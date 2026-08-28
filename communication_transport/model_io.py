from __future__ import annotations

import contextlib
import gc
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch
import torch.nn.functional as F

from .config import RunConfig


@dataclass(slots=True)
class Corpora:
    induction: torch.Tensor
    natural: torch.Tensor
    position: torch.Tensor
    block_size: int


@dataclass(slots=True)
class BehavioralReadout:
    induction_gain: float
    natural_loss: float
    induction_gain_by_sequence: np.ndarray
    natural_loss_by_sequence: np.ndarray


def _canonical_hf_name(name: str) -> str:
    if "/" in name:
        return name
    if name.lower().startswith("pythia-"):
        return f"EleutherAI/{name}"
    return name


def load_model(name: str, config: RunConfig):
    """Load a processed TransformerLens model, including the HF-v5 NeoX shim.

    TransformerLens still looks for ``embed_out`` on GPT-NeoX while recent
    Transformers exposes the same module as ``lm_head``.  Supplying the patched
    HF model keeps the conversion faithful and is removed once upstream no longer
    needs it.
    """

    if config.offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from transformer_lens import HookedTransformer

    canonical = _canonical_hf_name(name)
    kwargs = dict(
        fold_ln=True,
        center_writing_weights=True,
        center_unembed=True,
        device=config.device,
    )
    if "pythia" in canonical.lower():
        from transformers import AutoModelForCausalLM

        hf_model = AutoModelForCausalLM.from_pretrained(
            canonical, local_files_only=config.offline
        )
        if not hasattr(hf_model, "embed_out"):
            hf_model.embed_out = getattr(hf_model, "lm_head", None)
            if hf_model.embed_out is None:
                hf_model.embed_out = hf_model.get_output_embeddings()
        model = HookedTransformer.from_pretrained(
            canonical, hf_model=hf_model, **kwargs
        )
        del hf_model
    else:
        model = HookedTransformer.from_pretrained(canonical, **kwargs)
    model.eval()
    if hasattr(model, "set_use_attn_result"):
        model.set_use_attn_result(True)
    return model


def unload_model(model) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def model_summary(name: str, model) -> dict[str, object]:
    cfg = model.cfg
    learned_pos = (
        cfg.positional_embedding_type in ("standard", "shortformer")
        and getattr(model, "pos_embed", None) is not None
    )
    return {
        "model": name,
        "n_layers": int(cfg.n_layers),
        "n_heads": int(cfg.n_heads),
        "d_model": int(cfg.d_model),
        "d_head": int(cfg.d_head),
        "d_mlp": int(cfg.d_mlp),
        "d_vocab": int(cfg.d_vocab),
        "positional_scheme": str(cfg.positional_embedding_type),
        "rotary_dim": int(getattr(cfg, "rotary_dim", 0) or 0),
        "rotary_base": float(getattr(cfg, "rotary_base", 10_000.0) or 10_000.0),
        "rotary_adjacent_pairs": bool(getattr(cfg, "rotary_adjacent_pairs", False)),
        "parallel_attn_mlp": bool(getattr(cfg, "parallel_attn_mlp", False)),
        "residual_topology": (
            "parallel_attention_mlp"
            if bool(getattr(cfg, "parallel_attn_mlp", False))
            else "serial_attention_then_mlp"
        ),
        "learned_position_matrix": learned_pos,
        "inference_dtype": str(model.W_Q.dtype).replace("torch.", ""),
        "device": str(model.W_Q.device),
    }


def _valid_vocab(model) -> torch.Tensor:
    tok = model.tokenizer
    limit = min(int(model.cfg.d_vocab), len(tok))
    special = {
        value
        for value in (
            tok.bos_token_id,
            tok.eos_token_id,
            tok.pad_token_id,
            getattr(tok, "unk_token_id", None),
        )
        if value is not None
    }
    return torch.tensor([i for i in range(limit) if i not in special], dtype=torch.long)


def _bos(model) -> int:
    tok = model.tokenizer
    for value in (tok.bos_token_id, tok.eos_token_id, tok.pad_token_id):
        if value is not None:
            return int(value)
    return 0


def build_corpora(model, config: RunConfig) -> Corpora:
    rng = torch.Generator(device="cpu").manual_seed(config.seed)
    vocab = _valid_vocab(model)
    bos = _bos(model)
    n = config.n_sequences
    block = config.induction_block

    block_tokens = vocab[
        torch.randint(len(vocab), (n, block), generator=rng)
    ]
    induction = torch.cat(
        [torch.full((n, 1), bos, dtype=torch.long), block_tokens, block_tokens], dim=1
    )

    position_random = vocab[
        torch.randint(
            len(vocab), (n, config.position_length - 1), generator=rng
        )
    ]
    position = torch.cat(
        [torch.full((n, 1), bos, dtype=torch.long), position_random], dim=1
    )

    from datasets import load_dataset

    dataset = load_dataset("NeelNanda/pile-10k", split="train")
    natural_rows: list[list[int]] = []
    need = config.natural_length - 1
    for doc in dataset:
        ids = model.tokenizer(doc["text"], add_special_tokens=False)["input_ids"]
        if len(ids) >= need:
            natural_rows.append(ids[:need])
        if len(natural_rows) == n:
            break
    if len(natural_rows) != n:
        raise RuntimeError(
            f"natural corpus supplied {len(natural_rows)} usable rows, expected {n}"
        )
    natural = torch.cat(
        [
            torch.full((n, 1), bos, dtype=torch.long),
            torch.tensor(natural_rows, dtype=torch.long),
        ],
        dim=1,
    )
    return Corpora(
        induction=induction.to(config.device),
        natural=natural.to(config.device),
        position=position.to(config.device),
        block_size=block,
    )


def _sequence_nll(logits: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits.float(), dim=-1)
    return -log_probs[:, :-1].gather(
        -1, tokens[:, 1:, None]
    ).squeeze(-1)


def evaluate_behavior(
    model,
    corpora: Corpora,
    config: RunConfig,
    hooks_induction: list[tuple[str, Callable]] | None = None,
    hooks_natural: list[tuple[str, Callable]] | None = None,
) -> BehavioralReadout:
    first: list[torch.Tensor] = []
    second: list[torch.Tensor] = []
    natural_losses: list[torch.Tensor] = []
    b = config.batch_size
    block = corpora.block_size
    with torch.no_grad(), model.hooks(fwd_hooks=hooks_induction or []):
        for start in range(0, len(corpora.induction), b):
            toks = corpora.induction[start : start + b]
            loss = _sequence_nll(model(toks), toks)
            # First copy predicts block positions; second copy begins after the
            # first block and has access to its exact predecessor sequence.
            first.append(loss[:, :block].mean(dim=1).cpu())
            second.append(loss[:, block:].mean(dim=1).cpu())
    with torch.no_grad(), model.hooks(fwd_hooks=hooks_natural or []):
        for start in range(0, len(corpora.natural), b):
            toks = corpora.natural[start : start + b]
            natural_losses.append(_sequence_nll(model(toks), toks).mean(dim=1).cpu())
    first_v = torch.cat(first).double().numpy()
    second_v = torch.cat(second).double().numpy()
    natural_v = torch.cat(natural_losses).double().numpy()
    gain = first_v - second_v
    return BehavioralReadout(
        induction_gain=float(gain.mean()),
        natural_loss=float(natural_v.mean()),
        induction_gain_by_sequence=gain,
        natural_loss_by_sequence=natural_v,
    )


def detect_induction_heads(
    model, corpora: Corpora, config: RunConfig, topk: int = 5
) -> tuple[list[str], dict[str, float], dict[str, float]]:
    """Score every head at the induction and the duplicate-token offsets.

    With sequence [BOS, t_0..t_{B-1}, t_0..t_{B-1}], an induction head at the
    second-copy query t_k attends to the token AFTER the previous occurrence,
    i.e. offset -(B-1).  Offset -B is the duplicate-token pattern; it is kept
    as its own stored score because duplicate-token heads are useful labels.
    """

    names = {
        f"blocks.{layer}.attn.hook_pattern"
        for layer in range(model.cfg.n_layers)
    }
    sums = torch.zeros(
        model.cfg.n_layers, model.cfg.n_heads, dtype=torch.float64
    )
    dup_sums = torch.zeros_like(sums)
    counts = 0
    offset = corpora.block_size - 1
    with torch.no_grad():
        for start in range(0, len(corpora.induction), config.batch_size):
            toks = corpora.induction[start : start + config.batch_size]
            _, cache = model.run_with_cache(
                toks,
                return_type=None,
                names_filter=lambda name: name in names,
            )
            for layer in range(model.cfg.n_layers):
                pattern = cache[f"blocks.{layer}.attn.hook_pattern"]
                diagonal = pattern.diagonal(offset=-offset, dim1=2, dim2=3)
                # Only receiver positions in the copied block contribute; the
                # first two diagonal entries are first-block queries.
                score = diagonal[..., 2:].mean(dim=(0, 2)).double().cpu()
                sums[layer] += score * toks.shape[0]
                duplicate = pattern.diagonal(
                    offset=-(offset + 1), dim1=2, dim2=3
                )
                dup_score = duplicate[..., 1:].mean(dim=(0, 2)).double().cpu()
                dup_sums[layer] += dup_score * toks.shape[0]
            counts += toks.shape[0]
            del cache
    scores = sums / max(counts, 1)
    dup_scores = dup_sums / max(counts, 1)
    flat = [
        (float(scores[layer, head]), layer, head)
        for layer in range(model.cfg.n_layers)
        for head in range(model.cfg.n_heads)
    ]
    flat.sort(reverse=True)
    labels = [f"L{layer}H{head}" for _, layer, head in flat[:topk]]
    score_map = {
        f"L{layer}H{head}": float(scores[layer, head])
        for layer in range(model.cfg.n_layers)
        for head in range(model.cfg.n_heads)
    }
    dup_map = {
        f"L{layer}H{head}": float(dup_scores[layer, head])
        for layer in range(model.cfg.n_layers)
        for head in range(model.cfg.n_heads)
    }
    return labels, score_map, dup_map


def cache_activations(
    model,
    tokens: torch.Tensor,
    names: Iterable[str],
    config: RunConfig,
) -> dict[str, torch.Tensor]:
    wanted = set(names)
    pieces: dict[str, list[torch.Tensor]] = {name: [] for name in wanted}
    with torch.no_grad():
        for start in range(0, len(tokens), config.batch_size):
            batch = tokens[start : start + config.batch_size]
            _, cache = model.run_with_cache(
                batch,
                return_type=None,
                names_filter=lambda name: name in wanted,
            )
            for name in wanted:
                if name in cache:
                    pieces[name].append(cache[name].detach().float().cpu())
            del cache
    return {
        name: torch.cat(chunks, dim=0)
        for name, chunks in pieces.items()
        if chunks
    }


def head_z_means(
    model, tokens: torch.Tensor, config: RunConfig
) -> dict[int, torch.Tensor]:
    names = {
        f"blocks.{layer}.attn.hook_z" for layer in range(model.cfg.n_layers)
    }
    sums: dict[int, torch.Tensor] = {}
    count = 0
    with torch.no_grad():
        for start in range(0, len(tokens), config.batch_size):
            batch = tokens[start : start + config.batch_size]
            _, cache = model.run_with_cache(
                batch,
                return_type=None,
                names_filter=lambda name: name in names,
            )
            positions = batch.shape[0] * batch.shape[1]
            for layer in range(model.cfg.n_layers):
                value = cache[f"blocks.{layer}.attn.hook_z"].mean(dim=(0, 1))
                sums[layer] = sums.get(layer, torch.zeros_like(value)) + value * positions
            count += positions
            del cache
    return {layer: value / count for layer, value in sums.items()}


def parse_head(label: str) -> tuple[int, int]:
    if not label.startswith("L") or "H" not in label:
        raise ValueError(f"not a head label: {label}")
    split = label.index("H")
    return int(label[1:split]), int(label[split + 1 :])


def mean_ablation_hooks(
    head_labels: Iterable[str], means: dict[int, torch.Tensor]
) -> list[tuple[str, Callable]]:
    by_layer: dict[int, list[int]] = {}
    for label in head_labels:
        layer, head = parse_head(label)
        by_layer.setdefault(layer, []).append(head)

    hooks: list[tuple[str, Callable]] = []
    for layer, heads in by_layer.items():
        frozen = tuple(heads)

        def hook(z, hook=None, *, layer=layer, heads=frozen):
            indices = torch.tensor(heads, device=z.device)
            replacement = means[layer].to(device=z.device, dtype=z.dtype)[indices]
            z[:, :, indices] = replacement
            return z

        hooks.append((f"blocks.{layer}.attn.hook_z", hook))
    return hooks


@contextlib.contextmanager
def temporarily_replace(parameter: torch.Tensor, replacement: torch.Tensor):
    original = parameter.detach().clone()
    with torch.no_grad():
        parameter.copy_(replacement.to(parameter))
    try:
        yield
    finally:
        with torch.no_grad():
            parameter.copy_(original)
