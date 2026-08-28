from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(slots=True)
class ModelWeights:
    model_name: str
    Q: torch.Tensor  # [L,H,d,dh], column convention
    K: torch.Tensor
    V: torch.Tensor
    O: torch.Tensor  # [L,H,d,dh], transposed from TransformerLens W_O
    W_in: torch.Tensor  # [L,d_mlp,d], reader rows
    W_out: torch.Tensor  # [L,d_mlp,d], writer rows
    W_E: torch.Tensor  # [vocab,d], token writer rows
    W_pos: torch.Tensor | None  # [position,d], learned positional writer rows
    W_U: torch.Tensor  # [d,vocab], unembedding
    ln1_w: torch.Tensor
    ln2_w: torch.Tensor
    n_layers: int
    n_heads: int
    d_model: int
    d_head: int
    d_mlp: int
    parallel_attn_mlp: bool
    positional_scheme: str
    rotary_dim: int
    rotary_base: float
    # GPT-J rotates adjacent coordinate pairs (2j, 2j+1); GPT-NeoX (Pythia)
    # rotates the split-half pairs (j, j + rotary_dim/2).
    rotary_adjacent_pairs: bool = False

    def qk(self, layer: int, head: int) -> torch.Tensor:
        return self.Q[layer, head] @ self.K[layer, head].T

    def ov(self, layer: int, head: int) -> torch.Tensor:
        return self.O[layer, head] @ self.V[layer, head].T

    def head_label(self, layer: int, head: int) -> str:
        return f"L{layer}H{head}"

    @property
    def n_total_heads(self) -> int:
        return self.n_layers * self.n_heads

    def flat(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.reshape(self.n_total_heads, *tensor.shape[2:])


def extract_weights(model_name: str, model) -> ModelWeights:
    """Copy the processed model weights to CPU float64.

    The model has already had fixed LayerNorm gains folded into readers and
    writer dead directions centered by ``load_model``.  Input-dependent
    normalization gains remain activation-side quantities.
    """

    cfg = model.cfg
    with torch.no_grad():
        Q = model.W_Q.detach().double().cpu().contiguous()
        K = model.W_K.detach().double().cpu().contiguous()
        V = model.W_V.detach().double().cpu().contiguous()
        O = model.W_O.detach().double().cpu().transpose(-1, -2).contiguous()
        W_in = model.W_in.detach().double().cpu().transpose(1, 2).contiguous()
        W_out = model.W_out.detach().double().cpu().contiguous()
        W_E = model.W_E.detach().double().cpu().contiguous()
        W_U = model.W_U.detach().double().cpu().contiguous()
        W_pos = None
        if getattr(model, "W_pos", None) is not None:
            W_pos = model.W_pos.detach().double().cpu().contiguous()
        def folded_ln_weight(module) -> torch.Tensor:
            # TransformerLens replaces a folded LayerNorm with LayerNormPre,
            # which has no learned gain because that gain is already present in
            # the downstream readers.
            value = getattr(module, "w", None)
            if value is None:
                return torch.ones(cfg.d_model, dtype=torch.float64)
            return value.detach().double().cpu()

        ln1_w = torch.stack(
            [folded_ln_weight(model.blocks[layer].ln1) for layer in range(cfg.n_layers)]
        )
        ln2_w = torch.stack(
            [folded_ln_weight(model.blocks[layer].ln2) for layer in range(cfg.n_layers)]
        )
    expected = (cfg.n_layers, cfg.n_heads, cfg.d_model, cfg.d_head)
    if tuple(Q.shape) != expected or tuple(O.shape) != expected:
        raise ValueError(f"folded head factor shape mismatch: Q={tuple(Q.shape)}, O={tuple(O.shape)}, expected={expected}")
    return ModelWeights(
        model_name=model_name,
        Q=Q,
        K=K,
        V=V,
        O=O,
        W_in=W_in,
        W_out=W_out,
        W_E=W_E,
        W_pos=W_pos,
        W_U=W_U,
        ln1_w=ln1_w,
        ln2_w=ln2_w,
        n_layers=int(cfg.n_layers),
        n_heads=int(cfg.n_heads),
        d_model=int(cfg.d_model),
        d_head=int(cfg.d_head),
        d_mlp=int(cfg.d_mlp),
        parallel_attn_mlp=bool(getattr(cfg, "parallel_attn_mlp", False)),
        positional_scheme=str(cfg.positional_embedding_type),
        rotary_dim=int(getattr(cfg, "rotary_dim", 0) or 0),
        rotary_base=float(getattr(cfg, "rotary_base", 10_000.0) or 10_000.0),
        rotary_adjacent_pairs=bool(getattr(cfg, "rotary_adjacent_pairs", False)),
    )


def inner_grams(weights: ModelWeights) -> dict[str, torch.Tensor]:
    return {
        "Q": torch.einsum("lhdi,lhdj->lhij", weights.Q, weights.Q),
        "K": torch.einsum("lhdi,lhdj->lhij", weights.K, weights.K),
        "V": torch.einsum("lhdi,lhdj->lhij", weights.V, weights.V),
        "O": torch.einsum("lhdi,lhdj->lhij", weights.O, weights.O),
    }


def head_norms(weights: ModelWeights, grams: dict[str, torch.Tensor] | None = None):
    g = grams or inner_grams(weights)
    qk = torch.einsum("lhij,lhij->lh", g["Q"], g["K"])
    ov = torch.einsum("lhij,lhij->lh", g["O"], g["V"])
    return {"K": qk, "Q": qk, "V": ov, "writer": ov}


def psd_rank(matrix: torch.Tensor, rtol: float = 1.0e-9) -> int:
    values = torch.linalg.eigvalsh(0.5 * (matrix + matrix.T))
    if values.numel() == 0 or float(values.max()) <= 0:
        return 0
    return int(torch.count_nonzero(values > rtol * values.max()).item())
