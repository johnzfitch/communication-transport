from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from .config import RunConfig
from .model_io import Corpora, cache_activations, evaluate_behavior, parse_head
from .weights import ModelWeights


@dataclass(slots=True)
class RealizedResult:
    table: pd.DataFrame
    observations: list[str]


def _traffic_cache(model, tokens: torch.Tensor, config: RunConfig):
    names = set()
    for layer in range(model.cfg.n_layers):
        names.update(
            (
                f"blocks.{layer}.attn.hook_result",
                f"blocks.{layer}.hook_resid_pre",
                f"blocks.{layer}.ln1.hook_normalized",
                f"blocks.{layer}.ln1.hook_scale",
            )
        )
    names.add(f"blocks.{model.cfg.n_layers - 1}.hook_resid_post")
    return cache_activations(model, tokens, names, config)


def _normalized_abs_mean(value: torch.Tensor, denominator: torch.Tensor) -> float:
    return float((value.abs() / denominator.clamp_min(1.0e-12)).mean())


def _realized_one(
    weights: ModelWeights,
    edge,
    cache: dict[str, torch.Tensor],
) -> dict[str, object]:
    wl, wh = parse_head(edge.writer)
    rl, rh = parse_head(edge.reader)
    delta = cache[f"blocks.{wl}.attn.hook_result"][:, :, wh].double()
    x = cache[f"blocks.{rl}.ln1.hook_normalized"].double()
    scale = cache.get(f"blocks.{rl}.ln1.hook_scale")
    if scale is not None:
        scale_mean = float(scale.double().mean())
        scale_sd = float(scale.double().std())
    else:
        scale_mean = scale_sd = math.nan
    if edge.channel == "V":
        # (delta V_r) O_r^T is the exact dense OV read without materializing
        # the d_model-square operator for every selected edge.
        read = (delta @ weights.V[rl, rh]) @ weights.O[rl, rh].T
        operator_norm = float(
            torch.linalg.matrix_norm(weights.O[rl, rh] @ weights.V[rl, rh].T)
        )
        denominator = torch.linalg.vector_norm(delta, dim=-1) * operator_norm
        signed = float(read.mean())
        absolute = _normalized_abs_mean(torch.linalg.vector_norm(read, dim=-1), denominator)
        both = math.nan
    else:
        xi = x[:, 1:]
        xj = x[:, :-1]
        di = delta[:, 1:]
        dj = delta[:, :-1]
        q_factor = weights.Q[rl, rh]
        k_factor = weights.K[rl, rh]
        operator_norm = float(torch.linalg.matrix_norm(q_factor @ k_factor.T))
        if edge.channel == "K":
            read = torch.sum((xi @ q_factor) * (dj @ k_factor), dim=-1)
            denominator = (
                torch.linalg.vector_norm(xi, dim=-1)
                * torch.linalg.vector_norm(dj, dim=-1)
                * operator_norm
            )
        else:
            read = torch.sum((di @ q_factor) * (xj @ k_factor), dim=-1)
            denominator = (
                torch.linalg.vector_norm(di, dim=-1)
                * torch.linalg.vector_norm(xj, dim=-1)
                * operator_norm
            )
        both_term = torch.sum((di @ q_factor) * (dj @ k_factor), dim=-1)
        both_denominator = (
            torch.linalg.vector_norm(di, dim=-1)
            * torch.linalg.vector_norm(dj, dim=-1)
            * operator_norm
        )
        signed = float((read / denominator.clamp_min(1.0e-12)).mean())
        absolute = _normalized_abs_mean(read, denominator)
        both = _normalized_abs_mean(both_term, both_denominator)
    return {
        "model": weights.model_name,
        "row_kind": "edge_summary",
        "edge_id": f"{weights.model_name}:{edge.writer}->{edge.reader}:{edge.channel}",
        "writer": edge.writer,
        "reader": edge.reader,
        "writer_layer": wl,
        "reader_layer": rl,
        "channel": edge.channel,
        "potential_C": float(edge.C),
        "realized_signed": signed,
        "realized_absolute": absolute,
        "both_sides_absolute": both,
        "reader_ln_scale_mean": scale_mean,
        "reader_ln_scale_sd": scale_sd,
        "potential": bool(edge.selected),
        "stored_scope": "all_selected_head_edges_activation_summary",
    }


def _single_batch_cache(model, tokens: torch.Tensor, names: set[str], hooks=None):
    with torch.no_grad(), model.hooks(fwd_hooks=hooks or []):
        logits, cache = model.run_with_cache(
            tokens,
            return_type="logits",
            names_filter=lambda name: name in names,
        )
    return logits.detach().float().cpu(), {name: cache[name].detach().float().cpu() for name in names if name in cache}


def _causal_survival(
    model,
    weights: ModelWeights,
    edge,
    tokens: torch.Tensor,
    config: RunConfig,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    wl, wh = parse_head(edge.writer)
    rl, rh = parse_head(edge.reader)
    writer_name = f"blocks.{wl}.attn.hook_result"
    reader_result_name = f"blocks.{rl}.attn.hook_result"
    downstream_names = {
        f"blocks.{layer}.hook_resid_pre" for layer in range(rl, weights.n_layers)
    }
    downstream_names.update((writer_name, reader_result_name, f"blocks.{weights.n_layers - 1}.hook_resid_post"))
    clean_logits, clean_cache = _single_batch_cache(model, tokens, downstream_names)
    delta = clean_cache[writer_name][:, :, wh].to(model.W_Q.device)

    def subtract_writer(residual, hook=None):
        return residual - delta.to(residual)

    hooks = [(f"blocks.{rl}.hook_resid_pre", subtract_writer)]
    patched_logits, patched_cache = _single_batch_cache(model, tokens, downstream_names, hooks)
    reader_change = patched_cache[reader_result_name][:, :, rh] - clean_cache[reader_result_name][:, :, rh]
    reader_norm = float(torch.linalg.vector_norm(reader_change.double()))
    logit_change = float(torch.linalg.vector_norm((patched_logits - clean_logits).double()))
    logit_reference = float(torch.linalg.vector_norm(clean_logits.double()))
    clean_lp = torch.log_softmax(clean_logits, dim=-1)
    patch_lp = torch.log_softmax(patched_logits, dim=-1)
    targets = tokens.cpu()[:, 1:, None]
    clean_loss = float((-clean_lp[:, :-1].gather(-1, targets).squeeze(-1)).mean())
    patch_loss = float((-patch_lp[:, :-1].gather(-1, targets).squeeze(-1)).mean())
    summary = {
        "reader_output_change": reader_norm,
        "relative_logit_change": logit_change / max(logit_reference, 1.0e-300),
        "causal_natural_loss_change_subset": patch_loss - clean_loss,
    }
    # Task-relevant projector: at every traced position, the unembedding
    # direction of the actual next token (the continuation the position is
    # supposed to predict), applied per position and pooled in norm.
    token_targets = tokens.cpu()[:, 1:].reshape(-1)
    directions = weights.W_U[:, token_targets].T.reshape(
        tokens.shape[0], tokens.shape[1] - 1, weights.d_model
    )
    directions = directions / torch.linalg.vector_norm(
        directions, dim=-1, keepdim=True
    ).clamp_min(1.0e-12)

    def projected_norm(difference: torch.Tensor) -> float:
        aligned = difference.double()[:, :-1]
        component = (aligned * directions).sum(dim=-1)
        return float(torch.linalg.vector_norm(component))

    survival_rows = []
    for layer in range(rl, weights.n_layers):
        name = f"blocks.{layer}.hook_resid_pre"
        difference = patched_cache[name] - clean_cache[name]
        survival_rows.append(
            {
                "model": weights.model_name,
                "row_kind": "survival_curve",
                "edge_id": f"{weights.model_name}:{edge.writer}->{edge.reader}:{edge.channel}",
                "writer": edge.writer,
                "reader": edge.reader,
                "channel": edge.channel,
                "downstream_layer": layer,
                "whole_stream_survival_ratio": float(torch.linalg.vector_norm(difference.double())) / max(reader_norm, 1.0e-300),
                "next_token_projected_survival_ratio": projected_norm(difference) / max(reader_norm, 1.0e-300),
                "target_projector": "identity_plus_next_token_unembedding",
            }
        )
    final_difference = patched_cache[f"blocks.{weights.n_layers - 1}.hook_resid_post"] - clean_cache[f"blocks.{weights.n_layers - 1}.hook_resid_post"]
    survival_rows.append(
        {
            "model": weights.model_name,
            "row_kind": "survival_curve",
            "edge_id": f"{weights.model_name}:{edge.writer}->{edge.reader}:{edge.channel}",
            "writer": edge.writer,
            "reader": edge.reader,
            "channel": edge.channel,
            "downstream_layer": weights.n_layers,
            "whole_stream_survival_ratio": float(torch.linalg.vector_norm(final_difference.double())) / max(reader_norm, 1.0e-300),
            "next_token_projected_survival_ratio": projected_norm(final_difference) / max(reader_norm, 1.0e-300),
            "target_projector": "identity_plus_next_token_unembedding_final_residual",
        }
    )
    return summary, survival_rows


def _cached_head_delta(model, tokens: torch.Tensor, layer: int, head: int, config: RunConfig):
    name = f"blocks.{layer}.attn.hook_result"
    return cache_activations(model, tokens, [name], config)[name][:, :, head]


def _sequential_subtraction_hook(delta: torch.Tensor):
    state = {"start": 0}

    def hook(residual, hook=None):
        start = state["start"]
        stop = start + residual.shape[0]
        replacement = delta[start:stop].to(residual)
        state["start"] = stop
        return residual - replacement

    return hook


def _behavioral_patch(
    model,
    corpora: Corpora,
    edge,
    config: RunConfig,
) -> tuple[float, float]:
    wl, wh = parse_head(edge.writer)
    rl, _ = parse_head(edge.reader)
    delta_ind = _cached_head_delta(model, corpora.induction, wl, wh, config)
    delta_nat = _cached_head_delta(model, corpora.natural, wl, wh, config)
    clean = evaluate_behavior(model, corpora, config)
    changed = evaluate_behavior(
        model,
        corpora,
        config,
        hooks_induction=[
            (f"blocks.{rl}.hook_resid_pre", _sequential_subtraction_hook(delta_ind))
        ],
        hooks_natural=[
            (f"blocks.{rl}.hook_resid_pre", _sequential_subtraction_hook(delta_nat))
        ],
    )
    destroyed = 1.0 - changed.induction_gain / clean.induction_gain if abs(clean.induction_gain) > 1.0e-12 else math.nan
    return destroyed, changed.natural_loss - clean.natural_loss


def run_realized_traffic(
    model,
    weights: ModelWeights,
    corpora: Corpora,
    edges: pd.DataFrame,
    config: RunConfig,
) -> RealizedResult:
    selected = edges[
        edges["edge_class"].str.startswith("head_head_") & edges["selected"]
    ].copy()
    traffic_n = min(4, len(corpora.natural))
    traffic_tokens = corpora.natural[:traffic_n]
    traffic_config = RunConfig.debug(
        models=(weights.model_name,), output=config.output, experiments=("realized_traffic",)
    )
    traffic_config.device = config.device
    traffic_config.batch_size = traffic_n
    cache = _traffic_cache(model, traffic_tokens, traffic_config)
    summaries = [_realized_one(weights, edge, cache) for edge in selected.itertuples()]
    summary_frame = pd.DataFrame(summaries)
    rows = summaries.copy()
    if len(summary_frame):
        medians = summary_frame.groupby("channel")["realized_absolute"].transform("median")
        for row, median in zip(rows, medians, strict=True):
            row["realized"] = bool(row["realized_absolute"] >= median)
            row["survives_downstream_mixing"] = False

    causal_edges = selected.nlargest(
        min(config.max_realized_edges_per_model, len(selected)), "C"
    )
    survival_rows: list[dict[str, object]] = []
    causal_lookup: dict[str, dict[str, object]] = {}
    for index, edge in enumerate(causal_edges.itertuples()):
        try:
            causal, curves = _causal_survival(
                model, weights, edge, traffic_tokens, traffic_config
            )
            if index < min(3, len(causal_edges)):
                gain_destroyed, dnll = _behavioral_patch(model, corpora, edge, config)
                causal["causal_induction_gain_destroyed"] = gain_destroyed
                causal["causal_natural_loss_change"] = dnll
            edge_id = f"{weights.model_name}:{edge.writer}->{edge.reader}:{edge.channel}"
            causal_lookup[edge_id] = causal
            survival_rows.extend(curves)
        except Exception as exc:
            edge_id = f"{weights.model_name}:{edge.writer}->{edge.reader}:{edge.channel}"
            causal_lookup[edge_id] = {"local_error": f"{type(exc).__name__}: {exc}"}
    for row in rows:
        causal = causal_lookup.get(str(row["edge_id"]))
        if causal:
            row.update(causal)
            final = [
                value
                for value in survival_rows
                if value["edge_id"] == row["edge_id"]
                and value["downstream_layer"] == weights.n_layers
            ]
            if final:
                row["survives_downstream_mixing"] = bool(
                    final[0]["whole_stream_survival_ratio"] >= 0.1
                )
                row["survives_task_relevant"] = bool(
                    final[0].get("next_token_projected_survival_ratio", math.nan) >= 0.1
                )
    table = pd.concat(
        [pd.DataFrame(rows), pd.DataFrame(survival_rows)],
        ignore_index=True,
        sort=False,
    )
    edge_rows = table[table["row_kind"] == "edge_summary"] if len(table) else table
    correlation = (
        float(edge_rows[["potential_C", "realized_absolute"]].corr().iloc[0, 1])
        if len(edge_rows) > 2
        else math.nan
    )
    dormant = edge_rows.nlargest(max(1, min(5, len(edge_rows))), "potential_C")
    dormant_fraction = (
        float((~dormant["realized"].fillna(False).astype(bool)).mean())
        if len(dormant)
        else math.nan
    )
    observations = [
        f"Potential coupling and realized absolute traffic had Pearson correlation {correlation:.3f} over {len(edge_rows)} selected edges.",
        f"Among the strongest-potential examples, {dormant_fraction:.1%} fell below their channel's median realized traffic on the sampled inputs.",
        f"Causal survival curves were computed for {len(causal_lookup)} top edges by removing the writer contribution at the reader input and following the resulting perturbation.",
    ]
    return RealizedResult(table=table, observations=observations)
