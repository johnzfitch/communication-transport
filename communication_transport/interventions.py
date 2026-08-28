from __future__ import annotations

import math
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import torch

from .config import RunConfig
from .model_io import (
    BehavioralReadout,
    Corpora,
    evaluate_behavior,
    head_z_means,
    mean_ablation_hooks,
    parse_head,
    temporarily_replace,
)
from .weights import ModelWeights


@dataclass(slots=True)
class InterventionResult:
    records: pd.DataFrame
    clean: BehavioralReadout
    targets: dict[str, list[str]]
    observations: list[str]


def flatten_ov_factors(
    O: torch.Tensor, V: torch.Tensor, rtol: float = 1.0e-9
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Return factors for the Frobenius-matched flat spectrum of ``O V^T``."""

    Qo, Ro = torch.linalg.qr(O.double(), mode="reduced")
    Qv, Rv = torch.linalg.qr(V.double(), mode="reduced")
    Um, singular, Vhm = torch.linalg.svd(Ro @ Rv.T, full_matrices=False)
    threshold = rtol * max(float(singular.max()), 1.0)
    keep = singular > threshold
    rank = int(keep.sum().item())
    O_new = torch.zeros_like(O, dtype=torch.float64)
    V_new = torch.zeros_like(V, dtype=torch.float64)
    if rank:
        flat = float(torch.linalg.vector_norm(singular[keep])) / math.sqrt(rank)
        scale = math.sqrt(flat)
        O_new[:, :rank] = (Qo @ Um[:, keep]) * scale
        V_new[:, :rank] = (Qv @ Vhm[keep].T) * scale
    else:
        flat = 0.0
    original_norm = float(torch.linalg.vector_norm(singular))
    rebuilt = O_new @ V_new.T
    original = O.double() @ V.double().T
    return O_new, V_new, {
        "support_rank": rank,
        "flat_singular_value": flat,
        "original_frobenius": original_norm,
        "flat_frobenius": float(torch.linalg.matrix_norm(rebuilt)),
        "frob_residual": abs(float(torch.linalg.matrix_norm(rebuilt)) - original_norm),
        "operator_change": float(torch.linalg.matrix_norm(rebuilt - original))
        / max(original_norm, 1.0e-300),
    }


def _damage(clean: BehavioralReadout, changed: BehavioralReadout) -> tuple[float, float]:
    destroyed = 1.0 - changed.induction_gain / clean.induction_gain if abs(clean.induction_gain) > 1.0e-12 else math.nan
    return destroyed, changed.natural_loss - clean.natural_loss


def _row(
    *,
    model_name: str,
    intervention: str,
    target: str,
    clean: BehavioralReadout,
    changed: BehavioralReadout,
    scalar_change: float,
    spectral_metric: float,
    frame_metric: float,
    orientation_change: float,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    destroyed, dnll = _damage(clean, changed)
    row: dict[str, object] = {
        "model": model_name,
        "intervention": intervention,
        "target": target,
        "clean_induction_gain": clean.induction_gain,
        "intervened_induction_gain": changed.induction_gain,
        "fraction_gain_destroyed": destroyed,
        "clean_natural_loss": clean.natural_loss,
        "intervened_natural_loss": changed.natural_loss,
        "delta_natural_loss": dnll,
        "scalar_coupling_change": scalar_change,
        "spectral_metric": spectral_metric,
        "frame_metric": frame_metric,
        "operator_orientation_change": orientation_change,
        "induction_gain_by_sequence": changed.induction_gain_by_sequence.tolist(),
        "natural_loss_by_sequence": changed.natural_loss_by_sequence.tolist(),
    }
    if extra:
        row.update(extra)
    return row


def _outgoing_coupling_change(
    weights: ModelWeights,
    edges: pd.DataFrame,
    label: str,
    O_new: torch.Tensor,
    V_new: torch.Tensor,
) -> tuple[float, float]:
    affected = edges[
        (edges["writer"] == label)
        & edges["edge_class"].str.startswith("head_head_")
    ]
    if affected.empty:
        return math.nan, math.nan
    M = O_new @ V_new.T
    values: list[float] = []
    for edge in affected.itertuples():
        rl, rh = parse_head(edge.reader)
        if edge.channel == "K":
            R = weights.qk(rl, rh)
        elif edge.channel == "Q":
            R = weights.qk(rl, rh).T
        else:
            R = weights.ov(rl, rh)
        c = float(torch.linalg.matrix_norm(R @ M) / (torch.linalg.matrix_norm(R) * torch.linalg.matrix_norm(M)))
        values.append(abs(c - float(edge.C)) / max(float(edge.C), 1.0e-12))
    return float(np.median(values)), float(np.max(values))


def _plane_hooks(model, plane: np.ndarray | torch.Tensor):
    V = torch.as_tensor(plane, dtype=model.W_Q.dtype, device=model.W_Q.device)
    V = torch.linalg.qr(V, mode="reduced").Q

    def project(x, hook=None):
        return x - (x @ V) @ V.T

    names = [
        f"blocks.{layer}.hook_resid_pre" for layer in range(model.cfg.n_layers)
    ]
    names.append(f"blocks.{model.cfg.n_layers - 1}.hook_resid_post")
    return [(name, project) for name in names]


def _random_layer_matched_heads(
    weights: ModelWeights, labels: Iterable[str], seed: int
) -> list[str]:
    rng = np.random.default_rng(seed)
    excluded = set(labels)
    counts: dict[int, int] = {}
    for label in labels:
        layer, _ = parse_head(label)
        counts[layer] = counts.get(layer, 0) + 1
    chosen: list[str] = []
    for layer, count in counts.items():
        pool = [
            weights.head_label(layer, head)
            for head in range(weights.n_heads)
            if weights.head_label(layer, head) not in excluded
        ]
        if pool:
            chosen.extend(rng.choice(pool, min(count, len(pool)), replace=False).tolist())
    return chosen


def run_interventions(
    model,
    weights: ModelWeights,
    corpora: Corpora,
    edges: pd.DataFrame,
    head_census: pd.DataFrame,
    induction_heads: list[str],
    induction_community: list[str],
    planes: dict[str, np.ndarray],
    config: RunConfig,
) -> InterventionResult:
    clean = evaluate_behavior(model, corpora, config)
    means_ind = head_z_means(model, corpora.induction, config)
    means_nat = head_z_means(model, corpora.natural, config)

    copier = str(head_census.nlargest(1, "copying_score")["head"].iloc[0])
    suppressor = str(head_census.nsmallest(1, "copying_score")["head"].iloc[0])
    # The strongest K writer into behavioral induction heads is the
    # data-derived previous-token/copy feeder candidate.
    incoming = edges[
        edges["reader"].isin(induction_heads)
        & (edges["channel"] == "K")
        & edges["edge_class"].str.startswith("head_head_")
    ]
    previous = str(incoming.nlargest(1, "C")["writer"].iloc[0]) if len(incoming) else induction_heads[0]
    primary = list(dict.fromkeys(induction_heads + [copier, suppressor, previous]))
    random_heads = _random_layer_matched_heads(weights, primary, config.seed + 41)
    targets = {
        "behavioral_induction_heads": induction_heads,
        "induction_community": induction_community,
        "primary_single_heads": primary,
        "layer_matched_random_heads": random_heads,
    }
    rows: list[dict[str, object]] = []
    deletion_damage: dict[str, float] = {}

    # Community and single-head mean deletion.
    deletion_sets: list[tuple[str, list[str]]] = []
    if induction_community:
        deletion_sets.append(("induction_community", induction_community))
    deletion_sets.extend((label, [label]) for label in primary + random_heads[: len(primary)])
    for target, labels in deletion_sets:
        changed = evaluate_behavior(
            model,
            corpora,
            config,
            hooks_induction=mean_ablation_hooks(labels, means_ind),
            hooks_natural=mean_ablation_hooks(labels, means_nat),
        )
        record = _row(
            model_name=weights.model_name,
            intervention="mean_ablation",
            target=target,
            clean=clean,
            changed=changed,
            scalar_change=0.0,
            spectral_metric=0.0,
            frame_metric=0.0,
            orientation_change=math.nan,
            extra={"head_set": labels},
        )
        rows.append(record)
        deletion_damage[target] = float(record["fraction_gain_destroyed"])

    # Single-head sign flip and spectrum flattening use actual model weights.
    for label in primary + random_heads[: min(2, len(random_heads))]:
        layer, head = parse_head(label)
        # ``model.W_O``/``model.W_V`` are stacked read accessors in current
        # TransformerLens, not writable views.  Mutate the owning block
        # parameters so the intervention reaches the forward computation.
        original_O_tl = model.blocks[layer].attn.W_O[head]
        original_V_tl = model.blocks[layer].attn.W_V[head]
        with temporarily_replace(original_O_tl, -original_O_tl):
            changed = evaluate_behavior(model, corpora, config)
        gram_residual = float(
            torch.max(
                torch.abs(
                    original_O_tl.detach().double().T @ original_O_tl.detach().double()
                    - (-original_O_tl.detach().double()).T @ (-original_O_tl.detach().double())
                )
            )
        )
        denominator = deletion_damage.get(label, math.nan)
        changed_row = _row(
            model_name=weights.model_name,
            intervention="sign_flip",
            target=label,
            clean=clean,
            changed=changed,
            scalar_change=gram_residual,
            spectral_metric=0.0,
            frame_metric=0.0,
            orientation_change=2.0,
            extra={
                "ratio_to_mean_ablation_damage": (
                    (1.0 - changed.induction_gain / clean.induction_gain) / denominator
                    if math.isfinite(denominator) and abs(denominator) > 1.0e-12
                    else math.nan
                ),
                "copying_score_before": float(
                    head_census.loc[head_census["head"] == label, "copying_score"].iloc[0]
                ),
                "copying_score_after": -float(
                    head_census.loc[head_census["head"] == label, "copying_score"].iloc[0]
                ),
            },
        )
        rows.append(changed_row)

        O_col = weights.O[layer, head]
        V_col = weights.V[layer, head]
        O_flat, V_flat, flat_info = flatten_ov_factors(
            O_col, V_col, config.support_rtol
        )
        median_dc, max_dc = _outgoing_coupling_change(
            weights, edges, label, O_flat, V_flat
        )
        with ExitStack() as stack:
            stack.enter_context(
                temporarily_replace(original_O_tl, O_flat.T)
            )
            stack.enter_context(
                temporarily_replace(original_V_tl, V_flat)
            )
            changed_flat = evaluate_behavior(model, corpora, config)
        rows.append(
            _row(
                model_name=weights.model_name,
                intervention="spectrum_flattening",
                target=label,
                clean=clean,
                changed=changed_flat,
                scalar_change=median_dc,
                spectral_metric=flat_info["operator_change"],
                frame_metric=0.0,
                orientation_change=0.0,
                extra={
                    **flat_info,
                    "max_relative_coupling_change": max_dc,
                },
            )
        )

    # Positional/cluster/random plane deletions share one intervention path.
    for name, plane in planes.items():
        if plane.size == 0:
            continue
        hooks = _plane_hooks(model, plane)
        changed = evaluate_behavior(
            model,
            corpora,
            config,
            hooks_induction=hooks,
            hooks_natural=hooks,
        )
        projector = np.asarray(plane) @ np.asarray(plane).T
        rows.append(
            _row(
                model_name=weights.model_name,
                intervention="plane_deletion",
                target=name,
                clean=clean,
                changed=changed,
                scalar_change=math.nan,
                spectral_metric=float(np.linalg.norm(projector)),
                frame_metric=math.nan,
                orientation_change=math.nan,
                extra={"projector_dimension": int(np.asarray(plane).shape[1])},
            )
        )

    frame = pd.DataFrame(rows)
    sign = frame[frame["intervention"] == "sign_flip"]
    flat = frame[frame["intervention"] == "spectrum_flattening"]
    community = frame[
        (frame["intervention"] == "mean_ablation")
        & (frame["target"] == "induction_community")
    ]
    observations = [
        (
            f"Sign flips left affected Gram couplings unchanged to {float(sign['scalar_coupling_change'].max()):.3e} while changing induction gain by a median {float(sign['fraction_gain_destroyed'].median()):.1%}."
            if len(sign)
            else "No sign-flip target was constructible."
        ),
        (
            f"Spectrum flattening moved outgoing scalar couplings by a median {float(flat['scalar_coupling_change'].median()):.1%} and changed induction gain by a median {float(flat['fraction_gain_destroyed'].median()):.1%}."
            if len(flat)
            else "No spectrum-flattening target was constructible."
        ),
        (
            f"Mean-ablation of the induction community destroyed {float(community['fraction_gain_destroyed'].iloc[0]):.1%} of the measured induction gain."
            if len(community)
            else "No induction-community ablation was constructible."
        ),
    ]
    return InterventionResult(
        records=frame,
        clean=clean,
        targets=targets,
        observations=observations,
    )
