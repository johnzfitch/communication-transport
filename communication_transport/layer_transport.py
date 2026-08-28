from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from .config import RunConfig
from .wang_map import MAD_TO_SD, bh_qvalues
from .weights import ModelWeights, head_norms, inner_grams


@dataclass(slots=True)
class LayerTransportResult:
    transports: dict[str, np.ndarray]
    fit_table: pd.DataFrame
    span_profiles: pd.DataFrame
    covariant_edges: pd.DataFrame
    observations: list[str]


def _center_matrix(activation: torch.Tensor) -> torch.Tensor:
    matrix = activation.reshape(-1, activation.shape[-1]).double().T
    return matrix - matrix.mean(dim=1, keepdim=True)


def _fit_one(
    source_train: torch.Tensor,
    anchor_train: torch.Tensor,
    source_eval: torch.Tensor,
    anchor_eval: torch.Tensor,
    ridge_multiplier: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    covariance = source_train @ source_train.T
    ridge = ridge_multiplier * float(torch.trace(covariance)) / max(len(covariance), 1)
    cross = anchor_train @ source_train.T
    identity = torch.eye(len(covariance), dtype=torch.float64, device=covariance.device)
    fitted = torch.linalg.solve(covariance + ridge * identity, cross.T).T
    U, singular, Vh = torch.linalg.svd(fitted, full_matrices=False)
    compact = U @ Vh
    prediction = fitted @ source_eval
    compact_prediction = compact @ source_eval
    centered_target = anchor_eval - anchor_eval.mean(dim=1, keepdim=True)
    total = float(torch.sum(centered_target * centered_target))
    fitted_residual = float(torch.sum((anchor_eval - prediction) ** 2))
    compact_residual = float(torch.sum((anchor_eval - compact_prediction) ** 2))
    return compact.cpu(), {
        "ridge_absolute": ridge,
        "heldout_r2_linear": 1.0 - fitted_residual / max(total, 1.0e-300),
        "heldout_r2_polar": 1.0 - compact_residual / max(total, 1.0e-300),
        "condition_number": float(singular.max() / singular.min().clamp_min(1.0e-300)),
        "minimum_singular_value": float(singular.min()),
        "maximum_singular_value": float(singular.max()),
        "compact_orthogonality_residual": float(
            torch.linalg.matrix_norm(compact.T @ compact - identity)
        ),
    }


def _prepare_matrices(
    activations: dict[int, torch.Tensor], config: RunConfig
) -> tuple[list[int], dict[int, torch.Tensor], torch.Tensor, torch.Tensor, int]:
    layers = sorted(activations)
    matrices = {layer: _center_matrix(activations[layer]) for layer in layers}
    sequence_length = int(next(iter(activations.values())).shape[1])
    n_positions = next(iter(matrices.values())).shape[1]
    rng = np.random.default_rng(config.seed + 501)
    order = rng.permutation(n_positions)
    split = max(1, min(n_positions - 1, int(config.transport_train_fraction * n_positions)))
    train_idx = torch.tensor(order[:split], dtype=torch.long)
    eval_idx = torch.tensor(order[split:], dtype=torch.long)
    return layers, matrices, train_idx, eval_idx, sequence_length


def fit_layer_transports(
    activations: dict[int, torch.Tensor], config: RunConfig
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    layers, matrices, train_idx, eval_idx, _ = _prepare_matrices(activations, config)
    device = config.device if torch.cuda.is_available() else "cpu"
    anchors = sorted(set((layers[0], layers[len(layers) // 2], layers[-1])))
    arrays: dict[str, np.ndarray] = {}
    rows: list[dict[str, object]] = []
    for anchor in anchors:
        anchor_train = matrices[anchor][:, train_idx].to(device)
        anchor_eval = matrices[anchor][:, eval_idx].to(device)
        for layer in layers:
            source_train = matrices[layer][:, train_idx].to(device)
            source_eval = matrices[layer][:, eval_idx].to(device)
            for ridge_multiplier in config.transport_ridge_grid:
                compact, diagnostics = _fit_one(
                    source_train,
                    anchor_train,
                    source_eval,
                    anchor_eval,
                    ridge_multiplier,
                )
                key = f"anchor_{anchor}__layer_{layer}__ridge_{ridge_multiplier:.0e}"
                arrays[key] = compact.numpy()
                rows.append(
                    {
                        "anchor_layer": anchor,
                        "source_layer": layer,
                        "ridge_multiplier": ridge_multiplier,
                        "n_train_positions": len(train_idx),
                        "n_eval_positions": len(eval_idx),
                        **diagnostics,
                    }
                )
            del source_train, source_eval
        del anchor_train, anchor_eval
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return arrays, pd.DataFrame(rows)


def _transport_key(anchor: int, layer: int, ridge: float) -> str:
    return f"anchor_{anchor}__layer_{layer}__ridge_{ridge:.0e}"


def covariant_couplings(
    weights: ModelWeights,
    edges: pd.DataFrame,
    transports: dict[str, np.ndarray],
    anchor: int,
    ridge: float,
) -> pd.DataFrame:
    head = edges[edges["edge_class"].str.startswith("head_head_")].copy()
    grams = inner_grams(weights)
    norms = head_norms(weights, grams)
    Qlayer = {
        layer: torch.from_numpy(transports[_transport_key(anchor, layer, ridge)]).double()
        for layer in range(weights.n_layers)
    }
    left_gram = {"K": "Q", "Q": "K", "V": "O"}
    reader_right = {"K": weights.K, "Q": weights.Q, "V": weights.V}
    cov_values: list[float] = []
    for edge in head.itertuples():
        wl = int(edge.writer_layer)
        wh = int(str(edge.writer).split("H")[1])
        rl = int(edge.reader_layer)
        rh = int(str(edge.reader).split("H")[1])
        relative = Qlayer[rl].T @ Qlayer[wl]
        X = reader_right[edge.channel][rl, rh].T @ relative @ weights.O[wl, wh]
        numerator = float(
            torch.einsum(
                "ij,ik,kl,jl->",
                X,
                grams[left_gram[edge.channel]][rl, rh],
                X,
                grams["V"][wl, wh],
            )
        )
        denominator = float(
            norms[edge.channel][rl, rh] * norms["writer"][wl, wh]
        )
        cov_values.append(math.sqrt(max(numerator / max(denominator, 1.0e-300), 0.0)))
    head["C_covariant"] = cov_values
    head["C_covariant2"] = head["C_covariant"] ** 2
    head["covariant_theoretical_z"] = (
        head["C_covariant2"] - head["theoretical_mean"]
    ) / np.sqrt(np.clip(head["theoretical_variance"], 1.0e-300, None))
    return head


def _selection_overlap(values: pd.DataFrame, column: str, q_level: float):
    frame = values.copy()
    frame["z_stratified"] = np.nan
    frame["z_pooled"] = np.nan
    for edge_class, group in frame.groupby("edge_class"):
        x_all = group[column].to_numpy()
        center = np.median(x_all)
        scale = max(MAD_TO_SD * np.median(np.abs(x_all - center)), 1.0e-12)
        frame.loc[group.index, "z_pooled"] = (x_all - center) / scale
        for _, span_group in group.groupby("layer_span"):
            x = span_group[column].to_numpy()
            center_s = np.median(x)
            scale_s = max(MAD_TO_SD * np.median(np.abs(x - center_s)), 1.0e-12)
            frame.loc[span_group.index, "z_stratified"] = (x - center_s) / scale_s
    pooled_selected: set[int] = set()
    strat_selected: set[int] = set()
    for _, group in frame.groupby("edge_class"):
        qp = bh_qvalues(sps_norm_sf(frame.loc[group.index, "z_pooled"].to_numpy()))
        qs = bh_qvalues(sps_norm_sf(frame.loc[group.index, "z_stratified"].to_numpy()))
        pooled_selected.update(group.index[qp <= q_level].tolist())
        strat_selected.update(group.index[qs <= q_level].tolist())
    union = pooled_selected | strat_selected
    jaccard = len(pooled_selected & strat_selected) / len(union) if union else 1.0
    return jaccard, len(pooled_selected), len(strat_selected)


def sps_norm_sf(values: np.ndarray) -> np.ndarray:
    from scipy.special import ndtr

    return ndtr(-np.asarray(values, dtype=np.float64))


def span_profiles(
    raw_edges: pd.DataFrame,
    covariant: pd.DataFrame,
    config: RunConfig,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for edge_class, group in covariant.groupby("edge_class"):
        raw_centers = []
        cov_centers = []
        for span, span_group in group.groupby("layer_span"):
            raw = span_group["C"].to_numpy()
            cov = span_group["C_covariant"].to_numpy()
            raw_center = float(np.median(raw))
            cov_center = float(np.median(cov))
            raw_scale = float(MAD_TO_SD * np.median(np.abs(raw - raw_center)))
            cov_scale = float(MAD_TO_SD * np.median(np.abs(cov - cov_center)))
            raw_centers.append(raw_center)
            cov_centers.append(cov_center)
            rows.append(
                {
                    "model": str(span_group["model"].iloc[0]),
                    "edge_class": edge_class,
                    "layer_span": int(span),
                    "raw_median": raw_center,
                    "covariant_median": cov_center,
                    "raw_robust_scale": raw_scale,
                    "covariant_robust_scale": cov_scale,
                    "raw_median_theoretical_z": float(span_group["theoretical_z"].median()),
                    "covariant_median_theoretical_z": float(span_group["covariant_theoretical_z"].median()),
                }
            )
        raw_variance = float(np.var(raw_centers))
        cov_variance = float(np.var(cov_centers))
        collapse = 1.0 - cov_variance / raw_variance if raw_variance > 0 else math.nan
        for row in rows:
            if row["edge_class"] == edge_class:
                row["collapse_fraction"] = collapse
    raw_j, raw_pool, raw_strat = _selection_overlap(covariant, "C", config.fdr_q)
    cov_j, cov_pool, cov_strat = _selection_overlap(covariant, "C_covariant", config.fdr_q)
    for row in rows:
        row.update(
            {
                "raw_pooled_vs_stratified_selection_jaccard": raw_j,
                "covariant_pooled_vs_stratified_selection_jaccard": cov_j,
                "raw_pooled_selected": raw_pool,
                "raw_stratified_selected": raw_strat,
                "covariant_pooled_selected": cov_pool,
                "covariant_stratified_selected": cov_strat,
            }
        )
    return pd.DataFrame(rows)


def _pooled_layer_state(weights: ModelWeights, layer: int) -> torch.Tensor:
    state = torch.zeros(weights.d_model, weights.d_model, dtype=torch.float64)
    for factor_stack in (weights.Q, weights.K, weights.V, weights.O):
        for head in range(weights.n_heads):
            factor = factor_stack[layer, head]
            gram = factor @ factor.T
            state += gram / torch.trace(gram)
    return 0.5 * (state + state.T)


def weight_ot_transports(
    weights: ModelWeights, anchor: int, ridge: float
) -> tuple[dict[str, np.ndarray], list[dict[str, object]]]:
    states = [_pooled_layer_state(weights, layer) for layer in range(weights.n_layers)]
    Sa = states[anchor]
    eye = torch.eye(weights.d_model, dtype=torch.float64)

    def sqrt(matrix):
        values, vectors = torch.linalg.eigh(0.5 * (matrix + matrix.T))
        values = values.clamp_min(ridge)
        return (vectors * torch.sqrt(values)) @ vectors.T

    def invsqrt(matrix):
        values, vectors = torch.linalg.eigh(0.5 * (matrix + matrix.T))
        values = values.clamp_min(ridge)
        return (vectors * torch.rsqrt(values)) @ vectors.T

    Sa_root = sqrt(Sa + ridge * eye)
    arrays: dict[str, np.ndarray] = {}
    rows: list[dict[str, object]] = []
    for layer, state in enumerate(states):
        middle = Sa_root @ (state + ridge * eye) @ Sa_root
        transport = Sa_root @ invsqrt(middle) @ Sa_root
        arrays[f"weight_ot__anchor_{anchor}__layer_{layer}"] = transport.numpy()
        skew = 0.5 * (transport - transport.T)
        rows.append(
            {
                "anchor_layer": anchor,
                "source_layer": layer,
                "transport_kind": "weight_positive_ot",
                "ridge_multiplier": ridge,
                "compact_skew_norm": float(torch.linalg.matrix_norm(skew)),
                "symmetry_residual": float(torch.linalg.matrix_norm(transport - transport.T)),
            }
        )
    return arrays, rows


def _compact_angle_norm(Q: np.ndarray) -> float:
    angles = np.angle(np.linalg.eigvals(Q))
    return float(math.sqrt(float(np.sum(angles**2))))


def _fit_polar_on_indices(
    matrices: dict[int, torch.Tensor],
    anchor: int,
    layer: int,
    train_idx: torch.Tensor,
    eval_idx: torch.Tensor,
    ridge_multiplier: float,
    device: str,
    anchor_train_permutation: torch.Tensor | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    source_train = matrices[layer][:, train_idx].to(device)
    anchor_train = matrices[anchor][:, train_idx].to(device)
    if anchor_train_permutation is not None:
        anchor_train = anchor_train[:, anchor_train_permutation.to(anchor_train.device)]
    source_eval = matrices[layer][:, eval_idx].to(device)
    anchor_eval = matrices[anchor][:, eval_idx].to(device)
    compact, diagnostics = _fit_one(
        source_train, anchor_train, source_eval, anchor_eval, ridge_multiplier
    )
    return compact.numpy(), diagnostics


def run_layer_transport(
    weights: ModelWeights,
    edges: pd.DataFrame,
    residual_activations: dict[int, torch.Tensor],
    config: RunConfig,
) -> LayerTransportResult:
    transports, fit = fit_layer_transports(residual_activations, config)
    fit.insert(0, "model", weights.model_name)
    anchor = weights.n_layers // 2
    ridge = config.transport_ridge_grid[1]
    covariant = covariant_couplings(weights, edges, transports, anchor, ridge)
    profiles = span_profiles(edges, covariant, config)
    profiles["transport_control"] = "fitted"
    profile_frames = [profiles]
    extra_fit_rows: list[dict[str, object]] = []
    device = config.device if torch.cuda.is_available() else "cpu"
    layers, matrices, train_idx, eval_idx, sequence_length = _prepare_matrices(
        residual_activations, config
    )
    rng = np.random.default_rng(config.seed + 517)

    # Identity-transport control: collapse must be exactly zero by
    # construction, so any deviation flags a bookkeeping bug.
    identity_transports = {
        _transport_key(anchor, layer, ridge): np.eye(weights.d_model)
        for layer in layers
    }
    identity_cov = covariant_couplings(weights, edges, identity_transports, anchor, ridge)
    identity_profiles = span_profiles(edges, identity_cov, config)
    identity_profiles["transport_control"] = "identity"
    profile_frames.append(identity_profiles)
    identity_residual = float(
        np.max(np.abs(identity_cov["C_covariant"].to_numpy() - identity_cov["C"].to_numpy()))
    )
    identity_collapse = float(
        identity_profiles.groupby("edge_class")["collapse_fraction"].first().abs().max()
    )

    # Token-permuted correspondence: the fit-anything floor for the whole
    # covariant-collapse computation.
    permuted_collapses: list[float] = []
    permuted_compact: dict[int, list[float]] = {layer: [] for layer in layers}
    for draw in range(config.transport_control_draws):
        permutation = torch.tensor(
            rng.permutation(len(train_idx)), dtype=torch.long
        )
        draw_transports: dict[str, np.ndarray] = {}
        for layer in layers:
            compact, diagnostics = _fit_polar_on_indices(
                matrices, anchor, layer, train_idx, eval_idx, ridge, device,
                anchor_train_permutation=permutation,
            )
            draw_transports[_transport_key(anchor, layer, ridge)] = compact
            permuted_compact[layer].append(_compact_angle_norm(compact))
            extra_fit_rows.append(
                {
                    "model": weights.model_name,
                    "transport_kind": "token_permuted_fit",
                    "control_draw": draw,
                    "anchor_layer": anchor,
                    "source_layer": layer,
                    "ridge_multiplier": ridge,
                    **diagnostics,
                }
            )
        permuted_cov = covariant_couplings(weights, edges, draw_transports, anchor, ridge)
        permuted_profiles = span_profiles(edges, permuted_cov, config)
        permuted_profiles["transport_control"] = f"token_permuted_{draw}"
        profile_frames.append(permuted_profiles)
        permuted_collapses.append(
            float(permuted_profiles.groupby("edge_class")["collapse_fraction"].first().median())
        )

    # Random equal-size token partitions and the within-sequence bootstrap:
    # transport-fit noise floors for the compact angle.
    reference_compact = {
        layer: transports[_transport_key(anchor, layer, ridge)] for layer in layers
    }
    partition_changes: dict[int, list[float]] = {layer: [] for layer in layers}
    bootstrap_changes: dict[int, list[float]] = {layer: [] for layer in layers}
    n_train = len(train_idx)
    all_positions = torch.cat([train_idx, eval_idx])
    position_sequences = (all_positions // sequence_length).numpy()
    unique_sequences = np.unique(position_sequences)
    for draw in range(config.bootstrap_draws):
        partition = torch.tensor(
            rng.choice(len(all_positions), n_train, replace=False), dtype=torch.long
        )
        partition_idx = all_positions[partition]
        chosen = rng.choice(unique_sequences, len(unique_sequences), replace=True)
        counts = np.bincount(chosen, minlength=int(unique_sequences.max()) + 1)
        weights_per_position = counts[position_sequences]
        bootstrap_positions = all_positions[
            torch.tensor(np.repeat(np.arange(len(all_positions)), weights_per_position), dtype=torch.long)
        ]
        for layer in layers:
            compact_p, _ = _fit_polar_on_indices(
                matrices, anchor, layer, partition_idx, eval_idx, ridge, device
            )
            partition_changes[layer].append(
                _compact_angle_norm(compact_p @ reference_compact[layer].T)
            )
            compact_b, _ = _fit_polar_on_indices(
                matrices, anchor, layer, bootstrap_positions, eval_idx, ridge, device
            )
            bootstrap_changes[layer].append(
                _compact_angle_norm(compact_b @ reference_compact[layer].T)
            )
    for layer in layers:
        real_norm = _compact_angle_norm(reference_compact[layer])
        boot = np.asarray(bootstrap_changes[layer])
        part = np.asarray(partition_changes[layer])
        perm = np.asarray(permuted_compact[layer])
        extra_fit_rows.append(
            {
                "model": weights.model_name,
                "transport_kind": "compact_angle_calibration",
                "anchor_layer": anchor,
                "source_layer": layer,
                "ridge_multiplier": ridge,
                "real_compact_angle_norm": real_norm,
                "bootstrap_change_median": float(np.median(boot)) if len(boot) else math.nan,
                "bootstrap_change_robust_scale": (
                    float(1.4826 * np.median(np.abs(boot - np.median(boot))))
                    if len(boot)
                    else math.nan
                ),
                "partition_change_median": float(np.median(part)) if len(part) else math.nan,
                "token_permuted_compact_norm_median": float(np.median(perm)) if len(perm) else math.nan,
                "exceeds_bootstrap_floor": bool(
                    len(boot) and real_norm > float(np.median(boot))
                ),
            }
        )

    # Outlier-coordinate arm: gpt2-style massive stream coordinates dominate
    # the covariance and the condition number; refit with them projected out.
    anchor_rms = torch.sqrt((matrices[anchor] ** 2).mean(dim=1))
    outlier_coordinates = torch.argsort(anchor_rms, descending=True)[:2]
    deleted = {
        layer: matrix.clone() for layer, matrix in matrices.items()
    }
    for matrix in deleted.values():
        matrix[outlier_coordinates] = 0.0
    for layer in layers:
        _, diagnostics = _fit_polar_on_indices(
            deleted, anchor, layer, train_idx, eval_idx, ridge, device
        )
        extra_fit_rows.append(
            {
                "model": weights.model_name,
                "transport_kind": "outlier_deleted_fit",
                "anchor_layer": anchor,
                "source_layer": layer,
                "ridge_multiplier": ridge,
                "outlier_coordinates": [int(index) for index in outlier_coordinates],
                **diagnostics,
            }
        )

    ot_arrays, ot_rows = weight_ot_transports(weights, anchor, config.ridge_grid[1])
    transports.update(ot_arrays)
    fit = pd.concat(
        [fit, pd.DataFrame(ot_rows).assign(model=weights.model_name), pd.DataFrame(extra_fit_rows)],
        ignore_index=True,
        sort=False,
    )
    profiles = pd.concat(profile_frames, ignore_index=True, sort=False)
    fitted_profiles = profiles[profiles["transport_control"] == "fitted"]
    primary_fit = fit[
        (fit["anchor_layer"] == anchor)
        & (fit["ridge_multiplier"] == ridge)
        & fit.get("transport_kind").isna()
        & fit["heldout_r2_polar"].notna()
    ]
    fitted_collapse = float(
        fitted_profiles.groupby("edge_class")["collapse_fraction"].first().median()
    )
    permuted_floor = float(np.median(permuted_collapses)) if permuted_collapses else math.nan
    observations = [
        f"Activation-derived polar transports generalized with median held-out R2 {float(primary_fit['heldout_r2_polar'].median()):.3f} at the middle anchor.",
        f"Fitted transport collapsed span-center variance by median fraction {fitted_collapse:.3f} against a token-permuted fit-anything floor of {permuted_floor:.3f}; the identity control returned collapse {identity_collapse:.2e} and covariant-coupling residual {identity_residual:.2e}.",
        f"Pooled versus span-stratified selected-edge overlap changed from {float(fitted_profiles['raw_pooled_vs_stratified_selection_jaccard'].iloc[0]):.3f} to {float(fitted_profiles['covariant_pooled_vs_stratified_selection_jaccard'].iloc[0]):.3f} after transport.",
    ]
    return LayerTransportResult(
        transports=transports,
        fit_table=fit,
        span_profiles=profiles,
        covariant_edges=covariant,
        observations=observations,
    )

