from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from scipy import linalg as spla

from .config import RunConfig
from .model_io import parse_head
from .weights import ModelWeights, head_norms, inner_grams


@dataclass(slots=True)
class OperatorResult:
    records: pd.DataFrame
    head_census: pd.DataFrame
    thomas_wigner: pd.DataFrame
    arrays: dict[str, np.ndarray]
    observations: list[str]


def psd_function(
    matrix: torch.Tensor,
    function,
    *,
    rtol: float = 1.0e-12,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    matrix = 0.5 * (matrix + matrix.T)
    values, vectors = torch.linalg.eigh(matrix)
    scale = max(float(values.max()), 0.0)
    keep = values > max(rtol * scale, 0.0)
    transformed = torch.zeros_like(values)
    if keep.any():
        transformed[keep] = function(values[keep])
    return (vectors * transformed) @ vectors.T, values, keep


def psd_sqrt(matrix: torch.Tensor, rtol: float = 1.0e-12) -> torch.Tensor:
    return psd_function(matrix, torch.sqrt, rtol=rtol)[0]


def psd_log(matrix: torch.Tensor, floor: float) -> torch.Tensor:
    values, vectors = torch.linalg.eigh(0.5 * (matrix + matrix.T))
    values = values.clamp_min(floor)
    return (vectors * torch.log(values)) @ vectors.T


def partial_polar(
    matrix: torch.Tensor, rtol: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, float]:
    U, singular, Vh = torch.linalg.svd(matrix, full_matrices=False)
    threshold = rtol * max(float(singular.max()), 1.0)
    keep = singular > threshold
    rank = int(keep.sum().item())
    if rank:
        Ur = U[:, keep]
        Vr = Vh[keep].T
        sr = singular[keep]
        polar = Ur @ Vr.T
        right = (Vr * sr) @ Vr.T
        left = (Ur * sr) @ Ur.T
        reconstruction = polar @ right
    else:
        polar = torch.zeros_like(matrix)
        right = torch.zeros((matrix.shape[1], matrix.shape[1]), dtype=matrix.dtype)
        left = torch.zeros((matrix.shape[0], matrix.shape[0]), dtype=matrix.dtype)
        reconstruction = torch.zeros_like(matrix)
    residual = float(
        torch.linalg.matrix_norm(matrix - reconstruction)
        / max(float(torch.linalg.matrix_norm(matrix)), 1.0e-300)
    )
    return polar, right, left, singular, rank, residual


def _edge_factors(
    weights: ModelWeights,
    channel: str,
    writer_layer: int,
    writer_head: int,
    reader_layer: int,
    reader_head: int,
    grams: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if channel == "K":
        reader_left = weights.Q[reader_layer, reader_head]
        reader_right = weights.K[reader_layer, reader_head]
        left_metric = grams["Q"][reader_layer, reader_head]
    elif channel == "Q":
        reader_left = weights.K[reader_layer, reader_head]
        reader_right = weights.Q[reader_layer, reader_head]
        left_metric = grams["K"][reader_layer, reader_head]
    elif channel == "V":
        reader_left = weights.O[reader_layer, reader_head]
        reader_right = weights.V[reader_layer, reader_head]
        left_metric = grams["O"][reader_layer, reader_head]
    else:
        raise ValueError(f"operator core only supports K/Q/V, received {channel}")
    writer_left = weights.O[writer_layer, writer_head]
    writer_right = weights.V[writer_layer, writer_head]
    raw_cross = reader_right.T @ writer_left
    # A = stream_left @ stream_right.T.
    stream_left = reader_left @ raw_cross
    stream_right = writer_right
    return raw_cross, left_metric, grams["V"][writer_layer, writer_head], stream_left, stream_right


def _stream_energies(left: torch.Tensor, right: torch.Tensor):
    frob2 = float(torch.sum((left.T @ left) * (right.T @ right)))
    small = right.T @ left
    trace_a2 = float(torch.trace(small @ small))
    sym2 = max(0.5 * (frob2 + trace_a2), 0.0)
    skew2 = max(0.5 * (frob2 - trace_a2), 0.0)
    return frob2, sym2, skew2


def _selection(edges: pd.DataFrame, config: RunConfig) -> pd.DataFrame:
    head = edges[edges["edge_class"].str.startswith("head_head_")].copy()
    chosen: list[pd.DataFrame] = []
    rng = np.random.default_rng(config.seed)
    cap = config.max_operator_edges_per_channel
    for channel, group in head.groupby("channel"):
        n_high = min(int(math.ceil(0.65 * cap)), int(group["selected"].sum()))
        high = group[group["selected"]].nlargest(n_high, "empirical_z")
        remaining = group.drop(high.index)
        n_low = min(int(math.ceil(0.2 * cap)), int(remaining["avoidant"].sum()))
        low = remaining[remaining["avoidant"]].nsmallest(n_low, "empirical_z")
        remaining = remaining.drop(low.index)
        n_random = min(cap - len(high) - len(low), len(remaining))
        random = (
            remaining.sample(n_random, random_state=config.seed)
            if n_random
            else remaining.iloc[:0]
        )
        high = high.assign(analysis_selection="selected_high")
        low = low.assign(analysis_selection="selected_avoidant")
        random = random.assign(analysis_selection="unselected_control")
        chosen.extend((high, low, random))
    if not chosen:
        return head.iloc[:0]
    return pd.concat(chosen).sort_values(["channel", "writer_layer", "reader_layer"])


def head_operator_census(weights: ModelWeights) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for layer in range(weights.n_layers):
        for head in range(weights.n_heads):
            O = weights.O[layer, head]
            V = weights.V[layer, head]
            small = V.T @ O
            eigenvalues = torch.linalg.eigvals(small)
            denom = float(torch.sum(torch.abs(eigenvalues)))
            copying = float(eigenvalues.real.sum()) / max(denom, 1.0e-300)
            frob2, sym2, skew2 = _stream_energies(O, V)
            go = O.T @ O
            gv = V.T @ V
            ov = O.T @ V
            aa_norm2 = float(torch.trace(gv @ go @ gv @ go))
            cross = float(torch.trace(gv @ ov @ go @ ov.T))
            commutator_norm = math.sqrt(max(2.0 * aa_norm2 - 2.0 * cross, 0.0))
            nonnormality = commutator_norm / max(frob2, 1.0e-300)
            # Thin QR reduces the polar calculation to one d_head square SVD.
            Qo, Ro = torch.linalg.qr(O, mode="reduced")
            Qv, Rv = torch.linalg.qr(V, mode="reduced")
            middle = Ro @ Rv.T
            Um, singular, Vhm = torch.linalg.svd(middle, full_matrices=False)
            threshold = 1.0e-9 * max(float(singular.max()), 1.0)
            keep = singular > threshold
            rank = int(keep.sum().item())
            if rank:
                compact_small = Um[:, keep] @ Vhm[keep]
                trace_partial = float(torch.trace(compact_small @ (Qv.T @ Qo)))
                polar_distance = 0.5 * max(weights.d_model + rank - 2.0 * trace_partial, 0.0)
            else:
                polar_distance = 0.5 * weights.d_model
            residual = float(
                torch.linalg.vector_norm(singular[~keep])
                / max(float(torch.linalg.vector_norm(singular)), 1.0e-300)
            )
            signed_mass = float(eigenvalues.real.sum()) / max(
                float(torch.abs(eigenvalues.real).sum()), 1.0e-300
            )
            rows.append(
                {
                    "model": weights.model_name,
                    "head": weights.head_label(layer, head),
                    "layer": layer,
                    "copying_score": copying,
                    "signed_eigenvalue_mass": signed_mass,
                    "trace": float(torch.trace(small)),
                    "spectral_radius": float(torch.abs(eigenvalues).max()),
                    "nonnormality": nonnormality,
                    "symmetric_energy_fraction": sym2 / max(frob2, 1.0e-300),
                    "skew_energy_fraction": skew2 / max(frob2, 1.0e-300),
                    "support_rank": rank,
                    "polar_reconstruction_error": residual,
                    "polar_orientation_distance": polar_distance,
                    "largest_singular_value": float(singular.max()),
                }
            )
    return pd.DataFrame(rows)


def build_operator_cores(
    weights: ModelWeights, edges: pd.DataFrame, config: RunConfig
) -> tuple[pd.DataFrame, dict[str, np.ndarray], pd.DataFrame]:
    grams = inner_grams(weights)
    norms = head_norms(weights, grams)
    chosen = _selection(edges, config)
    records: list[dict[str, object]] = []
    ids: list[str] = []
    cores: list[np.ndarray] = []
    polars: list[np.ndarray] = []
    right_positive: list[np.ndarray] = []
    left_positive: list[np.ndarray] = []
    singular_values: list[np.ndarray] = []
    support_projectors_left: list[np.ndarray] = []
    support_projectors_right: list[np.ndarray] = []
    for edge in chosen.itertuples():
        wl, wh = int(edge.writer_layer), parse_head(str(edge.writer))[1]
        rl, rh = int(edge.reader_layer), parse_head(str(edge.reader))[1]
        raw, gm_left, gm_right, stream_left, stream_right = _edge_factors(
            weights, edge.channel, wl, wh, rl, rh, grams
        )
        metric_core = psd_sqrt(gm_left) @ raw @ psd_sqrt(gm_right)
        denominator = math.sqrt(
            float(norms[edge.channel][rl, rh] * norms["writer"][wl, wh])
        )
        core = metric_core / max(denominator, 1.0e-300)
        polar, right, left, singular, rank, reconstruction = partial_polar(
            core, config.support_rtol
        )
        Ul, _, _ = torch.linalg.svd(core, full_matrices=False)
        _, _, Vh = torch.linalg.svd(core, full_matrices=False)
        threshold = config.support_rtol * max(float(singular.max()), 1.0)
        keep = singular > threshold
        p_left = Ul[:, keep] @ Ul[:, keep].T if keep.any() else torch.zeros_like(left)
        Vr = Vh[keep].T
        p_right = Vr @ Vr.T if keep.any() else torch.zeros_like(right)
        frob2, sym2, skew2 = _stream_energies(stream_left, stream_right)
        edge_id = f"{weights.model_name}:{edge.writer}->{edge.reader}:{edge.channel}"
        determinant_sign = np.nan
        determinant_logabs = np.nan
        if rank == core.shape[0]:
            sign, logabs = torch.linalg.slogdet(core)
            determinant_sign = float(sign)
            determinant_logabs = float(logabs)
        records.append(
            {
                "model": weights.model_name,
                "edge_id": edge_id,
                "writer": edge.writer,
                "reader": edge.reader,
                "writer_layer": wl,
                "reader_layer": rl,
                "channel": edge.channel,
                "analysis_selection": edge.analysis_selection,
                "C": float(edge.C),
                "core_frobenius": float(torch.linalg.matrix_norm(core)),
                "core_C_residual": abs(float(torch.linalg.matrix_norm(core)) - float(edge.C)),
                "support_rank": rank,
                "polar_reconstruction_error": reconstruction,
                "support_projector_left_residual": float(torch.linalg.matrix_norm(p_left @ p_left - p_left)),
                "support_projector_right_residual": float(torch.linalg.matrix_norm(p_right @ p_right - p_right)),
                "stream_symmetric_energy": sym2,
                "stream_skew_energy": skew2,
                "stream_symmetric_fraction": sym2 / max(frob2, 1.0e-300),
                "stream_skew_fraction": skew2 / max(frob2, 1.0e-300),
                "signed_determinant": determinant_sign,
                "log_absolute_determinant": determinant_logabs,
            }
        )
        ids.append(edge_id)
        cores.append(core.numpy())
        polars.append(polar.numpy())
        right_positive.append(right.numpy())
        left_positive.append(left.numpy())
        singular_values.append(singular.numpy())
        support_projectors_left.append(p_left.numpy())
        support_projectors_right.append(p_right.numpy())
    arrays = {
        "edge_ids": np.asarray(ids, dtype="U256"),
        "normalized_core": np.stack(cores) if cores else np.empty((0, weights.d_head, weights.d_head)),
        "partial_polar": np.stack(polars) if polars else np.empty((0, weights.d_head, weights.d_head)),
        "right_positive": np.stack(right_positive) if right_positive else np.empty((0, weights.d_head, weights.d_head)),
        "left_positive": np.stack(left_positive) if left_positive else np.empty((0, weights.d_head, weights.d_head)),
        "singular_values": np.stack(singular_values) if singular_values else np.empty((0, weights.d_head)),
        "support_projector_left": np.stack(support_projectors_left) if support_projectors_left else np.empty((0, weights.d_head, weights.d_head)),
        "support_projector_right": np.stack(support_projectors_right) if support_projectors_right else np.empty((0, weights.d_head, weights.d_head)),
    }
    return pd.DataFrame(records), arrays, chosen


def _orthogonal_polar(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    U, singular, Vh = np.linalg.svd(matrix, full_matrices=False)
    return U @ Vh, (Vh.T * singular) @ Vh


def _rotation_angle_2d(matrix: np.ndarray) -> float:
    return float(
        math.atan2(
            matrix[1, 0] - matrix[0, 1],
            matrix[0, 0] + matrix[1, 1],
        )
    )


def _dominant_plane(orthogonal: np.ndarray) -> tuple[np.ndarray, float]:
    logarithm = spla.logm(orthogonal)
    logarithm = np.real_if_close(logarithm, tol=1_000).real
    skew = 0.5 * (logarithm - logarithm.T)
    U, singular, _ = np.linalg.svd(skew)
    if len(singular) < 2 or singular[0] < 1.0e-14:
        return U[:, :2], 0.0
    E = U[:, :2]
    block = E.T @ skew @ E
    if block[1, 0] < 0:
        E[:, 1] *= -1
        block = E.T @ skew @ E
    explained = float(np.sum(singular[:2] ** 2) / max(np.sum(singular**2), 1.0e-300))
    return E, explained


def _major_eigenvector(matrix: np.ndarray) -> tuple[float, np.ndarray]:
    values, vectors = np.linalg.eigh(0.5 * (matrix + matrix.T))
    return float(values[-1]), vectors[:, -1]


def _thomas_wigner_edge(
    weights: ModelWeights,
    edge,
    ridge: float,
) -> dict[str, object]:
    wl, wh = int(edge.writer_layer), parse_head(str(edge.writer))[1]
    rl, rh = int(edge.reader_layer), parse_head(str(edge.reader))[1]
    # V reader Gram G=M_r^T M_r and writer Gram H=M_w M_w^T.
    Vr, Or = weights.V[rl, rh], weights.O[rl, rh]
    Vw, Ow = weights.V[wl, wh], weights.O[wl, wh]
    go_r = Or.T @ Or
    gv_w = Vw.T @ Vw
    support_source = torch.cat((Vr, Ow), dim=1)
    U, singular, _ = torch.linalg.svd(support_source, full_matrices=False)
    keep = singular > 1.0e-10 * max(float(singular.max()), 1.0)
    basis = U[:, keep]
    G = (basis.T @ Vr) @ go_r @ (basis.T @ Vr).T
    H = (basis.T @ Ow) @ gv_w @ (basis.T @ Ow).T
    q = G / torch.trace(G)
    K = H / torch.trace(H)
    eye = torch.eye(len(q), dtype=torch.float64)
    A = psd_sqrt(q + ridge * eye, rtol=0.0).numpy()
    B = psd_sqrt(K + ridge * eye, rtol=0.0).numpy()
    Qab, _ = _orthogonal_polar(A @ B)
    E, explained = _dominant_plane(Qab)
    AE = 0.5 * (E.T @ A @ E + E.T @ A.T @ E)
    BE = 0.5 * (E.T @ B @ E + E.T @ B.T @ E)
    det_a = max(float(np.linalg.det(AE)), 1.0e-300)
    det_b = max(float(np.linalg.det(BE)), 1.0e-300)
    AE /= math.sqrt(det_a)
    BE /= math.sqrt(det_b)
    amax, va = _major_eigenvector(AE)
    bmax, vb = _major_eigenvector(BE)
    if np.linalg.det(np.column_stack((va, np.array([-va[1], va[0]])))) < 0:
        va = -va
    delta = math.atan2(va[0] * vb[1] - va[1] * vb[0], float(va @ vb))
    # Eigenvectors are axes; choose the representative in [-pi/2,pi/2].
    delta = ((delta + math.pi / 2) % math.pi) - math.pi / 2
    t = 0.5 * math.log(max(amax, 1.0e-300) / max(1.0 / amax, 1.0e-300))
    s = 0.5 * math.log(max(bmax, 1.0e-300) / max(1.0 / bmax, 1.0e-300))
    p = math.tanh(t)
    qg = math.tanh(s)
    alpha = 2.0 * delta
    predicted = math.atan2(
        -math.sin(alpha) * p * qg,
        1.0 + math.cos(alpha) * p * qg,
    )
    Qpair, _ = _orthogonal_polar(AE @ BE)
    pair = _rotation_angle_2d(Qpair)

    full_plane = basis.numpy() @ E
    cross_v = Vr.T @ Ow
    plane_t = torch.from_numpy(full_plane).to(dtype=torch.float64)
    model_2 = (
        (plane_t.T @ Or)
        @ cross_v
        @ (Vw.T @ plane_t)
    ).numpy()
    Qmodel, _ = _orthogonal_polar(model_2)
    model_angle = _rotation_angle_2d(Qmodel)
    Qreverse, _ = _orthogonal_polar(BE @ AE)
    reverse = _rotation_angle_2d(Qreverse)
    q_vr = torch.linalg.qr(Vr, mode="reduced").Q
    q_ow = torch.linalg.qr(Ow, mode="reduced").Q
    intersection_singular = np.linalg.svd((q_vr.T @ q_ow).numpy(), compute_uv=False)
    intersection_dimension = int(np.count_nonzero(intersection_singular > 1.0 - 1.0e-7))
    return {
        "model": weights.model_name,
        "edge_id": f"{weights.model_name}:{edge.writer}->{edge.reader}:V",
        "writer": edge.writer,
        "reader": edge.reader,
        "ridge": ridge,
        "union_support_dimension": int(len(q)),
        "intersection_support_dimension": intersection_dimension,
        "t": t,
        "s": s,
        "delta": delta,
        "alpha": alpha,
        "phi_predicted": predicted,
        "phi_pair": pair,
        "pair_formula_residual": abs(math.atan2(math.sin(pair - predicted), math.cos(pair - predicted))),
        "phi_model": model_angle,
        "model_prediction_residual": abs(math.atan2(math.sin(model_angle - predicted), math.cos(model_angle - predicted))),
        "phi_reverse": reverse,
        "forward_reverse_oddness_residual": abs(math.atan2(math.sin(pair + reverse), math.cos(pair + reverse))),
        "envelope": abs(p * qg),
        "envelope_occupancy": abs(math.sin(model_angle)) / max(abs(p * qg), 1.0e-12),
        "dominant_plane_compact_energy_fraction": explained,
    }


def run_thomas_wigner(
    weights: ModelWeights, analyzed_edges: pd.DataFrame, config: RunConfig
) -> pd.DataFrame:
    selected = analyzed_edges[
        (analyzed_edges["channel"] == "V")
        & (analyzed_edges["analysis_selection"] != "unselected_control")
    ]
    rows: list[dict[str, object]] = []
    for edge in selected.itertuples():
        for ridge in config.ridge_grid:
            try:
                rows.append(_thomas_wigner_edge(weights, edge, ridge))
            except Exception as exc:
                rows.append(
                    {
                        "model": weights.model_name,
                        "edge_id": f"{weights.model_name}:{edge.writer}->{edge.reader}:V",
                        "writer": edge.writer,
                        "reader": edge.reader,
                        "ridge": ridge,
                        "local_error": f"{type(exc).__name__}: {exc}",
                    }
                )
    return pd.DataFrame(rows)


def run_operator_experiments(
    weights: ModelWeights, edges: pd.DataFrame, config: RunConfig
) -> OperatorResult:
    records, arrays, analyzed = build_operator_cores(weights, edges, config)
    census = head_operator_census(weights)
    tw = run_thomas_wigner(weights, analyzed, config)
    core_residual = float(records["core_C_residual"].max()) if len(records) else math.nan
    valid_tw = tw[tw.get("local_error", pd.Series(index=tw.index, dtype=object)).isna()] if len(tw) else tw
    pair_residual = float(valid_tw["pair_formula_residual"].max()) if len(valid_tw) else math.nan
    model_residual = float(valid_tw["model_prediction_residual"].median()) if len(valid_tw) else math.nan
    observations = [
        f"Every retained small core reconstructed C with maximum residual {core_residual:.3e}.",
        f"The rank-two Thomas-Wigner formula reproduced its projected positive-pair polar angle with maximum angular residual {pair_residual:.3e} radians.",
        f"Real V-composition orientations differed from the canonical pair prediction by a median {model_residual:.3f} radians.",
    ]
    return OperatorResult(
        records=records,
        head_census=census,
        thomas_wigner=tw,
        arrays=arrays,
        observations=observations,
    )
