from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pandas as pd
import torch

from .config import RunConfig
from .graph_connection import role_bridge, typed_qk_map
from .operator_core import partial_polar, psd_log
from .model_io import parse_head
from .weights import ModelWeights


def rope_frequencies(weights: ModelWeights) -> np.ndarray:
    if weights.rotary_dim <= 0:
        return np.array([], dtype=np.float64)
    indices = np.arange(0, weights.rotary_dim, 2, dtype=np.float64)
    return weights.rotary_base ** (-indices / weights.rotary_dim)


def rope_matrix(weights: ModelWeights, position: int) -> np.ndarray:
    dimension = weights.d_head
    result = np.eye(dimension, dtype=np.float64)
    frequencies = rope_frequencies(weights)
    for plane, frequency in enumerate(frequencies):
        angle = position * frequency
        c, s = math.cos(angle), math.sin(angle)
        index = 2 * plane
        result[index : index + 2, index : index + 2] = ((c, -s), (s, c))
    return result


def _rational_frequency_rank(weights: ModelWeights) -> float:
    # Released Pythia uses base 10000.  In that case theta_j=10^(-8j/r),
    # and Eisenstein gives the degree of the common radical directly.
    if not math.isclose(weights.rotary_base, 10_000.0, rel_tol=0, abs_tol=1.0e-9):
        return math.nan
    if weights.rotary_dim <= 0:
        return 0.0
    denominator = weights.rotary_dim // math.gcd(weights.rotary_dim, 8)
    residues = {
        Fraction(-8 * j, weights.rotary_dim).numerator
        % Fraction(-8 * j, weights.rotary_dim).denominator
        for j in range(weights.rotary_dim // 2)
    }
    # The residue shortcut above collapses denominators inconsistently; the
    # exact degree is bounded by q and all consecutive powers expose q classes.
    return float(min(denominator, weights.rotary_dim // 2))


def _finite_dictionary_rank(weights: ModelWeights, length: int) -> int:
    frequencies = rope_frequencies(weights)
    if not len(frequencies):
        return 0
    positions = np.arange(length)[:, None]
    dictionary = np.concatenate(
        (np.cos(positions * frequencies), np.sin(positions * frequencies)), axis=1
    )
    return int(np.linalg.matrix_rank(dictionary, tol=1.0e-10))


def _visible_weight_rank(weights: ModelWeights) -> int:
    r = weights.rotary_dim
    if r <= 0:
        return 0
    factors = []
    for layer in range(weights.n_layers):
        for head in range(weights.n_heads):
            factors.extend((weights.Q[layer, head, :, :r], weights.K[layer, head, :, :r]))
    return int(torch.linalg.matrix_rank(torch.cat(factors, dim=1), atol=1.0e-9, rtol=0).item())


def _activation_rank(positional_matrix: np.ndarray) -> int:
    singular = np.linalg.svd(positional_matrix, compute_uv=False)
    return int(np.count_nonzero(singular > 1.0e-7 * max(singular.max(), 1.0)))


def _principal_cosines(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if not left.size or not right.size:
        return np.array([])
    left = np.linalg.qr(left)[0]
    right = np.linalg.qr(right)[0]
    return np.linalg.svd(left.T @ right, compute_uv=False)


def _loop_metrics(loop: torch.Tensor) -> dict[str, float]:
    compact, positive, _, _, _, reconstruction = partial_polar(loop, 1.0e-9)
    score = 0.5 * float(torch.linalg.matrix_norm(torch.eye(len(loop)) - compact) ** 2)
    positive_log = float(torch.linalg.matrix_norm(psd_log(positive, 1.0e-9)))
    q, r = torch.linalg.qr(loop)
    diagonal = torch.diag(r)
    dilation = float(torch.linalg.vector_norm(torch.log(torch.abs(diagonal).clamp_min(1.0e-12))))
    normalized_upper = torch.diag(1.0 / torch.abs(diagonal).clamp_min(1.0e-12)) @ r
    shear = float(torch.linalg.matrix_norm(torch.triu(normalized_upper, diagonal=1)))
    return {
        "polar_compact_score": score,
        "polar_positive_log_norm": positive_log,
        "polar_reconstruction_error": reconstruction,
        "iwasawa_dilation_norm": dilation,
        "iwasawa_shear_norm": shear,
        "qr_flag_orthogonality_residual": float(torch.linalg.matrix_norm(q.T @ q - torch.eye(len(q)))),
    }


def run_rope_transport(
    weights: ModelWeights,
    role_loops: pd.DataFrame,
    positional_matrix: np.ndarray,
    rw_planes: dict[str, np.ndarray],
    normalized_activations: dict[int, torch.Tensor] | None,
    config: RunConfig,
) -> tuple[pd.DataFrame, list[str]]:
    rows: list[dict[str, object]] = []
    scheme = weights.positional_scheme
    if weights.rotary_dim <= 0 or scheme != "rotary":
        rows.append(
            {
                "model": weights.model_name,
                "positional_scheme": scheme,
                "control": "learned_position_reference",
                "allocated_plane_count": 0,
                "pass_through_dimension": weights.d_head,
                "exact_rope_available": False,
                "activation_realized_rank": _activation_rank(positional_matrix),
            }
        )
        return pd.DataFrame(rows), [
            f"{weights.model_name} uses learned positional embeddings, so no fixed RoPE loop was subtracted."
        ]

    positions = (0, max(1, config.position_length // 2), config.position_length - 1)
    k_pos, j_pos, i_pos = positions
    R_ij = rope_matrix(weights, i_pos - j_pos)
    R_jk = rope_matrix(weights, j_pos - k_pos)
    R_ik = rope_matrix(weights, i_pos - k_pos)
    exact_loop = R_ij @ R_jk @ np.linalg.inv(R_ik)
    exact_residual = float(np.linalg.norm(exact_loop - np.eye(weights.d_head)))
    frequencies = rope_frequencies(weights)
    visible_factors = []
    for layer in range(weights.n_layers):
        for head in range(weights.n_heads):
            visible_factors.extend(
                (
                    weights.Q[layer, head, :, : weights.rotary_dim].numpy(),
                    weights.K[layer, head, :, : weights.rotary_dim].numpy(),
                )
            )
    visible_basis = np.linalg.qr(np.concatenate(visible_factors, axis=1))[0]
    pos_basis = np.linalg.svd(positional_matrix, full_matrices=False)[2].T
    pos_rank = _activation_rank(positional_matrix)
    cos_pos = _principal_cosines(visible_basis, pos_basis[:, : max(1, min(pos_rank, visible_basis.shape[1]))])
    posratio = rw_planes.get("posratio_plane", np.empty((weights.d_model, 0)))
    cos_rw = _principal_cosines(visible_basis, posratio)
    rows.append(
        {
            "model": weights.model_name,
            "positional_scheme": scheme,
            "control": "exact_rope",
            "exact_rope_available": True,
            "allocated_plane_count": weights.rotary_dim // 2,
            "distinct_active_character_count": len(np.unique(frequencies)),
            "rational_frequency_rank": _rational_frequency_rank(weights),
            "finite_window_dictionary_rank": _finite_dictionary_rank(weights, config.position_length),
            "learned_weight_visible_rank": _visible_weight_rank(weights),
            "activation_realized_rank": pos_rank,
            "pass_through_dimension": weights.d_head - weights.rotary_dim,
            "exact_loop_residual": exact_residual,
            "minimum_positional_span_cosine_to_rope_visible": float(cos_pos.min()) if len(cos_pos) else math.nan,
            "minimum_posratio_cosine_to_rope_visible": float(cos_rw.min()) if len(cos_rw) else math.nan,
        }
    )

    # Phase shuffling changes the character assignment but not flatness.
    rng = np.random.default_rng(config.seed + 701)
    shuffled = frequencies.copy()
    rng.shuffle(shuffled)
    def matrix_from_frequency(position, freq):
        result = np.eye(weights.d_head)
        for plane, value in enumerate(freq):
            angle = position * value
            result[2 * plane : 2 * plane + 2, 2 * plane : 2 * plane + 2] = (
                (math.cos(angle), -math.sin(angle)),
                (math.sin(angle), math.cos(angle)),
            )
        return result
    shuffled_loop = (
        matrix_from_frequency(i_pos - j_pos, shuffled)
        @ matrix_from_frequency(j_pos - k_pos, shuffled)
        @ np.linalg.inv(matrix_from_frequency(i_pos - k_pos, shuffled))
    )
    rows.append(
        {
            "model": weights.model_name,
            "positional_scheme": scheme,
            "control": "phase_shuffled_exact_rope",
            "exact_loop_residual": float(np.linalg.norm(shuffled_loop - np.eye(weights.d_head))),
            "allocated_plane_count": weights.rotary_dim // 2,
            "pass_through_dimension": weights.d_head - weights.rotary_dim,
        }
    )

    # Recompute representative role-complete loops with exact positional
    # transports inserted; the exact loop is divided out explicitly.
    valid_roles = role_loops[
        role_loops.get("local_error", pd.Series(index=role_loops.index, dtype=object)).isna()
    ] if len(role_loops) else role_loops
    valid_roles = valid_roles.drop_duplicates(["k", "j", "i", "bridge", "ridge"])
    for role in valid_roles.itertuples():
        k, j, i = role.k, role.j, role.i
        Tij = typed_qk_map(weights, j, i)
        Tjk = typed_qk_map(weights, k, j)
        Tik = typed_qk_map(weights, k, i)
        activation = None
        if normalized_activations is not None:
            activation = normalized_activations.get(parse_head(j)[0])
        try:
            bridge = role_bridge(
                weights,
                j,
                role.bridge,
                activation,
                float(role.ridge),
                config.support_rtol,
            )
            Rij = torch.from_numpy(R_ij).double()
            Rjk = torch.from_numpy(R_jk).double()
            Rik = torch.from_numpy(R_ik).double()
            learned = (
                Tij
                @ Rij
                @ torch.linalg.pinv(bridge, rtol=config.support_rtol)
                @ Tjk
                @ Rjk
                @ torch.linalg.pinv(Tik @ Rik, rtol=config.support_rtol)
            )
            # The exact flat contribution is identity at loop closure; retain
            # the explicit multiplication to make that subtraction auditable.
            exact_t = torch.from_numpy(exact_loop).double()
            residual = learned @ torch.linalg.inv(exact_t)
            rows.append(
                {
                    "model": weights.model_name,
                    "positional_scheme": scheme,
                    "control": "learned_role_complete_residual",
                    "k": k,
                    "j": j,
                    "i": i,
                    "bridge": role.bridge,
                    "ridge": float(role.ridge),
                    "exact_loop_residual": exact_residual,
                    **_loop_metrics(residual),
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "model": weights.model_name,
                    "positional_scheme": scheme,
                    "control": "learned_role_complete_residual",
                    "k": k,
                    "j": j,
                    "i": i,
                    "bridge": role.bridge,
                    "ridge": float(role.ridge),
                    "local_error": f"{type(exc).__name__}: {exc}",
                }
            )
    frame = pd.DataFrame(rows)
    learned_rows = frame[frame["control"] == "learned_role_complete_residual"]
    observations = [
        f"Exact RoPE loops closed with residual {exact_residual:.3e}; this is the flat positional control.",
        f"The rotary block allocated {weights.rotary_dim // 2} planes, had finite-window dictionary rank {_finite_dictionary_rank(weights, config.position_length)}, and left {weights.d_head - weights.rotary_dim} head coordinates position-transparent.",
        (
            f"After exact flat subtraction, learned role-complete loops had median compact score {float(learned_rows['polar_compact_score'].median()):.3g} and median positive-log norm {float(learned_rows['polar_positive_log_norm'].median()):.3g}."
            if len(learned_rows) and "polar_compact_score" in learned_rows
            else "Learned RoPE residuals were unresolved because no role-complete loop was available."
        ),
    ]
    return frame, observations
