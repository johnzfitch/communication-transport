from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from .config import RunConfig
from .model_io import Corpora, cache_activations
from .weights import ModelWeights


@dataclass(slots=True)
class RWPCAResult:
    table: pd.DataFrame
    planes: dict[str, np.ndarray]
    arrays: dict[str, np.ndarray]
    positional_matrix: np.ndarray
    natural_activation_matrix: np.ndarray
    observations: list[str]


def _add_normalized_state(state: torch.Tensor, factor: torch.Tensor) -> None:
    if (
        torch.cuda.is_available()
        and factor.device.type == "cpu"
        and factor.shape[1] > 4 * factor.shape[0]
    ):
        working = factor.float().cuda()
        gram = (working @ working.T).double().cpu()
        del working
    else:
        gram = factor @ factor.T
    trace = torch.trace(gram)
    if float(trace) > 0:
        state += gram / trace


def pooled_state(
    weights: ModelWeights,
    *,
    Q: torch.Tensor | None = None,
    K: torch.Tensor | None = None,
    V: torch.Tensor | None = None,
    O: torch.Tensor | None = None,
    interface_state: torch.Tensor | None = None,
) -> torch.Tensor:
    Q = weights.Q if Q is None else Q
    K = weights.K if K is None else K
    V = weights.V if V is None else V
    O = weights.O if O is None else O
    S = (
        interface_state.clone()
        if interface_state is not None
        else torch.zeros(weights.d_model, weights.d_model, dtype=torch.float64)
    )
    for layer in range(weights.n_layers):
        for head in range(weights.n_heads):
            for factor in (Q[layer, head], K[layer, head], V[layer, head], O[layer, head]):
                _add_normalized_state(S, factor)
    if interface_state is None:
        _add_normalized_state(S, weights.W_E.T)
        if weights.W_pos is not None:
            _add_normalized_state(S, weights.W_pos.T)
        _add_normalized_state(S, weights.W_U)
    return 0.5 * (S + S.T)


def _balanced_pair(left: torch.Tensor, right: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    Ql, Rl = torch.linalg.qr(left, mode="reduced")
    Qr, Rr = torch.linalg.qr(right, mode="reduced")
    U, singular, Vh = torch.linalg.svd(Rl @ Rr.T, full_matrices=False)
    root = torch.sqrt(singular.clamp_min(0.0))
    return (Ql @ U) * root, (Qr @ Vh.T) * root


def balanced_factors(weights: ModelWeights):
    Q = torch.empty_like(weights.Q)
    K = torch.empty_like(weights.K)
    O = torch.empty_like(weights.O)
    V = torch.empty_like(weights.V)
    for layer in range(weights.n_layers):
        for head in range(weights.n_heads):
            Q[layer, head], K[layer, head] = _balanced_pair(
                weights.Q[layer, head], weights.K[layer, head]
            )
            O[layer, head], V[layer, head] = _balanced_pair(
                weights.O[layer, head], weights.V[layer, head]
            )
    return Q, K, V, O


def _diagonal_rebalanced(
    weights: ModelWeights, seed: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    Q, K, V, O = (item.clone() for item in (weights.Q, weights.K, weights.V, weights.O))
    for layer in range(weights.n_layers):
        for head in range(weights.n_heads):
            qscale = torch.tensor(
                np.exp(rng.uniform(-0.8, 0.8, weights.d_head)), dtype=torch.float64
            )
            oscale = torch.tensor(
                np.exp(rng.uniform(-0.8, 0.8, weights.d_head)), dtype=torch.float64
            )
            Q[layer, head] *= qscale
            K[layer, head] /= qscale
            O[layer, head] *= oscale
            V[layer, head] /= oscale
    return Q, K, V, O


def _orth(matrix: np.ndarray, rank: int | None = None) -> np.ndarray:
    if matrix.size == 0:
        return np.empty((matrix.shape[0], 0))
    Q, R = np.linalg.qr(matrix)
    if rank is None:
        diagonal = np.abs(np.diag(R))
        rank = int(np.count_nonzero(diagonal > 1.0e-9 * max(diagonal.max(), 1.0)))
    return Q[:, :rank]


def _principal_cosines(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if left.shape[1] == 0 or right.shape[1] == 0:
        return np.array([])
    return np.linalg.svd(left.T @ right, compute_uv=False)


def _activation_matrices(model, corpora: Corpora, config: RunConfig):
    middle = model.cfg.n_layers // 2
    name = f"blocks.{middle}.hook_resid_pre"
    pos_cache = cache_activations(model, corpora.position, [name], config)[name]
    positional = pos_cache.mean(dim=0).double().numpy()
    positional -= positional.mean(axis=0, keepdims=True)
    natural_cache = cache_activations(model, corpora.natural, [name], config)[name]
    natural = natural_cache.reshape(-1, model.cfg.d_model).double().numpy()
    return positional, natural, middle


def _stable_cluster(indices: list[int], eigenvalues: np.ndarray, threshold: float) -> list[int]:
    selected = set(indices)
    changed = True
    log_values = np.log(np.clip(eigenvalues, 1.0e-300, None))
    while changed:
        changed = False
        for index in list(selected):
            for neighbor in (index - 1, index + 1):
                if 0 <= neighbor < len(eigenvalues) and abs(log_values[index] - log_values[neighbor]) < threshold and neighbor not in selected:
                    selected.add(neighbor)
                    changed = True
    return sorted(selected)


def run_rw_pca(
    model, weights: ModelWeights, corpora: Corpora, config: RunConfig
) -> RWPCAResult:
    positional, natural, middle = _activation_matrices(model, corpora, config)
    interface_state = torch.zeros(
        weights.d_model, weights.d_model, dtype=torch.float64
    )
    _add_normalized_state(interface_state, weights.W_E.T)
    if weights.W_pos is not None:
        _add_normalized_state(interface_state, weights.W_pos.T)
    _add_normalized_state(interface_state, weights.W_U)
    S = pooled_state(weights, interface_state=interface_state)
    eigvals_t, eigvecs_t = torch.linalg.eigh(S)
    eigvals = eigvals_t.flip(0).numpy()
    eigvecs = eigvecs_t.flip(1).numpy()
    top_n = min(20, weights.d_model)
    token = weights.W_E.numpy()
    pos_energy = max(float(np.linalg.norm(positional) ** 2), 1.0e-300)
    tok_energy = max(float(np.linalg.norm(token) ** 2), 1.0e-300)
    shares: dict[int, tuple[float, float, float]] = {}
    for index in range(top_n):
        vector = eigvecs[:, index]
        p = float(np.linalg.norm(positional @ vector) ** 2 / pos_energy * weights.d_model)
        t = float(np.linalg.norm(token @ vector) ** 2 / tok_energy * weights.d_model)
        shares[index] = (p, t, p / max(t, 1.0e-12))
    pair = sorted(sorted(shares, key=lambda i: -shares[i][2])[:2])
    top_pair = [0, 1]

    centered_natural = natural - natural.mean(axis=0, keepdims=True)
    pca_device = "cuda" if torch.cuda.is_available() else "cpu"
    natural_t = torch.from_numpy(centered_natural).float().to(pca_device)
    torch.manual_seed(config.seed + 304)
    _, _, activation_vectors = torch.pca_lowrank(
        natural_t, q=min(6, weights.d_model), center=False, niter=4
    )
    activation_pca = activation_vectors[:, :2].double().cpu().numpy()
    del natural_t, activation_vectors
    rms = np.sqrt(np.mean(natural**2, axis=0))
    outlier_indices = np.argsort(rms)[::-1][:2]
    outlier = np.zeros((weights.d_model, 2))
    outlier[outlier_indices, np.arange(2)] = 1.0

    gauge_states: dict[str, torch.Tensor] = {"trained": S}
    balanced = balanced_factors(weights)
    gauge_states["balanced"] = pooled_state(
        weights,
        Q=balanced[0],
        K=balanced[1],
        V=balanced[2],
        O=balanced[3],
        interface_state=interface_state,
    )
    for seed in (config.seed + 101, config.seed + 202):
        factors = _diagonal_rebalanced(weights, seed)
        gauge_states[f"diagonal_rebalance_{seed}"] = pooled_state(
            weights,
            Q=factors[0],
            K=factors[1],
            V=factors[2],
            O=factors[3],
            interface_state=interface_state,
        )
    gauge_planes: dict[str, np.ndarray] = {}
    table_rows: list[dict[str, object]] = []
    for gauge, state in gauge_states.items():
        values, vectors = torch.linalg.eigh(state)
        vectors_np = vectors.flip(1).numpy()
        local_shares = {}
        for index in range(min(10, weights.d_model)):
            v = vectors_np[:, index]
            p = float(np.linalg.norm(positional @ v) ** 2 / pos_energy * weights.d_model)
            t = float(np.linalg.norm(token @ v) ** 2 / tok_energy * weights.d_model)
            local_shares[index] = p / max(t, 1.0e-12)
        local_pair = sorted(sorted(local_shares, key=local_shares.get, reverse=True)[:2])
        gauge_planes[gauge] = vectors_np[:, local_pair]
        table_rows.append(
            {
                "model": weights.model_name,
                "row_kind": "gauge_plane",
                "gauge": gauge,
                "selected_bands": local_pair,
                "projector_dimension": 2,
            }
        )
    joined = _orth(np.concatenate(list(gauge_planes.values()), axis=1))
    projector_sum = sum(plane @ plane.T for plane in gauge_planes.values())
    projector_spectrum = np.linalg.eigvalsh(projector_sum)[::-1]

    # Cluster family; the 0.03 log-gap cluster is used for the deletion arm,
    # while all thresholds remain visible in the table.
    cluster_by_threshold: dict[float, list[int]] = {}
    for threshold in (0.01, 0.03, 0.1):
        cluster = _stable_cluster(pair, eigvals[:top_n], threshold)
        cluster_by_threshold[threshold] = cluster
        table_rows.append(
            {
                "model": weights.model_name,
                "row_kind": "spectral_cluster",
                "cluster_threshold": threshold,
                "cluster_indices": cluster,
                "projector_dimension": len(cluster),
            }
        )
    stable_indices = cluster_by_threshold[0.03]

    # Bootstrap the projector chosen by the same positional-specificity rule.
    rng = np.random.default_rng(config.seed + 303)
    boot_cosines: list[float] = []
    base_plane = eigvecs[:, pair]
    factor_pool = [
        weights.Q[l, h]
        for l in range(weights.n_layers)
        for h in range(weights.n_heads)
    ] + [
        weights.K[l, h]
        for l in range(weights.n_layers)
        for h in range(weights.n_heads)
    ] + [
        weights.V[l, h]
        for l in range(weights.n_layers)
        for h in range(weights.n_heads)
    ] + [
        weights.O[l, h]
        for l in range(weights.n_layers)
        for h in range(weights.n_heads)
    ]
    bootstrap_rank = min(32, weights.d_model)
    bootstrap_basis = torch.from_numpy(eigvecs[:, :bootstrap_rank]).double()
    projected_components: list[np.ndarray] = []
    for factor in factor_pool:
        projected = bootstrap_basis.T @ factor
        denominator = max(float(torch.sum(factor * factor)), 1.0e-300)
        projected_components.append((projected @ projected.T / denominator).numpy())
    projected_stack = np.stack(projected_components)
    for _ in range(config.bootstrap_draws):
        counts = np.bincount(
            rng.integers(0, len(factor_pool), len(factor_pool)),
            minlength=len(factor_pool),
        )
        state_small = np.einsum("n,nij->ij", counts, projected_stack)
        _, vectors_small = np.linalg.eigh(state_small)
        candidate = eigvecs[:, :bootstrap_rank] @ vectors_small[:, ::-1]
        candidate = candidate[:, : min(10, bootstrap_rank)]
        candidate_score = []
        for index in range(candidate.shape[1]):
            v = candidate[:, index]
            p = np.linalg.norm(positional @ v) ** 2 / pos_energy
            t = np.linalg.norm(token @ v) ** 2 / tok_energy
            candidate_score.append(p / max(t, 1.0e-12))
        selected = np.argsort(candidate_score)[::-1][:2]
        cosines = _principal_cosines(base_plane, candidate[:, selected])
        if len(cosines):
            boot_cosines.append(float(cosines.min()))

    for index in range(top_n):
        log_value = math.log(max(eigvals[index], 1.0e-300))
        other = np.delete(np.log(np.clip(eigvals[:top_n], 1.0e-300, None)), index)
        gap = float(np.min(np.abs(log_value - other))) if len(other) else math.inf
        p, t, ratio = shares[index]
        table_rows.append(
            {
                "model": weights.model_name,
                "row_kind": "eigendirection",
                "band": index,
                "eigenvalue": eigvals[index],
                "nearest_log_gap": gap,
                "position_multiple": p,
                "token_multiple": t,
                "position_ratio": ratio,
                "selected_posratio": index in pair,
            }
        )

    for beta in np.logspace(-2, 2, 17):
        logits = beta * eigvals
        probabilities = np.exp(logits - logits.max())
        probabilities /= probabilities.sum()
        entropy = -float(np.sum(probabilities * np.log(np.clip(probabilities, 1.0e-300, None))))
        table_rows.append(
            {
                "model": weights.model_name,
                "row_kind": "temperature",
                "beta": beta,
                "effective_rank": math.exp(entropy),
                "posratio_mass": float(probabilities[pair].sum()),
                "stable_cluster_mass": float(probabilities[stable_indices].sum()),
            }
        )

    rng_plane = _orth(rng.standard_normal((weights.d_model, 2)), 2)
    planes = {
        "posratio_plane": eigvecs[:, pair],
        "top_eigenvalue_plane": eigvecs[:, top_pair],
        "activation_pca_plane": activation_pca,
        "outlier_coordinate_plane": outlier,
        "joined_gauge_support": joined,
        "stable_spectral_cluster": eigvecs[:, stable_indices],
        "matched_random_plane": rng_plane,
    }
    arrays = {
        "pooled_state": S.numpy(),
        "eigenvalues": eigvals,
        "eigenvectors": eigvecs,
        "posratio_projector": planes["posratio_plane"] @ planes["posratio_plane"].T,
        "joined_gauge_projector": joined @ joined.T,
        "stable_cluster_projector": planes["stable_spectral_cluster"] @ planes["stable_spectral_cluster"].T,
        "summed_gauge_projector_spectrum": projector_spectrum,
    }
    cos_gauge = [
        float(_principal_cosines(gauge_planes["trained"], plane).min())
        for name, plane in gauge_planes.items()
        if name != "trained"
    ]
    observations = [
        f"The positional-specificity rule selected pooled bands {pair} in {weights.model_name}.",
        f"Across balanced and diagonal rebalancing gauges the selected planes had minimum principal cosine {min(cos_gauge) if cos_gauge else 1.0:.3f}, and their joined support had rank {joined.shape[1]}.",
        f"Bootstrap refits had median minimum principal cosine {float(np.median(boot_cosines)) if boot_cosines else math.nan:.3f} to the trained PosRatio plane.",
    ]
    return RWPCAResult(
        table=pd.DataFrame(table_rows),
        planes=planes,
        arrays=arrays,
        positional_matrix=positional,
        natural_activation_matrix=natural,
        observations=observations,
    )
