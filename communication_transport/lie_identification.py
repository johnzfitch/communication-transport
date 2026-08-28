from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd
import torch

from .config import RunConfig
from .model_io import parse_head
from .weights import ModelWeights


@dataclass(slots=True)
class LieResult:
    table: pd.DataFrame
    controls: pd.DataFrame
    observations: list[str]


def skew_vector(matrix: np.ndarray | torch.Tensor) -> np.ndarray:
    value = np.asarray(matrix, dtype=np.float64)
    indices = np.triu_indices(value.shape[0], 1)
    return math.sqrt(2.0) * value[indices]


def vector_skew(vector: np.ndarray, dimension: int) -> np.ndarray:
    result = np.zeros((dimension, dimension), dtype=np.float64)
    indices = np.triu_indices(dimension, 1)
    result[indices] = vector / math.sqrt(2.0)
    result[(indices[1], indices[0])] = -result[indices]
    return result


def orthonormalize_generators(generators: list[np.ndarray], rtol: float = 1.0e-10):
    if not generators:
        return np.empty((0, 0, 0)), np.array([])
    dimension = generators[0].shape[0]
    stacked = np.stack([skew_vector(generator) for generator in generators])
    _, singular, Vh = np.linalg.svd(stacked, full_matrices=False)
    keep = singular > rtol * max(float(singular.max()), 1.0)
    basis = np.stack([vector_skew(vector, dimension) for vector in Vh[keep]]) if keep.any() else np.empty((0, dimension, dimension))
    return basis, singular


def closure_defect(basis: np.ndarray) -> tuple[float, float, np.ndarray]:
    k = len(basis)
    if k < 2:
        return math.nan, 0.0, np.empty((0, basis.shape[1] if basis.ndim == 3 else 0, basis.shape[1] if basis.ndim == 3 else 0))
    vectors = np.stack([skew_vector(item) for item in basis])
    brackets = []
    numerator = 0.0
    denominator = 0.0
    for a in range(k):
        for b in range(a + 1, k):
            bracket = basis[a] @ basis[b] - basis[b] @ basis[a]
            vector = skew_vector(bracket)
            projected = vectors.T @ (vectors @ vector)
            numerator += float(np.sum((vector - projected) ** 2))
            denominator += float(np.sum(vector**2))
            brackets.append(bracket)
    if denominator <= 1.0e-24:
        return math.nan, denominator, np.stack(brackets)
    return numerator / denominator, denominator, np.stack(brackets)


def _cluster_multiplicities(values: np.ndarray, rtol: float = 1.0e-6) -> list[int]:
    if not len(values):
        return []
    values = np.sort(values)
    groups = [[values[0]]]
    scale = max(float(np.max(np.abs(values))), 1.0)
    for value in values[1:]:
        if abs(value - groups[-1][-1]) <= rtol * scale:
            groups[-1].append(value)
        else:
            groups.append([value])
    return [len(group) for group in groups]


def invariant_profile(basis: np.ndarray) -> dict[str, object]:
    k, m, _ = basis.shape
    defect, denominator, brackets = closure_defect(basis)
    vectors = np.stack([skew_vector(item) for item in basis]) if k else np.empty((0, m * (m - 1) // 2))
    bracket_vectors = np.stack([skew_vector(item) for item in brackets]) if len(brackets) else np.empty((0, vectors.shape[1] if k else 0))
    derived_dim = int(np.linalg.matrix_rank(bracket_vectors, tol=1.0e-8)) if bracket_vectors.size else 0
    structure = np.einsum("av,pv->pa", vectors, bracket_vectors) if bracket_vectors.size else np.empty((0, k))

    # Projected adjoint matrices, one per generator.
    ad = np.zeros((k, k, k), dtype=np.float64)
    for a in range(k):
        for b in range(k):
            bracket = basis[a] @ basis[b] - basis[b] @ basis[a]
            ad[a, :, b] = vectors @ skew_vector(bracket)
    stacked_center = np.concatenate([ad[a] for a in range(k)], axis=0) if k else np.empty((0, 0))
    center_dim = k - int(np.linalg.matrix_rank(stacked_center, tol=1.0e-8)) if k else 0
    rng = np.random.default_rng(17 + k + m)
    generic_coefficients = rng.standard_normal(k)
    generic_ad = np.einsum("a,aij->ij", generic_coefficients, ad) if k else np.empty((0, 0))
    estimated_rank = k - int(np.linalg.matrix_rank(generic_ad, tol=1.0e-7)) if k else 0
    killing = np.einsum("aij,bji->ab", ad, ad) if k else np.empty((0, 0))
    killing_eigenvalues = np.linalg.eigvalsh(0.5 * (killing + killing.T)) if k else np.array([])
    casimir_ambient = -np.einsum("aij,ajk->ik", basis, basis) if k else np.zeros((m, m))
    casimir_eigenvalues = np.linalg.eigvalsh(0.5 * (casimir_ambient + casimir_ambient.T))
    generic = np.einsum("a,aij->ij", generic_coefficients, basis) if k else np.zeros((m, m))
    generic_frequencies = np.sort(np.abs(np.imag(np.linalg.eigvals(generic))))

    commutant_dim: float = math.nan
    if m <= 32 and k:
        identity = np.eye(m)
        constraint_parts = []
        # Three generic combinations normally expose the full commutant while
        # avoiding a k-fold Kronecker matrix.
        for _ in range(min(3, k)):
            coefficients = rng.standard_normal(k)
            generator = np.einsum("a,aij->ij", coefficients, basis)
            constraint_parts.append(np.kron(identity, generator) - np.kron(generator.T, identity))
        constraint = np.concatenate(constraint_parts, axis=0)
        commutant_dim = float(m * m - np.linalg.matrix_rank(constraint, tol=1.0e-7))

    return {
        "generator_dimension": k,
        "ambient_dimension": m,
        "closure_defect": defect,
        "commutator_energy": denominator,
        "near_abelian": denominator <= 1.0e-24,
        "derived_algebra_dimension": derived_dim,
        "estimated_center_dimension": center_dim,
        "commutant_dimension": commutant_dim,
        "estimated_cartan_dimension": estimated_rank,
        "killing_spectrum": killing_eigenvalues.tolist(),
        "killing_negative_count": int(np.count_nonzero(killing_eigenvalues < -1.0e-8)),
        "killing_null_count": int(np.count_nonzero(np.abs(killing_eigenvalues) <= 1.0e-8)),
        "ambient_casimir_spectrum": casimir_eigenvalues.tolist(),
        "ambient_casimir_multiplicities": _cluster_multiplicities(casimir_eigenvalues),
        "generic_frequency_multiplicities": _cluster_multiplicities(generic_frequencies),
        # Projection makes the Jacobi identity exact in W only when the
        # projected bracket itself is a Lie bracket; compute that residual.
        "projected_jacobi_residual": _projected_jacobi_residual(ad),
    }


def _projected_jacobi_residual(ad: np.ndarray) -> float:
    k = len(ad)
    if k < 3:
        return 0.0
    numerator = 0.0
    denominator = 0.0
    for a, b, c in itertools.combinations(range(k), 3):
        # ad[a,:,b] are coefficients of [Ea,Eb].
        jacobi = (
            ad[:, :, c].T @ ad[a, :, b]
            + ad[:, :, a].T @ ad[b, :, c]
            + ad[:, :, b].T @ ad[c, :, a]
        )
        numerator += float(np.sum(jacobi**2))
        denominator += float(
            np.sum(ad[a, :, b] ** 2)
            + np.sum(ad[b, :, c] ** 2)
            + np.sum(ad[c, :, a] ** 2)
        )
    return math.sqrt(numerator / max(denominator, 1.0e-300))


def so_basis(dimension: int, ambient: int | None = None, offset: int = 0) -> np.ndarray:
    ambient = ambient or dimension
    generators = []
    for i in range(dimension):
        for j in range(i + 1, dimension):
            matrix = np.zeros((ambient, ambient))
            matrix[offset + i, offset + j] = 1.0 / math.sqrt(2.0)
            matrix[offset + j, offset + i] = -1.0 / math.sqrt(2.0)
            generators.append(matrix)
    return np.stack(generators) if generators else np.empty((0, ambient, ambient))


def block_so_basis(blocks: list[int]) -> np.ndarray:
    ambient = sum(blocks)
    parts = []
    offset = 0
    for block in blocks:
        parts.extend(so_basis(block, ambient, offset))
        offset += block
    return np.stack(parts)


def u_basis(n: int, *, special: bool = False, ambient: int | None = None) -> np.ndarray:
    base_dim = 2 * n
    ambient = ambient or base_dim
    generators = []
    # Real skew part A in diag(A,A).
    for i in range(n):
        for j in range(i + 1, n):
            A = np.zeros((n, n))
            A[i, j], A[j, i] = 1.0, -1.0
            matrix = np.block([[A, np.zeros_like(A)], [np.zeros_like(A), A]])
            generators.append(matrix)
    # Imaginary symmetric part iB -> [[0,-B],[B,0]].
    symmetric = []
    for i in range(n):
        B = np.zeros((n, n)); B[i, i] = 1.0
        symmetric.append(B)
    for i in range(n):
        for j in range(i + 1, n):
            B = np.zeros((n, n)); B[i, j] = B[j, i] = 1.0 / math.sqrt(2.0)
            symmetric.append(B)
    if special:
        # Remove the scalar imaginary identity direction.
        diagonal = np.stack([item for item in symmetric[:n]])
        coefficient_basis = np.linalg.svd(np.eye(n) - np.ones((n, n)) / n)[0][:, : n - 1]
        symmetric = [np.einsum("i,ijk->jk", coefficients, diagonal) for coefficients in coefficient_basis.T] + symmetric[n:]
    for B in symmetric:
        matrix = np.block([[np.zeros_like(B), -B], [B, np.zeros_like(B)]])
        generators.append(matrix)
    padded = []
    for matrix in generators:
        full = np.zeros((ambient, ambient)); full[:base_dim, :base_dim] = matrix
        padded.append(full)
    return orthonormalize_generators(padded)[0]


def _left_quaternion(q: np.ndarray) -> np.ndarray:
    a, b, c, d = q
    return np.array(
        [[a, -b, -c, -d], [b, a, -d, c], [c, d, a, -b], [d, -c, b, a]],
        dtype=np.float64,
    )


def sp_basis(n: int, ambient: int | None = None) -> np.ndarray:
    base_dim = 4 * n
    ambient = ambient or base_dim
    generators = []
    for i in range(n):
        for imaginary in (np.array([0, 1, 0, 0]), np.array([0, 0, 1, 0]), np.array([0, 0, 0, 1])):
            matrix = np.zeros((base_dim, base_dim))
            matrix[4 * i : 4 * i + 4, 4 * i : 4 * i + 4] = _left_quaternion(imaginary)
            generators.append(matrix)
    quaternion_basis = np.eye(4)
    for i in range(n):
        for j in range(i + 1, n):
            for q in quaternion_basis:
                block = _left_quaternion(q)
                matrix = np.zeros((base_dim, base_dim))
                matrix[4 * i : 4 * i + 4, 4 * j : 4 * j + 4] = block
                matrix[4 * j : 4 * j + 4, 4 * i : 4 * i + 4] = -block.T
                generators.append(matrix)
    padded = []
    for matrix in generators:
        full = np.zeros((ambient, ambient)); full[:base_dim, :base_dim] = matrix
        padded.append(full)
    return orthonormalize_generators(padded)[0]


@lru_cache(maxsize=1)
def compact_f4_basis() -> np.ndarray:
    from engine.readers.common import AlbertReaderContext
    from engine.readers.tangent import projector_matrix

    context = AlbertReaderContext(dtype=torch.float64)
    projector = projector_matrix(context).detach().cpu().numpy()
    values, vectors = np.linalg.eigh(projector)
    basis_vectors = vectors[:, values > 0.5].T
    return np.stack([vector_skew(vector, 26) for vector in basis_vectors])


def candidate_menu(ambient: int, target_dimension: int) -> dict[str, np.ndarray]:
    candidates: dict[str, np.ndarray] = {}
    for n in range(2, ambient + 1):
        if n * (n - 1) // 2 == target_dimension:
            candidates[f"so({n})_plus_trivial_{ambient-n}"] = so_basis(n, ambient)
    for n in range(2, ambient // 2 + 1):
        if n * n == target_dimension:
            candidates[f"u({n})_realified"] = u_basis(n, ambient=ambient)
        if n * n - 1 == target_dimension:
            candidates[f"su({n})_realified"] = u_basis(n, special=True, ambient=ambient)
    for n in range(1, ambient // 4 + 1):
        if n * (2 * n + 1) == target_dimension:
            candidates[f"sp({n})_realified"] = sp_basis(n, ambient)
    # Orthogonal block products with at most three blocks.
    for parts in range(2, 4):
        for cuts in itertools.combinations(range(1, ambient), parts - 1):
            blocks = np.diff((0, *cuts, ambient)).tolist()
            dim = sum(block * (block - 1) // 2 for block in blocks)
            if dim == target_dimension:
                name = " + ".join(f"so({block})" for block in blocks if block > 1)
                candidates[name] = block_so_basis(blocks)
                if len(candidates) >= 12:
                    break
        if len(candidates) >= 12:
            break
    if ambient == 26 and target_dimension == 52:
        candidates["F4_on_real_26_marked_control"] = compact_f4_basis()
    return candidates


def projector_chordal(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) != len(right) or not len(left):
        return math.nan
    L = np.stack([skew_vector(item) for item in left])
    R = np.stack([skew_vector(item) for item in right])
    overlap = float(np.sum((L @ R.T) ** 2))
    return max(2.0 - 2.0 * overlap / len(left), 0.0)


def fit_candidate_conjugation(
    learned: np.ndarray,
    candidate: np.ndarray,
    seed: int,
    n_initializations: int = 3,
    steps: int = 35,
) -> tuple[float, list[float]]:
    if len(learned) != len(candidate):
        return math.nan, []
    m = learned.shape[1]
    device = "cuda" if torch.cuda.is_available() and m <= 32 else "cpu"
    learned_t = torch.tensor(learned, dtype=torch.float64, device=device)
    candidate_t = torch.tensor(candidate, dtype=torch.float64, device=device)
    indices = torch.triu_indices(m, m, offset=1, device=device)
    rng = np.random.default_rng(seed)
    results = []
    for init in range(n_initializations):
        parameter = torch.zeros(m, m, dtype=torch.float64, device=device, requires_grad=True)
        if init:
            with torch.no_grad():
                parameter.copy_(torch.tensor(rng.normal(scale=0.05, size=(m, m)), dtype=torch.float64, device=device))
        optimizer = torch.optim.Adam([parameter], lr=0.08)
        for _ in range(steps):
            optimizer.zero_grad()
            skew = 0.5 * (parameter - parameter.T)
            Q = torch.matrix_exp(skew)
            moved = Q.unsqueeze(0) @ candidate_t @ Q.T.unsqueeze(0)
            Lvec = math.sqrt(2.0) * learned_t[:, indices[0], indices[1]]
            Mvec = math.sqrt(2.0) * moved[:, indices[0], indices[1]]
            overlap = torch.sum((Lvec @ Mvec.T) ** 2)
            loss = 2.0 - 2.0 * overlap / len(learned)
            loss.backward()
            optimizer.step()
        results.append(max(float(loss.detach().cpu()), 0.0))
    return min(results), results


def _ambient_basis(
    weights: ModelWeights,
    head_labels: list[str],
    extra_subspaces: list[np.ndarray],
    dimension: int,
) -> np.ndarray:
    columns = []
    for label in head_labels:
        layer, head = parse_head(label)
        columns.extend((weights.O[layer, head].numpy(), weights.V[layer, head].numpy()))
    columns.extend(value for value in extra_subspaces if value.size)
    if not columns:
        return np.eye(weights.d_model, min(dimension, weights.d_model))
    source = np.concatenate(columns, axis=1)
    U, _, _ = np.linalg.svd(source, full_matrices=False)
    return U[:, : min(dimension, U.shape[1])]


def _projected_head_generators(
    weights: ModelWeights,
    labels: list[str],
    ambient_basis: np.ndarray,
) -> list[np.ndarray]:
    U = torch.from_numpy(ambient_basis).double()
    generators = []
    for label in labels:
        layer, head = parse_head(label)
        projected = (U.T @ weights.O[layer, head]) @ (weights.V[layer, head].T @ U)
        skew = 0.5 * (projected - projected.T)
        if float(torch.linalg.matrix_norm(skew)) > 1.0e-12:
            generators.append(skew.numpy())
    return generators


def _choose_generator_dimensions(singular: np.ndarray, config: RunConfig) -> list[int]:
    if not len(singular):
        return []
    numerical = int(np.count_nonzero(singular > 1.0e-8 * max(singular.max(), 1.0)))
    choices = {numerical}
    choices.update(value for value in config.lie_generator_dims if value <= numerical)
    return sorted(value for value in choices if value >= 2)


def _learned_basis(generators: list[np.ndarray], dimension: int) -> tuple[np.ndarray, np.ndarray]:
    full, singular = orthonormalize_generators(generators)
    return full[:dimension], singular


def _layer_matched_random(
    weights: ModelWeights, labels: list[str], rng: np.random.Generator
) -> list[str]:
    counts: dict[int, int] = {}
    excluded = set(labels)
    for label in labels:
        layer, _ = parse_head(label)
        counts[layer] = counts.get(layer, 0) + 1
    result = []
    for layer, count in counts.items():
        pool = [weights.head_label(layer, h) for h in range(weights.n_heads) if weights.head_label(layer, h) not in excluded]
        if len(pool) < count:
            pool = [weights.head_label(layer, h) for h in range(weights.n_heads)]
        result.extend(rng.choice(pool, count, replace=False).tolist())
    return result


def exact_controls(config: RunConfig) -> pd.DataFrame:
    rows = []
    su5_small = u_basis(5, special=True, ambient=10)
    su5_offset = np.zeros((len(su5_small), 18, 18))
    su5_offset[:, 8:18, 8:18] = su5_small
    controls = {
        "so(9)+so(6)+so(2) dimension-52 impostor": block_so_basis([9, 6, 2]),
        "so(8)+su(5) dimension-52 impostor": np.concatenate(
            [so_basis(8, 18), su5_offset]
        ),
        "synthetic_compact_F4_on_26": compact_f4_basis(),
    }
    rng = np.random.default_rng(config.seed + 901)
    for name, basis in controls.items():
        basis = orthonormalize_generators(list(basis))[0]
        profile = invariant_profile(basis)
        rows.append({"control": name, **profile})
        corrupted = basis + rng.normal(scale=1.0e-3, size=basis.shape)
        corrupted = np.stack([0.5 * (item - item.T) for item in corrupted])
        corrupted = orthonormalize_generators(list(corrupted))[0]
        rows.append({"control": f"{name}_corrupted_1e-3", **invariant_profile(corrupted)})
    random_vectors = rng.standard_normal((10, 16 * 15 // 2))
    random_vectors = np.linalg.qr(random_vectors.T)[0].T
    random_plane = np.stack([vector_skew(vector, 16) for vector in random_vectors])
    rows.append({"control": "random_10_plane_in_so16", **invariant_profile(random_plane)})
    return pd.DataFrame(rows)


def run_lie_identification(
    weights: ModelWeights,
    induction_community: list[str],
    joined_positional_support: np.ndarray,
    activation_matrix: np.ndarray,
    config: RunConfig,
) -> LieResult:
    labels = induction_community or [
        weights.head_label(layer, head)
        for layer in range(weights.n_layers)
        for head in range(weights.n_heads)
    ][: min(weights.n_heads, 8)]
    activation_centered = activation_matrix - activation_matrix.mean(axis=0, keepdims=True)
    activation_device = "cuda" if torch.cuda.is_available() else "cpu"
    activation_tensor = torch.from_numpy(activation_centered).float().to(activation_device)
    requested_rank = min(26, activation_tensor.shape[0], activation_tensor.shape[1])
    torch.manual_seed(config.seed + 905)
    _, _, activation_vectors = torch.pca_lowrank(
        activation_tensor,
        q=requested_rank,
        center=False,
        niter=4,
    )
    activation_subspace = activation_vectors.double().cpu().numpy()
    del activation_tensor, activation_vectors
    extras = [joined_positional_support, activation_subspace]
    rows: list[dict[str, object]] = []
    learned_cache: dict[tuple[int, int], np.ndarray] = {}
    for ambient_requested in config.lie_ambient_dims:
        ambient_basis = _ambient_basis(weights, labels, extras, ambient_requested)
        m = ambient_basis.shape[1]
        raw_generators = _projected_head_generators(weights, labels, ambient_basis)
        full_basis, singular = orthonormalize_generators(raw_generators)
        for k in _choose_generator_dimensions(singular, config):
            basis = full_basis[:k]
            learned_cache[(m, k)] = basis
            profile = invariant_profile(basis)
            menu = candidate_menu(m, k)
            if not menu:
                rows.append(
                    {
                        "model": weights.model_name,
                        "row_kind": "learned_profile",
                        "ambient_source": "community_OV_plus_activation_plus_joined_RW",
                        "candidate": None,
                        **profile,
                        "generator_singular_values": singular.tolist(),
                    }
                )
            for candidate_name, candidate in menu.items():
                best, distribution = fit_candidate_conjugation(
                    basis,
                    candidate,
                    config.seed + m * 100 + k,
                    n_initializations=2 if m > 26 else 3,
                    steps=15 if m > 26 else 30,
                )
                rows.append(
                    {
                        "model": weights.model_name,
                        "row_kind": "candidate_fit",
                        "ambient_source": "community_OV_plus_activation_plus_joined_RW",
                        "candidate": candidate_name,
                        "candidate_chordal_distance": best,
                        "candidate_fit_distribution": distribution,
                        **profile,
                        "generator_singular_values": singular.tolist(),
                    }
                )

    # Full-pipeline layer-matched surrogates: refit ambient basis, generator
    # SVD, data-driven k, closure, and standard candidate distances each draw.
    rng = np.random.default_rng(config.seed + 902)
    reference_pairs = sorted(learned_cache)
    if reference_pairs:
        ref_m, ref_k = min(
            reference_pairs,
            key=lambda pair: abs(pair[0] - min(26, weights.d_model)) + abs(pair[1] - min(10, len(labels))),
        )
        reference = learned_cache[(ref_m, ref_k)]
        reference_defect = closure_defect(reference)[0]
        surrogate_defects = []
        surrogate_candidate: dict[str, list[float]] = {
            name: [] for name in candidate_menu(ref_m, ref_k)
        }
        for draw in range(config.surrogate_draws):
            surrogate_labels = _layer_matched_random(weights, labels, rng)
            ambient = _ambient_basis(weights, surrogate_labels, extras, ref_m)
            raw = _projected_head_generators(weights, surrogate_labels, ambient)
            full, singular = orthonormalize_generators(raw)
            numerical = int(np.count_nonzero(singular > 1.0e-8 * max(float(singular.max()) if len(singular) else 0.0, 1.0)))
            chosen_k = min(ref_k, numerical)
            if chosen_k < 2:
                continue
            surrogate = full[:chosen_k]
            defect = closure_defect(surrogate)[0]
            surrogate_defects.append(defect)
            if chosen_k == ref_k:
                for name, candidate in candidate_menu(ref_m, ref_k).items():
                    surrogate_candidate[name].append(projector_chordal(surrogate, candidate))
            rows.append(
                {
                    "model": weights.model_name,
                    "row_kind": "surrogate_refit",
                    "surrogate_kind": "layer_count_matched_random_community",
                    "surrogate_draw": draw,
                    "ambient_dimension": ref_m,
                    "generator_dimension": chosen_k,
                    "closure_defect": defect,
                    "reference_closure_defect": reference_defect,
                }
            )
        if surrogate_defects:
            center = float(np.median(surrogate_defects))
            scale = max(1.4826 * float(np.median(np.abs(np.asarray(surrogate_defects) - center))), 1.0e-12)
            rows.append(
                {
                    "model": weights.model_name,
                    "row_kind": "surrogate_summary",
                    "ambient_dimension": ref_m,
                    "generator_dimension": ref_k,
                    "closure_defect": reference_defect,
                    "surrogate_median": center,
                    "surrogate_robust_scale": scale,
                    "surrogate_separation": (reference_defect - center) / scale,
                    "surrogate_exceedance_count_lower": int(np.count_nonzero(np.asarray(surrogate_defects) <= reference_defect)),
                    "surrogate_draws": len(surrogate_defects),
                }
            )

    table = pd.DataFrame(rows)
    if len(table):
        ambient = pd.to_numeric(table.get("ambient_dimension"), errors="coerce")
        generators = pd.to_numeric(table.get("generator_dimension"), errors="coerce")
        table["saturated_full_so"] = generators == ambient * (ambient - 1.0) / 2.0
    controls = exact_controls(config)
    learned_profiles = table[table["row_kind"].isin(["learned_profile", "candidate_fit"])] if len(table) else table
    nonsaturated = learned_profiles[~learned_profiles["saturated_full_so"].fillna(False)] if len(learned_profiles) else learned_profiles
    best_fit = nonsaturated[nonsaturated["row_kind"] == "candidate_fit"].sort_values("candidate_chordal_distance").head(1) if len(nonsaturated) else nonsaturated
    finite_closure = nonsaturated["closure_defect"].dropna() if len(nonsaturated) else pd.Series(dtype=float)
    saturated_closure = learned_profiles.loc[
        learned_profiles["saturated_full_so"].fillna(False), "closure_defect"
    ].dropna() if len(learned_profiles) else pd.Series(dtype=float)
    if len(best_fit):
        conclusion = f"The closest listed candidate was {best_fit['candidate'].iloc[0]} at normalized chordal distance {float(best_fit['candidate_chordal_distance'].iloc[0]):.3f}; this is an identification comparison, not an exceptional claim."
    elif len(finite_closure):
        conclusion = "The learned generator planes had no dimension-compatible candidate in the declared classical menu."
    else:
        conclusion = "The community generators were abelian, rank-deficient, or otherwise unavailable for Lie identification."
    observations = [
        f"Across nonsaturated generator spans, the smallest measured closure defect was {float(finite_closure.min()) if len(finite_closure) else math.nan:.6g}; saturated full-so(m) rows are labeled separately.",
        (
            f"A saturated full-so(m) span closed at defect {float(saturated_closure.min()):.3g}, as it must."
            if len(saturated_closure)
            else "No learned span saturated its entire ambient orthogonal algebra."
        ),
        conclusion,
        "Exact dimension-52 classical impostors and the marked compact F4 control all closed numerically; dimension and closure alone therefore did not identify F4.",
    ]
    return LieResult(table=table, controls=controls, observations=observations)
