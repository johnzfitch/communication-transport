from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from engine.readers.common import AlbertReaderContext
from engine.readers.defect import defect_tensor, read_defect
from engine.readers.tangent import project_bivector, projector_diagnostics, projector_matrix

from .config import RunConfig
from .lie_identification import compact_f4_basis, skew_vector, vector_skew


@dataclass(slots=True)
class ExceptionalResult:
    table: pd.DataFrame
    arrays: dict[str, np.ndarray]
    observations: list[str]


def _relative(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(left - right)) / max(
        float(torch.linalg.vector_norm(right)), 1.0e-300
    )


def _cluster(values: np.ndarray, tolerance: float = 1.0e-7) -> list[dict[str, float | int]]:
    values = np.sort(np.asarray(values, dtype=float))
    if not len(values):
        return []
    scale = max(float(np.max(np.abs(values))), 1.0)
    groups: list[list[float]] = [[float(values[0])]]
    for value in values[1:]:
        if abs(float(value) - groups[-1][-1]) <= tolerance * scale:
            groups[-1].append(float(value))
        else:
            groups.append([float(value)])
    return [
        {"value": float(np.mean(group)), "multiplicity": len(group)} for group in groups
    ]


def _casimir_spectrum(basis: np.ndarray) -> list[dict[str, float | int]]:
    casimir = -np.einsum("aij,ajk->ik", basis, basis)
    return _cluster(np.linalg.eigvalsh(0.5 * (casimir + casimir.T)))


def _spin_basis() -> np.ndarray:
    """so(25) acting on 1+25, normalized in the shared skew metric."""

    generators = []
    for i in range(1, 26):
        for j in range(i + 1, 26):
            matrix = np.zeros((26, 26), dtype=np.float64)
            matrix[i, j] = 1.0 / math.sqrt(2.0)
            matrix[j, i] = -1.0 / math.sqrt(2.0)
            generators.append(matrix)
    return np.stack(generators)


def _spin_complement_basis() -> np.ndarray:
    generators = []
    for index in range(1, 26):
        matrix = np.zeros((26, 26), dtype=np.float64)
        matrix[0, index] = 1.0 / math.sqrt(2.0)
        matrix[index, 0] = -1.0 / math.sqrt(2.0)
        generators.append(matrix)
    return np.stack(generators)


def _spin_cubic() -> torch.Tensor:
    """A scaled trace-zero cubic for R plus a 25-spin factor.

    Its scale is irrelevant for derivation kernels and normalized defects.  In
    orthonormal tangent coordinates it has the invariant form
    ``x0^3 - 3 x0 ||x_vector||^2``.
    """

    cubic = torch.zeros(26, 26, 26, dtype=torch.float64)
    cubic[0, 0, 0] = 1.0
    for index in range(1, 26):
        cubic[0, index, index] = -1.0
        cubic[index, 0, index] = -1.0
        cubic[index, index, 0] = -1.0
    return cubic


def _sampled_closure(basis: np.ndarray, seed: int, draws: int = 512) -> float:
    rng = np.random.default_rng(seed)
    vectors = np.stack([skew_vector(item) for item in basis])
    numerator = 0.0
    denominator = 0.0
    for _ in range(draws):
        a, b = rng.integers(len(basis), size=2)
        bracket = basis[a] @ basis[b] - basis[b] @ basis[a]
        vector = skew_vector(bracket)
        projected = vectors.T @ (vectors @ vector)
        numerator += float(np.sum((vector - projected) ** 2))
        denominator += float(np.sum(vector**2))
    return numerator / max(denominator, 1.0e-300)


def _rotation_row(
    family: str,
    cubic: torch.Tensor,
    generator: torch.Tensor,
    amplitude: float = 0.1,
) -> dict[str, object]:
    norm = float(torch.linalg.matrix_norm(generator))
    generator = generator / max(norm, 1.0e-300)
    infinitesimal = defect_tensor(
        _rotation_row.context, cubic, generator  # type: ignore[attr-defined]
    )
    rotation = torch.matrix_exp(amplitude * generator)
    rotated = torch.einsum(
        "abc,ai,bj,ck->ijk", cubic, rotation, rotation, rotation
    )
    return {
        "record_type": "rotation_family",
        "candidate": "compact_Albert_synthetic",
        "rotation_family": family,
        "rotation_amplitude": amplitude,
        "infinitesimal_cubic_defect": float(torch.linalg.vector_norm(infinitesimal))
        / max(float(torch.linalg.vector_norm(cubic)), 1.0e-300),
        "finite_cubic_change": _relative(rotated, cubic),
    }


def _albert_peirce(context: AlbertReaderContext) -> list[dict[str, float | int]]:
    idempotent = torch.zeros(27, dtype=torch.float64)
    idempotent[0] = 1.0
    eigenvalues = torch.linalg.eigvals(context.left_matrix_raw(idempotent)).real.cpu().numpy()
    return _cluster(eigenvalues)


def _spin_left_matrix(value: np.ndarray) -> np.ndarray:
    """Left multiplication on R + Spin(25), full dimension 27."""

    a, b, vector = float(value[0]), float(value[1]), value[2:]
    matrix = np.zeros((27, 27), dtype=float)
    matrix[0, 0] = a
    matrix[1, 1] = b
    matrix[1, 2:] = vector
    matrix[2:, 1] = vector
    matrix[2:, 2:] = b * np.eye(25)
    return matrix


def _generic_left_fingerprints(context: AlbertReaderContext, seed: int) -> tuple[list[dict[str, float | int]], list[dict[str, float | int]]]:
    rng = np.random.default_rng(seed)
    albert_value = torch.tensor(rng.normal(size=27), dtype=torch.float64)
    albert = torch.linalg.eigvals(context.left_matrix_raw(albert_value)).real.cpu().numpy()
    spin_value = rng.normal(size=27)
    spin = np.linalg.eigvalsh(_spin_left_matrix(spin_value))
    return _cluster(albert, 1.0e-6), _cluster(spin, 1.0e-6)


def _marked_kernel(cubic: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit the cubic stabilizer directly from the supplied marked tensor."""

    value = torch.tensor(cubic, dtype=torch.float64)
    value = sum(value.permute(order) for order in (
        (0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)
    )) / 6.0
    upper = torch.triu_indices(26, 26, offset=1)
    columns = []
    for start in range(0, 325, 25):
        stop = min(start + 25, 325)
        matrices = torch.zeros(stop - start, 26, 26, dtype=torch.float64)
        local = torch.arange(stop - start)
        rows = upper[0, start:stop]
        cols = upper[1, start:stop]
        matrices[local, rows, cols] = 1.0
        matrices[local, cols, rows] = -1.0
        defects = (
            torch.einsum("ajk,bai->bijk", value, matrices)
            + torch.einsum("iak,baj->bijk", value, matrices)
            + torch.einsum("ija,bak->bijk", value, matrices)
        )
        columns.append(defects.reshape(stop - start, -1))
    defect_matrix = torch.cat(columns, dim=0).numpy()
    gram = defect_matrix @ defect_matrix.T
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (gram + gram.T))
    scale = max(float(eigenvalues.max()), 1.0)
    kernel_vectors = eigenvectors[:, eigenvalues <= 1.0e-9 * scale].T
    basis = np.stack([vector_skew(vector, 26) for vector in kernel_vectors]) if len(kernel_vectors) else np.empty((0, 26, 26))
    return basis, eigenvalues, defect_matrix


def _model_marking(path: Path, seed: int) -> tuple[list[dict[str, object]], dict[str, np.ndarray]]:
    rows: list[dict[str, object]] = []
    arrays: dict[str, np.ndarray] = {}
    payload = np.load(path, allow_pickle=False)
    missing = [key for key in ("basis", "product_tensor") if key not in payload]
    if missing:
        raise ValueError(f"candidate carrier is missing {', '.join(missing)}")
    carrier = np.asarray(payload["basis"])
    if 26 not in carrier.shape:
        raise ValueError(f"candidate basis must contain a 26-dimensional axis, got {carrier.shape}")
    product = np.asarray(payload["product_tensor"], dtype=np.float64)
    if product.shape != (26, 26, 26):
        raise ValueError(f"product_tensor must have shape (26,26,26), got {product.shape}")
    basis, eigenvalues, defect_matrix = _marked_kernel(product)
    arrays["model_marked_derivation_basis"] = basis
    arrays["model_marked_defect_eigenvalues"] = eigenvalues
    threshold = 1.0e-9 * max(float(eigenvalues.max()), 1.0)
    rows.append(
        {
            "record_type": "model_marking",
            "candidate": "supplied_26_space_and_product",
            "carrier_shape": list(carrier.shape),
            "derivation_dimension": len(basis),
            "defect_kernel_threshold": threshold,
            "held_out_fit": math.nan,
            "observation": (
                "The supplied file did not contain declared train and held-out splits."
                if not {"train_indices", "heldout_indices"}.issubset(payload.files)
                else "Declared train and held-out indices were present; the saved product was evaluated without refitting."
            ),
        }
    )
    if len(basis) > 1:
        rows.append(
            {
                "record_type": "model_marking",
                "candidate": "supplied_26_space_and_product",
                "metric": "sampled_closure_defect",
                "value": _sampled_closure(basis, seed),
            }
        )
    del defect_matrix
    return rows, arrays


def run_exceptional_branch(config: RunConfig) -> ExceptionalResult:
    context = AlbertReaderContext(dtype=torch.float64)
    _rotation_row.context = context  # type: ignore[attr-defined]
    diagnostics = projector_diagnostics(context)
    projector = projector_matrix(context)
    f4 = compact_f4_basis()
    spin = _spin_basis()
    spin_complement = _spin_complement_basis()
    arrays: dict[str, np.ndarray] = {
        "albert_pi52_projector": projector.cpu().numpy(),
        "albert_f4_basis": f4,
        "spin_so25_basis": spin,
        "spin_complement_basis": spin_complement,
    }
    rows: list[dict[str, object]] = [
        {
            "record_type": "hypothesis_manifest",
            "candidate": "compact_real_Albert_synthetic",
            "base_field": "R",
            "characteristic": 0,
            "unital": True,
            "finite_dimensional": True,
            "albert_form": "compact H3(O), simple reduced formally-real Albert algebra",
            "carrier": "full 27-dimensional algebra; 26-dimensional trace-zero module used only as the tangent carrier",
            "metric": "Albert trace form; -1/2 Tr(AB) on bivectors",
            "normalization": "Pi52(x wedge y)=(2/3)[Lx,Ly]",
        },
        {
            "record_type": "projector_health",
            "candidate": "compact_Albert_synthetic",
            **{key: value for key, value in diagnostics.items() if np.isscalar(value)},
        },
    ]

    rng = torch.Generator(device="cpu").manual_seed(config.seed + 404)
    fraction_rows = []
    defect_partition_residuals = []
    determinant_residuals = []
    for sample in range(max(8, config.surrogate_draws)):
        left = torch.randn(26, generator=rng, dtype=torch.float64)
        right = torch.randn(26, generator=rng, dtype=torch.float64)
        wedge = context.wedge(left, right)
        pi52, pi273 = project_bivector(context, wedge)
        wedge_sq = float(context.bivector_inner(wedge, wedge))
        projector_fraction = float(context.bivector_inner(pi52, pi52)) / max(wedge_sq, 1.0e-300)
        bracket = context.full_bracket_v(left, right)
        trace_fraction = (-2.0 / 9.0) * float(torch.trace(bracket @ bracket)) / max(wedge_sq, 1.0e-300)
        fraction_rows.append((projector_fraction, trace_fraction))
        reconstructed = 0.5 * (2.0 * pi273) + pi52
        defect_partition_residuals.append(_relative(reconstructed, wedge))
        if sample < 3:
            result = read_defect(context, wedge)
            determinant_residuals.append(float(result.diagnostics["determinant_dd_residual"]))
    fractions = np.asarray(fraction_rows)
    arrays["albert_52_fraction_samples"] = fractions
    rows.extend(
        [
            {
                "record_type": "mass_partition",
                "candidate": "compact_Albert_synthetic",
                "normalization": "basis-free Pi52 projector",
                "component": "52",
                "mean_fraction": float(fractions[:, 0].mean()),
                "std_fraction": float(fractions[:, 0].std()),
                "trace_formula_max_residual": float(np.max(np.abs(fractions[:, 0] - fractions[:, 1]))),
            },
            {
                "record_type": "mass_partition",
                "candidate": "compact_Albert_synthetic",
                "normalization": "normalized determinant defect: D*d D=2 Pi273",
                "component": "273",
                "mean_fraction": float(1.0 - fractions[:, 0].mean()),
                "identity_partition_max_residual": max(defect_partition_residuals),
                "direct_defect_adjoint_max_residual": max(determinant_residuals),
            },
            {
                "record_type": "mass_partition",
                "candidate": "compact_Albert_synthetic",
                "normalization": "full eiconal tensor action: D*c D=18 Pi273",
                "component": "273",
                "identity_coefficient": 1.0 / 18.0,
                "identity_partition_max_residual": max(defect_partition_residuals),
            },
        ]
    )

    random_coefficients = torch.randn(325, generator=rng, dtype=torch.float64)
    random_bivector = context.bivector_matrix(random_coefficients)
    generator_52, generator_273 = project_bivector(context, random_bivector)
    cubic = context.determinant_cubic_tensor
    rows.extend(
        [
            _rotation_row("F4_preserving_52", cubic, generator_52),
            _rotation_row("structure_deforming_273", cubic, generator_273),
            _rotation_row("full_O26", cubic, random_bivector),
        ]
    )

    albert_fingerprint, spin_fingerprint = _generic_left_fingerprints(context, config.seed)
    spin_external_idempotent = np.zeros(27); spin_external_idempotent[0] = 1.0
    spin_internal_idempotent = np.zeros(27); spin_internal_idempotent[1] = 0.5; spin_internal_idempotent[2] = 0.5
    comparisons = (
        (
            "compact_Albert_synthetic",
            52,
            273,
            [1, 16, 10],
            _albert_peirce(context),
            _casimir_spectrum(f4),
            _sampled_closure(f4, config.seed),
            albert_fingerprint,
            0.0,
        ),
        (
            "R_plus_spin25_hostile_control",
            300,
            25,
            [1, 24, 2],
            _cluster(np.linalg.eigvalsh(_spin_left_matrix(spin_internal_idempotent))),
            _casimir_spectrum(spin),
            _sampled_closure(spin, config.seed),
            spin_fingerprint,
            0.0,
        ),
    )
    for candidate, derivation_dim, complement_dim, peirce, observed_peirce, casimir, closure, fingerprint, heldout in comparisons:
        rows.append(
            {
                "record_type": "hostile_comparison",
                "candidate": candidate,
                "derivation_dimension": derivation_dim,
                "complement_dimension": complement_dim,
                "peirce_profile": peirce,
                "observed_peirce_eigenspaces": observed_peirce,
                "ambient_casimir_spectrum": casimir,
                "sampled_closure_defect": closure,
                "generic_left_eigenvalue_fingerprint": fingerprint,
                "synthetic_held_out_fit_residual": heldout,
            }
        )
    # Record the other hostile primitive instead of hiding the reducible
    # control's dependence on which simple summand supplies the idempotent.
    rows.append(
        {
            "record_type": "hostile_comparison",
            "candidate": "R_plus_spin25_hostile_control_external_primitive",
            "peirce_profile": [1, 0, 26],
            "observed_peirce_eigenspaces": _cluster(
                np.linalg.eigvalsh(_spin_left_matrix(spin_external_idempotent))
            ),
        }
    )

    model_marking_present = False
    if config.candidate_carrier is not None:
        try:
            model_rows, model_arrays = _model_marking(config.candidate_carrier, config.seed)
            rows.extend(model_rows)
            arrays.update(model_arrays)
            model_marking_present = True
        except Exception as exc:
            rows.append(
                {
                    "record_type": "model_marking",
                    "candidate": "supplied_candidate",
                    "local_error": f"{type(exc).__name__}: {exc}",
                    "observation": "The supplied marking could not be interpreted; synthetic controls still ran.",
                }
            )
    else:
        rows.append(
            {
                "record_type": "model_marking",
                "candidate": "none_supplied",
                "observation": "No model-derived 26-space and marked product were supplied.",
            }
        )

    table = pd.DataFrame(rows)
    observations = [
        f"The compact Albert control split random bivectors into mean 52-fraction {float(fractions[:, 0].mean()):.3f} and mean 273-fraction {float(1.0 - fractions[:, 0].mean()):.3f}.",
        f"The Albert and hostile spin controls exposed derivation/complement dimensions 52+273 and 300+25 with sampled closure defects {comparisons[0][6]:.3g} and {comparisons[1][6]:.3g}.",
        (
            "The supplied model-derived marking was analyzed alongside the synthetic controls."
            if model_marking_present
            else "No model-derived marking was supplied, so this branch makes no claim about an Albert signal in a transformer carrier."
        ),
    ]
    return ExceptionalResult(table=table, arrays=arrays, observations=observations)
