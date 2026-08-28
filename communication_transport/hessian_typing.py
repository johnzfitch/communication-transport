from __future__ import annotations

import math

import numpy as np
import pandas as pd
import torch

from .config import RunConfig
from .interventions import flatten_ov_factors
from .model_io import parse_head
from .weights import ModelWeights


def _state_in_union_support(
    factors: list[torch.Tensor], ridge: float
) -> tuple[torch.Tensor, torch.Tensor]:
    source = torch.cat(factors, dim=1)
    U, singular, _ = torch.linalg.svd(source, full_matrices=False)
    keep = singular > 1.0e-10 * max(float(singular.max()), 1.0)
    basis = U[:, keep]
    return basis, torch.eye(int(keep.sum()), dtype=torch.float64) * ridge


def _centered_log(state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    state = 0.5 * (state + state.T)
    state = state / torch.trace(state)
    values, vectors = torch.linalg.eigh(state)
    values = values.clamp_min(1.0e-300)
    log_values = torch.log(values)
    centered = log_values - log_values.mean()
    return (vectors * centered) @ vectors.T, values, vectors


def _logarithmic_mean(a: float, b: float) -> float:
    if abs(a - b) <= 1.0e-12 * max(a, b, 1.0):
        return 0.5 * (a + b)
    return (a - b) / (math.log(a) - math.log(b))


def gibbs_displacement(
    original: torch.Tensor,
    changed: torch.Tensor,
    ridge: float,
) -> dict[str, float]:
    d = original.shape[0]
    eye = torch.eye(d, dtype=torch.float64)
    q = original + ridge * eye
    qp = changed + ridge * eye
    x, weights, frame = _centered_log(q)
    xp, _, _ = _centered_log(qp)
    h = xp - x
    h_frame = frame.T @ h @ frame
    diagonal = torch.diag(h_frame)
    weighted_mean = float(torch.sum(weights * diagonal))
    spec = float(torch.sum(weights * diagonal * diagonal)) - weighted_mean**2
    frame_energy = 0.0
    for i in range(d):
        for j in range(i + 1, d):
            frame_energy += _logarithmic_mean(float(weights[i]), float(weights[j])) * float(h_frame[i, j] ** 2 + h_frame[j, i] ** 2)
    return {
        "gibbs_total_energy": spec + frame_energy,
        "gibbs_spectral_energy": max(spec, 0.0),
        "gibbs_frame_energy": max(frame_energy, 0.0),
        "log_displacement_norm": float(torch.linalg.matrix_norm(h)),
    }


def run_hessian_typing(
    weights: ModelWeights,
    interventions: pd.DataFrame,
    planes: dict[str, np.ndarray],
    config: RunConfig,
) -> tuple[pd.DataFrame, list[str]]:
    rows: list[dict[str, object]] = []
    targets = sorted(
        set(
            interventions.loc[
                interventions["intervention"].isin(["sign_flip", "spectrum_flattening"]),
                "target",
            ]
        )
    )
    for label in targets:
        layer, head = parse_head(label)
        O = weights.O[layer, head]
        V = weights.V[layer, head]
        H = O @ (V.T @ V) @ O.T
        O_flat, V_flat, _ = flatten_ov_factors(O, V, config.support_rtol)
        H_flat = O_flat @ (V_flat.T @ V_flat) @ O_flat.T
        for intervention, changed_factor in (
            ("sign_flip", -O),
            ("spectrum_flattening", O_flat),
        ):
            H_changed = (
                changed_factor @ (V.T @ V) @ changed_factor.T
                if intervention == "sign_flip"
                else H_flat
            )
            basis, _ = _state_in_union_support([O, changed_factor], config.ridge_grid[0])
            h0 = basis.T @ H @ basis
            h1 = basis.T @ H_changed @ basis
            for ridge in config.ridge_grid:
                metric = gibbs_displacement(h0, h1, ridge)
                rows.append(
                    {
                        "model": weights.model_name,
                        "intervention": intervention,
                        "target": label,
                        "ridge": ridge,
                        "support_dimension": len(h0),
                        "support_changed": False,
                        **metric,
                    }
                )
        for plane_name, plane_np in planes.items():
            plane = torch.as_tensor(plane_np, dtype=torch.float64)
            projector = torch.eye(weights.d_model, dtype=torch.float64) - plane @ plane.T
            O_deleted = projector @ O
            H_deleted = O_deleted @ (V.T @ V) @ O_deleted.T
            basis, _ = _state_in_union_support([O, O_deleted], config.ridge_grid[0])
            h0 = basis.T @ H @ basis
            h1 = basis.T @ H_deleted @ basis
            for ridge in config.ridge_grid:
                rows.append(
                    {
                        "model": weights.model_name,
                        "intervention": "plane_deletion",
                        "target": f"{label}:{plane_name}",
                        "ridge": ridge,
                        "support_dimension": len(h0),
                        "support_changed": int(torch.linalg.matrix_rank(O_deleted))
                        != int(torch.linalg.matrix_rank(O)),
                        **gibbs_displacement(h0, h1, ridge),
                    }
                )
    frame = pd.DataFrame(rows)
    base_ridge = config.ridge_grid[0]
    sign = frame[(frame["ridge"] == base_ridge) & (frame["intervention"] == "sign_flip")]
    flat = frame[(frame["ridge"] == base_ridge) & (frame["intervention"] == "spectrum_flattening")]
    observations = [
        (
            f"The Gibbs state metric assigned sign flips at most {float(sign['gibbs_total_energy'].max()):.3e} energy, exposing its structural blindness to orientation."
            if len(sign)
            else "No sign-flip Gibbs state was available."
        ),
        (
            f"Spectrum flattening carried median spectral energy {float(flat['gibbs_spectral_energy'].median()):.3e} and median frame energy {float(flat['gibbs_frame_energy'].median()):.3e}."
            if len(flat)
            else "No spectrum-flattening Gibbs state was available."
        ),
    ]
    return frame, observations
