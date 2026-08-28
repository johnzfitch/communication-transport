from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

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
)


@dataclass(slots=True)
class TernaryResult:
    records: pd.DataFrame
    observations: list[str]


def _set_key(heads: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    return tuple(sorted(set(heads), key=parse_head))


def _ablation_effect(
    clean: BehavioralReadout, changed: BehavioralReadout
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """A(S) in the two requested readouts, including per-prompt values."""

    induction = clean.induction_gain - changed.induction_gain
    natural = changed.natural_loss - clean.natural_loss
    induction_prompt = (
        clean.induction_gain_by_sequence - changed.induction_gain_by_sequence
    )
    natural_prompt = (
        changed.natural_loss_by_sequence - clean.natural_loss_by_sequence
    )
    return induction, natural, induction_prompt, natural_prompt


def _synergy(values: dict[tuple[str, ...], np.ndarray | float], triple: tuple[str, str, str]):
    i, j, k = triple
    return (
        values[_set_key((i, j, k))]
        - values[_set_key((i, j))]
        - values[_set_key((i, k))]
        - values[_set_key((j, k))]
        + values[_set_key((i,))]
        + values[_set_key((j,))]
        + values[_set_key((k,))]
    )


def _triangle_set(triangles: pd.DataFrame, limit: int) -> list[tuple[str, str, str]]:
    if not len(triangles):
        return []
    valid = triangles[
        triangles.get(
            "local_error", pd.Series(index=triangles.index, dtype=object)
        ).isna()
    ]
    if "row_kind" in valid:
        # Surrogate-model triangles are not circuit paths of the real model;
        # matched outside-community controls are and give the synergy strata.
        valid = valid[
            valid["row_kind"].isin(["v_triangle", "v_triangle_matched_control"])
        ]
    if "polar_method" in valid:
        preferred = valid[valid["polar_method"] == "ridge"]
        if len(preferred):
            valid = preferred.sort_values("ridge_value").drop_duplicates(["k", "j", "i"])
    triples = []
    for row in valid.itertuples():
        triple = (str(row.k), str(row.j), str(row.i))
        if len(set(triple)) == 3 and triple not in triples:
            triples.append(triple)
        if len(triples) >= limit:
            break
    return triples


def _edge_features(edges: pd.DataFrame, triple: tuple[str, str, str]) -> dict[str, float]:
    k, j, i = triple
    lookup = {
        (str(row.writer), str(row.reader), str(row.channel)): row
        for row in edges[edges["edge_class"].str.startswith("head_head_")].itertuples()
    }
    out: dict[str, float] = {}
    for channel in ("K", "Q", "V"):
        for name, pair in (("kj", (k, j)), ("ji", (j, i)), ("ki", (k, i))):
            row = lookup.get((*pair, channel))
            out[f"C_{channel}_{name}"] = float(row.C) if row is not None else 0.0
            out[f"z_{channel}_{name}"] = (
                float(row.empirical_z)
                if row is not None and hasattr(row, "empirical_z")
                else (
                    float(row.theoretical_z)
                    if row is not None and hasattr(row, "theoretical_z")
                    else 0.0
                )
            )
    layers = [parse_head(label)[0] for label in triple]
    out["span_kj"] = float(layers[1] - layers[0])
    out["span_ji"] = float(layers[2] - layers[1])
    out["span_ki"] = float(layers[2] - layers[0])
    return out


def _numeric_triangle_features(row: pd.Series) -> dict[str, float]:
    desired = (
        "path_residual_over_direct",
        "path_residual_over_composed",
        "path_residual_symmetric",
        "radial_residual",
        "positive_endpoint_distance",
        "compact_holonomy_score",
        "dominant_oriented_angle",
        "order_reversal_difference",
        "role_complete_minus_collapsed",
        "spectral_energy",
        "skew_energy",
        "routed_attention_mass",
        "norm_matched_ternary_interference",
        "endpoint_collision",
        "nonlinear_path_feature",
    )
    return {
        key: float(row.get(key, 0.0)) if math.isfinite(float(row.get(key, 0.0))) else 0.0
        for key in desired
    }


def _standardize(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0, keepdims=True)
    scale = train.std(axis=0, keepdims=True)
    scale[scale < 1.0e-12] = 1.0
    return (train - mean) / scale, (test - mean) / scale


def _ridge_predict(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, ridge: float = 1.0e-2) -> np.ndarray:
    x = np.column_stack([np.ones(len(train_x)), train_x])
    xt = np.column_stack([np.ones(len(test_x)), test_x])
    regularizer = ridge * np.eye(x.shape[1])
    regularizer[0, 0] = 0.0
    coefficients = np.linalg.solve(x.T @ x + regularizer, x.T @ train_y)
    return xt @ coefficients


def _polynomial_features(x: np.ndarray) -> np.ndarray:
    # Degree two is deliberately explicit so the fitted object is saved and
    # inspectable; cap interactions for the small one-machine census.
    width = min(x.shape[1], 24)
    pieces = [x]
    pieces.extend((x[:, a] * x[:, b])[:, None] for a in range(width) for b in range(a, width))
    return np.column_stack(pieces)


def _boosted_stumps(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    *,
    rounds: int = 48,
    rate: float = 0.08,
) -> np.ndarray:
    base = float(train_y.mean())
    prediction = np.full(len(train_y), base)
    output = np.full(len(test_x), base)
    for _ in range(rounds):
        residual = train_y - prediction
        best: tuple[float, int, float, float, float] | None = None
        for feature in range(train_x.shape[1]):
            values = train_x[:, feature]
            for threshold in np.unique(np.quantile(values, [0.2, 0.4, 0.6, 0.8])):
                left = values <= threshold
                if not left.any() or left.all():
                    continue
                lv = float(residual[left].mean())
                rv = float(residual[~left].mean())
                error = float(np.sum((residual - np.where(left, lv, rv)) ** 2))
                candidate = (error, feature, float(threshold), lv, rv)
                if best is None or candidate[0] < best[0]:
                    best = candidate
        if best is None:
            break
        _, feature, threshold, lv, rv = best
        prediction += rate * np.where(train_x[:, feature] <= threshold, lv, rv)
        output += rate * np.where(test_x[:, feature] <= threshold, lv, rv)
    return output


def _mlp_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    seed: int,
) -> np.ndarray:
    torch.manual_seed(seed)
    x = torch.tensor(train_x, dtype=torch.float32)
    y = torch.tensor(train_y[:, None], dtype=torch.float32)
    xt = torch.tensor(test_x, dtype=torch.float32)
    width = max(8, min(32, 2 * train_x.shape[1]))
    model = torch.nn.Sequential(
        torch.nn.Linear(train_x.shape[1], width),
        torch.nn.Tanh(),
        torch.nn.Linear(width, width // 2),
        torch.nn.Tanh(),
        torch.nn.Linear(width // 2, 1),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02, weight_decay=1.0e-3)
    for _ in range(160):
        optimizer.zero_grad()
        loss = torch.mean((model(x) - y) ** 2)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        return model(xt).squeeze(1).numpy().astype(float)


def _score(y: np.ndarray, prediction: np.ndarray) -> tuple[float, float]:
    mse = float(np.mean((y - prediction) ** 2))
    denominator = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum((y - prediction) ** 2)) / denominator if denominator > 1.0e-20 else math.nan
    return r2, math.sqrt(mse)


def _split_indices(
    triples: list[tuple[str, str, str]],
    communities: pd.DataFrame,
    seed: int,
    grouped: bool,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(triples)
    if n < 2:
        return np.arange(n), np.array([], dtype=int)
    rng = np.random.default_rng(seed)
    if not grouped:
        order = rng.permutation(n)
        cut = max(1, min(n - 1, int(round(0.75 * n))))
        return order[:cut], order[cut:]
    membership = dict(zip(communities.get("head", []), communities.get("community", [])))
    groups = [
        tuple(sorted({int(membership.get(head, -1)) for head in triple}))
        for triple in triples
    ]
    unique = list(dict.fromkeys(groups))
    rng.shuffle(unique)
    test_groups = set(unique[max(1, int(round(0.75 * len(unique)))) :])
    if not test_groups:
        # A strict held-community split is impossible when every triangle has
        # the same label.  A strict disjoint-head split is attempted instead.
        test = [n - 1]
        held_heads = set(triples[test[0]])
        train = [idx for idx, triple in enumerate(triples) if held_heads.isdisjoint(triple)]
        return np.asarray(train, dtype=int), np.asarray(test, dtype=int)
    test = np.asarray([idx for idx, group in enumerate(groups) if group in test_groups], dtype=int)
    held_heads = set(itertools.chain.from_iterable(triples[idx] for idx in test))
    train = np.asarray(
        [idx for idx, triple in enumerate(triples) if idx not in set(test) and held_heads.isdisjoint(triple)],
        dtype=int,
    )
    return train, test


def _prediction_rows(
    model_name: str,
    triples: list[tuple[str, str, str]],
    features: pd.DataFrame,
    outcomes: dict[str, np.ndarray],
    communities: pd.DataFrame,
    pairwise_columns: list[str],
    seed: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    all_columns = pairwise_columns + [column for column in features.columns if column not in pairwise_columns]
    for grouped in (False, True):
        train, test = _split_indices(triples, communities, seed, grouped)
        for target, y in outcomes.items():
            for feature_set, columns in (("pairwise", pairwise_columns), ("pairwise_plus_triangle", all_columns)):
                if len(train) < 2 or len(test) < 1 or not columns:
                    rows.append(
                        {
                            "record_type": "prediction",
                            "model": model_name,
                            "target": target,
                            "split": "grouped_by_head_and_community" if grouped else "ungrouped",
                            "feature_set": feature_set,
                            "estimator": "unavailable",
                            "n_train": len(train),
                            "n_test": len(test),
                            "observation": "No leak-free held-out fit was constructible at this triangle count.",
                        }
                    )
                    continue
                x = features[columns].to_numpy(float)
                train_x, test_x = _standardize(x[train], x[test])
                fits: dict[str, np.ndarray] = {
                    "ridge": _ridge_predict(train_x, y[train], test_x),
                    "gradient_boosted_stumps": _boosted_stumps(train_x, y[train], test_x),
                    "small_mlp": _mlp_predict(train_x, y[train], test_x, seed),
                }
                train_poly = _polynomial_features(train_x)
                test_poly = _polynomial_features(test_x)
                fits["degree2_kernel_ridge"] = _ridge_predict(train_poly, y[train], test_poly, ridge=0.1)
                for estimator, prediction in fits.items():
                    r2, rmse = _score(y[test], prediction)
                    rows.append(
                        {
                            "record_type": "prediction",
                            "model": model_name,
                            "target": target,
                            "split": "grouped_by_head_and_community" if grouped else "ungrouped",
                            "feature_set": feature_set,
                            "estimator": estimator,
                            "n_train": len(train),
                            "n_test": len(test),
                            "held_out_r2": r2,
                            "held_out_rmse": rmse,
                        }
                    )
    return rows


def _synthetic_control(seed: int) -> list[dict[str, object]]:
    rng = np.random.default_rng(seed + 10_029)
    x = rng.normal(size=(256, 5))
    # The interaction is norm matched to the additive term and invisible to a
    # purely pairwise linear description by construction.
    interaction = x[:, 0] * x[:, 1] * x[:, 2]
    interaction *= np.std(x[:, 3] + x[:, 4]) / max(np.std(interaction), 1.0e-12)
    y = x[:, 3] + x[:, 4] + interaction + 0.05 * rng.normal(size=len(x))
    train, test = np.arange(192), np.arange(192, 256)
    pair_train, pair_test = _standardize(x[train], x[test])
    full = np.column_stack([x, interaction])
    full_train, full_test = _standardize(full[train], full[test])
    rows = []
    for feature_set, a, b in (
        ("pairwise", pair_train, pair_test),
        ("pairwise_plus_known_ternary", full_train, full_test),
    ):
        prediction = _ridge_predict(a, y[train], b)
        r2, rmse = _score(y[test], prediction)
        rows.append(
            {
                "record_type": "positive_control",
                "model": "synthetic_norm_matched_ternary",
                "target": "known_nonpairwise_interference",
                "split": "held_out_samples",
                "feature_set": feature_set,
                "estimator": "ridge",
                "n_train": len(train),
                "n_test": len(test),
                "held_out_r2": r2,
                "held_out_rmse": rmse,
            }
        )
    return rows


def run_ternary_synergy(
    model_name: str,
    model,
    corpora: Corpora,
    clean: BehavioralReadout,
    triangles: pd.DataFrame,
    edges: pd.DataFrame,
    communities: pd.DataFrame,
    config: RunConfig,
) -> TernaryResult:
    triples = _triangle_set(triangles, config.max_synergy_triangles_per_model)
    if not triples:
        records = pd.DataFrame(_synthetic_control(config.seed))
        return TernaryResult(
            records=records,
            observations=[
                "No causal head triple was constructible, while the synthetic nonpairwise control still ran."
            ],
        )

    means = head_z_means(model, corpora.natural, config)
    requested_sets: set[tuple[str, ...]] = set()
    for triple in triples:
        for size in (1, 2, 3):
            requested_sets.update(_set_key(subset) for subset in itertools.combinations(triple, size))

    induction: dict[tuple[str, ...], float] = {}
    natural: dict[tuple[str, ...], float] = {}
    induction_prompt: dict[tuple[str, ...], np.ndarray] = {}
    natural_prompt: dict[tuple[str, ...], np.ndarray] = {}
    local_errors: dict[tuple[str, ...], str] = {}
    for head_set in sorted(requested_sets, key=lambda item: (len(item), item)):
        try:
            hooks = mean_ablation_hooks(head_set, means)
            changed = evaluate_behavior(model, corpora, config, hooks, hooks)
            a_ind, a_nat, p_ind, p_nat = _ablation_effect(clean, changed)
            induction[head_set], natural[head_set] = a_ind, a_nat
            induction_prompt[head_set], natural_prompt[head_set] = p_ind, p_nat
        except Exception as exc:
            local_errors[head_set] = f"{type(exc).__name__}: {exc}"

    base_triangles = triangles.copy()
    if "polar_method" in base_triangles:
        base_triangles = base_triangles[base_triangles["polar_method"] == "ridge"]
        base_triangles = base_triangles.sort_values("ridge_value").drop_duplicates(["k", "j", "i"])
    triangle_lookup = {
        (str(row.k), str(row.j), str(row.i)): pd.Series(row._asdict())
        for row in base_triangles.itertuples()
    }
    triple_rows: list[dict[str, object]] = []
    feature_rows: list[dict[str, float]] = []
    valid_triples: list[tuple[str, str, str]] = []
    y_induction: list[float] = []
    y_natural: list[float] = []
    membership = dict(zip(communities.get("head", []), communities.get("community", [])))
    for triple in triples:
        needed = [_set_key(subset) for size in (1, 2, 3) for subset in itertools.combinations(triple, size)]
        errors = [local_errors[item] for item in needed if item in local_errors]
        if errors:
            triple_rows.append(
                {
                    "record_type": "triple",
                    "model": model_name,
                    "k": triple[0],
                    "j": triple[1],
                    "i": triple[2],
                    "local_error": " | ".join(errors),
                }
            )
            continue
        syn_ind = float(_synergy(induction, triple))
        syn_nat = float(_synergy(natural, triple))
        prompt_ind = np.asarray(_synergy(induction_prompt, triple))
        prompt_nat = np.asarray(_synergy(natural_prompt, triple))
        row = {
            "record_type": "triple",
            "model": model_name,
            "k": triple[0],
            "j": triple[1],
            "i": triple[2],
            "community_labels": [int(membership.get(head, -1)) for head in triple],
            "induction_synergy": syn_ind,
            "natural_loss_synergy": syn_nat,
            "induction_prompt_synergy_mean": float(prompt_ind.mean()),
            "induction_prompt_synergy_std": float(prompt_ind.std()),
            "natural_prompt_synergy_mean": float(prompt_nat.mean()),
            "natural_prompt_synergy_std": float(prompt_nat.std()),
            "triple_ablation_induction_effect": induction[_set_key(triple)],
            "triple_ablation_natural_effect": natural[_set_key(triple)],
        }
        triangle_row = triangle_lookup.get(triple, pd.Series(dtype=float))
        features = _edge_features(edges, triple)
        features.update(_numeric_triangle_features(triangle_row))
        # Individual and pair causal effects are part of the pairwise baseline.
        for index, head in enumerate(triple):
            features[f"single_ablation_{index}"] = induction[_set_key((head,))]
        for index, pair in enumerate(itertools.combinations(triple, 2)):
            features[f"pair_ablation_{index}"] = induction[_set_key(pair)]
        row.update({f"feature_{key}": value for key, value in features.items()})
        triple_rows.append(row)
        feature_rows.append(features)
        valid_triples.append(triple)
        y_induction.append(syn_ind)
        y_natural.append(syn_nat)

    prediction_rows: list[dict[str, object]] = []
    if valid_triples:
        feature_frame = pd.DataFrame(feature_rows).fillna(0.0)
        pairwise_columns = [
            column
            for column in feature_frame.columns
            if column.startswith(("C_", "z_", "span_", "single_", "pair_"))
        ]
        prediction_rows = _prediction_rows(
            model_name,
            valid_triples,
            feature_frame,
            {
                "induction_synergy": np.asarray(y_induction),
                "natural_loss_synergy": np.asarray(y_natural),
            },
            communities,
            pairwise_columns,
            config.seed,
        )
    control_rows = _synthetic_control(config.seed)
    records = pd.DataFrame(triple_rows + prediction_rows + control_rows)
    observed = records[records.get("record_type", "") == "triple"]
    prediction = records[
        (records.get("record_type", "") == "prediction")
        & records.get("held_out_r2", pd.Series(index=records.index, dtype=float)).notna()
    ]
    observations = [
        (
            f"The {len(observed)} evaluated triples had median induction inclusion-exclusion synergy {float(observed['induction_synergy'].median()):.3g}."
            if len(observed) and "induction_synergy" in observed
            else "No interpretable causal triple remained after local run errors."
        ),
        (
            f"The best constructed held-out triangle model had R-squared {float(prediction['held_out_r2'].max()):.3g}; grouped and ungrouped rows are stored separately."
            if len(prediction)
            else "The configured triple count was too small for a held-out causal-synergy model."
        ),
    ]
    return TernaryResult(records=records, observations=observations)
