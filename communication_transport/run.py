from __future__ import annotations

import argparse
import json
import math
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
import pandas as pd
import torch

from .config import ALL_EXPERIMENTS, DEFAULT_MODELS, RunConfig, slugify_model
from .graph_connection import run_graph_connection
from .hessian_typing import run_hessian_typing
from .interventions import run_interventions
from .layer_transport import run_layer_transport
from .lie_identification import run_lie_identification
from .model_io import (
    build_corpora,
    cache_activations,
    detect_induction_heads,
    evaluate_behavior,
    load_model,
    model_summary,
    unload_model,
)
from .operator_core import run_operator_experiments
from .realized_traffic import run_realized_traffic
from .report import generate_figures, render_report
from .rope_transport import run_rope_transport
from .rw_pca import run_rw_pca
from .ternary_synergy import run_ternary_synergy
from .wang_map import run_wang_map
from .weights import extract_weights


T = TypeVar("T")


TABLE_PATHS = {
    "edges": "edges.parquet",
    "communities": "communities.parquet",
    "neuron_wires": "neuron_wires.parquet",
    "interventions": "interventions.parquet",
    "layer_span_profiles": "layer_span_profiles.parquet",
    "triangles": "triangles.parquet",
    "rw_pca": "rw_pca.parquet",
    "hessian_typing": "hessian_typing.parquet",
    "lie_identification": "lie_identification.parquet",
    "rope_transport": "rope_transport.parquet",
    "ternary_synergy": "ternary_synergy.parquet",
    "realized_traffic": "realized_traffic.parquet",
    # Plain supporting tables make every plotted or summarized value traceable.
    "map_families": "map_families.parquet",
    "dense_crosschecks": "dense_crosschecks.parquet",
    "operator_core_records": "operator_core_records.parquet",
    "head_operator_census": "head_operator_census.parquet",
    "thomas_wigner": "thomas_wigner.parquet",
    "layer_transport_fits": "layer_transport_fits.parquet",
}


def _json_default(value: object):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    raise TypeError(type(value).__name__)


def _parquet_safe(frame: pd.DataFrame) -> pd.DataFrame:
    value = frame.copy()
    for column in value.columns:
        if value[column].dtype != object:
            continue
        has_complex = value[column].map(
            lambda item: isinstance(item, (list, tuple, dict, np.ndarray))
        ).any()
        if has_complex:
            value[column] = value[column].map(
                lambda item: json.dumps(item, default=_json_default, sort_keys=True)
                if isinstance(item, (list, tuple, dict, np.ndarray))
                else item
            )
    return value


def _write_table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _parquet_safe(frame).to_parquet(path, index=False)


def _concat(frames: list[pd.DataFrame]) -> pd.DataFrame:
    nonempty = [frame for frame in frames if frame is not None]
    return pd.concat(nonempty, ignore_index=True, sort=False) if nonempty else pd.DataFrame()


def _save_tables(output: Path, tables: dict[str, list[pd.DataFrame]]) -> None:
    for name, relative in TABLE_PATHS.items():
        _write_table(_concat(tables[name]), output / relative)


def _save_arrays(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def _prefix_arrays(target: dict[str, np.ndarray], prefix: str, values: dict[str, np.ndarray]) -> None:
    for name, array in values.items():
        target[f"{prefix}__{name}"] = np.asarray(array)


def _attempt(
    stage: str,
    model_name: str | None,
    errors: list[dict[str, object]],
    operation: Callable[[], T],
) -> T | None:
    label = f"{model_name} / {stage}" if model_name else stage
    print(f"Running {label}...", flush=True)
    try:
        value = operation()
        print(f"Completed {label}.", flush=True)
        return value
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        errors.append(
            {
                "model": model_name,
                "stage": stage,
                "error": message,
                "traceback": traceback.format_exc(),
            }
        )
        print(f"Observed: {label} could not be interpreted because {message}; independent computations continue.", flush=True)
        return None


def _run_exceptional_branch(config: RunConfig):
    """Load the project-local Albert engine only when this branch is requested."""

    from .exceptional_branch import run_exceptional_branch

    return run_exceptional_branch(config)


def _append_observations(
    target: dict[str, list[str]], section: str, result: object | None
) -> None:
    if result is not None and hasattr(result, "observations"):
        target[section].extend(getattr(result, "observations"))
        for sentence in getattr(result, "observations"):
            print(f"Observed: {sentence}", flush=True)


def _activation_views(model, corpora, config: RunConfig):
    names = []
    for layer in range(model.cfg.n_layers):
        names.extend(
            (
                f"blocks.{layer}.hook_resid_pre",
                f"blocks.{layer}.ln1.hook_normalized",
            )
        )
    cached = cache_activations(model, corpora.natural, names, config)
    residual = {
        layer: cached[f"blocks.{layer}.hook_resid_pre"]
        for layer in range(model.cfg.n_layers)
        if f"blocks.{layer}.hook_resid_pre" in cached
    }
    normalized = {
        layer: cached[f"blocks.{layer}.ln1.hook_normalized"].reshape(
            -1, model.cfg.d_model
        ).double()
        for layer in range(model.cfg.n_layers)
        if f"blocks.{layer}.ln1.hook_normalized" in cached
    }
    return residual, normalized


def run(config: RunConfig) -> Path:
    output = config.output.resolve()
    config.output = output
    output.mkdir(parents=True, exist_ok=True)
    (output / "operator_cores").mkdir(exist_ok=True)
    (output / "exceptional_branch").mkdir(exist_ok=True)
    config.write()
    enabled = set(config.normalized_experiments())
    tables: dict[str, list[pd.DataFrame]] = {name: [] for name in TABLE_PATHS}
    arrays_operator: dict[str, np.ndarray] = {}
    arrays_transport: dict[str, np.ndarray] = {}
    arrays_rw: dict[str, np.ndarray] = {}
    summaries: list[dict[str, object]] = []
    errors: list[dict[str, object]] = []
    observations: dict[str, list[str]] = {key: [] for key in "ABCDEFG"}

    for model_name in config.models:
        model = _attempt("load processed model", model_name, errors, lambda: load_model(model_name, config))
        if model is None:
            continue
        try:
            summary = model_summary(model_name, model)
            summaries.append(summary)
            weights = _attempt("extract folded weights", model_name, errors, lambda: extract_weights(model_name, model))
            if weights is None:
                continue
            slug = slugify_model(model_name)

            map_result = _attempt("scalar communication map", model_name, errors, lambda: run_wang_map(weights, config))
            if map_result is not None:
                tables["edges"].append(map_result.edges)
                tables["neuron_wires"].append(map_result.neuron_wires)
                tables["map_families"].append(map_result.families)
                tables["dense_crosschecks"].append(map_result.dense_crosschecks)
                _append_observations(observations, "A", map_result)
            edges = map_result.edges if map_result is not None else pd.DataFrame()

            operator = None
            need_operator = bool(
                {"operator_core", "thomas_wigner", "interventions", "hessian_typing"}
                & enabled
            )
            if map_result is not None and need_operator:
                operator = _attempt("pair operator and Thomas-Wigner run", model_name, errors, lambda: run_operator_experiments(weights, edges, config))
                if operator is not None:
                    tables["operator_core_records"].append(operator.records)
                    tables["head_operator_census"].append(operator.head_census)
                    tables["thomas_wigner"].append(operator.thomas_wigner)
                    _prefix_arrays(arrays_operator, slug, operator.arrays)
                    _append_observations(observations, "B", operator)

            activation_stages = enabled - {"scalar_map", "operator_core", "thomas_wigner", "exceptional_branch"}
            corpora = None
            if activation_stages:
                corpora = _attempt("construct activation corpora", model_name, errors, lambda: build_corpora(model, config))
            induction_heads: list[str] = []
            induction_scores: dict[str, float] = {}
            if corpora is not None:
                detected = _attempt("behavioral induction-head readout", model_name, errors, lambda: detect_induction_heads(model, corpora, config))
                if detected is not None:
                    induction_heads, induction_scores = detected
                    summary["induction_heads"] = induction_heads
                    summary["induction_head_scores"] = induction_scores

            rw = None
            need_rw = bool(
                {
                    "rw_pca",
                    "interventions",
                    "hessian_typing",
                    "lie_identification",
                    "rope_transport",
                }
                & enabled
            )
            if corpora is not None and need_rw:
                rw = _attempt("RW-PCA and gauge census", model_name, errors, lambda: run_rw_pca(model, weights, corpora, config))
                if rw is not None:
                    tables["rw_pca"].append(rw.table)
                    _prefix_arrays(arrays_rw, slug, rw.arrays)
                    for plane_name, plane in rw.planes.items():
                        arrays_rw[f"{slug}__plane__{plane_name}"] = plane
                    _append_observations(observations, "D", rw)

            activations = None
            if corpora is not None and ({"layer_transport", "graph_connection", "role_complete_qk", "rope_transport"} & enabled):
                activations = _attempt("cache residual and normalized activation views", model_name, errors, lambda: _activation_views(model, corpora, config))
            residual_activations, normalized_activations = activations if activations is not None else ({}, {})

            graph = None
            if map_result is not None and ({"graph_connection", "role_complete_qk", "ternary_synergy", "interventions"} & enabled):
                graph = _attempt(
                    "weight-side graph connection",
                    model_name,
                    errors,
                    lambda: run_graph_connection(
                        weights,
                        edges,
                        induction_heads,
                        normalized_activations or None,
                        config,
                    ),
                )
                if graph is not None:
                    tables["communities"].append(graph.communities)
                    triangle = graph.triangles.copy()
                    triangle.insert(0, "row_kind", "v_triangle")
                    role = graph.role_loops.copy()
                    if len(role):
                        role.insert(0, "row_kind", "role_complete_qk_loop")
                    tables["triangles"].append(pd.concat([triangle, role], ignore_index=True, sort=False))
                    _append_observations(observations, "C", graph)

            layer = None
            if map_result is not None and residual_activations and "layer_transport" in enabled:
                layer = _attempt(
                    "layer transport",
                    model_name,
                    errors,
                    lambda: run_layer_transport(weights, edges, residual_activations, config),
                )
                if layer is not None:
                    tables["layer_span_profiles"].append(layer.span_profiles)
                    tables["layer_transport_fits"].append(layer.fit_table)
                    _prefix_arrays(arrays_transport, slug, layer.transports)
                    _append_observations(observations, "C", layer)

            interventions = None
            if (
                corpora is not None
                and operator is not None
                and graph is not None
                and rw is not None
                and "interventions" in enabled
            ):
                interventions = _attempt(
                    "causal interventions",
                    model_name,
                    errors,
                    lambda: run_interventions(
                        model,
                        weights,
                        corpora,
                        edges,
                        operator.head_census,
                        induction_heads,
                        graph.induction_community,
                        rw.planes,
                        config,
                    ),
                )
                if interventions is not None:
                    tables["interventions"].append(interventions.records)
                    _append_observations(observations, "B", interventions)

            if interventions is not None and rw is not None and "hessian_typing" in enabled:
                hessian = _attempt(
                    "Hessian-state intervention typing",
                    model_name,
                    errors,
                    lambda: run_hessian_typing(weights, interventions.records, rw.planes, config),
                )
                if hessian is not None:
                    frame, sentences = hessian
                    tables["hessian_typing"].append(frame)
                    observations["B"].extend(sentences)
                    for sentence in sentences:
                        print(f"Observed: {sentence}", flush=True)

            lie = None
            if graph is not None and rw is not None and "lie_identification" in enabled:
                lie = _attempt(
                    "carrier-agnostic Lie identification",
                    model_name,
                    errors,
                    lambda: run_lie_identification(
                        weights,
                        graph.induction_community,
                        rw.planes["joined_gauge_support"],
                        rw.natural_activation_matrix,
                        config,
                    ),
                )
                if lie is not None:
                    table = pd.concat(
                        [lie.table, lie.controls.assign(model=model_name, row_kind="exact_control")],
                        ignore_index=True,
                        sort=False,
                    )
                    tables["lie_identification"].append(table)
                    _append_observations(observations, "E", lie)

            if graph is not None and rw is not None and "rope_transport" in enabled:
                rope = _attempt(
                    "positional and RoPE transport",
                    model_name,
                    errors,
                    lambda: run_rope_transport(
                        weights,
                        graph.role_loops,
                        rw.positional_matrix,
                        rw.planes,
                        normalized_activations or None,
                        config,
                    ),
                )
                if rope is not None:
                    frame, sentences = rope
                    tables["rope_transport"].append(frame)
                    observations["D"].extend(sentences)
                    for sentence in sentences:
                        print(f"Observed: {sentence}", flush=True)

            clean = interventions.clean if interventions is not None else None
            if corpora is not None and graph is not None and map_result is not None and "ternary_synergy" in enabled:
                if clean is None:
                    clean = _attempt("clean behavioral baseline", model_name, errors, lambda: evaluate_behavior(model, corpora, config))
                if clean is not None:
                    synergy = _attempt(
                        "ternary co-ablation synergy",
                        model_name,
                        errors,
                        lambda: run_ternary_synergy(
                            model_name,
                            model,
                            corpora,
                            clean,
                            graph.triangles,
                            edges,
                            graph.communities,
                            config,
                        ),
                    )
                    if synergy is not None:
                        tables["ternary_synergy"].append(synergy.records)
                        _append_observations(observations, "F", synergy)

            if corpora is not None and map_result is not None and "realized_traffic" in enabled:
                realized = _attempt(
                    "realized and surviving traffic",
                    model_name,
                    errors,
                    lambda: run_realized_traffic(model, weights, corpora, edges, config),
                )
                if realized is not None:
                    tables["realized_traffic"].append(realized.table)
                    _append_observations(observations, "F", realized)

            _save_tables(output, tables)
            _save_arrays(output / "operator_cores" / "selected_edges.npz", arrays_operator)
            _save_arrays(output / "layer_transports.npz", arrays_transport)
            _save_arrays(output / "rw_pca_projectors.npz", arrays_rw)
            (output / "model_summary.json").write_text(
                json.dumps(summaries, indent=2, default=_json_default), encoding="utf-8"
            )
            (output / "errors.json").write_text(
                json.dumps(errors, indent=2, default=_json_default), encoding="utf-8"
            )
        finally:
            unload_model(model)

    if "exceptional_branch" in enabled:
        exceptional = _attempt(
            "independent exceptional branch",
            None,
            errors,
            lambda: _run_exceptional_branch(config),
        )
        if exceptional is not None:
            _write_table(exceptional.table, output / "exceptional_branch" / "comparison.parquet")
            _save_arrays(output / "exceptional_branch" / "arrays.npz", exceptional.arrays)
            _append_observations(observations, "G", exceptional)
    if not (output / "exceptional_branch" / "comparison.parquet").exists():
        _write_table(pd.DataFrame(), output / "exceptional_branch" / "comparison.parquet")
        _save_arrays(output / "exceptional_branch" / "arrays.npz", {})

    _save_tables(output, tables)
    _save_arrays(output / "operator_cores" / "selected_edges.npz", arrays_operator)
    _save_arrays(output / "layer_transports.npz", arrays_transport)
    _save_arrays(output / "rw_pca_projectors.npz", arrays_rw)
    (output / "model_summary.json").write_text(
        json.dumps(summaries, indent=2, default=_json_default), encoding="utf-8"
    )
    (output / "errors.json").write_text(
        json.dumps(errors, indent=2, default=_json_default), encoding="utf-8"
    )
    (output / "observations.json").write_text(
        json.dumps(observations, indent=2), encoding="utf-8"
    )
    _attempt("render figures", None, errors, lambda: generate_figures(output))
    _attempt("render report", None, errors, lambda: render_report(output))
    # Figure/report errors are added during rendering, so persist once more.
    (output / "errors.json").write_text(
        json.dumps(errors, indent=2, default=_json_default), encoding="utf-8"
    )
    if not (output / "report.md").exists():
        (output / "report.md").write_text(
            "# Transformer communication transport experiments\n\nObserved:\n\nThe report renderer failed; inspect `errors.json` and the saved computation tables.\n",
            encoding="utf-8",
        )
    print(f"Observed: runnable artifacts and the generated report are in {output}.", flush=True)
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the transformer communication-transport experiments.")
    parser.add_argument("--profile", choices=("debug", "standard"), default="standard")
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--experiments", nargs="+", default=["all"])
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--online", action="store_true", help="Allow Hugging Face network access instead of requiring the local cache.")
    parser.add_argument("--candidate-carrier", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = args.output or Path("outputs") / f"communication_transport_{args.profile}"
    experiments = tuple(
        item.strip()
        for group in args.experiments
        for item in group.split(",")
        if item.strip()
    )
    if args.profile == "debug":
        config = RunConfig.debug(models=args.models, output=output, experiments=experiments)
    else:
        config = RunConfig(models=tuple(args.models), output=output, experiments=experiments)
    config.device = args.device
    config.seed = args.seed
    config.offline = not args.online
    config.candidate_carrier = args.candidate_carrier
    run(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
