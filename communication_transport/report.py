from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path

# Keep Matplotlib's font cache in a writable temporary directory on managed machines.
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "communication-transport-matplotlib"),
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {"K": "#3569a8", "Q": "#d4772f", "V": "#3c8d68"}


def _read(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def _finite(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)


def _scalar_figure(edges: pd.DataFrame, path: Path) -> None:
    head = (
        edges[edges["edge_class"].astype(str).str.startswith("head_head_")]
        if "edge_class" in edges
        else edges
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    for channel in ("K", "Q", "V"):
        values = _finite(head[head.get("channel", "") == channel], "theoretical_z")
        if len(values):
            ax.hist(values.clip(-8, 8), bins=45, density=True, alpha=0.46, color=COLORS[channel], label=channel)
    ax.axvline(-2, color="#444444", linewidth=0.9, linestyle="--")
    ax.axvline(2, color="#444444", linewidth=0.9, linestyle="--")
    ax.set(xlabel="theoretical z (clipped to [-8, 8])", ylabel="density", title="Head-to-head coupling census")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(frameon=False, title="channel")
    _save(fig, path)


def _operator_figure(census: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.3, 5.0), constrained_layout=True)
    if len(census):
        for model, group in census.groupby("model"):
            ax.scatter(
                group["symmetric_energy_fraction"],
                group["skew_energy_fraction"],
                s=22,
                alpha=0.7,
                label=str(model),
            )
    ax.plot([0, 1], [1, 0], color="#777777", linewidth=0.9, linestyle="--")
    ax.set(
        xlabel="symmetric energy fraction",
        ylabel="skew energy fraction",
        xlim=(-0.02, 1.02),
        ylim=(-0.02, 1.02),
        title="Residual-stream OV operator parity",
    )
    if len(census):
        ax.legend(frameon=False, fontsize=8)
    _save(fig, path)


def _transport_figure(profiles: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True, sharey=False)
    if len(profiles):
        for channel, color in COLORS.items():
            group = profiles[profiles.get("edge_class", "") == f"head_head_{channel}"]
            if not len(group):
                continue
            aggregated = group.groupby("layer_span")[["raw_median", "covariant_median"]].median().sort_index()
            axes[0].plot(aggregated.index, aggregated["raw_median"], marker="o", color=color, label=channel)
            axes[1].plot(aggregated.index, aggregated["covariant_median"], marker="o", color=color, label=channel)
    axes[0].set(title="Raw", xlabel="layer span", ylabel="median coupling C")
    axes[1].set(title="After fitted covariant transport", xlabel="layer span")
    for ax in axes:
        ax.grid(alpha=0.2)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(frameon=False)
    fig.suptitle("Layer-span coupling profiles")
    _save(fig, path)


def _rw_figure(rw: pd.DataFrame, path: Path) -> None:
    eig = rw[rw.get("row_kind", "") == "eigendirection"]
    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    if len(eig):
        for model, group in eig.groupby("model"):
            selected = group.get("selected_posratio", False).fillna(False).astype(bool)
            ax.scatter(group.loc[~selected, "band"], group.loc[~selected, "position_ratio"], s=20, alpha=0.5)
            ax.scatter(group.loc[selected, "band"], group.loc[selected, "position_ratio"], s=58, marker="*", label=str(model))
    ax.set_yscale("log")
    ax.set(xlabel="pooled-state eigendirection", ylabel="position / token energy ratio", title="RW-PCA positional specificity")
    if len(eig):
        ax.legend(frameon=False, fontsize=8)
    _save(fig, path)


def _lie_figure(lie: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    learned = lie[lie.get("row_kind", "").isin(["learned_profile", "candidate_fit"])] if len(lie) else lie
    saturated_count = 0
    if len(learned):
        for model, group in learned.groupby("model"):
            ambient = pd.to_numeric(group["ambient_dimension"], errors="coerce")
            generators = pd.to_numeric(group["generator_dimension"], errors="coerce")
            saturated = generators == ambient * (ambient - 1.0) / 2.0
            saturated_count += int(saturated.sum())
            visible = group[~saturated]
            ax.scatter(visible["generator_dimension"], visible["closure_defect"], alpha=0.55, s=25, label=str(model))
    ax.set_yscale("log")
    ax.set(xlabel="generator-span dimension", ylabel="closure defect", title="Nonsaturated carrier-agnostic Lie closure census")
    if saturated_count:
        ax.text(
            0.02,
            0.04,
            f"{saturated_count} saturated full-so(m) rows omitted from scale",
            transform=ax.transAxes,
            fontsize=8,
            color="#555555",
        )
    if len(learned):
        ax.legend(frameon=False, fontsize=8)
    _save(fig, path)


def _realized_figures(realized: pd.DataFrame, scatter_path: Path, survival_path: Path) -> None:
    edge = realized[realized.get("row_kind", "") == "edge_summary"] if len(realized) else realized
    fig, ax = plt.subplots(figsize=(6.4, 5.0), constrained_layout=True)
    if len(edge):
        for channel, color in COLORS.items():
            group = edge[edge.get("channel", "") == channel]
            ax.scatter(group["potential_C"], group["realized_absolute"], color=color, alpha=0.65, s=24, label=channel)
    ax.set(xlabel="potential coupling C", ylabel="realized absolute traffic", title="Potential versus realized communication")
    if len(edge):
        ax.legend(frameon=False)
    _save(fig, scatter_path)

    curves = realized[realized.get("row_kind", "") == "survival_curve"] if len(realized) else realized
    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    if len(curves):
        for edge_id, group in curves.groupby("edge_id"):
            group = group.sort_values("downstream_layer")
            ax.plot(group["downstream_layer"], group["whole_stream_survival_ratio"], marker="o", alpha=0.55, linewidth=1.0)
    ax.axhline(0.1, color="#555555", linestyle="--", linewidth=0.9, label="stored survival label threshold")
    ax.set(xlabel="downstream layer", ylabel="whole-stream survival ratio", title="Downstream perturbation survival")
    ax.legend(frameon=False, fontsize=8)
    _save(fig, survival_path)


def _exceptional_figure(exceptional: pd.DataFrame, path: Path) -> None:
    mass = exceptional[exceptional.get("record_type", "") == "mass_partition"] if len(exceptional) else exceptional
    fig, ax = plt.subplots(figsize=(6.8, 4.5), constrained_layout=True)
    if len(mass):
        compact = mass.drop_duplicates("component")
        ax.bar(compact["component"].astype(str), compact["mean_fraction"], color=["#5967a8", "#b06357"][: len(compact)])
    ax.set(ylim=(0, 1), xlabel="Albert tangent summand", ylabel="mean random-wedge fraction", title="Synthetic basis-free 52/273 mass")
    _save(fig, path)


def generate_figures(output: Path) -> list[Path]:
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    edges = _read(output / "edges.parquet")
    census = _read(output / "head_operator_census.parquet")
    profiles = _read(output / "layer_span_profiles.parquet")
    rw = _read(output / "rw_pca.parquet")
    lie = _read(output / "lie_identification.parquet")
    realized = _read(output / "realized_traffic.parquet")
    exceptional = _read(output / "exceptional_branch" / "comparison.parquet")
    paths = [
        figures / "scalar_head_census.png",
        figures / "operator_parity.png",
        figures / "layer_span_transport.png",
        figures / "rw_pca_positional_specificity.png",
        figures / "lie_closure.png",
        figures / "potential_vs_realized.png",
        figures / "survival_curves.png",
        figures / "exceptional_mass.png",
    ]
    _scalar_figure(edges, paths[0])
    _operator_figure(census, paths[1])
    _transport_figure(profiles, paths[2])
    _rw_figure(rw, paths[3])
    _lie_figure(lie, paths[4])
    _realized_figures(realized, paths[5], paths[6])
    _exceptional_figure(exceptional, paths[7])
    return paths


def _fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}g}" if math.isfinite(value) else "unavailable"


def _observations(output: Path) -> dict[str, str]:
    edges = _read(output / "edges.parquet")
    wires = _read(output / "neuron_wires.parquet")
    operator = _read(output / "operator_core_records.parquet")
    interventions = _read(output / "interventions.parquet")
    profiles = _read(output / "layer_span_profiles.parquet")
    triangles = _read(output / "triangles.parquet")
    rw = _read(output / "rw_pca.parquet")
    lie = _read(output / "lie_identification.parquet")
    rope = _read(output / "rope_transport.parquet")
    synergy = _read(output / "ternary_synergy.parquet")
    realized = _read(output / "realized_traffic.parquet")
    exceptional = _read(output / "exceptional_branch" / "comparison.parquet")

    head = edges[edges.get("edge_class", "").astype(str).str.startswith("head_head_")] if len(edges) else edges
    selected = int(head.get("selected", pd.Series(dtype=bool)).fillna(False).sum()) if len(head) else 0
    map_sentence = f"The saved scalar maps contain {len(head)} head edges, {selected} selected high-coupling edges, and {len(wires)} retained signed neuron wires."

    core_residual = float(_finite(operator, "core_C_residual").max()) if len(_finite(operator, "core_C_residual")) else math.nan
    sign = interventions[interventions.get("intervention", "") == "sign_flip"] if len(interventions) else interventions
    sign_change = float(_finite(sign, "fraction_gain_destroyed").median()) if len(_finite(sign, "fraction_gain_destroyed")) else math.nan
    operator_sentence = f"Retained operator cores reconstructed scalar C to maximum residual {_fmt(core_residual)}, while sign flips changed the measured induction gain by median {_fmt(sign_change)} of baseline."

    collapse = float(_finite(profiles, "collapse_fraction").median()) if len(_finite(profiles, "collapse_fraction")) else math.nan
    tri = triangles[triangles.get("row_kind", "") == "v_triangle"] if len(triangles) and "row_kind" in triangles else triangles
    path = float(_finite(tri, "path_residual_symmetric").median()) if len(_finite(tri, "path_residual_symmetric")) else math.nan
    connection_sentence = f"Fitted layer transport changed span-center variance by median collapse fraction {_fmt(collapse)}, and V triangles had median symmetric path residual {_fmt(path)}."

    eig = rw[rw.get("row_kind", "") == "eigendirection"] if len(rw) else rw
    chosen = int(eig.get("selected_posratio", pd.Series(dtype=bool)).fillna(False).sum()) if len(eig) else 0
    exact = rope[rope.get("control", "") == "exact_rope"] if len(rope) else rope
    rope_residual = float(_finite(exact, "exact_loop_residual").max()) if len(_finite(exact, "exact_loop_residual")) else math.nan
    position_sentence = f"RW-PCA marked {chosen} model-specific positional bands, and exact RoPE loops closed with maximum residual {_fmt(rope_residual)} where rotary structure was available."

    learned = lie[lie.get("row_kind", "").isin(["learned_profile", "candidate_fit"])] if len(lie) else lie
    if len(learned):
        ambient = pd.to_numeric(learned["ambient_dimension"], errors="coerce")
        generators = pd.to_numeric(learned["generator_dimension"], errors="coerce")
        saturated = generators == ambient * (ambient - 1.0) / 2.0
        nonsaturated = learned[~saturated]
    else:
        nonsaturated = learned
    closure = float(_finite(nonsaturated, "closure_defect").min()) if len(_finite(nonsaturated, "closure_defect")) else math.nan
    algebra_sentence = f"The smallest nonsaturated carrier-agnostic learned closure defect was {_fmt(closure)}; saturated full-so(m) spans, candidate rows, and full-pipeline surrogate refits remain separately labeled."

    triples = synergy[synergy.get("record_type", "") == "triple"] if len(synergy) else synergy
    syn = float(_finite(triples, "induction_synergy").median()) if len(_finite(triples, "induction_synergy")) else math.nan
    traffic = realized[realized.get("row_kind", "") == "edge_summary"] if len(realized) else realized
    correlation = float(traffic[["potential_C", "realized_absolute"]].corr().iloc[0, 1]) if len(traffic) > 2 else math.nan
    causal_sentence = f"Evaluated head triples had median induction synergy {_fmt(syn)}, and potential coupling correlated {_fmt(correlation)} with realized traffic on sampled inputs."

    masses = exceptional[exceptional.get("record_type", "") == "mass_partition"] if len(exceptional) else exceptional
    f52 = masses[masses.get("component", "") == "52"] if len(masses) else masses
    fraction = float(_finite(f52, "mean_fraction").iloc[0]) if len(_finite(f52, "mean_fraction")) else math.nan
    model_marking = exceptional[exceptional.get("record_type", "") == "model_marking"] if len(exceptional) else exceptional
    supplied = bool(len(model_marking) and (model_marking.get("candidate", "") != "none_supplied").any())
    exceptional_sentence = (
        f"The synthetic Albert control assigned mean random-wedge 52-fraction {_fmt(fraction)}, and a supplied model-derived marking was evaluated."
        if supplied
        else f"The synthetic Albert control assigned mean random-wedge 52-fraction {_fmt(fraction)}; no model-derived 26-space and marked product were supplied."
    )
    return {
        "A": map_sentence,
        "B": operator_sentence,
        "C": connection_sentence,
        "D": position_sentence,
        "E": algebra_sentence,
        "F": causal_sentence,
        "G": exceptional_sentence,
    }


def render_report(output: Path) -> Path:
    observations = _observations(output)
    errors = []
    errors_path = output / "errors.json"
    if errors_path.exists():
        errors = json.loads(errors_path.read_text(encoding="utf-8"))
    sections = [
        ("A. Scalar map", "Wang reproduction, channel census, communities, RW-PCA context, and signed neuron wires are stored in the readable tables."),
        ("B. Pair operator", "Retained cores, symmetric/skew energy, sign flips, spectrum flattening, and the Thomas-Wigner comparison are recorded without turning discrepancies into gates."),
        ("C. Weight-side connection", "Layer-span transport, V-channel path residuals, role-complete Q/K loops, and bridge sensitivity are reported separately."),
        ("D. Positional geometry", "RW-PCA eigengaps, joined gauge support, exact RoPE flat transport, and learned residuals are kept distinct."),
        ("E. Algebra identification", "Closure profiles, classical-menu fits, and refitted surrogates are carrier-agnostic; F4 appears only as one marked control when dimensions permit."),
        ("F. Causal higher-order structure", "Mean co-ablation synergy, held-out prediction attempts, realized traffic, and downstream survival share explicit edge and triple labels."),
        ("G. Exceptional branch", "Synthetic compact Albert and hostile spin controls run independently of transformer interpretation; a model claim requires a supplied marking."),
    ]
    lines = [
        "# Transformer communication transport experiments",
        "",
        "This report was generated from the saved arrays and tables in this run directory.",
        "",
    ]
    for index, (title, description) in enumerate(sections):
        key = chr(ord("A") + index)
        lines.extend([f"## {title}", "", description, "", "Observed:", "", observations[key], ""])
    lines.extend(
        [
            "## Run errors",
            "",
            (
                f"{len(errors)} stage-level exceptions were recorded in `errors.json`; independent computations continued."
                if errors
                else "No stage-level exception was recorded."
            ),
            "",
        ]
    )
    path = output / "report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
