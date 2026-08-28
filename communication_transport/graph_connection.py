from __future__ import annotations

import dataclasses
import itertools
import math
from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd
import torch

from .config import RunConfig
from .model_io import parse_head
from .operator_core import partial_polar, psd_log
from .weights import ModelWeights


@dataclass(slots=True)
class GraphResult:
    communities: pd.DataFrame
    triangles: pd.DataFrame
    role_loops: pd.DataFrame
    induction_community: list[str]
    observations: list[str]


def build_communities(
    weights: ModelWeights,
    edges: pd.DataFrame,
    induction_heads: list[str],
    seed: int,
) -> tuple[pd.DataFrame, list[str], float]:
    selected = edges[
        edges["edge_class"].str.startswith("head_head_") & edges["selected"]
    ]
    graph = nx.Graph()
    graph.add_nodes_from(
        weights.head_label(layer, head)
        for layer in range(weights.n_layers)
        for head in range(weights.n_heads)
    )
    for edge in selected.itertuples():
        if graph.has_edge(edge.writer, edge.reader):
            graph[edge.writer][edge.reader]["weight"] += 1.0
            graph[edge.writer][edge.reader]["coupling"] += float(edge.C)
        else:
            graph.add_edge(
                edge.writer,
                edge.reader,
                weight=1.0,
                coupling=float(edge.C),
            )
    nontrivial = [component for component in nx.connected_components(graph) if len(component) > 1]
    if not nontrivial:
        rows = [
            {
                "model": weights.model_name,
                "head": node,
                "community": -1,
                "community_size": 1,
                "is_induction_head": node in induction_heads,
                "is_induction_community": False,
            }
            for node in graph.nodes
        ]
        return pd.DataFrame(rows), [], 0.0
    giant_nodes = max(nontrivial, key=len)
    giant = graph.subgraph(giant_nodes).copy()
    communities = sorted(
        nx.community.louvain_communities(giant, weight="weight", seed=seed),
        key=len,
        reverse=True,
    )
    modularity = float(nx.community.modularity(giant, communities, weight="weight"))
    overlaps = [len(set(induction_heads) & community) for community in communities]
    induction_index = int(np.argmax(overlaps)) if communities else -1
    induction_community = sorted(communities[induction_index]) if induction_index >= 0 else []
    membership: dict[str, int] = {}
    for index, community in enumerate(communities):
        for head in community:
            membership[head] = index
    rows = []
    for node in graph.nodes:
        index = membership.get(node, -1)
        size = len(communities[index]) if index >= 0 else 1
        rows.append(
            {
                "model": weights.model_name,
                "head": node,
                "community": index,
                "community_size": size,
                "is_induction_head": node in induction_heads,
                "is_induction_community": index == induction_index,
                "degree": int(graph.degree(node)),
                "weighted_degree": float(graph.degree(node, weight="weight")),
                "modularity": modularity,
            }
        )
    return pd.DataFrame(rows), induction_community, modularity


def _v_edge_factors(
    weights: ModelWeights, writer: str, reader: str
) -> tuple[torch.Tensor, torch.Tensor]:
    wl, wh = parse_head(writer)
    rl, rh = parse_head(reader)
    left = weights.O[rl, rh] @ (weights.V[rl, rh].T @ weights.O[wl, wh])
    right = weights.V[wl, wh]
    return left, right


def _project_low_rank(
    basis: torch.Tensor, factors: tuple[torch.Tensor, torch.Tensor]
) -> torch.Tensor:
    left, right = factors
    return (basis.T @ left) @ (right.T @ basis)


def _common_support(
    factor_sets: list[tuple[torch.Tensor, torch.Tensor]], rtol: float
) -> torch.Tensor:
    source = torch.cat([item for pair in factor_sets for item in pair], dim=1)
    U, singular, _ = torch.linalg.svd(source, full_matrices=False)
    keep = singular > rtol * max(float(singular.max()), 1.0)
    return U[:, keep]


def _orthogonal_polar(matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    U, singular, Vh = torch.linalg.svd(matrix, full_matrices=False)
    return U @ Vh, (Vh.T * singular) @ Vh


def _angle_from_orthogonal(matrix: np.ndarray) -> tuple[float, float, bool]:
    eigenvalues = np.linalg.eigvals(matrix)
    near_cut = bool(np.any(np.abs(eigenvalues + 1.0) < 1.0e-6))
    angles = np.angle(eigenvalues)
    dominant = float(angles[np.argmax(np.abs(angles))]) if len(angles) else 0.0
    if near_cut:
        return dominant, math.nan, True
    # The principal log of an orthogonal matrix is skew with singular values
    # equal to the rotation angles, so ||skew log H||_F^2 = sum_j theta_j^2
    # over all complex eigenvalues; rank-deficient partial polars contribute
    # zero angles on their null space.
    oriented = float(math.sqrt(float(np.sum(angles**2))))
    return dominant, oriented, False


def _triangle_statistics(
    weights: ModelWeights,
    triple: tuple[str, str, str],
    ridge: float,
    method: str,
    config: RunConfig,
) -> dict[str, object]:
    k, j, i = triple
    factors_jk = _v_edge_factors(weights, k, j)
    factors_ij = _v_edge_factors(weights, j, i)
    factors_ik = _v_edge_factors(weights, k, i)
    basis = _common_support([factors_jk, factors_ij, factors_ik], config.support_rtol)
    A_jk = _project_low_rank(basis, factors_jk)
    A_ij = _project_low_rank(basis, factors_ij)
    A_dir = _project_low_rank(basis, factors_ik)
    A_cmp = A_ij @ A_jk
    norm_jk = float(torch.linalg.matrix_norm(A_jk))
    norm_ij = float(torch.linalg.matrix_norm(A_ij))
    norm_dir = float(torch.linalg.matrix_norm(A_dir))
    norm_cmp = float(torch.linalg.matrix_norm(A_cmp))
    residual = float(torch.linalg.matrix_norm(A_dir - A_cmp))
    radial = math.log(max(norm_ij, 1.0e-300)) + math.log(max(norm_jk, 1.0e-300)) - math.log(max(norm_dir, 1.0e-300))

    if method == "ridge":
        identity = torch.eye(len(A_dir), dtype=torch.float64)
        Qd, Pd = _orthogonal_polar(A_dir + ridge * identity)
        Qc, Pc = _orthogonal_polar(A_cmp + ridge * identity)
    else:
        Qd, Pd, _, _, _, _ = partial_polar(A_dir, config.support_rtol)
        Qc, Pc, _, _, _, _ = partial_polar(A_cmp, config.support_rtol)
    H = Qc @ Qd.T
    compact_score = 0.5 * float(torch.linalg.matrix_norm(torch.eye(len(H)) - H) ** 2)
    positive_distance = float(
        torch.linalg.matrix_norm(
            psd_log(Pd, ridge) - psd_log(Pc, ridge)
        )
    )
    dominant_angle, oriented_norm, branch_cut = _angle_from_orthogonal(H.numpy())
    reversed_product = A_jk @ A_ij
    order_difference = float(torch.linalg.matrix_norm(A_cmp - reversed_product)) / max(norm_cmp, 1.0e-300)
    support_dimension = int(basis.shape[1])
    return {
        "model": weights.model_name,
        "triple_id": f"{k}>{j}>{i}",
        "k": k,
        "j": j,
        "i": i,
        "direct_edge_ids": [f"{k}->{i}:V"],
        "two_step_edge_ids": [f"{k}->{j}:V", f"{j}->{i}:V"],
        "direct_gain": math.log(max(norm_dir, 1.0e-300)),
        "composed_gain": math.log(max(norm_cmp, 1.0e-300)),
        "radial_residual": radial,
        "positive_endpoint_distance": positive_distance,
        "compact_holonomy_score": compact_score,
        "compact_holonomy_per_support_dim": compact_score / max(support_dimension, 1),
        "compact_log_norm": oriented_norm,
        "dominant_oriented_angle": dominant_angle,
        "order_reversal_difference": order_difference,
        "path_residual_over_direct": residual / max(norm_dir, 1.0e-300),
        "path_residual_over_composed": residual / max(norm_cmp, 1.0e-300),
        "path_residual_symmetric": residual / max(norm_dir + norm_cmp, 1.0e-300),
        "support_dimension": support_dimension,
        "ridge_value": ridge,
        "polar_method": method,
        "near_minus_one_branch_cut": branch_cut,
    }


def _span_pattern(triple: tuple[str, str, str]) -> tuple[int, int]:
    lk, lj, li = (parse_head(label)[0] for label in triple)
    return (lj - lk, li - lj)


def triangle_candidates(
    edges: pd.DataFrame, communities: pd.DataFrame
) -> dict[tuple[str, str, str], dict[str, object]]:
    """Every causal V triple with a direct edge, with its stratum labels."""

    v_edges = edges[(edges["edge_class"] == "head_head_V") & edges["selected"]]
    outgoing: dict[str, set[str]] = {}
    for edge in v_edges.itertuples():
        outgoing.setdefault(edge.writer, set()).add(edge.reader)
    direct_lookup = {
        (edge.writer, edge.reader): edge
        for edge in edges[edges["edge_class"] == "head_head_V"].itertuples()
    }
    induction = dict(zip(communities["head"], communities["is_induction_community"]))
    candidates: dict[tuple[str, str, str], dict[str, object]] = {}

    def register(k: str, j: str, i: str) -> None:
        direct = direct_lookup.get((k, i))
        first = direct_lookup.get((k, j))
        second = direct_lookup.get((j, i))
        if direct is None or first is None or second is None:
            return
        triple = (k, j, i)
        if triple in candidates or len({k, j, i}) != 3:
            return
        candidates[triple] = {
            "min_C": min(float(direct.C), float(first.C), float(second.C)),
            "span_pattern": _span_pattern(triple),
            "inside_induction_community": all(
                induction.get(label, False) for label in triple
            ),
        }

    for k, middle_set in outgoing.items():
        for j in middle_set:
            for i in outgoing.get(j, set()):
                register(k, j, i)
    if not candidates:
        # The selected graph can be triangle-free in very small models.  Use
        # top V edges for the two causal steps and retain the actual direct
        # candidate so the residual remains interpretable.
        top = edges[edges["edge_class"] == "head_head_V"].nlargest(
            min(250, len(edges)), "C"
        )
        top_out: dict[str, list[str]] = {}
        for edge in top.itertuples():
            top_out.setdefault(edge.writer, []).append(edge.reader)
        for k, mids in top_out.items():
            for j in mids:
                for i in top_out.get(j, []):
                    register(k, j, i)
    return candidates


def enumerate_triangles(
    weights: ModelWeights,
    edges: pd.DataFrame,
    communities: pd.DataFrame,
    config: RunConfig,
    cap: int | None = None,
) -> list[tuple[str, str, str]]:
    """Stratified triple census: round-robin over (community, span-pattern)
    strata in descending coupling order, so no stratum monopolizes the cap."""

    cap = cap or config.max_triangles_per_model
    candidates = triangle_candidates(edges, communities)
    strata: dict[tuple[bool, tuple[int, int]], list[tuple[float, tuple[str, str, str]]]] = {}
    for triple, info in candidates.items():
        key = (bool(info["inside_induction_community"]), info["span_pattern"])
        strata.setdefault(key, []).append((float(info["min_C"]), triple))
    for queue in strata.values():
        queue.sort(reverse=True)
    ordered: list[tuple[str, str, str]] = []
    # Induction-community strata first inside each round so they are never
    # starved by the (much larger) outside population.
    keys = sorted(strata, key=lambda key: (not key[0], key[1]))
    while len(ordered) < cap and any(strata[key] for key in keys):
        for key in keys:
            if strata[key]:
                ordered.append(strata[key].pop(0)[1])
                if len(ordered) >= cap:
                    break
    return ordered


def run_triangles(
    weights: ModelWeights,
    edges: pd.DataFrame,
    communities: pd.DataFrame,
    config: RunConfig,
) -> pd.DataFrame:
    membership = dict(zip(communities["head"], communities["community"]))
    induction = dict(zip(communities["head"], communities["is_induction_community"]))
    rows: list[dict[str, object]] = []
    analyzed = enumerate_triangles(weights, edges, communities, config)
    for triple in analyzed:
        for method, ridges in (
            ("truncated_support", (config.ridge_grid[0],)),
            ("ridge", config.ridge_grid),
        ):
            for ridge in ridges:
                try:
                    row = _triangle_statistics(weights, triple, ridge, method, config)
                    row["row_kind"] = "v_triangle"
                    row["community_labels"] = [membership.get(head, -1) for head in triple]
                    row["inside_induction_community"] = all(induction.get(head, False) for head in triple)
                    row["matched_null_id"] = None
                    rows.append(row)
                except Exception as exc:
                    rows.append(
                        {
                            "model": weights.model_name,
                            "row_kind": "v_triangle",
                            "k": triple[0],
                            "j": triple[1],
                            "i": triple[2],
                            "ridge_value": ridge,
                            "polar_method": method,
                            "local_error": f"{type(exc).__name__}: {exc}",
                        }
                    )
    rows.extend(
        _matched_control_rows(weights, edges, communities, analyzed, config)
    )
    return pd.DataFrame(rows)


def _matched_control_rows(
    weights: ModelWeights,
    edges: pd.DataFrame,
    communities: pd.DataFrame,
    analyzed: list[tuple[str, str, str]],
    config: RunConfig,
) -> list[dict[str, object]]:
    """Span-, channel-, and coupling-matched triangles outside the induction
    community, one pool per analyzed inside-community triangle."""

    candidates = triangle_candidates(edges, communities)
    inside = [
        triple
        for triple in analyzed
        if candidates.get(triple, {}).get("inside_induction_community")
    ]
    if not inside:
        return []
    analyzed_set = set(analyzed)
    outside_pool = [
        (triple, info)
        for triple, info in candidates.items()
        if not info["inside_induction_community"] and triple not in analyzed_set
    ]
    membership = dict(zip(communities["head"], communities["community"]))
    used: set[tuple[str, str, str]] = set()
    rows: list[dict[str, object]] = []
    per_target = max(1, config.max_matched_control_triangles // max(len(inside), 1))
    for target in inside:
        target_info = candidates[target]
        matches = sorted(
            (
                (abs(float(info["min_C"]) - float(target_info["min_C"])), triple)
                for triple, info in outside_pool
                if info["span_pattern"] == target_info["span_pattern"]
                and triple not in used
            ),
        )[:per_target]
        for _, triple in matches:
            used.add(triple)
            try:
                row = _triangle_statistics(
                    weights, triple, config.ridge_grid[0], "ridge", config
                )
                row["row_kind"] = "v_triangle_matched_control"
                row["community_labels"] = [membership.get(head, -1) for head in triple]
                row["inside_induction_community"] = False
                row["matched_null_id"] = f"{target[0]}>{target[1]}>{target[2]}"
                rows.append(row)
            except Exception as exc:
                rows.append(
                    {
                        "model": weights.model_name,
                        "row_kind": "v_triangle_matched_control",
                        "k": triple[0],
                        "j": triple[1],
                        "i": triple[2],
                        "matched_null_id": f"{target[0]}>{target[1]}>{target[2]}",
                        "local_error": f"{type(exc).__name__}: {exc}",
                    }
                )
            if len(rows) >= config.max_matched_control_triangles:
                return rows
    return rows


def _stream_frame_randomized(weights: ModelWeights, rng: np.random.Generator) -> ModelWeights:
    """Rotate every head factor's residual-stream frame to a Haar-random one
    while preserving its singular spectrum and its head-internal Gram."""

    def randomized(stack: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(stack)
        layers, heads, d, r = stack.shape
        for layer in range(layers):
            for head in range(heads):
                U, singular, Vh = torch.linalg.svd(stack[layer, head], full_matrices=False)
                gaussian = torch.tensor(
                    rng.standard_normal((d, r)), dtype=torch.float64
                )
                frame = torch.linalg.qr(gaussian, mode="reduced").Q
                out[layer, head] = (frame * singular) @ Vh
        return out

    return dataclasses.replace(
        weights,
        Q=randomized(weights.Q),
        K=randomized(weights.K),
        V=randomized(weights.V),
        O=randomized(weights.O),
    )


def run_triangle_surrogates(
    weights: ModelWeights,
    induction_heads: list[str],
    config: RunConfig,
) -> pd.DataFrame:
    """Full-pipeline surrogate: randomize stream frames spectrum-preserving,
    rebuild the map, refit nulls, reselect edges, rebuild communities, and
    recompute the triangle statistics per draw."""

    from .wang_map import add_empirical_selection, compute_head_edges

    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(config.seed + 811)
    for draw in range(config.triangle_surrogate_draws):
        surrogate = _stream_frame_randomized(weights, rng)
        edges = add_empirical_selection(compute_head_edges(surrogate), config.fdr_q)
        communities, _, _ = build_communities(
            surrogate, edges, induction_heads, config.seed + 811 + draw
        )
        triples = enumerate_triangles(
            surrogate,
            edges,
            communities,
            config,
            cap=config.max_surrogate_triangles_per_draw,
        )
        induction = dict(
            zip(communities["head"], communities["is_induction_community"])
        )
        for triple in triples:
            try:
                row = _triangle_statistics(
                    surrogate, triple, config.ridge_grid[0], "ridge", config
                )
                row["row_kind"] = "v_triangle_surrogate"
                row["surrogate_draw"] = draw
                row["inside_induction_community"] = all(
                    induction.get(head, False) for head in triple
                )
                rows.append(row)
            except Exception as exc:
                rows.append(
                    {
                        "model": weights.model_name,
                        "row_kind": "v_triangle_surrogate",
                        "surrogate_draw": draw,
                        "k": triple[0],
                        "j": triple[1],
                        "i": triple[2],
                        "local_error": f"{type(exc).__name__}: {exc}",
                    }
                )
        del surrogate, edges, communities
        import gc

        gc.collect()
    return pd.DataFrame(rows)


_TRIANGLE_CLASS_STATS = (
    ("compact_holonomy_per_support_dim", "compact"),
    ("positive_endpoint_distance", "endpoint"),
    ("path_residual_symmetric", "shape"),
    ("radial_residual_abs", "radial"),
    ("order_reversal_difference", "order"),
)


def triangle_surrogate_summary(
    model_name: str, real: pd.DataFrame, surrogate: pd.DataFrame
) -> tuple[list[dict[str, object]], str]:
    """Real-versus-surrogate separations per statistic plus the design's
    classification sentence, emitted from the comparison."""

    def prepared(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        if "radial_residual" in out:
            out["radial_residual_abs"] = pd.to_numeric(
                out["radial_residual"], errors="coerce"
            ).abs()
        return out

    real = prepared(real)
    surrogate = prepared(surrogate)
    rows: list[dict[str, object]] = []
    separations: dict[str, float] = {}
    for column, label in _TRIANGLE_CLASS_STATS:
        real_values = pd.to_numeric(real.get(column), errors="coerce").dropna()
        if not len(real_values) or "surrogate_draw" not in surrogate:
            continue
        draw_medians = (
            surrogate.assign(
                value=pd.to_numeric(surrogate.get(column), errors="coerce")
            )
            .dropna(subset=["value"])
            .groupby("surrogate_draw")["value"]
            .median()
        )
        if not len(draw_medians):
            continue
        center = float(draw_medians.median())
        scale = max(
            1.4826 * float((draw_medians - center).abs().median()), 1.0e-12
        )
        real_median = float(real_values.median())
        separation = (real_median - center) / scale
        separations[label] = separation
        rows.append(
            {
                "model": model_name,
                "row_kind": "triangle_surrogate_summary",
                "statistic": column,
                "real_median": real_median,
                "real_n": int(len(real_values)),
                "surrogate_median_of_draw_medians": center,
                "surrogate_robust_scale": scale,
                "separation": separation,
                "surrogate_draw_medians_below_real": int(
                    (draw_medians <= real_median).sum()
                ),
                "surrogate_draws": int(len(draw_medians)),
            }
        )
    below = sorted(label for label, value in separations.items() if value <= -2.0)
    above = sorted(label for label, value in separations.items() if value >= 2.0)
    if not separations:
        sentence = "No real-versus-surrogate triangle comparison was constructible."
    elif not below and not above:
        sentence = (
            "Against full-pipeline spectrum-preserving surrogates every triangle register was "
            "indistinguishable from random stream frames: the communities behave as scalar "
            "clusters at this instrument's resolution."
        )
    else:
        parts = []
        if below:
            parts.append(
                "below the surrogate distribution in " + ", ".join(below)
            )
        if above:
            parts.append(
                "above it in " + ", ".join(above)
            )
        residue = "compact" in below
        endpoint = "endpoint" in below
        if endpoint and not residue:
            reading = "endpoint-compatible subbundle structure"
        elif residue:
            reading = "partial parallel transport with reduced compact residue"
        else:
            reading = "mixed residual structure"
        sentence = (
            "Real triangles sit "
            + " and ".join(parts)
            + f" (|separation| >= 2 robust sigma), consistent with {reading}; all other registers match the surrogate null."
        )
    return rows, sentence


def _square_polar(matrix: torch.Tensor, rtol: float) -> torch.Tensor:
    return partial_polar(matrix, rtol)[0]


def role_bridge(
    weights: ModelWeights,
    label: str,
    kind: str,
    activation: torch.Tensor | None,
    ridge: float,
    rtol: float,
) -> torch.Tensor:
    layer, head = parse_head(label)
    Q = weights.Q[layer, head]
    K = weights.K[layer, head]
    if kind == "factor_overlap":
        return _square_polar(Q.T @ K, rtol)
    if kind == "gram":
        gq = Q.T @ Q
        gk = K.T @ K
        # Polar of the positive-factor comparison, retaining the trained
        # coordinate frame on each role fibre.
        qroot = torch.linalg.cholesky(gq + ridge * torch.eye(weights.d_head))
        kroot = torch.linalg.cholesky(gk + ridge * torch.eye(weights.d_head))
        return _square_polar(qroot.T @ kroot, rtol)
    if kind == "activation":
        if activation is None:
            raise ValueError("activation bridge requested without normalized activation samples")
        activation = activation.double()
        qcoord = activation @ Q
        kcoord = activation @ K
        solution = torch.linalg.solve(
            kcoord.T @ kcoord + ridge * torch.eye(weights.d_head),
            kcoord.T @ qcoord,
        )
        # Row q = row k @ solution; column q = solution.T column k.
        return _square_polar(solution.T, rtol)
    raise ValueError(f"unknown role bridge {kind}")


def typed_qk_map(weights: ModelWeights, writer: str, reader: str) -> torch.Tensor:
    """Causal typed map K_writer -> Q_reader through the writer's OV operator."""

    wl, wh = parse_head(writer)
    rl, rh = parse_head(reader)
    Qr = weights.Q[rl, rh]
    Kw = weights.K[wl, wh]
    Ow = weights.O[wl, wh]
    Vw = weights.V[wl, wh]
    result = Qr.T @ Ow @ Vw.T @ Kw
    return result / max(float(torch.linalg.matrix_norm(result)), 1.0e-300)


def _loop_readout(loop: torch.Tensor) -> tuple[float, float, int]:
    compact, positive = _orthogonal_polar(loop)
    score = 0.5 * float(
        torch.linalg.matrix_norm(torch.eye(len(compact)) - compact) ** 2
    )
    positive_log = float(torch.linalg.matrix_norm(psd_log(positive, 1.0e-9)))
    rank = int(torch.linalg.matrix_rank(loop).item())
    return score, positive_log, rank


def run_role_loops(
    weights: ModelWeights,
    triangles: pd.DataFrame,
    normalized_activations: dict[int, torch.Tensor] | None,
    config: RunConfig,
) -> pd.DataFrame:
    base = triangles[
        (triangles.get("row_kind", "v_triangle") == "v_triangle")
        & (triangles["polar_method"] == "ridge")
        & (triangles["ridge_value"] == config.ridge_grid[0])
    ].drop_duplicates(["k", "j", "i"])
    rows: list[dict[str, object]] = []
    for triangle in base.itertuples():
        k, j, i = triangle.k, triangle.j, triangle.i
        Tij = typed_qk_map(weights, j, i)
        Tjk = typed_qk_map(weights, k, j)
        Tik = typed_qk_map(weights, k, i)
        for ridge in config.ridge_grid:
            collapsed = Tij @ Tjk @ torch.linalg.pinv(Tik, rtol=config.support_rtol)
            collapsed_score, collapsed_positive, collapsed_rank = _loop_readout(collapsed)
            for bridge_kind in ("factor_overlap", "gram", "activation"):
                activation = None
                if normalized_activations is not None:
                    activation = normalized_activations.get(parse_head(j)[0])
                try:
                    bridge = role_bridge(
                        weights,
                        j,
                        bridge_kind,
                        activation,
                        ridge,
                        config.support_rtol,
                    )
                    loop = (
                        Tij
                        @ torch.linalg.pinv(bridge, rtol=config.support_rtol)
                        @ Tjk
                        @ torch.linalg.pinv(Tik, rtol=config.support_rtol)
                    )
                    score, positive, rank = _loop_readout(loop)
                    rows.append(
                        {
                            "model": weights.model_name,
                            "k": k,
                            "j": j,
                            "i": i,
                            "bridge": bridge_kind,
                            "ridge": ridge,
                            "compact_class_score": score,
                            "positive_log_norm": positive,
                            "support_dimension": rank,
                            "role_collapsed_compact_score": collapsed_score,
                            "role_collapsed_positive_log_norm": collapsed_positive,
                            "role_collapsed_support_dimension": collapsed_rank,
                            "role_complete_minus_collapsed": score - collapsed_score,
                        }
                    )
                except Exception as exc:
                    rows.append(
                        {
                            "model": weights.model_name,
                            "k": k,
                            "j": j,
                            "i": i,
                            "bridge": bridge_kind,
                            "ridge": ridge,
                            "local_error": f"{type(exc).__name__}: {exc}",
                        }
                    )
    frame = pd.DataFrame(rows)
    if len(frame):
        valid = frame[frame.get("local_error", pd.Series(index=frame.index, dtype=object)).isna()]
        sensitivity = (
            valid.groupby(["k", "j", "i", "ridge"])["compact_class_score"].std().rename("bridge_sensitivity")
        )
        frame = frame.merge(sensitivity, on=["k", "j", "i", "ridge"], how="left")
    return frame


def run_graph_connection(
    weights: ModelWeights,
    edges: pd.DataFrame,
    induction_heads: list[str],
    normalized_activations: dict[int, torch.Tensor] | None,
    config: RunConfig,
) -> GraphResult:
    communities, induction_community, modularity = build_communities(
        weights, edges, induction_heads, config.seed
    )
    triangles = run_triangles(weights, edges, communities, config)
    role = run_role_loops(weights, triangles, normalized_activations, config) if len(triangles) else pd.DataFrame()
    surrogates = run_triangle_surrogates(weights, induction_heads, config)
    valid_triangles = triangles[
        triangles.get("local_error", pd.Series(index=triangles.index, dtype=object)).isna()
    ] if len(triangles) else triangles
    real_rows = (
        valid_triangles[
            (valid_triangles.get("row_kind", "") == "v_triangle")
            & (valid_triangles.get("polar_method", "") == "ridge")
            & (valid_triangles.get("ridge_value", np.nan) == config.ridge_grid[0])
        ]
        if len(valid_triangles)
        else valid_triangles
    )
    valid_surrogates = (
        surrogates[
            surrogates.get(
                "local_error", pd.Series(index=surrogates.index, dtype=object)
            ).isna()
        ]
        if len(surrogates)
        else surrogates
    )
    summary_rows, classification = triangle_surrogate_summary(
        weights.model_name, real_rows, valid_surrogates
    )
    triangles = pd.concat(
        [triangles, surrogates, pd.DataFrame(summary_rows)],
        ignore_index=True,
        sort=False,
    )
    ridge_triangles = real_rows
    observations = [
        f"The selected head graph had modularity {modularity:.3f}; its induction-overlap community contained {len(induction_community)} heads.",
        (
            f"V-channel triangles had median compact score {float(ridge_triangles['compact_holonomy_score'].median()):.3g} and median symmetric path residual {float(ridge_triangles['path_residual_symmetric'].median()):.3g}."
            if len(ridge_triangles)
            else "No interpretable V-channel triangle was present at the configured edge budget."
        ),
        classification,
        (
            f"Role-complete Q/K loop class functions varied across bridge families by median standard deviation {float(role['bridge_sensitivity'].median()):.3g}."
            if len(role) and "bridge_sensitivity" in role
            else "Role-complete Q/K loops were unavailable because no typed triangle could be constructed."
        ),
    ]
    return GraphResult(
        communities=communities,
        triangles=triangles,
        role_loops=role,
        induction_community=induction_community,
        observations=observations,
    )
