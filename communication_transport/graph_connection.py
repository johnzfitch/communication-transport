from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd
import torch
from scipy import linalg as spla

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
    logarithm = np.real_if_close(spla.logm(matrix), tol=1_000).real
    oriented = float(np.linalg.norm(0.5 * (logarithm - logarithm.T)))
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
    return {
        "model": weights.model_name,
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
        "compact_log_norm": oriented_norm,
        "dominant_oriented_angle": dominant_angle,
        "order_reversal_difference": order_difference,
        "path_residual_over_direct": residual / max(norm_dir, 1.0e-300),
        "path_residual_over_composed": residual / max(norm_cmp, 1.0e-300),
        "path_residual_symmetric": residual / max(norm_dir + norm_cmp, 1.0e-300),
        "support_dimension": len(basis),
        "ridge_value": ridge,
        "polar_method": method,
        "near_minus_one_branch_cut": branch_cut,
    }


def enumerate_triangles(
    weights: ModelWeights,
    edges: pd.DataFrame,
    communities: pd.DataFrame,
    config: RunConfig,
) -> list[tuple[str, str, str]]:
    v_edges = edges[
        (edges["edge_class"] == "head_head_V") & edges["selected"]
    ]
    outgoing: dict[str, set[str]] = {}
    for edge in v_edges.itertuples():
        outgoing.setdefault(edge.writer, set()).add(edge.reader)
    direct_lookup = {
        (edge.writer, edge.reader): edge
        for edge in edges[edges["edge_class"] == "head_head_V"].itertuples()
    }
    candidates: list[tuple[float, tuple[str, str, str]]] = []
    for k, middle_set in outgoing.items():
        for j in middle_set:
            for i in outgoing.get(j, set()):
                direct = direct_lookup.get((k, i))
                if direct is None:
                    continue
                score = min(
                    float(direct.C),
                    float(direct_lookup[(k, j)].C),
                    float(direct_lookup[(j, i)].C),
                )
                candidates.append((score, (k, j, i)))
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
                    direct = direct_lookup.get((k, i))
                    if direct is not None:
                        candidates.append((float(direct.C), (k, j, i)))
    seen: set[tuple[str, str, str]] = set()
    ordered: list[tuple[str, str, str]] = []
    for _, triple in sorted(candidates, reverse=True):
        if triple not in seen:
            ordered.append(triple)
            seen.add(triple)
        if len(ordered) >= config.max_triangles_per_model:
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
    for triple in enumerate_triangles(weights, edges, communities, config):
        for method, ridges in (
            ("truncated_support", (config.ridge_grid[0],)),
            ("ridge", config.ridge_grid),
        ):
            for ridge in ridges:
                try:
                    row = _triangle_statistics(weights, triple, ridge, method, config)
                    row["community_labels"] = [membership.get(head, -1) for head in triple]
                    row["inside_induction_community"] = all(induction.get(head, False) for head in triple)
                    row["matched_null_id"] = None
                    rows.append(row)
                except Exception as exc:
                    rows.append(
                        {
                            "model": weights.model_name,
                            "k": triple[0],
                            "j": triple[1],
                            "i": triple[2],
                            "ridge_value": ridge,
                            "polar_method": method,
                            "local_error": f"{type(exc).__name__}: {exc}",
                        }
                    )
    return pd.DataFrame(rows)


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
        (triangles["polar_method"] == "ridge")
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
    valid_triangles = triangles[
        triangles.get("local_error", pd.Series(index=triangles.index, dtype=object)).isna()
    ] if len(triangles) else triangles
    ridge_triangles = valid_triangles[valid_triangles.get("polar_method", "") == "ridge"] if len(valid_triangles) else valid_triangles
    observations = [
        f"The selected head graph had modularity {modularity:.3f}; its induction-overlap community contained {len(induction_community)} heads.",
        (
            f"V-channel triangles had median compact score {float(ridge_triangles['compact_holonomy_score'].median()):.3g} and median symmetric path residual {float(ridge_triangles['path_residual_symmetric'].median()):.3g}."
            if len(ridge_triangles)
            else "No interpretable V-channel triangle was present at the configured edge budget."
        ),
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
