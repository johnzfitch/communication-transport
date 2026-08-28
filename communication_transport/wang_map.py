from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from scipy import stats as sps

from .config import RunConfig
from .weights import ModelWeights, head_norms, inner_grams, psd_rank


MAD_TO_SD = 1.4826


@dataclass(slots=True)
class MapResult:
    edges: pd.DataFrame
    neuron_wires: pd.DataFrame
    families: pd.DataFrame
    dense_crosschecks: pd.DataFrame
    observations: list[str]


def bh_qvalues(p_values: np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=np.float64)
    if values.size == 0:
        return values.copy()
    order = np.argsort(values, kind="mergesort")
    adjusted = values[order] * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    out = np.empty_like(adjusted)
    out[order] = np.minimum(adjusted, 1.0)
    return out


def _matrix_trace_square_from_factors(
    outer_gram: torch.Tensor, inner: torch.Tensor
) -> float:
    # G = outer inner outer^T and outer^T outer = outer_gram.
    return float(torch.trace(inner @ outer_gram @ inner @ outer_gram))


def _null_fields(
    numerator: float,
    tr_g: float,
    tr_h: float,
    tr_g2: float,
    tr_h2: float,
    d_model: int,
) -> tuple[float, float, float]:
    var_t = 2.0 / ((d_model - 1) * (d_model + 2))
    var_t *= max(tr_g2 - tr_g * tr_g / d_model, 0.0)
    var_t *= max(tr_h2 - tr_h * tr_h / d_model, 0.0)
    var_c2 = var_t / max(tr_g * tr_g * tr_h * tr_h, 1.0e-300)
    expected_t = tr_g * tr_h / d_model
    z = (numerator - expected_t) / math.sqrt(max(var_t, 1.0e-300))
    return 1.0 / d_model, var_c2, z


def _small_head_geometry(weights: ModelWeights):
    g = inner_grams(weights)
    norms = head_norms(weights, g)
    trace2: dict[str, torch.Tensor] = {
        key: torch.empty((weights.n_layers, weights.n_heads), dtype=torch.float64)
        for key in ("K", "Q", "V", "writer")
    }
    for layer in range(weights.n_layers):
        for head in range(weights.n_heads):
            gq, gk = g["Q"][layer, head], g["K"][layer, head]
            gv, go = g["V"][layer, head], g["O"][layer, head]
            trace2["K"][layer, head] = torch.trace(gq @ gk @ gq @ gk)
            trace2["Q"][layer, head] = trace2["K"][layer, head]
            trace2["V"][layer, head] = torch.trace(go @ gv @ go @ gv)
            trace2["writer"][layer, head] = trace2["V"][layer, head]
    return g, norms, trace2


def compute_head_edges(weights: ModelWeights) -> pd.DataFrame:
    g, norms, trace2 = _small_head_geometry(weights)
    left_gram = {"K": "Q", "Q": "K", "V": "O"}
    right_factor = {"K": weights.K, "Q": weights.Q, "V": weights.V}
    rows: list[dict[str, object]] = []
    for reader_layer in range(1, weights.n_layers):
        for writer_layer in range(reader_layer):
            for channel in ("K", "Q", "V"):
                # X[reader head, writer head, reader fibre, writer fibre].
                X = torch.einsum(
                    "hdi,gdj->hgij",
                    right_factor[channel][reader_layer],
                    weights.O[writer_layer],
                )
                gl = g[left_gram[channel]][reader_layer]
                gv = g["V"][writer_layer]
                numerator = torch.einsum(
                    "hgij,hik,hgkl,gjl->hg",
                    X,
                    gl,
                    X,
                    gv,
                ).clamp_min(0.0)
                denominator = (
                    norms[channel][reader_layer, :, None]
                    * norms["writer"][writer_layer, None, :]
                )
                C2 = numerator / denominator.clamp_min(1.0e-300)
                for reader_head in range(weights.n_heads):
                    for writer_head in range(weights.n_heads):
                        num = float(numerator[reader_head, writer_head])
                        trg = float(norms[channel][reader_layer, reader_head])
                        trh = float(norms["writer"][writer_layer, writer_head])
                        trg2 = float(trace2[channel][reader_layer, reader_head])
                        trh2 = float(trace2["writer"][writer_layer, writer_head])
                        mean, variance, z = _null_fields(
                            num, trg, trh, trg2, trh2, weights.d_model
                        )
                        reader_purity = trg2 / max(trg * trg, 1.0e-300)
                        writer_purity = trh2 / max(trh * trh, 1.0e-300)
                        rows.append(
                            {
                                "model": weights.model_name,
                                "writer_layer": writer_layer,
                                "writer": weights.head_label(writer_layer, writer_head),
                                "reader_layer": reader_layer,
                                "reader": weights.head_label(reader_layer, reader_head),
                                "channel": channel,
                                "edge_class": f"head_head_{channel}",
                                "layer_span": reader_layer - writer_layer,
                                "C": math.sqrt(max(float(C2[reader_head, writer_head]), 0.0)),
                                "C2": float(C2[reader_head, writer_head]),
                                "theoretical_mean": mean,
                                "theoretical_variance": variance,
                                "theoretical_z": z,
                                "reader_rank": psd_rank(gl[reader_head]),
                                "writer_rank": psd_rank(gv[writer_head]),
                                "reader_purity": reader_purity,
                                "writer_purity": writer_purity,
                                "signed_scalar_if_rank_one": np.nan,
                                "stored_scope": "all_head_edges",
                            }
                        )
    return pd.DataFrame(rows)


def add_empirical_selection(edges: pd.DataFrame, q_level: float) -> pd.DataFrame:
    out = edges.copy()
    out["empirical_center"] = np.nan
    out["empirical_scale"] = np.nan
    out["empirical_z"] = np.nan
    for (_, span), group in out.groupby(["edge_class", "layer_span"]):
        values = group["C"].to_numpy()
        center = float(np.median(values))
        scale = max(float(MAD_TO_SD * np.median(np.abs(values - center))), 1.0e-12)
        out.loc[group.index, "empirical_center"] = center
        out.loc[group.index, "empirical_scale"] = scale
        out.loc[group.index, "empirical_z"] = (values - center) / scale
    out["fdr_q_high"] = np.nan
    out["fdr_q_low"] = np.nan
    for _, group in out.groupby("edge_class"):
        z = out.loc[group.index, "empirical_z"].to_numpy(dtype=np.float64)
        out.loc[group.index, "fdr_q_high"] = bh_qvalues(sps.norm.sf(z))
        out.loc[group.index, "fdr_q_low"] = bh_qvalues(sps.norm.cdf(z))
    out["fdr_q"] = np.minimum(out["fdr_q_high"], out["fdr_q_low"])
    out["selected"] = out["fdr_q_high"] <= q_level
    out["avoidant"] = out["fdr_q_low"] <= q_level
    return out


def _dense_crosschecks(
    weights: ModelWeights, edges: pd.DataFrame, count: int, seed: int
) -> pd.DataFrame:
    if edges.empty or count <= 0:
        return pd.DataFrame()
    sample = edges.sample(min(count, len(edges)), random_state=seed)
    rows: list[dict[str, object]] = []
    for edge in sample.itertuples():
        wl, wh = int(edge.writer_layer), int(str(edge.writer).split("H")[1])
        rl, rh = int(edge.reader_layer), int(str(edge.reader).split("H")[1])
        writer = weights.ov(wl, wh)
        if edge.channel == "K":
            reader = weights.qk(rl, rh)
        elif edge.channel == "Q":
            reader = weights.qk(rl, rh).T
        else:
            reader = weights.ov(rl, rh)
        dense = float(
            torch.linalg.matrix_norm(reader @ writer)
            / (torch.linalg.matrix_norm(reader) * torch.linalg.matrix_norm(writer))
        )
        rows.append(
            {
                "model": weights.model_name,
                "writer": edge.writer,
                "reader": edge.reader,
                "channel": edge.channel,
                "factored_C": float(edge.C),
                "dense_C": dense,
                "absolute_residual": abs(dense - float(edge.C)),
            }
        )
    return pd.DataFrame(rows)


def _interface_gram_rows(
    weights: ModelWeights, head_edges: pd.DataFrame
) -> pd.DataFrame:
    """Compute the head/interface classes; rotary models have no learned POS row."""

    g, norms, trace2 = _small_head_geometry(weights)
    gram_device = "cuda" if torch.cuda.is_available() else "cpu"

    def residual_gram(factor: torch.Tensor) -> torch.Tensor:
        # Vocabulary interfaces are full rank and large only in their source
        # axis.  Form their d_model Gram once on the accelerator.
        working = factor.float().to(gram_device)
        result = (working @ working.T).double().cpu()
        del working
        return result

    factors = {
        "EMB": weights.W_E.T,
    }
    if weights.W_pos is not None:
        factors["POS"] = weights.W_pos.T
    rows: list[dict[str, object]] = []
    reader_outer = {"K": weights.K, "Q": weights.Q, "V": weights.V}
    reader_inner = {"K": g["Q"], "Q": g["K"], "V": g["O"]}
    for writer_name, factor in factors.items():
        H = residual_gram(factor)
        trh = float(torch.trace(H))
        trh2 = float(torch.sum(H * H))
        writer_rank = psd_rank(H)
        for layer in range(weights.n_layers):
            for head in range(weights.n_heads):
                for channel in ("K", "Q", "V"):
                    outer = reader_outer[channel][layer, head]
                    inner = reader_inner[channel][layer, head]
                    G = outer @ inner @ outer.T
                    num = float(torch.sum(G * H))
                    trg = float(norms[channel][layer, head])
                    trg2 = float(trace2[channel][layer, head])
                    mean, variance, z = _null_fields(
                        num, trg, trh, trg2, trh2, weights.d_model
                    )
                    c2 = num / max(trg * trh, 1.0e-300)
                    rows.append(
                        {
                            "model": weights.model_name,
                            "writer_layer": -1,
                            "writer": writer_name,
                            "reader_layer": layer,
                            "reader": weights.head_label(layer, head),
                            "channel": channel,
                            "edge_class": f"{writer_name.lower()}_head_{channel}",
                            "layer_span": layer + 1,
                            "C": math.sqrt(max(c2, 0.0)),
                            "C2": c2,
                            "theoretical_mean": mean,
                            "theoretical_variance": variance,
                            "theoretical_z": z,
                            "reader_rank": psd_rank(inner),
                            "writer_rank": writer_rank,
                            "reader_purity": trg2 / max(trg * trg, 1.0e-300),
                            "writer_purity": trh2 / max(trh * trh, 1.0e-300),
                            "signed_scalar_if_rank_one": np.nan,
                            "empirical_center": math.sqrt(1.0 / weights.d_model),
                            "empirical_scale": math.sqrt(max(variance, 0.0)),
                            "empirical_z": z,
                            "fdr_q_high": np.nan,
                            "fdr_q_low": np.nan,
                            "fdr_q": np.nan,
                            "selected": False,
                            "avoidant": False,
                            "stored_scope": "all_interface_head_edges",
                        }
                    )
    G_u = residual_gram(weights.W_U)
    trg = float(torch.trace(G_u))
    trg2 = float(torch.sum(G_u * G_u))
    for layer in range(weights.n_layers):
        for head in range(weights.n_heads):
            O, gv = weights.O[layer, head], g["V"][layer, head]
            H = O @ gv @ O.T
            trh = float(norms["writer"][layer, head])
            trh2 = float(trace2["writer"][layer, head])
            num = float(torch.sum(G_u * H))
            mean, variance, z = _null_fields(
                num, trg, trh, trg2, trh2, weights.d_model
            )
            c2 = num / max(trg * trh, 1.0e-300)
            rows.append(
                {
                    "model": weights.model_name,
                    "writer_layer": layer,
                    "writer": weights.head_label(layer, head),
                    "reader_layer": weights.n_layers,
                    "reader": "UNEMB",
                    "channel": "interface",
                    "edge_class": "head_unembed",
                    "layer_span": weights.n_layers - layer,
                    "C": math.sqrt(max(c2, 0.0)),
                    "C2": c2,
                    "theoretical_mean": mean,
                    "theoretical_variance": variance,
                    "theoretical_z": z,
                    "reader_rank": psd_rank(G_u),
                    "writer_rank": psd_rank(gv),
                    "reader_purity": trg2 / max(trg * trg, 1.0e-300),
                    "writer_purity": trh2 / max(trh * trh, 1.0e-300),
                    "signed_scalar_if_rank_one": np.nan,
                    "empirical_center": math.sqrt(1.0 / weights.d_model),
                    "empirical_scale": math.sqrt(max(variance, 0.0)),
                    "empirical_z": z,
                    "fdr_q_high": np.nan,
                    "fdr_q_low": np.nan,
                    "fdr_q": np.nan,
                    "selected": False,
                    "avoidant": False,
                    "stored_scope": "all_interface_head_edges",
                }
            )
    return pd.DataFrame(rows)


def _rank_one_variance(purity: np.ndarray | float, d_model: int) -> np.ndarray:
    purity_array = np.asarray(purity, dtype=np.float64)
    return (
        2.0
        / (d_model * (d_model + 2.0))
        * np.maximum(purity_array - 1.0 / d_model, 0.0)
    )


def _finish_mixed_class(
    weights: ModelWeights,
    config: RunConfig,
    edge_class: str,
    channel: str,
    pieces: list[dict[str, np.ndarray]],
) -> tuple[pd.DataFrame, dict[str, object]]:
    if not pieces:
        return pd.DataFrame(), {
            "model": weights.model_name,
            "edge_class": edge_class,
            "n_candidates": 0,
            "n_selected": 0,
            "n_avoidant": 0,
        }
    data = {
        key: np.concatenate([piece[key] for piece in pieces])
        for key in pieces[0]
    }
    c2 = np.clip(data["C2"], 0.0, None)
    coupling = np.sqrt(c2)
    theoretical_variance = np.maximum(data["theoretical_variance"], 1.0e-300)
    theoretical_z = (c2 - 1.0 / weights.d_model) / np.sqrt(theoretical_variance)
    empirical_z = np.zeros_like(coupling)
    empirical_center = np.zeros_like(coupling)
    empirical_scale = np.zeros_like(coupling)
    for span in np.unique(data["layer_span"]):
        mask = data["layer_span"] == span
        values = coupling[mask]
        center = float(np.median(values))
        scale = max(float(MAD_TO_SD * np.median(np.abs(values - center))), 1.0e-12)
        empirical_center[mask] = center
        empirical_scale[mask] = scale
        empirical_z[mask] = (values - center) / scale
    q_high = bh_qvalues(sps.norm.sf(empirical_z))
    q_low = bh_qvalues(sps.norm.cdf(empirical_z))
    selected = q_high <= config.fdr_q
    avoidant = q_low <= config.fdr_q

    cap = min(config.max_mixed_edges_per_class, len(coupling))
    high_order = np.flatnonzero(selected)[np.argsort(empirical_z[selected])[::-1]]
    low_order = np.flatnonzero(avoidant)[np.argsort(empirical_z[avoidant])]
    # Preserve both tails under the explicit artifact cap, then fill unused
    # space with the strongest unselected controls.
    half = cap // 2
    retained = list(high_order[:half]) + list(low_order[: cap - min(half, len(high_order))])
    retained_set = set(retained)
    if len(retained) < cap:
        extremes = np.argsort(np.abs(empirical_z))[::-1]
        for index in extremes:
            if int(index) not in retained_set:
                retained.append(int(index))
                retained_set.add(int(index))
            if len(retained) == cap:
                break

    rows: list[dict[str, object]] = []
    for index in retained:
        wl = int(data["writer_layer"][index])
        wi = int(data["writer_index"][index])
        rl = int(data["reader_layer"][index])
        ri = int(data["reader_index"][index])
        if edge_class == "head_neuron":
            writer, reader = weights.head_label(wl, wi), f"L{rl}N{ri}"
        elif edge_class.startswith("neuron_head_"):
            writer, reader = f"L{wl}N{wi}", weights.head_label(rl, ri)
        elif edge_class in ("emb_neuron", "pos_neuron"):
            writer, reader = edge_class.split("_")[0].upper(), f"L{rl}N{ri}"
        elif edge_class == "neuron_unembed":
            writer, reader = f"L{wl}N{wi}", "UNEMB"
        else:
            raise ValueError(edge_class)
        rows.append(
            {
                "model": weights.model_name,
                "writer_layer": wl,
                "writer": writer,
                "reader_layer": rl,
                "reader": reader,
                "channel": channel,
                "edge_class": edge_class,
                "layer_span": int(data["layer_span"][index]),
                "C": float(coupling[index]),
                "C2": float(c2[index]),
                "theoretical_mean": 1.0 / weights.d_model,
                "theoretical_variance": float(theoretical_variance[index]),
                "theoretical_z": float(theoretical_z[index]),
                "reader_rank": int(data["reader_rank"][index]),
                "writer_rank": int(data["writer_rank"][index]),
                "reader_purity": float(data["reader_purity"][index]),
                "writer_purity": float(data["writer_purity"][index]),
                "signed_scalar_if_rank_one": np.nan,
                "empirical_center": float(empirical_center[index]),
                "empirical_scale": float(empirical_scale[index]),
                "empirical_z": float(empirical_z[index]),
                "fdr_q_high": float(q_high[index]),
                "fdr_q_low": float(q_low[index]),
                "fdr_q": float(min(q_high[index], q_low[index])),
                "selected": bool(selected[index]),
                "avoidant": bool(avoidant[index]),
                "stored_scope": (
                    f"bounded_{cap}_edge_tail_table_from_{len(coupling)}_fully_scanned_candidates"
                ),
            }
        )
    summary = {
        "model": weights.model_name,
        "edge_class": edge_class,
        "n_candidates": len(coupling),
        "median_C": float(np.median(coupling)),
        "median_theoretical_z": float(np.median(theoretical_z)),
        "above_z2_fraction": float(np.mean(theoretical_z >= 2.0)),
        "below_z_minus2_fraction": float(np.mean(theoretical_z <= -2.0)),
        "n_selected": int(selected.sum()),
        "n_avoidant": int(avoidant.sum()),
        "n_stored": len(rows),
    }
    return pd.DataFrame(rows), summary


def scan_mixed_edges(
    weights: ModelWeights, config: RunConfig
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    """Fully scan the seven mixed classes and retain bounded two-tail tables."""

    grams, norms, trace2 = _small_head_geometry(weights)
    device = config.device if torch.cuda.is_available() and str(config.device).startswith("cuda") else "cpu"
    win = torch.nn.functional.normalize(weights.W_in.float(), dim=-1).to(device)
    wout = torch.nn.functional.normalize(weights.W_out.float(), dim=-1).to(device)
    O = weights.O.float().to(device)
    Q = weights.Q.float().to(device)
    K = weights.K.float().to(device)
    V = weights.V.float().to(device)
    pieces_by_class: dict[str, list[dict[str, np.ndarray]]] = {
        "head_neuron": [],
        "neuron_head_K": [],
        "neuron_head_Q": [],
        "neuron_head_V": [],
        "emb_neuron": [],
        "pos_neuron": [],
        "neuron_unembed": [],
    }

    def add(
        name: str,
        c2: torch.Tensor,
        wl: np.ndarray,
        wi: np.ndarray,
        rl: np.ndarray,
        ri: np.ndarray,
        span: np.ndarray,
        reader_rank: np.ndarray,
        writer_rank: np.ndarray,
        reader_purity: np.ndarray,
        writer_purity: np.ndarray,
        variance: np.ndarray,
    ) -> None:
        pieces_by_class[name].append(
            {
                "C2": c2.detach().float().cpu().numpy().reshape(-1).astype(np.float64),
                "writer_layer": wl.reshape(-1),
                "writer_index": wi.reshape(-1),
                "reader_layer": rl.reshape(-1),
                "reader_index": ri.reshape(-1),
                "layer_span": span.reshape(-1),
                "reader_rank": reader_rank.reshape(-1),
                "writer_rank": writer_rank.reshape(-1),
                "reader_purity": reader_purity.reshape(-1),
                "writer_purity": writer_purity.reshape(-1),
                "theoretical_variance": variance.reshape(-1),
            }
        )

    # Head writer -> neuron reader.  Same-layer attention-to-MLP flow exists
    # only in the serial topology.
    for reader_layer in range(weights.n_layers):
        writer_stop = reader_layer if weights.parallel_attn_mlp else reader_layer + 1
        for writer_layer in range(writer_stop):
            a = torch.einsum("nd,hdi->hni", win[reader_layer], O[writer_layer])
            c2 = torch.einsum(
                "hni,hij,hnj->hn", a, grams["V"][writer_layer].float().to(device), a
            ) / norms["writer"][writer_layer].float().to(device)[:, None].clamp_min(1.0e-30)
            h, n = c2.shape
            purity = (
                trace2["writer"][writer_layer] / norms["writer"][writer_layer].square().clamp_min(1.0e-300)
            ).numpy()
            add(
                "head_neuron", c2,
                np.full((h, n), writer_layer), np.broadcast_to(np.arange(h)[:, None], (h, n)),
                np.full((h, n), reader_layer), np.broadcast_to(np.arange(n)[None, :], (h, n)),
                np.full((h, n), reader_layer - writer_layer), np.ones((h, n), int),
                np.broadcast_to(np.array([[psd_rank(grams["V"][writer_layer, head]) for head in range(h)]]).T, (h, n)),
                np.ones((h, n)), np.broadcast_to(purity[:, None], (h, n)),
                np.broadcast_to(_rank_one_variance(purity, weights.d_model)[:, None], (h, n)),
            )

    # Neuron writer -> head reader, one exact small-factor calculation per channel.
    for reader_layer in range(1, weights.n_layers):
        for writer_layer in range(reader_layer):
            for channel, left, right, gram_name in (
                ("K", Q[reader_layer], K[reader_layer], "Q"),
                ("Q", K[reader_layer], Q[reader_layer], "K"),
                ("V", O[reader_layer], V[reader_layer], "O"),
            ):
                coordinates = torch.einsum("nd,hdi->hni", wout[writer_layer], right)
                c2 = torch.einsum(
                    "hni,hij,hnj->hn", coordinates, grams[gram_name][reader_layer].float().to(device), coordinates
                ) / norms[channel][reader_layer].float().to(device)[:, None].clamp_min(1.0e-30)
                h, n = c2.shape
                purity = (
                    trace2[channel][reader_layer] / norms[channel][reader_layer].square().clamp_min(1.0e-300)
                ).numpy()
                add(
                    f"neuron_head_{channel}", c2,
                    np.full((h, n), writer_layer), np.broadcast_to(np.arange(n)[None, :], (h, n)),
                    np.full((h, n), reader_layer), np.broadcast_to(np.arange(h)[:, None], (h, n)),
                    np.full((h, n), reader_layer - writer_layer),
                    np.broadcast_to(np.array([[psd_rank(grams[gram_name][reader_layer, head]) for head in range(h)]]).T, (h, n)),
                    np.ones((h, n), int), np.broadcast_to(purity[:, None], (h, n)), np.ones((h, n)),
                    np.broadcast_to(_rank_one_variance(purity, weights.d_model)[:, None], (h, n)),
                )

    # Interface -> neuron and neuron -> unembedding.  These dense quadratic
    # forms run on the configured accelerator and are evaluated for every neuron.
    interface_factors = {"emb_neuron": weights.W_E.T}
    if weights.W_pos is not None:
        interface_factors["pos_neuron"] = weights.W_pos.T
    for name, factor in interface_factors.items():
        factor_device = factor.float().to(device)
        gram_device = factor_device @ factor_device.T
        trace = float(torch.trace(gram_device))
        purity = float(torch.sum(gram_device * gram_device)) / max(trace * trace, 1.0e-300)
        rank = psd_rank(gram_device.double().cpu())
        for reader_layer in range(weights.n_layers):
            values = torch.einsum("nd,de,ne->n", win[reader_layer], gram_device, win[reader_layer]) / max(trace, 1.0e-30)
            n = len(values)
            add(
                name, values,
                np.full(n, -1), np.full(n, -1), np.full(n, reader_layer), np.arange(n),
                np.full(n, reader_layer + 1), np.ones(n, int), np.full(n, rank),
                np.ones(n), np.full(n, purity), np.full(n, _rank_one_variance(purity, weights.d_model)),
            )
        del gram_device, factor_device

    unembed = weights.W_U.float().to(device)
    unembed_gram = unembed @ unembed.T
    unembed_trace = float(torch.trace(unembed_gram))
    unembed_purity = float(torch.sum(unembed_gram * unembed_gram)) / max(unembed_trace**2, 1.0e-300)
    unembed_rank = psd_rank(unembed_gram.double().cpu())
    for writer_layer in range(weights.n_layers):
        values = torch.einsum("nd,de,ne->n", wout[writer_layer], unembed_gram, wout[writer_layer]) / max(unembed_trace, 1.0e-30)
        n = len(values)
        add(
            "neuron_unembed", values,
            np.full(n, writer_layer), np.arange(n), np.full(n, weights.n_layers), np.full(n, -1),
            np.full(n, weights.n_layers - writer_layer), np.full(n, unembed_rank), np.ones(n, int),
            np.full(n, unembed_purity), np.ones(n), np.full(n, _rank_one_variance(unembed_purity, weights.d_model)),
        )

    frames = []
    summaries = []
    channels = {
        "head_neuron": "neuron",
        "neuron_head_K": "K",
        "neuron_head_Q": "Q",
        "neuron_head_V": "V",
        "emb_neuron": "interface",
        "pos_neuron": "interface",
        "neuron_unembed": "interface",
    }
    for edge_class, pieces in pieces_by_class.items():
        if not pieces:
            continue
        frame, summary = _finish_mixed_class(
            weights, config, edge_class, channels[edge_class], pieces
        )
        frames.append(frame)
        summaries.append(summary)
    del win, wout, O, Q, K, V, unembed, unembed_gram
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame(), summaries


def _hist_quantile(counts: np.ndarray, edges: np.ndarray, q: float) -> float:
    cumulative = np.cumsum(counts, dtype=np.float64)
    if not len(cumulative) or cumulative[-1] == 0:
        return math.nan
    cumulative /= cumulative[-1]
    return float(np.interp(q, cumulative, edges[1:]))


def scan_neuron_wires(
    weights: ModelWeights, config: RunConfig
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    """Stream every causal neuron-neuron cosine and retain the strongest wires.

    Histograms summarize the complete class.  The parquet table is explicitly a
    bounded survivor table, preventing the billion-edge family from becoming an
    unreadable artifact.
    """

    device = config.device if torch.cuda.is_available() else "cpu"
    Wout = torch.nn.functional.normalize(weights.W_out.float(), dim=-1).to(device)
    Win = torch.nn.functional.normalize(weights.W_in.float(), dim=-1).to(device)
    bins = np.linspace(-1.0, 1.0, config.neuron_hist_bins + 1)
    hist = np.zeros((max(weights.n_layers - 1, 1), config.neuron_hist_bins), dtype=np.int64)
    top: list[tuple[float, int, int, int, int, float]] = []
    import heapq

    n_layer_pairs = max(weights.n_layers * (weights.n_layers - 1) // 2, 1)
    local_budget = max(
        256,
        int(math.ceil(3.0 * config.max_neuron_wires / n_layer_pairs)),
    )
    for wl in range(weights.n_layers):
        for rl in range(wl + 1, weights.n_layers):
            cosine = (Wout[wl] @ Win[rl].T).clamp(-1.0, 1.0).cpu().numpy()
            hist[rl - wl - 1] += np.histogram(cosine, bins=bins)[0]
            flat = cosine.ravel()
            local_n = min(local_budget, flat.size)
            if local_n:
                indices = np.argpartition(np.abs(flat), -local_n)[-local_n:]
                for index in indices:
                    value = float(flat[index])
                    entry = (
                        abs(value), wl, int(index // weights.d_mlp), rl,
                        int(index % weights.d_mlp), value,
                    )
                    if len(top) < config.max_neuron_wires:
                        heapq.heappush(top, entry)
                    elif entry[0] > top[0][0]:
                        heapq.heapreplace(top, entry)
            del cosine
    rows: list[dict[str, object]] = []
    for _, wl, wn, rl, rn, cosine in sorted(top, reverse=True):
        rows.append(
            {
                "model": weights.model_name,
                "writer_layer": wl,
                "writer": f"L{wl}N{wn}",
                "reader_layer": rl,
                "reader": f"L{rl}N{rn}",
                "channel": "neuron",
                "layer_span": rl - wl,
                "C": abs(cosine),
                "C2": cosine * cosine,
                "signed_cosine": cosine,
                "signed_scalar_if_rank_one": cosine,
                "stored_scope": f"top_{config.max_neuron_wires}_by_absolute_cosine",
            }
        )
    summaries: list[dict[str, object]] = []
    for index in range(weights.n_layers - 1):
        counts = hist[index]
        q25 = _hist_quantile(counts, bins, 0.25)
        med = _hist_quantile(counts, bins, 0.5)
        q75 = _hist_quantile(counts, bins, 0.75)
        sd = (q75 - q25) / 1.349
        centers = 0.5 * (bins[:-1] + bins[1:])
        exceed = int(counts[np.abs(centers) >= 0.2].sum())
        n_total = int(counts.sum())
        beta_tail = float(
            1.0
            - sps.beta.cdf(0.2**2, 0.5, (weights.d_model - 1) / 2.0)
        )
        summaries.append(
            {
                "model": weights.model_name,
                "edge_class": "neuron_neuron",
                "layer_span": index + 1,
                "n_candidates": n_total,
                "median_signed_cosine": med,
                "robust_scale": sd,
                "isotropic_scale": 1.0 / math.sqrt(weights.d_model),
                "widening": sd * math.sqrt(weights.d_model),
                "observed_abs_cos_ge_0_2": exceed,
                "beta_expected_abs_cos_ge_0_2": n_total * beta_tail,
            }
        )
    del Wout, Win
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return pd.DataFrame(rows), summaries


def _family_summary(edges: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for edge_class, group in edges.groupby("edge_class"):
        rows.append(
            {
                "model": str(group["model"].iloc[0]),
                "edge_class": edge_class,
                "n_candidates": len(group),
                "median_C": float(group["C"].median()),
                "median_theoretical_z": float(group["theoretical_z"].median()),
                "above_z2_fraction": float((group["theoretical_z"] >= 2).mean()),
                "below_z_minus2_fraction": float((group["theoretical_z"] <= -2).mean()),
                "n_selected": int(group.get("selected", pd.Series(False, index=group.index)).sum()),
                "n_avoidant": int(group.get("avoidant", pd.Series(False, index=group.index)).sum()),
            }
        )
    return pd.DataFrame(rows)


def run_wang_map(weights: ModelWeights, config: RunConfig) -> MapResult:
    head = add_empirical_selection(compute_head_edges(weights), config.fdr_q)
    cross = _dense_crosschecks(
        weights, head, config.dense_crosscheck_edges, config.seed
    )
    interface = _interface_gram_rows(weights, head)
    mixed, mixed_families = scan_mixed_edges(weights, config)
    wires, wire_families = scan_neuron_wires(weights, config)
    edges = pd.concat([head, interface, mixed], ignore_index=True, sort=False)
    families = pd.concat(
        [
            _family_summary(pd.concat([head, interface], ignore_index=True, sort=False)),
            pd.DataFrame(mixed_families),
            pd.DataFrame(wire_families),
        ],
        ignore_index=True,
        sort=False,
    )
    residual = float(cross["absolute_residual"].max()) if len(cross) else math.nan
    position_note = (
        "learned-position mixed and interface classes were included"
        if weights.W_pos is not None
        else "learned-position classes are not constructible for this rotary model"
    )
    observations = [
        f"Dense and factored head couplings agreed to a maximum absolute residual of {residual:.3e}.",
        f"The empirical head graph selected {int(head['selected'].sum())} high-coupling edges and identified {int(head['avoidant'].sum())} avoidant edges.",
        f"All {len(mixed_families)} constructible mixed classes were fully scanned and retained {len(mixed)} bounded tail records; {position_note}.",
        f"The full neuron-neuron scan retained {len(wires)} strongest signed wires; the table is explicitly bounded while span histograms describe every candidate.",
    ]
    return MapResult(
        edges=edges,
        neuron_wires=wires,
        families=families,
        dense_crosschecks=cross,
        observations=observations,
    )
