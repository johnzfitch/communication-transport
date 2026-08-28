from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
import torch

from communication_transport.config import RunConfig
from communication_transport.interventions import flatten_ov_factors
from communication_transport.lie_identification import closure_defect, fit_candidate_conjugation, so_basis
from communication_transport.model_io import mean_ablation_hooks
from communication_transport.operator_core import (
    _orthogonal_polar,
    _rotation_angle_2d,
    build_operator_cores,
    partial_polar,
)
from communication_transport.rope_transport import rope_matrix
from communication_transport.ternary_synergy import _synthetic_control
from communication_transport.wang_map import (
    _null_fields,
    add_empirical_selection,
    compute_head_edges,
    run_wang_map,
)
from communication_transport.weights import ModelWeights


def observed_close(label: str, left, right, tolerance: float) -> None:
    residual = float(np.linalg.norm(np.asarray(left) - np.asarray(right)))
    print(f"Observed: {label} residual={residual:.3e}")
    assert residual <= tolerance


@pytest.fixture(scope="module")
def tiny_weights() -> ModelWeights:
    generator = torch.Generator(device="cpu").manual_seed(71)
    L, H, d, dh, mlp, vocab = 2, 2, 6, 2, 4, 11

    def random(*shape):
        return torch.randn(*shape, generator=generator, dtype=torch.float64) / math.sqrt(d)

    return ModelWeights(
        model_name="tiny",
        Q=random(L, H, d, dh),
        K=random(L, H, d, dh),
        V=random(L, H, d, dh),
        O=random(L, H, d, dh),
        W_in=random(L, mlp, d),
        W_out=random(L, mlp, d),
        W_E=random(vocab, d),
        W_pos=random(9, d),
        W_U=random(d, vocab),
        ln1_w=torch.ones(L, d, dtype=torch.float64),
        ln2_w=torch.ones(L, d, dtype=torch.float64),
        n_layers=L,
        n_heads=H,
        d_model=d,
        d_head=dh,
        d_mlp=mlp,
        parallel_attn_mlp=False,
        positional_scheme="standard",
        rotary_dim=0,
        rotary_base=10_000.0,
    )


def tiny_config(tmp_path) -> RunConfig:
    config = RunConfig.debug(output=tmp_path)
    config.device = "cpu"
    config.max_neuron_wires = 20
    config.max_mixed_edges_per_class = 20
    config.neuron_hist_bins = 101
    return config


def test_dense_and_factored_head_coupling_agree(tiny_weights: ModelWeights) -> None:
    edges = add_empirical_selection(compute_head_edges(tiny_weights), 0.05)
    for row in edges.sample(6, random_state=3).itertuples():
        writer = tiny_weights.ov(row.writer_layer, int(row.writer.split("H")[1]))
        reader_head = int(row.reader.split("H")[1])
        if row.channel == "K":
            reader = tiny_weights.qk(row.reader_layer, reader_head)
        elif row.channel == "Q":
            reader = tiny_weights.qk(row.reader_layer, reader_head).T
        else:
            reader = tiny_weights.ov(row.reader_layer, reader_head)
        dense = float(
            torch.linalg.matrix_norm(reader @ writer)
            / (torch.linalg.matrix_norm(reader) * torch.linalg.matrix_norm(writer))
        )
        observed_close(f"dense/factored {row.channel}", dense, row.C, 2.0e-12)


def test_theoretical_haar_variance_matches_monte_carlo() -> None:
    rng = np.random.default_rng(44)
    d = 8
    G = np.diag(np.linspace(0.2, 2.1, d))
    H = np.diag(np.linspace(0.4, 1.7, d) ** 2)
    trg, trh = np.trace(G), np.trace(H)
    trg2, trh2 = np.trace(G @ G), np.trace(H @ H)
    _, predicted, _ = _null_fields(0.0, trg, trh, trg2, trh2, d)
    values = []
    for _ in range(1800):
        q, r = np.linalg.qr(rng.normal(size=(d, d)))
        q *= np.sign(np.diag(r))[None, :]
        values.append(np.trace(G @ q @ H @ q.T) / (trg * trh))
    observed = float(np.var(values, ddof=0))
    relative = abs(observed - predicted) / predicted
    print(f"Observed: Haar variance relative residual={relative:.3e}")
    assert relative < 0.14


def test_eighteen_map_classes_and_signed_neuron_identity(
    tiny_weights: ModelWeights, tmp_path
) -> None:
    result = run_wang_map(tiny_weights, tiny_config(tmp_path))
    classes = set(result.families["edge_class"])
    expected = {
        "head_head_K", "head_head_Q", "head_head_V",
        "emb_head_K", "emb_head_Q", "emb_head_V",
        "pos_head_K", "pos_head_Q", "pos_head_V",
        "head_unembed", "head_neuron",
        "neuron_head_K", "neuron_head_Q", "neuron_head_V",
        "emb_neuron", "pos_neuron", "neuron_unembed", "neuron_neuron",
    }
    print(f"Observed: map exposed {len(classes)} edge classes")
    assert expected == classes
    residual = np.max(np.abs(result.neuron_wires["C"] - result.neuron_wires["signed_cosine"].abs()))
    observed_close("absolute signed neuron cosine equals C", residual, 0.0, 1.0e-14)


def test_partial_polar_and_operator_support_projectors(
    tiny_weights: ModelWeights, tmp_path
) -> None:
    generator = torch.Generator().manual_seed(9)
    matrix = torch.randn(7, 5, generator=generator, dtype=torch.float64)
    polar, positive, _, _, _, residual = partial_polar(matrix, 1.0e-12)
    print(f"Observed: partial polar reconstruction residual={residual:.3e}")
    assert residual < 2.0e-12
    observed_close("polar reconstruction", polar @ positive, matrix, 2.0e-12)

    edges = add_empirical_selection(compute_head_edges(tiny_weights), 0.05)
    edges["selected"] = True
    edges["avoidant"] = False
    config = tiny_config(tmp_path)
    records, arrays, _ = build_operator_cores(tiny_weights, edges, config)
    assert len(records)
    for key in ("support_projector_left", "support_projector_right"):
        for projector in arrays[key]:
            observed_close(f"{key} idempotence", projector @ projector, projector, 2.0e-11)


def test_thomas_wigner_formula_and_order_reversal() -> None:
    t, s, delta = 0.7, 0.45, 0.31
    rotation = np.array(
        [[math.cos(delta), -math.sin(delta)], [math.sin(delta), math.cos(delta)]]
    )
    A = np.diag([math.exp(t / 2), math.exp(-t / 2)])
    B = rotation @ np.diag([math.exp(s / 2), math.exp(-s / 2)]) @ rotation.T
    qab, _ = _orthogonal_polar(A @ B)
    qba, _ = _orthogonal_polar(B @ A)
    observed = _rotation_angle_2d(qab)
    reversed_angle = _rotation_angle_2d(qba)
    p, q = math.tanh(t / 2), math.tanh(s / 2)
    predicted = math.atan2(
        -math.sin(2 * delta) * p * q,
        1 + math.cos(2 * delta) * p * q,
    )
    observed_close("Thomas-Wigner pair angle", observed, predicted, 2.0e-12)
    observed_close("positive-factor order reversal", observed, -reversed_angle, 2.0e-12)


def test_intervention_identities_and_stored_mean() -> None:
    generator = torch.Generator().manual_seed(19)
    O = torch.randn(9, 3, generator=generator, dtype=torch.float64)
    V = torch.randn(9, 3, generator=generator, dtype=torch.float64)
    observed_close("sign flip Gram", (-O).T @ (-O), O.T @ O, 1.0e-13)
    O_flat, V_flat, info = flatten_ov_factors(O, V)
    observed_close(
        "spectrum flatten Frobenius norm",
        torch.linalg.matrix_norm(O_flat @ V_flat.T),
        torch.linalg.matrix_norm(O @ V.T),
        2.0e-11,
    )
    assert info["support_rank"] == 3

    means = {0: torch.randn(2, 3, generator=generator)}
    z = torch.randn(4, 5, 2, 3, generator=generator)
    hook = mean_ablation_hooks(["L0H1"], means)[0][1]
    changed = hook(z.clone(), None)
    expected = means[0][1].expand(4, 5, 3)
    observed_close("mean-ablation stored contribution", changed[:, :, 1], expected, 1.0e-7)


def test_rope_is_flat_on_closed_position_loops(tiny_weights: ModelWeights) -> None:
    rotary = tiny_weights
    rotary.rotary_dim = 2
    rotary.positional_scheme = "rotary"
    r01 = rope_matrix(rotary, 7)
    r12 = rope_matrix(rotary, 11)
    r02 = rope_matrix(rotary, 18)
    observed_close("exact RoPE loop", r12 @ r01 @ np.linalg.inv(r02), np.eye(rotary.d_head), 2.0e-12)
    rotary.rotary_dim = 0
    rotary.positional_scheme = "standard"


def test_exact_lie_control_closes_and_candidate_fit_recovers() -> None:
    basis = so_basis(3)
    defect, energy, _ = closure_defect(basis)
    print(f"Observed: exact so(3) closure defect={defect:.3e}, energy={energy:.3e}")
    assert defect < 1.0e-28
    fitted, starts = fit_candidate_conjugation(
        basis, basis.copy(), seed=5, n_initializations=1, steps=4
    )
    print(f"Observed: exact candidate chordal distance={fitted:.3e}")
    assert fitted < 1.0e-10
    assert starts


def test_synthetic_ternary_positive_control_detects_increment() -> None:
    frame = pd.DataFrame(_synthetic_control(3))
    pair = float(frame.loc[frame["feature_set"] == "pairwise", "held_out_r2"].iloc[0])
    full = float(frame.loc[frame["feature_set"] == "pairwise_plus_known_ternary", "held_out_r2"].iloc[0])
    print(f"Observed: synthetic ternary incremental held-out R2={full - pair:.3f}")
    assert full > 0.98
    assert full - pair > 0.25
