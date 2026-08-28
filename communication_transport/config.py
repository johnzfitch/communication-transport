from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable


DEFAULT_MODELS = (
    "EleutherAI/pythia-70m",
    "EleutherAI/pythia-160m",
    "gpt2",
)

ALL_EXPERIMENTS = (
    "scalar_map",
    "operator_core",
    "interventions",
    "thomas_wigner",
    "layer_transport",
    "graph_connection",
    "role_complete_qk",
    "rw_pca",
    "hessian_typing",
    "lie_identification",
    "rope_transport",
    "ternary_synergy",
    "realized_traffic",
    "exceptional_branch",
)


@dataclass(slots=True)
class RunConfig:
    models: tuple[str, ...] = DEFAULT_MODELS
    experiments: tuple[str, ...] = ALL_EXPERIMENTS
    output: Path = Path("outputs/full_run")
    device: str = "cuda"
    seed: int = 0
    offline: bool = True
    inference_dtype: str = "float32"

    # The standard profile matches the requested compact corpora.
    n_sequences: int = 32
    induction_block: int = 128
    natural_length: int = 256
    position_length: int = 128
    batch_size: int = 8
    transport_train_fraction: float = 0.75

    fdr_q: float = 0.05
    support_rtol: float = 1.0e-9
    ridge_grid: tuple[float, ...] = (1.0e-9, 1.0e-7, 1.0e-5)
    transport_ridge_grid: tuple[float, ...] = (1.0e-6, 1.0e-4, 1.0e-2)
    dense_crosscheck_edges: int = 12
    max_operator_edges_per_channel: int = 96
    max_triangles_per_model: int = 200
    max_synergy_triangles_per_model: int = 150
    max_realized_edges_per_model: int = 12
    max_mixed_edges_per_class: int = 5_000
    max_neuron_wires: int = 100_000
    neuron_hist_bins: int = 2001
    surrogate_draws: int = 100
    # The full-pipeline triangle surrogate rebuilds the map per draw, so it
    # carries its own budget.
    triangle_surrogate_draws: int = 12
    max_surrogate_triangles_per_draw: int = 48
    max_matched_control_triangles: int = 200
    transport_control_draws: int = 3
    bootstrap_draws: int = 24
    lie_ambient_dims: tuple[int, ...] = (8, 12, 16, 20, 24, 26, 32, 40, 52)
    lie_generator_dims: tuple[int, ...] = (3, 6, 8, 10, 15, 21, 24, 28, 36, 52)
    candidate_carrier: Path | None = None
    source_spec: str = (
        r"C:\Users\johnz\Downloads\transformer_communication_transport_experiment(3).md"
        r" + C:\Users\johnz\Downloads\message_to_agent.md"
    )

    @classmethod
    def debug(
        cls,
        *,
        models: Iterable[str] | None = None,
        output: Path | str = Path("outputs/debug_run"),
        experiments: Iterable[str] | None = None,
    ) -> "RunConfig":
        """A runnable smoke profile; formulas and output schemas are unchanged."""

        return cls(
            models=tuple(models or ("EleutherAI/pythia-70m",)),
            experiments=tuple(experiments or ALL_EXPERIMENTS),
            output=Path(output),
            n_sequences=4,
            induction_block=32,
            natural_length=64,
            position_length=64,
            batch_size=2,
            dense_crosscheck_edges=4,
            max_operator_edges_per_channel=16,
            max_triangles_per_model=6,
            max_synergy_triangles_per_model=2,
            max_realized_edges_per_model=3,
            max_mixed_edges_per_class=250,
            max_neuron_wires=2_000,
            neuron_hist_bins=501,
            surrogate_draws=4,
            triangle_surrogate_draws=2,
            max_surrogate_triangles_per_draw=4,
            max_matched_control_triangles=6,
            transport_control_draws=2,
            bootstrap_draws=4,
            lie_ambient_dims=(8, 12, 16, 20, 26),
            lie_generator_dims=(3, 6, 8, 10, 15),
        )

    def normalized_experiments(self) -> tuple[str, ...]:
        if self.experiments == ("all",) or "all" in self.experiments:
            return ALL_EXPERIMENTS
        unknown = sorted(set(self.experiments) - set(ALL_EXPERIMENTS))
        if unknown:
            raise ValueError(f"unknown experiments: {', '.join(unknown)}")
        return self.experiments

    def as_json(self) -> dict[str, object]:
        out = asdict(self)
        out["output"] = str(self.output)
        out["candidate_carrier"] = (
            str(self.candidate_carrier) if self.candidate_carrier else None
        )
        return out

    def write(self) -> None:
        self.output.mkdir(parents=True, exist_ok=True)
        (self.output / "config.json").write_text(
            json.dumps(self.as_json(), indent=2), encoding="utf-8"
        )


def slugify_model(name: str) -> str:
    return name.split("/")[-1].replace("_", "-").lower()

