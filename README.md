# Communication Transport experiment outputs

This repository contains the complete output of the communication-transport experiments run on 2026-08-28 for:

- `EleutherAI/pythia-70m`
- `EleutherAI/pythia-160m`
- `gpt2`

The run completed without recorded execution errors (13/13 package tests pass). This run includes the full null-and-control battery: five algebra surrogate families with matched random-plane baselines at every stratum, full-pipeline spectrum-preserving triangle surrogates, identity/token-permuted/bootstrap/outlier layer-transport controls, an aggregate Thomas-Wigner arm on pooled states with genuine common support, a 150-triple co-ablation census with grouped splits, task-relevant survival projectors, per-role intervention labels, and an automatic external-anchors table.

Start with [report2.md](report2.md) for the full written results, or [report.md](report.md) for the machine-generated section summary and the Wang external-anchors table. The Parquet tables carry the row-level data and the NPZ files the numerical arrays.

Large NPZ artifacts are stored with Git LFS. Install Git LFS before cloning:

```powershell
git lfs install
git clone https://github.com/johnzfitch/communication-transport.git
```

The exact run settings are in [config.json](config.json), and [observations.json](observations.json) provides the machine-readable observation summary.

## Experiment runtime

The Python source used for the run is included in
[`communication_transport/`](communication_transport/). The captured Python and
package versions are in [`runtime-environment.txt`](runtime-environment.txt).
Generated Python bytecode and test caches are intentionally excluded.

## Debug profile run

[`debug_run2/`](debug_run2/) holds the small debug-profile smoke run of the same
pipeline (pythia-70m only, reduced corpora and caps, identical formulas and
output schema), kept for quick schema inspection without the full artifacts.
