# Transformer communication transport

This package implements the runnable experiment described in
`transformer_communication_transport_experiment(2).md`. It keeps weight-only,
activation-side, causal, carrier-agnostic Lie, and explicitly marked exceptional
analyses separate while writing their raw tables and arrays into one run folder.

Run the full cached-model experiment from the repository root with:

```powershell
python -m communication_transport.run `
  --models EleutherAI/pythia-70m EleutherAI/pythia-160m gpt2 `
  --experiments all `
  --output reruns/full_run
```

The `exceptional_branch` experiment imports `engine.readers` from the companion
Albert engine source tree, which must be available on `PYTHONPATH` for a full
exceptional result. If it is absent, that branch records the import error while
independent experiments continue. The remaining third-party package versions
from the completed run are recorded in
[`runtime-environment.txt`](../runtime-environment.txt).

Use `--profile debug` for the same formulas with smaller corpora and artifact
caps. Individual experiment names may be space-separated or comma-separated.
The runner records stage exceptions in `errors.json` and continues independent
computations. `report.md` is regenerated from the saved Parquet and NPZ files.

Run the package checks with:

```powershell
python -m pytest communication_transport\tests -q -s
```
