# Transformer communication transport

This package implements the runnable experiment described in
`transformer_communication_transport_experiment(3).md`, with the corrections
and added null/control arms from the external review work order
(`message_to_agent.md`). It keeps weight-only, activation-side, causal,
carrier-agnostic Lie, and explicitly marked exceptional analyses separate
while writing their raw tables and arrays into one run folder.

Run the full cached-model experiment from the repository root with:

```powershell
R:\transporting-lab\.venv314\Scripts\python.exe -m communication_transport.run `
  --models EleutherAI/pythia-70m EleutherAI/pythia-160m gpt2 `
  --experiments all `
  --output outputs/full_run2
```

`outputs/full_run` is the superseded 2026-08-27 run (its behavioral induction
labels carry the offset bug documented in its report); `outputs/full_run2` is
the corrected rerun.

Use `--profile debug` for the same formulas with smaller corpora and artifact
caps. Individual experiment names may be space-separated or comma-separated.
The runner records stage exceptions in `errors.json` and continues independent
computations. `report.md` is regenerated from the saved Parquet and NPZ files.

Run the package checks with:

```powershell
R:\transporting-lab\.venv314\Scripts\python.exe -m pytest communication_transport\tests -q -s
```
