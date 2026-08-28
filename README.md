# Communication Transport experiment outputs

This repository contains the complete output of the communication-transport experiments run on 2026-08-27 for:

- `EleutherAI/pythia-70m`
- `EleutherAI/pythia-160m`
- `gpt2`

The run completed without recorded execution errors. Start with [report.md](report.md) for the observed results, then use the Parquet tables for row-level analyses and the NPZ files for numerical arrays.

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
