# GPU Development Notes

This branch is a GPU development branch for ESAT NMF acceleration. It is not a
clean upstream-ready patch yet.

## Active Branch Policy

- Use `gpu-study-cuda-snapshot` as the remote synchronization branch for CUDA
  testing on `gb10`.
- Use normal fast-forward updates after the remote clone is initialized:

```bash
git pull --ff-only
```

- Avoid force-pushing this branch unless the remote working tree is explicitly
  reset afterward.
- Keep experimental local branches separate from the remote CUDA validation
  branch.

## Build Commands

CUDA build on GB10:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install maturin pytest numpy
maturin develop --release --features cuda
```

Metal build on Apple Silicon:

```bash
source .venv/bin/activate
maturin develop --release --features metal
```

CPU-only fallback build:

```bash
source .venv/bin/activate
maturin develop --release
```

## Backend Verification

Do not infer GPU execution from `use_gpu=True` or passing tests alone.
`ls_nmf_batched` returns a `backend` field. Check it when validating GPU paths.

Current limitation: the value is `"gpu"` or `"cpu"` and does not yet distinguish
`"cuda"` from `"metal"`.

## Current GPU Scope

Implemented GPU-batched paths:

- `BatchSA(use_gpu=True)` for LS-NMF
- `Bootstrap(use_gpu=True)`
- `Displacement(use_gpu=True)`
- `BSDISP(use_gpu=True)` through Bootstrap and Displacement

Known limitations:

- WS-NMF is not batched on GPU.
- `ls_nmf_batched` always runs the full `max_iter`; convergence tracking is not
  equivalent to the original Python path.
- Some metadata fields are still thinner than the original `SA.train` path.
- DISP still spends time in Python-side search logic.

## Candle Patch

This repo currently uses a local `[patch.crates-io]` override:

```toml
candle-core = { path = "vendor/candle-core-0.11.0" }
```

The immediate reason is Candle 0.11 Metal initialization safety. The vendored
copy keeps the branch buildable, but it creates a large diff and should not be
treated as the long-term dependency strategy.

Longer-term options:

- switch to an upstream Candle release if the issue is fixed;
- use a small fork with only the required patch;
- submit the Metal initialization fix upstream.

## Repository Hygiene

Do not commit:

- Jupyter `.ipynb_checkpoints/`
- notebook execution outputs unless they are intentional artifacts
- Cargo `target/` directories
- virtual environments
- benchmark logs and local profiling outputs

Large benchmark outputs should be stored outside git, or in a dedicated artifact
store if they need to be shared.
