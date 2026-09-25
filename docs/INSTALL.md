# Installation

Run every command from the repository root. Tested on x86_64 Linux (Ubuntu 20.04 under WSL2).

| task | Python environment | also needed |
|---|---|---|
| inference | `uv sync --extra cpu` | the [weights](#download-the-weights-and-the-example-volume) |
| training | `uv sync --extra cu118` | an NVIDIA GPU with a driver for CUDA 11.8 or newer, and the corpora |
| data generation | `uv sync` | g++ with C++17 and OpenMP, and a [MatViz3D clone](#set-up-data-generation) |

## Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

or `pip install uv`.

## Install the Python environment

```bash
uv sync --extra cpu
source .venv/bin/activate
```

This creates `.venv/` with Python 3.12 and every version from `uv.lock`, and installs `piml` in editable mode (under 300 MB to download). Use `--extra cu118` for the GPU build (about 2.8 GB; no CUDA toolkit needed) and a plain `uv sync` for data generation, which does not need torch. The two extras exclude each other; repeat the extra on every later `uv sync`, because a plain `uv sync` removes torch. The `dev` extra adds ruff and pytest.

Python 3.13 is not supported, because torch 2.3.1 has no wheels for it. If the machine has no Python 3.12, run `uv python install 3.12`.

Without uv's lockfile: `uv venv && uv pip install -e ".[cpu]"`. Keep the `-e`: the default paths are relative to the repository root, which `piml` finds from its own location.

To check a GPU install:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

It should print `2.3.1+cu118 True`.

## Download the weights and the example volume

The [weights folder](https://drive.google.com/drive/folders/1FDNxVCaYF80uvH8Wv0MCziz28GSd__Eu) holds the 34 checkpoints (81 MB), and the [example folder](https://drive.google.com/drive/folders/1RESub45cO6zhZbLnvxgWqiB0FYih34LK) one copper test volume, `sve_1160.hdf5` (5 MB). Keep the file names:

```
weights/<name>.pt
data/example/sve_1160.hdf5
```

Check that inference runs:

```bash
python -m piml.infer --ckpt weights/cu-nonlinear-rf13-w64-s0.pt --input data/example/sve_1160.hdf5
```

It should end with the relative error of the prediction against the solver's A stored in the example file.

## Set up data generation

Needed only to regenerate the corpora. The solver driver compiles against one header of MatViz3D, `fft_homog.hpp`, which exists only on the `qml-development` branch. Clone it at the pinned commit:

```bash
git clone --filter=blob:none --branch qml-development https://github.com/MME-NTU-KhPI/MatViz3D.git MatViz3D && git -C MatViz3D checkout 8c4ee49c7d8f16f8cb54ad9612cecd3db67cb894
```

Keep the clone after the build: `piml.corpus gen` hashes the header into every file it writes. [tools/README.md](../tools/README.md) covers the build and the rest.

## Paths

Every default path is relative to the repository root and has an environment variable that overrides it:

| variable | default | holds |
|---|---|---|
| `PIML_ROOT` | repository root | base of every default below |
| `PIML_DATA` | `data/` | corpora, under `periodic32/<corpus>/` |
| `PIML_SCRATCH` | `scratch/` | caches and solver working directories |
| `PIML_LOGS` | `logs/` | the run log and training traces |
| `PIML_CHECKPOINTS` | `checkpoints/` | weights written by training |
| `PIML_MATVIZ3D` | `MatViz3D/` | the MatViz3D clone |
| `PIML_MESH_REFINE` | `build/mesh_refine` | the compiled solver driver |

`tools/build_mesh_refine.sh` reads the last two as well.

## Troubleshooting

- `MatViz3D not found at .../MatViz3D`: the clone is missing or on the default branch, which lacks `fft_homog.hpp`. Run `git -C MatViz3D checkout 8c4ee49c7d8f16f8cb54ad9612cecd3db67cb894`, or set `PIML_MATVIZ3D`.
- `ModuleNotFoundError: No module named 'torch'`: a plain `uv sync` removed torch. Run `uv sync --extra cpu`.
- `FileNotFoundError: ... .venv/lib/python3.12/build/mesh_refine`: `piml` was installed without `-e`. Reinstall with `-e`, or `export PIML_ROOT="$PWD"`.
- `ImportError: attempted relative import with no known parent package`: a module was run by path. Run it as `python -m piml.<module>`.
