# Convolutional stress localization in 3D polycrystals

A convolutional network reads the elastic stiffness $\mathbf{C}(\mathbf{x})$ of every voxel of a periodic cubic polycrystal and predicts its stress-localization tensor $\mathbf{A}(\mathbf{x})$, a $6\times6$ matrix per voxel that gives the local stress under any macroscopic strain $\bar{\mathbf{E}}$:

$$\boldsymbol{\sigma}(\mathbf{x}) = \mathbf{A}(\mathbf{x}) : \bar{\mathbf{E}}$$

One prediction takes a fraction of a second on a CPU and replaces six FFT solves.

The repository holds the code, 34 trained checkpoints and the tools that regenerate the training data. The volumes are periodic cubes of $32^3$ voxels with 1,638 grains. The three materials share $c_{11}$ and $c_{12}$ and differ in $c_{44}$, which sets the anisotropy through the Zener ratio $2c_{44}/(c_{11}-c_{12})$: copper at 3.21 and two synthetic crystals at 5 and 8. For each material there is a linear network (no activation) and a nonlinear one (LeakyReLU), both with a receptive field of 13 voxels, each trained plain and through an equilibrium projection (`-proj`), a Fourier-space step that makes the predicted stress satisfy equilibrium.

## Layout

```
src/piml/   the Python package; each tool runs as python -m piml.<module>
tools/      the solver driver, its build script and the corpus regeneration guide
docs/       installation, inference, data format
```

Git ignores `weights/` and `data/`, which you download, and `MatViz3D/`, `build/`, `scratch/`, `logs/` and `checkpoints/`, which data generation and training create.

## Quick start

Install [uv](https://docs.astral.sh/uv/), then the pinned environment (Python 3.12, torch 2.3.1, CPU build):

```bash
git clone https://github.com/Danylo-Krasii/PIML-MS.git
cd PIML-MS
uv sync --extra cpu
source .venv/bin/activate
```

Download the 34 checkpoints (81 MB) from the [weights folder](https://drive.google.com/drive/folders/1FDNxVCaYF80uvH8Wv0MCziz28GSd__Eu) into `weights/`, and `sve_1160.hdf5` (5 MB) from the [example folder](https://drive.google.com/drive/folders/1RESub45cO6zhZbLnvxgWqiB0FYih34LK) into `data/example/`. Then predict $\mathbf{A}(\mathbf{x})$ for the example volume:

```bash
python -m piml.infer --ckpt weights/cu-nonlinear-rf13-w64-s0-proj.pt --input data/example/sve_1160.hdf5
```

It prints the shape of $\mathbf{A}$, whether the projection ran and, since the example file carries the solver's $\mathbf{A}$, the relative error against it. To apply 100 MPa of uniaxial stress along x and plot the stress:

```bash
python -m piml.infer --ckpt weights/cu-nonlinear-rf13-w64-s0-proj.pt --input data/example/sve_1160.hdf5 --load uniaxial-stress-x 100e6 --plot scratch/sve_1160
```

[docs/INFERENCE.md](docs/INFERENCE.md) covers loads, plots, your own volumes and the Python API. [docs/INSTALL.md](docs/INSTALL.md) covers GPU and data-generation setup, [docs/DATA.md](docs/DATA.md) the file formats, and [tools/README.md](tools/README.md) corpus regeneration.

## Checkpoints

Names follow `<corpus>-<arm>-rf13-w64-s<seed>[-proj].pt`, with corpus `cu`, `z5` or `z8`, arm `linear` or `nonlinear`, and seeds 0 to 2 (0 and 1 for the plain linear arm on `z5` and `z8`). Every network has 64 channels per layer and 592,764 parameters (`LocalizationNet` in `src/piml/network.py`).

- Use the checkpoint of your volume's material. The file does not record it, and a mismatch raises no error.
- A `-proj` checkpoint's raw output is not its prediction. `piml.infer` applies the projection by default; `piml.score` needs `--project eq`.
- The checkpoints were trained on $32^3$ volumes with about 20 voxels per grain; use volumes like these.

## Training

Training needs an NVIDIA GPU and the corpus in `data/periodic32/<corpus>/`, regenerated with [tools/README.md](tools/README.md). The released checkpoints used, per arm (`--linear` for the linear one, `--project-train` for `-proj`, `--which z5` or `z8` for the other corpora):

```bash
python -m piml.periodic sweep --which cu --rf 13 --width 64 --augment --norm none --max-steps 36000 --patience 6000 --sched-patience 20 --seeds 0 1 2
```

The defaults of these flags differ from the recipe, so pass all of them. Each run appends a row to `logs/results.jsonl` and saves its best-validation weights under `checkpoints/`. The first run of a corpus writes a 1.49 GB cache under `scratch/`. A nonlinear run took about 135 min on an RTX 4070 Laptop GPU (8 GB). cuDNN is nondeterministic, so reruns do not reproduce bitwise. [tools/queues/train_example.sh](tools/queues/train_example.sh) queues runs in the background.

## Data

Each corpus holds 200 volumes, one HDF5 file each, 3.02 GB for all three. They share seeds 1000 to 1199, so grain shapes and orientations are identical across materials. Seeds 1000 to 1139 are for training, 1140 to 1159 for validation and 1160 to 1199 for testing. The labels come from six solves per volume with the Moulinec-Suquet FFT solver of [MatViz3D](https://github.com/MME-NTU-KhPI/MatViz3D) at tolerance $10^{-5}$.

The corpora have no public copy yet. [tools/README.md](tools/README.md) regenerates them in about 1 to 2 hours per corpus on 12 cores.

## Modules

| module | purpose |
|---|---|
| `infer` | predict $\mathbf{A}(\mathbf{x})$ for one volume, apply a load, plot, export VTK |
| `loading`, `fields`, `viz` | loads, stress measures and plots used by `infer` |
| `network`, `microstructure`, `physics` | the network, crystal stiffness and Mandel conversions, the equilibrium projection |
| `periodic` | training (`sweep`) and the ridge baseline (`ridge`) |
| `score` | test error of a checkpoint |
| `affine_lsqr`, `born` | untrained baselines: the best linear kernel and the solver's iterates |
| `corpus`, `mesh_convergence` | corpus generation and checks, solver smoke test and resolution ladder |
| `paths` | default paths, each overridable by an environment variable (`PIML_DATA`, `PIML_SCRATCH` and others) |


## License

MIT, see [LICENSE](LICENSE).
