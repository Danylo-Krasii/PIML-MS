# Convolutional stress localization in 3D polycrystals

Predicts the stress-localization tensor $\mathbf{A}(\mathbf{x})$ of a periodic cubic polycrystal from the stiffness $\mathbf{C}(\mathbf{x})$ of its voxels, so the local stress under any macroscopic strain $\bar{\mathbf{E}}$ comes from one forward pass instead of an FFT solve:

$$\boldsymbol{\sigma}(\mathbf{x}) = \mathbf{A}(\mathbf{x}) : \bar{\mathbf{E}}$$

Includes 34 trained checkpoints for $32^3$ volumes of three cubic materials, the training code and the tools that regenerate the training data.

## Installation

See [docs/INSTALL.md](docs/INSTALL.md). In short:

```bash
uv sync --extra cpu          # --extra cu118 for training on an NVIDIA GPU
source .venv/bin/activate
```

Download the checkpoints from the [weights folder](https://drive.google.com/drive/folders/1FDNxVCaYF80uvH8Wv0MCziz28GSd__Eu) into `weights/` and the example volume from the [example folder](https://drive.google.com/drive/folders/1RESub45cO6zhZbLnvxgWqiB0FYih34LK) into `data/example/`.

## Running

> Every tool runs from the repository root as `python -m piml.<module>`; `--help` lists its arguments.

### Inference

```bash
python -m piml.infer --ckpt weights/cu-nonlinear-rf13-w64-s0-proj.pt --input data/example/sve_1160.hdf5 --load uniaxial-stress-x 100e6 --plot scratch/sve_1160
```

Checkpoints are named `<corpus>-<arm>-rf13-w64-s<seed>[-proj].pt`: corpus `cu` (copper), `z5` or `z8` (Zener ratio 5 and 8), arm `linear` or `nonlinear`, and `-proj` when trained through the equilibrium projection. Use the corpus of your volume's material; a nonlinear `-proj` checkpoint is the default choice. The checkpoints expect volumes like the training ones: $32^3$ voxels, about 20 voxels per grain.

Optional arguments:

```
--input FILE             Corpus HDF5 or .npz with C66, or voxels and C_grain (Pa, Mandel notation)
--generate SEED          Build a random volume the way the corpus does, instead of --input
--n                      Grid size for --generate (default: 32)
--ngrain                 Grain count for --generate (default: n^3 // 20)
--material               Cubic constants for --generate: Cu | Z5 | Z8 (default: Cu)
--c C11 C12 C44          Cubic constants in GPa; override --material
--project                Equilibrium projection: auto | none | eq; auto follows the checkpoint (default: auto)
--strain EXX ... EXZ     Macroscopic strain, tensor shear
--stress SXX ... SXZ     Macroscopic stress in Pa, applied through the effective stiffness <A>
--load CASE MAGNITUDE    Named load: uniaxial-stress-x, shear-xy, biaxial-xy, hydrostatic, ...; Pa for stress cases
--homogenize             Whose <A> turns a stress load into a strain: prediction | label (default: prediction)
--measures               Stress measures from xx yy zz xy yz xz vm p s1 s2 s3 tau (default: vm and the largest mean component)
--plot DIR               Write slice, distribution and cube PNGs per measure
--vtk FILE.vti           Write the stress field for ParaView
--out FILE.npz           Write A, and sigma and E_bar with a load
--device                 cpu | cuda (default: cpu)
--threads                Torch CPU threads (default: torch's choice)
```

### Training

Needs an NVIDIA GPU and the corpus in `data/periodic32/<corpus>/`. The recipe of the released checkpoints:

```bash
python -m piml.periodic sweep --which cu --rf 13 --width 64 --augment --norm none --max-steps 36000 --patience 6000 --sched-patience 20 --seeds 0 1 2
```

Add `--linear` for the linear arm, `--project-train` to train through the projection, and `--which z5` or `z8` for the other corpora. Each run saves its weights under `checkpoints/` and appends a row to `logs/results.jsonl`. [tools/queues/train_example.sh](tools/queues/train_example.sh) queues runs in the background.

### Scoring

```bash
python -m piml.score --ckpt weights/cu-nonlinear-rf13-w64-s0-proj.pt --which cu --project eq
```

Needs the corpus; `--project eq` for `-proj` checkpoints.

### Data generation

Needs the MatViz3D clone from [docs/INSTALL.md](docs/INSTALL.md#set-up-data-generation) and g++ with OpenMP.

```bash
tools/build_mesh_refine.sh                   # build the solver driver
python -m piml.mesh_convergence smoke        # check it against known answers
python -m piml.corpus gen --first 1000 --count 200 --material Cu --out data/periodic32/cu
python -m piml.corpus check data/periodic32/cu
```

The Zener 5 and Zener 8 corpora need `--c` and a higher `--maxit`; [tools/README.md](tools/README.md) has the exact commands.

## Documentation

| Document | Covers |
|---|---|
| [Installation](docs/INSTALL.md) | Environments, GPU build, weights download, MatViz3D clone, path variables |
| [Inference](docs/INFERENCE.md) | Choosing a checkpoint, loads, stress measures, plots, Python API, scoring |
| [Data format](docs/DATA.md) | Input requirements, Mandel notation, output and corpus file layouts |
| [Data generation](tools/README.md) | Solver driver, regenerating and checking the corpora, resolution ladder |

## License

MIT, see [LICENSE](LICENSE).
