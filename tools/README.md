# Data generation

This guide rebuilds the three corpora behind the released checkpoints (3 x 200 volumes, 3.02 GB). Python draws each volume's grains and orientations from its seed, and a C++ driver solves six load cases with the Moulinec-Suquet FFT solver of [MatViz3D](https://github.com/MME-NTU-KhPI/MatViz3D), one HDF5 file per volume.

| file | purpose |
|---|---|
| `mesh_refine.cpp` | the solver driver: applies six strains of 1e-4 to a voxel grain map and writes the stress fields |
| `build_mesh_refine.sh` | compiles the driver into `build/mesh_refine` |
| `queues/train_example.sh` | an example queue for unattended training (usage in its header) |

You need the MatViz3D clone from [INSTALL.md](../docs/INSTALL.md#set-up-data-generation), g++ with C++17 and OpenMP, and any environment from INSTALL.md; generation does not use torch. Run the commands from the repository root with the environment active.

## 1. Build the driver

```bash
tools/build_mesh_refine.sh
```

It prints `built <repository>/build/mesh_refine`. If the clone or the binary live elsewhere, export `PIML_MATVIZ3D` and `PIML_MESH_REFINE` first.

## 2. Run the smoke test

```bash
python -m piml.mesh_convergence smoke
```

It solves a single crystal, which must return A(x) = C in every voxel, and a two-grain laminate, which must carry continuous tractions across its interface. The first catches a wrong component order between Python and the solver. The last line reads `PASS` or `FAIL`.

## 3. Solve one volume

Copper volume 1160 is the example volume, so solving it gives the cost of one volume and compares your build with a stored label:

```bash
python -m piml.corpus gen --first 1160 --count 1 --workers 1 --out scratch/trial
python -c "import h5py, numpy as np; a = h5py.File('scratch/trial/sve_1160.hdf5')['A'][...].astype(float); b = h5py.File('data/example/sve_1160.hdf5')['A'][...].astype(float); print(np.array_equal(a, b), np.linalg.norm(a - b) / np.linalg.norm(b))"
```

`True` means your build reproduces the stored label bit for bit. On another CPU or compiler expect `False` with a relative difference near the solver tolerance.

## 4. Generate the three corpora

```bash
python -m piml.corpus gen --pool B --first 1000 --count 200 --n 32 --material Cu --tol 1e-5 --maxit 6000 --workers 12 --threads 1 --out data/periodic32/cu
python -m piml.corpus gen --pool B --first 1000 --count 200 --n 32 --material Z5 --c 168.4 121.4 117.5 --tol 1e-5 --maxit 9000 --workers 12 --threads 1 --out data/periodic32/z5
python -m piml.corpus gen --pool B --first 1000 --count 200 --n 32 --material Z8 --c 168.4 121.4 188.0 --tol 1e-5 --maxit 12000 --workers 12 --threads 1 --out data/periodic32/z8
```

These are the settings of the original corpora. The three share seeds, so they share grain maps and orientations and differ only in c44 (Zener ratio 3.21, 5 and 8).

- `corpus.py` knows only copper's constants, so `Z5` and `Z8` need `--c` (GPa).
- `--maxit` caps the solver iterations per load case and rises with anisotropy. A volume that hits the cap is written as `sve_<seed>.UNCONVERGED.hdf5`, and training then stops at the missing file.
- `--workers` sets how many volumes are solved at once. Keep `--threads 1`: another OpenMP thread count can change the rounding of the solver's convergence test.
- An interrupted command can be rerun as it is; it skips volumes already on disk.

Each corpus takes about 1 to 2 hours on 12 workers and about 1 GB on disk.

## 5. Check each corpus

```bash
python -m piml.corpus check data/periodic32/cu
```

`check` takes one directory and about 30 s. It tests that orientations are uniform over rotations, that grains wrap across the periodic faces like interior neighbours, and that the labels satisfy the average-strain theorem: the mean strain recovered from A must equal the applied one. Each line prints its statistic next to the expected value or bound.

## Reproducibility

A fresh build from MatViz3D commit `8c4ee49` can reproduce the stored labels bit for bit; only `secs` and the `solver` attribute then differ. `solver` is a hash of the driver binary and `fft_homog.hpp`, so every build writes its own. `-march=native` tunes the driver to the build CPU, so a build on another CPU can differ at rounding level. Grain maps, orientations and stiffnesses come from numpy's seeded generator and do not depend on the build.

## Optional: the 64^3 resolution ladder

The ladder measures the discretization error of the labels: how far a 32^3 solution sits from a 64^3 solution of the same microstructure averaged back to 32^3. Training and inference do not need it.

```bash
for M in Cu Z5 Z8; do python -m piml.mesh_convergence run --material $M --seeds 0 1 2 3 4 5 6 7 --grids 16 32 64 --tol 1e-5 --maxit 12000; done
for M in Cu Z5 Z8; do python -m piml.mesh_convergence analyze --grids 16 32 64 --material $M; done
```

It solves seeds 0 to 7 of pool S at three grids (about 4.5 h in total, 0.97 GB under `data/ladder/`), and `analyze` prints the relative difference between the 32^3 and 64^3 solutions.
