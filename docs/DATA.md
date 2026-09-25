# Data format

What a checkpoint expects as input, what `piml.infer` writes, and the layout of the corpus files.

## What a model needs

The only input is the stiffness field C(x) of a periodic volume. The model takes no strain, grain id or coordinate.

### Stiffness in Mandel notation

Each voxel needs a symmetric 6x6 stiffness matrix in Pa, in Mandel notation with component order xx, yy, zz, xy, yz, xz:

```
C66[I, J] = f_I f_J C_ijkl,   f = (1, 1, 1, sqrt2, sqrt2, sqrt2)
```

Normal-shear entries carry a factor sqrt2 and shear-shear entries a factor 2, so for copper in its crystal axes C66[3, 3] = 2 c44 = 150.8 GPa, where Voigt notation gives 75.4 GPa. `piml.microstructure.to_mandel4` converts a (..., 3, 3, 3, 3) tensor. The network reads the 21 upper-triangle entries (`np.triu_indices(6)` order).

Wrong units (GPa) or Voigt scaling raise no error and give a wrong A. To convert from Voigt order (xx, yy, zz, yz, xz, xy, no shear factors):

```python
import numpy as np

P = [0, 1, 2, 5, 3, 4]
f = np.array([1, 1, 1, np.sqrt(2), np.sqrt(2), np.sqrt(2)])
C66 = C_voigt[..., P, :][..., :, P] * np.outer(f, f)
```

### Grid and material

The network pads circularly, so it treats every volume as periodic. A `-proj` checkpoint and `--project eq` need a cubic n x n x n grid; a plain checkpoint also runs on an n1 x n2 x n3 box, untested. Keep about 20 voxels per grain, as in training (1,638 grains at 32^3).

A checkpoint standardizes its input with statistics of its own training corpus and does not record its material, so use the one whose prefix matches your volume: `cu-`, `z5-` or `z8-`. The training volumes are Voronoi tessellations with uniformly random orientations; the error on other cubic constants, textures or elongated grains is unmeasured.

## Input files

`--input` reads `.npz` files with numpy and anything else with h5py. It uses `C66` if present, otherwise `voxels` and `C_grain`. An optional `A` is a label to score against.

| key | shape | content |
|---|---|---|
| `C66` | (n1, n2, n3, 6, 6) | Mandel stiffness per voxel, Pa |
| `voxels` | (n1, n2, n3), integer | zero-based grain id per voxel, axes x, y, z |
| `C_grain` | (G, 6, 6) | Mandel stiffness per grain, Pa, so that `C_grain[voxels]` is C(x) |
| `A`, optional | (n1, n2, n3, 6, 6) or (n1, n2, n3, 36) | label |

The input is cast to float32 before the network, so float32 and float64 files give identical A.

### Building a volume

For grain orientations as Bunge Euler angles (phi1, Phi, phi2) in radians, rotate the crystal stiffness with `stiffness_field(..., source="fft")` and convert it to Mandel form. Always pass `source`; without it the function guesses the angle convention from the value range. This writes a random copper volume and predicts A for it:

```bash
python - <<'EOF'
import os

import numpy as np

from piml.mesh_convergence import voxelise
from piml.microstructure import cubic_stiffness, stiffness_field, to_mandel4

n, ngrain = 32, 32 ** 3 // 20
rng = np.random.default_rng(7)
voxels = voxelise(rng.random((ngrain, 3)), n)
euler = np.stack([rng.uniform(0, 2 * np.pi, ngrain),
                  np.arccos(rng.uniform(-1, 1, ngrain)),
                  rng.uniform(0, 2 * np.pi, ngrain)], axis=1)
C0 = cubic_stiffness(168.4e9, 121.4e9, 75.4e9)
C_grain = to_mandel4(stiffness_field(np.arange(ngrain), euler, C0, source="fft"))

os.makedirs("scratch", exist_ok=True)
np.savez("scratch/my_volume.npz", voxels=voxels, C_grain=C_grain)
EOF
python -m piml.infer --ckpt weights/cu-nonlinear-rf13-w64-s0-proj.pt --input scratch/my_volume.npz \
    --strain 1e-3 0 0 0 0 0 --out scratch/my_A.npz
```

For quaternions (w, x, y, z) that take crystal components to sample components, `piml.corpus.quat_to_matrix` gives the rotation matrix g, and `to_mandel4(np.einsum("gpi,gqj,grk,gsl,ijkl->gpqrs", g, g, g, g, C0))` the per-grain stiffness.

## Output

A has shape (n, n, n, 6, 6), float32, in Pa per unit Mandel strain: `A[x, y, z, i, k]` is stress component i at voxel (x, y, z) per unit strain component k.

```
sigma_M(x) = A(x) eps_M,   eps_M = (E_xx, E_yy, E_zz, sqrt2 E_xy, sqrt2 E_yz, sqrt2 E_xz)
```

E_xy is the tensor shear strain, half the engineering shear. `--out` writes an `.npz` with `A` and, with a load, `E_bar` (3, 3), `sigma` (n, n, n, 3, 3) in Pa, and `sigma_label` when the input carries A.

The projection enforces equilibrium only, not the average-strain condition <C^-1 A> = I.

### VTK file

`--vtk` writes ASCII VTK image data with one cell per voxel, spacing 1 and x fastest. Arrays: `sigma` (9 components, the active tensor), one array per measure, `grain` when the input has grain ids, and `sigma_label`, `<measure>_label` and `<measure>_error` (prediction minus label) when it carries A. All values are in Pa. A 32^3 file takes about 8 MB.

## Corpus files

Each corpus (`cu`, `z5`, `z8`) holds 200 files `data/periodic32/<corpus>/sve_<seed>.hdf5`, seeds 1000 to 1199, about 5 MB each. `PIML_DATA` moves the `data/` root. `data/example/sve_1160.hdf5` is a copy of `cu/sve_1160.hdf5`.

| dataset | shape | content |
|---|---|---|
| `voxels` | (32, 32, 32), int32 | zero-based grain id per voxel |
| `quat` | (1638, 4) | orientation quaternion (w, x, y, z) per grain |
| `C_grain` | (1638, 6, 6) | Mandel stiffness per grain, Pa |
| `A` | (32, 32, 32, 6, 6), float32 | the label |
| `sigma_avg` | (6, 6) | mean stress under each of the six load cases of 1e-4 |
| `iters`, `secs` | (6,) | solver iterations and seconds per load case |

Root attributes record the seed, the material and its constants (`c11_GPa`, `c12_GPa`, `c44_GPa`, `zener`), the solver settings (`tol` 1e-5, `maxit`, `load` 1e-4), `converged`, and `solver`, a hash of the driver binary and `fft_homog.hpp`. A rebuilt driver has a new hash; see [Reproducibility](../tools/README.md#reproducibility).

`piml.corpus gen` writes `sve_<seed>.hdf5.part` and renames it when complete. A volume that did not converge within `maxit` becomes `sve_<seed>.UNCONVERGED.hdf5`, which training never reads. Each run also appends its settings to `manifest.json`.

### Seed pools

A seed fixes the geometry and orientations of a generated volume. The pools do not overlap:

| pool | seeds | use |
|---|---|---|
| S | 0 to 63 | one geometry at 16^3, 32^3 and 64^3 for the resolution ladder |
| B | 1000 to 2499 | independent 32^3 volumes; the corpora use 1000 to 1199 |
| U | 100000 and up | unlabeled volumes, for example `--generate 100000` |

### Training cache

On first use `periodic.load` writes `scratch/periodic_<corpus>_1000_200.npz` (1.49 GB), holding `C` (200, 32, 32, 32, 21) and `A` (200, 32, 32, 32, 36), both float32. Delete it after regenerating a corpus, because `load` never rereads the HDF5 files while it exists.
