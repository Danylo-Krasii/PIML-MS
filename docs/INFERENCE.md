# Inference

`piml.infer` predicts A(x) for one periodic volume from its stiffness field and, given a load, returns the local stress with plots and a ParaView file. It needs a checkpoint and the CPU environment from [INSTALL.md](INSTALL.md), and neither the solver nor the corpora. `python -m piml.infer --help` lists every flag.

## Choosing a checkpoint

Names follow `<corpus>-<arm>-rf13-w64-s<seed>[-proj].pt`. Pick the corpus of your volume's material first (`cu`, `z5` or `z8`): a checkpoint does not record its material, and a mismatch gives a wrong field without any warning. Then prefer a nonlinear `-proj` file.

A `-proj` checkpoint was trained through the equilibrium projection, so its raw output is not its prediction. `--project auto`, the default, applies the projection whenever the checkpoint was trained through it; leave it on. `--project eq` projects the output of a plain checkpoint after the fact, and `--project none` turns the projection off. The projection needs a cubic grid.

## Predicting A(x)

```bash
python -m piml.infer --ckpt weights/cu-nonlinear-rf13-w64-s0-proj.pt --input data/example/sve_1160.hdf5
```

The output gives the shape of A, whether the projection ran, the equilibrium ratio of each of the six columns of A and, when the input carries the solver's A, the relative error against it. The equilibrium ratio is the solver's convergence measure; a value of 1e-3 or more on a `-proj` checkpoint means the projection did not run.

`--input` takes a corpus HDF5 file or an `.npz` of your own, with either `C66` (n, n, n, 6, 6) or `voxels` and `C_grain`, in Pa and Mandel notation ([DATA.md](DATA.md#input-files)). `--generate SEED` builds a random volume the way the corpus generator does, and `--material` or `--c C11 C12 C44` (GPa) sets its constants. Pair each material with its own checkpoint:

```bash
python -m piml.infer --ckpt weights/z8-nonlinear-rf13-w64-s0-proj.pt --generate 100000 --material Z8 --strain 1e-3 0 0 0 0 0 --out scratch/z8_100000.npz
```

`--out` writes A and, with a load, E_bar and the stress `sigma` ([DATA.md](DATA.md#output)).

## Applying a load and reading the stress

Since sigma(x) = A(x) : E_bar, one prediction serves every load. `--load CASE MAGNITUDE`, `--stress` or `--strain` applies one:

```bash
python -m piml.infer --ckpt weights/cu-nonlinear-rf13-w64-s0-proj.pt --input data/example/sve_1160.hdf5 --load uniaxial-stress-x 100e6
```

The report gives the applied E_bar, the mean stress and von Mises statistics of the prediction and, when the input carries the solver's A, the same for the label under the same E_bar together with the relative error of the stress field. A stress load becomes a strain through the effective stiffness <A>, the volume mean of A, so the prediction's mean stress equals the load; `--homogenize label` uses the stored A instead.

### Load cases

The magnitude m is in Pa for a stress case and dimensionless for a strain case. Unnamed components are zero, and a negative m reverses the load.

| case | load |
|---|---|
| `uniaxial-stress-x`, `uniaxial-stress-y`, `uniaxial-stress-z` | Sigma_ii = m |
| `uniaxial-strain-x`, `uniaxial-strain-y`, `uniaxial-strain-z` | E_ii = m |
| `shear-xy`, `shear-yz`, `shear-xz` | Sigma_ij = Sigma_ji = m |
| `biaxial-xy`, `biaxial-yz`, `biaxial-xz` | Sigma_ii = Sigma_jj = m |
| `hydrostatic` | Sigma = m I |

`--stress SXX SYY SZZ SXY SYZ SXZ` (Pa) and `--strain EXX EYY EZZ EXY EYZ EXZ` take any tensor, with tensor shear: half the engineering shear strain.

### Stress measures, plots and ParaView

`--measures` picks the fields to report, plot and export, all in Pa: the components `xx` to `xz`, von Mises `vm`, mean stress `p`, principal stresses `s1` >= `s2` >= `s3`, and maximum shear `tau`. The default is `vm` and the largest mean component.

`--plot DIR` writes three PNG files per measure: the mid-plane slice of prediction, label and their difference, the distribution over voxels, and the prediction on three faces of the cube. `--vtk FILE.vti` writes the stress tensor, the measures, the grain ids and, with a label, the label's fields and the errors, as VTK image data that ParaView opens ([DATA.md](DATA.md#vtk-file)).

## From Python

```python
import h5py

from piml.fields import measure, stress_field
from piml.infer import load_checkpoint, predict
from piml.loading import macro_strain

with h5py.File("data/example/sve_1160.hdf5", "r") as f:
    C66 = f["C_grain"][...][f["voxels"][...]]
A = predict(load_checkpoint("weights/cu-nonlinear-rf13-w64-s0-proj.pt"), C66)
E_bar = macro_strain(A, case="shear-xy", magnitude=50e6)
vm = measure(stress_field(A, E_bar), "vm")
```

`predict` takes the Mandel stiffness field (n1, n2, n3, 6, 6) in Pa and returns A, float32, in the same shape. `piml.fields`, `piml.loading` and `piml.viz` (`plot_slices`, `plot_distribution`, `plot_cube`, `write_vti`) document their arguments in their docstrings.

## Scoring a checkpoint on the test volumes

Scoring needs the regenerated corpus ([tools/README.md](../tools/README.md)) and a 1.49 GB cache of it, built once per corpus:

```bash
python -c "from piml.periodic import load; load('cu')"
python -m piml.score --ckpt weights/cu-nonlinear-rf13-w64-s0-proj.pt --which cu --project eq
```

For a `-proj` checkpoint read the `eq` row of the projection table; the `overall` line scores the raw output. For a plain checkpoint, drop `--project eq` and read `overall`. `--out DIR` saves per-volume and per-voxel errors.

## Grid size and grain size

The checkpoints were trained at 32^3 with about 20 voxels per grain. Other grids run, and `--generate` keeps that grain density at any `--n`, but volumes far from it are outside the training distribution.

## Runtime

On an 8-thread laptop CPU, predicting a 32^3 volume takes 0.08 to 0.20 s and a 64^3 volume 1.1 to 1.5 s; a full `python -m piml.infer` call takes 1.3 to 2.5 s, mostly start-up. `--device cuda` runs the network on a GPU in the `cu118` environment.
