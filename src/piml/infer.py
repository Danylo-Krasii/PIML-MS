"""Predict the stress-localization field A(x) of one periodic volume, and the stress under a load.

    python -m piml.infer --ckpt weights/cu-nonlinear-rf13-w64-s0-proj.pt --input data/example/sve_1160.hdf5
    python -m piml.infer --ckpt weights/z8-nonlinear-rf13-w64-s0-proj.pt --generate 100000 --material Z8 \
        --strain 1e-3 0 0 0 0 0 --out scratch/z8_100000.npz
    python -m piml.infer --ckpt weights/cu-nonlinear-rf13-w64-s0-proj.pt --input data/example/sve_1160.hdf5 \
        --load uniaxial-stress-x 100e6 --plot scratch/sve_1160 --vtk scratch/sve_1160/stress.vti

Prints the checkpoint settings, whether the equilibrium projection was applied, the shape of A, the
equilibrium ratio of the prediction per load column (physics.equilibrium_ratio, the solver's own
convergence metric: 1e-5 on the corpus labels, 1e-8 on a projected prediction after the float32 cast),
the relative Frobenius error against the stored A when the input carries one, the load report described
below when a load is given, and the runtime.

Conventions follow the corpus: Mandel notation, component order xx, yy, zz, xy, yz, xz, stiffness in Pa,
A[x, y, z, i, k] = Mandel stress i per unit Mandel strain k, so sigma_M(x) = A(x) eps_M with
eps_M = (e_xx, e_yy, e_zz, sqrt2 e_xy, sqrt2 e_yz, sqrt2 e_xz) and tensor shear strains. The network pads
circularly, so every volume is read as a periodic cell.

--input takes an HDF5 file in the corpus schema (voxels and C_grain, optionally A) or an .npz holding
C66 (n, n, n, 6, 6) or voxels and C_grain, optionally A. --generate builds the volume the way
`piml.corpus gen` does, with n^3 // 20 grains unless --ngrain is given, so --generate 1160 reproduces
the corpus volume sve_1160 of the chosen material.

A -proj checkpoint was trained through physics.project_equilibrium, so its raw output is not its
prediction. --project auto, the default, applies the projection whenever the checkpoint records
project_train; the projection needs a cubic grid. A checkpoint carries no material field, so pick the one
trained on the material of the volume. Accuracy was measured at 32^3 with about 20 voxels per grain.

A load is a macroscopic strain (--strain), a macroscopic stress in Pa (--stress) or a named case of
piml.loading (--load CASE MAGNITUDE, for example uniaxial-stress-x 100e6 or shear-xy 50e6). A stress is
turned into the strain E_bar = <A>^-1 : Sigma_bar through the effective stiffness <A> of the prediction,
or of the stored label with --homogenize label, so the mean stress of that field equals the load. With a
load the command prints E_bar, the mean stress and von Mises statistics, the same for the label at the
same E_bar when the input carries one, the relative error of the stress field and of its grain means
against the label, and the mean, 1st and 99th percentiles of each measure. --measures picks the stress
measures of piml.fields (default vm and the largest mean stress component); --plot DIR writes, per
measure, the mid-plane slices (label, prediction and their difference when a label exists), the
distribution over voxels and a 3D cube; --vtk writes sigma, the measures and the grain ids, and with a
label also sigma_label, <measure>_label and <measure>_error (prediction minus label), as VTK image data
for ParaView. Negative magnitudes may be written with an exponent, as in --load uniaxial-stress-x -100e6.

--out writes an .npz with A (float32, Pa per unit strain) and, with a load, sigma (n, n, n, 3, 3) in Pa,
E_bar (3, 3) and, when the input carries a label, sigma_label.
"""

import argparse
import re
import time
from collections import namedtuple
from pathlib import Path

import h5py
import numpy as np
import torch

from . import corpus
from . import mesh_convergence as mc
from .fields import COLS, LABELS, MEASURES, ROWS, grain_mean, measure, relative_error, stress_field, summary
from .loading import LOAD_CASES, load_case, macro_strain
from .microstructure import applied_strain, upper_triangle
from .physics import equilibrium_ratio, project_equilibrium
from .score import net_from_ckpt

Model = namedtuple("Model", "net xm xs ym ys project_train meta")
STATS = ("xm", "xs", "ym", "ys")
MODES = ("auto", "none", "eq")


def load_checkpoint(path, device="cpu"):
    """Network, standardization statistics, project_train flag and remaining metadata of one .pt file."""
    ck = torch.load(path, map_location="cpu", weights_only=True)
    meta = {k: v for k, v in ck.items() if k not in ("state", "project_train") + STATS}
    return Model(net_from_ckpt(ck).to(device), *(ck[k].float().to(device) for k in STATS),
                 bool(ck.get("project_train", False)), meta)


def random_volume(seed, n=32, ngrain=None, material="Cu", c=None):
    """Grain ids (n, n, n) int32 and Mandel stiffness per grain (ngrain, 6, 6) in Pa, built as in the corpus.

    material is a key of mesh_convergence.MATERIALS (Cu, Z5, Z8); c = (c11, c12, c44) in GPa overrides it.
    """
    if c is None:
        if material not in mc.MATERIALS:
            raise ValueError(f"unknown material {material!r}: use one of {sorted(mc.MATERIALS)} or pass c")
        c = mc.MATERIALS[material]
    key = "c11={:g} c12={:g} c44={:g}".format(*c)
    corpus.add_material(key, *c)
    pts, _, C_grain = corpus.microstructure(seed, ngrain or n ** 3 // 20, key)
    return mc.voxelise(pts, n), C_grain


def _projects(model, project):
    if project not in MODES:
        raise ValueError(f"project must be one of {MODES}, got {project!r}")
    return project == "eq" or (project == "auto" and model.project_train)


def predict(model, C66, project="auto"):
    """A (n1, n2, n3, 6, 6) float32, Pa per unit Mandel strain, from the Mandel stiffness field C66 in Pa.

    C66 has shape (n1, n2, n3, 6, 6). project "auto" applies physics.project_equilibrium when the
    checkpoint was trained through it, "eq" always applies it and "none" returns the raw network output.
    The projection runs in float64 and needs n1 = n2 = n3.
    """
    C66 = np.asarray(C66)
    if C66.ndim != 5 or C66.shape[-2:] != (6, 6):
        raise ValueError(f"C66 must have shape (n1, n2, n3, 6, 6), got {C66.shape}")
    grid = C66.shape[:3]
    eq = _projects(model, project)
    if eq and len(set(grid)) > 1:
        raise ValueError(f"the equilibrium projection needs a cubic grid, got {grid}; a checkpoint trained "
                         "through it (project_train) has no valid prediction on this grid")
    x = torch.from_numpy(upper_triangle(C66).astype(np.float32)).to(model.xm.device)
    with torch.no_grad():
        a = model.net((x.permute(3, 0, 1, 2)[None] - model.xm) / model.xs) * model.ys + model.ym
        A = a[0].permute(1, 2, 3, 0).reshape(grid + (6, 6))
        if eq:
            A = project_equilibrium(A[None].double())[0]
    return A.float().cpu().numpy()


stress = stress_field


def _read(path):
    opener = np.load if Path(path).suffix == ".npz" else (lambda p: h5py.File(p, "r"))
    with opener(path) as f:
        voxels = f["voxels"][...] if "voxels" in f else None
        if "C66" in f:
            C66 = f["C66"][...]
        elif voxels is not None and "C_grain" in f:
            C66 = f["C_grain"][...][voxels]
        else:
            raise SystemExit(f"{path} holds neither C66 nor voxels and C_grain")
        return C66, (f["A"][...] if "A" in f else None), voxels


def _mpa(values):
    return " ".join(f"{v / 1e6:.3f}" for v in values)


def _report(name, sigma):
    s = summary(sigma)
    print(f"{name} mean stress, MPa, xx yy zz xy yz xz: {_mpa(s['mean'])}")
    print(f"{name} von Mises, MPa: mean {s['vm_mean'] / 1e6:.3f}, median {s['vm_p50'] / 1e6:.3f}, "
          f"p99 {s['vm_p99'] / 1e6:.3f}, p99.9 {s['vm_p99.9'] / 1e6:.3f}, max {s['vm_max'] / 1e6:.3f}")
    return s


def _load(a, A, A_ref):
    """(E_bar, a one-line description) of the load given on the command line."""
    if a.strain:
        return macro_strain(strain=applied_strain(a.strain)), "macroscopic strain"
    case, magnitude = (a.load[0], float(a.load[1])) if a.load else (None, None)
    if case and load_case(case, magnitude)[0] == "strain":
        return macro_strain(case=case, magnitude=magnitude), f"{case} {magnitude:g}"
    if a.homogenize == "label" and A_ref is None:
        raise SystemExit("--homogenize label needs an input that carries the label A")
    A_hom = A if a.homogenize == "prediction" else A_ref
    through = f"through <A> of the {a.homogenize}"
    if a.stress:
        return macro_strain(A_hom, stress=applied_strain(a.stress)), f"macroscopic stress, {through}"
    return macro_strain(A_hom, case=case, magnitude=magnitude), f"{case} {magnitude:g} Pa, {through}"


def _outputs(a, sigma, sigma_ref, voxels):
    """Print the requested measures and write the plots and the VTK file."""
    mean = sigma.reshape(-1, 3, 3).mean(0)[ROWS, COLS]
    comp = ("xx", "yy", "zz", "xy", "yz", "xz")[int(np.argmax(np.abs(mean)))]
    names = list(dict.fromkeys(a.measures or ["vm", comp]))
    fields = {m: measure(sigma, m) for m in names}
    refs = {m: measure(sigma_ref, m) for m in names} if sigma_ref is not None else {}
    for m in names:
        stats = [(k, f) for k, f in (("prediction", fields[m]), ("label", refs.get(m))) if f is not None]
        cells = [f"{k} mean {f.mean() / 1e6:.3f} p1 {np.percentile(f, 1) / 1e6:.3f} "
                 f"p99 {np.percentile(f, 99) / 1e6:.3f}" for k, f in stats]
        print(f"{m}, MPa: " + "; ".join(cells))
    if a.plot:
        from . import viz
        d = Path(a.plot)
        for m in names:
            pair = {"label": refs[m], "prediction": fields[m]} if m in refs else {"prediction": fields[m]}
            viz.plot_slices(pair, d / f"slices_{m}.png", reference="label" if m in refs else None,
                            title=LABELS[m])
            viz.plot_distribution(pair, d / f"distribution_{m}.png", xlabel=LABELS[m],
                                  colors={"label": viz.INK, "prediction": viz.SERIES[2]})
            viz.plot_cube(fields[m], d / f"cube_{m}.png", title=f"{LABELS[m]}, prediction")
        print(f"wrote {3 * len(names)} PNG files to {d}")
    if a.vtk:
        from . import viz
        cells = {"sigma": sigma, **fields}
        if voxels is not None:
            cells["grain"] = voxels.astype(np.int32)
        if sigma_ref is not None:
            cells.update({"sigma_label": sigma_ref, **{f"{m}_label": refs[m] for m in names},
                          **{f"{m}_error": fields[m] - refs[m] for m in names}})
        print(f"wrote {viz.write_vti(a.vtk, cells)}")


def main():
    ap = argparse.ArgumentParser(prog="python -m piml.infer",
                                 description="Predict A(x) of one periodic volume with a checkpoint, and "
                                             "the stress under a macroscopic load.")
    ap._negative_number_matcher = re.compile(r"^-\.?\d")
    ap.add_argument("--ckpt", required=True, help="checkpoint .pt file")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", metavar="FILE",
                     help="corpus HDF5 (voxels, C_grain, optional A) or .npz "
                          "(C66, or voxels and C_grain; optional A)")
    src.add_argument("--generate", type=int, metavar="SEED", help="build a random volume as the corpus does")
    ap.add_argument("--n", type=int, default=32, help="grid size for --generate (default 32)")
    ap.add_argument("--ngrain", type=int, default=None, help="grain count for --generate (default n^3 // 20)")
    ap.add_argument("--material", default="Cu", choices=list(mc.MATERIALS),
                    help="cubic constants for --generate (default Cu)")
    ap.add_argument("--c", type=float, nargs=3, metavar=("C11", "C12", "C44"),
                    help="cubic constants in GPa, override --material")
    ap.add_argument("--project", default="auto", choices=MODES,
                    help="equilibrium projection; auto follows the checkpoint's project_train (default auto)")
    load = ap.add_mutually_exclusive_group()
    load.add_argument("--strain", type=float, nargs=6, metavar=("EXX", "EYY", "EZZ", "EXY", "EYZ", "EXZ"),
                      help="macroscopic strain, tensor shear")
    load.add_argument("--stress", type=float, nargs=6, metavar=("SXX", "SYY", "SZZ", "SXY", "SYZ", "SXZ"),
                      help="macroscopic stress in Pa, applied through the effective stiffness <A>")
    load.add_argument("--load", nargs=2, metavar=("CASE", "MAGNITUDE"),
                      help="named load: " + ", ".join(LOAD_CASES) +
                           "; the magnitude is in Pa for stress cases and a strain for strain cases")
    ap.add_argument("--homogenize", default="prediction", choices=["prediction", "label"],
                    help="whose <A> turns a stress load into a strain (default prediction)")
    ap.add_argument("--measures", nargs="+", choices=MEASURES, metavar="M",
                    help="stress measures to print, plot and export, from: " + " ".join(MEASURES) +
                         " (default vm and the largest mean stress component)")
    ap.add_argument("--plot", metavar="DIR", help="write slice, distribution and cube PNGs per measure")
    ap.add_argument("--vtk", metavar="FILE.vti",
                    help="write sigma, the measures and grain ids (and the label's, with the differences) "
                         "for ParaView")
    ap.add_argument("--out", metavar="OUT.npz", help="write A, and sigma and E_bar with a load")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="default cpu")
    ap.add_argument("--threads", type=int, default=None, help="torch CPU threads (default torch's choice)")
    a = ap.parse_args()
    if (a.plot or a.vtk or a.measures) and not (a.strain or a.stress or a.load):
        ap.error("--measures, --plot and --vtk need a load: --strain, --stress or --load")
    if a.homogenize == "label" and not (a.stress or a.load):
        ap.error("--homogenize label applies to a stress load: --stress or a stress case of --load")
    if a.homogenize == "label" and a.generate is not None:
        ap.error("--homogenize label needs an input that carries the label A, not --generate")
    if a.load:
        try:
            if not np.isfinite(float(a.load[1])):
                raise ValueError("the magnitude must be finite")
            kind = load_case(a.load[0], float(a.load[1]))[0]
        except ValueError as e:
            ap.error(f"--load {' '.join(a.load)}: {e}")
        if kind == "strain" and a.homogenize == "label":
            ap.error("--homogenize label applies to a stress load: --stress or a stress case of --load")
    for flag in ("strain", "stress"):
        if getattr(a, flag) and not np.isfinite(getattr(a, flag)).all():
            ap.error(f"--{flag} values must be finite")
    if a.threads:
        torch.set_num_threads(a.threads)

    t0 = time.time()
    model = load_checkpoint(a.ckpt, a.device)
    if a.input:
        C66, A_ref, voxels = _read(a.input)
        origin = a.input
    else:
        voxels, C_grain = random_volume(a.generate, a.n, a.ngrain, a.material, a.c)
        C66, A_ref = C_grain[voxels], None
        origin = f"generated seed {a.generate}, {a.c or a.material}, {len(C_grain)} grains"
    t1 = time.time()
    A = predict(model, C66, a.project)
    t2 = time.time()

    m = model.meta
    print(f"checkpoint {a.ckpt}: rf {m['rf']} width {m['width']} linear {m['linear']} "
          f"project_train {model.project_train}")
    print(f"volume {origin}: grid {C66.shape[:3]}")
    print(f"A {A.shape} {A.dtype}, Pa per unit Mandel strain; equilibrium projection "
          f"{'applied' if _projects(model, a.project) else 'not applied'} (--project {a.project})")
    if len(set(A.shape[:3])) == 1:
        r = equilibrium_ratio(torch.from_numpy(A)[None].double())[0].numpy()
        print("equilibrium ratio per load column xx yy zz xy yz xz: " + " ".join(f"{v:.3e}" for v in r))
    out = {"A": A}
    if A_ref is not None:
        A_ref = A_ref.astype(np.float64).reshape(A.shape)
        print(f"relative error against the stored A: {np.linalg.norm(A - A_ref) / np.linalg.norm(A_ref):.6f}")
    if a.strain or a.stress or a.load:
        E_bar, what = _load(a, A, A_ref)
        if not E_bar.any():
            raise SystemExit("the load is zero, so the stress is zero everywhere")
        out["E_bar"], out["sigma"] = E_bar, stress_field(A, E_bar)
        print(f"load: {what}")
        print("E_bar, xx yy zz xy yz xz: " + " ".join(f"{v:.4e}" for v in E_bar[ROWS, COLS]))
        _report("prediction", out["sigma"])
        sigma_ref = None
        if A_ref is not None:
            out["sigma_label"] = sigma_ref = stress_field(A_ref, E_bar)
            _report("label", sigma_ref)
            line = f"relative error of sigma against the label: {relative_error(out['sigma'], sigma_ref):.6f}"
            if voxels is not None:
                g = relative_error(grain_mean(out["sigma"], voxels), grain_mean(sigma_ref, voxels))
                line += f"; of its grain means: {g:.6f}"
            print(line)
        _outputs(a, out["sigma"], sigma_ref, voxels)
    tail = f", load and outputs {time.time() - t2:.2f} s" if "sigma" in out else ""
    print(f"runtime: load and build {t1 - t0:.2f} s, predict {t2 - t1:.2f} s{tail}")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        np.savez(a.out, **out)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
