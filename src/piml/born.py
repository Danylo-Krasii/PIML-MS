"""The Born ladder: how well the solver's own truncated series predicts A(x), untrained.

Started from the applied strain, iterate K of the Moulinec-Suquet basic scheme is the
Lippmann-Schwinger Neumann series truncated at order K. K = 0 is the Taylor field C(x),
K = 1 is an analytic translation-invariant kernel acting on C(x) and costs no training
at all. It is NOT the function class the linear network fits: at fixed C0 the stress
iterate A^1 = C - C:Gamma0:(C - C0) is quadratic in C(x), while the linear arm is affine
in C(x). Measured, rel|A^1(2C) - 2A^1(C)| = 0.19 against 0.00 at K = 0. Orders 2 and
above carry the grain-to-grain interaction terms, and their weight grows with contrast.

So this measures the line every learned model has to beat, and it needs no training.

The reference medium matters. Iterate K is an approximation about a chosen C0, and an
unfavourable C0 makes the analytic baseline a strawman, so C0 is recorded and swept.

Requires the solver driver (tools/build_mesh_refine.sh) and a corpus under data/periodic32/.

    python -m piml.born ladder --seeds 1000 1001 1002 --K 0 1 2 3 5 10
    python -m piml.born ladder --seeds 1160 ... 1199 --K 0 1 2 3 5 --scale 0.5
    python -m piml.born reference --seed 1000 --K 1 --scales 0.4 0.45 0.5 0.55 0.6
"""

import argparse
import numpy as np
import h5py

from . import mesh_convergence as mc
from . import corpus
from . import paths

LADDER = [0, 1, 2, 3, 5, 10]
CORPUS = {"cu": "periodic32/cu", "z5": "periodic32/z5", "z8": "periodic32/z8"}


def rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def load_converged(seed, d):
    with h5py.File(d / f"sve_{seed}.hdf5", "r") as f:
        return (f["A"][...].astype(float), f["voxels"][...], f["C_grain"][...],
                float(f.attrs["lambda0_Pa"]), float(f.attrs["mu0_Pa"]))


def iterate(vox, C, K, ref, threads):
    """A(x) from exactly K Green-operator updates. tol=0 so the loop never exits early."""
    A, iters, _ = mc.solve(vox, C, f"born_K{K}", tol=0.0, maxit=K, threads=threads, ref=ref)
    return A, [i[0] for i in iters]


def ladder(a):
    d = paths.DATA / CORPUS[a.which]
    rows = []
    for seed in a.seeds:
        Aref, vox, C, lam0, mu0 = load_converged(seed, d)
        lam0, mu0 = lam0 * a.scale, mu0 * a.scale
        grains = mc.grain_mean(Aref, vox)
        print(f"\nseed {seed}: reference medium lambda0 {lam0/1e9:.1f} GPa, mu0 {mu0/1e9:.1f} GPa "
              f"(scale {a.scale})")
        print(f"  {'K':>3} {'rel.err A':>11} {'grain-mean':>11} {'within-grain':>13}")
        for K in a.K:
            AK, _ = iterate(vox, C, K, (lam0, mu0), a.threads)
            gK = mc.grain_mean(AK, vox)
            e = rel(AK, Aref)
            eb = rel(gK, grains)
            ew = rel(AK - gK, Aref - grains)
            print(f"  {K:>3} {e:>11.4f} {eb:>11.4f} {ew:>13.4f}", flush=True)
            rows.append((seed, K, e, eb, ew))
    print("\nmean over seeds")
    print(f"  {'K':>3} {'rel.err A':>11} {'grain-mean':>11} {'within-grain':>13}")
    for K in a.K:
        sel = [r for r in rows if r[1] == K]
        print(f"  {K:>3} {np.mean([r[2] for r in sel]):>11.4f} "
              f"{np.mean([r[3] for r in sel]):>11.4f} {np.mean([r[4] for r in sel]):>13.4f}")
    print("\nOn periodic labels the trained linear arm reaches 0.0595 at RF 13 and 0.0540 at RF 31,")
    print("the constant-tensor floor is 0.2097 and the Taylor field 0.2062.")


def reference(a):
    """Sweep C0 at fixed K. The solver's own pick is one point on this curve."""
    d = paths.DATA / CORPUS[a.which]
    Aref, vox, C, lam0, mu0 = load_converged(a.seed, d)
    print(f"seed {a.seed}, K = {a.K}. Solver picked lambda0 {lam0/1e9:.1f}, mu0 {mu0/1e9:.1f} GPa")
    print(f"  {'scale':>7} {'lambda0 GPa':>12} {'mu0 GPa':>10} {'rel.err A':>11}")
    for s in a.scales:
        A, _ = iterate(vox, C, a.K, (lam0 * s, mu0 * s), a.threads)
        print(f"  {s:>7.2f} {lam0*s/1e9:>12.1f} {mu0*s/1e9:>10.1f} {rel(A, Aref):>11.4f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["ladder", "reference"])
    ap.add_argument("--which", default="cu", choices=list(CORPUS))
    ap.add_argument("--seeds", type=int, nargs="+", default=[1000, 1001, 1002])
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--K", type=int, nargs="+", default=LADDER)
    ap.add_argument("--scales", type=float, nargs="+", default=[0.6, 0.8, 1.0, 1.25, 1.5, 2.0])
    ap.add_argument("--scale", type=float, default=1.0,
                    help="multiply the solver's reference medium; 0.5 is the K=1 accuracy optimum")
    ap.add_argument("--threads", type=int, default=12)
    a = ap.parse_args()
    if a.cmd == "reference":
        a.K = a.K[0] if isinstance(a.K, list) else a.K
        reference(a)
    else:
        ladder(a)


if __name__ == "__main__":
    main()
