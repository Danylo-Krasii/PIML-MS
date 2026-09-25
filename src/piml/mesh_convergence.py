"""Mesh convergence: the same periodic Voronoi microstructure voxelised at three
resolutions and solved with the MatViz3D FFT solver, which shows how far the 32^3
labels sit from a refined solve.

The corpora sit at about 20 voxels per grain (2.71 voxels of grain edge). Geometry
and orientations are generated here, not by MatViz3D, because the application has
no way to load a microstructure; the statistics match the corpora (uniform random
orientations, ~20 voxels per grain at 32^3, periodic). This module also holds the
solver wrapper `solve` that `piml.corpus` and `piml.born` call.

The solver's Mandel component order is xx, yy, zz, yz, xz, xy. This module's
is xx, yy, zz, xy, yz, xz. `P` maps ours to the
solver's, `PINV` back. The smoke test fixes the permutation with a known
answer: a single crystal must return <sigma> = C : E to rounding, and a laminate
must carry continuous tractions across its interface. The traction jump scales
one for one with the solver tolerance (2.6e-4 at 1e-5, 1.1e-7 at 1e-8). The corpora
use 1e-5: at 1e-6 the 64^3 shear loads did not converge in 8000 iterations
(seed 0: 3661/3217/4030/8000/6540/8000, 933 s on 12 threads, against
2781/2301/2648/4066/4555/4510 at 32^3), so iteration count grows with
resolution here, and 3e-4 of solver error sits two orders below the few-percent
discretization differences being measured.

Usage:  python -m piml.mesh_convergence smoke
        python -m piml.mesh_convergence run   [--seeds 0 1 ...] [--grids 16 32 64] [--threads 12] [--tol 1e-5] [--maxit 12000]
        python -m piml.mesh_convergence analyze
Requires: numpy, scipy; the driver built from mesh_refine.cpp by tools/build_mesh_refine.sh
"""

import argparse
import os
import subprocess

import numpy as np
from scipy.spatial import cKDTree
from . import paths

from .microstructure import cubic_stiffness, to_mandel4

SCRATCH = str(paths.SCRATCH / "mesh")
DRIVER = str(paths.MESH_REFINE)
P = np.array([0, 1, 2, 4, 5, 3])          # ours -> solver
PINV = np.argsort(P)                      # solver -> ours
GRAINS_32 = 32 ** 3 // 20                 # 1638, the corpora's density
MATERIALS = {"Cu": (168.4, 121.4, 75.4), "Z5": (168.4, 121.4, 117.5), "Z8": (168.4, 121.4, 188.0)}


def random_rotation(rng):
    q = rng.normal(size=4); q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def microstructure(seed, ngrain=GRAINS_32, material="Cu"):
    """Seed points in the unit cube and one Mandel stiffness per grain, our order."""
    rng = np.random.default_rng(seed)
    pts = rng.random((ngrain, 3))
    C0 = cubic_stiffness(*(c * 1e9 for c in MATERIALS[material]))
    C = np.stack([to_mandel4(np.einsum('pi,qj,rk,sl,ijkl->pqrs', g, g, g, g, C0))
                  for g in (random_rotation(rng) for _ in range(ngrain))])
    return pts, C


def voxelise(pts, n):
    """Grain id per voxel, periodic nearest seed to the voxel centre. Shape (n, n, n), [x, y, z]."""
    offs = np.array([[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)])
    imgs = (pts[None] + offs[:, None]).reshape(-1, 3)
    owner = np.tile(np.arange(len(pts)), len(offs))
    c = (np.arange(n) + 0.5) / n
    X, Y, Z = np.meshgrid(c, c, c, indexing="ij")
    _, idx = cKDTree(imgs).query(np.stack([X, Y, Z], -1).reshape(-1, 3))
    return owner[idx].reshape(n, n, n).astype(np.int32)


def solve(vox, C, tag, tol=1e-6, maxit=8000, threads=8, ref=None):
    """Run the driver; return A(x) in our order, shape (n, n, n, 6, 6), plus iteration counts."""
    n = vox.shape[0]; d = os.path.join(SCRATCH, tag); os.makedirs(d, exist_ok=True)
    np.ascontiguousarray(vox.transpose(2, 1, 0)).astype(np.int32).tofile(os.path.join(d, "phase.i32"))
    Cs = C[:, P][:, :, P]
    np.ascontiguousarray(Cs).astype(np.float64).tofile(os.path.join(d, "C.f64"))
    env = dict(os.environ, OMP_NUM_THREADS=str(threads))
    cmd = [DRIVER, str(n), str(len(C)), os.path.join(d, "phase.i32"), os.path.join(d, "C.f64"),
           str(tol), str(maxit), os.path.join(d, "out")]
    if ref is not None:                      # fixed reference medium: needed so a Born iterate is
        cmd += [str(ref[0]), str(ref[1])]    # a stated approximation rather than a solver heuristic
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr)
    S = np.zeros((n, n, n, 6, 6)); E = np.zeros((6, 6)); iters = []
    for k in range(6):
        f = np.fromfile(os.path.join(d, f"out.{k}.f64"), dtype=np.float64).reshape(6, n, n, n)   # [c, z, y, x]
        S[..., :, k] = f.transpose(3, 2, 1, 0)[..., PINV]                                          # -> [x, y, z, c] ours
        E[P[k], k] = 1e-4                      # solver's unit load k is our component P[k]
        with open(os.path.join(d, f"out.{k}.txt")) as fh:
            it, status, secs = fh.readline().split()
        iters.append((int(it), status, float(secs)))
        os.remove(os.path.join(d, f"out.{k}.f64"))
    return S @ np.linalg.inv(E), iters, r.stdout


def block_mean(A, f):
    n = A.shape[0] // f
    return A.reshape(n, f, n, f, n, f, 6, 6).mean((1, 3, 5))


def boundary_mask(vox, depth=1):
    """Voxels within `depth` steps (6-neighbour) of a grain boundary, periodic."""
    m = np.zeros(vox.shape, bool)
    for ax in range(3):
        m |= vox != np.roll(vox, 1, ax); m |= vox != np.roll(vox, -1, ax)
    for _ in range(depth - 1):
        g = m.copy()
        for ax in range(3):
            g |= np.roll(m, 1, ax); g |= np.roll(m, -1, ax)
        m = g
    return m


def grain_mean(A, vox):
    """Per-voxel field replaced by its grain average: the between-grain part."""
    flat = A.reshape(-1, 36); ids = vox.reshape(-1)
    sums = np.zeros((ids.max() + 1, 36)); cnt = np.bincount(ids, minlength=ids.max() + 1)
    np.add.at(sums, ids, flat)
    return (sums / np.maximum(cnt, 1)[:, None])[ids].reshape(A.shape)


def smoke(threads):
    """Two known answers. A single crystal must give A(x) = C everywhere (0 iterations,
    fixes the component permutation). A two-grain laminate with layers normal to x
    must give piecewise-constant fields whose tractions on the interface, the
    xx, xy and xz stresses, are continuous; that one iterates."""
    rng = np.random.default_rng(7)
    C0 = cubic_stiffness()
    rot = lambda: to_mandel4(np.einsum('pi,qj,rk,sl,ijkl->pqrs', *([random_rotation(rng)] * 4), C0))
    C = rot()[None]
    A, iters, _ = solve(np.zeros((16, 16, 16), np.int32), C, "smoke", threads=threads)
    worst = np.abs(A - C[0]).max() / np.abs(C[0]).max()
    print(f"single crystal 16^3: max |A(x) - C| / max|C| = {worst:.2e}   (expect ~1e-12); "
          f"iterations {[i[0] for i in iters]}")
    ok1 = worst < 1e-8
    C2 = np.stack([rot(), rot()])
    vox = np.zeros((16, 16, 16), np.int32); vox[8:] = 1
    A, iters, _ = solve(vox, C2, "smoke_lam", tol=1e-7, threads=threads)
    Eb = np.zeros(6); Eb[0] = 1e-4
    sig = A @ Eb                                              # (n, n, n, 6) ours: xx yy zz xy yz xz
    within = max(sig[:8].std((0, 1, 2)).max(), sig[8:].std((0, 1, 2)).max()) / np.abs(sig).max()
    tr = np.array([0, 3, 5])                                  # xx, xy, xz: continuous across an x-normal interface
    jump = np.abs(sig[:8].mean((0, 1, 2))[tr] - sig[8:].mean((0, 1, 2))[tr]).max() / np.abs(sig).max()
    print(f"laminate 16^3 at tol 1e-7: within-layer spread {within:.2e}, traction jump {jump:.2e}  (both expect < 1e-5); "
          f"iterations {[i[0] for i in iters]}")
    ok2 = within < 1e-5 and jump < 1e-5
    print("PASS" if ok1 and ok2 else "FAIL")


def ladder_dir(material, out=None):
    return out or str(paths.DATA / "ladder" / material)


def run(seeds, grids, threads, tol, maxit, material="Cu", out=None):
    dest = ladder_dir(material, out)
    os.makedirs(SCRATCH, exist_ok=True); os.makedirs(dest, exist_ok=True)
    for s in seeds:
        pts, C = microstructure(s, material=material)
        out = {}
        for n in grids:
            vox = voxelise(pts, n)
            A, iters, log = solve(vox, C, f"{material}_seed{s}_n{n}", tol=tol, maxit=maxit, threads=threads)
            out[f"A{n}"] = A.astype(np.float32); out[f"vox{n}"] = vox
            print(f"seed {s} n={n:>2}: iters {[i[0] for i in iters]} "
                  f"{'ALL CONVERGED' if all(i[1] == 'converged' for i in iters) else 'NOT CONVERGED'} "
                  f"{sum(i[2] for i in iters):.0f} s", flush=True)
        out["C"] = C
        np.savez_compressed(os.path.join(dest, f"seed{s}.npz"), **out)


def analyze(grids, material="Cu", out=None):
    src = ladder_dir(material, out)
    files = sorted(f for f in os.listdir(src) if f.startswith("seed") and f.endswith(".npz"))
    if not files:
        print("nothing to analyze"); return
    fine = max(grids); rows = []
    hdr = None
    for f in files:
        z = np.load(os.path.join(src, f)); Af = z[f"A{fine}"].astype(float); C = z["C"]
        r = {}
        for n in grids:
            A = z[f"A{n}"].astype(float); vox = z[f"vox{n}"]
            Cx = C[vox]
            Af_down = block_mean(Af, fine // n)
            r[f"err_A_{n}"] = np.linalg.norm(A - Af_down) / np.linalg.norm(Af_down) if n != fine else 0.0
            Eb = np.zeros(6); Eb[0] = 1e-4
            sig, sigf = A @ Eb, Af_down @ Eb
            r[f"err_sig_{n}"] = np.linalg.norm(sig - sigf) / np.linalg.norm(sigf) if n != fine else 0.0
            if n != fine:
                pv = np.linalg.norm(A - Af_down, axis=(-2, -1)) / np.linalg.norm(Af_down, axis=(-2, -1))
                b1 = boundary_mask(vox, 1); b2 = boundary_mask(vox, 2)
                r[f"pv_gt5_{n}"] = (pv > 0.05).mean()
                r[f"pv_bnd1_{n}"] = pv[b1].mean(); r[f"pv_depth2plus_{n}"] = pv[~b2].mean()
                r[f"frac_depth2plus_{n}"] = (~b2).mean()
                Gc, Gf = grain_mean(A, vox), grain_mean(Af_down, vox)
                r[f"err_between_{n}"] = np.linalg.norm(Gc - Gf) / np.linalg.norm(Af_down)
                r[f"err_within_{n}"] = np.linalg.norm((A - Gc) - (Af_down - Gf)) / np.linalg.norm(Af_down)
            Afl, Cfl = A.reshape(-1, 36), Cx.reshape(-1, 36)
            r[f"asym_{n}"] = (np.linalg.norm(A - np.swapaxes(A, -1, -2), axis=(-2, -1)) / np.linalg.norm(A, axis=(-2, -1))).mean()
            Cc, Ac = Cfl - Cfl.mean(0), Afl - Afl.mean(0)
            r[f"alpha_{n}"] = ((Cc * Ac).sum(0) / (Cc * Cc).sum(0)).mean()
            r[f"avgstrain_{n}"] = np.linalg.norm(np.linalg.solve(Cx, A).reshape(-1, 6, 6).mean(0) - np.eye(6)) / np.sqrt(6)
        rows.append(r); hdr = hdr or list(r)
    print(f"{len(rows)} seeds, fine grid {fine}^3\n")
    for k in hdr:
        v = np.array([r[k] for r in rows])
        print(f"{k:>20}: mean {v.mean():.4e}  sd {v.std():.1e}")
    coarse = sorted(g for g in grids if g != fine)
    if len(coarse) >= 2:
        e1 = np.mean([r[f"err_A_{coarse[0]}"] for r in rows]); e2 = np.mean([r[f"err_A_{coarse[1]}"] for r in rows])
        print(f"\nerr_A ratio {coarse[0]}^3 / {coarse[1]}^3 = {e1 / e2:.3f}   "
              f"(sqrt(2) = 1.414 for L2 convergence of a field with jumps, 2 for first order)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["smoke", "run", "analyze"])
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(12)))
    ap.add_argument("--grids", type=int, nargs="+", default=[16, 32, 64])
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--tol", type=float, default=1e-5)
    ap.add_argument("--maxit", type=int, default=12000)
    ap.add_argument("--material", default="Cu", choices=list(MATERIALS))
    ap.add_argument("--out", default=None, help="ladder directory; default data/ladder/<material>")
    a = ap.parse_args()
    if a.cmd == "smoke":
        smoke(a.threads)
    elif a.cmd == "run":
        run(a.seeds, a.grids, a.threads, a.tol, a.maxit, a.material, a.out)
    else:
        analyze(a.grids, a.material, a.out)


if __name__ == "__main__":
    main()
