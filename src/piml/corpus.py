"""Periodic polycrystal corpus: geometry and orientations in Python, solved with the
MatViz3D FFT solver through the mesh_refine driver, one HDF5 per SVE.

The MatViz3D application cannot load a microstructure, so it cannot reuse one geometry
across resolutions or materials. Here the seed fixes the geometry and the orientations, and
the material is a separate argument, so the three corpora share seeds 1000..1199 and differ
only in c44.

Seed pools, disjoint:
    S  0..63       shared across 16/32/64^3 at a FIXED grain count (--ngrain 1638), so the
                   same geometry is voxelised at every resolution (mesh convergence used 0..7)
    B  1000..2499  independent SVEs at 32^3; the corpora use 1000..1199
    U  100000..    unlabeled, never solved, for volumes such as `piml.infer --generate 100000`

Conventions match mesh_convergence.py: Mandel, component order xx, yy, zz, xy, yz,
xz, A in Pa per unit strain, load magnitude 1e-4, voxels
indexed [x, y, z]. Orientations are Haar-uniform (normalized Gaussian quaternion).

Requires the solver driver built by tools/build_mesh_refine.sh (paths.MESH_REFINE) and the
MatViz3D clone it was compiled against (paths.MATVIZ3D), whose fft_homog.hpp enters the
solver hash stored in every file.

Usage:  python -m piml.corpus gen   --pool B --first 1000 --count 200 --n 32 --material Cu --out DIR [--workers 12]
        python -m piml.corpus gen   --pool S --first 0 --count 8 --n 64 --ngrain 1638 --material Cu --out DIR
        python -m piml.corpus check DIR
        python -m piml.corpus time  --n 32 --material Cu          # one single-thread solve, the cost figure
"""

import argparse
import hashlib
import json
import os
import sys
import time
from multiprocessing import Pool

import h5py
import numpy as np
from . import paths

from . import mesh_convergence as mc
from .microstructure import cubic_stiffness, to_mandel4

MATERIALS = {                                    # GPa, room temperature
    "Cu": (168.4, 121.4, 75.4),
}
POOLS = {"S": (0, 64), "B": (1000, 2500), "U": (100000, 10 ** 9)}
VERSION = "corpus.py 2026-09-08"


def zener(c11, c12, c44):
    return 2 * c44 / (c11 - c12)


def add_material(name, c11, c12, c44):
    MATERIALS[name] = (float(c11), float(c12), float(c44))


def quat_to_matrix(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def microstructure(seed, ngrain, material="Cu"):
    """Same random stream as mesh_convergence.microstructure, so pool S seeds reproduce
    the mesh convergence runs; returns the quaternions as well and takes the material as data."""
    rng = np.random.default_rng(seed)
    pts = rng.random((ngrain, 3))
    q = rng.normal(size=(ngrain, 4)); q /= np.linalg.norm(q, axis=1, keepdims=True)
    c11, c12, c44 = MATERIALS[material]
    C0 = cubic_stiffness(c11 * 1e9, c12 * 1e9, c44 * 1e9)
    C = np.stack([to_mandel4(np.einsum('pi,qj,rk,sl,ijkl->pqrs', g, g, g, g, C0))
                  for g in (quat_to_matrix(qq) for qq in q)])
    return pts, q, C


def reference_moduli(C):
    """The solver's reference medium, replicated from fft_homog.hpp pick_reference: the
    midpoint of the per-grain isotropic estimates, raised so that 2 C0 - C(x) stays positive
    definite (the C++ uses a 64-step power iteration for the largest eigenvalue; this uses
    the exact value, so the two agree unless the safeguard fires). Recorded because the Born
    iterates depend on it. Returns (lambda0, mu0, safeguard_fired)."""
    mu = 0.25 * (C[:, 3, 3] + C[:, 4, 4] + C[:, 5, 5])
    lam = (C[:, 0, 1] + C[:, 0, 2] + C[:, 1, 2]) / 3.0
    mu0 = 0.5 * (mu.min() + mu.max()); lam0 = 0.5 * (lam.min() + lam.max())
    lam_max = max(np.linalg.eigvalsh(c).max() for c in C)
    need = 0.55 * lam_max; have = min(3 * lam0 + 2 * mu0, 2 * mu0)
    fired = 0.0 < have < need
    if fired:
        k = need / have; mu0 *= k; lam0 *= k
    return float(lam0), float(mu0), bool(fired)


def solver_hash():
    h = hashlib.sha256()
    for p in (mc.DRIVER, str(paths.MATVIZ3D / "fft_homog.hpp")):
        with open(p, "rb") as f:
            h.update(f.read())
    return h.hexdigest()[:16]


def one(args):
    seed, pool, n, material, out, tol, maxit, threads, ngrain = args
    path = os.path.join(out, f"sve_{seed}.hdf5")
    if os.path.exists(path):
        return seed, "exists", 0.0
    ngrain = ngrain or n ** 3 // 20
    pts, q, C = microstructure(seed, ngrain, material)
    vox = mc.voxelise(pts, n)
    t0 = time.time()
    A, iters, _ = mc.solve(vox, C, f"work_{seed}", tol=tol, maxit=maxit, threads=threads)
    secs = time.time() - t0
    conv = all(i[1] == "converged" for i in iters)
    c11, c12, c44 = MATERIALS[material]
    lam0, mu0, fired = reference_moduli(C)
    E = np.zeros((6, 6)); np.fill_diagonal(E, 1e-4)
    tmp = path + ".part"
    with h5py.File(tmp, "w") as f:
        f["voxels"] = vox
        f["quat"] = q
        f["C_grain"] = C
        f.create_dataset("A", data=A.astype(np.float32), compression="gzip", compression_opts=1)
        f["sigma_avg"] = (A @ E).mean((0, 1, 2))
        f["iters"] = np.array([i[0] for i in iters]); f["secs"] = np.array([i[2] for i in iters])
        f.attrs.update(dict(seed=seed, pool=pool, n=n, ngrain=ngrain, material=material,
                            c11_GPa=c11, c12_GPa=c12, c44_GPa=c44, zener=zener(c11, c12, c44),
                            tol=tol, maxit=maxit, load=1e-4, converged=bool(conv),
                            order="Mandel xx,yy,zz,xy,yz,xz; A[x,y,z,i,k] = sigma_i per unit E_k, Pa",
                            orientation="quat (w,x,y,z), active rotation g; C_grain = g g g g : C0",
                            lambda0_Pa=lam0, mu0_Pa=mu0, reference_safeguard_fired=fired,
                            solver=solver_hash(), generator=VERSION, threads=threads))
    os.replace(tmp, path if conv else path.replace(".hdf5", ".UNCONVERGED.hdf5"))
    d = os.path.join(mc.SCRATCH, f"work_{seed}")
    for fn in ("phase.i32", "C.f64", *[f"out.{k}.txt" for k in range(6)]):
        try: os.remove(os.path.join(d, fn))
        except FileNotFoundError: pass
    try: os.rmdir(d)
    except OSError: pass
    return seed, "converged" if conv else "NOT CONVERGED", secs


def gen(a):
    os.makedirs(a.out, exist_ok=True)
    lo, hi = POOLS[a.pool]
    seeds = list(range(a.first, a.first + a.count))
    assert lo <= seeds[0] and seeds[-1] < hi, f"seeds outside pool {a.pool} range {lo}..{hi - 1}"
    with open(os.path.join(a.out, "manifest.json"), "a") as f:
        f.write(json.dumps(dict(pool=a.pool, seeds=[seeds[0], seeds[-1]], n=a.n, ngrain=a.ngrain or a.n ** 3 // 20, material=a.material,
                                tol=a.tol, maxit=a.maxit, solver=solver_hash(), generator=VERSION,
                                started=time.strftime("%Y-%m-%d %H:%M:%S"))) + "\n")
    jobs = [(s, a.pool, a.n, a.material, a.out, a.tol, a.maxit, a.threads, a.ngrain) for s in seeds]
    t0 = time.time(); done = 0; tot = 0.0
    with Pool(a.workers) as p:
        for seed, status, secs in p.imap_unordered(one, jobs):
            done += 1; tot += secs
            print(f"seed {seed}: {status} {secs:.0f} s   [{done}/{len(seeds)}, {time.time() - t0:.0f} s wall, "
                  f"{tot / done:.0f} s per SVE per worker]", flush=True)


def check(d):
    files = sorted(f for f in os.listdir(d) if f.endswith(".hdf5"))
    print(f"{len(files)} files in {d}")
    M = np.zeros((3, 3)); G = np.zeros(3); ng = 0; cosPhi = []; wrap = []; inner = []
    worst = dict(periodic=0.0, avgstrain=0.0, sym=0.0, unconv=0)
    for fn in files:
        with h5py.File(os.path.join(d, fn), "r") as f:
            vox = f["voxels"][...]; q = f["quat"][...]; A = f["A"][...].astype(float); C = f["C_grain"][...]
            worst["unconv"] += 0 if f.attrs["converged"] else 1
        # periodicity: same-grain adjacency across the wrap must match interior adjacency
        same_in = np.mean([(vox[:-1] == vox[1:]).mean(), (vox[:, :-1] == vox[:, 1:]).mean(), (vox[:, :, :-1] == vox[:, :, 1:]).mean()])
        same_wrap = np.mean([(vox[0] == vox[-1]).mean(), (vox[:, 0] == vox[:, -1]).mean(), (vox[:, :, 0] == vox[:, :, -1]).mean()])
        worst["periodic"] = max(worst["periodic"], abs(same_wrap - same_in)); wrap.append(same_wrap); inner.append(same_in)
        g = np.stack([quat_to_matrix(qq) for qq in q])
        M += np.einsum('gi,gj->ij', g[:, :, 0], g[:, :, 0]); G += g[:, :, 0].sum(0); ng += len(g)
        cosPhi.append(g[:, 2, 2])
        Cx = C[vox]
        worst["avgstrain"] = max(worst["avgstrain"], np.linalg.norm(np.linalg.solve(Cx, A).reshape(-1, 6, 6).mean(0) - np.eye(6)) / np.sqrt(6))
        Am = A.mean((0, 1, 2)); worst["sym"] = max(worst["sym"], np.linalg.norm(Am - Am.T) / np.linalg.norm(Am))
    M /= ng; G /= ng; cosPhi = np.concatenate(cosPhi)
    # KS on the empirical CDF. Comparing sorted values against uniform quantiles instead
    # returns exactly twice this, because cos Phi lives on [-1, 1] and carries density 1/2,
    # and would then fail the 1.36/sqrt(n) critical value spuriously.
    F = (np.sort(cosPhi) + 1) / 2
    m = len(cosPhi)
    ks = max(np.abs(F - np.arange(1, m + 1) / m).max(), np.abs(F - np.arange(m) / m).max())
    print(f"grains {ng}: <(g e1)(g e1)^T> - I/3 max |.| = {np.abs(M - np.eye(3) / 3).max():.4f} (expect ~{1 / np.sqrt(ng):.4f}), "
          f"|<g e1>| = {np.linalg.norm(G):.4f} (expect ~{1 / np.sqrt(ng):.4f}), KS(cos Phi vs uniform) = {ks:.4f} (expect ~{1.36 / np.sqrt(ng):.4f})")
    d = np.array(wrap) - np.array(inner)
    print(f"periodic same-grain adjacency, wrap minus interior: pooled mean {d.mean():+.4f} +- {d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float('nan'):.4f} "
          f"(expect 0 within the error), per-file worst |diff| {worst['periodic']:.4f} (bound 0.05)")
    print(f"||<C^-1 A> - I||/sqrt6 worst = {worst['avgstrain']:.1e} (bound 1e-8, float32 storage); <A> asymmetry worst = {worst['sym']:.1e}; "
          f"unconverged files {worst['unconv']} (bound 0)")


def timing(a):
    pts, q, C = microstructure(0, a.ngrain or a.n ** 3 // 20, a.material)
    vox = mc.voxelise(pts, a.n)
    lam0, mu0, fired = reference_moduli(C)
    print(f"reference medium (replicated rule): lambda0 {lam0 / 1e9:.2f} GPa, mu0 {mu0 / 1e9:.2f} GPa, safeguard fired {fired}")
    t0 = time.time(); A, iters, _ = mc.solve(vox, C, "timing", tol=a.tol, maxit=a.maxit, threads=1); secs = time.time() - t0
    print(f"{a.material} {a.n}^3 single thread: {secs:.0f} s per SVE, iterations {[i[0] for i in iters]}, "
          f"{[f'{i[2]:.0f}' for i in iters]} s per load")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["gen", "check", "time"])
    ap.add_argument("dir", nargs="?")
    ap.add_argument("--pool", default="B"); ap.add_argument("--first", type=int, default=1000); ap.add_argument("--count", type=int, default=200)
    ap.add_argument("--n", type=int, default=32); ap.add_argument("--ngrain", type=int, default=0, help="fixed grain count; default n^3/20")
    ap.add_argument("--material", default="Cu")
    ap.add_argument("--c", type=float, nargs=3, metavar=("C11", "C12", "C44"), help="GPa, defines --material if not in the table")
    ap.add_argument("--out"); ap.add_argument("--workers", type=int, default=12); ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--tol", type=float, default=1e-5); ap.add_argument("--maxit", type=int, default=6000)
    a = ap.parse_args()
    if a.c: add_material(a.material, *a.c)
    if a.cmd == "gen": gen(a)
    elif a.cmd == "check": check(a.dir)
    else: timing(a)


if __name__ == "__main__":
    main()
