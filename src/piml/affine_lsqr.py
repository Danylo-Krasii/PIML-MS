"""Exact least-squares optimum of the affine translation-invariant kernel of reach k over every
voxel of the training volumes, by conjugate gradients on the normal equations with the circular
convolution as the operator. The trained linear arm is read against this optimum.

    python -m piml.affine_lsqr --which cu --reach 3 --device cpu --volumes 10 --max-iter 50 --threads 2 \
        --out scratch/scores/lsqr_test
    python -m piml.affine_lsqr --check --threads 2

Model: A_hat = conv3d(C, W) + b with W of shape (36, 21, k, k, k) and circular padding (k - 1) / 2,
the function class of periodic.ridge and of the linear arm's composition. Objective: the sum of
squared errors over all voxels and channels of the training volumes, no standardization, so the
error sqrt(sum num / sum den) is the pooled statistic of piml.score and the solution is the class
optimum in that metric. --ridge lam adds lam * tr(X^T X) / (p + 1) * ||W||^2, the weight of
periodic.ridge, on the kernel only: periodic.ridge also penalises the bias and its solution then
carries the mean field along the invariant channel directions, which this solver removes.

The 21 stiffness channels of a rotated cubic crystal span a 9-dimensional affine subspace (12 exact
linear invariants; the centred channel covariance has 12 eigenvalues below 1e-13 of the largest),
so X^T X is singular: the prediction is unique, the kernel only up to those directions. The solver
iterates on the whitened field z = Lp (C - mu) of the varying channels, a right preconditioner that
also removes the mean-direction ill-conditioning, and maps back to (W, b) with W orthogonal to the
invariant directions at every tap. Kernel iterates are float64, field products float32 on --device,
reductions float64; the volumes are processed in batches of --batch. The adjoint of the kernel-to-
field map is the weight gradient of the convolution on the padded input, so <A x, y> = <x, A^T y>
to rounding; --check measures it.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import paths
from .network import to_nchw
from .periodic import load, splits

NOUT = 36
RIDGE_FIT = {"cu": {3: (0.1106, 0.1105)}}   # (train, test) of `periodic ridge --radii 1` on copper


def whitener(C, eps=1e-8):
    """mu (21,), Lp (r, 21) with z = Lp (C - mu) of unit covariance over the voxels of C (S,n,n,n,21),
    the null directions (21 - r, 21) of the centred covariance, and its eigenvalues over the largest."""
    ch = C.shape[-1]
    mu = np.zeros(ch)
    for v in C:
        mu += v.reshape(-1, ch).astype(np.float64).sum(0)
    mu /= C.size // ch
    G = np.zeros((ch, ch))
    for v in C:
        X = v.reshape(-1, ch).astype(np.float64) - mu
        G += X.T @ X
    w, U = np.linalg.eigh(G / (C.size // ch))
    keep = w > eps * w.max()
    return mu, (U[:, keep] / np.sqrt(w[keep])).T, U[:, ~keep].T, w / w.max()


class Affine:
    """x = (V, c) -> conv3d(z, V) + c on the whitened field of C (N, 21, n, n, n), by batches."""

    def __init__(self, C, mu, Lp, k, batch):
        self.C, self.k, self.pad, self.batch = C, k, (k - 1) // 2, batch
        self.mu = torch.as_tensor(mu, dtype=C.dtype, device=C.device)[None, :, None, None, None]
        self.Lp = torch.as_tensor(Lp, dtype=C.dtype, device=C.device)
        self.shape = (NOUT, Lp.shape[0], k, k, k)

    def batches(self):
        return [slice(i, i + self.batch) for i in range(0, len(self.C), self.batch)]

    def field(self, sl):
        z = torch.einsum("ij,njxyz->nixyz", self.Lp, self.C[sl] - self.mu)
        return F.pad(z, (self.pad,) * 6, mode="circular")

    def apply(self, V, c, sl):
        return F.conv3d(self.field(sl), V.to(self.C.dtype)) + c.to(self.C.dtype)[None, :, None, None, None]

    def adjoint(self, R, sl):
        return (torch.nn.grad.conv3d_weight(self.field(sl), self.shape, R).double(),
                R.double().sum((0, 2, 3, 4)))

    def to_kernel(self, V, c):
        """(W, b) on the raw channels: W_d = V_d Lp, b = c - sum_d W_d mu."""
        W = torch.einsum("oixyz,ij->ojxyz", V, self.Lp.double())
        return W, c - torch.einsum("ojxyz,j->o", W, self.mu.double().flatten())

    def to_kernel_t(self, gW, gb):
        """Adjoint of to_kernel."""
        m = self.Lp.double() @ self.mu.double().flatten()
        gV = torch.einsum("ojxyz,ij->oixyz", gW, self.Lp.double()) - (gb[:, None] * m[None])[:, :, None, None, None]
        return gV, gb


def predict(C, W, b, k, batch):
    """conv3d(C, W) + b on the raw channels, float32, one batch of volumes at a time."""
    Wf, bf = W.to(C.dtype), b.to(C.dtype)[None, :, None, None, None]
    for i in range(0, len(C), batch):
        yield i, F.conv3d(F.pad(C[i:i + batch], ((k - 1) // 2,) * 6, mode="circular"), Wf) + bf


def error(C, A, W, b, k, batch):
    """Pooled sqrt(sum num / sum den) over all voxels and channels, float64 reduction."""
    num = den = 0.0
    for i, P in predict(C, W, b, k, batch):
        num += float(((P - A[i:i + batch]).double() ** 2).sum())
        den += float((A[i:i + batch].double() ** 2).sum())
    return np.sqrt(num / den)


def cgls(op, Y, tol, max_iter, lam2=0.0, quiet=False):
    """CGLS on min ||A x - Y||^2 + lam2 ||W||^2 from x = 0, W the kernel of x on the raw channels.

    Returns the iterate with the smallest ||A^T r|| / ||A^T Y||: V, c, its iteration, ||r|| / ||Y||,
    ||A^T r|| / ||A^T Y||, and the number of iterations run; stops early once that residual exceeds
    100 times its best or has not improved for 50 iterations."""
    dev = Y.device
    V = torch.zeros(op.shape, dtype=torch.float64, device=dev)
    c = torch.zeros(NOUT, dtype=torch.float64, device=dev)
    R, Q = Y.clone(), torch.empty_like(Y)
    ynorm = np.sqrt(sum(float((R[sl].double() ** 2).sum()) for sl in op.batches()))
    sV = torch.zeros_like(V); sc = torch.zeros_like(c)
    for sl in op.batches():
        a, b_ = op.adjoint(R[sl], sl); sV += a; sc += b_
    rl = np.sqrt(lam2)
    gW = torch.zeros(NOUT, op.C.shape[1], op.k, op.k, op.k, dtype=torch.float64, device=dev)
    pV, pc = sV.clone(), sc.clone()
    gamma = float((sV ** 2).sum() + (sc ** 2).sum())
    g0 = np.sqrt(gamma)
    t0 = time.time()
    it, rel, nrel = 0, 1.0, 1.0
    bV, bc, bit, brel, bnrel = V.clone(), c.clone(), 0, 1.0, float("inf")
    for it in range(1, max_iter + 1):
        qq = 0.0
        for sl in op.batches():
            Q[sl] = op.apply(pV, pc, sl)
            qq += float((Q[sl].double() ** 2).sum())
        tW = rl * op.to_kernel(pV, pc)[0]
        qq += float((tW ** 2).sum())
        alpha = gamma / qq
        V += alpha * pV; c += alpha * pc
        gW -= alpha * tW
        rr = 0.0
        sV.zero_(); sc.zero_()
        for sl in op.batches():
            R[sl] -= alpha * Q[sl]
            rr += float((R[sl].double() ** 2).sum())
            a, b_ = op.adjoint(R[sl], sl); sV += a; sc += b_
        a, b_ = op.to_kernel_t(rl * gW, torch.zeros_like(c)); sV += a; sc += b_
        gnew = float((sV ** 2).sum() + (sc ** 2).sum())
        beta = gnew / gamma
        gamma = gnew
        pV = sV + beta * pV; pc = sc + beta * pc
        rel, nrel = np.sqrt(rr) / ynorm, np.sqrt(gamma) / g0
        if not quiet:
            print(f"  iter {it:4d}  ||r||/||y|| {rel:.6f}  ||A^T r||/||A^T y|| {nrel:.3e}  "
                  f"{time.time() - t0:.1f} s", flush=True)
        if nrel < bnrel:
            bV.copy_(V); bc.copy_(c)
            bit, brel, bnrel = it, rel, nrel
        if nrel < tol:
            break
        if nrel > 100 * bnrel or it - bit >= 50:
            if not quiet:
                print(f"  stopped at iter {it}: best ||A^T r||/||A^T y|| {bnrel:.3e} at iter {bit}", flush=True)
            break
    return bV, bc, bit, brel, bnrel, it


def data(which, volumes, device):
    """Train (first `volumes` of the 140) and test (40) volumes of the 200-volume cache, NCDHW."""
    C, A = load(which, 200)
    tr, _, te = splits(len(C))
    out = [to_nchw(x[ids]).to(device) for ids in (tr[:volumes], te) for x in (C, A)]
    return out, C[tr[:volumes]]


def solve(a):
    torch.set_num_threads(a.threads)
    t0 = time.time()
    (Ctr, Atr, Cte, Ate), Ctr_np = data(a.which, a.volumes, a.device)
    mu, Lp, _, ev = whitener(Ctr_np)
    del Ctr_np
    op = Affine(Ctr, mu, Lp, a.reach, a.batch)
    p = Ctr.shape[1] * a.reach ** 3
    lam2 = 0.0
    if a.ridge > 0:
        s2 = float((Ctr.double() ** 2).sum())
        lam2 = a.ridge * (a.reach ** 3 * s2 + Ctr[:, 0].numel()) / (p + 1)
    print(f"{a.which} reach {a.reach}: {len(Ctr)} training volumes, {Ctr[:, 0].numel():,} voxels, "
          f"{p + 1:,} coefficients per output row of which {Lp.shape[0] * a.reach ** 3 + 1:,} are "
          f"identifiable (channel rank {Lp.shape[0]}, largest dropped eigenvalue "
          f"{max(ev[ev <= 1e-8], default=0):.1e} of the largest); device {a.device}, batch {a.batch}, "
          f"tol {a.tol:g}, ridge {a.ridge:g}", flush=True)
    V, c, it, rel, nrel, ran = cgls(op, Atr, a.tol, a.max_iter, lam2)
    W, b = op.to_kernel(V, c)
    err_tr = error(Ctr, Atr, W, b, a.reach, a.batch)
    err_te = error(Cte, Ate, W, b, a.reach, a.batch)
    secs = time.time() - t0
    print(f"best iterate {it} of {ran} run, ||r||/||y|| {rel:.6f}, ||A^T r||/||A^T y|| {nrel:.3e}")
    print(f"train error {err_tr:.6f} ({len(Ctr)} volumes, all voxels), test error {err_te:.6f} "
          f"({len(Cte)} volumes), {secs:.0f} s")
    if a.out:
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        np.savez(out / "kernel.npz", W=W.cpu().numpy().astype(np.float32),
                 b=b.cpu().numpy().astype(np.float32))
        with open(out / "summary.json", "w") as f:
            json.dump({"reach": a.reach, "corpus": a.which, "volumes": len(Ctr),
                       "test_volumes": len(Cte), "iterations": it, "iterations_run": ran,
                       "max_iter": a.max_iter,
                       "tol": a.tol, "ridge": a.ridge, "final_residual": rel,
                       "final_normal_residual": nrel, "train_err": err_tr, "test_err": err_te,
                       "channel_rank": int(Lp.shape[0]), "device": a.device, "batch": a.batch,
                       "threads": a.threads, "elapsed": secs, "torch": torch.__version__}, f, indent=1)
        print(f"wrote {out}/")


def check(a):
    torch.set_num_threads(a.threads)
    g = torch.Generator().manual_seed(0)
    print("(a) adjointness, random kernel and fields on a 16^3 grid, 2 volumes")
    for dtype, bound in ((torch.float32, 1e-5), (torch.float64, 1e-12)):
        for k in (3, 5):
            C = torch.randn(2, 21, 16, 16, 16, generator=g).to(dtype)
            op = Affine(C, np.zeros(21), np.eye(21), k, 2)
            V = torch.randn(op.shape, generator=g).double(); c = torch.randn(NOUT, generator=g).double()
            Y = torch.randn(2, NOUT, 16, 16, 16, generator=g).to(dtype)
            Ax = op.apply(V, c, slice(0, 2))
            aV, ac = op.adjoint(Y, slice(0, 2))
            lhs = float((Ax.double() * Y.double()).sum())
            rhs = float((V * aV).sum() + (c * ac).sum())
            d = abs(lhs - rhs) / (float(Ax.double().norm()) * float(Y.double().norm()))
            print(f"    {str(dtype)[6:]} reach {k}: |<Ax,y> - <x,A^T y>| / (|Ax| |y|) = {d:.2e}  "
                  f"({'pass' if d < bound else 'FAIL'}, bound {bound:g})")

    print("(b) exact recovery of a random reach-3 kernel and bias on 4 cache volumes (cu)")
    C, A = load("cu", 200)
    tr, _, te = splits(len(C))
    rng = np.random.default_rng(0)
    ids = np.sort(rng.choice(tr, 8, replace=False))
    Cs = to_nchw(C[ids]).to(a.device)
    Cfit, Cother = Cs[:4], Cs[4:]
    mu, Lp, null, ev = whitener(C[ids[:4]])
    k = 3
    Wt = (torch.randn(NOUT, 21, k, k, k, generator=g) / 1e11).double().to(a.device)
    bt = (10 * torch.randn(NOUT, generator=g)).double().to(a.device)
    Y = torch.cat([P for _, P in predict(Cfit, Wt, bt, k, 4)])
    op = Affine(Cfit, mu, Lp, k, 4)
    print(f"    channel rank {Lp.shape[0]}, eigenvalues over the largest: "
          f"{np.array2string(ev, precision=1, max_line_width=200)}")
    V, c, it, rel, nrel, _ = cgls(op, Y, a.tol, a.max_iter, quiet=True)
    W, b = op.to_kernel(V, c)
    x_rec = torch.cat([W.reshape(NOUT, -1), b[:, None]], 1).cpu().numpy()
    x_true = torch.cat([Wt.reshape(NOUT, -1), bt[:, None]], 1).cpu().numpy()
    N = np.zeros((null.shape[0] * k ** 3, 21 * k ** 3 + 1))
    for j, v in enumerate(null):
        for d in range(k ** 3):
            N[j * k ** 3 + d, np.arange(21) * k ** 3 + d] = v
            N[j * k ** 3 + d, -1] = -v @ mu
    Qn, _ = np.linalg.qr(N.T)
    proj = lambda x: x - (x @ Qn) @ Qn.T
    d_id = np.linalg.norm(proj(x_rec - x_true)) / np.linalg.norm(proj(x_true))
    d_raw = np.linalg.norm(x_rec - x_true) / np.linalg.norm(x_true)
    Yo_t = torch.cat([P for _, P in predict(Cother, Wt, bt, k, 4)]).double()
    Yo_r = torch.cat([P for _, P in predict(Cother, W, b, k, 4)]).double()
    d_pred = float((Yo_r - Yo_t).norm() / Yo_t.norm())
    print(f"    {it} iterations, ||r||/||y|| {rel:.2e}, ||A^T r||/||A^T y|| {nrel:.2e}")
    print(f"    kernel error on the identifiable subspace {d_id:.2e} ({'pass' if d_id < 1e-4 else 'FAIL'}, "
          f"bound 1e-4); raw kernel difference {d_raw:.2e} (null space of {N.shape[0]} directions per "
          f"row); prediction difference on 4 other volumes {d_pred:.2e}")
    del Cs, Cfit, Cother, Y, op

    print("(c) copper reach 3, 20 training volumes, all voxels, against the 1,500-voxel ridge fit of "
          "periodic ridge (train 0.1106 on the first 20 training volumes, test 0.1105)")
    tr_doc, te_doc = RIDGE_FIT["cu"][3]
    Ctr = to_nchw(C[tr[:20]]).to(a.device); Atr = to_nchw(A[tr[:20]]).to(a.device)
    Cte = to_nchw(C[te]).to(a.device); Ate = to_nchw(A[te]).to(a.device)
    mu, Lp, _, _ = whitener(C[tr[:20]])
    del C, A
    op = Affine(Ctr, mu, Lp, k, a.batch)
    t0 = time.time()
    V, c, it, rel, nrel, _ = cgls(op, Atr, a.tol, a.max_iter)
    W, b = op.to_kernel(V, c)
    err_tr = error(Ctr, Atr, W, b, k, a.batch)
    err_te = error(Cte, Ate, W, b, k, a.batch)
    print(f"    {it} iterations in {time.time() - t0:.0f} s, ||r||/||y|| {rel:.6f}, "
          f"||A^T r||/||A^T y|| {nrel:.2e}")
    print(f"    train {err_tr:.6f} against {tr_doc} ({'pass' if err_tr <= tr_doc else 'FAIL'}, at or below); "
          f"test {err_te:.6f} against {te_doc}, difference {err_te - te_doc:+.4f} "
          f"({'pass' if abs(err_te - te_doc) < 0.002 else 'FAIL'}, bound 0.002)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="cu", choices=["cu", "z5", "z8"])
    ap.add_argument("--reach", type=int, default=3, help="receptive field k, odd; (36, 21, k, k, k) kernel")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--volumes", type=int, default=140, help="training volumes used, first of the 140")
    ap.add_argument("--max-iter", type=int, default=500)
    ap.add_argument("--tol", type=float, default=1e-6, help="stop when ||A^T r|| / ||A^T y|| falls below")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--ridge", type=float, default=0.0,
                    help="Tikhonov weight on the kernel, scaled like periodic.ridge (1e-6 there), "
                         "bias unpenalised; 0 is the plain optimum")
    ap.add_argument("--out", default=None)
    ap.add_argument("--check", action="store_true", help="the three known answers, CPU sizes")
    a = ap.parse_args()
    if a.reach % 2 == 0:
        raise SystemExit("--reach must be odd")
    check(a) if a.check else solve(a)


if __name__ == "__main__":
    main()
