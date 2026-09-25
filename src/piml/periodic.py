"""Training on the periodic corpora, and the ridge baseline.

Padding is circular, so every voxel on a face sees its true neighbour on the opposite face, and
the test error is taken over the whole cell. Each run appends one row to logs/results.jsonl and
saves its best-validation weights under checkpoints/.

    python -m piml.periodic sweep   --which cu --rf 13 --width 64 --augment --norm none \
        --max-steps 36000 --patience 6000 --sched-patience 20 --seeds 0 1 2 [--linear] [--project-train]
    python -m piml.periodic ridge   --which cu --radii 1 2
    python -m piml.periodic time    --rf 13
    python -m piml.periodic compare --rf 7 13 19 --seeds 0 1
    python -m piml.periodic norm    --rf 13 --seeds 0 1

`sweep` trains one arm across receptive fields and is the command behind the released
checkpoints. `compare` trains both arms at their original widths (linear 64, nonlinear 32), and
`norm` compares the three normalizations of the nonlinear arm.

REFS holds the untrained references per corpus, each on that corpus's 40 test volumes: the
constant tensor, the Taylor field A = C(x), and the Born iterates of orders 1 to 3 (`piml.born`).
"""

import argparse
import datetime as _dt
import json
import socket
import subprocess
import time as _time

import h5py
import numpy as np

from . import paths
from .microstructure import upper_triangle

_UNSET = object()
_COMMIT = _UNSET

# Untrained references per corpus, measured on that corpus's own 40 test volumes at its own
# optimal reference medium. Reading z8 against copper's numbers understates all of them.
REFS = {
    "cu": {"floor": 0.2097, "taylor": 0.2062, 1: 0.0546, 2: 0.0251, 3: 0.0162},
    "z5": {"floor": 0.3297, "taylor": 0.3487, 1: 0.1086, 2: 0.0527, 3: 0.0330},
    "z8": {"floor": 0.4604, "taylor": 0.5399, 1: 0.1954, 2: 0.1062, 3: 0.0686},
}
BORN = REFS["cu"]
FLOOR, TAYLOR = REFS["cu"]["floor"], REFS["cu"]["taylor"]
RESULTS = "results.jsonl"


def _commit():
    """Short git revision, or None outside a checkout. Resolved once."""
    global _COMMIT
    if _COMMIT is _UNSET:
        try:
            _COMMIT = subprocess.run(["git", "-C", str(paths.ROOT), "rev-parse", "--short", "HEAD"],
                                     capture_output=True, text=True, timeout=5,
                                     check=True).stdout.strip() or None
        except Exception:
            _COMMIT = None
    return _COMMIT


def record(**row):
    """Append one finished run to logs/results.jsonl immediately.

    One row per run, written as the run ends, so that losing a machine costs one run and
    not a whole grid.

    Rows carry the time and the code revision because the file has no other way to tell
    two runs of the same configuration apart.
    """
    row.setdefault("ts", _dt.datetime.now().astimezone().isoformat(timespec="seconds"))
    row.setdefault("commit", _commit())
    row.setdefault("host", socket.gethostname())
    row.setdefault("torch", __import__("torch").__version__)
    f = paths.ensure(paths.LOGS) / RESULTS
    with open(f, "a") as fh:
        fh.write(json.dumps(row) + "\n")
CORPUS = {"cu": "periodic32/cu", "z5": "periodic32/z5", "z8": "periodic32/z8"}


def load(which="cu", count=200, first=1000):
    """C as (S,n,n,n,21) and A as (S,n,n,n,36), cached."""
    cache = paths.SCRATCH / f"periodic_{which}_{first}_{count}.npz"
    if cache.exists():
        z = np.load(cache)
        return z["C"], z["A"]
    d = paths.DATA / CORPUS[which]
    Cs, As = [], []
    for s in range(first, first + count):
        with h5py.File(d / f"sve_{s}.hdf5", "r") as f:
            vox = f["voxels"][...]
            Cg = f["C_grain"][...]
            A = f["A"][...].astype(np.float32)
        Cs.append(upper_triangle(Cg[vox]).astype(np.float32))
        As.append(A.reshape(A.shape[:3] + (36,)))
    C, A = np.stack(Cs), np.stack(As)
    paths.ensure(cache.parent)
    np.savez(cache, C=C, A=A)
    return C, A


def splits(n, n_test=40, n_val=20):
    """Grouped by volume, fixed by index so a seed changes only initialization."""
    if n <= n_test + n_val:
        raise ValueError(f"{n} volumes cannot fill {n_test} test + {n_val} val and leave a train set")
    i = np.arange(n)
    return i[:n - n_test - n_val], i[n - n_test - n_val:n - n_test], i[n - n_test:]


def train(C, A, rf, tr, va, te, width=64, linear=True, norm="none", seed=0,
          max_steps=12000, patience_steps=3000, lr=1e-3, batch=8, dev="cuda", quiet=False,
          grad_clip=1.0, sched_patience=5, tag=None, rewind=3.0, max_rewinds=5,
          augment=False, lam=0.0, balance="none", balance_every=100, balance_alpha=0.9,
          balance_lr=None, project_train=False):
    """Returns a dict of results. Scored on the whole cell.

    Keys: err, params, steps, secs, best_step, nonfinite, rewinds, clipped, ckpt, and the recipe
    (patience_steps, sched_patience, lr, grad_clip, rewind_threshold, max_rewinds). `steps` is the loop length
    and `best_step` the step the returned weights were taken at; for an early-stopped run
    they differ by patience_steps.

    With augment=True each batch is acted on by a random element of the 48-element cube
    symmetry group, applied to the RAW fields before standardization since standardization
    is not equivariant, with the statistics equalized over the group's channel orbits. The
    action is exact on a periodic cell.

    With lam > 0 the loss gains lam times the batch mean of the squared spectral equilibrium
    ratio of the de-standardized prediction, taken after undoing the
    batch's group element. Validation and early stopping stay on data error alone.

    `balance` replaces the fixed lam by a rule that sets the weight during training:
    "uncertainty" trains two log-variances s = (s_d, s_p), loss = e^-s_d mse + e^-s_p r2 + s_d
    + s_p, in a second optimizer group at balance_lr (ten times lr when None); "gradnorm" sets
    the weight to ||grad mse|| / ||grad r2|| every balance_every steps, smoothed with
    balance_alpha, the first measurement taken as is. With project_train the de-standardized
    prediction is projected onto spectral equilibrium in the data frame and the data loss, the
    validation error and the test error are taken on the projected field; no penalty is added.
    The rewind guard, early stopping and the scheduler read the data error in every mode.
    """
    import torch
    import torch.nn as nn
    from .network import LocalizationNet, Octahedral, to_nchw
    if lam > 0 or balance != "none" or project_train:
        from .physics import (a36_to_cols, cols_to_a36, equilibrium_ratio_sq, project_equilibrium,
                              undo_octahedral)
    if balance not in ("none", "uncertainty", "gradnorm"):
        raise ValueError(f"balance {balance!r} is not none, uncertainty or gradnorm")

    torch.manual_seed(seed)
    np.random.seed(seed)
    X, Y = to_nchw(C).to(dev), to_nchw(A).to(dev)
    oct_ = Octahedral(dev=dev) if augment else None
    if oct_ is not None:
        xm, xs = Octahedral.orbit_stats(X[tr], oct_.orbits21)
        ym, ys = Octahedral.orbit_stats(Y[tr], oct_.orbits36)
    else:
        xm = X[tr].mean((0, 2, 3, 4), keepdim=True); xs = X[tr].std((0, 2, 3, 4), keepdim=True) + 1e-8
        ym = Y[tr].mean((0, 2, 3, 4), keepdim=True); ys = Y[tr].std((0, 2, 3, 4), keepdim=True) + 1e-8
    Xn = (X - xm) / xs

    net = LocalizationNet((rf - 1) // 2, width, cin=C.shape[-1], linear=linear,
                          padding_mode="circular", norm=norm).to(dev)
    npar = sum(p.numel() for p in net.parameters())
    groups = [{"params": net.parameters()}]
    s = None
    if balance == "uncertainty":
        s = torch.zeros(2, device=dev, requires_grad=True)
        groups.append({"params": [s], "lr": 10 * lr if balance_lr is None else balance_lr})
    opt = torch.optim.AdamW(groups, lr=lr, weight_decay=0.0)
    # ReduceLROnPlateau halves on the call AFTER patience consecutive non-improving ones,
    # so it needs (sched_patience + 1) * 100 stalled steps. Early stopping must outlive that
    # or the schedule never fires. min_lr stops the rate decaying to nothing.
    need = (sched_patience + 1) * 100
    if patience_steps <= need:
        raise ValueError(f"early stop at {patience_steps} stalled steps pre-empts the "
                         f"learning-rate schedule, which needs {need}")
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5,
                                                       patience=sched_patience, min_lr=1e-5)

    def project(p):
        return torch.cat([cols_to_a36(project_equilibrium(a36_to_cols(p[i:i + batch])))
                          for i in range(0, len(p), batch)])

    def score(ids):
        net.eval()
        with torch.no_grad():
            p = net(Xn[ids]) * ys + ym
            if project_train:
                p = project(p)
            return (torch.linalg.vector_norm(p - Y[ids]) / torch.linalg.vector_norm(Y[ids])).item()

    trace = None
    if tag is not None and (balance != "none" or project_train):
        d = paths.ensure(paths.LOGS / "traces")
        trace = d / f"{tag}.jsonl"
        k = 1
        while trace.exists():
            trace = d / f"{tag}-{k}.jsonl"
            k += 1

    nonfinite, clipped = 0, 0
    best, best_state, bad, step, stop = float("inf"), None, 0, 0, False
    best_step, rewinds = 0, 0
    last_eq, last_mse, last_r2, lam_eff, best_lam, best_s = None, None, None, None, None, None
    pars = [q for q in net.parameters() if q.requires_grad]
    t0 = _time.time()
    while step < max_steps and not stop:
        net.train()
        for i in range(0, len(tr), batch):
            b = np.random.choice(tr, size=min(batch, len(tr)), replace=False)
            if oct_ is not None:                       # act on raw fields, then standardize
                g = int(np.random.randint(48))
                xb, yb = oct_.apply(X[b], Y[b], g)
                xb = (xb - xm) / xs; yb = (yb - ym) / ys
            else:
                xb, yb = Xn[b], (Y[b] - ym) / ys
            opt.zero_grad(set_to_none=True)
            out = net(xb)
            if project_train:
                p = out * ys + ym
                if oct_ is not None:
                    p = undo_octahedral(oct_, p, g)
                loss = nn.functional.mse_loss((project(p) - ym) / ys, (Y[b] - ym) / ys)
            else:
                loss = nn.functional.mse_loss(out, yb)
            mse = loss
            if lam > 0 or balance != "none":
                p = out * ys + ym
                if oct_ is not None:
                    p = undo_octahedral(oct_, p, g)
                r2 = equilibrium_ratio_sq(a36_to_cols(p))
                r2m = r2.mean()
                if balance == "uncertainty":
                    loss = torch.exp(-s[0]) * mse + torch.exp(-s[1]) * r2m + s.sum()
                    lam_eff = torch.exp(s[0] - s[1]).detach()
                elif balance == "gradnorm":
                    if step % balance_every == 0:
                        gd = torch.cat([t.flatten() for t in torch.autograd.grad(mse, pars, retain_graph=True)])
                        gp = torch.cat([t.flatten() for t in torch.autograd.grad(r2m, pars, retain_graph=True)])
                        lam_hat = (torch.linalg.vector_norm(gd)
                                   / torch.linalg.vector_norm(gp).clamp_min(1e-30)).item()
                        lam_eff = (lam_hat if lam_eff is None
                                   else balance_alpha * lam_eff + (1 - balance_alpha) * lam_hat)
                    loss = mse + lam_eff * r2m
                else:
                    loss = mse + lam * r2m
                last_eq = r2.detach().sqrt().mean()
                last_r2 = r2m.detach()
            last_mse = mse.detach()
            loss.backward()
            gn = nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
            if torch.isfinite(gn):
                opt.step()
                if gn > grad_clip:
                    clipped += 1
            else:
                nonfinite += 1
            step += 1
            if step % 100 == 0:
                v = score(va)
                # A loss spike leaves the weights worse than the saved checkpoint, and
                # the scheduler makes it worse: it reads the spike as a plateau and shrinks the
                # step, so the climb back is slower than the fall. Go back to the best point and
                # take smaller steps from there. It is a no-op for a run that never spikes.
                if (best_state is not None and rewinds < max_rewinds
                        and (not np.isfinite(v) or v > rewind * best)):
                    net.load_state_dict(best_state)
                    for pg in opt.param_groups:
                        pg["lr"] = max(pg["lr"] * 0.5, 1e-5)
                    rewinds += 1
                    if not quiet:
                        print(f"      step {step:6d}: val {v:.4g} is {v/best:.1f}x best {best:.4f}; "
                              f"rewound to best, lr now {opt.param_groups[0]['lr']:.2e} "
                              f"({rewinds}/{max_rewinds})", flush=True)
                    bad = 0
                    continue
                sched.step(v)
                if v < best - 1e-5:
                    best, bad, best_step = v, 0, step
                    best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
                    best_lam = float(lam_eff) if lam_eff is not None else None
                    best_s = s.detach().tolist() if s is not None else None
                else:
                    bad += 100
                if step % 500 == 0:
                    w = float(lam_eff) if lam_eff is not None else None
                    if not quiet:
                        el = _time.time() - t0
                        rem = (max_steps - step) * el / step
                        stall = bad
                        cur_lr = opt.param_groups[0]["lr"]
                        print(f"      step {step:6d}/{max_steps}  val {v:.4f}  best {best:.4f}  "
                              f"lr {cur_lr:.2e}  {el/60:.0f} min elapsed, <= {rem/60:.0f} min left, "
                              f"{stall}/{patience_steps} steps since last gain"
                              + (f"  train eq ratio {last_eq.item():.3e}" if last_eq is not None else "")
                              + (f"  w {w:.3e}" if w is not None else "")
                              + (f"  s {s[0].item():.3f} {s[1].item():.3f}" if s is not None else ""),
                              flush=True)
                    if trace is not None:
                        with open(trace, "a") as fh:
                            fh.write(json.dumps({"step": step, "val": v, "mse": float(last_mse),
                                                 "r2": float(last_r2) if last_r2 is not None else None,
                                                 "lam_eff": w,
                                                 "s": s.detach().tolist() if s is not None else None,
                                                 "lr": opt.param_groups[0]["lr"]}) + "\n")
                if bad >= patience_steps:
                    stop = True; break
    net.load_state_dict(best_state)
    if nonfinite and not quiet:
        print(f"      {nonfinite} non-finite gradient steps skipped", flush=True)
    ckpt = None
    if tag is not None:
        # The tag names the configuration only in part, so two runs at different budgets or
        # learning rates would collide. Never overwrite; the row records the path actually used.
        d = paths.ensure(paths.CHECKPOINTS)
        ckpt = d / f"{tag}.pt"
        k = 1
        while ckpt.exists():
            ckpt = d / f"{tag}-{k}.pt"
            k += 1
        torch.save({"state": best_state, "rf": rf, "width": width, "linear": linear,
                    "norm": norm, "seed": seed, "cin": C.shape[-1], "best_step": best_step,
                    "xm": xm.cpu(), "xs": xs.cpu(), "ym": ym.cpu(), "ys": ys.cpu(),
                    "balance": balance, "lam_eff": best_lam, "project_train": project_train}, ckpt)
    extra = {}
    if balance != "none":
        extra.update(balance=balance, balance_every=balance_every, balance_alpha=balance_alpha,
                     balance_lr=opt.param_groups[-1]["lr"] if s is not None else None,
                     lam_eff=best_lam, s_data=best_s[0] if best_s else None,
                     s_phys=best_s[1] if best_s else None)
    if lam > 0 or balance != "none":
        extra["eq_ratio_train"] = float(last_eq) if last_eq is not None else None
    if project_train:
        extra["project_train"] = True
    return {"err": score(te), "params": npar, "steps": step, "secs": _time.time() - t0,
            "best_step": best_step, "nonfinite": nonfinite, "rewinds": rewinds, "clipped": clipped,
            "patience_steps": patience_steps, "sched_patience": sched_patience, "lr": lr,
            "grad_clip": grad_clip, "rewind_threshold": rewind, "max_rewinds": max_rewinds,
            "ckpt": (str(ckpt.relative_to(paths.ROOT)) if ckpt.is_relative_to(paths.ROOT)
                     else str(ckpt)) if ckpt else None, **extra}


def timing(a):
    C, A = load(a.which, a.count)
    print(f"corpus {a.which}: {C.shape[0]} volumes at {C.shape[1]}^3, "
          f"{C.nbytes/1e9:.2f} GB inputs + {A.nbytes/1e9:.2f} GB targets in fp32")
    tr, va, te = splits(len(C))
    import torch
    rf0 = a.rf[0] if isinstance(a.rf, list) else a.rf
    arms = ([(rf0, a.width, True), (rf0, a.width, False)] if getattr(a, "width", None)
            else [(rf0, 64, True), (rf0, 32, False)])
    for rf, w, lin in arms:
        torch.cuda.reset_peak_memory_stats()
        r = train(C, A, rf, tr, va, te, width=w, linear=lin, norm="none",
                  max_steps=a.steps, patience_steps=10 ** 9, augment=a.augment)
        e, npar, steps, secs = r["err"], r["params"], r["steps"], r["secs"]
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"  {'linear' if lin else 'nonlinear':>9} RF {rf} width {w}: {steps} steps in "
              f"{secs:.0f} s ({1000*secs/steps:.0f} ms/step), peak {peak:.2f} GB, {npar:,} params, "
              f"test {e:.4f}")


def compare(a):
    C, A = load(a.which, a.count)
    tr, va, te = splits(len(C))
    print(f"{a.which}: {len(tr)} train / {len(va)} val / {len(te)} test volumes, circular padding, "
          f"whole-cell scoring, norm={a.norm}, augment={a.augment}")
    r_ = REFS[a.which]
    print(f"untrained Born references on {a.which}: order 1 {r_[1]:.4f}, order 2 {r_[2]:.4f}, "
          f"order 3 {r_[3]:.4f}\n")
    print(f"  {'RF':>3} {'arm':>10} {'test rel.err':>13} {'params':>11} {'vs Born-2':>10}")
    total = len(a.rf) * 2 * len(a.seeds)
    done = 0
    for rf in a.rf:
        for lin, w in [(True, 64), (False, 32)]:
            es = []
            for s in a.seeds:
                arm = 'linear' if lin else 'nonlinear'
                print(f"  [{done+1}/{total}] starting {arm} RF {rf} width {w} seed {s}", flush=True)
                r = train(C, A, rf, tr, va, te, width=w, linear=lin, norm=a.norm, seed=s,
                          quiet=False, augment=a.augment,
                          tag=f"{a.which}-{arm}-rf{rf}-w{w}-s{s}-{a.norm}")
                done += 1
                e, npar, steps, secs = r["err"], r["params"], r["steps"], r["secs"]
                record(arm=arm, which=a.which, rf=rf, seed=s, width=w, norm=a.norm,
                       augment=a.augment, **r)
                print(f"  [{done}/{total}] {arm} RF {rf} seed {s}: {e:.4f} "
                      f"after {steps} steps in {secs/60:.0f} min", flush=True)
                es.append(e)
            m, sd = float(np.mean(es)), float(np.std(es))
            tag = "beats" if m < r_[2] else "loses to"
            print(f"  ROW {rf:>3} {'linear' if lin else 'nonlinear':>10} {m:>8.4f} +-{sd:.4f} "
                  f"{npar:>11,} {tag:>10} Born-2\n", flush=True)


def norm_ablation(a):
    """The leak test. InstanceNorm sees the whole cell; the other two do not."""
    C, A = load(a.which, a.count)
    tr, va, te = splits(len(C))
    print(f"nonlinear arm at RF {a.rf}, width 32, {len(a.seeds)} seeds. InstanceNorm statistics span")
    print("the whole cell, so it is not a matched-receptive-field model. The other two are.\n")
    print(f"  {'norm':>10} {'test rel.err':>14}")
    for norm in ["instance", "voxel", "none"]:
        es = [train(C, A, a.rf, tr, va, te, width=32, linear=False, norm=norm, seed=s, quiet=True,
                    augment=a.augment)["err"] for s in a.seeds]
        print(f"  {norm:>10} {np.mean(es):>9.4f} +-{np.std(es):.4f}", flush=True)


def sweep(a):
    """One arm across receptive fields.

    Born-1 applies the Green operator over the whole cell, so comparing it against a
    13-voxel kernel measures reach rather than training. At RF 31 on a 32 cubed cell the
    network finally has whole-cell reach and the comparison is matched.
    """
    C, A = load(a.which, a.count)
    tr, va, te = splits(len(C))
    arm = "linear" if a.linear else "nonlinear"
    w = a.width if getattr(a, "width", None) else (64 if a.linear else 32)
    ref = REFS[a.which]
    print(f"{arm} arm, {a.which}, reach sweep. augment={a.augment}, norm={a.norm}")
    if a.which == "cu":
        print("exact linear optima measured separately: RF 3 0.1105, RF 5 0.0850, RF 7 0.0733")
    print(f"untrained on {a.which}: constant {ref['floor']:.4f}, Taylor {ref['taylor']:.4f}, "
          f"Born-1 {ref[1]:.4f}, Born-2 {ref[2]:.4f}\n")
    for rf in a.rf:
        for s_ in a.seeds:
            print(f"  starting {arm} RF {rf} seed {s_}", flush=True)
            kw = {}
            if getattr(a, "max_steps", None):
                kw["max_steps"] = a.max_steps
            if getattr(a, "patience", None):
                kw["patience_steps"] = a.patience
            if getattr(a, "sched_patience", None):
                kw["sched_patience"] = a.sched_patience
            lam = getattr(a, "lam", 0.0) or 0.0
            bal = getattr(a, "balance", "none") or "none"
            proj = bool(getattr(a, "project_train", False))
            suffix = ((f"-lam{lam:g}" if lam > 0 else "")
                      + {"uncertainty": "-unc", "gradnorm": "-gn"}.get(bal, "")
                      + ("-proj" if proj else ""))
            r = train(C, A, rf, tr, va, te, width=w, linear=a.linear, norm=a.norm, seed=s_,
                      quiet=False, augment=a.augment, lam=lam, balance=bal,
                      balance_every=getattr(a, "balance_every", 100),
                      balance_alpha=getattr(a, "balance_alpha", 0.9),
                      balance_lr=getattr(a, "balance_lr", None), project_train=proj, **kw,
                      tag=f"{a.which}-{arm}-rf{rf}-w{w}-s{s_}-{a.norm}" + suffix)
            e, steps, secs = r["err"], r["steps"], r["secs"]
            record(arm=arm, which=a.which, rf=rf, seed=s_, width=w, norm=a.norm,
                   augment=a.augment, lam=lam, **({"budget": a.max_steps} if kw.get("max_steps") else {}), **r)
            print(f"  DONE {arm} RF {rf} seed {s_}: {e:.4f}  ({steps} steps, {secs/60:.0f} min, "
                  f"{'below' if e < ref[1] else 'above'} Born-1)\n", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["time", "compare", "norm", "ridge", "sweep"])
    ap.add_argument("--which", default="cu", choices=list(CORPUS))
    ap.add_argument("--count", type=int, default=200)
    ap.add_argument("--rf", type=int, nargs="+", default=[13])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--norm", default="none", choices=["instance", "voxel", "none"])
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--augment", action="store_true", help="48-fold cube symmetry, exact on a periodic cell")
    ap.add_argument("--radii", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--per-volume", type=int, default=1500)
    ap.add_argument("--linear", action="store_true")
    ap.add_argument("--max-steps", type=int, default=None,
                    help="override the 12000-step cap; the released checkpoints used 36000")
    ap.add_argument("--sched-patience", type=int, default=None,
                    help="validation calls before the LR halves, 100 steps each; default 5, the "
                         "released checkpoints used 20")
    ap.add_argument("--patience", type=int, default=None,
                    help="override the 3000-step early-stop budget; must exceed "
                         "(sched_patience+1)*100 or train() raises")
    ap.add_argument("--lam", type=float, default=0.0,
                    help="weight of the squared spectral equilibrium ratio in the loss (sweep only)")
    ap.add_argument("--balance", default="none", choices=["none", "uncertainty", "gradnorm"],
                    help="set the equilibrium weight during training instead of --lam (sweep only)")
    ap.add_argument("--balance-every", type=int, default=100,
                    help="steps between gradient-norm measurements under --balance gradnorm")
    ap.add_argument("--balance-alpha", type=float, default=0.9,
                    help="moving-average weight kept on the old value under --balance gradnorm")
    ap.add_argument("--balance-lr", type=float, default=None,
                    help="learning rate of the two log-variances under --balance uncertainty; ten times "
                         "the network's when unset")
    ap.add_argument("--project-train", action="store_true",
                    help="train, validate and test on the prediction projected onto spectral equilibrium "
                         "(sweep only)")
    ap.add_argument("--width", type=int, default=None,
                    help="override the arm's default width (linear 64, nonlinear 32); "
                         "use to separate architecture from capacity")
    a = ap.parse_args()
    if a.cmd == "sweep":
        sweep(a)
    elif a.cmd == "ridge":
        ridge(a)
    elif a.cmd == "time":
        a.rf = a.rf[0]; timing(a)
    elif a.cmd == "norm":
        a.rf = a.rf[0]; norm_ablation(a)
    else:
        compare(a)




def ridge(a):
    """The EXACT best linear kernel at radius r, by normal equations, with periodic wrapping.

    This is the ceiling for any linear model of that reach: gradient descent cannot beat a
    closed-form least-squares fit of the same function class. It therefore separates two
    explanations of a training plateau. If the trained network matches this, the optimizer is
    fine and the limit is reach. If the network is far above it, the training is at fault.

    Features are 21 stiffness channels over a (2r+1)^3 neighbourhood, so 567 at r=1 and
    2625 at r=2. Cost is dominated by accumulating X^T X.
    """
    C, A = load(a.which, a.count)
    tr, va, te = splits(len(C))
    n = C.shape[1]
    rng = np.random.default_rng(0)
    print(f"exact linear optimum, periodic wrapping, {a.per_volume} voxels per volume")
    print(f"  {'r':>2} {'RF':>3} {'features':>9} {'train':>9} {'test':>9}")
    for r in a.radii:
        k = 2 * r + 1
        p = 21 * k ** 3
        XtX = np.zeros((p + 1, p + 1)); XtY = np.zeros((p + 1, 36))
        def design(v):
            """(m, p+1) patches around m random voxels of volume v, wrapped periodically."""
            idx = rng.integers(0, n, size=(a.per_volume, 3))
            offs = np.stack(np.meshgrid(*[np.arange(-r, r + 1)] * 3, indexing="ij"), -1).reshape(-1, 3)
            nb = (idx[:, None, :] + offs[None]) % n
            patch = C[v][nb[..., 0], nb[..., 1], nb[..., 2]]        # (m, k^3, 21)
            X = patch.reshape(len(idx), -1).astype(np.float64)
            return np.hstack([X, np.ones((len(X), 1))]), A[v][idx[:, 0], idx[:, 1], idx[:, 2]].astype(np.float64)
        for v in tr:
            X, Y = design(v)
            XtX += X.T @ X; XtY += X.T @ Y
        w = np.linalg.solve(XtX + 1e-6 * np.trace(XtX) / (p + 1) * np.eye(p + 1), XtY)
        def err(ids):
            num = den = 0.0
            for v in ids:
                X, Y = design(v)
                num += float(((X @ w - Y) ** 2).sum()); den += float((Y ** 2).sum())
            return np.sqrt(num / den)
        print(f"  {r:>2} {k:>3} {p:>9,} {err(tr[:20]):>9.4f} {err(te):>9.4f}", flush=True)
    r_ = REFS[a.which]
    print(f"\n  untrained references on {a.which}: Born-1 {r_[1]:.4f}, Born-2 {r_[2]:.4f}, "
          f"constant {r_['floor']:.4f}, Taylor {r_['taylor']:.4f}")

if __name__ == "__main__":
    main()
