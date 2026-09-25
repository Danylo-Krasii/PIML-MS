"""Score one periodic checkpoint on its corpus's 40 test volumes, on CPU.

    CUDA_VISIBLE_DEVICES="" python -m piml.score --ckpt checkpoints/z5-nonlinear-rf13-w64-s0-none.pt \
        --which z5 --out scratch/scores/z5-nonlinear-s0

The overall number follows train().score() in periodic.py: inputs standardized with the
checkpoint's xm/xs, outputs de-standardized with ym/ys, no test-time augmentation, whole cell,
one Frobenius ratio over all 40 test volumes at once. It reproduces the test error that the
training run recorded to six digits.

Reads the 200-volume cache scratch/periodic_<which>_1000_200.npz, which `periodic.load` writes
on the first training run, and the corpus HDF5 files for the grain ids.

The forward pass is float32 like training. The norms are reduced in float64: torch's float32
CPU reduction over 47M elements drifts by 2e-3 relative, while the GPU float32 reduction of
the training run agrees with float64 to six digits.

Outputs under --out:
    per_volume.json   per test volume: ratio, num = ||p - a||^2, den = ||a||^2, so that any
                      subset reproduces the pooled statistic as sqrt(sum num / sum den);
                      grain_between_num = ||G(p) - G(a)||^2 and grain_within_num =
                      ||(p - G(p)) - (a - G(a))||^2 with G the grain-mean projector of
                      mesh_convergence.grain_mean, pooled over den the same way
    per_voxel.npy     (40, n, n, n) float32, ||p - a||_2 / ||a||_2 over the 36 channels
    per_voxel_num.npy, per_voxel_den.npy
                      the squared norms behind that ratio, so any voxel subset (boundary,
                      interior) pools the same way as the volumes
    pred_first.npy    (n, n, n, 36) float32 predicted A for the first test volume
    truth_first.npy   (n, n, n, 36) float32 reference A for the same volume
    per_volume_<mode>.json
                      the same per volume after each --project mode, float64 projection path
    summary.json      overall error, split indices, checkpoint metadata
"""

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from . import paths
from .network import LocalizationNet
from .mesh_convergence import grain_mean
from .periodic import CORPUS, splits
from .physics import (a36_to_cols, average_strain_deviation, c21_to_c6, equilibrium_ratio,
                      project_average_strain, project_equilibrium)


def net_from_ckpt(ck):
    net = LocalizationNet((ck["rf"] - 1) // 2, ck["width"], cin=ck["cin"], linear=ck["linear"],
                          padding_mode="circular", norm=ck["norm"])
    net.load_state_dict({k: v.float() for k, v in ck["state"].items()})
    return net.eval()


def projection_rows(P, Ate, Cte, modes):
    """Pooled rel.err, mean equilibrium ratio and mean average-strain deviation per projection.

    Float64, one test volume at a time. "both" is equilibrium then average strain.
    """
    acc = {m: {"num": 0.0, "den": 0.0, "ratio": [], "dev": [], "vnum": [], "vden": []} for m in modes}
    with torch.no_grad():
        for i in range(len(Ate)):
            p = a36_to_cols(P[i:i + 1]).double()
            y = torch.from_numpy(Ate[i:i + 1].astype(np.float64)).reshape(p.shape)
            c = c21_to_c6(torch.from_numpy(Cte[i:i + 1].astype(np.float64)))
            pe = project_equilibrium(p) if {"eq", "both"} & set(modes) else None
            for m in modes:
                q = {"none": p, "eq": pe}.get(m)
                if m == "strain":
                    q = project_average_strain(p, c)
                elif m == "both":
                    q = project_average_strain(pe, c)
                acc[m]["vnum"].append(float(((q - y) ** 2).sum()))
                acc[m]["vden"].append(float((y ** 2).sum()))
                acc[m]["num"] += acc[m]["vnum"][-1]
                acc[m]["den"] += acc[m]["vden"][-1]
                acc[m]["ratio"].append(float(equilibrium_ratio(q).mean()))
                acc[m]["dev"].append(float(average_strain_deviation(q, c)[0]))
    return [{"project": m, "err": float(np.sqrt(v["num"] / v["den"])),
             "eq_ratio_mean": float(np.mean(v["ratio"])), "avg_strain_dev_mean": float(np.mean(v["dev"])),
             "vol_num": v["vnum"], "vol_den": v["vden"]}
            for m, v in acc.items()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--which", required=True, choices=["cu", "z5", "z8"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--project", default="none", choices=["none", "eq", "strain", "both", "all"],
                    help="also score after inference projections; all = none, eq, strain, both")
    a = ap.parse_args()
    if not Path(a.ckpt).name.startswith(f"{a.which}-"):
        raise SystemExit(f"{a.ckpt} is not a {a.which} checkpoint; the .pt carries no corpus field")
    torch.set_num_threads(a.threads)
    t0 = time.time()

    ck = torch.load(a.ckpt, map_location="cpu")
    net = net_from_ckpt(ck)
    xm, xs, ym, ys = (ck[k].float() for k in ("xm", "xs", "ym", "ys"))

    cache = paths.SCRATCH / f"periodic_{a.which}_1000_200.npz"
    z = np.load(cache)
    _, _, te = splits(len(z["C"]))
    Cte, Ate = z["C"][te], z["A"][te]
    del z
    n = Cte.shape[1]

    X = torch.from_numpy(np.ascontiguousarray(Cte.transpose(0, 4, 1, 2, 3))).float()
    Y = torch.from_numpy(np.ascontiguousarray(Ate.transpose(0, 4, 1, 2, 3))).float()
    Xn = (X - xm) / xs

    with torch.no_grad():
        P = torch.cat([net(Xn[i:i + a.batch]) * ys + ym for i in range(0, len(te), a.batch)])
        D, Yd = (P - Y).double(), Y.double()
        num = (D ** 2).sum(dim=(1, 2, 3, 4)).numpy()
        den = (Yd ** 2).sum(dim=(1, 2, 3, 4)).numpy()
        overall = float(np.sqrt(num.sum() / den.sum()))
        per_vol = np.sqrt(num / den)
        vox_num = (D ** 2).sum(dim=1).numpy()
        vox_den = (Yd ** 2).sum(dim=1).numpy()
        per_vox = np.sqrt(vox_num / vox_den)
        gnum = np.zeros((2, len(te)))
        for i, v in enumerate(te):
            with h5py.File(paths.DATA / CORPUS[a.which] / f"sve_{1000 + v}.hdf5", "r") as f:
                vox = f["voxels"][...]
            p, y = (t[i].double().permute(1, 2, 3, 0).numpy() for t in (P, Y))
            Gp, Gy = grain_mean(p, vox), grain_mean(y, vox)
            gnum[:, i] = ((Gp - Gy) ** 2).sum(), (((p - Gp) - (y - Gy)) ** 2).sum()
        between, within = np.sqrt(gnum.sum(axis=1) / den.sum())
    secs = time.time() - t0

    print(f"checkpoint {a.ckpt}: rf {ck['rf']} width {ck['width']} linear {ck['linear']} "
          f"norm {ck['norm']} seed {ck['seed']} cin {ck['cin']} best_step {ck['best_step']}")
    print(f"corpus {a.which}, cache {cache.name}, test volumes {te[0]}..{te[-1]} ({len(te)}), "
          f"grid {n}^3, float32 forward, float64 reduction, {a.threads} threads")
    print(f"overall rel.err (train().score definition): {overall:.6f}")
    print(f"per-volume rel.err: mean {per_vol.mean():.5f} sd {per_vol.std():.5f} "
          f"min {per_vol.min():.5f} max {per_vol.max():.5f}")
    print(f"per-voxel rel.err: median {np.median(per_vox):.5f} p90 {np.percentile(per_vox, 90):.5f} "
          f"max {per_vox.max():.5f}")
    print(f"grain split (mesh_convergence.grain_mean, seeds {1000 + te[0]}..{1000 + te[-1]}): "
          f"between {between:.6f} within {within:.6f} sqrt(sum of squares) {np.hypot(between, within):.6f} "
          f"overall {overall:.6f} rel.diff {abs(np.hypot(between, within) - overall) / overall:.1e}")
    print(f"runtime {secs:.1f} s")

    proj = []
    if a.project != "none":
        modes = ["none", "eq", "strain", "both"] if a.project == "all" else [a.project]
        t1 = time.time()
        proj = projection_rows(P, Ate, Cte, modes)
        print("\ninference projections, float64; both = eq then strain")
        print(f"  {'project':>8} {'rel.err':>10} {'eq ratio mean':>14} {'||<C^-1 A> - I||_F mean':>24}")
        for r in proj:
            print(f"  {r['project']:>8} {r['err']:>10.6f} {r['eq_ratio_mean']:>14.4e} "
                  f"{r['avg_strain_dev_mean']:>24.4e}")
        print(f"projection runtime {time.time() - t1:.1f} s")

    if a.out:
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "per_volume.json", "w") as f:
            json.dump({int(v): {"ratio": float(r), "num": float(nu), "den": float(de),
                                "grain_between_num": float(gb), "grain_within_num": float(gw)}
                       for v, r, nu, de, gb, gw in zip(te, per_vol, num, den, *gnum)}, f, indent=1)
        np.save(out / "per_voxel.npy", per_vox.astype(np.float32))
        np.save(out / "per_voxel_num.npy", vox_num.astype(np.float32))
        np.save(out / "per_voxel_den.npy", vox_den.astype(np.float32))
        np.save(out / "pred_first.npy", P[0].permute(1, 2, 3, 0).contiguous().numpy().astype(np.float32))
        np.save(out / "truth_first.npy", Ate[0].astype(np.float32))
        meta = {k: (v if not hasattr(v, "shape") else None) for k, v in ck.items() if k != "state"}
        meta.update(ckpt=str(a.ckpt), which=a.which, cache=str(cache), test=[int(v) for v in te],
                    err=overall, grain_between=between, grain_within=within, secs=secs,
                    threads=a.threads, torch=torch.__version__,
                    per_voxel_def="||p-a||_2 / ||a||_2 over 36 channels")
        if proj:
            meta["projection"] = [{k: v for k, v in r.items() if not k.startswith("vol_")} for r in proj]
        with open(out / "summary.json", "w") as f:
            json.dump(meta, f, indent=1)
        for r in proj:
            with open(out / f"per_volume_{r['project']}.json", "w") as f:
                json.dump({int(v): {"ratio": float(np.sqrt(nu / de)), "num": nu, "den": de}
                           for v, nu, de in zip(te, r["vol_num"], r["vol_den"])}, f, indent=1)
        print(f"wrote {out}/")


if __name__ == "__main__":
    main()
