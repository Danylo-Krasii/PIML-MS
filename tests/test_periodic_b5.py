"""Regression tests for the rewind-guard and adaptive-weight fixes in `periodic.train`.

Each test runs `periodic.train` on a tiny random problem on the CPU and fails on the code
before the fix.
"""
import importlib
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")


@pytest.fixture
def periodic(tmp_path, monkeypatch):
    monkeypatch.setenv("PIML_LOGS", str(tmp_path / "logs"))
    monkeypatch.setenv("PIML_CHECKPOINTS", str(tmp_path / "ckpt"))
    from piml import paths
    importlib.reload(paths)
    from piml import periodic as mod
    importlib.reload(mod)
    return mod


def _data(n=6, g=8, seed=0):
    rng = np.random.default_rng(seed)
    C = rng.standard_normal((n, g, g, g, 21)).astype(np.float32)
    A = rng.standard_normal((n, g, g, g, 36)).astype(np.float32)
    return C, A, np.arange(4), np.arange(4, 5), np.arange(5, 6)


def _run(periodic, **kw):
    C, A, tr, va, te = _data()
    base = dict(width=4, linear=False, dev="cpu", quiet=True, max_steps=600, patience_steps=10 ** 6,
                sched_patience=10 ** 3, batch=2, lr=1e-2)
    base.update(kw)
    return periodic.train(C, A, 3, tr, va, te, **base)


def _trace(tmp_path, tag):
    return [json.loads(l) for l in open(tmp_path / "logs" / "traces" / f"{tag}.jsonl")]


def test_rewind_keeps_the_adaptive_learning_rate(periodic):
    # rewind=0.5 makes every validation after the first best a rewind
    r = _run(periodic, balance="uncertainty", balance_lr=1e-2, rewind=0.5, max_rewinds=3)
    assert r["rewinds"] == 3
    assert r["balance_lr_start"] == pytest.approx(1e-2)
    assert r["balance_lr"] == pytest.approx(1e-2)


def test_rewind_restores_log_variances_and_optimizer(periodic):
    r = _run(periodic, balance="uncertainty", balance_lr=1e-2, rewind=0.5, max_rewinds=1,
             max_steps=200)
    assert r["rewinds"] == 1
    # right after the rewind at step 200 the log-variances equal those at the best step
    assert r["s_after_last_rewind"] == pytest.approx([r["s_data"], r["s_phys"]])
    assert r["opt_restored"] is True


def test_rewinds_enter_the_trace(periodic, tmp_path):
    _run(periodic, balance="gradnorm", rewind=0.5, max_rewinds=2, tag="rw")
    rows = _trace(tmp_path, "rw")
    assert sum(1 for x in rows if x.get("event") == "rewind") == 2


def test_successive_rewinds_keep_halving(periodic, tmp_path):
    _run(periodic, balance="gradnorm", rewind=0.5, max_rewinds=3, tag="halve")
    lrs = [x["lr"] for x in _trace(tmp_path, "halve") if x.get("event") == "rewind"]
    assert lrs == pytest.approx([5e-3, 2.5e-3, 1.25e-3])


def test_final_weights_saved_beside_best(periodic):
    r = _run(periodic, tag="fin", max_steps=300)
    ck = torch.load(r["ckpt"], weights_only=False) if r["ckpt"].startswith("/") else None
    if ck is None:
        from piml import paths
        ck = torch.load(paths.ROOT / r["ckpt"], weights_only=False)
    assert "final_state" in ck
    assert set(ck["final_state"]) == set(ck["state"])


def test_clipping_covers_the_adaptive_weights(periodic):
    r = _run(periodic, balance="uncertainty", grad_clip=1e-8, max_steps=100)
    assert r["clipped_adaptive"] > 0


def test_cap_on_the_adaptive_weight(periodic):
    r = _run(periodic, balance="gradnorm", balance_cap=1e-3, max_steps=300)
    assert r["lam_eff"] <= 1e-3 + 1e-12
    r = _run(periodic, balance="uncertainty", balance_lr=0.5, balance_cap=2.0, max_steps=300)
    assert np.exp(r["s_data"] - r["s_phys"]) <= 2.0 * (1 + 1e-6)


def test_gradnorm_warmup(periodic, tmp_path):
    _run(periodic, balance="gradnorm", balance_warmup=1000, max_steps=500, tag="wu")
    row = _trace(tmp_path, "wu")[0]
    assert row["step"] == 500
    # the row is written after step 500; its last training step used the ramp at step 499
    assert row["lam_applied"] == pytest.approx(0.499 * row["lam_eff"])
