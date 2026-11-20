"""Stage 2 checks with known answers. Run:  python -m pytest tests/test_general.py -q

The last test trains two small models with the real scripts (about 4 minutes on a CPU), so it only runs
when asked for:  P2A_SLOW=1 python -m pytest tests/test_general.py -q      (inside a job, not on a login node)
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from p2a import FS, SEG_LEN                                                         # noqa: E402
from p2a.diffusion import Path as GPath                                             # noqa: E402
from p2a.gapfill import copy_neighbor, heart_period, linear_fill                    # noqa: E402
from p2a.general import (GeneralNet, general_draws, general_loss, general_sample,   # noqa: E402
                         hidden_mean, pointwise_crps, wave_scores)
from p2a.masks import GAP_SEC, IDX, add_gap, cell_mask, parse_signals, sample_mask  # noqa: E402
from p2a.metrics import crps                                                        # noqa: E402
from p2a.tri import TriDataset, make_synthetic3                                     # noqa: E402


def tiny(generative=True):
    return GeneralNet(generative=generative, base=8, mults=(1, 2), n_res=1, emb_dim=32)


def test_training_masks_cover_every_case():
    g = torch.Generator().manual_seed(0)
    m = sample_mask(4000, SEG_LEN, generator=g)
    assert m.shape == (4000, 3, SEG_LEN) and set(m.unique().tolist()) == {0.0, 1.0}
    hidden_per_signal = (1 - m).sum(-1)                               # (B, 3)
    assert (hidden_per_signal.sum(1) > 0).all(), "an example with nothing hidden teaches nothing"
    whole = hidden_per_signal == SEG_LEN
    gap = (hidden_per_signal > 0) & ~whole
    assert abs(whole.float().mean() - (0.4 + 0.4 ** 3 / 3)) < 0.03    # p_hide plus the forced ones
    assert abs(gap.float().mean() - 0.2) < 0.03
    lo, hi = int(GAP_SEC[0] * FS), int(GAP_SEC[1] * FS)
    widths = hidden_per_signal[gap]
    assert widths.min() >= lo - 1 and widths.max() <= hi
    # a gap is one contiguous span: exactly one 1 -> 0 edge and at most one 0 -> 1 edge
    edges = (m[:, :, 1:] != m[:, :, :-1]).sum(-1)[gap]
    assert edges.max() <= 2
    # every pattern of whole signals needed by the grid occurs
    pattern = set(map(tuple, whole.int().tolist()))
    assert len(pattern) == 8
    # one probability per signal (ABP, PPG, ECG): here ABP is always hidden and PPG never is
    m = sample_mask(4000, SEG_LEN, p_hide=(1.0, 0.0, 0.5), p_gap=(0.0, 0.0, 0.5), generator=g)
    h = (1 - m).sum(-1)
    assert (h[:, IDX["abp"]] == SEG_LEN).all() and (h[:, IDX["ppg"]] == 0).all()
    assert abs((h[:, IDX["ecg"]] == SEG_LEN).float().mean() - 0.5) < 0.03 and (h[:, IDX["ecg"]] > 0).all()


def test_cell_and_gap_masks():
    m = cell_mask(parse_signals("ppg,ecg"), 2, 100)
    assert m[:, IDX["abp"]].sum() == 0 and m[:, IDX["ppg"]].min() == 1 and m[:, IDX["ecg"]].min() == 1
    assert parse_signals("none") == () and parse_signals("") == ()
    with pytest.raises(ValueError):
        parse_signals("eeg")
    g = add_gap(torch.ones(2, 3, 100), "ppg", torch.tensor([10, 90]), 20)
    assert g[0, IDX["ppg"], 10:30].sum() == 0 and g[0, IDX["ppg"]].sum() == 80
    assert g[1, IDX["ppg"], 90:].sum() == 0 and g[1, IDX["ppg"]].sum() == 90     # clipped at the end
    assert g[:, IDX["abp"]].min() == 1 and g[:, IDX["ecg"]].min() == 1


@pytest.mark.parametrize("generative", [True, False])
def test_loss_only_sees_hidden_entries(generative):
    """The gradient of the loss with respect to the network output must be zero wherever the mask says
    'visible', and the visible part of the input must reach the network unchanged."""
    torch.manual_seed(0)
    b, n = 4, 256
    x0 = torch.randn(b, 3, n)
    mask = sample_mask(b, n, gap_sec=(0.2, 1.0), generator=torch.Generator().manual_seed(1))
    seen = {}

    class Probe(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.generative = generative
            self.out = torch.nn.Parameter(torch.randn(b, 3, n))

        def forward(self, x_t, t, x_vis, m, cal, cal_on):
            seen["x_vis"], seen["x_t"] = x_vis, x_t
            return self.out

    probe = Probe()
    loss = general_loss(probe, GPath("cosine"), x0, mask, torch.zeros(b, 2), torch.zeros(b)).mean()
    loss.backward()
    assert (probe.out.grad[mask == 1] == 0).all()
    assert (probe.out.grad[mask == 0] != 0).all()
    assert torch.equal(seen["x_vis"], x0)
    # channel restriction: only the chosen signal is scored
    err = torch.ones(b, 3, n)
    only = hidden_mean(err * torch.tensor([1.0, 5.0, 9.0]).view(1, 3, 1), torch.zeros(b, 3, n), channels=[IDX["ppg"]])
    assert torch.allclose(only, torch.full((b,), 5.0))


def test_network_ignores_values_behind_the_mask():
    """Changing a hidden entry of x_vis, or the noise at a visible entry, must not change the output."""
    torch.manual_seed(0)
    net = tiny().eval()
    b, n = 2, 256
    x0, x_t = torch.randn(b, 3, n), torch.randn(b, 3, n)
    mask = cell_mask(("ppg",), b, n)
    t, cal, on = torch.full((b,), 0.5), torch.zeros(b, 2), torch.zeros(b)
    ref = net(x_t, t, x0, mask, cal, on)
    x0b, x_tb = x0.clone(), x_t.clone()
    x0b[:, IDX["abp"]] += 3.0                  # ABP is hidden: its true values must not leak in
    x_tb[:, IDX["ppg"]] += 3.0                 # PPG is visible: its noise channel must be ignored
    assert torch.allclose(net(x_tb, t, x0b, mask, cal, on), ref, atol=1e-6)
    assert ref.shape == (b, 3, n)
    assert tiny(False)(None, None, x0, mask, cal, on).shape == (b, 3, n)


@pytest.mark.parametrize("kind", ["cosine", "linear"])
def test_sampler_keeps_visible_entries(kind):
    torch.manual_seed(0)
    net = tiny().eval()
    b, n = 3, 256
    x0 = torch.randn(b, 3, n)
    mask = add_gap(cell_mask(("abp", "ecg"), b, n), "abp", torch.tensor([20, 100, 200]), 40)
    out = general_sample(net, GPath(kind), x0, mask, torch.zeros(b, 2), torch.zeros(b), steps=4)
    assert torch.equal(out[mask == 1], x0[mask == 1])
    assert torch.isfinite(out).all() and not torch.allclose(out[mask == 0], x0[mask == 0])
    draws = general_draws(net, dict(mode="gen", path=kind), x0, mask, torch.zeros(b, 2), torch.zeros(b), k=3, steps=2)
    assert draws.shape == (b, 3, 3, n) and not torch.allclose(draws[:, 0], draws[:, 1])
    one = general_draws(tiny(False).eval(), dict(mode="reg"), x0, mask, torch.zeros(b, 2), torch.zeros(b))
    assert one.shape == (b, 1, 3, n) and torch.equal(one[:, 0][mask == 1], x0[mask == 1])


def test_pointwise_crps_matches_the_scalar_one():
    rng = np.random.default_rng(0)
    d, y = rng.standard_normal((50, 19, 7)), rng.standard_normal((50, 7))
    got = pointwise_crps(torch.from_numpy(d), torch.from_numpy(y)).numpy()
    want = np.stack([crps(d[:, :, j], y[:, j]) for j in range(7)], 1)       # p2a.metrics.crps, already tested
    assert np.allclose(got, want, atol=1e-10)
    one = pointwise_crps(torch.from_numpy(d[:, :1]), torch.from_numpy(y)).numpy()
    assert np.allclose(one, np.abs(d[:, 0] - y))
    # scores are taken on the hidden entries of the target only
    x0 = torch.zeros(2, 3, 100)
    draws = torch.zeros(2, 1, 3, 100)
    draws[:, :, IDX["ppg"], 40:60] = 2.0
    mask = add_gap(torch.ones(2, 3, 100), "ppg", torch.tensor([40, 0]), 20)
    s = wave_scores(draws, x0, mask, "ppg")
    assert np.allclose(s["mae_mean"], [2.0, 0.0]) and np.allclose(s["crps"], [2.0, 0.0]) and np.allclose(s["rmse_mean"], [2.0, 0.0])


def test_gap_fillers_on_a_periodic_signal():
    t = np.arange(SEG_LEN) / FS
    x = np.sin(2 * np.pi * 1.25 * t) + 0.3 * np.sin(2 * np.pi * 2.5 * t)     # period 0.8 s = 100 samples
    assert abs(heart_period(x) - 100) < 0.2 and abs(heart_period(x[:400]) - 100) < 0.2
    for start, width in [(500, 250), (1000, 250), (0, 250), (90, 60)]:
        y = copy_neighbor(x, start, width)
        assert np.abs(y - x).max() < 0.03, (start, width)
        assert np.array_equal(y[:start], x[:start].astype(np.float32))
    lin = linear_fill(x, 500, 250)
    assert np.abs(lin[500:750] - x[500:750]).mean() > 0.3                   # a line cannot follow the beats
    assert np.isfinite(copy_neighbor(np.zeros(SEG_LEN), 100, 100)).all()    # flat signal: falls back to the mean
    assert np.allclose(linear_fill(np.arange(10.0), 3, 4), np.arange(10.0))


def test_dataset_reads_three_signals(tmp_path):
    data = tmp_path / "synth3"
    make_synthetic3(data, n_subj=(3, 2, 2), seg_per_subj=4, seed=0)
    item = TriDataset(data, "train", pair="random")[3]
    assert item["x"].shape == (3, SEG_LEN) and torch.equal(item["x"][IDX["abp"]], item["abp"][0])
    assert torch.equal(item["x"][IDX["ppg"]], item["ppg"][0])
    assert abs(float(item["x"][IDX["ecg"]].mean())) < 1e-4 and abs(float(item["x"][IDX["ecg"]].std()) - 1) < 1e-2
    (data / "val" / "ecg.npy").rename(data / "val" / "ecg.moved")
    with pytest.raises(FileNotFoundError):
        TriDataset(data, "val")


@pytest.mark.skipif(os.environ.get("P2A_SLOW") != "1", reason="trains two small models; set P2A_SLOW=1")
def test_end_to_end_training_and_evaluation(tmp_path):
    """Synthetic data with a known structure, the real training script, both modes.
    After a short run:  filling a gap in the ABP from the rest of the ABP must be far easier than
    generating the ABP from nothing, and the saved checkpoint must load and evaluate."""
    data = tmp_path / "synth3"
    make_synthetic3(data, n_subj=(24, 6, 6), seg_per_subj=16, seed=0)

    for mode in ("reg", "gen"):
        out = tmp_path / f"run_{mode}"
        cmd = [sys.executable, str(ROOT / "scripts" / "train_general.py"), "--data", str(data), "--out", str(out),
               "--mode", mode, "--base", "16", "--steps", "400", "--bs", "16", "--lr", "1e-3", "--warmup", "20", "--val-every", "200",
               "--val-n", "64", "--val-sample-n", "16", "--val-k", "3", "--val-steps", "3", "--workers", "0"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-2000:]
        recs = [json.loads(line) for line in (out / "log.jsonl").read_text().splitlines()]
        vals = [x for x in recs if "val" in x]
        assert len(vals) == 2 and (out / "best.pt").exists() and (out / "last.pt").exists()
        assert vals[-1]["val"]["all"] < vals[0]["val"]["all"] or vals[-1]["best_step"] == 200
        v = vals[-1]["val"]
        assert v["abp|ppg"] < v["abp|none"], "seeing the pulse should help generate the pressure"
        assert np.isfinite(vals[-1]["abp_from_ppg"]["map_crps"])
        losses = [x["loss"] for x in recs if "loss" in x]
        assert losses[-1] < losses[0], "training loss did not go down"
        # resume: asking for the same number of steps again must be a no-op that keeps best.pt
        r2 = subprocess.run(cmd + ["--resume"], capture_output=True, text=True)
        assert r2.returncode == 0 and "resumed from step 400" in r2.stdout, r2.stderr[-2000:]

        ev = [sys.executable, str(ROOT / "scripts" / "evaluate_general.py"), "--data", str(data), "--ckpt",
              str(out / "best.pt"), "--n", "64", "--k", "5", "--steps", "8", "--bs", "32", "--workers", "0"]
        res = {}
        for name, extra in [("abp_from_none", ["--target", "abp", "--visible", "none"]),
                            ("abp_gap", ["--target", "abp", "--gap", "1", "--visible", "none"]),
                            ("ppg_from_abp", ["--target", "ppg", "--visible", "abp", "--cal", "fixed"])]:
            f = tmp_path / f"{mode}_{name}.json"
            r = subprocess.run(ev + extra + ["--out", str(f)], capture_output=True, text=True)
            assert r.returncode == 0, r.stderr[-2000:]
            res[name] = json.loads(f.read_text())
            z = np.load(f.with_suffix(".npz"))
            assert len(z["idx"]) == 64 and len(z["crps"]) == 64
        assert "metrics" in res["abp_from_none"] and "metrics" not in res["abp_gap"]
        assert set(np.load(tmp_path / f"{mode}_abp_from_none.npz").files) >= {"samples", "truth", "cal", "has_cal", "idx"}
        assert res["abp_gap"]["wave"]["mae_mean"] < 0.8 * res["abp_from_none"]["wave"]["mae_mean"], \
            "a gap with the rest of the signal visible should be much easier than generating from nothing"

    # design variants: one fixed mask started from the general model; per-signal mask probabilities
    for name, extra in [("ft", ["--init", str(tmp_path / "run_gen" / "best.pt"), "--fixed-visible", "ppg"]),
                        ("abpw", ["--p-hide", "0.6", "0.25", "0.4"])]:
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / "train_general.py"), "--data", str(data), "--out",
                            str(tmp_path / name), "--mode", "gen", "--base", "16", "--steps", "40", "--bs", "16",
                            "--warmup", "5", "--val-every", "20", "--val-n", "64", "--val-sample-n", "0", "--workers", "0"]
                           + extra, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-2000:]
        assert (name != "ft") or "initialised from" in r.stdout
    first = [json.loads(x) for x in (tmp_path / "ft" / "log.jsonl").read_text().splitlines() if "val" in x][0]
    scratch = [json.loads(x) for x in (tmp_path / "abpw" / "log.jsonl").read_text().splitlines() if "val" in x][0]
    assert first["val"]["abp|ppg"] < scratch["val"]["abp|ppg"], "starting from a trained model should beat 20 steps from scratch"
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "train_general.py"), "--data", str(data), "--out",
                        str(tmp_path / "bad"), "--mode", "reg", "--base", "16", "--steps", "2", "--workers", "0",
                        "--init", str(tmp_path / "run_gen" / "best.pt")], capture_output=True, text=True)
    assert r.returncode != 0 and "different model" in r.stderr

    # the model-free gap fillers run through the same script on the same gaps
    f = tmp_path / "copy.json"
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "evaluate_general.py"), "--data", str(data), "--baseline",
                        "copy", "--target", "abp", "--gap", "1", "--n", "64", "--bs", "32", "--workers", "0", "--out", str(f)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    copy = json.loads(f.read_text())
    assert copy["wave"]["mae_mean_mmhg"] < 6.0, "copying the neighbouring beat should nearly solve the synthetic gap"
    assert np.array_equal(np.load(f.with_suffix(".npz"))["start"], np.load(tmp_path / "gen_abp_gap.npz")["start"])


def test_prep_writes_the_ecg_and_leaves_stage1_files_unchanged(tmp_path):
    """A file laid out like a PulseDB subset (MATLAB v7.3 = HDF5, Signals stored as 1250 x 3 x N):
    --ecg must write channel 0 as ecg.npy and must not change PPG, ABP, the labels or the patient split."""
    h5py = pytest.importorskip("h5py")
    rng = np.random.default_rng(0)
    n, subjects = 60, ["p%03d" % (i // 6) for i in range(60)]
    t = np.arange(SEG_LEN) / FS
    sig = np.empty((n, 3, SEG_LEN), dtype=np.float64)
    for i in range(n):
        phi = (t * (60 + i) / 60) % 1.0
        sig[i, 0] = np.exp(-0.5 * ((phi - 0.05) / 0.012) ** 2) + 0.01 * i            # ECG, tagged by i
        sig[i, 1] = np.exp(-0.5 * ((phi - 0.25) / 0.08) ** 2)                        # PPG
        sig[i, 2] = 70 + 0.2 * i + 45 * np.exp(-0.5 * ((phi - 0.2) / 0.06) ** 2)     # ABP in mmHg
    mat = tmp_path / "fake_subset.mat"
    with h5py.File(mat, "w") as f:
        g = f.create_group("Subset")
        g.create_dataset("Signals", data=np.transpose(sig, (2, 1, 0)))               # MATLAB order
        g.create_dataset("SBP", data=sig[:, 2].max(1)[None])
        g.create_dataset("DBP", data=sig[:, 2].min(1)[None])
        refs = f.create_group("#refs#")
        ref = np.empty((1, n), dtype=h5py.ref_dtype)
        for i, name in enumerate(subjects):
            ref[0, i] = refs.create_dataset(f"s{i}", data=np.array([[ord(ch)] for ch in name], dtype=np.uint16)).ref
        g.create_dataset("Subject", data=ref)
    roots = {}
    for tag, extra in (("stage1", []), ("stage2", ["--ecg"])):
        roots[tag] = tmp_path / tag
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / "prep_pulsedb.py"), "--mat", str(mat), "--root",
                            str(roots[tag]), "--split", "train", "--val-frac", "0.2", "--jobs", "1"] + extra,
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-2000:]
    assert "ECG min/max" in r.stdout
    for split in ("train", "val"):
        a, b = roots["stage1"] / split, roots["stage2"] / split
        assert not (a / "ecg.npy").exists()
        for name in ("ppg.npy", "abp.npy"):
            assert np.array_equal(np.load(a / name), np.load(b / name))
        ma, mb = np.load(a / "meta.npz"), np.load(b / "meta.npz")
        assert np.array_equal(ma["subj"], mb["subj"]) and np.array_equal(ma["bp"], mb["bp"])
        ecg, abp = np.load(b / "ecg.npy"), np.load(b / "abp.npy")
        i = np.round((abp.min(1) - 70) / 0.2).astype(int)                            # which source row each segment is
        assert np.allclose(ecg, sig[i, 0].astype(np.float32), atol=1e-6)
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "compare_roots.py"), "--data", str(roots["stage2"]),
                        "--other", str(roots["stage1"])], capture_output=True, text=True)
    assert r.returncode == 0 and "ROOTS_EQUAL" in r.stdout, r.stdout + r.stderr
    np.save(roots["stage2"] / "val" / "ppg.npy", np.load(roots["stage2"] / "val" / "ppg.npy") + 1)
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "compare_roots.py"), "--data", str(roots["stage2"]),
                        "--other", str(roots["stage1"])], capture_output=True, text=True)
    assert r.returncode == 1 and "ROOTS_DIFFER" in r.stdout and "ppg: DIFFERENT" in r.stdout
