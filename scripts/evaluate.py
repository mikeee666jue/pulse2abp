"""Score a checkpoint (or only the reference forecasts) on a split.

    # reference forecasts only: climatology (calibration-free floor) and persistence (calibrated floor)
    python scripts/evaluate.py --data $DATA --split test --baselines-only --out results/b0.json

    # a model, no calibration
    python scripts/evaluate.py --data $DATA --split test --ckpt runs/m1/best.pt --cal none --out results/m1.json

    # conflict tests on the calibrated generator
    python scripts/evaluate.py --data $DATA --split test --ckpt runs/m3/best.pt --cal far   --out results/m3_far.json
    python scripts/evaluate.py --data $DATA --split test --ckpt runs/m3/best.pt --cal other --out results/m3_other.json
    python scripts/evaluate.py --data $DATA --split test --ckpt runs/m3/best.pt --cal fixed --w-ppg 2 --out results/m3_w2.json
    python scripts/evaluate.py --data $DATA --split test --ckpt runs/m3/best.pt --cal fixed --ppg noise:0.5 --out results/m3_noise.json

--cal   none | fixed (random same-patient reading) | far (same patient, most different pressure) | other (another patient)
--ppg   clean | <kind>:<severity in 0..1>, kinds: noise wander dropout burst clip
"""
import argparse
import json
import sys
from pathlib import Path as FsPath

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(FsPath(__file__).resolve().parents[1]))
from p2a.data import PulseDataset                                    # noqa: E402
from p2a.evalcore import load_checkpoint, predict                    # noqa: E402
from p2a.metrics import climatology_samples, persistence_samples, summarize  # noqa: E402


def within_subject_deltas(ds, n=20000, seed=0):
    """Training-set distribution of (target - calibration) for random same-subject pairs. (m, 3)"""
    rng = np.random.default_rng(seed)
    out = []
    for i in rng.choice(len(ds.subj), min(n, len(ds.subj)), replace=False):
        j = ds._partner(int(i))
        if j >= 0:
            out.append(ds.bp[i, :3] - ds.bp[j, :3])
    return np.asarray(out, dtype=np.float32)


def by_change(samples, truth, cal, has_cal, edges=(0, 5, 10, 20, 1e9)):
    """MAP error and tracking, stratified by how far pressure has moved since calibration."""
    rows = []
    gap = np.abs(truth[:, 2] - cal[:, 2])
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (has_cal > 0) & (gap >= lo) & (gap < hi)
        if m.sum() < 20:
            continue
        s = summarize(samples[m], truth[m], cal[m], has_cal[m])
        rows.append(dict(change_mmHg=f"{lo}-{hi if hi < 1e8 else 'inf'}", n=int(m.sum()), map_mae=s["map"]["mae"],
                         map_crps=s["map"]["crps"], cov90=s["map"]["cov90"], width90=s["map"]["width90"],
                         slope=s.get("tracking_map", {}).get("slope")))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--train-split", default="train")
    p.add_argument("--ckpt", default=None)
    p.add_argument("--baselines-only", action="store_true")
    p.add_argument("--cal", choices=["none", "fixed", "far", "other"], default="none")
    p.add_argument("--ppg", default="clean")
    p.add_argument("--n", type=int, default=20000, help="number of test segments (seeded subset)")
    p.add_argument("--k", type=int, default=39, help="posterior draws per segment (19 or 39 make the 90%% interval exact)")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--w-ppg", type=float, default=1.0)
    p.add_argument("--w-cal", type=float, default=1.0)
    p.add_argument("--bs", type=int, default=512)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-ema", action="store_true")
    p.add_argument("--save-waves", type=int, default=0, help="also save this many example waveforms (truth + draws)")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    FsPath(a.out).parent.mkdir(parents=True, exist_ok=True)
    pair = "fixed" if (a.baselines_only and a.cal == "none") else a.cal
    ds = PulseDataset(a.data, a.split, pair=pair, seed=a.seed, subset=a.n)
    result = dict(config=vars(a), n=len(ds))

    if a.baselines_only:
        train = PulseDataset(a.data, a.train_split, pair="fixed", seed=a.seed)
        truth = ds.bp[ds.index, :3]
        partner = np.array([ds._partner(int(i)) for i in ds.index])
        has_cal = (partner >= 0).astype(np.float32)
        cal = np.where(has_cal[:, None] > 0, ds.bp[np.maximum(partner, 0), :3], 0.0)
        clim = climatology_samples(train.bp, len(truth), seed=a.seed)
        result["climatology"] = summarize(clim, truth)
        result["train_mean"] = summarize(np.broadcast_to(train.bp[:, :3].mean(0), (len(truth), 1, 3)), truth)
        m = has_cal > 0
        pers = persistence_samples(cal[m], within_subject_deltas(train, seed=a.seed), seed=a.seed)
        result["persistence"] = summarize(pers, truth[m], cal[m])
        result["persistence_point"] = summarize(cal[m][:, None, :], truth[m], cal[m])
        result["persistence_by_change"] = by_change(pers, truth[m], cal[m], has_cal[m])
    else:
        model, targs = load_checkpoint(a.ckpt, device, ema=not a.no_ema)
        kind, sev = (a.ppg.split(":") + ["0"])[:2] if a.ppg != "clean" else ("clean", "0")
        loader = DataLoader(ds, batch_size=a.bs, num_workers=a.workers)
        res = predict(model, targs, loader, device, k=a.k if targs["mode"] == "gen" else 1, steps=a.steps,
                      w_ppg=a.w_ppg, w_cal=a.w_cal, with_cal=a.cal != "none", kind=kind, severity=float(sev),
                      n_jobs=max(1, a.workers), seed=a.seed, keep_waves=a.save_waves)
        result["train_args"] = targs
        result["metrics"] = summarize(res["samples"], res["truth"], res["cal"], res["has_cal"])
        result["wave_mae"] = float(res["wave_mae"].mean())
        result["flat_fraction"] = float(res["flat"].mean())
        if a.cal != "none" and targs["use_cal"]:
            result["by_change"] = by_change(res["samples"], res["truth"], res["cal"], res["has_cal"])
        np.savez_compressed(FsPath(a.out).with_suffix(".npz"), **res)

    FsPath(a.out).write_text(json.dumps(result, indent=1, default=float))
    key = "metrics" if "metrics" in result else "climatology"
    for name in ("sbp", "dbp", "map"):
        r = result[key][name]
        print(f"{key:12s} {name}: MAE {r['mae']:.2f}  ME±SD {r['me']:+.2f}±{r['sd']:.2f}  CRPS {r['crps']:.2f}  "
              f"cov90 {r['cov90']:.2f} (calibrated = {r['cov90_nominal']:.2f})  width90 {r['width90']:.1f}")
    print("hypotension:", result[key]["hypo"], " tracking:", result[key].get("tracking_map"))
    if "persistence" in result:
        r = result["persistence"]["map"]
        print(f"persistence  map: MAE {r['mae']:.2f}  CRPS {r['crps']:.2f}  cov90 {r['cov90']:.2f}")


if __name__ == "__main__":
    main()
