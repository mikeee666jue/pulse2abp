"""Stage 2 evaluation: one cell of the signal grid (E1) or one gap-filling condition (E2).

A cell is "generate --target from the signals in --visible". With --gap SEC the target is visible
except for one span of that length, and --visible lists which OTHER signals are given.

    # E1: ABP from PPG (the Stage 1 cell), ABP from PPG and ECG with an earlier reading, PPG from ABP
    python scripts/evaluate_general.py --data $DATA3 --ckpt runs/g1/best.pt --target abp --visible ppg --out results/general/g1/abp_from_ppg.json
    python scripts/evaluate_general.py --data $DATA3 --ckpt runs/g1/best.pt --target abp --visible ppg,ecg --cal fixed --out results/general/g1/abp_from_ppg+ecg_cal.json
    python scripts/evaluate_general.py --data $DATA3 --ckpt runs/g1/best.pt --target ppg --visible abp --out results/general/g1/ppg_from_abp.json
    # E2: a 2 s gap in the PPG, filled from the rest of the PPG alone, then with ABP and ECG visible
    python scripts/evaluate_general.py --data $DATA3 --ckpt runs/g1/best.pt --target ppg --gap 2 --visible none --out results/general/g1/ppg_gap2_alone.json
    python scripts/evaluate_general.py --data $DATA3 --ckpt runs/g1/best.pt --target ppg --gap 2 --visible abp,ecg --out results/general/g1/ppg_gap2_others.json
    # E2 floors without any model: copy the neighbouring beat, or a straight line
    python scripts/evaluate_general.py --data $DATA3 --baseline copy --target ppg --gap 2 --out results/general/base/ppg_gap2_copy.json

The same command works for the generator and for its regression twin (the checkpoint says which it is).
Gap positions depend only on --seed and the segment index, so every model and baseline fills the same gaps.

Scores, all on the hidden entries of the target:
    wave.crps        pointwise CRPS; equals the absolute error for the twin and the baselines (one prediction)
    wave.mae_mean    absolute error of the mean of the draws
    wave.rmse_mean   RMS error of the mean of the draws
  in the model's units (per-segment SD for PPG and ECG; for ABP also given in mmHg).
For a whole ABP waveform (--target abp, no gap) the checker also reads SBP/DBP/MAP off every draw, and the
output has the Stage 1 format, so scripts/reanalyze.py and scripts/steps_compare.py work on it unchanged.
"""
import argparse
import json
import sys
from pathlib import Path as FsPath

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(FsPath(__file__).resolve().parents[1]))
from p2a import ABP_STD, FS, z_to_abp                                        # noqa: E402
from p2a.checker import derive_bp                                            # noqa: E402
from p2a.gapfill import copy_neighbor, linear_fill                           # noqa: E402
from p2a.general import general_draws, load_general, wave_scores             # noqa: E402
from p2a.masks import IDX, SIGNALS, add_gap, cell_mask, parse_signals        # noqa: E402
from p2a.metrics import summarize                                            # noqa: E402
from p2a.tri import TriDataset                                               # noqa: E402


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--ckpt", default=None)
    p.add_argument("--baseline", choices=["none", "copy", "linear"], default="none", help="gap filler without a model")
    p.add_argument("--split", default="test")
    p.add_argument("--target", choices=SIGNALS, required=True)
    p.add_argument("--visible", default="none", help="comma-separated signals given to the model, or 'none'")
    p.add_argument("--gap", type=float, default=0.0, help="seconds hidden in the target; 0 = the whole target is hidden")
    p.add_argument("--cal", choices=["none", "fixed", "far", "other"], default="none")
    p.add_argument("--n", type=int, default=7200)
    p.add_argument("--k", type=int, default=19)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--bs", type=int, default=256)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-waves", type=int, default=0, help="keep truth, mask and draws of the first N segments")
    p.add_argument("--out", required=True)
    return p.parse_args()


def main():
    a = get_args()
    visible = parse_signals(a.visible)
    if a.target in visible:
        raise SystemExit("--target cannot also be in --visible (use --gap to hide part of it)")
    gap_w = int(round(a.gap * FS))
    if a.baseline != "none" and not gap_w:
        raise SystemExit("--baseline fills gaps: give --gap")
    if (a.baseline == "none") == (a.ckpt is None):
        raise SystemExit("give either --ckpt or --baseline")
    out = FsPath(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = TriDataset(a.data, a.split, pair=a.cal, seed=a.seed, subset=a.n)
    loader = DataLoader(ds, batch_size=a.bs, num_workers=a.workers)
    if a.baseline == "none":
        model, targs = load_general(a.ckpt, device)
    else:
        model, targs = None, dict(mode=a.baseline)
    gen = torch.Generator(device=device).manual_seed(a.seed)
    amp = torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda")
    c = IDX[a.target]
    whole_abp = a.target == "abp" and not gap_w
    acc = dict(crps=[], mae_mean=[], rmse_mean=[], idx=[], start=[])
    bp = dict(samples=[], truth=[], cal=[], has_cal=[], wave_mae=[], flat=[])
    kept = []

    for batch in loader:
        x0, cal = batch["x"].to(device), batch["cal"].to(device)
        b, _, length = x0.shape
        has_cal = batch["has_cal"].to(device) if a.cal != "none" else torch.zeros(b, device=device)
        mask = cell_mask(visible, b, length, device)
        start = np.zeros(b, dtype=np.int64)
        if gap_w:
            start = np.array([np.random.default_rng([a.seed, int(i)]).integers(length - gap_w + 1) for i in batch["idx"]])
            mask[:, c] = 1.0
            mask = add_gap(mask, a.target, torch.from_numpy(start).to(device), gap_w)
        if model is None:
            fill = copy_neighbor if a.baseline == "copy" else linear_fill
            y = x0.clone()
            sig = batch["x"][:, c].numpy()
            y[:, c] = torch.from_numpy(np.stack([fill(sig[i], int(start[i]), gap_w) for i in range(b)])).to(device)
            draws = y[:, None]
        else:
            with amp:
                draws = general_draws(model, targs, x0, mask, cal, has_cal, a.k, a.steps, gen).float()
        for key, v in wave_scores(draws, x0, mask, a.target).items():
            acc[key].append(v)
        acc["idx"].append(np.asarray(batch["idx"]))
        acc["start"].append(start)
        if whole_abp:
            waves = z_to_abp(draws[:, :, c]).cpu().numpy()
            truth_wave = z_to_abp(batch["x"][:, c].numpy())
            r = derive_bp(waves, n_jobs=max(1, a.workers))
            bp["samples"].append(r[..., :3]); bp["flat"].append(r[..., 3] == 0)
            bp["wave_mae"].append(np.abs(waves.mean(1) - truth_wave).mean(-1))
            bp["truth"].append(batch["bp"].numpy()); bp["cal"].append(batch["bp_cal"].numpy())
            bp["has_cal"].append(has_cal.cpu().numpy())
        if a.save_waves and sum(len(k_[0]) for k_ in kept) < a.save_waves:
            kept.append((batch["x"].numpy().astype(np.float32), mask.cpu().numpy().astype(np.uint8),
                         draws.cpu().numpy().astype(np.float16)))

    arr = {key: np.concatenate(v) for key, v in acc.items()}
    wave = {key: float(arr[key].mean()) for key in ("crps", "mae_mean", "rmse_mean")}
    if a.target == "abp":
        wave.update({key + "_mmhg": v * ABP_STD for key, v in list(wave.items())})
    result = dict(config=vars(a), train_args=targs, n=int(len(arr["idx"])), target=a.target, visible=list(visible),
                  gap_sec=a.gap, wave=wave)
    if whole_abp:
        bpa = {key: np.concatenate(v) for key, v in bp.items()}
        result["metrics"] = summarize(bpa["samples"], bpa["truth"], bpa["cal"], bpa["has_cal"])
        result["wave_mae"] = float(bpa["wave_mae"].mean())
        result["flat_fraction"] = float(bpa["flat"].mean())
        arr.update(bpa)
    if kept:
        arr["wave_truth"] = np.concatenate([k_[0] for k_ in kept])[:a.save_waves]
        arr["wave_mask"] = np.concatenate([k_[1] for k_ in kept])[:a.save_waves]
        arr["wave_draws"] = np.concatenate([k_[2] for k_ in kept])[:a.save_waves]
    np.savez_compressed(out.with_suffix(".npz"), **arr)
    out.write_text(json.dumps(result, indent=1, default=float))
    print(json.dumps(dict(target=a.target, visible=list(visible), gap_sec=a.gap, n=result["n"], wave=wave,
                          map=result.get("metrics", {}).get("map")), default=float))


if __name__ == "__main__":
    main()
