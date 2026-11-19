"""What does the pulse add? Two answers from one calibrated generator checkpoint.

    python scripts/information.py --data $DATA --ckpt runs/m3/best.pt --cal fixed --out results/m3_info.json

1. waveform level: denoiser-based mutual information I(abp ; ppg | calibration), with its noise-level
   profile and cutoff (see p2a/diffusion.information_profile and tests/test_information.py)
2. summary level: how much the posterior of SBP/DBP/MAP narrows when the pulse is switched on
   (draws with w_ppg = 0 against draws with w_ppg = 1)
Both are also split by how far pressure moved since calibration.
"""
import argparse
import json
import sys
from pathlib import Path as FsPath

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(FsPath(__file__).resolve().parents[1]))
from p2a.corrupt import corrupt                                           # noqa: E402
from p2a.data import PulseDataset                                         # noqa: E402
from p2a.diffusion import Path, information_profile, integrate_information  # noqa: E402
from p2a.evalcore import load_checkpoint, predict                         # noqa: E402
from p2a.metrics import posterior_shrinkage                               # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--cal", choices=["none", "fixed", "far", "other"], default="fixed")
    p.add_argument("--ppg", default="clean")
    p.add_argument("--n", type=int, default=4000)
    p.add_argument("--k", type=int, default=39)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--bs", type=int, default=256)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, targs = load_checkpoint(a.ckpt, device)
    assert targs["mode"] == "gen", "needs a generator checkpoint"
    kind, sev = (a.ppg.split(":") + ["0"])[:2] if a.ppg != "clean" else ("clean", "0")
    ds = PulseDataset(a.data, a.split, pair=a.cal, seed=a.seed, subset=a.n)
    loader = DataLoader(ds, batch_size=a.bs, num_workers=a.workers)
    path = Path(targs.get("path", "cosine"))
    gen = torch.Generator(device=device).manual_seed(a.seed)
    given_cal = a.cal != "none" and targs["use_cal"]

    sq, dm, gaps = [], [], []
    amp = torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda")
    for batch in loader:
        ppg = corrupt(batch["ppg"].to(device), kind, float(sev), gen)
        has_cal = batch["has_cal"].to(device) if given_cal else torch.zeros(len(ppg), device=device)
        with amp:
            lam, s, d = information_profile(model, path, batch["abp"].to(device), ppg, batch["cal"].to(device),
                                            has_cal, given_cal=given_cal, generator=gen)
        sq.append(s.float().cpu()); dm.append(d.float().cpu())
        gaps.append((batch["bp"][:, 2] - batch["bp_cal"][:, 2]).abs().numpy())
    sq, dm, gap = torch.cat(sq), torch.cat(dm), np.concatenate(gaps)
    info = integrate_information(lam.cpu(), sq, dm)
    per_item = info.pop("per_item_dm")
    info.pop("per_item_sq")
    result = dict(config=vars(a), n=len(ds), waveform_information=info)

    common = dict(k=a.k, steps=a.steps, with_cal=given_cal, kind=kind, severity=float(sev),
                  n_jobs=max(1, a.workers), seed=a.seed)
    without = predict(model, targs, loader, device, w_ppg=0.0, **common)
    with_ = predict(model, targs, loader, device, w_ppg=1.0, **common)
    result["summary_shrinkage"] = posterior_shrinkage(without["samples"], with_["samples"])

    if given_cal:
        rows = []
        for lo, hi in [(0, 5), (5, 10), (10, 20), (20, 1e9)]:
            m = (gap >= lo) & (gap < hi)
            if m.sum() >= 20:
                rows.append(dict(change_mmHg=f"{lo}-{hi if hi < 1e8 else 'inf'}", n=int(m.sum()),
                                 waveform_nats=float(per_item[m].mean()),
                                 map_nats=posterior_shrinkage(without["samples"][m], with_["samples"][m])["map"]["nats"]))
        result["by_change"] = rows

    FsPath(a.out).parent.mkdir(parents=True, exist_ok=True)
    FsPath(a.out).write_text(json.dumps(result, indent=1, default=float))
    print(json.dumps({k: v for k, v in result.items() if k != "config"}, indent=1, default=float))


if __name__ == "__main__":
    main()
