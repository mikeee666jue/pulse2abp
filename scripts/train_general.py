"""Stage 2: train the general model (--mode gen) or its regression twin (--mode reg).

Both see the same data and the same random masks (p2a.masks.sample_mask) and are trained on the hidden
entries only. The data root must have been prepared with the ECG (scripts/prep_pulsedb.py --ecg).

    python scripts/train_general.py --data $DATA3 --mode gen --out runs/g1
    python scripts/train_general.py --data $DATA3 --mode reg --out runs/g1_twin

Variants that test the design (same script, same data):
    # make "ABP from PPG" more frequent: hide ABP in 60% of examples, PPG in 25%, ECG in 40%
    python scripts/train_general.py --data $DATA3 --mode gen --p-hide 0.6 0.25 0.4 --out runs/g1_abpw
    # one mask only (PPG visible, the rest hidden): a single-task model trained with the Stage 2 code
    python scripts/train_general.py --data $DATA3 --mode gen --fixed-visible ppg --steps 15000 --out runs/s_ppg
    # start from the general model and continue on that one mask: does general training help the single task?
    python scripts/train_general.py --data $DATA3 --mode gen --init runs/g1/best.pt --fixed-visible ppg \
        --steps 5000 --lr 5e-5 --warmup 200 --val-every 500 --out runs/g1_ft_ppg

Model selection: best.pt is the checkpoint with the lowest validation loss over a fixed set of random
masks (the same masks, noise levels and noise at every check), so no single task decides which
checkpoint is kept. The log also carries the loss of six named cells and, every check, the Stage 1
quantity (MAP CRPS for ABP from PPG, from sampled waveforms) for a direct comparison with M1.
"""
import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path as FsPath

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(FsPath(__file__).resolve().parents[1]))
sys.path.insert(0, str(FsPath(__file__).resolve().parent))
from p2a.diffusion import Path                                              # noqa: E402
from p2a.general import build_general, general_loss, predict_abp            # noqa: E402
from p2a.masks import IDX, cell_mask, parse_signals, sample_mask            # noqa: E402
from p2a.metrics import summarize                                           # noqa: E402
from p2a.tri import TriDataset                                              # noqa: E402
from train import load_resume, save_checkpoint                              # noqa: E402  (same checkpoint format as Stage 1)

SELECT = "val loss, training masks"
# (name, visible signals, scored signal): logged at every validation check, not used for selection
CELLS = [("abp|ppg", ("ppg",), "abp"), ("abp|ecg", ("ecg",), "abp"), ("abp|ppg,ecg", ("ppg", "ecg"), "abp"),
         ("abp|none", (), "abp"), ("ppg|abp", ("abp",), "ppg"), ("ecg|ppg", ("ppg",), "ecg")]


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--train-split", default="train")
    p.add_argument("--val-split", default="val")
    p.add_argument("--mode", choices=["reg", "gen"], default="gen")
    p.add_argument("--path", choices=["cosine", "linear"], default="cosine")
    p.add_argument("--p-hide", type=float, nargs="+", default=[0.4],
                   help="probability that a whole signal is hidden: one value, or three for ABP PPG ECG")
    p.add_argument("--p-gap", type=float, nargs="+", default=[0.2],
                   help="probability that a signal gets one hidden span: one value, or three for ABP PPG ECG")
    p.add_argument("--fixed-visible", default=None,
                   help="train on ONE mask instead of random ones: these signals visible (e.g. 'ppg'), all others hidden")
    p.add_argument("--init", default=None, help="start from the weights of this checkpoint (fine-tuning)")
    p.add_argument("--p-drop-cal", type=float, default=0.5)
    p.add_argument("--base", type=int, default=64)
    p.add_argument("--steps", type=int, default=60_000)
    p.add_argument("--bs", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--val-every", type=int, default=2000)
    p.add_argument("--val-n", type=int, default=2048)
    p.add_argument("--val-sample-n", type=int, default=512, help="segments for the sampled ABP-from-PPG check (0 = skip)")
    p.add_argument("--val-k", type=int, default=19)
    p.add_argument("--val-steps", type=int, default=25)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", action="store_true", help="continue from <out>/last.pt if it exists")
    a = p.parse_args()
    if len(a.p_hide) not in (1, 3) or len(a.p_gap) not in (1, 3):
        p.error("--p-hide and --p-gap take one value or three (ABP PPG ECG)")
    if a.fixed_visible is not None and len(parse_signals(a.fixed_visible)) == len(IDX):
        p.error("--fixed-visible leaves nothing hidden")
    return a


def draw_mask(a, b, length, device, generator=None):
    """The masks this run trains on: one fixed mask, or random ones."""
    if a.fixed_visible is not None:
        return cell_mask(parse_signals(a.fixed_visible), b, length, device)
    return sample_mask(b, length, a.p_hide, a.p_gap, generator=generator, device=device)


@torch.no_grad()
def validate(model, path, loader, device, a):
    """Mean loss on the validation split with a fixed random stream, plus the named cells."""
    g = torch.Generator(device=device).manual_seed(12345)
    amp = torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda")
    tot, n = {}, 0
    for batch in loader:
        x0, cal, has = batch["x"].to(device), batch["cal"].to(device), batch["has_cal"].to(device)
        b, length = x0.shape[0], x0.shape[-1]
        mask = draw_mask(a, b, length, device, generator=g)
        cal_on = has * (torch.rand(b, generator=g, device=device) >= 0.5).float()
        t = (torch.arange(b, device=device) + torch.rand(b, generator=g, device=device)) / b   # evenly spread noise levels
        eps = torch.randn(x0.shape, generator=g, device=device)
        with amp:
            vals = {"all": general_loss(model, path, x0, mask, cal, cal_on, t, eps)}
            for name, vis, target in CELLS:
                vals[name] = general_loss(model, path, x0, cell_mask(vis, b, length, device), cal,
                                          torch.zeros(b, device=device), t, eps, channels=[IDX[target]])
        for key, v in vals.items():
            tot[key] = tot.get(key, 0.0) + float(v.float().sum())
        n += b
    return {key: v / n for key, v in tot.items()}


def main():
    a = get_args()
    torch.manual_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = FsPath(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(a), indent=1))

    train = TriDataset(a.data, a.train_split, pair="random")
    val = TriDataset(a.data, a.val_split, pair="fixed", subset=a.val_n)
    loader = DataLoader(train, batch_size=a.bs, shuffle=True, drop_last=True, num_workers=a.workers,
                        persistent_workers=a.workers > 0, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val, batch_size=a.bs, num_workers=min(2, a.workers))

    model = build_general(a).to(device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    print(f"device={device}  train segments={len(train)}  trainable params={sum(p.numel() for p in params)/1e6:.2f}M",
          flush=True)

    step = 0
    if a.init and not (a.resume and (out / "last.pt").exists()):
        ck = torch.load(a.init, map_location="cpu", weights_only=False)
        if ck["args"]["mode"] != a.mode or ck["args"]["base"] != a.base:
            raise SystemExit(f"--init: {a.init} is a different model (mode {ck['args']['mode']}, base {ck['args']['base']})")
        model.load_state_dict(ck.get("ema", ck["model"]))       # start from the weights that were evaluated
        ema.load_state_dict(ck.get("ema", ck["model"]))
        print(f"initialised from {a.init} (step {ck['step']})", flush=True)
    if a.resume and (out / "last.pt").exists():
        step = load_resume(out / "last.pt", model, ema, opt, a)
        torch.manual_seed(a.seed + step)
        print(f"resumed from step {step}", flush=True)
    best = dict(step=0, value=float("inf"), metric=SELECT)
    if step and (out / "best.pt").exists():
        best = torch.load(out / "best.pt", map_location="cpu", weights_only=False)["select"]
        print(f"best so far: step {best['step']}  {SELECT} {best['value']:.4f}", flush=True)

    path = Path(a.path)
    amp = torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda")
    log = open(out / "log.jsonl", "a")
    if step:
        log.write(json.dumps(dict(resumed_from=step)) + "\n"); log.flush()
    t0, running, logged = time.time(), 0.0, step
    model.train()
    while step < a.steps:
        for batch in loader:
            x0 = batch["x"].to(device, non_blocking=True)
            cal = batch["cal"].to(device, non_blocking=True)
            has = batch["has_cal"].to(device, non_blocking=True)
            b = x0.shape[0]
            mask = draw_mask(a, b, x0.shape[-1], device)
            cal_on = has * (torch.rand(b, device=device) >= a.p_drop_cal).float()
            frac = step / max(1, a.steps)
            scale = min(1.0, (step + 1) / a.warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * frac)))
            for g in opt.param_groups:
                g["lr"] = a.lr * scale
            with amp:
                loss = general_loss(model, path, x0, mask, cal, cal_on).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            with torch.no_grad():
                decay = min(a.ema, (1 + step) / (10 + step))
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.lerp_(pm, 1 - decay)
            running += loss.item()
            step += 1
            if step % 100 == 0:
                rec = dict(step=step, loss=running / (step - logged), lr=opt.param_groups[0]["lr"], sec=time.time() - t0)
                print(json.dumps(rec), flush=True)
                log.write(json.dumps(rec) + "\n"); log.flush()
                running, logged = 0.0, step
            if step % a.val_every == 0 or step == a.steps:
                v = validate(ema, path, val_loader, device, a)
                rec = dict(step=step, val=v)
                if a.val_sample_n > 0:      # the Stage 1 quantity: ABP from PPG, no reading
                    res = predict_abp(ema, a, val_loader, device, visible=("ppg",), k=a.val_k if a.mode == "gen" else 1,
                                      steps=a.val_steps, n_jobs=min(4, max(1, a.workers)), max_n=a.val_sample_n)
                    m = summarize(res["samples"], res["truth"])
                    rec["abp_from_ppg"] = dict(map_mae=m["map"]["mae"], map_crps=m["map"]["crps"],
                                               map_cov90=m["map"]["cov90"], n=int(len(res["truth"])))
                if v["all"] < best["value"]:
                    best = dict(step=step, value=float(v["all"]), metric=SELECT)
                    save_checkpoint(out / "best.pt", model, ema, a, step, select=best)
                rec["best_step"] = best["step"]
                print(json.dumps(rec), flush=True)
                log.write(json.dumps(rec) + "\n"); log.flush()
                save_checkpoint(out / "last.pt", model, ema, a, step, opt=opt if step < a.steps else None)
                model.train()
            if step >= a.steps:
                break
    print(f"done  best.pt is step {best['step']} ({SELECT} {best['value']:.4f})", flush=True)


if __name__ == "__main__":
    main()
