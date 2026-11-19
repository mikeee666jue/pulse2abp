"""Train the regression baseline (--mode reg) or the generator (--mode gen).

Examples
    # B1  regression baseline, pulse only
    python scripts/train.py --data $DATA --mode reg --out runs/b1
    # M1  conditional diffusion, pulse only
    python scripts/train.py --data $DATA --mode gen --out runs/m1
    # M2  + PaPaGei with LoRA adapters
    python scripts/train.py --data $DATA --mode gen --encoder lora --papagei-repo $PAPAGEI \
        --papagei-weights $PAPAGEI/weights/papagei_s.pt --out runs/m2_lora
    # M3  + calibration, both conditions dropped independently
    python scripts/train.py --data $DATA --mode gen --use-cal --out runs/m3
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
from p2a.corrupt import KINDS, corrupt            # noqa: E402
from p2a.data import PulseDataset                 # noqa: E402
from p2a.diffusion import Path, training_loss     # noqa: E402
from p2a.evalcore import build_model, predict     # noqa: E402
from p2a.metrics import summarize                 # noqa: E402


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--train-split", default="train")
    p.add_argument("--val-split", default="val")
    p.add_argument("--mode", choices=["reg", "gen"], default="gen")
    p.add_argument("--path", choices=["cosine", "linear"], default="cosine")
    p.add_argument("--use-cal", action="store_true")
    p.add_argument("--p-drop-ppg", type=float, default=0.1)
    p.add_argument("--p-drop-cal", type=float, default=0.5)
    p.add_argument("--encoder", choices=["none", "frozen", "lora", "full", "scratch"], default="none")
    p.add_argument("--papagei-repo", default=None)
    p.add_argument("--papagei-weights", default=None)
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--enc-lr-mult", type=float, default=1.0, help="encoder learning-rate multiplier (try 0.1 for --encoder full)")
    p.add_argument("--base", type=int, default=64)
    p.add_argument("--steps", type=int, default=100_000)
    p.add_argument("--bs", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--aug-corrupt", type=float, default=0.0, help="probability of corrupting a training batch's PPG")
    p.add_argument("--val-every", type=int, default=5000)
    p.add_argument("--val-n", type=int, default=1024)
    p.add_argument("--val-k", type=int, default=19)
    p.add_argument("--val-steps", type=int, default=25)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", action="store_true",
                   help="continue from <out>/last.pt if it exists (jobs on a preemptible partition get killed and requeued)")
    return p.parse_args()


# Settings that may differ between a checkpoint and the run that resumes it. Everything else must match.
RESUME_FREE = {"resume", "out", "data", "workers", "val_every", "val_n", "val_k", "val_steps"}


# Model selection: best.pt is the checkpoint with the lowest value of this on the validation split.
# For the regression baseline a point prediction's CRPS equals its MAE, so one rule serves every arm.
SELECT = "val map crps"


def save_checkpoint(path, model, ema, args, step, opt=None, **extra):
    """Write to a temporary file and rename it, so a job killed mid-write cannot leave a broken checkpoint.
    The optimizer state is only needed to resume: pass opt for an unfinished last.pt, leave it out otherwise."""
    ck = dict(model=model.state_dict(), ema=ema.state_dict(), args=vars(args), step=step, **extra)
    if opt is not None:
        ck["opt"] = opt.state_dict()
    tmp = path.with_name(path.stem + ".tmp.pt")
    torch.save(ck, tmp)
    tmp.replace(path)


def load_resume(path, model, ema, opt, args):
    """Load weights, EMA and optimizer state; return the step to continue from.
    Refuses to continue a run that was trained with different settings."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    diff = {k: (ck["args"].get(k), v) for k, v in vars(args).items()
            if k not in RESUME_FREE and ck["args"].get(k) != v}
    if diff:
        raise SystemExit(f"--resume: {path} was trained with different settings, as (checkpoint, now): {diff}")
    model.load_state_dict(ck["model"])
    ema.load_state_dict(ck["ema"])
    if ck["step"] < args.steps:
        if "opt" not in ck:
            raise SystemExit(f"--resume: {path} has no optimizer state (written before --resume existed)")
        opt.load_state_dict(ck["opt"])
    return ck["step"]


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = FsPath(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=1))

    pair = "random" if args.use_cal else "none"
    train = PulseDataset(args.data, args.train_split, pair=pair)
    val = PulseDataset(args.data, args.val_split, pair="fixed" if args.use_cal else "none", subset=args.val_n)
    loader = DataLoader(train, batch_size=args.bs, shuffle=True, drop_last=True, num_workers=args.workers,
                        persistent_workers=args.workers > 0, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val, batch_size=args.bs, num_workers=min(2, args.workers))

    model = build_model(args).to(device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    enc = [p for n, p in model.named_parameters() if n.startswith("encoder.") and p.requires_grad]
    rest = [p for n, p in model.named_parameters() if not n.startswith("encoder.") and p.requires_grad]
    groups = [dict(params=rest, lr=args.lr)] + ([dict(params=enc, lr=args.lr * args.enc_lr_mult)] if enc else [])
    opt = torch.optim.AdamW(groups, weight_decay=0.0)
    base_lrs = [g["lr"] for g in groups]
    n_train = sum(p.numel() for p in rest) + sum(p.numel() for p in enc)
    print(f"device={device}  train segments={len(train)}  trainable params={n_train/1e6:.2f}M "
          f"(encoder {sum(p.numel() for p in enc)/1e6:.2f}M)", flush=True)

    step = 0
    if args.resume and (out / "last.pt").exists():
        step = load_resume(out / "last.pt", model, ema, opt, args)
        torch.manual_seed(args.seed + step)     # do not replay the data order and noise of the first steps
        print(f"resumed from step {step}", flush=True)
    # best validation score so far and the step it came from; that model is the one stored in best.pt
    best = dict(step=0, value=float("inf"), metric=SELECT)
    if step and (out / "best.pt").exists():
        best = torch.load(out / "best.pt", map_location="cpu", weights_only=False)["select"]
        print(f"best so far: step {best['step']}  {SELECT} {best['value']:.3f}", flush=True)

    path = Path(args.path)
    amp = torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda")
    gen = torch.Generator(device=device).manual_seed(args.seed + step)
    log = open(out / "log.jsonl", "a")
    if step:    # steps after this one may already be in the log from the killed run: the later record wins
        log.write(json.dumps(dict(resumed_from=step)) + "\n"); log.flush()
    t0, running, logged = time.time(), 0.0, step
    model.train()
    while step < args.steps:
        for batch in loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items() if torch.is_tensor(v)}
            if args.aug_corrupt > 0 and torch.rand(1).item() < args.aug_corrupt:
                kind = KINDS[int(torch.randint(len(KINDS), (1,)))]
                batch["ppg"] = corrupt(batch["ppg"], kind, float(torch.rand(1)), gen)
            frac = step / max(1, args.steps)
            scale = min(1.0, (step + 1) / args.warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * frac)))
            for g, lr in zip(opt.param_groups, base_lrs):
                g["lr"] = lr * scale
            with amp:
                if args.mode == "gen":
                    loss = training_loss(model, path, batch, args.p_drop_ppg, args.p_drop_cal)
                else:
                    b = batch["ppg"].shape[0]
                    cal_on = batch["has_cal"] * (torch.rand(b, device=device) >= args.p_drop_cal).float()
                    pred = model(None, None, batch["ppg"], torch.ones(b, device=device), batch["cal"], cal_on)
                    loss = (pred.float() - batch["abp"]).abs().mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(rest + enc, 1.0)
            opt.step()
            with torch.no_grad():
                decay = min(args.ema, (1 + step) / (10 + step))
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.lerp_(pm, 1 - decay)
                for be, bm in zip(ema.buffers(), model.buffers()):
                    be.copy_(bm)
            running += loss.item()
            step += 1
            if step % 100 == 0:
                rec = dict(step=step, loss=running / (step - logged), lr=opt.param_groups[0]["lr"], sec=time.time() - t0)
                print(json.dumps(rec), flush=True)
                log.write(json.dumps(rec) + "\n"); log.flush()
                running, logged = 0.0, step
            if step % args.val_every == 0 or step == args.steps:
                res = predict(ema, args, val_loader, device, k=args.val_k if args.mode == "gen" else 1,
                              steps=args.val_steps, n_jobs=min(4, max(1, args.workers)))
                m = summarize(res["samples"], res["truth"], res["cal"], res["has_cal"])
                if m["map"]["crps"] < best["value"]:
                    best = dict(step=step, value=float(m["map"]["crps"]), metric=SELECT)
                    save_checkpoint(out / "best.pt", model, ema, args, step, select=best)
                rec = dict(step=step, val=dict(sbp=m["sbp"], dbp=m["dbp"], map=m["map"], hypo=m["hypo"],
                                               tracking=m.get("tracking_map"), wave_mae=float(res["wave_mae"].mean()),
                                               flat=float(res["flat"].mean())), best_step=best["step"])
                print(json.dumps(rec), flush=True)
                log.write(json.dumps(rec) + "\n"); log.flush()
                save_checkpoint(out / "last.pt", model, ema, args, step, opt=opt if step < args.steps else None)
                model.train()
            if step >= args.steps:
                break
    print(f"done  best.pt is step {best['step']} ({SELECT} {best['value']:.3f})", flush=True)


if __name__ == "__main__":
    main()
