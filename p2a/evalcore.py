"""Shared by train.py (quick validation) and evaluate.py: build a model, run it over a loader,
turn every output waveform into SBP/DBP/MAP with the checker."""
import numpy as np
import torch

from . import z_to_abp
from .checker import derive_bp
from .corrupt import corrupt
from .diffusion import Path, sample
from .unet1d import P2ANet


def build_model(a, pretrained=True):
    """a: argparse namespace or dict of the training args (also stored inside every checkpoint)."""
    a = a if isinstance(a, dict) else vars(a)
    encoder = None
    if a["encoder"] != "none":
        from .encoders import PPGEncoder
        encoder = PPGEncoder(a["papagei_repo"], a["papagei_weights"] if pretrained else None,
                             mode=a["encoder"], rank=a["lora_rank"], load=pretrained)
    return P2ANet(generative=a["mode"] == "gen", use_cal=a["use_cal"], base=a["base"], encoder=encoder)


def load_checkpoint(path, device, ema=True):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(ck["args"], pretrained=False)
    model.load_state_dict(ck["ema" if ema and "ema" in ck else "model"])
    return model.to(device).eval(), ck["args"]


@torch.no_grad()
def predict(model, args, loader, device, k=16, steps=50, w_ppg=1.0, w_cal=1.0, with_cal=True,
            kind="clean", severity=0.0, n_jobs=1, seed=0, keep_waves=0):
    """Returns numpy arrays: samples (N,K,3) sbp/dbp/map, truth (N,3), cal (N,3), has_cal (N,),
    wave_mae (N,) mean-waveform error in mmHg, flat (N,K) = checker found no beats in that draw."""
    args = args if isinstance(args, dict) else vars(args)
    path = Path(args.get("path", "cosine"))
    gen = torch.Generator(device=device).manual_seed(seed)
    out = dict(samples=[], truth=[], cal=[], has_cal=[], wave_mae=[], flat=[], idx=[], waves=[])
    amp = torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda")
    for batch in loader:
        ppg = corrupt(batch["ppg"].to(device), kind, severity, gen)
        cal = batch["cal"].to(device)
        has_cal = batch["has_cal"].to(device) if (with_cal and args["use_cal"]) else torch.zeros(len(ppg), device=device)
        with amp:
            if args["mode"] == "gen":
                draws = [sample(model, path, ppg, cal, has_cal, steps, w_ppg, w_cal, gen).float() for _ in range(k)]
                waves = torch.cat(draws, dim=1)                      # (B, K, L)
            else:
                waves = model(None, None, ppg, torch.ones(len(ppg), device=device), cal, has_cal).float()
        waves = z_to_abp(waves).cpu().numpy()
        truth_wave = z_to_abp(batch["abp"][:, 0].numpy())
        bp = derive_bp(waves, n_jobs=n_jobs)                         # (B, K, 4)
        out["samples"].append(bp[..., :3])
        out["flat"].append(bp[..., 3] == 0)
        out["wave_mae"].append(np.abs(waves.mean(1) - truth_wave).mean(-1))
        out["truth"].append(batch["bp"].numpy())
        out["cal"].append(batch["bp_cal"].numpy())
        out["has_cal"].append(has_cal.cpu().numpy())
        out["idx"].append(np.asarray(batch["idx"]))
        if keep_waves and sum(len(w) for w in out["waves"]) < keep_waves:
            out["waves"].append(np.concatenate([truth_wave[:, None], waves], axis=1).astype(np.float32))
    waves = out.pop("waves")
    res = {key: np.concatenate(v) for key, v in out.items()}
    if waves:
        res["waves"] = np.concatenate(waves)[:keep_waves]            # (n, 1 + K, L): truth first, then draws
    return res
