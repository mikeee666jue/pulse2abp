"""Stage 2: one model over ABP, PPG and ECG that generates whatever is hidden.

It is the Stage 1 network (p2a.unet1d.P2ANet, with the calibration input of M3) with two changes:
    * the single PPG input channel becomes three signal channels, any of which can be a target
    * the on/off flag for the whole PPG becomes a mask over signals and time (p2a.masks)

    generator        input = [noisy hidden entries (3), visible values (3), mask (3)]  ->  v-prediction (3)
    regression twin  input = [visible values (3), mask (3)]                            ->  the signals (3)

Both are trained and scored on the hidden entries only. With the mask "PPG visible, rest hidden" the
generator answers the Stage 1 question; M1 and M3 stay as the single-task references for that cell.

Nothing in Stage 1 (unet1d.py, diffusion.py, train.py, evaluate.py) is changed by this file.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import z_to_abp
from .checker import derive_bp
from .diffusion import Path, _b
from .masks import IDX, SIGNALS, cell_mask
from .unet1d import P2ANet, ResBlock, timestep_embedding


class GeneralNet(P2ANet):
    def __init__(self, generative=True, base=64, **kw):
        super().__init__(generative=generative, use_cal=True, base=base, **kw)
        c = len(SIGNALS)
        cin = (c if generative else 0) + 2 * c
        self.inp = nn.Conv1d(cin, self.inp.out_channels, 3, padding=1)
        last = self.out[-1].in_channels
        self.out = nn.Sequential(nn.GroupNorm(8, last), nn.SiLU(), nn.Conv1d(last, c, 3, padding=1))

    def forward(self, x_t, t, x_vis, mask, cal=None, cal_on=None):
        """x_t: (B,3,L) noisy signals or None (twin); t: (B,) in [0,1] or None;
        x_vis: (B,3,L) the signals (only entries with mask = 1 are used); mask: (B,3,L), 1 = visible;
        cal: (B,2) earlier SBP/DBP in z units; cal_on: (B,) 1 = reading given."""
        b, _, n = x_vis.shape
        feats = [x_vis * mask, mask]
        if self.generative:
            feats.insert(0, x_t * (1 - mask))        # visible entries never enter as noise
        x = torch.cat(feats, dim=1)
        pad = (-n) % self.down_factor
        if pad:
            x = F.pad(x, (0, pad), mode="reflect")

        if self.generative:
            emb = self.time_mlp(timestep_embedding(t, self.emb_dim))
        else:
            emb = torch.zeros(b, self.emb_dim, device=x.device, dtype=x.dtype)
        c_on = cal_on.view(b, 1).to(x.dtype)
        emb = emb + c_on * self.cal_mlp(cal) + (1 - c_on) * self.cal_null

        h = self.inp(x)
        skips = [h]
        for m in self.down:
            h = m(h, emb) if isinstance(m, ResBlock) else m(h)
            skips.append(h)
        h = self.mid2(self.mid_attn(self.mid1(h, emb)), emb)
        for m in self.up:
            h = m(torch.cat([h, skips.pop()], dim=1), emb) if isinstance(m, ResBlock) else m(h)
        return self.out(h)[..., :n]


def build_general(a):
    a = a if isinstance(a, dict) else vars(a)
    return GeneralNet(generative=a["mode"] == "gen", base=a["base"])


def load_general(path, device, ema=True):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = build_general(ck["args"])
    model.load_state_dict(ck["ema" if ema and "ema" in ck else "model"])
    return model.to(device).eval(), ck["args"]


# ----------------------------------------------------------------------------------------------
# Loss: on hidden entries only
# ----------------------------------------------------------------------------------------------

def hidden_mean(err, mask, channels=None):
    """Per example, the mean of `err` over its hidden entries -> (B,).
    channels: optional list of channel indices to score (the others are ignored even if hidden).
    Every example counts the same whether it hides 31 samples or three whole signals; otherwise short
    gaps would have almost no weight in the loss."""
    w = 1 - mask
    if channels is not None:
        keep = torch.zeros(mask.shape[1], device=mask.device)
        keep[list(channels)] = 1
        w = w * keep[None, :, None]
    return (err * w).flatten(1).sum(1) / w.flatten(1).sum(1).clamp(min=1)


def general_loss(model, path, x0, mask, cal, cal_on, t=None, eps=None, channels=None):
    """Per-example loss (B,). Generator: squared error of the v-prediction. Twin: absolute error.
    t and eps can be passed in to make a validation pass repeatable."""
    if not model.generative:
        pred = model(None, None, x0, mask, cal, cal_on)
        return hidden_mean((pred.float() - x0).abs(), mask, channels)
    b = x0.shape[0]
    t = torch.rand(b, device=x0.device) if t is None else t
    eps = torch.randn_like(x0) if eps is None else eps
    a, s = path.alpha_sigma(_b(t, x0))
    out = model(a * x0 + s * eps, t, x0, mask, cal, cal_on)
    return hidden_mean((out.float() - path.target(x0, eps, _b(t, x0))) ** 2, mask, channels)


# ----------------------------------------------------------------------------------------------
# Generation
# ----------------------------------------------------------------------------------------------

@torch.no_grad()
def general_sample(model, path, x0, mask, cal, cal_on, steps=50, generator=None):
    """One draw of the hidden entries given the visible ones (same deterministic sampler as Stage 1).
    Returns (B,3,L) in which the visible entries are copied from x0."""
    x = torch.randn(x0.shape, device=x0.device, generator=generator)
    ts = torch.linspace(1.0, 0.0, steps + 1, device=x0.device)
    for i in range(steps):
        t = ts[i].expand(x0.shape[0])
        out = model(x, t, x0, mask, cal, cal_on).float()
        x0_hat, eps_hat = path.to_x0_eps(x, out, _b(t, x))
        a, s = path.alpha_sigma(ts[i + 1])
        x = a * x0_hat + s * eps_hat
    return mask * x0 + (1 - mask) * x


@torch.no_grad()
def general_draws(model, args, x0, mask, cal, cal_on, k=19, steps=50, generator=None):
    """(B, K, 3, L): K draws from the generator, or the twin's single prediction (K = 1)."""
    args = args if isinstance(args, dict) else vars(args)
    if args["mode"] != "gen":
        pred = model(None, None, x0, mask, cal, cal_on).float()
        return (mask * x0 + (1 - mask) * pred)[:, None]
    path = Path(args.get("path", "cosine"))
    return torch.stack([general_sample(model, path, x0, mask, cal, cal_on, steps, generator) for _ in range(k)], 1)


# ----------------------------------------------------------------------------------------------
# Scores on waveforms
# ----------------------------------------------------------------------------------------------

def pointwise_crps(draws, truth):
    """Fair CRPS at every sample point. draws: (B, K, ...), truth: (B, ...) -> (B, ...).
    With K = 1 this is the absolute error, so a point prediction and a set of draws share one scale."""
    k = draws.shape[1]
    err = (draws - truth[:, None]).abs().mean(1)
    if k == 1:
        return err
    srt = draws.sort(dim=1).values
    coef = (2 * torch.arange(1, k + 1, device=draws.device, dtype=draws.dtype) - k - 1)
    coef = coef.view(1, k, *([1] * (draws.dim() - 2)))
    spread = (srt * coef).sum(1) / (k * (k - 1))        # = sum over pairs |x_i - x_j| / (2 K (K-1))
    return err - spread


@torch.no_grad()
def wave_scores(draws, x0, mask, signal):
    """Scores of one signal on its hidden entries, per example. draws: (B,K,3,L).
    Units are the model's: z units for ABP (multiply by ABP_STD for mmHg), per-segment SD for PPG and ECG."""
    c = IDX[signal]
    d, y, hid = draws[:, :, c], x0[:, c], 1 - mask[:, c]
    n = hid.sum(-1).clamp(min=1)
    mean = d.mean(1)
    out = dict(crps=(pointwise_crps(d, y) * hid).sum(-1) / n,
               mae_mean=((mean - y).abs() * hid).sum(-1) / n,
               rmse_mean=(((mean - y) ** 2 * hid).sum(-1) / n).sqrt())
    return {k: v.float().cpu().numpy() for k, v in out.items()}


@torch.no_grad()
def predict_abp(model, args, loader, device, visible=("ppg",), k=19, steps=50, with_cal=False, n_jobs=1,
                seed=0, max_n=None):
    """Generate whole ABP waveforms from the visible signals and read SBP/DBP/MAP off each with the checker.
    Returns the same arrays as p2a.evalcore.predict, so p2a.metrics.summarize and scripts/reanalyze.py
    apply to the general model exactly as they do to M1 and M3."""
    assert "abp" not in visible
    gen = torch.Generator(device=device).manual_seed(seed)
    out = dict(samples=[], truth=[], cal=[], has_cal=[], wave_mae=[], flat=[], idx=[])
    amp = torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda")
    done = 0
    for batch in loader:
        x0, cal = batch["x"].to(device), batch["cal"].to(device)
        has_cal = batch["has_cal"].to(device) if with_cal else torch.zeros(len(x0), device=device)
        mask = cell_mask(visible, len(x0), x0.shape[-1], device)
        with amp:
            draws = general_draws(model, args, x0, mask, cal, has_cal, k, steps, gen)
        waves = z_to_abp(draws[:, :, IDX["abp"]].float()).cpu().numpy()          # (B, K, L) mmHg
        truth_wave = z_to_abp(batch["x"][:, IDX["abp"]].numpy())
        bp = derive_bp(waves, n_jobs=n_jobs)
        out["samples"].append(bp[..., :3])
        out["flat"].append(bp[..., 3] == 0)
        out["wave_mae"].append(np.abs(waves.mean(1) - truth_wave).mean(-1))
        out["truth"].append(batch["bp"].numpy())
        out["cal"].append(batch["bp_cal"].numpy())
        out["has_cal"].append(has_cal.cpu().numpy())
        out["idx"].append(np.asarray(batch["idx"]))
        done += len(x0)
        if max_n and done >= max_n:
            break
    return {key: np.concatenate(v) for key, v in out.items()}
