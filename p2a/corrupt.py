"""PPG corruptions with a known label by construction.

All functions take a z-scored batch (B, 1, L) and a severity in [0, 1], and return a batch that
has been z-scored again (as a real front end would do before the model sees it).
"""
import math

import torch

from . import FS
from .data import zscore

KINDS = ("noise", "wander", "dropout", "burst", "clip")


def corrupt(ppg, kind, severity, generator=None):
    if kind == "clean" or severity <= 0:
        return ppg
    b, _, n = ppg.shape
    dev = ppg.device
    g = generator
    rnd = lambda *s: torch.rand(*s, generator=g, device=dev)
    t = torch.arange(n, device=dev) / FS
    x = ppg.clone()
    if kind == "noise":            # white noise; severity 1 -> noise std 1.5x the signal std
        x = x + 1.5 * severity * torch.randn(x.shape, generator=g, device=dev)
    elif kind == "wander":         # slow baseline drift, 0.05-0.4 Hz
        f = 0.05 + 0.35 * rnd(b, 1, 1)
        ph = 2 * math.pi * rnd(b, 1, 1)
        x = x + 4 * severity * torch.sin(2 * math.pi * f * t + ph)
    elif kind == "dropout":        # sensor off: a contiguous flat stretch, up to 80% of the window
        width = (0.8 * severity * n * (0.5 + 0.5 * rnd(b))).long().clamp(min=1)
        start = (rnd(b) * (n - width)).long()
        idx = torch.arange(n, device=dev)[None]
        mask = (idx >= start[:, None]) & (idx < (start + width)[:, None])
        hold = x.gather(2, start[:, None, None])          # hold the last good value
        x = torch.where(mask[:, None], hold, x)
    elif kind == "burst":          # motion: three short high-amplitude bursts of 2-8 Hz noise
        for _ in range(3):
            c = rnd(b, 1, 1) * 10
            env = torch.exp(-0.5 * ((t - c) / (0.3 + 0.7 * severity)) ** 2)
            f = 2 + 6 * rnd(b, 1, 1)
            x = x + 5 * severity * env * torch.sin(2 * math.pi * f * t + 2 * math.pi * rnd(b, 1, 1))
    elif kind == "clip":           # saturation: clip the top of each pulse
        hi = torch.quantile(x, 1 - 0.6 * severity, dim=-1, keepdim=True)
        x = torch.minimum(x, hi)
    else:
        raise ValueError(kind)
    return zscore(x)
