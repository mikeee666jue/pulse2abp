"""Which entries the general model sees and which it has to generate (Stage 2).

A mask is a float tensor (B, 3, L):  1 = visible (given to the model),  0 = hidden (to be generated).
The channel order is fixed everywhere in Stage 2:  0 = ABP,  1 = PPG,  2 = ECG.

Every task is a mask:
    translation   a whole signal is hidden               cell_mask(("ppg",))  ->  PPG visible, ABP and ECG hidden
    fill-in       one span of a signal is hidden          add_gap(mask, "abp", start, width)
The Stage 1 task (ABP from PPG) is the mask "PPG visible, everything else hidden".
"""
import math

import torch

from . import FS

SIGNALS = ("abp", "ppg", "ecg")
IDX = {s: i for i, s in enumerate(SIGNALS)}
GAP_SEC = (0.25, 8.0)       # shortest and longest hidden span, in seconds


def parse_signals(text):
    """'ppg,ecg' -> ('ppg', 'ecg');  '' or 'none' -> ()."""
    names = tuple(s for s in (text or "").replace(" ", "").split(",") if s and s != "none")
    for s in names:
        if s not in IDX:
            raise ValueError(f"unknown signal {s!r}; choose from {SIGNALS}")
    return names


def cell_mask(visible, b, length, device=None):
    """Whole signals: those named in `visible` are given, the others are hidden."""
    m = torch.zeros(b, len(SIGNALS), length, device=device)
    for s in visible:
        m[:, IDX[s]] = 1.0
    return m


def add_gap(mask, signal, start, width):
    """Copy of `mask` with samples [start, start + width) of `signal` hidden.
    start: (B,) long tensor or int; width: int."""
    m = mask.clone()
    b, _, n = m.shape
    start = torch.as_tensor(start, device=m.device).expand(b)
    idx = torch.arange(n, device=m.device)[None]
    hole = (idx >= start[:, None]) & (idx < start[:, None] + width)
    m[:, IDX[signal]] = torch.where(hole, torch.zeros_like(m[:, IDX[signal]]), m[:, IDX[signal]])
    return m


def sample_mask(b, length, p_hide=0.4, p_gap=0.2, gap_sec=GAP_SEC, generator=None, device=None):
    """Training masks. Each signal of each example is, independently,
        hidden    (probability p_hide)   the whole signal has to be generated
        gapped    (probability p_gap)    visible except one span; span length is log-uniform in gap_sec
        visible   (otherwise)
    p_hide and p_gap are one number for all signals, or three numbers in the order ABP, PPG, ECG
    (for example p_hide = (0.6, 0.25, 0.4) makes "ABP from PPG" more frequent).
    An example with nothing hidden would teach nothing, so one of its signals (picked uniformly) is hidden.
    With one number for all, the three signals are treated alike and every grid cell is seen in training."""
    c = len(SIGNALS)
    rnd = lambda *s: torch.rand(*s, generator=generator, device=device)
    p_hide = torch.as_tensor(p_hide, dtype=torch.float32, device=device).flatten().expand(c)
    p_gap = torch.as_tensor(p_gap, dtype=torch.float32, device=device).flatten().expand(c)
    u = rnd(b, c)
    hide = u < p_hide
    gap = (u >= p_hide) & (u < p_hide + p_gap)
    empty = ~(hide | gap).any(1)                                  # all three visible, nothing to learn from
    pick = torch.randint(c, (b,), generator=generator, device=device)
    hide[empty, pick[empty]] = True

    lo, hi = math.log(gap_sec[0] * FS), math.log(gap_sec[1] * FS)
    width = torch.exp(lo + (hi - lo) * rnd(b, c)).long().clamp(1, length - 1)
    start = (rnd(b, c) * (length - width + 1)).long()
    idx = torch.arange(length, device=device)[None, None]
    hole = (idx >= start[..., None]) & (idx < (start + width)[..., None])

    m = torch.ones(b, c, length, device=device)
    m = torch.where(gap[..., None] & hole, torch.zeros_like(m), m)
    m = torch.where(hide[..., None], torch.zeros_like(m), m)
    return m
