"""Datasets.

On-disk layout (written by scripts/prep_pulsedb.py or make_synthetic below), one folder per split:

    <root>/<split>/ppg.npy    float32 (N, 1250)   PPG as stored by the source
    <root>/<split>/abp.npy    float32 (N, 1250)   arterial pressure in mmHg
    <root>/<split>/meta.npz   subj (N,) int       subject code
                              bp   (N, 4)         checker sbp, dbp, map, n_beats  (the ground truth we score against)
                              sbp_lbl, dbp_lbl    the source's own labels, kept only as a cross-check
                              t    (N,)           segment start time in s within the record (nan if unknown)

"Calibration" for segment i is the SBP/DBP of a different segment j of the same subject, which stands in
for one earlier cuff reading. How j is chosen is the `pair` argument; the conflict tests are pair modes.
"""
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from . import SEG_LEN, FS, abp_to_z
from .checker import derive_bp


def zscore(x, eps=1e-6):
    """Per-segment z-score over the last axis (numpy or torch)."""
    m = x.mean(-1, keepdims=True) if isinstance(x, np.ndarray) else x.mean(-1, keepdim=True)
    s = x.std(-1, keepdims=True) if isinstance(x, np.ndarray) else x.std(-1, keepdim=True)
    return (x - m) / (s + eps)


class PulseDataset(Dataset):
    """pair:
        'none'   no calibration
        'random' a fresh random same-subject segment every call (training)
        'fixed'  a seeded same-subject segment, identical on every run (evaluation)
        'far'    the same-subject segment whose MAP differs most from the target (stale calibration)
        'other'  a seeded segment from a different subject (wrong-patient calibration)
    """

    def __init__(self, root, split, pair="none", seed=0, subset=None):
        d = Path(root) / split
        self.ppg = np.load(d / "ppg.npy", mmap_mode="r")
        self.abp = np.load(d / "abp.npy", mmap_mode="r")
        meta = np.load(d / "meta.npz")
        self.subj = meta["subj"].astype(np.int64)
        self.bp = meta["bp"].astype(np.float32)  # sbp, dbp, map, n_beats
        self.pair, self.seed = pair, seed
        n = len(self.subj)
        assert self.ppg.shape == (n, SEG_LEN) and self.abp.shape == (n, SEG_LEN), (self.ppg.shape, self.abp.shape)

        # group segments by subject: order[start[s]:start[s+1]] are the indices of subject s
        self.order = np.argsort(self.subj, kind="stable")
        sorted_subj = self.subj[self.order]
        self.uniq, self.start = np.unique(sorted_subj, return_index=True)
        self.start = np.append(self.start, n)
        self.group_of = np.searchsorted(self.uniq, self.subj)
        self.pos_in_group = np.empty(n, dtype=np.int64)
        self.pos_in_group[self.order] = np.arange(n) - self.start[self.group_of[self.order]]

        self.index = np.arange(n)
        if subset is not None and subset < n:
            self.index = np.sort(np.random.default_rng(seed).choice(n, subset, replace=False))

    def __len__(self):
        return len(self.index)

    def _partner(self, i):
        """Index of the calibration segment for target i, or -1."""
        if self.pair == "none":
            return -1
        g = self.group_of[i]
        a, b = self.start[g], self.start[g + 1]
        if self.pair == "other":
            rng = np.random.default_rng([self.seed, i])
            while True:
                j = int(rng.integers(len(self.subj)))
                if self.subj[j] != self.subj[i]:
                    return j
        if b - a < 2:
            return -1
        members = self.order[a:b]
        if self.pair == "far":
            gap = np.abs(self.bp[members, 2] - self.bp[i, 2])
            return int(members[int(gap.argmax())])
        rng = np.random.default_rng() if self.pair == "random" else np.random.default_rng([self.seed, i])
        r = int(rng.integers(b - a - 1))
        if r >= self.pos_in_group[i]:
            r += 1  # skip the target itself
        return int(members[r])

    def __getitem__(self, k):
        i = int(self.index[k])
        j = self._partner(i)
        ppg = zscore(np.asarray(self.ppg[i], dtype=np.float32))
        abp = abp_to_z(np.asarray(self.abp[i], dtype=np.float32))
        bp = self.bp[i, :3]
        if j >= 0:
            bp_cal = self.bp[j, :3]
            cal = abp_to_z(bp_cal[:2])
        else:
            bp_cal = np.zeros(3, dtype=np.float32)
            cal = np.zeros(2, dtype=np.float32)
        return {
            "ppg": torch.from_numpy(ppg)[None],
            "abp": torch.from_numpy(abp)[None],
            "cal": torch.from_numpy(np.asarray(cal, dtype=np.float32)),
            "has_cal": torch.tensor(float(j >= 0)),
            "bp": torch.from_numpy(np.asarray(bp, dtype=np.float32)),
            "bp_cal": torch.from_numpy(np.asarray(bp_cal, dtype=np.float32)),
            "idx": i,
            "cal_idx": j,
        }


def write_split(root, split, ppg, abp, subj, sbp_lbl=None, dbp_lbl=None, t=None, n_jobs=1):
    """Save one split and compute checker ground truth. Returns the meta dict."""
    d = Path(root) / split
    d.mkdir(parents=True, exist_ok=True)
    ppg = np.ascontiguousarray(ppg, dtype=np.float32)
    abp = np.ascontiguousarray(abp, dtype=np.float32)
    n = len(ppg)
    bp = derive_bp(abp, n_jobs=n_jobs)
    nan = np.full(n, np.nan, dtype=np.float32)
    meta = dict(
        subj=np.asarray(subj, dtype=np.int64),
        bp=bp.astype(np.float32),
        sbp_lbl=nan if sbp_lbl is None else np.asarray(sbp_lbl, dtype=np.float32),
        dbp_lbl=nan if dbp_lbl is None else np.asarray(dbp_lbl, dtype=np.float32),
        t=nan if t is None else np.asarray(t, dtype=np.float32),
    )
    np.save(d / "ppg.npy", ppg)
    np.save(d / "abp.npy", abp)
    np.savez(d / "meta.npz", **meta)
    return meta


# ----------------------------------------------------------------------------------------------
# Synthetic data. Used only to test the code path end to end; never reported as a result.
#
# The generator has a known information structure, which is what makes it useful as a test:
#   * each subject has a baseline pressure level that leaves NO trace in the PPG
#   * each segment has a within-subject deviation `delta` that DOES change PPG shape and heart rate
# So a calibration-free model can only recover `delta` (wide posterior on level), a calibration-only
# predictor knows the level but not the change, and a model using both should beat each of them.
# ----------------------------------------------------------------------------------------------

def _beats(rng, hr, n=SEG_LEN, fs=FS):
    """Beat phase in [0,1) for each sample, with 2% beat-to-beat jitter and a random start phase."""
    period = 60.0 / hr
    rr = period * (1 + 0.02 * rng.standard_normal(int(n / fs / period) + 4))
    onsets = np.cumsum(rr) - rr[0] - rng.uniform(0, period)
    t = np.arange(n) / fs
    k = np.clip(np.searchsorted(onsets, t, side="right") - 1, 0, len(rr) - 2)
    return (t - onsets[k]) / rr[k + 1]


def _bump(phi, mu, sd):
    return np.exp(-0.5 * ((phi - mu) / sd) ** 2)


def synth_segment(rng, level, pp_base, hr_base, delta):
    hr = np.clip(hr_base - 0.5 * delta + 4 * rng.standard_normal(), 40, 150)
    phi = _beats(rng, hr)
    mean_ap = level + delta
    pp = np.clip(pp_base + 0.3 * delta + 2 * rng.standard_normal(), 15, 110)
    shape = _bump(phi, 0.16, 0.055) + 0.38 * _bump(phi, 0.42, 0.07)
    shape = (shape - shape.min()) / (shape.max() - shape.min())
    abp = (mean_ap - pp / 3) + pp * shape + 0.5 * rng.standard_normal(len(phi))
    notch = np.clip(0.45 + 0.012 * delta, 0.1, 0.8)   # dicrotic height tracks delta
    width = np.clip(0.085 - 0.0006 * delta, 0.05, 0.12)  # upstroke width tracks delta
    ppg = _bump(phi, 0.24, width) + notch * _bump(phi, 0.55, 0.09) + 0.02 * rng.standard_normal(len(phi))
    ppg = (ppg - ppg.min()) / (ppg.max() - ppg.min())
    return ppg, abp


def make_synthetic(root, n_subj=(200, 30, 60), seg_per_subj=40, seed=0):
    """Write train/val/test splits with disjoint subjects."""
    rng = np.random.default_rng(seed)
    sid = 0
    for split, ns in zip(("train", "val", "test"), n_subj):
        ppg = np.empty((ns * seg_per_subj, SEG_LEN), dtype=np.float32)
        abp = np.empty_like(ppg)
        subj = np.empty(ns * seg_per_subj, dtype=np.int64)
        r = 0
        for _ in range(ns):
            level = np.clip(88 + 14 * rng.standard_normal(), 55, 130)
            pp_base = np.clip(50 + 10 * rng.standard_normal(), 25, 90)
            hr_base = np.clip(75 + 12 * rng.standard_normal(), 50, 110)
            for _ in range(seg_per_subj):
                ppg[r], abp[r] = synth_segment(rng, level, pp_base, hr_base, 9 * rng.standard_normal())
                subj[r] = sid
                r += 1
            sid += 1
        write_split(root, split, ppg, abp, subj)
