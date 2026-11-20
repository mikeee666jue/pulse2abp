"""Stage 2 data: the Stage 1 dataset plus the ECG.

On disk, a Stage 2 split folder is a Stage 1 split folder with one more file:

    <root>/<split>/ecg.npy    float32 (N, 1250)   ECG as stored by the source

written by `scripts/prep_pulsedb.py --ecg`. Each item carries, besides the Stage 1 fields,

    x   (3, 1250)   [ABP in z units, PPG z-scored per segment, ECG z-scored per segment]

in the channel order of p2a.masks. ABP keeps a fixed scale so that its level in mmHg survives;
PPG and ECG have arbitrary units, so each segment is standardised.
"""
from pathlib import Path

import numpy as np
import torch

from . import SEG_LEN
from .data import PulseDataset, _beats, _bump, write_split, zscore


class TriDataset(PulseDataset):
    def __init__(self, root, split, pair="none", seed=0, subset=None):
        super().__init__(root, split, pair=pair, seed=seed, subset=subset)
        f = Path(root) / split / "ecg.npy"
        if not f.exists():
            raise FileNotFoundError(f"{f} is missing: this dataset was prepared without the ECG. "
                                    f"Run scripts/prep_pulsedb.py with --ecg into a new root.")
        self.ecg = np.load(f, mmap_mode="r")
        assert self.ecg.shape == self.ppg.shape, (self.ecg.shape, self.ppg.shape)

    def __getitem__(self, k):
        item = super().__getitem__(k)
        ecg = np.nan_to_num(np.asarray(self.ecg[item["idx"]], dtype=np.float32))
        item["x"] = torch.cat([item["abp"], item["ppg"], torch.from_numpy(zscore(ecg))[None]], dim=0)
        return item


# ----------------------------------------------------------------------------------------------
# Synthetic data with an ECG. Used only to test the code path; never reported as a result.
# Same information structure as p2a.data.make_synthetic (the subject's pressure level leaves no trace
# in the PPG; the within-subject change does). The ECG shares the beat timing and nothing else, so it
# knows the heart rate but neither the level nor the pulse shape.
# ----------------------------------------------------------------------------------------------

def synth_segment3(rng, level, pp_base, hr_base, delta):
    hr = np.clip(hr_base - 0.5 * delta + 4 * rng.standard_normal(), 40, 150)
    phi = _beats(rng, hr)
    pp = np.clip(pp_base + 0.3 * delta + 2 * rng.standard_normal(), 15, 110)
    shape = _bump(phi, 0.16, 0.055) + 0.38 * _bump(phi, 0.42, 0.07)
    shape = (shape - shape.min()) / (shape.max() - shape.min())
    abp = (level + delta - pp / 3) + pp * shape + 0.5 * rng.standard_normal(len(phi))
    notch = np.clip(0.45 + 0.012 * delta, 0.1, 0.8)
    width = np.clip(0.085 - 0.0006 * delta, 0.05, 0.12)
    ppg = _bump(phi, 0.24, width) + notch * _bump(phi, 0.55, 0.09) + 0.02 * rng.standard_normal(len(phi))
    ppg = (ppg - ppg.min()) / (ppg.max() - ppg.min())
    ecg = (_bump(phi, 0.04, 0.012) - 0.25 * _bump(phi, 0.075, 0.012) + 0.18 * _bump(phi, 0.33, 0.05)
           + 0.02 * rng.standard_normal(len(phi)))
    return ecg, ppg, abp


def make_synthetic3(root, n_subj=(200, 30, 60), seg_per_subj=40, seed=0):
    """Write train/val/test splits with disjoint subjects, including ecg.npy."""
    rng = np.random.default_rng(seed)
    sid = 0
    for split, ns in zip(("train", "val", "test"), n_subj):
        n = ns * seg_per_subj
        ecg, ppg, abp = (np.empty((n, SEG_LEN), dtype=np.float32) for _ in range(3))
        subj = np.empty(n, dtype=np.int64)
        r = 0
        for _ in range(ns):
            level = np.clip(88 + 14 * rng.standard_normal(), 55, 130)
            pp_base = np.clip(50 + 10 * rng.standard_normal(), 25, 90)
            hr_base = np.clip(75 + 12 * rng.standard_normal(), 50, 110)
            for _ in range(seg_per_subj):
                ecg[r], ppg[r], abp[r] = synth_segment3(rng, level, pp_base, hr_base, 9 * rng.standard_normal())
                subj[r] = sid
                r += 1
            sid += 1
        write_split(root, split, ppg, abp, subj)
        np.save(Path(root) / split / "ecg.npy", ecg)
