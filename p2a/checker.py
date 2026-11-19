"""Independent checker: SBP / DBP / MAP from an arterial pressure waveform.

This is the only place blood-pressure numbers are computed from waveforms.
The same function scores the true ABP and every generated ABP, so truth and
prediction always go through identical beat detection. It never looks at the
dataset's own SBP/DBP labels; `scripts/prep_pulsedb.py` reports how well the
two agree, which is the cross-check on both.
"""
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.signal import find_peaks

from . import FS


def bp_from_waveform(abp, fs=FS):
    """One waveform in mmHg -> (sbp, dbp, map, n_beats).

    Systolic peaks and diastolic troughs are found by prominence, with the
    threshold set to 40% of the segment's pulse-pressure range. A dicrotic
    bump is a local maximum too, but its prominence is far below that, so it is
    not counted as a beat. MAP is the mean over whole beats (first trough to
    last trough) so a partial beat at either edge does not bias it.
    """
    abp = np.asarray(abp, dtype=np.float64)
    lo, hi = np.percentile(abp, [2, 98])
    prom = max(4.0, 0.4 * (hi - lo))
    dist = int(0.3 * fs)  # no two beats closer than 0.3 s (200 bpm)
    peaks, _ = find_peaks(abp, distance=dist, prominence=prom)
    troughs, _ = find_peaks(-abp, distance=dist, prominence=prom)
    if len(peaks) < 2 or len(troughs) < 2:
        # Not a pulsatile trace (flat line, heavy artefact, or a degenerate sample).
        # Fall back to robust percentiles and report zero beats so callers can count these.
        return float(hi), float(lo), float(abp.mean()), 0
    sbp = float(abp[peaks].mean())
    dbp = float(abp[troughs].mean())
    mean_ap = float(abp[troughs[0]:troughs[-1]].mean())
    return sbp, dbp, mean_ap, int(len(peaks))


def _chunk(arr):
    return np.array([bp_from_waveform(x) for x in arr], dtype=np.float32)


def derive_bp(abp, n_jobs=1, chunk=2000):
    """Batch version. abp: (..., L) in mmHg -> array (..., 4) = sbp, dbp, map, n_beats."""
    abp = np.asarray(abp)
    flat = abp.reshape(-1, abp.shape[-1])
    if n_jobs <= 1 or len(flat) <= chunk:
        out = _chunk(flat)
    else:
        parts = [flat[i:i + chunk] for i in range(0, len(flat), chunk)]
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            out = np.concatenate(list(ex.map(_chunk, parts)), axis=0)
    return out.reshape(*abp.shape[:-1], 4)
