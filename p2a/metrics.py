"""Metrics. Everything takes plain numpy arrays in mmHg.

`samples` always has shape (N, K): K draws for each of N test segments, for one quantity
(SBP, DBP or MAP). A point predictor is the K = 1 case.
"""
import numpy as np
from scipy.stats import rankdata

from . import HYPO_MAP


def crps(samples, y):
    """Continuous ranked probability score per item, in mmHg (lower is better).

    CRPS = E|X - y| - 1/2 E|X - X'|. With K = 1 it reduces to absolute error, so CRPS of a point
    predictor IS its MAE and the two can sit in the same column. The spread term uses the unbiased
    ("fair") estimator, dividing by K(K-1), and is computed from the sorted samples in O(K log K).
    """
    samples = np.asarray(samples, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    k = samples.shape[1]
    err = np.abs(samples - y[:, None]).mean(1)
    if k == 1:
        return err
    s = np.sort(samples, axis=1)
    w = 2 * np.arange(1, k + 1) - k - 1                      # sum_{a<b} (s_b - s_a) = sum_i w_i s_i
    spread = (s * w).sum(1) * 2 / (k * (k - 1))              # = E|X - X'| over distinct pairs
    return err - 0.5 * spread


def coverage(samples, y, level=0.9):
    """Coverage of the central interval, done with order statistics so the target is exact for finite K.

    Returns (observed coverage, nominal coverage, mean width). If the K draws and the truth come from
    the same distribution, the truth falls between the r-th smallest and r-th largest draw with
    probability exactly (K - 2r + 1) / (K + 1). That is the number to compare `observed` against, NOT
    `level`: with K = 8 a perfectly calibrated model covers 78%, not 90%. K = 19 (r = 1) and K = 39
    (r = 2) make the nominal value exactly 0.90, which is why they are the defaults.
    """
    k = samples.shape[1]
    r = max(1, int(round((1 - level) / 2 * (k + 1))))
    s = np.sort(samples, axis=1)
    lo, hi = s[:, r - 1], s[:, k - r]
    return float(((y >= lo) & (y <= hi)).mean()), (k - 2 * r + 1) / (k + 1), float((hi - lo).mean())


def auroc(score, label):
    """Area under the ROC curve via the rank-sum identity. nan if only one class is present."""
    label = np.asarray(label).astype(bool)
    n1, n0 = label.sum(), (~label).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(score)
    return float((r[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def tracking(pred, truth, cal):
    """How much of a true change since calibration the prediction follows.

    Ordinary least squares (with intercept) of (pred - cal) on (truth - cal). slope 0 = the prediction
    just repeats the calibration value; slope 1 = it follows the patient. Also returns the correlation.
    """
    dx, dy = truth - cal, pred - cal
    if len(dx) < 3 or dx.std() < 1e-6:
        return float("nan"), float("nan")
    slope = float(np.cov(dx, dy, bias=True)[0, 1] / dx.var())
    corr = float(np.corrcoef(dx, dy)[0, 1]) if dy.std() > 1e-9 else 0.0
    return slope, corr


def summarize(samples, truth, cal=None, has_cal=None):
    """samples: (N, K, 3) for sbp/dbp/map; truth: (N, 3); cal: (N, 3) calibration segment values."""
    out = {}
    point = samples.mean(1)
    for q, name in enumerate(("sbp", "dbp", "map")):
        err = point[:, q] - truth[:, q]
        cov, nominal, width = (coverage(samples[:, :, q], truth[:, q]) if samples.shape[1] > 1
                               else (float("nan"), float("nan"), 0.0))
        out[name] = dict(
            mae=float(np.abs(err).mean()),
            me=float(err.mean()),
            sd=float(err.std()),
            crps=float(crps(samples[:, :, q], truth[:, q]).mean()),
            cov90=cov,
            cov90_nominal=nominal,
            width90=width,
        )
    hypo = truth[:, 2] < HYPO_MAP
    if samples.shape[1] > 1:
        score = (samples[:, :, 2] < HYPO_MAP).mean(1) + 1e-6 * (-point[:, 2])  # P(MAP<65), ties broken by mean
    else:
        score = -point[:, 2]
    out["hypo"] = dict(prevalence=float(hypo.mean()), auroc=auroc(score, hypo))
    if cal is not None:
        m = np.ones(len(truth), bool) if has_cal is None else has_cal.astype(bool)
        if m.sum() > 2:
            slope, corr = tracking(point[m, 2], truth[m, 2], cal[m, 2])
            out["tracking_map"] = dict(slope=slope, corr=corr, n=int(m.sum()),
                                       mean_abs_change=float(np.abs(truth[m, 2] - cal[m, 2]).mean()))
    return out


# ---- reference forecasts every model has to beat ---------------------------------------------

def climatology_samples(train_bp, n, k=256, seed=0):
    """Calibration-free floor: ignore the input, draw from the training distribution. -> (n, k, 3)"""
    rng = np.random.default_rng(seed)
    draw = train_bp[rng.choice(len(train_bp), k, replace=False if k <= len(train_bp) else True)]
    return np.broadcast_to(draw[None, :, :3], (n, k, 3)).copy()


def persistence_samples(cal, train_delta, k=256, seed=0):
    """Calibrated floor: calibration value plus a draw from the training distribution of
    within-subject changes. cal: (n, 3); train_delta: (m, 3) -> (n, k, 3)"""
    rng = np.random.default_rng(seed)
    d = train_delta[rng.choice(len(train_delta), k, replace=len(train_delta) < k)]
    return cal[:, None, :3] + d[None, :, :3]


def posterior_shrinkage(samples_without, samples_with):
    """How much a condition narrows the posterior of SBP/DBP/MAP, from two sets of draws for the same
    segments (e.g. sampled with w_ppg = 0 and w_ppg = 1). samples_*: (N, K, 3).

    Returns, per quantity, the mean ratio of posterior standard deviations and the Gaussian-equivalent
    information 1/2 * mean log(var_without / var_with) in nats. Unlike the waveform-level estimate in
    diffusion.information_profile this needs only samples, so it is robust, but it sees only the three
    summary numbers and assumes roughly Gaussian posteriors.
    """
    out = {}
    for q, name in enumerate(("sbp", "dbp", "map")):
        v0 = samples_without[:, :, q].var(1, ddof=1) + 1e-6
        v1 = samples_with[:, :, q].var(1, ddof=1) + 1e-6
        out[name] = dict(sd_without=float(np.sqrt(v0).mean()), sd_with=float(np.sqrt(v1).mean()),
                         sd_ratio=float(np.sqrt(v1 / v0).mean()), nats=float(0.5 * np.log(v0 / v1).mean()))
    return out
