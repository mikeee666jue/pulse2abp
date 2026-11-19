"""Re-read saved evaluation draws (results/*.npz) and answer four questions without a GPU.

    python scripts/reanalyze.py --results results --data $DATA      # on the cluster (reads patient ids)
    python scripts/reanalyze.py --results results                   # anywhere (see --block)

1. Is a generator's distribution better than a constant error bar around a point prediction?
2. Within a patient, how much does the prediction move with the true pressure (pulse-following) and
   with the calibration reading (calibration-following)?  Between patients, how well is the level known?
3. How far do simple references get: the calibration reading shrunk toward the mean, and a linear
   combination of a pulse-only prediction with the calibration reading?
4. Which differences survive a patient-level bootstrap?

Why (2) replaces the "tracking slope" in p2a.metrics.tracking: that slope regresses (prediction -
calibration) on (truth - calibration). Calibration sits on both sides, so a CONSTANT prediction scores
0.5 and any model that ignores the calibration scores above that. It cannot be read as "follows x% of a
change". Here every variable has its patient mean removed first, so a constant scores 0, repeating the
calibration scores (pulse 0, calibration 1), and a perfect predictor scores (1, 0).

The linear references are cross-fitted by patient: coefficients fitted on half of the test patients score
the other half. No patient is scored by a fit that saw them, but the fit does use test patients; for a
final table, fit them on the training split instead.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from p2a.metrics import coverage, crps   # noqa: E402

Q = {"sbp": 0, "dbp": 1, "map": 2}


def gaussian_crps(mu, sd, y):
    z = (y - mu) / sd
    return sd * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / math.sqrt(math.pi))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results")
    p.add_argument("--arms", nargs="+", default=["b1", "m1", "m3"], help="names of .npz files, same segments in each")
    p.add_argument("--cal-from", default="m3", help="the arm whose file carries the calibration readings")
    p.add_argument("--pulse-only", default="b1", help="the arm used as the pulse-only input of the linear combination")
    p.add_argument("--quantity", choices=list(Q), default="map")
    p.add_argument("--data", default=None, help="dataset root; patient ids are read from <data>/<split>/meta.npz")
    p.add_argument("--split", default="test")
    p.add_argument("--block", type=int, default=400, help="without --data: assume each patient is this many consecutive rows")
    p.add_argument("--boot", type=int, default=1000)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    q = Q[a.quantity]

    Z = {n: np.load(Path(a.results) / f"{n}.npz") for n in a.arms}
    ref = Z[a.cal_from]
    idx, y = ref["idx"], ref["truth"][:, q].astype(np.float64)
    for n in a.arms:
        assert np.array_equal(Z[n]["idx"], idx), f"{n} was evaluated on different segments"
    cal = ref["cal"][:, q].astype(np.float64)
    assert (ref["has_cal"] > 0).all(), "some segments have no calibration reading"
    if a.data:
        subj = np.load(Path(a.data) / a.split / "meta.npz")["subj"][idx]
    else:
        subj = idx // a.block
        shifted = (idx + a.block // 2) // a.block
        share = lambda g: np.var(np.bincount(g, y)[np.unique(g)] / np.bincount(g)[np.unique(g)]) / y.var()
        print(f"no --data: taking patients as blocks of {a.block} rows. Between-block share of variance "
              f"{share(subj):.2f} (aligned) vs {share(shifted):.2f} (shifted half a block); aligned should be clearly higher.")
    subj = np.unique(subj, return_inverse=True)[1]
    S = {n: Z[n]["samples"][:, :, q].astype(np.float64) for n in a.arms}
    P = {n: S[n].mean(1) for n in a.arms}
    out = dict(quantity=a.quantity, segments=len(y), patients=int(subj.max() + 1))
    print(f"{a.quantity.upper()}: {len(y)} segments from {subj.max() + 1} patients\n")

    folds = [subj % 2 == 0, subj % 2 == 1]

    def crossfit(fn):
        r = np.empty(len(y))
        for te in folds:
            r[te] = fn(~te, te)
        return r

    def constant_bar(pred):          # bias and width fitted on the other half of the patients
        return crossfit(lambda tr, te: gaussian_crps(pred[te] + (y[tr] - pred[tr]).mean(), (y[tr] - pred[tr]).std(), y[te]))

    def linear(*feats):
        X = np.column_stack([np.ones(len(y)), *feats])
        return crossfit(lambda tr, te: X[te] @ np.linalg.lstsq(X[tr], y[tr], rcond=None)[0])

    print("1. Distribution vs constant error bar (CRPS, lower is better)")
    out["distribution"] = {}
    for n in a.arms:
        k = S[n].shape[1]
        row = dict(mae=np.abs(P[n] - y).mean(), crps_constant_bar=constant_bar(P[n]).mean())
        if k > 1:
            sd = S[n].std(1, ddof=1)
            cov, nominal, width = coverage(S[n], y)
            row.update(crps_own_draws=crps(S[n], y).mean(), coverage90=cov, nominal=nominal,
                       mean_posterior_sd=sd.mean(), error_sd=(P[n] - y).std(),
                       spearman_sd_vs_abs_error=spearmanr(sd, np.abs(P[n] - y))[0])
        out["distribution"][n] = row
        own = f"own draws {row['crps_own_draws']:.2f}" if k > 1 else "own draws   - "
        extra = (f"   posterior SD {row['mean_posterior_sd']:.1f} vs error SD {row['error_sd']:.1f}, "
                 f"rank corr(SD, |error|) {row['spearman_sd_vs_abs_error']:.2f}, coverage {row['coverage90']:.2f}") if k > 1 else ""
        print(f"   {n:4s} MAE {row['mae']:5.2f}   CRPS: {own}, mean + constant bar {row['crps_constant_bar']:.2f}{extra}")

    print("\n2. Within and between patients")
    count = np.bincount(subj)
    center = lambda v: (v - (np.bincount(subj, v) / count)[subj], np.bincount(subj, v) / count)
    yd, ym = center(y)
    cd, _ = center(cal)
    print(f"   true {a.quantity.upper()}: SD between patients {ym.std():.2f}, within a patient {yd.std():.2f} mmHg")
    out["within_between"] = {}
    for n in a.arms:
        pd_, pm = center(P[n])
        pulse, calib = np.linalg.lstsq(np.stack([yd, cd], 1), pd_, rcond=None)[0]
        row = dict(pulse_following=pulse, calibration_following=calib, within_corr=np.corrcoef(pd_, yd)[0, 1],
                   within_r2=1 - ((pd_ - yd) ** 2).sum() / (yd ** 2).sum(),
                   between_r2=1 - ((pm - ym) ** 2).sum() / ((ym - ym.mean()) ** 2).sum(),
                   patient_mean_error_sd=(pm - ym).std())
        out["within_between"][n] = row
        print(f"   {n:4s} pulse-following {pulse:.2f}  calibration-following {calib:+.2f}  within-patient corr {row['within_corr']:.2f}, "
              f"R2 {row['within_r2']:.2f}   between-patient R2 {row['between_r2']:.2f} (patient-mean error SD {row['patient_mean_error_sd']:.1f})")

    print("\n3. References (cross-fitted by patient)")
    po = P[a.pulse_only]
    refs = {"constant": linear(), "calibration as is": cal, "calibration shrunk toward the mean": linear(cal),
            f"{a.pulse_only} (pulse only)": po, f"{a.pulse_only} + calibration, linear": linear(po, cal)}
    refs.update({f"{n} (as trained)": P[n] for n in a.arms if n != a.pulse_only})
    out["references"] = {}
    for name, pred in refs.items():
        row = dict(mae=np.abs(pred - y).mean(), crps_constant_bar=constant_bar(pred).mean())
        out["references"][name] = row
        print(f"   {name:38s} MAE {row['mae']:5.2f}   CRPS with constant bar {row['crps_constant_bar']:5.2f}")
    w = np.linalg.lstsq(np.column_stack([np.ones(len(y)), po, cal]), y, rcond=None)[0]
    out["linear_weights"] = dict(pulse_only=w[1], calibration=w[2])
    print(f"   weights of the linear combination: {a.pulse_only} {w[1]:.2f}, calibration {w[2]:.2f}")

    print(f"\n4. Patient-level bootstrap, 95% intervals ({a.boot} resamples)")
    rng = np.random.default_rng(0)
    members = [np.where(subj == u)[0] for u in range(subj.max() + 1)]
    C = {n: crps(S[n], y) for n in a.arms}
    stack = refs[f"{a.pulse_only} + calibration, linear"]
    stats = {f"MAE {n} - {a.pulse_only}": (lambda i, n=n: np.abs(P[n][i] - y[i]).mean() - np.abs(po[i] - y[i]).mean())
             for n in a.arms if n != a.pulse_only}
    stats.update({f"MAE {n} - ({a.pulse_only} + calibration, linear)": (lambda i, n=n: np.abs(P[n][i] - y[i]).mean() - np.abs(stack[i] - y[i]).mean())
                  for n in a.arms if n != a.pulse_only})
    stats.update({f"CRPS {n} own draws - ({a.pulse_only} + constant bar)": (lambda i, n=n, cb=constant_bar(po): C[n][i].mean() - cb[i].mean())
                  for n in a.arms if S[n].shape[1] > 1})
    stats.update({f"coverage90 {n}": (lambda i, n=n: coverage(S[n][i], y[i])[0]) for n in a.arms if S[n].shape[1] > 1})
    draws = [np.concatenate([members[k] for k in rng.integers(len(members), size=len(members))]) for _ in range(a.boot)]
    out["bootstrap"] = {}
    everything = np.arange(len(y))
    for name, fn in stats.items():
        lo, hi = np.percentile([fn(i) for i in draws], [2.5, 97.5])
        out["bootstrap"][name] = dict(value=fn(everything), lo=lo, hi=hi)
        print(f"   {name:46s} {fn(everything):+.2f}  [{lo:+.2f}, {hi:+.2f}]")

    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
