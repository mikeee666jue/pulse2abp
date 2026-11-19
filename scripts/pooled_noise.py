"""Does a generator's spread adapt to pulse quality?  Reads saved draws, needs no GPU.

    python scripts/pooled_noise.py --clean results --noisy results/n7200_k19_s50 --data results/ids

Every test segment appears twice, once evaluated with the clean pulse and once with a noisy pulse, and the
rows are pooled, so nothing tells a predictor which rows are noisy. Compared on MAP CRPS:

    B1 + one fixed bar            regression with a single bias and width for all rows
    B1 + a bar per condition      the same, but told which rows are noisy (an upper bound for a fixed bar)
    M own draws                   a generator's own distribution
    M mean + one fixed bar        the generator's point prediction with a single width (isolates the spread)

Bars are cross-fitted by patient (fitted on half of the test patients, scored on the other half), as in
scripts/reanalyze.py, so they use test patients; for a final table fit them on the training split.
The clean files hold more segments and draws than the noisy ones: they are cut to the same segments and
thinned to the same number of draws. Each arm saw noise of the same strength but not the same samples.
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from p2a.metrics import crps   # noqa: E402


def gaussian_crps(mu, sd, y):
    z = (y - mu) / sd
    return sd * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / math.sqrt(math.pi))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--clean", default="results", help="folder with <arm>.npz from clean-pulse evaluations")
    p.add_argument("--noisy", default="results/n7200_k19_s50", help="folder with <arm>_noise.npz")
    p.add_argument("--data", required=True, help="dataset root; patient ids are read from <data>/<split>/meta.npz")
    p.add_argument("--split", default="test")
    p.add_argument("--point", default="b1", help="the point-prediction arm")
    p.add_argument("--generators", nargs="+", default=["m1", "m3"])
    p.add_argument("--quantity", type=int, default=2, help="0 SBP, 1 DBP, 2 MAP")
    p.add_argument("--boot", type=int, default=1000)
    a = p.parse_args()
    q = a.quantity
    rng = np.random.default_rng(0)

    ref = np.load(Path(a.noisy) / f"{a.generators[-1]}_noise.npz")
    idx, y = ref["idx"], ref["truth"][:, q].astype(float)
    k = ref["samples"].shape[1]
    subj = np.unique(np.load(Path(a.data) / a.split / "meta.npz")["subj"][idx], return_inverse=True)[1]

    def clean(arm):
        z = np.load(Path(a.clean) / f"{arm}.npz")
        pos = {v: i for i, v in enumerate(z["idx"])}
        s = z["samples"][np.array([pos[v] for v in idx])][:, :, q].astype(float)
        return s[:, rng.permutation(s.shape[1])[:k]] if s.shape[1] > k else s

    def noisy(arm):
        z = np.load(Path(a.noisy) / f"{arm}_noise.npz")
        assert np.array_equal(z["idx"], idx), f"{arm}: different segments"
        return z["samples"][:, :, q].astype(float)

    Y, G, cond = np.concatenate([y, y]), np.concatenate([subj, subj]), np.repeat([0, 1], len(y))
    folds = [G % 2 == 0, G % 2 == 1]

    def bars(pred, per_condition):
        r = np.empty(len(Y))
        for te in folds:
            for c in ((0, 1) if per_condition else (None,)):
                tr = ~te if c is None else (~te) & (cond == c)
                m = te if c is None else te & (cond == c)
                r[m] = gaussian_crps(pred[m] + (Y[tr] - pred[tr]).mean(), (Y[tr] - pred[tr]).std(), Y[m])
        return r

    point = np.concatenate([clean(a.point)[:, 0], noisy(a.point)[:, 0]])
    rows = {f"{a.point} + one fixed bar": bars(point, False), f"{a.point} + a bar per condition": bars(point, True)}
    for arm in a.generators:
        s = np.concatenate([clean(arm), noisy(arm)])
        sd, err = s.std(1, ddof=1), s.mean(1) - Y
        rows[f"{arm} own draws"] = crps(s, Y)
        rows[f"{arm} mean + one fixed bar"] = bars(s.mean(1), False)
        print(f"{arm}: mean draw SD clean {sd[cond == 0].mean():.2f}, noisy {sd[cond == 1].mean():.2f}; "
              f"error SD clean {err[cond == 0].std():.2f}, noisy {err[cond == 1].std():.2f}; "
              f"share of segments whose draw SD is larger with noise {np.mean(sd[cond == 1] > sd[cond == 0]):.2f}")

    print(f"\npooled clean + noisy, CRPS in mmHg, {len(Y)} rows from {G.max() + 1} patients, {k} draws")
    for name, v in rows.items():
        print(f"   {name:28s} all {v.mean():5.2f}   clean rows {v[cond == 0].mean():5.2f}   noisy rows {v[cond == 1].mean():5.2f}")

    members = [np.where(G == u)[0] for u in range(G.max() + 1)]
    draws = [np.concatenate([members[j] for j in rng.integers(len(members), size=len(members))]) for _ in range(a.boot)]
    pairs = [(f"{g} own draws", ref_name) for g in a.generators[:1]
             for ref_name in (f"{a.point} + one fixed bar", f"{a.point} + a bar per condition")]
    pairs += [(f"{g} own draws", f"{g} mean + one fixed bar") for g in a.generators]
    print(f"\ndifferences with 95% intervals from {a.boot} resamples of patients")
    for x, z in pairs:
        d = rows[x] - rows[z]
        lo, hi = np.percentile([d[i].mean() for i in draws], [2.5, 97.5])
        print(f"   {x} minus ({z}): {d.mean():+.2f}  [{lo:+.2f}, {hi:+.2f}]")


if __name__ == "__main__":
    main()
