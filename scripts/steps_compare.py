"""Compare evaluations of the same test segments row by row.  Reads saved draws, needs no GPU.

    python scripts/steps_compare.py --data results/ids
    python scripts/steps_compare.py --data results/ids --files \
        first=results/n7200_k19_s50/m3_clean_from_full.npz short=results/n7200_k19_s50/m3_s15k.npz

Written for the question "does the number of sampling steps change the evaluation?" (the default files:
one checkpoint evaluated with 50, 25 and 100 steps). It works for any evaluations that share segments and
number of draws, for example two checkpoints of one arm. For SBP, DBP and MAP it prints point error, CRPS,
90% coverage, interval width and the mean spread of the draws, then each file's difference from the first
file with a 95% interval from a patient-level bootstrap. With one draw per segment (a regression arm) only
the point error is printed.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from p2a.metrics import coverage, crps   # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--files", nargs="+", help="label=path.npz, the reference first",
                   default=["50 steps=results/n7200_k19_s50/m3_clean_from_full.npz", "25 steps=results/n7200_k19_s25/m3.npz",
                            "100 steps=results/n7200_k19_s100/m3.npz"])
    p.add_argument("--data", required=True, help="dataset root; patient ids are read from <data>/<split>/meta.npz")
    p.add_argument("--split", default="test")
    p.add_argument("--boot", type=int, default=1000)
    a = p.parse_args()

    Z = {f.split("=", 1)[0]: np.load(f.split("=", 1)[1]) for f in a.files}
    first = next(iter(Z))
    idx = Z[first]["idx"]
    subj = np.unique(np.load(Path(a.data) / a.split / "meta.npz")["subj"][idx], return_inverse=True)[1]
    members = [np.where(subj == u)[0] for u in range(subj.max() + 1)]
    rng = np.random.default_rng(0)
    boot = [np.concatenate([members[j] for j in rng.integers(len(members), size=len(members))]) for _ in range(a.boot)]
    everything = np.arange(len(idx))

    def interval(fn):
        lo, hi = np.percentile([fn(i) for i in boot], [2.5, 97.5])
        return f"{fn(everything):+.3f} [{lo:+.3f}, {hi:+.3f}]"

    for name, q in (("SBP", 0), ("DBP", 1), ("MAP", 2)):
        print(f"\n{name}: {len(idx)} segments from {subj.max() + 1} patients")
        rows = {}
        for k, z in Z.items():
            assert np.array_equal(z["idx"], idx), f"{k}: different segments"
            assert np.allclose(z["truth"], Z[first]["truth"]), f"{k}: different truth"
            s, y = z["samples"][:, :, q].astype(float), z["truth"][:, q].astype(float)
            rows[k] = dict(s=s, y=y, e=np.abs(s.mean(1) - y), many=s.shape[1] > 1)
            line = f"   {k:>10s}, {s.shape[1]} draws: MAE {rows[k]['e'].mean():5.2f}  mean error {(s.mean(1) - y).mean():+5.2f}"
            if rows[k]["many"]:
                cov, nominal, width = coverage(s, y)
                rows[k].update(sd=s.std(1, ddof=1), c=crps(s, y))
                line += (f"  CRPS {rows[k]['c'].mean():5.2f}  coverage {cov:.3f} (nominal {nominal:.2f})  width {width:5.1f}  "
                         f"mean draw SD {rows[k]['sd'].mean():5.2f}  error SD {(s.mean(1) - y).std():5.2f}")
            print(line)
        B = rows[first]
        for k, A in rows.items():
            if k == first:
                continue
            line = f"   {k} minus {first}: MAE {interval(lambda i: A['e'][i].mean() - B['e'][i].mean())}"
            if A["many"] and B["many"]:
                line += (f"   CRPS {interval(lambda i: A['c'][i].mean() - B['c'][i].mean())}   "
                         f"coverage {interval(lambda i: coverage(A['s'][i], A['y'][i])[0] - coverage(B['s'][i], B['y'][i])[0])}   "
                         f"draw SD ratio {A['sd'].mean() / B['sd'].mean():.3f}")
            print(line)


if __name__ == "__main__":
    main()
