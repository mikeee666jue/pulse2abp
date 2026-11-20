"""Tables for the gap experiment (E2). Reads the per-segment scores saved by scripts/evaluate_general.py
with --gap; needs no GPU.

    python scripts/gaps_table.py --data results/ids
    python scripts/gaps_table.py --data results/ids --gen g1 --twin g1_twin

For each target (abp, ppg, ecg) and gap length it prints the mean score on the hidden span for

    gen alone / gen others     the generator given nothing else, or given the two other signals
    twin alone / twin others   the regression twin, same two conditions
    copy, linear               the fillers that use no model

and then differences with a 95% interval from a bootstrap over patients. A cell that has not been run yet
is shown as a dash and its differences are skipped.

Scores. `crps` is the pointwise CRPS of the draws; for a single prediction (twin, fillers) it equals the
absolute error. `mae` is the absolute error of the mean of the draws. Units: mmHg for ABP, the segment's
own standard deviation for PPG and ECG. Every file must hold the same segments and the same gap positions.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from p2a import ABP_STD   # noqa: E402

GAPS = ("0.25", "1", "2", "4", "8")


def load(folder, target, gap, tag):
    f = Path(folder) / f"{target}_gap{gap}_{tag}.npz"
    if not f.exists():
        return None
    z = np.load(f)
    scale = ABP_STD if target == "abp" else 1.0
    return dict(idx=z["idx"], start=z["start"], crps=z["crps"].astype(float) * scale, mae=z["mae_mean"].astype(float) * scale)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results/general")
    p.add_argument("--gen", default="g1", help="folder of the generator under --results")
    p.add_argument("--twin", default="g1_twin")
    p.add_argument("--base", default="base")
    p.add_argument("--data", required=True, help="patient ids are read from <data>/<split>/meta.npz")
    p.add_argument("--split", default="test")
    p.add_argument("--boot", type=int, default=1000)
    a = p.parse_args()
    root = Path(a.results)
    subj_all = np.load(Path(a.data) / a.split / "meta.npz")["subj"]
    rng = np.random.default_rng(0)

    for target in ("abp", "ppg", "ecg"):
        unit = "mmHg" if target == "abp" else "per-segment SD"
        print(f"\n===== target {target} ({unit}) =====")
        print(f"{'gap s':>6} | {'gen alone':>19} | {'gen others':>19} | {'twin alone':>10} {'twin others':>11} | {'copy':>7} {'linear':>7}")
        print(f"{'':>6} | {'crps':>9} {'mae':>9} | {'crps':>9} {'mae':>9} | {'mae':>10} {'mae':>11} | {'mae':>7} {'mae':>7}")
        cells = {}
        for gap in GAPS:
            c = dict(ga=load(root / a.gen, target, gap, "alone"), go=load(root / a.gen, target, gap, "others"),
                     ta=load(root / a.twin, target, gap, "alone"), to=load(root / a.twin, target, gap, "others"),
                     cp=load(root / a.base, target, gap, "copy"), li=load(root / a.base, target, gap, "linear"))
            have = [v for v in c.values() if v is not None]
            for v in have[1:]:
                assert np.array_equal(v["idx"], have[0]["idx"]) and np.array_equal(v["start"], have[0]["start"]), \
                    f"{target} gap {gap}: files do not hold the same segments and gap positions"
            cells[gap] = c
            f = lambda v, k: "    -    " if v is None else f"{v[k].mean():9.4f}"
            print(f"{gap:>6} | {f(c['ga'], 'crps')} {f(c['ga'], 'mae')} | {f(c['go'], 'crps')} {f(c['go'], 'mae')} | "
                  f"{f(c['ta'], 'mae'):>10} {f(c['to'], 'mae'):>11} | {f(c['cp'], 'mae')[2:]:>7} {f(c['li'], 'mae')[2:]:>7}")

        print("\n   differences, 95% interval from a patient bootstrap (negative = the first one is lower)")
        pairs = (("gen others - gen alone, crps", "go", "crps", "ga", "crps"),
                 ("gen alone crps - copy mae", "ga", "crps", "cp", "mae"),
                 ("gen alone mae - copy mae", "ga", "mae", "cp", "mae"),
                 ("twin alone - copy, mae", "ta", "mae", "cp", "mae"),
                 ("gen alone mae - twin alone mae", "ga", "mae", "ta", "mae"),
                 ("twin others - twin alone, mae", "to", "mae", "ta", "mae"),
                 ("gen others mae - twin others mae", "go", "mae", "to", "mae"),
                 ("gen others crps - copy mae", "go", "crps", "cp", "mae"))
        for label, x, kx, y, ky in pairs:
            parts = []
            for gap in GAPS:
                c = cells[gap]
                if c[x] is None or c[y] is None:
                    parts.append(f"{gap}s: -")
                    continue
                d = c[x][kx] - c[y][ky]
                subj = np.unique(subj_all[c[x]["idx"]], return_inverse=True)[1]
                n = subj.max() + 1
                tot, cnt = np.bincount(subj, weights=d, minlength=n), np.bincount(subj, minlength=n)
                pick = rng.integers(n, size=(a.boot, n))
                lo, hi = np.percentile(tot[pick].sum(1) / cnt[pick].sum(1), [2.5, 97.5])
                parts.append(f"{gap}s: {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]")
            print(f"   {label:32s} " + "   ".join(parts))


if __name__ == "__main__":
    main()
