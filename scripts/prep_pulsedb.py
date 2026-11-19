"""Convert a PulseDB "Subset" .mat file into the npy layout this repo trains on.

PulseDB subset files (MATLAB v7.3 = HDF5) hold one struct `Subset` with
    Signals  N x 3 x 1250   channels in order ECG, PPG (filtered, normalised), ABP (mmHg)
    SBP, DBP N x 1          the dataset's own labels
    Subject  N x 1 cell     subject name
(see Generate_Subsets.m in https://github.com/pulselabteam/PulseDB)

    # training file -> train + val (5% of SUBJECTS held out, so val is subject-disjoint)
    python scripts/prep_pulsedb.py --mat VitalDB_Train_Subset.mat --root $DATA --split train --val-frac 0.05
    # calibration-free test set (subjects never seen in training)
    python scripts/prep_pulsedb.py --mat VitalDB_CalFree_Test_Subset.mat --root $DATA --split test

It prints a sanity report. READ IT: if the ABP range is not in mmHg, or the checker and the dataset
labels disagree by more than a few mmHg, the channel order or array orientation is wrong.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from p2a import SEG_LEN                      # noqa: E402
from p2a.data import write_split             # noqa: E402


def read_subset_h5(path, field="Subset", chunk=20000):
    """Chunked reader, so a 27 GB file never has to sit in RAM as float64."""
    import h5py
    f = h5py.File(path, "r")
    g = f[field]
    sig = g["Signals"]
    shape = sig.shape
    ax_len, ax_ch = shape.index(SEG_LEN), shape.index(3)
    ax_n = ({0, 1, 2} - {ax_len, ax_ch}).pop()
    n = shape[ax_n]
    ppg = np.empty((n, SEG_LEN), dtype=np.float32)
    abp = np.empty((n, SEG_LEN), dtype=np.float32)
    for i in range(0, n, chunk):
        sl = [slice(None)] * 3
        sl[ax_n] = slice(i, min(n, i + chunk))
        block = np.moveaxis(sig[tuple(sl)], (ax_n, ax_ch, ax_len), (0, 1, 2))   # (m, 3, 1250)
        ppg[i:i + block.shape[0]] = block[:, 1]
        abp[i:i + block.shape[0]] = block[:, 2]
        print(f"  read {min(n, i + chunk)}/{n}", flush=True)
    sbp = np.asarray(g["SBP"]).reshape(-1).astype(np.float32)
    dbp = np.asarray(g["DBP"]).reshape(-1).astype(np.float32)
    refs = np.asarray(g["Subject"]).reshape(-1)
    names = np.array(["".join(chr(c) for c in np.asarray(f[r]).reshape(-1)) for r in refs])
    return ppg, abp, sbp, dbp, names


def read_subset_mat73(path, field="Subset"):
    """Fallback: the loader PulseDB's own README uses. Needs RAM for the whole file."""
    from mat73 import loadmat
    d = loadmat(path)[field]
    sig = np.asarray(d["Signals"])
    names = np.array([str(np.asarray(s).reshape(-1)[0]) if not isinstance(s, str) else s for s in d["Subject"]])
    return (sig[:, 1].astype(np.float32), sig[:, 2].astype(np.float32),
            np.asarray(d["SBP"]).reshape(-1).astype(np.float32),
            np.asarray(d["DBP"]).reshape(-1).astype(np.float32), names)


def report(name, meta, ppg, abp):
    bp = meta["bp"]
    ok = bp[:, 3] > 0
    print(f"[{name}] segments {len(bp)}  subjects {len(np.unique(meta['subj']))}")
    print(f"  ABP percentiles 1/50/99: {np.percentile(abp[::50], [1, 50, 99]).round(1)} mmHg   "
          f"PPG min/max: {ppg[::50].min():.2f}/{ppg[::50].max():.2f}")
    print(f"  checker found no beats in {100 * (1 - ok.mean()):.2f}% of segments")
    if np.isfinite(meta["sbp_lbl"]).any():
        print(f"  checker vs dataset labels, mean abs difference: "
              f"SBP {np.abs(bp[ok, 0] - meta['sbp_lbl'][ok]).mean():.2f}  DBP {np.abs(bp[ok, 1] - meta['dbp_lbl'][ok]).mean():.2f} mmHg")
    print(f"  SBP {bp[ok, 0].mean():.1f}±{bp[ok, 0].std():.1f}  DBP {bp[ok, 1].mean():.1f}±{bp[ok, 1].std():.1f}  "
          f"MAP {bp[ok, 2].mean():.1f}±{bp[ok, 2].std():.1f}   MAP<65: {100 * (bp[ok, 2] < 65).mean():.1f}%")
    per = np.bincount(meta["subj"] - meta["subj"].min())
    per = per[per > 0]
    print(f"  segments per subject: median {int(np.median(per))}, min {per.min()}, max {per.max()}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mat", required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--val-frac", type=float, default=0.0, help="fraction of SUBJECTS moved to a 'val' split")
    p.add_argument("--mat73", action="store_true", help="use the mat73 loader instead of chunked h5py")
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    ppg, abp, sbp, dbp, names = (read_subset_mat73 if a.mat73 else read_subset_h5)(a.mat)
    root = Path(a.root)
    root.mkdir(parents=True, exist_ok=True)
    # one name -> code table shared by every split, so the same subject keeps the same code
    table_file = root / "subjects.json"
    table = json.loads(table_file.read_text()) if table_file.exists() else {}
    for nm in np.unique(names):
        table.setdefault(str(nm), len(table))
    table_file.write_text(json.dumps(table))
    subj = np.array([table[str(nm)] for nm in names], dtype=np.int64)

    parts = {a.split: np.ones(len(subj), bool)}
    if a.val_frac > 0:
        uniq = np.unique(subj)
        rng = np.random.default_rng(a.seed)
        val_subj = rng.choice(uniq, max(1, int(round(a.val_frac * len(uniq)))), replace=False)
        is_val = np.isin(subj, val_subj)
        parts = {a.split: ~is_val, "val": is_val}
    for name, m in parts.items():
        meta = write_split(root, name, ppg[m], abp[m], subj[m], sbp[m], dbp[m], n_jobs=a.jobs)
        report(name, meta, ppg[m], abp[m])


if __name__ == "__main__":
    main()
