"""Check that a Stage 2 data root (prepared with --ecg) holds exactly the Stage 1 data plus the ECG:
same patients in each split, same checker ground truth, same PPG and ABP arrays.

    SCRIPT=compare_roots.py sbatch --job-name=cmp-roots slurm/eval.sbatch --other /data/user_data/$USER/p2a/vital
    (slurm/eval.sbatch passes --data "$DATA", here the Stage 2 root, and --workers)

Prints ROOTS_EQUAL when every split matches. If it does not, do not compare Stage 2 numbers with Stage 1.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--data", required=True, help="Stage 2 root (with ecg.npy)")
p.add_argument("--other", required=True, help="Stage 1 root")
p.add_argument("--workers", type=int, default=0)
a = p.parse_args()

ok = True
for split in ("train", "val", "test"):
    d, o = Path(a.data) / split, Path(a.other) / split
    if not (d / "meta.npz").exists() and not (o / "meta.npz").exists():
        print(f"[{split}] in neither root, skipped", flush=True)
        continue
    m, mo = np.load(d / "meta.npz"), np.load(o / "meta.npz")
    same = dict(subj=np.array_equal(m["subj"], mo["subj"]), bp=np.array_equal(m["bp"], mo["bp"], equal_nan=True))
    for name in ("ppg", "abp"):
        x, y = np.load(d / f"{name}.npy", mmap_mode="r"), np.load(o / f"{name}.npy", mmap_mode="r")
        same[name] = x.shape == y.shape and all(np.array_equal(x[i:i + 50000], y[i:i + 50000]) for i in range(0, len(x), 50000))
    ecg = np.load(d / "ecg.npy", mmap_mode="r")
    same["ecg shape"] = ecg.shape == (len(m["subj"]), 1250)
    print(f"[{split}] segments {len(m['subj'])}  patients {len(np.unique(m['subj']))}  " +
          "  ".join(f"{k}: {'same' if v else 'DIFFERENT'}" for k, v in same.items()), flush=True)
    ok &= all(same.values())
print("ROOTS_EQUAL" if ok else "ROOTS_DIFFER")
sys.exit(0 if ok else 1)
