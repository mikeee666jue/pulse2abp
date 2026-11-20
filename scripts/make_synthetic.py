"""Write a small synthetic dataset for testing the code path (never a result).

    python scripts/make_synthetic.py --root /tmp/p2a_synth
    python scripts/make_synthetic.py --root /tmp/p2a_synth3 --ecg      # Stage 2: with an ECG channel
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from p2a.data import make_synthetic   # noqa: E402

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--subjects", type=int, nargs=3, default=[200, 30, 60], help="train val test")
    p.add_argument("--per-subject", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ecg", action="store_true", help="also write ecg.npy (Stage 2)")
    a = p.parse_args()
    if a.ecg:
        from p2a.tri import make_synthetic3
        make_synthetic = make_synthetic3
    make_synthetic(a.root, tuple(a.subjects), a.per_subject, a.seed)
    print("wrote", a.root)
