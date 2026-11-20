"""Simple gap fillers that use no learning. They are the floor for the fill-in experiment (E2):
a pulse repeats every beat, so copying the neighbouring beat is hard to beat on short gaps.
"""
import numpy as np

from . import FS


def heart_period(x, fs=FS, lo=0.3, hi=1.5):
    """Heart period in samples (a float), or None when the stretch is too short or flat to tell.
    It is the lag between lo and hi seconds at which the signal correlates best with itself, refined
    to a fraction of a sample by fitting a parabola through the peak."""
    a, b = int(lo * fs), int(hi * fs)
    if len(x) < 2 * b:
        b = len(x) // 2
    if b <= a + 3 or np.std(x) < 1e-6:
        return None
    x = np.asarray(x, dtype=np.float64)

    def corr(k):
        u, v = x[:-k] - x[:-k].mean(), x[k:] - x[k:].mean()
        d = np.sqrt((u * u).sum() * (v * v).sum())
        return float((u * v).sum() / d) if d > 0 else 0.0

    ac = np.array([corr(k) for k in range(a, b)])
    j = int(ac.argmax())
    if 0 < j < len(ac) - 1:
        curve = ac[j - 1] - 2 * ac[j] + ac[j + 1]
        shift = 0.5 * (ac[j - 1] - ac[j + 1]) / curve if curve < 0 else 0.0
    else:
        shift = 0.0
    return a + j + float(np.clip(shift, -0.5, 0.5))


def copy_neighbor(x, start, width, fs=FS):
    """Fill x[start:start+width] by repeating the signal a whole number of heart periods away.
    The period is estimated on the longer visible side; the gap is copied from the left when the left
    side holds at least one period, otherwise from the right. Falls back to the visible mean."""
    y = np.array(x, dtype=np.float32, copy=True)
    end = start + width
    left, right = y[:start], y[end:]
    period = heart_period(left if len(left) >= len(right) else right, fs)
    i = np.arange(start, end, dtype=np.float64)
    if period is not None and len(left) >= period + 1:
        src = i - period * np.ceil((i - start + 1) / period)                # nearest copy at or left of start - 1
        y[start:end] = np.interp(src, np.arange(start), left)
    elif period is not None and len(right) >= period + 1:
        src = i + period * np.ceil((end - i) / period)                      # nearest copy at or right of end
        y[start:end] = np.interp(src, np.arange(end, len(y)), right)
    else:
        vis = np.concatenate([left, right])
        y[start:end] = vis.mean() if len(vis) else 0.0
    return y


def linear_fill(x, start, width):
    """Straight line between the two edges of the gap (held flat when the gap touches an end)."""
    y = np.array(x, dtype=np.float32, copy=True)
    end = start + width
    a = y[start - 1] if start > 0 else (y[end] if end < len(y) else 0.0)
    b = y[end] if end < len(y) else a
    y[start:end] = np.linspace(a, b, width + 2, dtype=np.float32)[1:-1]
    return y
