"""Does the information estimate return the right number when the right number is known?

Toy problem with a closed-form answer. The "pulse" is a constant c ~ N(0,1) and the "pressure" is the
constant c + noise ~ N(0, s^2), both repeated over 16 samples. Then

    I(pressure ; pulse) = 1/2 * log(1 + 1/s^2)  nats      (0.347 for s = 1)

We train the real network class with the real training loss (pulse dropped half the time), then run the
real estimator and print its profile next to the exact one. Two things to look at in the output:
  * at low and middle log-SNR both estimators (sq, dm) track the exact integrand
  * at high log-SNR the exact integrand is ~0 but sq grows and dm goes negative: that is network error,
    and it is why integrate_information() stops where dm first turns non-positive
About two to three minutes on a laptop CPU.

    python tests/test_information.py
"""
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from p2a.diffusion import Path as GPath, information_profile, integrate_information, training_loss  # noqa: E402
from p2a.unet1d import P2ANet                                                                        # noqa: E402

LENGTH = 16


def toy_batch(b, noise_sd, length=LENGTH):
    c = torch.randn(b, 1, 1)
    x0 = c + noise_sd * torch.randn(b, 1, 1)
    return dict(abp=x0.expand(b, 1, length).contiguous(), ppg=c.expand(b, 1, length).contiguous(),
                cal=torch.zeros(b, 2), has_cal=torch.zeros(b))


def exact_integrand(logsnr, noise_sd, length=LENGTH):
    """A scalar with prior variance v seen through `length` samples at SNR g has MMSE v / (1 + length*g*v)."""
    g = math.exp(logsnr)
    v0, v1 = 1 + noise_sd ** 2, noise_sd ** 2          # prior variance without / with the pulse
    return length * g * (v0 / (1 + length * g * v0) - v1 / (1 + length * g * v1))


def run(noise_sd=1.0, kind="cosine", steps=2500, seed=0, verbose=True):
    torch.manual_seed(seed)
    net = P2ANet(generative=True, use_cal=False, base=16, mults=(1, 2))
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    path = GPath(kind)
    for i in range(steps):
        for g in opt.param_groups:
            g["lr"] = 1e-3 * 0.5 * (1 + math.cos(math.pi * i / steps))
        loss = training_loss(net, path, toy_batch(256, noise_sd), p_drop_ppg=0.5)
        opt.zero_grad()
        loss.backward()
        opt.step()
    net.eval()
    b = toy_batch(8192, noise_sd)
    lam, sq, dm = information_profile(net, path, b["abp"], b["ppg"], b["cal"], b["has_cal"], given_cal=False)
    res = integrate_information(lam, sq, dm)
    exact = 0.5 * math.log(1 + 1 / noise_sd ** 2)
    if verbose:
        print("logSNR      sq      dm   exact")
        for l, a, d in zip(lam.tolist(), sq.mean(0).tolist(), dm.mean(0).tolist()):
            print(f"{l:6.1f} {a:7.4f} {d:7.4f} {exact_integrand(l, noise_sd):7.4f}")
        print(f"cutoff at logSNR {res['cutoff_logsnr']:.1f}: sq {res['nats_sq']:.3f}  dm {res['nats_dm']:.3f}  "
              f"exact {exact:.3f} nats")
    return res, exact


def test_information_matches_closed_form():
    res, exact = run()
    assert not res["reached_end"], "expected the estimators to disagree at high SNR in this small model"
    assert abs(res["nats_dm"] - exact) < 0.15 * exact, (res["nats_dm"], exact)       # the reported estimate
    assert res["nats_dm"] <= res["nats_sq"] < 1.35 * exact, (res["nats_sq"], exact)   # the upper bracket
    assert res["per_item_dm"].shape == (8192,)


if __name__ == "__main__":
    test_information_matches_closed_form()
    print("ok")
