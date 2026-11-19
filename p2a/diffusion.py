"""Conditional generation of ABP given the pulse and (optionally) a calibration reading.

Forward process:  x_t = alpha(t) * x0 + sigma(t) * eps,   t in [0, 1],  eps ~ N(0, I)

Two paths share all the code below, which makes "diffusion vs flow matching" a one-flag ablation:
    cosine : alpha = cos(pi t / 2), sigma = sin(pi t / 2)   network predicts v = alpha*eps - sigma*x0
    linear : alpha = 1 - t,         sigma = t               network predicts u = eps - x0   (rectified flow)
In both cases the network output is d x_t / dt up to a constant, so one deterministic sampler serves both.

We predict v/u instead of eps because absolute pressure level is the whole point here: at t = 1 the
input is pure noise and an eps-prediction carries no information about x0, whereas v at t = 1 equals
-x0, so the network is trained to state its best guess of the level from the conditions alone.

Training drops the two conditions independently, which is what lets one network answer four
questions at sampling time: p(abp), p(abp | cal), p(abp | ppg), p(abp | ppg, cal).
"""
import math

import torch


class Path:
    def __init__(self, kind="cosine"):
        assert kind in ("cosine", "linear")
        self.kind = kind

    def alpha_sigma(self, t):
        if self.kind == "cosine":
            return torch.cos(0.5 * math.pi * t), torch.sin(0.5 * math.pi * t)
        return 1 - t, t

    def target(self, x0, eps, t):
        a, s = self.alpha_sigma(t)
        return a * eps - s * x0 if self.kind == "cosine" else eps - x0

    def to_x0_eps(self, x_t, out, t):
        """Invert the parameterisation: network output -> (x0 estimate, eps estimate)."""
        a, s = self.alpha_sigma(t)
        if self.kind == "cosine":
            return a * x_t - s * out, s * x_t + a * out
        return x_t - s * out, x_t + a * out


def _b(t, x):
    return t.view(-1, *([1] * (x.dim() - 1)))


def training_loss(model, path, batch, p_drop_ppg=0.1, p_drop_cal=0.5):
    """One training step's loss. batch holds abp, ppg (B,1,L), cal (B,2), has_cal (B,)."""
    x0, ppg = batch["abp"], batch["ppg"]
    b = x0.shape[0]
    t = torch.rand(b, device=x0.device)
    eps = torch.randn_like(x0)
    a, s = path.alpha_sigma(_b(t, x0))
    x_t = a * x0 + s * eps
    ppg_on = (torch.rand(b, device=x0.device) >= p_drop_ppg).float()
    cal_on = batch["has_cal"] * (torch.rand(b, device=x0.device) >= p_drop_cal).float()
    out = model(x_t, t, ppg, ppg_on, batch["cal"], cal_on)
    return ((out - path.target(x0, eps, _b(t, x0))) ** 2).mean()


def guided(model, x_t, t, ppg, cal, has_cal, w_ppg=1.0, w_cal=1.0):
    """Two-weight classifier-free guidance.

        out = out(-, -) + w_cal * [out(cal, -) - out(-, -)] + w_ppg * [out(cal, ppg) - out(cal, -)]

    w_cal = w_ppg = 1 is the plain conditional model. w_ppg > 1 amplifies whatever the pulse adds on
    top of the calibration; w_ppg = 0 ignores the pulse. For items without calibration (has_cal = 0)
    the middle term vanishes and this is ordinary single-condition guidance on the pulse.
    """
    one, zero = torch.ones_like(has_cal), torch.zeros_like(has_cal)
    if w_ppg == 1.0 and w_cal == 1.0:
        return model(x_t, t, ppg, one, cal, has_cal)
    cat = lambda *xs: torch.cat(xs, dim=0)
    out = model(cat(x_t, x_t, x_t), cat(t, t, t), cat(ppg, ppg, ppg), cat(zero, zero, one),
                cat(cal, cal, cal), cat(zero, has_cal, has_cal))
    o_00, o_c0, o_cp = out.chunk(3, dim=0)
    return o_00 + w_cal * (o_c0 - o_00) + w_ppg * (o_cp - o_c0)


@torch.no_grad()
def sample(model, path, ppg, cal, has_cal, steps=50, w_ppg=1.0, w_cal=1.0, generator=None):
    """Deterministic sampler (DDIM with eta = 0; for the linear path this is exactly Euler on the ODE).
    Randomness comes only from the starting noise, so different seeds give different posterior draws."""
    x = torch.randn(ppg.shape, device=ppg.device, generator=generator)
    ts = torch.linspace(1.0, 0.0, steps + 1, device=ppg.device)
    for i in range(steps):
        t = ts[i].expand(ppg.shape[0])
        out = guided(model, x, t, ppg, cal, has_cal, w_ppg, w_cal)
        x0, eps = path.to_x0_eps(x, out, _b(t, x))
        a, s = path.alpha_sigma(ts[i + 1])
        x = a * x0 + s * eps
    return x


@torch.no_grad()
def information_profile(model, path, x0, ppg, cal, has_cal, given_cal=True, logsnr=(-10.0, 6.0),
                        n_levels=33, generator=None):
    """What the pulse tells the model about the pressure waveform, level by noise level.

    For a Gaussian channel, mutual information is half the integral over log-SNR of the gap between the
    denoising errors without and with the extra condition (I-MMSE; Kong et al., ICLR 2023 / 2024):

        I(abp ; ppg | c) = 1/2 * integral  [ mse(eps | x_t, c) - mse(eps | x_t, ppg, c) ]  d(logSNR)

    with c = calibration (given_cal=True) or nothing. For perfect denoisers that gap can be written in
    two ways that are equal in expectation, and we return both per item and per level:

        sq = || eps_hat(x_t | c) - eps_hat(x_t | ppg, c) ||^2          never negative, low variance
        dm = || eps - eps_hat(x_t | c) ||^2 - || eps - eps_hat(x_t | ppg, c) ||^2   noisy, can go negative

    They stop being equal exactly where the network is not a good denoiser, which in practice is the
    high-SNR end: there the true gap is tiny, `sq` adds up approximation error over every sample of
    the waveform and grows, and `dm` drifts away from it (negative, in the toy). tests/test_information.py
    shows this on a problem with a known answer. integrate_information() therefore stops where `dm`
    first turns non-positive.

    Returns lam (n,), sq (B, n), dm (B, n).
    """
    b = x0.shape[0]
    cal_on = has_cal if given_cal else torch.zeros_like(has_cal)
    one, zero = torch.ones_like(has_cal), torch.zeros_like(has_cal)
    lam = torch.linspace(logsnr[0], logsnr[1], n_levels, device=x0.device)
    sq, dm = [], []
    for l in lam:
        r = torch.exp(-0.5 * l)                                   # sigma / alpha at this log-SNR
        t = ((2 / math.pi) * torch.atan(r) if path.kind == "cosine" else r / (1 + r)).expand(b)
        eps = torch.randn(x0.shape, device=x0.device, generator=generator)
        a, s = path.alpha_sigma(_b(t, x0))
        x_t = a * x0 + s * eps
        out = model(torch.cat([x_t, x_t]), torch.cat([t, t]), torch.cat([ppg, ppg]),
                    torch.cat([zero, one]), torch.cat([cal, cal]), torch.cat([cal_on, cal_on])).float()
        o_without, o_with = out.chunk(2)
        e_without = path.to_x0_eps(x_t, o_without, _b(t, x0))[1]
        e_with = path.to_x0_eps(x_t, o_with, _b(t, x0))[1]
        sq.append(((e_without - e_with) ** 2).flatten(1).sum(1))
        dm.append(((eps - e_without) ** 2).flatten(1).sum(1) - ((eps - e_with) ** 2).flatten(1).sum(1))
    return lam, torch.stack(sq, 1), torch.stack(dm, 1)


def integrate_information(lam, sq, dm):
    """Integrate the profile up to the log-SNR where the model stops being a trustworthy denoiser.

    The true gap is never negative (an extra condition cannot make the best denoiser worse), so the
    first place after its peak where the batch-mean `dm` drops to zero or below marks where network
    error has taken over, and we stop there. Up to that point:
        nats_dm  integral of dm : the estimate to report (close to the truth in the toy test)
        nats_sq  integral of sq : biased upward by approximation error, so read it as an upper bracket
    The integral is truncated, so whatever information lives above the cutoff is not counted; report
    the cutoff next to the number. `per_item_dm` is noisy for a single segment and is meant to be
    averaged within strata.
    """
    a, d = sq.mean(0), dm.mean(0)
    peak = int(d.argmax())
    below = (d[peak:] <= 0).nonzero()
    cut = max(2, peak + int(below[0]) if len(below) else len(lam))
    area = lambda y: 0.5 * torch.trapezoid(y[..., :cut], lam[:cut], dim=-1)
    return dict(nats_dm=float(area(d)), nats_sq=float(area(a)),
                cutoff_logsnr=float(lam[cut - 1]), reached_end=cut == len(lam),
                per_item_dm=area(dm).cpu().numpy(), per_item_sq=area(sq).cpu().numpy(),
                profile=dict(logsnr=lam.tolist(), sq=a.tolist(), dm=d.tolist()))
