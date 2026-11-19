"""Fast checks with known answers. Run:  python -m pytest tests/test_core.py -q

Set PAPAGEI_REPO=/path/to/papagei-foundation-model to also run the encoder / LoRA checks.
"""
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from p2a import FS, SEG_LEN                                    # noqa: E402
from p2a.checker import bp_from_waveform, derive_bp            # noqa: E402
from p2a.corrupt import KINDS, corrupt                         # noqa: E402
from p2a.data import PulseDataset, make_synthetic, synth_segment  # noqa: E402
from p2a.diffusion import Path as GPath, guided, sample, training_loss  # noqa: E402
from p2a.metrics import auroc, coverage, crps, tracking        # noqa: E402
from p2a.unet1d import P2ANet                                  # noqa: E402


def test_checker_recovers_known_pressures():
    """Build beats whose peak and trough are known exactly, with a dicrotic bump that must not count as a beat."""
    t = np.arange(SEG_LEN) / FS
    for hr, dbp, pp in [(60, 70, 50), (95, 45, 30), (130, 85, 70)]:
        phi = (t * hr / 60) % 1.0
        shape = np.exp(-0.5 * ((phi - 0.2) / 0.06) ** 2) + 0.35 * np.exp(-0.5 * ((phi - 0.5) / 0.07) ** 2)
        abp = dbp + pp * (shape - shape.min()) / (shape.max() - shape.min())
        sbp_hat, dbp_hat, map_hat, beats = bp_from_waveform(abp)
        assert abs(sbp_hat - (dbp + pp)) < 0.5 and abs(dbp_hat - dbp) < 0.5
        assert abs(beats - hr * 10 / 60) <= 1, "dicrotic bumps were counted as beats"
        assert dbp < map_hat < dbp + pp


def test_checker_flags_flat_line():
    assert bp_from_waveform(np.full(SEG_LEN, 80.0))[3] == 0
    assert derive_bp(np.zeros((3, 2, SEG_LEN))).shape == (3, 2, 4)


def test_crps_matches_gaussian_closed_form():
    rng = np.random.default_rng(0)
    mu, sd, y = 3.0, 2.0, np.array([1.0, 3.0, 8.0])
    z = (y - mu) / sd
    pdf = np.exp(-0.5 * z ** 2) / math.sqrt(2 * math.pi)
    cdf = 0.5 * (1 + np.vectorize(math.erf)(z / math.sqrt(2)))
    exact = sd * (z * (2 * cdf - 1) + 2 * pdf - 1 / math.sqrt(math.pi))
    est = crps(mu + sd * rng.standard_normal((3, 20000)), y)
    assert np.allclose(est, exact, rtol=0.03, atol=0.02)


def test_crps_of_point_prediction_is_absolute_error():
    assert np.allclose(crps(np.array([[2.0], [5.0]]), np.array([3.5, 1.0])), [1.5, 4.0])


def test_coverage_and_auroc_and_tracking():
    rng = np.random.default_rng(1)
    y = rng.standard_normal(4000)
    cov, nominal, width = coverage(rng.standard_normal((4000, 399)), y, 0.9)
    assert nominal == 0.9 and abs(cov - 0.9) < 0.02 and abs(width - 2 * 1.645) < 0.1
    for k, expect in [(8, 7 / 9), (19, 0.9), (39, 0.9)]:        # few draws: the target is NOT 0.9 unless K is chosen for it
        cov, nominal, _ = coverage(rng.standard_normal((20000, k)), rng.standard_normal(20000), 0.9)
        assert abs(nominal - expect) < 1e-12 and abs(cov - nominal) < 0.01
    label = rng.random(2000) < 0.3
    score = label + rng.standard_normal(2000)
    from sklearn.metrics import roc_auc_score
    assert abs(auroc(score, label) - roc_auc_score(label, score)) < 1e-9
    cal, change = rng.normal(90, 10, 500), rng.normal(0, 8, 500)
    assert abs(tracking(cal + 0.4 * change, cal + change, cal)[0] - 0.4) < 1e-6   # follows 40% of the change
    assert tracking(cal.copy(), cal + change, cal)[0] == 0.0                       # repeats the calibration


@pytest.mark.parametrize("kind", ["cosine", "linear"])
def test_path_parameterisation_inverts(kind):
    path = GPath(kind)
    x0, eps = torch.randn(4, 1, 64), torch.randn(4, 1, 64)
    t = torch.tensor([0.05, 0.3, 0.7, 0.95]).view(4, 1, 1)
    a, s = path.alpha_sigma(t)
    x0_hat, eps_hat = path.to_x0_eps(a * x0 + s * eps, path.target(x0, eps, t), t)
    assert torch.allclose(x0_hat, x0, atol=1e-5) and torch.allclose(eps_hat, eps, atol=1e-5)


def _batch(b=3, cal=True):
    return dict(abp=torch.randn(b, 1, SEG_LEN), ppg=torch.randn(b, 1, SEG_LEN), cal=torch.randn(b, 2),
                has_cal=torch.ones(b) if cal else torch.zeros(b))


@pytest.mark.parametrize("generative,use_cal", [(True, True), (True, False), (False, True), (False, False)])
def test_network_shapes(generative, use_cal):
    net = P2ANet(generative=generative, use_cal=use_cal, base=16)
    b = _batch()
    out = net(b["abp"] if generative else None, torch.rand(3), b["ppg"], torch.ones(3), b["cal"], b["has_cal"])
    assert out.shape == (3, 1, SEG_LEN)


def test_guidance_reduces_to_the_four_conditionals():
    torch.manual_seed(0)
    net = P2ANet(generative=True, use_cal=True, base=16).eval()
    for p in net.parameters():                  # output conv is zero-init; randomise so the test is not trivial
        torch.nn.init.normal_(p, std=0.05)
    b, t = _batch(), torch.rand(3)
    on, off = torch.ones(3), torch.zeros(3)
    x = b["abp"]
    with torch.no_grad():
        full = net(x, t, b["ppg"], on, b["cal"], on)
        assert torch.allclose(guided(net, x, t, b["ppg"], b["cal"], on, 1.0, 1.0), full, atol=1e-5)
        assert torch.allclose(guided(net, x, t, b["ppg"], b["cal"], on, 1.0 + 1e-7, 1.0), full, atol=1e-4)  # 3-pass route
        assert torch.allclose(guided(net, x, t, b["ppg"], b["cal"], on, 0.0, 1.0), net(x, t, b["ppg"], off, b["cal"], on), atol=1e-4)
        assert torch.allclose(guided(net, x, t, b["ppg"], b["cal"], on, 0.0, 0.0), net(x, t, b["ppg"], off, b["cal"], off), atol=1e-4)
        # a dropped pulse must not leak through: changing the PPG cannot change the output
        assert torch.allclose(net(x, t, b["ppg"], off, b["cal"], on), net(x, t, torch.randn_like(b["ppg"]), off, b["cal"], on), atol=1e-6)
        assert torch.allclose(net(x, t, b["ppg"], on, b["cal"], off), net(x, t, b["ppg"], on, torch.randn(3, 2), off), atol=1e-6)


@pytest.mark.parametrize("kind", ["cosine", "linear"])
def test_loss_and_sampler_run(kind):
    net = P2ANet(generative=True, use_cal=True, base=16)
    b = _batch()
    loss = training_loss(net, GPath(kind), b)
    loss.backward()
    assert torch.isfinite(loss)
    out = sample(net.eval(), GPath(kind), b["ppg"], b["cal"], b["has_cal"], steps=4, w_ppg=1.5)
    assert out.shape == (3, 1, SEG_LEN) and torch.isfinite(out).all()


@pytest.mark.parametrize("kind", KINDS)
def test_corruptions_change_signal_and_stay_normalised(kind):
    rng = np.random.default_rng(0)
    ppg = np.stack([synth_segment(rng, 90, 50, 75, 0.0)[0] for _ in range(4)])
    x = torch.tensor(ppg, dtype=torch.float32)[:, None]
    x = (x - x.mean(-1, keepdim=True)) / x.std(-1, keepdim=True)
    y = corrupt(x, kind, 0.7, torch.Generator().manual_seed(0))
    assert y.shape == x.shape and torch.isfinite(y).all()
    assert (y - x).abs().mean() > 0.05
    assert torch.allclose(y.mean(-1), torch.zeros(4, 1), atol=1e-4)
    assert torch.equal(corrupt(x, kind, 0.0), x)


def test_pairing_rules(tmp_path):
    make_synthetic(tmp_path, n_subj=(6, 2, 2), seg_per_subj=5, seed=0)
    for mode in ("random", "fixed", "far"):
        ds = PulseDataset(tmp_path, "train", pair=mode)
        for i in range(len(ds)):
            j = ds._partner(i)
            assert j != i and ds.subj[j] == ds.subj[i]
    fixed = PulseDataset(tmp_path, "train", pair="fixed")
    assert [fixed._partner(i) for i in range(30)] == [fixed._partner(i) for i in range(30)]
    far = PulseDataset(tmp_path, "train", pair="far")
    members = np.where(far.subj == far.subj[0])[0]
    assert far._partner(0) == members[np.abs(far.bp[members, 2] - far.bp[0, 2]).argmax()]
    other = PulseDataset(tmp_path, "train", pair="other")
    assert all(other.subj[other._partner(i)] != other.subj[i] for i in range(30))
    item = fixed[0]
    assert item["ppg"].shape == (1, SEG_LEN) and abs(float(item["ppg"].mean())) < 1e-4 and item["has_cal"] == 1
    assert PulseDataset(tmp_path, "train", pair="none")[0]["has_cal"] == 0


@pytest.mark.skipif("PAPAGEI_REPO" not in os.environ, reason="set PAPAGEI_REPO to test the PaPaGei wrapper")
def test_encoder_modes_and_lora_starts_at_identity():
    from p2a.encoders import PPGEncoder
    repo = os.environ["PAPAGEI_REPO"]
    torch.manual_seed(0)
    scratch = PPGEncoder(repo, mode="scratch").eval()
    vec, fmap = scratch(torch.randn(2, 1, SEG_LEN))
    assert vec.shape == (2, 512) and fmap.shape == (2, scratch.map_dim, 157)
    lora = PPGEncoder(repo, mode="lora", load=False).eval()
    # put the same base weights under the adapters
    state = {k.replace(".conv.weight", ".conv.base.weight").replace(".conv.bias", ".conv.base.bias"): v
             for k, v in scratch.net.state_dict().items()}
    missing, unexpected = lora.net.load_state_dict(state, strict=False)
    assert not unexpected and all(".A." in k or ".B." in k for k in missing)
    x = torch.randn(2, 1, SEG_LEN)
    v1, m1 = scratch(x)
    v2, m2 = lora(x)
    assert torch.allclose(v1, v2, atol=1e-5) and torch.allclose(m1, m2, atol=1e-5), "LoRA must start as a no-op"
    n_lora, n_full = lora.n_trainable(), scratch.n_trainable()
    assert 0 < n_lora < 0.25 * n_full
    assert all((".A." in n or ".B." in n) for n, p in lora.named_parameters() if p.requires_grad)
    net = P2ANet(generative=True, use_cal=True, base=16, encoder=scratch)
    b = _batch(2)
    assert net(b["abp"], torch.rand(2), b["ppg"], torch.ones(2), b["cal"], b["has_cal"]).shape == (2, 1, SEG_LEN)


@pytest.mark.skipif("PAPAGEI_REPO" not in os.environ, reason="set PAPAGEI_REPO to test the PaPaGei wrapper")
def test_lora_adapters_receive_gradient_once_training_starts():
    """At step 0 nothing upstream of the zero-initialised residual convs gets a gradient (by design).
    From the second step the LoRA B matrices must, and the frozen base convs never do."""
    from p2a.encoders import PPGEncoder
    torch.manual_seed(0)
    enc = PPGEncoder(os.environ["PAPAGEI_REPO"], mode="lora", load=False)
    net = P2ANet(generative=True, use_cal=True, base=16, encoder=enc).train()
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=1e-3)
    b = _batch(4)
    for _ in range(2):
        loss = training_loss(net, GPath("cosine"), b, p_drop_ppg=0.0)
        opt.zero_grad()
        loss.backward()
        opt.step()
    grad_b = sum(p.grad.abs().sum() for n, p in enc.named_parameters() if ".B." in n)
    assert grad_b > 0
    assert all(p.grad is None for n, p in enc.named_parameters() if ".base." in n)
    assert all(not m.training for m in enc.modules() if isinstance(m, torch.nn.BatchNorm1d))
