"""PaPaGei as the conditioning encoder, in four modes that together answer "does the PPG foundation
model help a generator, and is it the pretraining that helps?":

    frozen   pretrained weights, nothing trained                      (features only)
    lora     pretrained weights frozen + low-rank updates on every conv (the adapter arm)
    full     pretrained weights, everything trained
    scratch  same architecture, random init, everything trained       (control for pretraining)

PaPaGei-S is an 18-block 1D ResNet (about 5M weights), not a transformer, so "LoRA" here is a
low-rank update of each conv kernel: W + (alpha/r) * B(A(x)), with A a k-wide conv down to r channels
and B a 1x1 conv back up, B initialised to zero so training starts exactly at the pretrained model.

The PaPaGei code is NOT copied into this repo. Clone it and pass its path:
    git clone https://github.com/Nokia-Bell-Labs/papagei-foundation-model
Weights (papagei_s.pt) come from https://zenodo.org/records/13983110
"""
import importlib
import sys

import torch
import torch.nn as nn

PAPAGEI_S = dict(in_channels=1, base_filters=32, kernel_size=3, stride=2, groups=1,
                 n_block=18, n_classes=512, n_experts=3)


class LoRAConv1d(nn.Module):
    def __init__(self, base: nn.Conv1d, rank=8, alpha=16.0):
        super().__init__()
        assert base.groups == 1 and base.padding == (0,), "PaPaGei pads outside the conv"
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.A = nn.Conv1d(base.in_channels, rank, base.kernel_size[0], stride=base.stride[0], bias=False)
        self.B = nn.Conv1d(rank, base.out_channels, 1, bias=False)
        nn.init.kaiming_uniform_(self.A.weight, a=5 ** 0.5)
        nn.init.zeros_(self.B.weight)
        self.scale = alpha / rank

    def forward(self, x):
        return self.base(x) + self.scale * self.B(self.A(x))


class PPGEncoder(nn.Module):
    """forward(ppg) -> (pooled vector (B, 512), feature map (B, 192, 157)).

    The pooled vector is PaPaGei's 512-d embedding (the one its linear probes use). The feature map
    keeps time resolution, which a waveform generator needs: it concatenates the outputs of two
    intermediate blocks (after block 5: 64 channels x 157 steps; after block 9: 128 x 40), resampled
    to the longer length.
    """
    vec_dim = 512

    def __init__(self, repo, weights=None, mode="frozen", rank=8, taps=(5, 9), load=True):
        """load=False builds the structure only (used when restoring one of our own checkpoints)."""
        super().__init__()
        assert mode in ("frozen", "lora", "full", "scratch")
        assert repo, "pass --papagei-repo (path to a clone of the PaPaGei repository)"
        if repo not in sys.path:
            sys.path.insert(0, repo)
        resnet = importlib.import_module("models.resnet")
        self.net = resnet.ResNet1DMoE(**PAPAGEI_S, use_do=False)   # dropout off: we want features, not regularisation
        if mode != "scratch" and load:
            assert weights, f"mode={mode} needs --papagei-weights"
            state = torch.load(weights, map_location="cpu")
            state = {k.removeprefix("module."): v for k, v in state.items()}
            self.net.load_state_dict(state)
        self.mode, self.taps = mode, taps
        self.map_dim = sum(self.net.basicblock_list[i].out_channels for i in taps)
        if mode in ("frozen", "lora"):
            for p in self.net.parameters():
                p.requires_grad_(False)
        if mode == "lora":
            for m in list(self.net.modules()):
                if isinstance(m, resnet.MyConv1dPadSame):
                    m.conv = LoRAConv1d(m.conv, rank)

    def train(self, mode=True):
        # Pretrained variants keep BatchNorm on its pretraining statistics; only the from-scratch
        # control has to learn them.
        super().train(mode)
        if self.mode != "scratch":
            for m in self.net.modules():
                if isinstance(m, nn.BatchNorm1d):
                    m.eval()
        return self

    def forward(self, ppg):
        net = self.net
        with torch.set_grad_enabled(self.mode != "frozen" and torch.is_grad_enabled()):
            h = net.first_block_relu(net.first_block_bn(net.first_block_conv(ppg)))
            maps = []
            for i, block in enumerate(net.basicblock_list):
                h = block(h)
                if i in self.taps:
                    maps.append(h)
            vec = net.dense(net.final_relu(net.final_bn(h)).mean(-1))   # = outputs[0] of the PaPaGei README
            size = maps[0].shape[-1]
            fmap = torch.cat([nn.functional.interpolate(m, size=size, mode="linear", align_corners=False)
                              for m in maps], dim=1)
        return vec, fmap

    def n_trainable(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
