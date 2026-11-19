"""1D U-Net shared by the regression baseline and the generator.

The same backbone is used for both so that "point estimate vs distribution" is the only thing
that differs between arms B1 and M1:

    regression  : input = [PPG, mask]            -> ABP           (t is ignored)
    generation  : input = [noisy ABP, PPG, mask] -> v-prediction  (t is the noise level)

Conditioning enters in three places:
    * PPG waveform        as an input channel (zeroed, with mask channel 0, when dropped)
    * calibration (SBP, DBP of an earlier reading)  as a vector added to the time embedding
    * optional PPG encoder (PaPaGei)  pooled vector -> embedding, feature map -> bottleneck
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t, dim):
    """Sinusoidal features of t in [0, 1]."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
    args = t[:, None] * 1000.0 * freqs[None]
    return torch.cat([args.sin(), args.cos()], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, cin, cout, emb_dim, dropout=0.0):
        super().__init__()
        self.n1 = nn.GroupNorm(8, cin)
        self.c1 = nn.Conv1d(cin, cout, 3, padding=1)
        self.emb = nn.Linear(emb_dim, cout)
        self.n2 = nn.GroupNorm(8, cout)
        self.drop = nn.Dropout(dropout)
        self.c2 = nn.Conv1d(cout, cout, 3, padding=1)
        nn.init.zeros_(self.c2.weight)
        nn.init.zeros_(self.c2.bias)
        self.skip = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, emb):
        h = self.c1(F.silu(self.n1(x)))
        h = h + self.emb(emb)[:, :, None]
        h = self.c2(self.drop(F.silu(self.n2(h))))
        return h + self.skip(x)


class Attention(nn.Module):
    def __init__(self, ch, heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(8, ch)
        self.attn = nn.MultiheadAttention(ch, heads, batch_first=True)

    def forward(self, x):
        h = self.norm(x).transpose(1, 2)
        h, _ = self.attn(h, h, h, need_weights=False)
        return x + h.transpose(1, 2)


class P2ANet(nn.Module):
    def __init__(self, generative=True, use_cal=False, base=64, mults=(1, 2, 4, 4), n_res=2,
                 emb_dim=256, dropout=0.0, encoder=None):
        super().__init__()
        self.generative, self.use_cal, self.emb_dim = generative, use_cal, emb_dim
        self.down_factor = 2 ** (len(mults) - 1)
        self.encoder = encoder

        self.time_mlp = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim))
        if use_cal:
            self.cal_mlp = nn.Sequential(nn.Linear(2, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim))
            self.cal_null = nn.Parameter(torch.zeros(emb_dim))
        chs = [base * m for m in mults]
        if encoder is not None:
            self.enc_vec = nn.Sequential(nn.Linear(encoder.vec_dim, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim))
            self.enc_vec_null = nn.Parameter(torch.zeros(emb_dim))
            self.enc_map = nn.Conv1d(encoder.map_dim, chs[-1], 1)
            nn.init.zeros_(self.enc_map.weight)   # the encoder path starts as a no-op
            nn.init.zeros_(self.enc_map.bias)

        cin = (1 if generative else 0) + 2        # [noisy ABP] + PPG + PPG-present mask
        self.inp = nn.Conv1d(cin, chs[0], 3, padding=1)
        self.down = nn.ModuleList()
        skip_ch, c = [chs[0]], chs[0]
        for lvl, co in enumerate(chs):
            for _ in range(n_res):
                self.down.append(ResBlock(c, co, emb_dim, dropout))
                c = co
                skip_ch.append(c)
            if lvl < len(chs) - 1:
                self.down.append(nn.Conv1d(c, c, 3, stride=2, padding=1))
                skip_ch.append(c)
        self.mid1 = ResBlock(c, c, emb_dim, dropout)
        self.mid_attn = Attention(c)
        self.mid2 = ResBlock(c, c, emb_dim, dropout)
        self.up = nn.ModuleList()
        for lvl, co in reversed(list(enumerate(chs))):
            for _ in range(n_res + 1):
                self.up.append(ResBlock(c + skip_ch.pop(), co, emb_dim, dropout))
                c = co
            if lvl > 0:
                self.up.append(nn.Upsample(scale_factor=2, mode="nearest"))
        self.out = nn.Sequential(nn.GroupNorm(8, c), nn.SiLU(), nn.Conv1d(c, 1, 3, padding=1))

    def forward(self, x_t, t, ppg, ppg_on, cal=None, cal_on=None):
        """x_t: (B,1,L) noisy ABP or None; t: (B,) in [0,1] or None;
        ppg: (B,1,L); ppg_on, cal_on: (B,) 1 = condition present, 0 = dropped; cal: (B,2)."""
        b, _, n = ppg.shape
        on = ppg_on.view(b, 1, 1).to(ppg.dtype)
        feats = [ppg * on, on.expand(b, 1, n)]
        if self.generative:
            feats.insert(0, x_t)
        x = torch.cat(feats, dim=1)
        pad = (-n) % self.down_factor               # 1250 -> 1256 for a 3-level stride-2 stack
        if pad:
            x = F.pad(x, (0, pad), mode="reflect")

        if self.generative:
            emb = self.time_mlp(timestep_embedding(t, self.emb_dim))
        else:
            emb = torch.zeros(b, self.emb_dim, device=ppg.device, dtype=ppg.dtype)
        if self.use_cal:
            c_on = cal_on.view(b, 1).to(ppg.dtype)
            emb = emb + c_on * self.cal_mlp(cal) + (1 - c_on) * self.cal_null
        enc_map = None
        if self.encoder is not None:
            vec, enc_map = self.encoder(ppg)
            p_on = ppg_on.view(b, 1).to(ppg.dtype)
            emb = emb + p_on * self.enc_vec(vec) + (1 - p_on) * self.enc_vec_null
            enc_map = enc_map * on

        h = self.inp(x)
        skips = [h]
        for m in self.down:
            h = m(h, emb) if isinstance(m, ResBlock) else m(h)
            skips.append(h)
        if enc_map is not None:
            h = h + self.enc_map(F.interpolate(enc_map, size=h.shape[-1], mode="linear", align_corners=False))
        h = self.mid2(self.mid_attn(self.mid1(h, emb)), emb)
        for m in self.up:
            if isinstance(m, ResBlock):
                h = m(torch.cat([h, skips.pop()], dim=1), emb)
            else:
                h = m(h)
        return self.out(h)[..., :n]
