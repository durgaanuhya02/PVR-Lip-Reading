"""Stage 1 visual model: mouth video -> phoneme and viseme CTC posteriors.

Front-end: 3D convolution + 2D ResNet-18 per frame (the standard lip-reading
front-end of Stafylakis & Tzimiropoulos 2017 / Ma et al. 2021).
Back-end: Conformer encoder (Gulati et al. 2020) at the video frame rate.
Heads:
    phoneme CTC head on the final layer (40 = 39 phonemes + blank)
    viseme  CTC head on an intermediate layer (14 = 13 visemes + blank)
The viseme posteriors are what the Verify stage scores sentence candidates with.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.text.phonemes import PHONEME_ID_TO_VISEME_ID, PHONEME_VOCAB, VISEME_VOCAB


# --------------------------------------------------------------------------- front-end
class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.down = None
        if stride != 1 or cin != cout:
            self.down = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        idt = x if self.down is None else self.down(x)
        x = F.relu(self.bn1(self.conv1(x)))
        return F.relu(self.bn2(self.conv2(x)) + idt)


class Frontend3D(nn.Module):
    """(B, 1, T, 88, 88) -> (B, T, 512)"""

    def __init__(self, width=64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(1, width, (5, 7, 7), (1, 2, 2), (2, 3, 3), bias=False),
            nn.BatchNorm3d(width), nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 3, 3), (1, 2, 2), (0, 1, 1)),
        )
        chans = [width, width * 2, width * 4, width * 8]
        layers, cin = [], width
        for i, c in enumerate(chans):
            layers += [BasicBlock(cin, c, 1 if i == 0 else 2), BasicBlock(c, c)]
            cin = c
        self.trunk = nn.Sequential(*layers)
        self.out_dim = chans[-1]

    def forward(self, x):
        B, _, T = x.shape[:3]
        x = self.stem(x)                                   # (B, C, T, H, W)
        x = x.transpose(1, 2).flatten(0, 1)                # (B*T, C, H, W)
        x = self.trunk(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)         # (B*T, 512)
        return x.view(B, T, -1)


# --------------------------------------------------------------------------- Conformer
class FeedForward(nn.Module):
    def __init__(self, d, mult=4, p=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d * mult), nn.SiLU(), nn.Dropout(p),
                                 nn.Linear(d * mult, d), nn.Dropout(p))

    def forward(self, x):
        return self.net(x)


class ConvModule(nn.Module):
    def __init__(self, d, kernel=31, p=0.1):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.pw1 = nn.Conv1d(d, 2 * d, 1)
        self.dw = nn.Conv1d(d, d, kernel, padding=kernel // 2, groups=d)
        self.bn = nn.BatchNorm1d(d)
        self.pw2 = nn.Conv1d(d, d, 1)
        self.drop = nn.Dropout(p)

    def forward(self, x, pad_mask):
        x = self.ln(x).masked_fill(pad_mask.unsqueeze(-1), 0.0).transpose(1, 2)
        x = F.glu(self.pw1(x), dim=1)
        x = self.pw2(F.silu(self.bn(self.dw(x))))
        return self.drop(x.transpose(1, 2))


class ConformerBlock(nn.Module):
    def __init__(self, d, heads, kernel=31, p=0.1):
        super().__init__()
        self.ff1, self.ff2 = FeedForward(d, p=p), FeedForward(d, p=p)
        self.ln_att = nn.LayerNorm(d)
        self.att = nn.MultiheadAttention(d, heads, dropout=p, batch_first=True)
        self.drop = nn.Dropout(p)
        self.conv = ConvModule(d, kernel, p)
        self.ln_out = nn.LayerNorm(d)

    def forward(self, x, pad_mask):
        x = x + 0.5 * self.ff1(x)
        h = self.ln_att(x)
        x = x + self.drop(self.att(h, h, h, key_padding_mask=pad_mask, need_weights=False)[0])
        x = x + self.conv(x, pad_mask)
        x = x + 0.5 * self.ff2(x)
        return self.ln_out(x)


def sinusoidal(T, d, device):
    pos = torch.arange(T, device=device).unsqueeze(1)
    div = torch.exp(torch.arange(0, d, 2, device=device) * (-math.log(10000.0) / d))
    pe = torch.zeros(T, d, device=device)
    pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
    return pe


# --------------------------------------------------------------------------- full model
class VisualCTC(nn.Module):
    def __init__(self, d_model=256, heads=4, layers=12, viseme_layer=6, frontend_width=64, dropout=0.1):
        super().__init__()
        self.frontend = Frontend3D(frontend_width)
        self.proj = nn.Linear(self.frontend.out_dim, d_model)
        self.blocks = nn.ModuleList([ConformerBlock(d_model, heads, p=dropout) for _ in range(layers)])
        self.viseme_layer = viseme_layer
        self.phoneme_head = nn.Linear(d_model, len(PHONEME_VOCAB))
        self.viseme_head = nn.Linear(d_model, len(VISEME_VOCAB))

    def forward(self, frames, lengths):
        """frames (B, 1, T, 88, 88), lengths (B,) -> dict of log-probs (B, T, V)."""
        x = self.proj(self.frontend(frames))
        T = x.shape[1]
        x = x + sinusoidal(T, x.shape[2], x.device)
        pad_mask = torch.arange(T, device=x.device)[None] >= lengths.to(x.device)[:, None]
        viseme_logits = None
        for i, blk in enumerate(self.blocks, 1):
            x = blk(x, pad_mask)
            if i == self.viseme_layer:
                viseme_logits = self.viseme_head(x)
        return {
            "phoneme_logp": F.log_softmax(self.phoneme_head(x).float(), -1),
            "viseme_logp": F.log_softmax(viseme_logits.float(), -1),
            "lengths": lengths,
        }


def ctc_losses(out, batch, viseme_weight=0.3):
    """Joint loss: phoneme CTC + viseme_weight * viseme CTC (mean over batch)."""
    lens = out["lengths"].cpu()
    ph = F.ctc_loss(out["phoneme_logp"].transpose(0, 1), batch["phonemes"], lens,
                    batch["phoneme_lengths"], blank=0, zero_infinity=True)
    vi = F.ctc_loss(out["viseme_logp"].transpose(0, 1), batch["visemes"], lens,
                    batch["viseme_lengths"], blank=0, zero_infinity=True)
    return ph + viseme_weight * vi, {"phoneme_ctc": ph.item(), "viseme_ctc": vi.item()}


def visemes_from_phonemes(phoneme_logp):
    """Viseme log-probs from the phoneme head: the probability of a viseme is the sum of the
    probabilities of its phonemes (e.g. P + B + M -> BILABIAL; blank -> blank).
    (..., 40) -> (..., 14). An alternative to the viseme head when that head is weaker."""
    group = torch.tensor(PHONEME_ID_TO_VISEME_ID, device=phoneme_logp.device)
    return torch.stack([phoneme_logp[..., group == v].logsumexp(-1) for v in range(len(VISEME_VOCAB))], -1)


def greedy_decode(logp, lengths):
    """Best-path CTC decoding -> list of id lists (blanks and repeats removed)."""
    best = logp.argmax(-1).cpu()
    out = []
    for seq, n in zip(best, lengths.cpu()):
        ids, prev = [], 0
        for t in seq[:n].tolist():
            if t != prev and t != 0:
                ids.append(t)
            prev = t
        out.append(ids)
    return out
