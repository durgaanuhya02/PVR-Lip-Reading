"""Track B visual model: pre-trained Auto-AVSR encoder -> phoneme and viseme CTC posteriors.

Auto-AVSR (Ma et al., ICASSP 2023; https://github.com/mpc001/auto_avsr) is a 3D-CNN +
ResNet-18 front-end and a 12-layer Conformer (768-d) trained for lip reading on
thousands of hours of video. We keep its front-end and encoder, freeze them except
the top `train_layers` Conformer blocks, and train new heads on LRS3 30 h:
    phoneme CTC head on the final encoder output       (40 = 39 phonemes + blank)
    viseme  CTC head on an intermediate encoder block  (14 = 13 visemes + blank)

The Auto-AVSR code is not vendored: clone the repo and pass its path as `repo_dir`.
Its input pipeline (88x88 grayscale mouth crop, mean 0.421, std 0.165) is the same
as ours, so the frames from ROIDataset are used as they are.
"""
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.text.phonemes import PHONEME_VOCAB, VISEME_VOCAB

KEEP = ("frontend.", "proj_encoder.", "encoder.")


def load_encoder_state(path):
    """Front-end + encoder weights from an Auto-AVSR checkpoint (decoder and CTC are dropped)."""
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:     # Lightning checkpoint
        state = state["state_dict"]
    state = {k[len("model."):] if k.startswith("model.") else k: v for k, v in state.items()}
    return {k: v for k, v in state.items() if k.startswith(KEEP)}


class AutoAVSRCTC(nn.Module):
    def __init__(self, repo_dir="auto_avsr", checkpoint=None, train_layers=2, viseme_layer=6,
                 dropout=0.1, load_pretrained=True):
        super().__init__()
        if repo_dir not in sys.path:
            sys.path.insert(0, repo_dir)
        from espnet.nets.pytorch_backend.encoder.conformer_encoder import ConformerEncoder
        from espnet.nets.pytorch_backend.frontend.resnet import video_resnet

        self.frontend = video_resnet()
        self.proj_encoder = nn.Linear(512, 768)
        self.encoder = ConformerEncoder(attention_dim=768, attention_heads=12, linear_units=3072,
                                        num_blocks=12, cnn_module_kernel=31)
        if load_pretrained and checkpoint:
            missing, _ = self.load_state_dict(load_encoder_state(checkpoint), strict=False)
            missing = [k for k in missing if k.startswith(KEEP)]
            assert not missing, f"checkpoint is missing encoder weights: {missing[:5]}"

        blocks = self.encoder.encoders
        n_frozen = max(len(blocks) - train_layers, 0)
        self.frozen = [self.frontend, self.proj_encoder, self.encoder.embed, *blocks[:n_frozen]]
        for p in self.parameters():
            p.requires_grad = False
        for m in [*blocks[n_frozen:], self.encoder.after_norm]:
            for p in m.parameters():
                p.requires_grad = True

        self.viseme_layer = viseme_layer
        self.drop = nn.Dropout(dropout)
        self.viseme_norm = nn.LayerNorm(768)
        self.phoneme_head = nn.Linear(768, len(PHONEME_VOCAB))
        self.viseme_head = nn.Linear(768, len(VISEME_VOCAB))

    def train(self, mode=True):
        super().train(mode)
        for m in self.frozen:          # keep frozen BatchNorm statistics and dropout fixed
            m.eval()
        return self

    def forward(self, frames, lengths):
        """frames (B, 1, T, 88, 88) normalised grayscale, lengths (B,) -> dict of log-probs (B, T, V)."""
        T = frames.shape[2]
        x = self.proj_encoder(self.frontend(frames.transpose(1, 2)))       # (B, T, 768)
        mask = (torch.arange(T, device=x.device)[None] < lengths.to(x.device)[:, None]).unsqueeze(1)
        xs = self.encoder.embed(x)                                          # (x, pos_emb)
        viseme_logits = None
        for i, blk in enumerate(self.encoder.encoders, 1):
            xs, mask = blk(xs, mask)
            if i == self.viseme_layer:
                viseme_logits = self.viseme_head(self.drop(self.viseme_norm(xs[0])))
        x = self.encoder.after_norm(xs[0] if isinstance(xs, tuple) else xs)
        return {
            "phoneme_logp": F.log_softmax(self.phoneme_head(self.drop(x)).float(), -1),
            "viseme_logp": F.log_softmax(viseme_logits.float(), -1),
            "lengths": lengths,
        }
