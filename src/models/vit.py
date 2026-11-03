"""Track A visual model: VideoMAE ViT-Base -> phoneme and viseme CTC posteriors.

The clip is cut into 16-frame windows. Each window goes through a VideoMAE ViT-Base
(Tong et al., NeurIPS 2022; self-supervised on Kinetics-400), which splits it into
tubelets of 2 frames x 16 x 16 pixels, i.e. 8 time steps per window. Spatial tokens
are averaged per time step, repeated x2 back to the video frame rate (25 fps, so the
CTC heads get one step per frame, as in the Conformer model), and a small temporal
Transformer adds sentence-level context across windows.

To fit a single T4 the lower ViT layers are frozen (optionally with LoRA adapters on
their attention-output and MLP projections) and only the top `train_layers` layers are
fine-tuned.

Heads (same output as VisualCTC, so training, evaluation and PVR are unchanged):
    phoneme CTC head on the final temporal layer       (40 = 39 phonemes + blank)
    viseme  CTC head on an intermediate temporal layer (14 = 13 visemes + blank)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.dataset import MEAN, STD
from src.models.visual import sinusoidal
from src.text.phonemes import PHONEME_VOCAB, VISEME_VOCAB

IMAGENET_MEAN = (0.485, 0.456, 0.406)   # VideoMAE input normalisation
IMAGENET_STD = (0.229, 0.224, 0.225)


class ViTCTC(nn.Module):
    def __init__(self, pretrained="MCG-NJU/videomae-base", image_size=112, window=16,
                 train_layers=4, lora_r=8, lora_alpha=16, d_model=256, heads=4, layers=4,
                 viseme_layer=2, dropout=0.1, window_batch=32, vit_config=None, load_pretrained=True):
        """vit_config: dict of VideoMAEConfig fields for a small random ViT (tests only).
        load_pretrained=False builds the architecture without downloading weights
        (used when a trained checkpoint is loaded right after)."""
        super().__init__()
        from transformers import VideoMAEConfig, VideoMAEModel
        size = dict(image_size=image_size, num_frames=window)
        if vit_config:
            self.encoder = VideoMAEModel(VideoMAEConfig(**{**vit_config, **size}))
        elif load_pretrained:
            self.encoder, info = VideoMAEModel.from_pretrained(pretrained, output_loading_info=True, **size)
            # transformers 5 renamed VideoMAE's q/v biases, which then load as random weights
            assert not info["missing_keys"], (f"pre-trained ViT weights missing: {info['missing_keys'][:3]}; "
                                              "install transformers==4.57.1")
        else:
            self.encoder = VideoMAEModel(VideoMAEConfig.from_pretrained(pretrained, **size))
        cfg = self.encoder.config
        self.window, self.tubelet, self.image_size = window, cfg.tubelet_size, image_size
        self.patches = (image_size // cfg.patch_size) ** 2
        self.window_batch = window_batch
        self._freeze(train_layers, lora_r, lora_alpha)

        self.proj = nn.Linear(cfg.hidden_size, d_model)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, heads, 4 * d_model, dropout, batch_first=True, norm_first=True)
            for _ in range(layers)])
        self.viseme_layer = viseme_layer
        self.viseme_norm, self.phoneme_norm = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.phoneme_head = nn.Linear(d_model, len(PHONEME_VOCAB))
        self.viseme_head = nn.Linear(d_model, len(VISEME_VOCAB))
        self.register_buffer("img_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("img_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    def _freeze(self, train_layers, lora_r, lora_alpha):
        blocks = self.encoder.encoder.layer
        n_frozen = max(len(blocks) - train_layers, 0)
        if lora_r and n_frozen:
            from peft import LoraConfig, inject_adapter_in_model
            # VideoMAE computes query/key/value with F.linear on the raw weights (to add its q/v
            # biases), which bypasses LoRA wrappers, so LoRA goes on the attention output and MLP.
            inject_adapter_in_model(LoraConfig(r=lora_r, lora_alpha=lora_alpha,
                                               target_modules=["output.dense", "intermediate.dense"],
                                               layers_to_transform=list(range(n_frozen)), layers_pattern="layer"),
                                    self.encoder)
        for name, p in self.encoder.named_parameters():
            p.requires_grad = "lora_" in name
        for blk in blocks[n_frozen:]:
            for p in blk.parameters():
                p.requires_grad = True
        if getattr(self.encoder, "layernorm", None) is not None:
            for p in self.encoder.layernorm.parameters():
                p.requires_grad = True

    def forward(self, frames, lengths):
        """frames (B, 1, T, 88, 88) normalised grayscale, lengths (B,) -> dict of log-probs (B, T, V)."""
        B, _, T, H, W = frames.shape
        x = frames[:, 0] * STD + MEAN                                   # back to [0, 1]
        pad = (-T) % self.window
        if pad:
            x = F.pad(x, (0, 0, 0, 0, 0, pad))
        n_win = x.shape[1] // self.window
        x = F.interpolate(x.reshape(-1, 1, H, W), size=self.image_size, mode="bilinear", align_corners=False)
        x = (x.expand(-1, 3, -1, -1) - self.img_mean) / self.img_std
        x = x.view(B * n_win, self.window, 3, self.image_size, self.image_size)

        feats = []
        for chunk in x.split(self.window_batch):
            h = self.encoder(pixel_values=chunk).last_hidden_state      # (n, steps * patches, D)
            feats.append(h.view(len(chunk), -1, self.patches, h.shape[-1]).mean(2))
        x = torch.cat(feats).reshape(B, -1, feats[0].shape[-1])         # (B, n_win * steps, D)
        x = self.proj(x).repeat_interleave(self.tubelet, dim=1)[:, :T]  # one step per video frame
        x = x + sinusoidal(T, x.shape[2], x.device)

        pad_mask = torch.arange(T, device=x.device)[None] >= lengths.to(x.device)[:, None]
        viseme_logits = None
        for i, blk in enumerate(self.blocks, 1):
            x = blk(x, src_key_padding_mask=pad_mask)
            if i == self.viseme_layer:
                viseme_logits = self.viseme_head(self.viseme_norm(x))
        return {
            "phoneme_logp": F.log_softmax(self.phoneme_head(self.phoneme_norm(x)).float(), -1),
            "viseme_logp": F.log_softmax(viseme_logits.float(), -1),
            "lengths": lengths,
        }
