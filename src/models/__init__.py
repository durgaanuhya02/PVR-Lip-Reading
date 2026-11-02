"""Stage-1 visual models. All return the same dict (phoneme_logp, viseme_logp, lengths)."""


def build_visual(model_cfg, load_pretrained=True):
    """Build a visual model from the `model` section of a config.

    arch: conformer (3D-CNN + ResNet-18 + Conformer, from scratch)
          vit       (Track A: VideoMAE ViT-Base + temporal Transformer)
          autoavsr  (Track B: pre-trained Auto-AVSR encoder)
    load_pretrained=False skips downloading pre-trained weights, for when a trained
    checkpoint is loaded right after.
    """
    cfg = dict(model_cfg)
    arch = cfg.pop("arch", "conformer")
    if arch == "conformer":
        from src.models.visual import VisualCTC
        return VisualCTC(**cfg)
    if arch == "vit":
        from src.models.vit import ViTCTC
        return ViTCTC(**cfg, load_pretrained=load_pretrained)
    if arch == "autoavsr":
        from src.models.autoavsr import AutoAVSRCTC
        return AutoAVSRCTC(**cfg, load_pretrained=load_pretrained)
    raise ValueError(f"unknown arch: {arch}")
