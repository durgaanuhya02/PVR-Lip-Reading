"""Evaluate a trained Stage-1 visual model: PER, VER and confusion matrices.

Writes to --out:
    metrics.json                PER, VER, clip count
    confusion_phoneme.csv/.png  rows = reference phoneme, columns = predicted (+ <del>)
    confusion_viseme.csv/.png   the same for visemes
    top_confusions.txt          most frequent phoneme substitutions
    predictions.jsonl           per clip: reference and predicted phonemes / visemes
The phoneme confusion matrix is what the LLM stage uses to add realistic noise.

Usage:
    python -m src.eval_visual --checkpoint runs/visual_lrs3_vit/best.pt \
        --index lrs3_test_rois/index.jsonl --split test --out runs/visual_lrs3_vit/eval_test
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import ROIDataset, collate
from src.metrics import align, error_rate
from src.models import build_visual
from src.models.visual import greedy_decode
from src.text.phonemes import PHONEME_VOCAB, VISEME_VOCAB

DEL = "<del>"


def load_checkpoint(path, device):
    state = torch.load(path, map_location=device, weights_only=False)
    model = build_visual(state["config"]["model"], load_pretrained=False).to(device).eval()
    model.load_state_dict(state["model"])
    return model


def confusion(refs, hyps, vocab):
    """Counts over aligned pairs: rows = reference symbol, columns = predicted symbol or <del>.
    Insertions have no reference symbol and are counted separately."""
    labels = vocab[1:]                                    # drop the CTC blank
    col = {s: i for i, s in enumerate(labels + [DEL])}
    mat = np.zeros((len(labels), len(labels) + 1), dtype=np.int64)
    insertions = Counter()
    for r, h in zip(refs, hyps):
        for a, b in align(r, h):
            if a is None:
                insertions[b] += 1
            else:
                mat[col[a], col[DEL if b is None else b]] += 1
    return labels, mat, insertions


def save_matrix(labels, mat, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = labels + [DEL]
    lines = [",".join(["ref\\pred"] + cols)]
    lines += [",".join([l] + [str(v) for v in row]) for l, row in zip(labels, mat)]
    path.with_suffix(".csv").write_text("\n".join(lines) + "\n")

    norm = mat / np.maximum(mat.sum(1, keepdims=True), 1)          # row-normalised (recall per symbol)
    fig, ax = plt.subplots(figsize=(0.32 * len(cols) + 3, 0.32 * len(labels) + 2.5))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(cols)), cols, rotation=90, fontsize=7)
    ax.set_yticks(range(len(labels)), labels, fontsize=7)
    ax.set_xlabel("predicted"); ax.set_ylabel("reference"); ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout(); fig.savefig(path.with_suffix(".png"), dpi=200); plt.close(fig)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--index", required=True, nargs="+")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_checkpoint(args.checkpoint, device)
    ds = ROIDataset(args.index, args.split)
    if args.limit:
        ds.entries = ds.entries[:args.limit]
    dl = DataLoader(ds, args.batch_size, collate_fn=collate, num_workers=2)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    ph_ref, ph_hyp, vi_ref, vi_hyp, ids = [], [], [], [], []
    for batch in dl:
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            o = model(batch["frames"].to(device), batch["lengths"])
        ph_hyp += greedy_decode(o["phoneme_logp"], o["lengths"])
        vi_hyp += greedy_decode(o["viseme_logp"], o["lengths"])
        ph_ref += [r.tolist() for r in torch.split(batch["phonemes"], batch["phoneme_lengths"].tolist())]
        vi_ref += [r.tolist() for r in torch.split(batch["visemes"], batch["viseme_lengths"].tolist())]
        ids += batch["ids"]

    sym = lambda seqs, vocab: [[vocab[i] for i in s] for s in seqs]
    ph_ref_s, ph_hyp_s = sym(ph_ref, PHONEME_VOCAB), sym(ph_hyp, PHONEME_VOCAB)
    vi_ref_s, vi_hyp_s = sym(vi_ref, VISEME_VOCAB), sym(vi_hyp, VISEME_VOCAB)

    metrics = {"split": args.split, "clips": len(ids),
               "PER": error_rate(ph_ref, ph_hyp), "VER": error_rate(vi_ref, vi_hyp)}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    labels, mat, ins = confusion(ph_ref_s, ph_hyp_s, PHONEME_VOCAB)
    save_matrix(labels, mat, out / "confusion_phoneme", f"Phoneme confusion ({args.split}, PER {metrics['PER']:.1%})")
    np.save(out / "confusion_phoneme.npy", mat)
    subs = [(mat[i, j], labels[i], labels[j]) for i in range(len(labels)) for j in range(len(labels)) if i != j]
    top = sorted(subs, reverse=True)[:30]
    lines = [f"{r:>3} -> {h:<3} {n}" for n, r, h in top if n]
    lines += ["", "insertions: " + ", ".join(f"{s} {n}" for s, n in ins.most_common(10))]
    (out / "top_confusions.txt").write_text("\n".join(lines) + "\n")

    v_labels, v_mat, _ = confusion(vi_ref_s, vi_hyp_s, VISEME_VOCAB)
    save_matrix(v_labels, v_mat, out / "confusion_viseme", f"Viseme confusion ({args.split}, VER {metrics['VER']:.1%})")
    np.save(out / "confusion_viseme.npy", v_mat)

    with (out / "predictions.jsonl").open("w") as f:
        for i, a, b, c, d in zip(ids, ph_ref_s, ph_hyp_s, vi_ref_s, vi_hyp_s):
            f.write(json.dumps({"id": i, "ref_phonemes": " ".join(a), "hyp_phonemes": " ".join(b),
                                "ref_visemes": " ".join(c), "hyp_visemes": " ".join(d)}) + "\n")
    print(json.dumps(metrics, indent=2))
    print("\n".join(lines[:15]))


if __name__ == "__main__":
    main()
