"""Build a manifest for LRS-style datasets (LRS2, LRS3, LRS-VoxMM).

Expected layout (LRS2/3 convention):
    <root>/<video_or_speaker_id>/<clip>.mp4
    <root>/<video_or_speaker_id>/<clip>.txt     first line: "Text:  THE SPOKEN SENTENCE"

Every clip becomes one manifest entry:
    {"id": "lrs3/trainval/<folder>/<clip>", "video": ..., "start": 0, "end": null,
     "text": "the spoken sentence", "speaker": "<folder>", "split": ...}

Split options:
    --split test                    every clip goes to one split
    --split train --val-fraction 0.02
                                    hold out whole folders (videos / speakers) as validation,
                                    so validation speakers are unseen in training

Usage:
    python -m src.preprocess.build_lrs_manifest --root /tmp/lrs3/trainval --name lrs3/trainval \
        --split train --val-fraction 0.02 --out /kaggle/working/lrs3_trainval.jsonl
    python -m src.preprocess.build_lrs_manifest --root /tmp/lrs3/test --name lrs3/test --split test \
        --out /kaggle/working/lrs3_test.jsonl
"""
import argparse
import json
import random
from pathlib import Path

from src.text.phonemes import normalize_text


def read_lrs_text(path):
    """Transcript from an LRS .txt file ("Text:  ..." line); falls back to the first line."""
    lines = Path(path).read_text(encoding="utf-8", errors="ignore").splitlines()
    for line in lines:
        if line.startswith("Text:"):
            return line[len("Text:"):].strip()
    return lines[0].strip() if lines else ""


def build(root, name, split, val_fraction=0.0, seed=0):
    root = Path(root).resolve()
    entries = []
    for video in sorted(root.rglob("*.mp4")):
        txt = video.with_suffix(".txt")
        if not txt.exists():
            continue
        text = normalize_text(read_lrs_text(txt))
        if not text:
            continue
        rel = video.relative_to(root).with_suffix("")
        folder = rel.parts[0] if len(rel.parts) > 1 else "unknown"
        entries.append({
            "id": f"{name}/{rel.as_posix()}",
            "video": str(video),
            "start": 0,
            "end": None,
            "text": text,
            "speaker": folder,
            "split": split,
        })

    if val_fraction > 0:
        folders = sorted({e["speaker"] for e in entries})
        rng = random.Random(seed)
        val = set(rng.sample(folders, max(1, int(len(folders) * val_fraction))))
        for e in entries:
            if e["speaker"] in val:
                e["split"] = "val"
    return entries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="folder containing <id>/<clip>.mp4 + .txt")
    ap.add_argument("--name", required=True, help="prefix for clip ids, e.g. lrs3/trainval")
    ap.add_argument("--split", required=True, help="train / val / test (val is carved out of train)")
    ap.add_argument("--val-fraction", type=float, default=0.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    entries = build(args.root, args.name, args.split, args.val_fraction)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")
    counts = {s: sum(e["split"] == s for e in entries) for s in ("train", "val", "test")}
    print(f"wrote {len(entries)} clips to {out}: {counts}")


if __name__ == "__main__":
    main()
