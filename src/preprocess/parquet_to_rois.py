"""Convert pre-cropped LRS3 test parquet files (mattymchen/lrs3-test schema) into our ROI format.

Parquet columns: idx (int), audio (int16 list), video (T x H x W uint8, grayscale mouth crops),
label (transcript). The crops are written as 96x96 MP4s with an index.jsonl compatible with
src.preprocess.extract_rois, so the test set loads exactly like every other dataset.
These clips have no lip landmarks (the dataset loader substitutes zeros).

Usage:
    python -m src.preprocess.parquet_to_rois --parquet /tmp/lrs3_test/*.parquet --out /kaggle/working/lrs3_test_rois
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from src.preprocess.extract_rois import ROI_SIZE, write_video
from src.text.phonemes import normalize_text


def to_frames(video):
    """Nested lists / arrays -> uint8 (T, 96, 96) grayscale."""
    arr = np.stack([np.asarray(np.stack([np.asarray(r, dtype=np.uint8) for r in f]), dtype=np.uint8)
                    for f in video])
    if arr.ndim == 4:                                   # (T, H, W, C) -> gray
        arr = np.stack([cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in arr])
    if arr.shape[1:] != (ROI_SIZE, ROI_SIZE):
        arr = np.stack([cv2.resize(f, (ROI_SIZE, ROI_SIZE), interpolation=cv2.INTER_AREA) for f in arr])
    return arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="lrs3/test")
    ap.add_argument("--split", default="test")
    ap.add_argument("--fps", type=float, default=25.0)
    args = ap.parse_args()

    import pyarrow.parquet as pq
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    n, shapes = 0, set()
    with (out / "index.jsonl").open("w") as idx:
        for path in sorted(args.parquet):
            pf = pq.ParquetFile(path)
            for batch in pf.iter_batches(batch_size=16, columns=["idx", "video", "label"]):
                for row in batch.to_pylist():
                    raw_shape = np.asarray(row["video"][0]).shape
                    shapes.add(raw_shape)
                    frames = to_frames(row["video"])
                    rel = f"{args.name}/{row['idx']:05d}"
                    (out / rel).parent.mkdir(parents=True, exist_ok=True)
                    write_video(out / f"{rel}.mp4", frames, args.fps)
                    idx.write(json.dumps({
                        "id": rel, "video": None, "start": 0, "end": None,
                        "text": normalize_text(row["label"]), "speaker": "lrs3_test",
                        "split": args.split, "roi": f"{rel}.mp4", "landmarks": None,
                        "num_frames": len(frames), "detect_rate": None,
                    }) + "\n")
                    n += 1
    print(f"wrote {n} clips to {out} (original frame sizes: {sorted(shapes)})")


if __name__ == "__main__":
    main()
