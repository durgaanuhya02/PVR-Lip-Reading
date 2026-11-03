"""PyTorch dataset over preprocessed mouth ROIs (any dataset, via index.jsonl)."""
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.preprocess.extract_rois import read_video
from src.text.phonemes import (encode_phonemes, encode_visemes, phonemes_to_visemes,
                               text_to_phonemes)

CROP = 88
MEAN, STD = 0.421, 0.165   # grayscale normalisation used by Ma et al. for LRS/LRW
MIN_FRAMES = 5             # shorter clips are dropped (too short for CTC; some are written unreadable)


class ROIDataset(Dataset):
    """Returns mouth frames, lip landmarks and phoneme/viseme targets for one clip.

    Args:
        index_path: index.jsonl written by src.preprocess.extract_rois, or a list of them
                    (e.g. several preprocessing shards, or GRID + VoxMM)
        split: "train" / "val" / "test" (None = all)
        augment: random crop, horizontal flip and time masking (training only)
    """

    def __init__(self, index_path, split=None, augment=False, max_frames=None, max_clip_frames=None):
        paths = [index_path] if isinstance(index_path, (str, Path)) else index_path
        self.entries = []
        for p in paths:
            root = Path(p).parent
            for l in Path(p).read_text().splitlines():
                e = json.loads(l) if l.strip() else None
                if e and (split is None or e["split"] == split):
                    if max_clip_frames and e.get("num_frames", 0) > max_clip_frames:
                        continue                  # skip very long clips (GPU memory)
                    if e.get("num_frames", MIN_FRAMES) < MIN_FRAMES:
                        continue
                    e["_root"] = str(root)
                    self.entries.append(e)
        self.augment = augment
        self.max_frames = max_frames
        for e in self.entries:
            ph = text_to_phonemes(e["text"])
            e["phoneme_ids"] = encode_phonemes(ph)
            e["viseme_ids"] = encode_visemes(phonemes_to_visemes(ph))

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, i):
        e = self.entries[i]
        root = Path(e["_root"])
        try:
            frames = read_video(root / e["roi"])                      # (T, 96, 96) uint8
        except ValueError:                                            # unreadable video file
            print(f"skipping unreadable clip {e['id']}", flush=True)
            return self[(i + 1) % len(self)]
        if e.get("landmarks"):
            lm = np.load(root / e["landmarks"])["landmarks"].astype(np.float32)
        else:                                                         # pre-cropped data without landmarks
            lm = np.zeros((len(frames), 40, 3), np.float32)
        T = min(len(frames), len(lm))
        if self.max_frames:
            T = min(T, self.max_frames)
        frames, lm = frames[:T], lm[:T]

        if self.augment:
            y, x = random.randint(0, 96 - CROP), random.randint(0, 96 - CROP)
            if random.random() < 0.5:
                frames = frames[:, :, ::-1]
                lm[:, :, 0] = 1.0 - lm[:, :, 0]
            frames = frames[:, y:y + CROP, x:x + CROP]
            lm[:, :, 0] -= x / 96
            lm[:, :, 1] -= y / 96
        else:
            o = (96 - CROP) // 2
            frames = frames[:, o:o + CROP, o:o + CROP]
            lm[:, :, :2] -= o / 96

        frames = (torch.from_numpy(np.ascontiguousarray(frames)).float() / 255.0 - MEAN) / STD
        if self.augment:
            frames = time_mask(frames)
        return {
            "id": e["id"],
            "text": e["text"],
            "frames": frames,                                     # (T, 88, 88)
            "landmarks": torch.from_numpy(lm),                    # (T, 40, 3)
            "phonemes": torch.tensor(e["phoneme_ids"]),
            "visemes": torch.tensor(e["viseme_ids"]),
        }


def time_mask(frames, max_len=10, per_second=1.0, fps=25):
    """Replace random spans of frames with the clip mean (Ma et al., 2022)."""
    T = len(frames)
    for _ in range(int(T / fps * per_second)):
        n = random.randint(0, max_len)
        s = random.randint(0, max(T - n, 0))
        frames[s:s + n] = frames.mean(0)
    return frames


def collate(batch):
    lengths = torch.tensor([len(b["frames"]) for b in batch])
    T = int(lengths.max())
    frames = torch.zeros(len(batch), T, CROP, CROP)
    landmarks = torch.zeros(len(batch), T, 40, 3)
    for i, b in enumerate(batch):
        frames[i, :len(b["frames"])] = b["frames"]
        landmarks[i, :len(b["landmarks"])] = b["landmarks"]
    return {
        "ids": [b["id"] for b in batch],
        "texts": [b["text"] for b in batch],
        "frames": frames.unsqueeze(1),                            # (B, 1, T, 88, 88)
        "landmarks": landmarks,
        "lengths": lengths,
        "phonemes": torch.cat([b["phonemes"] for b in batch]),
        "phoneme_lengths": torch.tensor([len(b["phonemes"]) for b in batch]),
        "visemes": torch.cat([b["visemes"] for b in batch]),
        "viseme_lengths": torch.tensor([len(b["visemes"]) for b in batch]),
    }
