"""Extract mouth regions of interest (ROIs) and lip landmarks for every clip in a manifest.

For each clip it writes
    <out>/<id>.mp4   grayscale 96x96 mouth crops (MPEG-4, ~10x smaller than raw arrays)
    <out>/<id>.npz   landmarks float16 (T, 40, 3) lip landmarks, x/y as fraction of the crop
                     detected  bool    (T,)        False where no face was found (interpolated)
and appends a line to <out>/index.jsonl:
    the manifest entry + {"roi": "<id>.mp4", "landmarks": "<id>.npz", "num_frames": T, "detect_rate": r}

Face landmarks come from the MediaPipe Face Landmarker (478 points). The crop
centre is the mean of the 40 lip points, smoothed over time; the crop size is
fixed per clip from the median mouth width, so the mouth does not "zoom".

Safe to re-run: clips already listed in index.jsonl are skipped.

Usage:
    python -m src.preprocess.extract_rois --manifest data/grid_manifest.jsonl --out data/grid_rois --workers 3
"""
import argparse
import json
import multiprocessing as mp
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/1/face_landmarker.task")
MODEL_PATH = Path(__file__).resolve().parents[2] / "models_cache" / "face_landmarker.task"

# MediaPipe face-mesh indices of the outer (20) and inner (20) lip contours.
LIP_IDX = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185,
           78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]
LEFT_CORNER, RIGHT_CORNER = 61, 291

ROI_SIZE = 96
CROP_SCALE = 2.0      # crop side = CROP_SCALE * median mouth width
SMOOTH_WINDOW = 5     # frames, moving average of the crop centre


def ensure_model():
    if not MODEL_PATH.exists():
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {MODEL_URL}")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return MODEL_PATH


def make_landmarker():
    from mediapipe.tasks.python import BaseOptions, vision
    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(ensure_model())),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
    )
    return vision.FaceLandmarker.create_from_options(options)


def read_frames(path, start=0, end=None):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames, i = [], 0
    while True:
        ok, frame = cap.read()
        if not ok or (end is not None and i >= end):
            break
        if i >= start:
            frames.append(frame)
        i += 1
    cap.release()
    return frames, fps


def detect_landmarks(landmarker, frames, fps, every=1):
    """(T, 478, 3) landmarks in pixel units (x, y) and relative z, plus detection mask.
    With every > 1, only every N-th frame (and the last) is detected; the rest are
    interpolated by interpolate_missing()."""
    import mediapipe as mp_
    h, w = frames[0].shape[:2]
    out = np.zeros((len(frames), 478, 3), np.float32)
    found = np.zeros(len(frames), bool)
    for t, frame in enumerate(frames):
        if t % every and t != len(frames) - 1:
            continue
        image = mp_.Image(image_format=mp_.ImageFormat.SRGB,
                          data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        res = landmarker.detect_for_video(image, int(t * 1000 / fps))
        if res.face_landmarks:
            pts = res.face_landmarks[0]
            out[t] = [(p.x * w, p.y * h, p.z * w) for p in pts]
            found[t] = True
    return out, found


def interpolate_missing(points, found):
    """Linearly fill frames without a detection (edges copy the nearest frame)."""
    if found.all() or not found.any():
        return points
    t = np.arange(len(points))
    flat = points.reshape(len(points), -1)
    filled = np.stack([np.interp(t, t[found], flat[found, k]) for k in range(flat.shape[1])], 1)
    return filled.reshape(points.shape)


def crop_mouths(frames, lm, crop_scale=CROP_SCALE):
    lips = lm[:, LIP_IDX]                                        # (T, 40, 3)
    centre = lips[:, :, :2].mean(1)                              # (T, 2)
    kernel = np.ones(SMOOTH_WINDOW) / SMOOTH_WINDOW
    pad = SMOOTH_WINDOW // 2
    centre = np.stack([np.convolve(np.pad(centre[:, k], pad, mode="edge"), kernel, "valid")
                       for k in range(2)], 1)
    width = np.linalg.norm(lm[:, LEFT_CORNER, :2] - lm[:, RIGHT_CORNER, :2], axis=1)
    side = max(int(round(crop_scale * np.median(width))), 16)

    rois = np.zeros((len(frames), ROI_SIZE, ROI_SIZE), np.uint8)
    lips_crop = np.zeros_like(lips)
    for t, frame in enumerate(frames):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        x0 = int(round(centre[t, 0] - side / 2))
        y0 = int(round(centre[t, 1] - side / 2))
        padded = cv2.copyMakeBorder(gray, side, side, side, side, cv2.BORDER_REPLICATE)
        patch = padded[y0 + side:y0 + 2 * side, x0 + side:x0 + 2 * side]
        rois[t] = cv2.resize(patch, (ROI_SIZE, ROI_SIZE), interpolation=cv2.INTER_AREA)
        lips_crop[t, :, 0] = (lips[t, :, 0] - x0) / side
        lips_crop[t, :, 1] = (lips[t, :, 1] - y0) / side
        lips_crop[t, :, 2] = lips[t, :, 2] / side
    return rois, lips_crop.astype(np.float16)


_LANDMARKER = None


def process(entry, out_dir, every=1, crop_scale=CROP_SCALE):
    global _LANDMARKER
    frames, fps = read_frames(entry["video"], entry.get("start") or 0, entry.get("end"))
    if not frames:
        return {"id": entry["id"], "error": "no frames"}
    # A fresh landmarker per clip keeps MediaPipe's VIDEO-mode timestamps increasing. It is
    # closed even when the clip fails; otherwise every later clip in this worker fails with
    # "Input timestamp must be monotonically increasing".
    _LANDMARKER = make_landmarker()
    try:
        lm, found = detect_landmarks(_LANDMARKER, frames, fps, every)
    finally:
        _LANDMARKER.close()
        _LANDMARKER = None
    if not found.any():
        return {"id": entry["id"], "error": "no face detected"}
    lm = interpolate_missing(lm, found)
    rois, lips = crop_mouths(frames, lm, crop_scale)

    base = Path(out_dir) / entry["id"]
    base.parent.mkdir(parents=True, exist_ok=True)
    write_video(base.with_suffix(".mp4"), rois, fps)
    np.savez_compressed(base.with_suffix(".npz"), landmarks=lips, detected=found)
    return {**entry, "roi": f"{entry['id']}.mp4", "landmarks": f"{entry['id']}.npz",
            "num_frames": len(rois), "detect_rate": float(found.mean())}


def write_video(path, frames, fps):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                             (ROI_SIZE, ROI_SIZE), isColor=False)
    for f in frames:
        writer.write(f)
    writer.release()


def read_video(path):
    """Read a mouth-ROI video back as uint8 (T, 96, 96)."""
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f if f.ndim == 2 else cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
    cap.release()
    return np.stack(frames)


def _worker(args):
    entry, out_dir, every, crop_scale = args
    try:
        return process(entry, out_dir, every, crop_scale)
    except Exception as e:  # keep going; failures are logged
        return {"id": entry["id"], "error": repr(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None, help="process only the first N clips (testing)")
    ap.add_argument("--shard", default="0/1", help="i/n: process only the i-th of n equal parts")
    ap.add_argument("--detect-every", type=int, default=1,
                    help="run face detection on every N-th frame and interpolate the rest")
    ap.add_argument("--crop-scale", type=float, default=CROP_SCALE,
                    help="crop side = crop_scale x median mouth width (match other datasets' framing)")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path, error_path = out_dir / "index.jsonl", out_dir / "errors.jsonl"
    done = set()
    if index_path.exists():
        done = {json.loads(l)["id"] for l in index_path.read_text().splitlines() if l.strip()}

    entries = [json.loads(l) for l in Path(args.manifest).read_text().splitlines() if l.strip()]
    i, n = map(int, args.shard.split("/"))
    entries = entries[i::n]
    if args.limit:
        entries = entries[:args.limit]
    todo = [e for e in entries if e["id"] not in done]
    print(f"{len(entries)} clips, {len(done)} already done, {len(todo)} to process")
    ensure_model()

    t0 = time.time()
    with index_path.open("a") as idx, error_path.open("a") as err, mp.Pool(args.workers) as pool:
        jobs = [(e, out_dir, args.detect_every, args.crop_scale) for e in todo]
        for n, res in enumerate(pool.imap_unordered(_worker, jobs, chunksize=4), 1):
            (err if "error" in res else idx).write(json.dumps(res) + "\n")
            if n % 50 == 0 or n == len(todo):
                idx.flush(); err.flush()
                rate = n / (time.time() - t0)
                print(f"{n}/{len(todo)}  {rate:.2f} clips/s  eta {(len(todo) - n) / rate / 3600:.1f} h", flush=True)


if __name__ == "__main__":
    main()
