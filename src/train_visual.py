"""Train the Stage-1 visual model (phoneme + viseme CTC).

Resumes automatically from <out>/last.pt, so it survives Kaggle's 12-hour
session limit: just run the same command again.

Usage:
    python -m src.train_visual --config configs/visual_grid.yaml
    python -m src.train_visual --config configs/visual_grid.yaml --set epochs=2 batch_size=2   # overrides
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.data.dataset import ROIDataset, collate
from src.metrics import error_rate
from src.models.visual import VisualCTC, ctc_losses, greedy_decode


def load_config(path, overrides):
    cfg = yaml.safe_load(Path(path).read_text())
    for kv in overrides or []:
        k, v = kv.split("=", 1)
        cfg[k] = yaml.safe_load(v)
    return cfg


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def lr_at(step, total, warmup, peak):
    if step < warmup:
        return peak * (step + 1) / warmup
    return 0.5 * peak * (1 + math.cos(math.pi * (step - warmup) / max(total - warmup, 1)))


@torch.no_grad()
def evaluate(model, loader, device, max_batches=None):
    model.eval()
    ph_ref, ph_hyp, vi_ref, vi_hyp = [], [], [], []
    for n, batch in enumerate(loader):
        if max_batches and n >= max_batches:
            break
        out = model(batch["frames"].to(device), batch["lengths"])
        ph_hyp += greedy_decode(out["phoneme_logp"], out["lengths"])
        vi_hyp += greedy_decode(out["viseme_logp"], out["lengths"])
        ph_ref += torch.split(batch["phonemes"], batch["phoneme_lengths"].tolist())
        vi_ref += torch.split(batch["visemes"], batch["viseme_lengths"].tolist())
    model.train()
    return {"PER": error_rate([r.tolist() for r in ph_ref], ph_hyp),
            "VER": error_rate([r.tolist() for r in vi_ref], vi_hyp)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", help="override config values, e.g. epochs=2")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    set_seed(cfg.get("seed", 0))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg["out_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg))

    train_ds = ROIDataset(cfg["index"], cfg.get("train_split", "train"), augment=True,
                          max_clip_frames=cfg.get("max_train_frames"))
    val_ds = ROIDataset(cfg["index"], cfg.get("val_split", "val"), augment=False)
    nw = cfg.get("num_workers", 2)
    train_dl = DataLoader(train_ds, cfg["batch_size"], shuffle=True, collate_fn=collate,
                          num_workers=nw, drop_last=len(train_ds) > cfg["batch_size"],
                          persistent_workers=nw > 0)
    val_dl = DataLoader(val_ds, cfg["batch_size"], collate_fn=collate, num_workers=nw)
    print(f"device {device} | train {len(train_ds)} | val {len(val_ds)}")

    model = VisualCTC(**cfg["model"]).to(device)
    print(f"parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M")
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 0.01))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    total_steps = cfg["epochs"] * len(train_dl)
    warmup = int(cfg.get("warmup_epochs", 1) * len(train_dl))

    start_epoch, step, best = 0, 0, float("inf")
    ckpt = out_dir / "last.pt"
    if ckpt.exists():
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state["model"]); opt.load_state_dict(state["opt"])
        scaler.load_state_dict(state["scaler"])
        start_epoch, step, best = state["epoch"] + 1, state["step"], state["best"]
        print(f"resumed from epoch {start_epoch}")

    log = (out_dir / "log.jsonl").open("a")
    for epoch in range(start_epoch, cfg["epochs"]):
        t0, running = time.time(), []
        for batch in train_dl:
            for g in opt.param_groups:
                g["lr"] = lr_at(step, total_steps, warmup, cfg["lr"])
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                out = model(batch["frames"].to(device), batch["lengths"])
            loss, parts = ctc_losses(out, batch, cfg.get("viseme_weight", 0.3))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get("grad_clip", 5.0))
            scaler.step(opt); scaler.update()
            running.append(loss.item()); step += 1
            if step % cfg.get("log_every", 50) == 0:
                print(f"epoch {epoch} step {step} loss {np.mean(running[-50:]):.3f} {parts}", flush=True)

        metrics = evaluate(model, val_dl, device, cfg.get("val_batches"))
        rec = {"epoch": epoch, "step": step, "train_loss": float(np.mean(running)),
               **metrics, "minutes": (time.time() - t0) / 60}
        print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + "\n"); log.flush()

        state = {"model": model.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                 "epoch": epoch, "step": step, "best": min(best, metrics["PER"]), "config": cfg}
        torch.save(state, ckpt)
        if metrics["PER"] < best:
            best = metrics["PER"]
            torch.save({"model": model.state_dict(), "config": cfg, "epoch": epoch, **metrics},
                       out_dir / "best.pt")


if __name__ == "__main__":
    main()
