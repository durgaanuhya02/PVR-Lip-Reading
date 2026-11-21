"""LoRA fine-tuning of a causal LLM for phoneme->sentence reconstruction and repair.

Loss is computed on the target sentence only (prompt tokens are masked).
Uses 4-bit QLoRA automatically when a CUDA GPU and bitsandbytes are available.

Usage (Kaggle):
    python -m src.llm.train_p2t --config configs/llm_p2t.yaml
Local CPU smoke test:
    python -m src.llm.train_p2t --config configs/llm_p2t.yaml --set model_name=HuggingFaceTB/SmolLM2-135M max_steps=20
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from src.train_visual import load_config


def load_model(name, lora_cfg=None, adapter=None, quantize=True):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    kwargs = {}
    cuda = torch.cuda.is_available()
    if cuda and quantize:
        try:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
        except ImportError:
            pass
    model = AutoModelForCausalLM.from_pretrained(
        name, dtype=torch.float16 if cuda else torch.float32,
        device_map="auto" if cuda else None, **kwargs)
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    elif lora_cfg is not None:
        if "quantization_config" in kwargs:
            model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", **lora_cfg))
    return model, tok


def encode(tok, prompt, tgt, max_len):
    p = tok(prompt, add_special_tokens=True)["input_ids"]
    t = tok(tgt, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    # keep the whole target and cut the prompt from the left; cutting (p + t) from the right
    # left long examples with no target tokens at all, which gave NaN losses
    if len(t) >= max_len:
        t = t[-(max_len - 1):]
        p = []
    else:
        p = p[-(max_len - len(t)):]
    ids = p + t
    labels = ([-100] * len(p)) + t
    return ids, labels


def make_collate(tok, max_len):
    def collate(rows):
        enc = [encode(tok, r["prompt"], r["target"], max_len) for r in rows]
        L = max(len(i) for i, _ in enc)
        ids = torch.full((len(enc), L), tok.pad_token_id)
        labels = torch.full((len(enc), L), -100)
        att = torch.zeros((len(enc), L), dtype=torch.long)
        for k, (i, l) in enumerate(enc):          # right padding for training
            ids[k, :len(i)] = torch.tensor(i); labels[k, :len(l)] = torch.tensor(l); att[k, :len(i)] = 1
        return {"input_ids": ids, "labels": labels, "attention_mask": att}
    return collate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    random.seed(cfg.get("seed", 0)); torch.manual_seed(cfg.get("seed", 0))

    out_dir = Path(cfg["out_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg))
    model, tok = load_model(cfg["model_name"], cfg["lora"], quantize=cfg.get("quantize", True))
    model.print_trainable_parameters()

    rows = [json.loads(l) for l in Path(cfg["corpus"]).read_text().splitlines() if l.strip()]
    dl = DataLoader(rows, cfg["batch_size"], shuffle=True, collate_fn=make_collate(tok, cfg.get("max_len", 256)))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=0.0)
    total = cfg.get("max_steps") or cfg["epochs"] * len(dl)
    warm = max(1, int(0.03 * total))
    device = next(model.parameters()).device
    accum = cfg.get("grad_accum", 1)

    model.train(); step, micro, t0, losses = 0, 0, time.time(), []
    while step < total:
        for batch in dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss / accum
            loss.backward(); micro += 1; losses.append(loss.item() * accum)
            if micro % accum:
                continue
            lr = cfg["lr"] * min(1.0, (step + 1) / warm) * 0.5 * (1 + math.cos(math.pi * step / total))
            for g in opt.param_groups:
                g["lr"] = lr
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); opt.zero_grad(set_to_none=True); step += 1
            if step % cfg.get("log_every", 50) == 0 or step == total:
                print(f"step {step}/{total} loss {sum(losses[-50:]) / len(losses[-50:]):.4f} "
                      f"lr {lr:.2e} {(time.time() - t0) / 60:.1f} min", flush=True)
            if step % cfg.get("save_every", 1000) == 0 or step == total:
                model.save_pretrained(out_dir / "adapter"); tok.save_pretrained(out_dir / "adapter")
            if step >= total:
                break
    print("saved adapter to", out_dir / "adapter")


if __name__ == "__main__":
    main()
