"""Run the full lip-reading pipeline and compare systems on one split.

Systems (all share the same visual model and LLM):
    vallr_1best      greedy CTC phonemes -> LLM greedy sentence              (VALLR-style baseline)
    nbest_llm        LLM N-best, pick by LLM score only                      (reranking baseline)
    verify_phoneme   N-best reranked with phoneme-level visual score         (ablation)
    verify_viseme    N-best reranked with viseme-level visual score          (ours: C1 + C2)
    verify_repair    verify_viseme + localised repair of unsupported words   (ours: C1 + C2 + C3)

Writes <out>/outputs.jsonl (per clip, incl. every candidate and score) and <out>/metrics.json.
`--lam` weights the LLM score; tune it on the validation split with `--sweep`.

Usage:
    python -m src.pvr.run --visual runs/visual_grid/best.pt --index grid_rois/index.jsonl --split test \
        --llm meta-llama/Llama-3.2-1B --adapter runs/llm_p2t/adapter --out runs/pvr_test --n 5 --lam 0.5
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import ROIDataset, collate
from src.llm.train_p2t import load_model
from src.metrics import cer, error_rate, wer
from src.models import build_visual
from src.models.visual import greedy_decode
from src.pvr.generate import propose, repair
from src.pvr.verify import rerank, unsupported_words
from src.llm.prompts import MASK
from src.text.phonemes import PHONEME_VOCAB, decode_ids

SYSTEMS = ["vallr_1best", "nbest_llm", "verify_phoneme", "verify_viseme", "verify_repair"]


def load_visual(path, device):
    state = torch.load(path, map_location=device, weights_only=False)
    model = build_visual(state["config"]["model"], load_pretrained=False).to(device).eval()
    model.load_state_dict(state["model"])
    return model


def verify_and_repair(cands, vis_logp, model, tok, phonemes, lam, tau, n, rounds):
    ranked = rerank(cands, vis_logp, "viseme", lam)
    pool = dict(cands)
    for _ in range(rounds):
        best = ranked[0]["text"]
        bad = unsupported_words(vis_logp, best, tau)
        if not bad:
            break
        draft = [MASK if i in bad else w for i, w in enumerate(best.split())]
        new = [(t, s) for t, s in repair(model, tok, phonemes, draft, n) if t not in pool]
        if not new:
            break
        pool.update(new)
        ranked = rerank(list(pool.items()), vis_logp, "viseme", lam)
    return ranked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--visual", required=True)
    ap.add_argument("--index", required=True, nargs="+", help="one or more index.jsonl files")
    ap.add_argument("--split", default="test")
    ap.add_argument("--llm", required=True)
    ap.add_argument("--adapter")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--tau", type=float, default=-2.0, help="word support threshold (log-prob gap)")
    ap.add_argument("--repair-rounds", type=int, default=2)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--sweep", action="store_true", help="also report WER for a grid of lambda values")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    visual = load_visual(args.visual, device)
    llm, tok = load_model(args.llm, adapter=args.adapter)
    llm.eval()

    ds = ROIDataset(args.index, args.split)
    if args.limit:
        ds.entries = ds.entries[:args.limit]
    dl = DataLoader(ds, 1, collate_fn=collate)
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    with (out_dir / "outputs.jsonl").open("w") as f:
        for batch in dl:
            with torch.no_grad():
                o = visual(batch["frames"].to(device), batch["lengths"])
            ph_logp, vi_logp = o["phoneme_logp"][0].cpu(), o["viseme_logp"][0].cpu()
            phonemes = decode_ids(greedy_decode(o["phoneme_logp"], o["lengths"])[0], PHONEME_VOCAB)

            one = propose(llm, tok, phonemes, n=1)
            nbest = propose(llm, tok, phonemes, n=args.n)
            hyp = {"vallr_1best": one[0][0] if one else ""}
            if nbest:
                hyp["nbest_llm"] = max(nbest, key=lambda c: c[1])[0]
                hyp["verify_phoneme"] = rerank(nbest, ph_logp, "phoneme", args.lam)[0]["text"]
                vv = rerank(nbest, vi_logp, "viseme", args.lam)
                hyp["verify_viseme"] = vv[0]["text"]
                vr = verify_and_repair(nbest, vi_logp, llm, tok, phonemes, args.lam, args.tau,
                                       args.n, args.repair_rounds)
                hyp["verify_repair"] = vr[0]["text"]
            else:
                vv = []
                hyp.update({s: hyp["vallr_1best"] for s in SYSTEMS[1:]})

            rec = {"id": batch["ids"][0], "ref": batch["texts"][0], "phonemes": " ".join(phonemes),
                   "hyp": hyp, "candidates": vv,
                   "unsupported": {s: len(unsupported_words(vi_logp, h, args.tau)) if h else 0
                                   for s, h in hyp.items()},
                   "num_words": {s: len(h.split()) for s, h in hyp.items()},
                   "ref_phonemes": batch["phonemes"].tolist(),
                   "hyp_phoneme_ids": greedy_decode(o["phoneme_logp"], o["lengths"])[0]}
            records.append(rec)
            f.write(json.dumps(rec) + "\n"); f.flush()
            print(rec["id"], "| ref:", rec["ref"], "| repair:", hyp["verify_repair"], flush=True)

    refs = [r["ref"] for r in records]
    metrics = {"clips": len(records),
               "PER": error_rate([r["ref_phonemes"] for r in records], [r["hyp_phoneme_ids"] for r in records])}
    for s in SYSTEMS:
        hyps = [r["hyp"][s] for r in records]
        metrics[s] = {"WER": wer(refs, hyps), "CER": cer(refs, hyps),
                      "unsupported_word_rate": sum(r["unsupported"][s] for r in records)
                      / max(sum(r["num_words"][s] for r in records), 1)}
    if args.sweep:
        metrics["lambda_sweep"] = {}
        for lam in np.round(np.arange(0, 1.01, 0.1), 2):
            hyps = []
            for r in records:
                c = r["candidates"]
                hyps.append(max(c, key=lambda d: lam * d["llm"] + (1 - lam) * d["visual"])["text"] if c else "")
            metrics["lambda_sweep"][str(lam)] = wer(refs, hyps)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
