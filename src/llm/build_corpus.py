"""Build the text-only phoneme->sentence training corpus for the LLM.

Sources (combine freely):
    --text-file    one sentence per line (e.g. GRID training transcripts)
    --manifest     a dataset manifest/index; uses entries of --split
    --wikitext     WikiText-103 sentences via HuggingFace `datasets` (as in VALLR)

Each sentence becomes
    a reconstruct example with lightly corrupted phonemes (VALLR-style masking:
      random deletion / substitution / insertion, rate --noise), and
    with probability --repair-frac, a repair example whose draft has 1-3 words
      replaced by [?] (and sometimes swapped for a wrong word).

Output JSON lines: {"task", "prompt", "target", "text"}.

Usage:
    python -m src.llm.build_corpus --manifest data/grid_manifest.jsonl --split train --out data/llm_grid.jsonl
"""
import argparse
import json
import random
import re
from pathlib import Path

from src.llm.prompts import MASK, reconstruct_prompt, repair_prompt, target
from src.text.phonemes import PHONEMES, normalize_text, text_to_phonemes


def corrupt(phonemes, rate, rng):
    out = []
    for p in phonemes:
        r = rng.random()
        if r < rate / 3:
            continue                                   # deletion
        if r < 2 * rate / 3:
            out.append(rng.choice(PHONEMES))           # substitution
            continue
        out.append(p)
        if r > 1 - rate / 3:
            out.append(rng.choice(PHONEMES))           # insertion
    return out or phonemes


def make_draft(words, vocab, rng):
    draft = list(words)
    for i in rng.sample(range(len(words)), k=min(len(words), rng.randint(1, 3))):
        draft[i] = MASK
    # Sometimes leave one wrong word unmasked, so repair also learns to fix it.
    if vocab and rng.random() < 0.3:
        i = rng.randrange(len(draft))
        if draft[i] != MASK:
            draft[i] = rng.choice(vocab)
    return draft


def sentences_from_wikitext(limit):
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
    n = 0
    for row in ds:
        line = row["text"].strip()
        if not line or line.startswith("="):
            continue
        for s in re.split(r"(?<=[.!?])\s+", line):
            s = normalize_text(s.replace("@-@", "-").replace(" @,@ ", ","))
            if 3 <= len(s.split()) <= 30:
                yield s
                n += 1
                if limit and n >= limit:
                    return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--text-file")
    ap.add_argument("--manifest")
    ap.add_argument("--split", default="train")
    ap.add_argument("--wikitext", type=int, default=0, help="number of WikiText sentences (0 = none)")
    ap.add_argument("--noise", type=float, default=0.1)
    ap.add_argument("--repair-frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    sents = []
    if args.text_file:
        sents += [normalize_text(l) for l in Path(args.text_file).read_text().splitlines() if l.strip()]
    if args.manifest:
        for l in Path(args.manifest).read_text().splitlines():
            e = json.loads(l)
            if e["split"] == args.split:
                sents.append(normalize_text(e["text"]))
    if args.wikitext:
        sents += list(sentences_from_wikitext(args.wikitext))
    vocab = sorted({w for s in sents for w in s.split()})

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    n_rep = 0
    with out.open("w") as f:
        for s in sents:
            ph = text_to_phonemes(s)
            f.write(json.dumps({"task": "reconstruct", "prompt": reconstruct_prompt(corrupt(ph, args.noise, rng)),
                                "target": target(s), "text": s}) + "\n")
            if rng.random() < args.repair_frac:
                draft = make_draft(s.split(), vocab, rng)
                f.write(json.dumps({"task": "repair", "prompt": repair_prompt(corrupt(ph, args.noise, rng), draft),
                                    "target": target(s), "text": s}) + "\n")
                n_rep += 1
    print(f"{len(sents)} sentences -> {len(sents) + n_rep} examples ({n_rep} repair) in {out}")


if __name__ == "__main__":
    main()
