# Technical Description — See, Propose, Verify (PVR) for Lip Reading

This document describes exactly what is implemented in `src/` (every number below matches the code and configs). It is the basis for the paper's Method and Experimental Setup sections.

---

## 1. Problem and base paper

**Task.** Visual speech recognition (VSR): given a silent video of a speaker's mouth, output the spoken sentence.

**Base paper.** VALLR (Thomas, Fish & Bowden, ICCV 2025) decomposes VSR into two stages:
1. video → phoneme sequence (vision transformer + CTC head),
2. phoneme sequence → sentence (LoRA-fine-tuned Llama 3.2).

**Gap we address.** VALLR is *open-loop*: the sentence produced by the LLM is never checked against the video. The paper reports homophone/homophene errors (e.g. /s/–/z/, *your*–*you're*) and notes hallucination risk, but does not measure or correct either.

**Our contribution.** Close the loop with three inference-time mechanisms:

| ID | Mechanism | Purpose |
|---|---|---|
| C1 | Visual verification of LLM candidates | reject sentences the lips contradict |
| C2 | Viseme-level (homophene-tolerant) verification score | let vision judge what is visible, language what is not |
| C3 | Localised repair of unsupported words | fix only the words with weak visual evidence |
| — | Unsupported-word rate metric | quantify hallucination |
| C4 *(planned)* | Verified speaker self-adaptation | label-free adaptation to new speakers |

---

## 2. System overview

```
             STAGE 0                 STAGE 1                         STAGE 2                 STAGE 3 (ours)
 video ──► mouth ROI + ──► 3D-CNN + ResNet-18 + Conformer ──► greedy CTC ──► LLM (LoRA) ──► N candidates
           landmarks                 │                         phonemes                         │
                                     ├─ phoneme CTC head (layer 12)                             ▼
                                     └─ viseme  CTC head (layer 6) ───────────────► VERIFY (rerank) ──► REPAIR ──► sentence
```

---

## 3. Stage 0 — Data and preprocessing

### 3.1 Manifest (dataset-agnostic)
Every dataset is converted to JSON lines:
```json
{"id": "s1/bbaf2n", "video": "...", "start": 0, "end": null,
 "text": "bin blue at f two now", "speaker": "s1", "split": "train"}
```
Only `build_grid_manifest.py` knows about GRID; VoxMM will add its own builder and reuse everything else.

### 3.2 GRID protocol
- 33 speakers (s21 has no video), 1,000 sentences each, 75 frames at 25 fps (3 s).
- Grammar: `command colour preposition letter digit adverb` (51-word vocabulary).
- **Unseen-speaker split** (LipNet protocol): speakers **1, 2, 20, 22 → test**; 5 % of the remaining clips (random, seed 0) → validation; rest → train.
- Transcripts from `.align` files with `sil`/`sp` removed.

### 3.3 Mouth ROI extraction (`src/preprocess/extract_rois.py`)
| Step | Detail |
|---|---|
| Face landmarks | MediaPipe Face Landmarker (478 points), VIDEO mode, 1 face. On Kaggle every 2nd frame is detected (`--detect-every 2`) and the rest linearly interpolated (mean landmark deviation vs. full detection ≈ 0.1–0.5 % of crop width) |
| Lip points | 40 face-mesh indices (20 outer + 20 inner contour) |
| Crop centre | mean of the 40 lip points, smoothed by a 5-frame moving average |
| Crop size | fixed per clip: 2.0 × median mouth width (corner points 61–291) — avoids zoom jitter |
| Output frames | grayscale, resized to **96 × 96**, stored as MPEG-4 video (~35 KB/clip; ~1.2 GB for GRID) |
| Output landmarks | `(T, 40, 3)` float16, x/y as fraction of the crop, plus a per-frame detection mask |
| Missed detections | linearly interpolated |
| Scaling | multiprocessing, resumable (`index.jsonl`), sharding (`--shard i/n`) for Kaggle's 12 h limit |

### 3.4 Linguistic units (`src/text/phonemes.py`)
- **Phonemes:** 39 ARPAbet phonemes from the CMU Pronouncing Dictionary, stress removed (as in VALLR). Vocabulary = 39 + CTC blank = **40**.
- **Visemes:** 13 classes (+ blank = **14**), grouping phonemes that look alike on the lips (adapted from Cappelletta & Harte 2011; Bear et al. 2014):

| Viseme | Phonemes | Viseme | Phonemes |
|---|---|---|---|
| BILABIAL | P B M | RHOTIC | R ER |
| LABIODENTAL | F V | ROUND_CLOSE | W UW UH |
| DENTAL | TH DH | SPREAD_CLOSE | Y IY IH |
| ALVEOLAR | T D N L | SPREAD_MID | EH EY AE |
| SIBILANT | S Z | OPEN | AA AH AY AW |
| POSTALVEOLAR | SH ZH CH JH | ROUND_OPEN | AO OW OY |
| BACK | K G NG HH | | |

- Out-of-dictionary words: digits → number words; single letters → spelled pronunciation; otherwise letter-by-letter spelling. All GRID words are in CMUdict.

---

## 4. Stage 1 — Visual model (`src/models/visual.py`)

### 4.1 Architecture (29.6 M parameters)
| Block | Configuration | Output shape (T = 75) |
|---|---|---|
| Input | grayscale mouth crop, normalised (mean 0.421, std 0.165) | (B, 1, 75, 88, 88) |
| 3D stem | Conv3d 1→64, kernel (5,7,7), stride (1,2,2) + BN + ReLU + MaxPool3d (1,3,3)/(1,2,2) | (B, 64, 75, 22, 22) |
| 2D trunk | ResNet-18 (4 stages, 64/128/256/512 channels), shared over frames | (B·T, 512, 3, 3) |
| Pooling + projection | global average pool → Linear 512→256, + sinusoidal positions | (B, 75, 256) |
| Temporal encoder | 12 Conformer blocks: d = 256, 4 heads, FFN ×4 (macaron, ½-step), depthwise conv kernel 31, dropout 0.1 | (B, 75, 256) |
| **Viseme head** | Linear 256→14 on the output of **block 6** | (B, 75, 14) |
| **Phoneme head** | Linear 256→40 on the output of **block 12** | (B, 75, 40) |

Parameter split: front-end 11.2 M, Conformer 18.3 M, heads 0.01 M. Frame rate is kept at 25 fps (no temporal downsampling) so CTC always has more steps than target tokens.

**Design rationale.** The viseme head sits mid-network (hierarchical supervision: visible mouth shape first, phoneme identity later). Its posteriors are the visual evidence used by Verify.

### 4.2 Training objective
$$\mathcal{L} = \mathrm{CTC}(\text{phonemes}) + 0.3\cdot\mathrm{CTC}(\text{visemes})$$

### 4.3 Training setup (`configs/visual_grid.yaml`)
| Setting | Value |
|---|---|
| Optimiser | AdamW, weight decay 0.01 |
| Learning rate | peak 6e-4, 2 warm-up epochs (linear), cosine decay |
| Batch / epochs | 16 / 40 |
| Precision | fp16 autocast + gradient scaling (GPU) |
| Gradient clipping | 5.0 |
| Augmentation | random 88×88 crop (centre crop at test), horizontal flip p = 0.5 (landmarks mirrored), time masking (≤10 frames per second, replaced by clip mean) |
| Model selection | lowest validation PER (greedy CTC decoding) |
| Fault tolerance | `last.pt` every epoch, automatic resume |

---

## 5. Stage 2 — Phoneme-to-sentence LLM (`src/llm/`)

### 5.1 Model
Llama 3.2-1B, LoRA (r = 16, α = 32, dropout 0.05) on all attention and MLP projections (`q,k,v,o,gate,up,down`), loaded in 4-bit NF4 (QLoRA) on a T4 GPU.

### 5.2 Two tasks in one adapter
```
reconstruct:  ### Phonemes: B IH N B L UW AE T EH F T UW N AW
              ### Sentence: bin blue at f two now

repair:       ### Phonemes: B IH N B L UW AE T EH F T UW N AW
              ### Draft: bin blue at [?] two now
              ### Sentence: bin blue at f two now
```
The repair task (new) teaches the LLM to fill `[?]` slots given phonemes and context; it is what Stage 3's Repair step calls.

### 5.3 Training corpus (`build_corpus.py`)
| Source | Use |
|---|---|
| GRID training transcripts | in-domain sentences |
| WikiText-103 (50,000 sentences, 3–30 words) | general language, as in VALLR |

- **Phoneme corruption** (simulates recognition errors, VALLR-style): rate 0.1, split evenly into deletion / substitution / insertion.
- **Repair examples:** for 30 % of sentences; 1–3 words replaced by `[?]`; with probability 0.3 one other word is replaced by a wrong vocabulary word left unmasked.
- Loss on target tokens only (prompt masked). LR 2e-4, cosine with 3 % warm-up, batch 16 × 2 accumulation, max length 192 tokens, 3,000 steps.

---

## 6. Stage 3 — Propose · Verify · Repair (`src/pvr/`)

### 6.1 Propose
1. Greedy CTC decoding of the phoneme head → phoneme string $\hat{P}$.
2. LLM beam search (N = 5) with the reconstruct prompt → up to 5 distinct candidate sentences $S_1..S_N$.
3. LLM score: $\ell_{LM}(S) = \sum \log P_{LLM}(\text{tokens of } S \,\|\, \hat{P})$ (including end-of-sequence).

### 6.2 Verify (C1 + C2)
Each candidate is converted to its viseme sequence $V(S)$ (via CMUdict). Its visual score is the CTC log-likelihood under the viseme head:
$$\ell_{vis}(S) = \log P_{CTC}\big(V(S)\,\big|\,\text{video}\big)$$
Candidates are ranked by
$$\text{score}(S) = \lambda\,\ell_{LM}(S) + (1-\lambda)\,\ell_{vis}(S)$$
λ ∈ {0, 0.1, …, 1} is chosen on 300 validation clips.

**Why visemes (C2).** If two candidates differ only within a viseme class (p/b/m, s/z, f/v), $V(S)$ is identical, so $\ell_{vis}$ is identical and the LLM score decides; candidates whose mouth shapes contradict the video receive a much lower $\ell_{vis}$. (Verified by unit tests in `tests/test_verify.py`.)

### 6.3 Repair (C3)
1. **Forced alignment:** CTC Viterbi alignment of $V(S^*)$ (best candidate) to the viseme posteriors gives the frames of every viseme token.
2. **Word support:** for word $w$ with tokens $k$ and aligned frames $t$:
   $$\text{support}(w) = \operatorname{mean}_{k,t}\Big[\log p_t(k) - \max_v \log p_t(v)\Big] \le 0$$
   (0 = the video's top choice at every frame; very negative = contradicted).
3. Words with support < τ (τ = −2.0) are replaced by `[?]` and the LLM is prompted with the repair task (beam 5).
4. New candidates are scored with the *reconstruct* prompt (comparable LLM scores), added to the pool and re-verified. Up to 2 rounds; stops early if every word is supported.

### 6.4 Systems compared (same visual model and LLM for all)
| System | Description |
|---|---|
| `vallr_1best` | greedy phonemes → LLM greedy sentence (VALLR-style baseline) |
| `nbest_llm` | 5 candidates, highest LLM score (N-best baseline) |
| `verify_phoneme` | verification with the phoneme head (ablation of C2) |
| `verify_viseme` | **C1 + C2** |
| `verify_repair` | **C1 + C2 + C3** |

---

## 7. Evaluation

| Metric | Definition |
|---|---|
| WER / CER | word / character edit distance ÷ reference length (corpus level) |
| PER / VER | phoneme / viseme error rate of Stage 1 (greedy CTC) |
| **Unsupported-word rate** (new) | output words with support < τ ÷ total output words — a hallucination proxy |
| λ sweep | WER for λ = 0…1 on validation |
| Planned | homophene subset accuracy (GRID letter slot: b/p/m, f/v, …), per-speaker WER, latency, mean ± std over 3 seeds |

Outputs: `metrics.json` (all systems) and `outputs.jsonl` (per clip: reference, phonemes, every system's sentence, all candidates with LLM / visual / combined scores).

---

## 8. Compute and workflow

| Stage | Where | Notebook |
|---|---|---|
| Preprocessing (2 shards) | Kaggle CPU | `01_preprocess_grid` |
| Visual model training | Kaggle GPU T4 | `02_train_visual` |
| LLM fine-tuning + PVR evaluation | Kaggle GPU T4 | `03_llm_and_pvr` |
| Development, unit tests | laptop (CPU) | — |

Code is developed locally, pushed to GitHub and cloned by the notebooks (`KAGGLE_GUIDE.md`).

---

## 9. Code map

| File | Role |
|---|---|
| `src/preprocess/build_grid_manifest.py` | GRID → manifest, unseen-speaker split |
| `src/preprocess/extract_rois.py` | mouth ROI + lip landmarks |
| `src/text/phonemes.py` | text → phonemes → visemes |
| `src/data/dataset.py` | dataset, augmentation, batching |
| `src/models/visual.py` | visual model, CTC losses, greedy decoding |
| `src/train_visual.py` | Stage 1 training |
| `src/llm/{prompts,build_corpus,train_p2t}.py` | Stage 2 corpus and LoRA training |
| `src/pvr/generate.py` | Propose, LLM scoring, repair generation |
| `src/pvr/verify.py` | CTC scoring, reranking, forced alignment, word support |
| `src/pvr/run.py` | full pipeline + evaluation of all systems |
| `src/metrics.py` | WER, CER, PER, VER |
| `tests/test_verify.py` | unit tests for Verify |
| `configs/*.yaml` | hyper-parameters |
| `notebooks/` | Kaggle notebooks |

---

## 10. Status, limitations and next steps

**Done:** full pipeline implemented and tested end-to-end on 20 GRID clips (CPU); verification unit tests pass; GRID preprocessing running on Kaggle.

**Limitations to state honestly in the paper:**
- The visual encoder (CNN + Conformer on mouth crops) differs from VALLR's ViT-Base on full faces; the VALLR baseline is VALLR's *method* on our encoder, not their published numbers.
- GRID's fixed grammar makes the LLM stage easy; the benefit of verification is expected mainly on the letter slot (homophenes) and on unconstrained speech (VoxMM).
- Lip landmarks are extracted and stored but not yet used by the model (reserved for a landmark-stream extension).
- Hyper-parameters (viseme weight 0.3, τ = −2.0, N = 5) are initial choices; τ and N should be ablated.

**Next:**
1. Train Stage 1 and Stage 2 on full GRID (Kaggle) → first real results.
2. Homophene-subset evaluation and ablations (λ, τ, N, viseme-head layer).
3. VoxMM manifest builder and experiments.
4. C4 verified self-adaptation (optional).
5. Paper writing.
