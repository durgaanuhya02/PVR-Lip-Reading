# Final Approach — Propose · Verify · Repair (PVR)

**Working title:** *See, Propose, Verify: Visually Grounded LLM Decoding for Phoneme-Centric Lip Reading*

**Base paper we improve:** M. Thomas, E. Fish, R. Bowden, *"VALLR: Visual ASR Language Model for Lip Reading,"* ICCV 2025 ([arXiv 2503.21408](https://arxiv.org/abs/2503.21408)).

---

## 1. Problem with the base paper

VALLR is **open-loop**: the video model predicts one phoneme string, the LLM rewrites it into a sentence, and the sentence is **never checked against the video**. The paper itself reports the two consequences:

1. **Homophone / homophene errors** — e.g. *your* vs *you're*, /s/ vs /z/ (Sec. 5, Conclusion).
2. **Hallucination risk** — a fluent sentence that the lips do not support (Sec. 1), which is never measured.

Recent work sharpens the problem: *From Hype to Insight* (arXiv 2509.14880) finds that LLM decoders in lip reading improve mainly **lexical guessing**, not visual understanding, and that bigger decoders bring little. PVR answers this directly: every word the LLM writes must be supported by the video.

## 2. Related work and novelty check (Nov 2026)

| Work | What it does | LRS3 WER | Relation to us |
|---|---|---|---|
| VALLR (arXiv 2503.21408) | ViT → phonemes (CTC) → Llama 3.2-3B, 30 h only | 18.7 | Base paper; reproduced as our baseline |
| PV-ASR (arXiv 2507.18863) | Visual + lip-landmark features → phonemes → NLLB | 21.0 | Same two-stage family; no check against the video |
| FLARE (arXiv 2610.19559) | Lip + face streams, Whisper-distilled masked pre-training, Transformer or LLM decoder | 13.2 | Shows the value of a strong pre-trained encoder (our Track B) |
| From Hype to Insight (arXiv 2509.14880) | Study of LLM decoders on SSL encoders | 24.7 | Motivation: LLM gains are lexical; encoders matter most |
| VALLR-Pin (arXiv 2512.20032), DualHyp (arXiv 2510.13281) | N-best hypotheses as LLM input | — | Used inside our pipeline + as baseline |
| PECT (arXiv 2609.20839) | LLM trained on realistic phoneme errors | — | Used inside our pipeline (credited) |
| "Tackling the Phoneme-to-Text Conversion Bottleneck…" | Visemes vs phonemes as LLM input — phonemes better | — | Why the LLM reads phonemes, not visemes |
| Kim et al. 2022 (arXiv 2208.04498); ACM 2024 TTA | Pseudo-label speaker adaptation | — | Comparison point for C4 |
| DLLM-VSR (arXiv 2605.28456) | Reranks by *length* plausibility | — | Closest to C1, but not a visual score |

Open gaps (not found in the search):

| Idea | Status |
|---|---|
| **Re-scoring LLM sentence candidates by their fit to the video** | ✅ not found |
| **Localising unsupported words and repairing only those** | ✅ not found |
| **Viseme-tolerant verification score** | ✅ not found |
| **Unsupported-word rate as a hallucination metric for VSR** | ✅ not found |
| Viseme-aware phonetic retrieval for repair (C5) | ⚠ not yet checked |

→ The "done" ideas are **built into** the pipeline or used as **baselines**; the open gaps are our contributions. Several entries above were read only at abstract level — read the full papers before submission and re-check arXiv monthly.

## 3. Method

```
Video ──► (1) SEE: visual encoder (Track A or B) ──► phoneme CTC posteriors + viseme CTC posteriors
                                         │
                                         ▼  N-best phoneme strings (CTC beam search)
          (2) PROPOSE: phoneme→text LLM (trained on real Stage-1 errors) ──► N candidate sentences
                                         │
                                         ▼
          (3) VERIFY: each sentence → phonemes (CMU dict / G2P) → visemes
                      visual score = log P_CTC(viseme seq | video)      ← viseme-tolerant
                      final score  = λ · LLM score + (1 − λ) · visual score
                                         │
                                         ▼
          (4) REPAIR: forced-align best sentence to the video; words whose
                      segment score < τ are marked [?] and the LLM rewrites
                      only those words (≤ 2 iterations)
                      [C5: viseme-aware retrieval suggests candidate words]
                                         │
                                         ▼
                                   final sentence
```

### 3.1 Stage 1 — visual encoder, two tracks

PVR only needs CTC posteriors, so it is shown on two encoders:

| Track | Encoder | Training | Purpose |
|---|---|---|---|
| **A — low-data** (VALLR setting) | 3D-CNN + ResNet-18 + Conformer (current code, `configs/visual_lrs3.yaml`) | From scratch on LRS3 trainval 30 h | Fair comparison with VALLR's data setting |
| **B — strong encoder** | Public pre-trained VSR encoder (Auto-AVSR; AV-HuBERT or USR as alternatives), frozen or LoRA-tuned | New phoneme + viseme CTC heads trained on LRS3 trainval 30 h | Follows the "stronger encoder" finding; starts near state of the art; cheap on a T4 |

- Both tracks have a **phoneme CTC head** (39 + blank) and a **viseme CTC head** (~13 + blank); loss = phoneme CTC + 0.3 · viseme CTC.
- Track B uses thousands of hours of extra (pre-training) video, so its results are **reported separately** with the data amount labelled.
- Optional encoder additions if time allows: **lip-landmark stream** (40 MediaPipe points, already extracted; as in PV-ASR) and a **face stream** (as in FLARE).

### 3.2 Stage 2 — phoneme→text LLM

- Small recent LLM with **QLoRA 4-bit**: Llama 3.2-1B (main) or Qwen3-1.7B; 3B once at the end if time allows. A larger decoder is not worth the cost (*From Hype to Insight*).
- Training text: LRS3 train transcripts + WikiText-103, converted to phonemes (CMU dict, G2P fallback).
- **Noise like Stage 1 makes it** (PECT-style): instead of only random masking, corrupt the phonemes with the Stage-1 confusion matrix and with real CTC outputs on held-out LRS3 train clips.
- **N-best input** (VALLR-Pin-style): the LLM sees the top CTC phoneme strings, not just one.

### 3.3 C1 — Visual verification (core contribution)
- Generate N candidates (N = 5–10) from the LLM (beam / sampling).
- Convert each candidate to phonemes (CMU dict, G2P fallback for unknown words) and then visemes.
- Score with the **CTC log-likelihood of the candidate's sequence given the video** (`torch.nn.functional.ctc_loss`, one call per candidate) and combine with the LLM log-probability. λ tuned on validation.
- **Training-free** on the LLM side → cheap; works on top of any CTC visual model.

### 3.4 C2 — Viseme-tolerant scoring (principled split of responsibility)
- Verify with the **viseme** head, not the phoneme head: candidates differing only inside a viseme class (p/b/m, s/z, f/v) get the **same visual score**, so the decision is left to the LLM; candidates the lips actually contradict are penalised.
- Rule: *vision judges what is visible, language decides what is not.*
- Ablation: phoneme-level vs viseme-level vs mixed verification.

### 3.5 C3 — Localised repair
- CTC forced alignment gives a per-word visual support score.
- Words below threshold τ are replaced with a mask token and the LLM is asked to fill only those spans; the result is re-verified.
- Directly targets hallucinated / unsupported words instead of regenerating the whole sentence.

### 3.6 C4 (extension, if time) — Verified self-adaptation
- Use sentences with high verified score as pseudo-labels to adapt a LoRA module in the visual encoder per speaker (no labels).
- Comparison point: plain confidence-thresholded self-training (Kim et al. 2022). Our claim to test: *verification is a better pseudo-label filter than model confidence.*
- LRS3 has little data per speaker; run it only on speakers with enough clips (or on LRS-VoxMM dev).

### 3.7 C5 (extension) — Viseme-aware phonetic retrieval
- Targets VALLR's rare-word / proper-noun errors (e.g. *Aaronson* → *Erasmus*, Table 4), which the LLM cannot recover from memory.
- Index a large lexicon (CMU dict, ~130k words) and frequent WikiText phrases by their **viseme** sequence.
- For each span C3 marks as unsupported, retrieve the top-k entries by edit distance weighted with the Stage-1 viseme confusion matrix (same-viseme swaps are almost free), and give them to the LLM as candidate words for the infill.
- Every retrieved candidate is re-verified by C1, so retrieval can only add options — the video still decides.
- Ablation: repair with vs without retrieval; reported on a rare-word / proper-noun subset.

## 4. Main claim

> Adding visual verification and localised repair to a VALLR-style pipeline reduces **WER** and **unsupported words** compared to the same pipeline without them — on both a low-data encoder (Track A) and a strong pre-trained encoder (Track B).

Every comparison keeps the encoder, LLM, data and test set fixed; only C1–C3 change. Absolute WER on a T4 is not expected to beat VALLR's 18.7% in Track A.

## 5. Baselines (run in each track)

| Baseline | Purpose |
|---|---|
| VALLR-style 1-best → LLM (reproduced) | Main baseline |
| + N-best input to LLM (VALLR-Pin style) | Prior-work baseline for C1 |
| + N candidates reranked by LLM score only | Shows the visual score matters |
| CTC-only phoneme decoding + lexicon | Lower bound |

## 6. Metrics

| Metric | Measures |
|---|---|
| WER / CER | Standard accuracy |
| PER, VER | Phoneme / viseme error rate of Stage 1 |
| **Unsupported-word rate** (new) | % of output words whose aligned visual support < τ — hallucination proxy |
| Homophone / homophene subset accuracy | Ambiguity handling |
| Rare-word / proper-noun WER | C5 retrieval (words outside the top 10k of the training text) |
| Per-speaker WER before/after adaptation | C4 |
| Latency, #LLM calls | Cost of verify + repair |
| Mean ± std over 3 seeds; bootstrap significance | Reliability |

## 7. Data

| Dataset | Role | Split used | Access / notebook |
|---|---|---|---|
| **LRS3** trainval (~30 h, TED/TEDx) | Stage-1 training (Track A) and CTC-head training (Track B); no `pretrain` split | train + speaker-disjoint val (2% of folders held out) | Hugging Face (`HF_TOKEN`) · `01_lrs3_data` (3 shards) |
| **LRS3** test (1,321 clips) | Main test set — head-to-head with VALLR (18.7% WER) | test | Pre-cropped parquet files → `parquet_to_rois` · `01_lrs3_data` |
| **LRS-VoxMM** test | Second test set — unconstrained, cross-domain speech | test | Personal KAIST links (`VOXMM_URLS` secret) · `02_voxmm_data` |
| **LRS-VoxMM** dev (~27k clips) | Optional extra validation / C4 adaptation | val | Same · `02_voxmm_data` (3 shards) |
| **WildVSR** | Out-of-domain test, as reported by FLARE and *From Hype to Insight* | test | Access not yet checked |
| LRS3 train transcripts + WikiText-103 (50,000 sentences) + CMU dict | LLM phoneme→sentence training text, as in VALLR | — | Free · `04_llm_and_pvr` |
| GRID | Earlier development set only (pipeline tested end-to-end on 20 clips); not used for reported results | — | ✅ in `GRID/` |

Rules:
- **No test sentences** (LRS3 test, VoxMM test, WildVSR) in the LLM training text or in the C5 retrieval index.
- Track B: confirm from the checkpoint's documentation that it never trained on LRS3 test, and check speaker overlap with LRS-VoxMM.
- Our trainval crops are made with MediaPipe, while the LRS3 test crops come pre-cropped; notebook 03 shows both side by side and the crop scale is adjusted until they match. Track B uses the crop size and normalisation its checkpoint expects.
- All mouth ROIs are 96 × 96 grayscale; training clips longer than 250 frames (10 s) are skipped.

## 8. Compute

- Data preparation: Kaggle CPU notebooks (01, 02).
- Track A encoder: Kaggle T4 GPU, `configs/visual_lrs3.yaml` (d_model 256, 12 layers, viseme head at layer 6, 40 epochs, batch 16, lr 6e-4).
- Track B encoder: frozen pre-trained checkpoint; only CTC heads (+ optional LoRA) trained → a few T4 hours.
- LLM: 1B–1.7B with **QLoRA 4-bit** → fits a 16 GB T4.
- Verify + repair (+ retrieval): inference only.

## 9. Plan

| Weeks | Task |
|---|---|
| 1–2 | LRS3 trainval/test + LRS-VoxMM test preprocessing; phoneme + viseme labels; crop check against LRS3 test |
| 3–4 | Track A: train encoder + CTC heads; Track B: load checkpoint, train CTC heads; PER/VER on LRS3 val/test |
| 5 | Stage 2: QLoRA LLM with Stage-1-style noise and N-best input; reproduce VALLR-style baseline in both tracks |
| 6–7 | C1 verify + C2 viseme-tolerant scoring; tune λ, N on LRS3 val |
| 8 | C3 repair + unsupported-word metric |
| 9 | C5 phonetic retrieval (if C1–C3 work); rare-word WER |
| 10 | Cross-domain tests (LRS-VoxMM, WildVSR if available); C4 (if time); optional landmark stream |
| 11–12 | Ablations over 3 seeds, paper writing |

## 10. Risks

| Risk | Mitigation |
|---|---|
| Visual score too weak to separate candidates | Fall back to phoneme-level score; tune λ; report as finding |
| Strong encoder (Track B) leaves little for PVR to fix | Report per-track gains; focus on unsupported-word rate and homophene subset |
| Concurrent publication of verify-style rescoring | Keep C2 (viseme-tolerant) + C3 (repair) + new metric as distinct contributions; re-check arXiv monthly |
| Small model on a T4 → absolute WER well above VALLR's 18.7% (Track A) | Claim the **gain of PVR over our own reproduced baseline**, not the absolute number |
| Pre-trained checkpoint's front-end or crop does not match our ROIs | Re-crop with its own preprocessing script for Track B |
| Our trainval crops differ from the pre-cropped LRS3 test crops | Visual check in notebook 03; adjust crop scale |
| LRS-VoxMM links expire or WildVSR not accessible | LRS3 test alone still supports the main claims |
| Retrieval index leaks test sentences (C5) | Build it only from CMU dict + WikiText/BookCorpus; check overlap with test transcripts |
| LLM too slow | 1B model, small N, cache candidates |

## 11. Relation to MoVi-Swin

MoVi-Swin (the project's original encoder plan) is no longer part of this paper. Two of its ideas survive: the **viseme CTC head** (now needed by C2) and the **lip-landmark stream** (now an optional Stage-1 addition). A stronger Stage-1 encoder can be plugged in later without changing PVR.
