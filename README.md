# PVR: Viseme-Aware Visual Verification and Localized Repair for LLM-Based Visual Speech Recognition

Lip reading (visual speech recognition) turns silent video of a speaker's mouth into text.
Phoneme-centric systems such as **VALLR** (Thomas, Fish & Bowden, ICCV 2025) predict phonemes
from video and let a large language model (LLM) rebuild the sentence. They are **open-loop**:
the LLM's sentence is never checked against the video, which causes homophone errors and
sentences the lips do not support (hallucination).

**Propose-Verify-Repair (PVR)** closes the loop:

```
Silent video ──► visual encoder ──► phoneme CTC + viseme CTC
             ──► PROPOSE: LLM writes N candidate sentences
             ──► VERIFY:  score each candidate against the video (viseme-level CTC + LLM score)
             ──► REPAIR:  re-write only the words the video does not support
             ──► final sentence
```

| Contribution | What it does |
|---|---|
| **C1 Visual verification** | Re-ranks LLM candidates by their CTC likelihood given the video |
| **C2 Viseme-tolerant scoring** | Look-alike sounds (p/b/m, s/z) get equal visual scores; the LLM decides between them |
| **C3 Localised repair** | Forced alignment finds unsupported words; only those are re-written |
| **Unsupported-word rate** | New metric for visually unsupported (hallucinated) output words |

Full method, related work and plan: [`APPROACH.md`](APPROACH.md). Architecture figure: [`report/pvr_architecture.png`](report/pvr_architecture.png).

## Models

| Stage | Model | Config |
|---|---|---|
| Visual encoder, Track A | VideoMAE ViT-Base (112×112, 16-frame windows) + temporal Transformer; top 4 layers fine-tuned, LoRA on the rest | `configs/visual_lrs3_vit.yaml` |
| Visual encoder, Track B | Pre-trained [Auto-AVSR](https://github.com/mpc001/auto_avsr) encoder, top 2 blocks fine-tuned | `configs/visual_lrs3_autoavsr.yaml` |
| Phoneme → sentence LLM | Llama 3.2-1B with QLoRA (4-bit) | `configs/llm_p2t.yaml` |

Both visual encoders have a phoneme CTC head (39 phonemes + blank) and a viseme CTC head (13 visemes + blank),
trained with `loss = CTC(phonemes) + 0.3 · CTC(visemes)`.

## Data

| Dataset | Use | Clips |
|---|---|---|
| LRS3 trainval (30 h, VALLR's setting) | Training; 2 % of speakers held out for validation | ~31,900 |
| LRS3 test | Main test set | 1,321 |
| LRS-VoxMM test | Cross-domain test set | 2,125 |
| LRS3 transcripts + WikiText-103 + CMU dict | LLM phoneme → sentence training text | — |

Datasets are **not** included in this repository. Obtain them from their official sources under their licences.

## Running on Kaggle

The notebooks clone this repository, so they always run the latest pushed code.

| Notebook | Accelerator | What it does |
|---|---|---|
| `01_lrs3_data` | CPU | Downloads LRS3 (test, or one third of trainval per copy) and makes 96×96 mouth crops |
| `02_voxmm_data` | CPU | Makes mouth crops for LRS-VoxMM |
| `03_train_visual` | GPU T4 | Trains a visual encoder (`TRACK = 'vit'` or `'autoavsr'`), then writes PER / VER and confusion matrices |
| `04_llm_and_pvr` | GPU T4 | Fine-tunes the LLM and runs all PVR systems and baselines on LRS3 test and LRS-VoxMM test |

Secrets (Kaggle → Add-ons → Secrets): `HF_TOKEN` (notebooks 01, 04), `VOXMM_URLS` (02, only when downloading VoxMM by link).
Each notebook's output is added as an input to the next one. Training saves `last.pt` every 1,000 steps; to continue
after Kaggle's 12-hour limit, add the previous output as input and set `RESUME_FROM`.

## Code

```
src/
├─ preprocess/   LRS manifest builder, MediaPipe mouth cropping, parquet / URL helpers
├─ text/         text → phonemes (CMU dict) → visemes
├─ data/         dataset over mouth crops (augmentation, skipping unreadable clips)
├─ models/       ViT (Track A), Auto-AVSR (Track B), Conformer; shared CTC losses
├─ train_visual.py   train a visual encoder        eval_visual.py   PER / VER + confusion matrices
├─ llm/          phoneme → sentence corpus, prompts, QLoRA training
├─ pvr/          propose, verify (viseme CTC scoring, forced alignment), repair, evaluation
└─ metrics.py    WER, CER, PER, VER, alignment
configs/  notebooks/  tests/  report/
```

## Local setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |   Linux/macOS: source .venv/bin/activate
pip install torch torchvision opencv-python mediapipe numpy pyyaml cmudict transformers==4.57.1 peft==0.21.2 bitsandbytes matplotlib pytest
python -m pytest -q tests
```

## Status

- [x] LRS3 and LRS-VoxMM preprocessing
- [x] Visual encoders (Track A ViT, Track B Auto-AVSR) and PER / VER evaluation
- [ ] Visual encoder training — in progress
- [ ] LLM fine-tuning with realistic phoneme noise and N-best input
- [ ] PVR experiments, baselines and ablations
- [ ] Report / paper

## Results

*To be added.*
