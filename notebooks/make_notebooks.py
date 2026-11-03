"""Generates the Kaggle notebooks in this folder (run: python notebooks/make_notebooks.py).

01_lrs3_data        CPU   download LRS3 (trainval + test) from Hugging Face, make mouth ROIs
02_voxmm_data       CPU   download LRS-VoxMM from your KAIST links, make mouth ROIs
03_train_visual     GPU   train the phoneme/viseme CTC visual model on LRS3 trainval
04_llm_and_pvr      GPU   LLM fine-tuning + Propose-Verify-Repair on LRS3 test and LRS-VoxMM test

Secrets (Kaggle: Add-ons -> Secrets): HF_TOKEN (01, 04), VOXMM_URLS (02).
"""
import json
from pathlib import Path

REPO = "https://github.com/durgaanuhya02/MoVi-Swin-Motion--and-Viseme-Aware-Space-Time-Swin-Transformer-for-Robust-and-Efficient-Lip-Reading.git"
LRS3_REPO = "TheNHz/ellipsis-lrs3-raw"

SETUP = f"""# --- setup: clone the project code ---
import os, subprocess
if not os.path.exists("/kaggle/working/repo"):
    subprocess.run(["git", "clone", "--depth", "1", "{REPO}", "/kaggle/working/repo"], check=True)
os.chdir("/kaggle/working/repo")
!git log --oneline -1"""

SECRET = """# --- load a Kaggle secret into an environment variable (value is never printed) ---
def load_secret(name):
    from kaggle_secrets import UserSecretsClient
    os.environ[name] = UserSecretsClient().get_secret(name)
    print(f"{name} loaded")"""

CLEANUP = "!rm -rf /kaggle/working/repo   # keep only the data in the saved output"


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text}


def notebook(cells, gpu):
    return {"nbformat": 4, "nbformat_minor": 5, "cells": cells,
            "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                         "kaggle": {"accelerator": "nvidiaTeslaT4" if gpu else "none",
                                    "isInternetEnabled": True}}}


lrs3 = notebook([
    md("# 01 · LRS3 data (CPU)\n"
       f"Downloads LRS3 from the Hugging Face mirror `{LRS3_REPO}` (only the parts VALLR uses: "
       "**trainval** 10 GB and **test** 0.7 GB) and makes 96×96 mouth ROIs.\n\n"
       "Run this notebook **4 times** (one Kaggle notebook copy each), changing the first cell:\n"
       "| copy | `PART` | `SHARD` |\n|---|---|---|\n| lrs3-test | `'test'` | – |\n"
       "| lrs3-trainval-0 | `'trainval'` | `0` |\n| lrs3-trainval-1 | `'trainval'` | `1` |\n"
       "| lrs3-trainval-2 | `'trainval'` | `2` |\n\n"
       "Settings: Accelerator **None**, Internet **On**, secret **HF_TOKEN** attached. "
       "Use **Save Version → Save & Run All (Commit)**."),
    code("PART = 'test'            # 'test' or 'trainval'\nSHARD, NUM_SHARDS = 0, 3   # only used for trainval"),
    code(SETUP),
    code(SECRET + "\nload_secret('HF_TOKEN')"),
    code("!pip install -q mediapipe cmudict huggingface_hub\n!df -h /tmp /kaggle/working; nproc"),
    code(f"""from huggingface_hub import hf_hub_download, snapshot_download
REPO = "{LRS3_REPO}"
if PART == 'test':
    snapshot_download(REPO, repo_type="dataset", local_dir="/tmp/lrs3",
                      allow_patterns=["test-mattymchen/data/*.parquet"], token=os.environ["HF_TOKEN"])
    !python -m src.preprocess.parquet_to_rois --parquet /tmp/lrs3/test-mattymchen/data/*.parquet --out /kaggle/working/lrs3_test_rois
else:
    tar = hf_hub_download(REPO, "ainncy/trainval.tar.gz", repo_type="dataset",
                          local_dir="/tmp/lrs3", token=os.environ["HF_TOKEN"])
    !mkdir -p /tmp/lrs3_tv && tar -xzf {{tar}} -C /tmp/lrs3_tv && rm -f {{tar}}
    !find /tmp/lrs3_tv -maxdepth 3 -type d | head -5"""),
    code("""if PART == 'trainval':
    import glob
    # the folder whose sub-folders contain the .mp4/.txt clips
    root = sorted({os.path.dirname(os.path.dirname(p)) for p in glob.glob('/tmp/lrs3_tv/**/*.mp4', recursive=True)})[0]
    print('LRS3 trainval root:', root)
    !python -m src.preprocess.build_lrs_manifest --root {root} --name lrs3/trainval --split train --val-fraction 0.02 --out /kaggle/working/lrs3_trainval_manifest.jsonl
    OUT = f'/kaggle/working/lrs3_trainval_shard{SHARD}'
    !python -m src.preprocess.extract_rois --manifest /kaggle/working/lrs3_trainval_manifest.jsonl --out {OUT} --workers 4 --detect-every 2 --shard {SHARD}/{NUM_SHARDS} 2>&1 | grep -E 'clips|eta' | tail -20
    !wc -l {OUT}/index.jsonl {OUT}/errors.jsonl; du -sh {OUT}"""),
    code(CLEANUP),
], gpu=False)

voxmm = notebook([
    md("# 02 · LRS-VoxMM data (CPU)\n"
       "Downloads LRS-VoxMM with your personal KAIST links and makes 96×96 mouth ROIs.\n\n"
       "1. Add a Kaggle secret **VOXMM_URLS** containing the download links, one per line "
       "(copy them from `chrome://downloads`). They are never printed.\n"
       "2. Run once with `PART = 'test'`. Then, for the dev set (27k clips), run copies with "
       "`PART = 'dev'` and `SHARD = 0, 1, 2`.\n\n"
       "Settings: Accelerator **None**, Internet **On**. Use **Save & Run All (Commit)**."),
    code("PART = 'test'             # 'test' or 'dev'\nSHARD, NUM_SHARDS = 0, 1   # dev: use NUM_SHARDS = 3"),
    code(SETUP),
    code(SECRET + "\nload_secret('VOXMM_URLS')"),
    code("!pip install -q mediapipe cmudict\n!df -h /tmp /kaggle/working; nproc"),
    code("!python -m src.preprocess.download_urls --env VOXMM_URLS --out /tmp/lrs_voxmm\n"
         "!find /tmp/lrs_voxmm -maxdepth 3 -type d | head -20"),
    code("""import glob
# find the folder of this split: a directory named like the split whose sub-folders hold .mp4 clips
cands = sorted({os.path.dirname(os.path.dirname(p)) for p in glob.glob('/tmp/lrs_voxmm/**/*.mp4', recursive=True)})
root = [c for c in cands if PART in os.path.basename(c).lower()]
print('candidate roots:', cands)
assert root, f'no folder for {PART}: check the folder list above and set root manually'
root = root[0]; print('using', root)
SPLIT = 'test' if PART == 'test' else 'val'
!python -m src.preprocess.build_lrs_manifest --root {root} --name voxmm/{PART} --split {SPLIT} --out /kaggle/working/voxmm_{PART}_manifest.jsonl
OUT = f'/kaggle/working/voxmm_{PART}_shard{SHARD}'
!python -m src.preprocess.extract_rois --manifest /kaggle/working/voxmm_{PART}_manifest.jsonl --out {OUT} --workers 4 --detect-every 2 --shard {SHARD}/{NUM_SHARDS} 2>&1 | grep -E 'clips|eta' | tail -20
!wc -l {OUT}/index.jsonl {OUT}/errors.jsonl; du -sh {OUT}"""),
    code(CLEANUP),
], gpu=False)

AUTO_AVSR_COMMIT = "182b62837773ab01052d4ac21ef1d2203ea7d267"
AUTO_AVSR_CKPT = "1shcWXUK2iauRhW9NbwCc25FjU1CoMm8i"   # Google Drive id of vsr_trlrs3vox2_base.pth

train_visual = notebook([
    md("# 03 · Train the visual model on LRS3 trainval (GPU)\n"
       "Add inputs: the outputs of **lrs3-trainval-0/1/2** and **lrs3-test**. Accelerator **GPU T4**, Internet **On**.\n\n"
       "Run it twice (one notebook copy each), changing `TRACK` in the first cell:\n"
       "| copy | `TRACK` | model |\n|---|---|---|\n"
       "| visual-vit | `'vit'` | Track A: VideoMAE ViT-Base + temporal Transformer |\n"
       "| visual-autoavsr | `'autoavsr'` | Track B: pre-trained Auto-AVSR encoder + new CTC heads |\n\n"
       "The training saves `last.pt` every epoch. If the 12-hour session ends, add this notebook's own "
       "previous output as input, set `RESUME_FROM`, and run again. At the end it evaluates PER/VER and "
       "writes the confusion matrices (`eval_val/`, `eval_test/`)."),
    code("TRACK = 'vit'            # 'vit' (Track A) or 'autoavsr' (Track B)\n"
         "RESUME_FROM = None       # e.g. '/kaggle/input/<previous-output>/runs/visual_lrs3_vit'"),
    code(SETUP),
    code("!pip uninstall -y -q torchao   # Kaggle's old torchao makes peft refuse to load; not used here\n"
         "!pip install -q cmudict transformers==4.57.1 peft==0.21.2 gdown   # versions the models were tested with"),
    code("import glob, json\n"
         "INDEX = sorted(glob.glob('/kaggle/input/**/lrs3_trainval_shard*/index.jsonl', recursive=True))\n"
         "TEST_INDEX = glob.glob('/kaggle/input/**/lrs3_test_rois/index.jsonl', recursive=True)\n"
         "INDEX_JSON = json.dumps(INDEX)\nprint(INDEX, TEST_INDEX)\n"
         "CONFIG = f'configs/visual_lrs3_{TRACK}.yaml'\n"
         "OUT = f'/kaggle/working/runs/visual_lrs3_{TRACK}'"),
    md("### Track B only: Auto-AVSR code and checkpoint\n"
       "If the Google Drive download fails (quota), download `vsr_trlrs3vox2_base.pth` from the "
       "[Auto-AVSR model zoo](https://github.com/mpc001/auto_avsr), upload it as a Kaggle dataset, add it as "
       "input and set `CKPT` to its path."),
    code(f"""CKPT = '/kaggle/working/vsr_trlrs3vox2_base.pth'
if TRACK == 'autoavsr':
    if not os.path.exists('/kaggle/working/auto_avsr'):
        !git clone -q https://github.com/mpc001/auto_avsr /kaggle/working/auto_avsr
        !cd /kaggle/working/auto_avsr && git checkout -q {AUTO_AVSR_COMMIT}
    uploaded = glob.glob('/kaggle/input/**/vsr_trlrs3vox2_base.pth', recursive=True)
    if uploaded:
        CKPT = uploaded[0]
    elif not os.path.exists(CKPT):
        !gdown {AUTO_AVSR_CKPT} -O {{CKPT}}
    !ls -lh {{CKPT}}"""),
    md("### Check: do our trainval crops look like the pre-cropped LRS3 test crops?\n"
       "Top rows: our MediaPipe crops (trainval). Bottom rows: test-set crops. If the mouth size or position "
       "differs a lot, re-run notebook 01 with a different `--crop-scale` before training."),
    code("""import cv2, numpy as np, matplotlib.pyplot as plt
from src.data.dataset import ROIDataset
from src.preprocess.extract_rois import read_video
def frames_of(index, n):
    ds = ROIDataset(index)
    return [read_video(f"{e['_root']}/{e['roi']}")[10] for e in ds.entries[:n]]
row1 = frames_of(INDEX[0], 6); row2 = frames_of(TEST_INDEX[0], 6)
plt.figure(figsize=(12, 4)); plt.imshow(np.vstack([np.hstack(row1), np.hstack(row2)]), cmap='gray'); plt.axis('off')
plt.title('top: our trainval crops   |   bottom: LRS3 test crops'); plt.show()"""),
    code("import shutil\nos.makedirs(OUT, exist_ok=True)\n"
         "if RESUME_FROM:\n    for f in ['last.pt', 'best.pt', 'log.jsonl']:\n"
         "        if os.path.exists(f'{RESUME_FROM}/{f}'): shutil.copy(f'{RESUME_FROM}/{f}', OUT)"),
    code("EXTRA = f'model.checkpoint={CKPT}' if TRACK == 'autoavsr' else ''\n"
         "!python -m src.train_visual --config {CONFIG} --set 'index={INDEX_JSON}' out_dir={OUT} {EXTRA}"),
    code("!tail -5 {OUT}/log.jsonl"),
    md("## PER / VER and confusion matrices (validation and test)"),
    code("!python -m src.eval_visual --checkpoint {OUT}/best.pt --index {' '.join(INDEX)} --split val --out {OUT}/eval_val\n"
         "!python -m src.eval_visual --checkpoint {OUT}/best.pt --index {TEST_INDEX[0]} --split test --out {OUT}/eval_test"),
    code("from IPython.display import Image, display\n"
         "display(Image(f'{OUT}/eval_test/confusion_phoneme.png'))\n"
         "display(Image(f'{OUT}/eval_test/confusion_viseme.png'))\n"
         "!rm -f {OUT}/last.pt   # keep only best.pt in the saved output"),
], gpu=True)

llm_pvr = notebook([
    md("# 04 · LLM + Propose-Verify-Repair (GPU)\n"
       "Add inputs: LRS3 trainval shards, lrs3-test, voxmm-test, and the output of notebook 03 (`best.pt`); "
       "for Track B also run `git clone https://github.com/mpc001/auto_avsr /kaggle/working/auto_avsr` first. "
       "Accelerator **GPU T4**, Internet **On**, secret **HF_TOKEN** (with Llama 3.2 access).\n\n"
       "Without Llama access set `LLM = 'Qwen/Qwen2.5-1.5B'`."),
    code(SETUP),
    code("!pip install -q cmudict peft bitsandbytes"),
    code(SECRET + "\nload_secret('HF_TOKEN')"),
    code("import glob\n"
         "LRS3_TRAIN = sorted(glob.glob('/kaggle/input/**/lrs3_trainval_shard*/index.jsonl', recursive=True))\n"
         "LRS3_TEST = glob.glob('/kaggle/input/**/lrs3_test_rois/index.jsonl', recursive=True)[0]\n"
         "VOXMM_TEST = sorted(glob.glob('/kaggle/input/**/voxmm_test_shard*/index.jsonl', recursive=True))\n"
         "TRACK = 'vit'   # which notebook-03 model to use: 'vit' (Track A) or 'autoavsr' (Track B)\n"
         "VISUAL = glob.glob(f'/kaggle/input/**/visual_lrs3_{TRACK}/best.pt', recursive=True)[0]\n"
         "LLM = 'meta-llama/Llama-3.2-1B'\n"
         "TRAIN_ARGS, VOXMM_ARGS = ' '.join(LRS3_TRAIN), ' '.join(VOXMM_TEST)\n"
         "print(LRS3_TRAIN, LRS3_TEST, VOXMM_TEST, VISUAL, sep='\\n')"),
    md("## Stage 2 · phoneme→sentence corpus (LRS3 train transcripts + WikiText, as in VALLR) and LoRA"),
    code("!cat {TRAIN_ARGS} > /kaggle/working/lrs3_train_all.jsonl\n"
         "!python -m src.llm.build_corpus --manifest /kaggle/working/lrs3_train_all.jsonl --split train "
         "--wikitext 50000 --out /kaggle/working/llm_corpus.jsonl"),
    code("!python -m src.llm.train_p2t --config configs/llm_p2t.yaml --set model_name={LLM} max_steps=3000"),
    md("## Stage 3 · tune λ on LRS3 validation, then test on LRS3 and LRS-VoxMM"),
    code("ADAPTER = '/kaggle/working/runs/llm_p2t/adapter'\n"
         "!python -m src.pvr.run --visual {VISUAL} --index {TRAIN_ARGS} --split val --llm {LLM} "
         "--adapter {ADAPTER} --out /kaggle/working/runs/pvr_lrs3_val --n 5 --limit 300 --sweep | tail -30"),
    code("import json\nsweep = json.load(open('/kaggle/working/runs/pvr_lrs3_val/metrics.json'))['lambda_sweep']\n"
         "LAM = float(min(sweep, key=sweep.get)); print('best lambda', LAM, sweep)"),
    code("!python -m src.pvr.run --visual {VISUAL} --index {LRS3_TEST} --split test --llm {LLM} "
         "--adapter {ADAPTER} --out /kaggle/working/runs/pvr_lrs3_test --n 5 --lam {LAM} --sweep | tail -40"),
    code("!python -m src.pvr.run --visual {VISUAL} --index {VOXMM_ARGS} --split test --llm {LLM} "
         "--adapter {ADAPTER} --out /kaggle/working/runs/pvr_voxmm_test --n 5 --lam {LAM} --sweep | tail -40"),
], gpu=True)

here = Path(__file__).parent
for old in ["01_preprocess_grid", "02_train_visual", "03_llm_and_pvr"]:
    (here / f"{old}.ipynb").unlink(missing_ok=True)
for name, nb in [("01_lrs3_data", lrs3), ("02_voxmm_data", voxmm),
                 ("03_train_visual", train_visual), ("04_llm_and_pvr", llm_pvr)]:
    (here / f"{name}.ipynb").write_text(json.dumps(nb, indent=1))
    print("wrote", name)
