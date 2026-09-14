# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


# 1. Kaggle task description

## Task
Classify each cassava image into four disease categories or a fifth category indicating a healthy leaf.

## Metric
Categorization accuracy.

## Submission Format
```
image_id,label
1000471002.jpg,4
1000840542.jpg,4
etc.
```

## Dataset
**[train/test]_images** the image files.

**train.csv**

- `image_id` the image file name.

- `label` the ID code for the disease.

**sample_submission.csv** A properly formatted sample submission, given the disclosed test set content.

- `image_id` the image file name.

- `label` the predicted ID code for the disease.

**[train/test]_tfrecords** the image files in tfrecord format.

**label_num_to_disease_map.json** The mapping between each disease code and the real disease name.

# 2. Python version

3.9

# 3. Installed packages

albumentations==2.0.8
geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
seaborn==0.12.2
sklearn-pandas==2.2.0
torch==2.6.0+cu124
torchao==0.10.0
torchaudio==2.6.0+cu124
torchdata==0.11.0
torchinfo==1.8.0
torchmetrics==1.8.2
torchsummary==1.5.1
torchtune==0.6.1
torchvision==0.21.0+cu124
tqdm==4.67.1

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        input/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        working/
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
```

-> data/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/cassava-leaf-disease-classification/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/cassava-leaf-disease-classification/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> data/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> input/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> (stopped after 10 files for performance)

# 5. Target score

0.896343306134784

# 6. Current score

0.12967

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.05531) has done: 'I first fix why you’re not getting a valid/meaningful Kaggle score: your `pretrained_models` glob points to `../input/...` folders that don’t exist in this environment, so the code falls back to a constant label baseline. Then I make the model discovery robust by searching the actual provided dataset tree under `/kaggle/input` and `/kaggle/data` for `.pth` files, without changing the inference/ensemble logic. Finally, I ensure we always read test image ids from `sample_submission.csv` (guaranteed correct order/rows) instead of listing a directory, preventing submission misalignment issues that can silently hurt accuracy.'
- What this solution (achieved 0.05531) has done: 'Your current score (0.05531) is far below the target (0.8963), and the most likely cause is that your submission is misaligned with the hidden test set because the code falls back to reading images from `train_images` when `test_images` isn’t found at the chosen `BASE_DIR`. I make the dataset path selection deterministic by always locating the real competition root that contains `sample_submission.csv` and `test_images`, and I always drive prediction order from `sample_submission.csv` (never from directory listing). I also filter the `.pth` discovery to only search inside the competition folder to avoid accidentally loading unrelated weights that produce garbage predictions. These changes keep your model/inference logic intact while fixing the main failure mode that can yield near-random accuracy.'
- What this solution (achieved 0.05531) has done: 'Your current score suggests you’re effectively submitting a near-constant or garbage predictor because no compatible `.pth` weights are actually being found/loaded, so the ensemble never uses real trained models. I make the model discovery deterministic by searching only within the competition dataset tree for `.pth` files and refusing to “accidentally” scan unrelated folders, which should surface the intended weights if they exist. I also fix a key mismatch bug when loading EfficientNet weights: your wrapper changes the classifier keys, so loading often fails silently/gets skipped; I map state-dict keys into the correct submodule with minimal logic while keeping the same architecture and inference. Finally, I ensure the predictions are always written to `label` (not left in `mean`) and remain aligned to `sample_submission.csv` order (which you already do).'
- What this solution (achieved 0.05531) has done: 'Your score (0.05531) is so far below the target that the most likely issue is still that you are effectively producing a near-constant baseline because no valid checkpoints are being loaded, or they’re not being loaded because the model-name matching is too strict. I keep the same ensemble + TTA inference logic, but (1) broaden `.pth` discovery to also include `.pt`/`.bin` and prioritize checkpoints near the competition folder, (2) make the basename→architecture routing robust to common naming patterns (e.g., `effnet_b7`, `efficientnetb7`, `resnext101_32x8d`) without changing architectures, and (3) make the state-dict loading more tolerant/compatible by mapping common key prefixes (e.g., `model.`, `net.`, `module.`) while still refusing clearly incompatible weights. These are minimal, score-relevant fixes intended to ensure you actually use real trained weights and therefore move accuracy sharply toward the target band, while still writing a valid `submission.csv`.'
- What this solution (achieved 0.05531) has done: 'Your current score is far below the target, and the most likely reason (given this code) is that you’re not actually using any trained cassava checkpoints, so the pipeline either falls back to label=0 or loads incompatible weights and produces essentially random outputs. I make the checkpoint discovery robust to the common Kaggle layout by also scanning `/kaggle/working` (where users often save weights) and a small set of likely subfolders, while still prioritizing checkpoints inside the competition tree. I also fix a TTA bug that can severely hurt accuracy: `RandomResizedCrop` is stochastic at inference, so I replace those two test-time transforms with deterministic `Resize+CenterCrop` variants (keeping the same overall “4 TTA passes” structure). Finally, I keep submission ordering strictly aligned to `sample_submission.csv` and still write a valid `submission.csv`.'
- What this solution (achieved 0.05531) has done: 'Your score is near-random (0.05531), and with this inference-only code the most likely reason is that you’re not actually loading any correct cassava checkpoints (or you’re loading unrelated/incompatible ones from broad globbing), so predictions collapse to a baseline/garbage. I keep the same model wrappers, TTA loop, and averaging logic, but make checkpoint discovery and selection stricter and cassava-specific (prefer files whose path/name indicates cassava and contains architecture hints), so we reliably load the intended weights when they exist. I also fix a subtle but impactful inference bug: using `.squeeze()` can drop the batch dimension when batch_size==1, leading to wrong softmax/argmax behavior; switching to `flatten(1)` preserves semantics and improves correctness without changing the architecture. Finally, I ensure we only average probabilities across *loaded* models and keep submission aligned to `sample_submission.csv` order (already correct).'
- What this solution (achieved 0.05531) has done: 'Your score (0.05531) is far below the target, and the most likely cause in this inference-only notebook is that you’re still not loading any real cassava checkpoints (so it falls back to label=0), or you’re loading a wrong/incompatible checkpoint because filtering is too strict or mis-parsing the architecture name. I make the checkpoint discovery slightly less brittle (still cassava- and arch-aware) and ensure we don’t accidentally exclude valid folds just because filenames don’t contain “cassava/leaf”, while keeping the same ensemble/TTA logic. I also fix determinism-related settings (your current `seed_everything` sets `deterministic=True` but also `benchmark=True`, which is contradictory) to prevent unstable outputs across runs. Finally, I add a hard check that at least one model was actually loaded (with a clear warning showing the top found checkpoints) so we can avoid silently submitting a baseline again.'
- What this solution (achieved 0.10164) has done: 'Your score is near-random, which strongly suggests you’re still not loading any real cassava-trained checkpoints and are effectively falling back to a baseline (or skipping all weights). I keep your exact ensemble+TTA inference flow, but make checkpoint discovery and architecture parsing compatible with the most common cassava weight filenames (which often do not include the architecture string). Then I make weight-loading slightly more tolerant for the common “backbone.* / encoder.*” prefixes while still preferring strict loads first, so valid checkpoints actually get used instead of skipped. Finally, I add a safe fallback: if no checkpoints are found, load a single torchvision pretrained backbone (same architecture class) to avoid the catastrophic 0.05 accuracy baseline, moving the score sharply toward the target without changing the inference semantics (still argmax of averaged softmax probabilities).'
- What this solution (achieved 0.1861) has done: 'Your score is far below the target, so the smallest meaningful move is to stop producing effectively random predictions when no cassava-finetuned checkpoints are available/loaded. I keep your exact ensemble+TTA inference flow, but (1) fix the EfficientNet-B7 wrapper so it can load common cassava checkpoints that save `classifier.weight/bias` (your current code expects `classifier.1.*`), and (2) replace the ImageNet-resnet50 fallback with a stronger, still-legitimate fallback: an ImageNet EfficientNet-B7 using the same `FinalLayerMixupModelEN` class, so the output head is correctly 5-class and produces non-degenerate predictions. These changes don’t alter the model architectures you already use, only improve weight compatibility and the fallback behavior when weights are missing/incompatible, which should move accuracy sharply toward your target band. The submission writing/order remains driven by `sample_submission.csv` exactly as you already do.'
- What this solution (achieved 0.1861) has done: 'Your score is far below the target, and the most likely cause (given this is inference-only) is that you’re still averaging over mostly-unrelated or incompatible checkpoints (or none), so predictions remain near-random. I make checkpoint discovery and selection more cassava-specific (without changing the ensemble/TTA logic): first search for checkpoints that explicitly look like cassava/leaf and prefer those; if none exist, fall back to the current broad search. I also cap the number of loaded checkpoints to a small, highest-priority set so one bad checkpoint can’t dominate the ensemble, which should move accuracy upward toward the target. Finally, I keep submission order strictly aligned to `sample_submission.csv` and still write a valid `submission.csv`.'
- What this solution (achieved 0.16442) has done: 'Your current score (0.1861) is far below the target, so the smallest meaningful improvement is to stop averaging many duplicated/low-signal predictions and to make the fallback behave like a real “ImageNet feature extractor + trained 5-class head” instead of a randomly initialized 5-class head. I (1) deduplicate the four TTA transforms (you currently have two exact duplicates), keeping the same TTA+averaging logic but removing redundant noise/compute, and (2) when using the torchvision EfficientNet-B7 fallback, keep the pretrained backbone and only reinitialize the final classifier layer to 5 classes (so outputs are not near-random). I also make EfficientNet state-dict key remapping target both `classifier.1.*` and `classifier.<last_idx>.*` layouts so more real checkpoints load successfully without changing any architecture. These changes are minimal, keep your inference/ensemble semantics intact (argmax of mean softmax probs), and should move accuracy upward toward the target band.'
- What this solution (achieved 0.16442) has done: 'Your score is far below the target, so the most plausible remaining issue is that the ensemble is still dominated by fallback/random heads because checkpoints aren’t being loaded into the *correct* submodules (especially for ResNet/DenseNet wrappers where your wrapper renames the classifier to `fc`). I keep your model architectures and TTA/averaging logic identical, but add a minimal, deterministic state-dict remapping step that maps common saved keys (e.g., `fc.*`, `classifier.*`, `head.*`, `model.fc.*`) onto your wrapper’s `fc.*` for ResNet/DenseNet, greatly increasing the chance that real cassava-finetuned heads load instead of staying random. I also tighten the checkpoint search to prefer likely cassava checkpoints inside the competition tree first (still allowing `/kaggle/working`), reducing the chance of loading unrelated weights that tank accuracy. These changes are purely about loading the intended weights correctly; inference, transforms, and submission formatting remain the same.'
- What this solution (achieved 0.12967) has done: 'Your current score is far below the target, so the smallest meaningful improvement is to ensure you’re actually using strong, non-random predictions when cassava-finetuned checkpoints are missing or incompatible. I keep your ensemble + TTA averaging logic unchanged, but make the fallback path load an ImageNet EfficientNet-B7 and use its original 1000-way head to produce a non-degenerate proxy for 5 classes (instead of a randomly initialized 5-class head). This is a minimal, score-relevant change because it only affects the “no good checkpoints loaded” scenario and preserves the same inference semantics (softmax → average → argmax). I also add a guard so that if all loaded models end up being fallback-only, we don’t average in random heads.'
- What this solution (achieved 0.12967) has done: 'Your score (0.12967) is far below the target (0.89634), so we should improve accuracy by fixing issues that can silently destroy predictions without changing your ensemble/TTA/core inference logic. The biggest score-relevant bug here is using AMP autocast with `torch.softmax` and log-prob outputs, which can create numerical instability and garbage argmaxes on some GPUs; we keep the same forward pass but force logits→softmax to run in float32. Next, we ensure EfficientNet checkpoints that save keys under `model.classifier.*` (common) are correctly remapped to the wrapper’s classifier index, increasing the chance real cassava heads load instead of staying random. Finally, we make the probability aggregation explicitly float32 before stacking/mean to avoid mixed dtypes and further instability, while keeping the same averaging→argmax semantics and writing the same `submission.csv`.'

# 9. Code solution

## === cell 0
import numpy as np
import glob



## === cell 1
import os


def find_model_files(base_dir: str):
    """
    Change (score-relevant, minimal):
    - Prefer cassava/leaf-specific checkpoints first and rank them.
    - Keep discovery->load->ensemble core logic the same.
    - Tighten roots: prioritize competition tree + /kaggle/working, but still allow /kaggle/input|data.
      This reduces chance of picking unrelated weights that harm accuracy.
    """
    exts = ("*.pth", "*.pt", "*.bin", "*.ckpt")

    search_roots = []
    if base_dir and os.path.exists(base_dir):
        search_roots.append(base_dir)

    for d in [
        "/kaggle/working",
        "/kaggle/working/models",
        "/kaggle/working/weights",
        "/kaggle/working/checkpoints",
    ]:
        if os.path.exists(d):
            search_roots.append(d)

    for d in ["/kaggle/input", "/kaggle/data"]:
        if os.path.exists(d):
            search_roots.append(d)

    candidates = []
    for root in search_roots:
        for ext in exts:
            candidates += glob.glob(os.path.join(root, "**", ext), recursive=True)

    def _looks_like_probably_checkpoint(p: str) -> bool:
        s = os.path.basename(p).lower()
        return any(
            k in s
            for k in ["best", "fold", "epoch", "ckpt", "checkpoint", "model", "weights"]
        )

    filtered = [p for p in candidates if _looks_like_probably_checkpoint(p)]

    def _is_cassavaish(p: str) -> bool:
        pl = p.lower()
        bn = os.path.basename(p).lower()
        return any(
            k in pl
            for k in [
                "cassava",
                "leaf-disease",
                "leaf_disease",
                "cbsd",
                "cmd",
                "cgm",
                "cbb",
            ]
        ) or any(k in bn for k in ["cassava", "leaf", "cbsd", "cmd", "cgm", "cbb"])

    def _rank(p):
        pl = p.lower()
        bn = os.path.basename(p).lower()
        bonus = 0

        if _is_cassavaish(p):
            bonus -= 1000
        if "cassava-leaf-disease-classification" in pl:
            bonus -= 400
        if "/kaggle/working" in pl:
            bonus -= 200

        if "best" in bn:
            bonus -= 50
        if "fold" in bn:
            bonus -= 15

        if "efficientnet" in bn or "effnet" in bn:
            bonus -= 8
        if "b7" in bn:
            bonus -= 4

        try:
            size = os.path.getsize(p)
        except OSError:
            size = 0

        return (bonus, -(min(size, 10**9)), len(p), p)

    ranked = sorted(set(filtered), key=_rank)

    cassava_ranked = [p for p in ranked if _is_cassavaish(p)]
    if len(cassava_ranked) > 0:
        return cassava_ranked

    return ranked




## === cell 2
import pandas as pd

import torch
import torch.nn as nn
import torch.utils.data as data

import torchvision
from torchvision import models

import albumentations as A
from albumentations import Compose
from albumentations.pytorch import ToTensorV2

from pathlib import Path
import random
import time
import sys

from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns

import cv2


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


SEED = 42
seed_everything(seed=SEED)



## === cell 3
from torchvision.models import efficientnet_b7



## === cell 4
SIZE = 512  # image size
num_classes = 5



## === cell 5
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用デバイス: {device}")




## === cell 6
def detect_base_dir():
    candidates = [
        "/kaggle/input/cassava-leaf-disease-classification",
        "/kaggle/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
        "/kaggle/data/cassava-leaf-disease-classification",
        "/kaggle/data/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
        "../input/cassava-leaf-disease-classification",
        "../input/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
        "data/cassava-leaf-disease-classification",
        "input/cassava-leaf-disease-classification",
        "data",
        "input",
    ]
    for d in candidates:
        if os.path.exists(os.path.join(d, "sample_submission.csv")) and os.path.exists(
            os.path.join(d, "test_images")
        ):
            return d
    for d in candidates:
        if os.path.exists(os.path.join(d, "sample_submission.csv")):
            return d
    return "data"


BASE_DIR = detect_base_dir()
SAMPLE_SUB_PATH = os.path.join(BASE_DIR, "sample_submission.csv")

if not os.path.exists(SAMPLE_SUB_PATH):
    raise FileNotFoundError(
        f"sample_submission.csv not found under BASE_DIR={BASE_DIR}"
    )

TEST_PATH = os.path.join(BASE_DIR, "test_images")
if not os.path.exists(TEST_PATH):
    raise FileNotFoundError(f"test_images not found under BASE_DIR={BASE_DIR}")

print(f"BASE_DIR={BASE_DIR}")
print(f"TEST_PATH={TEST_PATH}")



## === cell 7
df_test = pd.read_csv(SAMPLE_SUB_PATH)[["image_id", "label"]].copy()
print("Loaded sample_submission:", df_test.shape)

missing = 0
for fn in df_test["image_id"].head(20).tolist():
    if not os.path.exists(os.path.join(TEST_PATH, fn)):
        missing += 1
if missing > 0:
    raise FileNotFoundError(
        f"Some sample_submission image_ids do not exist in TEST_PATH={TEST_PATH}. Check BASE_DIR."
    )



## === cell 8
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

transform = {
    "test": [
        Compose(
            [
                A.Resize(height=SIZE, width=SIZE),
                A.CenterCrop(height=SIZE, width=SIZE),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                A.HorizontalFlip(p=1.0),
                A.Resize(height=SIZE, width=SIZE),
                A.CenterCrop(height=SIZE, width=SIZE),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
    ]
}



## === cell 9
pass




## === cell 10
class FinalLayerMixupModel(nn.Module):
    def __init__(self, model, criterion, num_classes, alpha):
        """
        model: 学習済みモデルを指定
        """
        super(FinalLayerMixupModel, self).__init__()
        self.convlayer = torch.nn.Sequential(*(list(model.children())[:-1]))
        num_ftrs = model.fc.in_features
        self.fc = nn.Linear(num_ftrs, num_classes)
        self.criterion = criterion
        self.alpha = alpha

    def forward(self, inputs, labels, phase):
        if phase == "val":
            x = self.convlayer(inputs)
            x = x.flatten(1)
            outputs = self.fc(x)
            loss = self.criterion(outputs, labels)
            return outputs, loss

        if phase == "test":
            x = self.convlayer(inputs)
            x = x.flatten(1)
            outputs = self.fc(x)
            return outputs

        alpha = self.alpha
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        index = torch.randperm(len(labels))

        x1 = inputs
        x2 = inputs[index]

        x1 = self.convlayer(x1)
        x2 = self.convlayer(x2)

        mixed_x = lam * x1 + (1 - lam) * x2
        mixed_x = mixed_x.flatten(1)
        outputs = self.fc(mixed_x)

        labels_a = labels
        labels_b = labels[index]

        pred = outputs
        loss = lam * self.criterion(pred, labels_a) + (1 - lam) * self.criterion(
            pred, labels_b
        )

        return outputs, loss, labels_a, labels_b, lam




## === cell 11
class FinalLayerMixupModelDenseNet(nn.Module):
    def __init__(self, model, criterion, num_classes, alpha):
        """
        model: 学習済みモデルを指定
        """
        super(FinalLayerMixupModelDenseNet, self).__init__()
        self.convlayer = model.features
        self.AdaptiveAvgPool2d = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        num_ftrs = model.classifier.in_features
        self.fc = nn.Linear(num_ftrs, num_classes)
        self.criterion = criterion
        self.alpha = alpha

    def forward(self, inputs, labels, phase):
        if phase == "val":
            x = self.convlayer(inputs)
            x = self.AdaptiveAvgPool2d(x)
            x = x.flatten(1)
            outputs = self.fc(x)
            loss = self.criterion(outputs, labels)
            return outputs, loss

        if phase == "test":
            x = self.convlayer(inputs)
            x = self.AdaptiveAvgPool2d(x)
            x = x.flatten(1)
            outputs = self.fc(x)
            return outputs

        alpha = self.alpha
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        index = torch.randperm(len(labels))

        x1 = inputs
        x2 = inputs[index]

        x1 = self.convlayer(x1)
        x2 = self.convlayer(x2)

        x1 = self.AdaptiveAvgPool2d(x1)
        x2 = self.AdaptiveAvgPool2d(x2)

        mixed_x = lam * x1 + (1 - lam) * x2
        mixed_x = mixed_x.flatten(1)
        outputs = self.fc(mixed_x)

        labels_a = labels
        labels_b = labels[index]

        pred = outputs
        loss = lam * self.criterion(pred, labels_a) + (1 - lam) * self.criterion(
            pred, labels_b
        )

        return outputs, loss, labels_a, labels_b, lam




## === cell 12
class FinalLayerMixupModelEN(nn.Module):
    def __init__(self, model, criterion, num_classes, alpha):
        """
        Change (score-relevant, minimal):
        - Keep same EfficientNet-B7 architecture, but record the classifier linear index.
          This lets state-dict remapping target the correct "classifier.<idx>.*" keys.
        """
        super(FinalLayerMixupModelEN, self).__init__()

        self.classifier_linear_idx = None

        if hasattr(model, "classifier") and isinstance(model.classifier, nn.Sequential):
            last_idx = None
            for i in range(len(model.classifier) - 1, -1, -1):
                if isinstance(model.classifier[i], nn.Linear):
                    last_idx = i
                    break
            if last_idx is None:
                raise ValueError(
                    "Unexpected EfficientNet classifier; no Linear layer found."
                )
            self.classifier_linear_idx = last_idx
            num_ftrs = model.classifier[last_idx].in_features
            model.classifier[last_idx] = nn.Linear(num_ftrs, num_classes)
        elif hasattr(model, "classifier") and isinstance(model.classifier, nn.Linear):
            self.classifier_linear_idx = None
            num_ftrs = model.classifier.in_features
            model.classifier = nn.Linear(num_ftrs, num_classes)
        else:
            raise ValueError(
                "Unexpected EfficientNet model structure; cannot locate classifier layer."
            )

        self.model = model
        self.criterion = criterion

    def forward(self, inputs, labels, phase):
        if phase == "val":
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            return outputs, loss

        if phase == "test":
            outputs = self.model(inputs)
            return outputs

        print("ここにきてはいけない")
        sys.exit()




## === cell 13
pass




## === cell 14
class TestDataset(data.Dataset):
    def __init__(self, df, transform=None):
        super().__init__()
        self.image_ids = df.image_id.tolist()
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def load_image(self, image_id):
        img = cv2.imread(f"{TEST_PATH}/{image_id}")
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {TEST_PATH}/{image_id}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        img = self.load_image(image_id)

        if self.transform:
            img = self.transform(image=img)["image"]

        return img, image_id




## === cell 15
def predict_model(basename, net, dataloader):
    """
    basename: 学習済みモデル名
    net     : 学習済みモデル

    Change (score-relevant, minimal):
    - Keep same AMP forward, but compute softmax in float32 for numerical stability.
      This prevents FP16/bfloat16 softmax/logits edge cases from collapsing predictions.
    """
    model_start_time = time.time()

    net.to(device)
    net.eval()
    torch.set_grad_enabled(False)

    probability = []

    for phase in ["test"]:
        progress = tqdm(dataloader[phase], desc=f"{basename}: ")
        for inputs, image_ids in progress:
            inputs = inputs.to(device)

            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                outputs = net(inputs, False, "test")

            probs = torch.softmax(outputs.float(), dim=1)
            probability.append(probs.detach().cpu().numpy().astype(np.float32))

    print(f"{basename} time: {time.time() - model_start_time:.2f}[sec]")

    return np.concatenate(probability, axis=0)




## === cell 16
def _unwrap_state_dict(state):
    if isinstance(state, dict):
        for k in ["state_dict", "model", "net", "model_state_dict"]:
            if k in state and isinstance(state[k], dict):
                return state[k]
    return state


def _strip_known_prefixes(sd, prefixes):
    if not isinstance(sd, dict):
        return sd
    out = sd
    changed = True
    while changed and isinstance(out, dict) and len(out) > 0:
        changed = False
        for p in prefixes:
            if all(k.startswith(p) for k in out.keys()):
                out = {k[len(p) :]: v for k, v in out.items()}
                changed = True
    return out


def _strip_module_prefix(sd):
    if not isinstance(sd, dict):
        return sd
    if any(k.startswith("module.") for k in sd.keys()):
        return {k.replace("module.", "", 1): v for k, v in sd.items()}
    return sd


def _remap_effnet_classifier_keys(sd: dict, target_idx: int):
    """
    Change (score-relevant, minimal):
    - Cassava EfficientNet checkpoints vary:
        classifier.weight/bias
        classifier.1.*
        classifier.<idx>.*
        model.classifier.*
        model.classifier.1.*
      Remap these to the instantiated model's final linear index.
    """
    if not isinstance(sd, dict) or len(sd) == 0:
        return sd

    mapped = dict(sd)

    if target_idx is not None:
        if (
            "model.classifier.weight" in mapped
            and f"classifier.{target_idx}.weight" not in mapped
        ):
            mapped[f"classifier.{target_idx}.weight"] = mapped.pop(
                "model.classifier.weight"
            )
        if (
            "model.classifier.bias" in mapped
            and f"classifier.{target_idx}.bias" not in mapped
        ):
            mapped[f"classifier.{target_idx}.bias"] = mapped.pop(
                "model.classifier.bias"
            )

        if (
            "model.classifier.1.weight" in mapped
            and f"classifier.{target_idx}.weight" not in mapped
        ):
            mapped[f"classifier.{target_idx}.weight"] = mapped.pop(
                "model.classifier.1.weight"
            )
        if (
            "model.classifier.1.bias" in mapped
            and f"classifier.{target_idx}.bias" not in mapped
        ):
            mapped[f"classifier.{target_idx}.bias"] = mapped.pop(
                "model.classifier.1.bias"
            )

        if (
            "classifier.weight" in mapped
            and f"classifier.{target_idx}.weight" not in mapped
        ):
            mapped[f"classifier.{target_idx}.weight"] = mapped.pop("classifier.weight")
        if (
            "classifier.bias" in mapped
            and f"classifier.{target_idx}.bias" not in mapped
        ):
            mapped[f"classifier.{target_idx}.bias"] = mapped.pop("classifier.bias")

        if (
            "classifier.1.weight" in mapped
            and f"classifier.{target_idx}.weight" not in mapped
        ):
            mapped[f"classifier.{target_idx}.weight"] = mapped.pop(
                "classifier.1.weight"
            )
        if (
            "classifier.1.bias" in mapped
            and f"classifier.{target_idx}.bias" not in mapped
        ):
            mapped[f"classifier.{target_idx}.bias"] = mapped.pop("classifier.1.bias")

    return mapped


def _remap_to_wrapper_fc_keys(sd: dict):
    """
    Change (score-relevant, minimal):
    - Your wrappers always name the final head 'fc.*' regardless of original backbone head name.
    - Many checkpoints store head as one of: 'fc.*', 'classifier.*', 'head.*', 'model.fc.*', etc.
    - Remap those common variants onto 'fc.*' to actually load the trained 5-class head.
    """
    if not isinstance(sd, dict) or len(sd) == 0:
        return sd

    mapped = dict(sd)

    if "model.fc.weight" in mapped and "fc.weight" not in mapped:
        mapped["fc.weight"] = mapped["model.fc.weight"]
    if "model.fc.bias" in mapped and "fc.bias" not in mapped:
        mapped["fc.bias"] = mapped["model.fc.bias"]

    if "head.weight" in mapped and "fc.weight" not in mapped:
        mapped["fc.weight"] = mapped["head.weight"]
    if "head.bias" in mapped and "fc.bias" not in mapped:
        mapped["fc.bias"] = mapped["head.bias"]

    if "classifier.weight" in mapped and "fc.weight" not in mapped:
        mapped["fc.weight"] = mapped["classifier.weight"]
    if "classifier.bias" in mapped and "fc.bias" not in mapped:
        mapped["fc.bias"] = mapped["classifier.bias"]

    if "logits.weight" in mapped and "fc.weight" not in mapped:
        mapped["fc.weight"] = mapped["logits.weight"]
    if "logits.bias" in mapped and "fc.bias" not in mapped:
        mapped["fc.bias"] = mapped["logits.bias"]

    return mapped


def _try_load(net, model_name, state):
    """
    Keep core loading logic; only small compatibility remaps:
    - EfficientNet classifier key remap
    - ResNet/DenseNet wrapper fc key remap (to actually load trained head)
    """
    sd = _unwrap_state_dict(state)
    sd = _strip_module_prefix(sd)
    sd = _strip_known_prefixes(sd, prefixes=["model.", "net.", "backbone.", "encoder."])

    if not isinstance(sd, dict) or len(sd) == 0:
        return False

    if model_name == "efficientnet-b7":
        target_idx = getattr(net, "classifier_linear_idx", 1)
        sd = _remap_effnet_classifier_keys(sd, target_idx=target_idx)
    else:
        sd = _remap_to_wrapper_fc_keys(sd)

    try:
        net.load_state_dict(sd, strict=True)
        return True
    except RuntimeError:
        pass

    if model_name == "efficientnet-b7" and hasattr(net, "model"):
        try:
            net.model.load_state_dict(sd, strict=True)
            return True
        except RuntimeError:
            pass

        try:
            sd_pref = {("model." + k): v for k, v in sd.items()}
            net.load_state_dict(sd_pref, strict=True)
            return True
        except RuntimeError:
            pass

        try:
            sd_unpref = {
                k.replace("model.", "", 1): v
                for k, v in sd.items()
                if k.startswith("model.")
            }
            if len(sd_unpref) > 0:
                net.model.load_state_dict(sd_unpref, strict=True)
                return True
        except RuntimeError:
            pass

    missing_ok = (
        "fc.weight",
        "fc.bias",
        "classifier.1.weight",
        "classifier.1.bias",
        "classifier.weight",
        "classifier.bias",
        "head.weight",
        "head.bias",
        "logits.weight",
        "logits.bias",
    )
    try:
        incompatible = net.load_state_dict(sd, strict=False)
        missing = set(incompatible.missing_keys)
        unexpected = set(incompatible.unexpected_keys)
        if len(unexpected) == 0 and (
            len(missing) == 0 or missing.issubset(set(missing_ok))
        ):
            return True
    except Exception:
        pass

    return False


def _infer_model_name_from_basename(basename: str):
    b = basename.lower().replace("_", "").replace("-", "")
    if "resnet18" in b:
        return "resnet18"
    if "resnet50" in b:
        return "resnet50"
    if "resnet152" in b:
        return "resnet152"
    if "resnext101" in b or "resnext10132x8d" in b:
        return "resnext101"
    if "densenet201" in b:
        return "densenet201"
    if "efficientnetb7" in b or "effnetb7" in b:
        return "efficientnet-b7"
    return "unknown"


def _candidate_arches_for_checkpoint(model_name: str):
    supported = [
        "efficientnet-b7",
        "resnet50",
        "resnet18",
        "densenet201",
        "resnext101",
        "resnet152",
    ]
    if model_name in supported:
        return [model_name]
    return supported


MAX_MODELS_TO_USE = 6

pretrained_models = find_model_files(BASE_DIR)
pretrained_models = pretrained_models[:MAX_MODELS_TO_USE]

print(
    f"{len(pretrained_models)} model files selected (filtered, top-{MAX_MODELS_TO_USE})."
)
print("\n".join(np.array(pretrained_models[:50], dtype=str)))
if len(pretrained_models) > 50:
    print(f"... (showing first 50 of {len(pretrained_models)})")

probability = []
start_time = time.time()

loaded_any = 0

use_fallback_pretrained = len(pretrained_models) == 0

if use_fallback_pretrained:
    print(
        "WARNING: No pretrained model files found. Using a single torchvision pretrained efficientnet_b7 as fallback."
    )
    pretrained_models = ["__TORCHVISION_PRETRAINED_EFFICIENTNET_B7__"]

for pretrained_model in pretrained_models:
    if pretrained_model == "__TORCHVISION_PRETRAINED_EFFICIENTNET_B7__":
        basename = "torchvision_pretrained_efficientnet_b7"
        criterion = nn.CrossEntropyLoss()

        net = efficientnet_b7(
            weights=torchvision.models.EfficientNet_B7_Weights.IMAGENET1K_V1
        )

        class ImagenetEffB7Proxy5(nn.Module):
            def __init__(self, model_1000, criterion_):
                super().__init__()
                self.model = model_1000
                self.criterion = criterion_

                self.register_buffer(
                    "_edges",
                    torch.tensor([0, 200, 400, 600, 800, 1000], dtype=torch.long),
                )

            def forward(self, inputs, labels, phase):
                logits1000 = self.model(inputs)  # [B, 1000]
                probs = torch.softmax(logits1000.float(), dim=1)
                out5 = []
                for i in range(5):
                    s = self._edges[i].item()
                    e = self._edges[i + 1].item()
                    out5.append(probs[:, s:e].sum(dim=1, keepdim=True))
                probs5 = torch.cat(out5, dim=1).clamp_min(1e-12)
                logits5 = torch.log(probs5)

                if phase == "val":
                    loss = self.criterion(logits5, labels)
                    return logits5, loss
                if phase == "test":
                    return logits5

                print("ここにきてはいけない")
                sys.exit()

        net = ImagenetEffB7Proxy5(net, criterion)

        BATCH_SIZE = 10
        loaded_any += 1
    else:
        basename = os.path.splitext(os.path.basename(pretrained_model))[0]
        criterion = nn.CrossEntropyLoss()

        parsed = _infer_model_name_from_basename(basename)
        arches_to_try = _candidate_arches_for_checkpoint(parsed)

        state = torch.load(pretrained_model, map_location="cpu")

        net = None
        BATCH_SIZE = None
        used_arch = None

        for MODEL_NAME in arches_to_try:
            if MODEL_NAME == "resnet18":
                cand = models.resnet18(weights=None)
                cand = FinalLayerMixupModel(cand, criterion, num_classes, False)
                bs = 64
            elif MODEL_NAME == "resnet50":
                cand = models.resnet50(weights=None)
                cand = FinalLayerMixupModel(cand, criterion, num_classes, False)
                bs = 32
            elif MODEL_NAME == "resnet152":
                cand = models.resnet152(weights=None)
                cand = FinalLayerMixupModel(cand, criterion, num_classes, False)
                bs = 16
            elif MODEL_NAME == "resnext101":
                cand = models.resnext101_32x8d(weights=None)
                cand = FinalLayerMixupModel(cand, criterion, num_classes, False)
                bs = 12
            elif MODEL_NAME == "densenet201":
                cand = models.densenet201(weights=None)
                cand = FinalLayerMixupModelDenseNet(cand, criterion, num_classes, False)
                bs = 12
            elif MODEL_NAME == "efficientnet-b7":
                cand = efficientnet_b7(weights=None)
                cand = FinalLayerMixupModelEN(cand, criterion, num_classes, False)
                bs = 10
            else:
                continue

            ok = _try_load(cand, MODEL_NAME, state)
            if ok:
                net = cand
                BATCH_SIZE = bs
                used_arch = MODEL_NAME
                break
            else:
                del cand
                if device == "cuda":
                    torch.cuda.empty_cache()

        if net is None:
            print(
                f"WARNING: Failed to load weights for {basename} from {pretrained_model}. Skipping."
            )
            continue

        print(f"{basename}: loaded {pretrained_model} as arch={used_arch}")
        loaded_any += 1

    for param in net.parameters():
        param.requires_grad = False

    for tid, transform_ in enumerate(transform["test"]):
        print(f"transform loop={tid}")
        dataset = {"test": TestDataset(df_test, transform=transform_)}
        dataloader = {
            "test": torch.utils.data.DataLoader(
                dataset["test"],
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=2,
                pin_memory=(device == "cuda"),
            )
        }

        proba = predict_model(basename, net, dataloader)
        probability.append(proba)

    del net
    if device == "cuda":
        torch.cuda.empty_cache()

if loaded_any == 0 or len(probability) == 0:
    print(
        "WARNING: No supported models were loaded successfully. Falling back to label=0 baseline."
    )
    df_test["label"] = 0
else:
    proba_stack = np.stack(
        [p.astype(np.float32, copy=False) for p in probability], axis=0
    )
    proba_mean = np.mean(proba_stack, axis=0)
    df_test["mean"] = proba_mean.argmax(axis=1)
    print(f"total time: {time.time() - start_time:.2f}[sec]")



## === cell 17
if "mean" in df_test.columns:
    df_test["label"] = df_test["mean"].astype(int)



## === cell 18
df_test.head()



## === cell 19
df_test[["image_id", "label"]].to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", df_test[["image_id", "label"]].shape)
print(df_test[["image_id", "label"]].head())
