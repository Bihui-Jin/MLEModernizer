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

0.11584

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.11584) has done: 'I make the script robust to the current Kaggle environment by removing the missing `efficientnet_pytorch` dependency and by handling the case where no external pretrained `.pth` files are found (so it still produces a valid `submission.csv`). I also fix the dataset path detection so it correctly finds `/kaggle/input/cassava-leaf-disease-classification` (your current `../input/...` path and local fallback are what triggered the `FileNotFoundError`). Finally, I update the Albumentations v2 `RandomResizedCrop` call signature to the new API (it currently throws a validation error), and I guard the ensemble aggregation so it doesn’t crash when `probability` is empty.'
- What this solution (achieved 0.11584) has done: 'Your current score is low mainly because the script is inferring on the wrong folder (`train_images` instead of `test_images`) and because `df_test` is built from that directory rather than from `sample_submission.csv`, which can misalign rows/IDs and wreck accuracy. I make the dataset path logic always prefer `test_images` when present, and I build `df_test` from `sample_submission.csv` (canonical test set order) while still reading images from `test_images`. I also make checkpoint loading robust to common wrappers (`state_dict`, `module.` prefixes) so your available `.pth` files actually load, which should raise accuracy toward your target without changing the model architecture or inference approach. Finally, I ensure we always write a valid `submission.csv` with exactly the required columns and row count.'
- What this solution (achieved 0.11584) has done: 'Your score is far below target, so we should increase accuracy with minimal-risk fixes that don’t change the model architecture or training/inference approach. The biggest likely cause is that the model heads are randomly initialized because the wrapper replaces the final classifier but we never restore it from the checkpoint (common when checkpoints were saved from the base model, not the wrapper), so predictions become near-random. I modify checkpoint loading to (1) load into the underlying base model first when possible and (2) remap common key patterns so the final-layer weights land in `fc`/`classifier`/`model._fc` correctly, while keeping `strict=False` as before. I also ensure inference uses `torch.inference_mode()` for correctness/speed and keep the submission aligned to `sample_submission.csv` order.'
- What this solution (achieved 0.11584) has done: 'Your current score is far below target, so we should increase accuracy by fixing the most likely root cause: most checkpoints are not actually loading into the wrapper models’ final classifier (`fc`) because the saved keys typically target the base model’s `fc`/`classifier`, leaving your wrapper head randomly initialized (near-random predictions). I add a small, architecture-preserving key-remap step that maps common checkpoint keys (`fc.*`, `classifier.*`, `_fc.*`) into the wrapper’s expected names (`fc.*` or `model._fc.*`) before calling `load_state_dict`. I also ensure the base torchvision models have their final layers sized to `num_classes` before wrapping (does not change the wrapper logic, but makes weight shapes match and improves load success). These are minimal changes focused on making inference use the learned heads, which should move accuracy toward your target.'
- What this solution (achieved 0.11584) has done: 'Your current score is far below target, so we should increase accuracy by fixing the most likely reason predictions are effectively random: the wrapper classes (`FinalLayerMixupModel*`) create a *new* `fc` head that often does not get checkpoint weights because the checkpoint keys typically target the original base model head (`model.fc`, `classifier`, `_fc`) with different prefixes. I keep the architecture and inference/TTA logic the same, but make checkpoint loading reliably map common head keys into the wrapper’s `fc` (and `model._fc` for EfficientNet) and also ensure we don’t incorrectly load the base model in a way that bypasses the wrapper head. Additionally, I fix a small but important tensor-shape issue (`squeeze()` can drop the batch dimension when batch_size=1) to avoid silent misbehavior during inference, without changing the model computation for normal batch sizes. These are minimal, targeted changes that should move accuracy toward your target while preserving your overall approach and submission format.'
- What this solution (achieved 0.11584) has done: 'Your score is far below the target (0.11584 vs 0.8963), so the smallest high-impact fix is to ensure the model you load from each `.pth` checkpoint matches the checkpoint’s expected head and that head weights actually get loaded. Right now you replace the base model’s `fc/classifier` **before** wrapping it, then the wrapper creates a *new* `fc`, which often prevents the checkpoint head weights from landing where they should—leading to near-random predictions. I keep your architecture/wrapper and TTA/inference the same, but (1) stop pre-replacing the base head and (2) add a tiny, safe key-remap that copies checkpoint head weights into the wrapper head (`fc.*`) when shapes match. This should materially increase accuracy while preserving your core logic and still producing a valid `submission.csv`.'
- What this solution (achieved 0.11584) has done: 'The current score is far below the target, so we should increase accuracy by fixing the most likely “near-random prediction” cause while keeping your overall model/TTA/inference logic intact. The smallest high-impact change is to ensure the wrapper head (`net.fc`) actually receives the checkpoint’s classifier weights: after loading the checkpoint, we also (safely) copy matching weights from common checkpoint head keys into `net.fc` when shapes match. Additionally, we fix the `seed_everything` determinism settings (your code currently sets `deterministic=True` but also `benchmark=True`, which can introduce nondeterministic kernels) to make results stable without changing semantics. These are targeted changes that should move the score upward toward your target while still producing the same `submission.csv` format.'
- What this solution (achieved 0.11584) has done: 'Your score (0.11584) is far below the target (0.89634), so we should increase accuracy by fixing the most likely “near-random predictions” cause while keeping your model/TTA/inference core logic intact. The biggest issue is that your wrappers always create a fresh `fc` head, but the checkpoint head weights often live under different key names (e.g., `classifier.*` for DenseNet, `model.classifier.*`, etc.) and are not reliably mapped into the wrapper’s `fc`, leaving the head randomly initialized. I make a minimal, shape-safe head-key remap that is aware of which wrapper we’re using (ResNet-wrapper vs DenseNet-wrapper vs EfficientNet-wrapper), and I add a small sanity check to warn if the wrapper `fc` remains at its initial values after loading. These changes are targeted to make checkpoints actually load learned classifier weights and should move accuracy upward toward your target without changing architecture or inference semantics.'
- What this solution (achieved 0.11584) has done: 'Your score is far below the target, so the most likely minimal high-impact fix is that you are not actually using the trained weights at all (your `glob()` paths don’t exist in this environment, so `pretrained_models` is empty and you submit the default labels from `sample_submission.csv`, which matches the low score). I keep your model wrappers, TTA list, and inference logic the same, but change only the checkpoint discovery so it searches common Kaggle input locations (including the competition dataset folder) for `.pth` files and uses them if present. I also add a tiny safety check to skip corrupt/unloadable checkpoints instead of silently falling back to all-zeros, and keep the submission aligned to `sample_submission.csv`. These changes are directly aimed at loading real learned weights so accuracy moves toward your target.'
- What this solution (achieved 0.11584) has done: 'The current score suggests you are still effectively submitting near-random predictions, which most commonly happens here when the checkpoint classifier head weights don’t land in the wrapper’s `fc`/`model._fc` due to name mismatches (especially for ResNet where checkpoints often store `fc.*` but your wrapper expects `fc.*` while other keys may be `model.fc.*` or vice versa). I keep your exact model wrappers, TTA list, and inference loop, but make checkpoint loading more robust by (1) trying a few wrapper-aware key maps (including mapping base-model `fc.*` into wrapper `fc.*`, and also mapping `model.fc.*`/`backbone.fc.*` patterns), and (2) reporting/guarding against the case where the loaded state_dict doesn’t change the head at all. I also fix a subtle transform issue: `CenterCrop(512,512)` can fail/over-crop for smaller images; replacing it with `Resize(SIZE,SIZE)` for the “no-augmentation” TTA keeps the same semantics (test-time preprocessing) but avoids producing degenerate crops that can tank accuracy. These are minimal changes aimed directly at making inference use the trained weights and sane preprocessing, which should move accuracy upward toward your target.'
- What this solution (achieved 0.11584) has done: 'Your score is far below target, so we should focus on the smallest high-impact fixes that make your ensemble actually use meaningful, *stable* test-time predictions. The biggest issue is that your TTA list includes random augmentations (`RandomResizedCrop`, `Rotate`) which injects randomness at inference and can destroy accuracy when ensembling; we keep TTA/ensemble logic but make those transforms deterministic (center crop / fixed rotate angles). Next, checkpoint discovery currently picks up many irrelevant `.pth` files from other datasets, so we restrict/priority-filter to cassava-related paths/names to ensure we load the intended models (or fewer bad ones), which should move accuracy up toward your target. Finally, we make checkpoint selection deterministic and skip obviously incompatible checkpoints (e.g., wrong classifier shape) to avoid poisoning the mean-probability ensemble.'
- What this solution (achieved 0.11584) has done: 'Your current score is far below the target, so we should increase accuracy by fixing the most likely “near-random predictions” causes without changing your model wrappers, loss, or inference loop. The two minimal high-impact fixes here are (1) ensure checkpoint discovery is restricted to the cassava dataset/input (your current broad glob can pull unrelated `.pth` files and poison the ensemble), and (2) make TTA deterministic/valid for all images by removing `CenterCrop(512,512)` after `SmallestMaxSize` (it can fail or overly crop depending on image size/aspect), replacing it with a simple `Resize` (same evaluation semantics: test preprocessing). Finally, we keep the exact ensemble averaging, but we also ensure we only average over successful model+TTA predictions and that the submission is always aligned to `sample_submission.csv` order (already mostly true).'
- What this solution (achieved 0.11584) has done: 'Your score is far below the target, so the smallest high-impact change is to stop (a) averaging in random/over-aggressive TTAs that can destroy accuracy and (b) ensembling across potentially many weak/irrelevant checkpoints that “poison” the mean. I keep your exact model wrappers and inference loop, but make TTA deterministic and conservative (resize + optional horizontal flip only), and I prioritize/limit checkpoint loading to the best-looking cassava checkpoints (those with an obvious 5-class head), instead of blindly using everything found. This keeps the core logic the same (same models, same averaging), but removes the two main failure modes that commonly yield near-random 0.11 accuracy. The script still run end-to-end and always write a valid `submission.csv`.'
- What this solution (achieved 0.11584) has done: 'Your current score strongly suggests the submission is effectively random or defaulting to sample labels, so the smallest high-impact improvement is to ensure we actually load *usable* cassava checkpoints and don’t poison the ensemble with unrelated/incorrect ones. I keep your exact model wrappers, TTA list (resize + flip), and averaging logic, but (1) prioritize searching in `/kaggle/input` for non-competition “dataset” inputs that typically contain trained `.pth` (while still allowing the old search), and (2) validate candidate checkpoints by instantiating the inferred model and verifying that `load_state_dict` changes the head weights and produces correctly-shaped outputs on a tiny dummy batch. This stays within your current inference approach, but prevents averaging in broken/unmatched checkpoints (a common cause of ~0.11 accuracy). The script still always produce a valid `submission.csv` aligned to `sample_submission.csv`.'

# 9. Code solution

## === cell 0
import numpy as np
import glob




## === cell 1
def _discover_checkpoints():
    preferred_roots = [
        "/kaggle/input/**/cassava*",
        "/kaggle/input/**/leaf*",
        "/kaggle/input/**/cdd*",
        "/kaggle/input/**/weights*",
        "/kaggle/input/**/model*",
        "/kaggle/input/cassava-leaf-disease-classification",
        "../input/cassava-leaf-disease-classification",
        "../input/**/cassava*",
    ]

    patterns = []
    for root in preferred_roots:
        patterns += [
            f"{root}/*.pth",
            f"{root}/*/*.pth",
            f"{root}/*/*/*.pth",
            f"{root}/*/*/*/*.pth",
            f"{root}/*/*/*/*/*.pth",
        ]

    files = []
    for pat in patterns:
        files.extend(glob.glob(pat, recursive=True))

    if len(files) == 0:
        files = glob.glob("/kaggle/input/**/*.pth", recursive=True) + glob.glob(
            "../input/**/*.pth", recursive=True
        )

    keep_tokens = (
        "cassava",
        "leaf",
        "cdd",
        "resnet",
        "resnext",
        "densenet",
        "efficientnet",
        "seresnext",
        "vit",
    )
    out = []
    seen = set()
    for f in files:
        bn = f.lower()
        base = bn.split("/")[-1]
        if not any(t in bn for t in ("cassava", "leaf", "cdd")) and not any(
            t in base for t in keep_tokens
        ):
            continue
        if f not in seen:
            seen.add(f)
            out.append(f)

    out = sorted(out)
    return out


pretrained_models = _discover_checkpoints()

print(f"{len(pretrained_models)} models found.")
if len(pretrained_models) > 0:
    for p in np.sort(pretrained_models)[:50]:
        print(p)
    if len(pretrained_models) > 50:
        print(f"... ({len(pretrained_models)-50} more)")



## === cell 2
import pandas as pd

import torch
import torch.nn as nn
import torch.utils.data as data

from torchvision import models
import albumentations as A
from albumentations import Compose
from albumentations.pytorch import ToTensorV2

import os
from pathlib import Path
import random
import time
import sys

from tqdm import tqdm
import cv2


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


SEED = 42
seed_everything(seed=SEED)



## === cell 3
try:
    from efficientnet_pytorch import EfficientNet  # type: ignore
except Exception as e:
    EfficientNet = None
    print(
        f"efficientnet_pytorch not available; EfficientNet models will be skipped. ({type(e).__name__}: {e})"
    )



## === cell 4
SIZE = 512
num_classes = 5



## === cell 5
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用デバイス: {device}")




## === cell 6
def _pick_base_dir():
    candidates = [
        "/kaggle/input/cassava-leaf-disease-classification",
        "/kaggle/data/input/cassava-leaf-disease-classification",
        "/kaggle/data/cassava-leaf-disease-classification",
        "../input/cassava-leaf-disease-classification",
        "data/cassava-leaf-disease-classification",
        "data",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return candidates[0]


BASE_DIR = _pick_base_dir()

run_type = os.getenv("KAGGLE_KERNEL_RUN_TYPE", "")

TEST_IMAGES_CANDIDATES = [
    f"{BASE_DIR}/test_images",
    "/kaggle/input/cassava-leaf-disease-classification/test_images",
    "/kaggle/data/input/cassava-leaf-disease-classification/test_images",
]
TRAIN_IMAGES_CANDIDATES = [
    f"{BASE_DIR}/train_images",
    "/kaggle/input/cassava-leaf-disease-classification/train_images",
    "/kaggle/data/input/cassava-leaf-disease-classification/train_images",
]

TEST_PATH = None
for p in TEST_IMAGES_CANDIDATES:
    if os.path.exists(p):
        TEST_PATH = p
        break
if TEST_PATH is None:
    for p in TRAIN_IMAGES_CANDIDATES:
        if os.path.exists(p):
            TEST_PATH = p
            break

if TEST_PATH is None:
    raise FileNotFoundError(
        f"Could not find test_images/train_images under BASE_DIR={BASE_DIR}"
    )

print(f"BASE_DIR: {BASE_DIR}")
print(f"TEST_PATH: {TEST_PATH}")

if run_type == "Interactive":
    print("Test run in Kaggle environment (Interactive).")
else:
    print(f"Run type: {run_type or 'Unknown/Local'}")




## === cell 7
def _pick_sample_submission():
    candidates = [
        f"{BASE_DIR}/sample_submission.csv",
        "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv",
        "/kaggle/data/input/cassava-leaf-disease-classification/sample_submission.csv",
        "/kaggle/data/sample_submission.csv",
        "data/sample_submission.csv",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("sample_submission.csv not found in expected locations.")


sample_path = _pick_sample_submission()
df_test = pd.read_csv(sample_path)

if "image_id" not in df_test.columns:
    raise ValueError(
        f"sample_submission missing image_id column: columns={df_test.columns}"
    )

if "label" not in df_test.columns:
    df_test["label"] = 0

missing = 0
for iid in df_test["image_id"].head(50).tolist():
    if not os.path.exists(f"{TEST_PATH}/{iid}"):
        missing += 1
print(
    f"Loaded df_test from: {sample_path}, rows={len(df_test)} (first-50 missing images={missing})"
)

if run_type == "Interactive":
    df_test = df_test.iloc[:32].copy()
    print("Interactive: truncating df_test to 32 rows for speed.")
print(f"Number of test images (df_test): {len(df_test)}")



## === cell 8
if len(df_test) == 1:
    df_test.loc[1] = df_test.loc[0]
    print(df_test)



## === cell 9
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

transform = {
    "test": [
        Compose(
            [
                A.Resize(height=SIZE, width=SIZE),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                A.HorizontalFlip(p=1.0),
                A.Resize(height=SIZE, width=SIZE),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
    ]
}



## === cell 10
pass




## === cell 11
class FinalLayerMixupModel(nn.Module):
    def __init__(self, model, criterion, num_classes, alpha):
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
        if alpha and alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        index = torch.randperm(len(labels), device=labels.device)

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




## === cell 12
class FinalLayerMixupModelDenseNet(nn.Module):
    def __init__(self, model, criterion, num_classes, alpha):
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
        if alpha and alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        index = torch.randperm(len(labels), device=labels.device)

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




## === cell 13
class FinalLayerMixupModelEN(nn.Module):
    def __init__(self, model, criterion, num_classes, alpha):
        super(FinalLayerMixupModelEN, self).__init__()
        num_ftrs = model._fc.in_features
        model._fc = nn.Linear(num_ftrs, num_classes)
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




## === cell 14
pass




## === cell 15
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




## === cell 16
def predict_model(basename, net, dataloader):
    model_start_time = time.time()

    net.to(device)
    net.eval()

    probability = []
    with torch.inference_mode():
        for phase in ["test"]:
            progress = tqdm(dataloader[phase], desc=f"{basename}: ")
            for inputs, image_ids in progress:
                inputs = inputs.to(device)
                outputs = net(inputs, False, "test")
                probability.append(torch.softmax(outputs, dim=1).cpu().numpy())

    print(f"{basename} time: {time.time() - model_start_time:.2f}[sec]")
    return np.concatenate(probability, axis=0)




## === cell 17
def _extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for k in ["state_dict", "model_state_dict", "model", "net"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
    return ckpt


def _strip_module_prefix(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict
    keys = list(state_dict.keys())
    if len(keys) == 0:
        return state_dict
    if all(k.startswith("module.") for k in keys):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict


def _normalize_prefixes(state):
    if not isinstance(state, dict):
        return state
    out = {}
    for k, v in state.items():
        nk = k
        for pref in ("model.", "net."):
            if nk.startswith(pref):
                nk = nk.replace(pref, "", 1)
        out[nk] = v
    return out


def _remap_head_keys_for_wrapper(net, state):
    if not isinstance(state, dict):
        return state

    net_sd = net.state_dict()
    mapped = dict(state)

    def _copy(dst_key, src_key):
        if src_key not in state or dst_key not in net_sd:
            return False
        try:
            if tuple(state[src_key].shape) != tuple(net_sd[dst_key].shape):
                return False
        except Exception:
            return False
        mapped[dst_key] = state[src_key]
        return True

    if hasattr(net, "fc") and "fc.weight" in net_sd:
        for src_pref in [
            "fc",
            "model.fc",
            "net.fc",
            "backbone.fc",
            "encoder.fc",
            "module.fc",
            "module.model.fc",
            "module.backbone.fc",
        ]:
            _copy("fc.weight", f"{src_pref}.weight")
            _copy("fc.bias", f"{src_pref}.bias")

        for src_pref in [
            "classifier",
            "model.classifier",
            "net.classifier",
            "backbone.classifier",
            "encoder.classifier",
            "module.classifier",
            "module.model.classifier",
            "module.backbone.classifier",
        ]:
            _copy("fc.weight", f"{src_pref}.weight")
            _copy("fc.bias", f"{src_pref}.bias")

    if hasattr(net, "model") and hasattr(getattr(net, "model"), "_fc"):
        for src_pref in [
            "_fc",
            "model._fc",
            "net._fc",
            "backbone._fc",
            "module._fc",
            "module.model._fc",
            "module.backbone._fc",
        ]:
            _copy("model._fc.weight", f"{src_pref}.weight")
            _copy("model._fc.bias", f"{src_pref}.bias")

    return mapped


def _try_load_state_dict(net, state):
    attempts = []
    attempts.append(state)

    if isinstance(state, dict):
        attempts.append(_normalize_prefixes(state))

        st_wrap = {}
        for k, v in state.items():
            nk = k
            if nk.startswith("_fc."):
                nk = nk.replace("_fc.", "model._fc.", 1)
            st_wrap[nk] = v
        attempts.append(st_wrap)

        st_dense = {}
        for k, v in state.items():
            nk = k
            if nk.startswith("classifier."):
                nk = nk.replace("classifier.", "fc.", 1)
            st_dense[nk] = v
        attempts.append(st_dense)

        st_model_fc = dict(state)
        if "model.fc.weight" in st_model_fc and "fc.weight" not in st_model_fc:
            st_model_fc["fc.weight"] = st_model_fc["model.fc.weight"]
        if "model.fc.bias" in st_model_fc and "fc.bias" not in st_model_fc:
            st_model_fc["fc.bias"] = st_model_fc["model.fc.bias"]
        attempts.append(st_model_fc)

    best_msg = None
    for i, st in enumerate(attempts):
        try:
            missing, unexpected = net.load_state_dict(st, strict=False)
            best_msg = f"load attempt {i}: strict=False (missing={len(missing)}, unexpected={len(unexpected)})"
            return missing, unexpected, best_msg
        except Exception as e:
            best_msg = f"load attempt {i} failed: {type(e).__name__}: {e}"
            continue

    raise RuntimeError(best_msg or "Failed to load state_dict")


def _head_sanity(net):
    with torch.no_grad():
        if hasattr(net, "fc") and isinstance(net.fc, nn.Linear):
            w = net.fc.weight.detach().float().cpu()
            return float(w.abs().mean().item())
        if (
            hasattr(net, "model")
            and hasattr(net.model, "_fc")
            and isinstance(net.model._fc, nn.Linear)
        ):
            w = net.model._fc.weight.detach().float().cpu()
            return float(w.abs().mean().item())
    return None


def _ckpt_has_5class_head(state):
    if not isinstance(state, dict):
        return False
    cand_keys = [
        "fc.weight",
        "classifier.weight",
        "model.fc.weight",
        "model.classifier.weight",
        "net.fc.weight",
        "net.classifier.weight",
        "model._fc.weight",
        "_fc.weight",
    ]
    for k in cand_keys:
        if k in state and hasattr(state[k], "shape") and len(state[k].shape) == 2:
            if int(state[k].shape[0]) == num_classes:
                return True
    return False


def _infer_model_name_from_path(pth: str) -> str:
    b = os.path.splitext(os.path.basename(pth))[0].lower()
    if "resnet18" in b:
        return "resnet18"
    if "resnet50" in b:
        return "resnet50"
    if "resnet152" in b:
        return "resnet152"
    if "resnext101" in b or "resnext-101" in b:
        return "resnext101"
    if "densenet201" in b:
        return "densenet201"
    if "efficientnet-b7" in b or "effb7" in b:
        return "efficientnet-b7"
    return ""


def _build_net_for_name(model_name: str, criterion):
    if model_name == "resnet18":
        base = models.resnet18(weights=None)
        net = FinalLayerMixupModel(base, criterion, num_classes, False)
        bs = 64
        return net, bs
    if model_name == "resnet50":
        base = models.resnet50(weights=None)
        net = FinalLayerMixupModel(base, criterion, num_classes, False)
        bs = 32
        return net, bs
    if model_name == "resnet152":
        base = models.resnet152(weights=None)
        net = FinalLayerMixupModel(base, criterion, num_classes, False)
        bs = 16
        return net, bs
    if model_name == "resnext101":
        base = models.resnext101_32x8d(weights=None)
        net = FinalLayerMixupModel(base, criterion, num_classes, False)
        bs = 12
        return net, bs
    if model_name == "densenet201":
        base = models.densenet201(weights=None)
        net = FinalLayerMixupModelDenseNet(base, criterion, num_classes, False)
        bs = 12
        return net, bs
    if model_name == "efficientnet-b7":
        if EfficientNet is None:
            return None, None
        base = EfficientNet.from_name("efficientnet-b7")
        net = FinalLayerMixupModelEN(base, criterion, num_classes, False)
        bs = 10
        return net, bs
    return None, None


def _checkpoint_is_usable(pth: str) -> bool:
    model_name = _infer_model_name_from_path(pth)
    if model_name == "":
        return False
    if model_name.startswith("efficientnet") and EfficientNet is None:
        return False

    try:
        ckpt = torch.load(pth, map_location="cpu")
        state = _strip_module_prefix(_extract_state_dict(ckpt))
        state = _normalize_prefixes(state)
    except Exception:
        return False

    if not _ckpt_has_5class_head(state):
        return False

    criterion = nn.CrossEntropyLoss()
    net, _ = _build_net_for_name(model_name, criterion)
    if net is None:
        return False

    head_before = _head_sanity(net)
    state = _remap_head_keys_for_wrapper(net, state)

    try:
        _try_load_state_dict(net, state)
    except Exception:
        return False

    head_after = _head_sanity(net)
    if head_before is not None and head_after is not None:
        if abs(head_after - head_before) < 1e-10:
            return False

    try:
        net.eval()
        x = torch.zeros(2, 3, SIZE, SIZE)
        with torch.inference_mode():
            out = net(x, False, "test")
        if not (
            isinstance(out, torch.Tensor)
            and out.ndim == 2
            and out.shape[1] == num_classes
        ):
            return False
    except Exception:
        return False

    return True


MAX_MODELS = 6

probability = []
start_time = time.time()

if len(pretrained_models) == 0:
    print(
        "No pretrained model files found; using sample_submission.csv fallback (valid format, score will be low)."
    )
else:
    usable = []
    for pth in pretrained_models:
        if _checkpoint_is_usable(pth):
            bn = pth.lower()
            priority = 0
            if "cassava-leaf-disease-classification" in bn:
                priority += 1
            if "cassava" in bn:
                priority += 2
            if "leaf" in bn:
                priority += 1
            if "weight" in bn or "model" in bn or "checkpoint" in bn:
                priority += 1
            usable.append((priority, pth))

    if len(usable) == 0:
        print(
            "No usable checkpoints found after validation; predictions will remain default."
        )
    else:
        usable = sorted(usable, key=lambda x: (-x[0], x[1]))
        selected_models = [p for _, p in usable[:MAX_MODELS]]
        print(
            f"Selected {len(selected_models)} validated checkpoints (MAX_MODELS={MAX_MODELS})."
        )
        for p in selected_models:
            print("  ", p)

        for pretrained_model in selected_models:
            basename = os.path.splitext(os.path.basename(pretrained_model))[0]
            criterion = nn.CrossEntropyLoss()

            model_name = _infer_model_name_from_path(pretrained_model)
            net, BATCH_SIZE = _build_net_for_name(model_name, criterion)
            if net is None:
                print(
                    f"{basename}: unsupported or unavailable model ({model_name}) -> skipping."
                )
                continue

            print(f"{basename}: {model_name} ({pretrained_model})")

            head_before = _head_sanity(net)

            try:
                ckpt = torch.load(pretrained_model, map_location="cpu")
            except Exception as e:
                print(
                    f"{basename}: failed to torch.load -> skipping. ({type(e).__name__}: {e})"
                )
                continue

            state = _strip_module_prefix(_extract_state_dict(ckpt))
            state = _normalize_prefixes(state)

            state = _remap_head_keys_for_wrapper(net, state)
            if not _ckpt_has_5class_head(state):
                print(
                    f"{basename}: skipped (after remap, no obvious {num_classes}-class head in checkpoint)."
                )
                continue

            try:
                missing, unexpected, msg = _try_load_state_dict(net, state)
            except Exception as e:
                print(
                    f"{basename}: failed to load_state_dict -> skipping. ({type(e).__name__}: {e})"
                )
                continue

            head_after = _head_sanity(net)
            if head_before is not None and head_after is not None:
                if abs(head_after - head_before) < 1e-10:
                    print(
                        f"{basename}: WARNING head weights look unchanged after load (mean|w| before={head_before:.6g}, after={head_after:.6g}). Skipping to avoid poisoning ensemble."
                    )
                    continue

            print(f"{basename}: {msg}")

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
                        pin_memory=torch.cuda.is_available(),
                    )
                }

                proba = predict_model(basename, net, dataloader)
                if (
                    isinstance(proba, np.ndarray)
                    and proba.ndim == 2
                    and proba.shape[0] == len(df_test)
                    and proba.shape[1] == num_classes
                ):
                    probability.append(proba)
                else:
                    print(
                        f"{basename}: skipping this proba due to unexpected shape {getattr(proba, 'shape', None)}"
                    )

            del net
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if len(probability) == 0:
            print(
                "All models were skipped/unavailable; predictions will remain default."
            )
        else:
            probs = np.array(probability)  # (n_models*n_tta, n_samples, n_classes)
            df_test["mean"] = probs.mean(axis=0).argmax(axis=1)

            print(f"total time: {time.time() - start_time:.2f}[sec]")



## === cell 18
if "mean" in df_test.columns:
    df_test["label"] = df_test["mean"].astype(int)
elif "label" in df_test.columns:
    df_test["label"] = df_test["label"].astype(int)
else:
    df_test["label"] = 0



## === cell 19
sub = df_test[["image_id", "label"]].copy()
sub["label"] = sub["label"].astype(int)
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())
