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

0.8946811725596857

# 6. Current score

0.11584

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.11584) has done: 'I make the notebook run end-to-end by (1) removing the unavailable `efficientnet_pytorch` dependency and safely skipping EfficientNet-B7 weights if present, (2) fixing path detection so it always finds the Kaggle `/kaggle/input/cassava-leaf-disease-classification` images, and (3) updating Albumentations `RandomResizedCrop` to the v2 API to stop validation errors. I also make the ensemble aggregation robust (so `probability` always has the expected shape) and ensure a valid `submission.csv` is always written using `sample_submission.csv` for ordering. These are execution/format fixes and should improve score vs “no submission”, without changing the core ensemble/TTA inference logic beyond necessary compatibility.'
- What this solution (achieved 0.11584) has done: 'Your score is extremely low because the notebook is not actually finding/using the intended pretrained weights in your current environment (so predictions degrade to an almost-constant fallback), and because the current “test set” discovery can silently point at the wrong folder when paths differ. I make minimal, execution-safe changes to (1) locate pretrained `.pth` files robustly under the available `/kaggle/input` and `/kaggle/data` trees, (2) always use `sample_submission.csv` to define the exact test ordering while still reading images from the real `test_images/` directory, and (3) load checkpoints more safely by handling common key-prefix patterns (`module.`, `model.`, etc.) without changing the model architecture or inference logic. These changes should move accuracy upward toward your target by ensuring the ensemble is actually applied to the right test set in the right order, without altering the core model/ensemble/TTA semantics.'
- What this solution (achieved 0.11584) has done: 'Your score is low mainly because the notebook is ensembling *every* `.pth` it can find (including unrelated/incorrect checkpoints) and loading them with `strict=False`, which often yields near-random predictions that get averaged together. I make a minimal, score-directed change to only use checkpoints that match this solution’s supported architectures by filename, and I add a lightweight compatibility filter that skips checkpoints whose tensors don’t match the current model’s parameter shapes (instead of partially loading them). This preserves your core inference/ensemble/TTA logic, but prevents “bad models” from poisoning the average, which should move accuracy upward toward your target. I also keep the sample-submission ordering exactly as you already do and still always write a valid `submission.csv`.'
- What this solution (achieved 0.11584) has done: 'Your current score suggests the ensemble is effectively not using the intended trained weights (or is skipping most of them), so predictions collapse toward a near-constant class. To move accuracy upward toward the target with minimal logic change, I (1) load checkpoints more robustly by handling nested keys and common prefix patterns, and (2) avoid partially-loaded models by requiring a high overlap of matching tensor shapes before using a checkpoint (so “bad” checkpoints don’t poison the mean). I also enforce a deterministic, correct test ordering by always building predictions in the exact `sample_submission.csv` order and add a hard check that every test image exists (fail fast instead of silently degrading). The model architectures, TTA set, and averaging logic stay the same; this only fixes weight usage/alignment issues that are currently causing the low score.'
- What this solution (achieved 0.11584) has done: 'Your current score (0.11584) is far below the target (0.89468), so we should improve accuracy substantially while keeping your ensemble/TTA inference logic intact. The biggest issue is that your checkpoint compatibility check is overly strict: it requires ~80% of *all* model parameters to match, which almost always fails because your wrapper replaces the final classifier (`fc`/`classifier`) so those keys/shapes won’t match the checkpoint, causing most weights to be skipped and triggering the near-constant fallback. I relax compatibility to ignore classifier-head keys during overlap computation (still enforcing exact shape matches for the backbone) so real checkpoints load and contribute, and I add a small safeguard for `x.squeeze()` potentially dropping the batch dimension when batch_size==1 (a rare but harmful edge case). These are minimal changes that preserve your architecture, transforms, and averaging semantics, but should move the score sharply upward toward the target by actually using the intended pretrained weights.'
- What this solution (achieved 0.11584) has done: 'Your score is far below the target, so we should improve accuracy while keeping your ensemble/TTA inference logic intact. The main likely cause is that almost all checkpoints are being skipped because `_is_compatible_state_dict` compares overlap against the *entire* backbone key count and also fails when checkpoint keys are prefixed (e.g., `convlayer.*`, `fc.*`) relative to your wrapper, so you end up with a near-fallback prediction. I minimally (1) map common checkpoint keyspaces into your wrapper’s keyspace (e.g., `features.* -> convlayer.*`, `fc.* -> fc.*`, `classifier.* -> fc.*` as appropriate), and (2) fix the compatibility test to measure overlap against the keys actually present in the checkpoint (excluding head keys), so valid weights load instead of being skipped. This keeps architecture, TTA set, averaging, and argmax semantics unchanged, but ensures real trained weights are used, which should move the score upward toward the target band.'
- What this solution (achieved 0.11584) has done: 'Your current score is far below the target, so the most likely issue is that almost all checkpoints are still being skipped due to key remapping/compatibility being too strict or incorrect for torchvision models, leaving you with the fallback labels. I make a minimal, score-directed fix to load checkpoints reliably by (1) trying both the wrapper-keyspace and the original torchvision-keyspace when loading state_dicts, and (2) treating classifier-head keys as ignorable while still enforcing strict shape matching for backbone weights. This keeps your ensemble/TTA, architectures, transforms, and argmax submission semantics unchanged, but should ensure real trained weights actually get used (raising accuracy toward the target). I also add a small safety fix so batch size 1 never breaks concatenation, without changing predictions otherwise.'
- What this solution (achieved 0.11584) has done: 'Your score is far below the target, so we should improve accuracy by ensuring your ensemble actually uses the checkpoints you already have rather than silently skipping them. The main minimal fix is to load the *full* checkpoint with `strict=False` after choosing the best keyspace mapping, instead of pre-filtering out the classifier/head (your wrapper’s `fc` must be loaded to get meaningful logits; otherwise it stays random and accuracy collapses). To keep this safe, we still (a) choose between raw vs remapped keyspaces by backbone match ratio and (b) skip checkpoints with very low backbone compatibility, but we stop discarding the head weights. This preserves your model wrappers, transforms/TTA set, averaging/argmax semantics, and submission ordering, while making weight loading effective.'
- What this solution (achieved 0.11584) has done: 'Your score is far below the target, and the most likely reason is that most checkpoints are being (wrongly) rejected by the backbone-compatibility filter, so you effectively submit a near-constant fallback. I make a minimal, score-directed change to the checkpoint loading: (1) load the checkpoint with `weights_only=True` when available (safer, more consistent), (2) improve key handling by trying additional common prefixes (especially `encoder.`) and by selecting the best among multiple candidate “keyspace mappings” based on backbone match, and (3) relax the skip threshold slightly (0.80 → 0.60) while still requiring meaningful backbone matches so we use real weights instead of random heads. These changes preserve your models, transforms/TTA, ensembling-by-mean, and argmax submission semantics, but should make the ensemble actually use the intended trained weights, moving accuracy up toward your target.'
- What this solution (achieved 0.11584) has done: 'Your current score (0.11584) is far below the target (0.89468), so we should improve accuracy by ensuring the ensemble actually uses the intended checkpoints and that the loaded classifier head produces meaningful logits. I make minimal, score-directed changes to the checkpoint loading: (1) try additional common keyspaces (including direct torchvision model keys like `features.*`/`classifier.*` and resnet `layer*`/`fc.*`) instead of forcing everything into the wrapper space, and (2) load into the *base* torchvision model first, then wrap it—this preserves your exact architecture/wrapper logic while avoiding key-mismatch issues that currently cause random heads and near-constant predictions. I also fix a small but impactful bug in your resnet wrapper remapping (`"fc." -> "fc.fc."`) and keep the same TTA/transforms, averaging, and submission ordering. These changes are designed to be the smallest reliable way to move accuracy sharply upward toward your target without changing the inference semantics.'
- What this solution (achieved 0.11584) has done: 'Your current score (0.11584) is far below the target, so the most likely issue is still ineffective checkpoint loading: you load weights into the *base* torchvision model, but your selection routine sometimes chooses a “remapped->wrapper” state_dict that can never match the base model’s keys—leading to near-random predictions that get averaged. I make a minimal, score-directed fix by evaluating compatibility in the correct keyspace: compare “raw” weights against the base model, and compare “remapped->wrapper” weights against the wrapper model, then load into whichever model that keyspace belongs to. This preserves your core ensemble/TTA inference and architectures, but ensures real trained weights are actually used rather than silently producing garbage logits. I also keep the submission ordering strictly from `sample_submission.csv` as you already do.'
- What this solution (achieved 0.11584) has done: 'Your current score is far below the target, so we should improve accuracy by ensuring inference uses *valid, correctly-loaded checkpoints* and that the model wrapper produces meaningful logits. The main minimal fix is in `_remap_state_dict_to_wrapper_space` for ResNet-like models: it currently maps backbone keys under `convlayer.0.*`, but your wrapper’s `convlayer` is a `Sequential` of the original children, so `conv1/bn1/layer*` should map to `convlayer.{index}.*` (e.g., `conv1 -> convlayer.0`, `bn1 -> convlayer.1`, `layer1 -> convlayer.4`, etc.). With the correct remap, wrapper-space loading becomes truly compatible, so checkpoints won’t be “loaded” into the wrong submodule structure (which effectively yields random features and near-constant predictions). This preserves your ensemble/TTA and averaging/argmax semantics; it only fixes weight-key alignment so the existing logic can actually work.'
- What this solution (achieved 0.11584) has done: 'Your score is far below the target, so we should raise accuracy by ensuring the ensemble uses correctly ordered test images and that TTA doesn’t introduce randomness at inference. I make two minimal, score-directed fixes: (1) replace the `RandomResizedCrop` test-time transforms with deterministic `Resize`+`CenterCrop` variants (preserving the same “4 TTA passes” structure but removing stochasticity that hurts accuracy), and (2) enforce that predictions are aligned to `sample_submission.csv` order by building `df_test` directly from it (and removing the dummy-label duplication edge case). Core model wrappers, checkpoint loading logic, averaging/argmax semantics, and submission writing remain unchanged.'
- What this solution (achieved 0.11584) has done: 'Your current score is far below the target, so we should make the smallest change that plausibly fixes the “near-constant / wrong logits” failure mode without changing your ensemble/TTA or model definitions. The most likely remaining issue is that for many checkpoints you load weights into the *base* model, but your wrapper replaces `fc`/`classifier` with a fresh random head, so logits are essentially random even if the backbone is good. I minimally change the loading so that when we choose the base(raw) keyspace we also copy the checkpoint’s `fc.*` (or `classifier.*` for DenseNet) into the wrapper head when shapes match; when we choose wrapper(remapped), head weights already map to `fc.*` so they load. This keeps the same architectures, transforms, averaging, and argmax submission semantics, but ensures the trained classifier head is actually used, which should move accuracy strongly upward toward your target.'
- What this solution (achieved 0.11584) has done: 'Your score is far below the target, so we should make the smallest change that plausibly fixes a “wrong logits / random head” failure without changing your ensemble/TTA structure. Right now, when loading weights into the base model, the wrapper’s final `fc` is newly initialized unless the checkpoint head is copied—and your copy routine only works if the checkpoint head is already 5-class, which is often not true for Imagenet-pretrained (1000-class) checkpoints that might be present and currently poisoning the ensemble. I (1) filter out checkpoints whose head shape doesn’t match `num_classes` (so they’re skipped rather than averaged), and (2) additionally require a higher backbone match ratio for any checkpoint whose head cannot be loaded, preventing random heads from contributing. This keeps your models, TTA transforms, averaging and argmax semantics identical, but should move accuracy sharply upward toward the target by only ensembling checkpoints that can actually produce meaningful 5-class logits.'

# 9. Code solution

## === cell 0
import numpy as np
import glob



## === cell 1
import os

CANDIDATE_WEIGHT_GLOBS = [
    "../input/densenet201-04-2019data/*.pth",
    "../input/eb7m-seed70/*.pth",
    "/kaggle/input/**/**/*.pth",
    "/kaggle/input/**/*.pth",
    "/kaggle/data/**/**/*.pth",
    "/kaggle/data/**/*.pth",
]

SUPPORTED_NAME_PATTERNS = [
    "resnet18",
    "resnet50",
    "resnet152",
    "resnext101",
    "densenet201",
    "efficientnet-b7",  # still skipped later if efficientnet_pytorch is unavailable
]

pretrained_models_all = []
for g in CANDIDATE_WEIGHT_GLOBS:
    pretrained_models_all.extend(glob.glob(g, recursive=True))
pretrained_models_all = sorted(list(dict.fromkeys(pretrained_models_all)))


def _is_supported_weight_path(p: str) -> bool:
    bn = os.path.basename(p).lower()
    return any(pat in bn for pat in SUPPORTED_NAME_PATTERNS)


pretrained_models = [p for p in pretrained_models_all if _is_supported_weight_path(p)]

print(
    f"{len(pretrained_models)} supported-model candidates found (from {len(pretrained_models_all)} .pth files)."
)
if len(pretrained_models) > 0:
    print("\n".join(np.sort(pretrained_models)[:50]))
    if len(pretrained_models) > 50:
        print("... (truncated)")
else:
    print(
        "No supported pretrained models found in the specified folders. "
        "The notebook will still produce a valid submission (fallback)."
    )



## === cell 2
import pandas as pd

import torch
import torch.nn as nn
import torch.utils.data as data

import torchvision
from torchvision import models, transforms
import albumentations as A
from albumentations import Compose
from albumentations.pytorch import ToTensorV2

from pathlib import Path
import random
import json
import time
import pickle


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
    torch.backends.cudnn.benchmark = True


SEED = 42
seed_everything(seed=SEED)




## === cell 3
class _EfficientNetStub:
    @staticmethod
    def from_name(name: str):
        raise ModuleNotFoundError(
            "efficientnet_pytorch is not installed in this environment; "
            "EfficientNet weights will be skipped."
        )


EfficientNet = _EfficientNetStub



## === cell 4
SIZE = 512  # image size
num_classes = 5



## === cell 5
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用デバイス: {device}")



## === cell 6
CANDIDATE_BASE_DIRS = [
    "../input/cassava-leaf-disease-classification",
    "/kaggle/input/cassava-leaf-disease-classification",
    "/kaggle/data/cassava-leaf-disease-classification",
    "/kaggle/data/input/cassava-leaf-disease-classification",
    "/kaggle/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
    "/kaggle/data/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
]
BASE_DIR = None
for p in CANDIDATE_BASE_DIRS:
    if os.path.exists(p):
        BASE_DIR = p
        break
if BASE_DIR is None:
    BASE_DIR = "/kaggle/data/cassava-leaf-disease-classification"

TEST_PATH = None
for cand in [
    f"{BASE_DIR}/test_images",
    f"{BASE_DIR}/cassava-leaf-disease-classification/test_images",
    "/kaggle/input/cassava-leaf-disease-classification/test_images",
    "/kaggle/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification/test_images",
    "/kaggle/data/cassava-leaf-disease-classification/test_images",
    "/kaggle/data/cassava-leaf-disease-classification/cassava-leaf-disease-classification/test_images",
]:
    if os.path.isdir(cand):
        TEST_PATH = cand
        break

if TEST_PATH is None:
    TEST_PATH = f"{BASE_DIR}/train_images"

sample_sub_path = None
for cand in [
    f"{BASE_DIR}/sample_submission.csv",
    f"{BASE_DIR}/cassava-leaf-disease-classification/sample_submission.csv",
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv",
    "/kaggle/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification/sample_submission.csv",
    "/kaggle/data/cassava-leaf-disease-classification/sample_submission.csv",
]:
    if os.path.isfile(cand):
        sample_sub_path = cand
        break
if sample_sub_path is None:
    raise FileNotFoundError(
        "sample_submission.csv was not found in candidate locations."
    )

sample_sub = pd.read_csv(sample_sub_path)
test_files = sample_sub["image_id"].tolist()

print(f"BASE_DIR: {BASE_DIR}")
print(f"TEST_PATH: {TEST_PATH}")
print(f"sample_sub_path: {sample_sub_path}")
print(f"Number of test images from sample_submission: {len(test_files)}")

_missing = [
    img for img in test_files[:50] if not os.path.isfile(os.path.join(TEST_PATH, img))
]
if len(_missing) > 0:
    raise FileNotFoundError(
        f"TEST_PATH does not contain expected test images (e.g., { _missing[0] }). TEST_PATH={TEST_PATH}"
    )



## === cell 7
df_test = sample_sub[["image_id"]].copy()
df_test["label"] = 1



## === cell 8
pass



## === cell 9
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
                A.Resize(height=SIZE, width=SIZE),
                A.HorizontalFlip(p=1.0),
                A.CenterCrop(height=SIZE, width=SIZE),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                A.Resize(height=int(SIZE * 1.15), width=int(SIZE * 1.15)),
                A.CenterCrop(height=SIZE, width=SIZE),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                A.Resize(height=int(SIZE * 1.15), width=int(SIZE * 1.15)),
                A.HorizontalFlip(p=1.0),
                A.CenterCrop(height=SIZE, width=SIZE),
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
            x = torch.flatten(x, 1)
            outputs = self.fc(x)
            loss = self.criterion(outputs, labels)

            return outputs, loss

        if phase == "test":
            x = self.convlayer(inputs)
            x = torch.flatten(x, 1)
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
        mixed_x = torch.flatten(mixed_x, 1)
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
            x = torch.flatten(x, 1)
            outputs = self.fc(x)
            loss = self.criterion(outputs, labels)

            return outputs, loss

        if phase == "test":
            x = self.convlayer(inputs)
            x = self.AdaptiveAvgPool2d(x)
            x = torch.flatten(x, 1)
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
        mixed_x = torch.flatten(mixed_x, 1)
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
        img_path = f"{TEST_PATH}/{image_id}"
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {img_path}")
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
    """
    basename: 学習済みモデル名
    net     : 学習済みモデル
    """

    model_start_time = time.time()

    net.to(device)
    net.eval()
    torch.set_grad_enabled(False)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    probability = []

    for phase in ["test"]:
        progress = tqdm(dataloader[phase], desc=f"{basename}: ")

        for inputs, image_ids in progress:
            inputs = inputs.to(device)

            outputs = net(inputs, False, "test")
            if outputs.dim() == 1:
                outputs = outputs.unsqueeze(0)
            probability.append(torch.softmax(outputs, dim=1).cpu().numpy())

    print(f"{basename} time: {time.time() - model_start_time:.2f}[sec]")

    return np.concatenate(probability, axis=0)




## === cell 17
def _extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for k in [
            "state_dict",
            "model_state_dict",
            "model",
            "net",
            "weights",
            "params",
        ]:
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
    return ckpt


def _strip_prefix_if_present(
    state_dict, prefixes=("module.", "model.", "net.", "encoder.")
):
    if not isinstance(state_dict, dict) or len(state_dict) == 0:
        return state_dict
    keys = list(state_dict.keys())
    for pref in prefixes:
        if all(k.startswith(pref) for k in keys):
            return {k[len(pref) :]: v for k, v in state_dict.items()}
    return state_dict


def _cleanup_state_dict_keys(state_dict: dict) -> dict:
    if not isinstance(state_dict, dict):
        return state_dict
    cleaned = {}
    for k, v in state_dict.items():
        nk = k
        if nk.startswith("backbone."):
            nk = nk[len("backbone.") :]
        cleaned[nk] = v
    return cleaned


def _remap_state_dict_to_wrapper_space(state: dict, model_name: str) -> dict:
    """
    Why this change improves score toward target:
    - Wrapper backbone is `convlayer = Sequential(*(children()[:-1]))`, so ResNet child modules must be
      mapped to their correct indices (conv1->0, bn1->1, ..., layer4->7, avgpool->8). Without this,
      wrapper-space loading won't populate the intended layers.
    """
    if not isinstance(state, dict) or len(state) == 0:
        return state

    remapped = {}
    for k, v in state.items():
        nk = k

        if nk.startswith("convlayer.") or nk.startswith("fc."):
            remapped[nk] = v
            continue

        if model_name.startswith("densenet"):
            if nk.startswith("features."):
                nk = "convlayer." + nk[len("features.") :]
            elif nk.startswith("classifier."):
                nk = "fc." + nk[len("classifier.") :]
        else:
            if nk.startswith("fc."):
                nk = "fc." + nk[len("fc.") :]
            elif nk.startswith("conv1."):
                nk = "convlayer.0." + nk[len("conv1.") :]
            elif nk.startswith("bn1."):
                nk = "convlayer.1." + nk[len("bn1.") :]
            elif nk.startswith("layer1."):
                nk = "convlayer.4." + nk[len("layer1.") :]
            elif nk.startswith("layer2."):
                nk = "convlayer.5." + nk[len("layer2.") :]
            elif nk.startswith("layer3."):
                nk = "convlayer.6." + nk[len("layer3.") :]
            elif nk.startswith("layer4."):
                nk = "convlayer.7." + nk[len("layer4.") :]
            elif nk.startswith("avgpool."):
                nk = "convlayer.8." + nk[len("avgpool.") :]

        remapped[nk] = v

    return remapped


def _is_head_key(k: str) -> bool:
    k = str(k)
    return (
        k.startswith("fc.")
        or k.startswith("classifier.")
        or k.startswith("model._fc.")
        or k.endswith(".fc.weight")
        or k.endswith(".fc.bias")
        or k.endswith(".classifier.weight")
        or k.endswith(".classifier.bias")
    )


def _backbone_match_ratio(net: nn.Module, state: dict) -> float:
    """
    Ratio computed on checkpoint backbone keys (excluding head), to avoid over-rejecting
    valid checkpoints that don't include the wrapper head.
    """
    if not isinstance(state, dict) or len(state) == 0:
        return 0.0
    model_sd = net.state_dict()

    ckpt_keys = [
        k
        for k, v in state.items()
        if (not _is_head_key(k))
        and (k in model_sd)
        and torch.is_tensor(v)
        and torch.is_tensor(model_sd[k])
    ]
    if len(ckpt_keys) == 0:
        return 0.0

    matched = 0
    for k in ckpt_keys:
        if tuple(state[k].shape) == tuple(model_sd[k].shape):
            matched += 1
    return matched / max(1, len(ckpt_keys))


def _load_ckpt_safely(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _make_base_and_wrapper(MODEL_NAME: str, criterion):
    if MODEL_NAME == "resnet18":
        base = models.resnet18(weights=None)
        wrapper_ctor = lambda m: FinalLayerMixupModel(m, criterion, num_classes, False)
        batch_size = 64
    elif MODEL_NAME == "resnet50":
        base = models.resnet50(weights=None)
        wrapper_ctor = lambda m: FinalLayerMixupModel(m, criterion, num_classes, False)
        batch_size = 32
    elif MODEL_NAME == "resnet152":
        base = models.resnet152(weights=None)
        wrapper_ctor = lambda m: FinalLayerMixupModel(m, criterion, num_classes, False)
        batch_size = 16
    elif MODEL_NAME == "resnext101":
        base = models.resnext101_32x8d(weights=None)
        wrapper_ctor = lambda m: FinalLayerMixupModel(m, criterion, num_classes, False)
        batch_size = 12
    elif MODEL_NAME == "densenet201":
        base = models.densenet201(weights=None)
        wrapper_ctor = lambda m: FinalLayerMixupModelDenseNet(
            m, criterion, num_classes, False
        )
        batch_size = 12
    else:
        return None, None, None
    return base, wrapper_ctor, batch_size


def _head_is_compatible_for_num_classes(
    state_raw: dict, model_name: str, num_classes: int
) -> bool:
    """
    Why this change improves score toward target:
    - Checkpoints with non-5-class heads (e.g., ImageNet 1000-class) would leave our wrapper head random
      and poison the ensemble. Skipping them is a minimal, score-directed quality filter.
    """
    if not isinstance(state_raw, dict) or len(state_raw) == 0:
        return False
    if model_name.startswith("densenet"):
        w_key = "classifier.weight"
    else:
        w_key = "fc.weight"
    if w_key not in state_raw or not torch.is_tensor(state_raw[w_key]):
        return False
    w = state_raw[w_key]
    return (w.dim() == 2) and (w.shape[0] == num_classes)


def _maybe_copy_head_from_raw_ckpt_into_wrapper(
    base_model: nn.Module, wrapper_model: nn.Module, state_raw: dict, model_name: str
) -> bool:
    """
    Why this change improves score toward target:
    - When we load weights into the *base* model, the wrapper still has a freshly initialized `fc`,
      so logits become random even with a good backbone. Copying the checkpoint head into the wrapper
      (only when shapes match) preserves the intended classifier and should raise accuracy.
    """
    if not isinstance(state_raw, dict) or len(state_raw) == 0:
        return False

    copied = False
    with torch.no_grad():
        if model_name.startswith("densenet"):
            w_key, b_key = "classifier.weight", "classifier.bias"
        else:
            w_key, b_key = "fc.weight", "fc.bias"

        if hasattr(wrapper_model, "fc") and isinstance(wrapper_model.fc, nn.Linear):
            if (
                (w_key in state_raw)
                and torch.is_tensor(state_raw[w_key])
                and tuple(state_raw[w_key].shape)
                == tuple(wrapper_model.fc.weight.data.shape)
            ):
                wrapper_model.fc.weight.copy_(
                    state_raw[w_key].to(dtype=wrapper_model.fc.weight.dtype)
                )
                copied = True
            if (
                (b_key in state_raw)
                and torch.is_tensor(state_raw[b_key])
                and tuple(state_raw[b_key].shape)
                == tuple(wrapper_model.fc.bias.data.shape)
            ):
                wrapper_model.fc.bias.copy_(
                    state_raw[b_key].to(dtype=wrapper_model.fc.bias.dtype)
                )
                copied = True

    return copied


probability = []

start_time = time.time()

MIN_BACKBONE_RATIO = 0.60
MIN_BACKBONE_RATIO_NO_HEAD = 0.90  # stricter if we can't load a compatible 5-class head

if len(pretrained_models) == 0:
    print(
        "No pretrained models available; using sample_submission labels as a safe fallback."
    )
else:
    for pretrained_model in pretrained_models:
        basename = os.path.splitext(os.path.basename(pretrained_model))[0].lower()

        criterion = nn.CrossEntropyLoss()

        MODEL_NAME = None
        if "resnet18" in basename:
            MODEL_NAME = "resnet18"
        elif "resnet50" in basename:
            MODEL_NAME = "resnet50"
        elif "resnet152" in basename:
            MODEL_NAME = "resnet152"
        elif "resnext101" in basename:
            MODEL_NAME = "resnext101"
        elif "densenet201" in basename:
            MODEL_NAME = "densenet201"
        elif "efficientnet-b7" in basename:
            print(
                f"{basename}: efficientnet-b7 skipped (efficientnet_pytorch not installed)."
            )
            continue
        else:
            continue

        print(f"{basename}: {MODEL_NAME}  (weights: {pretrained_model})")

        ckpt = _load_ckpt_safely(pretrained_model)
        state_raw = _extract_state_dict(ckpt)
        state_raw = _strip_prefix_if_present(state_raw)
        state_raw = _cleanup_state_dict_keys(state_raw)

        base, wrapper_ctor, BATCH_SIZE = _make_base_and_wrapper(MODEL_NAME, criterion)
        if base is None:
            continue
        net = wrapper_ctor(base)

        state_wrapped = _remap_state_dict_to_wrapper_space(state_raw, MODEL_NAME)

        raw_ratio = _backbone_match_ratio(base, state_raw)
        wrapped_ratio = _backbone_match_ratio(net, state_wrapped)

        head_compatible_raw = _head_is_compatible_for_num_classes(
            state_raw, MODEL_NAME, num_classes
        )

        if max(raw_ratio, wrapped_ratio) < MIN_BACKBONE_RATIO:
            print(
                f"{basename}: skipped (low backbone match; raw={raw_ratio:.3f}, wrapper={wrapped_ratio:.3f})"
            )
            del net
            del base
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

        chosen_space = None
        head_copied = False

        if raw_ratio >= wrapped_ratio:
            chosen_space = "base(raw)"
            base.load_state_dict(state_raw, strict=False)

            if head_compatible_raw:
                head_copied = _maybe_copy_head_from_raw_ckpt_into_wrapper(
                    base, net, state_raw, MODEL_NAME
                )
            else:
                if raw_ratio < MIN_BACKBONE_RATIO_NO_HEAD:
                    print(
                        f"{basename}: skipped (raw backbone ok but head incompatible and raw_ratio<{MIN_BACKBONE_RATIO_NO_HEAD}; raw={raw_ratio:.3f})"
                    )
                    del net
                    del base
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue

            print(
                f"{basename}: chosen load space = {chosen_space} (raw_ratio={raw_ratio:.3f}, wrapper_ratio={wrapped_ratio:.3f}, head_compatible={head_compatible_raw}, head_copied={head_copied})"
            )
        else:
            chosen_space = "wrapper(remapped)"
            head_compatible_wrapped = _head_is_compatible_for_num_classes(
                (
                    {"fc.weight": state_wrapped.get("fc.weight", None)}
                    if isinstance(state_wrapped, dict)
                    else {}
                ),
                "resnet",
                num_classes,
            )
            if (
                not head_compatible_wrapped
                and wrapped_ratio < MIN_BACKBONE_RATIO_NO_HEAD
            ):
                print(
                    f"{basename}: skipped (wrapper backbone ok but head incompatible and wrapper_ratio<{MIN_BACKBONE_RATIO_NO_HEAD}; wrapper={wrapped_ratio:.3f})"
                )
                del net
                del base
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue

            net.load_state_dict(state_wrapped, strict=False)
            print(
                f"{basename}: chosen load space = {chosen_space} (raw_ratio={raw_ratio:.3f}, wrapper_ratio={wrapped_ratio:.3f})"
            )

        for param in net.parameters():
            param.requires_grad = False

        for tid, transform_ in enumerate(transform["test"]):
            dataset = {
                "test": TestDataset(df_test, transform=transform_),
            }
            dataloader = {
                "test": torch.utils.data.DataLoader(
                    dataset["test"],
                    batch_size=BATCH_SIZE,
                    shuffle=False,
                    num_workers=2,
                    pin_memory=(device == "cuda"),
                ),
            }

            proba = predict_model(basename, net, dataloader)
            probability.append(proba)

        del net
        del base
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if len(probability) > 0:
        mean_proba = np.stack(probability, axis=0).mean(axis=0)
        df_test["mean"] = mean_proba.argmax(axis=1)
    else:
        print(
            "All discovered models were skipped/unsupported; will fall back to sample_submission labels."
        )

print(f"total time: {time.time() - start_time:.2f}[sec]")



## === cell 18
if "mean" in df_test.columns:
    pred_map = dict(zip(df_test["image_id"].values, df_test["mean"].astype(int).values))
    sample_sub["label"] = sample_sub["image_id"].map(pred_map).fillna(1).astype(int)
    df_test = sample_sub
else:
    df_test = sample_sub



## === cell 19
df_test.head()



## === cell 20
df_test[["image_id", "label"]].to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", df_test[["image_id", "label"]].shape)
print(df_test[["image_id", "label"]].head())
