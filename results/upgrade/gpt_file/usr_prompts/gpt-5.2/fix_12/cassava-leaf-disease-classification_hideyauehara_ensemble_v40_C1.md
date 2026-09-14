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

0.8927168328800241

# 6. Current score

0.11584

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.11584) has done: 'I (1) remove the missing `efficientnet_pytorch` dependency by switching the EfficientNet-B7 branch to `torchvision.models.efficientnet_b7` with the same “replace final FC then load weights” intent, (2) make the dataset path detection robust for this environment so `test_images` is always found and `test_files/df_test` get created, and (3) update Albumentations transforms for v2.x (`RandomResizedCrop(size=(H,W))`) to fix the validation error. Finally, I fix the ensemble aggregation so it can’t crash when `pretrained_models` is empty or when probability arrays have unexpected shapes, and always write a valid `submission.csv` with the required columns. These changes keep the original inference/ensemble logic (multi-model + multi-TTA softmax averaging + argmax) while ensuring the notebook runs end-to-end.'
- What this solution (achieved 0.11584) has done: 'Your current 0.11584 is far below the 0.8927 target, and the most likely cause (given this exact code) is that the loaded checkpoints don’t match the model wrappers you’re using: for ResNet you’re replacing the entire `fc` module with a wrapper that introduces a *new* `fc`, so `load_state_dict(strict=True)` either fail or (if the checkpoints were saved from the wrapper) be extremely brittle across architectures; similarly EfficientNet-B7 in torchvision uses a different head naming than many community checkpoints. I keep the same ensemble + multi-TTA softmax averaging + argmax core logic, but change the wrappers so they *don’t re-create* classifier layers during inference; instead, we load the checkpoint into the original torchvision model, then wrap only for the `forward(inputs, labels, phase)` API. I also add a small, safe state-dict “key normalization” that remaps common checkpoint prefixes (`module.`, `model.`) so your checkpoints actually load, and I ensure the transform uses a deterministic resize+centercrop as the first TTA (reduces randomness at inference without changing the overall TTA approach). These are minimal inference-side fixes that should move accuracy sharply upward toward your target if those pretrained weights are valid for these backbones.'
- What this solution (achieved 0.11584) has done: 'Your low score (0.11584) strongly suggests the ensemble is effectively producing near-random predictions because the `.pth` checkpoints are not being loaded into the exact parameter key structure your instantiated torchvision models expect. I keep the same model backbones, same multi-model + multi-TTA softmax averaging + argmax core logic, but make checkpoint loading robust by (1) extracting the real `state_dict` from common checkpoint formats, (2) normalizing/remapping common key patterns (including `module.`/`model.` prefixes and `fc.` vs `model.fc.` wrapper artifacts), and (3) falling back to a safe non-strict load only when strict loading fails, while printing missing/unexpected keys for visibility. This is a minimal inference-side change that should move accuracy sharply upward toward your target if the provided pretrained weights are valid for these architectures. I also ensure the final submission always uses the official `sample_submission.csv` ordering for `image_id` so there is no accidental row-order mismatch.'
- What this solution (achieved 0.11584) has done: 'Your score (0.11584) is far below the target (0.8927), which strongly suggests the checkpoints are not being loaded into the exact parameter-key structure your instantiated torchvision models expect, causing near-random predictions. I keep the exact same ensemble + multi-TTA softmax averaging + argmax core logic, but make weight loading robust to common wrapper-induced key mismatches (especially `model.fc.*` vs `fc.*`, and `model.classifier.*` vs `classifier.*`) without changing architectures. I also force the final `df_test` ordering to match `sample_submission.csv` (to avoid any accidental row misalignment) while preserving your current inference loop and transforms. These minimal changes should move accuracy sharply upward toward your target if the provided `.pth` weights are valid.'
- What this solution (achieved 0.11584) has done: 'Your current score is far below the target, which is consistent with the ensemble producing effectively random predictions because the checkpoints are not actually being applied correctly at inference time (or are being loaded but the model is run with the wrong normalization/resize for those weights). I keep the exact same ensemble + multi-TTA softmax averaging + argmax core logic, but (1) force `cudnn.benchmark=False` for determinism (your current code sets both deterministic and benchmark True, which is conflicting), (2) fix the EfficientNet-B7 inference preprocessing to match torchvision EfficientNet defaults (resize to 600 + center crop 600, not 512) while keeping the same TTA structure, and (3) ensure we only average valid probability arrays (guard against any accidental shape mismatch that could silently corrupt the mean). These are minimal inference-side changes that should move accuracy sharply upward toward your target if the `.pth` weights correspond to these backbones, without changing model architectures or the overall inference procedure. The script still run end-to-end and always write a valid `submission.csv`.'
- What this solution (achieved 0.11584) has done: 'Your current score is far below the target, which is consistent with the model weights not being applied correctly at inference (so predictions become close to random). I keep your exact ensemble + multi-TTA softmax averaging + argmax core logic, but make checkpoint loading more robust to common head/key mismatches by (a) adapting the final classifier *after* inspecting the checkpoint tensor shapes (so the model head matches the checkpoint), and (b) adding minimal key remaps for EfficientNet’s `classifier.1.*` vs `classifier.*`. I also ensure test-time transforms remain deterministic for the “base” TTA (resize+centercrop) while preserving your existing TTA list structure. These changes are narrowly targeted to make your provided `.pth` checkpoints actually load into the intended backbones, which should move accuracy strongly upward toward your 0.8927 target without changing the inference semantics.'
- What this solution (achieved 0.11584) has done: 'Your score (0.11584) is far below the target (0.8927), so we should increase accuracy; the most likely blocker is that the model checkpoints still aren’t being applied correctly (partial/non-strict loads leaving random heads) and/or the input normalization/resize doesn’t match how the checkpoints were trained. I keep the exact same ensemble + multi-TTA softmax averaging + argmax core logic, but (1) make checkpoint loading stricter and safer by ensuring the classifier head matches the checkpoint and by refusing to proceed if the head didn’t load (to avoid random predictions dragging the ensemble down), and (2) align EfficientNet-B7 preprocessing to its expected default (600 center-crop) deterministically for the “base” TTA while preserving your TTA list. Finally, I make the submission ordering *exactly* match `sample_submission.csv` without any merge-induced reordering edge cases.'
- What this solution (achieved 0.11584) has done: 'Your score is far below the target, so the smallest likely “big gain” is to ensure your inference preprocessing matches what the provided checkpoints were trained with and to prevent weak/random transforms from diluting the ensemble. I keep the same multi-model + multi-TTA softmax averaging + argmax core logic, but make two minimal inference-only adjustments: (1) use EfficientNet-B7’s expected normalization (mean/std) when running EfficientNet checkpoints (while keeping ImageNet mean/std for ResNet/DenseNet), and (2) make the extra TTAs deterministic (center-crop based) instead of RandomResizedCrop/Rotate at test-time, since stochastic crops often destroy accuracy if the model wasn’t trained with identical test-time augmentation. I also keep the sample_submission ordering join as-is to guarantee alignment, and still write a valid `submission.csv`.'
- What this solution (achieved 0.11584) has done: 'Your current score is far below the target, so we should make the smallest inference-side changes that plausibly fix “near-random” predictions. The most likely root cause here is incorrect normalization for EfficientNet (using mean/std = 0.5/0.5 instead of ImageNet) and an incorrect head-key check that can mistakenly think the classifier loaded when it didn’t (because `classifier.1.*` is hardcoded even when the last layer is at `classifier[-1]`). I (1) switch EfficientNet preprocessing to standard ImageNet mean/std, and (2) make the “head loaded” verification compute the correct last-linear index for EfficientNet/DenseNet so we don’t silently run with random heads. These are minimal, keep the same ensemble + multi-TTA softmax averaging + argmax core logic, and should move accuracy sharply upward toward your target if the checkpoints are correct.'
- What this solution (achieved 0.11584) has done: 'I make two minimal inference-side fixes that are highly likely to move your score up toward the 0.8927 target without changing the ensemble/TTA/argmax core logic. First, I enforce that every loaded checkpoint’s head outputs exactly 5 classes (the competition requirement) and skip any checkpoint that isn’t a 5-class cassava model; otherwise a wrong-class head poison the averaged probabilities and produce near-random labels. Second, I make the head-shape inference and key remapping robust for common torchvision EfficientNet/ResNet/DenseNet checkpoint formats so valid weights actually load into the intended final layer, rather than silently leaving a random head. The rest (model list, transforms, multi-model + multi-TTA softmax averaging, submission ordering) stays the same and still writes a valid `submission.csv`.'
- What this solution (achieved 0.11584) has done: 'Your current score is far below the target, so the smallest likely improvement is to stop diluting the ensemble with partially-loaded or wrong-head checkpoints. I (1) enforce that the checkpoint actually contains a 5-class head tensor and skip any model that doesn’t, (2) after loading, verify that the model’s head weights changed compared to their initial random values (guards against “loaded but didn’t really load”), and (3) ensure the test dataframe is exactly the `sample_submission.csv` ordering from the start so every probability row aligns perfectly. This keeps the same backbones, the same multi-model + multi-TTA softmax averaging + argmax core logic, and still produces `submission.csv` end-to-end.'

# 9. Code solution

## === cell 0
import numpy as np
import glob



## === cell 1
pretrained_models = glob.glob(f"../input/ebmls-seed70/*.pth") + glob.glob(
    f"../input/eb7mseed71/*.pth"
)

print(f"{len(pretrained_models)} models found.")
if len(pretrained_models) > 0:
    print("\n".join(np.sort(pretrained_models)))
else:
    print(
        "No pretrained models found in ../input/ebmls-seed70 or ../input/eb7mseed71. Will fall back to sample_submission."
    )



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

import os
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
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
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
RUN_TYPE = os.getenv("KAGGLE_KERNEL_RUN_TYPE", "Batch")
if RUN_TYPE == "Interactive":
    print("Test run in Kaggle environment (Interactive).")
    BASE_DIR = "../input/cassava-leaf-disease-classification"
    TEST_PATH = f"{BASE_DIR}/train_images"
    test_files = os.listdir(TEST_PATH)[:32]
else:
    print("Kaggle/Batch-like run.")
    cand_base_dirs = [
        "../input/cassava-leaf-disease-classification",
        "/kaggle/input/cassava-leaf-disease-classification",
        "/kaggle/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
    ]
    BASE_DIR = None
    for c in cand_base_dirs:
        if os.path.exists(c):
            BASE_DIR = c
            break
    if BASE_DIR is None:
        BASE_DIR = "/kaggle/data/cassava-leaf-disease-classification"

    TEST_PATH = f"{BASE_DIR}/test_images"
    if not os.path.exists(TEST_PATH):
        TEST_PATH = f"{BASE_DIR}/cassava-leaf-disease-classification/test_images"
    if not os.path.exists(TEST_PATH):
        raise FileNotFoundError(
            f"Could not locate test_images under BASE_DIR={BASE_DIR}"
        )

    test_files = sorted(os.listdir(TEST_PATH))

print(f"BASE_DIR: {BASE_DIR}")
print(f"TEST_PATH: {TEST_PATH}")
print(f"Number of test images: {len(test_files)}")



## === cell 7
sample_sub_path = f"{BASE_DIR}/sample_submission.csv"
df_test = pd.read_csv(sample_sub_path)[["image_id"]].copy()
df_test["label"] = 1  # placeholder, overwritten after inference
print("Loaded sample_submission for ordering:", df_test.shape)



## === cell 8
if len(df_test) == 1:
    df_test.loc[1] = df_test.loc[0]
    print(df_test)



## === cell 9
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

EFF_MEAN = IMAGENET_MEAN
EFF_STD = IMAGENET_STD

EFF_SIZE = 600

transform = {
    "test": [
        Compose(
            [
                A.Resize(height=SIZE, width=SIZE, p=1.0),
                A.CenterCrop(height=SIZE, width=SIZE, p=1.0),
                A.Normalize(
                    mean=IMAGENET_MEAN, std=IMAGENET_STD, max_pixel_value=255.0, p=1.0
                ),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        )
    ]
}




## === cell 10
class ForwardWrapper(nn.Module):
    def __init__(self, model, criterion):
        super().__init__()
        self.model = model
        self.criterion = criterion

    def forward(self, inputs, labels, phase):
        if phase == "val":
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            return outputs, loss
        if phase == "test":
            return self.model(inputs)
        raise ValueError(f"Unsupported phase: {phase}")




## === cell 11
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




## === cell 12
def _extract_state_dict(obj):
    if isinstance(obj, dict):
        for key in ["state_dict", "model_state_dict", "model", "net", "weights"]:
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
    return obj


def _normalize_state_dict_keys(state_dict):
    state_dict = _extract_state_dict(state_dict)
    if not isinstance(state_dict, dict):
        return state_dict

    new_sd = {}
    for k, v in state_dict.items():
        nk = k

        for pref in ["module.", "model.", "net."]:
            if nk.startswith(pref):
                nk = nk[len(pref) :]

        if nk.startswith("backbone."):
            nk = nk[len("backbone.") :]

        if nk.startswith("model.fc."):
            nk = "fc." + nk[len("model.fc.") :]
        if nk.startswith("model.classifier."):
            nk = "classifier." + nk[len("model.classifier.") :]

        if nk.startswith("backbone.fc."):
            nk = "fc." + nk[len("backbone.fc.") :]
        if nk.startswith("backbone.classifier."):
            nk = "classifier." + nk[len("backbone.classifier.") :]

        if nk.startswith("classifier.1."):
            nk_alt = "classifier." + nk[len("classifier.1.") :]
            new_sd[nk_alt] = v
        if nk.startswith("classifier.") and (
            nk.endswith(".weight") or nk.endswith(".bias")
        ):
            nk_alt = "classifier.1." + nk[len("classifier.") :]
            new_sd[nk_alt] = v

        new_sd[nk] = v

    return new_sd


def _try_load(model, sd, strict):
    try:
        out = model.load_state_dict(sd, strict=strict)
        return True, out
    except Exception:
        return False, None


def _infer_head_out_features_from_sd(sd):
    candidates = [
        "fc.weight",
        "classifier.weight",
        "classifier.1.weight",
        "classifier.0.weight",
        "head.weight",
    ]
    for k in candidates:
        if k in sd and hasattr(sd[k], "shape") and len(sd[k].shape) == 2:
            return int(sd[k].shape[0])
    return None


def _adapt_model_head_to_checkpoint(model, basename, sd):
    out_features = _infer_head_out_features_from_sd(sd)
    if out_features is None:
        return None

    if out_features != num_classes:
        raise RuntimeError(
            f"{basename}: checkpoint head out_features={out_features} != num_classes={num_classes} (incompatible)"
        )

    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        if model.fc.out_features != out_features:
            in_f = model.fc.in_features
            model.fc = nn.Linear(in_f, out_features)
            print(f"[head-adapt] {basename}: set fc out_features={out_features}")
    elif hasattr(model, "classifier"):
        if isinstance(model.classifier, nn.Linear):
            if model.classifier.out_features != out_features:
                in_f = model.classifier.in_features
                model.classifier = nn.Linear(in_f, out_features)
                print(
                    f"[head-adapt] {basename}: set classifier out_features={out_features}"
                )
        elif isinstance(model.classifier, nn.Sequential) and len(model.classifier) > 0:
            last = model.classifier[-1]
            if isinstance(last, nn.Linear) and last.out_features != out_features:
                in_f = last.in_features
                model.classifier[-1] = nn.Linear(in_f, out_features)
                print(
                    f"[head-adapt] {basename}: set classifier[-1] out_features={out_features}"
                )
    return out_features


def _head_keys_for_model(model):
    keys = []
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        keys += ["fc.weight", "fc.bias"]

    if hasattr(model, "classifier"):
        if isinstance(model.classifier, nn.Linear):
            keys += ["classifier.weight", "classifier.bias"]
        elif isinstance(model.classifier, nn.Sequential) and len(model.classifier) > 0:
            last_linear_idx = None
            for i in range(len(model.classifier) - 1, -1, -1):
                if isinstance(model.classifier[i], nn.Linear):
                    last_linear_idx = i
                    break
            if last_linear_idx is not None:
                keys += [
                    f"classifier.{last_linear_idx}.weight",
                    f"classifier.{last_linear_idx}.bias",
                ]
            else:
                keys += ["classifier.weight", "classifier.bias"]

    return set(keys)


def _get_head_weight_tensor(model):
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        return model.fc.weight.detach().cpu()
    if hasattr(model, "classifier"):
        if isinstance(model.classifier, nn.Linear):
            return model.classifier.weight.detach().cpu()
        if isinstance(model.classifier, nn.Sequential) and len(model.classifier) > 0:
            for i in range(len(model.classifier) - 1, -1, -1):
                if isinstance(model.classifier[i], nn.Linear):
                    return model.classifier[i].weight.detach().cpu()
    return None


def _load_weights_strict_with_fallback(model, state, basename=""):
    raw_sd = _extract_state_dict(state)
    sd = _normalize_state_dict_keys(raw_sd)

    out_features = _infer_head_out_features_from_sd(sd)
    if out_features is None:
        raise RuntimeError(
            f"{basename}: could not find head weight tensor in checkpoint"
        )

    _adapt_model_head_to_checkpoint(model, basename, sd)

    ok, out = _try_load(model, sd, strict=True)
    if ok:
        return

    sd2 = dict(sd)

    if any(k.startswith("classifier.") for k in sd2.keys()) and hasattr(model, "fc"):
        for k in list(sd2.keys()):
            if k.startswith("classifier."):
                sd2["fc." + k[len("classifier.") :]] = sd2.pop(k)

    if any(k.startswith("fc.") for k in sd2.keys()) and hasattr(model, "classifier"):
        for k in list(sd2.keys()):
            if k.startswith("fc."):
                sd2["classifier." + k[len("fc.") :]] = sd2.pop(k)

    if hasattr(model, "classifier") and isinstance(model.classifier, nn.Sequential):
        if any(k.startswith("classifier.weight") for k in sd2.keys()) or any(
            k.startswith("classifier.bias") for k in sd2.keys()
        ):
            for suf in ["weight", "bias"]:
                k0 = f"classifier.{suf}"
                if k0 in sd2:
                    sd2[f"classifier.1.{suf}"] = sd2.pop(k0)

    ok, out = _try_load(model, sd2, strict=False)
    if not ok:
        raise RuntimeError(f"Failed to load checkpoint for {basename}")

    missing = list(out.missing_keys) if hasattr(out, "missing_keys") else []
    unexpected = list(out.unexpected_keys) if hasattr(out, "unexpected_keys") else []
    print(
        f"[load_state_dict non-strict] {basename}: missing={len(missing)} unexpected={len(unexpected)}"
    )
    if len(missing) > 0:
        print("  first missing keys:", missing[:10])
    if len(unexpected) > 0:
        print("  first unexpected keys:", unexpected[:10])

    head_keys = _head_keys_for_model(model)
    if len(head_keys) > 0:
        missing_set = set(missing)
        if any(k in missing_set for k in head_keys):
            raise RuntimeError(
                f"{basename}: checkpoint load left head params missing ({sorted(head_keys & missing_set)}); "
                f"refusing to use this model to avoid degrading ensemble accuracy."
            )




## === cell 13
def predict_model(basename, net, dataloader):
    model_start_time = time.time()

    net.to(device)
    net.eval()
    torch.set_grad_enabled(False)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    probability = []
    progress = tqdm(dataloader["test"], desc=f"{basename}: ")

    for inputs, image_ids in progress:
        inputs = inputs.to(device)
        outputs = net(inputs, False, "test")
        probability.append(torch.softmax(outputs, dim=1).cpu().numpy())

    print(f"{basename} time: {time.time() - model_start_time:.2f}[sec]")
    return np.concatenate(probability, axis=0)




## === cell 14
probability = []

start_time = time.time()

if len(pretrained_models) == 0:
    print(
        "No pretrained models available; using sample_submission as a valid fallback."
    )
    df_test = pd.read_csv(f"{BASE_DIR}/sample_submission.csv")
else:
    for pretrained_model in pretrained_models:
        basename = os.path.splitext(os.path.basename(pretrained_model))[0]
        criterion = nn.CrossEntropyLoss()

        local_size = SIZE
        local_mean = IMAGENET_MEAN
        local_std = IMAGENET_STD

        if "resnet18" in basename:
            MODEL_NAME = "resnet18"
            base = models.resnet18(weights=None)
            base.fc = nn.Linear(base.fc.in_features, num_classes)
            BATCH_SIZE = 64
        elif "resnet50" in basename:
            MODEL_NAME = "resnet50"
            base = models.resnet50(weights=None)
            base.fc = nn.Linear(base.fc.in_features, num_classes)
            BATCH_SIZE = 32
        elif "resnet152" in basename:
            MODEL_NAME = "resnet152"
            base = models.resnet152(weights=None)
            base.fc = nn.Linear(base.fc.in_features, num_classes)
            BATCH_SIZE = 16
        elif "resnext101" in basename:
            MODEL_NAME = "resnext101"
            base = models.resnext101_32x8d(weights=None)
            base.fc = nn.Linear(base.fc.in_features, num_classes)
            BATCH_SIZE = 12
        elif "densenet201" in basename:
            MODEL_NAME = "densenet201"
            base = models.densenet201(weights=None)
            base.classifier = nn.Linear(base.classifier.in_features, num_classes)
            BATCH_SIZE = 12
        elif "efficientnet-b7" in basename:
            MODEL_NAME = "efficientnet-b7"
            base = efficientnet_b7(weights=None)
            if hasattr(base, "classifier") and isinstance(
                base.classifier, nn.Sequential
            ):
                in_ftrs = base.classifier[-1].in_features
                base.classifier[-1] = nn.Linear(in_ftrs, num_classes)
            else:
                raise ValueError("Unexpected EfficientNet-B7 classifier structure.")
            BATCH_SIZE = 10
            local_size = EFF_SIZE
            local_mean = EFF_MEAN
            local_std = EFF_STD
        else:
            print(f"{basename} is not supported.")
            raise ValueError(f"Unsupported model in filename: {basename}")

        print(f"{basename}: {MODEL_NAME}")

        head_before = _get_head_weight_tensor(base)
        if head_before is not None:
            head_before = head_before.clone()

        state = torch.load(pretrained_model, map_location="cpu")

        try:
            _load_weights_strict_with_fallback(base, state, basename=basename)
        except Exception as e:
            print(f"[skip model] {basename}: {repr(e)}")
            del base
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

        head_after = _get_head_weight_tensor(base)
        if head_before is not None and head_after is not None:
            if torch.equal(head_before, head_after):
                print(
                    f"[skip model] {basename}: head weights unchanged after loading (likely mismatch); skipping to avoid degrading ensemble."
                )
                del base
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue

        net = ForwardWrapper(base, criterion)

        for param in net.parameters():
            param.requires_grad = False

        per_model_transforms = [
            Compose(
                [
                    A.Resize(height=local_size, width=local_size, p=1.0),
                    A.CenterCrop(height=local_size, width=local_size, p=1.0),
                    A.Normalize(
                        mean=local_mean, std=local_std, max_pixel_value=255.0, p=1.0
                    ),
                    ToTensorV2(p=1.0),
                ],
                p=1.0,
            ),
            Compose(
                [
                    A.Resize(height=local_size, width=local_size, p=1.0),
                    A.HorizontalFlip(p=1.0),
                    A.CenterCrop(height=local_size, width=local_size, p=1.0),
                    A.Normalize(
                        mean=local_mean, std=local_std, max_pixel_value=255.0, p=1.0
                    ),
                    ToTensorV2(p=1.0),
                ],
                p=1.0,
            ),
            Compose(
                [
                    A.Resize(height=local_size + 32, width=local_size + 32, p=1.0),
                    A.CenterCrop(height=local_size, width=local_size, p=1.0),
                    A.Normalize(
                        mean=local_mean, std=local_std, max_pixel_value=255.0, p=1.0
                    ),
                    ToTensorV2(p=1.0),
                ],
                p=1.0,
            ),
            Compose(
                [
                    A.Resize(height=local_size + 32, width=local_size + 32, p=1.0),
                    A.HorizontalFlip(p=1.0),
                    A.CenterCrop(height=local_size, width=local_size, p=1.0),
                    A.Normalize(
                        mean=local_mean, std=local_std, max_pixel_value=255.0, p=1.0
                    ),
                    ToTensorV2(p=1.0),
                ],
                p=1.0,
            ),
            Compose(
                [
                    A.Resize(height=local_size + 64, width=local_size + 64, p=1.0),
                    A.CenterCrop(height=local_size, width=local_size, p=1.0),
                    A.Normalize(
                        mean=local_mean, std=local_std, max_pixel_value=255.0, p=1.0
                    ),
                    ToTensorV2(p=1.0),
                ],
                p=1.0,
            ),
            Compose(
                [
                    A.Resize(height=local_size + 64, width=local_size + 64, p=1.0),
                    A.HorizontalFlip(p=1.0),
                    A.CenterCrop(height=local_size, width=local_size, p=1.0),
                    A.Normalize(
                        mean=local_mean, std=local_std, max_pixel_value=255.0, p=1.0
                    ),
                    ToTensorV2(p=1.0),
                ],
                p=1.0,
            ),
        ]

        for tid, transform_ in enumerate(per_model_transforms):
            print(f"transform loop={tid}")
            dataset = {"test": TestDataset(df_test, transform=transform_)}
            dataloader = {
                "test": torch.utils.data.DataLoader(
                    dataset["test"],
                    batch_size=BATCH_SIZE,
                    shuffle=False,
                    num_workers=2,
                    pin_memory=True,
                )
            }

            proba = predict_model(basename, net, dataloader)

            if proba.ndim != 2:
                raise RuntimeError(f"Bad proba ndim for {basename}: got {proba.ndim}")
            if proba.shape[0] != len(df_test):
                raise RuntimeError(
                    f"Bad proba N for {basename}: got {proba.shape[0]}, expected {len(df_test)}"
                )
            if proba.shape[1] != num_classes:
                raise RuntimeError(
                    f"Bad proba C for {basename}: got {proba.shape[1]}, expected {num_classes}"
                )
            probability.append(proba)

        del net
        del base
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if len(probability) == 0:
        print(
            "All models were skipped/unavailable; falling back to sample_submission labels."
        )
        df_test = pd.read_csv(f"{BASE_DIR}/sample_submission.csv")
    else:
        prob_array = np.stack(probability, axis=0)
        df_test["mean"] = prob_array.mean(axis=0).argmax(axis=1)

        print(f"total time: {time.time() - start_time:.2f}[sec]")



## === cell 15
if len(df_test) == 2 and df_test.loc[0, "image_id"] == df_test.loc[1, "image_id"]:
    df_test = pd.read_csv(f"{BASE_DIR}/sample_submission.csv")
else:
    if "mean" in df_test.columns:
        df_test["label"] = df_test["mean"].astype(int)

sample_df = pd.read_csv(f"{BASE_DIR}/sample_submission.csv")[["image_id"]]
df_test = df_test[["image_id", "label"]].copy()
df_test = sample_df.join(df_test.set_index("image_id"), on="image_id")
if df_test["label"].isna().any():
    df_test["label"] = df_test["label"].fillna(1).astype(int)
else:
    df_test["label"] = df_test["label"].astype(int)



## === cell 16
df_test.head()



## === cell 17
df_test[["image_id", "label"]].to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", df_test[["image_id", "label"]].shape)
print(df_test[["image_id", "label"]].head())
