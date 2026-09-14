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

0.8936234511937141

# 6. Current score

0.11584

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.11584) has done: 'I fix the missing EfficientNet dependency by replacing it with torchvision’s built-in `efficientnet_b7` implementation while keeping the same inference-time model selection logic. I also fix the Kaggle path detection so it always finds the dataset under `/kaggle/input/...` (your current logic falls back to a non-existent local `data/` folder). Albumentations v2 changed `RandomResizedCrop`’s signature, so I update those calls to the correct `size=(H,W)` form to remove the runtime validation error. Finally, I make ensembling robust when no external pretrained weights are found (create a safe fallback prediction aligned to `sample_submission.csv`) and ensure a valid `submission.csv` is always written.'
- What this solution (achieved 0.11584) has done: 'Your low score is consistent with the “no pretrained models found” fallback path: it writes the sample submission labels (mostly a single class), which scores near random. The minimal change to move toward your target is to reliably locate and load the provided `.pth` weights by expanding the search to the actual dataset directories you have (`/kaggle/input/...` and `/kaggle/data/...`), while keeping the same model selection and TTA/averaging logic. I also ensure the test file list is aligned to `sample_submission.csv` order (to avoid any accidental row-order mismatch) and make state-dict loading robust to common checkpoint formats (`state_dict` wrappers), without changing architecture or inference semantics. These changes should increase accuracy substantially toward your target, assuming the weights exist in the environment.'
- What this solution (achieved 0.11584) has done: 'Your current score strongly suggests the “no pretrained .pth found” fallback is still being triggered (or the wrong weights are being loaded), so the smallest meaningful improvement is to (1) reliably find weights anywhere under `/kaggle/input` or `/kaggle/data`, and (2) make checkpoint loading robust to common wrappers (`module.` prefix, `state_dict` nesting). I also force `df_test` to follow `sample_submission.csv` ordering and use that same `image_id` list for the dataset, eliminating any chance of misalignment between predictions and the submission rows. These changes preserve your existing model-selection logic, TTA loop, and averaging; they only unblock real inference with the intended weights so accuracy can move toward your target. If weights truly do not exist in the environment, it still produce a valid `submission.csv` as before.'
- What this solution (achieved 0.11584) has done: 'Your current score (0.11584) is consistent with silently using the fallback `sample_submission.csv` labels or with a prediction/row-order mismatch; both produce near-random accuracy. I make two minimal, score-relevant fixes: (1) force `df_test` (and the dataset image list) to follow `sample_submission.csv` ordering and filter to images that actually exist on disk, and (2) make checkpoint loading robust to common key mismatches (extra `fc/classifier` keys, nested wrappers) without changing the model selection logic, TTA loop, or averaging. I also ensure the ensemble always writes predictions for exactly the submission rows (keeping any missing-image rows filled with the baseline label rather than shifting alignment). These changes should move the score substantially upward toward your target by making sure real weights are used correctly and predictions align with Kaggle’s expected row order.'
- What this solution (achieved 0.11584) has done: 'Your score is near-random, which strongly indicates the inference is not using the intended trained weights correctly (or predictions are misaligned). I make the smallest changes that directly increase accuracy: (1) ensure test inference order exactly matches `sample_submission.csv` (no accidental shifting), (2) fix the EfficientNet wrapper so it replaces the correct classifier layer for torchvision EfficientNet (your current wrapper likely prevents the checkpoint head from loading, leaving a randomly initialized head), and (3) make state-dict loading slightly smarter by optionally reinitializing the classifier to match checkpoint output classes if needed. These changes preserve your existing model selection, TTA loop, averaging, and loss; they just unblock correct checkpoint loading and correct label alignment.'
- What this solution (achieved 0.11584) has done: 'Your current score is far below the target, so we should safely increase accuracy by ensuring the inference actually uses the intended trained heads instead of leaving randomly-initialized classifier layers. The main minimal fix is to stop overwriting the classifier layer inside the EfficientNet wrapper (and similarly for other wrappers) so that checkpoints can load their trained classifier weights; this preserves the same architecture at inference time (use the checkpoint’s architecture), and only affects whether weights load correctly. I also restrict the model list to those that match supported basenames (DenseNet201 / EfficientNet-B7) to avoid accidentally averaging in incompatible/random models found under `/kaggle/input/**/*.pth`. Finally, I keep your submission-row ordering exactly aligned to `sample_submission.csv`, so predictions cannot shift.'
- What this solution (achieved 0.11584) has done: 'Your score is near-random, so the smallest change likely to move you toward the target is to ensure the EfficientNet-B7 wrapper matches the checkpoint’s expected head keys: your current `FinalLayerMixupModelEN_TV` does not expose the trained classifier under `fc.*`, so many checkpoints won’t load their final layer and predict garbage. I minimally modify that wrapper to (a) expose `fc`/`classifier` consistently and (b) in `forward()` call the wrapped model directly so logits flow through the loaded head. I also update `_maybe_adjust_classifier_for_ckpt()` to handle torchvision EfficientNet keys (`model.classifier.1.*` / `classifier.1.*`) so the head shape matches the checkpoint before loading. Everything else (model selection, TTA loop, averaging, submission alignment) remains unchanged.'
- What this solution (achieved 0.11584) has done: 'Your score is near-random, so the most likely issue is that checkpoints are being loaded into a model whose final classification head does not match the checkpoint keys, leaving a randomly initialized head (especially for torchvision EfficientNet-B7). I keep your ensembling/TTA/inference loop intact and make a minimal, score-relevant fix: make the EfficientNet wrapper expose and use the classifier head in a way that is compatible with common checkpoint key patterns (e.g., `fc.*`, `classifier.*`, `model.classifier.1.*`). I also make the state-dict loader map common head key variants into the wrapper’s expected keys before loading, without changing architecture or inference semantics otherwise. These changes should move accuracy substantially upward toward your target by ensuring the trained heads actually load.'
- What this solution (achieved 0.11584) has done: 'Your current score is near-random, which most often happens here when the EfficientNet-B7 checkpoints’ trained classifier weights are not actually being used at inference (head keys don’t match, so the head stays randomly initialized). I make the smallest fix that keeps your model/loop/TTA logic intact: explicitly replace torchvision EfficientNet-B7’s classifier head with a 5-class Linear layer in the wrapper (matching what your DenseNet/ResNet wrappers already do) and expose it consistently as `fc`, so checkpoint loading reliably populates the trained head. I also broaden the head-key remapping to include common EfficientNet key patterns and make `_maybe_adjust_classifier_for_ckpt` correctly detect both 2D (weight) and 1D (bias) head tensors to adjust out_features before loading. These changes should substantially increase accuracy toward your target without changing the ensembling or inference semantics.'
- What this solution (achieved 0.11584) has done: 'Your score is near-random, which most plausibly comes from loading checkpoints whose classifier head weights are not actually being applied due to key mismatches (especially for EfficientNet-B7), leaving a randomly initialized head at inference. I make a minimal, score-critical change in checkpoint loading: add robust key remapping so common head keys load into the actual torchvision EfficientNet classifier (`model.classifier.1.*`) and DenseNet classifier (`model.classifier.*`), not just `fc.*`. I also stop resizing/replacing `net.fc` in a way that can break the link to the real head for EfficientNet, and instead resize the true underlying head module when needed. These changes preserve your existing model choices, TTA loop, averaging, and submission alignment, but should make inference use the trained heads and move accuracy strongly toward your target.'

# 9. Code solution

## === cell 0
import numpy as np
import glob

pretrained_models = []

pretrained_models += glob.glob(f"../input/densenet201-seed60/*.pth")
pretrained_models += glob.glob(
    f"../input/eb7slseed70/efficientnet-b7slseed70.best/*.pth"
)
pretrained_models += glob.glob(
    f"../input/eb7slseed70/efficientnet-b7sl_SEED70.best/*.pth"
)
pretrained_models += glob.glob(f"/kaggle/input/densenet201-seed60/*.pth")
pretrained_models += glob.glob(
    f"/kaggle/input/eb7slseed70/efficientnet-b7slseed70.best/*.pth"
)
pretrained_models += glob.glob(
    f"/kaggle/input/eb7slseed70/efficientnet-b7sl_SEED70.best/*.pth"
)
pretrained_models += glob.glob(f"/kaggle/data/densenet201-seed60/*.pth")
pretrained_models += glob.glob(
    f"/kaggle/data/eb7slseed70/efficientnet-b7slseed70.best/*.pth"
)
pretrained_models += glob.glob(
    f"/kaggle/data/eb7slseed70/efficientnet-b7sl_SEED70.best/*.pth"
)

pretrained_models += glob.glob(
    f"/kaggle/input/**/densenet201-seed60/*.pth", recursive=True
)
pretrained_models += glob.glob(
    f"/kaggle/input/**/eb7slseed70/efficientnet-b7slseed70.best/*.pth", recursive=True
)
pretrained_models += glob.glob(
    f"/kaggle/input/**/eb7slseed70/efficientnet-b7sl_SEED70.best/*.pth", recursive=True
)

all_found = []
all_found += glob.glob("/kaggle/input/**/*.pth", recursive=True)
all_found += glob.glob("/kaggle/data/**/*.pth", recursive=True)
pretrained_models += all_found

pretrained_models = sorted(set(pretrained_models))


def _is_supported_ckpt(path: str) -> bool:
    b = path.lower()
    base = path.lower().split("/")[-1]
    return ("densenet201" in base) or ("efficientnet-b7" in base) or ("eb7" in b)


pretrained_models = [p for p in pretrained_models if _is_supported_ckpt(p)]
pretrained_models = sorted(set(pretrained_models))

print(f"{len(pretrained_models)} supported models found.")
if len(pretrained_models) > 0:
    print("\n".join(np.sort(pretrained_models)[:200]))
    if len(pretrained_models) > 200:
        print(f"... (showing first 200 of {len(pretrained_models)})")



## === cell 1
import pandas as pd

import torch
import torch.nn as nn
import torch.utils.data as data

import torchvision
from torchvision import models, transforms
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
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


SEED = 42
seed_everything(seed=SEED)



## === cell 2
from torchvision.models import efficientnet_b7



## === cell 3
SIZE = 512
num_classes = 5



## === cell 4
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用デバイス: {device}")



## === cell 5
if os.path.isdir("/kaggle/input/cassava-leaf-disease-classification"):
    BASE_DIR = "/kaggle/input/cassava-leaf-disease-classification"
elif os.path.isdir("../input/cassava-leaf-disease-classification"):
    BASE_DIR = "../input/cassava-leaf-disease-classification"
elif os.path.isdir("/kaggle/data/cassava-leaf-disease-classification"):
    BASE_DIR = "/kaggle/data/cassava-leaf-disease-classification"
else:
    BASE_DIR = "data"

run_type = os.getenv("KAGGLE_KERNEL_RUN_TYPE", "")
if run_type == "Interactive":
    print("Test run in Kaggle environment (Interactive).")
    TEST_PATH = f"{BASE_DIR}/train_images"
    test_files = os.listdir(TEST_PATH)[:32]
elif run_type == "Batch":
    print("In Kaggle environment (Batch).")
    TEST_PATH = f"{BASE_DIR}/test_images"
    test_files = os.listdir(TEST_PATH)
else:
    if os.path.isdir(f"{BASE_DIR}/test_images"):
        print("Environment run type unknown; using test_images if available.")
        TEST_PATH = f"{BASE_DIR}/test_images"
        test_files = os.listdir(TEST_PATH)
    else:
        print("Environment run type unknown; using train_images subset.")
        TEST_PATH = f"{BASE_DIR}/train_images"
        test_files = os.listdir(TEST_PATH)[:32]

print(f"BASE_DIR: {BASE_DIR}")
print(f"TEST_PATH: {TEST_PATH}")
print(f"Number of test images: {len(test_files)}")



## === cell 6
sample_path = f"{BASE_DIR}/sample_submission.csv"
if os.path.isfile(sample_path):
    df_sample = pd.read_csv(sample_path)[["image_id", "label"]].copy()
else:
    df_sample = pd.DataFrame(sorted(test_files), columns=["image_id"])
    df_sample["label"] = 1

existing_set = set(test_files)
df_sample["exists"] = df_sample["image_id"].isin(existing_set)

df_test_run = df_sample[df_sample["exists"]].reset_index(drop=True).copy()

df_out = df_sample.drop(columns=["exists"]).copy()
df_out["label"] = df_out["label"].astype(int)

print(f"Rows in sample_submission: {len(df_out)}")
print(f"Images found on disk under TEST_PATH: {len(existing_set)}")
print(f"Rows used for inference (existing images): {len(df_test_run)}")



## === cell 7
if len(df_test_run) == 1:
    df_test_run.loc[1] = df_test_run.loc[0]
    print(df_test_run)



## === cell 8
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

transform = {
    "test": [
        Compose(
            [
                A.CenterCrop(height=SIZE, width=SIZE),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                A.HorizontalFlip(p=1.0),
                A.CenterCrop(height=SIZE, width=SIZE),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                A.RandomResizedCrop(size=(SIZE, SIZE)),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                A.RandomResizedCrop(size=(SIZE, SIZE)),
                A.HorizontalFlip(p=1.0),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                A.RandomResizedCrop(size=(SIZE, SIZE)),
                A.VerticalFlip(p=1.0),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                A.Rotate(p=1.0),
                A.RandomResizedCrop(size=(SIZE, SIZE)),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
    ]
}




## === cell 9
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
            x = x.squeeze()
            outputs = self.fc(x)
            loss = self.criterion(outputs, labels)

            return outputs, loss

        if phase == "test":
            x = self.convlayer(inputs)
            x = x.squeeze()
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
        mixed_x = mixed_x.squeeze()
        outputs = self.fc(mixed_x)

        labels_a = labels
        labels_b = labels[index]

        pred = outputs
        loss = lam * self.criterion(pred, labels_a) + (1 - lam) * self.criterion(
            pred, labels_b
        )

        return outputs, loss, labels_a, labels_b, lam




## === cell 10
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
            x = x.squeeze()
            outputs = self.fc(x)
            loss = self.criterion(outputs, labels)

            return outputs, loss

        if phase == "test":
            x = self.convlayer(inputs)
            x = self.AdaptiveAvgPool2d(x)
            x = x.squeeze()
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
        mixed_x = mixed_x.squeeze()
        outputs = self.fc(mixed_x)

        labels_a = labels
        labels_b = labels[index]

        pred = outputs
        loss = lam * self.criterion(pred, labels_a) + (1 - lam) * self.criterion(
            pred, labels_b
        )

        return outputs, loss, labels_a, labels_b, lam




## === cell 11
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
        raise SystemExit(1)




## === cell 12
class FinalLayerMixupModelEN_TV(nn.Module):
    def __init__(self, model, criterion, num_classes, alpha):
        """
        Keep the same wrapper logic, but ensure we expose the *actual* EfficientNet head module
        so checkpoint keys can be remapped to it reliably.
        """
        super().__init__()
        self.model = model
        self.criterion = criterion

        if not (
            hasattr(self.model, "classifier")
            and isinstance(self.model.classifier, nn.Sequential)
        ):
            raise ValueError(
                "Unexpected EfficientNet structure: missing Sequential classifier"
            )

        if not (
            len(self.model.classifier) >= 2
            and isinstance(self.model.classifier[-1], nn.Linear)
        ):
            raise ValueError("Unexpected EfficientNet classifier format")

        in_features = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Linear(in_features, num_classes)

        self.fc = self.model.classifier[-1]

    def forward(self, inputs, labels, phase):
        if phase == "val":
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            return outputs, loss
        if phase == "test":
            outputs = self.model(inputs)
            return outputs
        raise SystemExit(1)




## === cell 13
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




## === cell 14
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
            probability.append(torch.softmax(outputs, dim=1).cpu().numpy())

    print(f"{basename} time: {time.time() - model_start_time:.2f}[sec]")
    return np.concatenate(probability, axis=0)




## === cell 15
def _extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for k in ["state_dict", "model", "model_state_dict", "net", "network"]:
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
        return {k[len("module.") :]: v for k, v in state_dict.items()}
    return state_dict


def _remap_classifier_keys_minimal(state):
    """
    CHANGE (score-critical, minimal): many checkpoints save the head under different names.
    We remap common variants onto the *actual* head modules used by this notebook:
      - EfficientNet (torchvision): model.classifier.1.(weight|bias)
      - DenseNet (torchvision): model.classifier.(weight|bias)
      - ResNet-style wrapper: fc.(weight|bias)
    This improves the chance the trained head loads instead of leaving a random head.
    """
    if not isinstance(state, dict):
        return state

    if "model.classifier.1.weight" in state and "model.classifier.1.bias" in state:
        return state

    if "model.classifier.weight" in state and "model.classifier.bias" in state:
        return state

    new_state = dict(state)

    sources = [
        ("fc.weight", "fc.bias"),
        ("classifier.weight", "classifier.bias"),
        ("classifier.1.weight", "classifier.1.bias"),
        ("model.classifier.weight", "model.classifier.bias"),
        ("model.classifier.1.weight", "model.classifier.1.bias"),
        ("model.fc.weight", "model.fc.bias"),
        ("model._fc.weight", "model._fc.bias"),
        ("_fc.weight", "_fc.bias"),
        ("head.weight", "head.bias"),
        ("model.head.weight", "model.head.bias"),
    ]

    found = None
    for w_key, b_key in sources:
        if w_key in state and b_key in state:
            found = (state[w_key], state[b_key])
            break

    if found is None:
        return state

    w, b = found

    new_state.setdefault("fc.weight", w)
    new_state.setdefault("fc.bias", b)
    new_state.setdefault("model.fc.weight", w)
    new_state.setdefault("model.fc.bias", b)

    new_state.setdefault("model.classifier.weight", w)
    new_state.setdefault("model.classifier.bias", b)

    new_state.setdefault("classifier.weight", w)
    new_state.setdefault("classifier.bias", b)

    new_state.setdefault("model.classifier.1.weight", w)
    new_state.setdefault("model.classifier.1.bias", b)
    new_state.setdefault("classifier.1.weight", w)
    new_state.setdefault("classifier.1.bias", b)

    return new_state


def _maybe_adjust_head_for_ckpt(net, state):
    """
    CHANGE (score-critical, minimal): adjust the *real underlying* head module (not just net.fc),
    so shape matches the checkpoint and the trained head can load.
    """
    if not isinstance(state, dict):
        return

    cand = [
        "model.classifier.1.weight",
        "classifier.1.weight",
        "model.classifier.weight",
        "classifier.weight",
        "fc.weight",
        "model.fc.weight",
        "model._fc.weight",
        "_fc.weight",
        "model.classifier.1.bias",
        "classifier.1.bias",
        "model.classifier.bias",
        "classifier.bias",
        "fc.bias",
        "model.fc.bias",
        "model._fc.bias",
        "_fc.bias",
    ]

    out_features = None
    for k in cand:
        if k in state and hasattr(state[k], "shape"):
            shp = state[k].shape
            if len(shp) == 2:
                out_features = int(shp[0])
                break
            if len(shp) == 1:
                out_features = int(shp[0])
                break
    if out_features is None:
        return

    if (
        hasattr(net, "model")
        and hasattr(net.model, "classifier")
        and isinstance(net.model.classifier, nn.Sequential)
    ):
        last = net.model.classifier[-1]
        if isinstance(last, nn.Linear) and last.out_features != out_features:
            net.model.classifier[-1] = nn.Linear(last.in_features, out_features)
            if hasattr(net, "fc") and isinstance(getattr(net, "fc"), nn.Linear):
                net.fc = net.model.classifier[-1]
            return

    if (
        hasattr(net, "fc")
        and isinstance(net.fc, nn.Linear)
        and net.fc.out_features != out_features
    ):
        net.fc = nn.Linear(net.fc.in_features, out_features)
        return


def _load_state_dict_forgiving(net, state):
    missing, unexpected = net.load_state_dict(state, strict=False)
    if len(unexpected) > 0:
        print(f"WARNING: unexpected keys (showing up to 20): {unexpected[:20]}")
    if len(missing) > 0:
        print(f"WARNING: missing keys (showing up to 20): {missing[:20]}")
    return missing, unexpected


probability = []

start_time = time.time()

if len(pretrained_models) == 0:
    print(
        "WARNING: No pretrained .pth models found. Falling back to sample_submission.csv labels to produce a valid file."
    )
    df_out[["image_id", "label"]].to_csv("submission.csv", index=False)
    print(f"Saved fallback submission.csv with shape={df_out.shape}")
else:
    if len(df_test_run) == 0:
        print(
            "WARNING: No test images found on disk; writing baseline submission from sample_submission."
        )
        df_out[["image_id", "label"]].to_csv("submission.csv", index=False)
        print(f"Saved baseline submission.csv with shape={df_out.shape}")
    else:
        for pretrained_model in pretrained_models:
            basename = os.path.splitext(os.path.basename(pretrained_model))[0]

            criterion = nn.CrossEntropyLoss()

            if "resnet18" in basename:
                MODEL_NAME = "resnet18"
                net = models.resnet18(weights=None)
                net = FinalLayerMixupModel(net, criterion, num_classes, False)
                BATCH_SIZE = 64
            elif "resnet50" in basename:
                MODEL_NAME = "resnet50"
                net = models.resnet50(weights=None)
                net = FinalLayerMixupModel(net, criterion, num_classes, False)
                BATCH_SIZE = 32
            elif "resnet152" in basename:
                MODEL_NAME = "resnet152"
                net = models.resnet152(weights=None)
                net = FinalLayerMixupModel(net, criterion, num_classes, False)
                BATCH_SIZE = 16
            elif "resnext101" in basename:
                MODEL_NAME = "resnext101"
                net = models.resnext101_32x8d(weights=None)
                net = FinalLayerMixupModel(net, criterion, num_classes, False)
                BATCH_SIZE = 12
            elif "densenet201" in basename:
                MODEL_NAME = "densenet201"
                net = models.densenet201(weights=None)
                net = FinalLayerMixupModelDenseNet(net, criterion, num_classes, False)
                BATCH_SIZE = 12
            elif "efficientnet-b7" in basename or "eb7" in basename:
                MODEL_NAME = "efficientnet-b7"
                net = efficientnet_b7(weights=None)
                net = FinalLayerMixupModelEN_TV(net, criterion, num_classes, False)
                BATCH_SIZE = 10
            else:
                print(f"{basename} is not supported.")
                raise SystemExit(1)

            print(f"{basename}: {MODEL_NAME}  | loading: {pretrained_model}")

            ckpt = torch.load(pretrained_model, map_location="cpu")
            state = _strip_module_prefix(_extract_state_dict(ckpt))

            state = _remap_classifier_keys_minimal(state)
            _maybe_adjust_head_for_ckpt(net, state)
            _load_state_dict_forgiving(net, state)

            for param in net.parameters():
                param.requires_grad = False

            for tid, transform_ in enumerate(transform["test"]):
                print(f"transform loop={tid}")
                dataset = {
                    "test": TestDataset(df_test_run, transform=transform_),
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
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        prob_arr = np.stack(probability, axis=0)  # (n_models*n_tta, N_run, 5)
        mean_proba = prob_arr.mean(axis=0)  # (N_run, 5)
        pred_run = mean_proba.argmax(axis=1).astype(int)

        pred_map = dict(zip(df_test_run["image_id"].tolist(), pred_run.tolist()))
        df_out["label"] = (
            df_out["image_id"].map(pred_map).fillna(df_out["label"]).astype(int)
        )

        print(f"total time: {time.time() - start_time:.2f}[sec]")



## === cell 16
df_out.head()



## === cell 17
df_out["label"] = df_out["label"].astype(int)
df_out[["image_id", "label"]].to_csv("submission.csv", index=False)
print("Wrote submission.csv:", pd.read_csv("submission.csv").shape)
