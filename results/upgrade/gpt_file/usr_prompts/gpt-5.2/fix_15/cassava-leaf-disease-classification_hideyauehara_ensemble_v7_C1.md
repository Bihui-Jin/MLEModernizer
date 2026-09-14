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

0.8939256572982774

# 6. Current score

0.58707

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.11584) has done: 'I fix the incorrect `TEST_PATH` selection and file listing that currently includes a nested `test_images` directory name, which causes `cv2.imread` to fail with `.../test_images/test_images`. I make the test-file discovery robust by reading `sample_submission.csv` (authoritative image_id list) when available and by filtering `os.listdir` to image files only, preventing directories from entering the dataset. I also fix the Albumentations `RandomResizedCrop` API usage for v2 and make the forward pass safe for batch size 1 by replacing `squeeze()` with `flatten(1)`. These changes are execution/format correctness fixes (and should improve score versus the current “not yielded” state) while preserving the existing ensemble/TTA core logic.'
- What this solution (achieved 0.10762) has done: 'I fix the Albumentations v2 `RandomResizedCrop` API mismatch that currently crashes transform creation, which prevents `transform` from being defined and cascades into later NameError/KeyError failures. I keep the exact TTA/ensemble logic intact, only updating the transform call signature to the required `size=(H,W)` form so the pipeline runs end-to-end. I also make the prediction call robust to the wrapper model’s `forward` signature by passing a dummy tensor for `labels` during test (instead of a Python `False`), avoiding potential type/length issues. With these fixes, the script complete and write a valid `submission.csv` with the required columns and ordering from `sample_submission.csv`, which should materially improve accuracy versus the current 0.11584.'
- What this solution (achieved 0.10762) has done: 'Your very low accuracy is most consistent with the test-time transforms being incompatible with how these pretrained checkpoints were trained (notably `CenterCrop(512)` without a prior resize, plus stochastic `RandomResizedCrop` at inference). To move the score up toward the 0.894 target while preserving the ensemble/TTA core logic, I make the test-time preprocessing deterministic and standard for ImageNet-style models: `Resize(SIZE,SIZE)` followed by optional flips/rotate (no random crop). I also disable CuDNN benchmark (your code currently sets deterministic=True and benchmark=True simultaneously, which can create nondeterministic kernels and slightly unstable outputs). These are minimal, metric-aligned changes that keep the same models/loops/softmax averaging but should substantially improve correctness and thus accuracy.'
- What this solution (achieved 0.0) has done: 'Your current score (0.10762) is far below the target (0.8939), so we should only make small, low-risk changes that improve correctness of inference without changing the model ensemble/TTA core idea. The biggest likely issue is that your wrapper models create a *new randomly initialized final classifier layer* (fc/classifier) but then you load checkpoints with `strict=True`, which either fails silently in some environments or (more likely here) ends up not using the intended trained classifier weights—leading to near-random predictions. I minimally change the wrappers so they **do not replace the classifier layer at init**, and instead use the classifier that comes from the checkpoint (keeping the same forward logic). I also make checkpoint loading robust to common saved-dict formats (`state_dict` key, `module.` prefixes) while still preferring strict loading whenever keys match, because that should move accuracy sharply upward toward the target without changing the architecture or inference averaging.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with the script either not using the intended competition checkpoints (because the `../input/densenet201-04-2019data` / `../input/eb7m-seed70` paths don’t exist here) or loading them into mismatched architectures, which yields near-random predictions. To move accuracy upward toward the 0.8939 target with minimal logic changes, I keep your exact ensemble+TTA averaging but (1) discover `.pth` models from the actual available `/kaggle/input` tree, and (2) make checkpoint loading deterministically “best-effort strict” by inferring the architecture from checkpoint keys instead of filename substrings when needed. If no usable checkpoints are found, the code still writes a valid `submission.csv` (fallback), but the main goal is to actually use the provided competition models so the score rises substantially.'
- What this solution (achieved 0.0) has done: 'Your 0.0 accuracy strongly suggests the submission is not being scored on meaningful predictions, most likely because no valid competition checkpoints are being loaded (so the fallback ImageNet resnet18 predicts almost randomly for 5 cassava classes). To move the score up toward the 0.8939 target while keeping your ensemble+TTA inference logic intact, I (1) restrict checkpoint discovery to the competition dataset directory (avoids picking up unrelated `.pth` files), and (2) add a minimal “cassava head fix” when the fallback is used: replace the final FC to output 5 classes and load the closest available pretrained weights, so predictions are at least semantically valid. I also ensure the submission rows match `sample_submission.csv` ordering exactly (already mostly true) and keep all transforms/inference averaging unchanged. These are small, execution- and correctness-focused changes that should raise accuracy substantially from 0.0 without changing the overall approach.'
- What this solution (achieved 0.58857) has done: 'Your current 0.0 score is consistent with the model heads not matching the 5 cassava classes for common backbones (ResNet/DenseNet/EfficientNet), causing checkpoint loads to be non-strict and leaving randomly initialized final layers in place—leading to near-random predictions. To move accuracy up toward the 0.8939 target while preserving your ensemble+TTA inference logic, I minimally set each backbone’s classifier layer to `num_classes=5` *before* loading the checkpoint so the trained head weights can load correctly. I also make architecture selection prefer the inferred architecture from checkpoint keys (not just filename substrings) to reduce accidental mismatches. Finally, I keep your transforms, TTA averaging, and submission formatting unchanged, ensuring a valid `submission.csv` is produced.'
- What this solution (achieved 0.58819) has done: 'Your current score (0.58857) is well below the target (0.8939), so we should make small, low-risk inference fixes that improve correctness without changing your ensemble/TTA logic. The most likely accuracy killer here is a checkpoint mismatch: your wrapper modules (`convlayer`/`fc`) create different parameter names than common saved checkpoints (which usually save `model.*`, `features.*`, etc.), causing frequent non-strict loads and leaving parts randomly initialized. I (1) make checkpoint loading “wrapper-aware” by trying to load into `net.model` first when present (EfficientNet), and otherwise falling back to loading into the wrapper, and (2) expand architecture inference to detect ResNeXt and ResNet-152 so more checkpoints are instantiated with the correct backbone. These are minimal changes that should increase the fraction of correctly loaded checkpoints and move accuracy toward your target while preserving the same averaging/argmax submission semantics.'
- What this solution (achieved 0.58632) has done: 'I make two minimal inference-only fixes that tend to recover a lot of accuracy without changing your ensemble/TTA averaging core logic. First, I align EfficientNet-B7 checkpoint loading by supporting both common key styles (`classifier.*` and `model.classifier.*`) so strict loading succeeds more often instead of silently falling back to non-strict with a randomly initialized head. Second, I make checkpoint-architecture inference more reliable for ResNeXt/ResNet by fixing the current brittle key heuristics, again increasing the fraction of correctly matched backbones and thus improving predictions toward your target. Everything else (transforms list, model wrappers, softmax+mean+argmax, submission formatting/ordering) remains the same.'
- What this solution (achieved 0.58595) has done: 'Your score gap to the target is large (0.586 → 0.894), so we should make a small inference-only change that increases correctness without altering your ensemble/TTA averaging or model wrappers. The biggest likely remaining issue is that your EfficientNet-B7 wrapper (`FinalLayerMixupModelEN`) uses torchvision’s standard forward, but many Cassava EfficientNet checkpoints were trained with an extra pooling/dropout/head structure (often saved under `model.*`, or with `model._fc.*`/`_fc.*` keys), so loads can be partially non-strict and leave the head misaligned. I add a minimal key-normalization step for common EfficientNet head key patterns (`_fc`→`classifier.1`, and `model._fc`→`model.classifier.1`) before strict-loading attempts, which should increase the fraction of correctly-loaded EfficientNet checkpoints and move accuracy upward. Everything else (paths, transforms, softmax-mean, argmax submission) stays the same.'
- What this solution (achieved 0.59118) has done: 'Your current score (0.58595) is far below the target (0.89393), so we should make small, inference-only changes that increase the chance every checkpoint is loaded into the *correct* backbone and with a correctly-mapped classifier head. I (1) expand the architecture inference to reliably detect ResNet-18/34/101 and ResNeXt-50 (common in Cassava) so we don’t instantiate the wrong model and then fall back to non-strict loading, and (2) broaden EfficientNet head key normalization to cover more real-world checkpoint conventions (including `fc.*` and `model.fc.*`), again increasing strict-load success. These keep your ensemble + TTA + softmax-mean + argmax logic unchanged and should move accuracy upward toward the target. The script still writes a valid `submission.csv` with the required columns and ordering.'
- What this solution (achieved 0.58632) has done: 'Your current score (0.59118) is far below the target (0.89393), so we should make small, inference-only fixes that increase the fraction of checkpoints that load with the correct classifier head (avoiding randomly initialized heads). I (1) expand EfficientNet head key normalization to cover more real checkpoint conventions (e.g., `classifier.weight/bias`, `head.*`, and `model.head.*`) and (2) add a minimal DenseNet head key normalization (`fc.*` → `classifier.*`) which is a common mismatch. I also add a tiny safety check that warns (but does not change logic) when a checkpoint appears not to match 5 classes, which helps catch silent non-strict loads harming accuracy. The ensemble/TTA averaging, transforms, softmax-mean, and argmax submission semantics remain unchanged.'
- What this solution (achieved 0.58707) has done: 'Your current score (0.58632) is far below the target (0.89393), so we should make small inference-only fixes that improve correctness without changing your ensemble/TTA averaging or model wrappers. The biggest likely remaining issue is that many Cassava checkpoints are saved from *wrapped* models (e.g., keys prefixed with `convlayer.`/`fc.` or `backbone.`), so your current loader often falls back to non‑strict and silently leaves randomly initialized heads/features, hurting accuracy. I add minimal key-normalization to map common wrapper prefixes to your current wrapper module names and extend architecture inference to reliably detect ResNeXt50, so more checkpoints instantiate the right backbone and strict-load successfully. Everything else (transforms list, softmax-mean, argmax, submission ordering/format) stays the same and still writes `submission.csv`.'

# 9. Code solution

## === cell 0
import numpy as np
import glob



## === cell 1
import os

CANDIDATE_PTH = []
for pat in [
    "/kaggle/input/cassava-leaf-disease-classification/**/*.pth",
    "../input/cassava-leaf-disease-classification/**/*.pth",
    "/kaggle/input/**/*.pth",  # keep as a secondary fallback
    "../input/**/*.pth",
]:
    CANDIDATE_PTH.extend(glob.glob(pat, recursive=True))

preferred_keywords = [
    "cassava",
    "densenet201",
    "eb7",
    "efficientnet",
    "resnext",
    "resnet",
    "leaf",
]
preferred = [
    p for p in CANDIDATE_PTH if any(k in p.lower() for k in preferred_keywords)
]
pretrained_models = sorted(set(preferred if len(preferred) > 0 else CANDIDATE_PTH))

print(f"{len(pretrained_models)} models found.")
if len(pretrained_models) > 0:
    print("\n".join(pretrained_models[:50]))
    if len(pretrained_models) > 50:
        print(f"... ({len(pretrained_models)-50} more)")



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
SIZE = 512
num_classes = 5



## === cell 5
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用デバイス: {device}")



## === cell 6
DEFAULT_BASE_DIR = "../input/cassava-leaf-disease-classification"
if not os.path.isdir(DEFAULT_BASE_DIR):
    DEFAULT_BASE_DIR = "/kaggle/input/cassava-leaf-disease-classification"

BASE_DIR = DEFAULT_BASE_DIR

test_dir = f"{BASE_DIR}/test_images"
train_dir = f"{BASE_DIR}/train_images"

if os.path.isdir(test_dir):
    TEST_PATH = test_dir
else:
    TEST_PATH = train_dir  # local/debug fallback only

sample_sub_path = f"{BASE_DIR}/sample_submission.csv"
if os.path.isfile(sample_sub_path) and os.path.isdir(test_dir):
    df_test = pd.read_csv(sample_sub_path)
    test_files = df_test["image_id"].tolist()
else:
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    test_files = [
        f
        for f in os.listdir(TEST_PATH)
        if os.path.splitext(f.lower())[1] in exts
        and os.path.isfile(os.path.join(TEST_PATH, f))
    ]
    df_test = pd.DataFrame(test_files, columns=["image_id"])
    df_test["label"] = 1

print(f"BASE_DIR={BASE_DIR}")
print(f"TEST_PATH={TEST_PATH}")
print(f"Number of test images: {len(test_files)}")
print(df_test.head())



## === cell 7
if len(df_test) == 1:
    df_test.loc[1] = df_test.loc[0]
    print(df_test)



## === cell 8
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

transform = {
    "test": [
        Compose(
            [
                A.Resize(height=SIZE, width=SIZE, interpolation=cv2.INTER_LINEAR),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                A.HorizontalFlip(p=1.0),
                A.Resize(height=SIZE, width=SIZE, interpolation=cv2.INTER_LINEAR),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                A.VerticalFlip(p=1.0),
                A.Resize(height=SIZE, width=SIZE, interpolation=cv2.INTER_LINEAR),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                A.Rotate(limit=30, p=1.0),
                A.Resize(height=SIZE, width=SIZE, interpolation=cv2.INTER_LINEAR),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
    ]
}




## === cell 9
class FinalLayerMixupModel(nn.Module):
    def __init__(self, model, criterion):
        super(FinalLayerMixupModel, self).__init__()
        self.convlayer = torch.nn.Sequential(*(list(model.children())[:-1]))
        self.fc = model.fc
        self.criterion = criterion

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

        print("ここにきてはいけない")
        sys.exit()




## === cell 10
class FinalLayerMixupModelDenseNet(nn.Module):
    def __init__(self, model, criterion):
        super(FinalLayerMixupModelDenseNet, self).__init__()
        self.convlayer = model.features
        self.AdaptiveAvgPool2d = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = model.classifier
        self.criterion = criterion

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

        print("ここにきてはいけない")
        sys.exit()




## === cell 11
class FinalLayerMixupModelEN(nn.Module):
    def __init__(self, model, criterion):
        super(FinalLayerMixupModelEN, self).__init__()
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




## === cell 12
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
            raise FileNotFoundError(f"Image not found or unreadable: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        img = self.load_image(image_id)
        if self.transform:
            img = self.transform(image=img)["image"]
        return img, image_id




## === cell 13
def predict_model(basename, net, dataloader):
    model_start_time = time.time()

    net.to(device)
    net.eval()
    torch.set_grad_enabled(False)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    probability = []

    for phase in ["test"]:
        progress = tqdm(dataloader[phase], desc=f"{basename}: ")
        for inputs, image_ids in progress:
            inputs = inputs.to(device)
            dummy_labels = torch.zeros(
                inputs.size(0), dtype=torch.long, device=inputs.device
            )
            outputs = net(inputs, dummy_labels, "test")
            probability.append(torch.softmax(outputs, dim=1).cpu().numpy())

    print(f"{basename} time: {time.time() - model_start_time:.2f}[sec]")
    return np.concatenate(probability, axis=0)




## === cell 14
def _extract_state_dict(obj):
    if (
        isinstance(obj, dict)
        and "state_dict" in obj
        and isinstance(obj["state_dict"], dict)
    ):
        return obj["state_dict"]
    return obj


def _strip_module_prefix(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict
    if any(k.startswith("module.") for k in state_dict.keys()):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict


def _infer_arch_from_state_dict_keys(state):
    if not isinstance(state, dict) or len(state) == 0:
        return None
    keys = list(state.keys())
    keyset = " ".join(keys[:1200])

    if "features.denseblock" in keyset or "features.transition" in keyset:
        return "densenet201"

    if ("features.0.0.weight" in keyset or "features.0.1.weight" in keyset) and (
        "classifier.1.weight" in keyset
        or "classifier.1.bias" in keyset
        or "model.classifier.1.weight" in keyset
        or "model.classifier.1.bias" in keyset
        or "classifier.0.weight" in keyset
        or "classifier.0.bias" in keyset
        or "model.classifier.0.weight" in keyset
        or "model.classifier.0.bias" in keyset
        or "_fc.weight" in keyset
        or "_fc.bias" in keyset
        or "model._fc.weight" in keyset
        or "model._fc.bias" in keyset
        or "fc.weight" in keyset
        or "fc.bias" in keyset
        or "model.fc.weight" in keyset
        or "model.fc.bias" in keyset
        or "head.weight" in keyset
        or "head.bias" in keyset
        or "model.head.weight" in keyset
        or "model.head.bias" in keyset
        or "classifier.weight" in keyset
        or "classifier.bias" in keyset
        or "model.classifier.weight" in keyset
        or "model.classifier.bias" in keyset
    ):
        return "efficientnet-b7"

    if ("layer3.22.conv3.weight" in keyset or "layer3.23.conv3.weight" in keyset) and (
        "layer4.2.conv3.weight" in keyset
    ):
        return "resnext101"

    if ("layer3.5.conv3.weight" in keyset) and ("layer4.2.conv3.weight" in keyset):
        if "layer1.0.conv3.weight" in keyset or "layer2.0.conv3.weight" in keyset:
            return "resnext50"
        return "resnet50"

    if "layer3.35.conv3.weight" in keyset and "layer4.2.conv3.weight" in keyset:
        return "resnet152"

    if "layer3.5.conv2.weight" in keyset and "layer4.1.conv2.weight" in keyset:
        return "resnet18"
    if "layer3.5.conv2.weight" in keyset and "layer4.2.conv2.weight" in keyset:
        return "resnet34"
    if "layer3.22.conv3.weight" in keyset and "layer4.2.conv3.weight" in keyset:
        return "resnet101"

    if "fc.weight" in keyset and "layer1.0.conv1.weight" in keyset:
        return "resnet50"

    return None


def _set_head_to_num_classes_resnet(model, num_classes):
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def _set_head_to_num_classes_densenet(model, num_classes):
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)
    return model


def _set_head_to_num_classes_efficientnet_b7(model, num_classes):
    if isinstance(model.classifier, nn.Sequential) and len(model.classifier) > 0:
        last = model.classifier[-1]
        if isinstance(last, nn.Linear):
            in_features = last.in_features
            model.classifier[-1] = nn.Linear(in_features, num_classes)
            return model
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5, inplace=True), nn.Linear(2560, num_classes)
    )
    return model


def _remap_state_dict_keys_for_target(state, target_prefix):
    if not isinstance(state, dict) or len(state) == 0:
        return state
    remapped = {}
    for k, v in state.items():
        nk = k
        if target_prefix == "no_model_prefix":
            if nk.startswith("model."):
                nk = nk[len("model.") :]
        elif target_prefix == "add_model_prefix":
            if not nk.startswith("model."):
                nk = "model." + nk
        remapped[nk] = v
    return remapped


def _normalize_densenet_fc_keys(state):
    """
    Change is accuracy-relevant: some DenseNet cassava checkpoints save the final head
    as `fc.*` even though torchvision DenseNet uses `classifier.*`. Remapping helps
    strict-loading succeed and avoids random head weights.
    """
    if not isinstance(state, dict) or len(state) == 0:
        return state
    out = {}
    for k, v in state.items():
        nk = k
        if nk == "fc.weight":
            nk = "classifier.weight"
        elif nk == "fc.bias":
            nk = "classifier.bias"
        elif nk == "model.fc.weight":
            nk = "model.classifier.weight"
        elif nk == "model.fc.bias":
            nk = "model.classifier.bias"
        out[nk] = v
    return out


def _normalize_efficientnet_fc_keys(state):
    """
    Change is accuracy-relevant: expands key mapping for common EfficientNet head names
    so strict-loading succeeds more often (avoids random classifier head).
    """
    if not isinstance(state, dict) or len(state) == 0:
        return state
    out = {}
    for k, v in state.items():
        nk = k

        if nk == "_fc.weight":
            nk = "classifier.1.weight"
        elif nk == "_fc.bias":
            nk = "classifier.1.bias"
        elif nk == "model._fc.weight":
            nk = "model.classifier.1.weight"
        elif nk == "model._fc.bias":
            nk = "model.classifier.1.bias"

        elif nk == "fc.weight":
            nk = "classifier.1.weight"
        elif nk == "fc.bias":
            nk = "classifier.1.bias"
        elif nk == "model.fc.weight":
            nk = "model.classifier.1.weight"
        elif nk == "model.fc.bias":
            nk = "model.classifier.1.bias"

        elif nk == "classifier.weight":
            nk = "classifier.1.weight"
        elif nk == "classifier.bias":
            nk = "classifier.1.bias"
        elif nk == "model.classifier.weight":
            nk = "model.classifier.1.weight"
        elif nk == "model.classifier.bias":
            nk = "model.classifier.1.bias"

        elif nk == "head.weight":
            nk = "classifier.1.weight"
        elif nk == "head.bias":
            nk = "classifier.1.bias"
        elif nk == "model.head.weight":
            nk = "model.classifier.1.weight"
        elif nk == "model.head.bias":
            nk = "model.classifier.1.bias"

        out[nk] = v
    return out


def _normalize_wrapper_prefix_keys(state):
    """
    Change is accuracy-relevant: many saved Cassava checkpoints come from wrapper models
    and store weights under prefixes like `convlayer.*`, `fc.*`, `backbone.*` or `encoder.*`.
    Mapping these to this notebook's wrapper attribute names increases strict-load success
    (avoids random weights due to non-strict loading).
    """
    if not isinstance(state, dict) or len(state) == 0:
        return state

    out = {}
    for k, v in state.items():
        nk = k

        if nk.startswith("backbone."):
            nk = "model." + nk[len("backbone.") :]

        if nk.startswith("encoder."):
            nk = "convlayer." + nk[len("encoder.") :]

        if nk.startswith("net."):
            nk = nk[len("net.") :]

        out[nk] = v
    return out


def _warn_if_head_shape_mismatch(state, basename, num_classes_expected=5):
    """
    Change is score-relevant (debug): warns when a checkpoint head doesn't look like
    it targets 5 classes, which often implies a load mismatch / wrong backbone.
    Does not alter execution.
    """
    if not isinstance(state, dict):
        return
    for k in [
        "fc.weight",
        "classifier.weight",
        "classifier.1.weight",
        "_fc.weight",
        "head.weight",
        "model.fc.weight",
        "model.classifier.weight",
        "model.classifier.1.weight",
        "model._fc.weight",
        "model.head.weight",
    ]:
        w = state.get(k, None)
        if isinstance(w, torch.Tensor) and w.ndim == 2:
            out_dim = int(w.shape[0])
            if out_dim != num_classes_expected:
                print(
                    f"WARNING: {basename} head out_dim={out_dim} (expected {num_classes_expected}). "
                    f"This may indicate a mismatched architecture/head mapping and can hurt accuracy."
                )
            return


def _load_checkpoint_into_net(net, state, basename):
    state = _strip_module_prefix(state)

    state = _normalize_wrapper_prefix_keys(state)

    state = _normalize_densenet_fc_keys(state)
    state = _normalize_efficientnet_fc_keys(state)
    _warn_if_head_shape_mismatch(state, basename, num_classes_expected=num_classes)

    candidates = []
    candidates.append(("as_is", state))
    candidates.append(
        (
            "strip_model_prefix",
            _remap_state_dict_keys_for_target(state, "no_model_prefix"),
        )
    )
    candidates.append(
        (
            "add_model_prefix",
            _remap_state_dict_keys_for_target(state, "add_model_prefix"),
        )
    )

    target_modules = []
    if hasattr(net, "model"):
        target_modules.append(("net.model", net.model))
    target_modules.append(("net", net))

    for variant_name, st in candidates:
        for name, m in target_modules:
            try:
                m.load_state_dict(st, strict=True)
                print(
                    f"{basename}: strict load OK into {name} (variant={variant_name})"
                )
                return True
            except RuntimeError:
                continue

    name, m = target_modules[0]
    missing, unexpected = m.load_state_dict(state, strict=False)
    print(
        f"WARNING: Non-strict load used for {basename} into {name}. missing={len(missing)} unexpected={len(unexpected)}"
    )
    return False


if len(pretrained_models) == 0:
    print(
        "WARNING: No pretrained .pth models found. Falling back to torchvision resnet18 weights with a 5-class head for a valid submission."
    )
    pretrained_models = ["__TORCHVISION_RESNET18_FALLBACK__"]

probability = []
start_time = time.time()

for pretrained_model in pretrained_models:
    basename = (
        os.path.splitext(os.path.basename(pretrained_model))[0]
        if pretrained_model != "__TORCHVISION_RESNET18_FALLBACK__"
        else "resnet18_fallback"
    )

    criterion = nn.CrossEntropyLoss()

    inferred_model_name = None
    state = None
    if pretrained_model != "__TORCHVISION_RESNET18_FALLBACK__":
        raw = torch.load(pretrained_model, map_location="cpu")
        state = _extract_state_dict(raw)
        state = _strip_module_prefix(state)
        inferred_model_name = _infer_arch_from_state_dict_keys(state)

    model_hint = inferred_model_name if inferred_model_name is not None else None

    if pretrained_model == "__TORCHVISION_RESNET18_FALLBACK__":
        MODEL_NAME = "resnet18"
        net_base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        net_base = _set_head_to_num_classes_resnet(net_base, num_classes)
        net = FinalLayerMixupModel(net_base, criterion)
        BATCH_SIZE = 64

    elif (model_hint == "densenet201") or ("densenet201" in basename):
        MODEL_NAME = "densenet201"
        net_base = models.densenet201(weights=None)
        net_base = _set_head_to_num_classes_densenet(net_base, num_classes)
        net = FinalLayerMixupModelDenseNet(net_base, criterion)
        BATCH_SIZE = 12

    elif (
        (model_hint == "efficientnet-b7")
        or ("efficientnet-b7" in basename)
        or ("eb7" in basename.lower())
    ):
        MODEL_NAME = "efficientnet-b7"
        net_base = efficientnet_b7(weights=None)
        net_base = _set_head_to_num_classes_efficientnet_b7(net_base, num_classes)
        net = FinalLayerMixupModelEN(net_base, criterion)
        BATCH_SIZE = 10

    elif (model_hint == "resnet152") or ("resnet152" in basename):
        MODEL_NAME = "resnet152"
        net_base = models.resnet152(weights=None)
        net_base = _set_head_to_num_classes_resnet(net_base, num_classes)
        net = FinalLayerMixupModel(net_base, criterion)
        BATCH_SIZE = 16

    elif (model_hint == "resnext101") or ("resnext101" in basename):
        MODEL_NAME = "resnext101"
        net_base = models.resnext101_32x8d(weights=None)
        net_base = _set_head_to_num_classes_resnet(net_base, num_classes)
        net = FinalLayerMixupModel(net_base, criterion)
        BATCH_SIZE = 12

    elif (
        (model_hint == "resnext50")
        or ("resnext50" in basename)
        or ("32x4d" in basename)
    ):
        MODEL_NAME = "resnext50"
        net_base = models.resnext50_32x4d(weights=None)
        net_base = _set_head_to_num_classes_resnet(net_base, num_classes)
        net = FinalLayerMixupModel(net_base, criterion)
        BATCH_SIZE = 24

    elif (model_hint == "resnet101") or ("resnet101" in basename):
        MODEL_NAME = "resnet101"
        net_base = models.resnet101(weights=None)
        net_base = _set_head_to_num_classes_resnet(net_base, num_classes)
        net = FinalLayerMixupModel(net_base, criterion)
        BATCH_SIZE = 16

    elif (model_hint == "resnet34") or ("resnet34" in basename):
        MODEL_NAME = "resnet34"
        net_base = models.resnet34(weights=None)
        net_base = _set_head_to_num_classes_resnet(net_base, num_classes)
        net = FinalLayerMixupModel(net_base, criterion)
        BATCH_SIZE = 48

    elif (model_hint == "resnet18") or ("resnet18" in basename):
        MODEL_NAME = "resnet18"
        net_base = models.resnet18(weights=None)
        net_base = _set_head_to_num_classes_resnet(net_base, num_classes)
        net = FinalLayerMixupModel(net_base, criterion)
        BATCH_SIZE = 64

    elif (
        (model_hint == "resnet50") or ("resnet50" in basename) or ("resnet" in basename)
    ):
        MODEL_NAME = "resnet50"
        net_base = models.resnet50(weights=None)
        net_base = _set_head_to_num_classes_resnet(net_base, num_classes)
        net = FinalLayerMixupModel(net_base, criterion)
        BATCH_SIZE = 32

    else:
        print(
            f"{basename} is not supported and could not infer architecture from checkpoint keys."
        )
        sys.exit()

    print(f"{basename}: {MODEL_NAME}")

    if pretrained_model != "__TORCHVISION_RESNET18_FALLBACK__":
        _load_checkpoint_into_net(net, state, basename)

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
                pin_memory=True,
            )
        }
        proba = predict_model(basename, net, dataloader)
        probability.append(proba)

    del net
    torch.cuda.empty_cache()

prob_stack = np.stack(probability, axis=0)
df_test["mean"] = prob_stack.mean(axis=0).argmax(axis=1)

print(f"total time: {time.time() - start_time:.2f}[sec]")



## === cell 15
df_test["label"] = df_test["mean"].astype(int)



## === cell 16
df_test



## === cell 17
sub = df_test[["image_id", "label"]].copy()
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv", sub.shape)
print(sub.head())
