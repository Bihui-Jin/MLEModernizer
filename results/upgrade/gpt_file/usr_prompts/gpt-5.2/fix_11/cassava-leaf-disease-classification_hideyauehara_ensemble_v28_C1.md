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

0.8921124206708976

# 6. Current score

0.4716

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.05531) has done: 'I remove the hard dependency on `efficientnet_pytorch` (not installed) by skipping EfficientNet checkpoints if any are present, so inference can proceed with the available torchvision models. I also fix the Kaggle path detection so it always finds `../input/cassava-leaf-disease-classification/test_images` in this environment, avoiding the local `data/train_images` FileNotFoundError. Next, I update the Albumentations `RandomResizedCrop` calls to the v2 API (requires `size=(H,W)`), which currently crashes. Finally, I make the ensemble aggregation robust and always write a valid `submission.csv` with the correct columns, even if no checkpoints are found (fallback to a safe baseline).'
- What this solution (achieved 0.05531) has done: 'Your current 0.05531 score is consistent with “no checkpoints found → fallback all-zeros labels” (or effectively random), so the smallest change that can move you toward the 0.892 target is to actually load usable pretrained .pth models by fixing the input path discovery for those model files. I keep your model wrappers, inference loop, TTA, and averaging exactly as-is, but expand the checkpoint globbing to search the real dataset/input directories available in this environment (including nested folders). I also add a deterministic safeguard for the Albumentations `RandomResizedCrop` v2 signature by explicitly setting `scale`/`ratio` so it behaves stably across versions. This should raise the score substantially (toward your target) without changing the core modeling logic.'
- What this solution (achieved 0.05531) has done: 'Your current score (0.05531) is consistent with the code not finding any real `.pth` checkpoints, so it falls back to predicting a constant label (near-random accuracy). To move the score toward the 0.892 target with minimal logic changes, I (1) expand checkpoint discovery to include common Kaggle “Dataset” subfolders like `/kaggle/input/**/` (including `*.pt` as well), and (2) make test image path selection robust by checking both `test_images` and nested `cassava-leaf-disease-classification/test_images` under each candidate base. I also fix a subtle inference shape bug (`squeeze()` can drop the batch dimension when batch_size=1) that can silently ruin predictions for the last batch, without changing the model architecture. Everything else (models, TTA, averaging, submission format) stays the same.'
- What this solution (achieved 0.05531) has done: 'Your 0.05531 score strongly suggests your inference is effectively untrained because the loaded checkpoints’ classifier weights don’t match the instantiated model heads (you create new `fc`/`classifier` layers but then load a full state_dict, which silently fails or loads incorrectly depending on the saved keys). The smallest score-improving change (without altering architecture/loops/transforms) is to load checkpoint weights into the correct submodules: map common checkpoint key prefixes (`module.`, `model.`), and if the checkpoint contains only the backbone, load it into `convlayer`/`features` while separately loading `fc` when present. I also add an explicit “best-effort” loader that errors loudly if nothing was loaded (instead of producing near-random predictions), which should move accuracy sharply toward your target if the `.pth` files are valid cassava-trained weights. Everything else—model choices, TTA list, softmax-mean-argmax, and submission format—remains the same.'
- What this solution (achieved 0.10987) has done: 'Your low score is consistent with the “no usable checkpoints / incompatible weights” path, which makes predictions effectively constant or random; the smallest way to move toward the 0.892 target is to ensure we actually use a strong, compatible model at inference. I keep your wrappers, TTA list, softmax-mean-argmax aggregation, and submission writing exactly the same, but add a safe fallback: if no checkpoint produces valid predictions, run a single torchvision ImageNet-pretrained ResNet50 with the same inference loop (no training) so accuracy moves up substantially instead of staying near 0. I also fix a small bug in the fallback baseline (it currently sets `mean` to 0 instead of label 0), and make checkpoint compatibility checking stricter so we don’t accidentally load mismatched heads and ruin predictions. These changes are minimal, run fast, and preserve your evaluation semantics.'
- What this solution (achieved 0.74776) has done: 'Your current 0.10987 is far below the 0.892 target, and the biggest “minimal-change” win is to stop producing near-random outputs from an untrained head: the fallback currently uses an ImageNet-pretrained backbone but a randomly initialized cassava classifier layer. I keep your exact inference loop, TTAs, averaging, and wrapper classes, but add a tiny, legitimate calibration step for the fallback only: fit the final `fc` layer (and only that layer) on `train.csv` using frozen ResNet50 features, then use it for test inference. This preserves the same architecture and loss, doesn’t change evaluation semantics, and should move accuracy dramatically toward the target band while staying within time limits by using a small, deterministic number of steps and only training the linear head. I also make the checkpoint discovery prefer cassava-trained checkpoints if present, but the main improvement is ensuring the fallback is actually trained for 5-class cassava.'
- What this solution (achieved 0.74776) has done: 'Your current score (0.74776) is below the target (0.89211), so we should improve accuracy with the smallest, lowest-risk changes that don’t alter the core modeling/inference logic. The biggest likely issue is that your test-time preprocessing uses `CenterCrop(512,512)` and `RandomResizedCrop(512,512)` without first resizing, which distort/clip many cassava images (most are smaller than 512 on one side), hurting accuracy; we add a deterministic `LongestMaxSize` + `PadIfNeeded` before any crop so inputs match training-like assumptions. Separately, your normalization stats look non-standard; for the fallback ResNet50 (ImageNet backbone), switching to ImageNet mean/std (and using bicubic resize interpolation) typically yields a material accuracy lift while keeping the same architecture, loss, and inference aggregation. These changes should move the score upward toward the target while preserving your ensemble/TTA/softmax-mean-argmax semantics and still producing a valid `submission.csv`.'
- What this solution (achieved 0.65359) has done: 'Your score (0.74776) is well below the target (0.89211), so we should make a small, low-risk improvement that legitimately increases accuracy without changing the model architecture or inference semantics. The most likely easy win is that the fallback head fitting is too weak/unbalanced: it trains only 300 SGD steps on randomly shuffled data, so it underfits and is biased toward majority classes. I keep the exact same fallback model (ResNet50 convlayer frozen + linear fc trained with CrossEntropyLoss) and the same inference/TTA/mean-argmax aggregation, but (1) switch the fallback head optimizer to Adam (faster/stabler convergence for a linear head) and (2) add standard class weights computed from `train.csv` to reduce imbalance bias. These are minimal changes confined to the fallback training path and should move accuracy upward toward your target band.'
- What this solution (achieved 0.74028) has done: 'Your current score (0.65359) is far below the target (0.89211), so we should increase accuracy with the smallest, lowest-risk changes that keep your model/inference logic intact. The biggest improvement lever here is the fallback head fitting: it’s training only 300 mini-batches on random crops, so it underfits; increasing the number of steps and using a slightly more stable AdamW setup (still training only the final `fc`) should move you closer to the target without changing architecture, loss, or inference semantics. Separately, your fallback training uses random resized crops but your test pipeline heavily uses center crop; adding a deterministic center-crop pass during the fallback `fc` fitting (same images, same frozen backbone) improves train/test transform alignment with minimal change. Finally, I keep your ensemble/TTA/softmax-mean-argmax exactly as-is and still write a valid `submission.csv`.'
- What this solution (achieved 0.4716) has done: 'Your current score (0.74028) is well below the target (0.89211), so we should make a small change that reliably increases accuracy without changing your model architecture, loss, training loop structure, or ensemble semantics. The safest improvement lever here is the fallback head fitting: right now it trains the `fc` on heavily augmented random crops and separately on center crops, but it still uses train-time randomness and class imbalance may be under-addressed at the sampler level. I keep the exact “freeze convlayer + train only fc with CrossEntropyLoss + AdamW for fixed steps” approach, but add a `WeightedRandomSampler` (using `train.csv` label frequencies) so the fc head sees a balanced stream and converges better within the same number of steps. This typically gives a noticeable accuracy lift for cassava with minimal risk and still finishes within the time budget, while producing the same valid `submission.csv`.'

# 9. Code solution

## === cell 0
import numpy as np
import glob



## === cell 1
import os
from pathlib import Path

CANDIDATE_CKPT_ROOTS = [
    "../input",  # standard Kaggle notebooks
    "/kaggle/input",  # standard Kaggle
    "/kaggle/data",  # this environment shows data under /kaggle/data
    "/kaggle/data/input",  # mirror
    "/kaggle/working",  # working dir mirror
]

ckpt_patterns = []
for root in CANDIDATE_CKPT_ROOTS:
    if os.path.exists(root):
        for ext in ["pth", "pt"]:
            ckpt_patterns += [
                str(Path(root) / f"*.{ext}"),
                str(Path(root) / f"*/*.{ext}"),
                str(Path(root) / f"*/*/*.{ext}"),
                str(Path(root) / f"*/*/*/*.{ext}"),
                str(Path(root) / f"*/*/*/*/*.{ext}"),
                str(Path(root) / f"*/*/*/*/*/*.{ext}"),
                str(Path(root) / f"*/*/*/*/*/*/*.{ext}"),
            ]

pretrained_models = []
for pat in ckpt_patterns:
    pretrained_models.extend(glob.glob(pat))

pretrained_models = sorted(list(dict.fromkeys(pretrained_models)))

print(f"{len(pretrained_models)} models found.")
if len(pretrained_models) > 0:
    print("\n".join(pretrained_models[:50]))



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
    torch.backends.cudnn.benchmark = True


SEED = 42
seed_everything(seed=SEED)



## === cell 3
EFFNET_AVAILABLE = False
EfficientNet = None
try:
    from efficientnet_pytorch import EfficientNet  # type: ignore

    EFFNET_AVAILABLE = True
except ModuleNotFoundError:
    EFFNET_AVAILABLE = False
    EfficientNet = None
    print(
        "Warning: efficientnet_pytorch not available. Any EfficientNet checkpoints will be skipped."
    )



## === cell 4
SIZE = 512
num_classes = 5



## === cell 5
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用デバイス: {device}")



## === cell 6
CANDIDATE_BASE_DIRS = [
    "../input/cassava-leaf-disease-classification",
    "/kaggle/input/cassava-leaf-disease-classification",
    "../input/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
    "/kaggle/data/cassava-leaf-disease-classification",
    "/kaggle/data/input/cassava-leaf-disease-classification",
]

BASE_DIR = None
TEST_PATH = None

for base in CANDIDATE_BASE_DIRS:
    if not os.path.exists(base):
        continue

    cand1 = os.path.join(base, "test_images")
    cand2 = os.path.join(base, "cassava-leaf-disease-classification", "test_images")
    if os.path.exists(cand1):
        BASE_DIR = base
        TEST_PATH = cand1
        break
    if os.path.exists(cand2):
        BASE_DIR = base
        TEST_PATH = cand2
        break

if BASE_DIR is None:
    BASE_DIR = "/kaggle/data/cassava-leaf-disease-classification"
    cand1 = os.path.join(BASE_DIR, "test_images")
    cand2 = os.path.join(BASE_DIR, "cassava-leaf-disease-classification", "test_images")
    TEST_PATH = cand1 if os.path.exists(cand1) else cand2

test_files = sorted([f for f in os.listdir(TEST_PATH) if f.lower().endswith(".jpg")])

print(f"BASE_DIR: {BASE_DIR}")
print(f"TEST_PATH: {TEST_PATH}")
print(f"Number of test images: {len(test_files)}")



## === cell 7
df_test = pd.DataFrame(test_files, columns=["image_id"])
df_test["label"] = 1



## === cell 8
if len(df_test) == 1:
    df_test.loc[1] = df_test.loc[0]
    print(df_test)



## === cell 9
mean = [0.430, 0.497, 0.313]
std = [0.238, 0.240, 0.228]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

pre_crop = [
    A.LongestMaxSize(max_size=SIZE, interpolation=cv2.INTER_CUBIC, p=1.0),
    A.PadIfNeeded(
        min_height=SIZE,
        min_width=SIZE,
        border_mode=cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
        p=1.0,
    ),
]

transform = {
    "test": [
        Compose(
            pre_crop
            + [
                A.CenterCrop(SIZE, SIZE),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            pre_crop
            + [
                A.HorizontalFlip(p=1.0),
                A.CenterCrop(SIZE, SIZE),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            pre_crop
            + [
                A.RandomResizedCrop(
                    size=(SIZE, SIZE), scale=(0.8, 1.0), ratio=(0.75, 1.33), p=1.0
                ),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            pre_crop
            + [
                A.RandomResizedCrop(
                    size=(SIZE, SIZE), scale=(0.8, 1.0), ratio=(0.75, 1.33), p=1.0
                ),
                A.HorizontalFlip(p=1.0),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            pre_crop
            + [
                A.RandomResizedCrop(
                    size=(SIZE, SIZE), scale=(0.8, 1.0), ratio=(0.75, 1.33), p=1.0
                ),
                A.VerticalFlip(p=1.0),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            pre_crop
            + [
                A.Rotate(p=1.0),
                A.RandomResizedCrop(
                    size=(SIZE, SIZE), scale=(0.8, 1.0), ratio=(0.75, 1.33), p=1.0
                ),
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
        img = cv2.imread(f"{TEST_PATH}/{image_id}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        img = self.load_image(image_id)
        if self.transform:
            img = self.transform(image=img)["image"]
        return img, image_id




## === cell 16
class TrainDataset(data.Dataset):
    def __init__(self, df, img_dir, transform=None):
        super().__init__()
        self.image_ids = df.image_id.tolist()
        self.labels = df.label.astype(int).tolist()
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        label = self.labels[index]
        img = cv2.imread(os.path.join(self.img_dir, image_id))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(image=img)["image"]
        return img, torch.tensor(label, dtype=torch.long)




## === cell 17
def _extract_state_dict(obj):
    if isinstance(obj, dict):
        for k in ["state_dict", "model", "model_state_dict", "net", "weights"]:
            if k in obj and isinstance(obj[k], dict):
                return obj[k]
        if any(isinstance(v, torch.Tensor) for v in obj.values()):
            return obj
    return obj


def _strip_prefix(state_dict, prefixes=("module.", "model.", "net.")):
    if not isinstance(state_dict, dict):
        return state_dict
    out = {}
    for k, v in state_dict.items():
        nk = k
        for p in prefixes:
            if nk.startswith(p):
                nk = nk[len(p) :]
        out[nk] = v
    return out


def load_checkpoint_into_wrapper(net, state, model_name, basename=""):
    state = _extract_state_dict(state)
    if not isinstance(state, dict):
        raise ValueError(f"{basename}: checkpoint is not a state_dict-like dict.")

    state = _strip_prefix(state)

    if model_name == "efficientnet-b7":
        missing, unexpected = net.model.load_state_dict(state, strict=False)
        return missing, unexpected

    if isinstance(net, FinalLayerMixupModel):
        if any(k.startswith("convlayer.") or k.startswith("fc.") for k in state.keys()):
            missing, unexpected = net.load_state_dict(state, strict=False)
            return missing, unexpected

        conv_sd = {}
        fc_sd = {}
        for k, v in state.items():
            if k.startswith("fc."):
                fc_sd[k.replace("fc.", "", 1)] = v
            else:
                conv_sd[k] = v

        missing1, unexpected1 = net.convlayer.load_state_dict(conv_sd, strict=False)
        missing2, unexpected2 = (
            net.fc.load_state_dict(fc_sd, strict=False) if len(fc_sd) else ([], [])
        )
        return list(missing1) + list(missing2), list(unexpected1) + list(unexpected2)

    if isinstance(net, FinalLayerMixupModelDenseNet):
        if any(k.startswith("convlayer.") or k.startswith("fc.") for k in state.keys()):
            missing, unexpected = net.load_state_dict(state, strict=False)
            return missing, unexpected

        feat_sd = {}
        fc_sd = {}
        for k, v in state.items():
            if k.startswith("features."):
                feat_sd[k.replace("features.", "", 1)] = v
            elif k.startswith("classifier."):
                fc_sd[k.replace("classifier.", "", 1)] = v
            elif k.startswith("fc."):
                fc_sd[k.replace("fc.", "", 1)] = v
            else:
                pass

        missing1, unexpected1 = net.convlayer.load_state_dict(feat_sd, strict=False)
        missing2, unexpected2 = (
            net.fc.load_state_dict(fc_sd, strict=False) if len(fc_sd) else ([], [])
        )
        return list(missing1) + list(missing2), list(unexpected1) + list(unexpected2)

    missing, unexpected = net.load_state_dict(state, strict=False)
    return missing, unexpected


def predict_model(basename, net, dataloader):
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




## === cell 18
def fit_fallback_fc_only(net, train_loader, criterion, max_steps=300, lr=1e-3):
    net.to(device)
    net.train()

    for p in net.convlayer.parameters():
        p.requires_grad = False
    for p in net.fc.parameters():
        p.requires_grad = True

    optimizer = torch.optim.AdamW(net.fc.parameters(), lr=lr, weight_decay=1e-4)

    step = 0
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            feats = net.convlayer(inputs)
            feats = torch.flatten(feats, 1)
        logits = net.fc(feats)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        step += 1
        if step >= max_steps:
            break

    net.eval()
    return running_loss / max(1, step)




## === cell 19
probability = []
start_time = time.time()

usable_models = []
for pretrained_model in pretrained_models:
    base = os.path.splitext(os.path.basename(pretrained_model))[0]
    if "efficientnet-b7" in base and not EFFNET_AVAILABLE:
        print(f"Skipping {base} (efficientnet_pytorch not available).")
        continue
    usable_models.append(pretrained_model)

successful_ckpt_models = 0

for pretrained_model in usable_models:
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
    elif "efficientnet-b7" in basename:
        MODEL_NAME = "efficientnet-b7"
        net = EfficientNet.from_name(MODEL_NAME)
        net = FinalLayerMixupModelEN(net, criterion, num_classes, False)
        BATCH_SIZE = 10
    else:
        print(f"{basename} is not supported. Skipping.")
        continue

    print(f"{basename}: {MODEL_NAME}")

    state = torch.load(pretrained_model, map_location="cpu")

    missing, unexpected = load_checkpoint_into_wrapper(
        net if MODEL_NAME != "efficientnet-b7" else net,
        state,
        MODEL_NAME,
        basename=basename,
    )

    raw_keys = list(_strip_prefix(_extract_state_dict(state)).keys())
    has_classifier_keys = any(
        (k.startswith("fc.") or k.startswith("_fc.") or k.startswith("classifier."))
        for k in raw_keys
    )
    if not has_classifier_keys:
        print(
            f"{basename}: checkpoint seems incompatible (no classifier keys). Skipping to avoid degrading score."
        )
        del net
        torch.cuda.empty_cache()
        continue

    successful_ckpt_models += 1

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
        proba = predict_model(basename, net, dataloader)  # (N, 5)
        probability.append(proba)

    del net
    torch.cuda.empty_cache()

if len(probability) == 0:
    print(
        "No usable checkpoints found; running a ResNet50 fallback with a fitted final fc head to move score toward target."
    )

    train_csv_path = os.path.join(BASE_DIR, "train.csv")
    train_img_dir1 = os.path.join(BASE_DIR, "train_images")
    train_img_dir2 = os.path.join(
        BASE_DIR, "cassava-leaf-disease-classification", "train_images"
    )
    TRAIN_IMG_DIR = train_img_dir1 if os.path.exists(train_img_dir1) else train_img_dir2

    df_train = pd.read_csv(train_csv_path)

    class_counts = (
        df_train["label"]
        .value_counts()
        .reindex(range(num_classes), fill_value=0)
        .values
    )
    class_counts = np.maximum(class_counts, 1)
    class_weights = (class_counts.sum() / class_counts).astype(np.float32)
    class_weights = class_weights / class_weights.mean()
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32, device=device)

    criterion = nn.CrossEntropyLoss(weight=class_weights_t)

    backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    net = FinalLayerMixupModel(backbone, criterion, num_classes, False)

    train_transform_random = Compose(
        pre_crop
        + [
            A.RandomResizedCrop(
                size=(SIZE, SIZE), scale=(0.8, 1.0), ratio=(0.75, 1.33), p=1.0
            ),
            A.HorizontalFlip(p=0.5),
            A.Normalize(
                mean=IMAGENET_MEAN, std=IMAGENET_STD, max_pixel_value=255.0, p=1.0
            ),
            ToTensorV2(p=1.0),
        ],
        p=1.0,
    )

    train_transform_center = Compose(
        pre_crop
        + [
            A.CenterCrop(SIZE, SIZE),
            A.Normalize(
                mean=IMAGENET_MEAN, std=IMAGENET_STD, max_pixel_value=255.0, p=1.0
            ),
            ToTensorV2(p=1.0),
        ],
        p=1.0,
    )

    labels_np = df_train["label"].astype(int).values
    per_class_count = np.bincount(labels_np, minlength=num_classes).astype(np.float64)
    per_class_count = np.maximum(per_class_count, 1.0)
    sample_weights = (1.0 / per_class_count[labels_np]).astype(np.float64)
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_ds_random = TrainDataset(
        df_train, img_dir=TRAIN_IMG_DIR, transform=train_transform_random
    )
    train_loader_random = torch.utils.data.DataLoader(
        train_ds_random,
        batch_size=32,
        sampler=sampler,  # balanced sampling
        shuffle=False,  # must be False when sampler is provided
        num_workers=2,
        pin_memory=True,
    )

    train_ds_center = TrainDataset(
        df_train, img_dir=TRAIN_IMG_DIR, transform=train_transform_center
    )
    train_loader_center = torch.utils.data.DataLoader(
        train_ds_center,
        batch_size=32,
        sampler=sampler,  # balanced sampling
        shuffle=False,  # must be False when sampler is provided
        num_workers=2,
        pin_memory=True,
    )

    avg_loss1 = fit_fallback_fc_only(
        net, train_loader_random, criterion, max_steps=600, lr=1e-3
    )
    avg_loss2 = fit_fallback_fc_only(
        net, train_loader_center, criterion, max_steps=600, lr=5e-4
    )
    print(
        f"fallback fc fit done, avg_loss_random={avg_loss1:.4f}, avg_loss_center={avg_loss2:.4f}"
    )

    for param in net.parameters():
        param.requires_grad = False

    fallback_test_transforms = [
        Compose(
            pre_crop
            + [
                A.CenterCrop(SIZE, SIZE),
                A.Normalize(
                    mean=IMAGENET_MEAN, std=IMAGENET_STD, max_pixel_value=255.0, p=1.0
                ),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            pre_crop
            + [
                A.HorizontalFlip(p=1.0),
                A.CenterCrop(SIZE, SIZE),
                A.Normalize(
                    mean=IMAGENET_MEAN, std=IMAGENET_STD, max_pixel_value=255.0, p=1.0
                ),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            pre_crop
            + [
                A.RandomResizedCrop(
                    size=(SIZE, SIZE), scale=(0.8, 1.0), ratio=(0.75, 1.33), p=1.0
                ),
                A.Normalize(
                    mean=IMAGENET_MEAN, std=IMAGENET_STD, max_pixel_value=255.0, p=1.0
                ),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            pre_crop
            + [
                A.RandomResizedCrop(
                    size=(SIZE, SIZE), scale=(0.8, 1.0), ratio=(0.75, 1.33), p=1.0
                ),
                A.HorizontalFlip(p=1.0),
                A.Normalize(
                    mean=IMAGENET_MEAN, std=IMAGENET_STD, max_pixel_value=255.0, p=1.0
                ),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            pre_crop
            + [
                A.RandomResizedCrop(
                    size=(SIZE, SIZE), scale=(0.8, 1.0), ratio=(0.75, 1.33), p=1.0
                ),
                A.VerticalFlip(p=1.0),
                A.Normalize(
                    mean=IMAGENET_MEAN, std=IMAGENET_STD, max_pixel_value=255.0, p=1.0
                ),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            pre_crop
            + [
                A.Rotate(p=1.0),
                A.RandomResizedCrop(
                    size=(SIZE, SIZE), scale=(0.8, 1.0), ratio=(0.75, 1.33), p=1.0
                ),
                A.Normalize(
                    mean=IMAGENET_MEAN, std=IMAGENET_STD, max_pixel_value=255.0, p=1.0
                ),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
    ]

    for tid, transform_ in enumerate(fallback_test_transforms):
        print(f"fallback transform loop={tid}")
        dataset = {"test": TestDataset(df_test, transform=transform_)}
        dataloader = {
            "test": torch.utils.data.DataLoader(
                dataset["test"],
                batch_size=32,
                shuffle=False,
                num_workers=2,
                pin_memory=True,
            )
        }
        proba = predict_model("fallback_resnet50_imagenet_fitted_fc", net, dataloader)
        probability.append(proba)

    del net
    torch.cuda.empty_cache()

prob_stack = np.stack(probability, axis=0)  # (M, N, 5)
df_test["mean"] = prob_stack.mean(axis=0).argmax(axis=1)

print(f"successful_ckpt_models: {successful_ckpt_models}")
print(f"total time: {time.time() - start_time:.2f}[sec]")



## === cell 20
if len(df_test) == 2 and df_test.loc[0, "image_id"] == df_test.loc[1, "image_id"]:
    df_test = pd.read_csv(f"{BASE_DIR}/sample_submission.csv")
else:
    df_test["label"] = df_test["mean"].astype(int)



## === cell 21
df_test.head()



## === cell 22
submission = df_test[["image_id", "label"]].copy()
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())
