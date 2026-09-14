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

0.8933212450891508

# 6. Current score

0.28774

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.11584) has done: 'I fix the missing EfficientNet dependency by falling back to torchvision’s built-in EfficientNet-B7 when `efficientnet_pytorch` isn’t available, while keeping the rest of the ensemble logic intact. I also fix the Kaggle path detection so it always finds the provided `/kaggle/input/cassava-leaf-disease-classification` dataset and correctly lists test images, avoiding the current `FileNotFoundError`. Albumentations v2 API changes be addressed by updating `RandomResizedCrop`/`Rotate` usage to the new parameter names so transforms build successfully. Finally, I make the checkpoint loading and model head wiring robust (including EfficientNet key-mapping), and ensure the probability aggregation always produces a valid `(N, 5)` array so `submission.csv` is written correctly.'
- What this solution (achieved 0.11584) has done: 'Your low score is consistent with the model weights not being found/loaded, so you’re effectively submitting near-constant/random predictions (or the sample submission fallback). The minimal fix is to point `pretrained_models` to actual weight files that exist in this environment (your current `../input/densenet201-04-2019data/*.pth` and `../input/eb7m-seed70/*.pth` folders aren’t present in the provided data tree), while keeping the exact same ensemble/TTA/prediction logic. I also make the test image listing deterministic (sorted) to avoid any possible row-order mismatch with `sample_submission.csv`. Finally, I ensure we always align predictions to the official sample submission’s `image_id` order before writing `submission.csv`, which can materially affect accuracy if ordering ever differs.'
- What this solution (achieved 0.28774) has done: 'Your current score (0.11584) is far below the target (0.8933), and the most likely reason is that your script isn’t loading any real trained weights, so it falls back to essentially constant labels (the sample submission), which scores very poorly. The minimal, score-relevant fix is to stop scanning all of `/kaggle/input/**` for random `.pth/.pt` files and instead load the competition-provided trained model checkpoints if they exist; since none are in your provided tree, the next best minimal fix is to use ImageNet-pretrained weights for the same architectures (this keeps the exact same inference/ensemble/TTA pipeline but makes predictions non-random). Additionally, your ResNet/DenseNet wrappers currently create a new `fc`/`classifier` layer that is never loaded from checkpoint (so even if you had weights, the head would be random); we fix this by wiring the wrapper head to use the base model’s existing `fc/classifier` so checkpoint heads can load. Finally, we keep your sample_submission alignment merge (good) and ensure test image selection always matches sample_submission ordering.'

# 9. Code solution

## === cell 0
import numpy as np
import glob




## === cell 1
def find_pretrained_models():
    patterns = [
        "../input/**/**/*.pth",
        "../input/**/**/*.pt",
        "/kaggle/input/**/**/*.pth",
        "/kaggle/input/**/**/*.pt",
    ]
    found = []
    for pat in patterns:
        found.extend(glob.glob(pat, recursive=True))
    found = [
        p for p in found if not p.endswith(".pth.tar") and not p.endswith(".pt.tar")
    ]
    return sorted(set(found))


pretrained_models = find_pretrained_models()

print(f"{len(pretrained_models)} model files found (pth/pt).")
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
from torchvision import models, transforms  # 学習済みモデル、画像変換
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



## === cell 3
import sys

_EFFNET_BACKEND = None
EfficientNet = None
try:
    sys.path.append("/kaggle/input/package/EfficientNet-PyTorch-1.0")
    from efficientnet_pytorch import EfficientNet as _EfficientNetPyTorch

    EfficientNet = _EfficientNetPyTorch
    _EFFNET_BACKEND = "efficientnet_pytorch"
    print("Using efficientnet_pytorch backend.")
except Exception as e:
    _EFFNET_BACKEND = "torchvision"
    print(
        f"efficientnet_pytorch not found; will use torchvision efficientnet instead. ({type(e).__name__}: {e})"
    )



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
    "../input/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
    "/kaggle/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
]
BASE_DIR = None
for p in CANDIDATE_BASE_DIRS:
    if os.path.isdir(p):
        BASE_DIR = p
        break
if BASE_DIR is None:
    BASE_DIR = "data"

run_type = os.getenv("KAGGLE_KERNEL_RUN_TYPE")
print(f"KAGGLE_KERNEL_RUN_TYPE={run_type}")

if os.path.isdir(f"{BASE_DIR}/test_images"):
    TEST_PATH = f"{BASE_DIR}/test_images"
    test_files = sorted(os.listdir(TEST_PATH))
    print("Using test_images.")
elif os.path.isdir(f"{BASE_DIR}/train_images"):
    TEST_PATH = f"{BASE_DIR}/train_images"
    test_files = sorted(os.listdir(TEST_PATH))[:32]
    print("Using train_images (subset) because test_images not found.")
else:
    raise FileNotFoundError(
        f"Neither test_images nor train_images found under BASE_DIR={BASE_DIR}"
    )

print(f"BASE_DIR={BASE_DIR}")
print(f"TEST_PATH={TEST_PATH}")
print(f"Number of test images: {len(test_files)}")



## === cell 7
df_test = pd.DataFrame(test_files, columns=["image_id"])
df_test["label"] = 1



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
                A.RandomResizedCrop(
                    size=(SIZE, SIZE),
                    scale=(0.8, 1.0),
                    ratio=(0.75, 1.3333333333),
                    p=1.0,
                ),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                A.RandomResizedCrop(
                    size=(SIZE, SIZE),
                    scale=(0.8, 1.0),
                    ratio=(0.75, 1.3333333333),
                    p=1.0,
                ),
                A.HorizontalFlip(p=1.0),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                A.RandomResizedCrop(
                    size=(SIZE, SIZE),
                    scale=(0.8, 1.0),
                    ratio=(0.75, 1.3333333333),
                    p=1.0,
                ),
                A.VerticalFlip(p=1.0),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                A.Rotate(limit=(-30, 30), p=1.0),
                A.RandomResizedCrop(
                    size=(SIZE, SIZE),
                    scale=(0.8, 1.0),
                    ratio=(0.75, 1.3333333333),
                    p=1.0,
                ),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                A.Rotate(limit=(-30, 30), p=1.0),
                A.CenterCrop(height=SIZE, width=SIZE),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
    ]
}




## === cell 10
def _clean_state_dict_keys(state_dict):
    cleaned = {}
    for k, v in state_dict.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module.") :]
        cleaned[nk] = v
    return cleaned


def load_checkpoint_safely(model_or_wrapper, ckpt_path, strict=False):
    state = torch.load(ckpt_path, map_location="cpu")
    if (
        isinstance(state, dict)
        and "state_dict" in state
        and isinstance(state["state_dict"], dict)
    ):
        state = state["state_dict"]
    if isinstance(state, dict):
        state = _clean_state_dict_keys(state)
    missing, unexpected = model_or_wrapper.load_state_dict(state, strict=strict)
    if len(missing) > 0 or len(unexpected) > 0:
        print(
            f"load_state_dict(strict={strict}) missing={len(missing)} unexpected={len(unexpected)} for {os.path.basename(ckpt_path)}"
        )
    return missing, unexpected




## === cell 11
class FinalLayerMixupModel(nn.Module):
    def __init__(self, model, criterion, num_classes, alpha):
        """
        model: 学習済みモデルを指定
        """
        super(FinalLayerMixupModel, self).__init__()
        self.convlayer = torch.nn.Sequential(*(list(model.children())[:-1]))
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        self.fc = model.fc

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

        x1 = inputs  # torch.Size([64, 3, 256, 256])
        x2 = inputs[index]  # torch.Size([64, 3, 256, 256])

        x1 = self.convlayer(x1)  # torch.Size([64, 512, 1, 1])
        x2 = self.convlayer(x2)  # torch.Size([64, 512, 1, 1])

        mixed_x = lam * x1 + (1 - lam) * x2  # torch.Size([64, 512, 1, 1])
        mixed_x = mixed_x.squeeze()  # torch.Size([64, 512])
        outputs = self.fc(mixed_x)  # torch.Size([64, 5])

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
        model.classifier = nn.Linear(num_ftrs, num_classes)
        self.fc = model.classifier

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

        x1 = inputs  # torch.Size([12, 3, 512, 512])
        x2 = inputs[index]  # torch.Size([12, 3, 512, 512])

        x1 = self.convlayer(x1)  # torch.Size([12, 1920, 16, 16])
        x2 = self.convlayer(x2)  # torch.Size([12, 1920, 16, 16])

        x1 = self.AdaptiveAvgPool2d(x1)  # torch.Size([12, 1920, 1, 1])
        x2 = self.AdaptiveAvgPool2d(x2)  # torch.Size([12, 1920, 1, 1])

        mixed_x = lam * x1 + (1 - lam) * x2  # torch.Size([64, 1920, 1, 1])
        mixed_x = mixed_x.squeeze()  # torch.Size([64, 1920])
        outputs = self.fc(mixed_x)  # torch.Size([64, 5])

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
        self.criterion = criterion
        self.alpha = alpha

        if hasattr(model, "_fc"):
            num_ftrs = model._fc.in_features
            model._fc = nn.Linear(num_ftrs, num_classes)
            self.model = model
            self._backend = "efficientnet_pytorch"
        elif hasattr(model, "classifier"):
            if isinstance(model.classifier, nn.Sequential):
                num_ftrs = model.classifier[-1].in_features
                model.classifier[-1] = nn.Linear(num_ftrs, num_classes)
            else:
                num_ftrs = model.classifier.in_features
                model.classifier = nn.Linear(num_ftrs, num_classes)
            self.model = model
            self._backend = "torchvision"
        else:
            raise AttributeError(
                "Unsupported EfficientNet model structure (no _fc or classifier)."
            )

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
def build_torchvision_efficientnet_b7(num_classes):
    m = torchvision.models.efficientnet_b7(
        weights=torchvision.models.EfficientNet_B7_Weights.IMAGENET1K_V1
    )
    return m




## === cell 15
class TestDataset(data.Dataset):
    def __init__(self, df, transform=None):
        super().__init__()

        self.image_ids = df.image_id.tolist()
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def load_image(self, image_id):
        img = cv2.imread(f"{TEST_PATH}/{image_id}")  # (H, W, C) の numpy.ndarray
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {TEST_PATH}/{image_id}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR => RGB に変換
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
    net.eval()  # 検証モード
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




## === cell 17
probability = []

start_time = time.time()

sample_sub_path = f"{BASE_DIR}/sample_submission.csv"
df_sample = pd.read_csv(sample_sub_path)
sample_image_ids = df_sample["image_id"].tolist()

df_test = pd.DataFrame({"image_id": sample_image_ids})
df_test["label"] = 1

USE_IMAGENET_FALLBACK = True

supported = []
for p in pretrained_models:
    b = os.path.splitext(os.path.basename(p))[0].lower()
    if any(
        k in b
        for k in [
            "resnet18",
            "resnet50",
            "resnet152",
            "resnext101",
            "densenet201",
            "efficientnet-b7",
            "efficientnetb7",
            "eb7",
        ]
    ):
        supported.append(p)
pretrained_models = supported
print(f"{len(pretrained_models)} supported model files after filtering.")

model_specs = []
if len(pretrained_models) > 0:
    for p in pretrained_models:
        model_specs.append(("ckpt", p))
else:
    if not USE_IMAGENET_FALLBACK:
        print("No pretrained models found; writing sample_submission.csv as fallback.")
        df_sub = df_sample.copy()
        df_sub.to_csv("submission.csv", index=False)
    else:
        model_specs = [
            ("imagenet", "resnet50"),
            ("imagenet", "densenet201"),
            ("imagenet", "efficientnet-b7"),
        ]
        print(
            "No finetuned checkpoints found; using ImageNet pretrained backbones:",
            model_specs,
        )

if "df_sub" not in globals():
    for spec_type, spec_val in model_specs:
        if spec_type == "ckpt":
            pretrained_model = spec_val
            basename = os.path.splitext(os.path.basename(pretrained_model))[0]
            basename_l = basename.lower()
        else:
            pretrained_model = None
            basename = spec_val
            basename_l = spec_val.lower()

        criterion = nn.CrossEntropyLoss()

        if "resnet18" in basename_l:
            MODEL_NAME = "resnet18"
            if spec_type == "imagenet":
                net_base = models.resnet18(
                    weights=models.ResNet18_Weights.IMAGENET1K_V1
                )
            else:
                net_base = models.resnet18(weights=None)
            net = FinalLayerMixupModel(net_base, criterion, num_classes, False)
            BATCH_SIZE = 64
        elif "resnet50" in basename_l:
            MODEL_NAME = "resnet50"
            if spec_type == "imagenet":
                net_base = models.resnet50(
                    weights=models.ResNet50_Weights.IMAGENET1K_V2
                )
            else:
                net_base = models.resnet50(weights=None)
            net = FinalLayerMixupModel(net_base, criterion, num_classes, False)
            BATCH_SIZE = 32
        elif "resnet152" in basename_l:
            MODEL_NAME = "resnet152"
            if spec_type == "imagenet":
                net_base = models.resnet152(
                    weights=models.ResNet152_Weights.IMAGENET1K_V2
                )
            else:
                net_base = models.resnet152(weights=None)
            net = FinalLayerMixupModel(net_base, criterion, num_classes, False)
            BATCH_SIZE = 16
        elif "resnext101" in basename_l:
            MODEL_NAME = "resnext101"
            if spec_type == "imagenet":
                net_base = models.resnext101_32x8d(
                    weights=models.ResNeXt101_32X8D_Weights.IMAGENET1K_V2
                )
            else:
                net_base = models.resnext101_32x8d(weights=None)
            net = FinalLayerMixupModel(net_base, criterion, num_classes, False)
            BATCH_SIZE = 12
        elif "densenet201" in basename_l:
            MODEL_NAME = "densenet201"
            if spec_type == "imagenet":
                net_base = models.densenet201(
                    weights=models.DenseNet201_Weights.IMAGENET1K_V1
                )
            else:
                net_base = models.densenet201(weights=None)
            net = FinalLayerMixupModelDenseNet(net_base, criterion, num_classes, False)
            BATCH_SIZE = 12
        elif (
            ("efficientnet-b7" in basename_l)
            or ("efficientnetb7" in basename_l)
            or ("eb7" in basename_l)
        ):
            MODEL_NAME = "efficientnet-b7"
            if _EFFNET_BACKEND == "efficientnet_pytorch":
                net_base = EfficientNet.from_name("efficientnet-b7")
            else:
                net_base = build_torchvision_efficientnet_b7(num_classes=num_classes)
            net = FinalLayerMixupModelEN(net_base, criterion, num_classes, False)
            BATCH_SIZE = 10
        else:
            print(f"{basename} is not supported; skipping.")
            continue

        print(f"{basename}: {MODEL_NAME} ({spec_type})")

        if spec_type == "ckpt":
            try:
                if MODEL_NAME == "efficientnet-b7":
                    if hasattr(net, "model"):
                        load_checkpoint_safely(
                            net.model, pretrained_model, strict=False
                        )
                    else:
                        load_checkpoint_safely(net, pretrained_model, strict=False)
                else:
                    load_checkpoint_safely(net, pretrained_model, strict=False)
            except Exception as e:
                print(
                    f"Skipping {pretrained_model} due to load error: {type(e).__name__}: {e}"
                )
                del net
                torch.cuda.empty_cache()
                continue

        for param in net.parameters():
            param.requires_grad = False

        for tid, transform_ in enumerate(transform["test"]):
            print(f"transform loop={tid}")
            dataset = {
                "test": TestDataset(df_test, transform=transform_),
            }
            dataloader = {
                "test": torch.utils.data.DataLoader(
                    dataset["test"],
                    batch_size=BATCH_SIZE,
                    shuffle=False,
                    num_workers=2,
                    pin_memory=True,
                ),
            }

            proba = predict_model(basename, net, dataloader)
            probability.append(proba)

        del net
        torch.cuda.empty_cache()

    if len(probability) == 0:
        print("No models produced predictions; writing sample_submission.csv fallback.")
        df_sub = df_sample.copy()
        df_sub.to_csv("submission.csv", index=False)
    else:
        prob_arr = np.array(probability)
        if prob_arr.ndim != 3:
            raise ValueError(
                f"Unexpected probability array shape: {prob_arr.shape}. Expected 3D (K, N, C)."
            )
        df_test["mean"] = prob_arr.mean(axis=0).argmax(axis=1)

        print(f"total time: {time.time() - start_time:.2f}[sec]")



## === cell 18
if "df_sub" in globals():
    df_test = df_sub.copy()
else:
    if len(df_test) == 2 and df_test.loc[0, "image_id"] == df_test.loc[1, "image_id"]:
        df_test = pd.read_csv(f"{BASE_DIR}/sample_submission.csv")
    else:
        df_test["label"] = df_test["mean"].astype(int)



## === cell 19
df_test.head()



## === cell 20
df_out = df_sample[["image_id"]].merge(
    df_test[["image_id", "label"]], on="image_id", how="left"
)
df_out["label"] = df_out["label"].fillna(0).astype(int)

df_out[["image_id", "label"]].to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", df_out[["image_id", "label"]].shape)
print(df_out[["image_id", "label"]].head())
