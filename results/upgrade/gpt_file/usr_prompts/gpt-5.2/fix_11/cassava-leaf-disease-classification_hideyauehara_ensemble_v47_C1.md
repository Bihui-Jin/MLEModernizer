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

0.8942278634028408

# 6. Current score

0.11584

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.11584) has done: 'I fix the missing EfficientNet dependency by switching to torchvision’s built-in EfficientNet-B7 (same architecture family) and load weights safely with `map_location` so inference runs in Kaggle without custom packages. I make the dataset path detection robust (use `/kaggle/input/cassava-leaf-disease-classification` directly when present) so `test_files` is always defined and points to the real test set. I update the Albumentations `RandomResizedCrop` calls to the v2 API to remove the runtime validation error while preserving the same augmentation intent. Finally, I make the ensemble aggregation numerically safe (handle empty model list; stack and average correctly) and always write a valid `submission.csv` with required columns.'
- What this solution (achieved 0.11584) has done: 'Your current score (0.11584) suggests the submission is effectively near-random, which is consistent with the code overwriting the classifier head (creating a fresh `fc`/`classifier` layer) and then trying to load checkpoints whose keys don’t match—so the loaded weights won’t correspond to the intended model, even if the load “succeeds” in some cases. I keep the same overall ensemble/TTA inference logic, but fix the model wrappers so they do **not** replace the classifier head; instead they use the checkpoint’s own head exactly as saved, which should immediately move accuracy strongly upward toward your ~0.894 target. I also make the forward signatures compatible with your `predict_model` call without changing inference semantics, and I ensure the submission is aligned to `sample_submission.csv` ordering (safe, minimal, and avoids accidental row-order mismatch). These are minimal changes focused purely on restoring correct checkpoint loading/inference.'
- What this solution (achieved 0.11584) has done: 'Your current score (0.11584) is consistent with the model checkpoints not actually being used correctly: the code builds torchvision backbones with their default 1000-class heads, then loads 5-class cassava checkpoints with `strict=True`, which either errors or (if the checkpoint is wrapped) silently misaligns and produces junk. I keep your exact ensemble+TTA inference flow, but make checkpoint loading robust by (1) loading the real `state_dict` even if it’s nested under keys like `state_dict`/`model`, and (2) resizing only the final classifier layer to `num_classes=5` when needed, so the rest of the checkpoint weights load and predictions become meaningful. I also ensure `df_test` is always aligned to `sample_submission.csv` order in batch mode (prevents any row-order mismatch) without changing evaluation semantics. These are minimal, directly score-relevant fixes aimed at moving accuracy up toward your ~0.894 target.'
- What this solution (achieved 0.11584) has done: 'Your current score is far below the target, so the goal is to make predictions meaningful again with minimal risk. The main issue is that you’re loading checkpoints into the *wrapper* (`FinalLayerMixupModel*`) instead of into the underlying torchvision model, so most keys won’t match and you end up with essentially random heads/features. I change loading to target `net.model.load_state_dict(...)` (and strip common prefixes like `model.`/`net.`), and I also ensure the final classifier layer is adapted based on the checkpoint’s detected class count to avoid shape-mismatch discarding. Finally, I align the output rows to `sample_submission.csv` ordering explicitly (safe, minimal) and still write `submission.csv`.'
- What this solution (achieved 0.11584) has done: 'Your score is far below the target, so the smallest likely cause is that you’re not actually loading the cassava-trained classifier heads from the checkpoints (e.g., EfficientNet-B7 head key mismatch: your code expects `classifier.1.*` but torchvision uses `classifier.1.*` only for some versions, and many Kaggle checkpoints use `classifier.weight`/`classifier.bias` or different nesting). I keep your exact ensemble + TTA loop and torchvision backbones, but make checkpoint loading “head-aware”: detect the real head keys present, adapt the model head to 5 classes, and remap common EfficientNet head key patterns to torchvision’s expected keys. I also ensure the forward call works for all wrappers by calling `net.model(inputs)` inside `predict_model` (same semantics, just avoids any wrapper signature mismatch). These minimal fixes should turn predictions from near-random into meaningful, moving accuracy strongly upward toward the target band while preserving your overall approach and submission format.'
- What this solution (achieved 0.11584) has done: 'Your current score is far below the target, so the smallest likely fix is to ensure the torchvision backbones are instantiated with the correct 5-class classifier head before loading checkpoints (otherwise the 1000-class default head makes the load fail/partial and predictions become near-random). I keep your exact ensemble + TTA inference flow, but (1) adapt the final layer to `num_classes=5` (and only deviate if a checkpoint clearly indicates a different class count), and (2) make checkpoint key remapping a bit more robust for common EfficientNet/ResNet/DenseNet head naming so `strict=True` succeeds more often. Finally, I keep your submission alignment to `sample_submission.csv` to avoid any ordering mismatch, without changing evaluation semantics.'
- What this solution (achieved 0.11584) has done: 'Your score is far below the target, so the smallest change likely to move accuracy upward is to stop applying random augmentations at test-time: your current TTA includes `RandomResizedCrop`, which makes predictions unstable and typically harms accuracy for this competition. I keep the same ensemble+TTA structure and averaging logic, but replace the random crops with deterministic `Resize+CenterCrop` variants (and keep flips), so inference matches the training distribution more closely and becomes repeatable. I also set `cudnn.benchmark=False` to avoid nondeterministic kernel selection (stability helps accuracy consistency) while keeping the same model architectures and checkpoint loading logic. Submission writing and alignment to `sample_submission.csv` remain unchanged.'
- What this solution (achieved 0.11584) has done: 'Your current score is far below the target, which strongly suggests the checkpoints are not being applied correctly at inference time. I make two minimal, score-relevant fixes: (1) ensure we always instantiate each torchvision backbone with a 5-class head (or checkpoint-inferred head) *before* loading weights, and (2) broaden the checkpoint key-remapping to cover common EfficientNet-B7 head variants (`classifier.0.*`, `classifier.*`, `_fc.*`) so `strict=True` succeeds more often and the learned head is actually used. These changes keep your ensemble+TTA inference core logic identical, but should move predictions from near-random toward meaningful accuracy. The submission writing and alignment to `sample_submission.csv` remain unchanged.'
- What this solution (achieved 0.11584) has done: 'Your score gap to the target is large (0.11584 → ~0.894), so the smallest meaningful improvement is to ensure the ensemble actually uses the cassava-trained classifier heads from the checkpoints. I keep your exact ensemble + TTA structure, but fix model instantiation so each backbone is created with the correct `num_classes` **before** loading weights (torchvision supports this directly for these models), which prevents silent head mismatches and near-random predictions. I also make checkpoint loading slightly more robust by handling common “EMA” checkpoints and avoiding `strict=False` unless absolutely necessary, because partial loads can destroy accuracy. Finally, I keep your submission alignment to `sample_submission.csv` and still write `submission.csv`.'
- What this solution (achieved 0.11584) has done: 'The current score (0.11584) is far below the target, and the biggest likely cause is that your EfficientNet-B7 checkpoints aren’t being applied correctly because torchvision’s `efficientnet_b7` head is `classifier.1.*` while many cassava checkpoints store the head as `classifier.0.*` (or other variants), so `strict=False` ends up leaving the head random and predictions near-random. I make a minimal, head-aware remapping that detects whether the model’s classifier last layer is index 0 or 1 and maps checkpoint head keys accordingly (without changing your model choices, ensemble logic, or TTA). I also ensure we always build the backbone with the checkpoint-inferred number of classes *before* loading weights (so `strict=True` succeeds whenever possible), again keeping the same inference semantics. These changes are directly aimed at making the loaded checkpoints actually drive predictions, moving accuracy sharply upward toward the target band while still writing a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import numpy as np
import glob



## === cell 1
pretrained_models = (
    glob.glob(f"../input/densenet201-04-2019data/*.pth")
    + glob.glob(f"../input/resnet152-04-2019data/*.pth")
    + glob.glob(f"../input/eb7slseed70/efficientnet-b7sl_SEED70.best/*.pth")
)

print(f"{len(pretrained_models)} models found.")
if len(pretrained_models) > 0:
    print("\n".join(np.sort(pretrained_models)))



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
import sys
import time

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
from torchvision.models import efficientnet_b7



## === cell 4
SIZE = 512  # image size
num_classes = 5



## === cell 5
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用デバイス: {device}")



## === cell 6
default_base = Path("/kaggle/input/cassava-leaf-disease-classification")
if not default_base.exists():
    default_base = Path(
        "/kaggle/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification"
    )

if default_base.exists():
    BASE_DIR = str(default_base)
    run_type = os.getenv("KAGGLE_KERNEL_RUN_TYPE", "Batch")
    if run_type == "Interactive":
        print("Test run in Kaggle environment (interactive subset).")
        TEST_PATH = f"{BASE_DIR}/train_images"
        test_files = sorted(os.listdir(TEST_PATH))[:32]
    else:
        print("In Kaggle environment (batch/full test).")
        TEST_PATH = f"{BASE_DIR}/test_images"
        test_files = sorted(os.listdir(TEST_PATH))
else:
    print("In the local environment.")
    BASE_DIR = "data"
    run_type = os.getenv("KAGGLE_KERNEL_RUN_TYPE", "Batch")
    if Path(BASE_DIR, "test_images").exists():
        TEST_PATH = f"{BASE_DIR}/test_images"
    else:
        TEST_PATH = f"{BASE_DIR}/train_images"
    test_files = sorted(os.listdir(TEST_PATH))[:32]

print(f"BASE_DIR={BASE_DIR}")
print(f"TEST_PATH={TEST_PATH}")
print(f"Number of test images: {len(test_files)}")



## === cell 7
sample_sub_path = Path(BASE_DIR) / "sample_submission.csv"
if sample_sub_path.exists():
    df_test = pd.read_csv(sample_sub_path)
    if run_type == "Interactive":
        df_test = pd.DataFrame(test_files, columns=["image_id"])
        df_test["label"] = 1
else:
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
                A.Resize(height=SIZE, width=SIZE, p=1.0),
                A.CenterCrop(height=SIZE, width=SIZE, p=1.0),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                A.HorizontalFlip(p=1.0),
                A.Resize(height=SIZE, width=SIZE, p=1.0),
                A.CenterCrop(height=SIZE, width=SIZE, p=1.0),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                A.VerticalFlip(p=1.0),
                A.Resize(height=SIZE, width=SIZE, p=1.0),
                A.CenterCrop(height=SIZE, width=SIZE, p=1.0),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                A.HorizontalFlip(p=1.0),
                A.VerticalFlip(p=1.0),
                A.Resize(height=SIZE, width=SIZE, p=1.0),
                A.CenterCrop(height=SIZE, width=SIZE, p=1.0),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                A.Resize(height=SIZE + 32, width=SIZE + 32, p=1.0),
                A.CenterCrop(height=SIZE, width=SIZE, p=1.0),
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

        NOTE: Do NOT replace the classifier head here.
        """
        super(FinalLayerMixupModel, self).__init__()
        self.model = model
        self.criterion = criterion
        self.alpha = alpha

    def forward(self, inputs, labels=None, phase="test"):
        if phase == "val":
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            return outputs, loss

        if phase == "test":
            outputs = self.model(inputs)
            return outputs

        alpha = self.alpha
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        index = torch.randperm(len(labels), device=inputs.device)
        mixed_x = lam * inputs + (1 - lam) * inputs[index]
        outputs = self.model(mixed_x)

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
        NOTE: Do NOT replace classifier head.
        """
        super(FinalLayerMixupModelDenseNet, self).__init__()
        self.model = model
        self.criterion = criterion
        self.alpha = alpha

    def forward(self, inputs, labels=None, phase="test"):
        if phase == "val":
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            return outputs, loss

        if phase == "test":
            outputs = self.model(inputs)
            return outputs

        alpha = self.alpha
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        index = torch.randperm(len(labels), device=inputs.device)
        mixed_x = lam * inputs + (1 - lam) * inputs[index]
        outputs = self.model(mixed_x)

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
        """
        NOTE: Do NOT rewrite EfficientNet classifier head here.
        """
        super(FinalLayerMixupModelEN, self).__init__()
        self.model = model
        self.criterion = criterion
        self.alpha = alpha

    def forward(self, inputs, labels=None, phase="test"):
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
            raise FileNotFoundError(f"Image not found/readable: {TEST_PATH}/{image_id}")
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
    Minimal: call the underlying backbone directly (net.model) for inference.
    """
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
            outputs = net.model(inputs)
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
            "ema",
            "model_ema",
        ]:
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
    return ckpt


def _strip_any_prefixes(state_dict):
    """
    Minimal: strip common training prefixes if they dominate the keyspace.
    """
    if not isinstance(state_dict, dict):
        return state_dict

    prefixes = ["module.", "model.", "net."]
    keys = list(state_dict.keys())
    for p in prefixes:
        n = sum(k.startswith(p) for k in keys)
        if n >= max(1, int(0.8 * len(keys))):
            return {k[len(p) :]: v for k, v in state_dict.items()}
    return state_dict


def _remap_common_head_keys(
    state_dict: dict, model_name: str, backbone: nn.Module = None
) -> dict:
    """
    Score-critical minimal fix:
    - EfficientNet checkpoints often store head as classifier.weight/bias or classifier.0.* (timm-style),
      while torchvision EfficientNet uses classifier.1.* (Dropout is classifier[0], Linear is classifier[1]).
    - If we don't remap to the *actual* Linear index used by the instantiated backbone, the head stays random,
      leading to near-random predictions (very low accuracy).
    """
    if not isinstance(state_dict, dict):
        return state_dict
    sd = dict(state_dict)

    if model_name.startswith("resnet") or model_name.startswith("resnext"):
        if (
            ("classifier.weight" in sd)
            and ("classifier.bias" in sd)
            and ("fc.weight" not in sd)
        ):
            sd["fc.weight"] = sd.pop("classifier.weight")
            sd["fc.bias"] = sd.pop("classifier.bias")

    if model_name.startswith("densenet"):
        if (
            ("fc.weight" in sd)
            and ("fc.bias" in sd)
            and ("classifier.weight" not in sd)
        ):
            sd["classifier.weight"] = sd.pop("fc.weight")
            sd["classifier.bias"] = sd.pop("fc.bias")

    if model_name.startswith("efficientnet"):
        linear_idx = 1
        try:
            if backbone is not None and hasattr(backbone, "classifier"):
                for i in range(len(backbone.classifier)):
                    if isinstance(backbone.classifier[i], nn.Linear):
                        linear_idx = i
                        break
        except Exception:
            linear_idx = 1

        target_w = f"classifier.{linear_idx}.weight"
        target_b = f"classifier.{linear_idx}.bias"

        if (target_w not in sd) and (target_b not in sd):
            if ("classifier.weight" in sd) and ("classifier.bias" in sd):
                sd[target_w] = sd.pop("classifier.weight")
                sd[target_b] = sd.pop("classifier.bias")
            elif ("classifier.0.weight" in sd) and ("classifier.0.bias" in sd):
                sd[target_w] = sd.pop("classifier.0.weight")
                sd[target_b] = sd.pop("classifier.0.bias")
            elif ("classifier.1.weight" in sd) and ("classifier.1.bias" in sd):
                sd[target_w] = sd.pop("classifier.1.weight")
                sd[target_b] = sd.pop("classifier.1.bias")
            elif ("head.weight" in sd) and ("head.bias" in sd):
                sd[target_w] = sd.pop("head.weight")
                sd[target_b] = sd.pop("head.bias")
            elif ("_fc.weight" in sd) and ("_fc.bias" in sd):
                sd[target_w] = sd.pop("_fc.weight")
                sd[target_b] = sd.pop("_fc.bias")

    return sd


def infer_num_classes_from_state_dict(
    state_dict, model_name: str, fallback: int = 5
) -> int:
    if not isinstance(state_dict, dict):
        return fallback
    try:
        if model_name.startswith("resnet") or model_name.startswith("resnext"):
            w = state_dict.get("fc.weight", None)
            if w is not None and hasattr(w, "shape"):
                return int(w.shape[0])
        elif model_name.startswith("densenet"):
            w = state_dict.get("classifier.weight", None)
            if w is not None and hasattr(w, "shape"):
                return int(w.shape[0])
        elif model_name.startswith("efficientnet"):
            for key in [
                "classifier.1.weight",
                "classifier.weight",
                "classifier.0.weight",
                "head.weight",
                "_fc.weight",
            ]:
                w = state_dict.get(key, None)
                if w is not None and hasattr(w, "shape"):
                    return int(w.shape[0])
    except Exception:
        pass
    return fallback


def _set_backbone_num_classes(net: nn.Module, model_name: str, ncls: int):
    """
    Minimal: ensure the final layer matches checkpoint classes before loading,
    so strict=True can succeed and we don't end up with a random head.
    """
    if model_name.startswith("resnet") or model_name.startswith("resnext"):
        if hasattr(net.model, "fc") and isinstance(net.model.fc, nn.Linear):
            net.model.fc = nn.Linear(net.model.fc.in_features, ncls)
    elif model_name.startswith("densenet"):
        if hasattr(net.model, "classifier") and isinstance(
            net.model.classifier, nn.Linear
        ):
            net.model.classifier = nn.Linear(net.model.classifier.in_features, ncls)
    elif model_name.startswith("efficientnet"):
        if hasattr(net.model, "classifier") and isinstance(
            net.model.classifier, nn.Sequential
        ):
            for i in range(len(net.model.classifier)):
                if isinstance(net.model.classifier[i], nn.Linear):
                    net.model.classifier[i] = nn.Linear(
                        net.model.classifier[i].in_features, ncls
                    )
                    break




## === cell 18
probability = []

start_time = time.time()

if sample_sub_path.exists():
    df_test = pd.read_csv(sample_sub_path)
    if run_type == "Interactive":
        df_test = pd.DataFrame(test_files, columns=["image_id"])
        df_test["label"] = 1

if len(pretrained_models) == 0:
    print(
        "No pretrained .pth models found in the specified input folders. Writing a valid baseline submission."
    )
    if sample_sub_path.exists():
        df_test = pd.read_csv(sample_sub_path)
    df_test["label"] = df_test["label"].astype(int)
else:
    for pretrained_model in pretrained_models:
        basename = os.path.splitext(os.path.basename(pretrained_model))[0]
        criterion = nn.CrossEntropyLoss()

        ckpt = torch.load(pretrained_model, map_location="cpu")
        state_raw = _strip_any_prefixes(_extract_state_dict(ckpt))

        inferred_classes = None
        if "resnet18" in basename:
            MODEL_NAME = "resnet18"
            BATCH_SIZE = 64
        elif "resnet50" in basename:
            MODEL_NAME = "resnet50"
            BATCH_SIZE = 32
        elif "resnet152" in basename:
            MODEL_NAME = "resnet152"
            BATCH_SIZE = 16
        elif "resnext101" in basename:
            MODEL_NAME = "resnext101"
            BATCH_SIZE = 12
        elif "densenet201" in basename:
            MODEL_NAME = "densenet201"
            BATCH_SIZE = 12
        elif "efficientnet-b7" in basename:
            MODEL_NAME = "efficientnet-b7"
            BATCH_SIZE = 10
        else:
            print(f"{basename} is not supported.")
            sys.exit()

        inferred_classes = infer_num_classes_from_state_dict(
            state_raw, MODEL_NAME, fallback=num_classes
        )
        if inferred_classes <= 0 or inferred_classes > 1000:
            inferred_classes = num_classes

        if MODEL_NAME == "resnet18":
            backbone = models.resnet18(weights=None, num_classes=inferred_classes)
            net = FinalLayerMixupModel(backbone, criterion, inferred_classes, False)
        elif MODEL_NAME == "resnet50":
            backbone = models.resnet50(weights=None, num_classes=inferred_classes)
            net = FinalLayerMixupModel(backbone, criterion, inferred_classes, False)
        elif MODEL_NAME == "resnet152":
            backbone = models.resnet152(weights=None, num_classes=inferred_classes)
            net = FinalLayerMixupModel(backbone, criterion, inferred_classes, False)
        elif MODEL_NAME == "resnext101":
            backbone = models.resnext101_32x8d(
                weights=None, num_classes=inferred_classes
            )
            net = FinalLayerMixupModel(backbone, criterion, inferred_classes, False)
        elif MODEL_NAME == "densenet201":
            backbone = models.densenet201(weights=None, num_classes=inferred_classes)
            net = FinalLayerMixupModelDenseNet(
                backbone, criterion, inferred_classes, False
            )
        elif MODEL_NAME == "efficientnet-b7":
            backbone = efficientnet_b7(weights=None, num_classes=inferred_classes)
            net = FinalLayerMixupModelEN(backbone, criterion, inferred_classes, False)

        print(f"{basename}: {MODEL_NAME}")

        state = _remap_common_head_keys(state_raw, MODEL_NAME, backbone=net.model)

        _set_backbone_num_classes(net, MODEL_NAME, inferred_classes)

        try:
            net.model.load_state_dict(state, strict=True)
            print(
                f"[INFO] Loaded checkpoint into backbone with strict=True ({inferred_classes} classes)."
            )
        except RuntimeError as e:
            print(f"[WARN] strict=True failed for {basename}: {e}")
            missing, unexpected = net.model.load_state_dict(state, strict=False)
            print(
                f"[INFO] Loaded into backbone with strict=False. missing={len(missing)} unexpected={len(unexpected)}"
            )

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
                ),
            }

            proba = predict_model(basename, net, dataloader)
            probability.append(proba)

        del net
        torch.cuda.empty_cache()

    prob_arr = np.stack(probability, axis=0)  # [n_preds, n_images, n_classes]
    df_test["mean"] = prob_arr.mean(axis=0).argmax(axis=1)

    print(f"total time: {time.time() - start_time:.2f}[sec]")



## === cell 19
if "mean" in df_test.columns:
    df_test["label"] = df_test["mean"].astype(int)
else:
    df_test["label"] = df_test["label"].astype(int)



## === cell 20
df_test.head()



## === cell 21
if sample_sub_path.exists() and run_type != "Interactive":
    sub_template = pd.read_csv(sample_sub_path)
    df_test = sub_template[["image_id"]].merge(
        df_test[["image_id", "label"]], on="image_id", how="left"
    )
    df_test["label"] = df_test["label"].fillna(0).astype(int)

df_test[["image_id", "label"]].to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", df_test[["image_id", "label"]].shape)
print(df_test[["image_id", "label"]].head())
