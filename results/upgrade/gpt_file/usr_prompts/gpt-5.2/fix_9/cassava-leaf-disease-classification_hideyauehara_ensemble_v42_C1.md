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

0.8919613176186159

# 6. Current score

0.11584

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.11584) has done: 'I first fix the reason you currently don’t get a Kaggle score: your inference is accidentally running on `train_images` in the non-`/kaggle/input` branch, so it won’t match `sample_submission.csv` and can also fail submission expectations. I make the path resolution robust by always preferring the official `sample_submission.csv` (to define exact test `image_id` order/rows) and auto-detecting `test_images` from the provided filesystem (`/kaggle/data/...` in your environment). Then, while preserving your exact model logic, I make the TTA deterministic (use fixed-center crops instead of random crops/rotations at test time) so predictions are stable and typically improve accuracy versus noisy random test augmentation. Finally, I ensure the submission rows align exactly to the sample submission and that labels are valid integers 0–4.'
- What this solution (achieved 0.11584) has done: 'Your current score (0.11584) is far below the target (0.89196), and the biggest likely cause is that the loaded checkpoints’ parameter keys don’t match the wrapper modules (e.g., your wrappers add `convlayer`/`fc` names), so `load_state_dict(..., strict=False)` silently leaves most weights uninitialized, yielding near-random predictions. I keep your exact model selection and inference/TTA logic, but change checkpoint loading to robustly map common checkpoint formats (raw torchvision, your wrapper formats, and `state_dict` with `module.` prefixes) into the current wrapper module so that the intended pretrained weights actually get loaded. I also add a quick sanity print of how many parameters were matched to catch “loaded nothing” situations without changing evaluation semantics. These are minimal changes directly aimed at moving accuracy up toward your target.'
- What this solution (achieved 0.11584) has done: 'Your score is far below the target, so the most likely issue is still “wrong weights loaded” (near-random predictions). I keep your exact model wrappers/inference/TTA, but make checkpoint loading more robust for common Kaggle/Lightning formats by (1) stripping more prefixes, (2) handling `ema_state_dict`, and (3) systematically trying key-remaps that match your wrappers (`convlayer.*`, `fc.*`, `model.*`) and choosing the remap with the highest shape-matched key count. I also add a hard sanity check that refuses to run inference if a checkpoint matches too few parameters (this prevents silently producing random submissions and should move accuracy sharply upward toward your target). Finally, I keep submission row order strictly aligned to `sample_submission.csv` as you already do.'
- What this solution (achieved 0.11584) has done: 'Your current score (0.11584) is so far below the target (0.89196) that the most likely remaining issue is still “wrong model head is being used at inference”, even if weights load: your wrappers replace the classifier/FC with a *new random* layer (`self.fc` / `model.classifier`), so unless the checkpoint was trained with this exact wrapper, the checkpoint’s final-layer weights won’t be used correctly. I keep the same architectures, dataset, transforms, and TTA averaging, but adjust checkpoint loading to (1) detect and load classifier/FC weights into the correct place for each wrapper type, and (2) for `FinalLayerMixupModelEN`, avoid reinitializing the classifier when we load it from the checkpoint. This is a minimal, semantics-preserving correctness fix that should move accuracy sharply upward toward your target. I also keep strict submission alignment to `sample_submission.csv` unchanged.'
- What this solution (achieved 0.11584) has done: 'Your current score (0.11584) is far below the target (0.89196), so we should make the smallest correctness changes that plausibly recover the intended trained weights at inference time. The main issue is that your ResNet/DenseNet wrappers create a new random `fc` layer, but many common checkpoints store the trained head under `model.fc`/`fc`/`classifier`, so your loader often fails to map those weights into `self.fc`, leaving the classifier effectively random and collapsing accuracy. I keep your exact model choices, TTA, transforms, and inference loop, but make `load_checkpoint_robust` explicitly remap head keys into the wrapper `fc.*` (and keep a best-match selection), plus a small `squeeze` fix to avoid batch-size-1 shape bugs. These are minimal, directly score-relevant fixes and keep the rest of your pipeline unchanged while making the loaded models actually usable.'
- What this solution (achieved 0.11584) has done: 'Your score is far below the target, so the smallest high-impact fix is to ensure the test-time preprocessing matches what these checkpoints were almost certainly trained with: resize to a slightly larger size and then center-crop, instead of center-cropping directly (which can fail or heavily distort if the input image is smaller than 512 in either dimension). This keeps your model architectures, TTA structure (3 flips), inference loop, and loss untouched, but makes the input distribution consistent and should materially improve accuracy. I also add a defensive fallback to `cv2.resize` if any image still comes in smaller than the crop size, preventing silent failures or unintended behavior. Submission row order and format remain strictly aligned to `sample_submission.csv`.'

# 9. Code solution

## === cell 0
import numpy as np
import glob



## === cell 1
import os


def find_pretrained_models():
    candidates = []
    patterns = [
        "../input/eb7mseed71/*.pth",
        "/kaggle/input/eb7mseed71/*.pth",
        "/kaggle/input/*/*.pth",
        "/kaggle/input/*/*/*.pth",
        "/kaggle/data/*/*.pth",
        "/kaggle/data/*/*/*.pth",
    ]
    for pat in patterns:
        candidates.extend(glob.glob(pat))
    seen = set()
    uniq = []
    for p in candidates:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    for base in ["/kaggle/input", "/kaggle/data"]:
        if len(uniq) == 0 and os.path.isdir(base):
            for root, _, files in os.walk(base):
                for fn in files:
                    if fn.endswith(".pth"):
                        uniq.append(os.path.join(root, fn))
    return np.array(sorted(set(uniq)))


pretrained_models = find_pretrained_models()

print(f"{len(pretrained_models)} models found.")
if len(pretrained_models) > 0:
    print("\n".join(pretrained_models))
else:
    print(
        "No pretrained models found in known /kaggle/input or /kaggle/data locations."
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
    torch.backends.cudnn.benchmark = False


SEED = 42
seed_everything(seed=SEED)



## === cell 3
EfficientNet = None



## === cell 4
SIZE = 512  # image size
num_classes = 5



## === cell 5
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用デバイス: {device}")




## === cell 6
def resolve_sample_submission_path():
    candidates = [
        "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv",
        "/kaggle/data/cassava-leaf-disease-classification/sample_submission.csv",
        "/kaggle/input/sample_submission.csv",
        "/kaggle/data/sample_submission.csv",
        "sample_submission.csv",
        "data/sample_submission.csv",
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    for base in ["/kaggle/input", "/kaggle/data"]:
        if os.path.isdir(base):
            for root, _, files in os.walk(base):
                if "sample_submission.csv" in files:
                    return os.path.join(root, "sample_submission.csv")
    return None


def resolve_test_images_dir():
    candidates = [
        "/kaggle/input/cassava-leaf-disease-classification/test_images",
        "/kaggle/data/cassava-leaf-disease-classification/test_images",
        "/kaggle/input/test_images",
        "/kaggle/data/test_images",
        "data/test_images",
    ]
    for p in candidates:
        if os.path.isdir(p):
            return p
    for base in ["/kaggle/input", "/kaggle/data"]:
        if os.path.isdir(base):
            for root, dirs, _ in os.walk(base):
                if root.endswith("test_images") and os.path.isdir(root):
                    return root
    return None


sample_path = resolve_sample_submission_path()
if sample_path is None:
    raise FileNotFoundError(
        "Could not locate sample_submission.csv in /kaggle/input or /kaggle/data."
    )

TEST_PATH = resolve_test_images_dir()
if TEST_PATH is None:
    raise FileNotFoundError(
        "Could not locate test_images directory in /kaggle/input or /kaggle/data."
    )

print(f"sample_path: {sample_path}")
print(f"TEST_PATH: {TEST_PATH}")



## === cell 7
df_test = pd.read_csv(sample_path)
if "image_id" not in df_test.columns:
    raise ValueError("sample_submission.csv missing required column 'image_id'")
df_test["label"] = df_test.get("label", 0)

missing = []
for fn in df_test["image_id"].head(20).tolist():
    if not os.path.isfile(os.path.join(TEST_PATH, fn)):
        missing.append(fn)
if len(missing) > 0:
    print(
        "Warning: some test images not found under TEST_PATH (showing up to 5):",
        missing[:5],
    )



## === cell 8
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

RESIZE_TO = 560  # common "resize then crop" size for 512-crop models

transform = {
    "test": [
        Compose(
            [
                A.Resize(RESIZE_TO, RESIZE_TO, interpolation=cv2.INTER_LINEAR),
                A.CenterCrop(SIZE, SIZE),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                A.HorizontalFlip(p=1),
                A.Resize(RESIZE_TO, RESIZE_TO, interpolation=cv2.INTER_LINEAR),
                A.CenterCrop(SIZE, SIZE),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                A.VerticalFlip(p=1),
                A.Resize(RESIZE_TO, RESIZE_TO, interpolation=cv2.INTER_LINEAR),
                A.CenterCrop(SIZE, SIZE),
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
            x = x.squeeze(-1).squeeze(-1)
            outputs = self.fc(x)
            loss = self.criterion(outputs, labels)

            return outputs, loss

        if phase == "test":
            x = self.convlayer(inputs)
            x = x.squeeze(-1).squeeze(-1)
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
        mixed_x = mixed_x.squeeze(-1).squeeze(-1)
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
            x = x.squeeze(-1).squeeze(-1)
            outputs = self.fc(x)
            loss = self.criterion(outputs, labels)

            return outputs, loss

        if phase == "test":
            x = self.convlayer(inputs)
            x = self.AdaptiveAvgPool2d(x)
            x = x.squeeze(-1).squeeze(-1)
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
        mixed_x = mixed_x.squeeze(-1).squeeze(-1)
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
        import sys

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
            raise FileNotFoundError(f"Failed to read image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h, w = img.shape[:2]
        if h < SIZE or w < SIZE:
            new_h = max(h, SIZE)
            new_w = max(w, SIZE)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        return img

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        img = self.load_image(image_id)
        if self.transform:
            img = self.transform(image=img)["image"]
        return img, image_id




## === cell 13
def predict_model(basename, net, dataloader):
    """
    basename: 学習済みモデル名
    net     : 学習済みモデル
    """
    model_start_time = time.time()

    net.to(device)
    net.eval()
    torch.set_grad_enabled(False)

    probability = []

    progress = tqdm(dataloader["test"], desc=f"{basename}: ")
    for inputs, image_ids in progress:
        inputs = inputs.to(device)
        outputs = net(inputs, False, "test")
        probability.append(torch.softmax(outputs, dim=1).cpu().numpy())

    print(f"{basename} time: {time.time() - model_start_time:.2f}[sec]")
    return np.concatenate(probability, axis=0)




## === cell 14
from collections import OrderedDict


def _strip_prefix(state_dict, prefix):
    if not any(k.startswith(prefix) for k in state_dict.keys()):
        return state_dict
    return OrderedDict(
        (k[len(prefix) :], v) if k.startswith(prefix) else (k, v)
        for k, v in state_dict.items()
    )


def _select_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for key in [
            "ema_state_dict",
            "state_dict",
            "model_state_dict",
            "model",
            "net",
            "weights",
        ]:
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
    return ckpt


def _count_shape_matches(candidate_sd, net_sd):
    m = 0
    for k, v in candidate_sd.items():
        if k in net_sd and tuple(net_sd[k].shape) == tuple(v.shape):
            m += 1
    return m


def _fix_efficientnet_classifier_to_num_classes(net, num_classes):
    if isinstance(net, FinalLayerMixupModelEN):
        m = net.model
    else:
        m = net
    if hasattr(m, "classifier"):
        if isinstance(m.classifier, nn.Sequential):
            last = m.classifier[-1]
            if isinstance(last, nn.Linear) and last.out_features != num_classes:
                m.classifier[-1] = nn.Linear(last.in_features, num_classes)
        elif (
            isinstance(m.classifier, nn.Linear)
            and m.classifier.out_features != num_classes
        ):
            m.classifier = nn.Linear(m.classifier.in_features, num_classes)


def _add_head_key_remaps(state):
    """
    Change (score-relevant): explicitly map common trained-head keys into this solution's
    wrapper heads (FinalLayerMixupModel/FinalLayerMixupModelDenseNet use `fc.*`).
    Without this, the wrapper `fc` often stays random even when the backbone loads.
    """
    remaps = []

    remaps.append(("as_is", state))

    to_wrapper_fc = OrderedDict()
    for k, v in state.items():
        nk = k
        if nk.startswith("model.fc."):
            nk = "fc." + nk[len("model.fc.") :]
        elif nk.startswith("fc."):
            nk = "fc." + nk[len("fc.") :]
        elif nk.startswith("model.classifier."):
            nk = "fc." + nk[len("model.classifier.") :]
        elif nk.startswith("classifier."):
            nk = "fc." + nk[len("classifier.") :]
        to_wrapper_fc[nk] = v
    remaps.append(("head_to_wrapper_fc", to_wrapper_fc))

    to_model_prefix = OrderedDict()
    for k, v in state.items():
        to_model_prefix[k if k.startswith("model.") else ("model." + k)] = v
    remaps.append(("to_model_prefix", to_model_prefix))

    return remaps


def load_checkpoint_robust(net, ckpt_path, min_match_ratio=0.60):
    """
    Minimal but crucial fix for score: ensure checkpoint weights actually map into the wrapper module.
    Keeps the same architecture and inference; only improves correctness of weight loading.

    Additional change: enforce a minimum match ratio so we don't silently run random-weight inference.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = _select_state_dict(ckpt)
    if not isinstance(state, dict):
        raise ValueError(f"Unsupported checkpoint format at {ckpt_path}")

    for pref in ["module.", "model.", "net.", "network.", "backbone.", "encoder."]:
        state = _strip_prefix(state, pref)

    _fix_efficientnet_classifier_to_num_classes(net, num_classes)

    net_sd = net.state_dict()

    candidates = _add_head_key_remaps(state)

    remap_to_wrapper = OrderedDict()
    for k, v in state.items():
        nk = k

        if nk.startswith(("convlayer.", "fc.", "model.")):
            remap_to_wrapper[nk] = v
            continue

        if nk.startswith(
            (
                "conv1.",
                "bn1.",
                "layer1.",
                "layer2.",
                "layer3.",
                "layer4.",
            )
        ):
            remap_to_wrapper["convlayer." + nk] = v
            continue

        if nk.startswith("features."):
            remap_to_wrapper["convlayer." + nk[len("features.") :]] = v
            continue

        if nk.startswith("fc."):
            remap_to_wrapper["fc." + nk[len("fc.") :]] = v
            continue
        if nk.startswith("classifier."):
            remap_to_wrapper["fc." + nk[len("classifier.") :]] = v
            continue

        remap_to_wrapper["convlayer." + nk] = v

    candidates.append(("to_wrapper_convlayer_fc", remap_to_wrapper))

    best_name, best_sd, best_match = None, None, -1
    for name, cand in candidates:
        match = _count_shape_matches(cand, net_sd)
        if match > best_match:
            best_match = match
            best_name, best_sd = name, cand

    filtered = OrderedDict()
    for k, v in best_sd.items():
        if k in net_sd and tuple(net_sd[k].shape) == tuple(v.shape):
            filtered[k] = v

    missing_keys, unexpected_keys = net.load_state_dict(filtered, strict=False)

    match_ratio = len(filtered) / max(1, len(net_sd))
    if match_ratio < min_match_ratio:
        raise RuntimeError(
            f"Checkpoint '{ckpt_path}' matched too few parameters for this model wrapper "
            f"({len(filtered)}/{len(net_sd)} = {match_ratio:.2%}). "
            "This usually means wrong architecture/wrapper for the checkpoint, which would yield near-random accuracy."
        )

    return {
        "mode": f"filtered_nonstrict_bestmatch[{best_name}]",
        "matched": len(filtered),
        "total": len(net_sd),
        "match_ratio": match_ratio,
        "missing": len(missing_keys),
        "unexpected": len(unexpected_keys),
    }




## === cell 15
probability = []
start_time = time.time()

if len(pretrained_models) == 0:
    print(
        "No pretrained models available; writing sample_submission.csv as submission.csv to yield a valid file."
    )
    df_sub = pd.read_csv(sample_path)
    df_sub.to_csv("submission.csv", index=False)
    print(f"Wrote submission.csv from {sample_path}.")
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
        elif "efficientnet-b7" in basename:
            MODEL_NAME = "efficientnet-b7"
            net = models.efficientnet_b7(weights=None)
            net = FinalLayerMixupModelEN(net, criterion, num_classes, False)
            BATCH_SIZE = 10
        else:
            print(f"{basename} is not supported.")
            raise ValueError(f"Unsupported model in filename: {basename}")

        print(f"{basename}: {MODEL_NAME}")

        info = load_checkpoint_robust(net, pretrained_model)
        print(f"Checkpoint load info for {basename}: {info}")

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

            proba = predict_model(basename, net, dataloader)  # (N, 5)
            probability.append(proba)

        del net
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    proba_all = np.stack(probability, axis=0)  # (T, N, 5)
    df_test["mean"] = proba_all.mean(axis=0).argmax(axis=1).astype(np.int64)

    print(f"total time: {time.time() - start_time:.2f}[sec]")



## === cell 16
if "mean" in df_test.columns:
    df_test["label"] = df_test["mean"].astype(int)

df_test["label"] = df_test["label"].astype(int).clip(0, num_classes - 1)



## === cell 17
df_test.head()



## === cell 18
df_sub = pd.read_csv(sample_path)[["image_id"]]
df_sub = df_sub.merge(df_test[["image_id", "label"]], on="image_id", how="left")

if df_sub["label"].isna().any():
    n_miss = int(df_sub["label"].isna().sum())
    print(
        f"Warning: {n_miss} image_ids had no prediction; filling with 0 to keep submission valid."
    )
    df_sub["label"] = df_sub["label"].fillna(0)

df_sub["label"] = df_sub["label"].astype(int)

df_sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv:", os.path.abspath("submission.csv"), "rows=", len(df_sub))
print(df_sub.head())
