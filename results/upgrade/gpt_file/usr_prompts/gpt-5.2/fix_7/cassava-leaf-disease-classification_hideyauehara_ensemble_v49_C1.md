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

0.8943789664551224

# 6. Current score

0.10164

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.10052) has done: 'I fix the dataset path resolution so `TEST_PATH` always points to the actual folder that contains `.jpg` files (your current code sometimes builds `/.../test_images/test_images`, causing missing-file errors). I also make `df_test` come directly from `sample_submission.csv` (correct length/order) and only fill its `label` from predictions, which fixes the “length must match answers” submission error and avoids the `KeyError: 'mean'`. Finally, I make the image list filtering robust (only `*.jpg`) and keep the rest of your inference/ensemble logic unchanged so it runs end-to-end and writes a valid `submission.csv`.'
- What this solution (achieved 0.10389) has done: 'Your current 0.10052 score strongly suggests the model heads are effectively random (most likely because the loaded `.pth` weights don’t actually populate the custom `fc` layers you created, due to key mismatches like `fc.*` vs `model.fc.*` / `classifier.*`). To move the accuracy up toward the 0.894 target with minimal logic change, I (1) load checkpoints more correctly by remapping common head key patterns into your wrapper’s `fc.*`, and (2) ensure the backbone weights still load into `convlayer.*` properly (and keep `strict=False` as before for safety). I not change your architecture, transforms, ensembling, or inference loops—only the state_dict handling so the intended trained weights are actually used. This is the smallest change that plausibly closes most of the gap without altering evaluation semantics.'
- What this solution (achieved 0.10314) has done: 'Your score is far below the target, so the most likely issue is still that the intended trained heads/backbones are not being restored correctly from the `.pth` files, leaving inference close to random. I keep your model wrappers, transforms, ensembling, and inference loop intact, and only make the weight-loading more robust by (1) auto-detecting which part of the checkpoint corresponds to the wrapper (`convlayer.*`/`fc.*`) vs raw torchvision (`features.*`/`classifier.*`/`fc.*`), (2) mapping ResNet-style backbones into `convlayer.*` including common prefixes like `model.`/`backbone.`/`encoder.`, and (3) explicitly verifying that the classifier head weights actually loaded; if not, we fail fast instead of silently producing a near-random submission. This is a minimal change focused on correctness of restored weights, which is the smallest plausible step to move accuracy toward ~0.89.'
- What this solution (achieved 0.10314) has done: 'Your 0.10314 accuracy is far below the 0.894 target, so the smallest plausible fix is to ensure the trained weights are actually being used at inference time rather than silently falling back to mostly-random heads. I keep your model wrappers, transforms/TTA, ensembling, and inference loop unchanged, and only (1) expand checkpoint parsing to correctly extract state_dicts stored under common keys, (2) improve key remapping so that backbone weights land in `convlayer.*` and classifier weights land in `fc.*` (and for EfficientNet into `model.*`), and (3) make the head-load validation wrapper-aware (so DenseNet/EfficientNet don’t pass/fail incorrectly). These changes directly target the likely root cause of near-random performance while preserving evaluation semantics and producing the same `submission.csv` format/path.'
- What this solution (achieved 0.10164) has done: 'Your score (~0.10) is far below the target (~0.894), which is most consistent with your inference using essentially untrained/random classifier heads because the checkpoint keys still aren’t being mapped into the wrapper modules correctly. I keep your exact model wrappers, transforms/TTA, ensembling, and inference loop, and only strengthen checkpoint loading to (1) handle “wrapped” checkpoints more reliably, (2) correctly map torchvision-style ResNet/DenseNet/EfficientNet keys into your wrapper’s `convlayer.*` and `fc.*` (or `model.*`), and (3) validate that not just the head but also key backbone parts are loaded (fail-fast instead of silently producing random predictions). This is the smallest change that should move accuracy substantially toward the target without changing evaluation semantics. The script still write a valid `submission.csv` with the required columns.'

# 9. Code solution

## === cell 0
import numpy as np
import glob



## === cell 1
pretrained_models = glob.glob(f"../input/densenet201-04-2019data/*.pth") + glob.glob(
    f"../input/eb7m-seed70/*.pth"
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
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


SEED = 42
seed_everything(seed=SEED)



## === cell 3
import sys

EfficientNet = (
    None  # placeholder to preserve name usage; we won't import efficientnet_pytorch.
)



## === cell 4
SIZE = 512  # image size
num_classes = 5



## === cell 5
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用デバイス: {device}")




## === cell 6
def resolve_base_dir():
    candidates = [
        "/kaggle/input/cassava-leaf-disease-classification",
        "/kaggle/data/cassava-leaf-disease-classification",
        "../input/cassava-leaf-disease-classification",
        "data/cassava-leaf-disease-classification",
        "/kaggle/input",
        "/kaggle/data",
    ]
    for c in candidates:
        if os.path.exists(c):
            if os.path.isdir(os.path.join(c, "test_images")) and os.path.isfile(
                os.path.join(c, "sample_submission.csv")
            ):
                return c
            nested = os.path.join(c, "cassava-leaf-disease-classification")
            if os.path.isdir(os.path.join(nested, "test_images")) and os.path.isfile(
                os.path.join(nested, "sample_submission.csv")
            ):
                return nested
    raise FileNotFoundError(
        "Could not resolve cassava-leaf-disease-classification dataset directory."
    )


def resolve_image_dir(base_dir: str, split: str) -> str:
    """
    split: 'test_images' or 'train_images'
    Handles accidental nesting like .../test_images/test_images by picking the folder that actually contains jpgs.
    """
    root = os.path.join(base_dir, split)
    nested = os.path.join(root, split)
    for d in [root, nested]:
        if os.path.isdir(d):
            jpgs = [f for f in os.listdir(d) if f.lower().endswith(".jpg")]
            if len(jpgs) > 0:
                return d
    if os.path.isdir(root):
        return root
    raise FileNotFoundError(f"Could not resolve image dir for {split} under {base_dir}")


BASE_DIR = resolve_base_dir()

run_type = os.getenv("KAGGLE_KERNEL_RUN_TYPE", "")
interactive = run_type == "Interactive"

if interactive:
    print("Test run in Kaggle environment (interactive).")
    TEST_PATH = resolve_image_dir(BASE_DIR, "train_images")
    test_files = sorted(
        [f for f in os.listdir(TEST_PATH) if f.lower().endswith(".jpg")]
    )[:32]
else:
    print("In Kaggle environment (batch/local).")
    TEST_PATH = resolve_image_dir(BASE_DIR, "test_images")
    test_files = sorted(
        [f for f in os.listdir(TEST_PATH) if f.lower().endswith(".jpg")]
    )

print(f"BASE_DIR: {BASE_DIR}")
print(f"TEST_PATH: {TEST_PATH}")
print(f"Number of test images: {len(test_files)}")



## === cell 7
df_test = pd.read_csv(f"{BASE_DIR}/sample_submission.csv")
assert "image_id" in df_test.columns and "label" in df_test.columns
if interactive:
    df_test = df_test.iloc[: len(test_files)].copy()



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
                A.HorizontalFlip(p=1),
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
                    scale=(0.08, 1.0),
                    ratio=(0.75, 1.3333333333333333),
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
                    scale=(0.08, 1.0),
                    ratio=(0.75, 1.3333333333333333),
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
                A.Rotate(limit=30, p=1),
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




## === cell 13
class FinalLayerMixupModelEN(nn.Module):
    def __init__(self, model, criterion, num_classes, alpha):
        super(FinalLayerMixupModelEN, self).__init__()

        if hasattr(model, "classifier") and isinstance(model.classifier, nn.Sequential):
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_features, num_classes)
        else:
            raise ValueError(
                "Unsupported EfficientNet model head; expected torchvision EfficientNet with .classifier"
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
        img_path = os.path.join(TEST_PATH, image_id)
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
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
            probability.append(torch.softmax(outputs, dim=1).cpu().numpy())

    print(f"{basename} time: {time.time() - model_start_time:.2f}[sec]")

    return np.concatenate(probability, axis=0)




## === cell 17
def _normalize_state_dict_keys(sd: dict) -> dict:
    if isinstance(sd, dict):
        for key in (
            "state_dict",
            "model_state_dict",
            "model",
            "net",
            "student",
            "teacher",
        ):
            if key in sd and isinstance(sd[key], dict):
                sd = sd[key]
                break

    if not isinstance(sd, dict):
        return sd

    out = {}
    for k, v in sd.items():
        nk = k[7:] if isinstance(k, str) and k.startswith("module.") else k
        out[nk] = v
    return out


def _strip_known_prefixes(k: str) -> str:
    for p in (
        "model.",
        "net.",
        "module.",
        "backbone.",
        "encoder.",
        "student.",
        "teacher.",
    ):
        if k.startswith(p):
            return k[len(p) :]
    return k


def _looks_like_resnet_backbone_key(k: str) -> bool:
    return k.startswith(
        (
            "conv1.",
            "bn1.",
            "layer1.",
            "layer2.",
            "layer3.",
            "layer4.",
        )
    )


def _remap_to_wrapper(sd: dict, wrapper: nn.Module) -> dict:
    """
    Change rationale (score-moving, minimal logic change):
    Your ~0.10 accuracy strongly suggests the trained weights aren't landing into wrapper.convlayer / wrapper.fc,
    leaving random heads. We keep architecture/inference unchanged and only map keys more correctly.
    """
    if not isinstance(sd, dict):
        return sd

    sd2 = {}
    for k, v in sd.items():
        if isinstance(k, str):
            sd2[_strip_known_prefixes(k)] = v
        else:
            sd2[k] = v

    if any(
        isinstance(k, str)
        and (
            k.startswith("convlayer.")
            or k.startswith("fc.")
            or k.startswith("model.")
            or k.startswith("AdaptiveAvgPool2d.")
        )
        for k in sd2.keys()
    ):
        return sd2

    if isinstance(wrapper, FinalLayerMixupModel):
        remapped = {}
        for k, v in sd2.items():
            if not isinstance(k, str):
                continue

            if k.startswith("fc."):
                remapped["fc." + k[len("fc.") :]] = v
                continue
            if k.startswith("classifier."):
                remapped["fc." + k[len("classifier.") :]] = v
                continue

            if _looks_like_resnet_backbone_key(k) or k.startswith("downsample."):
                remapped["convlayer." + k] = v
                continue

            remapped["convlayer." + k] = v
        return remapped

    if isinstance(wrapper, FinalLayerMixupModelDenseNet):
        remapped = {}
        for k, v in sd2.items():
            if not isinstance(k, str):
                continue
            if k.startswith("features."):
                remapped["convlayer." + k[len("features.") :]] = v
            elif k.startswith("classifier."):
                remapped["fc." + k[len("classifier.") :]] = v
            elif k.startswith("fc."):
                remapped["fc." + k[len("fc.") :]] = v
            else:
                remapped[k] = v
        return remapped

    if isinstance(wrapper, FinalLayerMixupModelEN):
        remapped = {}
        for k, v in sd2.items():
            if not isinstance(k, str):
                continue

            if k.startswith("features.") or k.startswith("classifier."):
                remapped["model." + k] = v
            elif k.startswith("model."):
                remapped[k] = v
            else:
                remapped["model." + k] = v
        return remapped

    return sd2


def _assert_loaded_sufficient(wrapper: nn.Module, missing_keys: list, basename: str):
    """
    Change rationale: failing fast prevents silently producing near-random submissions.
    We require both: (A) classifier head loaded and (B) some backbone weights loaded.
    """
    if isinstance(wrapper, (FinalLayerMixupModel, FinalLayerMixupModelDenseNet)):
        head_prefix = "fc."
        backbone_prefix = "convlayer."
    elif isinstance(wrapper, FinalLayerMixupModelEN):
        head_prefix = "model.classifier."
        backbone_prefix = "model.features."
    else:
        head_prefix = "fc."
        backbone_prefix = ""

    head_missing = [
        k for k in missing_keys if isinstance(k, str) and k.startswith(head_prefix)
    ]
    if len(head_missing) > 0:
        raise RuntimeError(
            f"{basename}: classifier head weights not loaded (missing {len(head_missing)} keys under '{head_prefix}'). "
            f"Missing examples: {head_missing[:5]}"
        )

    if backbone_prefix:
        bb_missing = [
            k
            for k in missing_keys
            if isinstance(k, str) and k.startswith(backbone_prefix)
        ]
        if len(bb_missing) > 50:
            raise RuntimeError(
                f"{basename}: backbone appears not loaded correctly (missing {len(bb_missing)} keys under '{backbone_prefix}'). "
                f"This would likely yield very low accuracy. Missing examples: {bb_missing[:5]}"
            )




## === cell 18
probability = []
start_time = time.time()

if len(pretrained_models) == 0:
    print(
        "No pretrained .pth found in ../input; falling back to torchvision resnet50 pretrained on ImageNet for a valid submission."
    )
    pretrained_models = ["__torchvision_fallback_resnet50__"]

for pretrained_model in pretrained_models:
    basename = (
        os.path.splitext(os.path.basename(pretrained_model))[0]
        if pretrained_model != "__torchvision_fallback_resnet50__"
        else "torchvision_resnet50_imagenet"
    )

    criterion = nn.CrossEntropyLoss()

    if pretrained_model == "__torchvision_fallback_resnet50__":
        MODEL_NAME = "resnet50"
        weights = models.ResNet50_Weights.DEFAULT
        net = models.resnet50(weights=weights)
        net = FinalLayerMixupModel(net, criterion, num_classes, False)
        BATCH_SIZE = 32
    else:
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
            sys.exit()

        print(f"{basename}: {MODEL_NAME}")

        sd = torch.load(pretrained_model, map_location="cpu")
        sd = _normalize_state_dict_keys(sd)
        sd = _remap_to_wrapper(sd, net)

        missing, unexpected = net.load_state_dict(sd, strict=False)
        if len(missing) > 0 or len(unexpected) > 0:
            print(
                f"Warning load_state_dict(strict=False): missing={len(missing)} unexpected={len(unexpected)}"
            )

        _assert_loaded_sufficient(net, missing, basename)

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
                num_workers=min(4, os.cpu_count() or 1),
                pin_memory=(device == "cuda"),
            ),
        }

        proba = predict_model(basename, net, dataloader)
        probability.append(proba)

    del net
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if len(probability) == 0:
    df_test["mean"] = 0
else:
    prob_arr = np.stack(probability, axis=0)  # (n_preds, n_images, n_classes)
    df_test["mean"] = prob_arr.mean(axis=0).argmax(axis=1)

print(f"total time: {time.time() - start_time:.2f}[sec]")



## === cell 19
df_test["label"] = df_test["mean"].astype(int)



## === cell 20
df_test.head()



## === cell 21
sub_path = "submission.csv"
df_test[["image_id", "label"]].to_csv(sub_path, index=False)
print(
    f"Wrote submission to: {sub_path} with shape={df_test[['image_id','label']].shape}"
)
print(df_test[["image_id", "label"]].head())
