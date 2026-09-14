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

0.05531

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.05531) has done: 'I remove the hard dependency on `efficientnet_pytorch` (it isn’t installed) by skipping EfficientNet weights if present, while keeping the existing ResNet/DenseNet ensemble logic unchanged. I fix the Kaggle path detection so it always reads test images from the correct `/kaggle/input/...` location, preventing the `FileNotFoundError` cascade. I update the Albumentations `RandomResizedCrop` call to the v2 API (uses `size=(h,w)`), which currently crashes at transform construction. Finally, I make the probability aggregation robust (avoid the `AxisError` when no models are found or when shapes don’t stack), and always produce a valid `submission.csv` with the required columns by falling back to `sample_submission.csv` when needed.'
- What this solution (achieved 0.05531) has done: 'Your low score is consistent with a label/index misalignment bug: you’re averaging predictions in the filesystem-sorted `test_images` order, but then mapping them onto `sample_submission.csv`’s `image_id` order via a dict; if those orders differ, many predictions attach to the wrong image and accuracy collapses. I make the test dataframe follow `sample_submission.csv` exactly (same image_id order) and iterate over that order when loading images, so predictions and submission rows stay aligned. I also switch the single-image crop to a resize+center-crop style that won’t fail when images are smaller than 512, preventing silent data issues; this keeps the same “test-time augmentation ensemble” core logic intact. Finally, I keep everything else (models, weights loading, averaging, argmax) the same to preserve semantics while lifting the score toward your target.'
- What this solution (achieved 0.05531) has done: 'The very low accuracy is most consistent with the ensemble never actually running in this environment (your `pretrained_models` glob points to datasets that likely aren’t mounted), so the code falls back to predicting all zeros and scores ~class-prior level. I keep your model wrappers, transforms, and prediction/averaging logic unchanged, but fix model discovery to also look for `.pth` files anywhere under the provided cassava input tree so real weights are found and used. I also make `num_workers` adapt to the environment to avoid dataloader worker failures/timeouts, and keep the test order aligned to `sample_submission.csv` (already correct) to prevent accidental misalignment. The output still be a standard `submission.csv` with `image_id,label`.'
- What this solution (achieved 0.05531) has done: 'Your current score strongly suggests the loaded checkpoints don’t match the wrapped model keys, so `load_state_dict` is either failing silently elsewhere or loading the wrong tensors, leading to essentially random predictions. I keep your exact ensemble/TTA prediction core logic, but make checkpoint loading robust by (1) handling common checkpoint formats (`state_dict`, `model`, `net`) and (2) auto-stripping/adding the `fc.` prefix depending on whether the saved weights came from the wrapper or the raw backbone. I also ensure we don’t accidentally point `TEST_PATH` at `train_images` (which can desync from the true test set), by preferring the real `test_images` folder when present. These minimal fixes should move accuracy sharply upward toward your target without changing architecture, transforms, or averaging semantics.'
- What this solution (achieved 0.05531) has done: 'Your score is extremely low versus the target, so the most likely cause is that the model head weights are not being loaded into the wrapper’s `fc` layer (your current prefix-remap only handles `classifier.*`→`fc.*`, but most ResNet checkpoints save `fc.*` or `model.fc.*` and your wrapper expects `fc.*`). I keep your exact ensemble/TTA inference logic, but make checkpoint loading robust by (1) mapping common checkpoint key patterns like `model.fc.*`, `net.fc.*`, `module.fc.*` into the wrapper’s `fc.*`, and (2) separately trying to load backbone weights into `net.convlayer` while always loading head weights into `net.fc`, so you don’t end up with a random head. I also force `cudnn.benchmark=False` when `deterministic=True` to avoid non-deterministic kernel selection and subtle instability. These minimal changes should move accuracy sharply upward toward your target without changing the modeling approach or submission format.'
- What this solution (achieved 0.05531) has done: 'Your score is far below the target, so the most likely issue is that the ensemble is still effectively “untrained” at inference because checkpoints aren’t being matched to the wrapper modules (especially for ResNet: checkpoint backbone keys are usually `layer1.*` etc., but your wrapper’s backbone lives under `convlayer.*`). I keep your exact model wrappers, transforms/TTA loops, and averaging/argmax logic, but make checkpoint loading map common raw-backbone keys into `convlayer.*` and also correctly map DenseNet `features.*` into `convlayer.*`, while still loading `fc.*` into the wrapper head. I also stop scanning all of `/kaggle/input` for arbitrary `.pth` files (it can pick up unrelated weights) and instead prioritize only cassava-related folders/known model datasets, which reduces the chance of loading wrong weights and dragging accuracy down. These are minimal inference-only fixes that should move accuracy sharply upward toward your target while preserving the solution’s core behavior and producing the same `submission.csv` format.'
- What this solution (achieved 0.05531) has done: 'Your current score is far below the target, so the most probable issue is still “wrong/partial weights loaded” causing near-random predictions. I keep your exact ensemble + TTA inference flow, but make checkpoint loading safer by (1) selecting the correct state_dict when the checkpoint is nested and (2) remapping keys by *matching* to the wrapper’s expected keys (instead of blindly prefixing everything with `convlayer.`), which commonly breaks loading for ResNet/DenseNet wrappers. I also skip checkpoints that load with too many missing keys (likely wrong weights) to avoid dragging the mean down, while still producing a valid `submission.csv`. These are minimal inference-only changes and should move accuracy sharply upward toward your target.'

# 9. Code solution

## === cell 0
import numpy as np
import glob



## === cell 1
import os


def find_pth_files(search_roots):
    out = []
    for root in search_roots:
        if not root or not os.path.isdir(root):
            continue
        for dp, _, fn in os.walk(root):
            for f in fn:
                if f.lower().endswith(".pth"):
                    out.append(os.path.join(dp, f))
    return sorted(set(out))


pretrained_models = glob.glob(f"../input/densenet201-04-2019data/*.pth") + glob.glob(
    f"../input/eb7m-seed70/*.pth"
)

extra_roots = [
    "../input/cassava-leaf-disease-classification",
    "/kaggle/input/cassava-leaf-disease-classification",
    "../input/densenet201-04-2019data",
    "/kaggle/input/densenet201-04-2019data",
    "../input/eb7m-seed70",
    "/kaggle/input/eb7m-seed70",
]
pretrained_models = sorted(set(pretrained_models + find_pth_files(extra_roots)))

print(f"{len(pretrained_models)} models found.")
print("\n".join(np.sort(pretrained_models)[:200]))
if len(pretrained_models) > 200:
    print(f"... (showing first 200 of {len(pretrained_models)})")



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
    torch.backends.cudnn.benchmark = False


SEED = 42
seed_everything(seed=SEED)



## === cell 3
try:
    from efficientnet_pytorch import EfficientNet  # type: ignore

    HAS_EFFICIENTNET_PYTORCH = True
except Exception as e:
    EfficientNet = None
    HAS_EFFICIENTNET_PYTORCH = False
    print(
        "efficientnet_pytorch not available; EfficientNet models (if any) will be skipped."
    )



## === cell 4
SIZE = 512  # image size
num_classes = 5



## === cell 5
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用デバイス: {device}")



## === cell 6
KAGGLE_BASE = "../input/cassava-leaf-disease-classification"
if not os.path.isdir(KAGGLE_BASE):
    KAGGLE_BASE = "/kaggle/input/cassava-leaf-disease-classification"

if os.path.isdir(KAGGLE_BASE):
    BASE_DIR = KAGGLE_BASE
    TEST_PATH = f"{BASE_DIR}/test_images"
    if not os.path.isdir(TEST_PATH):
        TEST_PATH = f"{BASE_DIR}/train_images"
    print("In Kaggle environment.")
else:
    print("In the local environment.")
    BASE_DIR = "data"
    TEST_PATH = f"{BASE_DIR}/test_images"
    if not os.path.isdir(TEST_PATH):
        TEST_PATH = f"{BASE_DIR}/train_images"

print(f"TEST_PATH: {TEST_PATH}")

sample_path = f"{BASE_DIR}/sample_submission.csv"
sample_sub = pd.read_csv(sample_path)
df_test = sample_sub[["image_id"]].copy()
df_test["label"] = 1

print(f"Number of test images (from sample_submission): {len(df_test)}")
print("First 5 image_ids:", df_test["image_id"].head().tolist())



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
                A.SmallestMaxSize(max_size=SIZE, p=1.0),
                A.CenterCrop(SIZE, SIZE, p=1.0),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                A.HorizontalFlip(p=1),
                A.SmallestMaxSize(max_size=SIZE, p=1.0),
                A.CenterCrop(SIZE, SIZE, p=1.0),
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
                A.VerticalFlip(p=1),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                A.Rotate(p=1),
                A.RandomResizedCrop(size=(SIZE, SIZE)),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
    ]
}



## === cell 9
pass




## === cell 10
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




## === cell 11
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




## === cell 12
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




## === cell 13
pass




## === cell 14
class TestDataset(data.Dataset):
    def __init__(self, df, transform=None):
        super().__init__()

        self.image_ids = df.image_id.tolist()
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def load_image(self, image_id):
        img = cv2.imread(f"{TEST_PATH}/{image_id}")  # (H, W, C)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {TEST_PATH}/{image_id}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        img = self.load_image(image_id)
        if self.transform:
            img = self.transform(image=img)["image"]
        return img, image_id




## === cell 15
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
    torch.backends.cudnn.benchmark = False

    probability = []

    for phase in ["test"]:
        progress = tqdm(dataloader[phase], desc=f"{basename}: ")

        for inputs, image_ids in progress:
            inputs = inputs.to(device)

            outputs = net(inputs, False, "test")
            probability.append(torch.softmax(outputs, dim=1).cpu().numpy())

    print(f"{basename} time: {time.time() - model_start_time:.2f}[sec]")
    return np.concatenate(probability, axis=0)




## === cell 16
def _extract_state_dict(ckpt_obj):
    if isinstance(ckpt_obj, dict):
        for k in ["state_dict", "model", "net", "weights"]:
            if k in ckpt_obj and isinstance(ckpt_obj[k], dict):
                return ckpt_obj[k]
    return ckpt_obj


def _maybe_strip_any_prefix(sd, prefixes):
    if not isinstance(sd, dict):
        return sd
    changed = True
    while changed:
        changed = False
        for p in prefixes:
            if all(isinstance(k, str) and k.startswith(p) for k in sd.keys()):
                sd = {k[len(p) :]: v for k, v in sd.items()}
                changed = True
    return sd


def _remap_prefix(sd, src_prefix, dst_prefix):
    if not isinstance(sd, dict):
        return sd
    out = {}
    changed = False
    for k, v in sd.items():
        if isinstance(k, str) and k.startswith(src_prefix):
            out[dst_prefix + k[len(src_prefix) :]] = v
            changed = True
        else:
            out[k] = v
    return out if changed else sd


def _select_matching_keys(sd, target_keys):
    """
    Change (score): only keep keys that can actually load into the given wrapper.
    This avoids the previous behavior of prefixing *everything* with `convlayer.` which often
    makes weights unusable and yields near-random predictions.
    """
    if not isinstance(sd, dict):
        return sd
    if not target_keys:
        return sd

    direct = {k: v for k, v in sd.items() if k in target_keys}
    if len(direct) > 0:
        return direct

    conv_prefixed = {}
    for k, v in sd.items():
        if not isinstance(k, str):
            continue
        kk = k
        if kk in target_keys:
            conv_prefixed[kk] = v
            continue
        if ("convlayer." + kk) in target_keys:
            conv_prefixed["convlayer." + kk] = v
            continue
        if (
            kk.startswith("features.")
            and ("convlayer." + kk[len("features.") :]) in target_keys
        ):
            conv_prefixed["convlayer." + kk[len("features.") :]] = v
            continue
        if (
            kk.startswith("classifier.")
            and ("fc." + kk[len("classifier.") :]) in target_keys
        ):
            conv_prefixed["fc." + kk[len("classifier.") :]] = v
            continue
        if kk.startswith("head.") and ("fc." + kk[len("head.") :]) in target_keys:
            conv_prefixed["fc." + kk[len("head.") :]] = v
            continue
        if (
            kk.startswith("last_linear.")
            and ("fc." + kk[len("last_linear.") :]) in target_keys
        ):
            conv_prefixed["fc." + kk[len("last_linear.") :]] = v
            continue
        if kk.startswith("_fc.") and ("fc." + kk[len("_fc.") :]) in target_keys:
            conv_prefixed["fc." + kk[len("_fc.") :]] = v
            continue

    return conv_prefixed if len(conv_prefixed) > 0 else sd


def load_checkpoint_flexible(net, pretrained_model):
    """
    Change (score): robustly choose the correct sd and keep only wrapper-compatible keys.
    Then require that most parameters load; otherwise skip this checkpoint to avoid degrading the mean.
    """
    ckpt = torch.load(pretrained_model, map_location="cpu")
    sd = _extract_state_dict(ckpt)

    if not isinstance(sd, dict):
        return False, 10**9, 10**9, 0.0

    sd = _maybe_strip_any_prefix(sd, prefixes=["module.", "model.", "net."])

    sd = _remap_prefix(sd, "classifier.", "fc.")
    sd = _remap_prefix(sd, "head.", "fc.")
    sd = _remap_prefix(sd, "last_linear.", "fc.")
    sd = _remap_prefix(sd, "_fc.", "fc.")

    target_keys = set(net.state_dict().keys())
    sd = _select_matching_keys(sd, target_keys)

    missing, unexpected = net.load_state_dict(sd, strict=False)
    total = len(target_keys)
    loaded = total - len(missing)
    loaded_ratio = loaded / max(total, 1)

    return True, len(missing), len(unexpected), loaded_ratio


probability = []

start_time = time.time()

supported_models_used = 0

MAX_WORKERS = os.cpu_count() or 2
NUM_WORKERS = min(4, MAX_WORKERS)  # conservative for Kaggle-like envs
print(f"DataLoader num_workers set to: {NUM_WORKERS}")

MIN_LOADED_RATIO = 0.60

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
        BATCH_SIZE = 16
    elif "densenet201" in basename:
        MODEL_NAME = "densenet201"
        net = models.densenet201(weights=None)
        net = FinalLayerMixupModelDenseNet(net, criterion, num_classes, False)
        BATCH_SIZE = 16
    elif "efficientnet-b7" in basename:
        MODEL_NAME = "efficientnet-b7"
        if not HAS_EFFICIENTNET_PYTORCH:
            print(
                f"{basename}: EfficientNet skipped (efficientnet_pytorch not installed)."
            )
            continue
        net = EfficientNet.from_name(MODEL_NAME)
        net = FinalLayerMixupModelEN(net, criterion, num_classes, False)
        BATCH_SIZE = 16
    else:
        print(f"{basename} is not supported. Skipping.")
        continue

    print(f"{basename}: {MODEL_NAME}")

    if MODEL_NAME == "efficientnet-b7":
        state = torch.load(pretrained_model, map_location="cpu")
        state = _extract_state_dict(state)
        if isinstance(state, dict):
            state = _maybe_strip_any_prefix(
                state, prefixes=["module.", "model.", "net."]
            )
        try:
            net.model.load_state_dict(state, strict=True)
            loaded_ratio = 1.0
            print(f"{basename}: loaded (strict=True)")
        except Exception:
            missing, unexpected = net.model.load_state_dict(state, strict=False)
            loaded_ratio = 1.0 - (len(missing) / max(len(net.model.state_dict()), 1))
            print(
                f"{basename}: loaded (strict=False) missing={len(missing)} unexpected={len(unexpected)} loaded_ratio={loaded_ratio:.3f}"
            )
        if loaded_ratio < MIN_LOADED_RATIO:
            print(
                f"{basename}: skip (loaded_ratio {loaded_ratio:.3f} < {MIN_LOADED_RATIO})"
            )
            del net
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue
    else:
        ok, n_missing, n_unexpected, loaded_ratio = load_checkpoint_flexible(
            net, pretrained_model
        )
        print(
            f"{basename}: loaded (strict=False) missing={n_missing} unexpected={n_unexpected} loaded_ratio={loaded_ratio:.3f}"
        )
        if loaded_ratio < MIN_LOADED_RATIO:
            print(
                f"{basename}: skip (loaded_ratio {loaded_ratio:.3f} < {MIN_LOADED_RATIO})"
            )
            del net
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

    supported_models_used += 1

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
                num_workers=NUM_WORKERS,
                pin_memory=True,
            )
        }
        proba = predict_model(basename, net, dataloader)
        probability.append(proba)

    del net
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print(f"total time: {time.time() - start_time:.2f}[sec]")
print(
    f"supported_models_used: {supported_models_used}, total_pred_chunks: {len(probability)}"
)

if len(probability) > 0:
    prob_stack = np.stack(probability, axis=0)  # (n_chunks, n_samples, n_classes)
    df_test["mean"] = prob_stack.mean(axis=0).argmax(axis=1)
else:
    df_test["mean"] = 0



## === cell 17
sample_sub = sample_sub.copy()
sample_sub["label"] = df_test["mean"].astype(int).values
df_test = sample_sub[["image_id", "label"]].copy()



## === cell 18
df_test.head()



## === cell 19
df_test[["image_id", "label"]].to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", df_test[["image_id", "label"]].shape)
print(df_test[["image_id", "label"]].head())
