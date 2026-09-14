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

0.8955877908733756

# 6. Current score

0.75075

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.11584) has done: 'I fix the missing `efficientnet_pytorch` dependency by swapping it to `torchvision.models.efficientnet_b7` while keeping the same “replace final FC + load weights + TTA ensemble” core inference logic. I also make the dataset path detection robust for this environment so it always finds `../input/cassava-leaf-disease-classification/test_images` and creates `df_test` correctly. Albumentations v2 changed `RandomResizedCrop`’s signature, so I update those calls to the v2 API to eliminate the validation error. Finally, I guard against the “no models found” case (which caused the axis error) by falling back to a valid baseline submission from `sample_submission.csv`, ensuring a proper `submission.csv` is always produced.'
- What this solution (achieved 0.18012) has done: 'Your current 0.11584 score is consistent with an invalid/degenerate inference setup: in the “local environment” branch you only predict on the first 32 images, and if no pretrained weights are found you fall back to `sample_submission.csv` without producing any real predictions. To move the score toward the 0.8956 target, the smallest safe fix is to always run inference on the full Kaggle test set (all 2676 images) and to load at least a reasonable pretrained backbone when the competition weight files are missing. I keep your core logic (same model wrappers, same TTA averaging, same argmax) but (1) make `TEST_PATH` always point to the real `test_images` when it exists, (2) build `df_test` from `sample_submission.csv` to guarantee correct ordering/rowcount, and (3) if no `.pth` models are found, use a torchvision ImageNet-pretrained EfficientNet-B7 with its native classifier head (no wrapper) as a legitimate baseline so predictions are non-constant and cover all rows.'
- What this solution (achieved 0.57511) has done: 'Your current score is low mainly because the fallback path uses an ImageNet-pretrained EfficientNet-B7 but replaces its classifier with a random 5-class head, making predictions essentially random. To move toward the 0.8956 target with minimal change and without altering the overall inference/TTA/argmax logic, I keep the same EfficientNet-B7 baseline but calibrate its logits using the training-set label prior (computed from `train.csv`) to map ImageNet outputs into the 5 cassava classes in a reasonable, fully unsupervised-on-test way. This preserves your end-to-end pipeline and still writes a valid `submission.csv`, while typically producing a big jump versus random guessing. I also make the baseline deterministic and ensure ordering stays exactly as `sample_submission.csv`.'
- What this solution (achieved 0.75075) has done: 'Your score is far below the target, and the biggest bottleneck is the fallback mode: you’re applying random crops at test-time and a very weak “ImageNet→5 classes” mapping, which injects large noise and collapses accuracy. I keep your overall inference/TTA/argmax pipeline intact, but make the fallback deterministic and less destructive by (1) removing stochastic `RandomResizedCrop` from test TTA (replace with resize+centercrop variants) and (2) replacing the “sum every 5th ImageNet class” hack with a simple learned linear mapping from EfficientNet-B7 1000-d logits to 5 classes fitted on `train.csv` using the same frozen ImageNet model (no architectural/training loop changes; it’s just a closed-form ridge regression). This remains fully legitimate (uses only training labels), still produces `submission.csv`, and should move accuracy substantially toward the 0.8956 target without touching your pretrained-.pth ensemble path.'

# 9. Code solution

## === cell 0
import numpy as np
import glob



## === cell 1
pretrained_models = (
    glob.glob(f"../input/densenet201-04-2019data/*.pth")
    + glob.glob(f"../input/eb7m-seed70/*.pth")
    + glob.glob(f"../input/eb7mseed71/*.pth")
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
import sys

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
from torchvision.models import efficientnet_b7



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
    "../kaggle/input/cassava-leaf-disease-classification",
    "data/cassava-leaf-disease-classification",
    "/kaggle/data/cassava-leaf-disease-classification",
]
BASE_DIR = None
for p in CANDIDATE_BASE_DIRS:
    if os.path.isdir(p):
        BASE_DIR = p
        break
if BASE_DIR is None:
    BASE_DIR = "data"

if os.path.isdir(f"{BASE_DIR}/test_images"):
    TEST_PATH = f"{BASE_DIR}/test_images"
elif os.path.isdir(f"{BASE_DIR}/cassava-leaf-disease-classification/test_images"):
    TEST_PATH = f"{BASE_DIR}/cassava-leaf-disease-classification/test_images"
else:
    TEST_PATH = f"{BASE_DIR}/train_images"

print(f"BASE_DIR: {BASE_DIR}")
print(f"TEST_PATH: {TEST_PATH}")

SAMPLE_SUB_PATH = (
    f"{BASE_DIR}/sample_submission.csv"
    if os.path.isfile(f"{BASE_DIR}/sample_submission.csv")
    else "../input/cassava-leaf-disease-classification/sample_submission.csv"
)
df_test = pd.read_csv(SAMPLE_SUB_PATH)
print("Loaded sample_submission:", df_test.shape)

missing = [
    img
    for img in df_test["image_id"].tolist()[:10]
    if not os.path.isfile(f"{TEST_PATH}/{img}")
]
if len(missing) > 0:
    print("Warning: some images not found under TEST_PATH for first 10 ids:", missing)

print(f"Number of test images (from sample_submission): {len(df_test)}")

TRAIN_CSV_PATH = (
    f"{BASE_DIR}/train.csv"
    if os.path.isfile(f"{BASE_DIR}/train.csv")
    else "../input/cassava-leaf-disease-classification/train.csv"
)
if os.path.isfile(TRAIN_CSV_PATH):
    df_train = pd.read_csv(TRAIN_CSV_PATH)
    prior = df_train["label"].value_counts(normalize=True).sort_index()
    prior = (
        prior.reindex(range(num_classes))
        .fillna(1.0 / num_classes)
        .values.astype(np.float64)
    )
else:
    df_train = None
    prior = np.ones(num_classes, dtype=np.float64) / num_classes
prior = prior / prior.sum()
log_prior = np.log(prior + 1e-12)
print("Train label prior:", prior)

TRAIN_IMG_PATH = None
if os.path.isdir(f"{BASE_DIR}/train_images"):
    TRAIN_IMG_PATH = f"{BASE_DIR}/train_images"
elif os.path.isdir(f"{BASE_DIR}/cassava-leaf-disease-classification/train_images"):
    TRAIN_IMG_PATH = f"{BASE_DIR}/cassava-leaf-disease-classification/train_images"
else:
    TRAIN_IMG_PATH = f"{BASE_DIR}/train_images"
print(f"TRAIN_IMG_PATH: {TRAIN_IMG_PATH}")



## === cell 7
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

transform = {
    "test": [
        Compose(
            [
                A.Resize(height=SIZE, width=SIZE),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                A.Resize(height=SIZE, width=SIZE),
                A.HorizontalFlip(p=1.0),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                A.SmallestMaxSize(max_size=SIZE),
                A.CenterCrop(height=SIZE, width=SIZE),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                A.SmallestMaxSize(max_size=SIZE),
                A.CenterCrop(height=SIZE, width=SIZE),
                A.HorizontalFlip(p=1.0),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
    ]
}



## === cell 8
pass




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

        if hasattr(model, "classifier") and isinstance(model.classifier, nn.Sequential):
            num_ftrs = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(num_ftrs, num_classes)
        else:
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




## === cell 12
pass




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
            raise FileNotFoundError(f"Image not found/readable: {TEST_PATH}/{image_id}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        img = self.load_image(image_id)

        if self.transform:
            img = self.transform(image=img)["image"]

        return img, image_id




## === cell 14
class TrainDataset(data.Dataset):
    def __init__(self, df, img_dir, transform=None):
        super().__init__()
        self.image_ids = df.image_id.tolist()
        self.labels = df.label.astype(int).tolist()
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        label = self.labels[idx]
        img = cv2.imread(f"{self.img_dir}/{image_id}")
        if img is None:
            raise FileNotFoundError(
                f"Image not found/readable: {self.img_dir}/{image_id}"
            )
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(image=img)["image"]
        return img, label




## === cell 15
def predict_model(basename, net, dataloader, use_wrapper=True):
    """
    basename: 学習済みモデル名
    net     : 学習済みモデル
    use_wrapper: True if net.forward expects (inputs, labels, phase); False if standard torchvision forward(inputs)
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
            if use_wrapper:
                outputs = net(inputs, False, "test")
            else:
                outputs = net(inputs)
            probability.append(torch.softmax(outputs, dim=1).cpu().numpy())

    print(f"{basename} time: {time.time() - model_start_time:.2f}[sec]")
    return np.concatenate(probability, axis=0)




## === cell 16
def extract_logits_1000(net, dataloader, desc="extract"):
    net.to(device)
    net.eval()
    torch.set_grad_enabled(False)

    outs = []
    ys = []
    progress = tqdm(dataloader, desc=desc)
    for batch in progress:
        if len(batch) == 2:
            x, y = batch
            ys.append(y.numpy())
        else:
            x = batch[0]
        x = x.to(device)
        logits = net(x)  # (B,1000)
        outs.append(logits.detach().cpu().numpy())
    X = np.concatenate(outs, axis=0)
    y = np.concatenate(ys, axis=0) if len(ys) > 0 else None
    return X, y


def fit_ridge_multiclass(X, y, num_classes=5, reg=10.0):
    n, d = X.shape
    Y = np.zeros((n, num_classes), dtype=np.float64)
    Y[np.arange(n), y.astype(int)] = 1.0
    XtX = X.T @ X
    XtX_reg = XtX + reg * np.eye(d, dtype=np.float64)
    XtY = X.T @ Y
    W = np.linalg.solve(XtX_reg, XtY)  # (d, C)
    return W.astype(np.float32)


def softmax_np(z):
    z = z - z.max(axis=1, keepdims=True)
    ez = np.exp(z)
    return ez / (ez.sum(axis=1, keepdims=True) + 1e-12)




## === cell 17
probability = []
start_time = time.time()

if len(pretrained_models) == 0:
    print(
        "No pretrained .pth models found; using torchvision EfficientNet-B7 ImageNet-pretrained baseline."
    )
    try:
        from torchvision.models import EfficientNet_B7_Weights

        net = efficientnet_b7(weights=EfficientNet_B7_Weights.IMAGENET1K_V1)
    except Exception as e:
        print(
            "Could not load EfficientNet_B7_Weights; falling back to weights=None. Error:",
            repr(e),
        )
        net = efficientnet_b7(weights=None)

    W = None
    if df_train is not None and os.path.isdir(TRAIN_IMG_PATH):
        df_train_sub = df_train.sample(
            n=min(2048, len(df_train)), random_state=SEED
        ).reset_index(drop=True)

        train_tf = Compose(
            [
                A.SmallestMaxSize(max_size=SIZE),
                A.CenterCrop(height=SIZE, width=SIZE),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        )
        train_ds = TrainDataset(
            df_train_sub, img_dir=TRAIN_IMG_PATH, transform=train_tf
        )
        train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=8,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
        Xtr, ytr = extract_logits_1000(
            net, train_loader, desc="Fitting linear map (train logits)"
        )
        W = fit_ridge_multiclass(
            Xtr.astype(np.float64), ytr, num_classes=num_classes, reg=20.0
        )
        del Xtr, ytr, train_ds, train_loader
        torch.cuda.empty_cache()
        print("Fitted linear mapping W with shape:", W.shape)
    else:
        print(
            "Could not fit linear mapping (missing train.csv or train_images); using prior-weighted heuristic."
        )

    BATCH_SIZE = 10
    basename = "efficientnet-b7-imagenet"

    for tid, transform_ in enumerate(transform["test"]):
        print(f"transform loop={tid}")
        dataset = {"test": TestDataset(df_test, transform=transform_)}
        dataloader = {
            "test": torch.utils.data.DataLoader(
                dataset["test"],
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=4,
                pin_memory=True,
            )
        }

        outs = []
        net.to(device)
        net.eval()
        torch.set_grad_enabled(False)
        for inputs, _image_ids in tqdm(dataloader["test"], desc=f"{basename} logits: "):
            inputs = inputs.to(device)
            outs.append(net(inputs).detach().cpu().numpy())
        logits_1000 = np.concatenate(outs, axis=0)

        if W is not None:
            logits5 = logits_1000 @ W  # (N,5)
            proba5 = softmax_np(logits5.astype(np.float64))
        else:
            proba_1000 = softmax_np(logits_1000.astype(np.float64))
            n = proba_1000.shape[0]
            proba5 = np.zeros((n, num_classes), dtype=np.float64)
            for c in range(num_classes):
                proba5[:, c] = proba_1000[:, c::num_classes].sum(axis=1)
            proba5 = proba5 * prior.reshape(1, -1)
            proba5 = proba5 / (proba5.sum(axis=1, keepdims=True) + 1e-12)

        probability.append(proba5.astype(np.float32))

    del net
    torch.cuda.empty_cache()

else:
    for pretrained_model in pretrained_models:
        basename = os.path.splitext(os.path.basename(pretrained_model))[0]

        criterion = nn.CrossEntropyLoss()

        if "resnet18" in basename:
            MODEL_NAME = "resnet18"
            net = models.resnet18(pretrained=False)
            net = FinalLayerMixupModel(net, criterion, num_classes, False)
            BATCH_SIZE = 64
        elif "resnet50" in basename:
            MODEL_NAME = "resnet50"
            net = models.resnet50(pretrained=False)
            net = FinalLayerMixupModel(net, criterion, num_classes, False)
            BATCH_SIZE = 32
        elif "resnet152" in basename:
            MODEL_NAME = "resnet152"
            net = models.resnet152(pretrained=False)
            net = FinalLayerMixupModel(net, criterion, num_classes, False)
            BATCH_SIZE = 16
        elif "resnext101" in basename:
            MODEL_NAME = "resnext101"
            net = models.resnext101_32x8d(pretrained=False)
            net = FinalLayerMixupModel(net, criterion, num_classes, False)
            BATCH_SIZE = 12
        elif "densenet201" in basename:
            MODEL_NAME = "densenet201"
            net = models.densenet201(pretrained=False)
            net = FinalLayerMixupModelDenseNet(net, criterion, num_classes, False)
            BATCH_SIZE = 12
        elif "efficientnet-b7" in basename:
            MODEL_NAME = "efficientnet-b7"
            net = efficientnet_b7(weights=None)
            net = FinalLayerMixupModelEN(net, criterion, num_classes, False)
            BATCH_SIZE = 10
        else:
            print(f"{basename} is not supported.")
            sys.exit()

        print(f"{basename}: {MODEL_NAME}")

        state = torch.load(pretrained_model, map_location="cpu")
        try:
            net.load_state_dict(state)
        except RuntimeError:
            if isinstance(state, dict) and "state_dict" in state:
                sd = state["state_dict"]
            else:
                sd = state
            new_sd = {}
            for k, v in sd.items():
                nk = k.replace("module.", "")
                new_sd[nk] = v
            net.load_state_dict(new_sd, strict=False)

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
                    num_workers=4,
                    pin_memory=True,
                )
            }
            proba = predict_model(basename, net, dataloader, use_wrapper=True)
            probability.append(proba)

        del net
        torch.cuda.empty_cache()

proba_mean = np.mean(np.stack(probability, axis=0), axis=0)
df_test["mean"] = proba_mean.argmax(axis=1)

print(f"total time: {time.time() - start_time:.2f}[sec]")



## === cell 18
if "mean" in df_test.columns:
    df_test["label"] = df_test["mean"]

df_test = df_test[["image_id", "label"]]



## === cell 19
df_test.head()



## === cell 20
df_test[["image_id", "label"]].to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", df_test.shape)
print(df_test.head())
