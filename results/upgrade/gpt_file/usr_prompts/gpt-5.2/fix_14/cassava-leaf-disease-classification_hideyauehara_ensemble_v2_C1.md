# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import numpy as np
import glob



## === cell 1
import os
from pathlib import Path

pretrained_models = []

SEARCH_ROOTS = [
    "../input/cassava-leaf-disease-classification",
    "/kaggle/input/cassava-leaf-disease-classification",
    "/kaggle/data/cassava-leaf-disease-classification",
]

priority_ext = (".pth", ".pt")

found = []
for root in SEARCH_ROOTS:
    if not os.path.exists(root):
        continue
    try:
        with os.scandir(root) as it:
            for e in it:
                if e.is_file():
                    if e.name.endswith(priority_ext):
                        found.append(e.path)
                elif e.is_dir():
                    try:
                        with os.scandir(e.path) as it2:
                            for e2 in it2:
                                if e2.is_file() and e2.name.endswith(priority_ext):
                                    found.append(e2.path)
                    except PermissionError:
                        pass
    except PermissionError:
        pass

seen = set()
for p in found:
    if p not in seen:
        seen.add(p)
        pretrained_models.append(p)

print(f"{len(pretrained_models)} model files found under cassava dataset roots.")
if len(pretrained_models) > 0:
    print("\n".join(np.sort(pretrained_models)[:50]))
    if len(pretrained_models) > 50:
        print(f"... (showing first 50 of {len(pretrained_models)})")



## === cell 2
import pandas as pd

import torch
import torch.nn as nn
import torch.utils.data as data

import torchvision
from torchvision import models  # 学習済みモデル、画像変換
import albumentations as A
from albumentations import Compose
from albumentations.pytorch import ToTensorV2

import random
import json
import time
import pickle
import sys

from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns

import cv2

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


SEED = 42
seed_everything(seed=SEED)

try:
    cv2.setNumThreads(0)
except Exception:
    pass
try:
    cv2.ocl.setUseOpenCL(False)
except Exception:
    pass



## === cell 3
try:
    sys.path.append("/kaggle/input/package/EfficientNet-PyTorch-1.0")
    from efficientnet_pytorch import EfficientNet  # type: ignore

    EFFICIENTNET_BACKEND = "efficientnet_pytorch"
except Exception:
    EFFICIENTNET_BACKEND = "torchvision"

    class _TorchvisionEfficientNetCompat(nn.Module):
        def __init__(self, model: nn.Module):
            super().__init__()
            self.model = model
            self._fc = self.model.classifier

        def forward(self, x):
            return self.model(x)

    class EfficientNet:
        @staticmethod
        def from_name(name: str):
            if name != "efficientnet-b7":
                raise ValueError(
                    f"Only efficientnet-b7 is supported by fallback, got: {name}"
                )
            m = torchvision.models.efficientnet_b7(weights=None)
            return _TorchvisionEfficientNetCompat(m)


print(f"EfficientNet backend: {EFFICIENTNET_BACKEND}")



## === cell 4
SIZE = 512  # image size
num_classes = 5



## === cell 5
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用デバイス: {device}")



## === cell 6
BASE_CANDIDATES = [
    "../input/cassava-leaf-disease-classification",
    "/kaggle/input/cassava-leaf-disease-classification",
    "../input/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
    "/kaggle/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
]

BASE_DIR = None
for p in BASE_CANDIDATES:
    if os.path.exists(p):
        BASE_DIR = p
        break
if BASE_DIR is None:
    BASE_DIR = "/kaggle/data/cassava-leaf-disease-classification"

TEST_PATH = f"{BASE_DIR}/test_images"
if not os.path.isdir(TEST_PATH):
    raise FileNotFoundError(f"Could not find test_images under BASE_DIR={BASE_DIR}")

test_files = sorted(os.listdir(TEST_PATH))

if os.getenv("KAGGLE_KERNEL_RUN_TYPE") == "Interactive":
    test_files = test_files[:320]

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
                A.Rotate(limit=30, p=1.0),
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
    ]
}



## === cell 10
TRAIN_CSV = f"{BASE_DIR}/train.csv"
TRAIN_IMG_DIR = f"{BASE_DIR}/train_images"
if not os.path.isdir(TRAIN_IMG_DIR):
    raise FileNotFoundError(f"train_images not found under BASE_DIR={BASE_DIR}")

df_train = pd.read_csv(TRAIN_CSV)
print("Loaded train.csv:", df_train.shape)




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

        index = torch.randperm(len(labels), device=labels.device)

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

        index = torch.randperm(len(labels), device=labels.device)

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
        img = cv2.imread(f"{TEST_PATH}/{image_id}", cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(
                f"Image not found or unreadable: {TEST_PATH}/{image_id}"
            )
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
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_id = self.df.loc[idx, "image_id"]
        label = int(self.df.loc[idx, "label"])
        path = f"{self.img_dir}/{image_id}"
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Image not found or unreadable: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(image=img)["image"]
        return img, torch.tensor(label, dtype=torch.long)


train_transform = Compose(
    [
        A.RandomResizedCrop(
            size=(SIZE, SIZE),
            scale=(0.7, 1.0),
            ratio=(0.75, 1.3333333333),
            p=1.0,
        ),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=20, p=0.3),
        A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
        ToTensorV2(p=1.0),
    ],
    p=1.0,
)




## === cell 17
def predict_model(basename, net, dl):
    """
    basename: 学習済みモデル名
    net     : 学習済みモデル
    dl      : dataloader for test
    """
    model_start_time = time.time()

    net.to(device)
    net.eval()

    n = len(dl.dataset)
    all_proba = np.empty((n, num_classes), dtype=np.float32)

    use_cuda = device == "cuda"
    offset = 0
    with torch.inference_mode():
        for inputs, image_ids in dl:
            inputs = inputs.to(device, non_blocking=use_cuda)
            if use_cuda:
                inputs = inputs.to(memory_format=torch.channels_last)
            outputs = net(inputs, False, "test")
            probs = torch.softmax(outputs, dim=1)
            probs = probs.to(dtype=torch.float32).cpu().numpy()
            bs = probs.shape[0]
            all_proba[offset : offset + bs] = probs
            offset += bs

    print(f"{basename} time: {time.time() - model_start_time:.2f}[sec]")
    return all_proba




## === cell 18
probability = []
start_time = time.time()

df_sub = pd.read_csv(f"{BASE_DIR}/sample_submission.csv")
df_test = df_sub[["image_id"]].copy()
df_test["label"] = 1

_center_crop_tensor = Compose(
    [
        A.CenterCrop(height=SIZE, width=SIZE),
        A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
        ToTensorV2(p=1.0),
    ],
    p=1.0,
)


class CachedTestDataset(data.Dataset):
    def __init__(
        self,
        df,
        transform=None,
        cache_images=True,
        cache_center_tensor=True,
    ):
        super().__init__()
        self.image_ids = df.image_id.tolist()
        self.transform = transform
        self.cache_images = cache_images
        self.cache_center_tensor = cache_center_tensor

        self._cache_rgb = None
        self._cache_center_tensor = None

        if self.cache_images:
            cache_list = [None] * len(self.image_ids)
            for i, image_id in enumerate(self.image_ids):
                path = f"{TEST_PATH}/{image_id}"
                img = cv2.imread(path, cv2.IMREAD_COLOR)
                if img is None:
                    raise FileNotFoundError(f"Image not found or unreadable: {path}")
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                cache_list[i] = img
            self._cache_rgb = cache_list

        if self.cache_center_tensor:
            if self._cache_rgb is None:
                raise ValueError("cache_center_tensor=True requires cache_images=True")
            tcache = [None] * len(self.image_ids)
            for i, img in enumerate(self._cache_rgb):
                tcache[i] = _center_crop_tensor(image=img)["image"]
            self._cache_center_tensor = tcache

    def set_transform(self, transform):
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        image_id = self.image_ids[index]

        if (
            self._cache_center_tensor is not None
            and self.transform is transform["test"][0]
        ):
            img_t = self._cache_center_tensor[index]
            return img_t, image_id

        if self._cache_rgb is not None:
            img = self._cache_rgb[index]
        else:
            img = cv2.imread(f"{TEST_PATH}/{image_id}", cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(
                    f"Image not found or unreadable: {TEST_PATH}/{image_id}"
                )
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            img = self.transform(image=img)["image"]
        return img, image_id


def _seed_worker(worker_id: int):
    seed = SEED + worker_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_test_loader(ds, batch_size):
    cpu = os.cpu_count() or 2
    nw = min(4, max(2, cpu - 1))
    g = torch.Generator()
    g.manual_seed(SEED)
    return torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=nw,
        pin_memory=True if device == "cuda" else False,
        persistent_workers=True if nw > 0 else False,
        prefetch_factor=2 if nw > 0 else None,
        worker_init_fn=_seed_worker if nw > 0 else None,
        generator=g,
    )


def maybe_compile_for_inference(net: nn.Module) -> nn.Module:
    if device != "cuda":
        return net
    if not hasattr(torch, "compile"):
        return net
    try:
        return torch.compile(net, mode="reduce-overhead", fullgraph=False)
    except Exception as e:
        print(f"[WARN] torch.compile failed, running eager. Reason: {e}")
        return net


if device == "cuda":
    torch.backends.cudnn.benchmark = True

if len(pretrained_models) == 0:
    print(
        "No cassava checkpoints (.pth/.pt) found under competition dataset roots; training a ResNet50 on provided train.csv to produce non-constant predictions."
    )

    criterion = nn.CrossEntropyLoss()
    base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    net = FinalLayerMixupModel(base, criterion, num_classes, False).to(device)

    train_ds = TrainDataset(df_train, TRAIN_IMG_DIR, transform=train_transform)
    nw_tr = min(4, max(2, (os.cpu_count() or 2) - 1))
    g_tr = torch.Generator()
    g_tr.manual_seed(SEED)
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=16,
        shuffle=True,
        num_workers=nw_tr,
        pin_memory=True if device == "cuda" else False,
        drop_last=False,
        persistent_workers=True if nw_tr > 0 else False,
        prefetch_factor=2 if nw_tr > 0 else None,
        worker_init_fn=_seed_worker if nw_tr > 0 else None,
        generator=g_tr,
    )

    optimizer = torch.optim.AdamW(net.parameters(), lr=2e-4, weight_decay=1e-4)

    net.train()
    torch.set_grad_enabled(True)

    EPOCHS = 2  # keep as-is (core approach preserved)
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        correct = 0
        total = 0
        for imgs, labels in train_loader:
            imgs = imgs.to(device, non_blocking=(device == "cuda"))
            if device == "cuda":
                imgs = imgs.to(memory_format=torch.channels_last)
            labels = labels.to(device, non_blocking=(device == "cuda"))

            optimizer.zero_grad(set_to_none=True)

            out = net(imgs, labels, "train")
            outputs, loss = out[0], out[1]

            try:
                loss.backward()
            except RuntimeError as e:
                msg = str(e)
                if (
                    "CUBLAS_WORKSPACE_CONFIG" in msg
                    or "not deterministic because it uses CuBLAS" in msg
                ):
                    print(
                        "[WARN] Deterministic CuBLAS backward failed; disabling deterministic algorithms for training."
                    )
                    try:
                        torch.use_deterministic_algorithms(False)
                    except Exception:
                        pass
                    torch.backends.cudnn.deterministic = False
                    loss.backward()
                else:
                    raise

            optimizer.step()

            epoch_loss += loss.item() * labels.size(0)
            total += labels.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
        print(
            f"train epoch {epoch+1}/{EPOCHS} loss={epoch_loss/max(total,1):.6f} acc={correct/max(total,1):.6f}"
        )

    pretrained_models = ["__trained_resnet50__"]
    trained_net = net  # keep reference for inference
else:
    trained_net = None


cached_test_ds = CachedTestDataset(
    df_test, transform=transform["test"][0], cache_images=True, cache_center_tensor=True
)

for pretrained_model in pretrained_models:
    if pretrained_model == "__trained_resnet50__":
        basename = "trained_resnet50"
        MODEL_NAME = "resnet50"
        net = trained_net
        BATCH_SIZE = 32
        print(f"{basename}: {MODEL_NAME} (trained in-notebook)")
        for param in net.parameters():
            param.requires_grad = False
    else:
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
            print(f"{basename} is not supported.")
            sys.exit()

        print(f"{basename}: {MODEL_NAME}")

        state = torch.load(pretrained_model, map_location="cpu")

        if MODEL_NAME == "efficientnet-b7":
            missing, unexpected = net.model.load_state_dict(state, strict=False)
            if len(missing) + len(unexpected) > 0:
                print(
                    f"[WARN] EfficientNet state_dict load: missing={len(missing)}, unexpected={len(unexpected)}"
                )
        else:
            missing, unexpected = net.load_state_dict(state, strict=False)
            if len(missing) + len(unexpected) > 0:
                print(
                    f"[WARN] {MODEL_NAME} state_dict load: missing={len(missing)}, unexpected={len(unexpected)}"
                )

        for param in net.parameters():
            param.requires_grad = False

    net = net.to(device)
    if device == "cuda":
        net = net.to(memory_format=torch.channels_last)
    net = maybe_compile_for_inference(net)

    test_loader = make_test_loader(cached_test_ds, batch_size=BATCH_SIZE)

    for tid, transform_ in enumerate(transform["test"]):
        print(f"transform loop={tid}")
        cached_test_ds.set_transform(transform_)
        proba = predict_model(basename, net, test_loader)
        probability.append(proba)

    if pretrained_model != "__trained_resnet50__":
        del net
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

proba_stack = np.stack(probability, axis=0)  # (K, N, C)
mean_proba = proba_stack.mean(axis=0)  # (N, C)
df_test["label"] = mean_proba.argmax(axis=1).astype(int)

print(f"total time: {time.time() - start_time:.2f}[sec]")



## === cell 19
if len(df_test) == 2 and df_test.loc[0, "image_id"] == df_test.loc[1, "image_id"]:
    df_test = pd.read_csv(f"{BASE_DIR}/sample_submission.csv")



## === cell 20
df_test.head()



## === cell 21
df_test = df_test.merge(df_sub[["image_id"]], on="image_id", how="right")
df_test["label"] = df_test["label"].fillna(df_sub["label"]).astype(int)

df_test[["image_id", "label"]].to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", df_test[["image_id", "label"]].shape)
print(df_test[["image_id", "label"]].head())
