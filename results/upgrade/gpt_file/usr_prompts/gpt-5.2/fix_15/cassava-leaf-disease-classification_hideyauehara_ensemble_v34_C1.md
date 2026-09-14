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
    torch.cuda.manual_seed_all(seed)
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
from collections import OrderedDict


def _get_worker_cache(max_items=256):
    info = torch.utils.data.get_worker_info()
    if info is None:
        if not hasattr(_get_worker_cache, "_cache_main"):
            _get_worker_cache._cache_main = OrderedDict()
        return _get_worker_cache._cache_main

    if not hasattr(_get_worker_cache, "_cache_workers"):
        _get_worker_cache._cache_workers = {}

    wid = info.id
    cache = _get_worker_cache._cache_workers.get(wid)
    if cache is None:
        cache = OrderedDict()
        _get_worker_cache._cache_workers[wid] = cache
    return cache


def _cache_get_put(cache, key, loader_fn, max_items=256):
    if key in cache:
        v = cache.pop(key)
        cache[key] = v
        return v
    v = loader_fn()
    cache[key] = v
    if len(cache) > max_items:
        cache.popitem(last=False)
    return v




## === cell 14
class TestDataset(data.Dataset):
    def __init__(self, df, transform=None):
        super().__init__()
        self.image_ids = df.image_id.tolist()
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def load_image(self, image_id):
        cache = _get_worker_cache(max_items=256)

        def _load():
            img = cv2.imread(f"{TEST_PATH}/{image_id}")
            if img is None:
                raise FileNotFoundError(
                    f"Image not found/readable: {TEST_PATH}/{image_id}"
                )
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img

        return _cache_get_put(cache, ("test", image_id), _load, max_items=256)

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        img = self.load_image(image_id)

        if self.transform:
            img = self.transform(image=img)["image"]

        return img, image_id




## === cell 15
class TrainDataset(data.Dataset):
    def __init__(self, df, img_dir, transform=None):
        super().__init__()
        self.image_ids = df.image_id.tolist()
        self.labels = df.label.astype(int).tolist()
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def _load_rgb(self, image_id):
        cache = _get_worker_cache(max_items=256)

        def _load():
            img = cv2.imread(f"{self.img_dir}/{image_id}")
            if img is None:
                raise FileNotFoundError(
                    f"Image not found/readable: {self.img_dir}/{image_id}"
                )
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img

        return _cache_get_put(cache, (self.img_dir, image_id), _load, max_items=256)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        label = self.labels[idx]
        img = self._load_rgb(image_id)
        if self.transform:
            img = self.transform(image=img)["image"]
        return img, label




## === cell 16
def predict_model(basename, net, dataloader, use_wrapper=True):
    """
    basename: 学習済みモデル名
    net     : 学習済みモデル
    use_wrapper: True if net.forward expects (inputs, labels, phase); False if standard torchvision forward(inputs)
    """
    model_start_time = time.time()

    net.to(device)
    net.eval()

    dl = dataloader["test"]
    n = len(dl.dataset)
    proba = np.empty((n, num_classes), dtype=np.float32)
    offset = 0

    progress = tqdm(dl, desc=f"{basename}: ")
    with torch.inference_mode():
        for inputs, _image_ids in progress:
            bs = inputs.size(0)
            if device == "cuda":
                inputs = inputs.to(device, non_blocking=True).contiguous(
                    memory_format=torch.channels_last
                )
            else:
                inputs = inputs.to(device, non_blocking=True)
            if use_wrapper:
                outputs = net(inputs, False, "test")
            else:
                outputs = net(inputs)
            p = torch.softmax(outputs, dim=1).detach().cpu().numpy().astype(np.float32)
            proba[offset : offset + bs] = p
            offset += bs

    print(f"{basename} time: {time.time() - model_start_time:.2f}[sec]")
    return proba




## === cell 17
def efficientnet_b7_features(net, x):
    x = net.features(x)
    x = net.avgpool(x)
    x = torch.flatten(x, 1)
    return x


def fit_ridge_multiclass_streaming(
    feature_fn,
    net,
    dataloader,
    num_classes=5,
    reg=20.0,
    desc="fit ridge",
    compute_mean_std=False,
    add_bias=True,
):
    net.to(device)
    net.eval()

    XtX = None  # (d,d)
    XtY = None  # (d,C)
    d = None
    n_samples = 0

    n = 0
    mean_ = None
    M2 = None

    progress = tqdm(dataloader, desc=desc)
    with torch.inference_mode():
        for x, y in progress:
            if device == "cuda":
                x = x.to(device, non_blocking=True).contiguous(
                    memory_format=torch.channels_last
                )
            else:
                x = x.to(device, non_blocking=True)

            feats = (
                feature_fn(net, x).detach().cpu().numpy().astype(np.float64)
            )  # (B,d0)

            if add_bias:
                feats = np.concatenate(
                    [feats, np.ones((feats.shape[0], 1), dtype=feats.dtype)], axis=1
                )  # (B,d0+1)

            yy = y.numpy().astype(int)
            if d is None:
                d = feats.shape[1]
                XtX = np.zeros((d, d), dtype=np.float64)
                XtY = np.zeros((d, num_classes), dtype=np.float64)
                if compute_mean_std:
                    mean_ = np.zeros((d,), dtype=np.float64)
                    M2 = np.zeros((d,), dtype=np.float64)

            if compute_mean_std:
                b = feats.shape[0]
                n_new = n + b
                batch_mean = feats.mean(axis=0)
                batch_M2 = ((feats - batch_mean) ** 2).sum(axis=0)
                delta = batch_mean - mean_
                mean_ = mean_ + delta * (b / max(n_new, 1))
                M2 = M2 + batch_M2 + (delta * delta) * (n * b / max(n_new, 1))
                n = n_new

            Y = np.zeros((feats.shape[0], num_classes), dtype=np.float64)
            Y[np.arange(feats.shape[0]), yy] = 1.0

            XtX += feats.T @ feats
            XtY += feats.T @ Y
            n_samples += feats.shape[0]

    if n_samples > 0:
        XtX /= float(n_samples)
        XtY /= float(n_samples)

    XtX_reg = XtX + reg * np.eye(d, dtype=np.float64)
    W = np.linalg.solve(XtX_reg, XtY)  # (d,C)

    if compute_mean_std:
        var_ = M2 / max(n - 1, 1)
        std_ = np.sqrt(np.maximum(var_, 1e-12))
        return W.astype(np.float32), mean_.astype(np.float32), std_.astype(np.float32)

    return W.astype(np.float32), None, None


def softmax_np(z):
    z = z - z.max(axis=1, keepdims=True)
    ez = np.exp(z)
    return ez / (ez.sum(axis=1, keepdims=True) + 1e-12)


def apply_feature_standardize(feats, x_mean, x_std):
    return (feats - x_mean.reshape(1, -1)) / x_std.reshape(1, -1)


def make_holdout_split(df, holdout_frac=0.05, seed=42):
    rs = np.random.RandomState(seed)
    idx = np.arange(len(df))
    rs.shuffle(idx)
    n_hold = max(1, int(len(df) * holdout_frac))
    hold_idx = idx[:n_hold]
    tr_idx = idx[n_hold:]
    return df.iloc[tr_idx].reset_index(drop=True), df.iloc[hold_idx].reset_index(
        drop=True
    )




## === cell 18
def extract_features_dataset(
    feature_fn,
    net,
    dataset,
    batch_size,
    num_workers,
    prefetch_factor,
    persistent_workers,
    desc="extract feats",
):
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(persistent_workers and num_workers > 0),
        prefetch_factor=(prefetch_factor if (num_workers > 0) else None),
    )

    net.to(device)
    net.eval()

    feats_list = []
    y_list = []

    with torch.inference_mode():
        for xb, yb in tqdm(loader, desc=desc):
            if device == "cuda":
                xb = xb.to(device, non_blocking=True).contiguous(
                    memory_format=torch.channels_last
                )
            else:
                xb = xb.to(device, non_blocking=True)
            fb = feature_fn(net, xb).detach().cpu().numpy().astype(np.float32)
            feats_list.append(fb)
            y_list.append(np.asarray(yb.numpy(), dtype=np.int64))
    X = np.concatenate(feats_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    return X, y


def ridge_fit_from_features(
    X, y, num_classes, reg, add_bias=True, compute_mean_std=False
):
    X64 = X.astype(np.float64, copy=False)
    if add_bias:
        ones = np.ones((X64.shape[0], 1), dtype=np.float64)
        X64b = np.concatenate([X64, ones], axis=1)
    else:
        X64b = X64

    if compute_mean_std:
        mean_ = X64b.mean(axis=0)
        std_ = X64b.std(axis=0, ddof=1)
        std_ = np.maximum(std_, 1e-12)
    else:
        mean_ = None
        std_ = None

    if (mean_ is not None) and (std_ is not None):
        Xn = (X64b - mean_[None, :]) / std_[None, :]
    else:
        Xn = X64b

    Y = np.zeros((Xn.shape[0], num_classes), dtype=np.float64)
    Y[np.arange(Xn.shape[0]), y] = 1.0

    n = float(Xn.shape[0])
    XtX = (Xn.T @ Xn) / n
    XtY = (Xn.T @ Y) / n
    XtX_reg = XtX + reg * np.eye(XtX.shape[0], dtype=np.float64)
    W = np.linalg.solve(XtX_reg, XtY).astype(np.float32)

    if compute_mean_std:
        return W, mean_.astype(np.float32), std_.astype(np.float32)
    return W, None, None


def ridge_predict_from_features(X, W, x_mean=None, x_std=None, add_bias=True):
    X64 = X.astype(np.float64, copy=False)
    if add_bias:
        X64 = np.concatenate(
            [X64, np.ones((X64.shape[0], 1), dtype=np.float64)], axis=1
        )
    if x_mean is not None and x_std is not None:
        X64 = (X64 - x_mean.astype(np.float64)[None, :]) / x_std.astype(np.float64)[
            None, :
        ]
    logits = X64 @ W.astype(np.float64)
    return logits




## === cell 19
probability = []
start_time = time.time()

CPU_COUNT = os.cpu_count() or 4
DL_NUM_WORKERS = min(8, CPU_COUNT)
DL_PREFETCH = 4 if DL_NUM_WORKERS > 0 else None
DL_PERSIST = True

if device == "cuda":
    torch.backends.cudnn.benchmark = True

if device == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

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

    if hasattr(torch, "compile"):
        try:
            net = torch.compile(net, mode="reduce-overhead", fullgraph=False)
            print("torch.compile enabled for EfficientNet-B7.")
        except Exception as e:
            print(
                "torch.compile not available/failed; continuing without compile. Error:",
                repr(e),
            )

    W = None
    x_mean = None
    x_std = None
    ADD_BIAS = True  # keep consistent between fit and inference

    if df_train is not None and os.path.isdir(TRAIN_IMG_PATH):
        train_tfs = [
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

        train_bs = 32 if device == "cuda" else 16

        df_tr_fit, df_tr_hold = make_holdout_split(
            df_train, holdout_frac=0.05, seed=SEED
        )
        print("Ridge fit/holdout sizes:", len(df_tr_fit), len(df_tr_hold))

        reg_candidates = [2.0, 5.0, 10.0, 20.0, 40.0]

        fit_ds0 = TrainDataset(
            df_tr_fit, img_dir=TRAIN_IMG_PATH, transform=train_tfs[0]
        )
        hold_ds0 = TrainDataset(
            df_tr_hold, img_dir=TRAIN_IMG_PATH, transform=train_tfs[0]
        )

        X_fit0, y_fit0 = extract_features_dataset(
            efficientnet_b7_features,
            net,
            fit_ds0,
            batch_size=train_bs,
            num_workers=DL_NUM_WORKERS,
            prefetch_factor=DL_PREFETCH,
            persistent_workers=DL_PERSIST,
            desc="Extract train-fit features (view 1) for reg-search",
        )
        X_hold0, y_hold0 = extract_features_dataset(
            efficientnet_b7_features,
            net,
            hold_ds0,
            batch_size=train_bs,
            num_workers=DL_NUM_WORKERS,
            prefetch_factor=DL_PREFETCH,
            persistent_workers=DL_PERSIST,
            desc="Extract train-holdout features (view 1) for reg-search",
        )
        del fit_ds0, hold_ds0
        torch.cuda.empty_cache()

        best = None
        for reg_value in reg_candidates:
            W_tmp, m_tmp, s_tmp = ridge_fit_from_features(
                X_fit0,
                y_fit0,
                num_classes=num_classes,
                reg=reg_value,
                add_bias=ADD_BIAS,
                compute_mean_std=True,
            )
            logits = ridge_predict_from_features(
                X_hold0, W_tmp, m_tmp, s_tmp, add_bias=ADD_BIAS
            )
            pred = logits.argmax(axis=1).astype(np.int64)
            acc = float((pred == y_hold0).mean())
            print(f"reg={reg_value} holdout_acc={acc:.5f}")
            if (best is None) or (acc > best["acc"]):
                best = {
                    "acc": acc,
                    "reg": reg_value,
                    "W": W_tmp,
                    "m": m_tmp,
                    "s": s_tmp,
                }

        chosen_reg = float(best["reg"])
        print("Chosen ridge reg:", chosen_reg)

        W_accum = None
        for vid, train_tf in enumerate(train_tfs):
            print(f"Extracting full train features for view {vid+1}/{len(train_tfs)}")
            full_ds = TrainDataset(
                df_train.reset_index(drop=True),
                img_dir=TRAIN_IMG_PATH,
                transform=train_tf,
            )
            X_full, y_full = extract_features_dataset(
                efficientnet_b7_features,
                net,
                full_ds,
                batch_size=train_bs,
                num_workers=DL_NUM_WORKERS,
                prefetch_factor=DL_PREFETCH,
                persistent_workers=DL_PERSIST,
                desc=f"Extract full-train features (view {vid+1})",
            )
            del full_ds
            torch.cuda.empty_cache()

            if vid == 0:
                W_view, x_mean, x_std = ridge_fit_from_features(
                    X_full,
                    y_full,
                    num_classes=num_classes,
                    reg=chosen_reg,
                    add_bias=ADD_BIAS,
                    compute_mean_std=True,
                )
                print("Computed feature mean/std shapes:", x_mean.shape, x_std.shape)
            else:
                W_view, _, _ = ridge_fit_from_features(
                    X_full,
                    y_full,
                    num_classes=num_classes,
                    reg=chosen_reg,
                    add_bias=ADD_BIAS,
                    compute_mean_std=False,
                )

            if W_accum is None:
                W_accum = W_view.astype(np.float32)
            else:
                W_accum += W_view.astype(np.float32)

        W = (W_accum / float(len(train_tfs))).astype(np.float32)
        print("Fitted linear mapping W with shape:", W.shape)
    else:
        print(
            "Could not fit linear mapping (missing train.csv or train_images); using prior-weighted heuristic."
        )

    BATCH_SIZE = 16 if device == "cuda" else 8
    basename = "efficientnet-b7-imagenet"
    test_ds = TestDataset(df_test, transform=None)

    for tid, transform_ in enumerate(transform["test"]):
        print(f"transform loop={tid}")
        test_ds.transform = transform_
        test_loader = torch.utils.data.DataLoader(
            test_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=DL_NUM_WORKERS,
            pin_memory=True,
            persistent_workers=(DL_PERSIST and DL_NUM_WORKERS > 0),
            prefetch_factor=DL_PREFETCH,
        )

        n_test = len(test_ds)
        proba5 = np.empty((n_test, num_classes), dtype=np.float32)
        offset = 0

        net.to(device)
        net.eval()
        with torch.inference_mode():
            for inputs, _image_ids in tqdm(
                test_loader, desc=f"{basename} features->proba5: "
            ):
                bs = inputs.size(0)
                if device == "cuda":
                    inputs = inputs.to(device, non_blocking=True).contiguous(
                        memory_format=torch.channels_last
                    )
                else:
                    inputs = inputs.to(device, non_blocking=True)

                if W is not None:
                    feats = (
                        efficientnet_b7_features(net, inputs)
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.float32)
                    )
                    logits5 = ridge_predict_from_features(
                        feats, W, x_mean, x_std, add_bias=ADD_BIAS
                    )
                    p5 = softmax_np(logits5.astype(np.float64)).astype(np.float32)
                else:
                    out = (
                        net(inputs).detach().cpu().numpy().astype(np.float32)
                    )  # (B,1000)
                    proba_1000 = softmax_np(out.astype(np.float64))
                    p5 = np.zeros((bs, num_classes), dtype=np.float64)
                    for c in range(num_classes):
                        p5[:, c] = proba_1000[:, c::num_classes].sum(axis=1)
                    p5 = p5 * prior.reshape(1, -1)
                    p5 = p5 / (p5.sum(axis=1, keepdims=True) + 1e-12)
                    p5 = p5.astype(np.float32)

                proba5[offset : offset + bs] = p5
                offset += bs

        probability.append(proba5)

    del net
    torch.cuda.empty_cache()

else:
    test_ds = TestDataset(df_test, transform=None)

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
            test_ds.transform = transform_
            test_loader = torch.utils.data.DataLoader(
                test_ds,
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=DL_NUM_WORKERS,
                pin_memory=True,
                persistent_workers=(DL_PERSIST and DL_NUM_WORKERS > 0),
                prefetch_factor=DL_PREFETCH,
            )
            dataloader = {"test": test_loader}
            proba = predict_model(basename, net, dataloader, use_wrapper=True)
            probability.append(proba)

        del net
        torch.cuda.empty_cache()

proba_mean = np.mean(np.stack(probability, axis=0), axis=0)
df_test["mean"] = proba_mean.argmax(axis=1)

print(f"total time: {time.time() - start_time:.2f}[sec]")



## === cell 20
if "mean" in df_test.columns:
    df_test["label"] = df_test["mean"]

df_test = df_test[["image_id", "label"]]



## === cell 21
df_test.head()



## === cell 22
df_test[["image_id", "label"]].to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", df_test.shape)
print(df_test.head())
