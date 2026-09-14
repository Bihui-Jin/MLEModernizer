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

pretrained_models = (
    glob.glob(f"../input/densenet201-04-2019data/*.pth")
    + glob.glob(f"../input/eb7m-seed70/*.pth")
    + glob.glob(f"../input/eb7slseed70/efficientnet-b7sl_SEED70.best/*.pth")
)

print(f"{len(pretrained_models)} models found (initial globs).")
print("\n".join(np.sort(pretrained_models)))



## === cell 1
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
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


SEED = 42
seed_everything(seed=SEED)



## === cell 2
from torchvision.models import efficientnet_b7



## === cell 3
SIZE = 512  # image size
num_classes = 5



## === cell 4
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用デバイス: {device}")



## === cell 5
if os.path.isdir("../input/cassava-leaf-disease-classification"):
    print("In Kaggle environment.")
    BASE_DIR = "../input/cassava-leaf-disease-classification"
    TEST_PATH = f"{BASE_DIR}/test_images"
    TRAIN_PATH = f"{BASE_DIR}/train_images"
    test_files = sorted(os.listdir(TEST_PATH))
elif os.path.isdir("/kaggle/input/cassava-leaf-disease-classification"):
    print("In Kaggle environment (absolute path).")
    BASE_DIR = "/kaggle/input/cassava-leaf-disease-classification"
    TEST_PATH = f"{BASE_DIR}/test_images"
    TRAIN_PATH = f"{BASE_DIR}/train_images"
    test_files = sorted(os.listdir(TEST_PATH))
else:
    print("In the local environment.")
    BASE_DIR = "data/cassava-leaf-disease-classification"
    TEST_PATH = f"{BASE_DIR}/test_images"
    TRAIN_PATH = f"{BASE_DIR}/train_images"
    test_files = sorted(os.listdir(TEST_PATH))[:32]

print(f"Number of test  images: {len(test_files)}")


def _discover_pth_checkpoints(base_dir: str):
    candidates = []
    roots = []
    roots.append(Path(base_dir).resolve())
    roots.append(Path(base_dir).resolve().parent)  # e.g., /kaggle/input
    roots.append(Path("../input").resolve())
    roots.append(Path("/kaggle/input").resolve())
    roots.append(Path("data").resolve())
    roots.append(Path("input").resolve())

    seen = set()
    for r in roots:
        try:
            if r.exists() and r.is_dir():
                for p in r.rglob("*.pth"):
                    ps = str(p)
                    if ps not in seen:
                        candidates.append(ps)
                        seen.add(ps)
        except Exception:
            continue
    return candidates


def _is_supported_checkpoint(path: str) -> bool:
    b = os.path.splitext(os.path.basename(path))[0].lower()
    supported_tokens = [
        "resnet18",
        "resnet50",
        "resnet152",
        "resnext101",
        "densenet201",
        "efficientnet-b7",
        "cassava_local_train",  # added: our locally trained checkpoint naming
    ]
    return any(t in b for t in supported_tokens)


if len(pretrained_models) < 1:
    print("No initial checkpoints found via globs; skipping slow filesystem scan.")
    pretrained_models = []
else:
    pretrained_models = [p for p in pretrained_models if _is_supported_checkpoint(p)]
    print("Using initial pretrained_models list (no scan needed).")

pretrained_models = sorted(pretrained_models)
print(f"Final checkpoint count: {len(pretrained_models)}")
print("\n".join(pretrained_models[:50]))



## === cell 6
df_test = pd.DataFrame(test_files, columns=["image_id"])
df_test["label"] = 1



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
                A.CenterCrop(SIZE, SIZE),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                A.HorizontalFlip(p=1),
                A.CenterCrop(SIZE, SIZE),
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




## === cell 11
class FinalLayerMixupModelEN(nn.Module):
    def __init__(self, model, criterion, num_classes, alpha):
        super(FinalLayerMixupModelEN, self).__init__()

        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)

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
def _clean_state_dict_for_load(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict
    if "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
        state_dict = state_dict["state_dict"]

    cleaned = {}
    for k, v in state_dict.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module.") :]
        if nk.startswith("model."):
            nk = nk[len("model.") :]
        cleaned[nk] = v
    return cleaned




## === cell 13
from collections import OrderedDict


class _TensorLRUCache:
    def __init__(self, max_items: int):
        self.max_items = int(max_items)
        self._d = OrderedDict()

    def get(self, key):
        v = self._d.get(key, None)
        if v is not None:
            self._d.move_to_end(key, last=True)
        return v

    def set(self, key, value):
        self._d[key] = value
        self._d.move_to_end(key, last=True)
        if len(self._d) > self.max_items:
            self._d.popitem(last=False)

    def __len__(self):
        return len(self._d)


class TestDataset(data.Dataset):
    def __init__(
        self,
        df,
        transform=None,
        image_cache=None,
        tensor_cache=None,
        transform_id: int = 0,
    ):
        super().__init__()
        self.image_ids = df.image_id.tolist()
        self.transform = transform
        self.image_cache = image_cache  # dict image_id -> RGB np.ndarray
        self.tensor_cache = tensor_cache  # supports get/set
        self.transform_id = int(transform_id)

    def __len__(self):
        return len(self.image_ids)

    def load_image(self, image_id):
        if self.image_cache is not None:
            img = self.image_cache.get(image_id, None)
            if img is not None:
                return img
        img = cv2.imread(f"{TEST_PATH}/{image_id}")
        if img is None:
            raise FileNotFoundError(f"Image not found: {TEST_PATH}/{image_id}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.image_cache is not None:
            self.image_cache[image_id] = img
        return img

    def __getitem__(self, index):
        image_id = self.image_ids[index]

        if self.tensor_cache is not None:
            key = (self.transform_id, image_id)
            t = self.tensor_cache.get(key)
            if t is not None:
                return t, image_id

        img = self.load_image(image_id)
        if self.transform:
            t = self.transform(image=img)["image"]
        else:
            t = torch.from_numpy(img)

        if isinstance(t, torch.Tensor) and (not t.is_contiguous()):
            t = t.contiguous()

        if self.tensor_cache is not None:
            self.tensor_cache.set((self.transform_id, image_id), t)
        return t, image_id


class TrainDataset(data.Dataset):
    def __init__(self, df, transform=None):
        super().__init__()
        self.image_ids = df.image_id.tolist()
        self.labels = df.label.astype(int).tolist()
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def load_image(self, image_id):
        img = cv2.imread(f"{TRAIN_PATH}/{image_id}")
        if img is None:
            raise FileNotFoundError(f"Image not found: {TRAIN_PATH}/{image_id}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        y = self.labels[index]
        img = self.load_image(image_id)
        if self.transform:
            img = self.transform(image=img)["image"]
        return img, torch.tensor(y, dtype=torch.long)


train_transform = Compose(
    [
        A.RandomResizedCrop(
            size=(SIZE, SIZE),
            scale=(0.7, 1.0),
            ratio=(0.75, 1.3333333333),
            p=1.0,
        ),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
        ToTensorV2(p=1.0),
    ],
    p=1.0,
)

val_transform = Compose(
    [
        A.CenterCrop(SIZE, SIZE),
        A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
        ToTensorV2(p=1.0),
    ],
    p=1.0,
)




## === cell 14
def _stratified_split(df, seed=42, val_frac=0.1):
    rng = np.random.RandomState(seed)
    val_idx = []
    for c, g in df.groupby("label"):
        idx = g.index.to_numpy()
        rng.shuffle(idx)
        n_val = max(1, int(len(idx) * val_frac))
        val_idx.append(idx[:n_val])
    val_idx = np.concatenate(val_idx)
    is_val = df.index.isin(val_idx)
    return df.loc[~is_val].reset_index(drop=True), df.loc[is_val].reset_index(drop=True)


def _train_local_efficientnet_b7(save_path: str):
    train_csv_path = f"{BASE_DIR}/train.csv"
    df_train_all = pd.read_csv(train_csv_path)

    df_tr, df_va = _stratified_split(df_train_all, seed=SEED, val_frac=0.1)

    criterion = nn.CrossEntropyLoss()
    net = efficientnet_b7(weights="IMAGENET1K_V1")
    net = FinalLayerMixupModelEN(net, criterion, num_classes, False)

    net.to(device)
    net.train()

    BATCH_SIZE = 16 if torch.cuda.is_available() else 4
    num_workers = 2

    train_ds = TrainDataset(df_tr, transform=train_transform)
    val_ds = TrainDataset(df_va, transform=val_transform)

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )

    optimizer = torch.optim.AdamW(net.parameters(), lr=2e-4, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)

    best_acc = -1.0
    start = time.time()
    EPOCHS = 2  # unchanged
    for epoch in range(EPOCHS):
        net.train()
        tr_loss = 0.0
        tr_n = 0
        for xb, yb in tqdm(train_loader, desc=f"train epoch {epoch+1}/{EPOCHS}"):
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits, loss = net(xb, yb, "val")
            loss.backward()
            optimizer.step()

            tr_loss += float(loss.item()) * xb.size(0)
            tr_n += xb.size(0)

        scheduler.step()

        net.eval()
        correct = 0
        total = 0
        va_loss = 0.0
        with torch.no_grad():
            for xb, yb in tqdm(val_loader, desc=f"val epoch {epoch+1}/{EPOCHS}"):
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                logits, loss = net(xb, yb, "val")
                va_loss += float(loss.item()) * xb.size(0)
                pred = logits.argmax(dim=1)
                correct += int((pred == yb).sum().item())
                total += int(yb.numel())

        acc = correct / max(1, total)
        print(
            f"epoch {epoch+1}/{EPOCHS} train_loss={tr_loss/max(1,tr_n):.4f} val_loss={va_loss/max(1,total):.4f} val_acc={acc:.4f}"
        )

        if acc > best_acc:
            best_acc = acc
            torch.save(net.state_dict(), save_path)

    print(
        f"Local training finished in {time.time()-start:.1f}s, best_val_acc={best_acc:.4f}, saved={save_path}"
    )
    del net
    torch.cuda.empty_cache()


local_ckpt = "cassava_local_train_efficientnet-b7.pth"
if len(pretrained_models) == 0:
    print(
        "No pretrained models found; fallback local training will likely exceed the 600s budget."
    )
    if not os.path.exists(local_ckpt):
        _train_local_efficientnet_b7(local_ckpt)
    pretrained_models = [local_ckpt]
    print("Using locally trained checkpoint:", pretrained_models)




## === cell 15
def predict_model(basename, net, dataloader, n_samples: int, num_classes: int):
    """
    basename: 学習済みモデル名
    net     : 学習済みモデル
    """
    model_start_time = time.time()

    net.to(device)
    net.eval()

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    if device == "cuda":
        try:
            net = net.to(memory_format=torch.channels_last)
        except Exception:
            pass

    proba_all = np.empty((n_samples, num_classes), dtype=np.float32)
    offset = 0

    with torch.inference_mode():
        progress = tqdm(dataloader, desc=f"{basename}: ")
        for inputs, image_ids in progress:
            bs = inputs.size(0)
            if device == "cuda":
                inputs = inputs.contiguous(memory_format=torch.channels_last)
            inputs = inputs.to(device, non_blocking=True)
            outputs = net(inputs, False, "test")
            proba = (
                torch.softmax(outputs, dim=1)
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32, copy=False)
            )
            proba_all[offset : offset + bs] = proba
            offset += bs

    if offset != n_samples:
        proba_all = proba_all[:offset]

    print(f"{basename} time: {time.time() - model_start_time:.2f}[sec]")
    return proba_all




## === cell 16
probability = []

start_time = time.time()

sample_sub_path = f"{BASE_DIR}/sample_submission.csv"
df_sub = pd.read_csv(sample_sub_path)

df_test = df_sub[["image_id"]].copy()
df_test["label"] = 1  # placeholder, overwritten if we predict


def _seed_worker(worker_id: int):
    worker_seed = (SEED + worker_id) % (2**32 - 1)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


dl_generator = torch.Generator()
dl_generator.manual_seed(SEED)

if len(pretrained_models) == 0:
    print(
        "WARNING: No pretrained .pth files found. Submitting sample_submission labels (will score poorly)."
    )
else:
    pin = torch.cuda.is_available()
    image_ids = df_test["image_id"].tolist()

    def _build_tta_tensor_cache(df, transforms_list):
        test_image_cache = {}  # keep RGB np arrays, avoids repeated cv2 decode
        cache = {}  # (tid, image_id) -> torch.Tensor(C,H,W)

        for image_id in tqdm(df["image_id"].tolist(), desc="Precomputing TTA tensors"):
            img = cv2.imread(f"{TEST_PATH}/{image_id}")
            if img is None:
                raise FileNotFoundError(f"Image not found: {TEST_PATH}/{image_id}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            test_image_cache[image_id] = img

            for tid, tfm in enumerate(transforms_list):
                s = (
                    SEED * 1000003 + tid * 9176 + (hash(image_id) & 0xFFFFFFFF)
                ) & 0xFFFFFFFF
                random.seed(s)
                np.random.seed(s)

                t = tfm(image=img)["image"]
                if isinstance(t, torch.Tensor) and (not t.is_contiguous()):
                    t = t.contiguous()
                cache[(tid, image_id)] = t
        return cache

    class PrecomputedTTADataset(data.Dataset):
        def __init__(self, image_ids, precomputed_cache, transform_id: int):
            self.image_ids = list(image_ids)
            self.cache = precomputed_cache
            self.transform_id = int(transform_id)

        def __len__(self):
            return len(self.image_ids)

        def __getitem__(self, idx):
            image_id = self.image_ids[idx]
            return self.cache[(self.transform_id, image_id)], image_id

    precomputed_cache = _build_tta_tensor_cache(df_test, transform["test"])

    def _make_precomputed_loader(tid: int, batch_size: int):
        num_workers = 2 if torch.cuda.is_available() else 0
        ds = PrecomputedTTADataset(image_ids, precomputed_cache, transform_id=tid)
        return torch.utils.data.DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin,
            persistent_workers=(num_workers > 0),
            prefetch_factor=4 if num_workers > 0 else None,
            worker_init_fn=_seed_worker if num_workers > 0 else None,
            generator=dl_generator if num_workers > 0 else None,
        )

    loaders_by_tid_bs = {}

    for pretrained_model in pretrained_models:
        basename = os.path.splitext(os.path.basename(pretrained_model))[0]
        criterion = nn.CrossEntropyLoss()

        if "resnet18" in basename.lower():
            MODEL_NAME = "resnet18"
            net = models.resnet18(pretrained=False)
            net = FinalLayerMixupModel(net, criterion, num_classes, False)
            BATCH_SIZE = 64
        elif "resnet50" in basename.lower():
            MODEL_NAME = "resnet50"
            net = models.resnet50(pretrained=False)
            net = FinalLayerMixupModel(net, criterion, num_classes, False)
            BATCH_SIZE = 32
        elif "resnet152" in basename.lower():
            MODEL_NAME = "resnet152"
            net = models.resnet152(pretrained=False)
            net = FinalLayerMixupModel(net, criterion, num_classes, False)
            BATCH_SIZE = 16
        elif "resnext101" in basename.lower():
            MODEL_NAME = "resnext101"
            net = models.resnext101_32x8d(pretrained=False)
            net = FinalLayerMixupModel(net, criterion, num_classes, False)
            BATCH_SIZE = 12
        elif "densenet201" in basename.lower():
            MODEL_NAME = "densenet201"
            net = models.densenet201(pretrained=False)
            net = FinalLayerMixupModelDenseNet(net, criterion, num_classes, False)
            BATCH_SIZE = 12
        elif "efficientnet-b7" in basename.lower():
            MODEL_NAME = "efficientnet-b7"
            net = efficientnet_b7(weights=None)
            net = FinalLayerMixupModelEN(net, criterion, num_classes, False)
            BATCH_SIZE = 10
        else:
            print(f"{basename} is not supported; skipping this checkpoint.")
            continue

        print(f"{basename}: {MODEL_NAME} ({pretrained_model})")

        state = torch.load(pretrained_model, map_location="cpu")
        state = _clean_state_dict_for_load(state)

        loaded = False
        try:
            net.load_state_dict(state, strict=True)
            loaded = True
        except Exception:
            pass

        if not loaded:
            try:
                if MODEL_NAME == "efficientnet-b7":
                    net.model.load_state_dict(state, strict=True)
                    loaded = True
            except Exception:
                loaded = False

        if not loaded:
            print(f"Failed to load weights for {basename}; skipping.")
            del net
            torch.cuda.empty_cache()
            continue

        for param in net.parameters():
            param.requires_grad = False

        net.to(device)

        if hasattr(torch, "compile"):
            try:
                net = torch.compile(net, mode="reduce-overhead", fullgraph=False)
            except Exception:
                pass

        for tid, _ in enumerate(transform["test"]):
            key = (tid, BATCH_SIZE)
            if key not in loaders_by_tid_bs:
                loaders_by_tid_bs[key] = _make_precomputed_loader(tid, BATCH_SIZE)

            print(f"transform loop={tid}")
            dl = loaders_by_tid_bs[key]
            proba = predict_model(
                basename,
                net,
                dl,
                n_samples=len(image_ids),
                num_classes=num_classes,
            )
            probability.append(proba)

        del net
        torch.cuda.empty_cache()

    if len(probability) == 0:
        print(
            "WARNING: No models successfully loaded; falling back to sample_submission labels."
        )
        df_test = df_sub.copy()
    else:
        prob_array = np.asarray(probability)
        if prob_array.ndim != 3 or prob_array.shape[-1] != num_classes:
            raise ValueError(f"Unexpected probability array shape: {prob_array.shape}")

        df_test["mean"] = prob_array.mean(axis=0).argmax(axis=1)
        print(f"total time: {time.time() - start_time:.2f}[sec]")



## === cell 17
df_out = df_sub[["image_id"]].copy()
if "mean" in df_test.columns:
    df_out["label"] = df_test["mean"].astype(int).values
else:
    df_out["label"] = df_sub["label"].astype(int).values

df_out



## === cell 18
df_out[["image_id", "label"]].to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", df_out[["image_id", "label"]].shape)
print(df_out[["image_id", "label"]].head())
