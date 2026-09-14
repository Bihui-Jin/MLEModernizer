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
    + glob.glob(f"../input/resnet152-04-2019data/*.pth")
    + glob.glob(f"../input/eb7slseed70/efficientnet-b7sl_SEED70.best/*.pth")
)

pretrained_models += glob.glob("/kaggle/input/**/*.pth", recursive=True)
pretrained_models = sorted(list(set(pretrained_models)))

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
from torchvision.models import efficientnet_b7


class _TorchvisionEfficientNetWrapper(nn.Module):
    """
    Mimic the minimal interface used later: EfficientNet.from_name + attribute _fc.
    We map _fc to the classifier's final Linear layer.
    """

    def __init__(self, num_classes=1000):
        super().__init__()
        self.model = efficientnet_b7(weights=None)
        self._fc = self.model.classifier[-1]
        if num_classes is not None and self._fc.out_features != num_classes:
            self.model.classifier[-1] = nn.Linear(self._fc.in_features, num_classes)
            self._fc = self.model.classifier[-1]

    def forward(self, x):
        return self.model(x)


class EfficientNet:
    @staticmethod
    def from_name(name: str):
        if name != "efficientnet-b7":
            raise ValueError(
                f"Only efficientnet-b7 is supported in this fallback, got: {name}"
            )
        return _TorchvisionEfficientNetWrapper(num_classes=1000)




## === cell 4
SIZE = 512

EFFNET_FALLBACK_SIZE = 300  # EfficientNet-B3 default resolution

num_classes = 5



## === cell 5
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用デバイス: {device}")




## === cell 6
def _find_base_dir():
    candidates = [
        "../input/cassava-leaf-disease-classification",
        "/kaggle/input/cassava-leaf-disease-classification",
        "../input/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
        "/kaggle/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    if os.path.isdir("../input"):
        hits = glob.glob("../input/**/train.csv", recursive=True)
        if len(hits) > 0:
            return str(Path(hits[0]).parent)
    raise FileNotFoundError(
        "Could not locate cassava-leaf-disease-classification dataset directory."
    )


BASE_DIR = _find_base_dir()

sample_sub_path = f"{BASE_DIR}/sample_submission.csv"
if not os.path.isfile(sample_sub_path):
    raise FileNotFoundError(f"sample_submission.csv not found at: {sample_sub_path}")

if os.path.isdir(f"{BASE_DIR}/test_images"):
    TEST_PATH = f"{BASE_DIR}/test_images"
else:
    TEST_PATH = f"{BASE_DIR}/train_images"

if os.path.isdir(f"{BASE_DIR}/train_images"):
    TRAIN_PATH = f"{BASE_DIR}/train_images"
else:
    train_hits = glob.glob(f"{BASE_DIR}/**/train_images", recursive=True)
    if len(train_hits) == 0:
        raise FileNotFoundError(
            "Could not locate train_images directory under BASE_DIR."
        )
    TRAIN_PATH = train_hits[0]

df_test = pd.read_csv(sample_sub_path)[["image_id"]].copy()
df_test["image_id"] = df_test["image_id"].astype(str)
test_files = df_test["image_id"].tolist()

print(f"BASE_DIR: {BASE_DIR}")
print(f"TRAIN_PATH: {TRAIN_PATH}")
print(f"TEST_PATH: {TEST_PATH}")
print(f"Number of test images (from sample_submission): {len(test_files)}")

missing = [
    img_id for img_id in test_files[:50] if not os.path.isfile(f"{TEST_PATH}/{img_id}")
]
if len(missing) > 0:
    raise FileNotFoundError(
        f"Some expected test images are missing under TEST_PATH={TEST_PATH}. "
        f"Example missing: {missing[0]}"
    )



## === cell 7
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
                A.Rotate(p=1),
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
        num_ftrs = model._fc.in_features
        model._fc = nn.Linear(num_ftrs, num_classes)
        if hasattr(model, "model") and hasattr(model.model, "classifier"):
            model.model.classifier[-1] = model._fc

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
            raise FileNotFoundError(f"Failed to read image: {TEST_PATH}/{image_id}")
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




## === cell 17
class _ImageNetEffNetCassavaHead(nn.Module):
    def __init__(self):
        super().__init__()
        m = torchvision.models.efficientnet_b3(
            weights=torchvision.models.EfficientNet_B3_Weights.IMAGENET1K_V1
        )
        in_f = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_f, num_classes)
        self.model = m

    def forward(self, x, labels, phase):
        return self.model(x)




## === cell 18
class TrainDataset(data.Dataset):
    def __init__(self, df, transform=None):
        super().__init__()
        self.image_ids = df.image_id.tolist()
        self.labels = df.label.astype(int).tolist()
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        y = self.labels[index]
        img = cv2.imread(f"{TRAIN_PATH}/{image_id}")
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {TRAIN_PATH}/{image_id}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(image=img)["image"]
        return img, torch.tensor(y, dtype=torch.long)


def _eval_effnet_head(net, val_loader, criterion):
    net.eval()
    torch.set_grad_enabled(False)
    running_loss = 0.0
    correct = 0
    total = 0
    for xb, yb in val_loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        logits = net.model(xb)  # logits
        loss = criterion(logits, yb)

        running_loss += float(loss.item()) * xb.size(0)
        pred = logits.argmax(dim=1)
        correct += int((pred == yb).sum().item())
        total += int(yb.size(0))
    return running_loss / max(1, total), correct / max(1, total)


def _train_effnet_two_stage_fixed_epochs_bestval(
    net,
    train_loader,
    val_loader,
    class_weights=None,
    stage1_lr=2e-3,
    stage1_epochs=10,
    stage2_lr=2e-4,
    stage2_epochs=4,
    label_smoothing=0.1,
    stage2_unfreeze_last_n_blocks=2,
    grad_clip_norm=1.0,
):
    net.to(device)

    if class_weights is not None:
        class_weights_t = torch.tensor(
            class_weights, dtype=torch.float32, device=device
        )
        criterion = nn.CrossEntropyLoss(
            weight=class_weights_t, label_smoothing=float(label_smoothing)
        )
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=float(label_smoothing))

    best_state = None
    best_val_acc = -1.0

    def _maybe_update_best():
        nonlocal best_state, best_val_acc
        val_loss, val_acc = _eval_effnet_head(net, val_loader, criterion)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {
                k: v.detach().cpu().clone() for k, v in net.state_dict().items()
            }
        return val_loss, val_acc

    t0 = time.time()

    for p in net.parameters():
        p.requires_grad = False
    for p in net.model.classifier.parameters():
        p.requires_grad = True

    optimizer = torch.optim.AdamW(
        net.model.classifier.parameters(), lr=stage1_lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=stage1_epochs, eta_min=stage1_lr * 0.05
    )

    for ep in range(1, stage1_epochs + 1):
        net.model.features.eval()
        net.model.avgpool.eval()
        net.model.classifier.train()

        torch.backends.cudnn.benchmark = True
        torch.set_grad_enabled(True)

        running_loss = 0.0
        correct = 0
        total = 0

        for xb, yb in tqdm(train_loader, desc=f"stage1_head ep{ep}/{stage1_epochs}: "):
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = net.model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            if grad_clip_norm is not None and grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    net.model.classifier.parameters(), max_norm=float(grad_clip_norm)
                )
            optimizer.step()

            running_loss += float(loss.item()) * xb.size(0)
            pred = logits.argmax(dim=1)
            correct += int((pred == yb).sum().item())
            total += int(yb.size(0))

        scheduler.step()
        torch.backends.cudnn.benchmark = False

        train_loss = running_loss / max(1, total)
        train_acc = correct / max(1, total)
        val_loss, val_acc = _maybe_update_best()
        cur_lr = optimizer.param_groups[0]["lr"]
        print(
            f"stage1 ep{ep}: lr={cur_lr:.6f} train_loss={train_loss:.4f}, train_acc={train_acc:.4f} | val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        )

    for p in net.parameters():
        p.requires_grad = False
    for p in net.model.classifier.parameters():
        p.requires_grad = True

    blocks = list(net.model.features)
    n = int(stage2_unfreeze_last_n_blocks)
    n = max(1, min(n, len(blocks)))
    unfrozen_blocks = blocks[-n:]
    for b in unfrozen_blocks:
        for p in b.parameters():
            p.requires_grad = True

    params = list(net.model.classifier.parameters())
    for b in unfrozen_blocks:
        params += list(b.parameters())

    optimizer2 = torch.optim.AdamW(params, lr=stage2_lr, weight_decay=1e-4)
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer2, T_max=stage2_epochs, eta_min=stage2_lr * 0.05
    )

    for ep in range(1, stage2_epochs + 1):
        for b in blocks[:-n]:
            b.eval()
        for b in unfrozen_blocks:
            b.train()
        net.model.avgpool.eval()
        net.model.classifier.train()

        torch.backends.cudnn.benchmark = True
        torch.set_grad_enabled(True)

        running_loss = 0.0
        correct = 0
        total = 0

        for xb, yb in tqdm(
            train_loader, desc=f"stage2_last{n}blocks ep{ep}/{stage2_epochs}: "
        ):
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer2.zero_grad(set_to_none=True)
            logits = net.model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            if grad_clip_norm is not None and grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(params, max_norm=float(grad_clip_norm))
            optimizer2.step()

            running_loss += float(loss.item()) * xb.size(0)
            pred = logits.argmax(dim=1)
            correct += int((pred == yb).sum().item())
            total += int(yb.size(0))

        scheduler2.step()
        torch.backends.cudnn.benchmark = False

        train_loss = running_loss / max(1, total)
        train_acc = correct / max(1, total)
        val_loss, val_acc = _maybe_update_best()
        cur_lr = optimizer2.param_groups[0]["lr"]
        print(
            f"stage2 ep{ep}: lr={cur_lr:.6f} train_loss={train_loss:.4f}, train_acc={train_acc:.4f} | val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        )

    if best_state is not None:
        net.load_state_dict(best_state, strict=True)

    net.eval()
    torch.set_grad_enabled(False)
    print(
        f"train_effnet_two_stage: best_val_acc={best_val_acc:.4f}, total_time={time.time()-t0:.1f}s"
    )
    return net




## === cell 19
probability = []
start_time = time.time()

if len(pretrained_models) == 0:
    print(
        "No pretrained .pth models found. Training a lightweight EfficientNet classifier head on cassava train.csv (backbone frozen), then running the same TTA inference."
    )

    train_csv_path = f"{BASE_DIR}/train.csv"
    if not os.path.isfile(train_csv_path):
        raise FileNotFoundError(f"train.csv not found at: {train_csv_path}")
    df_train_full = pd.read_csv(train_csv_path)[["image_id", "label"]].copy()
    df_train_full["image_id"] = df_train_full["image_id"].astype(str)
    df_train_full["label"] = df_train_full["label"].astype(int)

    counts = df_train_full["label"].value_counts().sort_index()
    freq = counts.values.astype(np.float32)
    inv = freq.sum() / np.maximum(freq, 1.0)
    class_weights = (inv / inv.mean()).tolist()

    rng = np.random.RandomState(SEED)
    val_frac = 0.1
    train_idx = []
    val_idx = []
    for c in range(num_classes):
        idx_c = np.where(df_train_full["label"].values == c)[0]
        rng.shuffle(idx_c)
        n_val = max(1, int(len(idx_c) * val_frac))
        val_idx.extend(idx_c[:n_val].tolist())
        train_idx.extend(idx_c[n_val:].tolist())

    df_train = df_train_full.iloc[train_idx].reset_index(drop=True)
    df_val = df_train_full.iloc[val_idx].reset_index(drop=True)
    print(f"Train split: {len(df_train)}, Val split: {len(df_val)}")

    train_transform = Compose(
        [
            A.SmallestMaxSize(
                max_size=EFFNET_FALLBACK_SIZE, interpolation=cv2.INTER_CUBIC, p=1.0
            ),
            A.CenterCrop(
                height=EFFNET_FALLBACK_SIZE, width=EFFNET_FALLBACK_SIZE, p=1.0
            ),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ],
        p=1.0,
    )
    val_transform = Compose(
        [
            A.SmallestMaxSize(
                max_size=EFFNET_FALLBACK_SIZE, interpolation=cv2.INTER_CUBIC, p=1.0
            ),
            A.CenterCrop(
                height=EFFNET_FALLBACK_SIZE, width=EFFNET_FALLBACK_SIZE, p=1.0
            ),
            A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ],
        p=1.0,
    )

    net = _ImageNetEffNetCassavaHead()

    BATCH_SIZE_TRAIN = 32
    train_loader = torch.utils.data.DataLoader(
        TrainDataset(df_train, transform=train_transform),
        batch_size=BATCH_SIZE_TRAIN,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = torch.utils.data.DataLoader(
        TrainDataset(df_val, transform=val_transform),
        batch_size=32,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=False,
    )

    net = _train_effnet_two_stage_fixed_epochs_bestval(
        net,
        train_loader,
        val_loader,
        class_weights=class_weights,
        stage1_lr=2e-3,
        stage1_epochs=10,
        stage2_lr=2e-4,
        stage2_epochs=4,
        label_smoothing=0.1,
        stage2_unfreeze_last_n_blocks=2,
        grad_clip_norm=1.0,
    )

    transform_fallback_test = [
        Compose(
            [
                A.SmallestMaxSize(
                    max_size=EFFNET_FALLBACK_SIZE, interpolation=cv2.INTER_CUBIC, p=1.0
                ),
                A.CenterCrop(
                    height=EFFNET_FALLBACK_SIZE, width=EFFNET_FALLBACK_SIZE, p=1.0
                ),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                A.SmallestMaxSize(
                    max_size=EFFNET_FALLBACK_SIZE, interpolation=cv2.INTER_CUBIC, p=1.0
                ),
                A.HorizontalFlip(p=1),
                A.CenterCrop(
                    height=EFFNET_FALLBACK_SIZE, width=EFFNET_FALLBACK_SIZE, p=1.0
                ),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                A.RandomResizedCrop(
                    size=(EFFNET_FALLBACK_SIZE, EFFNET_FALLBACK_SIZE),
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
                    size=(EFFNET_FALLBACK_SIZE, EFFNET_FALLBACK_SIZE),
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
                A.SmallestMaxSize(
                    max_size=EFFNET_FALLBACK_SIZE, interpolation=cv2.INTER_CUBIC, p=1.0
                ),
                A.Rotate(p=1),
                A.CenterCrop(
                    height=EFFNET_FALLBACK_SIZE, width=EFFNET_FALLBACK_SIZE, p=1.0
                ),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
    ]

    BATCH_SIZE = 32
    for tid, transform_ in enumerate(transform_fallback_test):
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
        proba = predict_model("trained_imagenet_effnet_fallback_head", net, dataloader)
        probability.append(proba)

    proba_mean = np.mean(np.stack(probability, axis=0), axis=0)
    df_test["mean"] = proba_mean.argmax(axis=1)
    df_test["label"] = df_test["mean"].astype(int)

    print(f"total time: {time.time() - start_time:.2f}[sec]")
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
            net = EfficientNet.from_name(MODEL_NAME)
            net = FinalLayerMixupModelEN(net, criterion, num_classes, False)
            BATCH_SIZE = 10
        else:
            print(
                f"Skipping unsupported checkpoint (not in expected family): {pretrained_model}"
            )
            continue

        print(f"{basename}: {MODEL_NAME}")

        state = torch.load(pretrained_model, map_location="cpu")
        if MODEL_NAME == "efficientnet-b7":
            if "eb7m" in pretrained_model:
                net.model.load_state_dict(state, strict=True)
            else:
                net.load_state_dict(state, strict=True)
        else:
            net.load_state_dict(state, strict=True)

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
            proba = predict_model(basename, net, dataloader)
            probability.append(proba)

        del net
        torch.cuda.empty_cache()

    if len(probability) == 0:
        print(
            "No supported checkpoints were usable; falling back to sample_submission labels to produce a valid submission."
        )
        df_test = pd.read_csv(f"{BASE_DIR}/sample_submission.csv")[
            ["image_id", "label"]
        ].copy()
    else:
        proba_mean = np.mean(np.stack(probability, axis=0), axis=0)
        df_test["mean"] = proba_mean.argmax(axis=1)
        df_test["label"] = df_test["mean"].astype(int)

        print(f"total time: {time.time() - start_time:.2f}[sec]")



## === cell 20
if len(df_test) == 2 and df_test.loc[0, "image_id"] == df_test.loc[1, "image_id"]:
    df_test = pd.read_csv(f"{BASE_DIR}/sample_submission.csv")
else:
    if "label" not in df_test.columns:
        df_test["label"] = df_test["mean"]



## === cell 21
df_test.head()



## === cell 22
sub = df_test[["image_id", "label"]].copy()
sub["label"] = sub["label"].astype(int)
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())
