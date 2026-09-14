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
Label artwork images with significant attributes.

## Metric
Micro averaged F1 score.

## Submission Format
```
id,attribute_ids
00011f01965f141f5d1eea6592fa9862,0 1 2
00014abc91ed3e4bf1663fde8136fe80,0 1 2
0002e2054e303badc1a33463f6fb7973,0 1 2
```

## Dataset
Multiple modalities can be expected and the camera sources are unknown. The photographs are often centered for objects, and in the case where the museum artifact is an entire room, the images are scenic in nature.

Each object is annotated by a single annotator without a verification step. You should consider these annotations noisy.

The filename of each image is its `id`.

- **train.csv** gives the `attribute_ids` for the train images in **/train**
- **/test** contains the test images. You must predict the `attribute_ids` for these images.
- **sample_submission.csv** contains a submission in the correct format
- **labels.csv** provides descriptions of the attributes

# 2. Python version

3.8

# 3. Installed packages

albumentations==2.0.8
geopandas==0.14.4
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pillow==11.3.0
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
scipy==1.15.3
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
            description.md (81 lines)
            labels.csv (3475 lines)
            labels.csv.zip (28.4 kB)
            sample_submission.csv (21319 lines)
            sample_submission.csv.zip (426.2 kB)
            test.zip (3.8 GB)
            train.csv (120802 lines)
            train.csv.zip (3.2 MB)
            train.zip (21.4 GB)
            imet-2020-fgvc7/
                description.md (81 lines)
                labels.csv (3475 lines)
                ... and 7 other files
                imet-2020-fgvc7/
                test/
                    c48792ab798551af4bc8e7dec5d89c31.png (47.1 kB)
                    2e4ae3954a0167eedc8ac74256eace08.png (223.2 kB)
                    ... and 21316 other files
                    test/
                train/
                    cae7d65be972cbfa9e1af483219fbe26.png (212.7 kB)
                    bcbeb1d22d738370626f4686338e469f.png (137.5 kB)
                    ... and 120799 other files
                    train/
            test/
                c48792ab798551af4bc8e7dec5d89c31.png (47.1 kB)
                2e4ae3954a0167eedc8ac74256eace08.png (223.2 kB)
                ... and 21316 other files
                test/
            train/
                cae7d65be972cbfa9e1af483219fbe26.png (212.7 kB)
                bcbeb1d22d738370626f4686338e469f.png (137.5 kB)
                ... and 120799 other files
                train/
        input/
            description.md (81 lines)
            labels.csv (3475 lines)
            labels.csv.zip (28.4 kB)
            sample_submission.csv (21319 lines)
            sample_submission.csv.zip (426.2 kB)
            test.zip (3.8 GB)
            train.csv (120802 lines)
            train.csv.zip (3.2 MB)
            train.zip (21.4 GB)
            imet-2020-fgvc7/
                description.md (81 lines)
                labels.csv (3475 lines)
                ... and 7 other files
                imet-2020-fgvc7/
                test/
                    c48792ab798551af4bc8e7dec5d89c31.png (47.1 kB)
                    2e4ae3954a0167eedc8ac74256eace08.png (223.2 kB)
                    ... and 21316 other files
                    test/
                train/
                    cae7d65be972cbfa9e1af483219fbe26.png (212.7 kB)
                    bcbeb1d22d738370626f4686338e469f.png (137.5 kB)
                    ... and 120799 other files
                    train/
            test/
                c48792ab798551af4bc8e7dec5d89c31.png (47.1 kB)
                2e4ae3954a0167eedc8ac74256eace08.png (223.2 kB)
                ... and 21316 other files
                test/
                    c48792ab798551af4bc8e7dec5d89c31.png (47.1 kB)
                    2e4ae3954a0167eedc8ac74256eace08.png (223.2 kB)
                    ... and 21316 other files
                    test/
            train/
                cae7d65be972cbfa9e1af483219fbe26.png (212.7 kB)
                bcbeb1d22d738370626f4686338e469f.png (137.5 kB)
                ... and 120799 other files
                train/
                    cae7d65be972cbfa9e1af483219fbe26.png (212.7 kB)
                    bcbeb1d22d738370626f4686338e469f.png (137.5 kB)
                    ... and 120799 other files
                    train/
        working/
            imet-2020-fgvc7/
                description.md (81 lines)
                labels.csv (3475 lines)
                ... and 7 other files
                imet-2020-fgvc7/
                test/
                    c48792ab798551af4bc8e7dec5d89c31.png (47.1 kB)
                    2e4ae3954a0167eedc8ac74256eace08.png (223.2 kB)
                    ... and 21316 other files
                    test/
                train/
                    cae7d65be972cbfa9e1af483219fbe26.png (212.7 kB)
                    bcbeb1d22d738370626f4686338e469f.png (137.5 kB)
                    ... and 120799 other files
                    train/
```

-> data/imet-2020-fgvc7/labels.csv has 3474 rows and 2 columns.
The columns are: attribute_id, attribute_name

-> data/imet-2020-fgvc7/sample_submission.csv has 21318 rows and 2 columns.
The columns are: id, attribute_ids

-> data/imet-2020-fgvc7/train.csv has 120801 rows and 2 columns.
The columns are: id, attribute_ids

-> data/labels.csv has 3474 rows and 2 columns.
The columns are: attribute_id, attribute_name

-> data/sample_submission.csv has 21318 rows and 2 columns.
The columns are: id, attribute_ids

-> data/train.csv has 120801 rows and 2 columns.
The columns are: id, attribute_ids

-> (stopped after 10 files for performance)

# 5. Target score

0.5896283010295454

# 6. Current score

0.06655

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.00331) has done: 'I fix the missing weights issue by removing the dependency on an external `/kaggle/input/imet2020` directory and instead run a minimal, standard single-epoch training on the provided `train.csv` images, then do inference on test and write `submission.csv`. I also fix the device/dtype mismatch that caused CUDA tensors to be fed through a CPU-weight model by moving the model to `device` before inference and ensuring tensors are `float`. Finally, I make the image loading robust (fallback to PIL if OpenCV fails) and ensure we always generate at least one predicted label per row to avoid empty `attribute_ids`, producing a valid submission file with the required columns.'
- What this solution (achieved 0.2098) has done: 'I fix the albumentations `RandomResizedCrop` API change that currently crashes training by switching to the v2-compatible signature (explicit `size=(H, W)` and `scale=(...)`). I also fix a logic bug in the `ShuffleNet.fresh_params()` method (it references a non-existent attribute), which can break fine-tuning if used. Finally, I keep the model/training loop the same but make inference output formatting robust (string dtype) to guarantee a valid `submission.csv` with the required columns; these changes are score-neutral except that they allow the model to actually train, which should raise the score substantially toward the target.'
- What this solution (achieved 0.06655) has done: 'To move your score up toward the 0.5896 target with minimal disruption, I keep the exact same model/training loop/loss and instead fix the main score limiter: the current fixed threshold (0.10) is almost certainly badly calibrated for micro-F1 in this multi-label setting. I add a small in-training validation split and compute an out-of-fold probability threshold that maximizes micro-F1 (same evaluation semantics as Kaggle), then use that single threshold for test predictions. I also set `pretrained=True` using the torchvision-2.6 “weights” API to ensure pretrained weights are actually loaded (older `pretrained=` can silently degrade), which should improve representations without changing architecture. These changes are lightweight, keep runtime within budget, and should increase score substantially toward the target.'

# 9. Code solution

## === cell 0
import os
import json
import sys
import gc
import random
import time
from typing import Dict
from contextlib import contextmanager
from pathlib import Path
from collections import defaultdict, Counter

import cv2
from PIL import Image
import numpy as np
import pandas as pd
import scipy as sp

import sklearn.metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

from functools import partial
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.models as M
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models

from albumentations import (
    Compose,
    Normalize,
    Resize,
    RandomResizedCrop,
    RandomCrop,
    HorizontalFlip,
)
from albumentations.pytorch import ToTensorV2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device




## === cell 1
def init_logger(log_file="train.log"):
    from logging import getLogger, DEBUG, FileHandler, Formatter, StreamHandler

    log_format = "%(asctime)s %(levelname)s %(message)s"

    stream_handler = StreamHandler()
    stream_handler.setLevel(DEBUG)
    stream_handler.setFormatter(Formatter(log_format))

    file_handler = FileHandler(log_file)
    file_handler.setFormatter(Formatter(log_format))

    logger = getLogger("Herbarium")
    logger.setLevel(DEBUG)
    if not logger.handlers:
        logger.addHandler(stream_handler)
        logger.addHandler(file_handler)

    return logger


LOG_FILE = "train.log"
LOGGER = init_logger(LOG_FILE)


@contextmanager
def timer(name):
    t0 = time.time()
    LOGGER.info(f"[{name}] start")
    yield
    LOGGER.info(f"[{name}] done in {time.time() - t0:.0f} s.")


def seed_torch(seed=777):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


SEED = 777
seed_torch(SEED)



## === cell 2
submission = pd.read_csv("../input/imet-2020-fgvc7/sample_submission.csv")
train_df = pd.read_csv("../input/imet-2020-fgvc7/train.csv")



## === cell 3
N_CLASSES = 3474


def _read_image_rgb(file_path: str) -> np.ndarray:
    img = cv2.imread(file_path)
    if img is None:
        with Image.open(file_path) as im:
            im = im.convert("RGB")
            return np.array(im)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


class TrainDataset(Dataset):
    def __init__(self, df, labels, transform=None):
        self.df = df.reset_index(drop=True)
        self.labels = labels.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.df["id"].values[idx]
        file_path = f"../input/imet-2020-fgvc7/train/{file_name}.png"
        image = _read_image_rgb(file_path)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        label = self.labels.values[idx]
        target = torch.zeros(N_CLASSES, dtype=torch.float32)
        for cls in str(label).split():
            if cls != "":
                target[int(cls)] = 1.0

        return image.float(), target


class TestDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.df["id"].values[idx]
        file_path = f"../input/imet-2020-fgvc7/test/{file_name}.png"
        image = _read_image_rgb(file_path)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        return image.float()




## === cell 4
HEIGHT = 128
WIDTH = 128


def get_transforms(*, data):
    assert data in ("train", "valid")

    if data == "train":
        return Compose(
            [
                RandomResizedCrop(
                    size=(HEIGHT, WIDTH),
                    scale=(0.08, 1.0),
                    ratio=(0.75, 1.3333333333333333),
                ),
                Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )

    elif data == "valid":
        return Compose(
            [
                Resize(HEIGHT, WIDTH),
                Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )




## === cell 5
batch_size = 128

test_dataset = TestDataset(submission, transform=get_transforms(data="valid"))
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)




## === cell 6
class AvgPool(nn.Module):
    def forward(self, x):
        return F.avg_pool2d(x, x.shape[2:])


class ResNet(nn.Module):
    def __init__(
        self, num_classes, pretrained=False, net_cls=M.resnet50, dropout=False
    ):
        super().__init__()
        if pretrained:
            try:
                weights = M.ResNet50_Weights.DEFAULT
                self.net = net_cls(weights=weights)
            except Exception:
                self.net = net_cls(pretrained=True)
        else:
            try:
                self.net = net_cls(weights=None)
            except Exception:
                self.net = net_cls(pretrained=False)

        self.net.avgpool = AvgPool()
        if dropout:
            self.net.fc = nn.Sequential(
                nn.Dropout(),
                nn.Linear(self.net.fc.in_features, num_classes),
            )
        else:
            self.net.fc = nn.Linear(self.net.fc.in_features, num_classes)

    def fresh_params(self):
        return self.net.fc.parameters()

    def forward(self, x):
        return self.net(x)


class DenseNet(nn.Module):
    def __init__(self, num_classes, pretrained=False, net_cls=M.densenet121):
        super().__init__()
        if pretrained:
            try:
                weights = M.DenseNet121_Weights.DEFAULT
                self.net = net_cls(weights=weights)
            except Exception:
                self.net = net_cls(pretrained=True)
        else:
            try:
                self.net = net_cls(weights=None)
            except Exception:
                self.net = net_cls(pretrained=False)

        self.avg_pool = AvgPool()
        self.net.classifier = nn.Linear(self.net.classifier.in_features, num_classes)

    def fresh_params(self):
        return self.net.classifier.parameters()

    def forward(self, x):
        out = self.net.features(x)
        out = F.relu(out, inplace=True)
        out = self.avg_pool(out).view(out.size(0), -1)
        out = self.net.classifier(out)
        return out


class ShuffleNet(nn.Module):
    def __init__(
        self, num_classes, pretrained=False, net_cls=M.shufflenet_v2_x1_0, dropout=False
    ):
        super().__init__()
        if pretrained:
            try:
                weights = M.ShuffleNet_V2_X1_0_Weights.DEFAULT
                self.net = net_cls(weights=weights)
            except Exception:
                self.net = net_cls(pretrained=True)
        else:
            try:
                self.net = net_cls(weights=None)
            except Exception:
                self.net = net_cls(pretrained=False)

        if dropout:
            self.net.fc = nn.Sequential(
                nn.Dropout(),
                nn.Linear(self.net.fc.in_features, num_classes),
            )
        else:
            self.net.fc = nn.Linear(self.net.fc.in_features, num_classes)

    def fresh_params(self):
        return self.net.fc.parameters()

    def forward(self, x):
        return self.net(x)


class SqueezeNet(nn.Module):
    def __init__(
        self, num_classes, pretrained=False, net_cls=M.squeezenet1_0, dropout=False
    ):
        super().__init__()
        if pretrained:
            try:
                weights = M.SqueezeNet1_0_Weights.DEFAULT
                self.net = net_cls(weights=weights)
            except Exception:
                self.net = net_cls(pretrained=True)
        else:
            try:
                self.net = net_cls(weights=None)
            except Exception:
                self.net = net_cls(pretrained=False)

        fin_conv = nn.Conv2d(512, num_classes, kernel_size=1)
        self.net.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            fin_conv,
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def fresh_params(self):
        return self.net.classifier.parameters()

    def forward(self, x):
        out = self.net.features(x)
        out = self.net.classifier(out)
        out = out.view(-1, 3474)
        return out


class MobileNet(nn.Module):
    def __init__(
        self, num_classes, pretrained=False, net_cls=M.mobilenet_v2, dropout=False
    ):
        super().__init__()
        if pretrained:
            try:
                weights = M.MobileNet_V2_Weights.DEFAULT
                self.net = net_cls(weights=weights)
            except Exception:
                self.net = net_cls(pretrained=True)
        else:
            try:
                self.net = net_cls(weights=None)
            except Exception:
                self.net = net_cls(pretrained=False)

        self.net.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.net.last_channel, num_classes),
        )

    def fresh_params(self):
        return self.net.classifier.parameters()

    def forward(self, x):
        return self.net(x)


resnet50 = partial(ResNet, net_cls=M.resnet50)
densenet121 = partial(DenseNet, net_cls=M.densenet121)
shufflenet = partial(ShuffleNet, net_cls=M.shufflenet_v2_x1_0)
squeezenet = partial(SqueezeNet, net_cls=M.squeezenet1_0)
mobilenet = partial(MobileNet, net_cls=M.mobilenet_v2)



## === cell 7
model = densenet121(num_classes=N_CLASSES, pretrained=True).to(device)



## === cell 8
criterion = nn.BCEWithLogitsLoss(reduction="mean")
optimizer = Adam(model.parameters(), lr=1e-3)



## === cell 9
rng = np.random.RandomState(SEED)
perm = rng.permutation(len(train_df))
n_valid = int(0.05 * len(train_df))  # small holdout for threshold tuning
valid_idx = perm[:n_valid]
train_idx = perm[n_valid:]

train_split_df = train_df.iloc[train_idx].reset_index(drop=True)
valid_split_df = train_df.iloc[valid_idx].reset_index(drop=True)

train_dataset = TrainDataset(
    train_split_df,
    train_split_df["attribute_ids"],
    transform=get_transforms(data="train"),
)
valid_dataset = TrainDataset(
    valid_split_df,
    valid_split_df["attribute_ids"],
    transform=get_transforms(data="valid"),
)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
    drop_last=True,
)
valid_loader = DataLoader(
    valid_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
    drop_last=False,
)

with timer("train (1 epoch)"):
    model.train()
    tk0 = tqdm(enumerate(train_loader), total=len(train_loader))
    for i, (images, targets) in tk0:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        if (i + 1) % 50 == 0:
            tk0.set_postfix(loss=float(loss.detach().cpu().item()))

        if time.time() - tk0.start_t > 240:  # ~4 min cap for training
            LOGGER.info("Stopping training early due to runtime budget (safety cap).")
            break




## === cell 10
def micro_f1_from_probs(y_true: np.ndarray, y_prob: np.ndarray, thr: float) -> float:
    y_pred = (y_prob > thr).astype(np.uint8)
    return sklearn.metrics.f1_score(
        y_true.reshape(-1), y_pred.reshape(-1), average="micro", zero_division=0
    )


with timer("valid inference + threshold tuning"):
    model.eval()
    valid_probs = []
    valid_targets = []
    tk0 = tqdm(enumerate(valid_loader), total=len(valid_loader))
    for i, (images, targets) in tk0:
        images = images.to(device, non_blocking=True)
        with torch.no_grad():
            logits = model(images)
            prob = torch.sigmoid(logits).detach().to("cpu").numpy()
        valid_probs.append(prob)
        valid_targets.append(targets.numpy())

    valid_probs = np.concatenate(valid_probs, axis=0)
    valid_targets = np.concatenate(valid_targets, axis=0).astype(np.uint8)

    thr_grid = np.linspace(0.02, 0.30, 15)
    best_thr = 0.10
    best_f1 = -1.0
    for thr in thr_grid:
        f1 = micro_f1_from_probs(valid_targets, valid_probs, float(thr))
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)

    LOGGER.info(f"Chosen threshold={best_thr:.4f} (valid micro-F1={best_f1:.5f})")



## === cell 11
with timer("inference"):
    model.eval()

    preds = []
    tk0 = tqdm(enumerate(test_loader), total=len(test_loader))

    for i, images in tk0:
        images = images.to(device, non_blocking=True)

        with torch.no_grad():
            y_preds = model(images)

        preds.append(torch.sigmoid(y_preds).detach().to("cpu").numpy())



## === cell 12
threshold = best_thr

if len(preds) == 0:
    raise RuntimeError("No predictions were produced. Check test_loader/dataset.")

probs = np.concatenate(preds, axis=0)  # shape: (n_test, n_classes)
predictions = probs > threshold

submission["attribute_ids"] = submission["attribute_ids"].astype(str)

for i in range(predictions.shape[0]):
    ids = np.nonzero(predictions[i])[0]
    if ids.size == 0:
        ids = np.array([int(np.argmax(probs[i]))], dtype=np.int64)
    submission.loc[i, "attribute_ids"] = " ".join([str(int(x)) for x in ids])

submission.to_csv("submission.csv", index=False)
submission.head()
