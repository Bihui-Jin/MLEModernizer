# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

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

from functools import partial, lru_cache
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
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


SEED = 777
seed_torch(SEED)

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True



## === cell 2
submission = pd.read_csv("../input/imet-2020-fgvc7/sample_submission.csv")
train_df = pd.read_csv("../input/imet-2020-fgvc7/train.csv")



## === cell 3
N_CLASSES = 3474

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

_GLOBAL_IMG_CACHE = None
_GLOBAL_IMG_CACHE_MAX = 8000  # per worker process; bounded RAM usage


def _get_img_cache():
    global _GLOBAL_IMG_CACHE
    if _GLOBAL_IMG_CACHE is None:
        _GLOBAL_IMG_CACHE = {}
    return _GLOBAL_IMG_CACHE


def _read_image_rgb(file_path: str, resize_hw=None) -> np.ndarray:
    key = (file_path, resize_hw)
    cache = _get_img_cache()
    if key in cache:
        return cache[key]

    img = cv2.imread(file_path, cv2.IMREAD_COLOR)
    if img is None:
        with Image.open(file_path) as im:
            im = im.convert("RGB")
            arr = np.array(im)
            if resize_hw is not None:
                arr = cv2.resize(
                    arr, (resize_hw[1], resize_hw[0]), interpolation=cv2.INTER_AREA
                )
            out = np.ascontiguousarray(arr)
    else:
        if resize_hw is not None:
            img = cv2.resize(
                img, (resize_hw[1], resize_hw[0]), interpolation=cv2.INTER_AREA
            )
        out = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        out = np.ascontiguousarray(out)

    if _GLOBAL_IMG_CACHE_MAX > 0:
        if len(cache) >= _GLOBAL_IMG_CACHE_MAX:
            cache.pop(next(iter(cache)))
        cache[key] = out
    return out


class TrainDataset(Dataset):
    def __init__(
        self, df, targets_multihot: np.ndarray, transform=None, pre_resize_hw=None
    ):
        self.df = df.reset_index(drop=True)
        self.targets = targets_multihot  # shape (n, N_CLASSES), dtype float32
        self.transform = transform
        self._ids = self.df["id"].astype(str).values
        self.pre_resize_hw = pre_resize_hw
        self._base = "../input/imet-2020-fgvc7/train/"
        self._ext = ".png"

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self._ids[idx]
        file_path = self._base + file_name + self._ext
        image = _read_image_rgb(file_path, resize_hw=self.pre_resize_hw)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        target = torch.from_numpy(self.targets[idx])
        return image.float(), target


class TestDataset(Dataset):
    def __init__(self, df, transform=None, pre_resize_hw=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self._ids = self.df["id"].astype(str).values
        self.pre_resize_hw = pre_resize_hw
        self._base = "../input/imet-2020-fgvc7/test/"
        self._ext = ".png"

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self._ids[idx]
        file_path = self._base + file_name + self._ext
        image = _read_image_rgb(file_path, resize_hw=self.pre_resize_hw)

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
                RandomCrop(HEIGHT, WIDTH),
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

_cpu = os.cpu_count() or 4

_num_workers = min(8, max(2, _cpu - 2))

_pre_decode_resize = (256, 256)


def _worker_init_fn(worker_id: int):
    seed = SEED + worker_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    global _GLOBAL_IMG_CACHE
    _GLOBAL_IMG_CACHE = {}


_common_loader_kwargs = dict(
    num_workers=_num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(_num_workers > 0),
    prefetch_factor=2 if _num_workers > 0 else None,  # Speed: less RAM/queue overhead
    worker_init_fn=_worker_init_fn if _num_workers > 0 else None,
)

test_dataset = TestDataset(
    submission, transform=get_transforms(data="valid"), pre_resize_hw=_pre_decode_resize
)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    **_common_loader_kwargs,
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

if torch.cuda.is_available():
    model = model.to(memory_format=torch.channels_last)

_try_compile = hasattr(torch, "compile")
if _try_compile:
    try:
        model = torch.compile(model, mode="max-autotune", fullgraph=False)
        LOGGER.info("torch.compile enabled (mode=max-autotune).")
    except Exception as e:
        LOGGER.info(f"torch.compile not used due to: {repr(e)}")



## === cell 8
criterion = nn.BCEWithLogitsLoss(reduction="mean")
optimizer = Adam(model.parameters(), lr=1e-3)




## === cell 9
def _parse_labels_to_lists(attr_series: pd.Series):
    vals = attr_series.fillna("").astype(str).values
    out = []
    for s in vals:
        if s == "" or s.lower() == "nan":
            out.append([])
            continue
        toks = s.split()
        if len(toks) == 0:
            out.append([])
        else:
            out.append([int(t) for t in toks if t != ""])
    return out


all_label_lists = _parse_labels_to_lists(train_df["attribute_ids"])

_flat = np.fromiter((c for lst in all_label_lists for c in lst), dtype=np.int64)
_counts = np.bincount(_flat, minlength=N_CLASSES).astype(np.int64, copy=False)
class_priors = _counts / max(1, len(all_label_lists))
top_prior_classes = np.argsort(-class_priors)[:10].astype(np.int64)

rng = np.random.RandomState(SEED)
perm = rng.permutation(len(train_df))

n_valid = int(0.10 * len(train_df))
valid_idx = perm[:n_valid]
train_idx = perm[n_valid:]

train_split_df = train_df.iloc[train_idx].reset_index(drop=True)
valid_split_df = train_df.iloc[valid_idx].reset_index(drop=True)

train_label_lists = [all_label_lists[i] for i in train_idx]
valid_label_lists = [all_label_lists[i] for i in valid_idx]


def _multihot_vectorized(label_lists, n_classes: int, dtype):
    n = len(label_lists)
    row_counts = np.fromiter((len(lst) for lst in label_lists), dtype=np.int64, count=n)
    nnz = int(row_counts.sum())
    y = np.zeros((n, n_classes), dtype=dtype)
    if nnz == 0:
        return y
    rows = np.repeat(np.arange(n, dtype=np.int64), row_counts)
    cols = np.fromiter(
        (c for lst in label_lists for c in lst), dtype=np.int64, count=nnz
    )
    y[rows, cols] = 1
    return y


train_targets = _multihot_vectorized(train_label_lists, N_CLASSES, np.float32)
valid_targets_np_u8 = _multihot_vectorized(valid_label_lists, N_CLASSES, np.uint8)

train_dataset = TrainDataset(
    train_split_df,
    train_targets,
    transform=get_transforms(data="train"),
    pre_resize_hw=_pre_decode_resize,
)
valid_dataset = TrainDataset(
    valid_split_df,
    valid_targets_np_u8.astype(np.float32, copy=False),
    transform=get_transforms(data="valid"),
    pre_resize_hw=_pre_decode_resize,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    **_common_loader_kwargs,
)
valid_loader = DataLoader(
    valid_dataset,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    **_common_loader_kwargs,
)

with timer("train (1 epoch)"):
    model.train()
    tk0 = tqdm(enumerate(train_loader), total=len(train_loader))
    for i, (images, targets) in tk0:
        if torch.cuda.is_available():
            images = images.to(device, non_blocking=True).contiguous(
                memory_format=torch.channels_last
            )
            targets = targets.to(device, non_blocking=True)
        else:
            images = images.to(device)
            targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        if (i + 1) % 50 == 0:
            tk0.set_postfix(loss=float(loss.detach().cpu().item()))




## === cell 10
def micro_f1_from_preds(y_true_u8: np.ndarray, y_pred_u8: np.ndarray) -> float:
    y_true = y_true_u8.astype(np.uint8, copy=False)
    y_pred = y_pred_u8.astype(np.uint8, copy=False)
    tp = int(np.bitwise_and(y_true, y_pred).sum())
    fp = int(np.bitwise_and(1 - y_true, y_pred).sum())
    fn = int(np.bitwise_and(y_true, 1 - y_pred).sum())
    denom = 2 * tp + fp + fn
    if denom == 0:
        return 0.0
    return (2.0 * tp) / float(denom)


def _micro_f1_over_threshold_grid(
    y_true_u8: np.ndarray, probs: np.ndarray, thr_grid: np.ndarray
):
    y_true = y_true_u8.astype(np.uint8, copy=False)
    n, c = probs.shape
    y_true_sum = int(y_true.sum())

    probs_sorted = np.sort(probs, axis=0)  # ascending, shape (N, C)
    y_true_sorted = np.take_along_axis(
        y_true, np.argsort(probs, axis=0), axis=0
    ).astype(np.int32, copy=False)
    prefix = np.cumsum(y_true_sorted, axis=0, dtype=np.int64)
    total_pos_per_class = prefix[-1].astype(np.int64, copy=False)

    best_thr = float(thr_grid[0])
    best_f1 = -1.0

    for thr in thr_grid.astype(probs.dtype, copy=False):
        idx = np.searchsorted(probs_sorted, thr, side="right", axis=0)  # (C,)
        pred_pos_per_class = (n - idx).astype(np.int64, copy=False)

        head_pos = np.where(idx > 0, prefix[idx - 1, np.arange(c)], 0)
        tp_per_class = total_pos_per_class - head_pos
        tp = int(tp_per_class.sum())
        pred_sum = int(pred_pos_per_class.sum())
        fp = pred_sum - tp
        fn = y_true_sum - tp

        denom = 2 * tp + fp + fn
        f1 = (2.0 * tp) / float(denom) if denom > 0 else 0.0
        if f1 > best_f1:
            best_f1 = float(f1)
            best_thr = float(thr)

    return best_thr, best_f1


with timer("valid inference + calibration"):
    model.eval()
    n_valid_total = len(valid_dataset)

    valid_probs = np.empty((n_valid_total, N_CLASSES), dtype=np.float32)
    valid_logits = np.empty((n_valid_total, N_CLASSES), dtype=np.float32)

    ofs = 0
    tk0 = tqdm(enumerate(valid_loader), total=len(valid_loader))
    with torch.inference_mode():
        for i, (images, targets) in tk0:
            bs = images.shape[0]
            if torch.cuda.is_available():
                images = images.to(device, non_blocking=True).contiguous(
                    memory_format=torch.channels_last
                )
            else:
                images = images.to(device)

            logits = model(images)
            prob = torch.sigmoid(logits)

            valid_logits[ofs : ofs + bs] = logits.float().cpu().numpy()
            valid_probs[ofs : ofs + bs] = prob.float().cpu().numpy()
            ofs += bs

    valid_targets = valid_targets_np_u8

    base_thr = 0.20
    eps = 1e-6
    per_class_thr_raw = base_thr + 0.08 * np.log(
        (class_priors + eps) / (np.median(class_priors[class_priors > 0]) + eps)
    )
    per_class_thr_raw = np.clip(per_class_thr_raw, 0.03, 0.50)

    offset_grid = np.linspace(-0.08, 0.08, 17).astype(np.float32)
    best_offset = 0.0
    best_f1_pc = -1.0
    for off in offset_grid:
        thr_vec = np.clip(per_class_thr_raw + float(off), 0.02, 0.60).astype(
            np.float32, copy=False
        )
        y_pred = (valid_probs > thr_vec[None, :]).astype(np.uint8)
        f1 = micro_f1_from_preds(valid_targets, y_pred)
        if f1 > best_f1_pc:
            best_f1_pc = f1
            best_offset = float(off)

    per_class_thr = np.clip(per_class_thr_raw + best_offset, 0.02, 0.60).astype(
        np.float32
    )

    global_grid = np.linspace(0.02, 0.40, 39).astype(np.float32)
    best_global_thr, best_f1_global = _micro_f1_over_threshold_grid(
        valid_targets, valid_probs, global_grid
    )

    temp_grid = np.array([0.7, 0.85, 1.0, 1.2, 1.5], dtype=np.float32)
    best_temp = 1.0
    best_f1_pc_temp = -1.0
    for T in temp_grid:
        probs_T = 1.0 / (1.0 + np.exp(-(valid_logits / float(T))))
        y_pred = (probs_T > per_class_thr[None, :]).astype(np.uint8)
        f1 = micro_f1_from_preds(valid_targets, y_pred)
        if f1 > best_f1_pc_temp:
            best_f1_pc_temp = f1
            best_temp = float(T)

    best_method = "global"
    best_valid_f1 = best_f1_global

    if best_f1_pc > best_valid_f1:
        best_method = "per_class"
        best_valid_f1 = best_f1_pc

    if best_f1_pc_temp > best_valid_f1:
        best_method = "per_class_temp"
        best_valid_f1 = best_f1_pc_temp

    if best_method == "global":
        use_global_threshold = True
        use_temp_scaling = False
        LOGGER.info(
            f"Calibration chose GLOBAL thr={best_global_thr:.4f} (valid micro-F1={best_f1_global:.5f}); "
            f"per-class={best_f1_pc:.5f}; per-class+temp={best_f1_pc_temp:.5f}"
        )
    elif best_method == "per_class":
        use_global_threshold = False
        use_temp_scaling = False
        LOGGER.info(
            f"Calibration chose PER-CLASS offset={best_offset:+.4f} (valid micro-F1={best_f1_pc:.5f}); "
            f"global={best_f1_global:.5f}; per-class+temp={best_f1_pc_temp:.5f}"
        )
    else:
        use_global_threshold = False
        use_temp_scaling = True
        LOGGER.info(
            f"Calibration chose PER-CLASS+TEMP offset={best_offset:+.4f}, T={best_temp:.3f} "
            f"(valid micro-F1={best_f1_pc_temp:.5f}); global={best_f1_global:.5f}; per-class={best_f1_pc:.5f}"
        )

    true_k = valid_targets.sum(axis=1)
    expected_k = int(np.clip(np.round(np.median(true_k)), 1, 10))
    LOGGER.info(
        f"Calibrated fallback expected_k={expected_k} (median labels/image on holdout)"
    )



## === cell 11
with timer("inference"):
    model.eval()

    n_test = len(test_dataset)

    need_logits = ("use_temp_scaling" in globals()) and use_temp_scaling
    probs = np.empty((n_test, N_CLASSES), dtype=np.float32)
    test_logits = (
        np.empty((n_test, N_CLASSES), dtype=np.float32) if need_logits else None
    )

    ofs = 0
    tk0 = tqdm(enumerate(test_loader), total=len(test_loader))

    with torch.inference_mode():
        for i, images in tk0:
            bs = images.shape[0]
            if torch.cuda.is_available():
                images = images.to(device, non_blocking=True).contiguous(
                    memory_format=torch.channels_last
                )
            else:
                images = images.to(device)

            y_logits = model(images)
            y_prob = torch.sigmoid(y_logits)

            if need_logits:
                test_logits[ofs : ofs + bs] = y_logits.float().cpu().numpy()
            probs[ofs : ofs + bs] = y_prob.float().cpu().numpy()
            ofs += bs



## === cell 12
if "use_global_threshold" in globals() and use_global_threshold:
    predictions = probs > best_global_thr
else:
    if "use_temp_scaling" in globals() and use_temp_scaling:
        probs_T = 1.0 / (1.0 + np.exp(-(test_logits / float(best_temp))))
        predictions = probs_T > per_class_thr[None, :]
    else:
        predictions = probs > per_class_thr[None, :]

prior_boost = (0.05 * class_priors).astype(probs.dtype, copy=False)

pred_any = predictions.any(axis=1)
attr_out = [""] * predictions.shape[0]

idx_pos = np.flatnonzero(pred_any)
for i in idx_pos.tolist():
    ids = np.flatnonzero(predictions[i])
    attr_out[i] = " ".join(map(str, ids.tolist()))

idx_zero = np.flatnonzero(~pred_any)
if idx_zero.size > 0:
    score = probs[idx_zero] + prior_boost[None, :]
    topk = np.argpartition(-score, kth=expected_k - 1, axis=1)[:, :expected_k]
    row_scores = np.take_along_axis(score, topk, axis=1)
    order = np.argsort(-row_scores, axis=1)
    topk_sorted = np.take_along_axis(topk, order, axis=1)
    for j, i in enumerate(idx_zero.tolist()):
        attr_out[i] = " ".join(map(str, topk_sorted[j].tolist()))

submission["attribute_ids"] = attr_out
submission.to_csv("submission.csv", index=False)
submission.head()
