# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Create a classifier to predict the severity of diabetic retinopathy.

## Metric
Quadratic weighted kappa, which measures the agreement between two ratings. This metric typically varies from 0 (random agreement between raters) to 1 (complete agreement between raters). In the event that there is less agreement between the raters than expected by chance, this metric may go below 0. The quadratic weighted kappa is calculated between the scores assigned by the human rater and the predicted scores.

Images have five possible ratings, 0,1,2,3,4.  Each image is characterized by a tuple *(e*,*e)*, which corresponds to its scores by *Rater A* (human) and *Rater B* (predicted).  The quadratic weighted kappa is calculated as follows. First, an N x N histogram matrix *O* is constructed, such that *O* corresponds to the number of images that received a rating *i* by *A* and a rating *j* by *B*. An *N-by-N* matrix of weights, *w*, is calculated based on the difference between raters' scores:

An *N-by-N* histogram matrix of expected ratings, *E*, is calculated, assuming that there is no correlation between rating scores.  This is calculated as the outer product between each rater's histogram vector of ratings, normalized such that *E* and *O* have the same sum.

## Submission Format
```
id_code,diagnosis
0005cfc8afb6,0
003f0afdcd15,0
etc.
```

## Dataset
You are provided with a large set of retina images taken using [fundus photography](https://en.wikipedia.org/wiki/Fundus_photography) under a variety of imaging conditions.

Labels are on a scale of 0 to 4:

> 0 - No DR
> 1 - Mild
> 2 - Moderate
> 3 - Severe
> 4 - Proliferative DR

Images may contain artifacts, be out of focus, underexposed, or overexposed. The images were gathered from multiple clinics using a variety of cameras over an extended period of time, which will introduce further variation.

- **train.csv** - the training labels
- **test.csv** - the test set (you must predict the `diagnosis` value for these variables)
- **sample_submission.csv** - a sample submission file in the correct format
- **train.zip** - the training set images
- **test.zip** - the public test set images

# 2. Python version

3.9

# 3. Installed packages

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
sklearn-pandas==2.2.0
timm==1.0.19
torch==2.6.0+cu124
torchao==0.10.0
torchaudio==2.6.0+cu124
torchdata==0.11.0
torchinfo==1.8.0
torchmetrics==1.8.2
torchsummary==1.5.1
torchtune==0.6.1
torchvision==0.21.0+cu124

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (118 lines)
            sample_submission.csv (368 lines)
            sample_submission.csv.zip (3.2 kB)
            test.csv (368 lines)
            test.csv.zip (2.9 kB)
            test.zip (160 Bytes)
            test_images.zip (902.9 MB)
            train.csv (3296 lines)
            train.csv.zip (27.5 kB)
            train.zip (162 Bytes)
            train_images.zip (7.7 GB)
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
            test_images/
                218c822a3dd9.png (5.7 MB)
                0e82bcacc475.png (5.2 MB)
                ... and 365 other files
                test_images/
            train_images/
                184a185e7447.png (337.5 kB)
                c4aef0d88d1b.png (876.6 kB)
                ... and 3293 other files
                train_images/
        input/
            description.md (118 lines)
            sample_submission.csv (368 lines)
            sample_submission.csv.zip (3.2 kB)
            test.csv (368 lines)
            test.csv.zip (2.9 kB)
            test.zip (160 Bytes)
            test_images.zip (902.9 MB)
            train.csv (3296 lines)
            train.csv.zip (27.5 kB)
            train.zip (162 Bytes)
            train_images.zip (7.7 GB)
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
            test_images/
                218c822a3dd9.png (5.7 MB)
                0e82bcacc475.png (5.2 MB)
                ... and 365 other files
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
            train_images/
                184a185e7447.png (337.5 kB)
                c4aef0d88d1b.png (876.6 kB)
                ... and 3293 other files
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
        working/
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
```

-> data/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/aptos2019-blindness-detection/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> data/aptos2019-blindness-detection/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> data/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> input/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os
import glob
import math
import random
import time
import copy
import json
import pickle
import csv

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import functional as FT

from PIL import Image, ImageChops

import cv2
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold
import timm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

print("Device:", device)



## === cell 1
from torch import Tensor
from torch.jit.annotations import List, Optional


def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, flatten=False):
        super(GeM, self).__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps
        self.flatten = flatten

    def forward(self, x):
        x = gem(x, p=self.p, eps=self.eps)
        if self.flatten:
            x = x.flatten(1)
        return x

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + "p="
            + "{:.4f}".format(self.p.data.tolist()[0])
            + ", "
            + "eps="
            + str(self.eps)
            + ")"
        )


class Linear(nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if torch.jit.is_scripting():
            bias = self.bias.to(dtype=input.dtype) if self.bias is not None else None
            return F.linear(input, self.weight.to(dtype=input.dtype), bias=bias)
        else:
            return F.linear(input, self.weight, self.bias)


class ThreeStage_Model(nn.Module):
    def __init__(self, backbone=None):
        super(ThreeStage_Model, self).__init__()
        self.backbone = timm.models.tf_efficientnet_b4_ns(pretrained=False)
        self.backbone.global_pool = GeM(flatten=True)

        self.classifier = nn.Sequential(
            nn.SiLU(),
            nn.Linear(1000, 500),
            nn.SiLU(),
            nn.Linear(500, 5),
        )

        self.regressor = nn.Sequential(
            nn.SiLU(),
            nn.Linear(1000, 500),
            nn.SiLU(),
            nn.Linear(500, 1),
        )

        self.ordinal = nn.Sequential(
            nn.SiLU(),
            nn.Linear(1000, 500),
            nn.SiLU(),
            nn.Linear(500, 4),
        )

        self.final_regressor = nn.Sequential(
            nn.SiLU(),
            nn.Linear(10, 1),
        )

    def forward(self, x, final=False):
        x = self.backbone(x)
        c_out = self.classifier(x)
        r_out = self.regressor(x)
        o_out = self.ordinal(x)

        if final:
            out = torch.cat((c_out, r_out, o_out), 1)
            out = self.final_regressor(out)
            out = torch.sigmoid(out) * 4.5
            return out
        else:
            r_out = torch.sigmoid(r_out) * 4.5
            o_out = torch.sigmoid(o_out)
            return c_out, r_out, o_out


def build_model(pretrained_backbone: bool, device: torch.device):
    m = ThreeStage_Model()
    m.backbone = timm.create_model(
        "tf_efficientnet_b4_ns", pretrained=pretrained_backbone
    )
    m.backbone.global_pool = GeM(flatten=True)
    m = m.to(device)
    return m




## === cell 2
threshold = [0.75, 1.5, 2.5, 3.5]


def apply_thresholds(x_cont, thr):
    thr = np.asarray(thr, dtype=float)
    y = np.zeros_like(x_cont, dtype=int)
    for t in thr:
        y += (x_cont >= t).astype(int)
    return np.clip(y, 0, 4)


def _precompute_qwk_W(n_classes=5):
    denom = float((n_classes - 1) ** 2)
    idx = np.arange(n_classes, dtype=np.float64)
    W = (idx[:, None] - idx[None, :]) ** 2 / denom
    return W


_QWK_W5 = _precompute_qwk_W(5)


def _qwk_from_preds(y_true_int, y_pred_int, n_classes=5, W=None):
    y_true_int = np.asarray(y_true_int, dtype=np.int64)
    y_pred_int = np.asarray(y_pred_int, dtype=np.int64)
    if W is None:
        W = _precompute_qwk_W(n_classes)

    O = (
        np.bincount(
            y_true_int * n_classes + y_pred_int, minlength=n_classes * n_classes
        )
        .reshape(n_classes, n_classes)
        .astype(np.float64)
    )

    act_hist = O.sum(axis=1)
    pred_hist = O.sum(axis=0)
    E = np.outer(act_hist, pred_hist)
    sE = E.sum()
    sO = O.sum()
    if sE > 0:
        E *= sO / sE

    num = (W * O).sum()
    den = (W * E).sum()
    if den == 0:
        return 1.0 if num == 0 else 0.0
    return 1.0 - (num / den)


def optimize_thresholds_for_kappa(
    y_true, x_cont, init_thr=(0.75, 1.5, 2.5, 3.5), n_iter=3
):
    """
    Change (score-toward-target): slightly stronger threshold fitting improves QWK
    without changing model/loss/transform; this usually raises public score.
    """
    thr = np.array(init_thr, dtype=float)
    y_true = np.asarray(y_true, dtype=np.int64)
    x_cont = np.asarray(x_cont, dtype=np.float64)

    best_k = _qwk_from_preds(
        y_true, apply_thresholds(x_cont, thr), n_classes=5, W=_QWK_W5
    )

    bounds = [(0.0, 4.5)] * 4
    grid_sizes = [120, 120, 120, 120]

    for _ in range(n_iter):
        for i in range(4):
            lo, hi = bounds[i]
            candidates = np.linspace(lo, hi, grid_sizes[i])
            cur_best_thr_i = thr[i]
            cur_best_k = best_k
            for v in candidates:
                tmp = thr.copy()
                tmp[i] = v
                tmp = np.sort(tmp)
                k = _qwk_from_preds(
                    y_true, apply_thresholds(x_cont, tmp), n_classes=5, W=_QWK_W5
                )
                if k > cur_best_k:
                    cur_best_k = k
                    cur_best_thr_i = v
            thr[i] = cur_best_thr_i
            thr = np.sort(thr)
            best_k = cur_best_k

    for i in range(4):
        center = thr[i]
        lo = max(0.0, center - 0.25)
        hi = min(4.5, center + 0.25)
        candidates = np.linspace(lo, hi, 101)
        cur_best_thr_i = thr[i]
        cur_best_k = best_k
        for v in candidates:
            tmp = thr.copy()
            tmp[i] = v
            tmp = np.sort(tmp)
            k = _qwk_from_preds(
                y_true, apply_thresholds(x_cont, tmp), n_classes=5, W=_QWK_W5
            )
            if k > cur_best_k:
                cur_best_k = k
                cur_best_thr_i = v
        thr[i] = cur_best_thr_i
        thr = np.sort(thr)
        best_k = cur_best_k

    return thr.tolist(), float(best_k)




## === cell 3
class photometric_distort(object):
    def __call__(self, image):
        distortions = [
            FT.adjust_brightness,
            FT.adjust_contrast,
            FT.adjust_saturation,
            FT.adjust_hue,
        ]

        random.shuffle(distortions)

        for d in distortions:
            if random.random() < 0.5:
                if d.__name__ == "adjust_hue":
                    adjust_factor = random.uniform(-16 / 255.0, 16 / 255.0)
                else:
                    adjust_factor = random.uniform(0.7, 1.3)
                image = d(image, adjust_factor)

        return image


class cropTo4_3(object):
    def __call__(self, image):
        w, h = image.size
        if (w / h) >= (4 / 3):
            new_h = h
            new_w = int(h * 4 / 3)
        else:
            new_h = int(w * 3 / 4)
            new_w = w

        left = (w - new_w) / 2
        top = (h - new_h) / 2
        right = left + new_w
        bottom = top + new_h
        return image.crop((left, top, right, bottom))


class trim(object):
    def __call__(self, image):
        bg = Image.new(image.mode, image.size, image.getpixel((0, 0)))
        diff = ImageChops.difference(image, bg)
        diff = ImageChops.add(diff, diff, 2.0, -10)
        bbox = diff.getbbox()
        if bbox:
            return image.crop(bbox)
        return image




## === cell 4
BASE_INPUT = "/kaggle/input/aptos2019-blindness-detection"
train_csv_path = os.path.join(BASE_INPUT, "train.csv")
train_img_dir = os.path.join(BASE_INPUT, "train_images")
test_csv_path = os.path.join(BASE_INPUT, "test.csv")
test_img_dir = os.path.join(BASE_INPUT, "test_images")

train_df = pd.read_csv(train_csv_path)
test_ids_df = pd.read_csv(test_csv_path)
test_ids = np.squeeze(test_ids_df["id_code"].values)

transform1 = transforms.Compose(
    [
        trim(),
        cropTo4_3(),
        transforms.Resize((288, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.384, 0.258, 0.174], std=[0.124, 0.089, 0.094]),
    ]
)

net1 = build_model(pretrained_backbone=True, device=device)

candidate_patterns = [
    "/kaggle/input/**/0.926_B4_3stage_5epoch_320finetune.pkl",
    "/kaggle/input/**/*B4*3stage*finetune*.pkl",
    "/kaggle/input/**/*.pkl",
]
weight_path = None
for pat in candidate_patterns:
    matches = glob.glob(pat, recursive=True)
    if matches:
        exact = [
            m for m in matches if m.endswith("0.926_B4_3stage_5epoch_320finetune.pkl")
        ]
        weight_path = exact[0] if exact else matches[0]
        break

loaded_ckpt = False
if weight_path is not None and os.path.exists(weight_path):
    state = torch.load(weight_path, map_location="cpu")
    if (
        isinstance(state, dict)
        and "state_dict" in state
        and isinstance(state["state_dict"], dict)
    ):
        state = state["state_dict"]
    if isinstance(state, dict):
        new_state = {}
        for k, v in state.items():
            nk = k[7:] if k.startswith("module.") else k
            new_state[nk] = v
        state = new_state
    missing, unexpected = net1.load_state_dict(state, strict=False)
    loaded_ckpt = True
    print(f"Loaded checkpoint: {weight_path}")
    print(f"Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
else:
    print(
        "No external .pkl checkpoint found; using ImageNet-pretrained EfficientNet backbone weights."
    )

print(f"Using device: {device}")
print(f"Train images dir: {train_img_dir}")
print(f"Test images dir: {test_img_dir}")
print(f"Num train: {len(train_df)}; Num test ids: {len(test_ids)}")




## === cell 5
class IdImageDataset(Dataset):
    def __init__(self, df, img_dir, transform, has_label=False):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.has_label = has_label

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        idx = str(row["id_code"])
        image_name = os.path.join(self.img_dir, f"{idx}.png")
        img = Image.open(image_name).convert("RGB")
        img = self.transform(img)
        if self.has_label:
            y = int(row["diagnosis"])
            return idx, img, y
        return idx, img


class CachedIdImageDataset(Dataset):
    def __init__(self, df, img_dir, transform, has_label=False, cache=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.has_label = has_label
        self.cache = cache if cache is not None else {}

    def __len__(self):
        return len(self.df)

    def _load_and_transform(self, idx):
        image_name = os.path.join(self.img_dir, f"{idx}.png")
        img = Image.open(image_name).convert("RGB")
        img = self.transform(img)
        return img

    def __getitem__(self, i):
        row = self.df.iloc[i]
        idx = str(row["id_code"])
        img = self.cache.get(idx)
        if img is None:
            img = self._load_and_transform(idx)
            self.cache[idx] = img
        if self.has_label:
            y = int(row["diagnosis"])
            return idx, img, y
        return idx, img


class PreloadedTensorDataset(Dataset):
    def __init__(self, df, tensor_dict, has_label=False):
        self.df = df.reset_index(drop=True)
        self.tensor_dict = tensor_dict
        self.has_label = has_label

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        idx = str(row["id_code"])
        img = self.tensor_dict[idx]
        if self.has_label:
            y = int(row["diagnosis"])
            return idx, img, y
        return idx, img


def collate_fn_test(batch):
    ids = [b[0] for b in batch]
    imgs = torch.stack([b[1] for b in batch], dim=0)
    return ids, imgs


def collate_fn_train(batch):
    ids = [b[0] for b in batch]
    imgs = torch.stack([b[1] for b in batch], dim=0)
    ys = torch.tensor([b[2] for b in batch], dtype=torch.long)
    return ids, imgs, ys




## === cell 6
for p in net1.parameters():
    p.requires_grad = False

for p in net1.regressor.parameters():
    p.requires_grad = True

for name, module in [
    ("conv_head", getattr(net1.backbone, "conv_head", None)),
    ("bn2", getattr(net1.backbone, "bn2", None)),
    ("act2", getattr(net1.backbone, "act2", None)),
]:
    if module is not None:
        for p in module.parameters():
            p.requires_grad = True

val_frac = 0.2  # kept for reference, but we now use 5-fold OOF for threshold fit.

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

cpu_count = os.cpu_count() or 2
num_workers = min(4, max(2, cpu_count // 2)) if cpu_count > 2 else 0

batch_size = 8
epochs = 5
criterion = nn.SmoothL1Loss(beta=1.0)


def _preload_tensors(df, img_dir, transform, desc="preload"):
    out = {}
    t0 = time.time()
    for idx in df["id_code"].astype(str).tolist():
        path = os.path.join(img_dir, f"{idx}.png")
        img = Image.open(path).convert("RGB")
        out[idx] = transform(img)
    print(f"{desc}: cached {len(out)} images in {time.time()-t0:.1f}s")
    return out


train_tensors = _preload_tensors(
    train_df, train_img_dir, transform1, desc="train preload"
)
test_tensors = _preload_tensors(
    test_ids_df, test_img_dir, transform1, desc="test preload"
)

train_full_ds = PreloadedTensorDataset(train_df, train_tensors, has_label=True)

oof_pred = np.zeros(len(train_df), dtype=np.float32)
oof_true = train_df["diagnosis"].values.astype(int)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

init_state = {k: v.detach().cpu().clone() for k, v in net1.state_dict().items()}

fold_model = build_model(pretrained_backbone=False, device=device)
fold_model.load_state_dict(init_state, strict=True)

fold_times = []
for fold, (tr_idx, va_idx) in enumerate(
    skf.split(np.zeros(len(train_df)), oof_true), 1
):
    t0 = time.time()

    fold_model.load_state_dict(init_state, strict=True)

    for p in fold_model.parameters():
        p.requires_grad = False
    for p in fold_model.regressor.parameters():
        p.requires_grad = True
    for name, module in [
        ("conv_head", getattr(fold_model.backbone, "conv_head", None)),
        ("bn2", getattr(fold_model.backbone, "bn2", None)),
        ("act2", getattr(fold_model.backbone, "act2", None)),
    ]:
        if module is not None:
            for p in module.parameters():
                p.requires_grad = True

    trn_df = train_df.iloc[tr_idx].reset_index(drop=True)
    val_df = train_df.iloc[va_idx].reset_index(drop=True)

    train_ds = PreloadedTensorDataset(trn_df, train_tensors, has_label=True)
    val_ds = PreloadedTensorDataset(val_df, train_tensors, has_label=True)

    loader_kwargs = dict(
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn_train,
    )

    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        **loader_kwargs,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        **loader_kwargs,
    )

    params = [p for p in fold_model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=2e-4)

    fold_model.train()
    for ep in range(epochs):
        running = 0.0
        n = 0
        for _, imgs, ys in train_dl:
            imgs = imgs.to(device, non_blocking=True)
            y = ys.to(device, non_blocking=True).float().unsqueeze(1)

            optimizer.zero_grad(set_to_none=True)
            c_out, r_out, o_out = fold_model(imgs)
            loss = criterion(r_out, y)
            loss.backward()
            optimizer.step()

            running += loss.item() * imgs.size(0)
            n += imgs.size(0)

        print(
            f"Fold {fold}/5 - Epoch {ep+1}/{epochs} - train SmoothL1: {running/max(n,1):.5f}"
        )

    fold_model.eval()
    preds = []
    with torch.no_grad():
        for _, imgs, ys in val_dl:
            imgs = imgs.to(device, non_blocking=True)
            c_out, r_out, o_out = fold_model(imgs)
            preds.append(r_out.detach().squeeze(1).float().cpu().numpy())
    preds = np.concatenate(preds, axis=0)

    oof_pred[va_idx] = preds

    fold_time = time.time() - t0
    fold_times.append(fold_time)
    print(f"Fold {fold}/5 done in {fold_time:.1f}s")

opt_thr, opt_k = optimize_thresholds_for_kappa(
    oof_true, oof_pred, init_thr=threshold, n_iter=3
)
print("OOF optimized thresholds:", opt_thr)
print("OOF QWK with optimized thresholds:", opt_k)

final_model = build_model(pretrained_backbone=False, device=device)
final_model.load_state_dict(init_state, strict=True)

for p in final_model.parameters():
    p.requires_grad = False
for p in final_model.regressor.parameters():
    p.requires_grad = True
for name, module in [
    ("conv_head", getattr(final_model.backbone, "conv_head", None)),
    ("bn2", getattr(final_model.backbone, "bn2", None)),
    ("act2", getattr(final_model.backbone, "act2", None)),
]:
    if module is not None:
        for p in module.parameters():
            p.requires_grad = True

loader_kwargs_final = dict(
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
    collate_fn=collate_fn_train,
)

final_dl = DataLoader(
    train_full_ds,
    batch_size=batch_size,
    shuffle=True,
    **loader_kwargs_final,
)

params = [p for p in final_model.parameters() if p.requires_grad]
optimizer = optim.Adam(params, lr=2e-4)

final_model.train()
t0 = time.time()
for ep in range(epochs):
    running = 0.0
    n = 0
    for _, imgs, ys in final_dl:
        imgs = imgs.to(device, non_blocking=True)
        y = ys.to(device, non_blocking=True).float().unsqueeze(1)

        optimizer.zero_grad(set_to_none=True)
        c_out, r_out, o_out = final_model(imgs)
        loss = criterion(r_out, y)
        loss.backward()
        optimizer.step()

        running += loss.item() * imgs.size(0)
        n += imgs.size(0)

    print(
        f"Final train - Epoch {ep+1}/{epochs} - train SmoothL1: {running/max(n,1):.5f}"
    )

print(f"Final training time: {time.time()-t0:.1f}s")
final_model.eval()



## === cell 7
test_ds = PreloadedTensorDataset(test_ids_df, test_tensors, has_label=False)

loader_kwargs_test = dict(
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
    collate_fn=collate_fn_test,
)

test_dl = DataLoader(
    test_ds,
    batch_size=8,
    shuffle=False,
    **loader_kwargs_test,
)

submission = []
with torch.no_grad():
    for ids_batch, imgs in test_dl:
        imgs = imgs.to(device, non_blocking=True)
        c_out, r_out, o_out = final_model(imgs)

        r_out_cpu = r_out.detach().squeeze(1).float().cpu().numpy()
        P = apply_thresholds(r_out_cpu, opt_thr).astype(int)

        for _id, p in zip(ids_batch, P.tolist()):
            submission.append([_id, int(p)])

submission = np.array(submission, dtype=object)



## === cell 8
if len(submission) == 0:
    raise RuntimeError("Submission is empty; inference did not run.")

df = pd.DataFrame(submission, columns=["id_code", "diagnosis"])
df["id_code"] = df["id_code"].astype(str)
df["diagnosis"] = df["diagnosis"].astype(int)

df = test_ids_df.merge(df, on="id_code", how="left")
if df["diagnosis"].isna().any():
    missing = df[df["diagnosis"].isna()]["id_code"].tolist()[:5]
    raise RuntimeError(
        f"Some test ids are missing predictions (showing up to 5): {missing}"
    )

df.to_csv("submission.csv", index=False)

print(df.head())
print("Wrote submission.csv with shape:", df.shape)
print("submission.csv saved to:", os.path.abspath("submission.csv"))
