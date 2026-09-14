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

# 5. Target score

0.926329653636144

# 6. Current score

0.69522

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the missing weights issue by loading from the correct Kaggle input location if available, and otherwise fall back to running the same model with random weights so the pipeline still completes and writes a valid (non-empty) CSV. I also fix the CUDA crash by selecting CPU when no GPU is available, and make `ordinal2class_prob` device-safe (it currently mixes `.cuda()` and `.cpu()`). Finally, I make the image preprocessing robust by ensuring `trim()` always returns an image, preventing occasional `None` propagation into transforms. These changes are execution/stability fixes and preserve your core model/inference logic.'
- What this solution (achieved 0.64278) has done: 'Your 0.0 score is consistent with running an untrained (randomly initialized) model because the expected pretrained weights file is not present in your input paths. The smallest change to move toward your target is to (1) reliably locate/load weights if they exist anywhere under the competition dataset directory and (2) if they still don’t exist, train the *existing* `ThreeStage_Model` briefly on `train.csv` using the same regression head you already use at inference, so predictions become meaningful. I’m keeping your model, transforms, and regression-to-class mapping intact; the only “new” logic is a short, deterministic training loop and a safer weight search. The script still produces a valid `submission.csv` end-to-end within the time limit by using EfficientNet-B4 with a small batch size and few epochs.'
- What this solution (achieved 0.62784) has done: 'Your current pipeline is training only the regression branch but then thresholding with fixed cutoffs that likely don’t match the trained regressor’s calibration, which hurts quadratic weighted kappa. I keep your model, transforms, loss, and training loop structure intact, and add a minimal post-training step that fits 4 thresholds on the held-out validation split to maximize QWK, then reuse those thresholds for test predictions. This is a small, metric-aligned change that typically improves kappa a lot without changing the core modeling approach. I also switch test inference to batched DataLoader (same transforms/predictions) to stay within the 600s limit and ensure deterministic, consistent ordering.'
- What this solution (achieved 0.66402) has done: 'Your current score is below the target, so the smallest safe way to move upward is to improve metric alignment without changing the model or training loop. I keep your regressor-only training and inference exactly as-is, but (1) calibrate thresholds using out-of-fold (OOF) predictions from a small K-fold split instead of a single holdout, which reduces variance and usually raises public QWK, and (2) train for the same total number of epochs (2) but distributed across folds so compute stays similar. This keeps the core logic intact (same model, loss, optimizer, transforms, and regress→class thresholding) while making threshold fitting more reliable. The script still runs end-to-end and writes `submission.csv`.'
- What this solution (achieved 0.69522) has done: 'The timeout is dominated by expensive PIL preprocessing (trim/crop/resize) done repeatedly inside `__getitem__` plus slow single-sample Python loops in `regress2class` and the submission assembly loop. To preserve the exact model and transforms while cutting wall time, the script below adds a deterministic on-disk cache of the *transformed tensor* per image (so each image is decoded and transformed at most once per run), increases DataLoader throughput with safe settings, vectorizes thresholding, and avoids per-row Python appends when building the submission. These changes are provably equivalent to the original computations (same transforms, same network forward, same thresholds), just removing redundant work and Python overhead.'

# 9. Code solution

## === cell 0
import os
import random
import time
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from PIL import Image, ImageChops

from torchvision import transforms
from torchvision.transforms import functional as FT

import timm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True




## === cell 1
threshold = [0.75, 1.5, 2.5, 3.5]


def regress2class(out, thr=None):
    thr = threshold if thr is None else thr
    thr_t = torch.as_tensor(thr, device=out.device, dtype=out.dtype).view(
        1, -1
    )  # (1,4)
    outv = out.view(-1, 1)
    pred = (outv >= thr_t).sum(dim=1).to(torch.float32)
    return pred.cpu()


def ordinal2class_prob(out):
    dev = out.device
    pred_prob = torch.zeros(out.size(0), 5, device=dev)
    pred_prob[:, 0] = (1 - out[:, 0]).squeeze()
    pred_prob[:, 1] = (out[:, 0] * (1 - out[:, 1])).squeeze()
    pred_prob[:, 2] = (out[:, 1] * (1 - out[:, 2])).squeeze()
    pred_prob[:, 3] = (out[:, 2] * (1 - out[:, 3])).squeeze()
    pred_prob[:, 4] = out[:, 3].squeeze()
    return F.softmax(pred_prob, dim=1)


def regress2class_prob(out):
    pred_prob = torch.zeros((out.size(0), 5), device=out.device)
    for i in range(out.size(0)):
        if out[i] < 4.0:
            l1 = int(math.floor(out[i].item()))
            l2 = int(math.ceil(out[i].item()))
            pred_prob[i][l1] = 1 - (out[i] - l1)
            pred_prob[i][l2] = 1 - (l2 - out[i])
        else:
            pred_prob[i][4] = 1.0
    return pred_prob




## === cell 2
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


class Regressor(nn.Module):
    def __init__(self):
        super(Regressor, self).__init__()
        self.backbone = timm.models.tf_efficientnet_b5_ns(pretrained=False)
        self.backbone.global_pool = GeM(flatten=True)
        self.regressor = nn.Linear(1000, 1)

    def forward(self, x):
        x = self.backbone(x)
        out = self.regressor(x)
        out = torch.sigmoid(out) * 4.5
        return out


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
BASE = "../input/aptos2019-blindness-detection"
if not os.path.exists(BASE):
    if os.path.exists("/kaggle/input/aptos2019-blindness-detection"):
        BASE = "/kaggle/input/aptos2019-blindness-detection"
    elif os.path.exists("/kaggle/data/aptos2019-blindness-detection"):
        BASE = "/kaggle/data/aptos2019-blindness-detection"

TRAIN_CSV = os.path.join(BASE, "train.csv")
TEST_CSV = os.path.join(BASE, "test.csv")
TRAIN_IMG_DIR = os.path.join(BASE, "train_images")
TEST_IMG_DIR = os.path.join(BASE, "test_images")

print("BASE:", BASE)
print("TRAIN_CSV exists:", os.path.exists(TRAIN_CSV))
print("TEST_CSV exists:", os.path.exists(TEST_CSV))
print("TRAIN_IMG_DIR exists:", os.path.exists(TRAIN_IMG_DIR))
print("TEST_IMG_DIR exists:", os.path.exists(TEST_IMG_DIR))

train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)

train_df["id_code"] = train_df["id_code"].astype(str)
train_df["diagnosis"] = train_df["diagnosis"].astype(int)
test_df["id_code"] = test_df["id_code"].astype(str)

test_ids = test_df["id_code"].values

input_size = 384

transform = transforms.Compose(
    [
        trim(),
        cropTo4_3(),
        transforms.Resize((input_size * 3 // 4, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.384, 0.258, 0.174], std=[0.124, 0.089, 0.094]),
    ]
)

CACHE_DIR = os.path.join("/kaggle/working", f"aptos_cache_{input_size}")
os.makedirs(CACHE_DIR, exist_ok=True)
print("CACHE_DIR:", CACHE_DIR)




## === cell 5
def find_weight_file():
    candidates = [
        "../input/weights/B4_3stage_5epoch_320finetune.pkl",
        "/kaggle/input/weights/B4_3stage_5epoch_320finetune.pkl",
        os.path.join(BASE, "B4_3stage_5epoch_320finetune.pkl"),
        "/kaggle/input/aptos2019-blindness-detection/B4_3stage_5epoch_320finetune.pkl",
    ]
    for p in candidates:
        if p and os.path.exists(p):
            return p

    for root, _, files in os.walk(BASE):
        for fn in files:
            if fn == "B4_3stage_5epoch_320finetune.pkl":
                return os.path.join(root, fn)
    return None


net = ThreeStage_Model()
weight_path = find_weight_file()

loaded_weights = False
if weight_path is None:
    print(
        "WARNING: pretrained weights not found. Will train the existing regression branch to improve QWK."
    )
else:
    print("Loading weights:", weight_path)
    state = torch.load(weight_path, map_location="cpu")
    net.load_state_dict(state)
    loaded_weights = True

net = net.to(device)




## === cell 6
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import cohen_kappa_score


class AptosDataset(Dataset):
    def __init__(self, df, img_dir, transform, has_labels=True, cache_dir=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.has_labels = has_labels
        self.cache_dir = cache_dir

    def __len__(self):
        return len(self.df)

    def _cache_path(self, id_code: str):
        folder_tag = os.path.basename(self.img_dir.rstrip("/"))
        return os.path.join(self.cache_dir, f"{folder_tag}__{id_code}.pt")

    def _load_or_make_tensor(self, id_code: str):
        if self.cache_dir is None:
            img_path = os.path.join(self.img_dir, f"{id_code}.png")
            img = Image.open(img_path).convert("RGB")
            return self.transform(img)

        cp = self._cache_path(id_code)
        try:
            if os.path.exists(cp):
                return torch.load(cp, map_location="cpu", weights_only=False)
        except Exception:
            pass

        img_path = os.path.join(self.img_dir, f"{id_code}.png")
        img = Image.open(img_path).convert("RGB")
        x = self.transform(img)
        tmp = cp + ".tmp"
        try:
            torch.save(x, tmp)
            os.replace(tmp, cp)
        except Exception:
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except Exception:
                pass
        return x

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        id_code = str(row["id_code"])
        x = self._load_or_make_tensor(id_code)
        if self.has_labels:
            y = torch.tensor(row["diagnosis"], dtype=torch.float32)
            return x, y
        else:
            return x, id_code


def _qwk(y_true, y_pred_int):
    return cohen_kappa_score(y_true, y_pred_int, weights="quadratic")


def fit_thresholds_for_qwk(y_true_int, y_pred_float, init_thr=None):
    y_true_int = np.asarray(y_true_int, dtype=np.int64)
    y_pred_float = np.asarray(y_pred_float, dtype=np.float32)

    thr = np.array(
        init_thr if init_thr is not None else [0.75, 1.5, 2.5, 3.5], dtype=np.float32
    )
    thr = np.sort(thr)

    def apply_thr(pred, t):
        t = np.sort(t)
        return (
            (pred >= t[0]).astype(np.int64)
            + (pred >= t[1]).astype(np.int64)
            + (pred >= t[2]).astype(np.int64)
            + (pred >= t[3]).astype(np.int64)
        )

    best_thr = thr.copy()
    best_score = _qwk(y_true_int, apply_thr(y_pred_float, best_thr))

    step = 0.25
    for _ in range(25):
        improved = False
        for j in range(4):
            for delta in (-step, step):
                cand = best_thr.copy()
                cand[j] = cand[j] + delta
                cand = np.clip(cand, 0.0, 4.5)
                cand = np.sort(cand)
                if np.any(np.diff(cand) <= 1e-4):
                    continue
                sc = _qwk(y_true_int, apply_thr(y_pred_float, cand))
                if sc > best_score:
                    best_score = sc
                    best_thr = cand
                    improved = True
        if not improved:
            step *= 0.5
            if step < 0.01:
                break

    return best_thr.tolist(), float(best_score)


def _set_requires_grad(module, flag: bool):
    for p in module.parameters():
        p.requires_grad = flag


def _make_loader(ds, batch_size, shuffle, num_workers):
    kwargs = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
    )
    if kwargs["prefetch_factor"] is None:
        kwargs.pop("prefetch_factor")
    return DataLoader(ds, **kwargs)


val_thresholds = None

if not loaded_weights:
    n = len(train_df)
    rs = np.random.RandomState(42)
    perm = rs.permutation(n)

    folds = 3
    fold_sizes = [(n // folds) + (1 if i < (n % folds) else 0) for i in range(folds)]
    starts = np.cumsum([0] + fold_sizes[:-1]).tolist()

    batch_size = 10 if device.type == "cuda" else 4

    num_workers = min(4, (os.cpu_count() or 2))

    criterion = nn.SmoothL1Loss()

    head_only_epochs_per_fold = 1
    finetune_epochs_per_fold = 1  # brief unfreeze for small improvement
    lr_head = 3e-4
    lr_finetune = 1e-4

    oof_pred = np.zeros(n, dtype=np.float32)
    oof_true = train_df["diagnosis"].values.astype(np.int64)

    start_time = time.time()

    for fi in range(folds):
        va_start = starts[fi]
        va_end = va_start + fold_sizes[fi]
        va_idx = perm[va_start:va_end]
        tr_idx = np.concatenate([perm[:va_start], perm[va_end:]], axis=0)

        tr_df = train_df.iloc[tr_idx].copy()
        va_df = train_df.iloc[va_idx].copy()

        train_ds = AptosDataset(
            tr_df, TRAIN_IMG_DIR, transform, has_labels=True, cache_dir=CACHE_DIR
        )
        val_ds = AptosDataset(
            va_df, TRAIN_IMG_DIR, transform, has_labels=True, cache_dir=CACHE_DIR
        )

        train_loader = _make_loader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        val_loader = _make_loader(
            val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        _set_requires_grad(net.backbone, False)
        _set_requires_grad(net.regressor, True)
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, net.parameters()), lr=lr_head
        )

        net.train()
        for ep in range(1, head_only_epochs_per_fold + 1):
            ep_losses = []
            for xb, yb in train_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                _, r_out, _ = net(xb)
                r_out = r_out.squeeze(1)
                loss = criterion(r_out, yb)
                loss.backward()
                optimizer.step()
                ep_losses.append(loss.item())
            print(
                f"Fold {fi+1}/{folds} head-only epoch {ep}/{head_only_epochs_per_fold} "
                f"train_loss={float(np.mean(ep_losses)):.4f} elapsed={time.time()-start_time:.1f}s"
            )

        _set_requires_grad(net.backbone, True)
        _set_requires_grad(net.regressor, True)
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, net.parameters()), lr=lr_finetune
        )

        net.train()
        for ep in range(1, finetune_epochs_per_fold + 1):
            ep_losses = []
            for xb, yb in train_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                _, r_out, _ = net(xb)
                r_out = r_out.squeeze(1)
                loss = criterion(r_out, yb)
                loss.backward()
                optimizer.step()
                ep_losses.append(loss.item())
            print(
                f"Fold {fi+1}/{folds} finetune epoch {ep}/{finetune_epochs_per_fold} "
                f"train_loss={float(np.mean(ep_losses)):.4f} elapsed={time.time()-start_time:.1f}s"
            )

        net.eval()
        fold_preds = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                _, r_out, _ = net(xb)
                r_out = r_out.squeeze(1)
                fold_preds.append(r_out.detach().cpu().numpy())
        fold_preds = np.concatenate(fold_preds, axis=0) if fold_preds else np.array([])
        if len(fold_preds) != len(va_df):
            raise RuntimeError("OOF collection size mismatch.")
        oof_pred[va_idx] = fold_preds.astype(np.float32)

        fold_y = va_df["diagnosis"].values.astype(np.int64)
        fold_pred_i = (
            (fold_preds >= threshold[0]).astype(np.int64)
            + (fold_preds >= threshold[1]).astype(np.int64)
            + (fold_preds >= threshold[2]).astype(np.int64)
            + (fold_preds >= threshold[3]).astype(np.int64)
        )
        print(
            f"Fold {fi+1}/{folds} val_qwk(default_thr)={_qwk(fold_y, fold_pred_i):.4f}"
        )

    val_thresholds, val_qwk_fit = fit_thresholds_for_qwk(
        y_true_int=oof_true,
        y_pred_float=oof_pred,
        init_thr=threshold,
    )
    print("Fitted thresholds (OOF):", val_thresholds, "oof_qwk(fit_thr)=", val_qwk_fit)
    threshold = val_thresholds  # use for test inference
    net.eval()
else:
    net.eval()




## === cell 7
test_ds = AptosDataset(
    test_df, TEST_IMG_DIR, transform, has_labels=False, cache_dir=CACHE_DIR
)

batch_size = 10 if device.type == "cuda" else 4
num_workers = min(4, (os.cpu_count() or 2))

test_loader = DataLoader(
    test_ds,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=(device.type == "cuda"),
    persistent_workers=(num_workers > 0),
    prefetch_factor=4 if num_workers > 0 else 2,
)

all_ids = []
all_pred = []

with torch.no_grad():
    seen = 0
    for xb, id_codes in test_loader:
        if seen % 50 == 0:
            print("Predicting", seen, "/", len(test_ds))
        xb = xb.to(device, non_blocking=True)
        _, r_out, _ = net(xb)
        r_out = r_out.squeeze(1)
        pred = regress2class(r_out, thr=threshold).numpy().astype(np.int64)

        all_ids.extend(list(id_codes))
        all_pred.append(pred)
        seen += len(id_codes)

all_pred = (
    np.concatenate(all_pred, axis=0) if len(all_pred) else np.array([], dtype=np.int64)
)
submission = np.column_stack(
    [np.asarray(all_ids, dtype=object), all_pred.astype(object)]
)
print("Num predictions:", len(submission))




## === cell 8
df = pd.DataFrame(submission, columns=["id_code", "diagnosis"])

assert (
    len(df) == len(test_ids) and len(df) > 0
), "Submission DataFrame is empty or mis-sized."
df["id_code"] = df["id_code"].astype(str)
df["diagnosis"] = df["diagnosis"].astype(int).clip(0, 4)

df = df.set_index("id_code").loc[test_df["id_code"].astype(str).values].reset_index()

df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", df.shape)
print("Using thresholds:", threshold)
print(df.head())
