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

0.9045029843609144

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I remove the broken `pip install` cell and the hard-coded `../input/weights/...pth` dependency that causes the `FileNotFoundError`, and instead train the same EfficientNet-B4 regression model directly from the provided `train.csv`/`train_images` so the notebook runs in this environment. I also fix path handling to use `/kaggle/input/aptos2019-blindness-detection/...` and ensure we always write a valid `submission.csv` with the required columns and row order matching `test.csv`. To move the score toward the target (higher is better) without changing the core approach, I keep the same regression output/rounding scheme and add a small validation-based threshold calibration for discretization (still consistent with quadratic kappa optimization). Finally, I make execution robust (GPU/CPU fallback, deterministic seeds) and keep runtime under the limit by using a single EfficientNet-B4 fine-tuning run with conservative defaults.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is consistent with a submission-format/type issue (APTOS expects integer classes 0–4; if `diagnosis` is saved as float/object or misaligned, Kaggle can effectively score it as all-wrong). I make the smallest changes to (1) guarantee correct row alignment and integer dtype in `submission.csv`, and (2) reduce avoidable distribution shift by using ImageNet normalization for EfficientNet (a minimal preprocessing correction that typically improves kappa without changing the model/training loop). I also clamp/regression-scale to exactly the valid label range [0,4] before thresholding to stabilize discretization. Core architecture, loss, and training approach remain the same; runtime stays within the same order.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with a submission/evaluation mismatch rather than model quality, so I make the smallest changes that ensure Kaggle reads your predictions correctly and that the predicted labels are aligned and valid integers 0–4. Specifically, I (1) enforce `id_code` as string everywhere and sort predictions to exactly match `test.csv` order (no map-induced NaNs or silent misalignment), (2) guarantee `diagnosis` is a plain integer column (not numpy int64/object) by casting to Python `int` via `astype("int32")`, and (3) slightly stabilize threshold calibration by clamping/cleaning thresholds and using the already-computed validation predictions only (no change to model/training core logic). These are minimal, metric-aligned fixes intended to move the score upward from 0.0 toward your target without changing architecture, loss, or training loop behavior.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is almost certainly coming from an evaluation mismatch rather than model quality: this competition’s test set is unlabeled and Kaggle assigns QWK only on a hidden test set, so a true 0.0 commonly indicates predictions were effectively unreadable/invalid or heavily misaligned. I make minimal, score-relevant changes to harden submission validity: (1) enforce exact `id_code` order without a merge (direct assignment in test.csv order), (2) guarantee `diagnosis` is plain integer classes 0–4 (Python int-compatible) and never float/NaN, and (3) clamp/clean thresholds and predictions in one place to prevent edge-case class leakage. Core model, loss, training loop, and regression→thresholding approach stay the same; these changes aim to move you upward from 0.0 toward the target by ensuring Kaggle evaluates what you intend.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most likely caused by predictions not being evaluated as intended (e.g., hidden NaNs in `diagnosis`, wrong dtype/format, or an order mismatch that slipped through despite appearing correct). I keep your EfficientNet-B4 regression + MSE training and the same thresholding approach, but I harden the pipeline so it always outputs a strictly valid integer class column (Python int-compatible) with no NaNs and exact row alignment to `test.csv`. I also clamp/clean continuous outputs in one place (before any kappa/threshold logic) to avoid edge-case values that can destabilize threshold optimization. These are minimal, score-relevant changes intended to move you upward from 0.0 toward the target without altering the model or training semantics.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is most consistent with Kaggle effectively treating the submission as invalid/unreadable (even if the CSV writes), so I make the smallest changes that harden “submission correctness” (exact column dtypes/order, exact row alignment, and no hidden NaNs) without changing your model, loss, or training loop. To also move the score upward toward the target (higher-is-better) with minimal semantic impact, I switch the sigmoid scaling from `*4.5` to `*4.0` (still clamped to [0,4]) to better match the label range and reduce saturation at the upper end. I additionally compute optimized thresholds on a **concat of train+valid predictions** after training (same model; just uses more in-fold data to calibrate discretization for QWK). Finally, I ensure the submission `diagnosis` column is plain integer (int64) and `id_code` is string, and I write using the sample_submission template to match Kaggle’s expected schema exactly.'
- What this solution (achieved 0.0) has done: 'Your 0.0 is most consistent with Kaggle not evaluating your predictions as intended (often caused by a subtle label-type mismatch in the metric pipeline: QWK expects discrete integer classes, but your validation metric and threshold optimizer are comparing against float targets). I make the smallest change that keeps your EfficientNet-B4 regression + MSE training intact: cast targets to `int` inside validation/threshold optimization and compute kappa on integer labels only, ensuring thresholds are optimized against the right objective. I also ensure training targets are float (for MSE) but validation/threshold targets are int (for QWK), and keep the submission alignment/dtype enforcement exactly as you already do. This should move your score upward from 0.0 toward the target without changing the model, loss, or training loop.'

# 9. Code solution

## === cell 0
import os
import random
import time
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as transforms
from PIL import Image

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score

import timm


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

BASE = "/kaggle/input/aptos2019-blindness-detection"
TRAIN_CSV = os.path.join(BASE, "train.csv")
TEST_CSV = os.path.join(BASE, "test.csv")
SAMPLE_SUB_CSV = os.path.join(BASE, "sample_submission.csv")
TRAIN_DIR = os.path.join(BASE, "train_images")
TEST_DIR = os.path.join(BASE, "test_images")

assert os.path.exists(TRAIN_CSV), f"Missing: {TRAIN_CSV}"
assert os.path.exists(TEST_CSV), f"Missing: {TEST_CSV}"
assert os.path.exists(SAMPLE_SUB_CSV), f"Missing: {SAMPLE_SUB_CSV}"
assert os.path.exists(TRAIN_DIR), f"Missing: {TRAIN_DIR}"
assert os.path.exists(TEST_DIR), f"Missing: {TEST_DIR}"


def sanitize_continuous_preds(x, lo=0.0, hi=4.0):
    x = np.asarray(x, dtype=np.float32)
    x = np.nan_to_num(x, nan=(lo + hi) / 2.0, posinf=hi, neginf=lo)
    return np.clip(x, lo, hi)




## === cell 1
def apply_thresholds(preds_continuous, thresholds):
    thresholds = np.asarray(thresholds, dtype=np.float32)
    thresholds = np.clip(np.sort(thresholds), 0.0, 4.0)

    preds_continuous = np.asarray(preds_continuous, dtype=np.float32)
    preds = np.zeros_like(preds_continuous, dtype=np.int64)
    for t in thresholds:
        preds += (preds_continuous >= t).astype(np.int64)
    return preds


DEFAULT_THRESHOLDS = np.array([0.7, 1.5, 2.5, 3.5], dtype=np.float32)




## === cell 2
def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, flatten=False):
        super().__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps
        self.flatten = flatten

    def forward(self, x):
        x = gem(x, p=self.p, eps=self.eps)
        if self.flatten:
            x = x.flatten(1)
        return x

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p.data.tolist()[0]:.4f}, eps={self.eps})"


class Model(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone = timm.models.tf_efficientnet_b4_ns(pretrained=pretrained)
        self.backbone.global_pool = GeM(flatten=True)
        self.regressor = nn.Linear(1000, 1)

    def forward(self, x):
        x = self.backbone(x)
        out = self.regressor(x)
        out = torch.sigmoid(out) * 4.0
        out = torch.clamp(out, 0.0, 4.0)
        return out




## === cell 3
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_tfms = transforms.Compose(
    [
        transforms.Resize((384, 384)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.01),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)

valid_tfms = transforms.Compose(
    [
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)


class AptosDataset(Dataset):
    def __init__(self, df, img_dir, transforms_=None, with_labels=True):
        self.df = df.reset_index(drop=True).copy()
        self.df["id_code"] = self.df["id_code"].astype(str)
        self.img_dir = img_dir
        self.transforms = transforms_
        self.with_labels = with_labels

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, f"{row['id_code']}.png")
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
        img = Image.open(img_path).convert("RGB")
        if self.transforms is not None:
            img = self.transforms(img)
        if self.with_labels:
            y = torch.tensor(row["diagnosis"], dtype=torch.float32)
            return img, y
        return img, row["id_code"]




## === cell 4
train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)
sample_sub = pd.read_csv(SAMPLE_SUB_CSV)

train_df["id_code"] = train_df["id_code"].astype(str)
test_df["id_code"] = test_df["id_code"].astype(str)
sample_sub["id_code"] = sample_sub["id_code"].astype(str)

train_df["diagnosis"] = train_df["diagnosis"].astype(np.int64)

print(train_df.shape, test_df.shape)
print(train_df.head())

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold = 0
tr_idx, va_idx = next(iter(skf.split(train_df["id_code"], train_df["diagnosis"])))

tr_df = train_df.iloc[tr_idx].reset_index(drop=True)
va_df = train_df.iloc[va_idx].reset_index(drop=True)

train_ds = AptosDataset(tr_df, TRAIN_DIR, transforms_=train_tfms, with_labels=True)
valid_ds = AptosDataset(va_df, TRAIN_DIR, transforms_=valid_tfms, with_labels=True)

train_loader = DataLoader(
    train_ds, batch_size=8, shuffle=True, num_workers=2, pin_memory=True
)
valid_loader = DataLoader(
    valid_ds, batch_size=8, shuffle=False, num_workers=2, pin_memory=True
)




## === cell 5
model = Model(pretrained=True).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)


def run_valid(model, loader):
    model.eval()
    preds = []
    targs = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            out = model(x).squeeze(1)
            preds.append(out.detach().cpu().numpy())
            targs.append(y.detach().cpu().numpy())
    preds = np.concatenate(preds)
    targs = np.concatenate(targs)
    preds = sanitize_continuous_preds(preds, 0.0, 4.0)
    targs_int = np.clip(np.rint(targs), 0, 4).astype(np.int64)
    return preds, targs_int


EPOCHS = 2

start = time.time()
for epoch in range(EPOCHS):
    model.train()
    running = 0.0
    n = 0
    for x, y in train_loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        out = model(x).squeeze(1)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        running += loss.item() * x.size(0)
        n += x.size(0)

    va_preds_cont, va_y_int = run_valid(model, valid_loader)
    va_preds_round = np.clip(np.rint(va_preds_cont), 0, 4).astype(np.int64)
    kappa_round = cohen_kappa_score(va_y_int, va_preds_round, weights="quadratic")

    print(
        f"epoch {epoch+1}/{EPOCHS} train_loss={running/n:.4f} val_kappa_round={kappa_round:.4f} time={(time.time()-start):.1f}s"
    )




## === cell 6
def optimize_thresholds(
    y_true_int, y_pred_cont, init_thresholds=DEFAULT_THRESHOLDS, n_iters=30
):
    y_true_int = np.asarray(y_true_int, dtype=np.int64)
    y_pred_cont = sanitize_continuous_preds(y_pred_cont, 0.0, 4.0)

    thr = np.asarray(init_thresholds, dtype=np.float32).copy()
    thr = np.clip(np.sort(thr), 0.0, 4.0)

    def score(thresholds):
        thresholds = np.clip(
            np.sort(np.asarray(thresholds, dtype=np.float32)), 0.0, 4.0
        )
        pred_cls = apply_thresholds(y_pred_cont, thresholds)
        pred_cls = np.clip(pred_cls, 0, 4).astype(np.int64)
        return cohen_kappa_score(y_true_int, pred_cls, weights="quadratic")

    best = score(thr)
    for _ in range(n_iters):
        improved = False
        for i in range(4):
            for delta in (-0.2, -0.1, -0.05, 0.05, 0.1, 0.2):
                cand = thr.copy()
                cand[i] = cand[i] + delta
                cand = np.sort(cand)  # ensure monotonic
                cand = np.clip(cand, 0.0, 4.0)
                s = score(cand)
                if s > best:
                    best = s
                    thr = cand
                    improved = True
        if not improved:
            break
    return thr, best


tr_valid_ds_for_thr = AptosDataset(
    tr_df, TRAIN_DIR, transforms_=valid_tfms, with_labels=True
)
tr_valid_loader_for_thr = DataLoader(
    tr_valid_ds_for_thr, batch_size=8, shuffle=False, num_workers=2, pin_memory=True
)

tr_preds_cont, tr_y_int = run_valid(model, tr_valid_loader_for_thr)
va_preds_cont, va_y_int = run_valid(model, valid_loader)

all_preds_cont = np.concatenate([tr_preds_cont, va_preds_cont], axis=0)
all_y_int = np.concatenate([tr_y_int, va_y_int], axis=0)

best_thr, best_kappa = optimize_thresholds(
    all_y_int, all_preds_cont, DEFAULT_THRESHOLDS, n_iters=25
)

print("default thresholds:", DEFAULT_THRESHOLDS)
print("optimized thresholds:", best_thr)
print("in-fold kappa_thr:", best_kappa)




## === cell 7
test_ds = AptosDataset(test_df, TEST_DIR, transforms_=valid_tfms, with_labels=False)
test_loader = DataLoader(
    test_ds, batch_size=8, shuffle=False, num_workers=2, pin_memory=True
)

model.eval()
all_ids = []
all_preds_cont = []
with torch.no_grad():
    for x, ids in test_loader:
        x = x.to(device, non_blocking=True)
        out = model(x).squeeze(1).detach().cpu().numpy()
        all_preds_cont.append(out)
        all_ids.extend([str(i) for i in ids])

all_preds_cont = np.concatenate(all_preds_cont).astype(np.float32)
all_preds_cont = sanitize_continuous_preds(all_preds_cont, 0.0, 4.0)

all_preds_cls = apply_thresholds(all_preds_cont, best_thr)
all_preds_cls = np.clip(all_preds_cls, 0, 4).astype(np.int64)

assert len(all_ids) == len(test_df), "Prediction count mismatch with test.csv"
if not np.array_equal(
    np.array(all_ids, dtype=str), test_df["id_code"].astype(str).values
):
    raise AssertionError(
        "id_code order mismatch between loader outputs and test.csv; refusing to write a potentially misaligned submission."
    )

sub = sample_sub.copy()
sub["id_code"] = sub["id_code"].astype(str)

if not np.array_equal(sub["id_code"].values, test_df["id_code"].values):
    raise AssertionError(
        "sample_submission id_code order does not match test.csv in this environment."
    )

sub["diagnosis"] = np.asarray(all_preds_cls, dtype=np.int64)

assert sub["diagnosis"].notna().all(), "diagnosis contains NaNs"
assert (
    pd.Series(sub["diagnosis"]).between(0, 4).all()
), "Predictions must be within [0,4]"
assert list(sub.columns) == [
    "id_code",
    "diagnosis",
], "Submission columns must be id_code, diagnosis"
assert sub.shape[0] == test_df.shape[0], "Row count mismatch with test.csv"

sub_path = "submission.csv"
sub.to_csv(sub_path, index=False)
print("Wrote:", sub_path, "shape:", sub.shape)
print(sub.head())
print("diagnosis dtype:", sub["diagnosis"].dtype)
print("diagnosis value counts:\n", sub["diagnosis"].value_counts().sort_index())
