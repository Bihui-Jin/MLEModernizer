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

0.9004503090121132

# 6. Current score

0.69778

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the missing weights/runtime issues so the notebook runs end-to-end and always writes a non-empty `submission.csv`. Specifically, I (1) remove the failing `pip install` and add robust path checks for the weights file, (2) make device selection safe by falling back to CPU when CUDA isn’t available, and (3) ensure image loading/trim never returns `None` and predictions are clamped to valid classes 0–4. If the external weight file is absent, the code still generate a valid submission using the untrained model (score be low, but it run and produce a correctly formatted CSV).'
- What this solution (achieved 0.69778) has done: 'Your current 0.0 score is because the model is effectively untrained (the weights file isn’t present), so predictions are near-constant and kappa collapses. To move the score upward with minimal core-logic change, I keep your exact model/forward pass and inference loop, but (1) add a small training step on the provided `train.csv` images to learn usable weights, and (2) tune the four class thresholds on a held-out validation split specifically to maximize quadratic weighted kappa (the competition metric) while keeping the same “regression then threshold to classes” semantics. I also fix one bug where your `regress2class()` used `.data` and ignored device/shape, and I ensure the submission ordering matches `test.csv` exactly. This should reliably move the score toward your target without changing the model architecture or loss type (still regression with MSE).'

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
from PIL import Image, ImageChops

import timm
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import cohen_kappa_score

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)


def trim(im: Image.Image) -> Image.Image:
    """
    Ensure trim always returns an image (never None), otherwise transforms(...) can crash.
    """
    bg = Image.new(im.mode, im.size, im.getpixel((0, 0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -10)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)
    return im




## === cell 1
threshold = [0.7, 1.5, 2.5, 3.5]


def regress2class(out_1d: torch.Tensor, thr=None) -> torch.Tensor:
    """
    Map regression output to ordinal classes via thresholds.
    Fix: remove .data usage and ensure correct dtype/device behavior.
    """
    if thr is None:
        thr = threshold
    pred = torch.zeros_like(out_1d, dtype=torch.int64)
    for t in thr:
        pred += (out_1d >= t).to(torch.int64)
    return pred




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




## === cell 3
BASE = "../input/aptos2019-blindness-detection"
TRAIN_CSV = os.path.join(BASE, "train.csv")
TEST_CSV = os.path.join(BASE, "test.csv")
TRAIN_IMG_DIR = os.path.join(BASE, "train_images")
TEST_IMG_DIR = os.path.join(BASE, "test_images")

train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)

input_size = 384

train_tfms = transforms.Compose(
    [
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.384, 0.258, 0.174], std=[0.124, 0.089, 0.094]),
    ]
)
test_tfms = train_tfms


class APTOSDataset(Dataset):
    def __init__(self, df, img_dir, tfms, with_label: bool):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.tfms = tfms
        self.with_label = with_label

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img_path = os.path.join(self.img_dir, f"{row['id_code']}.png")
        img = Image.open(img_path).convert("RGB")
        img = trim(img)
        x = self.tfms(img)
        if self.with_label:
            y = torch.tensor(float(row["diagnosis"]), dtype=torch.float32)
            return x, y
        return x, row["id_code"]


splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
tr_idx, va_idx = next(splitter.split(train_df, train_df["diagnosis"].values))
tr_df = train_df.iloc[tr_idx].copy()
va_df = train_df.iloc[va_idx].copy()

train_loader = DataLoader(
    APTOSDataset(tr_df, TRAIN_IMG_DIR, train_tfms, with_label=True),
    batch_size=8,
    shuffle=True,
    num_workers=2,
    pin_memory=(device == "cuda"),
)
val_loader = DataLoader(
    APTOSDataset(va_df, TRAIN_IMG_DIR, train_tfms, with_label=True),
    batch_size=8,
    shuffle=False,
    num_workers=2,
    pin_memory=(device == "cuda"),
)



## === cell 4
net = Regressor().to(device)

WEIGHTS_CANDIDATES = [
    "../input/weights/D5_regre_70epoch.pkl",
    "/kaggle/input/weights/D5_regre_70epoch.pkl",
]
weights_path = next((p for p in WEIGHTS_CANDIDATES if os.path.exists(p)), None)

if weights_path is not None:
    print("Loading weights:", weights_path)
    state = torch.load(weights_path, map_location="cpu")
    net.load_state_dict(state, strict=True)
    trained_now = False
else:
    print(
        "Weights not found; training model on provided train set to obtain non-trivial predictions."
    )
    trained_now = True




## === cell 5
def evaluate_regression(model, loader):
    model.eval()
    ys = []
    outs = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            out = model(xb).squeeze(1)
            ys.append(yb.detach().cpu().numpy())
            outs.append(out.detach().cpu().numpy())
    y = np.concatenate(ys, axis=0)
    o = np.concatenate(outs, axis=0)
    return y, o


def qwk_from_thresholds(y_true_int, y_pred_reg, thr):
    pred_cls = np.zeros_like(y_pred_reg, dtype=np.int64)
    for t in thr:
        pred_cls += (y_pred_reg >= t).astype(np.int64)
    pred_cls = np.clip(pred_cls, 0, 4)
    return cohen_kappa_score(y_true_int, pred_cls, weights="quadratic")


def tune_thresholds(y_true_int, y_pred_reg, init_thr):
    """
    Minimal threshold tuning (coordinate ascent) to optimize the actual metric (QWK).
    Keeps the same regression->threshold->class semantics; only calibrates boundaries.
    """
    thr = np.array(init_thr, dtype=np.float64)
    best = qwk_from_thresholds(y_true_int, y_pred_reg, thr)

    for _ in range(5):
        improved = False
        for k in range(4):
            candidates = np.linspace(max(0.0, thr[k] - 0.6), min(4.5, thr[k] + 0.6), 25)
            local_best_thr = thr[k]
            local_best = best
            for c in candidates:
                thr_try = thr.copy()
                thr_try[k] = c
                thr_try = np.sort(np.clip(thr_try, 0.0, 4.5))
                score = qwk_from_thresholds(y_true_int, y_pred_reg, thr_try)
                if score > local_best:
                    local_best = score
                    local_best_thr = c
            if local_best > best + 1e-6:
                thr[k] = local_best_thr
                thr = np.sort(np.clip(thr, 0.0, 4.5))
                best = local_best
                improved = True
        if not improved:
            break
    return thr.tolist(), best




## === cell 6
if trained_now:
    net.train()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    EPOCHS = 2
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        running = 0.0
        n = 0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            out = net(xb).squeeze(1)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

            running += loss.item() * xb.size(0)
            n += xb.size(0)

        train_loss = running / max(1, n)
        y_val, o_val = evaluate_regression(net, val_loader)
        qwk_round = cohen_kappa_score(
            y_val.astype(int),
            np.clip(np.rint(o_val), 0, 4).astype(int),
            weights="quadratic",
        )
        print(
            f"Epoch {epoch}/{EPOCHS} - train_loss={train_loss:.4f} - val_qwk(round)={qwk_round:.4f} - time={time.time()-t0:.1f}s"
        )



## === cell 7
y_val, o_val = evaluate_regression(net, val_loader)
best_thr, best_qwk = tune_thresholds(y_val.astype(int), o_val, threshold)
threshold = best_thr  # update global threshold used in regress2class
print("Tuned thresholds:", threshold)
print("Val QWK after tuning:", best_qwk)

net.eval()



## === cell 8
test_ids = np.squeeze(test_df["id_code"].values)
test_loader = DataLoader(
    APTOSDataset(test_df, TEST_IMG_DIR, test_tfms, with_label=False),
    batch_size=8,
    shuffle=False,
    num_workers=2,
    pin_memory=(device == "cuda"),
)

rows = []
with torch.no_grad():
    for xb, id_codes in test_loader:
        xb = xb.to(device, non_blocking=True)
        out = net(xb).squeeze(1)  # [B]
        pred = regress2class(out, thr=threshold).detach().cpu().numpy().astype(int)
        for i, idc in enumerate(id_codes):
            rows.append([str(idc), int(np.clip(pred[i], 0, 4))])

sub_df = pd.DataFrame(rows, columns=["id_code", "diagnosis"])

sub_df = (
    sub_df.set_index("id_code").loc[test_df["id_code"].astype(str).values].reset_index()
)
assert len(sub_df) == len(test_df) and sub_df["diagnosis"].notna().all()

out_path = "submission.csv"
sub_df.to_csv(out_path, index=False)
print("Wrote", out_path, "with shape", sub_df.shape)
print(sub_df.head())
