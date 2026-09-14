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

0.918523708507293

# 6. Current score

0.60716

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5791) has done: 'I fix the immediate runtime blockers so the notebook runs end-to-end: (1) remove the dependency on a missing `../input/weights/...` checkpoint by falling back to training the same model on `train.csv` when no weights are found, and (2) make the code device-safe by using CUDA only if available (your current run is CPU-only). I also fix a few small but critical logic bugs that can break transforms/inference (`is` vs `==` for string comparison, `trim()` returning `None`, and `.cuda()` usage inside helper functions) without changing the model architecture or prediction semantics. Finally, I switch inference to batch DataLoader (same predictions, far faster) so it completes within the 600s limit and reliably writes a non-empty `submission.csv` with correct columns.'
- What this solution (achieved 0.59158) has done: 'Your current score is far below the target, so we should cautiously improve it without changing the model architecture or training loop design. The biggest score lever here (for QWK) is converting the regressor output to 0–4 classes: fixed thresholds `[0.75,1.5,2.5,3.5]` are rarely optimal, so we fit thresholds on a small validation split using QWK and then apply them to test predictions. This keeps the same model and same regressor head, but aligns post-processing to the metric and typically yields a large jump in kappa. We also make the fallback training deterministic and add a lightweight validation QWK print so you can confirm the improvement locally before submitting.'
- What this solution (achieved 0.59104) has done: 'Your current score is far below the target, so we should improve it with the smallest change that directly aligns predictions to the QWK metric. The main lever is the threshold tuning: your current coordinate-grid search is very coarse, so we keep the same regressor head and thresholding approach but replace the tuner with a lightweight discrete “coordinate descent + ternary refinement” that finds better thresholds on the same validation split (still deterministic and fast). We also ensure validation/test predictions use the same image directory logic (train vs test) without changing transforms or the model. This should move QWK upward without altering the model architecture, training loop style, or loss.'
- What this solution (achieved 0.59607) has done: 'Your current score is far below the target, so we should improve QWK while keeping the exact model/training core intact. The biggest safe lever is better threshold calibration: instead of tuning thresholds on a single small split (high variance), we tune them out-of-fold across several stratified folds and then average the thresholds, which usually generalizes better to the leaderboard without changing architecture or loss. We also ensure the “validation” predictions used for tuning are truly out-of-fold (no training leakage), and keep inference identical aside from using these more robust thresholds. All paths and submission formatting are preserved, and the script still runs within the time budget.'
- What this solution (achieved 0.57679) has done: 'The timeout is dominated by repeated full-image inference inside the 4-fold OOF threshold tuning (reading/transforming PNGs and running EfficientNet on ~3.3k images four times), plus per-epoch fallback training in each fold when no checkpoint exists. To preserve identical logic/semantics, the main speedup is to compute regression predictions for the entire training set once (with the same loaded model), then reuse those cached predictions to build OOF arrays for threshold tuning—this yields exactly the same OOF values because the model is fixed and not fold-dependent when a checkpoint is present. Additionally, DataLoader settings are tuned for throughput (more workers, persistent workers, prefetch, pinned memory) and we avoid pandas `.iloc` per sample by pre-materializing id/label arrays, cutting Python overhead without changing transforms or model outputs. No approximations, sampling, or training/architecture changes are introduced.'
- What this solution (achieved 0.57679) has done: 'Your current score (0.57679) is far below the target (0.9185), so we should push QWK upward with the smallest change that directly matches the metric while preserving your exact model/training logic. The biggest low-risk lever is fixing the post-processing alignment: your model’s regressor outputs are in the 0–4.5 range (via `*4.5`), but threshold tuning/clipping currently assumes a 0–4 range, which makes both tuning and final discretization suboptimal. I minimally update the threshold tuner/search space and monotone projection to operate in the correct 0–4.5 range (and allow the last threshold to go up to 4.5), keeping the same tuning algorithm, same model, same inference head. This change should improve the discretization for QWK and move the score toward the target without altering architecture or training loops.'
- What this solution (achieved 0.60716) has done: 'We keep your model, transforms, and training logic unchanged, and focus on improving the QWK-aligned post-processing with the smallest safe change. The main issue is that thresholds are tuned against potentially out-of-order/duplicate-projected candidates, and the tuner can get “stuck” because it only tests a few fixed offsets; we minimally strengthen it by adding a tiny local 1D search per threshold (still deterministic, still fast) while keeping the same coordinate-descent structure and the same 0–4.5 regression range. We also ensure the default starting thresholds are in the same 0–4.5 scale you trained for (simple rescale from the old 0–4 defaults), which is a direct metric-alignment fix and does not change model outputs. Finally, we keep submission formatting and row alignment identical, just producing better-calibrated discrete predictions.'
- What this solution (achieved 0.60716) has done: 'We keep your exact model, transforms, and training/inference heads unchanged, and only adjust the QWK-aligned post-processing to better match the target metric. The main minimal change is to tune thresholds on out-of-fold predictions using a small continuous optimizer (Nelder–Mead) with a monotonic re-parameterization so thresholds are always ordered; this typically improves kappa without touching the network. We also fix a subtle issue where `predict_regression()` casts labels to `int64` (can bias QWK tuning when labels are floats) by explicitly rounding/clipping labels to valid classes before scoring. Everything still runs end-to-end within the time budget and writes a valid `submission.csv` with correct ordering/columns.'

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
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as transforms
from torchvision.transforms import functional as FT
from PIL import Image, ImageChops

from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
import timm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)



## === cell 1
REG_MAX = 4.5
threshold = [
    0.75 * (REG_MAX / 4.0),
    1.5 * (REG_MAX / 4.0),
    2.5 * (REG_MAX / 4.0),
    3.5 * (REG_MAX / 4.0),
]


def regress2class(out):
    prediction = torch.zeros(out.size(0), device=out.device)
    for i in range(4):
        prediction += (out.data >= threshold[i]).squeeze()
    return prediction.detach().cpu()


def regress2class_with_thresholds(out_1d: torch.Tensor, thr):
    thr_t = torch.as_tensor(thr, device=out_1d.device, dtype=out_1d.dtype).view(1, -1)
    pred = (out_1d.view(-1, 1) >= thr_t).sum(dim=1)
    return pred.detach().cpu().numpy().astype(np.int64)


def ordinal2class_prob(out):
    pred_prob = torch.zeros(out.size(0), 5, device=out.device)
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
            l1 = int(math.floor(float(out[i])))
            l2 = int(math.ceil(float(out[i])))
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
TRAIN_CSV = os.path.join(BASE, "train.csv")
TEST_CSV = os.path.join(BASE, "test.csv")
TRAIN_IMG_DIR = os.path.join(BASE, "train_images")
TEST_IMG_DIR = os.path.join(BASE, "test_images")

test_df = pd.read_csv(TEST_CSV)
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

CKPT_CANDIDATES = [
    "../input/weights/B4_3stage_13epoch_320.pkl",
    "../input/weights/B4_3stage_13epoch_320.pth",
    "../input/weights/B4_3stage_13epoch_320.pt",
]
ckpt_path = next((p for p in CKPT_CANDIDATES if os.path.exists(p)), None)

net = ThreeStage_Model().to(device)

if ckpt_path is not None:
    state = torch.load(ckpt_path, map_location="cpu")
    if (
        isinstance(state, dict)
        and "state_dict" in state
        and isinstance(state["state_dict"], dict)
    ):
        state = state["state_dict"]
    if isinstance(state, dict):
        new_state = {}
        for k, v in state.items():
            new_state[k.replace("module.", "")] = v
        state = new_state
    net.load_state_dict(state, strict=True)
net.eval()




## === cell 5
class RetinopathyDataset(Dataset):
    def __init__(self, df, img_dir, transform):
        df = df.reset_index(drop=True)
        self.ids = df["id_code"].to_numpy()
        self.labels = df["diagnosis"].to_numpy(dtype=np.float32)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return self.ids.shape[0]

    def __getitem__(self, i):
        img_path = os.path.join(self.img_dir, f"{self.ids[i]}.png")
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        y = torch.tensor(self.labels[i], dtype=torch.float32)
        return img, y


train_df_full = pd.read_csv(TRAIN_CSV)

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
tr_idx, va_idx = next(sss.split(train_df_full["id_code"], train_df_full["diagnosis"]))
train_df = train_df_full.iloc[tr_idx].reset_index(drop=True)
val_df = train_df_full.iloc[va_idx].reset_index(drop=True)


def _loader_kwargs(batch_size, shuffle):
    num_workers = min(8, os.cpu_count() or 2)
    pin = device.type == "cuda"
    return dict(
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )


def train_fallback_if_needed():
    global net
    if ckpt_path is not None:
        return

    ds = RetinopathyDataset(train_df, TRAIN_IMG_DIR, transform)
    loader = DataLoader(ds, **_loader_kwargs(batch_size=8, shuffle=True))

    net.train()
    optimizer = optim.Adam(net.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    start = time.time()
    max_seconds = 420  # keep within 600s total notebook budget
    for epoch in range(1):  # keep core approach identical (simple 1-epoch fallback)
        for xb, yb in loader:
            if time.time() - start > max_seconds:
                break
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True).view(-1, 1)

            _, r_out, _ = net(xb)
            loss = criterion(r_out, yb)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        if time.time() - start > max_seconds:
            break

    net.eval()


train_fallback_if_needed()


def _labels_to_int(y_like):
    y = np.asarray(y_like)
    y = np.rint(y).astype(np.int64, copy=False)
    return np.clip(y, 0, 4)


@torch.no_grad()
def predict_regression(df_in, img_dir, batch_size=16):
    ds = RetinopathyDataset(df_in, img_dir, transform)
    loader = DataLoader(ds, **_loader_kwargs(batch_size=batch_size, shuffle=False))
    preds = np.empty(len(ds), dtype=np.float32)
    ys = np.empty(len(ds), dtype=np.float32)
    k = 0
    for xb, yb in loader:
        b = xb.size(0)
        xb = xb.to(device, non_blocking=True)
        _, r_out, _ = net(xb)
        preds[k : k + b] = (
            r_out.squeeze(1).detach().cpu().numpy().astype(np.float32, copy=False)
        )
        ys[k : k + b] = yb.detach().cpu().numpy().astype(np.float32, copy=False)
        k += b
    return preds, ys


def qwk_from_thresholds(y_true, y_pred_reg, thr):
    thr = np.asarray(thr, dtype=np.float64)
    y_true_i = _labels_to_int(y_true)

    y_pred_reg = np.asarray(y_pred_reg, dtype=np.float64)
    y_pred_cls = np.zeros_like(y_true_i, dtype=np.int64)
    y_pred_cls += (y_pred_reg >= thr[0]).astype(np.int64)
    y_pred_cls += (y_pred_reg >= thr[1]).astype(np.int64)
    y_pred_cls += (y_pred_reg >= thr[2]).astype(np.int64)
    y_pred_cls += (y_pred_reg >= thr[3]).astype(np.int64)
    return cohen_kappa_score(y_true_i, y_pred_cls, weights="quadratic")


def _project_monotone(thr, eps=1e-3, lo=0.0, hi=REG_MAX):
    thr = np.asarray(thr, dtype=np.float64).copy()
    thr[0] = np.clip(thr[0], lo, hi)
    for i in range(1, 4):
        thr[i] = max(thr[i], thr[i - 1] + eps)
        thr[i] = min(thr[i], hi)
    return thr


def tune_thresholds_refined(
    y_true, y_pred_reg, init_thr, n_passes=3, init_step=0.25, min_step=0.01
):
    thr = _project_monotone(init_thr)
    best = qwk_from_thresholds(y_true, y_pred_reg, thr)

    step = float(init_step)
    for _ in range(n_passes):
        improved = True
        while improved:
            improved = False
            for i in range(4):
                lo = 0.0 if i == 0 else thr[i - 1] + 1e-3
                hi = REG_MAX if i == 3 else thr[i + 1] - 1e-3
                if hi <= lo:
                    continue

                grid = np.linspace(-2.0, 2.0, 17, dtype=np.float64) * step
                candidates = thr[i] + grid
                candidates = np.clip(candidates, lo, hi)
                candidates = np.unique(candidates)

                local_best = best
                local_t = thr[i]
                for t in candidates:
                    thr_try = thr.copy()
                    thr_try[i] = t
                    thr_try = _project_monotone(thr_try)
                    score = qwk_from_thresholds(y_true, y_pred_reg, thr_try)
                    if score > local_best:
                        local_best = score
                        local_t = float(t)

                if local_best > best:
                    thr[i] = local_t
                    best = local_best
                    improved = True

            if not improved:
                break

        step = max(step / 2.0, min_step)

    return thr.tolist(), float(best)


@torch.no_grad()
def _predict_regression_with_model(model, df_in, img_dir, batch_size=16):
    ds = RetinopathyDataset(df_in, img_dir, transform)
    loader = DataLoader(ds, **_loader_kwargs(batch_size=batch_size, shuffle=False))
    preds = np.empty(len(ds), dtype=np.float32)
    ys = np.empty(len(ds), dtype=np.float32)
    k = 0
    for xb, yb in loader:
        b = xb.size(0)
        xb = xb.to(device, non_blocking=True)
        _, r_out, _ = model(xb)
        preds[k : k + b] = (
            r_out.squeeze(1).detach().cpu().numpy().astype(np.float32, copy=False)
        )
        ys[k : k + b] = yb.detach().cpu().numpy().astype(np.float32, copy=False)
        k += b
    return preds, ys


def tune_thresholds_from_true_oof(
    df_full,
    init_thr,
    n_splits=4,
    seed=42,
    batch_size=16,
):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    oof_y = df_full["diagnosis"].to_numpy(dtype=np.float32)
    oof_pred = np.zeros(len(df_full), dtype=np.float32)

    if ckpt_path is not None:
        df_full_for_ds = df_full[["id_code", "diagnosis"]].reset_index(drop=True)
        full_pred_reg, _ = _predict_regression_with_model(
            net, df_full_for_ds, TRAIN_IMG_DIR, batch_size=batch_size
        )
        full_pred_reg = full_pred_reg.astype(np.float32, copy=False)

        for fold, (_, va_i) in enumerate(
            skf.split(df_full["id_code"], df_full["diagnosis"]), 1
        ):
            oof_pred[va_i] = full_pred_reg[va_i]
            fold_qwk_default = qwk_from_thresholds(
                oof_y[va_i], oof_pred[va_i], init_thr
            )
            print(
                f"Fold {fold}/{n_splits} default-thr QWK (OOF preds):",
                float(fold_qwk_default),
            )
    else:
        def _fresh_net_like_loaded():
            m = ThreeStage_Model().to(device)
            m.eval()
            return m

        def _train_one_epoch_fallback(model, df_train, max_seconds=105, batch_size=8):
            ds = RetinopathyDataset(df_train, TRAIN_IMG_DIR, transform)
            loader = DataLoader(
                ds, **_loader_kwargs(batch_size=batch_size, shuffle=True)
            )
            model.train()
            optimizer = optim.Adam(model.parameters(), lr=1e-4)
            criterion = nn.MSELoss()

            start = time.time()
            for xb, yb in loader:
                if time.time() - start > max_seconds:
                    break
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True).view(-1, 1)
                _, r_out, _ = model(xb)
                loss = criterion(r_out, yb)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            model.eval()
            return model

        for fold, (tr_i, va_i) in enumerate(
            skf.split(df_full["id_code"], df_full["diagnosis"]), 1
        ):
            df_tr = df_full.iloc[tr_i].reset_index(drop=True)
            df_va = df_full.iloc[va_i].reset_index(drop=True)

            fold_model = _fresh_net_like_loaded()
            fold_model = _train_one_epoch_fallback(
                fold_model, df_tr, max_seconds=105, batch_size=8
            )

            va_pred_reg, _ = _predict_regression_with_model(
                fold_model, df_va, TRAIN_IMG_DIR, batch_size=batch_size
            )
            oof_pred[va_i] = va_pred_reg.astype(np.float32, copy=False)

            fold_qwk_default = qwk_from_thresholds(
                oof_y[va_i], oof_pred[va_i], init_thr
            )
            print(
                f"Fold {fold}/{n_splits} default-thr QWK (OOF preds):",
                float(fold_qwk_default),
            )

            del fold_model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    def _thr_from_u(u):
        u = np.asarray(u, dtype=np.float64).ravel()
        u = np.clip(u, -6.0, 6.0)
        base = 1.0 / (1.0 + np.exp(-u[0])) * REG_MAX
        gaps = np.exp(u[1:5]) * 1e-2  # positive, small base gap
        thr = np.cumsum(np.concatenate([[base], gaps]))
        thr = _project_monotone(thr, eps=1e-3, lo=0.0, hi=REG_MAX)
        return thr

    def _u_from_thr(thr):
        thr = _project_monotone(thr)
        base = float(thr[0])
        gaps = np.diff(thr)
        base01 = np.clip(base / REG_MAX, 1e-6, 1 - 1e-6)
        u0 = np.log(base01 / (1.0 - base01))
        gaps = np.clip(gaps, 1e-6, None)
        u_rest = np.log(gaps / 1e-2)
        u = np.concatenate([[u0], u_rest])
        if u.shape[0] != 5:
            u = np.pad(u, (0, 5 - u.shape[0]), mode="constant")
        return u

    thr0, q0 = tune_thresholds_refined(
        _labels_to_int(oof_y),
        oof_pred,
        init_thr,
        n_passes=3,
        init_step=0.25,
        min_step=0.01,
    )

    def _objective(u):
        thr = _thr_from_u(u)
        return -qwk_from_thresholds(oof_y, oof_pred, thr)

    u = _u_from_thr(thr0)
    best_u = u.copy()
    best_obj = _objective(best_u)

    step = 0.6
    start = time.time()
    time_budget = 20.0  # keep minimal runtime increase
    for _ in range(60):
        if time.time() - start > time_budget:
            break
        improved = False
        for j in range(best_u.shape[0]):
            for sgn in (-1.0, 1.0):
                cand = best_u.copy()
                cand[j] = cand[j] + sgn * step
                obj = _objective(cand)
                if obj < best_obj:
                    best_obj = obj
                    best_u = cand
                    improved = True
        if not improved:
            step *= 0.65
            if step < 0.03:
                break

    thr_best = _thr_from_u(best_u).tolist()
    qwk_best = -best_obj
    return thr_best, float(qwk_best)


val_pred_reg, val_y = predict_regression(val_df, TRAIN_IMG_DIR, batch_size=16)
tuned_thr_single, tuned_qwk_single = tune_thresholds_refined(
    _labels_to_int(val_y),
    val_pred_reg,
    threshold,
    n_passes=3,
    init_step=0.25,
    min_step=0.01,
)
base_qwk = qwk_from_thresholds(_labels_to_int(val_y), val_pred_reg, threshold)

tuned_thr, tuned_qwk_oof = tune_thresholds_from_true_oof(
    train_df_full, threshold, n_splits=4, seed=42, batch_size=16
)

print("Validation QWK with default thresholds (single split):", float(base_qwk))
print("Validation QWK with tuned thresholds (single split):", float(tuned_qwk_single))
print("Using thresholds (true OOF-tuned):", tuned_thr)
print("QWK on full true-OOF predictions (tuned):", float(tuned_qwk_oof))




## === cell 6
class TestDataset(Dataset):
    def __init__(self, ids, img_dir, transform):
        self.ids = list(ids)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        idx = self.ids[i]
        img_path = os.path.join(self.img_dir, f"{idx}.png")
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return idx, img


test_ds = TestDataset(test_ids, TEST_IMG_DIR, transform)
test_loader = DataLoader(test_ds, **_loader_kwargs(batch_size=16, shuffle=False))

submission_rows = []
with torch.no_grad():
    for ids, xb in test_loader:
        xb = xb.to(device, non_blocking=True)
        _, r_out, _ = net(xb)  # keep original inference head
        pred_cls = regress2class_with_thresholds(r_out.squeeze(1), tuned_thr)
        for i in range(len(ids)):
            submission_rows.append([ids[i], int(pred_cls[i])])



## === cell 7
df = pd.DataFrame(submission_rows, columns=["id_code", "diagnosis"])
df = df.set_index("id_code").loc[test_df["id_code"]].reset_index()

df.to_csv("submission.csv", index=False)

print(df.head())
print("Wrote submission.csv with shape:", df.shape)
print("Device:", device)
print("ckpt_path:", ckpt_path)
print("REG_MAX used for threshold tuning:", REG_MAX)
print("Initial threshold (scaled):", threshold)
print("Final tuned_thr:", tuned_thr)
