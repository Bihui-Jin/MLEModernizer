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

0.9103041346388836

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.02258) has done: 'I fix the missing weights issue by loading the model checkpoint only if it exists; otherwise, the script fall back to using the defined `Model` with ImageNet pretrained backbone weights so it can run end-to-end. I also fix runtime bugs (`Parameter` not imported, hard-coded `cuda:0` when GPU may be unavailable, and unsafe `.data` usage / scalar conversion) and ensure `test_ids` is read as a list of strings. Finally, I make the image path resolution robust to either `../input/...` or the provided `/kaggle/data/...` layout, and guarantee a valid `submission.csv` with the exact required columns is written.'
- What this solution (achieved -0.06179) has done: 'The main runtime error is a feature-dimension mismatch: `timm`’s `tf_efficientnet_b4_ns` returns 1000-d logits by default, but your `regressor` expects 1792 features. I fix this by creating the backbone with `num_classes=0` so it outputs pooled features, keeping your GeM pooling and regressor logic intact. I also make inference robust so a single unreadable image can’t abort the loop (which caused the empty submission), and I guarantee the written CSV is non-empty and aligned to `test.csv`’s `id_code` order. These changes are correctness/stability focused and should also improve score versus the accidental “linear on logits” behavior.'
- What this solution (achieved 0.10929) has done: 'Your current score is far below the target, so the most likely issue is not the backbone anymore but the *post-processing to discrete classes* for QWK. I keep your model exactly as-is and only (1) ensure EfficientNet runs in inference mode deterministically, (2) collect the raw regression outputs on the test set, and (3) replace the fixed hard-coded thresholds with *distribution-calibrated thresholds* derived from the training label distribution (a common minimal fix for this competition that usually improves QWK). This preserves your “regression -> thresholds -> class” core logic, but aligns the predicted class frequencies to the dataset’s expected ordinal distribution, which typically moves QWK up substantially. The script still run end-to-end and write a valid `submission.csv` with the required columns in `test.csv` order.'
- What this solution (achieved 0.03801) has done: 'Your score is far below the target, so we should improve toward it without changing the model/training core logic (there is no training here). The biggest lever consistent with your current “regression → thresholds → class” pipeline is to compute thresholds in a way that better matches the QWK/ordinal nature: instead of using test-prediction quantiles (which can be unstable), we fit thresholds on the training set by minimizing negative quadratic weighted kappa (classic APTOS fix) using the model’s own train predictions. This keeps the architecture, preprocessing, and regression head identical, but makes the discretization step metric-aligned and usually yields a large QWK gain. We also make inference more efficient/stable by batching and ensuring strict `id_code` order in the written submission.'
- What this solution (achieved 0.21984) has done: 'Your current score is far below the target, so the most likely remaining issue is that the “fit thresholds on train predictions” step is too weak/unstable, leaving your discrete class mapping poorly aligned with QWK. I keep your exact model and inference logic, but (1) make threshold fitting more robust by optimizing thresholds with a simple coordinate-descent over QWK using a better initialization (train label distribution + raw prediction quantiles) and more iterations, and (2) compute thresholds on out-of-fold (OOF) train predictions via a minimal K-fold split to avoid overly-optimistic/overfit thresholds. This preserves your “regression → thresholds → class” core approach and should materially improve QWK without changing architecture, loss, or training loops. The script still run end-to-end and write a valid `submission.csv` in `test.csv` order.'
- What this solution (achieved 0.1571) has done: 'Your gap to the target is large (0.21984 → 0.9103), so the biggest “minimal-change” lever is the discretization step for QWK. I keep your model and inference exactly the same, but change threshold fitting to be (1) truly out-of-fold (each fold’s predictions come from a separately initialized model so they vary, instead of the same fixed ImageNet model repeated 5x), and (2) optimized with a small grid-search refinement around the best coordinate-descent thresholds to better align with QWK. This avoids changing architecture/training loops (there is still no training), but makes the threshold calibration meaningful and typically yields a large QWK jump for APTOS. The script still run end-to-end and write a valid `submission.csv` with `id_code,diagnosis` in `test.csv` order.'
- What this solution (achieved 0.04719) has done: 'Your current score is far below the target, so we should improve QWK mainly by fixing the part that most affects it without changing your model: the thresholds calibration. Right now your “OOF” predictions are not truly out-of-fold because you reload the same fixed ImageNet model each fold, so threshold fitting is effectively done on in-sample predictions and becomes unstable/misaligned. I keep your architecture and inference exactly the same, but compute *true OOF raw predictions* by fitting thresholds via a lightweight, stratified K-fold split and calibrating thresholds on each fold’s holdout predictions (then averaging thresholds). Finally, I add a small safety fallback to handle any missing-image cases while keeping submission order identical to `test.csv`.'
- What this solution (achieved 0.07277) has done: 'Your current score is far below the target, so we should improve QWK with the smallest change that affects the metric most: how continuous regression outputs are converted into the 5 ordinal classes. I keep your model, transforms, and inference logic the same, but replace the current “average thresholds across folds trained on (folds != f)” scheme with a truly OOF calibration: for each fold, fit thresholds on that fold’s holdout predictions (not the in-fold ones), then apply those thresholds to that fold and score; finally fit one final set of thresholds on all valid OOF predictions for use on the test set. This keeps evaluation semantics (regression→thresholds→classes) intact while aligning discretization to QWK and avoiding the current mismatch that tends to collapse kappa. I also add a tiny safety improvement: fill any missing train/test predictions with the median valid raw score before threshold fitting/prediction (still legitimate, avoids NaNs poisoning calibration).'
- What this solution (achieved 0.17757) has done: 'The timeout is dominated by doing 5-fold training where each fold repeatedly reloads the full EfficientNet-B4 and (re)decodes large PNGs from disk; the threshold search is comparatively small. I preserve the exact model, loss, folds, and calibration logic, but remove redundant work by caching decoded+transformed images in RAM (train and test) so each image is read/processed once instead of many times across folds/epochs. I also avoid recreating the model/weights five times by loading the pretrained backbone once and cloning it per fold (same weights, same head-training semantics), and I speed up dataloading by using the in-memory tensors (so workers=0 is optimal and deterministic). These changes are equivalent in outputs (negligible FP differences only) but drastically reduce I/O and repeated preprocessing, which is what pushes you over 600s.'
- What this solution (achieved 0.0) has done: 'Your current score (0.17757) is far below the target (0.9103), so we should improve toward it with the smallest change that affects QWK the most while preserving your core “EffNet regression → thresholds → class” pipeline. The biggest issue is that you train a fold-specific regressor head for OOF calibration, but at test-time you predict with the *untrained* head (fresh ImageNet regressor), so the calibrated thresholds don’t match the test raw-score distribution. I keep your model, transforms, head-training procedure, and threshold-fitting logic intact, but switch test inference to use an ensemble of the same fold-trained head models (average raw predictions) so the thresholds and test predictions are on the same scale. This is a minimal semantic change (still regression→thresholds), should substantially raise QWK, and remains within the 600s budget thanks to your existing image caching.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.nn import Parameter
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import timm
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEED = 1337
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

CANDIDATE_ROOTS = [
    "../input/aptos2019-blindness-detection",
    "/kaggle/input/aptos2019-blindness-detection",
    "/kaggle/data/aptos2019-blindness-detection",
    "/kaggle/data/input/aptos2019-blindness-detection",
]
DATA_ROOT = next((p for p in CANDIDATE_ROOTS if os.path.exists(p)), None)
if DATA_ROOT is None:
    raise FileNotFoundError(
        "Could not find aptos2019-blindness-detection directory in expected locations."
    )

TEST_CSV = os.path.join(DATA_ROOT, "test.csv")
TEST_IMG_DIR = os.path.join(DATA_ROOT, "test_images")
TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train_images")

test_df = pd.read_csv(TEST_CSV)
test_ids = test_df["id_code"].astype(str).tolist()

train_df = pd.read_csv(TRAIN_CSV)
train_df["id_code"] = train_df["id_code"].astype(str)
train_df["diagnosis"] = train_df["diagnosis"].astype(int)

transforms_eval = transforms.Compose(
    [
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.384, 0.258, 0.174], std=[0.124, 0.089, 0.094]),
    ]
)
tranforms = transforms_eval  # keep backward-compatible name used below


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
            + "{:.4f}".format(self.p.detach().cpu().item())
            + ", "
            + "eps="
            + str(self.eps)
            + ")"
        )


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.backbone = timm.create_model(
            "tf_efficientnet_b4_ns",
            pretrained=True,
            num_classes=0,
            global_pool="",
        )
        self.backbone.global_pool = GeM(flatten=True)

        in_features = getattr(self.backbone, "num_features", None)
        if in_features is None:
            raise RuntimeError(
                "Backbone does not expose num_features; cannot build regressor."
            )
        self.regressor = nn.Linear(in_features, 1)

    def forward(self, x):
        x = self.backbone(x)
        out = self.regressor(x)
        out = torch.sigmoid(out) * 4.5
        return out


def quadratic_weighted_kappa(y_true, y_pred, num_classes=5):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    mask = (
        (y_true >= 0) & (y_true < num_classes) & (y_pred >= 0) & (y_pred < num_classes)
    )
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if y_true.size == 0:
        return 0.0

    O = np.zeros((num_classes, num_classes), dtype=np.float64)
    np.add.at(O, (y_true, y_pred), 1.0)

    act_hist = np.bincount(y_true, minlength=num_classes).astype(np.float64)
    pred_hist = np.bincount(y_pred, minlength=num_classes).astype(np.float64)
    E = np.outer(act_hist, pred_hist)
    if E.sum() > 0:
        E *= O.sum() / E.sum()

    idx = np.arange(num_classes, dtype=np.float64)
    W = (idx[:, None] - idx[None, :]) ** 2 / ((num_classes - 1) ** 2)

    denom = (W * E).sum()
    if denom == 0:
        return 0.0
    return 1.0 - (W * O).sum() / denom


def thresholds_to_pred(raw, thr):
    thr = np.asarray(thr, dtype=np.float32)
    raw = np.asarray(raw, dtype=np.float32)
    return (raw[:, None] >= thr[None, :]).sum(axis=1).astype(np.int64)


def _init_thresholds_from_label_dist(raw, y, eps=1e-6):
    raw = np.asarray(raw, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)

    hist = np.bincount(y, minlength=5).astype(np.float64)
    frac = hist / max(hist.sum(), 1.0)
    cum = np.cumsum(frac)
    q = np.clip(cum[:-1], eps, 1 - eps)
    return np.quantile(raw, q).astype(np.float32)


def fit_thresholds_qwk(raw_train, y_train, init_thr=None, iters=80):
    raw_train = np.asarray(raw_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int64)

    fixed = np.array([0.7, 1.5, 2.5, 3.5], dtype=np.float32)
    q_init = np.quantile(raw_train, [0.2, 0.4, 0.6, 0.8]).astype(np.float32)
    ld_init = _init_thresholds_from_label_dist(raw_train, y_train).astype(np.float32)

    if init_thr is None:
        init_thr = fixed
    else:
        init_thr = np.asarray(init_thr, dtype=np.float32)

    thr = (0.34 * fixed + 0.33 * q_init + 0.33 * ld_init).astype(np.float32)
    thr = 0.5 * thr + 0.5 * init_thr
    thr = np.sort(thr)

    eps = 1e-3
    best_thr = thr.copy()
    best_kappa = quadratic_weighted_kappa(
        y_train, thresholds_to_pred(raw_train, best_thr)
    )

    step = 0.5
    for _ in range(iters):
        improved = False
        for i in range(4):
            base = best_thr[i]
            for direction in (-1.0, 1.0):
                cand = best_thr.copy()
                cand[i] = base + direction * step

                if i > 0:
                    cand[i] = max(cand[i], cand[i - 1] + eps)
                if i < 3:
                    cand[i] = min(cand[i], cand[i + 1] - eps)

                k = quadratic_weighted_kappa(
                    y_train, thresholds_to_pred(raw_train, cand)
                )
                if k > best_kappa:
                    best_kappa = k
                    best_thr = cand
                    improved = True

        if not improved:
            step *= 0.5
            if step < 5e-3:
                break

    return best_thr.astype(np.float32), float(best_kappa)


def refine_thresholds_local_grid(raw_train, y_train, base_thr, span=0.25, step=0.05):
    raw_train = np.asarray(raw_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int64)
    base_thr = np.asarray(base_thr, dtype=np.float32)

    eps = 1e-3
    best_thr = base_thr.copy()
    best_k = quadratic_weighted_kappa(y_train, thresholds_to_pred(raw_train, best_thr))

    deltas = np.arange(-span, span + 1e-9, step, dtype=np.float32)

    for i in range(4):
        cur_best_thr_i = best_thr[i]
        for d in deltas:
            cand = best_thr.copy()
            cand[i] = cur_best_thr_i + float(d)
            if i > 0:
                cand[i] = max(cand[i], cand[i - 1] + eps)
            if i < 3:
                cand[i] = min(cand[i], cand[i + 1] - eps)
            k = quadratic_weighted_kappa(y_train, thresholds_to_pred(raw_train, cand))
            if k > best_k:
                best_k = k
                best_thr = cand

    return best_thr.astype(np.float32), float(best_k)


CANDIDATE_WEIGHT_PATHS = [
    "../input/weights/tf_efficientnet_b4_ns_regress.pth",
    "/kaggle/input/weights/tf_efficientnet_b4_ns_regress.pth",
    "/kaggle/data/input/weights/tf_efficientnet_b4_ns_regress.pth",
]
WEIGHT_PATH = next((p for p in CANDIDATE_WEIGHT_PATHS if os.path.exists(p)), None)


def _load_model():
    net = Model()
    if WEIGHT_PATH is not None:
        ckpt = torch.load(WEIGHT_PATH, map_location=device)
        if isinstance(ckpt, nn.Module):
            net = ckpt
        elif isinstance(ckpt, dict):
            state = ckpt.get("state_dict", ckpt)
            try:
                net.load_state_dict(state, strict=True)
            except RuntimeError:
                stripped = {}
                for k, v in state.items():
                    stripped[
                        k.replace("module.", "", 1) if k.startswith("module.") else k
                    ] = v
                net.load_state_dict(stripped, strict=False)

    net = net.to(device)
    net.eval()
    return net


def _load_image_tensor(img_path):
    img = Image.open(img_path).convert("RGB")
    return tranforms(img)


def _preload_image_cache(ids, img_dir):
    cache = {}
    ok = {}
    for img_id in ids:
        path = os.path.join(img_dir, f"{img_id}.png")
        try:
            cache[img_id] = _load_image_tensor(path)  # CPU tensor float32
            ok[img_id] = True
        except Exception:
            cache[img_id] = torch.zeros((3, 384, 384), dtype=torch.float32)
            ok[img_id] = False
    return cache, ok


class IdDataset(Dataset):
    def __init__(self, ids, img_dir, y=None, cache=None, ok_cache=None):
        self.ids = list(ids)
        self.img_dir = img_dir
        self.y = None if y is None else np.asarray(y, dtype=np.int64)
        self.cache = cache
        self.ok_cache = ok_cache

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        img_id = self.ids[i]
        if self.cache is not None and img_id in self.cache:
            x = self.cache[img_id]
            ok = (
                bool(self.ok_cache.get(img_id, True))
                if self.ok_cache is not None
                else True
            )
        else:
            path = os.path.join(self.img_dir, f"{img_id}.png")
            try:
                x = _load_image_tensor(path)
                ok = True
            except Exception:
                x = torch.zeros((3, 384, 384), dtype=torch.float32)
                ok = False

        if self.y is None:
            return x, ok, i
        return x, float(self.y[i]), ok, i


def _seed_worker(worker_id):
    s = (SEED + worker_id) % (2**32 - 1)
    np.random.seed(s)
    random.seed(s)
    torch.manual_seed(s)


def _make_loader(dataset, batch_size, shuffle, num_workers):
    g = torch.Generator()
    g.manual_seed(SEED)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        worker_init_fn=_seed_worker if num_workers > 0 else None,
        generator=g,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )


def _suggest_num_workers():
    try:
        cpu = os.cpu_count() or 2
    except Exception:
        cpu = 2
    return max(2, min(8, cpu // 2))


def predict_raw(ids, img_dir, net, batch_size=16, cache=None, ok_cache=None):
    raw = np.full((len(ids),), np.nan, dtype=np.float32)
    ok = np.zeros((len(ids),), dtype=bool)

    ds = IdDataset(ids, img_dir, y=None, cache=cache, ok_cache=ok_cache)
    num_workers = 0 if cache is not None else _suggest_num_workers()
    loader = _make_loader(
        ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    net.eval()
    with torch.no_grad():
        for xb, okb, ib in loader:
            xb = xb.to(device, non_blocking=True)
            out = net(xb).squeeze(1).detach().cpu().numpy().astype(np.float32)
            ib_np = ib.numpy()
            ok_np = np.asarray(okb.numpy(), dtype=bool)
            raw[ib_np] = out
            ok[ib_np] = ok_np

    return raw, ok


def _train_regressor_head_for_fold(
    base_net,
    train_ids,
    train_y,
    img_dir,
    epochs=2,
    batch_size=16,
    lr=3e-3,
    seed=1337,
    cache=None,
    ok_cache=None,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    net = base_net
    net.train()

    for p in net.backbone.parameters():
        p.requires_grad = False
    for p in net.regressor.parameters():
        p.requires_grad = True

    opt = torch.optim.Adam(net.regressor.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    ds = IdDataset(
        train_ids,
        img_dir,
        y=np.asarray(train_y, dtype=np.int64),
        cache=cache,
        ok_cache=ok_cache,
    )
    num_workers = 0 if cache is not None else _suggest_num_workers()
    loader = _make_loader(
        ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    with torch.enable_grad():
        for _ in range(int(epochs)):
            for xb, yb, okb, _ib in loader:
                ok_mask = okb.bool()
                if ok_mask.sum().item() == 0:
                    continue
                xb = xb[ok_mask].to(device, non_blocking=True)
                yb = torch.tensor(np.asarray(yb, dtype=np.float32), device=device).view(
                    -1, 1
                )
                yb = yb[ok_mask.to(device)]

                pred = net(xb)
                loss = loss_fn(pred, yb)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

    net.eval()
    return net


def make_true_oof_for_thresholds(
    train_df,
    img_dir,
    n_splits=5,
    seed=1337,
    batch_size=16,
    head_epochs=2,
):
    ids = train_df["id_code"].tolist()
    y = train_df["diagnosis"].values.astype(np.int64)

    train_cache, train_ok_cache = _preload_image_cache(ids, img_dir)

    rng = np.random.RandomState(seed)
    folds = np.full(len(ids), -1, dtype=np.int64)
    for c in range(5):
        cls_idx = np.where(y == c)[0]
        rng.shuffle(cls_idx)
        for k, j in enumerate(cls_idx):
            folds[j] = k % n_splits

    raw_oof = np.full((len(ids),), np.nan, dtype=np.float32)
    ok_oof = np.zeros((len(ids),), dtype=bool)

    base_template = _load_model()
    base_state = copy.deepcopy(base_template.state_dict())
    del base_template
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    for f in range(n_splits):
        tr_idx = np.where(folds != f)[0]
        ho_idx = np.where(folds == f)[0]

        base_net = Model().to(device)
        base_net.load_state_dict(base_state, strict=True)
        base_net.eval()

        fold_net = _train_regressor_head_for_fold(
            base_net=base_net,
            train_ids=[ids[i] for i in tr_idx],
            train_y=y[tr_idx],
            img_dir=img_dir,
            epochs=head_epochs,
            batch_size=batch_size,
            lr=3e-3,
            seed=seed + f,
            cache=train_cache,
            ok_cache=train_ok_cache,
        )

        fold_ids = [ids[i] for i in ho_idx]
        raw_f, ok_f = predict_raw(
            fold_ids,
            img_dir,
            net=fold_net,
            batch_size=batch_size,
            cache=train_cache,
            ok_cache=train_ok_cache,
        )
        raw_oof[ho_idx] = raw_f
        ok_oof[ho_idx] = ok_f

        del fold_net, base_net
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return raw_oof, ok_oof, y, folds, base_state, train_cache, train_ok_cache


raw_oof, ok_oof, y_train, folds, base_state, train_cache, train_ok_cache = (
    make_true_oof_for_thresholds(
        train_df,
        TRAIN_IMG_DIR,
        n_splits=5,
        seed=1337,
        batch_size=16,
        head_epochs=2,
    )
)

valid_oof = np.isfinite(raw_oof) & ok_oof
raw_fill = float(np.nanmedian(raw_oof[valid_oof])) if valid_oof.any() else 2.0
raw_oof_filled = raw_oof.copy()
raw_oof_filled[~valid_oof] = raw_fill

if valid_oof.sum() < 200:
    calibrated_thr = np.array([0.7, 1.5, 2.5, 3.5], dtype=np.float32)
    oof_kappa = None
else:
    thr_by_fold = []
    for f in range(5):
        ho_mask = (folds == f) & valid_oof
        if ho_mask.sum() < 40:
            continue

        thr_f, _ = fit_thresholds_qwk(
            raw_oof_filled[ho_mask],
            y_train[ho_mask],
            init_thr=(0.7, 1.5, 2.5, 3.5),
            iters=80,
        )
        thr_f, _ = refine_thresholds_local_grid(
            raw_oof_filled[ho_mask],
            y_train[ho_mask],
            base_thr=thr_f,
            span=0.25,
            step=0.05,
        )
        thr_by_fold.append(thr_f)

    thr_all, _ = fit_thresholds_qwk(
        raw_oof_filled[valid_oof],
        y_train[valid_oof],
        init_thr=(
            np.mean(np.stack(thr_by_fold, axis=0), axis=0)
            if len(thr_by_fold) > 0
            else (0.7, 1.5, 2.5, 3.5)
        ),
        iters=80,
    )
    calibrated_thr, _ = refine_thresholds_local_grid(
        raw_oof_filled[valid_oof],
        y_train[valid_oof],
        base_thr=thr_all,
        span=0.25,
        step=0.05,
    )
    calibrated_thr = np.sort(calibrated_thr.astype(np.float32))
    oof_kappa = quadratic_weighted_kappa(
        y_train[valid_oof],
        thresholds_to_pred(raw_oof_filled[valid_oof], calibrated_thr),
    )

test_cache, test_ok_cache = _preload_image_cache(test_ids, TEST_IMG_DIR)


def predict_raw_test_ensemble_from_fold_heads(
    test_ids,
    test_img_dir,
    base_state,
    train_df,
    train_img_dir,
    n_splits=5,
    seed=1337,
    batch_size=16,
    head_epochs=2,
    train_cache=None,
    train_ok_cache=None,
    test_cache=None,
    test_ok_cache=None,
):
    ids = train_df["id_code"].tolist()
    y = train_df["diagnosis"].values.astype(np.int64)

    rng = np.random.RandomState(seed)
    folds = np.full(len(ids), -1, dtype=np.int64)
    for c in range(5):
        cls_idx = np.where(y == c)[0]
        rng.shuffle(cls_idx)
        for k, j in enumerate(cls_idx):
            folds[j] = k % n_splits

    raw_sum = np.zeros((len(test_ids),), dtype=np.float32)
    ok_any = np.zeros((len(test_ids),), dtype=bool)

    for f in range(n_splits):
        tr_idx = np.where(folds != f)[0]

        base_net = Model().to(device)
        base_net.load_state_dict(base_state, strict=True)
        base_net.eval()

        fold_net = _train_regressor_head_for_fold(
            base_net=base_net,
            train_ids=[ids[i] for i in tr_idx],
            train_y=y[tr_idx],
            img_dir=train_img_dir,
            epochs=head_epochs,
            batch_size=batch_size,
            lr=3e-3,
            seed=seed + f,
            cache=train_cache,
            ok_cache=train_ok_cache,
        )

        raw_f, ok_f = predict_raw(
            test_ids,
            test_img_dir,
            net=fold_net,
            batch_size=batch_size,
            cache=test_cache,
            ok_cache=test_ok_cache,
        )
        raw_sum += np.nan_to_num(raw_f, nan=0.0).astype(np.float32)
        ok_any |= ok_f.astype(bool)

        del fold_net, base_net
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    raw_mean = raw_sum / float(n_splits)
    return raw_mean.astype(np.float32), ok_any


raw_test, ok_test = predict_raw_test_ensemble_from_fold_heads(
    test_ids=test_ids,
    test_img_dir=TEST_IMG_DIR,
    base_state=base_state,
    train_df=train_df,
    train_img_dir=TRAIN_IMG_DIR,
    n_splits=5,
    seed=1337,
    batch_size=16,
    head_epochs=2,
    train_cache=train_cache,
    train_ok_cache=train_ok_cache,
    test_cache=test_cache,
    test_ok_cache=test_ok_cache,
)

finite_test = np.isfinite(raw_test)
test_fill = (
    float(np.nanmedian(raw_test[finite_test])) if finite_test.any() else raw_fill
)
raw_test_filled = raw_test.copy()
raw_test_filled[~finite_test] = test_fill

test_pred = thresholds_to_pred(raw_test_filled, calibrated_thr)
test_pred = np.clip(test_pred, 0, 4).astype(np.int64)

print("Calibrated thresholds:", [float(x) for x in calibrated_thr])
if oof_kappa is not None:
    print(
        "OOF Train QWK using calibrated thresholds (on valid OOF preds):",
        float(oof_kappa),
    )
print("Test finite preds:", int(finite_test.sum()), "out of", len(test_ids))




## === cell 1
df = pd.DataFrame({"id_code": test_ids, "diagnosis": test_pred.astype(int)})
df["id_code"] = df["id_code"].astype(str)
df["diagnosis"] = df["diagnosis"].astype(int)

if df.empty:
    raise RuntimeError("Submission DataFrame is empty; inference produced no rows.")

df.to_csv("submission.csv", index=False)
print(df.head())
print("Wrote submission.csv with shape:", df.shape)
print(
    "diagnosis value counts:\n", df["diagnosis"].value_counts(dropna=False).sort_index()
)
