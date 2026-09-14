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

3.8

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

0.813172612933002

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -0.12777) has done: 'I fix the pipeline so it always produces a valid `submission.csv` even when the external pretrained weight file is not attached (your current crash). To keep the core model and preprocessing intact, I (1) make weight loading optional by falling back to torchvision’s built-in DenseNet121 ImageNet weights when the custom checkpoint is missing, and (2) fix the DenseNet feature-dimension mismatch by using `net.classifier.in_features` rather than a hardcoded value. I also make `load_state_dict` robust to common checkpoint formats (`state_dict` key, `module.` prefixes) without changing the architecture or inference logic. These changes unblock end-to-end execution and typically improve over random/uninitialized weights, moving score toward the target without altering evaluation semantics.'
- What this solution (achieved -0.35074) has done: 'Your current negative score is consistent with using ImageNet backbone features but a randomly initialized classifier head, so the smallest score-improving change (without altering your model or training loop) is to calibrate the 5-class outputs using the training label distribution. I keep your DenseNet121 + single-head inference exactly the same, but replace the raw `argmax` with a rank-based mapping that assigns predicted classes so the overall class counts match the empirical train priors (a common QWK post-processing that improves agreement for ordinal labels). This is purely post-processing on the same logits, doesn’t change architecture/loss/training, and typically moves QWK up substantially from near-random outputs. I also batch CSV writing to avoid per-row file opens (no semantic change) and ensure predictions are in {0..4}.'
- What this solution (achieved -0.03408) has done: 'Your score is far below the target (gap = -1.1639), so we should improve it with minimal risk while keeping the same model and inference. The biggest issue is that your current post-processing forces the test predictions to match the *train* class distribution, which can badly hurt QWK if the test distribution differs; instead, we use a tiny validation split on the training set to learn 4 optimal ordinal thresholds for the same expected-severity score and then apply them to test. This keeps the core logic intact (DenseNet121 backbone + softmax + expected severity), but aligns discretization directly to QWK, which typically moves the score upward substantially from negative. We also fix the transform to include ImageNet normalization (still just a transform change) so the ImageNet-fallback backbone operates in the feature space it was trained for, improving stability without changing architecture or loops.'
- What this solution (achieved 0.0) has done: 'Your current score is far below the target, so we should push it upward with the smallest risk changes that keep your DenseNet121 inference and expected-severity scoring intact. The main issue is that you calibrate thresholds on a random validation split but then still use that same (untrained/random-head) model; the thresholds become noise and QWK stays poor. I (1) load ImageNet weights correctly into the full DenseNet backbone **and** initialize the classifier in a sensible way when the custom checkpoint is missing, and (2) fit thresholds on **out-of-fold** predictions (stratified K-fold) to reduce overfitting/leakage in threshold tuning without changing your model or loss. These are minimal, metric-aligned changes that typically lift QWK substantially while preserving the core architecture and inference semantics.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os
import sys
import time
import datetime
import argparse
import os.path as osp
import random
from PIL import Image
import cv2
import csv

import torchvision as tv
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    from tensorboardX import SummaryWriter  # noqa: F401
except ModuleNotFoundError:
    SummaryWriter = None  # not used in this inference-only notebook/script

SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)



## === cell 1
name_file = "/kaggle/input/aptos2019-blindness-detection/test.csv"
with open(name_file, "r", newline="") as f:
    csv_file = csv.reader(f)
    content = []
    for line in csv_file:
        content.append(line[0] + ".png")
content = content[1:]  # drop header



## === cell 2
import torch
from torch import nn
import torchvision


class Baseline_single(nn.Module):
    def __init__(self, num_classes, loss_type="single BCE", **kwargs):
        super(Baseline_single, self).__init__()
        self.loss_type = loss_type

        backbone = torchvision.models.densenet121(pretrained=False)

        self.base = backbone.features
        self.feature_dim = backbone.classifier.in_features  # 1024 for densenet121

        if self.loss_type == "single BCE":
            self.ap = nn.AdaptiveAvgPool2d(1)
            self.classifiers = nn.Linear(
                in_features=self.feature_dim, out_features=num_classes
            )
            self.sigmoid = nn.Sigmoid()
            self.dropout = nn.Dropout(0.5)

    def freeze_base(self):
        for p in self.base.parameters():
            p.requires_grad = False

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True

    def forward(self, x1):
        x = self.base(x1)
        if self.loss_type == "single BCE":
            x = self.ap(x)
            x = self.dropout(x)
            x = x.view(x.size(0), -1)
            ys = self.classifiers(x)
        return ys




## === cell 3
def cv_imread(file_path):
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
    return cv_img


def crop_image_from_gray(img, tol=7):
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol

        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if check_shape == 0:
            return img
        else:
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            img = np.stack([img1, img2, img3], axis=-1)
        return img


def load_ben_yuan(image, sigmaX=10):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = crop_image_from_gray(image)
    image = cv2.resize(image, (512, 512))
    return image


class eye_dataset(Dataset):
    def __init__(self, txt_path, transform=None):
        self.imgs = list(txt_path)
        self.transform = transform

    def __getitem__(self, index):
        fn = self.imgs[index]
        img_path = "/kaggle/input/aptos2019-blindness-detection/test_images/" + fn
        img = cv_imread(img_path)
        img = load_ben_yuan(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, fn[:-4]

    def __len__(self):
        return len(self.imgs)




## === cell 4
def _find_model_path():
    candidates = [
        "/kaggle/input/temp-file/model_yuan512_dense121_00001_adam_max.pkl",
        "/kaggle/input/temp-file/model_yuan512_dense121_00001_adam_max.pth",
        "/kaggle/input/model_yuan512_dense121_00001_adam_max.pkl",
        "/kaggle/input/model_yuan512_dense121_00001_adam_max.pth",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p

    target_names = [
        "model_yuan512_dense121_00001_adam_max.pkl",
        "model_yuan512_dense121_00001_adam_max.pth",
    ]
    for root, _, files in os.walk("/kaggle/input"):
        for tn in target_names:
            if tn in files:
                return os.path.join(root, tn)

    return None


def _extract_state_dict(obj):
    if isinstance(obj, dict):
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            return obj["state_dict"]
        if "model_state_dict" in obj and isinstance(obj["model_state_dict"], dict):
            return obj["model_state_dict"]
    return obj  # assume raw state_dict


def _strip_module_prefix(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict
    if not any(k.startswith("module.") for k in state_dict.keys()):
        return state_dict
    return {k.replace("module.", "", 1): v for k, v in state_dict.items()}


def _init_classifier_from_train_priors(net, train_labels, eps=1e-6):
    """
    Change rationale (score-improving, minimal): if we fall back to ImageNet backbone but have a random
    classifier head, predictions are near-random and QWK can be negative. Initializing the classifier
    bias to match train priors yields a sane baseline distribution without changing the architecture.
    """
    y = np.asarray(train_labels, dtype=np.int64)
    counts = np.bincount(y, minlength=5).astype(np.float64)
    p = counts / max(counts.sum(), 1.0)
    p = np.clip(p, eps, 1.0)
    p = p / p.sum()
    bias = np.log(p)  # softmax(bias) = p when weights are ~0
    with torch.no_grad():
        net.classifiers.weight.zero_()
        net.classifiers.bias.copy_(
            torch.tensor(
                bias,
                dtype=net.classifiers.bias.dtype,
                device=net.classifiers.bias.device,
            )
        )


def _try_load_weights_or_fallback_imagenet(net, use_gpu, train_labels_for_prior=None):
    model_path = _find_model_path()
    if model_path is not None:
        state = torch.load(model_path, map_location="cuda" if use_gpu else "cpu")
        state = _extract_state_dict(state)
        state = _strip_module_prefix(state)
        missing, unexpected = net.load_state_dict(state, strict=False)
        print(f"Loaded checkpoint: {model_path}")
        if missing:
            print(f"Missing keys (showing up to 20): {missing[:20]}")
        if unexpected:
            print(f"Unexpected keys (showing up to 20): {unexpected[:20]}")
        return "custom_checkpoint"

    print(
        "WARNING: Custom model weights not found under /kaggle/input. "
        "Falling back to torchvision DenseNet121 ImageNet weights for the backbone "
        "and initializing the classifier to train label priors."
    )
    try:
        weights = torchvision.models.DenseNet121_Weights.DEFAULT
        pretrained = torchvision.models.densenet121(weights=weights)
    except Exception:
        pretrained = torchvision.models.densenet121(pretrained=True)

    net.base.load_state_dict(pretrained.features.state_dict(), strict=True)

    if train_labels_for_prior is not None:
        _init_classifier_from_train_priors(net, train_labels_for_prior)

    return "imagenet_fallback"


def _quadratic_weighted_kappa(y_true, y_pred, n_classes=5):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    assert y_true.shape == y_pred.shape

    O = np.zeros((n_classes, n_classes), dtype=np.float64)
    for a, b in zip(y_true, y_pred):
        if 0 <= a < n_classes and 0 <= b < n_classes:
            O[a, b] += 1.0

    hist_true = np.bincount(y_true, minlength=n_classes).astype(np.float64)
    hist_pred = np.bincount(y_pred, minlength=n_classes).astype(np.float64)
    E = np.outer(hist_true, hist_pred)
    E = E / E.sum() * O.sum() if E.sum() > 0 else E

    W = np.zeros((n_classes, n_classes), dtype=np.float64)
    for i in range(n_classes):
        for j in range(n_classes):
            W[i, j] = ((i - j) ** 2) / ((n_classes - 1) ** 2)

    num = (W * O).sum()
    den = (W * E).sum()
    return 1.0 - num / den if den > 0 else 0.0


def _apply_thresholds(scores, thresholds):
    t0, t1, t2, t3 = thresholds
    s = np.asarray(scores, dtype=np.float64)
    y = np.zeros_like(s, dtype=np.int64)
    y[s >= t0] = 1
    y[s >= t1] = 2
    y[s >= t2] = 3
    y[s >= t3] = 4
    return y


def _fit_thresholds_qwk(scores, y_true, n_classes=5, max_iter=3):
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(y_true, dtype=np.int64)

    qs = []
    for k in range(1, n_classes):
        q = np.quantile(s, np.mean(y < k))
        qs.append(float(q))
    thresholds = np.array(qs, dtype=np.float64)
    thresholds.sort()

    grid = np.unique(np.quantile(s, np.linspace(0.05, 0.95, 61))).astype(np.float64)

    best_thr = thresholds.copy()
    best_k = _quadratic_weighted_kappa(
        y, _apply_thresholds(s, best_thr), n_classes=n_classes
    )

    for _ in range(max_iter):
        improved = False
        for i in range(n_classes - 1):
            low = grid[0] if i == 0 else best_thr[i - 1] + 1e-6
            high = grid[-1] if i == n_classes - 2 else best_thr[i + 1] - 1e-6
            candidates = grid[(grid >= low) & (grid <= high)]
            if candidates.size == 0:
                continue

            local_best_thr = best_thr.copy()
            local_best_k = best_k

            for c in candidates:
                trial = best_thr.copy()
                trial[i] = c
                kappa = _quadratic_weighted_kappa(
                    y, _apply_thresholds(s, trial), n_classes=n_classes
                )
                if kappa > local_best_k:
                    local_best_k = kappa
                    local_best_thr = trial

            if local_best_k > best_k + 1e-12:
                best_k = local_best_k
                best_thr = local_best_thr
                improved = True

        if not improved:
            break

    best_thr.sort()
    return best_thr, best_k


def _stratified_kfold_indices(y, n_splits=5, seed=0):
    """
    Change rationale (score-improving, minimal): thresholds tuned on a single random split can overfit
    and be unstable; using stratified K-fold OOF scores makes the QWK-optimized thresholds more robust,
    improving expected leaderboard QWK without changing the model/inference.
    """
    y = np.asarray(y, dtype=np.int64)
    rng = np.random.RandomState(seed)
    folds = [[] for _ in range(n_splits)]
    for c in range(5):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        parts = np.array_split(idx, n_splits)
        for k in range(n_splits):
            folds[k].extend(parts[k].tolist())
    folds = [np.array(sorted(f), dtype=np.int64) for f in folds]
    return folds




## === cell 5
if __name__ == "__main__":
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(SEED)
    else:
        print("Currently using CPU (GPU is highly recommended)")

    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    transform2 = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ]
    )

    name_file = "/kaggle/input/aptos2019-blindness-detection/test.csv"
    with open(name_file, "r", newline="") as f:
        csv_file = csv.reader(f)
        content = []
        for line in csv_file:
            content.append(line[0] + ".png")
    content = content[1:]  # drop header

    test_data = eye_dataset(content, transform2)
    net = Baseline_single(num_classes=5)
    if use_gpu:
        net = net.cuda()

    train_df = pd.read_csv("/kaggle/input/aptos2019-blindness-detection/train.csv")
    train_labels_all = train_df["diagnosis"].values.astype(np.int64)
    train_ids_all = train_df["id_code"].values.astype(str)

    _ = _try_load_weights_or_fallback_imagenet(
        net, use_gpu, train_labels_for_prior=train_labels_all
    )

    dataloader_test = DataLoader(
        test_data, batch_size=8, shuffle=False, num_workers=2, pin_memory=use_gpu
    )

    class TrainEyeDataset(Dataset):
        def __init__(self, ids, labels, transform=None):
            self.ids = list(ids)
            self.labels = np.asarray(labels, dtype=np.int64)
            self.transform = transform

        def __len__(self):
            return len(self.ids)

        def __getitem__(self, i):
            id_code = self.ids[i]
            img_path = (
                "/kaggle/input/aptos2019-blindness-detection/train_images/"
                + id_code
                + ".png"
            )
            img = cv_imread(img_path)
            img = load_ben_yuan(img)
            if self.transform is not None:
                img = self.transform(img)
            return img, int(self.labels[i])

    n_splits = 5
    folds = _stratified_kfold_indices(train_labels_all, n_splits=n_splits, seed=SEED)

    oof_scores = np.zeros(len(train_labels_all), dtype=np.float64)
    oof_y = train_labels_all.copy()

    with torch.no_grad():
        net.eval()
        for k in range(n_splits):
            val_idx = folds[k]
            val_ids = train_ids_all[val_idx]
            val_labels = train_labels_all[val_idx]
            val_data = TrainEyeDataset(val_ids, val_labels, transform2)
            val_loader = DataLoader(
                val_data, batch_size=8, shuffle=False, num_workers=2, pin_memory=use_gpu
            )

            fold_scores = []
            for data, yb in tqdm(
                val_loader,
                total=len(val_loader),
                desc=f"Calibrating OOF fold {k+1}/{n_splits}",
            ):
                if use_gpu:
                    data = data.cuda(non_blocking=True)
                out = net(data)  # [B,5] logits
                prob = torch.softmax(out, dim=1)
                classes = torch.arange(5, device=prob.device, dtype=prob.dtype).view(
                    1, -1
                )
                exp_sev = (prob * classes).sum(dim=1)  # [B]
                fold_scores.extend(exp_sev.detach().float().cpu().numpy().tolist())

            oof_scores[val_idx] = np.asarray(fold_scores, dtype=np.float64)

    thresholds, oof_kappa = _fit_thresholds_qwk(
        oof_scores.tolist(), oof_y.tolist(), n_classes=5, max_iter=3
    )
    print(
        f"Fitted thresholds (OOF): {thresholds.tolist()} | OOF QWK (internal): {oof_kappa:.6f}"
    )

    ids = []
    severity_scores = []
    with torch.no_grad():
        net.eval()
        for _, item in tqdm(
            enumerate(dataloader_test), total=len(dataloader_test), desc="Predicting"
        ):
            data, name = item
            if use_gpu:
                data = data.cuda(non_blocking=True)
            out = net(data)  # [B,5] logits
            prob = torch.softmax(out, dim=1)
            classes = torch.arange(5, device=prob.device, dtype=prob.dtype).view(1, -1)
            exp_sev = (prob * classes).sum(dim=1)  # [B]
            severity_scores.extend(exp_sev.detach().float().cpu().numpy().tolist())
            ids.extend([str(n) for n in name])

    mapped = _apply_thresholds(severity_scores, thresholds).astype(int)

    sub_path = "/kaggle/working/submission.csv"
    sub_df = pd.DataFrame({"id_code": ids, "diagnosis": mapped})
    sub_df.to_csv(sub_path, index=False)

    sub_df_check = pd.read_csv(sub_path)
    assert list(sub_df_check.columns) == ["id_code", "diagnosis"]
    assert len(sub_df_check) == len(
        test_data
    ), f"Submission rows {len(sub_df_check)} != test rows {len(test_data)}"
    assert sub_df_check["diagnosis"].between(0, 4).all()
    print(f"Wrote {sub_path} with shape {sub_df_check.shape}")
