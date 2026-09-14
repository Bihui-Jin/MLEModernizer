# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.923001153423795

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix three execution blockers while keeping the model/inference logic unchanged: (1) make the runtime device-safe by selecting CUDA only if available (your environment has no GPU driver), (2) remove the dependency on a missing weights file by loading weights only if they exist and otherwise falling back to the included random initialization so a non-empty submission is still produced, and (3) ensure the image transforms never return `None` (your `trim()` sometimes does), which was silently breaking the pipeline. I also remove the external `pip install` step for timm since timm is already installed, preventing a hard failure from a missing wheel. The result run end-to-end and always write a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is coming from running an untrained (randomly initialized) model because the weights file path points to a non-existent dataset folder, so predictions are essentially random. The smallest change that legitimately improves score toward your target is to load the correct pretrained checkpoint by searching common Kaggle input locations for the expected filename (and only falling back to random init if truly absent). I also make inference a bit more robust by batching and using `torch.inference_mode()` (same semantics as `no_grad`) to avoid any accidental per-image issues, without changing the model or transforms. The submission schema/merge with `sample_submission.csv` remains unchanged.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is consistent with either (a) running with random weights because the checkpoint isn’t actually found/loaded, or (b) loading a checkpoint object that isn’t a raw `state_dict` (common on Kaggle), so `load_state_dict` silently fails to apply meaningful weights. I keep your model and inference logic the same, but make checkpoint loading robust by (1) searching additional realistic locations under the provided `/kaggle/data/...` tree and (2) correctly extracting the `state_dict` from typical checkpoint formats (`state_dict`, `model`, `model_state_dict`) and stripping a possible `module.` prefix. This is the smallest legitimate change that should move the score upward toward your target, because it ensures the intended trained weights are actually used. The submission writing and label post-processing stay identical.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score strongly suggests the intended trained checkpoint is still not being loaded (so the model is effectively random), or test images aren’t being found due to an input-path mismatch, leading to many default-0 predictions. I make two minimal, directly score-relevant fixes: (1) robustly resolve the dataset root and test_images directory from the actually available `/kaggle/...` paths, and (2) make checkpoint discovery slightly more flexible by also searching for common extensions and verifying that the loaded state dict meaningfully matches the model (otherwise warning clearly). These changes keep your model, transforms, thresholds, and inference semantics the same; they only ensure you’re using the correct files so the score can move up toward your target. The script still always produce a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with the model effectively behaving like random/untrained weights (checkpoint not found or not actually applied) and/or producing degenerate predictions due to missing images. I keep your model, transforms, thresholds, and inference logic the same, but (1) make checkpoint discovery and loading more robust by also scanning the working directory and accepting common checkpoint key patterns (including nested `model_state` variants), and (2) hard-fail if too many test images are missing to avoid silently submitting mostly zeros. These are minimal, directly score-relevant changes aimed at ensuring the intended trained weights and correct test images are actually used, which should increase QWK toward your target. The submission format and column names remain unchanged and a valid `submission.csv` is always written when inputs are sane.'
- What this solution (achieved 0.57088) has done: 'Main runtime is spent decoding PNGs and running heavy PIL transforms for every training epoch with a single-process DataLoader; this dominates wall time and causes the timeout. The core model and training loop are kept identical, but data loading is optimized by enabling multi-worker loading, pinned memory, persistent workers, and prefetching; additionally, we cache each transformed training image tensor in RAM so epoch 2 reuses it without re-decoding/re-transforming. For inference, we remove the per-sample `os.path.exists` check and rely on `try/except` around image open (equivalent behavior) and also enable multi-worker loading and pinned memory to reduce input stalls. These changes preserve evaluation semantics while cutting I/O and preprocessing overhead substantially.'
- What this solution (achieved 0.73577) has done: 'Your current score (0.57088) is far below the target (0.9230), so we should increase performance with minimal, score-relevant changes while keeping your architecture and overall approach intact. The biggest issue is that you’re only training the regressor head for 2 epochs on raw class labels (0–4), but your regressor output is scaled to 0–4.5; simply matching target scale and running a few more epochs (still the same loop/loss/model) should move QWK upward substantially. I also fix a small DataLoader setting that can error on some PyTorch versions (`prefetch_factor=None` when `num_workers=0`) and make the regression-to-class thresholds a tiny bit less biased by using midpoints (0.5,1.5,2.5,3.5), which better matches ordinal class boundaries for QWK without changing inference semantics. All paths, model definition, transforms, and the “train then infer then write submission.csv” pipeline remain the same.'
- What this solution (achieved 0.74011) has done: 'Your current score (0.73577) is well below the target (0.9230), so we should improve QWK with the smallest changes that keep your model/loop/loss intact. The most score-relevant issue is that you’re training against only the regression head `r_out` but at inference you also use `r_out`—however you’re extracting `r_out` from the non-final branch, while you already have a `final_regressor` designed to fuse classifier+regressor+ordinal outputs; using that existing fused head for both training and inference should raise agreement without changing architecture. I keep the same dataset, transforms, optimizer, MSE loss, and epochs, but switch the forward call to `final=True` in both training and inference and keep the same label scaling and thresholding. This is a minimal semantic alignment fix (train what you test) and should move your score upward toward the target band.'
- What this solution (achieved 0.7524) has done: 'The timeout is dominated by (1) repeatedly decoding and transforming large PNGs every epoch and again during threshold fitting, and (2) using a per-sample Python loop in `regress2class_prob` plus slower PIL decode. I keep the exact same model, losses, epochs, and transforms, but make the data pipeline provably equivalent and much faster by caching transformed tensors once on disk (so subsequent epochs/passes reuse them), using OpenCV for faster image decoding while preserving identical RGB values, and ensuring DataLoader workers persist and prefetch efficiently. I also vectorize a small hot loop (`regress2class_prob`) in a mathematically identical way and avoid re-building datasets/loaders unnecessarily. These changes reduce redundant CPU work and I/O without changing training semantics or outputs (beyond negligible float noise).'
- What this solution (achieved 0.0) has done: 'The timeout is dominated by doing 5-fold training of an EfficientNet-B4 for 6 epochs each (30 epochs total) plus repeatedly deserializing cached tensors from disk via `torch.load` on every batch. To stay within 600s without changing the learning objective or model, the main speedups are: (1) reuse the cached `.pt` tensors with a fast, deterministic in-process RAM cache (so each image tensor is loaded from disk at most once), (2) ensure DataLoader workers are actually effective by using a safe multiprocessing context and a collate function that avoids Python per-sample overhead, and (3) avoid redundant work in inference (simpler dataset/loader path and vectorized handling of missing). These changes preserve the exact same transforms, model forward, loss, epochs, folds, and threshold fitting logic—only removing repeated I/O and overhead that caused the timeout.'

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
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.transforms import functional as FT
from PIL import Image, ImageChops

from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold
import timm

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = (
    True  # input shape is constant (512x384 after transforms)
)

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

device = "cuda:0" if torch.cuda.is_available() else "cpu"




## === cell 1
threshold = [0.5, 1.5, 2.5, 3.5]


def regress2class(out):
    thr = out.new_tensor(threshold).view(1, -1)  # (1,4)
    return (out.view(-1, 1) >= thr).sum(dim=1).to(dtype=torch.float32).cpu()


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
    dev = out.device
    out = out.view(-1).to(torch.float32)
    n = out.numel()
    pred_prob = torch.zeros((n, 5), device=dev, dtype=torch.float32)

    is_ge4 = out >= 4.0
    out_clamped = torch.where(is_ge4, torch.full_like(out, 4.0), out)

    l1 = torch.floor(out_clamped).to(torch.long)
    l2 = torch.ceil(out_clamped).to(torch.long)

    w1 = 1.0 - (out_clamped - l1.to(out_clamped.dtype))
    w2 = 1.0 - (l2.to(out_clamped.dtype) - out_clamped)

    idx = torch.arange(n, device=dev)
    pred_prob[idx, l1] = w1
    pred_prob[idx, l2] = torch.maximum(pred_prob[idx, l2], w2)  # safe when l1==l2

    pred_prob[is_ge4, :] = 0.0
    pred_prob[is_ge4, 4] = 1.0
    return pred_prob


def apply_thresholds(preds_continuous, thr):
    preds_continuous = np.asarray(preds_continuous, dtype=np.float32).reshape(-1)
    thr = np.asarray(thr, dtype=np.float32).reshape(-1)
    return (preds_continuous[:, None] >= thr[None, :]).sum(axis=1).astype(np.int64)


def fit_thresholds_by_qwk(
    y_true,
    y_pred_cont,
    init_thr=(0.5, 1.5, 2.5, 3.5),
    iters=6,
    grid_step=0.05,
    window=0.75,
):
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    y_pred_cont = np.asarray(y_pred_cont, dtype=np.float32).reshape(-1)

    thr = np.array(init_thr, dtype=np.float32)

    def score_for_thr(t):
        pred = apply_thresholds(y_pred_cont, t)
        return cohen_kappa_score(y_true, pred, weights="quadratic")

    best_score = score_for_thr(thr)

    for _ in range(iters):
        improved = False
        for k in range(4):
            base = thr[k]
            lo = max(0.0, base - window)
            hi = min(4.5, base + window)
            grid = np.arange(lo, hi + 1e-9, grid_step, dtype=np.float32)

            best_k = base
            best_k_score = best_score
            for v in grid:
                t = thr.copy()
                t[k] = v
                if not (t[0] < t[1] < t[2] < t[3]):
                    continue
                s = score_for_thr(t)
                if s > best_k_score:
                    best_k_score = s
                    best_k = v

            if best_k != base:
                thr[k] = best_k
                best_score = best_k_score
                improved = True
        if not improved:
            break

    return thr.tolist(), float(best_score)




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
def resolve_aptos_root():
    candidates = [
        "../input/aptos2019-blindness-detection",
        "/kaggle/input/aptos2019-blindness-detection",
        "/kaggle/data/aptos2019-blindness-detection",
        "/kaggle/data/input/aptos2019-blindness-detection",
        "/kaggle/data/kaggle/data/aptos2019-blindness-detection",
    ]
    for p in candidates:
        if os.path.exists(os.path.join(p, "train.csv")) and os.path.exists(
            os.path.join(p, "test.csv")
        ):
            return p
    for root in [
        "../input",
        "/kaggle/input",
        "/kaggle/data",
        "/kaggle/data/input",
        "/kaggle/data/kaggle/data",
    ]:
        if not os.path.exists(root):
            continue
        try:
            for d in os.listdir(root):
                p = os.path.join(root, d)
                if os.path.isdir(p) and d == "aptos2019-blindness-detection":
                    if os.path.exists(os.path.join(p, "train.csv")) and os.path.exists(
                        os.path.join(p, "test.csv")
                    ):
                        return p
        except Exception:
            pass
    return None


APTOS_ROOT = resolve_aptos_root()
if APTOS_ROOT is None:
    raise FileNotFoundError(
        "Could not locate aptos2019-blindness-detection dataset root under provided paths."
    )

train_csv_path = os.path.join(APTOS_ROOT, "train.csv")
test_csv_path = os.path.join(APTOS_ROOT, "test.csv")
sample_sub_path = os.path.join(APTOS_ROOT, "sample_submission.csv")

train_img_dir = os.path.join(APTOS_ROOT, "train_images")
test_img_dir = os.path.join(APTOS_ROOT, "test_images")

alt_train_dirs = [
    train_img_dir,
    "../input/train_images",
    "/kaggle/input/aptos2019-blindness-detection/train_images",
    "/kaggle/data/aptos2019-blindness-detection/train_images",
    "/kaggle/data/train_images",
]
train_img_dir = None
for d in alt_train_dirs:
    if os.path.exists(d) and os.path.isdir(d):
        train_img_dir = d
        break
if train_img_dir is None:
    raise FileNotFoundError(
        "Could not locate train_images directory under provided paths."
    )

alt_test_dirs = [
    test_img_dir,
    "../input/test_images",
    "/kaggle/input/aptos2019-blindness-detection/test_images",
    "/kaggle/data/aptos2019-blindness-detection/test_images",
    "/kaggle/data/test_images",
]
test_img_dir = None
for d in alt_test_dirs:
    if os.path.exists(d) and os.path.isdir(d):
        test_img_dir = d
        break
if test_img_dir is None:
    raise FileNotFoundError(
        "Could not locate test_images directory under provided paths."
    )

train_df = pd.read_csv(train_csv_path)
test_df = pd.read_csv(test_csv_path)
test_ids = test_df["id_code"].astype(str).values

input_size = 512
transform = transforms.Compose(
    [
        trim(),
        cropTo4_3(),
        transforms.Resize((input_size * 3 // 4, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.384, 0.258, 0.174], std=[0.124, 0.089, 0.094]),
    ]
)

net = ThreeStage_Model().to(device)




## === cell 5
import cv2
import hashlib
import multiprocessing as mp
from collections import OrderedDict

CACHE_ROOT = "/kaggle/working/aptos_cache_v1"
os.makedirs(CACHE_ROOT, exist_ok=True)


def _safe_cache_key(path: str) -> str:
    h = hashlib.md5(path.encode("utf-8"), usedforsecurity=False).hexdigest()
    return h


def _seed_worker(worker_id):
    s = SEED + worker_id
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def _cache_path_for(image_path: str, split: str) -> str:
    subdir = os.path.join(CACHE_ROOT, f"{split}_tensors")
    os.makedirs(subdir, exist_ok=True)
    return os.path.join(subdir, _safe_cache_key(image_path) + ".pt")


def precompute_cache_for_df(df, img_dir, transform, split: str):
    t0 = time.time()
    n = len(df)
    hit = 0
    miss = 0
    for i in range(n):
        idx = str(df.iloc[i]["id_code"])
        image_path = os.path.join(img_dir, f"{idx}.png")
        cp = _cache_path_for(image_path, split)
        if os.path.exists(cp):
            hit += 1
            continue
        im = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if im is None:
            raise FileNotFoundError(f"Failed to read image: {image_path}")
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(im)
        img_t = transform(img)  # identical transform pipeline
        tmp = cp + ".tmp"
        torch.save(img_t, tmp)
        os.replace(tmp, cp)
        miss += 1
        if (i + 1) % 200 == 0:
            dt = time.time() - t0
            print(
                f"[cache:{split}] {i+1}/{n} | hit={hit} miss={miss} | {dt:.1f}s elapsed"
            )
    print(
        f"[cache:{split}] done {n} | hit={hit} miss={miss} | {time.time()-t0:.1f}s total"
    )


class TensorLRUCache:
    def __init__(self, max_items: int = 4096):
        self.max_items = int(max_items)
        self._d = OrderedDict()

    def get(self, key):
        v = self._d.get(key, None)
        if v is not None:
            self._d.move_to_end(key)
        return v

    def put(self, key, value):
        self._d[key] = value
        self._d.move_to_end(key)
        if len(self._d) > self.max_items:
            self._d.popitem(last=False)


_RAM_CACHE = TensorLRUCache(max_items=4096)


def _load_tensor_cached(path: str) -> torch.Tensor:
    t = _RAM_CACHE.get(path)
    if t is not None:
        return t
    t = torch.load(path, map_location="cpu")
    _RAM_CACHE.put(path, t)
    return t


class CachedTrainDataset(Dataset):
    def __init__(self, df, img_dir):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        idx = str(row["id_code"])
        y = int(row["diagnosis"])
        image_path = os.path.join(self.img_dir, f"{idx}.png")
        cp = _cache_path_for(image_path, "train")
        img_t = _load_tensor_cached(cp)
        return img_t, y


class CachedTestDataset(Dataset):
    def __init__(self, ids, img_dir):
        self.ids = list(ids)
        self.img_dir = img_dir

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        idx = str(self.ids[i])
        image_path = os.path.join(self.img_dir, f"{idx}.png")
        cp = _cache_path_for(image_path, "test")
        if not os.path.exists(cp):
            return idx, None
        img_t = _load_tensor_cached(cp)
        return idx, img_t


def _collate_train(batch):
    xs = [b[0] for b in batch]
    ys = torch.as_tensor([b[1] for b in batch], dtype=torch.long)
    return torch.stack(xs, dim=0), ys


def _collate_test(batch):
    ids = [b[0] for b in batch]
    imgs = [b[1] for b in batch]
    return ids, imgs


def _num_workers_for(device: str) -> int:
    if device == "cpu":
        return 0
    return min(4, (os.cpu_count() or 4))


def train_one_run(model, train_df, img_dir, device):
    model.train()

    num_workers = _num_workers_for(device)
    g = torch.Generator()
    g.manual_seed(SEED)

    dl_kwargs = dict(
        batch_size=4 if device == "cpu" else 8,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device != "cpu"),
        persistent_workers=(num_workers > 0),
        worker_init_fn=_seed_worker if num_workers > 0 else None,
        generator=g,
        drop_last=False,
        collate_fn=_collate_train,
    )
    if num_workers > 0:
        dl_kwargs["prefetch_factor"] = 4
        dl_kwargs["multiprocessing_context"] = mp.get_context("spawn")

    dl = DataLoader(CachedTrainDataset(train_df, img_dir), **dl_kwargs)

    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()

    epochs = 6
    label_scale = 4.5 / 4.0

    for ep in range(epochs):
        t0 = time.time()
        run_loss = 0.0
        n = 0
        for x, y in dl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).float().unsqueeze(1) * label_scale

            opt.zero_grad(set_to_none=True)

            out = model(x, final=True)  # (B,1) in [0,4.5]
            loss = loss_fn(out, y)

            loss.backward()
            opt.step()

            run_loss += float(loss.item()) * x.size(0)
            n += x.size(0)

        print(
            f"epoch {ep+1}/{epochs} - mse: {run_loss/max(1,n):.5f} - time: {time.time()-t0:.1f}s"
        )

    model.eval()
    return model


def predict_continuous_on_df(model, df, img_dir, device):
    model.eval()
    num_workers = _num_workers_for(device)
    dl_kwargs = dict(
        batch_size=8 if device != "cpu" else 4,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device != "cpu"),
        persistent_workers=(num_workers > 0),
        drop_last=False,
        collate_fn=_collate_train,
    )
    if num_workers > 0:
        dl_kwargs["prefetch_factor"] = 4
        dl_kwargs["multiprocessing_context"] = mp.get_context("spawn")

    ds = CachedTrainDataset(df, img_dir)
    dl = DataLoader(ds, **dl_kwargs)

    preds = []
    ys = []
    with torch.inference_mode():
        for x, y in dl:
            x = x.to(device, non_blocking=True)
            out = model(x, final=True).squeeze(1).detach().cpu().numpy()
            preds.append(out)
            ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(preds)


def fit_thresholds_oof(model_ctor, full_df, img_dir, device, n_splits=5):
    y = full_df["diagnosis"].astype(int).values
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    oof_pred = np.zeros(len(full_df), dtype=np.float32)
    last_model = None

    for fold, (tr_idx, va_idx) in enumerate(
        skf.split(np.zeros(len(full_df)), y), start=1
    ):
        print(f"\nOOF fold {fold}/{n_splits}: train={len(tr_idx)} valid={len(va_idx)}")
        fold_model = model_ctor().to(device)

        fold_train_df = full_df.iloc[tr_idx].reset_index(drop=True)
        fold_valid_df = full_df.iloc[va_idx].reset_index(drop=True)

        fold_model = train_one_run(fold_model, fold_train_df, img_dir, device)
        _, pred_va = predict_continuous_on_df(
            fold_model, fold_valid_df, img_dir, device
        )
        oof_pred[va_idx] = pred_va.astype(np.float32)

        if last_model is not None:
            del last_model
        last_model = fold_model  # keep last fold trained model for later reuse
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    learned_thr, oof_qwk = fit_thresholds_by_qwk(
        y_true=y,
        y_pred_cont=oof_pred,
        init_thr=threshold,
        iters=6,
        grid_step=0.05,
        window=0.75,
    )
    return learned_thr, float(oof_qwk), last_model


precompute_cache_for_df(train_df[["id_code"]], train_img_dir, transform, split="train")
precompute_cache_for_df(
    pd.DataFrame({"id_code": test_ids}), test_img_dir, transform, split="test"
)

trained_ckpt_path = "/kaggle/working/trained_b4_3stage_regressor_mse.pt"

learned_thr, oof_qwk, net = fit_thresholds_oof(
    model_ctor=ThreeStage_Model,
    full_df=train_df,
    img_dir=train_img_dir,
    device=device,
    n_splits=5,
)
threshold = learned_thr
print(f"\nLearned thresholds (OOF): {threshold} | OOF QWK: {oof_qwk:.5f}")

torch.save(net.state_dict(), trained_ckpt_path)
print(f"Saved trained checkpoint to: {trained_ckpt_path}")




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _try_get_data(self, timeout)
   1250         try:
-> 1251             data = self._data_queue.get(timeout=timeout)
   1252             return (True, data)

/usr/lib/python3.11/queue.py in get(self, block, timeout)
    179                         raise Empty
--> 180                     self.not_empty.wait(remaining)
    181             item = self._get()

/usr/lib/python3.11/threading.py in wait(self, timeout)
    330                 if timeout > 0:
--> 331                     gotit = waiter.acquire(True, timeout)
    332                 else:

/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/signal_handling.py in handler(signum, frame)
     72         # Python can still get and update the process status successfully.
---> 73         _error_if_any_worker_fails()
     74         if previous_handler is not None:

RuntimeError: DataLoader worker (pid 109) exited unexpectedly with exit code 1. Details are lost due to multiprocessing. Rerunning with num_workers=0 may give better error trace.

The above exception was the direct cause of the following exception:

RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/3036713594.py in <cell line: 0>()
    284 trained_ckpt_path = "/kaggle/working/trained_b4_3stage_regressor_mse.pt"
    285 
--> 286 learned_thr, oof_qwk, net = fit_thresholds_oof(
    287     model_ctor=ThreeStage_Model,
    288     full_df=train_df,

/tmp/ipykernel_55/3036713594.py in fit_thresholds_oof(model_ctor, full_df, img_dir, device, n_splits)
    254         fold_valid_df = full_df.iloc[va_idx].reset_index(drop=True)
    255 
--> 256         fold_model = train_one_run(fold_model, fold_train_df, img_dir, device)
    257         _, pred_va = predict_continuous_on_df(
    258             fold_model, fold_valid_df, img_dir, device

/tmp/ipykernel_55/3036713594.py in train_one_run(model, train_df, img_dir, device)
    185         run_loss = 0.0
    186         n = 0
--> 187         for x, y in dl:
    188             x = x.to(device, non_blocking=True)
    189             y = y.to(device, non_blocking=True).float().unsqueeze(1) * label_scale

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in __next__(self)
    706                 # TODO(https://github.com/pytorch/pytorch/issues/76750)
    707                 self._reset()  # type: ignore[call-arg]
--> 708             data = self._next_data()
    709             self._num_yielded += 1
    710             if (

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _next_data(self)
   1456 
   1457             assert not self._shutdown and self._tasks_outstanding > 0
-> 1458             idx, data = self._get_data()
   1459             self._tasks_outstanding -= 1
   1460             if self._dataset_kind == _DatasetKind.Iterable:

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _get_data(self)
   1408         elif self._pin_memory:
   1409             while self._pin_memory_thread.is_alive():
-> 1410                 success, data = self._try_get_data()
   1411                 if success:
   1412                     return data

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _try_get_data(self, timeout)
   1262             if len(failed_workers) > 0:
   1263                 pids_str = ", ".join(str(w.pid) for w in failed_workers)
-> 1264                 raise RuntimeError(
   1265                     f"DataLoader worker (pid(s) {pids_str}) exited unexpectedly"
   1266                 ) from e

RuntimeError: DataLoader worker (pid(s) 109) exited unexpectedly

## === cell 6
net.eval()

num_workers = _num_workers_for(device)
dl_kwargs = dict(
    batch_size=8 if device != "cpu" else 4,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=(device != "cpu"),
    persistent_workers=(num_workers > 0),
    drop_last=False,
    collate_fn=_collate_test,
)
if num_workers > 0:
    dl_kwargs["prefetch_factor"] = 4
    dl_kwargs["multiprocessing_context"] = mp.get_context("spawn")

dl = DataLoader(CachedTestDataset(test_ids, test_img_dir), **dl_kwargs)

submission = []
missing_images = 0

with torch.inference_mode():
    for ids, imgs in dl:
        valid_ids = []
        valid_imgs = []
        for _id, _img in zip(ids, imgs):
            if _img is None:
                missing_images += 1
                submission.append([str(_id), 0])
            else:
                valid_ids.append(str(_id))
                valid_imgs.append(_img)

        if len(valid_imgs) == 0:
            continue

        x = torch.stack(valid_imgs, dim=0).to(device, non_blocking=True)

        out = net(x, final=True)  # (B,1) in [0,4.5]
        pred = regress2class(out.data.squeeze(1)).numpy().astype(int)

        for _id, p in zip(valid_ids, pred):
            submission.append([_id, int(p)])

print(f"Finished inference. Missing images: {missing_images} / {len(test_ids)}")

if missing_images > 0:
    miss_rate = missing_images / max(1, len(test_ids))
    if miss_rate > 0.01:
        raise RuntimeError(
            f"Too many missing test images ({missing_images}/{len(test_ids)} = {miss_rate:.2%}). "
            f"Check test_img_dir={test_img_dir}"
        )

submission = np.array(submission, dtype=object)




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _try_get_data(self, timeout)
   1250         try:
-> 1251             data = self._data_queue.get(timeout=timeout)
   1252             return (True, data)

/usr/lib/python3.11/queue.py in get(self, block, timeout)
    179                         raise Empty
--> 180                     self.not_empty.wait(remaining)
    181             item = self._get()

/usr/lib/python3.11/threading.py in wait(self, timeout)
    330                 if timeout > 0:
--> 331                     gotit = waiter.acquire(True, timeout)
    332                 else:

/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/signal_handling.py in handler(signum, frame)
     72         # Python can still get and update the process status successfully.
---> 73         _error_if_any_worker_fails()
     74         if previous_handler is not None:

RuntimeError: DataLoader worker (pid 116) exited unexpectedly with exit code 1. Details are lost due to multiprocessing. Rerunning with num_workers=0 may give better error trace.

The above exception was the direct cause of the following exception:

RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/935167822.py in <cell line: 0>()
     21 
     22 with torch.inference_mode():
---> 23     for ids, imgs in dl:
     24         valid_ids = []
     25         valid_imgs = []

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in __next__(self)
    706                 # TODO(https://github.com/pytorch/pytorch/issues/76750)
    707                 self._reset()  # type: ignore[call-arg]
--> 708             data = self._next_data()
    709             self._num_yielded += 1
    710             if (

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _next_data(self)
   1456 
   1457             assert not self._shutdown and self._tasks_outstanding > 0
-> 1458             idx, data = self._get_data()
   1459             self._tasks_outstanding -= 1
   1460             if self._dataset_kind == _DatasetKind.Iterable:

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _get_data(self)
   1408         elif self._pin_memory:
   1409             while self._pin_memory_thread.is_alive():
-> 1410                 success, data = self._try_get_data()
   1411                 if success:
   1412                     return data

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _try_get_data(self, timeout)
   1262             if len(failed_workers) > 0:
   1263                 pids_str = ", ".join(str(w.pid) for w in failed_workers)
-> 1264                 raise RuntimeError(
   1265                     f"DataLoader worker (pid(s) {pids_str}) exited unexpectedly"
   1266                 ) from e

RuntimeError: DataLoader worker (pid(s) 116, 117) exited unexpectedly

## === cell 7
df = pd.DataFrame(submission, columns=["id_code", "diagnosis"])
df["id_code"] = df["id_code"].astype(str)
df["diagnosis"] = df["diagnosis"].astype(int).clip(0, 4)

sample_sub = pd.read_csv(sample_sub_path)
df = sample_sub[["id_code"]].merge(df, on="id_code", how="left")
df["diagnosis"] = df["diagnosis"].fillna(0).astype(int).clip(0, 4)

out_path = "submission.csv"
df.to_csv(out_path, index=False)
print(df.head())
print(f"Wrote {out_path} with shape {df.shape}")
print(f"Using APTOS_ROOT={APTOS_ROOT}")
print(f"Using train_img_dir={train_img_dir}")
print(f"Using test_img_dir={test_img_dir}")
print(f"Using trained checkpoint={trained_ckpt_path}")
print(f"Using learned thresholds={threshold}")
