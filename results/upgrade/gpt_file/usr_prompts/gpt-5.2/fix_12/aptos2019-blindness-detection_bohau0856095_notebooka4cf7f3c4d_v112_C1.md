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

0.5976815189516772

# 6. Current score

0.19733

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I (1) fix the missing weights path by automatically falling back to “no-weights” inference if the file isn’t present, so the notebook always runs end-to-end and produces a CSV. I (2) fix the CUDA/CPU dtype mismatch by moving the model to `device` *before* loading the state dict (and loading the state dict onto the same device), which resolves the “Input type and weight type should be the same” runtime error. I (3) make the inference loop robust to missing/corrupt images (so it can’t silently produce an empty submission) and ensure the output is aligned to `sample_submission.csv` ordering and written as `submission.csv`. These changes are execution/stability fixes; without the competition-provided weights the score may be low, but the pipeline be valid and runnable.'
- What this solution (achieved -0.01401) has done: 'Your current 0.0 score is consistent with running inference from a randomly initialized model because the weights file isn’t present; the smallest legitimate way to move toward the target is to ensure the model uses ImageNet pretrained weights (still the same architecture and inference logic) instead of random init. I keep all heads/combination logic unchanged, but switch the timm backbone creation to `pretrained=True` (and keep your external weights loading if found, which still override). I also remove the unused transform1 to avoid confusion and keep inference using the same `transform2` pipeline you already apply. This should materially increase QWK above 0.0 while preserving core semantics and producing the same valid `submission.csv`.'
- What this solution (achieved -0.02067) has done: 'Your current score is far below the target, so we should improve genuine model signal with minimal semantic changes. The biggest issue is that your `combine3output()` treats the ordinal head incorrectly (it does `argmax` over 4 logits), which breaks the intended ordinal decoding and can easily yield near-random predictions even with a reasonable backbone. I keep the same model, heads, and inference loop, but fix ordinal post-processing to use `sigmoid` + `ordinal2class_prob()` (already defined) and then average probabilities from the three heads before taking `argmax`. This preserves your “combine 3 outputs” core idea while aligning the ordinal/regression outputs to their intended meaning, which should move QWK upward toward the target.'
- What this solution (achieved 0.01609) has done: 'Your current score is far below the target, so we should add a small, metric-aligned improvement that increases signal without changing the model/heads or adding training. The biggest low-risk win for QWK is to calibrate the final class decision via tuned thresholds on the regression head using the training labels (this preserves your existing “3-head average” logic; we only change the final argmax step). I compute optimal 4 cutpoints on a held-out split of `train.csv` by maximizing QWK, then use those cutpoints to convert the averaged expected score into the final integer label for test. This keeps the architecture and inference intact, but aligns discretization to QWK, which should move the score upward toward your target.'
- What this solution (achieved 0.00242) has done: 'Your current score (0.016) is far below the target (0.598), so we should add a minimal, metric-aligned calibration step that preserves your model and inference logic but improves discretization for QWK. Right now cutpoints are tuned on a small random subset with a coarse grid, which can easily underfit the thresholds and leave a lot of QWK on the table. I keep the same 3-head probability averaging and the same tuning approach, but (1) tune cutpoints on out-of-fold (OOF) predictions over the full train set (no training; just inference) and (2) slightly refine the search grid in a very small extra pass; this typically increases QWK materially while keeping semantics intact. I also ensure the tuned cutpoints are used consistently in test inference (already done) and keep runtime under 600s by using a DataLoader and a capped image size (unchanged) with batched inference.'
- What this solution (achieved -0.02936) has done: 'Your current score is far below the target, so we should improve genuine signal with minimal semantic risk. The largest issue is that your regression head is being squashed through a sigmoid and then scaled to 0–4.5, which collapses dynamic range and tends to predict near the center; this harms QWK even after threshold tuning. I keep the exact same model/heads and the same “average 3 head probabilities + cutpoint tuning” pipeline, but change only the regression post-processing to a bounded linear map (clamp) instead of sigmoid, and use the same mapping consistently in both cutpoint tuning and test inference. This should move predictions away from near-constant outputs and improve QWK toward the target while preserving the core logic and producing the same valid submission.csv.'
- What this solution (achieved 0.0) has done: 'Your current score (-0.02936) is far below the target (0.59768), so we should make a minimal change that increases genuine signal without changing the model/heads or adding training. The biggest likely cause is a mismatch between the checkpoint’s expected preprocessing and your current normalization (ImageNet mean/std), which can destroy performance even if weights load correctly; we keep the same resize/crop but switch to EfficientNet’s native `timm` data config (mean/std + interpolation) and apply it consistently for both train-Oof cutpoint tuning and test inference. To preserve your existing logic and avoid introducing new modeling, we won’t touch the architecture, heads, loss, or threshold tuning approach—only the image normalization pipeline. This should move QWK upward toward the target while keeping the submission format identical.'
- What this solution (achieved 0.01308) has done: 'Your current 0.0 score suggests the model is producing nearly constant/garbage predictions, most likely because the regression head is being clamped to [0,4] without matching the checkpoint’s original post-processing, which can collapse signal. To move the score toward the 0.598 target with minimal semantic change, I keep the exact same model/3-head averaging and cutpoint tuning, but switch the regression continuous mapping back to the more common bounded-sigmoid scaling (0–4) and use it consistently in both cutpoint tuning and test inference. I also batch test inference (same transforms/model, just faster/less error-prone) to ensure the full submission is produced reliably within the time limit. No architecture, loss, or training loops are changed; only regression post-processing + inference batching are adjusted to improve QWK.'
- What this solution (achieved 0.19057) has done: 'Your current score is far below the target (gap ≈ -0.585), so we should add a small, metric-aligned improvement without changing the model architecture or training loops. The biggest low-risk win for QWK here is to use the training set to calibrate how we mix the three heads (regression/classification/ordinal) and how we map the mixed expected score into 0–4. I keep your exact backbone + heads and the same inference pipeline, but (1) learn nonnegative weights for the 3-head average on OOF predictions by maximizing QWK and (2) retune cutpoints using those weighted scores. This typically improves signal substantially vs an equal 1/3 average while preserving core semantics and still producing a valid `submission.csv`.'
- What this solution (achieved 0.19733) has done: 'Your current gap to the target is large (0.1906 → 0.5977), so the smallest legitimate way to improve QWK without changing the model/heads is to make the calibration (head-mixing + cutpoints) more robust and less overfit. I keep your exact backbone + 3-head scoring, but tune the head weights on QWK directly (with fixed cuts) and then tune cutpoints afterwards (instead of re-tuning cuts inside every weight candidate), which stabilizes the search and typically improves generalization. I also switch calibration from in-sample to simple 3-fold OOF (same inference, no training) so the tuned parameters are less biased and more likely to help the leaderboard score. All paths and submission format remain unchanged and the script still runs end-to-end within the time budget.'

# 9. Code solution

## === cell 0
import os
import math
import random
import time
import glob
import copy
import json
import pickle
import csv
import warnings

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as transforms
from torchvision.transforms import functional as FT

from PIL import Image, ImageChops

from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold
import timm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

print("device:", device)



## === cell 1
from torch import Tensor
from torch.jit.annotations import List, Optional, Tuple


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


class backboneNet_efficient(nn.Module):
    """
    Core logic unchanged: EfficientNet-B4 backbone -> global pooling -> 3 heads.

    Uses ImageNet pretrained weights so that when competition weights are not found,
    the backbone is not randomly initialized.
    """

    def __init__(self):
        super(backboneNet_efficient, self).__init__()
        self.backbone = timm.create_model(
            "tf_efficientnet_b4_ns",
            pretrained=True,
            features_only=True,
            out_indices=(4,),
        )
        self.num_features = self.backbone.feature_info.channels()[-1]

        self.global_pool = GeM(flatten=True)
        self.drop_rate = 0.4

        self.rg_cls = Linear(self.num_features, 1, bias=True)
        self.cls_cls = Linear(self.num_features, 5, bias=True)
        self.ord_cls = Linear(self.num_features, 4, bias=True)

    def forward(self, x):
        feats = self.backbone(x)[0]
        x = self.global_pool(feats)
        if self.drop_rate > 0.0:
            x = F.dropout(x, p=self.drop_rate, training=self.training)

        r_out = self.rg_cls(x)
        c_out = self.cls_cls(x)
        o_out = self.ord_cls(x)
        return r_out, c_out, o_out




## === cell 2
threshold = [0.75, 1.5, 2.5, 3.5]


def regress2class(out):
    prediction = 0
    for i in range(4):
        prediction += (out.data >= threshold[i]).squeeze().cpu().item()
    return int(prediction)


def ordinal2class_prob(out):
    pred_prob = torch.zeros(out.size(0), 5, device=out.device, dtype=out.dtype)
    pred_prob[:, 0] = (1 - out[:, 0]).squeeze()
    pred_prob[:, 1] = (out[:, 0] * (1 - out[:, 1])).squeeze()
    pred_prob[:, 2] = (out[:, 1] * (1 - out[:, 2])).squeeze()
    pred_prob[:, 3] = (out[:, 2] * (1 - out[:, 3])).squeeze()
    pred_prob[:, 4] = out[:, 3].squeeze()
    return F.softmax(pred_prob, dim=1)


def regress2class_prob(out):
    pred_prob = torch.zeros((out.size(0), 5), device=out.device, dtype=out.dtype)
    for i in range(out.size(0)):
        if out[i] < 4.0:
            l1 = int(math.floor(out[i].item()))
            l2 = int(math.ceil(out[i].item()))
            pred_prob[i][l1] = 1 - (out[i] - l1)
            pred_prob[i][l2] = 1 - (l2 - out[i])
        else:
            pred_prob[i][4] = 1.0
    return pred_prob


def score_from_prob(prob: torch.Tensor) -> torch.Tensor:
    w = torch.arange(5, device=prob.device, dtype=prob.dtype).view(1, -1)
    return (prob * w).sum(dim=1)


def apply_cutpoints(score: np.ndarray, cuts) -> np.ndarray:
    cuts = np.asarray(cuts, dtype=np.float32)
    pred = np.zeros_like(score, dtype=np.int64)
    pred += score >= cuts[0]
    pred += score >= cuts[1]
    pred += score >= cuts[2]
    pred += score >= cuts[3]
    return np.clip(pred, 0, 4)


def tune_cutpoints(
    y_true: np.ndarray,
    score: np.ndarray,
    init_cuts=None,
    iters=2,
    grid_n=25,
    widths=(0.45, 0.60, 0.60, 0.45),
):
    y_true = np.asarray(y_true, dtype=np.int64)
    score = np.asarray(score, dtype=np.float32)

    if init_cuts is None:
        cuts = np.array([0.75, 1.5, 2.5, 3.5], dtype=np.float32)
    else:
        cuts = np.array(init_cuts, dtype=np.float32)

    for _ in range(iters):
        for i in range(4):
            base = float(cuts[i])
            width = float(widths[i])
            grid = np.linspace(base - width, base + width, grid_n).astype(np.float32)

            best_k = -1e9
            best_v = base
            for v in grid:
                tmp = cuts.copy()
                tmp[i] = v
                tmp = np.maximum.accumulate(
                    tmp + np.array([0.0, 1e-3, 2e-3, 3e-3], dtype=np.float32)
                )
                tmp = np.clip(tmp, 0.05, 3.95)

                pred = apply_cutpoints(score, tmp)
                k = cohen_kappa_score(y_true, pred, weights="quadratic")
                if k > best_k:
                    best_k = k
                    best_v = float(tmp[i])

            cuts[i] = best_v
            cuts = np.maximum.accumulate(
                cuts + np.array([0.0, 1e-3, 2e-3, 3e-3], dtype=np.float32)
            )
            cuts = np.clip(cuts, 0.05, 3.95)

    final_pred = apply_cutpoints(score, cuts)
    final_k = cohen_kappa_score(y_true, final_pred, weights="quadratic")
    return cuts.tolist(), float(final_k)


def regress_raw_to_continuous(r_out: torch.Tensor) -> torch.Tensor:
    r = torch.sigmoid(r_out) * 4.0
    return torch.clamp(r, 0.0, 4.0)


def combine3output(r_out, c_out, o_out, cutpoints=None):
    """
    Keeps the same 3-head probability averaging.
    Uses tuned cutpoints on expected score when provided (QWK-aligned discretization).
    """
    c_prob = F.softmax(c_out, dim=1)

    r_cont = regress_raw_to_continuous(r_out)
    r_cont = r_cont.squeeze(1) if r_cont.ndim == 2 else r_cont
    r_prob = regress2class_prob(r_cont)

    o_prob = ordinal2class_prob(torch.sigmoid(o_out))

    prob = (r_prob + c_prob + o_prob) / 3.0

    if cutpoints is None:
        pred = int(torch.argmax(prob, dim=1).item())
        return max(0, min(4, pred))

    s = float(score_from_prob(prob).detach().cpu().numpy()[0])
    pred = int(apply_cutpoints(np.array([s], dtype=np.float32), cutpoints)[0])
    return max(0, min(4, pred))


def tune_head_weights_fixed_cuts(
    y_true: np.ndarray,
    s_r: np.ndarray,
    s_c: np.ndarray,
    s_o: np.ndarray,
    cuts,
):
    """
    Change is directly score-improving/stabilizing: we tune (wr,wc,wo) against QWK
    with a fixed discretization (cuts), avoiding re-tuning cuts inside each weight
    candidate (which overfits and is noisy). Core logic (weighted mixing) is unchanged.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    s_r = np.asarray(s_r, dtype=np.float32)
    s_c = np.asarray(s_c, dtype=np.float32)
    s_o = np.asarray(s_o, dtype=np.float32)

    best_k = -1e9
    best_w = (1 / 3, 1 / 3, 1 / 3)

    for wr_i in range(0, 11):
        wr = wr_i / 10.0
        for wc_i in range(0, 11 - wr_i):
            wc = wc_i / 10.0
            wo = 1.0 - wr - wc
            if wo < 0:
                continue
            score = wr * s_r + wc * s_c + wo * s_o
            pred = apply_cutpoints(score, cuts)
            k = cohen_kappa_score(y_true, pred, weights="quadratic")
            if k > best_k:
                best_k = k
                best_w = (wr, wc, wo)

    wr0, wc0, wo0 = best_w
    candidates = []
    for dwr in np.linspace(-0.10, 0.10, 11):
        for dwc in np.linspace(-0.10, 0.10, 11):
            wr = wr0 + float(dwr)
            wc = wc0 + float(dwc)
            wo = 1.0 - wr - wc
            if wr < 0 or wc < 0 or wo < 0:
                continue
            candidates.append((wr, wc, wo))

    for wr, wc, wo in candidates:
        score = wr * s_r + wc * s_c + wo * s_o
        pred = apply_cutpoints(score, cuts)
        k = cohen_kappa_score(y_true, pred, weights="quadratic")
        if k > best_k:
            best_k = k
            best_w = (float(wr), float(wc), float(wo))

    return best_w, float(best_k)




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
DATA_DIR = "../input/aptos2019-blindness-detection"
if not os.path.exists(DATA_DIR):
    alt = "/kaggle/input/aptos2019-blindness-detection"
    if os.path.exists(alt):
        DATA_DIR = alt

TEST_CSV = os.path.join(DATA_DIR, "test.csv")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")
SAMPLE_SUB = os.path.join(DATA_DIR, "sample_submission.csv")
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images")

WEIGHTS_PATH = "../input/weights/2.pth"
WEIGHTS_CANDIDATES = [
    WEIGHTS_PATH,
    "/kaggle/input/weights/2.pth",
    os.path.join(DATA_DIR, "2.pth"),
    os.path.join(DATA_DIR, "weights", "2.pth"),
]

test_df = pd.read_csv(TEST_CSV)
test_ids = test_df["id_code"].astype(str).values

data_cfg = timm.data.resolve_model_data_config(
    timm.create_model("tf_efficientnet_b4_ns", pretrained=False)
)
mean = data_cfg.get("mean", (0.485, 0.456, 0.406))
std = data_cfg.get("std", (0.229, 0.224, 0.225))
interp = data_cfg.get("interpolation", "bicubic")

transform2 = transforms.Compose(
    [
        transforms.Resize(
            (280, 280),
            interpolation=(
                transforms.InterpolationMode.BICUBIC
                if str(interp).lower().startswith("bi")
                else transforms.InterpolationMode.BILINEAR
            ),
        ),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
)

net3 = backboneNet_efficient().to(device)
net3.eval()

found_weights = None
for p in WEIGHTS_CANDIDATES:
    if p and os.path.exists(p):
        found_weights = p
        break

if found_weights is None:
    warnings.warn(
        f"Could not find weights file. Tried: {WEIGHTS_CANDIDATES}. "
        "Proceeding with ImageNet-pretrained backbone; submission will be valid."
    )
else:
    state = torch.load(found_weights, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict):
        new_state = {}
        for k, v in state.items():
            nk = k
            if nk.startswith("module."):
                nk = nk[len("module.") :]
            new_state[nk] = v
        state = new_state

    missing, unexpected = net3.load_state_dict(state, strict=False)
    print("Loaded weights from:", found_weights)
    print("Missing keys:", len(missing), "Unexpected keys:", len(unexpected))




## === cell 5
class TrainImageDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_dir: str, transform):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        id_code = str(row["id_code"])
        y = int(row["diagnosis"])
        img_path = os.path.join(self.img_dir, f"{id_code}.png")
        if not os.path.exists(img_path):
            return id_code, None, y
        try:
            img = Image.open(img_path).convert("RGB")
            x = self.transform(img)
            return id_code, x, y
        except Exception:
            return id_code, None, y


def collate_skip_missing(batch):
    ids, xs, ys = zip(*batch)
    keep = [i for i, x in enumerate(xs) if x is not None]
    if len(keep) == 0:
        return [], None, torch.empty((0,), dtype=torch.long)
    ids2 = [ids[i] for i in keep]
    x2 = torch.stack([xs[i] for i in keep], dim=0)
    y2 = torch.tensor([ys[i] for i in keep], dtype=torch.long)
    return ids2, x2, y2




## === cell 6
tuned_cutpoints = None
tuned_head_weights = (1 / 3, 1 / 3, 1 / 3)

if os.path.exists(TRAIN_CSV) and os.path.exists(TRAIN_IMG_DIR):
    train_df = pd.read_csv(TRAIN_CSV)
    train_df["id_code"] = train_df["id_code"].astype(str)
    train_df["diagnosis"] = train_df["diagnosis"].astype(int)

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    oof_targets = []
    oof_sr = []
    oof_sc = []
    oof_so = []
    missing_or_failed = 0

    net3.eval()
    t0 = time.time()

    for fold, (_, val_idx) in enumerate(skf.split(train_df, train_df["diagnosis"])):
        val_df = train_df.iloc[val_idx].reset_index(drop=True)
        ds = TrainImageDataset(val_df, TRAIN_IMG_DIR, transform2)

        bs = 16 if device.type == "cuda" else 8
        loader = DataLoader(
            ds,
            batch_size=bs,
            shuffle=False,
            num_workers=2,
            pin_memory=(device.type == "cuda"),
            collate_fn=collate_skip_missing,
        )

        with torch.no_grad():
            for b, (ids_b, x_b, y_b) in enumerate(loader):
                if x_b is None or len(ids_b) == 0:
                    missing_or_failed += len(y_b)
                    continue

                x_b = x_b.to(device, non_blocking=True)
                r_out, c_out, o_out = net3(x_b)

                c_prob = F.softmax(c_out, dim=1)

                r_cont = regress_raw_to_continuous(r_out)
                r_cont = r_cont.squeeze(1) if r_cont.ndim == 2 else r_cont
                r_prob = regress2class_prob(r_cont)

                o_prob = ordinal2class_prob(torch.sigmoid(o_out))

                sr = score_from_prob(r_prob).detach().cpu().numpy().astype(np.float32)
                sc = score_from_prob(c_prob).detach().cpu().numpy().astype(np.float32)
                so = score_from_prob(o_prob).detach().cpu().numpy().astype(np.float32)

                oof_sr.append(sr)
                oof_sc.append(sc)
                oof_so.append(so)
                oof_targets.append(y_b.numpy().astype(np.int64))

        print(f"OOF fold {fold+1}/3 done")

    if len(oof_targets) > 0:
        oof_targets = np.concatenate(oof_targets, axis=0)
        oof_sr = np.concatenate(oof_sr, axis=0)
        oof_sc = np.concatenate(oof_sc, axis=0)
        oof_so = np.concatenate(oof_so, axis=0)
    else:
        oof_targets = np.array([], dtype=np.int64)
        oof_sr = np.array([], dtype=np.float32)
        oof_sc = np.array([], dtype=np.float32)
        oof_so = np.array([], dtype=np.float32)

    print("OOF calibration elapsed sec:", round(time.time() - t0, 2))

    if len(oof_targets) >= 200:
        base_score = (oof_sr + oof_sc + oof_so) / 3.0

        cuts1, k1 = tune_cutpoints(
            oof_targets, base_score, init_cuts=threshold, iters=2, grid_n=29
        )
        cuts2, k2 = tune_cutpoints(
            oof_targets,
            base_score,
            init_cuts=cuts1,
            iters=1,
            grid_n=41,
            widths=(0.20, 0.25, 0.25, 0.20),
        )

        w, kw = tune_head_weights_fixed_cuts(oof_targets, oof_sr, oof_sc, oof_so, cuts2)

        weighted_score = w[0] * oof_sr + w[1] * oof_sc + w[2] * oof_so
        cuts_w, k_wcuts = tune_cutpoints(
            oof_targets,
            weighted_score,
            init_cuts=cuts2,
            iters=2,
            grid_n=41,
            widths=(0.20, 0.25, 0.25, 0.20),
        )

        tuned_head_weights = w
        tuned_cutpoints = cuts_w

        print(
            "Calibration: used",
            len(oof_targets),
            "OOF train images; missing/failed:",
            missing_or_failed,
        )
        print("Equal-weight coarse cuts:", cuts1, "QWK:", k1)
        print("Equal-weight refined cuts:", cuts2, "QWK:", k2)
        print(
            "Tuned head weights (wr,wc,wo):", tuned_head_weights, "QWK@fixedcuts:", kw
        )
        print("Tuned cutpoints on weighted score:", tuned_cutpoints, "QWK:", k_wcuts)
    else:
        warnings.warn(
            f"Not enough training images to tune cutpoints/weights (got {len(oof_targets)}); falling back to argmax(prob)."
        )
else:
    warnings.warn(
        "train.csv/train_images not found; cannot tune cutpoints. Falling back to argmax(prob)."
    )




## === cell 7
class TestImageDataset(Dataset):
    def __init__(self, ids, img_dir: str, transform):
        self.ids = [str(x) for x in ids]
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        id_code = self.ids[i]
        img_path = os.path.join(self.img_dir, f"{id_code}.png")
        if not os.path.exists(img_path):
            return id_code, None
        try:
            img = Image.open(img_path).convert("RGB")
            x = self.transform(img)
            return id_code, x
        except Exception:
            return id_code, None


def collate_test_skip_missing(batch):
    ids, xs = zip(*batch)
    keep = [i for i, x in enumerate(xs) if x is not None]
    if len(keep) == 0:
        return [], None, ids  # ids_all returned for accounting
    ids2 = [ids[i] for i in keep]
    x2 = torch.stack([xs[i] for i in keep], dim=0)
    return ids2, x2, ids


submission_dict = {}
missing_images = 0
failed_images = 0

test_ds = TestImageDataset(test_ids, TEST_IMG_DIR, transform2)
test_loader = DataLoader(
    test_ds,
    batch_size=32 if device.type == "cuda" else 16,
    shuffle=False,
    num_workers=2,
    pin_memory=(device.type == "cuda"),
    collate_fn=collate_test_skip_missing,
)

wr, wc, wo = tuned_head_weights

net3.eval()
with torch.no_grad():
    seen = 0
    for b, (ids_b, x_b, ids_all_b) in enumerate(test_loader):
        seen += len(ids_all_b)
        if b % 10 == 0:
            print(seen, "/", len(test_ids))

        if x_b is None or len(ids_b) == 0:
            missing_images += len(ids_all_b)
            continue

        missing_images += len(ids_all_b) - len(ids_b)

        x_b = x_b.to(device, non_blocking=True)
        r_out, c_out, o_out = net3(x_b)

        c_prob = F.softmax(c_out, dim=1)

        r_cont = regress_raw_to_continuous(r_out)
        r_cont = r_cont.squeeze(1) if r_cont.ndim == 2 else r_cont
        r_prob = regress2class_prob(r_cont)

        o_prob = ordinal2class_prob(torch.sigmoid(o_out))

        prob = wr * r_prob + wc * c_prob + wo * o_prob

        if tuned_cutpoints is None:
            preds = torch.argmax(prob, dim=1).detach().cpu().numpy().astype(np.int64)
        else:
            s = score_from_prob(prob).detach().cpu().numpy().astype(np.float32)
            preds = apply_cutpoints(s, tuned_cutpoints).astype(np.int64)

        for _id, _p in zip(ids_b, preds):
            submission_dict[_id] = int(max(0, min(4, _p)))

print("missing_images:", missing_images, "failed_images:", failed_images)
print("Using head weights (wr,wc,wo):", tuned_head_weights)
print("Using cutpoints:", tuned_cutpoints)



## === cell 8
rows = []
for idx in test_ids:
    rows.append([str(idx), int(submission_dict.get(str(idx), 0))])

df = pd.DataFrame(rows, columns=["id_code", "diagnosis"])
df["id_code"] = df["id_code"].astype(str)
df["diagnosis"] = df["diagnosis"].astype(int).clip(0, 4)

if os.path.exists(SAMPLE_SUB):
    sample = pd.read_csv(SAMPLE_SUB)
    df = sample[["id_code"]].merge(df, on="id_code", how="left")
    df["diagnosis"] = df["diagnosis"].fillna(0).astype(int).clip(0, 4)
else:
    df = df.sort_values("id_code").reset_index(drop=True)

assert len(df) > 0, "Submission DataFrame should not be empty"
assert list(df.columns) == ["id_code", "diagnosis"]

out_path = "submission.csv"
df.to_csv(out_path, index=False)
print("Wrote:", out_path, "rows:", len(df))
print(df.head())
