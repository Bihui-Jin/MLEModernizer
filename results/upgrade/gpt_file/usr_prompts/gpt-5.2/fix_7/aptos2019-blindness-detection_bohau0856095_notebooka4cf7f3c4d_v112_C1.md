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

0.00242

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I (1) fix the missing weights path by automatically falling back to “no-weights” inference if the file isn’t present, so the notebook always runs end-to-end and produces a CSV. I (2) fix the CUDA/CPU dtype mismatch by moving the model to `device` *before* loading the state dict (and loading the state dict onto the same device), which resolves the “Input type and weight type should be the same” runtime error. I (3) make the inference loop robust to missing/corrupt images (so it can’t silently produce an empty submission) and ensure the output is aligned to `sample_submission.csv` ordering and written as `submission.csv`. These changes are execution/stability fixes; without the competition-provided weights the score may be low, but the pipeline be valid and runnable.'
- What this solution (achieved -0.01401) has done: 'Your current 0.0 score is consistent with running inference from a randomly initialized model because the weights file isn’t present; the smallest legitimate way to move toward the target is to ensure the model uses ImageNet pretrained weights (still the same architecture and inference logic) instead of random init. I keep all heads/combination logic unchanged, but switch the timm backbone creation to `pretrained=True` (and keep your external weights loading if found, which still override). I also remove the unused transform1 to avoid confusion and keep inference using the same `transform2` pipeline you already apply. This should materially increase QWK above 0.0 while preserving core semantics and producing the same valid `submission.csv`.'
- What this solution (achieved -0.02067) has done: 'Your current score is far below the target, so we should improve genuine model signal with minimal semantic changes. The biggest issue is that your `combine3output()` treats the ordinal head incorrectly (it does `argmax` over 4 logits), which breaks the intended ordinal decoding and can easily yield near-random predictions even with a reasonable backbone. I keep the same model, heads, and inference loop, but fix ordinal post-processing to use `sigmoid` + `ordinal2class_prob()` (already defined) and then average probabilities from the three heads before taking `argmax`. This preserves your “combine 3 outputs” core idea while aligning the ordinal/regression outputs to their intended meaning, which should move QWK upward toward the target.'
- What this solution (achieved 0.01609) has done: 'Your current score is far below the target, so we should add a small, metric-aligned improvement that increases signal without changing the model/heads or adding training. The biggest low-risk win for QWK is to calibrate the final class decision via tuned thresholds on the regression head using the training labels (this preserves your existing “3-head average” logic; we only change the final argmax step). I compute optimal 4 cutpoints on a held-out split of `train.csv` by maximizing QWK, then use those cutpoints to convert the averaged expected score into the final integer label for test. This keeps the architecture and inference intact, but aligns discretization to QWK, which should move the score upward toward your target.'
- What this solution (achieved 0.00242) has done: 'Your current score (0.016) is far below the target (0.598), so we should add a minimal, metric-aligned calibration step that preserves your model and inference logic but improves discretization for QWK. Right now cutpoints are tuned on a small random subset with a coarse grid, which can easily underfit the thresholds and leave a lot of QWK on the table. I keep the same 3-head probability averaging and the same tuning approach, but (1) tune cutpoints on out-of-fold (OOF) predictions over the full train set (no training; just inference) and (2) slightly refine the search grid in a very small extra pass; this typically increases QWK materially while keeping semantics intact. I also ensure the tuned cutpoints are used consistently in test inference (already done) and keep runtime under 600s by using a DataLoader and a capped image size (unchanged) with batched inference.'

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

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import Dataset, DataLoader, random_split

import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import functional as FT

from PIL import Image, ImageChops, ImageEnhance, ImageOps

import cv2

from sklearn.metrics import cohen_kappa_score
import timm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

print("device:", device)



## === cell 1
import warnings
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

    Change (score improvement toward target):
    - Use ImageNet pretrained weights (pretrained=True) so that when competition weights
      are not found, the backbone is not randomly initialized (which yields ~0 score).
    - If a competition weights file is found, we still load it afterward (overrides).
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
    """
    Change (score improvement toward target, core semantics preserved):
    - Keep the same "maximize QWK by adjusting 4 cutpoints" approach, but allow:
      * More robust tuning by using more data (we'll pass OOF scores from full train)
      * Slightly finer search grid on the same simple coordinate-descent procedure
    """
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


def combine3output(r_out, c_out, o_out, cutpoints=None):
    """
    Keeps the same 3-head probability averaging.
    Uses tuned cutpoints on expected score when provided (QWK-aligned discretization).
    """
    c_prob = F.softmax(c_out, dim=1)

    r_cont = torch.sigmoid(r_out) * 4.5
    r_prob = regress2class_prob(r_cont.squeeze(1) if r_cont.ndim == 2 else r_cont)

    o_prob = ordinal2class_prob(torch.sigmoid(o_out))

    prob = (r_prob + c_prob + o_prob) / 3.0

    if cutpoints is None:
        pred = int(torch.argmax(prob, dim=1).item())
        return max(0, min(4, pred))

    s = float(score_from_prob(prob).detach().cpu().numpy()[0])
    pred = int(apply_cutpoints(np.array([s], dtype=np.float32), cutpoints)[0])
    return max(0, min(4, pred))




## === cell 3
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




## === cell 4
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




## === cell 5
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

transform2 = transforms.Compose(
    [
        transforms.Resize((280, 280)),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
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
        "Proceeding with ImageNet-pretrained backbone; submission will be valid and score should improve vs random init."
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




## === cell 6
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




## === cell 7
tuned_cutpoints = None

if os.path.exists(TRAIN_CSV) and os.path.exists(TRAIN_IMG_DIR):
    train_df = pd.read_csv(TRAIN_CSV)
    train_df["id_code"] = train_df["id_code"].astype(str)
    train_df["diagnosis"] = train_df["diagnosis"].astype(int)

    ds = TrainImageDataset(train_df, TRAIN_IMG_DIR, transform2)

    bs = 16 if device.type == "cuda" else 8
    loader = DataLoader(
        ds,
        batch_size=bs,
        shuffle=False,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_skip_missing,
    )

    all_scores = []
    all_targets = []
    missing_or_failed = 0

    net3.eval()
    with torch.no_grad():
        for b, (ids_b, x_b, y_b) in enumerate(loader):
            if x_b is None or len(ids_b) == 0:
                missing_or_failed += len(y_b)
                continue

            x_b = x_b.to(device, non_blocking=True)
            r_out, c_out, o_out = net3(x_b)

            c_prob = F.softmax(c_out, dim=1)

            r_cont = torch.sigmoid(r_out) * 4.5
            r_prob = regress2class_prob(
                r_cont.squeeze(1) if r_cont.ndim == 2 else r_cont
            )

            o_prob = ordinal2class_prob(torch.sigmoid(o_out))

            prob = (r_prob + c_prob + o_prob) / 3.0
            s = score_from_prob(prob).detach().cpu().numpy().astype(np.float32)

            all_scores.append(s)
            all_targets.append(y_b.numpy().astype(np.int64))

            if b % 50 == 0:
                print("OOF inference batches:", b)

    if len(all_targets) > 0:
        all_scores = np.concatenate(all_scores, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
    else:
        all_scores = np.array([], dtype=np.float32)
        all_targets = np.array([], dtype=np.int64)

    if len(all_targets) >= 200:
        cuts1, k1 = tune_cutpoints(
            all_targets, all_scores, init_cuts=threshold, iters=2, grid_n=29
        )
        cuts2, k2 = tune_cutpoints(
            all_targets,
            all_scores,
            init_cuts=cuts1,
            iters=1,
            grid_n=41,
            widths=(0.20, 0.25, 0.25, 0.20),
        )
        tuned_cutpoints = cuts2
        print(
            "Cutpoint tuning: used",
            len(all_targets),
            "train images; missing/failed:",
            missing_or_failed,
        )
        print("Coarse cuts:", cuts1, "QWK:", k1)
        print("Refined cuts:", cuts2, "QWK:", k2)
    else:
        warnings.warn(
            f"Not enough training images to tune cutpoints (got {len(all_targets)}); falling back to argmax(prob)."
        )
else:
    warnings.warn(
        "train.csv/train_images not found; cannot tune cutpoints. Falling back to argmax(prob)."
    )



## === cell 8
submission = []
missing_images = 0
failed_images = 0

with torch.no_grad():
    for i, idx in enumerate(test_ids):
        if i % 25 == 0:
            print(i, "/", len(test_ids))

        image_name = os.path.join(TEST_IMG_DIR, f"{idx}.png")
        if not os.path.exists(image_name):
            missing_images += 1
            submission.append([idx, 0])
            continue

        try:
            img = Image.open(image_name).convert("RGB")
            img2 = transform2(img).unsqueeze(0).to(device)

            r_out, c_out, o_out = net3(img2)
            pred3 = combine3output(r_out, c_out, o_out, cutpoints=tuned_cutpoints)
            submission.append([idx, pred3])
        except Exception:
            failed_images += 1
            submission.append([idx, 0])

print("missing_images:", missing_images, "failed_images:", failed_images)
submission = np.array(submission, dtype=object)



## === cell 9
df = pd.DataFrame(submission, columns=["id_code", "diagnosis"])
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
