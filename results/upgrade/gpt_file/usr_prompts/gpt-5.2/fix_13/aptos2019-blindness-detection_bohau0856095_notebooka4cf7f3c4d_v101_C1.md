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

0.9121463446464624

# 6. Current score

0.68726

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix three execution blockers while keeping the model and prediction logic unchanged: (1) make the script run on CPU when no GPU is available, (2) robustly locate the pretrained weights file (or fail fast with a clear message if it’s truly missing), and (3) fix small transform/utility bugs that can silently return `None` (trim) or use incorrect string comparison (`is` vs `==`). I also make the inference loop tolerant to missing/corrupt images and ensure `submission.csv` is always written with the correct columns and row count matching `test.csv`. These changes are score-neutral (they don’t change the architecture, thresholds, or inference semantics) but ensure you get a valid submission file end-to-end.'
- What this solution (achieved 0.00014) has done: 'I fix the runtime blocker by making the weights loading robust: if the specified external weights file isn’t present (common on Kaggle unless an extra dataset is attached), the script fall back to using the model’s built-in pretrained ImageNet weights so it still runs end-to-end and produces a non-degenerate submission (improving score from 0.0 toward the target). I keep the model architecture and inference logic the same (still using the regressor head + fixed thresholds) and only adjust the `pretrained` flag and state-dict loading behavior. I also fix the current inference call to use the correct output for regression (`r_out` returned by `net(img)` is already scaled), and ensure the submission is always aligned to `test.csv` with correct columns. These changes are minimal, unblock execution, and should substantially improve score versus the current all-zeros / non-running state.'
- What this solution (achieved 0.06369) has done: 'Your score is far below the target, and the biggest issue is that (when custom weights are missing) the model’s heads are randomly initialized, producing near-random ordinal outputs and a near-zero QWK. To move the score toward the target without changing the model architecture or training, I add a minimal, metric-aligned post-processing step: fit optimal regression-to-class thresholds on the provided training set using out-of-fold (stratified) predictions, then apply those learned thresholds to test predictions. This keeps your existing backbone/model/inference logic intact (still using `r_out` and converting to classes), but replaces the hardcoded thresholds with data-driven ones, which is a standard way to improve QWK for this competition. I also make sure we run efficiently within time by caching computed image tensors for threshold fitting and using batch inference (no change to model, just I/O efficiency).'
- What this solution (achieved 0.74374) has done: 'Your current score is far below the target, so we should improve (not degrade) performance with minimal risk while keeping the same model and regression→threshold→class core logic. The biggest likely issue is that you’re running with randomly initialized heads when the custom weight file is missing; I add a minimal, legitimate “heads-only” training step on the provided training images to learn those heads while keeping the EfficientNet backbone frozen (architecture unchanged). Then we keep your existing out-of-fold threshold fitting, but make it consistent with the newly trained heads by reusing the same inference path. This should move QWK substantially upward toward your target without changing the model structure, loss type for inference, or submission semantics.'
- What this solution (achieved 0.74046) has done: 'We keep your model, training loop, and regression→threshold→class logic intact, but improve the “heads-only” training signal to better match the QWK/ordinal nature of the task while staying minimal. Specifically, we (1) compute regression loss against the class *midpoints* (0.0–4.0) but with a small label-smoothing style offset via expected value from the classifier probabilities, (2) train the `final_regressor` jointly by actually using the `final=True` path during training, and (3) fit thresholds using out-of-fold predictions from that final regressor output (same thresholding step, just on the intended final output). These are small, metric-aligned tweaks that typically move QWK upward from your current 0.7437 without changing architecture or adding early stopping/sampling, and still finish within the time budget. Submission writing/format stays identical.'
- What this solution (achieved 0.72466) has done: 'I keep your model architecture and the overall “heads-only train → OOF threshold fit → test inference → threshold to class” pipeline unchanged, but make two minimal, metric-aligned fixes that typically improve QWK a lot for this competition. First, I compute OOF predictions using the *same* post-training model (not a potentially miscalibrated snapshot) by training the heads inside each fold on the fold’s training split, then predicting on that fold’s validation split; this avoids threshold fitting on biased predictions and usually increases QWK toward your target. Second, I make the heads-only training deterministic and more stable by switching the regressor targets from “smoothed class” to the true class midpoints (0–4) while still keeping your existing losses/loops/heads intact, which tends to improve ordinal calibration without changing inference semantics. All paths and submission format remain identical, and the script still writes `submission.csv` end-to-end within the time budget.'
- What this solution (achieved 0.70895) has done: 'The timeout is dominated by (1) loading and transforming ~3.3k large PNGs sequentially in Python/PIL, and (2) doing 5-fold inference/training by repeatedly stacking Python lists of tensors. I keep the exact same model, losses, and threshold-fitting logic, but make data input and batching fast by using a proper `Dataset` + multi-worker `DataLoader`, precomputing train tensors once into a contiguous array, and using pinned-memory + non_blocking GPU copies. I also vectorize the few remaining Python loops that are purely postprocessing (threshold application / regress2class_prob) to remove overhead without changing outputs. Finally, I avoid reloading weights repeatedly inside the CV loop by caching the loaded state dict once (same bytes, same semantics).'
- What this solution (achieved 0.70895) has done: 'We’re far below the target (0.70895 vs 0.9121), so we should improve, but with the smallest changes that keep your “heads-only train → OOF threshold fit → final-regressor inference → threshold to class” core pipeline intact. The biggest score limiter in your current code is that the fold models used for OOF threshold fitting are starting from scratch (no ImageNet backbone) when the custom weight file is missing, because `use_pretrained_backbone` is a single global flag; this makes OOF predictions low-quality and leads to poor learned thresholds. I make a minimal fix: always initialize fold models with an ImageNet-pretrained backbone when no custom weights are provided (so OOF threshold fitting is meaningful), and I also ensure the main model uses `pretrained_backbone=True` before heads-only training in that same case. Finally, I keep everything else the same, only adding a tiny safety clamp of regression outputs to [0, 4.5] before thresholding (doesn’t change semantics, but avoids rare numerical spillover hurting kappa).'
- What this solution (achieved 0.71154) has done: 'I make a minimal, metric-aligned improvement by ensuring the same post-training calibration is used for test as for the out-of-fold threshold fitting: after learning thresholds from OOF final-regressor predictions, we run a short “temperature-like” linear rescaling of the *regression outputs* (not changing the model) fitted on OOF to better align predicted score distribution to labels before thresholding. This keeps your core pipeline intact (same model, heads-only training, OOF threshold fitting, final-regressor inference) but typically improves QWK because QWK is very sensitive to small shifts around thresholds. I also make one small consistency fix: use the learned thresholds to compute and print an OOF QWK with the same rescaling applied, so we can confirm calibration logic is coherent. All changes are deterministic, lightweight, and still produce `submission.csv` with the same required format.'
- What this solution (achieved 0.68185) has done: 'To move your QWK up toward 0.912 with minimal disruption, I keep your exact model and “heads-only train → OOF calibration/threshold fit → test inference” pipeline but remove a single major mismatch: your fold models and main model are only trained when the external weights are missing, so when weights exist (or when fold training is skipped) the final regressor is never adapted to this dataset, making OOF thresholds/calibration less meaningful. I therefore always run a short heads-only fine-tune (same loss/loops, backbone still frozen) on the training set for the main model, and inside each fold (regardless of weight_path), so OOF predictions are produced by a model that has actually been trained with the same final=True output you use at inference. I also make the affine calibration robust by fitting it on **rank-preserving** mapping (simple least-squares stays) but applying it only if it improves OOF QWK vs identity, preventing harmful over/under-scaling that can depress kappa. These are small, metric-aligned changes that should raise score from ~0.71 toward your target without changing architecture or inference semantics.'
- What this solution (achieved 0.68726) has done: 'Your current score (0.68185) is far below the target (0.91215), so we should improve it with minimal, metric-aligned changes while keeping your exact model and pipeline. The biggest limiter in your code is that during heads-only training (both main model and folds) you train against raw class integers (0–4) while your regressor outputs are constrained to [0, 4.5]; switching the regression targets to class midpoints (still on the same 0–4 scale, just better calibrated for thresholding/QWK) is a small change that typically boosts QWK without changing architecture or inference semantics. I also add a minimal class-imbalance handling to the CrossEntropy loss (computed from train label frequencies) to reduce the bias toward class 0, which commonly improves QWK for APTOS. Finally, I keep your OOF threshold+optional affine calibration logic intact, but I compute the affine fit on clipped predictions to avoid rare extreme values skewing the calibration.'

# 9. Code solution

## === cell 0
import os
import glob
import random
import time
import math
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import torchvision.transforms as transforms
from torchvision.transforms import functional as FT
from PIL import Image, ImageChops

from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold
import timm

device = "cuda:0" if torch.cuda.is_available() else "cpu"

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

try:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
except Exception:
    pass



## === cell 1
threshold = [0.75, 1.5, 2.5, 3.5]


def regress2class(out, thr=None):
    if thr is None:
        thr = threshold
    out = out.view(-1)
    prediction = torch.zeros(out.size(0), dtype=torch.long)
    for i in range(4):
        prediction += (out.data >= thr[i]).to(torch.long).cpu()
    return prediction


def ordinal2class_prob(out):
    pred_prob = torch.zeros(out.size(0), 5, device=out.device, dtype=out.dtype)
    pred_prob[:, 0] = (1 - out[:, 0]).squeeze()
    pred_prob[:, 1] = (out[:, 0] * (1 - out[:, 1])).squeeze()
    pred_prob[:, 2] = (out[:, 1] * (1 - out[:, 2])).squeeze()
    pred_prob[:, 3] = (out[:, 2] * (1 - out[:, 3])).squeeze()
    pred_prob[:, 4] = out[:, 3].squeeze()
    return F.softmax(pred_prob, dim=1)


def regress2class_prob(out):
    out = out.view(-1).to(dtype=torch.float32)
    pred_prob = torch.zeros((out.size(0), 5), device=out.device, dtype=out.dtype)

    lt4 = out < 4.0
    if lt4.any():
        o = out[lt4]
        l1 = torch.floor(o).to(torch.long)
        l2 = torch.ceil(o).to(torch.long)
        w2 = o - l1.to(o.dtype)
        w1 = 1.0 - w2
        rows = torch.arange(o.size(0), device=out.device)
        pred_prob[lt4.nonzero(as_tuple=True)[0][rows], l1] = w1
        pred_prob[lt4.nonzero(as_tuple=True)[0][rows], l2] = 1.0 - (l2.to(o.dtype) - o)
    ge4 = ~lt4
    if ge4.any():
        pred_prob[ge4, 4] = 1.0
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
    def __init__(self, backbone=None, pretrained_backbone=False):
        super(ThreeStage_Model, self).__init__()

        self.backbone = timm.models.tf_efficientnet_b4_ns(
            pretrained=pretrained_backbone
        )
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
TEST_CSV_CANDIDATES = [
    "../input/aptos2019-blindness-detection/test.csv",
    "/kaggle/input/aptos2019-blindness-detection/test.csv",
    "/kaggle/data/aptos2019-blindness-detection/test.csv",
]
TEST_IMG_DIR_CANDIDATES = [
    "../input/aptos2019-blindness-detection/test_images",
    "/kaggle/input/aptos2019-blindness-detection/test_images",
    "/kaggle/data/aptos2019-blindness-detection/test_images",
]
TRAIN_CSV_CANDIDATES = [
    "../input/aptos2019-blindness-detection/train.csv",
    "/kaggle/input/aptos2019-blindness-detection/train.csv",
    "/kaggle/data/aptos2019-blindness-detection/train.csv",
]
TRAIN_IMG_DIR_CANDIDATES = [
    "../input/aptos2019-blindness-detection/train_images",
    "/kaggle/input/aptos2019-blindness-detection/train_images",
    "/kaggle/data/aptos2019-blindness-detection/train_images",
]


def first_existing_path(candidates, kind="file"):
    for p in candidates:
        if kind == "file" and os.path.isfile(p):
            return p
        if kind == "dir" and os.path.isdir(p):
            return p
    return None


test_csv_path = first_existing_path(TEST_CSV_CANDIDATES, kind="file")
test_img_dir = first_existing_path(TEST_IMG_DIR_CANDIDATES, kind="dir")
train_csv_path = first_existing_path(TRAIN_CSV_CANDIDATES, kind="file")
train_img_dir = first_existing_path(TRAIN_IMG_DIR_CANDIDATES, kind="dir")

if test_csv_path is None:
    raise FileNotFoundError(
        f"Could not find test.csv in candidates: {TEST_CSV_CANDIDATES}"
    )
if test_img_dir is None:
    raise FileNotFoundError(
        f"Could not find test_images/ in candidates: {TEST_IMG_DIR_CANDIDATES}"
    )
if train_csv_path is None:
    raise FileNotFoundError(
        f"Could not find train.csv in candidates: {TRAIN_CSV_CANDIDATES}"
    )
if train_img_dir is None:
    raise FileNotFoundError(
        f"Could not find train_images/ in candidates: {TRAIN_IMG_DIR_CANDIDATES}"
    )

test_df = pd.read_csv(test_csv_path)
test_ids = test_df["id_code"].astype(str).values

train_df = pd.read_csv(train_csv_path)
train_df["id_code"] = train_df["id_code"].astype(str)
train_df["diagnosis"] = train_df["diagnosis"].astype(int)

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

WEIGHT_CANDIDATES = [
    "../input/weights/B4_3stage_17epoch_320.pkl",
    "/kaggle/input/weights/B4_3stage_17epoch_320.pkl",
    "/kaggle/data/weights/B4_3stage_17epoch_320.pkl",
]
WEIGHT_GLOBS = [
    "../input/**/B4_3stage_17epoch_320.pkl",
    "/kaggle/input/**/B4_3stage_17epoch_320.pkl",
    "/kaggle/data/**/B4_3stage_17epoch_320.pkl",
]

weight_path = first_existing_path(WEIGHT_CANDIDATES, kind="file")
if weight_path is None:
    for g in WEIGHT_GLOBS:
        hits = glob.glob(g, recursive=True)
        if hits:
            weight_path = hits[0]
            break

use_pretrained_backbone = weight_path is None

net = ThreeStage_Model(pretrained_backbone=use_pretrained_backbone).to(device)

cached_state = None
if weight_path is not None:
    cached_state = torch.load(weight_path, map_location="cpu")
    state = (
        cached_state
        if device == "cpu"
        else {k: v.to(device) for k, v in cached_state.items()}
    )
    missing, unexpected = net.load_state_dict(state, strict=False)
    if len(unexpected) > 0:
        print("Warning: unexpected keys while loading weights:", unexpected[:10])
    if len(missing) > 0:
        print("Warning: missing keys while loading weights:", missing[:10])
else:
    print(
        "Warning: custom weights file B4_3stage_17epoch_320.pkl not found; "
        "using ImageNet-pretrained backbone with randomly initialized heads (will train heads)."
    )

net.eval()



## === cell 5
from torch.utils.data import Dataset, DataLoader


class ImageTensorDataset(Dataset):
    def __init__(self, img_dir, ids, transform):
        self.img_dir = img_dir
        self.ids = list(map(str, ids))
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        idx = self.ids[i]
        image_path = os.path.join(self.img_dir, f"{idx}.png")
        try:
            with Image.open(image_path) as im:
                img = im.convert("RGB")
            t = self.transform(img)
            return idx, t
        except Exception:
            return idx, None


def _seed_worker(worker_id):
    seed = 42 + worker_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_images_as_tensors(img_dir, ids, transform, batch_size=32, num_workers=None):
    if num_workers is None:
        cpu = os.cpu_count() or 2
        num_workers = min(4, cpu)

    ds = ImageTensorDataset(img_dir, ids, transform)
    g = torch.Generator()
    g.manual_seed(42)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
        worker_init_fn=_seed_worker,
        generator=g,
        collate_fn=lambda batch: batch,
    )

    tensors = []
    ids_ok = []
    missing = 0
    for batch in dl:
        for idx, t in batch:
            if t is None:
                missing += 1
            else:
                ids_ok.append(idx)
                tensors.append(t)
    return ids_ok, tensors, missing


def infer_regression_batch(tensors, batch_size=32, use_final=False, model=None):
    outs = []
    if model is None:
        model = net
    model.eval()
    pin = torch.cuda.is_available()
    with torch.no_grad():
        for i in range(0, len(tensors), batch_size):
            batch_cpu = torch.stack(tensors[i : i + batch_size], dim=0)
            if pin:
                batch_cpu = batch_cpu.pin_memory()
            batch = batch_cpu.to(device, non_blocking=pin)
            if use_final:
                out = model(batch, final=True)
                outs.append(out.view(-1).detach().float().cpu().numpy())
            else:
                _, r_out, _ = model(batch, final=False)
                outs.append(r_out.view(-1).detach().float().cpu().numpy())
    return (
        np.concatenate(outs, axis=0) if len(outs) else np.zeros((0,), dtype=np.float32)
    )


def qwk(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


def apply_thresholds(preds, thr):
    thr = np.asarray(list(thr), dtype=np.float64).reshape(1, -1)
    preds = np.asarray(preds, dtype=np.float64).reshape(-1, 1)
    out = (preds >= thr).sum(axis=1).astype(np.int64)
    return np.clip(out, 0, 4)


def fit_thresholds_simple(preds, y_true, init_thr=(0.75, 1.5, 2.5, 3.5)):
    preds = np.asarray(preds).reshape(-1).astype(np.float64)
    y_true = np.asarray(y_true).astype(int)

    thr = np.array(init_thr, dtype=np.float64)

    step = 0.20
    best_thr = thr.copy()
    best_score = qwk(y_true, apply_thresholds(preds, best_thr))

    for _ in range(45):
        improved = False
        for k in range(4):
            candidates = best_thr[k] + step * np.array(
                [-2, -1, 0, 1, 2], dtype=np.float64
            )
            for cand in candidates:
                trial = best_thr.copy()
                trial[k] = cand
                trial = np.sort(trial)
                if trial[0] < 0.0 or trial[-1] > 4.5:
                    continue
                score = qwk(y_true, apply_thresholds(preds, trial))
                if score > best_score + 1e-10:
                    best_score = score
                    best_thr = trial
                    improved = True
        if not improved:
            step *= 0.5
            if step < 5e-4:
                break

    return best_thr.tolist(), float(best_score)


def fit_affine_calibration(preds, y_true):
    preds = np.asarray(preds, dtype=np.float64).reshape(-1)
    preds = np.clip(preds, 0.0, 4.5)
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    if preds.size == 0:
        return 1.0, 0.0
    x = preds
    y = y_true
    vx = float(np.var(x))
    if vx < 1e-12:
        return 1.0, 0.0
    a = float(np.cov(x, y, bias=True)[0, 1] / vx)
    b = float(y.mean() - a * x.mean())
    return a, b


def apply_affine(preds, a, b):
    preds = np.asarray(preds, dtype=np.float64)
    return preds * a + b


t0 = time.time()
train_ids_used, train_tensors, missing_train = load_images_as_tensors(
    train_img_dir,
    train_df["id_code"].values,
    transform,
    batch_size=32,
    num_workers=None,
)
id2label = dict(zip(train_df["id_code"].values, train_df["diagnosis"].values))
train_labels = [int(id2label[i]) for i in train_ids_used]
print(
    f"Train images loaded: {len(train_tensors)} (missing/unreadable skipped: {missing_train}) "
    f"in {time.time()-t0:.1f}s"
)

_counts = np.bincount(np.array(train_labels, dtype=np.int64), minlength=5).astype(
    np.float64
)
_ce_weights = _counts.sum() / np.maximum(_counts, 1.0)
_ce_weights = (_ce_weights / _ce_weights.mean()).astype(np.float32)
CE_CLASS_WEIGHTS = torch.tensor(_ce_weights, dtype=torch.float32, device=device)

REG_TARGETS = torch.tensor(
    [0.0, 1.0, 2.0, 3.0, 4.0], dtype=torch.float32, device=device
)




## === cell 6
def train_heads_only(
    net, tensors, labels, epochs=1, batch_size=32, lr=3e-4, weight_decay=1e-4
):
    if len(tensors) == 0:
        return

    for p in net.backbone.parameters():
        p.requires_grad = False
    for m in [net.classifier, net.regressor, net.ordinal, net.final_regressor]:
        for p in m.parameters():
            p.requires_grad = True

    params = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    ce = nn.CrossEntropyLoss(weight=CE_CLASS_WEIGHTS)
    bce = nn.BCELoss()
    mse = nn.MSELoss()

    n = len(tensors)
    order = np.arange(n)

    net.train()
    pin = torch.cuda.is_available()
    for ep in range(epochs):
        np.random.shuffle(order)
        t0 = time.time()
        total_loss = 0.0
        steps = 0

        for i in range(0, n, batch_size):
            idxs = order[i : i + batch_size]
            batch_cpu = torch.stack([tensors[j] for j in idxs], dim=0)
            if pin:
                batch_cpu = batch_cpu.pin_memory()
            batch = batch_cpu.to(device, non_blocking=pin)
            y = torch.tensor([labels[j] for j in idxs], device=device, dtype=torch.long)

            opt.zero_grad(set_to_none=True)

            c_out, r_out, o_out = net(batch, final=False)
            final_out = net(batch, final=True).view(-1)

            loss_c = ce(c_out, y)

            y_target = REG_TARGETS[y].detach()
            loss_r = mse(r_out.view(-1), y_target)
            loss_f = mse(final_out, y_target)

            y_float = y.float()
            ord_t = torch.stack([(y_float > k).float() for k in range(4)], dim=1)
            loss_o = bce(o_out, ord_t)

            loss = loss_c + 0.5 * loss_r + 0.5 * loss_o + 0.5 * loss_f

            loss.backward()
            opt.step()

            total_loss += float(loss.detach().cpu())
            steps += 1

        dt = time.time() - t0
        print(
            f"Heads-only train epoch {ep+1}/{epochs} - loss {total_loss/max(steps,1):.4f} - {dt:.1f}s"
        )

    net.eval()


if len(train_tensors) >= 200:
    train_heads_only(
        net,
        train_tensors,
        train_labels,
        epochs=1,
        batch_size=32,
        lr=3e-4,
        weight_decay=1e-4,
    )
else:
    print("Warning: too few train images for heads-only training; skipping.")



## === cell 7
learned_thresholds = threshold
oof_qwk = None

calib_a, calib_b = 1.0, 0.0

if len(train_tensors) >= 200:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_arr = np.array(train_labels, dtype=int)
    oof_preds = np.zeros(len(train_tensors), dtype=np.float64)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(y_arr)), y_arr), 1):
        fold_model = ThreeStage_Model(pretrained_backbone=(cached_state is None)).to(
            device
        )

        if cached_state is not None:
            state = (
                cached_state
                if device == "cpu"
                else {k: v.to(device) for k, v in cached_state.items()}
            )
            fold_model.load_state_dict(state, strict=False)

        tr_tensors = [train_tensors[i] for i in tr_idx]
        tr_labels = [train_labels[i] for i in tr_idx]
        va_tensors = [train_tensors[i] for i in va_idx]

        if len(tr_tensors) >= 200:
            train_heads_only(
                fold_model,
                tr_tensors,
                tr_labels,
                epochs=1,
                batch_size=32,
                lr=3e-4,
                weight_decay=1e-4,
            )
        else:
            fold_model.eval()

        va_preds = infer_regression_batch(
            va_tensors, batch_size=32, use_final=True, model=fold_model
        )
        oof_preds[va_idx] = va_preds
        print(
            f"Fold {fold}: trained on {len(tr_idx)} and inferred {len(va_idx)} val preds"
        )

        del fold_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    thr_id, q_id = fit_thresholds_simple(
        np.clip(oof_preds, 0.0, 4.5), y_arr, init_thr=threshold
    )

    a_tmp, b_tmp = fit_affine_calibration(oof_preds, y_arr)
    oof_preds_cal = np.clip(apply_affine(oof_preds, a_tmp, b_tmp), 0.0, 4.5)
    thr_cal, q_cal = fit_thresholds_simple(oof_preds_cal, y_arr, init_thr=threshold)

    if q_cal > q_id + 1e-10:
        calib_a, calib_b = a_tmp, b_tmp
        learned_thresholds, oof_qwk = thr_cal, q_cal
        used = "affine"
    else:
        calib_a, calib_b = 1.0, 0.0
        learned_thresholds, oof_qwk = thr_id, q_id
        used = "identity"

    final_oof_preds = np.clip(apply_affine(oof_preds, calib_a, calib_b), 0.0, 4.5)
    oof_pred_class = apply_thresholds(final_oof_preds, learned_thresholds)
    oof_qwk_check = qwk(y_arr, oof_pred_class)

    print("OOF QWK selected:", float(oof_qwk), "using", used)
    print("OOF QWK check (selected calib + learned thr):", float(oof_qwk_check))
    print("Affine calibration (a, b):", float(calib_a), float(calib_b))
    print("Learned thresholds:", learned_thresholds)
else:
    print("Warning: too few train images loaded; using default thresholds:", threshold)



## === cell 8
t0 = time.time()
test_ids_ok, test_tensors, missing_images = load_images_as_tensors(
    test_img_dir, test_ids, transform, batch_size=32, num_workers=None
)
print(
    f"Test images loaded: {len(test_tensors)} (missing/unreadable skipped: {missing_images}) "
    f"in {time.time()-t0:.1f}s"
)

test_preds_reg = np.zeros(len(test_ids), dtype=np.float64)

if len(test_tensors) > 0:
    ok_preds = infer_regression_batch(
        test_tensors, batch_size=32, use_final=True, model=net
    )
    ok_map = {k: float(v) for k, v in zip(test_ids_ok, ok_preds)}
    for i, idx in enumerate(test_ids):
        test_preds_reg[i] = ok_map.get(idx, 0.0)

test_preds_reg = apply_affine(test_preds_reg, calib_a, calib_b)
test_preds_reg = np.clip(test_preds_reg, 0.0, 4.5)

test_pred_class = apply_thresholds(test_preds_reg, learned_thresholds)

submission = pd.DataFrame(
    {"id_code": test_ids, "diagnosis": test_pred_class.astype(int)}
)
submission["diagnosis"] = submission["diagnosis"].astype(int).clip(0, 4)

submission = test_df[["id_code"]].merge(submission, on="id_code", how="left")
submission["diagnosis"] = submission["diagnosis"].fillna(0).astype(int).clip(0, 4)

submission.to_csv("submission.csv", index=False)

if submission.shape[0] == 0:
    raise RuntimeError(
        "Submission DataFrame is empty; cannot write a valid submission."
    )
if list(submission.columns) != ["id_code", "diagnosis"]:
    raise RuntimeError(f"Unexpected submission columns: {submission.columns.tolist()}")
if submission.shape[0] != test_df.shape[0]:
    raise RuntimeError(
        f"Submission row count {submission.shape[0]} does not match test.csv {test_df.shape[0]}"
    )

print("Wrote submission.csv with shape:", submission.shape)
print("Missing/unreadable test images handled:", int(missing_images))
print("Used thresholds:", learned_thresholds)
print("Used affine calibration (a, b):", calib_a, calib_b)
print(submission.head())
