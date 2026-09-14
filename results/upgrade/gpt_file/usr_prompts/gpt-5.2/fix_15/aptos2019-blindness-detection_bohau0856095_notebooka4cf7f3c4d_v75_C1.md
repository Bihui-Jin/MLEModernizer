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

0.9189743985652252

# 6. Current score

0.0287

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I (1) make weight loading non-fatal so the notebook always runs end-to-end and writes a `submission.csv` even when the external pretrained weights file is missing. Then (2) fix the GPU/CPU dtype mismatch by ensuring the model parameters are moved to `device` after loading (or after initialization) before inference. Finally (3) fix the submission alignment KeyError by building the output directly from `test.csv` order via a dict mapping, guaranteeing exact row order and required columns without fragile reindexing.'
- What this solution (achieved -0.09521) has done: 'Your current 0.0 score is consistent with running inference using randomly initialized weights, so the smallest change that can move the score toward the 0.919 target is to ensure the model actually uses pretrained weights. I therefore (1) load ImageNet pretrained weights for the EfficientNet-B4 backbone when the competition finetuned weights file is missing, and (2) also switch to `final=True` inference when finetuned weights are found (since those checkpoints typically expect the final regressor head), while keeping your existing regression-to-class thresholding and submission formatting unchanged. These are minimal, score-relevant changes that preserve the same model architecture and inference semantics, but avoid the “random model” failure mode. The script still run end-to-end and always write `submission.csv`.'
- What this solution (achieved 0.03045) has done: 'Your score is far below the target, so the most likely issue is that you’re still effectively using a mis-specified checkpoint vs model (or never actually loading finetuned weights), making predictions behave like a near-random regressor/classifier. I (1) make finetuned-weight loading robust by auto-detecting whether the checkpoint expects `final=True` (10-dim head) or the normal heads, and route inference accordingly, and (2) keep your exact model and thresholding but add a minimal, metric-relevant post-processing step: fit optimal regression-to-class thresholds on a small train/val split using quadratic weighted kappa and then apply those thresholds to test predictions (this preserves your core regression→class semantics but calibrates it toward the metric). These changes are directly score-relevant for QWK and are minimal compared to changing the architecture or training. The script still runs end-to-end and always writes a valid `submission.csv`.'
- What this solution (achieved 0.02362) has done: 'Your current score (0.03045) is far below the target (0.91897), so we should make a small, score-relevant change that increases QWK without changing the model or training approach. The biggest likely issue is that the threshold calibration is being done on a random val split, which can be unstable and can pick thresholds that don’t generalize; switching to out-of-fold (OOF) threshold calibration gives a much more reliable mapping from regression outputs to 0–4 classes while preserving your exact inference and thresholding semantics. I keep your same model, same weight-loading behavior, same transforms, and same prediction code, but replace the single split with 5-fold stratified OOF predictions to fit thresholds. This is minimal, stays within time, and should move the score materially upward toward the target.'
- What this solution (achieved -0.08207) has done: 'Your score is far below the target, so we should make a small but high-impact, metric-relevant change that doesn’t alter your model or training loops: calibrate the regression→class thresholds in a way that directly optimizes Quadratic Weighted Kappa. Your current coordinate-search threshold optimizer is brittle and can get stuck; replacing it with a standard Nelder–Mead optimization over 4 ordered thresholds usually yields a materially better mapping while preserving the exact same prediction semantics (still thresholding the same regressor outputs). I keep your OOF setup, model loading, transforms, and inference untouched, and only swap the threshold fitting routine plus add safe ordering constraints. This should move QWK upward toward the 0.919 target without changing architecture or adding extra training.'
- What this solution (achieved 0.0287) has done: 'Your current score is far below the target, so we should make a small but high-impact metric-aligned change without changing your model or any training loops: calibrate the regression→class thresholds on OOF predictions using the same distribution as the true labels. The main issue is that your regressor outputs are typically poorly calibrated (often compressed), and unconstrained optimization can pick thresholds that don’t match class frequencies, yielding very low/negative QWK. I replace the Nelder–Mead threshold search with a minimal, deterministic “distribution-matching” thresholding based on OOF regression quantiles per class (and then a tiny local refinement around those thresholds). This preserves your exact inference, architecture, and “count thresholds crossed” semantics, but makes the discretization far more stable and usually much closer to strong QWK.'
- What this solution (achieved 0.0287) has done: 'Your current score is far below the target, so the most likely reason is that the finetuned weights are not actually being loaded (and the ImageNet fallback for EfficientNet-B4 alone is not enough), causing near-random predictions. I make weight discovery succeed in this environment by searching for any 3-stage EfficientNet-B4 checkpoint in the available dataset tree and by handling common checkpoint formats more robustly (including nested dicts and different key prefixes). I also harden the “final vs non-final” inference-mode detection by using the loaded state dict keys and tensor shapes, so you don’t silently run the wrong head. These are minimal changes that keep your model and threshold calibration logic intact but should materially increase QWK toward your target when the correct finetuned checkpoint exists in the environment.'
- What this solution (achieved 0.0287) has done: 'Your score is far below the target, so we need a small change that improves correctness of the learned thresholds without changing the model or training approach. The biggest score-killer here is that your OOF calibration currently uses only per-fold validation predictions (each image predicted by a potentially different batchnorm/statistics state), which can make the regressor output scale inconsistent and ruin global thresholds. I keep your exact architecture and inference path, but I switch the OOF generation to be a true “single model snapshot” prediction by running inference on the full train set once (still no training), then do the same quantile+local-refine threshold calibration on those consistent predictions. This is minimal, stays within time, preserves your regression→threshold→class semantics, and should move QWK materially upward toward the target.'
- What this solution (achieved 0.0287) has done: 'Your score (0.0287) is far below the target (0.91897), so we need a correctness fix that materially improves QWK without changing your model or any training loops. The biggest likely issue is a backbone mismatch: your `ThreeStage_Model` uses EfficientNet-B4, but your weight finder prioritizes a `B4_...` filename while the code comment/history suggests a B4/B5 mix; loading a mismatched checkpoint with `strict=False` can silently leave most of the backbone random, producing near-random predictions and very low QWK. I add a minimal, shape-based compatibility check to only accept checkpoints that actually match the model’s backbone stem/conv shapes, and otherwise fall back to ImageNet-pretrained B4 (instead of partially-random weights). This preserves the core architecture and inference semantics, but avoids the “silently partially loaded random model” failure mode and should move the score sharply upward toward the target.'

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
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as transforms
from torchvision.transforms import functional as FT
from PIL import Image, ImageChops

import timm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)



## === cell 1
threshold = [0.75, 1.5, 2.5, 3.5]


def regress2class(out: torch.Tensor) -> torch.Tensor:
    """
    out: shape (B,) float in [0, 4.5]
    returns: shape (B,) long-like float tensor on CPU
    """
    prediction = torch.zeros(out.size(0), device=out.device)
    for i in range(4):
        prediction += (out >= threshold[i]).squeeze()
    return prediction.detach().cpu()


def ordinal2class_prob(out: torch.Tensor) -> torch.Tensor:
    pred_prob = torch.zeros(out.size(0), 5, device=out.device, dtype=out.dtype)
    pred_prob[:, 0] = (1 - out[:, 0]).squeeze()
    pred_prob[:, 1] = (out[:, 0] * (1 - out[:, 1])).squeeze()
    pred_prob[:, 2] = (out[:, 1] * (1 - out[:, 2])).squeeze()
    pred_prob[:, 3] = (out[:, 2] * (1 - out[:, 3])).squeeze()
    pred_prob[:, 4] = out[:, 3].squeeze()
    return F.softmax(pred_prob, dim=1)


def regress2class_prob(out: torch.Tensor) -> torch.Tensor:
    pred_prob = torch.zeros((out.size(0), 5), device=out.device, dtype=out.dtype)
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
def _resolve_data_dir():
    candidates = [
        "/kaggle/input/aptos2019-blindness-detection",
        "/kaggle/data/aptos2019-blindness-detection",
        "/kaggle/data/aptos2019-blindness-detection/aptos2019-blindness-detection",
        "../input/aptos2019-blindness-detection",
        "../kaggle/input/aptos2019-blindness-detection",
        "../kaggle/data/aptos2019-blindness-detection",
        "../kaggle/data/aptos2019-blindness-detection/aptos2019-blindness-detection",
    ]

    def has_required(d):
        if not os.path.exists(os.path.join(d, "test.csv")):
            return False
        if os.path.isdir(os.path.join(d, "test_images")):
            return True
        if os.path.isdir(os.path.join(d, "test_images", "test_images")):
            return True
        return False

    for d in candidates:
        if has_required(d):
            return d

    search_roots = ["/kaggle/data", "/kaggle/input", "../input"]
    for root in search_roots:
        if not os.path.isdir(root):
            continue
        for r, dirs, files in os.walk(root):
            if "test.csv" in files and ("test_images" in dirs):
                return r

    raise FileNotFoundError(
        "Could not locate aptos2019-blindness-detection directory. "
        "Tried: " + ", ".join(candidates)
    )


DATA_DIR = _resolve_data_dir()
TEST_CSV = os.path.join(DATA_DIR, "test.csv")
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")

TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")
if not os.path.isdir(TEST_IMG_DIR) and os.path.isdir(
    os.path.join(TEST_IMG_DIR, "test_images")
):
    TEST_IMG_DIR = os.path.join(TEST_IMG_DIR, "test_images")

TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images")
if not os.path.isdir(TRAIN_IMG_DIR) and os.path.isdir(
    os.path.join(TRAIN_IMG_DIR, "train_images")
):
    TRAIN_IMG_DIR = os.path.join(TRAIN_IMG_DIR, "train_images")

test_ids = pd.read_csv(TEST_CSV)["id_code"].astype(str).values
train_df = pd.read_csv(TRAIN_CSV)
train_df["id_code"] = train_df["id_code"].astype(str)

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

net = ThreeStage_Model()


def _find_weight_file():
    preferred_names = [
        "B4_3stage_15epoch_finetune512.pkl",
        "B4_3stage_15epoch_finetune512.pth",
    ]
    candidate_weight_paths = [
        "/kaggle/input/weights/B4_3stage_15epoch_finetune512.pkl",
        "/kaggle/input/weights/B4_3stage_15epoch_finetune512.pth",
        "/kaggle/data/weights/B4_3stage_15epoch_finetune512.pkl",
        "/kaggle/data/weights/B4_3stage_15epoch_finetune512.pth",
        "/kaggle/input/aptos2019-blindness-detection/weights/B4_3stage_15epoch_finetune512.pkl",
        "/kaggle/input/aptos2019-blindness-detection/weights/B4_3stage_15epoch_finetune512.pth",
        "/kaggle/data/aptos2019-blindness-detection/weights/B4_3stage_15epoch_finetune512.pkl",
        "/kaggle/data/aptos2019-blindness-detection/weights/B4_3stage_15epoch_finetune512.pth",
        "/kaggle/data/aptos2019-blindness-detection/aptos2019-blindness-detection/weights/B4_3stage_15epoch_finetune512.pkl",
        "/kaggle/data/aptos2019-blindness-detection/aptos2019-blindness-detection/weights/B4_3stage_15epoch_finetune512.pth",
        "../input/weights/B4_3stage_15epoch_finetune512.pkl",
        "../input/weights/B4_3stage_15epoch_finetune512.pth",
        "../input/aptos2019-blindness-detection/weights/B4_3stage_15epoch_finetune512.pkl",
        "../input/aptos2019-blindness-detection/weights/B4_3stage_15epoch_finetune512.pth",
    ]
    for p in candidate_weight_paths:
        if os.path.exists(p):
            return p

    search_roots = [
        "/kaggle/input",
        "/kaggle/data",
        "/kaggle/working",
        "../input",
        "../kaggle/input",
        "../kaggle/data",
        "../kaggle/working",
    ]

    for root in search_roots:
        if not os.path.isdir(root):
            continue
        for r, _, files in os.walk(root):
            for fn in files:
                if fn in preferred_names:
                    return os.path.join(r, fn)

    for root in search_roots:
        if not os.path.isdir(root):
            continue
        for r, _, files in os.walk(root):
            for fn in files:
                lfn = fn.lower()
                if (
                    (lfn.endswith(".pth") or lfn.endswith(".pkl"))
                    and ("3stage" in lfn)
                    and ("b4" in lfn)
                ):
                    return os.path.join(r, fn)

    return None


def _extract_state_dict(state_obj):
    if isinstance(state_obj, dict):
        for k in [
            "state_dict",
            "model_state_dict",
            "model",
            "net",
            "weights",
            "params",
            "ema_state_dict",
        ]:
            if (
                k in state_obj
                and isinstance(state_obj[k], dict)
                and len(state_obj[k]) > 0
            ):
                return state_obj[k]
    return state_obj


def _strip_prefix_if_present(state_dict, prefix):
    if not isinstance(state_dict, dict) or not state_dict:
        return state_dict
    keys = list(state_dict.keys())
    if all(k.startswith(prefix) for k in keys):
        return {k[len(prefix) :]: v for k, v in state_dict.items()}
    return state_dict


def _detect_infer_with_final_from_state(state_dict):
    if not isinstance(state_dict, dict) or len(state_dict) == 0:
        return False
    keys = set(state_dict.keys())

    if any(k.startswith("final_regressor.") for k in keys):
        w_key = "final_regressor.1.weight"
        if w_key in state_dict and hasattr(state_dict[w_key], "shape"):
            try:
                return int(state_dict[w_key].shape[1]) == 10
            except Exception:
                return True
        return True

    return False


def _is_checkpoint_compatible_with_model(state_dict, model):
    """
    Change rationale (score-relevant): if we load an incompatible checkpoint with strict=False,
    most of the backbone can remain randomly initialized, producing near-random predictions and very low QWK.
    This check ensures we only accept finetuned weights that match the model's backbone tensor shapes.
    """
    if not isinstance(state_dict, dict) or len(state_dict) == 0:
        return False

    model_sd = model.state_dict()

    keys_to_check = [
        "backbone.conv_stem.weight",
        "backbone.bn1.weight",
        "backbone.bn1.bias",
    ]

    for k in keys_to_check:
        if (
            k in state_dict
            and k in model_sd
            and hasattr(state_dict[k], "shape")
            and hasattr(model_sd[k], "shape")
        ):
            if tuple(state_dict[k].shape) != tuple(model_sd[k].shape):
                return False

    head_keys = [
        "regressor.1.weight",  # ThreeStage regressor: SiLU, Linear(1000->500), SiLU, Linear(500->1)
        "classifier.1.weight",
        "ordinal.1.weight",
    ]
    for k in head_keys:
        if (
            k in state_dict
            and k in model_sd
            and hasattr(state_dict[k], "shape")
            and hasattr(model_sd[k], "shape")
        ):
            if tuple(state_dict[k].shape) != tuple(model_sd[k].shape):
                return False

    return True


loaded_path = _find_weight_file()
USING_FINETUNED_WEIGHTS = False
INFER_WITH_FINAL = False

if loaded_path is not None:
    state = torch.load(loaded_path, map_location="cpu")
    state = _extract_state_dict(state)
    state = _strip_prefix_if_present(state, "module.")
    state = _strip_prefix_if_present(state, "model.")
    state = _strip_prefix_if_present(state, "net.")

    if _is_checkpoint_compatible_with_model(state, net):
        INFER_WITH_FINAL = _detect_infer_with_final_from_state(state)
        missing, unexpected = net.load_state_dict(state, strict=False)
        USING_FINETUNED_WEIGHTS = True

        print("Loaded weights:", loaded_path)
        print("INFER_WITH_FINAL:", INFER_WITH_FINAL)
        if len(missing) or len(unexpected):
            print(
                f"NOTE: load_state_dict strict=False (missing={len(missing)}, unexpected={len(unexpected)})"
            )
            print("Example missing keys:", missing[:5])
            print("Example unexpected keys:", unexpected[:5])
    else:
        print(
            "WARNING: Found weights but they are incompatible with this EfficientNet-B4 model; ignoring:",
            loaded_path,
        )
        loaded_path = None

if loaded_path is None:
    print(
        "WARNING: No compatible finetuned weights file found. Falling back to ImageNet-pretrained EfficientNet-B4 backbone "
        "to avoid near-random predictions from partial/incompatible loading."
    )
    net.backbone = timm.models.tf_efficientnet_b4_ns(pretrained=True)
    net.backbone.global_pool = GeM(flatten=True)

net = net.to(device)
net.eval()

print("Resolved DATA_DIR:", DATA_DIR)
print("Resolved TEST_IMG_DIR:", TEST_IMG_DIR)
print("Resolved TRAIN_IMG_DIR:", TRAIN_IMG_DIR)
print("Num test ids:", len(test_ids))
print("Num train rows:", len(train_df))
print("USING_FINETUNED_WEIGHTS:", USING_FINETUNED_WEIGHTS)




## === cell 5
class ImageDataset(Dataset):
    def __init__(self, ids, img_dir, transform=None, labels=None):
        self.ids = list(ids)
        self.img_dir = img_dir
        self.transform = transform
        self.labels = None if labels is None else list(labels)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        image_name = os.path.join(self.img_dir, f"{img_id}.png")
        img = Image.open(image_name).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        if self.labels is None:
            return img_id, img
        return img_id, img, int(self.labels[idx])


def _predict_regression(ids, img_dir, batch_size=8):
    ds = ImageDataset(ids, img_dir, transform=transform, labels=None)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    out_ids = []
    out_reg = []
    with torch.inference_mode():
        for batch in dl:
            batch_ids, batch_imgs = batch
            batch_imgs = batch_imgs.to(device, non_blocking=True)
            if USING_FINETUNED_WEIGHTS and INFER_WITH_FINAL:
                r_out = net(batch_imgs, final=True)  # (B,1)
            else:
                _, r_out, _ = net(batch_imgs)  # (B,1)
            out_ids.extend([str(x) for x in batch_ids])
            out_reg.append(r_out.squeeze(1).detach().cpu().float())
    out_reg = torch.cat(out_reg, dim=0).numpy()
    return out_ids, out_reg


def _apply_thresholds(reg_pred, thr):
    reg_pred = np.asarray(reg_pred)
    diag = np.zeros(reg_pred.shape[0], dtype=np.int64)
    for t in thr:
        diag += (reg_pred >= t).astype(np.int64)
    return diag


def _qwk_for_thresholds(y_true, reg_pred, thr):
    pred = _apply_thresholds(reg_pred, thr)
    return float(
        cohen_kappa_score(np.asarray(y_true, dtype=np.int64), pred, weights="quadratic")
    )


def _quantile_thresholds_match_label_dist(y_true, reg_pred):
    """
    Pick thresholds by matching the reg_pred distribution to the true label distribution.
    Preserves the exact same thresholding semantics, but stabilizes discretization for QWK.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    reg_pred = np.asarray(reg_pred, dtype=np.float64)

    counts = np.bincount(y_true, minlength=5).astype(np.float64)
    probs = counts / counts.sum()
    cum = np.cumsum(probs)
    q = cum[:4]
    thr = np.quantile(reg_pred, q).astype(np.float64)

    thr = np.clip(thr, 0.0, 4.5)
    for i in range(1, 4):
        if thr[i] <= thr[i - 1] + 1e-3:
            thr[i] = thr[i - 1] + 1e-3
    thr = np.clip(thr, 0.0, 4.5)
    return thr.tolist()


def _local_refine_thresholds(
    y_true, reg_pred, thr_init, steps=(0.2, 0.1, 0.05), span=0.6
):
    """
    Deterministic small grid refinement around a good initialization.
    Preserves thresholding semantics and stays lightweight.
    """
    best_thr = np.array(thr_init, dtype=np.float64)
    best_k = _qwk_for_thresholds(y_true, reg_pred, best_thr)

    for step in steps:
        improved = True
        while improved:
            improved = False
            for i in range(4):
                base = best_thr.copy()
                candidates = np.arange(
                    base[i] - span, base[i] + span + 1e-12, step, dtype=np.float64
                )
                for v in candidates:
                    cand = base.copy()
                    cand[i] = float(np.clip(v, 0.0, 4.5))
                    cand = np.sort(cand)
                    for j in range(1, 4):
                        if cand[j] <= cand[j - 1] + 1e-3:
                            cand[j] = cand[j - 1] + 1e-3
                    cand = np.clip(cand, 0.0, 4.5)
                    k = _qwk_for_thresholds(y_true, reg_pred, cand)
                    if k > best_k + 1e-6:
                        best_k = k
                        best_thr = cand
                        improved = True
        span = max(span * 0.6, step * 3.0)

    return best_thr.tolist(), best_k


train_ids_all = train_df["id_code"].values.astype(str)
train_y_all = train_df["diagnosis"].values.astype(int)

t0 = time.time()
train_pred_ids, train_reg_all = _predict_regression(
    train_ids_all, TRAIN_IMG_DIR, batch_size=8
)
assert len(train_pred_ids) == len(train_ids_all)

train_pred_map = {i: float(r) for i, r in zip(train_pred_ids, train_reg_all)}
train_reg_ordered = np.array(
    [train_pred_map[i] for i in train_ids_all], dtype=np.float64
)

n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
fold_qwk = []
for fold, (_, va_idx) in enumerate(skf.split(train_ids_all, train_y_all), 1):
    val_y = train_y_all[va_idx]
    val_reg = train_reg_ordered[va_idx]
    fold_pred = _apply_thresholds(val_reg, threshold)
    k = cohen_kappa_score(val_y, fold_pred, weights="quadratic")
    fold_qwk.append(float(k))
    print(f"Fold {fold}/{n_splits}: baseline-threshold QWK={k:.5f} (n={len(va_idx)})")

thr0 = _quantile_thresholds_match_label_dist(train_y_all, train_reg_ordered)
k0 = _qwk_for_thresholds(train_y_all, train_reg_ordered, thr0)
thr_opt, qwk_opt = _local_refine_thresholds(train_y_all, train_reg_ordered, thr0)

print("Threshold calibration done in %.1fs" % (time.time() - t0))
print("Baseline fold QWK (mean):", float(np.mean(fold_qwk)))
print("Initial thresholds (hardcoded):", threshold)
print("Quantile-match thresholds:", thr0, "QWK_on_train_pred:", k0)
print("Refined thresholds:", thr_opt, "QWK_on_train_pred:", qwk_opt)

threshold = thr_opt



## === cell 6
submission_ids, test_reg = _predict_regression(test_ids, TEST_IMG_DIR, batch_size=8)
submission_diag = _apply_thresholds(test_reg, threshold).tolist()

assert len(submission_ids) == len(test_ids), "Prediction count mismatch"
assert len(submission_ids) > 0, "No predictions generated"

pred_map = {i: int(d) for i, d in zip(submission_ids, submission_diag)}

ordered_diag = []
missing_ids = []
for tid in list(test_ids):
    if tid in pred_map:
        ordered_diag.append(pred_map[tid])
    else:
        missing_ids.append(tid)
        ordered_diag.append(0)

if missing_ids:
    print(
        f"WARNING: {len(missing_ids)} test ids missing predictions; filled with 0. Example: {missing_ids[:5]}"
    )

df = pd.DataFrame(
    {"id_code": test_ids.astype(str), "diagnosis": np.array(ordered_diag, dtype=int)}
)

out_path = "submission.csv"
df.to_csv(out_path, index=False)
print("Wrote", out_path, "with shape", df.shape)
print(df.head())
