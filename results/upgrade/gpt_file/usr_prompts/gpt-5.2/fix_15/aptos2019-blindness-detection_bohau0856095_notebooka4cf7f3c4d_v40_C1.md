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

0.8989433704725162

# 6. Current score

0.26034

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I (1) remove the failing custom `pip install` and instead rely on the already-installed `timm`, (2) make the code run on CPU when no GPU is available by selecting `cuda` only if present, and (3) fix the missing weights issue by loading from any available weights file if present, otherwise falling back to the model’s random initialization so a non-empty submission is always produced. I also make `trim()` always return an image (it currently can return `None`), and I correct a small Python logic bug (`is` vs `==`) in the augmentation class (score-neutral here since it’s not used at inference). Finally, I ensure the script always writes `submission.csv` with the required columns and correct row count.'
- What this solution (achieved 0.01221) has done: 'Your current 0.0 score is consistent with a near-constant/low-quality prediction caused by missing pretrained competition weights; the single most direct way to move toward the 0.8989 target is to ensure the model uses a meaningful initialization even when custom weight files aren’t available. I keep your exact model and inference logic, but add a safe fallback: load ImageNet-pretrained EfficientNet weights (via timm) when the competition checkpoint can’t be found, so predictions are no longer essentially random. I also make the regression→class mapping return integer class tensors directly on CPU and keep thresholds unchanged to preserve evaluation semantics. The rest of the pipeline (transforms, forward pass, CSV writing and alignment) stays the same.'
- What this solution (achieved 0.01787) has done: 'Your current score is far below the target, so the smallest meaningful way to move toward it is to make inference use the model’s intended final regression head (which combines classifier+regressor+ordinal outputs) rather than only the intermediate regressor output. This preserves your architecture and weights exactly, but switches the forward call to `final=True`, which is consistent with how a “ThreeStage” model is typically meant to be used at inference. I also add a safe weight-loading fallback that attempts to load only the backbone weights when no full competition checkpoint exists, so you still get better-than-random features without changing the model design. The submission format/path is kept identical and still writes `submission.csv` with correct ordering.'
- What this solution (achieved 0.1012) has done: 'Your score is far below the target, so the smallest high-impact change is to make the regression→class conversion consistent with the model’s output range (0–4.5) by calibrating the thresholds on the training set using out-of-fold (OOF) predictions and optimizing quadratic weighted kappa. This keeps your model, transforms, and inference forward pass identical, but replaces the fixed hardcoded thresholds with data-driven ones that better align with the competition metric. To stay within the 600s budget, the calibration runs on a small, stratified subset of training images and uses a few quick coordinate-descent passes. The learned thresholds are then applied to all test predictions and a valid `submission.csv` is written as before.'
- What this solution (achieved 0.1012) has done: 'Your current score (0.1012) is far below the target (0.8989), so we should increase performance with the smallest change that preserves your model and inference semantics. The biggest remaining gap is that threshold calibration is done on in-sample predictions, which tends to overfit and produces thresholds that don’t generalize to the Kaggle test set. I keep the exact model, transforms, and regression→class mechanism, but change calibration to use out-of-fold (OOF) predictions on the same stratified subset, then optimize thresholds on those OOF predictions (still fast enough). This typically improves QWK materially while remaining a minimal, metric-aligned adjustment.'
- What this solution (achieved 0.1012) has done: 'Your score is far below the target, so we need a real (but still minimal) inference-quality boost without changing your model architecture or training loop (you have none). The biggest issue is that your EfficientNet backbones are currently created with `pretrained=False`, and your fallback only loads ImageNet weights when no competition checkpoint exists; in practice this often means you’re running random heads/backbone and threshold calibration can’t rescue QWK. I keep your exact model and `final=True` inference, but make the backbone initialization consistently use ImageNet pretrained weights (both when no checkpoint is found, and also as a “best-effort” initializer before loading a partial checkpoint). This preserves semantics while making predictions meaningful and should move QWK substantially toward the target; submission formatting and paths stay unchanged.'
- What this solution (achieved 0.11212) has done: 'Your current score is far below the target, so we need a real inference-quality improvement without changing your model design. The biggest limiter is that the threshold calibration subset is sampled uniformly per class, which strongly distorts the true label distribution and leads to thresholds that don’t transfer well to the test set under QWK. I keep your exact model, transforms, and “final=True regression→thresholds→class” pipeline, but change calibration to (1) sample a subset that preserves the original class distribution and (2) optimize thresholds on OOF predictions by alternating between choosing thresholds and re-running OOF evaluation, which tends to yield more stable thresholds. This is a minimal, metric-aligned change and still writes a valid `submission.csv` in the same format.'
- What this solution (achieved 0.1) has done: 'Your score is still far below the target, so we should increase it with the smallest metric-aligned change that doesn’t alter your model architecture or add training. The highest-impact minimal improvement here is to make inference more robust via test-time augmentation (TTA) using the same exact preprocessing plus a horizontal flip, then average the regression outputs before thresholding. This keeps your pipeline identical (same model, same forward `final=True`, same thresholding approach) but reduces prediction noise and typically improves QWK. I also keep the calibrated thresholds, and apply TTA consistently during the calibration OOF predictions so thresholds match the inference behavior.'
- What this solution (achieved 0.0) has done: 'Your score (0.1) is far below the target (0.8989), so we should increase it with the smallest changes that preserve your architecture and inference semantics. The biggest likely issue is mismatched input preprocessing for `tf_efficientnet_b4_ns/b5_ns`: these timm models expect their own mean/std and interpolation/crop behavior, and using custom normalization can severely hurt predictions even with pretrained weights. I keep your exact model, `final=True` inference, OOF threshold calibration, and TTA logic, but switch preprocessing to timm’s `resolve_data_config/create_transform` for the backbone to align inputs correctly. I also make the weight loading a bit more robust by stripping common prefixes and skipping non-tensor entries, which can prevent silently-bad partial loads.'
- What this solution (achieved -0.15467) has done: 'Your current score (0.0) is far below the target (0.8989), so we need to materially improve predictive signal with minimal, metric-aligned changes while keeping your model and inference pipeline intact. The biggest practical issue is that you are using ImageNet-pretrained EfficientNet features but leaving all custom heads (classifier/regressor/ordinal/final_regressor) randomly initialized when the competition checkpoint isn’t found, which makes outputs essentially noise and QWK near 0. I keep your exact architecture and no-training approach, but add a tiny “threshold-only” calibration step that uses the training label distribution to set thresholds so that predicted class proportions match the training proportions (a standard, fast post-processing for ordinal regression). This uses the same `final=True` regression outputs you already produce and does not change any model layers, loss, or training loops, but typically moves score substantially above 0 when weights are missing. I also ensure we don’t attempt the slow OOF threshold optimization unless a real checkpoint is actually loaded (so we stay within time and avoid optimizing thresholds on pure noise).'
- What this solution (achieved -0.15467) has done: 'Your current score is far below the target, so we should increase predictive signal with the smallest changes that don’t alter your architecture or add training. The biggest issue is that your timm preprocessing is configured for `tf_efficientnet_b4_ns` but you are actually feeding images to a `tf_efficientnet_b5_ns` backbone inside the forward pass, so inputs are normalized/resized incorrectly and predictions collapse. I make the transform config match the actual backbone used (B5), while keeping your model, `final=True` inference, TTA, and threshold calibration logic identical. I also clamp regression outputs to the expected [0, 4.5] range before thresholding/calibration to prevent rare numeric outliers from harming kappa.'
- What this solution (achieved 0.0) has done: 'Your score is far below the target, so we should increase predictive signal with the smallest changes that keep your model and inference pipeline intact. The main bug hurting performance is a preprocessing mismatch: your transform is built for `tf_efficientnet_b4_ns` while the actual `final=True` forward uses the B4 backbone but your separate `Regressor` (and earlier iterations) mixed B5 settings; to avoid any backbone/transform drift and improve stability, we derive the timm transform directly from `net.backbone` after constructing the model. Next, to prevent the extremely negative kappa seen when outputs are poorly calibrated, we switch the “no checkpoint” thresholding from label-distribution matching on a small subset to the same OOF coordinate-descent calibration you already use (but on a smaller subset) so thresholds are metric-optimized even without a competition checkpoint. Finally, we keep TTA and the `final=True` path unchanged and still write a valid `submission.csv` with correct ordering and columns.'
- What this solution (achieved -0.00355) has done: 'Your current score (0.0) is far below the target, so we need to add predictive signal with minimal changes while keeping your exact model and inference pipeline. Right now, when no competition checkpoint is found, the custom heads (including `final_regressor`) remain randomly initialized, so `final=True` outputs are essentially noise; threshold calibration can’t recover from that. The smallest high-impact fix is to switch to a deterministic, pretrained-only fallback path: if no checkpoint is loaded, run inference using a frozen ImageNet-pretrained `Regressor` model instead (same preprocessing/TTA/thresholding), and calibrate thresholds using that regressor’s outputs. If a competition checkpoint is found, your current `ThreeStage_Model(final=True)` path remains unchanged.'
- What this solution (achieved 0.26034) has done: 'Your current score is far below the target, so we should increase predictive signal with the smallest safe change that preserves your architecture and inference semantics. The biggest issue is that, when no competition checkpoint is found (common on Kaggle), you fall back to a different model (B5 regressor) whose outputs are not aligned with the B4 three-stage head/thresholding calibration; this tends to produce very low/negative QWK. I keep your exact models and inference flow, but change the “no checkpoint” fallback to use the same backbone as the main model (B4) so preprocessing/output distribution matches what your thresholds can calibrate. I also make the threshold optimization more robust by using a slightly larger, distribution-preserving calibration subset (still bounded for runtime) and by sorting thresholds at the end to enforce monotonicity.'

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

from sklearn.metrics import cohen_kappa_score  # used for threshold calibration
from sklearn.model_selection import StratifiedKFold
import timm
from timm.data import resolve_data_config, create_transform

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)



## === cell 1
threshold = [0.75, 1.5, 2.5, 3.5]


def regress2class(out, thr=None):
    thr = threshold if thr is None else thr
    out_cpu = out.detach().view(-1).to("cpu")
    pred = torch.zeros(out_cpu.size(0), dtype=torch.int64)
    for t in thr:
        pred += (out_cpu >= t).to(torch.int64)
    return pred


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
        val = float(out[i].detach().cpu().item())
        if val < 4.0:
            l1 = int(math.floor(val))
            l2 = int(math.ceil(val))
            pred_prob[i][l1] = 1 - (val - l1)
            pred_prob[i][l2] = 1 - (l2 - val)
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
        self.backbone = timm.models.tf_efficientnet_b5_ns(pretrained=True)
        self.backbone.global_pool = GeM(flatten=True)
        self.regressor = nn.Linear(1000, 1)

    def forward(self, x):
        x = self.backbone(x)
        out = self.regressor(x)
        out = torch.sigmoid(out) * 4.5
        return out


class RegressorB4(nn.Module):
    def __init__(self):
        super(RegressorB4, self).__init__()
        self.backbone = timm.models.tf_efficientnet_b4_ns(pretrained=True)
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

        self.backbone = timm.models.tf_efficientnet_b4_ns(pretrained=True)
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
TEST_CSV = "../input/aptos2019-blindness-detection/test.csv"
TEST_IMG_DIR = "../input/aptos2019-blindness-detection/test_images"
TRAIN_CSV = "../input/aptos2019-blindness-detection/train.csv"
TRAIN_IMG_DIR = "../input/aptos2019-blindness-detection/train_images"

test_df = pd.read_csv(TEST_CSV)
test_ids = test_df["id_code"].astype(str).values

train_df = pd.read_csv(TRAIN_CSV)
train_df["id_code"] = train_df["id_code"].astype(str)
train_df["diagnosis"] = train_df["diagnosis"].astype(int)

input_size = 380

net = ThreeStage_Model()

reg_fallback = RegressorB4()

_data_config_net = resolve_data_config({}, model=net.backbone)
_data_config_net["input_size"] = (3, input_size, input_size)
_timm_transform_net = create_transform(**_data_config_net, is_training=False)

_data_config_reg = resolve_data_config({}, model=reg_fallback.backbone)
_data_config_reg["input_size"] = (3, input_size, input_size)
_timm_transform_reg = create_transform(**_data_config_reg, is_training=False)

candidate_weight_paths = [
    "../input/weights/B4_3stage_47epoch_CLAHE.pkl",
    "../input/weights/B4_3stage_47epoch_CLAHE.pth",
    "../input/weights/B4_3stage_47epoch_CLAHE.pt",
]
scan_patterns = [
    "../input/**/*.pkl",
    "../input/**/*.pth",
    "../input/**/*.pt",
]
scanned = []
for pat in scan_patterns:
    scanned.extend(glob.glob(pat, recursive=True))

preferred = [
    p
    for p in scanned
    if ("3stage" in os.path.basename(p).lower() and "b4" in os.path.basename(p).lower())
]
all_candidates = candidate_weight_paths + preferred + scanned

weight_path = None
for p in all_candidates:
    if p and os.path.isfile(p):
        weight_path = p
        break


def _sanitize_state_dict(state):
    if (
        isinstance(state, dict)
        and "state_dict" in state
        and isinstance(state["state_dict"], dict)
    ):
        state = state["state_dict"]
    if not isinstance(state, dict):
        return None
    out = {}
    for k, v in state.items():
        if not torch.is_tensor(v):
            continue
        nk = k
        for pref in ("model.", "net.", "module."):
            if nk.startswith(pref):
                nk = nk[len(pref) :]
        out[nk] = v
    return out


loaded_checkpoint = False

if weight_path is None:
    print(
        "No competition weights found under ../input. Will use ImageNet-pretrained B4 regressor fallback for inference + calibration."
    )
else:
    print("Loading weights from:", weight_path)
    state = torch.load(weight_path, map_location="cpu")
    state = _sanitize_state_dict(state)
    if state is None:
        print(
            "WARNING: Could not parse checkpoint state_dict; will use ImageNet-pretrained B4 regressor fallback."
        )
    else:
        missing, unexpected = net.load_state_dict(state, strict=False)
        loaded_checkpoint = True
        if missing:
            print(
                f"WARNING: Missing keys when loading weights (showing up to 10): {missing[:10]}"
            )
        if unexpected:
            print(
                f"WARNING: Unexpected keys when loading weights (showing up to 10): {unexpected[:10]}"
            )

if loaded_checkpoint:
    model_for_pred = net
    _timm_transform = _timm_transform_net
    print("Inference model: ThreeStage_Model(final=True)")
else:
    model_for_pred = reg_fallback
    _timm_transform = _timm_transform_reg
    print("Inference model: RegressorB4() (fallback)")

transform = transforms.Compose(
    [
        trim(),
        cropTo4_3(),
        _timm_transform,
    ]
)

net = net.to(device)
net.eval()
reg_fallback = reg_fallback.to(device)
reg_fallback.eval()

print("loaded_checkpoint:", loaded_checkpoint)
print("timm data_config (net):", _data_config_net)
print("timm data_config (regressor):", _data_config_reg)



## === cell 5
USE_TTA = True


def _predict_one_image(model, image_path, use_tta=USE_TTA):
    if not os.path.isfile(image_path):
        return None
    img = Image.open(image_path).convert("RGB")

    def _forward_one(x):
        if isinstance(model, ThreeStage_Model):
            return model(x, final=True)
        return model(x)

    if not use_tta:
        x = transform(img).unsqueeze(0).to(device)
        with torch.inference_mode():
            out = _forward_one(x)  # [1,1] in [0,4.5]
        out = out.clamp(0.0, 4.5)
        return float(out.squeeze().detach().cpu().item())

    img_flip = FT.hflip(img)
    x1 = transform(img).unsqueeze(0).to(device)
    x2 = transform(img_flip).unsqueeze(0).to(device)
    with torch.inference_mode():
        out1 = _forward_one(x1)
        out2 = _forward_one(x2)
        out = 0.5 * (out1 + out2)
    out = out.clamp(0.0, 4.5)
    return float(out.squeeze().detach().cpu().item())


def _qwk_from_thresholds(y_true, y_pred_reg, thr):
    y_pred_cls = np.zeros_like(y_true, dtype=np.int64)
    for t in thr:
        y_pred_cls += (y_pred_reg >= t).astype(np.int64)
    y_pred_cls = np.clip(y_pred_cls, 0, 4)
    return cohen_kappa_score(y_true, y_pred_cls, weights="quadratic")


def optimize_thresholds_coordinate_descent(y_true, y_pred_reg, init_thr, n_passes=6):
    thr = np.array(init_thr, dtype=np.float64)
    best = _qwk_from_thresholds(y_true, y_pred_reg, thr)

    step_grid = [0.25, 0.1, 0.05]

    for _ in range(n_passes):
        improved_any = False
        for k in range(len(thr)):
            for step in step_grid:
                for direction in (-1.0, 1.0):
                    cand = thr.copy()
                    cand[k] = cand[k] + direction * step

                    lo = 0.0 if k == 0 else cand[k - 1] + 1e-3
                    hi = 4.5 if k == len(thr) - 1 else cand[k + 1] - 1e-3
                    cand[k] = float(np.clip(cand[k], lo, hi))

                    score = _qwk_from_thresholds(y_true, y_pred_reg, cand)
                    if score > best:
                        thr = cand
                        best = score
                        improved_any = True
        if not improved_any:
            break

    thr = np.sort(thr)
    thr[0] = float(np.clip(thr[0], 0.0, thr[1] - 1e-3))
    thr[1] = float(np.clip(thr[1], thr[0] + 1e-3, thr[2] - 1e-3))
    thr[2] = float(np.clip(thr[2], thr[1] + 1e-3, thr[3] - 1e-3))
    thr[3] = float(np.clip(thr[3], thr[2] + 1e-3, 4.5))

    return thr.tolist(), best


MAX_CALIB_SAMPLES = 1200  # still bounded to keep runtime reasonable
N_SPLITS = 5

y_all = train_df["diagnosis"].values.astype(np.int64)
idx_all = np.arange(len(train_df))
rng = np.random.RandomState(42)

if len(train_df) <= MAX_CALIB_SAMPLES:
    subset_idx = idx_all
else:
    subset_idx = []
    for cls in range(5):
        cls_idx = idx_all[y_all == cls]
        rng.shuffle(cls_idx)
        frac = float((y_all == cls).mean())
        take = int(round(MAX_CALIB_SAMPLES * frac))
        take = max(1, min(len(cls_idx), take))
        subset_idx.extend(cls_idx[:take].tolist())
    subset_idx = np.array(subset_idx, dtype=int)

subset_df = train_df.iloc[subset_idx].reset_index(drop=True)
subset_true = subset_df["diagnosis"].values.astype(np.int64)

print(
    "Calibrating thresholds on subset size:",
    len(subset_df),
    "loaded_checkpoint:",
    loaded_checkpoint,
)
print("Subset class counts:", np.bincount(subset_true, minlength=5))
print("TTA enabled for calibration/inference:", USE_TTA)

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
subset_pred_oof = np.full(len(subset_df), np.nan, dtype=np.float64)

for fold, (_, val_idx) in enumerate(
    skf.split(np.zeros(len(subset_df)), subset_true), 1
):
    t0 = time.time()
    for j in val_idx:
        row = subset_df.iloc[j]
        image_path = os.path.join(TRAIN_IMG_DIR, f"{row.id_code}.png")
        val = _predict_one_image(model_for_pred, image_path, use_tta=USE_TTA)
        if val is None:
            val = 0.0
        subset_pred_oof[j] = val
    print(f"Fold {fold}/{N_SPLITS} done in {time.time()-t0:.1f}s")

nan_mask = np.isnan(subset_pred_oof)
if nan_mask.any():
    subset_pred_oof[nan_mask] = 0.0
subset_pred_oof = np.clip(subset_pred_oof, 0.0, 4.5)

init_thr = threshold
thr_work = init_thr
best_work = _qwk_from_thresholds(subset_true, subset_pred_oof, thr_work)

for _round in range(2):
    thr_work, best_work = optimize_thresholds_coordinate_descent(
        subset_true, subset_pred_oof, init_thr=thr_work, n_passes=10
    )

print(
    "Initial thresholds:",
    init_thr,
    "OOF QWK:",
    _qwk_from_thresholds(subset_true, subset_pred_oof, init_thr),
)
print("Optimized thresholds:", thr_work, "OOF QWK:", best_work)

threshold = thr_work



## === cell 6
submission = []

with torch.inference_mode():
    for i, idx in enumerate(test_ids):
        image_name = os.path.join(TEST_IMG_DIR, f"{idx}.png")
        if not os.path.isfile(image_name):
            submission.append([idx, 0])
            continue

        pred_reg = _predict_one_image(model_for_pred, image_name, use_tta=USE_TTA)
        if pred_reg is None:
            submission.append([idx, 0])
            continue

        pred_reg = float(np.clip(pred_reg, 0.0, 4.5))

        out_tensor = torch.tensor([pred_reg], dtype=torch.float32)
        pred = regress2class(out_tensor, thr=threshold)
        submission.append([idx, int(pred.item())])

submission_df = pd.DataFrame(submission, columns=["id_code", "diagnosis"])
print("Submission rows:", len(submission_df), "Expected:", len(test_ids))
print(submission_df.head())



## === cell 7
submission_df["id_code"] = submission_df["id_code"].astype(str)
submission_df["diagnosis"] = submission_df["diagnosis"].astype(int).clip(0, 4)

submission_df = submission_df.set_index("id_code").reindex(test_ids).reset_index()
submission_df.to_csv("submission.csv", index=False)

print("Wrote submission.csv")
print(pd.read_csv("submission.csv").head())
print("Shape:", pd.read_csv("submission.csv").shape)
print("Final thresholds used:", threshold)
print("TTA used:", USE_TTA)
print("loaded_checkpoint:", loaded_checkpoint)
print("Inference model type:", type(model_for_pred).__name__)
