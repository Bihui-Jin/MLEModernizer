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

0.9293190275094716

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I (1) fix the missing weights path by auto-discovering the provided `.pth` file(s) under `/kaggle/input` and loading the checkpoint robustly (handling both full state dicts and wrapped dicts), (2) fix the CUDA/CPU dtype/device mismatch by moving the model to the target device *after* loading weights and ensuring all model parameters are on the same device as the input, and (3) ensure test image paths resolve correctly in this environment and always write a non-empty `submission.csv` with the exact required columns. These are execution/stability fixes; they keep your model and inference logic unchanged and should yield a valid submission file end-to-end. If no weights are actually present, the script fall back to random-init (still producing a valid CSV), but it warn that score be poor.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with a submission/label mismatch rather than a model issue, so I make two minimal changes: (1) build the submission starting from `sample_submission.csv` to guarantee correct row order and IDs, and (2) ensure predictions are mapped back to that exact ID order via a dict lookup. This keeps your model, preprocessing, thresholds, and inference semantics unchanged while removing the most common cause of a zero kappa (misaligned `id_code` ↔ `diagnosis`). I also add a strict sanity-check that all sample IDs were predicted to prevent silently writing a partially mismatched file.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with either (a) failing to load the intended weights (silently running random-init) or (b) producing near-constant predictions due to using only the regression head with fixed thresholds. I make two minimal, inference-only changes that preserve your model and preprocessing: (1) make weight discovery prefer checkpoints that actually match `backboneNet_efficient` by checking key names/shapes before loading, and (2) keep your current regression-threshold prediction but add a very small, metric-relevant improvement by averaging the regression-derived class with the argmax of the classifier head (still same network outputs, no retraining). This should move the score up substantially toward the target while keeping the pipeline stable and still writing a valid `submission.csv` aligned to `sample_submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with “random-init weights” (or wrong checkpoint picked), which produces near-random/constant predictions even though the submission format is correct. I keep your model and inference logic the same, but make weight selection stricter by preferring checkpoints that match *both* (a) many parameter shapes and (b) the EfficientNet-B4 feature dimension (1792) seen in `backboneNet_efficient`, so we’re far more likely to actually load the intended trained weights. I also load with `strict=False` only when needed (and report missing/unexpected keys) so the run doesn’t silently fail into random-init due to minor key-prefix differences, while still using the best-matching weights. This should move the score upward toward the target without changing architecture, transforms, or the prediction rule.'
- What this solution (achieved 0.05806) has done: 'The run currently stops because no compatible checkpoint is found, which prevents `test_images_dir` from being defined and causes the later `NameError`. I make checkpoint loading robust by (a) searching common Kaggle locations for any `.pth/.pt/.ckpt`, and (b) if none are found, falling back to `timm` pretrained EfficientNet-B4 backbone weights (keeping the same heads and inference logic) so the notebook can still produce a valid submission. I also make the state-dict loading handle the common “backbone.*” prefix mismatch so weights that were saved from a wrapped model can load cleanly. These changes are minimal, unblock end-to-end execution, and should raise score well above the random-init baseline.'
- What this solution (achieved 0.0) has done: 'Your current score (0.05806) is far below the target (0.9293), so we should improve predictions without changing the model architecture or training loop (there is no training here). The main low-risk, high-impact issue is that your post-processing uses fixed thresholds and an ad-hoc average of regression-class and classifier argmax, which is usually badly calibrated for quadratic weighted kappa. I keep the exact same network forward pass and outputs, but replace the hard-coded thresholds with thresholds fit on the provided training labels using out-of-fold predictions (same inference semantics, just better calibration), and then apply those learned thresholds to test predictions. This is minimal (only threshold calibration + consistent mapping) and should move the kappa substantially upward toward your target.'
- What this solution (achieved 0.0) has done: 'Your score gap is large (0.0 vs target 0.9293), and the biggest likely cause in this exact script is that your “OOF threshold fitting” is not truly out-of-fold because it uses a single fixed `net2` for all folds; if that model is not well-trained on this dataset (or weights aren’t the intended ones), the learned thresholds won’t rescue performance. Without changing your model architecture or adding training, the minimal legitimate way to move kappa upward is to (1) make inference use all three heads (regression, classifier, ordinal) in a calibrated probability space, then (2) fit only the final class-thresholds on OOF predictions (same metric-aligned calibration step, but with a stronger continuous signal). I also cache per-image predictions during OOF so the 5-fold loop doesn’t repeatedly decode the same images, keeping runtime under the limit while preserving the inference semantics. Submission writing remains aligned to `sample_submission.csv` to avoid ID/order mistakes that can collapse kappa.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is far from the 0.929 target, so the smallest high-impact fix is to stop calibrating thresholds on “OOF” predictions generated by a single fixed model (not truly out-of-fold), which can produce badly miscalibrated cutpoints and near-random class mapping. I keep your exact model and continuous severity computation unchanged, but fit thresholds on a simple held-out split instead (same metric-aligned post-processing, just correctly calibrated against unseen labels). I also clamp and enforce strictly increasing thresholds to avoid degenerate solutions that collapse most predictions into one class (a common cause of very low/zero QWK). Submission alignment to `sample_submission.csv` is preserved exactly so IDs/order can’t zero out the score.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is far below the 0.9293 target, so we need a minimal change that legitimately improves predictions without altering the model architecture or training loop. The biggest likely issue is that the “continuous severity” is currently dominated by the classifier/ordinal heads, which are often randomly initialized when only backbone weights are available; this can collapse predictions and tank QWK. I keep your exact forward pass and threshold calibration, but (1) detect whether the heads were actually loaded from a checkpoint and (2) if not, compute severity from the regression head only (which matches your original design intent), while still using the same optimized thresholds fitting step. This is an inference-only gating change that should move the score upward toward the target while keeping submission alignment and format unchanged.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 vs target 0.9293 gap is very large, and the most likely cause here is that `USE_AUX_HEADS=False` makes you rely only on a regression head that is *randomly initialized* when no compatible checkpoint is found, so calibration cannot recover meaningful signal. I keep your exact model/forward pass and the same “continuous severity → optimized thresholds → discretize” core logic, but change the fallback initialization so that when no checkpoint is available we still load ImageNet weights for the backbone and also initialize the three heads to a sensible “near-identity severity” starting point (regression ~ expected class, classifier ~ bias toward class priors, ordinal ~ monotonic). This is inference-only and preserves evaluation semantics while making the calibration step able to produce non-degenerate predictions, which should move QWK up substantially toward your target. I also make weight discovery slightly stricter to avoid accidentally choosing an unrelated `.pth` that partially matches shapes but produces garbage outputs.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 (far below the 0.9293 target) is most consistent with predictions being essentially untrained/constant or badly calibrated, even though the submission format is correct. To move the score upward with minimal semantic changes, I keep your exact model, transforms, and “continuous severity → thresholds → discretize” pipeline, but make the threshold fitting more reliable by optimizing on the full training set (in-sample calibration) instead of a single 80/20 split. I also make inference batched (same model outputs, just faster) so the full-train threshold fit stays within the 600s budget, and I clamp the continuous predictions into the valid [0, 4.5] range to avoid degenerate threshold solutions. Submission writing remains aligned to `sample_submission.csv` IDs exactly.'

# 9. Code solution

## === cell 0
import os
import math
import random
import time
import glob

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import torchvision.transforms as transforms
from torchvision.transforms import functional as FT

from PIL import Image, ImageChops
import cv2

import timm

from sklearn.metrics import cohen_kappa_score
from scipy.optimize import minimize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)




## === cell 1
def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, flatten=False):
        super().__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps
        self.flatten = flatten

    def forward(self, x):
        x = gem(x, p=self.p, eps=self.eps)
        if self.flatten:
            x = x.flatten(1)
        return x


class Linear(nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if torch.jit.is_scripting():
            bias = self.bias.to(dtype=input.dtype) if self.bias is not None else None
            return F.linear(input, self.weight.to(dtype=input.dtype), bias=bias)
        return F.linear(input, self.weight, self.bias)


class backboneNet_efficient(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        self.backbone = timm.create_model(
            "tf_efficientnet_b4_ns",
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        self.num_features = getattr(self.backbone, "num_features", 1792)
        self.drop_rate = 0.4

        self.rg_cls = Linear(self.num_features, 1, bias=True)
        self.cls_cls = Linear(self.num_features, 5, bias=True)
        self.ord_cls = Linear(self.num_features, 4, bias=True)

    def forward(self, x):
        feats = self.backbone(x)  # (N, C)
        if self.drop_rate and self.drop_rate > 0.0:
            feats = F.dropout(feats, p=self.drop_rate, training=self.training)
        x15 = self.rg_cls(feats)
        x16 = self.cls_cls(feats)
        x17 = self.ord_cls(feats)
        return x15, x16, x17




## === cell 2
threshold = [0.75, 1.5, 2.5, 3.5]


def regress2class(out):
    prediction = 0
    for i in range(4):
        prediction += (out.data >= threshold[i]).squeeze().cpu().item()
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


def combine3output(r_out, c_out, o_out):
    R = regress2class(r_out.data)
    _, C = torch.max(c_out.data, 1)
    C = C.squeeze().item()
    _, O = torch.max(o_out.data, 1)
    O = O.squeeze().item()
    P = (R + C + O) / 3.0
    P = int(round(P))
    return P




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




## === cell 4
CANDIDATE_DATA_DIRS = [
    "/kaggle/input/aptos2019-blindness-detection",
    "../input/aptos2019-blindness-detection",
    "/kaggle/data/aptos2019-blindness-detection",
    "../data/aptos2019-blindness-detection",
]
DATA_DIR = next(
    (p for p in CANDIDATE_DATA_DIRS if os.path.exists(os.path.join(p, "test.csv"))),
    None,
)
if DATA_DIR is None:
    raise FileNotFoundError(
        f"Could not find aptos2019-blindness-detection data dir. Tried: {CANDIDATE_DATA_DIRS}"
    )

sample_sub = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")
test_df = pd.read_csv(f"{DATA_DIR}/test.csv")
train_df = pd.read_csv(f"{DATA_DIR}/train.csv")

sample_ids = sample_sub["id_code"].astype(str).values
test_ids = test_df["id_code"].astype(str).values


def _unwrap_state_dict(state):
    if (
        isinstance(state, dict)
        and "state_dict" in state
        and isinstance(state["state_dict"], dict)
    ):
        state = state["state_dict"]
    if (
        isinstance(state, dict)
        and "model" in state
        and isinstance(state["model"], dict)
    ):
        state = state["model"]
    if isinstance(state, dict) and "net" in state and isinstance(state["net"], dict):
        state = state["net"]
    if isinstance(state, dict):
        if any(k.startswith("module.") for k in state.keys()):
            state = {k.replace("module.", "", 1): v for k, v in state.items()}
    return state


def _maybe_add_or_strip_backbone_prefix(state, model):
    if not isinstance(state, dict):
        return state
    model_keys = set(model.state_dict().keys())
    state_keys = set(state.keys())

    if len(state_keys) == 0:
        return state

    if not any(k.startswith("backbone.") for k in state_keys) and any(
        k.startswith("backbone.") for k in model_keys
    ):
        added = {("backbone." + k): v for k, v in state.items()}
        inter_old = len(model_keys.intersection(state_keys))
        inter_new = len(model_keys.intersection(set(added.keys())))
        if inter_new > inter_old:
            return added

    if any(k.startswith("backbone.") for k in state_keys) and not any(
        k.startswith("backbone.") for k in model_keys
    ):
        stripped = {k.replace("backbone.", "", 1): v for k, v in state.items()}
        inter_old = len(model_keys.intersection(state_keys))
        inter_new = len(model_keys.intersection(set(stripped.keys())))
        if inter_new > inter_old:
            return stripped

    return state


def find_best_weights_file_for_model(model, search_roots):
    candidates = []
    for root in search_roots:
        candidates.extend(glob.glob(os.path.join(root, "**", "*.pth"), recursive=True))
        candidates.extend(glob.glob(os.path.join(root, "**", "*.pt"), recursive=True))
        candidates.extend(glob.glob(os.path.join(root, "**", "*.ckpt"), recursive=True))
    candidates = list(dict.fromkeys(candidates))

    if not candidates:
        return None, None

    model_sd = model.state_dict()
    total_params = len(model_sd)

    best = None
    best_tuple = None

    for path in candidates:
        try:
            state = torch.load(path, map_location="cpu")
            state = _unwrap_state_dict(state)
            state = _maybe_add_or_strip_backbone_prefix(state, model)
            if not isinstance(state, dict) or len(state) == 0:
                continue

            shape_match = 0
            for k, v in state.items():
                if (
                    k in model_sd
                    and hasattr(v, "shape")
                    and model_sd[k].shape == v.shape
                ):
                    shape_match += 1
            matched_ratio = shape_match / max(1, total_params)

            has_any_head = int(
                any(
                    k in state
                    for k in ["rg_cls.weight", "cls_cls.weight", "ord_cls.weight"]
                )
            )

            head_dim_ok = 0
            head_shapes_ok = 0
            for head_key, expected in [
                ("rg_cls.weight", (1, getattr(model, "num_features", 1792))),
                ("cls_cls.weight", (5, getattr(model, "num_features", 1792))),
                ("ord_cls.weight", (4, getattr(model, "num_features", 1792))),
            ]:
                if head_key in state and hasattr(state[head_key], "shape"):
                    if state[head_key].shape[1] == expected[1]:
                        head_dim_ok += 1
                    if tuple(state[head_key].shape) == expected:
                        head_shapes_ok += 1

            score_tuple = (
                has_any_head,
                matched_ratio,
                head_dim_ok,
                head_shapes_ok,
                shape_match,
            )
            if best_tuple is None or score_tuple > best_tuple:
                best_tuple = score_tuple
                best = path
        except Exception:
            continue

    return best, best_tuple


transform2 = transforms.Compose(
    [
        transforms.Resize((280, 280)),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

net2 = backboneNet_efficient(pretrained=False)

SEARCH_ROOTS = [
    "/kaggle/input",
    "../input",
    "/kaggle/data",
    "../data",
    DATA_DIR,
]
WEIGHTS_PATH, WEIGHTS_SCORE = find_best_weights_file_for_model(net2, SEARCH_ROOTS)

loaded_from_checkpoint = False
if (
    WEIGHTS_PATH is not None
    and WEIGHTS_SCORE is not None
    and WEIGHTS_SCORE[0] >= 1
    and WEIGHTS_SCORE[1] >= 0.05
):
    print("Using weights:", WEIGHTS_PATH)
    print(
        "Checkpoint match score (has_any_head, matched_ratio, head_dim_ok, head_shapes_ok, shape_match):",
        WEIGHTS_SCORE,
    )
    state = torch.load(WEIGHTS_PATH, map_location="cpu")
    state = _unwrap_state_dict(state)
    state = _maybe_add_or_strip_backbone_prefix(state, net2)

    try:
        net2.load_state_dict(state, strict=True)
        print("Loaded weights with strict=True")
        loaded_from_checkpoint = True
    except RuntimeError as e:
        missing, unexpected = net2.load_state_dict(state, strict=False)
        print("Loaded weights with strict=False due to RuntimeError:", str(e)[:300])
        print(f"Missing keys: {len(missing)}; Unexpected keys: {len(unexpected)}")
        loaded = len(net2.state_dict()) - len(missing)
        if loaded >= 0.05 * len(net2.state_dict()):
            loaded_from_checkpoint = True

if not loaded_from_checkpoint:
    print(
        "WARNING: No compatible external checkpoint found. "
        "Falling back to timm pretrained EfficientNet-B4 backbone weights."
    )
    net2 = backboneNet_efficient(pretrained=True)

    y = train_df["diagnosis"].astype(int).values
    counts = np.bincount(y, minlength=5).astype(np.float64)
    priors = counts / max(1.0, counts.sum())
    priors = np.clip(priors, 1e-6, 1.0)
    prior_logits = np.log(priors)
    prior_logits = prior_logits - prior_logits.mean()

    with torch.no_grad():
        mean_sev = float((priors * np.arange(5)).sum())
        p = np.clip(mean_sev / 4.5, 1e-4, 1.0 - 1e-4)
        b = float(np.log(p / (1.0 - p)))
        net2.rg_cls.weight.zero_()
        net2.rg_cls.bias.fill_(b)

        net2.cls_cls.weight.zero_()
        net2.cls_cls.bias.copy_(
            torch.tensor(prior_logits, dtype=net2.cls_cls.bias.dtype)
        )

        cdf = np.cumsum(priors)
        p_gt = 1.0 - cdf[:4]
        p_gt = np.clip(p_gt, 1e-4, 1.0 - 1e-4)
        ord_bias = np.log(p_gt / (1.0 - p_gt))
        net2.ord_cls.weight.zero_()
        net2.ord_cls.bias.copy_(torch.tensor(ord_bias, dtype=net2.ord_cls.bias.dtype))

USE_AUX_HEADS = True
print("USE_AUX_HEADS:", USE_AUX_HEADS)

net2 = net2.to(device)
net2.eval()

test_images_dir = os.path.join(DATA_DIR, "test_images")
train_images_dir = os.path.join(DATA_DIR, "train_images")
if not os.path.isdir(test_images_dir):
    raise FileNotFoundError(f"test_images directory not found at: {test_images_dir}")
if not os.path.isdir(train_images_dir):
    raise FileNotFoundError(f"train_images directory not found at: {train_images_dir}")




## === cell 5
def apply_thresholds(preds_cont, thrs):
    thrs = np.asarray(thrs, dtype=np.float64)
    x = np.asarray(preds_cont, dtype=np.float64)
    return np.digitize(x, thrs).astype(np.int64)


def qwk(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


def _enforce_increasing_thresholds(thrs, min_gap=1e-3, lo=0.0, hi=4.5):
    thrs = np.asarray(thrs, dtype=np.float64).copy()
    thrs = np.clip(np.sort(thrs), lo, hi)
    for i in range(1, len(thrs)):
        if thrs[i] <= thrs[i - 1] + min_gap:
            thrs[i] = min(hi, thrs[i - 1] + min_gap)
    return thrs


def fit_thresholds(y_true, preds_cont, init=(0.75, 1.5, 2.5, 3.5)):
    y_true = np.asarray(y_true, dtype=np.int64)
    preds_cont = np.asarray(preds_cont, dtype=np.float64)

    def _loss(thrs):
        thrs = _enforce_increasing_thresholds(thrs)
        y_pred = apply_thresholds(preds_cont, thrs)
        return -qwk(y_true, y_pred)

    bounds = [(0.0, 4.5), (0.0, 4.5), (0.0, 4.5), (0.0, 4.5)]
    res = minimize(
        _loss, x0=np.array(init, dtype=np.float64), method="Powell", bounds=bounds
    )
    thrs = _enforce_increasing_thresholds(res.x)
    return thrs, -_loss(thrs)


def infer_continuous_severity(id_list, images_dir, cache=None, batch_size=16):
    preds = np.zeros((len(id_list),), dtype=np.float32)

    batch_tensors = []
    batch_positions = []
    batch_ids = []

    def _flush_batch():
        if not batch_tensors:
            return
        x = torch.stack(batch_tensors, dim=0).to(device, non_blocking=True)
        rg_out, cls_out, ord_out = net2(x)

        rg_cont = (torch.sigmoid(rg_out) * 4.5).squeeze(1)  # (B,)

        if USE_AUX_HEADS:
            cls_prob = F.softmax(cls_out, dim=1)  # (B,5)
            cls_ev = (
                cls_prob * torch.arange(5, device=cls_prob.device, dtype=cls_prob.dtype)
            ).sum(dim=1)

            ord_prob = ordinal2class_prob(torch.sigmoid(ord_out))  # (B,5)
            ord_ev = (
                ord_prob * torch.arange(5, device=ord_prob.device, dtype=ord_prob.dtype)
            ).sum(dim=1)

            cont = (rg_cont + cls_ev + ord_ev) / 3.0
        else:
            cont = rg_cont

        cont = cont.clamp(0.0, 4.5)

        cont_np = cont.detach().to("cpu").numpy().astype(np.float32)
        for pos, idx, v in zip(batch_positions, batch_ids, cont_np):
            preds[pos] = float(v)
            if cache is not None:
                cache[idx] = preds[pos]

        batch_tensors.clear()
        batch_positions.clear()
        batch_ids.clear()

    with torch.no_grad():
        for i, idx in enumerate(id_list):
            if cache is not None and idx in cache:
                preds[i] = float(cache[idx])
                continue

            image_name = os.path.join(images_dir, f"{idx}.png")
            img = cv2.imread(image_name)
            if img is None:
                preds[i] = 0.0
                if cache is not None:
                    cache[idx] = preds[i]
                continue

            img = crop_image_from_gray(img)
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            x1 = transform2(img)  # (3,H,W), CPU tensor
            batch_tensors.append(x1)
            batch_positions.append(i)
            batch_ids.append(idx)

            if len(batch_tensors) >= batch_size:
                _flush_batch()

        _flush_batch()

    return preds


train_ids = train_df["id_code"].astype(str).values
train_y = train_df["diagnosis"].astype(int).values

cache = {}
t0 = time.time()

train_preds_cont = infer_continuous_severity(
    train_ids, train_images_dir, cache=cache, batch_size=16
)
print(f"Train inference done. elapsed={time.time()-t0:.1f}s cache_size={len(cache)}")

opt_thrs, train_qwk = fit_thresholds(
    train_y, train_preds_cont, init=(0.75, 1.5, 2.5, 3.5)
)
print("Optimized thresholds (full-train fit):", opt_thrs)
print("Train QWK (threshold-fit objective):", train_qwk)



## === cell 6
test_cache = {}
test_preds_cont = infer_continuous_severity(
    test_ids, test_images_dir, cache=test_cache, batch_size=16
)
test_preds_cls = apply_thresholds(test_preds_cont, opt_thrs)

pred_by_id = {idx: int(p) for idx, p in zip(test_ids, test_preds_cls)}

missing = [i for i in sample_ids if i not in pred_by_id]
if missing:
    raise RuntimeError(
        f"Missing predictions for {len(missing)} ids (showing up to 10): {missing[:10]}"
    )

df = sample_sub.copy()
df["id_code"] = df["id_code"].astype(str)
df["diagnosis"] = df["id_code"].map(pred_by_id).astype(int)

if df.shape[0] != sample_sub.shape[0]:
    raise RuntimeError(
        f"Submission rows ({df.shape[0]}) != sample_submission rows ({sample_sub.shape[0]})"
    )

df.to_csv("submission.csv", index=False)

print(df.head())
print("Wrote submission.csv with shape:", df.shape)
print("Diagnosis value counts:\n", df["diagnosis"].value_counts().sort_index())
