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

0.9139246525010014

# 6. Current score

0.09471

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix three execution blockers with minimal, score-neutral changes: (1) remove the obsolete `pip install` cell and use the already-installed `timm`, (2) make device selection robust by falling back to CPU when CUDA isn’t available, and (3) load model weights only if the referenced file exists, otherwise proceed without crashing (still producing a valid submission). I also fix minor transform bugs that can break preprocessing (`is` vs `==` and `trim()` returning `None`) and ensure the submission is created for all test rows. These changes preserve the model architecture and inference logic; they only make the notebook run end-to-end and write a non-empty `submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is because the model is effectively untrained: it tries to load weights from a path that doesn’t exist in the provided dataset tree, so you’re predicting with random weights. I keep the same model and inference logic, but make weight loading robust by searching common Kaggle locations (including `/kaggle/input/**`) for the referenced filename and loading it if found. If weights still can’t be found, the code still generate a valid submission, but the main expected score lift toward your target comes from actually loading the trained checkpoint. I also ensure the loaded checkpoint format is handled safely (`state_dict` vs raw) without changing the model architecture.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score strongly indicates the submission is essentially random, which happens here because the checkpoint path is very likely not found and the model runs with random weights. To move the score toward your target with minimal, core-logic-preserving changes, I (1) make checkpoint discovery more robust by searching for the actual filename across `/kaggle/input` (including nested competition folders) and (2) broaden the loader to handle common checkpoint formats (raw `state_dict`, `state_dict` under different keys, Lightning `state_dict`, and non-strict head mismatches) while keeping the same architecture and inference path. I also add a deterministic tie-down (eval/no-grad already) and keep the same thresholds and prediction method. These changes should materially increase score if the intended weights exist somewhere in the provided dataset tree; otherwise the code still produces a valid `submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is consistent with predicting with random weights and/or mis-loading an existing checkpoint; the smallest score-improving change is to reliably locate and correctly load the intended pretrained weights if they exist anywhere under Kaggle’s mounted input folders. I keep the exact same model, transforms, and prediction (regression + fixed thresholds), but strengthen checkpoint discovery (search by basename and also by common extensions) and make the loader robust to more real-world checkpoint formats (Lightning `state_dict`, nested keys, and `ema` weights) while still preferring strict loading when possible. If no weights are found, the code still run end-to-end and produce a valid `submission.csv` (but score remain low). These changes are directly aimed at moving the score up toward your target by enabling the intended trained model to be used.'
- What this solution (achieved 0.13281) has done: 'Your 0.0 score is consistent with running inference using random weights (because the checkpoint path doesn’t exist in the provided dataset), so the smallest change that should move you toward the target is to actually use a strong pretrained initialization while keeping your exact model and inference logic. I keep the same `ThreeStage_Model`, the same regression-to-class thresholds, and the same transforms, but set `pretrained=True` for the EfficientNet-B4 backbone so the network isn’t random even without a custom checkpoint. I also enable a safe `channels_last` + autocast inference path on CUDA to stay within the 600s budget without changing outputs meaningfully. The submission writing and row alignment stay exactly the same.'
- What this solution (achieved 0.14808) has done: 'We keep your exact model, transforms, and regression-to-class thresholding, but fix a likely score bottleneck: you’re currently ignoring the model’s `final_regressor` head and only using the intermediate `r_out`. With minimal semantic change, we switch inference to `net(img, final=True)` so predictions use the intended 3-stage fusion output, which should move QWK substantially upward toward your target if the head is meaningfully trained (or at least better calibrated than `r_out`). We also make checkpoint loading slightly more robust by accepting common `*.pkl`/`*.pth` names via a wider basename search (still no architecture change). Everything else (paths, submission schema, thresholds, no training) stays the same and still produce `submission.csv` end-to-end.'
- What this solution (achieved 0.18391) has done: 'Your current score is far below the target, so we should make a small, metric-aligned improvement without changing the model architecture or training approach. The biggest low-risk gain for QWK in this setup is calibrating the 4 regression-to-class thresholds using a quick out-of-fold procedure on the provided training set, while keeping your exact network and `regress2class` logic (just making the thresholds data-driven). We run a single pass of inference on a small validation split (no training), search for thresholds that maximize quadratic weighted kappa, then use those thresholds for test-time discretization. This preserves your inference semantics (regression output + thresholding), stays within Kaggle constraints, and typically moves QWK materially upward compared to fixed generic thresholds.'
- What this solution (achieved 0.20231) has done: 'Your current gap to the target is large, and the biggest low-risk gain without changing the model/training is to calibrate the regression→class thresholds more robustly for QWK. I keep your exact model, transforms, and inference (`net(..., final=True)`), but tune thresholds using full train out-of-fold (5-fold) predictions instead of a single small holdout, which reduces variance and typically improves QWK alignment. I also make `regress2class` device-safe (avoid CPU tensor accumulation) so calibration and test discretization are consistent and deterministic. Everything else (paths, submission format, no training, no architecture changes) stays the same and still writes `submission.csv`.'
- What this solution (achieved 0.20231) has done: 'Your score is far below the target, so we should increase performance with minimal, metric-aligned changes while preserving your model and inference semantics. The biggest current issue is that “OOF” threshold tuning is not actually out-of-fold (you tune on in-sample predictions), which overfits thresholds and can hurt generalization/QWK on the leaderboard. I change threshold calibration to be truly 5-fold OOF: for each fold, tune thresholds on the other 4 folds then apply them to the held-out fold to build unbiased OOF discrete predictions, and finally pick global thresholds by tuning on the full OOF continuous predictions. This keeps your model, transforms, forward pass (`final=True`), and regression→thresholding approach intact, but should move QWK materially upward toward your target.'
- What this solution (achieved 0.22942) has done: 'Main runtime is dominated by per-image Python overhead and PIL/transform work inside tight loops (3295 train + 367 test), plus a slow pure-Python QWK/threshold tuning implementation. I keep the exact same model and transforms, but batch inference with a DataLoader (multi-worker, pinned memory, persistent workers) to amortize GPU kernel launch and Python overhead, and I ensure images are closed promptly. I also replace QWK and threshold application with vectorized NumPy implementations that are mathematically identical, drastically reducing the threshold-tuning time. Finally, I avoid repeated tensor construction in `regress2class` by using the global tuned thresholds at the end in NumPy for test predictions (same semantics).'
- What this solution (achieved 0.09471) has done: 'Your current score (0.22942) is far below the target (0.9139), so we should make a small change that materially improves predictions without altering your model architecture or training loop. The most likely remaining bottleneck is a train/test preprocessing mismatch: you apply CLAHE with a hard black background outside the circle, which can shift image statistics and hurt a pretrained EfficientNet’s features; we instead preserve the original background by applying CLAHE only inside the mask and keeping pixels outside unchanged. This keeps the same transforms and model semantics (still CLAHE + same resize/normalize), but reduces distribution shift and typically improves QWK. Additionally, we ensure any “missing image” case doesn’t silently inject zeros into calibration by excluding missing entries from threshold tuning/QWK computations (still producing a valid submission with 0s for missing test rows).'

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
import torchvision.transforms as transforms
from torchvision.transforms import functional as FT
from PIL import Image, ImageChops

import timm
import cv2

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True



## === cell 1
threshold = [0.75, 1.5, 2.5, 3.5]


_thr_cache = {"thr": None, "device": None, "dtype": None, "vals": None}


def _get_thr_tensor(out):
    global _thr_cache, threshold
    dev, dt = out.device, out.dtype
    vals = tuple(float(x) for x in threshold)
    if (
        _thr_cache["thr"] is None
        or _thr_cache["device"] != dev
        or _thr_cache["dtype"] != dt
        or _thr_cache["vals"] != vals
    ):
        _thr_cache["thr"] = torch.tensor(threshold, device=dev, dtype=dt)
        _thr_cache["device"] = dev
        _thr_cache["dtype"] = dt
        _thr_cache["vals"] = vals
    return _thr_cache["thr"]


def regress2class(out):
    out = out.view(-1)
    thr = _get_thr_tensor(out)
    pred = (out[:, None] >= thr[None, :]).sum(dim=1)
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


class fundus_clahe_circle(object):
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8), min_mask_coverage=0.10):
        self.clip_limit = float(clip_limit)
        self.tile_grid_size = tuple(tile_grid_size)
        self.min_mask_coverage = float(min_mask_coverage)

    def __call__(self, image):
        img = np.array(image)
        if img.ndim != 3 or img.shape[2] != 3:
            return image
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        h, w = bgr.shape[:2]
        r = int(0.48 * min(h, w))
        if r <= 0:
            return image
        cx, cy = w // 2, h // 2
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, (cx, cy), r, 255, thickness=-1)

        if (mask.mean() / 255.0) < self.min_mask_coverage:
            return image

        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size
        )
        l2 = clahe.apply(l)
        lab2 = cv2.merge([l2, a, b])
        bgr2 = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

        out = bgr.copy()
        out[mask == 255] = bgr2[mask == 255]

        rgb_out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb_out)




## === cell 4
def _find_files_by_basename(search_roots, basenames, max_hits=200):
    if isinstance(basenames, str):
        basenames = [basenames]
    basenames = set(basenames)

    hits = []
    for root in search_roots:
        if not root or not os.path.exists(root):
            continue
        for dirpath, _, filenames in os.walk(root):
            fn_set = set(filenames)
            intersect = basenames.intersection(fn_set)
            if intersect:
                for bn in sorted(intersect):
                    hits.append(os.path.join(dirpath, bn))
                    if len(hits) >= max_hits:
                        return hits
    return hits


def _extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for k in [
            "state_dict",
            "model_state_dict",
            "model",
            "net",
            "weights",
            "ema",
            "model_ema",
        ]:
            if k in ckpt and isinstance(ckpt[k], dict) and len(ckpt[k]) > 0:
                return ckpt[k]
        for k in ["checkpoint", "module"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                inner = ckpt[k]
                for kk in [
                    "state_dict",
                    "model_state_dict",
                    "model",
                    "net",
                    "weights",
                    "ema",
                    "model_ema",
                ]:
                    if (
                        kk in inner
                        and isinstance(inner[kk], dict)
                        and len(inner[kk]) > 0
                    ):
                        return inner[kk]
        tensor_vals = [v for v in ckpt.values() if torch.is_tensor(v)]
        if len(tensor_vals) > 0:
            return ckpt
    return ckpt


def _strip_prefixes(state_dict):
    if not isinstance(state_dict, dict) or not state_dict:
        return state_dict

    prefixes = [
        "module.",
        "model.",
        "net.",
        "backbone.",
    ]
    changed = True
    sd = state_dict
    while changed:
        changed = False
        if not isinstance(sd, dict) or not sd:
            break
        first_key = next(iter(sd.keys()))
        for p in prefixes:
            if first_key.startswith(p):
                sd = {k.replace(p, "", 1): v for k, v in sd.items()}
                changed = True
                break
    return sd


def _load_checkpoint_safely(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = _extract_state_dict(ckpt)
    state_dict = _strip_prefixes(state_dict)

    try:
        model.load_state_dict(state_dict, strict=True)
        strict_used = True
    except RuntimeError as e:
        print(
            "WARNING: strict load failed, trying strict=False. Error was:\n",
            str(e)[:800],
        )
        incompatible = model.load_state_dict(state_dict, strict=False)
        try:
            missing = incompatible.missing_keys
            unexpected = incompatible.unexpected_keys
        except Exception:
            missing, unexpected = [], []
        print(
            "Loaded with strict=False. Missing keys:",
            len(missing),
            "Unexpected keys:",
            len(unexpected),
        )
        strict_used = False
    return model, strict_used


def _qwk(y_true, y_pred, n_classes=5):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    mask = (y_true >= 0) & (y_true < n_classes) & (y_pred >= 0) & (y_pred < n_classes)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if y_true.size == 0:
        return 0.0

    O = np.bincount(
        y_true * n_classes + y_pred, minlength=n_classes * n_classes
    ).astype(np.float64)
    O = O.reshape(n_classes, n_classes)

    act_hist = np.bincount(y_true, minlength=n_classes).astype(np.float64)
    pred_hist = np.bincount(y_pred, minlength=n_classes).astype(np.float64)
    E = np.outer(act_hist, pred_hist)
    sE = E.sum()
    if sE > 0:
        E = E / sE * O.sum()

    idx = np.arange(n_classes, dtype=np.float64)
    W = (idx[:, None] - idx[None, :]) ** 2 / ((n_classes - 1) ** 2)

    num = (W * O).sum()
    den = (W * E).sum()
    if den == 0:
        return 0.0
    return 1.0 - num / den


def _apply_thresholds(preds_cont, thr):
    preds_cont = np.asarray(preds_cont, dtype=np.float64)
    thr = np.asarray(list(thr), dtype=np.float64)
    cls = (preds_cont[:, None] >= thr[None, :]).sum(axis=1).astype(np.int64)
    return np.clip(cls, 0, 4)


def _tune_thresholds(preds_cont, y_true, init_thr, iters=4, grid_points=41):
    preds_cont = np.asarray(preds_cont, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.int64)

    thr = np.array(init_thr, dtype=np.float64)
    thr.sort()
    best_thr = thr.copy()
    best_score = _qwk(y_true, _apply_thresholds(preds_cont, best_thr))

    lo = max(0.0, np.percentile(preds_cont, 1))
    hi = min(4.5, np.percentile(preds_cont, 99))
    lo, hi = float(lo), float(hi)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = 0.0, 4.5

    for _ in range(iters):
        for k in range(4):
            left = lo if k == 0 else best_thr[k - 1] + 1e-4
            right = hi if k == 3 else best_thr[k + 1] - 1e-4
            if right <= left:
                continue
            candidates = np.linspace(left, right, grid_points)
            local_best_t = best_thr[k]
            local_best_score = best_score
            for t in candidates:
                cand = best_thr.copy()
                cand[k] = t
                score = _qwk(y_true, _apply_thresholds(preds_cont, cand))
                if score > local_best_score:
                    local_best_score = score
                    local_best_t = t
            best_thr[k] = local_best_t
            best_score = local_best_score

    return best_thr.tolist(), float(best_score)


from torch.utils.data import Dataset, DataLoader


class ImageIdDataset(Dataset):
    def __init__(self, ids, img_dir, transform):
        self.ids = np.asarray(ids)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return self.ids.shape[0]

    def __getitem__(self, i):
        idx = self.ids[i]
        image_name = f"{self.img_dir}/{idx}.png"
        if not os.path.exists(image_name):
            return idx, None
        with Image.open(image_name) as im:
            img = im.convert("RGB")
        img = self.transform(img)
        return idx, img


def _collate_keep_missing(batch):
    ids = [b[0] for b in batch]
    imgs = [b[1] for b in batch]
    present_idx = [i for i, x in enumerate(imgs) if x is not None]
    if len(present_idx) == 0:
        return ids, None, present_idx
    imgs_t = torch.stack([imgs[i] for i in present_idx], dim=0)
    return ids, imgs_t, present_idx


def _infer_continuous_for_ids(
    ids, img_dir, transform, net, batch_size=16, num_workers=2
):
    ds = ImageIdDataset(ids, img_dir, transform)
    use_cuda = torch.cuda.is_available() and str(device).startswith("cuda")
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_cuda,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
        collate_fn=_collate_keep_missing,
    )

    preds = np.zeros(len(ds), dtype=np.float64)
    missing = 0
    id2pos = {str(i): p for p, i in enumerate(ids)}

    net.eval()
    with torch.no_grad():
        for step, (id_list, imgs, present_idx) in enumerate(loader):
            if step % 50 == 0:
                print("Inference batches:", step, "of", math.ceil(len(ds) / batch_size))
            if imgs is None:
                missing += len(id_list)
                continue

            if use_cuda:
                imgs = imgs.to(device, non_blocking=True).contiguous(
                    memory_format=torch.channels_last
                )
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    out = net(imgs, final=True)
            else:
                imgs = imgs.to(device)
                out = net(imgs, final=True)

            out_np = out.detach().float().cpu().view(-1).numpy()
            for j, bi in enumerate(present_idx):
                pos = id2pos[str(id_list[bi])]
                preds[pos] = float(out_np[j])

            missing += len(id_list) - len(present_idx)
    return preds, missing




## === cell 5
TEST_CSV_PATH = "../input/aptos2019-blindness-detection/test.csv"
SAMPLE_SUB_PATH = "../input/aptos2019-blindness-detection/sample_submission.csv"
TEST_IMG_DIR = "../input/aptos2019-blindness-detection/test_images"

TRAIN_CSV_PATH = "../input/aptos2019-blindness-detection/train.csv"
TRAIN_IMG_DIR = "../input/aptos2019-blindness-detection/train_images"

test_df = pd.read_csv(TEST_CSV_PATH)
test_ids = np.squeeze(test_df["id_code"].values)

input_size = 380
transform = transforms.Compose(
    [
        trim(),
        cropTo4_3(),
        fundus_clahe_circle(clip_limit=2.0, tile_grid_size=(8, 8)),
        transforms.Resize((input_size * 3 // 4, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.384, 0.258, 0.174], std=[0.124, 0.089, 0.094]),
    ]
)

net = ThreeStage_Model()

weights_path = "../input/weights/B4_3stage_45epoch_CLAHE.pkl"
weights_basename = os.path.basename(weights_path)

candidate_basenames = {
    weights_basename,
    os.path.splitext(weights_basename)[0] + ".pth",
    os.path.splitext(weights_basename)[0] + ".pt",
    os.path.splitext(weights_basename)[0] + ".bin",
    os.path.splitext(weights_basename)[0] + ".ckpt",
    os.path.splitext(weights_basename)[0] + ".pkl",
    "B4_3stage_45epoch_CLAHE.pkl",
    "B4_3stage_45epoch_CLAHE.pth",
    "B4_3stage_45epoch_CLAHE.pt",
    "B4_3stage_45epoch_CLAHE.ckpt",
}

resolved_path = None
if os.path.exists(weights_path):
    resolved_path = weights_path
else:
    search_roots = [
        "../input",
        "/kaggle/input",
        "/kaggle/data/input",
        "/kaggle/working",
    ]
    hits = _find_files_by_basename(
        search_roots, list(candidate_basenames), max_hits=200
    )
    if hits:

        def _rank(p):
            p_low = p.lower()
            bonus = 0
            if "weight" in p_low:
                bonus -= 5
            if "ckpt" in p_low or "checkpoint" in p_low:
                bonus -= 3
            if "model" in p_low:
                bonus -= 1
            return (bonus, len(p), p)

        hits = sorted(hits, key=_rank)
        resolved_path = hits[0]

if resolved_path and os.path.exists(resolved_path):
    net, strict_used = _load_checkpoint_safely(net, resolved_path)
    print("Loaded weights:", resolved_path, "| strict:", strict_used)
else:
    print(
        f"WARNING: weights file not found (looked for {sorted(candidate_basenames)} under ../input and /kaggle/*). "
        "Proceeding with ImageNet-pretrained backbone (better than random, but below target if the intended DR checkpoint is missing)."
    )

net = net.to(device)
net.eval()

use_cuda = torch.cuda.is_available() and device.startswith("cuda")
if use_cuda:
    net = net.to(memory_format=torch.channels_last)



## === cell 6
train_df = pd.read_csv(TRAIN_CSV_PATH)
train_ids = train_df["id_code"].values
train_y = train_df["diagnosis"].values.astype(int)

K = 5
rng = np.random.RandomState(42)
perm = rng.permutation(len(train_ids))
fold_id = np.empty(len(train_ids), dtype=np.int64)
for i, p in enumerate(perm):
    fold_id[p] = i % K

t0 = time.time()
num_workers = 2
batch_size = 16 if use_cuda else 4
oof_preds_cont, missing_train = _infer_continuous_for_ids(
    train_ids,
    TRAIN_IMG_DIR,
    transform,
    net,
    batch_size=batch_size,
    num_workers=num_workers,
)

print(
    "Train inference done. Missing train images:",
    missing_train,
    "time(s):",
    round(time.time() - t0, 2),
)

missing_mask = oof_preds_cont == 0.0
if missing_train > 0:
    keep_idx = np.where(~missing_mask)[0]
else:
    keep_idx = np.arange(len(train_ids))

base_pred_cls = _apply_thresholds(oof_preds_cont[keep_idx], threshold)
base_qwk = _qwk(train_y[keep_idx], base_pred_cls)
print("Base thresholds:", threshold, "Train(QWK, filtered) :", round(base_qwk, 6))

oof_pred_cls_true = np.zeros(len(train_ids), dtype=np.int64)
fold_thr = []
for f in range(K):
    tr_idx = np.where((fold_id != f) & np.isin(np.arange(len(train_ids)), keep_idx))[0]
    va_idx = np.where((fold_id == f) & np.isin(np.arange(len(train_ids)), keep_idx))[0]

    thr_f, qwk_tr = _tune_thresholds(
        oof_preds_cont[tr_idx], train_y[tr_idx], threshold, iters=5, grid_points=41
    )
    fold_thr.append(thr_f)
    oof_pred_cls_true[va_idx] = _apply_thresholds(oof_preds_cont[va_idx], thr_f)

oof_qwk_true = _qwk(train_y[keep_idx], oof_pred_cls_true[keep_idx])
print("True OOF QWK (foldwise tuned, filtered):", round(oof_qwk_true, 6))

tuned_thr, tuned_qwk = _tune_thresholds(
    oof_preds_cont[keep_idx], train_y[keep_idx], threshold, iters=5, grid_points=41
)
print(
    "Tuned thresholds (global):",
    [round(x, 6) for x in tuned_thr],
    "Train tuned QWK (filtered):",
    round(tuned_qwk, 6),
)

threshold = tuned_thr



## === cell 7
t0 = time.time()
batch_size = 16 if use_cuda else 4
test_preds_cont, missing_images = _infer_continuous_for_ids(
    test_ids,
    TEST_IMG_DIR,
    transform,
    net,
    batch_size=batch_size,
    num_workers=num_workers,
)
test_pred_cls = _apply_thresholds(test_preds_cont, threshold)

if missing_images:
    print("WARNING: missing images:", missing_images)

submission = np.empty((len(test_ids), 2), dtype=object)
submission[:, 0] = test_ids
submission[:, 1] = test_pred_cls.astype(int)

print("Test inference time(s):", round(time.time() - t0, 2))



## === cell 8
df = pd.DataFrame(submission, columns=["id_code", "diagnosis"])
df["diagnosis"] = df["diagnosis"].astype(int)

sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
df = sample_sub[["id_code"]].merge(df, on="id_code", how="left")
df["diagnosis"] = df["diagnosis"].fillna(0).astype(int)

out_path = "submission.csv"
df.to_csv(out_path, index=False)
print("Wrote:", out_path, "rows:", len(df))
print(df.head())
