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

0.9197503695792306

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the missing-weight crash by searching for the provided checkpoint under the Kaggle input directory (and fall back to running with random weights if it truly doesn’t exist, so a CSV is still produced). I also fix the hardcoded CUDA device to automatically use GPU when available or CPU otherwise, which resolves the “no NVIDIA driver” runtime error. Finally, I make the `trim()` transform always return an image (it currently returns `None` for some inputs) and switch inference to `torch.no_grad()` with a DataLoader so the submission is reliably non-empty and generated end-to-end.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is consistent with a silent “random-weights inference” situation (checkpoint not found/loaded), plus your current post-processing ignores the model’s stronger “final” head that is designed to combine classifier/regressor/ordinal outputs. I make two minimal, score-relevant changes: (1) make checkpoint loading robust to common key mismatches (e.g., `model.`, `net.` prefixes) and fall back to a clear warning only if truly missing; and (2) switch inference to use `net(batch, final=True)` and then apply the same `regress2class` thresholds on that final regression output. This preserves your architecture and overall semantics (still produces integer classes 0–4 from a regressed score), but should move the score upward toward the target by actually using the intended trained head and correctly loading weights. The submission writing and row alignment with `sample_submission.csv` are kept intact.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 is most consistent with “weights not actually loaded” (so inference is effectively random), even if the script completes and writes a CSV. I make checkpoint loading more tolerant (handle common wrappers like `{'state_dict': ...}` plus prefix stripping, and fall back to `strict=False` while reporting missing/unexpected keys) so the intended trained weights are actually used. I also make the test image path resolution robust to either `../input/...` or `/kaggle/input/...` so you don’t silently read from a non-existent directory, and I clamp predictions to `[0,4]` to avoid any out-of-range edge cases that can hurt kappa. Core model/forward logic, transforms, and the “final head + regress2class thresholds” inference semantics are preserved.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with the model never actually using trained weights (checkpoint not found/loaded correctly), so predictions are effectively random even though a CSV is produced. I make the checkpoint search include Kaggle’s `/kaggle/data/input/...` mirror and also ensure we choose the most likely correct file (prefer exact name matches before fuzzy glob hits). I also make checkpoint loading robust to checkpoints saved as `{"model": ...}` / `{"net": ...}` / `{"state_dict": ...}` and explicitly print a hard failure if nothing loads, so you don’t silently submit random predictions. These are minimal, score-relevant changes that preserve your architecture, transforms, and inference semantics (`final=True` + `regress2class` thresholds).'
- What this solution (achieved 0.0) has done: 'Your 0.0 score strongly suggests the model is still effectively running with random weights (checkpoint not found or not actually matching this architecture), so the most score-relevant minimal change is to reliably locate and load a valid checkpoint or fail loudly instead of silently continuing. I tighten checkpoint discovery to prefer files that look like the intended 3-stage B4 weights, add a more robust state_dict extraction (including nested keys and EMA), and verify that a meaningful fraction of parameters were loaded before proceeding. If a checkpoint truly can’t be loaded, the script still write a valid submission.csv (so you can test), but it clearly warn that the score remain near-random. Core model, transforms, inference (`final=True` + `regress2class`), and submission formatting are preserved.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is still most consistent with “the checkpoint isn’t actually being loaded into the right module names,” so the model effectively predicts near-random classes even though a valid CSV is produced. I make checkpoint loading slightly more robust by (a) trying a small set of key-normalization variants (including the common case where `backbone.` should NOT be stripped) and (b) selecting the best-matching variant by parameter coverage, then loading that one. I also fix a subtle inference bug where `regress2class()` always creates predictions on CPU float and relies on `.data` and shape quirks; I keep identical thresholding semantics but implement it safely and deterministically on the correct device and dtype. These are minimal changes that preserve your architecture, transforms, and “final head + thresholds” evaluation semantics, and are directly aimed at moving the score up toward the target.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score strongly indicates the submission predictions are effectively random, which in your pipeline most plausibly happens when the test images are not actually being read from the intended directory (path resolves but points to the wrong nested folder) and/or when the checkpoint is “loaded” but not into the correct keys (coverage looks OK but semantically mismatched). I make two minimal, score-relevant fixes: (1) make `resolve_base_dir()` choose the *best* candidate by verifying that `test_images/` contains the expected PNGs from `test.csv` (not just that the directory exists), and (2) make checkpoint selection prefer a plausible large weights file (by size floor and name) and refuse tiny/invalid checkpoints that lead to near-random inference. These changes preserve your exact model, transforms, inference (`final=True` + `regress2class` thresholds), and submission formatting, but should move the score upward toward your target by ensuring you run inference on the real test set with real trained weights.'

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
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as transforms
from torchvision.transforms import functional as FT
from PIL import Image, ImageChops

from sklearn.metrics import cohen_kappa_score
import timm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)



## === cell 1
threshold = [0.75, 1.5, 2.5, 3.5]


def regress2class(out: torch.Tensor) -> torch.Tensor:
    out = out.view(-1)
    thr = torch.tensor(threshold, device=out.device, dtype=out.dtype).view(1, -1)
    pred = (out.view(-1, 1) >= thr).sum(dim=1)
    return pred


def ordinal2class_prob(out):
    pred_prob = torch.zeros(out.size(0), 5).to(out.device)
    pred_prob[:, 0] = (1 - out[:, 0]).squeeze()
    pred_prob[:, 1] = (out[:, 0] * (1 - out[:, 1])).squeeze()
    pred_prob[:, 2] = (out[:, 1] * (1 - out[:, 2])).squeeze()
    pred_prob[:, 3] = (out[:, 2] * (1 - out[:, 3])).squeeze()
    pred_prob[:, 4] = out[:, 3].squeeze()
    return F.softmax(pred_prob, dim=1)


def regress2class_prob(out):
    pred_prob = torch.zeros((out.size(0), 5)).to(out.device)
    for i in range(out.size(0)):
        if out[i] < 4.0:
            l1 = int(math.floor(out[i]))
            l2 = int(math.ceil(out[i]))
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
def _candidate_score(base_dir: str) -> float:
    test_csv = os.path.join(base_dir, "test.csv")
    test_img_dir = os.path.join(base_dir, "test_images")
    if not (os.path.exists(test_csv) and os.path.isdir(test_img_dir)):
        return -1.0

    try:
        df = pd.read_csv(test_csv)
        if "id_code" not in df.columns or len(df) == 0:
            return -1.0
        ids = df["id_code"].astype(str).tolist()
    except Exception:
        return -1.0

    hits = 0
    for idc in ids[: min(25, len(ids))]:
        if os.path.exists(os.path.join(test_img_dir, f"{idc}.png")):
            hits += 1

    png_count = len(glob.glob(os.path.join(test_img_dir, "*.png")))
    s = 0.0
    s += hits * 10.0
    s += min(png_count, 500) / 50.0  # small tie-breaker
    path_lower = base_dir.lower()
    if "aptos2019-blindness-detection" in path_lower:
        s += 1.0
    if "/kaggle/input/" in path_lower:
        s += 0.5
    if (
        path_lower.rstrip("/").endswith("test_images")
        or "test_images/test_images" in path_lower
    ):
        s -= 0.5
    return s


def resolve_base_dir():
    candidates = [
        "../input/aptos2019-blindness-detection",
        "/kaggle/input/aptos2019-blindness-detection",
        "/kaggle/data/input/aptos2019-blindness-detection",
        "/kaggle/data/aptos2019-blindness-detection",
    ]

    discovered = glob.glob(
        "/kaggle/**/aptos2019-blindness-detection/test.csv", recursive=True
    )
    for p in discovered:
        candidates.append(os.path.dirname(p))

    scored = []
    for c in sorted(set(candidates)):
        scored.append((c, _candidate_score(c)))

    scored = sorted(scored, key=lambda x: (-x[1], x[0]))
    best_dir, best_score = (
        scored[0] if scored else ("../input/aptos2019-blindness-detection", -1.0)
    )
    if best_score < 0:
        return "../input/aptos2019-blindness-detection"
    return best_dir


BASE = resolve_base_dir()
TEST_CSV = os.path.join(BASE, "test.csv")
TEST_IMG_DIR = os.path.join(BASE, "test_images")

test_df = pd.read_csv(TEST_CSV)
test_ids = test_df["id_code"].astype(str).tolist()

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


def find_checkpoint():
    preferred_names = [
        "B4_3stage_10epoch_finetune512.pkl",
        "B4_3stage_10epoch_finetune512.pth",
        "B4_3stage_10epoch_finetune512.pt",
    ]
    roots = [
        "../input",
        "/kaggle/input",
        "/kaggle/data/input",
        "/kaggle/data",
        BASE,
    ]

    candidates = []
    for r in roots:
        for name in preferred_names:
            candidates += glob.glob(os.path.join(r, "**", name), recursive=True)

    if not candidates:
        fuzzy_patterns = [
            "**/*3stage*512*.pkl",
            "**/*3stage*512*.pth",
            "**/*3stage*512*.pt",
            "**/*B4*3stage*.pkl",
            "**/*B4*3stage*.pth",
            "**/*B4*3stage*.pt",
        ]
        for r in roots:
            for pat in fuzzy_patterns:
                candidates += glob.glob(os.path.join(r, pat), recursive=True)

    if not candidates:
        return None

    def score(p):
        fn = os.path.basename(p).lower()
        s = 0.0
        if "b4" in fn:
            s += 10.0
        if "3stage" in fn or "three" in fn:
            s += 10.0
        if "512" in fn:
            s += 5.0
        if "finetune" in fn:
            s += 2.0
        if fn.endswith(".pth") or fn.endswith(".pt"):
            s += 1.0

        try:
            mb = os.path.getsize(p) / (1024 * 1024)
            if mb >= 20:
                s += 8.0
            elif mb >= 5:
                s += 2.0
            else:
                s -= 10.0  # strongly downrank tiny "checkpoints"
            s += min(mb, 200) / 50.0
        except OSError:
            s -= 5.0
        return s

    candidates = sorted(set(candidates), key=lambda p: (-score(p), p))
    return candidates[0]


def _extract_state_dict(obj):
    if isinstance(obj, dict):
        for k in (
            "state_dict",
            "model_state_dict",
            "model",
            "net",
            "weights",
            "params",
        ):
            if k in obj and isinstance(obj[k], dict):
                obj = obj[k]
                break
        for k in ("ema", "ema_state_dict"):
            if k in obj and isinstance(obj[k], dict):
                if any(isinstance(kk, str) for kk in obj[k].keys()):
                    obj = obj[k]
                    break
    return obj


def _normalize_state_dict_keys_variant(state_dict, variant: str):
    if not isinstance(state_dict, dict):
        return state_dict
    sd = state_dict

    if variant in ("keep_backbone", "strip_backbone", "strip_more"):
        if len(sd) > 0:
            keys = list(sd.keys())
            if all(
                isinstance(k, str) and k.startswith("module.")
                for k in keys[: min(50, len(keys))]
            ):
                sd = {k[len("module.") :]: v for k, v in sd.items()}

    if variant in ("keep_backbone", "strip_backbone", "strip_more"):
        keys = list(sd.keys())
        for p in ("model.", "net."):
            if len(keys) > 0 and all(
                isinstance(k, str) and k.startswith(p)
                for k in keys[: min(50, len(keys))]
            ):
                sd = {k[len(p) :]: v for k, v in sd.items()}
                keys = list(sd.keys())

    if variant == "strip_backbone":
        keys = list(sd.keys())
        if len(keys) > 0 and all(
            isinstance(k, str) and k.startswith("backbone.")
            for k in keys[: min(50, len(keys))]
        ):
            sd = {k[len("backbone.") :]: v for k, v in sd.items()}

    if variant == "strip_more":
        keys = list(sd.keys())
        for p in ("encoder.",):
            if len(keys) > 0 and all(
                isinstance(k, str) and k.startswith(p)
                for k in keys[: min(50, len(keys))]
            ):
                sd = {k[len(p) :]: v for k, v in sd.items()}
                keys = list(sd.keys())

    return sd


def _try_load_state(model, state):
    if not isinstance(state, dict) or not any(isinstance(k, str) for k in state.keys()):
        return False, None, None, 0.0
    try:
        model.load_state_dict(state, strict=True)
        model_keys = set(model.state_dict().keys())
        coverage = len(set(state.keys()) & model_keys) / max(1, len(model_keys))
        return True, 0, 0, float(coverage)
    except RuntimeError:
        missing, unexpected = model.load_state_dict(state, strict=False)
        model_keys = set(model.state_dict().keys())
        loaded_keys = set(state.keys()) & model_keys
        coverage = len(loaded_keys) / max(1, len(model_keys))
        return True, len(missing), len(unexpected), float(coverage)


def _load_checkpoint_into_model(model, ckpt_path):
    raw = torch.load(ckpt_path, map_location="cpu")
    raw = _extract_state_dict(raw)

    best = (
        False,
        None,
        None,
        0.0,
        None,
        None,
    )  # ok, missing, unexpected, coverage, variant, state
    for variant in ("keep_backbone", "strip_backbone", "strip_more"):
        state = _normalize_state_dict_keys_variant(raw, variant)
        ok, missing, unexpected, coverage = _try_load_state(model, state)
        if ok and (
            coverage > best[3] + 1e-12
            or (
                abs(coverage - best[3]) <= 1e-12
                and (best[1] is None or missing < best[1])
            )
        ):
            best = (ok, missing, unexpected, coverage, variant, state)

    if not best[0]:
        raise RuntimeError(
            "Checkpoint does not look like a compatible PyTorch state_dict."
        )

    model.load_state_dict(best[5], strict=(best[1] == 0 and best[2] == 0))
    return True, best[1], best[2], best[3], best[4]


ckpt_path = find_checkpoint()
loaded_ok = False
missing_n = None
unexpected_n = None
coverage = 0.0
variant_used = None

if ckpt_path is not None and os.path.exists(ckpt_path):
    try:
        loaded_ok, missing_n, unexpected_n, coverage, variant_used = (
            _load_checkpoint_into_model(net, ckpt_path)
        )
        print("Loaded checkpoint:", ckpt_path)
        print(
            "Key normalization variant used:",
            variant_used,
            "| Missing keys:",
            missing_n,
            "| Unexpected keys:",
            unexpected_n,
            "| Coverage:",
            round(float(coverage), 4),
        )
        if coverage < 0.2:
            loaded_ok = False
            print(
                "WARNING: Checkpoint matched too few parameters; treating as NOT loaded (coverage < 0.2)."
            )
    except Exception as e:
        loaded_ok = False
        print("WARNING: Failed to load checkpoint:", ckpt_path)
        print("Exception:", repr(e))
else:
    print("WARNING: No checkpoint found. Inference will run with random weights.")

net = net.to(device)
net.eval()

if not loaded_ok:
    print(
        "WARNING: Model weights were NOT loaded successfully. Expect very low Kaggle score (~random)."
    )

print("Base dir used:", BASE)
print("Test images dir:", TEST_IMG_DIR)
print("Test images png count:", len(glob.glob(os.path.join(TEST_IMG_DIR, "*.png"))))
print("Checkpoint used:", ckpt_path)




## === cell 5
class TestDataset(Dataset):
    def __init__(self, ids, img_dir, transform=None):
        self.ids = ids
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id_code = self.ids[idx]
        image_name = os.path.join(self.img_dir, f"{id_code}.png")
        img = Image.open(image_name).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return id_code, img


ds = TestDataset(test_ids, TEST_IMG_DIR, transform=transform)
loader = DataLoader(
    ds, batch_size=8, shuffle=False, num_workers=2, pin_memory=torch.cuda.is_available()
)

submission = []
with torch.no_grad():
    for batch_ids, batch_imgs in loader:
        batch_imgs = batch_imgs.to(device, non_blocking=True)

        final_out = net(batch_imgs, final=True)  # [B,1] scaled to 0..4.5
        preds = regress2class(final_out.squeeze(1)).detach().cpu().numpy().astype(int)

        preds = np.clip(preds, 0, 4)

        for i, id_code in enumerate(batch_ids):
            submission.append([id_code, int(preds[i])])

submission = np.array(submission, dtype=object)



## === cell 6
df = pd.DataFrame(submission, columns=["id_code", "diagnosis"])
df["id_code"] = df["id_code"].astype(str)
df["diagnosis"] = df["diagnosis"].astype(int)

sample_path = os.path.join(BASE, "sample_submission.csv")
if os.path.exists(sample_path):
    sample = pd.read_csv(sample_path)
    df = sample[["id_code"]].merge(df, on="id_code", how="left")
    df["diagnosis"] = df["diagnosis"].fillna(0).astype(int)

df.to_csv("submission.csv", index=False)
print(df.head())
print("Wrote submission.csv with shape:", df.shape)
print("Base dir used:", BASE)
print("Checkpoint used:", ckpt_path)
print("Weights loaded OK:", loaded_ok)
print("Load coverage:", round(float(coverage), 4))
print("Variant used:", variant_used)
