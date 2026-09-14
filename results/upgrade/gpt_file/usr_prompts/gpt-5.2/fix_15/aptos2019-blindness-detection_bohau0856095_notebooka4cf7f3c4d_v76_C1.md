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

0.9217688344313156

# 6. Current score

-0.01006

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the missing weight file crash by loading the model weights only if the file exists, and otherwise fall back to a deterministic, valid baseline prediction so a submission is always produced. I also fix the CUDA crash by selecting `cuda` only when it’s actually available (otherwise run on CPU). To prevent transform failures and empty outputs, I make `trim()` always return an image and add safe image loading with a fallback, ensuring we always generate exactly one prediction per `id_code`. Finally, I keep the model/transform logic intact when weights are present, so the score can improve toward the target when the expected weights exist in the Kaggle environment.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is coming from always predicting class 0 when the weight file isn’t found, which produces a valid submission but very poor QWK. To move the score toward the 0.92 target with minimal logic changes, I keep your exact model/transform/prediction path when weights exist, but when they don’t, I switch to a deterministic label-prior baseline: predict the training-set majority class (computed from `train.csv`). This preserves evaluation semantics (still outputs integer 0–4 labels) and typically yields a noticeably higher QWK than all-zeros, without changing architecture or training. I also ensure `train.csv` is read from the same DATA_ROOT so it works in this environment.'
- What this solution (achieved 0.19197) has done: 'Your 0.0 score is almost certainly because the model weights are never found/loaded, so the script falls back to a constant-label submission. To move the score upward toward the 0.92 target with minimal changes and without altering your model logic, I (1) robustly search for the weight file under `/kaggle/input/**` (where Kaggle datasets actually mount), and (2) if weights still aren’t found, replace the constant/majority fallback with a deterministic lightweight baseline that uses simple image brightness (no training loop, no architecture change) to produce non-constant 0–4 predictions. This keeps the existing inference path identical when weights exist, but avoids the worst-case 0.0 kappa behavior when they don’t. The submission writing and column format remain unchanged.'
- What this solution (achieved 0.23183) has done: 'Your current score (0.19197) is far below the target (0.9218), so we should improve performance without changing your core model/inference logic. The most likely limiter is that the weights aren’t being loaded correctly (or are loaded but don’t match the checkpoint structure), forcing the fallback baseline; we make weight loading robust to common checkpoint formats (raw state_dict, nested `state_dict`, `model`, `net`) and tolerant of `module.` prefixes so it actually uses the trained model when available. If weights still can’t be loaded, we keep your brightness fallback but make it slightly more informative by using two simple, deterministic image statistics (mean brightness + contrast) and mapping them through per-class centroids computed from a small capped training subset (still no training loop, and very lightweight). These changes are minimal, deterministic, and directly aimed at moving QWK upward toward the target while preserving your architecture and evaluation semantics.'
- What this solution (achieved 0.23183) has done: 'Your score is far below the target, so we should make a small, legitimate improvement that increases QWK without changing your model/training logic. The biggest leverage with minimal risk is fixing the regression-to-class thresholds: your current thresholds are generic, but QWK is very sensitive to calibration; we can learn optimal 4 cutpoints from `train.csv` by matching the distribution of your model’s continuous outputs to the true label quantiles. This keeps the same network, same forward pass, and same inference approach; it only adjusts how we bin the regressor output into 0–4, which directly targets the metric. If weights are not found, we keep your centroid fallback unchanged.'
- What this solution (achieved 0.23183) has done: 'Your current score (0.23183) is far below the target (0.92177), so we need a real-but-minimal lift without changing the model/training/inference core when weights are available. The most likely reason you’re stuck low is that `has_weights` is falsely staying `False` due to an overly strict “missing keys < 20” heuristic, which can fail even when the checkpoint is correct; I change this to a robust load that treats the model as loaded if any backbone/regressor weights match, while still keeping `strict=False`. Additionally, when weights are available, your threshold calibration currently uses `net(x)` (non-final) but you ultimately classify only `r_out`; I instead calibrate cutpoints using `net(x, final=True)` so thresholds align with what you should be using for a single scalar severity score, and then also use that same final output at test-time (same network, same weights—just the intended head). These two changes are directly aimed at moving QWK upward toward the target while keeping architecture and evaluation semantics intact.'
- What this solution (achieved 0.23183) has done: 'Your current score is far below the target, so the priority is to ensure you’re actually using the trained weights; right now `has_weights` can remain `False` due to an overly strict “matched>=50 and has_backbone” gate, even when a compatible checkpoint is found and loaded with `strict=False`. I make weight loading acceptance depend on a more reliable signal (how many keys were successfully loaded, and presence of key prefixes), while keeping the same model and inference path. Next, because QWK is very sensitive to rounding, I make `regress2class()` strictly monotonic and dtype-safe (avoid `.data` and implicit CPU float accumulation), without changing semantics. Finally, I keep your threshold calibration but make it robust to edge cases (ensure strictly increasing cutpoints), which usually improves QWK when weights are actually used.'
- What this solution (achieved 0.23183) has done: 'Your current score is far below the target, so we should improve QWK by making sure you actually use the trained checkpoint when it exists, without changing the model itself. The most likely blocker is that `_load_weights_robust` can reject a successfully-loaded checkpoint due to its heuristic gate, leaving `has_weights=False` and forcing the weak fallback. I change weight loading to accept the checkpoint whenever `load_state_dict(strict=False)` loads a meaningful number of tensors (and handle common nested keys + `module.` prefixes), while keeping the same network and inference. I also make threshold calibration deterministic and aligned with the same `final=True` scalar used at test time, but otherwise keep your prediction pipeline identical.'
- What this solution (achieved 0.23183) has done: 'Your score gap to the target is large (0.23183 → 0.92177), and the most likely cause is still that the checkpoint is not being accepted/used, so you’re effectively submitting the weak fallback. I make the weight-loading acceptance robust (accept when a meaningful number of tensors actually match, instead of requiring an arbitrary ≥50) and ensure the loaded weights are moved to the correct device before inference. Then, if weights are loaded, I slightly improve the last-mile mapping to integer labels for QWK by calibrating thresholds using an in-sample discrete search that directly maximizes QWK on a small capped train subset (still no training loop, same model outputs; only binning changes). If weights are still unavailable, the fallback path remains unchanged to preserve stability.'
- What this solution (achieved -0.06348) has done: 'Your current score (0.23183) is far below the target (0.92177), so we need a small but meaningful lift without changing the model architecture, training, or the basic inference loop. The most likely cause is still that `has_weights` is frequently false (or weights don’t exist), forcing the weak centroid fallback; I strengthen the fallback by learning a deterministic mapping from (mean, std) → class directly from `train.csv` (still no training loop, just statistics), which typically beats nearest-centroid. Additionally, when weights are available, I improve the threshold calibration to use a tiny, deterministic coordinate search that directly maximizes QWK on a capped subset, but keep the same `final=True` scalar output and the same binning semantics (only cutpoints change). These changes are score-relevant, lightweight, and keep runtime within limits while preserving your core model logic.'
- What this solution (achieved 0.10522) has done: 'Your current score is far below the target, and the negative QWK strongly suggests the fallback path is producing systematically wrong (likely inverted) ordinal labels. I keep your model/inference unchanged when weights load, but I make the no-weights fallback *monotonic* by mapping a simple severity proxy (mean gray brightness) to classes using training-label quantiles—this avoids inverted mappings and typically moves QWK upward from negative with minimal logic change. I also remove the risky multiclass-logreg fallback (which can easily learn an inverted boundary and hurt QWK), and keep centroids only as a last resort. Finally, I ensure deterministic ordering and robust image reading so the submission remains valid and stable.'
- What this solution (achieved -0.10709) has done: 'Your current score (0.10522) is far below the target (0.92177), so we should make the smallest change that legitimately improves QWK without changing the model or training logic. The biggest weakness is the no-weights fallback: using raw mean brightness can be inverted (camera/exposure varies), which can systematically misorder severity. I keep the exact same model path when weights load, but strengthen only the fallback by using a more stable “dark lesion proxy” (fraction of dark pixels in the fundus) and fitting its 4 class cutpoints from `train.csv` label quantiles (still deterministic, no training loop). I also add a tiny safeguard to prefer the highest-resolution, centered crop for the fallback feature so predictions are less noisy, keeping runtime within limits.'
- What this solution (achieved -0.06114) has done: 'Your score is far below the target (higher-is-better), and the current negative QWK strongly indicates the no-weights fallback is producing an ordinal mapping that’s effectively misordered on the test set. To move the score upward with minimal change, I keep your entire model path unchanged when weights are available, but I replace the fallback “dark pixel fraction” proxy with a more stable, DR-relevant proxy: the **normalized green-channel standard deviation** (captures lesion/texture contrast more robustly across exposure). I fit its 4 cutpoints deterministically from `train.csv` label-quantiles (same idea as you already used), and apply the same cutpoints at test time; this preserves evaluation semantics (still outputs 0–4 integers) and avoids the inversion issue that can drive QWK negative. I keep your existing centroid/majority fallbacks as last resorts and keep runtime within 600s by capping per-class sampling as before.'
- What this solution (achieved -0.01006) has done: 'Your current negative QWK implies the no-weights fallback is producing an *ordinally miscalibrated* mapping on test; the smallest legitimate way to move toward the 0.92 target is to keep your entire weighted-model path unchanged, but make the fallback mapping more DR-relevant and monotonic. I keep your existing green-std/mean proxy, but (1) compute it on a ring-like mask that ignores the black background and (2) learn fallback cutpoints by directly maximizing QWK on a small capped training subset via the same deterministic coordinate search you already use for model thresholds. This does not change architecture, loss, or training loops, and it only affects the fallback path (which is what’s currently hurting you). The rest of the pipeline, submission formatting, and I/O paths remain identical.'

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
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import functional as FT
from PIL import Image, ImageChops
import cv2

from sklearn.metrics import cohen_kappa_score
import timm

device = "cuda" if torch.cuda.is_available() else "cpu"

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



## === cell 1
threshold = [0.75, 1.5, 2.5, 3.5]


def regress2class(out: torch.Tensor) -> torch.Tensor:
    out = out.reshape(-1)
    thr = torch.tensor(threshold, device=out.device, dtype=out.dtype).view(1, -1)
    pred = (out.view(-1, 1) >= thr).sum(dim=1).to(torch.int64)
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
DATA_ROOT = "../input/aptos2019-blindness-detection"
TEST_CSV = os.path.join(DATA_ROOT, "test.csv")
TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
TEST_IMG_DIR = os.path.join(DATA_ROOT, "test_images")


def find_weight_file(filename: str):
    candidates = [
        "../input/weights/" + filename,
        os.path.join(DATA_ROOT, "weights", filename),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p

    roots = ["../input", "/kaggle/input", "/kaggle/data/input"]
    for root in roots:
        if not os.path.exists(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            if filename in filenames:
                return os.path.join(dirpath, filename)
    return None


def _extract_state_dict(ckpt_obj):
    if isinstance(ckpt_obj, dict):
        for k in ["state_dict", "model_state_dict", "model", "net", "weights"]:
            if k in ckpt_obj and isinstance(ckpt_obj[k], dict):
                return ckpt_obj[k]
        if any(isinstance(v, torch.Tensor) for v in ckpt_obj.values()):
            return ckpt_obj
    return None


def _strip_module_prefix(sd: dict):
    if not isinstance(sd, dict) or len(sd) == 0:
        return sd
    if all(k.startswith("module.") for k in sd.keys()):
        return {k[len("module.") :]: v for k, v in sd.items()}
    return sd


def _count_loaded_tensors(model_sd: dict, sd: dict) -> int:
    n = 0
    for k, v in sd.items():
        if k in model_sd and isinstance(v, torch.Tensor):
            mv = model_sd[k]
            if isinstance(mv, torch.Tensor) and tuple(mv.shape) == tuple(v.shape):
                n += 1
    return n


def _load_weights_robust(net: nn.Module, weight_path: str):
    """
    Accept checkpoints when a meaningful fraction of tensors match (prevents false rejection
    that would force the fallback and yield very low QWK).
    """
    try:
        ckpt = torch.load(weight_path, map_location="cpu")
        sd = _extract_state_dict(ckpt)
        sd = _strip_module_prefix(sd)
        if sd is None or not isinstance(sd, dict) or len(sd) == 0:
            return False, (0, 0, 0, 0)

        model_sd = net.state_dict()
        loaded_compatible = _count_loaded_tensors(model_sd, sd)

        incompatible = net.load_state_dict(sd, strict=False)
        missing_keys = list(getattr(incompatible, "missing_keys", []))
        unexpected_keys = list(getattr(incompatible, "unexpected_keys", []))

        sd_keys = set(sd.keys())
        has_backbone = any(k.startswith("backbone.") for k in sd_keys)
        has_head = any(
            k.startswith("final_regressor.")
            or k.startswith("regressor.")
            or k.startswith("classifier.")
            for k in sd_keys
        )

        total_model_tensors = max(1, len(model_sd))
        frac_loaded = loaded_compatible / float(total_model_tensors)

        ok = (frac_loaded >= 0.15 and (has_backbone or has_head)) or (
            loaded_compatible >= 25 and has_backbone
        )
        return bool(ok), (
            loaded_compatible,
            len(sd_keys),
            len(missing_keys),
            len(unexpected_keys),
        )
    except Exception:
        return False, (0, 0, 0, 0)


test_ids = pd.read_csv(TEST_CSV)
test_ids = np.squeeze(test_ids["id_code"].values)

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

weight_filename = "B4_3stage_20epoch_finetune512.pkl"
weight_path = find_weight_file(weight_filename)

has_weights = False
matched_info = (0, 0, 0, 0)
if weight_path is not None:
    has_weights, matched_info = _load_weights_robust(net, weight_path)

net = net.to(device)
net.eval()

fallback_label = 0
if os.path.exists(TRAIN_CSV):
    train_df = pd.read_csv(TRAIN_CSV)
    if "diagnosis" in train_df.columns and len(train_df) > 0:
        fallback_label = int(train_df["diagnosis"].value_counts().idxmax())

TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train_images")

centroids = None

fallback_gstd_thr = None


def _safe_gray_mean_std(path: str):
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    if im is None:
        return None
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    return float(gray.mean()), float(gray.std())


def _safe_green_std_norm(path: str, crop_frac: float = 0.8):
    """
    Deterministic fallback feature: std(green_channel)/mean(green_channel) on a central crop.
    This is typically more stable than raw brightness and less prone to inverted ordering.
    """
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    if im is None:
        return None
    g = im[:, :, 1].astype(np.float32)

    h, w = g.shape[:2]
    if h < 10 or w < 10:
        return None

    ch = int(h * crop_frac)
    cw = int(w * crop_frac)
    y0 = max(0, (h - ch) // 2)
    x0 = max(0, (w - cw) // 2)
    gc = g[y0 : y0 + ch, x0 : x0 + cw]

    m = float(np.mean(gc))
    s = float(np.std(gc))
    if not np.isfinite(m) or not np.isfinite(s):
        return None
    return float(s / (m + 1e-6))


def _safe_green_std_norm_masked(path: str, crop_frac: float = 0.86):
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    if im is None:
        return None

    g = im[:, :, 1].astype(np.float32)
    h, w = g.shape[:2]
    if h < 20 or w < 20:
        return None

    ch = int(h * crop_frac)
    cw = int(w * crop_frac)
    y0 = max(0, (h - ch) // 2)
    x0 = max(0, (w - cw) // 2)
    gc = g[y0 : y0 + ch, x0 : x0 + cw]

    p = float(np.percentile(gc, 10))
    thr = max(5.0, p)
    mask = gc > thr
    if mask.sum() < 500:  # too little signal
        return None

    vals = gc[mask]
    m = float(np.mean(vals))
    s = float(np.std(vals))
    if not np.isfinite(m) or not np.isfinite(s):
        return None
    return float(s / (m + 1e-6))


def _qwk_coordinate_search_from_feature(feat: np.ndarray, y: np.ndarray):
    """
    Deterministic small coordinate search over 4 cutpoints to maximize QWK, mirroring the
    model-threshold search but applied to the fallback feature (no training, just calibration).
    """
    feat = np.asarray(feat, dtype=np.float32).reshape(-1)
    y = np.asarray(y, dtype=np.int64).reshape(-1)
    if feat.shape[0] != y.shape[0] or feat.shape[0] < 200:
        return None

    fracs = [np.mean(y <= k) for k in [0, 1, 2, 3]]
    fracs = np.clip(np.maximum.accumulate(fracs), 1e-3, 1 - 1e-3)
    thr0 = np.array([float(np.quantile(feat, q)) for q in fracs], dtype=np.float32)

    def apply_thr(thr):
        thr = np.asarray(thr, dtype=np.float32).reshape(4)
        eps = 1e-6
        for i in range(1, 4):
            if thr[i] <= thr[i - 1] + eps:
                thr[i] = thr[i - 1] + eps
        pr = np.zeros(feat.shape[0], dtype=np.int64)
        pr += (feat >= thr[0]).astype(np.int64)
        pr += (feat >= thr[1]).astype(np.int64)
        pr += (feat >= thr[2]).astype(np.int64)
        pr += (feat >= thr[3]).astype(np.int64)
        pr = np.clip(pr, 0, 4)
        return pr, thr

    base_pred, thr0 = apply_thr(thr0)
    best_kappa = cohen_kappa_score(y, base_pred, weights="quadratic")
    best_thr = thr0.copy()

    deltas = np.array(
        [-0.35, -0.22, -0.12, -0.06, 0.0, 0.06, 0.12, 0.22, 0.35], dtype=np.float32
    )
    for _ in range(3):
        improved = False
        for i in range(4):
            cand_best_k = best_kappa
            cand_best_thr = best_thr.copy()
            for d in deltas:
                thr_c = best_thr.copy()
                thr_c[i] = float(thr_c[i] + d)
                pr_c, thr_c2 = apply_thr(thr_c)
                k = cohen_kappa_score(y, pr_c, weights="quadratic")
                if k > cand_best_k + 1e-6:
                    cand_best_k = k
                    cand_best_thr = thr_c2.copy()
            if cand_best_k > best_kappa + 1e-6:
                best_kappa = cand_best_k
                best_thr = cand_best_thr
                improved = True
        if not improved:
            break

    return best_thr.tolist()


if (not has_weights) and os.path.isdir(TRAIN_IMG_DIR) and os.path.exists(TRAIN_CSV):
    train_df = pd.read_csv(TRAIN_CSV)[["id_code", "diagnosis"]].copy()
    train_df["diagnosis"] = train_df["diagnosis"].astype(int).clip(0, 4)

    per_class_cap = 140
    parts = []
    for cls in range(5):
        sub = train_df[train_df["diagnosis"] == cls]
        if len(sub) > 0:
            parts.append(sub.head(per_class_cap))
    sample_df = pd.concat(parts, axis=0, ignore_index=True)

    feats = {cls: [] for cls in range(5)}

    all_gstd = []
    all_gstd_y = []

    for rid, cls in zip(sample_df["id_code"].tolist(), sample_df["diagnosis"].tolist()):
        p = os.path.join(TRAIN_IMG_DIR, f"{rid}.png")

        try:
            ms = _safe_gray_mean_std(p)
        except Exception:
            ms = None
        if ms is not None:
            m, s = ms
            feats[int(cls)].append((m, s))

        try:
            gsn = _safe_green_std_norm_masked(p, crop_frac=0.86)
        except Exception:
            gsn = None
        if gsn is not None:
            all_gstd.append(float(gsn))
            all_gstd_y.append(int(cls))

    c = []
    for cls in range(5):
        arr = np.array(feats[cls], dtype=float)
        if arr.shape[0] >= 5:
            c.append(arr.mean(axis=0))
        else:
            c.append(np.array([np.nan, np.nan], dtype=float))
    c = np.stack(c, axis=0)
    if np.isfinite(c).sum() >= 6:
        global_mean = np.nanmean(c, axis=0)
        c = np.where(np.isfinite(c), c, global_mean)
        centroids = c  # shape (5,2)

    if len(all_gstd) >= 200:
        feat = np.asarray(all_gstd, dtype=np.float32)
        y = np.asarray(all_gstd_y, dtype=np.int64)
        thr = _qwk_coordinate_search_from_feature(feat=feat, y=y)
        if thr is not None and len(thr) == 4:
            fallback_gstd_thr = thr


def _calibrate_thresholds_qwk_search(
    net, transform, train_csv, train_img_dir, device, max_per_class=140
):
    """
    Keep the exact same scalar output net(x, final=True), but tune the 4 cutpoints with a small
    deterministic coordinate search to directly maximize QWK.
    """
    df = pd.read_csv(train_csv)[["id_code", "diagnosis"]].copy()
    df["diagnosis"] = df["diagnosis"].astype(int).clip(0, 4)

    parts = []
    for cls in range(5):
        sub = df[df["diagnosis"] == cls]
        if len(sub) > 0:
            parts.append(sub.head(max_per_class))
    sdf = pd.concat(parts, axis=0, ignore_index=True)
    if len(sdf) < 120:
        return None

    preds = []
    y = []
    net.eval()
    with torch.no_grad():
        for rid, cls in zip(sdf["id_code"].tolist(), sdf["diagnosis"].tolist()):
            p = os.path.join(train_img_dir, f"{rid}.png")
            try:
                img = Image.open(p).convert("RGB")
            except Exception:
                continue
            x = transform(img).unsqueeze(0).to(device)
            try:
                v = float(net(x, final=True).reshape(-1).detach().cpu().item())
            except Exception:
                continue
            preds.append(v)
            y.append(int(cls))

    if len(preds) < 120:
        return None

    preds = np.asarray(preds, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)

    fracs = [np.mean(y <= k) for k in [0, 1, 2, 3]]
    fracs = np.clip(np.maximum.accumulate(fracs), 1e-3, 1 - 1e-3)
    thr0 = np.array([float(np.quantile(preds, q)) for q in fracs], dtype=np.float32)
    thr0 = np.clip(thr0, 0.0, 4.5)

    def apply_thr(thr):
        thr = np.asarray(thr, dtype=np.float32).reshape(
            4,
        )
        thr = np.clip(thr, 0.0, 4.5)
        eps = 1e-4
        for i in range(1, 4):
            if thr[i] <= thr[i - 1] + eps:
                thr[i] = min(4.5, thr[i - 1] + eps)
        pr = np.zeros(preds.shape[0], dtype=np.int64)
        pr += (preds >= thr[0]).astype(np.int64)
        pr += (preds >= thr[1]).astype(np.int64)
        pr += (preds >= thr[2]).astype(np.int64)
        pr += (preds >= thr[3]).astype(np.int64)
        pr = np.clip(pr, 0, 4)
        return pr, thr

    base_pred, thr0 = apply_thr(thr0)
    best_kappa = cohen_kappa_score(y, base_pred, weights="quadratic")
    best_thr = thr0.copy()

    deltas = np.array(
        [-0.35, -0.22, -0.12, -0.06, 0.0, 0.06, 0.12, 0.22, 0.35], dtype=np.float32
    )
    for _ in range(3):
        improved = False
        for i in range(4):
            cand_best_k = best_kappa
            cand_best_thr = best_thr.copy()
            for d in deltas:
                thr_c = best_thr.copy()
                thr_c[i] = float(thr_c[i] + d)
                pr_c, thr_c2 = apply_thr(thr_c)
                k = cohen_kappa_score(y, pr_c, weights="quadratic")
                if k > cand_best_k + 1e-6:
                    cand_best_k = k
                    cand_best_thr = thr_c2.copy()
            if cand_best_k > best_kappa + 1e-6:
                best_kappa = cand_best_k
                best_thr = cand_best_thr
                improved = True
        if not improved:
            break

    return best_thr.tolist()


if has_weights and os.path.exists(TRAIN_CSV) and os.path.isdir(TRAIN_IMG_DIR):
    thr = _calibrate_thresholds_qwk_search(
        net=net,
        transform=transform,
        train_csv=TRAIN_CSV,
        train_img_dir=TRAIN_IMG_DIR,
        device=device,
        max_per_class=140,
    )
    if thr is not None and len(thr) == 4:
        threshold = thr



## === cell 5
submission = []

for i, idx in enumerate(test_ids):
    image_name = os.path.join(TEST_IMG_DIR, f"{idx}.png")
    if has_weights:
        try:
            img = Image.open(image_name).convert("RGB")
        except Exception:
            img = Image.new("RGB", (input_size, input_size), (0, 0, 0))
        img = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            out = net(img, final=True).reshape(-1)
            pred = regress2class(out)
        submission.append([idx, int(pred.item())])
    else:
        pred_label = fallback_label

        gsn = None
        try:
            gsn = _safe_green_std_norm_masked(image_name, crop_frac=0.86)
        except Exception:
            gsn = None

        if (fallback_gstd_thr is not None) and (gsn is not None):
            v = float(gsn)
            thr = np.asarray(fallback_gstd_thr, dtype=np.float32).reshape(4)
            pred_label = int(
                (v >= thr[0]) + (v >= thr[1]) + (v >= thr[2]) + (v >= thr[3])
            )
        else:
            ms = None
            try:
                ms = _safe_gray_mean_std(image_name)
            except Exception:
                ms = None

            if centroids is not None and ms is not None:
                try:
                    x = np.array([float(ms[0]), float(ms[1])], dtype=float)
                    d = ((centroids - x[None, :]) ** 2).sum(axis=1)
                    pred_label = int(np.argmin(d))
                except Exception:
                    pred_label = fallback_label

        submission.append([idx, int(pred_label)])

submission = np.array(submission, dtype=object)



## === cell 6
df = pd.DataFrame(submission, columns=["id_code", "diagnosis"])

df["id_code"] = df["id_code"].astype(str)
df["diagnosis"] = (
    pd.to_numeric(df["diagnosis"], errors="coerce").fillna(0).astype(int).clip(0, 4)
)

assert len(df) == len(test_ids) and len(df) > 0

df.to_csv("submission.csv", index=False)
print(df.head())
print(
    "Wrote submission.csv with",
    len(df),
    "rows. Weights_loaded=",
    has_weights,
    "device=",
    device,
    "fallback_label=",
    fallback_label,
    "weight_path=",
    weight_path,
    "loaded_compatible=",
    matched_info[0],
    "ckpt_keys=",
    matched_info[1],
    "missing_keys=",
    matched_info[2],
    "unexpected_keys=",
    matched_info[3],
    "centroid_baseline=",
    centroids is not None,
    "gstd_thr_baseline=",
    fallback_gstd_thr is not None,
    "thresholds=",
    threshold,
)
