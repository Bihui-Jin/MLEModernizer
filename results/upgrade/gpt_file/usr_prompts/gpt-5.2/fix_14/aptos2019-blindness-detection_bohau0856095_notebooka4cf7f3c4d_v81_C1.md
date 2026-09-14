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

0.9232816474245888

# 6. Current score

0.23509

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I (1) remove the failing `pip install` and make `timm` use the already-installed version, (2) make the code run on CPU when no GPU is available, and (3) fix the transform bug where `trim()` can return `None`, which was breaking inference and leading to an empty submission. I also fix a small logic bug in `ordinal2class_prob` (CPU/GPU mix) and make checkpoint loading robust by automatically picking an available weights file if the referenced one is missing; if no weights exist, it still run end-to-end with random weights and produce a valid `submission.csv`. Finally, I ensure test IDs are read correctly and inference iterates without printing every row (to stay within runtime limits) while preserving the core model and prediction logic.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is most consistent with a format/alignment or label-range issue rather than model quality, so the smallest score-improving change is to ensure predictions are in the valid class set {0,1,2,3,4} and that the submission order matches `test.csv` exactly. I keep your model and transforms identical, but make `regress2class` run purely on the same device and explicitly clamp outputs before thresholding to avoid any out-of-range/NaN edge cases. I also force the final `df` to be reindexed to `test.csv` order (even if anything ever gets appended out of order) and assert the class range before writing the CSV. These are minimal, semantics-preserving changes that prevent “silent invalid submission” patterns and should move the score upward toward the target if the weights are being loaded.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with the checkpoint not actually loading (so you’re submitting near-random predictions), rather than a modeling issue, since your submission format/alignment code is already robust. I make checkpoint discovery/load stricter and more compatible with common training wrappers by (1) searching recursively under `WEIGHTS_DIR`, (2) auto-stripping `module.` / `model.` / `net.` prefixes, and (3) selecting the candidate checkpoint that yields the highest key-overlap with your model (so we’re far more likely to load the intended weights). These are minimal changes that preserve your model and inference logic while making it much more likely you’re using real trained weights, which should move your score up toward the target. I also add a hard failure if `WEIGHTS_DIR` exists but no weights are found, to avoid silently producing a bad submission.'
- What this solution (achieved 0.0) has done: 'I (1) make checkpoint loading non-fatal by falling back to running with randomly initialized weights when no compatible checkpoint is found, so the notebook always completes and writes a submission CSV. I (2) fix the device mismatch that caused CUDA inputs to hit a CPU model by moving the model to `device` immediately after creation and ensuring checkpoint loading happens into the already-device-placed model (or loaded on CPU then applied). I (3) harden the inference loop so a single bad image can’t abort the whole run, guaranteeing a non-empty, correctly ordered `submission.csv` with valid class values 0–4. These changes preserve your model architecture and prediction logic while unblocking end-to-end execution.'
- What this solution (achieved -0.02166) has done: 'Your 0.0 score is almost certainly coming from predicting with randomly initialized weights (checkpoint not found/loaded), not from the submission format (which is already correct and clamped to 0–4). The smallest score-improving change is to make checkpoint loading strict: if `WEIGHTS_DIR` exists but no compatible checkpoint is found, fail fast instead of silently submitting random predictions. I also add a fallback to automatically use `timm`’s pretrained EfficientNet weights only when no external checkpoint is available, which keeps your architecture identical but yields meaningful predictions and should move QWK upward toward the target. Finally, I keep the inference semantics the same, but add a tiny robustness fix to always call `net(img, final=False)` explicitly to avoid accidental signature misuse.'
- What this solution (achieved 0.09924) has done: 'Your current negative QWK is consistent with “predicting the wrong thing” for this ordinal task: you’re discarding the model’s classification and ordinal heads and only thresholding the regressor output, which often collapses predictions to a narrow range and hurts kappa. I keep your exact model, weights loading, transforms, and inference loop structure, but change only the final prediction rule to use the model’s intended “three-stage” fusion: convert classifier logits to probabilities, convert ordinal outputs to class probabilities, convert regressor output to class probabilities, then sum them and take argmax. This is a minimal semantic fix (no new training, no architecture changes) that should move the score upward toward your target. I also make the probability combination robust (clamp ordinal outputs to [0,1]) and keep the submission ordering/format checks unchanged.'
- What this solution (achieved 0.19546) has done: 'Your current score (0.09924) is far below the target, and the most likely remaining issue (given your already-correct submission formatting and improved head-fusion) is a distribution mismatch: the argmax over summed probabilities can still be badly calibrated for QWK on this dataset. I keep the exact same model, weights loading, transforms, and the same three-head probability fusion, but add a minimal, metric-aligned post-processing step: fit 4 optimal thresholds on a small held-out split of the provided training set to maximize QWK, then apply those thresholds to the expected class value from the fused probabilities. This preserves the core inference semantics (still using your fused per-class probabilities) while typically moving QWK substantially upward on APTOS without retraining the network. I also keep deterministic behavior and ensure the threshold-fitting is lightweight (no image loading in the tuning loop, only model inference on a small subset).'
- What this solution (achieved 0.19768) has done: 'Your current score is far below the target, so we should increase performance with the smallest metric-aligned change that doesn’t alter the model or training. The biggest remaining lever is the threshold calibration: it’s currently fit on only 600 samples and uses a coarse coordinate grid, which is often too weak for QWK on APTOS. I keep your exact three-head probability fusion and expected-value approach, but (1) calibrate on the full validation split (no subsampling), and (2) make threshold fitting slightly more thorough (more grid points + a couple more coordinate-descent passes) while still staying within the runtime budget. This should move QWK upward toward the target without changing architecture, transforms, or inference semantics.'
- What this solution (achieved 0.19292) has done: 'Your score gap to the target is large, so the smallest legitimate lever is stronger metric-aligned threshold calibration while keeping your model, transforms, and fused-probability inference intact. I (1) speed up and stabilize calibration by doing it batched over the full validation split (larger batch, more DataLoader workers) and (2) make the threshold search slightly more thorough (a few more coordinate-descent passes and a denser grid) without changing the prediction rule. I also add deterministic settings to reduce run-to-run noise, and keep the submission ordering/format checks identical so the CSV remains valid.'
- What this solution (achieved 0.20803) has done: 'Your current gap to the target is large (0.19292 vs 0.92328), so we should improve score with a metric-aligned change that preserves your model and fused-probability inference logic. The main issue is that your threshold calibration uses only one fixed split and a simple coordinate grid, which can overfit or land in a poor local optimum for QWK. I keep the same “expected value + 4 thresholds” post-processing, but make calibration more reliable by (1) doing out-of-fold (OOF) threshold fitting across 5 stratified folds (so thresholds generalize), and (2) applying a lightweight class-prior correction on the fused probabilities based on train-label distribution (no new model, just calibration), then fitting thresholds on OOF expected values. Inference stays the same except it uses the new fitted thresholds and (optionally) the same prior correction.'
- What this solution (achieved 0.20471) has done: 'Your current QWK (0.208) is far below the target, so we should improve metric-aligned post-processing without changing the model or inference semantics. The biggest low-risk lever is to make threshold calibration more robust: instead of fitting thresholds only on OOF expected values with a local coordinate grid, we keep that but add a tiny “global” search initialization (based on label quantiles) and then refine; this often avoids poor local optima and improves QWK. I also fix a small inefficiency/bug in the OOF loop (`all_pos_fold` is unused) and make calibration/inference use batched DataLoaders for test too (same predictions, faster/less variance), staying within runtime. All changes preserve your architecture, heads fusion, expected-value prediction rule, and submission formatting.'
- What this solution (achieved 0.23509) has done: 'Your current score is far below target, so we should improve QWK with the smallest metric-aligned post-processing changes while keeping your model, transforms, and inference intact. I keep the same fused per-class probabilities and expected-value approach, but make the class-prior correction optional and tune its strength (power) jointly with thresholds using OOF predictions, selecting the setting that maximizes OOF QWK. I also remove the `softmax` from `ordinal2class_prob` (it incorrectly re-normalizes an already-valid class-probability construction and can distort the fusion), which is a minimal semantics fix that typically improves calibration for QWK. Finally, I fit thresholds using a slightly more robust two-start initialization (fixed + quantile) for each candidate power and pick the best by OOF QWK, then apply the chosen (power, thresholds) to test.'

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
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.transforms import functional as FT
from PIL import Image, ImageChops

from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold
import timm

device = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DATA_DIR = "../input/aptos2019-blindness-detection"
WEIGHTS_DIR = "../input/weights"



## === cell 1
threshold = [0.75, 1.5, 2.5, 3.5]


def regress2class(out):
    out = out.detach()
    out = torch.nan_to_num(out, nan=0.0, posinf=4.5, neginf=0.0).clamp(0.0, 4.5)
    prediction = torch.zeros(out.size(0), device=out.device, dtype=torch.long)
    for i in range(4):
        prediction += (out >= threshold[i]).to(torch.long).squeeze()
    return prediction


def ordinal2class_prob(out):
    out = out.clamp(0.0, 1.0)
    pred_prob = torch.zeros(out.size(0), 5, device=out.device, dtype=out.dtype)
    pred_prob[:, 0] = (1 - out[:, 0]).squeeze()
    pred_prob[:, 1] = (out[:, 0] * (1 - out[:, 1])).squeeze()
    pred_prob[:, 2] = (out[:, 1] * (1 - out[:, 2])).squeeze()
    pred_prob[:, 3] = (out[:, 2] * (1 - out[:, 3])).squeeze()
    pred_prob[:, 4] = out[:, 3].squeeze()
    pred_prob = torch.nan_to_num(pred_prob, nan=0.0, posinf=0.0, neginf=0.0).clamp(
        min=0.0
    )
    pred_prob = pred_prob / (pred_prob.sum(dim=1, keepdim=True) + 1e-12)
    return pred_prob


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
    pred_prob = pred_prob / (pred_prob.sum(dim=1, keepdim=True) + 1e-12)
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
def _extract_state_dict(obj):
    if isinstance(obj, nn.Module):
        return obj.state_dict()

    if isinstance(obj, dict):
        for k in ["state_dict", "model_state_dict", "model", "net", "weights"]:
            if k in obj:
                v = obj[k]
                if isinstance(v, nn.Module):
                    return v.state_dict()
                if isinstance(v, dict):
                    return v
        if len(obj) > 0 and all(isinstance(k, str) for k in obj.keys()):
            return obj
    return obj


def _strip_prefix_if_present(state_dict, prefix):
    if not isinstance(state_dict, dict):
        return state_dict
    if not any(k.startswith(prefix) for k in state_dict.keys()):
        return state_dict
    return {k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)}


def _normalize_state_keys(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict
    for pref in ["module.", "model.", "net."]:
        state_dict = _strip_prefix_if_present(state_dict, pref)
    return state_dict


def _score_state_dict_match(model, state_dict):
    if not isinstance(state_dict, dict):
        return -1
    model_keys = set(model.state_dict().keys())
    sd_keys = set(state_dict.keys())
    return len(model_keys & sd_keys)


def _collect_checkpoint_candidates(search_dirs):
    patterns = ["*.pkl", "*.pth", "*.pt", "*.bin", "*.ckpt"]
    candidates = []
    for d in search_dirs:
        if not d or not os.path.exists(d):
            continue
        for pat in patterns:
            candidates.extend(glob.glob(os.path.join(d, "**", pat), recursive=True))
    seen = set()
    out = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _find_best_checkpoint(
    model, weights_dir, preferred_path, extra_search_dirs=None, min_overlap_ratio=0.70
):
    candidates = []
    if preferred_path and os.path.exists(preferred_path):
        candidates.append(preferred_path)

    search_dirs = [weights_dir]
    if extra_search_dirs:
        search_dirs.extend(extra_search_dirs)

    candidates.extend(_collect_checkpoint_candidates(search_dirs))

    seen = set()
    candidates = [c for c in candidates if not (c in seen or seen.add(c))]

    best = None
    best_score = -1
    best_info = None

    model_nkeys = len(model.state_dict().keys())
    min_overlap = int(model_nkeys * float(min_overlap_ratio))

    for path in candidates:
        try:
            raw = torch.load(path, map_location="cpu")
            sd = _normalize_state_keys(_extract_state_dict(raw))
            s = _score_state_dict_match(model, sd)
            if s >= min_overlap and s > best_score:
                best_score = s
                best = path
                best_info = (s, len(sd) if isinstance(sd, dict) else -1)
        except Exception:
            continue

    return best, best_score, best_info




## === cell 5
class ImageIdDataset(Dataset):
    def __init__(self, ids, labels=None, img_dir=None, transform=None):
        self.ids = [str(x) for x in ids]
        self.labels = None if labels is None else np.asarray(labels, dtype=np.int64)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        idx = self.ids[i]
        path = os.path.join(self.img_dir, f"{idx}.png")
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        if self.labels is None:
            return idx, img
        return idx, img, int(self.labels[i])


def fused_prob_from_outputs(c_out, r_out, o_out):
    p_cls = F.softmax(c_out, dim=1)
    p_reg = regress2class_prob(r_out.squeeze(1))
    p_ord = ordinal2class_prob(o_out)
    p = p_cls + p_reg + p_ord
    p = torch.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0).clamp(min=0.0)
    p = p / (p.sum(dim=1, keepdim=True) + 1e-12)
    return p


def apply_prior_correction(p, train_prior, power=0.35, eps=1e-12):
    train_prior = np.asarray(train_prior, dtype=np.float64)
    train_prior = np.clip(train_prior, eps, 1.0)
    train_prior = train_prior / train_prior.sum()
    w = torch.tensor(train_prior, device=p.device, dtype=p.dtype).view(1, -1).pow(power)
    p2 = p * w
    p2 = p2 / (p2.sum(dim=1, keepdim=True) + 1e-12)
    return p2


def apply_thresholds(x, thr):
    thr = np.asarray(thr, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    return (
        (x >= thr[0]).astype(np.int64)
        + (x >= thr[1]).astype(np.int64)
        + (x >= thr[2]).astype(np.int64)
        + (x >= thr[3]).astype(np.int64)
    )


def _init_thresholds_by_label_quantiles(y_true, x_pred, eps=1e-3):
    y_true = np.asarray(y_true, dtype=np.int64)
    x_pred = np.asarray(x_pred, dtype=np.float64)
    thr = []
    for c in [0, 1, 2, 3]:
        xc = x_pred[y_true == c]
        xn = x_pred[y_true == (c + 1)]
        if len(xc) == 0 or len(xn) == 0:
            thr.append(0.5 + c)
            continue
        t = 0.5 * (np.quantile(xc, 0.90) + np.quantile(xn, 0.10))
        thr.append(float(t))
    thr = np.array(thr, dtype=np.float64)
    thr = np.maximum.accumulate(thr)
    thr[0] = np.clip(thr[0], 0.0, 4.5)
    thr[1] = np.clip(thr[1], thr[0] + eps, 4.5)
    thr[2] = np.clip(thr[2], thr[1] + eps, 4.5)
    thr[3] = np.clip(thr[3], thr[2] + eps, 4.5)
    return thr.tolist()


def fit_thresholds_for_qwk(
    y_true,
    x_pred,
    init_thr=(0.5, 1.5, 2.5, 3.5),
    n_iter=5,
    grid_points=61,
    span=1.25,
):
    y_true = np.asarray(y_true, dtype=np.int64)
    x_pred = np.asarray(x_pred, dtype=np.float64)

    thr = np.array(init_thr, dtype=np.float64)

    def score(th):
        y_hat = apply_thresholds(x_pred, th)
        return cohen_kappa_score(y_true, y_hat, weights="quadratic")

    best = score(thr)

    for _ in range(int(n_iter)):
        for k in range(4):
            lo = 0.0 if k == 0 else thr[k - 1] + 1e-3
            hi = 4.5 if k == 3 else thr[k + 1] - 1e-3
            if lo >= hi:
                continue

            grid_lo = max(lo, thr[k] - span)
            grid_hi = min(hi, thr[k] + span)
            grid = np.linspace(grid_lo, grid_hi, int(grid_points))

            best_k = thr[k]
            best_k_score = best
            for v in grid:
                cand = thr.copy()
                cand[k] = v
                s = score(cand)
                if s > best_k_score:
                    best_k_score = s
                    best_k = v
            thr[k] = best_k
            best = best_k_score

    thr = np.maximum.accumulate(thr)
    thr[0] = np.clip(thr[0], 0.0, 4.5)
    thr[3] = np.clip(thr[3], 0.0, 4.5)
    thr[1] = np.clip(thr[1], thr[0] + 1e-3, 4.5)
    thr[2] = np.clip(thr[2], thr[1] + 1e-3, 4.5)
    thr[3] = np.clip(thr[3], thr[2] + 1e-3, 4.5)
    return thr.tolist(), float(best)




## === cell 6
test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
test_ids = test_df["id_code"].astype(str).tolist()

input_size = 380

transform = transforms.Compose(
    [
        trim(),
        cropTo4_3(),
        transforms.Resize((input_size * 3 // 4, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.384, 0.258, 0.174], std=[0.124, 0.089, 0.094]),
    ]
)

preferred_ckpt = os.path.join(WEIGHTS_DIR, "B4_3stage_9epoch_finetune2.pkl")

_tmp_net_for_search = ThreeStage_Model(pretrained_backbone=False).to(device)
ckpt_path, overlap, info = _find_best_checkpoint(
    _tmp_net_for_search,
    WEIGHTS_DIR,
    preferred_ckpt,
    extra_search_dirs=[DATA_DIR, os.path.join("..", "input")],
    min_overlap_ratio=0.70,
)
del _tmp_net_for_search

if ckpt_path is None:
    if os.path.exists(WEIGHTS_DIR):
        candidates = _collect_checkpoint_candidates([WEIGHTS_DIR])
        raise RuntimeError(
            f"No suitable checkpoint found under WEIGHTS_DIR={WEIGHTS_DIR}. "
            f"Found {len(candidates)} candidate files but none match the model. "
            "Aborting to avoid producing a near-random submission."
        )

    print(
        f"Warning: WEIGHTS_DIR={WEIGHTS_DIR} does not exist; using ImageNet-pretrained backbone "
        "weights as a minimal, architecture-preserving fallback."
    )
    net = ThreeStage_Model(pretrained_backbone=True).to(device)
else:
    net = ThreeStage_Model(pretrained_backbone=False).to(device)
    state = torch.load(ckpt_path, map_location="cpu")
    state = _normalize_state_keys(_extract_state_dict(state))
    missing, unexpected = net.load_state_dict(state, strict=False)
    print(f"Loaded checkpoint: {ckpt_path}")
    print(
        f"State_dict key overlap with model: {overlap} / {len(net.state_dict().keys())}"
    )
    if missing:
        print(f"Missing keys (first 20): {missing[:20]}")
    if unexpected:
        print(f"Unexpected keys (first 20): {unexpected[:20]}")

net.eval()



## === cell 7
train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
train_ids = train_df["id_code"].astype(str).tolist()
train_y = train_df["diagnosis"].astype(int).values

_counts = np.bincount(train_y, minlength=5).astype(np.float64)
train_prior = (_counts / (_counts.sum() + 1e-12)).tolist()

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

_calib_num_workers = 2 if os.name != "nt" else 0
calib_batch = 32 if device == "cuda" else 12

oof_exp_base = np.empty(len(train_ids), dtype=np.float64)
oof_exp_base.fill(np.nan)

for fold, (_, va_idx) in enumerate(skf.split(np.zeros(len(train_y)), train_y), start=1):
    va_ids = [train_ids[i] for i in va_idx]
    va_y = train_y[va_idx]

    calib_ds = ImageIdDataset(
        va_ids,
        labels=va_y,
        img_dir=os.path.join(DATA_DIR, "train_images"),
        transform=transform,
    )
    calib_loader = DataLoader(
        calib_ds,
        batch_size=calib_batch,
        shuffle=False,
        num_workers=_calib_num_workers,
        pin_memory=(device == "cuda"),
    )

    all_exp_fold = []

    with torch.no_grad():
        for ids_batch, imgs, ys in calib_loader:
            imgs = imgs.to(device, non_blocking=(device == "cuda"))
            c_out, r_out, o_out = net(imgs, final=False)
            p = fused_prob_from_outputs(c_out, r_out, o_out)  # (B,5)
            exp = (p * torch.arange(5, device=p.device, dtype=p.dtype).view(1, -1)).sum(
                dim=1
            )
            all_exp_fold.append(exp.detach().cpu().numpy())

    exp_fold = (
        np.concatenate(all_exp_fold, axis=0)
        if len(all_exp_fold)
        else np.array([], dtype=np.float64)
    )
    if exp_fold.shape[0] != len(va_idx):
        raise RuntimeError(
            f"Fold {fold}: expected {len(va_idx)} exp values, got {exp_fold.shape[0]}"
        )

    oof_exp_base[va_idx] = exp_fold
    print(f"Fold {fold}: collected {len(va_idx)} OOF expected values")

if np.isnan(oof_exp_base).any():
    raise RuntimeError(
        "OOF calibration failed: some training samples did not receive OOF predictions."
    )


def _apply_prior_on_expected(x_exp, prior, power):
    return x_exp


store_prob = True
if store_prob:
    oof_p = np.empty((len(train_ids), 5), dtype=np.float16)
    oof_p[:] = np.nan

    for fold, (_, va_idx) in enumerate(
        skf.split(np.zeros(len(train_y)), train_y), start=1
    ):
        va_ids = [train_ids[i] for i in va_idx]
        va_y = train_y[va_idx]

        calib_ds = ImageIdDataset(
            va_ids,
            labels=va_y,
            img_dir=os.path.join(DATA_DIR, "train_images"),
            transform=transform,
        )
        calib_loader = DataLoader(
            calib_ds,
            batch_size=calib_batch,
            shuffle=False,
            num_workers=_calib_num_workers,
            pin_memory=(device == "cuda"),
        )

        prob_chunks = []
        with torch.no_grad():
            for ids_batch, imgs, ys in calib_loader:
                imgs = imgs.to(device, non_blocking=(device == "cuda"))
                c_out, r_out, o_out = net(imgs, final=False)
                p = fused_prob_from_outputs(c_out, r_out, o_out)  # (B,5)
                prob_chunks.append(p.detach().cpu().numpy().astype(np.float16))

        p_fold = (
            np.concatenate(prob_chunks, axis=0)
            if len(prob_chunks)
            else np.zeros((0, 5), dtype=np.float16)
        )
        if p_fold.shape[0] != len(va_idx):
            raise RuntimeError(
                f"Fold {fold}: expected {len(va_idx)} prob rows, got {p_fold.shape[0]}"
            )
        oof_p[va_idx] = p_fold
        print(f"Fold {fold}: stored {len(va_idx)} OOF probability rows")

    if np.isnan(oof_p.astype(np.float32)).any():
        raise RuntimeError("OOF probability storage failed: NaNs encountered.")

powers = [0.0, 0.15, 0.25, 0.35, 0.5]
best_cfg = None

for pw in powers:
    p32 = oof_p.astype(np.float32)
    if pw > 0.0:
        w = (np.asarray(train_prior, dtype=np.float32) ** float(pw)).reshape(1, 5)
        p32 = p32 * w
        p32 = p32 / (p32.sum(axis=1, keepdims=True) + 1e-12)
    exp_pw = (
        (p32 * np.arange(5, dtype=np.float32).reshape(1, 5))
        .sum(axis=1)
        .astype(np.float64)
    )

    init_thr_q = _init_thresholds_by_label_quantiles(train_y, exp_pw)

    thr1, qwk1 = fit_thresholds_for_qwk(
        train_y, exp_pw, init_thr=threshold, n_iter=11, grid_points=161, span=1.75
    )
    thr2, qwk2 = fit_thresholds_for_qwk(
        train_y, exp_pw, init_thr=init_thr_q, n_iter=11, grid_points=161, span=1.75
    )

    if qwk2 > qwk1:
        fitted_thr, best_qwk = thr2, qwk2
        init_used = "quantile"
    else:
        fitted_thr, best_qwk = thr1, qwk1
        init_used = "fixed"

    print(f"[power={pw}] OOF QWK: {best_qwk:.6f} (init={init_used}) thr={fitted_thr}")

    if (best_cfg is None) or (best_qwk > best_cfg["qwk"]):
        best_cfg = {"power": float(pw), "thr": fitted_thr, "qwk": float(best_qwk)}

selected_power = best_cfg["power"]
fitted_thr = best_cfg["thr"]
best_qwk = best_cfg["qwk"]

print("Train prior:", train_prior)
print("Selected prior power:", selected_power)
print("Selected fitted thresholds (OOF):", fitted_thr, " | OOF QWK:", best_qwk)




## === cell 8
class TestImageDataset(Dataset):
    def __init__(self, ids, img_dir, transform):
        self.ids = [str(x) for x in ids]
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        idx = self.ids[i]
        path = os.path.join(self.img_dir, f"{idx}.png")
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return idx, img


test_img_dir = os.path.join(DATA_DIR, "test_images")
test_ds = TestImageDataset(test_ids, img_dir=test_img_dir, transform=transform)
_test_num_workers = 2 if os.name != "nt" else 0
test_batch = 32 if device == "cuda" else 12
test_loader = DataLoader(
    test_ds,
    batch_size=test_batch,
    shuffle=False,
    num_workers=_test_num_workers,
    pin_memory=(device == "cuda"),
)

submission = []
failed_images = 0

with torch.no_grad():
    for ids_batch, imgs in test_loader:
        try:
            imgs = imgs.to(device, non_blocking=(device == "cuda"))
            c_out, r_out, o_out = net(imgs, final=False)
            p = fused_prob_from_outputs(c_out, r_out, o_out)  # (B,5)

            if selected_power > 0.0:
                p = apply_prior_correction(
                    p, train_prior=train_prior, power=selected_power
                )

            exp = (p * torch.arange(5, device=p.device, dtype=p.dtype).view(1, -1)).sum(
                dim=1
            )
            exp_np = exp.detach().cpu().numpy()
            pred_np = apply_thresholds(exp_np, fitted_thr).astype(np.int64)
            pred_np = np.clip(pred_np, 0, 4)

            for idx, pred in zip(ids_batch, pred_np.tolist()):
                submission.append([str(idx), int(pred)])
        except Exception:
            failed_images += len(ids_batch)
            for idx in ids_batch:
                submission.append([str(idx), 0])

submission = np.array(submission, dtype=object)

df = pd.DataFrame(submission, columns=["id_code", "diagnosis"])
df["id_code"] = df["id_code"].astype(str)
df["diagnosis"] = df["diagnosis"].astype(int)

if len(df) == 0:
    raise RuntimeError(
        "Submission DataFrame is empty; inference did not produce any rows."
    )
if len(df) != len(test_ids):
    raise RuntimeError(
        f"Submission row count {len(df)} != test row count {len(test_ids)}"
    )

df = test_df[["id_code"]].merge(df, on="id_code", how="left", validate="one_to_one")
if df["diagnosis"].isna().any():
    raise RuntimeError(
        "Some test ids did not receive predictions; submission would be invalid/misaligned."
    )
df["diagnosis"] = df["diagnosis"].astype(int).clip(0, 4)

u = sorted(df["diagnosis"].unique().tolist())
if (df["diagnosis"].min() < 0) or (df["diagnosis"].max() > 4):
    raise RuntimeError("Predicted diagnosis out of valid range 0..4.")
if len(u) == 0:
    raise RuntimeError("No predictions found.")

df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", df.shape)
if failed_images:
    print("Warning: failed images encountered:", failed_images)
print("Selected prior power:", selected_power)
print("Diagnosis unique values:", u)
