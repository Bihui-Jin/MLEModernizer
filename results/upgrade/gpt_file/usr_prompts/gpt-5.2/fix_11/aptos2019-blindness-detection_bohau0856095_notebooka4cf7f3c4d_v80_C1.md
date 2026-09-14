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

0.923969709786398

# 6. Current score

0.13175

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I (1) remove the failing custom `pip install` and instead use the already-installed `timm`, (2) make the code robust to CPU-only Kaggle sessions by selecting `cuda` only when available and by removing hard `.cuda()` tensor allocations, and (3) fix the missing weights issue by safely falling back to an untrained model (still producing a valid, non-empty submission) while emitting a clear warning. I also fix a couple of transform bugs (`trim()` returning `None`, and `is` vs `==` for string comparison) that can cause runtime errors. Finally, I keep the model/thresholding logic intact and ensure `submission.csv` is always written with the correct columns and row count.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is because the model is running with random weights (the specified `WEIGHTS_PATH` doesn’t exist in this environment), so the smallest meaningful improvement is to (a) correctly locate a weights file if one is available under the provided dataset tree and load it, and (b) if no weights exist, keep your exact inference logic but calibrate the 4 thresholds on a train-validation split to maximize quadratic weighted kappa and then apply those thresholds at test-time. This preserves your model architecture and inference semantics (still using the regressor head + `regress2class`) while making the predictions far less arbitrary, moving score upward toward your target. I also ensure `test_ids` extraction is robust (uses `id_code` explicitly) and that the submission ordering matches `test.csv`. All changes are directly tied to making predictions consistent and improving kappa without changing the core modeling approach.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is coming from running the ThreeStage model with random weights (no valid checkpoint was loaded), so the smallest legitimate move toward your target is to (1) reliably find and load an actual compatible checkpoint if it exists anywhere under `../input`, and (2) if no checkpoint exists, at least make inference+thresholding consistent and kappa-aligned by calibrating the regression thresholds on a held-out validation split and then using those thresholds for test predictions. I keep your model, transforms, and `regress2class` semantics intact, but fix a key bug where calibrated thresholds were not actually used by `regress2class` (it referenced the global `threshold` list, but you were passing raw model outputs in a slightly mismatched shape in some places). I also switch test inference to the same batched path used for calibration to reduce per-image variance/IO overhead without changing the model logic, and ensure the submission is still written exactly in the required format and order.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with producing nearly-constant or effectively random predictions (often caused by missing/incorrect weights), so the smallest change that legitimately moves toward your target is to (1) make checkpoint discovery/load more reliable by preferring filenames that match your exact model (B4 + 3stage + epoch/finetune keywords) and by properly handling common checkpoint wrappers, and (2) calibrate the regression thresholds on a validation split using only the images that were successfully inferred (fixing the current mismatch logic, which often prevents calibration from being applied at all). These keep your model architecture and inference semantics intact (same ThreeStage model, still using regressor output + thresholding), but make the discrete predictions align better with QWK. I also make the validation inference join by `id_code` so kappa is computed against the correct labels even if any images are skipped (which avoids silently wrong calibration). The submission format, ordering, and paths remain unchanged, and it still writes `submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is consistent with either (a) weights not loading so predictions are effectively random/near-constant, or (b) a submission alignment/type issue (e.g., `diagnosis` not being clean ints 0–4 in the exact `test.csv` order). I make the smallest changes that directly improve QWK without changing your model/inference semantics: (1) strengthen checkpoint loading to correctly handle common wrapper formats and reject obviously-incompatible state dicts, and (2) calibrate the 4 regression thresholds using a slightly more robust coordinate-descent search that scales steps to the model’s output range, then apply those thresholds for test predictions. I also make test-time ordering airtight by building the submission directly from `test.csv` order (so even if any id is skipped, it fails early with a clear error as before). These changes keep your ThreeStage model and regressor+thresholding approach intact, but should move your score substantially upward toward the target when any reasonable weights are found/loaded.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is overwhelmingly likely due to missing/incorrect weights leading to near-random predictions, so the smallest meaningful move toward your target is to (1) make checkpoint discovery/loading more robust and actually compatible with your `ThreeStage_Model` by handling nested/wrapped state_dicts and common key prefixes, and (2) ensure thresholds are always calibrated and applied on a correctly-aligned validation set (by joining on `id_code`) so the discrete predictions are optimized for QWK. I keep your model, transforms, inference path, and regressor-to-class semantics intact; the only behavioral changes are safer weight loading and more reliable threshold calibration/application. I also ensure the submission is built strictly in `test.csv` order with integer labels 0–4, so there’s no silent format/alignment penalty.'
- What this solution (achieved 0.63091) has done: 'Your score is 0.0 because the model is still effectively untrained in this environment (no compatible checkpoint is actually being loaded), so predictions are near-random and threshold calibration can’t rescue it. The smallest legitimate improvement toward your target is to train the exact same `ThreeStage_Model` regressor head for a short, fixed number of epochs on the provided `train_images/` (no architecture/loss changes) and then calibrate thresholds on a held-out split for QWK. I also fix a subtle weight-loading bug where stripping `"backbone."` from keys can prevent proper loading when a checkpoint actually exists. Finally, I keep the same inference+thresholding semantics and ensure the submission is written in `test.csv` order with integer labels 0–4.'
- What this solution (achieved 0.13175) has done: 'The timeout is dominated by doing full-image PIL transforms (especially `trim()` converting to numpy and scanning pixels) for every training/validation/test image, plus the fallback training path that runs 5 epochs over ~2600 images. To preserve core logic and accuracy, the biggest safe win is to cache the deterministic preprocessing results (post-`transform`) to disk and reuse them across training/val/test passes, eliminating repeated decoding + trim/crop/resize work. Additionally, we avoid the expensive recursive glob search unless the expected weights path is missing, and we reduce DataLoader overhead via better worker/prefetch settings and reusing a single dataset cache. All changes are equivalence-preserving (same images, same transforms, same model forward/training), just avoiding repeated computation and I/O.'

# 9. Code solution

## === cell 0
import os
import random
import time
import math
import warnings
import glob

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import torchvision.transforms as transforms
from torchvision.transforms import functional as FT
from PIL import Image

from sklearn.metrics import cohen_kappa_score
import timm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False



## === cell 1
threshold = [0.75, 1.5, 2.5, 3.5]


def regress2class(out):
    x = out.detach().view(-1).to("cpu")
    t = torch.tensor(threshold, dtype=x.dtype, device=x.device).view(1, -1)
    return (x.view(-1, 1) >= t).sum(dim=1)


def ordinal2class_prob(out):
    pred_prob = torch.zeros(out.size(0), 5, device=out.device)
    pred_prob[:, 0] = (1 - out[:, 0]).squeeze()
    pred_prob[:, 1] = (out[:, 0] * (1 - out[:, 1])).squeeze()
    pred_prob[:, 2] = (out[:, 1] * (1 - out[:, 2])).squeeze()
    pred_prob[:, 3] = (out[:, 2] * (1 - out[:, 3])).squeeze()
    pred_prob[:, 4] = out[:, 3].squeeze()
    return F.softmax(pred_prob, dim=1)


def regress2class_prob(out):
    out = out.view(-1).to(dtype=torch.float32)
    pred_prob = torch.zeros((out.size(0), 5), device=out.device, dtype=out.dtype)
    out_c = torch.clamp(out, 0.0, 4.0)
    l1 = torch.floor(out_c).to(torch.long)
    l2 = torch.ceil(out_c).to(torch.long)
    w1 = 1.0 - (out_c - l1.to(out.dtype))
    w2 = 1.0 - (l2.to(out.dtype) - out_c)

    pred_prob.scatter_(1, l1.unsqueeze(1), w1.unsqueeze(1))
    pred_prob.scatter_add_(1, l2.unsqueeze(1), w2.unsqueeze(1))

    mask_hi = out >= 4.0
    if mask_hi.any():
        pred_prob[mask_hi] = 0.0
        pred_prob[mask_hi, 4] = 1.0
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
        arr = np.asarray(image)
        if arr.ndim == 2:
            arr = arr[..., None]
        bg = arr[0, 0].astype(np.int16)
        diff = np.abs(arr.astype(np.int16) - bg)
        diff = diff * 2 - 10
        diff = np.clip(diff, 0, 255).astype(np.uint8)
        mask = diff.any(axis=-1) if diff.ndim == 3 else (diff != 0)
        if not mask.any():
            return image
        ys, xs = np.where(mask)
        left, right = int(xs.min()), int(xs.max()) + 1
        top, bottom = int(ys.min()), int(ys.max()) + 1
        return image.crop((left, top, right, bottom))




## === cell 4
BASE_DIR = "../input/aptos2019-blindness-detection"
TEST_CSV = os.path.join(BASE_DIR, "test.csv")
TEST_IMG_DIR = os.path.join(BASE_DIR, "test_images")
TRAIN_CSV = os.path.join(BASE_DIR, "train.csv")
TRAIN_IMG_DIR = os.path.join(BASE_DIR, "train_images")

WEIGHTS_PATH = "../input/weights/B4_3stage_7epoch_finetune2.pkl"

test_ids_df = pd.read_csv(TEST_CSV)
test_ids_df["id_code"] = test_ids_df["id_code"].astype(str)
test_ids = test_ids_df["id_code"].tolist()

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

net = ThreeStage_Model()


def _find_weights_fallback():
    candidates = []
    for pat in [
        "../input/**/*.pkl",
        "../input/**/*.pth",
        "../input/**/*.pt",
        "../input/**/*.bin",
    ]:
        candidates.extend(glob.glob(pat, recursive=True))

    def _score_path(p):
        name = os.path.basename(p).lower()
        s = 0
        if "b4" in name or "efficientnet_b4" in name:
            s += 6
        if "3stage" in name or "three" in name:
            s += 6
        if any(k in name for k in ["aptos", "blind", "retina", "dr"]):
            s += 3
        if any(k in name for k in ["epoch", "finetune", "fold", "best", "final"]):
            s += 2
        try:
            s += min(os.path.getsize(p) / 1e8, 6.0)
        except Exception:
            pass
        return s

    candidates = [p for p in candidates if os.path.isfile(p)]
    if not candidates:
        return None
    candidates = sorted(candidates, key=_score_path, reverse=True)
    return candidates[0]


def _extract_state_dict(obj):
    if isinstance(obj, dict):
        for k in [
            "state_dict",
            "model_state_dict",
            "model",
            "net",
            "weights",
            "ema",
            "teacher",
            "module",
        ]:
            if k in obj and isinstance(obj[k], dict):
                obj = obj[k]
                break

    if isinstance(obj, dict) and all(isinstance(v, torch.Tensor) for v in obj.values()):
        return obj
    return None


def _normalize_state_keys(sd):
    fixed = {}
    for k, v in sd.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module.") :]
        if nk.startswith("model."):
            nk = nk[len("model.") :]
        if nk.startswith("net."):
            nk = nk[len("net.") :]
        fixed[nk] = v
    return fixed


def _try_load_checkpoint(path):
    raw = torch.load(path, map_location="cpu")
    sd = _extract_state_dict(raw)
    if sd is None:
        raise RuntimeError(
            "Could not extract a tensor state_dict from the checkpoint object."
        )
    sd = _normalize_state_keys(sd)

    model_keys = set(net.state_dict().keys())
    sd_keys = set(sd.keys())
    overlap = len(model_keys & sd_keys) / max(1, len(model_keys))
    if overlap < 0.25:
        raise RuntimeError(
            f"Checkpoint appears incompatible (key-overlap={overlap:.2%})."
        )

    missing, unexpected = net.load_state_dict(sd, strict=False)
    return overlap, missing, unexpected


loaded_weights_path = None
loaded_ok = False

if os.path.exists(WEIGHTS_PATH):
    loaded_weights_path = WEIGHTS_PATH
else:
    loaded_weights_path = _find_weights_fallback()

if loaded_weights_path is not None:
    try:
        overlap, missing, unexpected = _try_load_checkpoint(loaded_weights_path)
        loaded_ok = True
        print(f"Loaded weights: {loaded_weights_path} (key-overlap={overlap:.2%})")
        if missing:
            print(
                f"Warning: missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}"
            )
        if unexpected:
            print(
                f"Warning: unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}"
            )
    except Exception as e:
        warnings.warn(
            f"Found weights at {loaded_weights_path} but failed to load safely: {e}. Proceeding without weights."
        )
        loaded_weights_path = None
        loaded_ok = False
else:
    warnings.warn(
        f"Weights not found at {WEIGHTS_PATH} and no alternative weights discovered under ../input. "
        "Proceeding without weights; will do a minimal training pass to avoid a 0.0 QWK submission."
    )

net = net.to(device)



## === cell 5
CACHE_DIR = "./_cache_aptos_tensors_380"
os.makedirs(CACHE_DIR, exist_ok=True)


def _cache_path(img_path: str) -> str:
    try:
        st = os.stat(img_path)
        sig = f"{st.st_size}_{int(st.st_mtime)}"
    except Exception:
        sig = "nosig"
    base = os.path.basename(img_path)
    return os.path.join(CACHE_DIR, f"{base}.{sig}.pt")


def _load_or_compute_tensor(img_path: str) -> torch.Tensor:
    cp = _cache_path(img_path)
    if os.path.exists(cp):
        return torch.load(cp, map_location="cpu")
    img = Image.open(img_path).convert("RGB")
    x = transform(img)
    tmp = cp + f".tmp_{os.getpid()}_{random.randint(0, 2**31-1)}"
    torch.save(x, tmp)
    os.replace(tmp, cp)
    return x


class _AptosInferDataset(torch.utils.data.Dataset):
    def __init__(self, id_list, img_dir):
        self.ids = list(id_list)
        self.img_dir = img_dir

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        idx = self.ids[i]
        p = os.path.join(self.img_dir, f"{idx}.png")
        x = _load_or_compute_tensor(p)
        return idx, x


def _infer_regression_outputs(id_list, img_dir, batch_size=16):
    ds = _AptosInferDataset(id_list, img_dir)

    num_workers = min(8, (os.cpu_count() or 2))
    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
    )

    outs = []
    ids_out = []
    net.eval()
    with torch.no_grad():
        for b_ids, x in dl:
            x = x.to(device, non_blocking=True)
            _, r_out, _ = net(x)
            outs.append(r_out.detach().float().cpu().view(-1))
            ids_out.extend(list(b_ids))
    outs = torch.cat(outs, dim=0).numpy() if outs else np.array([], dtype=np.float32)
    return ids_out, outs


def _apply_thresholds(reg_out_1d, thr):
    x = np.asarray(reg_out_1d, dtype=np.float64)
    t = np.asarray(thr, dtype=np.float64).reshape(1, -1)
    pred = (x.reshape(-1, 1) >= t).sum(axis=1).astype(np.int64)
    return np.clip(pred, 0, 4)


def _kappa(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


def _calibrate_thresholds(y_true, y_pred_cont, init_thr):
    thr = np.array(init_thr, dtype=np.float64)

    y_pred_cont = np.asarray(y_pred_cont, dtype=np.float64)
    y_pred_cont = np.clip(y_pred_cont, 0.0, 4.5)

    best_thr = thr.copy()
    best = _kappa(y_true, _apply_thresholds(y_pred_cont, best_thr))

    p05, p95 = np.percentile(y_pred_cont, [5, 95])
    spread = max(0.25, float(p95 - p05))
    base_steps = np.array(
        [-0.50, -0.35, -0.25, -0.15, -0.08, 0.0, 0.08, 0.15, 0.25, 0.35, 0.50],
        dtype=np.float64,
    )
    grid = base_steps * (spread / 2.0)

    for _ in range(10):
        improved = False
        for i in range(4):
            cur_best = best
            cur_thr = best_thr.copy()
            base = best_thr[i]
            for d in grid:
                cand = best_thr.copy()
                cand[i] = float(np.clip(base + d, 0.0, 4.5))
                if i > 0 and cand[i] <= cand[i - 1] + 1e-6:
                    continue
                if i < 3 and cand[i] >= cand[i + 1] - 1e-6:
                    continue
                score = _kappa(y_true, _apply_thresholds(y_pred_cont, cand))
                if score > cur_best:
                    cur_best = score
                    cur_thr = cand
            if cur_best > best + 1e-8:
                best = cur_best
                best_thr = cur_thr
                improved = True
        if not improved:
            break
        grid *= 0.65

    return best_thr.tolist(), best




## === cell 6
class _AptosTrainDataset(torch.utils.data.Dataset):
    def __init__(self, df, img_dir):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        idx = str(row["id_code"])
        y = int(row["diagnosis"])
        p = os.path.join(self.img_dir, f"{idx}.png")
        x = _load_or_compute_tensor(p)
        return x, torch.tensor(float(y), dtype=torch.float32)


def _set_trainable_for_minimal_regression(net):
    for p in net.parameters():
        p.requires_grad = False
    for p in net.regressor.parameters():
        p.requires_grad = True


def _train_regressor_minimal(
    net, train_df, img_dir, epochs=5, batch_size=16, lr=3e-4, num_workers=2
):
    net.train()
    ds = _AptosTrainDataset(train_df, img_dir)

    num_workers = min(8, max(0, num_workers))
    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
    )

    params = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)
    loss_fn = nn.MSELoss()

    t0 = time.time()
    for ep in range(epochs):
        running = 0.0
        n = 0
        for x, y in dl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).view(-1)

            _, r_out, _ = net(x)
            r_out = r_out.view(-1)

            loss = loss_fn(r_out, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            running += float(loss.item()) * x.size(0)
            n += x.size(0)

        print(
            f"Epoch {ep+1}/{epochs} - train MSE: {running/max(1,n):.4f} - elapsed: {time.time()-t0:.1f}s"
        )

    net.eval()


train_df = pd.read_csv(TRAIN_CSV)
train_df["id_code"] = train_df["id_code"].astype(str)
train_df["diagnosis"] = train_df["diagnosis"].astype(int)

rng = np.random.default_rng(42)
perm = rng.permutation(len(train_df))
split = int(0.8 * len(train_df))
tr_idx = perm[:split]
va_idx = perm[split:]

tr_df = train_df.iloc[tr_idx].reset_index(drop=True)
val_df = train_df.iloc[va_idx].reset_index(drop=True)

if not loaded_ok:
    _set_trainable_for_minimal_regression(net)
    _train_regressor_minimal(
        net, tr_df, TRAIN_IMG_DIR, epochs=5, batch_size=16, lr=3e-4, num_workers=2
    )

val_ids = val_df["id_code"].tolist()
val_ids_inferred, val_out = _infer_regression_outputs(
    val_ids, TRAIN_IMG_DIR, batch_size=16
)

if len(val_out) > 0 and len(val_ids_inferred) == len(val_out):
    pred_df = pd.DataFrame(
        {"id_code": val_ids_inferred, "pred": val_out.astype(np.float64)}
    )
    pred_df = pred_df.merge(val_df[["id_code", "diagnosis"]], on="id_code", how="inner")

    if len(pred_df) >= 200:
        y_true = pred_df["diagnosis"].to_numpy(np.int64)
        y_pred_cont = pred_df["pred"].to_numpy(np.float64)
        new_thr, best_k = _calibrate_thresholds(y_true, y_pred_cont, threshold)
        threshold = new_thr
        print(
            "Calibrated thresholds:",
            threshold,
            "val QWK:",
            best_k,
            "n_val_used:",
            len(pred_df),
        )
    else:
        warnings.warn(
            f"Too few validation samples inferred for stable calibration (n={len(pred_df)}). Keeping default thresholds."
        )
else:
    warnings.warn(
        f"Validation inference failed/empty (n_out={len(val_out)}, n_ids={len(val_ids_inferred)}). Keeping default thresholds."
    )



## === cell 7
test_ids_inferred, test_out = _infer_regression_outputs(
    test_ids, TEST_IMG_DIR, batch_size=16
)

if len(test_ids_inferred) != len(test_ids) or len(test_out) != len(test_ids):
    missing = set(test_ids) - set(test_ids_inferred)
    raise RuntimeError(
        f"Test inference mismatch: inferred {len(test_ids_inferred)} / {len(test_ids)} images. "
        f"Missing examples (up to 5): {list(sorted(missing))[:5]}"
    )

test_pred = _apply_thresholds(test_out.astype(np.float64), threshold).astype(int)

df = pd.DataFrame({"id_code": test_ids_inferred, "diagnosis": test_pred})
df["id_code"] = df["id_code"].astype(str)
df["diagnosis"] = df["diagnosis"].astype(int).clip(0, 4)

expected_n = len(test_ids_df)
if len(df) != expected_n:
    raise RuntimeError(
        f"Row count mismatch: got {len(df)} predictions but expected {expected_n}."
    )

df = df.set_index("id_code").loc[test_ids_df["id_code"].tolist()].reset_index()

df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", df.shape)
print(df.head())
