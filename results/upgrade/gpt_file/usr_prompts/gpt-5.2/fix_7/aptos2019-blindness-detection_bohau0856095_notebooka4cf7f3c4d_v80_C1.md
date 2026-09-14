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

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I (1) remove the failing custom `pip install` and instead use the already-installed `timm`, (2) make the code robust to CPU-only Kaggle sessions by selecting `cuda` only when available and by removing hard `.cuda()` tensor allocations, and (3) fix the missing weights issue by safely falling back to an untrained model (still producing a valid, non-empty submission) while emitting a clear warning. I also fix a couple of transform bugs (`trim()` returning `None`, and `is` vs `==` for string comparison) that can cause runtime errors. Finally, I keep the model/thresholding logic intact and ensure `submission.csv` is always written with the correct columns and row count.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is because the model is running with random weights (the specified `WEIGHTS_PATH` doesn’t exist in this environment), so the smallest meaningful improvement is to (a) correctly locate a weights file if one is available under the provided dataset tree and load it, and (b) if no weights exist, keep your exact inference logic but calibrate the 4 thresholds on a train-validation split to maximize quadratic weighted kappa and then apply those thresholds at test-time. This preserves your model architecture and inference semantics (still using the regressor head + `regress2class`) while making the predictions far less arbitrary, moving score upward toward your target. I also ensure `test_ids` extraction is robust (uses `id_code` explicitly) and that the submission ordering matches `test.csv`. All changes are directly tied to making predictions consistent and improving kappa without changing the core modeling approach.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is coming from running the ThreeStage model with random weights (no valid checkpoint was loaded), so the smallest legitimate move toward your target is to (1) reliably find and load an actual compatible checkpoint if it exists anywhere under `../input`, and (2) if no checkpoint exists, at least make inference+thresholding consistent and kappa-aligned by calibrating the regression thresholds on a held-out validation split and then using those thresholds for test predictions. I keep your model, transforms, and `regress2class` semantics intact, but fix a key bug where calibrated thresholds were not actually used by `regress2class` (it referenced the global `threshold` list, but you were passing raw model outputs in a slightly mismatched shape in some places). I also switch test inference to the same batched path used for calibration to reduce per-image variance/IO overhead without changing the model logic, and ensure the submission is still written exactly in the required format and order.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with producing nearly-constant or effectively random predictions (often caused by missing/incorrect weights), so the smallest change that legitimately moves toward your target is to (1) make checkpoint discovery/load more reliable by preferring filenames that match your exact model (B4 + 3stage + epoch/finetune keywords) and by properly handling common checkpoint wrappers, and (2) calibrate the regression thresholds on a validation split using only the images that were successfully inferred (fixing the current mismatch logic, which often prevents calibration from being applied at all). These keep your model architecture and inference semantics intact (same ThreeStage model, still using regressor output + thresholding), but make the discrete predictions align better with QWK. I also make the validation inference join by `id_code` so kappa is computed against the correct labels even if any images are skipped (which avoids silently wrong calibration). The submission format, ordering, and paths remain unchanged, and it still writes `submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is consistent with either (a) weights not loading so predictions are effectively random/near-constant, or (b) a submission alignment/type issue (e.g., `diagnosis` not being clean ints 0–4 in the exact `test.csv` order). I make the smallest changes that directly improve QWK without changing your model/inference semantics: (1) strengthen checkpoint loading to correctly handle common wrapper formats and reject obviously-incompatible state dicts, and (2) calibrate the 4 regression thresholds using a slightly more robust coordinate-descent search that scales steps to the model’s output range, then apply those thresholds for test predictions. I also make test-time ordering airtight by building the submission directly from `test.csv` order (so even if any id is skipped, it fails early with a clear error as before). These changes keep your ThreeStage model and regressor+thresholding approach intact, but should move your score substantially upward toward the target when any reasonable weights are found/loaded.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is overwhelmingly likely due to missing/incorrect weights leading to near-random predictions, so the smallest meaningful move toward your target is to (1) make checkpoint discovery/loading more robust and actually compatible with your `ThreeStage_Model` by handling nested/wrapped state_dicts and common key prefixes, and (2) ensure thresholds are always calibrated and applied on a correctly-aligned validation set (by joining on `id_code`) so the discrete predictions are optimized for QWK. I keep your model, transforms, inference path, and regressor-to-class semantics intact; the only behavioral changes are safer weight loading and more reliable threshold calibration/application. I also ensure the submission is built strictly in `test.csv` order with integer labels 0–4, so there’s no silent format/alignment penalty.'

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
from PIL import Image, ImageChops

from sklearn.metrics import cohen_kappa_score
import timm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)



## === cell 1
threshold = [0.75, 1.5, 2.5, 3.5]


def regress2class(out):
    prediction = torch.zeros(out.size(0), device="cpu")
    for i in range(4):
        prediction += (out.data >= threshold[i]).squeeze().detach().cpu()
    return prediction


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
            l1 = int(math.floor(float(out[i].item())))
            l2 = int(math.ceil(float(out[i].item())))
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
        for pref in ("module.", "model.", "net.", "backbone."):
            pass
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

if os.path.exists(WEIGHTS_PATH):
    loaded_weights_path = WEIGHTS_PATH
else:
    loaded_weights_path = _find_weights_fallback()

if loaded_weights_path is not None:
    try:
        overlap, missing, unexpected = _try_load_checkpoint(loaded_weights_path)
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
else:
    warnings.warn(
        f"Weights not found at {WEIGHTS_PATH} and no alternative weights discovered under ../input. "
        "Proceeding with randomly-initialized model; thresholds will be calibrated but score will likely remain far from target."
    )

net = net.to(device)
net.eval()




## === cell 5
def _infer_regression_outputs(id_list, img_dir, batch_size=8):
    outs = []
    ids = []
    n = len(id_list)
    with torch.no_grad():
        for start in range(0, n, batch_size):
            batch_ids = id_list[start : start + batch_size]
            imgs = []
            valid_ids = []
            for idx in batch_ids:
                p = os.path.join(img_dir, f"{idx}.png")
                if not os.path.exists(p):
                    continue
                img = Image.open(p).convert("RGB")
                img = transform(img)
                imgs.append(img)
                valid_ids.append(idx)
            if not imgs:
                continue
            x = torch.stack(imgs, dim=0).to(device)
            _, r_out, _ = net(x)
            outs.append(r_out.detach().float().cpu().view(-1))
            ids.extend(valid_ids)
    if outs:
        outs = torch.cat(outs, dim=0).numpy()
    else:
        outs = np.array([], dtype=np.float32)
    return ids, outs


def _apply_thresholds(reg_out_1d, thr):
    pred = np.zeros_like(reg_out_1d, dtype=np.int64)
    for t in thr:
        pred += (reg_out_1d >= t).astype(np.int64)
    return np.clip(pred, 0, 4)


def _kappa(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


def _calibrate_thresholds(y_true, y_pred_cont, init_thr):
    thr = np.array(init_thr, dtype=np.float64)

    y_pred_cont = np.asarray(y_pred_cont, dtype=np.float64)
    y_pred_cont = np.clip(y_pred_cont, 0.0, 4.5)

    best_thr = thr.copy()
    best = _kappa(y_true, _apply_thresholds(y_pred_cont, best_thr))

    p10, p90 = np.percentile(y_pred_cont, [10, 90])
    spread = max(0.2, float(p90 - p10))
    base_steps = np.array(
        [-0.30, -0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20, 0.30], dtype=np.float64
    )
    grid = base_steps * (spread / 2.0)

    for _ in range(8):
        improved = False
        for i in range(4):
            cur_best = best
            cur_thr = best_thr.copy()
            base = best_thr[i]
            for d in grid:
                cand = best_thr.copy()
                cand[i] = base + d
                cand[i] = float(np.clip(cand[i], 0.0, 4.5))
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
        grid *= 0.6

    return best_thr.tolist(), best


train_df = pd.read_csv(TRAIN_CSV)
train_df["id_code"] = train_df["id_code"].astype(str)
train_df["diagnosis"] = train_df["diagnosis"].astype(int)

rng = np.random.default_rng(42)
perm = rng.permutation(len(train_df))
split = int(0.8 * len(train_df))
va_idx = perm[split:]
val_df = train_df.iloc[va_idx].reset_index(drop=True)

val_ids = val_df["id_code"].tolist()
val_ids_inferred, val_out = _infer_regression_outputs(
    val_ids, TRAIN_IMG_DIR, batch_size=8
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



## === cell 6
test_ids_inferred, test_out = _infer_regression_outputs(
    test_ids, TEST_IMG_DIR, batch_size=8
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
