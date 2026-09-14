# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.9212175220602284

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I make the weight loading robust to the Kaggle dataset layout by searching under `../input` for the expected `.pkl` (instead of assuming a missing `weights/` dataset), so the notebook can actually run inference. I also fix the CUDA/CPU dtype mismatch by moving the model to `device` before loading the state dict (and coercing tensors to the right device/dtype), which resolves the `FloatTensor` vs `cuda.FloatTensor` runtime error. Finally, I keep the model and transforms unchanged and ensure the code always writes a valid `submission.csv` with the required columns and row alignment.'
- What this solution (achieved 0.0) has done: 'I fix the runtime failure by removing the hard dependency on a missing external `.pkl` weight file and instead run inference with the existing model definition and preprocessing so the notebook completes and writes `submission.csv`. To keep core logic intact, the model architecture, transforms, and prediction mapping (`regress2class` with the given thresholds) are unchanged; only the weight-loading behavior becomes optional and non-fatal. I also make the input directory resolution robust to the actual Kaggle filesystem you showed (`/kaggle/input/...`), so the script finds `train.csv/test.csv` and image folders reliably. This should move the score from the current broken/invalid pipeline (0.0) to a valid nonzero submission, while staying within minimal necessary edits.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is consistent with running inference without the intended trained weights (or with weights not being found/loaded), which yield near-random predictions and a kappa close to 0. The smallest score-relevant change is to (1) make weight discovery more robust by also checking common “working/weights” locations and by allowing non-strict loading when the checkpoint contains extra/missing keys (common across training wrappers), while still preferring strict loading when it matches. Additionally, because the metric is quadratic weighted kappa on ordinal classes, a minimal post-processing improvement is to tune the 4 regression-to-class thresholds on a small validation split from `train.csv` (using the same regressor head output you already use), then apply those thresholds to test predictions; this keeps the model and inference core logic intact but aligns discretization to kappa. The rest of the pipeline (model, transforms, prediction source `r_out`) stays unchanged, and the script still always writes a valid `submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with the intended trained weights not actually being loaded, leaving the network at random initialization; the smallest score-relevant fix is to make checkpoint discovery/loading more compatible with common saved formats and key prefixes. I keep the model, transforms, and `r_out -> regress2class` mapping intact, but expand weight-file search to include any matching `.pkl/.pth/.pt` in the input tree and improve state-dict extraction (handling `model`, `net`, `ema`, and nested `state_dict` cases). I also ensure the loaded tensors are moved to the right device/dtype before `load_state_dict` to avoid silent partial loads and make the script fail loudly if no weights are found (since otherwise it predict near-random and stay near 0.0). Threshold tuning stays the same logic, but now run when weights are properly loaded, which should move score sharply upward toward your target.'
- What this solution (achieved 0.0) has done: 'The notebook currently fails because it hard-errors when the expected checkpoint file isn’t present in the Kaggle dataset, so inference never runs and no submission can be produced. I keep the exact model, transforms, and regression→class mapping intact, but change weight discovery to first look for the expected filename and, if not found, automatically fall back to any plausible checkpoint in the input tree (still loading via the same `load_state_dict`). If no usable weights exist anywhere, we proceed with random weights (so it runs end-to-end and writes a valid `submission.csv`), but we clearly warn that the score be near 0—this is the smallest change that both fixes the runtime error and preserves the intended logic when weights are available. All paths and submission formatting are kept compatible with the competition requirements.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is almost certainly coming from not actually using the intended trained weights (random-init predictions tend to yield kappa near 0). I make weight loading deterministic and stricter: first search only for the expected checkpoint name, then load it with better key-unwrapping (including common `model`/`state_dict`/`ema` nesting) and safer prefix stripping; if the expected weights cannot be found, the script now fail loudly instead of silently producing a near-random submission. I also keep your exact model/inference logic, but ensure the tuned thresholds are actually used consistently by running the same `apply_thresholds` logic in test inference (so tuning affects predictions without changing model behavior). These are minimal, score-relevant changes intended to move you off 0.0 and toward the target band.'
- What this solution (achieved 0.0) has done: 'I fix the hard failure caused by missing external weights by making the checkpoint load optional again so the notebook always runs end-to-end and writes `submission.csv`. Because your current score is 0.0 and the target is high, the only legitimate way to move toward the target would be to load the intended trained weights; since they are not present anywhere under the provided `/kaggle/input` tree, the best we can do is ensure a valid (but likely low-scoring) submission is produced rather than crashing. I keep the model, transforms, inference, and threshold-tuning logic unchanged; the only logic change is to skip threshold tuning when weights aren’t loaded (tuning on random outputs is meaningless/noisy). The submission formatting and id alignment checks remain intact.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 is consistent with the model running with random initialization because the intended checkpoint isn’t being found/loaded, which makes kappa collapse near 0. I make weight loading both more robust (better recursive state_dict extraction and prefix stripping) and stricter about not silently proceeding with random weights (raise an error if no checkpoint is found), since the only legitimate minimal change that can move you toward 0.92 is actually using trained weights. I also ensure the threshold tuning and test-time threshold application are using the exact same `threshold` values (your core logic already does this; we keep it) and add a small deterministic guard (cuDNN flags) to avoid accidental nondeterministic drift. All model architecture, transforms, inference approach, and regression→class mapping remain unchanged.'
- What this solution (achieved 0.0) has done: 'I fix the hard failure caused by requiring an external checkpoint that isn’t present in your `/kaggle/input` tree, so the notebook can run end-to-end and always write a valid `submission.csv`. Because the metric is quadratic weighted kappa and your pipeline already includes regression→class thresholding, I keep the exact model/inference logic but make threshold tuning conditional: it only run if weights were successfully loaded (tuning on random weights is unstable and can hurt). I also keep your robust input/image directory resolution unchanged and ensure inference is wrapped in `torch.inference_mode()` to avoid any accidental grad state issues. These changes are minimal and directly address the runtime error + avoid unnecessary score-damaging behavior when weights are missing.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with submitting predictions from a randomly initialized network because the intended trained checkpoint is not being found/loaded (or is being partially loaded with key mismatches). The smallest score-relevant change is to (1) search more broadly for any plausible EfficientNet-B4 3-stage checkpoint file in the provided `/kaggle/input` tree and (2) make state-dict key normalization smarter (only strip `backbone.` if it actually helps) while still preferring strict loading. I’m also making the notebook fail loudly if no weights are loaded, because proceeding would almost certainly keep kappa near 0 and cannot move you toward 0.92 legitimately. Everything else (model, transforms, regression head usage, threshold tuning logic, submission formatting) stays the same.'
- What this solution (achieved 0.0) has done: 'I remove the hard failure when the trained checkpoint can’t be found so the notebook always runs end-to-end and writes a valid `submission.csv` (your current crash is why you can’t iterate toward the target). Because tuning thresholds on random/untrained outputs is meaningless and can further destabilize predictions, I make threshold tuning conditional on successful weight loading while keeping the exact same tuning logic when weights do load. I also keep all model definitions, transforms, and inference logic unchanged; the only functional change is to allow a fallback “run without weights” path that still produces a valid submission. This move you from “no submission / 0.0” to a valid submission, and if you later attach the intended checkpoint file, the same code automatically use it and apply threshold tuning for kappa.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is overwhelmingly likely to be caused by running inference with random weights (because no checkpoint is actually present/loaded), which cannot realistically move toward the 0.92 target. The most score-relevant minimal change is therefore to (1) fail loudly if weights are not found/loaded (so you don’t unknowingly submit near-random predictions again), and (2) broaden checkpoint discovery slightly to include more common filenames while keeping the same model/inference logic. I also make threshold tuning use a slightly larger validation split to reduce noise (same tuning logic, just more stable), and ensure regression outputs are flattened robustly to avoid shape edge-cases. The rest (model architecture, transforms, inference loop, and submission formatting) stays the same.'

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

from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedShuffleSplit
import timm

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_grad_enabled(False)

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



## === cell 1
threshold = [0.75, 1.5, 2.5, 3.5]


def regress2class(out):
    prediction = torch.zeros(out.size(0), dtype=torch.long)
    for i in range(4):
        prediction += (out >= threshold[i]).squeeze().cpu().long()
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
def find_weight_file(filename: str, roots):
    candidate_paths = [
        "../input/weights/" + filename,
        "../input/aptos2019-blindness-detection/weights/" + filename,
        "../kaggle/input/weights/" + filename,
        "../kaggle/input/aptos2019-blindness-detection/weights/" + filename,
        "/kaggle/input/weights/" + filename,
        "/kaggle/input/aptos2019-blindness-detection/weights/" + filename,
        "/kaggle/working/weights/" + filename,
        "/kaggle/working/" + filename,
        "../input/" + filename,
        "../input/aptos2019-blindness-detection/" + filename,
        "../kaggle/input/" + filename,
        "../kaggle/input/aptos2019-blindness-detection/" + filename,
        "/kaggle/input/" + filename,
        "/kaggle/input/aptos2019-blindness-detection/" + filename,
    ]
    for p in candidate_paths:
        if os.path.exists(p):
            return p

    for root in roots:
        if not os.path.exists(root):
            continue
        for dirpath, _, filenames in os.walk(root):
            if filename in filenames:
                return os.path.join(dirpath, filename)
    return None


def find_weight_file_flexible(base_filename: str, roots):
    exts = ["", ".pkl", ".pth", ".pt", ".bin"]
    for ext in exts:
        name = base_filename if base_filename.endswith(ext) else (base_filename + ext)
        p = find_weight_file(name, roots)
        if p is not None:
            return p

    stem = os.path.splitext(os.path.basename(base_filename))[0].lower()
    best = None
    best_score = -1
    for root in roots:
        if not os.path.exists(root):
            continue
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                low = fn.lower()
                if not low.endswith((".pkl", ".pth", ".pt", ".bin")):
                    continue
                score = 0
                tokens = [
                    "b4",
                    "b5",
                    "3stage",
                    "stage",
                    "aptos",
                    "blind",
                    "retina",
                    "efficientnet",
                    "3_stage",
                    "three",
                ]
                for t in tokens:
                    if t in low:
                        score += 1
                if stem and stem in low:
                    score += 3
                cand = os.path.join(dirpath, fn)
                if score > best_score or (
                    score == best_score and best is not None and len(cand) < len(best)
                ):
                    best = cand
                    best_score = score
    return best


def _strip_known_prefixes_from_state_dict_keys(state, prefixes):
    keys = list(state.keys())
    for pref in prefixes:
        if keys and all(k.startswith(pref) for k in keys):
            state = {k.replace(pref, "", 1): v for k, v in state.items()}
            keys = list(state.keys())
    return state


def extract_state_dict(ckpt_obj):
    state = ckpt_obj
    if isinstance(state, dict):
        for _ in range(6):
            if not isinstance(state, dict):
                break

            for key in [
                "state_dict",
                "model_state_dict",
                "model",
                "net",
                "ema",
                "student",
                "module",
                "weights",
                "params",
            ]:
                if (
                    key in state
                    and isinstance(state[key], dict)
                    and len(state[key]) > 0
                ):
                    state = state[key]
                    break
            else:
                for key in ["checkpoint", "ckpt"]:
                    if (
                        key in state
                        and isinstance(state[key], dict)
                        and len(state[key]) > 0
                    ):
                        state = state[key]
                        break
                else:
                    break

    if not isinstance(state, dict):
        raise TypeError(f"Unsupported checkpoint type after extraction: {type(state)}")

    if (
        "state_dict" in state
        and isinstance(state["state_dict"], dict)
        and len(state) < 10
    ):
        state = state["state_dict"]

    return state


def resolve_input_dir():
    candidates = [
        "/kaggle/input/aptos2019-blindness-detection",
        "../input/aptos2019-blindness-detection",
        "/kaggle/data/aptos2019-blindness-detection",
        "../kaggle/data/aptos2019-blindness-detection",
    ]
    for c in candidates:
        if os.path.exists(os.path.join(c, "test.csv")):
            return c
    for root in ["/kaggle/input", "../input", "/kaggle/data", "../kaggle/data"]:
        if not os.path.exists(root):
            continue
        for dirpath, _, filenames in os.walk(root):
            if "test.csv" in filenames and "train.csv" in filenames:
                return dirpath
    raise FileNotFoundError(
        "Could not locate competition input directory containing train.csv/test.csv."
    )


def resolve_image_dir(input_dir: str, split: str):
    base = os.path.join(input_dir, f"{split}_images")
    if os.path.exists(base):
        return base
    alt = os.path.join(input_dir, "aptos2019-blindness-detection", f"{split}_images")
    if os.path.exists(alt):
        return alt
    for root in [input_dir, os.path.join(input_dir, "aptos2019-blindness-detection")]:
        if not os.path.exists(root):
            continue
        cand = os.path.join(root, f"{split}_images")
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError(f"Could not find {split}_images under {input_dir}.")


INPUT_DIR = resolve_input_dir()
test_df = pd.read_csv(os.path.join(INPUT_DIR, "test.csv"))
test_ids = test_df["id_code"].values

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

net = ThreeStage_Model().to(device)
net.eval()

weight_candidates = [
    "B4_3stage_20epoch_finetune.pkl",
    "B4_3stage_20epoch_finetune.pth",
    "B4_3stage_20epoch_finetune.pt",
    "b4_3stage_20epoch_finetune.pkl",
    "b4_3stage_20epoch_finetune.pth",
    "b4_3stage_20epoch_finetune.pt",
]

search_roots = [
    "/kaggle/input",
    "../input",
    "/kaggle/data",
    "../kaggle/data",
    "/kaggle/working",
]

weight_path = None
picked_name = None
for wname in weight_candidates:
    wp = find_weight_file_flexible(wname, roots=search_roots)
    if wp is not None:
        weight_path = wp
        picked_name = wname
        break
if weight_path is None:
    weight_path = find_weight_file_flexible(weight_candidates[0], roots=search_roots)
    picked_name = weight_candidates[0]

loaded = False
load_mode = "none"
load_error = None

if weight_path is not None:
    try:
        ckpt = torch.load(weight_path, map_location="cpu")
        state = extract_state_dict(ckpt)

        model_keys = set(net.state_dict().keys())

        def score_match(state_dict):
            keys = set(state_dict.keys())
            return len(keys & model_keys)

        candidates = []
        candidates.append(("as_is", state))
        candidates.append(
            (
                "strip_module_model_net_student_encoder",
                _strip_known_prefixes_from_state_dict_keys(
                    state,
                    prefixes=["module.", "model.", "net.", "student.", "encoder."],
                ),
            )
        )
        candidates.append(
            (
                "strip_plus_backbone",
                _strip_known_prefixes_from_state_dict_keys(
                    state,
                    prefixes=[
                        "module.",
                        "model.",
                        "net.",
                        "student.",
                        "encoder.",
                        "backbone.",
                    ],
                ),
            )
        )

        best_name, best_state, best_match = None, None, -1
        for name, st in candidates:
            m = score_match(st)
            if m > best_match:
                best_name, best_state, best_match = name, st, m

        state = best_state

        model_param = next(net.parameters())
        target_device = model_param.device
        target_dtype = model_param.dtype
        for k, v in list(state.items()):
            if torch.is_tensor(v):
                state[k] = v.to(device=target_device, dtype=target_dtype)

        try:
            net.load_state_dict(state, strict=True)
            loaded = True
            load_mode = f"strict/{best_name}"
        except RuntimeError as e:
            net.load_state_dict(state, strict=False)
            loaded = True
            load_mode = f"non_strict/{best_name}"
            load_error = str(e)[:1200]
    except Exception as e:
        loaded = False
        load_mode = "failed"
        load_error = repr(e)

print("Using device:", device)
print("INPUT_DIR:", INPUT_DIR)
print(
    "Loaded weights:",
    loaded,
    "| mode:",
    load_mode,
    "| picked_name:",
    picked_name,
    "| path:",
    weight_path,
)
if load_error is not None:
    print("Weight load note/error (truncated):\n", load_error)
print("Num test images:", len(test_ids))

if not loaded:
    raise FileNotFoundError(
        "No trained checkpoint could be loaded. "
        "To move toward the target score, attach the intended weight file to the notebook/dataset "
        "or place it under /kaggle/input or /kaggle/working, then rerun."
    )




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/2629820611.py in <cell line: 0>()
    311 # Change (score-relevant): fail loudly if weights are not loaded; otherwise submissions stay near 0.0.
    312 if not loaded:
--> 313     raise FileNotFoundError(
    314         "No trained checkpoint could be loaded. "
    315         "To move toward the target score, attach the intended weight file to the notebook/dataset "

FileNotFoundError: No trained checkpoint could be loaded. To move toward the target score, attach the intended weight file to the notebook/dataset or place it under /kaggle/input or /kaggle/working, then rerun.

## === cell 5
def infer_regression_outputs(ids, img_dir, batch_size=8):
    outs = []
    with torch.inference_mode():
        for start in range(0, len(ids), batch_size):
            batch_ids = ids[start : start + batch_size]
            imgs = []
            for idx in batch_ids:
                image_name = os.path.join(img_dir, f"{idx}.png")
                img = Image.open(image_name).convert("RGB")
                imgs.append(transform(img))
            x = torch.stack(imgs, dim=0).to(device, non_blocking=True)
            _, r_out, _ = net(x)
            outs.append(r_out.detach().float().view(-1).cpu().numpy())
    return np.concatenate(outs, axis=0)


def apply_thresholds(reg_out_np, thr):
    reg_out_t = torch.from_numpy(reg_out_np.astype(np.float32)).view(-1)
    pred = torch.zeros(reg_out_t.size(0), dtype=torch.long)
    for i in range(4):
        pred += (reg_out_t >= thr[i]).long()
    return pred.numpy().astype(int)


def tune_thresholds_for_kappa(y_true, y_reg, init_thr, n_iters=12, step=0.05):
    thr = np.array(init_thr, dtype=np.float32).copy()
    best_kappa = cohen_kappa_score(
        y_true, apply_thresholds(y_reg, thr), weights="quadratic"
    )

    for _ in range(n_iters):
        improved = False
        for j in range(4):
            candidates = []
            for delta in [-2, -1, 0, 1, 2]:
                cand = thr.copy()
                cand[j] = float(cand[j] + delta * step)
                cand = np.clip(cand, 0.0, 4.5)
                cand.sort()
                k = cohen_kappa_score(
                    y_true, apply_thresholds(y_reg, cand), weights="quadratic"
                )
                candidates.append((k, cand))
            candidates.sort(key=lambda x: x[0], reverse=True)
            if candidates[0][0] > best_kappa + 1e-8:
                best_kappa = candidates[0][0]
                thr = candidates[0][1]
                improved = True
        if not improved:
            break
    return thr.tolist(), float(best_kappa)


train_csv_path = os.path.join(INPUT_DIR, "train.csv")
train_df = pd.read_csv(train_csv_path)
train_img_dir = resolve_image_dir(INPUT_DIR, "train")

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
tr_idx, va_idx = next(
    sss.split(train_df["id_code"].values, train_df["diagnosis"].values)
)
val_df = train_df.iloc[va_idx].reset_index(drop=True)

val_ids = val_df["id_code"].values
val_y = val_df["diagnosis"].values.astype(int)

val_reg = infer_regression_outputs(val_ids, train_img_dir, batch_size=8)
tuned_thr, tuned_kappa = tune_thresholds_for_kappa(
    val_y, val_reg, threshold, n_iters=12, step=0.05
)
print("Threshold tuning kappa (val):", tuned_kappa)
print("Old thresholds:", threshold)
print("New thresholds:", tuned_thr)
threshold = tuned_thr



## === cell 6
submission = []
test_img_dir = resolve_image_dir(INPUT_DIR, "test")

with torch.inference_mode():
    for i in range(0, len(test_ids), 8):
        if i % 50 == 0:
            print(i, "/", len(test_ids))

        batch_ids = test_ids[i : i + 8]
        imgs = []
        for idx in batch_ids:
            image_name = os.path.join(test_img_dir, f"{idx}.png")
            img = Image.open(image_name).convert("RGB")
            imgs.append(transform(img))
        x = torch.stack(imgs, dim=0).to(device, non_blocking=True)

        _, r_out, _ = net(x)
        r_np = r_out.detach().float().view(-1).cpu().numpy()
        preds = apply_thresholds(r_np, threshold)
        for idx, p in zip(batch_ids, preds):
            submission.append([idx, int(p)])

submission = np.array(submission, dtype=object)



## === cell 7
df = pd.DataFrame(submission, columns=["id_code", "diagnosis"])

if df.empty:
    raise RuntimeError("Submission DataFrame is empty; inference did not run.")
df["diagnosis"] = df["diagnosis"].astype(int).clip(0, 4)

df = test_df.merge(df, on="id_code", how="left", validate="one_to_one")
if df["diagnosis"].isna().any():
    missing = df.loc[df["diagnosis"].isna(), "id_code"].head(5).tolist()
    raise RuntimeError(f"Missing predictions for some ids (showing up to 5): {missing}")

df = df[["id_code", "diagnosis"]]
df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", df.shape)
print(df.head())
