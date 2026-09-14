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

0.8948946735643287

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the missing weight-file crash by loading from the available `/kaggle/data` paths and adding a safe fallback that still produces a submission if weights aren’t present. I also fix the OpenCV CLAHE bug by converting the `cv2.split()` tuple into a mutable list before assignment. Finally, I make the trimming transform always return an image (it currently can return `None`), and I guard against unreadable images so the loop can complete and the submission CSV is never empty. These changes keep the same model/logic and only address runtime correctness so you can get a valid scored submission.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score strongly suggests the submission predictions are effectively random (likely because the intended pretrained weight file isn’t being found/loaded), even though the code now runs and writes a CSV. I make the smallest change that improves score toward the target by (1) ensuring we can actually locate and load the weights if they exist anywhere under `/kaggle` (without changing the model), and (2) loading robustly whether the checkpoint is a raw `state_dict` or wrapped in a dict (common in Kaggle). I also ensure the submission rows are in the exact same order as `test.csv` and are integer-clamped to `[0,4]` (no semantic change, just safety). These changes preserve your model and inference logic; they only address the likely root cause of the 0.0 score: missing/incorrect checkpoint loading.'
- What this solution (achieved 0.0) has done: 'I remove the hard assertion that aborts when the checkpoint file isn’t present, so the notebook always runs end-to-end and produces a valid `submission.csv`. To avoid the downstream `NameError`, I ensure `test_img_dir` is defined regardless of whether weights are found. To fix the `KeyError` during reordering, I build the submission DataFrame directly in `test.csv` order (rather than reindexing with potentially mismatched dtypes/duplicates), while still clamping predictions to valid integer classes `[0,4]`. These changes keep your model, transforms, and prediction fusion logic intact and only address execution/blocking issues and submission alignment.'
- What this solution (achieved 0.0637) has done: 'Your 0.0 score is consistent with the model running inference with random weights (the log already warns about missing checkpoints), so the smallest score-improving change is to ensure `tf_efficientnet_b4_ns` actually has meaningful weights even when the competition-specific `.pkl` isn’t found. I keep your architecture and fusion logic unchanged, but add a safe fallback to load the standard timm pretrained backbone weights (only for the backbone) when the custom checkpoint is missing or largely incompatible. I also tighten checkpoint loading by preferring `strict=True` first (then falling back to `strict=False`), so we don’t silently run with mostly-random parameters when a partially-matching file is found. This should move the score upward toward your target without changing evaluation semantics or the inference pipeline.'
- What this solution (achieved -0.00719) has done: 'Your current gap to the target (0.0637 → 0.8949) is very large, so the most likely minimal win is to ensure inference uses the same preprocessing the checkpoint expects and that the loaded weights actually match the model’s input head. I keep the model and fusion logic identical, but fix two score-critical issues: (1) timm EfficientNet-B4 expects ImageNet normalization unless the checkpoint was trained with different stats, so we use timm’s default `resolve_data_config` mean/std for this backbone (a common reason for near-random predictions), and (2) when loading a checkpoint, we only accept it as “loaded” if it meaningfully matches (low missing/unexpected keys); otherwise we fall back to timm pretrained and clearly report it. I also add a small, legitimate test-time augmentation (horizontal flip average) which preserves the same inference semantics (still single-model inference, same fusion) but typically boosts QWK with minimal code change. These changes should move the score upward toward the target without altering your architecture, heads, losses, or training approach.'
- What this solution (achieved 0.00691) has done: 'Your current score (-0.00719) is far below the target (0.8949), so we should only make minimal, high-impact fixes that keep your model and inference pipeline intact while addressing the most likely cause of near-random predictions: a mismatch between the checkpoint’s expected preprocessing and what you’re feeding at inference. I keep your architecture and probability-fusion logic unchanged, but (1) add a CLAHE-aware transform path (since your checkpoint name includes “CLAHE”) by applying CLAHE inside the PIL transform in a controlled way, and (2) choose preprocessing (mean/std) from the actual model you instantiate (b4 vs b5 differences) to avoid silent config mismatch. These are small, legitimate inference-time fixes that commonly move QWK from ~0 toward a reasonable range when weights are actually meaningful, without changing training, losses, or heads. The script still run end-to-end and write a valid `submission.csv`.'
- What this solution (achieved 0.00691) has done: 'Your current score (0.00691) is far below the target (0.89489), so the smallest likely high-impact move is to ensure the EfficientNet backbone is using the exact input normalization it was trained with rather than a possibly-mismatched `resolve_data_config` call using a string. I keep your model, heads, fusion logic, and TTA identical, but (1) resolve mean/std directly from the instantiated `net.backbone` (authoritative), and (2) make checkpoint loading accept common key prefixes (e.g., `backbone.`/`model.`) so a “found” checkpoint doesn’t silently fail to load most weights. These changes are narrowly targeted at the most common cause of near-random QWK in this setup: preprocessing + checkpoint key mismatch. The script still runs end-to-end and writes `submission.csv` in the correct format.'
- What this solution (achieved 0.0017) has done: 'I fix the inference crash by correcting the backbone feature dimension: with timm EfficientNet, calling the model directly returns logits (1000), so your heads (expecting 1792 features) get a shape mismatch. The minimal fix is to forward through `backbone.forward_features()` and then apply your existing GeM pool so the heads receive the intended feature vector. I also make the submission-building robust by ensuring `submission` is always a DataFrame even if an exception occurs, which fixes the `merge` TypeError and guarantees `submission.csv` is written. These changes preserve your model architecture and fusion logic; they only repair the forward path and output formatting so the notebook runs end-to-end.'
- What this solution (achieved 0.0) has done: 'Your current score (0.0017) is far below the target (0.8949), so we should make the smallest high-impact changes that keep your model and inference logic intact but fix two common “near-random score” causes: (1) accidentally loading the wrong timm backbone weights (including classifier head keys) instead of only the feature extractor, and (2) producing class predictions via plain argmax on an uncalibrated fused distribution rather than using an expected-value regression with your existing `threshold` discretization (more aligned with QWK). I therefore (a) load only the backbone feature weights via `timm.create_model(..., num_classes=0)` and filter keys to avoid polluting your pooled feature space, and (b) keep the same fused `c_prob + r_prob + o_prob` but convert it to a scalar expected value and discretize with your existing thresholds. These changes preserve architecture, transforms, and fusion semantics, while usually moving QWK materially upward versus argmax-on-randomish probabilities. The script remains end-to-end and still writes a valid `submission.csv`.'

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
from PIL import Image, ImageChops
import cv2

import timm

device = "cuda:0" if torch.cuda.is_available() else "cpu"

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)



## === cell 1
threshold = [0.75, 1.5, 2.5, 3.5]


def regress2class(out):
    prediction = torch.zeros(out.size(0))
    for i in range(4):
        prediction += (out.data >= threshold[i]).squeeze().cpu()
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
    if out.ndim == 2 and out.size(1) == 1:
        out = out.squeeze(1)
    pred_prob = torch.zeros((out.size(0), 5), device=out.device)

    for i in range(out.size(0)):
        if out[i] < 4.0:
            l1 = int(math.floor(out[i].item()))
            l2 = int(math.ceil(out[i].item()))
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
        in_features = getattr(self.backbone, "num_features", 1000)
        self.regressor = nn.Linear(in_features, 1)

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

        in_features = getattr(self.backbone, "num_features", 1000)

        self.classifier = nn.Sequential(
            nn.SiLU(),
            nn.Linear(in_features, 500),
            nn.SiLU(),
            nn.Linear(500, 5),
        )

        self.regressor = nn.Sequential(
            nn.SiLU(),
            nn.Linear(in_features, 500),
            nn.SiLU(),
            nn.Linear(500, 1),
        )

        self.ordinal = nn.Sequential(
            nn.SiLU(),
            nn.Linear(in_features, 500),
            nn.SiLU(),
            nn.Linear(500, 4),
        )

        self.final_regressor = nn.Sequential(
            nn.SiLU(),
            nn.Linear(10, 1),
        )

    def forward(self, x, final=False):
        feats = self.backbone.forward_features(x)  # (B, C, H, W)
        x = self.backbone.global_pool(feats)  # (B, C)

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


class applyCLAHE_PIL(object):
    def __init__(self, clipLimit=40.0, tileGridSize=(4, 4)):
        self.clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)

    def __call__(self, image: Image.Image) -> Image.Image:
        if image.mode != "RGB":
            image = image.convert("RGB")
        rgb = np.array(image)  # HWC, uint8
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        lab_planes = list(cv2.split(lab))
        lab_planes[0] = self.clahe.apply(lab_planes[0])
        lab = cv2.merge(lab_planes)
        rgb2 = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return Image.fromarray(rgb2)




## === cell 4
DATA_ROOT = "/kaggle/data/aptos2019-blindness-detection"
ALT_DATA_ROOT = "/kaggle/data"

test_csv_path = os.path.join(DATA_ROOT, "test.csv")
if not os.path.exists(test_csv_path):
    test_csv_path = os.path.join(ALT_DATA_ROOT, "test.csv")

test_df = pd.read_csv(test_csv_path)
test_df["id_code"] = test_df["id_code"].astype(str)
test_ids = test_df["id_code"].values

input_size = 512

net = ThreeStage_Model()

WEIGHT_BASENAME = "B4_3stage_60epoch_AdamW_CLAHE.pkl"
WEIGHT_BASENAME_ALTS = [
    WEIGHT_BASENAME,
    WEIGHT_BASENAME.replace(".pkl", ".pt"),
    WEIGHT_BASENAME.replace(".pkl", ".pth"),
    WEIGHT_BASENAME.replace(".pkl", ".bin"),
]

WEIGHT_CANDIDATES = [
    "/kaggle/input/weights/B4_3stage_60epoch_AdamW_CLAHE.pkl",
    "/kaggle/input/weights/B4_3stage_60epoch_AdamW_CLAHE.pth",
    "/kaggle/input/weights/B4_3stage_60epoch_AdamW_CLAHE.pt",
    "/kaggle/data/weights/B4_3stage_60epoch_AdamW_CLAHE.pkl",
    "/kaggle/data/B4_3stage_60epoch_AdamW_CLAHE.pkl",
    os.path.join(DATA_ROOT, "B4_3stage_60epoch_AdamW_CLAHE.pkl"),
    "/kaggle/working/B4_3stage_60epoch_AdamW_CLAHE.pkl",
    "/kaggle/working/weights/B4_3stage_60epoch_AdamW_CLAHE.pkl",
]


def _find_weight_under(base_dir: str, names):
    if not os.path.isdir(base_dir):
        return None
    for root, dirs, files in os.walk(base_dir):
        for nm in names:
            if nm in files:
                return os.path.join(root, nm)
    return None


def _extract_state_dict(ckpt_obj):
    if isinstance(ckpt_obj, dict):
        for k in ["state_dict", "model", "model_state_dict", "net", "weights"]:
            if k in ckpt_obj and isinstance(ckpt_obj[k], dict):
                return ckpt_obj[k]
    return ckpt_obj


def _strip_module_prefix_if_present(state):
    if isinstance(state, dict):
        has_module_prefix = any(k.startswith("module.") for k in state.keys())
        if has_module_prefix:
            return {k.replace("module.", "", 1): v for k, v in state.items()}
    return state


def _strip_known_prefixes(
    state: dict,
    prefixes=(
        "backbone.",
        "model.",
        "net.",
        "module.backbone.",
        "module.model.",
        "module.net.",
    ),
):
    if not isinstance(state, dict) or len(state) == 0:
        return state
    keys = list(state.keys())
    for p in prefixes:
        if all(k.startswith(p) for k in keys):
            return {k[len(p) :]: v for k, v in state.items()}
    return state


def _try_load_variants_into_net(net: nn.Module, state: dict):
    attempts = []

    attempts.append(("as_is", state))

    if isinstance(state, dict) and len(state) > 0:
        if not any(
            k.startswith(
                (
                    "classifier.",
                    "regressor.",
                    "ordinal.",
                    "final_regressor.",
                    "backbone.",
                )
            )
            for k in state.keys()
        ):
            attempts.append(
                ("prefix_backbone", {f"backbone.{k}": v for k, v in state.items()})
            )

    last_err = None
    for name, st in attempts:
        try:
            net.load_state_dict(st, strict=True)
            return True, f"Loaded weights (strict=True, variant={name})"
        except Exception as e:
            last_err = e
            try:
                incompatible = net.load_state_dict(st, strict=False)
                missing = list(getattr(incompatible, "missing_keys", []))
                unexpected = list(getattr(incompatible, "unexpected_keys", []))
                if (len(missing) + len(unexpected)) <= 50:
                    return True, (
                        f"Loaded weights (strict=False, mostly matching, variant={name}); "
                        f"missing={len(missing)}, unexpected={len(unexpected)}"
                    )
            except Exception as e2:
                last_err = e2
                continue
    return False, f"Failed to load checkpoint variants; last_error={repr(last_err)}"


weight_path = next((p for p in WEIGHT_CANDIDATES if os.path.exists(p)), None)
if weight_path is None:
    weight_path = _find_weight_under("/kaggle/input", WEIGHT_BASENAME_ALTS)
if weight_path is None:
    weight_path = _find_weight_under("/kaggle", WEIGHT_BASENAME_ALTS)

loaded_custom = False
if weight_path is not None:
    ckpt = torch.load(weight_path, map_location="cpu")
    state = _extract_state_dict(ckpt)
    state = _strip_module_prefix_if_present(state)
    state = _strip_known_prefixes(state)

    loaded_custom, msg = _try_load_variants_into_net(net, state)
    print(msg, "from:", weight_path)
else:
    print("WARNING: Custom checkpoint not found under /kaggle.")

if not loaded_custom:
    pretrained_backbone = timm.create_model(
        "tf_efficientnet_b4_ns", pretrained=True, num_classes=0
    )
    backbone_state = pretrained_backbone.state_dict()

    backbone_state = {
        k: v for k, v in backbone_state.items() if not k.startswith("classifier")
    }

    net.backbone.load_state_dict(backbone_state, strict=False)
    print(
        "Loaded timm pretrained feature backbone weights (num_classes=0) as fallback."
    )

cfg = getattr(net.backbone, "default_cfg", {}) or {}
mean = cfg.get("mean", (0.485, 0.456, 0.406))
std = cfg.get("std", (0.229, 0.224, 0.225))
interp = cfg.get("interpolation", "bilinear")
_interp_map = {
    "nearest": transforms.InterpolationMode.NEAREST,
    "bilinear": transforms.InterpolationMode.BILINEAR,
    "bicubic": transforms.InterpolationMode.BICUBIC,
    "lanczos": transforms.InterpolationMode.LANCZOS,
}
interp_mode = _interp_map.get(
    str(interp).lower(), transforms.InterpolationMode.BILINEAR
)

transform = transforms.Compose(
    [
        trim(),
        cropTo4_3(),
        transforms.Resize((input_size * 3 // 4, input_size), interpolation=interp_mode),
        applyCLAHE_PIL(clipLimit=40.0, tileGridSize=(4, 4)),
        transforms.ToTensor(),
        transforms.Normalize(mean=list(mean), std=list(std)),
    ]
)

net = net.to(device)
net.eval()

test_img_dir = os.path.join(DATA_ROOT, "test_images")
if not os.path.isdir(test_img_dir):
    test_img_dir = os.path.join(ALT_DATA_ROOT, "test_images")



## === cell 5
submission_rows = []


def _predict_probs(img_tensor_1x3xhxw: torch.Tensor) -> torch.Tensor:
    c_out, r_out, o_out = net(img_tensor_1x3xhxw)
    c_prob = F.softmax(c_out, dim=1)
    r_prob = regress2class_prob(r_out.data)
    o_prob = ordinal2class_prob(o_out.data)
    return c_prob + r_prob + o_prob


def _prob_to_class_from_expected_value(fuse_prob_1x5: torch.Tensor) -> int:
    probs = fuse_prob_1x5.squeeze(0)
    probs = probs / (probs.sum() + 1e-12)
    classes = torch.arange(5, device=probs.device, dtype=probs.dtype)
    expected = (probs * classes).sum().unsqueeze(0)  # shape (1,)
    pred = int(regress2class(expected).item())
    return int(max(0, min(4, pred)))


with torch.no_grad():
    for i, idx in enumerate(test_ids):
        image_name = os.path.join(test_img_dir, f"{idx}.png")

        try:
            img = Image.open(image_name).convert("RGB")
        except Exception:
            submission_rows.append([idx, 0])
            continue

        img_t = transform(img).unsqueeze(0).to(device)

        fuse_prob = _predict_probs(img_t)

        img_t_flip = torch.flip(img_t, dims=[3])  # horizontal flip (W dimension)
        fuse_prob_flip = _predict_probs(img_t_flip)

        fuse_prob = (fuse_prob + fuse_prob_flip) / 2.0

        pred = _prob_to_class_from_expected_value(fuse_prob)
        submission_rows.append([idx, pred])

submission = pd.DataFrame(submission_rows, columns=["id_code", "diagnosis"])
submission["id_code"] = submission["id_code"].astype(str)
submission["diagnosis"] = submission["diagnosis"].astype(int)



## === cell 6
df = test_df[["id_code"]].merge(submission, on="id_code", how="left")
df["diagnosis"] = df["diagnosis"].fillna(0).astype(int).clip(0, 4)

assert (
    len(df) == len(test_df) and len(df) > 0
), "Submission DataFrame is empty or mis-sized."
assert list(df.columns) == ["id_code", "diagnosis"], "Submission columns mismatch."

df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", df.shape)
print(df.head())
