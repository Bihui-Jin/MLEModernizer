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

0.9237334929400204

# 6. Current score

-0.00265

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -0.03395) has done: 'I fix the missing weight file crash by making the script robust to Kaggle path differences and by cleanly falling back to a built-in pretrained EfficientNet if the custom weights are unavailable (so a non-empty submission is always produced). I also fix the CUDA hardcoding by selecting `cuda` only when available, which resolves the “no NVIDIA driver” runtime error. Finally, I prevent the custom `trim()` transform from returning `None` (which can break transforms) by returning the original image when no bbox is found, and I speed up inference with a DataLoader (same model logic, just batched execution) to stay within the time limit while producing `submission.csv` in the required format.'
- What this solution (achieved 0.02381) has done: 'Your current score is far below the target, so the most direct way to move toward 0.9237 is to use the model’s intended “final” fusion head at inference instead of only the regressor head, and to convert its continuous output into 0–4 labels using the same thresholds logic already present. This preserves the architecture and weights while aligning prediction with the model design, which should substantially increase kappa versus the current near-random result. I also ensure predictions are clipped to [0,4] and keep ordering aligned with `test.csv` to avoid any accidental misalignment. No training is introduced; this is an inference-only change that should remain within the time limit and still write a valid `submission.csv`.'
- What this solution (achieved 0.11769) has done: 'Your current score is far below the target, so the smallest high-impact change is to align inference post-processing with quadratic weighted kappa by tuning the 4 regression-to-class thresholds on a holdout split of the training set, then using those learned thresholds for test predictions. This keeps your exact model architecture and inference path (`final=True`) intact, and only adjusts the discretization step that strongly controls kappa. To stay within time, I reuse the same transforms and run a single-pass optimization (coordinate descent) over thresholds using the already-installed `cohen_kappa_score`. The rest of the pipeline (paths, loading, batching, and submission formatting) stays the same.'
- What this solution (achieved 0.11769) has done: 'Your current score (0.11769) is far below the target (0.9237), so we should push performance upward with the smallest change that preserves your exact model and inference path. The most likely issue is a misalignment between `val_y` and `val_pred_cont` because `predict_regression` returns predictions in DataLoader order, while `val_y` comes from `train_df.iloc[val_idx]`; if DataLoader reorders IDs for any reason or duplicates/missing files occur, threshold tuning becomes essentially random and hurts QWK badly. I make the validation prediction return a `DataFrame` keyed by `id_code` and then explicitly align `val_pred_cont` to `val_ids` before tuning, without changing the model, transforms, or training approach. I also make `regress2class` device-safe (not required for submission, but it prevents subtle dtype/device issues) and keep everything else identical.'
- What this solution (achieved 0.02171) has done: 'Your current score (0.11769) is far below the target (0.9237), so we need a safe, minimal change that legitimately improves kappa without altering the model or training logic. The biggest likely drag is that the model expects EfficientNet-B4 inputs normalized with ImageNet stats (since the fallback uses ImageNet pretrained weights, and the provided weights were almost certainly trained with the same convention), but your pipeline uses custom mean/std; this mismatch can severely degrade predictions. I switch normalization to ImageNet mean/std while keeping all transforms, model forward path (`final=True`), and threshold tuning exactly the same. I also add a small safety clamp to ensure monotonic thresholds during tuning (already mostly done) and keep submission alignment unchanged.'
- What this solution (achieved 0.0) has done: 'Your current score is far below the target, so the smallest likely high-impact fix is to match the input preprocessing to the `tf_efficientnet_b4_ns/b5_ns` family’s expected normalization (these models typically perform best with Inception-style mean/std, not ImageNet). I keep your exact model, `final=True` inference path, threshold tuning logic, and submission formatting unchanged, and only adjust the normalization constants to the EfficientNet/Inception convention. This should move predictions away from near-random behavior without changing architecture or training. I also keep the tuned thresholds monotonic as you already do, and leave all paths and batching intact.'
- What this solution (achieved 0.02171) has done: 'Your current 0.0 score strongly suggests the submission is being evaluated as essentially constant/wrong due to a preprocessing mismatch, not a model/thresholding bug (your ordering and CSV schema look correct). The smallest high-impact change that preserves your exact model and inference flow is to use the normalization that `tf_efficientnet_b4_ns/b5_ns` expects in `timm` (model-specific mean/std), instead of hardcoding `[0.5,0.5,0.5]`. I implement this by constructing the transform from `net.backbone.default_cfg` (or `net.backbone.pretrained_cfg` fallback), keeping all resizing/cropping and threshold-tuning logic identical. This should move predictions from near-random toward the model’s intended input distribution, improving QWK toward your target without altering architecture or training.'
- What this solution (achieved 0.02171) has done: 'Your current score is far below the target, so we should push performance upward with minimal, safe changes that keep your exact model and inference path intact. The most likely remaining issue is input normalization: for `timm` EfficientNet models the correct preprocessing stats are stored in `model.pretrained_cfg` (often `default_cfg` can be empty/mismatched), and a wrong mean/std can make predictions near-random. I switch to extracting mean/std from `net.backbone.pretrained_cfg` first (then fall back), and I also ensure we use the matching `input_size` from that cfg when available (still just resizing, same transform steps). Everything else (architecture, `final=True` inference, threshold tuning, submission alignment/format) stays the same.'
- What this solution (achieved 0.0) has done: 'Your score is far below the target, so we should push QWK up with the smallest change that keeps your model/inference logic intact. The biggest remaining likely issue is a mismatch between how the backbone was trained and the preprocessing you feed it: `tf_efficientnet_b4_ns/b5_ns` in `timm` expects a specific resize+crop policy and normalization, and relying on `pretrained_cfg` from a non-pretrained backbone can silently give fallback `(0.5,0.5,0.5)` stats. I build the eval transform directly from `timm.data.resolve_model_data_config` + `create_transform` (center-crop, correct interpolation, correct mean/std) while keeping your existing `trim()` and `cropTo4_3()` steps and keeping `final=True` inference + threshold tuning unchanged. This is a preprocessing-only correction, so it preserves architecture/training semantics and should move predictions away from near-random toward a much higher kappa.'
- What this solution (achieved 0.0) has done: 'Your current score (0.0) suggests the submission is effectively garbage for QWK, and the most likely cause in this pipeline is that the test images are not being found/loaded consistently (silent mismatch of `*_img_dir` due to Kaggle path/layout differences), leading to failed/empty/degenerate predictions that still write a CSV. I make the image directory resolution robust by searching the known folder layouts under your detected `DATA_ROOT` (including the common nested `aptos2019-blindness-detection/aptos2019-blindness-detection/...` structure) and asserting that a sample image exists before inference. This is a minimal change that preserves your exact model, transforms, threshold tuning, and inference logic, but prevents “valid CSV with wrong/constant predictions” caused by reading the wrong directory. Everything else (final head inference, threshold tuning, submission formatting/alignment) remains identical.'
- What this solution (achieved -0.00265) has done: 'Your current 0.0 QWK strongly suggests the model is producing essentially uninformative/constant predictions, and in this pipeline the most likely cause is that the checkpoint (when present) was trained with a *different* input preprocessing than the `timm` default eval transform you’re currently using. I keep your model, `final=True` inference, and threshold-tuning logic identical, but change only the image preprocessing to the standard APTOS baseline (Ben Graham-style circular crop + resize + ImageNet normalization), which is a common requirement for these DR models and should move QWK upward toward your target. I also make the weight search slightly more robust by including `/kaggle/input/` root scanning for the exact filename (no logic change if your current path works). The rest of your pipeline (paths, batching, alignment, CSV writing) stays the same.'

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
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.transforms import functional as FT
from PIL import Image, ImageChops

from sklearn.metrics import cohen_kappa_score
import timm

device = "cuda" if torch.cuda.is_available() else "cpu"

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)



## === cell 1
threshold = [0.75, 1.5, 2.5, 3.5]


def regress2class(out):
    out_cpu = out.detach().view(-1).cpu()
    prediction = torch.zeros(out_cpu.size(0), dtype=torch.int64)
    for i in range(4):
        prediction += (out_cpu >= threshold[i]).to(torch.int64)
    return prediction


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


def ben_graham_preprocess(pil_img, out_size=380, tol=7):
    img = np.array(pil_img)
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    if img.shape[2] == 4:
        img = img[:, :, :3]

    gray = img.mean(axis=2)
    mask = gray > tol

    if mask.any():
        coords = np.argwhere(mask)
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1
        img = img[y0:y1, x0:x1]

    h, w = img.shape[:2]
    r = min(h, w) // 2
    cy, cx = h // 2, w // 2

    y0, y1 = cy - r, cy + r
    x0, x1 = cx - r, cx + r
    img = img[y0:y1, x0:x1]

    hh, ww = img.shape[:2]
    Y, X = np.ogrid[:hh, :ww]
    dist = (Y - hh / 2) ** 2 + (X - ww / 2) ** 2
    circle = dist <= (min(hh, ww) / 2) ** 2

    out = np.zeros_like(img)
    out[circle] = img[circle]

    pil = Image.fromarray(out)
    pil = pil.resize((out_size, out_size), resample=Image.BILINEAR)
    return pil




## === cell 4
BASE_INPUT_CANDIDATES = [
    "/kaggle/input/aptos2019-blindness-detection",
    "../input/aptos2019-blindness-detection",
]


def first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


DATA_ROOT = first_existing(BASE_INPUT_CANDIDATES)
if DATA_ROOT is None:
    raise FileNotFoundError(
        "Could not find aptos2019-blindness-detection dataset folder in expected locations."
    )

test_csv_path = os.path.join(DATA_ROOT, "test.csv")
train_csv_path = os.path.join(DATA_ROOT, "train.csv")

test_df = pd.read_csv(test_csv_path)
test_ids = test_df["id_code"].astype(str).values

train_df = pd.read_csv(train_csv_path)
train_df["id_code"] = train_df["id_code"].astype(str)
train_df["diagnosis"] = train_df["diagnosis"].astype(int)


def resolve_image_dir(root, split_name):
    candidates = [
        os.path.join(root, f"{split_name}_images"),
        os.path.join(root, "aptos2019-blindness-detection", f"{split_name}_images"),
        os.path.join(
            root,
            "aptos2019-blindness-detection",
            "aptos2019-blindness-detection",
            f"{split_name}_images",
        ),
        os.path.join(os.path.dirname(root), f"{split_name}_images"),
        os.path.join(
            os.path.dirname(root),
            "aptos2019-blindness-detection",
            f"{split_name}_images",
        ),
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    return None


test_img_dir = resolve_image_dir(DATA_ROOT, "test")
train_img_dir = resolve_image_dir(DATA_ROOT, "train")
if test_img_dir is None or train_img_dir is None:
    raise FileNotFoundError(
        f"Could not resolve image directories. Got train_img_dir={train_img_dir}, test_img_dir={test_img_dir}, DATA_ROOT={DATA_ROOT}"
    )

_sample_train = os.path.join(train_img_dir, f"{train_df['id_code'].iloc[0]}.png")
_sample_test = os.path.join(test_img_dir, f"{test_df['id_code'].iloc[0]}.png")
if not os.path.exists(_sample_train):
    raise FileNotFoundError(f"Train image not found at expected path: {_sample_train}")
if not os.path.exists(_sample_test):
    raise FileNotFoundError(f"Test image not found at expected path: {_sample_test}")

input_size = 380

WEIGHT_NAME = "B4_3stage_3epoch_finetune3.pkl"
WEIGHT_CANDIDATES = [
    "/kaggle/input/weights/" + WEIGHT_NAME,
    "../input/weights/" + WEIGHT_NAME,
]

weight_path = first_existing(WEIGHT_CANDIDATES)
if weight_path is None and os.path.isdir("/kaggle/input"):
    for root, _, files in os.walk("/kaggle/input"):
        if WEIGHT_NAME in files:
            weight_path = os.path.join(root, WEIGHT_NAME)
            break

net = ThreeStage_Model()

if weight_path is not None:
    state = torch.load(weight_path, map_location="cpu")
    net.load_state_dict(state, strict=True)
else:
    pretrained_backbone = timm.create_model("tf_efficientnet_b4_ns", pretrained=True)
    pretrained_backbone.global_pool = GeM(flatten=True)
    net.backbone.load_state_dict(pretrained_backbone.state_dict(), strict=True)

EFFNET_MEAN = [0.485, 0.456, 0.406]
EFFNET_STD = [0.229, 0.224, 0.225]

transform = transforms.Compose(
    [
        trim(),
        cropTo4_3(),
        transforms.Lambda(lambda im: ben_graham_preprocess(im, out_size=input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=EFFNET_MEAN, std=EFFNET_STD),
    ]
)

net = net.to(device)
net.eval()




## === cell 5
class ImageDataset(Dataset):
    def __init__(self, ids, img_dir, transform=None, labels=None):
        self.ids = list(ids)
        self.img_dir = img_dir
        self.transform = transform
        self.labels = None if labels is None else list(labels)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        idx = self.ids[i]
        image_name = os.path.join(self.img_dir, f"{idx}.png")
        img = Image.open(image_name).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        if self.labels is None:
            return idx, img
        return idx, img, int(self.labels[i])


def predict_regression_df(model, ids, img_dir, tfm, batch_size=8, num_workers=2):
    ds = ImageDataset(ids, img_dir, transform=tfm, labels=None)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    preds = []
    out_ids = []
    with torch.no_grad():
        for batch_ids, batch_imgs in dl:
            batch_imgs = batch_imgs.to(device, non_blocking=True)
            out = model(batch_imgs, final=True).squeeze(1)
            out = torch.clamp(out, 0.0, 4.0)
            preds.append(out.detach().cpu().numpy())
            out_ids.extend([str(x) for x in batch_ids])
    preds = np.concatenate(preds, axis=0)
    return pd.DataFrame({"id_code": np.array(out_ids, dtype=str), "pred": preds})


def apply_thresholds(preds, thr):
    thr = list(thr)
    out = np.zeros_like(preds, dtype=np.int64)
    for t in thr:
        out += (preds >= t).astype(np.int64)
    return out


def tune_thresholds(y_true, y_pred_cont, init_thr=(0.75, 1.5, 2.5, 3.5), steps=12):
    thr = np.array(init_thr, dtype=np.float64)

    def score(thr_vec):
        y_hat = apply_thresholds(y_pred_cont, thr_vec)
        return cohen_kappa_score(y_true, y_hat, weights="quadratic")

    thr = np.clip(thr, 0.0, 4.0)
    thr.sort()

    best = score(thr)

    step_sizes = np.linspace(0.25, 0.02, steps)
    for step in step_sizes:
        improved = True
        while improved:
            improved = False
            for i in range(4):
                for delta in (-step, step):
                    cand = thr.copy()
                    cand[i] = cand[i] + delta
                    cand = np.clip(cand, 0.0, 4.0)
                    cand.sort()
                    s = score(cand)
                    if s > best + 1e-8:
                        thr, best = cand, s
                        improved = True
    return thr.tolist(), float(best)


from sklearn.model_selection import StratifiedShuffleSplit

splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, val_idx = next(
    splitter.split(train_df["id_code"].values, train_df["diagnosis"].values)
)

val_ids = train_df.iloc[val_idx]["id_code"].values.astype(str)
val_y = train_df.iloc[val_idx]["diagnosis"].values.astype(int)

val_pred_df = predict_regression_df(
    net, val_ids, train_img_dir, transform, batch_size=8, num_workers=2
)

val_pred_df = val_pred_df.set_index("id_code").loc[val_ids].reset_index()
val_pred_cont = val_pred_df["pred"].values.astype(np.float64)

assert len(val_pred_cont) == len(val_y)

t0 = time.time()
best_thr, best_qwk = tune_thresholds(val_y, val_pred_cont, init_thr=threshold, steps=12)
threshold = best_thr
print(
    "Tuned thresholds:",
    threshold,
    "val_qwk:",
    best_qwk,
    "tuning_sec:",
    round(time.time() - t0, 2),
)



## === cell 6
test_pred_df = predict_regression_df(
    net, test_ids, test_img_dir, transform, batch_size=8, num_workers=2
)

test_pred_df = (
    test_pred_df.set_index("id_code")
    .loc[test_df["id_code"].astype(str).values]
    .reset_index()
)

test_pred_cls = apply_thresholds(test_pred_df["pred"].values, threshold).astype(int)

df = pd.DataFrame(
    {
        "id_code": test_pred_df["id_code"].astype(str).values,
        "diagnosis": test_pred_cls.astype(int),
    }
)

if df.empty:
    raise RuntimeError("Submission DataFrame is empty; inference produced no rows.")

df.to_csv("submission.csv", index=False)
print(df.head())
print("Wrote submission.csv with", len(df), "rows")
print("Resolved train_img_dir:", train_img_dir)
print("Resolved test_img_dir:", test_img_dir)
print("Using input_size:", input_size)
print("Using normalization mean/std:", EFFNET_MEAN, EFFNET_STD)
print("Using weight_path:", weight_path)
