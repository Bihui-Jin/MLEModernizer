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

0.9045937707959956

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I (1) remove the failing `pip install` and instead use the already-installed `timm` to keep the environment stable, (2) make device selection safe by using CUDA only if available (fixes the “no NVIDIA driver” crash), and (3) fix the missing weights path by automatically looking for the checkpoint in common Kaggle input locations and loading it with `map_location`. I also fix a couple of small transform/utility bugs that can silently break preprocessing (`trim` returning `None`, and `is` vs `==` for string compare) so inference always produces non-empty predictions. Finally, I ensure the submission is written with the required columns and correct ordering to `submission.csv`.'
- What this solution (achieved 0.0) has done: 'I (1) make checkpoint loading robust by falling back to a “no-weights” inference mode that still produces a valid `submission.csv` instead of crashing when the weight file isn’t provided, and (2) fix the CUDA/CPU dtype mismatch by ensuring the model weights and inputs land on the same device after loading. These changes are minimal and don’t alter the model architecture or prediction logic; they only make execution stable in Kaggle’s environment. With weights available, the score should increase from 0.0 toward the target because you actually be using the trained checkpoint instead of failing or outputting empty/invalid predictions.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is most consistent with “predictions are essentially random / all-one-class” because the intended checkpoint is not being found/loaded in the Kaggle environment, so the model runs with random weights. The smallest change to move toward the target is to make checkpoint discovery also search `../input/**` for *any* plausible EfficientNet-B4 3-stage weights file (not just the single hardcoded name), and load it with tolerant key-handling (state_dict nesting, module-prefix stripping, and `strict=False` fallback) so it actually uses the provided trained weights if present. I also keep the exact inference path (using the regressor head + your fixed thresholds) but add a safe, deterministic “autothreshold calibration on train.csv” fallback only when weights are missing, which should improve above 0.0 without changing the model architecture or training. The script still always produce a valid `submission.csv` with correct ordering and columns.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score strongly suggests the submission is valid-format but the predictions are effectively useless (most commonly: the intended checkpoint is not actually being found/loaded, so the network runs with random weights). I make the smallest change that increases the chance of loading the real trained weights by (1) searching also under `/kaggle/input/**` (some environments resolve paths differently) and (2) handling common checkpoint wrappers (`model`, `net`, `ema_state_dict`) in addition to `state_dict`, plus a last-resort “shape-matched subset load” that still preserves your exact model and inference logic. I not change the model architecture, transforms, or prediction method; only checkpoint discovery/loading robustness so the same model can actually use the provided weights. This should move the score upward toward your target if the weights exist in the notebook environment.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with a silent inference mismatch (wrong checkpoint loaded or not loaded, and/or using the wrong forward branch), yielding near-random/all-one-class predictions. I keep your exact model and preprocessing, but make the checkpoint search/load slightly more deterministic and correct by (1) preferring an exact filename match anywhere under `../input` and `/kaggle/input`, and (2) handling common “wrapped” checkpoints more reliably (including nested `state_dict` and `module.` prefixes) while avoiding accidentally loading an unrelated model. Finally, I use the model’s intended `final=True` regressor output (still your same architecture and same thresholding) because your checkpoint name suggests a 3-stage “final regressor” was trained; this typically moves QWK up substantially versus using the intermediate `regressor` head.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is consistent with the model running with random weights (checkpoint not found/loaded) and/or a mild preprocessing mismatch with how the checkpoint was trained. I keep your exact model and inference path, but make checkpoint discovery include `/kaggle/working/**` (where users sometimes place weights) and make loading prefer the most plausible match while rejecting tiny/unrelated files. I also make inference faster and more stable (no score change intent) by batching in a DataLoader, and I only apply the “threshold calibration fallback” when weights are truly missing (unchanged behavior). These changes are minimal and focused on actually using the intended trained checkpoint so the score can move upward toward your 0.9046 target.'
- What this solution (achieved 0.03519) has done: 'Your 0.0 score is almost certainly because the model is still running with random weights (checkpoint not found in this dataset), so the smallest legitimate way to move toward the target is to reliably use pretrained weights that are actually available (`timm` ImageNet weights) without changing your model architecture or inference logic. I keep your exact model and `final=True` forward path, but enable `pretrained=True` for the EfficientNet-B4 backbone **only when no checkpoint is loaded**, so you get a meaningful DR signal instead of random outputs. I also adjust the “no-checkpoint” threshold calibration to match the model’s 0..4.5 output range more sensibly by mapping class priors to quantiles of the regressor output (still only used when weights are missing). The submission writing/ordering remains identical.'
- What this solution (achieved -0.10946) has done: 'Your current score (0.03519) is far below the target (0.9046), so we should push performance up with minimal, low-risk changes that preserve your exact model/inference logic. The biggest likely issue is a train/inference preprocessing mismatch: `tf_efficientnet_b4_ns` expects ImageNet normalization by default, but you’re using custom mean/std; switching to timm’s default config normalization/resize parameters (while keeping your trim/crop steps) typically yields a large jump without changing the model architecture or prediction method. I also add a tiny safety fix to ensure the final `df` ordering exactly matches `test.csv` (avoid any accidental merge ordering edge cases), and keep the rest identical. These changes are targeted to improve QWK by making the backbone see inputs in the distribution it was pretrained/trained on, while still producing the same submission schema.'
- What this solution (achieved -0.10754) has done: 'Your current score is far below the target (higher-is-better), so we should make the smallest changes that plausibly increase QWK without changing your model architecture or inference semantics. The most likely cause of a negative QWK here is a preprocessing mismatch: you use timm’s ImageNet mean/std but still force a fixed 380 resize, whereas tf_efficientnet_b4_ns expects its own default input size/interpolation/crop_pct; aligning those typically gives a large, low-risk boost. I keep your trim + 4:3 crop, but switch the resize to the model’s resolved `input_size` and use timm’s recommended interpolation. I also add a deterministic test-time augmentation (simple horizontal flip average) which preserves the same model and thresholds but usually improves agreement for this task.'
- What this solution (achieved 0.0) has done: 'Your score is far below the target (higher-is-better), so the most likely issue is that the model outputs aren’t being converted to classes in a way that matches the metric’s ordinal nature. I keep your exact model, weights-loading logic, transforms, and TTA, but replace the fixed hand-picked thresholds with thresholds fit on the training set using out-of-fold predictions from your current model (no training, just inference), directly optimizing quadratic weighted kappa. This is a minimal, metric-aligned post-processing change that typically yields a large QWK jump without altering the model architecture or forward pass. If checkpoint weights are missing, it still works (using the current backbone weights) and produce a valid `submission.csv`.'
- What this solution (achieved -0.00785) has done: 'Your current 0.0 score is consistent with a calibration failure: even with a reasonable model output, poorly chosen thresholds can map most predictions to the wrong ordinal bins, tanking QWK. I keep your exact model, transforms, TTA, and inference path, but replace the current slow/fragile OOF-threshold fitting with a deterministic, metric-aligned threshold optimizer that directly maximizes quadratic weighted kappa on the training set predictions (no training involved). This is a minimal post-processing change that typically moves QWK upward substantially while preserving evaluation semantics (still regressor→4 cutpoints→{0..4}). I also guard threshold fitting so it only runs when it can finish within the time budget, ensuring the notebook always writes a valid `submission.csv`.'
- What this solution (achieved -0.00785) has done: 'Your current negative QWK is most consistent with a calibration/threshold mismatch: even if the regression outputs contain some signal, mapping them to {0..4} with thresholds fit on the *same* train set can still be unstable and can also overfit in a way that hurts test QWK. To move the score upward toward the target with minimal semantic change, I keep your exact model, transforms, and regressor→4-threshold discretization, but fit thresholds using fast 5-fold out-of-fold (OOF) predictions and then optimize thresholds on those OOF predictions (still no training, just inference). I also make the threshold optimizer slightly more robust by performing a deterministic coordinate-descent search over a bounded range using OOF predictions, and I only run OOF fitting when no checkpoint is loaded (when weights are loaded, we keep your original fixed thresholds for stability). Submission writing/order stays identical and still produces `submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your current negative QWK is most consistent with the fallback path (no checkpoint found) producing regressor outputs that don’t align well with your fixed thresholds, so the class mapping is effectively miscalibrated. To move the score upward with minimal semantic change, I keep your exact model and inference (regressor → 4 thresholds → {0..4}, same TTA), but (only when no checkpoint is loaded) fit the 4 thresholds directly to maximize quadratic weighted kappa on out-of-fold predictions using a fast, deterministic coordinate-descent search. This avoids overfitting to in-sample predictions and is more metric-aligned than quantiles alone, while preserving your core logic. I also ensure OOF fitting always uses only the fold’s validation images (no leakage) and stays within the time budget so `submission.csv` is always produced.'

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

import torchvision.transforms as transforms
from torchvision.transforms import functional as FT
from PIL import Image, ImageChops

from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold
import timm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.set_grad_enabled(False)

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)



## === cell 1
threshold = [0.75, 1.5, 2.5, 3.5]


def regress2class(out):
    out = out.view(-1)
    prediction = torch.zeros(out.size(0), dtype=torch.int64)
    for i in range(4):
        prediction += (out.data >= threshold[i]).to(torch.int64).cpu()
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
    out = out.view(-1)
    pred_prob = torch.zeros((out.size(0), 5), device=out.device)
    for i in range(out.size(0)):
        v = float(out[i].item())
        if v < 4.0:
            l1 = int(math.floor(v))
            l2 = int(math.ceil(v))
            pred_prob[i][l1] = 1 - (v - l1)
            pred_prob[i][l2] = 1 - (l2 - v)
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
from torch.utils.data import Dataset, DataLoader

TEST_CSV = "../input/aptos2019-blindness-detection/test.csv"
TEST_IMG_DIR = "../input/aptos2019-blindness-detection/test_images"
TRAIN_CSV = "../input/aptos2019-blindness-detection/train.csv"
TRAIN_IMG_DIR = "../input/aptos2019-blindness-detection/train_images"

test_df = pd.read_csv(TEST_CSV)
test_ids = test_df["id_code"].astype(str).values

_probe_model = timm.create_model("tf_efficientnet_b4_ns", pretrained=True)
timm_cfg = timm.data.resolve_model_data_config(_probe_model)
mean, std = timm_cfg["mean"], timm_cfg["std"]
input_size = int(timm_cfg["input_size"][-1])

transform = transforms.Compose(
    [
        trim(),
        cropTo4_3(),
        transforms.Resize(
            (input_size * 3 // 4, input_size),
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]
)


def find_checkpoint():
    roots = ["../input", "/kaggle/input", "../working", "/kaggle/working"]
    exact_name = "B4_3stage_39epoch_CLAHE"

    exact_hits = []
    for root in roots:
        for ext in ("pkl", "pth", "pt", "bin"):
            exact_hits += glob.glob(
                os.path.join(root, "**", f"{exact_name}.{ext}"), recursive=True
            )

    exact_hits = [p for p in exact_hits if os.path.isfile(p)]
    exact_hits = [p for p in exact_hits if os.path.getsize(p) > 1_000_000]
    if exact_hits:
        exact_hits = sorted(exact_hits, key=lambda p: (len(p), p))
        return exact_hits[0]

    candidates = []
    keywords = (
        "b4_3stage",
        "3stage_39epoch",
        "clahe",
        "efficientnet_b4",
        "tf_efficientnet_b4",
    )
    for root in roots:
        for ext in ("*.pkl", "*.pth", "*.pt", "*.bin"):
            for p in glob.glob(os.path.join(root, "**", ext), recursive=True):
                if not os.path.isfile(p):
                    continue
                if os.path.getsize(p) <= 1_000_000:
                    continue
                bn = os.path.basename(p).lower()
                if any(k in bn for k in keywords):
                    candidates.append(p)

    if candidates:
        candidates = sorted(candidates, key=lambda p: (len(p), -os.path.getsize(p), p))
        return candidates[0]
    return None


def _extract_state_dict(obj):
    if isinstance(obj, dict):
        for key in ("state_dict", "model_state_dict", "model", "net", "ema_state_dict"):
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
    return obj if isinstance(obj, dict) else None


def _sanitize_state_dict(state):
    state = _extract_state_dict(state)
    if not isinstance(state, dict):
        return None

    new_state = {}
    for k, v in state.items():
        nk = k
        if isinstance(nk, str) and nk.startswith("module."):
            nk = nk.replace("module.", "", 1)
        new_state[nk] = v
    return new_state


def _load_state_dict_best_effort(model, state):
    try:
        model.load_state_dict(state, strict=True)
        return True, "strict=True"
    except Exception:
        try:
            incompatible = model.load_state_dict(state, strict=False)
            missing = getattr(incompatible, "missing_keys", [])
            unexpected = getattr(incompatible, "unexpected_keys", [])
            return (
                True,
                f"strict=False (missing={len(missing)}, unexpected={len(unexpected)})",
            )
        except Exception as e2:
            model_sd = model.state_dict()
            filtered = {}
            for k, v in state.items():
                if (
                    k in model_sd
                    and hasattr(v, "shape")
                    and hasattr(model_sd[k], "shape")
                    and tuple(v.shape) == tuple(model_sd[k].shape)
                ):
                    filtered[k] = v
            if len(filtered) == 0:
                return False, f"failed: {type(e2).__name__}: {e2}"
            incompatible = model.load_state_dict(filtered, strict=False)
            missing = getattr(incompatible, "missing_keys", [])
            unexpected = getattr(incompatible, "unexpected_keys", [])
            return (
                True,
                f"shape-matched subset (loaded={len(filtered)}, missing={len(missing)}, unexpected={len(unexpected)})",
            )


ckpt_path = find_checkpoint()
weights_loaded = False
load_msg = "none"

net = ThreeStage_Model(pretrained_backbone=(ckpt_path is None))

if ckpt_path is not None:
    state = torch.load(ckpt_path, map_location="cpu")
    state = _sanitize_state_dict(state)
    if state is not None:
        weights_loaded, load_msg = _load_state_dict_best_effort(net, state)
        if not weights_loaded:
            print(
                f"WARNING: checkpoint found at {ckpt_path} but could not be loaded; using fallback weights. ({load_msg})"
            )
    else:
        print(
            f"WARNING: checkpoint found at {ckpt_path} but not a state_dict-like object; using fallback weights."
        )
else:
    print(
        "WARNING: no suitable checkpoint found under ../input, /kaggle/input, ../working, or /kaggle/working. "
        "Proceeding with timm pretrained backbone."
    )

net = net.to(device)
net.eval()

print(
    f"Device: {device} | Weights loaded: {weights_loaded} | Load mode: {load_msg} | Checkpoint: {ckpt_path}"
)


class TestDS(Dataset):
    def __init__(self, ids, img_dir, transform):
        self.ids = list(ids)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        idx = self.ids[i]
        image_name = f"{self.img_dir}/{idx}.png"
        if not os.path.isfile(image_name):
            raise FileNotFoundError(f"Missing test image: {image_name}")
        img = Image.open(image_name).convert("RGB")
        img = self.transform(img)
        return idx, img


class TrainDS(Dataset):
    def __init__(self, ids, labels, img_dir, transform):
        self.ids = list(ids)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        idx = self.ids[i]
        image_name = f"{self.img_dir}/{idx}.png"
        if not os.path.isfile(image_name):
            raise FileNotFoundError(f"Missing train image: {image_name}")
        img = Image.open(image_name).convert("RGB")
        img = self.transform(img)
        y = int(self.labels[i])
        return idx, img, y


def _predict_regression(ids, img_dir, batch_size):
    ds = TestDS(ids, img_dir, transform)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )
    outs = []
    out_ids = []
    for batch in dl:
        b_ids, imgs = batch
        imgs = imgs.to(device, non_blocking=True)

        r1 = net(imgs, final=True)
        r2 = net(torch.flip(imgs, dims=[3]), final=True)
        r_out = (r1 + r2) / 2.0

        outs.append(r_out.detach().float().cpu().view(-1).numpy())
        out_ids.extend([str(x) for x in b_ids])
    outs = np.concatenate(outs, axis=0)
    return np.array(out_ids, dtype=object), outs


def _apply_thresholds(reg_out, thr):
    reg_out = np.asarray(reg_out, dtype=np.float64).reshape(-1)
    pred = np.zeros_like(reg_out, dtype=np.int64)
    for t in thr:
        pred += (reg_out >= float(t)).astype(np.int64)
    pred = np.clip(pred, 0, 4)
    return pred


def fit_thresholds_by_oof_qwk(train_csv_path=TRAIN_CSV, n_splits=5, time_budget_s=520):
    """
    Change rationale (score-moving, minimal semantics change):
    - Keeps the exact same prediction semantics: regression output -> 4 cutpoints -> {0..4}.
    - Improves calibration by optimizing thresholds for QWK on *OOF* predictions (less overfit than in-sample),
      which is the most likely cause of negative/near-zero public score when weights are missing.
    """
    global threshold
    t0 = time.time()

    if not os.path.isfile(train_csv_path):
        print("WARNING: TRAIN_CSV not found; keeping existing thresholds.")
        return

    tr = pd.read_csv(train_csv_path)
    if "diagnosis" not in tr.columns:
        print("WARNING: TRAIN_CSV missing diagnosis; keeping existing thresholds.")
        return

    tr_ids = tr["id_code"].astype(str).values
    y = tr["diagnosis"].astype(int).values

    bs = 12 if device.type == "cuda" else 6

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_pred = np.zeros(len(tr_ids), dtype=np.float64)
    oof_mask = np.zeros(len(tr_ids), dtype=bool)

    for fold, (_, val_idx) in enumerate(skf.split(tr_ids, y), 1):
        if time.time() - t0 > time_budget_s * 0.80:
            print(
                f"WARNING: time budget reached during OOF; using partial OOF (folds done: {fold-1})."
            )
            break
        val_ids = tr_ids[val_idx]
        _, p = _predict_regression(val_ids, TRAIN_IMG_DIR, batch_size=bs)
        oof_pred[val_idx] = p
        oof_mask[val_idx] = True

    if not oof_mask.any():
        print("WARNING: OOF predictions not computed; keeping existing thresholds.")
        return

    oof_pred_used = oof_pred[oof_mask]
    y_used = y[oof_mask]

    thr0 = np.quantile(oof_pred_used, [0.2, 0.4, 0.6, 0.8]).astype(np.float64)
    thr0 = np.clip(thr0, 0.05, 4.45)
    thr0 = np.maximum.accumulate(thr0)

    best_thr = thr0.copy()
    best_kappa = cohen_kappa_score(
        y_used, _apply_thresholds(oof_pred_used, best_thr), weights="quadratic"
    )

    step_schedule = [0.30, 0.15, 0.08, 0.04, 0.02]
    for step in step_schedule:
        improved = True
        while improved and (time.time() - t0) < time_budget_s:
            improved = False
            for i in range(4):
                cur = float(best_thr[i])
                candidates = (cur - step, cur, cur + step)
                for cand in candidates:
                    trial = best_thr.copy()
                    trial[i] = float(cand)
                    trial = np.clip(trial, 0.05, 4.45)
                    trial = np.maximum.accumulate(trial)
                    kappa = cohen_kappa_score(
                        y_used,
                        _apply_thresholds(oof_pred_used, trial),
                        weights="quadratic",
                    )
                    if kappa > best_kappa + 1e-10:
                        best_kappa = kappa
                        best_thr = trial
                        improved = True

    threshold = [float(x) for x in best_thr.tolist()]
    print(
        f"Fitted thresholds via OOF QWK optimization (OOF QWK={best_kappa:.6f}): {threshold} | fit_time={time.time()-t0:.1f}s"
    )


if not weights_loaded:
    fit_thresholds_by_oof_qwk()
else:
    print(f"Using fixed thresholds (checkpoint-loaded stability): {threshold}")

bs = 8 if device.type == "cuda" else 4
num_workers = 2

ds = TestDS(test_ids, TEST_IMG_DIR, transform)
dl = DataLoader(
    ds,
    batch_size=bs,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=(device.type == "cuda"),
)

submission = []
for batch in dl:
    ids, imgs = batch
    imgs = imgs.to(device, non_blocking=True)

    r1 = net(imgs, final=True)
    r2 = net(torch.flip(imgs, dims=[3]), final=True)  # horizontal flip
    r_out = (r1 + r2) / 2.0

    pred = regress2class(r_out.data.squeeze(1))
    for k in range(len(ids)):
        submission.append([str(ids[k]), int(pred[k].item())])

submission = np.array(submission, dtype=object)



## === cell 5
df = pd.DataFrame(submission, columns=["id_code", "diagnosis"])

df = test_df[["id_code"]].merge(df, on="id_code", how="left")
df["diagnosis"] = df["diagnosis"].fillna(0).astype(int)

out_path = "submission.csv"
df.to_csv(out_path, index=False)

print(df.head())
print(f"Wrote {out_path} with shape={df.shape} and columns={list(df.columns)}")
print(df["diagnosis"].value_counts(dropna=False).sort_index())
