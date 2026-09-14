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

0.8990568894881139

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the missing weights issue by automatically falling back to the already-installed `timm` (instead of trying to install a non-existent wheel) and by loading the model weights only if they exist; otherwise the notebook still run end-to-end. I also fix the CUDA crash by selecting `cuda` only when available, defaulting to CPU so it runs in Kaggle’s no-GPU environment. To prevent an empty submission, I make inference robust to image/transform edge cases (e.g., `trim()` returning `None`) and ensure every `id_code` gets a prediction. Finally, I write `submission.csv` with the required columns and row count.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is almost certainly coming from running with randomly initialized weights (the warning shows weights usually aren’t found in your environment), which produce near-random predictions and tank QWK. I keep your exact model and preprocessing, but make the weights lookup robust to the actual Kaggle file layout you showed (the dataset exists under `/kaggle/input/...`), and also allow loading common checkpoint formats (`state_dict` wrapped dicts). Additionally, I switch inference to use the model’s `final=True` head (it’s part of your existing architecture and intended for a single regression output), then apply your same rounding/clipping to 0–4. These changes are minimal, preserve core logic, and should move the score up toward your target by ensuring you’re actually using the trained checkpoint and the correct prediction head.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score indicates the model is still effectively untrained at inference time (random weights), which happens because no checkpoint is found/loaded in the current environment. I keep your exact model, transforms, and rounding/clipping, but expand the checkpoint search to include common Kaggle dataset locations (including `/kaggle/input/**` subfolders) and also accept typical wrapper keys like `model`, `net`, and `ema_state_dict`. This is the smallest change that should move QWK sharply upward toward your target by ensuring the intended trained weights are actually used. I also add a hard check: if no weights are found, we still generate a valid submission, but we print an explicit warning showing the searched paths to avoid silently submitting random predictions again.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is consistent with still not loading the intended trained checkpoint, so the main change is to make weight loading succeed reliably by (1) searching for any plausible checkpoint file in the Kaggle input tree (not just a single hardcoded basename) and (2) correctly extracting a `state_dict` even when it’s nested under common keys and/or stored with `module.`/`model.` prefixes. I keep your exact model, transforms, and inference/rounding logic, but add a strict sanity check that warns if too few parameters were actually loaded (a common silent failure case with `strict=False`). This is the smallest safe change expected to move QWK sharply upward toward your target by ensuring inference uses the trained weights instead of random initialization, while still producing a valid `submission.csv` in all cases.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with inference using random weights (checkpoint not found or not actually loaded into the model), so the smallest change to move toward the 0.899 target is to make checkpoint discovery/loading succeed reliably and fail loudly when it doesn’t. I keep your exact model, transforms, and rounding/clipping logic, but (1) expand the checkpoint search to include the current working directory and common Kaggle dataset roots, (2) make state-dict extraction handle more real-world wrapper formats and prefixes, and (3) add a strict sanity check that aborts submission generation if no usable checkpoint is loaded (preventing another near-random 0.0 submission). This preserves core semantics while making it much more likely you actually run the intended trained weights. The script still writes a valid `submission.csv` when weights are correctly found/loaded.'
- What this solution (achieved 0.0) has done: 'I remove the hard failure when no checkpoint is found (so the notebook always produces a valid `submission.csv`) while still attempting to discover and load weights when they exist. I also fix the device/dtype mismatch by ensuring the model is moved to the selected device *after* any weight loading, and by explicitly aligning the input tensor dtype/device to the model’s parameters during inference. Finally, I keep your exact model, transforms, and rounding/clipping semantics unchanged, only making the pipeline robust so it runs end-to-end and avoids the current runtime errors.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score strongly suggests the submission is effectively random or degenerate (e.g., nearly constant), which most commonly happens here when the intended checkpoint is not found/loaded or the wrong file is picked from the broad fallback glob. I make the smallest change that increases the chance of loading the *correct* trained weights by (1) searching explicitly for typical “best/epoch/fold” checkpoint names for this exact 3-stage EfficientNet-B4 model, and (2) ranking candidates so we prefer clearly-relevant files over arbitrary `.pth/.pkl` matches. I also add a lightweight sanity check on predictions (distribution + variance) so we can catch “all zeros / constant” outputs before writing the submission (still writing a valid CSV, but warning loudly). Core model, transforms, and rounding/clipping are unchanged; only checkpoint discovery/selection and safety diagnostics are adjusted to move QWK upward toward your target.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is still most consistent with predictions coming from a mismatched/noisy checkpoint load (or no checkpoint at all), so the minimal change to move toward the 0.899 target is to (1) make checkpoint selection prefer files that actually match the model by trying top-ranked candidates until one loads with a high loaded_ratio, and (2) enforce the correct inference head/output shape robustly (flattening to 1D before rounding). This keeps your exact model, transforms, and rounding/clipping semantics, but greatly reduces the chance you silently load the wrong `.pth/.pkl` and submit near-random/constant outputs. The pipeline remains end-to-end and always writes a valid `submission.csv`; if no sufficiently-matching checkpoint is found, it warn clearly (so you don’t accidentally submit another random 0.0). All changes are directly tied to using the intended trained weights reliably, which is the biggest lever to raise QWK from 0.0 toward the target.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is most consistent with the test images not being found due to a wrong `test_images` directory resolution, causing many/all predictions to fall back to 0 (and thus QWK ≈ 0). I keep your model, transforms, and rounding exactly the same, but make the image directory detection robust by auto-selecting the first existing directory among common Kaggle layouts (including the nested `.../test_images/test_images` case visible in your file tree). I also add a quick sanity print of the resolved `TEST_IMG_DIR` and how many test images exist there, so you don’t silently submit an “all missing images → all zeros” file again. These are minimal changes directly tied to producing non-degenerate predictions, which should increase QWK toward your target.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is still most consistent with either (a) using the wrong checkpoint (e.g., optimizer/scheduler or a different architecture) or (b) silently “loading” a checkpoint but with a badly mismatched state_dict, yielding effectively random/degenerate predictions. I keep your exact model, transforms, and inference semantics, but make checkpoint selection/load more reliable by (1) preferring candidates that actually match *this* model via a quick key-shape compatibility precheck (before load_state_dict) and (2) only accepting checkpoints that load with a high parameter match ratio, otherwise continuing to the next candidate instead of using a weak fallback. This is the smallest change that should materially raise QWK toward your target by ensuring you’re truly using the intended trained weights, while still producing a valid `submission.csv` end-to-end. I also add a final warning that explicitly lists the top tried checkpoints and their match ratios so you can verify what was actually loaded.'
- What this solution (achieved 0.0) has done: 'Your 0.0 QWK is consistent with submitting essentially random/degenerate predictions, which in your pipeline most plausibly comes from (a) failing to load a compatible trained checkpoint and/or (b) using a different head than the checkpoint was trained for. I keep your exact model, transforms, and rounding/clipping semantics, but make checkpoint loading more reliable by also checking for checkpoints shipped inside the competition dataset directory and by accepting checkpoints that match at least one of the model’s common submodules (not only the full combined model). If a checkpoint matches only the backbone/classifier/regressor/ordinal but not `final_regressor`, I automatically fall back to `final=False` inference and use your existing `regress2class()` thresholds (already defined) to convert regression output to classes—this preserves your core approach while avoiding the “wrong head” failure mode. The script still always produce a valid `submission.csv`, but now it’s much less likely to silently run with unusable weights and score ~0.0.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most likely coming from a mismatch between what the model was trained to output and how we convert outputs to the 0–4 labels at inference time (especially if the checkpoint was trained using the regressor+ordinal/classifier branches rather than the `final=True` head). To move the score upward toward the 0.899 target with minimal changes, I (1) automatically choose the inference path based on which head actually matches the loaded checkpoint (prefer `final=True` only when `final_regressor.*` is present; otherwise use the existing `regress2class()` thresholding on `r_out`), and (2) make checkpoint acceptance slightly less brittle while still rejecting obviously-wrong files (so we actually load a compatible model instead of effectively-random weights). This preserves your architecture, transforms, and label mapping logic, but prevents the common “loaded backbone only + wrong head used” failure mode that yields near-random/degenerate submissions. The script still run end-to-end and always write a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import random
import glob
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import torchvision.transforms as T
from PIL import Image, ImageChops

import timm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)



## === cell 1
threshold = [0.7, 1.5, 2.5, 3.5]


def regress2class(out):
    prediction = torch.zeros(out.size(0))
    for i in range(4):
        prediction += (out.data >= threshold[i]).squeeze().cpu()
    return prediction




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
DATA_ROOT_CANDIDATES = [
    "../input/aptos2019-blindness-detection",
    "/kaggle/input/aptos2019-blindness-detection",
    "/kaggle/data/aptos2019-blindness-detection",
    "/kaggle/input/aptos2019-blindness-detection/aptos2019-blindness-detection",
    "/kaggle/data/aptos2019-blindness-detection/aptos2019-blindness-detection",
]
DATA_ROOT = None
for p in DATA_ROOT_CANDIDATES:
    if os.path.exists(p):
        DATA_ROOT = p
        break
if DATA_ROOT is None:
    DATA_ROOT = "../input/aptos2019-blindness-detection"

TEST_CSV = os.path.join(DATA_ROOT, "test.csv")

TEST_IMG_DIR_CANDIDATES = [
    os.path.join(DATA_ROOT, "test_images"),
    os.path.join(DATA_ROOT, "test_images", "test_images"),
    "/kaggle/input/aptos2019-blindness-detection/test_images",
    "/kaggle/input/aptos2019-blindness-detection/test_images/test_images",
    "/kaggle/data/aptos2019-blindness-detection/test_images",
    "/kaggle/data/aptos2019-blindness-detection/test_images/test_images",
]
TEST_IMG_DIR = None
for p in TEST_IMG_DIR_CANDIDATES:
    if os.path.isdir(p):
        TEST_IMG_DIR = p
        break
if TEST_IMG_DIR is None:
    TEST_IMG_DIR = os.path.join(DATA_ROOT, "test_images")

test_ids_df = pd.read_csv(TEST_CSV)
test_ids = np.squeeze(test_ids_df.values)

existing_test_imgs = glob.glob(os.path.join(TEST_IMG_DIR, "*.png"))
print(f"Resolved DATA_ROOT={DATA_ROOT}")
print(
    f"Resolved TEST_IMG_DIR={TEST_IMG_DIR} (found {len(existing_test_imgs)} .png files)"
)

input_size = 320

tfm = T.Compose(
    [
        trim(),
        T.Resize((input_size, (input_size * 3) // 4)),
        T.ToTensor(),
        T.Normalize(mean=[0.384, 0.258, 0.174], std=[0.124, 0.089, 0.094]),
    ]
)

net = ThreeStage_Model()

WEIGHTS_BASENAME = "B4_3stage_75epoch_AdamW.pkl"

LIKELY_NAMES = [
    WEIGHTS_BASENAME,
    "B4_3stage_best.pth",
    "B4_3stage_best.pkl",
    "B4_3stage.pth",
    "B4_3stage.pkl",
    "three_stage_b4.pth",
    "three_stage_b4.pkl",
    "3stage_b4.pth",
    "3stage_b4.pkl",
    "best.pth",
    "best.pkl",
    "model_best.pth",
    "model_best.pkl",
]

SEARCH_ROOTS = [
    ".",
    "..",
    "/kaggle/input",
    "/kaggle/data",
    "/kaggle/working",
    DATA_ROOT,
    os.path.join(DATA_ROOT, "weights"),
    os.path.join(DATA_ROOT, "checkpoints"),
    os.path.join(DATA_ROOT, "models"),
]


def _gather_candidates():
    cands = []
    subdirs = [
        "",
        "weights",
        "checkpoints",
        "models",
        "model",
        "ckpt",
        "output",
        "outputs",
    ]
    for root in SEARCH_ROOTS:
        for sd in subdirs:
            for nm in LIKELY_NAMES:
                cands.append(
                    os.path.join(root, sd, nm) if sd else os.path.join(root, nm)
                )

    for nm in LIKELY_NAMES:
        for pat in [
            f"/kaggle/input/**/{nm}",
            f"/kaggle/data/**/{nm}",
            f"/kaggle/working/**/{nm}",
        ]:
            cands.extend(sorted(glob.glob(pat, recursive=True)))

    fallback_patterns = [
        "/kaggle/input/**/*.pkl",
        "/kaggle/input/**/*.pth",
        "/kaggle/input/**/*.pt",
        "/kaggle/data/**/*.pkl",
        "/kaggle/data/**/*.pth",
        "/kaggle/data/**/*.pt",
        "/kaggle/working/**/*.pkl",
        "/kaggle/working/**/*.pth",
        "/kaggle/working/**/*.pt",
    ]
    fallback_files = []
    for pat in fallback_patterns:
        fallback_files.extend(glob.glob(pat, recursive=True))
    cands.extend(sorted(set(fallback_files)))

    seen = set()
    out = []
    for p in cands:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def _is_likely_ckpt(path: str) -> bool:
    name = os.path.basename(path).lower()
    if any(x in name for x in ["tf_efficientnet", "efficientnet", "b4", "b5"]):
        return True
    if any(x in name for x in ["aptos", "blind", "retina", "dr"]):
        return True
    if any(x in name for x in ["3stage", "three", "stage"]):
        return True
    if any(x in name for x in ["fold", "epoch", "best", "final"]):
        return True
    if any(x in name for x in ["weight", "ckpt", "checkpoint"]):
        return True
    return False


def _rank_ckpt(path: str) -> tuple:
    name = os.path.basename(path).lower()
    score = 0
    if "3stage" in name or "three" in name:
        score += 50
    if "b4" in name:
        score += 25
    if "efficientnet" in name:
        score += 10
    if "best" in name:
        score += 20
    if "75epoch" in name:
        score += 10
    if name.endswith(".pth") or name.endswith(".pt"):
        score += 2
    if name.endswith(".pkl"):
        score += 1
    if any(x in name for x in ["optimizer", "sched", "scheduler", "adamw"]):
        score -= 5
    score += 5 if _is_likely_ckpt(path) else -100
    return (score, -len(path))


def _strip_prefix(sd, prefix):
    out = {}
    for k, v in sd.items():
        if isinstance(k, str) and k.startswith(prefix):
            out[k[len(prefix) :]] = v
        else:
            out[k] = v
    return out


def _extract_state_dict(maybe_ckpt):
    if isinstance(maybe_ckpt, dict):
        for key in [
            "state_dict",
            "model_state_dict",
            "model",
            "net",
            "ema_state_dict",
            "ema",
            "teacher",
            "student",
            "module",
            "weights",
        ]:
            if key in maybe_ckpt and isinstance(maybe_ckpt[key], dict):
                sd = maybe_ckpt[key]
                for pref in ["model.", "net.", "module."]:
                    sd = _strip_prefix(sd, pref)
                return sd

        if all(isinstance(k, str) for k in maybe_ckpt.keys()):
            any_tensor = any(torch.is_tensor(v) for v in maybe_ckpt.values())
            if any_tensor:
                sd = maybe_ckpt
                for pref in ["model.", "net.", "module."]:
                    sd = _strip_prefix(sd, pref)
                return sd
    return maybe_ckpt


def _compatibility_ratio(model: nn.Module, sd: dict) -> float:
    if not isinstance(sd, dict):
        return 0.0
    msd = model.state_dict()
    match = 0
    total = 0
    for k, v in msd.items():
        total += 1
        if k in sd and torch.is_tensor(sd[k]) and tuple(sd[k].shape) == tuple(v.shape):
            match += 1
    return match / max(1, total)


def _module_compat(model: nn.Module, sd: dict, prefix: str) -> float:
    if not isinstance(sd, dict):
        return 0.0
    msd = model.state_dict()
    keys = [k for k in msd.keys() if k.startswith(prefix)]
    if len(keys) == 0:
        return 0.0
    match = 0
    for k in keys:
        if (
            k in sd
            and torch.is_tensor(sd[k])
            and tuple(sd[k].shape) == tuple(msd[k].shape)
        ):
            match += 1
    return match / max(1, len(keys))


WEIGHTS_CANDIDATES = _gather_candidates()
existing = [p for p in WEIGHTS_CANDIDATES if isinstance(p, str) and os.path.isfile(p)]
existing_sorted = sorted(existing, key=_rank_ckpt, reverse=True)

loaded_ok = False
weights_path = None

tried = []

use_final_head = False

MAX_TRIES = min(60, len(existing_sorted))
for cand in existing_sorted[:MAX_TRIES]:
    try:
        state = torch.load(cand, map_location="cpu")
        state = _extract_state_dict(state)

        sd = state if isinstance(state, dict) else {}

        comp_full = _compatibility_ratio(net, sd)
        comp_backbone = _module_compat(net, sd, "backbone.")
        comp_classifier = _module_compat(net, sd, "classifier.")
        comp_regressor = _module_compat(net, sd, "regressor.")
        comp_ordinal = _module_compat(net, sd, "ordinal.")
        comp_final = _module_compat(net, sd, "final_regressor.")

        tried.append((cand, comp_full, comp_backbone, comp_regressor, comp_final))

        acceptable = (comp_full >= 0.90) or (
            comp_backbone >= 0.98 and comp_regressor >= 0.90
        )
        if not acceptable:
            continue

        missing, unexpected = net.load_state_dict(sd, strict=False)

        total_params = len(list(net.state_dict().keys()))
        loaded_params = total_params - len(missing)
        loaded_ratio = loaded_params / max(1, total_params)

        use_final_head = comp_final >= 0.90

        if loaded_ratio >= 0.70 and comp_backbone >= 0.98 and comp_regressor >= 0.90:
            weights_path = cand
            loaded_ok = True
            print(f"Loaded weights (accepted): {weights_path}")
            print(
                f"load_state_dict strict=False; loaded_ratio={loaded_ratio:.3f}, "
                f"missing={len(missing)}, unexpected={len(unexpected)}, "
                f"compat_full={comp_full:.3f}, compat_backbone={comp_backbone:.3f}, "
                f"compat_regressor={comp_regressor:.3f}, compat_final={comp_final:.3f}"
            )
            print(
                f"Inference will use {'final=True head' if use_final_head else 'regressor+thresholds (final=False)'}"
            )
            break
        else:
            net = ThreeStage_Model()
    except Exception:
        net = ThreeStage_Model()
        continue

if not loaded_ok:
    top_tried = sorted(tried, key=lambda x: x[1], reverse=True)[:8]
    print(
        "WARNING: No sufficiently-matching checkpoint was loaded; proceeding with random initialization."
    )
    if len(top_tried) > 0:
        print("Top tried checkpoints by compat_full:")
        for p, rfull, rbb, rreg, rfin in top_tried:
            print(
                f"  compat_full={rfull:.3f} compat_backbone={rbb:.3f} compat_regressor={rreg:.3f} compat_final={rfin:.3f}  path={p}"
            )
    else:
        print("No checkpoints were tried (none found on disk).")
    use_final_head = False

net = net.to(device)
net.eval()



## === cell 5
submission = []
missing_imgs = 0

model_param = next(net.parameters())
model_device = model_param.device
model_dtype = model_param.dtype

preds_for_check = []

with torch.no_grad():
    for i, idx in enumerate(test_ids):
        image_name = os.path.join(TEST_IMG_DIR, f"{idx}.png")
        if not os.path.exists(image_name):
            missing_imgs += 1
            pred = 0
            submission.append([idx, pred])
            preds_for_check.append(pred)
            continue

        img = Image.open(image_name).convert("RGB")
        img = tfm(img).unsqueeze(0).to(device=model_device, dtype=model_dtype)

        if use_final_head:
            out = net(img, final=True)
            out = out.view(-1)
            pred = torch.round(out.data[0]).cpu().item()
            pred = int(max(0, min(4, pred)))
        else:
            _, r_out, _ = net(img, final=False)  # r_out already sigmoid*4.5 in forward
            pred = int(regress2class(r_out).item())
            pred = int(max(0, min(4, pred)))

        submission.append([idx, pred])
        preds_for_check.append(pred)

print(f"Done. Missing images: {missing_imgs}")

preds_arr = np.asarray(preds_for_check, dtype=np.int64)
uniq, cnt = np.unique(preds_arr, return_counts=True)
dist = dict(zip(uniq.tolist(), cnt.tolist()))
print(f"Prediction class distribution: {dist}")
if preds_arr.std() < 0.05:
    print(
        "WARNING: Predictions are nearly constant (very low std). "
        "This often indicates missing/mismatched weights and can yield ~0.0 QWK."
    )

submission = np.array(submission, dtype=object)



## === cell 6
df = pd.DataFrame(submission, columns=["id_code", "diagnosis"])

df["id_code"] = df["id_code"].astype(str)
df["diagnosis"] = df["diagnosis"].astype(int)
df = df.set_index("id_code").reindex(test_ids_df["id_code"].astype(str)).reset_index()

df["diagnosis"] = df["diagnosis"].fillna(0).astype(int)

out_path = "submission.csv"
df.to_csv(out_path, index=False)
print(df.head())
print(f"Wrote {out_path} with shape {df.shape}")
