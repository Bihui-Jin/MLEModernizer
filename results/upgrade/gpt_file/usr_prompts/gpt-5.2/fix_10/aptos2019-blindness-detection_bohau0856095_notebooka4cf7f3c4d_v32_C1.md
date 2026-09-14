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

0.9145905451464686

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.01221) has done: 'I fix the missing-weights crash by making the model loading robust: if the expected weight file is unavailable, the script fall back to an ImageNet-pretrained backbone and still run end-to-end (score likely be lower than the target but you get a valid submission). I also fix the CUDA runtime error by auto-selecting CPU when no GPU is available, keeping the same inference logic. Finally, I fix two small transform bugs that can silently break preprocessing (`is` vs `==` for hue, and `trim()` returning `None` when no bbox is found), and switch inference to `torch.no_grad()` to avoid unnecessary memory use while preserving outputs.'
- What this solution (achieved 0.01221) has done: 'I (1) make weight loading robust by searching common Kaggle input locations and falling back to an ImageNet-pretrained backbone if the custom `.pkl` weights are not present, so the notebook always runs end-to-end and writes `submission.csv`. I (2) fix the CUDA dtype/device mismatch by ensuring the entire model is moved to the selected device after loading weights (and verifying parameter device). I (3) make the inference loop resilient to missing/corrupt images (count and skip with a safe default) so the submission length always matches `test.csv`. These changes are execution/stability fixes; if the custom weights are available they preserve the original inference semantics and should allow score to move toward the target (since the intended weights are used).'
- What this solution (achieved 0.01221) has done: 'Your current score is far below the target because the pipeline is almost certainly running with the fallback ImageNet backbone (custom weights not found), which makes the regressor head essentially random for this task. To move the score toward the target with minimal changes, I (1) expand and prioritize weight-file discovery to reliably locate the provided `.pkl` within the competition dataset directories, (2) make the state-dict loading tolerant to minor key mismatches (while still preferring a strict load first) so it doesn’t silently fail into the fallback path, and (3) print a clear warning and stop using the fallback unless weights truly cannot be found. These changes preserve your model and inference logic; they primarily ensure you actually use the intended trained weights, which should raise QWK substantially toward the target.'
- What this solution (achieved 0.0) has done: 'Your score is extremely far below the target, which strongly suggests the intended trained weights are still not being loaded, so the model is effectively untrained for this task. I make the smallest change that’s most likely to move QWK upward: expand weight-file discovery to search for the exact expected filename anywhere under `/kaggle/input` and also accept common checkpoint extensions, and make the loader handle checkpoints that store the model under common keys like `model`, `model_state_dict`, or `net`. I also add a hard fail if no custom weights are found (instead of silently falling back to ImageNet), because the fallback keep you near-random and cannot move you toward 0.91. These changes preserve your model architecture and inference semantics; they only ensure the correct weights are actually used and thus should move the score sharply toward the target.'
- What this solution (achieved 0.01787) has done: 'You’re failing before inference because the script hard-errors when it can’t find the custom checkpoint; in this environment there is no `../input/weights` dataset mounted, so weight discovery returns `None`. I keep your model/inference logic unchanged, but make the run end-to-end by enabling a controlled fallback (ImageNet pretrained backbone) when weights truly aren’t available, instead of crashing. To avoid a silent “random head” as much as possible without changing architecture/training, I also switch to using the `final=True` head (which exists in your model) during inference and then apply the same `regress2class` thresholds, which is a minimal semantic correction if the intended checkpoint was trained for the final regressor. The code always write a valid `submission.csv` with correct columns and length.'
- What this solution (achieved 0.0) has done: 'Your score is far below the target, and the printouts/logic strongly suggest you’re running with the ImageNet-backbone fallback because the intended custom checkpoint is not being found/loaded. The most direct minimal change to move QWK upward toward the target is to (1) make weight discovery actually locate the checkpoint inside the available dataset tree and (2) support more common checkpoint containers (e.g., nested keys like `ema_state_dict`, `checkpoint`, etc.) without changing the model or inference logic. I keep your model, transforms, thresholds, and `final=True` inference unchanged, but I hard-require custom weights (raise with a clear error) because the fallback cannot plausibly approach 0.91. These changes are small, execution-safe, and directly aimed at using the intended trained weights so the score moves toward the target.'
- What this solution (achieved 0.01787) has done: 'I fix the immediate runtime failure by making the checkpoint requirement conditional: if the custom weights truly don’t exist in this Kaggle environment, the code fall back to a pretrained backbone (same model, same inference path) so it can run end-to-end and always write `submission.csv`. I also broaden weight discovery slightly (search any `.pkl/.pth/.pt/.bin` under `/kaggle/input` that contains key substrings like `3stage`, `b4`, `clahe`) to maximize the chance of finding the intended checkpoint if it is present under an unexpected name. These changes are minimal and localized to the weight-loading block; the model architecture, transforms, thresholds, and inference semantics remain the same. This should move the score upward if the weights can be found/loaded; otherwise it at least produce a valid submission instead of crashing.'
- What this solution (achieved 0.0) has done: 'Your current score is far below the target, which strongly suggests you are still not loading the intended trained checkpoint and are instead running the ImageNet-backbone fallback (random head → near-random QWK). I make a minimal, execution-safe change to *prefer and require* the custom weights by searching more reliably (including common “weights” filenames without relying on keywords) and by validating that the loaded state_dict actually matches your model’s keys at a high ratio—otherwise we fail fast instead of silently producing a bad submission. This preserves your model architecture and inference logic; it only changes checkpoint discovery/validation so the run is much more likely to use the correct trained weights and move QWK toward your target. The script still produce `submission.csv` when weights are found/loaded; if not, it error clearly rather than generating a low-scoring submission.'

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
BASE = "../input/aptos2019-blindness-detection"
ALT_BASE = "/kaggle/input/aptos2019-blindness-detection"

if not os.path.exists(BASE) and os.path.exists(ALT_BASE):
    BASE = ALT_BASE

TEST_CSV = f"{BASE}/test.csv"
TEST_IMG_DIR = f"{BASE}/test_images"

WEIGHT_PATH = "../input/weights/B4_3stage_35epoch_CLAHE.pkl"

test_ids = pd.read_csv(TEST_CSV)["id_code"].values

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


def _find_weight_file():
    target_base = os.path.splitext(os.path.basename(WEIGHT_PATH))[0]
    allowed_exts = [".pkl", ".pth", ".pt", ".bin"]
    allowed_names = [target_base + ext for ext in allowed_exts]

    candidates = [WEIGHT_PATH]
    common_dirs = [
        "../input/weights",
        "/kaggle/input/weights",
        "../input",
        "/kaggle/input",
        BASE,
        os.path.dirname(BASE),
        os.path.join(os.path.dirname(BASE), "aptos2019-blindness-detection"),
        os.path.join(BASE, "aptos2019-blindness-detection"),
    ]
    for d in common_dirs:
        for nm in allowed_names:
            candidates.append(os.path.join(d, nm))

    for c in candidates:
        if c and os.path.exists(c):
            return c

    search_roots = [
        "/kaggle/input",
        "../input",
        "/kaggle/data",
        BASE,
        os.path.dirname(BASE),
    ]
    exts = set(allowed_exts)
    max_depth = 8

    best = None
    best_score = -1

    def score_name(fn: str) -> int:
        low = fn.lower()
        s = 0
        for tok in ["b4", "3stage", "3_stage", "clahe", "35epoch", "epoch"]:
            if tok in low:
                s += 3
        if target_base.lower() in low:
            s += 10
        for tok in ["weight", "weights", "ckpt", "checkpoint", "model"]:
            if tok in low:
                s += 2
        return s

    for root in search_roots:
        if not root or not os.path.exists(root):
            continue
        root = os.path.abspath(root)
        for dirpath, dirnames, filenames in os.walk(root):
            rel = os.path.relpath(dirpath, root)
            depth = 0 if rel == "." else rel.count(os.sep) + 1
            if depth > max_depth:
                dirnames[:] = []
                continue

            for fn in filenames:
                if os.path.splitext(fn)[1].lower() not in exts:
                    continue
                s = score_name(fn)
                if s > best_score:
                    best_score = s
                    best = os.path.join(dirpath, fn)

    if best is not None and best_score >= 5:
        return best

    return None


def _extract_state_dict(state):
    if not isinstance(state, dict):
        raise ValueError("Unsupported checkpoint format (expected a dict-like object).")

    candidate_keys = [
        "state_dict",
        "model_state_dict",
        "model",
        "net",
        "weights",
        "checkpoint",
        "ckpt",
        "ema_state_dict",
        "ema",
    ]

    def _looks_like_state_dict(d):
        return (
            isinstance(d, dict)
            and len(d) > 0
            and all(hasattr(v, "shape") or torch.is_tensor(v) for v in d.values())
        )

    for key in candidate_keys:
        if key in state:
            inner = state[key]
            if _looks_like_state_dict(inner):
                return inner
            if isinstance(inner, dict):
                for key2 in candidate_keys:
                    if key2 in inner and _looks_like_state_dict(inner[key2]):
                        return inner[key2]

    if _looks_like_state_dict(state):
        return state

    raise ValueError("Unsupported checkpoint format (could not locate a state_dict).")


def _load_state_dict_robust(model: nn.Module, state):
    state = _extract_state_dict(state)

    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}

    model_keys = set(model.state_dict().keys())
    state_keys = set(state.keys())
    overlap = len(model_keys & state_keys) / max(1, len(model_keys))

    if overlap < 0.85:
        raise RuntimeError(
            f"Checkpoint appears incompatible with model (key overlap {overlap:.2%})."
        )

    try:
        model.load_state_dict(state, strict=True)
        return True, "strict"
    except RuntimeError:
        missing, unexpected = model.load_state_dict(state, strict=False)
        loaded_keys = len(model_keys & state_keys) - len(unexpected)
        if loaded_keys < 0.85 * len(model_keys):
            raise RuntimeError(
                f"Too few keys loaded ({loaded_keys}/{len(model_keys)}); missing={len(missing)} unexpected={len(unexpected)}"
            )
        return (
            True,
            f"non-strict (missing={len(missing)}, unexpected={len(unexpected)})",
        )


weight_file = _find_weight_file()
loaded_custom_weights, load_mode = False, "none"

if weight_file is None:
    raise FileNotFoundError(
        "Could not locate the trained checkpoint. "
        "Please add/mount the weights dataset so the intended model can be loaded."
    )

state = torch.load(weight_file, map_location="cpu")
loaded_custom_weights, load_mode = _load_state_dict_robust(net, state)

net = net.to(device)
net.eval()

_param_dev = next(net.parameters()).device
print("Custom weights loaded:", loaded_custom_weights, "| load mode:", load_mode)
print("Loaded weights from:", weight_file)
print("Using device:", device, "| model param device:", _param_dev)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_56/161652918.py in <cell line: 0>()
    190 # which yields near-random predictions for this task and cannot move toward the target.
    191 if weight_file is None:
--> 192     raise FileNotFoundError(
    193         "Could not locate the trained checkpoint. "
    194         "Please add/mount the weights dataset so the intended model can be loaded."

FileNotFoundError: Could not locate the trained checkpoint. Please add/mount the weights dataset so the intended model can be loaded.

## === cell 5
submission = []
missing_or_failed = 0

with torch.no_grad():
    for idx in test_ids:
        image_name = f"{TEST_IMG_DIR}/{idx}.png"
        try:
            img = Image.open(image_name).convert("RGB")
            img = transform(img).unsqueeze(0).to(device)

            out = net(img, final=True)
            pred = regress2class(out.data.squeeze(1))
            submission.append([idx, int(pred.item())])
        except Exception:
            missing_or_failed += 1
            submission.append([idx, 0])

submission = np.array(submission, dtype=object)
print("Inference done. Missing/failed images:", missing_or_failed)



## === cell 6
df = pd.DataFrame(submission, columns=["id_code", "diagnosis"])

df["id_code"] = df["id_code"].astype(str)
df["diagnosis"] = df["diagnosis"].astype(int).clip(0, 4)

assert (
    len(df) == len(test_ids) and len(df) > 0
), "Submission DataFrame is empty or misaligned with test_ids"
assert list(df.columns) == ["id_code", "diagnosis"]

df.to_csv("submission.csv", index=False)
print(df.head())
print("Wrote submission.csv with shape:", df.shape)
