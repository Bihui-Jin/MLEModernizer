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

0.9007424399066876

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'The main runtime failures come from (1) hardcoding `cuda:0` in an environment without an NVIDIA driver, and (2) trying to load a weights file that doesn’t exist in the provided `/kaggle/input` tree. I make device selection robust (CPU fallback) and make weight loading conditional: if weights are missing, the script still run end-to-end with random weights (score be poor, but it produce a valid non-empty `submission.csv`). I also fix a couple of small logic/device bugs in the helper functions (incorrect `.cpu()` placement and unsafe `.data` usage) and ensure the custom `trim()` transform always returns an image to avoid `None` propagating into the pipeline. Finally, I keep I/O paths aligned with Kaggle’s `/kaggle/input/...` structure and write `submission.csv` with the required columns.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is coming from running inference with random weights (the script can’t find the pretrained checkpoint), so the smallest change that moves you toward the target is to reliably locate and load the provided weights from any of the common `/kaggle/input/**` locations. I keep your model and transforms identical, but expand the checkpoint search (including the dataset directory and common filename variants) and make the loader handle common wrapper formats (`state_dict`, `model`, `net`, `module.` prefixes). This should produce meaningful predictions and increase QWK toward your target while still writing a valid `submission.csv` end-to-end. If no weights are found, it still fall back to random weights (so it never fails to produce a CSV).'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is almost certainly because the checkpoint still isn’t being found/loaded (so you’re predicting with random weights), even though the script runs and writes a CSV. I make checkpoint discovery robust by (1) searching more broadly under `/kaggle/input` with a prioritized filename match, and (2) handling common checkpoint wrappers plus the frequent EfficientNet prefix mismatch (`backbone.` vs `module.backbone.` etc.) without changing the model. I also move the model to `device` before loading to avoid any subtle dtype/device inconsistencies, and I add a tiny sanity print to confirm whether weights were actually applied. These are minimal, inference-only changes intended to move QWK upward toward your 0.9007 target by ensuring you use the intended pretrained weights when present.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score indicates the model is still effectively untrained at inference time (either no checkpoint is loaded, or the loaded weights barely match the model keys), so the smallest score-moving change is to make checkpoint discovery/load actually succeed. I keep your exact model/transform/inference logic, but (1) broaden checkpoint search to include `/kaggle/working` and common filenames, and (2) make state-dict extraction/remapping handle more real-world wrappers and prefix patterns. I also add a hard check: if the overlap of loaded keys is too low, we warn loudly (still produce a CSV), so you can see when you’re submitting random-ish predictions. These are inference-only, minimal changes expected to move QWK upward toward your 0.9007 target by ensuring the intended pretrained weights are applied.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is consistent with effectively random predictions, which in this script happens when the pretrained checkpoint is not actually found/loaded or when the inference head used doesn’t match the checkpoint (e.g., checkpoint trained with `final=True` but you call `net(img)` and ignore the final regressor). I make checkpoint discovery also search under `/kaggle/input/**/` for common filenames (including zipped/alternative names) and strengthen the state-dict unwrapping/remapping so the weight overlap is high. Then, without changing the model or transforms, I make inference automatically use `final=True` only when the loaded checkpoint contains `final_regressor` weights (otherwise keep your current 3-head path). This is a minimal, inference-only change that should move QWK upward toward your target by ensuring you’re using the intended pretrained weights and the correct forward path.'
- What this solution (achieved 0.01221) has done: 'Your 0.0 score is consistent with a “valid CSV but essentially random predictions”, which most often happens here because no real checkpoint is being loaded (APTOS dataset doesn’t ship your custom `B4_3stage_34epoch_CLAHE.*` weights). The smallest legitimate change that should move QWK upward toward your 0.9007 target is to use a pretrained backbone (ImageNet) for the exact same `tf_efficientnet_b4_ns` model when no external checkpoint is found, while keeping your transforms, thresholds, and inference logic unchanged. I also make the checkpoint search a bit more robust for common Kaggle dataset layouts, but still fall back safely so a submission is always produced. This keeps “core logic” intact (same model class/forward path) while avoiding random-weight inference.'
- What this solution (achieved 0.06105) has done: 'Your current score is far below the target, and the most likely cause is that you’re still doing inference with essentially untrained weights (no real checkpoint exists in the provided inputs, and ImageNet-pretrained backbone alone is not sufficient with your current head), so predictions are close to random. The smallest score-moving change that preserves your core model/forward/inference semantics is to add a lightweight validation-based calibration step: fit the 4 regression-to-class thresholds on a held-out split of the provided `train.csv` using your existing model outputs, then reuse those thresholds for test prediction. This doesn’t change the architecture, loss, or training loops (there are none), but it aligns the final discretization to maximize QWK given the model’s current regressor behavior, which should move the score materially upward toward the target. I also ensure the pipeline stays CPU-safe, keeps the same transforms, and still always writes a valid `submission.csv`.'

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



## === cell 1
threshold = [0.75, 1.5, 2.5, 3.5]


def regress2class(out: torch.Tensor) -> torch.Tensor:
    """
    out: shape (B,) float in [0, 4.5]
    returns: shape (B,) float classes (0..4) on CPU
    """
    out = out.detach()
    pred = torch.zeros(out.size(0), device=out.device, dtype=torch.float32)
    for t in threshold:
        pred += (out >= t).to(torch.float32)
    return pred.cpu()


def ordinal2class_prob(out: torch.Tensor) -> torch.Tensor:
    out = out.detach()
    pred_prob = torch.zeros(out.size(0), 5, device=out.device, dtype=torch.float32)
    pred_prob[:, 0] = (1 - out[:, 0]).squeeze()
    pred_prob[:, 1] = (out[:, 0] * (1 - out[:, 1])).squeeze()
    pred_prob[:, 2] = (out[:, 1] * (1 - out[:, 2])).squeeze()
    pred_prob[:, 3] = (out[:, 2] * (1 - out[:, 3])).squeeze()
    pred_prob[:, 4] = out[:, 3].squeeze()
    return F.softmax(pred_prob, dim=1)


def regress2class_prob(out: torch.Tensor) -> torch.Tensor:
    out = out.detach().view(-1)
    pred_prob = torch.zeros((out.size(0), 5), device=out.device, dtype=torch.float32)
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
        self.backbone = timm.models.tf_efficientnet_b5_ns(pretrained=True)
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

        self.backbone = timm.models.tf_efficientnet_b4_ns(pretrained=True)
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
DATA_ROOT = "/kaggle/input/aptos2019-blindness-detection"
TEST_CSV = os.path.join(DATA_ROOT, "test.csv")
TEST_IMG_DIR = os.path.join(DATA_ROOT, "test_images")
TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train_images")

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

net = ThreeStage_Model().to(device)
net.eval()


def _find_existing_checkpoint():
    explicit_candidates = [
        "/kaggle/input/weights/B4_3stage_34epoch_CLAHE.pkl",
        "/kaggle/input/weights/B4_3stage_34epoch_CLAHE.pth",
        "/kaggle/input/weights/B4_3stage_34epoch_CLAHE.pt",
        os.path.join(DATA_ROOT, "B4_3stage_34epoch_CLAHE.pkl"),
        os.path.join(DATA_ROOT, "B4_3stage_34epoch_CLAHE.pth"),
        os.path.join(DATA_ROOT, "B4_3stage_34epoch_CLAHE.pt"),
        "/kaggle/input/aptos2019-blindness-detection/B4_3stage_34epoch_CLAHE.pkl",
        "/kaggle/input/aptos2019-blindness-detection/B4_3stage_34epoch_CLAHE.pth",
        "/kaggle/input/aptos2019-blindness-detection/B4_3stage_34epoch_CLAHE.pt",
        "/kaggle/working/B4_3stage_34epoch_CLAHE.pkl",
        "/kaggle/working/B4_3stage_34epoch_CLAHE.pth",
        "/kaggle/working/B4_3stage_34epoch_CLAHE.pt",
    ]
    for p in explicit_candidates:
        if os.path.exists(p):
            return p

    preferred_substrings = [
        "b4_3stage_34epoch_clahe",
        "b4-3stage-34epoch-clahe",
        "b4_3stage_clahe",
        "b4_3stage",
        "three_stage",
        "threestage",
        "efficientnet_b4",
        "effb4",
        "aptos",
        "blindness",
        "retina",
        "dr",
    ]

    exts = {".pth", ".pt", ".pkl", ".bin"}
    search_roots = ["/kaggle/input", "/kaggle/working"]

    best = None
    best_score = -1

    for base in search_roots:
        if not os.path.exists(base):
            continue
        for root, _, files in os.walk(base):
            for fn in files:
                low = fn.lower()
                ext = os.path.splitext(low)[1]
                if ext not in exts:
                    continue

                score = 0
                if any(s in low for s in preferred_substrings):
                    score += 6
                if "b4" in low:
                    score += 3
                if "3stage" in low or "three" in low:
                    score += 2
                if "clahe" in low:
                    score += 2
                if "final" in low:
                    score += 1

                if score < 7:
                    continue

                p = os.path.join(root, fn)
                if score > best_score:
                    best = p
                    best_score = score

    return best


def _extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for _ in range(6):
            moved = False
            for key in [
                "state_dict",
                "model_state_dict",
                "model",
                "net",
                "network",
                "weights",
                "params",
                "ema_state_dict",
                "ema",
                "student",
            ]:
                if key in ckpt and isinstance(ckpt[key], dict):
                    ckpt = ckpt[key]
                    moved = True
                    break
            if not moved:
                break

    if isinstance(ckpt, dict) and any(
        isinstance(k, str) and k.startswith("module.") for k in ckpt.keys()
    ):
        ckpt = {k.replace("module.", "", 1): v for k, v in ckpt.items()}

    return ckpt


def _remap_prefixes_if_needed(state_dict, model):
    if not isinstance(state_dict, dict) or len(state_dict) == 0:
        return state_dict

    model_keys = set(model.state_dict().keys())
    sd_keys = set(state_dict.keys())

    if len(model_keys.intersection(sd_keys)) > 50:
        return state_dict

    candidates = [state_dict]

    for pref in ["model.", "net.", "network.", "encoder."]:
        if any(k.startswith(pref) for k in sd_keys):
            candidates.append(
                {k[len(pref) :]: v for k, v in state_dict.items() if k.startswith(pref)}
            )

    if any(k.startswith("backbone.") for k in model_keys) and not any(
        k.startswith("backbone.") for k in sd_keys
    ):
        candidates.append({("backbone." + k): v for k, v in state_dict.items()})
    if any(k.startswith("backbone.") for k in sd_keys) and not any(
        k.startswith("backbone.") for k in model_keys
    ):
        candidates.append(
            {
                k[len("backbone.") :]: v
                for k, v in state_dict.items()
                if k.startswith("backbone.")
            }
        )

    if any(k.startswith("model.backbone.") for k in sd_keys):
        candidates.append(
            {k.replace("model.", "", 1): v for k, v in state_dict.items()}
        )

    if any(k.startswith("model.") for k in sd_keys):
        candidates.append(
            {
                k[len("model.") :]: v
                for k, v in state_dict.items()
                if k.startswith("model.")
            }
        )

    best_sd = state_dict
    best_overlap = len(model_keys.intersection(sd_keys))
    for cand in candidates:
        overlap = len(model_keys.intersection(set(cand.keys())))
        if overlap > best_overlap:
            best_overlap = overlap
            best_sd = cand

    return best_sd


weights_path = _find_existing_checkpoint()
use_final_forward = False  # will be set based on checkpoint contents (score-relevant)

if weights_path is None:
    print(
        "WARNING: No custom pretrained weights found under /kaggle/input or /kaggle/working."
    )
    print(
        "Proceeding with ImageNet-pretrained tf_efficientnet_b4_ns backbone (default)."
    )
else:
    print("Loading weights:", weights_path)
    ckpt = torch.load(weights_path, map_location="cpu")
    state = _extract_state_dict(ckpt)
    state = _remap_prefixes_if_needed(state, net)

    missing, unexpected = net.load_state_dict(state, strict=False)

    overlap = (
        len(set(net.state_dict().keys()).intersection(set(state.keys())))
        if isinstance(state, dict)
        else 0
    )
    total = len(net.state_dict().keys())
    print(
        f"load_state_dict: missing={len(missing)} unexpected={len(unexpected)} overlap={overlap}/{total}"
    )

    if isinstance(state, dict) and any(
        k.startswith("final_regressor.") for k in state.keys()
    ):
        use_final_forward = True
    print(
        (
            "Inference will use final=True:"
            if use_final_forward
            else "Inference will use final=False:"
        ),
        use_final_forward,
    )

    if overlap < 50:
        print(
            "WARNING: Very low checkpoint key overlap; predictions may be near-random and score will be poor."
        )

net = net.to(device)
net.eval()




## === cell 5
def _predict_regression_value(img_path: str) -> float:
    img = Image.open(img_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        if use_final_forward:
            r = net(x, final=True)  # (1,1)
        else:
            _, r, _ = net(x)  # (1,1)
    return float(r.squeeze().detach().cpu().item())


def _apply_thresholds(values: np.ndarray, thr: list) -> np.ndarray:
    out = np.zeros(values.shape[0], dtype=np.int64)
    for t in thr:
        out += (values >= t).astype(np.int64)
    return out


def _fit_thresholds_qwk(y_true: np.ndarray, y_pred_cont: np.ndarray, init_thr: list):
    thr = np.array(init_thr, dtype=np.float64)

    def score(thr_vec):
        y_pred = _apply_thresholds(y_pred_cont, thr_vec.tolist())
        return cohen_kappa_score(y_true, y_pred, weights="quadratic")

    best = score(thr)

    bounds = [(0.0, 4.5), (0.0, 4.5), (0.0, 4.5), (0.0, 4.5)]
    steps = [0.25, 0.10, 0.05]

    for step in steps:
        improved = True
        it = 0
        while improved and it < 30:
            improved = False
            it += 1
            for i in range(4):
                candidates = []
                for delta in [-2 * step, -step, 0.0, step, 2 * step]:
                    cand = thr.copy()
                    cand[i] = cand[i] + delta
                    cand[i] = min(max(cand[i], bounds[i][0]), bounds[i][1])
                    cand = np.sort(cand)  # keep ordered
                    candidates.append(cand)
                for cand in candidates:
                    sc = score(cand)
                    if sc > best + 1e-9:
                        best = sc
                        thr = cand
                        improved = True

    return thr.tolist(), best


train_df = pd.read_csv(TRAIN_CSV)

perm = np.random.RandomState(42).permutation(len(train_df))
val_size = min(512, max(256, int(0.2 * len(train_df))))
val_idx = perm[:val_size]
val_df = train_df.iloc[val_idx].reset_index(drop=True)

print(f"Calibrating thresholds on {len(val_df)} validation images...")

val_pred_cont = np.zeros(len(val_df), dtype=np.float32)
val_true = val_df["diagnosis"].values.astype(np.int64)

t0 = time.time()
for i, row in val_df.iterrows():
    img_path = os.path.join(TRAIN_IMG_DIR, f"{row['id_code']}.png")
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Missing train image for calibration: {img_path}")
    val_pred_cont[i] = _predict_regression_value(img_path)
    if (i + 1) % 64 == 0:
        print(f"  {i+1}/{len(val_df)} done")

print(f"Calibration inference time: {time.time()-t0:.1f}s")

orig_qwk = cohen_kappa_score(
    val_true, _apply_thresholds(val_pred_cont, threshold), weights="quadratic"
)
print("Original thresholds:", threshold, "val QWK:", orig_qwk)

fitted_thr, fitted_qwk = _fit_thresholds_qwk(val_true, val_pred_cont, threshold)
print("Fitted thresholds:", fitted_thr, "val QWK:", fitted_qwk)

threshold = fitted_thr



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_56/3401098722.py in <cell line: 0>()
     71     if not os.path.exists(img_path):
     72         raise FileNotFoundError(f"Missing train image for calibration: {img_path}")
---> 73     val_pred_cont[i] = _predict_regression_value(img_path)
     74     if (i + 1) % 64 == 0:
     75         print(f"  {i+1}/{len(val_df)} done")

/tmp/ipykernel_56/3401098722.py in _predict_regression_value(img_path)
      6             r = net(x, final=True)  # (1,1)
      7         else:
----> 8             _, r, _ = net(x)  # (1,1)
      9     return float(r.squeeze().detach().cpu().item())
     10 

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/tmp/ipykernel_56/1153186077.py in forward(self, x, final)
     87         x = self.backbone(x)
     88 
---> 89         c_out = self.classifier(x)
     90         r_out = self.regressor(x)
     91         o_out = self.ordinal(x)

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/container.py in forward(self, input)
    248     def forward(self, input):
    249         for module in self:
--> 250             input = module(input)
    251         return input
    252 

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/linear.py in forward(self, input)
    123 
    124     def forward(self, input: Tensor) -> Tensor:
--> 125         return F.linear(input, self.weight, self.bias)
    126 
    127     def extra_repr(self) -> str:

RuntimeError: mat1 and mat2 shapes cannot be multiplied (1x1000 and 1792x500)

## === cell 6
submission = []
with torch.no_grad():
    for i, idx in enumerate(test_ids):
        image_name = os.path.join(TEST_IMG_DIR, f"{idx}.png")
        if not os.path.exists(image_name):
            raise FileNotFoundError(f"Missing test image: {image_name}")

        img = Image.open(image_name).convert("RGB")
        img = transform(img).unsqueeze(0).to(device)

        if use_final_forward:
            r_out = net(img, final=True)  # shape (1,1)
        else:
            _, r_out, _ = net(img)  # shape (1,1)

        pred = regress2class(r_out.squeeze(1))
        submission.append([idx, int(pred.item())])

submission = np.array(submission, dtype=object)
print("Predictions:", submission.shape)



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_56/853105729.py in <cell line: 0>()
     12             r_out = net(img, final=True)  # shape (1,1)
     13         else:
---> 14             _, r_out, _ = net(img)  # shape (1,1)
     15 
     16         pred = regress2class(r_out.squeeze(1))

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/tmp/ipykernel_56/1153186077.py in forward(self, x, final)
     87         x = self.backbone(x)
     88 
---> 89         c_out = self.classifier(x)
     90         r_out = self.regressor(x)
     91         o_out = self.ordinal(x)

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/container.py in forward(self, input)
    248     def forward(self, input):
    249         for module in self:
--> 250             input = module(input)
    251         return input
    252 

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/linear.py in forward(self, input)
    123 
    124     def forward(self, input: Tensor) -> Tensor:
--> 125         return F.linear(input, self.weight, self.bias)
    126 
    127     def extra_repr(self) -> str:

RuntimeError: mat1 and mat2 shapes cannot be multiplied (1x1000 and 1792x500)

## === cell 7
df = pd.DataFrame(submission, columns=["id_code", "diagnosis"])

df["id_code"] = df["id_code"].astype(str)
df["diagnosis"] = df["diagnosis"].astype(int)

assert (
    len(df) == len(test_ids) and len(df) > 0
), "Submission DataFrame is empty or wrong length."
df.to_csv("submission.csv", index=False)
print(df.head())
print("Wrote submission.csv with", len(df), "rows")

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
AssertionError                            Traceback (most recent call last)
/tmp/ipykernel_56/3177915982.py in <cell line: 0>()
      5 
      6 assert (
----> 7     len(df) == len(test_ids) and len(df) > 0
      8 ), "Submission DataFrame is empty or wrong length."
      9 df.to_csv("submission.csv", index=False)

AssertionError: Submission DataFrame is empty or wrong length.
