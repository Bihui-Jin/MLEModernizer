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

0.9134323743353088

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the missing-weights crash by making weight loading optional and clearly warning when the file is not present, so the notebook can still run end-to-end and write a submission.csv. I also fix the CUDA crash by selecting `cuda` only when it’s actually available, otherwise falling back to CPU. To prevent an empty submission, I ensure the test image directory is resolved robustly across the provided `/kaggle/input/...` layout and I handle edge cases where transforms return `None` (from `trim`). These changes keep your model and prediction logic intact while making the pipeline runnable and producing a valid `.csv` submission.'
- What this solution (achieved 0.0) has done: 'I fix the core runtime error by making the model’s feature dimension match the EfficientNet backbone output: `timm` returns 1000-d logits by default unless `num_classes=0` is set, which caused the 1000 vs 1792 matrix multiply crash. I keep the architecture and inference semantics the same, only changing backbone construction to output pooled features. Then I make submission writing robust even if inference fails partway by building the submission directly from `test_ids` and filling missing predictions, which eliminates the indexing KeyError and guarantees a valid `submission.csv`. These changes are necessary for end-to-end execution and should improve score versus a broken/empty submission.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with a miscalibrated discretization from regression-to-class (fixed thresholds) rather than a broken submission file, since your pipeline now writes a valid CSV. To move the score up toward the 0.913 target without changing the model/training, I keep the exact network/inference but tune the four cutpoints on the training set using out-of-fold predictions to directly maximize quadratic weighted kappa. This is a minimal semantic change (only post-processing) and is standard for this competition’s metric. I also make the threshold search deterministic and fast (small coordinate-descent over a bounded range), and then apply the learned thresholds to the test predictions before writing `submission.csv`.'
- What this solution (achieved 0.0) has done: 'You’re getting a `NameError` because `TestDataset` is referenced in cell 5 before it is defined (it’s defined later in cell 6). I fix this with a minimal change: define `TestDataset` earlier (in cell 5, before `infer_regression`) and then reuse it in cell 6 without changing the model, training/inference loop, or threshold-fitting logic. This make the OOF inference run, enable threshold calibration (which should lift the score from 0.0 toward the target), and still produce a valid `submission.csv`. I also keep all paths and I/O intact and ensure the submission columns match the required format.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with a submission that’s valid CSV-wise but has predictions effectively uninformative (e.g., untrained/random weights because finetuned weights weren’t found), plus threshold calibration being both extremely slow and potentially unstable because it runs full OOF inference on all 3295 train images. To move the score up toward the 0.913 target with minimal semantic change, I keep your model/inference exactly the same but (1) make weight discovery succeed for common Kaggle layouts (including `.pth`/`.pt` in the dataset folder), and (2) if no finetuned weights are found, fail fast instead of producing a misleading 0-score submission. I also cap the threshold calibration runtime by using a single deterministic stratified holdout for calibration (still legitimate post-processing) so the notebook finishes within 600s; this should improve discretization without changing the network. Finally, I ensure the submission uses the sample_submission row order and exact required columns to avoid any accidental misalignment.'
- What this solution (achieved 0.0) has done: 'I (1) remove the hard crash when finetuned weights are missing so the notebook can run end-to-end and still create a valid `submission.csv`, and (2) fix the CUDA dtype/device mismatch by ensuring the model and all its parameters are moved onto the selected `device` *before* any inference (the current error shows model weights stayed on CPU while inputs were on GPU). I also make inference robust by forcing `net.eval()` right before predicting and adding a small safety fallback so threshold calibration won’t crash the whole run if weights are unavailable (it then use the default thresholds). These are minimal changes that preserve your model/loops and should move the score up from 0.0 (broken inference) toward the target by producing non-trivial predictions.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with the notebook producing a *valid* CSV but with essentially untrained/random predictions because the finetuned weights are not actually being found/loaded, so I focus on making weight discovery/loading succeed without changing your model or inference logic. I also make the load step tolerant of common checkpoint formats (`model`, `net`, nested dicts) and add an explicit hard failure if no finetuned weights are available (since otherwise you reliably submit garbage and score ~0). Finally, I keep your threshold calibration exactly as-is but ensure it only runs when finetuned weights truly loaded, so the discretization is meaningful and moves QWK up toward the target.'
- What this solution (achieved 0.0) has done: 'I remove the hard failure when finetuned weights are missing so the notebook can run end-to-end and produce a valid `submission.csv` (your current code stops in cell 4). I also fix the GPU/CPU mismatch that crashes inference by ensuring the model is moved to `device` immediately after creation and again after loading weights (some checkpoint loads can leave parameters on CPU). Finally, I keep your calibration and discretization logic intact, but make it conditional: threshold fitting run only when finetuned weights successfully loaded; otherwise it fall back to the default thresholds so the pipeline still completes and writes a submission.'

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
import cv2

from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedShuffleSplit
import timm

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)



## === cell 1
threshold = [0.75, 1.5, 2.5, 3.5]


def regress2class(out: torch.Tensor) -> torch.Tensor:
    out = out.view(-1)
    pred = torch.zeros(out.size(0), device=out.device, dtype=torch.long)
    for t in threshold:
        pred += (out >= t).to(torch.long)
    return pred


def regress2class_with_thresholds(out: torch.Tensor, thr) -> torch.Tensor:
    out = out.view(-1)
    pred = torch.zeros(out.size(0), device=out.device, dtype=torch.long)
    for t in thr:
        pred += (out >= t).to(torch.long)
    return pred


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
            l1 = int(math.floor(out[i]))
            l2 = int(math.ceil(out[i]))
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
    def __init__(self, pretrained_backbone=False):
        super(Regressor, self).__init__()
        self.backbone = timm.models.tf_efficientnet_b5_ns(
            pretrained=pretrained_backbone, num_classes=0
        )
        self.backbone.global_pool = GeM(flatten=True)

        in_features = getattr(self.backbone, "num_features", 1000)
        self.regressor = nn.Linear(in_features, 1)

    def forward(self, x):
        x = self.backbone(x)
        out = self.regressor(x)
        out = torch.sigmoid(out) * 4.5
        return out


class ThreeStage_Model(nn.Module):
    def __init__(self, backbone=None, pretrained_backbone=False):
        super(ThreeStage_Model, self).__init__()

        self.backbone = timm.models.tf_efficientnet_b4_ns(
            pretrained=pretrained_backbone, num_classes=0
        )
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


class trim_safe(object):
    def __call__(self, image):
        try:
            out = trim()(image)
            if out is None or out.size[0] < 2 or out.size[1] < 2:
                return image
            return out
        except Exception:
            return image




## === cell 4
CANDIDATE_BASES = [
    "/kaggle/input/aptos2019-blindness-detection",
    "/kaggle/data/aptos2019-blindness-detection",
    "../input/aptos2019-blindness-detection",
    "../data/aptos2019-blindness-detection",
]
BASE_INPUT = None
for p in CANDIDATE_BASES:
    if os.path.isdir(p) and os.path.isfile(os.path.join(p, "test.csv")):
        BASE_INPUT = p
        break
if BASE_INPUT is None:
    BASE_INPUT = "/kaggle/input/aptos2019-blindness-detection"
print("BASE_INPUT:", BASE_INPUT)

test_csv_path = os.path.join(BASE_INPUT, "test.csv")
test_df = pd.read_csv(test_csv_path)
test_ids = test_df["id_code"].astype(str).tolist()

train_csv_path = os.path.join(BASE_INPUT, "train.csv")
train_df = pd.read_csv(train_csv_path)
train_df["id_code"] = train_df["id_code"].astype(str)
train_df["diagnosis"] = train_df["diagnosis"].astype(int)

input_size = 380

_tmp_model = timm.create_model("tf_efficientnet_b4_ns", pretrained=False)
cfg = getattr(_tmp_model, "pretrained_cfg", {}) or {}
mean = cfg.get("mean", (0.485, 0.456, 0.406))
std = cfg.get("std", (0.229, 0.224, 0.225))
del _tmp_model

transform = transforms.Compose(
    [
        trim_safe(),
        cropTo4_3(),
        transforms.Resize((input_size * 3 // 4, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=list(mean), std=list(std)),
    ]
)


def _can_use_pretrained():
    try:
        torch.hub.set_dir(os.path.join("/kaggle", "working", "torchhub"))
    except Exception:
        pass
    try:
        _ = timm.create_model("tf_efficientnet_b4_ns", pretrained=True)
        return True
    except Exception as e:
        print(
            "Pretrained weights not available locally; using pretrained_backbone=False. Reason:",
            repr(e),
        )
        return False


use_pretrained = _can_use_pretrained()

net = ThreeStage_Model(pretrained_backbone=use_pretrained).to(device)


def _find_first_existing(paths):
    for p in paths:
        if p and os.path.isfile(p):
            return p
    return None


def _search_weights_by_name(search_roots, filenames):
    hits = []
    for root in search_roots:
        if not root or not os.path.isdir(root):
            continue
        for dirpath, _, files in os.walk(root):
            for fn in files:
                if fn in filenames:
                    hits.append(os.path.join(dirpath, fn))
    hits = sorted(set(hits))
    return hits


weight_filenames = {
    "B4_3stage_4epoch_finetune.pkl",
    "B4_3stage_4epoch_finetune.pt",
    "B4_3stage_4epoch_finetune.pth",
}

candidate_paths = [
    "../input/weights/B4_3stage_4epoch_finetune.pth",
    "../input/weights/B4_3stage_4epoch_finetune.pt",
    "../input/weights/B4_3stage_4epoch_finetune.pkl",
    "/kaggle/input/weights/B4_3stage_4epoch_finetune.pth",
    "/kaggle/input/weights/B4_3stage_4epoch_finetune.pt",
    "/kaggle/input/weights/B4_3stage_4epoch_finetune.pkl",
    os.path.join(BASE_INPUT, "B4_3stage_4epoch_finetune.pth"),
    os.path.join(BASE_INPUT, "B4_3stage_4epoch_finetune.pt"),
    os.path.join(BASE_INPUT, "B4_3stage_4epoch_finetune.pkl"),
    "/kaggle/working/B4_3stage_4epoch_finetune.pth",
    "/kaggle/working/B4_3stage_4epoch_finetune.pt",
    "/kaggle/working/B4_3stage_4epoch_finetune.pkl",
]

search_roots = [
    "/kaggle/input",
    "/kaggle/data",
    "/kaggle/working",
    "../input",
    "../data",
    BASE_INPUT,
]

found = _find_first_existing(candidate_paths)
if found is None:
    hits = _search_weights_by_name(search_roots, weight_filenames)
    found = hits[0] if len(hits) else None


def _coerce_state_dict(state):
    if isinstance(state, dict):
        for k in ["state_dict", "model_state_dict", "model", "net", "weights"]:
            if k in state and isinstance(state[k], dict):
                state = state[k]
                break

    sd = state
    if isinstance(sd, dict):
        keys = list(sd.keys())
        if len(keys) and all(
            isinstance(k, str) and k.startswith("module.") for k in keys
        ):
            sd = {k[len("module.") :]: v for k, v in sd.items()}
    return sd


loaded_finetuned = False
if found is not None:
    state = torch.load(found, map_location="cpu")
    sd = _coerce_state_dict(state)
    try:
        net.load_state_dict(sd, strict=True)
        loaded_finetuned = True
        print("Loaded weights (strict=True):", found)
    except Exception as e:
        missing, unexpected = net.load_state_dict(sd, strict=False)
        loaded_finetuned = True
        print("Loaded weights (strict=False):", found)
        print("Reason strict=True failed:", repr(e))
        print("missing:", len(missing), "unexpected:", len(unexpected))
else:
    print(
        "WARNING: No competition finetuned weights found. Running with backbone pretrained="
        f"{use_pretrained} (when locally available)."
    )

net = net.to(device)
net.eval()

print("loaded_finetuned:", loaded_finetuned)



## === cell 5
train_img_dir = os.path.join(BASE_INPUT, "train_images")
if not os.path.isdir(train_img_dir):
    for alt in [
        "/kaggle/input/train_images",
        "/kaggle/data/train_images",
        "../input/train_images",
        "../data/train_images",
        "/kaggle/input/aptos2019-blindness-detection/train_images",
        "/kaggle/data/aptos2019-blindness-detection/train_images",
    ]:
        if os.path.isdir(alt):
            train_img_dir = alt
            break
print("train_img_dir:", train_img_dir)


class TrainDataset(Dataset):
    def __init__(self, df, img_dir, transform):
        self.ids = df["id_code"].astype(str).tolist()
        self.y = df["diagnosis"].astype(int).tolist()
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        idx = self.ids[i]
        y = self.y[i]
        image_name = os.path.join(self.img_dir, f"{idx}.png")

        img = None
        if os.path.exists(image_name):
            bgr = cv2.imread(image_name, cv2.IMREAD_COLOR)
            if bgr is not None:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb)

        if img is None:
            img = Image.new("RGB", (input_size, input_size), (0, 0, 0))

        x = self.transform(img)
        return idx, x, int(y)


class TestDataset(Dataset):
    def __init__(self, ids, img_dir, transform):
        self.ids = [str(x) for x in ids]
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        idx = self.ids[i]
        image_name = os.path.join(self.img_dir, f"{idx}.png")

        img = None
        if os.path.exists(image_name):
            bgr = cv2.imread(image_name, cv2.IMREAD_COLOR)
            if bgr is not None:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb)

        if img is None:
            img = Image.new("RGB", (input_size, input_size), (0, 0, 0))

        x = self.transform(img)
        return idx, x


def infer_regression(ids_list, img_dir, batch_size=8):
    net.eval()
    ds = TestDataset(ids_list, img_dir, transform)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=(device == "cuda"),
    )
    preds = np.zeros((len(ids_list),), dtype=np.float32)
    with torch.no_grad():
        start = 0
        for _, x in dl:
            x = x.to(device, non_blocking=True)
            r_out = net(x, final=True).squeeze(1)  # [B]
            r_out = r_out.detach().float().cpu().numpy()
            preds[start : start + len(r_out)] = r_out
            start += len(r_out)
    return preds


def qwk(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


def fit_thresholds(y_true, y_cont, init_thr=None, n_iter=10):
    if init_thr is None:
        thr = np.array([0.75, 1.5, 2.5, 3.5], dtype=np.float32)
    else:
        thr = np.array(init_thr, dtype=np.float32)

    y_true = np.asarray(y_true, dtype=int)
    y_cont = np.asarray(y_cont, dtype=np.float32)

    def apply_thr(t):
        return np.digitize(y_cont, t, right=False).astype(int)

    best = qwk(y_true, apply_thr(thr))
    steps = [0.25, 0.1, 0.05]
    for step in steps:
        for _ in range(n_iter):
            improved = False
            for k in range(4):
                lo = 0.0 if k == 0 else float(thr[k - 1] + 1e-3)
                hi = 4.5 if k == 3 else float(thr[k + 1] - 1e-3)
                candidates = [thr[k] - step, thr[k], thr[k] + step]
                for cand in candidates:
                    if not (lo <= cand <= hi):
                        continue
                    t2 = thr.copy()
                    t2[k] = cand
                    sc = qwk(y_true, apply_thr(t2))
                    if sc > best + 1e-12:
                        best = sc
                        thr = t2
                        improved = True
            if not improved:
                break
    thr = np.sort(thr)
    return thr.tolist(), best


if loaded_finetuned:
    y_all = train_df["diagnosis"].values
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED)
    _, cal_idx = next(sss.split(train_df["id_code"].values, y_all))
    cal_ids = train_df.iloc[cal_idx]["id_code"].astype(str).tolist()
    cal_y = train_df.iloc[cal_idx]["diagnosis"].values.astype(int)

    cal_pred = infer_regression(cal_ids, train_img_dir, batch_size=8)
    learned_thr, cal_qwk = fit_thresholds(
        y_true=cal_y, y_cont=cal_pred, init_thr=threshold, n_iter=12
    )
    print("Initial thresholds:", threshold)
    print("Learned thresholds:", learned_thr)
    print("Holdout QWK (for calibration only):", cal_qwk)
    threshold = learned_thr
else:
    print("Skipping threshold calibration because finetuned weights were not loaded.")
    print("Using default thresholds:", threshold)



## === cell 6
test_img_dir = os.path.join(BASE_INPUT, "test_images")
if not os.path.isdir(test_img_dir):
    for alt in [
        "/kaggle/input/test_images",
        "/kaggle/data/test_images",
        "../input/test_images",
        "../data/test_images",
        "/kaggle/input/aptos2019-blindness-detection/test_images",
        "/kaggle/data/aptos2019-blindness-detection/test_images",
    ]:
        if os.path.isdir(alt):
            test_img_dir = alt
            break
print("test_img_dir:", test_img_dir)

net.eval()
ds = TestDataset(test_ids, test_img_dir, transform)
dl = DataLoader(
    ds, batch_size=8, shuffle=False, num_workers=2, pin_memory=(device == "cuda")
)

pred_map = {}
with torch.no_grad():
    for ids, x in dl:
        x = x.to(device, non_blocking=True)
        r_out = net(x, final=True)  # [B,1]
        pred = regress2class_with_thresholds(r_out.squeeze(1), threshold)
        for _id, _p in zip(ids, pred.detach().cpu().numpy().tolist()):
            pred_map[str(_id)] = int(_p)



## === cell 7
sample_path = os.path.join(BASE_INPUT, "sample_submission.csv")
sample_df = pd.read_csv(sample_path)
sample_df["id_code"] = sample_df["id_code"].astype(str)

sample_df["diagnosis"] = sample_df["id_code"].map(pred_map).fillna(0).astype(int)

out_path = "submission.csv"
sample_df.to_csv(out_path, index=False)
print("Wrote", out_path, "with shape", sample_df.shape)
print("Final thresholds used:", threshold)
print("Loaded finetuned weights:", loaded_finetuned)
print(sample_df.head())
