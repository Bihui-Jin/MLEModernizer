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

0.9029520543793256

# 6. Current score

-0.07538

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I (1) fix the missing checkpoint crash by making inference robust to “no weights available” so the notebook always completes and writes a non-empty `submission.csv`. I (2) fix the CUDA/CPU dtype mismatch by ensuring the loaded `state_dict` is applied before moving the model to the target device, and by moving any tensor outputs through consistent device handling. Finally, I (3) speed up and stabilize inference using a simple `Dataset`+`DataLoader` (no change to model/transform logic) so all test rows are predicted and the submission format matches `sample_submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is consistent with predicting almost all zeros, which happens here because you’re taking the regression head (`r_out`) and applying fixed thresholds that don’t match the model’s intended inference path. I keep the same model and checkpoint-loading logic, but switch inference to the model’s `final=True` forward path (the only place where the `final_regressor` is used), then convert its 0–4.5 regression output into 0–4 classes with the same thresholding function. I also make `regress2class` device/shape-safe so it can’t silently misbehave due to squeezing/broadcasting. These are minimal changes aimed at moving QWK up toward your target by using the trained ensemble head rather than a single sub-head.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is almost certainly because the checkpoint never loads (so you’re predicting from random weights), which makes QWK near/below 0. To move toward your target with minimal change, I keep the same model and inference path (`final=True`) but (1) broaden checkpoint discovery to also search common Kaggle locations (including any `.pkl/.pth/.pt` with the same stem), and (2) add a strict “must-have-weights” fallback to load *any* available EfficientNet-B4/B5 DR checkpoint in the environment if the exact filename isn’t present. This preserves your core architecture and thresholding, but makes it far more likely you actually use trained weights and get a non-zero score. The submission writing stays identical and still always produces `submission.csv`.'
- What this solution (achieved -0.00088) has done: 'I fix the two blockers preventing an end-to-end run: (1) the hard crash when no checkpoint is found, by loading *timm* pretrained weights as a safe fallback so you never run random weights, and (2) the CUDA/CPU dtype mismatch by ensuring the model is moved to the target device before inference (and also after any checkpoint load). These changes keep your architecture, transforms, and inference logic (`final=True` + the same thresholding) intact, while making the pipeline robust enough to always write a valid `submission.csv`. The pretrained fallback should also materially improve score from “not yielded / random” toward your target, without changing training/evaluation semantics beyond using non-random weights when your custom checkpoint is unavailable.'
- What this solution (achieved -0.00088) has done: 'Your current score (-0.00088) indicates the model outputs are essentially untrained for this task (timm ImageNet backbone + random heads), so the smallest effective change is to ensure we load an actual APTOS-trained checkpoint if one exists in the environment. I keep your model, transforms, and `final=True` inference path identical, but improve checkpoint discovery and loading by (1) explicitly prioritizing any checkpoint whose state_dict contains your `final_regressor` keys (a strong signal it matches this exact architecture), and (2) if none match, selecting the “best” candidate by maximum key overlap with your model’s state_dict rather than just the first path found. This should move QWK materially upward toward your 0.90 target without changing training/eval semantics, and still always writes a valid `submission.csv`.'
- What this solution (achieved 0.14916) has done: 'Your score is far below the target, so we should increase QWK with the smallest change that preserves your model and inference logic. The biggest issue is that your `regress2class` thresholds are generic; QWK is very sensitive to these cut points, and using fixed thresholds often collapses predictions toward 0–1, yielding near-zero kappa. I keep the same network (`final=True` regression output) and the same class-conversion idea, but fit the four thresholds on the training set using out-of-fold-style predictions (no label leakage from test) by directly optimizing QWK on a held-out validation split. Then I apply those learned thresholds to test predictions and write a valid `submission.csv` exactly as required.'
- What this solution (achieved 0.16008) has done: 'Your current score (0.14916) is far below the target (0.90295), so we should increase QWK with the smallest possible change that preserves your architecture and inference path. Right now you fit thresholds on only one small validation split, which is noisy and can pick poor cut points that don’t generalize to the test set. I keep the exact same model (`final=True` regression head) and the same threshold-optimization method, but fit thresholds using out-of-fold predictions over the full training set (stratified K-fold) to make the thresholds much more stable and closer to what QWK needs. Then I run test inference exactly as you already do, using the learned thresholds, and write `submission.csv` in the same required format.'
- What this solution (achieved -0.07538) has done: 'Your score (0.16008) is far below the target (0.90295), so we should increase QWK with the smallest change that preserves your architecture and inference path. The most likely limiter is that you’re learning thresholds from OOF predictions generated by a backbone+heads that are not actually APTOS-trained (often only the backbone loads, leaving heads essentially random), so threshold fitting can’t rescue performance. I keep your model, transforms, and `final=True` regression inference unchanged, but (1) make checkpoint loading stricter by requiring that `final_regressor` weights are present (otherwise it’s not your trained 3-stage model), and (2) if that’s not available, fall back to a deterministic, label-aligned post-processing: fit thresholds by matching the training label distribution (quantile cutpoints) rather than QWK optimization on noisy pseudo-predictions. This avoids overfitting thresholds to junk outputs and typically moves QWK upward materially versus the current setup, while still producing a valid `submission.csv`.'
- What this solution (achieved 0.16008) has done: 'Your current score is far below the target, so we should push QWK upward with the smallest changes that keep your model/inference logic intact. The biggest likely issue is that when no real 3-stage checkpoint is available, you’re effectively using an ImageNet backbone with random heads, and the “label-prior quantile thresholds” can still produce systematically miscalibrated classes. I keep your `ThreeStage_Model`, `final=True` regression inference, and thresholding approach, but (1) make weight loading prefer any checkpoint that matches *either* `final_regressor` or the older non-final heads (`classifier/regressor/ordinal`) so we load the best available APTOS-ish weights instead of falling back too often, and (2) if we still don’t have a trained final head, learn thresholds by directly optimizing QWK on OOF predictions (as you already do when trained) rather than using label-prior quantiles. This is a minimal, score-relevant change: it doesn’t alter architecture or training, only makes sure we use the best available weights and a metric-aligned threshold fit.'
- What this solution (achieved -0.07538) has done: 'Your current score (0.16008) is far below the target (0.90295), so we should increase QWK with the smallest change that preserves your model and inference semantics. The biggest likely issue now is that you may be optimizing and applying thresholds for the `final=True` regression output even when the loaded checkpoint does not actually contain a trained `final_regressor` head (so those predictions are poorly calibrated and thresholding can’t rescue them). I keep the same model and `final=True` inference, but make checkpoint loading *strictly prefer and require* a `final_regressor`-containing checkpoint when available, and only then do QWK threshold optimization; otherwise we fall back to a distribution-matching threshold fit (quantile cutpoints) which is more stable than QWK-optimizing on junk predictions. This is a minimal, score-relevant change: no architecture/training loop changes, just making sure thresholds are learned in a way that matches whether the final head is truly trained.'

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
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.transforms import functional as FT
from PIL import Image, ImageChops

from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold
import timm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)



## === cell 1
threshold = np.array([0.75, 1.5, 2.5, 3.5], dtype=np.float32)


def regress2class(out, thr=None):
    thr = threshold if thr is None else np.asarray(thr, dtype=np.float32)
    out = out.detach()
    if out.ndim > 1:
        out = out.view(out.size(0), -1).squeeze(1)
    out_cpu = out.to("cpu")
    prediction = torch.zeros(out_cpu.size(0), device="cpu", dtype=torch.float32)
    for i in range(4):
        prediction += (out_cpu >= float(thr[i])).to(prediction.dtype)
    return prediction


def ordinal2class_prob(out):
    out = out.to(device)
    pred_prob = torch.zeros(out.size(0), 5, device=device)
    pred_prob[:, 0] = (1 - out[:, 0]).squeeze()
    pred_prob[:, 1] = (out[:, 0] * (1 - out[:, 1])).squeeze()
    pred_prob[:, 2] = (out[:, 1] * (1 - out[:, 2])).squeeze()
    pred_prob[:, 3] = (out[:, 2] * (1 - out[:, 3])).squeeze()
    pred_prob[:, 4] = out[:, 3].squeeze()
    return F.softmax(pred_prob, dim=1)


def regress2class_prob(out):
    out = out.to(device)
    pred_prob = torch.zeros((out.size(0), 5), device=device)
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
BASE_INPUT = "/kaggle/input/aptos2019-blindness-detection"
if not os.path.exists(os.path.join(BASE_INPUT, "test.csv")):
    BASE_INPUT = "/kaggle/data/aptos2019-blindness-detection"

test_csv_path = os.path.join(BASE_INPUT, "test.csv")
test_df = pd.read_csv(test_csv_path)
test_ids = np.squeeze(test_df["id_code"].values)

train_csv_path = os.path.join(BASE_INPUT, "train.csv")
train_df = pd.read_csv(train_csv_path)

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

ckpt_name = "B4_3stage_33epoch_CLAHE.pkl"
ckpt_stem = os.path.splitext(ckpt_name)[0]
candidate_paths = []

candidate_paths += glob.glob(
    os.path.join("/kaggle/input", "**", "weights", ckpt_name), recursive=True
)
candidate_paths += glob.glob(
    os.path.join("/kaggle/data", "**", "weights", ckpt_name), recursive=True
)

for base in ["/kaggle/input", "/kaggle/data", "/kaggle/working"]:
    for ext in ["pkl", "pth", "pt", "bin"]:
        candidate_paths += glob.glob(
            os.path.join(base, "**", f"{ckpt_stem}.{ext}"), recursive=True
        )
        candidate_paths += glob.glob(
            os.path.join(base, "**", "weights", f"{ckpt_stem}.{ext}"), recursive=True
        )
        candidate_paths += glob.glob(
            os.path.join(base, "**", "models", f"{ckpt_stem}.{ext}"), recursive=True
        )

keywords = [
    "aptos",
    "blind",
    "retina",
    "dr",
    "diabet",
    "efficientnet",
    "b4",
    "b5",
    "3stage",
    "clahe",
]
for base in ["/kaggle/input", "/kaggle/data"]:
    for ext in ["pkl", "pth", "pt", "bin"]:
        for p in glob.glob(os.path.join(base, "**", f"*.{ext}"), recursive=True):
            lp = p.lower()
            if any(k in lp for k in keywords):
                candidate_paths.append(p)

seen = set()
candidate_paths = [
    p for p in candidate_paths if os.path.exists(p) and not (p in seen or seen.add(p))
]


def _extract_state_dict(obj):
    if isinstance(obj, nn.Module):
        return obj.state_dict()
    if not isinstance(obj, dict):
        return None
    for key in ["state_dict", "model_state_dict", "model", "net", "weights", "params"]:
        if key in obj:
            candidate = obj[key]
            if isinstance(candidate, nn.Module):
                return candidate.state_dict()
            if isinstance(candidate, dict):
                return candidate
    if all(isinstance(k, str) for k in obj.keys()):
        return obj
    return None


def _sanitize_state_dict(sd):
    new_state = {}
    for k, v in sd.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module.") :]
        new_state[nk] = v
    return new_state


def _score_checkpoint_path(path, model_keys_set):
    """
    Change (score-relevant, minimal): for QWK to improve, we need a checkpoint that actually trains
    the `final_regressor` used by `net(x, final=True)`. We still keep a fallback (3-head checkpoint)
    so the notebook runs, but we will *prefer final-head checkpoints strongly* and we will know when
    we don't have one (so threshold learning can switch to a safer method).
    """
    try:
        raw = torch.load(path, map_location="cpu")
        sd = _extract_state_dict(raw)
        if sd is None:
            return -1, 0, 0, 0
        sd = _sanitize_state_dict(sd)
        keys = list(sd.keys())
        keys_set = set(keys)

        overlap = len(keys_set & model_keys_set)
        has_final = any(k.startswith("final_regressor") for k in keys)
        has_classifier = any(k.startswith("classifier") for k in keys)
        has_regressor = any(k.startswith("regressor") for k in keys)
        has_ordinal = any(k.startswith("ordinal") for k in keys)

        has_three_heads = has_classifier and has_regressor and has_ordinal

        score = (
            (500000 if has_final else 0)  # stronger preference than before
            + (30000 if has_three_heads else 0)
            + overlap
        )
        return score, overlap, int(has_final), int(has_three_heads)
    except Exception:
        return -1, 0, 0, 0


model_keys_set = set(net.state_dict().keys())
scored = []
for p in candidate_paths:
    s, overlap, has_final, has_three = _score_checkpoint_path(p, model_keys_set)
    if s >= 0:
        scored.append((s, overlap, has_final, has_three, p))
scored.sort(reverse=True, key=lambda x: (x[0], x[1], x[2], x[3]))

ckpt_path = scored[0][4] if len(scored) > 0 else None

before_sum = float(sum(p.detach().abs().sum().cpu() for p in net.parameters()))

loaded_ok = False
missing = unexpected = None
load_source = None
has_trained_final_head = False

if ckpt_path is not None:
    try:
        state_raw = torch.load(ckpt_path, map_location="cpu")
        state = _extract_state_dict(state_raw)
        if state is not None:
            state = _sanitize_state_dict(state)
            has_trained_final_head = any(
                k.startswith("final_regressor") for k in state.keys()
            )

            missing, unexpected = net.load_state_dict(state, strict=False)
            loaded_ok = True
            load_source = f"checkpoint:{ckpt_path}"
    except Exception as e:
        print("Checkpoint load error:", repr(e))
        loaded_ok = False

if not loaded_ok:
    try:
        pretrained_backbone = timm.models.tf_efficientnet_b4_ns(pretrained=True)
        pretrained_backbone.global_pool = GeM(flatten=True)
        net.backbone.load_state_dict(pretrained_backbone.state_dict(), strict=False)
        loaded_ok = True
        load_source = "timm_pretrained_backbone(tf_efficientnet_b4_ns)"
        has_trained_final_head = False
    except Exception as e:
        print("Pretrained backbone init error:", repr(e))
        loaded_ok = False

after_sum = float(sum(p.detach().abs().sum().cpu() for p in net.parameters()))
param_changed = abs(after_sum - before_sum) > 1e-3

if not loaded_ok:
    raise RuntimeError(
        "No usable weights could be loaded (neither checkpoint nor timm pretrained). "
        f"Found candidates: {len(candidate_paths)}. First candidate: {ckpt_path}"
    )

net = net.to(device)
net.eval()

print("BASE_INPUT:", BASE_INPUT)
print("Weights source:", load_source)
print("Has trained final head:", bool(has_trained_final_head))
print("Checkpoint candidates found:", len(candidate_paths))
print("Scored candidates:", len(scored))
if len(scored) > 0:
    print(
        "Top-scored checkpoint:",
        scored[0][4],
        "| has_final:",
        bool(scored[0][2]),
        "| has_three_heads:",
        bool(scored[0][3]),
        "| overlap:",
        scored[0][1],
    )
print("Loaded OK:", loaded_ok, "| params_changed:", param_changed)
if missing is not None and unexpected is not None:
    print(
        "load_state_dict missing keys:",
        len(missing),
        "| unexpected keys:",
        len(unexpected),
    )




## === cell 5
class RetinaDataset(Dataset):
    def __init__(self, df, img_dir, transform=None, with_label=False):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.with_label = with_label

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_id = self.df.loc[idx, "id_code"]
        image_name = os.path.join(self.img_dir, f"{img_id}.png")
        img = Image.open(image_name).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        if self.with_label:
            y = int(self.df.loc[idx, "diagnosis"])
            return img, y
        return img, img_id


def _apply_thresholds_np(preds_cont, thr):
    thr = np.asarray(thr, dtype=np.float32)
    preds_cont = np.asarray(preds_cont, dtype=np.float32).reshape(-1)
    return (preds_cont[:, None] >= thr[None, :]).sum(axis=1).astype(np.int64)


def _fit_thresholds_by_qwk(preds_cont, y_true, init_thr):
    preds_cont = np.asarray(preds_cont, dtype=np.float32).reshape(-1)
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    thr = np.array(init_thr, dtype=np.float32)

    def score(thr_):
        y_pred = _apply_thresholds_np(preds_cont, thr_)
        return cohen_kappa_score(y_true, y_pred, weights="quadratic")

    best = score(thr)

    for step in [0.25, 0.1, 0.05, 0.02]:
        improved = True
        while improved:
            improved = False
            for i in range(4):
                for delta in (-step, step):
                    cand = thr.copy()
                    cand[i] = cand[i] + delta
                    cand = np.clip(cand, 0.0, 4.5)
                    cand.sort()
                    s = score(cand)
                    if s > best + 1e-8:
                        thr, best = cand, s
                        improved = True
    return thr, best


def _fit_thresholds_by_label_quantiles(preds_cont, y_true):
    """
    Change (score-relevant, minimal): if the final head isn't trained, QWK-optimizing thresholds on
    noisy pseudo-regression outputs tends to overfit and can harm public LB. A safer, stable fallback
    is to match the training label distribution via quantiles of the predictions.
    """
    preds_cont = np.asarray(preds_cont, dtype=np.float32).reshape(-1)
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    counts = np.bincount(y_true, minlength=5).astype(np.float64)
    cdf = np.cumsum(counts) / counts.sum()
    qs = [cdf[0], cdf[1], cdf[2], cdf[3]]
    thr = np.quantile(preds_cont, qs).astype(np.float32)
    thr = np.clip(thr, 0.0, 4.5)
    thr.sort()
    return thr


train_img_dir = os.path.join(BASE_INPUT, "train_images")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

oof_preds = np.zeros(len(train_df), dtype=np.float32)
oof_y = train_df["diagnosis"].values.astype(np.int64)

for fold, (_, val_idx) in enumerate(
    skf.split(train_df["id_code"].values, oof_y), start=1
):
    df_val = train_df.iloc[val_idx].reset_index(drop=True)
    val_ds = RetinaDataset(df_val, train_img_dir, transform=transform, with_label=True)
    val_dl = DataLoader(
        val_ds,
        batch_size=8,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    fold_preds = []
    with torch.inference_mode():
        for xb, _yb in val_dl:
            xb = xb.to(device, non_blocking=True)
            out = net(xb, final=True).view(-1).detach().float().cpu().numpy()
            fold_preds.append(out)
    fold_preds = np.concatenate(fold_preds, axis=0)
    oof_preds[val_idx] = fold_preds

    if has_trained_final_head:
        fold_thr, fold_qwk = _fit_thresholds_by_qwk(
            fold_preds, oof_y[val_idx], threshold
        )
        print(f"Fold {fold}: val_qwk={float(fold_qwk):.6f} | thr={fold_thr.tolist()}")
    else:
        fold_thr = _fit_thresholds_by_label_quantiles(fold_preds, oof_y[val_idx])
        fold_qwk = cohen_kappa_score(
            oof_y[val_idx],
            _apply_thresholds_np(fold_preds, fold_thr),
            weights="quadratic",
        )
        print(
            f"Fold {fold}: val_qwk={float(fold_qwk):.6f} | thr(quantile)={fold_thr.tolist()}"
        )

if has_trained_final_head:
    learned_thr, oof_qwk = _fit_thresholds_by_qwk(oof_preds, oof_y, threshold)
    threshold = learned_thr
    print("OOF-learned thresholds (QWK-optimized):", threshold.tolist())
    print("OOF QWK using learned thresholds:", float(oof_qwk))
else:
    threshold = _fit_thresholds_by_label_quantiles(oof_preds, oof_y)
    oof_qwk = cohen_kappa_score(
        oof_y, _apply_thresholds_np(oof_preds, threshold), weights="quadratic"
    )
    print("OOF-learned thresholds (label-quantile fallback):", threshold.tolist())
    print("OOF QWK using quantile thresholds:", float(oof_qwk))

print("OOF label counts:", pd.Series(oof_y).value_counts().sort_index().to_dict())
print(
    "OOF pred counts:",
    pd.Series(_apply_thresholds_np(oof_preds, threshold))
    .value_counts()
    .sort_index()
    .to_dict(),
)




## === cell 6
class TestRetinaDataset(Dataset):
    def __init__(self, ids, img_dir, transform=None):
        self.ids = list(ids)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        image_name = os.path.join(self.img_dir, f"{img_id}.png")
        img = Image.open(image_name).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, img_id


test_img_dir = os.path.join(BASE_INPUT, "test_images")
ds = TestRetinaDataset(test_ids, test_img_dir, transform=transform)

dl = DataLoader(
    ds, batch_size=8, shuffle=False, num_workers=2, pin_memory=torch.cuda.is_available()
)

all_ids = []
all_preds = []

with torch.inference_mode():
    for xb, idb in dl:
        xb = xb.to(device, non_blocking=True)
        out = net(xb, final=True)  # shape [B, 1], range ~[0, 4.5]
        pred = regress2class(out, thr=threshold)
        all_ids.extend(list(idb))
        all_preds.extend(pred.to(torch.int64).cpu().numpy().tolist())

submission = pd.DataFrame(
    {
        "id_code": pd.Series(all_ids, dtype=str),
        "diagnosis": pd.Series(all_preds, dtype=int),
    }
)

submission = test_df.merge(submission, on="id_code", how="left")
if submission["diagnosis"].isna().any():
    submission["diagnosis"] = submission["diagnosis"].fillna(0).astype(int)

submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
print("diagnosis value counts:\n", submission["diagnosis"].value_counts().sort_index())
print("Final thresholds used:", threshold.tolist())
print("Has trained final head:", bool(has_trained_final_head))
