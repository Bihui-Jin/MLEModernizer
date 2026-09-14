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

0.912829436921204

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.24331) has done: 'I (1) remove the failing `pip install` and make `timm` usage compatible with the already-installed version, (2) make device selection robust by falling back to CPU when CUDA isn’t available (fixing the “no NVIDIA driver” crash), and (3) fix weight-loading so it doesn’t hard-fail when the referenced checkpoint file isn’t present (and instead loads any available `.pth/.pt` in `../input/weights` or runs with the initialized model). Finally, I ensure inference always runs end-to-end and writes a non-empty `submission.csv` with the exact required columns and id alignment. These changes are execution/stability fixes; they don’t alter the core model definition or prediction logic beyond safely handling missing files/hardware.'
- What this solution (achieved 0.0) has done: 'The runtime error comes from a feature-dimension mismatch: with `timm` EfficientNet models, `self.backbone(x)` returns the classifier logits (1000-d) rather than pooled features, so your `Linear(n_features,1)` expects 1792 but receives 1000. The minimal fix is to replace the backbone classifier head with an identity layer so the forward pass returns pooled features, preserving the rest of the architecture and checkpoint-loading logic. I also make `regress2class` device-safe and slightly more robust (no `.data`), and I keep all paths and submission formatting unchanged so a valid `submission.csv` is always written.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is most consistent with a “submission is valid CSV but predictions are effectively uninformative,” which can happen when the checkpoint isn’t loaded (wrong path) and the randomly-initialized regressor outputs almost-constant values that mostly map to class 2 via your thresholds. To move the score toward the target with minimal, core-logic-preserving changes, I (1) make checkpoint discovery robust by searching common Kaggle input locations (while keeping the original preferred path first), and (2) fix the B4 backbone head removal more completely using `reset_classifier(0)` when available so the feature dimension matches `num_features` deterministically. Everything else (same model, GeM pooling, sigmoid*4.5 regression, fixed thresholds, and per-image inference loop) stays the same, and the script still writes `submission.csv` in the required format.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with the model running essentially untrained (or loading the wrong weights), producing near-constant predictions that map poorly to the 5 ordinal classes. To move the score upward toward the target with minimal disruption, I (1) make checkpoint discovery prefer the exact expected filename anywhere under `../input/**` (not just `../input/weights`), and (2) align the `Regressor` definition with the actual inference `Model` (B4 + head removed + GeM + correct `num_features`) so that if the checkpoint contains that architecture it loads cleanly instead of partially/incorrectly. Everything else (thresholds, sigmoid*4.5 regression, per-image inference loop, and submission formatting) stays the same to preserve evaluation semantics.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with a submission that is technically valid but badly miscalibrated for quadratic weighted kappa, which commonly happens when fixed regression thresholds don’t match the model’s output scale/distribution on this environment. I keep your exact model/inference logic (EfficientNet-B4 -> GeM -> Linear -> sigmoid*4.5) and only add a minimal, metric-aligned post-processing step: fit 4 optimal thresholds on a small validation split of the provided training set using your current model outputs, then apply those thresholds to test predictions. This preserves the core architecture and prediction semantics (still regression + thresholding), but adapts the thresholds to the actual loaded checkpoint (or untrained model) so the kappa moves upward toward your target. I also make image path resolution robust to both `../input/...` and `/kaggle/input/...` without changing the dataset used, and ensure `submission.csv` is always written.'

# 9. Code solution

## === cell 0
import os
import glob
import random
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torchvision.transforms as transforms
from PIL import Image, ImageChops

import timm

device = "cuda:0" if torch.cuda.is_available() else "cpu"


def trim(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0, 0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -10)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)
    return im


def _resolve_data_root():
    candidates = [
        "../input/aptos2019-blindness-detection",
        "/kaggle/input/aptos2019-blindness-detection",
        "../input",
        "/kaggle/input",
    ]
    for c in candidates:
        if os.path.exists(os.path.join(c, "train.csv")) and os.path.exists(
            os.path.join(c, "test.csv")
        ):
            return c
    return "../input/aptos2019-blindness-detection"


DATA_ROOT = _resolve_data_root()
TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
TEST_CSV = os.path.join(DATA_ROOT, "test.csv")
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train_images")
TEST_IMG_DIR = os.path.join(DATA_ROOT, "test_images")

print("DATA_ROOT:", DATA_ROOT)
print("Device:", device)



## === cell 1
threshold = [0.7, 1.5, 2.5, 3.5]


def regress2class(out: torch.Tensor, thr=None) -> torch.Tensor:
    """
    Convert regression output (float) to ordinal class {0..4} using thresholds.
    Keeps tensor on CPU for safe int conversion later.
    """
    if thr is None:
        thr = threshold
    out_cpu = out.detach().view(-1).cpu()
    prediction = torch.zeros(out_cpu.size(0), dtype=torch.long)
    for i in range(4):
        prediction += (out_cpu >= float(thr[i])).long()
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
        self.backbone = timm.create_model("tf_efficientnet_b4_ns", pretrained=False)

        if hasattr(self.backbone, "reset_classifier"):
            self.backbone.reset_classifier(0)
        else:
            if hasattr(self.backbone, "classifier"):
                self.backbone.classifier = nn.Identity()
            if hasattr(self.backbone, "fc"):
                self.backbone.fc = nn.Identity()
            if hasattr(self.backbone, "head"):
                self.backbone.head = nn.Identity()

        self.backbone.global_pool = GeM(flatten=True)

        n_features = getattr(self.backbone, "num_features", 1000)
        self.regressor = nn.Linear(n_features, 1)

    def forward(self, x):
        x = self.backbone(x)
        out = self.regressor(x)
        out = torch.sigmoid(out) * 4.5
        return out




## === cell 3
test_df = pd.read_csv(TEST_CSV)
test_ids = np.squeeze(test_df["id_code"].values)

input_size = 384

tranforms = transforms.Compose(
    [
        transforms.Resize(int(input_size * 1.15)),
        transforms.CenterCrop((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.backbone = timm.create_model("tf_efficientnet_b4_ns", pretrained=False)

        if hasattr(self.backbone, "reset_classifier"):
            self.backbone.reset_classifier(0)
        else:
            if hasattr(self.backbone, "classifier"):
                self.backbone.classifier = nn.Identity()
            if hasattr(self.backbone, "fc"):
                self.backbone.fc = nn.Identity()
            if hasattr(self.backbone, "head"):
                self.backbone.head = nn.Identity()

        self.backbone.global_pool = GeM(flatten=True)

        n_features = getattr(self.backbone, "num_features", 1000)
        self.regressor = nn.Linear(n_features, 1)

    def forward(self, x):
        x = self.backbone(x)
        out = self.regressor(x)
        out = torch.sigmoid(out) * 4.5
        return out


def _load_checkpoint_into_model(model, ckpt_path, device):
    obj = torch.load(ckpt_path, map_location=device)
    if isinstance(obj, nn.Module):
        model = obj
        model.to(device)
        return model
    if isinstance(obj, dict):
        state = obj.get("state_dict", obj)
        if isinstance(state, dict):
            new_state = {}
            for k, v in state.items():
                nk = k[7:] if k.startswith("module.") else k
                new_state[nk] = v
            model.load_state_dict(new_state, strict=False)
        return model
    return model


net = Model().to(device)

preferred_ckpt = "../input/weights/0.912_tf_efficientnet_b4_ns_regress.pth"

search_globs = [
    preferred_ckpt,
    "../input/**/0.912_tf_efficientnet_b4_ns_regress.pth",
    "/kaggle/input/**/0.912_tf_efficientnet_b4_ns_regress.pth",
    "../input/weights/*.pth",
    "../input/weights/*.pt",
    "../input/**/*.pth",
    "../input/**/*.pt",
    "/kaggle/input/**/*.pth",
    "/kaggle/input/**/*.pt",
]
ckpt_candidates = []
for g in search_globs:
    ckpt_candidates.extend(glob.glob(g, recursive=True))


def _ckpt_rank(p: str) -> tuple:
    base = os.path.basename(p).lower()
    exact = 0 if base == "0.912_tf_efficientnet_b4_ns_regress.pth" else 1
    return (
        exact,
        0 if "0.912" in base else 1,
        0 if "efficientnet_b4" in base else 1,
        0 if "regress" in base else 1,
        len(p),
        p,
    )


ckpt_candidates = sorted(list(dict.fromkeys(ckpt_candidates)), key=_ckpt_rank)

if len(ckpt_candidates) > 0 and os.path.exists(ckpt_candidates[0]):
    net = _load_checkpoint_into_model(net, ckpt_candidates[0], device)

net.eval()

print(
    "Checkpoint used:",
    (
        ckpt_candidates[0]
        if len(ckpt_candidates) > 0
        else "None found (running untrained)"
    ),
)
print("Backbone num_features:", getattr(net.backbone, "num_features", "NA"))



## === cell 4
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import cohen_kappa_score
from scipy.optimize import minimize


def _qwk(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


def _apply_thresholds(preds_1d: np.ndarray, thr: np.ndarray) -> np.ndarray:
    thr = np.asarray(thr, dtype=np.float64)
    thr = np.sort(thr)
    return np.digitize(preds_1d, thr, right=False).astype(np.int64)


def _fit_thresholds(y_true: np.ndarray, preds: np.ndarray, init_thr=None):
    y_true = y_true.astype(np.int64)
    preds = preds.astype(np.float64)

    if init_thr is None:
        init_thr = np.array([0.7, 1.5, 2.5, 3.5], dtype=np.float64)
    else:
        init_thr = np.asarray(init_thr, dtype=np.float64)

    def loss(thr):
        thr_sorted = np.sort(thr)
        y_pred = _apply_thresholds(preds, thr_sorted)
        return -_qwk(y_true, y_pred)

    res = minimize(
        loss,
        init_thr,
        method="Nelder-Mead",
        options={"maxiter": 400, "xatol": 1e-4, "fatol": 1e-4},
    )
    thr_best = np.sort(res.x)
    return thr_best, res


def _predict_regression_for_ids(ids, img_dir, batch_size=8, max_n=None):
    outs = []
    ids_used = []
    batch = []
    batch_ids = []
    with torch.no_grad():
        for k, idx in enumerate(ids):
            if max_n is not None and k >= max_n:
                break
            image_name = os.path.join(img_dir, f"{idx}.png")
            img = Image.open(image_name).convert("RGB")
            img = trim(img)
            tens = tranforms(img)
            batch.append(tens)
            batch_ids.append(idx)
            if len(batch) == batch_size:
                x = torch.stack(batch, dim=0).to(device)
                r_out = net(x).view(-1).detach().cpu().numpy()
                outs.append(r_out)
                ids_used.extend(batch_ids)
                batch, batch_ids = [], []
        if len(batch) > 0:
            x = torch.stack(batch, dim=0).to(device)
            r_out = net(x).view(-1).detach().cpu().numpy()
            outs.append(r_out)
            ids_used.extend(batch_ids)
    return np.concatenate(outs, axis=0), np.array(ids_used)


train_df = pd.read_csv(TRAIN_CSV)
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
tr_idx, va_idx = next(
    sss.split(train_df["id_code"].values, train_df["diagnosis"].values)
)
val_df = train_df.iloc[va_idx].reset_index(drop=True)

val_preds, val_ids_used = _predict_regression_for_ids(
    val_df["id_code"].values, TRAIN_IMG_DIR, batch_size=8
)
val_y = val_df["diagnosis"].values[: len(val_preds)]

thr_fitted, opt_res = _fit_thresholds(
    val_y, val_preds, init_thr=np.array(threshold, dtype=np.float64)
)
val_pred_cls = _apply_thresholds(val_preds, thr_fitted)
val_qwk = _qwk(val_y, val_pred_cls)

print("Default thresholds:", threshold)
print("Fitted thresholds:", thr_fitted.tolist())
print("Validation QWK (fitted thresholds):", float(val_qwk))



## === cell 5
submission = []

with torch.no_grad():
    for i, idx in enumerate(test_ids):
        image_name = os.path.join(TEST_IMG_DIR, f"{idx}.png")
        img = Image.open(image_name).convert("RGB")
        img = trim(img)
        img = tranforms(img).unsqueeze(0).to(device)

        r_out = net(img).view(-1)
        pred = regress2class(r_out, thr=thr_fitted)
        submission.append([idx, int(pred.item())])

submission = np.array(submission, dtype=object)



## === cell 6
df = pd.DataFrame(submission, columns=["id_code", "diagnosis"])

df["id_code"] = df["id_code"].astype(str)
df["diagnosis"] = df["diagnosis"].astype(int)

df = df.set_index("id_code").reindex(test_df["id_code"].astype(str)).reset_index()

df.to_csv("submission.csv", index=False)
print(df.head())
print("Wrote submission.csv with shape:", df.shape)
print(
    "Diagnosis value counts:\n", df["diagnosis"].value_counts(dropna=False).sort_index()
)
