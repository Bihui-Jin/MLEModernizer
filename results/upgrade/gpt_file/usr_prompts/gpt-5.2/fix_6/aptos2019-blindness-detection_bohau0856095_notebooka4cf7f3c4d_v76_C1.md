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

0.9217688344313156

# 6. Current score

0.23183

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the missing weight file crash by loading the model weights only if the file exists, and otherwise fall back to a deterministic, valid baseline prediction so a submission is always produced. I also fix the CUDA crash by selecting `cuda` only when it’s actually available (otherwise run on CPU). To prevent transform failures and empty outputs, I make `trim()` always return an image and add safe image loading with a fallback, ensuring we always generate exactly one prediction per `id_code`. Finally, I keep the model/transform logic intact when weights are present, so the score can improve toward the target when the expected weights exist in the Kaggle environment.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is coming from always predicting class 0 when the weight file isn’t found, which produces a valid submission but very poor QWK. To move the score toward the 0.92 target with minimal logic changes, I keep your exact model/transform/prediction path when weights exist, but when they don’t, I switch to a deterministic label-prior baseline: predict the training-set majority class (computed from `train.csv`). This preserves evaluation semantics (still outputs integer 0–4 labels) and typically yields a noticeably higher QWK than all-zeros, without changing architecture or training. I also ensure `train.csv` is read from the same DATA_ROOT so it works in this environment.'
- What this solution (achieved 0.19197) has done: 'Your 0.0 score is almost certainly because the model weights are never found/loaded, so the script falls back to a constant-label submission. To move the score upward toward the 0.92 target with minimal changes and without altering your model logic, I (1) robustly search for the weight file under `/kaggle/input/**` (where Kaggle datasets actually mount), and (2) if weights still aren’t found, replace the constant/majority fallback with a deterministic lightweight baseline that uses simple image brightness (no training loop, no architecture change) to produce non-constant 0–4 predictions. This keeps the existing inference path identical when weights exist, but avoids the worst-case 0.0 kappa behavior when they don’t. The submission writing and column format remain unchanged.'
- What this solution (achieved 0.23183) has done: 'Your current score (0.19197) is far below the target (0.9218), so we should improve performance without changing your core model/inference logic. The most likely limiter is that the weights aren’t being loaded correctly (or are loaded but don’t match the checkpoint structure), forcing the fallback baseline; we make weight loading robust to common checkpoint formats (raw state_dict, nested `state_dict`, `model`, `net`) and tolerant of `module.` prefixes so it actually uses the trained model when available. If weights still can’t be loaded, we keep your brightness fallback but make it slightly more informative by using two simple, deterministic image statistics (mean brightness + contrast) and mapping them through per-class centroids computed from a small capped training subset (still no training loop, and very lightweight). These changes are minimal, deterministic, and directly aimed at moving QWK upward toward the target while preserving your architecture and evaluation semantics.'
- What this solution (achieved 0.23183) has done: 'Your score is far below the target, so we should make a small, legitimate improvement that increases QWK without changing your model/training logic. The biggest leverage with minimal risk is fixing the regression-to-class thresholds: your current thresholds are generic, but QWK is very sensitive to calibration; we can learn optimal 4 cutpoints from `train.csv` by matching the distribution of your model’s continuous outputs to the true label quantiles. This keeps the same network, same forward pass, and same inference approach; it only adjusts how we bin the regressor output into 0–4, which directly targets the metric. If weights are not found, we keep your centroid fallback unchanged.'

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
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import functional as FT
from PIL import Image, ImageChops
import cv2

from sklearn.metrics import cohen_kappa_score
import timm

device = "cuda" if torch.cuda.is_available() else "cpu"

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



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
DATA_ROOT = "../input/aptos2019-blindness-detection"
TEST_CSV = os.path.join(DATA_ROOT, "test.csv")
TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
TEST_IMG_DIR = os.path.join(DATA_ROOT, "test_images")


def find_weight_file(filename: str):
    candidates = [
        "../input/weights/" + filename,
        os.path.join(DATA_ROOT, "weights", filename),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p

    roots = ["../input", "/kaggle/input", "/kaggle/data/input"]
    for root in roots:
        if not os.path.exists(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            if filename in filenames:
                return os.path.join(dirpath, filename)
    return None


def _extract_state_dict(ckpt_obj):
    if isinstance(ckpt_obj, dict):
        for k in ["state_dict", "model_state_dict", "model", "net", "weights"]:
            if k in ckpt_obj and isinstance(ckpt_obj[k], dict):
                return ckpt_obj[k]
        if any(isinstance(v, torch.Tensor) for v in ckpt_obj.values()):
            return ckpt_obj
    return None


def _strip_module_prefix(sd: dict):
    if not isinstance(sd, dict) or len(sd) == 0:
        return sd
    if all(k.startswith("module.") for k in sd.keys()):
        return {k[len("module.") :]: v for k, v in sd.items()}
    return sd


test_ids = pd.read_csv(TEST_CSV)
test_ids = np.squeeze(test_ids["id_code"].values)

input_size = 512

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

weight_filename = "B4_3stage_20epoch_finetune512.pkl"
weight_path = find_weight_file(weight_filename)

has_weights = False
if weight_path is not None:
    try:
        ckpt = torch.load(weight_path, map_location="cpu")
        sd = _extract_state_dict(ckpt)
        sd = _strip_module_prefix(sd)
        if sd is not None:
            missing, unexpected = net.load_state_dict(sd, strict=False)
            if len(missing) < 20:
                has_weights = True
    except Exception:
        has_weights = False

net = net.to(device)
net.eval()

fallback_label = 0
if os.path.exists(TRAIN_CSV):
    train_df = pd.read_csv(TRAIN_CSV)
    if "diagnosis" in train_df.columns and len(train_df) > 0:
        fallback_label = int(train_df["diagnosis"].value_counts().idxmax())

TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train_images")

centroids = None
if (not has_weights) and os.path.isdir(TRAIN_IMG_DIR) and os.path.exists(TRAIN_CSV):
    train_df = pd.read_csv(TRAIN_CSV)[["id_code", "diagnosis"]].copy()

    per_class_cap = 60  # small cap for speed; deterministic via original CSV order
    rows = []
    for cls in range(5):
        cls_rows = train_df[train_df["diagnosis"] == cls]
        if len(cls_rows) == 0:
            continue
        rows.append(cls_rows.head(per_class_cap))
    sample_df = pd.concat(rows, axis=0, ignore_index=True)

    feats = {cls: [] for cls in range(5)}
    for rid, cls in zip(sample_df["id_code"].tolist(), sample_df["diagnosis"].tolist()):
        p = os.path.join(TRAIN_IMG_DIR, f"{rid}.png")
        try:
            im = cv2.imread(p, cv2.IMREAD_COLOR)
            if im is None:
                continue
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            m = float(gray.mean())
            s = float(gray.std())
            feats[int(cls)].append((m, s))
        except Exception:
            continue

    c = []
    for cls in range(5):
        arr = np.array(feats[cls], dtype=float)
        if arr.shape[0] >= 5:
            c.append(arr.mean(axis=0))
        else:
            c.append(np.array([np.nan, np.nan], dtype=float))
    c = np.stack(c, axis=0)

    if np.isfinite(c).sum() >= 6:
        global_mean = np.nanmean(c, axis=0)
        c = np.where(np.isfinite(c), c, global_mean)
        centroids = c  # shape (5,2)


def _calibrate_thresholds_from_train(
    net, transform, train_csv, train_img_dir, device, max_per_class=120
):
    df = pd.read_csv(train_csv)[["id_code", "diagnosis"]].copy()
    df["diagnosis"] = df["diagnosis"].astype(int).clip(0, 4)

    parts = []
    for cls in range(5):
        sub = df[df["diagnosis"] == cls]
        if len(sub) > 0:
            parts.append(sub.head(max_per_class))
    sdf = pd.concat(parts, axis=0, ignore_index=True)
    if len(sdf) < 50:
        return None  # too small / something wrong

    preds = []
    y = []
    net.eval()
    with torch.no_grad():
        for rid, cls in zip(sdf["id_code"].tolist(), sdf["diagnosis"].tolist()):
            p = os.path.join(train_img_dir, f"{rid}.png")
            try:
                img = Image.open(p).convert("RGB")
            except Exception:
                continue
            x = transform(img).unsqueeze(0).to(device)
            try:
                _, r_out, _ = net(x)
                v = float(r_out.squeeze().detach().cpu().item())
            except Exception:
                continue
            preds.append(v)
            y.append(int(cls))

    if len(preds) < 50:
        return None

    preds = np.asarray(preds, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)

    fracs = [np.mean(y <= k) for k in [0, 1, 2, 3]]
    fracs = np.clip(np.maximum.accumulate(fracs), 1e-3, 1 - 1e-3)

    thr = [float(np.quantile(preds, q)) for q in fracs]

    thr = np.clip(np.maximum.accumulate(thr), 0.0, 4.5).tolist()
    return thr


if has_weights and os.path.exists(TRAIN_CSV) and os.path.isdir(TRAIN_IMG_DIR):
    thr = _calibrate_thresholds_from_train(
        net=net,
        transform=transform,
        train_csv=TRAIN_CSV,
        train_img_dir=TRAIN_IMG_DIR,
        device=device,
        max_per_class=120,
    )
    if thr is not None and len(thr) == 4:
        threshold = thr



## === cell 5
submission = []

for i, idx in enumerate(test_ids):
    image_name = os.path.join(TEST_IMG_DIR, f"{idx}.png")
    if has_weights:
        try:
            img = Image.open(image_name).convert("RGB")
        except Exception:
            img = Image.new("RGB", (input_size, input_size), (0, 0, 0))
        img = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            _, r_out, _ = net(img)
            pred = regress2class(r_out.data.squeeze(1))
        submission.append([idx, int(pred.item())])
    else:
        pred_label = fallback_label
        if centroids is not None:
            try:
                im = cv2.imread(image_name, cv2.IMREAD_COLOR)
                if im is not None:
                    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                    x = np.array([float(gray.mean()), float(gray.std())], dtype=float)
                    d = ((centroids - x[None, :]) ** 2).sum(axis=1)
                    pred_label = int(np.argmin(d))
            except Exception:
                pred_label = fallback_label
        submission.append([idx, int(pred_label)])

submission = np.array(submission, dtype=object)



## === cell 6
df = pd.DataFrame(submission, columns=["id_code", "diagnosis"])

df["id_code"] = df["id_code"].astype(str)
df["diagnosis"] = (
    pd.to_numeric(df["diagnosis"], errors="coerce").fillna(0).astype(int).clip(0, 4)
)

assert len(df) == len(test_ids) and len(df) > 0

df.to_csv("submission.csv", index=False)
print(df.head())
print(
    "Wrote submission.csv with",
    len(df),
    "rows. Weights_loaded=",
    has_weights,
    "device=",
    device,
    "fallback_label=",
    fallback_label,
    "weight_path=",
    weight_path,
    "centroid_baseline=",
    centroids is not None,
    "thresholds=",
    threshold,
)
