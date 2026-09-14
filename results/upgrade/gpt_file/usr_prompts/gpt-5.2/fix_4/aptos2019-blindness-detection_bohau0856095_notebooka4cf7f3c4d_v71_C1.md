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

0.9218033364342396

# 6. Current score

0.02131

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I make the code robust to the Kaggle CPU-only environment by selecting `cuda` only if available and mapping weights to CPU when loading. I also fix a couple of small but critical logic bugs in transforms (`trim()` returning `None` sometimes and the incorrect `is` string comparison) that can break preprocessing. Since your provided weights path doesn’t exist in this dataset, I add a minimal fallback that runs end-to-end by generating a valid submission via a simple baseline derived from the training label distribution (so the notebook always produces a non-empty `submission.csv`). These changes are focused on unblocking execution and producing a valid CSV; if you later provide the correct weights file, the same script automatically use it.'
- What this solution (achieved 0.02131) has done: 'I fix the immediate crash when loading a timm pretrained backbone by allowing the missing `global_pool.p` key (your model replaces `global_pool` with GeM, so that parameter doesn’t exist in the pretrained state dict). Then I prevent the cascade `NameError`s by making sure `train_df` and `train_label_mode` are always defined (even if weight-loading fails), so threshold tuning and the submission writer can run. Finally, I keep the same inference core logic, but make the test prediction loop robust to missing/corrupt images so the notebook always produces a valid `submission.csv`.'

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
from sklearn.model_selection import train_test_split
import timm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

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
DATA_ROOT_CANDIDATES = [
    "../input/aptos2019-blindness-detection",
    "/kaggle/input/aptos2019-blindness-detection",
    "../input",
    "/kaggle/input",
    "/kaggle/data/aptos2019-blindness-detection",
    "/kaggle/data",
]
DATA_ROOT = None
for cand in DATA_ROOT_CANDIDATES:
    if os.path.exists(os.path.join(cand, "train.csv")) and os.path.exists(
        os.path.join(cand, "test.csv")
    ):
        DATA_ROOT = cand
        break
if DATA_ROOT is None:
    raise FileNotFoundError(
        "Could not locate dataset root. Checked: " + str(DATA_ROOT_CANDIDATES)
    )

WEIGHTS_PATH = "../input/weights/B4_3stage_17epoch_finetune.pkl"

test_df = pd.read_csv(f"{DATA_ROOT}/test.csv")
test_ids = np.squeeze(test_df["id_code"].values)

train_df = pd.read_csv(f"{DATA_ROOT}/train.csv")
train_label_mode = int(train_df["diagnosis"].mode().iloc[0])

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

has_weights = os.path.exists(WEIGHTS_PATH)
if has_weights:
    state = torch.load(WEIGHTS_PATH, map_location=device)
    net.load_state_dict(state, strict=True)
    print("Loaded weights:", WEIGHTS_PATH)
else:
    print("WARNING: weights file not found:", WEIGHTS_PATH)
    print("Initializing backbone from timm pretrained weights.")
    pretrained_backbone = timm.models.tf_efficientnet_b4_ns(pretrained=True)
    missing, unexpected = net.backbone.load_state_dict(
        pretrained_backbone.state_dict(), strict=False
    )
    print(
        "Backbone load_state_dict strict=False; missing:",
        missing,
        "unexpected:",
        unexpected,
    )
    del pretrained_backbone




## === cell 5
def find_best_thresholds(y_true, y_pred_cont, init_thr=None, n_iters=2):
    thr = np.array(
        init_thr if init_thr is not None else [0.75, 1.5, 2.5, 3.5], dtype=np.float64
    )

    def apply_thr(pred, t):
        pred = np.asarray(pred)
        out = np.zeros_like(pred, dtype=np.int64)
        out += pred >= t[0]
        out += pred >= t[1]
        out += pred >= t[2]
        out += pred >= t[3]
        return out

    best_kappa = cohen_kappa_score(
        y_true, apply_thr(y_pred_cont, thr), weights="quadratic"
    )

    for _ in range(n_iters):
        for i in range(4):
            cur = thr[i]
            grid = np.arange(cur - 0.35, cur + 0.3501, 0.025)
            grid = np.clip(grid, 0.05, 4.25)
            for cand in grid:
                t2 = thr.copy()
                t2[i] = cand
                if not (t2[0] < t2[1] < t2[2] < t2[3]):
                    continue
                k = cohen_kappa_score(
                    y_true, apply_thr(y_pred_cont, t2), weights="quadratic"
                )
                if k > best_kappa:
                    best_kappa = k
                    thr = t2
    return thr.tolist(), float(best_kappa)


USE_THRESHOLD_TUNING = True

if USE_THRESHOLD_TUNING:
    tr_ids = train_df["id_code"].values
    y = train_df["diagnosis"].values.astype(int)

    tr_idx, va_idx = train_test_split(
        np.arange(len(tr_ids)),
        test_size=0.15,
        random_state=42,
        stratify=y,
    )
    max_val = 256
    va_idx = va_idx[:max_val]

    y_true = y[va_idx]
    y_pred_cont = []

    net.eval()
    with torch.no_grad():
        for k in va_idx:
            image_name = f"{DATA_ROOT}/train_images/{tr_ids[k]}.png"
            if not os.path.exists(image_name):
                alt = f"{DATA_ROOT}/train_images/{tr_ids[k]}.jpg"
                image_name = alt if os.path.exists(alt) else image_name

            try:
                img = Image.open(image_name).convert("RGB")
            except Exception:
                y_pred_cont.append(float(train_label_mode))
                continue

            img = transform(img).unsqueeze(0).to(device)
            _, r_out, _ = net(img)
            y_pred_cont.append(float(r_out.squeeze(1).item()))

    new_thr, val_kappa = find_best_thresholds(
        y_true, y_pred_cont, init_thr=threshold, n_iters=2
    )
    print("Threshold tuning val_kappa:", val_kappa)
    print("Old thresholds:", threshold, "New thresholds:", new_thr)
    threshold = new_thr



## === cell 6
submission = []

with torch.no_grad():
    for i, idx in enumerate(test_ids):
        if i % 50 == 0:
            print(i, "/", len(test_ids))

        image_name = f"{DATA_ROOT}/test_images/{idx}.png"
        if not os.path.exists(image_name):
            alt = f"{DATA_ROOT}/test_images/{idx}.jpg"
            image_name = alt if os.path.exists(alt) else image_name

        try:
            img = Image.open(image_name).convert("RGB")
            img = transform(img).unsqueeze(0).to(device)
            _, r_out, _ = net(img)
            pred = regress2class(r_out.data.squeeze(1))
            submission.append([idx, int(pred.item())])
        except Exception:
            submission.append([idx, int(train_label_mode)])

submission = np.array(submission, dtype=object)



## === cell 7
df = pd.DataFrame(submission, columns=["id_code", "diagnosis"])

df["id_code"] = df["id_code"].astype(str)
df["diagnosis"] = (
    pd.to_numeric(df["diagnosis"], errors="coerce").fillna(train_label_mode).astype(int)
)
df["diagnosis"] = df["diagnosis"].clip(0, 4)

df = test_df.merge(df, on="id_code", how="left")
df["diagnosis"] = df["diagnosis"].fillna(train_label_mode).astype(int)
df = df[["id_code", "diagnosis"]]

out_path = "submission.csv"
df.to_csv(out_path, index=False)
print("Wrote:", out_path)
print(df.head())
print("Rows:", len(df), "Cols:", df.shape[1])
print("Diagnosis value counts:\n", df["diagnosis"].value_counts().sort_index())
