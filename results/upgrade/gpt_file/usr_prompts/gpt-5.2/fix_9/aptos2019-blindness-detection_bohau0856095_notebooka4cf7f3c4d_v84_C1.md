# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

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
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import functional as FT
from PIL import Image, ImageChops

from sklearn.metrics import cohen_kappa_score
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import timm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



## === cell 1
threshold = [0.75, 1.5, 2.5, 3.5]


def regress2class(out):
    out = out.view(-1)
    prediction = torch.zeros(out.size(0), dtype=torch.int64)
    for i in range(4):
        prediction += (out >= threshold[i]).to(torch.int64).cpu()
    return prediction


def ordinal2class_prob(out):
    out = out.to(dtype=torch.float32)
    pred_prob = torch.zeros(out.size(0), 5, device=out.device, dtype=torch.float32)
    pred_prob[:, 0] = (1 - out[:, 0]).squeeze()
    pred_prob[:, 1] = (out[:, 0] * (1 - out[:, 1])).squeeze()
    pred_prob[:, 2] = (out[:, 1] * (1 - out[:, 2])).squeeze()
    pred_prob[:, 3] = (out[:, 2] * (1 - out[:, 3])).squeeze()
    pred_prob[:, 4] = out[:, 3].squeeze()
    return F.softmax(pred_prob, dim=1)


def regress2class_prob(out):
    out = out.view(-1).to(dtype=torch.float32)
    pred_prob = torch.zeros((out.size(0), 5), device=out.device, dtype=torch.float32)
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
    def __init__(self, pretrained_backbone=False):
        super(Regressor, self).__init__()
        self.backbone = timm.models.tf_efficientnet_b5_ns(
            pretrained=pretrained_backbone
        )
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
TEST_CSV = "../input/aptos2019-blindness-detection/test.csv"
TEST_IMG_DIR = "../input/aptos2019-blindness-detection/test_images"

if not os.path.exists(TEST_CSV):
    alt = "/kaggle/input/aptos2019-blindness-detection/test.csv"
    if os.path.exists(alt):
        TEST_CSV = alt

if not os.path.isdir(TEST_IMG_DIR):
    alt_dir = "/kaggle/input/aptos2019-blindness-detection/test_images"
    if os.path.isdir(alt_dir):
        TEST_IMG_DIR = alt_dir

test_ids_df = pd.read_csv(TEST_CSV)
test_ids = test_ids_df["id_code"].astype(str).values

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


def find_weight_file():
    candidates = [
        "../input/weights/B4_3stage_4epoch_finetune2_512.pkl",
        "/kaggle/input/weights/B4_3stage_4epoch_finetune2_512.pkl",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    for pat in [
        "../input/**/B4_3stage_4epoch_finetune2_512.pkl",
        "/kaggle/input/**/B4_3stage_4epoch_finetune2_512.pkl",
    ]:
        hits = glob.glob(pat, recursive=True)
        if hits:
            return hits[0]
    return None


weight_path = find_weight_file()
weights_loaded = False

use_timm_pretrained_backbone = weight_path is None
net = ThreeStage_Model(pretrained_backbone=use_timm_pretrained_backbone)

if weight_path is not None:
    state = torch.load(weight_path, map_location="cpu")
    net.load_state_dict(state)
    weights_loaded = True
else:
    print(
        "WARNING: Pretrained competition weights not found. "
        "Using timm pretrained EfficientNet backbone."
    )

net = net.to(device)
net.eval()

TRAIN_CSV = "../input/aptos2019-blindness-detection/train.csv"
TRAIN_IMG_DIR = "../input/aptos2019-blindness-detection/train_images"
if not os.path.exists(TRAIN_CSV):
    alt = "/kaggle/input/aptos2019-blindness-detection/train.csv"
    if os.path.exists(alt):
        TRAIN_CSV = alt
if not os.path.isdir(TRAIN_IMG_DIR):
    alt_dir = "/kaggle/input/aptos2019-blindness-detection/train_images"
    if os.path.isdir(alt_dir):
        TRAIN_IMG_DIR = alt_dir

calibrator = None
calibrator_kind = None

if not weights_loaded:
    train_df = pd.read_csv(TRAIN_CSV)
    train_df = train_df.sample(n=min(2400, len(train_df)), random_state=42).reset_index(
        drop=True
    )

    feat_device = torch.device("cpu")
    net.to(feat_device)
    net.eval()

    X_feats = []
    y_labels = []

    with torch.no_grad():
        for _, row in train_df.iterrows():
            img_id = str(row["id_code"])
            y = int(row["diagnosis"])
            img_path = os.path.join(TRAIN_IMG_DIR, f"{img_id}.png")
            if not os.path.exists(img_path):
                continue
            img = Image.open(img_path).convert("RGB")
            x = transform(img).unsqueeze(0).to(feat_device)

            c_out, r_out, o_out = net(
                x
            )  # heads are random but deterministic; we use their joint 10-D signal
            feats10 = (
                torch.cat(
                    [
                        c_out.to(torch.float32),
                        r_out.to(torch.float32),
                        o_out.to(torch.float32),
                    ],
                    dim=1,
                )
                .detach()
                .cpu()
                .numpy()
                .reshape(-1)
            )  # (10,)

            X_feats.append(feats10)
            y_labels.append(y)

    net.to(device)
    net.eval()

    if len(X_feats) >= 120 and len(set(y_labels)) >= 2:
        X_feats = np.stack(X_feats, axis=0)
        y_labels = np.array(y_labels, dtype=np.float32)

        counts = np.bincount(y_labels.astype(int), minlength=5).astype(np.float64)
        inv = 1.0 / np.maximum(counts, 1.0)
        w = inv[y_labels.astype(int)]
        w = w / np.mean(w)
        w = np.clip(w, 0.5, 3.0).astype(np.float64)

        calibrator = make_pipeline(
            StandardScaler(with_mean=True, with_std=True),
            Ridge(alpha=2.0, solver="svd", random_state=42),
        )
        calibrator.fit(X_feats, y_labels, ridge__sample_weight=w)
        calibrator_kind = "ridge_regression_feats10"
        print(
            f"Calibrator (ridge regressor, 10-D features) fit on {len(y_labels)} train images."
        )
    else:
        print(
            "WARNING: Not enough train data to fit calibrator; will fallback to original heads."
        )



## === cell 5
submission = []
missing = 0

with torch.no_grad():
    for idx in test_ids:
        image_name = os.path.join(TEST_IMG_DIR, f"{idx}.png")
        if not os.path.exists(image_name):
            missing += 1
            submission.append([idx, 0])
            continue

        img = Image.open(image_name).convert("RGB")
        img_t = transform(img).unsqueeze(0).to(device)

        if weights_loaded or (calibrator is None):
            c_out, r_out, o_out = net(img_t)

            p_c = F.softmax(c_out.to(torch.float32), dim=1)  # (1, 5)
            p_r = regress2class_prob(r_out.squeeze(1))  # (1, 5)
            p_o = ordinal2class_prob(o_out)  # (1, 5)

            p = (p_c + p_r + p_o) / 3.0
            pred = torch.argmax(p, dim=1).item()
        else:
            c_out, r_out, o_out = net(img_t)
            feats10 = (
                torch.cat(
                    [
                        c_out.to(torch.float32),
                        r_out.to(torch.float32),
                        o_out.to(torch.float32),
                    ],
                    dim=1,
                )
                .detach()
                .float()
                .cpu()
                .numpy()
                .reshape(1, -1)
            )  # (1, 10)

            if calibrator_kind.startswith("ridge_regression"):
                y_hat = float(calibrator.predict(feats10)[0])
                y_hat = float(np.clip(y_hat, 0.0, 4.5))
                pred = int(
                    regress2class(torch.tensor([y_hat], dtype=torch.float32))[0].item()
                )
            else:
                proba = calibrator.predict_proba(feats10)[0]
                pred = int(np.argmax(proba))

        submission.append([idx, int(pred)])

submission = np.array(submission, dtype=object)
print("Missing test images:", missing, "out of", len(test_ids))



## === cell 6
df = pd.DataFrame(submission, columns=["id_code", "diagnosis"])
df["id_code"] = df["id_code"].astype(str)
df["diagnosis"] = df["diagnosis"].astype(int)

SAMPLE_SUB = "../input/aptos2019-blindness-detection/sample_submission.csv"
if not os.path.exists(SAMPLE_SUB):
    alt = "/kaggle/input/aptos2019-blindness-detection/sample_submission.csv"
    if os.path.exists(alt):
        SAMPLE_SUB = alt

if os.path.exists(SAMPLE_SUB):
    sample = pd.read_csv(SAMPLE_SUB)
    df = sample[["id_code"]].merge(df, on="id_code", how="left")
    df["diagnosis"] = df["diagnosis"].fillna(0).astype(int)

assert len(df) > 0, "Submission DataFrame is empty"
df.to_csv("submission.csv", index=False)
print(df.head())
print("Wrote submission.csv with shape:", df.shape)
