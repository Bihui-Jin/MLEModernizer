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
TEST_CSV = "../input/aptos2019-blindness-detection/test.csv"
TEST_IMG_DIR = "../input/aptos2019-blindness-detection/test_images"
TRAIN_CSV = "../input/aptos2019-blindness-detection/train.csv"
TRAIN_IMG_DIR = "../input/aptos2019-blindness-detection/train_images"

test_df = pd.read_csv(TEST_CSV)
test_ids = test_df["id_code"].astype(str).values

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


def find_checkpoint():
    roots = [
        "../input",
        "/kaggle/input",
    ]

    candidates = []
    for root in roots:
        candidates += [
            os.path.join(root, "weights", "B4_3stage_39epoch_CLAHE.pkl"),
            os.path.join(
                root, "aptos2019-blindness-detection", "B4_3stage_39epoch_CLAHE.pkl"
            ),
        ]

    for root in roots:
        candidates += glob.glob(
            os.path.join(root, "**", "B4_3stage_39epoch_CLAHE.*"), recursive=True
        )

    keywords = ("b4", "efficientnet", "3stage", "stage", "aptos", "blind", "clahe")
    for root in roots:
        for ext in ("*.pkl", "*.pth", "*.pt", "*.bin"):
            for p in glob.glob(os.path.join(root, "**", ext), recursive=True):
                bn = os.path.basename(p).lower()
                if any(k in bn for k in keywords):
                    candidates.append(p)

    seen = set()
    uniq = []
    for p in candidates:
        if p not in seen:
            seen.add(p)
            uniq.append(p)

    for p in uniq:
        if os.path.isfile(p):
            return p
    return None


def _extract_state_dict(obj):
    if isinstance(obj, dict):
        for key in ("state_dict", "model", "net", "ema_state_dict", "model_state_dict"):
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
    except Exception as e1:
        try:
            missing, unexpected = model.load_state_dict(state, strict=False)
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
            missing, unexpected = model.load_state_dict(filtered, strict=False)
            return (
                True,
                f"shape-matched subset (loaded={len(filtered)}, missing={len(missing)}, unexpected={len(unexpected)})",
            )


ckpt_path = find_checkpoint()
weights_loaded = False
load_msg = "none"

if ckpt_path is not None:
    state = torch.load(ckpt_path, map_location="cpu")
    state = _sanitize_state_dict(state)
    if state is not None:
        weights_loaded, load_msg = _load_state_dict_best_effort(net, state)
        if not weights_loaded:
            print(
                f"WARNING: checkpoint found at {ckpt_path} but could not be loaded; using random weights. ({load_msg})"
            )
    else:
        print(
            f"WARNING: checkpoint found at {ckpt_path} but not a state_dict-like object; using random weights."
        )
else:
    print(
        "WARNING: no suitable checkpoint found under ../input or /kaggle/input. "
        "Proceeding with random weights; will try a small threshold calibration fallback from train.csv."
    )

net = net.to(device)
net.eval()

print(
    f"Device: {device} | Weights loaded: {weights_loaded} | Load mode: {load_msg} | Checkpoint: {ckpt_path}"
)




## === cell 5
def calibrate_thresholds_to_label_distribution(train_csv_path=TRAIN_CSV):
    global threshold
    if not os.path.isfile(train_csv_path):
        return
    tr = pd.read_csv(train_csv_path)
    if "diagnosis" not in tr.columns:
        return
    y = tr["diagnosis"].astype(int).values
    counts = np.bincount(y, minlength=5).astype(float)
    probs = counts / max(1.0, counts.sum())
    cum = np.cumsum(probs)  # P(y<=k)
    new_thr = [
        float(cum[0] * 4.5),
        float(cum[1] * 4.5),
        float(cum[2] * 4.5),
        float(cum[3] * 4.5),
    ]
    new_thr = np.clip(new_thr, 0.05, 4.45)
    new_thr = np.maximum.accumulate(new_thr).tolist()
    threshold = new_thr
    print(f"Calibrated thresholds (weights missing) to train distribution: {threshold}")


if not weights_loaded:
    calibrate_thresholds_to_label_distribution()

submission = []
for i, idx in enumerate(test_ids):
    image_name = f"{TEST_IMG_DIR}/{idx}.png"
    if not os.path.isfile(image_name):
        raise FileNotFoundError(f"Missing test image: {image_name}")

    img = Image.open(image_name).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)

    _, r_out, _ = net(img)
    pred = regress2class(r_out.data.squeeze(1))
    submission.append([idx, int(pred.item())])

submission = np.array(submission, dtype=object)



## === cell 6
df = pd.DataFrame(submission, columns=["id_code", "diagnosis"])

df = df.merge(test_df[["id_code"]], on="id_code", how="right")
df["diagnosis"] = df["diagnosis"].fillna(0).astype(int)

out_path = "submission.csv"
df.to_csv(out_path, index=False)

print(df.head())
print(f"Wrote {out_path} with shape={df.shape} and columns={list(df.columns)}")
print(df["diagnosis"].value_counts(dropna=False).sort_index())
