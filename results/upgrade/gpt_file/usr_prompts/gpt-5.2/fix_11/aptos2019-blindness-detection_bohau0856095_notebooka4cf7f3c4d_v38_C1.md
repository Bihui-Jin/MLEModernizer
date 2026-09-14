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

import timm
import cv2

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



## === cell 1
threshold = [0.75, 1.5, 2.5, 3.5]


def regress2class(out):
    out = out.view(-1)
    thr = torch.tensor(threshold, device=out.device, dtype=out.dtype)
    pred = (out[:, None] >= thr[None, :]).sum(dim=1)
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

        self.backbone = timm.models.tf_efficientnet_b4_ns(pretrained=True)
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


class fundus_clahe_circle(object):
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8), min_mask_coverage=0.10):
        self.clip_limit = float(clip_limit)
        self.tile_grid_size = tuple(tile_grid_size)
        self.min_mask_coverage = float(min_mask_coverage)

    def __call__(self, image):
        img = np.array(image)
        if img.ndim != 3 or img.shape[2] != 3:
            return image
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        h, w = bgr.shape[:2]
        r = int(0.48 * min(h, w))
        if r <= 0:
            return image
        cx, cy = w // 2, h // 2
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, (cx, cy), r, 255, thickness=-1)

        if (mask.mean() / 255.0) < self.min_mask_coverage:
            return image

        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size
        )
        l2 = clahe.apply(l)
        lab2 = cv2.merge([l2, a, b])
        bgr2 = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

        bgr2 = cv2.bitwise_and(bgr2, bgr2, mask=mask)

        rgb2 = cv2.cvtColor(bgr2, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb2)




## === cell 4
def _find_files_by_basename(search_roots, basenames, max_hits=200):
    if isinstance(basenames, str):
        basenames = [basenames]
    basenames = set(basenames)

    hits = []
    for root in search_roots:
        if not root or not os.path.exists(root):
            continue
        for dirpath, _, filenames in os.walk(root):
            fn_set = set(filenames)
            intersect = basenames.intersection(fn_set)
            if intersect:
                for bn in sorted(intersect):
                    hits.append(os.path.join(dirpath, bn))
                    if len(hits) >= max_hits:
                        return hits
    return hits


def _extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for k in [
            "state_dict",
            "model_state_dict",
            "model",
            "net",
            "weights",
            "ema",
            "model_ema",
        ]:
            if k in ckpt and isinstance(ckpt[k], dict) and len(ckpt[k]) > 0:
                return ckpt[k]
        for k in ["checkpoint", "module"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                inner = ckpt[k]
                for kk in [
                    "state_dict",
                    "model_state_dict",
                    "model",
                    "net",
                    "weights",
                    "ema",
                    "model_ema",
                ]:
                    if (
                        kk in inner
                        and isinstance(inner[kk], dict)
                        and len(inner[kk]) > 0
                    ):
                        return inner[kk]
        tensor_vals = [v for v in ckpt.values() if torch.is_tensor(v)]
        if len(tensor_vals) > 0:
            return ckpt
    return ckpt


def _strip_prefixes(state_dict):
    if not isinstance(state_dict, dict) or not state_dict:
        return state_dict

    prefixes = [
        "module.",
        "model.",
        "net.",
        "backbone.",
    ]
    changed = True
    sd = state_dict
    while changed:
        changed = False
        if not isinstance(sd, dict) or not sd:
            break
        first_key = next(iter(sd.keys()))
        for p in prefixes:
            if first_key.startswith(p):
                sd = {k.replace(p, "", 1): v for k, v in sd.items()}
                changed = True
                break
    return sd


def _load_checkpoint_safely(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = _extract_state_dict(ckpt)
    state_dict = _strip_prefixes(state_dict)

    try:
        model.load_state_dict(state_dict, strict=True)
        strict_used = True
    except RuntimeError as e:
        print(
            "WARNING: strict load failed, trying strict=False. Error was:\n",
            str(e)[:800],
        )
        incompatible = model.load_state_dict(state_dict, strict=False)
        try:
            missing = incompatible.missing_keys
            unexpected = incompatible.unexpected_keys
        except Exception:
            missing, unexpected = [], []
        print(
            "Loaded with strict=False. Missing keys:",
            len(missing),
            "Unexpected keys:",
            len(unexpected),
        )
        strict_used = False
    return model, strict_used


def _qwk(y_true, y_pred, n_classes=5):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    O = np.zeros((n_classes, n_classes), dtype=np.float64)
    for a, b in zip(y_true, y_pred):
        if 0 <= a < n_classes and 0 <= b < n_classes:
            O[a, b] += 1.0
    act_hist = np.bincount(y_true, minlength=n_classes).astype(np.float64)
    pred_hist = np.bincount(y_pred, minlength=n_classes).astype(np.float64)
    E = np.outer(act_hist, pred_hist)
    E = E / E.sum() * O.sum() if E.sum() > 0 else E

    W = np.zeros((n_classes, n_classes), dtype=np.float64)
    for i in range(n_classes):
        for j in range(n_classes):
            W[i, j] = ((i - j) ** 2) / ((n_classes - 1) ** 2)

    num = (W * O).sum()
    den = (W * E).sum()
    if den == 0:
        return 0.0
    return 1.0 - num / den


def _apply_thresholds(preds_cont, thr):
    preds_cont = np.asarray(preds_cont, dtype=np.float64)
    thr = list(thr)
    cls = np.zeros(preds_cont.shape[0], dtype=np.int64)
    for t in thr:
        cls += (preds_cont >= t).astype(np.int64)
    return np.clip(cls, 0, 4)


def _tune_thresholds(preds_cont, y_true, init_thr, iters=4, grid_points=41):
    preds_cont = np.asarray(preds_cont, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.int64)

    thr = np.array(init_thr, dtype=np.float64)
    thr.sort()
    best_thr = thr.copy()
    best_score = _qwk(y_true, _apply_thresholds(preds_cont, best_thr))

    lo = max(0.0, np.percentile(preds_cont, 1))
    hi = min(4.5, np.percentile(preds_cont, 99))
    lo, hi = float(lo), float(hi)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = 0.0, 4.5

    for _ in range(iters):
        for k in range(4):
            left = lo if k == 0 else best_thr[k - 1] + 1e-4
            right = hi if k == 3 else best_thr[k + 1] - 1e-4
            if right <= left:
                continue
            candidates = np.linspace(left, right, grid_points)
            local_best_t = best_thr[k]
            local_best_score = best_score
            for t in candidates:
                cand = best_thr.copy()
                cand[k] = t
                score = _qwk(y_true, _apply_thresholds(preds_cont, cand))
                if score > local_best_score:
                    local_best_score = score
                    local_best_t = t
            best_thr[k] = local_best_t
            best_score = local_best_score

    return best_thr.tolist(), float(best_score)




## === cell 5
TEST_CSV_PATH = "../input/aptos2019-blindness-detection/test.csv"
SAMPLE_SUB_PATH = "../input/aptos2019-blindness-detection/sample_submission.csv"
TEST_IMG_DIR = "../input/aptos2019-blindness-detection/test_images"

TRAIN_CSV_PATH = "../input/aptos2019-blindness-detection/train.csv"
TRAIN_IMG_DIR = "../input/aptos2019-blindness-detection/train_images"

test_df = pd.read_csv(TEST_CSV_PATH)
test_ids = np.squeeze(test_df["id_code"].values)

input_size = 380
transform = transforms.Compose(
    [
        trim(),
        cropTo4_3(),
        fundus_clahe_circle(clip_limit=2.0, tile_grid_size=(8, 8)),
        transforms.Resize((input_size * 3 // 4, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.384, 0.258, 0.174], std=[0.124, 0.089, 0.094]),
    ]
)

net = ThreeStage_Model()

weights_path = "../input/weights/B4_3stage_45epoch_CLAHE.pkl"
weights_basename = os.path.basename(weights_path)

candidate_basenames = {
    weights_basename,
    os.path.splitext(weights_basename)[0] + ".pth",
    os.path.splitext(weights_basename)[0] + ".pt",
    os.path.splitext(weights_basename)[0] + ".bin",
    os.path.splitext(weights_basename)[0] + ".ckpt",
    os.path.splitext(weights_basename)[0] + ".pkl",
    "B4_3stage_45epoch_CLAHE.pkl",
    "B4_3stage_45epoch_CLAHE.pth",
    "B4_3stage_45epoch_CLAHE.pt",
    "B4_3stage_45epoch_CLAHE.ckpt",
}

resolved_path = None
if os.path.exists(weights_path):
    resolved_path = weights_path
else:
    search_roots = [
        "../input",
        "/kaggle/input",
        "/kaggle/data/input",
        "/kaggle/working",
    ]
    hits = _find_files_by_basename(
        search_roots, list(candidate_basenames), max_hits=200
    )
    if hits:

        def _rank(p):
            p_low = p.lower()
            bonus = 0
            if "weight" in p_low:
                bonus -= 5
            if "ckpt" in p_low or "checkpoint" in p_low:
                bonus -= 3
            if "model" in p_low:
                bonus -= 1
            return (bonus, len(p), p)

        hits = sorted(hits, key=_rank)
        resolved_path = hits[0]

if resolved_path and os.path.exists(resolved_path):
    net, strict_used = _load_checkpoint_safely(net, resolved_path)
    print("Loaded weights:", resolved_path, "| strict:", strict_used)
else:
    print(
        f"WARNING: weights file not found (looked for {sorted(candidate_basenames)} under ../input and /kaggle/*). "
        "Proceeding with ImageNet-pretrained backbone (better than random, but below target if the intended DR checkpoint is missing)."
    )

net = net.to(device)
net.eval()

use_cuda = torch.cuda.is_available() and device.startswith("cuda")
if use_cuda:
    net = net.to(memory_format=torch.channels_last)



## === cell 6
train_df = pd.read_csv(TRAIN_CSV_PATH)
train_ids = train_df["id_code"].values
train_y = train_df["diagnosis"].values.astype(int)

K = 5
rng = np.random.RandomState(42)
perm = rng.permutation(len(train_ids))
fold_id = np.empty(len(train_ids), dtype=np.int64)
for i, p in enumerate(perm):
    fold_id[p] = i % K

oof_preds_cont = np.zeros(len(train_ids), dtype=np.float64)
missing_train = 0

t0 = time.time()
with torch.no_grad():
    for i in range(len(train_ids)):
        if i % 200 == 0:
            print("OOF inference", i, "of", len(train_ids))
        idx = train_ids[i]
        image_name = f"{TRAIN_IMG_DIR}/{idx}.png"
        if not os.path.exists(image_name):
            missing_train += 1
            oof_preds_cont[i] = 0.0
            continue

        img = Image.open(image_name).convert("RGB")
        img = transform(img).unsqueeze(0)

        if use_cuda:
            img = img.to(device, non_blocking=True).contiguous(
                memory_format=torch.channels_last
            )
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                out = net(img, final=True)
        else:
            img = img.to(device)
            out = net(img, final=True)

        oof_preds_cont[i] = float(out.detach().float().cpu().view(-1)[0].item())

print(
    "Train inference done. Missing train images:",
    missing_train,
    "time(s):",
    round(time.time() - t0, 2),
)

base_pred_cls = _apply_thresholds(oof_preds_cont, threshold)
base_qwk = _qwk(train_y, base_pred_cls)
print("Base thresholds:", threshold, "Train(QWK, in-sample) :", round(base_qwk, 6))

oof_pred_cls_true = np.zeros(len(train_ids), dtype=np.int64)
fold_thr = []
for f in range(K):
    tr_idx = np.where(fold_id != f)[0]
    va_idx = np.where(fold_id == f)[0]

    thr_f, qwk_tr = _tune_thresholds(
        oof_preds_cont[tr_idx], train_y[tr_idx], threshold, iters=5, grid_points=41
    )
    fold_thr.append(thr_f)
    oof_pred_cls_true[va_idx] = _apply_thresholds(oof_preds_cont[va_idx], thr_f)

oof_qwk_true = _qwk(train_y, oof_pred_cls_true)
print("True OOF QWK (foldwise tuned):", round(oof_qwk_true, 6))

tuned_thr, tuned_qwk = _tune_thresholds(
    oof_preds_cont, train_y, threshold, iters=5, grid_points=41
)
print(
    "Tuned thresholds (global):",
    [round(x, 6) for x in tuned_thr],
    "Train tuned QWK (in-sample on OOF-cont):",
    round(tuned_qwk, 6),
)

threshold = tuned_thr



## === cell 7
submission = []
missing_images = 0

with torch.no_grad():
    for i, idx in enumerate(test_ids):
        if i % 50 == 0:
            print("Predicting", i, "of", len(test_ids))

        image_name = f"{TEST_IMG_DIR}/{idx}.png"
        if not os.path.exists(image_name):
            missing_images += 1
            submission.append([idx, 0])
            continue

        img = Image.open(image_name).convert("RGB")
        img = transform(img).unsqueeze(0)

        if use_cuda:
            img = img.to(device, non_blocking=True).contiguous(
                memory_format=torch.channels_last
            )
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                final_out = net(img, final=True)
        else:
            img = img.to(device)
            final_out = net(img, final=True)

        pred = regress2class(final_out.view(-1))
        submission.append([idx, int(pred.item())])

if missing_images:
    print("WARNING: missing images:", missing_images)

submission = np.array(submission, dtype=object)



## === cell 8
df = pd.DataFrame(submission, columns=["id_code", "diagnosis"])
df["diagnosis"] = df["diagnosis"].astype(int)

sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
df = sample_sub[["id_code"]].merge(df, on="id_code", how="left")
df["diagnosis"] = df["diagnosis"].fillna(0).astype(int)

out_path = "submission.csv"
df.to_csv(out_path, index=False)
print("Wrote:", out_path, "rows:", len(df))
print(df.head())
