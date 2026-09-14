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

from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold
import timm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



## === cell 1
threshold = [0.75, 1.5, 2.5, 3.5]


def regress2class(out):
    if out.ndim > 1:
        out = out.squeeze(1)
    prediction = torch.zeros(out.size(0), device="cpu")
    out_cpu = out.detach().to("cpu")
    for i in range(4):
        prediction += (out_cpu >= threshold[i]).to(torch.float32)
    return prediction


def ordinal2class_prob(out):
    dev = out.device
    pred_prob = torch.zeros(out.size(0), 5, device=dev)
    pred_prob[:, 0] = (1 - out[:, 0]).squeeze()
    pred_prob[:, 1] = (out[:, 0] * (1 - out[:, 1])).squeeze()
    pred_prob[:, 2] = (out[:, 1] * (1 - out[:, 2])).squeeze()
    pred_prob[:, 3] = (out[:, 2] * (1 - out[:, 3])).squeeze()
    pred_prob[:, 4] = out[:, 3].squeeze()
    return F.softmax(pred_prob, dim=1)


def regress2class_prob(out):
    if out.ndim > 1:
        out = out.squeeze(1)
    dev = out.device
    pred_prob = torch.zeros((out.size(0), 5), device=dev, dtype=torch.float32)
    out_det = out.detach()
    for i in range(out_det.size(0)):
        oi = float(out_det[i].item())
        if oi < 4.0:
            l1 = int(math.floor(oi))
            l2 = int(math.ceil(oi))
            pred_prob[i, l1] = 1.0 - (oi - l1)
            pred_prob[i, l2] = 1.0 - (l2 - oi)
        else:
            pred_prob[i, 4] = 1.0
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
DATA_DIR = "/kaggle/input/aptos2019-blindness-detection"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")

if not os.path.exists(TEST_CSV):
    DATA_DIR = "/kaggle/data/aptos2019-blindness-detection"
    TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
    TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images")
    TEST_CSV = os.path.join(DATA_DIR, "test.csv")
    TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")

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

net = ThreeStage_Model().to(device)

SEARCH_ROOTS = ["/kaggle/input", "/kaggle/data", "../input"]
SEARCH_NAMES_EXACT = {
    "B4_3stage_5epoch_finetune.pkl",
    "B4_3stage_5epoch_finetune.pth",
    "B4_3stage_5epoch_finetune.pt",
}
SEARCH_EXTS = {".pth", ".pt", ".pkl", ".bin"}
SEARCH_SUBSTRINGS = ["3stage", "b4", "efficientnet_b4", "finetune"]

WEIGHTS_CANDIDATES = []
for p in [
    "../input/weights/B4_3stage_5epoch_finetune.pkl",
    "../input/weights/B4_3stage_5epoch_finetune.pth",
    "../input/weights/B4_3stage_5epoch_finetune.pt",
    "/kaggle/input/weights/B4_3stage_5epoch_finetune.pkl",
    "/kaggle/input/weights/B4_3stage_5epoch_finetune.pth",
    "/kaggle/input/weights/B4_3stage_5epoch_finetune.pt",
]:
    WEIGHTS_CANDIDATES.append(p)

for root in SEARCH_ROOTS:
    if os.path.isdir(root):
        for dirpath, dirnames, filenames in os.walk(root):
            for fn in filenames:
                if fn in SEARCH_NAMES_EXACT:
                    WEIGHTS_CANDIDATES.append(os.path.join(dirpath, fn))
                else:
                    ext = os.path.splitext(fn)[1].lower()
                    if ext in SEARCH_EXTS:
                        low = fn.lower()
                        if any(s in low for s in SEARCH_SUBSTRINGS):
                            WEIGHTS_CANDIDATES.append(os.path.join(dirpath, fn))

_seen = set()
WEIGHTS_CANDIDATES = [p for p in WEIGHTS_CANDIDATES if not (p in _seen or _seen.add(p))]
weights_path = next((p for p in WEIGHTS_CANDIDATES if os.path.exists(p)), None)


def _extract_state_dict(state_obj):
    if isinstance(state_obj, dict):
        for k in ["ema_state_dict", "model_ema", "ema", "state_dict_ema"]:
            if k in state_obj and isinstance(state_obj[k], dict):
                state_obj = state_obj[k]
                break

    if isinstance(state_obj, dict):
        for k in [
            "state_dict",
            "model_state_dict",
            "model",
            "net",
            "encoder",
            "weights",
        ]:
            if k in state_obj and isinstance(state_obj[k], dict):
                state_obj = state_obj[k]
                break

    if not isinstance(state_obj, dict) or len(state_obj) == 0:
        return None

    def _looks_like_state_dict(d):
        if not isinstance(d, dict) or len(d) == 0:
            return False
        for v in d.values():
            if isinstance(v, (torch.Tensor, torch.nn.Parameter)):
                return True
        return False

    if not _looks_like_state_dict(state_obj):
        for v in state_obj.values():
            if isinstance(v, dict) and _looks_like_state_dict(v):
                state_obj = v
                break

    if not _looks_like_state_dict(state_obj):
        return None

    def _strip_prefix(d, prefix):
        if any(isinstance(k, str) and k.startswith(prefix) for k in d.keys()):
            return {k.replace(prefix, "", 1): v for k, v in d.items()}
        return d

    state_obj = _strip_prefix(state_obj, "module.")
    state_obj = _strip_prefix(state_obj, "model.")
    state_obj = _strip_prefix(state_obj, "net.")

    if not all(isinstance(k, str) for k in state_obj.keys()):
        return None
    return state_obj


has_weights = False
if weights_path is not None:
    try:
        state = torch.load(weights_path, map_location="cpu")
        state = _extract_state_dict(state)
        if state is None:
            raise ValueError(
                "Unsupported checkpoint format (no usable state_dict found)."
            )

        try:
            net.load_state_dict(state, strict=True)
            has_weights = True
        except RuntimeError:
            missing, unexpected = net.load_state_dict(state, strict=False)
            print("Warning: strict=True load failed, used strict=False instead.")
            print("Missing keys (first 30):", missing[:30])
            print("Unexpected keys (first 30):", unexpected[:30])
            has_weights = True
    except Exception as e:
        print(f"Warning: failed to load weights from {weights_path}: {e}")
        has_weights = False

print(
    f"Device: {device} | weights_loaded: {has_weights} | weights_path: {weights_path}"
)



## === cell 5
from torch.utils.data import Dataset, DataLoader


class APTOSDataset(Dataset):
    def __init__(self, df, img_dir, transform, with_labels=True):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.with_labels = with_labels

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        idx = str(row["id_code"])
        image_name = os.path.join(self.img_dir, f"{idx}.png")
        img = Image.open(image_name).convert("RGB")
        img = self.transform(img)
        if self.with_labels:
            y = int(row["diagnosis"])
            return img, torch.tensor([y], dtype=torch.float32)
        return img, idx


def _calibrate_thresholds(val_y_true, val_r_pred):
    global threshold

    y = np.asarray(val_y_true, dtype=int)
    r = np.asarray(val_r_pred, dtype=float)

    def qwk_for(thr):
        thr = list(thr)
        pred = np.zeros_like(r, dtype=int)
        for t in thr:
            pred += (r >= t).astype(np.int32)
        pred = np.clip(pred, 0, 4)
        return cohen_kappa_score(y, pred, weights="quadratic")

    best_thr = threshold[:]
    best = qwk_for(best_thr)

    step = 0.05
    for _ in range(6):
        improved = False
        for j in range(4):
            for delta in (-step, step):
                cand = best_thr[:]
                cand[j] = float(cand[j] + delta)
                cand = sorted(cand)
                cand = [max(0.0, min(4.5, c)) for c in cand]
                sc = qwk_for(cand)
                if sc > best:
                    best = sc
                    best_thr = cand
                    improved = True
        if not improved:
            step *= 0.5

    threshold = best_thr
    print("Threshold calibration done.")
    print("New threshold:", threshold, "| OOF QWK:", best)


def _train_one_model(net, dl_tr, epochs=3, lr=1e-4):
    net.train()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.SmoothL1Loss(beta=0.5)
    start = time.time()

    for ep in range(epochs):
        running = 0.0
        n = 0
        for xb, yb in dl_tr:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            _, r_out, _ = net(xb)
            loss = loss_fn(r_out, yb)
            loss.backward()
            opt.step()
            running += float(loss.item()) * xb.size(0)
            n += xb.size(0)

        print(
            f"Epoch {ep+1}/{epochs} | loss={running/max(n,1):.4f} | elapsed={time.time()-start:.1f}s"
        )


def _oof_threshold_calibration(train_df, n_splits=3):
    y = train_df["diagnosis"].astype(int).values
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED + 123)

    oof_pred = np.zeros(len(train_df), dtype=np.float32)

    bs = 16 if device.type == "cuda" else 6
    num_workers = 2 if os.name != "nt" else 0

    for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(y)), y), 1):
        print(f"OOF fold {fold}/{n_splits} | train={len(tr_idx)} val={len(va_idx)}")
        net_fold = ThreeStage_Model().to(device)

        ds_tr = APTOSDataset(
            train_df.iloc[tr_idx].reset_index(drop=True),
            TRAIN_IMG_DIR,
            transform,
            with_labels=True,
        )
        ds_va = APTOSDataset(
            train_df.iloc[va_idx].reset_index(drop=True),
            TRAIN_IMG_DIR,
            transform,
            with_labels=True,
        )

        dl_tr = DataLoader(
            ds_tr,
            batch_size=bs,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
        )
        dl_va = DataLoader(
            ds_va,
            batch_size=bs,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
        )

        _train_one_model(net_fold, dl_tr, epochs=3, lr=1e-4)

        net_fold.eval()
        preds = []
        with torch.no_grad():
            for xb, _yb in dl_va:
                xb = xb.to(device, non_blocking=True)
                _, r_out, _ = net_fold(xb)
                preds.append(r_out.detach().cpu().numpy().reshape(-1))
        preds = np.concatenate(preds, axis=0).astype(np.float32)
        oof_pred[va_idx] = preds

        del net_fold
        if device.type == "cuda":
            torch.cuda.empty_cache()

    _calibrate_thresholds(y, oof_pred)


def _train_if_needed():
    global has_weights

    if has_weights:
        return

    if not (os.path.exists(TRAIN_CSV) and os.path.isdir(TRAIN_IMG_DIR)):
        print("Warning: train data not found; falling back to all-zero predictions.")
        return

    train_df = pd.read_csv(TRAIN_CSV)
    train_df["diagnosis"] = train_df["diagnosis"].astype(int)
    train_df = train_df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    _oof_threshold_calibration(train_df, n_splits=3)

    ds_tr_full = APTOSDataset(train_df, TRAIN_IMG_DIR, transform, with_labels=True)

    bs = 18 if device.type == "cuda" else 6
    num_workers = 2 if os.name != "nt" else 0
    dl_tr_full = DataLoader(
        ds_tr_full,
        batch_size=bs,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    _train_one_model(net, dl_tr_full, epochs=3, lr=1e-4)

    has_weights = True  # trained weights in-memory


_train_if_needed()



## === cell 6
test_ds = APTOSDataset(
    test_df.assign(diagnosis=0), TEST_IMG_DIR, transform, with_labels=False
)
bs_inf = 24 if device.type == "cuda" else 8
num_workers = 2 if os.name != "nt" else 0
test_dl = DataLoader(
    test_ds,
    batch_size=bs_inf,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=(device.type == "cuda"),
)

submission_rows = []
net.eval()
with torch.no_grad():
    if not has_weights:
        for idx in test_ids:
            submission_rows.append([idx, 0])
    else:
        for xb, idxs in test_dl:
            xb = xb.to(device, non_blocking=True)
            _, r_out, _ = net(xb)
            preds = regress2class(r_out).numpy().astype(int).tolist()
            for idx, p in zip(idxs, preds):
                submission_rows.append([str(idx), int(p)])

pred_map = {rid: diag for rid, diag in submission_rows}
submission_rows = [[rid, int(pred_map.get(rid, 0))] for rid in test_ids]
submission = np.array(submission_rows, dtype=object)



## === cell 7
df = pd.DataFrame(submission, columns=["id_code", "diagnosis"])

df["id_code"] = df["id_code"].astype(str)
df["diagnosis"] = df["diagnosis"].astype(int).clip(0, 4)

assert (
    len(df) == len(test_ids) and len(df) > 0
), "Submission DataFrame should not be empty and must match test size."
df.to_csv("submission.csv", index=False)
print(df.head())
print("Wrote submission.csv with shape:", df.shape)
print("Final thresholds used:", threshold)
