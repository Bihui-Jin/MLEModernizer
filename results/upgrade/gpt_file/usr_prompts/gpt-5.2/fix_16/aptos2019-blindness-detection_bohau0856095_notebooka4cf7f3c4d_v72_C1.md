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
import hashlib
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as transforms
from torchvision.transforms import functional as FT
from PIL import Image, ImageChops, ImageFile

from sklearn.metrics import cohen_kappa_score
import timm

device = "cuda:0" if torch.cuda.is_available() else "cpu"

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

ImageFile.LOAD_TRUNCATED_IMAGES = True



## === cell 1
threshold = [0.75, 1.5, 2.5, 3.5]

_thr_cache = {}  # (device, dtype) -> tensor


def _get_thr_tensor(out):
    key = (out.device, out.dtype)
    t = _thr_cache.get(key)
    if t is None:
        t = torch.tensor(threshold, device=out.device, dtype=out.dtype).view(1, -1)
        _thr_cache[key] = t
    return t


def regress2class(out):
    out = out.view(-1)
    thr = _get_thr_tensor(out)
    prediction = (out.view(-1, 1) >= thr).sum(dim=1).to(torch.int64)
    return prediction.cpu()


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
    pred_prob = torch.zeros((out.size(0), 5), device=out.device, dtype=out.dtype)

    lt4 = out < 4.0
    if lt4.any():
        v = out[lt4]
        l1 = torch.floor(v).to(torch.long).clamp(0, 4)
        l2 = torch.ceil(v).to(torch.long).clamp(0, 4)
        w1 = 1.0 - (v - l1.to(v.dtype))
        w2 = 1.0 - (l2.to(v.dtype) - v)
        idx = torch.nonzero(lt4, as_tuple=False).squeeze(1)
        pred_prob[idx, l1] = w1
        pred_prob[idx, l2] = w2

    ge4 = ~lt4
    if ge4.any():
        pred_prob[ge4, 4] = 1.0

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
BASE_INPUT_CANDIDATES = [
    "../input/aptos2019-blindness-detection",
    "/kaggle/input/aptos2019-blindness-detection",
    "../input",
    "/kaggle/input",
    "/kaggle/data/aptos2019-blindness-detection",
    "/kaggle/data",
]


def resolve_base_input():
    for base in BASE_INPUT_CANDIDATES:
        if os.path.isfile(os.path.join(base, "test.csv")) and os.path.isdir(
            os.path.join(base, "test_images")
        ):
            return base
    for base in ["../input", "/kaggle/input", "/kaggle/data"]:
        if os.path.isdir(base):
            hit = glob.glob(
                os.path.join(base, "**", "aptos2019-blindness-detection"),
                recursive=True,
            )
            for h in hit:
                if os.path.isfile(os.path.join(h, "test.csv")) and os.path.isdir(
                    os.path.join(h, "test_images")
                ):
                    return h
    raise FileNotFoundError(
        "Could not resolve dataset base path containing test.csv and test_images/."
    )


BASE_INPUT = resolve_base_input()
TRAIN_CSV = os.path.join(BASE_INPUT, "train.csv")
TEST_CSV = os.path.join(BASE_INPUT, "test.csv")
TRAIN_IMG_DIR = os.path.join(BASE_INPUT, "train_images")
TEST_IMG_DIR = os.path.join(BASE_INPUT, "test_images")

train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)

test_ids = test_df["id_code"].astype(str).values

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

train_transform = transforms.Compose(
    [
        trim(),
        cropTo4_3(),
        transforms.Resize((input_size * 3 // 4, input_size)),
        photometric_distort(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.384, 0.258, 0.174], std=[0.124, 0.089, 0.094]),
    ]
)

CACHE_DIR = "/kaggle/working/aptos_cache_tensors"
os.makedirs(CACHE_DIR, exist_ok=True)


def _transform_fingerprint(transform_obj, input_size_):
    s = f"{repr(transform_obj)}|input_size={input_size_}"
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:12]


VAL_FP = _transform_fingerprint(transform, input_size)
TRN_FP = _transform_fingerprint(train_transform, input_size)




## === cell 5
class TrainDataset(Dataset):
    def __init__(self, df, img_dir, transform, cache_tag=None):
        df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.cache_tag = cache_tag  # e.g., "val_<fp>" or "train_<fp>"
        self.id_codes = df["id_code"].astype(str).values
        self.y = df["diagnosis"].astype(np.float32).values

    def __len__(self):
        return self.id_codes.shape[0]

    def _cache_path(self, id_code):
        if self.cache_tag is None:
            return None
        return os.path.join(CACHE_DIR, f"{self.cache_tag}__{str(id_code)}.pt")

    def __getitem__(self, i):
        id_code = self.id_codes[i]
        y = float(self.y[i])

        cpath = self._cache_path(id_code)
        if cpath is not None and os.path.isfile(cpath):
            img = torch.load(cpath, map_location="cpu", weights_only=False)
            return img, torch.tensor([y], dtype=torch.float32), id_code

        image_name = os.path.join(self.img_dir, f"{id_code}.png")
        with Image.open(image_name) as im:
            img = im.convert("RGB")
            img = self.transform(img)

        if cpath is not None:
            tmp = cpath + ".tmp"
            torch.save(img, tmp)
            os.replace(tmp, cpath)

        return img, torch.tensor([y], dtype=torch.float32), id_code


def find_weight_file(base_input):
    explicit = [
        "../input/weights/B4_3stage_5epoch_finetune512.pkl",
        "../input/B4_3stage_5epoch_finetune512.pkl",
        "/kaggle/input/weights/B4_3stage_5epoch_finetune512.pkl",
        "/kaggle/input/B4_3stage_5epoch_finetune512.pkl",
        "/kaggle/data/weights/B4_3stage_5epoch_finetune512.pkl",
        "/kaggle/data/B4_3stage_5epoch_finetune512.pkl",
        os.path.join(base_input, "B4_3stage_5epoch_finetune512.pkl"),
        os.path.join(base_input, "weights", "B4_3stage_5epoch_finetune512.pkl"),
    ]
    for c in explicit:
        if os.path.isfile(c):
            return c

    search_roots = [
        "../input",
        "/kaggle/input",
        "/kaggle/data",
        base_input,
        "/kaggle",
    ]
    patterns_rel = [
        "**/B4_3stage_5epoch_finetune512.pkl",
        "**/*3stage*finetune*512*.pkl",
        "**/*3stage*.pkl",
        "**/*3stage*.pth",
        "**/*3stage*.pt",
        "**/*efficientnet*b4*.pth",
        "**/*efficientnet*b4*.pt",
        "**/*b4*3stage*.pth",
        "**/*b4*3stage*.pt",
    ]

    hits_all = []
    for root in search_roots:
        if not os.path.isdir(root):
            continue
        for pr in patterns_rel:
            hits_all.extend(glob.glob(os.path.join(root, pr), recursive=True))

    hits_all = sorted(set(hits_all), key=lambda p: (len(p.split(os.sep)), len(p), p))
    return hits_all[0] if len(hits_all) > 0 else None


def extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for k in ["state_dict", "model_state_dict", "model", "net", "weights"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
    return ckpt


_MODEL_KEYS = set(ThreeStage_Model().state_dict().keys())


def clean_state_dict_keys(state):
    if not isinstance(state, dict) or len(state) == 0:
        return state

    prefixes = [
        "module.",
        "model.",
        "net.",
        "backbone.",
    ]

    def strip_once(sd, pref):
        return {k[len(pref) :] if k.startswith(pref) else k: v for k, v in sd.items()}

    best_state = state
    best_overlap = len(set(state.keys()) & _MODEL_KEYS)

    for pref in prefixes:
        stripped = strip_once(state, pref)
        overlap = len(set(stripped.keys()) & _MODEL_KEYS)
        if overlap > best_overlap:
            best_overlap = overlap
            best_state = stripped

    improved = True
    while improved:
        improved = False
        for pref in prefixes:
            stripped = strip_once(best_state, pref)
            overlap = len(set(stripped.keys()) & _MODEL_KEYS)
            if overlap > best_overlap:
                best_overlap = overlap
                best_state = stripped
                improved = True

    return best_state


def apply_thresholds_np(preds, thr):
    thr = np.asarray(list(thr), dtype=np.float32).reshape(1, -1)
    preds = np.asarray(preds, dtype=np.float32).reshape(-1, 1)
    return (preds >= thr).sum(axis=1).astype(np.int64)


def tune_thresholds_for_qwk(y_true_int, y_pred_float, init_thr, iters=2):
    y_true_int = np.asarray(y_true_int, dtype=np.int64).reshape(-1)
    y_pred_float = np.asarray(y_pred_float, dtype=np.float32).reshape(-1)

    thr = np.array(init_thr, dtype=np.float32)
    thr = np.clip(thr, 0.0, 4.5)
    thr.sort()

    def score(thr_):
        y_hat = apply_thresholds_np(y_pred_float, thr_)
        return cohen_kappa_score(y_true_int, y_hat, weights="quadratic")

    best = score(thr)

    for _ in range(int(iters)):
        for j in range(4):
            lo = 0.0 if j == 0 else float(thr[j - 1] + 1e-3)
            hi = 4.5 if j == 3 else float(thr[j + 1] - 1e-3)
            if hi <= lo:
                continue

            grid = np.linspace(lo, hi, 41, dtype=np.float32)
            local_best = best
            local_thr = float(thr[j])

            for v in grid:
                cand = thr.copy()
                cand[j] = v
                cand.sort()
                s = score(cand)
                if s > local_best:
                    local_best = s
                    local_thr = float(v)

            thr[j] = local_thr
            thr.sort()
            best = local_best

    return thr.tolist(), float(best)


def tune_thresholds_for_qwk_scipy_multistart(
    y_true_int, y_pred_float, init_thr, seeds=(0, 1, 2, 3, 4)
):
    return None, None


def oof_threshold_calibration_true_oof(
    base_state_dict,
    train_df_,
    train_img_dir_,
    base_threshold,
    n_splits=3,
    batch_size=8,
    seed=42,
    fold_epochs=3,
):
    from sklearn.model_selection import StratifiedKFold

    y_all = train_df_["diagnosis"].astype(int).values
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    thrs = []
    kappas = []

    _nw = min(8, max(2, (os.cpu_count() or 4) // 2))
    _dl_kwargs = dict(
        num_workers=_nw,
        pin_memory=(device != "cpu"),
        persistent_workers=(_nw > 0),
        prefetch_factor=4 if _nw > 0 else None,
    )
    _dl_kwargs = {k: v for k, v in _dl_kwargs.items() if v is not None}

    for fold, (tr_idx, va_idx) in enumerate(
        skf.split(np.zeros(len(y_all)), y_all), start=1
    ):
        tr_df = train_df_.iloc[tr_idx]
        va_df = train_df_.iloc[va_idx]

        tr_ds_fold = TrainDataset(
            tr_df,
            train_img_dir_,
            train_transform,
            cache_tag=f"oof_tr_f{fold}_{TRN_FP}",
        )
        va_ds_fold = TrainDataset(
            va_df,
            train_img_dir_,
            transform,
            cache_tag=f"oof_va_f{fold}_{VAL_FP}",
        )

        tr_dl_fold = DataLoader(
            tr_ds_fold,
            batch_size=batch_size,
            shuffle=True,
            **_dl_kwargs,
        )
        va_dl_fold = DataLoader(
            va_ds_fold,
            batch_size=batch_size,
            shuffle=False,
            **_dl_kwargs,
        )

        net_f = ThreeStage_Model().to(device)
        if base_state_dict is not None:
            try:
                net_f.load_state_dict(base_state_dict, strict=True)
            except Exception:
                net_f.load_state_dict(base_state_dict, strict=False)

        optimizer = torch.optim.AdamW(net_f.parameters(), lr=1e-4, weight_decay=1e-4)
        loss_fn = nn.MSELoss()

        net_f.train()
        for ep in range(int(fold_epochs)):
            for x, y, _ in tr_dl_fold:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                pred = net_f(x, final=True)
                loss = loss_fn(pred, y)
                loss.backward()
                optimizer.step()

        net_f.eval()
        preds = []
        ytrue = []
        with torch.no_grad():
            for x, y, _ in va_dl_fold:
                x = x.to(device, non_blocking=True)
                p = net_f(x, final=True).squeeze(1).detach().cpu().numpy()
                preds.append(p)
                ytrue.append(y.squeeze(1).numpy().astype(int))
        preds = np.concatenate(preds)
        ytrue = np.concatenate(ytrue)

        thr0 = list(base_threshold)
        old_k = cohen_kappa_score(
            ytrue, apply_thresholds_np(preds, thr0), weights="quadratic"
        )

        sc_thr, sc_kappa = tune_thresholds_for_qwk_scipy_multistart(
            ytrue, preds, thr0, seeds=(0, 1, 2, 3, 4)
        )
        if sc_thr is None:
            sc_thr, sc_kappa = tune_thresholds_for_qwk(ytrue, preds, thr0, iters=2)

        sc_thr, sc_kappa = tune_thresholds_for_qwk(ytrue, preds, sc_thr, iters=1)

        thrs.append(np.asarray(sc_thr, dtype=np.float64))
        kappas.append(float(sc_kappa))

        print(
            f"True-OOF fold {fold}/{n_splits}: base_qwk={old_k:.4f} tuned_qwk={sc_kappa:.4f} thr={sc_thr}"
        )

        del net_f
        if device != "cpu":
            torch.cuda.empty_cache()

    thr_avg = np.mean(np.stack(thrs, axis=0), axis=0)
    thr_avg = np.clip(thr_avg, 0.0, 4.5)
    thr_avg.sort()
    return thr_avg.tolist(), float(np.mean(kappas))


weight_path = find_weight_file(BASE_INPUT)

net = ThreeStage_Model().to(device)

loaded = False
load_error = None
_loaded_state_for_folds = None
if weight_path is not None:
    try:
        ckpt = torch.load(weight_path, map_location="cpu", weights_only=False)
        state = extract_state_dict(ckpt)
        state = clean_state_dict_keys(state)
        try:
            net.load_state_dict(state, strict=True)
            loaded = True
        except RuntimeError as e:
            load_error = str(e)
            net.load_state_dict(state, strict=False)
            loaded = True
        if loaded:
            _loaded_state_for_folds = {
                k: v.detach().cpu() for k, v in net.state_dict().items()
            }
    except Exception as e:
        load_error = repr(e)
        loaded = False

print(f"Using device: {device}")
print(f"Dataset base: {BASE_INPUT}")
print(f"Num train images: {len(train_df)}")
print(f"Num test images: {len(test_df)}")
print(f"Weights found: {weight_path}")
print(f"Weights loaded: {loaded}")
if load_error is not None:
    print(f"Weight load note: {load_error[:400]}")

perm = np.random.RandomState(42).permutation(len(train_df))
val_n = max(1, int(0.15 * len(train_df)))
val_idx = perm[:val_n]
trn_idx = perm[val_n:]

trn_ds = TrainDataset(
    train_df.iloc[trn_idx], TRAIN_IMG_DIR, train_transform, cache_tag=f"train_{TRN_FP}"
)
val_ds = TrainDataset(
    train_df.iloc[val_idx], TRAIN_IMG_DIR, transform, cache_tag=f"val_{VAL_FP}"
)

_nw = min(8, max(2, (os.cpu_count() or 4) // 2))
_common_dl_kwargs = dict(
    num_workers=_nw,
    pin_memory=(device != "cpu"),
    persistent_workers=(_nw > 0),
    prefetch_factor=4 if _nw > 0 else None,
)

trn_dl = DataLoader(
    trn_ds,
    batch_size=8,
    shuffle=True,
    **{k: v for k, v in _common_dl_kwargs.items() if v is not None},
)
val_dl = DataLoader(
    val_ds,
    batch_size=8,
    shuffle=False,
    **{k: v for k, v in _common_dl_kwargs.items() if v is not None},
)

if not loaded:
    net.train()
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    epochs = 8  # fixed epoch budget (no early stopping)
    start_t = time.time()
    for ep in range(epochs):
        ep_loss = 0.0
        n_seen = 0
        for x, y, _ in trn_dl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            pred = net(x, final=True)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            bs = x.size(0)
            ep_loss += float(loss.detach().cpu()) * bs
            n_seen += bs

            if time.time() - start_t > 520:
                break

        net.eval()
        y_true = []
        y_pred = []
        with torch.no_grad():
            for x, y, _ in val_dl:
                x = x.to(device, non_blocking=True)
                pred = net(x, final=True).squeeze(1).detach()
                yhat = regress2class(pred).numpy().astype(int)
                ytrue = y.squeeze(1).numpy().astype(int)
                y_true.append(ytrue)
                y_pred.append(yhat)
        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)
        kappa = cohen_kappa_score(y_true, y_pred, weights="quadratic")

        net.train()
        print(
            f"epoch {ep+1}/{epochs} loss={ep_loss/max(1,n_seen):.4f} val_qwk={kappa:.4f}"
        )

        if time.time() - start_t > 520:
            print("Time budget reached; proceeding to inference.")
            break

    net.eval()
    _loaded_state_for_folds = {k: v.detach().cpu() for k, v in net.state_dict().items()}
else:
    net.eval()

net.eval()
val_preds_float = []
val_true_int = []
with torch.no_grad():
    for x, y, _ in val_dl:
        x = x.to(device, non_blocking=True)
        pred = net(x, final=True).squeeze(1).detach().cpu().numpy()
        val_preds_float.append(pred)
        val_true_int.append(y.squeeze(1).numpy().astype(int))
val_preds_float = np.concatenate(val_preds_float)
val_true_int = np.concatenate(val_true_int)

old_thr = list(threshold)
old_kappa = cohen_kappa_score(
    val_true_int, apply_thresholds_np(val_preds_float, old_thr), weights="quadratic"
)

oof_thr, oof_mean_kappa = oof_threshold_calibration_true_oof(
    base_state_dict=_loaded_state_for_folds,
    train_df_=train_df,
    train_img_dir_=TRAIN_IMG_DIR,
    base_threshold=old_thr,
    n_splits=3,
    batch_size=8,
    seed=42,
    fold_epochs=3,
)

oof_thr_ref, oof_ref_kappa = tune_thresholds_for_qwk(
    val_true_int, val_preds_float, oof_thr, iters=1
)

sc_thr, sc_kappa = tune_thresholds_for_qwk_scipy_multistart(
    val_true_int, val_preds_float, old_thr, seeds=(0, 1, 2, 3, 4, 5, 6, 7)
)
if sc_thr is None:
    cand_thr, cand_kappa = tune_thresholds_for_qwk(
        val_true_int, val_preds_float, old_thr, iters=2
    )
else:
    cand_thr, cand_kappa = sc_thr, sc_kappa
cand_thr, cand_kappa = tune_thresholds_for_qwk(
    val_true_int, val_preds_float, cand_thr, iters=1
)

print(
    f"Single-split: old_thr={old_thr} old_qwk={old_kappa:.4f} tuned_thr={cand_thr} tuned_qwk={cand_kappa:.4f}"
)
print(
    f"True-OOF(3-fold) avg thr={oof_thr} mean_fold_qwk={oof_mean_kappa:.4f} -> heldout_ref_thr={oof_thr_ref} heldout_ref_qwk={oof_ref_kappa:.4f}"
)

best_thr = old_thr
best_k = old_kappa
if cand_kappa >= best_k:
    best_thr, best_k = cand_thr, cand_kappa
if oof_ref_kappa >= best_k:
    best_thr, best_k = oof_thr_ref, oof_ref_kappa

threshold = list(best_thr)
_thr_cache.clear()  # ensure subsequent calls use updated thresholds
print(f"Final thresholds selected (heldout best): {threshold} qwk={best_k:.4f}")




## === cell 6
class TestDataset(Dataset):
    def __init__(self, ids, img_dir, transform, cache_tag=None):
        self.ids = np.asarray(list(ids), dtype=object)
        self.img_dir = img_dir
        self.transform = transform
        self.cache_tag = cache_tag

    def __len__(self):
        return int(self.ids.shape[0])

    def _cache_path(self, id_code):
        if self.cache_tag is None:
            return None
        return os.path.join(CACHE_DIR, f"{self.cache_tag}__{str(id_code)}.pt")

    def __getitem__(self, i):
        id_code = str(self.ids[i])
        cpath = self._cache_path(id_code)
        if cpath is not None and os.path.isfile(cpath):
            img = torch.load(cpath, map_location="cpu", weights_only=False)
            return img, id_code

        image_name = os.path.join(self.img_dir, f"{id_code}.png")
        with Image.open(image_name) as im:
            img = im.convert("RGB")
            img = self.transform(img)

        if cpath is not None:
            tmp = cpath + ".tmp"
            torch.save(img, tmp)
            os.replace(tmp, cpath)

        return img, id_code


ds = TestDataset(test_ids, TEST_IMG_DIR, transform, cache_tag=f"test_{VAL_FP}")
dl = DataLoader(
    ds,
    batch_size=8,
    shuffle=False,
    **{k: v for k, v in _common_dl_kwargs.items() if v is not None},
)

submission_rows = []
with torch.no_grad():
    for images, ids in dl:
        images = images.to(device, non_blocking=True)

        out1 = net(images, final=True).squeeze(1)

        images_flip = torch.flip(images, dims=[3])  # horizontal flip (W dimension)
        out2 = net(images_flip, final=True).squeeze(1)

        out = 0.5 * (out1 + out2)

        preds = regress2class(out).numpy().astype(int)
        submission_rows.extend([[str(idx), int(p)] for idx, p in zip(ids, preds)])

submission = pd.DataFrame(submission_rows, columns=["id_code", "diagnosis"])



## === cell 7
if submission.shape[0] == 0:
    raise RuntimeError(
        "Submission DataFrame is empty; inference did not produce any rows."
    )

submission["id_code"] = submission["id_code"].astype(str)
submission["diagnosis"] = submission["diagnosis"].astype(int)

submission = test_df[["id_code"]].merge(submission, on="id_code", how="left")
if submission["diagnosis"].isna().any():
    submission["diagnosis"] = submission["diagnosis"].fillna(0).astype(int)

submission.to_csv("submission.csv", index=False)
print(submission.head())
print(f"Wrote submission.csv with {len(submission)} rows.")
print(f"Final thresholds used: {threshold}")
