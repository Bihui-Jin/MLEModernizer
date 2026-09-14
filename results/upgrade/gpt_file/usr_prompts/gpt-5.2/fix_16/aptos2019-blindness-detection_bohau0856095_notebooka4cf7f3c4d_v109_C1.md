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

from sklearn.metrics import (
    cohen_kappa_score,
)  # kept to preserve original imports/semantics
from sklearn.model_selection import StratifiedKFold
import timm

import cv2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True



## === cell 1
threshold = [0.75, 1.5, 2.5, 3.5]


def regress2class(out):
    out = out.view(-1)
    thr = torch.tensor(threshold, device=out.device, dtype=out.dtype).view(1, -1)
    prediction = (out.view(-1, 1) >= thr).sum(dim=1).to(torch.int64)
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
    out = out.view(-1).to(torch.float32)
    pred_prob = torch.zeros((out.size(0), 5), device=out.device, dtype=out.dtype)

    ge4 = out >= 4.0
    pred_prob[ge4, 4] = 1.0

    lt4 = ~ge4
    if lt4.any():
        o = out[lt4]
        l1 = torch.floor(o).to(torch.int64)
        l2 = torch.ceil(o).to(torch.int64)
        w1 = 1.0 - (o - l1.to(o.dtype))
        w2 = 1.0 - (l2.to(o.dtype) - o)

        rows = torch.nonzero(lt4, as_tuple=False).view(-1)
        pred_prob[rows, l1] = w1
        pred_prob[rows, l2] = w2

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


def _zero_init_linear(m: nn.Module):
    if isinstance(m, nn.Linear):
        nn.init.zeros_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)




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
BASE_CANDIDATES = [
    "/kaggle/input/aptos2019-blindness-detection",
    "../input/aptos2019-blindness-detection",
]
base = None
for b in BASE_CANDIDATES:
    if os.path.exists(b):
        base = b
        break
if base is None:
    base = "../input/aptos2019-blindness-detection"  # fallback

train_csv_path = os.path.join(base, "train.csv")
test_csv_path = os.path.join(base, "test.csv")
train_img_dir = os.path.join(base, "train_images")
test_img_dir = os.path.join(base, "test_images")

train_df = pd.read_csv(train_csv_path)
test_df = pd.read_csv(test_csv_path)

train_df["id_code"] = train_df["id_code"].astype(str)
test_df["id_code"] = test_df["id_code"].astype(str)

train_ids = train_df["id_code"].values
train_y = train_df["diagnosis"].astype(int).values
test_ids = test_df["id_code"].values

input_size = 384

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
print("Using timm data config mean/std:", mean, std)

transform = transforms.Compose(
    [
        trim(),
        cropTo4_3(),
        transforms.Resize((input_size * 3 // 4, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]
)

net = ThreeStage_Model().to(device)

ckpt_name = "B4_3stage_17epoch_320finetune.pkl"
candidate_paths = [
    "../input/weights/" + ckpt_name,
    "/kaggle/input/weights/" + ckpt_name,
]
candidate_paths += glob.glob("/kaggle/input/**/" + ckpt_name, recursive=True)

ckpt_path = None
for p in candidate_paths:
    if os.path.exists(p):
        ckpt_path = p
        break

if ckpt_path is not None:
    state = torch.load(ckpt_path, map_location="cpu")
    if (
        isinstance(state, dict)
        and "state_dict" in state
        and isinstance(state["state_dict"], dict)
    ):
        state = state["state_dict"]
    try:
        net.load_state_dict(state, strict=True)
        print("Loaded checkpoint (strict=True):", ckpt_path)
    except Exception as e:
        missing, unexpected = net.load_state_dict(state, strict=False)
        print("Loaded checkpoint (strict=False):", ckpt_path)
        print("Strict load failed with:", repr(e))
        if missing:
            print("Missing keys (loaded with strict=False):", len(missing))
        if unexpected:
            print("Unexpected keys (loaded with strict=False):", len(unexpected))
else:
    print(
        f"WARNING: Checkpoint '{ckpt_name}' not found; using ImageNet-pretrained backbone weights only."
    )
    net.classifier.apply(_zero_init_linear)
    net.regressor.apply(_zero_init_linear)
    net.ordinal.apply(_zero_init_linear)
    net.final_regressor.apply(_zero_init_linear)

net.eval()

print("Num train images:", len(train_ids))
print("Train images dir exists:", os.path.exists(train_img_dir))
print("Num test images:", len(test_ids))
print("Test images dir exists:", os.path.exists(test_img_dir))

CPU_COUNT = os.cpu_count() or 2
DEFAULT_NW = min(4, max(2, CPU_COUNT // 2))
print("CPU_COUNT:", CPU_COUNT, "DEFAULT_NW:", DEFAULT_NW)

INFER_BS = 32 if torch.cuda.is_available() else 8
CACHE_BS = 64 if torch.cuda.is_available() else 32

CACHE_DIR = "/kaggle/working/tensor_cache"
os.makedirs(CACHE_DIR, exist_ok=True)



## === cell 5
from torch.utils.data import Dataset, DataLoader

_MEAN = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
_STD = np.array(std, dtype=np.float32).reshape(1, 1, 3)


def _trim_bbox_np(rgb: np.ndarray):
    bg = rgb[0, 0].astype(np.int16)  # (3,)
    diff = np.abs(rgb.astype(np.int16) - bg[None, None, :]).astype(np.int16)
    diff = np.clip(diff * 2 - 10, 0, 255).astype(np.uint8)
    mask = diff.max(axis=2) > 0
    if not mask.any():
        return None
    ys, xs = np.where(mask)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    return x0, y0, x1, y1


def _crop_to_4_3_np(rgb: np.ndarray):
    h, w = rgb.shape[:2]
    if (w / h) >= (4 / 3):
        new_h = h
        new_w = int(h * 4 / 3)
    else:
        new_h = int(w * 3 / 4)
        new_w = w
    left = int(round((w - new_w) / 2))
    top = int(round((h - new_h) / 2))
    right = left + new_w
    bottom = top + new_h
    left = max(0, left)
    top = max(0, top)
    right = min(w, right)
    bottom = min(h, bottom)
    return rgb[top:bottom, left:right]


def _resize_np(rgb: np.ndarray, out_h: int, out_w: int):
    return cv2.resize(rgb, (out_w, out_h), interpolation=cv2.INTER_LINEAR)


def _to_tensor_normalized(rgb: np.ndarray):
    x = rgb.astype(np.float32) / 255.0
    x = (x - _MEAN) / _STD
    x = np.transpose(x, (2, 0, 1))
    return torch.from_numpy(x)  # float32 CPU tensor


def fast_transform_from_path(image_path: str):
    bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if bgr is None:
        rgb = np.zeros((input_size, input_size, 3), dtype=np.uint8)
    else:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        bbox = _trim_bbox_np(rgb)
        if bbox is not None:
            x0, y0, x1, y1 = bbox
            rgb = rgb[y0:y1, x0:x1]

        rgb = _crop_to_4_3_np(rgb)
        rgb = _resize_np(rgb, input_size * 3 // 4, input_size)

    return _to_tensor_normalized(rgb)


class ImgDataset(Dataset):
    def __init__(self, ids, img_dir, transform=None, labels=None, use_fast_cv=True):
        self.ids = list(ids)
        self.img_dir = img_dir
        self.transform = transform
        self.labels = None if labels is None else np.asarray(labels).astype(int)
        self.use_fast_cv = use_fast_cv

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        idx = self.ids[i]
        image_name = os.path.join(self.img_dir, f"{idx}.png")

        if self.use_fast_cv:
            x = fast_transform_from_path(image_name)
        else:
            try:
                img = Image.open(image_name).convert("RGB")
            except Exception:
                img = Image.new("RGB", (input_size, input_size), (0, 0, 0))
            x = self.transform(img) if self.transform is not None else img

        if self.labels is None:
            return str(idx), x
        return str(idx), x, int(self.labels[i])


class TensorCacheDataset(Dataset):
    def __init__(self, ids, tensor_cache, labels=None):
        self.ids = list(ids)
        self.tensor_cache = tensor_cache  # dict id->CPU tensor CHW float32
        self.labels = None if labels is None else np.asarray(labels).astype(int)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        idx = str(self.ids[i])
        x = self.tensor_cache[idx]
        if self.labels is None:
            return idx, x
        return idx, x, int(self.labels[i])


def _make_loader(ds, bs, shuffle, nw):
    pin = torch.cuda.is_available()
    return DataLoader(
        ds,
        batch_size=bs,
        shuffle=shuffle,
        num_workers=nw,
        pin_memory=pin,
        persistent_workers=(nw > 0),
        prefetch_factor=4 if nw > 0 else None,  # higher prefetch reduces idle GPU
        drop_last=False,
    )


def build_tensor_cache(
    ids, img_dir, transform, bs=CACHE_BS, nw=DEFAULT_NW, cache_name="cache.pt"
):
    cache_path = os.path.join(CACHE_DIR, cache_name)
    if os.path.exists(cache_path):
        obj = torch.load(cache_path, map_location="cpu")
        if isinstance(obj, dict) and len(obj) == len(ids):
            return obj

    ds = ImgDataset(ids, img_dir, transform=transform, labels=None, use_fast_cv=True)
    dl = _make_loader(ds, bs=bs, shuffle=False, nw=nw)

    cache = {}
    with torch.inference_mode():
        for ids_batch, x in dl:
            x = x.contiguous()
            for k, t in zip(ids_batch, x.unbind(0)):
                cache[str(k)] = t.contiguous()

    torch.save(cache, cache_path)
    return cache


def infer_regression_from_cache(ids, tensor_cache, bs=INFER_BS, nw=DEFAULT_NW):
    ds = TensorCacheDataset(ids, tensor_cache, labels=None)
    dl = _make_loader(ds, bs=bs, shuffle=False, nw=nw)

    out_all = []
    ids_all = []
    with torch.inference_mode():
        for ids_batch, x in dl:
            x = x.to(device, non_blocking=True)
            out = net(x, final=True).view(-1).detach().cpu().numpy().astype(np.float32)
            out_all.append(out)
            ids_all.extend(ids_batch)
    out_all = (
        np.concatenate(out_all, axis=0) if out_all else np.zeros((0,), dtype=np.float32)
    )
    return np.array(ids_all), out_all


def apply_thresholds(pred_cont, thr_list):
    thr = np.array(thr_list, dtype=np.float32).reshape(1, -1)
    pred_cont = np.asarray(pred_cont, dtype=np.float32).reshape(-1, 1)
    return (pred_cont >= thr).sum(axis=1).astype(np.int64)


def qwk(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


def tune_thresholds_oof(pred_cont, y_true, init_thr, n_iter=2):
    thr = np.array(init_thr, dtype=np.float32)

    def project(t):
        t = np.clip(t, 0.0, 4.5)
        t = np.sort(t)
        min_gap = 1e-3
        for i in range(1, 4):
            if t[i] <= t[i - 1] + min_gap:
                t[i] = min(t[i - 1] + min_gap, 4.5)
        return t

    thr = project(thr)
    best_thr = thr.copy()
    best_score = qwk(y_true, apply_thresholds(pred_cont, best_thr))

    steps = [0.25, 0.10, 0.05]
    for _ in range(n_iter):
        for step in steps:
            for j in range(4):
                candidates = []
                for delta in (-step, 0.0, step):
                    t = best_thr.copy()
                    t[j] = t[j] + delta
                    t = project(t)
                    candidates.append(t)
                local_best_thr = best_thr
                local_best_score = best_score
                for t in candidates:
                    s = qwk(y_true, apply_thresholds(pred_cont, t))
                    if s > local_best_score:
                        local_best_score = s
                        local_best_thr = t
                best_thr, best_score = local_best_thr, local_best_score

    return best_thr.tolist(), float(best_score)




## === cell 6
def train_heads_and_final_regressor(
    net: ThreeStage_Model,
    ids,
    y,
    img_dir,
    bs: int = 16,
    nw: int = DEFAULT_NW,
    epochs: int = 4,
    lr: float = 5e-3,
):
    net.train()

    for p in net.parameters():
        p.requires_grad_(False)
    for head in (net.classifier, net.regressor, net.ordinal, net.final_regressor):
        for p in head.parameters():
            p.requires_grad_(True)

    ds = ImgDataset(ids, img_dir, transform=transform, labels=y, use_fast_cv=True)
    dl = _make_loader(ds, bs=bs, shuffle=True, nw=nw)

    opt = torch.optim.AdamW(
        list(net.classifier.parameters())
        + list(net.regressor.parameters())
        + list(net.ordinal.parameters())
        + list(net.final_regressor.parameters()),
        lr=lr,
        weight_decay=1e-4,
    )
    loss_fn = nn.MSELoss()

    for ep in range(1, epochs + 1):
        total_loss = 0.0
        n = 0
        for _, x, yb in dl:
            x = x.to(device, non_blocking=True)
            yb = torch.as_tensor(yb, device=device, dtype=torch.float32).view(-1, 1)

            with torch.inference_mode():
                feats = net.backbone(x)

            c_out = net.classifier(feats)
            r_out = net.regressor(feats)
            o_out = net.ordinal(feats)
            fusion = torch.cat((c_out, r_out, o_out), dim=1)

            out = net.final_regressor(fusion)
            out = torch.sigmoid(out) * 4.5

            loss = loss_fn(out, yb)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            total_loss += float(loss.item()) * x.size(0)
            n += x.size(0)

        print(
            f"heads+final train epoch {ep}/{epochs} - mse: {total_loss / max(1,n):.5f}"
        )

    net.eval()
    return net


def _fresh_model_like(base_net: ThreeStage_Model) -> ThreeStage_Model:
    m = ThreeStage_Model().to(device)
    m.eval()
    return m


if ckpt_path is None:
    net = train_heads_and_final_regressor(
        net,
        train_ids,
        train_y,
        train_img_dir,
        bs=16,
        nw=DEFAULT_NW,
        epochs=4,
        lr=5e-3,
    )



## === cell 7
print("Building train tensor cache...")
train_cache = build_tensor_cache(
    train_ids,
    train_img_dir,
    transform=transform,
    bs=CACHE_BS,
    nw=DEFAULT_NW,
    cache_name="train_cache.pt",
)
print("Train tensor cache size:", len(train_cache))

print("Building test tensor cache...")
test_cache = build_tensor_cache(
    test_ids,
    test_img_dir,
    transform=transform,
    bs=CACHE_BS,
    nw=DEFAULT_NW,
    cache_name="test_cache.pt",
)
print("Test tensor cache size:", len(test_cache))

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_pred = np.zeros(len(train_df), dtype=np.float32)

if ckpt_path is None:
    for fold, (tr_idx, va_idx) in enumerate(skf.split(train_ids, train_y), 1):
        fold_net = _fresh_model_like(net)

        fold_net.classifier.apply(_zero_init_linear)
        fold_net.regressor.apply(_zero_init_linear)
        fold_net.ordinal.apply(_zero_init_linear)
        fold_net.final_regressor.apply(_zero_init_linear)

        fold_net = train_heads_and_final_regressor(
            fold_net,
            train_ids[tr_idx],
            train_y[tr_idx],
            train_img_dir,
            bs=16,
            nw=DEFAULT_NW,
            epochs=3,  # kept as in original script
            lr=5e-3,
        )

        _net_prev = net
        net = fold_net
        va_ids = train_ids[va_idx]
        _, va_pred = infer_regression_from_cache(
            va_ids, train_cache, bs=INFER_BS, nw=DEFAULT_NW
        )
        net = _net_prev

        oof_pred[va_idx] = va_pred
        print(
            f"Fold {fold}: trained on {len(tr_idx)} and inferred {len(va_idx)} val images"
        )
else:
    for fold, (tr_idx, va_idx) in enumerate(skf.split(train_ids, train_y), 1):
        va_ids = train_ids[va_idx]
        _, va_pred = infer_regression_from_cache(
            va_ids, train_cache, bs=INFER_BS, nw=DEFAULT_NW
        )
        oof_pred[va_idx] = va_pred
        print(f"Fold {fold}: inferred {len(va_idx)} train images")

best_thr, best_oof_qwk = tune_thresholds_oof(oof_pred, train_y, threshold, n_iter=2)
print("OOF QWK (with tuned thresholds):", best_oof_qwk)
print("Tuned thresholds:", best_thr)

threshold = best_thr




## === cell 8
def infer_regression_tta_from_cache(ids, tensor_cache, bs=INFER_BS, nw=DEFAULT_NW):
    ds = TensorCacheDataset(ids, tensor_cache, labels=None)
    dl = _make_loader(ds, bs=bs, shuffle=False, nw=nw)

    out_all = []
    ids_all = []
    with torch.inference_mode():
        for ids_batch, x in dl:
            x = x.to(device, non_blocking=True)

            out0 = net(x, final=True).view(-1)

            x_h = torch.flip(x, dims=[3])
            out1 = net(x_h, final=True).view(-1)

            x_v = torch.flip(x, dims=[2])
            out2 = net(x_v, final=True).view(-1)

            out = (out0 + out1 + out2) / 3.0
            out_all.append(out.detach().cpu().numpy().astype(np.float32))
            ids_all.extend(ids_batch)
    out_all = (
        np.concatenate(out_all, axis=0) if out_all else np.zeros((0,), dtype=np.float32)
    )
    return np.array(ids_all), out_all


def calibrate_to_oof(test_pred, oof_pred, clip_min=0.0, clip_max=4.5):
    test_pred = np.asarray(test_pred, dtype=np.float32)
    oof_pred = np.asarray(oof_pred, dtype=np.float32)

    m_test, s_test = float(test_pred.mean()), float(test_pred.std() + 1e-6)
    m_oof, s_oof = float(oof_pred.mean()), float(oof_pred.std() + 1e-6)

    scale = s_oof / s_test
    shift = m_oof - scale * m_test

    scale = float(np.clip(scale, 0.8, 1.25))
    shift = float(np.clip(shift, -0.5, 0.5))

    cal = test_pred * scale + shift
    cal = np.clip(cal, clip_min, clip_max).astype(np.float32)
    return cal, {
        "scale": scale,
        "shift": shift,
        "m_test": m_test,
        "s_test": s_test,
        "m_oof": m_oof,
        "s_oof": s_oof,
    }


test_ids_out, test_pred_cont = infer_regression_tta_from_cache(
    test_ids, test_cache, bs=INFER_BS, nw=DEFAULT_NW
)

pred_map = {k: v for k, v in zip(test_ids_out.tolist(), test_pred_cont.tolist())}
test_pred_cont_aligned = np.array(
    [pred_map.get(str(i), 0.0) for i in test_ids], dtype=np.float32
)

test_pred_cont_cal, cal_stats = calibrate_to_oof(test_pred_cont_aligned, oof_pred)
print("Calibration stats:", cal_stats)

test_pred_cls = apply_thresholds(test_pred_cont_cal, threshold)

pred_rows = [[str(i), int(p)] for i, p in zip(test_ids, test_pred_cls.tolist())]
print("Predicted rows:", len(pred_rows))

if len(pred_rows) == 0:
    raise RuntimeError("No predictions were generated; submission would be empty.")

pred_df = pd.DataFrame(pred_rows, columns=["id_code", "diagnosis"])
pred_df["id_code"] = pred_df["id_code"].astype(str)
pred_df["diagnosis"] = pred_df["diagnosis"].astype(int)

sub_df = test_df.copy()
sub_df["id_code"] = sub_df["id_code"].astype(str)
sub_df = sub_df.merge(pred_df, on="id_code", how="left")

if sub_df["diagnosis"].isna().any():
    sub_df["diagnosis"] = sub_df["diagnosis"].fillna(0).astype(int)

sub_df.to_csv("submission.csv", index=False)
print(sub_df.head())
print("Wrote submission.csv with shape:", sub_df.shape)
print("Unique diagnosis values:", sorted(sub_df["diagnosis"].unique().tolist()))
print("Final thresholds used:", threshold)
