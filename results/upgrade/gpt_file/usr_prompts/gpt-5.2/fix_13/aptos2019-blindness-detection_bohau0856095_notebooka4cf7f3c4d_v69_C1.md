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
import time
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as transforms
from torchvision.transforms import functional as FT
from PIL import Image, ImageChops

import timm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)



## === cell 1
threshold = [0.75, 1.5, 2.5, 3.5]


def regress2class(out: torch.Tensor) -> torch.Tensor:
    """
    out: shape (B,) or (B,1) on any device.
    returns: shape (B,) on CPU, dtype float
    """
    if out.ndim == 2 and out.size(1) == 1:
        out = out.squeeze(1)
    prediction = torch.zeros(out.size(0), device=out.device, dtype=torch.float32)
    for i in range(4):
        prediction += (out >= threshold[i]).to(torch.float32)
    return prediction.detach().cpu()


def ordinal2class_prob(out: torch.Tensor) -> torch.Tensor:
    pred_prob = torch.zeros(out.size(0), 5, device=out.device, dtype=out.dtype)
    pred_prob[:, 0] = (1 - out[:, 0]).squeeze()
    pred_prob[:, 1] = (out[:, 0] * (1 - out[:, 1])).squeeze()
    pred_prob[:, 2] = (out[:, 1] * (1 - out[:, 2])).squeeze()
    pred_prob[:, 3] = (out[:, 2] * (1 - out[:, 3])).squeeze()
    pred_prob[:, 4] = out[:, 3].squeeze()
    return F.softmax(pred_prob, dim=1)


def regress2class_prob(out: torch.Tensor) -> torch.Tensor:
    pred_prob = torch.zeros((out.size(0), 5), device=out.device, dtype=out.dtype)
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
TEST_CSV = "../input/aptos2019-blindness-detection/test.csv"
TEST_IMG_DIR = "../input/aptos2019-blindness-detection/test_images"
TRAIN_CSV = "../input/aptos2019-blindness-detection/train.csv"
TRAIN_IMG_DIR = "../input/aptos2019-blindness-detection/train_images"

test_df = pd.read_csv(TEST_CSV)
test_ids = test_df["id_code"].astype(str).values

train_df = pd.read_csv(TRAIN_CSV)

input_size = 380

APTOS_MEAN = [0.384, 0.258, 0.174]
APTOS_STD = [0.124, 0.089, 0.094]
TIMM_IMAGENET_MEAN = [0.485, 0.456, 0.406]
TIMM_IMAGENET_STD = [0.229, 0.224, 0.225]


def make_transform(
    mean,
    std,
    interpolation=transforms.InterpolationMode.BILINEAR,
    augment: bool = False,
):
    ops = [
        trim(),
        cropTo4_3(),
    ]
    if augment:
        ops.append(photometric_distort())
    ops += [
        transforms.Resize(
            (input_size * 3 // 4, input_size),
            interpolation=interpolation,
            antialias=True,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]
    return transforms.Compose(ops)


transform = make_transform(APTOS_MEAN, APTOS_STD, augment=False)


class TestDataset(Dataset):
    def __init__(self, ids, img_dir, transform=None):
        self.ids = list(ids)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id_code = self.ids[idx]
        image_name = os.path.join(self.img_dir, f"{id_code}.png")
        img = Image.open(image_name).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return id_code, img


class TrainDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_dir: str, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        id_code = str(row["id_code"])
        y = float(row["diagnosis"])
        image_name = os.path.join(self.img_dir, f"{id_code}.png")
        img = Image.open(image_name).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, torch.tensor([y], dtype=torch.float32)


def find_checkpoint(preferred_path: str):
    if os.path.isfile(preferred_path):
        return preferred_path

    preferred_base = os.path.basename(preferred_path).lower()
    preferred_stem = os.path.splitext(preferred_base)[0]

    candidates = []
    for pat in [
        "../input/**/*.pkl",
        "../input/**/*.pth",
        "../input/**/*.pt",
        "../input/**/*.ckpt",
        "/kaggle/input/**/*.pkl",
        "/kaggle/input/**/*.pth",
        "/kaggle/input/**/*.pt",
        "/kaggle/input/**/*.ckpt",
    ]:
        candidates.extend(glob.glob(pat, recursive=True))

    def looks_like_model_file(p: str) -> bool:
        b = os.path.basename(p).lower()
        if any(
            x in b for x in ["submission", "sample_submission", "train.csv", "test.csv"]
        ):
            return False
        return b.endswith((".pt", ".pth", ".pkl", ".ckpt"))

    candidates = [c for c in candidates if looks_like_model_file(c)]
    if not candidates:
        return None

    extra_preferred_name_tokens = [
        "b4_3stage",
        "b4-3stage",
        "three_stage",
        "3stage",
        "aptos",
        "blindness",
        "retinopathy",
        "dr",
        "kappa",
        "effnet",
        "efficientnet",
        "tf_efficientnet_b4_ns",
        "tf_efficientnet_b5_ns",
    ]

    def score_path(p: str) -> tuple:
        b = os.path.basename(p).lower()
        pl = p.lower()
        s = 0

        if b == preferred_base:
            s += 5000
        if preferred_stem in b:
            s += 1200

        for tok in extra_preferred_name_tokens:
            if tok in b:
                s += 120

        for kw, w in [
            ("weights", 50),
            ("checkpoint", 50),
            ("ckpt", 40),
            ("fold", 15),
            ("best", 80),
            ("final", 30),
            ("finetune", 60),
            ("epoch", 10),
            ("0.9", 5),
        ]:
            if kw in pl:
                s += w

        depth = pl.count(os.sep)
        return (-s, depth, len(p))

    candidates_sorted = sorted(candidates, key=score_path)
    return candidates_sorted[0]


def _extract_state_dict(obj):
    if isinstance(obj, dict):
        for k in ["state_dict", "model", "model_state_dict", "net", "weights"]:
            if k in obj and isinstance(obj[k], dict):
                return obj[k]
    return obj


def _clean_state_dict(sd):
    if not isinstance(sd, dict):
        return sd
    cleaned = {}
    for k, v in sd.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module.") :]
        if nk.startswith("model."):
            nk = nk[len("model.") :]
        cleaned[nk] = v
    return cleaned


def _try_adapt_timm_classifier_to_three_stage(net: ThreeStage_Model, sd: dict) -> dict:
    if not isinstance(sd, dict):
        return sd
    if ("classifier.weight" not in sd) or ("classifier.bias" not in sd):
        return sd

    W = sd.get("classifier.weight", None)
    b = sd.get("classifier.bias", None)
    if not (torch.is_tensor(W) and torch.is_tensor(b)):
        return sd

    try:
        if W.ndim == 2 and W.shape[0] == 5:
            sd2 = dict(sd)

            sd2["classifier.3.weight"] = W.clone()
            sd2["classifier.3.bias"] = b.clone()

            k = torch.arange(5, device=W.device, dtype=W.dtype).view(5, 1)
            Wr = (W * k).sum(dim=0, keepdim=True) / 4.0
            br = (b * k.squeeze(1)).sum().view(1) / 4.0
            sd2["regressor.3.weight"] = Wr.clone()
            sd2["regressor.3.bias"] = br.clone()

            sd2["ordinal.3.weight"] = W[1:5].clone()
            sd2["ordinal.3.bias"] = b[1:5].clone()
            return sd2
    except Exception:
        return sd
    return sd


def _torch_load_maybe_weights_only(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")
    except Exception:
        return torch.load(path, map_location="cpu")


def _load_checkpoint_safely(
    model: nn.Module, ckpt_path: str
) -> tuple[bool, dict, dict]:
    state = _torch_load_maybe_weights_only(ckpt_path)
    sd = _clean_state_dict(_extract_state_dict(state))
    if not isinstance(sd, dict) or len(sd) == 0:
        return False, {}, {"reason": "empty_or_invalid_state_dict"}

    if isinstance(model, ThreeStage_Model):
        sd = _try_adapt_timm_classifier_to_three_stage(model, sd)

    load_res = model.load_state_dict(sd, strict=False)

    model_keys = set(model.state_dict().keys())
    sd_keys = set(sd.keys())
    matched = len(model_keys & sd_keys)
    match_ratio = matched / max(1, len(model_keys))

    has_backbone = any(k.startswith("backbone.") for k in sd_keys)
    loaded_ok = (match_ratio >= 0.20) or (has_backbone and match_ratio >= 0.12)

    info = {
        "matched": matched,
        "total_model_keys": len(model_keys),
        "match_ratio": match_ratio,
        "has_backbone": has_backbone,
        "missing_keys_n": len(load_res.missing_keys),
        "unexpected_keys_n": len(load_res.unexpected_keys),
    }
    return loaded_ok, sd, info


net = ThreeStage_Model()
ckpt_path = find_checkpoint("../input/weights/B4_3stage_15epoch_finetune.pkl")

loaded_checkpoint = False
use_final_head = False
ckpt_info = {}

if ckpt_path is not None:
    loaded_checkpoint, sd_used, ckpt_info = _load_checkpoint_safely(net, ckpt_path)
    if loaded_checkpoint:
        use_final_head = any(k.startswith("final_regressor.") for k in sd_used.keys())
    else:
        ckpt_path = None

if ckpt_path is None:
    net.backbone = timm.models.tf_efficientnet_b4_ns(pretrained=True)
    net.backbone.global_pool = GeM(flatten=True)
    loaded_checkpoint = False
    use_final_head = False

if not loaded_checkpoint:
    cfg = getattr(net.backbone, "default_cfg", {}) or {}
    mean = cfg.get("mean", TIMM_IMAGENET_MEAN)
    std = cfg.get("std", TIMM_IMAGENET_STD)
    interp = cfg.get("interpolation", "bilinear")
    interp_map = {
        "bilinear": transforms.InterpolationMode.BILINEAR,
        "bicubic": transforms.InterpolationMode.BICUBIC,
        "nearest": transforms.InterpolationMode.NEAREST,
        "lanczos": transforms.InterpolationMode.LANCZOS,
    }
    transform = make_transform(
        mean,
        std,
        interpolation=interp_map.get(interp, transforms.InterpolationMode.BILINEAR),
        augment=False,
    )

net = net.to(device)
net.eval()

print(
    f"device={device} ckpt_found={ckpt_path is not None} ckpt_path={ckpt_path} ckpt_loaded={loaded_checkpoint} "
    f"use_final_head={use_final_head} threshold={threshold} ckpt_info={ckpt_info}"
)



## === cell 5
ds = TestDataset(test_ids, TEST_IMG_DIR, transform=transform)
dl = DataLoader(
    ds, batch_size=8, shuffle=False, num_workers=2, pin_memory=torch.cuda.is_available()
)

submission_rows = []


def _run_inference(current_device: torch.device):
    rows = []
    net.to(current_device)
    net.eval()
    with torch.no_grad():
        for id_codes, imgs in dl:
            imgs = imgs.to(current_device, non_blocking=True)
            if use_final_head:
                out = net(imgs, final=True).squeeze(1)
            else:
                _, r_out, _ = net(imgs)
                out = r_out.squeeze(1)

            preds = regress2class(out)  # CPU tensor float
            preds = preds.numpy().astype(np.int64)
            for i, idc in enumerate(id_codes):
                rows.append([str(idc), int(preds[i])])
    return rows


def _qwk_from_preds(
    y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 5
) -> float:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    O = np.zeros((n_classes, n_classes), dtype=np.float64)
    for a, b in zip(y_true, y_pred):
        if 0 <= a < n_classes and 0 <= b < n_classes:
            O[a, b] += 1.0

    act_hist = O.sum(axis=1)
    pred_hist = O.sum(axis=0)
    E = np.outer(act_hist, pred_hist)
    if E.sum() > 0:
        E = E * (O.sum() / E.sum())

    W = np.zeros((n_classes, n_classes), dtype=np.float64)
    for i in range(n_classes):
        for j in range(n_classes):
            W[i, j] = ((i - j) ** 2) / ((n_classes - 1) ** 2)

    num = (W * O).sum()
    den = (W * E).sum()
    if den == 0:
        return 0.0
    return 1.0 - (num / den)


def _apply_thresholds(outputs_1d: np.ndarray, thr: list[float]) -> np.ndarray:
    outputs_1d = np.asarray(outputs_1d, dtype=np.float64)
    thr = [float(t) for t in thr]
    pred = np.zeros(outputs_1d.shape[0], dtype=np.int64)
    for t in thr:
        pred += (outputs_1d >= t).astype(np.int64)
    return pred


def _stratified_train_subset(max_items: int = 768, seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    df = train_df.copy()
    parts = []
    per_class = max(1, max_items // 5)
    for c in [0, 1, 2, 3, 4]:
        d = df[df["diagnosis"] == c]
        if len(d) == 0:
            continue
        idx = np.arange(len(d))
        rng.shuffle(idx)
        parts.append(d.iloc[idx[:per_class]])
    sub = pd.concat(parts, axis=0)
    if len(sub) < max_items:
        rem = df.loc[~df.index.isin(sub.index)]
        idx = np.arange(len(rem))
        rng.shuffle(idx)
        sub = pd.concat([sub, rem.iloc[idx[: (max_items - len(sub))]]], axis=0)
    return sub.iloc[:max_items].reset_index(drop=True)


def _light_train_regressor_head_when_no_ckpt(
    current_device: torch.device,
    max_items: int = 1536,
    epochs: int = 3,
    lr: float = 2e-3,
) -> None:
    """
    Change (score-relevant, minimal):
    - keep frozen backbone in eval mode during head training (prevents BN stat drift).
    - enable photometric_distort for this light training ONLY (helps head generalize),
      without changing model architecture or inference transform.
    """
    sub = _stratified_train_subset(max_items=max_items, seed=42)

    mean = transform.transforms[-1].mean  # Normalize
    std = transform.transforms[-1].std
    interp = transforms.InterpolationMode.BILINEAR
    ds_tr = TrainDataset(
        sub,
        TRAIN_IMG_DIR,
        transform=make_transform(mean, std, interpolation=interp, augment=True),
    )

    dl_tr = DataLoader(
        ds_tr,
        batch_size=8,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    net.to(current_device)

    for p in net.parameters():
        p.requires_grad = False
    for p in net.regressor.parameters():
        p.requires_grad = True

    net.backbone.eval()
    net.regressor.train()

    opt = torch.optim.Adam(net.regressor.parameters(), lr=lr)
    loss_fn = nn.SmoothL1Loss(beta=1.0)

    start = time.time()
    for ep in range(int(epochs)):
        total = 0.0
        n = 0
        for imgs, y in dl_tr:
            imgs = imgs.to(current_device, non_blocking=True)
            y = y.to(current_device, non_blocking=True).view(-1, 1)

            _, r_out, _ = net(imgs)
            r_out = r_out.view(-1, 1)

            loss = loss_fn(r_out, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            total += float(loss.detach().cpu().item()) * imgs.size(0)
            n += imgs.size(0)

        avg = total / max(1, n)
        print(
            f"Light-train regressor head (no ckpt) epoch {ep+1}/{epochs} loss={avg:.5f}"
        )

    net.eval()
    for p in net.parameters():
        p.requires_grad = True

    print(f"Light training time: {time.time()-start:.1f}s on device={current_device}")


def _calibrate_thresholds_by_qwk_on_train_subset(
    current_device: torch.device,
    max_items: int = 1024,
    grid_step: float = 0.04,
    iters: int = 2,
) -> list[float]:
    """
    Change (score-relevant, minimal): calibrate on a slightly larger stratified subset.
    This keeps the same QWK-maximizing threshold search but reduces variance vs 768.
    """
    sub = _stratified_train_subset(max_items=max_items, seed=42)
    ids_subset = sub["id_code"].astype(str).values
    y_true = sub["diagnosis"].values.astype(np.int64)

    ds_tr = TestDataset(ids_subset, TRAIN_IMG_DIR, transform=transform)
    dl_tr = DataLoader(
        ds_tr,
        batch_size=8,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    preds = []
    net.to(current_device)
    net.eval()
    with torch.no_grad():
        for _, imgs in dl_tr:
            imgs = imgs.to(current_device, non_blocking=True)
            if use_final_head:
                out = net(imgs, final=True).squeeze(1)
            else:
                _, r_out, _ = net(imgs)
                out = r_out.squeeze(1)
            preds.append(out.detach().float().cpu().numpy())
    outputs = np.clip(np.concatenate(preds, axis=0).astype(np.float64), 0.0, 4.5)

    counts = pd.Series(y_true).value_counts().sort_index()
    probs = (counts / counts.sum()).reindex([0, 1, 2, 3, 4]).fillna(0).values
    cum = np.cumsum(probs)
    q = np.clip(cum[:4], 1e-6, 1 - 1e-6)
    thr = np.quantile(outputs, q).astype(np.float64)
    thr = np.clip(thr, 0.0, 4.49)
    for i in range(1, 4):
        if thr[i] <= thr[i - 1]:
            thr[i] = min(4.49, thr[i - 1] + 1e-3)

    for _ in range(int(iters)):
        for k in range(4):
            lo = 0.0 if k == 0 else (thr[k - 1] + 1e-3)
            hi = 4.49 if k == 3 else (thr[k + 1] - 1e-3)
            if hi <= lo:
                continue
            cand = np.arange(lo, hi + 1e-9, float(grid_step), dtype=np.float64)
            best_t = thr[k]
            best_score = -1e9
            for t in cand:
                thr_try = thr.copy()
                thr_try[k] = float(t)
                y_pred = _apply_thresholds(outputs, thr_try.tolist())
                score = _qwk_from_preds(y_true, y_pred, n_classes=5)
                if score > best_score:
                    best_score = score
                    best_t = float(t)
            thr[k] = best_t

    thr = np.clip(thr, 0.0, 4.49)
    for i in range(1, 4):
        if thr[i] <= thr[i - 1]:
            thr[i] = min(4.49, thr[i - 1] + 1e-3)

    y_pred_final = _apply_thresholds(outputs, thr.tolist())
    print(
        f"Train-subset QWK after threshold calibration: {_qwk_from_preds(y_true, y_pred_final):.6f}"
    )
    return [float(x) for x in thr.tolist()]


try:
    if not loaded_checkpoint:
        _light_train_regressor_head_when_no_ckpt(
            device, max_items=1536, epochs=3, lr=2e-3
        )

        threshold = _calibrate_thresholds_by_qwk_on_train_subset(
            device, max_items=1024, grid_step=0.04, iters=2
        )
        print(f"Fallback QWK-optimized thresholds (no checkpoint): {threshold}")

    submission_rows = _run_inference(device)

except RuntimeError as e:
    msg = str(e).lower()
    if ("input type" in msg and "weight type" in msg) or (
        "expected all tensors" in msg and "same device" in msg
    ):
        device = torch.device("cpu")

        if not loaded_checkpoint:
            _light_train_regressor_head_when_no_ckpt(
                device, max_items=1536, epochs=3, lr=2e-3
            )
            threshold = _calibrate_thresholds_by_qwk_on_train_subset(
                device, max_items=1024, grid_step=0.04, iters=2
            )
            print(
                f"Fallback QWK-optimized thresholds (no checkpoint, CPU): {threshold}"
            )

        submission_rows = _run_inference(device)
    else:
        raise

submission = pd.DataFrame(submission_rows, columns=["id_code", "diagnosis"])

if len(submission) != len(test_df):
    submission = test_df.merge(submission, on="id_code", how="left")
    submission["diagnosis"] = submission["diagnosis"].fillna(0).astype(int)
else:
    submission["diagnosis"] = submission["diagnosis"].astype(int)

submission = submission[["id_code", "diagnosis"]]
submission.to_csv("submission.csv", index=False)

print(submission.head())
print(f"Saved submission.csv with shape={submission.shape} on device={device}")
print(
    "Pred label distribution:",
    submission["diagnosis"].value_counts(dropna=False).sort_index().to_dict(),
)
