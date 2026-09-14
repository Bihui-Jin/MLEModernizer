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

0.9139656294404443

# 6. Current score

0.56623

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the missing weights issue by loading the model checkpoint only if it exists; otherwise the script still run end-to-end and create a valid submission (with a clear warning that score be low without weights). I also fix the CUDA crash by selecting `cuda` only when available and otherwise running on CPU, plus ensure all tensors are created on the correct device. Finally, I prevent an empty submission by iterating correctly over `id_code`, handling missing/corrupt images safely, and always writing a non-empty `submission.csv` with the required columns.'
- What this solution (achieved 0.66111) has done: 'Your current 0.0 score is consistent with running an untrained (random-weight) network because the checkpoint path doesn’t exist in the provided dataset tree, so the single most important change is to (legitimately) train the exact same model on `train.csv` + `train_images` inside the notebook and then run inference on `test_images`. To keep changes minimal and preserve core logic, I’m not changing the architecture, loss function, or prediction mapping; I’m only adding a short training phase (same forward path using the regressor head and your `regress2class` thresholds) plus a small train/val split to pick the best epoch by quadratic weighted kappa. This should move the score substantially upward toward your target (0.914) while staying within Kaggle constraints and still writing a valid `submission.csv`. I also keep the existing transform logic, only reusing it for both train and test to avoid train/test preprocessing mismatch.'
- What this solution (achieved 0.0) has done: 'The timeout is most likely coming from the “weights not found → train from scratch” fallback, plus expensive Python/PIL transforms and non-optimal dataloader settings during that training. To guarantee finishing under 600s without changing model/loop/loss semantics, the main fix is to (1) stop doing the expensive `/kaggle/input` directory walk for weights, and (2) make the “no weights” path fail fast (still preserving logic: this script is intended to be inference with pretrained weights). Additionally, I speed up inference by enabling `torch.compile` when available (numerically equivalent) and by reusing/caching small tensors and tightening dataloader overhead; the architecture, TTA, and thresholding remain identical.'
- What this solution (achieved 0.56623) has done: 'I fix the missing-weights failure by switching back to a minimal “train from scratch” fallback using the same model, loss (regression with MSE), and the same `regress2class` thresholds for kappa evaluation, so the notebook runs end-to-end and can achieve a non-zero score. I also fix the TTA runtime error by only applying 90° rotations when the resized image is square; since your resize is 380x285, I keep only the shape-preserving horizontal flip TTA (2-way), which preserves the same inference semantics without tensor shape mismatches. Finally, I ensure the submission is always written as `submission.csv` with correct `id_code,diagnosis` alignment.'

# 9. Code solution

## === cell 0
import os
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

import torchvision.transforms as transforms
from torchvision.transforms import functional as FT
from PIL import Image, ImageChops

from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedShuffleSplit
import timm

device = "cuda" if torch.cuda.is_available() else "cpu"
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

if device == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

Image.MAX_IMAGE_PIXELS = None



## === cell 1
threshold = [0.75, 1.5, 2.5, 3.5]

_THR_CACHE = {}


def _get_thr_tensor(dev, dtype):
    key = (str(dev), str(dtype))
    t = _THR_CACHE.get(key, None)
    if t is None:
        t = torch.as_tensor(threshold, device=dev, dtype=dtype).view(1, -1)
        _THR_CACHE[key] = t
    return t


def regress2class(out):
    out = out.view(-1)
    thr = _get_thr_tensor(out.device, out.dtype)
    return (out.view(-1, 1) >= thr).sum(dim=1).to(torch.int64)


def ordinal2class_prob(out):
    out = out.view(-1, 4)
    pred_prob = torch.zeros(out.size(0), 5, device=out.device)
    pred_prob[:, 0] = 1 - out[:, 0]
    pred_prob[:, 1] = out[:, 0] * (1 - out[:, 1])
    pred_prob[:, 2] = out[:, 1] * (1 - out[:, 2])
    pred_prob[:, 3] = out[:, 2] * (1 - out[:, 3])
    pred_prob[:, 4] = out[:, 3]
    return F.softmax(pred_prob, dim=1)


def regress2class_prob(out):
    out = out.view(-1)
    pred_prob = torch.zeros((out.size(0), 5), device=out.device)

    v = out.clamp_min(0.0)
    is_ge4 = v >= 4.0
    v_lt4 = v[~is_ge4]

    if v_lt4.numel() > 0:
        l1 = torch.floor(v_lt4).to(torch.int64)
        l2 = torch.ceil(v_lt4).to(torch.int64)
        w1 = 1.0 - (v_lt4 - l1.to(v_lt4.dtype))
        w2 = 1.0 - (l2.to(v_lt4.dtype) - v_lt4)

        idx = torch.nonzero(~is_ge4, as_tuple=False).view(-1)
        pred_prob[idx, l1] = w1
        pred_prob[idx, l2] = w2

    if is_ge4.any():
        pred_prob[is_ge4, 4] = 1.0

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
import cv2

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

try:
    from PIL import ImageFile

    ImageFile.LOAD_TRUNCATED_IMAGES = True
except Exception:
    pass


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
        try:
            img = np.array(image)  # RGB uint8
            if img.size == 0:
                return image

            bg = img[0, 0, :]  # background reference color
            diff = cv2.absdiff(img, bg.reshape(1, 1, 3))
            gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)

            _, mask = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)
            coords = cv2.findNonZero(mask)
            if coords is None:
                return image
            x, y, w, h = cv2.boundingRect(coords)
            if w <= 0 or h <= 0:
                return image
            return Image.fromarray(img[y : y + h, x : x + w, :])
        except Exception:
            return image




## === cell 4
DATA_DIR = "../input/aptos2019-blindness-detection"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")

train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)

train_df["id_code"] = train_df["id_code"].astype(str)
train_df["diagnosis"] = train_df["diagnosis"].astype(int)
test_df["id_code"] = test_df["id_code"].astype(str)

test_ids = test_df["id_code"].tolist()

input_size = 380

_MEAN = np.array([0.384, 0.258, 0.174], dtype=np.float32)
_STD = np.array([0.124, 0.089, 0.094], dtype=np.float32)
_H = input_size * 3 // 4
_W = input_size


def _cv_resize_rgb_uint8(img_rgb_uint8: np.ndarray) -> np.ndarray:
    return cv2.resize(img_rgb_uint8, (_W, _H), interpolation=cv2.INTER_LINEAR)


def _to_norm_tensor_from_rgb_uint8(img_rgb_uint8: np.ndarray) -> torch.Tensor:
    x = img_rgb_uint8.astype(np.float32) / 255.0
    x = (x - _MEAN) / _STD
    x = np.transpose(x, (2, 0, 1))  # CHW
    return torch.from_numpy(x)


class BaseTransformFast:
    def __init__(self):
        self._trim = trim()
        self._crop = cropTo4_3()

    def __call__(self, img_pil: Image.Image) -> torch.Tensor:
        img_pil = self._trim(img_pil)
        img_pil = self._crop(img_pil)
        img = np.array(img_pil, dtype=np.uint8)  # RGB
        img = _cv_resize_rgb_uint8(img)
        return _to_norm_tensor_from_rgb_uint8(img)


class TrainTransformFast:
    def __init__(self):
        self._trim = trim()
        self._crop = cropTo4_3()
        self._photo = photometric_distort()

    def __call__(self, img_pil: Image.Image) -> torch.Tensor:
        img_pil = self._trim(img_pil)
        img_pil = self._crop(img_pil)
        img_pil = self._photo(img_pil)
        img = np.array(img_pil, dtype=np.uint8)
        img = _cv_resize_rgb_uint8(img)
        return _to_norm_tensor_from_rgb_uint8(img)


base_transform = BaseTransformFast()
train_transform = TrainTransformFast()


def make_tta_batch_from_base_tensor_batch(x_bchw: torch.Tensor) -> torch.Tensor:
    B, C, H, W = x_bchw.shape
    tta = x_bchw.new_empty((B, 2, C, H, W))
    tta[:, 0] = x_bchw
    tta[:, 1] = torch.flip(x_bchw, dims=(3,))  # horizontal flip
    return tta




## === cell 5
class AptosDataset(torch.utils.data.Dataset):
    def __init__(self, df, img_dir, transform):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img_path = os.path.join(self.img_dir, f"{row['id_code']}.png")
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (input_size, input_size), (0, 0, 0))
        img = self.transform(img)
        if "diagnosis" in self.df.columns:
            y = int(row["diagnosis"])
            return img, y
        return img, row["id_code"]


def predict_kappa(model, loader):
    model.eval()
    ys = []
    ps = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            _, r_out, _ = model(xb)
            pred = (
                regress2class(r_out.data.squeeze(1)).detach().cpu().numpy().astype(int)
            )
            ps.append(pred)
            ys.append(yb.numpy().astype(int))
    ys = np.concatenate(ys)
    ps = np.concatenate(ps)
    return cohen_kappa_score(ys, ps, weights="quadratic")


class BaseTensorTestDataset(torch.utils.data.Dataset):
    def __init__(self, id_codes, img_dir, transform, enable_cache=True):
        self.id_codes = list(id_codes)
        self.img_dir = img_dir
        self.transform = transform
        self.enable_cache = enable_cache
        self._cache = {}  # id_code -> torch.Tensor or None

    def __len__(self):
        return len(self.id_codes)

    def __getitem__(self, i):
        id_code = self.id_codes[i]
        if self.enable_cache and id_code in self._cache:
            return id_code, self._cache[id_code]

        img_path = os.path.join(self.img_dir, f"{id_code}.png")
        try:
            img = Image.open(img_path).convert("RGB")
            x = self.transform(img)
        except Exception:
            x = None

        if self.enable_cache:
            self._cache[id_code] = x
        return id_code, x




## === cell 6
net = ThreeStage_Model().to(device)

WEIGHT_FILENAME = "B4_3stage_37epoch_CLAHE.pkl"
WEIGHTS_CANDIDATES = [
    "../input/weights/B4_3stage_37epoch_CLAHE.pkl",
    os.path.join(DATA_DIR, "weights", WEIGHT_FILENAME),
    f"/kaggle/input/weights/{WEIGHT_FILENAME}",
    f"/kaggle/input/aptos2019-blindness-detection/weights/{WEIGHT_FILENAME}",
]

for d in (
    "/kaggle/input/aptos2019-blindness-detection",
    "/kaggle/input/weights",
    "/kaggle/input/dataset",
    "/kaggle/input",
):
    WEIGHTS_CANDIDATES.append(os.path.join(d, WEIGHT_FILENAME))
    WEIGHTS_CANDIDATES.append(os.path.join(d, "weights", WEIGHT_FILENAME))

_seen = set()
WEIGHTS_CANDIDATES = [p for p in WEIGHTS_CANDIDATES if not (p in _seen or _seen.add(p))]

weights_path = next((p for p in WEIGHTS_CANDIDATES if os.path.exists(p)), None)

trained_from_scratch = False
if weights_path is not None:
    state = torch.load(weights_path, map_location="cpu")
    net.load_state_dict(state)
    print("Loaded weights:", weights_path)
else:
    print(
        "WARNING: Model weights not found. Training from scratch (same model + MSE on regressor output) "
        "to produce a non-zero submission score."
    )
    trained_from_scratch = True

net.eval()

if device == "cuda":
    net = net.to(memory_format=torch.channels_last)

if device == "cuda":
    try:
        net = torch.compile(net, mode="reduce-overhead", fullgraph=False)
        print("torch.compile enabled for inference/training.")
    except Exception as e:
        print("torch.compile not available/failed; using eager. Reason:", repr(e))



## === cell 7
if trained_from_scratch:
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    idx_tr, idx_va = next(sss.split(train_df["id_code"], train_df["diagnosis"]))
    df_tr = train_df.iloc[idx_tr].reset_index(drop=True)
    df_va = train_df.iloc[idx_va].reset_index(drop=True)

    train_ds = AptosDataset(df_tr, TRAIN_IMG_DIR, train_transform)
    val_ds = AptosDataset(df_va, TRAIN_IMG_DIR, base_transform)

    cpu_cnt = os.cpu_count() or 2
    num_workers = min(4, cpu_cnt)

    def _seed_worker(worker_id):
        seed = 42 + worker_id
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=8 if device == "cuda" else 2,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
        worker_init_fn=_seed_worker if num_workers > 0 else None,
    )

    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=16 if device == "cuda" else 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
        worker_init_fn=_seed_worker if num_workers > 0 else None,
    )

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(net.parameters(), lr=2e-4, weight_decay=1e-4)

    best_kappa = -1e9
    best_state = None

    epochs = 2  # keep within 600s; still a legitimate training step to avoid random predictions
    t0 = time.time()
    for ep in range(1, epochs + 1):
        net.train()
        ep_loss = 0.0
        n = 0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            y = yb.to(device, non_blocking=True).float().view(-1, 1)

            if device == "cuda":
                xb = xb.contiguous(memory_format=torch.channels_last)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device_type=("cuda" if device == "cuda" else "cpu"),
                enabled=(device == "cuda"),
            ):
                _, r_out, _ = net(xb)
                loss = criterion(r_out, y)
            loss.backward()
            optimizer.step()

            ep_loss += float(loss.detach().cpu()) * xb.size(0)
            n += xb.size(0)

        net.eval()
        kappa = predict_kappa(net, val_loader)
        avg_loss = ep_loss / max(1, n)
        print(f"epoch {ep}/{epochs} - train_mse={avg_loss:.4f} - val_qwk={kappa:.5f}")

        if kappa > best_kappa:
            best_kappa = kappa
            best_state = {
                k: v.detach().cpu().clone() for k, v in net.state_dict().items()
            }

    if best_state is not None:
        net.load_state_dict(best_state)
    print(f"Training done in {time.time()-t0:.1f}s. Best val QWK={best_kappa:.5f}")
    net.eval()



## === cell 8
submission = []
missing = 0

test_ds = BaseTensorTestDataset(
    test_ids, TEST_IMG_DIR, base_transform, enable_cache=True
)

cpu_cnt = os.cpu_count() or 2
num_workers = min(8, cpu_cnt)


def _collate_base_fast(batch):
    bsz = len(batch)
    ids = [None] * bsz
    valid = torch.empty(bsz, dtype=torch.bool)
    x_list = []
    for i, (idc, x) in enumerate(batch):
        ids[i] = idc
        ok = x is not None
        valid[i] = ok
        if ok:
            x_list.append(x)
    xb = torch.stack(x_list, dim=0) if x_list else None
    return ids, valid, xb


def _seed_worker(worker_id):
    seed = 42 + worker_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


test_loader = torch.utils.data.DataLoader(
    test_ds,
    batch_size=32 if device == "cuda" else 4,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=(device == "cuda"),
    persistent_workers=(num_workers > 0),
    prefetch_factor=4 if num_workers > 0 else None,
    collate_fn=_collate_base_fast,
    worker_init_fn=_seed_worker if num_workers > 0 else None,
)

per_image_T = 2
t0 = time.time()

net.eval()
with torch.inference_mode():
    for ids, valid_mask, xb_cpu in test_loader:
        if not bool(valid_mask.all()):
            for id_code, ok in zip(ids, valid_mask):
                if not bool(ok):
                    missing += 1
                    submission.append([id_code, 0])

        if xb_cpu is None:
            continue

        if device == "cuda":
            xb = xb_cpu.to(device, non_blocking=True).contiguous(
                memory_format=torch.channels_last
            )
        else:
            xb = xb_cpu.to(device)

        tta = make_tta_batch_from_base_tensor_batch(xb)  # [B,2,C,H,W]
        tta = tta.view(-1, *tta.shape[2:])  # [B*2,C,H,W]

        with torch.amp.autocast(
            device_type=("cuda" if device == "cuda" else "cpu"),
            enabled=(device == "cuda"),
        ):
            _, r_out, _ = net(tta)  # [B*2,1]

        r_out = r_out.view(-1)  # [B*2]
        bsz = xb_cpu.shape[0]
        r_mean = r_out.view(bsz, per_image_T).mean(dim=1)  # [B]
        pred = regress2class(r_mean).detach().cpu().numpy().astype(int)

        vi = 0
        for id_code, ok in zip(ids, valid_mask):
            if bool(ok):
                submission.append([id_code, int(pred[vi])])
                vi += 1

dt = time.time() - t0
if missing > 0:
    print(
        f"WARNING: {missing} test images were missing/unreadable; defaulted their predictions to 0."
    )
print(f"TTA+inference done in {dt:.1f}s for {len(test_ids)} test images.")

submission = np.array(submission, dtype=object)



## === cell 9
df = pd.DataFrame(submission, columns=["id_code", "diagnosis"])

df["id_code"] = df["id_code"].astype(str)
df["diagnosis"] = (
    pd.to_numeric(df["diagnosis"], errors="coerce").fillna(0).astype(int).clip(0, 4)
)

df = test_df[["id_code"]].merge(df, on="id_code", how="left")
df["diagnosis"] = df["diagnosis"].fillna(0).astype(int).clip(0, 4)

out_path = "submission.csv"
df.to_csv(out_path, index=False)

print(df.head())
print("Wrote:", out_path, "rows:", len(df), "cols:", list(df.columns))
print("Diagnosis distribution:", df["diagnosis"].value_counts().sort_index().to_dict())
