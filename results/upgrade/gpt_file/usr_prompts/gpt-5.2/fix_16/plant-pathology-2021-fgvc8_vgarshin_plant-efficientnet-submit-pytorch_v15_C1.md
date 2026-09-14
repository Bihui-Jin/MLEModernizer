# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Detect apple diseases from images.

## Metric
Mean F1-Score

## Submission Format
labels should be a space-delimited list.

The file should contain a header and have the following format:

```
image, labels
85f8cb619c66b863.jpg,healthy
ad8770db05586b59.jpg,healthy
c7b03e718489f3ca.jpg,healthy
```

## Dataset
**train.csv** - the training set metadata.

- `image` - the image ID.
- `labels` - the target classes, a space delimited list of all diseases found in the image. Unhealthy leaves with too many diseases to classify visually will have the `complex` class, and may also have a subset of the diseases identified.

**sample_submission.csv** - A sample submission file in the correct format.

- `image`
- `labels`

**train_images** - The training set images.

**test_images** - The test set images. This competition has a hidden test set: only three images are provided here as samples while the remaining 5,000 images will be available to your notebook once it is submitted.

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
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
sklearn-pandas==2.2.0
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
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
        input/
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
        working/
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
```

-> data/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> data/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os
import gc
import sys
import json
import time
import cv2
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.utils.data as data
from torch.utils.data.sampler import SequentialSampler

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
torch.set_num_interop_threads(1)

try:
    display  # noqa: F821
except NameError:

    def display(x):
        print(x)




## === cell 1
import torchvision
from torchvision import models

KAGGLE = True
if not KAGGLE:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("device:", DEVICE)



## === cell 2
TEST = True
VER = "v6"

_CANDIDATES = [
    "../input/plant-pathology-2021-fgvc8",
    "/kaggle/input/plant-pathology-2021-fgvc8",
    "../input/plant-pathology-2021-fgvc8/plant-pathology-2021-fgvc8",
    "/kaggle/input/plant-pathology-2021-fgvc8/plant-pathology-2021-fgvc8",
    "/kaggle/input/plant-pathology-2021-fgvc8/plant-pathology-2021-fgvc8/plant-pathology-2021-fgvc8",
]
DATA_PATH = None
for p in _CANDIDATES:
    if os.path.isdir(p) and os.path.isfile(os.path.join(p, "sample_submission.csv")):
        DATA_PATH = p
        break
if DATA_PATH is None:
    DATA_PATH = "../input/plant-pathology-2021-fgvc8"

MDLS_CANDIDATES = [
    f"../input/plant-models-{VER}",
    f"/kaggle/input/plant-models-{VER}",
    f"../input/plant-models-{VER}/plant-models-{VER}",
    f"/kaggle/input/plant-models-{VER}/plant-models-{VER}",
]

MDLS_PATH = None
for p in MDLS_CANDIDATES:
    if os.path.isdir(p) and os.path.isfile(os.path.join(p, "params.json")):
        MDLS_PATH = p
        break


def _find_pretrained_bundle(ver: str):
    roots = ["/kaggle/input", "../input"]
    for root in roots:
        if not os.path.isdir(root):
            continue
        candidate_dirs = []
        for d in os.listdir(root):
            if f"plant-models-{ver}" in d:
                candidate_dirs.append(os.path.join(root, d))
        walk_roots = candidate_dirs if candidate_dirs else [root]
        for wr in walk_roots:
            for dirpath, dirnames, filenames in os.walk(wr):
                if "params.json" in filenames:
                    has_any_weight = any(
                        fn.startswith("model_best_") and fn.endswith(".pth")
                        for fn in filenames
                    )
                    if has_any_weight:
                        return dirpath
    return None


if MDLS_PATH is None:
    found = _find_pretrained_bundle(VER)
    if found is not None:
        MDLS_PATH = found
    else:
        MDLS_PATH = f"../input/plant-models-{VER}"  # keep original default

TTAS = [0, 1]
FOLDS = [0, 1, 2]
IMGS_PATH = f"{DATA_PATH}/test_images" if TEST else f"{DATA_PATH}/train_images"

start_time = time.time()
print("DATA_PATH:", DATA_PATH)
print("IMGS_PATH:", IMGS_PATH)
print("MDLS_PATH:", MDLS_PATH)



## === cell 3
params_path = f"{MDLS_PATH}/params.json"
HAS_PARAMS = os.path.isfile(params_path)
HAS_ANY_WEIGHTS = False
if os.path.isdir(MDLS_PATH):
    try:
        HAS_ANY_WEIGHTS = any(
            fn.startswith("model_best_") and fn.endswith(".pth")
            for fn in os.listdir(MDLS_PATH)
        )
    except Exception:
        HAS_ANY_WEIGHTS = False
HAS_PRETRAINED = bool(HAS_PARAMS and HAS_ANY_WEIGHTS)

params = None
LABELS_ = None
LABELS = None
WORKERS = 2

if HAS_PRETRAINED:
    with open(params_path) as file:
        params = json.load(file)
    LABELS_ = params["labels_"]  # list-like (model output order)
    LABELS = params["labels"]  # dict-like index->label strings
    WORKERS = 2 if KAGGLE else int(params.get("workers", 2))
    print("loaded pretrained params keys:", sorted(list(params.keys())))
else:
    print(
        f"WARNING: Missing/partial pretrained bundle at {MDLS_PATH} "
        f"(params.json={HAS_PARAMS}, any_model_best_*.pth={HAS_ANY_WEIGHTS}). "
        "Will create a valid fallback submission."
    )



## === cell 4
sample_path = os.path.join(DATA_PATH, "sample_submission.csv")
if not os.path.isfile(sample_path):
    raise FileNotFoundError(f"sample_submission.csv not found at: {sample_path}")

df_sub = pd.read_csv(sample_path)
if "image" not in df_sub.columns or "labels" not in df_sub.columns:
    raise ValueError(
        f"Unexpected sample_submission.csv columns: {df_sub.columns.tolist()}"
    )

df_sub["labels"] = "healthy"
display(df_sub.head())
print("n_test_images (from sample_submission):", len(df_sub))

if not os.path.isdir(IMGS_PATH):
    raise FileNotFoundError(f"Image directory not found: {IMGS_PATH}")




## === cell 5
def flip(img, axis=0):
    if axis == 1:
        return img[::-1, :]
    elif axis == 2:
        return img[:, ::-1]
    elif axis == 3:
        return img[::-1, ::-1]
    else:
        return img


class PlantDataset(data.Dataset):
    def __init__(self, df, size, labels, transform=None, tta=0, enable_cache=True):
        self.df = df.reset_index(drop=True)
        self.size = int(size)
        self.labels = labels
        self.transform = transform
        self.tta = int(tta)
        self.enable_cache = bool(enable_cache)

        self._base_cache = {} if (self.enable_cache and self.labels is None) else None

    def __len__(self):
        return self.df.shape[0]

    def _load_base_tensor(self, index: int) -> torch.Tensor:
        row = self.df.iloc[index]
        img_path = f"{IMGS_PATH}/{row.image}"
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image read failed: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.size, self.size))
        img = img.astype(np.float32) / 255.0
        if self.transform is not None:
            img = self.transform(image=img)["image"]
        img = img.transpose(2, 0, 1)  # CHW
        return torch.from_numpy(np.ascontiguousarray(img))

    def __getitem__(self, index):
        if self.labels:
            row = self.df.iloc[index]
            img_path = f"{IMGS_PATH}/{row.image}"
            img = cv2.imread(img_path)
            if img is None:
                raise FileNotFoundError(f"Image read failed: {img_path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.size, self.size))
            img = img.astype(np.float32) / 255.0
            if self.transform is not None:
                img = self.transform(image=img)["image"]
            img = img.transpose(2, 0, 1)
            label = np.zeros(len(self.labels)).astype(np.float32)
            for lbl in row.labels.split():
                label[self.labels[lbl]] = 1
            return (torch.tensor(img), torch.tensor(label))

        if self._base_cache is not None:
            base = self._base_cache.get(index)
            if base is None:
                base = self._load_base_tensor(index)
                self._base_cache[index] = base
        else:
            base = self._load_base_tensor(index)

        return base


class EffNet(nn.Module):
    """
    EfficientNet backbone -> replace classifier with Identity -> custom FC head.
    (Preserves original core logic; uses torchvision EfficientNet.)
    """

    def __init__(self, params, out_dim):
        super().__init__()
        backbone = params["backbone"]

        name_map = {
            "efficientnet-b0": "efficientnet_b0",
            "efficientnet-b1": "efficientnet_b1",
            "efficientnet-b2": "efficientnet_b2",
            "efficientnet-b3": "efficientnet_b3",
            "efficientnet-b4": "efficientnet_b4",
            "efficientnet-b5": "efficientnet_b5",
            "efficientnet-b6": "efficientnet_b6",
            "efficientnet-b7": "efficientnet_b7",
        }
        tv_name = name_map.get(backbone, backbone)

        if not hasattr(models, tv_name):
            raise ValueError(
                f"Unknown backbone '{backbone}' (mapped to '{tv_name}'). "
                f"Available torchvision: {[n for n in dir(models) if 'efficientnet' in n]}"
            )

        self.enet = getattr(models, tv_name)(weights=None)

        if not hasattr(self.enet, "classifier") or not isinstance(
            self.enet.classifier, nn.Sequential
        ):
            raise RuntimeError(
                "Unexpected EfficientNet structure in torchvision version."
            )

        last_linear = self.enet.classifier[-1]
        if not isinstance(last_linear, nn.Linear):
            raise RuntimeError(
                "Unexpected classifier last layer type; expected nn.Linear."
            )
        nc = last_linear.in_features

        self.enet.classifier = nn.Identity()

        drop = float(params["dropout"])
        self.myfc = nn.Sequential(
            nn.Dropout(drop),
            nn.Linear(nc, int(nc / 4)),
            nn.Dropout(drop),
            nn.Linear(int(nc / 4), out_dim),
        )

    def extract(self, x):
        return self.enet(x)

    def forward(self, x):
        x = self.extract(x)
        x = self.myfc(x)
        return x




## === cell 6
models_ens = []
if HAS_PRETRAINED:
    use_channels_last = DEVICE.type == "cuda"
    for n_fold in FOLDS:
        model = EffNet(params, out_dim=len(LABELS_))
        path = f"{MDLS_PATH}/model_best_{n_fold}.pth"
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing model weights: {path}")
        state_dict = torch.load(path, map_location="cpu")
        model.load_state_dict(state_dict, strict=True)
        model.float()
        model.eval()
        model.to(DEVICE)
        if use_channels_last:
            model = model.to(memory_format=torch.channels_last)
        models_ens.append(model)
        print("loaded:", path)

    del state_dict, model
    gc.collect()
else:
    print("Skipping model loading because pretrained files are not available.")



## === cell 7
datasets, loaders = [], []
if HAS_PRETRAINED:
    n_workers = 0

    dl_kwargs = dict(
        batch_size=int(params["batch_size"]),
        sampler=None,  # set below
        num_workers=n_workers,
        pin_memory=(DEVICE.type == "cuda"),
        drop_last=False,
        persistent_workers=False,
    )

    dataset = PlantDataset(
        df=df_sub,
        size=params["img_size"],
        labels=None,
        transform=None,
        tta=0,
        enable_cache=True,  # cache decode/resize/normalize once
    )
    datasets.append(dataset)
    loader = torch.utils.data.DataLoader(
        dataset,
        **dl_kwargs,
        sampler=SequentialSampler(dataset),
    )
    loaders.append(loader)

    print(
        "n_loaders:",
        len(loaders),
        "num_workers:",
        n_workers,
        "(single-process loader so cached decode/resize is reused across models/TTAs)",
    )
else:
    print("Skipping dataloaders because pretrained files are not available.")




## === cell 8
def get_labels(row, labels, ths):
    idxs = [i for i, x in enumerate(row) if x > ths[str(i)]]
    row_labels = [labels[str(i)] for i in idxs]
    row_out = (
        "healthy"
        if ("healthy" in row_labels or len(row_labels) == 0)
        else " ".join(row_labels)
    )
    return row_out


_FALLBACK_LABELS = [
    "scab",
    "rust",
    "frogeye_leaf_spot",
    "powdery_mildew",
    "complex",
    "healthy",
]


def _heuristic_labels_from_image(img_bgr: np.ndarray) -> str:
    if img_bgr is None:
        return "healthy"

    img = cv2.resize(img_bgr, (384, 384), interpolation=cv2.INTER_AREA)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0].astype(np.uint8)
    s = hsv[:, :, 1].astype(np.uint8)
    v = hsv[:, :, 2].astype(np.uint8)

    white_mask = (s < 45) & (v > 190)
    white_ratio = float(np.mean(white_mask))

    orange_mask = (h >= 5) & (h <= 25) & (s > 90) & (v > 70)
    orange_ratio = float(np.mean(orange_mask))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
    blackhat = cv2.morphologyEx(gray_blur, cv2.MORPH_BLACKHAT, kernel)
    bh_mean, bh_std = float(blackhat.mean()), float(blackhat.std())
    bh_thr = bh_mean + 1.5 * bh_std
    dark_spot_mask = blackhat > bh_thr
    dark_spot_ratio = float(np.mean(dark_spot_mask))

    edges = cv2.Canny(gray_blur, 60, 140)
    edge_ratio = float(np.mean(edges > 0))

    labels = []

    if white_ratio > 0.055:
        labels.append("powdery_mildew")
    if orange_ratio > 0.020:
        labels.append("rust")

    if dark_spot_ratio > 0.020:
        labels.append("scab")
    elif dark_spot_ratio > 0.010 and edge_ratio > 0.060:
        labels.append("frogeye_leaf_spot")

    if len(labels) >= 2:
        labels.append("complex")

    if len(labels) == 0:
        return "healthy"

    return " ".join(labels)


def _extract_fallback_features(img_bgr: np.ndarray) -> np.ndarray:
    if img_bgr is None:
        return np.zeros(10, dtype=np.float32)

    img = cv2.resize(img_bgr, (256, 256), interpolation=cv2.INTER_AREA)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0].astype(np.uint8)
    s = hsv[:, :, 1].astype(np.uint8)
    v = hsv[:, :, 2].astype(np.uint8)

    white_ratio = float(np.mean((s < 45) & (v > 190)))
    orange_ratio = float(np.mean((h >= 5) & (h <= 25) & (s > 90) & (v > 70)))
    green_ratio = float(np.mean((h >= 35) & (h <= 95) & (s > 60) & (v > 40)))
    dark_ratio = float(np.mean(v < 60))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    blackhat = cv2.morphologyEx(gray_blur, cv2.MORPH_BLACKHAT, kernel)
    bh_mean = float(blackhat.mean())
    bh_std = float(blackhat.std())
    dark_spot_ratio = float(np.mean(blackhat > (bh_mean + 1.25 * bh_std)))

    edges = cv2.Canny(gray_blur, 60, 140)
    edge_ratio = float(np.mean(edges > 0))

    g_mean = float(gray.mean()) / 255.0
    g_std = float(gray.std()) / 255.0

    s_mean = float(s.mean()) / 255.0

    feats = np.array(
        [
            white_ratio,
            orange_ratio,
            green_ratio,
            dark_ratio,
            dark_spot_ratio,
            edge_ratio,
            g_mean,
            g_std,
            s_mean,
            1.0,
        ],
        dtype=np.float32,
    )
    return feats


def _multilabel_f1_macro(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    eps = 1e-12
    f1s = []
    for c in range(y_true.shape[1]):
        tp = float(np.sum((y_true[:, c] == 1) & (y_pred[:, c] == 1)))
        fp = float(np.sum((y_true[:, c] == 0) & (y_pred[:, c] == 1)))
        fn = float(np.sum((y_true[:, c] == 1) & (y_pred[:, c] == 0)))
        denom = 2 * tp + fp + fn
        f1 = (2 * tp) / (denom + eps)
        f1s.append(f1)
    return float(np.mean(f1s))


if HAS_PRETRAINED:
    N = len(df_sub)
    C = len(LABELS_)

    probs_sum_gpu = torch.zeros((N, C), device=DEVICE, dtype=torch.float32)
    n_terms = 0

    use_channels_last = DEVICE.type == "cuda"
    do_ttas = TTAS[:]  # preserve original set/order

    def _apply_tta(x: torch.Tensor, tta: int) -> torch.Tensor:
        if tta == 1:
            return x.flip(-2)  # H flip
        elif tta == 2:
            return x.flip(-1)  # W flip
        elif tta == 3:
            return x.flip(-2).flip(-1)
        return x

    with torch.inference_mode():
        for i, model in enumerate(models_ens):
            offset = 0
            for img_data in loaders[0]:
                bs = img_data.shape[0]

                if use_channels_last:
                    img_data = img_data.contiguous(memory_format=torch.channels_last)

                img_data = img_data.to(DEVICE, non_blocking=True)

                for tta in do_ttas:
                    x = _apply_tta(img_data, int(tta))
                    p = model(x).sigmoid()  # (bs, C)
                    probs_sum_gpu[offset : offset + bs].add_(p)
                    n_terms += 1

                offset += bs

            print(f"model {i} -> done (accumulated {n_terms} terms so far)")

    probs = (
        (probs_sum_gpu / float(n_terms))
        .detach()
        .to("cpu")
        .numpy()
        .astype(np.float32, copy=False)
    )  # (N, C)

    ths_arr = np.array([params["ths"][str(i)] for i in range(C)], dtype=np.float32)
    is_pos = probs > ths_arr[None, :]
    idx_to_lbl = np.array([LABELS[str(i)] for i in range(C)], dtype=object)

    out_labels = ["healthy"] * N
    for r in range(N):
        idxs = np.flatnonzero(is_pos[r])
        if idxs.size == 0:
            continue
        row_lbls = idx_to_lbl[idxs].tolist()
        if ("healthy" in row_lbls) or (len(row_lbls) == 0):
            continue
        out_labels[r] = " ".join(row_lbls)

    df_sub["labels"] = out_labels
else:
    train_csv_path = os.path.join(DATA_PATH, "train.csv")
    train_imgs_path = os.path.join(DATA_PATH, "train_images")
    if not os.path.isfile(train_csv_path):
        raise FileNotFoundError(f"train.csv not found at: {train_csv_path}")
    if not os.path.isdir(train_imgs_path):
        raise FileNotFoundError(f"train_images dir not found at: {train_imgs_path}")

    df_train = pd.read_csv(train_csv_path)
    label_list = _FALLBACK_LABELS[:]  # fixed competition labels

    y = np.zeros((len(df_train), len(label_list)), dtype=np.int8)
    lbl_to_idx = {l: i for i, l in enumerate(label_list)}
    for i, s in enumerate(df_train["labels"].astype(str).values):
        for l in s.split():
            if l in lbl_to_idx:
                y[i, lbl_to_idx[l]] = 1

    X = np.zeros((len(df_train), 10), dtype=np.float32)
    for i, img_name in enumerate(df_train["image"].values):
        img_path = os.path.join(train_imgs_path, img_name)
        img_bgr = cv2.imread(img_path)
        X[i] = _extract_fallback_features(img_bgr)
        if (i + 1) % 2000 == 0:
            print(f"features train: {i+1}/{len(df_train)}")

    from sklearn.model_selection import StratifiedKFold
    from sklearn.linear_model import LogisticRegression

    n_splits = 5
    oof_prob = np.zeros((len(df_train), len(label_list)), dtype=np.float32)

    for c, lbl in enumerate(label_list):
        pos = int(y[:, c].sum())
        neg = int(len(y) - pos)
        if pos < 20 or neg < 20:
            oof_prob[:, c] = float(pos) / float(len(y))
            continue

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        for tr_idx, va_idx in skf.split(X, y[:, c]):
            model = LogisticRegression(
                solver="lbfgs",
                max_iter=200,
                class_weight="balanced",
            )
            model.fit(X[tr_idx], y[tr_idx, c])
            oof_prob[va_idx, c] = model.predict_proba(X[va_idx])[:, 1].astype(
                np.float32
            )

    thresholds = np.full(len(label_list), 0.5, dtype=np.float32)
    for c, lbl in enumerate(label_list):
        best_t = thresholds[c]
        best_score = -1.0
        grid = np.linspace(0.05, 0.95, 19, dtype=np.float32)
        for t in grid:
            y_pred_tmp = (oof_prob > thresholds[None, :]).astype(np.int8)
            y_pred_tmp[:, c] = (oof_prob[:, c] > t).astype(np.int8)
            score = _multilabel_f1_macro(y, y_pred_tmp)
            if score > best_score:
                best_score = score
                best_t = float(t)
        thresholds[c] = best_t
        print(f"tuned threshold: {lbl} -> {thresholds[c]:.2f}")

    final_models = []
    for c, lbl in enumerate(label_list):
        pos = int(y[:, c].sum())
        neg = int(len(y) - pos)
        if pos < 20 or neg < 20:
            final_models.append(None)
            continue
        model = LogisticRegression(
            solver="lbfgs",
            max_iter=200,
            class_weight="balanced",
        )
        model.fit(X, y[:, c])
        final_models.append(model)

    X_test = np.zeros((len(df_sub), 10), dtype=np.float32)
    for i, img_name in enumerate(df_sub["image"].values):
        img_path = os.path.join(IMGS_PATH, img_name)
        img_bgr = cv2.imread(img_path)
        X_test[i] = _extract_fallback_features(img_bgr)

    prob_test = np.zeros((len(df_sub), len(label_list)), dtype=np.float32)
    for c, mdl in enumerate(final_models):
        if mdl is None:
            prob_test[:, c] = float(y[:, c].mean())
        else:
            prob_test[:, c] = mdl.predict_proba(X_test)[:, 1].astype(np.float32)

    pred_bin = (prob_test > thresholds[None, :]).astype(np.int8)

    out = []
    healthy_idx = lbl_to_idx["healthy"]
    for i in range(len(df_sub)):
        idxs = np.where(pred_bin[i] == 1)[0].tolist()
        if len(idxs) == 0:
            out.append("healthy")
            continue
        labels_i = [label_list[j] for j in idxs]
        if "healthy" in labels_i and len(labels_i) > 1:
            labels_i = [l for l in labels_i if l != "healthy"]
        if len(labels_i) == 0:
            out.append("healthy")
        else:
            out.append(" ".join(labels_i))
    df_sub["labels"] = out

elapsed_time = time.time() - start_time
print(f"time elapsed: {elapsed_time // 60:.0f} min {elapsed_time % 60:.0f} sec")



## === cell 9
print("value counts:")
print(df_sub.labels.value_counts().head(20))
display(df_sub.head())



## === cell 10
sub_path = "submission.csv"
df_sub[["image", "labels"]].to_csv(sub_path, index=False)
print("wrote:", sub_path, "rows:", len(df_sub))
print(df_sub.dtypes)
