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
import random
import cv2
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.utils.data as data
from torch.utils.data.sampler import SequentialSampler
import torchvision

KAGGLE = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICE:", DEVICE)

TEST = True
VER = "v0"
if KAGGLE:
    DATA_PATH = "../input/plant-pathology-2021-fgvc8"
    MDLS_PATH = f"../input/plant-efficientnet-train-pytorch/models_{VER}"
else:
    DATA_PATH = "./data"
    MDLS_PATH = f"./models_{VER}"

TTAS = [0, 1, 2]
FOLDS = [0]
IMGS_PATH = f"{DATA_PATH}/test_images" if TEST else f"{DATA_PATH}/train_images"

start_time = time.time()
print("DATA_PATH:", DATA_PATH)
print("MDLS_PATH:", MDLS_PATH)
print("IMGS_PATH:", IMGS_PATH)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

try:
    cv2.setNumThreads(0)
except Exception:
    pass

if DEVICE.type == "cuda":
    INFER_WORKERS = min(8, max(2, (os.cpu_count() or 4) // 2))
    INFER_PREFETCH = 4
else:
    INFER_WORKERS = min(4, max(1, (os.cpu_count() or 2) // 2))
    INFER_PREFETCH = 2 if INFER_WORKERS > 0 else None

try:
    torch.multiprocessing.set_sharing_strategy("file_system")
except Exception:
    pass

INFERENCE_CONTEXT = torch.inference_mode



## === cell 1
sub_path_candidates = [
    f"{DATA_PATH}/sample_submission.csv",
    "../input/sample_submission.csv",
    "../input/plant-pathology-2021-fgvc8/sample_submission.csv",
]
sub_path = None
for p in sub_path_candidates:
    if os.path.isfile(p):
        sub_path = p
        break
if sub_path is None:
    raise FileNotFoundError(
        f"Could not find sample_submission.csv in candidates: {sub_path_candidates}"
    )

df_sub = pd.read_csv(sub_path)
if "image" not in df_sub.columns:
    raise ValueError(
        f"sample_submission.csv missing 'image' column: cols={df_sub.columns.tolist()}"
    )
df_sub["labels"] = "healthy"
print("Loaded sample_submission:", sub_path, "shape:", df_sub.shape)
df_sub.head()



## === cell 2
use_external_model = False
params = None
ths = None
LABELS_ = None
LABELS = None

params_path = f"{MDLS_PATH}/params.json"
ths_path = f"{MDLS_PATH}/ths.json"

if os.path.isfile(params_path) and os.path.isfile(ths_path):
    with open(params_path) as file:
        params = json.load(file)
    LABELS_ = params["labels_"]  # list of label names (in model output order)
    LABELS = params["labels"]
    WORKERS = (
        4
        if (KAGGLE and DEVICE.type == "cuda")
        else (2 if KAGGLE else params.get("workers", 2))
    )
    print("loaded params:", params)

    with open(ths_path) as file:
        ths = json.load(file)
    print("thresholds loaded; n_ths =", len(ths))

    if isinstance(LABELS, dict) and ("0" not in LABELS) and (0 in LABELS):
        LABELS = {str(k): v for k, v in LABELS.items()}
    elif (
        isinstance(LABELS, dict)
        and ("0" not in LABELS)
        and all(isinstance(k, str) for k in LABELS.keys())
    ):
        vals = list(LABELS.values())
        if len(vals) and all(isinstance(v, int) for v in vals):
            inv = {str(v): k for k, v in LABELS.items()}
            LABELS = inv

    ths = {str(k): float(v) for k, v in ths.items()}

    ok = True
    for n_fold in FOLDS:
        wpath = f"{MDLS_PATH}/model_best_{n_fold}.pth"
        if not os.path.isfile(wpath):
            print("Missing model weights:", wpath)
            ok = False
            break
    use_external_model = ok
else:
    WORKERS = 4 if (KAGGLE and DEVICE.type == "cuda") else 2

print("use_external_model:", use_external_model)

use_train_fallback = not use_external_model




## === cell 3
def flip(img, axis=0):
    if axis == 1:
        return img[
            ::-1,
            :,
        ]
    elif axis == 2:
        return img[
            :,
            ::-1,
        ]
    elif axis == 3:
        return img[
            ::-1,
            ::-1,
        ]
    else:
        return img


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class PlantDataset(data.Dataset):
    def __init__(
        self,
        df,
        size,
        labels,
        transform=None,
        tta=0,
        imgs_path=None,
        normalize=True,
        shared_cache=None,
        shared_cache_order=None,
    ):
        self.df = df.reset_index(drop=True)
        self.size = int(size)
        self.labels = labels
        self.transform = transform
        self.tta = int(tta)
        self.imgs_path = imgs_path if imgs_path is not None else IMGS_PATH
        self.normalize = bool(normalize)

        self._paths = (self.imgs_path + "/" + self.df["image"].astype(str)).tolist()

        if shared_cache is None:
            self._cache_u8 = {}  # path -> np.uint8(H,W,3) RGB resized
            self._cache_order = []
        else:
            self._cache_u8 = shared_cache
            self._cache_order = (
                shared_cache_order if shared_cache_order is not None else []
            )

        if not self.labels:
            self._max_cache = 8192
        else:
            self._max_cache = 20000  # allow full train+val in-memory resized cache

        if self.labels:
            parsed = []
            for s in self.df["labels"].astype(str).tolist():
                idxs = []
                for lbl in s.split():
                    j = self.labels.get(lbl, None)
                    if j is not None:
                        idxs.append(int(j))
                parsed.append(np.array(idxs, dtype=np.int64))
            self._label_idxs = parsed
        else:
            self._label_idxs = None

        self._n_classes = len(self.labels) if self.labels else 0

    def __len__(self):
        return self.df.shape[0]

    def _get_base_u8(self, img_path: str):
        cached = self._cache_u8.get(img_path, None)
        if cached is not None:
            return cached

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_LINEAR)

        if len(self._cache_order) >= self._max_cache:
            old = self._cache_order.pop(0)
            self._cache_u8.pop(old, None)

        self._cache_u8[img_path] = img  # uint8 RGB HWC
        self._cache_order.append(img_path)
        return img

    def __getitem__(self, index):
        img_path = self._paths[index]
        img = self._get_base_u8(img_path)

        if not self.labels:
            if self.tta == 1:
                img = img[::-1, :, :]
            elif self.tta == 2:
                img = img[:, ::-1, :]
            elif self.tta == 3:
                img = img[::-1, ::-1, :]

        if self.transform is not None:
            img = self.transform(image=img)["image"]  # expects HWC
            if isinstance(img, np.ndarray) and img.ndim == 3 and img.shape[0] in (1, 3):
                x = torch.from_numpy(np.ascontiguousarray(img))
            else:
                img = np.ascontiguousarray(img.transpose(2, 0, 1))
                x = torch.from_numpy(img)
        else:
            x_np = img.astype(np.float32, copy=False) / 255.0
            if self.normalize:
                x_np = (x_np - IMAGENET_MEAN) / IMAGENET_STD
            x_np = np.ascontiguousarray(x_np.transpose(2, 0, 1))
            x = torch.from_numpy(x_np)  # float32 CPU

        if self.labels:
            label = np.zeros(self._n_classes, dtype=np.float32)
            idxs = self._label_idxs[index]
            if idxs.size:
                label[idxs] = 1.0
            return x, torch.from_numpy(label)
        else:
            return x




## === cell 4
class EffNet(nn.Module):
    def __init__(self, params, out_dim):
        super(EffNet, self).__init__()
        backbone = params["backbone"]

        tv_map = {
            "efficientnet-b0": torchvision.models.efficientnet_b0,
            "efficientnet-b1": torchvision.models.efficientnet_b1,
            "efficientnet-b2": torchvision.models.efficientnet_b2,
            "efficientnet-b3": torchvision.models.efficientnet_b3,
            "efficientnet-b4": torchvision.models.efficientnet_b4,
            "efficientnet-b5": torchvision.models.efficientnet_b5,
            "efficientnet-b6": torchvision.models.efficientnet_b6,
            "efficientnet-b7": torchvision.models.efficientnet_b7,
        }

        if backbone not in tv_map:
            raise ValueError(
                f"Unsupported backbone '{backbone}' without efficientnet_pytorch. "
                f"Supported: {sorted(tv_map.keys())} or use params['backbone']=='resnext'."
            )

        self.enet = tv_map[backbone](weights=None)

        if hasattr(self.enet, "classifier") and isinstance(
            self.enet.classifier, nn.Sequential
        ):
            nc = None
            for m in reversed(self.enet.classifier):
                if isinstance(m, nn.Linear):
                    nc = m.in_features
                    break
            if nc is None:
                raise RuntimeError(
                    "Could not infer EfficientNet feature dim from torchvision classifier."
                )
            self.enet.classifier = nn.Identity()
        else:
            raise RuntimeError(
                "Unexpected torchvision EfficientNet structure; no .classifier Sequential found."
            )

        self.myfc = nn.Sequential(
            nn.Dropout(params["dropout"]),
            nn.Linear(nc, int(nc / 4)),
            nn.Dropout(params["dropout"]),
            nn.Linear(int(nc / 4), out_dim),
        )

    def extract(self, x):
        return self.enet(x)

    def forward(self, x):
        x = self.extract(x)
        x = self.myfc(x)
        return x


class ResNext(nn.Module):
    def __init__(self, params, out_dim):
        super(ResNext, self).__init__()
        self.rsnxt = torchvision.models.resnext50_32x4d(weights=None)
        nc = self.rsnxt.fc.in_features
        self.rsnxt.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(nc, int(nc / 4)),
            nn.ReLU(),
            nn.Dropout(params["dropout"]),
            nn.Linear(int(nc / 4), out_dim),
        )
        if torch.cuda.device_count() > 1:
            self.rsnxt = nn.DataParallel(self.rsnxt)

    def forward(self, x):
        return self.rsnxt(x)




## === cell 5
models_list = []
if use_external_model:
    for n_fold in FOLDS:
        if params["backbone"] == "resnext":
            model = ResNext(params=params, out_dim=len(LABELS_))
        else:
            model = EffNet(params=params, out_dim=len(LABELS_))

        path = "{}/model_best_{}.pth".format(MDLS_PATH, n_fold)
        state_dict = torch.load(path, map_location="cpu")

        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError:
            new_state = {}
            for k, v in state_dict.items():
                nk = k.replace("module.", "") if k.startswith("module.") else k
                new_state[nk] = v
            model.load_state_dict(new_state, strict=True)

        model.float()
        model.eval()
        model.to(DEVICE)
        models_list.append(model)
        print("loaded:", path)

    del state_dict, model
    gc.collect()
else:
    print("External model not available; will train a fallback model for predictions.")



## === cell 6
train_path_candidates = [
    f"{DATA_PATH}/train.csv",
    "../input/train.csv",
    "../input/plant-pathology-2021-fgvc8/train.csv",
]
train_path = None
for p in train_path_candidates:
    if os.path.isfile(p):
        train_path = p
        break
if train_path is None:
    raise FileNotFoundError(
        f"Could not find train.csv in candidates: {train_path_candidates}"
    )

df_train = pd.read_csv(train_path)
print("Loaded train:", train_path, "shape:", df_train.shape)
df_train["labels"] = df_train["labels"].fillna("").astype(str)

all_labs = set()
for s in df_train["labels"].tolist():
    for t in str(s).split():
        if t.strip():
            all_labs.add(t.strip())
LABELS_ = sorted(list(all_labs))
label_to_idx = {l: i for i, l in enumerate(LABELS_)}
idx_to_label = {str(i): l for i, l in enumerate(LABELS_)}
LABELS = idx_to_label  # used by get_labels()

print("n_classes:", len(LABELS_), "classes:", LABELS_)

if params is None:
    params = {
        "backbone": "efficientnet-b0",
        "img_size": 224,
        "dropout": 0.2,
        "batch_size": 32,
    }

n = len(df_train)
perm = np.random.RandomState(SEED).permutation(n)
val_frac = 0.1
n_val = int(n * val_frac)
val_idx = perm[:n_val]
trn_idx = perm[n_val:]
df_trn = df_train.iloc[trn_idx].reset_index(drop=True)
df_val = df_train.iloc[val_idx].reset_index(drop=True)
print("train/val sizes:", df_trn.shape, df_val.shape)

TRAIN_IMGS_PATH = f"{DATA_PATH}/train_images"




## === cell 7
def get_labels(row, labels, ths):
    idxs = [i for i, x in enumerate(row) if x > ths.get(str(i), 0.5)]
    labs = [labels.get(str(i), str(i)) for i in idxs]
    out = "healthy" if ("healthy" in labs or len(labs) == 0) else " ".join(labs)
    return out


def f1_micro_from_logits(logits, y_true, ths_dict):
    th_arr = np.fromiter(
        (ths_dict.get(str(i), 0.5) for i in range(logits.shape[1])),
        dtype=np.float32,
        count=logits.shape[1],
    )
    y_pred = (logits > th_arr[None, :]).astype(np.int32)
    y_true = y_true.astype(np.int32)
    tp = (y_pred & y_true).sum()
    fp = (y_pred & (1 - y_true)).sum()
    fn = ((1 - y_pred) & y_true).sum()
    denom = 2 * tp + fp + fn
    return (2 * tp / denom) if denom > 0 else 0.0




## === cell 8
def preload_test_images_u8(image_names, imgs_path, size):
    paths = [f"{imgs_path}/{str(x)}" for x in image_names]
    n = len(paths)
    arr = np.empty((n, size, size, 3), dtype=np.uint8)
    for i, p in enumerate(paths):
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {p}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
        arr[i] = img
    return arr


def u8_batch_to_tensor_chw_float(u8_hwc_batch, normalize=True):
    x = u8_hwc_batch.astype(np.float32, copy=False) / 255.0
    if normalize:
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
    x = np.ascontiguousarray(x.transpose(0, 3, 1, 2))  # NCHW
    return torch.from_numpy(x)  # float32 CPU


def apply_tta_u8(u8, tta):
    if tta == 1:
        return u8[:, ::-1, :, :]
    elif tta == 2:
        return u8[:, :, ::-1, :]
    elif tta == 3:
        return u8[:, ::-1, ::-1, :]
    return u8


TEST_U8 = None
TEST_TTA_TENSORS = {}

if use_external_model or use_train_fallback:
    TEST_U8 = preload_test_images_u8(
        df_sub["image"].tolist(), f"{DATA_PATH}/test_images", int(params["img_size"])
    )
    print("Preloaded test images:", TEST_U8.shape, TEST_U8.dtype)

    for tta in TTAS:
        u8_tta = apply_tta_u8(TEST_U8, int(tta))
        x_cpu = u8_batch_to_tensor_chw_float(u8_tta, normalize=True)  # float32 CPU
        if DEVICE.type == "cuda":
            x_cpu = x_cpu.pin_memory()
        TEST_TTA_TENSORS[int(tta)] = x_cpu
    print(
        "Prepared TTA tensors:",
        {k: tuple(v.shape) for k, v in TEST_TTA_TENSORS.items()},
    )

datasets, loaders = [], []
if use_external_model:
    print(
        "Skipping unused DataLoader construction for external model inference (inference uses precomputed TEST_TTA_TENSORS)."
    )



## === cell 9
if use_train_fallback:
    if params["backbone"] == "resnext":
        model = ResNext(params=params, out_dim=len(LABELS_))
    else:
        model = EffNet(params=params, out_dim=len(LABELS_))
    model = model.to(DEVICE).float()

    _shared_cache = {}
    _shared_cache_order = []

    trn_ds = PlantDataset(
        df=df_trn,
        size=params["img_size"],
        labels=label_to_idx,
        transform=None,
        tta=0,
        imgs_path=TRAIN_IMGS_PATH,
        normalize=True,
        shared_cache=_shared_cache,
        shared_cache_order=_shared_cache_order,
    )
    val_ds = PlantDataset(
        df=df_val,
        size=params["img_size"],
        labels=label_to_idx,
        transform=None,
        tta=0,
        imgs_path=TRAIN_IMGS_PATH,
        normalize=True,
        shared_cache=_shared_cache,
        shared_cache_order=_shared_cache_order,
    )

    TRAIN_WORKERS = min(WORKERS, 2) if DEVICE.type == "cuda" else 0

    trn_loader = torch.utils.data.DataLoader(
        trn_ds,
        batch_size=params["batch_size"],
        shuffle=True,
        num_workers=TRAIN_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
        persistent_workers=(TRAIN_WORKERS > 0),
        prefetch_factor=2 if TRAIN_WORKERS > 0 else None,
        drop_last=False,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=params["batch_size"],
        shuffle=False,
        num_workers=TRAIN_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
        persistent_workers=(TRAIN_WORKERS > 0),
        prefetch_factor=2 if TRAIN_WORKERS > 0 else None,
        drop_last=False,
    )

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    EPOCHS = 2 if DEVICE.type == "cuda" else 1

    for epoch in range(EPOCHS):
        model.train()
        tr_loss = 0.0
        n_seen = 0
        for xb, yb in trn_loader:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            bs = xb.size(0)
            tr_loss += float(loss.item()) * bs
            n_seen += bs
        tr_loss /= max(n_seen, 1)

        model.eval()
        va_loss = 0.0
        va_seen = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(DEVICE, non_blocking=True)
                yb = yb.to(DEVICE, non_blocking=True)
                out = model(xb)
                loss = criterion(out, yb)
                bs = xb.size(0)
                va_loss += float(loss.item()) * bs
                va_seen += bs
        va_loss /= max(va_seen, 1)
        print(
            f"epoch {epoch+1}/{EPOCHS} train_loss={tr_loss:.4f} val_loss={va_loss:.4f}"
        )

    model.eval()
    val_probs = []
    val_true = []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(DEVICE, non_blocking=True)
            prob = model(xb).sigmoid().detach().cpu().numpy()
            val_probs.append(prob)
            val_true.append(yb.detach().cpu().numpy())
    val_probs = np.concatenate(val_probs, axis=0)
    val_true = np.concatenate(val_true, axis=0)

    ths = {}
    grid = np.linspace(0.15, 0.85, 15)
    for i in range(val_probs.shape[1]):
        best_t = 0.5
        best_f1 = -1.0
        y_t = val_true[:, i].astype(np.int32)
        p = val_probs[:, i]
        for t in grid:
            y_p = (p > t).astype(np.int32)
            tp = (y_p & y_t).sum()
            fp = (y_p & (1 - y_t)).sum()
            fn = ((1 - y_p) & y_t).sum()
            denom = 2 * tp + fp + fn
            f1 = (2 * tp / denom) if denom > 0 else 0.0
            if f1 > best_f1:
                best_f1 = f1
                best_t = float(t)
        ths[str(i)] = best_t

    print(
        "Calibrated thresholds (first 10):", {k: ths[k] for k in list(ths.keys())[:10]}
    )

    models_list = [model]

    datasets, loaders = [], []
    print(
        "Fallback: skipping unused DataLoader construction (inference uses precomputed TEST_TTA_TENSORS)."
    )



## === cell 10
if len(models_list) > 0:
    n_imgs = len(df_sub)
    n_classes = len(LABELS_)

    pred_sum_t = torch.zeros((n_imgs, n_classes), dtype=torch.float32, device="cpu")
    n_preds = 0

    ths_local = ths if ths is not None else {}
    th_arr = np.fromiter(
        (ths_local.get(str(i), 0.5) for i in range(n_classes)),
        dtype=np.float32,
        count=n_classes,
    )
    idx2lab = [LABELS.get(str(i), str(i)) for i in range(n_classes)]

    old_bench = torch.backends.cudnn.benchmark
    if DEVICE.type == "cuda":
        torch.backends.cudnn.benchmark = True

    bs = int(params["batch_size"])
    use_channels_last = DEVICE.type == "cuda"

    with INFERENCE_CONTEXT():
        for i, model in enumerate(models_list):
            model.eval()
            if use_channels_last:
                model = model.to(memory_format=torch.channels_last)

            for tta in TTAS:
                x_all_cpu = TEST_TTA_TENSORS[int(tta)]
                if use_channels_last:
                    x_all_cpu = x_all_cpu.contiguous(memory_format=torch.channels_last)

                pos = 0
                while pos < n_imgs:
                    x_cpu = x_all_cpu[pos : pos + bs]
                    x = x_cpu.to(DEVICE, non_blocking=True)

                    pred = model(x).sigmoid().detach().to("cpu")
                    pred_sum_t[pos : pos + pred.shape[0]].add_(pred)
                    pos += pred.shape[0]

                n_preds += 1
                print(f"model {i} | tta {tta} -> done; accumulated")

    if DEVICE.type == "cuda":
        torch.backends.cudnn.benchmark = old_bench

    logits = (pred_sum_t / float(n_preds)).numpy()
    pred_mask = logits > th_arr[None, :]

    col_names = np.array(idx2lab, dtype=object)
    out_labels = ["healthy"] * n_imgs
    for r in range(n_imgs):
        idxs = np.flatnonzero(pred_mask[r])
        if idxs.size == 0:
            continue
        labs = col_names[idxs]
        if np.any(labs == "healthy"):
            continue
        out_labels[r] = " ".join(labs.tolist())

    df_sub["labels"] = out_labels
else:
    df_sub["labels"] = "healthy"

elapsed_time = time.time() - start_time
print(f"time elapsed: {elapsed_time // 60:.0f} min {elapsed_time % 60:.0f} sec")



## === cell 11
print("value counts:")
print(df_sub.labels.value_counts().head(20))
df_sub.head()



## === cell 12
df_sub = df_sub[["image", "labels"]].copy()

df_sub["labels"] = df_sub["labels"].fillna("healthy").astype(str).str.strip()
df_sub.loc[df_sub["labels"].eq(""), "labels"] = "healthy"

df_sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", df_sub.shape)



## === cell 13
assert os.path.isfile("submission.csv")
sub_check = pd.read_csv("submission.csv")
print("submission.csv columns:", sub_check.columns.tolist(), "shape:", sub_check.shape)
with open("submission.csv", "r") as f:
    for _ in range(5):
        print(f.readline().rstrip())
