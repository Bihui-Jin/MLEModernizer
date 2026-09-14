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
import json
import time
import cv2
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.utils.data as data
from torch.utils.data.sampler import SequentialSampler

import torchvision
from torchvision import models

torch.backends.cudnn.benchmark = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

KAGGLE = True
print(
    "torch:",
    torch.__version__,
    "| torchvision:",
    torchvision.__version__,
    "| device:",
    DEVICE,
)



## === cell 1
TEST = True
VER = "v100"

if KAGGLE:
    DATA_PATH = "../input/plant-pathology-2021-fgvc8"
    _input_root = "../input"
    _candidates = []
    if os.path.isdir(_input_root):
        for d in os.listdir(_input_root):
            if d.startswith("plant-models-") and os.path.isdir(
                os.path.join(_input_root, d)
            ):
                _candidates.append(d)
    _candidates = sorted(_candidates)
    if len(_candidates) > 0:
        MDLS_PATH = os.path.join(_input_root, _candidates[-1])
        print("auto-detected MDLS_PATH:", MDLS_PATH)
    else:
        MDLS_PATH = f"../input/plant-models-{VER}"
        print(
            "WARNING: could not auto-detect plant-models-*; using fallback MDLS_PATH:",
            MDLS_PATH,
        )
else:
    DATA_PATH = "./data"
    MDLS_PATH = f"./models_{VER}"

TH = 0.4
TTAS = [0, 1, 2, 3]
FOLDS = [0]

IMGS_PATH = f"{DATA_PATH}/test_images" if TEST else f"{DATA_PATH}/train_images"
start_time = time.time()

print("DATA_PATH:", DATA_PATH)
print("IMGS_PATH:", IMGS_PATH)
print("MDLS_PATH:", MDLS_PATH)




## === cell 2
def _derive_labels_from_train(train_csv_path: str):
    df = pd.read_csv(train_csv_path)
    uniq = set()
    for s in df["labels"].astype(str).values:
        for t in s.split():
            uniq.add(t)
    uniq = sorted(list(uniq))
    labels_ = {k: i for i, k in enumerate(uniq)}  # str -> int
    labels = {str(i): k for k, i in labels_.items()}  # str(int) -> str
    return labels_, labels


params_path = f"{MDLS_PATH}/params.json"
train_csv_path = f"{DATA_PATH}/train.csv"

if os.path.exists(params_path):
    with open(params_path) as file:
        params = json.load(file)
    LABELS_ = params["labels_"]
    LABELS = params["labels"]
    if KAGGLE:
        WORKERS = min(8, os.cpu_count() or 2)
    else:
        WORKERS = int(params.get("workers", 2))
    CLASS_TH = params.get("class_th", None)
    print("loaded params from:", params_path)
else:
    LABELS_, LABELS = _derive_labels_from_train(train_csv_path)
    params = {
        "img_size": 512,
        "batch_size": 16,
        "dropout": 0.3,
        "backbone": "efficientnet_b0",
        "workers": 2,
    }
    WORKERS = 2
    CLASS_TH = None
    print("params.json not found; using derived labels + defaults")

print("n_classes:", len(LABELS_))
print("example class mapping:", list(LABELS_.items())[:5])
if CLASS_TH is not None:
    print("class_th provided in params.json (will use per-class thresholds if usable)")



## === cell 3
sub_path = f"{DATA_PATH}/sample_submission.csv"
df_sub = pd.read_csv(sub_path)
if "image" not in df_sub.columns or "labels" not in df_sub.columns:
    raise ValueError("sample_submission.csv must have columns: image, labels")

df_sub["labels"] = "healthy"
print("submission template shape:", df_sub.shape)
df_sub.head()




## === cell 4
def flip(img, axis=0):
    if axis == 1:
        return img[::-1, :, :]
    elif axis == 2:
        return img[:, ::-1, :]
    elif axis == 3:
        return img[::-1, ::-1, :]
    else:
        return img


def _resize_shorter_side(img, target):
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return img
    if h < w:
        new_h = target
        new_w = int(round(w * (target / h)))
    else:
        new_w = target
        new_h = int(round(h * (target / w)))
    return cv2.resize(img, (new_w, new_h))


def _center_crop(img, size):
    h, w = img.shape[:2]
    if h < size or w < size:
        img = cv2.resize(img, (max(w, size), max(h, size)))
        h, w = img.shape[:2]
    y0 = (h - size) // 2
    x0 = (w - size) // 2
    return img[y0 : y0 + size, x0 : x0 + size, :]


class PlantDataset(data.Dataset):
    def __init__(
        self, df, size, labels, transform=None, tta=0, imgs_path=None, cache_base=False
    ):
        self.df = df.reset_index(drop=True)
        self.size = int(size)
        self.labels = labels
        self.transform = transform
        self.tta = int(tta)
        self.imgs_path = imgs_path

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        self.is_test = self.labels is None
        self.cache_base = bool(cache_base and self.is_test)
        self._cache = {} if self.cache_base else None

    def __len__(self):
        return self.df.shape[0]

    def _load_base_hwc_float01(self, index: int):
        if self._cache is not None:
            key = str(self.df.iloc[index].image)
            x = self._cache.get(key, None)
            if x is not None:
                return x

        row = self.df.iloc[index]
        img_name = row.image
        img_path = f"{self.imgs_path}/{img_name}"
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found/readable: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = _resize_shorter_side(img, self.size)
        img = _center_crop(img, self.size)
        img = np.ascontiguousarray(img)

        img = img.astype(np.float32) / 255.0  # HWC float in [0,1]

        if self._cache is not None:
            self._cache[key] = img
        return img

    def _hwc_to_chw_norm(self, img_hwc_float01: np.ndarray):
        img = (img_hwc_float01 - self.mean) / self.std
        img = img.transpose(2, 0, 1)  # CHW
        return np.ascontiguousarray(img)

    def __getitem__(self, index):
        if self.is_test:
            img = self._load_base_hwc_float01(index)

            if self.tta == 1:
                img = img[::-1, :, :]  # vertical flip (H)
            elif self.tta == 2:
                img = img[:, ::-1, :]  # horizontal flip (W)
            elif self.tta == 3:
                img = img[::-1, ::-1, :]  # both
            img = np.ascontiguousarray(img)

            img = self._hwc_to_chw_norm(img)
            return torch.tensor(img)

        row = self.df.iloc[index]
        img_name = row.image
        img_path = f"{self.imgs_path}/{img_name}"
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found/readable: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = _resize_shorter_side(img, self.size)
        img = _center_crop(img, self.size)
        img = np.ascontiguousarray(img)

        img = img.astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        img = img.transpose(2, 0, 1)

        label = np.zeros(len(self.labels), dtype=np.float32)
        for lbl in str(row.labels).split():
            label[self.labels[lbl]] = 1.0
        return torch.tensor(img), torch.tensor(label)




## === cell 5
class EffNet(nn.Module):
    def __init__(self, params, out_dim):
        super().__init__()
        backbone = params.get("backbone", "efficientnet_b0")

        if backbone != "efficientnet_b0":
            backbone = "efficientnet_b0"

        self.enet = models.efficientnet_b0(weights=None)

        nc = self.enet.classifier[-1].in_features
        self.enet.classifier = nn.Identity()

        self.myfc = nn.Sequential(
            nn.Dropout(float(params.get("dropout", 0.3))),
            nn.Linear(nc, int(nc / 4)),
            nn.Dropout(float(params.get("dropout", 0.3))),
            nn.Linear(int(nc / 4), out_dim),
        )

    def extract(self, x):
        return self.enet(x)

    def forward(self, x):
        x = self.extract(x)
        x = self.myfc(x)
        return x




## === cell 6
def _unwrap_state_dict(ckpt_obj):
    if isinstance(ckpt_obj, dict):
        for k in ["state_dict", "model", "net", "model_state_dict"]:
            if k in ckpt_obj and isinstance(ckpt_obj[k], dict):
                return ckpt_obj[k]
    return ckpt_obj


def _strip_module_prefix(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict
    if len(state_dict) == 0:
        return state_dict
    has_module = any(str(k).startswith("module.") for k in state_dict.keys())
    if not has_module:
        return state_dict
    return {str(k)[7:]: v for k, v in state_dict.items()}


def _remap_head_keys_for_effnet(state_dict: dict):
    if not isinstance(state_dict, dict) or len(state_dict) == 0:
        return state_dict

    out = {}
    for k, v in state_dict.items():
        nk = k

        if nk.startswith("enet.classifier."):
            nk = "myfc." + nk[len("enet.classifier.") :]
        elif nk.startswith("classifier."):
            nk = "myfc." + nk[len("classifier.") :]

        if nk.startswith("head."):
            nk = "myfc." + nk[len("head.") :]
        elif nk.startswith("fc."):
            nk = "myfc." + nk[len("fc.") :]

        out[nk] = v
    return out


models_list = []
for n_fold in FOLDS:
    model = EffNet(params, out_dim=len(LABELS_))
    path = f"{MDLS_PATH}/model_best_{n_fold}.pth"
    if os.path.exists(path):
        ckpt = torch.load(path, map_location="cpu")
        state_dict = _unwrap_state_dict(ckpt)
        state_dict = _strip_module_prefix(state_dict)
        state_dict = _remap_head_keys_for_effnet(state_dict)

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        n_total = len(model.state_dict())
        n_loaded = n_total - len(missing)

        print(f"loaded weights: {path} | loaded_tensors: {n_loaded}/{n_total}")
        if n_loaded < max(5, int(0.2 * n_total)):
            print(
                "WARNING: very few tensors loaded; predictions may be close to random. "
                "Check that MDLS_PATH points to the correct plant-models-* dataset."
            )
        if len(missing) > 0:
            print(
                "WARNING: missing keys when loading:",
                missing[:10],
                ("..." if len(missing) > 10 else ""),
            )
        if len(unexpected) > 0:
            print(
                "WARNING: unexpected keys when loading:",
                unexpected[:10],
                ("..." if len(unexpected) > 10 else ""),
            )
        del ckpt, state_dict
    else:
        print("WARNING: weights not found, using random init:", path)

    model = model.to(DEVICE).float().eval()
    models_list.append(model)

gc.collect()



## === cell 7
datasets, loaders = [], []

base_dataset = PlantDataset(
    df=df_sub,
    size=params["img_size"],
    labels=None,
    transform=None,
    tta=0,
    imgs_path=IMGS_PATH,
    cache_base=True,
)

for tta in TTAS:
    if tta == 0:
        dataset = base_dataset
    else:
        dataset = PlantDataset(
            df=df_sub,
            size=params["img_size"],
            labels=None,
            transform=None,
            tta=tta,
            imgs_path=IMGS_PATH,
            cache_base=False,
        )
        dataset._cache = base_dataset._cache

    datasets.append(dataset)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(params["batch_size"]),
        sampler=SequentialSampler(dataset),
        num_workers=int(WORKERS),
        pin_memory=(DEVICE.type == "cuda"),
        persistent_workers=(int(WORKERS) > 0),
        prefetch_factor=2 if int(WORKERS) > 0 else None,
    )
    loaders.append(loader)

len(df_sub), len(loaders)




## === cell 8
def get_labels(row, labels, th, class_th=None):
    idx = []
    for i, x in enumerate(row):
        _th = float(class_th[i]) if class_th is not None else float(th)
        if x > _th:
            idx.append(i)

    if len(idx) == 0:
        return "healthy"

    names = [labels[str(i)] for i in idx]

    if ("healthy" in names) and (len(names) > 1):
        names = [n for n in names if n != "healthy"]

    if len(names) == 0:
        return "healthy"

    return " ".join(names)


def _infer_logits_for_df(df_images, imgs_path, cache_base=True):
    ds_list, ld_list = [], []
    base_ds = PlantDataset(
        df=df_images,
        size=params["img_size"],
        labels=None,
        transform=None,
        tta=0,
        imgs_path=imgs_path,
        cache_base=cache_base,
    )
    for tta in TTAS:
        if tta == 0:
            dset = base_ds
        else:
            dset = PlantDataset(
                df=df_images,
                size=params["img_size"],
                labels=None,
                transform=None,
                tta=tta,
                imgs_path=imgs_path,
                cache_base=False,
            )
            dset._cache = base_ds._cache
        ds_list.append(dset)
        ld = torch.utils.data.DataLoader(
            dset,
            batch_size=int(params["batch_size"]),
            sampler=SequentialSampler(dset),
            num_workers=int(WORKERS),
            pin_memory=(DEVICE.type == "cuda"),
            persistent_workers=(int(WORKERS) > 0),
            prefetch_factor=2 if int(WORKERS) > 0 else None,
        )
        ld_list.append(ld)

    n = len(df_images)
    c = len(LABELS_)
    pred_sum_local = np.zeros((n, c), dtype=np.float32)
    n_runs_local = 0

    with torch.inference_mode():
        for model in models_list:
            for loader in ld_list:
                all_preds = np.empty((n, c), dtype=np.float32)
                offset = 0
                for img_data in loader:
                    bs = img_data.size(0)
                    img_data = img_data.to(DEVICE, non_blocking=True)
                    preds = (
                        model(img_data)
                        .sigmoid()
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.float32, copy=False)
                    )
                    all_preds[offset : offset + bs] = preds
                    offset += bs
                if offset != n:
                    raise ValueError(
                        f"Pred fill mismatch: filled {offset}, expected {n}"
                    )
                pred_sum_local += all_preds
                n_runs_local += 1

    return pred_sum_local / max(n_runs_local, 1)


def _labels_to_multi_hot(series_labels, labels_):
    y = np.zeros((len(series_labels), len(labels_)), dtype=np.int8)
    for r, s in enumerate(series_labels.astype(str).values):
        for t in s.split():
            if t in labels_:
                y[r, labels_[t]] = 1
    return y


def _mean_f1_multilabel(y_true_bin, y_pred_bin, eps=1e-9):
    tp = (y_true_bin & y_pred_bin).sum(axis=1).astype(np.float32)
    fp = ((1 - y_true_bin) & y_pred_bin).sum(axis=1).astype(np.float32)
    fn = (y_true_bin & (1 - y_pred_bin)).sum(axis=1).astype(np.float32)
    f1 = (2 * tp) / (2 * tp + fp + fn + eps)
    return float(f1.mean())


def _apply_thresholds(probs, th_arr):
    return (probs > th_arr[None, :]).astype(np.int8)


def _search_class_thresholds_oof(df_train, max_images_per_fold=1200):
    n = len(df_train)
    n_folds_eff = max(1, len(FOLDS))
    fold_id = np.arange(n) % n_folds_eff  # deterministic, uses len(FOLDS)

    probs_oof = np.zeros((n, len(LABELS_)), dtype=np.float32)

    img_root = f"{DATA_PATH}/train_images"
    for fold in range(n_folds_eff):
        idx_all = np.where(fold_id == fold)[0]
        if len(idx_all) == 0:
            continue

        if len(idx_all) > int(max_images_per_fold):
            sel_idx = idx_all[: int(max_images_per_fold)]
        else:
            sel_idx = idx_all

        df_fold = df_train.iloc[sel_idx].reset_index(drop=True)
        probs = _infer_logits_for_df(df_fold, imgs_path=img_root, cache_base=True)

        probs_oof[sel_idx] = probs
        print(f"OOF inference fold {fold}: {len(sel_idx)} images")

    y_true = _labels_to_multi_hot(df_train["labels"], LABELS_)

    filled = (probs_oof.sum(axis=1) > 0).astype(bool)
    if filled.sum() < 50:
        print("WARNING: too few OOF predictions computed; skipping threshold search.")
        return None

    probs_use = probs_oof[filled]
    y_use = y_true[filled]

    grid = np.array(
        [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55], dtype=np.float32
    )
    best_global_th = float(TH)
    best_global_score = -1.0
    for t in grid:
        pred = (probs_use > t).astype(np.int8)
        sc = _mean_f1_multilabel(y_use, pred)
        if sc > best_global_score:
            best_global_score = sc
            best_global_th = float(t)
    print("OOF best global TH:", best_global_th, "| meanF1:", best_global_score)

    th_arr = np.full((len(LABELS_),), best_global_th, dtype=np.float32)
    base_pred = _apply_thresholds(probs_use, th_arr)
    base_score = _mean_f1_multilabel(y_use, base_pred)

    for ci in range(len(LABELS_)):
        best_t = th_arr[ci]
        best_s = base_score
        for t in grid:
            th_try = th_arr.copy()
            th_try[ci] = t
            pred_try = _apply_thresholds(probs_use, th_try)
            sc = _mean_f1_multilabel(y_use, pred_try)
            if sc > best_s:
                best_s = sc
                best_t = float(t)
        th_arr[ci] = best_t
        base_score = best_s

    print("OOF per-class threshold search done | meanF1:", base_score)
    return th_arr


class_th_arr = None
c = len(LABELS_)
if CLASS_TH is not None:
    if isinstance(CLASS_TH, dict):
        class_th_arr = np.array(
            [float(CLASS_TH.get(LABELS[str(i)], TH)) for i in range(c)],
            dtype=np.float32,
        )
    elif isinstance(CLASS_TH, (list, tuple)) and len(CLASS_TH) == c:
        class_th_arr = np.array([float(x) for x in CLASS_TH], dtype=np.float32)
    else:
        class_th_arr = None
        print(
            "WARNING: class_th found but not in a usable format; will try OOF search or fall back to global TH"
        )

if class_th_arr is None:
    try:
        df_train_full = pd.read_csv(train_csv_path)
        class_th_arr = _search_class_thresholds_oof(
            df_train_full, max_images_per_fold=1200
        )
    except Exception as e:
        print("WARNING: OOF threshold search failed; using global TH. Error:", repr(e))
        class_th_arr = None

n = len(df_sub)
pred_sum = np.zeros((n, c), dtype=np.float32)
n_runs = 0

with torch.inference_mode():
    for mi, model in enumerate(models_list):
        for ti, loader in enumerate(loaders):
            all_preds = np.empty((n, c), dtype=np.float32)
            offset = 0
            for img_data in loader:
                bs = img_data.size(0)
                img_data = img_data.to(DEVICE, non_blocking=True)
                preds = (
                    model(img_data)
                    .sigmoid()
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32, copy=False)
                )
                all_preds[offset : offset + bs] = preds
                offset += bs
            if offset != n:
                raise ValueError(f"Pred fill mismatch: filled {offset}, expected {n}")
            pred_sum += all_preds
            n_runs += 1
            print(f"model {mi} | tta {ti} done")

logits = pred_sum / max(n_runs, 1)

df_sub["labels"] = [get_labels(x, LABELS, TH, class_th=class_th_arr) for x in logits]

elapsed_time = time.time() - start_time
print(f"time elapsed: {elapsed_time // 60:.0f} min {elapsed_time % 60:.0f} sec")



## === cell 9
print("value counts:")
print(df_sub.labels.value_counts().head(20))
df_sub.head()



## === cell 10
out_path = "submission.csv"
df_sub[["image", "labels"]].to_csv(out_path, index=False)
print("wrote:", out_path, "| shape:", df_sub.shape)
print(df_sub.head())
