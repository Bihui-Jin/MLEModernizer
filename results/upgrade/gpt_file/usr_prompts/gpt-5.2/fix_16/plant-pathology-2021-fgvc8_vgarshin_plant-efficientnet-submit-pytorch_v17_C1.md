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

# 5. Target score

0.8065189289012012

# 6. Current score

0.24507

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.24507) has done: 'I remove the hard dependency on an external “plant-models-v101” dataset (which is missing) and instead make the notebook train the same ResNeXt50-based multilabel classifier from `train.csv` and `train_images`, then run inference on `test_images` and write `submission.csv`. I keep the core model architecture (ResNeXt50 + small FC head) and sigmoid multilabel semantics, and fix the label mapping/thresholding so it always outputs space-delimited labels with a safe “healthy” fallback. I also fix path resolution and ensure variables like `IMGS_PATH`, `TTAS`, and `FOLDS` are always defined so later cells don’t crash. Finally, I ensure the submission has exactly the required columns and `.csv` suffix.'
- What this solution (achieved 0.24507) has done: 'The timeout is dominated by CPU-side image decode/resize/normalize being repeated many times: every validation epoch and (more importantly) every TTA pass builds/iterates a separate DataLoader, causing 4× decoding overhead on the full test set. To fix this without changing the model or training semantics, I (1) build validation/test base caches once via a fast single-pass DataLoader that just materializes the cached `.npy` files, then reuse them; and (2) run all TTAs in one inference pass per batch by flipping the cached tensor batch in-memory instead of iterating four separate loaders. I also remove unnecessary per-sample `os.path.exists` checks by relying on a prebuilt cache index, and reduce Python overhead in label encoding with a pre-tokenized split. These changes preserve identical preprocessing, TTA definitions, thresholds, and training loop behavior while cutting repeated I/O and DataLoader iteration time drastically.'
- What this solution (achieved 0.24507) has done: 'Your current score is far below the target, so we should safely increase it with minimal semantic changes. The biggest issue is label/threshold calibration for mean F1 in a multilabel setting: a fixed 0.5 threshold is usually too strict and yields lots of false negatives, crushing F1. I keep your exact model, loss, training loop, and TTA logic, but (1) compute per-class decision thresholds on the validation set to maximize F1 per class (with a small grid), and (2) use those learned thresholds at test time (still with the same “healthy” fallback and “remove healthy if others predicted” rule). This typically boosts multilabel F1 substantially without changing the core approach or adding extra training.'
- What this solution (achieved 0.24507) has done: 'Your current score is far below the target, so we make small, metric-aligned fixes that typically raise mean F1 without changing the model or training loop. The biggest bug hurting F1 is that your current threshold tuning optimizes *per-class* F1, while Kaggle evaluates *per-image mean F1* across labels; we tune a small set of **global thresholds** directly for per-image F1 on the validation set and use the best one at test time. We also fix TTA flipping to match your dataset semantics (TTA=1 should be vertical flip over height, TTA=2 horizontal flip over width), which otherwise makes TTA predictions inconsistent and degrades accuracy. Finally, we keep your “healthy fallback”/“remove healthy if others predicted” rule but apply it consistently during validation threshold selection so the tuned threshold matches submission post-processing.'
- What this solution (achieved 0.24507) has done: 'Main runtime is dominated by (1) precomputing and writing base-cache `.npy` files for val/test (full dataset pass with disk I/O), and (2) repeated CPU-side `.cpu().numpy()` transfers inside the inner TTA loops. To fit the 600s limit without changing the model or training/inference semantics, I remove the eager cache materialization passes (keep on-demand caching, which is equivalent) and replace per-batch NumPy accumulation with a single preallocated CPU torch tensor to accumulate probabilities, converting to NumPy only once at the end. I also keep the exact same TTAs, thresholds search grid, and postprocessing, and preserve determinism settings. These changes reduce redundant passes and drastically cut Python overhead and device↔host synchronization while producing the same predictions up to negligible float differences.'
- What this solution (achieved 0.24507) has done: 'Your score gap to the target is large (0.245 → 0.806), so we should improve F1 without changing the model/training core. The main bug is that your validation-time TTA loop is *double-flipping* because `val_ds` applies `tta` flips in `__getitem__`, and then the tuning code flips again, which corrupts probabilities and yields a bad threshold/top-k choice. I force validation/test datasets used for inference/tuning to return the **base (non-flipped) tensor** and keep *all* TTA flipping exclusively in the inference loops (same TTAs as you already use), so threshold tuning matches test-time behavior. Additionally, I fix the cache temp filename so `np.save()` doesn’t create `*.npy.npy` and then fail to `os.replace`, which silently disables caching and can destabilize/slow inference (without changing semantics).'
- What this solution (achieved 0.24507) has done: 'Your current score (0.245) is far below the target (0.806), so we should increase mean F1 with minimal, metric-aligned changes. The largest issue is the mismatch between how validation F1 is computed/tuned and Kaggle’s evaluation: Kaggle uses per-image F1 on **sets of labels**, while your current tuning compares raw multi-hot vectors (and uses a different “healthy fallback” logic than the submission). I change only the validation-time scoring/tuning to compute F1 the same way the submission is generated (space-delimited labels, remove-healthy-if-others, and healthy fallback), then tune a single global threshold/top-k against that exact metric and apply it at test time. This preserves your model, loss, training loop, preprocessing, and TTA definitions, but should move the score substantially toward the target.'
- What this solution (achieved 0.24507) has done: 'Your current score (0.245) is far below the target (0.806), so we should increase mean F1 with minimal, metric-aligned changes. The biggest likely issue is that validation threshold tuning doesn’t match Kaggle’s “set-of-labels per image” evaluation because it uses raw multi-hot vectors that still include the special `complex` handling and label co-occurrence patterns; we tune using the exact same string/set post-processing that we write to submission, and compute F1 on sets (not vectors) on the validation split. We also fix a critical dataset bug: when `labels is not None` (train/val), your `PlantDataset.__getitem__` currently returns the *base tensor only* and never applies `tta` (good) but also never applies the configured `tta` for inference datasets with labels—however your tuning loop flips tensors again, so we keep all flipping exclusively in the inference loop and ensure the dataset never flips for val/test (already intended) while making it explicit to avoid accidental future mismatch. Finally, we slightly expand the global threshold search grid (still cheap) because mean-F1 is very sensitive around low thresholds, and your current coarse grid can easily miss the good region, especially with only 2 epochs of training.'

# 9. Code solution

## === cell 0
import os
import gc
import time
import cv2
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
from torch.utils.data.sampler import SequentialSampler
from torch.utils.data import DataLoader

KAGGLE = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

start_time = time.time()


def _first_existing(paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


DATA_PATH = _first_existing(
    [
        "/kaggle/input/plant-pathology-2021-fgvc8",
        "/kaggle/input/plant-pathology-2021-fgvc8/plant-pathology-2021-fgvc8",
        "../input/plant-pathology-2021-fgvc8",
        "../input/plant-pathology-2021-fgvc8/plant-pathology-2021-fgvc8",
        "/kaggle/data/plant-pathology-2021-fgvc8",
        "../data/plant-pathology-2021-fgvc8",
        "/kaggle/input",
        "/kaggle/data",
        "../input",
        "../data",
        "./data/plant-pathology-2021-fgvc8",
    ]
)

if DATA_PATH is None:
    raise FileNotFoundError("Could not find competition data directory.")


def _resolve_file(fname):
    cands = [
        os.path.join(DATA_PATH, fname),
        os.path.join(DATA_PATH, "plant-pathology-2021-fgvc8", fname),
    ]
    out = _first_existing(cands)
    if out is None:
        raise FileNotFoundError(f"Could not find required file {fname} in {cands}")
    return out


def _resolve_dir(dname):
    cands = [
        os.path.join(DATA_PATH, dname),
        os.path.join(DATA_PATH, "plant-pathology-2021-fgvcvc8", dname),
        os.path.join(DATA_PATH, "plant-pathology-2021-fgvc8", dname),
    ]
    out = _first_existing(cands)
    if out is None:
        raise FileNotFoundError(f"Could not find required directory {dname} in {cands}")
    return out


TRAIN_CSV = _resolve_file("train.csv")
SAMPLE_SUB = _resolve_file("sample_submission.csv")
TRAIN_IMGS_PATH = _resolve_dir("train_images")
TEST_IMGS_PATH = _resolve_dir("test_images")

print("DATA_PATH:", DATA_PATH)
print("TRAIN_CSV:", TRAIN_CSV)
print("SAMPLE_SUB:", SAMPLE_SUB)
print("TRAIN_IMGS_PATH:", TRAIN_IMGS_PATH)
print("TEST_IMGS_PATH:", TEST_IMGS_PATH)
print("DEVICE:", DEVICE)

TTAS = [0, 1, 2, 3]
FOLDS = [0]

CACHE_DIR = os.path.join(
    "/kaggle/working" if os.path.exists("/kaggle/working") else ".", "_pp2021_cache"
)
os.makedirs(CACHE_DIR, exist_ok=True)
print("CACHE_DIR:", CACHE_DIR)



## === cell 1
df_train = pd.read_csv(TRAIN_CSV)
df_sub = pd.read_csv(SAMPLE_SUB)

all_labels = sorted(
    {lbl for s in df_train["labels"].astype(str).values for lbl in s.split()}
)
LABELS_ = all_labels[:]  # index -> label
LABELS = {l: i for i, l in enumerate(LABELS_)}  # label -> index

params = {
    "img_size": 256,
    "batch_size": 32,
    "dropout": 0.5,
    "mean": (0.485, 0.456, 0.406),
    "std": (0.229, 0.224, 0.225),
}

_cpu = os.cpu_count() or 4
WORKERS = min(6, max(2, _cpu // 2))

print("train rows:", len(df_train), "| test rows:", len(df_sub))
print("num classes:", len(LABELS_))
print("labels:", LABELS_)



## === cell 2
df_sub = df_sub[["image", "labels"]].copy()
df_sub["labels"] = "healthy"
print(df_sub.head())
print("test images:", len(df_sub))




## === cell 3
def flip(img, axis=0):
    if axis == 1:
        return img[::-1, :, :]
    elif axis == 2:
        return img[:, ::-1, :]
    elif axis == 3:
        return img[::-1, ::-1, :]
    else:
        return img


class PlantDataset(data.Dataset):
    def __init__(
        self,
        df,
        img_dir,
        size,
        labels,
        transform=None,
        tta=0,
        train_mode=False,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        cache_base=False,
        cache_dir=None,
        cache_tag="",
    ):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.size = int(size)
        self.labels = labels  # dict label->index or None
        self.transform = transform
        self.tta = int(tta)
        self.train_mode = bool(train_mode)

        self.cache_base = bool(cache_base) and (not self.train_mode)

        self.mean = np.asarray(mean, dtype=np.float32).reshape(3, 1, 1)
        self.std = np.asarray(std, dtype=np.float32).reshape(3, 1, 1)

        self.images = self.df["image"].astype(str).values

        if self.labels is not None:
            lbl_series = self.df["labels"].astype(str).values
            tokenized = [s.split() for s in lbl_series]
            num_classes = len(self.labels)
            y = np.zeros((len(self.images), num_classes), dtype=np.float32)
            for i, toks in enumerate(tokenized):
                for tok in toks:
                    j = self.labels.get(tok)
                    if j is not None:
                        y[i, j] = 1.0
            self._labels_tensor = torch.from_numpy(y)
            self._num_classes = num_classes
        else:
            self._labels_tensor = None
            self._num_classes = 0

        self._base_cache = {} if self.cache_base else None

        self.cache_dir = cache_dir if (self.cache_base and cache_dir) else None
        self.cache_tag = str(cache_tag) if cache_tag else "base"
        if self.cache_dir is not None:
            os.makedirs(self.cache_dir, exist_ok=True)
            self._cache_index = set(os.listdir(self.cache_dir))
        else:
            self._cache_index = None

    def __len__(self):
        return self.images.shape[0]

    def _cache_fname(self, index: int) -> str:
        img_name = self.images[index]
        return f"{self.cache_tag}_s{self.size}_{img_name}.npy"

    def _cache_path(self, index: int) -> str:
        return os.path.join(self.cache_dir, self._cache_fname(index))

    def _load_base_tensor(self, index: int) -> torch.Tensor:
        if self.cache_dir is not None:
            fname = self._cache_fname(index)
            if self._cache_index is not None and fname in self._cache_index:
                cpath = os.path.join(self.cache_dir, fname)
                arr = np.load(cpath, allow_pickle=False, mmap_mode="r")
                arr = np.asarray(arr, dtype=np.float32)
                return torch.from_numpy(arr)

        img_name = self.images[index]
        img_path = os.path.join(self.img_dir, img_name)

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Image not found/readable: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_LINEAR)

        img = img.astype(np.float32) * (1.0 / 255.0)

        if self.train_mode:
            r = np.random.rand()
            if r < 0.33:
                img = flip(img, axis=1)
            elif r < 0.66:
                img = flip(img, axis=2)

        img = img.transpose(2, 0, 1)  # CHW
        img = (img - self.mean) / self.std
        img = np.ascontiguousarray(img, dtype=np.float32)

        if self.cache_dir is not None:
            cpath = self._cache_path(index)
            tmp = cpath + f".tmp.{os.getpid()}"
            try:
                np.save(tmp, img, allow_pickle=False)  # writes tmp + ".npy"
                os.replace(tmp + ".npy", cpath)
                if self._cache_index is not None:
                    self._cache_index.add(os.path.basename(cpath))
            except Exception:
                try:
                    if os.path.exists(tmp + ".npy"):
                        os.remove(tmp + ".npy")
                except Exception:
                    pass

        return torch.from_numpy(img)

    def __getitem__(self, index):
        if self._base_cache is not None:
            base = self._base_cache.get(index)
            if base is None:
                base = self._load_base_tensor(index)
                self._base_cache[index] = base
        else:
            base = self._load_base_tensor(index)

        if self._labels_tensor is not None:
            return base, self._labels_tensor[index]

        if self.tta == 1:
            return torch.flip(base, dims=(1,))  # H (vertical)
        elif self.tta == 2:
            return torch.flip(base, dims=(2,))  # W (horizontal)
        elif self.tta == 3:
            return torch.flip(base, dims=(1, 2))
        else:
            return base


class ResNext(nn.Module):
    def __init__(self, params, out_dim):
        super(ResNext, self).__init__()
        self.rsnxt = torchvision.models.resnext50_32x4d(
            weights=torchvision.models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1
        )
        nc = self.rsnxt.fc.in_features
        self.rsnxt.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(nc, int(nc / 4)),
            nn.ReLU(),
            nn.Dropout(params["dropout"]),
            nn.Linear(int(nc / 4), out_dim),
        )

    def forward(self, x):
        return self.rsnxt(x)




## === cell 4
idx = np.arange(len(df_train))
rng = np.random.RandomState(42)
rng.shuffle(idx)
val_size = int(0.1 * len(idx))
val_idx = idx[:val_size]
trn_idx = idx[val_size:]

df_trn = df_train.iloc[trn_idx].reset_index(drop=True)
df_val = df_train.iloc[val_idx].reset_index(drop=True)

train_ds = PlantDataset(
    df_trn,
    TRAIN_IMGS_PATH,
    params["img_size"],
    LABELS,
    train_mode=True,
    mean=params["mean"],
    std=params["std"],
    cache_base=False,  # keep disabled: training uses randomness
)

val_cache_dir = os.path.join(CACHE_DIR, "val_base")
val_ds = PlantDataset(
    df_val,
    TRAIN_IMGS_PATH,
    params["img_size"],
    LABELS,
    train_mode=False,
    tta=0,
    mean=params["mean"],
    std=params["std"],
    cache_base=True,
    cache_dir=val_cache_dir,
    cache_tag="val",
)

_common_loader_kwargs = dict(
    num_workers=WORKERS,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(WORKERS > 0),
    prefetch_factor=(4 if WORKERS > 0 else None),
)

train_loader = DataLoader(
    train_ds,
    batch_size=params["batch_size"],
    shuffle=True,
    drop_last=True,
    **{k: v for k, v in _common_loader_kwargs.items() if v is not None},
)
val_loader = DataLoader(
    val_ds,
    batch_size=params["batch_size"],
    shuffle=False,
    **{k: v for k, v in _common_loader_kwargs.items() if v is not None},
)


def _materialize_base_cache(ds: PlantDataset, batch_size: int, loader_kwargs: dict):
    return


_materialize_base_cache(
    val_ds, batch_size=params["batch_size"], loader_kwargs=_common_loader_kwargs
)

model = ResNext(params, out_dim=len(LABELS_)).to(DEVICE)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

EPOCHS = 2

for epoch in range(EPOCHS):
    model.train()
    tr_loss = 0.0
    for xb, yb in train_loader:
        xb = xb.to(DEVICE, non_blocking=True)
        yb = yb.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        tr_loss += loss.item() * xb.size(0)

    tr_loss /= len(train_loader.dataset) if len(train_loader.dataset) else 1

    model.eval()
    va_loss = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            va_loss += loss.item() * xb.size(0)
    va_loss /= len(val_loader.dataset) if len(val_loader.dataset) else 1

    print(
        f"epoch {epoch+1}/{EPOCHS} | train loss: {tr_loss:.4f} | val loss: {va_loss:.4f}"
    )

models_list = [model]
gc.collect()




## === cell 5
def _apply_submission_postprocess(
    mask: np.ndarray, healthy_idx: int | None
) -> np.ndarray:
    if healthy_idx is None:
        return mask
    mask2 = mask.copy()
    if mask2.ndim != 2:
        return mask2
    has_other = mask2.sum(axis=1) > mask2[:, healthy_idx]
    mask2[has_other, healthy_idx] = False
    return mask2


def _mask_from_probs_with_topk(
    probs: np.ndarray,
    th: float,
    topk: int | None,
    healthy_idx: int | None,
) -> np.ndarray:
    base = probs > float(th)

    if topk is not None and int(topk) > 0:
        k = int(topk)
        top_idx = np.argpartition(-probs, kth=min(k - 1, probs.shape[1] - 1), axis=1)[
            :, :k
        ]
        keep = np.zeros_like(base, dtype=bool)
        rows = np.arange(probs.shape[0])[:, None]
        keep[rows, top_idx] = True
        base = base & keep

    base = _apply_submission_postprocess(base, healthy_idx=healthy_idx)
    return base


def _mask_with_healthy_fallback(
    mask: np.ndarray, healthy_idx: int | None
) -> np.ndarray:
    if healthy_idx is None:
        return mask
    m = mask.copy()
    empty = m.sum(axis=1) == 0
    if np.any(empty):
        m[empty, healthy_idx] = True
    return m


def _labels_from_mask_row(mask_row: np.ndarray, labels_list, healthy_idx: int | None):
    idxs = np.flatnonzero(mask_row)
    if idxs.size == 0:
        return ("healthy",)
    if healthy_idx is not None and idxs.size > 1:
        idxs = idxs[idxs != healthy_idx]
        if idxs.size == 0:
            return ("healthy",)
    return tuple(labels_list[j] for j in idxs.tolist())


def _f1_imagewise_labelsets(true_sets, pred_sets, eps=1e-12) -> float:
    f1s = []
    for t, p in zip(true_sets, pred_sets):
        t = set(t)
        p = set(p)
        inter = len(t & p)
        denom = len(t) + len(p)
        f1s.append((2.0 * inter) / (denom + eps))
    return float(np.mean(f1s))


def tune_global_threshold_and_topk_on_val(
    models_list,
    val_ds: PlantDataset,
    loader_kwargs: dict,
    device,
    ttas=(0, 1, 2, 3),
    th_grid=None,
    topk_grid=None,
):
    if th_grid is None:
        th_grid = np.array(
            [
                0.02,
                0.04,
                0.06,
                0.08,
                0.10,
                0.12,
                0.14,
                0.16,
                0.18,
                0.20,
                0.22,
                0.24,
                0.26,
                0.28,
                0.30,
                0.32,
                0.34,
                0.36,
                0.38,
                0.40,
                0.42,
                0.44,
                0.46,
                0.48,
                0.50,
            ],
            dtype=np.float32,
        )
    if topk_grid is None:
        topk_grid = [None, 1, 2, 3, 4]

    val_inf_loader = DataLoader(
        val_ds,
        batch_size=params["batch_size"],
        shuffle=False,
        sampler=SequentialSampler(val_ds),
        **{k: v for k, v in loader_kwargs.items() if v is not None},
    )

    n_val = len(val_ds)
    n_classes = len(LABELS_)

    probs_sum_t = torch.zeros((n_val, n_classes), dtype=torch.float32, device="cpu")
    n_accum = 0

    prev_bench = torch.backends.cudnn.benchmark
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    healthy_idx = LABELS.get("healthy", None)

    true_sets = []
    for s in df_val["labels"].astype(str).values:
        toks = s.split()
        if len(toks) == 0:
            toks = ["healthy"]
        true_sets.append(tuple(toks))

    offset = 0
    with torch.inference_mode():
        for xb, _yb in val_inf_loader:
            bs = xb.size(0)

            if torch.cuda.is_available():
                xb = xb.to(device, non_blocking=True).to(
                    memory_format=torch.channels_last
                )
            else:
                xb = xb.to(device)

            for mdl in models_list:
                mdl.eval()
                for tj in ttas:
                    if tj == 1:
                        x_t = torch.flip(xb, dims=(2,))  # H
                    elif tj == 2:
                        x_t = torch.flip(xb, dims=(3,))  # W
                    elif tj == 3:
                        x_t = torch.flip(xb, dims=(2, 3))
                    else:
                        x_t = xb

                    probs = mdl(x_t).sigmoid().detach()
                    probs_sum_t[offset : offset + bs].add_(
                        probs.to("cpu", non_blocking=True)
                    )
                    n_accum += 1

            offset += bs

    torch.backends.cudnn.benchmark = prev_bench
    probs_mean = (probs_sum_t / float(n_accum)).numpy()

    best_th = 0.5
    best_topk = None
    best_f1 = -1.0

    for th in th_grid:
        for topk in topk_grid:
            mask = _mask_from_probs_with_topk(
                probs=probs_mean,
                th=float(th),
                topk=topk,
                healthy_idx=healthy_idx,
            )
            mask = _mask_with_healthy_fallback(mask, healthy_idx=healthy_idx)

            pred_sets = [
                _labels_from_mask_row(mask[i], LABELS_, healthy_idx=healthy_idx)
                for i in range(mask.shape[0])
            ]
            f1 = _f1_imagewise_labelsets(true_sets=true_sets, pred_sets=pred_sets)
            if f1 > best_f1:
                best_f1 = f1
                best_th = float(th)
                best_topk = topk

    return best_th, best_topk, best_f1


GLOBAL_TH, GLOBAL_TOPK, VAL_F1_AT_TUNED = tune_global_threshold_and_topk_on_val(
    models_list=models_list,
    val_ds=val_ds,
    loader_kwargs=_common_loader_kwargs,
    device=DEVICE,
    ttas=TTAS,
)

print("Tuned GLOBAL threshold:", GLOBAL_TH)
print("Tuned GLOBAL topk cap:", GLOBAL_TOPK)
print(
    "Val mean F1 at tuned (th, topk) w/ submission-style postprocess:", VAL_F1_AT_TUNED
)

PER_CLASS_TH = np.full((len(LABELS_),), GLOBAL_TH, dtype=np.float32)
healthy_idx = LABELS.get("healthy", None)



## === cell 6
test_cache_dir = os.path.join(CACHE_DIR, "test_base")
test_base_ds = PlantDataset(
    df=df_sub,
    img_dir=TEST_IMGS_PATH,
    size=params["img_size"],
    labels=None,
    transform=None,
    tta=0,
    train_mode=False,
    mean=params["mean"],
    std=params["std"],
    cache_base=True,
    cache_dir=test_cache_dir,
    cache_tag="test",
)

_materialize_base_cache(
    test_base_ds, batch_size=params["batch_size"], loader_kwargs=_common_loader_kwargs
)

test_loader = DataLoader(
    test_base_ds,
    batch_size=params["batch_size"],
    sampler=SequentialSampler(test_base_ds),
    **{k: v for k, v in _common_loader_kwargs.items() if v is not None},
)

print("Num TTAs:", len(TTAS), "| Num models:", len(models_list))



## === cell 7
n_test = len(df_sub)
n_classes = len(LABELS_)

probs_sum_t = torch.zeros((n_test, n_classes), dtype=torch.float32, device="cpu")
n_accum = 0

_infer_benchmark_prev = torch.backends.cudnn.benchmark
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

with torch.inference_mode():
    for mi, mdl in enumerate(models_list):
        mdl.eval()
        if torch.cuda.is_available():
            try:
                mdl = mdl.to(memory_format=torch.channels_last)
            except Exception:
                pass

        offset = 0
        for xb in test_loader:
            bs = xb.size(0)
            if torch.cuda.is_available():
                xb = xb.to(DEVICE, non_blocking=True).to(
                    memory_format=torch.channels_last
                )
            else:
                xb = xb.to(DEVICE)

            for tj in TTAS:
                if tj == 1:
                    x_t = torch.flip(xb, dims=(2,))  # H
                elif tj == 2:
                    x_t = torch.flip(xb, dims=(3,))  # W
                elif tj == 3:
                    x_t = torch.flip(xb, dims=(2, 3))
                else:
                    x_t = xb

                probs = mdl(x_t).sigmoid().detach()
                probs_sum_t[offset : offset + bs].add_(
                    probs.to("cpu", non_blocking=True)
                )
                n_accum += 1

            offset += bs

        print(f"model {mi} -> accumulated {n_accum} (model*tta*batches)")

torch.backends.cudnn.benchmark = _infer_benchmark_prev

probs_mean = (probs_sum_t / float(n_accum)).numpy()

mask = _mask_from_probs_with_topk(
    probs=probs_mean,
    th=float(GLOBAL_TH),
    topk=GLOBAL_TOPK,
    healthy_idx=healthy_idx,
)

out = []
for i in range(n_test):
    idxs = np.flatnonzero(mask[i])
    if idxs.size == 0:
        out.append("healthy")
        continue
    if healthy_idx is not None and idxs.size > 1:
        idxs = idxs[idxs != healthy_idx]
        if idxs.size == 0:
            out.append("healthy")
            continue
    out.append(" ".join(LABELS_[j] for j in idxs.tolist()))

df_sub["labels"] = out

elapsed_time = time.time() - start_time
print(f"time elapsed: {elapsed_time // 60:.0f} min {elapsed_time % 60:.0f} sec")



## === cell 8
print("value counts:")
print(df_sub["labels"].value_counts().head(20))
df_sub.head()



## === cell 9
sub_path = "submission.csv"
df_sub[["image", "labels"]].to_csv(sub_path, index=False)
print("Wrote:", sub_path, "| rows:", len(df_sub))
print(df_sub.head())
print("Submission columns:", list(pd.read_csv(sub_path).columns))
