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
            num_classes = len(self.labels)
            y = np.zeros((len(self.images), num_classes), dtype=np.float32)
            for i, s in enumerate(lbl_series):
                for tok in s.split():
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

    def __len__(self):
        return self.images.shape[0]

    def _cache_path(self, index: int) -> str:
        img_name = self.images[index]
        base = f"{self.cache_tag}_s{self.size}_{img_name}.npy"
        return os.path.join(self.cache_dir, base)

    def _load_base_tensor(self, index: int) -> torch.Tensor:
        if self.cache_dir is not None:
            cpath = self._cache_path(index)
            if os.path.exists(cpath):
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
            try:
                tmp = cpath + f".tmp.{os.getpid()}"
                np.save(tmp, img, allow_pickle=False)
                os.replace(tmp, cpath)
            except Exception:
                try:
                    if os.path.exists(tmp):
                        os.remove(tmp)
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
            return torch.flip(base, dims=(1,))
        elif self.tta == 2:
            return torch.flip(base, dims=(2,))
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
    prefetch_factor=(
        4 if WORKERS > 0 else None
    ),  # Speed: slightly deeper prefetch to hide IO/resize.
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


class TTADataset(data.Dataset):
    def __init__(self, base_ds: PlantDataset, tta: int):
        self.base_ds = base_ds
        self.tta = int(tta)

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        x = self.base_ds[idx]  # tensor CHW normalized
        if self.tta == 1:
            return torch.flip(x, dims=(1,))
        elif self.tta == 2:
            return torch.flip(x, dims=(2,))
        elif self.tta == 3:
            return torch.flip(x, dims=(1, 2))
        return x


loaders = []
for tta in TTAS:
    dataset = TTADataset(test_base_ds, tta=tta)
    loader = DataLoader(
        dataset,
        batch_size=params["batch_size"],
        sampler=SequentialSampler(dataset),
        **{k: v for k, v in _common_loader_kwargs.items() if v is not None},
    )
    loaders.append(loader)

print("Num TTAs:", len(loaders), "| Num models:", len(models_list))



## === cell 6
TH = 0.5


def get_labels_from_probs(probs_row, labels_list, th=0.5):
    idxs = [i for i, p in enumerate(probs_row) if p > th]
    if not idxs:
        return "healthy"
    lbls = [labels_list[i] for i in idxs]
    if "healthy" in lbls and len(lbls) > 1:
        lbls = [l for l in lbls if l != "healthy"]
    return " ".join(lbls) if lbls else "healthy"


n_test = len(df_sub)
n_classes = len(LABELS_)

probs_sum = np.zeros((n_test, n_classes), dtype=np.float32)
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

        for tj, loader in enumerate(loaders):
            offset = 0
            for img_data in loader:
                bs = img_data.size(0)
                if torch.cuda.is_available():
                    img_data = img_data.to(DEVICE, non_blocking=True).to(
                        memory_format=torch.channels_last
                    )
                else:
                    img_data = img_data.to(DEVICE)

                probs = mdl(img_data).sigmoid()
                probs_sum[offset : offset + bs] += probs.detach().cpu().numpy()
                offset += bs

            n_accum += 1
            print(f"model {mi} | tta {tj} -> accumulated")

torch.backends.cudnn.benchmark = _infer_benchmark_prev

probs_mean = probs_sum / float(n_accum)

mask = probs_mean > TH
out = []
healthy_idx = LABELS.get("healthy", None)

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



## === cell 7
print("value counts:")
print(df_sub["labels"].value_counts().head(20))
df_sub.head()



## === cell 8
sub_path = "submission.csv"
df_sub[["image", "labels"]].to_csv(sub_path, index=False)
print("Wrote:", sub_path, "| rows:", len(df_sub))
print(df_sub.head())
print("Submission columns:", list(pd.read_csv(sub_path).columns))
