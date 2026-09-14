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

0.8034164358264089

# 6. Current score

0.24507

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.24507) has done: 'I fix the path bug that makes training crash by ensuring the dataset uses its own `imgs_path` rather than the global `IMGS_PATH`, which was causing doubled `train_images/train_images/...` paths. I also fix the test image existence filtering that was incorrectly dropping almost all rows due to case-sensitivity and directory structure differences, which led to an empty submission and the row-count assertion failure. These changes keep the same model, inference, and thresholding logic, but make the pipeline run end-to-end and reliably produce a full-length `submission.csv`. Finally, I make the final submission align exactly to `sample_submission.csv` ordering to guarantee the correct row count and image IDs.'
- What this solution (achieved 0.24507) has done: 'The timeout is dominated by repeated JPEG decode+resize work across 4 TTAs (and possibly multiple checkpoints), because each TTA creates a separate Dataset/DataLoader that re-reads the same images from disk. I keep the exact same model, weights, TTAs, thresholding, and averaging logic, but change inference to load/normalize each image once and generate the 4 flips on-the-fly on GPU/CPU tensors (equivalent to the current flip+normalize). I also remove repeated `os.listdir` scans per TTA dataset and reduce Python overhead in the accumulation loop by preallocating and using a single loader pass. These changes preserve evaluation semantics while cutting I/O and preprocessing by ~4×, which is typically the difference between timing out and finishing under 600s.'

# 9. Code solution

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
from torchvision import models

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

KAGGLE = True
TEST = True
VER = "v5"

if KAGGLE:
    DATA_PATH = "../input/plant-pathology-2021-fgvc8"
    MDLS_PATH = f"../input/plant-models-{VER}"
else:
    DATA_PATH = "./data"
    MDLS_PATH = f"./models_{VER}"

TH = 0.5
TTAS = [0, 1, 2, 3]
FOLDS = [0]

IMGS_PATH = f"{DATA_PATH}/test_images" if TEST else f"{DATA_PATH}/train_images"
start_time = time.time()

torch.backends.cudnn.benchmark = True



## === cell 1
print("Using torchvision EfficientNet backbone; no external pip installs required.")




## === cell 2
def build_params_from_train_csv(data_path):
    train_csv = os.path.join(data_path, "train.csv")
    df = pd.read_csv(train_csv)
    all_labels = sorted(
        {l for s in df["labels"].astype(str).tolist() for l in s.split()}
    )
    labels_ = all_labels  # index -> label
    labels = {lbl: i for i, lbl in enumerate(labels_)}  # label -> index
    return {
        "labels_": labels_,
        "labels": labels,
        "img_size": 512,  # used for resize in inference
        "batch_size": 16,
        "dropout": 0.2,
        "backbone": "efficientnet_b0",
        "workers": 2,
    }


params_path = os.path.join(MDLS_PATH, "params.json")
if os.path.exists(params_path):
    with open(params_path, "r") as file:
        params = json.load(file)
    print("loaded params from:", params_path)
else:
    params = build_params_from_train_csv(DATA_PATH)
    print("WARNING: params.json not found at", params_path)
    print("Using fallback params derived from train.csv")

LABELS_ = params["labels_"]
LABELS = params["labels"]

_default_workers = int(params.get("workers", 2))
if KAGGLE:
    WORKERS = max(2, min(8, (os.cpu_count() or 4) // 2))
else:
    WORKERS = max(2, _default_workers)

print("num classes:", len(LABELS_))
print("dataloader workers:", WORKERS)




## === cell 3
def resolve_images_dir(base_path, split):
    d1 = os.path.join(base_path, split)
    if not os.path.isdir(d1):
        raise FileNotFoundError(f"Images directory not found: {d1}")

    d2 = os.path.join(d1, split)  # possible nested structure

    def has_jpg(d):
        try:
            for fn in os.listdir(d):
                if fn.lower().endswith(".jpg"):
                    return True
        except Exception:
            return False
        return False

    if has_jpg(d1):
        return d1
    if os.path.isdir(d2) and has_jpg(d2):
        return d2
    return d2 if os.path.isdir(d2) else d1


if TEST:
    IMGS_PATH = resolve_images_dir(DATA_PATH, "test_images")
    sub_path = os.path.join(DATA_PATH, "sample_submission.csv")
    df_sub = pd.read_csv(sub_path)
    assert set(df_sub.columns) >= {
        "image",
        "labels",
    }, f"Bad sample_submission columns: {df_sub.columns}"
    df_sub = df_sub[["image", "labels"]].copy()
else:
    IMGS_PATH = resolve_images_dir(DATA_PATH, "train_images")
    df_sub = pd.read_csv(os.path.join(DATA_PATH, "train.csv"))[
        ["image", "labels"]
    ].copy()

existing_lower = {
    fn.lower() for fn in os.listdir(IMGS_PATH) if fn.lower().endswith(".jpg")
}
missing_mask = ~df_sub["image"].astype(str).str.lower().isin(existing_lower)
missing = df_sub.loc[missing_mask, "image"]
if len(missing) > 0:
    print(
        f"WARNING: {len(missing)} images listed in csv not found under {IMGS_PATH} (first 5):",
        missing.head().tolist(),
    )

df_sub["labels"] = "healthy"
print("IMGS_PATH:", IMGS_PATH)
print("df_sub shape:", df_sub.shape)
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


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class PlantDataset(data.Dataset):
    def __init__(self, df, size, labels, transform=None, tta=0, imgs_path=None):
        self.df = df.reset_index(drop=True)
        self.size = int(size)
        self.labels = labels
        self.transform = transform
        self.tta = tta
        self.imgs_path = imgs_path if imgs_path is not None else IMGS_PATH

        self._lower_to_real = None
        try:
            self._lower_to_real = {
                fn.lower(): fn
                for fn in os.listdir(self.imgs_path)
                if fn.lower().endswith(".jpg")
            }
        except Exception:
            self._lower_to_real = None

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img_name = str(row.image)
        real_name = img_name
        if self._lower_to_real is not None:
            real_name = self._lower_to_real.get(img_name.lower(), img_name)

        img_path = os.path.join(self.imgs_path, real_name)
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found or unreadable: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.size, self.size))
        img = img.astype(np.float32) / 255.0

        if self.transform is not None:
            img = self.transform(image=img)["image"]

        if self.labels:
            img = img.transpose(2, 0, 1)
            label = np.zeros(len(self.labels), dtype=np.float32)
            for lbl in str(row.labels).split():
                if lbl in self.labels:
                    label[self.labels[lbl]] = 1.0
            return torch.tensor(img), torch.tensor(label)
        else:
            img = flip(img, axis=self.tta)
            img = (img - IMAGENET_MEAN) / IMAGENET_STD
            img = img.transpose(2, 0, 1)
            return torch.tensor(img.copy())


class PlantDatasetBaseInfer(data.Dataset):
    """Inference-only dataset that returns normalized CHW float32 tensor without any TTA applied."""

    def __init__(self, df, size, imgs_path=None, lower_to_real=None):
        self.df = df.reset_index(drop=True)
        self.size = int(size)
        self.imgs_path = imgs_path if imgs_path is not None else IMGS_PATH

        self._lower_to_real = lower_to_real
        if self._lower_to_real is None:
            try:
                self._lower_to_real = {
                    fn.lower(): fn
                    for fn in os.listdir(self.imgs_path)
                    if fn.lower().endswith(".jpg")
                }
            except Exception:
                self._lower_to_real = None

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img_name = str(row.image)
        real_name = img_name
        if self._lower_to_real is not None:
            real_name = self._lower_to_real.get(img_name.lower(), img_name)

        img_path = os.path.join(self.imgs_path, real_name)
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found or unreadable: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.size, self.size))
        img = img.astype(np.float32) / 255.0
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        img = img.transpose(2, 0, 1)
        return torch.from_numpy(img.copy())




## === cell 5
class EffNet(nn.Module):
    def __init__(self, params, out_dim):
        super().__init__()
        backbone = params.get("backbone", "efficientnet_b0")
        if backbone != "efficientnet_b0":
            backbone = "efficientnet_b0"

        self.enet = models.efficientnet_b0(weights=None)
        nc = self.enet.classifier[1].in_features
        self.enet.classifier = nn.Identity()

        dp = float(params.get("dropout", 0.2))
        self.myfc = nn.Sequential(
            nn.Dropout(dp),
            nn.Linear(nc, int(nc / 4)),
            nn.Dropout(dp),
            nn.Linear(int(nc / 4), out_dim),
        )

    def extract(self, x):
        return self.enet(x)

    def forward(self, x):
        x = self.extract(x)
        x = self.myfc(x)
        return x




## === cell 6
def discover_checkpoints(mdls_path):
    if not os.path.isdir(mdls_path):
        return []
    ckpts = []
    for fn in os.listdir(mdls_path):
        if fn.startswith("model_best_") and fn.endswith(".pth"):
            ckpts.append(os.path.join(mdls_path, fn))
    return sorted(ckpts)


models_list = []
all_missing = True

ckpt_paths = []
for n_fold in FOLDS:
    ckpt_paths.append(os.path.join(MDLS_PATH, f"model_best_{n_fold}.pth"))
if not any(os.path.exists(p) for p in ckpt_paths):
    ckpt_paths = discover_checkpoints(MDLS_PATH)

if len(ckpt_paths) == 0:
    print("WARNING: no checkpoints found under:", MDLS_PATH)
else:
    print("Found checkpoints:", len(ckpt_paths))
    for p in ckpt_paths[:10]:
        print(" -", p)

for ckpt_path in ckpt_paths if len(ckpt_paths) > 0 else [None]:
    model = EffNet(params, out_dim=len(LABELS_))
    if ckpt_path is not None and os.path.exists(ckpt_path):
        state_dict = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=True)
        all_missing = False
        print("loaded:", ckpt_path)
    else:
        if ckpt_path is not None:
            print("WARNING: checkpoint missing:", ckpt_path)

    model.eval()
    model.to(DEVICE)
    if DEVICE.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    models_list.append(model)

gc.collect()




## === cell 7
def seed_everything(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True  # kept as original default intent


def train_if_needed_and_return_models():
    if not all_missing:
        return models_list, False

    seed_everything(42)
    print(
        "No checkpoints found; training a fallback model from train.csv to improve score vs all-'healthy'."
    )

    train_csv = os.path.join(DATA_PATH, "train.csv")
    df_train = pd.read_csv(train_csv).copy()

    train_imgs_path = resolve_images_dir(DATA_PATH, "train_images")

    existing_lower = {
        fn.lower() for fn in os.listdir(train_imgs_path) if fn.lower().endswith(".jpg")
    }
    m = df_train["image"].astype(str).str.lower().isin(existing_lower)
    if (~m).sum() > 0:
        print(f"WARNING: dropping {(~m).sum()} training rows with missing images")
        df_train = df_train.loc[m].reset_index(drop=True)

    if len(df_train) < 2:
        print(
            "WARNING: training set too small after filtering; skipping training fallback."
        )
        model = EffNet(params, out_dim=len(LABELS_)).to(DEVICE)
        model.eval()
        return [model], True

    rng = np.random.RandomState(42)
    idx = np.arange(len(df_train))
    rng.shuffle(idx)
    split = int(0.9 * len(idx))
    split = min(max(split, 1), len(idx) - 1)  # ensure both sides non-empty
    tr_idx, va_idx = idx[:split], idx[split:]
    df_tr = df_train.iloc[tr_idx].reset_index(drop=True)
    df_va = df_train.iloc[va_idx].reset_index(drop=True)

    train_ds = PlantDataset(
        df=df_tr,
        size=params["img_size"],
        labels=LABELS,
        transform=None,
        tta=0,
        imgs_path=train_imgs_path,
    )
    val_ds = PlantDataset(
        df=df_va,
        size=params["img_size"],
        labels=LABELS,
        transform=None,
        tta=0,
        imgs_path=train_imgs_path,
    )

    dl_kwargs = dict(
        num_workers=WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(WORKERS > 0),
        prefetch_factor=4 if WORKERS > 0 else None,
    )
    if dl_kwargs["prefetch_factor"] is None:
        dl_kwargs.pop("prefetch_factor")

    bs = int(params["batch_size"])
    drop_last_train = len(train_ds) >= bs

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=bs,
        shuffle=True,
        drop_last=drop_last_train,
        **dl_kwargs,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=bs,
        shuffle=False,
        drop_last=False,
        **dl_kwargs,
    )

    model = EffNet(params, out_dim=len(LABELS_)).to(DEVICE)
    if DEVICE.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    model.train()

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-2)

    epochs = 2 if torch.cuda.is_available() else 1

    best_val = float("inf")
    best_state = None

    mean_t = torch.tensor(IMAGENET_MEAN, device=DEVICE).view(1, 3, 1, 1)
    std_t = torch.tensor(IMAGENET_STD, device=DEVICE).view(1, 3, 1, 1)

    for ep in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0
        n_tr = 0
        for xb, yb in train_loader:
            xb = xb.to(DEVICE, non_blocking=True).float()
            if DEVICE.type == "cuda":
                xb = xb.contiguous(memory_format=torch.channels_last)
            xb = (xb - mean_t) / std_t
            yb = yb.to(DEVICE, non_blocking=True).float()

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            bs_ = xb.size(0)
            tr_loss += loss.item() * bs_
            n_tr += bs_

        model.eval()
        va_loss = 0.0
        n_va = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(DEVICE, non_blocking=True).float()
                if DEVICE.type == "cuda":
                    xb = xb.contiguous(memory_format=torch.channels_last)
                xb = (xb - mean_t) / std_t
                yb = yb.to(DEVICE, non_blocking=True).float()
                logits = model(xb)
                loss = criterion(logits, yb)
                bs_ = xb.size(0)
                va_loss += loss.item() * bs_
                n_va += bs_

        tr_loss /= max(n_tr, 1)
        va_loss /= max(n_va, 1)
        print(
            f"epoch {ep}/{epochs} | train_loss={tr_loss:.4f} | val_loss={va_loss:.4f}"
        )

        if va_loss < best_val:
            best_val = va_loss
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)

    model.eval()
    return [model], True


models_list, trained_fallback = train_if_needed_and_return_models()
gc.collect()



## === cell 8
_lower_to_real_infer = None
try:
    _lower_to_real_infer = {
        fn.lower(): fn for fn in os.listdir(IMGS_PATH) if fn.lower().endswith(".jpg")
    }
except Exception:
    _lower_to_real_infer = None

infer_ds = PlantDatasetBaseInfer(
    df=df_sub,
    size=params["img_size"],
    imgs_path=IMGS_PATH,
    lower_to_real=_lower_to_real_infer,
)

dl_infer_kwargs = dict(
    num_workers=WORKERS,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(WORKERS > 0),
    prefetch_factor=4 if WORKERS > 0 else None,
)
if dl_infer_kwargs["prefetch_factor"] is None:
    dl_infer_kwargs.pop("prefetch_factor")

infer_loader = torch.utils.data.DataLoader(
    infer_ds,
    batch_size=int(params["batch_size"]),
    sampler=SequentialSampler(infer_ds),
    **dl_infer_kwargs,
)

len(df_sub), len(infer_loader)




## === cell 9
def get_labels(row_probs, idx2lbl, th):
    idxs = [i for i, x in enumerate(row_probs) if x > th]
    lbls = [idx2lbl[i] for i in idxs]
    lbls = "healthy" if ("healthy" in lbls or len(lbls) == 0) else " ".join(lbls)
    return lbls


IDX2LBL = {i: lbl for i, lbl in enumerate(LABELS_)}

if models_list is None or len(models_list) == 0:
    print("No models available; writing default 'healthy' submission.")
    df_sub["labels"] = "healthy"
else:
    n_images = len(df_sub)
    n_classes = len(LABELS_)
    probs_accum = np.zeros((n_images, n_classes), dtype=np.float32)
    n_passes = 0

    def apply_tta_chw(x, tta):
        if tta == 1:
            return x.flip(dims=(2,))  # vertical: H
        elif tta == 2:
            return x.flip(dims=(3,))  # horizontal: W
        elif tta == 3:
            return x.flip(dims=(2, 3))  # both
        else:
            return x

    with torch.no_grad():
        for i, model in enumerate(models_list):
            offset = 0
            for xb in infer_loader:
                xb = xb.to(DEVICE, non_blocking=True).float()
                if DEVICE.type == "cuda":
                    xb = xb.contiguous(memory_format=torch.channels_last)

                for tta in TTAS:
                    x_tta = apply_tta_chw(xb, tta)
                    preds = model(x_tta).sigmoid()
                    bs = preds.shape[0]
                    probs_accum[offset : offset + bs] += preds.detach().cpu().numpy()
                    n_passes += 1

                offset += xb.shape[0]
            print(f"model {i} | all TTAs -> done")

    probs_mean = probs_accum / float(n_passes)

    healthy_idx = LABELS.get("healthy", None)
    above = probs_mean > TH

    out_labels = []
    for r in range(n_images):
        if healthy_idx is not None and above[r, healthy_idx]:
            out_labels.append("healthy")
            continue
        idxs = np.flatnonzero(above[r])
        if idxs.size == 0:
            out_labels.append("healthy")
        else:
            out_labels.append(" ".join(IDX2LBL[int(k)] for k in idxs))
    df_sub["labels"] = out_labels

elapsed_time = time.time() - start_time
print(f"time elapsed: {elapsed_time // 60:.0f} min {elapsed_time % 60:.0f} sec")



## === cell 10
print("value counts:")
print(df_sub["labels"].value_counts().head(20))
df_sub.head()



## === cell 11
sample = pd.read_csv(os.path.join(DATA_PATH, "sample_submission.csv"))[
    ["image", "labels"]
].copy()
df_pred = df_sub[["image", "labels"]].copy()
df_pred = sample[["image"]].merge(df_pred, on="image", how="left")
df_pred["labels"] = df_pred["labels"].fillna("healthy")

df_pred.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", df_pred.shape)
print(df_pred.head())



## === cell 12
assert os.path.exists("submission.csv"), "submission.csv was not created"
sub = pd.read_csv("submission.csv")
assert list(sub.columns) == ["image", "labels"], f"Bad columns: {sub.columns}"
expected_n = pd.read_csv(os.path.join(DATA_PATH, "sample_submission.csv")).shape[0]
assert (
    sub.shape[0] == expected_n
), f"Row count mismatch vs sample_submission: got {sub.shape[0]}, expected {expected_n}"
assert sub["image"].isna().sum() == 0, "Found NaN image ids"
assert sub["labels"].isna().sum() == 0, "Found NaN labels"
print("Submission file is valid.")
