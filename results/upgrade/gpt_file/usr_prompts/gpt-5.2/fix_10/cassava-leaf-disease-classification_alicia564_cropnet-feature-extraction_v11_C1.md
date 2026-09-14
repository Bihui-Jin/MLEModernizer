# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Classify each cassava image into four disease categories or a fifth category indicating a healthy leaf.

## Metric
Categorization accuracy.

## Submission Format
```
image_id,label
1000471002.jpg,4
1000840542.jpg,4
etc.
```

## Dataset
**[train/test]_images** the image files.

**train.csv**

- `image_id` the image file name.

- `label` the ID code for the disease.

**sample_submission.csv** A properly formatted sample submission, given the disclosed test set content.

- `image_id` the image file name.

- `label` the predicted ID code for the disease.

**[train/test]_tfrecords** the image files in tfrecord format.

**label_num_to_disease_map.json** The mapping between each disease code and the real disease name.

# 2. Python version

3.13

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        input/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        working/
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
```

-> data/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/cassava-leaf-disease-classification/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/cassava-leaf-disease-classification/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> data/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> input/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from PIL import Image
import io

os.environ.setdefault("PYTHONHASHSEED", "42")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True  # as in original

os.environ.setdefault("PILLOW_SILENCE_DEPRECATION", "1")
os.environ.pop("PILLOW_USE_CFFI_ACCESSOR", None)

Image.MAX_IMAGE_PIXELS = None
try:
    Image.preinit()
except Exception:
    pass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Torch:", torch.__version__, "Device:", device)

DATA_DIR = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_CSV_PATH = f"{DATA_DIR}/train.csv"
SAMPLE_SUB_PATH = f"{DATA_DIR}/sample_submission.csv"
TRAIN_IMG_DIR = f"{DATA_DIR}/train_images"
TEST_IMG_DIR = f"{DATA_DIR}/test_images"

assert os.path.exists(TRAIN_CSV_PATH), f"Missing {TRAIN_CSV_PATH}"
assert os.path.exists(SAMPLE_SUB_PATH), f"Missing {SAMPLE_SUB_PATH}"
assert os.path.isdir(TRAIN_IMG_DIR), f"Missing {TRAIN_IMG_DIR}"
assert os.path.isdir(TEST_IMG_DIR), f"Missing {TEST_IMG_DIR}"

train_df = pd.read_csv(TRAIN_CSV_PATH)
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

train_df["path"] = TRAIN_IMG_DIR.rstrip("/") + "/" + train_df["image_id"].astype(str)
train_df["label"] = train_df["label"].astype(int)

print(train_df.head())
print("Train rows:", len(train_df), "Test rows:", len(sample_sub))




## === cell 1
from sklearn.model_selection import train_test_split

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)),
    test_size=0.2,
    random_state=SEED,
    stratify=train_df["label"].values,
)

train_split = train_df.iloc[train_idx].reset_index(drop=True)
valid_split = train_df.iloc[valid_idx].reset_index(drop=True)

NUM_CLASSES = int(train_df["label"].nunique())
assert NUM_CLASSES == 5, f"Expected 5 classes for cassava, got {NUM_CLASSES}"
print("NUM_CLASSES:", NUM_CLASSES, "Train/Valid:", len(train_split), len(valid_split))




## === cell 2
from torchvision.transforms import v2 as T
from torchvision.transforms.functional import InterpolationMode

IMG_SIZE = 224
BATCH_SIZE = 32

CPU_COUNT = os.cpu_count() or 2

NUM_WORKERS = min(4, max(2, CPU_COUNT - 1))
PREFETCH_FACTOR = (
    4 if NUM_WORKERS > 0 else None
)  # improve overlap of decode/augment with GPU

train_tfms = T.Compose(
    [
        T.Resize((IMG_SIZE, IMG_SIZE), interpolation=InterpolationMode.BILINEAR),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomRotation(45, interpolation=InterpolationMode.BILINEAR),
        T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
        T.ToTensor(),  # converts to float32 in [0,1]
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)

valid_tfms = T.Compose(
    [
        T.Resize((IMG_SIZE, IMG_SIZE), interpolation=InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)


class CassavaDataset(Dataset):
    def __init__(
        self,
        df,
        transform=None,
        has_labels=True,
        cache_transformed=False,
        cache_decoded=False,
        img_size=224,  # kept for signature compatibility; resize moved to transforms above
    ):
        self.transform = transform
        self.has_labels = has_labels

        self.image_ids = df["image_id"].tolist() if "image_id" in df.columns else None
        self.paths = df["path"].tolist()
        self.labels = (
            df["label"].to_numpy(np.int64)
            if has_labels and "label" in df.columns
            else None
        )

        self.cache_transformed = bool(cache_transformed)
        self._cache = {} if self.cache_transformed else None

        self.cache_decoded = bool(cache_decoded)
        self._img_cache = {} if self.cache_decoded else None

    def __len__(self):
        return len(self.paths)

    def _load_image(self, img_path, idx=None):
        if self.cache_decoded and idx is not None:
            cached = self._img_cache.get(idx)
            if cached is not None:
                return Image.open(io.BytesIO(cached)).convert("RGB")

        with open(img_path, "rb") as f:
            b = f.read()

        if self.cache_decoded and idx is not None:
            self._img_cache[idx] = b

        return Image.open(io.BytesIO(b)).convert("RGB")

    def __getitem__(self, idx):
        if self.cache_transformed:
            cached = self._cache.get(idx)
            if cached is not None:
                return cached

        img = self._load_image(self.paths[idx], idx=idx)
        if self.transform:
            img = self.transform(img)

        if self.has_labels:
            out = (img, int(self.labels[idx]))
        else:
            out = (img, self.image_ids[idx])

        if self.cache_transformed:
            self._cache[idx] = out
        return out


train_ds = CassavaDataset(
    train_split,
    transform=train_tfms,
    has_labels=True,
    cache_transformed=False,
    cache_decoded=False,
    img_size=IMG_SIZE,
)
valid_ds = CassavaDataset(
    valid_split,
    transform=valid_tfms,
    has_labels=True,
    cache_transformed=False,
    cache_decoded=True,  # keep caching for valid; now much cheaper and safer
    img_size=IMG_SIZE,
)


def _seed_worker(worker_id):
    worker_seed = (SEED + worker_id) % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


g = torch.Generator()
g.manual_seed(SEED)

common_loader_kwargs = dict(
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(NUM_WORKERS > 0),
    prefetch_factor=PREFETCH_FACTOR,
    worker_init_fn=_seed_worker if NUM_WORKERS > 0 else None,
    generator=g,
)
common_loader_kwargs = {k: v for k, v in common_loader_kwargs.items() if v is not None}

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    **common_loader_kwargs,
)
valid_loader = DataLoader(
    valid_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    **common_loader_kwargs,
)

print(
    "CPU_COUNT:",
    CPU_COUNT,
    "NUM_WORKERS:",
    NUM_WORKERS,
    "PREFETCH_FACTOR:",
    PREFETCH_FACTOR,
)
print("Batches:", len(train_loader), len(valid_loader))




## === cell 3
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, NUM_CLASSES)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

try:
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=1, verbose=True
    )
except TypeError:
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=1
    )


def batch_correct(logits, y):
    return (logits.argmax(dim=1) == y).sum()


if hasattr(torch, "compile"):
    try:
        model = torch.compile(model, mode="reduce-overhead")
        print("torch.compile enabled")
    except Exception as e:
        print("torch.compile unavailable, continuing eager. Reason:", repr(e))

print(model)




## === cell 4
EPOCHS = 6  # keep identical

best_val_acc = -1.0
best_state = None
patience = 2
pat = 0

for epoch in range(1, EPOCHS + 1):
    model.train()
    tr_loss = 0.0
    tr_correct = 0
    n_tr = 0

    for xb, yb in train_loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        bs = yb.size(0)
        tr_loss += loss.item() * bs
        tr_correct += int(batch_correct(logits.detach(), yb).item())
        n_tr += bs

    tr_loss /= max(1, n_tr)
    tr_acc = tr_correct / max(1, n_tr)

    model.eval()
    va_loss = 0.0
    va_correct = 0
    n_va = 0

    with torch.inference_mode():
        for xb, yb in valid_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            logits = model(xb)
            loss = criterion(logits, yb)

            bs = yb.size(0)
            va_loss += loss.item() * bs
            va_correct += int(batch_correct(logits, yb).item())
            n_va += bs

    va_loss /= max(1, n_va)
    va_acc = va_correct / max(1, n_va)

    scheduler.step(va_acc)

    print(
        f"Epoch {epoch:02d}/{EPOCHS} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | val loss {va_loss:.4f} acc {va_acc:.4f}"
    )

    if va_acc > best_val_acc + 1e-6:
        best_val_acc = va_acc
        best_state = {
            k: v.detach().cpu().clone() for k, v in model.state_dict().items()
        }
        pat = 0
    else:
        pat += 1
        if pat >= patience:
            print(
                f"Early stopping triggered at epoch {epoch}. Best val acc: {best_val_acc:.4f}"
            )
            break

if best_state is not None:
    model.load_state_dict(best_state)
print("Best val acc:", best_val_acc)




## === cell 5
test_df = sample_sub.copy()

test_df["path"] = TEST_IMG_DIR.rstrip("/") + "/" + test_df["image_id"].astype(str)

alt_dir = os.path.join(TEST_IMG_DIR, "test_images")
if os.path.isdir(alt_dir):
    missing_mask = ~test_df["path"].map(os.path.exists)
    if int(missing_mask.sum()) > 0:
        test_df.loc[missing_mask, "path"] = (
            alt_dir.rstrip("/")
            + "/"
            + test_df.loc[missing_mask, "image_id"].astype(str)
        )

missing_paths = int((~test_df["path"].map(os.path.exists)).sum())
print("Missing test image files:", missing_paths)

test_ds = CassavaDataset(
    test_df[["image_id", "path"]],
    transform=valid_tfms,
    has_labels=False,
    cache_transformed=False,
    cache_decoded=True,  # reduces repeated decode overhead across workers
    img_size=IMG_SIZE,
)

test_loader = DataLoader(
    test_ds,
    batch_size=64,
    shuffle=False,
    **common_loader_kwargs,
)




## === cell 6
model.eval()
preds = []

with torch.inference_mode():
    for xb, ids in test_loader:
        xb = xb.to(device, non_blocking=True)
        logits = model(xb)
        pred = logits.argmax(dim=1).detach().cpu().numpy().astype(int).tolist()
        preds.extend(pred)

assert len(preds) == len(sample_sub), "Prediction count mismatch vs sample_submission."

submission_df = sample_sub.copy()
submission_df["label"] = np.asarray(preds, dtype=np.int64)

submission_path = "/kaggle/working/submission.csv"
submission_df.to_csv(submission_path, index=False)

print("Submission file created:", submission_path)
print(submission_df.head())

assert (
    submission_df.shape[0] == sample_sub.shape[0]
), "Row count mismatch vs sample_submission."
assert list(submission_df.columns) == [
    "image_id",
    "label",
], "Submission columns mismatch."
assert submission_path.endswith(".csv") and os.path.exists(
    submission_path
), "Submission file missing or wrong suffix."
assert submission_df["label"].between(0, 4).all(), "Labels must be integers in [0,4]."
print("Submission looks valid. Rows:", submission_df.shape[0])
