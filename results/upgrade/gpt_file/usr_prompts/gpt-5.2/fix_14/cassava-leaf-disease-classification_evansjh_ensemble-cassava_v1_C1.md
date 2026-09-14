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
from pathlib import Path
import io
import glob
import struct

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

DATA_ROOT = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_CSV = f"{DATA_ROOT}/train.csv"
SAMPLE_CSV = f"{DATA_ROOT}/sample_submission.csv"
TRAIN_TFRECORD_DIR = f"{DATA_ROOT}/train_tfrecords"
TEST_TFRECORD_DIR = f"{DATA_ROOT}/test_tfrecords"
SUBMISSION_PATH = "/kaggle/working/submission.csv"

assert os.path.exists(TRAIN_CSV), f"Missing {TRAIN_CSV}"
assert os.path.exists(SAMPLE_CSV), f"Missing {SAMPLE_CSV}"

train_df = pd.read_csv(TRAIN_CSV)
sample_df = pd.read_csv(SAMPLE_CSV)
if list(sample_df.columns) != ["image_id", "label"]:
    raise ValueError(
        f"Unexpected sample submission columns: {sample_df.columns.tolist()}"
    )

print(train_df.head())
print(sample_df.head())


## === cell 1
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision import transforms

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.benchmark = True

try:
    cpu_cnt = os.cpu_count() or 2
    torch.set_num_threads(max(1, min(8, cpu_cnt)))
except Exception:
    pass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)
print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)


## === cell 2
TRAIN_IMG_DIR = f"{DATA_ROOT}/train_images"
TEST_IMG_DIR = f"{DATA_ROOT}/test_images"
assert os.path.isdir(TRAIN_IMG_DIR), f"Missing {TRAIN_IMG_DIR}"
assert os.path.isdir(TEST_IMG_DIR), f"Missing {TEST_IMG_DIR}"

num_classes = 5

img_size = 384
train_tfms = transforms.Compose(
    [
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.IMAGENET),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)

valid_tfms = transforms.Compose(
    [
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)


## === cell 3
from sklearn.model_selection import StratifiedShuffleSplit

FORCE_IMAGE_FOLDERS = True

splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=SEED)
train_idx, val_idx = next(splitter.split(train_df["image_id"], train_df["label"]))

tr_df = train_df.iloc[train_idx].reset_index(drop=True)
va_df = train_df.iloc[val_idx].reset_index(drop=True)

import multiprocessing as mp

batch_size = 32 if device.type == "cuda" else 16
cpu_cnt = os.cpu_count() or 2

if device.type == "cuda":
    num_workers = min(4, max(2, cpu_cnt // 4))
else:
    num_workers = min(8, max(2, cpu_cnt // 2))

prefetch_factor = 2 if num_workers > 0 else None


def _seed_worker(worker_id: int):
    seed = SEED + worker_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


g = torch.Generator()
g.manual_seed(SEED)

pin_mem = device.type == "cuda"
pin_mem_dev = "cuda" if pin_mem else ""

train_name_to_rec = None
test_name_to_rec = None

print(
    "TFRecords present:",
    (len(glob.glob(os.path.join(TRAIN_TFRECORD_DIR, "*.tfrec"))) > 0)
    and (len(glob.glob(os.path.join(TEST_TFRECORD_DIR, "*.tfrec"))) > 0),
    "| Using image folders due to FORCE_IMAGE_FOLDERS:",
    FORCE_IMAGE_FOLDERS,
)


## === cell 4
from PIL import Image


class CassavaTrainDataset(Dataset):
    def __init__(self, df, tfrecord_offset_index=None, transform=None, img_dir=None):
        df = df.reset_index(drop=True)
        self.image_ids = df["image_id"].to_numpy()
        self.labels = df["label"].to_numpy(np.int64)
        self.tfrecord_offset_index = tfrecord_offset_index
        self.transform = transform
        self.img_dir = img_dir

        self._name_bytes = None
        self._rec = None
        self._acc = None

    def __len__(self):
        return self.image_ids.shape[0]

    def __getitem__(self, idx):
        label = int(self.labels[idx])

        img_path = os.path.join(self.img_dir, str(self.image_ids[idx]))
        img = Image.open(img_path)
        if img.mode != "RGB":
            img = img.convert("RGB")

        if self.transform is not None:
            img = self.transform(img)
        return img, label


class CassavaTestDataset(Dataset):
    def __init__(
        self, image_ids, tfrecord_offset_index=None, transform=None, img_dir=None
    ):
        self.image_ids = np.asarray(list(image_ids))
        self.tfrecord_offset_index = tfrecord_offset_index
        self.transform = transform
        self.img_dir = img_dir

        self._name_bytes = None
        self._rec = None
        self._acc = None

    def __len__(self):
        return self.image_ids.shape[0]

    def __getitem__(self, idx):
        image_id = str(self.image_ids[idx])

        img_path = os.path.join(self.img_dir, image_id)
        img = Image.open(img_path)
        if img.mode != "RGB":
            img = img.convert("RGB")

        if self.transform is not None:
            img = self.transform(img)
        return img, image_id


train_ds = CassavaTrainDataset(
    tr_df, train_name_to_rec, transform=train_tfms, img_dir=TRAIN_IMG_DIR
)
val_ds = CassavaTrainDataset(
    va_df, train_name_to_rec, transform=valid_tfms, img_dir=TRAIN_IMG_DIR
)

mp_ctx = "fork" if hasattr(os, "fork") else None

train_loader = DataLoader(
    train_ds,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=pin_mem,
    pin_memory_device=pin_mem_dev,
    persistent_workers=(num_workers > 0),
    prefetch_factor=prefetch_factor if num_workers > 0 else None,
    worker_init_fn=_seed_worker if num_workers > 0 else None,
    generator=g,
    multiprocessing_context=mp_ctx if num_workers > 0 else None,
)
val_loader = DataLoader(
    val_ds,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=pin_mem,
    pin_memory_device=pin_mem_dev,
    persistent_workers=(num_workers > 0),
    prefetch_factor=prefetch_factor if num_workers > 0 else None,
    worker_init_fn=_seed_worker if num_workers > 0 else None,
    multiprocessing_context=mp_ctx if num_workers > 0 else None,
)

print("train/val sizes:", len(train_ds), len(val_ds))
print("batch_size:", batch_size, "num_workers:", num_workers)


## === cell 5
model = torchvision.models.efficientnet_b0(
    weights=torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1
)
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, num_classes)
model = model.to(device)

ENABLE_TORCH_COMPILE = False
if ENABLE_TORCH_COMPILE and hasattr(torch, "compile"):
    try:
        model = torch.compile(model)
    except Exception:
        pass

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))




## === cell 6
@torch.inference_mode()
def evaluate(model, loader):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_n = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total_n += bs
    return total_loss / total_n, total_correct / total_n


def train_one_epoch(model, loader):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_n = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            logits = model(x)
            loss = criterion(logits, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total_n += bs

    return total_loss / total_n, total_correct / total_n




## === cell 7
epochs = 3

best_val_acc = -1.0
best_buf = None

for epoch in range(1, epochs + 1):
    tr_loss, tr_acc = train_one_epoch(model, train_loader)
    va_loss, va_acc = evaluate(model, val_loader)
    print(
        f"epoch {epoch}/{epochs} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | val loss {va_loss:.4f} acc {va_acc:.4f}"
    )

    if va_acc > best_val_acc:
        best_val_acc = va_acc
        buf = io.BytesIO()
        cpu_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        torch.save(cpu_state, buf)
        best_buf = buf.getvalue()

print("best_val_acc:", best_val_acc)

if best_buf is not None:
    model.load_state_dict(torch.load(io.BytesIO(best_buf), map_location=device))


## === cell 8
test_ids = sample_df["image_id"].values
test_ds = CassavaTestDataset(
    test_ids,
    test_name_to_rec,
    transform=valid_tfms,
    img_dir=TEST_IMG_DIR,
)

test_loader = DataLoader(
    test_ds,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=pin_mem,
    pin_memory_device=pin_mem_dev,
    persistent_workers=(num_workers > 0),
    prefetch_factor=prefetch_factor if num_workers > 0 else None,
    worker_init_fn=_seed_worker if num_workers > 0 else None,
    multiprocessing_context=mp_ctx if num_workers > 0 else None,
)

model.eval()
all_preds = np.empty(len(test_ds), dtype=np.int64)

with torch.inference_mode():
    offset = 0
    for x, image_ids in test_loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy().astype(np.int64)
        bs = preds.shape[0]
        all_preds[offset : offset + bs] = preds
        offset += bs

all_ids = list(test_ids)
sub_df = pd.DataFrame({"image_id": all_ids, "label": all_preds.astype(int)})
assert sub_df.shape[0] == sample_df.shape[0]
assert list(sub_df.columns) == ["image_id", "label"]
assert (sub_df["image_id"].values == sample_df["image_id"].values).all()

sub_df.to_csv(SUBMISSION_PATH, index=False)
print("Wrote:", SUBMISSION_PATH)
print(sub_df.head())


## === cell 9
print(sub_df["label"].value_counts().sort_index())
print("submission rows:", len(sub_df))
print("unique ids:", sub_df["image_id"].nunique())
print("submission path exists:", os.path.exists(SUBMISSION_PATH))
print(
    "submission file size:",
    os.path.getsize(SUBMISSION_PATH) if os.path.exists(SUBMISSION_PATH) else None,
)
