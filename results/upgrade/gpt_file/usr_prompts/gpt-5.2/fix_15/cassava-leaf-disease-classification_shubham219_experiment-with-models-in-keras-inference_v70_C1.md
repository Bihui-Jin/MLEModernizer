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

3.9

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
import glob
import numpy as np
import pandas as pd

SEED = 42
DEBUG = False

np.random.seed(SEED)

DATA_ROOT_CANDIDATES = [
    "/kaggle/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
    "/kaggle/input/cassava-leaf-disease-classification",
    "/kaggle/data/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
    "/kaggle/data/cassava-leaf-disease-classification",
]


def _is_valid_root(root: str) -> bool:
    return (
        os.path.isdir(root)
        and os.path.isdir(os.path.join(root, "test_images"))
        and os.path.isfile(os.path.join(root, "sample_submission.csv"))
    )


DATA_ROOT = None
for p in DATA_ROOT_CANDIDATES:
    if _is_valid_root(p):
        DATA_ROOT = p
        break

if DATA_ROOT is None:
    sample_candidates = [
        "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv",
        "/kaggle/input/sample_submission.csv",
        "/kaggle/data/cassava-leaf-disease-classification/sample_submission.csv",
        "/kaggle/data/sample_submission.csv",
    ]
    found_sample = next((p for p in sample_candidates if os.path.isfile(p)), None)
    if found_sample is not None:
        base = os.path.dirname(found_sample)
        possible_roots = [base] + [
            os.path.join(base, d)
            for d in os.listdir(base)
            if os.path.isdir(os.path.join(base, d))
        ]
        for r in possible_roots:
            if _is_valid_root(r):
                DATA_ROOT = r
                break

if DATA_ROOT is None:
    raise FileNotFoundError(
        "Could not locate a valid cassava-leaf-disease-classification dataset root containing "
        "'test_images/' and 'sample_submission.csv' under expected Kaggle locations."
    )

TEST_IMG_DIR = os.path.join(DATA_ROOT, "test_images")
SAMPLE_SUB_PATH = os.path.join(DATA_ROOT, "sample_submission.csv")
TRAIN_CSV_PATH = os.path.join(DATA_ROOT, "train.csv")
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train_images")

if not os.path.isdir(TEST_IMG_DIR):
    raise FileNotFoundError(f"Expected test_images directory at: {TEST_IMG_DIR}")
if not os.path.isfile(SAMPLE_SUB_PATH):
    raise FileNotFoundError(f"Expected sample_submission.csv at: {SAMPLE_SUB_PATH}")
if not os.path.isfile(TRAIN_CSV_PATH):
    raise FileNotFoundError(f"Expected train.csv at: {TRAIN_CSV_PATH}")
if not os.path.isdir(TRAIN_IMG_DIR):
    raise FileNotFoundError(f"Expected train_images directory at: {TRAIN_IMG_DIR}")

print("DATA_ROOT:", DATA_ROOT)
print("TRAIN_CSV_PATH:", TRAIN_CSV_PATH)
print("TRAIN_IMG_DIR:", TRAIN_IMG_DIR)
print("TEST_IMG_DIR:", TEST_IMG_DIR)
print("SAMPLE_SUB_PATH:", SAMPLE_SUB_PATH)



## === cell 1
import random
import torch
import torch.nn as nn
from torchvision import models

random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

torch.set_num_threads(max(1, min(8, os.cpu_count() or 2)))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Torch device:", device)

my_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
in_features = my_model.classifier[1].in_features
my_model.classifier[1] = nn.Linear(in_features, 5)
my_model = my_model.to(device)

if device.type == "cuda":
    my_model = my_model.to(memory_format=torch.channels_last)

if hasattr(torch, "compile"):
    try:
        my_model = torch.compile(my_model, mode="reduce-overhead")
        print("Using torch.compile for faster execution.")
    except Exception as e:
        print("torch.compile unavailable/failed; using eager. Reason:", repr(e))

_MEAN = (
    torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1).to(device)
)
_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1).to(device)

with torch.inference_mode():
    x0 = torch.zeros(1, 3, 224, 224, device=device)
    if device.type == "cuda":
        x0 = x0.contiguous(memory_format=torch.channels_last)
    out = my_model(x0)
print("Model ready. Output dim:", out.shape[-1])



## === cell 2
train_df = pd.read_csv(TRAIN_CSV_PATH)
if not {"image_id", "label"}.issubset(train_df.columns):
    raise ValueError("train.csv must contain image_id and label columns.")

train_df = train_df.copy()
train_df["image_id"] = train_df["image_id"].astype(str)
train_df["label"] = train_df["label"].astype(int)
train_df["path"] = (TRAIN_IMG_DIR.rstrip("/") + "/" + train_df["image_id"]).astype(str)

idx = np.arange(len(train_df))
rng = np.random.RandomState(SEED)
rng.shuffle(idx)
split = int(0.9 * len(idx))
tr_idx, va_idx = idx[:split], idx[split:]
tr_df = train_df.iloc[tr_idx].reset_index(drop=True)
va_df = train_df.iloc[va_idx].reset_index(drop=True)

print("Train size:", len(tr_df), "Val size:", len(va_df))

from torchvision.io import read_image, ImageReadMode
from torchvision.transforms.functional import InterpolationMode
import torchvision.transforms.functional as F


@torch.jit.script
def _preprocess_script(
    img_uint8_chw: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
) -> torch.Tensor:
    x = img_uint8_chw.to(dtype=torch.float32).div(255.0)
    x = F.resize(
        x,
        size=[256, 256],
        interpolation=InterpolationMode.BILINEAR,
        antialias=True,
    )
    x = F.center_crop(x, output_size=[224, 224])
    x = (x - mean) / std
    return x


_MEAN_CPU = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
_STD_CPU = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


def preprocess_tensor_uint8_chw(img_uint8_chw: torch.Tensor) -> torch.Tensor:
    return _preprocess_script(img_uint8_chw, _MEAN_CPU, _STD_CPU)


class CassavaDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.paths = df["path"].to_numpy(dtype=str)
        self.labels = df["label"].to_numpy(dtype=np.int64)

    def __len__(self):
        return self.paths.shape[0]

    def __getitem__(self, i):
        p = self.paths[i]
        y = int(self.labels[i])
        img_t = read_image(p, mode=ImageReadMode.RGB)  # uint8 CHW
        x = preprocess_tensor_uint8_chw(img_t)
        return x, y


train_ds = CassavaDataset(tr_df)
val_ds = CassavaDataset(va_df)

for p in my_model.parameters():
    p.requires_grad = False
for p in my_model.classifier.parameters():
    p.requires_grad = True

batch_size = 64
pin = torch.cuda.is_available()

cpu_cnt = os.cpu_count() or 2
num_workers = max(2, min(8, cpu_cnt))
g = torch.Generator()
g.manual_seed(SEED)

_prefetch = 4 if num_workers > 0 else None

train_loader = torch.utils.data.DataLoader(
    train_ds,
    batch_size=batch_size,
    shuffle=True,
    generator=g,
    num_workers=num_workers,
    pin_memory=pin,
    pin_memory_device="cuda" if pin else "",
    persistent_workers=(num_workers > 0),
    prefetch_factor=_prefetch,
    drop_last=True,
)
val_loader = torch.utils.data.DataLoader(
    val_ds,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=pin,
    pin_memory_device="cuda" if pin else "",
    persistent_workers=(num_workers > 0),
    prefetch_factor=_prefetch,
)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(my_model.classifier.parameters(), lr=3e-4)


def eval_acc(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.inference_mode():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            if device.type == "cuda":
                xb = xb.contiguous(memory_format=torch.channels_last)
            logits = model(xb)
            pred = torch.argmax(logits, dim=1)
            correct += (pred == yb).sum().item()
            total += yb.numel()
    return correct / max(total, 1)


if device.type == "cuda":
    with torch.inference_mode():
        xw = torch.zeros(batch_size, 3, 224, 224, device=device).contiguous(
            memory_format=torch.channels_last
        )
        _ = my_model(xw)
    torch.cuda.synchronize()

my_model.train()
epochs = 2
for ep in range(1, epochs + 1):
    my_model.train()
    running_loss = 0.0
    seen = 0
    torch.set_grad_enabled(True)
    for xb, yb in train_loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        if device.type == "cuda":
            xb = xb.contiguous(memory_format=torch.channels_last)

        optimizer.zero_grad(set_to_none=True)
        logits = my_model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        bs = xb.size(0)
        running_loss += float(loss.detach()) * bs
        seen += bs

    torch.set_grad_enabled(False)
    tr_loss = running_loss / max(seen, 1)
    va_acc = eval_acc(my_model, val_loader)
    print(f"Epoch {ep}/{epochs} - train_loss: {tr_loss:.4f} - val_acc: {va_acc:.4f}")

my_model.eval()



## === cell 3
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
if "image_id" not in sample_sub.columns:
    raise ValueError("sample_submission.csv does not contain 'image_id' column")

df_test = pd.DataFrame({"image_id": sample_sub["image_id"].astype(str).values})
df_test["path"] = (TEST_IMG_DIR.rstrip("/") + "/" + df_test["image_id"]).astype(str)

print("Test rows:", len(df_test))
print(df_test.head())



## === cell 4
from torchvision.io import read_image, ImageReadMode


class CassavaTestDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.paths = df["path"].to_numpy(dtype=str)

    def __len__(self):
        return self.paths.shape[0]

    def __getitem__(self, i):
        p = self.paths[i]
        img_t = read_image(p, mode=ImageReadMode.RGB)  # uint8 CHW
        x = preprocess_tensor_uint8_chw(img_t)
        return x


batch_size = 64
test_ds = CassavaTestDataset(df_test.reset_index(drop=True))
pin = torch.cuda.is_available()

cpu_cnt = os.cpu_count() or 2
num_workers = max(2, min(8, cpu_cnt))
_prefetch = 4 if num_workers > 0 else None

test_loader = torch.utils.data.DataLoader(
    test_ds,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=pin,
    pin_memory_device="cuda" if pin else "",
    persistent_workers=(num_workers > 0),
    prefetch_factor=_prefetch,
)

pred_labels = np.empty(len(df_test), dtype=np.int64)
offset = 0
with torch.inference_mode():
    for xb in test_loader:
        xb = xb.to(device, non_blocking=True)
        if device.type == "cuda":
            xb = xb.contiguous(memory_format=torch.channels_last)
        logits = my_model(xb)
        batch_pred = torch.argmax(logits, dim=1).detach().cpu().numpy().astype(np.int64)
        pred_labels[offset : offset + len(batch_pred)] = batch_pred
        offset += len(batch_pred)

final_csv = df_test[["image_id"]].copy()
final_csv["label"] = pred_labels

if len(final_csv) != len(sample_sub):
    raise RuntimeError("Submission row count mismatch vs sample_submission.")
if final_csv["label"].isna().any():
    raise RuntimeError("NaNs found in predicted labels.")
if not np.issubdtype(final_csv["label"].dtype, np.integer):
    final_csv["label"] = final_csv["label"].astype(int)

final_csv.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", final_csv.shape)
print(final_csv.head())



## === cell 5
final_csv.head()
