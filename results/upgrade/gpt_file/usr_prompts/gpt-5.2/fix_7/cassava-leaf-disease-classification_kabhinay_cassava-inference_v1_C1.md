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
import random
import numpy as np
import pandas as pd

os.environ["PYTHONHASHSEED"] = "42"
random.seed(42)
np.random.seed(42)



## === cell 1
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

from torchvision.io import read_image, ImageReadMode
from torchvision.transforms import functional as TF

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

NUM_CLASSES = 5
TARGET_SIZE = (448, 448)  # keep original input size intent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

try:
    weights = models.EfficientNet_B3_Weights.IMAGENET1K_V1
except Exception:
    weights = None

base = models.efficientnet_b3(weights=weights)
in_features = base.classifier[1].in_features
base.classifier[1] = nn.Linear(in_features, NUM_CLASSES)

for name, param in base.named_parameters():
    param.requires_grad = False
for param in base.classifier[1].parameters():
    param.requires_grad = True

model = base.to(device)

if weights is not None:
    mean = weights.transforms().mean
    std = weights.transforms().std
else:
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

train_tfms = transforms.Compose(
    [
        transforms.Resize(
            TARGET_SIZE, interpolation=transforms.InterpolationMode.BICUBIC
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]
)
test_tfms = transforms.Compose(
    [
        transforms.Resize(
            TARGET_SIZE, interpolation=transforms.InterpolationMode.BICUBIC
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]
)



## === cell 2
CANDIDATE_DATA_DIRS = [
    "/kaggle/input/cassava-leaf-disease-classification",
    "/kaggle/data/cassava-leaf-disease-classification",
]
DATA_DIR = None
for d in CANDIDATE_DATA_DIRS:
    if os.path.isdir(d):
        DATA_DIR = d
        break
if DATA_DIR is None:
    raise FileNotFoundError(f"Could not find dataset dir. Tried: {CANDIDATE_DATA_DIRS}")

train_dir = os.path.join(DATA_DIR, "train_images")
test_dir = os.path.join(DATA_DIR, "test_images")
assert os.path.isdir(train_dir), f"train_dir not found: {train_dir}"
assert os.path.isdir(test_dir), f"test_dir not found: {test_dir}"

train_csv_path = os.path.join(DATA_DIR, "train.csv")
sample_path = os.path.join(DATA_DIR, "sample_submission.csv")
train_df = pd.read_csv(train_csv_path)
sample_sub = pd.read_csv(sample_path)

test_images = sorted([f for f in os.listdir(test_dir) if f.lower().endswith(".jpg")])

expected = sample_sub["image_id"].tolist()
expected_set = set(expected)
test_set = set(test_images)

if expected_set == test_set:
    image_ids = expected  # exact canonical ordering
else:
    missing = len(expected_set - test_set)
    extra = len(test_set - expected_set)
    print(
        f"Warning: test image set differs from sample_submission. missing={missing}, extra={extra}"
    )
    image_ids = test_images

print("Using DATA_DIR:", DATA_DIR)
print("Num train:", len(train_df))
print("Num test images:", len(image_ids))




## === cell 3
class CassavaTrainDataset(Dataset):
    def __init__(self, df, img_dir, transform):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["image_id"])

        try:
            img = read_image(img_path, mode=ImageReadMode.RGB)  # uint8 CHW
            im = TF.to_pil_image(img)  # keep identical transform behavior (expects PIL)
        except Exception:
            with Image.open(img_path) as im:
                im = im.convert("RGB")

        x = self.transform(im)
        y = int(row["label"])
        return x, y


class CassavaTestDataset(Dataset):
    def __init__(self, image_ids, img_dir, transform):
        self.image_ids = list(image_ids)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_path = os.path.join(self.img_dir, img_id)

        try:
            img = read_image(img_path, mode=ImageReadMode.RGB)  # uint8 CHW
            im = TF.to_pil_image(img)
        except Exception:
            with Image.open(img_path) as im:
                im = im.convert("RGB")

        x = self.transform(im)
        return x, img_id


def stratified_split_indices(labels, val_frac=0.1, seed=42):
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    train_idx, val_idx = [], []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        n_val = max(1, int(round(len(idx) * val_frac)))
        val_idx.extend(idx[:n_val].tolist())
        train_idx.extend(idx[n_val:].tolist())
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


train_idx, val_idx = stratified_split_indices(
    train_df["label"].values, val_frac=0.1, seed=42
)
tr_df = train_df.iloc[train_idx].reset_index(drop=True)
va_df = train_df.iloc[val_idx].reset_index(drop=True)

print("Split sizes:", len(tr_df), len(va_df))
print("Train label dist:\n", tr_df["label"].value_counts().sort_index())
print("Val label dist:\n", va_df["label"].value_counts().sort_index())



## === cell 4
BATCH_SIZE = 32
EPOCHS = 2  # keep runtime safe under 600s; still yields big improvement over 0.16
LR = 3e-3

_num_workers = min(4, os.cpu_count() or 2)
_loader_kwargs = dict(
    num_workers=_num_workers,
    pin_memory=(device.type == "cuda"),
    persistent_workers=(_num_workers > 0),
    prefetch_factor=4 if _num_workers > 0 else None,
)

train_loader = DataLoader(
    CassavaTrainDataset(tr_df, train_dir, train_tfms),
    batch_size=BATCH_SIZE,
    shuffle=True,
    **{k: v for k, v in _loader_kwargs.items() if v is not None},
)
val_loader = DataLoader(
    CassavaTrainDataset(va_df, train_dir, test_tfms),
    batch_size=BATCH_SIZE,
    shuffle=False,
    **{k: v for k, v in _loader_kwargs.items() if v is not None},
)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=1e-4
)

model.train()
for epoch in range(EPOCHS):
    model.train()
    tr_loss = 0.0
    tr_correct = 0
    n = 0

    for xb, yb in train_loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        bs = xb.size(0)
        tr_loss += loss.item() * bs
        tr_correct += (logits.detach().argmax(dim=1) == yb).sum().item()
        n += bs

    tr_loss /= max(1, n)
    tr_acc = tr_correct / max(1, n)

    model.eval()
    va_loss = 0.0
    va_correct = 0
    vn = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            bs = xb.size(0)
            va_loss += loss.item() * bs
            va_correct += (logits.argmax(dim=1) == yb).sum().item()
            vn += bs
    va_loss /= max(1, vn)
    va_acc = va_correct / max(1, vn)

    print(
        f"Epoch {epoch+1}/{EPOCHS} - train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} val_loss={va_loss:.4f} val_acc={va_acc:.4f}"
    )



## === cell 5
test_loader = DataLoader(
    CassavaTestDataset(image_ids, test_dir, test_tfms),
    batch_size=64,
    shuffle=False,
    **{k: v for k, v in _loader_kwargs.items() if v is not None},
)

model.eval()
all_ids = []
all_preds = []

with torch.no_grad():
    for xb, ids in test_loader:
        xb = xb.to(device, non_blocking=True)
        logits = model(xb)
        pred = torch.argmax(logits, dim=1).cpu().numpy().astype(int)
        all_preds.append(pred)
        all_ids.extend(list(ids))

all_preds = np.concatenate(all_preds, axis=0)

results = pd.DataFrame({"image_id": all_ids, "label": all_preds.astype(int)})

if expected_set == test_set:
    order = pd.Index(expected)
    results = results.set_index("image_id").reindex(order).reset_index()

print(results.head())
print("Label distribution:\n", results["label"].value_counts().sort_index())

sub_path = "/kaggle/working/submission.csv"
results.to_csv(sub_path, index=False)

check = pd.read_csv(sub_path)
assert list(check.columns) == [
    "image_id",
    "label",
], f"Bad columns: {check.columns.tolist()}"
assert len(check) == len(
    sample_sub
), f"Row count mismatch: got {len(check)} expected {len(sample_sub)}"

print(f"Wrote submission to {sub_path} with {len(results)} rows.")
