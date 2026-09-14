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

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

ROOT_DIR = "../input/cassava-leaf-disease-classification/"
print("ROOT_DIR exists:", os.path.exists(ROOT_DIR))
print("ROOT_DIR listing (first 20):", os.listdir(ROOT_DIR)[:20])



## === cell 1
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

print("PyTorch:", torch.__version__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(SEED)



## === cell 2
TRAIN_CSV = os.path.join(ROOT_DIR, "train.csv")
SAMPLE_SUB = os.path.join(ROOT_DIR, "sample_submission.csv")
TRAIN_DIR = os.path.join(ROOT_DIR, "train_images")
TEST_DIR = os.path.join(ROOT_DIR, "test_images")

train_df = pd.read_csv(TRAIN_CSV)
sample_sub = pd.read_csv(SAMPLE_SUB)

print("train_df:", train_df.shape, train_df.columns.tolist())
print("sample_sub:", sample_sub.shape, sample_sub.columns.tolist())
print(
    "TRAIN_DIR exists:",
    os.path.exists(TRAIN_DIR),
    "num_files:",
    len(os.listdir(TRAIN_DIR)),
)
print(
    "TEST_DIR exists:",
    os.path.exists(TEST_DIR),
    "num_files:",
    len(os.listdir(TEST_DIR)),
)

NUM_CLASSES = 5



## === cell 3
from torchvision.io import read_image
from torchvision.transforms import v2 as T

IMG_SIZE = 300

train_tfms = T.Compose(
    [
        T.Resize((IMG_SIZE, IMG_SIZE), antialias=True),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=10),
        T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
        T.ToDtype(
            torch.float32, scale=True
        ),  # equivalent to ToTensor() scaling to [0,1]
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)

valid_tfms = T.Compose(
    [
        T.Resize((IMG_SIZE, IMG_SIZE), antialias=True),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)


class CassavaDataset(Dataset):
    def __init__(self, df, img_dir, tfms, has_labels=True):
        df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.tfms = tfms
        self.has_labels = has_labels

        self.image_ids = df["image_id"].to_numpy()
        if has_labels:
            self.labels = df["label"].to_numpy(dtype=np.int64)
        else:
            self.labels = None

    def __len__(self):
        return self.image_ids.shape[0]

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        path = os.path.join(self.img_dir, img_id)

        x = read_image(path)  # RGB by default for jpeg; returns CHW uint8
        if x.shape[0] == 1:
            x = x.expand(3, -1, -1)
        elif x.shape[0] == 4:
            x = x[:3]  # drop alpha if present
        x = self.tfms(x)

        if self.has_labels:
            y = int(self.labels[idx])
            return x, y
        return x, img_id




## === cell 4
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, NUM_CLASSES)
model = model.to(device)
model = model.to(memory_format=torch.channels_last)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

idx = np.arange(len(train_df))
rng = np.random.RandomState(SEED)
rng.shuffle(idx)
split = int(0.9 * len(train_df))
tr_idx, va_idx = idx[:split], idx[split:]
tr_df = train_df.iloc[tr_idx].reset_index(drop=True)
va_df = train_df.iloc[va_idx].reset_index(drop=True)

train_ds = CassavaDataset(tr_df, TRAIN_DIR, train_tfms, has_labels=True)
valid_ds = CassavaDataset(va_df, TRAIN_DIR, valid_tfms, has_labels=True)

BATCH_SIZE = 32

_cpu = os.cpu_count() or 2
NUM_WORKERS = min(4, max(2, _cpu // 2))

common_dl_kwargs = dict(
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(NUM_WORKERS > 0),
    prefetch_factor=4 if NUM_WORKERS > 0 else None,
)

train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False, **common_dl_kwargs
)
valid_loader = DataLoader(
    valid_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, **common_dl_kwargs
)

print("Train/Valid sizes:", len(train_ds), len(valid_ds))
print("NUM_WORKERS:", NUM_WORKERS)




## === cell 5
def accuracy_from_logits(logits, y):
    preds = torch.argmax(logits, dim=1)
    return (preds == y).float().mean().item()


EPOCHS = 3  # keep same as original code for minimal runtime while improving backbone

for epoch in range(1, EPOCHS + 1):
    model.train()
    tr_loss = 0.0
    tr_acc = 0.0
    n_tr = 0

    for xb, yb in train_loader:
        xb = xb.to(device, non_blocking=True).contiguous(
            memory_format=torch.channels_last
        )
        yb = yb.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        bs = xb.size(0)
        tr_loss += loss.item() * bs
        tr_acc += accuracy_from_logits(logits.detach(), yb) * bs
        n_tr += bs

    model.eval()
    va_loss = 0.0
    va_acc = 0.0
    n_va = 0
    with torch.no_grad():
        for xb, yb in valid_loader:
            xb = xb.to(device, non_blocking=True).contiguous(
                memory_format=torch.channels_last
            )
            yb = yb.to(device, non_blocking=True)
            logits = model(xb)
            loss = criterion(logits, yb)

            bs = xb.size(0)
            va_loss += loss.item() * bs
            va_acc += accuracy_from_logits(logits, yb) * bs
            n_va += bs

    print(
        f"Epoch {epoch}/{EPOCHS} | "
        f"train loss {tr_loss/n_tr:.4f} acc {tr_acc/n_tr:.4f} | "
        f"valid loss {va_loss/n_va:.4f} acc {va_acc/n_va:.4f}"
    )



## === cell 6
test_images = sample_sub["image_id"].tolist()
test_df = pd.DataFrame({"image_id": test_images})

test_ds = CassavaDataset(test_df, TEST_DIR, valid_tfms, has_labels=False)
test_loader = DataLoader(
    test_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, **common_dl_kwargs
)

model.eval()
preds = []
ids = []
with torch.no_grad():
    for xb, batch_ids in test_loader:
        xb = xb.to(device, non_blocking=True).contiguous(
            memory_format=torch.channels_last
        )
        logits = model(xb)
        batch_preds = torch.argmax(logits, dim=1).cpu().numpy().astype(int).tolist()
        preds.extend(batch_preds)
        ids.extend(list(batch_ids))

print("Num test images:", len(test_images))
print("Num preds:", len(preds))



## === cell 7
sub = pd.DataFrame({"image_id": ids, "label": preds})

sub = sample_sub[["image_id"]].merge(sub, on="image_id", how="left")

assert (
    sub.shape[0] == sample_sub.shape[0]
), "Submission row count must match sample_submission."
assert list(sub.columns) == [
    "image_id",
    "label",
], "Submission columns must be exactly ['image_id','label']."
assert sub["image_id"].equals(
    sample_sub["image_id"]
), "image_id order must match sample_submission."
assert sub["label"].notnull().all(), "All test images must have predictions."
sub["label"] = sub["label"].astype(int)
assert sub["label"].between(0, 4).all(), "Labels must be integers in [0,4]."

print(sub.head())

out_path = "submission.csv"
sub.to_csv(out_path, index=False)
print("Wrote:", out_path, "size(bytes):", os.path.getsize(out_path))
print("Done.")
