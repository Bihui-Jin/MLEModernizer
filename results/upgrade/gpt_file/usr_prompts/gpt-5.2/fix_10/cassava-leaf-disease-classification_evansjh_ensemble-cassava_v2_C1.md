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
import json
import time
import random
import numpy as np
import pandas as pd

DATA_DIR = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
SAMPLE_SUB = os.path.join(DATA_DIR, "sample_submission.csv")
TRAIN_IMAGE_DIR = os.path.join(DATA_DIR, "train_images")
TEST_IMAGE_DIR = os.path.join(DATA_DIR, "test_images")
MAP_JSON = os.path.join(DATA_DIR, "label_num_to_disease_map.json")

assert os.path.exists(TRAIN_CSV), f"Missing: {TRAIN_CSV}"
assert os.path.exists(SAMPLE_SUB), f"Missing: {SAMPLE_SUB}"
assert os.path.exists(TRAIN_IMAGE_DIR), f"Missing: {TRAIN_IMAGE_DIR}"
assert os.path.exists(TEST_IMAGE_DIR), f"Missing: {TEST_IMAGE_DIR}"

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

train_df = pd.read_csv(TRAIN_CSV)
sample_df = pd.read_csv(SAMPLE_SUB)

for col in ["image_id", "label"]:
    assert col in train_df.columns, f"train.csv missing column: {col}"
for col in ["image_id", "label"]:
    assert col in sample_df.columns, f"sample_submission.csv missing column: {col}"

train_df["label"] = train_df["label"].astype(int)

if os.path.exists(MAP_JSON):
    with open(MAP_JSON, "r") as f:
        label_map = json.load(f)
else:
    label_map = None

print("train_df:", train_df.shape, "sample_df:", sample_df.shape)
print("label counts:", train_df["label"].value_counts().to_dict())



## === cell 1
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

try:
    import torchvision
    from torchvision.io import ImageReadMode
    from torchvision.io import read_file, decode_jpeg

    try:
        torchvision.set_image_backend("accimage")
        _IMG_BACKEND = torchvision.get_image_backend()
    except Exception:
        _IMG_BACKEND = torchvision.get_image_backend()

    try:
        from torchvision.transforms import v2 as T  # torchvision>=0.15

        _HAS_V2 = True
    except Exception:
        import torchvision.transforms as T

        _HAS_V2 = False
except Exception as e:
    raise RuntimeError(
        "torchvision is required for this minimal image baseline. "
        "It is typically available on Kaggle. Import failed with: " + str(e)
    )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device, "| torchvision image backend:", _IMG_BACKEND)

_cpu = os.cpu_count() or 2
torch.set_num_threads(min(8, _cpu))
torch.set_num_interop_threads(1)

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
if device.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def _seed_worker(worker_id: int):
    seed = SEED + worker_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _read_rgb_image(path: str) -> torch.Tensor:
    data = read_file(path)
    img = decode_jpeg(data, mode=ImageReadMode.RGB)  # uint8, CHW
    return img


class CassavaTrainDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        df = df.reset_index(drop=True)
        self.transform = transform
        self.image_paths = [os.path.join(img_dir, p) for p in df["image_id"].tolist()]
        self.labels = df["label"].astype(int).tolist()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = _read_rgb_image(self.image_paths[idx])
        if self.transform is not None:
            img = self.transform(img)
        y = self.labels[idx]
        return img, y


class CassavaValDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        df = df.reset_index(drop=True)
        self.transform = transform
        self.image_paths = [os.path.join(img_dir, p) for p in df["image_id"].tolist()]
        self.labels = df["label"].astype(int).tolist()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = _read_rgb_image(self.image_paths[idx])
        if self.transform is not None:
            img = self.transform(img)
        return img, self.labels[idx]


class CassavaTestDataset(Dataset):
    def __init__(self, image_ids, img_dir, transform=None):
        self.image_ids = list(image_ids)
        self.image_paths = [os.path.join(img_dir, p) for p in self.image_ids]
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img = _read_rgb_image(self.image_paths[idx])
        if self.transform is not None:
            img = self.transform(img)
        return self.image_ids[idx], img


IMG_SIZE = 224


class ToFloat01(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(dtype=torch.float32).div_(255.0)


if _HAS_V2:
    train_tfms = T.Compose(
        [
            T.Resize((IMG_SIZE, IMG_SIZE), antialias=True),
            T.RandomHorizontalFlip(p=0.5),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    test_tfms = T.Compose(
        [
            T.Resize((IMG_SIZE, IMG_SIZE), antialias=True),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
else:
    train_tfms = T.Compose(
        [
            T.Resize((IMG_SIZE, IMG_SIZE), antialias=True),
            T.RandomHorizontalFlip(p=0.5),
            ToFloat01(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    test_tfms = T.Compose(
        [
            T.Resize((IMG_SIZE, IMG_SIZE), antialias=True),
            ToFloat01(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

val_frac = 0.1
parts = []
for lbl, grp in train_df.groupby("label"):
    grp = grp.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    n_val = max(1, int(len(grp) * val_frac))
    parts.append((grp.iloc[n_val:], grp.iloc[:n_val]))
train_parts = [p[0] for p in parts]
val_parts = [p[1] for p in parts]
tr_df = (
    pd.concat(train_parts).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
)
va_df = pd.concat(val_parts).sample(frac=1.0, random_state=SEED).reset_index(drop=True)

print("split:", tr_df.shape, va_df.shape)

train_ds = CassavaTrainDataset(tr_df, TRAIN_IMAGE_DIR, transform=train_tfms)
val_ds = CassavaValDataset(va_df, TRAIN_IMAGE_DIR, transform=test_tfms)

BATCH_SIZE = 64 if device.type == "cuda" else 32

NUM_WORKERS = min(16, max(2, _cpu - 1))
PIN_MEMORY = device.type == "cuda"
PERSISTENT = NUM_WORKERS > 0
PREFETCH_FACTOR = 4 if NUM_WORKERS > 0 else None

g = torch.Generator()
g.manual_seed(SEED)

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    generator=g,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    persistent_workers=PERSISTENT,
    prefetch_factor=PREFETCH_FACTOR,
    worker_init_fn=_seed_worker,
)
val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    persistent_workers=PERSISTENT,
    prefetch_factor=PREFETCH_FACTOR,
    worker_init_fn=_seed_worker,
)

NUM_CLASSES = int(train_df["label"].nunique())
assert NUM_CLASSES == 5, f"Expected 5 classes, got {NUM_CLASSES}"

try:
    weights = torchvision.models.ResNet18_Weights.DEFAULT
    model = torchvision.models.resnet18(weights=weights)
except Exception:
    model = torchvision.models.resnet18(weights=None)

model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

model = model.to(device)
if device.type == "cuda":
    model = model.to(memory_format=torch.channels_last)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

_COMPILED = False
if device.type == "cuda":
    try:
        model = torch.compile(model)
        _COMPILED = True
    except Exception:
        _COMPILED = False
print("torch.compile:", _COMPILED)


class CUDAPrefetcher:
    def __init__(self, loader, device, channels_last: bool):
        self.loader = loader
        self.device = device
        self.channels_last = channels_last and (device.type == "cuda")
        self.stream = torch.cuda.Stream() if device.type == "cuda" else None

    def __iter__(self):
        if self.device.type != "cuda":
            for batch in self.loader:
                yield batch
            return

        it = iter(self.loader)

        def _preload():
            nonlocal next_batch
            try:
                batch = next(it)
            except StopIteration:
                next_batch = None
                return
            with torch.cuda.stream(self.stream):
                if (
                    isinstance(batch, (list, tuple))
                    and len(batch) == 2
                    and torch.is_tensor(batch[0])
                ):
                    x, y = batch
                    x = x.to(self.device, non_blocking=True)
                    if self.channels_last:
                        x = x.contiguous(memory_format=torch.channels_last)
                    y = y.to(self.device, non_blocking=True)
                    next_batch = (x, y)
                else:
                    img_ids, x = batch
                    x = x.to(self.device, non_blocking=True)
                    if self.channels_last:
                        x = x.contiguous(memory_format=torch.channels_last)
                    next_batch = (img_ids, x)

        next_batch = None
        _preload()
        while next_batch is not None:
            torch.cuda.current_stream().wait_stream(self.stream)
            batch = next_batch
            _preload()
            yield batch


def eval_acc(model, loader):
    model.eval()
    correct = 0
    total = 0
    loader_it = CUDAPrefetcher(loader, device, channels_last=True)
    with torch.inference_mode():
        for x, y in loader_it:
            if device.type != "cuda":
                x = x.to(device, non_blocking=True)
                if device.type == "cuda":
                    x = x.contiguous(memory_format=torch.channels_last)
                y = y.to(device, non_blocking=True)
            logits = model(x)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()
    return correct / max(1, total)




## === cell 2
EPOCHS = 2 if device.type == "cuda" else 1

start = time.time()
best_val = -1.0

for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0
    n = 0

    train_it = CUDAPrefetcher(train_loader, device, channels_last=True)
    for x, y in train_it:
        if device.type != "cuda":
            x = x.to(device, non_blocking=True)
            if device.type == "cuda":
                x = x.contiguous(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        bs = y.size(0)
        running_loss += float(loss.item()) * bs
        n += bs

    train_loss = running_loss / max(1, n)
    val_acc = eval_acc(model, val_loader)
    best_val = max(best_val, val_acc)

    elapsed = time.time() - start
    print(
        f"epoch {epoch}/{EPOCHS} - train_loss: {train_loss:.4f} - val_acc: {val_acc:.4f} - elapsed: {elapsed:.1f}s"
    )

print("best_val_acc:", best_val)



## === cell 3
test_image_ids = sample_df["image_id"].tolist()
test_ds = CassavaTestDataset(test_image_ids, TEST_IMAGE_DIR, transform=test_tfms)
test_loader = DataLoader(
    test_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    persistent_workers=PERSISTENT,
    prefetch_factor=PREFETCH_FACTOR,
    worker_init_fn=_seed_worker,
)

model.eval()
pred_labels = []
pred_ids = []

test_it = CUDAPrefetcher(test_loader, device, channels_last=True)
with torch.inference_mode():
    for img_ids, x in test_it:
        if device.type != "cuda":
            x = x.to(device, non_blocking=True)
            if device.type == "cuda":
                x = x.contiguous(memory_format=torch.channels_last)
        logits = model(x)
        preds = logits.argmax(dim=1).detach().cpu().numpy().astype(int).tolist()
        pred_labels.extend(preds)
        pred_ids.extend(list(img_ids))

assert (
    pred_ids == test_image_ids
), "Test predictions are not aligned with sample_submission order."
assert len(pred_labels) == len(sample_df)

submission_df = pd.DataFrame({"image_id": pred_ids, "label": pred_labels})

submission_path = "/kaggle/working/submission.csv"
submission_df.to_csv(submission_path, index=False)

assert list(submission_df.columns) == ["image_id", "label"]
assert len(submission_df) == len(sample_df)

print(f"Submission file saved at: {submission_path}")
print(submission_df.head(10))
