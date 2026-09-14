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
from torchvision.transforms import functional as TF

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
try:
    torch.use_deterministic_algorithms(True)
except Exception:
    pass

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

_mean_t = torch.tensor(mean).view(3, 1, 1)
_std_t = torch.tensor(std).view(3, 1, 1)

train_colorjitter = transforms.ColorJitter(
    brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02
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
import glob
import tensorflow as tf

train_tfrecord_dir = os.path.join(DATA_DIR, "train_tfrecords")
test_tfrecord_dir = os.path.join(DATA_DIR, "test_tfrecords")
assert os.path.isdir(
    train_tfrecord_dir
), f"train_tfrecord_dir not found: {train_tfrecord_dir}"
assert os.path.isdir(
    test_tfrecord_dir
), f"test_tfrecord_dir not found: {test_tfrecord_dir}"

train_tfrec_files = sorted(glob.glob(os.path.join(train_tfrecord_dir, "*.tfrec")))
test_tfrec_files = sorted(glob.glob(os.path.join(test_tfrecord_dir, "*.tfrec")))
print(
    "Num train tfrecs:",
    len(train_tfrec_files),
    "Num test tfrecs:",
    len(test_tfrec_files),
)

_label_map = dict(
    zip(
        train_df["image_id"].astype(str).tolist(),
        train_df["label"].astype(int).tolist(),
    )
)

_TFREC_FEATURES = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
}


def _parse_tfrec_to_torch(example_proto):
    ex = tf.io.parse_single_example(example_proto, _TFREC_FEATURES)
    img_bytes = ex["image"]
    img_name = ex["image_name"]
    img = tf.io.decode_jpeg(img_bytes, channels=3)  # uint8 HWC
    return img, img_name




## === cell 4
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

_tr_set = set(tr_df["image_id"].astype(str).tolist())
_va_set = set(va_df["image_id"].astype(str).tolist())




## === cell 5
from PIL import Image

TARGET_H, TARGET_W = TARGET_SIZE

_keys = tf.constant(train_df["image_id"].astype(str).values)
_vals = tf.constant(train_df["label"].astype(np.int64).values)
_label_table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(_keys, _vals),
    default_value=tf.constant(-1, dtype=tf.int64),
)

_tr_keys = tf.constant(tr_df["image_id"].astype(str).values)
_va_keys = tf.constant(va_df["image_id"].astype(str).values)
_true_bool = tf.constant(True)
_false_bool = tf.constant(False)
_tr_table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(
        _tr_keys, tf.fill([tf.shape(_tr_keys)[0]], _true_bool)
    ),
    default_value=_false_bool,
)
_va_table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(
        _va_keys, tf.fill([tf.shape(_va_keys)[0]], _true_bool)
    ),
    default_value=_false_bool,
)

try:
    tf.random.set_seed(42)
except Exception:
    pass
try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass


class _TFRecordIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, tfrec_files, mode, id_set=None, return_id=False):
        super().__init__()
        self.tfrec_files = list(tfrec_files)
        self.mode = mode  # "train" | "val" | "test"
        self.id_set = (
            id_set  # kept for API compatibility; actual filtering is done via TF tables
        )
        self.return_id = return_id

    def __iter__(self):
        opts = tf.data.Options()
        opts.experimental_deterministic = True

        ds = tf.data.TFRecordDataset(
            self.tfrec_files,
            num_parallel_reads=tf.data.AUTOTUNE,
        ).with_options(opts)

        def _parse_and_resize(raw):
            img_u8, img_name = _parse_tfrec_to_torch(raw)  # img: uint8 HWC
            img_u8 = tf.image.resize(
                img_u8,
                [TARGET_H, TARGET_W],
                method=tf.image.ResizeMethod.BICUBIC,
                antialias=True,
            )
            img_u8 = tf.cast(tf.clip_by_value(img_u8, 0.0, 255.0), tf.uint8)
            return img_u8, img_name

        ds = ds.map(_parse_and_resize, num_parallel_calls=tf.data.AUTOTUNE)

        if self.mode == "train":
            ds = ds.filter(lambda img, name: _tr_table.lookup(name))
        elif self.mode == "val":
            ds = ds.filter(lambda img, name: _va_table.lookup(name))
        else:
            pass

        ds = ds.prefetch(tf.data.AUTOTUNE)

        for img_u8, img_name in ds:
            img_name = img_name.numpy().decode("utf-8")
            x = torch.from_numpy(img_u8.numpy()).permute(2, 0, 1).contiguous()
            x = x.float().div_(255.0)

            if self.mode == "train":
                if torch.rand((), dtype=torch.float32).item() < 0.5:
                    x = TF.hflip(x)
                x = train_colorjitter(x)

            x = TF.normalize(x, _mean_t, _std_t)

            if self.mode == "test":
                yield x, img_name
            else:
                y = int(_label_map[img_name])
                yield x, y


def _seed_worker(worker_id):
    seed = 42 + worker_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)




## === cell 6
BATCH_SIZE = 32
EPOCHS = 2  # keep same as provided
LR = 3e-3

_cpu = os.cpu_count() or 2
_num_workers = min(8, max(2, _cpu // 2))

_loader_kwargs = dict(
    num_workers=_num_workers,
    pin_memory=(device.type == "cuda"),
    persistent_workers=(_num_workers > 0),
    prefetch_factor=4 if _num_workers > 0 else None,
    worker_init_fn=_seed_worker,
)

train_loader = DataLoader(
    _TFRecordIterableDataset(train_tfrec_files, mode="train", id_set=_tr_set),
    batch_size=BATCH_SIZE,
    shuffle=False,  # IterableDataset cannot be shuffled by DataLoader; TFRecords already provide random-ish order.
    **{k: v for k, v in _loader_kwargs.items() if v is not None},
)

val_loader = DataLoader(
    _TFRecordIterableDataset(train_tfrec_files, mode="val", id_set=_va_set),
    batch_size=BATCH_SIZE,
    shuffle=False,
    **{k: v for k, v in _loader_kwargs.items() if v is not None},
)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=1e-4
)

if device.type == "cuda":
    model = model.to(memory_format=torch.channels_last)

if device.type == "cuda":
    _mean_t_dev = _mean_t.to(device, non_blocking=True)
    _std_t_dev = _std_t.to(device, non_blocking=True)
else:
    _mean_t_dev = _mean_t
    _std_t_dev = _std_t




## === cell 7
model.train()
for epoch in range(EPOCHS):
    model.train()
    tr_loss = 0.0
    tr_correct = 0
    n = 0

    for xb, yb in train_loader:
        if device.type == "cuda":
            xb = xb.to(device, non_blocking=True).contiguous(
                memory_format=torch.channels_last
            )
        else:
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
            if device.type == "cuda":
                xb = xb.to(device, non_blocking=True).contiguous(
                    memory_format=torch.channels_last
                )
            else:
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




## === cell 8
test_loader = DataLoader(
    _TFRecordIterableDataset(test_tfrec_files, mode="test", id_set=None),
    batch_size=64,
    shuffle=False,
    **{k: v for k, v in _loader_kwargs.items() if v is not None},
)

model.eval()
all_ids = []
all_preds = []

with torch.no_grad():
    for xb, ids in test_loader:
        if device.type == "cuda":
            xb = xb.to(device, non_blocking=True).contiguous(
                memory_format=torch.channels_last
            )
        else:
            xb = xb.to(device, non_blocking=True)
        logits = model(xb)
        pred = torch.argmax(logits, dim=1).cpu().numpy().astype(int)
        all_preds.append(pred)
        all_ids.extend(list(ids))

all_preds = np.concatenate(all_preds, axis=0)

results = pd.DataFrame({"image_id": all_ids, "label": all_preds.astype(int)})

expected_set = set(expected)
if expected_set == set(results["image_id"].tolist()):
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
