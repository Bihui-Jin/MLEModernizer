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
TRAIN_TFREC_DIR = os.path.join(DATA_DIR, "train_tfrecords")
TEST_TFREC_DIR = os.path.join(DATA_DIR, "test_tfrecords")

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
from torch.utils.data import DataLoader

try:
    import torchvision
    from torchvision.transforms import v2 as T  # torchvision>=0.15

    _HAS_V2 = True
except Exception:
    import torchvision
    import torchvision.transforms as T

    _HAS_V2 = False

import tensorflow as tf

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

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

tf.random.set_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

IMG_SIZE = 224

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

    class ToFloat01(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x.to(dtype=torch.float32).div_(255.0)

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

NUM_CLASSES = int(train_df["label"].nunique())
assert NUM_CLASSES == 5, f"Expected 5 classes, got {NUM_CLASSES}"


def _list_tfrec_files(dir_path: str):
    if not os.path.isdir(dir_path):
        return []
    files = [
        os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith(".tfrec")
    ]
    return sorted(files)


train_tfrecs = _list_tfrec_files(TRAIN_TFREC_DIR)
test_tfrecs = _list_tfrec_files(TEST_TFREC_DIR)
assert len(train_tfrecs) > 0, "No train tfrecords found"
assert len(test_tfrecs) > 0, "No test tfrecords found"
print("tfrecords:", len(train_tfrecs), "train |", len(test_tfrecs), "test")

_TF_FEATURES = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),
}


def _parse_example(example_proto):
    ex = tf.io.parse_single_example(example_proto, _TF_FEATURES)
    return ex["image_name"], ex["image"], ex["target"]


def _build_train_index(tfrec_files):
    index = {}
    for tfp in tfrec_files:
        ds = tf.data.TFRecordDataset(tfp, num_parallel_reads=1)
        ds = ds.map(
            _parse_example, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True
        )
        i = 0
        for name, _img, y in ds:
            n = name.numpy().decode("utf-8")
            index[n] = (tfp, i, int(y.numpy()))
            i += 1
    return index


t0 = time.time()
train_index = _build_train_index(train_tfrecs)
print("indexed train tfrecords:", len(train_index), "in", f"{time.time()-t0:.1f}s")
assert len(train_index) == len(train_df), "TFRecord train count mismatch with train.csv"


def _make_split_records(df):
    recs = []
    for img_id, y in zip(df["image_id"].tolist(), df["label"].astype(int).tolist()):
        tfp, ridx, y2 = train_index[img_id]
        assert y2 == int(y), "Label mismatch between CSV and TFRecord"
        recs.append((tfp, ridx, int(y)))
    return recs


train_records = _make_split_records(tr_df)
val_records = _make_split_records(va_df)


class TFRecordIterable(torch.utils.data.IterableDataset):
    def __init__(self, records, training: bool, transform, seed: int):
        super().__init__()
        self.records = records
        self.training = training
        self.transform = transform
        self.seed = seed

        self.by_file = {}
        for tfp, ridx, y in records:
            self.by_file.setdefault(tfp, []).append((ridx, y))

        for tfp in self.by_file:
            self.by_file[tfp].sort(key=lambda t: t[0])

        self.flat = []
        for tfp, lst in self.by_file.items():
            for j in range(len(lst)):
                self.flat.append((tfp, j))

    def __len__(self):
        return len(self.records)

    def _make_tf_dataset_for_file(self, tfp: str):
        ds = tf.data.TFRecordDataset(tfp, num_parallel_reads=1)
        ds = ds.map(
            _parse_example, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True
        )
        return ds

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        if info is None:
            worker_id, num_workers = 0, 1
        else:
            worker_id, num_workers = info.id, info.num_workers

        idxs = np.arange(len(self.flat), dtype=np.int64)[worker_id::num_workers]

        if self.training:
            rng = np.random.RandomState(self.seed + worker_id)
            rng.shuffle(idxs)

        file_cache = {}
        for idx in idxs:
            tfp, j = self.flat[int(idx)]
            if tfp not in file_cache:
                ridx_y = self.by_file[tfp]
                need_idxs = set(r for r, _ in ridx_y)
                ds = self._make_tf_dataset_for_file(tfp)
                keep = {}
                k = 0
                for name, img_bytes, y_tf in ds:
                    if k in need_idxs:
                        keep[k] = (img_bytes.numpy(), int(y_tf.numpy()))
                        if len(keep) == len(need_idxs):
                            break
                    k += 1
                file_cache[tfp] = (ridx_y, keep)

            ridx_y, keep = file_cache[tfp]
            ridx, y = ridx_y[j]
            img_bytes, y2 = keep[ridx]
            img = torchvision.io.decode_jpeg(
                torch.frombuffer(img_bytes, dtype=torch.uint8), device="cpu"
            )
            img = img.contiguous()
            if self.transform is not None:
                img = self.transform(img)
            if self.training or True:
                yield img, int(y2)


BATCH_SIZE = 64 if device.type == "cuda" else 32
NUM_WORKERS = 0
PIN_MEMORY = device.type == "cuda"
PERSISTENT = False
PREFETCH_FACTOR = None

train_ds = TFRecordIterable(
    train_records, training=True, transform=train_tfms, seed=SEED
)
val_ds = TFRecordIterable(val_records, training=False, transform=test_tfms, seed=SEED)

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
)
val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
)

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

        def _move_x(x: torch.Tensor) -> torch.Tensor:
            x = x.to(self.device, non_blocking=True)
            if self.channels_last:
                x = x.contiguous(memory_format=torch.channels_last)
            return x

        def _preload():
            nonlocal next_batch
            try:
                batch = next(it)
            except StopIteration:
                next_batch = None
                return
            with torch.cuda.stream(self.stream):
                x, y = batch
                next_batch = (_move_x(x), torch.as_tensor(y, device=self.device))

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
test_id_to_pos = {img_id: i for i, img_id in enumerate(test_image_ids)}
pred_labels = np.empty(len(test_image_ids), dtype=np.int64)

_TEST_TF_FEATURES = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
}


def _parse_test_example(example_proto):
    ex = tf.io.parse_single_example(example_proto, _TEST_TF_FEATURES)
    return ex["image_name"], ex["image"]


test_ds_tf = tf.data.TFRecordDataset(test_tfrecs, num_parallel_reads=1)
test_ds_tf = test_ds_tf.map(
    _parse_test_example, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True
)
test_ds_tf = test_ds_tf.prefetch(tf.data.AUTOTUNE)

model.eval()
with torch.inference_mode():
    batch_imgs = []
    batch_pos = []
    for name, img_bytes in test_ds_tf:
        img_id = name.numpy().decode("utf-8")
        pos = test_id_to_pos.get(img_id, None)
        if pos is None:
            continue
        img = torchvision.io.decode_jpeg(
            torch.frombuffer(img_bytes.numpy(), dtype=torch.uint8), device="cpu"
        )
        img = img.contiguous()
        img = test_tfms(img)
        batch_imgs.append(img)
        batch_pos.append(pos)

        if len(batch_imgs) == BATCH_SIZE:
            x = torch.stack(batch_imgs, dim=0)
            if device.type == "cuda":
                x = x.to(device, non_blocking=True).contiguous(
                    memory_format=torch.channels_last
                )
            else:
                x = x.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1).detach().cpu().numpy().astype(np.int64)
            pred_labels[np.array(batch_pos, dtype=np.int64)] = preds
            batch_imgs.clear()
            batch_pos.clear()

    if batch_imgs:
        x = torch.stack(batch_imgs, dim=0)
        if device.type == "cuda":
            x = x.to(device, non_blocking=True).contiguous(
                memory_format=torch.channels_last
            )
        else:
            x = x.to(device)
        logits = model(x)
        preds = logits.argmax(dim=1).detach().cpu().numpy().astype(np.int64)
        pred_labels[np.array(batch_pos, dtype=np.int64)] = preds

submission_df = pd.DataFrame(
    {"image_id": test_image_ids, "label": pred_labels.astype(int).tolist()}
)

submission_path = "/kaggle/working/submission.csv"
submission_df.to_csv(submission_path, index=False)

assert list(submission_df.columns) == ["image_id", "label"]
assert len(submission_df) == len(sample_df)

print(f"Submission file saved at: {submission_path}")
print(submission_df.head(10))
