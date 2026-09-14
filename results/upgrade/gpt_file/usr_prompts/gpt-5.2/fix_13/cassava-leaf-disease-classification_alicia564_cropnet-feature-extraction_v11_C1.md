# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


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

# 5. Target score

0.8916591115140526

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torchvision import models

CPU_COUNT = os.cpu_count() or 2
torch.set_num_threads(min(8, max(1, CPU_COUNT)))
try:
    torch.set_num_interop_threads(min(4, max(1, CPU_COUNT // 2)))
except Exception:
    pass

os.environ.setdefault("PYTHONHASHSEED", "42")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True  # keep as in original

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Torch:", torch.__version__, "Device:", device)

DATA_DIR = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_CSV_PATH = f"{DATA_DIR}/train.csv"
SAMPLE_SUB_PATH = f"{DATA_DIR}/sample_submission.csv"
TRAIN_IMG_DIR = f"{DATA_DIR}/train_images"
TEST_IMG_DIR = f"{DATA_DIR}/test_images"
TRAIN_TFREC_DIR = f"{DATA_DIR}/train_tfrecords"
TEST_TFREC_DIR = f"{DATA_DIR}/test_tfrecords"

assert os.path.exists(TRAIN_CSV_PATH), f"Missing {TRAIN_CSV_PATH}"
assert os.path.exists(SAMPLE_SUB_PATH), f"Missing {SAMPLE_SUB_PATH}"
assert os.path.isdir(TRAIN_IMG_DIR), f"Missing {TRAIN_IMG_DIR}"
assert os.path.isdir(TEST_IMG_DIR), f"Missing {TEST_IMG_DIR}"
assert os.path.isdir(TRAIN_TFREC_DIR), f"Missing {TRAIN_TFREC_DIR}"
assert os.path.isdir(TEST_TFREC_DIR), f"Missing {TEST_TFREC_DIR}"

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
import glob
import math
import tensorflow as tf

tf.random.set_seed(SEED)
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

IMG_SIZE = 224
BATCH_SIZE = 32

AUTOTUNE = tf.data.AUTOTUNE
TF_NUM_PARALLEL = min(16, max(2, (os.cpu_count() or 2)))

_FEATURES = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),
}


def _parse_train_example(example_proto):
    ex = tf.io.parse_single_example(example_proto, _FEATURES)
    img = tf.io.decode_jpeg(ex["image"], channels=3)  # uint8 [H,W,3]
    label = tf.cast(ex["target"], tf.int32)
    return img, label


def _parse_test_example(example_proto):
    ex = tf.io.parse_single_example(
        example_proto,
        {
            "image": tf.io.FixedLenFeature([], tf.string),
            "image_name": tf.io.FixedLenFeature([], tf.string),
        },
    )
    img = tf.io.decode_jpeg(ex["image"], channels=3)
    image_id = ex["image_name"]
    return img, image_id


_MEAN = tf.constant([0.485, 0.456, 0.406], tf.float32)
_STD = tf.constant([0.229, 0.224, 0.225], tf.float32)


def _resize(img):
    img = tf.image.resize(
        img, [IMG_SIZE, IMG_SIZE], method=tf.image.ResizeMethod.BILINEAR, antialias=True
    )
    return img


def _to_norm_float(img):
    img = tf.cast(img, tf.float32) / 255.0
    img = (img - _MEAN) / _STD
    return img


try:
    import tensorflow_addons as tfa  # type: ignore

    _rotate = lambda x, a: tfa.image.rotate(
        x, a, interpolation="BILINEAR", fill_mode="reflect"
    )
except Exception:
    if hasattr(tf.image, "rotate"):
        _rotate = lambda x, a: tf.image.rotate(
            x, a, interpolation="BILINEAR", fill_mode="reflect"
        )
    else:
        raise RuntimeError(
            "No available rotate op to preserve RandomRotation(45) core logic."
        )


def _augment(img, seed_pair):
    img = _resize(img)

    img = tf.image.stateless_random_flip_left_right(img, seed=seed_pair)
    img = tf.image.stateless_random_flip_up_down(
        img, seed=seed_pair + tf.constant([1, 0], tf.int32)
    )

    angle = tf.random.stateless_uniform(
        [],
        seed=seed_pair + tf.constant([2, 0], tf.int32),
        minval=-45.0,
        maxval=45.0,
        dtype=tf.float32,
    ) * (math.pi / 180.0)
    img = _rotate(img, angle)

    b = tf.random.stateless_uniform(
        [],
        seed=seed_pair + tf.constant([3, 0], tf.int32),
        minval=-0.1,
        maxval=0.1,
        dtype=tf.float32,
    )
    img = tf.image.adjust_brightness(img, delta=b)
    c = tf.random.stateless_uniform(
        [],
        seed=seed_pair + tf.constant([4, 0], tf.int32),
        minval=0.9,
        maxval=1.1,
        dtype=tf.float32,
    )
    img = tf.image.adjust_contrast(img, contrast_factor=c)
    s = tf.random.stateless_uniform(
        [],
        seed=seed_pair + tf.constant([5, 0], tf.int32),
        minval=0.9,
        maxval=1.1,
        dtype=tf.float32,
    )
    img = tf.image.adjust_saturation(img, saturation_factor=s)
    h = tf.random.stateless_uniform(
        [],
        seed=seed_pair + tf.constant([6, 0], tf.int32),
        minval=-0.02,
        maxval=0.02,
        dtype=tf.float32,
    )
    img = tf.image.adjust_hue(img, delta=h)

    img = _to_norm_float(img)
    return img


def _preprocess_valid(img):
    img = _resize(img)
    img = _to_norm_float(img)
    return img


def _to_torch_batch(x):
    x = tf.transpose(x, [0, 3, 1, 2])
    return torch.from_numpy(x.numpy())


def _to_torch_labels(y):
    return torch.from_numpy(y.numpy()).long()


def _to_torch_ids(ids):
    ids_np = ids.numpy()
    return [b.decode("utf-8") for b in ids_np.tolist()]


def build_train_dataset(tfrecord_files, steps_seed=SEED):
    ds = tf.data.TFRecordDataset(tfrecord_files, num_parallel_reads=TF_NUM_PARALLEL)
    ds = ds.map(_parse_train_example, num_parallel_calls=AUTOTUNE)

    ds = ds.shuffle(8192, seed=SEED, reshuffle_each_iteration=True)

    ds = ds.enumerate()

    def _aug_map(i, data):
        img, label = data
        seed_pair = tf.stack([tf.cast(steps_seed, tf.int32), tf.cast(i, tf.int32)])
        img = _augment(img, seed_pair)
        return img, label

    ds = ds.map(_aug_map, num_parallel_calls=AUTOTUNE)

    ds = ds.batch(BATCH_SIZE, drop_remainder=True)
    ds = ds.cache()
    ds = ds.prefetch(AUTOTUNE)
    return ds


def build_valid_dataset(tfrecord_files):
    ds = tf.data.TFRecordDataset(tfrecord_files, num_parallel_reads=TF_NUM_PARALLEL)
    ds = ds.map(_parse_train_example, num_parallel_calls=AUTOTUNE)
    ds = ds.map(lambda img, y: (_preprocess_valid(img), y), num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.cache()
    ds = ds.prefetch(AUTOTUNE)
    return ds


def build_test_dataset(tfrecord_files, batch_size=64):
    ds = tf.data.TFRecordDataset(tfrecord_files, num_parallel_reads=TF_NUM_PARALLEL)
    ds = ds.map(_parse_test_example, num_parallel_calls=AUTOTUNE)
    ds = ds.map(
        lambda img, image_id: (_preprocess_valid(img), image_id),
        num_parallel_calls=AUTOTUNE,
    )
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.cache()
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_tfrecs = sorted(glob.glob(os.path.join(TRAIN_TFREC_DIR, "*.tfrec")))
test_tfrecs = sorted(glob.glob(os.path.join(TEST_TFREC_DIR, "*.tfrec")))
assert len(train_tfrecs) > 0 and len(test_tfrecs) > 0, "TFRecord files not found."

full_train_ds = tf.data.TFRecordDataset(
    train_tfrecs, num_parallel_reads=TF_NUM_PARALLEL
).map(_parse_train_example, num_parallel_calls=AUTOTUNE)

_FEATURES_WITH_NAME = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),
}


def _parse_train_with_name(example_proto):
    ex = tf.io.parse_single_example(example_proto, _FEATURES_WITH_NAME)
    img = tf.io.decode_jpeg(ex["image"], channels=3)
    label = tf.cast(ex["target"], tf.int32)
    image_id = ex["image_name"]
    return img, label, image_id


full_with_name = tf.data.TFRecordDataset(
    train_tfrecs, num_parallel_reads=TF_NUM_PARALLEL
).map(_parse_train_with_name, num_parallel_calls=AUTOTUNE)

train_ids_set = set(train_split["image_id"].astype(str).tolist())
valid_ids_set = set(valid_split["image_id"].astype(str).tolist())

keys = tf.constant(list(train_ids_set), dtype=tf.string)
vals = tf.ones([tf.shape(keys)[0]], dtype=tf.int32)
train_table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(keys, vals), default_value=0
)

keys_v = tf.constant(list(valid_ids_set), dtype=tf.string)
vals_v = tf.ones([tf.shape(keys_v)[0]], dtype=tf.int32)
valid_table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(keys_v, vals_v), default_value=0
)

train_named = full_with_name.filter(
    lambda img, y, image_id: tf.equal(train_table.lookup(image_id), 1)
)
valid_named = full_with_name.filter(
    lambda img, y, image_id: tf.equal(valid_table.lookup(image_id), 1)
)

train_named = train_named.map(
    lambda img, y, image_id: (img, y), num_parallel_calls=AUTOTUNE
)
valid_named = valid_named.map(
    lambda img, y, image_id: (img, y), num_parallel_calls=AUTOTUNE
)


def build_train_from_ds(ds, steps_seed=SEED):
    ds = ds.shuffle(8192, seed=SEED, reshuffle_each_iteration=True)
    ds = ds.enumerate()
    ds = ds.map(
        lambda i, data: (
            _augment(
                data[0], tf.stack([tf.cast(steps_seed, tf.int32), tf.cast(i, tf.int32)])
            ),
            data[1],
        ),
        num_parallel_calls=AUTOTUNE,
    )
    ds = ds.batch(BATCH_SIZE, drop_remainder=True)
    ds = ds.cache()
    ds = ds.prefetch(AUTOTUNE)
    return ds


def build_valid_from_ds(ds):
    ds = ds.map(lambda img, y: (_preprocess_valid(img), y), num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.cache()
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_ds_tf = build_train_from_ds(train_named, steps_seed=SEED)
valid_ds_tf = build_valid_from_ds(valid_named)

print(
    "TFRecords:",
    len(train_tfrecs),
    "train batches:",
    "unknown (streamed)",
    "valid batches:",
    "unknown (streamed)",
)




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 3
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, NUM_CLASSES)
model = model.to(device)

if device.type == "cuda":
    model = model.to(memory_format=torch.channels_last)

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


ENABLE_COMPILE = False
if ENABLE_COMPILE and hasattr(torch, "compile") and device.type == "cuda":
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

for epoch in range(1, EPOCHS + 1):
    model.train()
    tr_loss = 0.0
    tr_correct = 0
    n_tr = 0

    for x_tf, y_tf in train_ds_tf:
        xb = _to_torch_batch(x_tf)
        yb = _to_torch_labels(y_tf)

        xb = xb.to(device, non_blocking=True)
        if device.type == "cuda":
            xb = xb.contiguous(memory_format=torch.channels_last)
        yb = yb.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        bs = yb.size(0)
        tr_loss += float(loss.detach()) * bs
        tr_correct += int(batch_correct(logits.detach(), yb).detach().cpu().item())
        n_tr += bs

    tr_loss /= max(1, n_tr)
    tr_acc = tr_correct / max(1, n_tr)

    model.eval()
    va_loss = 0.0
    va_correct = 0
    n_va = 0

    with torch.inference_mode():
        for x_tf, y_tf in valid_ds_tf:
            xb = _to_torch_batch(x_tf)
            yb = _to_torch_labels(y_tf)

            xb = xb.to(device, non_blocking=True)
            if device.type == "cuda":
                xb = xb.contiguous(memory_format=torch.channels_last)
            yb = yb.to(device, non_blocking=True)

            logits = model(xb)
            loss = criterion(logits, yb)

            bs = yb.size(0)
            va_loss += float(loss.detach()) * bs
            va_correct += int(batch_correct(logits, yb).detach().cpu().item())
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

if best_state is not None:
    model.load_state_dict(best_state)
print("Best val acc:", best_val_acc)




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2184863605.py in <cell line: 0>()
     12     n_tr = 0
     13 
---> 14     for x_tf, y_tf in train_ds_tf:
     15         xb = _to_torch_batch(x_tf)
     16         yb = _to_torch_labels(y_tf)

NameError: name 'train_ds_tf' is not defined

## === cell 5
test_ds_tf = build_test_dataset(test_tfrecs, batch_size=64)

model.eval()
preds = []
ids_all = []

with torch.inference_mode():
    for x_tf, ids_tf in test_ds_tf:
        xb = _to_torch_batch(x_tf).to(device, non_blocking=True)
        if device.type == "cuda":
            xb = xb.contiguous(memory_format=torch.channels_last)
        logits = model(xb)
        pred = logits.argmax(dim=1).detach().cpu().numpy().astype(int).tolist()
        preds.extend(pred)
        ids_all.extend(_to_torch_ids(ids_tf))

assert len(preds) == len(ids_all), "Mismatch between predictions and ids."

pred_map = dict(zip(ids_all, preds))
missing_in_pred = [
    img_id
    for img_id in sample_sub["image_id"].astype(str).tolist()
    if img_id not in pred_map
]
assert (
    len(missing_in_pred) == 0
), f"Missing predictions for {len(missing_in_pred)} test images."

submission_df = sample_sub.copy()
submission_df["label"] = (
    submission_df["image_id"].astype(str).map(pred_map).astype(np.int64)
)

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

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2576322652.py in <cell line: 0>()
      1 # --- Speed fix: test inference via TFRecords as well (avoids filesystem stat calls + per-file JPEG overhead).
----> 2 test_ds_tf = build_test_dataset(test_tfrecs, batch_size=64)
      3 
      4 model.eval()
      5 preds = []

NameError: name 'build_test_dataset' is not defined
