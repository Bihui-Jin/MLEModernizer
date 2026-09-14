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

# 5. Target score

0.8158053792686613

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.50486) has done: 'The failures come from `keras_cv` being incompatible in this environment (protobuf/TF/Keras version mismatch) and from referencing a non-existent `EfficientNetV2B3` symbol, which prevents `model` from being created and cascades into later `NameError`s. To keep the same core idea (ImageNet-pretrained EfficientNetV2 with 448×448 input and 5-class softmax), I remove `keras_cv` and switch to the built-in `tf.keras.applications.EfficientNetV2B3` for inference-only, which is stable on Kaggle. I also add a small path-resolver so the notebook works whether the dataset is mounted at `/kaggle/input/...` or mirrored under `/kaggle/data/...`. Finally, I ensure preprocessing matches the chosen application model and that a valid `submission.csv` is always written with the correct columns and row order.'
- What this solution (achieved 0.16143) has done: 'I fix the TensorFlow import crash (`MessageFactory.GetPrototype`) by removing the TensorFlow dependency entirely and switching to a stable built-in Keras application model for inference. To improve accuracy toward your target, I keep the same “ImageNet-pretrained CNN + softmax head” core idea but use `EfficientNetB3` with its matching `preprocess_input`, since it’s widely supported in Kaggle’s default environment and avoids the TF/protobuf mismatch. I also keep your dataset path resolution and submission-order alignment logic, ensuring `submission.csv` is always written with the exact required columns and row count. These changes are minimal and directly address both the runtime failure and the low score due to the model never successfully running in this environment.'

# 9. Code solution

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
from torch.utils.data import DataLoader
from torchvision import models, transforms

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

_label_map = dict(
    zip(
        train_df["image_id"].astype(str).tolist(),
        train_df["label"].astype(int).tolist(),
    )
)



## === cell 4
import tensorflow as tf

tf.random.set_seed(42)

TARGET_H, TARGET_W = TARGET_SIZE
_mean_np = np.asarray(mean, dtype=np.float32)
_std_np = np.asarray(std, dtype=np.float32)

train_tfrec_dir = os.path.join(DATA_DIR, "train_tfrecords")
test_tfrec_dir = os.path.join(DATA_DIR, "test_tfrecords")
assert os.path.isdir(train_tfrec_dir), f"train_tfrecords not found: {train_tfrec_dir}"
assert os.path.isdir(test_tfrec_dir), f"test_tfrecords not found: {test_tfrec_dir}"

train_tfrec_files = sorted(
    [
        os.path.join(train_tfrec_dir, f)
        for f in os.listdir(train_tfrec_dir)
        if f.endswith(".tfrec")
    ]
)
test_tfrec_files = sorted(
    [
        os.path.join(test_tfrec_dir, f)
        for f in os.listdir(test_tfrec_dir)
        if f.endswith(".tfrec")
    ]
)
assert len(train_tfrec_files) > 0, "No train tfrecords found"
assert len(test_tfrec_files) > 0, "No test tfrecords found"

_FEATURES = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_id": tf.io.FixedLenFeature([], tf.string),
    "label": tf.io.FixedLenFeature([], tf.int64),
}


def _parse_example(example_proto):
    ex = tf.io.parse_single_example(example_proto, _FEATURES)
    img = tf.io.decode_jpeg(ex["image"], channels=3)  # uint8 HWC
    img = tf.image.resize(
        img, [TARGET_H, TARGET_W], method=tf.image.ResizeMethod.BICUBIC, antialias=True
    )
    img = tf.cast(img, tf.float32) / 255.0  # float32 [0,1]
    return img, ex["label"], ex["image_id"]


def _augment_train(img, label, image_id):
    img = tf.cond(
        tf.random.uniform(()) < 0.5, lambda: tf.image.flip_left_right(img), lambda: img
    )
    b = 0.1
    c = 0.1
    s = 0.1
    h = 0.02
    img = tf.image.random_brightness(img, max_delta=b)
    img = tf.image.random_contrast(img, lower=max(0.0, 1.0 - c), upper=1.0 + c)
    img = tf.image.random_saturation(img, lower=max(0.0, 1.0 - s), upper=1.0 + s)
    img = tf.image.random_hue(img, max_delta=h)
    img = tf.clip_by_value(img, 0.0, 1.0)
    return img, label, image_id


def _normalize(img, label, image_id):
    img = (img - _mean_np) / _std_np
    return img, label, image_id


_options = tf.data.Options()
_options.experimental_deterministic = True


def make_train_dataset(batch_size):
    ds = tf.data.TFRecordDataset(train_tfrec_files, num_parallel_reads=tf.data.AUTOTUNE)
    ds = ds.with_options(_options)
    ds = ds.map(_parse_example, num_parallel_calls=tf.data.AUTOTUNE)
    tr_ids = tf.constant(tr_df["image_id"].astype(str).tolist())
    tr_table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            tr_ids, tf.ones_like(tr_ids, dtype=tf.int32)
        ),
        default_value=0,
    )

    def _keep(img, label, image_id):
        return tr_table.lookup(image_id) > 0

    ds = ds.filter(_keep)

    ds = ds.shuffle(4096, seed=42, reshuffle_each_iteration=True)
    ds = ds.map(_augment_train, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(_normalize, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def make_val_dataset(batch_size):
    ds = tf.data.TFRecordDataset(train_tfrec_files, num_parallel_reads=tf.data.AUTOTUNE)
    ds = ds.with_options(_options)
    ds = ds.map(_parse_example, num_parallel_calls=tf.data.AUTOTUNE)

    va_ids = tf.constant(va_df["image_id"].astype(str).tolist())
    va_table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            va_ids, tf.ones_like(va_ids, dtype=tf.int32)
        ),
        default_value=0,
    )

    def _keep(img, label, image_id):
        return va_table.lookup(image_id) > 0

    ds = ds.filter(_keep)

    ds = ds.map(_normalize, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def make_test_dataset(batch_size):
    ds = tf.data.TFRecordDataset(test_tfrec_files, num_parallel_reads=tf.data.AUTOTUNE)
    ds = ds.with_options(_options)
    ds = ds.map(_parse_example, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(_normalize, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def _tf_batch_to_torch(x_tf):
    x_dl = tf.experimental.dlpack.to_dlpack(x_tf)
    x_t = torch.utils.dlpack.from_dlpack(x_dl)  # torch tensor on CPU
    return x_t




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 5
BATCH_SIZE = 32
EPOCHS = 2  # keep same as provided
LR = 3e-3

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=1e-4
)

if device.type == "cuda":
    model = model.to(memory_format=torch.channels_last)

train_ds = make_train_dataset(BATCH_SIZE)
val_ds = make_val_dataset(BATCH_SIZE)



## === cell 6
model.train()
for epoch in range(EPOCHS):
    model.train()
    tr_loss = 0.0
    tr_correct = 0
    n = 0

    for x_tf, y_tf, _id_tf in train_ds:
        xb = _tf_batch_to_torch(x_tf).permute(0, 3, 1, 2).contiguous()  # NHWC->NCHW
        yb = _tf_batch_to_torch(tf.cast(y_tf, tf.int64)).contiguous()

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
        for x_tf, y_tf, _id_tf in val_ds:
            xb = _tf_batch_to_torch(x_tf).permute(0, 3, 1, 2).contiguous()
            yb = _tf_batch_to_torch(tf.cast(y_tf, tf.int64)).contiguous()

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



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
InvalidArgumentError                      Traceback (most recent call last)
/tmp/ipykernel_11/1679008029.py in <cell line: 0>()
      7     n = 0
      8 
----> 9     for x_tf, y_tf, _id_tf in train_ds:
     10         xb = _tf_batch_to_torch(x_tf).permute(0, 3, 1, 2).contiguous()  # NHWC->NCHW
     11         yb = _tf_batch_to_torch(tf.cast(y_tf, tf.int64)).contiguous()

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/iterator_ops.py in __next__(self)
    824   def __next__(self):
    825     try:
--> 826       return self._next_internal()
    827     except errors.OutOfRangeError:
    828       raise StopIteration

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/iterator_ops.py in _next_internal(self)
    774     # to communicate that there is no more data to iterate over.
    775     with context.execution_mode(context.SYNC):
--> 776       ret = gen_dataset_ops.iterator_get_next(
    777           self._iterator_resource,
    778           output_types=self._flat_output_types,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/gen_dataset_ops.py in iterator_get_next(iterator, output_types, output_shapes, name)
   3084       return _result
   3085     except _core._NotOkStatusException as e:
-> 3086       _ops.raise_from_not_ok_status(e, name)
   3087     except _core._FallbackException:
   3088       pass

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py in raise_from_not_ok_status(e, name)
   6000 def raise_from_not_ok_status(e, name) -> NoReturn:
   6001   e.message += (" name: " + str(name if name is not None else ""))
-> 6002   raise core._status_to_exception(e) from None  # pylint: disable=protected-access
   6003 
   6004 

InvalidArgumentError: {{function_node __wrapped__IteratorGetNext_output_types_3_device_/job:localhost/replica:0/task:0/device:CPU:0}} Error in user-defined function passed to ParallelMapDatasetV2:3 transformation with iterator: Iterator::Root::Prefetch::MapAndBatch::ParallelMapV2::Shuffle::Filter::ParallelMapV2: Feature: image_id (data type: string) is required but could not be found.
	 [[{{node ParseSingleExample/ParseExample/ParseExampleV2}}]] [Op:IteratorGetNext] name: 

## === cell 7
test_ds = make_test_dataset(batch_size=64)

model.eval()
all_ids = []
all_preds = []

with torch.no_grad():
    for x_tf, _y_tf, id_tf in test_ds:
        xb = _tf_batch_to_torch(x_tf).permute(0, 3, 1, 2).contiguous()
        if device.type == "cuda":
            xb = xb.to(device, non_blocking=True).contiguous(
                memory_format=torch.channels_last
            )
        else:
            xb = xb.to(device, non_blocking=True)

        logits = model(xb)
        pred = torch.argmax(logits, dim=1).cpu().numpy().astype(int)
        all_preds.append(pred)

        ids_np = id_tf.numpy()
        all_ids.extend([b.decode("utf-8") for b in ids_np])

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

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
InvalidArgumentError                      Traceback (most recent call last)
/tmp/ipykernel_11/2716119519.py in <cell line: 0>()
      6 
      7 with torch.no_grad():
----> 8     for x_tf, _y_tf, id_tf in test_ds:
      9         xb = _tf_batch_to_torch(x_tf).permute(0, 3, 1, 2).contiguous()
     10         if device.type == "cuda":

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/iterator_ops.py in __next__(self)
    824   def __next__(self):
    825     try:
--> 826       return self._next_internal()
    827     except errors.OutOfRangeError:
    828       raise StopIteration

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/iterator_ops.py in _next_internal(self)
    774     # to communicate that there is no more data to iterate over.
    775     with context.execution_mode(context.SYNC):
--> 776       ret = gen_dataset_ops.iterator_get_next(
    777           self._iterator_resource,
    778           output_types=self._flat_output_types,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/gen_dataset_ops.py in iterator_get_next(iterator, output_types, output_shapes, name)
   3084       return _result
   3085     except _core._NotOkStatusException as e:
-> 3086       _ops.raise_from_not_ok_status(e, name)
   3087     except _core._FallbackException:
   3088       pass

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py in raise_from_not_ok_status(e, name)
   6000 def raise_from_not_ok_status(e, name) -> NoReturn:
   6001   e.message += (" name: " + str(name if name is not None else ""))
-> 6002   raise core._status_to_exception(e) from None  # pylint: disable=protected-access
   6003 
   6004 

InvalidArgumentError: {{function_node __wrapped__IteratorGetNext_output_types_3_device_/job:localhost/replica:0/task:0/device:CPU:0}} Error in user-defined function passed to ParallelMapDatasetV2:28 transformation with iterator: Iterator::Root::Prefetch::MapAndBatch::ParallelMapV2: Feature: image_id (data type: string) is required but could not be found.
	 [[{{node ParseSingleExample/ParseExample/ParseExampleV2}}]] [Op:IteratorGetNext] name:
