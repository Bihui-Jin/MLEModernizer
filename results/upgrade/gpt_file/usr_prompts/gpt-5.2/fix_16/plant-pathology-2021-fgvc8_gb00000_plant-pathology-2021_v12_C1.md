# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Detect apple diseases from images.

## Metric
Mean F1-Score

## Submission Format
labels should be a space-delimited list.

The file should contain a header and have the following format:

```
image, labels
85f8cb619c66b863.jpg,healthy
ad8770db05586b59.jpg,healthy
c7b03e718489f3ca.jpg,healthy
```

## Dataset
**train.csv** - the training set metadata.

- `image` - the image ID.
- `labels` - the target classes, a space delimited list of all diseases found in the image. Unhealthy leaves with too many diseases to classify visually will have the `complex` class, and may also have a subset of the diseases identified.

**sample_submission.csv** - A sample submission file in the correct format.

- `image`
- `labels`

**train_images** - The training set images.

**test_images** - The test set images. This competition has a hidden test set: only three images are provided here as samples while the remaining 5,000 images will be available to your notebook once it is submitted.

# 2. Python version

3.9

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
        input/
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
        working/
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
```

-> data/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> data/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_auto_jit=2")

import numpy as np
import pandas as pd

import google.protobuf  # noqa: F401



## === cell 1
import sys

try:
    import tensorflow as tf
except Exception:
    if "tensorflow" in sys.modules:
        del sys.modules["tensorflow"]
    os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
    os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)
    import tensorflow as tf  # noqa: F401

print("TensorFlow version:", tf.__version__)
print("Num GPUs Available:", len(tf.config.list_physical_devices("GPU")))

tf.random.set_seed(42)
np.random.seed(42)

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass



## === cell 2
from tensorflow.keras.applications.resnet50 import (
    preprocess_input,
    ResNet50,
)
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model



## === cell 3
sam_sub = pd.read_csv("/kaggle/input/plant-pathology-2021-fgvc8/sample_submission.csv")
sam_sub.head()



## === cell 4
train_dir = "/kaggle/input/plant-pathology-2021-fgvc8/train_images"
test_dir = "/kaggle/input/plant-pathology-2021-fgvc8/test_images"



## === cell 5
train = pd.read_csv("/kaggle/input/plant-pathology-2021-fgvc8/train.csv")
train.head()



## === cell 6
test_df = sam_sub[["image"]].copy()
test_ids = test_df["image"].tolist()
test_df.head()



## === cell 7
all_classes = sorted(
    {lab for s in train["labels"].astype(str).tolist() for lab in s.split() if lab}
)
assert len(all_classes) > 0, "No classes found in train labels."

num_classes = len(all_classes)
y_cols = [f"y_{c}" for c in all_classes]

lbl_series = train["labels"].astype(str).str.get_dummies(sep=" ")
lbl_series = lbl_series.reindex(columns=all_classes, fill_value=0).astype(np.float32)
y = lbl_series.to_numpy(dtype=np.float32, copy=False)

train_ml = train.copy()
train_ml[y_cols] = y

train_ml.head()



## === cell 8
AUTOTUNE = tf.data.AUTOTUNE

IMG_H, IMG_W = 224, 336
BATCH_TRAIN = 16
BATCH_TEST = 32
VAL_SPLIT = 0.1

rng = np.random.RandomState(42)
idx = np.arange(len(train_ml))
rng.shuffle(idx)
val_size = int(np.floor(len(idx) * VAL_SPLIT))
val_idx = idx[:val_size]
train_idx = idx[val_size:]

train_split = train_ml.iloc[train_idx].reset_index(drop=True)
val_split = train_ml.iloc[val_idx].reset_index(drop=True)

train_paths = (
    train_dir.rstrip("/") + "/" + train_split["image"].astype(str)
).to_numpy()
val_paths = (train_dir.rstrip("/") + "/" + val_split["image"].astype(str)).to_numpy()
train_y = train_split[y_cols].to_numpy(dtype=np.float32, copy=False)
val_y = val_split[y_cols].to_numpy(dtype=np.float32, copy=False)

_rot_layer = tf.keras.layers.RandomRotation(
    factor=20.0 / 180.0, fill_mode="reflect", seed=42
)


@tf.function(reduce_retracing=True)
def _decode_resize(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)  # dataset is jpg
    img = tf.image.resize(img, (IMG_H, IMG_W), method=tf.image.ResizeMethod.BILINEAR)
    img = preprocess_input(img)  # ResNet50 preprocessing
    img = tf.ensure_shape(img, (IMG_H, IMG_W, 3))
    return img


@tf.function(reduce_retracing=True)
def _augment(img, seed):
    img = tf.image.stateless_random_flip_left_right(img, seed=seed)

    dx = tf.random.stateless_uniform(
        [], seed=seed + tf.constant([1, 0], tf.int32), minval=-0.1, maxval=0.1
    )
    dy = tf.random.stateless_uniform(
        [], seed=seed + tf.constant([2, 0], tf.int32), minval=-0.1, maxval=0.1
    )
    tx = tf.cast(tf.round(dx * tf.cast(IMG_W, tf.float32)), tf.int32)
    ty = tf.cast(tf.round(dy * tf.cast(IMG_H, tf.float32)), tf.int32)
    img = tf.roll(img, shift=[ty, tx], axis=[0, 1])

    img = _rot_layer(tf.expand_dims(img, 0), training=True)[0]
    img = tf.ensure_shape(img, (IMG_H, IMG_W, 3))
    return img


import zlib

_ds_options = tf.data.Options()
_ds_options.experimental_deterministic = True
try:
    _ds_options.experimental_optimization.map_parallelization = True
    _ds_options.experimental_optimization.parallel_batch = True
except Exception:
    pass


def _make_stateless_seeds_from_paths(paths: np.ndarray) -> np.ndarray:
    p = paths.astype(str, copy=False)
    out = np.empty((len(p), 2), dtype=np.int32)
    for i, s in enumerate(p):
        out[i, 0] = np.int32(zlib.crc32(s.encode("utf-8")) & 0xFFFFFFFF)
        out[i, 1] = np.int32(42)
    return out


train_seeds = _make_stateless_seeds_from_paths(train_paths)
val_seeds = _make_stateless_seeds_from_paths(val_paths)


def make_train_ds(paths, labels, seeds, batch_size):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels, seeds)).with_options(
        _ds_options
    )

    ds = ds.map(
        lambda p, y_, s: (_decode_resize(p), y_, s),
        num_parallel_calls=AUTOTUNE,
        deterministic=False,
    ).cache()

    ds = ds.shuffle(
        buffer_size=min(len(paths), 4096), seed=42, reshuffle_each_iteration=True
    )

    @tf.function(reduce_retracing=True)
    def _map_fn(img, lab, seed):
        img = _augment(img, seed)
        return img, lab

    ds = ds.map(_map_fn, num_parallel_calls=AUTOTUNE, deterministic=False)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def make_eval_ds(paths, labels, batch_size):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels)).with_options(_ds_options)

    ds = ds.map(
        lambda p, y_: (_decode_resize(p), y_),
        num_parallel_calls=AUTOTUNE,
        deterministic=False,
    ).cache()

    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_ds = make_train_ds(train_paths, train_y, train_seeds, BATCH_TRAIN)
val_ds = make_eval_ds(val_paths, val_y, BATCH_TRAIN)

print("Train samples:", len(train_paths), "Val samples:", len(val_paths))



## === cell 9
base = ResNet50(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 336, 3),
    pooling="avg",
)
base.trainable = False  # key speedup

x = base.output
out = Dense(num_classes, activation="sigmoid")(x)
trained_model_sub = Model(inputs=base.input, outputs=out)

trained_model_sub.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
)

trained_model_sub.summary()



## === cell 10
history = trained_model_sub.fit(
    train_ds,
    validation_data=val_ds,
    epochs=2,
    verbose=1,
)



## === cell 11
test_paths = (test_dir.rstrip("/") + "/" + test_df["image"].astype(str)).to_numpy()


def make_test_ds(paths, batch_size):
    ds = tf.data.Dataset.from_tensor_slices(paths).with_options(_ds_options)
    ds = ds.map(
        _decode_resize, num_parallel_calls=AUTOTUNE, deterministic=False
    ).cache()
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


test_ds = make_test_ds(test_paths, BATCH_TEST)

y_pred = trained_model_sub.predict(
    test_ds,
    verbose=1,
)
y_pred = np.asarray(y_pred)
print("Pred shape:", y_pred.shape)



## === cell 12
prevalence = y.mean(axis=0)  # fraction of positives per class
thr = np.clip(0.5 * (0.6 + (1.0 - prevalence)), 0.25, 0.75).astype(np.float32)

mask = y_pred >= thr  # (N, C) boolean
argmax_idx = np.argmax(y_pred, axis=1)

classes_arr = np.asarray(all_classes, dtype=object)
pred_labels = []
for i in range(mask.shape[0]):
    row = mask[i]
    if row.any():
        pred_labels.append(" ".join(classes_arr[row].tolist()))
    else:
        pred_labels.append(classes_arr[int(argmax_idx[i])])

print("Example preds:", pred_labels[:5])



## === cell 13
pred_map = pd.DataFrame({"image": test_ids, "labels": pred_labels})


def _normalize_label_str(s: str) -> str:
    toks = [t for t in str(s).strip().split() if t]
    if len(toks) <= 1:
        return toks[0] if toks else ""
    tokset = set(toks)
    ordered = [c for c in all_classes if c in tokset]
    return " ".join(ordered)


pred_map["labels"] = pred_map["labels"].map(_normalize_label_str)

sub = sam_sub[["image"]].merge(pred_map, on="image", how="left")
sub["labels"] = sub["labels"].fillna("healthy")

sub = sub[["image", "labels"]]

assert list(sub.columns) == ["image", "labels"]
assert len(sub) == len(sam_sub)

sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())
