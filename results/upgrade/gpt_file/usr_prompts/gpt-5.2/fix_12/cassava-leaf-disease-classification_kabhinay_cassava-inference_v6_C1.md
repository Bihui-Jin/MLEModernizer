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

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator,
)  # kept to preserve original imports

tf.random.set_seed(42)
np.random.seed(42)

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

AUTOTUNE = tf.data.AUTOTUNE

print("TF version:", tf.__version__)
print("Num GPUs:", len(tf.config.list_physical_devices("GPU")))

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass



## === cell 1
IMG_SIZE = 448
NUM_CLASSES = 5

base = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    pooling="avg",
)
base.trainable = True
for layer in base.layers[:-30]:
    layer.trainable = False

inp = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="image")
x = tf.keras.applications.efficientnet.preprocess_input(inp)
x = base(x, training=False)
x = tf.keras.layers.Dropout(0.2)(x)
out = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)
model_v4 = tf.keras.Model(inputs=inp, outputs=out, name="cassava_effnetb0_448")

model_v4.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
    steps_per_execution=16,
)

model_v4.summary()



## === cell 2
DATA_DIR = "../input/cassava-leaf-disease-classification"
train_csv_path = os.path.join(DATA_DIR, "train.csv")
train_img_dir = os.path.join(DATA_DIR, "train_images")
test_img_dir = os.path.join(DATA_DIR, "test_images")
sample_path = os.path.join(DATA_DIR, "sample_submission.csv")

train_tfrecord_dir = os.path.join(DATA_DIR, "train_tfrecords")
test_tfrecord_dir = os.path.join(DATA_DIR, "test_tfrecords")

train_df = pd.read_csv(train_csv_path)
sample_sub = pd.read_csv(sample_path)

assert set(train_df.columns) >= {"image_id", "label"}
assert set(sample_sub.columns) >= {"image_id", "label"}

val_frac = 0.1
GLOBAL_SEED = 42

train_df["label"] = train_df["label"].astype(str)

rng = np.random.RandomState(GLOBAL_SEED)
val_indices = []
for lbl, idx in train_df.groupby("label").indices.items():
    idx = np.asarray(idx, dtype=np.int64)
    rng.shuffle(idx)
    n_val = int(np.floor(len(idx) * val_frac))
    if n_val > 0:
        val_indices.append(idx[:n_val])
val_idx = (
    np.sort(np.concatenate(val_indices))
    if val_indices
    else np.array([], dtype=np.int64)
)

val_df = train_df.iloc[val_idx].reset_index(drop=True)
trn_df = train_df.drop(train_df.index[val_idx]).reset_index(drop=True)

_has_gpu = len(tf.config.list_physical_devices("GPU")) > 0
batch_size = 32 if _has_gpu else 16


@tf.function
def _decode_resize_from_bytes(img_bytes):
    img = tf.image.decode_jpeg(img_bytes, channels=3)  # dataset are jpgs
    img = tf.image.resize(
        img, [IMG_SIZE, IMG_SIZE], method=tf.image.ResizeMethod.BILINEAR, antialias=True
    )
    img = tf.cast(img, tf.float32)  # keep float32
    return img


@tf.function
def _decode_resize(path):
    img_bytes = tf.io.read_file(path)
    return _decode_resize_from_bytes(img_bytes)


@tf.function
def _augment(img, seed):
    seed = tf.convert_to_tensor(seed, dtype=tf.int32)
    img = tf.image.stateless_random_flip_left_right(
        img, seed=seed + tf.constant([1, 0], tf.int32)
    )

    angle = tf.random.stateless_uniform(
        [], seed=seed + tf.constant([2, 0], tf.int32), minval=-10.0, maxval=10.0
    ) * (np.pi / 180.0)

    dx = tf.random.stateless_uniform(
        [], seed=seed + tf.constant([3, 0], tf.int32), minval=-0.05, maxval=0.05
    )
    dy = tf.random.stateless_uniform(
        [], seed=seed + tf.constant([4, 0], tf.int32), minval=-0.05, maxval=0.05
    )

    zoom = tf.random.stateless_uniform(
        [], seed=seed + tf.constant([5, 0], tf.int32), minval=0.9, maxval=1.1
    )

    cx = (IMG_SIZE - 1) / 2.0
    cy = (IMG_SIZE - 1) / 2.0
    cos_a = tf.cos(angle)
    sin_a = tf.sin(angle)

    inv_zoom = 1.0 / zoom

    tx = dx * tf.cast(IMG_SIZE, tf.float32)
    ty = dy * tf.cast(IMG_SIZE, tf.float32)

    a0 = inv_zoom * cos_a
    a1 = inv_zoom * sin_a
    a3 = -inv_zoom * sin_a
    a4 = inv_zoom * cos_a

    a2 = cx - a0 * cx - a1 * cy + tx
    a5 = cy - a3 * cx - a4 * cy + ty

    transform = tf.stack([a0, a1, a2, a3, a4, a5, 0.0, 0.0])[tf.newaxis, :]
    img = tf.raw_ops.ImageProjectiveTransformV3(
        images=img[tf.newaxis, ...],
        transforms=transform,
        output_shape=tf.constant([IMG_SIZE, IMG_SIZE], tf.int32),
        interpolation="BILINEAR",
        fill_mode="REFLECT",
        fill_value=0.0,
    )[0]
    return img


_DATA_OPTIONS = tf.data.Options()
_DATA_OPTIONS.experimental_deterministic = False
try:
    _DATA_OPTIONS.experimental_optimization.apply_default_optimizations = True
    _DATA_OPTIONS.experimental_optimization.map_parallelization = True
    _DATA_OPTIONS.experimental_optimization.map_fusion = True
    _DATA_OPTIONS.experimental_slack = True
except Exception:
    pass

_SHUFFLE_BUFFER = min(4096, len(trn_df))

_TRAIN_FEATURES = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),
}
_TEST_FEATURES = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
}


def _list_tfrec_files(tfrec_dir, prefix):
    pattern = os.path.join(tfrec_dir, f"{prefix}*.tfrec")
    files = tf.io.gfile.glob(pattern)
    files = sorted(files)
    if not files:
        raise FileNotFoundError(f"No TFRecord files found: {pattern}")
    return files


def _parse_train_example(example_proto):
    ex = tf.io.parse_single_example(example_proto, _TRAIN_FEATURES)
    img = _decode_resize_from_bytes(ex["image"])
    y = tf.cast(ex["target"], tf.int32)
    return img, y


def _parse_test_example(example_proto):
    ex = tf.io.parse_single_example(example_proto, _TEST_FEATURES)
    img = _decode_resize_from_bytes(ex["image"])
    name = ex["image_name"]
    return img, name


def _disk_cache_path(name: str) -> str:
    return os.path.join("/kaggle/working", f"tfdata_cache_{name}")


def _make_seed_stream(global_seed: int):
    base = tf.constant([global_seed, 0], dtype=tf.int64)
    ctr = tf.data.Dataset.counter()  # int64, increasing

    def _mk(i):
        i = tf.cast(i, tf.int64)
        s0 = base[0] ^ (i * tf.constant(1103515245, tf.int64))
        s1 = base[1] ^ (i * tf.constant(12345, tf.int64))
        return tf.cast(tf.stack([s0, s1]), tf.int32)

    return ctr.map(_mk, num_parallel_calls=AUTOTUNE)


def _make_test_ds_from_tfrecords(tfrec_files, batch_size):
    ds = tf.data.TFRecordDataset(
        tfrec_files, num_parallel_reads=AUTOTUNE, compression_type=None
    )
    ds = ds.with_options(_DATA_OPTIONS)
    ds = ds.map(_parse_test_example, num_parallel_calls=AUTOTUNE, deterministic=False)
    ds = ds.map(lambda img, name: img, num_parallel_calls=AUTOTUNE, deterministic=False)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_tfrec_files = _list_tfrec_files(train_tfrecord_dir, "ld_train")
test_tfrec_files = _list_tfrec_files(test_tfrecord_dir, "ld_test")

n_train = len(trn_df)
n_val = len(val_df)

decoded_all = (
    tf.data.TFRecordDataset(train_tfrec_files, num_parallel_reads=AUTOTUNE)
    .with_options(_DATA_OPTIONS)
    .map(_parse_train_example, num_parallel_calls=AUTOTUNE, deterministic=False)
    .cache(_disk_cache_path("decoded_all_448"))
)

val_unbatched = decoded_all.take(n_val).cache(
    _disk_cache_path("val_decoded_448_unbatched")
)
train_unbatched = (
    decoded_all.skip(n_val)
    .take(n_train)
    .cache(_disk_cache_path("train_decoded_448_unbatched"))
)


def _to_oh(img, y):
    y_oh = tf.one_hot(tf.cast(y, tf.int32), NUM_CLASSES, dtype=tf.float32)
    return img, y_oh


val_ds_v4 = (
    val_unbatched.map(_to_oh, num_parallel_calls=AUTOTUNE, deterministic=False)
    .batch(batch_size, drop_remainder=False)
    .prefetch(AUTOTUNE)
)

train_unbatched = train_unbatched.shuffle(
    buffer_size=_SHUFFLE_BUFFER, seed=GLOBAL_SEED, reshuffle_each_iteration=True
)

seed_ds = _make_seed_stream(GLOBAL_SEED)
train_unbatched = tf.data.Dataset.zip((train_unbatched, seed_ds))


def _aug_map(data, seed2):
    img, y = data
    img = _augment(img, seed2)
    y_oh = tf.one_hot(tf.cast(y, tf.int32), NUM_CLASSES, dtype=tf.float32)
    return img, y_oh


train_ds_v4 = (
    train_unbatched.map(_aug_map, num_parallel_calls=AUTOTUNE, deterministic=False)
    .batch(batch_size, drop_remainder=False)
    .prefetch(AUTOTUNE)
)

test_v4 = sample_sub[["image_id"]].copy()
test_batch_size = 128 if _has_gpu else 64
test_ds_v4 = _make_test_ds_from_tfrecords(test_tfrec_files, batch_size=test_batch_size)

print("Train/Val sizes:", len(trn_df), len(val_df))
print("Batch size:", batch_size, "| Test batch size:", test_batch_size)
print("Test size:", len(test_v4))



## === cell 3
epochs = 2

history = model_v4.fit(
    train_ds_v4,
    validation_data=val_ds_v4,
    epochs=epochs,
    verbose=1,
)



## === cell 4
pred_v4 = model_v4.predict(test_ds_v4, verbose=1)
pred_v4 = np.asarray(pred_v4)

print("Raw pred shape:", pred_v4.shape, "dtype:", pred_v4.dtype)

if pred_v4.ndim == 2 and pred_v4.shape[1] >= 5:
    predicted_class_indices_v4 = np.argmax(pred_v4[:, :5], axis=1)
elif pred_v4.ndim == 1:
    predicted_class_indices_v4 = pred_v4.astype(int)
else:
    predicted_class_indices_v4 = np.argmax(pred_v4, axis=-1)

predicted_class_indices_v4 = predicted_class_indices_v4.astype(int)

results_v4 = pd.DataFrame(
    {"image_id": test_v4["image_id"].values, "label": predicted_class_indices_v4}
)

assert (
    results_v4.shape[0] == sample_sub.shape[0]
), "Submission row count mismatch vs sample_submission"
assert list(results_v4.columns) == ["image_id", "label"], "Wrong submission columns"
assert results_v4["label"].between(0, 4).all(), "Labels out of expected range 0..4"

out_path = "/kaggle/working/submission.csv"
results_v4.to_csv(out_path, index=False)
print(f"Wrote {out_path}")
print(results_v4.head())
