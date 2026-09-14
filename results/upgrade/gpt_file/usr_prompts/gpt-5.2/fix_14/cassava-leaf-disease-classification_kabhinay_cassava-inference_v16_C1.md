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

2.7

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

import tensorflow as tf

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
try:
    tf.random.set_seed(SEED)
except Exception:
    pass

print("TF version:", tf.__version__)



## === cell 1
pass



## === cell 2
DATA_DIR = "/kaggle/input/cassava-leaf-disease-classification"
if not os.path.exists(DATA_DIR):
    DATA_DIR = "/kaggle/data/cassava-leaf-disease-classification"

train_csv = os.path.join(DATA_DIR, "train.csv")
sample_path = os.path.join(DATA_DIR, "sample_submission.csv")
train_dir = os.path.join(DATA_DIR, "train_images")
test_dir = os.path.join(DATA_DIR, "test_images")

assert os.path.exists(train_csv), train_csv
assert os.path.exists(sample_path), sample_path
assert os.path.isdir(train_dir), train_dir
assert os.path.isdir(test_dir), test_dir

train_df = pd.read_csv(train_csv)
sample_df = pd.read_csv(sample_path)
test_df = sample_df[["image_id"]].copy()

print("train_df:", train_df.shape, "test_df:", test_df.shape)



## === cell 3
IMG_SIZE = (448, 448)
BATCH_SIZE = 8

train_df_shuf = train_df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
val_frac = 0.1
val_size = int(len(train_df_shuf) * val_frac)
val_df = train_df_shuf.iloc[:val_size].copy()
trn_df = train_df_shuf.iloc[val_size:].copy()

AUTOTUNE = tf.data.AUTOTUNE
IMG_H, IMG_W = IMG_SIZE

num_classes = int(train_df["label"].nunique())
print("num_classes:", num_classes)

tfrecord_train_dir = os.path.join(DATA_DIR, "train_tfrecords")
tfrecord_test_dir = os.path.join(DATA_DIR, "test_tfrecords")


def _list_tfrec_files(directory):
    if not tf.io.gfile.exists(directory):
        return []
    files = tf.io.gfile.glob(os.path.join(directory, "*.tfrec"))
    return sorted(files)


train_tfrec_files_all = _list_tfrec_files(tfrecord_train_dir)
test_tfrec_files = _list_tfrec_files(tfrecord_test_dir)


def _split_train_tfrec_files(files, val_frac):
    def _extract_shard_idx(path):
        base = os.path.basename(path)
        try:
            s = base.split("ld_train", 1)[1].split("-", 1)[0]
            return int(s)
        except Exception:
            return None

    parsed = []
    for f in files:
        idx = _extract_shard_idx(f)
        parsed.append((f, idx))

    if any(idx is None for _, idx in parsed) or len(parsed) == 0:
        return files, []

    parsed.sort(key=lambda x: x[1])
    n = len(parsed)
    n_val = int(round(n * val_frac))
    if n_val <= 0:
        return [p[0] for p in parsed], []
    val_files = [p[0] for p in parsed[:n_val]]
    trn_files = [p[0] for p in parsed[n_val:]]
    return trn_files, val_files


train_tfrec_files, val_tfrec_files = _split_train_tfrec_files(
    train_tfrec_files_all, val_frac
)

_use_tfrec = (len(train_tfrec_files_all) > 0) and (len(test_tfrec_files) > 0)
print(
    "Using TFRecords:",
    _use_tfrec,
    "n_train_tfrec_all:",
    len(train_tfrec_files_all),
    "n_train_tfrec_used:",
    len(train_tfrec_files),
    "n_val_tfrec_used:",
    len(val_tfrec_files),
    "n_test_tfrec:",
    len(test_tfrec_files),
)

_tfrec_feature_desc = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),
}

_PI_OVER_180 = tf.constant(np.pi / 180.0, dtype=tf.float32)
_SEED_CONST = tf.constant(SEED, dtype=tf.int32)
_ONE = tf.constant([1, 1], tf.int32)
_TWO = tf.constant([2, 2], tf.int32)
_THREE = tf.constant([3, 3], tf.int32)


@tf.function
def _parse_tfrec(example_proto):
    ex = tf.io.parse_single_example(example_proto, _tfrec_feature_desc)
    img = tf.image.decode_jpeg(ex["image"], channels=3)
    img = tf.image.resize(img, [IMG_H, IMG_W], method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32) / 255.0
    image_id = ex["image_name"]
    label = ex["target"]
    return image_id, img, label


@tf.function
def _to_onehot(label):
    return tf.one_hot(tf.cast(label, tf.int32), depth=num_classes, dtype=tf.float32)


def tfa_image_rotate(image, angle):
    angle = tf.cast(angle, tf.float32)
    cos_a = tf.cos(angle)
    sin_a = tf.sin(angle)
    cx = (tf.cast(IMG_W, tf.float32) - 1.0) / 2.0
    cy = (tf.cast(IMG_H, tf.float32) - 1.0) / 2.0
    a0 = cos_a
    a1 = -sin_a
    b0 = sin_a
    b1 = cos_a
    a2 = cx - a0 * cx - a1 * cy
    b2 = cy - b0 * cx - b1 * cy
    transform = tf.stack([a0, a1, a2, b0, b1, b2, 0.0, 0.0])
    transform = tf.reshape(transform, [1, 8])
    image4 = tf.expand_dims(image, 0)
    out = tf.raw_ops.ImageProjectiveTransformV3(
        images=image4,
        transforms=transform,
        output_shape=tf.constant([IMG_H, IMG_W], dtype=tf.int32),
        interpolation="BILINEAR",
        fill_mode="REFLECT",
        fill_value=0.0,
    )
    return tf.squeeze(out, 0)


@tf.function
def _augment_stateless(img, seed2):
    seed2 = tf.cast(seed2, tf.int32)

    img = tf.image.stateless_random_flip_left_right(img, seed=seed2)

    shift_x = tf.cast(tf.round(0.05 * tf.cast(IMG_W, tf.float32)), tf.int32)
    shift_y = tf.cast(tf.round(0.05 * tf.cast(IMG_H, tf.float32)), tf.int32)
    pad = tf.maximum(shift_x, shift_y)
    img = tf.image.pad_to_bounding_box(img, pad, pad, IMG_H + 2 * pad, IMG_W + 2 * pad)
    img = tf.image.stateless_random_crop(img, size=[IMG_H, IMG_W, 3], seed=seed2 + _ONE)

    scale = tf.random.stateless_uniform([], seed=seed2 + _TWO, minval=0.9, maxval=1.1)
    new_h = tf.cast(tf.round(scale * tf.cast(IMG_H, tf.float32)), tf.int32)
    new_w = tf.cast(tf.round(scale * tf.cast(IMG_W, tf.float32)), tf.int32)
    img2 = tf.image.resize(img, [new_h, new_w], method=tf.image.ResizeMethod.BILINEAR)
    img2 = tf.image.resize_with_crop_or_pad(img2, IMG_H, IMG_W)

    angle = (
        tf.random.stateless_uniform([], seed=seed2 + _THREE, minval=-15.0, maxval=15.0)
        * _PI_OVER_180
    )
    img2 = tfa_image_rotate(img2, angle)
    return img2


def _dataset_options():
    opts = tf.data.Options()
    opts.experimental_deterministic = False
    try:
        opts.experimental_optimization.apply_default_optimizations = True
        opts.experimental_optimization.map_parallelization = True
        opts.experimental_optimization.parallel_batch = True
        opts.experimental_optimization.autotune_buffers = True
    except Exception:
        pass
    return opts


@tf.function
def _seed_from_image_id(image_id):
    h = tf.strings.to_hash_bucket_fast(image_id, 2**31 - 1)
    return tf.stack([_SEED_CONST, tf.cast(h, tf.int32)])


def _make_base_ds_from_files(files, shuffle_files, shuffle_records, seed):
    file_ds = tf.data.Dataset.from_tensor_slices(files)
    if shuffle_files:
        file_ds = file_ds.shuffle(len(files), seed=seed, reshuffle_each_iteration=True)
    ds = file_ds.interleave(
        lambda f: tf.data.TFRecordDataset(f),
        cycle_length=AUTOTUNE,
        num_parallel_calls=AUTOTUNE,
        deterministic=False,
    )
    ds = ds.with_options(_dataset_options())
    ds = ds.apply(tf.data.experimental.ignore_errors())
    if shuffle_records:
        ds = ds.shuffle(buffer_size=8192, seed=seed, reshuffle_each_iteration=True)
    return ds


def make_train_ds():
    if not _use_tfrec:
        raise RuntimeError("TFRecords not found; expected in this dataset layout.")
    files = train_tfrec_files if len(train_tfrec_files) > 0 else train_tfrec_files_all

    ds = _make_base_ds_from_files(
        files=files, shuffle_files=True, shuffle_records=True, seed=SEED
    )

    ds = ds.map(_parse_tfrec, num_parallel_calls=AUTOTUNE)
    ds = ds.cache()

    @tf.function
    def _map_train(image_id, img, label):
        seed2 = _seed_from_image_id(image_id)
        img = _augment_stateless(img, seed2)
        return img, _to_onehot(label)

    ds = ds.map(_map_train, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def make_val_ds():
    if not _use_tfrec:
        raise RuntimeError("TFRecords not found; expected in this dataset layout.")
    files = val_tfrec_files
    if len(files) == 0:
        val_ids = val_df["image_id"].values.astype(np.str_)
        val_keys = tf.constant(val_ids)
        val_vals = tf.ones([val_keys.shape[0]], dtype=tf.bool)
        val_table_local = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(val_keys, val_vals),
            default_value=False,
        )

        ds = _make_base_ds_from_files(
            files=train_tfrec_files_all,
            shuffle_files=False,
            shuffle_records=False,
            seed=SEED,
        )
        ds = ds.map(_parse_tfrec, num_parallel_calls=AUTOTUNE)
        ds = ds.filter(lambda image_id, img, label: val_table_local.lookup(image_id))
    else:
        ds = _make_base_ds_from_files(
            files=files, shuffle_files=False, shuffle_records=False, seed=SEED
        )
        ds = ds.map(_parse_tfrec, num_parallel_calls=AUTOTUNE)

    ds = ds.map(
        lambda image_id, img, label: (img, _to_onehot(label)),
        num_parallel_calls=AUTOTUNE,
    )
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.cache()
    ds = ds.prefetch(AUTOTUNE)
    return ds


def make_test_ds():
    if not _use_tfrec:
        raise RuntimeError("TFRecords not found; expected in this dataset layout.")
    ds = _make_base_ds_from_files(
        files=test_tfrec_files, shuffle_files=False, shuffle_records=False, seed=SEED
    )

    ds = ds.map(_parse_tfrec, num_parallel_calls=AUTOTUNE)
    ds = ds.map(lambda image_id, img, label: img, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_ds = make_train_ds()
val_ds = make_val_ds()
test_ds = make_test_ds()

steps_per_epoch = None
val_steps = None

class_indices = {str(i): i for i in range(num_classes)}
print("class_indices:", class_indices)



## === cell 4
inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu")(inputs)
x = tf.keras.layers.MaxPooling2D()(x)
x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
x = tf.keras.layers.MaxPooling2D()(x)
x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
x = tf.keras.layers.MaxPooling2D()(x)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

EPOCHS = 3

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_steps=val_steps,
    verbose=1,
)



## === cell 5
pred = model.predict(test_ds, verbose=1)

n_test = test_df.shape[0]
pred = pred[:n_test]

idx_to_label_str = {v: k for k, v in class_indices.items()}
pred_idx = np.argmax(pred, axis=1).astype(np.int64)

lut = np.empty((num_classes,), dtype=np.int64)
for v, k in idx_to_label_str.items():
    lut[int(v)] = int(k)
pred_labels = lut[pred_idx]

submission = pd.DataFrame(
    {"image_id": test_df["image_id"].values, "label": pred_labels}
)

assert submission.shape[0] == sample_df.shape[0]
assert list(submission.columns) == ["image_id", "label"]

out_path = "/kaggle/working/submission.csv"
submission.to_csv(out_path, index=False)
print(submission.head())
print("Wrote", out_path, "with shape:", submission.shape)
