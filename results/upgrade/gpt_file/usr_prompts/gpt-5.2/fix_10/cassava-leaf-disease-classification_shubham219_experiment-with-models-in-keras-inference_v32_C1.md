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

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)

import glob
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.applications import EfficientNetB3

SEED = 42
DEBUG = False

tf.random.set_seed(SEED)
np.random.seed(SEED)
try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

DATA_DIR = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")
SAMPLE_SUB = os.path.join(DATA_DIR, "sample_submission.csv")

assert os.path.exists(TRAIN_CSV), f"Missing train.csv at {TRAIN_CSV}"
assert os.path.exists(TRAIN_IMG_DIR), f"Missing train_images at {TRAIN_IMG_DIR}"
assert os.path.exists(TEST_IMG_DIR), f"Missing test_images at {TEST_IMG_DIR}"
assert os.path.exists(SAMPLE_SUB), f"Missing sample_submission.csv at {SAMPLE_SUB}"

print("TF version:", tf.__version__)
print("Data dir:", DATA_DIR)



## === cell 1
IMG_SIZE = (300, 300)  # kept consistent with original target_size
BATCH_SIZE = 32
EPOCHS = 3  # keep identical fixed training budget

train_df = pd.read_csv(TRAIN_CSV)

train_df["path"] = TRAIN_IMG_DIR + "/" + train_df["image_id"].astype(str)
train_df["label"] = train_df["label"].astype(np.int32)

idx = np.arange(len(train_df))
rng = np.random.RandomState(SEED)
rng.shuffle(idx)
val_size = int(0.1 * len(idx))
val_idx = idx[:val_size]
trn_idx = idx[val_size:]

df_train = train_df.iloc[trn_idx].reset_index(drop=True)
df_val = train_df.iloc[val_idx].reset_index(drop=True)

AUTO = tf.data.AUTOTUNE
IMG_H, IMG_W = IMG_SIZE


@tf.function
def _decode_resize_rescale(path):
    img = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img, channels=3, dct_method="INTEGER_FAST")
    img = tf.image.resize(img, [IMG_H, IMG_W], method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32) / 255.0
    return img


@tf.function
def _stateless_seed_from_index(i):
    i = tf.cast(i, tf.int32)
    return tf.stack([tf.constant(SEED, tf.int32), i], axis=0)


@tf.function
def _random_affine_equivalent(img, seed2):
    img = tf.image.stateless_random_flip_left_right(img, seed=seed2)

    max_dx = tf.cast(tf.round(0.05 * tf.cast(IMG_W, tf.float32)), tf.int32)
    max_dy = tf.cast(tf.round(0.05 * tf.cast(IMG_H, tf.float32)), tf.int32)
    dx = tf.random.stateless_uniform(
        [],
        seed=seed2 + tf.constant([1, 0], tf.int32),
        minval=-max_dx,
        maxval=max_dx + 1,
        dtype=tf.int32,
    )
    dy = tf.random.stateless_uniform(
        [],
        seed=seed2 + tf.constant([2, 0], tf.int32),
        minval=-max_dy,
        maxval=max_dy + 1,
        dtype=tf.int32,
    )
    img = tf.roll(img, shift=[dy, dx], axis=[0, 1])

    angle = tf.random.stateless_uniform(
        [],
        seed=seed2 + tf.constant([3, 0], tf.int32),
        minval=-15.0 * np.pi / 180.0,
        maxval=15.0 * np.pi / 180.0,
        dtype=tf.float32,
    )
    cos_a = tf.math.cos(angle)
    sin_a = tf.math.sin(angle)

    scale = tf.random.stateless_uniform(
        [],
        seed=seed2 + tf.constant([4, 0], tf.int32),
        minval=0.9,
        maxval=1.1,
        dtype=tf.float32,
    )

    cx = (tf.cast(IMG_W, tf.float32) - 1.0) / 2.0
    cy = (tf.cast(IMG_H, tf.float32) - 1.0) / 2.0

    a0 = scale * cos_a
    a1 = -scale * sin_a
    b0 = scale * sin_a
    b1 = scale * cos_a

    a2 = cx - a0 * cx - a1 * cy
    b2 = cy - b0 * cx - b1 * cy
    transform = tf.stack([a0, a1, a2, b0, b1, b2, 0.0, 0.0], axis=0)

    img = tf.raw_ops.ImageProjectiveTransformV3(
        images=tf.expand_dims(img, axis=0),
        transforms=tf.expand_dims(transform, axis=0),
        output_shape=tf.constant([IMG_H, IMG_W], dtype=tf.int32),
        interpolation="BILINEAR",
        fill_mode="REFLECT",
        fill_value=0.0,
    )
    img = tf.squeeze(img, axis=0)
    return img


@tf.function
def _train_map_enum(i, data):
    path, label = data
    img = _decode_resize_rescale(path)
    seed2 = _stateless_seed_from_index(i)
    img = _random_affine_equivalent(img, seed2)
    return img, label


@tf.function
def _val_map(path, label):
    img = _decode_resize_rescale(path)
    return img, label


_TFDATA_OPTIONS = tf.data.Options()
try:
    _TFDATA_OPTIONS.experimental_optimization.apply_default_optimizations = True
except Exception:
    pass
try:
    _TFDATA_OPTIONS.experimental_deterministic = True
except Exception:
    pass

TRAIN_CACHE = os.path.join(
    "/kaggle/working", f"train_cache_seed{SEED}_sz{IMG_H}x{IMG_W}.tfrecord"
)
VAL_CACHE = os.path.join(
    "/kaggle/working", f"val_cache_seed{SEED}_sz{IMG_H}x{IMG_W}.tfrecord"
)


def _bytes_feature(v):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[v]))


def _int64_feature(v):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[int(v)]))


def _serialize_example(img, label):
    img_bytes = tf.io.serialize_tensor(img).numpy()
    ex = tf.train.Example(
        features=tf.train.Features(
            feature={
                "img": _bytes_feature(img_bytes),
                "label": _int64_feature(label),
            }
        )
    )
    return ex.SerializeToString()


def _write_cache_tfrecord(df, out_path, is_train):
    paths = df["path"].to_numpy(dtype=str)
    labels = df["label"].to_numpy(dtype=np.int32)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if is_train:
        shuffle_buf = min(len(df), 4096)
        ds = ds.shuffle(
            buffer_size=shuffle_buf, seed=SEED, reshuffle_each_iteration=True
        )
        ds = ds.enumerate()
        ds = ds.map(_train_map_enum, num_parallel_calls=AUTO, deterministic=True)
    else:
        ds = ds.map(_val_map, num_parallel_calls=AUTO, deterministic=True)

    with tf.io.TFRecordWriter(out_path) as w:
        for item in ds:
            img, label = item
            w.write(_serialize_example(img, label.numpy()))
    return out_path


def _parse_cached_example(example_proto):
    feat = {
        "img": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([], tf.int64),
    }
    x = tf.io.parse_single_example(example_proto, feat)
    img = tf.io.parse_tensor(x["img"], out_type=tf.float32)
    img = tf.ensure_shape(img, [IMG_H, IMG_W, 3])
    label = tf.cast(x["label"], tf.int32)
    return img, label


def make_train_ds(df, batch_size):
    if not tf.io.gfile.exists(TRAIN_CACHE):
        print("Building train TFRecord cache at:", TRAIN_CACHE)
        _write_cache_tfrecord(df, TRAIN_CACHE, is_train=True)
    else:
        print("Using existing train TFRecord cache:", TRAIN_CACHE)

    ds = tf.data.TFRecordDataset(TRAIN_CACHE, num_parallel_reads=AUTO)
    ds = ds.map(_parse_cached_example, num_parallel_calls=AUTO, deterministic=True)
    ds = ds.with_options(_TFDATA_OPTIONS)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTO)
    return ds


def make_val_ds(df, batch_size):
    if not tf.io.gfile.exists(VAL_CACHE):
        print("Building val TFRecord cache at:", VAL_CACHE)
        _write_cache_tfrecord(df, VAL_CACHE, is_train=False)
    else:
        print("Using existing val TFRecord cache:", VAL_CACHE)

    ds = tf.data.TFRecordDataset(VAL_CACHE, num_parallel_reads=AUTO)
    ds = ds.map(_parse_cached_example, num_parallel_calls=AUTO, deterministic=True)
    ds = ds.with_options(_TFDATA_OPTIONS)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTO)
    return ds


train_ds = make_train_ds(df_train, BATCH_SIZE)
val_ds = make_val_ds(df_val, BATCH_SIZE)

inputs = Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
base = EfficientNetB3(include_top=False, weights="imagenet", input_tensor=inputs)
x = GlobalAveragePooling2D()(base.output)
x = Dropout(0.2, seed=SEED)(x)
outputs = Dense(5, activation="softmax")(x)
my_model = Model(inputs=inputs, outputs=outputs)

base.trainable = False
my_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

steps_per_epoch = max(1, len(df_train) // BATCH_SIZE)
val_steps = max(1, len(df_val) // BATCH_SIZE)

my_model.fit(
    train_ds,
    epochs=1,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_ds,
    validation_steps=val_steps,
    verbose=1,
)

base.trainable = True
my_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

my_model.fit(
    train_ds,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_ds,
    validation_steps=val_steps,
    verbose=1,
)



## === cell 2
test_images = glob.glob(os.path.join(TEST_IMG_DIR, "*.jpg"))
test_images.sort()  # deterministic order

test_paths = np.asarray(test_images, dtype=str)
test_image_ids = np.asarray([os.path.basename(p) for p in test_paths], dtype=object)


@tf.function
def _test_map(path):
    img = _decode_resize_rescale(path)
    return img


def make_test_ds(paths, batch_size=128):
    ds = tf.data.Dataset.from_tensor_slices(paths)
    ds = ds.map(_test_map, num_parallel_calls=AUTO, deterministic=True)
    ds = ds.cache()
    ds = ds.with_options(_TFDATA_OPTIONS)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTO)
    return ds


test_ds = make_test_ds(test_paths, batch_size=128)

pred_test = my_model.predict(test_ds, verbose=1)
pred_test_labels = np.argmax(pred_test, axis=-1).astype(int)

final_submission = pd.DataFrame({"image_id": test_image_ids, "label": pred_test_labels})

sample = pd.read_csv(SAMPLE_SUB, usecols=["image_id"])
label_map = dict(
    zip(final_submission["image_id"].to_numpy(), final_submission["label"].to_numpy())
)
final_csv = sample.copy()
final_csv["label"] = final_csv["image_id"].map(label_map).fillna(0).astype(int)

final_csv.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", final_csv.shape)
print(final_csv.head())



## === cell 3
final_csv.head()
