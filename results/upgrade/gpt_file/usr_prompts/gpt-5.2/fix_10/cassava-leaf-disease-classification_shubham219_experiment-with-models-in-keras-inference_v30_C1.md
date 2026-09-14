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

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import glob
import pandas as pd
import numpy as np

import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.applications import EfficientNetB3

SEED = 42
DEBUG = False

tf.keras.utils.set_random_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass
try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

CANDIDATE_BASES = [
    "/kaggle/input/cassava-leaf-disease-classification",
    "../input/cassava-leaf-disease-classification",
    "/kaggle/data/cassava-leaf-disease-classification",
    "/kaggle/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
]
BASE_PATH = None
for p in CANDIDATE_BASES:
    if tf.io.gfile.exists(p):
        BASE_PATH = p
        break
if BASE_PATH is None:
    raise FileNotFoundError(
        "Could not find cassava-leaf-disease-classification dataset folder in expected locations."
    )

TRAIN_CSV_PATH = os.path.join(BASE_PATH, "train.csv")
TRAIN_IMG_DIR = os.path.join(BASE_PATH, "train_images")
TEST_IMG_DIR = os.path.join(BASE_PATH, "test_images")
SAMPLE_SUB_PATH = os.path.join(BASE_PATH, "sample_submission.csv")
TRAIN_TFREC_DIR = os.path.join(BASE_PATH, "train_tfrecords")
TEST_TFREC_DIR = os.path.join(BASE_PATH, "test_tfrecords")

print("Using BASE_PATH:", BASE_PATH)
print("TensorFlow:", tf.__version__)
print("Train CSV exists:", tf.io.gfile.exists(TRAIN_CSV_PATH))
print("Train images dir exists:", tf.io.gfile.exists(TRAIN_IMG_DIR))
print("Test images dir exists:", tf.io.gfile.exists(TEST_IMG_DIR))
print("Train TFRecords dir exists:", tf.io.gfile.exists(TRAIN_TFREC_DIR))
print("Test TFRecords dir exists:", tf.io.gfile.exists(TEST_TFREC_DIR))

options = tf.data.Options()
options.experimental_optimization.apply_default_optimizations = True
options.experimental_optimization.autotune_buffers = True
options.experimental_optimization.map_parallelization = True
options.experimental_optimization.parallel_batch = True
options.deterministic = True




## === cell 1
IMG_SIZE = (300, 300)
N_CLASSES = 5

base = EfficientNetB3(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    pooling="avg",
)
x = tf.keras.layers.Dropout(0.2)(base.output)
out = tf.keras.layers.Dense(N_CLASSES, activation="softmax")(x)
my_model = Model(inputs=base.input, outputs=out)

my_model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)




## === cell 2
train_df = pd.read_csv(TRAIN_CSV_PATH)
train_df["path"] = train_df["image_id"].map(lambda x: os.path.join(TRAIN_IMG_DIR, x))

preprocess_fn = tf.keras.applications.efficientnet.preprocess_input

BATCH_SIZE = 32
EPOCHS = 3  # keep as-is (core logic)
AUTOTUNE = tf.data.AUTOTUNE

paths = train_df["path"].values
labels = train_df["label"].values.astype(np.int32)

rng = np.random.RandomState(SEED)
idx = np.arange(len(train_df))
rng.shuffle(idx)
val_size = int(round(len(idx) * 0.1))
val_idx = idx[-val_size:]
train_idx = idx[:-val_size]

train_paths, train_labels = paths[train_idx], labels[train_idx]
val_paths, val_labels = paths[val_idx], labels[val_idx]

use_tfrecords = (
    tf.io.gfile.exists(TRAIN_TFREC_DIR)
    and len(tf.io.gfile.glob(os.path.join(TRAIN_TFREC_DIR, "*.tfrec"))) > 0
)
train_tfrec_files = (
    sorted(tf.io.gfile.glob(os.path.join(TRAIN_TFREC_DIR, "*.tfrec")))
    if use_tfrecords
    else []
)
if use_tfrecords:
    print("Using TFRecords for training:", len(train_tfrec_files), "files")
else:
    print("TFRecords not found; falling back to JPEG paths pipeline.")


def _stateless_augment(img, seed):
    img = tf.image.stateless_random_flip_left_right(img, seed=seed)

    tx = tf.random.stateless_uniform(
        [], seed=seed + tf.constant([2, 0], tf.int32), minval=-0.05, maxval=0.05
    )
    ty = tf.random.stateless_uniform(
        [], seed=seed + tf.constant([3, 0], tf.int32), minval=-0.05, maxval=0.05
    )
    img = tf.raw_ops.ImageProjectiveTransformV3(
        images=tf.expand_dims(img, 0),
        transforms=tf.reshape(
            tf.stack(
                [
                    1.0,
                    0.0,
                    -tx * tf.cast(IMG_SIZE[1], tf.float32),
                    0.0,
                    1.0,
                    -ty * tf.cast(IMG_SIZE[0], tf.float32),
                    0.0,
                    0.0,
                ]
            ),
            [1, 8],
        ),
        output_shape=tf.constant([IMG_SIZE[0], IMG_SIZE[1]], dtype=tf.int32),
        interpolation="BILINEAR",
        fill_mode="REFLECT",
        fill_value=0.0,
    )
    img = tf.squeeze(img, 0)

    z = tf.random.stateless_uniform(
        [], seed=seed + tf.constant([4, 0], tf.int32), minval=-0.1, maxval=0.1
    )
    h = tf.cast(IMG_SIZE[0], tf.float32)
    w = tf.cast(IMG_SIZE[1], tf.float32)
    new_h = tf.cast(tf.round(h * (1.0 - z)), tf.int32)
    new_w = tf.cast(tf.round(w * (1.0 - z)), tf.int32)
    img = tf.image.resize_with_crop_or_pad(img, new_h, new_w)
    img = tf.image.resize(img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR)

    return img


_TFREC_FEATURES = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),
}


def _parse_tfrec(example_proto):
    ex = tf.io.parse_single_example(example_proto, _TFREC_FEATURES)
    img = tf.image.decode_jpeg(ex["image"], channels=3)
    img = tf.image.resize(img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32)
    label = tf.cast(ex["target"], tf.int32)
    return img, label


def _decode_resize_only(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32)
    return img, label


def _train_map(img, label, idx_in_epoch):
    seed = tf.stack([tf.cast(SEED, tf.int32), tf.cast(idx_in_epoch, tf.int32)], axis=0)
    img = _stateless_augment(img, seed)
    img = preprocess_fn(img)
    return img, label


def _val_map(img, label):
    img = preprocess_fn(img)
    return img, label


SHUFFLE_BUFFER = 4096

if use_tfrecords:
    raw = tf.data.TFRecordDataset(
        train_tfrec_files,
        num_parallel_reads=AUTOTUNE,
        compression_type=None,
    ).with_options(options)
    raw = raw.map(_parse_tfrec, num_parallel_calls=AUTOTUNE)

    n_total = len(train_df)
    is_val = np.zeros(n_total, dtype=np.bool_)
    is_val[val_idx] = True
    is_val_tf = tf.constant(is_val)

    def _keep_train(i, img_label):
        return tf.logical_not(tf.gather(is_val_tf, i))

    def _keep_val(i, img_label):
        return tf.gather(is_val_tf, i)

    def _drop_i(i, img_label):
        return img_label

    raw = raw.shuffle(
        buffer_size=min(SHUFFLE_BUFFER, n_total),
        seed=SEED,
        reshuffle_each_iteration=True,
    )

    enum = raw.enumerate()

    train_base = enum.filter(_keep_train).map(_drop_i, num_parallel_calls=AUTOTUNE)
    val_base = enum.filter(_keep_val).map(_drop_i, num_parallel_calls=AUTOTUNE)

    train_ds = (
        train_base.enumerate()
        .map(
            lambda i, img_label: _train_map(img_label[0], img_label[1], i),
            num_parallel_calls=AUTOTUNE,
        )
        .batch(BATCH_SIZE, drop_remainder=False)
        .prefetch(AUTOTUNE)
    )

    val_ds = (
        val_base.map(_val_map, num_parallel_calls=AUTOTUNE)
        .batch(BATCH_SIZE, drop_remainder=False)
        .prefetch(AUTOTUNE)
    )
else:
    train_base = tf.data.Dataset.from_tensor_slices(
        (train_paths, train_labels)
    ).with_options(options)
    train_base = train_base.map(_decode_resize_only, num_parallel_calls=AUTOTUNE)
    train_base = train_base.shuffle(
        buffer_size=min(SHUFFLE_BUFFER, len(train_paths)),
        seed=SEED,
        reshuffle_each_iteration=True,
    )

    train_ds = (
        train_base.enumerate()
        .map(
            lambda i, img_label: _train_map(img_label[0], img_label[1], i),
            num_parallel_calls=AUTOTUNE,
        )
        .batch(BATCH_SIZE, drop_remainder=False)
        .prefetch(AUTOTUNE)
    )

    val_base = tf.data.Dataset.from_tensor_slices((val_paths, val_labels)).with_options(
        options
    )
    val_base = val_base.map(_decode_resize_only, num_parallel_calls=AUTOTUNE)
    val_ds = (
        val_base.map(_val_map, num_parallel_calls=AUTOTUNE)
        .batch(BATCH_SIZE, drop_remainder=False)
        .prefetch(AUTOTUNE)
    )

history = my_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    verbose=1,
)




## === cell 3
test_images = tf.io.gfile.glob(os.path.join(TEST_IMG_DIR, "*.jpg"))
if len(test_images) == 0:
    raise FileNotFoundError(f"No .jpg files found under {TEST_IMG_DIR}")
test_images = sorted(test_images)

df_test = pd.DataFrame(test_images, columns=["path"])


def _test_decode_resize(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32)
    img = preprocess_fn(img)
    return img


def make_test_ds(batch_size=64):
    ds = tf.data.Dataset.from_tensor_slices(df_test["path"].values).with_options(
        options
    )
    ds = ds.map(_test_decode_resize, num_parallel_calls=AUTOTUNE)
    ds = ds.cache()
    ds = ds.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)
    return ds




## === cell 4
test_ds = make_test_ds(batch_size=128)

pred_test = my_model.predict(
    test_ds,
    verbose=1,
)
pred_test_labels = np.argmax(pred_test, axis=-1).astype(int)

final_submission = df_test.copy()
final_submission["image_id"] = final_submission["path"].str.rsplit("/", n=1).str[-1]
final_submission["label"] = pred_test_labels

sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
final_csv = sample_sub[["image_id"]].merge(
    final_submission[["image_id", "label"]],
    on="image_id",
    how="left",
)

final_csv["label"] = final_csv["label"].fillna(0).astype(int)

final_csv.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", final_csv.shape)
print(final_csv.head())




## === cell 5
assert os.path.exists("submission.csv"), "submission.csv was not created"
chk = pd.read_csv("submission.csv")
assert list(chk.columns) == ["image_id", "label"], "Submission columns are incorrect"
assert len(chk) == 2676, f"Unexpected submission length: {len(chk)}"
assert chk["label"].between(0, 4).all(), "Labels must be integers in [0, 4]"
chk.head()
