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

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import glob
import random
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

SEED = 42
DEBUG = False

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(
        max(1, (os.cpu_count() or 4) // 2)
    )
    tf.config.threading.set_inter_op_parallelism_threads(2)
except Exception:
    pass

CANDIDATE_ROOTS = [
    "../input/cassava-leaf-disease-classification",
    "/kaggle/input/cassava-leaf-disease-classification",
    "/kaggle/data/cassava-leaf-disease-classification",
]
DATA_ROOT = None
for r in CANDIDATE_ROOTS:
    if os.path.exists(r):
        DATA_ROOT = r
        break

if DATA_ROOT is None:
    raise FileNotFoundError(
        "Could not find cassava dataset folder in expected locations: "
        + ", ".join(CANDIDATE_ROOTS)
    )

TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
TEST_IMG_DIR = os.path.join(DATA_ROOT, "test_images")
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train_images")
SAMPLE_SUB = os.path.join(DATA_ROOT, "sample_submission.csv")
TRAIN_TFREC_DIR = os.path.join(DATA_ROOT, "train_tfrecords")
TEST_TFREC_DIR = os.path.join(DATA_ROOT, "test_tfrecords")

assert os.path.exists(TRAIN_CSV), f"Missing {TRAIN_CSV}"
assert os.path.exists(TEST_IMG_DIR), f"Missing {TEST_IMG_DIR}"
assert os.path.exists(TRAIN_IMG_DIR), f"Missing {TRAIN_IMG_DIR}"
assert os.path.exists(SAMPLE_SUB), f"Missing {SAMPLE_SUB}"

print("DATA_ROOT:", DATA_ROOT)
print("TensorFlow:", tf.__version__)

train_df = pd.read_csv(TRAIN_CSV)

train_df["path"] = TRAIN_IMG_DIR.rstrip("/") + "/" + train_df["image_id"].astype(str)
train_df["label"] = train_df["label"].astype(int)

val_frac = 0.15
train_split = train_df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
n_val = int(len(train_split) * val_frac)
val_df = train_split.iloc[:n_val].copy()
trn_df = train_split.iloc[n_val:].copy()

IMG_SIZE = (300, 300)
BATCH_SIZE = 32
NUM_CLASSES = 5
EPOCHS = 3 if not DEBUG else 1

AUTOTUNE = tf.data.AUTOTUNE

aug = keras.Sequential(
    [
        layers.RandomFlip("horizontal", seed=SEED),
        layers.RandomRotation(0.0277777778, seed=SEED),  # ~10 degrees / 360
        layers.RandomTranslation(0.05, 0.05, seed=SEED),
        layers.RandomZoom(0.10, 0.10, seed=SEED),
    ],
    name="augmentation",
)


@tf.function
def _decode_resize(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32) / 255.0  # rescale=1/255
    return img


_TFREC_FEATURES = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),
}


@tf.function
def _decode_resize_from_bytes(img_bytes):
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32) / 255.0
    return img


@tf.function
def _parse_tfrecord(example_proto):
    ex = tf.io.parse_single_example(example_proto, _TFREC_FEATURES)
    img = _decode_resize_from_bytes(ex["image"])
    label = tf.cast(ex["target"], tf.int32)
    return img, label


@tf.function
def _augment_and_onehot_map(img, label):
    img = aug(img, training=True)
    label_oh = tf.one_hot(label, NUM_CLASSES)
    return img, label_oh


@tf.function
def _onehot_map(img, label):
    label_oh = tf.one_hot(label, NUM_CLASSES)
    return img, label_oh


def _maybe_prefetch_to_device(ds):
    try:
        gpus = tf.config.list_logical_devices("GPU")
        if gpus:
            return ds.apply(tf.data.experimental.prefetch_to_device("/GPU:0"))
    except Exception:
        pass
    return ds.prefetch(AUTOTUNE)


def _dataset_options():
    opts = tf.data.Options()
    opts.experimental_deterministic = True
    opts.experimental_optimization.apply_default_optimizations = True
    opts.experimental_optimization.map_parallelization = True
    opts.experimental_optimization.parallel_batch = True
    return opts


def _list_tfrecord_files(folder):
    if not os.path.exists(folder):
        return []
    return sorted(glob.glob(os.path.join(folder, "*.tfrec")))


def make_train_val_ds_from_tfrecords(val_fraction=0.15):
    tfrec_files = _list_tfrecord_files(TRAIN_TFREC_DIR)
    if len(tfrec_files) == 0:
        return None, None

    n_total = len(train_df)
    n_val_local = int(n_total * val_fraction)

    val_idx = np.arange(n_val_local, dtype=np.int64)
    trn_idx = np.arange(n_val_local, n_total, dtype=np.int64)

    ds_all = tf.data.TFRecordDataset(tfrec_files, num_parallel_reads=AUTOTUNE)
    ds_all = ds_all.with_options(_dataset_options())
    ds_all = ds_all.map(
        _parse_tfrecord, num_parallel_calls=AUTOTUNE, deterministic=True
    )

    ds_enum = ds_all.enumerate()  # (i, (img,label))

    val_idx_t = tf.constant(val_idx)
    trn_idx_t = tf.constant(trn_idx)

    def _in_set(i, idx_tensor):
        return tf.reduce_any(tf.equal(i, idx_tensor))

    val_raw = ds_enum.filter(lambda i, x: _in_set(i, val_idx_t)).map(
        lambda i, x: x, num_parallel_calls=AUTOTUNE, deterministic=True
    )
    train_raw = ds_enum.filter(lambda i, x: _in_set(i, trn_idx_t)).map(
        lambda i, x: x, num_parallel_calls=AUTOTUNE, deterministic=True
    )

    train_raw = train_raw.cache()
    val_raw = val_raw.cache()

    shuffle_buf = min(n_total - n_val_local, 8192)
    train_raw = train_raw.shuffle(
        buffer_size=shuffle_buf, seed=SEED, reshuffle_each_iteration=True
    )

    train_ds = train_raw.map(
        _augment_and_onehot_map, num_parallel_calls=AUTOTUNE, deterministic=True
    )
    val_ds = val_raw.map(_onehot_map, num_parallel_calls=AUTOTUNE, deterministic=True)

    train_ds = train_ds.batch(BATCH_SIZE, drop_remainder=False)
    val_ds = val_ds.batch(BATCH_SIZE, drop_remainder=False)

    train_ds = _maybe_prefetch_to_device(train_ds)
    val_ds = _maybe_prefetch_to_device(val_ds)
    return train_ds, val_ds


train_ds, val_ds = make_train_val_ds_from_tfrecords(val_fraction=val_frac)

if train_ds is None or val_ds is None:

    @tf.function
    def _decode_only_map(path, label):
        img = _decode_resize(path)
        return img, label

    def make_train_ds(df):
        paths = df["path"].to_numpy(dtype=str)
        labels = df["label"].to_numpy(dtype=np.int32)
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        ds = ds.with_options(_dataset_options())

        ds = ds.shuffle(
            buffer_size=min(len(df), 8192), seed=SEED, reshuffle_each_iteration=True
        )

        ds = ds.map(
            _decode_only_map, num_parallel_calls=AUTOTUNE, deterministic=True
        ).cache()

        ds = ds.map(
            _augment_and_onehot_map, num_parallel_calls=AUTOTUNE, deterministic=True
        )
        ds = ds.batch(BATCH_SIZE, drop_remainder=False)
        ds = _maybe_prefetch_to_device(ds)
        return ds

    def make_val_ds(df):
        paths = df["path"].to_numpy(dtype=str)
        labels = df["label"].to_numpy(dtype=np.int32)
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        ds = ds.with_options(_dataset_options())

        ds = ds.map(
            _decode_only_map, num_parallel_calls=AUTOTUNE, deterministic=True
        ).cache()

        ds = ds.map(_onehot_map, num_parallel_calls=AUTOTUNE, deterministic=True)
        ds = ds.batch(BATCH_SIZE, drop_remainder=False)
        ds = _maybe_prefetch_to_device(ds)
        return ds

    train_ds = make_train_ds(trn_df)
    val_ds = make_val_ds(val_df)

inputs = keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
x = layers.MaxPooling2D()(x)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
my_model = keras.Model(inputs, outputs)

my_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

my_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    verbose=1,
)



## === cell 1
sample_sub_df = pd.read_csv(SAMPLE_SUB)
test_image_ids = sample_sub_df["image_id"].astype(str).tolist()
test_images = [os.path.join(TEST_IMG_DIR, img_id) for img_id in test_image_ids]

if len(test_images) == 0 or not os.path.exists(test_images[0]):
    raise FileNotFoundError(f"Test images not found under {TEST_IMG_DIR}")

df_test = pd.DataFrame({"path": test_images})


@tf.function
def _test_map(path):
    img = _decode_resize(path)
    return img


_TEST_TFREC_FEATURES = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
}


@tf.function
def _parse_test_tfrecord(example_proto):
    ex = tf.io.parse_single_example(example_proto, _TEST_TFREC_FEATURES)
    img = _decode_resize_from_bytes(ex["image"])
    image_id = ex["image_name"]
    return img, image_id


def make_test_ds(batch_size=128):
    tfrec_files = _list_tfrecord_files(TEST_TFREC_DIR)
    if len(tfrec_files) > 0:
        ds = tf.data.TFRecordDataset(tfrec_files, num_parallel_reads=AUTOTUNE)
        ds = ds.with_options(_dataset_options())
        ds = ds.map(
            _parse_test_tfrecord, num_parallel_calls=AUTOTUNE, deterministic=True
        )
        ds = ds.batch(batch_size, drop_remainder=False)
        ds = _maybe_prefetch_to_device(ds)
        return ds, True

    paths = np.asarray(df_test["path"].values, dtype=str)
    ds = tf.data.Dataset.from_tensor_slices(paths)
    ds = ds.with_options(_dataset_options())
    ds = ds.map(_test_map, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = _maybe_prefetch_to_device(ds)
    return ds, False




## === cell 2
test_ds, test_from_tfrecord = make_test_ds(batch_size=128)

if test_from_tfrecord:
    ids_ds = test_ds.map(
        lambda _img, _id: _id, num_parallel_calls=AUTOTUNE, deterministic=True
    ).unbatch()
    test_image_ids_order = (
        [
            x.decode("utf-8")
            for x in ids_ds.batch(4096).as_numpy_iterator().__next__().tolist()
        ]
        if len(sample_sub_df) <= 4096
        else [
            x.decode("utf-8")
            for batch in ids_ds.batch(4096).as_numpy_iterator()
            for x in batch.tolist()
        ]
    )

    img_only_ds = test_ds.map(
        lambda img, _id: img, num_parallel_calls=AUTOTUNE, deterministic=True
    )
    pred_test = my_model.predict(img_only_ds, verbose=1)
else:
    pred_test = my_model.predict(test_ds, verbose=1)
    test_image_ids_order = [os.path.basename(p) for p in df_test["path"].tolist()]

pred_test_labels = np.argmax(pred_test, axis=-1).astype(int)

final_submission = pd.DataFrame(
    {"image_id": test_image_ids_order, "label": pred_test_labels}
)

sample_sub = pd.read_csv(SAMPLE_SUB)
final_csv = sample_sub[["image_id"]].merge(
    final_submission[["image_id", "label"]],
    on="image_id",
    how="left",
)

if final_csv["label"].isna().any():
    fill_label = int(pd.Series(pred_test_labels).mode().iloc[0])
    final_csv["label"] = final_csv["label"].fillna(fill_label).astype(int)
else:
    final_csv["label"] = final_csv["label"].astype(int)

final_csv.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", final_csv.shape)
print(final_csv.head())



## === cell 3
assert os.path.exists("submission.csv"), "submission.csv was not created"
sub_check = pd.read_csv("submission.csv")
assert list(sub_check.columns) == [
    "image_id",
    "label",
], f"Bad columns: {sub_check.columns.tolist()}"
assert len(sub_check) == len(
    pd.read_csv(SAMPLE_SUB)
), "Row count mismatch vs sample_submission.csv"
assert sub_check["label"].between(0, 4).all(), "Labels out of range [0,4]"
sub_check.head()
