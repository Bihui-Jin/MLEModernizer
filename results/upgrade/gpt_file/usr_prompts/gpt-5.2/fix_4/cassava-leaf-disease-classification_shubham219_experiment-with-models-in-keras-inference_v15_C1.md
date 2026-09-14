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
import glob
import math
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.applications import EfficientNetB3
from sklearn.model_selection import train_test_split

SEED = 42
DEBUG = False

os.environ["PYTHONHASHSEED"] = str(SEED)
try:
    tf.keras.utils.set_random_seed(SEED)
except Exception:
    tf.random.set_seed(SEED)
np.random.seed(SEED)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

AUTOTUNE = tf.data.AUTOTUNE

DATA_ROOT_CANDIDATES = [
    "/kaggle/input/cassava-leaf-disease-classification",
    "../input/cassava-leaf-disease-classification",
]
DATA_ROOT = None
for c in DATA_ROOT_CANDIDATES:
    if os.path.exists(c):
        DATA_ROOT = c
        break
if DATA_ROOT is None:
    raise FileNotFoundError(
        "Could not find cassava dataset folder in expected locations."
    )

TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train_images")
TEST_IMG_DIR = os.path.join(DATA_ROOT, "test_images")
SAMPLE_SUB = os.path.join(DATA_ROOT, "sample_submission.csv")
TRAIN_TFREC_DIR = os.path.join(DATA_ROOT, "train_tfrecords")
TEST_TFREC_DIR = os.path.join(DATA_ROOT, "test_tfrecords")

for p in [TRAIN_CSV, TRAIN_IMG_DIR, TEST_IMG_DIR, SAMPLE_SUB]:
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing expected path: {p}")

print("Using DATA_ROOT:", DATA_ROOT)
print("TensorFlow:", tf.__version__)



## === cell 1
df = pd.read_csv(TRAIN_CSV)
if DEBUG:
    df = df.sample(2000, random_state=SEED).reset_index(drop=True)

df["path"] = TRAIN_IMG_DIR.rstrip("/") + "/" + df["image_id"].astype(str)

if not os.path.isdir(TRAIN_IMG_DIR):
    raise FileNotFoundError(f"Training image directory missing: {TRAIN_IMG_DIR}")
if df["path"].isna().any():
    raise ValueError("Found NaN paths after path construction.")

df_train, df_val = train_test_split(
    df, test_size=0.15, random_state=SEED, stratify=df["label"]
)

IMG_SIZE = (300, 300)
BATCH_SIZE = 32 if not DEBUG else 16
NUM_CLASSES = 5


def _decode_resize(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32) * (1.0 / 255.0)  # same as rescale=1/255
    return img


data_augment = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip(mode="horizontal", seed=SEED),
        tf.keras.layers.RandomRotation(
            factor=15.0 / 180.0, seed=SEED, fill_mode="reflect"
        ),
        tf.keras.layers.RandomTranslation(
            height_factor=0.10, width_factor=0.10, seed=SEED, fill_mode="reflect"
        ),
        tf.keras.layers.RandomZoom(
            height_factor=(-0.10, 0.10),
            width_factor=(-0.10, 0.10),
            seed=SEED,
            fill_mode="reflect",
        ),
    ],
    name="data_augment",
)

_TRAIN_FEATURES = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),
}
_TEST_FEATURES = {"image": tf.io.FixedLenFeature([], tf.string)}


def _decode_resize_from_bytes(img_bytes):
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32) * (1.0 / 255.0)
    return img


def _configure_ds(ds):
    opts = tf.data.Options()
    opts.experimental_deterministic = False
    opts.threading.private_threadpool_size = max(8, (os.cpu_count() or 8))
    opts.threading.max_intra_op_parallelism = 1
    return ds.with_options(opts)


def _get_train_tfrecord_files():
    files = sorted(glob.glob(os.path.join(TRAIN_TFREC_DIR, "*.tfrec")))
    return files


def _parse_train_example(serialized):
    ex = tf.io.parse_single_example(serialized, _TRAIN_FEATURES)
    x = _decode_resize_from_bytes(ex["image"])
    y = tf.cast(ex["target"], tf.int32)
    return x, y


def _parse_test_example(serialized):
    ex = tf.io.parse_single_example(serialized, _TEST_FEATURES)
    x = _decode_resize_from_bytes(ex["image"])
    return x


def make_train_ds(paths, labels, batch_size):
    train_tfrec_files = _get_train_tfrecord_files()
    if not train_tfrec_files:
        paths = tf.convert_to_tensor(paths, dtype=tf.string)
        labels = tf.convert_to_tensor(labels, dtype=tf.int32)
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        ds = ds.shuffle(
            buffer_size=min(int(paths.shape[0]), 8192),
            seed=SEED,
            reshuffle_each_iteration=True,
        )

        def _map_fn(p, y):
            x = _decode_resize(p)
            x = data_augment(x, training=True)
            return x, y

        ds = ds.map(_map_fn, num_parallel_calls=AUTOTUNE)
        ds = ds.batch(batch_size, drop_remainder=False)
        ds = ds.prefetch(AUTOTUNE)
        return _configure_ds(ds)

    ds = tf.data.TFRecordDataset(train_tfrec_files, num_parallel_reads=AUTOTUNE)
    ds = _configure_ds(ds)
    ds = ds.shuffle(buffer_size=8192, seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(_parse_train_example, num_parallel_calls=AUTOTUNE)

    def _aug_map(x, y):
        x = data_augment(x, training=True)
        return x, y

    ds = ds.map(_aug_map, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def make_val_ds(paths, labels, batch_size):
    train_tfrec_files = _get_train_tfrecord_files()
    if not train_tfrec_files:
        paths = tf.convert_to_tensor(paths, dtype=tf.string)
        labels = tf.convert_to_tensor(labels, dtype=tf.int32)
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))

        def _map_fn(p, y):
            x = _decode_resize(p)
            return x, y

        ds = ds.map(_map_fn, num_parallel_calls=AUTOTUNE)
        ds = ds.batch(batch_size, drop_remainder=False)
        ds = ds.prefetch(AUTOTUNE)
        return _configure_ds(ds)

    val_ids = tf.constant(df_val["image_id"].values.astype(str))
    val_table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            val_ids, tf.ones_like(val_ids, dtype=tf.int32)
        ),
        default_value=0,
    )

    def _parse_with_id(serialized):
        ex = tf.io.parse_single_example(
            serialized,
            {
                "image": tf.io.FixedLenFeature([], tf.string),
                "target": tf.io.FixedLenFeature([], tf.int64),
                "image_name": tf.io.FixedLenFeature([], tf.string),
            },
        )
        x = _decode_resize_from_bytes(ex["image"])
        y = tf.cast(ex["target"], tf.int32)
        img_id = ex["image_name"]
        return x, y, img_id

    try:
        ds = tf.data.TFRecordDataset(train_tfrec_files, num_parallel_reads=AUTOTUNE)
        ds = _configure_ds(ds)
        ds = ds.map(_parse_with_id, num_parallel_calls=AUTOTUNE)

        def _is_val(x, y, img_id):
            return tf.equal(val_table.lookup(img_id), 1)

        ds = ds.filter(_is_val)
        ds = ds.map(lambda x, y, img_id: (x, y), num_parallel_calls=AUTOTUNE)
        ds = ds.batch(batch_size, drop_remainder=False)
        ds = ds.prefetch(AUTOTUNE)
        return ds
    except Exception:
        paths = tf.convert_to_tensor(paths, dtype=tf.string)
        labels = tf.convert_to_tensor(labels, dtype=tf.int32)
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))

        def _map_fn(p, y):
            x = _decode_resize(p)
            return x, y

        ds = ds.map(_map_fn, num_parallel_calls=AUTOTUNE)
        ds = ds.batch(batch_size, drop_remainder=False)
        ds = ds.prefetch(AUTOTUNE)
        return _configure_ds(ds)


train_ds = make_train_ds(df_train["path"].values, df_train["label"].values, BATCH_SIZE)
val_ds = make_val_ds(df_val["path"].values, df_val["label"].values, BATCH_SIZE)

train_steps = math.ceil(len(df_train) / BATCH_SIZE)
val_steps = math.ceil(len(df_val) / BATCH_SIZE)

inputs = Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
base = EfficientNetB3(include_top=False, weights="imagenet", input_tensor=inputs)
x = base.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.2)(x)
outputs = Dense(NUM_CLASSES, activation="softmax")(x)
my_model = Model(inputs=inputs, outputs=outputs)

base.trainable = False

my_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
    jit_compile=True,
)

EPOCHS = 3 if not DEBUG else 1

history = my_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    steps_per_epoch=train_steps,
    validation_steps=val_steps,
    verbose=1,
)

base.trainable = True
my_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
    jit_compile=True,
)
history_ft = my_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=1 if not DEBUG else 1,
    steps_per_epoch=train_steps,
    validation_steps=val_steps,
    verbose=1,
)



## === cell 2
test_tfrec_files = sorted(glob.glob(os.path.join(TEST_TFREC_DIR, "*.tfrec")))

if not test_tfrec_files:
    test_images = glob.glob(os.path.join(TEST_IMG_DIR, "*.jpg"))
    if len(test_images) == 0:
        raise FileNotFoundError(f"No test images found in {TEST_IMG_DIR}")
    df_test = pd.DataFrame({"path": test_images})

    def make_test_ds(paths, batch_size=128):
        paths = tf.convert_to_tensor(np.asarray(paths, dtype=object), dtype=tf.string)
        ds = tf.data.Dataset.from_tensor_slices(paths)

        def _map_fn(p):
            x = _decode_resize(p)
            return x

        ds = ds.map(_map_fn, num_parallel_calls=AUTOTUNE)
        ds = ds.batch(batch_size, drop_remainder=False)
        ds = ds.prefetch(AUTOTUNE)
        return _configure_ds(ds)

    test_ds = make_test_ds(df_test["path"].values, batch_size=128)
    pred_test = my_model.predict(test_ds, verbose=1)
    pred_test_labels = np.argmax(pred_test, axis=-1)

    final_submission = df_test.copy()
    final_submission["image_id"] = final_submission["path"].map(os.path.basename)
    final_submission["label"] = pred_test_labels.astype(int)

    final_csv = final_submission[["image_id", "label"]].copy()

    sample = pd.read_csv(SAMPLE_SUB)
    final_csv = sample[["image_id"]].merge(final_csv, on="image_id", how="left")
    if final_csv["label"].isna().any():
        fill_label = int(pd.Series(pred_test_labels).mode().iloc[0])
        final_csv["label"] = final_csv["label"].fillna(fill_label).astype(int)

    final_csv.to_csv("submission.csv", index=False)
    print("Wrote submission.csv with shape:", final_csv.shape)
    print(final_csv.head())
else:
    def make_test_ds_from_tfrecords(tfrec_files, batch_size=128):
        ds = tf.data.TFRecordDataset(tfrec_files, num_parallel_reads=AUTOTUNE)
        ds = _configure_ds(ds)
        ds = ds.map(_parse_test_example, num_parallel_calls=AUTOTUNE)
        ds = ds.batch(batch_size, drop_remainder=False)
        ds = ds.prefetch(AUTOTUNE)
        return ds

    test_ds = make_test_ds_from_tfrecords(test_tfrec_files, batch_size=128)
    pred_test = my_model.predict(test_ds, verbose=1)
    pred_test_labels = np.argmax(pred_test, axis=-1).astype(int)

    sample = pd.read_csv(SAMPLE_SUB)
    if len(sample) != len(pred_test_labels):
        raise ValueError(
            f"Prediction count {len(pred_test_labels)} != sample_submission rows {len(sample)}. "
            "Cannot safely align predictions."
        )

    final_csv = sample[["image_id"]].copy()
    final_csv["label"] = pred_test_labels
    final_csv.to_csv("submission.csv", index=False)
    print("Wrote submission.csv with shape:", final_csv.shape)
    print(final_csv.head())



## === cell 3
assert os.path.exists("submission.csv"), "submission.csv was not created."
chk = pd.read_csv("submission.csv")
assert list(chk.columns) == [
    "image_id",
    "label",
], f"Bad submission columns: {chk.columns.tolist()}"
assert len(chk) == len(
    pd.read_csv(SAMPLE_SUB)
), "Submission row count does not match sample_submission."
assert chk["label"].between(0, 4).all(), "Labels out of expected range 0..4"
chk.head()
