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
import random
import numpy as np
import pandas as pd

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

ROOT_DIR = "/kaggle/input/cassava-leaf-disease-classification/"
TRAIN_CSV = os.path.join(ROOT_DIR, "train.csv")
SAMPLE_SUB = os.path.join(ROOT_DIR, "sample_submission.csv")
TRAIN_DIR = os.path.join(ROOT_DIR, "train_images")
TEST_DIR = os.path.join(ROOT_DIR, "test_images")
TRAIN_TFRECORDS_DIR = os.path.join(ROOT_DIR, "train_tfrecords")
TEST_TFRECORDS_DIR = os.path.join(ROOT_DIR, "test_tfrecords")

print("ROOT_DIR exists:", os.path.exists(ROOT_DIR))
print("Train CSV exists:", os.path.exists(TRAIN_CSV))
print("Train images dir exists:", os.path.exists(TRAIN_DIR))
print("Test images dir exists:", os.path.exists(TEST_DIR))
print("Train tfrecords dir exists:", os.path.exists(TRAIN_TFRECORDS_DIR))
print("Test tfrecords dir exists:", os.path.exists(TEST_TFRECORDS_DIR))
print("Sample submission exists:", os.path.exists(SAMPLE_SUB))
print(
    "n_test_images_files:",
    len(os.listdir(TEST_DIR)) if os.path.exists(TEST_DIR) else None,
)



## === cell 1
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

tf.random.set_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism()
except Exception as e:
    print("Determinism setting not available:", repr(e))

try:
    tf.config.threading.set_inter_op_parallelism_threads(0)
    tf.config.threading.set_intra_op_parallelism_threads(0)
except Exception as e:
    print("Threading setting not available:", repr(e))

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
    except Exception as e:
        print("GPU memory growth setting not available:", repr(e))

print("TensorFlow:", tf.__version__)
print("GPUs:", gpus)



## === cell 2
train_df = pd.read_csv(TRAIN_CSV)
sub_df = pd.read_csv(SAMPLE_SUB)

assert set(train_df.columns) == {"image_id", "label"}
assert set(sub_df.columns) == {"image_id", "label"}

train_df["filepath"] = TRAIN_DIR.rstrip("/") + "/" + train_df["image_id"].astype(str)
sub_df["filepath"] = TEST_DIR.rstrip("/") + "/" + sub_df["image_id"].astype(str)

print("Train size:", len(train_df), "Test size:", len(sub_df))

NUM_CLASSES = int(train_df["label"].nunique())
print("NUM_CLASSES:", NUM_CLASSES)



## === cell 3
IMG_SIZE = 300
BATCH_SIZE = 32
EPOCHS = 3  # unchanged

from sklearn.model_selection import train_test_split

train_idx, val_idx = train_test_split(
    np.arange(len(train_df)),
    test_size=0.1,
    random_state=SEED,
    stratify=train_df["label"].values,
)

train_part = train_df.iloc[train_idx].reset_index(drop=True)
val_part = train_df.iloc[val_idx].reset_index(drop=True)

print("Train/Val sizes:", len(train_part), len(val_part))



## === cell 4
AUTOTUNE = tf.data.AUTOTUNE

_USE_INMEM_CACHE_FOR_FILE_PIPELINE = True


def decode_image(path):
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE], method="bilinear")
    img = tf.cast(img, tf.float32) / 255.0
    return img


_FEATURES = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),
}


def _decode_from_bytes(encoded):
    img = tf.io.decode_jpeg(encoded, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE], method="bilinear")
    img = tf.cast(img, tf.float32) / 255.0
    return img


def _parse_train_example(example_proto):
    ex = tf.io.parse_single_example(example_proto, _FEATURES)
    img = _decode_from_bytes(ex["image"])
    label = tf.cast(ex["target"], tf.int32)
    return img, label


_TEST_FEATURES = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
}


def _parse_test_example(example_proto):
    ex = tf.io.parse_single_example(example_proto, _TEST_FEATURES)
    img = _decode_from_bytes(ex["image"])
    return img


def _list_tfrec_files(folder):
    if not tf.io.gfile.exists(folder):
        return []
    files = tf.io.gfile.glob(os.path.join(folder, "*.tfrec"))
    return sorted(files)


TRAIN_TFRECS = _list_tfrec_files(TRAIN_TFRECORDS_DIR)
TEST_TFRECS = _list_tfrec_files(TEST_TFRECORDS_DIR)
print("Found train tfrecs:", len(TRAIN_TFRECS), "test tfrecs:", len(TEST_TFRECS))

tfrec_mode_available = len(TRAIN_TFRECS) > 0 and len(TEST_TFRECS) > 0
print("TFRecord mode available:", tfrec_mode_available)

_options = tf.data.Options()
_options.deterministic = False

opt = _options.experimental_optimization
for name, value in [
    ("apply_default_optimizations", True),
    ("map_parallelization", True),
    ("autotune_buffers", True),
    ("parallel_batch", True),
]:
    try:
        if hasattr(opt, name):
            setattr(opt, name, value)
    except Exception as e:
        print(f"Could not set optimization option {name}:", repr(e))

USE_PREFETCH_TO_DEVICE = bool(gpus)
PREFETCH_DEVICE = "/GPU:0" if USE_PREFETCH_TO_DEVICE else None


def _maybe_prefetch_to_device(ds):
    if USE_PREFETCH_TO_DEVICE:
        try:
            ds = ds.apply(
                tf.data.experimental.prefetch_to_device(
                    PREFETCH_DEVICE, buffer_size=AUTOTUNE
                )
            )
        except Exception as e:
            print("prefetch_to_device not available:", repr(e))
    return ds


def make_ds(df, training):
    paths = df["filepath"].to_numpy()
    labels = df["label"].to_numpy(dtype=np.int32)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels)).with_options(_options)

    if training:
        ds = ds.shuffle(8192, seed=SEED, reshuffle_each_iteration=True)

    ds = ds.map(
        lambda path, label: (decode_image(path), label),
        num_parallel_calls=AUTOTUNE,
        deterministic=False,
    )

    ds = ds.apply(tf.data.experimental.ignore_errors())

    if _USE_INMEM_CACHE_FOR_FILE_PIPELINE:
        ds = ds.cache()

    ds = ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)
    ds = _maybe_prefetch_to_device(ds)
    return ds


def _tfrecord_count_records(tfrec_path: str) -> int:
    return sum(1 for _ in tf.data.TFRecordDataset(tfrec_path))


def _count_tfrecords_cached(tfrec_files, cache_path: str):
    if not tfrec_files:
        return np.zeros((0,), dtype=np.int64)

    try:
        if os.path.exists(cache_path):
            arr = np.load(cache_path)
            if len(arr) == len(tfrec_files):
                return arr.astype(np.int64, copy=False)
    except Exception as e:
        print("Could not load TFRecord counts cache:", repr(e))

    counts = np.empty((len(tfrec_files),), dtype=np.int64)
    for i, f in enumerate(tfrec_files):
        counts[i] = _tfrecord_count_records(f)

    try:
        np.save(cache_path, counts)
    except Exception as e:
        print("Could not save TFRecord counts cache:", repr(e))
    return counts


def make_stream_from_tfrecords(tfrecs, shuffle_files, shuffle_records, parse_fn):
    files = tf.data.Dataset.from_tensor_slices(tfrecs)
    if shuffle_files and len(tfrecs) > 1:
        files = files.shuffle(len(tfrecs), seed=SEED, reshuffle_each_iteration=True)

    cycle_len = min(16, max(1, len(tfrecs)))

    def _dataset_for_file(f):
        return tf.data.TFRecordDataset(
            f,
            buffer_size=16 * 1024 * 1024,
        )

    ds = files.interleave(
        _dataset_for_file,
        cycle_length=cycle_len,
        num_parallel_calls=AUTOTUNE,
        deterministic=False,
    )

    if shuffle_records:
        ds = ds.shuffle(8192, seed=SEED, reshuffle_each_iteration=True)

    ds = ds.map(parse_fn, num_parallel_calls=AUTOTUNE, deterministic=False)
    ds = ds.apply(tf.data.experimental.ignore_errors())
    ds = ds.with_options(_options)
    return ds


STEPS_PER_EPOCH = None
VAL_STEPS = None

if tfrec_mode_available:
    n_files = len(TRAIN_TFRECS)
    n_val_files = max(1, int(round(0.1 * n_files)))
    n_train_files = n_files - n_val_files
    train_tfrecs_split = TRAIN_TFRECS[:n_train_files]
    val_tfrecs_split = TRAIN_TFRECS[n_train_files:]

    print(
        "TFRec file split:",
        "n_train_files=",
        len(train_tfrecs_split),
        "n_val_files=",
        len(val_tfrecs_split),
    )

    train_counts = _count_tfrecords_cached(
        train_tfrecs_split, cache_path="train_tfrec_record_counts.npy"
    )
    val_counts = _count_tfrecords_cached(
        val_tfrecs_split, cache_path="val_tfrec_record_counts.npy"
    )
    n_train_records = int(train_counts.sum())
    n_val_records = int(val_counts.sum())

    STEPS_PER_EPOCH = int(np.ceil(n_train_records / BATCH_SIZE))
    VAL_STEPS = int(np.ceil(n_val_records / BATCH_SIZE))
    print("Exact n_train_records:", n_train_records, "steps/epoch:", STEPS_PER_EPOCH)
    print("Exact n_val_records:", n_val_records, "val_steps:", VAL_STEPS)

    train_ds = make_stream_from_tfrecords(
        train_tfrecs_split,
        shuffle_files=True,
        shuffle_records=True,
        parse_fn=_parse_train_example,
    )
    train_ds = train_ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)
    train_ds = _maybe_prefetch_to_device(train_ds)
    train_ds = train_ds.repeat()  # requires explicit steps_per_epoch below

    val_ds = make_stream_from_tfrecords(
        val_tfrecs_split,
        shuffle_files=False,
        shuffle_records=False,
        parse_fn=_parse_train_example,
    )
    val_ds = val_ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)
    val_ds = _maybe_prefetch_to_device(val_ds)
    val_ds = val_ds.repeat()
else:
    train_ds = make_ds(train_part, training=True)
    val_ds = make_ds(val_part, training=False)



## === cell 5
data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal", seed=SEED),
        layers.RandomRotation(0.05, seed=SEED),
        layers.RandomZoom(0.1, seed=SEED),
    ],
    name="aug",
)


class AugLayer(layers.Layer):
    def call(self, x, training=None):
        return data_augmentation(x, training=training)


model = keras.Sequential(
    [
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        AugLayer(),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.25, seed=SEED),
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ]
)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
    steps_per_execution=32,
)

model.summary()



## === cell 6
fit_kwargs = {}
if STEPS_PER_EPOCH is not None:
    fit_kwargs["steps_per_epoch"] = STEPS_PER_EPOCH
if VAL_STEPS is not None:
    fit_kwargs["validation_steps"] = VAL_STEPS

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    verbose=2,
    **fit_kwargs,
)




## === cell 7
def make_test_ds(df):
    paths = df["filepath"].to_numpy()
    ds = tf.data.Dataset.from_tensor_slices(paths).with_options(_options)

    ds = ds.map(
        lambda path: decode_image(path),
        num_parallel_calls=AUTOTUNE,
        deterministic=False,
    )

    ds = ds.apply(tf.data.experimental.ignore_errors())

    if _USE_INMEM_CACHE_FOR_FILE_PIPELINE:
        ds = ds.cache()

    ds = ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)
    ds = _maybe_prefetch_to_device(ds)
    return ds


if tfrec_mode_available:

    def _dataset_for_file(f):
        return tf.data.TFRecordDataset(
            f,
            buffer_size=16 * 1024 * 1024,
        )

    test_ds = (
        tf.data.Dataset.from_tensor_slices(TEST_TFRECS)
        .interleave(
            _dataset_for_file,
            cycle_length=min(16, len(TEST_TFRECS)) if len(TEST_TFRECS) else 1,
            num_parallel_calls=AUTOTUNE,
            deterministic=False,
        )
        .map(_parse_test_example, num_parallel_calls=AUTOTUNE, deterministic=False)
        .apply(tf.data.experimental.ignore_errors())
        .with_options(_options)
        .batch(BATCH_SIZE, drop_remainder=False)
        .prefetch(AUTOTUNE)
    )
    test_ds = _maybe_prefetch_to_device(test_ds)
else:
    test_ds = make_test_ds(sub_df)

probs = model.predict(test_ds, verbose=0)
preds = probs.argmax(axis=1).astype(int)

print("Preds length:", len(preds), "Expected:", len(sub_df))

submission = pd.DataFrame(
    {
        "image_id": sub_df["image_id"].values,
        "label": preds,
    }
)

assert len(submission) == len(sub_df)
assert submission["label"].dtype.kind in ("i", "u")
assert submission.columns.tolist() == ["image_id", "label"]

submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
