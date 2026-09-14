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
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

ROOT_DIR = "/kaggle/input/cassava-leaf-disease-classification/"
print("ROOT_DIR exists:", os.path.exists(ROOT_DIR))
print("ROOT_DIR listing (head):", sorted(os.listdir(ROOT_DIR))[:20])




## === cell 1
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print("TensorFlow:", tf.__version__)

try:
    tf.keras.utils.set_random_seed(SEED)
except Exception as e:
    print("set_random_seed unavailable:", repr(e))

try:
    tf.config.experimental.enable_op_determinism()
except Exception as e:
    print("enable_op_determinism unavailable:", repr(e))

try:
    tf.data.experimental.enable_debug_mode(False)
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass




## === cell 2
TRAIN_CSV = os.path.join(ROOT_DIR, "train.csv")
TRAIN_DIR = os.path.join(ROOT_DIR, "train_images")
TEST_DIR = os.path.join(ROOT_DIR, "test_images")
SAMPLE_SUB = os.path.join(ROOT_DIR, "sample_submission.csv")
TEST_TFRECORD_DIR = os.path.join(ROOT_DIR, "test_tfrecords")

train_df = pd.read_csv(TRAIN_CSV)
sample_sub = pd.read_csv(SAMPLE_SUB)

print("train_df:", train_df.shape, train_df.columns.tolist())
print("sample_sub:", sample_sub.shape, sample_sub.columns.tolist())

print("TRAIN_DIR exists:", os.path.exists(TRAIN_DIR))
print("TEST_DIR exists:", os.path.exists(TEST_DIR))
print("TEST_TFRECORD_DIR exists:", os.path.exists(TEST_TFRECORD_DIR))




## === cell 3
MODEL_PATH = (
    "/kaggle/input/cassava-leaf-disease-first-look-and-training/best_model.hdf5"
)

import glob


def _find_pretrained_model_path(default_path: str) -> str:
    if os.path.exists(default_path):
        return default_path

    hits = glob.glob("/kaggle/input/*/best_model.hdf5")
    if hits:
        for h in hits:
            if "cassava-leaf-disease-first-look-and-training" in h:
                return h
        return hits[0]
    return default_path


MODEL_PATH = _find_pretrained_model_path(MODEL_PATH)

new_model = None
if os.path.exists(MODEL_PATH):
    new_model = keras.models.load_model(MODEL_PATH, compile=False)
    print("Loaded existing model:", MODEL_PATH)
else:
    print(
        "Pretrained model not found; will train a lightweight fallback model from train_images."
    )




## === cell 4
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = 5
EPOCHS = 3  # preserve original intent: short training

AUTOTUNE = tf.data.AUTOTUNE


@tf.function
def _decode_resize_only(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(
        img, [IMG_SIZE, IMG_SIZE], method=tf.image.ResizeMethod.BILINEAR
    )
    img = tf.cast(img, tf.float32) / 255.0
    return img


@tf.function
def _augment(img):
    img = tf.image.random_flip_left_right(img, seed=SEED)
    img = tf.image.random_brightness(img, max_delta=0.1, seed=SEED)
    return img


def decode_and_resize(path, label=None, training=False):
    img = _decode_resize_only(path)
    if training:
        img = _augment(img)
    if label is None:
        return img
    return img, tf.cast(label, tf.int32)




## === cell 5
if new_model is None:
    train_paths = (
        TRAIN_DIR.rstrip("/") + "/" + train_df["image_id"].astype(str)
    ).to_numpy()
    train_labels = train_df["label"].astype(np.int32).to_numpy()

    rng = np.random.RandomState(SEED)
    idx = np.arange(len(train_paths))
    rng.shuffle(idx)
    train_paths = train_paths[idx]
    train_labels = train_labels[idx]

    val_size = int(0.1 * len(train_paths))
    val_paths = train_paths[:val_size]
    val_labels = train_labels[:val_size]
    tr_paths = train_paths[val_size:]
    tr_labels = train_labels[val_size:]

    options = tf.data.Options()
    options.experimental_deterministic = True

    tr_ds = tf.data.Dataset.from_tensor_slices((tr_paths, tr_labels)).with_options(
        options
    )
    tr_ds = tr_ds.map(
        lambda p, y: decode_and_resize(p, y, training=True),
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    )
    tr_ds = tr_ds.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
    tr_ds = tr_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

    va_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_labels)).with_options(
        options
    )
    va_ds = va_ds.map(
        lambda p, y: decode_and_resize(p, y, training=False),
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    ).cache()
    va_ds = va_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2, seed=SEED)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
    new_model = keras.Model(inputs, outputs)

    new_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
        jit_compile=True,
    )

    history = new_model.fit(tr_ds, validation_data=va_ds, epochs=EPOCHS, verbose=2)
    print(
        "Finished training fallback model. Final val_accuracy:",
        float(history.history.get("val_accuracy", [np.nan])[-1]),
    )




## === cell 6
FEATURES = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
}


@tf.function
def _parse_test_example(example_proto):
    ex = tf.io.parse_single_example(example_proto, FEATURES)
    img = tf.image.decode_jpeg(ex["image"], channels=3)
    img = tf.image.resize(
        img, [IMG_SIZE, IMG_SIZE], method=tf.image.ResizeMethod.BILINEAR
    )
    img = tf.cast(img, tf.float32) / 255.0
    return ex["image_name"], img


options = tf.data.Options()
options.experimental_deterministic = True
try:
    options.experimental_optimization.apply_default_optimizations = True
    options.experimental_optimization.map_parallelization = True
    options.experimental_optimization.parallel_batch = True
    options.autotune.enabled = True
except Exception:
    pass

tfrec_files = sorted(glob.glob(os.path.join(TEST_TFRECORD_DIR, "*.tfrec")))
use_tfrecords = len(tfrec_files) > 0
print("Found test tfrecords:", len(tfrec_files), "use_tfrecords:", use_tfrecords)

if use_tfrecords:
    compression_type = None
    if tfrec_files and (
        tfrec_files[0].endswith(".gz") or tfrec_files[0].endswith(".tfrec.gz")
    ):
        compression_type = "GZIP"

    raw_ds = tf.data.TFRecordDataset(
        tfrec_files, num_parallel_reads=AUTOTUNE, compression_type=compression_type
    ).with_options(options)

    parsed_ds = raw_ds.map(
        _parse_test_example, num_parallel_calls=AUTOTUNE, deterministic=True
    )

    names_ds = parsed_ds.map(
        lambda n, img: n, num_parallel_calls=AUTOTUNE, deterministic=True
    )
    imgs_ds = parsed_ds.map(
        lambda n, img: img, num_parallel_calls=AUTOTUNE, deterministic=True
    )

    names_ds = names_ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)
    imgs_ds = imgs_ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)

    try:
        new_model.predict_function = tf.function(
            new_model.predict_function, jit_compile=True
        )
    except Exception:
        pass

    probs = new_model.predict(imgs_ds, verbose=0)
    preds_arr = probs.argmax(axis=1).astype(int)

    test_names = np.concatenate([b.numpy().astype("U") for b in names_ds], axis=0)

    if len(test_names) != len(preds_arr):
        raise RuntimeError(
            f"Name/pred length mismatch: {len(test_names)} vs {len(preds_arr)}"
        )

    sample_names = sample_sub["image_id"].to_numpy(dtype=str)
    if len(sample_names) == len(test_names) and np.array_equal(
        sample_names, test_names
    ):
        preds = preds_arr.tolist()
    else:
        pred_map = dict(zip(test_names.tolist(), preds_arr.tolist()))
        preds = [
            pred_map.get(img_id, None) for img_id in sample_sub["image_id"].tolist()
        ]
else:
    test_images = sorted(os.listdir(TEST_DIR))
    print("n test images:", len(test_images), "head:", test_images[:5])

    test_paths = (
        TEST_DIR.rstrip("/") + "/" + pd.Series(test_images, dtype="string")
    ).to_numpy(dtype=str)

    test_ds = tf.data.Dataset.from_tensor_slices(test_paths).with_options(options)
    test_ds = test_ds.map(
        lambda p: decode_and_resize(p, label=None, training=False),
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    )
    test_ds = test_ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)

    try:
        new_model.predict_function = tf.function(
            new_model.predict_function, jit_compile=True
        )
    except Exception:
        pass

    probs = new_model.predict(test_ds, verbose=0)
    preds = probs.argmax(axis=1).astype(int).tolist()

print(
    "n preds:",
    len(preds),
    "unique preds (head):",
    sorted(set([p for p in preds if p is not None]))[:10],
)




## === cell 7
sub = pd.DataFrame({"image_id": sample_sub["image_id"].tolist(), "label": preds})

if sub["label"].isna().any():
    fallback_label = int(train_df["label"].mode().iloc[0])
    sub["label"] = sub["label"].fillna(fallback_label).astype(int)
else:
    sub["label"] = sub["label"].astype(int)

assert sub.shape[0] == sample_sub.shape[0], "Row count mismatch with sample_submission"
assert list(sub.columns) == ["image_id", "label"], "Submission columns mismatch"

print(sub.head())
print("Submission shape:", sub.shape, "NaNs:", sub.isna().sum().to_dict())

sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv")
