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

import glob
import json
import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow.keras as k

from PIL import Image  # kept as in original (even if unused)

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

try:
    from tensorflow.keras import mixed_precision

    mixed_precision.set_global_policy("mixed_float16")
except Exception:
    pass

import warnings

warnings.filterwarnings("ignore")

print("TF:", tf.__version__)
print("Eager:", tf.executing_eagerly())




## === cell 1
models = []
image_size = 512
n_splits = 2
model_name = "effnetb0"

fold_name = "-fold.weights.h5"

INPUT_DIR = "../input/cassava-leaf-disease-classification"
TRAIN_CSV = os.path.join(INPUT_DIR, "train.csv")
SAMPLE_SUB = os.path.join(INPUT_DIR, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(INPUT_DIR, "train_images")
TEST_IMG_DIR = os.path.join(INPUT_DIR, "test_images")

TRAIN_TFREC_DIR = os.path.join(INPUT_DIR, "train_tfrecords")
TEST_TFREC_DIR = os.path.join(INPUT_DIR, "test_tfrecords")
TRAIN_TFRECS = sorted(glob.glob(os.path.join(TRAIN_TFREC_DIR, "*.tfrec")))
TEST_TFRECS = sorted(glob.glob(os.path.join(TEST_TFREC_DIR, "*.tfrec")))
print("Train tfrecords:", len(TRAIN_TFRECS), "Test tfrecords:", len(TEST_TFRECS))

train_df = pd.read_csv(TRAIN_CSV)
sample_sub = pd.read_csv(SAMPLE_SUB)

num_classes = train_df["label"].nunique()
print("Train shape:", train_df.shape, "num_classes:", num_classes)
print("Sample submission shape:", sample_sub.shape)




## === cell 2
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input

AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 16  # keep moderate for 512x512 within Kaggle GPU RAM
EPOCHS = 3  # unchanged

DATA_OPTS = tf.data.Options()
DATA_OPTS.experimental_deterministic = True
try:
    DATA_OPTS.experimental_optimization.apply_default_optimizations = True
    DATA_OPTS.experimental_optimization.map_parallelization = True
    DATA_OPTS.experimental_optimization.autotune_buffers = True
except Exception:
    pass
try:
    DATA_OPTS.threading.private_threadpool_size = 16
    DATA_OPTS.threading.max_intra_op_parallelism = 1
except Exception:
    pass

_TFREC_FEATURES = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),
    "image_name": tf.io.FixedLenFeature([], tf.string),
}


def _detect_tfrecord_compression(path):
    p = str(path).lower()
    if p.endswith(".gz") or p.endswith(".gzip"):
        return "GZIP"
    return ""


_TFREC_COMPRESSION = (
    _detect_tfrecord_compression(TRAIN_TFRECS[0]) if TRAIN_TFRECS else ""
)


def _decode_resize_preprocess(img_bytes):
    img = tf.io.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, [image_size, image_size], method="bilinear")
    img = preprocess_input(img)
    return img


@tf.function
def parse_train_example(example_proto):
    ex = tf.io.parse_single_example(example_proto, _TFREC_FEATURES)
    img = _decode_resize_preprocess(ex["image"])
    label = tf.cast(ex["target"], tf.int32)
    return img, tf.one_hot(label, depth=num_classes)


@tf.function
def parse_test_example_with_name(example_proto):
    ex = tf.io.parse_single_example(example_proto, _TFREC_FEATURES)
    img = _decode_resize_preprocess(ex["image"])
    name = ex["image_name"]
    return img, name


def make_train_dataset_from_tfrecs(tfrecs, training=True):
    ds = tf.data.TFRecordDataset(
        tfrecs, num_parallel_reads=AUTOTUNE, compression_type=_TFREC_COMPRESSION
    )
    ds = ds.with_options(DATA_OPTS)
    if training:
        ds = ds.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(parse_train_example, num_parallel_calls=AUTOTUNE)
    ds = ds.cache()
    ds = ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)
    return ds


def make_test_dataset_from_tfrecs_with_names(tfrecs):
    ds = tf.data.TFRecordDataset(
        tfrecs, num_parallel_reads=AUTOTUNE, compression_type=_TFREC_COMPRESSION
    )
    ds = ds.with_options(DATA_OPTS)
    ds = ds.map(parse_test_example_with_name, num_parallel_calls=AUTOTUNE)
    ds = ds.cache()
    ds = ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)
    return ds


def build_model():
    inputs = k.Input(shape=(image_size, image_size, 3))
    base = EfficientNetB0(include_top=False, weights="imagenet", input_tensor=inputs)
    x = k.layers.GlobalAveragePooling2D()(base.output)
    x = k.layers.Dropout(0.2)(x)
    outputs = k.layers.Dense(num_classes, activation="softmax", dtype="float32")(x)
    model = k.Model(inputs, outputs)
    model.compile(
        optimizer=k.optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model




## === cell 3
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)


@tf.function
def _parse_target_only(example_proto):
    ex = tf.io.parse_single_example(example_proto, _TFREC_FEATURES)
    return ex["target"]


def _tfrecord_major_label(tfrec_path, max_scan=256, batch_size=512):
    ds = tf.data.TFRecordDataset([tfrec_path], compression_type=_TFREC_COMPRESSION)
    ds = (
        ds.take(max_scan)
        .map(_parse_target_only, num_parallel_calls=AUTOTUNE)
        .batch(batch_size)
    )

    counts = np.zeros((num_classes,), dtype=np.int64)
    seen = 0
    for b in ds:
        arr = b.numpy().astype(np.int32, copy=False)
        if arr.size:
            counts += np.bincount(arr, minlength=num_classes)
            seen += arr.size
            if seen >= max_scan:
                break
    if seen == 0:
        return 0
    return int(counts.argmax())


tfrec_major_labels = np.fromiter(
    (_tfrecord_major_label(p) for p in TRAIN_TFRECS),
    dtype=np.int32,
    count=len(TRAIN_TFRECS),
)

saved_paths = []
for fold, (tr_shard_idx, va_shard_idx) in enumerate(
    skf.split(np.arange(len(TRAIN_TFRECS)), tfrec_major_labels), start=1
):
    tr_tfrecs = [TRAIN_TFRECS[i] for i in tr_shard_idx]
    va_tfrecs = [TRAIN_TFRECS[i] for i in va_shard_idx]

    tr_ds = make_train_dataset_from_tfrecs(tr_tfrecs, training=True)
    va_ds = make_train_dataset_from_tfrecs(va_tfrecs, training=False)

    model = build_model()

    ckpt_path = f"{model_name}{fold}{fold_name}"

    callbacks = [
        k.callbacks.ModelCheckpoint(
            ckpt_path,
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=True,
            mode="max",
            verbose=1,
        ),
        k.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=1, verbose=1
        ),
    ]

    model.fit(
        tr_ds,
        validation_data=va_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )

    model.load_weights(ckpt_path)
    models.append(model)
    saved_paths.append(ckpt_path)

print("Saved fold model weights:", saved_paths)
print("Ensemble size:", len(models))




## === cell 4
test_ds = make_test_dataset_from_tfrecs_with_names(TEST_TFRECS)

majority_label = int(train_df["label"].value_counts().idxmax())

if len(models) == 0:
    all_names = []
    for _, batch_names in test_ds:
        all_names.append(batch_names.numpy())
    image_names_all = np.concatenate(all_names).astype("S").astype(str).tolist()
    pred_labels = np.full((len(image_names_all),), majority_label, dtype=int)
else:
    all_names = []
    for _, batch_names in test_ds:
        all_names.append(batch_names.numpy())
    image_names_all = np.concatenate(all_names).astype("S").astype(str).tolist()
    n_test = len(image_names_all)

    sum_probs = None
    for m in models:
        probs_m = m.predict(test_ds.map(lambda x, n: x), verbose=0)
        if sum_probs is None:
            sum_probs = probs_m.astype(np.float32, copy=False)
        else:
            sum_probs += probs_m.astype(np.float32, copy=False)

    avg_probs = sum_probs / float(len(models))
    pred_labels = avg_probs.argmax(axis=1).astype(int)

sub_raw = pd.DataFrame({"image_id": np.array(image_names_all), "label": pred_labels})
sub = sample_sub[["image_id"]].merge(sub_raw, on="image_id", how="left")

if sub["label"].isna().any():
    sub["label"] = sub["label"].fillna(majority_label).astype(int)
else:
    sub["label"] = sub["label"].astype(int)

print(sub.head())
print("Submission shape:", sub.shape)
print("Any missing labels:", sub["label"].isna().any())




## === cell 5
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with columns:", list(sub.columns))
print("submission.csv exists:", os.path.exists("submission.csv"))
print("Rows:", len(sub), "Expected:", len(sample_sub))
