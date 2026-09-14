# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


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

# 5. Target score

0.8890903596252644

# 6. Current score

0.61099

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.61099) has done: 'The timeout is dominated by training EfficientNetB0 at 512×512 for 2 folds (full training done twice) and by extra dataset passes during test-name extraction. I keep the exact same model/epochs/folds/loss, but remove avoidable overhead by (1) enabling deterministic but faster tf.data pipelines (non-blocking prefetch, parallel map already used), (2) avoiding a full pass over the test dataset just to collect names by parsing names + images once and predicting in the same pass, and (3) making sure Keras uses the tf.data pipeline efficiently (fixed steps, no extra iterator rebuilds) and avoiding repeated graph retracing. These changes preserve evaluation semantics and outputs (up to negligible FP differences) while cutting redundant input-pipeline work and Python overhead.'

# 9. Code solution

## === cell 0
import os

if os.environ.get("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "").lower() == "python":
    os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)

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

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

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




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
models = []
image_size = 512
n_splits = 2
model_name = "effnetb0"

fold_name = "-fold.weights.h5"

INPUT_DIR = "../input/cassava-leaf-disease-classification"
TRAIN_CSV = os.path.join(INPUT_DIR, "train.csv")
SAMPLE_SUB = os.path.join(INPUT_DIR, "sample_submission.csv")

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
BATCH_SIZE = 16  # unchanged
EPOCHS = 3  # unchanged

USE_DISK_CACHE = False

DATA_OPTS = tf.data.Options()
DATA_OPTS.experimental_deterministic = True

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


@tf.function(reduce_retracing=True)
def _decode_resize_preprocess(img_bytes):
    img = tf.image.decode_jpeg(
        img_bytes, channels=3, fancy_upscaling=False, dct_method="INTEGER_FAST"
    )
    img = tf.image.resize(
        img,
        [image_size, image_size],
        method=tf.image.ResizeMethod.BILINEAR,
        antialias=False,
    )
    img = preprocess_input(img)
    img.set_shape([image_size, image_size, 3])
    return img


@tf.function(reduce_retracing=True)
def parse_train_example(example_proto):
    ex = tf.io.parse_single_example(example_proto, _TFREC_FEATURES)
    img = _decode_resize_preprocess(ex["image"])
    label = tf.cast(ex["target"], tf.int32)
    return img, tf.one_hot(label, depth=num_classes)


@tf.function(reduce_retracing=True)
def parse_test_example_with_name(example_proto):
    ex = tf.io.parse_single_example(example_proto, _TFREC_FEATURES)
    img = _decode_resize_preprocess(ex["image"])
    name = ex["image_name"]
    return img, name


def make_train_dataset_from_tfrecs(tfrecs, training=True, cache_path=None):
    ds = tf.data.TFRecordDataset(
        tfrecs,
        num_parallel_reads=AUTOTUNE,
        compression_type=_TFREC_COMPRESSION,
    ).with_options(DATA_OPTS)

    if training:
        ds = ds.shuffle(8192, seed=SEED, reshuffle_each_iteration=True)

    ds = ds.map(parse_train_example, num_parallel_calls=AUTOTUNE, deterministic=True)

    if (cache_path is not None) and USE_DISK_CACHE:
        ds = ds.cache(cache_path)
    elif not training:
        ds = ds.cache()

    ds = ds.batch(BATCH_SIZE, drop_remainder=False)

    try:
        ds = ds.apply(
            tf.data.experimental.prefetch_to_device("/GPU:0", buffer_size=AUTOTUNE)
        )
    except Exception:
        ds = ds.prefetch(AUTOTUNE)
    return ds


def make_test_dataset_from_tfrecs_with_names(tfrecs, cache_in_memory=True):
    ds = tf.data.TFRecordDataset(
        tfrecs,
        num_parallel_reads=AUTOTUNE,
        compression_type=_TFREC_COMPRESSION,
    ).with_options(DATA_OPTS)

    ds = ds.map(
        parse_test_example_with_name, num_parallel_calls=AUTOTUNE, deterministic=True
    )

    if cache_in_memory:
        ds = ds.cache()

    ds = ds.batch(BATCH_SIZE, drop_remainder=False)

    try:
        ds = ds.apply(
            tf.data.experimental.prefetch_to_device("/GPU:0", buffer_size=AUTOTUNE)
        )
    except Exception:
        ds = ds.prefetch(AUTOTUNE)
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

imgs_per_shard = 1338
n_shards = len(TRAIN_TFRECS)
n_imgs = len(train_df)
expected = n_shards * imgs_per_shard
if expected < n_imgs:
    raise RuntimeError(
        f"TFRecord shard size assumption failed: n_shards*1338={expected} < n_imgs={n_imgs}"
    )

row_idx = np.arange(n_imgs, dtype=np.int32)
shard_idx = (row_idx // imgs_per_shard).astype(np.int32)
shard_idx = np.minimum(shard_idx, n_shards - 1)

labels = train_df["label"].to_numpy(np.int32, copy=False)
flat = shard_idx.astype(np.int64) * int(num_classes) + labels.astype(np.int64)
counts = np.bincount(flat, minlength=n_shards * int(num_classes)).reshape(
    n_shards, num_classes
)
tfrec_major_labels = counts.argmax(axis=1).astype(np.int32, copy=False)

saved_paths = []
for fold, (tr_shard_idx, va_shard_idx) in enumerate(
    skf.split(np.arange(len(TRAIN_TFRECS)), tfrec_major_labels), start=1
):
    tr_tfrecs = [TRAIN_TFRECS[i] for i in tr_shard_idx]
    va_tfrecs = [TRAIN_TFRECS[i] for i in va_shard_idx]

    tr_cache = f"./cache_train_fold{fold}"
    va_cache = f"./cache_val_fold{fold}"
    tr_ds = make_train_dataset_from_tfrecs(
        tr_tfrecs, training=True, cache_path=tr_cache
    )
    va_ds = make_train_dataset_from_tfrecs(
        va_tfrecs, training=False, cache_path=va_cache
    )

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

    tr_steps = int(tf.data.experimental.cardinality(tr_ds).numpy())
    va_steps = int(tf.data.experimental.cardinality(va_ds).numpy())

    model.fit(
        tr_ds,
        validation_data=va_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        steps_per_epoch=tr_steps,
        validation_steps=va_steps,
        verbose=1,
    )

    model.load_weights(ckpt_path)
    models.append(model)
    saved_paths.append(ckpt_path)

print("Saved fold model weights:", saved_paths)
print("Ensemble size:", len(models))




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1019327440.py in <cell line: 0>()
     59     va_steps = int(tf.data.experimental.cardinality(va_ds).numpy())
     60 
---> 61     model.fit(
     62         tr_ds,
     63         validation_data=va_ds,

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/utils/progbar.py in update(self, current, values, finalize)
    117 
    118             if self.target is not None:
--> 119                 numdigits = int(math.log10(self.target)) + 1
    120                 bar = ("%" + str(numdigits) + "d/%d") % (current, self.target)
    121                 bar = f"\x1b[1m{bar}\x1b[0m "

ValueError: math domain error

## === cell 4
test_ds = make_test_dataset_from_tfrecs_with_names(TEST_TFRECS, cache_in_memory=False)

majority_label = int(train_df["label"].value_counts().idxmax())

if len(models) == 0:
    name_chunks = []
    for _, batch_names in test_ds:
        name_chunks.append(batch_names.numpy())
    image_names_all = np.concatenate(name_chunks).astype("S").astype(str).tolist()
    pred_labels = np.full((len(image_names_all),), majority_label, dtype=int)
else:
    name_chunks = []
    sum_probs_chunks = []

    for batch_imgs, batch_names in test_ds:
        name_chunks.append(batch_names.numpy())

        batch_sum = None
        for m in models:
            probs = m(batch_imgs, training=False).numpy().astype(np.float32, copy=False)
            if batch_sum is None:
                batch_sum = probs
            else:
                batch_sum += probs
        sum_probs_chunks.append(batch_sum)

    image_names_all = np.concatenate(name_chunks).astype("S").astype(str).tolist()
    sum_probs = np.concatenate(sum_probs_chunks, axis=0)
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
