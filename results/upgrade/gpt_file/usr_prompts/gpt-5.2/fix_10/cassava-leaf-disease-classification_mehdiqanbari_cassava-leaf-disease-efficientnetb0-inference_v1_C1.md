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

3.14

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

import random
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

AUTOTUNE = tf.data.AUTOTUNE

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

print("TF version:", tf.__version__)



## === cell 1
BATCH_SIZE = 16
IMG_SIZE = (224, 224)
EPOCHS = 5  # Keep runtime reasonable while producing a non-random, trainable model.

DATA_ROOT = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
TRAIN_DIR = os.path.join(DATA_ROOT, "train_images")
TEST_DIR = os.path.join(DATA_ROOT, "test_images")
SAMPLE_SUB = os.path.join(DATA_ROOT, "sample_submission.csv")

TRAIN_TFREC_DIR = os.path.join(DATA_ROOT, "train_tfrecords")
TEST_TFREC_DIR = os.path.join(DATA_ROOT, "test_tfrecords")




## === cell 2
def build_model():
    base_model = EfficientNetB0(weights=None, include_top=False)

    model = Sequential(
        [
            tf.keras.layers.Input(shape=(*IMG_SIZE, 3)),
            base_model,
            GlobalAveragePooling2D(),
            Dense(5, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
        jit_compile=True,
    )
    return model


model = build_model()
model.summary()



## === cell 3
train_df = pd.read_csv(TRAIN_CSV)

num_classes = 5

_FEATURES_LABELED_LABEL = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "label": tf.io.FixedLenFeature([], tf.int64),
}
_FEATURES_LABELED_TARGET = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),
}
_FEATURES_UNLABELED = {
    "image": tf.io.FixedLenFeature([], tf.string),
}


@tf.function
def _decode_and_resize_from_bytes(img_bytes):
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32) / 255.0
    return img


def _list_tfrec_files(tfrec_dir):
    files = tf.io.gfile.glob(os.path.join(tfrec_dir, "*.tfrec"))
    files = sorted(files)
    if not files:
        raise FileNotFoundError(f"No .tfrec files found in: {tfrec_dir}")
    return files


def _detect_label_key(tfrec_files):
    probe = tf.data.TFRecordDataset(tfrec_files[:1]).take(1)
    for rec in probe:
        try:
            ex = tf.io.parse_single_example(rec, _FEATURES_LABELED_TARGET)
            _ = ex["target"]
            return "target"
        except Exception:
            pass
        try:
            ex = tf.io.parse_single_example(rec, _FEATURES_LABELED_LABEL)
            _ = ex["label"]
            return "label"
        except Exception:
            pass
    raise ValueError(
        "Could not detect label key in TFRecord (neither 'target' nor 'label')."
    )


train_tfrec_files = _list_tfrec_files(TRAIN_TFREC_DIR)
LABEL_KEY = _detect_label_key(train_tfrec_files)
print("Detected TFRecord label key:", LABEL_KEY)


@tf.function
def _parse_labeled_tfrecord(example_proto):
    if LABEL_KEY == "target":
        ex = tf.io.parse_single_example(example_proto, _FEATURES_LABELED_TARGET)
        y_raw = ex["target"]
    else:
        ex = tf.io.parse_single_example(example_proto, _FEATURES_LABELED_LABEL)
        y_raw = ex["label"]
    x = _decode_and_resize_from_bytes(ex["image"])
    y = tf.one_hot(tf.cast(y_raw, tf.int32), depth=num_classes)
    return x, y


@tf.function
def _parse_unlabeled_tfrecord(example_proto):
    ex = tf.io.parse_single_example(example_proto, _FEATURES_UNLABELED)
    x = _decode_and_resize_from_bytes(ex["image"])
    return x


def _make_tfrec_ds(tfrec_files, labeled, training=False):
    ds = tf.data.TFRecordDataset(tfrec_files, num_parallel_reads=AUTOTUNE)

    if labeled:
        ds = ds.map(
            _parse_labeled_tfrecord, num_parallel_calls=AUTOTUNE, deterministic=True
        )
    else:
        ds = ds.map(
            _parse_unlabeled_tfrecord, num_parallel_calls=AUTOTUNE, deterministic=True
        )

    if training:
        ds = ds.shuffle(buffer_size=8192, seed=SEED, reshuffle_each_iteration=True)

    ds = ds.batch(BATCH_SIZE, drop_remainder=False)

    opts = tf.data.Options()
    opts.experimental_slack = True
    ds = ds.with_options(opts)

    ds = ds.prefetch(AUTOTUNE)
    return ds


rng = np.random.RandomState(SEED)
perm_files = rng.permutation(len(train_tfrec_files))
val_size_files = max(1, int(round(0.1 * len(train_tfrec_files))))
val_tfrec_files = [train_tfrec_files[i] for i in perm_files[:val_size_files]]
trn_tfrec_files = [train_tfrec_files[i] for i in perm_files[val_size_files:]]

train_ds = _make_tfrec_ds(trn_tfrec_files, labeled=True, training=True)
val_ds = _make_tfrec_ds(val_tfrec_files, labeled=True, training=False)


def _shard_counts(total_examples: int, num_shards: int) -> np.ndarray:
    q, r = divmod(int(total_examples), int(num_shards))
    counts = np.full((num_shards,), q, dtype=np.int64)
    if r:
        counts[:r] += 1
    return counts


total_train = int(len(train_df))
num_shards = int(len(train_tfrec_files))
counts_per_file = _shard_counts(total_train, num_shards)

val_idx = perm_files[:val_size_files].astype(np.int64)
trn_idx = perm_files[val_size_files:].astype(np.int64)

n_train = int(counts_per_file[trn_idx].sum())
n_val = int(counts_per_file[val_idx].sum())

steps_per_epoch = int(np.ceil(n_train / BATCH_SIZE))
validation_steps = int(np.ceil(n_val / BATCH_SIZE))

ckpt_path = "/kaggle/working/best_model.weights.h5"
callbacks = [
    ModelCheckpoint(
        ckpt_path,
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        save_weights_only=True,
        verbose=1,
    ),
    ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=1, min_lr=1e-6, verbose=1
    ),
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
)

if os.path.exists(ckpt_path):
    model.load_weights(ckpt_path)



## === cell 4
sample_sub = pd.read_csv(SAMPLE_SUB)
test_files = sample_sub["image_id"].astype(str).tolist()


def _make_test_ds():
    tfrec_files = _list_tfrec_files(TEST_TFREC_DIR)

    detected_name_key = None
    probe_ds = tf.data.TFRecordDataset(tfrec_files[:1]).take(1)
    for r in probe_ds:
        for cand in ("image_name", "image_id"):
            try:
                _probe_features = {
                    "image": tf.io.FixedLenFeature([], tf.string),
                    cand: tf.io.FixedLenFeature([], tf.string),
                }
                _ = tf.io.parse_single_example(r, _probe_features)[cand]
                detected_name_key = cand
                break
            except Exception:
                continue

    if detected_name_key is not None:
        features = {
            "image": tf.io.FixedLenFeature([], tf.string),
            detected_name_key: tf.io.FixedLenFeature([], tf.string),
        }

        @tf.function
        def _parse_test(example_proto):
            ex = tf.io.parse_single_example(example_proto, features)
            x = _decode_and_resize_from_bytes(ex["image"])
            return ex[detected_name_key], x

        ds = tf.data.TFRecordDataset(tfrec_files, num_parallel_reads=AUTOTUNE).map(
            _parse_test, num_parallel_calls=AUTOTUNE, deterministic=True
        )
        return ds.batch(BATCH_SIZE).prefetch(AUTOTUNE), True

    test_paths = np.array([os.path.join(TEST_DIR, f) for f in test_files], dtype=object)

    @tf.function
    def _decode_and_resize_path(path):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR)
        img = tf.cast(img, tf.float32) / 255.0
        return img

    ds = tf.data.Dataset.from_tensor_slices(test_paths).map(
        _decode_and_resize_path, num_parallel_calls=AUTOTUNE, deterministic=True
    )
    ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return ds, False


test_ds, test_has_name = _make_test_ds()

if test_has_name:
    names_collected = []
    preds_collected = []

    for names, imgs in test_ds:
        batch_preds = model.predict_on_batch(imgs)
        preds_collected.append(batch_preds)
        names_collected.extend(names.numpy().astype("U"))

    predictions = np.concatenate(preds_collected, axis=0)
    pred_labels = np.argmax(predictions, axis=1).astype(int)

    order = {img_id: i for i, img_id in enumerate(names_collected)}
    idx = np.fromiter(
        (order[f] for f in test_files), dtype=np.int64, count=len(test_files)
    )
    predicted_labels = pred_labels[idx]
else:
    predictions = model.predict(test_ds, verbose=1)
    predicted_labels = np.argmax(predictions, axis=1).astype(int)

assert len(predicted_labels) == len(sample_sub), (
    len(predicted_labels),
    len(sample_sub),
)

sub = sample_sub.copy()
sub["label"] = predicted_labels.astype(int)

sub.to_csv("submission.csv", index=False)
print("Submission file created: submission.csv")
print(sub.head())
