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

3.12

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
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import json
import random
import datetime

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import efficientnet_v2

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

try:
    _CPU = os.cpu_count() or 8
    tf.config.threading.set_intra_op_parallelism_threads(max(1, _CPU // 2))
    tf.config.threading.set_inter_op_parallelism_threads(max(1, min(4, _CPU // 4)))
except Exception:
    pass

print("TF:", tf.__version__)
print("Eager:", tf.executing_eagerly())



## === cell 1
WORK_DIR = "/kaggle/input/cassava-leaf-disease-classification/"

train_csv_path = os.path.join(WORK_DIR, "train.csv")
sample_sub_path = os.path.join(WORK_DIR, "sample_submission.csv")
train_img_dir = os.path.join(WORK_DIR, "train_images")
test_img_dir = os.path.join(WORK_DIR, "test_images")

train_tfrec_dir = os.path.join(WORK_DIR, "train_tfrecords")
test_tfrec_dir = os.path.join(WORK_DIR, "test_tfrecords")

assert os.path.exists(sample_sub_path), sample_sub_path
assert os.path.isdir(test_img_dir), test_img_dir
assert os.path.isdir(test_tfrec_dir), test_tfrec_dir
assert os.path.isdir(train_tfrec_dir), train_tfrec_dir

sample_sub = pd.read_csv(sample_sub_path)

print(sample_sub.shape, sample_sub.columns.tolist())

NUM_CLASSES = 5
print("NUM_CLASSES:", NUM_CLASSES)




## === cell 2
class SigmoidFocalCrossEntropy(tf.keras.losses.Loss):
    def __init__(self, alpha=0.25, gamma=2.0, from_logits=False, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        if self.from_logits:
            y_pred = tf.sigmoid(y_pred)
        y_pred = tf.clip_by_value(
            y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon()
        )
        cross_entropy = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(
            1 - y_pred
        )
        weight = self.alpha * y_true + (1 - self.alpha) * (1 - y_true)
        focal_loss = weight * ((1 - y_pred) ** self.gamma) * cross_entropy
        return tf.reduce_sum(focal_loss, axis=-1)


custom_objects = {"SigmoidFocalCrossEntropy": SigmoidFocalCrossEntropy}



## === cell 3
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 6  # unchanged


def build_model(num_classes=NUM_CLASSES, img_size=IMG_SIZE):
    inputs = keras.Input(shape=(img_size, img_size, 3))
    x = efficientnet_v2.preprocess_input(inputs)
    base = efficientnet_v2.EfficientNetV2B0(
        include_top=False, weights="imagenet", input_tensor=x
    )
    base.trainable = False
    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    return model


_DATA_OPTS = tf.data.Options()
_DATA_OPTS.experimental_deterministic = (
    False  # faster input pipeline; preserves training semantics
)

try:
    _DATA_OPTS.threading.private_threadpool_size = max(4, (os.cpu_count() or 8) // 2)
except Exception:
    pass

_FEATURES = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64, default_value=-1),
    "label": tf.io.FixedLenFeature([], tf.int64, default_value=-1),
}

_TEST_FEATURES = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string, default_value=""),
}


@tf.function(reduce_retracing=True)
def _decode_resize_from_bytes(img_bytes):
    img = tf.image.decode_jpeg(img_bytes, channels=3, dct_method="INTEGER_FAST")
    img = tf.image.resize_with_pad(
        img, IMG_SIZE, IMG_SIZE, method="bilinear", antialias=False
    )
    img = tf.cast(img, tf.float32)
    return img


@tf.function(reduce_retracing=True)
def _to_one_hot(label):
    return tf.one_hot(label, depth=NUM_CLASSES)


@tf.function(reduce_retracing=True)
def _parse_train_example(serialized):
    ex = tf.io.parse_single_example(serialized, _FEATURES)
    img = _decode_resize_from_bytes(ex["image"])
    y_raw = tf.where(ex["target"] >= 0, ex["target"], ex["label"])
    y = _to_one_hot(tf.cast(y_raw, tf.int32))
    return img, y


@tf.function(reduce_retracing=True)
def _parse_test_example(serialized):
    ex = tf.io.parse_single_example(serialized, _TEST_FEATURES)
    img = _decode_resize_from_bytes(ex["image"])
    return img


def _count_examples_from_filenames(tfrec_files):
    total = 0
    for f in tfrec_files:
        base = os.path.basename(f)
        try:
            n_part = base.split("-")[-1].split(".")[0]
            total += int(n_part)
        except Exception:
            return None
    return total


def _make_tfrec_dataset(tfrec_files, parse_fn, shuffle=False, cache=False):
    ds = tf.data.TFRecordDataset(tfrec_files, num_parallel_reads=tf.data.AUTOTUNE)
    ds = ds.with_options(_DATA_OPTS)

    if shuffle:
        ds = ds.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)

    ds = ds.map(parse_fn, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)

    if cache:
        ds = ds.cache()

    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds




## === cell 4
train_files = sorted(tf.io.gfile.glob(os.path.join(train_tfrec_dir, "ld_train*.tfrec")))
if not train_files:
    raise FileNotFoundError(f"No train TFRecords found in {train_tfrec_dir}")

n_files = len(train_files)
n_val = max(1, int(round(0.1 * n_files)))
val_files = train_files[:n_val]
trn_files = train_files[n_val:]

print(
    f"Train TFRecords: {len(trn_files)} files | Val TFRecords: {len(val_files)} files"
)

train_ds = _make_tfrec_dataset(
    trn_files, _parse_train_example, shuffle=True, cache=False
)
val_ds = _make_tfrec_dataset(val_files, _parse_train_example, shuffle=False, cache=True)

model = build_model()

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

n_trn = _count_examples_from_filenames(trn_files)
n_val_ex = _count_examples_from_filenames(val_files)
steps_per_epoch = int(np.ceil(n_trn / BATCH_SIZE)) if n_trn is not None else None
val_steps = int(np.ceil(n_val_ex / BATCH_SIZE)) if n_val_ex is not None else None

print("Estimated train examples:", n_trn, "steps_per_epoch:", steps_per_epoch)
print("Estimated val examples:", n_val_ex, "val_steps:", val_steps)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_steps=val_steps,
    verbose=1,
)



## === cell 5
test_files = tf.io.gfile.glob(os.path.join(test_tfrec_dir, "ld_test*.tfrec"))
test_files = sorted(test_files)
if not test_files:
    raise FileNotFoundError(f"No test TFRecords found in {test_tfrec_dir}")

test_ds = _make_tfrec_dataset(
    test_files, _parse_test_example, shuffle=False, cache=True
)

_n_test = _count_examples_from_filenames(test_files)
test_steps = int(np.ceil(_n_test / BATCH_SIZE)) if _n_test is not None else None
print("Test examples:", _n_test, "test_steps:", test_steps)

pred_probs = model.predict(test_ds, steps=test_steps, verbose=1)
pred_labels = np.argmax(pred_probs, axis=1).astype(int)

pred_labels = pred_labels[: len(sample_sub)]

submission = pd.DataFrame(
    {"image_id": sample_sub["image_id"].values, "label": pred_labels}
)
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
print("File exists:", os.path.exists("submission.csv"))
