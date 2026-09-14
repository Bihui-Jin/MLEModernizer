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
import glob
import json
import shutil
import random
import datetime

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

SEED = 123
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

print("TF:", tf.__version__)




## === cell 1
df = pd.read_csv("../input/cassava-leaf-disease-classification/train.csv")
WORK_DIR = "../input/cassava-leaf-disease-classification/"
OUTPUT_DIR = "./"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

TRAIN_TFRECORDS = sorted(
    glob.glob(os.path.join(WORK_DIR, "train_tfrecords", "*.tfrec"))
)
TEST_TFRECORDS = sorted(glob.glob(os.path.join(WORK_DIR, "test_tfrecords", "*.tfrec")))

if len(TRAIN_TFRECORDS) == 0:
    raise FileNotFoundError(
        "No train tfrecords found at: " + os.path.join(WORK_DIR, "train_tfrecords")
    )
if len(TEST_TFRECORDS) == 0:
    raise FileNotFoundError(
        "No test tfrecords found at: " + os.path.join(WORK_DIR, "test_tfrecords")
    )

print("Found train tfrecords:", len(TRAIN_TFRECORDS))
print("Found test tfrecords:", len(TEST_TFRECORDS))




## === cell 2
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
NUM_CLASSES = 5
VALIDATION_SPLIT = 0.1

AUTO = tf.data.AUTOTUNE

_FEATURE_DESC = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),
}


def _decode_and_resize_jpeg(image_bytes):
    img = tf.image.decode_jpeg(image_bytes, channels=3)
    img = tf.image.resize(img, IMAGE_SIZE, method="bilinear", antialias=False)
    img = tf.cast(img, tf.float32)
    return img


def _parse_train_example(ex):
    x = tf.io.parse_single_example(ex, _FEATURE_DESC)
    img = _decode_and_resize_jpeg(x["image"])
    label = tf.one_hot(
        tf.cast(x["target"], tf.int32), depth=NUM_CLASSES, dtype=tf.float32
    )
    return img, label


def _parse_test_example(ex):
    x = tf.io.parse_single_example(
        ex,
        {
            "image": tf.io.FixedLenFeature([], tf.string),
            "image_name": tf.io.FixedLenFeature([], tf.string),
        },
    )
    img = _decode_and_resize_jpeg(x["image"])
    return img, x["image_name"]


num_shards = len(TRAIN_TFRECORDS)
n_valid_shards = max(1, int(round(num_shards * VALIDATION_SPLIT)))


def _stable_shard_key(path: str) -> int:
    import hashlib

    h = hashlib.md5(path.encode("utf-8")).hexdigest()
    return int(h[:8], 16)


shard_keys = np.array([_stable_shard_key(p) for p in TRAIN_TFRECORDS], dtype=np.uint32)
order = np.argsort(shard_keys, kind="mergesort")
valid_shard_idx = set(order[:n_valid_shards].tolist())
train_files = [p for i, p in enumerate(TRAIN_TFRECORDS) if i not in valid_shard_idx]
valid_files = [p for i, p in enumerate(TRAIN_TFRECORDS) if i in valid_shard_idx]

print(
    f"Shard split -> train shards: {len(train_files)} valid shards: {len(valid_files)}"
)

opts = tf.data.Options()
opts.experimental_deterministic = True
opts.autotune.enabled = True

train_raw = tf.data.TFRecordDataset(train_files, num_parallel_reads=AUTO).with_options(
    opts
)
valid_raw = tf.data.TFRecordDataset(valid_files, num_parallel_reads=AUTO).with_options(
    opts
)

train_parsed = train_raw.map(_parse_train_example, num_parallel_calls=AUTO)
train_cache_path = os.path.join(
    OUTPUT_DIR, f"train_parsed_cache_{IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}.tfdata"
)
train_parsed = train_parsed.cache(train_cache_path)

valid_dataset = (
    valid_raw.map(_parse_train_example, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTO)
)

train_dataset = train_parsed.shuffle(
    2048, seed=SEED, reshuffle_each_iteration=True
).batch(BATCH_SIZE, drop_remainder=False)

n_total = len(df)
n_valid = int(round(n_total * (len(valid_files) / max(1, num_shards))))
n_train = n_total - n_valid
train_steps = int(np.ceil(n_train / BATCH_SIZE))
valid_steps = int(np.ceil(n_valid / BATCH_SIZE))

print("Approx Train size:", n_train, "Approx Valid size:", n_valid)
print("Train steps:", train_steps, "Valid steps:", valid_steps)




## === cell 3
def data_augmentation():
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal", seed=SEED),
            tf.keras.layers.RandomTranslation(0.2, 0.2, fill_mode="reflect", seed=SEED),
            tf.keras.layers.RandomRotation(0.2, fill_mode="reflect", seed=SEED),
            tf.keras.layers.RandomZoom(0.2, seed=SEED),
            tf.keras.layers.RandomContrast(0.2, seed=SEED),
        ]
    )


data_aug = data_augmentation()


@tf.function(reduce_retracing=True)
def _aug_batch(x, y):
    return data_aug(x, training=True), y


@tf.function(reduce_retracing=True)
def _maybe_aug(x, y, do_aug: tf.Tensor):
    x = tf.cond(do_aug, lambda: data_aug(x, training=True), lambda: x)
    return x, y


plain_ds = train_dataset
aug_ds = train_dataset.map(_aug_batch, num_parallel_calls=AUTO)

train_dataset = plain_ds.concatenate(aug_ds).concatenate(aug_ds).prefetch(AUTO)
train_steps = int(np.ceil((n_train * 3) / BATCH_SIZE))

print("augmentation train dataset prepared")
print("Updated train steps (after concat):", train_steps)




## === cell 4
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




## === cell 5
class _FixedHps:
    def __init__(self, learning_rate=1e-3, dropout_rate=0.2, dense_units=256):
        self._vals = {
            "learning_rate": float(learning_rate),
            "dropout_rate": float(dropout_rate),
            "dense_units": int(dense_units),
        }

    def Float(self, name, **kwargs):
        return self._vals[name]

    def Int(self, name, **kwargs):
        return self._vals[name]


best_hps = _FixedHps(learning_rate=1e-3, dropout_rate=0.2, dense_units=256)




## === cell 6
def build_model(hp):
    efficientweight = "/kaggle/input/test55/efficientnetv2s.h5"
    if isinstance(efficientweight, str) and os.path.exists(efficientweight):
        weights_arg = efficientweight
    else:
        weights_arg = "imagenet"

    learning_rate = hp.Float(
        "learning_rate", min_value=1e-4, max_value=1e-2, sampling="log"
    )
    dropout_rate = hp.Float("dropout_rate", min_value=0, max_value=0.5)
    dense_units = hp.Int("dense_units", min_value=128, max_value=512, step=32)

    base = tf.keras.applications.efficientnet_v2.EfficientNetV2S(
        weights=weights_arg, include_top=False, input_shape=(224, 224, 3)
    )
    x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    x = tf.keras.layers.Dense(dense_units, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    outputs = tf.keras.layers.Dense(5, activation="softmax")(x)
    model = tf.keras.Model(inputs=base.input, outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=SigmoidFocalCrossEntropy(alpha=0.25, gamma=2, from_logits=False),
        metrics=[tf.keras.metrics.CategoricalAccuracy(name="accuracy")],
    )
    return model




## === cell 7
def Model_Training(best_hps, train_dataset, valid_dataset, epochs=10):
    model = build_model(best_hps)

    history = model.fit(
        train_dataset,
        epochs=epochs,
        steps_per_epoch=train_steps,
        validation_data=valid_dataset,
        validation_steps=valid_steps,
        callbacks=[],
        verbose=2,
    )

    try:
        plt.plot(history.history["accuracy"])
        plt.plot(history.history["val_accuracy"])
        plt.title("Model accuracy")
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch")
        plt.legend(["Train", "Valid"], loc="upper left")
        plt.show()
        plt.clf()

        plt.plot(history.history["loss"])
        plt.plot(history.history["val_loss"])
        plt.title("Model loss")
        plt.ylabel("Loss")
        plt.xlabel("Loss")
        plt.legend(["Train", "Valid"], loc="upper left")
        plt.show()
        plt.clf()
    except Exception as e:
        print("Plotting skipped due to:", repr(e))

    return model


model = Model_Training(best_hps, train_dataset, valid_dataset, epochs=10)




## === cell 8
sample_sub = pd.read_csv(os.path.join(WORK_DIR, "sample_submission.csv"))

test_raw = tf.data.TFRecordDataset(
    TEST_TFRECORDS, num_parallel_reads=AUTO
).with_options(opts)
test_parsed = test_raw.map(_parse_test_example, num_parallel_calls=AUTO)

test_batched = test_parsed.batch(64, drop_remainder=False).prefetch(AUTO)

all_names = []
all_probs = []

for imgs, names in test_batched:
    probs = model(imgs, training=False).numpy()
    all_probs.append(probs)
    all_names.extend([n.decode("utf-8") for n in names.numpy().tolist()])

probs = np.concatenate(all_probs, axis=0)
pred_cls = np.argmax(probs, axis=-1).astype(np.int64)

if len(all_names) != len(pred_cls):
    raise RuntimeError(f"Mismatch names({len(all_names)}) vs preds({len(pred_cls)})")

pred_map = dict(zip(all_names, pred_cls.tolist()))
pred_labels = [int(pred_map[img_id]) for img_id in sample_sub["image_id"].tolist()]

submission = pd.DataFrame({"image_id": sample_sub["image_id"], "label": pred_labels})
submission_path = os.path.join(OUTPUT_DIR, "submission.csv")
submission.to_csv(submission_path, index=False)
print("Wrote submission.csv with shape:", submission.shape, "to:", submission_path)

folder = "/kaggle/working/dataset"
try:
    shutil.rmtree(folder)
    print("Folder Deleted")
except OSError as e:
    print(f"Error: {e}")
