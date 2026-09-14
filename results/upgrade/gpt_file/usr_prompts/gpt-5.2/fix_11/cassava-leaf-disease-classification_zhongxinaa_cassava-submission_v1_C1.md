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
import random
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

print("TF:", tf.__version__)
print("Eager:", tf.executing_eagerly())

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.optimizer.set_jit(True)  # XLA JIT where applicable
except Exception as e:
    print("Warning: could not enable XLA JIT:", e)

try:
    tf.data.experimental.enable_debug_mode(False)
except Exception:
    pass

AUTOTUNE = tf.data.AUTOTUNE
options = tf.data.Options()
try:
    options.experimental_deterministic = False
except Exception:
    try:
        options.deterministic = False
    except Exception:
        pass
try:
    options.experimental_optimization.apply_default_optimizations = True
    options.experimental_optimization.map_parallelization = True
    options.experimental_optimization.parallel_batch = True
    options.experimental_optimization.autotune_buffers = True
    options.experimental_optimization.experimental_slack = True
except Exception:
    pass
try:
    options.experimental_slack = True
except Exception:
    pass




## === cell 1
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



## === cell 2
WORK_DIR = "/kaggle/input/cassava-leaf-disease-classification/"
TRAIN_CSV = os.path.join(WORK_DIR, "train.csv")
SAMPLE_SUB = os.path.join(WORK_DIR, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(WORK_DIR, "train_images")
TEST_IMG_DIR = os.path.join(WORK_DIR, "test_images")

TRAIN_TFREC_DIR = os.path.join(WORK_DIR, "train_tfrecords")
TEST_TFREC_DIR = os.path.join(WORK_DIR, "test_tfrecords")

assert os.path.exists(TRAIN_CSV), TRAIN_CSV
assert os.path.exists(SAMPLE_SUB), SAMPLE_SUB
assert os.path.isdir(TRAIN_IMG_DIR), TRAIN_IMG_DIR
assert os.path.isdir(TEST_IMG_DIR), TEST_IMG_DIR
assert os.path.isdir(TRAIN_TFREC_DIR), TRAIN_TFREC_DIR
assert os.path.isdir(TEST_TFREC_DIR), TEST_TFREC_DIR

train_df = pd.read_csv(TRAIN_CSV, usecols=["image_id", "label"])
sample_df = pd.read_csv(SAMPLE_SUB, usecols=["image_id", "label"])

print(train_df.head())
print(sample_df.head())
print("Train rows:", len(train_df), "Test rows:", len(sample_df))



## === cell 3
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 5
EPOCHS = 3  # keep runtime within limits; enough to produce a non-random submission

from sklearn.model_selection import train_test_split

train_ids = train_df["image_id"].values
train_labels = train_df["label"].astype("int32").values

tr_ids, va_ids, tr_y, va_y = train_test_split(
    train_ids, train_labels, test_size=0.1, random_state=SEED, stratify=train_labels
)

tr_files = np.array([os.path.join(TRAIN_IMG_DIR, f) for f in tr_ids], dtype=object)
va_files = np.array([os.path.join(TRAIN_IMG_DIR, f) for f in va_ids], dtype=object)
tr_y = tr_y.astype(np.int32, copy=False)
va_y = va_y.astype(np.int32, copy=False)


@tf.function
def _decode_resize_from_path(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, IMG_SIZE, method="bilinear")
    img = tf.cast(img, tf.float32)
    return img


@tf.function
def _augment_stateless_lr_flip(img, image_id):
    h = tf.strings.to_hash_bucket_fast(image_id, 2**31 - 1)
    seed = tf.stack([tf.cast(SEED, tf.int32), tf.cast(h, tf.int32)])
    return tf.image.stateless_random_flip_left_right(img, seed=seed)


@tf.function
def _train_decode_and_aug(path, label):
    image_id = tf.strings.split(path, os.sep)[-1]
    img = _decode_resize_from_path(path)
    img = _augment_stateless_lr_flip(img, image_id)
    return img, tf.cast(label, tf.int32)


@tf.function
def _eval_decode(path, label):
    img = _decode_resize_from_path(path)
    return img, tf.cast(label, tf.int32)


def _make_train_ds(files, labels):
    ds = tf.data.Dataset.from_tensor_slices((files, labels)).with_options(options)
    ds = ds.map(_eval_decode, num_parallel_calls=AUTOTUNE).cache()
    ds = ds.shuffle(4096, seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(
        lambda img, y: (_augment_stateless_lr_flip(img, tf.constant("", tf.string)), y),
        num_parallel_calls=AUTOTUNE,
    )
    ds = tf.data.Dataset.from_tensor_slices((files, labels)).with_options(options)
    ds = ds.map(
        lambda p, y: (_decode_resize_from_path(p), tf.strings.split(p, os.sep)[-1], y),
        num_parallel_calls=AUTOTUNE,
    ).cache()
    ds = ds.shuffle(4096, seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(
        lambda img, image_id, y: (
            _augment_stateless_lr_flip(img, image_id),
            tf.cast(y, tf.int32),
        ),
        num_parallel_calls=AUTOTUNE,
    )
    ds = (
        ds.batch(BATCH_SIZE, drop_remainder=True)
        .prefetch(AUTOTUNE)
        .with_options(options)
    )
    return ds


def _make_val_ds(files, labels):
    ds = tf.data.Dataset.from_tensor_slices((files, labels)).with_options(options)
    ds = ds.map(_eval_decode, num_parallel_calls=AUTOTUNE).cache()
    ds = (
        ds.batch(BATCH_SIZE, drop_remainder=False)
        .prefetch(AUTOTUNE)
        .with_options(options)
    )
    return ds


tr_ds = _make_train_ds(tr_files, tr_y)
val_ds = _make_val_ds(va_files, va_y)

steps_per_epoch = len(tr_ids) // BATCH_SIZE
validation_steps = int(np.ceil(len(va_ids) / BATCH_SIZE))

base = tf.keras.applications.EfficientNetV2B0(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
)
base.trainable = False  # first train only the head for speed/stability

inputs = keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = inputs
x = tf.keras.applications.efficientnet_v2.preprocess_input(x)
x = base(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2, seed=SEED)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()



## === cell 4
history = model.fit(
    tr_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    verbose=2,
)

base.trainable = True
for layer in base.layers[:-40]:
    layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

history_ft = model.fit(
    tr_ds,
    validation_data=val_ds,
    epochs=1,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    verbose=2,
)



## === cell 5
test_image_ids = sample_df["image_id"].values
test_files = np.array(
    [os.path.join(TEST_IMG_DIR, f) for f in test_image_ids], dtype=object
)


@tf.function
def _test_decode(path):
    img = _decode_resize_from_path(path)
    return img


test_ds = tf.data.Dataset.from_tensor_slices(test_files).with_options(options)
test_ds = test_ds.map(_test_decode, num_parallel_calls=AUTOTUNE).cache()
test_ds = test_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE).with_options(options)

pred_steps = int(np.ceil(len(test_image_ids) / BATCH_SIZE))
pred_probs = model.predict(test_ds, verbose=1, steps=pred_steps)
pred_labels = np.argmax(pred_probs, axis=1).astype(int)

submission = pd.DataFrame(
    {"image_id": test_image_ids, "label": pred_labels[: len(test_image_ids)]}
)
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with rows:", len(submission))
assert os.path.exists("submission.csv")
