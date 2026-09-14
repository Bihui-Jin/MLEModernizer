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

0.4333635539437896

# 6. Current score

0.63901

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.63864) has done: 'I remove the expensive 512×512 resize by switching to 224×224 (this preserves the same model topology and training loop while cutting compute by ~5×), and I streamline the test pipeline to avoid iterating the TFRecord dataset twice (names and images) which was doubling decode/resize cost. I also eliminate the unused pandas CSV train/val split work (the TFRecord-based split is what’s actually used) to save startup time and memory. Finally, I keep determinism/seeds intact while enabling more efficient `tf.data` graph execution (cache test decoding, fuse maps, and avoid Python-side concatenation where possible) without changing evaluation semantics.'
- What this solution (achieved 0.63901) has done: 'The crash happens before any training because the Kaggle TFRecord reader triggers a known protobuf/TensorFlow compatibility issue (`MessageFactory.GetPrototype`). The smallest robust fix in this environment is to avoid TFRecords entirely and switch the input pipeline to read JPEGs from `train_images/` and `test_images/`, while keeping the same model, loss, optimizer, epochs, batch size, and augmentation semantics. I also ensure the train/val split is deterministic and stratified (so score should remain stable and typically improve slightly vs a file-based TFRecord split), and I keep the submission format exactly `image_id,label` written to `submission.csv`. All paths remain under `/kaggle/input/cassava-leaf-disease-classification`, and the script runs end-to-end without relying on TFRecord/protobuf.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

tf.config.optimizer.set_jit(True)

IMAGE_SIZE = (224, 224)

BATCH_SIZE = 8
EPOCHS = 2  # keep identical training schedule

BASE_DIR = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_CSV = os.path.join(BASE_DIR, "train.csv")
SAMPLE_SUB = os.path.join(BASE_DIR, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(BASE_DIR, "train_images")
TEST_IMG_DIR = os.path.join(BASE_DIR, "test_images")


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

AUTOTUNE = tf.data.AUTOTUNE

DATASET_OPTIONS = tf.data.Options()
DATASET_OPTIONS.experimental_deterministic = False
DATASET_OPTIONS.experimental_optimization.apply_default_optimizations = True
DATASET_OPTIONS.experimental_optimization.map_parallelization = True
DATASET_OPTIONS.experimental_optimization.parallel_batch = True
DATASET_OPTIONS.threading.private_threadpool_size = 0
DATASET_OPTIONS.threading.max_intra_op_parallelism = 0

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

train_df = pd.read_csv(TRAIN_CSV)
train_df["label"] = train_df["label"].astype(int)
assert train_df["label"].between(0, 4).all()

val_frac = 0.1

idx_all = np.arange(len(train_df))
train_idx_parts = []
val_idx_parts = []
for c in range(5):
    idx_c = idx_all[train_df["label"].values == c]
    rng = np.random.RandomState(SEED + c)
    rng.shuffle(idx_c)
    n_val_c = max(1, int(round(len(idx_c) * val_frac)))
    val_idx_parts.append(idx_c[:n_val_c])
    train_idx_parts.append(idx_c[n_val_c:])

val_idx = np.concatenate(val_idx_parts)
train_idx = np.concatenate(train_idx_parts)

rng = np.random.RandomState(SEED)
rng.shuffle(train_idx)
rng.shuffle(val_idx)

trn_df = train_df.iloc[train_idx].reset_index(drop=True)
val_df = train_df.iloc[val_idx].reset_index(drop=True)

print("Train/Val sizes:", len(trn_df), len(val_df))
print(
    "Train label distribution:\n",
    trn_df["label"].value_counts(normalize=True).sort_index(),
)
print(
    "Val label distribution:\n",
    val_df["label"].value_counts(normalize=True).sort_index(),
)


@tf.function
def _decode_resize_normalize_from_path(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(
        img_bytes, channels=3, dct_method="INTEGER_FAST", ratio=2
    )
    img = tf.image.resize(img, IMAGE_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32) / 255.0
    return img


@tf.function
def _load_train_example(image_id, label):
    path = tf.strings.join([TRAIN_IMG_DIR, "/", image_id])
    img = _decode_resize_normalize_from_path(path)
    label = tf.cast(label, tf.int32)
    return img, tf.one_hot(label, depth=5)


@tf.function
def _load_test_example(image_id):
    path = tf.strings.join([TEST_IMG_DIR, "/", image_id])
    img = _decode_resize_normalize_from_path(path)
    return img, image_id


@tf.function
def augment(img, label):
    img = tf.image.random_flip_left_right(img, seed=SEED)
    pad = 16
    img = tf.image.resize_with_crop_or_pad(
        img, IMAGE_SIZE[0] + pad, IMAGE_SIZE[1] + pad
    )
    img = tf.image.random_crop(img, size=[IMAGE_SIZE[0], IMAGE_SIZE[1], 3], seed=SEED)
    return img, label


def _make_train_dataset(df):
    ds = tf.data.Dataset.from_tensor_slices((df["image_id"].values, df["label"].values))
    ds = ds.with_options(DATASET_OPTIONS)
    ds = ds.shuffle(
        buffer_size=min(len(df), 8192), seed=SEED, reshuffle_each_iteration=True
    )
    ds = ds.map(_load_train_example, num_parallel_calls=AUTOTUNE)
    ds = ds.map(augment, num_parallel_calls=AUTOTUNE)
    ds = ds.repeat()
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def _make_val_dataset(df):
    ds = tf.data.Dataset.from_tensor_slices((df["image_id"].values, df["label"].values))
    ds = ds.with_options(DATASET_OPTIONS)
    ds = ds.map(_load_train_example, num_parallel_calls=AUTOTUNE)
    ds = ds.cache()
    ds = ds.repeat()
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_ds = _make_train_dataset(trn_df)
val_ds = _make_val_dataset(val_df)

steps_per_epoch = int(np.ceil(len(trn_df) / BATCH_SIZE))
validation_steps = int(np.ceil(len(val_df) / BATCH_SIZE))

print("steps_per_epoch:", steps_per_epoch, "validation_steps:", validation_steps)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
num_classes = 5

inputs = keras.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
x = layers.Conv2D(16, 3, padding="same", activation="relu")(inputs)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
x = layers.MaxPooling2D()(x)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)

model = keras.Model(inputs, outputs)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()



## === cell 2
history = model.fit(
    train_ds,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_ds,
    validation_steps=validation_steps,
    verbose=1,
)



## === cell 3
test_df = pd.read_csv(SAMPLE_SUB)

test_ds = tf.data.Dataset.from_tensor_slices(test_df["image_id"].values)
test_ds = test_ds.with_options(DATASET_OPTIONS)
test_ds = test_ds.map(_load_test_example, num_parallel_calls=AUTOTUNE).cache()
batched = test_ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)

probs = model.predict(
    batched.map(lambda img, name: img, num_parallel_calls=AUTOTUNE), verbose=0
)

names_batches = []
for _, name_b in batched.as_numpy_iterator():
    names_batches.append(name_b)
names = np.concatenate(names_batches, axis=0)

if names.dtype.kind in ("S",):  # fixed-width bytes
    names = np.char.decode(names, "utf-8")
else:
    names = names.astype("U")

pred_labels = np.argmax(probs, axis=1).astype(np.int32)

name_to_label = dict(zip(names.tolist(), pred_labels.tolist()))
ordered_labels = test_df["image_id"].map(name_to_label).astype(int).to_numpy()
assert len(ordered_labels) == len(test_df)

submission = pd.DataFrame(
    {"image_id": test_df["image_id"].values, "label": ordered_labels}
)
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
print("Submission columns:", submission.columns.tolist())
