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

2.7

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

0.8634028407373829

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.61099) has done: 'I fix the environment/runtime failures first by removing the TensorFlow import from the initial “version print” cell (it triggers the `MessageFactory.GetPrototype` crash) and making the script self-contained using only the local Cassava dataset. Then I fix the data generator error by converting `train_df["label"]` to string for `class_mode="categorical"`, and I ensure `NUM_CLASSES` is defined before it’s used so the fallback model path runs. Finally, I make inference robust by always producing a prediction array aligned to `sample_submission.csv` and writing `submission.csv` with the required columns and row count.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

os.environ["PYTHONHASHSEED"] = "0"
np.random.seed(0)

print(
    "Running self-contained Cassava pipeline (train small CNN from train_images and infer on test_images)."
)



## === cell 1
"""Robust Bi-Tempered Logistic Loss Based on Bregman Divergences.

Source: https://bit.ly/3jSol8T

NOTE: Preserved for original logic context. Not executed in this runtime.
"""
pass



## === cell 2
"""Gambler's loss helpers (not executed in this runtime)."""
pass



## === cell 3
model_v1_path = "/kaggle/input/only-xception-with-cropping/saved-model-11-0.879"
model_v2_path = "/kaggle/input/gambler-s-loss-cassava/saved-model-05-0.860"
model_v3_path = (
    "/kaggle/input/bitempered-loss-only-xception-with-cropping/saved-model-15-0.839"
)

all_model_paths = [model_v1_path, model_v2_path, model_v3_path]
have_external_models = False
print("External models disabled; training an in-notebook model instead.")



## === cell 4
DATA_DIR = "/kaggle/input/cassava-leaf-disease-classification"
train_csv_path = os.path.join(DATA_DIR, "train.csv")
sample_path = os.path.join(DATA_DIR, "sample_submission.csv")

TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")

train_df = pd.read_csv(train_csv_path)
sample_sub = pd.read_csv(sample_path)

print("Train rows:", len(train_df), "Test rows:", len(sample_sub))
print("Train label distribution:\n", train_df["label"].value_counts().sort_index())

train_df["label"] = train_df["label"].astype(str)
NUM_CLASSES = train_df["label"].nunique()
print("NUM_CLASSES:", NUM_CLASSES)



## === cell 5
import math
import tensorflow as tf
from tensorflow.keras import layers, models

tf.random.set_seed(0)
try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

AUTOTUNE = tf.data.AUTOTUNE

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 3  # unchanged core logic

train_df_shuffled = train_df.sample(frac=1.0, random_state=0).reset_index(drop=True)
val_frac = 0.1
val_size = int(len(train_df_shuffled) * val_frac)
val_df = train_df_shuffled.iloc[:val_size].copy()
tr_df = train_df_shuffled.iloc[val_size:].copy()

classes_sorted = sorted(train_df["label"].unique().tolist())
class_to_index = {c: i for i, c in enumerate(classes_sorted)}
index_to_class = {i: c for c, i in class_to_index.items()}


def _paths_and_labels(df, img_dir):
    paths = (img_dir.rstrip("/") + "/" + df["image_id"].values.astype(str)).tolist()
    labels = df["label"].values.astype(str).tolist()
    return paths, labels


tr_paths, tr_labels_str = _paths_and_labels(tr_df, TRAIN_IMG_DIR)
val_paths, val_labels_str = _paths_and_labels(val_df, TRAIN_IMG_DIR)


def _decode_and_resize(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(
        img, [IMG_SIZE, IMG_SIZE], method=tf.image.ResizeMethod.BILINEAR
    )
    img = tf.cast(img, tf.float32) * (1.0 / 255.0)
    return img


_GLOBAL_SEED = tf.constant([0, 0], dtype=tf.int32)


def _augment_stateless(img, seed):
    img = tf.image.stateless_random_flip_left_right(img, seed=seed)

    max_dx = tf.cast(tf.round(0.05 * IMG_SIZE), tf.int32)
    max_dy = tf.cast(tf.round(0.05 * IMG_SIZE), tf.int32)
    dx = tf.random.stateless_uniform(
        [],
        seed=seed + tf.constant([1, 0], tf.int32),
        minval=-max_dx,
        maxval=max_dx + 1,
        dtype=tf.int32,
    )
    dy = tf.random.stateless_uniform(
        [],
        seed=seed + tf.constant([2, 0], tf.int32),
        minval=-max_dy,
        maxval=max_dy + 1,
        dtype=tf.int32,
    )
    img = tf.roll(img, shift=[dy, dx], axis=[0, 1])

    z = tf.random.stateless_uniform(
        [],
        seed=seed + tf.constant([3, 0], tf.int32),
        minval=0.9,
        maxval=1.1,
        dtype=tf.float32,
    )
    crop_size = tf.cast(tf.round(tf.cast(IMG_SIZE, tf.float32) / z), tf.int32)
    crop_size = tf.clip_by_value(crop_size, 1, IMG_SIZE)
    img = tf.image.resize_with_crop_or_pad(img, crop_size, crop_size)
    img = tf.image.resize(
        img, [IMG_SIZE, IMG_SIZE], method=tf.image.ResizeMethod.BILINEAR
    )

    angle = tf.random.stateless_uniform(
        [],
        seed=seed + tf.constant([4, 0], tf.int32),
        minval=-10.0,
        maxval=10.0,
        dtype=tf.float32,
    ) * (math.pi / 180.0)

    c = tf.cos(angle)
    s = tf.sin(angle)
    cx = (IMG_SIZE - 1) / 2.0
    cy = (IMG_SIZE - 1) / 2.0
    tx = cx - c * cx + s * cy
    ty = cy - s * cx - c * cy
    transform = tf.stack([c, -s, tx, s, c, ty, 0.0, 0.0], axis=0)
    img = tf.raw_ops.ImageProjectiveTransformV3(
        images=tf.expand_dims(img, 0),
        transforms=tf.expand_dims(transform, 0),
        output_shape=tf.constant([IMG_SIZE, IMG_SIZE], dtype=tf.int32),
        fill_mode="REFLECT",
        interpolation="BILINEAR",
        fill_value=0.0,
    )
    img = tf.squeeze(img, 0)
    return img


def _make_train_ds(paths, labels_str):
    labels_idx = [class_to_index[s] for s in labels_str]
    labels_idx = tf.constant(labels_idx, dtype=tf.int32)

    ds = tf.data.Dataset.from_tensor_slices((tf.constant(paths), labels_idx))
    ds = ds.shuffle(buffer_size=len(paths), seed=0, reshuffle_each_iteration=True)

    ds = ds.enumerate()

    def _map_fn(i, elems):
        path, y = elems
        img = _decode_and_resize(path)
        seed = _GLOBAL_SEED + tf.stack([tf.cast(i, tf.int32), 0])
        img = _augment_stateless(img, seed)
        y = tf.one_hot(y, depth=NUM_CLASSES, dtype=tf.float32)
        return img, y

    ds = ds.map(_map_fn, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def _make_val_ds(paths, labels_str):
    labels_idx = [class_to_index[s] for s in labels_str]
    labels_idx = tf.constant(labels_idx, dtype=tf.int32)

    ds = tf.data.Dataset.from_tensor_slices((tf.constant(paths), labels_idx))

    def _map_fn(path, y):
        img = _decode_and_resize(path)
        y = tf.one_hot(y, depth=NUM_CLASSES, dtype=tf.float32)
        return img, y

    ds = ds.map(_map_fn, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_ds = _make_train_ds(tr_paths, tr_labels_str)
val_ds = _make_val_ds(val_paths, val_labels_str)

model = models.Sequential(
    [
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.2),
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ]
)

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

steps_per_epoch = int(math.ceil(len(tr_df) / float(BATCH_SIZE)))
val_steps = int(math.ceil(len(val_df) / float(BATCH_SIZE)))

history = model.fit(
    train_ds,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_ds,
    validation_steps=val_steps,
    verbose=2,
)

test_df = sample_sub[["image_id"]].copy()
test_paths = (
    TEST_IMG_DIR.rstrip("/") + "/" + test_df["image_id"].values.astype(str)
).tolist()

test_ds = tf.data.Dataset.from_tensor_slices(tf.constant(test_paths))


def _map_test(path):
    return _decode_and_resize(path)


test_ds = test_ds.map(_map_test, num_parallel_calls=AUTOTUNE)
test_ds = test_ds.batch(BATCH_SIZE, drop_remainder=False)
test_ds = test_ds.prefetch(AUTOTUNE)

probs = model.predict(test_ds, verbose=1)
pred_idx = np.argmax(probs, axis=1)

pred_labels = np.array([int(index_to_class[i]) for i in pred_idx], dtype=np.int64)
predicted_class_indices_new = pred_labels

print(
    "Pred label distribution (test):",
    pd.Series(predicted_class_indices_new).value_counts().sort_index().to_dict(),
)



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 6
submission = sample_sub[["image_id"]].copy()
submission["label"] = predicted_class_indices_new.astype(int)

out_path = "/kaggle/working/submission.csv"
submission.to_csv(out_path, index=False)

print("Wrote submission:", out_path)
print(submission.head())
print("Rows:", len(submission))
assert out_path.endswith(".csv")
assert list(submission.columns) == ["image_id", "label"]
assert len(submission) == len(sample_sub)

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/97550037.py in <cell line: 0>()
      1 submission = sample_sub[["image_id"]].copy()
----> 2 submission["label"] = predicted_class_indices_new.astype(int)
      3 
      4 out_path = "/kaggle/working/submission.csv"
      5 submission.to_csv(out_path, index=False)

NameError: name 'predicted_class_indices_new' is not defined
