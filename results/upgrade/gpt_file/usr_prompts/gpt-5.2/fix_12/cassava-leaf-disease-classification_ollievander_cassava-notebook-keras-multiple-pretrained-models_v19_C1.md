# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.8807796917497733

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

INPUT_DIR = "../input/cassava-leaf-disease-classification/"
OUTPUT_DIR = "./"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAIN_PATH = os.path.join(INPUT_DIR, "train_images")
TEST_PATH = os.path.join(INPUT_DIR, "test_images")

print("TRAIN_PATH exists:", os.path.exists(TRAIN_PATH))
print("TEST_PATH exists:", os.path.exists(TEST_PATH))
print("INPUT_DIR exists:", os.path.exists(INPUT_DIR))




## === cell 1
import json
import math
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

SEED = 100
tf.random.set_seed(SEED)
np.random.seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.data.experimental.enable_debug_mode = False  # no-op on older TF, safe if present
except Exception:
    pass

print("TensorFlow:", tf.__version__)




## === cell 2
train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv"))
sample_sub = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv"))

print(train.shape, sample_sub.shape)
train.head()




## === cell 3
with open(os.path.join(INPUT_DIR, "label_num_to_disease_map.json")) as f:
    classes = json.load(f)

train["class"] = train["label"].apply(lambda x: classes[str(x)])
train[["image_id", "label", "class"]].head()




## === cell 4
from sklearn.model_selection import train_test_split

train = train.copy()
train["path"] = train["image_id"].apply(lambda x: os.path.join(TRAIN_PATH, str(x)))

train = train.astype({"image_id": "str", "label": "str", "path": "str", "class": "str"})

train_df, val_df = train_test_split(
    train, test_size=0.05, random_state=SEED, stratify=train["label"].values
)

print("train_df:", train_df.shape, "val_df:", val_df.shape)
train_df.head()




## === cell 5
batch_size = 4
img_size = (512, 512)

label_to_index = {str(i): i for i in range(train["label"].nunique())}
num_classes = len(label_to_index)

AUTO = tf.data.AUTOTUNE

TFREC_TRAIN_DIR = os.path.join(INPUT_DIR, "train_tfrecords")
TFREC_TEST_DIR = os.path.join(INPUT_DIR, "test_tfrecords")

train_tfrecs = tf.io.gfile.glob(os.path.join(TFREC_TRAIN_DIR, "*.tfrec"))
test_tfrecs = tf.io.gfile.glob(os.path.join(TFREC_TEST_DIR, "*.tfrec"))
train_tfrecs = sorted(train_tfrecs)
test_tfrecs = sorted(test_tfrecs)

print("Found train tfrecs:", len(train_tfrecs), "test tfrecs:", len(test_tfrecs))

TFREC_FEATURES = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),
}


def _parse_train_min(example_proto):
    ex = tf.io.parse_single_example(example_proto, TFREC_FEATURES)
    return ex["image"], ex["image_name"], ex["target"]


@tf.function
def _decode_train(image_bytes, image_name, target):
    img = tf.image.decode_jpeg(image_bytes, channels=3)
    img = tf.image.resize(img, img_size, method="bilinear")
    img = tf.cast(img, tf.float32)  # keep 0..255, model has Rescaling
    label_int = tf.cast(target, tf.int32)
    y = tf.one_hot(label_int, depth=num_classes, dtype=tf.float32)
    return img, y


@tf.function
def _parse_and_decode_train(example_proto):
    ex = tf.io.parse_single_example(example_proto, TFREC_FEATURES)
    return _decode_train(ex["image"], ex["image_name"], ex["target"])


@tf.function
def _parse_test_example(example_proto):
    ex = tf.io.parse_single_example(example_proto, TFREC_FEATURES)
    img = tf.image.decode_jpeg(ex["image"], channels=3)
    img = tf.image.resize(img, img_size, method="bilinear")
    img = tf.cast(img, tf.float32)  # keep 0..255, model has Rescaling
    image_id = ex["image_name"]
    return img, image_id


train_paths = train_df["path"].values
val_paths = val_df["path"].values

print("num_classes:", num_classes)


def _decode_image_from_path(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, img_size, method="bilinear")
    img = tf.cast(img, tf.float32)  # keep 0..255, model has Rescaling
    return img


def _make_tfrecord_train_val_datasets(train_tfrecs, val_tfrecs, batch_size):
    def _read_tfrecs(filenames, training):
        opts = tf.data.Options()
        opts.experimental_deterministic = True

        ds = tf.data.Dataset.from_tensor_slices(filenames)

        if training:
            ds = ds.shuffle(len(filenames), seed=SEED, reshuffle_each_iteration=True)

        ds = ds.interleave(
            lambda fn: tf.data.TFRecordDataset(fn, num_parallel_reads=AUTO),
            cycle_length=min(8, len(filenames)) if len(filenames) else 1,
            num_parallel_calls=AUTO,
            deterministic=True,
        )
        ds = ds.with_options(opts)

        if training:
            ds = ds.shuffle(8192, seed=SEED, reshuffle_each_iteration=True)

        ds = ds.map(
            _parse_and_decode_train, num_parallel_calls=AUTO, deterministic=True
        )

        if not training:
            ds = ds.cache()

        ds = ds.batch(batch_size, drop_remainder=False)
        ds = ds.prefetch(AUTO)
        return ds

    train_ds = _read_tfrecs(train_tfrecs, training=True)
    val_ds = _read_tfrecs(val_tfrecs, training=False)
    return train_ds, val_ds


n_train_files = len(train_tfrecs)
val_file_count = max(1, int(round(0.05 * n_train_files))) if n_train_files else 0
val_tfrecs = train_tfrecs[:val_file_count]
train_tfrecs_used = train_tfrecs[val_file_count:]

print(
    "TFRecord split -> train files:",
    len(train_tfrecs_used),
    "val files:",
    len(val_tfrecs),
)

train_ds, val_ds = _make_tfrecord_train_val_datasets(
    train_tfrecs_used, val_tfrecs, batch_size=batch_size
)




## === cell 6
inputs = keras.Input(shape=(img_size[0], img_size[1], 3))

x = layers.Rescaling(1.0 / 255)(inputs)

aug = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.10),
    ],
    name="augmentation",
)
x = aug(x)

x = layers.Conv2D(16, 3, padding="same", activation="relu")(x)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)

model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
    jit_compile=True,
)
model.summary()




## === cell 7
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=2,
    verbose=1,
)




## === cell 8
TEST_DIR = TEST_PATH
test_images = sample_sub["image_id"].astype(str).tolist()

missing = [fn for fn in test_images if not os.path.exists(os.path.join(TEST_DIR, fn))]
print("Missing test images:", len(missing))
if len(missing) > 0:
    print("First few missing:", missing[:5])




## === cell 9
def make_test_ds(tfrecs, batch_size):
    opts = tf.data.Options()
    opts.experimental_deterministic = True

    ds = tf.data.Dataset.from_tensor_slices(tfrecs)
    ds = ds.interleave(
        lambda fn: tf.data.TFRecordDataset(fn, num_parallel_reads=AUTO),
        cycle_length=min(8, len(tfrecs)) if len(tfrecs) else 1,
        num_parallel_calls=AUTO,
        deterministic=True,
    )
    ds = ds.map(_parse_test_example, num_parallel_calls=AUTO, deterministic=True)
    ds = ds.batch(batch_size, drop_remainder=False).prefetch(AUTO)
    ds = ds.with_options(opts)
    return ds


test_ds = make_test_ds(test_tfrecs, batch_size=16)

probs = model.predict(test_ds.map(lambda x, y: x), verbose=0)

all_ids = np.concatenate(
    [ids for _, ids in test_ds.as_numpy_iterator()], axis=0
).astype("U")
pred_labels = probs.argmax(axis=1).astype(int)

id_to_pred = dict(zip(all_ids.tolist(), pred_labels.tolist()))
pred_labels_ordered = [id_to_pred[iid] for iid in test_images]

print("Preds:", len(pred_labels_ordered), "Expected:", len(test_images))
print("First 10 preds:", pred_labels_ordered[:10])

if len(pred_labels_ordered) != len(test_images):
    raise RuntimeError(
        f"Prediction length mismatch: got {len(pred_labels_ordered)} preds for {len(test_images)} images"
    )

submission = pd.DataFrame({"image_id": test_images, "label": pred_labels_ordered})
submission["image_id"] = submission["image_id"].astype(str)
submission["label"] = submission["label"].astype(int)

sub_path = os.path.join(OUTPUT_DIR, "submission.csv")
submission.to_csv(sub_path, index=False)

print("Wrote:", sub_path, "shape:", submission.shape)
submission.head()
