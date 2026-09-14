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

3.13

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
import re
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

os.environ.setdefault("PYTHONHASHSEED", "42")

import tensorflow as tf

tf.keras.utils.set_random_seed(42)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

AUTOTUNE = tf.data.AUTOTUNE

print("TF version:", tf.__version__)



## === cell 1
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications.efficientnet import preprocess_input

label_to_disease = pd.read_json(
    "/kaggle/input/cassava-leaf-disease-classification/label_num_to_disease_map.json",
    typ="series",
)
train_csv = pd.read_csv("/kaggle/input/cassava-leaf-disease-classification/train.csv")

train_csv["disease"] = train_csv["label"].map(label_to_disease)
train_csv["path"] = (
    "/kaggle/input/cassava-leaf-disease-classification/train_images/"
    + train_csv["image_id"]
)

train_csv["label_encoded"] = LabelEncoder().fit_transform(train_csv["disease"])

train_csv["disease"] = train_csv["disease"].astype(str)
train_csv["label"] = train_csv["label"].astype(str)

train, valid = train_test_split(
    train_csv, test_size=0.2, stratify=train_csv["label"], random_state=42
)

NUM_CLASSES = int(train_csv["label_encoded"].nunique())
print("NUM_CLASSES:", NUM_CLASSES, "train:", len(train), "valid:", len(valid))



## === cell 2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomRotation(0.125, fill_mode="nearest", seed=42),
        tf.keras.layers.RandomTranslation(0.2, 0.2, fill_mode="nearest", seed=42),
        tf.keras.layers.RandomZoom(0.2, 0.2, fill_mode="nearest", seed=42),
        tf.keras.layers.RandomFlip("horizontal_and_vertical", seed=42),
    ],
    name="data_augmentation",
)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

TRAIN_TFRECORD_DIR = "/kaggle/input/cassava-leaf-disease-classification/train_tfrecords"
TEST_TFRECORD_DIR = "/kaggle/input/cassava-leaf-disease-classification/test_tfrecords"

train_tfrec_files = sorted(
    tf.io.gfile.glob(os.path.join(TRAIN_TFRECORD_DIR, "*.tfrec"))
)
test_tfrec_files = sorted(tf.io.gfile.glob(os.path.join(TEST_TFRECORD_DIR, "*.tfrec")))

if not train_tfrec_files:
    raise FileNotFoundError(f"No TFRecords found in {TRAIN_TFRECORD_DIR}")
if not test_tfrec_files:
    raise FileNotFoundError(f"No TFRecords found in {TEST_TFRECORD_DIR}")

_TFREC_FEATURES = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),
    "image_name": tf.io.FixedLenFeature([], tf.string),
}

TFREC_COMPRESSION = None  # uncompressed TFRecords


@tf.function
def _parse_raw(example_proto):
    ex = tf.io.parse_single_example(example_proto, _TFREC_FEATURES)
    img_bytes = ex["image"]
    label = tf.cast(ex["target"], tf.int32)
    image_id = ex["image_name"]
    return img_bytes, label, image_id


@tf.function
def _decode_and_preprocess(img_bytes, label, image_id):
    image = tf.io.decode_jpeg(img_bytes, channels=3, dct_method="INTEGER_FAST")
    image = tf.image.resize(image, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    image = tf.cast(image, tf.float32)
    image = preprocess_input(image)
    return image, label, image_id


@tf.function
def _to_xy(image, label, image_id):
    y = tf.one_hot(label, depth=NUM_CLASSES, dtype=tf.float32)
    return image, y


@tf.function
def _augment_xy(image, y):
    image = data_augmentation(image, training=True)
    return image, y


def _extract_shard_id(tfrec_path: str) -> int:
    m = re.search(r"ld_train(\d+)-", os.path.basename(tfrec_path))
    return int(m.group(1)) if m else -1


all_shard_ids = np.array(
    [_extract_shard_id(p) for p in train_tfrec_files], dtype=np.int32
)
if np.any(all_shard_ids < 0):
    raise ValueError(
        "Unexpected TFRecord naming; cannot extract shard ids for fast split."
    )

rng = np.random.RandomState(42)
perm = rng.permutation(len(train_tfrec_files))
n_valid = int(round(0.2 * len(train_tfrec_files)))
valid_idx = np.sort(perm[:n_valid])
train_idx = np.sort(perm[n_valid:])

train_tfrec_split = [train_tfrec_files[i] for i in train_idx]
valid_tfrec_split = [train_tfrec_files[i] for i in valid_idx]

train_opts = tf.data.Options()
train_opts.experimental_deterministic = False
valid_opts = tf.data.Options()
valid_opts.experimental_deterministic = False


def _build_base_ds(tfrec_files, opts):
    ds = tf.data.TFRecordDataset(
        tfrec_files,
        num_parallel_reads=AUTOTUNE,
        compression_type=TFREC_COMPRESSION,
    ).with_options(opts)
    ds = ds.map(_parse_raw, num_parallel_calls=AUTOTUNE, deterministic=False)
    return ds


train_base_raw = _build_base_ds(train_tfrec_split, train_opts)
valid_base_raw = _build_base_ds(valid_tfrec_split, valid_opts)

shuffle_buf = int(min(len(train), 4096))

train_ds = (
    train_base_raw.cache()
    .shuffle(shuffle_buf, seed=42, reshuffle_each_iteration=True)
    .map(_decode_and_preprocess, num_parallel_calls=AUTOTUNE, deterministic=False)
    .map(_to_xy, num_parallel_calls=AUTOTUNE, deterministic=False)
    .map(_augment_xy, num_parallel_calls=AUTOTUNE, deterministic=False)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTOTUNE)
)

valid_ds = (
    valid_base_raw.cache()
    .map(_decode_and_preprocess, num_parallel_calls=AUTOTUNE, deterministic=False)
    .map(_to_xy, num_parallel_calls=AUTOTUNE, deterministic=False)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTOTUNE)
)


class EarlyStoppingCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if self.model.stop_training:
            print(f"Early stopping triggered at epoch {epoch + 1}.")


early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True,
)

learning_rate_reduction = ReduceLROnPlateau(
    monitor="val_loss",
    patience=2,
    factor=0.5,
    min_lr=1e-6,
    verbose=1,
)

base_model = EfficientNetB0(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3),
)
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
preds = Dense(NUM_CLASSES, activation="softmax")(x)
model = Model(inputs=base_model.input, outputs=preds)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
    steps_per_execution=10,
)

history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=10,
    callbacks=[early_stopping, learning_rate_reduction, EarlyStoppingCallback()],
    verbose=1,
)



## === cell 3
import numpy as np
import pandas as pd

sample_sub = pd.read_csv(
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
)

test_opts = tf.data.Options()
test_opts.experimental_deterministic = False


@tf.function
def _to_x_id(image, label, image_id):
    return image, image_id


test_ds_with_ids = (
    tf.data.TFRecordDataset(
        test_tfrec_files,
        num_parallel_reads=AUTOTUNE,
        compression_type=TFREC_COMPRESSION,
    )
    .with_options(test_opts)
    .map(_parse_raw, num_parallel_calls=AUTOTUNE, deterministic=False)
    .cache()  # cache parsed bytes/label/id, not decoded images (faster + lower memory)
    .map(_decode_and_preprocess, num_parallel_calls=AUTOTUNE, deterministic=False)
    .map(_to_x_id, num_parallel_calls=AUTOTUNE, deterministic=False)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTOTUNE)
)

pred_labels_list = []
ids_list = []
for batch_images, batch_ids in test_ds_with_ids:
    probs = model(batch_images, training=False).numpy()
    pred_labels_list.append(np.argmax(probs, axis=1).astype(np.int32))
    ids_list.append(batch_ids.numpy())

pred_labels = np.concatenate(pred_labels_list, axis=0).astype(int)
ids_bytes = np.concatenate(ids_list, axis=0)

test_ids = tf.strings.unicode_decode(
    ids_bytes, "UTF-8"
).numpy()  # unused, but forces shape check
test_ids = tf.strings.as_string(ids_bytes).numpy()  # fallback representation if needed
test_ids = tf.strings.regex_replace(ids_bytes, b"$", b"").numpy()
test_ids = np.char.decode(test_ids.astype("S"), "utf-8")

pred_map = dict(zip(test_ids, pred_labels))
ordered_preds = sample_sub["image_id"].map(pred_map)
ordered_preds = ordered_preds.fillna(0).astype(int).values

submission_df = pd.DataFrame(
    {"image_id": sample_sub["image_id"].values, "label": ordered_preds}
)
submission_path = "/kaggle/working/submission.csv"
submission_df.to_csv(submission_path, index=False)

print(f"Submission file created: {submission_path}")
print(submission_df.head())
print("Submission shape:", submission_df.shape)
print("Unique labels in submission:", np.unique(submission_df["label"].values))
