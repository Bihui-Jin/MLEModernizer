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
import random
import numpy as np
import pandas as pd

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf
from tensorflow import keras

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

print("TF version:", tf.__version__)



## === cell 1
from tensorflow.keras.applications.efficientnet import preprocess_input

DATA_DIR = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_CSV_PATH = f"{DATA_DIR}/train.csv"
TRAIN_IMG_DIR = f"{DATA_DIR}/train_images"
SAMPLE_SUB_PATH = f"{DATA_DIR}/sample_submission.csv"
TEST_IMG_DIR = f"{DATA_DIR}/test_images"

train_csv = pd.read_csv(TRAIN_CSV_PATH)
train_csv["path"] = TRAIN_IMG_DIR + "/" + train_csv["image_id"]
train_csv["label"] = train_csv["label"].astype(int)


def stratified_split_df(df: pd.DataFrame, label_col: str, test_size: float, seed: int):
    rng = np.random.RandomState(seed)
    train_parts = []
    valid_parts = []
    for lbl, g in df.groupby(label_col, sort=False):
        idx = g.index.to_numpy()
        rng.shuffle(idx)
        n_valid = int(np.round(len(idx) * test_size))
        valid_idx = idx[:n_valid]
        train_idx = idx[n_valid:]
        valid_parts.append(df.loc[valid_idx])
        train_parts.append(df.loc[train_idx])
    train_df = (
        pd.concat(train_parts, axis=0)
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )
    valid_df = (
        pd.concat(valid_parts, axis=0)
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )
    return train_df, valid_df


train, valid = stratified_split_df(train_csv, "label", test_size=0.2, seed=SEED)

BATCH_SIZE = 32
IMG_SIZE = (224, 224)

AUTOTUNE = tf.data.AUTOTUNE

train_paths = train["path"].to_numpy()
train_labels = train["label"].to_numpy(np.int32)

valid_paths = valid["path"].to_numpy()
valid_labels = valid["label"].to_numpy(np.int32)


def _read_decode_resize(path):
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img_bytes, channels=3, dct_method="INTEGER_FAST")
    img = tf.image.resize(
        img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR, antialias=False
    )
    img = tf.cast(img, tf.float32)
    return img


data_augmentation = keras.Sequential(
    [
        keras.layers.RandomFlip("horizontal_and_vertical", seed=SEED),
        keras.layers.RandomRotation(0.125, fill_mode="nearest", seed=SEED),
        keras.layers.RandomZoom(
            height_factor=(-0.2, 0.2), width_factor=(-0.2, 0.2), seed=SEED
        ),
        keras.layers.RandomTranslation(
            height_factor=0.2, width_factor=0.2, fill_mode="nearest", seed=SEED
        ),
    ],
    name="aug",
)


def _decode_only(path, label):
    img = _read_decode_resize(path)
    return img, label


def _augment_and_preprocess(img, label):
    img = data_augmentation(img, training=True)
    img = preprocess_input(img)
    return img, label


def _preprocess_only(img, label):
    img = preprocess_input(img)
    return img, label


DS_OPTIONS = tf.data.Options()
DS_OPTIONS.deterministic = True
try:
    DS_OPTIONS.autotune.enabled = True
except Exception:
    pass


def make_ds(paths, labels, training, use_cache: bool):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.with_options(DS_OPTIONS)

    if training:
        buf = min(len(paths), 8192)
        ds = ds.shuffle(buffer_size=buf, seed=SEED, reshuffle_each_iteration=True)

    ds = ds.map(_decode_only, num_parallel_calls=AUTOTUNE, deterministic=True)

    if use_cache:
        ds = ds.cache()

    if training:
        ds = ds.map(
            _augment_and_preprocess, num_parallel_calls=AUTOTUNE, deterministic=True
        )
    else:
        ds = ds.map(_preprocess_only, num_parallel_calls=AUTOTUNE, deterministic=True)

    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_dataset = make_ds(train_paths, train_labels, training=True, use_cache=False)
valid_dataset = make_ds(valid_paths, valid_labels, training=False, use_cache=True)



## === cell 2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.models import Model


class EarlyStoppingCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if getattr(self.model, "stop_training", False):
            print(f"Early stopping triggered at epoch {epoch + 1}.")


early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True,
    verbose=1,
)

learning_rate_reduction = ReduceLROnPlateau(
    monitor="val_loss",
    patience=2,
    factor=0.5,
    min_lr=1e-6,
    verbose=1,
)



## === cell 3
inputs = Input(shape=(224, 224, 3))
base = EfficientNetB0(include_top=False, weights="imagenet", input_tensor=inputs)
base.trainable = False

x = GlobalAveragePooling2D()(base.output)
x = Dropout(0.2)(x)
outputs = Dense(5, activation="softmax")(x)

model = Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"],
    steps_per_execution=32,
)

model.summary()



## === cell 4
history = model.fit(
    train_dataset,
    validation_data=valid_dataset,
    epochs=12,
    callbacks=[early_stopping, learning_rate_reduction, EarlyStoppingCallback()],
    verbose=1,
)



## === cell 5
base.trainable = True
for layer in base.layers[:-20]:
    layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"],
    steps_per_execution=32,
)

history_ft = model.fit(
    train_dataset,
    validation_data=valid_dataset,
    epochs=6,
    callbacks=[early_stopping, learning_rate_reduction, EarlyStoppingCallback()],
    verbose=1,
)



## === cell 6
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
test_paths = [
    os.path.join(TEST_IMG_DIR, img_id) for img_id in sample_sub["image_id"].tolist()
]
test_paths = np.array(test_paths)


def _decode_only_no_label(path):
    img = _read_decode_resize(path)
    return img


def _preprocess_only_no_label(img):
    img = preprocess_input(img)
    return img


def make_test_ds(paths, use_cache: bool):
    ds = tf.data.Dataset.from_tensor_slices(paths)
    ds = ds.with_options(DS_OPTIONS)
    ds = ds.map(_decode_only_no_label, num_parallel_calls=AUTOTUNE, deterministic=True)
    if use_cache:
        ds = ds.cache()
    ds = ds.map(
        _preprocess_only_no_label, num_parallel_calls=AUTOTUNE, deterministic=True
    )
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


test_dataset = make_test_ds(test_paths, use_cache=True)

probs = model.predict(
    test_dataset,
    verbose=1,
)
preds = np.argmax(probs, axis=1).astype(int)

submission_df = pd.DataFrame({"image_id": sample_sub["image_id"], "label": preds})
submission_path = "/kaggle/working/submission.csv"
submission_df.to_csv(submission_path, index=False)

print("Submission file created:", submission_path)
print(submission_df.head())
print("Rows:", len(submission_df), "Cols:", submission_df.shape[1])
