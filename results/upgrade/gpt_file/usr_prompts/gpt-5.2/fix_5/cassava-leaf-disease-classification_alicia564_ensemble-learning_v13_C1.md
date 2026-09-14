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
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.efficientnet import preprocess_input

DATA_DIR = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_CSV_PATH = f"{DATA_DIR}/train.csv"
TRAIN_IMG_DIR = f"{DATA_DIR}/train_images"
SAMPLE_SUB_PATH = f"{DATA_DIR}/sample_submission.csv"
TEST_IMG_DIR = f"{DATA_DIR}/test_images"

train_csv = pd.read_csv(TRAIN_CSV_PATH)
train_csv["path"] = TRAIN_IMG_DIR + "/" + train_csv["image_id"]
train_csv["label"] = train_csv["label"].astype(int)

train, valid = train_test_split(
    train_csv, test_size=0.2, stratify=train_csv["label"], random_state=SEED
)

BATCH_SIZE = 32
IMG_SIZE = (224, 224)

train_steps = int(np.ceil(len(train) / BATCH_SIZE))
valid_steps = int(np.ceil(len(valid) / BATCH_SIZE))

AUTOTUNE = tf.data.AUTOTUNE

train_paths = train["path"].to_numpy()
train_labels = train["label"].to_numpy(np.int32)

valid_paths = valid["path"].to_numpy()
valid_labels = valid["label"].to_numpy(np.int32)


def _read_decode_resize(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)  # matches JPG inputs
    img = tf.image.resize(img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32)
    return img


@tf.function
def _augment(img, label):
    img = tf.image.random_flip_left_right(img, seed=SEED)
    img = tf.image.random_flip_up_down(img, seed=SEED)

    batch = tf.expand_dims(img, 0)

    rot = tf.random.uniform([], -45.0, 45.0, seed=SEED) * (np.pi / 180.0)
    zoom = tf.random.uniform([], 0.8, 1.2, seed=SEED)
    shear = tf.random.uniform([], -0.2, 0.2, seed=SEED)
    tx = tf.random.uniform([], -0.2, 0.2, seed=SEED) * tf.cast(IMG_SIZE[1], tf.float32)
    ty = tf.random.uniform([], -0.2, 0.2, seed=SEED) * tf.cast(IMG_SIZE[0], tf.float32)

    cos_r = tf.math.cos(rot)
    sin_r = tf.math.sin(rot)

    cx = tf.cast(IMG_SIZE[1], tf.float32) / 2.0
    cy = tf.cast(IMG_SIZE[0], tf.float32) / 2.0

    r00 = cos_r
    r01 = -sin_r
    r10 = sin_r
    r11 = cos_r

    sh00 = 1.0
    sh01 = tf.math.tan(shear)
    sh10 = 0.0
    sh11 = 1.0

    a00 = r00 * sh00 + r01 * sh10
    a01 = r00 * sh01 + r01 * sh11
    a10 = r10 * sh00 + r11 * sh10
    a11 = r10 * sh01 + r11 * sh11

    a00 *= zoom
    a01 *= zoom
    a10 *= zoom
    a11 *= zoom

    A = tf.stack([[a00, a01], [a10, a11]])
    det = a00 * a11 - a01 * a10
    invA = (1.0 / det) * tf.stack([[a11, -a01], [-a10, a00]])

    b = tf.stack([tx + cx - (a00 * cx + a01 * cy), ty + cy - (a10 * cx + a11 * cy)])
    invb = -tf.linalg.matvec(invA, b)

    transform = tf.stack(
        [invA[0, 0], invA[0, 1], invb[0], invA[1, 0], invA[1, 1], invb[1], 0.0, 0.0]
    )
    transform = tf.expand_dims(transform, 0)

    batch = tf.raw_ops.ImageProjectiveTransformV3(
        images=batch,
        transforms=transform,
        output_shape=tf.constant([IMG_SIZE[0], IMG_SIZE[1]], dtype=tf.int32),
        interpolation="BILINEAR",
        fill_mode="NEAREST",
        fill_value=0.0,
    )
    img = tf.squeeze(batch, 0)

    img = preprocess_input(img)
    return img, label


@tf.function
def _preprocess_only(img, label):
    img = preprocess_input(img)
    return img, label


def make_ds(paths, labels, training):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(
            buffer_size=len(paths), seed=SEED, reshuffle_each_iteration=True
        )

    def _load(path, label):
        img = _read_decode_resize(path)
        return img, label

    ds = ds.map(_load, num_parallel_calls=AUTOTUNE, deterministic=True)
    if training:
        ds = ds.map(_augment, num_parallel_calls=AUTOTUNE, deterministic=True)
    else:
        ds = ds.map(_preprocess_only, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_dataset = make_ds(train_paths, train_labels, training=True)
valid_dataset = make_ds(valid_paths, valid_labels, training=False)



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
)

model.summary()



## === cell 4
history = model.fit(
    train_dataset,
    validation_data=valid_dataset,
    epochs=12,
    steps_per_epoch=train_steps,
    validation_steps=valid_steps,
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
)

history_ft = model.fit(
    train_dataset,
    validation_data=valid_dataset,
    epochs=6,
    steps_per_epoch=train_steps,
    validation_steps=valid_steps,
    callbacks=[early_stopping, learning_rate_reduction, EarlyStoppingCallback()],
    verbose=1,
)



## === cell 6
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
test_paths = [
    os.path.join(TEST_IMG_DIR, img_id) for img_id in sample_sub["image_id"].tolist()
]
test_paths = np.array(test_paths)


def make_test_ds(paths):
    ds = tf.data.Dataset.from_tensor_slices(paths)

    def _load(path):
        img = _read_decode_resize(path)
        img = preprocess_input(img)
        return img

    ds = ds.map(_load, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


test_dataset = make_test_ds(test_paths)
test_steps = int(np.ceil(len(test_paths) / BATCH_SIZE))

probs = model.predict(
    test_dataset,
    steps=test_steps,
    verbose=1,
)
preds = np.argmax(probs, axis=1).astype(int)

submission_df = pd.DataFrame({"image_id": sample_sub["image_id"], "label": preds})
submission_path = "/kaggle/working/submission.csv"
submission_df.to_csv(submission_path, index=False)

print("Submission file created:", submission_path)
print(submission_df.head())
print("Rows:", len(submission_df), "Cols:", submission_df.shape[1])
