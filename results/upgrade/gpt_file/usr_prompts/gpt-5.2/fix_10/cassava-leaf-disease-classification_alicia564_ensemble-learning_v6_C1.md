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

# 5. Target score

0.8596252644303415

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import re
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf

tf.keras.utils.set_random_seed(42)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass
try:
    tf.config.threading.set_intra_op_parallelism_threads(0)  # let TF decide
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

print("TF version:", tf.__version__)
print("Eager:", tf.executing_eagerly())



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.preprocessing import LabelEncoder

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
print("NUM_CLASSES:", NUM_CLASSES)



## === cell 2
import math
import glob

AUTOTUNE = tf.data.AUTOTUNE
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SHUFFLE_SEED = 42

train_paths = train["path"].to_numpy()
train_lbls = train["label_encoded"].to_numpy(np.int32)
valid_paths = valid["path"].to_numpy()
valid_lbls = valid["label_encoded"].to_numpy(np.int32)

IMG_H = tf.constant(IMG_SIZE[0], tf.int32)
IMG_W = tf.constant(IMG_SIZE[1], tf.int32)
IMG_H_F = tf.cast(IMG_H, tf.float32)
IMG_W_F = tf.cast(IMG_W, tf.float32)
PI = tf.constant(math.pi, tf.float32)
OUT_SHAPE = tf.stack([IMG_H, IMG_W])

TRAIN_TFREC_GLOB = (
    "/kaggle/input/cassava-leaf-disease-classification/train_tfrecords/*.tfrec"
)
TEST_TFREC_GLOB = (
    "/kaggle/input/cassava-leaf-disease-classification/test_tfrecords/*.tfrec"
)
train_tfrec_files = sorted(glob.glob(TRAIN_TFREC_GLOB))
test_tfrec_files = sorted(glob.glob(TEST_TFREC_GLOB))
if len(train_tfrec_files) == 0:
    raise FileNotFoundError(f"No TFRecord files found at {TRAIN_TFREC_GLOB}")
if len(test_tfrec_files) == 0:
    raise FileNotFoundError(f"No TFRecord files found at {TEST_TFREC_GLOB}")

FEATURES = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "label": tf.io.FixedLenFeature([], tf.int64),
    "image_name": tf.io.FixedLenFeature([], tf.string),
}


@tf.function
def _decode_resize_from_bytes(image_bytes):
    image = tf.image.decode_jpeg(image_bytes, channels=3)
    image = tf.image.resize(
        image, (IMG_H, IMG_W), method=tf.image.ResizeMethod.BILINEAR
    )
    image = tf.cast(image, tf.float32)
    return image


@tf.function
def _one_hot(label):
    return tf.one_hot(label, depth=NUM_CLASSES, dtype=tf.float32)


@tf.function
def _train_augment_and_preprocess(image):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)

    angle = tf.random.uniform((), -45.0, 45.0, dtype=tf.float32) * (PI / 180.0)

    tx = tf.random.uniform((), -0.2, 0.2, dtype=tf.float32) * IMG_H_F
    ty = tf.random.uniform((), -0.2, 0.2, dtype=tf.float32) * IMG_W_F

    zx = tf.random.uniform((), 0.8, 1.2, dtype=tf.float32)
    zy = tf.random.uniform((), 0.8, 1.2, dtype=tf.float32)

    shear = tf.random.uniform((), -0.2, 0.2, dtype=tf.float32)

    cos_a = tf.cos(angle)
    sin_a = tf.sin(angle)

    a0 = cos_a / zx
    a1 = (-sin_a + shear) / zx
    b0 = sin_a / zy
    b1 = cos_a / zy

    cx = IMG_W_F / 2.0
    cy = IMG_H_F / 2.0

    a2 = cx - a0 * cx - a1 * cy - ty
    b2 = cy - b0 * cx - b1 * cy - tx

    transform = tf.stack([a0, a1, a2, b0, b1, b2, 0.0, 0.0])[tf.newaxis, :]

    image = tf.raw_ops.ImageProjectiveTransformV3(
        images=image[tf.newaxis, ...],
        transforms=transform,
        output_shape=OUT_SHAPE,
        interpolation="BILINEAR",
        fill_mode="REFLECT",
        fill_value=0.0,
    )[0]

    image = preprocess_input(image)
    return image


@tf.function
def _valid_preprocess(image):
    return preprocess_input(image)


@tf.function
def _parse_train_example(ex):
    x = tf.io.parse_single_example(ex, FEATURES)
    img = _decode_resize_from_bytes(x["image"])
    lbl = tf.cast(x["label"], tf.int32)
    return img, lbl


@tf.function
def _parse_test_example(ex):
    x = tf.io.parse_single_example(ex, FEATURES)
    img = _decode_resize_from_bytes(x["image"])
    name = x["image_name"]
    return img, name


ds_opts = tf.data.Options()
ds_opts.deterministic = True

SHUFFLE_BUFFER = 4096

train_ds = tf.data.TFRecordDataset(
    train_tfrec_files, num_parallel_reads=AUTOTUNE
).with_options(ds_opts)
train_ds = train_ds.shuffle(
    buffer_size=SHUFFLE_BUFFER, seed=SHUFFLE_SEED, reshuffle_each_iteration=True
)
train_ds = train_ds.map(_parse_train_example, num_parallel_calls=AUTOTUNE)
train_ds = train_ds.map(
    lambda x, y: (_train_augment_and_preprocess(x), _one_hot(y)),
    num_parallel_calls=AUTOTUNE,
)
train_ds = train_ds.apply(tf.data.experimental.ignore_errors())
train_ds = train_ds.batch(BATCH_SIZE, drop_remainder=True).prefetch(AUTOTUNE)

valid_ds = tf.data.TFRecordDataset(
    train_tfrec_files, num_parallel_reads=AUTOTUNE
).with_options(ds_opts)
valid_names = tf.constant(valid["image_id"].to_numpy(dtype=str))
valid_table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(
        valid_names, tf.ones_like(valid_names, dtype=tf.int64)
    ),
    default_value=0,
)


def _is_valid(ex):
    x = tf.io.parse_single_example(
        ex, {"image_name": tf.io.FixedLenFeature([], tf.string)}
    )
    return tf.equal(valid_table.lookup(x["image_name"]), 1)


valid_ds = valid_ds.filter(_is_valid)
valid_ds = valid_ds.map(_parse_train_example, num_parallel_calls=AUTOTUNE)
valid_ds = valid_ds.map(
    lambda x, y: (_valid_preprocess(x), _one_hot(y)),
    num_parallel_calls=AUTOTUNE,
)
valid_ds = valid_ds.apply(tf.data.experimental.ignore_errors())
valid_ds = valid_ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)



## === cell 3
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback


class EarlyStoppingCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if self.model.stop_training:
            print(f"Early stopping triggered at epoch {epoch + 1}.")


early_stopping = EarlyStopping(
    monitor="val_loss", patience=3, restore_best_weights=True
)

learning_rate_reduction = ReduceLROnPlateau(
    monitor="val_loss", patience=2, factor=0.5, min_lr=1e-6, verbose=1
)



## === cell 4
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model

num_classes = NUM_CLASSES

base_model = EfficientNetB0(
    include_top=False, weights="imagenet", input_shape=(224, 224, 3)
)
base_model.trainable = False  # keep transfer learning simple and stable

x = GlobalAveragePooling2D()(base_model.output)
x = Dropout(0.2)(x)
outputs = Dense(num_classes, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=outputs)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

steps_per_epoch = int(np.ceil(len(train_paths) / BATCH_SIZE))
validation_steps = int(np.ceil(len(valid_paths) / BATCH_SIZE))

history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=10,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    callbacks=[early_stopping, learning_rate_reduction, EarlyStoppingCallback()],
    verbose=1,
)



## === cell 5
import pandas as pd

sample_sub = pd.read_csv(
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
)

assert "model" in globals(), "Model was not created; training likely failed."

sample_ids = sample_sub["image_id"].astype(str).to_numpy()
sample_ids_tf = tf.constant(sample_ids)

sample_table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(
        sample_ids_tf,
        tf.ones([tf.shape(sample_ids_tf)[0]], dtype=tf.int64),
    ),
    default_value=0,
)

test_raw = tf.data.TFRecordDataset(
    test_tfrec_files, num_parallel_reads=AUTOTUNE
).with_options(ds_opts)


def _keep_in_sample(ex):
    x = tf.io.parse_single_example(
        ex, {"image_name": tf.io.FixedLenFeature([], tf.string)}
    )
    return tf.equal(sample_table.lookup(x["image_name"]), 1)


test_ds = test_raw.filter(_keep_in_sample)
test_ds = test_ds.map(_parse_test_example, num_parallel_calls=AUTOTUNE)
test_ds = test_ds.map(
    lambda img, name: (preprocess_input(img), name), num_parallel_calls=AUTOTUNE
)
test_ds = test_ds.apply(tf.data.experimental.ignore_errors())
test_ds = test_ds.batch(64, drop_remainder=False).prefetch(AUTOTUNE)

test_img_ds = test_ds.map(lambda x, n: x, num_parallel_calls=AUTOTUNE)
probs = model.predict(test_img_ds, verbose=0)
preds = np.argmax(probs, axis=1).astype(int)

names = []
for batch in test_ds.map(lambda x, n: n):
    names.extend([n.decode("utf-8") for n in batch.numpy().tolist()])

if len(names) != len(preds):
    raise RuntimeError(
        f"Mismatch between names ({len(names)}) and preds ({len(preds)})."
    )

name_to_pred = dict(zip(names, preds.tolist()))
all_preds = [
    int(name_to_pred[iid]) for iid in sample_sub["image_id"].astype(str).tolist()
]

submission_df = pd.DataFrame({"image_id": sample_sub["image_id"], "label": all_preds})

submission_path = "/kaggle/working/submission.csv"
submission_df.to_csv(submission_path, index=False)
print(f"Submission file created: {submission_path}")
print(submission_df.head())
print("Submission shape:", submission_df.shape)
print("Unique labels predicted:", sorted(submission_df["label"].unique().tolist()))

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
UnboundLocalError                         Traceback (most recent call last)
/tmp/ipykernel_11/2149955980.py in <cell line: 0>()
     44 # Predict on images
     45 test_img_ds = test_ds.map(lambda x, n: x, num_parallel_calls=AUTOTUNE)
---> 46 probs = model.predict(test_img_ds, verbose=0)
     47 preds = np.argmax(probs, axis=1).astype(int)
     48 

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/backend/tensorflow/trainer.py in predict(self, x, batch_size, verbose, steps, callbacks)
    567         callbacks.on_predict_end()
    568         outputs = tree.map_structure_up_to(
--> 569             batch_outputs, potentially_ragged_concat, outputs
    570         )
    571         return tree.map_structure(convert_to_np_if_not_ragged, outputs)

UnboundLocalError: cannot access local variable 'batch_outputs' where it is not associated with a value
