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

# 5. Target score

0.8401329706860079

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.06353) has done: 'The changes keep the model architecture and training schedule unchanged while eliminating the per‑image Python loop used for inference. By batching test predictions through an `ImageDataGenerator` and adding multiprocessing workers to the training generators, disk‑I/O and Python overhead are dramatically reduced, allowing the whole script to finish well under the 600‑second limit. All other logic, paths, and hyper‑parameters remain identical.'
- What this solution (achieved 0.19619) has done: 'We replace the slow ImageDataGenerator pipelines with efficient tf.data datasets and move the data‑augmentation into the model (using built‑in Keras augmentation layers). This removes the heavy Python‑level image loading loop, lets TensorFlow parallelize I/O and preprocessing, and keeps the same model architecture and training schedule, so accuracy is unchanged while runtime drops below the 600 s limit.'
- What this solution (achieved 0.69544) has done: 'I fixed the protobuf import error that prevented TensorFlow from loading, removed unsupported image‑augmentation calls (`tf.image.random_zoom` and the old `random_rotation`), and cleaned up the dataset pipeline. These changes let the script run end‑to‑end, correctly create the training/validation datasets, train the model, and write a valid `submission.csv` file, moving the score toward the target.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import tensorflow as tf

tf.config.optimizer.set_jit(True)
tf.config.experimental.enable_tensor_float_32_execution(True)

options = tf.data.Options()
options.experimental_threading.private_threadpool_size = 8
options.experimental_optimization.apply_default_optimizations = True
tf.data.experimental.enable_debug_mode(False)

import json, random, shutil, datetime
import numpy as np, pandas as pd

from tensorflow.keras import (
    layers,
    models,
    callbacks,
    optimizers,
    applications,
    mixed_precision,
)
from tensorflow.keras.utils import load_img, img_to_array

mixed_precision.set_global_policy("mixed_float16")

BASE_INPUT = os.getenv("KAGGLE_INPUT_DIR", "/kaggle/input")
WORK_DIR = os.path.join(BASE_INPUT, "cassava-leaf-disease-classification")

preprocess_fn = applications.efficientnet_v2.preprocess_input
IMG_SIZE = (224, 224)

BATCH_SIZE = 256


def make_dataset(df, img_dir, training=False):
    """Create a tf.data.Dataset from a dataframe of image ids."""
    paths = tf.strings.join([img_dir, "/", df["image_id"].values])
    has_labels = "label" in df.columns
    labels = df["label"].values if has_labels else None

    if has_labels:
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    else:
        ds = tf.data.Dataset.from_tensor_slices(paths)

    def _load(path, label=None):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, IMG_SIZE)
        img = tf.cast(img, tf.float32)
        img = preprocess_fn(img)
        return (img, label) if has_labels else img

    if has_labels:
        ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
    else:
        ds = ds.map(
            lambda p: _load(p), num_parallel_calls=tf.data.AUTOTUNE, deterministic=False
        )

    ds = ds.cache()
    ds = ds.with_options(options)

    if training:
        ds = ds.map(
            lambda x, y: (tf.image.random_flip_left_right(x), y),
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=False,
        )
        ds = ds.shuffle(1024)

    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


train_df = pd.read_csv(os.path.join(WORK_DIR, "train.csv"))
train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)

val_fraction = 0.2
val_size = int(len(train_df) * val_fraction)
val_df = train_df.iloc[:val_size].reset_index(drop=True)
train_df = train_df.iloc[val_size:].reset_index(drop=True)

train_dir = os.path.join(WORK_DIR, "train_images")
train_dataset = make_dataset(train_df, train_dir, training=True)
val_dataset = make_dataset(val_df, train_dir, training=False)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def build_model(input_shape=(224, 224, 3), num_classes=5):
    inputs = layers.Input(shape=input_shape)
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.111)(x)  # ~20 degrees
    x = layers.RandomZoom(0.2)(x)
    base = applications.efficientnet_v2.EfficientNetV2B0(
        weights="imagenet", include_top=False, input_tensor=x
    )
    base.trainable = False
    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=optimizers.Adam(),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


model = build_model()
checkpoint_cb = callbacks.ModelCheckpoint(
    "best_model.h5", save_best_only=True, monitor="val_accuracy", mode="max"
)
reduce_lr_cb = callbacks.ReduceLROnPlateau(
    monitor="val_accuracy", factor=0.5, patience=2, min_lr=1e-6, mode="max"
)

model.fit(
    train_dataset,
    epochs=8,
    validation_data=val_dataset,
    callbacks=[checkpoint_cb, reduce_lr_cb],
    verbose=2,
)

model.load_weights("best_model.h5")

for layer in model.layers:
    if isinstance(layer, tf.keras.Model):  # the EfficientNet sub‑model
        layer.trainable = True

model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.fit(
    train_dataset,
    epochs=8,  # extended fine‑tuning
    validation_data=val_dataset,
    callbacks=[reduce_lr_cb],
    verbose=2,
)

model.load_weights("best_model.h5")


## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1750975417.py in <cell line: 0>()
     20 
     21 
---> 22 model = build_model()
     23 checkpoint_cb = callbacks.ModelCheckpoint(
     24     "best_model.h5", save_best_only=True, monitor="val_accuracy", mode="max"

/tmp/ipykernel_11/1750975417.py in build_model(input_shape, num_classes)
      1 def build_model(input_shape=(224, 224, 3), num_classes=5):
----> 2     inputs = layers.Input(shape=input_shape)
      3     x = layers.RandomFlip("horizontal")(inputs)
      4     x = layers.RandomRotation(0.111)(x)  # ~20 degrees
      5     x = layers.RandomZoom(0.2)(x)

NameError: name 'layers' is not defined

## === cell 2
test_dir = os.path.join(WORK_DIR, "test_images")
test_images = [f for f in os.listdir(test_dir) if f.lower().endswith(".jpg")]
test_df = pd.DataFrame({"image_id": test_images})

test_dataset = make_dataset(test_df, test_dir, training=False)

preds = model.predict(test_dataset, verbose=2)
labels = np.argmax(preds, axis=1).astype(int)

submission = pd.DataFrame({"image_id": test_df["image_id"], "label": labels})
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}, shape: {submission.shape}")

## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/207888205.py in <cell line: 0>()
----> 1 test_dir = os.path.join(WORK_DIR, "test_images")
      2 test_images = [f for f in os.listdir(test_dir) if f.lower().endswith(".jpg")]
      3 test_df = pd.DataFrame({"image_id": test_images})
      4 
      5 test_dataset = make_dataset(test_df, test_dir, training=False)

NameError: name 'WORK_DIR' is not defined
