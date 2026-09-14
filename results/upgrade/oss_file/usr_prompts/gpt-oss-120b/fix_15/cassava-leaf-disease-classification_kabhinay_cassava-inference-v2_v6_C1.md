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

0.8819885161680266

# 6. Current score

0.61099

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.11584) has done: 'The fix removes the failing custom loss imports and the attempts to load unavailable SavedModel files, and replaces them with a lightweight transfer‑learning pipeline using EfficientNet‑B0. The new code loads the training CSV, builds a simple image data pipeline, trains a few epochs (fast enough for the notebook), predicts on the test set, and writes a properly formatted `submission.csv`. This resolves the import and model‑loading errors and ensures a valid submission file is produced, while keeping the overall approach (image classification with a CNN) unchanged.'
- What this solution (achieved 0.61099) has done: 'The script is revised to replace the slow `ImageDataGenerator` pipelines with efficient `tf.data` pipelines that decode, resize, and (for training) augment images using TensorFlow’s native operations.  The new pipelines use `shuffle`, `batch`, `prefetch`, and deterministic seeds, eliminating costly Python‑level multiprocessing and per‑epoch disk I/O while keeping the model architecture, loss, optimizer, and training epochs unchanged.  Test data loading is also switched to a `tf.data` pipeline, and filenames are captured directly from the sorted file list to preserve the original submission order.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
try:
    from google.protobuf import message_factory

    if not hasattr(message_factory.MessageFactory, "GetPrototype"):

        def _GetPrototype(self, descriptor):
            return self.GetMessageClass(descriptor)

        message_factory.MessageFactory.GetPrototype = _GetPrototype
except Exception:
    pass  # If protobuf is not available, TensorFlow will raise its own error

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, mixed_precision
from tensorflow.keras.applications import EfficientNetB0

np.random.seed(42)
tf.random.set_seed(42)
tf.config.optimizer.set_jit(True)  # Enable XLA compilation

mixed_precision.set_global_policy("mixed_float16")




## === cell 1
DATA_ROOT = "../input/cassava-leaf-disease-classification"
TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train_images")
TEST_IMG_DIR = os.path.join(DATA_ROOT, "test_images")
SAMPLE_SUBMIT = os.path.join(DATA_ROOT, "sample_submission.csv")
OUTPUT_SUBMIT = "/kaggle/working/submission.csv"




## === cell 2
train_df = pd.read_csv(TRAIN_CSV)
train_df["label"] = train_df["label"].astype(str)




## === cell 3
IMG_SIZE = (224, 224)
BATCH_SIZE = 256
AUTOTUNE = tf.data.experimental.AUTOTUNE
TRAIN_VAL_SPLIT = 0.1
SEED = 42

train_paths = [os.path.join(TRAIN_IMG_DIR, fname) for fname in train_df["image_id"]]
train_labels = train_df["label"].astype(int).values

full_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))

full_ds = full_ds.shuffle(
    buffer=len(train_paths), seed=SEED, reshuffle_each_iteration=False
)

val_size = int(TRAIN_VAL_SPLIT * len(train_paths))
val_ds = full_ds.take(val_size)
train_ds = full_ds.skip(val_size)


def decode_and_preprocess(path, label, training=False):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = img / 255.0  # rescale

    if training:
        img = tf.image.random_flip_left_right(img, seed=SEED)
        angle = tf.random.uniform([], -20.0, 20.0, seed=SEED) * (3.14159265 / 180.0)
        img = tfa.image.rotate(img, angle) if tf.__version__ >= "2.5" else img
        img = tf.keras.preprocessing.image.random_shift(
            img,
            0.1,
            0.1,
            row_axis=0,
            col_axis=1,
            channel_axis=2,
            fill_mode="nearest",
            seed=SEED,
        )
        img = tf.keras.preprocessing.image.random_zoom(
            img,
            (0.9, 1.1),
            row_axis=0,
            col_axis=1,
            channel_axis=2,
            fill_mode="nearest",
            seed=SEED,
        )
    return img, label


train_ds = train_ds.map(
    lambda p, l: decode_and_preprocess(p, l, training=True), num_parallel_calls=AUTOTUNE
)
train_ds = train_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

val_ds = val_ds.map(
    lambda p, l: decode_and_preprocess(p, l, training=False),
    num_parallel_calls=AUTOTUNE,
)
val_ds = val_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1223974952.py in <cell line: 0>()
     14 
     15 # Shuffle once with a fixed seed for reproducibility
---> 16 full_ds = full_ds.shuffle(
     17     buffer=len(train_paths), seed=SEED, reshuffle_each_iteration=False
     18 )

TypeError: DatasetV2.shuffle() got an unexpected keyword argument 'buffer'

## === cell 4
base_model = EfficientNetB0(
    include_top=False, weights="imagenet", input_shape=IMG_SIZE + (3,), pooling="avg"
)
base_model.trainable = False  # freeze base

inputs = layers.Input(shape=IMG_SIZE + (3,))
x = base_model(inputs, training=False)
outputs = layers.Dense(5, activation="softmax")(x)
model = models.Model(inputs, outputs)

model.compile(
    optimizer=optimizers.Adam(),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

EPOCHS_INITIAL = 5
model.fit(
    train_ds,
    epochs=EPOCHS_INITIAL,
    validation_data=val_ds,
    verbose=2,
)

base_model.trainable = True
model.compile(
    optimizer=optimizers.Adam(1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
EPOCHS_FINE = 8
model.fit(
    train_ds,
    epochs=EPOCHS_FINE,
    validation_data=val_ds,
    verbose=2,
)




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2576958373.py in <cell line: 0>()
     17 EPOCHS_INITIAL = 5
     18 model.fit(
---> 19     train_ds,
     20     epochs=EPOCHS_INITIAL,
     21     validation_data=val_ds,

NameError: name 'train_ds' is not defined

## === cell 5
test_files = sorted([f for f in os.listdir(TEST_IMG_DIR) if f.lower().endswith(".jpg")])
test_paths = [os.path.join(TEST_IMG_DIR, f) for f in test_files]


def decode_test(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = img / 255.0
    return img


test_ds = tf.data.Dataset.from_tensor_slices(test_paths)
test_ds = test_ds.map(decode_test, num_parallel_calls=AUTOTUNE)
test_ds = test_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

pred_probs = model.predict(test_ds, verbose=2)
pred_labels = np.argmax(pred_probs, axis=1)




## === cell 6
output_dir = os.path.dirname(OUTPUT_SUBMIT)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

submission = pd.DataFrame({"image_id": test_files, "label": pred_labels})
submission.to_csv(OUTPUT_SUBMIT, index=False)

print("Submission saved to {}".format(OUTPUT_SUBMIT))
