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

0.885312783318223

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.05531) has done: 'The changes increase data‑loading parallelism by using all available CPU cores and enabling multiprocessing for both training/validation and test generators, and they pass the same settings to `model.fit`. This speeds up image preprocessing and queue filling without altering the network architecture, training epochs, batch size, or any other algorithmic logic, so the resulting model and predictions remain unchanged.'
- What this solution (achieved 0.05531) has done: 'The fix speeds up the slow image‑loading pipeline by maximizing parallel workers and enabling multiprocessing for both the generators and the training loop; this reduces I/O bottlenecks while keeping the model, epochs, batch size, and augmentation unchanged, so the results remain identical.'

# 9. Code solution

## === cell 0
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input
from tensorflow.keras.models import Model

tf.config.optimizer.set_jit(True)

if tf.config.list_physical_devices("GPU"):
    from tensorflow.keras import mixed_precision

    mixed_precision.set_global_policy("mixed_float16")

tf.config.threading.set_intra_op_parallelism_threads(os.cpu_count())
tf.config.threading.set_inter_op_parallelism_threads(os.cpu_count())
tf.keras.utils.set_random_seed(42)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
BASE_DIR = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_DIR = os.path.join(BASE_DIR, "train_images")
TEST_DIR = os.path.join(BASE_DIR, "test_images")

with open(os.path.join(BASE_DIR, "label_num_to_disease_map.json")) as f:
    label_map = json.load(f)

train_df = pd.read_csv(os.path.join(BASE_DIR, "train.csv"))
print(f"Training samples: {len(train_df)}")




## === cell 2
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 256
NUM_CLASSES = 5
EPOCHS_FROZEN = 12  # training with backbone frozen
EPOCHS_UNFROZEN = 5  # fine‑tuning after unfreezing

train_df["subset"] = np.random.rand(len(train_df)) < 0.8
train_subset = train_df[train_df["subset"]].reset_index(drop=True)
val_subset = train_df[~train_df["subset"]].reset_index(drop=True)




## === cell 3
AUTOTUNE = tf.data.AUTOTUNE


def _load_and_preprocess(path, label=None, augment=False):
    """Read an image file, decode, resize and rescale. Apply augmentation if required."""
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])
    image = tf.cast(image, tf.float32) / 255.0

    if augment:
        image = tf.image.random_flip_left_right(image, seed=42)
        angles = tf.random.uniform([], -0.111, 0.111, seed=42)
        image = tfa.image.rotate(image, angles, fill_mode="nearest")
        scales = tf.random.uniform([], 0.8, 1.2, seed=42)
        new_size = tf.cast(tf.round([IMG_HEIGHT, IMG_WIDTH] * scales), tf.int32)
        image = tf.image.resize(image, new_size)
        image = tf.image.resize_with_crop_or_pad(image, IMG_HEIGHT, IMG_WIDTH)
        dx = tf.random.uniform([], -0.1, 0.1, seed=42) * IMG_WIDTH
        dy = tf.random.uniform([], -0.1, 0.1, seed=42) * IMG_HEIGHT
        image = tfa.image.translate(image, [dx, dy], fill_mode="nearest")
    if label is None:
        return image
    else:
        return image, label


train_paths = [os.path.join(TRAIN_DIR, fname) for fname in train_subset["image_id"]]
train_labels = train_subset["label"].values.astype(np.int32)

val_paths = [os.path.join(TRAIN_DIR, fname) for fname in val_subset["image_id"]]
val_labels = val_subset["label"].values.astype(np.int32)

train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
train_ds = train_ds.shuffle(
    buffer=len(train_paths), seed=42, reshuffle_each_iteration=False
)
train_ds = train_ds.map(
    lambda p, l: _load_and_preprocess(p, l, augment=True), num_parallel_calls=AUTOTUNE
)
train_ds = train_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
val_ds = val_ds.map(
    lambda p, l: _load_and_preprocess(p, l, augment=False), num_parallel_calls=AUTOTUNE
)
val_ds = val_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/358612407.py in <cell line: 0>()
     40 # Training dataset with augmentation
     41 train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
---> 42 train_ds = train_ds.shuffle(
     43     buffer=len(train_paths), seed=42, reshuffle_each_iteration=False
     44 )

TypeError: DatasetV2.shuffle() got an unexpected keyword argument 'buffer'

## === cell 4
data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal", seed=42),
        tf.keras.layers.RandomRotation(0.1, seed=42),  # ~20°
        tf.keras.layers.RandomZoom(0.2, seed=42),
        tf.keras.layers.RandomTranslation(0.1, 0.1, seed=42),
    ]
)

inputs = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
x = data_augmentation(inputs)  # augmentation attached to the model
base_model = EfficientNetB0(weights="imagenet", include_top=False, input_tensor=x)
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.2)(x)
outputs = Dense(NUM_CLASSES, activation="softmax")(x)

model = Model(inputs=inputs, outputs=outputs)
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()




## === cell 5
model.fit(
    train_ds,
    epochs=EPOCHS_FROZEN,
    validation_data=val_ds,
    verbose=1,
)

base_model.trainable = True
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),  # smaller LR for fine‑tuning
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.fit(
    train_ds,
    epochs=EPOCHS_UNFROZEN,
    validation_data=val_ds,
    verbose=1,
)




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3247866995.py in <cell line: 0>()
      2     train_ds,
      3     epochs=EPOCHS_FROZEN,
----> 4     validation_data=val_ds,
      5     verbose=1,
      6 )

NameError: name 'val_ds' is not defined

## === cell 6
test_files = [os.path.join(TEST_DIR, fname) for fname in os.listdir(TEST_DIR)]

test_ds = tf.data.Dataset.from_tensor_slices(test_files)
test_ds = test_ds.map(
    lambda p: _load_and_preprocess(p, augment=False), num_parallel_calls=AUTOTUNE
)
test_ds = test_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

pred_probs = model.predict(test_ds, verbose=1)
pred_labels = np.argmax(pred_probs, axis=1)

submission = pd.DataFrame(
    {"image_id": [os.path.basename(p) for p in test_files], "label": pred_labels}
)

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
FailedPreconditionError                   Traceback (most recent call last)
/tmp/ipykernel_11/2514242363.py in <cell line: 0>()
      8 test_ds = test_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
      9 
---> 10 pred_probs = model.predict(test_ds, verbose=1)
     11 pred_labels = np.argmax(pred_probs, axis=1)
     12 

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py in raise_from_not_ok_status(e, name)
   6000 def raise_from_not_ok_status(e, name) -> NoReturn:
   6001   e.message += (" name: " + str(name if name is not None else ""))
-> 6002   raise core._status_to_exception(e) from None  # pylint: disable=protected-access
   6003 
   6004 

FailedPreconditionError: {{function_node __wrapped__IteratorGetNext_output_types_1_device_/job:localhost/replica:0/task:0/device:CPU:0}} /kaggle/input/cassava-leaf-disease-classification/test_images/test_images; Is a directory
	 [[{{node ReadFile}}]] [Op:IteratorGetNext] name: 

## === cell 7
print(pd.read_csv(submission_path).head())

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/428350302.py in <cell line: 0>()
----> 1 print(pd.read_csv(submission_path).head())

NameError: name 'submission_path' is not defined
