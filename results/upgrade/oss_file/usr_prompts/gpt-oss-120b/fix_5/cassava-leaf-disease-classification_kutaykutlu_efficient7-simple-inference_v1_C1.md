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

0.8738289513448172

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.11584) has done: 'I remove the failing KaggleDatasets import, add the missing K import for the swish activation, replace the unavailable pretrained model with a minimal EfficientNet‑B3‑based model (using ImageNet weights) so that a model object exists, and fix the FixedDropout implementation. These changes make the notebook run end‑to‑end and generate a proper `submission.csv` while keeping the original pipeline logic intact.'

# 9. Code solution

## === cell 0
import math, re, os
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from kaggle_datasets import KaggleDatasets
except Exception:
    KaggleDatasets = None
from tensorflow import keras
from tensorflow.keras import backend as K
from functools import partial
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.layers import Dense, Dropout, Input, GlobalAveragePooling2D

from tensorflow.keras.mixed_precision import experimental as mixed_precision

policy = mixed_precision.Policy("mixed_float16")
mixed_precision.set_policy(policy)

print("Tensorflow version " + tf.__version__)

AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 32  # increased batch size reduces number of steps per epoch
IMAGE_SIZE = [512, 512]
CLASSES = ["0", "1", "2", "3", "4"]




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def swish_activation(x):
    return K.sigmoid(x) * x


class FixedDropout(tf.keras.layers.Dropout):
    def _get_noise_shape(self, inputs):
        if self.noise_shape is None:
            return None
        symbolic_shape = tf.keras.backend.shape(inputs)
        return tuple(
            symbolic_shape[axis] if shape is None else shape
            for axis, shape in enumerate(self.noise_shape)
        )


def decode_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.reshape(image, [*IMAGE_SIZE, 3])
    return image


def read_tfrecord(example, labeled):
    tfrecord_format = (
        {
            "image": tf.io.FixedLenFeature([], tf.string),
            "target": tf.io.FixedLenFeature([], tf.int64),
        }
        if labeled
        else {
            "image": tf.io.FixedLenFeature([], tf.string),
            "image_name": tf.io.FixedLenFeature([], tf.string),
        }
    )
    example = tf.io.parse_single_example(example, tfrecord_format)
    image = decode_image(example["image"])
    if labeled:
        label = tf.cast(example["target"], tf.int32)
        return image, label
    idnum = example["image_name"]
    return image, idnum


def load_dataset(filenames, labeled=True, ordered=False):
    ignore_order = tf.data.Options()
    if not ordered:
        ignore_order.experimental_deterministic = False  # speed‑up
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTOTUNE)
    dataset = dataset.with_options(ignore_order)
    dataset = dataset.map(
        partial(read_tfrecord, labeled=labeled), num_parallel_calls=AUTOTUNE
    )
    return dataset


TEST_FILENAMES = tf.io.gfile.glob(
    "../input/cassava-leaf-disease-classification/test_tfrecords/ld_test*.tfrec"
)


def count_data_items(filenames):
    n = [
        int(re.compile(r"-([0-9]*)\.").search(filename).group(1))
        for filename in filenames
    ]
    return np.sum(n)


NUM_TEST_IMAGES = count_data_items(TEST_FILENAMES)


def data_augment(image, label):
    image = tf.image.random_flip_left_right(image)
    return image, label


def get_test_dataset(ordered=False):
    dataset = load_dataset(TEST_FILENAMES, labeled=False, ordered=ordered)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTOTUNE)
    return dataset


print("Test data shapes:")
for image, idnum in get_test_dataset().take(3):
    print(image.numpy().shape, idnum.numpy().shape)
print("Test data IDs:", idnum.numpy().astype("U"))




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/640338909.py in <cell line: 0>()
     83 
     84 print("Test data shapes:")
---> 85 for image, idnum in get_test_dataset().take(3):
     86     print(image.numpy().shape, idnum.numpy().shape)
     87 print("Test data IDs:", idnum.numpy().astype("U"))

/tmp/ipykernel_11/640338909.py in get_test_dataset(ordered)
     76 
     77 def get_test_dataset(ordered=False):
---> 78     dataset = load_dataset(TEST_FILENAMES, labeled=False, ordered=ordered)
     79     dataset = dataset.batch(BATCH_SIZE)
     80     dataset = dataset.prefetch(AUTOTUNE)

/tmp/ipykernel_11/640338909.py in load_dataset(filenames, labeled, ordered)
     46     if not ordered:
     47         ignore_order.experimental_deterministic = False  # speed‑up
---> 48     dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTOTUNE)
     49     dataset = dataset.with_options(ignore_order)
     50     dataset = dataset.map(

NameError: name 'AUTOTUNE' is not defined

## === cell 2
base_model = EfficientNetB3(
    weights="imagenet", include_top=False, input_shape=(*IMAGE_SIZE, 3)
)
base_model.trainable = False
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation=swish_activation)(x)
x = Dropout(0.3)(x)
output = Dense(len(CLASSES), activation="softmax")(x)
model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)
print("Model built successfully.")




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3976663578.py in <cell line: 0>()
      1 base_model = EfficientNetB3(
----> 2     weights="imagenet", include_top=False, input_shape=(*IMAGE_SIZE, 3)
      3 )
      4 base_model.trainable = False
      5 x = base_model.output

NameError: name 'IMAGE_SIZE' is not defined

## === cell 3
TRAIN_FILENAMES = tf.io.gfile.glob(
    "../input/cassava-leaf-disease-classification/train_tfrecords/ld_train*.tfrec"
)

raw_train_ds = load_dataset(TRAIN_FILENAMES, labeled=True, ordered=False)
raw_train_ds = raw_train_ds.cache()
raw_train_ds = raw_train_ds.shuffle(1000, seed=42, reshuffle_each_iteration=False)

total_train_items = count_data_items(TRAIN_FILENAMES)
val_count = int(0.20 * total_train_items)

val_ds = raw_train_ds.take(val_count).batch(BATCH_SIZE).prefetch(AUTOTUNE)
train_ds = raw_train_ds.skip(val_count).batch(BATCH_SIZE).prefetch(AUTOTUNE)

print(
    f"Training on {total_train_items - val_count} samples, validating on {val_count} samples."
)

model.fit(train_ds, validation_data=val_ds, epochs=3, verbose=2)




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1800913774.py in <cell line: 0>()
      4 
      5 # Load, parse, and cache the training data (no extra cast needed – decode_image already returns float32)
----> 6 raw_train_ds = load_dataset(TRAIN_FILENAMES, labeled=True, ordered=False)
      7 raw_train_ds = raw_train_ds.cache()
      8 raw_train_ds = raw_train_ds.shuffle(1000, seed=42, reshuffle_each_iteration=False)

/tmp/ipykernel_11/640338909.py in load_dataset(filenames, labeled, ordered)
     46     if not ordered:
     47         ignore_order.experimental_deterministic = False  # speed‑up
---> 48     dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTOTUNE)
     49     dataset = dataset.with_options(ignore_order)
     50     dataset = dataset.map(

NameError: name 'AUTOTUNE' is not defined

## === cell 4
test_ds = get_test_dataset(ordered=True)

print("Computing predictions...")
test_images_ds = test_ds.map(lambda image, idnum: image)
probabilities = model.predict(test_images_ds, verbose=0)
predictions = np.argmax(probabilities, axis=-1)
print(predictions)

print("Generating submission.csv file...")
test_ids_ds = test_ds.map(lambda image, idnum: idnum).unbatch()
test_ids = next(iter(test_ids_ds.batch(NUM_TEST_IMAGES))).numpy().astype("U")
np.savetxt(
    "submission.csv",
    np.rec.fromarrays([test_ids, predictions]),
    fmt=["%s", "%d"],
    delimiter=",",
    header="image_id,label",
    comments="",
)

from csv import reader

with open("submission.csv", "r") as read_obj:
    csv_reader = reader(read_obj)
    for row in csv_reader:
        print(row)

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3718938308.py in <cell line: 0>()
      1 # Use the single test‑dataset definition from earlier.
----> 2 test_ds = get_test_dataset(ordered=True)
      3 
      4 print("Computing predictions...")
      5 # Extract only images for prediction.

/tmp/ipykernel_11/640338909.py in get_test_dataset(ordered)
     76 
     77 def get_test_dataset(ordered=False):
---> 78     dataset = load_dataset(TEST_FILENAMES, labeled=False, ordered=ordered)
     79     dataset = dataset.batch(BATCH_SIZE)
     80     dataset = dataset.prefetch(AUTOTUNE)

/tmp/ipykernel_11/640338909.py in load_dataset(filenames, labeled, ordered)
     46     if not ordered:
     47         ignore_order.experimental_deterministic = False  # speed‑up
---> 48     dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTOTUNE)
     49     dataset = dataset.with_options(ignore_order)
     50     dataset = dataset.map(

NameError: name 'AUTOTUNE' is not defined
