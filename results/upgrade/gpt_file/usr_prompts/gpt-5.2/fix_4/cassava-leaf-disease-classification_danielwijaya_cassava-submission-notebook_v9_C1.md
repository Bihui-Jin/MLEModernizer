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

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1

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

0.8850105772136597

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from functools import partial
from sklearn.model_selection import train_test_split
import re
import random

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

print("TensorFlow:", tf.__version__)

print("Input root exists:", os.path.exists("/kaggle/input"))
print(
    "Competition path exists:",
    os.path.exists("/kaggle/input/cassava-leaf-disease-classification"),
)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
GCS_PATH = "/kaggle/input/cassava-leaf-disease-classification"

AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 64  # original was 16*8
IMAGE_SIZE = [512, 512]
NUM_CLASSES = 5


def dataset_sizes(filenames):
    n = [int(re.compile(r"-([0-9]*)\.").search(fn).group(1)) for fn in filenames]
    return int(np.sum(n))


TRAIN_FILENAMES = tf.io.gfile.glob(GCS_PATH + "/train_tfrecords/ld_train*.tfrec")
TEST_FILENAMES = tf.io.gfile.glob(GCS_PATH + "/test_tfrecords/ld_test*.tfrec")
NUM_TRAIN_IMAGES = dataset_sizes(TRAIN_FILENAMES)
NUM_TEST_IMAGES = dataset_sizes(TEST_FILENAMES)

print("Train tfrecords:", len(TRAIN_FILENAMES), "images:", NUM_TRAIN_IMAGES)
print("Test  tfrecords:", len(TEST_FILENAMES), "images:", NUM_TEST_IMAGES)



## === cell 2
test_df = pd.read_csv(GCS_PATH + "/sample_submission.csv")
print(test_df.head())
print("sample_submission rows:", len(test_df))




## === cell 3
@tf.function
def decode_img(img):
    img = tf.io.decode_jpeg(img, channels=3)  # uint8 [H,W,3]
    img = tf.image.resize(img, IMAGE_SIZE, method="bilinear", antialias=False)
    img = tf.cast(img, tf.float32) / 255.0
    return img


@tf.function
def read_tfrecord_labeled(example):
    tfrec_format = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "target": tf.io.FixedLenFeature([], tf.int64),
    }
    example = tf.io.parse_single_example(example, tfrec_format)
    img = decode_img(example["image"])
    label = tf.cast(example["target"], tf.int32)
    return img, label


@tf.function
def read_tfrecord_unlabeled(example):
    tfrec_format = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "image_name": tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(example, tfrec_format)
    img = decode_img(example["image"])
    image_id = example["image_name"]
    return img, image_id


def _dataset_options(ordered: bool):
    options = tf.data.Options()
    options.experimental_deterministic = bool(ordered)
    options.experimental_optimization.apply_default_optimizations = True
    options.experimental_optimization.map_parallelization = True
    options.experimental_optimization.parallel_batch = True
    options.experimental_optimization.autotune_buffers = True
    return options


def load_dataset(filenames, labeled=True, ordered=False):
    ds = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTOTUNE).with_options(
        _dataset_options(ordered)
    )
    if labeled:
        ds = ds.map(read_tfrecord_labeled, num_parallel_calls=AUTOTUNE)
    else:
        ds = ds.map(read_tfrecord_unlabeled, num_parallel_calls=AUTOTUNE)
    return ds


def get_training_data(ordered=False):
    ds = load_dataset(TRAIN_FILENAMES, labeled=True, ordered=ordered)
    ds = ds.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.cache("/kaggle/working/cache_train_ds")
    ds = ds.prefetch(AUTOTUNE)
    return ds


def get_test_data(ordered=False):
    ds = load_dataset(TEST_FILENAMES, labeled=False, ordered=ordered)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.cache("/kaggle/working/cache_test_ds")
    ds = ds.prefetch(AUTOTUNE)
    return ds




## === cell 4
train_ds = get_training_data(ordered=False)

val_batches = max(1, int((0.1 * NUM_TRAIN_IMAGES) // BATCH_SIZE))
val_ds = train_ds.take(val_batches)
train_ds2 = train_ds.skip(val_batches)

data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomZoom(0.1),
    ],
    name="augment",
)

inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))
x = data_augmentation(inputs)
x = x * 255.0
x = tf.keras.applications.efficientnet.preprocess_input(x)

base = tf.keras.applications.EfficientNetB5(
    include_top=False,
    weights="imagenet",
    input_tensor=None,
    input_shape=(*IMAGE_SIZE, 3),
    pooling="avg",
)
base.trainable = False  # start with frozen backbone for stability/speed

x = base(x, training=False)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)
model_20 = tf.keras.Model(inputs, outputs)

model_20.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"],
)

EPOCHS_STAGE1 = 2
history1 = model_20.fit(
    train_ds2, validation_data=val_ds, epochs=EPOCHS_STAGE1, verbose=1
)

base.trainable = True
for layer in base.layers[:-40]:
    layer.trainable = False

model_20.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"],
)
EPOCHS_STAGE2 = 1
history2 = model_20.fit(
    train_ds2, validation_data=val_ds, epochs=EPOCHS_STAGE2, verbose=1
)

print("Computing predictions...")
test_ds = get_test_data(ordered=True)

test_images_ds = test_ds.map(lambda image, image_id: image, num_parallel_calls=AUTOTUNE)
probabilities = model_20.predict(test_images_ds, verbose=1)
predictions = np.argmax(probabilities, axis=-1).astype(np.int32)
print("Predictions shape:", predictions.shape)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_55/3945315221.py in <cell line: 0>()
----> 1 train_ds = get_training_data(ordered=False)
      2 
      3 val_batches = max(1, int((0.1 * NUM_TRAIN_IMAGES) // BATCH_SIZE))
      4 val_ds = train_ds.take(val_batches)
      5 train_ds2 = train_ds.skip(val_batches)

/tmp/ipykernel_55/2333481449.py in get_training_data(ordered)
     59 
     60 def get_training_data(ordered=False):
---> 61     ds = load_dataset(TRAIN_FILENAMES, labeled=True, ordered=ordered)
     62     ds = ds.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
     63     ds = ds.batch(BATCH_SIZE, drop_remainder=False)

/tmp/ipykernel_55/2333481449.py in load_dataset(filenames, labeled, ordered)
     49 def load_dataset(filenames, labeled=True, ordered=False):
     50     ds = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTOTUNE).with_options(
---> 51         _dataset_options(ordered)
     52     )
     53     if labeled:

/tmp/ipykernel_55/2333481449.py in _dataset_options(ordered)
     43     options.experimental_optimization.map_parallelization = True
     44     options.experimental_optimization.parallel_batch = True
---> 45     options.experimental_optimization.autotune_buffers = True
     46     return options
     47 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/util/options.py in __setattr__(self, name, value)
     59       object.__setattr__(self, name, value)
     60     else:
---> 61       raise AttributeError("Cannot set the property {} on {}.".format(
     62           name,
     63           type(self).__name__))

AttributeError: Cannot set the property autotune_buffers on OptimizationOptions.

## === cell 5
print("Generating submission.csv file...")

test_id_batches = []
for _, image_id in test_ds:
    test_id_batches.append(image_id.numpy())
test_ids = np.concatenate(test_id_batches, axis=0).astype("U")

pred_df = pd.DataFrame({"image_id": test_ids, "label": predictions.astype(int)})
sub = test_df[["image_id"]].merge(pred_df, on="image_id", how="left")

assert (
    sub["label"].notna().all()
), "Some test ids in sample_submission were not predicted."
assert len(sub) == len(test_df)

out_path = "submission.csv"
sub.to_csv(out_path, index=False)
print(sub.head())
print("Wrote:", out_path, "rows:", len(sub))
with open(out_path, "r") as f:
    for _ in range(5):
        print(f.readline().rstrip())

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2717329417.py in <cell line: 0>()
      4 # Instead, iterate batches once and concatenate; deterministic order preserved by ordered=True.
      5 test_id_batches = []
----> 6 for _, image_id in test_ds:
      7     test_id_batches.append(image_id.numpy())
      8 test_ids = np.concatenate(test_id_batches, axis=0).astype("U")

NameError: name 'test_ds' is not defined
