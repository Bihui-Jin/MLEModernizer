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

0.8838017527954065

# 6. Current score

0.51495

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.61099) has done: 'I remove the failing external EfficientNet/kerasapplications imports that crash due to an incompatibility with the current protobuf/TensorFlow stack, since they are not needed for inference here. Because the referenced pre-trained `.h5` file is not available in your environment, I replace that load with a minimal on-the-fly TFRecord-based training + inference pipeline using the same TFRecord decoding logic you already have, so the notebook runs end-to-end. I also fix a real bug in `load_dataset()` where the dataset options weren’t applied, and ensure test ordering is deterministic so `image_id` aligns with predictions. Finally, I always write a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.55717) has done: 'I fix the TensorFlow import crash caused by the protobuf 6.x incompatibility by forcing the pure-Python protobuf implementation before importing TensorFlow, which resolves the `MessageFactory.GetPrototype` error. Then I keep your TFRecord decoding/training/inference pipeline intact, but make a minimal score-improving change by unfreezing the EfficientNet backbone after a short warmup and fine-tuning with a low learning rate (same architecture, same loss/metric). I also ensure the test dataset remains deterministic so `image_id` ordering stays aligned with predictions, and that `submission.csv` is always written with the correct columns and row order.'
- What this solution (achieved 0.51981) has done: 'You’re hitting the TensorFlow import crash because `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION` is forced to `"cpp"`, but in this environment protobuf’s C++ extension (`google.protobuf.pyext._message`) isn’t available; switching to the pure-Python protobuf implementation fixes TF import and unblocks all downstream cells. After TF loads, the later `NameError`s disappear because `tf`, `GCS_PATH`, and the TFRecord filename lists be defined as intended. I also make the test TFRecord reading deterministic and keep the submission aligned to `sample_submission.csv` ordering so `image_id` matches predictions, ensuring a valid `submission.csv` is always written. Core model/training/inference logic stays the same (EfficientNetB0 warmup + fine-tune on TFRecords).'
- What this solution (achieved 0.51495) has done: 'You’re failing immediately on TensorFlow import because protobuf 6.x removed/changed `MessageFactory.GetPrototype`, and setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` alone doesn’t prevent TensorFlow from hitting that path in this environment. The minimal reliable fix is to force the pure-Python protobuf module early (before importing TF) and pin the implementation version env var so TF doesn’t attempt the C++/fast path. After TF imports, the rest of your pipeline can run unchanged, and it again write a correctly ordered `submission.csv`. This change is score-neutral (it only fixes the runtime crash), so your achieved score behavior should remain comparable once it runs end-to-end.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"

import re
import random
from functools import partial

import numpy as np
import pandas as pd

import tensorflow as tf
import matplotlib.pyplot as plt

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("TensorFlow:", tf.__version__)
print(
    "protobuf python implementation:",
    os.environ.get("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"),
    "version:",
    os.environ.get("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"),
)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
AUTOTUNE = tf.data.experimental.AUTOTUNE
GCS_PATH = "/kaggle/input/cassava-leaf-disease-classification"

BATCH_SIZE = 32
IMAGE_SIZE = [512, 512]
NUM_CLASSES = 5

TRAIN_FILENAMES = tf.io.gfile.glob(GCS_PATH + "/train_tfrecords/ld_train*.tfrec")
TEST_FILENAMES = tf.io.gfile.glob(GCS_PATH + "/test_tfrecords/ld_test*.tfrec")

TRAIN_FILENAMES = sorted(TRAIN_FILENAMES)
TEST_FILENAMES = sorted(TEST_FILENAMES)


def dataset_sizes(filenames):
    n = [int(re.compile(r"-([0-9]*)\.").search(fn).group(1)) for fn in filenames]
    return int(np.sum(n))


NUM_TRAIN_IMAGES = dataset_sizes(TRAIN_FILENAMES)
NUM_TEST_IMAGES = dataset_sizes(TEST_FILENAMES)

print(
    "TFRecords:",
    len(TRAIN_FILENAMES),
    "train files;",
    len(TEST_FILENAMES),
    "test files",
)
print("Images:", NUM_TRAIN_IMAGES, "train;", NUM_TEST_IMAGES, "test")



## === cell 2
sample_sub_path = GCS_PATH + "/sample_submission.csv"
test_df = pd.read_csv(sample_sub_path)
print(test_df.head(), "\nrows:", len(test_df))




## === cell 3
def to_float32(image, label):
    return tf.cast(image, tf.float32), label


def decode_img(img):
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.resize(
        img, IMAGE_SIZE, method=tf.image.ResizeMethod.BILINEAR, antialias=True
    )
    img = tf.cast(img, tf.float32) / 255.0
    return img


def read_tfrecord(example, labeled):
    if labeled:
        TFREC_FORMAT = {
            "image": tf.io.FixedLenFeature([], tf.string),
            "target": tf.io.FixedLenFeature([], tf.int64),
        }
    else:
        TFREC_FORMAT = {
            "image": tf.io.FixedLenFeature([], tf.string),
            "image_name": tf.io.FixedLenFeature([], tf.string),
        }
    example = tf.io.parse_single_example(example, TFREC_FORMAT)
    img = decode_img(example["image"])
    if labeled:
        label = tf.cast(example["target"], tf.int32)
        return img, label
    else:
        id_num = example["image_name"]
        return img, id_num


def load_dataset(filenames, labeled=True, ordered=False):
    options = tf.data.Options()
    options.experimental_deterministic = bool(ordered)

    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTOTUNE)
    dataset = dataset.with_options(options)  # ensure options are applied
    dataset = dataset.map(
        partial(read_tfrecord, labeled=labeled), num_parallel_calls=AUTOTUNE
    )
    return dataset


def get_training_data():
    ds = load_dataset(TRAIN_FILENAMES, labeled=True, ordered=False)
    ds = ds.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
    ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return ds


def get_test_data(ordered=False):
    ds = load_dataset(TEST_FILENAMES, labeled=False, ordered=ordered)
    ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return ds


train_ds = get_training_data()
test_ds_ordered = get_test_data(ordered=True)



## === cell 4
base = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(*IMAGE_SIZE, 3),
)
base.trainable = False

inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))
x = tf.keras.applications.efficientnet.preprocess_input(inputs)
x = base(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2, seed=SEED)(x)
outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)
model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["sparse_categorical_accuracy"],
)

EPOCHS_WARMUP = 2
EPOCHS_FINETUNE = 2
steps_per_epoch = max(1, NUM_TRAIN_IMAGES // BATCH_SIZE)

print("Training (warmup, frozen backbone)...")
history1 = model.fit(
    train_ds,
    epochs=EPOCHS_WARMUP,
    steps_per_epoch=steps_per_epoch,
    verbose=1,
)

print("Fine-tuning (unfreeze backbone, low LR)...")
base.trainable = True
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["sparse_categorical_accuracy"],
)

history2 = model.fit(
    train_ds,
    epochs=EPOCHS_FINETUNE,
    steps_per_epoch=steps_per_epoch,
    verbose=1,
)

print("Computing predictions...")
test_images_ds = test_ds_ordered.map(lambda image, idnum: image)
probabilities = model.predict(test_images_ds, verbose=1)
predictions = np.argmax(probabilities, axis=-1).astype(np.int64)

print("Pred shape:", predictions.shape, "unique:", np.unique(predictions))



## === cell 5
print("Generating submission.csv file...")

test_ids_ds = test_ds_ordered.map(lambda image, idnum: idnum).unbatch()
test_ids = next(iter(test_ids_ds.batch(NUM_TEST_IMAGES))).numpy().astype("U")

assert len(test_ids) == len(predictions), (len(test_ids), len(predictions))

sub = pd.DataFrame({"image_id": test_ids, "label": predictions})
sub = sub.set_index("image_id").reindex(test_df["image_id"]).reset_index()

if sub["label"].isna().any():
    missing = sub[sub["label"].isna()]["image_id"].head(10).tolist()
    raise ValueError(
        f"Missing predictions for some image_ids after reindex. Examples: {missing}"
    )

sub["label"] = sub["label"].astype(np.int64)

sub_path = "submission.csv"
sub.to_csv(sub_path, index=False)
print("Wrote:", sub_path, "rows:", len(sub))
print(sub.head())
