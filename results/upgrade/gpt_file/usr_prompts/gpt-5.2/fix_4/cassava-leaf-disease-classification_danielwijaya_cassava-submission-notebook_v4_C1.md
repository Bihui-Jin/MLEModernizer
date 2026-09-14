# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.8851616802659413

# 6. Current score

0.74963

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.2003) has done: 'I remove the failing `kaggle_datasets`/protobuf-dependent import and the missing third-party EfficientNet package imports, since they are not needed to read TFRecords or build a model in TF 2.18. I also replace the missing external `.h5` model load with an in-notebook EfficientNetB6 model (same overall intent: EfficientNet-based classifier) and load pretrained ImageNet weights so the pipeline produces sensible predictions without extra files. I fix the TFRecord decode logic to resize instead of reshaping (the JPEGs are variable-sized, and reshaping can silently corrupt) and ensure the test IDs align with predictions. Finally, I guarantee a valid `submission.csv` with the exact required columns is written to the working directory.'
- What this solution (achieved 0.2003) has done: 'I fix the TensorFlow import crash caused by the protobuf 6.x incompatibility by explicitly forcing the pure-Python protobuf implementation before importing TensorFlow. Then I correct a preprocessing bug that currently double-normalizes inputs (Rescaling + EfficientNet preprocess), which severely hurts accuracy and is consistent with the very low current score. Finally, I keep the TFRecord decoding and submission alignment logic intact, but make the submission ID extraction robust (no reliance on `NUM_TEST_IMAGES`), ensuring a valid `submission.csv` is always written.'
- What this solution (achieved 0.74963) has done: 'I fix the TensorFlow import crash by pinning protobuf to the pure-Python implementation and forcing a compatible python-protobuf version before importing TensorFlow (the current `MessageFactory.GetPrototype` error is a known protobuf 6.x incompatibility). Then I keep your model and TFRecord pipeline intact, but add the missing training step on `train_tfrecords` so the EfficientNetB6 head is actually fit to cassava labels (your current code never trains, which explains the ~0.20 accuracy). Finally, I ensure the test dataset order is deterministic and that the submission is written with correct IDs/labels to `submission.csv` in the working directory.'

# 9. Code solution

## === cell 0
import os
import sys
import subprocess

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

try:
    import google.protobuf  # noqa: F401
    import pkgutil, importlib
    import pkg_resources

    ver = pkg_resources.get_distribution("protobuf").version
    if ver.startswith("6."):
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "protobuf<5"]
        )
        importlib.invalidate_caches()
        for m in list(sys.modules.keys()):
            if m.startswith("google.protobuf"):
                del sys.modules[m]
except Exception as e:
    print("protobuf adjustment warning:", repr(e))

import numpy as np
import pandas as pd

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames[:5]:
        print(os.path.join(dirname, filename))




## === cell 1
import tensorflow as tf
import matplotlib.pyplot as plt
from functools import partial
import re
import random

print("TensorFlow:", tf.__version__)

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)




## === cell 2
NUM_CLASSES = 5
IMAGE_SIZE = [512, 512]

inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3), name="image")
x = tf.keras.applications.efficientnet.preprocess_input(inputs)

base = tf.keras.applications.EfficientNetB6(
    include_top=False,
    weights="imagenet",
    input_tensor=x,
    pooling="avg",
)
base.trainable = False

outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax", name="pred")(
    base.output
)
model_15 = tf.keras.Model(inputs=inputs, outputs=outputs, name="effnetb6_cassava")

model_15.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)
model_15.summary()




## === cell 3
DATA_PATH = "/kaggle/input/cassava-leaf-disease-classification"

test_df = pd.read_csv(f"{DATA_PATH}/sample_submission.csv")
print(test_df.head())
print("Sample submission rows:", len(test_df))

AUTOTUNE = tf.data.experimental.AUTOTUNE
GCS_PATH = DATA_PATH

BATCH_SIZE = 16 * 8  # original value
CLASSES = ["1", "2", "3", "4", "5"]  # kept although not used directly


def dataset_sizes(filenames):
    n = [int(re.compile(r"-([0-9]*)\.").search(fn).group(1)) for fn in filenames]
    return int(np.sum(n))


TEST_FILENAMES = tf.io.gfile.glob(GCS_PATH + "/test_tfrecords/ld_test*.tfrec")
TEST_FILENAMES = sorted(TEST_FILENAMES)
NUM_TEST_IMAGES = dataset_sizes(TEST_FILENAMES)

TRAIN_FILENAMES = tf.io.gfile.glob(GCS_PATH + "/train_tfrecords/ld_train*.tfrec")
TRAIN_FILENAMES = sorted(TRAIN_FILENAMES)
NUM_TRAIN_IMAGES = dataset_sizes(TRAIN_FILENAMES)

print("Num TEST TFRecord files:", len(TEST_FILENAMES))
print("NUM_TEST_IMAGES:", NUM_TEST_IMAGES)
print("Num TRAIN TFRecord files:", len(TRAIN_FILENAMES))
print("NUM_TRAIN_IMAGES:", NUM_TRAIN_IMAGES)




## === cell 4
def to_float32(image, label_or_id):
    return tf.cast(image, tf.float32), label_or_id


def decode_img(img_bytes):
    img = tf.io.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, IMAGE_SIZE, method="bilinear")
    img = tf.cast(img, tf.float32)
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
        image_id = example["image_name"]
        return img, image_id


def load_dataset(filenames, labeled=True, ordered=False):
    options = tf.data.Options()
    if not ordered:
        options.experimental_deterministic = False
    else:
        options.experimental_deterministic = True

    ds = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTOTUNE)
    ds = ds.with_options(options)
    ds = ds.map(partial(read_tfrecord, labeled=labeled), num_parallel_calls=AUTOTUNE)
    return ds


def get_train_data(ordered=False):
    ds = load_dataset(filenames=TRAIN_FILENAMES, labeled=True, ordered=ordered)
    if not ordered:
        ds = ds.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.map(to_float32, num_parallel_calls=AUTOTUNE)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def get_test_data(ordered=False):
    ds = load_dataset(filenames=TEST_FILENAMES, labeled=False, ordered=ordered)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.map(to_float32, num_parallel_calls=AUTOTUNE)
    ds = ds.prefetch(AUTOTUNE)
    return ds




## === cell 5
train_ds = get_train_data(ordered=False)

EPOCHS = 1
print("Training (head only) for", EPOCHS, "epoch(s)...")
steps_per_epoch = int(np.ceil(NUM_TRAIN_IMAGES / BATCH_SIZE))
model_15.fit(train_ds, epochs=EPOCHS, steps_per_epoch=steps_per_epoch, verbose=1)




## === cell 6
test_ds = get_test_data(ordered=True)

print("Computing predictions...")
test_images_ds = test_ds.map(lambda image, image_id: image, num_parallel_calls=AUTOTUNE)

prob1 = model_15.predict(test_images_ds, verbose=1)
probabilities = prob1
predictions = np.argmax(probabilities, axis=-1).astype(np.int64)

print("Predictions shape:", predictions.shape)
print("Predictions (first 20):", predictions[:20])




## === cell 7
print("Generating submission.csv file...")

test_ids_ds = test_ds.map(
    lambda image, image_id: image_id, num_parallel_calls=AUTOTUNE
).unbatch()
test_ids = np.concatenate([x.numpy() for x in test_ids_ds.batch(2048)]).astype("U")

if len(test_ids) != len(predictions):
    raise RuntimeError(f"Length mismatch: ids={len(test_ids)} preds={len(predictions)}")

sub = pd.DataFrame({"image_id": test_ids, "label": predictions})
sub = test_df[["image_id"]].merge(sub, on="image_id", how="left")
if sub["label"].isna().any():
    raise RuntimeError(
        "Some test image_ids did not receive predictions after merge; check ordering/ids."
    )

sub["label"] = sub["label"].astype(int)
sub.to_csv("submission.csv", index=False)

print(sub.head())
print("Wrote submission.csv with rows:", len(sub))
with open("submission.csv", "r") as f:
    for _ in range(5):
        print(f.readline().strip())
