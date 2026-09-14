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

0.8927168328800241

# 6. Current score

0.05531

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.10987) has done: 'The runtime fails early due to incompatible/unused imports (`kaggle_datasets`, custom EfficientNet/keras_applications) and missing external pretrained model files, which prevents `model_15` etc. from being defined. To make the notebook run end-to-end and still follow the same “TFRecord → EfficientNet → predict → submission” semantics, I replace the unavailable pretrained-model loading with a single built-in `tf.keras.applications.EfficientNetB0` model and keep the same inference/prediction flow. I also fix TFRecord decoding to avoid forcing a wrong fixed shape (which can break) by resizing after decode, and I correct `dataset.with_options(...)` (must be assigned) for determinism. Finally, I ensure the generated `submission.csv` matches `sample_submission.csv` ordering and has correct `image_id,label` columns.'
- What this solution (achieved 0.05531) has done: 'We need to fix the TensorFlow import crash (`AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'`), which is a protobuf/TensorFlow incompatibility that prevents the entire pipeline from running. The safest minimal fix in this Kaggle environment is to force the Python protobuf implementation before importing TensorFlow and restart the TF import cleanly, keeping the rest of your TFRecord → EfficientNet → predict → submission flow unchanged. I also add a tiny defensive check to ensure TFRecord filenames are found and keep submission generation identical (same columns/order as `sample_submission.csv`). These changes are score-neutral by themselves, but they unblock execution so you can produce a valid submission and then iterate toward the target score.'
- What this solution (achieved 0.05531) has done: 'I fix the TensorFlow import crash by forcing a protobuf version compatible with TF 2.18 (protobuf<6) via a pip install at runtime before importing TensorFlow; this is the minimal, standard Kaggle-side workaround for the `MessageFactory.GetPrototype` error. Then I fix TFRecord parsing for the test set by accepting both possible ID keys (`"id"` and `"image_id"`) so we don’t crash when the TFRecords use a different feature name than expected. Finally, I ensure we always produce a valid `submission.csv` with exactly the `image_id,label` columns in the same order as `sample_submission.csv`, keeping the rest of your EfficientNetB0 inference flow unchanged.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        pass



## === cell 1
import sys
import subprocess

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "protobuf<6"])

import tensorflow as tf
import matplotlib.pyplot as plt
from functools import partial
import re
import random

print("TensorFlow:", tf.__version__)
print("Protobuf:", __import__("google.protobuf").protobuf.__version__)

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)



## === cell 2
NUM_CLASSES = 5
IMAGE_SIZE = [512, 512]


def build_model(img_size=(512, 512), num_classes=5):
    inputs = tf.keras.Input(shape=(img_size[0], img_size[1], 3))
    base = tf.keras.applications.EfficientNetB0(
        include_top=False, weights="imagenet", input_tensor=inputs, pooling="avg"
    )
    x = tf.keras.layers.Dropout(0.2)(base.output)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    return model


model = build_model(tuple(IMAGE_SIZE), NUM_CLASSES)



## === cell 3
DATA_ROOT = "/kaggle/input/cassava-leaf-disease-classification"

test_df = pd.read_csv(f"{DATA_ROOT}/sample_submission.csv")
print(test_df.head())

AUTOTUNE = tf.data.AUTOTUNE
GCS_PATH = DATA_ROOT
BATCH_SIZE = 128  # preserves original intent (16*8)


def dataset_sizes(filenames):
    n = [
        int(re.compile(r"-([0-9]*)\.").search(filename).group(1))
        for filename in filenames
    ]
    return int(np.sum(n))


TEST_FILENAMES = tf.io.gfile.glob(GCS_PATH + "/test_tfrecords/ld_test*.tfrec")
if len(TEST_FILENAMES) == 0:
    raise FileNotFoundError(
        f"No TFRecords found under: {GCS_PATH}/test_tfrecords/ld_test*.tfrec"
    )

NUM_TEST_IMAGES = dataset_sizes(TEST_FILENAMES)
print("Found TFRecords:", len(TEST_FILENAMES), "NUM_TEST_IMAGES:", NUM_TEST_IMAGES)




## === cell 4
def to_float32(image, label):
    return tf.cast(image, tf.float32), label


def decode_img(img_bytes):
    img = tf.io.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, IMAGE_SIZE, method="bilinear")
    img = tf.cast(img, tf.float32) / 255.0
    return img


def read_tfrecord(example, labeled):
    if labeled:
        TFREC_FORMAT = {
            "image": tf.io.FixedLenFeature([], tf.string),
            "target": tf.io.FixedLenFeature([], tf.int64, default_value=-1),
            "label": tf.io.FixedLenFeature([], tf.int64, default_value=-1),
        }
    else:
        TFREC_FORMAT = {
            "image": tf.io.FixedLenFeature([], tf.string),
            "id": tf.io.FixedLenFeature([], tf.string, default_value=""),
            "image_id": tf.io.FixedLenFeature([], tf.string, default_value=""),
        }

    example = tf.io.parse_single_example(example, TFREC_FORMAT)
    img = decode_img(example["image"])

    if labeled:
        label = example["target"]
        label = tf.where(label >= 0, label, example["label"])
        label = tf.cast(label, tf.int32)
        return img, label
    else:
        image_id = tf.where(
            tf.strings.length(example["id"]) > 0, example["id"], example["image_id"]
        )
        return img, image_id


def load_dataset(filenames, labeled=True, ordered=False):
    options = tf.data.Options()
    if not ordered:
        options.experimental_deterministic = False  # speed
    ds = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTOTUNE)
    ds = ds.with_options(options)  # must be assigned
    ds = ds.map(partial(read_tfrecord, labeled=labeled), num_parallel_calls=AUTOTUNE)
    return ds


def get_test_data(ordered=False):
    ds = load_dataset(TEST_FILENAMES, labeled=False, ordered=ordered)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(AUTOTUNE)
    return ds




## === cell 5
test_ds = get_test_data(ordered=True)
test_ds = test_ds.map(to_float32, num_parallel_calls=AUTOTUNE)

print("Computing predictions...")
test_images_ds = test_ds.map(lambda image, image_id: image, num_parallel_calls=AUTOTUNE)

probabilities = model.predict(test_images_ds, verbose=1)
predictions = np.argmax(probabilities, axis=-1).astype(int)

if len(predictions) != NUM_TEST_IMAGES:
    raise RuntimeError(
        f"Prediction count mismatch: got {len(predictions)} vs expected {NUM_TEST_IMAGES}. "
        "Check TFRecord parsing/batching."
    )

print("Predictions shape:", predictions.shape, "Unique labels:", np.unique(predictions))



## === cell 6
print("Generating submission.csv file...")

test_ids_ds = test_ds.map(lambda image, image_id: image_id).unbatch()
test_ids = next(iter(test_ids_ds.batch(NUM_TEST_IMAGES))).numpy()

test_ids = np.array(
    [
        x.decode("utf-8") if isinstance(x, (bytes, bytearray, np.bytes_)) else str(x)
        for x in test_ids
    ],
    dtype=object,
)

assert len(test_ids) == len(predictions), (len(test_ids), len(predictions))

sub = pd.DataFrame({"image_id": test_ids, "label": predictions})

sub = test_df[["image_id"]].merge(sub, on="image_id", how="left")
sub["label"] = sub["label"].fillna(0).astype(int)

sub.to_csv("submission.csv", index=False)
print(sub.head())
print("Wrote submission.csv with rows:", len(sub))

with open("submission.csv", "r") as f:
    for _ in range(5):
        print(f.readline().strip())
