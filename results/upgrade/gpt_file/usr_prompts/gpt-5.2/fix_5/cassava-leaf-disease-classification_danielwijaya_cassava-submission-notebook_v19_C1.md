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

0.8918102145663342

# 6. Current score

0.77803

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.61099) has done: 'I remove the broken external `efficientnet/kerasapplications` imports that crash under the current protobuf version, and replace the missing `.h5` model loading (dataset not present) with an in-notebook EfficientNet backbone that keeps the same “predict probabilities then argmax” submission logic. I also fix the TFRecord pipeline bug where `dataset.with_options()` wasn’t applied, and make sure image IDs and predictions are aligned in the same deterministic order. Finally, I ensure a valid `submission.csv` (with the required header and columns) is written to the working directory end-to-end.'
- What this solution (achieved 0.77803) has done: 'We fix the immediate TensorFlow import crash caused by an incompatible protobuf build by forcing the pure-Python protobuf implementation before importing TensorFlow (this is a known workaround for the `MessageFactory.GetPrototype` error). Then we make a minimal, score-improving correction to the image preprocessing so it matches EfficientNet’s expected input normalization (using the built-in `preprocess_input` instead of simple `/255.0`), keeping the same model and training loop. We also ensure TFRecord decoding resizes images (not just reshapes) to the configured `IMAGE_SIZE`, so inference uses correctly scaled inputs. Finally, we keep deterministic test ordering and write a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.77803) has done: 'The crash happens before any training because TensorFlow 2.18 imports protobuf internals that no longer expose `MessageFactory.GetPrototype` under protobuf==6.x, so the current environment-variable workaround is insufficient. I add a tiny, safe monkey-patch before importing TensorFlow that aliases `GetPrototype` to `GetMessageClass` when needed, which is a known compatibility shim and does not change model logic. Then I keep the existing model/training/inference pipeline intact, only adding a small robustness fix to decode `image_name` bytes into proper UTF-8 strings so the submission `image_id` column is guaranteed correct. The output remains a valid `submission.csv` in the working directory with the required columns.'
- What this solution (achieved 0.77803) has done: 'The crash happens because the protobuf shim only patches `MessageFactory.GetPrototype` but TensorFlow is calling `GetPrototype` on a *MessageFactory instance* that still lacks that attribute under protobuf 6.x. I replace the shim with a safer, broader patch that aliases `GetPrototype` to `GetMessageClass` on both the `MessageFactory` class and the default factory instance, applied before importing TensorFlow. This is score-neutral (it only restores compatibility) and keeps your model/training/inference logic unchanged. I also add a small fallback for counting test IDs that avoids a hard dependency on TFRecord filename patterns, preventing runtime errors if the regex ever fails.'

# 9. Code solution

## === cell 0
import os
import re
import sys
import random
import numpy as np
import pandas as pd

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "3")

try:
    from google.protobuf import message_factory as _mf

    def _patch_message_factory_getprototype():
        MF = getattr(_mf, "MessageFactory", None)
        if (
            MF is not None
            and not hasattr(MF, "GetPrototype")
            and hasattr(MF, "GetMessageClass")
        ):

            def _GetPrototype(self, descriptor):
                return self.GetMessageClass(descriptor)

            setattr(MF, "GetPrototype", _GetPrototype)

        default_factory = getattr(_mf, "_DEFAULT_FACTORY", None)
        if (
            default_factory is not None
            and not hasattr(default_factory, "GetPrototype")
            and hasattr(default_factory, "GetMessageClass")
        ):

            def _inst_GetPrototype(descriptor, _df=default_factory):
                return _df.GetMessageClass(descriptor)

            setattr(default_factory, "GetPrototype", _inst_GetPrototype)

    _patch_message_factory_getprototype()
except Exception as e:
    print("Warning: protobuf shim not applied:", repr(e))

import tensorflow as tf
import matplotlib.pyplot as plt

from functools import partial
from sklearn.model_selection import train_test_split

SEED = 1337
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("TF version:", tf.__version__)
print("Num GPUs:", len(tf.config.list_physical_devices("GPU")))



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
DATA_DIR = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
SAMPLE_SUB = os.path.join(DATA_DIR, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images")
TEST_TFREC_DIR = os.path.join(DATA_DIR, "test_tfrecords")

assert os.path.exists(TRAIN_CSV), TRAIN_CSV
assert os.path.exists(SAMPLE_SUB), SAMPLE_SUB
assert os.path.exists(TRAIN_IMG_DIR), TRAIN_IMG_DIR
assert os.path.exists(TEST_TFREC_DIR), TEST_TFREC_DIR

train_df = pd.read_csv(TRAIN_CSV)
print(train_df.head())
print("Train rows:", len(train_df), "num classes:", train_df["label"].nunique())

NUM_CLASSES = 5
IMAGE_SIZE = [512, 512]
BATCH_SIZE = 64  # keep close to original effective batch size (16*8)
EPOCHS = 3  # keep runtime reasonable; no early stopping introduced

trn_df, val_df = train_test_split(
    train_df,
    test_size=0.1,
    random_state=SEED,
    stratify=train_df["label"],
)

preprocess_input = tf.keras.applications.efficientnet.preprocess_input


def _load_image_from_path(path):
    img = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMAGE_SIZE, method="bilinear")
    img = tf.cast(img, tf.float32)
    img = preprocess_input(img)  # expects float32 in [0,255]
    return img


def make_ds_from_df(df, training=True):
    paths = tf.constant([os.path.join(TRAIN_IMG_DIR, x) for x in df["image_id"].values])
    labels = tf.constant(df["label"].values, dtype=tf.int32)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(4096, seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(
        lambda p, y: (_load_image_from_path(p), y), num_parallel_calls=tf.data.AUTOTUNE
    )
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


train_ds = make_ds_from_df(trn_df, training=True)
val_ds = make_ds_from_df(val_df, training=False)

base = tf.keras.applications.EfficientNetB4(
    include_top=False,
    weights="imagenet",
    input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
)
base.trainable = False  # fine-tuning off to keep training stable and quick

inputs = tf.keras.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
x = inputs
x = base(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)
model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"],
)

model.summary()

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    verbose=1,
)



## === cell 2
test_df = pd.read_csv(SAMPLE_SUB)
print(test_df.head())
AUTOTUNE = tf.data.AUTOTUNE
GCS_PATH = DATA_DIR
CLASSES = ["1", "2", "3", "4", "5"]


def dataset_sizes(filenames):
    sizes = []
    pat = re.compile(r"-([0-9]+)\.tfrec$")
    for filename in filenames:
        m = pat.search(filename)
        if m:
            sizes.append(int(m.group(1)))
    if sizes:
        return int(np.sum(sizes))
    return int(len(test_df))


TEST_FILENAMES = tf.io.gfile.glob(GCS_PATH + "/test_tfrecords/ld_test*.tfrec")
TEST_FILENAMES = sorted(TEST_FILENAMES)  # deterministic order
NUM_TEST_IMAGES = dataset_sizes(TEST_FILENAMES)
print("Num test tfrecords:", len(TEST_FILENAMES), "Num test images:", NUM_TEST_IMAGES)




## === cell 3
def to_float32(image, label):
    return tf.cast(image, tf.float32), label


def decode_img(img_bytes):
    img = tf.io.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, IMAGE_SIZE, method="bilinear")
    img = tf.cast(img, tf.float32)
    img = preprocess_input(img)
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
        return (img, label)
    else:
        idNum = tf.ensure_shape(example["image_name"], [])
        idNum = tf.strings.regex_replace(
            idNum, rb"\x00+$", b""
        )  # strip trailing nulls if any
        return (img, idNum)


def load_dataset(filenames, labeled=True, ordered=False):
    options = tf.data.Options()
    options.experimental_deterministic = bool(ordered)

    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTOTUNE)
    dataset = dataset.with_options(options)
    dataset = dataset.map(
        partial(read_tfrecord, labeled=labeled), num_parallel_calls=AUTOTUNE
    )
    return dataset


def get_test_data(ordered=False):
    dataset = load_dataset(filenames=TEST_FILENAMES, labeled=False, ordered=ordered)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTOTUNE)
    return dataset




## === cell 4
test_ds = get_test_data(ordered=True)

print("Computing predictions...")
test_images_ds = test_ds.map(lambda image, idnum: image, num_parallel_calls=AUTOTUNE)

probabilities = model.predict(test_images_ds, verbose=1)
predictions = np.argmax(probabilities, axis=-1).astype(np.int64)

print("Predictions shape:", predictions.shape, "unique:", np.unique(predictions))



## === cell 5
print("Generating submission.csv file...")
test_ids_ds = test_ds.map(lambda image, idnum: idnum).unbatch()
test_ids_raw = next(iter(test_ids_ds.batch(NUM_TEST_IMAGES))).numpy()

test_ids = np.array(
    [
        x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else str(x)
        for x in test_ids_raw
    ],
    dtype=object,
)

assert len(test_ids) == len(predictions), (len(test_ids), len(predictions))

sub = pd.DataFrame({"image_id": test_ids, "label": predictions})
sub.to_csv("submission.csv", index=False)

print(sub.head())
print("Wrote submission.csv with rows:", len(sub))
with open("submission.csv", "r") as f:
    for _ in range(5):
        print(f.readline().rstrip())
