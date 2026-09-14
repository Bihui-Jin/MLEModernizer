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

0.8491991538229072

# 6. Current score

0.75747

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.10688) has done: 'I fix the immediate import crash caused by an incompatibility between `kaggle_datasets` and the installed `protobuf` by removing the unused `KaggleDatasets` import (the code uses local `../input/...` paths anyway). Then I fix the missing weights issue by replacing the external `.h5` dependency (not present in this environment) with a standard built-in Keras ImageNet backbone (same inference-only flow) so the notebook can run end-to-end and generate predictions. Finally, I ensure the TFRecord parsing, deterministic ordered test iteration, and submission writing are correct and aligned to the test IDs, producing a valid `submission.csv`.'
- What this solution (achieved 0.13453) has done: 'I fix the crash caused by the protobuf/TensorFlow descriptor incompatibility by setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` before importing TensorFlow. I also fix the broken cell numbering/order (your script starts at cell 0 and skips cell 1), so it runs cleanly in the provided “cells” runner. Finally, to move accuracy toward the target (your current 0.10688 is far below), I keep the same ResNet152 backbone but minimally add training on `train.csv` + `train_images` with a simple train/valid split; this preserves the overall architecture (ResNet backbone + Dense softmax head) while making predictions meaningful for this dataset, then it write a valid `submission.csv`.'
- What this solution (achieved 0.75747) has done: 'I remove expensive full-image resizing to 512×512 and instead decode/resize using TensorFlow’s built-in “decode+resize” path at the model’s native input size (still feeding exactly what the same ResNet backbone expects), which drastically cuts CPU/GPU time per step while preserving the same architecture and training loop. I also eliminate the extra second pass over the test dataset that currently iterates in Python to collect IDs, by making the prediction dataset yield `(image, id)` and collecting IDs in a single streamed pass in-graph. Finally, I tune the `tf.data` pipeline to avoid caching the entire TFRecord dataset in memory (which can cause stalls) and to use non-blocking prefetching and optimized dataset options; these are correctness-preserving pipeline changes only.'
- What this solution (achieved 0.75747) has done: 'I fix the immediate runtime crash by switching the protobuf implementation setting from `upb` (which triggers the `MessageFactory.GetPrototype` error in this environment) to the safe pure-Python implementation **before** importing TensorFlow. I also fix an image decoding bug: `decode_and_crop_jpeg` with a zero crop window is invalid; replacing it with `decode_jpeg` keeps the same “decode → resize → scale” semantics and avoids sporadic decoding failures. To nudge accuracy upward toward your target without changing the model/training approach, I align preprocessing with the selected backbone by using `tf.keras.applications.resnet.preprocess_input` (ResNet152 is in the `resnet` family) and keep the rest of the pipeline intact. Finally, I keep submission creation deterministic and ensure `image_id` strings are correctly decoded from bytes before writing `submission.csv`.'
- What this solution (achieved 0.75747) has done: 'We fix the TensorFlow import crash (`MessageFactory` / protobuf mismatch) by forcing the pure-Python protobuf implementation *and* disabling the C++ implementation via `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION=2`, and we do it before any TensorFlow import. Next, we fix a logic error in the test prediction pipeline: your `test_ds` already yields `(image, id)`, but you were mapping it with a lambda that expects two inputs—this can crash or silently mis-handle structure—so we remove that map and iterate the dataset directly. These changes are correctness/stability fixes and should preserve your existing model/training logic and (if anything) help the score by ensuring the intended pipeline actually runs end-to-end and writes a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import math, re
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from functools import partial
from sklearn.model_selection import train_test_split

print("Tensorflow version " + tf.__version__)

tf.keras.utils.set_random_seed(42)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    gpus = tf.config.list_physical_devices("GPU")
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

AUTOTUNE = tf.data.AUTOTUNE

try:
    tf.data.experimental.enable_debug_mode(False)
except Exception:
    pass

strategy = tf.distribute.get_strategy()

GCS_PATH = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_CSV = os.path.join(GCS_PATH, "train.csv")
TRAIN_IMG_DIR = os.path.join(GCS_PATH, "train_images")
TRAIN_TFREC_DIR = os.path.join(GCS_PATH, "train_tfrecords")
TEST_TFREC_DIR = os.path.join(GCS_PATH, "test_tfrecords")
SAMPLE_SUB = os.path.join(GCS_PATH, "sample_submission.csv")

IMAGE_SIZE = [224, 224]
CLASSES = ["0", "1", "2", "3", "4"]
BATCH_SIZE = 16
EPOCHS = 3  # keep as-is

print("Train CSV:", TRAIN_CSV)
print("Train image dir:", TRAIN_IMG_DIR)
print("Train tfrecords dir:", TRAIN_TFREC_DIR)
print("Test tfrecords dir:", TEST_TFREC_DIR)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
@tf.function
def decode_image(image_bytes):
    image = tf.image.decode_jpeg(image_bytes, channels=3)
    image = tf.image.resize(image, IMAGE_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    image = tf.cast(image, tf.float32) / 255.0
    return image




## === cell 2
TEST_FILENAMES = tf.io.gfile.glob(os.path.join(TEST_TFREC_DIR, "*.tfrec"))
print("Num test tfrecords:", len(TEST_FILENAMES))
print(TEST_FILENAMES[:3])




## === cell 3
@tf.function
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

    img_bytes = example["image"]
    image = decode_image(img_bytes)

    if labeled:
        label = tf.cast(example["target"], tf.int32)
        return image, label
    idnum = example["image_name"]
    return image, idnum


def load_dataset(filenames, labeled=True, ordered=False):
    opts = tf.data.Options()
    if not ordered:
        opts.experimental_deterministic = False
    opts.experimental_optimization.apply_default_optimizations = True
    opts.experimental_optimization.map_parallelization = True
    opts.experimental_optimization.parallel_batch = True

    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTOTUNE)
    dataset = dataset.with_options(opts)
    dataset = dataset.map(
        partial(read_tfrecord, labeled=labeled), num_parallel_calls=AUTOTUNE
    )
    return dataset


def get_test_dataset(ordered=False):
    dataset = load_dataset(TEST_FILENAMES, labeled=False, ordered=ordered)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=False)
    dataset = dataset.prefetch(AUTOTUNE)
    return dataset




## === cell 4
def count_data_items(filenames):
    n = [
        int(re.compile(r"-([0-9]*)\.").search(filename).group(1))
        for filename in filenames
    ]
    return np.sum(n)




## === cell 5
train_df = pd.read_csv(TRAIN_CSV)
train_df["filepath"] = (
    TRAIN_IMG_DIR.rstrip("/") + "/" + train_df["image_id"].astype(str)
)
train_df["label"] = train_df["label"].astype(np.int64)

trn_df, val_df = train_test_split(
    train_df, test_size=0.1, random_state=42, stratify=train_df["label"]
)
print("Train/Valid sizes:", len(trn_df), len(val_df))

TRAIN_FILENAMES = tf.io.gfile.glob(os.path.join(TRAIN_TFREC_DIR, "*.tfrec"))
print("Num train tfrecords:", len(TRAIN_FILENAMES))
print(TRAIN_FILENAMES[:3])
print("Train items (from filenames):", int(count_data_items(TRAIN_FILENAMES)))


def make_trainval_ds_from_tfrecords(filenames, training):
    ds = load_dataset(filenames, labeled=True, ordered=not training)

    if training:
        ds = ds.shuffle(8192, seed=42, reshuffle_each_iteration=True)

    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


TRAIN_FILENAMES_SORTED = sorted(TRAIN_FILENAMES)
rng = np.random.RandomState(42)
perm = rng.permutation(len(TRAIN_FILENAMES_SORTED))
TRAIN_FILENAMES_SHUFFLED = [TRAIN_FILENAMES_SORTED[i] for i in perm]

n_valid_files = max(1, int(round(0.1 * len(TRAIN_FILENAMES_SHUFFLED))))
VAL_FILENAMES = TRAIN_FILENAMES_SHUFFLED[:n_valid_files]
TRN_FILENAMES = TRAIN_FILENAMES_SHUFFLED[n_valid_files:]

print(
    "TFRecord file split -> train files:",
    len(TRN_FILENAMES),
    "valid files:",
    len(VAL_FILENAMES),
)

train_ds = make_trainval_ds_from_tfrecords(TRN_FILENAMES, training=True)
valid_ds = make_trainval_ds_from_tfrecords(VAL_FILENAMES, training=False)



## === cell 6
with strategy.scope():
    backbone = tf.keras.applications.ResNet152(
        include_top=False,
        weights="imagenet",
        input_shape=(*IMAGE_SIZE, 3),
        pooling="avg",
    )
    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3), dtype=tf.float32)

    x = tf.keras.applications.resnet.preprocess_input(inputs * 255.0)

    x = backbone(x, training=False)
    outputs = tf.keras.layers.Dense(len(CLASSES), activation="softmax")(x)
    trained_model = tf.keras.Model(inputs, outputs)

    trained_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

trained_model.summary()



## === cell 7
backbone.trainable = False
history1 = trained_model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=EPOCHS,
    verbose=1,
)

backbone.trainable = True
for layer in backbone.layers[:-30]:
    layer.trainable = False

trained_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

history2 = trained_model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=1,
    verbose=1,
)



## === cell 8
test_ds = get_test_dataset(ordered=True)

print("Computing predictions...")

pred_ds = test_ds.prefetch(AUTOTUNE)

all_probs = []
all_ids = []

for x_batch, id_batch in pred_ds:
    all_probs.append(trained_model(x_batch, training=False).numpy())
    all_ids.append(id_batch.numpy())

prob = np.concatenate(all_probs, axis=0)
predictions = np.argmax(prob, axis=-1).astype(np.int64)

test_ids_bytes = np.concatenate(all_ids, axis=0)
if test_ids_bytes.dtype.kind in ("S", "O"):
    test_ids = np.array([x.decode("utf-8") for x in test_ids_bytes], dtype="U")
else:
    test_ids = test_ids_bytes.astype("U")

print(predictions[:20], " ... total:", len(predictions))

print("Generating submission.csv file...")
assert len(test_ids) == len(
    predictions
), f"IDs ({len(test_ids)}) and predictions ({len(predictions)}) length mismatch"

sub = pd.DataFrame({"image_id": test_ids, "label": predictions.astype(np.int64)})

sample = pd.read_csv(SAMPLE_SUB)
sub = sample[["image_id"]].merge(sub, on="image_id", how="left")
if sub["label"].isna().any():
    fill_label = int(train_df["label"].mode().iloc[0])
    sub["label"] = sub["label"].fillna(fill_label).astype(np.int64)

sub.to_csv("submission.csv", index=False)

print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
print("Label value counts:\n", sub["label"].value_counts().sort_index())
