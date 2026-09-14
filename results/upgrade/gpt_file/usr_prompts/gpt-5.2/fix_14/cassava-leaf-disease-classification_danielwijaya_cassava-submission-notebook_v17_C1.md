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

0.8821396192203083

# 6. Current score

0.72048

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.72048) has done: 'The timeout is dominated by training EfficientNetB5 at 456×456 for 3 epochs plus input pipeline overhead; we can speed this up without changing the model or training loop by (a) enabling mixed precision on GPU (keeps architecture/loss/epochs identical and is standard for EfficientNet), (b) caching decoded/resized images to avoid repeating expensive JPEG decode+resize across epochs, and (c) using `model.predict`/vectorized dataset mapping for test inference to avoid Python-side loops and repeated `.numpy()` transfers. We also ensure TFRecord file order is stable and determinism remains enabled, and we add sensible `steps_per_epoch`/`validation_steps` computed from TFRecord counts to prevent any accidental extra iteration. These changes keep the same data, same labels, same model, same epochs, and same evaluation semantics, but reduce wall-clock time significantly.'
- What this solution (achieved 0.72048) has done: 'We fix the immediate runtime crash that happens on importing TensorFlow by pinning protobuf to the Python implementation at runtime, which avoids the known `MessageFactory.GetPrototype` incompatibility seen with some TF/protobuf builds. Then we keep the same model/training logic but make the TFRecord pipeline truly deterministic for validation/test and ensure IDs are decoded correctly to strings so the merge cannot silently misalign. Finally, we keep the submission-writing logic the same but add a small safety check to guarantee the output has exactly the sample submission rows and required column names, producing a valid `submission.csv`.'
- What this solution (achieved 0.72048) has done: 'I fix the immediate TensorFlow/protobuf crash by switching to a safe, Kaggle-compatible workaround (forcing the pure-Python protobuf runtime before importing TensorFlow, plus a fallback that removes the unsafe env var if TensorFlow still fails). Then I keep your exact model/training/inference logic, but make test ID decoding robust (handle `np.bytes_` and `tf.Tensor`/object cases) so the submission merge cannot silently produce NaNs or misalignment. Finally, I ensure the TFRecord size parsing can’t crash if a filename doesn’t match the regex, and that the script always writes a valid `submission.csv` with the required columns and row count. These are correctness/stability fixes and should not change the training semantics; any score change would come only from eliminating accidental ID misalignment.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import re
import sys
import random
import numpy as np
import pandas as pd

try:
    import tensorflow as tf
except Exception as e:
    print(
        "First TensorFlow import failed, retrying without PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION override."
    )
    os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
    import tensorflow as tf  # noqa: F401

import matplotlib.pyplot as plt

from functools import partial
from sklearn.model_selection import train_test_split

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

tf.config.optimizer.set_jit(True)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

if tf.config.list_physical_devices("GPU"):
    try:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        print("Mixed precision enabled: mixed_float16")
    except Exception as e:
        print("Could not enable mixed precision:", repr(e))

print("TensorFlow:", tf.__version__)
print("Num GPUs:", len(tf.config.list_physical_devices("GPU")))



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
AUTOTUNE = tf.data.AUTOTUNE

DATA_ROOT = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
SAMPLE_SUB = os.path.join(DATA_ROOT, "sample_submission.csv")

GCS_PATH = DATA_ROOT  # keep name used later

BATCH_SIZE = 32
IMAGE_SIZE = [456, 456]
NUM_CLASSES = 5

train_df = pd.read_csv(TRAIN_CSV)
print("train_df:", train_df.shape, train_df.columns.tolist())
test_df = pd.read_csv(SAMPLE_SUB)
print("sample_submission:", test_df.shape, test_df.columns.tolist())

TRAIN_FILENAMES = sorted(
    tf.io.gfile.glob(os.path.join(GCS_PATH, "train_tfrecords/ld_train*.tfrec"))
)
TEST_FILENAMES = sorted(
    tf.io.gfile.glob(os.path.join(GCS_PATH, "test_tfrecords/ld_test*.tfrec"))
)

_re_size = re.compile(r"-([0-9]+)\.tfrec$")


def dataset_sizes(filenames):
    total = 0
    for fn in filenames:
        m = _re_size.search(fn)
        if m:
            total += int(m.group(1))
    return int(total)


NUM_TRAIN_IMAGES = dataset_sizes(TRAIN_FILENAMES)
NUM_TEST_IMAGES = dataset_sizes(TEST_FILENAMES)

print(
    "TFRecords:",
    len(TRAIN_FILENAMES),
    "train files,",
    len(TEST_FILENAMES),
    "test files",
)
print("NUM_TRAIN_IMAGES:", NUM_TRAIN_IMAGES, "NUM_TEST_IMAGES:", NUM_TEST_IMAGES)




## === cell 2
def decode_img(img_bytes):
    img = tf.io.decode_jpeg(img_bytes, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)  # [0,1]
    img = tf.image.resize(
        img, IMAGE_SIZE, method=tf.image.ResizeMethod.BILINEAR, antialias=False
    )
    img = tf.ensure_shape(img, [IMAGE_SIZE[0], IMAGE_SIZE[1], 3])
    return img


def read_tfrecord(example, labeled):
    if labeled:
        tfrec_format = {
            "image": tf.io.FixedLenFeature([], tf.string),
            "target": tf.io.FixedLenFeature([], tf.int64),
        }
    else:
        tfrec_format = {
            "image": tf.io.FixedLenFeature([], tf.string),
            "image_name": tf.io.FixedLenFeature([], tf.string),
        }

    example = tf.io.parse_single_example(example, tfrec_format)
    img = decode_img(example["image"])

    if labeled:
        label = tf.cast(example["target"], tf.int32)
        return img, label
    else:
        image_id = example["image_name"]
        return img, image_id


def load_dataset(filenames, labeled=True, ordered=False):
    options = tf.data.Options()
    options.experimental_optimization.map_parallelization = True
    options.experimental_optimization.parallel_batch = True
    options.experimental_optimization.apply_default_optimizations = True
    options.threading.private_threadpool_size = 0  # let TF manage threads
    options.threading.max_intra_op_parallelism = 0
    options.deterministic = bool(ordered)

    ds = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTOTUNE)
    ds = ds.with_options(options)
    ds = ds.map(partial(read_tfrecord, labeled=labeled), num_parallel_calls=AUTOTUNE)
    return ds


def get_train_data(filenames, ordered=False):
    ds = load_dataset(filenames=filenames, labeled=True, ordered=ordered)
    ds = ds.cache()
    ds = ds.shuffle(8192, seed=SEED, reshuffle_each_iteration=True)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def get_valid_data(filenames, ordered=True):
    ds = load_dataset(filenames=filenames, labeled=True, ordered=ordered)
    ds = ds.cache()
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def get_test_data(ordered=True):
    ds = load_dataset(filenames=TEST_FILENAMES, labeled=False, ordered=ordered)
    ds = ds.cache()
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds




## === cell 3
train_files, valid_files = train_test_split(
    TRAIN_FILENAMES, test_size=0.15, random_state=SEED
)

num_train_images_split = dataset_sizes(train_files)
num_valid_images_split = dataset_sizes(valid_files)
steps_per_epoch = (
    int(np.ceil(num_train_images_split / BATCH_SIZE))
    if num_train_images_split
    else None
)
validation_steps = (
    int(np.ceil(num_valid_images_split / BATCH_SIZE))
    if num_valid_images_split
    else None
)
print("Split sizes:", num_train_images_split, "train,", num_valid_images_split, "valid")
print("Steps:", steps_per_epoch, "steps/epoch,", validation_steps, "validation_steps")

train_ds = get_train_data(train_files, ordered=False)
valid_ds = get_valid_data(valid_files, ordered=True)

base = tf.keras.applications.EfficientNetB5(
    include_top=False,
    weights="imagenet",
    input_shape=(None, None, 3),
    pooling="avg",
)

inputs = tf.keras.Input(shape=(None, None, 3))
x = tf.keras.applications.efficientnet.preprocess_input(inputs)
x = base(x, training=False)

outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax", dtype="float32")(x)
model_21 = tf.keras.Model(inputs, outputs)

model_21.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["sparse_categorical_accuracy"],
    run_eagerly=False,
)

model_21.summary()



## === cell 4
EPOCHS = 3

print("Training...")
history = model_21.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    verbose=2,
)

print("Computing predictions...")
test_ds = get_test_data(ordered=True)

test_imgs_ds = test_ds.map(lambda x, image_id: x, num_parallel_calls=AUTOTUNE)

test_ids_list = []
for batch_ids in test_ds.map(lambda x, image_id: image_id).as_numpy_iterator():
    test_ids_list.append(np.asarray(batch_ids))
test_ids_bytes = np.concatenate(test_ids_list, axis=0)


def _decode_id(v):
    if isinstance(v, np.ndarray) and v.shape == ():
        v = v.item()
    if isinstance(v, (bytes, bytearray, np.bytes_)):
        return v.decode("utf-8")
    return str(v)


test_ids = np.array([_decode_id(t) for t in test_ids_bytes], dtype="U")

probs = model_21.predict(test_imgs_ds, verbose=0)
predictions = np.argmax(probs, axis=-1).astype(np.int64)

print("Predictions shape:", predictions.shape, "unique labels:", np.unique(predictions))
print("Test IDs shape:", test_ids.shape)



## === cell 5
print("Generating submission.csv file...")

assert len(test_ids) == len(predictions), (len(test_ids), len(predictions))

sub_pred = pd.DataFrame({"image_id": test_ids, "label": predictions})
sub = test_df[["image_id"]].merge(sub_pred, on="image_id", how="left")

if sub["label"].isna().any():
    fallback = int(train_df["label"].mode().iloc[0])
    sub["label"] = sub["label"].fillna(fallback).astype(int)
else:
    sub["label"] = sub["label"].astype(int)

sub = sub[["image_id", "label"]]
assert sub.shape[0] == test_df.shape[0], (sub.shape, test_df.shape)
assert list(sub.columns) == ["image_id", "label"]

sub.to_csv("submission.csv", index=False)
print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)

with open("submission.csv", "r") as f:
    for _ in range(5):
        print(f.readline().strip())
