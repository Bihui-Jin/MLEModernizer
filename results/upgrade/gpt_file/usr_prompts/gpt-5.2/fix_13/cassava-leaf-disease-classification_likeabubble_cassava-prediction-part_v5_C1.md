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

3.13

# 3. Installed packages

numpy==1.26.4
protobuf==6.33.0
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

0.8399818676337262

# 6. Current score

0.61248

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.15919) has done: 'I fix the protobuf/TensorFlow import crash by forcing the pure-Python protobuf implementation early, which avoids the `MessageFactory.GetPrototype` attribute error in this environment. Since your external pre-trained model path doesn’t exist, I replace only that failing dependency with a small built-in Keras application model (keeping the same “load model → predict → argmax” pipeline) so the notebook runs end-to-end. I also ensure the TFRecord parsing matches the Cassava dataset keys (`image`, `image_name`), keep deterministic ordering, and write a valid `submission.csv` with the exact required header/columns. These changes are aimed at correctness and producing a valid submission; score is expected to be reasonable but not guaranteed without the original checkpoint.'
- What this solution (achieved 0.09268) has done: 'I fix the TensorFlow/protobuf crash by setting the protobuf environment variables before TensorFlow is imported (your current cell ordering imports TF too early). Then I correct the Xception construction error by using the required 299×299 input size when `include_top=True` with ImageNet weights, and resize TFRecord images to match so inference runs. Finally, I keep your existing “predict → map 1000→5 → argmax → write submission.csv” core logic intact, ensuring `image_id` alignment with `sample_submission.csv` and that a valid `submission.csv` is always produced.'
- What this solution (achieved 0.09268) has done: 'We fix the protobuf/TensorFlow import crash by forcing the pure-Python protobuf implementation at process start (via `sitecustomize.py`) and by importing TensorFlow only after verifying those env vars are actually set, with a safe fallback that restarts the interpreter if needed. Then we keep your exact inference core logic (Xception → predict 1000 → map to 5 → argmax) but correct a major submission-alignment bug: the TFRecord names are `bytes` and must be decoded and normalized to match `sample_submission.csv` keys, otherwise almost everything becomes NaN and gets filled with a single class (causing the very low score). Finally, we make the TFRecord item counting robust (no reliance on filename regex) and ensure we always write a valid `submission.csv` with correct columns and row count.'
- What this solution (achieved 0.61173) has done: 'You’re not getting a score because the script as provided won’t run on Kaggle: it starts at `cell 0` (but my output must start at cell 1), and more importantly it tries to `os.execv` the interpreter, which typically breaks notebook execution and prevents `submission.csv` from being written. I keep your exact model/inference/calibration core logic (Xception → predict 1000 → linear Dense to 5 → argmax) but remove the restart mechanism and instead ensure protobuf env vars are set before importing TensorFlow, which addresses the crash without killing the run. I also make the test-id extraction robust by taking names directly from the already-batched `test_ds` alongside images (avoids any off-by-one mismatch between `pred` and `test_names_ds.take(len(pred))`), which should improve correctness and therefore accuracy vs. the “filled with one class” failure mode. Finally, I still align to `sample_submission.csv` and always write a valid `submission.csv`.'
- What this solution (achieved 0.61248) has done: 'I fix the TensorFlow/protobuf crash by setting the protobuf environment variables before any TensorFlow-related import and by importing `google.protobuf` first to lock in the pure-Python implementation. I also remove the `sitecustomize.py` mechanism (it doesn’t help once the process has already started) and keep the rest of your model/inference/calibration pipeline unchanged. Finally, I make the TFRecord name decoding slightly more robust (handling missing/empty names) while preserving your existing deterministic ordering and submission alignment to `sample_submission.csv`. These changes are score-neutral in intent but should unblock execution and ensure the submission is produced correctly, which should improve score versus a crashed/invalid run.'
- What this solution (achieved 0.61248) has done: 'I fix the protobuf/TensorFlow crash by forcing the pure-Python protobuf backend *before* anything from `google.protobuf` is imported, using the supported `google.protobuf.internal.api_implementation` hook (instead of importing `google.protobuf` directly, which can trigger the broken C++ path). Then I keep your exact inference + calibration logic (Xception→1000-dim preds→single Dense(5) calibration→argmax) unchanged, only making the TensorFlow import ordering safe and deterministic. Finally, I keep the TFRecord parsing and submission alignment intact, ensuring `submission.csv` is always written with the correct columns and row count.'
- What this solution (achieved 0.61173) has done: 'We fix the TensorFlow/protobuf crash by forcing the pure-Python protobuf implementation *before* any `google.protobuf` import occurs, and by importing TensorFlow only after that lock-in; the current code still triggers the broken C++ path, causing the `MessageFactory.GetPrototype` error. Then we keep your exact model/inference/calibration pipeline (Xception→1000-dim→Dense(5) calibration→argmax) unchanged, but add a safe fallback to `/kaggle/data/...` if `/kaggle/input/...` isn’t mounted in this environment so the TFRecords are always found. Finally, we keep submission alignment logic intact and ensure `submission.csv` is written with the required columns and row count.'
- What this solution (achieved 0.61248) has done: 'I fix the protobuf/TensorFlow import crash by forcing the pure-Python protobuf implementation *before* any `google.protobuf`/TensorFlow import happens, using a safe early hook and only then importing TensorFlow. I keep your exact modeling/inference/calibration pipeline (Xception→1000-dim preds→Dense(5) calibration→argmax) unchanged, focusing only on stability and correct end-to-end execution. I also make the dataset path resolution slightly more robust (supporting both `/kaggle/input/...` and the provided `/kaggle/data/...` layout) without changing the data semantics. Finally, I keep the same submission alignment logic but ensure IDs are consistently decoded/normalized so the written `submission.csv` always matches `sample_submission.csv` row order.'

# 9. Code solution

## === cell 0
import os, sys

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

try:
    from google.protobuf.internal import api_implementation as _api_impl  # noqa: E402

    try:
        _api_impl._SetType("python")
    except Exception:
        pass
except Exception:
    pass

import re
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras

print("Python:", sys.version)
print("TF:", tf.__version__)
print("Keras:", keras.__version__)
print(
    "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION:",
    os.environ.get("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"),
)
print(
    "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION:",
    os.environ.get("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"),
)

tf.random.set_seed(42)
np.random.seed(42)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
NUM_CLASSES = 5
IMG_SIZE = 299

base = keras.applications.Xception(
    include_top=True,
    weights="imagenet",
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
)

inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

x = keras.applications.xception.preprocess_input(inputs * 255.0)

outputs = base(x, training=False)  # (None, 1000)
model = keras.Model(inputs, outputs)
model.trainable = False



## === cell 2
model.summary()



## === cell 3
CANDIDATE_DIRS = [
    "/kaggle/input/cassava-leaf-disease-classification",
    "/kaggle/data/cassava-leaf-disease-classification",
    "/kaggle/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
    "/kaggle/data/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
]
DATA_DIR = None
for d in CANDIDATE_DIRS:
    if tf.io.gfile.exists(d):
        DATA_DIR = d
        break
assert DATA_DIR is not None, "Could not find dataset directory in known locations."
print("Using DATA_DIR:", DATA_DIR)

train_filenames = tf.io.gfile.glob(f"{DATA_DIR}/train_tfrecords/*.tfrec")
train_filenames = sorted(train_filenames)
print("Train tfrecords:", len(train_filenames))
assert len(train_filenames) > 0, "No train TFRecord files found; check the path."

test_filenames = tf.io.gfile.glob(f"{DATA_DIR}/test_tfrecords/*.tfrec")
test_filenames = sorted(test_filenames)
print("Test tfrecords:", len(test_filenames))
assert len(test_filenames) > 0, "No test TFRecord files found; check the path."

sample_path = f"{DATA_DIR}/sample_submission.csv"
sample = pd.read_csv(sample_path)
print("Sample rows:", len(sample), "cols:", list(sample.columns))
assert list(sample.columns) == [
    "image_id",
    "label",
], "Unexpected sample submission format."




## === cell 4
def read_tfrec(example):
    fmt = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "image_name": tf.io.FixedLenFeature([], tf.string, default_value=""),
        "image_id": tf.io.FixedLenFeature([], tf.string, default_value=""),
        "target": tf.io.FixedLenFeature([], tf.int64, default_value=-1),
        "label": tf.io.FixedLenFeature([], tf.int64, default_value=-1),
    }
    ex = tf.io.parse_single_example(example, fmt)
    image = tf.image.decode_jpeg(ex["image"], channels=3)
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE), antialias=True)
    image = tf.cast(image, tf.float32) / 255.0

    name = ex["image_name"]
    name = tf.where(tf.equal(name, ""), ex["image_id"], name)

    y = tf.cast(ex["target"], tf.int32)
    y2 = tf.cast(ex["label"], tf.int32)
    y = tf.where(tf.equal(y, -1), y2, y)

    return image, name, y




## === cell 5
options = tf.data.Options()
options.experimental_deterministic = True

train_ds = tf.data.TFRecordDataset(train_filenames, num_parallel_reads=tf.data.AUTOTUNE)
train_ds = train_ds.with_options(options)
train_ds = train_ds.map(read_tfrec, num_parallel_calls=tf.data.AUTOTUNE)

train_ds_labeled = train_ds.filter(lambda img, name, y: y >= 0)

BATCH = 64
train_img_ds = (
    train_ds_labeled.map(lambda img, name, y: img, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH)
    .prefetch(tf.data.AUTOTUNE)
)
train_y_ds = (
    train_ds_labeled.map(lambda img, name, y: y, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH)
    .prefetch(tf.data.AUTOTUNE)
)

test_ds = tf.data.TFRecordDataset(test_filenames, num_parallel_reads=tf.data.AUTOTUNE)
test_ds = test_ds.with_options(options)
test_ds = test_ds.map(read_tfrec, num_parallel_calls=tf.data.AUTOTUNE)
test_ds = test_ds.batch(BATCH).prefetch(tf.data.AUTOTUNE)

test_image_ds = test_ds.map(
    lambda img, name, y: img, num_parallel_calls=tf.data.AUTOTUNE
)
test_name_batches_ds = test_ds.map(
    lambda img, name, y: name, num_parallel_calls=tf.data.AUTOTUNE
)



## === cell 6
train_prob_1000 = model.predict(train_img_ds, verbose=1)
train_y = np.concatenate([y.numpy() for y in train_y_ds], axis=0).astype(np.int64)

print(
    "Train prob_1000:",
    train_prob_1000.shape,
    "Train y:",
    train_y.shape,
    "y unique:",
    np.unique(train_y, return_counts=True),
)

calib = keras.Sequential(
    [
        keras.Input(shape=(train_prob_1000.shape[1],)),
        keras.layers.Dense(NUM_CLASSES, activation=None, use_bias=True),
    ]
)

calib.compile(
    optimizer=keras.optimizers.Adam(learning_rate=3e-3),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
)

calib.fit(
    train_prob_1000,
    train_y,
    batch_size=256,
    epochs=6,
    verbose=2,
    shuffle=True,
)



## === cell 7
prob_1000 = model.predict(test_image_ds, verbose=1)
logits_5 = calib.predict(prob_1000, verbose=1)
pred = np.argmax(logits_5, axis=-1).astype(np.int64)
print("Test logits_5 shape:", logits_5.shape)
print("Pred shape:", pred.shape, "Pred unique:", np.unique(pred, return_counts=True))




## === cell 8
def _normalize_image_id(x: str) -> str:
    x = (x or "").strip().replace("\\", "/")
    x = x.split("/")[-1]  # drop any directory prefix
    if not x:
        return x
    if x.isdigit():
        x = x + ".jpg"
    elif not x.lower().endswith(".jpg"):
        x = x + ".jpg"
    return x


test_ids = []
for batch_names in test_name_batches_ds.as_numpy_iterator():
    for v in batch_names:
        if isinstance(v, (bytes, bytearray, np.bytes_)):
            s = bytes(v).decode("utf-8", errors="ignore")
        else:
            s = str(v)
        test_ids.append(_normalize_image_id(s))

print("IDs read:", len(test_ids), "Preds:", len(pred))
assert len(test_ids) == len(
    pred
), "Mismatch between number of test ids and predictions."

id2pred = {test_ids[i]: int(pred[i]) for i in range(len(pred))}

vals, counts = np.unique(list(id2pred.values()), return_counts=True)
fill_value = int(vals[np.argmax(counts)])

sub = sample.copy()
sub["label"] = sub["image_id"].map(id2pred).fillna(fill_value).astype(np.int64)

assert len(sub) == len(sample), "Submission row count mismatch."
assert (
    sub["label"].between(0, NUM_CLASSES - 1).all()
), "Predicted labels out of range 0-4."

sub_path = "submission.csv"
sub.to_csv(sub_path, index=False)
print("Wrote:", sub_path, "rows:", len(sub))

with open(sub_path, "r", encoding="utf-8") as f:
    for _ in range(5):
        print(f.readline().rstrip("\n"))
