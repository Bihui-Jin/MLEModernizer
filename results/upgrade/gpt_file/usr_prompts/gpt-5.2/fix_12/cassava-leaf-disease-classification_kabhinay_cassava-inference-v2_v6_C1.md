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

2.7

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

0.8819885161680266

# 6. Current score

0.05531

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.61099) has done: 'I fix the immediate runtime/import failure caused by an incompatible protobuf implementation (the `MessageFactory.GetPrototype` error) by forcing TensorFlow to use the pure-Python protobuf backend before importing TensorFlow. Next, I make the SavedModel loading robust: those `/kaggle/input/...` model directories are not present in your provided dataset, so I detect missing model paths and fall back to a simple, deterministic baseline submission (majority class from `train.csv`) to ensure a valid `submission.csv` is always produced end-to-end. This keeps the core ensemble/prediction logic intact when the models exist, but prevents crashes when they don’t. The output always match the required submission format and `.csv` suffix.'
- What this solution (achieved 0.05531) has done: 'I fix the immediate TensorFlow import crash caused by the protobuf `MessageFactory.GetPrototype` mismatch by forcing the pure-Python protobuf runtime **and** ensuring TensorFlow is imported via the v1-compat path that works reliably in this environment. Next, I remove the dependency on missing external SavedModel directories (which currently forces a weak majority-class fallback and caps accuracy) by switching to using the competition-provided TFRecords for inference with a small Keras CNN trained on `train.csv` labels—this keeps the overall “train a classifier then predict test” semantics while making the pipeline self-contained. I keep the data paths unchanged, ensure deterministic behavior, and write a valid `submission.csv` with the exact required columns. This should substantially improve score from ~0.61 toward the target by using actual image content rather than a constant label.'
- What this solution (achieved 0.05531) has done: 'I fix the TensorFlow/protobuf import crash by forcing the pure-Python protobuf backend early and avoiding TF1 graph/session code paths that are incompatible with the current runtime. Next, I keep your exact dataset + model architecture intent (simple CNN trained on TFRecords) but run training/inference in standard TF2/Keras eager mode so there’s no “empty Session graph” failure. I also fix the optimizer compile error by switching from the TF1 `AdamOptimizer` object to the Keras Adam optimizer (same algorithm/learning rate semantics). Finally, I ensure inference preserves `sample_submission.csv` ordering and writes a valid `/kaggle/working/submission.csv`.'

# 9. Code solution

## === cell 0
from __future__ import print_function
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import json
import random
import numpy as np
import pandas as pd

import tensorflow as tf

try:
    tf.random.set_seed(42)
except Exception:
    try:
        tf.compat.v1.set_random_seed(42)
    except Exception:
        pass

np.random.seed(42)
random.seed(42)

DATA_DIR = "/kaggle/input/cassava-leaf-disease-classification"
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")
TRAIN_CSV_PATH = os.path.join(DATA_DIR, "train.csv")
TRAIN_TFREC_DIR = os.path.join(DATA_DIR, "train_tfrecords")
TEST_TFREC_DIR = os.path.join(DATA_DIR, "test_tfrecords")

print("TF version:", getattr(tf, "__version__", "unknown"))
print("Eager execution:", tf.executing_eagerly())
print("Train TFRecords dir exists:", os.path.isdir(TRAIN_TFREC_DIR))
print("Test TFRecords dir exists:", os.path.isdir(TEST_TFREC_DIR))



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
IMG_SIZE = 448  # keep consistent with the original script's input size
NUM_CLASSES = 5

_ID_KEYS_TO_TRY = ["image_id", "id"]


def _parse_id_feature(ex):
    for k in _ID_KEYS_TO_TRY:
        if k in ex:
            return ex[k]
    return tf.constant("", dtype=tf.string)


def parse_train_example(example_proto):
    features = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([], tf.int64),
        "id": tf.io.FixedLenFeature([], tf.string, default_value=""),
        "image_id": tf.io.FixedLenFeature([], tf.string, default_value=""),
    }
    ex = tf.io.parse_single_example(example_proto, features)
    img = tf.image.decode_jpeg(ex["image"], channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)  # [0,1]
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE], method="bilinear")
    label = tf.cast(ex["label"], tf.int32)
    label_oh = tf.one_hot(label, NUM_CLASSES, dtype=tf.float32)
    return img, label_oh


def parse_test_example(example_proto):
    features = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "id": tf.io.FixedLenFeature([], tf.string, default_value=""),
        "image_id": tf.io.FixedLenFeature([], tf.string, default_value=""),
    }
    ex = tf.io.parse_single_example(example_proto, features)
    img = tf.image.decode_jpeg(ex["image"], channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE], method="bilinear")
    image_id = _parse_id_feature(ex)
    return img, image_id


def list_tfrecs(directory):
    if not os.path.isdir(directory):
        return []
    files = []
    for fn in sorted(os.listdir(directory)):
        if (
            fn.endswith(".tfrec")
            or fn.endswith(".tfrecord")
            or fn.endswith(".tfrecords")
        ):
            files.append(os.path.join(directory, fn))
    return files


train_tfrecs = list_tfrecs(TRAIN_TFREC_DIR)
test_tfrecs = list_tfrecs(TEST_TFREC_DIR)

print("Num train tfrecs:", len(train_tfrecs))
print("Num test tfrecs:", len(test_tfrecs))
if len(train_tfrecs) == 0 or len(test_tfrecs) == 0:
    raise IOError(
        "Expected TFRecords under %s and %s" % (TRAIN_TFREC_DIR, TEST_TFREC_DIR)
    )



## === cell 2
BATCH_SIZE = 16  # preserve original inference batch size
EPOCHS = 3  # preserve original setting

try:
    AUTOTUNE = tf.data.AUTOTUNE
except Exception:
    AUTOTUNE = tf.data.experimental.AUTOTUNE


def make_train_dataset(tfrecs, shuffle_buffer=2048):
    ds = tf.data.TFRecordDataset(tfrecs, num_parallel_reads=AUTOTUNE)
    ds = ds.map(parse_train_example, num_parallel_calls=AUTOTUNE)
    ds = ds.shuffle(shuffle_buffer, seed=42, reshuffle_each_iteration=True)
    ds = ds.repeat()
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def make_valid_dataset(tfrecs):
    ds = tf.data.TFRecordDataset(tfrecs, num_parallel_reads=AUTOTUNE)
    ds = ds.map(parse_train_example, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def make_test_dataset(tfrecs):
    ds = tf.data.TFRecordDataset(tfrecs, num_parallel_reads=AUTOTUNE)
    ds = ds.map(parse_test_example, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


if len(train_tfrecs) >= 4:
    valid_tfrecs = train_tfrecs[-2:]
    trn_tfrecs = train_tfrecs[:-2]
else:
    valid_tfrecs = train_tfrecs[-1:]
    trn_tfrecs = train_tfrecs[:-1]

train_ds = make_train_dataset(trn_tfrecs)
valid_ds = make_valid_dataset(valid_tfrecs)
test_ds = make_test_dataset(test_tfrecs)

train_df = pd.read_csv(TRAIN_CSV_PATH)
num_train = int(train_df.shape[0])
num_valid = (
    int(np.ceil(num_train * (len(valid_tfrecs) / float(len(train_tfrecs)))))
    if len(train_tfrecs)
    else 0
)

steps_per_epoch = max(1, int(np.ceil((num_train - num_valid) / float(BATCH_SIZE))))
validation_steps = max(1, int(np.ceil(num_valid / float(BATCH_SIZE))))
print("num_train:", num_train, "num_valid(approx):", num_valid)
print("steps_per_epoch:", steps_per_epoch, "validation_steps:", validation_steps)



## === cell 3
keras = tf.keras
layers = tf.keras.layers

inp = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="image")
x = layers.Conv2D(32, 3, padding="same", activation="relu")(inp)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
out = layers.Dense(NUM_CLASSES, activation="softmax", name="probs")(x)
model = keras.Model(inp, out)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

print("Model built and compiled.")



## === cell 4
history = model.fit(
    train_ds,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_data=valid_ds,
    validation_steps=validation_steps,
    verbose=1,
)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
InvalidArgumentError                      Traceback (most recent call last)
/tmp/ipykernel_11/122227621.py in <cell line: 0>()
      1 # Fix: remove TF1 Session/graph usage which caused "Session graph is empty" under eager TF2.
      2 # Train with standard Keras fit() on tf.data datasets.
----> 3 history = model.fit(
      4     train_ds,
      5     epochs=EPOCHS,

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/execute.py in quick_execute(op_name, num_outputs, inputs, attrs, ctx, name)
     57       e.message += " name: " + name
     58     raise core._status_to_exception(e) from None
---> 59   except TypeError as e:
     60     keras_symbolic_tensors = [x for x in inputs if _is_keras_symbolic_tensor(x)]
     61     if keras_symbolic_tensors:

InvalidArgumentError: Graph execution error:

Detected at node ParseSingleExample/ParseExample/ParseExampleV2 defined at (most recent call last):
<stack traces unavailable>
Error in user-defined function passed to ParallelMapDatasetV2:2 transformation with iterator: Iterator::Root::Prefetch::BatchV2::ShuffleAndRepeat::ParallelMapV2: Feature: label (data type: int64) is required but could not be found.
	 [[{{node ParseSingleExample/ParseExample/ParseExampleV2}}]]
	 [[IteratorGetNext]] [Op:__inference_multi_step_on_iterator_1970]

## === cell 5
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
sample_image_ids = sample_sub["image_id"].astype(str).values.tolist()
sample_index = dict((img_id, i) for i, img_id in enumerate(sample_image_ids))

test_preds = np.zeros((len(sample_image_ids),), dtype=np.int64)

num_seen = 0
num_mapped = 0

for bx, bid in test_ds:
    bid_np = bid.numpy()
    ids = []
    for x in bid_np.tolist():
        if isinstance(x, (bytes, bytearray)):
            ids.append(x.decode("utf-8"))
        else:
            ids.append(str(x))

    probs = model.predict_on_batch(bx)
    cls = np.argmax(probs, axis=1).astype(np.int64)

    for img_id, c in zip(ids, cls):
        num_seen += 1
        if img_id in sample_index:
            test_preds[sample_index[img_id]] = int(c)
            num_mapped += 1

print("Test samples seen:", num_seen, "Expected:", len(sample_image_ids))
print("Mapped into sample_submission order:", num_mapped)

submission = pd.DataFrame({"image_id": sample_image_ids, "label": test_preds})
assert submission.shape[0] == sample_sub.shape[0]
assert list(submission.columns) == ["image_id", "label"]

out_path = "/kaggle/working/submission.csv"
submission.to_csv(out_path, index=False)
print("Wrote:", out_path)
print(submission.head())
