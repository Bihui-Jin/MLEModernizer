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

0.5238742822605017

# 6. Current score

0.61099

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.61099) has done: 'Main bottlenecks are (1) decoding/resizing JPEGs from disk for every epoch using the image folder instead of the much faster TFRecords you already have, and (2) reading the test TFRecords twice (once to get names, once to get images). I switch training/validation input to TFRecords (same preprocessing/labels/one-hot, same model/training loop) and derive train/valid TFRecord splits deterministically from `train.csv` so semantics stay the same. I also make the test dataset emit `(image, image_name)` and run a single pass to collect both predictions and names, eliminating the extra TFRecord scan. These changes preserve the core model and loss/training logic while removing redundant and slower I/O, which is the typical cause of the 10-minute timeout here.'

# 9. Code solution

## === cell 0
import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)

import glob
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Activation,
    Dropout,
    Flatten,
    Dense,
)

from sklearn.model_selection import train_test_split

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

tf.config.experimental.enable_op_determinism()

BASE_PATH = "../input/cassava-leaf-disease-classification"
TRAIN_CSV = os.path.join(BASE_PATH, "train.csv")
TRAIN_IMG_DIR = os.path.join(BASE_PATH, "train_images")
TEST_IMG_GLOB = os.path.join(BASE_PATH, "test_images", "*.jpg")
SAMPLE_SUB = os.path.join(BASE_PATH, "sample_submission.csv")

TRAIN_TFREC_GLOB = os.path.join(BASE_PATH, "train_tfrecords", "*.tfrec")
TEST_TFREC_GLOB = os.path.join(BASE_PATH, "test_tfrecords", "*.tfrec")

assert os.path.exists(TRAIN_CSV), f"Missing {TRAIN_CSV}"
assert os.path.isdir(TRAIN_IMG_DIR), f"Missing {TRAIN_IMG_DIR}"
assert os.path.exists(SAMPLE_SUB), f"Missing {SAMPLE_SUB}"

print("TF version:", tf.__version__)
print("Keras version:", tf.keras.__version__)

train_tfrec_files = sorted(glob.glob(TRAIN_TFREC_GLOB))
test_tfrec_files = sorted(glob.glob(TEST_TFREC_GLOB))
print(
    "Found train tfrecords:",
    len(train_tfrec_files),
    "test tfrecords:",
    len(test_tfrec_files),
)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
df_train = pd.read_csv(TRAIN_CSV)

df_train["path"] = (TRAIN_IMG_DIR + "/" + df_train["image_id"]).astype(str)
df_train["label"] = df_train["label"].astype(str)

train_df, valid_df = train_test_split(
    df_train, test_size=0.15, random_state=SEED, stratify=df_train["label"]
)

print("Train size:", len(train_df), "Valid size:", len(valid_df))
print("Label distribution (train):", train_df["label"].value_counts().to_dict())



## === cell 2
IMG_SIZE = 224
SIZE = (IMG_SIZE, IMG_SIZE)
BATCH_SIZE = 32

AUTOTUNE = tf.data.AUTOTUNE

classes = sorted(df_train["label"].unique().tolist())
NUM_CLASSES = len(classes)
class_to_idx = {c: i for i, c in enumerate(classes)}
print("NUM_CLASSES:", NUM_CLASSES)
print("Class indices:", class_to_idx)


def _decode_resize_rescale(image_bytes):
    img = tf.io.decode_jpeg(image_bytes, channels=3)
    img = tf.image.resize(
        img, [IMG_SIZE, IMG_SIZE], method=tf.image.ResizeMethod.BILINEAR
    )
    img = tf.cast(img, tf.float32) * (1.0 / 255.0)
    return img


@tf.function
def _load_train(path, label_idx):
    image_bytes = tf.io.read_file(path)
    img = _decode_resize_rescale(image_bytes)
    y = tf.one_hot(label_idx, NUM_CLASSES, dtype=tf.float32)
    return img, y


@tf.function
def _load_test(path):
    image_bytes = tf.io.read_file(path)
    img = _decode_resize_rescale(image_bytes)
    return img


def make_train_valid_ds_from_jpegs(train_df, valid_df, batch_size):
    train_paths = train_df["path"].to_numpy()
    valid_paths = valid_df["path"].to_numpy()
    train_y = train_df["label"].map(class_to_idx).to_numpy(np.int32)
    valid_y = valid_df["label"].map(class_to_idx).to_numpy(np.int32)

    options = tf.data.Options()
    options.experimental_deterministic = (
        False  # correctness-preserving for supervised training
    )

    train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_y))
    train_ds = train_ds.with_options(options)
    train_ds = train_ds.shuffle(
        buffer_size=len(train_paths), seed=SEED, reshuffle_each_iteration=True
    )
    train_ds = train_ds.map(_load_train, num_parallel_calls=AUTOTUNE)

    train_ds = train_ds.batch(batch_size, drop_remainder=False)
    train_ds = train_ds.prefetch(AUTOTUNE)

    valid_ds = tf.data.Dataset.from_tensor_slices((valid_paths, valid_y))
    valid_ds = valid_ds.with_options(options)
    valid_ds = valid_ds.map(_load_train, num_parallel_calls=AUTOTUNE)

    valid_ds = valid_ds.batch(batch_size, drop_remainder=False)
    valid_ds = valid_ds.prefetch(AUTOTUNE)
    return train_ds, valid_ds


_FEATURES = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),
    "image_name": tf.io.FixedLenFeature([], tf.string),
}


@tf.function
def _parse_train_example(example_proto):
    ex = tf.io.parse_single_example(example_proto, _FEATURES)
    img = _decode_resize_rescale(ex["image"])
    y = tf.one_hot(tf.cast(ex["target"], tf.int32), NUM_CLASSES, dtype=tf.float32)
    return img, y


@tf.function
def _parse_test_example(example_proto):
    ex = tf.io.parse_single_example(example_proto, _FEATURES)
    img = _decode_resize_rescale(ex["image"])
    return img, ex["image_name"]


def make_train_valid_ds_from_tfrecords(train_files, valid_files, batch_size):
    options = tf.data.Options()
    options.experimental_deterministic = (
        False  # correctness-preserving for training throughput
    )

    train_ds = tf.data.TFRecordDataset(
        train_files, num_parallel_reads=AUTOTUNE
    ).with_options(options)
    train_ds = train_ds.map(_parse_train_example, num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.shuffle(
        buffer_size=8192, seed=SEED, reshuffle_each_iteration=True
    )

    train_ds = train_ds.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)

    valid_ds = tf.data.TFRecordDataset(
        valid_files, num_parallel_reads=AUTOTUNE
    ).with_options(options)
    valid_ds = valid_ds.map(_parse_train_example, num_parallel_calls=AUTOTUNE)

    valid_ds = valid_ds.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)
    return train_ds, valid_ds


def _tfrec_index_to_path(idx: int) -> str:
    return os.path.join(BASE_PATH, "train_tfrecords", f"ld_train{idx:02d}-1338.tfrec")


def _compute_shard_ids(image_ids: np.ndarray) -> np.ndarray:
    ids_int = np.fromiter(
        (int(s.split(".")[0]) for s in image_ids), dtype=np.int64, count=len(image_ids)
    )
    return (ids_int % 14).astype(np.int32)


def _tfrecord_files_for_split(split_df: pd.DataFrame) -> list:
    shard_ids = np.unique(_compute_shard_ids(split_df["image_id"].to_numpy()))
    return [_tfrec_index_to_path(int(i)) for i in shard_ids.tolist()]


if len(train_tfrec_files) > 0:
    train_files = _tfrecord_files_for_split(train_df)
    valid_files = _tfrecord_files_for_split(valid_df)
    print(
        "Using TFRecords for training. Train shards:",
        len(train_files),
        "Valid shards:",
        len(valid_files),
    )
    train_ds, valid_ds = make_train_valid_ds_from_tfrecords(
        train_files, valid_files, BATCH_SIZE
    )
else:
    print("TFRecords not found; falling back to JPEG loading (slower).")
    train_ds, valid_ds = make_train_valid_ds_from_jpegs(train_df, valid_df, BATCH_SIZE)


def make_test_ds_from_tfrecords(test_files, batch_size):
    options = tf.data.Options()
    options.experimental_deterministic = True  # keep fixed order for name alignment

    ds = tf.data.TFRecordDataset(test_files, num_parallel_reads=AUTOTUNE).with_options(
        options
    )
    ds = ds.map(_parse_test_example, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)
    return ds


if len(test_tfrec_files) > 0:
    test_ds = make_test_ds_from_tfrecords(test_tfrec_files, BATCH_SIZE)
    df_test = None
else:
    test_images = sorted(glob.glob(TEST_IMG_GLOB))
    df_test = pd.DataFrame(
        {
            "path": test_images,
            "image_id": [os.path.basename(p) for p in test_images],
        }
    )

    def make_test_ds(test_paths, batch_size):
        options = tf.data.Options()
        options.experimental_deterministic = True
        ds = tf.data.Dataset.from_tensor_slices(test_paths)
        ds = ds.with_options(options)
        ds = ds.map(_load_test, num_parallel_calls=AUTOTUNE)
        ds = ds.batch(batch_size, drop_remainder=False)
        ds = ds.prefetch(AUTOTUNE)
        return ds

    test_ds = make_test_ds(df_test["path"].to_numpy(), BATCH_SIZE)



## === cell 3
model = Sequential(
    [
        Conv2D(32, (3, 3), input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        Activation("relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3)),
        Activation("relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, (3, 3)),
        Activation("relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(256),
        Activation("relu"),
        Dropout(0.5),
        Dense(NUM_CLASSES),
        Activation("softmax"),
    ]
)

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()



## === cell 4
EPOCHS = 3

history = model.fit(train_ds, validation_data=valid_ds, epochs=EPOCHS, verbose=1)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NotFoundError                             Traceback (most recent call last)
/tmp/ipykernel_11/403313252.py in <cell line: 0>()
      1 EPOCHS = 3
      2 
----> 3 history = model.fit(train_ds, validation_data=valid_ds, epochs=EPOCHS, verbose=1)
      4 

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

NotFoundError: Graph execution error:

Detected at node IteratorGetNext defined at (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main

  File "<frozen runpy>", line 88, in _run_code

  File "/usr/local/lib/python3.11/dist-packages/colab_kernel_launcher.py", line 37, in <module>

  File "/usr/local/lib/python3.11/dist-packages/traitlets/config/application.py", line 992, in launch_instance

  File "/usr/local/lib/python3.11/dist-packages/ipykernel/kernelapp.py", line 712, in start

  File "/usr/local/lib/python3.11/dist-packages/tornado/platform/asyncio.py", line 211, in start

  File "/usr/lib/python3.11/asyncio/base_events.py", line 608, in run_forever

  File "/usr/lib/python3.11/asyncio/base_events.py", line 1936, in _run_once

  File "/usr/lib/python3.11/asyncio/events.py", line 84, in _run

  File "/usr/local/lib/python3.11/dist-packages/ipykernel/kernelbase.py", line 510, in dispatch_queue

  File "/usr/local/lib/python3.11/dist-packages/ipykernel/kernelbase.py", line 499, in process_one

  File "/usr/local/lib/python3.11/dist-packages/ipykernel/kernelbase.py", line 406, in dispatch_shell

  File "/usr/local/lib/python3.11/dist-packages/ipykernel/kernelbase.py", line 730, in execute_request

  File "/usr/local/lib/python3.11/dist-packages/ipykernel/ipkernel.py", line 383, in do_execute

  File "/usr/local/lib/python3.11/dist-packages/ipykernel/zmqshell.py", line 528, in run_cell

  File "/usr/local/lib/python3.11/dist-packages/IPython/core/interactiveshell.py", line 2975, in run_cell

  File "/usr/local/lib/python3.11/dist-packages/IPython/core/interactiveshell.py", line 3030, in _run_cell

  File "/usr/local/lib/python3.11/dist-packages/IPython/core/async_helpers.py", line 78, in _pseudo_sync_runner

  File "/usr/local/lib/python3.11/dist-packages/IPython/core/interactiveshell.py", line 3257, in run_cell_async

  File "/usr/local/lib/python3.11/dist-packages/IPython/core/interactiveshell.py", line 3473, in run_ast_nodes

  File "/usr/local/lib/python3.11/dist-packages/IPython/core/interactiveshell.py", line 3553, in run_code

  File "/tmp/ipykernel_11/403313252.py", line 3, in <cell line: 0>

  File "/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py", line 117, in error_handler

  File "/usr/local/lib/python3.11/dist-packages/keras/src/backend/tensorflow/trainer.py", line 371, in fit

  File "/usr/local/lib/python3.11/dist-packages/keras/src/backend/tensorflow/trainer.py", line 219, in function

  File "/usr/local/lib/python3.11/dist-packages/keras/src/backend/tensorflow/trainer.py", line 132, in multi_step_on_iterator

../input/cassava-leaf-disease-classification/train_tfrecords/ld_train08-1338.tfrec; No such file or directory
	 [[{{node IteratorGetNext}}]] [Op:__inference_multi_step_on_iterator_2265]

## === cell 5
if len(test_tfrec_files) > 0:
    test_names = []
    test_probs = []

    for batch_imgs, batch_names in test_ds:
        batch_pred = model.predict_on_batch(batch_imgs)
        test_probs.append(batch_pred)
        test_names.extend([n.decode("utf-8") for n in batch_names.numpy().tolist()])

    pred_test = np.concatenate(test_probs, axis=0)
    df_test = pd.DataFrame({"image_id": test_names})
else:
    pred_test = model.predict(test_ds, verbose=1)

pred_test_labels = np.argmax(pred_test, axis=-1).astype(int)

final_submission = pd.DataFrame(
    {
        "image_id": df_test["image_id"].values,
        "label": pred_test_labels,
    }
)

sample = pd.read_csv(SAMPLE_SUB)
assert len(final_submission) == len(
    sample
), f"Pred rows {len(final_submission)} != sample {len(sample)}"

final_submission = sample[["image_id"]].merge(
    final_submission, on="image_id", how="left"
)
missing = final_submission["label"].isna().sum()
assert missing == 0, f"Some test image_ids missing predictions: {missing}"

final_submission["label"] = final_submission["label"].astype(int)
final_submission.to_csv("submission.csv", index=False)
print(final_submission.head())
print("Wrote submission.csv with shape:", final_submission.shape)
