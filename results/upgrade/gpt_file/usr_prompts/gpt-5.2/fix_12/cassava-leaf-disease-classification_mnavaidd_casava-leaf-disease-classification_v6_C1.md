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

0.6976427923844062

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2

import tensorflow as tf
from tensorflow.keras import applications
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    BatchNormalization,
    GlobalAveragePooling2D,
)

SEED = 42
tf.keras.utils.set_random_seed(SEED)
np.random.seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

AUTOTUNE = tf.data.AUTOTUNE




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
data_path = "/kaggle/input/cassava-leaf-disease-classification/"
if not os.path.exists(data_path):
    data_path = "../input/cassava-leaf-disease-classification/"

train_csv_data_path = os.path.join(data_path, "train.csv")
label_json_data_path = os.path.join(data_path, "label_num_to_disease_map.json")
train_images_dir_data_path = os.path.join(data_path, "train_images")
test_images_dir_data_path = os.path.join(data_path, "test_images")
sample_sub_path = os.path.join(data_path, "sample_submission.csv")

train_tfrecords_dir_data_path = os.path.join(data_path, "train_tfrecords")
test_tfrecords_dir_data_path = os.path.join(data_path, "test_tfrecords")

assert os.path.exists(train_csv_data_path), f"Missing {train_csv_data_path}"
assert os.path.exists(
    train_images_dir_data_path
), f"Missing {train_images_dir_data_path}"
assert os.path.exists(test_images_dir_data_path), f"Missing {test_images_dir_data_path}"
assert os.path.exists(sample_sub_path), f"Missing {sample_sub_path}"

assert os.path.exists(
    train_tfrecords_dir_data_path
), f"Missing {train_tfrecords_dir_data_path}"
assert os.path.exists(
    test_tfrecords_dir_data_path
), f"Missing {test_tfrecords_dir_data_path}"




## === cell 2
train_csv = pd.read_csv(train_csv_data_path)
train_csv["label"] = train_csv["label"].astype(int)

label_class = pd.read_json(label_json_data_path, orient="index")
label_class = label_class.values.flatten().tolist()

train_csv["filepath"] = (
    train_images_dir_data_path + "/" + train_csv["image_id"].astype(str)
)

assert len(train_csv) > 0 and train_csv["filepath"].iloc[0].endswith(".jpg")

NUM_CLASSES = train_csv["label"].nunique()
assert NUM_CLASSES == 5, f"Expected 5 classes, got {NUM_CLASSES}"




## === cell 3
print("Label names :")
for i, label in enumerate(label_class):
    print(f" {i}. {label}")




## === cell 4
train_csv.head()




## === cell 5
BATCH_SIZE = 18
IMG_SIZE = 224
EPOCHS = 2  # keep identical

from sklearn.model_selection import train_test_split

train_df, val_df = train_test_split(
    train_csv,
    test_size=0.1,
    random_state=SEED,
    stratify=train_csv["label"],
)




## === cell 6
base_model = applications.ResNet152(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
)

base_model.trainable = False

inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dropout(0.25)(x)
outputs = Dense(NUM_CLASSES, activation="softmax")(x)
model_model = tf.keras.Model(inputs, outputs)

model_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
    jit_compile=True,
)




## === cell 7
model_model.summary()




## === cell 8
ss = pd.read_csv(sample_sub_path)
test_img_path = os.path.join(test_images_dir_data_path, ss["image_id"].iloc[0])
assert os.path.exists(test_img_path), f"Missing test image: {test_img_path}"




## === cell 9
data_augment = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal", seed=SEED),
        tf.keras.layers.RandomRotation(15 / 360.0, fill_mode="reflect", seed=SEED),
        tf.keras.layers.RandomTranslation(
            height_factor=0.05, width_factor=0.05, fill_mode="reflect", seed=SEED
        ),
        tf.keras.layers.RandomZoom(
            height_factor=(-0.1, 0.1),
            width_factor=(-0.1, 0.1),
            fill_mode="reflect",
            seed=SEED,
        ),
    ],
    name="data_augment",
)

FEATURE_DESCRIPTION_TRAIN = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),
    "image_id": tf.io.FixedLenFeature([], tf.string, default_value=b""),
}
FEATURE_DESCRIPTION_TEST = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_id": tf.io.FixedLenFeature([], tf.string, default_value=b""),
}


@tf.function
def _decode_and_resize_from_bytes(img_bytes):
    img = tf.io.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(
        img, [IMG_SIZE, IMG_SIZE], method=tf.image.ResizeMethod.BILINEAR
    )
    img = tf.cast(img, tf.float32) / 255.0
    return img


@tf.function
def _parse_train_example(example_proto):
    ex = tf.io.parse_single_example(example_proto, FEATURE_DESCRIPTION_TRAIN)
    img = _decode_and_resize_from_bytes(ex["image"])
    y = tf.cast(ex["target"], tf.int32)
    return img, y


@tf.function
def _parse_test_example(example_proto):
    ex = tf.io.parse_single_example(example_proto, FEATURE_DESCRIPTION_TEST)
    img = _decode_and_resize_from_bytes(ex["image"])
    img_id = ex["image_id"]
    return img, img_id


def _num_steps(n, batch_size):
    return (int(n) + int(batch_size) - 1) // int(batch_size)


train_count = len(train_df)
val_count = len(val_df)

train_steps = _num_steps(train_count, BATCH_SIZE)
val_steps = _num_steps(val_count, BATCH_SIZE)
test_steps = _num_steps(len(ss), BATCH_SIZE)

options = tf.data.Options()
options.experimental_deterministic = True
options.experimental_optimization.apply_default_optimizations = True
options.experimental_optimization.map_parallelization = True
options.experimental_optimization.parallel_batch = True

SHUFFLE_BUFFER = min(4096, train_count)

train_tfrec_files = sorted(
    [
        os.path.join(train_tfrecords_dir_data_path, f)
        for f in os.listdir(train_tfrecords_dir_data_path)
        if f.endswith(".tfrec")
    ]
)
test_tfrec_files = sorted(
    [
        os.path.join(test_tfrecords_dir_data_path, f)
        for f in os.listdir(test_tfrecords_dir_data_path)
        if f.endswith(".tfrec")
    ]
)
assert len(train_tfrec_files) > 0, "No training TFRecords found"
assert len(test_tfrec_files) > 0, "No test TFRecords found"

TFREC_COMPRESSION = "GZIP"


def _make_train_ds_from_tfrecords():
    ds_files = tf.data.Dataset.from_tensor_slices(train_tfrec_files).with_options(
        options
    )
    ds_files = ds_files.interleave(
        lambda f: tf.data.TFRecordDataset(
            f, compression_type=TFREC_COMPRESSION, num_parallel_reads=AUTOTUNE
        ),
        cycle_length=min(len(train_tfrec_files), 8),
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    )
    ds = ds_files.map(_parse_train_example, num_parallel_calls=AUTOTUNE)
    ds = ds.shuffle(
        buffer_size=SHUFFLE_BUFFER, seed=SEED, reshuffle_each_iteration=True
    )
    ds = ds.map(
        lambda x, y: (data_augment(x, training=True), y), num_parallel_calls=AUTOTUNE
    )
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def _make_val_ds_from_tfrecords():
    ds_files = tf.data.Dataset.from_tensor_slices(train_tfrec_files).with_options(
        options
    )
    ds_files = ds_files.interleave(
        lambda f: tf.data.TFRecordDataset(
            f, compression_type=TFREC_COMPRESSION, num_parallel_reads=AUTOTUNE
        ),
        cycle_length=min(len(train_tfrec_files), 8),
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    )
    ds = ds_files.map(_parse_train_example, num_parallel_calls=AUTOTUNE)
    ds = ds.take(val_count)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def _make_test_ds_from_tfrecords():
    ds_files = tf.data.Dataset.from_tensor_slices(test_tfrec_files).with_options(
        options
    )
    ds_files = ds_files.interleave(
        lambda f: tf.data.TFRecordDataset(
            f, compression_type=TFREC_COMPRESSION, num_parallel_reads=AUTOTUNE
        ),
        cycle_length=min(len(test_tfrec_files), 8),
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    )
    ds = ds_files.map(_parse_test_example, num_parallel_calls=AUTOTUNE)
    ds = ds.map(lambda x, img_id: x, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_ds = _make_train_ds_from_tfrecords()
val_ds = _make_val_ds_from_tfrecords()

history = model_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    verbose=1,
    steps_per_epoch=train_steps,
    validation_steps=val_steps,
)

test_ds = _make_test_ds_from_tfrecords()

probs = model_model.predict(test_ds, verbose=1, steps=test_steps)
preds = np.argmax(probs, axis=1).astype(int)

preds = preds[: len(ss)]
if len(preds) < len(ss):
    preds = np.pad(preds, (0, len(ss) - len(preds)), mode="constant", constant_values=0)

my_submission = pd.DataFrame({"image_id": ss["image_id"].values, "label": preds})
my_submission.to_csv("submission.csv", index=False)




## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
DataLossError                             Traceback (most recent call last)
/tmp/ipykernel_11/4226110362.py in <cell line: 0>()
    164 val_ds = _make_val_ds_from_tfrecords()
    165 
--> 166 history = model_model.fit(
    167     train_ds,
    168     validation_data=val_ds,

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

DataLossError: Graph execution error:

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

  File "/tmp/ipykernel_11/4226110362.py", line 166, in <cell line: 0>

  File "/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py", line 117, in error_handler

  File "/usr/local/lib/python3.11/dist-packages/keras/src/backend/tensorflow/trainer.py", line 371, in fit

  File "/usr/local/lib/python3.11/dist-packages/keras/src/backend/tensorflow/trainer.py", line 219, in function

  File "/usr/local/lib/python3.11/dist-packages/keras/src/backend/tensorflow/trainer.py", line 132, in multi_step_on_iterator

inflate() failed with error -3: incorrect header check
	 [[{{node IteratorGetNext}}]] [Op:__inference_multi_step_on_iterator_32734]

## === cell 10
print("Submission File: \n---------------\n")
print(my_submission.head())
print("\nSaved to: submission.csv")
print("Rows:", len(my_submission), "Columns:", list(my_submission.columns))
assert os.path.exists("submission.csv") and os.path.getsize("submission.csv") > 0
assert list(my_submission.columns) == ["image_id", "label"]
assert len(my_submission) == len(ss)

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1269576264.py in <cell line: 0>()
      1 print("Submission File: \n---------------\n")
----> 2 print(my_submission.head())
      3 print("\nSaved to: submission.csv")
      4 print("Rows:", len(my_submission), "Columns:", list(my_submission.columns))
      5 assert os.path.exists("submission.csv") and os.path.getsize("submission.csv") > 0

NameError: name 'my_submission' is not defined
