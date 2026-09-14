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

0.7765185856754306

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import re
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.setdefault("PYTHONHASHSEED", "42")

import tensorflow as tf

tf.keras.utils.set_random_seed(42)
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
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.preprocessing import LabelEncoder

label_to_disease = pd.read_json(
    "/kaggle/input/cassava-leaf-disease-classification/label_num_to_disease_map.json",
    typ="series",
)
train_csv = pd.read_csv("/kaggle/input/cassava-leaf-disease-classification/train.csv")

train_csv["disease"] = train_csv["label"].map(label_to_disease)
train_csv["path"] = (
    "/kaggle/input/cassava-leaf-disease-classification/train_images/"
    + train_csv["image_id"]
)

train_csv["label_encoded"] = LabelEncoder().fit_transform(train_csv["disease"])

train_csv["disease"] = train_csv["disease"].astype(str)
train_csv["label"] = train_csv["label"].astype(str)

train, valid = train_test_split(
    train_csv, test_size=0.2, stratify=train_csv["label"], random_state=42
)

NUM_CLASSES = int(train_csv["label_encoded"].nunique())



## === cell 2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomRotation(0.125, fill_mode="nearest", seed=42),
        tf.keras.layers.RandomTranslation(0.2, 0.2, fill_mode="nearest", seed=42),
        tf.keras.layers.RandomZoom(0.2, 0.2, fill_mode="nearest", seed=42),
        tf.keras.layers.RandomFlip("horizontal_and_vertical", seed=42),
    ],
    name="data_augmentation",
)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

TRAIN_TFRECORD_DIR = "/kaggle/input/cassava-leaf-disease-classification/train_tfrecords"
TEST_TFRECORD_DIR = "/kaggle/input/cassava-leaf-disease-classification/test_tfrecords"

train_tfrec_files = sorted(
    tf.io.gfile.glob(os.path.join(TRAIN_TFRECORD_DIR, "*.tfrec"))
)
test_tfrec_files = sorted(tf.io.gfile.glob(os.path.join(TEST_TFRECORD_DIR, "*.tfrec")))

if not train_tfrec_files:
    raise FileNotFoundError(f"No TFRecords found in {TRAIN_TFRECORD_DIR}")
if not test_tfrec_files:
    raise FileNotFoundError(f"No TFRecords found in {TEST_TFRECORD_DIR}")

_TFREC_FEATURES = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),
    "image_name": tf.io.FixedLenFeature([], tf.string),
}

TFREC_COMPRESSION = "GZIP"


@tf.function
def _parse_and_preprocess(example_proto):
    ex = tf.io.parse_single_example(example_proto, _TFREC_FEATURES)
    image = tf.image.decode_jpeg(ex["image"], channels=3)
    image = tf.image.resize(image, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    image = tf.cast(image, tf.float32)
    image = preprocess_input(image)
    label = tf.cast(ex["target"], tf.int32)
    image_id = ex["image_name"]
    return image, label, image_id


@tf.function
def _one_hot(image, label, image_id):
    return image, tf.one_hot(label, depth=NUM_CLASSES, dtype=tf.float32), image_id


@tf.function
def _augment(image, label, image_id):
    image = data_augmentation(image, training=True)
    return image, label, image_id


def _extract_shard_id(tfrec_path: str) -> int:
    m = re.search(r"ld_train(\d+)-", os.path.basename(tfrec_path))
    return int(m.group(1)) if m else -1


all_shard_ids = np.array(
    [_extract_shard_id(p) for p in train_tfrec_files], dtype=np.int32
)
if np.any(all_shard_ids < 0):
    raise ValueError(
        "Unexpected TFRecord naming; cannot extract shard ids for fast split."
    )

rng = np.random.RandomState(42)
perm = rng.permutation(len(train_tfrec_files))
n_valid = int(round(0.2 * len(train_tfrec_files)))
valid_idx = np.sort(perm[:n_valid])
train_idx = np.sort(perm[n_valid:])

train_tfrec_split = [train_tfrec_files[i] for i in train_idx]
valid_tfrec_split = [train_tfrec_files[i] for i in valid_idx]

train_opts = tf.data.Options()
train_opts.experimental_deterministic = (
    False  # safe for performance; model ops remain deterministic
)
valid_opts = tf.data.Options()
valid_opts.experimental_deterministic = False


def _build_base_ds(tfrec_files, opts):
    return (
        tf.data.TFRecordDataset(
            tfrec_files,
            num_parallel_reads=AUTOTUNE,
            compression_type=TFREC_COMPRESSION,
        )
        .with_options(opts)
        .map(_parse_and_preprocess, num_parallel_calls=AUTOTUNE)
    )


train_base = _build_base_ds(train_tfrec_split, train_opts)
valid_base = _build_base_ds(valid_tfrec_split, valid_opts)

shuffle_buf = int(min(len(train), 4096))

train_ds = (
    train_base.shuffle(shuffle_buf, seed=42, reshuffle_each_iteration=True)
    .map(_augment, num_parallel_calls=AUTOTUNE)
    .map(_one_hot, num_parallel_calls=AUTOTUNE)
    .map(lambda x, y, _id: (x, y), num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTOTUNE)
)

valid_ds = (
    valid_base.map(_one_hot, num_parallel_calls=AUTOTUNE)
    .map(lambda x, y, _id: (x, y), num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTOTUNE)
)


class EarlyStoppingCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if self.model.stop_training:
            print(f"Early stopping triggered at epoch {epoch + 1}.")


early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True,
)

learning_rate_reduction = ReduceLROnPlateau(
    monitor="val_loss",
    patience=2,
    factor=0.5,
    min_lr=1e-6,
    verbose=1,
)

base_model = EfficientNetB0(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3),
)
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
preds = Dense(NUM_CLASSES, activation="softmax")(x)
model = Model(inputs=base_model.input, outputs=preds)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
    steps_per_execution=10,
)

history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=10,
    callbacks=[early_stopping, learning_rate_reduction, EarlyStoppingCallback()],
    verbose=1,
)



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
DataLossError                             Traceback (most recent call last)
/tmp/ipykernel_12/2909931311.py in <cell line: 0>()
    169 )
    170 
--> 171 history = model.fit(
    172     train_ds,
    173     validation_data=valid_ds,

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

Detected at node IteratorGetNextAsOptional defined at (most recent call last):
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

  File "/tmp/ipykernel_12/2909931311.py", line 171, in <cell line: 0>

  File "/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py", line 117, in error_handler

  File "/usr/local/lib/python3.11/dist-packages/keras/src/backend/tensorflow/trainer.py", line 371, in fit

  File "/usr/local/lib/python3.11/dist-packages/keras/src/backend/tensorflow/trainer.py", line 219, in function

  File "/usr/local/lib/python3.11/dist-packages/keras/src/backend/tensorflow/trainer.py", line 199, in multi_step_on_iterator

inflate() failed with error -3: incorrect header check
	 [[{{node IteratorGetNextAsOptional}}]] [Op:__inference_multi_step_on_iterator_18552]

## === cell 3
import numpy as np
import pandas as pd
import os
from tensorflow.keras.applications.efficientnet import preprocess_input

sample_sub = pd.read_csv(
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
)

test_opts = tf.data.Options()
test_opts.experimental_deterministic = False


@tf.function
def _parse_and_preprocess_test(example_proto):
    ex = tf.io.parse_single_example(example_proto, _TFREC_FEATURES)
    image = tf.image.decode_jpeg(ex["image"], channels=3)
    image = tf.image.resize(image, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    image = tf.cast(image, tf.float32)
    image = preprocess_input(image)
    image_id = ex["image_name"]
    return image, image_id


test_ds = (
    tf.data.TFRecordDataset(
        test_tfrec_files,
        num_parallel_reads=AUTOTUNE,
        compression_type=TFREC_COMPRESSION,
    )
    .with_options(test_opts)
    .map(_parse_and_preprocess_test, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTOTUNE)
)

probs = model.predict(
    test_ds.map(lambda x, _id: x, num_parallel_calls=AUTOTUNE), verbose=0
)

ids_list = []
for _, b in test_ds:
    ids_list.append(b.numpy())
ids_bytes = np.concatenate(ids_list, axis=0)

test_ids = np.array([x.decode("utf-8") for x in ids_bytes.astype("S")], dtype=object)
pred_labels = np.argmax(probs, axis=1).astype(int)

pred_map = dict(zip(test_ids, pred_labels))
ordered_preds = sample_sub["image_id"].map(pred_map).astype(int).values

submission_df = pd.DataFrame(
    {"image_id": sample_sub["image_id"].values, "label": ordered_preds}
)
submission_path = "/kaggle/working/submission.csv"
submission_df.to_csv(submission_path, index=False)

print(f"Submission file created: {submission_path}")
print(submission_df.head())
print("Submission shape:", submission_df.shape)

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
DataLossError                             Traceback (most recent call last)
/tmp/ipykernel_12/4264493832.py in <cell line: 0>()
     39 
     40 # Predict on images, but keep IDs aligned by collecting them in the same single pass.
---> 41 probs = model.predict(
     42     test_ds.map(lambda x, _id: x, num_parallel_calls=AUTOTUNE), verbose=0
     43 )

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py in raise_from_not_ok_status(e, name)
   6000 def raise_from_not_ok_status(e, name) -> NoReturn:
   6001   e.message += (" name: " + str(name if name is not None else ""))
-> 6002   raise core._status_to_exception(e) from None  # pylint: disable=protected-access
   6003 
   6004 

DataLossError: {{function_node __wrapped__IteratorGetNext_output_types_1_device_/job:localhost/replica:0/task:0/device:CPU:0}} inflate() failed with error -3: incorrect header check [Op:IteratorGetNext] name:
