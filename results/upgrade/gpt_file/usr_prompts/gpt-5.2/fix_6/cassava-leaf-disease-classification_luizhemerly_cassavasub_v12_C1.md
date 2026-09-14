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
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
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

0.8679359323058325

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.61622) has done: 'I remove the protobuf “python” fallback that’s triggering the `MessageFactory.GetPrototype` crash under TF 2.18 + protobuf 6, so TensorFlow can import cleanly. Then I fix the test `tf.data` pipeline so it doesn’t return a string tensor alongside the image during `model.predict` (XLA can’t compile DT_STRING inputs), while still collecting `image_id`s for the submission. Finally, I ensure the submission is aligned to the sample submission order and always writes a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.61846) has done: 'I fix the TensorFlow import crash caused by the protobuf 6 API change by pinning protobuf to the pure-Python implementation via environment variables before importing TensorFlow. Then I make the TFRecord parsing more robust by supporting both possible label feature names (`target` and `label`) so training always uses the correct labels (this should materially improve accuracy toward your target without changing the model/training loop). Finally, I keep the existing “predict images only / collect ids separately” logic and ensure the submission is aligned to `sample_submission.csv` and written as `submission.csv`.'
- What this solution (achieved 0.61958) has done: 'I fix the TensorFlow import crash by removing the protobuf pure-Python fallback (it’s what triggers the `MessageFactory.GetPrototype` error under TF 2.18 + protobuf 6 in Kaggle). Then I keep your TFRecord parsing/training logic the same, but make the label feature selection robust without introducing invalid labels (so training isn’t silently corrupted). Finally, I ensure test IDs are decoded cleanly and the written `submission.csv` is aligned to `sample_submission.csv` with the exact required columns.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "cpp"
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

import re
import numpy as np
import pandas as pd
import tensorflow as tf

SEED = 42
tf.keras.utils.set_random_seed(SEED)
np.random.seed(SEED)

DATA_DIR = "/kaggle/input/cassava-leaf-disease-classification/"
AUTOTUNE = tf.data.AUTOTUNE

TRAIN_FILENAMES = sorted(tf.io.gfile.glob(DATA_DIR + "train_tfrecords/*.tfrec"))
TEST_FILENAMES = sorted(tf.io.gfile.glob(DATA_DIR + "test_tfrecords/*.tfrec"))

split_ind = int(0.9 * len(TRAIN_FILENAMES))
TRAINING_FILENAMES, VALID_FILENAMES = (
    TRAIN_FILENAMES[:split_ind],
    TRAIN_FILENAMES[split_ind:],
)

CLASS_NAMES = pd.read_json(DATA_DIR + "label_num_to_disease_map.json", typ="series")
NUM_CLASSES = len(CLASS_NAMES)

BATCH_SIZE = 16
IMG_HEIGHT = 400
IMG_WIDTH = 300
IMAGE_SIZE = (IMG_HEIGHT, IMG_WIDTH)

print("TF:", tf.__version__)
print(
    "Train tfrecs:",
    len(TRAIN_FILENAMES),
    "-> train/valid:",
    len(TRAINING_FILENAMES),
    len(VALID_FILENAMES),
)
print("Test tfrecs:", len(TEST_FILENAMES))
print("Num classes:", NUM_CLASSES)


def count_data_items(filenames):
    n = [int(re.compile(r"-([0-9]*)\.").search(fn).group(1)) for fn in filenames]
    return int(np.sum(n))


NUM_TRAIN_IMAGES = count_data_items(TRAIN_FILENAMES)
NUM_TEST_IMAGES = count_data_items(TEST_FILENAMES)
print("NUM_TRAIN_IMAGES:", NUM_TRAIN_IMAGES)
print("NUM_TEST_IMAGES:", NUM_TEST_IMAGES)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_55/423489.py in <cell line: 0>()
     11 import numpy as np
     12 import pandas as pd
---> 13 import tensorflow as tf
     14 
     15 SEED = 42

/usr/local/lib/python3.11/dist-packages/tensorflow/__init__.py in <module>
     47 _tf2.enable()
     48 
---> 49 from tensorflow._api.v2 import __internal__
     50 from tensorflow._api.v2 import __operators__
     51 from tensorflow._api.v2 import audio

/usr/local/lib/python3.11/dist-packages/tensorflow/_api/v2/__internal__/__init__.py in <module>
      6 import sys as _sys
      7 
----> 8 from tensorflow._api.v2.__internal__ import autograph
      9 from tensorflow._api.v2.__internal__ import decorator
     10 from tensorflow._api.v2.__internal__ import dispatch

/usr/local/lib/python3.11/dist-packages/tensorflow/_api/v2/__internal__/autograph/__init__.py in <module>
      6 import sys as _sys
      7 
----> 8 from tensorflow.python.autograph.core.ag_ctx import control_status_ctx # line: 34
      9 from tensorflow.python.autograph.impl.api import tf_convert # line: 493

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/core/ag_ctx.py in <module>
     19 import threading
     20 
---> 21 from tensorflow.python.autograph.utils import ag_logging
     22 from tensorflow.python.util.tf_export import tf_export
     23 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/utils/__init__.py in <module>
     15 """Utility module that contains APIs usable in the generated code."""
     16 
---> 17 from tensorflow.python.autograph.utils.context_managers import control_dependency_on_returns
     18 from tensorflow.python.autograph.utils.misc import alias_tensors
     19 from tensorflow.python.autograph.utils.tensor_list import dynamic_list_append

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/utils/context_managers.py in <module>
     17 import contextlib
     18 
---> 19 from tensorflow.python.framework import ops
     20 from tensorflow.python.ops import tensor_array_ops
     21 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py in <module>
     31 
     32 from google.protobuf import message
---> 33 from tensorflow.core.framework import attr_value_pb2
     34 from tensorflow.core.framework import full_type_pb2
     35 from tensorflow.core.framework import function_pb2

/usr/local/lib/python3.11/dist-packages/tensorflow/core/framework/attr_value_pb2.py in <module>
      3 # source: tensorflow/core/framework/attr_value.proto
      4 """Generated protocol buffer code."""
----> 5 from google.protobuf.internal import builder as _builder
      6 from google.protobuf import descriptor as _descriptor
      7 from google.protobuf import descriptor_pool as _descriptor_pool

/usr/local/lib/python3.11/dist-packages/google/protobuf/internal/builder.py in <module>
     16 
     17 from google.protobuf.internal import enum_type_wrapper
---> 18 from google.protobuf.internal import python_message
     19 from google.protobuf import message as _message
     20 from google.protobuf import reflection as _reflection

/usr/local/lib/python3.11/dist-packages/google/protobuf/internal/python_message.py in <module>
     36 import weakref
     37 
---> 38 from google.protobuf import descriptor as descriptor_mod
     39 from google.protobuf import message as message_mod
     40 from google.protobuf import text_format

/usr/local/lib/python3.11/dist-packages/google/protobuf/descriptor.py in <module>
     27   # TODO: Remove this import after fix api_implementation
     28   if _message is None:
---> 29     from google.protobuf.pyext import _message
     30   _USE_C_DESCRIPTORS = True
     31 

ImportError: cannot import name '_message' from 'google.protobuf.pyext' (/usr/local/lib/python3.11/dist-packages/google/protobuf/pyext/__init__.py)

## === cell 1
from functools import partial


def decode_image(image_bytes):
    image = tf.image.decode_jpeg(image_bytes, channels=3)
    image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH], method="bilinear")
    image = tf.cast(image, tf.float32) / 255.0
    return image


def _augment_image(image):
    image = tf.image.random_flip_left_right(image, seed=SEED)
    image = tf.image.random_brightness(image, max_delta=0.10)
    image = tf.image.random_contrast(image, lower=0.90, upper=1.10)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image


def read_tfrecord(example, labeled):
    if labeled:
        feature_description = {
            "image_name": tf.io.FixedLenFeature([], tf.string),
            "image": tf.io.FixedLenFeature([], tf.string),
            "target": tf.io.FixedLenFeature([], tf.int64, default_value=-1),
            "label": tf.io.FixedLenFeature([], tf.int64, default_value=-1),
        }
    else:
        feature_description = {
            "image_name": tf.io.FixedLenFeature([], tf.string),
            "image": tf.io.FixedLenFeature([], tf.string),
        }

    example = tf.io.parse_single_example(example, feature_description)
    image = decode_image(example["image"])

    if labeled:
        image = _augment_image(image)

        target = tf.cast(example["target"], tf.int32)
        label = tf.cast(example["label"], tf.int32)

        y = tf.where(label >= 0, label, target)
        y = tf.where(y >= 0, y, tf.zeros_like(y))

        y = tf.one_hot(y, depth=NUM_CLASSES, dtype=tf.float32)
        return image, y

    idnum = example["image_name"]
    return image, idnum


def load_dataset(filenames, labeled=True, ordered=False):
    opts = tf.data.Options()
    if not ordered:
        opts.experimental_deterministic = False
    ds = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTOTUNE).with_options(
        opts
    )
    ds = ds.map(partial(read_tfrecord, labeled=labeled), num_parallel_calls=AUTOTUNE)
    return ds


def get_training_dataset(filenames):
    ds = load_dataset(filenames, labeled=True, ordered=False)
    ds = ds.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def get_validation_dataset(filenames):
    ds = load_dataset(filenames, labeled=True, ordered=True)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def get_test_dataset_for_predict(filenames):
    ds = load_dataset(filenames, labeled=False, ordered=True)  # yields (image, id)
    ds = ds.map(lambda image, idnum: image, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def get_test_dataset_for_ids(filenames):
    ds = load_dataset(filenames, labeled=False, ordered=True)  # yields (image, id)
    ds = ds.map(lambda image, idnum: idnum, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_dataset = get_training_dataset(TRAINING_FILENAMES)
valid_dataset = get_validation_dataset(VALID_FILENAMES)
test_dataset_pred = get_test_dataset_for_predict(TEST_FILENAMES)
test_dataset_ids = get_test_dataset_for_ids(TEST_FILENAMES)




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/781886485.py in <cell line: 0>()
     94 
     95 
---> 96 train_dataset = get_training_dataset(TRAINING_FILENAMES)
     97 valid_dataset = get_validation_dataset(VALID_FILENAMES)
     98 test_dataset_pred = get_test_dataset_for_predict(TEST_FILENAMES)

NameError: name 'TRAINING_FILENAMES' is not defined

## === cell 2
def build_model():
    inputs = tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    x = tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    return model


model = build_model()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=[tf.keras.metrics.CategoricalAccuracy(name="acc")],
)
model.summary()



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/632332773.py in <cell line: 0>()
     13 
     14 
---> 15 model = build_model()
     16 model.compile(
     17     optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),

/tmp/ipykernel_55/632332773.py in build_model()
      1 def build_model():
----> 2     inputs = tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
      3     x = tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu")(inputs)
      4     x = tf.keras.layers.MaxPool2D()(x)
      5     x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)

NameError: name 'tf' is not defined

## === cell 3
EPOCHS = 3

history = model.fit(
    train_dataset,
    validation_data=valid_dataset,
    epochs=EPOCHS,
    verbose=2,
)



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/554224017.py in <cell line: 0>()
      1 EPOCHS = 3
      2 
----> 3 history = model.fit(
      4     train_dataset,
      5     validation_data=valid_dataset,

NameError: name 'model' is not defined

## === cell 4
probabilities = model.predict(test_dataset_pred, verbose=1)
predictions = np.argmax(probabilities, axis=1).astype(np.int64)
print("probabilities:", probabilities.shape, "predictions:", predictions.shape)
print("predictions[:10]:", predictions[:10])

test_ids = np.concatenate(
    [batch.numpy().astype("U") for batch in test_dataset_ids], axis=0
)
assert len(test_ids) == len(predictions), (len(test_ids), len(predictions))

submission = pd.DataFrame({"image_id": test_ids, "label": predictions})

sample_path = DATA_DIR + "sample_submission.csv"
if tf.io.gfile.exists(sample_path):
    sample = pd.read_csv(sample_path)
    submission = sample[["image_id"]].merge(submission, on="image_id", how="left")
    submission["label"] = submission["label"].fillna(0).astype(np.int64)

submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote:", "submission.csv", "rows:", len(submission))
print("label value counts:\n", submission["label"].value_counts().sort_index())

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2689524390.py in <cell line: 0>()
----> 1 probabilities = model.predict(test_dataset_pred, verbose=1)
      2 predictions = np.argmax(probabilities, axis=1).astype(np.int64)
      3 print("probabilities:", probabilities.shape, "predictions:", predictions.shape)
      4 print("predictions[:10]:", predictions[:10])
      5 

NameError: name 'model' is not defined
