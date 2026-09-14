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

0.8927168328800241

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.10987) has done: 'The runtime fails early due to incompatible/unused imports (`kaggle_datasets`, custom EfficientNet/keras_applications) and missing external pretrained model files, which prevents `model_15` etc. from being defined. To make the notebook run end-to-end and still follow the same “TFRecord → EfficientNet → predict → submission” semantics, I replace the unavailable pretrained-model loading with a single built-in `tf.keras.applications.EfficientNetB0` model and keep the same inference/prediction flow. I also fix TFRecord decoding to avoid forcing a wrong fixed shape (which can break) by resizing after decode, and I correct `dataset.with_options(...)` (must be assigned) for determinism. Finally, I ensure the generated `submission.csv` matches `sample_submission.csv` ordering and has correct `image_id,label` columns.'
- What this solution (achieved 0.05531) has done: 'We need to fix the TensorFlow import crash (`AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'`), which is a protobuf/TensorFlow incompatibility that prevents the entire pipeline from running. The safest minimal fix in this Kaggle environment is to force the Python protobuf implementation before importing TensorFlow and restart the TF import cleanly, keeping the rest of your TFRecord → EfficientNet → predict → submission flow unchanged. I also add a tiny defensive check to ensure TFRecord filenames are found and keep submission generation identical (same columns/order as `sample_submission.csv`). These changes are score-neutral by themselves, but they unblock execution so you can produce a valid submission and then iterate toward the target score.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "cpp")

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        pass



## === cell 1
import tensorflow as tf
import matplotlib.pyplot as plt
from functools import partial
import re
import random

print("TensorFlow:", tf.__version__)

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_55/1054075065.py in <cell line: 0>()
----> 1 import tensorflow as tf
      2 import matplotlib.pyplot as plt
      3 from functools import partial
      4 import re
      5 import random

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



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/986044606.py in <cell line: 0>()
     14 
     15 
---> 16 model = build_model(tuple(IMAGE_SIZE), NUM_CLASSES)
     17 

/tmp/ipykernel_55/986044606.py in build_model(img_size, num_classes)
      4 
      5 def build_model(img_size=(512, 512), num_classes=5):
----> 6     inputs = tf.keras.Input(shape=(img_size[0], img_size[1], 3))
      7     base = tf.keras.applications.EfficientNetB0(
      8         include_top=False, weights="imagenet", input_tensor=inputs, pooling="avg"

NameError: name 'tf' is not defined

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




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/587849750.py in <cell line: 0>()
      4 print(test_df.head())
      5 
----> 6 AUTOTUNE = tf.data.AUTOTUNE
      7 GCS_PATH = DATA_ROOT
      8 BATCH_SIZE = 128  # preserves original intent (16*8)

NameError: name 'tf' is not defined

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
            "target": tf.io.FixedLenFeature([], tf.int64),
        }
    else:
        TFREC_FORMAT = {
            "image": tf.io.FixedLenFeature([], tf.string),
            "id": tf.io.FixedLenFeature([], tf.string),
        }

    example = tf.io.parse_single_example(example, TFREC_FORMAT)
    img = decode_img(example["image"])

    if labeled:
        label = tf.cast(example["target"], tf.int32)
        return img, label
    else:
        image_id = example["id"]
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
print("Predictions shape:", predictions.shape, "Unique labels:", np.unique(predictions))



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4034832626.py in <cell line: 0>()
----> 1 test_ds = get_test_data(ordered=True)
      2 test_ds = test_ds.map(to_float32, num_parallel_calls=AUTOTUNE)
      3 
      4 print("Computing predictions...")
      5 test_images_ds = test_ds.map(lambda image, image_id: image, num_parallel_calls=AUTOTUNE)

/tmp/ipykernel_55/3523675038.py in get_test_data(ordered)
     46 
     47 def get_test_data(ordered=False):
---> 48     ds = load_dataset(TEST_FILENAMES, labeled=False, ordered=ordered)
     49     ds = ds.batch(BATCH_SIZE)
     50     ds = ds.prefetch(AUTOTUNE)

NameError: name 'TEST_FILENAMES' is not defined

## === cell 6
print("Generating submission.csv file...")

test_ids_ds = test_ds.map(lambda image, image_id: image_id).unbatch()
test_ids = next(iter(test_ids_ds.batch(NUM_TEST_IMAGES))).numpy()

test_ids = np.array(
    [
        x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else str(x)
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

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/168045621.py in <cell line: 0>()
      1 print("Generating submission.csv file...")
      2 
----> 3 test_ids_ds = test_ds.map(lambda image, image_id: image_id).unbatch()
      4 test_ids = next(iter(test_ids_ds.batch(NUM_TEST_IMAGES))).numpy()
      5 

NameError: name 'test_ds' is not defined
