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
Predict whether a lesion is malignant (0 denotes **benign**, and 1 indicates **malignant**).

## Metric
Area under the ROC curve.

## Submission Format
For each `image_name` in the test set, you must predict the probability (`target`) that the sample is **malignant**. The file should contain a header and have the following format:

```
image_name,target
ISIC_0052060,0.7
ISIC_0052349,0.9
ISIC_0058510,0.8
ISIC_0073313,0.5
ISIC_0073502,0.5
etc.
```

## Dataset 
The images are provided in DICOM format.

Images are also provided in JPEG and TFRecord format (in the `jpeg` and `tfrecords` directories, respectively). Images in TFRecord format have been resized to a uniform 1024x1024.

Metadata is also provided outside of the DICOM format, in CSV files. See the `Columns` section for a description.

### Files
- **train.csv** - the training set
- **test.csv** - the test set
- **sample_submission.csv** - a sample submission file in the correct format

### Columns
- `image_name` - unique identifier, points to filename of related DICOM image
- `patient_id` - unique patient identifier
- `sex` - the sex of the patient (when unknown, will be blank)
- `age_approx` - approximate patient age at time of imaging
- `anatom_site_general_challenge` - location of imaged site
- `diagnosis` - detailed diagnosis information (train only)
- `benign_malignant` - indicator of malignancy of imaged lesion
- `target` - binarized version of the target variable

# 2. Python version

3.8

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (176 lines)
            jpeg.zip (24.7 GB)
            sample_submission.csv (4143 lines)
            sample_submission.csv.zip (16.4 kB)
            test.csv (4143 lines)
            test.csv.zip (42.5 kB)
            test.zip (6.4 GB)
            tfrecords.zip (9.3 GB)
            train.csv (28985 lines)
            train.csv.zip (299.7 kB)
            train.zip (46.0 GB)
            jpeg/
                test/
                    ISIC_1440063.jpg (1.1 MB)
                    ISIC_0815802.jpg (853.1 kB)
                    ... and 4140 other files
                train/
                    ISIC_1845271.jpg (1.0 MB)
                    ISIC_1970027.jpg (138.4 kB)
                    ... and 28982 other files
            siim-isic-melanoma-classification/
                description.md (176 lines)
                jpeg.zip (24.7 GB)
                ... and 9 other files
                jpeg/
                    test/
                        ISIC_1440063.jpg (1.1 MB)
                        ISIC_0815802.jpg (853.1 kB)
                        ... and 4140 other files
                    train/
                        ISIC_1845271.jpg (1.0 MB)
                        ISIC_1970027.jpg (138.4 kB)
                        ... and 28982 other files
                siim-isic-melanoma-classification/
                test/
                    ISIC_0052212.dcm (1.5 MB)
                    ISIC_0076545.dcm (4.0 MB)
                    ... and 4140 other files
                    test/
                tfrecords/
                    test00-2071.tfrec (579.6 MB)
                    test01-2071.tfrec (583.5 MB)
                    ... and 14 other files
                train/
                    ISIC_0015719.dcm (2.4 MB)
                    ISIC_0068279.dcm (1.3 MB)
                    ... and 28982 other files
                    train/
            test/
                ISIC_0052212.dcm (1.5 MB)
                ISIC_0076545.dcm (4.0 MB)
                ... and 4140 other files
                test/
            tfrecords/
                test00-2071.tfrec (579.6 MB)
                test01-2071.tfrec (583.5 MB)
                ... and 14 other files
            train/
                ISIC_0015719.dcm (2.4 MB)
                ISIC_0068279.dcm (1.3 MB)
                ... and 28982 other files
                train/
        input/
            description.md (176 lines)
            jpeg.zip (24.7 GB)
            sample_submission.csv (4143 lines)
            sample_submission.csv.zip (16.4 kB)
            test.csv (4143 lines)
            test.csv.zip (42.5 kB)
            test.zip (6.4 GB)
            tfrecords.zip (9.3 GB)
            train.csv (28985 lines)
            train.csv.zip (299.7 kB)
            train.zip (46.0 GB)
            jpeg/
                test/
                    ISIC_1440063.jpg (1.1 MB)
                    ISIC_0815802.jpg (853.1 kB)
                    ... and 4140 other files
                train/
                    ISIC_1845271.jpg (1.0 MB)
                    ISIC_1970027.jpg (138.4 kB)
                    ... and 28982 other files
            siim-isic-melanoma-classification/
                description.md (176 lines)
                jpeg.zip (24.7 GB)
                ... and 9 other files
                jpeg/
                    test/
                        ISIC_1440063.jpg (1.1 MB)
                        ISIC_0815802.jpg (853.1 kB)
                        ... and 4140 other files
                    train/
                        ISIC_1845271.jpg (1.0 MB)
                        ISIC_1970027.jpg (138.4 kB)
                        ... and 28982 other files
                siim-isic-melanoma-classification/
                test/
                    ISIC_0052212.dcm (1.5 MB)
                    ISIC_0076545.dcm (4.0 MB)
                    ... and 4140 other files
                    test/
                tfrecords/
                    test00-2071.tfrec (579.6 MB)
                    test01-2071.tfrec (583.5 MB)
                    ... and 14 other files
                train/
                    ISIC_0015719.dcm (2.4 MB)
                    ISIC_0068279.dcm (1.3 MB)
                    ... and 28982 other files
                    train/
            test/
                ISIC_0052212.dcm (1.5 MB)
                ISIC_0076545.dcm (4.0 MB)
                ... and 4140 other files
                test/
                    ISIC_0052212.dcm (1.5 MB)
                    ISIC_0076545.dcm (4.0 MB)
                    ... and 4140 other files
                    test/
            tfrecords/
                test00-2071.tfrec (579.6 MB)
                test01-2071.tfrec (583.5 MB)
                ... and 14 other files
            train/
                ISIC_0015719.dcm (2.4 MB)
                ISIC_0068279.dcm (1.3 MB)
                ... and 28982 other files
                train/
                    ISIC_0015719.dcm (2.4 MB)
                    ISIC_0068279.dcm (1.3 MB)
                    ... and 28982 other files
                    train/
        working/
            siim-isic-melanoma-classification/
                description.md (176 lines)
                jpeg.zip (24.7 GB)
                ... and 9 other files
                jpeg/
                    test/
                        ISIC_1440063.jpg (1.1 MB)
                        ISIC_0815802.jpg (853.1 kB)
                        ... and 4140 other files
                    train/
                        ISIC_1845271.jpg (1.0 MB)
                        ISIC_1970027.jpg (138.4 kB)
                        ... and 28982 other files
                siim-isic-melanoma-classification/
                test/
                    ISIC_0052212.dcm (1.5 MB)
                    ISIC_0076545.dcm (4.0 MB)
                    ... and 4140 other files
                    test/
                tfrecords/
                    test00-2071.tfrec (579.6 MB)
                    test01-2071.tfrec (583.5 MB)
                    ... and 14 other files
                train/
                    ISIC_0015719.dcm (2.4 MB)
                    ISIC_0068279.dcm (1.3 MB)
                    ... and 28982 other files
                    train/
```

-> data/sample_submission.csv has 4142 rows and 2 columns.
The columns are: image_name, target

-> data/siim-isic-melanoma-classification/sample_submission.csv has 4142 rows and 2 columns.
The columns are: image_name, target

-> data/siim-isic-melanoma-classification/test.csv has 4142 rows and 5 columns.
The columns are: image_name, patient_id, sex, age_approx, anatom_site_general_challenge

-> data/siim-isic-melanoma-classification/train.csv has 28984 rows and 8 columns.
The columns are: image_name, patient_id, sex, age_approx, anatom_site_general_challenge, diagnosis, benign_malignant, target

-> data/test.csv has 4142 rows and 5 columns.
The columns are: image_name, patient_id, sex, age_approx, anatom_site_general_challenge

-> data/train.csv has 28984 rows and 8 columns.
The columns are: image_name, patient_id, sex, age_approx, anatom_site_general_challenge, diagnosis, benign_malignant, target

-> input/sample_submission.csv has 4142 rows and 2 columns.
The columns are: image_name, target

-> (stopped after 10 files for performance)

# 5. Target score

0.8972131511597005

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "cpp")

import re
import math
import random

import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow.keras.layers as L
import tensorflow.keras.backend as K

from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

print("TensorFlow:", tf.__version__)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_11/1519948277.py in <cell line: 0>()
     13 import pandas as pd
     14 
---> 15 import tensorflow as tf
     16 import tensorflow.keras.layers as L
     17 import tensorflow.keras.backend as K

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
print("Listing /kaggle/input exists:", os.path.exists("/kaggle/input"))
print(
    "Listing competition dataset folder exists:",
    os.path.exists("/kaggle/input/siim-isic-melanoma-classification"),
)



## === cell 2
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print("Running on TPU ", tpu.master())
except Exception:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy()

print("REPLICAS: ", strategy.num_replicas_in_sync)



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1604652109.py in <cell line: 0>()
     10     strategy = tf.distribute.TPUStrategy(tpu)
     11 else:
---> 12     strategy = tf.distribute.get_strategy()
     13 
     14 print("REPLICAS: ", strategy.num_replicas_in_sync)

NameError: name 'tf' is not defined

## === cell 3
AUTO = tf.data.AUTOTUNE

BASE_PATH = "/kaggle/input/siim-isic-melanoma-classification"

EPOCHS = 10
BATCH_SIZE = 8 * strategy.num_replicas_in_sync
IMAGE_SIZE = [1024, 1024]

print("BASE_PATH:", BASE_PATH)
print("BATCH_SIZE:", BATCH_SIZE, "IMAGE_SIZE:", IMAGE_SIZE, "EPOCHS:", EPOCHS)



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2474941991.py in <cell line: 0>()
----> 1 AUTO = tf.data.AUTOTUNE
      2 
      3 BASE_PATH = "/kaggle/input/siim-isic-melanoma-classification"
      4 
      5 EPOCHS = 10

NameError: name 'tf' is not defined

## === cell 4
HEIGHT = IMAGE_SIZE[0]
WIDTH = IMAGE_SIZE[1]
CHANNELS = 3




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2541564553.py in <cell line: 0>()
----> 1 HEIGHT = IMAGE_SIZE[0]
      2 WIDTH = IMAGE_SIZE[1]
      3 CHANNELS = 3
      4 
      5 

NameError: name 'IMAGE_SIZE' is not defined

## === cell 5
def append_path(pre):
    return np.vectorize(lambda file: os.path.join(BASE_PATH, pre, file))




## === cell 6
sub = pd.read_csv(f"{BASE_PATH}/sample_submission.csv")
sub.head()



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2547788584.py in <cell line: 0>()
----> 1 sub = pd.read_csv(f"{BASE_PATH}/sample_submission.csv")
      2 sub.head()
      3 

NameError: name 'BASE_PATH' is not defined

## === cell 7
train = pd.read_csv(f"{BASE_PATH}/train.csv")
test = pd.read_csv(f"{BASE_PATH}/test.csv")
print(train.shape, test.shape)



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4253402684.py in <cell line: 0>()
----> 1 train = pd.read_csv(f"{BASE_PATH}/train.csv")
      2 test = pd.read_csv(f"{BASE_PATH}/test.csv")
      3 print(train.shape, test.shape)
      4 

NameError: name 'BASE_PATH' is not defined

## === cell 8
pass



## === cell 9
TRAINING_FILENAMES_ALL = sorted(tf.io.gfile.glob(f"{BASE_PATH}/tfrecords/train*.tfrec"))
TEST_FILENAMES = sorted(tf.io.gfile.glob(f"{BASE_PATH}/tfrecords/test*.tfrec"))

if len(TRAINING_FILENAMES_ALL) == 0 or len(TEST_FILENAMES) == 0:
    raise FileNotFoundError(
        "TFRecord files not found. Expected under "
        f"{BASE_PATH}/tfrecords. Found train={len(TRAINING_FILENAMES_ALL)}, test={len(TEST_FILENAMES)}"
    )

VALIDATION_FRACTION = 0.1
n_val = max(1, int(len(TRAINING_FILENAMES_ALL) * VALIDATION_FRACTION))
VALIDATION_FILENAMES = TRAINING_FILENAMES_ALL[:n_val]
TRAINING_FILENAMES = TRAINING_FILENAMES_ALL[n_val:]

print(
    "TFRecords:",
    len(TRAINING_FILENAMES_ALL),
    "train files,",
    len(TRAINING_FILENAMES),
    "used for training,",
    len(VALIDATION_FILENAMES),
    "used for validation",
)
print("Example train tfrec:", TRAINING_FILENAMES[0])
print("Example test tfrec:", TEST_FILENAMES[0])

CLASSES = [0, 1]



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3951788159.py in <cell line: 0>()
----> 1 TRAINING_FILENAMES_ALL = sorted(tf.io.gfile.glob(f"{BASE_PATH}/tfrecords/train*.tfrec"))
      2 TEST_FILENAMES = sorted(tf.io.gfile.glob(f"{BASE_PATH}/tfrecords/test*.tfrec"))
      3 
      4 if len(TRAINING_FILENAMES_ALL) == 0 or len(TEST_FILENAMES) == 0:
      5     raise FileNotFoundError(

NameError: name 'tf' is not defined

## === cell 10
pass



## === cell 11
LABELED_TFREC_FORMAT = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),
}
UNLABELED_TFREC_FORMAT = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
}


@tf.function
def decode_image(image_data):
    image = tf.image.decode_jpeg(image_data, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.reshape(image, [*IMAGE_SIZE, 3])
    return image


@tf.function
def read_labeled_tfrecord(example):
    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)
    image = decode_image(example["image"])
    label = tf.cast(example["target"], tf.int32)
    return image, label


@tf.function
def read_unlabeled_tfrecord(example):
    example = tf.io.parse_single_example(example, UNLABELED_TFREC_FORMAT)
    image = decode_image(example["image"])
    idnum = example["image_name"]
    return image, idnum


def load_dataset(filenames, labeled=True, ordered=False):
    opts = tf.data.Options()
    try:
        opts.experimental_optimization.apply_default_optimizations = True
        opts.experimental_optimization.map_parallelization = True
    except Exception:
        pass
    try:
        opts.experimental_deterministic = bool(ordered)
    except Exception:
        pass

    ds = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)
    ds = ds.with_options(opts)
    ds = ds.map(
        read_labeled_tfrecord if labeled else read_unlabeled_tfrecord,
        num_parallel_calls=AUTO,
        deterministic=ordered,
    )
    return ds


@tf.function
def data_augment(image, label):
    image = tf.image.random_flip_left_right(image)
    return image, label


def get_training_dataset():
    ds = load_dataset(TRAINING_FILENAMES, labeled=True, ordered=False)
    ds = ds.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
    ds = ds.repeat()
    ds = ds.map(data_augment, num_parallel_calls=AUTO)
    ds = ds.batch(BATCH_SIZE, drop_remainder=True)
    ds = ds.prefetch(AUTO)
    return ds


def get_validation_dataset(ordered=False):
    ds = load_dataset(VALIDATION_FILENAMES, labeled=True, ordered=ordered)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(AUTO)
    return ds


def get_test_dataset(ordered=False):
    ds = load_dataset(TEST_FILENAMES, labeled=False, ordered=ordered)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(AUTO)
    return ds


def count_data_items(filenames):
    n = [
        int(re.compile(r"-([0-9]*)\.").search(filename).group(1))
        for filename in filenames
    ]
    return np.sum(n)


NUM_TRAINING_IMAGES = int(count_data_items(TRAINING_FILENAMES))
NUM_VALIDATION_IMAGES = int(count_data_items(VALIDATION_FILENAMES))
NUM_TEST_IMAGES = int(count_data_items(TEST_FILENAMES))

STEPS_PER_EPOCH = max(1, NUM_TRAINING_IMAGES // BATCH_SIZE)
VALIDATION_STEPS = max(1, math.ceil(NUM_VALIDATION_IMAGES / BATCH_SIZE))
TEST_STEPS = max(1, math.ceil(NUM_TEST_IMAGES / BATCH_SIZE))

print(
    f"Dataset: {NUM_TRAINING_IMAGES} train, {NUM_VALIDATION_IMAGES} val, {NUM_TEST_IMAGES} test"
)
print(
    "STEPS_PER_EPOCH:",
    STEPS_PER_EPOCH,
    "VALIDATION_STEPS:",
    VALIDATION_STEPS,
    "TEST_STEPS:",
    TEST_STEPS,
)




## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1367920186.py in <cell line: 0>()
      4 # - adding explicit parallel calls and prefetch; keeping the exact same augmentation op
      5 LABELED_TFREC_FORMAT = {
----> 6     "image": tf.io.FixedLenFeature([], tf.string),
      7     "target": tf.io.FixedLenFeature([], tf.int64),
      8 }

NameError: name 'tf' is not defined

## === cell 12
def build_lrfn(
    lr_start=0.00001,
    lr_max=0.0001,
    lr_min=0.000001,
    lr_rampup_epochs=20,
    lr_sustain_epochs=0,
    lr_exp_decay=0.8,
):
    lr_max = lr_max * strategy.num_replicas_in_sync

    def lrfn(epoch):
        if epoch < lr_rampup_epochs:
            lr = (lr_max - lr_start) / lr_rampup_epochs * epoch + lr_start
        elif epoch < lr_rampup_epochs + lr_sustain_epochs:
            lr = lr_max
        else:
            lr = (lr_max - lr_min) * lr_exp_decay ** (
                epoch - lr_rampup_epochs - lr_sustain_epochs
            ) + lr_min
        return lr

    return lrfn




## === cell 13
with strategy.scope():
    model = tf.keras.Sequential(
        [
            tf.keras.applications.EfficientNetB6(
                input_shape=(*IMAGE_SIZE, 3), weights="imagenet", include_top=False
            ),
            L.GlobalAveragePooling2D(),
            L.Dense(512, activation="relu"),
            L.Dense(128, activation="relu"),
            L.Dense(1, activation="sigmoid"),
        ]
    )

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    jit_compile=True,
)
model.summary()



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3608558494.py in <cell line: 0>()
----> 1 with strategy.scope():
      2     model = tf.keras.Sequential(
      3         [
      4             tf.keras.applications.EfficientNetB6(
      5                 input_shape=(*IMAGE_SIZE, 3), weights="imagenet", include_top=False

NameError: name 'strategy' is not defined

## === cell 14
with strategy.scope():
    model2 = tf.keras.Sequential(
        [
            tf.keras.applications.EfficientNetB3(
                input_shape=(*IMAGE_SIZE, 3), weights="imagenet", include_top=False
            ),
            L.GlobalAveragePooling2D(),
            L.Dense(512, activation="relu"),
            L.Dropout(0.3),
            L.Dense(256, activation="relu"),
            L.Dropout(0.25),
            L.Dense(128, activation="relu"),
            L.Dropout(0.2),
            L.Dense(1, activation="sigmoid"),
        ]
    )

model2.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    jit_compile=True,
)
model2.summary()



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2144621244.py in <cell line: 0>()
----> 1 with strategy.scope():
      2     model2 = tf.keras.Sequential(
      3         [
      4             tf.keras.applications.EfficientNetB3(
      5                 input_shape=(*IMAGE_SIZE, 3), weights="imagenet", include_top=False

NameError: name 'strategy' is not defined

## === cell 15
lrfn = build_lrfn()
lr_schedule = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=0)

STEPS_PER_EPOCH = max(1, NUM_TRAINING_IMAGES // BATCH_SIZE)

train_ds = get_training_dataset()
val_ds = get_validation_dataset(ordered=True)



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3144116089.py in <cell line: 0>()
----> 1 lrfn = build_lrfn()
      2 lr_schedule = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=0)
      3 
      4 STEPS_PER_EPOCH = max(1, NUM_TRAINING_IMAGES // BATCH_SIZE)
      5 

/tmp/ipykernel_11/886858925.py in build_lrfn(lr_start, lr_max, lr_min, lr_rampup_epochs, lr_sustain_epochs, lr_exp_decay)
      7     lr_exp_decay=0.8,
      8 ):
----> 9     lr_max = lr_max * strategy.num_replicas_in_sync
     10 
     11     def lrfn(epoch):

NameError: name 'strategy' is not defined

## === cell 16
history = model.fit(
    train_ds,
    epochs=EPOCHS,
    callbacks=[lr_schedule],
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_data=val_ds,
    validation_steps=VALIDATION_STEPS,
    verbose=1,
)



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3566844467.py in <cell line: 0>()
----> 1 history = model.fit(
      2     train_ds,
      3     epochs=EPOCHS,
      4     callbacks=[lr_schedule],
      5     steps_per_epoch=STEPS_PER_EPOCH,

NameError: name 'model' is not defined

## === cell 17
history2 = model2.fit(
    train_ds,
    epochs=EPOCHS,
    callbacks=[lr_schedule],
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_data=val_ds,
    validation_steps=VALIDATION_STEPS,
    verbose=1,
)



## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1258982805.py in <cell line: 0>()
----> 1 history2 = model2.fit(
      2     train_ds,
      3     epochs=EPOCHS,
      4     callbacks=[lr_schedule],
      5     steps_per_epoch=STEPS_PER_EPOCH,

NameError: name 'model2' is not defined

## === cell 18
pass



## === cell 19
test_ds = get_test_dataset(ordered=True)
print("Computing predictions...")

test_ids_list = []
for _, ids in test_ds:
    test_ids_list.append(ids.numpy())
test_ids = np.concatenate(test_ids_list, axis=0).astype("U")

test_img_ds = test_ds.map(lambda img, ids: img, num_parallel_calls=AUTO)

probabilities1 = model.predict(test_img_ds, steps=TEST_STEPS, verbose=1)
probabilities2 = model2.predict(test_img_ds, steps=TEST_STEPS, verbose=1)

probabilities = 0.5 * probabilities1 + 0.5 * probabilities2
probabilities = np.clip(probabilities.reshape(-1), 0.0, 1.0)

if len(probabilities) != len(test_ids):
    min_len = min(len(probabilities), len(test_ids))
    probabilities = probabilities[:min_len]
    test_ids = test_ids[:min_len]



## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2842440331.py in <cell line: 0>()
----> 1 test_ds = get_test_dataset(ordered=True)
      2 print("Computing predictions...")
      3 
      4 # CHANGE (timeout): Collect test_ids without expensive unbatch()/re-batch() reshaping.
      5 # This preserves ordering (ordered=True) and yields identical IDs.

NameError: name 'get_test_dataset' is not defined

## === cell 20
sub1 = sub.copy()
sub2 = sub.copy()



## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1207565556.py in <cell line: 0>()
----> 1 sub1 = sub.copy()
      2 sub2 = sub.copy()
      3 

NameError: name 'sub' is not defined

## === cell 21
print("Generating submission.csv file...")
print("Test IDs:", len(test_ids), "Preds:", probabilities.shape)



## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/645848708.py in <cell line: 0>()
      1 print("Generating submission.csv file...")
----> 2 print("Test IDs:", len(test_ids), "Preds:", probabilities.shape)
      3 

NameError: name 'test_ids' is not defined

## === cell 22
pred_df = pd.DataFrame({"image_name": test_ids, "target": probabilities})
pred_df.head()



## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2425912406.py in <cell line: 0>()
----> 1 pred_df = pd.DataFrame({"image_name": test_ids, "target": probabilities})
      2 pred_df.head()
      3 

NameError: name 'test_ids' is not defined

## === cell 23
sub_out = sub[["image_name"]].merge(
    pred_df, on="image_name", how="left", validate="one_to_one"
)

if sub_out["target"].isna().any():
    sub_out["target"] = sub_out["target"].fillna(
        float(np.nanmean(sub_out["target"].values))
    )

sub_out["target"] = np.clip(sub_out["target"].astype(np.float32), 0.0, 1.0)

sub_out.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub_out.shape)
sub_out.head()



## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3948434114.py in <cell line: 0>()
----> 1 sub_out = sub[["image_name"]].merge(
      2     pred_df, on="image_name", how="left", validate="one_to_one"
      3 )
      4 
      5 if sub_out["target"].isna().any():

NameError: name 'sub' is not defined

## === cell 24
sub1.head()



## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1135934688.py in <cell line: 0>()
----> 1 sub1.head()
      2 

NameError: name 'sub1' is not defined

## === cell 25
sub2.head()



## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3201570645.py in <cell line: 0>()
----> 1 sub2.head()
      2 

NameError: name 'sub2' is not defined

## === cell 26
pass



## === cell 27
pass



## === cell 28
pass



## === cell 29
sub_out.to_csv("submission-sample.csv", index=False)
sub_out.head()



## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/305830529.py in <cell line: 0>()
----> 1 sub_out.to_csv("submission-sample.csv", index=False)
      2 sub_out.head()
      3 

NameError: name 'sub_out' is not defined

## === cell 30
pass



## === cell 31
pass



## === cell 32
pass



## === cell 33
print("Final submission file ready at: submission.csv")



## === cell 34
model.save("EffNetB6-Melanoma.h5")
model2.save("EffNetB3-Melanoma.h5")
print("Saved model files.")

## --- ERROR in cell 34, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/516393683.py in <cell line: 0>()
----> 1 model.save("EffNetB6-Melanoma.h5")
      2 model2.save("EffNetB3-Melanoma.h5")
      3 print("Saved model files.")

NameError: name 'model' is not defined
