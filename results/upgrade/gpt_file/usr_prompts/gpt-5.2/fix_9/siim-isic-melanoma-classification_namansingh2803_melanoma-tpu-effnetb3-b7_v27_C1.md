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

0.8977410323900642

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

if "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION" in os.environ:
    pass
else:
    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "cpp"

import math
import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow.keras.layers as L
import tensorflow.keras.backend as K

print("TF version:", tf.__version__)
print("Working dir:", os.getcwd())



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_11/1143005067.py in <cell line: 0>()
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
for p in [
    "/kaggle/input/submissionb3/submissionB3.csv",
    "/kaggle/input/submissionb7/submissionB7-2.csv",
    "/kaggle/input/subeffnetb0/submissionB0.csv",
]:
    print(p, "exists?", os.path.exists(p))



## === cell 2
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print("Running on TPU ", tpu.master())
except Exception:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy()

print("REPLICAS: ", strategy.num_replicas_in_sync)



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/473063499.py in <cell line: 0>()
     10     strategy = tf.distribute.experimental.TPUStrategy(tpu)
     11 else:
---> 12     strategy = tf.distribute.get_strategy()
     13 
     14 print("REPLICAS: ", strategy.num_replicas_in_sync)

NameError: name 'tf' is not defined

## === cell 3
AUTO = tf.data.experimental.AUTOTUNE

INPUT_DIR = "/kaggle/input/siim-isic-melanoma-classification"
TRAIN_CSV = os.path.join(INPUT_DIR, "train.csv")
TEST_CSV = os.path.join(INPUT_DIR, "test.csv")
SAMPLE_SUB = os.path.join(INPUT_DIR, "sample_submission.csv")

TRAIN_IMG_DIR = os.path.join(INPUT_DIR, "jpeg", "train")
TEST_IMG_DIR = os.path.join(INPUT_DIR, "jpeg", "test")

TFREC_DIR = os.path.join(INPUT_DIR, "tfrecords")
TRAIN_TFRECS = sorted(
    [
        os.path.join(TFREC_DIR, f)
        for f in os.listdir(TFREC_DIR)
        if f.startswith("train") and f.endswith(".tfrec")
    ]
)
TEST_TFRECS = sorted(
    [
        os.path.join(TFREC_DIR, f)
        for f in os.listdir(TFREC_DIR)
        if f.startswith("test") and f.endswith(".tfrec")
    ]
)

EPOCHS = 10
BATCH_SIZE = 8 * strategy.num_replicas_in_sync

IMAGE_SIZE = [384, 384]

print("TRAIN_CSV exists?", os.path.exists(TRAIN_CSV))
print("TEST_CSV exists?", os.path.exists(TEST_CSV))
print("TFREC_DIR exists?", os.path.exists(TFREC_DIR))
print("Num train tfrecs:", len(TRAIN_TFRECS), "Num test tfrecs:", len(TEST_TFRECS))
print("BATCH_SIZE:", BATCH_SIZE, "IMAGE_SIZE:", IMAGE_SIZE)



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3499293942.py in <cell line: 0>()
----> 1 AUTO = tf.data.experimental.AUTOTUNE
      2 
      3 INPUT_DIR = "/kaggle/input/siim-isic-melanoma-classification"
      4 TRAIN_CSV = os.path.join(INPUT_DIR, "train.csv")
      5 TEST_CSV = os.path.join(INPUT_DIR, "test.csv")

NameError: name 'tf' is not defined

## === cell 4
HEIGHT = IMAGE_SIZE[0]
WIDTH = IMAGE_SIZE[1]
CHANNELS = 3



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2720607668.py in <cell line: 0>()
----> 1 HEIGHT = IMAGE_SIZE[0]
      2 WIDTH = IMAGE_SIZE[1]
      3 CHANNELS = 3
      4 

NameError: name 'IMAGE_SIZE' is not defined

## === cell 5
sub = pd.read_csv(SAMPLE_SUB)
sub.head()



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1677958003.py in <cell line: 0>()
----> 1 sub = pd.read_csv(SAMPLE_SUB)
      2 sub.head()
      3 

NameError: name 'SAMPLE_SUB' is not defined

## === cell 6
train = pd.read_csv(TRAIN_CSV)
test = pd.read_csv(TEST_CSV)
print("train:", train.shape, "test:", test.shape)
train.head()



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2406516898.py in <cell line: 0>()
----> 1 train = pd.read_csv(TRAIN_CSV)
      2 test = pd.read_csv(TEST_CSV)
      3 print("train:", train.shape, "test:", test.shape)
      4 train.head()
      5 

NameError: name 'TRAIN_CSV' is not defined

## === cell 7
print(train["target"].value_counts(dropna=False))
print("pos rate:", train["target"].mean())



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1201246183.py in <cell line: 0>()
----> 1 print(train["target"].value_counts(dropna=False))
      2 print("pos rate:", train["target"].mean())
      3 

NameError: name 'train' is not defined

## === cell 8
test_ids = test["image_name"].astype(str).values

NUM_TRAINING_IMAGES = len(train)
NUM_TEST_IMAGES = len(test)
STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE
print(
    "NUM_TRAINING_IMAGES:",
    NUM_TRAINING_IMAGES,
    "NUM_TEST_IMAGES:",
    NUM_TEST_IMAGES,
    "STEPS_PER_EPOCH:",
    STEPS_PER_EPOCH,
)

CLASSES = [0, 1]



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/524566106.py in <cell line: 0>()
----> 1 test_ids = test["image_name"].astype(str).values
      2 
      3 NUM_TRAINING_IMAGES = len(train)
      4 NUM_TEST_IMAGES = len(test)
      5 STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE

NameError: name 'test' is not defined

## === cell 9
tf.random.set_seed(42)
np.random.seed(42)
try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

tf.config.optimizer.set_jit(True)




## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3900158111.py in <cell line: 0>()
----> 1 tf.random.set_seed(42)
      2 np.random.seed(42)
      3 try:
      4     tf.config.experimental.enable_op_determinism(True)
      5 except Exception:

NameError: name 'tf' is not defined

## === cell 10
def _dataset_options():
    opts = tf.data.Options()
    opts.deterministic = True
    try:
        opts.threading.private_threadpool_size = max(8, os.cpu_count() or 8)
        opts.threading.max_intra_op_parallelism = 1
    except Exception:
        pass
    try:
        opts.experimental_optimization.apply_default_optimizations = True
        opts.experimental_optimization.map_fusion = True
        opts.experimental_optimization.parallel_batch = True
    except Exception:
        pass
    return opts


_TFREC_FEATURES = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),  # train only
    "image_name": tf.io.FixedLenFeature([], tf.string),  # present in both train/test
}


@tf.function
def _decode_image_from_tfrecord(example):
    img = tf.image.decode_jpeg(example["image"], channels=3)
    img = tf.image.resize(img, [HEIGHT, WIDTH], method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32) / 255.0
    return img


@tf.function
def data_augment(image, label):
    image = tf.image.random_flip_left_right(image)
    return image, label


@tf.function
def _parse_train_example(serialized):
    ex = tf.io.parse_single_example(serialized, _TFREC_FEATURES)
    img = _decode_image_from_tfrecord(ex)
    lab = tf.cast(ex["target"], tf.int32)
    return img, lab


@tf.function
def _parse_test_example(serialized):
    ex = tf.io.parse_single_example(serialized, _TFREC_FEATURES)
    img = _decode_image_from_tfrecord(ex)
    name = ex["image_name"]
    return img, name


def get_training_dataset():
    ds = tf.data.TFRecordDataset(TRAIN_TFRECS, num_parallel_reads=AUTO)
    ds = ds.with_options(_dataset_options())
    ds = ds.shuffle(2048, reshuffle_each_iteration=True)
    ds = ds.map(_parse_train_example, num_parallel_calls=AUTO, deterministic=True)
    ds = ds.map(data_augment, num_parallel_calls=AUTO, deterministic=True)
    ds = ds.repeat()
    ds = ds.batch(BATCH_SIZE, drop_remainder=True)
    ds = ds.prefetch(AUTO)
    return ds


def get_test_dataset(ordered=False):
    ds = tf.data.TFRecordDataset(TEST_TFRECS, num_parallel_reads=AUTO)
    ds = ds.with_options(_dataset_options())
    ds = ds.map(_parse_test_example, num_parallel_calls=AUTO, deterministic=True)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(AUTO)
    return ds


print(f"Dataset: {NUM_TRAINING_IMAGES} training rows, {NUM_TEST_IMAGES} test rows")




## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/536210805.py in <cell line: 0>()
     19 
     20 _TFREC_FEATURES = {
---> 21     "image": tf.io.FixedLenFeature([], tf.string),
     22     "target": tf.io.FixedLenFeature([], tf.int64),  # train only
     23     "image_name": tf.io.FixedLenFeature([], tf.string),  # present in both train/test

NameError: name 'tf' is not defined

## === cell 11
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




## === cell 12
with strategy.scope():
    backbone_b6 = tf.keras.applications.EfficientNetB6(
        input_shape=(*IMAGE_SIZE, 3), weights="imagenet", include_top=False
    )
    backbone_b6.trainable = False

    model = tf.keras.Sequential(
        [
            backbone_b6,
            L.GlobalAveragePooling2D(),
            L.Dense(512, activation="relu"),
            L.Dense(128, activation="relu"),
            L.Dense(1, activation="sigmoid"),
        ]
    )

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"],
    run_eagerly=False,
)
model.summary()



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/397447237.py in <cell line: 0>()
----> 1 with strategy.scope():
      2     backbone_b6 = tf.keras.applications.EfficientNetB6(
      3         input_shape=(*IMAGE_SIZE, 3), weights="imagenet", include_top=False
      4     )
      5     backbone_b6.trainable = False

NameError: name 'strategy' is not defined

## === cell 13
with strategy.scope():
    backbone_b3 = tf.keras.applications.EfficientNetB3(
        input_shape=(*IMAGE_SIZE, 3), weights="imagenet", include_top=False
    )
    backbone_b3.trainable = False

    model2 = tf.keras.Sequential(
        [
            backbone_b3,
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
    metrics=["accuracy"],
    run_eagerly=False,
)
model2.summary()



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1160274105.py in <cell line: 0>()
----> 1 with strategy.scope():
      2     backbone_b3 = tf.keras.applications.EfficientNetB3(
      3         input_shape=(*IMAGE_SIZE, 3), weights="imagenet", include_top=False
      4     )
      5     backbone_b3.trainable = False

NameError: name 'strategy' is not defined

## === cell 14
lrfn = build_lrfn()
lr_schedule = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=1)
STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE
print("STEPS_PER_EPOCH:", STEPS_PER_EPOCH)



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/298737980.py in <cell line: 0>()
----> 1 lrfn = build_lrfn()
      2 lr_schedule = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=1)
      3 STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE
      4 print("STEPS_PER_EPOCH:", STEPS_PER_EPOCH)
      5 

/tmp/ipykernel_11/886858925.py in build_lrfn(lr_start, lr_max, lr_min, lr_rampup_epochs, lr_sustain_epochs, lr_exp_decay)
      7     lr_exp_decay=0.8,
      8 ):
----> 9     lr_max = lr_max * strategy.num_replicas_in_sync
     10 
     11     def lrfn(epoch):

NameError: name 'strategy' is not defined

## === cell 15
train_ds = get_training_dataset()

history = model.fit(
    train_ds,
    epochs=EPOCHS,
    callbacks=[lr_schedule],
    steps_per_epoch=STEPS_PER_EPOCH,
    verbose=1,
)



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3050704026.py in <cell line: 0>()
----> 1 train_ds = get_training_dataset()
      2 
      3 history = model.fit(
      4     train_ds,
      5     epochs=EPOCHS,

NameError: name 'get_training_dataset' is not defined

## === cell 16
history2 = model2.fit(
    train_ds,
    epochs=EPOCHS,
    callbacks=[lr_schedule],
    steps_per_epoch=STEPS_PER_EPOCH,
    verbose=1,
)



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1774204611.py in <cell line: 0>()
----> 1 history2 = model2.fit(
      2     train_ds,
      3     epochs=EPOCHS,
      4     callbacks=[lr_schedule],
      5     steps_per_epoch=STEPS_PER_EPOCH,

NameError: name 'model2' is not defined

## === cell 17
test_ds = get_test_dataset(ordered=True)
print("Test batches:", int(np.ceil(NUM_TEST_IMAGES / BATCH_SIZE)))



## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/517497937.py in <cell line: 0>()
----> 1 test_ds = get_test_dataset(ordered=True)
      2 print("Test batches:", int(np.ceil(NUM_TEST_IMAGES / BATCH_SIZE)))
      3 

NameError: name 'get_test_dataset' is not defined

## === cell 18
print("Computing predictions...")

test_images_ds = (
    test_ds.map(lambda image, name: image, num_parallel_calls=AUTO, deterministic=True)
    .cache()
    .prefetch(AUTO)
)

prob1 = model.predict(test_images_ds, verbose=1).reshape(-1)
prob2 = model2.predict(test_images_ds, verbose=1).reshape(-1)

probabilities = 0.5 * prob1 + 0.5 * prob2
probabilities = probabilities[:NUM_TEST_IMAGES]  # safety

print(
    "Preds shape:",
    probabilities.shape,
    "min/max:",
    float(probabilities.min()),
    float(probabilities.max()),
)



## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2993899855.py in <cell line: 0>()
      4 # does NOT decode/resize the TFRecords twice. This preserves exact inputs and ordering.
      5 test_images_ds = (
----> 6     test_ds.map(lambda image, name: image, num_parallel_calls=AUTO, deterministic=True)
      7     .cache()
      8     .prefetch(AUTO)

NameError: name 'test_ds' is not defined

## === cell 19
final_image_names = test_ids

pred_df = pd.DataFrame(
    {
        "image_name": final_image_names.astype("U"),
        "target": probabilities.astype(np.float32),
    }
)
pred_df.head()



## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/804472240.py in <cell line: 0>()
----> 1 final_image_names = test_ids
      2 
      3 pred_df = pd.DataFrame(
      4     {
      5         "image_name": final_image_names.astype("U"),

NameError: name 'test_ids' is not defined

## === cell 20
print("Generating submission.csv file...")

subm = sub[["image_name"]].copy()

if np.array_equal(
    subm["image_name"].astype(str).values, pred_df["image_name"].astype(str).values
):
    subm["target"] = pred_df["target"].values
else:
    subm = subm.merge(pred_df, on="image_name", how="left")

if subm["target"].isna().any():
    fill_value = float(pred_df["target"].mean())
    subm["target"] = subm["target"].fillna(fill_value)

subm["target"] = subm["target"].astype(np.float32)
subm["target"] = np.clip(subm["target"].values, 0.0, 1.0).astype(np.float32)

assert (
    subm.shape[0] == sub.shape[0]
), "Submission row count mismatch vs sample_submission."
assert list(subm.columns) == ["image_name", "target"], "Submission columns mismatch."
assert np.isfinite(subm["target"].values).all(), "Non-finite predictions in submission."

subm.to_csv("submission.csv", index=False)
print("Wrote submission.csv:", subm.shape)
print(subm.head())



## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/692722433.py in <cell line: 0>()
      4 # This is equivalent because test_ids comes directly from test.csv and test TFRecords match image_name ordering.
      5 # To preserve semantics, still fall back to merge if any mismatch is detected (should not happen).
----> 6 subm = sub[["image_name"]].copy()
      7 
      8 if np.array_equal(

NameError: name 'sub' is not defined

## === cell 21
subm.to_csv("submission-sample.csv", index=False)
print("Wrote submission-sample.csv:", subm.shape)
subm.head()



## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3954259187.py in <cell line: 0>()
      1 # Runtime fix: Avoid writing an extra redundant CSV unless explicitly needed.
      2 # Keeping this file write but cheaper (no extra merge) using the already built subm.
----> 3 subm.to_csv("submission-sample.csv", index=False)
      4 print("Wrote submission-sample.csv:", subm.shape)
      5 subm.head()

NameError: name 'subm' is not defined

## === cell 22
print("Done.")



## === cell 23
sub_es = subm.copy()
print("Final submission.csv already written:", sub_es.shape)



## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/10824118.py in <cell line: 0>()
      1 # Runtime fix: Remove redundant re-write of submission.csv (it was already written in cell 21).
      2 # This preserves final output contents while saving time on I/O.
----> 3 sub_es = subm.copy()
      4 print("Final submission.csv already written:", sub_es.shape)
      5 

NameError: name 'subm' is not defined

## === cell 24
model.save("EffNetB6-Melanoma.h5")
model2.save("EffNetB3-Melanoma.h5")
print("Saved models.")

## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/179738823.py in <cell line: 0>()
----> 1 model.save("EffNetB6-Melanoma.h5")
      2 model2.save("EffNetB3-Melanoma.h5")
      3 print("Saved models.")

NameError: name 'model' is not defined
