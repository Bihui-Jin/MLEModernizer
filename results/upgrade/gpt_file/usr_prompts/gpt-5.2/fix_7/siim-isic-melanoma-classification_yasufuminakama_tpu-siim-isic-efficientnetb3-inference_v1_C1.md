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

0.8271057910391251

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'I remove the expensive full 1024×1024 inference by resizing to EfficientNetB3’s native 300×300 inside the TFRecord decode (this preserves the same model architecture/weights and only makes the input pipeline match how these public weights were trained/used). I also eliminate Python-side accumulation of IDs/preds and `.numpy()` roundtrips by letting `model.predict()` stream over the dataset once, while separately collecting IDs with a light, deterministic dataset pass. Finally, I drop an unused helper (`append_path`) and avoid unnecessary `tf.reshape` to 1024×1024 that forces huge tensors, which is the dominant timeout cause; all changes are input-pipeline/inference plumbing and keep evaluation semantics identical.'
- What this solution (achieved 0.32696) has done: 'I fix the TensorFlow import crash by forcing the Python protobuf implementation before importing TensorFlow (this resolves the `MessageFactory.GetPrototype` error in Kaggle’s environment). Then I remove the hard dependency on the missing external weights file by switching EfficientNetB3 to use built-in ImageNet weights (same architecture and head), which should move your AUC meaningfully above the 0.5 baseline without changing the core approach. Finally, I make the TFRecord parsing robust to both common key variants (`image_name` vs `image_id`) so the test id collection doesn’t silently fail, and I ensure the submission is written as `submission.csv` with the required columns and correct alignment to `sample_submission.csv`.'

# 9. Code solution

## === cell 0
import os, sys, re, math

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "cpp")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow.keras.layers as L

print("TensorFlow:", tf.__version__)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_11/2747818125.py in <cell line: 0>()
     10 import pandas as pd
     11 
---> 12 import tensorflow as tf
     13 import tensorflow.keras.layers as L
     14 

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
from tensorflow.keras.applications import EfficientNetB3



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_11/54437960.py in <cell line: 0>()
----> 1 from tensorflow.keras.applications import EfficientNetB3
      2 

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
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2057276366.py in <cell line: 0>()
      1 SEED = 42
----> 2 tf.random.set_seed(SEED)
      3 np.random.seed(SEED)
      4 os.environ["PYTHONHASHSEED"] = str(SEED)
      5 

NameError: name 'tf' is not defined

## === cell 3
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



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/473063499.py in <cell line: 0>()
     10     strategy = tf.distribute.experimental.TPUStrategy(tpu)
     11 else:
---> 12     strategy = tf.distribute.get_strategy()
     13 
     14 print("REPLICAS: ", strategy.num_replicas_in_sync)

NameError: name 'tf' is not defined

## === cell 4
AUTO = tf.data.experimental.AUTOTUNE

BASE_PATH = "/kaggle/input/siim-isic-melanoma-classification"
TFRECORD_PATH = os.path.join(BASE_PATH, "tfrecords")

EPOCHS = 1
BATCH_SIZE = 8 * strategy.num_replicas_in_sync

IMAGE_SIZE = [300, 300]



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/156854282.py in <cell line: 0>()
----> 1 AUTO = tf.data.experimental.AUTOTUNE
      2 
      3 BASE_PATH = "/kaggle/input/siim-isic-melanoma-classification"
      4 TFRECORD_PATH = os.path.join(BASE_PATH, "tfrecords")
      5 

NameError: name 'tf' is not defined

## === cell 5
sub = pd.read_csv(os.path.join(BASE_PATH, "sample_submission.csv"))
print("Sample submission shape:", sub.shape)
print("Sample submission columns:", list(sub.columns))



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/392543540.py in <cell line: 0>()
----> 1 sub = pd.read_csv(os.path.join(BASE_PATH, "sample_submission.csv"))
      2 print("Sample submission shape:", sub.shape)
      3 print("Sample submission columns:", list(sub.columns))
      4 

NameError: name 'BASE_PATH' is not defined

## === cell 6
TRAINING_FILENAMES = tf.io.gfile.glob(os.path.join(TFRECORD_PATH, "train*.tfrec"))
TEST_FILENAMES = tf.io.gfile.glob(os.path.join(TFRECORD_PATH, "test*.tfrec"))

if len(TRAINING_FILENAMES) == 0 or len(TEST_FILENAMES) == 0:
    raise FileNotFoundError(
        f"TFRecords not found. TRAIN={len(TRAINING_FILENAMES)} TEST={len(TEST_FILENAMES)} "
        f"under {TFRECORD_PATH}"
    )

TRAINING_FILENAMES = sorted(TRAINING_FILENAMES)
TEST_FILENAMES = sorted(TEST_FILENAMES)




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3785473418.py in <cell line: 0>()
----> 1 TRAINING_FILENAMES = tf.io.gfile.glob(os.path.join(TFRECORD_PATH, "train*.tfrec"))
      2 TEST_FILENAMES = tf.io.gfile.glob(os.path.join(TFRECORD_PATH, "test*.tfrec"))
      3 
      4 if len(TRAINING_FILENAMES) == 0 or len(TEST_FILENAMES) == 0:
      5     raise FileNotFoundError(

NameError: name 'tf' is not defined

## === cell 7
def decode_image(image_data):
    image = tf.image.decode_jpeg(image_data, channels=3)
    image = tf.image.resize(image, IMAGE_SIZE, method="bilinear", antialias=True)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.ensure_shape(image, [*IMAGE_SIZE, 3])
    return image


def read_labeled_tfrecord(example):
    LABELED_TFREC_FORMAT = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "target": tf.io.FixedLenFeature([], tf.int64),
    }
    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)
    image = decode_image(example["image"])
    label = tf.cast(example["target"], tf.int32)
    return image, label


def read_unlabeled_tfrecord(example):
    UNLABELED_TFREC_FORMAT = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "image_name": tf.io.FixedLenFeature([], tf.string, default_value=b""),
        "image_id": tf.io.FixedLenFeature([], tf.string, default_value=b""),
    }
    example = tf.io.parse_single_example(example, UNLABELED_TFREC_FORMAT)
    image = decode_image(example["image"])
    idnum = tf.cond(
        tf.strings.length(example["image_name"]) > 0,
        lambda: example["image_name"],
        lambda: example["image_id"],
    )
    return image, idnum


def load_dataset(filenames, labeled=True, ordered=False):
    opt = tf.data.Options()
    if not ordered:
        opt.experimental_deterministic = False

    ds = tf.data.TFRecordDataset(
        filenames,
        num_parallel_reads=AUTO,
        compression_type=None,
        buffer_size=8 * 1024 * 1024,
    )
    ds = ds.with_options(opt)
    ds = ds.map(
        read_labeled_tfrecord if labeled else read_unlabeled_tfrecord,
        num_parallel_calls=AUTO,
    )
    return ds


def data_augment(image, label):
    image = tf.image.random_flip_left_right(image)
    return image, label


def get_training_dataset():
    ds = load_dataset(TRAINING_FILENAMES, labeled=True)
    ds = ds.map(data_augment, num_parallel_calls=AUTO)
    ds = ds.repeat()
    ds = ds.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
    ds = ds.batch(BATCH_SIZE, drop_remainder=True)
    ds = ds.prefetch(AUTO)
    return ds


def get_test_dataset(ordered=False):
    ds = load_dataset(TEST_FILENAMES, labeled=False, ordered=ordered)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(AUTO)
    return ds


def count_data_items(filenames):
    n = [int(re.compile(r"-([0-9]*)\.").search(fn).group(1)) for fn in filenames]
    return int(np.sum(n))


NUM_TRAINING_IMAGES = count_data_items(TRAINING_FILENAMES)
NUM_TEST_IMAGES = count_data_items(TEST_FILENAMES)
STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE

print(
    f"Dataset: {NUM_TRAINING_IMAGES} training images, {NUM_TEST_IMAGES} unlabeled test images"
)
print(f"Shards: {len(TRAINING_FILENAMES)} train, {len(TEST_FILENAMES)} test")
print(f"BATCH_SIZE={BATCH_SIZE}, STEPS_PER_EPOCH={STEPS_PER_EPOCH}")




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3135376627.py in <cell line: 0>()
     80 
     81 
---> 82 NUM_TRAINING_IMAGES = count_data_items(TRAINING_FILENAMES)
     83 NUM_TEST_IMAGES = count_data_items(TEST_FILENAMES)
     84 STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE

NameError: name 'TRAINING_FILENAMES' is not defined

## === cell 8
def build_lrfn(
    lr_start=0.00001,
    lr_max=0.000075,
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




## === cell 9
with strategy.scope():
    backbone = EfficientNetB3(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet",
        pooling=None,
    )
    model = tf.keras.Sequential(
        [
            backbone,
            L.GlobalAveragePooling2D(),
            L.Dense(1, activation="sigmoid"),
        ]
    )

model.compile(optimizer="adam", loss="binary_crossentropy")

print("Model built. Backbone weights:", "imagenet")



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1925973853.py in <cell line: 0>()
----> 1 with strategy.scope():
      2     backbone = EfficientNetB3(
      3         input_shape=(*IMAGE_SIZE, 3),
      4         include_top=False,
      5         weights="imagenet",

NameError: name 'strategy' is not defined

## === cell 10
test_ds = get_test_dataset(ordered=True)

print("Collecting test ids (ordered) ...")
id_ds = test_ds.map(lambda x, y: y, num_parallel_calls=AUTO).unbatch()
test_ids = np.fromiter((x.numpy() for x in id_ds), dtype=object, count=NUM_TEST_IMAGES)


def _to_str_array(arr):
    out = []
    for x in arr:
        if isinstance(x, (bytes, bytearray)):
            out.append(x.decode("utf-8"))
        else:
            out.append(str(x))
    return np.asarray(out)


test_ids = _to_str_array(test_ids)
if (len(test_ids) == 0) or (np.mean([len(s) for s in test_ids]) < 3):
    print("TFRecord ids look empty; falling back to test.csv image_name order.")
    test_csv = pd.read_csv(os.path.join(BASE_PATH, "test.csv"))
    test_ids = test_csv["image_name"].astype(str).values

print("Computing predictions with model.predict (streamed) ...")
preds = model.predict(
    test_ds.map(lambda x, y: x, num_parallel_calls=AUTO),
    verbose=0,
)
probabilities = preds.reshape(-1)



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1936527169.py in <cell line: 0>()
----> 1 test_ds = get_test_dataset(ordered=True)
      2 
      3 print("Collecting test ids (ordered) ...")
      4 id_ds = test_ds.map(lambda x, y: y, num_parallel_calls=AUTO).unbatch()
      5 test_ids = np.fromiter((x.numpy() for x in id_ds), dtype=object, count=NUM_TEST_IMAGES)

/tmp/ipykernel_11/3135376627.py in get_test_dataset(ordered)
     69 
     70 def get_test_dataset(ordered=False):
---> 71     ds = load_dataset(TEST_FILENAMES, labeled=False, ordered=ordered)
     72     ds = ds.batch(BATCH_SIZE)
     73     ds = ds.prefetch(AUTO)

NameError: name 'TEST_FILENAMES' is not defined

## === cell 11
print("Generating submission.csv file...")
preds = probabilities.astype(np.float32)
preds = np.clip(preds, 0.0, 1.0)

if len(preds) != len(test_ids):
    min_len = min(len(preds), len(test_ids))
    preds = preds[:min_len]
    test_ids = test_ids[:min_len]

pred_df = pd.DataFrame({"image_name": test_ids, "target": preds})
print("Pred df shape:", pred_df.shape)
print(pred_df.head())

sub_out = sub[["image_name"]].merge(pred_df, on="image_name", how="left")

if sub_out["target"].isna().any():
    sub_out["target"] = sub_out["target"].fillna(
        float(np.nanmean(pred_df["target"].values))
    )

sub_out.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub_out.shape)
print(sub_out.head())
print("submission.csv columns:", list(sub_out.columns))
print("Missing targets after merge:", int(sub_out["target"].isna().sum()))

## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/603273368.py in <cell line: 0>()
      1 print("Generating submission.csv file...")
----> 2 preds = probabilities.astype(np.float32)
      3 preds = np.clip(preds, 0.0, 1.0)
      4 
      5 if len(preds) != len(test_ids):

NameError: name 'probabilities' is not defined
