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
Create a classifier to predict the severity of diabetic retinopathy.

## Metric
Quadratic weighted kappa, which measures the agreement between two ratings. This metric typically varies from 0 (random agreement between raters) to 1 (complete agreement between raters). In the event that there is less agreement between the raters than expected by chance, this metric may go below 0. The quadratic weighted kappa is calculated between the scores assigned by the human rater and the predicted scores.

Images have five possible ratings, 0,1,2,3,4.  Each image is characterized by a tuple *(e*,*e)*, which corresponds to its scores by *Rater A* (human) and *Rater B* (predicted).  The quadratic weighted kappa is calculated as follows. First, an N x N histogram matrix *O* is constructed, such that *O* corresponds to the number of images that received a rating *i* by *A* and a rating *j* by *B*. An *N-by-N* matrix of weights, *w*, is calculated based on the difference between raters' scores:

An *N-by-N* histogram matrix of expected ratings, *E*, is calculated, assuming that there is no correlation between rating scores.  This is calculated as the outer product between each rater's histogram vector of ratings, normalized such that *E* and *O* have the same sum.

## Submission Format
```
id_code,diagnosis
0005cfc8afb6,0
003f0afdcd15,0
etc.
```

## Dataset
You are provided with a large set of retina images taken using [fundus photography](https://en.wikipedia.org/wiki/Fundus_photography) under a variety of imaging conditions.

Labels are on a scale of 0 to 4:

> 0 - No DR
> 1 - Mild
> 2 - Moderate
> 3 - Severe
> 4 - Proliferative DR

Images may contain artifacts, be out of focus, underexposed, or overexposed. The images were gathered from multiple clinics using a variety of cameras over an extended period of time, which will introduce further variation.

- **train.csv** - the training labels
- **test.csv** - the test set (you must predict the `diagnosis` value for these variables)
- **sample_submission.csv** - a sample submission file in the correct format
- **train.zip** - the training set images
- **test.zip** - the public test set images

# 2. Python version

3.7

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (118 lines)
            sample_submission.csv (368 lines)
            sample_submission.csv.zip (3.2 kB)
            test.csv (368 lines)
            test.csv.zip (2.9 kB)
            test.zip (160 Bytes)
            test_images.zip (902.9 MB)
            train.csv (3296 lines)
            train.csv.zip (27.5 kB)
            train.zip (162 Bytes)
            train_images.zip (7.7 GB)
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
            test_images/
                218c822a3dd9.png (5.7 MB)
                0e82bcacc475.png (5.2 MB)
                ... and 365 other files
                test_images/
            train_images/
                184a185e7447.png (337.5 kB)
                c4aef0d88d1b.png (876.6 kB)
                ... and 3293 other files
                train_images/
        input/
            description.md (118 lines)
            sample_submission.csv (368 lines)
            sample_submission.csv.zip (3.2 kB)
            test.csv (368 lines)
            test.csv.zip (2.9 kB)
            test.zip (160 Bytes)
            test_images.zip (902.9 MB)
            train.csv (3296 lines)
            train.csv.zip (27.5 kB)
            train.zip (162 Bytes)
            train_images.zip (7.7 GB)
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
            test_images/
                218c822a3dd9.png (5.7 MB)
                0e82bcacc475.png (5.2 MB)
                ... and 365 other files
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
            train_images/
                184a185e7447.png (337.5 kB)
                c4aef0d88d1b.png (876.6 kB)
                ... and 3293 other files
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
        working/
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
```

-> data/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/aptos2019-blindness-detection/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> data/aptos2019-blindness-detection/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> data/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> input/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> (stopped after 10 files for performance)

# 5. Target score

0.7333452683918757

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the TensorFlow/Keras import errors by switching from `tensorflow.python.keras` to the supported `tensorflow.keras` API, which also resolves the `BatchNormalization` and `ImageDataGenerator` failures. I also remove notebook-only shell commands (`!mkdir`) and replace them with `os.makedirs` so the script runs as a plain Python file in Kaggle. To ensure inference works even when no external weight file is available, I make weight loading optional and fall back to a freshly initialized model (still producing a valid `submission.csv`). Finally, I fix a few execution-order/name issues (like `MODEL_NAME` not being defined due to earlier import failures) and update deprecated `.fit_generator/.predict_generator` to `.fit/.predict` for compatibility.'
- What this solution (achieved 0.0) has done: 'I fix the TensorFlow import crash (`MessageFactory GetPrototype`) by forcing the pure-Python protobuf implementation before importing TensorFlow, which is a common Kaggle runtime incompatibility. Then I fix the score=0.0 issue by ensuring inference uses a real trained model: automatically locate and load the provided conv1 weight file from any attached input dataset (instead of only one hardcoded directory), while keeping the architecture/training logic unchanged. I also make the submission alignment robust by using `test["id_code"]` order (not generator filenames) and writing integer diagnoses 0–4 with the exact required columns. These changes are minimal, execution-blocking/score-critical, and keep the rest of the pipeline intact.'
- What this solution (achieved 0.0) has done: 'I fix the TensorFlow/protobuf crash by setting the protobuf environment variables before any TensorFlow import (the current placement is too late), which unblocks the whole pipeline. Then I keep the same model/inference logic but make prediction robust by explicitly limiting prediction steps to the generator length so it can’t hang or mismatch. Finally, I keep the submission formatting/alignment identical to the sample submission and ensure the CSV is always written with the required columns and correct row count, which should move the score up from 0.0 to a meaningful value (assuming weights are found/loaded).'
- What this solution (achieved 0.0) has done: 'I fix the TensorFlow/protobuf crash by moving the protobuf environment-variable settings to the very top of the script (before any TensorFlow-related import), which is the root cause of the `MessageFactory.GetPrototype` error. I also make the image directory resolution robust (some Kaggle setups have the files under `/kaggle/input/...` instead of `../input/...`) without changing any modeling/training logic. Finally, I keep the same weight-loading logic but ensure the code always finds the correct dataset root and writes a correctly formatted `submission.csv` aligned to `test.csv`, which should move the score up from 0.0 (random/unrun) toward the target if pretrained weights are present.'
- What this solution (achieved 0.0) has done: 'I fix the TensorFlow/protobuf crash that stops execution by setting the protobuf env vars before any TensorFlow-related import and by forcing a compatible import order. Then I keep the exact same data pipeline/model code, but add a safe fallback to load a matching local weights file if the hardcoded dataset isn’t attached (otherwise you keep getting effectively-random predictions and ~0.0 kappa). Finally, I ensure prediction length and submission alignment always match `test.csv`/`sample_submission.csv`, and that `submission.csv` is written with the required columns and row count.'

# 9. Code solution

## === cell 0
import os
import math
import warnings

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "cpp")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
os.environ.setdefault("PYTHONHASHSEED", "42")

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
np.random.seed(42)

TRAINING = False




## === cell 1
def _resolve_input_dir(preferred_rel_path="../input/aptos2019-blindness-detection"):
    candidates = [
        preferred_rel_path,
        "/kaggle/input/aptos2019-blindness-detection",
        "/kaggle/data/aptos2019-blindness-detection",
        "../input",
        "/kaggle/input",
        "/kaggle/data",
    ]
    for p in candidates:
        p = os.path.abspath(p)
        if os.path.isdir(p) and os.path.isfile(os.path.join(p, "train.csv")):
            return p

    for base in ["/kaggle/input", "/kaggle/data", os.path.abspath("../input")]:
        if os.path.isdir(base):
            for root, _, files in os.walk(base):
                if "train.csv" in files and "test.csv" in files:
                    return root
    return os.path.abspath(preferred_rel_path)


INPUT_DIR = _resolve_input_dir("../input/aptos2019-blindness-detection")
TRAIN_CSV = os.path.join(INPUT_DIR, "train.csv")
TEST_CSV = os.path.join(INPUT_DIR, "test.csv")
TRAIN_IMG_DIR = os.path.join(INPUT_DIR, "train_images")
TEST_IMG_DIR = os.path.join(INPUT_DIR, "test_images")

print("Using INPUT_DIR:", INPUT_DIR)
print("TRAIN_CSV exists:", os.path.isfile(TRAIN_CSV))
print("TEST_CSV exists:", os.path.isfile(TEST_CSV))
print("TRAIN_IMG_DIR exists:", os.path.isdir(TRAIN_IMG_DIR))
print("TEST_IMG_DIR exists:", os.path.isdir(TEST_IMG_DIR))



## === cell 2
train = pd.read_csv(TRAIN_CSV)
test = pd.read_csv(TEST_CSV)

print("Number of train samples: ", train.shape[0])
print("Number of test samples: ", test.shape[0])



## === cell 3
train["id_code"] = train["id_code"].apply(lambda x: str(x) + ".png")
test["id_code"] = test["id_code"].apply(lambda x: str(x) + ".png")
train["diagnosis"] = train["diagnosis"].astype(str)



## === cell 4
import cv2

try:
    import tensorflow as tf
except Exception as e:
    print("Initial TensorFlow import failed:", repr(e))
    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
    import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = 224
NB_CHANNELS = 3
NB_CLASSES = 5  # 0, 1, 2, 3, 4
BATCH_SIZE = 32
TEST_BATCH_SIZE = 1

print("TensorFlow version:", tf.__version__)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_55/2007936305.py in <cell line: 0>()
      5 try:
----> 6     import tensorflow as tf
      7 except Exception as e:

/usr/local/lib/python3.11/dist-packages/tensorflow/__init__.py in <module>
     48 
---> 49 from tensorflow._api.v2 import __internal__
     50 from tensorflow._api.v2 import __operators__

/usr/local/lib/python3.11/dist-packages/tensorflow/_api/v2/__internal__/__init__.py in <module>
      7 
----> 8 from tensorflow._api.v2.__internal__ import autograph
      9 from tensorflow._api.v2.__internal__ import decorator

/usr/local/lib/python3.11/dist-packages/tensorflow/_api/v2/__internal__/autograph/__init__.py in <module>
      7 
----> 8 from tensorflow.python.autograph.core.ag_ctx import control_status_ctx # line: 34
      9 from tensorflow.python.autograph.impl.api import tf_convert # line: 493

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/core/ag_ctx.py in <module>
     20 
---> 21 from tensorflow.python.autograph.utils import ag_logging
     22 from tensorflow.python.util.tf_export import tf_export

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/utils/__init__.py in <module>
     16 
---> 17 from tensorflow.python.autograph.utils.context_managers import control_dependency_on_returns
     18 from tensorflow.python.autograph.utils.misc import alias_tensors

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/utils/context_managers.py in <module>
     18 
---> 19 from tensorflow.python.framework import ops
     20 from tensorflow.python.ops import tensor_array_ops

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py in <module>
     32 from google.protobuf import message
---> 33 from tensorflow.core.framework import attr_value_pb2
     34 from tensorflow.core.framework import full_type_pb2

/usr/local/lib/python3.11/dist-packages/tensorflow/core/framework/attr_value_pb2.py in <module>
      4 """Generated protocol buffer code."""
----> 5 from google.protobuf.internal import builder as _builder
      6 from google.protobuf import descriptor as _descriptor

/usr/local/lib/python3.11/dist-packages/google/protobuf/internal/builder.py in <module>
     17 from google.protobuf.internal import enum_type_wrapper
---> 18 from google.protobuf.internal import python_message
     19 from google.protobuf import message as _message

/usr/local/lib/python3.11/dist-packages/google/protobuf/internal/python_message.py in <module>
     37 
---> 38 from google.protobuf import descriptor as descriptor_mod
     39 from google.protobuf import message as message_mod

/usr/local/lib/python3.11/dist-packages/google/protobuf/descriptor.py in <module>
     28   if _message is None:
---> 29     from google.protobuf.pyext import _message
     30   _USE_C_DESCRIPTORS = True

ImportError: cannot import name '_message' from 'google.protobuf.pyext' (/usr/local/lib/python3.11/dist-packages/google/protobuf/pyext/__init__.py)

During handling of the above exception, another exception occurred:

ImportError                               Traceback (most recent call last)
/tmp/ipykernel_55/2007936305.py in <cell line: 0>()
      9     os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
     10     os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
---> 11     import tensorflow as tf
     12 
     13 from tensorflow.keras.preprocessing.image import ImageDataGenerator

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

## === cell 5
"""
crops black parts around the image (intensity is <= tol)
"""


def crop_image(img, tol=10):
    def crop_image_1(img2d):
        mask = img2d > tol
        return img2d[np.ix_(mask.any(1), mask.any(0))]

    if img.ndim == 2:
        return crop_image_1(img)

    elif img.ndim == 3:
        img_cpy = img.copy()
        try:
            h, w, _ = img.shape
            img1 = cv2.resize(crop_image_1(img[:, :, 0]), (w, h))
            img2 = cv2.resize(crop_image_1(img[:, :, 1]), (w, h))
            img3 = cv2.resize(crop_image_1(img[:, :, 2]), (w, h))

            img[:, :, 0] = img1
            img[:, :, 1] = img2
            img[:, :, 2] = img3
        except Exception:
            return img_cpy

        return img

    return img


"""
crops black parts and enhances image (Ben Graham's method)
"""


def preprocess_image(img):
    img = crop_image(img)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), IMG_SIZE / 10), -4, 128)
    return img




## === cell 6
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    validation_split=0.2,
    horizontal_flip=True,
    preprocessing_function=preprocess_image,
)

train_gen = train_datagen.flow_from_dataframe(
    dataframe=train,
    directory=TRAIN_IMG_DIR,
    x_col="id_code",
    y_col="diagnosis",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    target_size=(IMG_SIZE, IMG_SIZE),
    subset="training",
    shuffle=True,
)

val_gen = train_datagen.flow_from_dataframe(
    dataframe=train,
    directory=TRAIN_IMG_DIR,
    x_col="id_code",
    y_col="diagnosis",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    target_size=(IMG_SIZE, IMG_SIZE),
    subset="validation",
    shuffle=True,
)

test_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0, preprocessing_function=preprocess_image
)

test_gen = test_datagen.flow_from_dataframe(
    dataframe=test,
    directory=TEST_IMG_DIR,
    x_col="id_code",
    batch_size=TEST_BATCH_SIZE,
    class_mode=None,
    target_size=(IMG_SIZE, IMG_SIZE),
    shuffle=False,
)



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/661613053.py in <cell line: 0>()
----> 1 train_datagen = ImageDataGenerator(
      2     rescale=1.0 / 255.0,
      3     validation_split=0.2,
      4     horizontal_flip=True,
      5     preprocessing_function=preprocess_image,

NameError: name 'ImageDataGenerator' is not defined

## === cell 7
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    GlobalAveragePooling2D,
    Dense,
    Dropout,
    BatchNormalization,
    Conv2D,
    MaxPooling2D,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from tensorflow.keras.applications.resnet50 import ResNet50

MODEL_NAME = "conv1"

NB_WARMUP_EPOCHS = 2
NB_EPOCHS = 30
INITIAL_LR = 1e-3

weights_path_template = os.path.join(
    "../input/aptos-2019-conv1-weights/", "{}_weights.hdf5"
)
log_path_template = os.path.join("logs/", "{}_training_log.csv")



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_55/1558381928.py in <cell line: 0>()
----> 1 from tensorflow.keras.models import Model
      2 from tensorflow.keras.layers import (
      3     Input,
      4     GlobalAveragePooling2D,
      5     Dense,

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

## === cell 8
os.makedirs("weights", exist_ok=True)
os.makedirs("logs", exist_ok=True)



## === cell 9
"""
ResNet50 based model
"""


def get_resnet50(input_shape, nb_out):
    inputs = Input(shape=input_shape)
    base_model = ResNet50(weights="imagenet", include_top=False, input_tensor=inputs)

    x = GlobalAveragePooling2D()(base_model.output)
    x = Dropout(0.5)(x)

    x = Dense(2048, activation="relu")(x)
    x = Dropout(0.5)(x)

    x = Dense(1024, activation="relu")(x)
    x = Dropout(0.5)(x)

    output = Dense(nb_out, activation="softmax", name="final_output")(x)
    model = Model(inputs, output)
    return model




## === cell 10
"""
simple CNN
"""


def get_conv1(input_shape, nb_out):
    inputs = Input(shape=input_shape)

    x = Conv2D(64, (7, 7), activation="relu")(inputs)
    x = MaxPooling2D((2, 2))(x)
    x = BatchNormalization()(x)

    x = Conv2D(64, (7, 7), activation="relu")(x)
    x = MaxPooling2D((2, 2))(x)
    x = BatchNormalization()(x)

    x = Conv2D(128, (5, 5), activation="relu")(x)
    x = MaxPooling2D((2, 2))(x)
    x = BatchNormalization()(x)

    x = Conv2D(256, (3, 3), activation="relu")(x)
    x = MaxPooling2D((2, 2))(x)
    x = BatchNormalization()(x)

    x = Conv2D(512, (3, 3), activation="relu")(x)
    x = MaxPooling2D((2, 2))(x)
    x = BatchNormalization()(x)

    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)

    x = Dense(2048, activation="relu")(x)
    x = Dropout(0.5)(x)

    x = Dense(1024, activation="relu")(x)
    x = Dropout(0.5)(x)

    output = Dense(nb_out, activation="softmax", name="final_output")(x)
    model = Model(inputs, output)
    return model




## === cell 11
"""
returns model

Score fix: ensure we actually load the pre-trained weights if they exist anywhere under ../input or /kaggle/input.
The previous behavior often resulted in random weights -> ~0 score.
Core model logic is unchanged; only weight path discovery is made robust.
"""


def _find_weight_file(filename):
    candidate = os.path.join("../input/aptos-2019-conv1-weights", filename)
    if os.path.isfile(candidate):
        return candidate

    candidate2 = os.path.join("/kaggle/input/aptos-2019-conv1-weights", filename)
    if os.path.isfile(candidate2):
        return candidate2

    for base_dir in ["../input", "/kaggle/input", "/kaggle/data"]:
        if not os.path.isdir(base_dir):
            continue
        for root, _, files in os.walk(base_dir):
            if filename in files:
                return os.path.join(root, filename)
    return None


def get_model(name, input_shape, nb_out):
    models = {
        "resnet50": get_resnet50,
        "conv1": get_conv1,
    }

    if name not in models:
        raise ValueError(f"No model named '{name}'")

    model = models[name](input_shape, nb_out)

    expected_weights_path = weights_path_template.format(name)
    weights_filename = os.path.basename(expected_weights_path)

    weights_path = (
        expected_weights_path
        if os.path.isfile(expected_weights_path)
        else _find_weight_file(weights_filename)
    )

    if weights_path and os.path.isfile(weights_path):
        model.load_weights(weights_path)
        print(f"loaded model from {weights_path}")
    else:
        print(
            f"weights not found for {weights_filename}; using randomly initialized weights"
        )

    return model




## === cell 12
"""
trains a ResNet50-based model
"""


def train_resnet50(model, train_generator, val_generator, weights_path, log_path):
    for i in range(len(model.layers)):
        model.layers[i].trainable = False
    for i in range(-5, 0):
        model.layers[i].trainable = True

    metrics_list = ["accuracy"]
    optimizer = Adam(learning_rate=INITIAL_LR)

    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=metrics_list
    )

    mc = ModelCheckpoint(
        weights_path, monitor="val_loss", save_best_only=True, verbose=1
    )
    es = EarlyStopping(
        monitor="val_loss", mode="min", restore_best_weights=True, verbose=1
    )
    cl = CSVLogger(log_path)

    step_size_train = max(1, train_generator.n // train_generator.batch_size)
    step_size_val = max(1, val_generator.n // val_generator.batch_size)

    model.fit(
        train_generator,
        steps_per_epoch=step_size_train,
        validation_data=val_generator,
        validation_steps=step_size_val,
        epochs=NB_WARMUP_EPOCHS,
        callbacks=[mc, cl],
        verbose=1,
    )

    train_generator.reset()
    val_generator.reset()

    for i in range(len(model.layers)):
        model.layers[i].trainable = True

    optimizer = Adam(learning_rate=INITIAL_LR)
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=metrics_list
    )

    mc = ModelCheckpoint(
        weights_path, monitor="val_loss", save_best_only=True, verbose=1
    )
    es = EarlyStopping(
        monitor="val_loss", mode="min", restore_best_weights=True, verbose=1
    )
    cl = CSVLogger(log_path)

    step_size_train = max(1, train_generator.n // train_generator.batch_size)
    step_size_val = max(1, val_generator.n // val_generator.batch_size)

    model.fit(
        train_generator,
        steps_per_epoch=step_size_train,
        validation_data=val_generator,
        validation_steps=step_size_val,
        epochs=NB_WARMUP_EPOCHS,
        callbacks=[mc, cl],
        verbose=1,
    )




## === cell 13
"""
trains the simple CNN
"""


def train_conv1(model, train_generator, val_generator, weights_path, log_path):
    metrics_list = ["accuracy"]
    optimizer = Adam(learning_rate=INITIAL_LR)

    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=metrics_list
    )

    mc = ModelCheckpoint(
        weights_path, monitor="val_loss", save_best_only=True, verbose=1
    )
    es = EarlyStopping(
        monitor="val_loss", mode="min", restore_best_weights=True, verbose=1
    )
    cl = CSVLogger(log_path)

    step_size_train = max(1, train_generator.n // train_generator.batch_size)
    step_size_val = max(1, val_generator.n // val_generator.batch_size)

    model.fit(
        train_generator,
        steps_per_epoch=step_size_train,
        validation_data=val_generator,
        validation_steps=step_size_val,
        epochs=NB_EPOCHS,
        callbacks=[mc, cl],
        verbose=1,
    )




## === cell 14
def train_model(name, input_shape, nb_out, train_generator, val_generator):
    model = get_model(name, input_shape, nb_out)

    trainers = {"resnet50": train_resnet50, "conv1": train_conv1}

    if name not in trainers:
        raise ValueError(f"No model named '{name}'")

    trainers[name](
        model,
        train_generator,
        val_generator,
        weights_path_template.format(name),
        log_path_template.format(name),
    )




## === cell 15
if TRAINING:
    train_model(
        MODEL_NAME, (IMG_SIZE, IMG_SIZE, NB_CHANNELS), NB_CLASSES, train_gen, val_gen
    )



## === cell 16
model = get_model(MODEL_NAME, (IMG_SIZE, IMG_SIZE, NB_CHANNELS), NB_CLASSES)

test_gen.reset()

pred_steps = len(test_gen)
preds = model.predict(test_gen, steps=pred_steps, verbose=1)
predictions = np.argmax(preds, axis=1).astype(int)

predictions = predictions[: len(test)]

results = pd.DataFrame(
    {
        "id_code": test["id_code"].str.replace(".png", "", regex=False),
        "diagnosis": predictions,
    }
)

sample_sub_path = os.path.join(INPUT_DIR, "sample_submission.csv")
if os.path.isfile(sample_sub_path):
    sample_sub = pd.read_csv(sample_sub_path)
    if "id_code" in sample_sub.columns and len(sample_sub) == len(results):
        results = sample_sub[["id_code"]].merge(results, on="id_code", how="left")
        results["diagnosis"] = results["diagnosis"].fillna(0).astype(int)

results["diagnosis"] = results["diagnosis"].clip(0, 4).astype(int)
results.to_csv("submission.csv", index=False)
print(results.head(10))
print("Wrote submission.csv with shape:", results.shape)

## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/21619133.py in <cell line: 0>()
----> 1 model = get_model(MODEL_NAME, (IMG_SIZE, IMG_SIZE, NB_CHANNELS), NB_CLASSES)
      2 
      3 test_gen.reset()
      4 
      5 pred_steps = len(test_gen)

NameError: name 'MODEL_NAME' is not defined
