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
Detect apple diseases from images.

## Metric
Mean F1-Score

## Submission Format
labels should be a space-delimited list.

The file should contain a header and have the following format:

```
image, labels
85f8cb619c66b863.jpg,healthy
ad8770db05586b59.jpg,healthy
c7b03e718489f3ca.jpg,healthy
```

## Dataset
**train.csv** - the training set metadata.

- `image` - the image ID.
- `labels` - the target classes, a space delimited list of all diseases found in the image. Unhealthy leaves with too many diseases to classify visually will have the `complex` class, and may also have a subset of the diseases identified.

**sample_submission.csv** - A sample submission file in the correct format.

- `image`
- `labels`

**train_images** - The training set images.

**test_images** - The test set images. This competition has a hidden test set: only three images are provided here as samples while the remaining 5,000 images will be available to your notebook once it is submitted.

# 2. Python version

3.10

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
        input/
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
        working/
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
```

-> data/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> data/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> (stopped after 10 files for performance)

# 5. Target score

0.8192243767313039

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
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import random
import numpy as np
import pandas as pd
import tensorflow as tf

print("TF version:", tf.__version__)

SEED = 1337
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_11/3374505029.py in <cell line: 0>()
     10 import numpy as np
     11 import pandas as pd
---> 12 import tensorflow as tf
     13 
     14 print("TF version:", tf.__version__)

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
output_dir = "./"
test_dir = "../input/plant-pathology-2021-fgvc8/test_images/"

efficientB7 = "../input/efficientb7/effb7"
efficientB7_weights = "../input/conve01/eff7-e12/epoch-12"

resnet50 = "../input/resnet50/Model-Resnet"
resnet50_weights = "../input/resnet50weights/last_epoch-20"

inceptionv3 = "../input/inceptionv3/Model-InceptionV3"
inceptionv3_weights = "../input/inceptionv3-weights/epoch-12"

image_dims = (300, 300, 3)

data_set = pd.read_csv("../input/plant-pathology-2021-fgvc8/train.csv")

dataset_labels = [
    "complex",
    "frog_eye_leaf_spot",
    "healthy",
    "powdery_mildew",
    "rust",
    "scab",
]
print("Num classes:", len(dataset_labels), dataset_labels)

assert os.path.isdir(test_dir), f"test_dir not found: {test_dir}"



## === cell 2
from tensorflow import concat
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import (
    Dense,
    BatchNormalization,
    Dropout,
    GlobalMaxPool2D,
    Conv2D,
    InputLayer,
)


def _is_savedmodel_dir(path: str) -> bool:
    return os.path.isdir(path) and (
        os.path.exists(os.path.join(path, "saved_model.pb"))
        or os.path.exists(os.path.join(path, "saved_model.pbtxt"))
    )


def load_savedmodel_as_layer(savedmodel_dir: str):
    """
    Robust loader:
    - If directory is a SavedModel -> use TFSMLayer
    - Else try tf.keras.models.load_model
    """
    if not os.path.exists(savedmodel_dir):
        raise FileNotFoundError(f"Backbone path does not exist: {savedmodel_dir}")

    if _is_savedmodel_dir(savedmodel_dir):
        for endpoint in ("serving_default", "call"):
            try:
                layer = tf.keras.layers.TFSMLayer(
                    savedmodel_dir, call_endpoint=endpoint
                )
                return layer
            except Exception:
                continue
        return tf.keras.layers.TFSMLayer(
            savedmodel_dir, call_endpoint="serving_default"
        )

    try:
        loaded = tf.keras.models.load_model(savedmodel_dir, compile=False)
        return loaded
    except Exception as e:
        raise OSError(
            f"Could not load backbone at {savedmodel_dir} as SavedModel or Keras model. Error: {e}"
        ) from e


def get_efficientnetb7_backbone_layer():
    return tf.keras.applications.EfficientNetB7(
        include_top=False,
        weights="imagenet",
        input_shape=image_dims,
    )


class MultiLabel(Model):
    def __init__(self):
        super().__init__()
        self.model_backbone = load_savedmodel_as_layer(resnet50)
        self.model = Sequential()
        self.model.add(InputLayer(input_shape=image_dims))
        self.model.add(self.model_backbone)
        self.model.add(Conv2D(filters=1024, kernel_size=(1, 1), padding="same"))
        self.model.add(BatchNormalization(momentum=0.7))
        self.model.add(Dropout(0.2))
        self.model.add(Conv2D(filters=256, kernel_size=(1, 1), padding="same"))
        self.model.add(BatchNormalization(momentum=0.7))
        self.model.add(Dropout(0.1))
        self.model.add(Conv2D(filters=128, kernel_size=(1, 1), padding="same"))
        self.model.add(GlobalMaxPool2D())
        self.model.add(Dense(units=6, activation="sigmoid"))

    def call(self, predict_input):
        return self.model(predict_input)

    def create_model(self):
        return self.model


class MultiLabel_2(Model):
    def __init__(self):
        super().__init__()
        self.model_backbone = load_savedmodel_as_layer(inceptionv3)
        self.model = Sequential()
        self.model.add(InputLayer(input_shape=image_dims))
        self.model.add(self.model_backbone)
        self.model.add(Conv2D(filters=1024, kernel_size=(1, 1), padding="same"))
        self.model.add(BatchNormalization(momentum=0.7))
        self.model.add(Dropout(0.2))
        self.model.add(Conv2D(filters=1024, kernel_size=(1, 1), padding="same"))
        self.model.add(Conv2D(filters=2400, kernel_size=(1, 1), padding="same"))

        self.model_pred1_2 = Conv2D(filters=256, kernel_size=(1, 1), padding="same")
        self.model_pred1_3 = BatchNormalization()
        self.model_pred1_4 = Conv2D(filters=128, kernel_size=(1, 1), padding="same")
        self.model_pred1_5 = GlobalMaxPool2D()
        self.model_pred1_6 = Dense(units=1, activation="sigmoid")

        self.model_pred2_2 = Conv2D(filters=256, kernel_size=(1, 1), padding="same")
        self.model_pred2_3 = BatchNormalization()
        self.model_pred2_4 = Conv2D(filters=128, kernel_size=(1, 1), padding="same")
        self.model_pred2_5 = GlobalMaxPool2D()
        self.model_pred2_6 = Dense(units=1, activation="sigmoid")

        self.model_pred3_2 = Conv2D(filters=256, kernel_size=(1, 1), padding="same")
        self.model_pred3_3 = BatchNormalization()
        self.model_pred3_4 = Conv2D(filters=128, kernel_size=(1, 1), padding="same")
        self.model_pred3_5 = GlobalMaxPool2D()
        self.model_pred3_6 = Dense(units=1, activation="sigmoid")

        self.model_pred4_2 = Conv2D(filters=256, kernel_size=(1, 1), padding="same")
        self.model_pred4_3 = BatchNormalization()
        self.model_pred4_4 = Conv2D(filters=128, kernel_size=(1, 1), padding="same")
        self.model_pred4_5 = GlobalMaxPool2D()
        self.model_pred4_6 = Dense(units=1, activation="sigmoid")

        self.model_pred5_2 = Conv2D(filters=256, kernel_size=(1, 1), padding="same")
        self.model_pred5_3 = BatchNormalization()
        self.model_pred5_4 = Conv2D(filters=128, kernel_size=(1, 1), padding="same")
        self.model_pred5_5 = GlobalMaxPool2D()
        self.model_pred5_6 = Dense(units=1, activation="sigmoid")

        self.model_pred6_2 = Conv2D(filters=256, kernel_size=(1, 1), padding="same")
        self.model_pred6_3 = BatchNormalization()
        self.model_pred6_4 = Conv2D(filters=128, kernel_size=(1, 1), padding="same")
        self.model_pred6_5 = GlobalMaxPool2D()
        self.model_pred6_6 = Dense(units=1, activation="sigmoid")

    def call(self, predict_input):
        predict_output = self.model(predict_input)
        pred_1 = predict_output[:, :, :, :400]
        pred_2 = predict_output[:, :, :, 400:800]
        pred_3 = predict_output[:, :, :, 800:1200]
        pred_4 = predict_output[:, :, :, 1200:1600]
        pred_5 = predict_output[:, :, :, 1600:2000]
        pred_6 = predict_output[:, :, :, 2000:2400]

        pred_1 = self.model_pred1_2(pred_1)
        pred_1 = self.model_pred1_3(pred_1)
        pred_1 = self.model_pred1_4(pred_1)
        pred_1 = self.model_pred1_5(pred_1)
        pred_1 = self.model_pred1_6(pred_1)

        pred_2 = self.model_pred2_2(pred_2)
        pred_2 = self.model_pred2_3(pred_2)
        pred_2 = self.model_pred2_4(pred_2)
        pred_2 = self.model_pred2_5(pred_2)
        pred_2 = self.model_pred2_6(pred_2)

        pred_3 = self.model_pred3_2(pred_3)
        pred_3 = self.model_pred3_3(pred_3)
        pred_3 = self.model_pred3_4(pred_3)
        pred_3 = self.model_pred3_5(pred_3)
        pred_3 = self.model_pred3_6(pred_3)

        pred_4 = self.model_pred4_2(pred_4)
        pred_4 = self.model_pred4_3(pred_4)
        pred_4 = self.model_pred4_4(pred_4)
        pred_4 = self.model_pred4_5(pred_4)
        pred_4 = self.model_pred4_6(pred_4)

        pred_5 = self.model_pred5_2(pred_5)
        pred_5 = self.model_pred5_3(pred_5)
        pred_5 = self.model_pred5_4(pred_5)
        pred_5 = self.model_pred5_5(pred_5)
        pred_5 = self.model_pred5_6(pred_5)

        pred_6 = self.model_pred6_2(pred_6)
        pred_6 = self.model_pred6_3(pred_6)
        pred_6 = self.model_pred6_4(pred_6)
        pred_6 = self.model_pred6_5(pred_6)
        pred_6 = self.model_pred6_6(pred_6)

        return concat([pred_1, pred_2, pred_3, pred_4, pred_5, pred_6], axis=1)

    def create_model(self):
        return self.model


class MultiLabel_3(Model):
    def __init__(self, backbone_layer=None):
        super().__init__()
        if backbone_layer is None:
            if os.path.exists(efficientB7):
                self.model_backbone = load_savedmodel_as_layer(efficientB7)
            else:
                print(
                    f"WARNING: backbone path not found: {efficientB7} -> using tf.keras.applications.EfficientNetB7 imagenet weights"
                )
                self.model_backbone = get_efficientnetb7_backbone_layer()
        else:
            self.model_backbone = backbone_layer

        self.model = Sequential()
        self.model.add(InputLayer(input_shape=image_dims))
        self.model.add(self.model_backbone)
        self.model.add(Conv2D(filters=1024, kernel_size=(1, 1), padding="same"))
        self.model.add(BatchNormalization(momentum=0.7))
        self.model.add(Dropout(0.2))
        self.model.add(Conv2D(filters=1024, kernel_size=(1, 1), padding="same"))
        self.model.add(Conv2D(filters=2400, kernel_size=(1, 1), padding="same"))

        self.model_pred1_2 = Conv2D(filters=1024, kernel_size=(1, 1), padding="same")
        self.model_pred1_3 = BatchNormalization()
        self.model_pred1_4 = Conv2D(filters=512, kernel_size=(1, 1), padding="same")
        self.model_pred1_5 = GlobalMaxPool2D()
        self.model_pred1_6 = Dense(units=1, activation="sigmoid")

        self.model_pred2_2 = Conv2D(filters=1024, kernel_size=(1, 1), padding="same")
        self.model_pred2_3 = BatchNormalization()
        self.model_pred2_4 = Conv2D(filters=512, kernel_size=(1, 1), padding="same")
        self.model_pred2_5 = GlobalMaxPool2D()
        self.model_pred2_6 = Dense(units=1, activation="sigmoid")

        self.model_pred3_2 = Conv2D(filters=1024, kernel_size=(1, 1), padding="same")
        self.model_pred3_3 = BatchNormalization()
        self.model_pred3_4 = Conv2D(filters=512, kernel_size=(1, 1), padding="same")
        self.model_pred3_5 = GlobalMaxPool2D()
        self.model_pred3_6 = Dense(units=1, activation="sigmoid")

        self.model_pred4_2 = Conv2D(filters=1024, kernel_size=(1, 1), padding="same")
        self.model_pred4_3 = BatchNormalization()
        self.model_pred4_4 = Conv2D(filters=512, kernel_size=(1, 1), padding="same")
        self.model_pred4_5 = GlobalMaxPool2D()
        self.model_pred4_6 = Dense(units=1, activation="sigmoid")

        self.model_pred5_2 = Conv2D(filters=1024, kernel_size=(1, 1), padding="same")
        self.model_pred5_3 = BatchNormalization()
        self.model_pred5_4 = Conv2D(filters=512, kernel_size=(1, 1), padding="same")
        self.model_pred5_5 = GlobalMaxPool2D()
        self.model_pred5_6 = Dense(units=1, activation="sigmoid")

        self.model_pred6_2 = Conv2D(filters=1024, kernel_size=(1, 1), padding="same")
        self.model_pred6_3 = BatchNormalization()
        self.model_pred6_4 = Conv2D(filters=512, kernel_size=(1, 1), padding="same")
        self.model_pred6_5 = GlobalMaxPool2D()
        self.model_pred6_6 = Dense(units=1, activation="sigmoid")

    def call(self, predict_input):
        predict_output = self.model(predict_input)
        pred_1 = predict_output[:, :, :, :400]
        pred_2 = predict_output[:, :, :, 400:800]
        pred_3 = predict_output[:, :, :, 800:1200]
        pred_4 = predict_output[:, :, :, 1200:1600]
        pred_5 = predict_output[:, :, :, 1600:2000]
        pred_6 = predict_output[:, :, :, 2000:2400]

        pred_1 = self.model_pred1_2(pred_1)
        pred_1 = self.model_pred1_3(pred_1)
        pred_1 = self.model_pred1_4(pred_1)
        pred_1 = self.model_pred1_5(pred_1)
        pred_1 = self.model_pred1_6(pred_1)

        pred_2 = self.model_pred2_2(pred_2)
        pred_2 = self.model_pred2_3(pred_2)
        pred_2 = self.model_pred2_4(pred_2)
        pred_2 = self.model_pred2_5(pred_2)
        pred_2 = self.model_pred2_6(pred_2)

        pred_3 = self.model_pred3_2(pred_3)
        pred_3 = self.model_pred3_3(pred_3)
        pred_3 = self.model_pred3_4(pred_3)
        pred_3 = self.model_pred3_5(pred_3)
        pred_3 = self.model_pred3_6(pred_3)

        pred_4 = self.model_pred4_2(pred_4)
        pred_4 = self.model_pred4_3(pred_4)
        pred_4 = self.model_pred4_4(pred_4)
        pred_4 = self.model_pred4_5(pred_4)
        pred_4 = self.model_pred4_6(pred_4)

        pred_5 = self.model_pred5_2(pred_5)
        pred_5 = self.model_pred5_3(pred_5)
        pred_5 = self.model_pred5_4(pred_5)
        pred_5 = self.model_pred5_5(pred_5)
        pred_5 = self.model_pred5_6(pred_5)

        pred_6 = self.model_pred6_2(pred_6)
        pred_6 = self.model_pred6_3(pred_6)
        pred_6 = self.model_pred6_4(pred_6)
        pred_6 = self.model_pred6_5(pred_6)
        pred_6 = self.model_pred6_6(pred_6)

        return concat([pred_1, pred_2, pred_3, pred_4, pred_5, pred_6], axis=1)

    def create_model(self):
        return self.model




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_11/4212067997.py in <cell line: 0>()
----> 1 from tensorflow import concat
      2 from tensorflow.keras import Sequential, Model
      3 from tensorflow.keras.layers import (
      4     Dense,
      5     BatchNormalization,

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

## === cell 3
if __name__ == "__main__":
    model = MultiLabel_3()
    model.build(input_shape=[None, image_dims[0], image_dims[1], image_dims[2]])

    if os.path.exists(efficientB7_weights):
        model.load_weights(efficientB7_weights)
        print("Loaded weights:", efficientB7_weights)
    else:
        print(
            f"WARNING: Weights file not found: {efficientB7_weights} -> using backbone default weights and random head weights"
        )

    images_path_list = sorted(list(os.listdir(test_dir)))

    AUTO = tf.data.AUTOTUNE

    def _load_and_preprocess(name):
        path = tf.strings.join([test_dir, name])
        img_bytes = tf.io.read_file(path)
        img = tf.io.decode_jpeg(img_bytes, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)  # [0,1]
        img.set_shape([None, None, 3])
        img = tf.image.resize(img, [image_dims[0], image_dims[1]])
        img = img * 255.0  # keep original behavior
        img.set_shape([image_dims[0], image_dims[1], 3])
        return name, img

    BATCH_SIZE = 32

    ds = tf.data.Dataset.from_tensor_slices(tf.constant(images_path_list))
    ds = ds.map(_load_and_preprocess, num_parallel_calls=AUTO)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTO)

    @tf.function(reduce_retracing=True)
    def _predict_batch(batch_imgs):
        return model(batch_imgs, training=False)

    classes = dataset_labels
    n_classes = len(classes)
    healthy_idx = classes.index("healthy") if "healthy" in classes else -1

    out_names = []
    out_labels = []

    for batch_names, batch_imgs in ds:
        preds = _predict_batch(batch_imgs)  # (B, 6)
        preds_np = preds.numpy()

        above = preds_np > 0.5

        for i in range(preds_np.shape[0]):
            name = batch_names[i].numpy().decode("utf-8")
            idxs = np.flatnonzero(above[i])

            if idxs.size == 0:
                if healthy_idx >= 0:
                    label_str = "healthy"
                else:
                    max_i = int(np.argmax(preds_np[i]))
                    max_i = max(0, min(max_i, n_classes - 1))
                    label_str = str(classes[max_i])
            else:
                label_str = " ".join(str(classes[j]) for j in idxs.tolist())

            out_names.append(name)
            out_labels.append(label_str)

    csv_pd = pd.DataFrame({"image": out_names, "labels": out_labels})
    csv_path = os.path.join(output_dir, "submission.csv")
    csv_pd.to_csv(csv_path, index=False)
    print("Wrote:", csv_path, "rows:", len(csv_pd))

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/65537021.py in <cell line: 0>()
      1 if __name__ == "__main__":
----> 2     model = MultiLabel_3()
      3     model.build(input_shape=[None, image_dims[0], image_dims[1], image_dims[2]])
      4 
      5     if os.path.exists(efficientB7_weights):

NameError: name 'MultiLabel_3' is not defined
