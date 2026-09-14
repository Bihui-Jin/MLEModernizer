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
Given a dataset of images from digital pathology scans, predict if the center 32x32px region of a patch contains at least one pixel of tumor tissue. Tumor tissue in the outer region of the patch does not influence the label. 

## Metric
Area under the ROC curve.

## Submission Format
For each `id` in the test set, you must predict a probability that center 32x32px region of a patch contains at least one pixel of tumor tissue. The file should contain a header and have the following format:

```
id,label
0b2ea2a822ad23fdb1b5dd26653da899fbd2c0d5,0
95596b92e5066c5c52466c90b69ff089b39f2737,0
248e6738860e2ebcf6258cdc1f32f299e0c76914,0
etc.
```

## Dataset
Files are named with an image `id`. The `train_labels.csv` file provides the ground truth for the images in the `train` folder. You are predicting the labels for the images in the `test` folder.

# 2. Python version

3.13

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (63 lines)
            sample_submission.csv (45562 lines)
            sample_submission.csv.zip (1.1 MB)
            test.zip (1.1 GB)
            train.zip (4.2 GB)
            train_labels.csv (174465 lines)
            train_labels.csv.zip (4.2 MB)
            histopathologic-cancer-detection/
                description.md (63 lines)
                sample_submission.csv (45562 lines)
                ... and 5 other files
                histopathologic-cancer-detection/
                test/
                    7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                    c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                    ... and 45559 other files
                    test/
                train/
                    bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                    0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                    ... and 174462 other files
                    train/
            test/
                7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                ... and 45559 other files
                test/
            train/
                bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                ... and 174462 other files
                train/
        input/
            description.md (63 lines)
            sample_submission.csv (45562 lines)
            sample_submission.csv.zip (1.1 MB)
            test.zip (1.1 GB)
            train.zip (4.2 GB)
            train_labels.csv (174465 lines)
            train_labels.csv.zip (4.2 MB)
            histopathologic-cancer-detection/
                description.md (63 lines)
                sample_submission.csv (45562 lines)
                ... and 5 other files
                histopathologic-cancer-detection/
                test/
                    7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                    c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                    ... and 45559 other files
                    test/
                train/
                    bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                    0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                    ... and 174462 other files
                    train/
            test/
                7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                ... and 45559 other files
                test/
                    7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                    c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                    ... and 45559 other files
                    test/
            train/
                bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                ... and 174462 other files
                train/
                    bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                    0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                    ... and 174462 other files
                    train/
        working/
            histopathologic-cancer-detection/
                description.md (63 lines)
                sample_submission.csv (45562 lines)
                ... and 5 other files
                histopathologic-cancer-detection/
                test/
                    7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                    c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                    ... and 45559 other files
                    test/
                train/
                    bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                    0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                    ... and 174462 other files
                    train/
```

-> data/histopathologic-cancer-detection/sample_submission.csv has 45561 rows and 2 columns.
The columns are: id, label

-> data/histopathologic-cancer-detection/train_labels.csv has 174464 rows and 2 columns.
The columns are: id, label

-> data/sample_submission.csv has 45561 rows and 2 columns.
The columns are: id, label

-> data/train_labels.csv has 174464 rows and 2 columns.
The columns are: id, label

-> input/histopathologic-cancer-detection/sample_submission.csv has 45561 rows and 2 columns.
The columns are: id, label

-> input/histopathologic-cancer-detection/train_labels.csv has 174464 rows and 2 columns.
The columns are: id, label

-> (stopped after 10 files for performance)

# 5. Target score

0.753736212726282

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.66296) has done: 'I fix the Keras runtime/import issue that triggers the `MessageFactory.GetPrototype` error by forcing Keras to use the TensorFlow backend (compatible in Kaggle) before importing `keras`. I also remove the now-unsupported `workers/use_multiprocessing/max_queue_size` arguments from `fit()` and `predict()` (new Keras API), which currently prevents training/inference from running at all. Finally, I keep the model/training logic the same and ensure predictions are generated and written to a valid `submission.csv` with the required `id,label` columns.'
- What this solution (achieved 0.90752) has done: 'I fix the two runtime blockers: (1) the protobuf `MessageFactory.GetPrototype` crash by pinning a compatible pure-Python protobuf implementation before importing TensorFlow/Keras, and (2) the `AUC` metric shape mismatch by changing the metric to binary AUC while keeping the same 2-class softmax output/loss. These are execution/correctness fixes that should also improve AUC because the metric now be computed correctly during training/validation (previously it crashed). I keep your model architecture, preprocessing, data sampling, and training loop intact, and ensure the script always writes a valid `submission.csv` with `id,label`.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "cpp")
os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import random

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

print("Python:", os.sys.version)



## === cell 1
BASE = "/kaggle/input/histopathologic-cancer-detection"
train_images = f"{BASE}/train/"
test_images = f"{BASE}/test/"
train_csv = f"{BASE}/train_labels.csv"
sample_sub_csv = f"{BASE}/sample_submission.csv"

assert os.path.exists(train_images), f"Missing path: {train_images}"
assert os.path.exists(test_images), f"Missing path: {test_images}"
assert os.path.exists(train_csv), f"Missing path: {train_csv}"
assert os.path.exists(sample_sub_csv), f"Missing path: {sample_sub_csv}"



## === cell 2
test_df = pd.read_csv(sample_sub_csv)
print("Test Set Size:", test_df.shape)
test_df.head()



## === cell 3
train_df = pd.read_csv(train_csv)
train_df["label"] = train_df["label"].astype(int)

print("Train Set Size:", train_df.shape)
train_df.head()



## === cell 4
BATCH_SIZE = 64
TARGET_SIZE = (96, 96)




## === cell 5
def standardize_np(img_float_0_1: np.ndarray) -> np.ndarray:
    mean = img_float_0_1.mean(dtype=np.float32)
    var = img_float_0_1.var(dtype=np.float32)
    std = np.sqrt(var, dtype=np.float32)
    return (img_float_0_1 - mean) / (std + 1e-7)




## === cell 6
from PIL import Image
from sklearn.model_selection import train_test_split

train_split_df, val_split_df = train_test_split(
    train_df, test_size=0.10, random_state=SEED, stratify=train_df["label"]
)


def load_image(path, target_size=(96, 96)):
    with Image.open(path) as im:
        im = im.convert("RGB")
        if im.size != (target_size[1], target_size[0]):
            im = im.resize((target_size[1], target_size[0]), resample=Image.BILINEAR)
        arr = np.asarray(im, dtype=np.float32) * (1.0 / 255.0)
    arr = standardize_np(arr).astype(np.float32, copy=False)
    return arr


MAX_TRAIN = 30000
MAX_VAL = 6000

train_split_df_sub = train_split_df.sample(
    n=min(MAX_TRAIN, len(train_split_df)), random_state=SEED
).reset_index(drop=True)
val_split_df_sub = val_split_df.sample(
    n=min(MAX_VAL, len(val_split_df)), random_state=SEED
).reset_index(drop=True)

print("Train subset:", train_split_df_sub.shape, "Val subset:", val_split_df_sub.shape)



## === cell 7
try:
    import keras
    from keras import layers
    from keras.layers import (
        Layer,
        GlobalAveragePooling2D,
        GlobalMaxPooling2D,
        Dense,
        Conv2D,
    )
    from keras.initializers import lecun_normal
    from keras.utils import register_keras_serializable
except Exception as e:
    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
    import importlib

    for m in ["tensorflow", "keras"]:
        if m in list(importlib.sys.modules.keys()):
            importlib.sys.modules.pop(m, None)
    import keras
    from keras import layers
    from keras.layers import (
        Layer,
        GlobalAveragePooling2D,
        GlobalMaxPooling2D,
        Dense,
        Conv2D,
    )
    from keras.initializers import lecun_normal
    from keras.utils import register_keras_serializable


@register_keras_serializable(package="Custom")
class CBAM(Layer):
    def __init__(self, channels, reduction_ratio=16, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.reduction_ratio = reduction_ratio

        self.global_avg_pool = GlobalAveragePooling2D()
        self.global_max_pool = GlobalMaxPooling2D()
        self.fc1 = Dense(
            units=channels // reduction_ratio,
            activation="selu",
            kernel_initializer=lecun_normal(),
        )
        self.fc2 = Dense(
            units=channels,
            activation="sigmoid",
            kernel_initializer=lecun_normal(),
        )

        self.conv = Conv2D(
            filters=1,
            kernel_size=7,
            padding="same",
            activation="sigmoid",
            kernel_initializer=lecun_normal(),
        )

    def call(self, inputs, **kwargs):
        avg_pooled = self.global_avg_pool(inputs)
        max_pooled = self.global_max_pool(inputs)
        avg_fc = self.fc2(self.fc1(avg_pooled))
        max_fc = self.fc2(self.fc1(max_pooled))
        channel_attention = avg_fc + max_fc
        channel_attention = keras.ops.expand_dims(channel_attention, axis=1)
        channel_attention = keras.ops.expand_dims(channel_attention, axis=1)
        channel_refined = inputs * channel_attention

        avg_spatial = keras.ops.mean(channel_refined, axis=-1, keepdims=True)
        max_spatial = keras.ops.max(channel_refined, axis=-1, keepdims=True)
        spatial_attention = self.conv(
            keras.ops.concatenate([avg_spatial, max_spatial], axis=-1)
        )
        spatial_refined = channel_refined * spatial_attention
        return spatial_refined

    def get_config(self):
        config = super().get_config()
        config.update(
            {"channels": self.channels, "reduction_ratio": self.reduction_ratio}
        )
        return config


@register_keras_serializable(package="Custom")
class CustomAlphaDropout(Layer):
    def __init__(self, rate, **kwargs):
        super().__init__(**kwargs)
        self.rate = rate

    def call(self, inputs, training=None):
        if training is None:
            training = False
        if not training:
            return inputs
        keep_prob = 1.0 - self.rate
        rnd = keras.random.uniform(
            shape=keras.ops.shape(inputs), minval=0.0, maxval=1.0, seed=SEED
        )
        binary_tensor = keras.ops.floor(keep_prob + rnd)
        outputs = inputs * binary_tensor / keep_prob
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({"rate": self.rate})
        return config


def build_model(input_shape=(96, 96, 3)):
    inputs = keras.Input(shape=input_shape)

    x = layers.Conv2D(32, 3, padding="same", kernel_initializer=lecun_normal())(inputs)
    x = layers.Activation("selu")(x)
    x = CBAM(32)(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, 3, padding="same", kernel_initializer=lecun_normal())(x)
    x = layers.Activation("selu")(x)
    x = CBAM(64)(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(128, 3, padding="same", kernel_initializer=lecun_normal())(x)
    x = layers.Activation("selu")(x)
    x = CBAM(128)(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(256, 3, padding="same", kernel_initializer=lecun_normal())(x)
    x = layers.Activation("selu")(x)
    x = CBAM(256)(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="selu", kernel_initializer=lecun_normal())(x)
    x = CustomAlphaDropout(0.2)(x)
    outputs = layers.Dense(2, activation="softmax")(x)

    model = keras.Model(inputs, outputs)
    return model


cnn = build_model(input_shape=(96, 96, 3))
cnn.summary()



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_11/4261454019.py in <cell line: 0>()
      4 try:
----> 5     import keras
      6     from keras import layers

/usr/local/lib/python3.11/dist-packages/keras/__init__.py in <module>
      1 # DO NOT EDIT. Generated by api_gen.sh
----> 2 from keras.api import DTypePolicy
      3 from keras.api import FloatDTypePolicy

/usr/local/lib/python3.11/dist-packages/keras/api/__init__.py in <module>
      7 
----> 8 from keras.api import activations
      9 from keras.api import applications

/usr/local/lib/python3.11/dist-packages/keras/api/activations/__init__.py in <module>
      6 
----> 7 from keras.src.activations import deserialize
      8 from keras.src.activations import get

/usr/local/lib/python3.11/dist-packages/keras/src/__init__.py in <module>
----> 1 from keras.src import activations
      2 from keras.src import applications
      3 from keras.src import backend

/usr/local/lib/python3.11/dist-packages/keras/src/activations/__init__.py in <module>
      2 
----> 3 from keras.src.activations.activations import celu
      4 from keras.src.activations.activations import elu

/usr/local/lib/python3.11/dist-packages/keras/src/activations/activations.py in <module>
----> 1 from keras.src import backend
      2 from keras.src import ops
      3 from keras.src.api_export import keras_export

/usr/local/lib/python3.11/dist-packages/keras/src/backend/__init__.py in <module>
      9 from keras.src.api_export import keras_export
---> 10 from keras.src.backend.common.dtypes import result_type
     11 from keras.src.backend.common.keras_tensor import KerasTensor

/usr/local/lib/python3.11/dist-packages/keras/src/backend/common/__init__.py in <module>
      1 from keras.src.backend.common import backend_utils
----> 2 from keras.src.backend.common.dtypes import result_type
      3 from keras.src.backend.common.variables import AutocastScope

/usr/local/lib/python3.11/dist-packages/keras/src/backend/common/dtypes.py in <module>
      4 from keras.src.backend import config
----> 5 from keras.src.backend.common.variables import standardize_dtype
      6 

/usr/local/lib/python3.11/dist-packages/keras/src/backend/common/variables.py in <module>
     10 from keras.src.backend.common.stateless_scope import in_stateless_scope
---> 11 from keras.src.utils.module_utils import tensorflow as tf
     12 from keras.src.utils.naming import auto_name

/usr/local/lib/python3.11/dist-packages/keras/src/utils/__init__.py in <module>
----> 1 from keras.src.utils.audio_dataset_utils import audio_dataset_from_directory
      2 from keras.src.utils.dataset_utils import split_dataset
      3 from keras.src.utils.file_utils import get_file

/usr/local/lib/python3.11/dist-packages/keras/src/utils/audio_dataset_utils.py in <module>
      3 from keras.src.api_export import keras_export
----> 4 from keras.src.utils import dataset_utils
      5 from keras.src.utils.module_utils import tensorflow as tf

/usr/local/lib/python3.11/dist-packages/keras/src/utils/dataset_utils.py in <module>
      8 
----> 9 from keras.src import tree
     10 from keras.src.api_export import keras_export

/usr/local/lib/python3.11/dist-packages/keras/src/tree/__init__.py in <module>
----> 1 from keras.src.tree.tree_api import assert_same_paths
      2 from keras.src.tree.tree_api import assert_same_structure
      3 from keras.src.tree.tree_api import flatten

/usr/local/lib/python3.11/dist-packages/keras/src/tree/tree_api.py in <module>
      7 if optree.available:
----> 8     from keras.src.tree import optree_impl as tree_impl
      9 elif dmtree.available:

/usr/local/lib/python3.11/dist-packages/keras/src/tree/optree_impl.py in <module>
     12 if backend() == "tensorflow":
---> 13     from tensorflow.python.trackable.data_structures import ListWrapper
     14     from tensorflow.python.trackable.data_structures import _DictWrapper

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
/tmp/ipykernel_11/4261454019.py in <cell line: 0>()
     23         if m in list(importlib.sys.modules.keys()):
     24             importlib.sys.modules.pop(m, None)
---> 25     import keras
     26     from keras import layers
     27     from keras.layers import (

/usr/local/lib/python3.11/dist-packages/keras/__init__.py in <module>
      1 # DO NOT EDIT. Generated by api_gen.sh
----> 2 from keras.api import DTypePolicy
      3 from keras.api import FloatDTypePolicy
      4 from keras.api import Function
      5 from keras.api import Initializer

/usr/local/lib/python3.11/dist-packages/keras/api/__init__.py in <module>
      6 
      7 
----> 8 from keras.api import activations
      9 from keras.api import applications
     10 from keras.api import backend

/usr/local/lib/python3.11/dist-packages/keras/api/activations/__init__.py in <module>
      5 """
      6 
----> 7 from keras.src.activations import deserialize
      8 from keras.src.activations import get
      9 from keras.src.activations import serialize

/usr/local/lib/python3.11/dist-packages/keras/src/__init__.py in <module>
----> 1 from keras.src import activations
      2 from keras.src import applications
      3 from keras.src import backend
      4 from keras.src import constraints
      5 from keras.src import datasets

/usr/local/lib/python3.11/dist-packages/keras/src/activations/__init__.py in <module>
      1 import types
      2 
----> 3 from keras.src.activations.activations import celu
      4 from keras.src.activations.activations import elu
      5 from keras.src.activations.activations import exponential

/usr/local/lib/python3.11/dist-packages/keras/src/activations/activations.py in <module>
----> 1 from keras.src import backend
      2 from keras.src import ops
      3 from keras.src.api_export import keras_export
      4 
      5 

/usr/local/lib/python3.11/dist-packages/keras/src/backend/__init__.py in <module>
      8 
      9 from keras.src.api_export import keras_export
---> 10 from keras.src.backend.common.dtypes import result_type
     11 from keras.src.backend.common.keras_tensor import KerasTensor
     12 from keras.src.backend.common.keras_tensor import any_symbolic_tensors

/usr/local/lib/python3.11/dist-packages/keras/src/backend/common/__init__.py in <module>
      1 from keras.src.backend.common import backend_utils
----> 2 from keras.src.backend.common.dtypes import result_type
      3 from keras.src.backend.common.variables import AutocastScope
      4 from keras.src.backend.common.variables import Variable as KerasVariable
      5 from keras.src.backend.common.variables import get_autocast_scope

/usr/local/lib/python3.11/dist-packages/keras/src/backend/common/dtypes.py in <module>
      3 from keras.src.api_export import keras_export
      4 from keras.src.backend import config
----> 5 from keras.src.backend.common.variables import standardize_dtype
      6 
      7 BOOL_TYPES = ("bool",)

/usr/local/lib/python3.11/dist-packages/keras/src/backend/common/variables.py in <module>
     10 from keras.src.backend.common.stateless_scope import in_stateless_scope
     11 from keras.src.utils.module_utils import tensorflow as tf
---> 12 from keras.src.utils.naming import auto_name
     13 
     14 

/usr/local/lib/python3.11/dist-packages/keras/src/utils/__init__.py in <module>
----> 1 from keras.src.utils.audio_dataset_utils import audio_dataset_from_directory
      2 from keras.src.utils.dataset_utils import split_dataset
      3 from keras.src.utils.file_utils import get_file
      4 from keras.src.utils.image_dataset_utils import image_dataset_from_directory
      5 from keras.src.utils.image_utils import array_to_img

/usr/local/lib/python3.11/dist-packages/keras/src/utils/audio_dataset_utils.py in <module>
      2 
      3 from keras.src.api_export import keras_export
----> 4 from keras.src.utils import dataset_utils
      5 from keras.src.utils.module_utils import tensorflow as tf
      6 from keras.src.utils.module_utils import tensorflow_io as tfio

/usr/local/lib/python3.11/dist-packages/keras/src/utils/dataset_utils.py in <module>
      7 import numpy as np
      8 
----> 9 from keras.src import tree
     10 from keras.src.api_export import keras_export
     11 from keras.src.utils import io_utils

/usr/local/lib/python3.11/dist-packages/keras/src/tree/__init__.py in <module>
----> 1 from keras.src.tree.tree_api import assert_same_paths
      2 from keras.src.tree.tree_api import assert_same_structure
      3 from keras.src.tree.tree_api import flatten
      4 from keras.src.tree.tree_api import flatten_with_path
      5 from keras.src.tree.tree_api import is_nested

/usr/local/lib/python3.11/dist-packages/keras/src/tree/tree_api.py in <module>
      6 
      7 if optree.available:
----> 8     from keras.src.tree import optree_impl as tree_impl
      9 elif dmtree.available:
     10     from keras.src.tree import dmtree_impl as tree_impl

/usr/local/lib/python3.11/dist-packages/keras/src/tree/optree_impl.py in <module>
     11 # Register backend-specific node classes
     12 if backend() == "tensorflow":
---> 13     from tensorflow.python.trackable.data_structures import ListWrapper
     14     from tensorflow.python.trackable.data_structures import _DictWrapper
     15 

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
import math


class HCDSequence(keras.utils.Sequence):
    def __init__(
        self,
        df,
        img_dir,
        target_size=(96, 96),
        batch_size=64,
        shuffle=False,
        seed=42,
        with_labels=True,
    ):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.target_size = target_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.with_labels = with_labels

        self.ids = self.df["id"].astype(str).values
        self.labels = (
            self.df["label"].values.astype(np.int64, copy=False)
            if with_labels and "label" in self.df.columns
            else None
        )

        self.indexes = np.arange(len(self.ids), dtype=np.int64)
        self.rng = np.random.RandomState(self.seed)
        self.on_epoch_end()

    def __len__(self):
        return (len(self.indexes) + self.batch_size - 1) // self.batch_size

    def on_epoch_end(self):
        if self.shuffle:
            self.rng.shuffle(self.indexes)

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = min(len(self.indexes), start + self.batch_size)
        batch_idx = self.indexes[start:end]
        b = len(batch_idx)

        x = np.empty((b, self.target_size[0], self.target_size[1], 3), dtype=np.float32)
        if self.with_labels:
            y = np.empty((b,), dtype=np.int64)

        for i, j in enumerate(batch_idx):
            img_id = self.ids[j]
            x[i] = load_image(
                os.path.join(self.img_dir, f"{img_id}.tif"),
                target_size=self.target_size,
            )
            if self.with_labels:
                y[i] = self.labels[j]

        return (x, y) if self.with_labels else x


train_seq = HCDSequence(
    train_split_df_sub,
    train_images,
    target_size=TARGET_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED,
    with_labels=True,
)
val_seq = HCDSequence(
    val_split_df_sub,
    train_images,
    target_size=TARGET_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
    seed=SEED,
    with_labels=True,
)

binary_auc = keras.metrics.AUC(name="auc")


def auc_on_positive_class(y_true, y_pred):
    return binary_auc(y_true, y_pred[:, 1])


cnn.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=[auc_on_positive_class],
)

EPOCHS = 2  # unchanged

history = cnn.fit(
    train_seq,
    validation_data=val_seq,
    epochs=EPOCHS,
    verbose=1,
)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1491366698.py in <cell line: 0>()
      2 
      3 
----> 4 class HCDSequence(keras.utils.Sequence):
      5     def __init__(
      6         self,

NameError: name 'keras' is not defined

## === cell 9
test_ids = test_df["id"].astype(str).values

test_only_df = pd.DataFrame({"id": test_ids})
test_seq = HCDSequence(
    test_only_df,
    test_images,
    target_size=TARGET_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
    seed=SEED,
    with_labels=False,
)

test_preds = cnn.predict(
    test_seq,
    verbose=1,
)

print(
    "test_preds shape:",
    test_preds.shape,
    "min/max:",
    float(test_preds.min()),
    float(test_preds.max()),
)



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1003619772.py in <cell line: 0>()
      2 
      3 test_only_df = pd.DataFrame({"id": test_ids})
----> 4 test_seq = HCDSequence(
      5     test_only_df,
      6     test_images,

NameError: name 'HCDSequence' is not defined

## === cell 10
tumor_probs = test_preds[:, 1].astype(np.float32)



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1115617278.py in <cell line: 0>()
----> 1 tumor_probs = test_preds[:, 1].astype(np.float32)
      2 

NameError: name 'test_preds' is not defined

## === cell 11
submission = pd.DataFrame({"id": test_ids, "label": tumor_probs})
submission.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3230850356.py in <cell line: 0>()
----> 1 submission = pd.DataFrame({"id": test_ids, "label": tumor_probs})
      2 submission.to_csv("submission.csv", index=False)
      3 
      4 print("Wrote submission.csv with shape:", submission.shape)
      5 print(submission.head())

NameError: name 'tumor_probs' is not defined

## === cell 12
frequency_distribution = (submission["label"].round(3).describe()).to_frame(
    name="label_stats"
)
print(frequency_distribution)

## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/448144222.py in <cell line: 0>()
----> 1 frequency_distribution = (submission["label"].round(3).describe()).to_frame(
      2     name="label_stats"
      3 )
      4 print(frequency_distribution)

NameError: name 'submission' is not defined
