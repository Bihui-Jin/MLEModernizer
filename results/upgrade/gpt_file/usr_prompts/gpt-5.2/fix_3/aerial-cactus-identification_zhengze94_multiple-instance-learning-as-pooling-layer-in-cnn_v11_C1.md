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
Create a classifier to predict whether an image contains a cactus.

## Metric
Area under the ROC curve.

## Submission Format
For each ID in the test set, you must predict a probability for the `has_cactus` variable. The file should contain a header and have the following format:

```
id,has_cactus
000940378805c44108d287872b2f04ce.jpg,0.5
0017242f54ececa4512b4d7937d1e21e.jpg,0.5
001ee6d8564003107853118ab87df407.jpg,0.5
etc.
```

## Dataset
This dataset contains a large number of 32 x 32 thumbnail images containing aerial photos of a cactus. The file name of an image corresponds to its `id`.

- **train/** - the training set images
- **test/** - the test set images (you must predict the labels of these)
- **train.csv** - the training set labels, indicates whether the image has a cactus (`has_cactus = 1`)
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.7

# 3. Installed packages

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pillow==11.3.0
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
scipy==1.15.3
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
tqdm==4.67.1

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (56 lines)
            sample_submission.csv (3326 lines)
            sample_submission.csv.zip (67.3 kB)
            test.zip (3.5 MB)
            train.csv (14176 lines)
            train.csv.zip (285.6 kB)
            train.zip (15.0 MB)
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
            test/
                76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                ... and 3323 other files
                test/
            train/
                775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                ... and 14173 other files
                train/
        input/
            description.md (56 lines)
            sample_submission.csv (3326 lines)
            sample_submission.csv.zip (67.3 kB)
            test.zip (3.5 MB)
            train.csv (14176 lines)
            train.csv.zip (285.6 kB)
            train.zip (15.0 MB)
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
            test/
                76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                ... and 3323 other files
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
            train/
                775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                ... and 14173 other files
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
        working/
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
```

-> data/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> data/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> data/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> data/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> input/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> input/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> (stopped after 10 files for performance)

# 5. Target score

0.7965

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.91051) has done: 'I fix the TensorFlow/protobuf crash by pinning protobuf to the compatible pure-Python implementation before importing TensorFlow. I fix the image-loading logic so train/test arrays have consistent shapes (32×32×3) and deterministic ordering that matches the sample submission IDs, avoiding the current ragged `X_test` error. I update the custom `noisyand` layer to work with TF/Keras 2.18 shape objects (remove the deprecated `.value` access) so the model can build and train. Finally, I ensure the script writes a valid `submission.csv` with columns `id,has_cactus` using model `predict` (Keras doesn’t have `predict_proba/predict_classes`) and update plotting keys to the current Keras history names.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "cpp"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd

BASE_INPUT = "/kaggle/input"
DATASET_DIR_CANDIDATES = [
    os.path.join(BASE_INPUT, "aerial-cactus-identification"),
    os.path.join(
        BASE_INPUT, "aerial-cactus-identification", "aerial-cactus-identification"
    ),
    os.path.join(BASE_INPUT, "data", "aerial-cactus-identification"),
]
DATASET_DIR = next((p for p in DATASET_DIR_CANDIDATES if os.path.exists(p)), None)
if DATASET_DIR is None:
    DATASET_DIR = os.path.join("..", "input")

print("Using DATASET_DIR:", DATASET_DIR)
print(
    "Top-level /kaggle/input entries (if present):",
    os.listdir(BASE_INPUT) if os.path.exists(BASE_INPUT) else "N/A",
)



## === cell 1
import matplotlib.pyplot as plt
import glob
import os



## === cell 2
from tqdm import tqdm
import tensorflow as tf
import tensorflow.keras  # keep original import pattern
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Conv2D,
    Flatten,
    Dropout,
    MaxPooling2D,
    Activation,
    BatchNormalization,
    LeakyReLU,
    GlobalAveragePooling2D,
)
from tensorflow.keras.optimizers import Adam, SGD
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from PIL import Image
import cv2

print("TensorFlow:", tf.__version__)



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_11/2863236412.py in <cell line: 0>()
      1 from tqdm import tqdm
----> 2 import tensorflow as tf
      3 import tensorflow.keras  # keep original import pattern
      4 from tensorflow.keras.preprocessing.image import ImageDataGenerator
      5 from tensorflow.keras.models import Sequential

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
import os
import pandas as pd
import numpy as np
import cv2

TRAIN_DIR_CANDIDATES = [
    os.path.join(DATASET_DIR, "train"),
    os.path.join(DATASET_DIR, "train", "train"),
    os.path.join("..", "input", "train", "train"),
]
TEST_DIR_CANDIDATES = [
    os.path.join(DATASET_DIR, "test"),
    os.path.join(DATASET_DIR, "test", "test"),
    os.path.join("..", "input", "test", "test"),
]
TRAIN_CSV_CANDIDATES = [
    os.path.join(DATASET_DIR, "train.csv"),
    os.path.join("..", "input", "train.csv"),
]
SAMPLE_SUB_CANDIDATES = [
    os.path.join(DATASET_DIR, "sample_submission.csv"),
    os.path.join("..", "input", "sample_submission.csv"),
]

TRAIN_DIR = next((p for p in TRAIN_DIR_CANDIDATES if os.path.exists(p)), None)
TEST_DIR = next((p for p in TEST_DIR_CANDIDATES if os.path.exists(p)), None)
TRAIN_CSV_PATH = next((p for p in TRAIN_CSV_CANDIDATES if os.path.exists(p)), None)
SAMPLE_SUB_PATH = next((p for p in SAMPLE_SUB_CANDIDATES if os.path.exists(p)), None)

if (
    TRAIN_DIR is None
    or TEST_DIR is None
    or TRAIN_CSV_PATH is None
    or SAMPLE_SUB_PATH is None
):
    raise FileNotFoundError(
        f"Could not locate required files.\n"
        f"TRAIN_DIR={TRAIN_DIR}\nTEST_DIR={TEST_DIR}\nTRAIN_CSV_PATH={TRAIN_CSV_PATH}\nSAMPLE_SUB_PATH={SAMPLE_SUB_PATH}"
    )

print("TRAIN_DIR:", TRAIN_DIR)
print("TEST_DIR:", TEST_DIR)
print("TRAIN_CSV_PATH:", TRAIN_CSV_PATH)
print("SAMPLE_SUB_PATH:", SAMPLE_SUB_PATH)


def read_img_rgb_32(path):
    """Read image, ensure RGB 32x32, float32 in [0,1]."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img.shape[0] != 32 or img.shape[1] != 32:
        img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)
    return img.astype(np.float32) / 255.0




## === cell 4
train_csv = pd.read_csv(TRAIN_CSV_PATH)

X_train = np.empty((len(train_csv), 32, 32, 3), dtype=np.float32)
Y_train = train_csv["has_cactus"].astype(np.int32).values

for i, img_id in enumerate(tqdm(train_csv["id"].values, desc="Loading train images")):
    X_train[i] = read_img_rgb_32(os.path.join(TRAIN_DIR, img_id))

sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
test_ids = sample_sub["id"].values

X_test = np.empty((len(test_ids), 32, 32, 3), dtype=np.float32)
for i, img_id in enumerate(tqdm(test_ids, desc="Loading test images")):
    X_test[i] = read_img_rgb_32(os.path.join(TEST_DIR, img_id))

print("Training data shape:", X_train.shape, "=>", Y_train.shape)
print("Test data shape:", X_test.shape)



## === cell 5
import numpy as np
from matplotlib import pyplot as plt

plt.rcParams["axes.grid"] = False



## === cell 6
fig, axes = plt.subplots(1, 5, figsize=(15, 4))
axes[0].imshow(X_train[0])
axes[0].set_title("Has cactus:" + str(Y_train[0]))
axes[1].imshow(X_train[1])
axes[1].set_title("Has cactus:" + str(Y_train[1]))
axes[2].imshow(X_train[2])
axes[2].set_title("Has cactus:" + str(Y_train[2]))
axes[3].imshow(X_train[1000])
axes[3].set_title("Has cactus:" + str(Y_train[1000]))
axes[4].imshow(X_train[1050])
axes[4].set_title("Has cactus:" + str(Y_train[1050]))
plt.show()



## === cell 7
from scipy.ndimage import gaussian_filter


def img_sharpen(img):
    blurred_f = gaussian_filter(img, 2)
    filter_blurred_f = gaussian_filter(blurred_f, 2)
    alpha = 15
    sharpened = blurred_f + alpha * (blurred_f - filter_blurred_f)
    return sharpened




## === cell 8
sharp_img_xtrain = [img_sharpen(im) for im in tqdm(X_train, desc="Sharpening train")]



## === cell 9
fig, axes = plt.subplots(1, 5, figsize=(15, 4))
axes[0].imshow(np.clip(sharp_img_xtrain[0], 0, 1))
axes[0].set_title("Has cactus:" + str(Y_train[0]))
axes[1].imshow(np.clip(sharp_img_xtrain[1], 0, 1))
axes[1].set_title("Has cactus:" + str(Y_train[1]))
axes[2].imshow(np.clip(sharp_img_xtrain[2], 0, 1))
axes[2].set_title("Has cactus:" + str(Y_train[2]))
axes[3].imshow(np.clip(sharp_img_xtrain[1000], 0, 1))
axes[3].set_title("Has cactus:" + str(Y_train[1000]))
axes[4].imshow(np.clip(sharp_img_xtrain[1050], 0, 1))
axes[4].set_title("Has cactus:" + str(Y_train[1050]))
plt.show()



## === cell 10
from sklearn.model_selection import train_test_split
from numpy import array

sharp_xtrain = array(sharp_img_xtrain, dtype=np.float32)

x_train, x_val, y_train, y_val = train_test_split(
    X_train, Y_train, test_size=0.2, random_state=42, stratify=Y_train
)



## === cell 11
import tensorflow as tf


class noisyand(tf.keras.layers.Layer):
    def __init__(self, num_classes, a=20, **kwargs):
        self.num_classes = num_classes
        self.a = max(1, a)
        super(noisyand, self).__init__(**kwargs)

    def build(self, input_shape):
        channels = int(input_shape[-1])
        self.b = self.add_weight(
            name="b",
            shape=(1, channels),
            initializer="uniform",
            trainable=True,
        )
        super(noisyand, self).build(input_shape)

    def call(self, x):
        mean = tf.reduce_mean(x, axis=[1, 2])
        num = tf.nn.sigmoid(self.a * (mean - self.b)) - tf.nn.sigmoid(-self.a * self.b)
        den = tf.nn.sigmoid(self.a * (1 - self.b)) - tf.nn.sigmoid(-self.a * self.b)
        return num / den

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[3])




## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_11/4267176039.py in <cell line: 0>()
----> 1 import tensorflow as tf
      2 
      3 
      4 class noisyand(tf.keras.layers.Layer):
      5     def __init__(self, num_classes, a=20, **kwargs):

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

## === cell 12
def define_model(input_shape=(32, 32, 3), num_classes=1):
    model = Sequential()
    model.add(
        Conv2D(
            64,
            kernel_size=(3, 3),
            activation="relu",
            padding="same",
            input_shape=input_shape,
        )
    )

    model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D())

    model.add(Conv2D(128, (3, 3), activation="relu"))
    model.add(MaxPooling2D())

    model.add(Conv2D(128, (3, 3), activation="relu"))
    model.add(Conv2D(128, (1, 1), activation="relu"))

    model.add(noisyand(num_classes + 1))
    model.add(Dense(num_classes, activation="sigmoid"))

    return model




## === cell 13
model = define_model()
model.summary()



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2566095353.py in <cell line: 0>()
----> 1 model = define_model()
      2 model.summary()
      3 

/tmp/ipykernel_11/1920032264.py in define_model(input_shape, num_classes)
      1 def define_model(input_shape=(32, 32, 3), num_classes=1):
----> 2     model = Sequential()
      3     model.add(
      4         Conv2D(
      5             64,

NameError: name 'Sequential' is not defined

## === cell 14
model.compile(
    loss=tensorflow.keras.losses.binary_crossentropy,
    optimizer=tensorflow.keras.optimizers.RMSprop(),
    metrics=["accuracy"],
)



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/836276833.py in <cell line: 0>()
----> 1 model.compile(
      2     loss=tensorflow.keras.losses.binary_crossentropy,
      3     optimizer=tensorflow.keras.optimizers.RMSprop(),
      4     metrics=["accuracy"],
      5 )

NameError: name 'model' is not defined

## === cell 15
epoch = 30
history = model.fit(
    x_train,
    y_train,
    batch_size=32,
    epochs=epoch,
    verbose=1,
    validation_data=(x_val, y_val),
)



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/929167923.py in <cell line: 0>()
      1 epoch = 30
----> 2 history = model.fit(
      3     x_train,
      4     y_train,
      5     batch_size=32,

NameError: name 'model' is not defined

## === cell 16
acc = history.history.get("accuracy", history.history.get("acc"))
epochs_ = range(0, epoch)
plt.plot(list(epochs_), acc, label="training accuracy")

acc_val = history.history.get("val_accuracy", history.history.get("val_acc"))
plt.scatter(list(epochs_), acc_val, label="validation accuracy")
plt.ylim([0.85, 1.0])
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Accuracy Plot of Model")
plt.legend()
plt.show()



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2686534850.py in <cell line: 0>()
----> 1 acc = history.history.get("accuracy", history.history.get("acc"))
      2 epochs_ = range(0, epoch)
      3 plt.plot(list(epochs_), acc, label="training accuracy")
      4 
      5 acc_val = history.history.get("val_accuracy", history.history.get("val_acc"))

NameError: name 'history' is not defined

## === cell 17
loss = history.history["loss"]
epochs_ = range(0, epoch)
plt.plot(list(epochs_), loss, label="training loss")

loss_val = history.history["val_loss"]
plt.scatter(list(epochs_), loss_val, label="validation loss")
plt.ylim([0, 0.5])
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss Plot of Model")
plt.legend()
plt.show()



## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3640430162.py in <cell line: 0>()
----> 1 loss = history.history["loss"]
      2 epochs_ = range(0, epoch)
      3 plt.plot(list(epochs_), loss, label="training loss")
      4 
      5 loss_val = history.history["val_loss"]

NameError: name 'history' is not defined

## === cell 18
submission_set = pd.read_csv(SAMPLE_SUB_PATH)
submission_set.head()



## === cell 19
X_test_sharp = np.clip(
    np.array(
        [img_sharpen(im) for im in tqdm(X_test, desc="Sharpening test")],
        dtype=np.float32,
    ),
    0.0,
    1.0,
)

preds = (
    model.predict(X_test_sharp, batch_size=256, verbose=0).ravel().astype(np.float32)
)
submission_set["has_cactus"] = preds

submission_path = "submission.csv"
submission_set.to_csv(submission_path, index=False)
print("Wrote:", submission_path)
submission_set.head()



## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1935718637.py in <cell line: 0>()
     11 
     12 preds = (
---> 13     model.predict(X_test_sharp, batch_size=256, verbose=0).ravel().astype(np.float32)
     14 )
     15 submission_set["has_cactus"] = preds

NameError: name 'model' is not defined

## === cell 20
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

y_val_proba = model.predict(x_val, verbose=0).ravel()

fpr, tpr, _ = roc_curve(y_val, y_val_proba)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(
    fpr, tpr, color="darkorange", lw=1.5, label="ROC curve (area = %0.4f)" % roc_auc
)
plt.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--")
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Validation ROC")
plt.legend(loc="lower right")
plt.show()

print("Validation AUC:", roc_auc)

## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3410913427.py in <cell line: 0>()
      2 from sklearn.metrics import roc_curve, auc
      3 
----> 4 y_val_proba = model.predict(x_val, verbose=0).ravel()
      5 
      6 fpr, tpr, _ = roc_curve(y_val, y_val_proba)

NameError: name 'model' is not defined
