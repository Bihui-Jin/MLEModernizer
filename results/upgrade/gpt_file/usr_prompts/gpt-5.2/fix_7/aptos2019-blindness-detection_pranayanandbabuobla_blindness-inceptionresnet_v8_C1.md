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

3.13

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

0.3338402537284629

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.09079) has done: 'The timeout is dominated by Python-side image decoding/augmentation via `tf.numpy_function` + OpenCV, plus extra overhead from forcing pure-Python protobuf and unbounded tf.data parallelism. I keep the exact same model, preprocessing, augmentation semantics, and training loop, but make the input pipeline faster by (1) switching to a pure-TensorFlow decode/augment path (so it can run in graph and parallelize efficiently), (2) adding deterministic, bounded parallelism and prefetching, and (3) removing the protobuf pure-Python fallback that slows TF startup and graph execution. These changes preserve the algorithm and outputs up to negligible floating-point differences while significantly reducing per-step input overhead.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "cpp")

import random
import warnings

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.inception_resnet_v2 import (
    InceptionResNetV2,
    preprocess_input,
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

warnings.filterwarnings("ignore")




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_11/1269094182.py in <cell line: 0>()
     13 import numpy as np
     14 import pandas as pd
---> 15 import tensorflow as tf
     16 from sklearn.model_selection import train_test_split
     17 from tensorflow.keras.applications.inception_resnet_v2 import (

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
class SimpleEpochLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        msg = f"Epoch {epoch + 1}: " + ", ".join(
            [
                f"{k}={v:.4f}"
                for k, v in logs.items()
                if isinstance(v, (int, float, np.floating))
            ]
        )
        print(msg)




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1290240819.py in <cell line: 0>()
----> 1 class SimpleEpochLogger(tf.keras.callbacks.Callback):
      2     def on_epoch_end(self, epoch, logs=None):
      3         logs = logs or {}
      4         msg = f"Epoch {epoch + 1}: " + ", ".join(
      5             [

NameError: name 'tf' is not defined

## === cell 2
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

try:
    _CPU = os.cpu_count() or 4
    tf.data.experimental.threading.private_threadpool_size = min(16, max(4, _CPU))
except Exception:
    pass



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2209801417.py in <cell line: 0>()
      1 SEED = 42
      2 np.random.seed(SEED)
----> 3 tf.random.set_seed(SEED)
      4 random.seed(SEED)
      5 

NameError: name 'tf' is not defined

## === cell 3
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    print("GPUs available:")
    for gpu in gpus:
        print(gpu)
else:
    print("No GPU available. Using CPU.")



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2586442577.py in <cell line: 0>()
----> 1 gpus = tf.config.list_physical_devices("GPU")
      2 if gpus:
      3     print("GPUs available:")
      4     for gpu in gpus:
      5         print(gpu)

NameError: name 'tf' is not defined

## === cell 4
TRAIN_IMG_DIR = "../input/aptos2019-blindness-detection/train_images"
TEST_IMG_DIR = "../input/aptos2019-blindness-detection/test_images"



## === cell 5
train_df = pd.read_csv("../input/aptos2019-blindness-detection/train.csv")
test_df = pd.read_csv("../input/aptos2019-blindness-detection/test.csv")
sample_sub = pd.read_csv("../input/aptos2019-blindness-detection/sample_submission.csv")




## === cell 6
def get_image_path(id_code, is_train=True):
    ext = ".png"
    if is_train:
        return os.path.join(TRAIN_IMG_DIR, id_code + ext)
    else:
        return os.path.join(TEST_IMG_DIR, id_code + ext)




## === cell 7
train_df["filepath"] = TRAIN_IMG_DIR + "/" + train_df["id_code"].astype(str) + ".png"
test_df["filepath"] = TEST_IMG_DIR + "/" + test_df["id_code"].astype(str) + ".png"



## === cell 8
IMG_SIZE = 299



## === cell 9
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_transform = ImageDataGenerator(
    horizontal_flip=True,
    brightness_range=(0.8, 1.2),
    rotation_range=180,
    shear_range=20,
    zoom_range=(0.8, 1.2),
    width_shift_range=0.2,
    height_shift_range=0.2,
    fill_mode="reflect",
)



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_11/656049729.py in <cell line: 0>()
----> 1 from tensorflow.keras.preprocessing.image import ImageDataGenerator
      2 
      3 train_transform = ImageDataGenerator(
      4     horizontal_flip=True,
      5     brightness_range=(0.8, 1.2),

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

## === cell 10
valid_transform = ImageDataGenerator()




## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4159910136.py in <cell line: 0>()
----> 1 valid_transform = ImageDataGenerator()
      2 
      3 

NameError: name 'ImageDataGenerator' is not defined

## === cell 11
def load_and_preprocess_image(path, transform=None):
    image = cv2.imread(path)
    if image is None:
        raise ValueError(f"Image not found at path: {path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

    if transform is not None:
        image = transform.random_transform(image)
        image = transform.standardize(image)

    image = image.astype(np.float32)
    image = preprocess_input(image)  # correct preprocessing for InceptionResNetV2
    return image




## === cell 12
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(
        self,
        df,
        batch_size=32,
        transform=None,
        is_train=True,
        num_classes=5,
        shuffle=True,
    ):
        self.df = df.reset_index(drop=True)
        self.batch_size = int(batch_size)
        self.transform = transform
        self.is_train = bool(is_train)
        self.num_classes = int(num_classes)
        self.shuffle = bool(shuffle)

        self.filepaths = self.df["filepath"].to_numpy(dtype=object)
        if self.is_train:
            y_int = self.df["diagnosis"].to_numpy(dtype=np.int32)
            self.labels_oh = tf.keras.utils.to_categorical(
                y_int, num_classes=self.num_classes
            ).astype(np.float32)
        else:
            self.labels_oh = None

        self.indexes = np.arange(len(self.df), dtype=np.int32)
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.indexes) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        batch_idx = self.indexes[
            index * self.batch_size : (index + 1) * self.batch_size
        ]
        bs = batch_idx.shape[0]

        images = np.empty((bs, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)

        if self.is_train:
            labels = self.labels_oh[batch_idx]
            for i, j in enumerate(batch_idx):
                images[i] = load_and_preprocess_image(
                    self.filepaths[j], transform=self.transform
                )
            return images, labels
        else:
            for i, j in enumerate(batch_idx):
                images[i] = load_and_preprocess_image(
                    self.filepaths[j], transform=self.transform
                )
            return images




## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2203349669.py in <cell line: 0>()
----> 1 class DataGenerator(tf.keras.utils.Sequence):
      2     def __init__(
      3         self,
      4         df,
      5         batch_size=32,

NameError: name 'tf' is not defined

## === cell 13
train_df_split, valid_df_split = train_test_split(
    train_df, test_size=0.2, random_state=SEED, stratify=train_df["diagnosis"]
)

BATCH_SIZE = 32

_AUTOTUNE = tf.data.AUTOTUNE
_CPU = os.cpu_count() or 4
_NUM_PARALLEL = min(16, max(4, _CPU))

_BRIGHT_LO, _BRIGHT_HI = 0.8, 1.2
_ROT_DEG = 180.0
_ZOOM_LO, _ZOOM_HI = 0.8, 1.2
_SHIFT_FRAC = 0.2  # width/height shift range


def _decode_resize(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE), method="bilinear", antialias=True)
    img = tf.cast(img, tf.float32)
    return img


def _apply_train_aug(img):
    img = tf.image.random_flip_left_right(img, seed=SEED)

    factor = tf.random.uniform([], _BRIGHT_LO, _BRIGHT_HI, seed=SEED)
    img = tf.clip_by_value(img * factor, 0.0, 255.0)

    angle = tf.random.uniform([], -_ROT_DEG, _ROT_DEG, seed=SEED) * (np.pi / 180.0)
    if hasattr(tf.image, "rotate"):
        img = tf.image.rotate(img, angles=angle, fill_mode="reflect")
    else:
        k = tf.random.uniform([], 0, 4, dtype=tf.int32, seed=SEED)
        img = tf.image.rot90(img, k=k)

    zoom = tf.random.uniform([], _ZOOM_LO, _ZOOM_HI, seed=SEED)
    crop_size = tf.cast(
        tf.round(tf.cast(IMG_SIZE, tf.float32) * tf.minimum(1.0, zoom)), tf.int32
    )
    crop_size = tf.clip_by_value(crop_size, 1, IMG_SIZE)
    img = tf.image.random_crop(img, size=[crop_size, crop_size, 3], seed=SEED)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE), method="bilinear", antialias=True)

    pad = tf.cast(tf.round(tf.cast(IMG_SIZE, tf.float32) * _SHIFT_FRAC), tf.int32)
    img = tf.pad(img, [[pad, pad], [pad, pad], [0, 0]], mode="REFLECT")
    img = tf.image.random_crop(img, size=[IMG_SIZE, IMG_SIZE, 3], seed=SEED)

    return img


def _preprocess_inception(img):
    return preprocess_input(img)


def make_train_ds(df, batch_size):
    paths = df["filepath"].to_numpy(dtype=str)
    y_int = df["diagnosis"].to_numpy(dtype=np.int32)
    y_oh = tf.keras.utils.to_categorical(y_int, num_classes=5).astype(np.float32)

    ds = tf.data.Dataset.from_tensor_slices((paths, y_oh))
    ds = ds.shuffle(len(df), seed=SEED, reshuffle_each_iteration=True)

    def _map(path, label):
        img = _decode_resize(path)
        img = _apply_train_aug(img)
        img = _preprocess_inception(img)
        img.set_shape((IMG_SIZE, IMG_SIZE, 3))
        label = tf.ensure_shape(label, (5,))
        return img, label

    opts = tf.data.Options()
    opts.experimental_deterministic = True
    ds = ds.with_options(opts)
    ds = ds.map(_map, num_parallel_calls=_NUM_PARALLEL, deterministic=True)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(_AUTOTUNE)
    return ds


def make_valid_ds(df, batch_size):
    paths = df["filepath"].to_numpy(dtype=str)
    y_int = df["diagnosis"].to_numpy(dtype=np.int32)
    y_oh = tf.keras.utils.to_categorical(y_int, num_classes=5).astype(np.float32)

    ds = tf.data.Dataset.from_tensor_slices((paths, y_oh))

    def _map(path, label):
        img = _decode_resize(path)
        img = _preprocess_inception(img)
        img.set_shape((IMG_SIZE, IMG_SIZE, 3))
        label = tf.ensure_shape(label, (5,))
        return img, label

    opts = tf.data.Options()
    opts.experimental_deterministic = True
    ds = ds.with_options(opts)
    ds = ds.map(_map, num_parallel_calls=_NUM_PARALLEL, deterministic=True)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(_AUTOTUNE)
    return ds


train_ds = make_train_ds(train_df_split, BATCH_SIZE)
valid_ds = make_valid_ds(valid_df_split, BATCH_SIZE)



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3935721824.py in <cell line: 0>()
----> 1 train_df_split, valid_df_split = train_test_split(
      2     train_df, test_size=0.2, random_state=SEED, stratify=train_df["diagnosis"]
      3 )
      4 
      5 BATCH_SIZE = 32

NameError: name 'train_test_split' is not defined

## === cell 14
try:
    base_model = InceptionResNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
except Exception as e:
    print(
        f"Warning: could not load imagenet weights due to: {e}\nFalling back to random initialization."
    )
    base_model = InceptionResNetV2(
        include_top=False,
        weights=None,
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )

for layer in base_model.layers:
    layer.trainable = True



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2471691972.py in <cell line: 0>()
      1 try:
----> 2     base_model = InceptionResNetV2(
      3         include_top=False,

NameError: name 'InceptionResNetV2' is not defined

During handling of the above exception, another exception occurred:

NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2471691972.py in <cell line: 0>()
      9         f"Warning: could not load imagenet weights due to: {e}\nFalling back to random initialization."
     10     )
---> 11     base_model = InceptionResNetV2(
     12         include_top=False,
     13         weights=None,

NameError: name 'InceptionResNetV2' is not defined

## === cell 15
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(100)(x)
x = Dropout(0.3)(x)
predictions = Dense(5, activation="softmax")(x)
model = Model(inputs=base_model.input, outputs=predictions)

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1027498892.py in <cell line: 0>()
----> 1 x = base_model.output
      2 x = GlobalAveragePooling2D()(x)
      3 x = Dense(100)(x)
      4 x = Dropout(0.3)(x)
      5 predictions = Dense(5, activation="softmax")(x)

NameError: name 'base_model' is not defined

## === cell 16
checkpoint = ModelCheckpoint(
    "best_model.keras",
    monitor="val_accuracy",
    verbose=1,
    save_best_only=True,
    mode="max",
)
earlystop = EarlyStopping(
    monitor="val_accuracy",
    patience=10,
    verbose=1,
    mode="max",
    restore_best_weights=True,
)
reduce_lr = ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=5, verbose=1, min_lr=1e-7
)



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2830324852.py in <cell line: 0>()
----> 1 checkpoint = ModelCheckpoint(
      2     "best_model.keras",
      3     monitor="val_accuracy",
      4     verbose=1,
      5     save_best_only=True,

NameError: name 'ModelCheckpoint' is not defined

## === cell 17
EPOCHS = 1

history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=valid_ds,
    callbacks=[SimpleEpochLogger(), checkpoint, earlystop, reduce_lr],
    verbose=1,
)



## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3852321298.py in <cell line: 0>()
      1 EPOCHS = 1
      2 
----> 3 history = model.fit(
      4     train_ds,
      5     epochs=EPOCHS,

NameError: name 'model' is not defined

## === cell 18
if os.path.exists("best_model.keras"):
    model = tf.keras.models.load_model("best_model.keras")




## === cell 19
def make_test_ds(df, batch_size):
    paths = df["filepath"].to_numpy(dtype=str)
    ds = tf.data.Dataset.from_tensor_slices(paths)

    def _map(path):
        img = _decode_resize(path)
        img = _preprocess_inception(img)
        img.set_shape((IMG_SIZE, IMG_SIZE, 3))
        return img

    opts = tf.data.Options()
    opts.experimental_deterministic = True
    ds = ds.with_options(opts)
    ds = ds.map(_map, num_parallel_calls=_NUM_PARALLEL, deterministic=True)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(_AUTOTUNE)
    return ds


test_ds = make_test_ds(test_df, BATCH_SIZE)

preds = model.predict(
    test_ds,
    verbose=1,
)
test_df["diagnosis"] = np.argmax(preds, axis=1).astype(int)



## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1114817604.py in <cell line: 0>()
     18 
     19 
---> 20 test_ds = make_test_ds(test_df, BATCH_SIZE)
     21 
     22 preds = model.predict(

NameError: name 'BATCH_SIZE' is not defined

## === cell 20
submission_csv = "submission.csv"

sub = sample_sub[["id_code"]].copy()
sub = sub.merge(test_df[["id_code", "diagnosis"]], on="id_code", how="left")

sub["diagnosis"] = sub["diagnosis"].fillna(0).astype(int)

sub.to_csv(submission_csv, index=False)
print(f"Submission file saved as {submission_csv}")
print(sub.head())

## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/2447660254.py in <cell line: 0>()
      2 
      3 sub = sample_sub[["id_code"]].copy()
----> 4 sub = sub.merge(test_df[["id_code", "diagnosis"]], on="id_code", how="left")
      5 
      6 sub["diagnosis"] = sub["diagnosis"].fillna(0).astype(int)

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4106             if is_iterator(key):
   4107                 key = list(key)
-> 4108             indexer = self.columns._get_indexer_strict(key, "columns")[1]
   4109 
   4110         # take() does not accept boolean indexers

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _get_indexer_strict(self, key, axis_name)
   6198             keyarr, indexer, new_indexer = self._reindex_non_unique(keyarr)
   6199 
-> 6200         self._raise_if_missing(keyarr, indexer, axis_name)
   6201 
   6202         keyarr = self.take(indexer)

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _raise_if_missing(self, key, indexer, axis_name)
   6250 
   6251             not_found = list(ensure_index(key)[missing_mask.nonzero()[0]].unique())
-> 6252             raise KeyError(f"{not_found} not in index")
   6253 
   6254     @overload

KeyError: "['diagnosis'] not in index"
