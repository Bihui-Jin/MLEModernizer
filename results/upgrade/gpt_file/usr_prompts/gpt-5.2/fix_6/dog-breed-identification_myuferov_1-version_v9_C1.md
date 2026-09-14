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
Given a dataset of images of dogs, predict the breed of each image.

## Metric
Multi Class Log Loss.

## Submission Format
For each image in the test set, you must predict a probability for each of the different breeds. The file should contain a header and have the following format:
```
id,affenpinscher,afghan_hound,..,yorkshire_terrier
000621fb3cbb32d8935728e48679680e,0.0083,0.0,...,0.0083
etc.
```

## Dataset Description
- `train.zip` - the training set, you are provided the breed for these dogs
- `test.zip` - the test set, you must predict the probability of each breed for each image
- `sample_submission.csv` - a sample submission file in the correct format
- `labels.csv` - the breeds for the images in the train set

# 2. Python version

3.8

# 3. Installed packages

geopandas==0.14.4
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
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
tf_keras==2.18.0
tqdm==4.67.1

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (169 lines)
            labels.csv (9200 lines)
            labels.csv.zip (201.6 kB)
            sample_submission.csv (1024 lines)
            sample_submission.csv.zip (32.1 kB)
            test.zip (36.4 MB)
            train.zip (324.7 MB)
            dog-breed-identification/
                description.md (169 lines)
                labels.csv (9200 lines)
                ... and 5 other files
                dog-breed-identification/
                test/
                    bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                    53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                    ... and 1021 other files
                    test/
                train/
                    868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                    7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                    ... and 9197 other files
                    train/
            test/
                bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                ... and 1021 other files
                test/
            train/
                868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                ... and 9197 other files
                train/
        input/
            description.md (169 lines)
            labels.csv (9200 lines)
            labels.csv.zip (201.6 kB)
            sample_submission.csv (1024 lines)
            sample_submission.csv.zip (32.1 kB)
            test.zip (36.4 MB)
            train.zip (324.7 MB)
            dog-breed-identification/
                description.md (169 lines)
                labels.csv (9200 lines)
                ... and 5 other files
                dog-breed-identification/
                test/
                    bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                    53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                    ... and 1021 other files
                    test/
                train/
                    868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                    7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                    ... and 9197 other files
                    train/
            test/
                bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                ... and 1021 other files
                test/
                    bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                    53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                    ... and 1021 other files
                    test/
            train/
                868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                ... and 9197 other files
                train/
                    868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                    7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                    ... and 9197 other files
                    train/
        working/
            dog-breed-identification/
                description.md (169 lines)
                labels.csv (9200 lines)
                ... and 5 other files
                dog-breed-identification/
                test/
                    bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                    53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                    ... and 1021 other files
                    test/
                train/
                    868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                    7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                    ... and 9197 other files
                    train/
```

-> data/dog-breed-identification/labels.csv has 9199 rows and 2 columns.
The columns are: id, breed

-> data/dog-breed-identification/sample_submission.csv has 1023 rows and 121 columns.
The columns are: id, affenpinscher, afghan_hound, african_hunting_dog, airedale, american_staffordshire_terrier, appenzeller, australian_terrier, basenji, basset, beagle, bedlington_terrier, bernese_mountain_dog, black-and-tan_coonhound, blenheim_spaniel... and 106 more columns

-> data/labels.csv has 9199 rows and 2 columns.
The columns are: id, breed

-> data/sample_submission.csv has 1023 rows and 121 columns.
The columns are: id, affenpinscher, afghan_hound, african_hunting_dog, airedale, american_staffordshire_terrier, appenzeller, australian_terrier, basenji, basset, beagle, bedlington_terrier, bernese_mountain_dog, black-and-tan_coonhound, blenheim_spaniel... and 106 more columns

-> input/dog-breed-identification/labels.csv has 9199 rows and 2 columns.
The columns are: id, breed

-> input/dog-breed-identification/sample_submission.csv has 1023 rows and 121 columns.
The columns are: id, affenpinscher, afghan_hound, african_hunting_dog, airedale, american_staffordshire_terrier, appenzeller, australian_terrier, basenji, basset, beagle, bedlington_terrier, bernese_mountain_dog, black-and-tan_coonhound, blenheim_spaniel... and 106 more columns

-> (stopped after 10 files for performance)

# 5. Target score

5.57999

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 4.78552) has done: 'I fix the Keras/TensorFlow import incompatibilities causing the protobuf `GetPrototype` crash by using `tf.keras` consistently, and I remove notebook-only syntax (`%matplotlib inline`) so the script runs as a .py. I update deprecated/removed Keras 3 module paths (e.g., `keras.layers.convolutional`, `keras.utils.generic_utils`) and fix `ImageDataGenerator` usage so the generators are defined. I also switch the final activation to `softmax` (multi-class log loss expects a probability distribution) while keeping the same CNN architecture and training loop, and ensure the checkpoint saves to a `.keras` file and is loaded correctly. Finally, I build the submission by following `sample_submission.csv` column order to guarantee a valid `.csv` with the required header and aligned `id`s.'
- What this solution (achieved 4.34651) has done: 'I fix the training crash by telling `EarlyStopping` and `ModelCheckpoint` how to interpret `val_categorical_accuracy` (mode='max'), which unblocks model.fit and ensures the checkpoint file is actually created. To address the protobuf `MessageFactory.GetPrototype` error seen at import time in some Kaggle images, I force the pure-Python protobuf implementation before importing TensorFlow (a minimal environment-compatibility fix). I keep the same CNN architecture, preprocessing, generators, and training loop; no score-tuning changes are introduced beyond making it run. Finally, I make model loading conditional on the checkpoint existing (fallback to the in-memory model) so submission generation always completes and writes a valid `submission.csv`.'
- What this solution (achieved 4.47448) has done: 'I fix the import-time crash (`MessageFactory` missing `GetPrototype`) by forcing a protobuf version that is compatible with TensorFlow 2.18, without changing your model/training logic. To keep the run stable in Kaggle, I also add a small fallback that disables the “python protobuf implementation” override if it still triggers the error, then re-import TensorFlow cleanly. Finally, I ensure the submission is always written in the exact `sample_submission.csv` column order and that predictions are numerically safe probabilities (row-normalized and clipped), which is score-neutral but prevents invalid log-loss edge cases.'
- What this solution (achieved 4.66913) has done: 'I fix the import-time protobuf/TensorFlow crash by setting a compatible protobuf implementation *before* importing TensorFlow (this is the root cause of the `MessageFactory.GetPrototype` error). I keep your exact CNN, generators, training loop, and submission construction unchanged so the modeling/evaluation semantics stay the same and the score should remain in the same range (already better than the target, so no intentional score-tuning changes). I also make the TensorFlow import resilient by retrying with a fallback environment setting if the first import still fails in this Kaggle image. Finally, I keep the submission writing exactly as you already do to guarantee a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "cpp")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.model_selection import train_test_split

try:
    import tensorflow as tf
except Exception as e:
    os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
    os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)
    import tensorflow as tf  # noqa: F401

from tensorflow import keras
from tensorflow.keras import backend
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator,
    load_img,
    img_to_array,
)
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

np.random.seed(42)
tf.random.set_seed(42)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_11/3012723533.py in <cell line: 0>()
     19 try:
---> 20     import tensorflow as tf
     21 except Exception as e:

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
/tmp/ipykernel_11/3012723533.py in <cell line: 0>()
     23     os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
     24     os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)
---> 25     import tensorflow as tf  # noqa: F401
     26 
     27 from tensorflow import keras

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
def gen_graph(history, title):
    if history is None:
        return
    if (
        "categorical_accuracy" in history.history
        and "val_categorical_accuracy" in history.history
    ):
        plt.plot(history.history["categorical_accuracy"])
        plt.plot(history.history["val_categorical_accuracy"])
        plt.title("Accuracy " + title)
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch")
        plt.legend(["train", "validation"], loc="upper left")
        plt.show()

    if "fbeta" in history.history and "val_fbeta" in history.history:
        plt.plot(history.history["fbeta"])
        plt.plot(history.history["val_fbeta"])
        plt.title("fbeta " + title)
        plt.ylabel("fbeta")
        plt.xlabel("Epoch")
        plt.legend(["train", "validation"], loc="upper left")
        plt.show()




## === cell 2
def fbeta(y_true, y_pred, beta=2):
    y_pred = backend.clip(y_pred, 0, 1)
    tp = backend.sum(backend.round(backend.clip(y_true * y_pred, 0, 1)), axis=1)
    fp = backend.sum(backend.round(backend.clip(y_pred - y_true, 0, 1)), axis=1)
    fn = backend.sum(backend.round(backend.clip(y_true - y_pred, 0, 1)), axis=1)
    p = tp / (tp + fp + backend.epsilon())
    r = tp / (tp + fn + backend.epsilon())
    bb = beta**2
    fbeta_score = backend.mean((1 + bb) * (p * r) / (bb * p + r + backend.epsilon()))
    return fbeta_score




## === cell 3
DATA_DIR = "../input/dog-breed-identification"

labels_path = os.path.join(DATA_DIR, "labels.csv")
sample_sub_path = os.path.join(DATA_DIR, "sample_submission.csv")
train_dir = os.path.join(DATA_DIR, "train")
test_dir = os.path.join(DATA_DIR, "test")

assert os.path.exists(labels_path), f"Missing: {labels_path}"
assert os.path.exists(sample_sub_path), f"Missing: {sample_sub_path}"
assert os.path.isdir(train_dir), f"Missing dir: {train_dir}"
assert os.path.isdir(test_dir), f"Missing dir: {test_dir}"

df_train = pd.read_csv(labels_path)
df_test = pd.read_csv(sample_sub_path)

jpg_train = os.path.join(train_dir, "{}.jpg")
jpg_test = os.path.join(test_dir, "{}.jpg")



## === cell 4
df_train.head()



## === cell 5
df_test.head()



## === cell 6
labels = df_train["breed"]
one_hot = pd.get_dummies(labels, sparse=False)



## === cell 7
one_hot_labels = np.asarray(one_hot, dtype=np.float32)
one_hot_labels.shape



## === cell 8
im_resize = 64
num_class = one_hot_labels.shape[1]
num_class



## === cell 9
x_train = []
y_train = []
x_test = []



## === cell 10
i = 0
for f, breed in tqdm(df_train.values, total=len(df_train)):
    img = load_img(jpg_train.format(f), target_size=(im_resize, im_resize))
    img_resized = img_to_array(img)
    x_train.append(img_resized)
    y_train.append(one_hot_labels[i])
    i += 1



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1463876322.py in <cell line: 0>()
      1 i = 0
      2 for f, breed in tqdm(df_train.values, total=len(df_train)):
----> 3     img = load_img(jpg_train.format(f), target_size=(im_resize, im_resize))
      4     img_resized = img_to_array(img)
      5     x_train.append(img_resized)

NameError: name 'load_img' is not defined

## === cell 11
for f in tqdm(df_test["id"].values, total=len(df_test)):
    img = load_img(jpg_test.format(f), target_size=(im_resize, im_resize))
    img_resized = img_to_array(img)
    x_test.append(img_resized)



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/476603467.py in <cell line: 0>()
      1 for f in tqdm(df_test["id"].values, total=len(df_test)):
----> 2     img = load_img(jpg_test.format(f), target_size=(im_resize, im_resize))
      3     img_resized = img_to_array(img)
      4     x_test.append(img_resized)
      5 

NameError: name 'load_img' is not defined

## === cell 12
X_train, X_valid, Y_train, Y_valid = train_test_split(
    np.array(x_train, dtype=np.float32),
    np.array(y_train, dtype=np.float32),
    shuffle=True,
    test_size=0.2,
    random_state=42,
    stratify=np.argmax(np.array(y_train), axis=1),
)



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
AxisError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3394648863.py in <cell line: 0>()
      5     test_size=0.2,
      6     random_state=42,
----> 7     stratify=np.argmax(np.array(y_train), axis=1),
      8 )
      9 

/usr/local/lib/python3.11/dist-packages/numpy/core/fromnumeric.py in argmax(a, axis, out, keepdims)
   1227     """
   1228     kwds = {'keepdims': keepdims} if keepdims is not np._NoValue else {}
-> 1229     return _wrapfunc(a, 'argmax', axis=axis, out=out, **kwds)
   1230 
   1231 

/usr/local/lib/python3.11/dist-packages/numpy/core/fromnumeric.py in _wrapfunc(obj, method, *args, **kwds)
     57 
     58     try:
---> 59         return bound(*args, **kwds)
     60     except TypeError:
     61         # A TypeError occurs if the object does have such a method in its

AxisError: axis 1 is out of bounds for array of dimension 1

## === cell 13
del x_train, y_train



## === cell 14
datagen = ImageDataGenerator(rescale=1.0 / 255.0)



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3425902930.py in <cell line: 0>()
----> 1 datagen = ImageDataGenerator(rescale=1.0 / 255.0)
      2 

NameError: name 'ImageDataGenerator' is not defined

## === cell 15
model = Sequential()

model.add(
    Conv2D(
        32,
        (3, 3),
        padding="same",
        input_shape=(im_resize, im_resize, 3),
        activation="relu",
        kernel_initializer="he_uniform",
    )
)
model.add(
    Conv2D(
        32,
        (3, 3),
        activation="relu",
        kernel_initializer="he_uniform",
        padding="same",
    )
)
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(
    Conv2D(
        64,
        (3, 3),
        activation="relu",
        kernel_initializer="he_uniform",
        padding="same",
    )
)
model.add(
    Conv2D(
        64,
        (3, 3),
        activation="relu",
        kernel_initializer="he_uniform",
        padding="same",
    )
)
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(
    Conv2D(
        128,
        (3, 3),
        activation="relu",
        kernel_initializer="he_uniform",
        padding="same",
    )
)
model.add(
    Conv2D(
        128,
        (3, 3),
        activation="relu",
        kernel_initializer="he_uniform",
        padding="same",
    )
)
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(256, activation="relu", kernel_initializer="he_uniform"))
model.add(Dropout(0.5))

model.add(Dense(num_class, activation="softmax"))



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3771841444.py in <cell line: 0>()
----> 1 model = Sequential()
      2 
      3 model.add(
      4     Conv2D(
      5         32,

NameError: name 'Sequential' is not defined

## === cell 16
print(model.summary())



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3518542503.py in <cell line: 0>()
----> 1 print(model.summary())
      2 

NameError: name 'model' is not defined

## === cell 17
model.compile(
    optimizer="Adam",
    loss="categorical_crossentropy",
    metrics=[categorical_accuracy, fbeta],
)



## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/854596951.py in <cell line: 0>()
----> 1 model.compile(
      2     optimizer="Adam",
      3     loss="categorical_crossentropy",
      4     metrics=[categorical_accuracy, fbeta],
      5 )

NameError: name 'model' is not defined

## === cell 18
train_generator = datagen.flow(X_train, Y_train, batch_size=128, shuffle=True)
valid_generator = datagen.flow(X_valid, Y_valid, batch_size=128, shuffle=False)



## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1524288797.py in <cell line: 0>()
----> 1 train_generator = datagen.flow(X_train, Y_train, batch_size=128, shuffle=True)
      2 valid_generator = datagen.flow(X_valid, Y_valid, batch_size=128, shuffle=False)
      3 

NameError: name 'datagen' is not defined

## === cell 19
earlystop = EarlyStopping(
    monitor="val_categorical_accuracy",
    mode="max",
    min_delta=0.0,
    patience=5,
    restore_best_weights=False,
)

checkpoint_path = "model_best.keras"
checkpoint_callback = ModelCheckpoint(
    checkpoint_path,
    monitor="val_categorical_accuracy",
    mode="max",
    save_best_only=True,
    verbose=1,
)



## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1793709027.py in <cell line: 0>()
----> 1 earlystop = EarlyStopping(
      2     monitor="val_categorical_accuracy",
      3     mode="max",
      4     min_delta=0.0,
      5     patience=5,

NameError: name 'EarlyStopping' is not defined

## === cell 20
batch_size = 128
Epochs = 50

history_rmsprop = model.fit(
    train_generator,
    callbacks=[earlystop, checkpoint_callback],
    epochs=Epochs,
    steps_per_epoch=len(train_generator),
    validation_data=valid_generator,
    validation_steps=len(valid_generator),
    verbose=2,
)



## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3996850582.py in <cell line: 0>()
      2 Epochs = 50
      3 
----> 4 history_rmsprop = model.fit(
      5     train_generator,
      6     callbacks=[earlystop, checkpoint_callback],

NameError: name 'model' is not defined

## === cell 21
gen_graph(history_rmsprop, "training curves")



## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3864733437.py in <cell line: 0>()
----> 1 gen_graph(history_rmsprop, "training curves")
      2 

NameError: name 'history_rmsprop' is not defined

## === cell 22
if os.path.exists(checkpoint_path):
    model = keras.models.load_model(checkpoint_path, custom_objects={"fbeta": fbeta})
else:
    print(f"Warning: checkpoint not found at {checkpoint_path}; using in-memory model.")



## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1086495418.py in <cell line: 0>()
----> 1 if os.path.exists(checkpoint_path):
      2     model = keras.models.load_model(checkpoint_path, custom_objects={"fbeta": fbeta})
      3 else:
      4     print(f"Warning: checkpoint not found at {checkpoint_path}; using in-memory model.")
      5 

NameError: name 'checkpoint_path' is not defined

## === cell 23
x_test_arr = np.array(x_test, dtype=np.float32) / 255.0
preds = model.predict(x_test_arr, batch_size=128, verbose=0)



## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3616670199.py in <cell line: 0>()
      1 x_test_arr = np.array(x_test, dtype=np.float32) / 255.0
----> 2 preds = model.predict(x_test_arr, batch_size=128, verbose=0)
      3 

NameError: name 'model' is not defined

## === cell 24
sample_cols = df_test.columns.tolist()  # includes "id" then breed columns
breed_cols = sample_cols[1:]

sub = pd.DataFrame(preds, columns=breed_cols)
sub.insert(0, "id", df_test["id"].values)

probs = sub[breed_cols].to_numpy(dtype=np.float64)
probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
row_sums = probs.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1.0
probs = probs / row_sums
probs = np.clip(probs, 1e-15, 1.0)
probs = probs / probs.sum(axis=1, keepdims=True)

sub[breed_cols] = probs.astype(np.float32)
sub = sub[["id"] + breed_cols]  # enforce order

sub.head(5)



## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4148024444.py in <cell line: 0>()
      2 breed_cols = sample_cols[1:]
      3 
----> 4 sub = pd.DataFrame(preds, columns=breed_cols)
      5 sub.insert(0, "id", df_test["id"].values)
      6 

NameError: name 'preds' is not defined

## === cell 25
out_path = "submission.csv"
sub.to_csv(out_path, index=False)
print(f"Wrote submission: {out_path} with shape={sub.shape}")
print(
    "Columns match sample_submission:", sub.columns.tolist() == df_test.columns.tolist()
)

## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/936213813.py in <cell line: 0>()
      1 out_path = "submission.csv"
----> 2 sub.to_csv(out_path, index=False)
      3 print(f"Wrote submission: {out_path} with shape={sub.shape}")
      4 print(
      5     "Columns match sample_submission:", sub.columns.tolist() == df_test.columns.tolist()

NameError: name 'sub' is not defined
