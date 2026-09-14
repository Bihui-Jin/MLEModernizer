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
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
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

0.993

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "cpp"

import glob
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

import tensorflow as tf
import tf_keras as keras
from tf_keras.models import Sequential
from tf_keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tf_keras.optimizers import RMSprop
from tf_keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split

np.random.seed(27)
tf.random.set_seed(27)

INPUT_ROOT = "../input"
COMP_ROOT = os.path.join(INPUT_ROOT, "aerial-cactus-identification")
if os.path.exists(COMP_ROOT):
    DATA_ROOT = COMP_ROOT
else:
    DATA_ROOT = INPUT_ROOT

print("Using DATA_ROOT:", DATA_ROOT)
print("Listing ../input:", os.listdir(INPUT_ROOT)[:20])

TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
SAMPLE_SUB = os.path.join(DATA_ROOT, "sample_submission.csv")


def resolve_dir(root, name):
    """
    Kaggle can have either:
      root/train/*.jpg
    or nested variants. The earlier logic preferred root/train/train which is
    wrong for this dataset and yields zero images.

    We try the most likely candidates in a safe order and return the first that exists.
    """
    candidates = [
        os.path.join(root, name),  # expected: .../train or .../test
        os.path.join(root, name, name),  # fallback: .../train/train
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    return candidates[0]


TRAIN_DIR = resolve_dir(DATA_ROOT, "train")
TEST_DIR = resolve_dir(DATA_ROOT, "test")

print("TRAIN_CSV:", TRAIN_CSV, "exists:", os.path.exists(TRAIN_CSV))
print("SAMPLE_SUB:", SAMPLE_SUB, "exists:", os.path.exists(SAMPLE_SUB))
print("TRAIN_DIR:", TRAIN_DIR, "exists:", os.path.isdir(TRAIN_DIR))
print("TEST_DIR:", TEST_DIR, "exists:", os.path.isdir(TEST_DIR))




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_11/760024234.py in <cell line: 0>()
     11 import matplotlib.pyplot as plt
     12 
---> 13 import tensorflow as tf
     14 import tf_keras as keras
     15 from tf_keras.models import Sequential

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
def read_pix(jpg_dir):
    filenames = glob.glob(os.path.join(jpg_dir, "*.jpg"))
    filenames = sorted(filenames)  # deterministic order
    img_array = np.zeros((len(filenames), 32, 32, 3), dtype=np.uint8)
    img_index = []
    for idx, filename in enumerate(filenames):
        im_tmp = matplotlib.image.imread(filename)
        if im_tmp.dtype != np.uint8:
            im_tmp = (
                (im_tmp * 255.0).astype(np.uint8)
                if im_tmp.max() <= 1.0
                else im_tmp.astype(np.uint8)
            )
        img_array[idx] = im_tmp[:, :, :3]
        img_index.append(os.path.basename(filename))
    return img_array, img_index


def prepare_data(img_array, img_index, train_response):
    y_train_series, y_test_series = train_test_split(
        train_response,
        shuffle=True,
        random_state=12,
        test_size=0.25,
        stratify=train_response.values,
    )
    y_train = y_train_series.values[:, np.newaxis]
    y_test = y_test_series.values[:, np.newaxis]

    pos = {fn: i for i, fn in enumerate(img_index)}
    train_index = [pos[idx] for idx in y_train_series.index]
    test_index = [pos[idx] for idx in y_test_series.index]

    x_train = img_array[train_index]
    x_test = img_array[test_index]
    return x_train, y_train, x_test, y_test, y_train_series, y_test_series


def simple_oversample(x, y, oversample_class, oversample_factor=3):
    new_x = x.copy()
    new_y = y.copy()
    oversample_mask = np.ravel(y == oversample_class)
    oversample_x = x[oversample_mask]
    oversample_y = y[oversample_mask]
    for _ in range(oversample_factor):
        new_x = np.concatenate([new_x, oversample_x], axis=0)
        new_y = np.concatenate([new_y, oversample_y], axis=0)
    return new_x, new_y




## === cell 2
train_labels = pd.read_csv(TRAIN_CSV)
sample_submission = pd.read_csv(SAMPLE_SUB)

train_img_array, train_img_index = read_pix(TRAIN_DIR)
test_img_array, test_img_index = read_pix(TEST_DIR)

train_response = train_labels["has_cactus"].copy()
train_response.index = train_labels["id"]

train_response = train_response.loc[train_img_index]

print("Train images:", train_img_array.shape, "Train labels:", train_response.shape)
print(
    "Test images:", test_img_array.shape, "Sample submission:", sample_submission.shape
)

if train_img_array.shape[0] == 0:
    raise RuntimeError(f"No training images found in TRAIN_DIR={TRAIN_DIR}")
if test_img_array.shape[0] == 0:
    raise RuntimeError(f"No test images found in TEST_DIR={TEST_DIR}")



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/789506470.py in <cell line: 0>()
----> 1 train_labels = pd.read_csv(TRAIN_CSV)
      2 sample_submission = pd.read_csv(SAMPLE_SUB)
      3 
      4 train_img_array, train_img_index = read_pix(TRAIN_DIR)
      5 test_img_array, test_img_index = read_pix(TEST_DIR)

NameError: name 'TRAIN_CSV' is not defined

## === cell 3
print("Image Labels")
print(train_labels.head(2))
print("\nImage Labels (Series)")
print(train_response.head(2))
print("\nSubmission Example")
print(sample_submission.head(2))



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/893640727.py in <cell line: 0>()
      1 print("Image Labels")
----> 2 print(train_labels.head(2))
      3 print("\nImage Labels (Series)")
      4 print(train_response.head(2))
      5 print("\nSubmission Example")

NameError: name 'train_labels' is not defined

## === cell 4
print("Some examples")
np.random.seed(27)
if train_img_array.shape[0] > 0:
    inspect = np.random.randint(low=0, high=train_img_array.shape[0], size=9)
    for i in range(9):
        pic_id = inspect[i]
        plt.subplot(330 + 1 + i)
        plt.imshow(train_img_array[pic_id].astype(np.uint8))
        plt.axis("off")
    plt.show()

    print("Image Labels")
    print(train_response.loc[np.array(train_img_index)[inspect]])
else:
    print("No training images to display.")



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2667934997.py in <cell line: 0>()
      1 print("Some examples")
      2 np.random.seed(27)
----> 3 if train_img_array.shape[0] > 0:
      4     inspect = np.random.randint(low=0, high=train_img_array.shape[0], size=9)
      5     for i in range(9):

NameError: name 'train_img_array' is not defined

## === cell 5
count_classes = train_response.value_counts().sort_index()
n0 = int(count_classes.get(0, 0))
n1 = int(count_classes.get(1, 0))

pd.DataFrame({"count": [n0, n1]}, index=["no cactus", "cactus"]).plot(
    kind="bar", legend=False
)
plt.xticks(rotation=0)
plt.title("Number of instances in training data")
plt.show()

ratio = (n1 / n0) if n0 > 0 else np.nan
print(
    "Ratio (cactus vs. no cactus): {:.2f}".format(
        ratio if np.isfinite(ratio) else float("nan")
    )
)



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3644101381.py in <cell line: 0>()
      1 # Fix crosstab column name robustness: depending on pandas versions/inputs,
      2 # the column can be named differently. We compute the ratio directly.
----> 3 count_classes = train_response.value_counts().sort_index()
      4 # Expected labels are 0 and 1; keep safe defaults
      5 n0 = int(count_classes.get(0, 0))

NameError: name 'train_response' is not defined

## === cell 6
x_train, y_train, x_test, y_test, y_train_series, y_test_series = prepare_data(
    train_img_array, train_img_index, train_response
)

count_train = y_train_series.value_counts().sort_index()
n0t = int(count_train.get(0, 0))
n1t = int(count_train.get(1, 0))

pd.DataFrame({"count": [n0t, n1t]}, index=["no cactus", "cactus"]).plot(
    kind="bar", legend=False
)
plt.xticks(rotation=0)
plt.title("Number of instances in train split")
plt.show()

ratio_t = (n1t / n0t) if n0t > 0 else np.nan
print(
    "Ratio (cactus vs. no cactus): {:.2f}".format(
        ratio_t if np.isfinite(ratio_t) else float("nan")
    )
)



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/503184604.py in <cell line: 0>()
      1 x_train, y_train, x_test, y_test, y_train_series, y_test_series = prepare_data(
----> 2     train_img_array, train_img_index, train_response
      3 )
      4 
      5 count_train = y_train_series.value_counts().sort_index()

NameError: name 'train_img_array' is not defined

## === cell 7
print("Some images from the training set")
np.random.seed(26)
if y_train.shape[0] > 0:
    inspect = np.random.randint(low=0, high=y_train.shape[0], size=9)
    for i in range(9):
        pic_id = inspect[i]
        plt.subplot(330 + 1 + i)
        plt.imshow(x_train[pic_id].astype(np.uint8))
        plt.axis("off")
    plt.show()

    print("Accompanying labels")
    print(y_train[inspect].ravel())
else:
    print("Empty y_train; cannot display samples.")



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3710765340.py in <cell line: 0>()
      1 print("Some images from the training set")
      2 np.random.seed(26)
----> 3 if y_train.shape[0] > 0:
      4     inspect = np.random.randint(low=0, high=y_train.shape[0], size=9)
      5     for i in range(9):

NameError: name 'y_train' is not defined

## === cell 8
batch_size = 128
num_classes = 2
epochs = 12

datagen = ImageDataGenerator(
    rotation_range=360,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode="nearest",
)

x_train, y_train, x_test, y_test, y_train_series, y_test_series = prepare_data(
    train_img_array, train_img_index, train_response
)

x_train, y_train = simple_oversample(
    x_train, y_train, oversample_class=0, oversample_factor=2
)
x_test, y_test = simple_oversample(
    x_test, y_test, oversample_class=0, oversample_factor=2
)

print("Number of 'no cactus' samples in y_train: {}".format(int((y_train == 0).sum())))
print("Number of 'cactus' samples in y_train: {}".format(int((y_train == 1).sum())))



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/317377435.py in <cell line: 0>()
      3 epochs = 12
      4 
----> 5 datagen = ImageDataGenerator(
      6     rotation_range=360,
      7     width_shift_range=0.2,

NameError: name 'ImageDataGenerator' is not defined

## === cell 9
print("Some images from the oversampled training set")
np.random.seed(26)
if y_train.shape[0] > 0:
    inspect = np.random.randint(low=0, high=y_train.shape[0], size=9)
    for i in range(9):
        pic_id = inspect[i]
        plt.subplot(330 + 1 + i)
        plt.imshow(x_train[pic_id].astype(np.uint8))
        plt.axis("off")
    plt.show()

    print("Accompanying labels")
    print(y_train[inspect].ravel())
else:
    print("Empty y_train after oversampling; cannot display samples.")



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2938237277.py in <cell line: 0>()
      1 print("Some images from the oversampled training set")
      2 np.random.seed(26)
----> 3 if y_train.shape[0] > 0:
      4     inspect = np.random.randint(low=0, high=y_train.shape[0], size=9)
      5     for i in range(9):

NameError: name 'y_train' is not defined

## === cell 10
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

train_generator = datagen.flow(x_train, y_train, batch_size=batch_size, shuffle=True)
test_datagen = ImageDataGenerator()
validation_generator = test_datagen.flow(
    x_test, y_test, batch_size=batch_size, shuffle=False
)




## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1101043566.py in <cell line: 0>()
----> 1 x_train = x_train.astype("float32") / 255.0
      2 x_test = x_test.astype("float32") / 255.0
      3 
      4 train_generator = datagen.flow(x_train, y_train, batch_size=batch_size, shuffle=True)
      5 test_datagen = ImageDataGenerator()

NameError: name 'x_train' is not defined

## === cell 11
def convnet_model():
    model = Sequential()
    model.add(Conv2D(32, (5, 5), input_shape=(32, 32, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (5, 5), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(128, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(
        loss="binary_crossentropy",
        optimizer=RMSprop(learning_rate=1e-4),
        metrics=["accuracy"],
    )
    return model




## === cell 12
model = convnet_model()

hist = model.fit(
    train_generator,
    steps_per_epoch=int(np.ceil(x_train.shape[0] / 32.0)),
    epochs=20,
    validation_data=(x_test, y_test),
    verbose=2,
)

scores = model.evaluate(x_test, y_test, verbose=0)
print("CNN error: {:.2f}".format(100 - scores[1] * 100))



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4114583003.py in <cell line: 0>()
----> 1 model = convnet_model()
      2 
      3 hist = model.fit(
      4     train_generator,
      5     steps_per_epoch=int(np.ceil(x_train.shape[0] / 32.0)),

/tmp/ipykernel_11/332688819.py in convnet_model()
      1 def convnet_model():
----> 2     model = Sequential()
      3     model.add(Conv2D(32, (5, 5), input_shape=(32, 32, 3), activation="relu"))
      4     model.add(MaxPooling2D(pool_size=(2, 2)))
      5     model.add(Conv2D(64, (5, 5), activation="relu"))

NameError: name 'Sequential' is not defined

## === cell 13
plt.plot(hist.history.get("accuracy", []))
plt.plot(hist.history.get("val_accuracy", []))
plt.title("model accuracy")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(["train", "test"], loc="upper left")
plt.show()

plt.plot(hist.history.get("loss", []))
plt.plot(hist.history.get("val_loss", []))
plt.title("model loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train", "test"], loc="upper left")
plt.show()



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1942356055.py in <cell line: 0>()
----> 1 plt.plot(hist.history.get("accuracy", []))
      2 plt.plot(hist.history.get("val_accuracy", []))
      3 plt.title("model accuracy")
      4 plt.ylabel("accuracy")
      5 plt.xlabel("epoch")

NameError: name 'hist' is not defined

## === cell 14
test_x = test_img_array.astype("float32") / 255.0
if test_x.shape[0] == 0:
    raise RuntimeError("Test array is empty; cannot create submission.")

probability = model.predict(test_x, batch_size=batch_size, verbose=0).ravel()

pred_df = pd.DataFrame({"id": test_img_index, "has_cactus": probability})
res = sample_submission[["id"]].merge(pred_df, on="id", how="left")

res["has_cactus"] = res["has_cactus"].astype("float32").fillna(0.5).clip(0.0, 1.0)

res.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", res.shape)
print(res.head())

## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2777496244.py in <cell line: 0>()
----> 1 test_x = test_img_array.astype("float32") / 255.0
      2 if test_x.shape[0] == 0:
      3     raise RuntimeError("Test array is empty; cannot create submission.")
      4 
      5 probability = model.predict(test_x, batch_size=batch_size, verbose=0).ravel()

NameError: name 'test_img_array' is not defined
