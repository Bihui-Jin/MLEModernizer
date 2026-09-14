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

3.9

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
scikit-image==0.25.2
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

0.951

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import random

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "cpp"

random.seed(42)
np.random.seed(42)

DATA_PATH = "/kaggle/input/aerial-cactus-identification/"
TRAIN_ZIP = os.path.join(DATA_PATH, "train.zip")
TEST_ZIP = os.path.join(DATA_PATH, "test.zip")
TRAIN_CSV = os.path.join(DATA_PATH, "train.csv")
SAMPLE_SUB = os.path.join(DATA_PATH, "sample_submission.csv")

TRAIN_EXTRACT_DIR = "./training"
TEST_EXTRACT_DIR = "./test"

os.makedirs(TRAIN_EXTRACT_DIR, exist_ok=True)
os.makedirs(TEST_EXTRACT_DIR, exist_ok=True)

print("Listing a few input files for sanity:")
for dirname, _, filenames in os.walk(DATA_PATH):
    for filename in filenames[:5]:
        print(os.path.join(dirname, filename))
    break



## === cell 1
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from zipfile import ZipFile



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_11/2567708867.py in <cell line: 0>()
----> 1 from tensorflow.keras.preprocessing.image import ImageDataGenerator
      2 from zipfile import ZipFile
      3 

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
files_dataframe = pd.read_csv(TRAIN_CSV, dtype={"id": str})
files_dataframe["has_cactus"] = files_dataframe["has_cactus"].astype(int).astype(str)
files_dataframe.head()



## === cell 3
train_extract_target = os.path.join(TRAIN_EXTRACT_DIR, "train")
test_extract_target = os.path.join(TEST_EXTRACT_DIR, "test")

if (not os.path.exists(train_extract_target)) or (
    len(os.listdir(train_extract_target)) == 0
):
    with ZipFile(TRAIN_ZIP, "r") as zipper:
        zipper.extractall(TRAIN_EXTRACT_DIR)

if (not os.path.exists(test_extract_target)) or (
    len(os.listdir(test_extract_target)) == 0
):
    with ZipFile(TEST_ZIP, "r") as zipper:
        zipper.extractall(TEST_EXTRACT_DIR)

base_train_dir = train_extract_target
print("Training sample paths:")
print((files_dataframe["id"].head(2)).tolist())

print(
    "Train dir exists:",
    os.path.exists(base_train_dir),
    "num_files:",
    len(os.listdir(base_train_dir)) if os.path.exists(base_train_dir) else -1,
)
print(
    "Test dir exists:",
    os.path.exists(test_extract_target),
    "num_files:",
    len(os.listdir(test_extract_target)) if os.path.exists(test_extract_target) else -1,
)



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2276268470.py in <cell line: 0>()
      5     len(os.listdir(train_extract_target)) == 0
      6 ):
----> 7     with ZipFile(TRAIN_ZIP, "r") as zipper:
      8         zipper.extractall(TRAIN_EXTRACT_DIR)
      9 

NameError: name 'ZipFile' is not defined

## === cell 4
import matplotlib.pyplot as plt

class_reparts = files_dataframe["has_cactus"].value_counts()
ax = class_reparts.plot.bar()



## === cell 5
total_samples = files_dataframe["has_cactus"].size
print("Total number of samples: ", total_samples)
has_cactus_weight = total_samples / (2 * class_reparts["1"])
no_cactus_weight = total_samples / (2 * class_reparts["0"])
class_weights = {0: no_cactus_weight, 1: has_cactus_weight}
print("Class weights: ", class_weights)



## === cell 6
from matplotlib.image import imread

plt.figure(figsize=(36, 12))
for i, k in enumerate(np.random.randint(0, len(files_dataframe), size=(20,))):
    p = os.path.join(base_train_dir, files_dataframe["id"].iloc[k])
    if os.path.exists(p):
        plt.subplot(4, 5, i + 1)
        plt.imshow(imread(p))
        plt.title("Label :" + str(files_dataframe["has_cactus"].iloc[k]))



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1173472791.py in <cell line: 0>()
      3 plt.figure(figsize=(36, 12))
      4 for i, k in enumerate(np.random.randint(0, len(files_dataframe), size=(20,))):
----> 5     p = os.path.join(base_train_dir, files_dataframe["id"].iloc[k])
      6     if os.path.exists(p):
      7         plt.subplot(4, 5, i + 1)

NameError: name 'base_train_dir' is not defined

## === cell 7
import skimage.exposure as exposure


def preprocess(img):
    p2, p98 = np.percentile(img, (2, 98))
    img_rescale = exposure.rescale_intensity(img, in_range=(p2, p98))
    return img_rescale


plt.figure(figsize=(12, 4))
p0 = os.path.join(base_train_dir, files_dataframe["id"].iloc[0])
if os.path.exists(p0):
    plt.subplot(121)
    img = imread(p0)
    plt.imshow(img)
    plt.title("Before histogram equalization")

    plt.subplot(122)
    img2 = preprocess(img)
    plt.imshow(img2)
    plt.title("After histogram equalization")



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1414196208.py in <cell line: 0>()
      9 
     10 plt.figure(figsize=(12, 4))
---> 11 p0 = os.path.join(base_train_dir, files_dataframe["id"].iloc[0])
     12 if os.path.exists(p0):
     13     plt.subplot(121)

NameError: name 'base_train_dir' is not defined

## === cell 8
generator = ImageDataGenerator(
    samplewise_center=True,
    samplewise_std_normalization=True,
    vertical_flip=True,
    horizontal_flip=True,
    rotation_range=45,
    validation_split=0.1,
    shear_range=10,
    preprocessing_function=preprocess,
)

noPreprocessGenerator = ImageDataGenerator(
    samplewise_center=True,
    samplewise_std_normalization=True,
    vertical_flip=True,
    horizontal_flip=True,
    rotation_range=45,
    validation_split=0.1,
)

noAugmentationGenerator = ImageDataGenerator(
    samplewise_center=True,
    samplewise_std_normalization=True,
    vertical_flip=False,
    horizontal_flip=False,
    preprocessing_function=preprocess,
)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1950073671.py in <cell line: 0>()
----> 1 generator = ImageDataGenerator(
      2     samplewise_center=True,
      3     samplewise_std_normalization=True,
      4     vertical_flip=True,
      5     horizontal_flip=True,

NameError: name 'ImageDataGenerator' is not defined

## === cell 9
training_generator = generator.flow_from_dataframe(
    dataframe=files_dataframe,
    directory=base_train_dir,
    x_col="id",
    y_col="has_cactus",
    class_mode="categorical",
    target_size=(32, 32),
    batch_size=32,
    subset="training",
    shuffle=True,
)

validation_generator = generator.flow_from_dataframe(
    dataframe=files_dataframe,
    directory=base_train_dir,
    x_col="id",
    y_col="has_cactus",
    class_mode="categorical",
    target_size=(32, 32),
    batch_size=32,
    subset="validation",
    shuffle=True,
)

noproc_training_generator = noPreprocessGenerator.flow_from_dataframe(
    dataframe=files_dataframe,
    directory=base_train_dir,
    x_col="id",
    y_col="has_cactus",
    class_mode="categorical",
    target_size=(32, 32),
    batch_size=32,
    subset="training",
    shuffle=True,
)

noproc_validation_generator = noPreprocessGenerator.flow_from_dataframe(
    dataframe=files_dataframe,
    directory=base_train_dir,
    x_col="id",
    y_col="has_cactus",
    class_mode="categorical",
    target_size=(32, 32),
    batch_size=32,
    subset="validation",
    shuffle=True,
)

print("class_indices:", training_generator.class_indices)
print("train n:", training_generator.n, "val n:", validation_generator.n)



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1645541903.py in <cell line: 0>()
----> 1 training_generator = generator.flow_from_dataframe(
      2     dataframe=files_dataframe,
      3     directory=base_train_dir,
      4     x_col="id",
      5     y_col="has_cactus",

NameError: name 'generator' is not defined

## === cell 10
import keras
from keras import layers, models

plt.figure(figsize=(36, 12))
imgs, labels = next(validation_generator)
plotindx = 1
for img, label in zip(imgs, labels):
    plt.subplot(7, 5, plotindx)
    plt.imshow(img)
    labl = 0
    if label[0] == 0:
        labl = 1
    plt.title("Label :" + str(labl))
    plotindx += 1



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_11/1503780813.py in <cell line: 0>()
----> 1 import keras
      2 from keras import layers, models
      3 
      4 plt.figure(figsize=(36, 12))
      5 imgs, labels = next(validation_generator)

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
      9 from keras.src.backend.common.stateless_scope import get_stateless_scope
     10 from keras.src.backend.common.stateless_scope import in_stateless_scope
---> 11 from keras.src.utils.module_utils import tensorflow as tf
     12 from keras.src.utils.naming import auto_name
     13 

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

## === cell 11
model = models.Sequential()

model.add(
    layers.Conv2D(
        32, (5, 5), padding="valid", activation="relu", input_shape=(32, 32, 3)
    )
)
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.2))

model.add(layers.Conv2D(64, (3, 3), padding="valid", activation="relu"))
model.add(layers.Conv2D(64, (3, 3), padding="valid", activation="relu"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.2))

model.add(layers.Conv2D(128, (3, 3), padding="valid", activation="relu"))
model.add(layers.Conv2D(128, (3, 3), padding="valid", activation="relu"))
model.add(layers.Dropout(0.2))

model.add(layers.Flatten())
model.add(layers.Dense(256, activation="relu"))
model.add(layers.Dense(256, activation="relu"))
model.add(layers.Dropout(0.1))
model.add(layers.Dense(128, activation="relu"))
model.add(layers.Dense(2, activation="softmax"))

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/53444105.py in <cell line: 0>()
----> 1 model = models.Sequential()
      2 
      3 model.add(
      4     layers.Conv2D(
      5         32, (5, 5), padding="valid", activation="relu", input_shape=(32, 32, 3)

NameError: name 'models' is not defined

## === cell 12
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=5, min_lr=0.001)

CKPT_PATH = "/tmp/checkpoint.keras"
save_best_model = ModelCheckpoint(
    CKPT_PATH,
    monitor="val_accuracy",
    mode="max",
    save_best_only=True,
    verbose=1,
)

steps_per_epoch = max(
    1, int(np.ceil(training_generator.n / training_generator.batch_size))
)
validation_steps = max(
    1, int(np.ceil(validation_generator.n / validation_generator.batch_size))
)
print("steps_per_epoch:", steps_per_epoch, "validation_steps:", validation_steps)



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_11/4022882467.py in <cell line: 0>()
----> 1 from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
      2 
      3 reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=5, min_lr=0.001)
      4 
      5 CKPT_PATH = "/tmp/checkpoint.keras"

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

/usr/local/lib/python3.11/dist-packages/tensorflow/python/trackable/data_structures.py in <module>
     24   from .....third_party import wrapt  # pylint: disable=relative-beyond-top-level
     25 
---> 26 from tensorflow.python.eager import def_function
     27 from tensorflow.python.eager import function as defun
     28 from tensorflow.python.ops import variables

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/def_function.py in <module>
     18 
     19 # Config Options
---> 20 from tensorflow.python.eager.polymorphic_function.eager_function_run import run_functions_eagerly
     21 from tensorflow.python.eager.polymorphic_function.eager_function_run import functions_run_eagerly
     22 

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

## === cell 13
history = model.fit(
    training_generator,
    validation_data=validation_generator,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    verbose=1,
    epochs=4,
    class_weight=class_weights,
    callbacks=[reduce_lr, save_best_model],
)

print("Checkpoint exists after training:", os.path.exists(CKPT_PATH), CKPT_PATH)



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1504224473.py in <cell line: 0>()
----> 1 history = model.fit(
      2     training_generator,
      3     validation_data=validation_generator,
      4     steps_per_epoch=steps_per_epoch,
      5     validation_steps=validation_steps,

NameError: name 'model' is not defined

## === cell 14
test_generator = noAugmentationGenerator.flow_from_directory(
    directory=TEST_EXTRACT_DIR,
    classes=["test"],  # ensures filenames are "test/<id>.jpg"
    class_mode=None,
    target_size=(32, 32),
    batch_size=32,
    shuffle=False,
)
print("test n:", test_generator.n)
print("Example test filenames:", test_generator.filenames[:3])



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/799160124.py in <cell line: 0>()
      1 # Fix: flow_from_directory must point to the directory that contains the "test/" folder.
      2 # Previously it used TEST_EXTRACT_DIR (./test) while extracted images are under ./test/test/.
----> 3 test_generator = noAugmentationGenerator.flow_from_directory(
      4     directory=TEST_EXTRACT_DIR,
      5     classes=["test"],  # ensures filenames are "test/<id>.jpg"

NameError: name 'noAugmentationGenerator' is not defined

## === cell 15
from keras.models import load_model

if os.path.exists(CKPT_PATH):
    model = load_model(CKPT_PATH)

proba = model.predict(test_generator, verbose=1)

if proba.ndim == 2 and proba.shape[1] == 2:
    idx_pos = training_generator.class_indices.get("1", 1)
    has_cactus = proba[:, idx_pos]
else:
    has_cactus = proba.reshape(-1)

sample = pd.read_csv(SAMPLE_SUB, dtype={"id": str})

pred_df = pd.DataFrame(
    {
        "id": [os.path.basename(f) for f in test_generator.filenames],
        "has_cactus": has_cactus.astype("float32"),
    }
)

output = sample[["id"]].merge(pred_df, on="id", how="left")
output["has_cactus"] = output["has_cactus"].astype("float32").fillna(0.5)

output.to_csv("submission.csv", index=False)
print(output.head())
print("Wrote submission.csv with shape:", output.shape)



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_11/2555947183.py in <cell line: 0>()
----> 1 from keras.models import load_model
      2 
      3 if os.path.exists(CKPT_PATH):
      4     model = load_model(CKPT_PATH)
      5 

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

/usr/local/lib/python3.11/dist-packages/tensorflow/python/trackable/data_structures.py in <module>
     24   from .....third_party import wrapt  # pylint: disable=relative-beyond-top-level
     25 
---> 26 from tensorflow.python.eager import def_function
     27 from tensorflow.python.eager import function as defun
     28 from tensorflow.python.ops import variables

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/def_function.py in <module>
     18 
     19 # Config Options
---> 20 from tensorflow.python.eager.polymorphic_function.eager_function_run import run_functions_eagerly
     21 from tensorflow.python.eager.polymorphic_function.eager_function_run import functions_run_eagerly
     22 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/eager_function_run.py in <module>
     16 """Eager semantics for polymorphic function."""
     17 
---> 18 from tensorflow.python.util import deprecation
     19 from tensorflow.python.util.tf_export import tf_export
     20 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/deprecation.py in <module>
     19 import re
     20 
---> 21 from tensorflow.python.framework import strict_mode
     22 from tensorflow.python.platform import tf_logging as logging
     23 from tensorflow.python.util import decorator_utils

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

## === cell 16
import shutil

try:
    shutil.rmtree("test")
except OSError:
    print("Test files already erased or not present")
try:
    shutil.rmtree("training")
except OSError:
    print("Training files already erased or not present")
