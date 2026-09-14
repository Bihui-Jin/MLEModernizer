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
Classify each cassava image into four disease categories or a fifth category indicating a healthy leaf.

## Metric
Categorization accuracy.

## Submission Format
```
image_id,label
1000471002.jpg,4
1000840542.jpg,4
etc.
```

## Dataset
**[train/test]_images** the image files.

**train.csv**

- `image_id` the image file name.

- `label` the ID code for the disease.

**sample_submission.csv** A properly formatted sample submission, given the disclosed test set content.

- `image_id` the image file name.

- `label` the predicted ID code for the disease.

**[train/test]_tfrecords** the image files in tfrecord format.

**label_num_to_disease_map.json** The mapping between each disease code and the real disease name.

# 2. Python version

3.9

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        input/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        working/
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
```

-> data/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/cassava-leaf-disease-classification/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/cassava-leaf-disease-classification/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> data/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> input/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> (stopped after 10 files for performance)

# 5. Target score

0.8735267452402539

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.17862) has done: 'I set the protobuf implementation to the pure‑Python version before importing TensorFlow to avoid the “MessageFactory” error, replace the missing model load with building a MobileNetV2 model that uses ImageNet weights, train it briefly on the provided data, and then generate predictions so a proper `submission.csv` is written.'
- What this solution (achieved 0.05979) has done: 'We speed up the pipeline by enabling mixed‑precision (GPU‑friendly) computation and by parallelising data loading during training – both keep the exact model, augmentations, and training schedule intact, just make each epoch run faster.'
- What this solution (achieved 0.12892) has done: 'The script is sped up by parallelizing data loading during training and by batch‑predicting all test images instead of running a separate prediction for each file. These changes keep the exact model, hyper‑parameters, and augmentation pipeline, so the trained model and its predictions remain unchanged while eliminating the most time‑consuming Python‑level loops.'
- What this solution (achieved 0.41517) has done: 'The changes add multi‑process image loading to the generators and the training loop, which drastically reduces data‑augmentation overhead without altering the model, loss, or training schedule. By using all CPU cores (`workers=os.cpu_count()`) and `use_multiprocessing=True`, each batch is prepared in parallel, keeping the same augmentations, epochs, and architecture, so the final predictions remain identical. Only the necessary imports and arguments are added; all other logic stays untouched.'
- What this solution (achieved 0.10426) has done: 'The fix moves the protobuf environment setting to the very first cell, updates the mixed‑precision imports to the current TensorFlow API, and ensures `multiprocessing` is imported before it’s used so the training call can access `cpu_count()`. These minimal changes remove the import error and allow the script to run end‑to‑end, producing a valid `submission.csv` while keeping the original model and training logic unchanged.'
- What this solution (achieved 0.10314) has done: 'The changes add multiprocessing for model fitting, set explicit random seeds for reproducibility, and replace the Python loop that loads test images with a parallel `tf.data` pipeline that applies the same resizing and scaling. These adjustments keep the exact model architecture, data augmentations, and training procedure while removing the most time‑consuming I/O bottlenecks, allowing the script to finish well within the 600‑second limit.'

# 9. Code solution

## === cell 0
import os
import random

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "cpp"

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import applications
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.mixed_precision import Policy, set_global_policy
import multiprocessing  # retained for potential future use

seed = 42
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

policy = Policy("mixed_float16")
set_global_policy(policy)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_11/3369773692.py in <cell line: 0>()
      7 import pandas as pd
      8 import numpy as np
----> 9 import tensorflow as tf
     10 from tensorflow.keras import applications
     11 from tensorflow.keras.preprocessing.image import ImageDataGenerator

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
train_csv_path = "../input/cassava-leaf-disease-classification/train.csv"
label_json_path = (
    "../input/cassava-leaf-disease-classification/label_num_to_disease_map.json"
)
images_dir_path = "../input/cassava-leaf-disease-classification/train_images"



## === cell 2
train_csv = pd.read_csv(train_csv_path)
train_csv["label"] = train_csv["label"].astype("string")

label_class = pd.read_json(label_json_path, orient="index")
label_class = (
    label_class.values.flatten().tolist()
)  # not used further but kept for compatibility



## === cell 3
IMG_SIZE = 288
BATCH_SIZE = 12
EPOCHS = 30  # increased epochs for better learning
lr = 1e-4  # slightly higher LR for fine‑tuning



## === cell 4
train_gen = ImageDataGenerator(
    rotation_range=270,
    width_shift_range=0.2,
    height_shift_range=0.2,
    brightness_range=[0.1, 0.9],
    shear_range=25,
    zoom_range=0.3,
    channel_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
    rescale=1 / 255,
    validation_split=0.2,
)

valid_gen = ImageDataGenerator(rescale=1 / 255, validation_split=0.2)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3160226.py in <cell line: 0>()
----> 1 train_gen = ImageDataGenerator(
      2     rotation_range=270,
      3     width_shift_range=0.2,
      4     height_shift_range=0.2,
      5     brightness_range=[0.1, 0.9],

NameError: name 'ImageDataGenerator' is not defined

## === cell 5
train_generator = train_gen.flow_from_dataframe(
    dataframe=train_csv,
    directory=images_dir_path,
    x_col="image_id",
    y_col="label",
    target_size=(IMG_SIZE, IMG_SIZE),
    class_mode="categorical",
    batch_size=BATCH_SIZE,
    shuffle=True,
    subset="training",
    seed=seed,
)

valid_generator = valid_gen.flow_from_dataframe(
    dataframe=train_csv,
    directory=images_dir_path,
    x_col="image_id",
    y_col="label",
    target_size=(IMG_SIZE, IMG_SIZE),
    class_mode="categorical",
    batch_size=BATCH_SIZE,
    shuffle=False,
    subset="validation",
    seed=seed,
)



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/546695600.py in <cell line: 0>()
----> 1 train_generator = train_gen.flow_from_dataframe(
      2     dataframe=train_csv,
      3     directory=images_dir_path,
      4     x_col="image_id",
      5     y_col="label",

NameError: name 'train_gen' is not defined

## === cell 6
PRECISION = Precision()
RECALL = Recall()


def F1_score(y_true, y_pred):
    PRECISION.reset_state()
    RECALL.reset_state()
    PRECISION.update_state(y_true, y_pred)
    RECALL.update_state(y_true, y_pred)
    p = PRECISION.result()
    r = RECALL.result()
    return 2 * ((p * r) / (p + r + 1e-23))




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2649388661.py in <cell line: 0>()
----> 1 PRECISION = Precision()
      2 RECALL = Recall()
      3 
      4 
      5 def F1_score(y_true, y_pred):

NameError: name 'Precision' is not defined

## === cell 7
BASE0 = applications.MobileNetV2(
    include_top=False,
    input_shape=[IMG_SIZE, IMG_SIZE, 3],
    weights="imagenet",
    pooling="avg",
)
BASE0.trainable = True  # unfreeze for better learning


def build_model():
    model = tf.keras.Sequential([BASE0, Dropout(0.5), Dense(5, activation="softmax")])
    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9),
        metrics=["accuracy"],
    )
    return model




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2997141711.py in <cell line: 0>()
----> 1 BASE0 = applications.MobileNetV2(
      2     include_top=False,
      3     input_shape=[IMG_SIZE, IMG_SIZE, 3],
      4     weights="imagenet",
      5     pooling="avg",

NameError: name 'applications' is not defined

## === cell 8
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    "CassavaLeafDiseaseModel.h5", monitor="val_loss", save_best_only=True
)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1833660544.py in <cell line: 0>()
----> 1 checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
      2     "CassavaLeafDiseaseModel.h5", monitor="val_loss", save_best_only=True
      3 )
      4 

NameError: name 'tf' is not defined

## === cell 9
model0 = build_model()
model0.fit(
    train_generator,
    validation_data=valid_generator,
    epochs=EPOCHS,
    callbacks=[checkpoint_cb],
    verbose=2,
)  # removed unsupported workers arguments



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3381855056.py in <cell line: 0>()
----> 1 model0 = build_model()
      2 model0.fit(
      3     train_generator,
      4     validation_data=valid_generator,
      5     epochs=EPOCHS,

NameError: name 'build_model' is not defined

## === cell 10
if os.path.exists("CassavaLeafDiseaseModel.h5"):
    model0.load_weights("CassavaLeafDiseaseModel.h5")

ss = pd.read_csv("../input/cassava-leaf-disease-classification/sample_submission.csv")
test_images_dir = "../input/cassava-leaf-disease-classification/test_images"

test_paths = ss["image_id"].apply(lambda x: os.path.join(test_images_dir, x)).values

test_ds = tf.data.Dataset.from_tensor_slices(test_paths)


def _load_and_preprocess(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = img / 255.0
    return img


test_ds = test_ds.map(_load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
test_ds = test_ds.batch(BATCH_SIZE * 4).prefetch(tf.data.AUTOTUNE)

preds_batch = model0.predict(test_ds, verbose=0)
preds = np.argmax(preds_batch, axis=1).astype(int).tolist()

my_submission = pd.DataFrame({"image_id": ss.image_id, "label": preds})
my_submission.to_csv("submission.csv", index=False)



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2846741833.py in <cell line: 0>()
      7 test_paths = ss["image_id"].apply(lambda x: os.path.join(test_images_dir, x)).values
      8 
----> 9 test_ds = tf.data.Dataset.from_tensor_slices(test_paths)
     10 
     11 

NameError: name 'tf' is not defined

## === cell 11
print("Submission File: \n---------------\n")
print(my_submission.head())

## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3014216753.py in <cell line: 0>()
      1 print("Submission File: \n---------------\n")
----> 2 print(my_submission.head())

NameError: name 'my_submission' is not defined
