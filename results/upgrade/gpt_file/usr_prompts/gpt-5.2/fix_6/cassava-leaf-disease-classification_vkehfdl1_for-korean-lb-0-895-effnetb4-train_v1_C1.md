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
tqdm==4.67.1

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

0.0507

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.6719) has done: 'The timeout is dominated by the Python-level prediction loop: for each test image you run 4 crops × 4 TTA runs = 16 separate `model.predict` calls, causing massive per-call overhead and repeated augmentation work. I keep the exact same cropping + TTA + probability-sum voting logic, but batch all crops and all TTA runs into a single TensorFlow graph execution per image (and run the model directly instead of `predict`) to remove thousands of small Keras predict calls. I also make the scan/crop step avoid extra list/array conversions and enable TF `prefetch`/`cache` where it is provably equivalent for validation. These changes preserve semantics (same augment layers, same number of TTA runs, same summation/vote) while drastically reducing overhead to fit the 600s limit.'
- What this solution (achieved 0.67451) has done: 'The crash happens before any modeling because forcing `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` is incompatible with the installed `protobuf==6.33.0` in this Kaggle environment, triggering the `MessageFactory.GetPrototype` AttributeError during TensorFlow import. I remove that forced protobuf setting (and keep everything else the same) so TensorFlow loads correctly and the pipeline can run end-to-end. Since your current score (0.6719) is already far above the target (0.0507), I not make any score-improving changes; the edits are purely to fix the runtime error and ensure a valid `submission.csv` is produced.'

# 9. Code solution

## === cell 0
import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "cpp")

import random
import warnings

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from sklearn.utils import shuffle

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.applications import EfficientNetB0

from tensorflow.keras.optimizers.schedules import CosineDecay

warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

print("TF:", tf.__version__)

BASE_INPUT = "../input/cassava-leaf-disease-classification"
if not os.path.exists(BASE_INPUT):
    BASE_INPUT = "/kaggle/input/cassava-leaf-disease-classification"
print("BASE_INPUT:", BASE_INPUT)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_11/3596374052.py in <cell line: 0>()
     17 from sklearn.utils import shuffle
     18 
---> 19 import tensorflow as tf
     20 from tensorflow.keras import layers
     21 from tensorflow.keras.layers import Dense, Input

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
training_folder = f"{BASE_INPUT}/train_images/"



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1063145998.py in <cell line: 0>()
----> 1 training_folder = f"{BASE_INPUT}/train_images/"
      2 

NameError: name 'BASE_INPUT' is not defined

## === cell 2
samples_df = pd.read_csv(f"{BASE_INPUT}/train.csv")
samples_df = shuffle(samples_df, random_state=SEED).reset_index(drop=True)
samples_df["filepath"] = training_folder + samples_df["image_id"]
samples_df.head()



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/66674230.py in <cell line: 0>()
----> 1 samples_df = pd.read_csv(f"{BASE_INPUT}/train.csv")
      2 samples_df = shuffle(samples_df, random_state=SEED).reset_index(drop=True)
      3 samples_df["filepath"] = training_folder + samples_df["image_id"]
      4 samples_df.head()
      5 

NameError: name 'BASE_INPUT' is not defined

## === cell 3
training_df = samples_df[:1000].copy()
validation_df = samples_df[1000:1200].copy()



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3563462268.py in <cell line: 0>()
----> 1 training_df = samples_df[:1000].copy()
      2 validation_df = samples_df[1000:1200].copy()
      3 

NameError: name 'samples_df' is not defined

## === cell 4
batch_size = 8
image_size = 512
input_shape = (image_size, image_size, 3)
dropout_rate = 0.4

num_classes = 5

print(
    "Train rows:",
    len(training_df),
    "Val rows:",
    len(validation_df),
    "Classes:",
    num_classes,
)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3996677512.py in <cell line: 0>()
      8 print(
      9     "Train rows:",
---> 10     len(training_df),
     11     "Val rows:",
     12     len(validation_df),

NameError: name 'training_df' is not defined

## === cell 5
training_data = tf.data.Dataset.from_tensor_slices(
    (training_df.filepath.values, training_df.label.values)
)
validation_data = tf.data.Dataset.from_tensor_slices(
    (validation_df.filepath.values, validation_df.label.values)
)




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2312537126.py in <cell line: 0>()
----> 1 training_data = tf.data.Dataset.from_tensor_slices(
      2     (training_df.filepath.values, training_df.label.values)
      3 )
      4 validation_data = tf.data.Dataset.from_tensor_slices(
      5     (validation_df.filepath.values, validation_df.label.values)

NameError: name 'tf' is not defined

## === cell 6
def load_image_and_label_from_path(image_path, label):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (image_size, image_size), method="bilinear")
    img = tf.cast(img, tf.float32) / 255.0
    return img, tf.cast(label, tf.int32)


AUTOTUNE = tf.data.AUTOTUNE

training_data = training_data.map(
    load_image_and_label_from_path, num_parallel_calls=AUTOTUNE
)
validation_data = validation_data.map(
    load_image_and_label_from_path, num_parallel_calls=AUTOTUNE
)



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1419873516.py in <cell line: 0>()
      7 
      8 
----> 9 AUTOTUNE = tf.data.AUTOTUNE
     10 
     11 training_data = training_data.map(

NameError: name 'tf' is not defined

## === cell 7
training_data_batches = (
    training_data.shuffle(buffer_size=1000, seed=SEED)
    .batch(batch_size)
    .prefetch(buffer_size=AUTOTUNE)
)
validation_data_batches = (
    validation_data.cache()
    .shuffle(buffer_size=1000, seed=SEED)
    .batch(batch_size)
    .prefetch(buffer_size=AUTOTUNE)
)



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4176168424.py in <cell line: 0>()
      1 training_data_batches = (
----> 2     training_data.shuffle(buffer_size=1000, seed=SEED)
      3     .batch(batch_size)
      4     .prefetch(buffer_size=AUTOTUNE)
      5 )

NameError: name 'training_data' is not defined

## === cell 8
data_augmentation_layers = tf.keras.Sequential(
    [
        layers.RandomCrop(height=image_size, width=image_size),
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.25),
        layers.RandomZoom((-0.2, 0.0)),
        layers.RandomContrast((0.2, 0.2)),
    ],
    name="train_aug",
)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3076235504.py in <cell line: 0>()
----> 1 data_augmentation_layers = tf.keras.Sequential(
      2     [
      3         layers.RandomCrop(height=image_size, width=image_size),
      4         layers.RandomFlip("horizontal_and_vertical"),
      5         layers.RandomRotation(0.25),

NameError: name 'tf' is not defined

## === cell 9
efficientnet_backbone = EfficientNetB0(
    weights="imagenet",
    include_top=False,
    input_shape=input_shape,
)

inputs = Input(shape=input_shape)
augmented = data_augmentation_layers(inputs)
features = efficientnet_backbone(augmented, training=True)
pooling = layers.GlobalAveragePooling2D()(features)
dropout = layers.Dropout(dropout_rate)(pooling)
outputs = Dense(num_classes, activation="softmax")(dropout)
model = Model(inputs=inputs, outputs=outputs)

model.summary()



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3404337033.py in <cell line: 0>()
----> 1 efficientnet_backbone = EfficientNetB0(
      2     weights="imagenet",
      3     include_top=False,
      4     input_shape=input_shape,
      5 )

NameError: name 'EfficientNetB0' is not defined

## === cell 10
epochs = 15
decay_steps = int(round(len(training_df) / batch_size)) * epochs
cosine_decay = CosineDecay(
    initial_learning_rate=1e-4, decay_steps=decay_steps, alpha=0.3
)

callbacks = [
    ModelCheckpoint(
        filepath="best_model.keras", monitor="val_loss", save_best_only=True
    )
]

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=tf.keras.optimizers.Adam(learning_rate=cosine_decay),
    metrics=["accuracy"],
)



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2705035582.py in <cell line: 0>()
      1 epochs = 15
----> 2 decay_steps = int(round(len(training_df) / batch_size)) * epochs
      3 cosine_decay = CosineDecay(
      4     initial_learning_rate=1e-4, decay_steps=decay_steps, alpha=0.3
      5 )

NameError: name 'training_df' is not defined

## === cell 11
history = model.fit(
    training_data_batches,
    epochs=epochs,
    validation_data=validation_data_batches,
    callbacks=callbacks,
)

model = tf.keras.models.load_model("best_model.keras")




## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1142442891.py in <cell line: 0>()
----> 1 history = model.fit(
      2     training_data_batches,
      3     epochs=epochs,
      4     validation_data=validation_data_batches,
      5     callbacks=callbacks,

NameError: name 'model' is not defined

## === cell 12
def scan_over_image(img_path, crop_size=512):
    """
    Extract 512x512 images covering the whole original image (4 crops).
    """
    with Image.open(img_path) as im:
        im = im.convert("RGB")
        img = np.asarray(im)

    img_height, img_width = img.shape[0], img.shape[1]

    pad_h = max(0, crop_size - img_height)
    pad_w = max(0, crop_size - img_width)
    if pad_h > 0 or pad_w > 0:
        img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
        img_height, img_width = img.shape[0], img.shape[1]

    x0 = 0
    x1 = img_width - crop_size
    y0 = 0
    y1 = img_height - crop_size

    crops = np.empty((4, crop_size, crop_size, 3), dtype=img.dtype)
    crops[0] = img[y0 : y0 + crop_size, x0 : x0 + crop_size, :]
    crops[1] = img[y1 : y1 + crop_size, x0 : x0 + crop_size, :]
    crops[2] = img[y0 : y0 + crop_size, x1 : x1 + crop_size, :]
    crops[3] = img[y1 : y1 + crop_size, x1 : x1 + crop_size, :]
    return crops




## === cell 13
test_time_augmentation_layers = tf.keras.Sequential(
    [
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomZoom((-0.2, 0.2)),
        layers.RandomContrast((0.2, 0.2)),
    ],
    name="tta_aug",
)




## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3111700228.py in <cell line: 0>()
----> 1 test_time_augmentation_layers = tf.keras.Sequential(
      2     [
      3         layers.RandomFlip("horizontal_and_vertical"),
      4         layers.RandomZoom((-0.2, 0.2)),
      5         layers.RandomContrast((0.2, 0.2)),

NameError: name 'tf' is not defined

## === cell 14
@tf.function(reduce_retracing=True)
def _tta_predict_sum(batch_images, tta_runs: tf.Tensor):
    n = tf.shape(batch_images)[0]
    probs_sum = tf.zeros((n, num_classes), dtype=tf.float32)

    for _ in tf.range(tta_runs):
        aug = test_time_augmentation_layers(batch_images, training=True)
        probs_sum += model(aug, training=False)
    return probs_sum


def predict_and_vote(image_filename, folder, TTA_runs=4):
    """
    Run the model over 4 local areas of the given image, with light TTA, then vote by summed probabilities.
    """
    local_images = scan_over_image(
        os.path.join(folder, image_filename), crop_size=image_size
    )
    local_images = tf.convert_to_tensor(local_images, dtype=tf.float32) / 255.0

    probs_sum_per_crop = _tta_predict_sum(
        local_images, tf.constant(TTA_runs, dtype=tf.int32)
    )

    global_predictions = tf.reduce_sum(probs_sum_per_crop, axis=0)
    final_prediction = int(tf.argmax(global_predictions, axis=-1).numpy())
    return final_prediction




## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/820840435.py in <cell line: 0>()
----> 1 @tf.function(reduce_retracing=True)
      2 def _tta_predict_sum(batch_images, tta_runs: tf.Tensor):
      3     n = tf.shape(batch_images)[0]
      4     probs_sum = tf.zeros((n, num_classes), dtype=tf.float32)
      5 

NameError: name 'tf' is not defined

## === cell 15
def run_predictions_over_image_list(image_list, folder):
    predictions = []
    with tqdm(total=len(image_list), mininterval=0.5) as pbar:
        for image_filename in image_list:
            predictions.append(predict_and_vote(image_filename, folder))
            pbar.update(1)
    return predictions




## === cell 16
map_path = f"{BASE_INPUT}/label_num_to_disease_map.json"
with open(map_path, "r") as f:
    print(f.read())



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/583559851.py in <cell line: 0>()
----> 1 map_path = f"{BASE_INPUT}/label_num_to_disease_map.json"
      2 with open(map_path, "r") as f:
      3     print(f.read())
      4 

NameError: name 'BASE_INPUT' is not defined

## === cell 17
test_folder = f"{BASE_INPUT}/test_images/"

test_files = sorted([f for f in os.listdir(test_folder) if f.lower().endswith(".jpg")])

submission_df = pd.DataFrame({"image_id": test_files})
submission_df["label"] = 0
submission_df.head()



## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3017771018.py in <cell line: 0>()
----> 1 test_folder = f"{BASE_INPUT}/test_images/"
      2 
      3 test_files = sorted([f for f in os.listdir(test_folder) if f.lower().endswith(".jpg")])
      4 
      5 submission_df = pd.DataFrame({"image_id": test_files})

NameError: name 'BASE_INPUT' is not defined

## === cell 18
submission_df["label"] = run_predictions_over_image_list(
    submission_df["image_id"].tolist(), test_folder
)

submission_df["label"] = submission_df["label"].astype(int)
submission_df = submission_df[["image_id", "label"]]

submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)
print("Wrote", submission_path, "with shape", submission_df.shape)
print(submission_df.head())

## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3327521904.py in <cell line: 0>()
      1 submission_df["label"] = run_predictions_over_image_list(
----> 2     submission_df["image_id"].tolist(), test_folder
      3 )
      4 
      5 submission_df["label"] = submission_df["label"].astype(int)

NameError: name 'submission_df' is not defined
