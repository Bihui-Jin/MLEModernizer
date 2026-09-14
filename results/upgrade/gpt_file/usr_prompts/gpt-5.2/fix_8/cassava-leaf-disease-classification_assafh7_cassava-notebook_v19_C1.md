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

albumentations==2.0.8
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
pillow==11.3.0
protobuf==6.33.0
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

0.0018

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.2201) has done: 'The timeout is dominated by slow Python-side image loading/augmentation in `ImageDataGenerator.flow_from_dataframe` plus long training (up to 40 epochs) on a heavy backbone. To keep the exact same model and training semantics while reducing wall time, I (1) switch to Keras’ `tf.data`-based `image_dataset_from_directory` with the same 80/20 split and the same augmentation operations implemented as TensorFlow graph ops (so the augmentation *logic* is unchanged), (2) add caching/prefetching and parallel decoding, and (3) avoid any extra one-off batches/plots that trigger I/O. This keeps the architecture, loss, optimizer, early stopping, epochs, and evaluation logic intact, but removes the major input pipeline bottleneck that commonly causes >10 minute runs.'
- What this solution (achieved 0.2201) has done: 'I fix two runtime blockers: the protobuf/TensorFlow import crash caused by forcing the pure-Python protobuf implementation, and the `tf.data` augmentation error caused by passing `AUTOTUNE` into `tf.map_fn(parallel_iterations=...)` (must be a positive int). I keep your exact model, optimizer, loss, training loop, and `image_dataset_from_directory` approach unchanged, only making the minimum adjustments needed for it to run end-to-end. After these fixes, `train_datagen_flow`/`valid_datagen_flow` be defined correctly so training and prediction complete, and a valid `submission.csv` with the required columns be written to `/kaggle/working`. These changes are score-neutral in intent (they just restore the pipeline so it can train/predict as designed).'
- What this solution (achieved 0.76196) has done: 'I fix the TensorFlow/protobuf import crash by pinning protobuf to the pure-Python implementation *before* importing TensorFlow (this avoids the `MessageFactory.GetPrototype` mismatch seen with protobuf 6.x). Then I fix the `NotFoundError` in the `image_dataset_from_directory` pipeline by avoiding fragile symlinks and instead copying only missing images into the label-folders (Kaggle’s environment can make symlink targets disappear/not resolve as expected). These changes keep your exact model, optimizer, loss, epochs, and `tf.data` augmentation logic intact; they only unblock I/O and restore end-to-end training/prediction. Finally, I ensure `submission.csv` is always written with the required columns and row count.'

# 9. Code solution

## === cell 0
import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "cpp")

import json
import warnings
import shutil

import numpy as np
import pandas as pd

import cv2
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import albumentations as aug

warnings.simplefilter("ignore")

SEED = 42
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

try:
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except Exception:
    pass

base_walk = "/kaggle/input"
for dirname, _, filenames in os.walk(base_walk):
    for filename in filenames[:3]:
        print(os.path.join(dirname, filename))
    break

print("TensorFlow:", tf.__version__)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_11/2289737720.py in <cell line: 0>()
     18 import seaborn as sns
     19 
---> 20 import tensorflow as tf
     21 from tensorflow.keras.layers import Dropout, Dense
     22 from tensorflow.keras.models import Sequential

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
candidate_paths = [
    "/kaggle/input/cassava-leaf-disease-classification",
    "/kaggle/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
    "../input/cassava-leaf-disease-classification",
]
general_path = None
for p in candidate_paths:
    if os.path.exists(p):
        general_path = p if p.endswith("/") else p + "/"
        break

if general_path is None:
    raise FileNotFoundError(
        "Could not locate cassava dataset directory. Checked: "
        + ", ".join(candidate_paths)
    )

print("Using general_path:", general_path)
print("Exists:", os.path.exists(general_path))
print(os.listdir(general_path)[:10])



## === cell 2
with open(os.path.join(general_path, "label_num_to_disease_map.json"), "r") as file:
    map_classes = json.loads(file.read())
map_classes = {int(k): v for k, v in map_classes.items()}
print(json.dumps(map_classes, indent=4))



## === cell 3
input_files = os.listdir(os.path.join(general_path, "train_images"))
print(f"Number of train images: {len(input_files)}")



## === cell 4
img_shapes = {}
print("Skipping image shape scan for performance (not used downstream).")



## === cell 5
df_train = pd.read_csv(os.path.join(general_path, "train.csv"))
df_train["class_name"] = df_train["label"].map(map_classes)
df_train.head()



## === cell 6
print("Skipping class distribution plot for performance (not used downstream).")




## === cell 7
def visualize_batch(image_ids, labels, class_names):
    plt.figure(figsize=(16, 12))
    for ind, (image_id, label, class_name) in enumerate(
        zip(image_ids, labels, class_names)
    ):
        if ind >= 9:
            break
        plt.subplot(3, 3, ind + 1)
        img = cv2.imread(os.path.join(general_path, "train_images", image_id))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.title(f"Class {label}:  {class_name}", fontsize=12)
        plt.axis("off")
    plt.show()




## === cell 8
print("Skipping class-0 sample visualization for performance.")



## === cell 9
print("Skipping class-1 sample visualization for performance.")



## === cell 10
print("Skipping class-2 sample visualization for performance.")



## === cell 11
print("Skipping class-3 sample visualization for performance.")



## === cell 12
print("Skipping class-4 sample visualization for performance.")




## === cell 13
def plot_augmentation(image_id, transform):
    plt.figure(figsize=(12, 12))
    img = cv2.imread(os.path.join(general_path, "train_images", image_id))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.subplot(2, 2, 1)
    plt.imshow(img)
    plt.axis("off")
    plt.title("original")

    for i in range(2, 5):
        plt.subplot(2, 2, i)
        x = transform(image=img)["image"]
        plt.imshow(x)
        plt.axis("off")
        plt.title(f"Augmentation -{i-1}")

    plt.show()




## === cell 14
transform_shift_scale_rotate = aug.ShiftScaleRotate(
    p=1.0,
    shift_limit=(-0.3, 0.3),
    scale_limit=(-0.1, 0.1),
    rotate_limit=(-180, 180),
    interpolation=0,
    border_mode=4,
)
print("Skipping augmentation demo (ShiftScaleRotate).")



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2281295473.py in <cell line: 0>()
----> 1 transform_shift_scale_rotate = aug.ShiftScaleRotate(
      2     p=1.0,
      3     shift_limit=(-0.3, 0.3),
      4     scale_limit=(-0.1, 0.1),
      5     rotate_limit=(-180, 180),

NameError: name 'aug' is not defined

## === cell 15
transform_coarse_dropout = aug.CoarseDropout(
    p=1.0,
    max_holes=100,
    max_height=50,
    max_width=50,
    min_holes=30,
    min_height=20,
    min_width=20,
)
print("Skipping augmentation demo (CoarseDropout).")



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3987900503.py in <cell line: 0>()
----> 1 transform_coarse_dropout = aug.CoarseDropout(
      2     p=1.0,
      3     max_holes=100,
      4     max_height=50,
      5     max_width=50,

NameError: name 'aug' is not defined

## === cell 16
transform_hsv = aug.HueSaturationValue(
    hue_shift_limit=0,
    sat_shift_limit=(40, 80),
    val_shift_limit=(40, 80),
    p=1.0,
)
print("Skipping augmentation demo (HueSaturationValue).")



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3810376697.py in <cell line: 0>()
----> 1 transform_hsv = aug.HueSaturationValue(
      2     hue_shift_limit=0,
      3     sat_shift_limit=(40, 80),
      4     val_shift_limit=(40, 80),
      5     p=1.0,

NameError: name 'aug' is not defined

## === cell 17
transform_clahe = aug.CLAHE(
    p=1.0,
    clip_limit=(10, 30),
    tile_grid_size=(10, 10),
)
print("Skipping augmentation demo (CLAHE).")



## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3952940614.py in <cell line: 0>()
----> 1 transform_clahe = aug.CLAHE(
      2     p=1.0,
      3     clip_limit=(10, 30),
      4     tile_grid_size=(10, 10),
      5 )

NameError: name 'aug' is not defined

## === cell 18
transform_fog = aug.RandomFog(p=1.0)
print("Skipping augmentation demo (RandomFog).")



## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/670643645.py in <cell line: 0>()
----> 1 transform_fog = aug.RandomFog(p=1.0)
      2 print("Skipping augmentation demo (RandomFog).")
      3 

NameError: name 'aug' is not defined

## === cell 19
transform_sunflare = aug.RandomSunFlare(p=1.0)
print("Skipping augmentation demo (RandomSunFlare).")



## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2837337668.py in <cell line: 0>()
----> 1 transform_sunflare = aug.RandomSunFlare(p=1.0)
      2 print("Skipping augmentation demo (RandomSunFlare).")
      3 

NameError: name 'aug' is not defined

## === cell 20
transform_brightness_contrast = aug.RandomBrightnessContrast(p=1.0)
print("Skipping augmentation demo (RandomBrightnessContrast).")



## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2271824893.py in <cell line: 0>()
----> 1 transform_brightness_contrast = aug.RandomBrightnessContrast(p=1.0)
      2 print("Skipping augmentation demo (RandomBrightnessContrast).")
      3 

NameError: name 'aug' is not defined

## === cell 21
transform_randomcrop = aug.RandomCrop(p=1.0, height=512, width=512)
print("Skipping augmentation demo (RandomCrop).")



## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3314888882.py in <cell line: 0>()
----> 1 transform_randomcrop = aug.RandomCrop(p=1.0, height=512, width=512)
      2 print("Skipping augmentation demo (RandomCrop).")
      3 

NameError: name 'aug' is not defined

## === cell 22
transform_rgbshift = aug.RGBShift(p=1.0)
print("Skipping augmentation demo (RGBShift).")



## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1753909571.py in <cell line: 0>()
----> 1 transform_rgbshift = aug.RGBShift(p=1.0)
      2 print("Skipping augmentation demo (RGBShift).")
      3 

NameError: name 'aug' is not defined

## === cell 23
transform_snow = aug.RandomSnow(p=1.0)
print("Skipping augmentation demo (RandomSnow).")



## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3767122330.py in <cell line: 0>()
----> 1 transform_snow = aug.RandomSnow(p=1.0)
      2 print("Skipping augmentation demo (RandomSnow).")
      3 

NameError: name 'aug' is not defined

## === cell 24
transform_hflip = aug.HorizontalFlip(p=1.0)
print("Skipping augmentation demo (HorizontalFlip).")



## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/96173452.py in <cell line: 0>()
----> 1 transform_hflip = aug.HorizontalFlip(p=1.0)
      2 print("Skipping augmentation demo (HorizontalFlip).")
      3 

NameError: name 'aug' is not defined

## === cell 25
transform_vflip = aug.VerticalFlip(p=1.0)
print("Skipping augmentation demo (VerticalFlip).")



## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2828069834.py in <cell line: 0>()
----> 1 transform_vflip = aug.VerticalFlip(p=1.0)
      2 print("Skipping augmentation demo (VerticalFlip).")
      3 

NameError: name 'aug' is not defined

## === cell 26
transform_transpose = aug.Transpose(p=1.0)
print("Skipping augmentation demo (Transpose).")



## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/698223049.py in <cell line: 0>()
----> 1 transform_transpose = aug.Transpose(p=1.0)
      2 print("Skipping augmentation demo (Transpose).")
      3 

NameError: name 'aug' is not defined

## === cell 27
img_width, img_height = 224, 224

train = pd.read_csv(os.path.join(general_path, "train.csv"))
train["label"] = train["label"].astype("string")
train.head()



## === cell 28
AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 64
VAL_SPLIT = 0.2

train_images_dir = os.path.join(general_path, "train_images")

work_root = "/kaggle/working"
symlink_root = os.path.join(work_root, "train_images_by_label")
os.makedirs(symlink_root, exist_ok=True)

for lbl in map(str, range(5)):
    os.makedirs(os.path.join(symlink_root, lbl), exist_ok=True)

copied = 0
missing_sources = 0
for image_id, lbl in zip(train["image_id"].values, train["label"].values):
    src = os.path.join(train_images_dir, image_id)
    dst = os.path.join(symlink_root, lbl, image_id)
    if not os.path.exists(src):
        missing_sources += 1
        continue
    if not os.path.exists(dst):
        shutil.copy2(src, dst)
        copied += 1

print(f"Prepared label folders at: {symlink_root}")
print(f"Copied {copied} images (missing sources: {missing_sources}).")

train_ds = tf.keras.utils.image_dataset_from_directory(
    symlink_root,
    labels="inferred",
    label_mode="categorical",
    batch_size=BATCH_SIZE,
    image_size=(img_width, img_height),
    shuffle=True,
    seed=SEED,
    validation_split=VAL_SPLIT,
    subset="training",
)

valid_ds = tf.keras.utils.image_dataset_from_directory(
    symlink_root,
    labels="inferred",
    label_mode="categorical",
    batch_size=BATCH_SIZE,
    image_size=(img_width, img_height),
    shuffle=False,
    seed=SEED,
    validation_split=VAL_SPLIT,
    subset="validation",
)

SHEAR_RANGE = 0.2
ZOOM_RANGE = 0.2


def _augment(images, labels):
    images = tf.cast(images, tf.float32)

    images = tf.image.random_flip_left_right(images, seed=SEED)
    images = tf.image.random_flip_up_down(images, seed=SEED)

    batch_size = tf.shape(images)[0]
    zoom = tf.random.stateless_uniform(
        [batch_size, 1, 1, 1],
        seed=tf.stack([SEED, 123]),
        minval=1.0 - ZOOM_RANGE,
        maxval=1.0 + ZOOM_RANGE,
        dtype=tf.float32,
    )
    in_h = tf.cast(img_height, tf.float32)
    in_w = tf.cast(img_width, tf.float32)
    new_h = tf.cast(tf.round(in_h / zoom[:, 0, 0, 0]), tf.int32)
    new_w = tf.cast(tf.round(in_w / zoom[:, 0, 0, 0]), tf.int32)

    def _zoom_one(img, nh, nw):
        nh = tf.clip_by_value(nh, 1, img_height)
        nw = tf.clip_by_value(nw, 1, img_width)
        img2 = tf.image.resize_with_crop_or_pad(img, nh, nw)
        img2 = tf.image.resize(img2, [img_height, img_width], method="bilinear")
        return img2

    images = tf.map_fn(
        lambda t: _zoom_one(t[0], t[1], t[2]),
        (images, new_h, new_w),
        fn_output_signature=tf.float32,
        parallel_iterations=16,
    )

    shear = tf.random.stateless_uniform(
        [batch_size],
        seed=tf.stack([SEED, 456]),
        minval=-SHEAR_RANGE,
        maxval=SHEAR_RANGE,
        dtype=tf.float32,
    )
    a0 = tf.ones_like(shear)
    a1 = -shear
    a2 = tf.zeros_like(shear)
    b0 = tf.zeros_like(shear)
    b1 = tf.ones_like(shear)
    b2 = tf.zeros_like(shear)
    c0 = tf.zeros_like(shear)
    c1 = tf.zeros_like(shear)
    transforms = tf.stack([a0, a1, a2, b0, b1, b2, c0, c1], axis=1)

    images = tf.raw_ops.ImageProjectiveTransformV3(
        images=images,
        transforms=transforms,
        output_shape=tf.constant([img_height, img_width], dtype=tf.int32),
        interpolation="BILINEAR",
        fill_mode="REFLECT",
        fill_value=0.0,
    )
    return images, labels


train_ds = (
    train_ds.map(_augment, num_parallel_calls=AUTOTUNE).cache().prefetch(AUTOTUNE)
)
valid_ds = valid_ds.cache().prefetch(AUTOTUNE)

train_datagen_flow = train_ds
valid_datagen_flow = valid_ds



## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1868295559.py in <cell line: 0>()
----> 1 AUTOTUNE = tf.data.AUTOTUNE
      2 BATCH_SIZE = 64
      3 VAL_SPLIT = 0.2
      4 
      5 train_images_dir = os.path.join(general_path, "train_images")

NameError: name 'tf' is not defined

## === cell 29
x, y = next(iter(train_datagen_flow.take(1)))
print("Batch X shape:", x.shape, "Batch y shape:", y.shape)



## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3702806991.py in <cell line: 0>()
----> 1 x, y = next(iter(train_datagen_flow.take(1)))
      2 print("Batch X shape:", x.shape, "Batch y shape:", y.shape)
      3 

NameError: name 'train_datagen_flow' is not defined

## === cell 30
from tensorflow.keras.applications import EfficientNetB0

backbone = EfficientNetB0(
    weights="imagenet",
    input_shape=(img_width, img_height, 3),
    pooling="avg",
    include_top=False,
)
backbone.summary()



## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_11/3204427137.py in <cell line: 0>()
----> 1 from tensorflow.keras.applications import EfficientNetB0
      2 
      3 backbone = EfficientNetB0(
      4     weights="imagenet",
      5     input_shape=(img_width, img_height, 3),

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

## === cell 31
opt = Adam(learning_rate=5e-4)

model = Sequential()
model.add(backbone)
model.add(Dropout(0.25))
model.add(Dense(5, activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
model.summary()



## --- ERROR in cell 31, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2312972228.py in <cell line: 0>()
----> 1 opt = Adam(learning_rate=5e-4)
      2 
      3 model = Sequential()
      4 model.add(backbone)
      5 model.add(Dropout(0.25))

NameError: name 'Adam' is not defined

## === cell 32
from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)



## --- ERROR in cell 32, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_11/1749115285.py in <cell line: 0>()
----> 1 from tensorflow.keras.callbacks import EarlyStopping
      2 
      3 early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
      4 

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

## === cell 33
history = model.fit(
    train_datagen_flow,
    validation_data=valid_datagen_flow,
    epochs=40,
    verbose=1,
    callbacks=[early_stop],
)



## --- ERROR in cell 33, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2408604926.py in <cell line: 0>()
----> 1 history = model.fit(
      2     train_datagen_flow,
      3     validation_data=valid_datagen_flow,
      4     epochs=40,
      5     verbose=1,

NameError: name 'model' is not defined

## === cell 34
model.save("submission.h5")



## --- ERROR in cell 34, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3691005991.py in <cell line: 0>()
----> 1 model.save("submission.h5")
      2 

NameError: name 'model' is not defined

## === cell 35
print("Skipping training curves plot for performance.")



## === cell 36
ss = pd.read_csv(os.path.join(general_path, "sample_submission.csv"))
test_images_dir = os.path.join(general_path, "test_images")

test_paths = tf.constant(
    [os.path.join(test_images_dir, x) for x in ss["image_id"].values]
)


def _load_test(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [img_width, img_height], method="bilinear")
    img = tf.cast(img, tf.float32)
    return img


test_ds = (
    tf.data.Dataset.from_tensor_slices(test_paths)
    .map(_load_test, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

pred_proba = model.predict(test_ds, verbose=1)
preds = np.argmax(pred_proba, axis=1).astype(int)

my_submission = pd.DataFrame({"image_id": ss["image_id"].values, "label": preds})
my_submission.to_csv("submission.csv", index=False)

print(my_submission.head())
print("Wrote submission.csv with rows:", len(my_submission))
print("submission.csv exists:", os.path.exists("submission.csv"))

## --- ERROR in cell 36, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3209397974.py in <cell line: 0>()
      2 test_images_dir = os.path.join(general_path, "test_images")
      3 
----> 4 test_paths = tf.constant(
      5     [os.path.join(test_images_dir, x) for x in ss["image_id"].values]
      6 )

NameError: name 'tf' is not defined
