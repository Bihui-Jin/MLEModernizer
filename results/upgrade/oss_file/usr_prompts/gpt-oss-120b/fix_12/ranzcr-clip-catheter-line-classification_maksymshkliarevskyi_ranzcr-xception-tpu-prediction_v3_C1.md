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
Detect the presence and position of catheters and lines on chest x-rays.

## Metric
Area under the ROC curve for each label, with the final score being the average of the individual AUCs of each predicted column.

## Submission Format
For each ID in the test set, you must predict a probability for all target variables. The file should contain a header and have the following format:
```
StudyInstanceUID,ETT - Abnormal,ETT - Borderline,ETT - Normal,NGT - Abnormal,NGT - Borderline,NGT - Incompletely Imaged,NGT - Normal,CVC - Abnormal,CVC - Borderline,CVC - Normal,Swan Ganz Catheter Present
1.2.826.0.1.3680043.8.498.62451881164053375557257228990443168843,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.83721761279899623084220697845011427274,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.12732270010839808189235995393981377825,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.11769539755086084996287023095028033598,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.87838627504097587943394933987052577153,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.53211840524738036417560823327351887819,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.93555795394184819372299157360228027866,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.52241894131170494723503100795076463919,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.36500167484503936720548852591033878284,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.86199852603457900780565655267977637728,0,0,0,0,0,0,0,0,0,0,0
```

## Dataset
`train.csv` contains image IDs, binary labels, and patient IDs.

TFRecords are available for both train and test.

We've also included `train_annotations.csv`. These are segmentation annotations for training samples that have them. They are included solely as additional information for competitors.

- train.csv - contains image IDs, binary labels, and patient IDs.
- sample_submission.csv - a sample submission file in the correct format
- test - test images
- train - training images

### Columns
- `StudyInstanceUID` - unique ID for each image
- `ETT - Abnormal` - endotracheal tube placement abnormal
- `ETT - Borderline` - endotracheal tube placement borderline abnormal
- `ETT - Normal` - endotracheal tube placement normal
- `NGT - Abnormal` - nasogastric tube placement abnormal
- `NGT - Borderline` - nasogastric tube placement borderline abnormal
- `NGT - Incompletely Imaged` - nasogastric tube placement inconclusive due to imaging
- `NGT - Normal` - nasogastric tube placement borderline normal
- `CVC - Abnormal` - central venous catheter placement abnormal
- `CVC - Borderline` - central venous catheter placement borderline abnormal
- `CVC - Normal` - central venous catheter placement normal
- `Swan Ganz Catheter Present`
- `PatientID` - unique ID for each patient in the dataset

# 2. Python version

3.9

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (172 lines)
            sample_submission.csv (3010 lines)
            sample_submission.csv.zip (64.2 kB)
            test.zip (642.8 MB)
            train.csv (27075 lines)
            train.csv.zip (798.6 kB)
            train.zip (5.8 GB)
            train_annotations.csv (16262 lines)
            train_annotations.csv.zip (1.4 MB)
            ranzcr-clip-catheter-line-classification/
                description.md (172 lines)
                sample_submission.csv (3010 lines)
                ... and 7 other files
                ranzcr-clip-catheter-line-classification/
                test/
                    1.2.826.0.1.3680043.8.498.57477258980718966370268402373568359767.jpg (201.5 kB)
                    1.2.826.0.1.3680043.8.498.70369997506092680321332747830211112877.jpg (176.2 kB)
                    ... and 3007 other files
                    test/
                train/
                    1.2.826.0.1.3680043.8.498.22695482406757766723043400617436591679.jpg (306.8 kB)
                    1.2.826.0.1.3680043.8.498.67485046061813043086989609654834659261.jpg (315.8 kB)
                    ... and 27072 other files
                    train/
            test/
                1.2.826.0.1.3680043.8.498.57477258980718966370268402373568359767.jpg (201.5 kB)
                1.2.826.0.1.3680043.8.498.70369997506092680321332747830211112877.jpg (176.2 kB)
                ... and 3007 other files
                test/
            train/
                1.2.826.0.1.3680043.8.498.22695482406757766723043400617436591679.jpg (306.8 kB)
                1.2.826.0.1.3680043.8.498.67485046061813043086989609654834659261.jpg (315.8 kB)
                ... and 27072 other files
                train/
        input/
            description.md (172 lines)
            sample_submission.csv (3010 lines)
            sample_submission.csv.zip (64.2 kB)
            test.zip (642.8 MB)
            train.csv (27075 lines)
            train.csv.zip (798.6 kB)
            train.zip (5.8 GB)
            train_annotations.csv (16262 lines)
            train_annotations.csv.zip (1.4 MB)
            ranzcr-clip-catheter-line-classification/
                description.md (172 lines)
                sample_submission.csv (3010 lines)
                ... and 7 other files
                ranzcr-clip-catheter-line-classification/
                test/
                    1.2.826.0.1.3680043.8.498.57477258980718966370268402373568359767.jpg (201.5 kB)
                    1.2.826.0.1.3680043.8.498.70369997506092680321332747830211112877.jpg (176.2 kB)
                    ... and 3007 other files
                    test/
                train/
                    1.2.826.0.1.3680043.8.498.22695482406757766723043400617436591679.jpg (306.8 kB)
                    1.2.826.0.1.3680043.8.498.67485046061813043086989609654834659261.jpg (315.8 kB)
                    ... and 27072 other files
                    train/
            test/
                1.2.826.0.1.3680043.8.498.57477258980718966370268402373568359767.jpg (201.5 kB)
                1.2.826.0.1.3680043.8.498.70369997506092680321332747830211112877.jpg (176.2 kB)
                ... and 3007 other files
                test/
                    1.2.826.0.1.3680043.8.498.57477258980718966370268402373568359767.jpg (201.5 kB)
                    1.2.826.0.1.3680043.8.498.70369997506092680321332747830211112877.jpg (176.2 kB)
                    ... and 3007 other files
                    test/
            train/
                1.2.826.0.1.3680043.8.498.22695482406757766723043400617436591679.jpg (306.8 kB)
                1.2.826.0.1.3680043.8.498.67485046061813043086989609654834659261.jpg (315.8 kB)
                ... and 27072 other files
                train/
                    1.2.826.0.1.3680043.8.498.22695482406757766723043400617436591679.jpg (306.8 kB)
                    1.2.826.0.1.3680043.8.498.67485046061813043086989609654834659261.jpg (315.8 kB)
                    ... and 27072 other files
                    train/
        working/
            ranzcr-clip-catheter-line-classification/
                description.md (172 lines)
                sample_submission.csv (3010 lines)
                ... and 7 other files
                ranzcr-clip-catheter-line-classification/
                test/
                    1.2.826.0.1.3680043.8.498.57477258980718966370268402373568359767.jpg (201.5 kB)
                    1.2.826.0.1.3680043.8.498.70369997506092680321332747830211112877.jpg (176.2 kB)
                    ... and 3007 other files
                    test/
                train/
                    1.2.826.0.1.3680043.8.498.22695482406757766723043400617436591679.jpg (306.8 kB)
                    1.2.826.0.1.3680043.8.498.67485046061813043086989609654834659261.jpg (315.8 kB)
                    ... and 27072 other files
                    train/
```

-> data/ranzcr-clip-catheter-line-classification/sample_submission.csv has 3009 rows and 10 columns.
The columns are: StudyInstanceUID, ETT - Abnormal, ETT - Borderline, ETT - Normal, NGT - Abnormal, NGT - Borderline, NGT - Incompletely Imaged, NGT - Normal, CVC - Abnormal, CVC - Borderline

-> data/ranzcr-clip-catheter-line-classification/train.csv has 27074 rows and 13 columns.
The columns are: StudyInstanceUID, ETT - Abnormal, ETT - Borderline, ETT - Normal, NGT - Abnormal, NGT - Borderline, NGT - Incompletely Imaged, NGT - Normal, CVC - Abnormal, CVC - Borderline, CVC - Normal, Swan Ganz Catheter Present, PatientID

-> data/ranzcr-clip-catheter-line-classification/train_annotations.csv has 16261 rows and 3 columns.
The columns are: StudyInstanceUID, label, data

-> data/sample_submission.csv has 3009 rows and 10 columns.
The columns are: StudyInstanceUID, ETT - Abnormal, ETT - Borderline, ETT - Normal, NGT - Abnormal, NGT - Borderline, NGT - Incompletely Imaged, NGT - Normal, CVC - Abnormal, CVC - Borderline

-> data/train.csv has 27074 rows and 13 columns.
The columns are: StudyInstanceUID, ETT - Abnormal, ETT - Borderline, ETT - Normal, NGT - Abnormal, NGT - Borderline, NGT - Incompletely Imaged, NGT - Normal, CVC - Abnormal, CVC - Borderline, CVC - Normal, Swan Ganz Catheter Present, PatientID

-> data/train_annotations.csv has 16261 rows and 3 columns.
The columns are: StudyInstanceUID, label, data

-> (stopped after 10 files for performance)

# 5. Target score

0.949966397922312

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.52253) has done: 'I fix the import errors, add missing pandas and Keras imports, remove the problematic mixed‑precision setup, make the data‑folder path detection robust, and ensure all variables (train, test paths, labels, model, etc.) are defined before they are used. These changes remove the runtime crashes while keeping the original modeling approach unchanged, allowing the script to run end‑to‑end and generate a proper `submission.csv` file.'
- What this solution (achieved 0.52253) has done: 'The changes enable in‑memory caching of the training and validation pipelines (removing the explicit `cache=False` arguments) so images are decoded only once and reused across epochs, which dramatically cuts I/O overhead while keeping the exact model, dataset splits, augmentations, and training logic unchanged.'

# 9. Code solution

## === cell 0
import os, random, warnings

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "cpp"

import numpy as np
import pandas as pd
import tensorflow as tf

random.seed(0)
np.random.seed(0)
tf.random.set_seed(0)

warnings.simplefilter("ignore")

cand_dirs = [
    "/kaggle/input/ranzcr-clip-catheter-line-classification",
    "../input/ranzcr-clip-catheter-line-classification",
    "data/ranzcr-clip-catheter-line-classification",
]
WORK_DIR = next((d for d in cand_dirs if os.path.isdir(d)), None)
if WORK_DIR is None:
    raise FileNotFoundError("Could not locate the dataset directory.")
print("Using WORK_DIR:", WORK_DIR)
print("Directory contents sample:", os.listdir(WORK_DIR)[:5])



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_11/848455222.py in <cell line: 0>()
      6 import numpy as np
      7 import pandas as pd
----> 8 import tensorflow as tf
      9 
     10 random.seed(0)

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
print("Train images count:", len(os.listdir(os.path.join(WORK_DIR, "train"))))



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4050538634.py in <cell line: 0>()
----> 1 print("Train images count:", len(os.listdir(os.path.join(WORK_DIR, "train"))))
      2 

NameError: name 'WORK_DIR' is not defined

## === cell 2
train = pd.read_csv(os.path.join(WORK_DIR, "train.csv"))
ss = pd.read_csv(os.path.join(WORK_DIR, "sample_submission.csv"))

train_images = (
    os.path.join(WORK_DIR, "train") + "/" + train["StudyInstanceUID"] + ".jpg"
)
test_images = os.path.join(WORK_DIR, "test") + "/" + ss["StudyInstanceUID"] + ".jpg"

label_cols = ss.columns[1:]
labels = train[label_cols].values

print("Label columns:", label_cols.tolist())
print("Training data preview:")
print(train.head())



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2480626682.py in <cell line: 0>()
----> 1 train = pd.read_csv(os.path.join(WORK_DIR, "train.csv"))
      2 ss = pd.read_csv(os.path.join(WORK_DIR, "sample_submission.csv"))
      3 
      4 train_images = (
      5     os.path.join(WORK_DIR, "train") + "/" + train["StudyInstanceUID"] + ".jpg"

NameError: name 'WORK_DIR' is not defined

## === cell 3
from sklearn.model_selection import train_test_split  # added import

BATCH_SIZE = 64
EPOCHS = 30
TARGET_SIZE = 224




## === cell 4
def build_decoder(with_labels=True, target_size=(TARGET_SIZE, TARGET_SIZE), ext="jpg"):
    def decode(path):
        file_bytes = tf.io.read_file(path)
        if ext == "png":
            img = tf.image.decode_png(file_bytes, channels=3)
        elif ext in ("jpg", "jpeg"):
            img = tf.image.decode_jpeg(file_bytes, channels=3)
        else:
            raise ValueError("Image extension not supported")
        img = tf.cast(img, tf.float32) / 255.0
        img = tf.image.resize(img, target_size)
        return img

    def decode_with_labels(path, label):
        return decode(path), label

    return decode_with_labels if with_labels else decode


def build_augmenter(with_labels=True):
    def augment(img):
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        img = tf.image.adjust_brightness(img, 0.1)
        return img

    def augment_with_labels(img, label):
        return augment(img), label

    return augment_with_labels if with_labels else augment


def build_dataset(
    paths,
    labels=None,
    bsize=32,
    cache=True,
    decode_fn=None,
    augment_fn=None,
    augment=True,
    repeat=True,
    shuffle=1024,
    cache_dir=None,
):
    if cache_dir is not None and cache:
        os.makedirs(cache_dir, exist_ok=True)

    if decode_fn is None:
        decode_fn = build_decoder(labels is not None)

    if augment_fn is None:
        augment_fn = build_augmenter(labels is not None)

    AUTO = tf.data.experimental.AUTOTUNE
    slices = (paths, labels) if labels is not None else paths

    dset = tf.data.Dataset.from_tensor_slices(slices)
    dset = dset.map(decode_fn, num_parallel_calls=AUTO)
    if cache:
        if cache_dir is not None:
            dset = dset.cache(cache_dir)
        else:
            dset = dset.cache()
    dset = dset.map(augment_fn, num_parallel_calls=AUTO) if augment else dset
    dset = dset.repeat() if repeat else dset
    dset = dset.shuffle(shuffle) if shuffle else dset
    dset = dset.batch(bsize).prefetch(AUTO)

    return dset




## === cell 5
test_df = build_dataset(
    test_images,
    bsize=BATCH_SIZE,
    repeat=False,
    shuffle=False,
    augment=False,
    cache=False,
)



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2092789202.py in <cell line: 0>()
      1 test_df = build_dataset(
----> 2     test_images,
      3     bsize=BATCH_SIZE,
      4     repeat=False,
      5     shuffle=False,

NameError: name 'test_images' is not defined

## === cell 6
from tensorflow.keras.applications import Xception
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

pretrained_path = os.path.join(WORK_DIR, "Xception_750_TPU.h5")
if os.path.exists(pretrained_path):
    model = tf.keras.models.load_model(pretrained_path)
    print("Loaded pretrained model from:", pretrained_path)
else:
    print("Pretrained model not found – building a fresh Xception model.")
    base = Xception(
        weights="imagenet", include_top=False, input_shape=(TARGET_SIZE, TARGET_SIZE, 3)
    )
    base.trainable = True
    x = layers.GlobalAveragePooling2D(dtype="float32")(base.output)
    output = layers.Dense(len(label_cols), activation="sigmoid", dtype="float32")(x)
    model = tf.keras.models.Model(inputs=base.input, outputs=output)
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(name="auc")],
    )
    model.summary()



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_11/1791589548.py in <cell line: 0>()
----> 1 from tensorflow.keras.applications import Xception
      2 from tensorflow.keras import layers
      3 from tensorflow.keras.optimizers import Adam
      4 from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
      5 

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

## === cell 7
print(f"Model has {len(model.layers)} layers.")



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3285467344.py in <cell line: 0>()
----> 1 print(f"Model has {len(model.layers)} layers.")
      2 

NameError: name 'model' is not defined

## === cell 8
if not os.path.exists(pretrained_path):
    train_paths, val_paths, train_lbls, val_lbls = train_test_split(
        train_images.values, labels, test_size=0.2, random_state=42
    )
    import math

    train_steps = math.ceil(len(train_paths) / BATCH_SIZE)
    val_steps = math.ceil(len(val_paths) / BATCH_SIZE)

    train_ds = build_dataset(
        train_paths,
        train_lbls,
        bsize=BATCH_SIZE,
        augment=True,
        repeat=True,
        cache=True,
        cache_dir=None,
    )
    val_ds = build_dataset(
        val_paths,
        val_lbls,
        bsize=BATCH_SIZE,
        augment=False,
        repeat=False,
        shuffle=False,
        cache=True,
        cache_dir=None,
    )

    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True, monitor="val_auc"),
        ReduceLROnPlateau(patience=3, factor=0.5, monitor="val_auc"),
    ]

    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        steps_per_epoch=train_steps,
        validation_data=val_ds,
        validation_steps=val_steps,
        callbacks=callbacks,
        verbose=1,
    )
    model.save(pretrained_path)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1120627864.py in <cell line: 0>()
----> 1 if not os.path.exists(pretrained_path):
      2     train_paths, val_paths, train_lbls, val_lbls = train_test_split(
      3         train_images.values, labels, test_size=0.2, random_state=42
      4     )
      5     import math

NameError: name 'pretrained_path' is not defined

## === cell 9
preds = model.predict(test_df, verbose=1)
ss.loc[:, label_cols] = preds
submission_path = "submission.csv"
ss.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/523133000.py in <cell line: 0>()
----> 1 preds = model.predict(test_df, verbose=1)
      2 ss.loc[:, label_cols] = preds
      3 submission_path = "submission.csv"
      4 ss.to_csv(submission_path, index=False)
      5 print(f"Submission saved to {submission_path}")

NameError: name 'model' is not defined

## === cell 10
print(ss.head())

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3037783917.py in <cell line: 0>()
----> 1 print(ss.head())

NameError: name 'ss' is not defined
