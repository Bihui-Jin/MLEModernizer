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

0.9380388595462292

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.50383) has done: 'I fix the environment crash caused by an incompatible protobuf import by removing the protobuf version introspection that triggers the `MessageFactory.GetPrototype` error. Then I fix inference by removing the redundant `.numpy()` call after `predict_on_batch`, since it already returns a NumPy array in TF/Keras, which currently stops submission generation. Finally, I make the model input shape consistent (H, W, 3) to avoid subtle shape mismatches, while keeping the exact architecture and weights behavior unchanged, and ensure the submission has all 11 required columns even if the provided sample file is missing some. These changes are execution/stability fixes and should allow an end-to-end run that writes a valid `submission.csv`.'
- What this solution (achieved 0.52349) has done: 'I remove the protobuf introspection that’s triggering the `MessageFactory.GetPrototype` crash so the notebook runs in TF 2.18/protobuf 6.x. Then I fix the incorrect dataset root path so the code actually finds the competition files in this environment (`/kaggle/data/...` instead of `../input/...`), which is currently causing the model to fall back to ImageNet weights and tank the score. Finally, I correct the sample-submission path used for UID ordering to ensure predictions align 1:1 with the expected test IDs and that the written `submission.csv` has all 11 required target columns.'
- What this solution (achieved 0.48602) has done: 'I remove the protobuf introspection that’s crashing TensorFlow under protobuf 6.x so the notebook runs, and I make the TFRecords discovery robust by falling back to `/kaggle/input/...` when `/kaggle/data/...` doesn’t contain the expected `test_tfrecords` folder. Then I ensure we always use the competition’s `sample_submission.csv` from the dataset root for correct UID ordering/alignment (instead of a potentially truncated/altered one), and I keep the model/weights logic unchanged so score improves back toward the target by actually using the intended pretrained weights + correct test set ordering. Finally, I guarantee the submission contains all 11 required target columns even if the sample submission file is missing some columns in this environment.'
- What this solution (achieved 0.49782) has done: 'I remove the remaining protobuf-related crash by ensuring we never touch TensorFlow/protobuf message-factory introspection at import time (this is what triggers the `MessageFactory.GetPrototype` AttributeError under protobuf 6.x). Then I fix a major score-killer: your sample_submission in this environment is missing target columns (it has only 10), so your merge currently drops/reorders targets incorrectly; I instead build the submission strictly from the official `train.csv` target columns (11) and align to the official `sample_submission.csv` UID order. Finally, I keep the exact model/weights logic unchanged, but add a safe fallback to image-folder inference when TFRecords are absent, guaranteeing an end-to-end run that writes a valid `submission.csv` with all 11 required columns.'
- What this solution (achieved 0.56517) has done: 'I fix the protobuf/TensorFlow crash that’s currently happening at import time by forcing the pure-Python protobuf implementation before importing TensorFlow (this is the smallest reliable workaround for protobuf>=6.x `MessageFactory.GetPrototype` issues). Then I fix a likely major score/validity issue by ensuring the submission always contains all 11 required target columns: this environment’s `sample_submission.csv` is missing some columns, so we must build the final submission using the full target list from `train.csv` and add any missing columns to the sample UID frame before merging. Finally, I keep your model/inference logic unchanged, but make the UID decoding robust and preserve sample ordering exactly so predictions align 1:1 with the expected test IDs.'
- What this solution (achieved 0.45051) has done: 'I fix the TensorFlow import crash caused by the protobuf 6.x incompatibility by forcing the C++ protobuf implementation (and unsetting the pure-Python override) before importing TensorFlow, which is the most reliable workaround in this environment. Then I correct the submission-column logic: your `sample_submission.csv` here is missing some required targets, so we always construct the submission columns from `train.csv` (all 11 targets) and reindex the sample to those columns without dropping anything. Finally, I keep your model/inference pipeline unchanged but make the UID alignment deterministic by reindexing predictions to the sample UID order before writing `submission.csv`.'
- What this solution (achieved 0.49421) has done: 'You’re currently crashing at TensorFlow import due to the protobuf 6.x `MessageFactory.GetPrototype` incompatibility; I apply the known Kaggle-safe workaround by forcing the pure-Python protobuf implementation *before* importing TensorFlow. Then I keep your model/data pipeline intact but fix a validity issue: the provided sample_submission in this environment is missing required target columns, so we always construct the submission columns from `train.csv` (all 11) and reindex the sample accordingly. Finally, I add a tiny robustness fix so TFRecord filenames are resolved correctly when `test_tfrecords` exists (some environments expose them with different extensions), without changing any modeling logic.'
- What this solution (achieved 0.44625) has done: 'You’re failing immediately because TensorFlow can’t import with protobuf set to the C++ implementation in this environment; switching to the pure-Python protobuf implementation before importing TensorFlow resolves the `_message` ImportError. Once TF imports, the downstream `NameError: tf is not defined` cascade disappears because later cells depend on that first import. I also ensure we always load the correct competition `train.csv` to derive the full 11 target columns (since the provided `sample_submission.csv` here is missing columns), and we still write a valid `submission.csv` with all required columns and correct UID alignment. No model architecture, weights logic, or inference semantics are changed—this is strictly an environment/import fix plus submission column robustness.'
- What this solution (achieved 0.49095) has done: 'We need to fix the TensorFlow import crash caused by the protobuf 6.x `MessageFactory.GetPrototype` incompatibility; the most reliable minimal fix in this environment is to pin protobuf to the pure-Python implementation *and* force the C++ descriptor backend off before importing TensorFlow. Then we fix a correctness/score issue stemming from inconsistent sample submission columns in this environment by always building the submission schema from `train.csv`’s 11 targets and reindexing predictions accordingly. Finally, we keep your model/inference logic unchanged, but make dataset root selection deterministic and robust so it consistently finds the actual competition dataset files and doesn’t silently fall back to ImageNet weights or wrong UID order, which is a major score killer.'

# 9. Code solution

## === cell 0
import os, sys

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "cpp"
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_DESCRIPTORS", None)

import tensorflow as tf
import pandas as pd
import numpy as np

print("Python:", sys.version)
print("TensorFlow:", tf.__version__)
print(
    "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=",
    os.environ.get("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"),
)
print(
    "PROTOCOL_BUFFERS_PYTHON_DESCRIPTORS=",
    os.environ.get("PROTOCOL_BUFFERS_PYTHON_DESCRIPTORS"),
)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_55/3215842770.py in <cell line: 0>()
      9 os.environ.pop("PROTOCOL_BUFFERS_PYTHON_DESCRIPTORS", None)
     10 
---> 11 import tensorflow as tf
     12 import pandas as pd
     13 import numpy as np

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
W = H = 338
N_CLASSES = 11
autotune = tf.data.experimental.AUTOTUNE

features = {
    "StudyInstanceUID": tf.io.FixedLenFeature([], tf.string),
    "image": tf.io.FixedLenFeature([], tf.string),
}

BASE_CANDIDATES = [
    "/kaggle/data/ranzcr-clip-catheter-line-classification",
    "/kaggle/input/ranzcr-clip-catheter-line-classification",
    "/kaggle/data/ranzcr-clip-catheter-line-classification/ranzcr-clip-catheter-line-classification",
    "/kaggle/input/ranzcr-clip-catheter-line-classification/ranzcr-clip-catheter-line-classification",
]


def pick_base_input(candidates):
    for p in candidates:
        if os.path.isdir(p) and os.path.exists(os.path.join(p, "train.csv")):
            if os.path.isdir(os.path.join(p, "test")) or os.path.isdir(
                os.path.join(p, "test_tfrecords")
            ):
                return p
    for p in candidates:
        if os.path.isdir(p) and os.path.exists(os.path.join(p, "train.csv")):
            return p
    for p in candidates:
        if os.path.isdir(p):
            return p
    return candidates[0]


BASE_INPUT = pick_base_input(BASE_CANDIDATES)

train_csv_path = os.path.join(BASE_INPUT, "train.csv")
if not os.path.exists(train_csv_path):
    train_csv_path = "/kaggle/data/train.csv"
train_df = pd.read_csv(train_csv_path)

target_cols = [
    c for c in train_df.columns if c not in ["StudyInstanceUID", "PatientID"]
]
assert (
    len(target_cols) == N_CLASSES
), f"Expected {N_CLASSES} targets, got {len(target_cols)}: {target_cols}"

mean = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
std = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)

test_tfrecords_dir = os.path.join(BASE_INPUT, "test_tfrecords")
test_images_dir = os.path.join(BASE_INPUT, "test")

weight_dir = "/kaggle/input/cassava2020weights"
if not os.path.isdir(weight_dir):
    weight_dir = "/kaggle/data/cassava2020weights"

if os.path.isdir(test_tfrecords_dir):
    test_tfrecords = sorted(
        [
            f
            for f in os.listdir(test_tfrecords_dir)
            if f.endswith(".tfrec")
            or f.endswith(".tfrecord")
            or f.endswith(".tfrecords")
        ]
    )
else:
    test_tfrecords = []

print("BASE_INPUT:", BASE_INPUT)
print("train_csv_path:", train_csv_path)
print(
    "Found test tfrecords:",
    len(test_tfrecords),
    "dir exists:",
    os.path.isdir(test_tfrecords_dir),
)
print("Found test images dir exists:", os.path.isdir(test_images_dir))
print("Weights dir exists:", os.path.isdir(weight_dir))
print("Targets:", target_cols)

model_map = {
    "efficientb3": [
        tf.keras.applications.EfficientNetB3,
        os.path.join(weight_dir, "ranzcr_efficientb3.h5"),
    ],
    "efficientb5": [
        tf.keras.applications.EfficientNetB5,
        os.path.join(weight_dir, "ranzcr_efficientb5.h5"),
    ],
    "efficientb7": [
        tf.keras.applications.EfficientNetB7,
        os.path.join(
            weight_dir, "ranzcr_efficientb5.h5"
        ),  # kept as-is (original mapping)
    ],
}




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/242780798.py in <cell line: 0>()
      1 W = H = 338
      2 N_CLASSES = 11
----> 3 autotune = tf.data.experimental.AUTOTUNE
      4 
      5 features = {

NameError: name 'tf' is not defined

## === cell 2
def parse_example(sample):
    sample = tf.io.parse_single_example(sample, features)
    image = tf.image.decode_png(sample["image"], channels=3)
    image = tf.image.resize(image, (H, W))
    image_id = sample["StudyInstanceUID"]
    return image, image_id


def preprocess(images, labels):
    images = tf.cast(images, tf.float32) / 255.0
    images = (images - mean) / std
    return images, labels


def get_count(fname):
    return 1868 if "15-1881.tfrec" in fname else 1881


def get_tfrecord(indices):
    files = [os.path.join(test_tfrecords_dir, test_tfrecords[i]) for i in indices]
    sizes = [get_count(file) for file in files]
    return files, int(np.sum(sizes))


def get_model(
    base_model,
    baseline_weight=None,
    init_weight=None,
    lr=0.001,
    optimizer=tf.optimizers.Adam,
):
    base_model = base_model(
        include_top=False, input_shape=(H, W, 3), pooling="avg", weights=baseline_weight
    )
    base_out = base_model.output
    out = tf.keras.layers.Dropout(0.3)(base_out)
    out = tf.keras.layers.Dense(N_CLASSES, activation="sigmoid")(out)
    model = tf.keras.models.Model(inputs=base_model.input, outputs=out)

    model.compile(
        optimizer=optimizer(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC()],
    )

    if init_weight and os.path.exists(init_weight):
        try:
            model.load_weights(init_weight)
            print(f"Weight loaded from {init_weight}")
        except Exception as e:
            print(f"Load weight from {init_weight} failed, {e}")
    else:
        if init_weight:
            print(
                f"Weight file not found: {init_weight} (will use baseline_weight={baseline_weight})"
            )
    return model




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1669984220.py in <cell line: 0>()
     28     init_weight=None,
     29     lr=0.001,
---> 30     optimizer=tf.optimizers.Adam,
     31 ):
     32     base_model = base_model(

NameError: name 'tf' is not defined

## === cell 3
def _read_jpg(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (H, W))
    return img


def _path_to_uid(path):
    parts = tf.strings.split(path, os.sep)
    fname = parts[-1]
    uid = tf.strings.regex_replace(fname, r"\.jpg$", "")
    return uid


if len(test_tfrecords) > 0:
    files = [os.path.join(test_tfrecords_dir, c) for c in test_tfrecords]
    test_data = tf.data.TFRecordDataset(files, num_parallel_reads=autotune)
    test_data = test_data.map(parse_example, num_parallel_calls=autotune)
    source = "tfrecords"
else:
    if not os.path.isdir(test_images_dir):
        raise FileNotFoundError(
            f"Neither TFRecords found in {test_tfrecords_dir} nor test images dir found at {test_images_dir}."
        )

    sample_path = os.path.join(BASE_INPUT, "sample_submission.csv")
    if not os.path.exists(sample_path):
        sample_path = "/kaggle/data/sample_submission.csv"
    sample_sub_uids = pd.read_csv(sample_path, usecols=["StudyInstanceUID"])
    uids = sample_sub_uids["StudyInstanceUID"].astype(str).tolist()

    paths = [os.path.join(test_images_dir, f"{uid}.jpg") for uid in uids]
    missing = [p for p in paths if not os.path.exists(p)]
    if len(missing) > 0:
        print(
            f"Warning: {len(missing)} expected test images not found by UID path; falling back to directory listing."
        )
        paths = sorted(
            [
                os.path.join(test_images_dir, f)
                for f in os.listdir(test_images_dir)
                if f.lower().endswith(".jpg")
            ]
        )
        if len(paths) == 0:
            raise FileNotFoundError(f"No .jpg files found in {test_images_dir}.")

    path_ds = tf.data.Dataset.from_tensor_slices(paths)
    test_data = path_ds.map(
        lambda p: (_read_jpg(p), _path_to_uid(p)), num_parallel_calls=autotune
    )
    source = "jpg"

test_data = test_data.batch(16)
test_data = test_data.map(preprocess, num_parallel_calls=autotune)
test_data = test_data.prefetch(autotune)

print(f"Test dataset ready from {source}.")



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3270925599.py in <cell line: 0>()
     13 
     14 
---> 15 if len(test_tfrecords) > 0:
     16     files = [os.path.join(test_tfrecords_dir, c) for c in test_tfrecords]
     17     test_data = tf.data.TFRecordDataset(files, num_parallel_reads=autotune)

NameError: name 'test_tfrecords' is not defined

## === cell 4
base_mode, weight_path = model_map["efficientb5"]

baseline = "imagenet" if not (weight_path and os.path.exists(weight_path)) else None
model = get_model(base_mode, baseline_weight=baseline, init_weight=weight_path)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/881491829.py in <cell line: 0>()
----> 1 base_mode, weight_path = model_map["efficientb5"]
      2 
      3 # Keep original behavior: if weights not found, fall back to imagenet.
      4 baseline = "imagenet" if not (weight_path and os.path.exists(weight_path)) else None
      5 model = get_model(base_mode, baseline_weight=baseline, init_weight=weight_path)

NameError: name 'model_map' is not defined

## === cell 5
preds = []
image_ids = []

sample_path = os.path.join(BASE_INPUT, "sample_submission.csv")
if not os.path.exists(sample_path):
    sample_path = "/kaggle/data/sample_submission.csv"
sample_sub = pd.read_csv(sample_path)

if "StudyInstanceUID" not in sample_sub.columns:
    raise ValueError(
        f"sample_submission is missing StudyInstanceUID column: {sample_path}"
    )

sub_cols = ["StudyInstanceUID"] + target_cols
sample_sub = sample_sub.copy()
for c in target_cols:
    if c not in sample_sub.columns:
        sample_sub[c] = 0.0
sample_sub = sample_sub[sub_cols].copy()

for batch_images, batch_image_id in test_data:
    if isinstance(batch_image_id, tf.Tensor):
        batch_ids = batch_image_id.numpy()
        if getattr(batch_ids, "dtype", None) is not None and batch_ids.dtype.kind in {
            "S",
            "O",
        }:
            batch_ids = [
                x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else str(x)
                for x in batch_ids.tolist()
            ]
        else:
            batch_ids = batch_ids.astype(str).tolist()
    else:
        batch_ids = [str(x) for x in batch_image_id]

    image_ids.extend(batch_ids)
    batch_pred = model.predict_on_batch(batch_images)  # NumPy array
    preds.append(batch_pred)

preds = np.concatenate(preds, axis=0)

if len(image_ids) != preds.shape[0]:
    raise ValueError(
        f"Mismatch: got {len(image_ids)} ids but {preds.shape[0]} predictions"
    )

test_pred_df = pd.DataFrame({"StudyInstanceUID": image_ids})
for i, c in enumerate(target_cols):
    test_pred_df[c] = preds[:, i].astype(np.float32)

test_pred_df = test_pred_df.groupby("StudyInstanceUID", as_index=False)[
    target_cols
].mean()

sub_df = sample_sub[["StudyInstanceUID"]].merge(
    test_pred_df, on="StudyInstanceUID", how="left"
)
for c in target_cols:
    sub_df[c] = sub_df[c].fillna(0.0).astype(np.float32)

sub_df = sub_df[sub_cols]
sub_df.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", sub_df.shape)
print(sub_df.head())
print("Submission columns:", list(sub_df.columns))

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2438930230.py in <cell line: 0>()
      2 image_ids = []
      3 
----> 4 sample_path = os.path.join(BASE_INPUT, "sample_submission.csv")
      5 if not os.path.exists(sample_path):
      6     sample_path = "/kaggle/data/sample_submission.csv"

NameError: name 'BASE_INPUT' is not defined
