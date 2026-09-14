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
Predict the genetic subtype of glioblastoma using MRI (magnetic resonance imaging) scans to detect for the presence of MGMT promoter methylation.

## Metric
Area under the ROC curve between the predicted probability and the observed target.

## Submission Format
For each `BraTS21ID` in the test set, you must predict a probability for the target `MGMT_value`. The file should contain a header and have the following format:

```
BraTS21ID,MGMT_value
00001,0.5
00013,0.5
00015,0.5
etc.
```

## Dataset
- **train/** - folder containing the training files, with each top-level folder representing a subject. **NOTE:** There are some unexpected issues with the following three cases in the training dataset, participants can exclude the cases during training: `[00109, 00123, 00709]`. We have checked and confirmed that the testing dataset is free from such issues.
- **train_labels.csv** - file containing the target `MGMT_value` for each subject in the training data (e.g. the presence of MGMT promoter methylation)
- **test/** - the test files, which use the same structure as `train/`; your task is to predict the `MGMT_value` for each subject in the test data. **NOTE**: the total size of the rerun test set (Public and Private) is ~5x the size of the Public test set
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.10

# 3. Installed packages

geopandas==0.14.4
keras-tuner==1.4.7
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
pydicom==3.0.1
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
tqdm==4.67.1

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (202 lines)
            sample_submission.csv (60 lines)
            sample_submission.csv.zip (382 Bytes)
            test.zip (1.3 GB)
            train.zip (10.2 GB)
            train_labels.csv (527 lines)
            train_labels.csv.zip (1.4 kB)
            rsna-miccai-brain-tumor-radiogenomic-classification/
                description.md (202 lines)
                sample_submission.csv (60 lines)
                ... and 5 other files
                rsna-miccai-brain-tumor-radiogenomic-classification/
                test/
                    00002/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00019/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 58 other folders
                train/
                    00000/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00003/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 525 other folders
            test/
                00002/
                    FLAIR/
                        Image-387.dcm (525.4 kB)
                        Image-388.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 29 other files
                    T1wCE/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 382 other files
                00019/
                    FLAIR/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 30 other files
                    T1wCE/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T2w/
                        Image-258.dcm (525.4 kB)
                        Image-259.dcm (525.4 kB)
                        ... and 127 other files
                ... and 58 other folders
            train/
                00000/
                    FLAIR/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 398 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 31 other files
                    T1wCE/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 406 other files
                00003/
                    FLAIR/
                        Image-387.dcm (525.4 kB)
                        Image-388.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 31 other files
                    T1wCE/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 406 other files
                ... and 525 other folders
        input/
            description.md (202 lines)
            sample_submission.csv (60 lines)
            sample_submission.csv.zip (382 Bytes)
            test.zip (1.3 GB)
            train.zip (10.2 GB)
            train_labels.csv (527 lines)
            train_labels.csv.zip (1.4 kB)
            rsna-miccai-brain-tumor-radiogenomic-classification/
                description.md (202 lines)
                sample_submission.csv (60 lines)
                ... and 5 other files
                rsna-miccai-brain-tumor-radiogenomic-classification/
                test/
                    00002/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00019/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 58 other folders
                train/
                    00000/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00003/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 525 other folders
            test/
                00002/
                    FLAIR/
                        Image-387.dcm (525.4 kB)
                        Image-388.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 29 other files
                    T1wCE/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 382 other files
                00019/
                    FLAIR/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 30 other files
                    T1wCE/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T2w/
                        Image-258.dcm (525.4 kB)
                        Image-259.dcm (525.4 kB)
                        ... and 127 other files
                ... and 58 other folders
            train/
                00000/
                    FLAIR/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 398 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 31 other files
                    T1wCE/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 406 other files
                00003/
                    FLAIR/
                        Image-387.dcm (525.4 kB)
                        Image-388.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 31 other files
                    T1wCE/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 406 other files
                ... and 525 other folders
        working/
            rsna-miccai-brain-tumor-radiogenomic-classification/
                description.md (202 lines)
                sample_submission.csv (60 lines)
                ... and 5 other files
                rsna-miccai-brain-tumor-radiogenomic-classification/
                test/
                    00002/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00019/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 58 other folders
                train/
                    00000/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00003/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 525 other folders
```

-> data/rsna-miccai-brain-tumor-radiogenomic-classification/sample_submission.csv has 59 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> data/rsna-miccai-brain-tumor-radiogenomic-classification/train_labels.csv has 526 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> data/sample_submission.csv has 59 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> data/train_labels.csv has 526 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> input/rsna-miccai-brain-tumor-radiogenomic-classification/sample_submission.csv has 59 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> input/rsna-miccai-brain-tumor-radiogenomic-classification/train_labels.csv has 526 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> (stopped after 10 files for performance)

# 5. Target score

-1.0

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.51294) has done: 'I fix the environment-breaking import error by forcing TensorFlow to use the pure-Python protobuf implementation (this resolves the `MessageFactory.GetPrototype` crash) before importing TensorFlow. I fix the DICOM reader call for pydicom v3 by replacing the removed `read_file` with `dcmread`, and update the Keras `Rescaling` layer path so it works in TF/Keras 2.18. Finally, I fix metric/early-stopping naming issues so tuning/training runs, and generate a valid submission by using the positive-class probability (not argmax labels) and keeping IDs aligned with `sample_submission.csv`.'
- What this solution (achieved 0.41529) has done: 'I fix the TensorFlow/protobuf crash by setting the required environment variables before any TensorFlow-related imports and by using a safe fallback that forces the pure-Python protobuf runtime. I also make the import order deterministic and add a small compatibility tweak for progress bars (`tqdm.auto`) so it runs reliably in Kaggle notebooks/scripts. These changes are score-neutral and only unblock execution so your existing modeling/training/prediction logic can run end-to-end. The script still train the same model and write a valid `submission.csv` with the required columns and alignment to `sample_submission.csv`.'
- What this solution (achieved 0.48118) has done: 'I fix the TensorFlow/protobuf crash by forcing the pure-Python protobuf runtime *before any protobuf-related imports* and by importing `google.protobuf` early so the environment variable takes effect reliably in Kaggle. I keep your data loading, preprocessing, model, tuning, and training logic the same, only adding a small safety fallback around the TensorFlow import so the notebook always runs end-to-end. I also ensure the submission is written as `submission.csv` with the exact required columns and aligned to `sample_submission.csv` (your current code already does this, so it stays unchanged). These changes are intended to be score-neutral; they unblock execution so you can get a valid submission and keep the existing performance trajectory toward your target.'
- What this solution (achieved 0.55176) has done: 'The crash happens before training: TensorFlow 2.18 imports a newer protobuf API where `MessageFactory.GetPrototype` was removed, so forcing the pure-Python protobuf runtime alone isn’t sufficient in this environment. I fix this by using a protobuf/TensorFlow-safe setup: stop forcing the Python protobuf backend and instead force the C++ backend, then import TensorFlow; this is execution-only and score-neutral. I also keep everything else (data loading, model, tuner, training, prediction, and submission formatting) unchanged so behavior stays consistent and still writes a valid `submission.csv`. This should run end-to-end and keep your score trajectory stable toward the target band.'
- What this solution (achieved 0.46588) has done: 'I fix the TensorFlow/protobuf import crash by forcing the pure-Python protobuf runtime *before* importing anything that could load protobuf/TensorFlow, and I keep the rest of your pipeline (data loading, model, tuner, training, inference, and submission formatting) unchanged. This addresses the `MessageFactory.GetPrototype` failure so training can start reliably in this Kaggle environment. I also add a small, safe fallback so if the first TensorFlow import attempt fails, it retries with the alternate protobuf backend, which is execution-stability only and score-neutral. The script still write a valid `submission.csv` with `BraTS21ID,MGMT_value` aligned to `sample_submission.csv`.'
- What this solution (achieved 0.47412) has done: 'I fix the TensorFlow import crash by removing the unsupported protobuf “cpp” backend setting (it causes `google.protobuf.pyext._message` import failure here) and forcing the safe pure-Python protobuf backend *before* TensorFlow/Keras is imported. This also restore the missing globals (`SEED`, `to_categorical`, etc.) by ensuring cell 1 executes successfully and keeps import order deterministic. I fix the `keras_tuner` import crash by forcing KerasTuner to use `tf.keras` (setting `KERAS_BACKEND=tensorflow` and importing `tensorflow.keras`), which is compatible with TF 2.18 in Kaggle. These changes are execution/stability focused and keep your model/training logic the same, so they should produce a valid `submission.csv` end-to-end and move from “Not yielded” to a valid scored submission.'
- What this solution (achieved 0.45647) has done: 'I fix the TensorFlow/protobuf import crash by setting the protobuf implementation environment variables before *any* protobuf/TensorFlow-related imports and adding a safe fallback that retries TensorFlow import with the alternate protobuf backend if needed. I also ensure KerasTuner uses `tf.keras` (compatible with TF 2.18 here) without changing your model architecture, training loop, or prediction logic. These changes are execution/stability focused and should be score-neutral (or within negligible variance), while guaranteeing the pipeline runs end-to-end and writes a valid `submission.csv` with the required columns and alignment. No feature extraction, model layers, loss, or inference aggregation logic is changed.'
- What this solution (achieved 0.41882) has done: 'Your run is blocked by a TensorFlow/protobuf incompatibility: importing TensorFlow triggers `MessageFactory.GetPrototype` errors with the currently forced pure-Python protobuf runtime. I make the protobuf/TensorFlow import robust by (1) not forcing the Python protobuf backend up-front and (2) adding a controlled two-step fallback that retries TensorFlow import with the opposite protobuf backend, then verifies `tf` is usable. This is execution/stability focused and keeps your data pipeline, model, tuning, training, inference, and submission formatting unchanged, so score behavior should remain essentially the same while producing a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import glob
import random
from pathlib import Path
import importlib

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import pydicom
import cv2

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


def _import_tensorflow_with_protobuf_backend(backend: str):
    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = backend
    if backend == "python":
        os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "3"
    else:
        os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

    importlib.invalidate_caches()

    import tensorflow as tf  # noqa: F401

    _ = tf.constant(1)
    return tf


try:
    tf = _import_tensorflow_with_protobuf_backend("cpp")
except Exception:
    tf = _import_tensorflow_with_protobuf_backend("python")

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_11/793496611.py in <cell line: 0>()
     45 try:
---> 46     tf = _import_tensorflow_with_protobuf_backend("cpp")
     47 except Exception:

/tmp/ipykernel_11/793496611.py in _import_tensorflow_with_protobuf_backend(backend)
     35 
---> 36     import tensorflow as tf  # noqa: F401
     37 

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
/tmp/ipykernel_11/793496611.py in <cell line: 0>()
     46     tf = _import_tensorflow_with_protobuf_backend("cpp")
     47 except Exception:
---> 48     tf = _import_tensorflow_with_protobuf_backend("python")
     49 
     50 from tensorflow import keras

/tmp/ipykernel_11/793496611.py in _import_tensorflow_with_protobuf_backend(backend)
     34     importlib.invalidate_caches()
     35 
---> 36     import tensorflow as tf  # noqa: F401
     37 
     38     # Touch a trivial op to ensure TF is usable

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
data_dir = Path("../input/rsna-miccai-brain-tumor-radiogenomic-classification/")

mri_types = ["FLAIR", "T1w", "T2w", "T1wCE"]
excluded_images = [109, 123, 709]  # Bad images per competition note

train_df = pd.read_csv(data_dir / "train_labels.csv")
test_df = pd.read_csv(data_dir / "sample_submission.csv")
sample_submission = pd.read_csv(data_dir / "sample_submission.csv")

train_df["BraTS21ID"] = train_df["BraTS21ID"].astype(int)
test_df["BraTS21ID"] = test_df["BraTS21ID"].astype(int)
sample_submission["BraTS21ID"] = sample_submission["BraTS21ID"].astype(int)

train_df = train_df[~train_df.BraTS21ID.isin(excluded_images)].reset_index(drop=True)

train_df.head(), test_df.head()




## === cell 2
def load_dicom(path, size=224):
    """
    Reads a DICOM image, standardizes pixel values to [0,1], rescales to [0,255], and resizes.
    pydicom v3 removed read_file; use dcmread.
    """
    dicom = pydicom.dcmread(path)
    data = dicom.pixel_array.astype(np.float32)

    mx = np.max(data)
    if mx > 0:
        data = data / mx
    data = (data * 255.0).astype(np.uint8)

    return cv2.resize(data, (size, size), interpolation=cv2.INTER_AREA)


def get_all_image_paths(brats21id, image_type, folder="train"):
    """Returns paths for a slice window (25%-75%) sampled every `interval` images."""
    assert image_type in mri_types

    patient_path = os.path.join(
        "../input/rsna-miccai-brain-tumor-radiogenomic-classification/%s/" % folder,
        str(brats21id).zfill(5),
    )

    paths = sorted(
        glob.glob(os.path.join(patient_path, image_type, "*")),
        key=lambda x: int(os.path.splitext(x)[0].split("-")[-1]),
    )

    num_images = len(paths)
    if num_images == 0:
        return np.array([], dtype=object)

    start = int(num_images * 0.25)
    end = int(num_images * 0.75)

    interval = 3
    if num_images < 10:
        interval = 1

    return np.array(paths[start:end:interval])


def get_all_images(brats21id, image_type, folder="train", size=32):
    paths = get_all_image_paths(brats21id, image_type, folder)
    return [load_dicom(path, size) for path in paths]


def get_all_data_for_train(image_type, image_size=32):
    X, y, train_ids = [], [], []

    for i in tqdm(train_df.index):
        row = train_df.loc[i]
        brats_id = int(row["BraTS21ID"])
        images = get_all_images(brats_id, image_type, folder="train", size=image_size)
        label = int(row["MGMT_value"])

        if len(images) == 0:
            continue

        X += images
        y += [label] * len(images)
        train_ids += [brats_id] * len(images)

    return np.array(X), np.array(y), np.array(train_ids)


def get_all_data_for_test(image_type, image_size=32):
    X, test_ids = [], []

    for i in tqdm(test_df.index):
        row = test_df.loc[i]
        brats_id = int(row["BraTS21ID"])
        images = get_all_images(brats_id, image_type, folder="test", size=image_size)

        if len(images) == 0:
            continue

        X += images
        test_ids += [brats_id] * len(images)

    return np.array(X), np.array(test_ids)




## === cell 3
X, y, trainidt = get_all_data_for_train("T1wCE", image_size=32)
X_test, testidt = get_all_data_for_test("T1wCE", image_size=32)

X.shape, y.shape, trainidt.shape, X_test.shape, testidt.shape



## === cell 4
X_train, X_valid, y_train, y_valid, trainidt_train, trainidt_valid = train_test_split(
    X, y, trainidt, test_size=0.2, random_state=SEED, stratify=y
)

X_train = tf.expand_dims(X_train, axis=-1)
X_valid = tf.expand_dims(X_valid, axis=-1)
X_test = tf.expand_dims(X_test, axis=-1)

X_train.shape, X_valid.shape, X_test.shape



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2856654212.py in <cell line: 0>()
      1 X_train, X_valid, y_train, y_valid, trainidt_train, trainidt_valid = train_test_split(
----> 2     X, y, trainidt, test_size=0.2, random_state=SEED, stratify=y
      3 )
      4 
      5 X_train = tf.expand_dims(X_train, axis=-1)

NameError: name 'SEED' is not defined

## === cell 5
y_train = to_categorical(y_train, num_classes=2)
y_valid = to_categorical(y_valid, num_classes=2)

y_train.shape, y_valid.shape



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1686704136.py in <cell line: 0>()
----> 1 y_train = to_categorical(y_train, num_classes=2)
      2 y_valid = to_categorical(y_valid, num_classes=2)
      3 
      4 y_train.shape, y_valid.shape
      5 

NameError: name 'to_categorical' is not defined

## === cell 6
import keras_tuner as kt


def make_model(hp):
    inputs = keras.Input(shape=X_train.shape[1:])

    x = layers.Rescaling(1.0 / 255.0)(inputs)

    x = layers.Conv2D(64, kernel_size=(4, 4), activation="relu", name="Conv_1")(x)
    x = layers.MaxPool2D(pool_size=(2, 2))(x)

    x = layers.Conv2D(32, kernel_size=(2, 2), activation="relu", name="Conv_2")(x)
    x = layers.MaxPool2D(pool_size=(1, 1))(x)

    x = layers.Dropout(hp.Float("dense_dropout", min_value=0.0, max_value=0.7))(x)
    x = layers.Flatten()(x)

    x = layers.Dense(32, activation="relu")(x)
    outputs = layers.Dense(2, activation="softmax")(x)

    model = keras.Model(inputs, outputs)

    roc_auc = tf.keras.metrics.AUC(name="roc_auc", curve="ROC")
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=[roc_auc])
    return model




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_11/333106160.py in <cell line: 0>()
----> 1 import keras_tuner as kt
      2 
      3 
      4 def make_model(hp):
      5     inputs = keras.Input(shape=X_train.shape[1:])

/usr/local/lib/python3.11/dist-packages/keras_tuner/__init__.py in <module>
      6 
      7 
----> 8 from keras_tuner import applications
      9 from keras_tuner import engine
     10 from keras_tuner import errors

/usr/local/lib/python3.11/dist-packages/keras_tuner/applications/__init__.py in <module>
      6 
      7 
----> 8 from keras_tuner.src.applications.augment import HyperImageAugment
      9 from keras_tuner.src.applications.efficientnet import HyperEfficientNet
     10 from keras_tuner.src.applications.resnet import HyperResNet

/usr/local/lib/python3.11/dist-packages/keras_tuner/src/__init__.py in <module>
     14 
     15 
---> 16 from keras_tuner.src import applications
     17 from keras_tuner.src import oracles
     18 from keras_tuner.src import tuners

/usr/local/lib/python3.11/dist-packages/keras_tuner/src/applications/__init__.py in <module>
     14 
     15 
---> 16 from keras_tuner.src.applications.augment import HyperImageAugment
     17 from keras_tuner.src.applications.efficientnet import HyperEfficientNet
     18 from keras_tuner.src.applications.resnet import HyperResNet

/usr/local/lib/python3.11/dist-packages/keras_tuner/src/applications/augment.py in <module>
     15 
     16 from keras_tuner.src.api_export import keras_tuner_export
---> 17 from keras_tuner.src.backend import keras
     18 from keras_tuner.src.backend import ops
     19 from keras_tuner.src.backend import random

/usr/local/lib/python3.11/dist-packages/keras_tuner/src/backend/__init__.py in <module>
     24 """
     25 
---> 26 from keras_tuner.src.backend import config
     27 from keras_tuner.src.backend import io
     28 from keras_tuner.src.backend import keras

/usr/local/lib/python3.11/dist-packages/keras_tuner/src/backend/config.py in <module>
     13 # limitations under the License.
     14 
---> 15 import keras
     16 
     17 

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

## === cell 7
tuner = kt.tuners.BayesianOptimization(
    make_model, objective="val_loss", max_trials=5, overwrite=True, seed=SEED
)

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_roc_auc", mode="max", patience=3, restore_best_weights=True
    )
]

tuner.search(
    X_train, y_train, validation_split=0.2, callbacks=callbacks, verbose=1, epochs=20
)



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3618434137.py in <cell line: 0>()
----> 1 tuner = kt.tuners.BayesianOptimization(
      2     make_model, objective="val_loss", max_trials=5, overwrite=True, seed=SEED
      3 )
      4 
      5 callbacks = [

NameError: name 'kt' is not defined

## === cell 8
best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
model = make_model(best_hp)

history = model.fit(X_train, y_train, validation_split=0.2, epochs=50, verbose=1)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2474235039.py in <cell line: 0>()
----> 1 best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
      2 model = make_model(best_hp)
      3 
      4 history = model.fit(X_train, y_train, validation_split=0.2, epochs=50, verbose=1)
      5 

NameError: name 'tuner' is not defined

## === cell 9
y_pred_valid = model.predict(X_valid, verbose=0)[:, 1]

result = pd.DataFrame(
    {"BraTS21ID": trainidt_valid.astype(int), "MGMT_value": y_pred_valid.astype(float)}
)
result2 = result.groupby("BraTS21ID", as_index=False).mean()

result2 = result2.merge(
    train_df, on="BraTS21ID", how="inner", suffixes=("_pred", "_true")
)
auc = roc_auc_score(result2["MGMT_value_true"], result2["MGMT_value_pred"])
print(f"Validation AUC={auc:.5f}")



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1815458939.py in <cell line: 0>()
----> 1 y_pred_valid = model.predict(X_valid, verbose=0)[:, 1]
      2 
      3 result = pd.DataFrame(
      4     {"BraTS21ID": trainidt_valid.astype(int), "MGMT_value": y_pred_valid.astype(float)}
      5 )

NameError: name 'model' is not defined

## === cell 10
y_pred_test = model.predict(X_test, verbose=0)[:, 1]

test_slice_df = pd.DataFrame(
    {"BraTS21ID": testidt.astype(int), "MGMT_value": y_pred_test.astype(float)}
)
test_patient_df = test_slice_df.groupby("BraTS21ID", as_index=False).mean()

sub = sample_submission[["BraTS21ID"]].merge(
    test_patient_df, on="BraTS21ID", how="left"
)
sub["MGMT_value"] = sub["MGMT_value"].fillna(0.5).clip(0.0, 1.0)

sub.to_csv("submission.csv", index=False)
sub.head(), sub.shape

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3113505528.py in <cell line: 0>()
----> 1 y_pred_test = model.predict(X_test, verbose=0)[:, 1]
      2 
      3 test_slice_df = pd.DataFrame(
      4     {"BraTS21ID": testidt.astype(int), "MGMT_value": y_pred_test.astype(float)}
      5 )

NameError: name 'model' is not defined
