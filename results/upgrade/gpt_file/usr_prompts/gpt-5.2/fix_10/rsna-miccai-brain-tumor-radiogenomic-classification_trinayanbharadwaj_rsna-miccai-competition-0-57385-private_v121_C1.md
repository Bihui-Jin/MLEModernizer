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

3.9

# 3. Installed packages

No external packages required in the script and installed.

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

- What this solution (achieved 0.53059) has done: 'I fix the environment-breaking import error by using the stable `tensorflow.keras` import path and forcing TensorFlow to use the pure-Python protobuf implementation, which avoids the `MessageFactory.GetPrototype` crash. Then I fix the training crash by making the AUC metric compatible with a 2-class softmax + sparse labels (use `SparseCategoricalAccuracy` for training stability while keeping the same loss/outputs). Finally, I make test loading robust to missing/failed cases (skip or zero-fill instead of indexing `None`) and ensure the submission IDs are formatted exactly like `sample_submission.csv` (zero-padded strings), writing a valid `submission.csv`.'
- What this solution (achieved 0.53059) has done: 'I fix the environment-breaking protobuf crash that happens before any training by ensuring the `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION` variables are set **before** TensorFlow (and anything that pulls protobuf) is imported, and I also add a safe fallback to the default protobuf implementation if the pure-Python path still fails. I keep the model/training/prediction logic identical, only making small robustness edits (cast labels to integer for `sparse_categorical_crossentropy` and avoid accidental float labels). Finally, I keep the submission formatting/alignment exactly as `sample_submission.csv` and ensure a `submission.csv` is always written.'
- What this solution (achieved 0.53059) has done: 'I fix the TensorFlow/protobuf crash by ensuring the environment variables are set before *any* protobuf-using imports (including `pydicom`), and by adding a safe retry that restarts the Python process once with the fallback implementation if the first import still fails. This is a correctness/stability fix and should be score-neutral (it does not change the model, data, or training semantics). I also keep all paths and the existing model/training/inference logic identical, only adjusting the import order and making the TensorFlow import robust so the notebook runs end-to-end and writes `submission.csv`. No changes are made to the architecture, loss, or training loop.'
- What this solution (achieved 0.54471) has done: 'I fix the protobuf/TensorFlow import crash by forcing a safe protobuf runtime version and by importing TensorFlow before any other protobuf-using libraries (like `pydicom`), removing the brittle “restart with cpp/python protobuf” logic that still triggers the `MessageFactory.GetPrototype` error. I keep the model, training loop, data loading heuristics, ensembling, and submission formatting identical, only making the import sequence and environment variables robust so the notebook runs end-to-end. This should be score-neutral (same training/inference semantics) while ensuring a valid `submission.csv` is always produced in the Kaggle environment. I also keep the original bad-case filtering and ID zero-padding unchanged.'
- What this solution (achieved 0.54471) has done: 'I fix the immediate runtime crash (`MessageFactory.GetPrototype`) by setting the protobuf environment variables *before any protobuf-using imports* and by safely falling back to the C++ protobuf implementation if the pure-Python path still fails. This is an execution/stability fix and is intended to be score-neutral (no model/data/training semantic changes). I also correct a clear inference bug where slice predictions for models 2 and 3 were accidentally produced using `model_T2` instead of their respective models, which should legitimately improve AUC toward a higher score. Finally, I keep the submission formatting exactly aligned to `sample_submission.csv` and ensure `submission.csv` is always written.'
- What this solution (achieved 0.54471) has done: 'I fix the immediate TensorFlow/protobuf crash by setting the protobuf env vars before any other imports and by retrying the TensorFlow import once in a clean way (cpp fallback) when the `MessageFactory.GetPrototype` AttributeError occurs; the current try/except doesn’t reliably recover because the partially-imported modules remain in memory. I keep your model, training loop, data loading, ensembling, and submission formatting unchanged to be score-neutral beyond enabling the run to complete. I also renumber the cells starting from 1 (your provided script starts at cell 0) while preserving the original order and logic. Finally, I ensure `submission.csv` is always written with the exact required columns and ID formatting.'
- What this solution (achieved 0.54471) has done: 'I fix the TensorFlow/protobuf `MessageFactory.GetPrototype` crash by setting the protobuf env vars before any protobuf-related imports and by performing a clean, reliable retry (purging already-imported `google.protobuf` / `tensorflow` modules) when the first import fails. This is an execution/stability fix and should be score-neutral because it does not change model architecture, training loop, data loading heuristics, or ensembling logic. I also keep the submission formatting exactly aligned to `sample_submission.csv` and ensure `submission.csv` is always written. Finally, I renumber the cells to start from 1 while preserving the original order and logic.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "cpp")
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

import sys
import random
import numpy as np
import pandas as pd


def _import_tensorflow_with_clean_retry():
    """
    Import TensorFlow robustly. If a protobuf-related crash happens during import,
    purge partially imported modules and retry once with the same (cpp) protobuf.
    This is purely a stability fix; it does not change model/data semantics.
    """
    try:
        import tensorflow as tf  # noqa: E402

        return tf
    except Exception as e:
        msg = str(e)
        if (
            ("GetPrototype" not in msg)
            and ("MessageFactory" not in msg)
            and ("protobuf" not in msg.lower())
        ):
            raise

        for m in list(sys.modules.keys()):
            if m.startswith(("tensorflow", "keras", "google.protobuf")):
                sys.modules.pop(m, None)

        import importlib  # noqa: E402

        importlib.invalidate_caches()

        import tensorflow as tf  # noqa: E402

        return tf


tf = _import_tensorflow_with_clean_retry()

import pydicom  # noqa: E402
from tensorflow.keras import layers  # noqa: E402

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

DATA_ROOT = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification"
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
TEST_DIR = os.path.join(DATA_ROOT, "test")
LABELS_CSV = os.path.join(DATA_ROOT, "train_labels.csv")
SAMPLE_SUB = os.path.join(DATA_ROOT, "sample_submission.csv")

print("TF version:", tf.__version__)
print("Train dir exists:", os.path.isdir(TRAIN_DIR))
print("Test dir exists:", os.path.isdir(TEST_DIR))
print("Labels exist:", os.path.isfile(LABELS_CSV))



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_11/1755533960.py in _import_tensorflow_with_clean_retry()
     23     try:
---> 24         import tensorflow as tf  # noqa: E402
     25 

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
/tmp/ipykernel_11/1755533960.py in <cell line: 0>()
     49 
     50 
---> 51 tf = _import_tensorflow_with_clean_retry()
     52 
     53 import pydicom  # noqa: E402

/tmp/ipykernel_11/1755533960.py in _import_tensorflow_with_clean_retry()
     44         importlib.invalidate_caches()
     45 
---> 46         import tensorflow as tf  # noqa: E402
     47 
     48         return tf

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
IMG_PX_SIZE = 150
CHANNELS = 3
N_SLICES_PER_CASE = 6
MRI_FOLDER_INDEX_T2W = 3  # keep original logic "mri_type[3]" after sorting

BAD_TRAIN_CASES = set(["00109", "00123", "00709"])


def _safe_dcm_to_float(path):
    """Read DICOM safely, return float32 2D array or None on failure."""
    try:
        ds = pydicom.dcmread(path, force=True)
        arr = ds.pixel_array.astype(np.float32)
        if arr.ndim != 2:
            return None
        if not np.isfinite(arr).all():
            return None
        return arr
    except Exception:
        return None


def _resize_to_150(img2d):
    """Resize 2D image to (150,150) using TF bilinear; returns float32."""
    t = tf.convert_to_tensor(img2d[..., None], dtype=tf.float32)  # (H,W,1)
    t = tf.image.resize(
        t, (IMG_PX_SIZE, IMG_PX_SIZE), method="bilinear", antialias=True
    )
    t = tf.squeeze(t, axis=-1)
    return t.numpy().astype(np.float32)


def _normalize_img(img2d):
    """Min-max normalize to [0,1]; returns float32."""
    mn = float(np.min(img2d))
    mx = float(np.max(img2d))
    if mx <= mn:
        return None
    out = (img2d - mn) / (mx - mn)
    if not np.isfinite(out).all():
        return None
    return out.astype(np.float32)


def load_case_slices_T2W(case_path):
    """
    Load up to N_SLICES_PER_CASE slices from T2W folder using original heuristics.
    Returns list length N_SLICES_PER_CASE; if not enough slices found, pads with last/zeros.
    """
    mri_types = sorted([f.path for f in os.scandir(case_path) if f.is_dir()])
    if len(mri_types) <= MRI_FOLDER_INDEX_T2W:
        return None

    t2_path = mri_types[MRI_FOLDER_INDEX_T2W]
    dcm_files = sorted(
        [
            f.path
            for f in os.scandir(t2_path)
            if f.is_file() and f.name.lower().endswith(".dcm")
        ]
    )

    selected = []
    for fp in dcm_files:
        arr = _safe_dcm_to_float(fp)
        if arr is None:
            continue
        if arr.sum() <= 100000:
            continue
        arr = _resize_to_150(arr)
        arr = _normalize_img(arr)
        if arr is None:
            continue
        stacked = np.stack([arr, arr, arr], axis=-1)  # (150,150,3)
        if stacked.sum() <= 2000:
            continue
        selected.append(stacked)
        if len(selected) >= N_SLICES_PER_CASE:
            break

    if len(selected) == 0:
        for fp in dcm_files[:10]:
            arr = _safe_dcm_to_float(fp)
            if arr is None:
                continue
            arr = _resize_to_150(arr)
            arr = _normalize_img(arr)
            if arr is None:
                continue
            stacked = np.stack([arr, arr, arr], axis=-1)
            selected.append(stacked)
            break

    if len(selected) == 0:
        selected = [np.zeros((IMG_PX_SIZE, IMG_PX_SIZE, CHANNELS), dtype=np.float32)]

    while len(selected) < N_SLICES_PER_CASE:
        selected.append(selected[-1].copy())

    return selected[:N_SLICES_PER_CASE]




## === cell 2
labels_df = pd.read_csv(LABELS_CSV)
labels_df["BraTS21ID"] = labels_df["BraTS21ID"].astype(str).str.zfill(5)

train_cases = sorted([d.name for d in os.scandir(TRAIN_DIR) if d.is_dir()])
train_cases = [cid for cid in train_cases if cid not in BAD_TRAIN_CASES]

test_cases = sorted([d.name for d in os.scandir(TEST_DIR) if d.is_dir()])

labels_df = labels_df[labels_df["BraTS21ID"].isin(train_cases)].reset_index(drop=True)

print(
    "Train cases:",
    len(train_cases),
    "Labels rows:",
    len(labels_df),
    "Test cases:",
    len(test_cases),
)
print("Label distribution:\n", labels_df["MGMT_value"].value_counts(dropna=False))




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1873004028.py in <cell line: 0>()
----> 1 labels_df = pd.read_csv(LABELS_CSV)
      2 labels_df["BraTS21ID"] = labels_df["BraTS21ID"].astype(str).str.zfill(5)
      3 
      4 train_cases = sorted([d.name for d in os.scandir(TRAIN_DIR) if d.is_dir()])
      5 train_cases = [cid for cid in train_cases if cid not in BAD_TRAIN_CASES]

NameError: name 'LABELS_CSV' is not defined

## === cell 3
def build_slice_dataset(case_ids, base_dir, labels_map=None, max_cases=None):
    X = []
    y = []
    used_cases = 0
    for cid in case_ids:
        if max_cases is not None and used_cases >= max_cases:
            break
        case_path = os.path.join(base_dir, cid)
        slices = load_case_slices_T2W(case_path)
        if slices is None:
            continue
        X.extend(slices)
        if labels_map is not None:
            y_val = int(labels_map[cid])
            y.extend([y_val] * len(slices))
        used_cases += 1
    X = np.asarray(X, dtype=np.float32)
    if labels_map is None:
        return X
    y = np.asarray(y, dtype=np.int32)
    return X, y


labels_map = dict(zip(labels_df["BraTS21ID"].values, labels_df["MGMT_value"].values))

X_all, y_all = build_slice_dataset(
    labels_df["BraTS21ID"].tolist(), TRAIN_DIR, labels_map=labels_map, max_cases=None
)
print(
    "Training slices:", X_all.shape, "y:", y_all.shape, "y mean:", float(y_all.mean())
)



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3207947991.py in <cell line: 0>()
     22 
     23 
---> 24 labels_map = dict(zip(labels_df["BraTS21ID"].values, labels_df["MGMT_value"].values))
     25 
     26 X_all, y_all = build_slice_dataset(

NameError: name 'labels_df' is not defined

## === cell 4
idx = np.arange(len(y_all))
np.random.shuffle(idx)
split = int(0.85 * len(idx))
tr_idx, va_idx = idx[:split], idx[split:]

X_tr, y_tr = X_all[tr_idx], y_all[tr_idx]
X_va, y_va = X_all[va_idx], y_all[va_idx]

print("Train/val:", X_tr.shape, X_va.shape)


def build_model():
    inp = tf.keras.Input(shape=(IMG_PX_SIZE, IMG_PX_SIZE, CHANNELS))
    x = layers.Conv2D(16, 3, padding="same", activation="relu")(inp)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.25)(x)
    out = layers.Dense(2, activation="softmax")(x)
    model = tf.keras.Model(inp, out)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")],
    )
    return model


model_T2 = build_model()
model_T2_2 = build_model()
model_T2_3 = build_model()



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/649864831.py in <cell line: 0>()
----> 1 idx = np.arange(len(y_all))
      2 np.random.shuffle(idx)
      3 split = int(0.85 * len(idx))
      4 tr_idx, va_idx = idx[:split], idx[split:]
      5 

NameError: name 'y_all' is not defined

## === cell 5
BATCH_SIZE = 32
EPOCHS = 3


def train_one(model, seed_offset=0):
    tf.random.set_seed(SEED + seed_offset)
    history = model.fit(
        X_tr,
        y_tr,
        validation_data=(X_va, y_va),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=2,
    )
    return history


train_one(model_T2, seed_offset=1)
train_one(model_T2_2, seed_offset=2)
train_one(model_T2_3, seed_offset=3)




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1016075007.py in <cell line: 0>()
     16 
     17 
---> 18 train_one(model_T2, seed_offset=1)
     19 train_one(model_T2_2, seed_offset=2)
     20 train_one(model_T2_3, seed_offset=3)

NameError: name 'model_T2' is not defined

## === cell 6
def load_test_T2W_images(path_test):
    arrays = [[] for _ in range(N_SLICES_PER_CASE)]
    case_ids = sorted([f.name for f in os.scandir(path_test) if f.is_dir()])

    n_failed = 0
    for cid in case_ids:
        case_path = os.path.join(path_test, cid)
        slices = load_case_slices_T2W(case_path)
        if slices is None:
            n_failed += 1
            slices = [
                np.zeros((IMG_PX_SIZE, IMG_PX_SIZE, CHANNELS), dtype=np.float32)
            ] * N_SLICES_PER_CASE
        for i in range(N_SLICES_PER_CASE):
            arrays[i].append(slices[i])

    arrays = [np.asarray(a, dtype=np.float32) for a in arrays]
    print(
        "Number of T2 images loaded are",
        ", ".join(str(len(a)) for a in arrays),
        "| failed cases:",
        n_failed,
    )
    return case_ids, arrays


test = TEST_DIR
test_case_ids, arrays = load_test_T2W_images(test)
pixels_1, pixels_2, pixels_3, pixels_4, pixels_5, pixels_6 = arrays



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/197332851.py in <cell line: 0>()
     25 
     26 
---> 27 test = TEST_DIR
     28 test_case_ids, arrays = load_test_T2W_images(test)
     29 pixels_1, pixels_2, pixels_3, pixels_4, pixels_5, pixels_6 = arrays

NameError: name 'TEST_DIR' is not defined

## === cell 7
preds_1 = model_T2.predict(pixels_1, verbose=0)
prediction_1 = preds_1[:, 1]
preds_2 = model_T2.predict(pixels_2, verbose=0)
prediction_2 = preds_2[:, 1]
preds_3 = model_T2.predict(pixels_3, verbose=0)
prediction_3 = preds_3[:, 1]
preds_4 = model_T2.predict(pixels_4, verbose=0)
prediction_4 = preds_4[:, 1]
preds_5 = model_T2.predict(pixels_5, verbose=0)
prediction_5 = preds_5[:, 1]
preds_6 = model_T2.predict(pixels_6, verbose=0)
prediction_6 = preds_6[:, 1]

preds_101 = model_T2_2.predict(pixels_1, verbose=0)
prediction_101 = preds_101[:, 1]
preds_102 = model_T2_2.predict(pixels_2, verbose=0)
prediction_102 = preds_102[:, 1]
preds_103 = model_T2_2.predict(pixels_3, verbose=0)
prediction_103 = preds_103[:, 1]
preds_104 = model_T2_2.predict(pixels_4, verbose=0)
prediction_104 = preds_104[:, 1]
preds_105 = model_T2_2.predict(pixels_5, verbose=0)
prediction_105 = preds_105[:, 1]
preds_106 = model_T2_2.predict(pixels_6, verbose=0)
prediction_106 = preds_106[:, 1]

preds_201 = model_T2_3.predict(pixels_1, verbose=0)
prediction_201 = preds_201[:, 1]
preds_202 = model_T2_3.predict(pixels_2, verbose=0)
prediction_202 = preds_202[:, 1]
preds_203 = model_T2_3.predict(pixels_3, verbose=0)
prediction_203 = preds_203[:, 1]
preds_204 = model_T2_3.predict(pixels_4, verbose=0)
prediction_204 = preds_204[:, 1]
preds_205 = model_T2_3.predict(pixels_5, verbose=0)
prediction_205 = preds_205[:, 1]
preds_206 = model_T2_3.predict(pixels_6, verbose=0)
prediction_206 = preds_206[:, 1]




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2476749253.py in <cell line: 0>()
----> 1 preds_1 = model_T2.predict(pixels_1, verbose=0)
      2 prediction_1 = preds_1[:, 1]
      3 preds_2 = model_T2.predict(pixels_2, verbose=0)
      4 prediction_2 = preds_2[:, 1]
      5 preds_3 = model_T2.predict(pixels_3, verbose=0)

NameError: name 'model_T2' is not defined

## === cell 8
def create_sub(
    case_ids,
    p1,
    p2,
    p3,
    p4,
    p5,
    p6,
    p101,
    p102,
    p103,
    p104,
    p105,
    p106,
    p201,
    p202,
    p203,
    p204,
    p205,
    p206,
):
    prediction = (
        p1.astype(np.float32)
        + p2.astype(np.float32)
        + p3.astype(np.float32)
        + p4.astype(np.float32)
        + p5.astype(np.float32)
        + p6.astype(np.float32)
        + p101.astype(np.float32)
        + p102.astype(np.float32)
        + p103.astype(np.float32)
        + p104.astype(np.float32)
        + p105.astype(np.float32)
        + p106.astype(np.float32)
        + p201.astype(np.float32)
        + p202.astype(np.float32)
        + p203.astype(np.float32)
        + p204.astype(np.float32)
        + p205.astype(np.float32)
        + p206.astype(np.float32)
    ) / 18.0

    df = pd.DataFrame(
        {
            "BraTS21ID": pd.Series(case_ids, dtype=str).str.zfill(5),
            "MGMT_value": prediction,
        }
    )
    return df


sub_df = create_sub(
    test_case_ids,
    prediction_1,
    prediction_2,
    prediction_3,
    prediction_4,
    prediction_5,
    prediction_6,
    prediction_101,
    prediction_102,
    prediction_103,
    prediction_104,
    prediction_105,
    prediction_106,
    prediction_201,
    prediction_202,
    prediction_203,
    prediction_204,
    prediction_205,
    prediction_206,
)

print(sub_df.head())
print(
    "Rows:",
    len(sub_df),
    "MGMT_value range:",
    (float(sub_df["MGMT_value"].min()), float(sub_df["MGMT_value"].max())),
)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2475231519.py in <cell line: 0>()
     51 
     52 sub_df = create_sub(
---> 53     test_case_ids,
     54     prediction_1,
     55     prediction_2,

NameError: name 'test_case_ids' is not defined

## === cell 9
sample = pd.read_csv(SAMPLE_SUB)
sample["BraTS21ID"] = sample["BraTS21ID"].astype(str).str.zfill(5)

sub_df["BraTS21ID"] = sub_df["BraTS21ID"].astype(str).str.zfill(5)
sub_df = sample[["BraTS21ID"]].merge(sub_df, on="BraTS21ID", how="left")

sub_df["MGMT_value"] = sub_df["MGMT_value"].astype(float).fillna(0.5).clip(0.0, 1.0)

sub_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub_df.shape)
print(sub_df.head())
print("Any NaNs:", sub_df.isna().any().to_dict())

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3581874581.py in <cell line: 0>()
----> 1 sample = pd.read_csv(SAMPLE_SUB)
      2 sample["BraTS21ID"] = sample["BraTS21ID"].astype(str).str.zfill(5)
      3 
      4 sub_df["BraTS21ID"] = sub_df["BraTS21ID"].astype(str).str.zfill(5)
      5 sub_df = sample[["BraTS21ID"]].merge(sub_df, on="BraTS21ID", how="left")

NameError: name 'SAMPLE_SUB' is not defined
