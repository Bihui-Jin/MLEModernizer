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

0.8928

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.9832) has done: 'The immediate blocker is the TensorFlow import crash caused by an incompatible protobuf runtime (your current `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` setting triggers the `MessageFactory.GetPrototype` error in this Kaggle image). I switch to the default C++ protobuf implementation and force it to be set before importing TensorFlow, which fixes the runtime error and lets training/prediction complete. To fix the “Invalid submission: ... same number of rows” issue, I make the data root detection deterministic (use the correct Kaggle folder), and I always build predictions in exactly the same row order as `sample_submission.csv` and validate against it before writing `submission.csv`. These are execution/format fixes and should be score-neutral beyond negligible numeric differences.'
- What this solution (achieved 0.99344) has done: 'I fix the TensorFlow/protobuf crash by forcing the pure-Python protobuf implementation *before any TensorFlow import* (the current “pop env vars” still leaves the runtime picking an incompatible protobuf path in this Kaggle image). I also make dataset root detection more robust by checking for both CSVs and ZIPs so we always unzip the intended files, and I ensure the test directory is the one actually containing exactly the IDs in `sample_submission.csv`. Finally, I keep the core model/training logic unchanged, but I hard-validate that prediction length and ID order exactly match `sample_submission.csv` before writing `submission.csv`, preventing the “same number of rows” submission error.'
- What this solution (achieved 0.5) has done: 'I fix the TensorFlow/protobuf import crash by removing the forced pure-Python protobuf setting (it’s what triggers the `MessageFactory.GetPrototype` error in this Kaggle image) and explicitly forcing the default C++ implementation before importing TensorFlow. This is an execution-only fix and should be score-neutral (negligible numeric differences at most). I also keep your dataset root detection, generators, training loop, and submission alignment checks unchanged, so the pipeline still trains and writes a valid `submission.csv` matching `sample_submission.csv` row order.'
- What this solution (achieved 0.5) has done: 'I fix the TensorFlow import crash by removing the forced pure-Python protobuf environment variables (they trigger the `MessageFactory.GetPrototype` error in this Kaggle image) and explicitly forcing the default C++ protobuf implementation before TensorFlow is imported. Then I make the data root detection deterministic for this dataset layout and ensure we always unzip into a clean, known folder so we don’t accidentally pick up nested `.../test/test` folders that can cause row-count mismatches. Finally, I keep your generators/model/training intact but tighten submission alignment: predictions be generated in exactly the `sample_submission.csv` row order and we validate count and IDs before writing `submission.csv` to guarantee Kaggle accepts it.'
- What this solution (achieved 0.5) has done: 'I fix the data extraction/directory discovery so `TRAIN_DIR` and `TEST_DIR` are correctly found after unzipping (the current `rglob('train')` misses the typical `train/` folder because it’s searching for an exact-name directory without a trailing slash match). I also fix the TensorFlow/protobuf import crash by forcing the pure-Python protobuf implementation (and disabling C descriptors) *before* importing TensorFlow, which avoids the missing `google.protobuf.pyext._message` error in this environment. These changes unblock training and prediction, and should move your AUC up from 0.5 toward the target by producing real model probabilities instead of failing/degenerate output. Finally, I keep your model, generators, training loop, and submission alignment checks intact so the evaluation semantics remain the same and the submission is guaranteed valid.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_DISABLE_C_DESCRIPTORS", None)
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "cpp")

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

for dirname, _, filenames in os.walk("/kaggle/input/aerial-cactus-identification"):
    for i, filename in enumerate(filenames[:5]):
        print(os.path.join(dirname, filename))
    break



## === cell 1
import subprocess, pathlib, sys, shutil

workdir = "/kaggle/working"
os.makedirs(workdir, exist_ok=True)


def run(cmd):
    print(cmd)
    subprocess.check_call(cmd, shell=True)


CANDIDATES = [
    "/kaggle/input/aerial-cactus-identification/aerial-cactus-identification",
    "/kaggle/input/aerial-cactus-identification",
]

DATA_ROOT = None
for p in CANDIDATES:
    if (
        os.path.isfile(os.path.join(p, "train.csv"))
        and os.path.isfile(os.path.join(p, "sample_submission.csv"))
        and os.path.isfile(os.path.join(p, "train.zip"))
        and os.path.isfile(os.path.join(p, "test.zip"))
    ):
        DATA_ROOT = p
        break
if DATA_ROOT is None:
    raise FileNotFoundError(f"Could not find dataset root in candidates: {CANDIDATES}")

print("Using DATA_ROOT:", DATA_ROOT)

EXTRACT_ROOT = os.path.join(workdir, "aerial_cactus_extracted")
if os.path.isdir(EXTRACT_ROOT):
    shutil.rmtree(EXTRACT_ROOT, ignore_errors=True)
os.makedirs(EXTRACT_ROOT, exist_ok=True)

run(f"cp -f {DATA_ROOT}/train.csv {workdir}/train.csv")
run(f"cp -f {DATA_ROOT}/sample_submission.csv {workdir}/sample_submission.csv")
run(f"unzip -oq {DATA_ROOT}/train.zip -d {EXTRACT_ROOT}")
run(f"unzip -oq {DATA_ROOT}/test.zip -d {EXTRACT_ROOT}")

train_csv_path = os.path.join(workdir, "train.csv")
sample_sub_path = os.path.join(workdir, "sample_submission.csv")
train_df_full = pd.read_csv(train_csv_path)
sample_df = pd.read_csv(sample_sub_path)

expected_train_ids = set(train_df_full["id"].astype(str).tolist())
expected_test_ids = set(sample_df["id"].astype(str).tolist())
print("Expected train images:", len(expected_train_ids))
print("Expected test images :", len(expected_test_ids))


def find_best_image_dir(root, leaf, expected_ids):
    """
    FIX: Robustly locate the correct extracted train/test directory.
    We score candidate directories named leaf by coverage of expected_ids
    and prefer shallower paths. This handles nested `.../test/test` layouts.
    """
    root_p = pathlib.Path(root)
    candidates = []
    for p in root_p.rglob(leaf):
        if p.is_dir():
            if any(p.glob("*.jpg")):
                candidates.append(p)

    if not candidates:
        return None, 0.0

    ids_list = list(expected_ids)
    best = None
    best_cov = -1.0
    best_depth = 10**9

    for p in candidates:
        cov = float(np.mean([os.path.isfile(p / i) for i in ids_list]))
        depth = len(p.parts)

        parts = p.parts
        dup_penalty = int(len(parts) >= 2 and parts[-1] == leaf and parts[-2] == leaf)

        if (cov > best_cov) or (
            cov == best_cov
            and (dup_penalty, depth) < (0 if best is None else 0, best_depth)
        ):
            best, best_cov, best_depth = p, cov, depth

        if cov >= 0.9999:
            break

    return (str(best) if best is not None else None), best_cov


TRAIN_DIR, train_cov = find_best_image_dir(EXTRACT_ROOT, "train", expected_train_ids)
TEST_DIR, test_cov = find_best_image_dir(EXTRACT_ROOT, "test", expected_test_ids)

print("Detected TRAIN_DIR:", TRAIN_DIR, "coverage:", train_cov)
print("Detected TEST_DIR :", TEST_DIR, "coverage:", test_cov)

if TRAIN_DIR is None or train_cov < 0.99:
    print("Debug: extracted top-level:", sorted(os.listdir(EXTRACT_ROOT))[:50])
    raise FileNotFoundError(
        f"Could not locate extracted TRAIN_DIR with sufficient coverage. TRAIN_DIR={TRAIN_DIR}, coverage={train_cov}"
    )
if TEST_DIR is None or test_cov < 0.99:
    print("Debug: extracted top-level:", sorted(os.listdir(EXTRACT_ROOT))[:50])
    raise FileNotFoundError(
        f"Could not locate extracted TEST_DIR with sufficient coverage. TEST_DIR={TEST_DIR}, coverage={test_cov}"
    )



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/1858195486.py in <cell line: 0>()
    104 if TRAIN_DIR is None or train_cov < 0.99:
    105     print("Debug: extracted top-level:", sorted(os.listdir(EXTRACT_ROOT))[:50])
--> 106     raise FileNotFoundError(
    107         f"Could not locate extracted TRAIN_DIR with sufficient coverage. TRAIN_DIR={TRAIN_DIR}, coverage={train_cov}"
    108     )

FileNotFoundError: Could not locate extracted TRAIN_DIR with sufficient coverage. TRAIN_DIR=None, coverage=0.0

## === cell 2
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

print("tf version:", tf.__version__)
print("Available GPUs:", tf.config.list_physical_devices("GPU"))

tf.keras.utils.set_random_seed(42)
np.random.seed(42)



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_11/3421389312.py in <cell line: 0>()
      1 import matplotlib.pyplot as plt
      2 
----> 3 import tensorflow as tf
      4 from tensorflow import keras
      5 from sklearn.model_selection import train_test_split

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

## === cell 3
df = pd.read_csv("/kaggle/working/train.csv")
print(df.head())
df.has_cactus.value_counts().plot.bar()
plt.show()



## === cell 4
from tensorflow.keras.utils import load_img

filename = df.id.iloc[10]
print("Example file:", filename)
image_path = os.path.join(TRAIN_DIR, filename)
print("Resolved path:", image_path, "exists:", os.path.isfile(image_path))
image = load_img(image_path)
plt.imshow(image)
plt.axis("off")
plt.show()



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_11/2647921589.py in <cell line: 0>()
----> 1 from tensorflow.keras.utils import load_img
      2 
      3 filename = df.id.iloc[10]
      4 print("Example file:", filename)
      5 image_path = os.path.join(TRAIN_DIR, filename)

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

## === cell 5
try:
    train_test_split
except NameError:
    from sklearn.model_selection import train_test_split

train_df, validate_df = train_test_split(
    df, test_size=0.20, random_state=42, stratify=df["has_cactus"]
)
train_df = train_df.reset_index(drop=True)
validate_df = validate_df.reset_index(drop=True)

print(train_df.shape, validate_df.shape)

train_exists = (
    train_df["id"].apply(lambda x: os.path.isfile(os.path.join(TRAIN_DIR, x))).mean()
)
valid_exists = (
    validate_df["id"].apply(lambda x: os.path.isfile(os.path.join(TRAIN_DIR, x))).mean()
)
print("Train files present ratio:", train_exists)
print("Valid files present ratio:", valid_exists)
if train_exists < 0.99 or valid_exists < 0.99:
    missing = (
        train_df.loc[
            ~train_df["id"].apply(lambda x: os.path.isfile(os.path.join(TRAIN_DIR, x))),
            "id",
        ]
        .head(5)
        .tolist()
    )
    raise FileNotFoundError(
        f"Some training images are missing under TRAIN_DIR={TRAIN_DIR}. Example missing: {missing}"
    )



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1989387007.py in <cell line: 0>()
     13 
     14 train_exists = (
---> 15     train_df["id"].apply(lambda x: os.path.isfile(os.path.join(TRAIN_DIR, x))).mean()
     16 )
     17 valid_exists = (

/usr/local/lib/python3.11/dist-packages/pandas/core/series.py in apply(self, func, convert_dtype, args, by_row, **kwargs)
   4922             args=args,
   4923             kwargs=kwargs,
-> 4924         ).apply()
   4925 
   4926     def _reindex_indexer(

/usr/local/lib/python3.11/dist-packages/pandas/core/apply.py in apply(self)
   1425 
   1426         # self.func is Callable
-> 1427         return self.apply_standard()
   1428 
   1429     def agg(self):

/usr/local/lib/python3.11/dist-packages/pandas/core/apply.py in apply_standard(self)
   1505         #  Categorical (GH51645).
   1506         action = "ignore" if isinstance(obj.dtype, CategoricalDtype) else None
-> 1507         mapped = obj._map_values(
   1508             mapper=curried, na_action=action, convert=self.convert_dtype
   1509         )

/usr/local/lib/python3.11/dist-packages/pandas/core/base.py in _map_values(self, mapper, na_action, convert)
    919             return arr.map(mapper, na_action=na_action)
    920 
--> 921         return algorithms.map_array(arr, mapper, na_action=na_action, convert=convert)
    922 
    923     @final

/usr/local/lib/python3.11/dist-packages/pandas/core/algorithms.py in map_array(arr, mapper, na_action, convert)
   1741     values = arr.astype(object, copy=False)
   1742     if na_action is None:
-> 1743         return lib.map_infer(values, mapper, convert=convert)
   1744     else:
   1745         return lib.map_infer_mask(

lib.pyx in pandas._libs.lib.map_infer()

/tmp/ipykernel_11/1989387007.py in <lambda>(x)
     13 
     14 train_exists = (
---> 15     train_df["id"].apply(lambda x: os.path.isfile(os.path.join(TRAIN_DIR, x))).mean()
     16 )
     17 valid_exists = (

/usr/lib/python3.11/posixpath.py in join(a, *p)

TypeError: expected str, bytes or os.PathLike object, not NoneType

## === cell 6
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rotation_range=15,
    rescale=1.0 / 255.0,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
)

valid_datagen = ImageDataGenerator(rescale=1.0 / 255.0)



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_11/4120657998.py in <cell line: 0>()
----> 1 from tensorflow.keras.preprocessing.image import ImageDataGenerator
      2 
      3 train_datagen = ImageDataGenerator(
      4     rotation_range=15,
      5     rescale=1.0 / 255.0,

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
BATCH_SIZE = 2**10
IMAGE_SIZE = (32, 32)
INPUT_SHAPE = (32, 32, 3)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=TRAIN_DIR,
    x_col="id",
    y_col="has_cactus",
    target_size=IMAGE_SIZE,
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    class_mode="raw",
    shuffle=True,
    seed=42,
)

validation_generator = valid_datagen.flow_from_dataframe(
    dataframe=validate_df,
    directory=TRAIN_DIR,
    x_col="id",
    y_col="has_cactus",
    target_size=IMAGE_SIZE,
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    class_mode="raw",
    shuffle=False,
)

print("train_generator batches:", len(train_generator))
print("validation_generator batches:", len(validation_generator))
if len(train_generator) == 0 or len(validation_generator) == 0:
    raise ValueError(
        f"Generator length is 0. TRAIN_DIR={TRAIN_DIR}. "
        f"Check that images exist and dataframe ids match filenames."
    )



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/360972142.py in <cell line: 0>()
      3 INPUT_SHAPE = (32, 32, 3)
      4 
----> 5 train_generator = train_datagen.flow_from_dataframe(
      6     dataframe=train_df,
      7     directory=TRAIN_DIR,

NameError: name 'train_datagen' is not defined

## === cell 8
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    Flatten,
    Dense,
    BatchNormalization,
    Dropout,
    AveragePooling2D,
)
from tensorflow.keras.callbacks import EarlyStopping

model = Sequential(
    [
        Conv2D(
            filters=64,
            kernel_size=(4, 4),
            strides=(1, 1),
            activation="relu",
            input_shape=INPUT_SHAPE,
            padding="same",
        ),
        BatchNormalization(),
        AveragePooling2D(pool_size=(3, 3)),
        Dropout(0.2),
        Flatten(),
        Dense(128, activation="relu"),
        Dense(32, activation="relu"),
        Dropout(0.45),
        Dense(1, activation="sigmoid"),
    ]
)

earlystop = EarlyStopping(patience=4, restore_best_weights=True)
model.compile(loss="binary_crossentropy", optimizer="nadam", metrics=["accuracy"])
model.summary()



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_11/624455915.py in <cell line: 0>()
----> 1 from tensorflow.keras.models import Sequential
      2 from tensorflow.keras.layers import (
      3     Conv2D,
      4     Flatten,
      5     Dense,

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

## === cell 9
history = model.fit(
    train_generator,
    epochs=30,
    validation_data=validation_generator,
    callbacks=[earlystop],
    verbose=2,
)



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/731527035.py in <cell line: 0>()
----> 1 history = model.fit(
      2     train_generator,
      3     epochs=30,
      4     validation_data=validation_generator,
      5     callbacks=[earlystop],

NameError: name 'model' is not defined

## === cell 10
pd.DataFrame(history.history).plot()
plt.grid(True)
plt.show()



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/206874453.py in <cell line: 0>()
----> 1 pd.DataFrame(history.history).plot()
      2 plt.grid(True)
      3 plt.show()
      4 

NameError: name 'history' is not defined

## === cell 11
sample_sub_path = os.path.join("/kaggle/working", "sample_submission.csv")
sub_df = pd.read_csv(sample_sub_path)

assert (
    "id" in sub_df.columns and "has_cactus" in sub_df.columns
), "sample_submission.csv must contain columns: id, has_cactus"

test_exists = (
    sub_df["id"].apply(lambda x: os.path.isfile(os.path.join(TEST_DIR, x))).mean()
)
print("Test files present ratio:", test_exists)
if test_exists < 0.99:
    missing = (
        sub_df.loc[
            ~sub_df["id"].apply(lambda x: os.path.isfile(os.path.join(TEST_DIR, x))),
            "id",
        ]
        .head(10)
        .tolist()
    )
    raise FileNotFoundError(
        f"Some test images are missing under TEST_DIR={TEST_DIR}. Example missing: {missing}"
    )

test_gen = ImageDataGenerator(rescale=1.0 / 255.0)
test_generator = test_gen.flow_from_dataframe(
    dataframe=sub_df[["id"]],  # preserve exact row order of sample_submission
    directory=TEST_DIR,
    x_col="id",
    y_col=None,
    class_mode=None,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

print("test_generator batches:", len(test_generator))
if len(test_generator) == 0:
    raise ValueError(f"Test generator length is 0. TEST_DIR={TEST_DIR}.")

pred = model.predict(test_generator, verbose=0).reshape(-1)

n = len(sub_df)
if len(pred) != n:
    raise ValueError(
        f"Prediction length mismatch: got {len(pred)} predictions for {n} test ids."
    )

sub_df["has_cactus"] = pred.astype(np.float32)

print(sub_df.head())
print(
    "Pred min/max:",
    float(sub_df["has_cactus"].min()),
    float(sub_df["has_cactus"].max()),
)



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/877432743.py in <cell line: 0>()
      7 
      8 test_exists = (
----> 9     sub_df["id"].apply(lambda x: os.path.isfile(os.path.join(TEST_DIR, x))).mean()
     10 )
     11 print("Test files present ratio:", test_exists)

/usr/local/lib/python3.11/dist-packages/pandas/core/series.py in apply(self, func, convert_dtype, args, by_row, **kwargs)
   4922             args=args,
   4923             kwargs=kwargs,
-> 4924         ).apply()
   4925 
   4926     def _reindex_indexer(

/usr/local/lib/python3.11/dist-packages/pandas/core/apply.py in apply(self)
   1425 
   1426         # self.func is Callable
-> 1427         return self.apply_standard()
   1428 
   1429     def agg(self):

/usr/local/lib/python3.11/dist-packages/pandas/core/apply.py in apply_standard(self)
   1505         #  Categorical (GH51645).
   1506         action = "ignore" if isinstance(obj.dtype, CategoricalDtype) else None
-> 1507         mapped = obj._map_values(
   1508             mapper=curried, na_action=action, convert=self.convert_dtype
   1509         )

/usr/local/lib/python3.11/dist-packages/pandas/core/base.py in _map_values(self, mapper, na_action, convert)
    919             return arr.map(mapper, na_action=na_action)
    920 
--> 921         return algorithms.map_array(arr, mapper, na_action=na_action, convert=convert)
    922 
    923     @final

/usr/local/lib/python3.11/dist-packages/pandas/core/algorithms.py in map_array(arr, mapper, na_action, convert)
   1741     values = arr.astype(object, copy=False)
   1742     if na_action is None:
-> 1743         return lib.map_infer(values, mapper, convert=convert)
   1744     else:
   1745         return lib.map_infer_mask(

lib.pyx in pandas._libs.lib.map_infer()

/tmp/ipykernel_11/877432743.py in <lambda>(x)
      7 
      8 test_exists = (
----> 9     sub_df["id"].apply(lambda x: os.path.isfile(os.path.join(TEST_DIR, x))).mean()
     10 )
     11 print("Test files present ratio:", test_exists)

/usr/lib/python3.11/posixpath.py in join(a, *p)

TypeError: expected str, bytes or os.PathLike object, not NoneType

## === cell 12
submission = sub_df[["id", "has_cactus"]].copy()
out_path = "/kaggle/working/submission.csv"

expected = pd.read_csv(sample_sub_path)
if len(submission) != len(expected):
    raise ValueError(f"Row mismatch: {len(submission)} vs {len(expected)}")
if list(submission.columns) != ["id", "has_cactus"]:
    raise ValueError(f"Column mismatch: {submission.columns.tolist()}")
if submission["id"].tolist() != expected["id"].tolist():
    raise ValueError("ID order mismatch vs sample_submission.csv")

submission.to_csv(out_path, index=False)
print("Wrote:", out_path, "rows:", len(submission))



## === cell 13
print(submission.columns.tolist())
print(submission.isna().sum())
print(submission.head())
print(submission["has_cactus"].describe())



## === cell 14
print(os.listdir("/kaggle/working")[:50])
print("Submission exists:", os.path.isfile("/kaggle/working/submission.csv"))

expected = pd.read_csv(sample_sub_path)
assert len(submission) == len(
    expected
), f"Row mismatch: {len(submission)} vs {len(expected)}"
assert list(submission.columns) == ["id", "has_cactus"]
assert (
    submission["id"].tolist() == expected["id"].tolist()
), "ID order mismatch vs sample_submission.csv"
print("Submission validated with rows:", len(expected))

## --- ERROR in outputing the csv:
Invalid submission: Submission and answers should have the same number of rows
