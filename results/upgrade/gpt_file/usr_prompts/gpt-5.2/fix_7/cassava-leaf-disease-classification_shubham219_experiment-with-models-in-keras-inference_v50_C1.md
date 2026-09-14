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

0.6177092777274101

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.17339) has done: 'I remove the incompatible `tensorflow_hub` import that is triggering the protobuf `MessageFactory.GetPrototype` crash, since it’s not used by your pipeline. I also replace the notebook `!pip install` cells with a safe in-notebook install fallback and, more importantly, add a robust import/load path: if the external `efficientnet` package or the provided weight file is unavailable, we fall back to a standard `tf.keras.applications.EfficientNetB4` model so `my_model` is always defined. Finally, I ensure the test image path is correct for this environment, fix generator sizing, and write a valid `submission.csv` with `image_id,label` columns.'
- What this solution (achieved 0.0938) has done: 'We need to fix the TensorFlow import crash happening immediately in cell 0 (`MessageFactory` protobuf incompatibility). The most reliable minimal fix in Kaggle is to force the pure-Python protobuf implementation before importing TensorFlow, which avoids the failing C++/upb path. After that, we keep your existing inference-only pipeline intact, but improve score toward the 0.617 target by ensuring the intended pretrained `.h5` model loads when available and, if it doesn’t, compiling the fallback model and loading ImageNet weights correctly (otherwise predictions are effectively random, matching your very low 0.173 score). Finally, we keep the submission formatting checks and guarantee `submission.csv` is written.'
- What this solution (achieved 0.05531) has done: 'We fix the immediate TensorFlow import crash (`MessageFactory.GetPrototype`) by forcing TensorFlow to use the pure-Python protobuf runtime *before* TensorFlow is imported, and by explicitly setting `TF_USE_LEGACY_KERAS=1` (a common Kaggle TF compatibility requirement) plus avoiding any TF/keras imports until after those env vars are set. Then we keep your inference-only pipeline intact, but ensure the fallback model’s head is not random (which is why accuracy is near chance): if the custom `.h5` isn’t available, we load the base EfficientNetB4 ImageNet weights and use the standard EfficientNet preprocessing; this preserves the architecture while making predictions meaningful. Finally, we keep the submission alignment with `sample_submission.csv` and guarantee `submission.csv` is created with the exact required columns and row count.'
- What this solution (achieved 0.17377) has done: 'We fix the TensorFlow import crash (`MessageFactory.GetPrototype`) by setting additional protobuf-related environment variables *before* importing TensorFlow, which is the root cause preventing the pipeline from running at all. Since your current score (0.05531) is far below target, we also remove the “zeros-initialized” random/untrained classification head in the fallback model and instead use the standard ImageNet-pretrained EfficientNetB4 top classifier (1000 classes) and map its predictions deterministically into 5 labels; this is a minimal change that makes fallback predictions non-degenerate without altering your inference-only flow. We keep all paths and submission formatting logic intact and still write `submission.csv` with the required columns and row count. Finally, we add a tiny safety fallback for preprocessing to ensure it matches the actual model used.'
- What this solution (achieved 0.31203) has done: 'We fix the TensorFlow import crash (`MessageFactory` / protobuf mismatch) by setting the protobuf/TF environment variables *as early as possible* and adding a safe fallback to force the pure-Python protobuf backend before importing TensorFlow. Then we remove the currently-harmful “ImageNet-top then mod-5” mapping (it produces near-random labels and matches your low score) by ensuring the intended 5-class `.h5` model loads when available, and otherwise building a 5-class EfficientNetB4 head (same backbone) so predictions are at least aligned to the cassava label space. Finally, we make preprocessing conditional on which EfficientNet implementation is actually used (efficientnet.tfkeras vs tf.keras.applications) to avoid silent input scaling mismatches, and keep the exact required `submission.csv` format and ordering.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "3")
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import glob
import numpy as np
import pandas as pd

SEED = 42
DEBUG = False

os.environ["PYTHONHASHSEED"] = str(SEED)
np.random.seed(SEED)



## === cell 1
import sys
import subprocess


def _pip_uninstall(pkg: str):
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "uninstall", "-y", "-q", pkg]
        )
        print(f"Uninstalled (if present): {pkg}")
    except Exception as e:
        print(f"Warning: uninstall {pkg} failed (continuing): {pkg} -> {e}")


_pip_uninstall("tensorflow-hub")  # not used; can pull incompatible deps on some images
_pip_uninstall("protobuf")



## === cell 2
try:
    import tensorflow as tf
except Exception as e:
    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "3"
    os.environ["TF_USE_LEGACY_KERAS"] = "1"
    import importlib

    for m in list(sys.modules.keys()):
        if (
            m.startswith("tensorflow")
            or m.startswith("google.protobuf")
            or m.startswith("protobuf")
        ):
            sys.modules.pop(m, None)
    importlib.invalidate_caches()
    import tensorflow as tf  # retry

tf.random.set_seed(SEED)

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

print("TF version:", tf.__version__)




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/1515940568.py in <cell line: 0>()
      1 try:
----> 2     import tensorflow as tf
      3 except Exception as e:

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
     31 
---> 32 from google.protobuf import message
     33 from tensorflow.core.framework import attr_value_pb2

ModuleNotFoundError: No module named 'google.protobuf'

During handling of the above exception, another exception occurred:

ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/1515940568.py in <cell line: 0>()
     17             sys.modules.pop(m, None)
     18     importlib.invalidate_caches()
---> 19     import tensorflow as tf  # retry
     20 
     21 tf.random.set_seed(SEED)

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
     30 from numpy import typing as npt
     31 
---> 32 from google.protobuf import message
     33 from tensorflow.core.framework import attr_value_pb2
     34 from tensorflow.core.framework import full_type_pb2

ModuleNotFoundError: No module named 'google.protobuf'

## === cell 3
def _maybe_pip_install(wheel_path: str):
    if os.path.exists(wheel_path):
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", wheel_path]
            )
            print(f"Installed: {wheel_path}")
        except Exception as e:
            print(f"Warning: failed to install {wheel_path}: {e}")
    else:
        print(f"Wheel not found (skipping): {wheel_path}")


_maybe_pip_install(
    "/kaggle/input/kerasapplication/Keras_Applications-1.0.8-py3-none-any.whl"
)
_maybe_pip_install("/kaggle/input/efficientnet/efficientnet-1.1.1-py3-none-any.whl")



## === cell 4
my_model = None
USING_EFFICIENTNET_TFKERAS = False  # controls preprocessing choice

try:
    from efficientnet.tfkeras import EfficientNetB4 as EffNetB4_tfk  # noqa: F401

    USING_EFFICIENTNET_TFKERAS = True
    print("Imported efficientnet.tfkeras")
except Exception as e:
    print("Warning: could not import efficientnet.tfkeras:", repr(e))

candidate_weight_paths = [
    "../input/experiment-with-models-using-keras-with-updates/effnetB4_v0.25.h5",
    "/kaggle/input/experiment-with-models-using-keras-with-updates/effnetB4_v0.25.h5",
]

weight_path = None
for p in candidate_weight_paths:
    if os.path.exists(p):
        weight_path = p
        break

if weight_path is not None:
    try:
        my_model = load_model(weight_path, compile=False)
        print("Loaded model from:", weight_path)
    except Exception as e1:
        print("Warning: first load_model failed:", repr(e1))
        try:
            custom_objects = {}
            if USING_EFFICIENTNET_TFKERAS:
                from efficientnet.tfkeras import EfficientNetB4 as EffB4_custom

                custom_objects["EfficientNetB4"] = EffB4_custom
            my_model = load_model(
                weight_path, compile=False, custom_objects=custom_objects
            )
            print("Loaded model from with custom_objects:", weight_path)
        except Exception as e2:
            print(
                "Warning: failed to load .h5 model, will fall back to base model:",
                repr(e2),
            )
            my_model = None

if my_model is None:
    from tensorflow.keras import layers, Model

    if USING_EFFICIENTNET_TFKERAS:
        from efficientnet.tfkeras import EfficientNetB4

        base = EfficientNetB4(
            include_top=False, weights="imagenet", input_shape=(300, 300, 3)
        )
    else:
        from tensorflow.keras.applications import EfficientNetB4

        base = EfficientNetB4(
            include_top=False, weights="imagenet", input_shape=(300, 300, 3)
        )

    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(5, activation="softmax")(x)
    my_model = Model(inputs=base.input, outputs=out)
    print(
        "Built fallback EfficientNetB4(include_top=False, ImageNet weights) + 5-class head model."
    )



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/1908412705.py in <cell line: 0>()
     45 
     46 if my_model is None:
---> 47     from tensorflow.keras import layers, Model
     48 
     49     if USING_EFFICIENTNET_TFKERAS:

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
     30 from numpy import typing as npt
     31 
---> 32 from google.protobuf import message
     33 from tensorflow.core.framework import attr_value_pb2
     34 from tensorflow.core.framework import full_type_pb2

ModuleNotFoundError: No module named 'google.protobuf'

## === cell 5
test_globs = [
    "../input/cassava-leaf-disease-classification/test_images/*.jpg",
    "/kaggle/input/cassava-leaf-disease-classification/test_images/*.jpg",
    "/kaggle/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification/test_images/*.jpg",
]
test_images = []
for g in test_globs:
    test_images = glob.glob(g)
    if len(test_images) > 0:
        print("Using test glob:", g, "count:", len(test_images))
        break

if len(test_images) == 0:
    raise FileNotFoundError(
        "Could not find any test images under expected /kaggle/input paths."
    )

df_test = pd.DataFrame(test_images, columns=["path"])


def make_test_gen(batch_size=64):
    if USING_EFFICIENTNET_TFKERAS:
        from efficientnet.tfkeras import preprocess_input
    else:
        from tensorflow.keras.applications.efficientnet import preprocess_input

    my_test_idg = ImageDataGenerator(preprocessing_function=preprocess_input)
    test_gen = my_test_idg.flow_from_dataframe(
        dataframe=df_test,
        x_col="path",
        y_col=None,
        batch_size=batch_size,
        seed=SEED,
        shuffle=False,
        class_mode=None,
        target_size=(300, 300),
    )
    return test_gen




## === cell 6
test_gen = make_test_gen(batch_size=128)
steps = int(np.ceil(test_gen.n / test_gen.batch_size))

pred_test = my_model.predict(test_gen, steps=steps, verbose=1)
pred_test_labels = np.argmax(pred_test, axis=-1).astype(int)

final_submission = df_test.copy()
final_submission["image_id"] = final_submission["path"].str.split("/").str[-1]
final_submission["label"] = pred_test_labels

final_csv = final_submission[["image_id", "label"]]

sample_path_candidates = [
    "../input/cassava-leaf-disease-classification/sample_submission.csv",
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv",
    "/kaggle/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification/sample_submission.csv",
]
sample_path = None
for p in sample_path_candidates:
    if os.path.exists(p):
        sample_path = p
        break

if sample_path is not None:
    sample_sub = pd.read_csv(sample_path)
    final_csv = sample_sub[["image_id"]].merge(final_csv, on="image_id", how="left")
    if final_csv["label"].isna().any():
        fill_label = int(pd.Series(pred_test_labels).mode().iloc[0])
        final_csv["label"] = final_csv["label"].fillna(fill_label).astype(int)

final_csv["label"] = final_csv["label"].astype(int)
final_csv.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", final_csv.shape)
print(final_csv.head())



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/2616038696.py in <cell line: 0>()
----> 1 test_gen = make_test_gen(batch_size=128)
      2 steps = int(np.ceil(test_gen.n / test_gen.batch_size))
      3 
      4 pred_test = my_model.predict(test_gen, steps=steps, verbose=1)
      5 pred_test_labels = np.argmax(pred_test, axis=-1).astype(int)

/tmp/ipykernel_11/4145046182.py in make_test_gen(batch_size)
     24         from efficientnet.tfkeras import preprocess_input
     25     else:
---> 26         from tensorflow.keras.applications.efficientnet import preprocess_input
     27 
     28     my_test_idg = ImageDataGenerator(preprocessing_function=preprocess_input)

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
     30 from numpy import typing as npt
     31 
---> 32 from google.protobuf import message
     33 from tensorflow.core.framework import attr_value_pb2
     34 from tensorflow.core.framework import full_type_pb2

ModuleNotFoundError: No module named 'google.protobuf'

## === cell 7
assert os.path.exists("submission.csv"), "submission.csv was not created."
check = pd.read_csv("submission.csv")
assert list(check.columns) == [
    "image_id",
    "label",
], f"Bad columns: {check.columns.tolist()}"
assert len(check) == len(
    df_test
), f"Row count mismatch: submission={len(check)} test={len(df_test)}"
assert check["label"].between(0, 4).all(), "Labels out of range 0-4."
check.head()

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
AssertionError                            Traceback (most recent call last)
/tmp/ipykernel_11/3210326103.py in <cell line: 0>()
----> 1 assert os.path.exists("submission.csv"), "submission.csv was not created."
      2 check = pd.read_csv("submission.csv")
      3 assert list(check.columns) == [
      4     "image_id",
      5     "label",

AssertionError: submission.csv was not created.
