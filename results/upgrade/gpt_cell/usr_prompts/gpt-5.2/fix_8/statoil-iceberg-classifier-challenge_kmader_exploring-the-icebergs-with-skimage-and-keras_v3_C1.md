# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


# 1. Kaggle task description

## Task
Predict whether an image contains a ship or an iceberg.

## Metric
Log loss.

## Submission Format
For each id in the test set, you must predict the probability that the image contains an iceberg (a number between 0 and 1). The file should contain a header and have the following format:

```
id,is_iceberg
809385f7,0.5
7535f0cd,0.4
3aa99a38,0.9
etc.
```

## Dataset
The labels are provided by human experts and geographic knowledge on the target. All the images are 75x75 images with two bands.

The data (`train.json`, `test.json`) is presented in `json` format.

The files consist of a list of images, and for each image, you can find the following fields:

- **id** - the id of the image
- **band_1, band_2** - the [flattened](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.ndarray.flatten.html) image data. Each band has 75x75 pixel values in the list, so the list has 5625 elements. Note that these values are not the normal non-negative integers in image files since they have physical meanings - these are **float** numbers with unit being [dB](https://en.wikipedia.org/wiki/Decibel). Band 1 and Band 2 are signals characterized by radar backscatter produced from different polarizations at a particular incidence angle. The polarizations correspond to HH (transmit/receive horizontally) and HV (transmit horizontally and receive vertically).
- **inc_angle** - the incidence angle of which the image was taken. Note that this field has missing data marked as "na", and those images with "na" incidence angles are all in the training data to prevent leakage.
- **is_iceberg** - the target variable, set to 1 if it is an iceberg, and 0 if it is a ship. This field only exists in `train.json`.

Please note that we have included machine-generated images in the test set to prevent hand labeling. They are excluded in scoring.

sample_submission.csv: The submission file in the correct format:

# 2. Python version

3.6

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
scikit-image==0.25.2
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
tf_keras==2.18.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (99 lines)
            sample_submission.csv (322 lines)
            sample_submission.csv.7z (2.0 kB)
            sample_submission.csv.zip (2.0 kB)
            test.json (1 lines)
            test.json.7z (8.9 MB)
            train.json (1 lines)
            train.json.7z (35.8 MB)
            statoil-iceberg-classifier-challenge/
                description.md (99 lines)
                sample_submission.csv (322 lines)
                ... and 6 other files
                statoil-iceberg-classifier-challenge/
        input/
            description.md (99 lines)
            sample_submission.csv (322 lines)
            sample_submission.csv.7z (2.0 kB)
            sample_submission.csv.zip (2.0 kB)
            test.json (1 lines)
            test.json.7z (8.9 MB)
            train.json (1 lines)
            train.json.7z (35.8 MB)
            statoil-iceberg-classifier-challenge/
                description.md (99 lines)
                sample_submission.csv (322 lines)
                ... and 6 other files
                statoil-iceberg-classifier-challenge/
        working/
            statoil-iceberg-classifier-challenge/
                description.md (99 lines)
                sample_submission.csv (322 lines)
                ... and 6 other files
                statoil-iceberg-classifier-challenge/
```

-> data/sample_submission.csv has 321 rows and 2 columns.
The columns are: id, is_iceberg

-> data/statoil-iceberg-classifier-challenge/sample_submission.csv has 321 rows and 2 columns.
The columns are: id, is_iceberg

-> data/statoil-iceberg-classifier-challenge/test.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "array",
  "items": {
    "type": "object",
    "properties": {
      "id": {
        "type": "string"
      },
      "band_1": {
        "type": "array",
        "items": {
          "type": "number"
        }
      },
      "band_2": {
        "type": "array",
        "items": {
          "type": "number"
        }
      },
      "inc_angle": {
        "type": [
          "number",
          "string"
        ]
      }
    },
    "required": [
      "band_1",
      "band_2",
      "id",
      "inc_angle"
    ]
  }
}

-> data/statoil-iceberg-classifier-challenge/train.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "array",
  "items": {
    "type": "object",
    "properties": {
      "id": {
        "type": "string"
      },
      "band_1": {
        "type": "array",
        "items": {
          "type": "number"
        }
      },
      "band_2": {
        "type": "array",
        "items": {
          "type": "number"
        }
      },
      "inc_angle": {
        "type": [
          "number",
          "string"
        ]
      },
      "is_iceberg": {
        "type": "integer"
      }
    },
    "required": [
      "band_1",
      "band_2",
      "id",
      "inc_angle",
      "is_iceberg"
    ]
  }
}

-> data/test.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "array",
  "items": {
    "type": "object",
    "properties": {
      "id": {
        "type": "string"
      },
      "band_1": {
        "type": "array",
        "items": {
          "type": "number"
        }
      },
      "band_2": {
        "type": "array",
        "items": {
          "type": "number"
        }
      },
      "inc_angle": {
        "type": [
          "number",
          "string"
        ]
      }
    },
    "required": [
      "band_1",
      "band_2",
      "id",
      "inc_angle"
    ]
  }
}

-> data/train.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "array",
  "items": {
    "type": "object",
    "properties": {
      "id": {
        "type": "string"
      },
      "band_1": {
        "type": "array",
        "items": {
          "type": "number"
        }
      },
      "band_2": {
        "type": "array",
        "items": {
          "type": "number"
        }
      },
      "inc_angle": {
        "type": [
          "number",
          "string"
        ]
      },
      "is_iceberg": {
        "type": "integer"
      }
    },
    "required": [
      "band_1",
      "band_2",
      "id",
      "inc_angle",
      "is_iceberg"
    ]
  }
}

-> input/sample_submission.csv has 321 rows and 2 columns.
The columns are: id, is_iceberg

-> (stopped after 10 files for performance)

# 5. Target score

0.31379

# 6. Current score

0.47533

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 16.93927) has done: 'Diagnosis: The crash happens at import time because `skimage.util.montage` no longer exists in scikit-image 0.25.x; `montage2d` has been moved/removed from that location. This is an API-path mismatch, not a missing package (scikit-image is installed). The rest of the notebook likely only needs `montage2d` for optional visualization, so we should restore the expected symbol without changing downstream logic.  

Patch summary: In cell 0, replace the failing import with a small compatibility shim that first tries the old import, then falls back to `skimage.util.montage`, and finally defines a minimal local `montage2d` implementation if neither is available. This keeps the same name (`montage2d`) in the namespace, preserving later-cell expectations.  

Updated cells: Only cell 0 is modified.  

Compatibility notes for cell k+1: No variables used by cell 1 are changed (`os`, `np`, `pd`, `base_path` remain identical). `montage2d` remains defined, so any later visualization code continues to work.  

Assumptions: `montage2d` is only used for plotting/inspection and does not affect training/inference outputs; the fallback implementation is sufficient for 2D grayscale montage visualization if needed.'
- What this solution (achieved 0.45157) has done: 'Diagnosis: The crash happens when importing/using `keras.utils.np_utils.to_categorical` under Keras 3.x in this environment; that import path triggers a protobuf-related incompatibility (`MessageFactory.GetPrototype`). The notebook only needs one-hot encoding of `is_iceberg`, so we can avoid the problematic Keras utility call while preserving identical semantics. The fix is to replace `to_categorical(train_df['is_iceberg'])` with a deterministic NumPy one-hot encoding that yields the same shape and dtype behavior expected by later cells.

Patch summary: In cell 5 only, remove the Keras `to_categorical` import/use and generate `y` via `np.eye(2)[...]` from the integer labels. Keep the same `train_test_split` call, random_state, and test_size so downstream training remains unchanged.

Updated cells: cell 5 below.

Compatibility notes for cell k+1: `y_train`/`y_test` remain 2-column one-hot arrays compatible with the final `Dense(2, activation='softmax')` and `binary_crossentropy` used in cell 6; `X_train`/`X_test` are unchanged.

Assumptions: `train_df['is_iceberg']` contains only 0/1 integers as per the dataset schema; `numpy` is already imported earlier (cell 0).'
- What this solution (achieved 0.48323) has done: 'Diagnosis: The crash happens during `simple_cnn.summary()` in cell 6 when importing/using `keras==3.8.0` in this Python 3.6 environment. Keras 3 depends on newer protobuf internals and triggers `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'` with the available protobuf runtime. The environment also provides `tf_keras==2.18.0`, which is the compatible Keras implementation for this runtime and supports the same `Sequential`/layers/compile/fit API used here. The minimal fix is to import the model and layers from `tf_keras` instead of `keras`, leaving the architecture and training semantics unchanged.

Patch summary: In cell 6 only, replace `from keras...` imports with `from tf_keras...` imports so model construction/summary works without the protobuf incompatibility. No other logic is changed.

Updated cells: (cell 6 only)

Compatibility notes for cell k+1: `simple_cnn` remains a Keras-like model object with the same `.fit(...)` signature, so cell 7 continues to run unchanged.

Assumptions: `tf_keras==2.18.0` is correctly installed and usable in the target runtime (it is listed in the environment packages).'
- What this solution (achieved 0.53748) has done: 'Diagnosis: The crash happens while importing/initializing `tf_keras` in cell 6, raising `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'`. This is a known incompatibility between TensorFlow/Keras protobuf bindings and newer `protobuf` versions, and it can be avoided by forcing protobuf to use the pure-Python implementation. Since cell 6 is the first point where `tf_keras` is imported, setting the environment variable before those imports resolves the issue deterministically.

Patch summary: In cell 6, set `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` (and version=2 for stability) *before* importing anything from `tf_keras`. Keep the model definition/architecture and compilation exactly the same.

Updated cells: Only cell 6 is changed.

Compatibility notes for cell k+1: `simple_cnn` remains a compiled `tf_keras` `Sequential` model with the same layers and interface, so `simple_cnn.fit(...)` in cell 7 works unchanged.

Assumptions: The environment allows setting `os.environ` at runtime before importing `tf_keras`, and the installed `tf_keras==2.18.0` works with the pure-Python protobuf backend.'
- What this solution (achieved 0.47998) has done: 'The crash happens during `tf_keras` import/model construction because the protobuf runtime in this environment is too new for the installed TensorFlow/Keras stack, leading to `MessageFactory` missing `GetPrototype`. The existing environment-variable workaround is applied too late (after protobuf may already be imported indirectly), so it doesn’t take effect. The minimal fix is to ensure the pure-Python protobuf implementation is selected *before* any protobuf-dependent imports by proactively setting the env vars and reloading `google.protobuf` if it was already loaded. This keeps the model definition/training logic unchanged and preserves the `simple_cnn` object for cell 7.'
- What this solution (achieved 0.41521) has done: 'Diagnosis: The crash happens during `tf_keras` import/initialization because the installed `protobuf` runtime in this environment is incompatible with the TensorFlow/Keras stack being used, leading to `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'`. The attempted workaround (forcing `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`) is not sufficient here because the incompatible `protobuf` module is already installed/used and the API mismatch remains. The minimal deterministic fix is to monkey-patch `google.protobuf.message_factory.MessageFactory` to provide `GetPrototype` as an alias for `GetMessageClass` (or a safe fallback), restoring the attribute expected by the TF/Keras stack without changing model logic.

Patch summary: In cell 6 only, import `google.protobuf.message_factory` early and add a small compatibility shim that defines `MessageFactory.GetPrototype` when missing. Keep the rest of the model code unchanged so architecture/training semantics are identical.

Updated cells: Only cell 6 is modified below.

Compatibility notes for cell k+1: `simple_cnn` is still created/compiled with the same name and API, so `simple_cnn.fit(...)` in cell 7 work unchanged.

Assumptions: `google.protobuf` is installed (it is, since the error originates from it) and `MessageFactory` exists; if `GetMessageClass` is not available, the fallback raise a clear error at call time rather than silently altering behavior.'
- What this solution (achieved 0.47533) has done: 'Your current score (0.41521 log loss) is worse than the target (0.31379), so we should make a small, low-risk improvement without changing the model/loop. The biggest issue is a train/validation split bug: you’re using `test_size=0.5`, which throws away half the data for training and typically hurts log loss; changing it to 0.2 keeps the same training approach but improves generalization toward the target. To keep results stable and avoid semantic drift, I also set deterministic seeds (NumPy + TF) and ensure `inc_angle` “na” doesn’t inject dtype issues by parsing it to numeric (not used by the model, but it prevents subtle downstream inconsistencies). The submission remains identical in format and filename (`predictions.csv`) and uses the same probability column (`test_predictions[:,1]`).'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

try:
    from skimage.util.montage import montage2d  # older scikit-image
except Exception:
    try:
        from skimage.util import montage as _montage_mod  # newer scikit-image

        montage2d = _montage_mod  # skimage.util.montage returns a montage image
    except Exception:

        def montage2d(arr, fill=0):
            arr = np.asarray(arr)
            if arr.ndim != 3:
                raise ValueError(
                    "montage2d fallback expects an array of shape (n_images, height, width)"
                )
            n, h, w = arr.shape
            n_cols = int(np.ceil(np.sqrt(n)))
            n_rows = int(np.ceil(n / n_cols))
            out = np.full((n_rows * h, n_cols * w), fill, dtype=arr.dtype)
            for i in range(n):
                r, c = divmod(i, n_cols)
                out[r * h : (r + 1) * h, c * w : (c + 1) * w] = arr[i]
            return out


import matplotlib.pyplot as plt

base_path = os.path.join("..", "input")




## === cell 1
def load_and_format(in_path):
    out_df = pd.read_json(in_path)

    if "inc_angle" in out_df.columns:
        out_df["inc_angle"] = pd.to_numeric(out_df["inc_angle"], errors="coerce")

    out_images = out_df.apply(
        lambda c_row: [
            np.stack([c_row["band_1"], c_row["band_2"]], -1).reshape((75, 75, 2))
        ],
        1,
    )
    out_images = np.stack(out_images).squeeze()
    return out_df, out_images


train_df, train_images = load_and_format(os.path.join(base_path, "train.json"))
print("training", train_df.shape, "loaded", train_images.shape)
test_df, test_images = load_and_format(os.path.join(base_path, "test.json"))
print("testing", test_df.shape, "loaded", test_images.shape)
train_df.sample(3)



## === cell 2
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.matshow(train_images[0, :, :, 0])
ax1.set_title("Band 1")
ax2.matshow(train_images[0, :, :, 1])
ax2.set_title("Band 2")



## === cell 3
fig, (ax1s, ax2s) = plt.subplots(2, 2, figsize=(8, 8))
obj_list = dict(
    ships=train_df.query("is_iceberg==0").sample(16).index,
    icebergs=train_df.query("is_iceberg==1").sample(16).index,
)
for ax1, ax2, (obj_type, idx_list) in zip(ax1s, ax2s, obj_list.items()):
    ax1.imshow(montage2d(train_images[idx_list, :, :, 0]))
    ax1.set_title("%s Band 1" % obj_type)
    ax1.axis("off")
    ax2.imshow(montage2d(train_images[idx_list, :, :, 1]))
    ax2.set_title("%s Band 2" % obj_type)
    ax2.axis("off")



## === cell 4
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 12))
idx_list = test_df.sample(49).index
obj_type = "Test Data"
ax1.imshow(montage2d(test_images[idx_list, :, :, 0]))
ax1.set_title("%s Band 1" % obj_type)
ax1.axis("off")
ax2.imshow(montage2d(test_images[idx_list, :, :, 1]))
ax2.set_title("%s Band 2" % obj_type)
ax2.axis("off")



## === cell 5
from sklearn.model_selection import train_test_split

y = np.eye(2, dtype=np.float32)[train_df["is_iceberg"].astype(int).to_numpy()]

X_train, X_test, y_train, y_test = train_test_split(
    train_images,
    y,
    random_state=2017,
    test_size=0.2,
    stratify=train_df["is_iceberg"].astype(int).to_numpy(),
)
print("Train", X_train.shape, y_train.shape)
print("Validation", X_test.shape, y_test.shape)



## === cell 6
import os
import sys
import importlib

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"

os.environ["PYTHONHASHSEED"] = "2017"
np.random.seed(2017)

import google.protobuf  # noqa: F401
from google.protobuf import message_factory as _message_factory

if not hasattr(_message_factory.MessageFactory, "GetPrototype"):
    if hasattr(_message_factory.MessageFactory, "GetMessageClass"):

        def _GetPrototype(self, descriptor):
            return self.GetMessageClass(descriptor)

        _message_factory.MessageFactory.GetPrototype = _GetPrototype
    else:

        def _GetPrototype(self, descriptor):
            raise AttributeError(
                "protobuf MessageFactory has neither GetPrototype nor GetMessageClass; "
                "incompatible protobuf version for this environment."
            )

        _message_factory.MessageFactory.GetPrototype = _GetPrototype

if "google.protobuf" in sys.modules:
    importlib.reload(sys.modules["google.protobuf"])

import tensorflow as tf

tf.random.set_seed(2017)

from tf_keras.models import Sequential
from tf_keras.layers import (
    Conv2D,
    BatchNormalization,
    Dropout,
    MaxPooling2D,
    GlobalMaxPooling2D,
    Dense,
)

simple_cnn = Sequential()
simple_cnn.add(BatchNormalization(input_shape=(75, 75, 2)))
for i in range(4):
    simple_cnn.add(Conv2D(8 * 2**i, kernel_size=(3, 3)))
    simple_cnn.add(MaxPooling2D((2, 2)))
simple_cnn.add(GlobalMaxPooling2D())
simple_cnn.add(Dropout(0.5))
simple_cnn.add(Dense(8))
simple_cnn.add(Dense(2, activation="softmax"))
simple_cnn.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
simple_cnn.summary()



## === cell 7
simple_cnn.fit(
    X_train, y_train, validation_data=(X_test, y_test), epochs=10, shuffle=True
)



## === cell 8
test_predictions = simple_cnn.predict(test_images)

pred_df = test_df[["id"]].copy()
pred_df["is_iceberg"] = test_predictions[:, 1]
pred_df.to_csv("predictions.csv", index=False)
pred_df.sample(3)
