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

0.71765

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 0.40445) has done: 'I remove the failing `skimage.util.montage` import (it is not available in this environment) and replace montage visualizations with a small local montage helper so the exploratory cells run without extra dependencies. I also fix the input path to point at the provided Kaggle dataset location and make the script resilient by not depending on earlier plotting cells having run. Finally, I fix the Keras utility import (`to_categorical`) to avoid the protobuf-related error and ensure the pipeline trains, predicts, and writes a valid `submission.csv` with the required `id,is_iceberg` columns.'
- What this solution (achieved 0.45563) has done: 'I fix the protobuf-related crash by avoiding `tf_keras.utils.to_categorical` and instead building the 2-column one-hot targets with NumPy (same semantics, no dependency on the failing protobuf path). I also add lightweight determinism (seeds + TF deterministic ops where available) to stabilize training and slightly improve logloss without changing the model or training loop. Finally, I ensure the loss is numerically aligned with the softmax+one-hot setup by using `categorical_crossentropy` (same objective intent, but correct pairing), and keep the submission format/columns exactly as required.'
- What this solution (achieved 0.40606) has done: 'I fix the TensorFlow/Keras protobuf crash by removing the dependency on `tensorflow`/`tf_keras` (which is triggering `MessageFactory.GetPrototype`) and instead using the already-installed standalone `keras==3.8.0` API to build and train the exact same CNN architecture. I keep the model layers, loss, optimizer, epochs, and data split semantics identical so the core logic and evaluation intent remain unchanged, while ensuring the notebook runs end-to-end. I also make the import paths consistent for Keras 3 and keep the submission writing exactly in `id,is_iceberg` format to produce a valid `submission.csv`.'
- What this solution (achieved 0.39703) has done: 'I fix the crash in the Keras import/compile step by switching the backend to `jax` (to avoid the TensorFlow/protobuf `MessageFactory.GetPrototype` issue in this environment) while keeping the same model, loss, optimizer, epochs, and split logic. I also make the backend choice deterministic by setting `KERAS_BACKEND` before importing `keras`, and add a small fallback to `numpy` backend if `jax` isn’t available. These changes are execution-focused and should also improve logloss versus a non-running/unstable setup by ensuring the CNN actually trains and predicts consistently. The submission writing remains unchanged and still produce a valid `submission.csv` with `id,is_iceberg`.'
- What this solution (achieved 17.96568) has done: 'I fix the protobuf-related crash by preventing Keras from initializing any TensorFlow/protobuf code paths (which is what triggers the `MessageFactory.GetPrototype` error) and force a safe backend before `keras` is imported. I do this by setting `KERAS_BACKEND` explicitly and also sanitizing the environment to avoid TensorFlow being imported indirectly. This is execution-focused and score-neutral in intent (same model, loss, optimizer, epochs, and split); it simply makes training run reliably end-to-end again so you can generate a valid `submission.csv`. No changes are made to the CNN architecture, training loop, or submission formatting.'
- What this solution (achieved 0.40728) has done: 'I fix the backend/runtime issue by switching Keras to a trainable backend (TensorFlow if available; otherwise JAX), because the NumPy backend cannot run `fit()` and is causing the current failure/degenerate training that leads to very poor logloss. I also ensure the backend environment variables are set before importing `keras`, so the selection actually takes effect and avoids the protobuf crash path as much as possible. The model architecture, loss, optimizer, epochs, and train/validation split stay identical; changes are limited to making training execute correctly and produce calibrated probabilities. Finally, I keep the submission writing intact (`id,is_iceberg`) and ensure it always outputs a valid `submission.csv`.'
- What this solution (achieved 0.37782) has done: 'I fix the `MessageFactory.GetPrototype` crash by preventing Keras from selecting the TensorFlow backend (which is triggering the protobuf incompatibility in this environment) and forcing a trainable non-TF backend before importing `keras`. To keep the core model/training logic identical, I not change the architecture, loss, optimizer, epochs, or the train/validation split; the only functional change is backend selection so `fit()` actually runs. I also add a small, safe fallback chain (JAX → Torch) to ensure the script runs end-to-end in Kaggle even if one backend isn’t available. Finally, I keep the submission writing exactly as required (`id,is_iceberg`) to guarantee a valid `submission.csv`.'
- What this solution (achieved 0.38418) has done: 'I fix the crash in the Keras initialization by forcing a non-TensorFlow backend *before any Keras import happens* (the current protobuf `MessageFactory.GetPrototype` error is triggered when TF/protobuf paths are touched). To keep changes minimal and score-improving toward your target, I keep the exact same CNN architecture/training loop but switch the loss to `sparse_categorical_crossentropy` and train on integer labels (same softmax objective, fewer opportunities for one-hot shape/backend issues), which is typically slightly better calibrated for logloss here. I also ensure the backend choice is deterministic and prefers JAX/Torch (both avoid the TF/protobuf issue in this environment) while still writing a valid `submission.csv` with `id,is_iceberg`.'
- What this solution (achieved 0.3873) has done: 'I fix the protobuf-related `MessageFactory.GetPrototype` crash by preventing Keras from ever touching the TensorFlow backend and explicitly selecting a non-TF backend (JAX preferred, Torch fallback) before importing `keras`. This is a minimal execution fix that keeps your CNN architecture, loss (`sparse_categorical_crossentropy`), optimizer, epochs, and train/validation split unchanged. I also add a small safety check to fail fast if only the TensorFlow backend is available (since that’s the source of the crash here) rather than dying mid-import. The rest of the pipeline (training, predicting, and writing `submission.csv` with `id,is_iceberg`) remains the same.'
- What this solution (achieved 0.4007) has done: 'I fix the protobuf `MessageFactory.GetPrototype` crash by ensuring Keras never imports/initializes TensorFlow/protobuf code paths: we force a non-TF backend (JAX preferred, Torch fallback) *and* proactively block TensorFlow imports before importing `keras`. I also make backend selection more robust by setting key environment flags early and verifying the chosen backend after import, so the model build/compile doesn’t die mid-way. These changes are execution/stability focused and keep your CNN architecture, loss, optimizer, epochs, and train/validation split semantics the same. Once training runs, the existing prediction and `submission.csv` writing logic produce a valid Kaggle submission with `id,is_iceberg`.'
- What this solution (achieved 0.38516) has done: 'I keep your CNN, training loop, epochs, and data split semantics the same, but make two small changes that typically improve logloss without changing the core approach: (1) standardize each image channel using training-set mean/std and apply the same transform to validation/test (stabilizes optimization and probability calibration), and (2) use `padding="same"` in the Conv2D layers so spatial information isn’t prematurely shrunk (a minor architectural tweak that often reduces underfitting on 75×75 inputs). I also add a minimal incidence-angle feature as a third input channel (replicated across the 75×75 grid) so the model can use this known helpful signal without changing the training loop structure. The submission writing stays identical (`id,is_iceberg` to `submission.csv`). These changes are aimed to reduce your logloss from ~0.4007 toward 0.31379.'
- What this solution (achieved 0.71765) has done: 'We’re currently worse than the target (0.38516 vs 0.31379, lower is better), so the safest way to move toward the target with minimal risk is to improve probability calibration without changing the CNN architecture or training loop. I keep the exact same model/loss/epochs/split, but add a tiny amount of label smoothing (implemented via a custom loss wrapper) which often reduces logloss by preventing overconfident wrong predictions. I also ensure the final probabilities are clipped away from exact 0/1 (logloss-safe) and keep all paths/columns identical so the submission remains valid. These changes are small, metric-aligned, and should improve logloss toward the target without altering the core approach.'

# 9. Code solution

## === cell 0
import os
import sys
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SEED = 2017
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

CANDIDATE_BASE_PATHS = [
    "/kaggle/input/statoil-iceberg-classifier-challenge",
    "/kaggle/input",
    os.path.join("..", "input"),
    os.path.join("..", "data"),
]
base_path = None
for p in CANDIDATE_BASE_PATHS:
    if os.path.exists(os.path.join(p, "train.json")) and os.path.exists(
        os.path.join(p, "test.json")
    ):
        base_path = p
        break
if base_path is None:
    p = "/kaggle/input/statoil-iceberg-classifier-challenge/statoil-iceberg-classifier-challenge"
    if os.path.exists(os.path.join(p, "train.json")):
        base_path = p
    else:
        raise FileNotFoundError(
            "Could not locate train.json/test.json under expected Kaggle input paths."
        )

print("Using base_path:", base_path)


def montage2d_local(img_stack, grid_shape=None, fill_value=0.0):
    """
    img_stack: (N, H, W) array
    returns: (grid_H, grid_W) montage image
    """
    img_stack = np.asarray(img_stack)
    if img_stack.ndim != 3:
        raise ValueError("montage2d_local expects a 3D array (N, H, W).")
    n, h, w = img_stack.shape
    if grid_shape is None:
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))
    else:
        rows, cols = grid_shape
    out = np.full((rows * h, cols * w), fill_value, dtype=img_stack.dtype)
    for i in range(n):
        r, c = divmod(i, cols)
        if r >= rows:
            break
        out[r * h : (r + 1) * h, c * w : (c + 1) * w] = img_stack[i]
    return out




## === cell 1
def load_and_format(in_path):
    out_df = pd.read_json(in_path)

    if "inc_angle" in out_df.columns:
        out_df["inc_angle"] = pd.to_numeric(out_df["inc_angle"], errors="coerce")

    out_images = out_df.apply(
        lambda c_row: [
            np.stack([c_row["band_1"], c_row["band_2"]], -1).reshape((75, 75, 2))
        ],
        axis=1,
    )
    out_images = np.stack(out_images).squeeze().astype(np.float32)
    return out_df, out_images


train_df, train_images = load_and_format(os.path.join(base_path, "train.json"))
print("training", train_df.shape, "loaded", train_images.shape)
test_df, test_images = load_and_format(os.path.join(base_path, "test.json"))
print("testing", test_df.shape, "loaded", test_images.shape)
train_df.sample(3, random_state=SEED)



## === cell 2
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.matshow(train_images[0, :, :, 0])
ax1.set_title("Band 1")
ax2.matshow(train_images[0, :, :, 1])
ax2.set_title("Band 2")
plt.tight_layout()
plt.show()



## === cell 3
fig, (ax1s, ax2s) = plt.subplots(2, 2, figsize=(8, 8))
obj_list = dict(
    ships=train_df.query("is_iceberg==0").sample(16, random_state=SEED).index,
    icebergs=train_df.query("is_iceberg==1").sample(16, random_state=SEED).index,
)
for ax1, ax2, (obj_type, idx_list) in zip(ax1s, ax2s, obj_list.items()):
    ax1.imshow(montage2d_local(train_images[idx_list, :, :, 0]))
    ax1.set_title("%s Band 1" % obj_type)
    ax1.axis("off")
    ax2.imshow(montage2d_local(train_images[idx_list, :, :, 1]))
    ax2.set_title("%s Band 2" % obj_type)
    ax2.axis("off")
plt.tight_layout()
plt.show()



## === cell 4
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 12))
idx_list = test_df.sample(49, random_state=SEED).index
obj_type = "Test Data"
ax1.imshow(montage2d_local(test_images[idx_list, :, :, 0]))
ax1.set_title("%s Band 1" % obj_type)
ax1.axis("off")
ax2.imshow(montage2d_local(test_images[idx_list, :, :, 1]))
ax2.set_title("%s Band 2" % obj_type)
ax2.axis("off")
plt.tight_layout()
plt.show()



## === cell 5
from sklearn.model_selection import train_test_split

y = train_df["is_iceberg"].values.astype(np.int64)

X_train, X_test, y_train, y_test, df_train, df_valid = train_test_split(
    train_images,
    y,
    train_df[["inc_angle"]].copy(),
    random_state=SEED,
    test_size=0.5,
    stratify=y,
)

print("Train", X_train.shape, y_train.shape, df_train.shape)
print("Validation", X_test.shape, y_test.shape, df_valid.shape)




## === cell 6
def make_inc_angle_channel(inc_series, fill_value):
    inc = pd.to_numeric(inc_series, errors="coerce").astype(np.float32).to_numpy()
    inc = np.where(np.isfinite(inc), inc, fill_value).astype(np.float32)
    inc = inc.reshape((-1, 1, 1, 1))
    return np.tile(inc, (1, 75, 75, 1)).astype(np.float32)


train_inc_fill = float(
    np.nanmedian(pd.to_numeric(df_train["inc_angle"], errors="coerce"))
)
if not np.isfinite(train_inc_fill):
    train_inc_fill = 0.0

X_train = np.concatenate(
    [X_train, make_inc_angle_channel(df_train["inc_angle"], train_inc_fill)], axis=-1
)
X_test = np.concatenate(
    [X_test, make_inc_angle_channel(df_valid["inc_angle"], train_inc_fill)], axis=-1
)
test_images = np.concatenate(
    [test_images, make_inc_angle_channel(test_df["inc_angle"], train_inc_fill)], axis=-1
)

print("After inc_angle channel:")
print("Train", X_train.shape, "Valid", X_test.shape, "Test", test_images.shape)

ch_mean = X_train.mean(axis=(0, 1, 2), keepdims=True).astype(np.float32)
ch_std = X_train.std(axis=(0, 1, 2), keepdims=True).astype(np.float32)
ch_std = np.maximum(ch_std, 1e-6)

X_train = (X_train - ch_mean) / ch_std
X_test = (X_test - ch_mean) / ch_std
test_images = (test_images - ch_mean) / ch_std



## === cell 7
import importlib.util

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

available_backends = []
if importlib.util.find_spec("jax") is not None:
    available_backends.append("jax")
if importlib.util.find_spec("torch") is not None:
    available_backends.append("torch")

os.environ["KERAS_BACKEND"] = (
    "jax"
    if "jax" in available_backends
    else "torch" if "torch" in available_backends else ""
)

if not os.environ["KERAS_BACKEND"]:
    raise RuntimeError(
        "No non-TensorFlow Keras backend found (jax/torch). "
        "TensorFlow backend is avoided due to protobuf MessageFactory.GetPrototype crash."
    )


class _BlockedModule:
    def __getattr__(self, name):
        raise ImportError(
            "TensorFlow is blocked to avoid protobuf crash in this environment."
        )


sys.modules.setdefault("tensorflow", _BlockedModule())
sys.modules.setdefault("tf_keras", _BlockedModule())

import keras
from keras.models import Sequential
from keras.layers import (
    Conv2D,
    BatchNormalization,
    Dropout,
    MaxPooling2D,
    GlobalMaxPooling2D,
    Dense,
)

try:
    keras.utils.set_random_seed(SEED)
except Exception:
    pass


def sparse_cce_with_label_smoothing(epsilon=0.05):
    base = keras.losses.SparseCategoricalCrossentropy(from_logits=False)

    def _loss(y_true, y_pred):
        y_true = keras.ops.cast(y_true, "int32")
        y_onehot = keras.ops.one_hot(y_true, num_classes=2)
        y_onehot = y_onehot * (1.0 - epsilon) + (epsilon / 2.0)
        return base(y_onehot, y_pred)

    return _loss


simple_cnn = Sequential()
simple_cnn.add(BatchNormalization(input_shape=(75, 75, 3)))
for i in range(4):
    simple_cnn.add(Conv2D(8 * 2**i, kernel_size=(3, 3), padding="same"))
    simple_cnn.add(MaxPooling2D((2, 2)))
simple_cnn.add(GlobalMaxPooling2D())
simple_cnn.add(Dropout(0.5))
simple_cnn.add(Dense(8))
simple_cnn.add(Dense(2, activation="softmax"))

simple_cnn.compile(
    optimizer="adam",
    loss=sparse_cce_with_label_smoothing(epsilon=0.05),
    metrics=["accuracy"],
)
simple_cnn.summary()
print("Keras backend:", keras.backend.backend())



## === cell 8
history = simple_cnn.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=10,
    shuffle=True,
    verbose=2,
)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2832695594.py in <cell line: 0>()
----> 1 history = simple_cnn.fit(
      2     X_train,
      3     y_train,
      4     validation_data=(X_test, y_test),
      5     epochs=10,

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

    [... skipping hidden 14 frame]

/tmp/ipykernel_11/2099498401.py in _loss(y_true, y_pred)
     59         y_onehot = keras.ops.one_hot(y_true, num_classes=2)
     60         y_onehot = y_onehot * (1.0 - epsilon) + (epsilon / 2.0)
---> 61         return base(y_onehot, y_pred)
     62 
     63     return _loss

ValueError: Arguments `target` and `output` must have the same shape up until the last dimension: target.shape=(32, 2), output.shape=(32, 2)

## === cell 9
test_predictions = simple_cnn.predict(test_images, verbose=0)

iceberg_prob = np.clip(test_predictions[:, 1].astype(np.float64), 1e-6, 1.0 - 1e-6)

pred_df = test_df[["id"]].copy()
pred_df["is_iceberg"] = iceberg_prob

submission_path = "submission.csv"
pred_df.to_csv(submission_path, index=False)
print("Wrote:", submission_path, "shape:", pred_df.shape)
print(pred_df.head(3))
