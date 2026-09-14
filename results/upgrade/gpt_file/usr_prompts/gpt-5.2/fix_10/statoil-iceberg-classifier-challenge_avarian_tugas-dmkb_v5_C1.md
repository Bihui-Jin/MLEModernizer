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
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
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

9.29066

# 6. Current score

0.28703

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 0.35002) has done: 'I fix the Keras import/runtime issues caused by the installed Keras 3 stack by switching to `tf_keras` (which is available in your environment) and updating a few deprecated/renamed APIs (BatchNormalization import path, Adam `lr` arg, ReduceLROnPlateau `epsilon`). I also fix the notebook cell numbering (start at cell 1) while preserving the same model architecture, training loop, and preprocessing so the core logic stays unchanged. Finally, I ensure the pipeline trains, loads the best weights, runs inference, and writes a valid `submission.csv` with the required `id,is_iceberg` columns.'
- What this solution (achieved 1.0172) has done: 'I fix the runtime crash happening at the `tf_keras` import by switching the code to use `tensorflow.keras` (TF-backed Keras) which avoids the protobuf `MessageFactory.GetPrototype` issue in this environment. I keep the model architecture, preprocessing, training loop, and submission formatting identical so the evaluation semantics don’t change. I also keep all paths and output `submission.csv` unchanged, ensuring the script runs end-to-end and produces a valid CSV with `id,is_iceberg`. This should restore a normal score (your current 0.35002 is far better than the target 9.29066, so we avoid any score-changing “improvements” and focus on correctness/stability).'
- What this solution (achieved 0.31006) has done: 'I fix the TensorFlow/Keras import crash by forcing a safe protobuf Python implementation before importing TensorFlow, which avoids the `MessageFactory.GetPrototype` error in this environment. Then I fix the `ModelCheckpoint` filepath extension requirement in Keras 3 by switching the checkpoint to a `.weights.h5` filename and setting `save_weights_only=True`, and ensure the same path is used when reloading weights. These are runtime/IO correctness fixes that preserve your model architecture, preprocessing, training loop, and submission formatting, so they should be essentially score-neutral (and your current score is already far better than the very loose target). Finally, I renumber cells starting from 1 and keep all paths/output (`./submission.csv`) unchanged so it runs end-to-end and produces a valid submission.'
- What this solution (achieved 0.3687) has done: 'We fix the runtime crash in the TensorFlow import (`MessageFactory.GetPrototype`) by forcing the pure-Python protobuf implementation *and* disabling the C++ fast-path via `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION=2` before importing TensorFlow. To keep the core model/training logic identical, we won’t change architecture, preprocessing, optimizer, epochs, or callbacks—only the environment/import ordering and a couple of defensive checks. We also renumber cells starting at 1 (Kaggle-compatible) while preserving the same code flow. Finally, we ensure the submission is written as `./submission.csv` with exactly the required `id,is_iceberg` columns.'
- What this solution (achieved 0.329) has done: 'We fix the TensorFlow import crash (`MessageFactory.GetPrototype`) by forcing the pure-Python protobuf implementation and version **before any protobuf/TensorFlow-dependent imports**, and by importing TensorFlow in a guarded way that avoids eager loading of problematic protobuf symbols. This is a runtime-only change and keeps your model, training loop, preprocessing, and submission formatting identical (so it should be score-neutral). I also renumber cells starting at 1 (Kaggle-friendly) while preserving the original order/content, and add a small path fallback to ensure the JSON files are found reliably. Finally, the script still write `./submission.csv` with the exact required `id,is_iceberg` columns.'
- What this solution (achieved 0.37174) has done: 'I fix the TensorFlow import crash (`MessageFactory` has no `GetPrototype`) by forcing a compatible protobuf runtime *before any TensorFlow/protobuf-related imports* and clearing any preloaded protobuf modules that can lock in the wrong implementation. This is a runtime-only change and keeps your model, preprocessing, training loop, and submission formatting identical, so it should be score-neutral (your current logloss is already far better than the target, and lower is better). I also renumber cells starting at 1 (Kaggle-friendly) while preserving the original order and ensure the checkpoint and submission paths remain unchanged. The script then run end-to-end and write `./submission.csv` with `id,is_iceberg`.'
- What this solution (achieved 0.30868) has done: 'We fix the TensorFlow import crash (`MessageFactory` has no `GetPrototype`) by forcing the pure-Python protobuf implementation *and* preventing TensorFlow from using the C++ protobuf fast-path before any TensorFlow/protobuf imports occur. This is a runtime-only change that keeps your model architecture, training loop, preprocessing, and submission formatting identical, so it should be score-neutral (and your current logloss is already far better than the very loose target). We also renumber the cells to start at 1 (Kaggle-friendly) while preserving order and ensure the checkpoint filename remains compatible with the Keras/TensorFlow stack and is loaded correctly. The script then run end-to-end and write `./submission.csv` with exactly `id,is_iceberg`.'
- What this solution (achieved 0.31294) has done: 'We fix the runtime crash occurring when importing TensorFlow/Keras (`MessageFactory.GetPrototype`) by forcing the pure-Python protobuf implementation *and* proactively preventing any protobuf modules from being imported before TensorFlow, including removing any preloaded `google.protobuf` modules. This is a minimal, runtime-only change that preserves your exact model, preprocessing, training loop, and submission formatting, so it should be score-neutral (your current logloss is already far better than the very loose target). We also keep the checkpoint filename as `.weights.h5` (required) and ensure the code reliably finds `train.json/test.json` from the Kaggle paths you listed. Finally, the script run end-to-end and write `./submission.csv` with the required `id,is_iceberg` columns.'
- What this solution (achieved 0.28703) has done: 'We fix the TensorFlow import crash (`MessageFactory` has no `GetPrototype`) by forcing the pure-Python protobuf implementation *before any TensorFlow/protobuf imports* and clearing any already-imported protobuf modules, then importing TensorFlow in a guarded way. This is a runtime-only change and keeps your model architecture, preprocessing, training loop, and submission formatting identical, so it should be score-neutral (and you’re already far better than the very loose target). We also renumber cells to start at 1 (Kaggle-friendly) while preserving the original order and ensure the checkpoint filename remains compatible (`.weights.h5`) and is reloaded correctly. Finally, the script run end-to-end and write `./submission.csv` with exactly `id,is_iceberg`.'

# 9. Code solution

## === cell 0
import os
import sys

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

for k in list(sys.modules.keys()):
    if k.startswith("google.protobuf"):
        del sys.modules[k]

import numpy as np  # noqa: E402

np.random.seed(666)

import cv2  # noqa: F401, E402
import pandas as pd  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: F401, E402
from subprocess import check_output  # noqa: E402

for p in ["../input", "/kaggle/input", "/kaggle/data/input"]:
    if os.path.exists(p):
        print(f"Listing: {p}")
        try:
            print(check_output(["ls", p]).decode("utf8"))
        except Exception as e:
            print("ls failed:", e)
        break



## === cell 1
DATA_CANDIDATES = [
    "/kaggle/data/input/statoil-iceberg-classifier-challenge",
    "/kaggle/data/input",
    "/kaggle/input/statoil-iceberg-classifier-challenge",
    "/kaggle/input",
    "../input",
    "/kaggle/data/statoil-iceberg-classifier-challenge",
    "/kaggle/data",
]

data_dir = None
for d in DATA_CANDIDATES:
    if os.path.exists(os.path.join(d, "train.json")) and os.path.exists(
        os.path.join(d, "test.json")
    ):
        data_dir = d
        break

if data_dir is None:
    raise FileNotFoundError(
        "Could not find train.json/test.json in expected Kaggle input locations."
    )

train = pd.read_json(os.path.join(data_dir, "train.json"))
test = pd.read_json(os.path.join(data_dir, "test.json"))

print("Loaded train:", train.shape, "test:", test.shape)
print("Data dir:", data_dir)



## === cell 2
X_band_1 = np.array(
    [np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_1"]]
)
X_band_2 = np.array(
    [np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_2"]]
)
X_train = np.concatenate(
    [
        X_band_1[:, :, :, np.newaxis],
        X_band_2[:, :, :, np.newaxis],
        ((X_band_1 + X_band_2) / 2)[:, :, :, np.newaxis],
    ],
    axis=-1,
)
Y_train = train["is_iceberg"].astype(np.float32).values

print("X_train:", X_train.shape, "Y_train:", Y_train.shape)



## === cell 3
from matplotlib import pyplot  # noqa: F401, E402

import tensorflow as tf  # noqa: E402

from tensorflow.keras.preprocessing.image import ImageDataGenerator  # noqa: F401, E402
from tensorflow.keras.models import Sequential  # noqa: E402
from tensorflow.keras.layers import (  # noqa: E402
    Conv2D,
    MaxPooling2D,
    Dense,
    Dropout,
    Input,
    Flatten,
)
from tensorflow.keras.layers import GlobalMaxPooling2D  # noqa: F401, E402
from tensorflow.keras.layers import BatchNormalization  # noqa: F401, E402
from tensorflow.keras.layers import Concatenate  # noqa: F401, E402
from tensorflow.keras.models import Model  # noqa: F401, E402
from tensorflow.keras.optimizers import Adam  # noqa: E402
from tensorflow.keras.callbacks import (  # noqa: E402
    ModelCheckpoint,
    Callback,  # noqa: F401
    EarlyStopping,
    ReduceLROnPlateau,
)

print("TF version:", tf.__version__)




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 4
def get_callbacks(filepath, patience=2):
    es = EarlyStopping(monitor="val_loss", patience=patience, mode="min")
    msave = ModelCheckpoint(
        filepath,
        save_best_only=True,
        save_weights_only=True,
        monitor="val_loss",
        mode="min",
    )
    return [es, msave]




## === cell 5
def getModel():
    model = Sequential()

    model.add(Conv2D(8, kernel_size=(3, 3), activation="relu", input_shape=(75, 75, 3)))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(16, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(32, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Flatten())

    model.add(Dense(512, activation="relu"))
    model.add(Dropout(0.2))

    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.2))

    model.add(Dense(1, activation="sigmoid"))

    optimizer = Adam(learning_rate=0.001)
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    return model


model = getModel()
model.summary()



## === cell 6
batch_size = 32
earlyStopping = EarlyStopping(monitor="val_loss", patience=10, verbose=0, mode="min")

CKPT_PATH = "./mdl_wts.weights.h5"
mcp_save = ModelCheckpoint(
    CKPT_PATH,
    save_best_only=True,
    save_weights_only=True,
    monitor="val_loss",
    mode="min",
)

reduce_lr_loss = ReduceLROnPlateau(
    monitor="val_loss", factor=0.1, patience=7, verbose=1, min_delta=1e-4, mode="min"
)



## === cell 7
model = getModel()
model.fit(
    X_train,
    Y_train,
    batch_size=batch_size,
    epochs=50,
    verbose=1,
    callbacks=[earlyStopping, mcp_save, reduce_lr_loss],
    validation_split=0.25,
)



## === cell 8
if os.path.exists(CKPT_PATH):
    model.load_weights(CKPT_PATH)
else:
    print("WARNING: checkpoint not found at", CKPT_PATH, "- using last-epoch weights.")

score = model.evaluate(X_train, Y_train, verbose=1)
print("Train score:", score[0])
print("Train accuracy:", score[1])



## === cell 9
X_band_test_1 = np.array(
    [np.array(band).astype(np.float32).reshape(75, 75) for band in test["band_1"]]
)
X_band_test_2 = np.array(
    [np.array(band).astype(np.float32).reshape(75, 75) for band in test["band_2"]]
)
X_test = np.concatenate(
    [
        X_band_test_1[:, :, :, np.newaxis],
        X_band_test_2[:, :, :, np.newaxis],
        ((X_band_test_1 + X_band_test_2) / 2)[:, :, :, np.newaxis],
    ],
    axis=-1,
)

pred_test = model.predict(X_test, batch_size=256, verbose=1)



## === cell 10
submission = pd.DataFrame(
    {
        "id": test["id"].astype(str).values,
        "is_iceberg": pred_test.reshape((pred_test.shape[0],)),
    }
)

submission["is_iceberg"] = submission["is_iceberg"].astype(np.float32).clip(0.0, 1.0)

print(submission.head(10))
print(submission.shape)

out_path = "./submission.csv"
submission.to_csv(out_path, index=False)
print("Wrote:", out_path, "size:", os.path.getsize(out_path), "bytes")
print("Submission columns:", list(submission.columns))
