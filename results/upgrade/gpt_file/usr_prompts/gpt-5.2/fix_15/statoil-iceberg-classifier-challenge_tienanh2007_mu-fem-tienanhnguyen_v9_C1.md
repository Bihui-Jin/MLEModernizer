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

16.34687

# 6. Current score

0.3739

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 0.87779) has done: 'I fix the import/runtime crash by switching from `tf_keras` (which is triggering a protobuf `MessageFactory.GetPrototype` issue in this environment) to `tensorflow.keras`, while keeping the same model, training loop, and data pipeline. I also fix the optimizer construction error by removing the deprecated `decay` argument (no functional replacement needed here) so `Adam` initializes correctly. Finally, I make sure weights are loaded only after a successful training run and that `submission.csv` is always written with the exact required columns and row alignment.'
- What this solution (achieved 0.41154) has done: 'I fix the TensorFlow/Keras import crash by avoiding `tensorflow` in this environment and switching to `tf_keras` (the installed TF-Keras package) while keeping the same Sequential CNN, compile settings, and fit call. I also fix the `ModelCheckpoint` filepath extension issue by changing it to a valid `.keras` filename (Keras 3 requirement), which also restore the `callbacks` variable so training runs. Finally, I keep the same submission generation logic but ensure the weights are loaded from the new path and that the CSV is always written with the required `id,is_iceberg` columns.'
- What this solution (achieved 0.30181) has done: 'I fix the import/runtime crash caused by `tf_keras` hitting a protobuf incompatibility (`MessageFactory.GetPrototype`) by switching back to `tensorflow.keras`, which is the most stable Keras backend in Kaggle’s environment for this competition. I keep the exact same CNN architecture, optimizer settings, train/valid split, and training loop so evaluation semantics remain unchanged. I also keep the checkpoint file as `.keras` (required by Keras 3) and ensure we always write a valid `submission.csv` with the required `id,is_iceberg` columns aligned to `test.json`.'
- What this solution (achieved 0.38505) has done: 'I fix the TensorFlow import crash (`MessageFactory.GetPrototype`) by removing the TensorFlow dependency entirely and switching to the installed `tf_keras` package so the same Keras-style Sequential CNN can build/train without protobuf issues. I keep the model architecture, loss, optimizer hyperparameters, train/valid split, and training loop unchanged to preserve evaluation semantics. I also make the checkpoint saving/loading compatible with Keras 3 by saving full model to a `.keras` file and then reloading it (instead of `load_weights`), ensuring inference always uses the best checkpoint. Finally, I keep the submission formatting identical but ensure probabilities are clipped and the CSV is always written as `submission.csv` with `id,is_iceberg`.'
- What this solution (achieved 0.41038) has done: 'I fix the runtime crash caused by importing `tf_keras` (protobuf `MessageFactory.GetPrototype` incompatibility) by switching to `tensorflow.keras`, which is available via the installed `keras==3.8.0` package’s TensorFlow backend. I keep the exact same CNN architecture, optimizer hyperparameters, data pipeline, train/valid split, callbacks, and training loop so evaluation semantics remain unchanged and score impact is only from the model actually running correctly. I also make the checkpointing robust under Keras 3 by saving a full `.keras` model and reloading it for inference, and ensure the submission is always written as `submission.csv` with `id,is_iceberg` aligned to `test.json`. These changes are directly targeted at unblocking training/inference end-to-end and should move the score toward the target simply by producing a valid, properly-trained model rather than crashing.'
- What this solution (achieved 0.28512) has done: 'I fix the runtime crash caused by importing `tensorflow` in this environment (protobuf `MessageFactory.GetPrototype` issue) by switching the code to use the installed `tf_keras` package instead, while keeping the same Sequential CNN, compile settings, and training loop. I also make checkpointing compatible with Keras 3 by saving a full model to a `.keras` file and reloading it for inference exactly as your logic intends. Finally, I ensure the submission file is always written as `submission.csv` with the required `id,is_iceberg` columns aligned to `test.json`. These changes are execution/stability fixes and should keep score in the same ballpark while unblocking end-to-end runs.'
- What this solution (achieved 0.33114) has done: 'I fix the crash in the Keras import stack (`tf_keras` triggering the protobuf `MessageFactory.GetPrototype` error) by switching to `tensorflow.keras`, which is the most reliable way to run this classic CNN code in Kaggle while keeping the exact same model architecture, compile settings, and training loop. I also make the data pipeline robust to `inc_angle == "na"` by adding it as a numeric feature (with safe imputation) without changing the CNN core, which should improve log loss modestly toward your target. Finally, I keep checkpointing as a full `.keras` model (Keras 3 compatible) and ensure `submission.csv` is always written with the required `id,is_iceberg` columns aligned to `test.json`.'
- What this solution (achieved 0.3521) has done: 'I fix the runtime crash coming from importing TensorFlow in this Kaggle image by switching the code to the installed `tf_keras` package (same Keras API/semantics) while keeping the exact same CNN, optimizer settings, and training loop. I also keep checkpointing as a full `.keras` model (Keras 3 requirement) and reload it for inference exactly as your pipeline intends. Finally, I make the imports/version prints robust across backends and ensure the submission is always written as `submission.csv` with the required `id,is_iceberg` columns aligned to `test.json`. These changes are execution/stability fixes and should keep the score in the same ballpark (moving toward the target by making sure the model trains/runs reliably).'
- What this solution (achieved 0.3524) has done: 'I fix the immediate runtime crash caused by `tf_keras` importing protobuf symbols that are incompatible in this environment by switching imports to the stable `tensorflow.keras` API (same Keras semantics for this model). I keep the exact same CNN architecture, optimizer hyperparameters, data pipeline, train/validation split, callbacks, and training loop so the core logic and evaluation semantics remain the same. I also keep checkpointing to a `.keras` file (required by Keras 3+) and ensure we always reload the best saved model if it exists. Finally, I ensure the submission is always written as `submission.csv` with the required `id,is_iceberg` columns aligned to `test.json`.'
- What this solution (achieved 0.29898) has done: 'I fix the import crash by removing the TensorFlow backend usage (which triggers the protobuf `MessageFactory.GetPrototype` issue here) and switching the code to the installed `tf_keras` package while keeping the same model architecture, compile settings, callbacks, and training loop. I also make checkpointing compatible and reliable by saving a full model to a `.keras` file and reloading it for inference exactly as your current pipeline intends. Finally, I keep the exact submission format (`id,is_iceberg`) and ensure predictions are correctly shaped, aligned to `test.json`, clipped to [0, 1], and written to `submission.csv`.'
- What this solution (achieved 0.38218) has done: 'I fix the runtime crash in your import cell by avoiding `tf_keras`, which is triggering the protobuf `MessageFactory.GetPrototype` incompatibility, and instead use `tensorflow.keras` (the TensorFlow-backed Keras API) while keeping your exact model architecture, compile settings, callbacks, and training loop unchanged. I also make the input-path listing robust to both Kaggle-style folder layouts so the notebook doesn’t fail before training starts. Finally, I keep checkpointing to a `.keras` file and ensure we always reload it if created, then write a valid `submission.csv` with the required `id,is_iceberg` columns aligned to `test.json` (score impact should be positive simply because the pipeline runs reliably end-to-end).'
- What this solution (achieved 0.32331) has done: 'I fix the TensorFlow import crash (`MessageFactory.GetPrototype`) by switching the model code to use the installed `tf_keras` package (which provides the Keras 2.x API without importing `tensorflow`), keeping the exact same CNN architecture, optimizer hyperparameters, callbacks, and fit call. I also make checkpointing/loading compatible with this backend by saving the full model to a `.keras` file and reloading it via `tf_keras.models.load_model` for inference. Finally, I ensure the pipeline always writes a valid `submission.csv` with the required `id,is_iceberg` columns aligned to `test.json` and clipped to `[0,1]`. These changes are primarily runtime/stability fixes; any score change should come only from the model successfully training and predicting end-to-end.'
- What this solution (achieved 0.3739) has done: 'I fix the runtime crash in the import cell by removing `tf_keras` (which is triggering the protobuf `MessageFactory.GetPrototype` error) and using the TensorFlow-backed Keras API (`tensorflow.keras`) instead, while keeping the same model architecture and training loop. I also make checkpointing compatible with this backend by saving a full model to a `.keras` file and reloading it only if it exists. Finally, I keep the same submission generation logic but ensure predictions are flattened, clipped to `[0,1]`, and written as `submission.csv` with exactly `id,is_iceberg` aligned to `test.json`.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

from subprocess import check_output

for p in ("../input", "/kaggle/input"):
    if os.path.exists(p):
        try:
            print(f"Listing {p}:\n", check_output(["ls", p]).decode("utf8"))
        except Exception as e:
            print(f"Could not list {p}: {e}")


def _resolve_input_path(fname: str) -> str:
    candidates = [
        os.path.join("../input", fname),
        os.path.join("../input", "statoil-iceberg-classifier-challenge", fname),
        os.path.join("/kaggle/input", fname),
        os.path.join("/kaggle/input", "statoil-iceberg-classifier-challenge", fname),
        os.path.join(
            "/kaggle/input",
            "statoil-iceberg-classifier-challenge",
            "statoil-iceberg-classifier-challenge",
            fname,
        ),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return candidates[0]


train_path = _resolve_input_path("train.json")
test_path = _resolve_input_path("test.json")

train_df = pd.read_json(train_path)
test_df = pd.read_json(test_path)

print("train_df:", train_df.shape, "test_df:", test_df.shape)
print("train_path:", train_path)
print("test_path:", test_path)



## === cell 1
x_band1 = np.array(
    [np.array(band).astype(np.float32).reshape(75, 75) for band in train_df["band_1"]]
)
x_band2 = np.array(
    [np.array(band).astype(np.float32).reshape(75, 75) for band in train_df["band_2"]]
)
X_train = np.concatenate(
    [x_band1[:, :, :, np.newaxis], x_band2[:, :, :, np.newaxis]], axis=-1
)
y_train = np.array(train_df["is_iceberg"]).astype(np.float32)
print("Xtrain:", X_train.shape, "y:", y_train.shape)

x_band1 = np.array(
    [np.array(band).astype(np.float32).reshape(75, 75) for band in test_df["band_1"]]
)
x_band2 = np.array(
    [np.array(band).astype(np.float32).reshape(75, 75) for band in test_df["band_2"]]
)
X_test = np.concatenate(
    [x_band1[:, :, :, np.newaxis], x_band2[:, :, :, np.newaxis]], axis=-1
)
print("Xtest:", X_test.shape)

inc_train = pd.to_numeric(train_df["inc_angle"], errors="coerce").astype(np.float32)
inc_test = pd.to_numeric(test_df["inc_angle"], errors="coerce").astype(np.float32)
inc_fill = float(np.nanmedian(inc_train.values))
inc_train = inc_train.fillna(inc_fill).values.reshape(-1, 1)
inc_test = inc_test.fillna(inc_fill).values.reshape(-1, 1)

print("inc_train:", inc_train.shape, "inc_test:", inc_test.shape, "fill:", inc_fill)



## === cell 2
from matplotlib import pyplot  # kept as originally imported (even if unused)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # kept for parity
from tensorflow.keras.models import Sequential  # kept for parity
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Dense,
    Dropout,
    Flatten,
    Activation,
    Input,
    Concatenate,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from sklearn.model_selection import train_test_split

print("Using backend: tensorflow.keras")
print("tf:", getattr(tf, "__version__", "unknown"))
print("keras:", getattr(keras, "__version__", "unknown"))




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 3
def getModel():
    img_in = Input(shape=(75, 75, 2), name="img")
    x = Conv2D(64, kernel_size=(3, 3), activation="relu")(img_in)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
    x = Dropout(0.2)(x)

    x = Conv2D(64, kernel_size=(3, 3), activation="relu")(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = Dropout(0.2)(x)

    x = Conv2D(128, kernel_size=(3, 3), activation="relu")(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = Dropout(0.2)(x)

    x = Conv2D(64, kernel_size=(3, 3), activation="relu")(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = Dropout(0.2)(x)

    x = Flatten()(x)

    ang_in = Input(shape=(1,), name="inc_angle")
    x = Concatenate()([x, ang_in])

    x = Dense(512)(x)
    x = Activation("relu")(x)
    x = Dropout(0.2)(x)

    x = Dense(256)(x)
    x = Activation("relu")(x)
    x = Dropout(0.2)(x)

    out = Dense(1)(x)
    out = Activation("sigmoid")(out)

    gmodel = keras.Model(inputs=[img_in, ang_in], outputs=out)

    mypotim = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    gmodel.compile(loss="binary_crossentropy", optimizer=mypotim, metrics=["accuracy"])
    gmodel.summary()
    return gmodel


def get_callbacks(filepath, patience=2):
    es = EarlyStopping(
        monitor="val_loss", patience=patience, mode="min", restore_best_weights=False
    )
    msave = ModelCheckpoint(
        filepath,
        save_best_only=True,
        monitor="val_loss",
        mode="min",
        save_weights_only=False,  # save full model to .keras
    )
    return [es, msave]


file_path = "./best_model.keras"
callbacks = get_callbacks(filepath=file_path, patience=5)



## === cell 4
X_train_cv, X_valid, y_train_cv, y_valid, inc_train_cv, inc_valid = train_test_split(
    X_train, y_train, inc_train, random_state=1, train_size=0.75, stratify=y_train
)
print(
    "Train split:",
    X_train_cv.shape,
    y_train_cv.shape,
    inc_train_cv.shape,
    "Valid split:",
    X_valid.shape,
    y_valid.shape,
    inc_valid.shape,
)



## === cell 5
model = getModel()
model.fit(
    [X_train_cv, inc_train_cv],
    y_train_cv,
    batch_size=24,
    epochs=50,
    verbose=1,
    validation_data=([X_valid, inc_valid], y_valid),
    callbacks=callbacks,
)

if os.path.exists(file_path):
    model = keras.models.load_model(file_path)

predicted_test = model.predict([X_test, inc_test], verbose=1)
predicted_test = np.asarray(predicted_test).reshape(-1)

submit_df = pd.DataFrame(
    {"id": test_df["id"].astype(str).values, "is_iceberg": predicted_test}
)
submit_df["is_iceberg"] = submit_df["is_iceberg"].astype(float).clip(0.0, 1.0)

submit_path = "./submission.csv"
submit_df.to_csv(submit_path, index=False)

print("Wrote:", submit_path, "shape:", submit_df.shape)
print(submit_df.head())
