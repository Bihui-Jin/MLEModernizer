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
google-api-python-client==2.177.0
ipython==7.34.0
ipython-genutils==0.2.0
ipython_pygments_lexers==1.1.1
ipython-sql==0.5.0
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

0.24988

# 6. Current score

0.41989

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 0.40192) has done: 'I fix the dataset-shape bugs by deriving train/test sizes from the loaded JSON instead of hardcoding 1604/8424, so indexing and data cardinality match. I also resolve the Keras import/runtime error by removing legacy/unused visualization utilities (`vis_utils`, `model_to_dot`, protobuf-dependent pieces) and keeping only the layers/models used for training. Finally, I ensure predictions are 1D probabilities aligned with `test_df['id']` and write a valid `submission.csv` with the required columns and `.csv` suffix.'
- What this solution (achieved 0.4445) has done: 'I fix the Keras runtime crash by switching from the standalone `keras` import (which is triggering a protobuf `MessageFactory.GetPrototype` incompatibility in this environment) to the Kaggle-stable `tf_keras` package you already have installed, without changing the model architecture or training loop. I also add a small, score-improving but still minimal preprocessing step (per-image standardization) that preserves the same input shape and model semantics, and typically improves log loss for this competition. Finally, I ensure the submission probabilities are clipped to a safe (0,1) range to avoid any potential numerical issues with log loss, and keep the submission format exactly `id,is_iceberg` in a `.csv` file.'
- What this solution (achieved 0.59693) has done: 'We fix the protobuf `MessageFactory.GetPrototype` crash by avoiding the `tf_keras` import path that triggers it in this environment and instead use `tensorflow.keras`, which is compatible with the Kaggle TensorFlow/protobuf stack. All model architecture, optimizer/loss, epochs, batch size, and preprocessing (including your per-image standardization and 3-channel construction) are kept the same to preserve evaluation semantics while restoring end-to-end execution. We also add a small deterministic seeding block (score-neutral/stability) and keep the submission writer exactly `id,is_iceberg` to `submission.csv`.'
- What this solution (achieved 0.61735) has done: 'We fix the crash in the TensorFlow/Keras import stack caused by an incompatible protobuf/tensorflow pairing (`MessageFactory.GetPrototype`) by switching to the environment-stable `tf_keras` package (already installed) and explicitly forcing legacy Keras behavior, without changing your model architecture or training loop. We keep your preprocessing (3-channel construction + per-image standardization) identical, and keep deterministic seeds for stability. After the model runs end-to-end, we ensure predictions are a 1D float array aligned to `test_df["id"]` and write a valid `submission.csv` with columns `id,is_iceberg`. These changes should both unblock execution and (since your current run is failing before training) move the score back toward your target by producing a properly trained submission.'
- What this solution (achieved 0.32024) has done: 'I fix the runtime crash caused by importing `tf_keras` (protobuf `MessageFactory.GetPrototype` incompatibility) by switching to the environment-stable `tensorflow.keras` import path while keeping the exact same model architecture, compile settings, and training loop. I also keep your existing preprocessing (3-channel construction + per-image standardization) and seeding intact, because those are part of your current scoring behavior. Finally, I ensure the submission is written as `submission.csv` with the required `id,is_iceberg` columns and predictions are 1D clipped probabilities aligned to `test_df["id"]`.'
- What this solution (achieved 0.53496) has done: 'I fix the runtime crash caused by the TensorFlow/protobuf incompatibility (`MessageFactory.GetPrototype`) by switching the Keras imports to the already-installed `tf_keras` package, which avoids that protobuf path in this environment. I keep your model architecture, preprocessing (3-channel construction + per-image standardization), optimizer/loss, and training loop unchanged so the evaluation semantics remain the same. I also add a small compatibility guard so the backend image data format call won’t fail across Keras variants, and I keep the submission writing exactly `id,is_iceberg` to `submission.csv`.'
- What this solution (achieved 0.37569) has done: 'I fix the protobuf crash by avoiding the `tf_keras` import path that triggers `MessageFactory.GetPrototype` in this environment, and instead import Keras from `tensorflow.keras` (same layers/model API, same architecture and training loop). I keep your preprocessing (3-channel construction + per-image standardization), optimizer/loss, epochs, and batch size unchanged to preserve evaluation semantics and move logloss back down toward the target. I also add a small, score-neutral safety guard so `inc_angle` parsing doesn’t break anything if later used, and keep the submission writing exactly `id,is_iceberg` to `submission.csv`.'
- What this solution (achieved 0.50998) has done: 'I fix the runtime crash in the TensorFlow/Keras import stack (`MessageFactory.GetPrototype`) by avoiding `tensorflow.keras` entirely and using the already-installed `tf_keras` package instead, while keeping your exact model architecture, preprocessing, optimizer/loss, epochs, and batch size unchanged. I also add a small compatibility fallback so the script still runs even if a backend call like `set_image_data_format` is unavailable. These changes are execution-unblocking and should restore the previously working training/inference path, which is expected to improve log loss from the currently worse (higher) score toward your target. The submission writing remains `submission.csv` with columns `id,is_iceberg` aligned to `test_df["id"]`.'
- What this solution (achieved 0.4979) has done: 'I fix the runtime crash caused by importing `tf_keras`, which is triggering a protobuf `MessageFactory.GetPrototype` incompatibility in this environment, by switching the Keras imports to `tensorflow.keras` (same API, same model/loss/training loop). I keep your preprocessing (3-channel construction + per-image standardization), model architecture, optimizer/loss, and training settings unchanged so evaluation semantics remain the same while restoring end-to-end execution. I also keep deterministic seeding and ensure the submission is written as `submission.csv` with columns `id,is_iceberg`, with predictions shaped and clipped safely for log loss.'
- What this solution (achieved 0.38147) has done: 'I fix the protobuf `MessageFactory.GetPrototype` crash by avoiding the `tensorflow`/`tensorflow.keras` import path in this environment and switching to the already-installed `tf_keras` package, which is compatible here. This is an execution-unblocking change that keeps your exact model architecture, compile settings, preprocessing, and training loop intact (so evaluation semantics stay the same). I also keep seeding/stability and ensure the submission is written as a valid `submission.csv` with `id,is_iceberg` aligned to `test_df["id"]` and probabilities clipped for log loss safety. These changes should both restore end-to-end execution and move log loss down from the current 0.4979 toward your target.'
- What this solution (achieved 0.34519) has done: 'I fix the crash in the Keras import stack (`MessageFactory.GetPrototype`) by avoiding the incompatible `tf_keras` package in this environment and switching to the stable `tensorflow.keras` API, keeping your exact model architecture, compile settings, preprocessing, and training loop unchanged. I also make the import logic robust with a small fallback so the notebook runs regardless of which Keras backend is available. These changes are execution-unblocking and should also improve your score simply by letting the model train/infer successfully with the same semantics as intended. The submission writing remain `submission.csv` with `id,is_iceberg` aligned to `test_df["id"]` and clipped probabilities for logloss safety.'
- What this solution (achieved 0.41989) has done: 'I fix the runtime crash (`MessageFactory` has no `GetPrototype`) by avoiding the TensorFlow/Keras import path that triggers the protobuf incompatibility in this environment, and instead importing from the installed `tf_keras` package with legacy Keras enabled. This is an execution-unblocking change and keeps your model architecture, preprocessing, optimizer/loss, epochs, and batch size unchanged, so it should also move log loss down toward your target simply by producing a proper trained submission. I also keep a small import fallback so the notebook still runs if one backend import fails, without changing training semantics. The submission writing remains `submission.csv` with the required `id,is_iceberg` columns and clipped probabilities.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

random.seed(SEED)
np.random.seed(SEED)

INPUT_DIR_CANDIDATES = [
    "../input",
    "/kaggle/input",
    "/kaggle/data/input",
    "/kaggle/data",
]
INPUT_DIR = None
for d in INPUT_DIR_CANDIDATES:
    if os.path.isdir(d):
        if os.path.exists(os.path.join(d, "train.json")) and os.path.exists(
            os.path.join(d, "test.json")
        ):
            INPUT_DIR = d
            break
if INPUT_DIR is None:
    INPUT_DIR = "../input"

print("Using INPUT_DIR:", INPUT_DIR)
print("Files in INPUT_DIR:", os.listdir(INPUT_DIR)[:20])



## === cell 1
train_df = pd.read_json(os.path.join(INPUT_DIR, "train.json"))
test_df = pd.read_json(os.path.join(INPUT_DIR, "test.json"))

for df in (train_df, test_df):
    if "inc_angle" in df.columns:
        df["inc_angle"] = pd.to_numeric(df["inc_angle"], errors="coerce")

try:
    display(train_df.head())
except NameError:
    print(train_df.head())
print("train_df shape:", train_df.shape, " test_df shape:", test_df.shape)




## === cell 2
def _to_image_array(series):
    return np.array(
        [np.array(b, dtype=np.float32).reshape(75, 75) for b in series],
        dtype=np.float32,
    )


def _standardize_per_image(x, eps=1e-6):
    mean = x.mean(axis=(1, 2, 3), keepdims=True)
    std = x.std(axis=(1, 2, 3), keepdims=True)
    return (x - mean) / (std + eps)


X_band_1 = _to_image_array(train_df["band_1"])
X_band_2 = _to_image_array(train_df["band_2"])

n_train = len(train_df)
X_band = np.zeros((n_train, 75, 75, 3), dtype=np.float32)
X_band[:, :, :, 0] = X_band_1
X_band[:, :, :, 1] = X_band_2
X_band[:, :, :, 2] = (X_band_1 + X_band_2) / 2.0

X_band = _standardize_per_image(X_band)

target = train_df["is_iceberg"].astype(np.float32).values
print("X_band:", X_band.shape, "target:", target.shape)



## === cell 3
try:
    import tensorflow as tf  # still used for seeding; safe to import

    try:
        tf.random.set_seed(SEED)
    except Exception:
        pass

    import tf_keras as keras
    from tf_keras import layers
    from tf_keras.layers import (
        Input,
        Dense,
        Activation,
        BatchNormalization,
        Flatten,
        Conv2D,
        MaxPooling2D,
    )
    from tf_keras.models import Model
    from tf_keras import backend as K

    try:
        K.set_image_data_format("channels_last")
    except Exception as e:
        print("Warning: could not set image data format:", repr(e))

    print("Using tensorflow version:", getattr(tf, "__version__", "unknown"))
    print("Using tf_keras version:", getattr(keras, "__version__", "unknown"))
except Exception as e_main:
    print(
        "Warning: tf_keras import failed, falling back to standalone keras. Error:",
        repr(e_main),
    )
    import keras
    from keras import layers
    from keras.layers import (
        Input,
        Dense,
        Activation,
        BatchNormalization,
        Flatten,
        Conv2D,
        MaxPooling2D,
    )
    from keras.models import Model
    from keras import backend as K

    try:
        K.set_image_data_format("channels_last")
    except Exception as e:
        print("Warning: could not set image data format:", repr(e))

    print("Using keras version:", getattr(keras, "__version__", "unknown"))




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 4
def Iceberg_model(input_shape):
    X_in = Input(input_shape)

    X = Conv2D(16, kernel_size=(5, 5), input_shape=(75, 75, 3))(X_in)
    X = BatchNormalization()(X)
    X = Activation("relu")(X)
    X = MaxPooling2D(pool_size=(2, 2))(X)

    X = Conv2D(32, kernel_size=(5, 5))(X)
    X = BatchNormalization()(X)
    X = Activation("relu")(X)
    X = MaxPooling2D(pool_size=(2, 2))(X)

    X = Conv2D(64, kernel_size=(5, 5))(X)
    X = BatchNormalization()(X)
    X = Activation("relu")(X)
    X = MaxPooling2D(pool_size=(2, 2))(X)

    X = Flatten()(X)

    X = Dense(128)(X)
    X = Activation("relu")(X)

    X = Dense(64)(X)
    X = Activation("relu")(X)

    X = Dense(1)(X)
    X = Activation("sigmoid")(X)

    model = Model(inputs=X_in, outputs=X, name="Iceberg_model")
    return model




## === cell 5
IcebergModel = Iceberg_model((75, 75, 3))
IcebergModel.summary()



## === cell 6
IcebergModel.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])



## === cell 7
history = IcebergModel.fit(x=X_band, y=target, epochs=20, batch_size=128, verbose=2)



## === cell 8
eval_out = IcebergModel.evaluate(x=X_band, y=target, verbose=0)
print("Train loss, acc:", eval_out)



## === cell 9
from sklearn.metrics import classification_report

pred_label = IcebergModel.predict(x=X_band, verbose=0).reshape(-1)
pred_bin = (pred_label > 0.5).astype(int)
print(classification_report(target.astype(int), pred_bin))



## === cell 10
X_band_test_1 = _to_image_array(test_df["band_1"])
X_band_test_2 = _to_image_array(test_df["band_2"])

n_test = len(test_df)
X_test = np.zeros((n_test, 75, 75, 3), dtype=np.float32)
X_test[:, :, :, 0] = X_band_test_1
X_test[:, :, :, 1] = X_band_test_2
X_test[:, :, :, 2] = (X_band_test_1 + X_band_test_2) / 2.0

X_test = _standardize_per_image(X_test)

print("X_test:", X_test.shape)

pred = IcebergModel.predict(x=X_test, verbose=0).reshape(-1)
pred = np.clip(pred, 1e-6, 1.0 - 1e-6).astype(np.float32)

sub_df = pd.DataFrame({"id": test_df["id"].values, "is_iceberg": pred})
sub_path = "submission.csv"
sub_df.to_csv(sub_path, index=False)

print("Wrote:", sub_path)
try:
    display(sub_df.head())
except NameError:
    print(sub_df.head())
print("Submission shape:", sub_df.shape)
print("Submission columns:", sub_df.columns.tolist())
