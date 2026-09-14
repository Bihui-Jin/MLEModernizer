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

3.7

# 3. Installed packages



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

0.3247

# 6. Current score

0.45327

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.69334) has done: 'We fix the environment/runtime crash that happens on `import tensorflow` by forcing the pure-Python protobuf implementation before TensorFlow loads. Then we fix the model save/load failure caused by the `Lambda(lambda x: x, ...)` layer under newer Keras safe deserialization by replacing it with an equivalent safe layer (`InputLayer`) while keeping the same architecture semantics. Finally, we ensure the best model is available for inference (fallback to the in-memory trained model if needed) and always write a valid `submission.csv` with the required `id,is_iceberg` columns.'
- What this solution (achieved 0.43616) has done: 'The immediate blocker is the TensorFlow import crash caused by an incompatible protobuf runtime; fixing that cleanly requires forcing a compatible protobuf version before importing TensorFlow (the current env var workaround isn’t sufficient in this Kaggle image). After it runs, the 0.693 logloss indicates the model is effectively untrained/mis-calibrated; the smallest legitimate improvement (without changing architecture or training loop) is to apply the standard per-image normalization used for this competition (zero-mean/unit-std per band) and to correctly use `inc_angle` as an additional input feature by folding it into the existing 3rd channel (keeping input shape and architecture unchanged). Finally, we keep the same submission format but add safety checks to ensure predictions align with test ids and always write a valid `submission.csv`.'
- What this solution (achieved 0.69316) has done: 'We fix the TensorFlow import crash by pinning protobuf to the pure-Python implementation *and* patching the specific `MessageFactory.GetPrototype` incompatibility that triggers the error in this Kaggle image. This is a runtime-only compatibility shim and does not change your model/training logic. Then we keep everything else the same so training/inference run end-to-end and a valid `submission.csv` is always written. With TF loading correctly again, your existing preprocessing (per-image standardization + incidence-angle folding) and training should execute and is expected to improve logloss versus the current broken run.'
- What this solution (achieved 0.45327) has done: 'I fix the protobuf shim that currently throws an AttributeError by safely patching both the class and instance method cases (and making it a no-op when not needed), so TensorFlow can import reliably. I also add a small, score-neutral safety fallback to load the best checkpoint with `compile=False` if Keras deserialization/compile metadata causes issues, while keeping the same model/training logic. Finally, I keep all paths and submission formatting the same, ensuring `submission.csv` is always written with `id,is_iceberg` aligned to the test ids.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "3")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import math
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import google.protobuf  # noqa: F401
from google.protobuf import message_factory as _message_factory


def _ensure_getprototype():
    MF = _message_factory.MessageFactory

    def _GetPrototype(self, descriptor):
        if hasattr(self, "GetMessageClass"):
            return self.GetMessageClass(descriptor)
        if hasattr(_message_factory, "GetMessageClass"):
            return _message_factory.GetMessageClass(descriptor)
        raise AttributeError("No GetMessageClass available to emulate GetPrototype")

    if not hasattr(MF, "GetPrototype"):
        setattr(MF, "GetPrototype", _GetPrototype)

    try:
        inst = MF()
        if not hasattr(inst, "GetPrototype"):
            setattr(inst, "GetPrototype", _GetPrototype.__get__(inst, MF))
    except Exception:
        pass


_ensure_getprototype()

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, InputLayer
from tensorflow.keras.layers import SeparableConv2D, BatchNormalization, MaxPooling2D
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    TensorBoard,
    CSVLogger,
    ReduceLROnPlateau,
)

np.random.seed(1)
tf.random.set_seed(1)

print("TensorFlow version:", tf.__version__)




## === cell 1
def show_final_history(history):
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].set_title("loss")
    ax[0].plot(history.epoch, history.history.get("loss", []), label="Train loss")
    ax[0].plot(
        history.epoch, history.history.get("val_loss", []), label="Validation loss"
    )
    ax[1].set_title("acc")
    ax[1].plot(
        history.epoch,
        history.history.get("accuracy", history.history.get("acc", [])),
        label="Train acc",
    )
    ax[1].plot(
        history.epoch,
        history.history.get("val_accuracy", history.history.get("val_acc", [])),
        label="Validation acc",
    )
    ax[0].legend()
    ax[1].legend()
    plt.show()




## === cell 2
train_path = "/kaggle/input/train.json"
test_path = "/kaggle/input/test.json"
if not os.path.exists(train_path) or not os.path.exists(test_path):
    train_path = "/kaggle/input/statoil-iceberg-classifier-challenge/train.json"
    test_path = "/kaggle/input/statoil-iceberg-classifier-challenge/test.json"

train = pd.read_json(train_path)
test = pd.read_json(test_path)


def _to_band_array(series):
    return np.array(
        [np.array(band, dtype=np.float32).reshape(75, 75) for band in series]
    )


def _per_image_standardize(x):
    mean = x.mean(axis=(1, 2), keepdims=True)
    std = x.std(axis=(1, 2), keepdims=True)
    return (x - mean) / (std + 1e-6)


def _clean_inc_angle(s):
    s = s.replace("na", np.nan)
    ang = pd.to_numeric(s, errors="coerce").astype(np.float32)
    return ang.values


X_band_1 = _to_band_array(train["band_1"])
X_band_2 = _to_band_array(train["band_2"])

X_band_1 = _per_image_standardize(X_band_1)
X_band_2 = _per_image_standardize(X_band_2)

inc_train = _clean_inc_angle(train["inc_angle"])
inc_median = np.nanmedian(inc_train)
inc_train = np.where(np.isfinite(inc_train), inc_train, inc_median).astype(np.float32)

inc_train_scaled = ((inc_train - inc_median) / 10.0).astype(np.float32)  # ~order 0.1-1
inc_train_map = inc_train_scaled.reshape(-1, 1, 1)

avg_band = (X_band_1 + X_band_2) / 2.0
third_channel = avg_band + inc_train_map  # broadcast to 75x75

X_train = np.concatenate(
    [
        X_band_1[:, :, :, np.newaxis],
        X_band_2[:, :, :, np.newaxis],
        third_channel[:, :, :, np.newaxis],
    ],
    axis=-1,
).astype(np.float32)

target_train = train["is_iceberg"].astype(np.float32).values

x_train, x_val, y_train, y_val = train_test_split(
    X_train, target_train, random_state=1, train_size=0.80, stratify=target_train
)

print("Train/Val shapes:", x_train.shape, x_val.shape, y_train.shape, y_val.shape)
print("Inc angle median:", float(inc_median))




## === cell 3
def ConvBlock(model, layers, filters):
    for _ in range(layers):
        model.add(SeparableConv2D(filters, (3, 3), activation="relu"))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))


def FCN():
    model = Sequential()
    model.add(InputLayer(input_shape=(75, 75, 3)))
    ConvBlock(model, 1, 64)
    ConvBlock(model, 1, 128)
    ConvBlock(model, 1, 128)
    ConvBlock(model, 1, 64)
    model.add(Flatten())
    model.add(Dense(1024, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation="sigmoid"))
    return model


model = FCN()
model.summary()



## === cell 4
best_model_weights = "./base.model.keras"  # standard keras format filename
checkpoint = ModelCheckpoint(
    best_model_weights,
    monitor="val_loss",
    verbose=1,
    save_best_only=True,
    mode="min",
    save_weights_only=False,
)

earlystop = EarlyStopping(
    monitor="val_loss",
    min_delta=0.001,
    patience=10,
    verbose=1,
    mode="auto",
)

tensorboard = TensorBoard(
    log_dir="./logs",
    histogram_freq=0,
    write_graph=True,
    write_images=False,
)

csvlogger = CSVLogger(
    filename="training_csv.log",
    separator=",",
    append=False,
)

reduce = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=40,
    verbose=1,
    mode="auto",
    cooldown=1,
)

callbacks = [checkpoint, tensorboard, csvlogger, reduce, earlystop]



## === cell 5
opt = SGD(learning_rate=1e-4, momentum=0.95)
opt1 = Adam(learning_rate=2e-4)

model.compile(
    loss="binary_crossentropy",
    optimizer=opt1,
    metrics=["accuracy"],
)

history = model.fit(
    x_train,
    y_train,
    validation_data=(x_val, y_val),
    batch_size=32,
    verbose=1,
    epochs=200,
    callbacks=callbacks,
)



## === cell 6
show_final_history(history)



## === cell 7
if os.path.exists(best_model_weights):
    try:
        best_model = tf.keras.models.load_model(best_model_weights)
    except Exception as e:
        print("load_model failed, retrying with compile=False. Error:", repr(e))
        best_model = tf.keras.models.load_model(best_model_weights, compile=False)
else:
    best_model = model

model_score = best_model.evaluate(x_val, y_val, verbose=1)
print("Model Val Loss:", model_score[0])
print("Model Val Accuracy:", model_score[1])



## === cell 8
band1_test = _to_band_array(test["band_1"])
band2_test = _to_band_array(test["band_2"])

band1_test = _per_image_standardize(band1_test)
band2_test = _per_image_standardize(band2_test)

inc_test = _clean_inc_angle(test["inc_angle"])
inc_test = np.where(np.isfinite(inc_test), inc_test, inc_median).astype(np.float32)
inc_test_scaled = ((inc_test - inc_median) / 10.0).astype(np.float32)
inc_test_map = inc_test_scaled.reshape(-1, 1, 1)

avg_band_test = (band1_test + band2_test) / 2.0
third_channel_test = avg_band_test + inc_test_map

x_test = np.concatenate(
    [
        band1_test[:, :, :, np.newaxis],
        band2_test[:, :, :, np.newaxis],
        third_channel_test[:, :, :, np.newaxis],
    ],
    axis=-1,
).astype(np.float32)

predictions = best_model.predict(x_test, batch_size=32, verbose=1)



## === cell 9
preds = predictions.reshape((-1,)).astype(np.float64)
preds = np.clip(preds, 0.0, 1.0)

test_ids = test["id"].values
if len(test_ids) != len(preds):
    raise RuntimeError(
        f"Prediction length mismatch: ids={len(test_ids)} preds={len(preds)}"
    )

submission = pd.DataFrame({"id": test_ids, "is_iceberg": preds})

out_path = "submission.csv"
submission.to_csv(out_path, index=False)
print("Wrote submission:", out_path)
print(submission.head())
print("Submission shape:", submission.shape)
