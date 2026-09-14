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

0.27455

# 6. Current score

0.30636

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 0.34736) has done: 'The timeout is dominated by training 200 epochs even though the model typically converges earlier; since you already configured EarlyStopping but never used it, the main fix is to actually include it in `callbacks` so training stops once `val_loss` plateaus (this preserves the same training semantics you intended: best weights still come from `ModelCheckpoint`). To further reduce per-epoch overhead without changing the model, loss, or data, the input pipeline is upgraded to cache + deterministic prefetch and the dataset is built with fixed `steps_per_epoch`/`validation_steps` to avoid extra iterator bookkeeping. TensorBoard writing is disabled by default (it’s pure overhead in Kaggle runs) while keeping the callback object available if you re-enable it. All paths, architecture, optimizer, and evaluation remain the same; only unnecessary work is removed.'
- What this solution (achieved 0.30636) has done: 'The crash happens before any training due to an incompatibility between TensorFlow and the installed protobuf runtime; it’s triggered during `import tensorflow` and shows up as `MessageFactory.GetPrototype` missing. The minimal, Kaggle-standard fix is to force protobuf to use the pure-Python implementation *before* importing TensorFlow, which avoids that API mismatch without changing your model/training logic. I also keep deterministic settings but move all environment-variable configuration above the TensorFlow import so it actually takes effect. No architecture, loss, optimizer, or data processing is changed, so this should be score-neutral while making the notebook run end-to-end and write `submission.csv`.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import math
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Lambda
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

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

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
BASE_INPUT = "/kaggle/input/statoil-iceberg-classifier-challenge"
train_path = os.path.join(BASE_INPUT, "train.json")
test_path = os.path.join(BASE_INPUT, "test.json")

train = pd.read_json(train_path)
test = pd.read_json(test_path)

X_band_1 = np.asarray(train["band_1"].to_list(), dtype=np.float32).reshape(-1, 75, 75)
X_band_2 = np.asarray(train["band_2"].to_list(), dtype=np.float32).reshape(-1, 75, 75)

X_train = np.concatenate(
    [
        X_band_1[..., np.newaxis],
        X_band_2[..., np.newaxis],
        ((X_band_1 + X_band_2) / 2.0)[..., np.newaxis],
    ],
    axis=-1,
).astype(np.float32, copy=False)

target_train = train["is_iceberg"].astype(np.float32).values

x_train, x_val, y_train, y_val = train_test_split(
    X_train, target_train, random_state=1, train_size=0.80, stratify=target_train
)

BATCH_SIZE = 32

options = tf.data.Options()
options.experimental_deterministic = True

train_steps = int(math.ceil(len(x_train) / BATCH_SIZE))
val_steps = int(math.ceil(len(x_val) / BATCH_SIZE))

train_ds = (
    tf.data.Dataset.from_tensor_slices((x_train, y_train))
    .with_options(options)
    .cache()
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(tf.data.AUTOTUNE)
)
val_ds = (
    tf.data.Dataset.from_tensor_slices((x_val, y_val))
    .with_options(options)
    .cache()
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(tf.data.AUTOTUNE)
)




## === cell 3
def ConvBlock(model, layers, filters):
    for _ in range(layers):
        model.add(SeparableConv2D(filters, (3, 3), activation="relu"))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))


def FCN():
    model = Sequential()
    model.add(Lambda(lambda x: x, input_shape=(75, 75, 3)))
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
best_model_weights = "./base.weights.h5"

checkpoint = ModelCheckpoint(
    filepath=best_model_weights,
    monitor="val_loss",
    verbose=1,
    save_best_only=True,
    mode="min",
    save_weights_only=True,
)

earlystop = EarlyStopping(
    monitor="val_loss",
    min_delta=0.001,
    patience=10,
    verbose=1,
    mode="auto",
    restore_best_weights=False,  # keep core behavior: use checkpoint for best weights
)

tensorboard = TensorBoard(
    log_dir="./logs",
    histogram_freq=0,
    write_graph=False,
    write_images=False,
    profile_batch=0,
)

csvlogger = CSVLogger(filename="training_csv.log", separator=",", append=False)

reduce = ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=40, verbose=1, mode="auto", cooldown=1
)

callbacks = [checkpoint, earlystop, csvlogger, reduce]



## === cell 5
opt = SGD(learning_rate=1e-4, momentum=0.95)
opt1 = Adam(learning_rate=1e-3)  # kept for parity with original code (unused)

model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

history = model.fit(
    train_ds,
    validation_data=val_ds,
    verbose=1,
    epochs=200,
    callbacks=callbacks,
    steps_per_epoch=train_steps,
    validation_steps=val_steps,
)



## === cell 6
show_final_history(history)



## === cell 7
model.load_weights(best_model_weights)
model_score = model.evaluate(val_ds, verbose=1, steps=val_steps)
print("Model Val Loss:", model_score[0])
print("Model Val Accuracy:", model_score[1])



## === cell 8
band1_test = np.asarray(test["band_1"].to_list(), dtype=np.float32).reshape(-1, 75, 75)
band2_test = np.asarray(test["band_2"].to_list(), dtype=np.float32).reshape(-1, 75, 75)

x_test = np.concatenate(
    [
        band1_test[..., np.newaxis],
        band2_test[..., np.newaxis],
        ((band1_test + band2_test) / 2.0)[..., np.newaxis],
    ],
    axis=-1,
).astype(np.float32, copy=False)

options = tf.data.Options()
options.experimental_deterministic = True

test_ds = (
    tf.data.Dataset.from_tensor_slices(x_test)
    .with_options(options)
    .cache()
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(tf.data.AUTOTUNE)
)
predictions = model.predict(test_ds, verbose=1)



## === cell 9
submission = pd.DataFrame(
    {"id": test["id"].values, "is_iceberg": predictions.reshape(-1)}
)
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print("Wrote:", submission_path, "with shape:", submission.shape)
print(submission.head())
