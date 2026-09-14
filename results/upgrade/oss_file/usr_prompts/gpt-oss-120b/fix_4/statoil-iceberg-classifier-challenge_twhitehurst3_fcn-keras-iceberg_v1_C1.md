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

No external packages required in the script and installed.

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

0.69312

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 0.69315) has done: 'I replace the broken keras imports with tensorflow‑keras, remove the invalid URL line, fix the model‑visualisation calls, adjust the callback imports, and correct the prediction call. These changes resolve the import and name errors, allow the script to run end‑to‑end, and ensure a proper .csv submission is written, while preserving the original network architecture and training procedure.'
- What this solution (achieved 0.69319) has done: 'I fixed the protobuf import issue by forcing the Python implementation before loading TensorFlow, corrected the model‑checkpoint filename and set it to save only weights (so it can be loaded later), switched the optimizer to Adam for faster convergence, and renumbered the cells to start at 1 while keeping the original workflow unchanged. These changes resolve all runtime errors and enable proper training, which should lower the log‑loss toward the target value.'
- What this solution (achieved 0.69312) has done: 'The fix updates the checkpoint filename so it complies with Keras’s `save_weights_only=True` requirement, adds proper normalization of the image data (crucial for reducing log‑loss), and adjusts the model‑weight loading path. These changes resolve the runtime errors and improve training stability, moving the validation log‑loss toward the target value while preserving the original network architecture.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras import Sequential, Model, Input
from tensorflow.keras.layers import (
    Dense,
    Flatten,
    Dropout,
    Lambda,
    SeparableConv2D,
    BatchNormalization,
    MaxPooling2D,
    Concatenate,
)
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    TensorBoard,
    CSVLogger,
    ReduceLROnPlateau,
)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def show_final_history(history):
    """Plot loss and accuracy curves."""
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].set_title("loss")
    ax[0].plot(history.epoch, history.history["loss"], label="Train loss")
    ax[0].plot(history.epoch, history.history["val_loss"], label="Validation loss")
    ax[1].set_title("accuracy")
    ax[1].plot(history.epoch, history.history.get("accuracy", []), label="Train acc")
    ax[1].plot(
        history.epoch, history.history.get("val_accuracy", []), label="Validation acc"
    )
    ax[0].legend()
    ax[1].legend()
    plt.show()




## === cell 2
train = pd.read_json("../input/train.json")
test = pd.read_json("../input/test.json")

X_band_1 = np.array(
    [np.array(b).astype(np.float32).reshape(75, 75) for b in train["band_1"]]
)
X_band_2 = np.array(
    [np.array(b).astype(np.float32).reshape(75, 75) for b in train["band_2"]]
)

X_train = np.concatenate(
    [
        X_band_1[..., np.newaxis],
        X_band_2[..., np.newaxis],
        ((X_band_1 + X_band_2) / 2)[..., np.newaxis],
    ],
    axis=-1,
)

y_train_full = train["is_iceberg"].values

x_train, x_val, y_train, y_val = train_test_split(
    X_train, y_train_full, random_state=1, train_size=0.80
)

train_mean = X_train.mean()
train_std = X_train.std()
X_train = (X_train - train_mean) / train_std
x_train = (x_train - train_mean) / train_std
x_val = (x_val - train_mean) / train_std




## === cell 3
def ConvBlock(model, layers, filters):
    """Add a series of separable conv → BN → max‑pool blocks."""
    for _ in range(layers):
        model.add(SeparableConv2D(filters, (3, 3), activation="relu", padding="same"))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))


def FCN():
    """Fully‑convolutional network as used in the original notebook."""
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
best_model_weights = "best_weights.weights.h5"

checkpoint = ModelCheckpoint(
    best_model_weights,
    monitor="val_loss",
    verbose=1,
    save_best_only=True,
    mode="min",
    save_weights_only=True,  # store only weights, compatible with load_weights
)

earlystop = EarlyStopping(
    monitor="val_loss", min_delta=0.001, patience=10, verbose=1, mode="min"
)

tensorboard = TensorBoard(
    log_dir="./logs",
    histogram_freq=0,
    batch_size=16,
    write_graph=True,
    write_images=False,
)

csvlogger = CSVLogger(filename="training_csv.log", separator=",", append=False)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=40, verbose=1, mode="min", cooldown=1
)

callbacks = [checkpoint, earlystop, tensorboard, csvlogger, reduce_lr]



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/418645585.py in <cell line: 0>()
     15 )
     16 
---> 17 tensorboard = TensorBoard(
     18     log_dir="./logs",
     19     histogram_freq=0,

TypeError: TensorBoard.__init__() got an unexpected keyword argument 'batch_size'

## === cell 5
opt = Adam(learning_rate=1e-4)

model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

history = model.fit(
    x_train,
    y_train,
    validation_data=(x_val, y_val),
    batch_size=32,
    epochs=30,
    verbose=1,
    callbacks=callbacks,
)



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1844587920.py in <cell line: 0>()
     10     epochs=30,
     11     verbose=1,
---> 12     callbacks=callbacks,
     13 )
     14 

NameError: name 'callbacks' is not defined

## === cell 6
show_final_history(history)



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/298676776.py in <cell line: 0>()
----> 1 show_final_history(history)
      2 

NameError: name 'history' is not defined

## === cell 7
model.load_weights(best_model_weights)
val_loss, val_acc = model.evaluate(x_val, y_val, verbose=1)
print("Validation LogLoss:", val_loss)
print("Validation Accuracy:", val_acc)



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/1986767449.py in <cell line: 0>()
----> 1 model.load_weights(best_model_weights)
      2 val_loss, val_acc = model.evaluate(x_val, y_val, verbose=1)
      3 print("Validation LogLoss:", val_loss)
      4 print("Validation Accuracy:", val_acc)
      5 

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/h5py/_hl/files.py in __init__(self, name, mode, driver, libver, userblock_size, swmr, rdcc_nslots, rdcc_nbytes, rdcc_w0, track_order, fs_strategy, fs_persist, fs_threshold, fs_page_size, page_buf_size, min_meta_keep, min_raw_keep, locking, alignment_threshold, alignment_interval, meta_block_size, **kwds)
    562                                  fs_persist=fs_persist, fs_threshold=fs_threshold,
    563                                  fs_page_size=fs_page_size)
--> 564                 fid = make_fid(name, mode, userblock_size, fapl, fcpl, swmr=swmr)
    565 
    566             if isinstance(libver, tuple):

/usr/local/lib/python3.11/dist-packages/h5py/_hl/files.py in make_fid(name, mode, userblock_size, fapl, fcpl, swmr)
    236         if swmr and swmr_support:
    237             flags |= h5f.ACC_SWMR_READ
--> 238         fid = h5f.open(name, flags, fapl=fapl)
    239     elif mode == 'r+':
    240         fid = h5f.open(name, h5f.ACC_RDWR, fapl=fapl)

h5py/_objects.pyx in h5py._objects.with_phil.wrapper()

h5py/_objects.pyx in h5py._objects.with_phil.wrapper()

h5py/h5f.pyx in h5py.h5f.open()

FileNotFoundError: [Errno 2] Unable to synchronously open file (unable to open file: name = 'best_weights.weights.h5', errno = 2, error message = 'No such file or directory', flags = 0, o_flags = 0)

## === cell 8
band1_test = np.array(
    [np.array(b).astype(np.float32).reshape(75, 75) for b in test["band_1"]]
)
band2_test = np.array(
    [np.array(b).astype(np.float32).reshape(75, 75) for b in test["band_2"]]
)

x_test = np.concatenate(
    [
        band1_test[..., np.newaxis],
        band2_test[..., np.newaxis],
        ((band1_test + band2_test) / 2)[..., np.newaxis],
    ],
    axis=-1,
)

x_test = (x_test - train_mean) / train_std

predictions = model.predict(x_test, batch_size=32, verbose=1).ravel()



## === cell 9
submission = pd.DataFrame({"id": test["id"], "is_iceberg": predictions})
submission.to_csv("IceBerg_FCN_sub.csv", index=False)
print("Submission saved to IceBerg_FCN_sub.csv")
