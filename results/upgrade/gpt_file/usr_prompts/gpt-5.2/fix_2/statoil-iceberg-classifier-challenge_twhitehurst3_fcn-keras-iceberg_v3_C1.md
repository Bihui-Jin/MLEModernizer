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

0.3247

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import math
import shutil
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
train_path = "/kaggle/input/train.json"
test_path = "/kaggle/input/test.json"
if not os.path.exists(train_path) or not os.path.exists(test_path):
    train_path = "/kaggle/input/statoil-iceberg-classifier-challenge/train.json"
    test_path = "/kaggle/input/statoil-iceberg-classifier-challenge/test.json"

train = pd.read_json(train_path)
test = pd.read_json(test_path)

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
        ((X_band_1 + X_band_2) / 2.0)[:, :, :, np.newaxis],
    ],
    axis=-1,
).astype(np.float32)

target_train = train["is_iceberg"].astype(np.float32).values

x_train, x_val, y_train, y_val = train_test_split(
    X_train, target_train, random_state=1, train_size=0.80, stratify=target_train
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
best_model_weights = "./base.model.keras"  # use a standard keras format filename
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
best_model = tf.keras.models.load_model(best_model_weights)
model_score = best_model.evaluate(x_val, y_val, verbose=1)
print("Model Val Loss:", model_score[0])
print("Model Val Accuracy:", model_score[1])



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/4106823620.py in <cell line: 0>()
      1 # NOTE: Load best model (ModelCheckpoint saved the full model because save_weights_only=False).
----> 2 best_model = tf.keras.models.load_model(best_model_weights)
      3 model_score = best_model.evaluate(x_val, y_val, verbose=1)
      4 print("Model Val Loss:", model_score[0])
      5 print("Model Val Accuracy:", model_score[1])

/usr/local/lib/python3.11/dist-packages/keras/src/saving/saving_api.py in load_model(filepath, custom_objects, compile, safe_mode)
    187 
    188     if is_keras_zip or is_keras_dir or is_hf:
--> 189         return saving_lib.load_model(
    190             filepath,
    191             custom_objects=custom_objects,

/usr/local/lib/python3.11/dist-packages/keras/src/saving/saving_lib.py in load_model(filepath, custom_objects, compile, safe_mode)
    365             )
    366         with open(filepath, "rb") as f:
--> 367             return _load_model_from_fileobj(
    368                 f, custom_objects, compile, safe_mode
    369             )

/usr/local/lib/python3.11/dist-packages/keras/src/saving/saving_lib.py in _load_model_from_fileobj(fileobj, custom_objects, compile, safe_mode)
    442             config_json = f.read()
    443 
--> 444         model = _model_from_config(
    445             config_json, custom_objects, compile, safe_mode
    446         )

/usr/local/lib/python3.11/dist-packages/keras/src/saving/saving_lib.py in _model_from_config(config_json, custom_objects, compile, safe_mode)
    431     # Construct the model from the configuration file in the archive.
    432     with ObjectSharingScope():
--> 433         model = deserialize_keras_object(
    434             config_dict, custom_objects, safe_mode=safe_mode
    435         )

/usr/local/lib/python3.11/dist-packages/keras/src/saving/serialization_lib.py in deserialize_keras_object(config, custom_objects, safe_mode, **kwargs)
    716     with custom_obj_scope, safe_mode_scope:
    717         try:
--> 718             instance = cls.from_config(inner_config)
    719         except TypeError as e:
    720             raise TypeError(

/usr/local/lib/python3.11/dist-packages/keras/src/models/sequential.py in from_config(cls, config, custom_objects)
    353                 )
    354             else:
--> 355                 layer = serialization_lib.deserialize_keras_object(
    356                     layer_config,
    357                     custom_objects=custom_objects,

/usr/local/lib/python3.11/dist-packages/keras/src/saving/serialization_lib.py in deserialize_keras_object(config, custom_objects, safe_mode, **kwargs)
    716     with custom_obj_scope, safe_mode_scope:
    717         try:
--> 718             instance = cls.from_config(inner_config)
    719         except TypeError as e:
    720             raise TypeError(

/usr/local/lib/python3.11/dist-packages/keras/src/layers/core/lambda_layer.py in from_config(cls, config, custom_objects, safe_mode)
    188             and fn_config["class_name"] == "__lambda__"
    189         ):
--> 190             cls._raise_for_lambda_deserialization("function", safe_mode)
    191             inner_config = fn_config["config"]
    192             fn = python_utils.func_load(

/usr/local/lib/python3.11/dist-packages/keras/src/layers/core/lambda_layer.py in _raise_for_lambda_deserialization(arg_name, safe_mode)
    170     def _raise_for_lambda_deserialization(arg_name, safe_mode):
    171         if safe_mode:
--> 172             raise ValueError(
    173                 "The `{arg_name}` of this `Lambda` layer is a Python lambda. "
    174                 "Deserializing it is unsafe. If you trust the source of the "

ValueError: The `{arg_name}` of this `Lambda` layer is a Python lambda. Deserializing it is unsafe. If you trust the source of the config artifact, you can override this error by passing `safe_mode=False` to `from_config()`, or calling `keras.config.enable_unsafe_deserialization().

## === cell 8
band1_test = np.array(
    [np.array(band).astype(np.float32).reshape(75, 75) for band in test["band_1"]]
)
band2_test = np.array(
    [np.array(band).astype(np.float32).reshape(75, 75) for band in test["band_2"]]
)

x_test = np.concatenate(
    [
        band1_test[:, :, :, np.newaxis],
        band2_test[:, :, :, np.newaxis],
        ((band1_test + band2_test) / 2.0)[:, :, :, np.newaxis],
    ],
    axis=-1,
).astype(np.float32)

predictions = best_model.predict(x_test, batch_size=32, verbose=1)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2801902167.py in <cell line: 0>()
     16 
     17 # NOTE: Keras models use predict(), not predict_proba().
---> 18 predictions = best_model.predict(x_test, batch_size=32, verbose=1)
     19 

NameError: name 'best_model' is not defined

## === cell 9
submission = pd.DataFrame(
    {"id": test["id"].values, "is_iceberg": predictions.reshape((-1,))}
)
submission["is_iceberg"] = submission["is_iceberg"].clip(0.0, 1.0)

out_path = "IceBerg_FCN_sub.csv"
submission.to_csv(out_path, index=False)
print("Wrote submission:", out_path)
print(submission.head())



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/329831753.py in <cell line: 0>()
      1 submission = pd.DataFrame(
----> 2     {"id": test["id"].values, "is_iceberg": predictions.reshape((-1,))}
      3 )
      4 # Safety: clip to [0,1] to avoid any numerical edge cases for log loss.
      5 submission["is_iceberg"] = submission["is_iceberg"].clip(0.0, 1.0)

NameError: name 'predictions' is not defined

## === cell 10
pass
