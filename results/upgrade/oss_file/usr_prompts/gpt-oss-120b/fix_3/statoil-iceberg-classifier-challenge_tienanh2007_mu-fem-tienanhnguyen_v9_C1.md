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

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

train_path = os.path.join("input", "train.json")
test_path = os.path.join("input", "test.json")

train_df = pd.read_json(train_path)
test_df = pd.read_json(test_path)

print("Loaded data:", train_df.shape, test_df.shape)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/3073724824.py in <cell line: 0>()
      7 test_path = os.path.join("input", "test.json")
      8 
----> 9 train_df = pd.read_json(train_path)
     10 test_df = pd.read_json(test_path)
     11 

/usr/local/lib/python3.11/dist-packages/pandas/io/json/_json.py in read_json(path_or_buf, orient, typ, dtype, convert_axes, convert_dates, keep_default_dates, precise_float, date_unit, encoding, encoding_errors, lines, chunksize, compression, nrows, storage_options, dtype_backend, engine)
    789         convert_axes = True
    790 
--> 791     json_reader = JsonReader(
    792         path_or_buf,
    793         orient=orient,

/usr/local/lib/python3.11/dist-packages/pandas/io/json/_json.py in __init__(self, filepath_or_buffer, orient, typ, dtype, convert_axes, convert_dates, keep_default_dates, precise_float, date_unit, encoding, lines, chunksize, compression, nrows, storage_options, encoding_errors, dtype_backend, engine)
    902             self.data = filepath_or_buffer
    903         elif self.engine == "ujson":
--> 904             data = self._get_data_from_filepath(filepath_or_buffer)
    905             self.data = self._preprocess_data(data)
    906 

/usr/local/lib/python3.11/dist-packages/pandas/io/json/_json.py in _get_data_from_filepath(self, filepath_or_buffer)
    958             and not file_exists(filepath_or_buffer)
    959         ):
--> 960             raise FileNotFoundError(f"File {filepath_or_buffer} does not exist")
    961         else:
    962             warnings.warn(

FileNotFoundError: File input/train.json does not exist

## === cell 1
x_band1 = np.array(
    [np.array(band, dtype=np.float32).reshape(75, 75) for band in train_df["band_1"]]
)
x_band2 = np.array(
    [np.array(band, dtype=np.float32).reshape(75, 75) for band in train_df["band_2"]]
)
X_train = np.concatenate([x_band1[..., np.newaxis], x_band2[..., np.newaxis]], axis=-1)
y_train = np.array(train_df["is_iceberg"])
print("X_train shape:", X_train.shape)

x_band1_test = np.array(
    [np.array(band, dtype=np.float32).reshape(75, 75) for band in test_df["band_1"]]
)
x_band2_test = np.array(
    [np.array(band, dtype=np.float32).reshape(75, 75) for band in test_df["band_2"]]
)
X_test = np.concatenate(
    [x_band1_test[..., np.newaxis], x_band2_test[..., np.newaxis]], axis=-1
)
print("X_test shape:", X_test.shape)




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2660553207.py in <cell line: 0>()
      1 # Load and reshape band data
      2 x_band1 = np.array(
----> 3     [np.array(band, dtype=np.float32).reshape(75, 75) for band in train_df["band_1"]]
      4 )
      5 x_band2 = np.array(

NameError: name 'train_df' is not defined

## === cell 2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Dense,
    Dropout,
    Flatten,
    Activation,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 3
def getModel():
    gmodel = Sequential()
    gmodel.add(
        Conv2D(64, kernel_size=(3, 3), activation="relu", input_shape=(75, 75, 2))
    )
    gmodel.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    gmodel.add(Dropout(0.2))

    gmodel.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
    gmodel.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    gmodel.add(Dropout(0.2))

    gmodel.add(Conv2D(128, kernel_size=(3, 3), activation="relu"))
    gmodel.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    gmodel.add(Dropout(0.2))

    gmodel.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
    gmodel.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    gmodel.add(Dropout(0.2))

    gmodel.add(Flatten())

    gmodel.add(Dense(512))
    gmodel.add(Activation("relu"))
    gmodel.add(Dropout(0.2))

    gmodel.add(Dense(256))
    gmodel.add(Activation("relu"))
    gmodel.add(Dropout(0.2))

    gmodel.add(Dense(1, activation="sigmoid"))

    optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    gmodel.compile(
        loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"]
    )
    gmodel.summary()
    return gmodel


def get_callbacks(filepath, patience=2):
    es = EarlyStopping(monitor="val_loss", patience=patience, mode="min", verbose=1)
    msave = ModelCheckpoint(
        filepath, save_best_only=True, monitor="val_loss", mode="min", verbose=1
    )
    return [es, msave]


file_path = "model_weights.keras"
callbacks = get_callbacks(filepath=file_path, patience=5)




## === cell 4
X_train_cv, X_valid, y_train_cv, y_valid = train_test_split(
    X_train, y_train, random_state=1, train_size=0.75
)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/501157537.py in <cell line: 0>()
      1 # Split training data
      2 X_train_cv, X_valid, y_train_cv, y_valid = train_test_split(
----> 3     X_train, y_train, random_state=1, train_size=0.75
      4 )
      5 

NameError: name 'X_train' is not defined

## === cell 5
model = getModel()
model.fit(
    X_train_cv,
    y_train_cv,
    batch_size=24,
    epochs=50,
    verbose=1,
    validation_data=(X_valid, y_valid),
    callbacks=callbacks,
)




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/178697159.py in <cell line: 0>()
      1 model = getModel()
      2 model.fit(
----> 3     X_train_cv,
      4     y_train_cv,
      5     batch_size=24,

NameError: name 'X_train_cv' is not defined

## === cell 6
predicted_test = model.predict(X_test, batch_size=24, verbose=0)




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2942391587.py in <cell line: 0>()
----> 1 predicted_test = model.predict(X_test, batch_size=24, verbose=0)
      2 
      3 

NameError: name 'X_test' is not defined

## === cell 7
submit_df = pd.DataFrame({"id": test_df["id"], "is_iceberg": predicted_test.flatten()})
submit_df.to_csv("naive_submission.csv", index=False)
print("Submission saved to naive_submission.csv")

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2049207458.py in <cell line: 0>()
----> 1 submit_df = pd.DataFrame({"id": test_df["id"], "is_iceberg": predicted_test.flatten()})
      2 submit_df.to_csv("naive_submission.csv", index=False)
      3 print("Submission saved to naive_submission.csv")

NameError: name 'test_df' is not defined
