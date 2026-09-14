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

0.35989

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 0.35326) has done: 'The fixes address mismatched array sizes by using the actual number of training and test samples, remove unnecessary Keras imports that cause protobuf errors, and correctly shape the prediction array before creating the submission CSV.'
- What this solution (achieved 1.94699) has done: 'The fix switches to TensorFlow Keras to avoid the protobuf import error, adds a validation split with early‑stopping (so the model keeps the best weights and yields a lower log‑loss), computes and prints the validation log‑loss, and then creates the required `submission.csv` file. No changes are made to the network architecture or core preprocessing.'
- What this solution (achieved 0.56703) has done: 'Implemented fixes to resolve import errors, corrected validation prediction logic, and added simple data normalization to improve model training stability and log‑loss. The script now runs end‑to‑end and writes a proper `submission.csv` file.'
- What this solution (achieved 0.34378) has done: 'Implemented two focused fixes:  
1. Replaced the failing `import tensorflow` with `import tf_keras as tf`, which matches the available package and resolves the protobuf import error.  
2. Increased the training epochs to 100 (still using early stopping) to allow the model more opportunity to converge, nudging the validation log‑loss toward the target score.  

The rest of the pipeline remains unchanged, and the script now runs end‑to‑end and writes a proper `submission.csv`.'
- What this solution (achieved 0.35989) has done: 'Implemented fixes to resolve the protobuf import error by using `tf_keras` exclusively for Keras components, and added lightweight regularization (Dropout layers) plus a modest increase in training epochs to improve validation log‑loss without altering the overall architecture. These changes keep the core logic intact while nudging the score toward the target and ensure a proper `submission.csv` is written.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import numpy as np
import pandas as pd
from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



## === cell 1
train_df = pd.read_json("../input/train.json")
test_df = pd.read_json("../input/test.json")
train_df.head()



## === cell 2
X_band_1 = np.array(
    [np.array(band).astype(np.float32).reshape(75, 75) for band in train_df["band_1"]]
)
X_band_2 = np.array(
    [np.array(band).astype(np.float32).reshape(75, 75) for band in train_df["band_2"]]
)

n_train = len(train_df)
X_band = np.zeros([n_train, 75, 75, 3], dtype=np.float32)
for t in range(n_train):
    X_band[t, :, :, 0] = X_band_1[t]
    X_band[t, :, :, 1] = X_band_2[t]
    X_band[t, :, :, 2] = (X_band_1[t] + X_band_2[t]) / 2

mean = X_band.mean()
std = X_band.std()
X_band = (X_band - mean) / std



## === cell 3
import tf_keras as tf
from tf_keras import layers, Model, backend as K
from tf_keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 4
def Iceberg_model(input_shape):
    X_in = layers.Input(shape=input_shape)

    X = layers.Conv2D(16, kernel_size=(5, 5))(X_in)
    X = layers.BatchNormalization()(X)
    X = layers.Activation("relu")(X)
    X = layers.MaxPooling2D(pool_size=(2, 2))(X)

    X = layers.Conv2D(32, kernel_size=(5, 5))(X)
    X = layers.BatchNormalization()(X)
    X = layers.Activation("relu")(X)
    X = layers.MaxPooling2D(pool_size=(2, 2))(X)

    X = layers.Conv2D(64, kernel_size=(5, 5))(X)
    X = layers.BatchNormalization()(X)
    X = layers.Activation("relu")(X)
    X = layers.MaxPooling2D(pool_size=(2, 2))(X)

    X = layers.Flatten()(X)

    X = layers.Dense(128)(X)
    X = layers.Activation("relu")(X)
    X = layers.Dropout(0.3)(X)  # lightweight regularization

    X = layers.Dense(64)(X)
    X = layers.Activation("relu")(X)
    X = layers.Dropout(0.3)(X)  # lightweight regularization

    X = layers.Dense(1)(X)
    X = layers.Activation("sigmoid")(X)

    model = Model(inputs=X_in, outputs=X, name="Iceberg_model")
    return model




## === cell 5
IcebergModel = Iceberg_model((75, 75, 3))



## === cell 6
IcebergModel.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])



## === cell 7
target = train_df["is_iceberg"].values
early_stop = EarlyStopping(patience=5, restore_best_weights=True, verbose=1)

history = IcebergModel.fit(
    x=X_band,
    y=target,
    epochs=150,  # a bit more epochs for better convergence
    batch_size=128,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=2,
)



## === cell 8
from sklearn.metrics import log_loss, classification_report

val_size = int(0.2 * n_train)
X_val = X_band[-val_size:]
y_val = target[-val_size:]

val_pred = IcebergModel.predict(X_val).reshape(-1)
val_logloss = log_loss(y_val, val_pred)
print(f"Validation LogLoss: {val_logloss:.5f}")

print(classification_report(y_val, (val_pred > 0.5).astype(int)))



## === cell 9
X_band_test_1 = np.array(
    [np.array(band).astype(np.float32).reshape(75, 75) for band in test_df["band_1"]]
)
X_band_test_2 = np.array(
    [np.array(band).astype(np.float32).reshape(75, 75) for band in test_df["band_2"]]
)
n_test = len(test_df)
X_test = np.zeros([n_test, 75, 75, 3], dtype=np.float32)
for t in range(n_test):
    X_test[t, :, :, 0] = X_band_test_1[t]
    X_test[t, :, :, 1] = X_band_test_2[t]
    X_test[t, :, :, 2] = (X_band_test_1[t] + X_band_test_2[t]) / 2

X_test = (X_test - mean) / std



## === cell 10
pred = IcebergModel.predict(X_test).reshape(-1)
sub_df = pd.DataFrame({"id": test_df["id"], "is_iceberg": pred})
sub_df.to_csv("submission.csv", index=False)
