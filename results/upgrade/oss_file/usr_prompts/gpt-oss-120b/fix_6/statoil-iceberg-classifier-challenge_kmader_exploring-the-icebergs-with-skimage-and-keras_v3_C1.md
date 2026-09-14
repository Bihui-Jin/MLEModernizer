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

0.37955

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 0.41695) has done: 'The changes fix file discovery by pointing the default search directory to Kaggle’s standard `/kaggle/input` location and switch the model imports to TensorFlow‑Keras, which resolves the incompatibility error with the installed Keras version. Minor adjustments ensure the prediction column is written correctly. No core modelling logic is altered, preserving the original approach while enabling the script to run end‑to‑end and produce a valid `predictions.csv` file.'
- What this solution (achieved 0.37955) has done: 'I replace the TensorFlow‑Keras imports with pure Keras (compatible with the installed keras 3) to fix the protobuf MessageFactory error, and I modestly increase model capacity and training epochs (while keeping the overall architecture) so the log‑loss moves closer to the target. The rest of the pipeline stays unchanged, and the script now reliably writes a predictions.csv file.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from skimage.util import montage2d  # older versions
except ImportError:
    from skimage.util import montage as montage2d  # newer versions

base_path = os.getenv("KAGGLE_INPUT_DIR", "/kaggle/input")
if not os.path.isdir(base_path):
    base_path = os.path.join(os.getcwd(), "input")


def find_file(filename: str) -> str:
    """Search for *filename* inside *base_path* and its sub‑folders.
    Returns the first matching path or raises FileNotFoundError."""
    for root, _, files in os.walk(base_path):
        if filename in files:
            return os.path.join(root, filename)
    raise FileNotFoundError(f"{filename} not found under {base_path}")


def load_and_format(in_path):
    """Load the json file and reshape the two bands into (75,75,2) images."""
    df = pd.read_json(in_path)
    images = np.stack(
        df.apply(
            lambda r: np.stack(
                [np.array(r["band_1"]), np.array(r["band_2"])], axis=-1
            ).reshape(75, 75, 2),
            axis=1,
        ).values
    )
    return df, images


train_json = find_file("train.json")
test_json = find_file("test.json")

train_df, train_images = load_and_format(train_json)
print("training", train_df.shape, "loaded", train_images.shape)

test_df, test_images = load_and_format(test_json)
print("testing", test_df.shape, "loaded", test_images.shape)




## === cell 1
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.matshow(train_images[0, :, :, 0])
ax1.set_title("Band 1")
ax2.matshow(train_images[0, :, :, 1])
ax2.set_title("Band 2")
plt.close(fig)  # close to avoid display in non‑interactive runs




## === cell 2
fig, (ax1s, ax2s) = plt.subplots(2, 2, figsize=(8, 8))
obj_list = dict(
    ships=train_df.query("is_iceberg==0").sample(16, random_state=42).index,
    icebergs=train_df.query("is_iceberg==1").sample(16, random_state=42).index,
)
for ax1, ax2, (obj_type, idx_list) in zip(ax1s, ax2s, obj_list.items()):
    ax1.imshow(montage2d(train_images[idx_list, :, :, 0]))
    ax1.set_title(f"{obj_type} Band 1")
    ax1.axis("off")
    ax2.imshow(montage2d(train_images[idx_list, :, :, 1]))
    ax2.set_title(f"{obj_type} Band 2")
    ax2.axis("off")
plt.close(fig)




## === cell 3
from sklearn.model_selection import train_test_split

y = train_df["is_iceberg"].values.astype("float32")

X_train, X_val, y_train, y_val = train_test_split(
    train_images,
    y,
    test_size=0.5,
    random_state=2017,
    shuffle=True,
    stratify=y,
)
print("Train", X_train.shape, y_train.shape)
print("Validation", X_val.shape, y_val.shape)




## === cell 4
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

simple_cnn = Sequential()
simple_cnn.add(BatchNormalization(input_shape=(75, 75, 2)))
for i in range(4):
    simple_cnn.add(Conv2D(16 * (2**i), kernel_size=(3, 3), activation="relu"))
    simple_cnn.add(MaxPooling2D(pool_size=(2, 2)))
simple_cnn.add(GlobalMaxPooling2D())
simple_cnn.add(Dropout(0.5))
simple_cnn.add(Dense(8, activation="relu"))
simple_cnn.add(Dense(1, activation="sigmoid"))

simple_cnn.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
simple_cnn.summary()




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 5
simple_cnn.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=30,  # more epochs for better convergence
    shuffle=True,
    verbose=2,
)




## === cell 6
test_predictions = simple_cnn.predict(test_images, verbose=0)




## === cell 7
pred_df = test_df[["id"]].copy()
pred_df["is_iceberg"] = test_predictions.ravel()
pred_df.to_csv("predictions.csv", index=False)
print("Saved predictions to predictions.csv")
pred_df.sample(3)
