# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import log_loss, accuracy_score

np.random.seed(1)




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


def parse_angle(series):
    """Convert inc_angle to numeric, coercing 'na' to NaN."""
    return pd.to_numeric(series, errors="coerce")


angles = parse_angle(train["inc_angle"]).values  # may contain NaN

x_train, x_val, y_train, y_val, angle_train, angle_val = train_test_split(
    X_train, y_train_full, angles, random_state=1, train_size=0.80
)

train_mean = np.nanmean(X_train, axis=(0, 1, 2, 3), keepdims=True)
train_std = np.nanstd(X_train, axis=(0, 1, 2, 3), keepdims=True)

train_std = np.where(train_std == 0, 1, train_std)

X_train = (X_train - train_mean) / train_std
x_train = (x_train - train_mean) / train_std
x_val = (x_val - train_mean) / train_std

X_train = np.nan_to_num(X_train)
x_train = np.nan_to_num(x_train)
x_val = np.nan_to_num(x_val)

angle_median = np.nanmedian(angles)
angle_train = np.where(np.isnan(angle_train), angle_median, angle_train)
angle_val = np.where(np.isnan(angle_val), angle_median, angle_val)

angle_mean = angle_train.mean()
angle_std = angle_train.std()
angle_std = angle_std if angle_std != 0 else 1.0
angle_train = (angle_train - angle_mean) / angle_std
angle_val = (angle_val - angle_mean) / angle_std

x_train_flat = x_train.reshape((x_train.shape[0], -1))
x_val_flat = x_val.reshape((x_val.shape[0], -1))

x_train_flat = np.hstack([x_train_flat, angle_train.reshape(-1, 1)])
x_val_flat = np.hstack([x_val_flat, angle_val.reshape(-1, 1)])

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
x_test = np.nan_to_num(x_test)

x_test_flat = x_test.reshape((x_test.shape[0], -1))

angle_test = parse_angle(test["inc_angle"]).values
angle_test = np.where(np.isnan(angle_test), angle_median, angle_test)
angle_test = (angle_test - angle_mean) / angle_std
x_test_flat = np.hstack([x_test_flat, angle_test.reshape(-1, 1)])



## === cell 3
callbacks = []



## === cell 4
class_weights = class_weight.compute_class_weight(
    class_weight="balanced", classes=np.unique(y_train), y=y_train
)
class_weights_dict = dict(zip(np.unique(y_train), class_weights))
sample_weights = np.vectorize(class_weights_dict.get)(y_train)

model = GradientBoostingClassifier(
    n_estimators=500,  # increased from 300
    learning_rate=0.05,
    max_depth=4,  # increased from 3
    random_state=1,
)

model.fit(x_train_flat, y_train, sample_weight=sample_weights)

val_pred_proba = model.predict_proba(x_val_flat)[:, 1]
val_loss = log_loss(y_val, val_pred_proba)
val_acc = accuracy_score(y_val, (val_pred_proba > 0.5).astype(int))

print("Validation LogLoss:", val_loss)
print("Validation Accuracy:", val_acc)



## === cell 5
predictions = model.predict_proba(x_test_flat)[:, 1]

submission = pd.DataFrame({"id": test["id"], "is_iceberg": predictions})
submission.to_csv("IceBerg_FCN_sub.csv", index=False)
print("Submission saved to IceBerg_FCN_sub.csv")
