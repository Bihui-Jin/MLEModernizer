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

0.4825

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.69315) has done: 'I replace the broken keras imports with tensorflow‑keras, remove the invalid URL line, fix the model‑visualisation calls, adjust the callback imports, and correct the prediction call. These changes resolve the import and name errors, allow the script to run end‑to‑end, and ensure a proper .csv submission is written, while preserving the original network architecture and training procedure.'
- What this solution (achieved 0.69319) has done: 'I fixed the protobuf import issue by forcing the Python implementation before loading TensorFlow, corrected the model‑checkpoint filename and set it to save only weights (so it can be loaded later), switched the optimizer to Adam for faster convergence, and renumbered the cells to start at 1 while keeping the original workflow unchanged. These changes resolve all runtime errors and enable proper training, which should lower the log‑loss toward the target value.'
- What this solution (achieved 0.69312) has done: 'The fix updates the checkpoint filename so it complies with Keras’s `save_weights_only=True` requirement, adds proper normalization of the image data (crucial for reducing log‑loss), and adjusts the model‑weight loading path. These changes resolve the runtime errors and improve training stability, moving the validation log‑loss toward the target value while preserving the original network architecture.'
- What this solution (achieved 0.34493) has done: 'I fix the TensorBoard initialization error, ensure the callbacks list is created before training, increase training epochs and adjust early‑stopping to allow better convergence, and add class‑weight balancing to improve validation log‑loss. These changes resolve the runtime failures and should lower the log‑loss toward the target while keeping the original model architecture intact.'
- What this solution (achieved 0.49351) has done: 'The changes set a global NumPy seed for reproducibility, reduce the GradientBoostingClassifier trees from 300 to 100 (cutting training time by ~3× while keeping the same model type), and keep all data handling identical. These tweaks stay within the original algorithmic logic and produce the same predictions format.'
- What this solution (achieved 0.49885) has done: 'I reduced the training cost by lowering the number of trees and limiting the features each tree examines, which cuts the per‑estimator work dramatically while keeping the same GradientBoostingClassifier model and overall pipeline unchanged. These hyper‑parameter tweaks preserve deterministic behavior and the same data preprocessing, so the predictions remain comparable but the fit finishes well within the 600 s limit.'
- What this solution (achieved 0.48245) has done: 'I keep the same preprocessing and GradientBoosting model but increase the number of trees to give the algorithm more capacity, then select the iteration that gives the lowest validation log‑loss using the staged predictions. This modest change stays within the original pipeline while usually reduces log‑loss, moving the score closer to the target.'
- What this solution (achieved 0.4825) has done: 'I slightly increase the model capacity by raising the number of trees to 500 and lowering the learning rate to 0.03. This keeps the original GradientBoosting pipeline unchanged while giving the ensemble more opportunity to find a lower‑loss iteration, which should move the validation log‑loss (and thus the final Kaggle score) closer to the target 0.27455.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import log_loss, accuracy_score

np.random.seed(1)


def _import_plt():
    import matplotlib.pyplot as plt

    return plt


def show_final_history(history):
    """Plot loss and accuracy curves."""
    plt = _import_plt()
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




## === cell 1
train = pd.read_json("../input/train.json")
test = pd.read_json("../input/test.json")

band1_train = np.array(train["band_1"].tolist(), dtype=np.float32)  # (n, 5625)
band2_train = np.array(train["band_2"].tolist(), dtype=np.float32)  # (n, 5625)

n_train = band1_train.shape[0]
X_full = np.empty((n_train, 5625 * 3), dtype=np.float32)
X_full[:, :5625] = band1_train
X_full[:, 5625:11250] = band2_train
X_full[:, 11250:] = ((band1_train + band2_train) / 2).astype(np.float32)

y_train_full = train["is_iceberg"].values


def parse_angle(series):
    """Convert inc_angle to numeric, coercing 'na' to NaN."""
    return pd.to_numeric(series, errors="coerce")


angles = parse_angle(train["inc_angle"]).values  # may contain NaN

train_mean = np.nanmean(X_full, axis=0, keepdims=True).astype(np.float32)
train_std = np.nanstd(X_full, axis=0, keepdims=True).astype(np.float32)
train_std = np.where(train_std == 0, 1, train_std)

X_full -= train_mean
X_full /= train_std
np.nan_to_num(X_full, copy=False)

x_train, x_val, y_train, y_val, angle_train, angle_val = train_test_split(
    X_full, y_train_full, angles, random_state=1, train_size=0.80
)

angle_median = np.nanmedian(angles)

angle_train = np.where(np.isnan(angle_train), angle_median, angle_train)
angle_val = np.where(np.isnan(angle_val), angle_median, angle_val)

angle_mean = angle_train.mean()
angle_std = angle_train.std()
angle_std = angle_std if angle_std != 0 else 1.0

angle_train = (angle_train - angle_mean) / angle_std
angle_val = (angle_val - angle_mean) / angle_std

x_train = np.column_stack([x_train, angle_train])
x_val = np.column_stack([x_val, angle_val])

band1_test = np.array(test["band_1"].tolist(), dtype=np.float32)
band2_test = np.array(test["band_2"].tolist(), dtype=np.float32)
n_test = band1_test.shape[0]
x_test = np.empty((n_test, 5625 * 3), dtype=np.float32)
x_test[:, :5625] = band1_test
x_test[:, 5625:11250] = band2_test
x_test[:, 11250:] = ((band1_test + band2_test) / 2).astype(np.float32)

x_test -= train_mean
x_test /= train_std
np.nan_to_num(x_test, copy=False)

angle_test = parse_angle(test["inc_angle"]).values
angle_test = np.where(np.isnan(angle_test), angle_median, angle_test)
angle_test = (angle_test - angle_mean) / angle_std
x_test = np.column_stack([x_test, angle_test])




## === cell 2
callbacks = []  # placeholder for compatibility; not used by GradientBoosting




## === cell 3
class_weights = class_weight.compute_class_weight(
    class_weight="balanced", classes=np.unique(y_train), y=y_train
)
class_weights_dict = dict(zip(np.unique(y_train), class_weights))

unique_classes = np.array(list(class_weights_dict.keys()))
weights_array = np.array(list(class_weights_dict.values()), dtype=np.float32)
sample_weights = weights_array[np.searchsorted(unique_classes, y_train)]

model = GradientBoostingClassifier(
    n_estimators=500,  # increased from 300
    learning_rate=0.03,  # decreased to keep training stable
    max_depth=4,
    max_features="sqrt",
    subsample=0.9,
    random_state=1,
)

model.fit(x_train, y_train, sample_weight=sample_weights)

val_pred_list = list(model.staged_predict_proba(x_val))
val_losses = [log_loss(y_val, preds[:, 1]) for preds in val_pred_list]
best_iter = int(np.argmin(val_losses))  # zero‑based index
best_val_loss = val_losses[best_iter]

val_acc = accuracy_score(y_val, (val_pred_list[best_iter][:, 1] > 0.5).astype(int))

print("Best validation iteration (0‑based):", best_iter)
print("Validation LogLoss (best):", best_val_loss)
print("Validation Accuracy (best):", val_acc)




## === cell 4
test_pred_list = list(model.staged_predict_proba(x_test))
best_test_pred = test_pred_list[best_iter][:, 1]

submission = pd.DataFrame({"id": test["id"], "is_iceberg": best_test_pred})
submission.to_csv("IceBerg_FCN_sub.csv", index=False)
print("Submission saved to IceBerg_FCN_sub.csv")
