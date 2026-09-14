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

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 0.69315) has done: 'I replace the broken keras imports with tensorflow‑keras, remove the invalid URL line, fix the model‑visualisation calls, adjust the callback imports, and correct the prediction call. These changes resolve the import and name errors, allow the script to run end‑to‑end, and ensure a proper .csv submission is written, while preserving the original network architecture and training procedure.'
- What this solution (achieved 0.69319) has done: 'I fixed the protobuf import issue by forcing the Python implementation before loading TensorFlow, corrected the model‑checkpoint filename and set it to save only weights (so it can be loaded later), switched the optimizer to Adam for faster convergence, and renumbered the cells to start at 1 while keeping the original workflow unchanged. These changes resolve all runtime errors and enable proper training, which should lower the log‑loss toward the target value.'
- What this solution (achieved 0.69312) has done: 'The fix updates the checkpoint filename so it complies with Keras’s `save_weights_only=True` requirement, adds proper normalization of the image data (crucial for reducing log‑loss), and adjusts the model‑weight loading path. These changes resolve the runtime errors and improve training stability, moving the validation log‑loss toward the target value while preserving the original network architecture.'
- What this solution (achieved 0.34493) has done: 'I fix the TensorBoard initialization error, ensure the callbacks list is created before training, increase training epochs and adjust early‑stopping to allow better convergence, and add class‑weight balancing to improve validation log‑loss. These changes resolve the runtime failures and should lower the log‑loss toward the target while keeping the original model architecture intact.'
- What this solution (achieved 0.49351) has done: 'The changes set a global NumPy seed for reproducibility, reduce the GradientBoostingClassifier trees from 300 to 100 (cutting training time by ~3× while keeping the same model type), and keep all data handling identical. These tweaks stay within the original algorithmic logic and produce the same predictions format.'

# 9. Code solution

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


angles = parse_angle(train["inc_angle"]).values  # shape (n_samples,)
x_train, x_val, y_train, y_val, angle_train, angle_val = train_test_split(
    X_train, y_train_full, angles, random_state=1, train_size=0.80
)

train_mean = X_train.mean()
train_std = X_train.std()
X_train = (X_train - train_mean) / train_std
x_train = (x_train - train_mean) / train_std
x_val = (x_val - train_mean) / train_std

angle_mean = angle_train.mean()
angle_std = angle_train.std()
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
x_test_flat = x_test.reshape((x_test.shape[0], -1))

angle_test = parse_angle(test["inc_angle"]).values
angle_test = np.where(np.isnan(angle_test), np.median(angles), angle_test)
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
    n_estimators=300,  # increased from 100 to give more capacity
    learning_rate=0.05,
    max_depth=3,
    random_state=1,
)

model.fit(x_train_flat, y_train, sample_weight=sample_weights)

val_pred_proba = model.predict_proba(x_val_flat)[:, 1]
val_loss = log_loss(y_val, val_pred_proba)
val_acc = accuracy_score(y_val, (val_pred_proba > 0.5).astype(int))

print("Validation LogLoss:", val_loss)
print("Validation Accuracy:", val_acc)




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1436204526.py in <cell line: 0>()
     12 )
     13 
---> 14 model.fit(x_train_flat, y_train, sample_weight=sample_weights)
     15 
     16 val_pred_proba = model.predict_proba(x_val_flat)[:, 1]

/usr/local/lib/python3.11/dist-packages/sklearn/ensemble/_gb.py in fit(self, X, y, sample_weight, monitor)
    427         # trees use different types for X and y, checking them separately.
    428 
--> 429         X, y = self._validate_data(
    430             X, y, accept_sparse=["csr", "csc", "coo"], dtype=DTYPE, multi_output=True
    431         )

/usr/local/lib/python3.11/dist-packages/sklearn/base.py in _validate_data(self, X, y, reset, validate_separately, **check_params)
    582                 y = check_array(y, input_name="y", **check_y_params)
    583             else:
--> 584                 X, y = check_X_y(X, y, **check_params)
    585             out = X, y
    586 

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in check_X_y(X, y, accept_sparse, accept_large_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, multi_output, ensure_min_samples, ensure_min_features, y_numeric, estimator)
   1104         )
   1105 
-> 1106     X = check_array(
   1107         X,
   1108         accept_sparse=accept_sparse,

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in check_array(array, accept_sparse, accept_large_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, ensure_min_samples, ensure_min_features, estimator, input_name)
    919 
    920         if force_all_finite:
--> 921             _assert_all_finite(
    922                 array,
    923                 input_name=input_name,

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in _assert_all_finite(X, allow_nan, msg_dtype, estimator_name, input_name)
    159                 "#estimators-that-handle-nan-values"
    160             )
--> 161         raise ValueError(msg_err)
    162 
    163 

ValueError: Input X contains NaN.
GradientBoostingClassifier does not accept missing values encoded as NaN natively. For supervised learning, you might want to consider sklearn.ensemble.HistGradientBoostingClassifier and Regressor which accept missing values encoded as NaNs natively. Alternatively, it is possible to preprocess the data, for instance by using an imputer transformer in a pipeline or drop samples with missing values. See https://scikit-learn.org/stable/modules/impute.html You can find a list of all estimators that handle NaN values at the following page: https://scikit-learn.org/stable/modules/impute.html#estimators-that-handle-nan-values

## === cell 5
predictions = model.predict_proba(x_test_flat)[:, 1]

submission = pd.DataFrame({"id": test["id"], "is_iceberg": predictions})
submission.to_csv("IceBerg_FCN_sub.csv", index=False)
print("Submission saved to IceBerg_FCN_sub.csv")

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/288887439.py in <cell line: 0>()
----> 1 predictions = model.predict_proba(x_test_flat)[:, 1]
      2 
      3 submission = pd.DataFrame({"id": test["id"], "is_iceberg": predictions})
      4 submission.to_csv("IceBerg_FCN_sub.csv", index=False)
      5 print("Submission saved to IceBerg_FCN_sub.csv")

/usr/local/lib/python3.11/dist-packages/sklearn/ensemble/_gb.py in predict_proba(self, X)
   1353             If the ``loss`` does not support probabilities.
   1354         """
-> 1355         raw_predictions = self.decision_function(X)
   1356         try:
   1357             return self._loss._raw_prediction_to_proba(raw_predictions)

/usr/local/lib/python3.11/dist-packages/sklearn/ensemble/_gb.py in decision_function(self, X)
   1259             array of shape (n_samples,).
   1260         """
-> 1261         X = self._validate_data(
   1262             X, dtype=DTYPE, order="C", accept_sparse="csr", reset=False
   1263         )

/usr/local/lib/python3.11/dist-packages/sklearn/base.py in _validate_data(self, X, y, reset, validate_separately, **check_params)
    563             raise ValueError("Validation should be done on X, y or both.")
    564         elif not no_val_X and no_val_y:
--> 565             X = check_array(X, input_name="X", **check_params)
    566             out = X
    567         elif no_val_X and not no_val_y:

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in check_array(array, accept_sparse, accept_large_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, ensure_min_samples, ensure_min_features, estimator, input_name)
    919 
    920         if force_all_finite:
--> 921             _assert_all_finite(
    922                 array,
    923                 input_name=input_name,

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in _assert_all_finite(X, allow_nan, msg_dtype, estimator_name, input_name)
    159                 "#estimators-that-handle-nan-values"
    160             )
--> 161         raise ValueError(msg_err)
    162 
    163 

ValueError: Input X contains NaN.
GradientBoostingClassifier does not accept missing values encoded as NaN natively. For supervised learning, you might want to consider sklearn.ensemble.HistGradientBoostingClassifier and Regressor which accept missing values encoded as NaNs natively. Alternatively, it is possible to preprocess the data, for instance by using an imputer transformer in a pipeline or drop samples with missing values. See https://scikit-learn.org/stable/modules/impute.html You can find a list of all estimators that handle NaN values at the following page: https://scikit-learn.org/stable/modules/impute.html#estimators-that-handle-nan-values
