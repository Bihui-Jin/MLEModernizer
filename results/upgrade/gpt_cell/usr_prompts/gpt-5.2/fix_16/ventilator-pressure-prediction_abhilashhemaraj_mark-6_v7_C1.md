# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Given time series of breaths, predict the airway pressure in the respiratory circuit during the breath, given the time series of control inputs.

The best submissions will take lung attributes compliance and resistance into account.

## Metric
Mean absolute error between the predicted and actual pressures during the inspiratory phase of each breath. The expiratory phase is not scored.

## Submission Format
For each `id` in the test set, you must predict a value for the `pressure` variable. The file should contain a header and have the following format:

```
id,pressure
1,20
2,23
3,24
etc.
```

## Dataset
The ventilator data used in this competition was produced using a modified [open-source ventilator](https://pvp.readthedocs.io/) connected to an [artificial bellows test lung](https://www.ingmarmed.com/product/quicklung/) via a respiratory circuit. The diagram below illustrates the setup, with the two control inputs highlighted in green and the state variable (airway pressure) to predict in blue. The first control input is a continuous variable from 0 to 100 representing the percentage the inspiratory solenoid valve is open to let air into the lung (i.e., 0 is completely closed and no air is let in and 100 is completely open). The second control input is a binary variable representing whether the exploratory valve is open (1) or closed (0) to let air out.

![Ventilator diagram](https://raw.githubusercontent.com/google/deluca-lung/main/assets/2020-10-02%20Ventilator%20diagram.svg)

Each time series represents an approximately 3-second breath. The files are organized such that each row is a time step in a breath and gives the two control signals, the resulting airway pressure, and relevant attributes of the lung, described below.

### Files
- **train.csv** - the training set
- **test.csv** - the test set
- **sample_submission.csv** - a sample submission file in the correct format

### Columns
- `id` - globally-unique time step identifier across an entire file
- `breath_id` - globally-unique time step for breaths
- `R` - lung attribute indicating how restricted the airway is (in cmH2O/L/S). Physically, this is the change in pressure per change in flow (air volume per time). Intuitively, one can imagine blowing up a balloon through a straw. We can change `R` by changing the diameter of the straw, with higher `R` being harder to blow.
- `C` - lung attribute indicating how compliant the lung is (in mL/cmH2O). Physically, this is the change in volume per change in pressure. Intuitively, one can imagine the same balloon example. We can change `C` by changing the thickness of the balloon’s latex, with higher `C` having thinner latex and easier to blow.
- `time_step` - the actual time stamp.
- `u_in` - the control input for the inspiratory solenoid valve. Ranges from 0 to 100.
- `u_out` - the control input for the exploratory solenoid valve. Either 0 or 1.
- `pressure` - the airway pressure measured in the respiratory circuit, measured in cmH2O.

# 2. Python version

3.9

# 3. Installed packages

geopandas==0.14.4
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
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1
tqdm==4.67.1

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (82 lines)
            sample_submission.csv (603601 lines)
            sample_submission.csv.zip (1.3 MB)
            test.csv (603601 lines)
            test.csv.zip (8.5 MB)
            train.csv (5432401 lines)
            train.csv.zip (93.8 MB)
            ventilator-pressure-prediction/
                description.md (82 lines)
                sample_submission.csv (603601 lines)
                ... and 5 other files
                ventilator-pressure-prediction/
        input/
            description.md (82 lines)
            sample_submission.csv (603601 lines)
            sample_submission.csv.zip (1.3 MB)
            test.csv (603601 lines)
            test.csv.zip (8.5 MB)
            train.csv (5432401 lines)
            train.csv.zip (93.8 MB)
            ventilator-pressure-prediction/
                description.md (82 lines)
                sample_submission.csv (603601 lines)
                ... and 5 other files
                ventilator-pressure-prediction/
        working/
            ventilator-pressure-prediction/
                description.md (82 lines)
                sample_submission.csv (603601 lines)
                ... and 5 other files
                ventilator-pressure-prediction/
```

-> data/sample_submission.csv has 603600 rows and 2 columns.
Here is some information about the columns:
id (int64) has range: 1.00 - 2000.00, 0 nan values
pressure (int64) has 1 unique values: [0]

-> data/test.csv has 603600 rows and 7 columns.
Here is some information about the columns:
C (int64) has 3 unique values: [20, 50, 10]
R (int64) has 3 unique values: [50, 5, 20]
breath_id (int64) has range: 2436.00 - 124535.00, 0 nan values
id (int64) has range: 1.00 - 2000.00, 0 nan values
time_step (float64) has range: 0.00 - 2.73, 0 nan values
u_in (float64) has range: 0.00 - 100.00, 0 nan values
u_out (int64) has 2 unique values: [0, 1]

-> data/train.csv has 5432400 rows and 8 columns.
Here is some information about the columns:
C (int64) has 3 unique values: [10, 20, 50]
R (int64) has 3 unique values: [5, 50, 20]
breath_id (int64) has range: 1194.00 - 124492.00, 0 nan values
id (int64) has range: 1.00 - 2000.00, 0 nan values
pressure (float64) has range: 3.52 - 41.48, 0 nan values
time_step (float64) has range: 0.00 - 2.73, 0 nan values
u_in (float64) has range: 0.00 - 100.00, 0 nan values
u_out (int64) has 2 unique values: [0, 1]

-> data/ventilator-pressure-prediction/sample_submission.csv has 603600 rows and 2 columns.
Here is some information about the columns:
id (int64) has range: 1.00 - 2000.00, 0 nan values
pressure (int64) has 1 unique values: [0]

-> data/ventilator-pressure-prediction/test.csv has 603600 rows and 7 columns.
Here is some information about the columns:
C (int64) has 3 unique values: [20, 50, 10]
R (int64) has 3 unique values: [50, 5, 20]
breath_id (int64) has range: 2436.00 - 124535.00, 0 nan values
id (int64) has range: 1.00 - 2000.00, 0 nan values
time_step (float64) has range: 0.00 - 2.73, 0 nan values
u_in (float64) has range: 0.00 - 100.00, 0 nan values
u_out (int64) has 2 unique values: [0, 1]

-> data/ventilator-pressure-prediction/train.csv has 5432400 rows and 8 columns.
Here is some information about the columns:
C (int64) has 3 unique values: [10, 20, 50]
R (int64) has 3 unique values: [5, 50, 20]
breath_id (int64) has range: 1194.00 - 124492.00, 0 nan values
id (int64) has range: 1.00 - 2000.00, 0 nan values
pressure (float64) has range: 3.52 - 41.48, 0 nan values
time_step (float64) has range: 0.00 - 2.73, 0 nan values
u_in (float64) has range: 0.00 - 100.00, 0 nan values
u_out (int64) has 2 unique values: [0, 1]

-> input/sample_submission.csv has 603600 rows and 2 columns.
Here is some information about the columns:
id (int64) has range: 1.00 - 2000.00, 0 nan values
pressure (int64) has 1 unique values: [0]

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import sys
import subprocess

try:
    import google.protobuf  # noqa: F401
    from google.protobuf import __version__ as _pb_ver
except Exception:
    _pb_ver = None


def _major(v):
    try:
        return int(str(v).split(".")[0])
    except Exception:
        return None


if _pb_ver is None or (_major(_pb_ver) is not None and _major(_pb_ver) >= 5):
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "protobuf==4.25.3"]
    )
    for m in list(sys.modules):
        if m.startswith("google.protobuf"):
            del sys.modules[m]

import numpy as np
import pandas as pd
import random
import itertools
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Bidirectional, Dense
from sklearn.pipeline import Pipeline

random.seed(7)
np.random.seed(7)
tf.random.set_seed(7)

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames[:50]:
        print(os.path.join(dirname, filename))


## === cell 1
directory = "../input/ventilator-pressure-prediction"
if not os.path.exists(directory):
    directory = "/kaggle/data/ventilator-pressure-prediction"
if not os.path.exists(directory):
    directory = "/kaggle/data"

train = pd.read_csv(os.path.join(directory, "train.csv"))
test = pd.read_csv(os.path.join(directory, "test.csv"))
sub = pd.read_csv(os.path.join(directory, "sample_submission.csv"))

print(train.shape, test.shape, sub.shape)



## === cell 2
_ = train[["R", "C", "time_step", "u_in", "u_out"]].values



## === cell 3
FEATURES = ["R", "C", "time_step", "u_in", "u_out"]
SEQ_LEN = 80


def make_breath_sequences(df, features, target_col=None, seq_len=80):
    df = df.sort_values(["breath_id", "time_step"]).reset_index(drop=True)
    breath_sizes = df.groupby("breath_id").size().values
    if not np.all(breath_sizes == seq_len):
        raise ValueError(
            f"Expected all breaths to have length {seq_len}, got sizes like: {np.unique(breath_sizes)[:10]}"
        )
    x = df[features].to_numpy(dtype=np.float32).reshape(-1, seq_len, len(features))
    if target_col is None:
        return x
    y = df[target_col].to_numpy(dtype=np.float32).reshape(-1, seq_len, 1)
    return x, y


inputs, targets = make_breath_sequences(
    train, FEATURES, target_col="pressure", seq_len=SEQ_LEN
)
print("inputs:", inputs.shape, "targets:", targets.shape)



## === cell 4
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *

try:
    from tensorflow.keras.wrappers.scikit_learn import KerasRegressor  # type: ignore
except ModuleNotFoundError:
    try:
        from scikeras.wrappers import KerasRegressor  # type: ignore
    except Exception:
        KerasRegressor = None

if KerasRegressor is None:
    from sklearn.base import BaseEstimator, RegressorMixin

    class KerasRegressor(BaseEstimator, RegressorMixin):
        def __init__(
            self, build_fn=None, epochs=1, batch_size=32, verbose=0, **fit_kwargs
        ):
            self.build_fn = build_fn
            self.epochs = epochs
            self.batch_size = batch_size
            self.verbose = verbose
            self.fit_kwargs = fit_kwargs
            self.model_ = None

        def fit(self, X, y):
            if self.build_fn is None:
                raise ValueError("build_fn must be provided")
            self.model_ = self.build_fn()
            self.model_.fit(
                X,
                y,
                epochs=self.epochs,
                batch_size=self.batch_size,
                verbose=self.verbose,
                **self.fit_kwargs,
            )
            return self

        def predict(self, X):
            if self.model_ is None:
                raise ValueError("This KerasRegressor instance is not fitted yet.")
            preds = self.model_.predict(X, batch_size=self.batch_size, verbose=0)
            return np.asarray(preds)




## === cell 5
train.shape




## === cell 6
def lstm_model():
    model = Sequential()
    model.add(Input(shape=(inputs.shape[1], inputs.shape[2])))
    model.add(LSTM(320, return_sequences=True))
    model.add(LSTM(320, return_sequences=True))
    model.add(Dense(160, activation="relu"))
    model.add(Dense(1, kernel_initializer="normal"))
    model.compile(
        loss="mae",
        optimizer="adam",
        metrics=[tensorflow.keras.metrics.RootMeanSquaredError()],
    )
    return model




## === cell 7
def bi_lstm_model():
    model = Sequential()
    model.add(Input(shape=(inputs.shape[1], inputs.shape[2])))
    model.add(Bidirectional(LSTM(640, return_sequences=True)))
    model.add(Bidirectional(LSTM(320, return_sequences=True)))
    model.add(LSTM(320, return_sequences=True))
    model.add(LSTM(320, return_sequences=True))
    model.add(Dense(320, activation="relu"))
    model.add(Dense(160, activation="relu"))
    model.add(Dense(1, kernel_initializer="normal"))
    model.compile(
        loss="mae",
        optimizer="adam",
        metrics=[tensorflow.keras.metrics.RootMeanSquaredError()],
    )
    return model




## === cell 8
from sklearn.pipeline import Pipeline as SkPipeline

pipeline = SkPipeline(
    [
        (
            "model",
            KerasRegressor(
                build_fn=bi_lstm_model, epochs=500, batch_size=80, verbose=1
            ),
        ),
    ]
)

pipeline.fit(inputs, targets)


## === cell 9
print("n_breaths_train:", inputs.shape[0], "rows:", inputs.shape[0] * SEQ_LEN)



## === cell 10
test_sorted = test.sort_values(["breath_id", "time_step"]).reset_index(drop=True)
test_inputs = make_breath_sequences(
    test_sorted, FEATURES, target_col=None, seq_len=SEQ_LEN
)
print("test_inputs:", test_inputs.shape)

pred = pipeline.predict(
    test_inputs
)  # expected (n_breaths, 80, 1) or (n_breaths, 80, 1)-like
pred = np.asarray(pred).reshape(
    -1
)  # flatten to per-row predictions in the sorted test order



## === cell 11
pred_by_id = pd.DataFrame({"id": test_sorted["id"].values, "pressure": pred})
pred_by_id = pred_by_id.sort_values("id").reset_index(drop=True)

sub_out = sub[["id"]].merge(pred_by_id, on="id", how="left")
if sub_out["pressure"].isna().any():
    raise ValueError("Missing predictions for some ids; alignment failed.")



## === cell 12
submission_path = "submission.csv"
sub_out.to_csv(submission_path, index=False)
print("Wrote:", submission_path, "shape:", sub_out.shape)
print(sub_out.head())



## === cell 13
sub_out.to_csv("mark_3.csv", index=False)
print("Wrote: mark_3.csv")
