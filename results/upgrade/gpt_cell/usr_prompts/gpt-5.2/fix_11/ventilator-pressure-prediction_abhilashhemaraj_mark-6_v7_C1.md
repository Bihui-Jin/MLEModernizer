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

# 5. Target score

1.9106

# 6. Current score

None

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
import os

import sys
import subprocess


def _ensure_compatible_protobuf():
    try:
        import google.protobuf  # noqa: F401
        from google.protobuf import __version__ as pb_version

        major = int(pb_version.split(".", 1)[0])
        if major >= 5:
            raise RuntimeError(f"Incompatible protobuf version detected: {pb_version}")
    except Exception:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "protobuf>=4.21.0,<5"]
        )
        for m in list(sys.modules.keys()):
            if m.startswith("google.protobuf"):
                del sys.modules[m]


_ensure_compatible_protobuf()

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

import pandas as pd
from tqdm.auto import tqdm

tqdm.pandas()
import matplotlib.pyplot as plt

get_ipython().run_line_magic("matplotlib", "inline")
import seaborn as sns
import numpy as np
import time
import itertools
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import *
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingRegressor

import random

random.seed(7)


## === cell 2
directory = '../input/ventilator-pressure-prediction'
train = pd.read_csv(os.path.join(directory, 'train.csv'))
test = pd.read_csv(os.path.join(directory, 'test.csv'))
sub = pd.read_csv(os.path.join(directory, 'sample_submission.csv'))


## === cell 3
train[['R', 'C', 'time_step', 'u_in', 'u_out']].values


## === cell 4
X = train[['R', 'C', 'time_step', 'u_in', 'u_out']].values
Y = train.pressure.values

sample_length = 80
input_dataset = tf.keras.preprocessing.timeseries_dataset_from_array(
  X, None, sequence_length=sample_length, sequence_stride=sample_length)
target_dataset = tf.keras.preprocessing.timeseries_dataset_from_array(
  Y, None, sequence_length=sample_length, sequence_stride=sample_length)

for batch in zip(input_dataset, target_dataset):
  inputs, targets = batch
  assert np.array_equal(inputs[0], X[:sample_length])

  assert np.array_equal(targets[1], Y[sample_length:2*sample_length])
  break


## === cell 5
import time
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *

try:
    from tensorflow.keras.wrappers.scikit_learn import KerasRegressor  # type: ignore
except ModuleNotFoundError:
    try:
        from scikeras.wrappers import KerasRegressor  # type: ignore
    except Exception:
        KerasRegressor = (
            None  # Allows notebook to run if the wrapper is not used later.
        )

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import *
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingRegressor


## === cell 6
train.shape


## === cell 7
def lstm_model():
    model = Sequential()
    model.add(Input(shape = (inputs.shape[1], inputs.shape[2])))
    model.add(LSTM(320, return_sequences=True))
    model.add(LSTM(320, return_sequences=True))
    model.add(Dense(160, activation = 'relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mae', optimizer='adam', metrics = [tensorflow.keras.metrics.RootMeanSquaredError()])
    return model


## === cell 8
def bi_lstm_model():
    model = Sequential()
    model.add(Input(shape = (inputs.shape[1], inputs.shape[2])))
    model.add(Bidirectional(LSTM(640, return_sequences=True)))
    model.add(Bidirectional(LSTM(320, return_sequences=True)))
    model.add(LSTM(320, return_sequences=True))
    model.add(LSTM(320, return_sequences=True))
    model.add(Dense(320, activation = 'relu'))
    model.add(Dense(160, activation = 'relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mae', optimizer='adam', metrics = [tensorflow.keras.metrics.RootMeanSquaredError()])
    return model


## === cell 9
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


pipeline = Pipeline(
    steps=[
        (
            "model",
            KerasRegressor(
                build_fn=bi_lstm_model, epochs=500, batch_size=80, verbose=1
            ),
        )
    ]
)

pipeline.fit(inputs, targets)


## === cell 10
4024000/80


## === cell 11
test_array = test[["R", "C", "time_step", "u_in", "u_out"]].to_numpy()
n_rows = test_array.shape[0]
seq_len = 80
if n_rows % seq_len != 0:
    raise ValueError(
        f"test rows ({n_rows}) must be divisible by sequence_length ({seq_len}) to reshape."
    )
n_seq = n_rows // seq_len

sub_array = pipeline.predict(test_array.reshape(n_seq, seq_len, 5))


## === cell 12
sub_array =list(itertools.chain(*sub_array))


## === cell 14
pd.DataFrame({'id':test.id, 'pressure':sub_array}).to_csv('mark_3.csv', index=False)
