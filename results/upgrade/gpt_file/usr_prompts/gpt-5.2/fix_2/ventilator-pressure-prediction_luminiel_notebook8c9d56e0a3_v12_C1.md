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

3.10

# 3. Installed packages

No external packages required in the script and installed.

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

1.391755329978957

# 6. Current score

17.65486

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

- What this solution (achieved 17.65486) has done: 'Your notebook currently fails to yield a score because it errors at `mean` (undefined) after writing `submission.csv`, so I remove that breaking code path and make submission-writing unconditional at the end. To move the MAE down toward your target (lower is better) with minimal change and identical core modeling logic, I also fix two feature bugs that silently hurt generalization: your lag features were leaking across breaths (global `shift`), and one-hot columns for `R`/`C` can mismatch between train/test if created separately. Finally, I generate predictions from the existing PyTorch LSTM via a simple fold-ensemble (average of the already-trained fold models) and write them into the required `id,pressure` format.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
import seaborn as sns
import matplotlib.pyplot as plt



## === cell 2
train = pd.read_csv("/kaggle/input/ventilator-pressure-prediction/train.csv")



## === cell 3
train.head()



## === cell 4
train.nunique().to_frame()



## === cell 5
first = train[train["breath_id"] == 1]
second = train[train["breath_id"] == 2]



## === cell 6
first.plot(x="time_step", y="u_in", kind="line", figsize=(12, 3))
second.plot(x="time_step", y="u_in", kind="line", figsize=(12, 3))
first.plot(x="time_step", y="u_out", kind="line", figsize=(12, 3))
second.plot(x="time_step", y="u_out", kind="line", figsize=(12, 3))
first.plot(x="time_step", y="pressure", kind="line", figsize=(12, 3))
second.plot(x="time_step", y="pressure", kind="line", figsize=(12, 3))



## === cell 7
train["u_in_cumsum"] = (train["u_in"]).groupby(train["breath_id"]).cumsum()
for df in [train]:
    df["u_in_first"] = df.groupby("breath_id")["u_in"].transform("first")
    df["u_in_min"] = df.groupby("breath_id")["u_in"].transform("min")
    df["u_in_mean"] = df.groupby("breath_id")["u_in"].transform("mean")
    df["u_in_median"] = df.groupby("breath_id")["u_in"].transform("median")
    df["u_in_max"] = df.groupby("breath_id")["u_in"].transform("max")
    df["u_in_last"] = df.groupby("breath_id")["u_in"].transform("last")
    df["area"] = df["time_step"] * df["u_in"]
    df["area"] = df.groupby("breath_id")["area"].cumsum()

    df["u_in_lag2"] = df.groupby("breath_id")["u_in"].shift(2).fillna(0)
    df["u_in_lag4"] = df.groupby("breath_id")["u_in"].shift(4).fillna(0)

    g = df.groupby("breath_id")["u_in"]
    df["ewm_u_in_mean"] = g.ewm(halflife=10).mean().reset_index(level=0, drop=True)
    df["ewm_u_in_std"] = g.ewm(halflife=10).std().reset_index(level=0, drop=True)
    df["ewm_u_in_corr"] = g.ewm(halflife=10).corr().reset_index(level=0, drop=True)

    df["rolling_10_mean"] = (
        g.rolling(window=10, min_periods=1).mean().reset_index(level=0, drop=True)
    )
    df["rolling_10_max"] = (
        g.rolling(window=10, min_periods=1).max().reset_index(level=0, drop=True)
    )
    df["rolling_10_std"] = (
        g.rolling(window=10, min_periods=1).std().reset_index(level=0, drop=True)
    )
    df.fillna(0, inplace=True)



## === cell 8
train



## === cell 9
from sklearn.experimental import enable_hist_gradient_boosting  # noqa: F401
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error



## === cell 10
X_train = train.drop(["pressure", "id", "breath_id"], axis=1)
y_train = train["pressure"]



## === cell 11
X_train



## === cell 12
regressor = HistGradientBoostingRegressor(
    max_iter=100, loss="least_absolute_deviation", early_stopping=False
)
regressor.fit(X_train, y_train)



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
InvalidParameterError                     Traceback (most recent call last)
/tmp/ipykernel_55/2124222077.py in <cell line: 0>()
      2     max_iter=100, loss="least_absolute_deviation", early_stopping=False
      3 )
----> 4 regressor.fit(X_train, y_train)
      5 

/usr/local/lib/python3.11/dist-packages/sklearn/ensemble/_hist_gradient_boosting/gradient_boosting.py in fit(self, X, y, sample_weight)
    351             Fitted estimator.
    352         """
--> 353         self._validate_params()
    354 
    355         fit_start_time = time()

/usr/local/lib/python3.11/dist-packages/sklearn/base.py in _validate_params(self)
    598         accepted constraints.
    599         """
--> 600         validate_parameter_constraints(
    601             self._parameter_constraints,
    602             self.get_params(deep=False),

/usr/local/lib/python3.11/dist-packages/sklearn/utils/_param_validation.py in validate_parameter_constraints(parameter_constraints, params, caller_name)
     95                 )
     96 
---> 97             raise InvalidParameterError(
     98                 f"The {param_name!r} parameter of {caller_name} must be"
     99                 f" {constraints_str}. Got {param_val!r} instead."

InvalidParameterError: The 'loss' parameter of HistGradientBoostingRegressor must be a str among {'quantile', 'poisson', 'squared_error', 'absolute_error'} or an instance of 'sklearn._loss.loss.BaseLoss'. Got 'least_absolute_deviation' instead.

## === cell 13
y_pred = regressor.predict(X_train)



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NotFittedError                            Traceback (most recent call last)
/tmp/ipykernel_55/534512031.py in <cell line: 0>()
----> 1 y_pred = regressor.predict(X_train)
      2 

/usr/local/lib/python3.11/dist-packages/sklearn/ensemble/_hist_gradient_boosting/gradient_boosting.py in predict(self, X)
   1482             The predicted values.
   1483         """
-> 1484         check_is_fitted(self)
   1485         # Return inverse link of raw predictions after converting
   1486         # shape (n_samples, 1) to (n_samples,)

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in check_is_fitted(estimator, attributes, msg, all_or_any)
   1388 
   1389     if not fitted:
-> 1390         raise NotFittedError(msg % {"name": type(estimator).__name__})
   1391 
   1392 

NotFittedError: This HistGradientBoostingRegressor instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.

## === cell 14
print("MAE =", mean_absolute_error(y_pred, y_train))



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1801806736.py in <cell line: 0>()
----> 1 print("MAE =", mean_absolute_error(y_pred, y_train))
      2 

NameError: name 'y_pred' is not defined

## === cell 15
test = pd.read_csv("/kaggle/input/ventilator-pressure-prediction/test.csv")
sub = pd.read_csv("/kaggle/input/ventilator-pressure-prediction/sample_submission.csv")



## === cell 16
test["u_in_cumsum"] = (test["u_in"]).groupby(test["breath_id"]).cumsum()
for df in [test]:
    df["u_in_first"] = df.groupby("breath_id")["u_in"].transform("first")
    df["u_in_min"] = df.groupby("breath_id")["u_in"].transform("min")
    df["u_in_mean"] = df.groupby("breath_id")["u_in"].transform("mean")
    df["u_in_median"] = df.groupby("breath_id")["u_in"].transform("median")
    df["u_in_max"] = df.groupby("breath_id")["u_in"].transform("max")
    df["u_in_last"] = df.groupby("breath_id")["u_in"].transform("last")
    df["area"] = df["time_step"] * df["u_in"]
    df["area"] = df.groupby("breath_id")["area"].cumsum()

    df["u_in_lag2"] = df.groupby("breath_id")["u_in"].shift(2).fillna(0)
    df["u_in_lag4"] = df.groupby("breath_id")["u_in"].shift(4).fillna(0)

    g = df.groupby("breath_id")["u_in"]
    df["ewm_u_in_mean"] = g.ewm(halflife=10).mean().reset_index(level=0, drop=True)
    df["ewm_u_in_std"] = g.ewm(halflife=10).std().reset_index(level=0, drop=True)
    df["ewm_u_in_corr"] = g.ewm(halflife=10).corr().reset_index(level=0, drop=True)

    df["rolling_10_mean"] = (
        g.rolling(window=10, min_periods=1).mean().reset_index(level=0, drop=True)
    )
    df["rolling_10_max"] = (
        g.rolling(window=10, min_periods=1).max().reset_index(level=0, drop=True)
    )
    df["rolling_10_std"] = (
        g.rolling(window=10, min_periods=1).std().reset_index(level=0, drop=True)
    )
    df.fillna(0, inplace=True)



## === cell 17
X_test = test.drop(["id", "breath_id"], axis=1)



## === cell 18
X_test



## === cell 19
sub["pressure"] = regressor.predict(X_test)



## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NotFittedError                            Traceback (most recent call last)
/tmp/ipykernel_55/1039524368.py in <cell line: 0>()
----> 1 sub["pressure"] = regressor.predict(X_test)
      2 

/usr/local/lib/python3.11/dist-packages/sklearn/ensemble/_hist_gradient_boosting/gradient_boosting.py in predict(self, X)
   1482             The predicted values.
   1483         """
-> 1484         check_is_fitted(self)
   1485         # Return inverse link of raw predictions after converting
   1486         # shape (n_samples, 1) to (n_samples,)

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in check_is_fitted(estimator, attributes, msg, all_or_any)
   1388 
   1389     if not fitted:
-> 1390         raise NotFittedError(msg % {"name": type(estimator).__name__})
   1391 
   1392 

NotFittedError: This HistGradientBoostingRegressor instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.

## === cell 20
sub.to_csv("submission_hgb.csv", index=False)



## === cell 21
print("HistGradientBoosting baseline submission written to submission_hgb.csv")



## === cell 22
import numpy as np
import pandas as pd
import math
import time
import pickle
import argparse
import sklearn.preprocessing
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import KFold

debug = False


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)


set_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device



## === cell 23
train = pd.read_csv("/kaggle/input/ventilator-pressure-prediction/train.csv")
test = pd.read_csv("/kaggle/input/ventilator-pressure-prediction/test.csv")
sub = pd.read_csv("/kaggle/input/ventilator-pressure-prediction/sample_submission.csv")




## === cell 24
def create_features(df, all_dummies_columns=None):
    df = df.copy()

    df["area"] = df["time_step"] * df["u_in"]
    df["area"] = df.groupby("breath_id")["area"].cumsum()

    df["u_in_cumsum"] = df["u_in"].groupby(df["breath_id"]).cumsum()

    df["u_in_lag2"] = df.groupby("breath_id")["u_in"].shift(2).fillna(0)
    df["u_in_lag4"] = df.groupby("breath_id")["u_in"].shift(4).fillna(0)

    df["R"] = df["R"].astype(str)
    df["C"] = df["C"].astype(str)
    df = pd.get_dummies(df)

    if all_dummies_columns is not None:
        df = df.reindex(columns=all_dummies_columns, fill_value=0)

    g = df.groupby("breath_id")["u_in"]
    df["ewm_u_in_mean"] = g.ewm(halflife=10).mean().reset_index(level=0, drop=True)
    df["ewm_u_in_std"] = g.ewm(halflife=10).std().reset_index(level=0, drop=True)
    df["ewm_u_in_corr"] = g.ewm(halflife=10).corr().reset_index(level=0, drop=True)

    df["rolling_10_mean"] = (
        g.rolling(window=10, min_periods=1).mean().reset_index(level=0, drop=True)
    )
    df["rolling_10_max"] = (
        g.rolling(window=10, min_periods=1).max().reset_index(level=0, drop=True)
    )
    df["rolling_10_std"] = (
        g.rolling(window=10, min_periods=1).std().reset_index(level=0, drop=True)
    )

    df = df.fillna(0)

    for col in ["id", "breath_id", "pressure"]:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)

    return df


class Dataset(torch.utils.data.Dataset):
    def __init__(self, X, y, w):
        if y is None:
            y = np.zeros(len(X), dtype=np.float32)
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)
        self.w = w.astype(np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i], self.w[i]




## === cell 25
combo = pd.concat([train.drop(columns=["pressure"]), test], axis=0, ignore_index=True)
combo_feat = create_features(combo)
all_cols = combo_feat.columns

train_feat = create_features(train, all_dummies_columns=all_cols)
test_feat = create_features(test, all_dummies_columns=all_cols)

rs = sklearn.preprocessing.RobustScaler()
train_feat_scaled = rs.fit_transform(train_feat)
test_feat_scaled = rs.transform(test_feat)

X_all = train_feat_scaled.reshape(-1, 80, train_feat_scaled.shape[-1])
y_all = train.pressure.values.reshape(-1, 80)
w_all = 1 - train.u_out.values.reshape(-1, 80)  # kept (not used in loss)

X_test_all = test_feat_scaled.reshape(-1, 80, test_feat_scaled.shape[-1])

input_size = X_all.shape[2]
print(
    "n_breaths_train:",
    len(X_all),
    "n_breaths_test:",
    len(X_test_all),
    "input_size:",
    input_size,
)




## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_55/1339371008.py in <cell line: 0>()
      4 all_cols = combo_feat.columns
      5 
----> 6 train_feat = create_features(train, all_dummies_columns=all_cols)
      7 test_feat = create_features(test, all_dummies_columns=all_cols)
      8 

/tmp/ipykernel_55/3740871567.py in create_features(df, all_dummies_columns)
     20         df = df.reindex(columns=all_dummies_columns, fill_value=0)
     21 
---> 22     g = df.groupby("breath_id")["u_in"]
     23     df["ewm_u_in_mean"] = g.ewm(halflife=10).mean().reset_index(level=0, drop=True)
     24     df["ewm_u_in_std"] = g.ewm(halflife=10).std().reset_index(level=0, drop=True)

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in groupby(self, by, axis, level, as_index, sort, group_keys, observed, dropna)
   9181             raise TypeError("You have to supply one of 'by' and 'level'")
   9182 
-> 9183         return DataFrameGroupBy(
   9184             obj=self,
   9185             keys=by,

/usr/local/lib/python3.11/dist-packages/pandas/core/groupby/groupby.py in __init__(self, obj, keys, axis, level, grouper, exclusions, selection, as_index, sort, group_keys, observed, dropna)
   1327 
   1328         if grouper is None:
-> 1329             grouper, exclusions, obj = get_grouper(
   1330                 obj,
   1331                 keys,

/usr/local/lib/python3.11/dist-packages/pandas/core/groupby/grouper.py in get_grouper(obj, key, axis, level, sort, observed, validate, dropna)
   1041                 in_axis, level, gpr = False, gpr, None
   1042             else:
-> 1043                 raise KeyError(gpr)
   1044         elif isinstance(gpr, Grouper) and gpr.key is not None:
   1045             # Add key to exclusions

KeyError: 'breath_id'

## === cell 26
class Model(nn.Module):
    def __init__(self, input_size):
        hidden = [400, 300, 200, 100]
        super().__init__()
        self.lstm1 = nn.LSTM(
            input_size, hidden[0], batch_first=True, bidirectional=True
        )
        self.lstm2 = nn.LSTM(
            2 * hidden[0], hidden[1], batch_first=True, bidirectional=True
        )
        self.lstm3 = nn.LSTM(
            2 * hidden[1], hidden[2], batch_first=True, bidirectional=True
        )
        self.lstm4 = nn.LSTM(
            2 * hidden[2], hidden[3], batch_first=True, bidirectional=True
        )
        self.fc1 = nn.Linear(2 * hidden[3], 50)
        self.selu = nn.SELU()
        self.fc2 = nn.Linear(50, 1)
        self._reinitialize()

    def _reinitialize(self):
        for name, p in self.named_parameters():
            if "lstm" in name:
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(p.data)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(p.data)
                elif "bias_ih" in name:
                    p.data.fill_(0)
                    n = p.size(0)
                    p.data[(n // 4) : (n // 2)].fill_(1)
                elif "bias_hh" in name:
                    p.data.fill_(0)
            elif "fc" in name:
                if "weight" in name:
                    nn.init.xavier_uniform_(p.data)
                elif "bias" in name:
                    p.data.fill_(0)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x, _ = self.lstm3(x)
        x, _ = self.lstm4(x)
        x = self.fc1(x)
        x = self.selu(x)
        x = self.fc2(x)
        return x




## === cell 27
model = Model(input_size)
for name, p in model.named_parameters():
    print("%-32s %s" % (name, tuple(p.shape)))



## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2930058235.py in <cell line: 0>()
----> 1 model = Model(input_size)
      2 for name, p in model.named_parameters():
      3     print("%-32s %s" % (name, tuple(p.shape)))
      4 

NameError: name 'input_size' is not defined

## === cell 28
criterion = torch.nn.L1Loss()


def evaluate(model, loader_val):
    tb = time.time()
    was_training = model.training
    model.eval()

    loss_sum = 0
    n_sum = 0
    y_pred_all = []

    for ibatch, (x, y, w) in enumerate(loader_val):
        n = y.size(0)
        x = x.to(device)
        y = y.to(device)

        with torch.no_grad():
            y_pred = model(x).squeeze()

        loss = criterion(y_pred, y)
        n_sum += n
        loss_sum += n * loss.item()
        y_pred_all.append(y_pred.cpu().detach().numpy())

    loss_val = loss_sum / n_sum
    model.train(was_training)

    d = {
        "loss": loss_val,
        "time": time.time() - tb,
        "y_pred": np.concatenate(y_pred_all, axis=0),
    }
    return d




## === cell 29
nfold = 2
kfold = KFold(n_splits=nfold, shuffle=True, random_state=228)
epochs = 15
lr = 1e-3
batch_size = 1024
max_grad_norm = 1000
log = {}

for ifold, (idx_train, idx_val) in enumerate(kfold.split(X_all)):
    print("Fold %d" % ifold)
    tb = time.time()
    model = Model(input_size)
    model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=10)

    X_train_f = X_all[idx_train]
    y_train_f = y_all[idx_train]
    w_train_f = w_all[idx_train]
    X_val_f = X_all[idx_val]
    y_val_f = y_all[idx_val]
    w_val_f = w_all[idx_val]

    dataset_train = Dataset(X_train_f, y_train_f, w_train_f)
    dataset_val = Dataset(X_val_f, y_val_f, w_val_f)
    loader_train = torch.utils.data.DataLoader(
        dataset_train, shuffle=True, batch_size=batch_size, drop_last=True
    )
    loader_val = torch.utils.data.DataLoader(
        dataset_val, shuffle=False, batch_size=batch_size, drop_last=False
    )

    losses_train = []
    losses_val = []
    lrs = []
    time_val = 0

    print("epoch loss_train loss_val lr time")
    for iepoch in range(epochs):
        loss_train = 0
        n_sum = 0

        for ibatch, (x, y, w) in enumerate(loader_train):
            n = y.size(0)
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            y_pred = model(x).squeeze()
            loss = criterion(y_pred, y)

            loss_train += n * loss.item()
            n_sum += n

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        val = evaluate(model, loader_val)
        loss_val = val["loss"]
        time_val += val["time"]

        losses_train.append(loss_train / n_sum)
        losses_val.append(loss_val)
        lrs.append(optimizer.param_groups[0]["lr"])

        print(
            "%3d %9.6f %9.6f %7.3e %7.1f %6.1f"
            % (
                iepoch + 1,
                losses_train[-1],
                losses_val[-1],
                lrs[-1],
                time.time() - tb,
                time_val,
            )
        )

        scheduler.step(losses_val[-1])

    ofilename = "model%d.pth" % ifold
    torch.save(model.state_dict(), ofilename)
    print(ofilename, "written")

    log["fold%d" % ifold] = {
        "loss_train": np.array(losses_train),
        "loss_val": np.array(losses_val),
        "learning_rate": np.array(lrs),
        "y_pred": val["y_pred"],
        "idx": idx_val,
    }

    if ifold >= 1:  # keep original time-limit safeguard
        break




## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1700104812.py in <cell line: 0>()
      8 log = {}
      9 
---> 10 for ifold, (idx_train, idx_val) in enumerate(kfold.split(X_all)):
     11     print("Fold %d" % ifold)
     12     tb = time.time()

NameError: name 'X_all' is not defined

## === cell 30
def predict_test_with_saved_models(
    X_test_breaths, input_size, nfold=2, batch_size=1024
):
    dataset_test = Dataset(
        X_test_breaths, None, np.zeros((len(X_test_breaths), 80), dtype=np.float32)
    )
    loader_test = torch.utils.data.DataLoader(
        dataset_test, shuffle=False, batch_size=batch_size, drop_last=False
    )

    preds_folds = []
    for ifold in range(nfold):
        path = f"model{ifold}.pth"
        if not os.path.exists(path):
            continue
        m = Model(input_size).to(device)
        m.load_state_dict(torch.load(path, map_location=device))
        m.eval()

        pred_chunks = []
        with torch.no_grad():
            for x, y, w in loader_test:
                x = x.to(device)
                p = m(x).squeeze(-1)  # (batch, 80)
                pred_chunks.append(p.detach().cpu().numpy())
        preds_folds.append(np.concatenate(pred_chunks, axis=0))

    if len(preds_folds) == 0:
        raise RuntimeError("No saved fold models found; cannot create submission.")

    pred_mean = np.mean(preds_folds, axis=0)  # (n_breaths_test, 80)
    return pred_mean.reshape(-1)


test_pred = predict_test_with_saved_models(
    X_test_all, input_size, nfold=nfold, batch_size=batch_size
)
sub_out = sub.copy()
sub_out["pressure"] = test_pred.astype(np.float32)

sub_out = sub_out[["id", "pressure"]]
assert len(sub_out) == len(test), "Submission length must match test rows."

sub_out.to_csv("submission.csv", index=False)
print("Final submission written to submission.csv with rows:", len(sub_out))
print(sub_out.head())



## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3386402120.py in <cell line: 0>()
     35 
     36 test_pred = predict_test_with_saved_models(
---> 37     X_test_all, input_size, nfold=nfold, batch_size=batch_size
     38 )
     39 sub_out = sub.copy()

NameError: name 'X_test_all' is not defined

## === cell 31
print("done")
