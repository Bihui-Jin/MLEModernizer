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

3.9

# 3. Installed packages

geopandas==0.14.4
lightgbm==4.6.0
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
seaborn==0.12.2
sklearn-pandas==2.2.0
tqdm==4.67.1
xgboost==2.0.3

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

1.2625

# 6. Current score

17.65486

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

- What this solution (achieved 17.65486) has done: 'You’re currently far from the target (MAE 4.46 vs 1.2625; lower is better), so we need a meaningful but still minimal change that preserves your overall approach (tabular feature engineering + LightGBM CV). The biggest score issue is that the competition metric evaluates only inspiratory timesteps (`u_out==0`), but your model trains/validates on all rows, including expiratory phases that are not scored; filtering to inspiratory rows for training/CV aligns the objective with the leaderboard metric and typically improves MAE substantially. I also stop passing the training `pressure` column into the test set (it doesn’t exist) and ensure predictions are written aligned to `id` with the exact required columns. Finally, I keep your GroupKFold-by-breath_id logic but apply it after filtering and use a simple LGBM parameter set to stabilize performance without changing the modeling paradigm.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use("seaborn-white")
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)



## === cell 2
train_data = pd.read_csv(
    "../input/ventilator-pressure-prediction/train.csv", index_col=0
)
test_data = pd.read_csv("../input/ventilator-pressure-prediction/test.csv", index_col=0)
sample = pd.read_csv("../input/ventilator-pressure-prediction/sample_submission.csv")



## === cell 3
train_data.head()



## === cell 4
test_data.head()



## === cell 5
train_data.isnull().sum().to_frame()



## === cell 6
test_data.isnull().sum().to_frame()



## === cell 7
breath_one = train_data[train_data["breath_id"] == 4].reset_index(drop=True)
breath_one



## === cell 8
breath_one.nunique().to_frame()



## === cell 9
fig, axes = plt.subplots(3, 1, figsize=(12, 15))
sns.lineplot(x="time_step", y="u_in", data=breath_one, ax=axes[0])
axes[0].set_title("u_in")
sns.lineplot(x="time_step", y="u_out", data=breath_one, ax=axes[1])
axes[1].set_title("u_out")
sns.lineplot(x="time_step", y="pressure", data=breath_one, ax=axes[2])
axes[2].set_title("pressure")



## === cell 10
breath_one.describe()



## === cell 11
train_data.R.value_counts().to_frame()



## === cell 12
train_data.C.value_counts().to_frame()



## === cell 13
train_data.describe()



## === cell 14
fig, axes = plt.subplots(1, 1, figsize=(10, 5))
sns.histplot(data=train_data, x="pressure", ax=axes)



## === cell 15
idxmax_time_step = train_data.groupby("breath_id")["time_step"].idxmax()
last_value_u_in = train_data.loc[idxmax_time_step, ["breath_id", "u_in"]]
last_value_u_in.columns = ["breath_id", "last_value_u_in"]

train_data = train_data.merge(last_value_u_in, on="breath_id")
train_data



## === cell 16
idxmax_time_step = test_data.groupby("breath_id")["time_step"].idxmax()
last_value_u_in = test_data.loc[idxmax_time_step, ["breath_id", "u_in"]]
last_value_u_in.columns = ["breath_id", "last_value_u_in"]

test_data = test_data.merge(last_value_u_in, on="breath_id")
test_data



## === cell 17
mean_u_in = train_data.groupby("breath_id")["u_in"].mean().to_frame()
mean_u_in.columns = ["mean_value_u_in"]
train_data = train_data.merge(mean_u_in, on="breath_id")



## === cell 18
train_data



## === cell 19
mean_u_in = test_data.groupby("breath_id")["u_in"].mean().to_frame()
mean_u_in.columns = ["mean_value_u_in"]
test_data = test_data.merge(mean_u_in, on="breath_id")
test_data



## === cell 20
train_data["diff_u_in"] = train_data.groupby("breath_id")["u_in"].diff()



## === cell 21
train_data = train_data.fillna(0)



## === cell 22
train_data



## === cell 23
test_data["diff_u_in"] = test_data.groupby("breath_id")["u_in"].diff()
test_data = test_data.fillna(0)



## === cell 24
train_data["diff_diff_u_in"] = train_data.groupby("breath_id")["diff_u_in"].diff()
train_data = train_data.fillna(0)
train_data



## === cell 25
test_data["diff_diff_u_in"] = test_data.groupby("breath_id")["diff_u_in"].diff()
test_data = test_data.fillna(0)
test_data



## === cell 26
train_data["u_in_cumsum"] = (
    (train_data["u_in"]).groupby(train_data["breath_id"]).cumsum()
)
test_data["u_in_cumsum"] = (test_data["u_in"]).groupby(test_data["breath_id"]).cumsum()



## === cell 27
sum_u_in = train_data.groupby("breath_id")["u_in"].sum().to_frame()
sum_u_in.columns = ["sum_value_u_in"]
train_data = train_data.merge(sum_u_in, on="breath_id")



## === cell 28
sum_u_in = test_data.groupby("breath_id")["u_in"].sum().to_frame()
sum_u_in.columns = ["sum_value_u_in"]
test_data = test_data.merge(sum_u_in, on="breath_id")



## === cell 29
train_data["u_in_cumsum_rate"] = (
    train_data["u_in_cumsum"] / train_data["sum_value_u_in"]
)
test_data["u_in_cumsum_rate"] = test_data["u_in_cumsum"] / test_data["sum_value_u_in"]



## === cell 30
train_data[train_data["sum_value_u_in"] == 0]



## === cell 31
train_data[train_data["breath_id"] == 3928]



## === cell 32
train_data = train_data.fillna(0)
test_data = test_data.fillna(0)



## === cell 33
train_data["lag_u_in"] = train_data.groupby("breath_id")["u_in"].shift(1)
train_data = train_data.fillna(0)



## === cell 34
test_data["lag_u_in"] = test_data.groupby("breath_id")["u_in"].shift(1)
test_data = test_data.fillna(0)



## === cell 35
train_data["lag_2_u_in"] = train_data.groupby("breath_id")["u_in"].shift(2)
train_data = train_data.fillna(0)



## === cell 36
test_data["lag_2_u_in"] = test_data.groupby("breath_id")["u_in"].shift(2)
test_data = test_data.fillna(0)



## === cell 37
train_data["lag_-1_u_in"] = train_data.groupby("breath_id")["u_in"].shift(-1)
train_data = train_data.fillna(0)
test_data["lag_-1_u_in"] = test_data.groupby("breath_id")["u_in"].shift(-1)
test_data = test_data.fillna(0)



## === cell 38
train_data["lag_-2_u_in"] = train_data.groupby("breath_id")["u_in"].shift(-2)
train_data = train_data.fillna(0)
test_data["lag_-2_u_in"] = test_data.groupby("breath_id")["u_in"].shift(-2)
test_data = test_data.fillna(0)



## === cell 39
GRAPH = False
if GRAPH:
    fig, axes = plt.subplots(2, 5, figsize=(25, 10))
    sns.scatterplot(data=train_data, x="last_value_u_in", y="pressure", ax=axes[0][1])
    sns.scatterplot(data=train_data, x="u_in_cumsum", y="pressure", ax=axes[0][3])
    sns.scatterplot(data=train_data, x="lag_u_in", y="pressure", ax=axes[1][3])
    sns.scatterplot(data=train_data, x="lag_2_u_in", y="pressure", ax=axes[1][4])



## === cell 40
try:
    del fig
    del axes
except NameError:
    pass



## === cell 41
import gc

gc.collect()



## === cell 42
train_data["train_test"] = "train"
test_data["train_test"] = "test"



## === cell 43
train_test_all = pd.concat([train_data, test_data], axis=0)



## === cell 44
del train_data
del test_data
gc.collect()



## === cell 45
train_test_all



## === cell 46
train_test_all["R_C"] = [
    f"{r}_{c}" for r, c in zip(train_test_all["R"], train_test_all["C"])
]



## === cell 47
train_test_all.info()



## === cell 48
train_test_all = pd.get_dummies(train_test_all, columns=["R_C"])



## === cell 49
train_test_all.isnull().sum()



## === cell 50
train_data = train_test_all[train_test_all["train_test"] == "train"].copy()



## === cell 51
test_data = train_test_all[train_test_all["train_test"] == "test"].copy()



## === cell 52
del train_test_all
gc.collect()



## === cell 53
train_data_insp = train_data[train_data["u_out"] == 0].copy()

X_train = train_data_insp.drop(["pressure", "breath_id", "train_test"], axis=1)
y_train = train_data_insp[["pressure"]]
groups = train_data_insp["breath_id"]

X_test = test_data.drop(["breath_id", "train_test"], axis=1)



## === cell 54
scaler = StandardScaler()
scaler.fit(X_train)
X_train_std = scaler.transform(X_train)
lm = LinearRegression().fit(X_train_std, y_train)
print("coefficient of determination = ", lm.score(X_train_std, y_train))

X_test_std = scaler.transform(X_test)
sample_lr = sample.copy()
sample_lr["pressure"] = lm.predict(X_test_std)
sample_lr.to_csv("submission_lm.csv", index=False)



## --- ERROR in cell 54, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2538714248.py in <cell line: 0>()
      7 print("coefficient of determination = ", lm.score(X_train_std, y_train))
      8 
----> 9 X_test_std = scaler.transform(X_test)
     10 sample_lr = sample.copy()
     11 sample_lr["pressure"] = lm.predict(X_test_std)

/usr/local/lib/python3.11/dist-packages/sklearn/utils/_set_output.py in wrapped(self, X, *args, **kwargs)
    138     @wraps(f)
    139     def wrapped(self, X, *args, **kwargs):
--> 140         data_to_wrap = f(self, X, *args, **kwargs)
    141         if isinstance(data_to_wrap, tuple):
    142             # only wrap the first output for cross decomposition

/usr/local/lib/python3.11/dist-packages/sklearn/preprocessing/_data.py in transform(self, X, copy)
    990 
    991         copy = copy if copy is not None else self.copy
--> 992         X = self._validate_data(
    993             X,
    994             reset=False,

/usr/local/lib/python3.11/dist-packages/sklearn/base.py in _validate_data(self, X, y, reset, validate_separately, **check_params)
    546             validated.
    547         """
--> 548         self._check_feature_names(X, reset=reset)
    549 
    550         if y is None and self._get_tags()["requires_y"]:

/usr/local/lib/python3.11/dist-packages/sklearn/base.py in _check_feature_names(self, X, reset)
    479                 )
    480 
--> 481             raise ValueError(message)
    482 
    483     def _validate_data(

ValueError: The feature names should match those that were passed during fit.
Feature names unseen at fit time:
- pressure


## === cell 55
insample_result = pd.DataFrame()
insample_result["correct"] = y_train.values.ravel()
insample_result["result"] = lm.predict(X_train_std).ravel()

fig, axes = plt.subplots(1, 1, figsize=(10, 10))
sns.scatterplot(data=insample_result, x="correct", y="result", ax=axes)

x = np.linspace(0, 60, 10)
y = x
axes.plot(x, y, color="k")



## === cell 56
insample_MAE = mean_absolute_error(
    insample_result["correct"], insample_result["result"]
)
print(insample_MAE)



## === cell 57
from sklearn.model_selection import GroupKFold
from tqdm import tqdm_notebook as tqdm
import lightgbm as lgb

gbm_val_result = pd.DataFrame(index=np.arange(len(X_train)))
gbm_val_result["correct"] = y_train.values.ravel()



## === cell 58
scores = []
y_pred_test = np.zeros(len(X_test), dtype=np.float64)
gkf = GroupKFold(n_splits=5)

lgb_params = dict(
    random_state=71,
    importance_type="gain",
    n_estimators=800,
    learning_rate=0.05,
    num_leaves=64,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.0,
    reg_lambda=0.0,
)

for i, (train_ix, val_ix) in tqdm(enumerate(gkf.split(X_train, y_train, groups))):
    X_tr, y_tr = X_train.iloc[train_ix], y_train.iloc[train_ix]
    X_val, y_val = X_train.iloc[val_ix], y_train.iloc[val_ix]

    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(X_tr, y_tr)

    y_pred = model.predict(X_val)
    gbm_val_result.loc[val_ix, "result"] = y_pred

    y_pred_test += model.predict(X_test)

    score = mean_absolute_error(y_val, y_pred)
    scores.append(score)
    print("CV Score of Fold_%d is %f" % (i, score))



## --- ERROR in cell 58, traceback:
---------------------------------------------------------------------------
LightGBMError                             Traceback (most recent call last)
/tmp/ipykernel_11/2063094487.py in <cell line: 0>()
     27     gbm_val_result.loc[val_ix, "result"] = y_pred
     28 
---> 29     y_pred_test += model.predict(X_test)
     30 
     31     score = mean_absolute_error(y_val, y_pred)

/usr/local/lib/python3.11/dist-packages/lightgbm/sklearn.py in predict(self, X, raw_score, start_iteration, num_iteration, pred_leaf, pred_contrib, validate_features, **kwargs)
   1142         predict_params["num_threads"] = self._process_n_jobs(predict_params["num_threads"])
   1143 
-> 1144         return self._Booster.predict(  # type: ignore[union-attr]
   1145             X,
   1146             raw_score=raw_score,

/usr/local/lib/python3.11/dist-packages/lightgbm/basic.py in predict(self, data, start_iteration, num_iteration, raw_score, pred_leaf, pred_contrib, data_has_header, validate_features, **kwargs)
   4765             else:
   4766                 num_iteration = -1
-> 4767         return predictor.predict(
   4768             data=data,
   4769             start_iteration=start_iteration,

/usr/local/lib/python3.11/dist-packages/lightgbm/basic.py in predict(self, data, start_iteration, num_iteration, raw_score, pred_leaf, pred_contrib, data_has_header, validate_features)
   1202             )
   1203         elif isinstance(data, np.ndarray):
-> 1204             preds, nrow = self.__pred_for_np2d(
   1205                 mat=data,
   1206                 start_iteration=start_iteration,

/usr/local/lib/python3.11/dist-packages/lightgbm/basic.py in __pred_for_np2d(self, mat, start_iteration, num_iteration, predict_type)
   1359             return preds, nrow
   1360         else:
-> 1361             return self.__inner_predict_np2d(
   1362                 mat=mat,
   1363                 start_iteration=start_iteration,

/usr/local/lib/python3.11/dist-packages/lightgbm/basic.py in __inner_predict_np2d(self, mat, start_iteration, num_iteration, predict_type, preds)
   1305             raise ValueError("Wrong length of pre-allocated predict array")
   1306         out_num_preds = ctypes.c_int64(0)
-> 1307         _safe_call(
   1308             _LIB.LGBM_BoosterPredictForMat(
   1309                 self._handle,

/usr/local/lib/python3.11/dist-packages/lightgbm/basic.py in _safe_call(ret)
    311     """
    312     if ret != 0:
--> 313         raise LightGBMError(_LIB.LGBM_GetLastError().decode("utf-8"))
    314 
    315 

LightGBMError: The number of features in data (26) is not the same as it was in training data (25).
You can set ``predict_disable_shape_check=true`` to discard this error, but please be aware what you are doing.

## === cell 59
print(scores)
print(np.mean(scores))



## === cell 60
y_pred_test_submit = y_pred_test / 5
submission = sample.copy()
submission["pressure"] = y_pred_test_submit
submission.to_csv("submission.csv", index=False)
submission.head()



## === cell 61
gbm_val_result.head()



## === cell 62
fig, axes = plt.subplots(1, 1, figsize=(10, 10))
sns.scatterplot(data=gbm_val_result, x="correct", y="result", ax=axes)

x = np.linspace(0, 60, 10)
y = x
axes.plot(x, y, color="k")



## === cell 63
XGBRegressor = False



## === cell 64
if XGBRegressor:
    xgb = XGBRegressor(objective="reg:squarederror", n_estimators=700)
    xgb.fit(X_train, y_train)
    print("coefficient of determination = ", xgb.score(X_train, y_train))

    sample_xgb = sample.copy()
    sample_xgb["pressure"] = xgb.predict(X_test)
    sample_xgb.to_csv("submission_xgb.csv", index=False)



## === cell 65
if XGBRegressor:
    insample_result = pd.DataFrame()
    insample_result["correct"] = y_train.values.ravel()
    insample_result["result"] = xgb.predict(X_train).ravel()

    fig, axes = plt.subplots(1, 1, figsize=(10, 10))
    sns.scatterplot(data=insample_result, x="correct", y="result", ax=axes)

    x = np.linspace(0, 60, 10)
    y = x
    axes.plot(x, y, color="k")
