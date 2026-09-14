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

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 17.65486) has done: 'You’re currently far from the target (MAE 4.46 vs 1.2625; lower is better), so we need a meaningful but still minimal change that preserves your overall approach (tabular feature engineering + LightGBM CV). The biggest score issue is that the competition metric evaluates only inspiratory timesteps (`u_out==0`), but your model trains/validates on all rows, including expiratory phases that are not scored; filtering to inspiratory rows for training/CV aligns the objective with the leaderboard metric and typically improves MAE substantially. I also stop passing the training `pressure` column into the test set (it doesn’t exist) and ensure predictions are written aligned to `id` with the exact required columns. Finally, I keep your GroupKFold-by-breath_id logic but apply it after filtering and use a simple LGBM parameter set to stabilize performance without changing the modeling paradigm.'
- What this solution (achieved 3.55632) has done: 'I fix the feature-mismatch bugs by ensuring train and test use the exact same feature columns (dropping `pressure` from the test split and aligning columns before scaling/prediction). I also make the inspiratory-only training consistent end-to-end by using the same filtered training frame for both the linear model and LightGBM CV while keeping your existing feature engineering and GroupKFold-by-breath_id logic. Finally, I make the submission creation robust by sorting by `id` and writing a valid `submission.csv` with exactly `id,pressure`. These changes are minimal but should materially reduce MAE because they remove runtime errors and align training with the scored phase.'

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
test_u_out_by_id = test_data[["id", "u_out"]].copy()

train_data_insp = train_data[train_data["u_out"] == 0].copy()

X_train = train_data_insp.drop(["pressure", "breath_id", "train_test"], axis=1)
y_train = train_data_insp[["pressure"]]
groups = train_data_insp["breath_id"]

X_test = test_data.drop(["breath_id", "train_test"], axis=1, errors="ignore")
if "pressure" in X_test.columns:
    X_test = X_test.drop(["pressure"], axis=1)

X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

assert list(X_train.columns) == list(X_test.columns)
assert X_train.shape[1] == X_test.shape[1]



## --- ERROR in cell 53, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/56715978.py in <cell line: 0>()
      1 # Change (score-relevant): store test u_out mask by id so we can enforce "expiratory not scored"
      2 # behavior at inference without changing training logic (training stays inspiratory-only).
----> 3 test_u_out_by_id = test_data[["id", "u_out"]].copy()
      4 
      5 train_data_insp = train_data[train_data["u_out"] == 0].copy()

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4106             if is_iterator(key):
   4107                 key = list(key)
-> 4108             indexer = self.columns._get_indexer_strict(key, "columns")[1]
   4109 
   4110         # take() does not accept boolean indexers

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _get_indexer_strict(self, key, axis_name)
   6198             keyarr, indexer, new_indexer = self._reindex_non_unique(keyarr)
   6199 
-> 6200         self._raise_if_missing(keyarr, indexer, axis_name)
   6201 
   6202         keyarr = self.take(indexer)

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _raise_if_missing(self, key, indexer, axis_name)
   6250 
   6251             not_found = list(ensure_index(key)[missing_mask.nonzero()[0]].unique())
-> 6252             raise KeyError(f"{not_found} not in index")
   6253 
   6254     @overload

KeyError: "['id'] not in index"

## === cell 54
scaler = StandardScaler()
scaler.fit(X_train)
X_train_std = scaler.transform(X_train)
lm = LinearRegression().fit(X_train_std, y_train)
print("coefficient of determination = ", lm.score(X_train_std, y_train))

X_test_std = scaler.transform(X_test)
sample_lr = sample.copy()
sample_lr["pressure"] = lm.predict(X_test_std)
sample_lr = sample_lr.sort_values("id")
sample_lr.to_csv("submission_lm.csv", index=False)



## --- ERROR in cell 54, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3389211057.py in <cell line: 0>()
      1 scaler = StandardScaler()
----> 2 scaler.fit(X_train)
      3 X_train_std = scaler.transform(X_train)
      4 lm = LinearRegression().fit(X_train_std, y_train)
      5 print("coefficient of determination = ", lm.score(X_train_std, y_train))

NameError: name 'X_train' is not defined

## === cell 55
insample_result = pd.DataFrame()
insample_result["correct"] = y_train.values.ravel()
insample_result["result"] = lm.predict(X_train_std).ravel()

fig, axes = plt.subplots(1, 1, figsize=(10, 10))
sns.scatterplot(data=insample_result, x="correct", y="result", ax=axes)

x = np.linspace(0, 60, 10)
y = x
axes.plot(x, y, color="k")



## --- ERROR in cell 55, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1729177926.py in <cell line: 0>()
      1 insample_result = pd.DataFrame()
----> 2 insample_result["correct"] = y_train.values.ravel()
      3 insample_result["result"] = lm.predict(X_train_std).ravel()
      4 
      5 fig, axes = plt.subplots(1, 1, figsize=(10, 10))

NameError: name 'y_train' is not defined

## === cell 56
insample_MAE = mean_absolute_error(
    insample_result["correct"], insample_result["result"]
)
print(insample_MAE)



## --- ERROR in cell 56, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/808643562.py in <cell line: 0>()
      1 insample_MAE = mean_absolute_error(
----> 2     insample_result["correct"], insample_result["result"]
      3 )
      4 print(insample_MAE)
      5 

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4100             if self.columns.nlevels > 1:
   4101                 return self._getitem_multilevel(key)
-> 4102             indexer = self.columns.get_loc(key)
   4103             if is_integer(indexer):
   4104                 indexer = [indexer]

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/range.py in get_loc(self, key)
    415                 raise KeyError(key) from err
    416         if isinstance(key, Hashable):
--> 417             raise KeyError(key)
    418         self._check_indexing_error(key)
    419         raise KeyError(key)

KeyError: 'correct'

## === cell 57
from sklearn.model_selection import GroupKFold
from tqdm import tqdm_notebook as tqdm
import lightgbm as lgb

gbm_val_result = pd.DataFrame(index=np.arange(len(X_train)))
gbm_val_result["correct"] = y_train.values.ravel()



## --- ERROR in cell 57, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3560147849.py in <cell line: 0>()
      3 import lightgbm as lgb
      4 
----> 5 gbm_val_result = pd.DataFrame(index=np.arange(len(X_train)))
      6 gbm_val_result["correct"] = y_train.values.ravel()
      7 

NameError: name 'X_train' is not defined

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
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1313716257.py in <cell line: 0>()
      1 scores = []
----> 2 y_pred_test = np.zeros(len(X_test), dtype=np.float64)
      3 gkf = GroupKFold(n_splits=5)
      4 
      5 lgb_params = dict(

NameError: name 'X_test' is not defined

## === cell 59
print(scores)
print(np.mean(scores))



## === cell 60
pressure_grid = np.sort(train_data["pressure"].unique())


def snap_to_grid(pred, grid):
    pred = np.asarray(pred, dtype=np.float64)
    idx = np.searchsorted(grid, pred, side="left")
    idx = np.clip(idx, 1, len(grid) - 1)
    left = grid[idx - 1]
    right = grid[idx]
    choose_right = (pred - left) > (right - pred)
    return np.where(choose_right, right, left)




## === cell 61
y_pred_test_submit = y_pred_test / 5

sub = pd.DataFrame({"id": X_test["id"].values, "pressure": y_pred_test_submit})
sub = sub.merge(test_u_out_by_id, on="id", how="left")

baseline_pressure = float(np.median(train_data["pressure"].values))
sub.loc[sub["u_out"] == 1, "pressure"] = baseline_pressure

sub["pressure"] = snap_to_grid(sub["pressure"].values, pressure_grid)

submission = sample[["id"]].merge(sub[["id", "pressure"]], on="id", how="left")
submission = submission.sort_values("id")
submission.to_csv("submission.csv", index=False)
submission.head()



## --- ERROR in cell 61, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3648434248.py in <cell line: 0>()
      1 # Build submission robustly by id, then apply expiratory handling + grid snapping.
----> 2 y_pred_test_submit = y_pred_test / 5
      3 
      4 sub = pd.DataFrame({"id": X_test["id"].values, "pressure": y_pred_test_submit})
      5 sub = sub.merge(test_u_out_by_id, on="id", how="left")

NameError: name 'y_pred_test' is not defined

## === cell 62
gbm_val_result.head()



## --- ERROR in cell 62, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3216970509.py in <cell line: 0>()
----> 1 gbm_val_result.head()
      2 

NameError: name 'gbm_val_result' is not defined

## === cell 63
fig, axes = plt.subplots(1, 1, figsize=(10, 10))
sns.scatterplot(data=gbm_val_result, x="correct", y="result", ax=axes)

x = np.linspace(0, 60, 10)
y = x
axes.plot(x, y, color="k")



## --- ERROR in cell 63, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1252981949.py in <cell line: 0>()
      1 fig, axes = plt.subplots(1, 1, figsize=(10, 10))
----> 2 sns.scatterplot(data=gbm_val_result, x="correct", y="result", ax=axes)
      3 
      4 x = np.linspace(0, 60, 10)
      5 y = x

NameError: name 'gbm_val_result' is not defined

## === cell 64
XGBRegressor = False



## === cell 65
if XGBRegressor:
    xgb = XGBRegressor(objective="reg:squarederror", n_estimators=700)
    xgb.fit(X_train, y_train)
    print("coefficient of determination = ", xgb.score(X_train, y_train))

    sample_xgb = sample.copy()
    sample_xgb["pressure"] = xgb.predict(X_test)
    sample_xgb = sample_xgb.sort_values("id")
    sample_xgb.to_csv("submission_xgb.csv", index=False)



## === cell 66
if XGBRegressor:
    insample_result = pd.DataFrame()
    insample_result["correct"] = y_train.values.ravel()
    insample_result["result"] = xgb.predict(X_train).ravel()

    fig, axes = plt.subplots(1, 1, figsize=(10, 10))
    sns.scatterplot(data=insample_result, x="correct", y="result", ax=axes)

    x = np.linspace(0, 60, 10)
    y = x
    axes.plot(x, y, color="k")
