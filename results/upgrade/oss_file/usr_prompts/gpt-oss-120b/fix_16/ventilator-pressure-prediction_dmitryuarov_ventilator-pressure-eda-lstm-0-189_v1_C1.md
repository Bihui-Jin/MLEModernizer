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

catboost==1.2.8
geopandas==0.14.4
lightgbm==4.6.0
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
optuna==4.5.0
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
plotly==5.24.1
plotly-express==0.4.1
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

# 5. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold
from lightgbm import LGBMRegressor

from joblib import Parallel, delayed

pd.set_option("display.max_columns", None)

dtypes = {
    "R": "int8",
    "C": "int8",
    "breath_id": "int32",
    "id": "int16",
    "time_step": "float32",
    "u_in": "float32",
    "u_out": "int8",
    "pressure": "float32",
}
train = pd.read_csv("../input/ventilator-pressure-prediction/train.csv", dtype=dtypes)
test = pd.read_csv("../input/ventilator-pressure-prediction/test.csv", dtype=dtypes)
ss = pd.read_csv("../input/ventilator-pressure-prediction/sample_submission.csv")



## === cell 1
train.head(3)



## === cell 2
print(f"Length of TRAIN dataset: {len(train)}")
print(f"Length of TEST dataset: {len(test)}")
print("")
print("Missing values in TRAIN dataset")
for i in train.iloc[:, 0:-1].columns.tolist():
    print(f"{i}: {train[i].isna().sum()}")
print("")
print("Missing values in TEST dataset")
for i in test.iloc[:, 0:-1].columns.tolist():
    print(f"{i}: {test[i].isna().sum()}")
print("")
print(f'Number of breaths in train dataset: {train["breath_id"].nunique()}')
print(f'Number of breaths in test dataset: {test["breath_id"].nunique()}')
print(
    f'The number of observations for each breath: {train["breath_id"].value_counts().reset_index()["breath_id"].unique()[0]}'
)




## === cell 3
def features(df):
    df = df.copy()
    df["area"] = df["time_step"] * df["u_in"]
    g = df.groupby("breath_id", sort=False)

    df["area"] = g["area"].cumsum()
    df["u_in_cumsum"] = g["u_in"].cumsum()

    df["u_in_lag2"] = g["u_in"].shift(2).fillna(0)
    df["u_in_lag4"] = g["u_in"].shift(4).fillna(0)

    df["ewm_u_in_mean"] = g["u_in"].transform(lambda x: x.ewm(halflife=10).mean())
    df["ewm_u_in_std"] = g["u_in"].transform(lambda x: x.ewm(halflife=10).std())

    df["rolling_10_mean"] = g["u_in"].transform(
        lambda x: x.rolling(window=10, min_periods=1).mean()
    )
    df["rolling_10_max"] = g["u_in"].transform(
        lambda x: x.rolling(window=10, min_periods=1).max()
    )
    df["rolling_10_std"] = g["u_in"].transform(
        lambda x: x.rolling(window=10, min_periods=1).std()
    )

    df["expand_mean"] = g["u_in"].transform(lambda x: x.expanding(2).mean())
    df["expand_max"] = g["u_in"].transform(lambda x: x.expanding(2).max())
    df["expand_std"] = g["u_in"].transform(lambda x: x.expanding(2).std())

    df["R"] = df["R"].astype(str)
    df["C"] = df["C"].astype(str)
    df = pd.get_dummies(df, columns=["R", "C"], dtype=np.uint8)

    return df


full = pd.concat([train, test], axis=0, ignore_index=True)
full = features(full)

train = full.iloc[: len(train)].reset_index(drop=True).copy()
test = full.iloc[len(train) :].reset_index(drop=True).copy()

import gc

gc.collect()



## === cell 4
train = train.fillna(0)
test = test.fillna(0)



## === cell 5
y = train["pressure"].values
train = train.drop(["pressure", "id", "breath_id"], axis=1)
test = test.drop(["id", "breath_id"], axis=1, errors="ignore")
test = test.drop(["pressure"], axis=1, errors="ignore")



## === cell 6
scaler = RobustScaler()
train = scaler.fit_transform(train).astype(np.float32)
test = scaler.transform(test).astype(np.float32)



## === cell 7
import lightgbm as lgb

kf = KFold(n_splits=5, shuffle=True, random_state=228)


def run_fold(fold_idx, tr_idx, val_idx):
    X_tr, X_val = train[tr_idx], train[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]

    model = LGBMRegressor(
        n_estimators=4000,
        learning_rate=0.01,
        num_leaves=128,
        objective="mae",
        metric="mae",
        n_jobs=1,
        verbose=-1,
        random_state=fold_idx + 228,
    )
    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric="mae",
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)],
    )
    preds = model.predict(test)
    return preds


fold_results = Parallel(n_jobs=5, backend="threading")(
    delayed(run_fold)(fold, tr_idx, val_idx)
    for fold, (tr_idx, val_idx) in enumerate(kf.split(train))
)

test_preds = list(fold_results)



## === cell 8
ss["pressure"] = np.mean(test_preds, axis=0)
ss.to_csv("submission.csv", index=False)
