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
Predict the fare amount for a taxi ride given the pickup and dropoff locations.

## Metric
Root mean-squared error.

## Submission Format
For each `key` in the test set, you must predict a value for the `fare_amount` variable. The file should contain a header and have the following format:

```
key,fare_amount
2015-01-27 13:08:24.0000002,11.00
2015-02-27 13:08:24.0000002,12.05
2015-03-27 13:08:24.0000002,11.23
2015-04-27 13:08:24.0000002,14.17
2015-05-27 13:08:24.0000002,15.12
etc
```

## Dataset
- **train.csv** - Input features and target `fare_amount` values for the training set (about 55M rows).
- **test.csv** - Input features for the test set (about 10K rows). Your goal is to predict `fare_amount` for each row.
- **sample_submission.csv** - a sample submission file in the correct format (columns `key` and `fare_amount`). This file 'predicts' `fare_amount` to be $`11.35` for all rows, which is the mean `fare_amount` from the training set.

### Data fields
**ID**

- **key** - Unique `string` identifying each row in both the training and test sets. Comprised of **pickup_datetime** plus a unique integer, but this doesn't matter, it should just be used as a unique ID field.Required in your submission CSV. Not necessarily needed in the training set, but could be useful to simulate a 'submission file' while doing cross-validation within the training set.

**Features**

- **pickup_datetime** - `timestamp` value indicating when the taxi ride started.
- **pickup_longitude** - `float` for longitude coordinate of where the taxi ride started.
- **pickup_latitude** - `float` for latitude coordinate of where the taxi ride started.
- **dropoff_longitude** - `float` for longitude coordinate of where the taxi ride ended.
- **dropoff_latitude** - `float` for latitude coordinate of where the taxi ride ended.
- **passenger_count** - `integer` indicating the number of passengers in the taxi ride.

**Target**

- **fare_amount** - `float` dollar amount of the cost of the taxi ride. This value is only in the training set; this is what you are predicting in the test set and it is required in your submission CSV.

# 2. Python version

3.7

# 3. Installed packages

geopandas==0.14.4
geopy==2.4.1
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
xgboost==2.0.3

# 4. Data file paths

```
/
    kaggle/
        data/
            GCP-Coupons-Instructions.rtf (486 Bytes)
            description.md (100 lines)
            labels.csv (55413943 lines)
            labels.csv.zip (1.6 GB)
            sample_submission.csv (9915 lines)
            sample_submission.csv.zip (76.2 kB)
            test.csv (9915 lines)
            test.csv.zip (273.0 kB)
            train.csv (55423857 lines)
            train.csv.zip (1.6 GB)
            new-york-city-taxi-fare-prediction/
                GCP-Coupons-Instructions.rtf (486 Bytes)
                description.md (100 lines)
                ... and 8 other files
                new-york-city-taxi-fare-prediction/
        input/
            GCP-Coupons-Instructions.rtf (486 Bytes)
            description.md (100 lines)
            labels.csv (55413943 lines)
            labels.csv.zip (1.6 GB)
            sample_submission.csv (9915 lines)
            sample_submission.csv.zip (76.2 kB)
            test.csv (9915 lines)
            test.csv.zip (273.0 kB)
            train.csv (55423857 lines)
            train.csv.zip (1.6 GB)
            new-york-city-taxi-fare-prediction/
                GCP-Coupons-Instructions.rtf (486 Bytes)
                description.md (100 lines)
                ... and 8 other files
                new-york-city-taxi-fare-prediction/
        working/
            new-york-city-taxi-fare-prediction/
                GCP-Coupons-Instructions.rtf (486 Bytes)
                description.md (100 lines)
                ... and 8 other files
                new-york-city-taxi-fare-prediction/
```

-> data/labels.csv has 55413942 rows and 8 columns.
The columns are: key, fare_amount, pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count

-> data/new-york-city-taxi-fare-prediction/labels.csv has 55413942 rows and 8 columns.
The columns are: key, fare_amount, pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count

-> data/new-york-city-taxi-fare-prediction/sample_submission.csv has 9914 rows and 2 columns.
The columns are: key, fare_amount

-> data/new-york-city-taxi-fare-prediction/test.csv has 9914 rows and 7 columns.
The columns are: key, pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count

-> data/new-york-city-taxi-fare-prediction/train.csv has 55423856 rows and 8 columns.
The columns are: key, fare_amount, pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count

-> data/sample_submission.csv has 9914 rows and 2 columns.
The columns are: key, fare_amount

-> data/test.csv has 9914 rows and 7 columns.
The columns are: key, pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count

-> data/train.csv has 55423856 rows and 8 columns.
The columns are: key, fare_amount, pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count

-> (stopped after 10 files for performance)

# 5. Target score

3.86972

# 6. Current score

5.78902

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 5.86517) has done: 'The changes remove the notebook‑only magic command, fix the XGBoost training call (use the current objective name and obtain the best iteration correctly), and adjust the prediction step to use the trained Booster without accessing a non‑existent attribute. These fixes allow the pipeline to run end‑to‑end and produce a properly formatted `submission.csv` file, moving the solution toward the target RMSE.'
- What this solution (achieved 5.8033) has done: 'I keep the overall pipeline unchanged but give XGBoost a more suitable set of hyper‑parameters (learning rate, depth, subsampling) and allow it to run longer so it can find a better model. I also remove the premature rounding of predictions (and clip any negative values) which can inflate the RMSE. Finally I add a quick validation RMSE printout so we can see the improvement before writing the submission.'
- What this solution (achieved 5.78902) has done: 'Implemented a log‑transformation of the target variable for the XGBoost (and Linear Regression) model, then exponentiated predictions back to the original scale before evaluating RMSE and creating the submission. This simple calibration often reduces error on skewed fare data, moving the validation RMSE closer to the target value.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from geopy.distance import great_circle
from sklearn import metrics, ensemble, linear_model
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns




## === cell 1
print(os.listdir("../input"))




## === cell 2
test = pd.read_csv("../input/test.csv")




## === cell 3
test.dtypes




## === cell 4
types = {
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "uint8",
}




## === cell 5
train = pd.read_csv("../input/train.csv", nrows=500000, dtype=types)




## === cell 6
train.head()




## === cell 7
train.describe()




## === cell 8
sns.distplot(train["fare_amount"])




## === cell 9
sns.distplot(train["passenger_count"])




## === cell 10
train.isnull().sum()




## === cell 11
train.dropna(inplace=True)




## === cell 12
train = train[train["fare_amount"] > 0]
train = train[train["pickup_longitude"] < -72]
train = train[(train["pickup_latitude"] > 40) & (train["pickup_latitude"] < 44)]
train = train[train["dropoff_longitude"] < -72]
train = train[(train["dropoff_latitude"] > 40) & (train["dropoff_latitude"] < 44)]
train = train[(train["passenger_count"] > 0) & (train["passenger_count"] < 10)]




## === cell 13
train.describe()




## === cell 14
def dist_calc(df):
    for i, row in df.iterrows():
        df.at[i, "distance"] = great_circle(
            (row["pickup_latitude"], row["pickup_longitude"]),
            (row["dropoff_latitude"], row["dropoff_longitude"]),
        ).km




## === cell 15
dist_calc(train)
dist_calc(test)




## === cell 16
train["pickup_datetime"] = train["pickup_datetime"].str.replace(" UTC", "")
train["pickup_datetime"] = pd.to_datetime(
    train["pickup_datetime"], format="%Y-%m-%d %H:%M:%S"
)




## === cell 17
test["pickup_datetime"] = test["pickup_datetime"].str.replace(" UTC", "")
test["pickup_datetime"] = pd.to_datetime(
    test["pickup_datetime"], format="%Y-%m-%d %H:%M:%S"
)




## === cell 18
train["hour"] = train.pickup_datetime.dt.hour
train["weekday"] = train.pickup_datetime.dt.weekday
train["month"] = train.pickup_datetime.dt.month
train["year"] = train.pickup_datetime.dt.year

test["hour"] = test.pickup_datetime.dt.hour
test["weekday"] = test.pickup_datetime.dt.weekday
test["month"] = test.pickup_datetime.dt.month
test["year"] = test.pickup_datetime.dt.year




## === cell 19
test.head()




## === cell 20
plt.figure(figsize=(15, 8))
sns.heatmap(
    train.drop(["key", "pickup_datetime"], axis=1).corr(), annot=True, fmt=".4f"
)




## === cell 21
X = train.drop(
    [
        "key",
        "fare_amount",
        "pickup_datetime",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
    ],
    axis=1,
)
y_original = train["fare_amount"]
y = np.log1p(y_original)




## === cell 22
X.head()




## === cell 23
y.head()




## === cell 24
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)




## === cell 25
test_pred = test.drop(
    [
        "key",
        "pickup_datetime",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
    ],
    axis=1,
)




## === cell 26
lm = LinearRegression()
lm.fit(X_train, y_train)
print("Linear train R²:", lm.score(X_train, y_train))
print("Linear val   R²:", lm.score(X_test, y_test))




## === cell 27
y_pred_log = lm.predict(X)
y_pred = np.expm1(y_pred_log)
lrmse = np.sqrt(metrics.mean_squared_error(y_original, y_pred))
print("Linear RMSE on full training data (original scale):", lrmse)




## === cell 28
LinearPredictions = np.round(np.expm1(lm.predict(test_pred)), 2)




## === cell 29
linear_submission = pd.DataFrame(
    {"key": test["key"], "fare_amount": LinearPredictions},
    columns=["key", "fare_amount"],
)




## === cell 30
def XGBoost(X_tr, X_va, y_tr, y_va):
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dvalid = xgb.DMatrix(X_va, label=y_va)

    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "eta": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "seed": 42,
        "verbosity": 0,
    }

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=2000,
        evals=[(dvalid, "valid")],
        early_stopping_rounds=50,
        verbose_eval=False,
    )
    return model




## === cell 31
xgbm = XGBoost(X_train, X_test, y_train, y_test)




## === cell 32
val_pred_log = xgbm.predict(xgb.DMatrix(X_test))
val_pred = np.expm1(val_pred_log)
val_rmse = np.sqrt(mean_squared_error(y_original.iloc[y_test.index], val_pred))
print("XGBoost validation RMSE (original scale):", val_rmse)




## --- ERROR in cell 32, traceback:
---------------------------------------------------------------------------
IndexError                                Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in _get_list_axis(self, key, axis)
   1713         try:
-> 1714             return self.obj._take_with_is_copy(key, axis=axis)
   1715         except IndexError as err:

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in _take_with_is_copy(self, indices, axis)
   4152         """
-> 4153         result = self.take(indices=indices, axis=axis)
   4154         # Maybe set copy if we didn't actually change the index.

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in take(self, indices, axis, **kwargs)
   4132 
-> 4133         new_data = self._mgr.take(
   4134             indices,

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/managers.py in take(self, indexer, axis, verify)
    890         n = self.shape[axis]
--> 891         indexer = maybe_convert_indices(indexer, n, verify=verify)
    892 

/usr/local/lib/python3.11/dist-packages/pandas/core/indexers/utils.py in maybe_convert_indices(indices, n, verify)
    281         if mask.any():
--> 282             raise IndexError("indices are out-of-bounds")
    283     return indices

IndexError: indices are out-of-bounds

The above exception was the direct cause of the following exception:

IndexError                                Traceback (most recent call last)
/tmp/ipykernel_11/1612429778.py in <cell line: 0>()
      2 val_pred_log = xgbm.predict(xgb.DMatrix(X_test))
      3 val_pred = np.expm1(val_pred_log)
----> 4 val_rmse = np.sqrt(mean_squared_error(y_original.iloc[y_test.index], val_pred))
      5 print("XGBoost validation RMSE (original scale):", val_rmse)
      6 

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in __getitem__(self, key)
   1189             maybe_callable = com.apply_if_callable(key, self.obj)
   1190             maybe_callable = self._check_deprecated_callable_usage(key, maybe_callable)
-> 1191             return self._getitem_axis(maybe_callable, axis=axis)
   1192 
   1193     def _is_scalar_access(self, key: tuple):

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in _getitem_axis(self, key, axis)
   1741         # a list of integers
   1742         elif is_list_like_indexer(key):
-> 1743             return self._get_list_axis(key, axis=axis)
   1744 
   1745         # a single integer

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in _get_list_axis(self, key, axis)
   1715         except IndexError as err:
   1716             # re-raise with different error message, e.g. test_getitem_ndarray_3d
-> 1717             raise IndexError("positional indexers are out-of-bounds") from err
   1718 
   1719     def _getitem_axis(self, key, axis: AxisInt):

IndexError: positional indexers are out-of-bounds

## === cell 33
XGBPredictions = np.expm1(xgbm.predict(xgb.DMatrix(test_pred)))
XGBPredictions = np.clip(XGBPredictions, 0, None)  # ensure no negative fares




## === cell 34
XGB_submission = pd.DataFrame(
    {"key": test["key"], "fare_amount": XGBPredictions}, columns=["key", "fare_amount"]
)




## === cell 35
XGB_submission.head()




## === cell 36
submission = XGB_submission




## === cell 37
submission.to_csv("submission.csv", index=False)




## === cell 38
print("Submission file written to submission.csv with shape:", submission.shape)
