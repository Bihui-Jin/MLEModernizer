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
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0

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

3.84109

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

- What this solution (achieved 4.55639) has done: 'I fixed the import of the deprecated NumPy object type, corrected the date‑feature extraction, implemented a proper Haversine distance calculation, kept the original latitude/longitude features for the model, and slightly improved the RandomForest hyper‑parameters. These changes resolve the runtime error and provide more informative distance features, helping the validation RMSE move closer to the target while preserving the original modelling approach.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os

print(os.listdir("../input"))




## === cell 1
train = pd.read_csv("../input/train.csv", nrows=2000000)
test = pd.read_csv("../input/test.csv")
train.head()




## === cell 2
test.head()




## === cell 3
train.shape




## === cell 4
test.shape




## === cell 5
train.dtypes.value_counts()




## === cell 6
test.dtypes.value_counts()




## === cell 7
train.isnull().sum()




## === cell 8
train = train.dropna()
train.isnull().sum()




## === cell 9
(train == 0).astype(int).sum()




## === cell 10
train = train.loc[~(train == 0).any(axis=1)]




## === cell 11
(train == 0).astype(int).sum()




## === cell 12
train.shape




## === cell 13
train.describe()




## === cell 14
train.describe()




## === cell 15
train.dtypes.value_counts()




## === cell 16
object_data = train.dtypes == object
categoricals = train.columns[object_data]
categoricals




## === cell 17
train.drop("key", axis=1, inplace=True)
train.head()




## === cell 18
import datetime as dt


def date_extraction(data):
    data["pickup_datetime"] = pd.to_datetime(data["pickup_datetime"])
    data["year"] = data["pickup_datetime"].dt.year
    data["month"] = data["pickup_datetime"].dt.month
    data["weekday"] = data["pickup_datetime"].dt.weekday  # 0 = Monday
    data["hour"] = data["pickup_datetime"].dt.hour
    data.drop("pickup_datetime", axis=1, inplace=True)
    return data


date_extraction(train)




## === cell 19
train.head()




## === cell 20
date_extraction(test)
test.head()




## === cell 21
def long_lat_distance(x):
    x["Longitude_distance"] = np.radians(x["pickup_longitude"] - x["dropoff_longitude"])
    x["Latitude_distance"] = np.radians(x["pickup_latitude"] - x["dropoff_latitude"])
    x["distance_travelled/10e3"] = (
        (x["Longitude_distance"] ** 2 + x["Latitude_distance"] ** 2) ** 0.5
    ) * 1000
    return x




## === cell 22
for x in [train, test]:
    long_lat_distance(x)
train.head()




## === cell 23
def haversine(x):
    """
    Compute Haversine distance (km) between pickup and drop‑off points.
    """
    R = 6371.0  # Earth radius in km
    lat1 = np.radians(x["pickup_latitude"])
    lon1 = np.radians(x["pickup_longitude"])
    lat2 = np.radians(x["dropoff_latitude"])
    lon2 = np.radians(x["dropoff_longitude"])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    x["haversine_km"] = R * c
    return x




## === cell 24
for x in [train, test]:
    haversine(x)
train.head()




## === cell 25
train.dtypes.value_counts()




## === cell 26
train.head()




## === cell 27
test.head()




## === cell 28
train.describe()




## === cell 29
print("Are there any nulls in the train data: ")
print(train.isnull().sum())

print("\nAre there any nulls in the test data: ")
print(test.isnull().sum())




## === cell 30
train["fare_amount"] = np.log1p(train["fare_amount"])

from sklearn.ensemble import RandomForestRegressor

feature_cols = [x for x in train.columns if x != "fare_amount"]
X = train[feature_cols]
y = train["fare_amount"]




## === cell 31
correlations = X.corrwith(y)
correlations = abs(correlations * 100)
correlations.sort_values(ascending=False, inplace=True)

correlations




## === cell 32
ax = correlations.plot(kind="bar")
ax.set(ylim=[-1, 1], ylabel="pearson correlation")




## === cell 33
train.head()




## === cell 34
train_1 = train.copy()
train_1.head()




## === cell 35
train_1["haversine_km"] = train_1["haversine_km"].round(2)
train_1["distance_travelled/10e3"] = train_1["distance_travelled/10e3"].round(2)

train_1.head()




## === cell 36
train_1.describe()




## === cell 37
test_1 = test.copy()
test_1.head()




## === cell 38
from sklearn.model_selection import train_test_split

feat_cols = [x for x in train_1.columns if x != "fare_amount"]
X_1 = train_1[feat_cols]
y_1 = train_1["fare_amount"]
X_train, X_test, y_train, y_test = train_test_split(
    X_1, y_1, test_size=0.25, random_state=42
)




## === cell 39
rf = RandomForestRegressor(
    n_estimators=800, max_features="sqrt", random_state=42, n_jobs=-1
)
rf = rf.fit(X_train, y_train)




## --- ERROR in cell 39, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2278520300.py in <cell line: 0>()
      3     n_estimators=800, max_features="sqrt", random_state=42, n_jobs=-1
      4 )
----> 5 rf = rf.fit(X_train, y_train)
      6 
      7 

/usr/local/lib/python3.11/dist-packages/sklearn/ensemble/_forest.py in fit(self, X, y, sample_weight)
    343         if issparse(y):
    344             raise ValueError("sparse multilabel-indicator for y is not supported.")
--> 345         X, y = self._validate_data(
    346             X, y, multi_output=True, accept_sparse="csc", dtype=DTYPE
    347         )

/usr/local/lib/python3.11/dist-packages/sklearn/base.py in _validate_data(self, X, y, reset, validate_separately, **check_params)
    582                 y = check_array(y, input_name="y", **check_y_params)
    583             else:
--> 584                 X, y = check_X_y(X, y, **check_params)
    585             out = X, y
    586 

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in check_X_y(X, y, accept_sparse, accept_large_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, multi_output, ensure_min_samples, ensure_min_features, y_numeric, estimator)
   1120     )
   1121 
-> 1122     y = _check_y(y, multi_output=multi_output, y_numeric=y_numeric, estimator=estimator)
   1123 
   1124     check_consistent_length(X, y)

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in _check_y(y, multi_output, y_numeric, estimator)
   1130     """Isolated part of check_X_y dedicated to y validation"""
   1131     if multi_output:
-> 1132         y = check_array(
   1133             y,
   1134             accept_sparse="csr",

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

ValueError: Input y contains NaN.

## === cell 40
from sklearn.metrics import mean_squared_error

val_pred_log = rf.predict(X_test)
val_pred = np.expm1(val_pred_log)  # back to original fare scale
val_true = np.expm1(y_test)
rmse = np.sqrt(mean_squared_error(val_true, val_pred))
print(f"Validation RMSE (log‑target back‑transformed): {rmse:.5f}")




## --- ERROR in cell 40, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/2939435917.py in <cell line: 0>()
      2 
      3 # Validation predictions (still in log‑space)
----> 4 val_pred_log = rf.predict(X_test)
      5 val_pred = np.expm1(val_pred_log)  # back to original fare scale
      6 val_true = np.expm1(y_test)

/usr/local/lib/python3.11/dist-packages/sklearn/ensemble/_forest.py in predict(self, X)
    985 
    986         # avoid storing the output of every estimator by summing them here
--> 987         if self.n_outputs_ > 1:
    988             y_hat = np.zeros((X.shape[0], self.n_outputs_), dtype=np.float64)
    989         else:

AttributeError: 'RandomForestRegressor' object has no attribute 'n_outputs_'

## === cell 41
test.head()




## === cell 42
test_1.drop("key", axis=1, inplace=True)
test_1.head()




## === cell 43
final_prediction = np.expm1(rf.predict(test_1))

NYCtaxiFare_submission = pd.DataFrame(
    {"key": test["key"], "fare_amount": final_prediction}
)
NYCtaxiFare_submission.to_csv("NYCtaxiFare_prediction.csv", index=False)




## --- ERROR in cell 43, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/3672988471.py in <cell line: 0>()
      1 # Predict on test set and revert the log transformation
----> 2 final_prediction = np.expm1(rf.predict(test_1))
      3 
      4 NYCtaxiFare_submission = pd.DataFrame(
      5     {"key": test["key"], "fare_amount": final_prediction}

/usr/local/lib/python3.11/dist-packages/sklearn/ensemble/_forest.py in predict(self, X)
    985 
    986         # avoid storing the output of every estimator by summing them here
--> 987         if self.n_outputs_ > 1:
    988             y_hat = np.zeros((X.shape[0], self.n_outputs_), dtype=np.float64)
    989         else:

AttributeError: 'RandomForestRegressor' object has no attribute 'n_outputs_'

## === cell 44
NYCtaxiFare_submission.head()

## --- ERROR in cell 44, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2256468999.py in <cell line: 0>()
----> 1 NYCtaxiFare_submission.head()

NameError: name 'NYCtaxiFare_submission' is not defined
