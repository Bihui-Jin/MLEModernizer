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

3.9

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

5.69253

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

- What this solution (achieved 113.72277) has done: 'I remove the unsupported `normalize` argument from the LinearRegression constructor and adjust the submission creation so the CSV has the required `key` column (instead of using it as an index). These fixes resolve the runtime errors and ensure a proper submission file is written.'

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

TRAIN_PATH = "../input/train.csv"
TEST_PATH = "../input/test.csv"

train_data = pd.read_csv(TRAIN_PATH, nrows=2_000_000)  # sample for speed
test_data = pd.read_csv(TEST_PATH)




## === cell 1
def add_travel_vector_features(df):
    df["abs_diff_longitude"] = (df["dropoff_longitude"] - df["pickup_longitude"]).abs()
    df["abs_diff_latitude"] = (df["dropoff_latitude"] - df["pickup_latitude"]).abs()


add_travel_vector_features(train_data)
add_travel_vector_features(test_data)



## === cell 2
R = 6372.0  # Earth radius in km


def haversine(df):
    lat1 = np.radians(df["pickup_latitude"])
    lon1 = np.radians(df["pickup_longitude"])
    lat2 = np.radians(df["dropoff_latitude"])
    lon2 = np.radians(df["dropoff_longitude"])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    km = R * c
    return km * 0.621371  # convert to miles


train_data["Distance"] = haversine(train_data)
test_data["Distance"] = haversine(test_data)



## === cell 3
airport_lat = np.radians(40.6413)
airport_lon = np.radians(-73.7781)


def airport_distances(df):
    lat = np.radians(df["pickup_latitude"])
    lon = np.radians(df["pickup_longitude"])
    dlat = airport_lat - lat
    dlon = airport_lon - lon
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat) * np.cos(airport_lat) * np.sin(dlon / 2) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    km = R * c
    return km * 0.621371


train_data["pickup_distance_a"] = airport_distances(train_data)

lat = np.radians(test_data["pickup_latitude"])
lon = np.radians(test_data["pickup_longitude"])
dlat = airport_lat - lat
dlon = airport_lon - lon
a = np.sin(dlat / 2) ** 2 + np.cos(lat) * np.cos(airport_lat) * np.sin(dlon / 2) ** 2
c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
km = R * c
test_data["pickup_distance_a"] = km * 0.621371


def airport_distances_drop(df):
    lat = np.radians(df["dropoff_latitude"])
    lon = np.radians(df["dropoff_longitude"])
    dlat = airport_lat - lat
    dlon = airport_lon - lon
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat) * np.cos(airport_lat) * np.sin(dlon / 2) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    km = R * c
    return km * 0.621371


train_data["dropoff_distance_a"] = airport_distances_drop(train_data)
test_data["dropoff_distance_a"] = airport_distances_drop(test_data)



## === cell 4
train_data["pickup_datetime"] = pd.to_datetime(train_data["pickup_datetime"])
test_data["pickup_datetime"] = pd.to_datetime(test_data["pickup_datetime"])

train_data["weekday"] = train_data["pickup_datetime"].dt.weekday
test_data["weekday"] = test_data["pickup_datetime"].dt.weekday

train_one_hot = pd.get_dummies(train_data["weekday"], prefix="wd")
test_one_hot = pd.get_dummies(test_data["weekday"], prefix="wd")

train_data = pd.concat([train_data, train_one_hot], axis=1)
test_data = pd.concat([test_data, test_one_hot], axis=1)

train_data.drop(
    columns=[
        "pickup_datetime",
        "weekday",
        "pickup_latitude",
        "pickup_longitude",
        "dropoff_latitude",
        "dropoff_longitude",
    ],
    inplace=True,
)
test_data.drop(
    columns=[
        "pickup_datetime",
        "weekday",
        "pickup_latitude",
        "pickup_longitude",
        "dropoff_latitude",
        "dropoff_longitude",
    ],
    inplace=True,
)



## === cell 5
num_cols = [
    "abs_diff_longitude",
    "abs_diff_latitude",
    "Distance",
    "pickup_distance_a",
    "dropoff_distance_a",
]
for col in num_cols:
    mean = train_data[col].mean()
    var = train_data[col].var()
    train_data[col] = (train_data[col] - mean) / var
    test_data[col] = (test_data[col] - mean) / var  # use training stats



## === cell 6
X = train_data.drop(["key", "fare_amount"], axis=1)
y = train_data["fare_amount"]
y_log = np.log1p(y)

X_train, X_valid, y_train_log, y_valid_log = train_test_split(
    X, y_log, test_size=0.01, random_state=42
)

rf = RandomForestRegressor(
    n_estimators=150,
    max_depth=15,
    min_samples_split=2,
    min_samples_leaf=1,
    n_jobs=-1,
    random_state=42,
)

rf.fit(X_train, y_train_log)



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3049508889.py in <cell line: 0>()
     16 )
     17 
---> 18 rf.fit(X_train, y_train_log)
     19 

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
RandomForestRegressor does not accept missing values encoded as NaN natively. For supervised learning, you might want to consider sklearn.ensemble.HistGradientBoostingClassifier and Regressor which accept missing values encoded as NaNs natively. Alternatively, it is possible to preprocess the data, for instance by using an imputer transformer in a pipeline or drop samples with missing values. See https://scikit-learn.org/stable/modules/impute.html You can find a list of all estimators that handle NaN values at the following page: https://scikit-learn.org/stable/modules/impute.html#estimators-that-handle-nan-values

## === cell 7
valid_pred_log = rf.predict(X_valid)
valid_pred = np.expm1(valid_pred_log)
y_valid = np.expm1(y_valid_log)

rmse = np.sqrt(mean_squared_error(y_valid, valid_pred))
print(f"Validation RMSE: {rmse:.5f}")



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/4171356454.py in <cell line: 0>()
----> 1 valid_pred_log = rf.predict(X_valid)
      2 valid_pred = np.expm1(valid_pred_log)
      3 y_valid = np.expm1(y_valid_log)
      4 
      5 rmse = np.sqrt(mean_squared_error(y_valid, valid_pred))

/usr/local/lib/python3.11/dist-packages/sklearn/ensemble/_forest.py in predict(self, X)
    985 
    986         # avoid storing the output of every estimator by summing them here
--> 987         if self.n_outputs_ > 1:
    988             y_hat = np.zeros((X.shape[0], self.n_outputs_), dtype=np.float64)
    989         else:

AttributeError: 'RandomForestRegressor' object has no attribute 'n_outputs_'

## === cell 8
test_features = test_data.drop("key", axis=1)
test_pred_log = rf.predict(test_features)
test_pred = np.expm1(test_pred_log)
test_pred = np.clip(test_pred, a_min=0, a_max=None)
test_pred = np.round(test_pred, 2)

submission = pd.DataFrame({"key": test_data["key"], "fare_amount": test_pred})
submission.to_csv("submission.csv", index=False)
print("Submission file saved as submission.csv")

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/2362896778.py in <cell line: 0>()
      1 test_features = test_data.drop("key", axis=1)
----> 2 test_pred_log = rf.predict(test_features)
      3 test_pred = np.expm1(test_pred_log)
      4 test_pred = np.clip(test_pred, a_min=0, a_max=None)
      5 test_pred = np.round(test_pred, 2)

/usr/local/lib/python3.11/dist-packages/sklearn/ensemble/_forest.py in predict(self, X)
    985 
    986         # avoid storing the output of every estimator by summing them here
--> 987         if self.n_outputs_ > 1:
    988             y_hat = np.zeros((X.shape[0], self.n_outputs_), dtype=np.float64)
    989         else:

AttributeError: 'RandomForestRegressor' object has no attribute 'n_outputs_'
