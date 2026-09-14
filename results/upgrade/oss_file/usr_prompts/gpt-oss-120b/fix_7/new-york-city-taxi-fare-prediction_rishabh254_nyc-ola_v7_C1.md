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

5.73504

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 1079.44035) has done: 'I add a haversine‑distance feature (and keep passenger count) and feed it to the same linear‑regression pipeline you already use. This modest feature boost is expected to drop the RMSE from the huge 895 → near the target ≈ 5.7 without changing the core modelling approach.'
- What this solution (achieved 10.20663) has done: 'The fix focuses on reducing I/O and unnecessary work, removing the unused plot, and letting the RandomForest use all available CPU cores. We read only the columns needed for training (skipping the `key` column), do the same for the test set while keeping `key` for the submission, drop the matplotlib scatter which adds overhead, and set `n_jobs=-1` so the tree ensemble fully exploits parallelism. These changes keep every computation that influences the model identical, so the predictions and validation score remain unchanged while the overall runtime fits well inside the 600‑second limit.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O
import os
from sklearnex import patch_sklearn

patch_sklearn()

print(os.listdir("../input"))




## === cell 1
dtype_train = {
    "fare_amount": np.float32,
    "pickup_longitude": np.float32,
    "pickup_latitude": np.float32,
    "dropoff_longitude": np.float32,
    "dropoff_latitude": np.float32,
    "passenger_count": np.int8,
}
usecols_train = list(dtype_train.keys())
data = pd.read_csv(
    "../input/train.csv",
    nrows=20_000_000,
    dtype=dtype_train,
    usecols=usecols_train,
    low_memory=False,
)




## === cell 2
def add_diff_features(df):
    """Add cheap absolute longitude/latitude differences."""
    df["abs_diff_longitude"] = (df.dropoff_longitude - df.pickup_longitude).abs()
    df["abs_diff_latitude"] = (df.dropoff_latitude - df.pickup_latitude).abs()


def add_distance_features(df):
    """Add haversine distance (km) – called only after filtering."""
    lon1 = np.radians(df.pickup_longitude.values.astype(np.float32))
    lat1 = np.radians(df.pickup_latitude.values.astype(np.float32))
    lon2 = np.radians(df.dropoff_longitude.values.astype(np.float32))
    lat2 = np.radians(df.dropoff_latitude.values.astype(np.float32))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    df["distance"] = (6371.0 * c).astype(np.float32)  # Earth radius in km


add_diff_features(data)




## === cell 3
print(data.isnull().sum())
print("Old size: %d" % len(data))
data = data.dropna(how="any", axis="rows")
print("New size: %d" % len(data))




## === cell 4
print("Old size: %d" % len(data))
data = data[(data.abs_diff_longitude < 1.0) & (data.abs_diff_latitude < 1.0)]
print("New size: %d" % len(data))
add_distance_features(data)




## === cell 5
from sklearn.model_selection import train_test_split

y = data.fare_amount.astype(np.float32)
X = data.drop("fare_amount", axis=1)

train_df, val_df, train_y, val_y = train_test_split(
    X, y, test_size=0.2, random_state=42
)

train_y = train_y.replace([np.inf, -np.inf], np.nan).dropna()
val_y = val_y.replace([np.inf, -np.inf], np.nan).dropna()

train_y_log = np.log1p(train_y).astype(np.float32)
val_y_log = np.log1p(val_y).astype(np.float32)




## === cell 6
def get_input_matrix(df):
    return np.column_stack(
        (
            df.abs_diff_longitude.values.astype(np.float32),
            df.abs_diff_latitude.values.astype(np.float32),
            df.distance.values.astype(np.float32),
            df.passenger_count.values.astype(np.float32),
            np.ones(len(df), dtype=np.float32),  # intercept term
        )
    )


train_X = get_input_matrix(train_df)
val_X = get_input_matrix(val_df)

print("train_X shape:", train_X.shape)
print("train_y shape:", train_y.shape)




## === cell 7
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(
    n_estimators=400,  # more trees for better fit
    max_depth=None,  # allow deeper trees
    n_jobs=-1,
    random_state=42,
)

rf.fit(train_X, train_y_log)
print("RandomForestRegressor trained on log target.")




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2432524965.py in <cell line: 0>()
      9 
     10 # Fit using NumPy arrays to avoid dtype‑related checks in sklearnex
---> 11 rf.fit(train_X, train_y_log)
     12 print("RandomForestRegressor trained on log target.")
     13 

/usr/local/lib/python3.11/dist-packages/daal4py/sklearn/_n_jobs_support.py in n_jobs_wrapper(self, *args, **kwargs)
    120         old_n_threads = get_n_threads()
    121         if n_jobs == old_n_threads:
--> 122             return method(self, *args, **kwargs)
    123 
    124         try:

/usr/local/lib/python3.11/dist-packages/sklearnex/ensemble/_forest.py in fit(self, X, y, sample_weight)
   1173 
   1174     def fit(self, X, y, sample_weight=None):
-> 1175         dispatch(
   1176             self,
   1177             "fit",

/usr/local/lib/python3.11/dist-packages/sklearnex/_device_offload.py in dispatch(obj, method_name, branches, *args, **kwargs)
    145 
    146         while backend is None:
--> 147             backend, patching_status = _get_backend(obj, method_name, *hostargs)
    148 
    149         if backend:

/usr/local/lib/python3.11/dist-packages/sklearnex/_device_offload.py in _get_backend(obj, method_name, *data)
     45 
     46     if cpu_device:
---> 47         patching_status = obj._onedal_cpu_supported(method_name, *data)
     48         return patching_status.get_status(), patching_status
     49 

/usr/local/lib/python3.11/dist-packages/sklearnex/ensemble/_forest.py in _onedal_cpu_supported(self, method_name, *data)
   1046 
   1047         if method_name == "fit":
-> 1048             patching_status, X, y, sample_weight = self._onedal_fit_ready(
   1049                 patching_status, *data
   1050             )

/usr/local/lib/python3.11/dist-packages/sklearnex/ensemble/_forest.py in _onedal_fit_ready(self, patching_status, X, y, sample_weight)
    982                 )
    983             else:
--> 984                 X, y = check_X_y(
    985                     X,
    986                     y,

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

/usr/local/lib/python3.11/dist-packages/daal4py/sklearn/utils/validation.py in _assert_all_finite(X, allow_nan, msg_dtype, estimator_name, input_name)
    132         elif dt == np.float32:
    133             if not d4p.daal_assert_all_finite(x_for_daal, allow_nan, 1):
--> 134                 raise ValueError(err)
    135     # First try an O(n) time, O(1) space solution for the common case that
    136     # everything is finite; fall back to O(n) space np.isfinite to prevent

ValueError: Input y contains NaN, infinity or a value too large for dtype('float32').

## === cell 8
dtype_test = {
    "key": object,
    "pickup_longitude": np.float32,
    "pickup_latitude": np.float32,
    "dropoff_longitude": np.float32,
    "dropoff_latitude": np.float32,
    "passenger_count": np.int8,
}
test_df = pd.read_csv(
    "../input/test.csv",
    dtype=dtype_test,
    low_memory=False,
)

add_diff_features(test_df)
add_distance_features(test_df)

test_X = get_input_matrix(test_df)




## === cell 9
val_y_log_pred = rf.predict(val_X)
val_y_pred = np.expm1(val_y_log_pred)

test_y_log_pred = rf.predict(test_X)
test_y_pred = np.expm1(test_y_log_pred)

from sklearn.metrics import mean_squared_error

val_rmse = np.sqrt(mean_squared_error(val_y, val_y_pred))
print("Validation RMSE:", val_rmse)

test_y_pred = np.round(test_y_pred, 2)

submission = pd.DataFrame(
    {"key": test_df["key"], "fare_amount": test_y_pred},
    columns=["key", "fare_amount"],
)
submission.to_csv("submission.csv", index=False)

print("Files in cwd:", os.listdir("."))

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NotFittedError                            Traceback (most recent call last)
/tmp/ipykernel_11/1374087302.py in <cell line: 0>()
----> 1 val_y_log_pred = rf.predict(val_X)
      2 val_y_pred = np.expm1(val_y_log_pred)
      3 
      4 test_y_log_pred = rf.predict(test_X)
      5 test_y_pred = np.expm1(test_y_log_pred)

/usr/local/lib/python3.11/dist-packages/daal4py/sklearn/_n_jobs_support.py in n_jobs_wrapper(self, *args, **kwargs)
    120         old_n_threads = get_n_threads()
    121         if n_jobs == old_n_threads:
--> 122             return method(self, *args, **kwargs)
    123 
    124         try:

/usr/local/lib/python3.11/dist-packages/sklearnex/_device_offload.py in wrapper(self, *args, **kwargs)
    180     @wraps(func)
    181     def wrapper(self, *args, **kwargs) -> Any:
--> 182         result = func(self, *args, **kwargs)
    183         if not (len(args) == 0 and len(kwargs) == 0):
    184             data = (*args, *kwargs.values())[0]

/usr/local/lib/python3.11/dist-packages/sklearnex/ensemble/_forest.py in predict(self, X)
   1188     @wrap_output_data
   1189     def predict(self, X):
-> 1190         check_is_fitted(self)
   1191         return dispatch(
   1192             self,

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in check_is_fitted(estimator, attributes, msg, all_or_any)
   1388 
   1389     if not fitted:
-> 1390         raise NotFittedError(msg % {"name": type(estimator).__name__})
   1391 
   1392 

NotFittedError: This RandomForestRegressor instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.
