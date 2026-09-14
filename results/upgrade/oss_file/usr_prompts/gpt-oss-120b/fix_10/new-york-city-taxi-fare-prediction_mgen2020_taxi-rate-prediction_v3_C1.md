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

5.689

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 936.92806) has done: 'I remove the deprecated `normalize` argument from `LinearRegression`, ensure the model variable persists for prediction, and adjust the submission writing step to output a proper CSV with the required columns and no index column. This resolves the runtime errors and creates a valid `submission.csv` file.'
- What this solution (achieved 7.98509) has done: 'Implemented a lightweight acceleration by patching scikit‑learn with Intel® ‑ oneAPI optimizations before the RandomForest model is instantiated. This change does not alter any algorithmic parameters, data preprocessing, or evaluation logic, but it significantly speeds up tree construction and prediction, keeping the entire pipeline within the 600 s limit.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
usecols = [
    "key",
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
dtypes = {
    "key": "object",
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int8",
}
td = pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv",
    usecols=usecols,
    dtype=dtypes,
    nrows=10_000_000,
)
if len(td) > 2_000_000:
    td = td.sample(frac=2_000_000 / len(td), random_state=42).reset_index(drop=True)
td.head()



## === cell 2
td.shape



## === cell 3
td.info()



## === cell 4
test_usecols = [col for col in usecols if col != "fare_amount"]
test_dtypes = {k: v for k, v in dtypes.items() if k != "fare_amount"}

ted = pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv",
    usecols=test_usecols,  # test does not have fare_amount
    dtype=test_dtypes,
)
ted.head()



## === cell 5
ted.info()



## === cell 6
td.isna().sum()



## === cell 7
td["Difference_longitude"] = np.abs(td["pickup_longitude"] - td["dropoff_longitude"])
td["Difference_latitude"] = np.abs(td["pickup_latitude"] - td["dropoff_latitude"])
ted["Difference_longitude"] = np.abs(ted["pickup_longitude"] - ted["dropoff_longitude"])
ted["Difference_latitude"] = np.abs(ted["pickup_latitude"] - ted["dropoff_latitude"])



## === cell 8
print(f"Before Dropping null values: {len(td)}")
td.dropna(inplace=True)
print(f"After Dropping null values: {len(td)}")



## === cell 9
plot = td[:2000].plot.scatter("Difference_longitude", "Difference_latitude")



## === cell 10
td = td[(td["Difference_longitude"] < 5.0) & (td["Difference_latitude"] < 5.0)]



## === cell 11
td["pickup_datetime"] = pd.to_datetime(td["pickup_datetime"])
td["pickuptime"] = td["pickup_datetime"].dt.hour * 100 + td["pickup_datetime"].dt.minute
td["Weekday"] = td["pickup_datetime"].dt.weekday.map(
    {
        0: "Monday",
        1: "Tuesday",
        2: "Wednesday",
        3: "Thursday",
        4: "Friday",
        5: "Saturday",
        6: "Sunday",
    }
)

ted["pickup_datetime"] = pd.to_datetime(ted["pickup_datetime"])
ted["pickuptime"] = (
    ted["pickup_datetime"].dt.hour * 100 + ted["pickup_datetime"].dt.minute
)
ted["Weekday"] = ted["pickup_datetime"].dt.weekday.map(
    {
        0: "Monday",
        1: "Tuesday",
        2: "Wednesday",
        3: "Thursday",
        4: "Friday",
        5: "Saturday",
        6: "Sunday",
    }
)



## === cell 12
td.head()



## === cell 13
td.head()



## === cell 14
ted.head()



## === cell 15
td.drop("pickup_datetime", inplace=True, axis=1)
ted.drop("pickup_datetime", inplace=True, axis=1)

th = pd.get_dummies(td["Weekday"], dtype=np.float32)
teh = pd.get_dummies(ted["Weekday"], dtype=np.float32)

td = pd.concat([td, th], axis=1)
ted = pd.concat([ted, teh], axis=1)

td.drop("Weekday", axis=1, inplace=True)
ted.drop("Weekday", inplace=True, axis=1)

td.head()




## === cell 16
def add_haversine_features(df):
    R = 6373.0
    lat1 = np.radians(df["pickup_latitude"].values.astype(np.float32))
    lon1 = np.radians(df["pickup_longitude"].values.astype(np.float32))
    lat2 = np.radians(df["dropoff_latitude"].values.astype(np.float32))
    lon2 = np.radians(df["dropoff_longitude"].values.astype(np.float32))

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    df["Distance"] = (R * c * 0.621).round(2)  # miles

    lat3 = np.full(len(df), np.radians(40.6413111), dtype=np.float32)
    lon3 = np.full(len(df), np.radians(-73.7781391), dtype=np.float32)

    dlon_pickup = lon3 - lon1
    dlat_pickup = lat3 - lat1
    a1 = (
        np.sin(dlat_pickup / 2) ** 2
        + np.cos(lat1) * np.cos(lat3) * np.sin(dlon_pickup / 2) ** 2
    )
    df["Pickup_Distance_airport"] = (
        R * 2 * np.arctan2(np.sqrt(a1), np.sqrt(1 - a1)) * 0.621
    ).round(2)

    dlon_dropoff = lon3 - lon2
    dlat_dropoff = lat3 - lat2
    a2 = (
        np.sin(dlon_dropoff / 2) ** 2
        + np.cos(lat2) * np.cos(lat3) * np.sin(dlon_dropoff / 2) ** 2
    )
    df["Dropoff_Distance_airport"] = (
        R * 2 * np.arctan2(np.sqrt(a2), np.sqrt(1 - a2)) * 0.621
    ).round(2)

    df.drop(
        [
            "pickup_longitude",
            "pickup_latitude",
            "dropoff_longitude",
            "dropoff_latitude",
        ],
        axis=1,
        inplace=True,
    )
    return df


td = add_haversine_features(td)
ted = add_haversine_features(ted)

for col in ["Difference_longitude", "Difference_latitude"]:
    td[col] = np.abs(td[col] - td[col].mean()) / td[col].var()
    ted[col] = np.abs(ted[col] - ted[col].mean()) / ted[col].var()

td.shape



## === cell 17
ted.shape



## === cell 18
from sklearnex import patch_sklearn

patch_sklearn()

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

X = td.drop(["key", "fare_amount"], axis=1).astype(np.float32)
y = td["fare_amount"].astype(np.float32)

y_log = np.log1p(y).astype(np.float32)

median_vals = X.median()
X = X.fillna(median_vals)

X_train, X_val, y_train_log, y_val = train_test_split(
    X, y_log, test_size=0.01, random_state=80
)

rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=None,
    n_jobs=5,
    random_state=42,
    min_samples_leaf=1,
)
rf.fit(X_train, y_train_log)

val_pred_log = rf.predict(X_val)
val_pred = np.expm1(val_pred_log)  # back to original scale
rmse = np.sqrt(mean_squared_error(y_val, val_pred))
print(f"Validation RMSE: {rmse:.4f}")



## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2682009702.py in <cell line: 0>()
     27     min_samples_leaf=1,
     28 )
---> 29 rf.fit(X_train, y_train_log)
     30 
     31 # Predict on validation, invert log transform

/usr/local/lib/python3.11/dist-packages/daal4py/sklearn/_n_jobs_support.py in n_jobs_wrapper(self, *args, **kwargs)
    130             )
    131             set_n_threads(n_jobs)
--> 132             return method(self, *args, **kwargs)
    133         finally:
    134             set_n_threads(old_n_threads)

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

## === cell 19
test_features = ted.drop("key", axis=1).astype(np.float32)
test_features = test_features.fillna(median_vals)  # use training medians
test_pred_log = rf.predict(test_features)
pred = np.expm1(test_pred_log)
pred = np.round(pred, 2)

print(pred[:5])



## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NotFittedError                            Traceback (most recent call last)
/tmp/ipykernel_11/3248123929.py in <cell line: 0>()
      1 test_features = ted.drop("key", axis=1).astype(np.float32)
      2 test_features = test_features.fillna(median_vals)  # use training medians
----> 3 test_pred_log = rf.predict(test_features)
      4 pred = np.expm1(test_pred_log)
      5 pred = np.round(pred, 2)

/usr/local/lib/python3.11/dist-packages/daal4py/sklearn/_n_jobs_support.py in n_jobs_wrapper(self, *args, **kwargs)
    130             )
    131             set_n_threads(n_jobs)
--> 132             return method(self, *args, **kwargs)
    133         finally:
    134             set_n_threads(old_n_threads)

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

## === cell 20
Submission = pd.DataFrame({"key": ted["key"], "fare_amount": pred})
Submission.to_csv("submission.csv", index=False)

## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3739353413.py in <cell line: 0>()
----> 1 Submission = pd.DataFrame({"key": ted["key"], "fare_amount": pred})
      2 Submission.to_csv("submission.csv", index=False)

NameError: name 'pred' is not defined
