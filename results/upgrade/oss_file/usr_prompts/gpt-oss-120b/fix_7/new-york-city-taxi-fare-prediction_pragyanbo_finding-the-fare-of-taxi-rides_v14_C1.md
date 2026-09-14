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

4.54269

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 5.9093) has done: 'I removed the failing TensorFlow sections, fixed the notebook‑only `%matplotlib inline` line, added a modest amount of extra training data and a simple hyper‑parameter tweak, and introduced a lightweight GradientBoosting model. The script now evaluates both models on a validation split, picks the one with the lower RMSE, and writes a proper `submission.csv` file containing the required `key` and `fare_amount` columns.'
- What this solution (achieved 5.73305) has done: 'The changes reduce the number of trees and depth for the RandomForest and GradientBoosting models, which cuts the most expensive training loops while keeping the same model types and feature set. These hyper‑parameter adjustments preserve the overall logic and still produce valid predictions, allowing the entire pipeline to finish well under the 600‑second limit.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from sklearnex import patch_sklearn

patch_sklearn()  # patches RandomForestRegressor & GradientBoostingRegressor in place


## === cell 1
train_path = "../input/train.csv"
test_path = "../input/test.csv"
train_df = pd.read_csv(train_path, nrows=500_000)  # increased from 200k
test_df = pd.read_csv(test_path)


## === cell 2
train_df.dropna(inplace=True)
train_df = train_df[train_df["fare_amount"] > 0]




## === cell 3
def haversine_distance(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6371 * c
    miles = km * 0.621371
    return miles


train_df["distance"] = haversine_distance(
    train_df["pickup_latitude"],
    train_df["pickup_longitude"],
    train_df["dropoff_latitude"],
    train_df["dropoff_longitude"],
)
test_df["distance"] = haversine_distance(
    test_df["pickup_latitude"],
    test_df["pickup_longitude"],
    test_df["dropoff_latitude"],
    test_df["dropoff_longitude"],
)

train_df = train_df[train_df["distance"] < 15]
train_df = train_df[
    (train_df["passenger_count"] != 0) & (train_df["passenger_count"] < 10)
]

train_dt = pd.to_datetime(train_df["pickup_datetime"])
train_df["hour"] = train_dt.dt.hour
train_df["minute"] = train_dt.dt.minute
train_df["dayofweek"] = train_dt.dt.dayofweek
train_df["month"] = train_dt.dt.month
train_df["year"] = train_dt.dt.year

test_dt = pd.to_datetime(test_df["pickup_datetime"])
test_df["hour"] = test_dt.dt.hour
test_df["minute"] = test_dt.dt.minute
test_df["dayofweek"] = test_dt.dt.dayofweek
test_df["month"] = test_dt.dt.month
test_df["year"] = test_dt.dt.year

train_df["log_distance"] = np.log1p(train_df["distance"])
test_df["log_distance"] = np.log1p(test_df["distance"])

train_df["distance_per_passenger"] = train_df["distance"] / train_df["passenger_count"]
test_df["distance_per_passenger"] = test_df["distance"] / test_df["passenger_count"]


## === cell 4
feat_cols = [
    "distance",
    "log_distance",
    "distance_per_passenger",
    "passenger_count",
    "hour",
    "minute",
    "dayofweek",
    "month",
    "year",
]

median_vals = train_df[feat_cols].median()
test_df[feat_cols] = test_df[feat_cols].fillna(median_vals)

X = train_df[feat_cols]
y = train_df["fare_amount"]
y_log = np.log1p(y)

X_train, X_val, y_train_log, y_val = train_test_split(
    X, y_log, test_size=0.2, random_state=42
)


## === cell 5
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=15,
    max_features="sqrt",
    n_jobs=5,
    random_state=42,
)
rf.fit(X_train, y_train_log)
rf_val_pred = np.expm1(rf.predict(X_val))
rf_rmse = mean_squared_error(y_val, rf_val_pred, squared=False)

gbr = GradientBoostingRegressor(
    n_estimators=300,  # slightly more boosting rounds for better fit
    learning_rate=0.05,
    max_depth=5,
    random_state=42,
)
gbr.fit(X_train, y_train_log)
gbr_val_pred = np.expm1(gbr.predict(X_val))
gbr_rmse = mean_squared_error(y_val, gbr_val_pred, squared=False)


## === cell 6
print(f"RF RMSE: {rf_rmse:.4f} | GBR RMSE: {gbr_rmse:.4f}")

test_features = test_df[feat_cols]

rf_test_log = rf.predict(test_features)
gbr_test_log = gbr.predict(test_features)
test_pred_log = (rf_test_log + gbr_test_log) / 2.0

test_pred = np.expm1(test_pred_log)
test_pred = np.clip(test_pred, a_min=0, a_max=None)

submission = pd.DataFrame({"key": test_df["key"], "fare_amount": test_pred})
submission.to_csv("submission.csv", index=False)

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/2524161937.py in <cell line: 0>()
      4 test_features = test_df[feat_cols]
      5 
----> 6 rf_test_log = rf.predict(test_features)
      7 gbr_test_log = gbr.predict(test_features)
      8 test_pred_log = (rf_test_log + gbr_test_log) / 2.0

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
   1189     def predict(self, X):
   1190         check_is_fitted(self)
-> 1191         return dispatch(
   1192             self,
   1193             "predict",

/usr/local/lib/python3.11/dist-packages/sklearnex/_device_offload.py in dispatch(obj, method_name, branches, *args, **kwargs)
    150             queue = QM.get_global_queue()
    151             patching_status.write_log(queue=queue, transferred_to_host=False)
--> 152             return branches["onedal"](obj, *hostargs, **hostkwargs, queue=queue)
    153         else:
    154             if sklearn_array_api and not has_usm_data:

/usr/local/lib/python3.11/dist-packages/sklearnex/ensemble/_forest.py in _onedal_predict(self, X, queue)
   1165             )  # Warning, order of dtype matters
   1166 
-> 1167         return self._onedal_estimator.predict(X, queue=queue)
   1168 
   1169     def _onedal_score(self, X, y, sample_weight=None, queue=None):

/usr/local/lib/python3.11/dist-packages/onedal/_device_offload.py in wrapper(self, *args, **kwargs)
     57         with QM.manage_global_queue(queue, *args) as queue:
     58             kwargs["queue"] = queue
---> 59             result = func(self, *args, **kwargs)
     60         return result
     61 

/usr/local/lib/python3.11/dist-packages/onedal/ensemble/forest.py in predict(self, X, queue)
    611     def predict(self, X, queue=None):
    612         _, xp, _ = _get_sycl_namespace(X)
--> 613         return xp.reshape(self._predict(X), -1)
    614 
    615 

/usr/local/lib/python3.11/dist-packages/onedal/ensemble/forest.py in _predict(self, X, hparams)
    381 
    382         if not use_raw_input:
--> 383             X = _check_array(
    384                 X,
    385                 dtype=[np.float64, np.float32],

/usr/local/lib/python3.11/dist-packages/onedal/utils/validation.py in _check_array(array, dtype, accept_sparse, order, copy, force_all_finite, ensure_2d, accept_large_sparse, _finite_keyword)
    166                 force_all_finite = False
    167         else:
--> 168             _daal4py_assert_all_finite(array)
    169             force_all_finite = False
    170     check_kwargs = {

/usr/local/lib/python3.11/dist-packages/daal4py/sklearn/utils/validation.py in _assert_all_finite(X, allow_nan, msg_dtype, estimator_name, input_name)
    129         if dt == np.float64:
    130             if not d4p.daal_assert_all_finite(x_for_daal, allow_nan, 0):
--> 131                 raise ValueError(err)
    132         elif dt == np.float32:
    133             if not d4p.daal_assert_all_finite(x_for_daal, allow_nan, 1):

ValueError: Input  contains NaN, infinity or a value too large for dtype('float64').
