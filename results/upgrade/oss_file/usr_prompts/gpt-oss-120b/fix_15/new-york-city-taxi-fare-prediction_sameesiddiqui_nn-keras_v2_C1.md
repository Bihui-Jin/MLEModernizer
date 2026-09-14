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

3.62636

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 5.66488) has done: 'The update replaces the single‑threaded `GradientBoostingRegressor` with the much faster `HistGradientBoostingRegressor`, which uses a histogram‑based algorithm that yields the same gradient‑boosting semantics but runs orders of magnitude quicker on large data. The rest of the pipeline—including feature engineering, one‑hot encoding, and evaluation—remains unchanged, preserving result accuracy.'
- What this solution (achieved 5.6255) has done: 'I increase the training sample size and adjust the HistGradientBoostingRegressor hyper‑parameters (more trees, smaller learning rate, deeper trees) which are small, targeted changes expected to lower the validation RMSE and bring the score nearer the target while keeping the original pipeline unchanged.'
- What this solution (achieved 5.53135) has done: 'I add a more informative distance feature (haversine distance) to the feature set and slightly strengthen the gradient‑boosting model (more trees, lower learning rate, deeper trees). These minimal changes are expected to reduce the RMSE, moving the score closer to the target while keeping the overall pipeline unchanged.'

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

datatypes = {
    "fare_amount": "float32",
    "pickup_datetime": "str",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "uint8",
}
usecols_train = list(datatypes.keys())  # columns needed for training (key not required)

train_df = pd.read_csv(
    "../input/train.csv",
    nrows=1_000_000,
    dtype=datatypes,
    usecols=usecols_train,
)




## === cell 1
def convert_to_one_hot(column, num_buckets, df, starting_index=0):
    """One‑hot encode an integer column with values in [starting_index, starting_index+num_buckets)."""
    df_size = df.shape[0]
    one_hots = np.zeros((df_size, num_buckets), dtype=np.float32)
    indices = df[column].values.astype(np.int32) - starting_index
    one_hots[np.arange(df_size), indices] = 1.0
    return one_hots


def distance_between_points(df):
    """Add haversine and Manhattan distance columns to the dataframe."""
    lat1 = np.radians(df["pickup_latitude"].values)
    lon1 = np.radians(df["pickup_longitude"].values)
    lat2 = np.radians(df["dropoff_latitude"].values)
    lon2 = np.radians(df["dropoff_longitude"].values)

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    earth_radius_km = 6371.0
    haversine = earth_radius_km * c
    df["haversine_dist"] = haversine.astype(np.float32)

    lat_km = 111.32
    lon_km = 111.32 * np.cos((lat1 + lat2) / 2)
    manhattan = (np.abs(dlat) * lat_km) + (np.abs(dlon) * lon_km)
    df["manhattan_dist"] = manhattan.astype(np.float32)


def extract_date_details(df):
    """Parse pickup_datetime and add year and hour columns."""
    dt = pd.to_datetime(
        df["pickup_datetime"], format="%Y-%m-%d %H:%M:%S", errors="coerce"
    )
    df["year"] = dt.dt.year.astype(np.int16)
    df["hour"] = dt.dt.hour.astype(np.int16)


_quantiles_cache = {}


def _compute_quantiles(column):
    """Cache 10‑quantile cut points for a given column based on the training dataframe."""
    if column not in _quantiles_cache:
        _quantiles_cache[column] = np.quantile(
            train_df[column].values, [0.1 * i for i in range(1, 10)]
        )
    return _quantiles_cache[column]


def bucketize_feature(df, column):
    """Return integer bucket indices (0‑9) for the column based on training quantiles."""
    quantiles = _compute_quantiles(column)
    binned = np.digitize(df[column].values, quantiles, right=False)
    binned = np.clip(binned, 0, 9).astype(np.int8)
    return binned


def feature_cross(a1, a2):
    """Cross two 0‑9 integer vectors into a 100‑dim one‑hot matrix."""
    rows = a1.shape[0]
    cross = np.zeros((rows, 100), dtype=np.float32)
    cross[np.arange(rows), (a1 * 10) + a2] = 1.0
    return cross


def extract_features(df):
    """Create the feature matrix matching the training pipeline."""
    distance_between_points(df)
    extract_date_details(df)

    p_long = bucketize_feature(df, "pickup_longitude")
    p_lat = bucketize_feature(df, "pickup_latitude")
    d_long = bucketize_feature(df, "dropoff_longitude")
    d_lat = bucketize_feature(df, "dropoff_latitude")

    p_lat_x_long = feature_cross(p_lat, p_long)
    d_lat_x_long = feature_cross(d_lat, d_long)

    year_onehot = convert_to_one_hot("year", 7, df, 2009)  # years 2009‑2015 inclusive
    hour_onehot = convert_to_one_hot("hour", 24, df, 0)

    n_rows = len(df)
    total_features = (
        p_lat_x_long.shape[1]
        + d_lat_x_long.shape[1]
        + year_onehot.shape[1]
        + hour_onehot.shape[1]
        + 3  # manhattan, haversine, passenger_count
    )
    X = np.empty((n_rows, total_features), dtype=np.float32)

    idx = 0
    X[:, idx : idx + p_lat_x_long.shape[1]] = p_lat_x_long
    idx += p_lat_x_long.shape[1]

    X[:, idx : idx + d_lat_x_long.shape[1]] = d_lat_x_long
    idx += d_lat_x_long.shape[1]

    X[:, idx : idx + year_onehot.shape[1]] = year_onehot
    idx += year_onehot.shape[1]

    X[:, idx : idx + hour_onehot.shape[1]] = hour_onehot
    idx += hour_onehot.shape[1]

    X[:, idx] = df["manhattan_dist"].values.astype(np.float32)
    idx += 1
    X[:, idx] = df["haversine_dist"].values.astype(np.float32)
    idx += 1
    X[:, idx] = df["passenger_count"].values.astype(np.float32)

    return X




## === cell 2
train_X = extract_features(train_df.copy())
train_y = train_df["fare_amount"].values.astype(np.float32)

gbr = HistGradientBoostingRegressor(
    max_iter=1200,
    learning_rate=0.015,
    max_depth=12,
    random_state=42,
    early_stopping=False,
)

gbr.fit(train_X, train_y)




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
IntCastingNaNError                        Traceback (most recent call last)
/tmp/ipykernel_55/2505850690.py in <cell line: 0>()
      1 # Prepare training data
----> 2 train_X = extract_features(train_df.copy())
      3 train_y = train_df["fare_amount"].values.astype(np.float32)
      4 
      5 # Model definition (same hyper‑parameters as the original solution)

/tmp/ipykernel_55/1503696058.py in extract_features(df)
     72     """Create the feature matrix matching the training pipeline."""
     73     distance_between_points(df)
---> 74     extract_date_details(df)
     75 
     76     p_long = bucketize_feature(df, "pickup_longitude")

/tmp/ipykernel_55/1503696058.py in extract_date_details(df)
     37         df["pickup_datetime"], format="%Y-%m-%d %H:%M:%S", errors="coerce"
     38     )
---> 39     df["year"] = dt.dt.year.astype(np.int16)
     40     df["hour"] = dt.dt.hour.astype(np.int16)
     41 

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in astype(self, dtype, copy, errors)
   6641         else:
   6642             # else, only a single dtype is given
-> 6643             new_data = self._mgr.astype(dtype=dtype, copy=copy, errors=errors)
   6644             res = self._constructor_from_mgr(new_data, axes=new_data.axes)
   6645             return res.__finalize__(self, method="astype")

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/managers.py in astype(self, dtype, copy, errors)
    428             copy = False
    429 
--> 430         return self.apply(
    431             "astype",
    432             dtype=dtype,

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/managers.py in apply(self, f, align_keys, **kwargs)
    361                 applied = b.apply(f, **kwargs)
    362             else:
--> 363                 applied = getattr(b, f)(**kwargs)
    364             result_blocks = extend_blocks(applied, result_blocks)
    365 

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/blocks.py in astype(self, dtype, copy, errors, using_cow, squeeze)
    756             values = values[0, :]  # type: ignore[call-overload]
    757 
--> 758         new_values = astype_array_safe(values, dtype, copy=copy, errors=errors)
    759 
    760         new_values = maybe_coerce_values(new_values)

/usr/local/lib/python3.11/dist-packages/pandas/core/dtypes/astype.py in astype_array_safe(values, dtype, copy, errors)
    235 
    236     try:
--> 237         new_values = astype_array(values, dtype, copy=copy)
    238     except (ValueError, TypeError):
    239         # e.g. _astype_nansafe can fail on object-dtype of strings

/usr/local/lib/python3.11/dist-packages/pandas/core/dtypes/astype.py in astype_array(values, dtype, copy)
    180 
    181     else:
--> 182         values = _astype_nansafe(values, dtype, copy=copy)
    183 
    184     # in pandas we don't store numpy str dtypes, so convert to object

/usr/local/lib/python3.11/dist-packages/pandas/core/dtypes/astype.py in _astype_nansafe(arr, dtype, copy, skipna)
     99 
    100     elif np.issubdtype(arr.dtype, np.floating) and dtype.kind in "iu":
--> 101         return _astype_float_to_int_nansafe(arr, dtype, copy)
    102 
    103     elif arr.dtype == object:

/usr/local/lib/python3.11/dist-packages/pandas/core/dtypes/astype.py in _astype_float_to_int_nansafe(values, dtype, copy)
    143     """
    144     if not np.isfinite(values).all():
--> 145         raise IntCastingNaNError(
    146             "Cannot convert non-finite values (NA or inf) to integer"
    147         )

IntCastingNaNError: Cannot convert non-finite values (NA or inf) to integer

## === cell 3
datatypes_test = datatypes.copy()
del datatypes_test["fare_amount"]  # test file does not contain the label
usecols_test = list(datatypes_test.keys())

test_df = pd.read_csv(
    "../input/test.csv",
    dtype=datatypes_test,
    usecols=usecols_test,
)

test_X = extract_features(test_df.copy())
test_pred = gbr.predict(test_X)

submission = pd.DataFrame({"key": test_df["key"], "fare_amount": test_pred})

submission.to_csv("submission.csv", index=False)

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
IntCastingNaNError                        Traceback (most recent call last)
/tmp/ipykernel_55/2769156308.py in <cell line: 0>()
     11 
     12 # Build test features and predict
---> 13 test_X = extract_features(test_df.copy())
     14 test_pred = gbr.predict(test_X)
     15 

/tmp/ipykernel_55/1503696058.py in extract_features(df)
     72     """Create the feature matrix matching the training pipeline."""
     73     distance_between_points(df)
---> 74     extract_date_details(df)
     75 
     76     p_long = bucketize_feature(df, "pickup_longitude")

/tmp/ipykernel_55/1503696058.py in extract_date_details(df)
     37         df["pickup_datetime"], format="%Y-%m-%d %H:%M:%S", errors="coerce"
     38     )
---> 39     df["year"] = dt.dt.year.astype(np.int16)
     40     df["hour"] = dt.dt.hour.astype(np.int16)
     41 

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in astype(self, dtype, copy, errors)
   6641         else:
   6642             # else, only a single dtype is given
-> 6643             new_data = self._mgr.astype(dtype=dtype, copy=copy, errors=errors)
   6644             res = self._constructor_from_mgr(new_data, axes=new_data.axes)
   6645             return res.__finalize__(self, method="astype")

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/managers.py in astype(self, dtype, copy, errors)
    428             copy = False
    429 
--> 430         return self.apply(
    431             "astype",
    432             dtype=dtype,

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/managers.py in apply(self, f, align_keys, **kwargs)
    361                 applied = b.apply(f, **kwargs)
    362             else:
--> 363                 applied = getattr(b, f)(**kwargs)
    364             result_blocks = extend_blocks(applied, result_blocks)
    365 

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/blocks.py in astype(self, dtype, copy, errors, using_cow, squeeze)
    756             values = values[0, :]  # type: ignore[call-overload]
    757 
--> 758         new_values = astype_array_safe(values, dtype, copy=copy, errors=errors)
    759 
    760         new_values = maybe_coerce_values(new_values)

/usr/local/lib/python3.11/dist-packages/pandas/core/dtypes/astype.py in astype_array_safe(values, dtype, copy, errors)
    235 
    236     try:
--> 237         new_values = astype_array(values, dtype, copy=copy)
    238     except (ValueError, TypeError):
    239         # e.g. _astype_nansafe can fail on object-dtype of strings

/usr/local/lib/python3.11/dist-packages/pandas/core/dtypes/astype.py in astype_array(values, dtype, copy)
    180 
    181     else:
--> 182         values = _astype_nansafe(values, dtype, copy=copy)
    183 
    184     # in pandas we don't store numpy str dtypes, so convert to object

/usr/local/lib/python3.11/dist-packages/pandas/core/dtypes/astype.py in _astype_nansafe(arr, dtype, copy, skipna)
     99 
    100     elif np.issubdtype(arr.dtype, np.floating) and dtype.kind in "iu":
--> 101         return _astype_float_to_int_nansafe(arr, dtype, copy)
    102 
    103     elif arr.dtype == object:

/usr/local/lib/python3.11/dist-packages/pandas/core/dtypes/astype.py in _astype_float_to_int_nansafe(values, dtype, copy)
    143     """
    144     if not np.isfinite(values).all():
--> 145         raise IntCastingNaNError(
    146             "Cannot convert non-finite values (NA or inf) to integer"
    147         )

IntCastingNaNError: Cannot convert non-finite values (NA or inf) to integer
