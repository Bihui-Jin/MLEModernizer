# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
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

# 5. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np
from sklearnex.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer  # fast median imputer

np.random.seed(42)




## === cell 1
usecols_train = [
    "key",
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]

dtype_train = {
    "key": "object",
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int8",
}
train = pd.read_csv(
    "../input/train.csv",
    nrows=2_000_000,
    usecols=usecols_train,
    dtype=dtype_train,
    parse_dates=["pickup_datetime"],
)

usecols_test = [
    "key",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
dtype_test = {
    "key": "object",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int8",
}
test = pd.read_csv(
    "../input/test.csv",
    usecols=usecols_test,
    dtype=dtype_test,
    parse_dates=["pickup_datetime"],
)




## === cell 2
def haversine_vec(lon1, lat1, lon2, lat2):
    """Vectorized haversine distance (km) using NumPy."""
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371 * c


def add_features(df):
    dt = df["pickup_datetime"]
    df["pickup_hour"] = dt.dt.hour.astype("int8")
    df["pickup_dayofweek"] = dt.dt.dayofweek.astype("int8")
    df["pickup_month"] = dt.dt.month.astype("int8")

    df["distance"] = haversine_vec(
        df["pickup_longitude"].values,
        df["pickup_latitude"].values,
        df["dropoff_longitude"].values,
        df["dropoff_latitude"].values,
    ).astype("float32")

    df["distance_log"] = np.log1p(df["distance"]).astype("float32")

    df["lat_diff"] = (df["dropoff_latitude"] - df["pickup_latitude"]).astype("float32")
    df["lon_diff"] = (df["dropoff_longitude"] - df["pickup_longitude"]).astype(
        "float32"
    )
    df["euclidean_dist"] = np.sqrt(df["lat_diff"] ** 2 + df["lon_diff"] ** 2).astype(
        "float32"
    )

    hour_rad = 2 * np.pi * df["pickup_hour"] / 24.0
    dow_rad = 2 * np.pi * df["pickup_dayofweek"] / 7.0
    df["hour_sin"] = np.sin(hour_rad).astype("float32")
    df["hour_cos"] = np.cos(hour_rad).astype("float32")
    df["dow_sin"] = np.sin(dow_rad).astype("float32")
    df["dow_cos"] = np.cos(dow_rad).astype("float32")

    return df


train = add_features(train)
test = add_features(test)

train = train.drop(columns=["pickup_datetime"])
test = test.drop(columns=["pickup_datetime"])




## === cell 3
target_raw = train["fare_amount"].values

valid_mask = (
    (target_raw > 0)
    & np.isfinite(target_raw)
    & (train["distance"] <= 200)  # distance filter
)

features_df = (
    train.drop(columns=["fare_amount", "key"]).loc[valid_mask].astype("float32")
)
target = np.log1p(target_raw[valid_mask])

X_test_df = test.drop(columns=["key"]).astype("float32")

features_arr = features_df.to_numpy(dtype=np.float32, copy=False)
test_arr = X_test_df.to_numpy(dtype=np.float32, copy=False)

imputer = SimpleImputer(strategy="median")
X_imp = imputer.fit_transform(features_arr)  # shape (n_samples, n_features)
X_test_imp = imputer.transform(test_arr)  # test matrix

del features_df, X_test_df, features_arr, test_arr
import gc

gc.collect()




## === cell 4
rf = RandomForestRegressor(
    random_state=42,
    n_estimators=400,  # more trees for stronger ensemble
    max_depth=35,  # allow a bit deeper trees
    n_jobs=-1,
)
rf.fit(X_imp, target)




## === cell 5
test_pred_log = rf.predict(X_test_imp)
test_pred = np.expm1(test_pred_log)  # inverse of log1p
test_pred = np.maximum(test_pred, 0.0)  # ensure non‑negative fares

submission = pd.DataFrame(
    {
        "key": test["key"],
        "fare_amount": test_pred,
    }
)
output_path = "submission.csv"
submission.to_csv(output_path, index=False)
print(f"Submission written to {output_path}")
