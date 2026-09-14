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

3.12

# 3. Installed packages

geopandas==0.14.4
joblib==1.5.2
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

# 5. Code solution

## === cell 0
import os
import gc
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

os.environ["OMP_NUM_THREADS"] = str(os.cpu_count() or 1)

dtypes = {
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int8",
}
train_path = "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv"
test_path = "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv"

train_usecols = [
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
test_usecols = [
    "key",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]

ny_lat_min, ny_lat_max = 40.4772, 45.0153
ny_lon_min, ny_lon_max = -79.7624, -71.7517


def process_chunk(chunk):
    """Filter invalid rows and build feature matrix + target vector."""
    mask = (
        chunk["pickup_longitude"].between(ny_lon_min, ny_lon_max)
        & chunk["pickup_latitude"].between(ny_lat_min, ny_lat_max)
        & chunk["dropoff_longitude"].between(ny_lon_min, ny_lon_max)
        & chunk["dropoff_latitude"].between(ny_lat_min, ny_lat_max)
        & chunk["passenger_count"].between(1, 6)
        & chunk["fare_amount"].between(1, 200)
    )
    filtered = chunk.loc[mask].dropna()
    dt = filtered["pickup_datetime"]
    year = dt.dt.year.to_numpy(copy=False).astype(np.float32)
    month = dt.dt.month.to_numpy(copy=False).astype(np.float32)
    day = dt.dt.day.to_numpy(copy=False).astype(np.float32)
    hour = dt.dt.hour.to_numpy(copy=False).astype(np.float32)
    weekday = dt.dt.weekday.to_numpy(copy=False).astype(np.float32)

    numeric_part = (
        filtered[
            [
                "pickup_longitude",
                "pickup_latitude",
                "dropoff_longitude",
                "dropoff_latitude",
                "passenger_count",
            ]
        ]
        .to_numpy(copy=False)
        .astype(np.float32)
    )

    feats = np.column_stack((numeric_part, year, month, day, hour, weekday))
    target = filtered["fare_amount"].to_numpy(copy=False).astype(np.float32)
    return feats, target


all_feats = []
all_targs = []

rng = np.random.default_rng(42)  # deterministic shuffling later

for chunk in pd.read_csv(
    train_path,
    dtype=dtypes,
    usecols=train_usecols,
    parse_dates=["pickup_datetime"],
    chunksize=10_000_000,
):
    feats, targ = process_chunk(chunk)
    all_feats.append(feats)
    all_targs.append(targ)
    del chunk, feats, targ
    gc.collect()

X_all = np.concatenate(all_feats, axis=0)
y_all_raw = np.concatenate(all_targs, axis=0)

del all_feats, all_targs
gc.collect()

perm = rng.permutation(X_all.shape[0])
split_idx = int(0.8 * X_all.shape[0])
train_idx, val_idx = perm[:split_idx], perm[split_idx:]

X_train = X_all[train_idx]
y_train_raw = y_all_raw[train_idx]
X_val = X_all[val_idx]
y_val_raw = y_all_raw[val_idx]

del X_all, y_all_raw, perm, train_idx, val_idx
gc.collect()

test_df = pd.read_csv(
    test_path,
    dtype={k: v for k, v in dtypes.items() if k != "fare_amount"},
    usecols=test_usecols,
    parse_dates=["pickup_datetime"],
)

dt_test = test_df["pickup_datetime"]
year = dt_test.dt.year.to_numpy(copy=False).astype(np.float32)
month = dt_test.dt.month.to_numpy(copy=False).astype(np.float32)
day = dt_test.dt.day.to_numpy(copy=False).astype(np.float32)
hour = dt_test.dt.hour.to_numpy(copy=False).astype(np.float32)
weekday = dt_test.dt.weekday.to_numpy(copy=False).astype(np.float32)

numeric_test = (
    test_df[
        [
            "pickup_longitude",
            "pickup_latitude",
            "dropoff_longitude",
            "dropoff_latitude",
            "passenger_count",
        ]
    ]
    .to_numpy(copy=False)
    .astype(np.float32)
)

test_features = np.column_stack((numeric_test, year, month, day, hour, weekday))




## === cell 1
y_train = np.log1p(y_train_raw.astype(np.float32))
y_val = np.log1p(y_val_raw.astype(np.float32))

del y_train_raw, y_val_raw
gc.collect()

xgb_model = XGBRegressor(
    objective="reg:squarederror",
    learning_rate=0.1,
    n_estimators=300,  # reduced number of trees for speed
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    n_jobs=-1,
    random_state=42,
    tree_method="hist",
    max_bin=64,  # fewer bins → faster histogram construction
    grow_policy="lossguide",  # limit tree growth overhead
    max_leaves=256,  # cap leaf count for quicker training
    eval_metric="rmse",
)

xgb_model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=50,
    verbose=False,
)

val_pred = np.expm1(xgb_model.predict(X_val))
val_rmse = mean_squared_error(y_val_raw, val_pred, squared=False)
print("Validation RMSE:", val_rmse)

del X_train, X_val, y_train, y_val, val_pred
gc.collect()




## === cell 2
test_pred = np.expm1(xgb_model.predict(test_features))
submission = pd.DataFrame({"key": test_df["key"], "fare_amount": test_pred})
submission.to_csv("submission.csv", index=False)
print("Submission written to submission.csv")
