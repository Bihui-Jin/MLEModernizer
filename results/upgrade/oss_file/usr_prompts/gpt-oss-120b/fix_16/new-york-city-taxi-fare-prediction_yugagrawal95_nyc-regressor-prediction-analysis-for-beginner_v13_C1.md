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
lightgbm==4.6.0
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

3.62101

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 4.77064) has done: 'I ensure the prediction array matches the original test keys by keeping a copy of the original keys, filtering the key series alongside the test rows that are removed, and then building a full‑size submission where any rows that were dropped receive the overall mean fare as a fallback. This resolves the length‑mismatch error and guarantees a valid `submission.csv` with the correct columns.'
- What this solution (achieved 4.39435) has done: 'I increase the training sample size (read more rows) and strengthen the LightGBM model by using more leaves and estimators, which should lower the validation RMSE and move the score closer to the target while keeping the overall workflow unchanged.'
- What this solution (achieved 4.11726) has done: 'I increase the training sample to 500 k rows, add a simple interaction feature `passenger_distance = distance * passenger_count`, and give LightGBM a larger capacity (more leaves, more trees, lower learning rate). These modest changes keep the original pipeline intact while expected to lower the validation RMSE and move the score closer to the target.'
- What this solution (achieved 4.15303) has done: 'I speed up the notebook by removing the expensive plotting commands and by lowering the estimator counts for the RandomForest and LightGBM models (keeping the same model types). These changes cut down on unnecessary I/O and reduce the heavy training loops while preserving the overall algorithm and feature engineering.'
- What this solution (achieved 4.08886) has done: 'The updates keep every modeling step unchanged but eliminate costly DataFrame‑to‑NumPy conversions inside the training loops. By converting the feature matrices and target vectors to contiguous ``float32`` NumPy arrays once, both scikit‑learn’s RandomForest and LightGBM can operate on memory‑efficient structures, cutting down Python‑level overhead and fitting time while preserving exact model configurations and evaluation logic.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import lightgbm as lgb



## === cell 1
use_cols = [
    "key",
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
dtype_map = {
    "key": "object",
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "uint8",
}
train_data = pd.read_csv(
    "../input/train.csv",
    usecols=use_cols,
    dtype=dtype_map,
    nrows=2_000_000,
    parse_dates=["pickup_datetime"],
    infer_datetime_format=True,
)
test_data = pd.read_csv(
    "../input/test.csv",
    usecols=[c for c in use_cols if c != "fare_amount"],
    dtype={k: v for k, v in dtype_map.items() if k != "fare_amount"},
    parse_dates=["pickup_datetime"],
    infer_datetime_format=True,
)



## === cell 2
train_data = train_data.dropna()
train_data = train_data[train_data.fare_amount < 400]
train_data = train_data[train_data.passenger_count <= 6]

lat_min, lat_max = test_data.pickup_latitude.min(), test_data.pickup_latitude.max()
lon_min, lon_max = test_data.pickup_longitude.min(), test_data.pickup_longitude.max()
drop_lat_min, drop_lat_max = (
    test_data.dropoff_latitude.min(),
    test_data.dropoff_latitude.max(),
)
drop_lon_min, drop_lon_max = (
    test_data.dropoff_longitude.min(),
    test_data.dropoff_longitude.max(),
)

mask = (
    train_data["pickup_latitude"].between(-90, 90)
    & train_data["pickup_longitude"].between(-180, 180)
    & train_data["dropoff_latitude"].between(-90, 90)
    & train_data["dropoff_longitude"].between(-180, 180)
    & train_data["pickup_latitude"].between(lat_min, lat_max)
    & train_data["pickup_longitude"].between(lon_min, lon_max)
    & train_data["dropoff_latitude"].between(drop_lat_min, drop_lat_max)
    & train_data["dropoff_longitude"].between(drop_lon_min, drop_lon_max)
)
train_data = train_data.loc[mask].copy()

zero_mask = (
    (train_data["pickup_latitude"] != 0)
    & (train_data["pickup_longitude"] != 0)
    & (train_data["dropoff_latitude"] != 0)
    & (train_data["dropoff_longitude"] != 0)
)
train_data = train_data.loc[zero_mask].copy()

zero_mask_test = (
    (test_data["pickup_latitude"] != 0)
    & (test_data["pickup_longitude"] != 0)
    & (test_data["dropoff_latitude"] != 0)
    & (test_data["dropoff_longitude"] != 0)
)
test_data = test_data.loc[zero_mask_test].copy()




## === cell 3
def degree_to_rad(degree):
    return degree * (np.pi / 180)


def haversine_distance(pickup_lat, pickup_long, dropoff_lat, dropoff_long):
    from_lat = degree_to_rad(pickup_lat)
    from_long = degree_to_rad(pickup_long)
    to_lat = degree_to_rad(dropoff_lat)
    to_long = degree_to_rad(dropoff_long)

    lat_diff = to_lat - from_lat
    long_diff = to_long - from_long

    a = (
        np.sin(lat_diff / 2) ** 2
        + np.cos(from_lat) * np.cos(to_lat) * np.sin(long_diff / 2) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return 6371.01 * c  # Earth radius in km


train_data["distance"] = haversine_distance(
    train_data.pickup_latitude,
    train_data.pickup_longitude,
    train_data.dropoff_latitude,
    train_data.dropoff_longitude,
)
test_data["distance"] = haversine_distance(
    test_data.pickup_latitude,
    test_data.pickup_longitude,
    test_data.dropoff_latitude,
    test_data.dropoff_longitude,
)

for df in (train_data, test_data):
    df["Year"] = df["pickup_datetime"].dt.year
    df["Month"] = df["pickup_datetime"].dt.month
    df["Date"] = df["pickup_datetime"].dt.day
    df["Day_of_Week"] = df["pickup_datetime"].dt.dayofweek
    df["Hour"] = df["pickup_datetime"].dt.hour
    df["passenger_distance"] = df["passenger_count"] * df["distance"]
    df["log_distance"] = np.log1p(df["distance"])
    df["hour_sin"] = np.sin(2 * np.pi * df["Hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["Hour"] / 24)



## === cell 4
test_keys_original = test_data["key"].copy()
test_data = test_data.drop(columns=["key", "pickup_datetime"])
train_data = train_data.drop(columns=["key", "pickup_datetime"])



## === cell 5
X = train_data.drop(columns="fare_amount").values.astype("float32")
y = train_data["fare_amount"].values.astype("float32")
X_test = test_data.values.astype("float32")



## === cell 6
X_train_np, X_val_np, y_train_np, y_val_np = train_test_split(
    X, y, test_size=0.3, random_state=0
)



## === cell 7
rf = RandomForestRegressor(max_depth=200, n_estimators=50, n_jobs=-1, random_state=0)
rf.fit(X_train_np, y_train_np)
rf_pred = rf.predict(X_val_np)
rf_rmse = np.sqrt(mean_squared_error(y_val_np, rf_pred))
print("RandomForest RMSE:", rf_rmse)



## === cell 8
model_lgb = lgb.LGBMRegressor(
    objective="regression",
    num_leaves=2047,
    learning_rate=0.005,
    n_estimators=2000,
    max_bin=255,
    bagging_fraction=0.8,
    bagging_freq=5,
    feature_fraction=0.8,
    random_state=42,
    n_jobs=-1,
)
model_lgb.fit(
    X_train_np,
    y_train_np,
    eval_set=[(X_val_np, y_val_np)],
    eval_metric="rmse",
    verbose=-1,
)
lgb_pred = model_lgb.predict(X_val_np)
lgb_rmse = np.sqrt(mean_squared_error(y_val_np, lgb_pred))
print("LightGBM RMSE:", lgb_rmse)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/692095051.py in <cell line: 0>()
     11     n_jobs=-1,
     12 )
---> 13 model_lgb.fit(
     14     X_train_np,
     15     y_train_np,

TypeError: LGBMRegressor.fit() got an unexpected keyword argument 'verbose'

## === cell 9
best_model = model_lgb if lgb_rmse < rf_rmse else rf
best_model.fit(X, y)



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3013560443.py in <cell line: 0>()
----> 1 best_model = model_lgb if lgb_rmse < rf_rmse else rf
      2 best_model.fit(X, y)
      3 

NameError: name 'lgb_rmse' is not defined

## === cell 10
test_preds = best_model.predict(X_test)

full_submission = pd.DataFrame({"key": test_keys_original, "fare_amount": np.nan})
full_submission.loc[full_submission["key"].isin(test_data["key"]), "fare_amount"] = (
    test_preds
)
full_submission["fare_amount"].fillna(y.mean(), inplace=True)
full_submission.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1230321585.py in <cell line: 0>()
----> 1 test_preds = best_model.predict(X_test)
      2 
      3 full_submission = pd.DataFrame({"key": test_keys_original, "fare_amount": np.nan})
      4 full_submission.loc[full_submission["key"].isin(test_data["key"]), "fare_amount"] = (
      5     test_preds

NameError: name 'best_model' is not defined
