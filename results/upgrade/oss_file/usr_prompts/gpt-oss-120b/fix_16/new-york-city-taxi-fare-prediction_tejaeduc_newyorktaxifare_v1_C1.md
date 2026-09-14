# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

3.9611

# 6. Current score

7.38189

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 5.77441) has done: 'I added an hour feature (which is predictive for fare), removed the unnecessary rounding of predictions, and tuned the XGBoost parameters slightly to improve RMSE while keeping the original modeling pipeline intact. These changes should move the score closer to the target without altering the core logic.'
- What this solution (achieved 5.90397) has done: 'Implemented the missing imports, data loading, feature engineering (distance & hour), XGBoost training, and proper alignment of test features. Added a validation split to check RMSE and ensured the final predictions are saved to **submission.csv** with the required columns `key` and `fare_amount`. All previous undefined variables are removed and the pipeline now runs end‑to‑end, producing a valid submission file.'
- What this solution (achieved 5.83279) has done: 'I increase the training sample size, add two inexpensive temporal features (day‑of‑week and month), and let XGBoost train a bit longer with early stopping. These changes keep the original pipeline intact while giving the model more data and predictive signals, which should lower the RMSE toward the target.'
- What this solution (achieved 6.57536) has done: 'Implemented two lightweight enhancements aimed at nudging the RMSE toward the target without altering the core modeling pipeline.  
1. Removed extreme fare outliers ( 0 ≤ fare ≤ 300 ) to reduce noise.  
2. Enriched the feature set with the raw pickup/drop‑off longitude and latitude columns, giving XGBoost more spatial signals.  
3. Slightly increased the maximum number of trees (to 1200) so early‑stopping can exploit the extra features, while keeping all other hyper‑parameters unchanged. These changes are minimal, preserve the original logic, and are expected to lower the validation RMSE, moving the score closer to the target.'
- What this solution (achieved 5.48065) has done: 'I add a simple log‑transform of the target (training on log1p fare and exponentiating predictions) and clip predictions to the realistic range [0, 300]. I also add an inexpensive “is_weekend” binary feature and include it in the model. These tweaks keep the original pipeline intact while giving the model a better‑behaved target and a modestly richer feature set, which should lower the validation RMSE and move the score toward the target.'
- What this solution (achieved 7.0478) has done: 'We trim the training input to a smaller yet representative subset (2 million rows) so that XGBoost can finish well within the 600 s limit while preserving the exact preprocessing, feature engineering, model type and evaluation logic. The change only reduces the amount of data loaded; all transformations, model architecture, and metrics remain unchanged, guaranteeing identical semantics on the sampled data.'
- What this solution (achieved 9.31745) has done: 'Implemented three targeted tweaks to move the RMSE closer to the target while keeping the core pipeline intact:  
1. Train the model on the raw `fare_amount` instead of a log‑transformed target, removing the extra exponentiation step that can bias large fares.  
2. Introduce a squared distance feature `distance_km_sq` to give XGBoost an additional non‑linear signal without altering the existing feature set.  
3. Update the validation and test prediction handling to reflect the raw‑target training. These minimal changes preserve the original modeling logic and should lower the validation RMSE toward the desired score.'
- What this solution (achieved 7.38189) has done: 'I train the model on a log‑transformed fare amount (log1p) and convert the predictions back with expm1 before clipping. This small change keeps the same feature engineering and XGBoost setup while reducing the effect of heavy‑tailed fare values, which tends to lower RMSE and move the score nearer the target. The rest of the pipeline (validation split, early stopping, submission file) remains unchanged.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

np.random.seed(42)

train_path = os.path.abspath("../input/train.csv")
test_path = os.path.abspath("../input/test.csv")

usecols = [
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
    "key",
]

test_usecols = [c for c in usecols if c != "fare_amount"]

dtype_map = {
    "fare_amount": np.float32,
    "pickup_longitude": np.float32,
    "pickup_latitude": np.float32,
    "dropoff_longitude": np.float32,
    "dropoff_latitude": np.float32,
    "passenger_count": np.int8,
    "key": "object",
}

df = pd.read_csv(
    train_path,
    usecols=usecols,
    dtype=dtype_map,
    parse_dates=["pickup_datetime"],
    nrows=2_000_000,
)

df.dropna(
    subset=[
        "fare_amount",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
    ],
    inplace=True,
)
df = df[(df["fare_amount"] >= 0) & (df["fare_amount"] <= 300)]

lon_min, lon_max = -74.5, -73.5
lat_min, lat_max = 40.5, 41.0
df = df.loc[
    (df["pickup_longitude"].between(lon_min, lon_max))
    & (df["dropoff_longitude"].between(lon_min, lon_max))
    & (df["pickup_latitude"].between(lat_min, lat_max))
    & (df["dropoff_latitude"].between(lat_min, lat_max))
]


def haversine_distance(lat1, lon1, lat2, lon2):
    """Vectorized haversine distance (km) between two points."""
    R = 6371.0
    lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
    lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    )
    return 2 * R * np.arcsin(np.sqrt(a))


df["distance_km"] = haversine_distance(
    df["pickup_latitude"],
    df["pickup_longitude"],
    df["dropoff_latitude"],
    df["dropoff_longitude"],
).astype(np.float32)

df["distance_km_sq"] = (df["distance_km"] ** 2).astype(np.float32)
df["log_distance"] = np.log1p(df["distance_km"]).astype(np.float32)
df["hour"] = df["pickup_datetime"].dt.hour.astype(np.int8)
df["dayofweek"] = df["pickup_datetime"].dt.dayofweek.astype(np.int8)
df["month"] = df["pickup_datetime"].dt.month.astype(np.int8)
df["is_weekend"] = (df["dayofweek"] >= 5).astype(np.int8)

feature_cols = [
    "distance_km",
    "distance_km_sq",
    "log_distance",
    "hour",
    "passenger_count",
    "dayofweek",
    "month",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "is_weekend",
]

X = df[feature_cols].to_numpy(copy=False, dtype=np.float32)

y_log = np.log1p(df["fare_amount"]).to_numpy(copy=False, dtype=np.float32)

X_train, X_val, y_train_log, y_val_log = train_test_split(
    X, y_log, test_size=0.2, random_state=42
)

model = XGBRegressor(
    n_estimators=1500,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    tree_method="hist",
    n_jobs=5,
    reg_lambda=1.0,
    random_state=42,
    verbosity=0,
    eval_metric="rmse",
)

model.fit(
    X_train,
    y_train_log,
    eval_set=[(X_val, y_val_log)],
    early_stopping_rounds=30,
    verbose=False,
)

val_pred_log = model.predict(X_val)
val_pred = np.expm1(val_pred_log)
val_pred = np.clip(val_pred, 0, 300)

val_rmse = np.sqrt(mean_squared_error(np.expm1(y_val_log), val_pred))
print(f"Validation RMSE: {val_rmse:.4f}")



## === cell 1
test_df = pd.read_csv(
    test_path,
    usecols=test_usecols,
    dtype=dtype_map,
    parse_dates=["pickup_datetime"],
)

test_df["distance_km"] = haversine_distance(
    test_df["pickup_latitude"],
    test_df["pickup_longitude"],
    test_df["dropoff_latitude"],
    test_df["dropoff_longitude"],
).astype(np.float32)

test_df["distance_km_sq"] = (test_df["distance_km"] ** 2).astype(np.float32)
test_df["log_distance"] = np.log1p(test_df["distance_km"]).astype(np.float32)
test_df["hour"] = test_df["pickup_datetime"].dt.hour.astype(np.int8)
test_df["dayofweek"] = test_df["pickup_datetime"].dt.dayofweek.astype(np.int8)
test_df["month"] = test_df["pickup_datetime"].dt.month.astype(np.int8)
test_df["is_weekend"] = (test_df["dayofweek"] >= 5).astype(np.int8)

X_test = test_df[feature_cols].to_numpy(copy=False, dtype=np.float32)



## === cell 2
test_pred_log = model.predict(X_test)
test_pred = np.expm1(test_pred_log)
test_pred = np.clip(test_pred, 0, 300)

submission = pd.DataFrame({"key": test_df["key"], "fare_amount": test_pred})
output_path = os.path.abspath("submission.csv")
submission.to_csv(output_path, index=False)
print(f"Submission file written to {output_path}")
