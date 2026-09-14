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

3.12

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

3.51454

# 6. Current score

4.27738

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 5.34116) has done: 'I fixed the haversine distance calculation (which was swapping latitude/longitude) and added a small log‑transform model with slightly tuned XGBoost parameters; this improves distance quality and lets the regressor work on a smoother target, which is expected to lower the RMSE toward the target score. The final prediction uses this updated model and writes a proper `submission.csv`.'
- What this solution (achieved 5.29195) has done: 'I keep the overall pipeline unchanged but improve the log‑transformed XGBoost model that generates the final submission. By increasing the number of trees and using a slightly lower learning rate we can obtain a modest gain in validation RMSE, moving the score closer to the target. The only change is in the definition of `log_model` (cell 66); the rest of the script stays the same and still writes a valid `submission.csv`.'
- What this solution (achieved 5.30754) has done: 'Implemented a fix for the feature‑mismatch error that prevented model prediction.  
The test data now includes the required **log_distance** column by selecting the full `features1` list (used during training). This change lets both the regular XGB model and the log‑transformed XGB model generate predictions and write a valid `submission.csv`. No other logic was altered, preserving the original pipeline and score‑related behavior.'
- What this solution (achieved 5.30298) has done: 'I add sinusoidal hour features (`sin_hour`, `cos_hour`) to better capture daily patterns, include them in the feature set used for training and prediction, and keep the rest of the pipeline unchanged. This small augmentation should lower the RMSE toward the target while preserving the original model logic.'
- What this solution (achieved 5.22106) has done: 'I keep the existing preprocessing and feature engineering unchanged and only modify the final modeling step. The log‑transformed XGBoost model that is already trained be combined with the earlier XGBRegressor (`xgb_model3`) by averaging their predictions both on the validation split and on the test set. This simple ensemble usually lowers RMSE modestly without altering the core pipeline, moving the score nearer to the target. The submission file is still written to `submission.csv` with the required columns.'
- What this solution (achieved 5.19945) has done: 'I increase the capacity of the main XGBoost model (more trees, lower learning rate) and add a tiny weight‑search on the validation set to blend the log‑transformed model with the raw model. This keeps the overall pipeline unchanged while giving a modest RMSE reduction, moving the score closer to the target.'
- What this solution (achieved 4.31648) has done: 'The script now reads a smaller sample (5 million rows) to keep the same preprocessing and model definitions while cutting the training workload roughly in half, allowing the full training‑and‑prediction pipeline to complete within the 600‑second limit. No core logic, model architecture, or evaluation steps are altered.'
- What this solution (achieved 4.27738) has done: 'I added a simple “day” feature from the pickup datetime, included it in the feature list, and modestly increased the capacity of both XGBoost models (more trees, slightly lower learning rates) to improve predictive power while staying within the original pipeline structure. These tweaks are expected to lower the validation RMSE and bring the overall score closer to the target without changing the core modeling approach.'

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error



## === cell 1
dtypes = {
    "key": "object",
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int8",
}
train_df = pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv",
    nrows=5_000_000,  # reduced sample to stay within time budget
    dtype=dtypes,
    parse_dates=False,  # parse later after cleaning
    low_memory=False,
)
test_df = pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv",
    dtype={k: v for k, v in dtypes.items() if k != "fare_amount"},
    parse_dates=False,
    low_memory=False,
)



## === cell 2
train_df = train_df.dropna()
train_df = train_df[train_df["fare_amount"] >= 0]

mask1 = (train_df["pickup_latitude"] < -90) | (train_df["pickup_latitude"] > 90)
mask2 = (train_df["pickup_longitude"] < -180) | (train_df["pickup_longitude"] > 180)
mask3 = (train_df["dropoff_latitude"] < -90) | (train_df["dropoff_latitude"] > 90)
mask4 = (train_df["dropoff_longitude"] < -180) | (train_df["dropoff_longitude"] > 180)
train_df = train_df[~(mask1 | mask2 | mask3 | mask4)]

mask = (train_df["passenger_count"] <= 0) | (train_df["passenger_count"] > 6)
train_df = train_df[~mask]




## === cell 3
def haversine(df):
    lon1, lat1 = df["pickup_longitude"], df["pickup_latitude"]
    lon2, lat2 = df["dropoff_longitude"], df["dropoff_latitude"]
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    a = (
        np.sin((lat2 - lat1) / 2.0) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2.0) ** 2
    )
    df["distance"] = 6371 * 2 * np.arcsin(np.sqrt(a))
    df["distance"] = df["distance"].astype(np.float32)


haversine(train_df)
haversine(test_df)

train_df["log_distance"] = np.log1p(train_df["distance"]).astype(np.float32)
test_df["log_distance"] = np.log1p(test_df["distance"]).astype(np.float32)



## === cell 4
train_df = train_df[~((train_df["distance"] > 0) & (train_df["fare_amount"] == 0))]



## === cell 5
train_df["pickup_datetime"] = pd.to_datetime(train_df["pickup_datetime"])
test_df["pickup_datetime"] = pd.to_datetime(test_df["pickup_datetime"])

train_df["year"] = train_df["pickup_datetime"].dt.year.astype(np.int16)
train_df["month"] = train_df["pickup_datetime"].dt.month.astype(np.int8)
train_df["weekday"] = train_df["pickup_datetime"].dt.dayofweek.astype(np.int8)
train_df["hour"] = train_df["pickup_datetime"].dt.hour.astype(np.int8)
train_df["sin_hour"] = np.sin(2 * np.pi * train_df["hour"] / 24).astype(np.float32)
train_df["cos_hour"] = np.cos(2 * np.pi * train_df["hour"] / 24).astype(np.float32)
train_df["day"] = train_df["pickup_datetime"].dt.day.astype(np.int8)

test_df["year"] = test_df["pickup_datetime"].dt.year.astype(np.int16)
test_df["month"] = test_df["pickup_datetime"].dt.month.astype(np.int8)
test_df["weekday"] = test_df["pickup_datetime"].dt.dayofweek.astype(np.int8)
test_df["hour"] = test_df["pickup_datetime"].dt.hour.astype(np.int8)
test_df["sin_hour"] = np.sin(2 * np.pi * test_df["hour"] / 24).astype(np.float32)
test_df["cos_hour"] = np.cos(2 * np.pi * test_df["hour"] / 24).astype(np.float32)
test_df["day"] = test_df["pickup_datetime"].dt.day.astype(np.int8)



## === cell 6
features1 = [
    "distance",
    "log_distance",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
    "year",
    "month",
    "weekday",
    "hour",
    "sin_hour",
    "cos_hour",
    "day",  # newly added feature
]
target = "fare_amount"

X1 = train_df[features1].astype(np.float32)
y1 = train_df[target].astype(np.float32)

X_train1, X_val1, y_train1, y_val1 = train_test_split(
    X1, y1, test_size=0.2, random_state=42
)




## === cell 7
def test_model(model, X_train, y_train, X_val, y_val):
    model.fit(X_train, y_train)
    y_pred_val = model.predict(X_val)
    y_pred_train = model.predict(X_train)
    mse_val = mean_squared_error(y_val, y_pred_val)
    mse_train = mean_squared_error(y_train, y_pred_train)
    rmse_val = np.sqrt(mse_val)
    rmse_train = np.sqrt(mse_train)
    print(f"{type(model).__name__}: RMSE on validation set: {rmse_val}")
    print(f"{type(model).__name__}: RMSE on train set: {rmse_train}")




## === cell 8
from xgboost import XGBRegressor

xgb_model3 = XGBRegressor(
    n_estimators=800,  # increased from 500
    learning_rate=0.03,  # slightly lower to keep training stable
    max_depth=5,
    n_jobs=-1,
    objective="reg:squarederror",
    random_state=42,
    tree_method="hist",
)
test_model(xgb_model3, X_train1, y_train1, X_val1, y_val1)



## === cell 9
log_model = XGBRegressor(
    n_estimators=1200,  # increased from 1000
    learning_rate=0.02,  # slightly lower learning rate
    max_depth=6,
    subsample=0.9,
    colsample_bytree=0.9,
    min_child_weight=1,
    objective="reg:squarederror",
    n_jobs=-1,
    random_state=42,
    tree_method="hist",
)

y_train_log = np.log1p(y_train1).astype(np.float32)
y_val_log = np.log1p(y_val1).astype(np.float32)

log_model.fit(X_train1.values, y_train_log)

val_pred_log = np.expm1(log_model.predict(X_val1.values))
val_pred_raw = xgb_model3.predict(X_val1.values)

weights = np.linspace(0, 1, 101)
best_rmse = np.inf
best_w = 0.5
for w in weights:
    ensemble_pred = w * val_pred_log + (1 - w) * val_pred_raw
    rmse = np.sqrt(mean_squared_error(y_val1, ensemble_pred))
    if rmse < best_rmse:
        best_rmse = rmse
        best_w = w
print(f"Best ensemble weight for log model: {best_w:.2f}, RMSE: {best_rmse:.5f}")



## === cell 10
X1_full = X1.values  # already float32
y1_full = y1.values

xgb_model_full = XGBRegressor(
    n_estimators=800,  # match validation model
    learning_rate=0.03,
    max_depth=5,
    n_jobs=-1,
    objective="reg:squarederror",
    random_state=42,
    tree_method="hist",
)
xgb_model_full.fit(X1_full, y1_full)

log_model_full = XGBRegressor(
    n_estimators=1200,  # match validation log model
    learning_rate=0.02,
    max_depth=6,
    subsample=0.9,
    colsample_bytree=0.9,
    min_child_weight=1,
    objective="reg:squarederror",
    n_jobs=-1,
    random_state=42,
    tree_method="hist",
)
log_model_full.fit(X1_full, np.log1p(y1_full).astype(np.float32))



## === cell 11
test_pred = test_df[features1].astype(np.float32)

test_pred_log = np.expm1(log_model_full.predict(test_pred.values))
test_pred_raw = xgb_model_full.predict(test_pred.values)

test_pred_ens = best_w * test_pred_log + (1 - best_w) * test_pred_raw

submission = pd.DataFrame(
    {"key": test_df["key"], "fare_amount": test_pred_ens},
    columns=["key", "fare_amount"],
)
submission.to_csv("submission.csv", index=False)
