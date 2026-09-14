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

bayesian-optimization==3.1.0
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

3.21281

# 6. Current score

4.30276

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 7.15707) has done: 'I fix the XGBoost training error by providing labels for the validation DMatrix, which resolves the size‑mismatch crash and lets the model be created. After that the subsequent cells that reference `model2`, `y_pred_test`, and the submission CSV run correctly, producing a valid “submission.csv” file with the required columns.'
- What this solution (achieved 4.4783) has done: 'I add a more accurate haversine distance feature, train the model on a log‑1p transformed target (and revert the predictions back), and keep the rest of the pipeline unchanged. These small changes usually lower RMSE without altering the core XGBoost logic, moving the score closer to the target.'
- What this solution (achieved 4.12443) has done: 'The fix adds data‑cleaning to remove NaNs and unreasonable fare values (which caused the XGBoost label error), ensures the training parameters are defined before they are used, and keeps the original feature engineering and log‑transform logic. These minimal changes let the pipeline run end‑to‑end and output a correctly formatted **submission.csv**, while preserving the core model logic and improving the RMSE toward the target.'
- What this solution (achieved 4.19949) has done: 'Implemented faster XGBoost training by enabling the histogram tree method and capping the maximum number of boosting rounds. These changes keep the exact training workflow, early‑stopping behavior, and feature engineering untouched while dramatically reducing computation time.'
- What this solution (achieved 4.27336) has done: 'I keep the overall pipeline unchanged but make a small regularization‑focused tweak to the XGBoost parameters: reduce tree depth, add a modest gamma, increase `min_child_weight`, and slightly lower the column‑ and row‑sampling rates. These changes usually curb over‑fitting, which should lower the validation RMSE and move the score closer to the target while preserving the core model, feature engineering, and log‑target handling.'
- What this solution (achieved 4.1125) has done: 'I remove the log‑1p target transformation and train the XGBoost model directly on the raw fare amounts. The rest of the pipeline (feature engineering, regularisation parameters, early‑stopping) stays unchanged, so we keep the core logic while expectedly lowering the RMSE toward the target.'
- What this solution (achieved 4.30276) has done: 'Implemented targeted speed‑ups while keeping the model, features, and training logic unchanged.  
Key changes:  
- Added explicit `nthread` to XGBoost to use all CPU cores efficiently.  
- Reduced the maximum boost rounds from 4000 to 2000 (early stopping still pick the optimal iteration).  
- Minor refactor in `transform` to compute Manhattan distances inline, avoiding repeated function call overhead.  
These adjustments cut runtime dramatically without altering any feature engineering, loss, or model architecture.'

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


def manhattan_dist(lat1, lon1, lat2, lon2):
    """Simple Manhattan distance in degrees (approximate)."""
    return np.abs(lat1 - lat2) + np.abs(lon1 - lon2)


def haversine_dist(lat1, lon1, lat2, lon2):
    """Haversine distance in kilometers."""
    R = 6371.0  # Earth radius in km
    lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
    lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c




## === cell 1
dtype_spec = {
    "fare_amount": "float32",
    "pickup_datetime": "object",  # keep as string for slicing
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int8",
}
df = pd.read_csv(
    "../input/train.csv",
    nrows=4_000_000,
    usecols=[1, 2, 3, 4, 5, 6, 7],  # fare_amount + features (no key)
    dtype=dtype_spec,
)

df["pickup_datetime"] = df["pickup_datetime"].str.slice(0, 16)
df["pickup_datetime"] = pd.to_datetime(
    df["pickup_datetime"], utc=True, format="%Y-%m-%d %H:%M"
)

df = df.dropna()
df = df[(df["fare_amount"] > 0) & (df["fare_amount"] < 500)]


def transform(data):
    data["hour"] = data["pickup_datetime"].dt.hour
    data["day"] = data["pickup_datetime"].dt.day
    data["month"] = data["pickup_datetime"].dt.month
    data["year"] = data["pickup_datetime"].dt.year

    data["hour_sin"] = np.sin(2 * np.pi * data["hour"] / 24)
    data["hour_cos"] = np.cos(2 * np.pi * data["hour"] / 24)

    data = data.drop("pickup_datetime", axis=1)

    nyc_lat, nyc_lon = 40.7141667, -74.0063889
    jfk_lat, jfk_lon = 40.6441666667, -73.7822222222
    ewr_lat, ewr_lon = 40.69, -74.175
    lgr_lat, lgr_lon = 40.77, -73.87

    data["distance_to_center"] = np.abs(data["pickup_latitude"] - nyc_lat) + np.abs(
        data["pickup_longitude"] - nyc_lon
    )
    data["pickup_distance_to_jfk"] = np.abs(data["pickup_latitude"] - jfk_lat) + np.abs(
        data["pickup_longitude"] - jfk_lon
    )
    data["dropoff_distance_to_jfk"] = np.abs(
        data["dropoff_latitude"] - jfk_lat
    ) + np.abs(data["dropoff_longitude"] - jfk_lon)
    data["pickup_distance_to_ewr"] = np.abs(data["pickup_latitude"] - ewr_lat) + np.abs(
        data["pickup_longitude"] - ewr_lon
    )
    data["dropoff_distance_to_ewr"] = np.abs(
        data["dropoff_latitude"] - ewr_lat
    ) + np.abs(data["dropoff_longitude"] - ewr_lon)
    data["pickup_distance_to_lgr"] = np.abs(data["pickup_latitude"] - lgr_lat) + np.abs(
        data["pickup_longitude"] - lgr_lon
    )
    data["dropoff_distance_to_lgr"] = np.abs(
        data["dropoff_latitude"] - lgr_lat
    ) + np.abs(data["dropoff_longitude"] - lgr_lon)

    data["long_dist"] = data["pickup_longitude"] - data["dropoff_longitude"]
    data["lat_dist"] = data["pickup_latitude"] - data["dropoff_latitude"]

    data["dist"] = np.abs(data["pickup_latitude"] - data["dropoff_latitude"]) + np.abs(
        data["pickup_longitude"] - data["dropoff_longitude"]
    )
    data["haversine_dist"] = haversine_dist(
        data["pickup_latitude"],
        data["pickup_longitude"],
        data["dropoff_latitude"],
        data["dropoff_longitude"],
    )
    data["haversine_to_manhattan"] = data["haversine_dist"] / (data["dist"] + 1e-6)
    data["haversine_sq"] = data["haversine_dist"] ** 2

    feature_cols = data.columns
    data[feature_cols] = data[feature_cols].astype(np.float32)
    return data


df = transform(df)




## === cell 2
X = df.drop("fare_amount", axis=1)
y_log = np.log1p(df["fare_amount"]).astype(np.float32)  # low‑precision label

X_train, X_valid, y_train_log, y_valid_log = train_test_split(
    X, y_log, test_size=0.2, random_state=42
)

dtrain = xgb.DMatrix(X_train, label=y_train_log)
dvalid = xgb.DMatrix(X_valid, label=y_valid_log)

params = {
    "objective": "reg:squarederror",
    "max_depth": 8,
    "eta": 0.01,
    "gamma": 0.1,
    "colsample_bytree": 0.9,
    "min_child_weight": 5,
    "subsample": 0.9,
    "eval_metric": "rmse",
    "seed": 42,
    "tree_method": "hist",
    "predictor": "cpu_predictor",
    "max_bin": 128,
    "nthread": 4,  # utilize multiple cores for faster training
}




## === cell 3
model2 = xgb.train(
    params,
    dtrain,
    num_boost_round=2000,  # reduced max rounds; early stopping still selects optimum
    evals=[(dvalid, "validation")],
    early_stopping_rounds=30,
    verbose_eval=False,
)

y_pred_log = model2.predict(dvalid)
y_pred = np.expm1(y_pred_log)

y_train_pred_log = model2.predict(dtrain)
y_train_pred = np.expm1(y_train_pred_log)

print("Validation RMSE:", np.sqrt(mean_squared_error(np.expm1(y_valid_log), y_pred)))
print(
    "Training RMSE:", np.sqrt(mean_squared_error(np.expm1(y_train_log), y_train_pred))
)




## === cell 4
test = pd.read_csv("../input/test.csv").set_index("key")
test["pickup_datetime"] = test["pickup_datetime"].str.slice(0, 16)
test["pickup_datetime"] = pd.to_datetime(
    test["pickup_datetime"], utc=True, format="%Y-%m-%d %H:%M"
)

test = transform(test)

dtest = xgb.DMatrix(test)
y_pred_test_log = model2.predict(dtest)
y_pred_test = np.expm1(y_pred_test_log)

submission = pd.DataFrame({"key": test.index, "fare_amount": y_pred_test})
submission.to_csv("submission.csv", index=False)
