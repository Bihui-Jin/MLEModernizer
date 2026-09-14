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

3.47309

# 6. Current score

4.82911

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 8.79815) has done: 'I fix the runtime error caused by using the nonexistent attribute `best_ntree_limit` on the XGBoost Booster. The prediction simply use the model’s default behaviour (all trees) which resolves the undefined variable and allows the script to generate a proper `taxi_fare_submission.csv` file.'
- What this solution (achieved 5.07565) has done: 'I keep the overall workflow and feature engineering unchanged, but improve the XGBoost model by using the up‑to‑date regression objective, adding a modest learning rate, depth and subsampling, increasing the number of boosting rounds with proper early stopping, and training on the log‑transformed fare amount (then back‑transforming the predictions). These tweaks are minimal yet directly target a lower RMSE, moving the score closer to the desired 3.47 ± 10 % range.'
- What this solution (achieved 8.70071) has done: 'I train the XGBoost model on the original fare amount (removing the log‑transform) and lower the learning rate while allowing more boosting rounds with early stopping. This aligns the training objective directly with the RMSE metric used for evaluation, so the model’s predictions are calibrated to the target scale. The prediction step is also adjusted to use the raw output without back‑transforming. These minimal changes keep the overall workflow and feature engineering intact while moving the validation RMSE closer to the target value.'
- What this solution (achieved 5.55844) has done: 'I re‑introduce a log‑transform of the target fare (using `log1p` and `expm1` for back‑transform) and slightly tighten the XGBoost hyper‑parameters (lower learning rate, deeper trees, higher subsample/colsample). These changes keep the overall pipeline and feature engineering intact while improving prediction calibration, which should lower the RMSE toward the target value.'
- What this solution (achieved 8.65348) has done: 'The update removes the log‑transform of the target, training XGBoost directly on the raw fare amount and using the raw model predictions for the submission. This aligns the training objective with the competition’s RMSE metric, which is expected to lower the validation score and move it nearer the target value.'
- What this solution (achieved 5.40756) has done: 'The update switches the target to a log‑transform (log1p) during training, which better matches the distribution of fares and typically lowers RMSE. After prediction the values are back‑transformed with `expm1`. A slightly lower learning rate, deeper trees, and a longer early‑stopping patience allow the model to learn more without changing the overall pipeline or feature engineering. The submission file is still written as `taxi_fare_submission.csv` with the required columns.'
- What this solution (achieved 4.58486) has done: 'I add robust preprocessing to remove NaNs/outliers and create a useful distance feature, switch the model to train directly on the raw fare amount (removing the problematic log‑transform), and adjust XGBoost parameters slightly. These fixes eliminate the label‑NaN error, ensure the model variable exists for test prediction, and provide a more informative feature set, which together should lower the validation RMSE toward the target while keeping the core workflow unchanged.'
- What this solution (achieved 4.82911) has done: 'I increase the training sample size, add cyclic time features (sin/cos of hour, weekday, month) to give the model richer temporal information, and slightly adjust XGBoost hyper‑parameters (lower learning rate, deeper trees, longer early‑stopping patience) – all minimal, targeted changes that keep the original workflow intact while aiming to lower the validation RMSE toward the desired target.'

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb




## === cell 1
train_path = "../input/train.csv"
train_df = pd.read_csv(train_path, nrows=5_000_000)


def haversine_distance(lat1, lon1, lat2, lon2):
    """Vectorised Haversine distance in kilometers."""
    R = 6371.0
    lat1_rad, lat2_rad = np.radians(lat1), np.radians(lat2)
    lon1_rad, lon2_rad = np.radians(lon1), np.radians(lon2)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
    df["pickup_hour"] = df["pickup_datetime"].dt.hour
    df["pickup_weekday"] = df["pickup_datetime"].dt.weekday
    df["pickup_month"] = df["pickup_datetime"].dt.month

    df["hour_sin"] = np.sin(2 * np.pi * df["pickup_hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["pickup_hour"] / 24)
    df["weekday_sin"] = np.sin(2 * np.pi * df["pickup_weekday"] / 7)
    df["weekday_cos"] = np.cos(2 * np.pi * df["pickup_weekday"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["pickup_month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["pickup_month"] / 12)

    df["distance"] = haversine_distance(
        df["pickup_latitude"],
        df["pickup_longitude"],
        df["dropoff_latitude"],
        df["dropoff_longitude"],
    )

    numeric_cols = [
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
        "pickup_hour",
        "pickup_weekday",
        "pickup_month",
        "distance",
        "hour_sin",
        "hour_cos",
        "weekday_sin",
        "weekday_cos",
        "month_sin",
        "month_cos",
    ]
    for col in numeric_cols:
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)

    df = df.drop(columns=["pickup_datetime", "key"])
    return df


train_processed = preprocess(train_df)

train_processed = train_processed.dropna(subset=["fare_amount"])
train_processed = train_processed[
    (train_processed["fare_amount"] > 0) & (train_processed["fare_amount"] <= 500)
]

y = train_processed["fare_amount"]
X = train_processed.drop(columns=["fare_amount"])




## === cell 2
x_train, x_valid, y_train, y_valid = train_test_split(
    X, y, random_state=0, test_size=0.2
)




## === cell 3
def XGBmodel(x_train, x_valid, y_train, y_valid):
    dtrain = xgb.DMatrix(x_train, label=y_train)
    dvalid = xgb.DMatrix(x_valid, label=y_valid)

    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "eta": 0.03,  # lower learning rate for finer fitting
        "max_depth": 12,  # slightly deeper trees
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "seed": 0,
    }

    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=5000,
        evals=[(dvalid, "valid")],
        early_stopping_rounds=200,
        verbose_eval=False,
    )
    return model




## === cell 4
model = XGBmodel(x_train, x_valid, y_train, y_valid)

valid_pred = model.predict(xgb.DMatrix(x_valid))
valid_pred = np.clip(valid_pred, 0, None)

valid_rmse = np.sqrt(((valid_pred - y_valid) ** 2).mean())
print(f"Validation RMSE (raw fare): {valid_rmse:.4f}")




## === cell 5
test_path = "../input/test.csv"
test_df = pd.read_csv(test_path)

test_key = test_df["key"].values

test_processed = preprocess(test_df)

test_pred = model.predict(xgb.DMatrix(test_processed))
test_pred = np.clip(test_pred, 0, None)

submission = pd.DataFrame({"key": test_key, "fare_amount": np.round(test_pred, 2)})
submission_path = "taxi_fare_submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
