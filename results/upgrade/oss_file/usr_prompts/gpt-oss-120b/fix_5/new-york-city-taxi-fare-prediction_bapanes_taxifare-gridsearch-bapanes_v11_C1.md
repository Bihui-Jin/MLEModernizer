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

No external packages required in the script and installed.

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

3.916084767058486

# 6. Current score

4.57121

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 4.42697) has done: 'The fix adds simple preprocessing to remove or impute NaN values, creates extra time‑based features (hour and weekday) to improve model performance, and ensures the same processing is applied to the test set before prediction. The script now runs end‑to‑end, fits the RandomForest, prints a validation RMSE, and writes a correctly formatted `submission_file.csv`.'
- What this solution (achieved 4.44445) has done: 'I added cyclic hour and weekday features (sin/cos) to better capture temporal patterns, included them in the feature list, and increased the number of trees in the RandomForest (n_estimators = 300) with a slightly deeper max_depth. The same feature engineering is applied to the test set, ensuring consistent preprocessing while keeping the original model structure unchanged. These modest tweaks should reduce the validation RMSE, moving the score closer to the target.'
- What this solution (achieved 4.57121) has done: 'I filter out extreme fare values, train the model on a log‑transformed target to reduce skew, and convert predictions back with `expm1` before evaluating and creating the submission. These tweaks keep the same RandomForest core while improving RMSE toward the target.'

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


def haversine(lon1, lat1, lon2, lat2):
    """Vectorised haversine distance in kilometres."""
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    km = 6371.0 * 2 * np.arcsin(np.sqrt(a))
    return km


TRAIN_PATH = "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv"
TEST_PATH = "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv"

df_train = pd.read_csv(TRAIN_PATH, nrows=500_000)




## === cell 1
df_train["distance"] = haversine(
    df_train["pickup_longitude"],
    df_train["pickup_latitude"],
    df_train["dropoff_longitude"],
    df_train["dropoff_latitude"],
)

df_train["pickup_datetime"] = pd.to_datetime(
    df_train["pickup_datetime"], errors="coerce"
)
df_train["hour"] = df_train["pickup_datetime"].dt.hour
df_train["weekday"] = df_train["pickup_datetime"].dt.weekday

df_train["hour_sin"] = np.sin(2 * np.pi * df_train["hour"] / 24)
df_train["hour_cos"] = np.cos(2 * np.pi * df_train["hour"] / 24)
df_train["weekday_sin"] = np.sin(2 * np.pi * df_train["weekday"] / 7)
df_train["weekday_cos"] = np.cos(2 * np.pi * df_train["weekday"] / 7)

feature_cols = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
    "distance",
    "hour",
    "weekday",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
]

df_train = df_train[(df_train["fare_amount"] > 0) & (df_train["fare_amount"] < 200)]

df_train = df_train.dropna(subset=feature_cols + ["fare_amount"])

X = df_train[feature_cols].values
y = np.log1p(df_train["fare_amount"].values)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)




## === cell 2
rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=20,
    min_samples_leaf=2,
    n_jobs=-1,
    random_state=42,
)
rf.fit(X_train, y_train)




## === cell 3
val_pred_log = rf.predict(X_val)
val_pred = np.expm1(val_pred_log)
val_pred = np.clip(val_pred, 0, None)

rmse = mean_squared_error(np.expm1(y_val), val_pred, squared=False)
print(f"Validation RMSE: {rmse:.4f}")




## === cell 4
df_test = pd.read_csv(TEST_PATH)

df_test["distance"] = haversine(
    df_test["pickup_longitude"],
    df_test["pickup_latitude"],
    df_test["dropoff_longitude"],
    df_test["dropoff_latitude"],
)

df_test["pickup_datetime"] = pd.to_datetime(df_test["pickup_datetime"], errors="coerce")
df_test["hour"] = df_test["pickup_datetime"].dt.hour
df_test["weekday"] = df_test["pickup_datetime"].dt.weekday

df_test["hour_sin"] = np.sin(2 * np.pi * df_test["hour"] / 24)
df_test["hour_cos"] = np.cos(2 * np.pi * df_test["hour"] / 24)
df_test["weekday_sin"] = np.sin(2 * np.pi * df_test["weekday"] / 7)
df_test["weekday_cos"] = np.cos(2 * np.pi * df_test["weekday"] / 7)

X_test = df_test[feature_cols].fillna(0).values

test_pred_log = rf.predict(X_test)
test_pred = np.expm1(test_pred_log)
test_pred = np.clip(test_pred, 0, None)

submission = pd.DataFrame({"key": df_test["key"], "fare_amount": test_pred})
submission.to_csv("submission_file.csv", index=False)
print("Submission file 'submission_file.csv' written.")
