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

3.95459

# 6. Current score

5.67963

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 5.62692) has done: 'I fix the XGBoost prediction call (using `iteration_range` instead of the unsupported `ntree_limit`) and prevent the test set from being filtered by the trip‑distance cutoff, which could misalign the predictions with the test keys. These minimal changes eliminate the runtime error and ensure a correctly sized submission file while keeping the original model and feature logic intact.'
- What this solution (achieved 5.67963) has done: 'I increase the amount of training data used and relax the distance filter slightly, which should give the XGBoost model a richer set of examples and improve its RMSE, moving the score closer to the target. The changes only adjust the constants in the first cell; all core logic, feature engineering, and model code remain unchanged.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb

MAX_TRAINING_SIZE = 300_000  # increased from 100_000

THRESHOLD_TRIP_DISTANCE = 30.0  # increased from 25.0
KMS_PER_RADIAN = 6371.0088


def haversine(coord1, coord2):
    lat1, lon1 = np.radians(coord1)
    lat2, lon2 = np.radians(coord2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return KMS_PER_RADIAN * 2 * np.arcsin(np.sqrt(a))


def _possible_paths(filename):
    """Return a list of plausible locations for a given dataset file."""
    base_paths = [
        os.path.join("data", filename),  # original relative path
        os.path.join("kaggle", "input", filename),  # /kaggle/input/<file>
        os.path.join("kaggle", "input", "new-york-city-taxi-fare-prediction", filename),
        os.path.join(
            "kaggle", "working", "new-york-city-taxi-fare-prediction", filename
        ),
        os.path.join("kaggle", "working", filename),
        os.path.join(
            "/", "kaggle", "input", "new-york-city-taxi-fare-prediction", filename
        ),
        os.path.join("/", "kaggle", "input", filename),
    ]
    return [p for p in base_paths if os.path.exists(p)]


def get_file_path(filename):
    """Find the first existing path for a dataset file; raise if none found."""
    candidates = _possible_paths(filename)
    if not candidates:
        raise FileNotFoundError(
            f"Unable to locate {filename} in any expected directory."
        )
    return candidates[0]


train_path = get_file_path("train.csv")
test_path = get_file_path("test.csv")

df_train = pd.read_csv(
    train_path, nrows=MAX_TRAINING_SIZE, parse_dates=["pickup_datetime"]
)
df_test = pd.read_csv(test_path, parse_dates=["pickup_datetime"])

test_key = df_test["key"].copy()

df_train.drop(columns=["key"], inplace=True)
df_test.drop(columns=["key"], inplace=True)



## === cell 1
df_train = df_train[
    (df_train.fare_amount >= 0) & (df_train.passenger_count.between(1, 6))
].dropna()




## === cell 2
def add_features(df, drop_long_distance=True):
    """Create engineered features; optionally drop extreme distances."""
    df["trip_distance"] = df.apply(
        lambda row: haversine(
            (row["pickup_latitude"], row["pickup_longitude"]),
            (row["dropoff_latitude"], row["dropoff_longitude"]),
        ),
        axis=1,
    )
    if drop_long_distance:
        df = df[df["trip_distance"] < THRESHOLD_TRIP_DISTANCE]

    df["hour"] = df["pickup_datetime"].dt.hour
    df["day"] = df["pickup_datetime"].dt.day
    df["month"] = df["pickup_datetime"].dt.month
    df["weekday"] = df["pickup_datetime"].dt.weekday
    df["year"] = df["pickup_datetime"].dt.year
    return df


df_train = add_features(df_train, drop_long_distance=True)
df_test = add_features(df_test, drop_long_distance=False)

drop_cols = [
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
]
df_train = df_train.drop(columns=drop_cols)
df_test = df_test.drop(columns=drop_cols)



## === cell 3
y = df_train["fare_amount"]
X = df_train.drop(columns=["fare_amount"])

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.01, random_state=42)



## === cell 4
params = {
    "max_depth": 8,
    "eta": 0.03,
    "subsample": 1,
    "colsample_bytree": 0.8,
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "verbosity": 0,
}
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)

model = xgb.train(
    params,
    dtrain,
    num_boost_round=5000,
    evals=[(dval, "validation")],
    early_stopping_rounds=10,
    verbose_eval=False,
)



## === cell 5
dtest = xgb.DMatrix(df_test)

best_ntree = getattr(model, "best_ntree_limit", None)
if best_ntree is None:
    best_ntree = (
        (model.best_iteration + 1)
        if model.best_iteration >= 0
        else model.num_boost_round
    )

pred = model.predict(dtest, iteration_range=(0, best_ntree))

submission = pd.DataFrame({"key": test_key, "fare_amount": np.round(pred, 2)})
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")
