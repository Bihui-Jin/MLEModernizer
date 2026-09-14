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

4.33523

# 6. Current score

5.41337

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 5.76451) has done: 'Implemented a regression‑focused pipeline: cleaned data, engineered a distance feature, extracted hour of day, removed the erroneous classification steps, built a simple Keras model with a single linear output, and saved a proper CSV submission. Fixed syntax errors and ensured the submission file is correctly written.'
- What this solution (achieved 5.651) has done: 'I remove the unsupported `subsample` argument from the `HistGradientBoostingRegressor` initialization, which fixes the TypeError and allows the model to be trained. This change also restores the subsequent prediction and submission steps, so a valid `submission.csv` is generated. No other core logic is altered.'
- What this solution (achieved 5.6603) has done: 'I keep the overall pipeline unchanged but slightly strengthen the gradient‑boosting model, which is the main lever for RMS‑error. By increasing the number of boosting iterations, deepening the trees, and lowering the learning rate, the model can capture more complex relationships (especially the distance and hour features) without altering any core logic or data handling. These modest hyper‑parameter tweaks are expected to reduce the validation RMSE and move the leaderboard score closer to the target 4.33523.'
- What this solution (achieved 5.63013) has done: 'I add a few inexpensive time‑based features (weekday and month) to give the model more temporal context, and I modestly strengthen the gradient‑boosting regressor (more trees, deeper trees, slightly lower learning rate). These changes keep the original pipeline intact but should lower the validation RMSE, moving the score closer to the target.'
- What this solution (achieved 5.63853) has done: 'I added a line to store the test set’s `key` column before it gets dropped, defining `test_keys` so the submission dataframe can be built without a NameError. No other logic was changed, preserving the model and feature engineering.'
- What this solution (achieved 5.27407) has done: 'I add a few cheap interaction features (distance × hour, distance × passenger_count) to give the model more signal, and after the quick validation I refit the same model on the full training set so the final predictions use all available data. These minimal tweaks keep the original pipeline intact while nudging the RMS E downward toward the target.'
- What this solution (achieved 5.25065) has done: 'I add a cheap but informative feature `dist_per_hour` (distance divided by hour+1) to give the model a sense of speed, and I modestly strengthen the Gradient‑Boosting model by increasing the number of boosting rounds, deepening the trees and using a slightly smaller learning rate with a lower `min_samples_leaf`. These targeted changes keep the original pipeline intact while aiming to lower the validation RMSE toward the target score.'
- What this solution (achieved 5.42858) has done: 'I added two inexpensive features (`dist_per_passenger` and `log_passenger_count`) to give the model more signal, switched the target to a log‑space regression (training on `log1p(fare_amount)` and exponentiating predictions back), and slightly strengthened the gradient‑boosting model (more iterations, deeper trees, smaller learning rate, lower `min_samples_leaf`). These changes keep the original pipeline intact while aiming to lower the validation RMSE enough to fall within the target tolerance band.'
- What this solution (achieved 5.41433) has done: 'I increase the training sample size, add a few inexpensive temporal features (weekend flag and log‑hour), and slightly tune the gradient‑boosting hyper‑parameters to reduce over‑fitting while keeping the same model type and overall pipeline. These minimal changes are expected to lower the validation RMSE, moving the score closer to the target.'
- What this solution (achieved 5.41337) has done: 'I tighten the gradient‑boosting model to capture more signal without altering the overall pipeline. By increasing the number of boosting iterations and lowering the learning rate while slightly reducing tree depth, the model can fit the data more accurately and lower the RMS‑error toward the target. The only change is the hyper‑parameter settings in the model construction cell.'

# 9. Code solution

## === cell 0
import os, warnings

warnings.filterwarnings("ignore")
print(os.listdir("../input"))




## === cell 1
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error




## === cell 2
dtype_dict = {
    "fare_amount": np.float32,
    "pickup_longitude": np.float32,
    "pickup_latitude": np.float32,
    "dropoff_longitude": np.float32,
    "dropoff_latitude": np.float32,
    "passenger_count": np.int8,
}
train = pd.read_csv(
    "../input/train.csv",
    nrows=4_000_000,  # use more rows for better learning
    usecols=[
        "key",
        "fare_amount",
        "pickup_datetime",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
    ],
    dtype=dtype_dict,
    low_memory=False,
)




## === cell 3
train = train.loc[(train["fare_amount"] > 0) & (train["fare_amount"] < 200)]
train = train.loc[(train["pickup_longitude"] > -150) & (train["pickup_longitude"] < 0)]
train = train.loc[(train["pickup_latitude"] > 0) & (train["pickup_latitude"] < 80)]
train = train.loc[
    (train["dropoff_longitude"] > -150) & (train["dropoff_longitude"] < 0)
]
train = train.loc[(train["dropoff_latitude"] > 0) & (train["dropoff_latitude"] < 80)]
train = train.loc[train["passenger_count"] <= 8]




## === cell 4
test = pd.read_csv("../input/test.csv")
test_keys = test["key"].values




## === cell 5
def haversine(lon1, lat1, lon2, lat2):
    """Great‑circle distance in kilometres."""
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371 * c


def add_distance(df):
    df["distance"] = haversine(
        df["pickup_longitude"],
        df["pickup_latitude"],
        df["dropoff_longitude"],
        df["dropoff_latitude"],
    )


add_distance(train)
add_distance(test)




## === cell 6
train_dt = pd.to_datetime(train["pickup_datetime"])
test_dt = pd.to_datetime(test["pickup_datetime"])

train["hour"] = train_dt.dt.hour
test["hour"] = test_dt.dt.hour

train["weekday"] = train_dt.dt.weekday
test["weekday"] = test_dt.dt.weekday

train["month"] = train_dt.dt.month
test["month"] = test_dt.dt.month

train["is_weekend"] = (train["weekday"] >= 5).astype(np.int8)
test["is_weekend"] = (test["weekday"] >= 5).astype(np.int8)

train["log_hour"] = np.log1p(train["hour"]).astype(np.float32)
test["log_hour"] = np.log1p(test["hour"]).astype(np.float32)

train["log_distance"] = np.log1p(train["distance"])
test["log_distance"] = np.log1p(test["distance"])

train["distance_sq"] = train["distance"] ** 2
test["distance_sq"] = test["distance"] ** 2

train["hour_sin"] = np.sin(2 * np.pi * train["hour"] / 24)
train["hour_cos"] = np.cos(2 * np.pi * train["hour"] / 24)
test["hour_sin"] = np.sin(2 * np.pi * test["hour"] / 24)
test["hour_cos"] = np.cos(2 * np.pi * test["hour"] / 24)

train["dist_hour"] = train["distance"] * train["hour"]
test["dist_hour"] = test["distance"] * test["hour"]

train["passenger_distance"] = train["distance"] * train["passenger_count"]
test["passenger_distance"] = test["distance"] * test["passenger_count"]

train["dist_per_hour"] = train["distance"] / (train["hour"] + 1)
test["dist_per_hour"] = test["distance"] / (test["hour"] + 1)

train["dist_per_passenger"] = train["distance"] / (train["passenger_count"] + 1)
test["dist_per_passenger"] = test["distance"] / (test["passenger_count"] + 1)

train["log_passenger_count"] = np.log1p(train["passenger_count"])
test["log_passenger_count"] = np.log1p(test["passenger_count"])

cols_to_drop = [
    "key",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
]
train.drop(columns=cols_to_drop, inplace=True)
test.drop(columns=cols_to_drop, inplace=True)




## === cell 7
train.dropna(inplace=True)
test.dropna(inplace=True)




## === cell 8
train = train.loc[train["fare_amount"] <= 100]  # cap extreme fares




## === cell 9
y_original = train["fare_amount"].values.astype(np.float32)
y = np.log1p(y_original).astype(np.float32)

train.drop(columns=["fare_amount"], inplace=True)

X = train.values.astype(np.float32)
X_test = test.values.astype(np.float32)




## === cell 10
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_test = scaler.transform(X_test)




## === cell 11
X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.01, random_state=42)




## === cell 12
def rmse(y_true, y_pred):
    """Root Mean Squared Error."""
    return np.sqrt(mean_squared_error(y_true, y_pred))




## === cell 13
model = HistGradientBoostingRegressor(
    max_iter=3000,  # increase number of trees
    learning_rate=0.005,  # finer step size
    max_depth=12,  # modest depth to control over‑fit
    min_samples_leaf=5,
    random_state=42,
)

model.fit(X_tr, y_tr)

val_pred_log = model.predict(X_val)
val_pred = np.expm1(val_pred_log)
val_true = np.expm1(y_val)
val_rmse = rmse(val_true, val_pred)
print(f"Validation RMSE: {val_rmse:.4f}")

model.fit(X, y)




## === cell 14
preds_log = model.predict(X_test).flatten()
preds = np.expm1(preds_log)
preds = np.where(preds < 0, 0, preds)




## === cell 15
submission = pd.DataFrame({"key": test_keys, "fare_amount": preds})
submission.to_csv("submission.csv", index=False)




## === cell 16
print("Submission written. Files in current directory:")
print(os.listdir("."))
