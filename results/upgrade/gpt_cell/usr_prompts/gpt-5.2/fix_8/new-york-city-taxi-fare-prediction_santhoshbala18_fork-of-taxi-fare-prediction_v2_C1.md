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

3.70513

# 6. Current score

6.02741

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 5.3769) has done: 'The crash comes from an API change in LightGBM: in v4.x, `lgbm.Dataset(..., silent=True)` is no longer a valid argument, so the constructor raises `TypeError`. The minimal fix is to remove the unsupported `silent` keyword while keeping the same training data and parameters. This preserves the model training logic and ensures `model` is created for cell 10. No other behavior or data flow is changed.'
- What this solution (achieved 5.35838) has done: 'To move RMSE down toward your target with minimal logic changes, I fix two feature bugs that currently hurt performance: your `is_airport` longitude check uses positive longitudes (NYC is negative) and the airport rule uses `OR` instead of a bounding-box `AND`, so it flags many non-airport trips. I also correct a LightGBM parameter typo (`reg_aplha` → `reg_alpha`) so regularization is actually applied as intended, and make the train/valid split deterministic to stabilize score iteration-to-iteration. Core model (LightGBM GBDT regressor with the same features and training call) stays the same, and the script still writes a valid `submission.csv`.'
- What this solution (achieved 5.3592) has done: 'Your current RMSE (5.358) is worse than the target (3.705), so we should make small, legitimate improvements that preserve your LightGBM approach and feature set. The biggest issue is that you train on a random subset of 1e6 rows without filtering out extreme/erroneous fares and coordinate zeros, which injects a lot of noise; adding standard NYC Taxi Fare cleaning (fare caps + removing zero coordinates) typically improves RMSE substantially without changing model logic. I also ensure the model uses `best_iteration` by adding a proper validation set + `valid_sets` (same model/training API, no early stopping), and clip negative predictions to 0 to avoid RMSE blow-ups from invalid negatives. All paths stay the same and the script still produces a valid `submission.csv`.'
- What this solution (achieved 5.67617) has done: 'I make small, score-relevant improvements without changing your LightGBM training approach or feature set: (1) add a simple, standard outlier/consistency filter tying fare to trip distance to remove obvious noise while preserving most data, (2) ensure train/test feature columns are perfectly aligned to avoid subtle column-order issues, and (3) use the same cleaned-feature list for both validation and submission. These changes typically reduce RMSE materially on this competition and should move you closer to the 3.705 target while keeping the model, parameters, and training call intact. The script still run end-to-end within the same paths and write a valid `submission.csv`.'
- What this solution (achieved 5.72498) has done: 'Your current RMSE is worse than the target (lower is better), so we make the smallest changes that typically produce a real improvement without changing your LightGBM approach. The main issue is that `pickup_datetime`-derived features are created but then dropped (hour/day/month/weekday), so the model can’t use time patterns that strongly affect fares; we keep those features by not dropping them. We also add one very standard engineered feature (haversine-based Manhattan distance approximation via `abs(dlat)+abs(dlon)` in km) to complement your existing great-circle distance with minimal added logic. Finally, we make sure train/test feature columns are aligned in the same order and fill any missing values defensively to avoid subtle train/test mismatches.'
- What this solution (achieved 5.69567) has done: 'Your current RMSE (5.72498) is worse than the target (3.70513), so we should make small, legitimate improvements that reduce noise without changing the LightGBM approach. The biggest missing piece is filtering clearly-invalid passenger counts in the test set (the competition’s test includes 0 and other odd values), which can create outlier predictions that hurt RMSE; we clip passenger_count into a sane range for both train and test. Next, we add a minimal, standard location feature (distance to NYC center for pickup/dropoff) that often gives a noticeable RMSE drop while keeping the same model/training loop and feature-building pattern. Finally, we keep your existing cleaning/features intact and ensure the engineered features are applied consistently to train and test.'
- What this solution (achieved 6.02741) has done: 'We need to reduce RMSE (5.695 → 3.705), so we should remove obvious label/feature noise while keeping your LightGBM setup and feature engineering intact. The biggest missing cleaning piece is filtering out invalid `fare_amount==0` and implausible passenger counts/coordinates more consistently, plus removing very short “distance” rides with high fares and vice versa using a gentle fare-per-km band (this typically helps a lot on this competition without changing the model). I also align train/test preprocessing by applying the same inf/NaN handling to both, and keep your existing validation/training procedure unchanged. Finally, I keep the submission format identical while adding a small post-process cap to avoid extreme predictions that can inflate RMSE.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os

print(os.listdir("../input"))



## === cell 1
df = pd.read_csv("../input/train.csv", nrows=10**6)
test_set = pd.read_csv("../input/test.csv")




## === cell 2
def distance(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295  # Pi/180
    a = (
        0.5
        - np.cos((lat2 - lat1) * p) / 2
        + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    )
    return 12742 * np.arcsin(np.sqrt(a))  # 2*R*asin...


df["distance_km"] = distance(
    df.pickup_latitude, df.pickup_longitude, df.dropoff_latitude, df.dropoff_longitude
)

test_set["distance_km"] = distance(
    test_set.pickup_latitude,
    test_set.pickup_longitude,
    test_set.dropoff_latitude,
    test_set.dropoff_longitude,
)

KM_PER_DEG_LAT = 111.32
df["manhattan_km"] = (
    df["pickup_latitude"] - df["dropoff_latitude"]
).abs() * KM_PER_DEG_LAT + (df["pickup_longitude"] - df["dropoff_longitude"]).abs() * (
    KM_PER_DEG_LAT * np.cos(np.deg2rad(df["pickup_latitude"].clip(-90, 90)))
)
test_set["manhattan_km"] = (
    test_set["pickup_latitude"] - test_set["dropoff_latitude"]
).abs() * KM_PER_DEG_LAT + (
    test_set["pickup_longitude"] - test_set["dropoff_longitude"]
).abs() * (
    KM_PER_DEG_LAT * np.cos(np.deg2rad(test_set["pickup_latitude"].clip(-90, 90)))
)

NYC_LAT, NYC_LON = 40.7128, -74.0060
df["pickup_center_km"] = distance(
    df["pickup_latitude"], df["pickup_longitude"], NYC_LAT, NYC_LON
)
df["dropoff_center_km"] = distance(
    df["dropoff_latitude"], df["dropoff_longitude"], NYC_LAT, NYC_LON
)
test_set["pickup_center_km"] = distance(
    test_set["pickup_latitude"], test_set["pickup_longitude"], NYC_LAT, NYC_LON
)
test_set["dropoff_center_km"] = distance(
    test_set["dropoff_latitude"], test_set["dropoff_longitude"], NYC_LAT, NYC_LON
)



## === cell 3
BB = (-75, -73, 40, 41.5)


def select_within_boundingbox(df, BB):
    return (
        (df.pickup_longitude >= BB[0])
        & (df.pickup_longitude <= BB[1])
        & (df.pickup_latitude >= BB[2])
        & (df.pickup_latitude <= BB[3])
        & (df.dropoff_longitude >= BB[0])
        & (df.dropoff_longitude <= BB[1])
        & (df.dropoff_latitude >= BB[2])
        & (df.dropoff_latitude <= BB[3])
    )


print("Old size: %d" % len(df))
df = df[select_within_boundingbox(df, BB)]

df = df[(df.passenger_count >= 1) & (df.passenger_count <= 6)]
print("New size: %d" % len(df))

df["passenger_count"] = df["passenger_count"].clip(lower=1, upper=6)
test_set["passenger_count"] = test_set["passenger_count"].clip(lower=1, upper=6)




## === cell 4
def add_datetime_info(dataset):
    dataset["pickup_datetime"] = pd.to_datetime(dataset["pickup_datetime"])
    dataset["hour"] = dataset.pickup_datetime.dt.hour
    dataset["day"] = dataset.pickup_datetime.dt.day
    dataset["month"] = dataset.pickup_datetime.dt.month
    dataset["weekday"] = dataset.pickup_datetime.dt.weekday
    return dataset


df = add_datetime_info(df)
test_set = add_datetime_info(test_set)




## === cell 5
def add_flags(dataset):
    dataset["is_night"] = np.where(
        (
            ((dataset["hour"] >= 20) & (dataset["hour"] <= 23))
            | ((dataset["hour"] >= 0) & (dataset["hour"] < 6))
        ),
        1,
        0,
    )

    dataset["is_airport"] = np.where(
        (dataset["dropoff_longitude"] >= -73.90)
        & (dataset["dropoff_longitude"] <= -73.75)
        & (dataset["dropoff_latitude"] >= 40.62)
        & (dataset["dropoff_latitude"] <= 40.78),
        1,
        0,
    )

    dataset["is_surge"] = np.where(
        (
            (dataset["hour"] >= 16)
            & (dataset["hour"] < 20)
            & (dataset["weekday"] != 5)
            & (dataset["weekday"] != 6)
        ),
        1,
        0,
    )
    return dataset


df = add_flags(df)
test_set = add_flags(test_set)



## === cell 6
df = df.drop(df[df["fare_amount"] < 0].index, axis=0)
df = df[df["fare_amount"] > 0]
df = df[df["fare_amount"] <= 250]

coord_cols = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
]
for c in coord_cols:
    df = df[df[c] != 0]

df = df[(df["distance_km"] > 0) & (df["distance_km"] < 200)]
df = df[df["fare_amount"] >= 2.5]  # base fare in NYC; removes corrupt very small labels

df = df[df["fare_amount"] <= 3.0 + 10.0 * df["distance_km"] + 10.0]

fare_per_km = df["fare_amount"] / df["distance_km"].clip(lower=0.2)
df = df[(fare_per_km >= 0.5) & (fare_per_km <= 80.0)]



## === cell 7
plt.scatter(df["is_night"], df["fare_amount"], c="r")
plt.show()



## === cell 8
from sklearn.model_selection import train_test_split

drop_cols = ["key", "fare_amount", "pickup_datetime"]
feature_cols = [c for c in df.columns if c not in drop_cols]

X = df[feature_cols]
y = df["fare_amount"]

X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42
)



## === cell 9
import lightgbm as lgbm

params = {
    "boosting_type": "gbdt",
    "objective": "regression",
    "nthread": -1,
    "verbose": 0,
    "num_leaves": 31,
    "learning_rate": 0.05,
    "max_depth": -1,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.6,
    "reg_alpha": 1,
    "reg_lambda": 0.001,
    "metric": "rmse",
    "min_split_gain": 0.5,
    "min_child_weight": 1,
    "min_child_samples": 10,
    "scale_pos_weight": 1,
}

train_set = lgbm.Dataset(X_train, y_train)
valid_set = lgbm.Dataset(X_test, y_test, reference=train_set)
model = lgbm.train(
    params,
    train_set=train_set,
    num_boost_round=300,
    valid_sets=[valid_set],
    valid_names=["valid"],
)



## === cell 10
predictions = model.predict(X_test, num_iteration=model.best_iteration)



## === cell 11
import math
from sklearn.metrics import mean_squared_error

rmse = math.sqrt(mean_squared_error(y_test, predictions))
print(rmse)



## === cell 12
test_set_features = test_set.reindex(columns=feature_cols)
test_set_features = test_set_features.replace([np.inf, -np.inf], np.nan).fillna(0)

test_set_key = test_set["key"]

y_pred_final = model.predict(test_set_features, num_iteration=model.best_iteration)

y_pred_final = np.clip(y_pred_final, 0, 250)

submission = pd.DataFrame(
    {"key": test_set_key, "fare_amount": y_pred_final}, columns=["key", "fare_amount"]
)
submission.to_csv("submission.csv", index=False)
print("Submitted")



## === cell 13
submission.shape
