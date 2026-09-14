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

3.11

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

3.7787

# 6. Current score

5.4456

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 5.09489) has done: 'You’re dropping rows from `test` during cleaning, which makes `new_pred` shorter than the original `key` list and causes the length-mismatch error when building the submission. I keep the same XGBoost regressor and the same feature set, but change test-time cleaning to *not drop rows*; instead we flag invalid rows and still produce predictions for all 9914 keys (filling invalid rows with a simple fallback), so the submission is always aligned. I also fix the incorrect `StandardScaler` usage (you were fitting but never transforming your train/test matrices properly) in a minimal way that doesn’t change the core model. Finally, I write a valid `submission1.csv` with exactly `key` and `fare_amount` columns.'
- What this solution (achieved 5.06994) has done: 'Your current RMSE (5.09489) is worse than the target (3.7787), so we should cautiously improve generalization without changing the core model/feature set. The largest low-risk gain here is making train-time filtering consistent with your test-time “invalid” logic: right now you drop invalid geos/passenger_count in train, but you *don’t* drop or flag out-of-range fares (and you also never remove negative/huge fares), which injects noisy labels and hurts RMSE. I add minimal fare_amount sanity filtering (standard for this competition) and use a deterministic median fallback computed on the cleaned y_train; everything else (XGBRegressor, scaling, features, train/test split, submission schema) stays the same. This should move the score down toward the target band without altering the modeling approach.'
- What this solution (achieved 5.06994) has done: 'To move RMSE down toward your target with minimal risk, I keep the same XGBRegressor and the same raw feature set, but make train/test preprocessing consistent by applying the exact same invalid-row rules to the training sample (instead of only dropping some zero/out-of-range rows). This reduces label noise from clearly-bad coordinates and passenger counts that still slip into training and typically improves RMSE without changing the model. I also compute the fallback fare from the cleaned training target (same as you already do) and keep the submission aligned to all test keys. Everything still runs end-to-end within the same paths and writes `submission1.csv` with the required columns.'
- What this solution (achieved 5.63047) has done: 'I fix the runtime errors caused by using the deprecated/invalid `clip(min=...)` argument in pandas 2.2 by switching to the correct `clip(lower=...)`, which unblocks feature engineering for both train and test. I also remove a heavy, unnecessary re-read of `train.csv` inside the test preprocessing (it wastes time/memory and isn’t used for the model), while keeping the same model, features, and training loop. Finally, I ensure the test feature pipeline produces `test_scaled` deterministically so predictions are created for all test keys and a valid `submission1.csv` is written with the required columns.'
- What this solution (achieved 5.62686) has done: 'Your current RMSE (5.63047) is worse than the target (3.7787), so we should make small, safe improvements that reduce label noise and better align training with how you score on Kaggle, without changing the model or feature set. The biggest low-risk win is to train on a larger sample (still within runtime) while keeping the same cleaning rules; 1M rows is often too small/biased for this competition. I also refit the final model on all cleaned training data (not just the train split) after validating RMSE, which usually improves leaderboard RMSE with no semantic change. Finally, I keep the exact same test-time “don’t drop rows” logic and submission alignment so the CSV remains valid.'
- What this solution (achieved 5.4456) has done: 'To reduce RMSE toward your target without changing the core model/feature set, I keep the same XGBRegressor and the same two features (`passenger_count`, `log1p_distance_km`), but remove redundant/contradictory train cleaning and make the remaining cleaning consistent and vectorized. The main score-improving change is to cap extreme `passenger_count` values instead of dropping them (reduces distribution shift between train/test) while still filtering obviously invalid geo coordinates and implausible fares. I also keep the same scaling approach but ensure the fallback fare is computed from the cleaned training target (not the pre-cleaned one) and that test invalid rows are handled identically without dropping any test rows, preserving submission alignment.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd

import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error as MSE



## === cell 2
train = pd.read_csv(
    "../input/new-york-city-taxi-fare-prediction/train.csv", nrows=3000000
)
test = pd.read_csv("../input/new-york-city-taxi-fare-prediction/test.csv")
train.shape, test.shape



## === cell 3
train.head()



## === cell 4
train.isnull().sum()



## === cell 5
train = train.dropna(axis="rows")
test.isnull().sum()



## === cell 6
train.head()



## === cell 7
train["fare_amount"].describe()



## === cell 8
pass



## === cell 9
train.drop(["key"], axis=1, inplace=True)



## === cell 10
train.drop(["pickup_datetime"], axis=1, inplace=True)



## === cell 11
train.dropna(inplace=True)

required_cols = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]

train["passenger_count"] = train["passenger_count"].clip(lower=1, upper=6)

invalid_train = (
    train[required_cols].isna().any(axis=1)
    | (train["pickup_longitude"] == 0)
    | (train["pickup_latitude"] == 0)
    | (train["dropoff_longitude"] == 0)
    | (train["dropoff_latitude"] == 0)
    | (
        train["passenger_count"] == 0
    )  # after clipping, shouldn't occur, but keep for safety
    | (train["pickup_longitude"] < -75)
    | (train["pickup_longitude"] > -72)
    | (train["pickup_latitude"] < 40)
    | (train["pickup_latitude"] > 42)
    | (train["dropoff_longitude"] < -75)
    | (train["dropoff_longitude"] > -72)
    | (train["dropoff_latitude"] < 40)
    | (train["dropoff_latitude"] > 42)
)
train = train.loc[~invalid_train].copy()

train = train[(train["fare_amount"] > 0) & (train["fare_amount"] <= 250)].copy()




## === cell 12
def haversine_km(lon1, lat1, lon2, lat2):
    lon1 = np.radians(lon1.astype(float))
    lat1 = np.radians(lat1.astype(float))
    lon2 = np.radians(lon2.astype(float))
    lat2 = np.radians(lat2.astype(float))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(a))
    return 6371.0 * c


dist_km = haversine_km(
    train["pickup_longitude"].values,
    train["pickup_latitude"].values,
    train["dropoff_longitude"].values,
    train["dropoff_latitude"].values,
)
train["distance_km"] = dist_km

train["log1p_distance_km"] = np.log1p(train["distance_km"].clip(lower=0.0))

train.drop(
    [
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "distance_km",
    ],
    axis=1,
    inplace=True,
)



## === cell 13
X, y = train.drop("fare_amount", axis=1), train["fare_amount"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=12
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



## === cell 14
xgb_r = xgb.XGBRegressor(
    objective="reg:squarederror",
    n_estimators=400,
    seed=123,
)
xgb_r.fit(X_train_scaled, y_train)



## === cell 15
y_pred = xgb_r.predict(X_test_scaled)
rmse = np.sqrt(MSE(y_test, y_pred))
print("RMSE : % f" % (rmse))



## === cell 16
scaler_full = StandardScaler()
X_full_scaled = scaler_full.fit_transform(X)

xgb_r_full = xgb.XGBRegressor(
    objective="reg:squarederror",
    n_estimators=400,
    seed=123,
)
xgb_r_full.fit(X_full_scaled, y)



## === cell 17
test_raw = pd.read_csv("../input/new-york-city-taxi-fare-prediction/test.csv")
test_feat = test_raw.copy()

test_feat.drop(["pickup_datetime"], axis=1, inplace=True)

keys = test_feat["key"].copy()
test_feat.drop(["key"], axis=1, inplace=True)



## === cell 18
required_cols = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
for c in required_cols:
    if c not in test_feat.columns:
        raise ValueError(f"Missing expected column in test: {c}")

test_feat = test_feat.copy()
test_feat["passenger_count"] = test_feat["passenger_count"].clip(lower=1, upper=6)

invalid = (
    test_feat[required_cols].isna().any(axis=1)
    | (test_feat["pickup_longitude"] == 0)
    | (test_feat["pickup_latitude"] == 0)
    | (test_feat["dropoff_longitude"] == 0)
    | (test_feat["dropoff_latitude"] == 0)
    | (
        test_feat["passenger_count"] == 0
    )  # after clipping, shouldn't occur, but keep for safety
    | (test_feat["pickup_longitude"] < -75)
    | (test_feat["pickup_longitude"] > -72)
    | (test_feat["pickup_latitude"] < 40)
    | (test_feat["pickup_latitude"] > 42)
    | (test_feat["dropoff_longitude"] < -75)
    | (test_feat["dropoff_longitude"] > -72)
    | (test_feat["dropoff_latitude"] < 40)
    | (test_feat["dropoff_latitude"] > 42)
)

train_feature_medians = X.median(numeric_only=True)

test_feat_filled = test_feat.copy()

nyc_defaults = {
    "pickup_longitude": -73.985428,
    "pickup_latitude": 40.748817,
    "dropoff_longitude": -73.985428,
    "dropoff_latitude": 40.748817,
}

for col in [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]:
    if col in nyc_defaults:
        test_feat_filled[col] = test_feat_filled[col].fillna(nyc_defaults[col])
    else:
        test_feat_filled[col] = test_feat_filled[col].fillna(
            test_feat_filled[col].median()
        )

for col, val in nyc_defaults.items():
    test_feat_filled.loc[invalid, col] = val

test_feat_filled.loc[invalid, "passenger_count"] = float(
    train_feature_medians.get("passenger_count", 1.0)
)

test_feat_filled["distance_km"] = haversine_km(
    test_feat_filled["pickup_longitude"].values,
    test_feat_filled["pickup_latitude"].values,
    test_feat_filled["dropoff_longitude"].values,
    test_feat_filled["dropoff_latitude"].values,
)

test_feat_filled["log1p_distance_km"] = np.log1p(
    test_feat_filled["distance_km"].clip(lower=0.0)
)

test_model = test_feat_filled[["passenger_count", "log1p_distance_km"]].copy()
test_model = test_model.fillna(train_feature_medians)

test_scaled = scaler_full.transform(test_model)



## === cell 19
new_pred = xgb_r_full.predict(test_scaled)

fallback_fare = float(np.median(y.values))
new_pred = np.asarray(new_pred, dtype=float)
new_pred[invalid.values] = fallback_fare

new_pred = np.clip(new_pred, 0.0, 250.0)



## === cell 20
submission = pd.DataFrame({"key": keys.values, "fare_amount": new_pred})
submission.head()



## === cell 21
submission.to_csv("submission1.csv", index=False)
print("Wrote submission1.csv with shape:", submission.shape)
print("Columns:", submission.columns.tolist())
print(submission.head())
