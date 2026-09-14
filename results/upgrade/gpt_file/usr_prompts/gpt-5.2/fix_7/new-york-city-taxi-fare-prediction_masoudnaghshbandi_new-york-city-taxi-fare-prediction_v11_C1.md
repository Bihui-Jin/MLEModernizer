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

3.10

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

4.53844

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 4.87043) has done: 'You’re getting a length mismatch because you filter/drop rows in `test` (so predictions are for fewer rows), but then you assign those predictions back to an unfiltered `sample_new` with all rows. I fix this by keeping `key` intact, building predictions only for rows that survive cleaning, and then merging back onto the full test keys so every row gets a `fare_amount` (using the model’s training-mean as a safe fallback for dropped/invalid rows). I also fix the incorrect scaler usage (`fit_transform(X_train, X_test)` doesn’t do what you intend) without changing the core modeling logic (you weren’t actually using scaled features anyway). Finally, the script always write a valid `submission1.csv` with exactly the required columns `key,fare_amount`.'
- What this solution (achieved 4.83917) has done: 'You’re currently underperforming the target (RMSE 4.87 vs 3.78; lower is better), so the smallest safe move is to improve feature signal without changing the model/training loop. I keep the same XGBoost regressor and training approach, but add a minimal, standard NYC-taxi feature: Haversine distance (computed from the same lat/lon columns) for both train and test. I also fix one obvious no-op/typo in the test cleaning block that can accidentally drop/keep rows inconsistently, which can slightly hurt generalization. Everything else (rows loaded, filters, model params, submission writing/merge fallback) stays intact and still writes `submission1.csv`.'
- What this solution (achieved 4.53568) has done: 'You’re below the target (4.839 > 3.7787, lower is better), so we make the smallest legitimate signal improvements without changing the core model/training loop. We keep the same XGBRegressor and same training approach, but (1) add a few standard time-based features extracted from `pickup_datetime` (hour, dayofweek, month) and (2) add a simple Manhattan-distance feature in lat/lon space alongside your existing Haversine distance. We also apply the exact same basic zero/passenger filters to `test_features` as you already do for `train` (previously, train/test cleaning was slightly inconsistent), while keeping the same submission merge+fallback logic to guarantee a valid `submission1.csv`.'
- What this solution (achieved 4.73308) has done: 'You’re currently worse than the target (RMSE 4.53568 vs 3.7787; lower is better), so we make the smallest legitimate signal/robustness improvements without changing your model type or training loop. The main score drag here is that `fare_amount` outliers remain in the training sample; adding the standard NYC Taxi fare filter (keep fares in a reasonable range) typically improves generalization while preserving your approach. We also make the XGBoost objective modern/consistent (`reg:squarederror`) to avoid legacy behavior differences, and we actually apply the already-fitted scaler to train/test features (your current code fits it but never uses the scaled arrays), which is a minimal fix that often yields a modest RMSE gain. Submission writing/merge+fallback logic stays the same to guarantee a valid `submission1.csv`.'
- What this solution (achieved 4.57519) has done: 'Your current gap to the target is large (4.73308 vs 3.7787; lower is better), so we need a modest but still “same-core-logic” boost in signal without changing the model type or training loop. The biggest safe lift here is to add a couple of standard geospatial features that use only the existing columns: (1) Haversine distance in **miles** (fares are in USD and NYC heuristics align better in miles), and (2) simple coordinate interaction features (`abs_dlon`, `abs_dlat`). We keep your exact XGBRegressor setup (same n_estimators/training flow/loss), keep your existing cleaning, and only expand the feature matrix consistently for train and test. We also keep the same submission merge+fallback behavior to guarantee a valid `submission1.csv`.'
- What this solution (achieved 4.53844) has done: 'To move RMSE down toward your 3.7787 target (current 4.57519; lower is better) without changing the core XGBRegressor/training loop, I add a couple of standard “same-input-columns” geospatial features that usually give a modest lift: bearing (direction of travel) and simple NYC-center distances (pickup/dropoff distance to Manhattan). I also make train/test cleaning fully consistent by applying the same passenger_count filter (`!=208`) to test and clamping negative predictions to 0 (a safe, metric-aligned postprocess that often reduces RMSE because fares can’t be negative). Everything else (data loading size, model type, n_estimators, scaling, submission merge+fallback) stays the same and still writes a valid `submission1.csv`.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd

import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error as MSE



## === cell 2
train = pd.read_csv(
    "../input/new-york-city-taxi-fare-prediction/train.csv", nrows=1000000
)
test = pd.read_csv("../input/new-york-city-taxi-fare-prediction/test.csv")



## === cell 3
train.shape, test.shape



## === cell 4
train.head()



## === cell 5
train.isnull().sum()



## === cell 6
train = train.dropna(how="any", axis="rows")



## === cell 7
test.isnull().sum()



## === cell 8
train.head()



## === cell 9
train["fare_amount"].describe()



## === cell 10
train.drop(train[train["pickup_longitude"] == 0].index, axis=0, inplace=True)
train.drop(train[train["pickup_latitude"] == 0].index, axis=0, inplace=True)
train.drop(train[train["dropoff_longitude"] == 0].index, axis=0, inplace=True)
train.drop(train[train["dropoff_latitude"] == 0].index, axis=0, inplace=True)
train.drop(train[train["passenger_count"] == 208].index, axis=0, inplace=True)
train.drop(train[train["passenger_count"] > 5].index, axis=0, inplace=True)
train.drop(train[train["passenger_count"] == 0].index, axis=0, inplace=True)



## === cell 11
train.head()



## === cell 12
train["pickup_datetime"] = pd.to_datetime(
    train["pickup_datetime"], errors="coerce", utc=True
)
train["pickup_hour"] = train["pickup_datetime"].dt.hour.astype("float32")
train["pickup_dayofweek"] = train["pickup_datetime"].dt.dayofweek.astype("float32")
train["pickup_month"] = train["pickup_datetime"].dt.month.astype("float32")



## === cell 13
train.drop(["key"], axis=1, inplace=True)



## === cell 14
train.drop(["pickup_datetime"], axis=1, inplace=True)



## === cell 15
train.dropna(inplace=True)

train.drop(
    train.index[
        (train.pickup_longitude < -75)
        | (train.pickup_longitude > -72)
        | (train.pickup_latitude < 40)
        | (train.pickup_latitude > 42)
    ],
    inplace=True,
)
train.drop(
    train.index[
        (train.dropoff_longitude < -75)
        | (train.dropoff_longitude > -72)
        | (train.dropoff_latitude < 40)
        | (train.dropoff_latitude > 42)
    ],
    inplace=True,
)

train = train[(train["fare_amount"] >= 2.5) & (train["fare_amount"] <= 200.0)].copy()



## === cell 16
train.head()




## === cell 17
def haversine_km(lon1, lat1, lon2, lat2):
    lon1 = np.radians(lon1.astype(float))
    lat1 = np.radians(lat1.astype(float))
    lon2 = np.radians(lon2.astype(float))
    lat2 = np.radians(lat2.astype(float))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371.0 * c  # Earth radius in km




## === cell 18
def bearing_rad(lon1, lat1, lon2, lat2):
    lon1 = np.radians(lon1.astype(float))
    lat1 = np.radians(lat1.astype(float))
    lon2 = np.radians(lon2.astype(float))
    lat2 = np.radians(lat2.astype(float))
    dlon = lon2 - lon1
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return np.arctan2(y, x)  # [-pi, pi]


NYC_LON = -73.985428  # approx Times Square / Midtown
NYC_LAT = 40.748817



## === cell 19
train["haversine_km"] = haversine_km(
    train["pickup_longitude"],
    train["pickup_latitude"],
    train["dropoff_longitude"],
    train["dropoff_latitude"],
)
train["haversine_miles"] = (train["haversine_km"] * 0.621371).astype("float32")

train["manhattan"] = (
    (train["pickup_longitude"] - train["dropoff_longitude"]).abs()
    + (train["pickup_latitude"] - train["dropoff_latitude"]).abs()
).astype("float32")
train["abs_dlon"] = (
    (train["pickup_longitude"] - train["dropoff_longitude"]).abs().astype("float32")
)
train["abs_dlat"] = (
    (train["pickup_latitude"] - train["dropoff_latitude"]).abs().astype("float32")
)

train["bearing"] = bearing_rad(
    train["pickup_longitude"],
    train["pickup_latitude"],
    train["dropoff_longitude"],
    train["dropoff_latitude"],
).astype("float32")
train["pickup_dist_to_nyc_km"] = haversine_km(
    train["pickup_longitude"], train["pickup_latitude"], NYC_LON, NYC_LAT
).astype("float32")
train["dropoff_dist_to_nyc_km"] = haversine_km(
    train["dropoff_longitude"], train["dropoff_latitude"], NYC_LON, NYC_LAT
).astype("float32")



## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/439101027.py in <cell line: 0>()
     25     train["dropoff_latitude"],
     26 ).astype("float32")
---> 27 train["pickup_dist_to_nyc_km"] = haversine_km(
     28     train["pickup_longitude"], train["pickup_latitude"], NYC_LON, NYC_LAT
     29 ).astype("float32")

/tmp/ipykernel_11/1919308816.py in haversine_km(lon1, lat1, lon2, lat2)
      2     lon1 = np.radians(lon1.astype(float))
      3     lat1 = np.radians(lat1.astype(float))
----> 4     lon2 = np.radians(lon2.astype(float))
      5     lat2 = np.radians(lat2.astype(float))
      6     dlon = lon2 - lon1

AttributeError: 'float' object has no attribute 'astype'

## === cell 20
X, y = train.drop("fare_amount", axis=1), train["fare_amount"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=12
)



## === cell 21
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



## === cell 22
xgb_r = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=400, seed=123)



## === cell 23
xgb_r.fit(X_train_scaled, y_train)



## === cell 24
y_pred = xgb_r.predict(X_test_scaled)



## === cell 25
rmse = np.sqrt(MSE(y_test, y_pred))
print("RMSE : % f" % (rmse))



## === cell 26
test.head()



## === cell 27
test_features = test.copy()



## === cell 28
test_features = test_features.dropna()

test_features.drop(
    test_features[test_features["pickup_longitude"] == 0].index, axis=0, inplace=True
)
test_features.drop(
    test_features[test_features["pickup_latitude"] == 0].index, axis=0, inplace=True
)
test_features.drop(
    test_features[test_features["dropoff_longitude"] == 0].index, axis=0, inplace=True
)
test_features.drop(
    test_features[test_features["dropoff_latitude"] == 0].index, axis=0, inplace=True
)

test_features.drop(
    test_features[test_features["passenger_count"] == 208].index, axis=0, inplace=True
)
test_features.drop(
    test_features[test_features["passenger_count"] > 5].index, axis=0, inplace=True
)
test_features.drop(
    test_features[test_features["passenger_count"] == 0].index, axis=0, inplace=True
)

test_features.drop(
    test_features.index[
        (test_features.pickup_longitude < -75)
        | (test_features.pickup_longitude > -72)
        | (test_features.pickup_latitude < 40)
        | (test_features.pickup_latitude > 42)
    ],
    inplace=True,
)

test_features.drop(
    test_features.index[
        (test_features.dropoff_longitude < -75)
        | (test_features.dropoff_longitude > -72)
        | (test_features.dropoff_latitude < 40)
        | (test_features.dropoff_latitude > 42)
    ],
    inplace=True,
)



## === cell 29
test_features.head()



## === cell 30
test_features["pickup_datetime"] = pd.to_datetime(
    test_features["pickup_datetime"], errors="coerce", utc=True
)
test_features["pickup_hour"] = test_features["pickup_datetime"].dt.hour.astype(
    "float32"
)
test_features["pickup_dayofweek"] = test_features[
    "pickup_datetime"
].dt.dayofweek.astype("float32")
test_features["pickup_month"] = test_features["pickup_datetime"].dt.month.astype(
    "float32"
)



## === cell 31
test_features = test_features.drop(["pickup_datetime"], axis=1)



## === cell 32
test_features["haversine_km"] = haversine_km(
    test_features["pickup_longitude"],
    test_features["pickup_latitude"],
    test_features["dropoff_longitude"],
    test_features["dropoff_latitude"],
)
test_features["haversine_miles"] = (test_features["haversine_km"] * 0.621371).astype(
    "float32"
)

test_features["manhattan"] = (
    (test_features["pickup_longitude"] - test_features["dropoff_longitude"]).abs()
    + (test_features["pickup_latitude"] - test_features["dropoff_latitude"]).abs()
).astype("float32")
test_features["abs_dlon"] = (
    (test_features["pickup_longitude"] - test_features["dropoff_longitude"])
    .abs()
    .astype("float32")
)
test_features["abs_dlat"] = (
    (test_features["pickup_latitude"] - test_features["dropoff_latitude"])
    .abs()
    .astype("float32")
)

test_features["bearing"] = bearing_rad(
    test_features["pickup_longitude"],
    test_features["pickup_latitude"],
    test_features["dropoff_longitude"],
    test_features["dropoff_latitude"],
).astype("float32")
test_features["pickup_dist_to_nyc_km"] = haversine_km(
    test_features["pickup_longitude"],
    test_features["pickup_latitude"],
    NYC_LON,
    NYC_LAT,
).astype("float32")
test_features["dropoff_dist_to_nyc_km"] = haversine_km(
    test_features["dropoff_longitude"],
    test_features["dropoff_latitude"],
    NYC_LON,
    NYC_LAT,
).astype("float32")



## --- ERROR in cell 32, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/52688835.py in <cell line: 0>()
     31     test_features["dropoff_latitude"],
     32 ).astype("float32")
---> 33 test_features["pickup_dist_to_nyc_km"] = haversine_km(
     34     test_features["pickup_longitude"],
     35     test_features["pickup_latitude"],

/tmp/ipykernel_11/1919308816.py in haversine_km(lon1, lat1, lon2, lat2)
      2     lon1 = np.radians(lon1.astype(float))
      3     lat1 = np.radians(lat1.astype(float))
----> 4     lon2 = np.radians(lon2.astype(float))
      5     lat2 = np.radians(lat2.astype(float))
      6     dlon = lon2 - lon1

AttributeError: 'float' object has no attribute 'astype'

## === cell 33
test_keys_valid = test_features["key"].copy()
X_test_pred = test_features.drop(["key"], axis=1)

X_test_pred_scaled = scaler.transform(X_test_pred)



## === cell 34
new_pred = xgb_r.predict(X_test_pred_scaled)

new_pred = np.maximum(new_pred, 0.0)



## === cell 35
fallback = float(y_train.mean())

submission = test[["key"]].copy()
pred_df = pd.DataFrame({"key": test_keys_valid.values, "fare_amount": new_pred})
submission = submission.merge(pred_df, on="key", how="left")
submission["fare_amount"] = submission["fare_amount"].fillna(fallback)



## === cell 36
submission.head()



## === cell 37
submission = submission[["key", "fare_amount"]]
submission.to_csv("submission1.csv", index=False)
print("Wrote submission1.csv with shape:", submission.shape)
print(submission.columns.tolist())



## === cell 38
assert submission.shape[0] == test.shape[0]
assert list(submission.columns) == ["key", "fare_amount"]
assert submission["fare_amount"].notna().all()
submission.describe(include="all")
