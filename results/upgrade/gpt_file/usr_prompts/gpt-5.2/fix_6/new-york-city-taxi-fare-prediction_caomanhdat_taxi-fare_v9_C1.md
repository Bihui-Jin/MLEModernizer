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

4.79008

# 6. Current score

6.58884

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 6.23058) has done: 'I fix the datetime feature extraction to be compatible with modern pandas (replacing deprecated `.dt.week`/`.dt.weekofyear` with ISO calendar week) so your preprocessing runs. Then I ensure no datetime-typed columns leak into the model matrix by coercing all feature columns to numeric, which resolves the RandomForest `DTypePromotionError`. Finally, I keep your same model and training approach but make the train/test feature alignment explicit and write a valid `submission.csv` with the required columns.'
- What this solution (achieved 6.27152) has done: 'Your current RMSE (6.23058) is worse than the target (4.79008), so we should make a small, legitimate improvement without changing the core model/training approach. The biggest low-risk gain here is adding standard NYC Taxi data sanity filters (latitude/longitude bounds and outlier-distance removal) which reduces label noise and typically improves RMSE for this exact competition while keeping your same RandomForest and features. I also ensure we train on a numeric, NaN-free matrix consistently and clip negative predictions to 0 to avoid obviously invalid fares that can hurt RMSE. These changes are minimal, metric-aligned, and keep the same overall pipeline structure.'
- What this solution (achieved 6.26375) has done: 'You’re worse than the target (6.27152 vs 4.79008; lower is better), so we should make a small, safe improvement without changing your core approach (same RF model, same features, same training flow). The biggest issue is your current distance filter (`distance_travelled < 1.0`) is far too strict for NYC trips and discards many valid rides, hurting generalization; widening this to a more realistic bound typically reduces RMSE. I also compute a proper geodesic (haversine) distance feature alongside your existing Euclidean-in-degrees distance (keeping your original feature intact) and add a simple `abs_lon/abs_lat` sum as a tiny extra signal—these are minimal feature additions that keep the same model/training semantics but usually help this competition. Finally, I keep your existing sanity filters but slightly tighten only the most harmful outliers (e.g., absurd fares) while preserving valid longer trips.'
- What this solution (achieved 6.27378) has done: 'Your current RMSE (6.26375) is still worse than the target (4.79008), so we make a small, legitimate improvement without changing the core approach (same RandomForestRegressor, same training flow, same basic feature engineering). The biggest low-risk gain is to correct the overly-strict `distance_travelled < 0.30` filter: that filter is in “degrees” and is effectively removing many valid trips and distorting the training distribution; switching that particular bound to use your already-computed `haversine_km` is a minimal fix that typically improves RMSE. We keep your existing NYC bounding-box and fare/passenger filters, but make the distance filtering consistent and avoid double-filtering that conflicts. The submission writing stays the same and still produce a valid `submission.csv`.'
- What this solution (achieved 6.58884) has done: 'We’re currently worse than the target (6.27378 vs 4.79008; lower is better), so the smallest legitimate push toward the target is to reduce noise in the training labels and align the model to the heavy-tailed fare distribution without changing your core model/training loop. I add two standard NYC Taxi cleanup steps: remove rows where pickup/dropoff coordinates are identical (zero-distance but nonzero fare) and drop extreme outliers in `haversine_km` vs `fare_amount` using a very conservative speed-like cap. Then I train the same RandomForest on `log1p(fare_amount)` and invert with `expm1` at prediction time (same model, same features, same training flow), which typically improves RMSE for this competition by stabilizing large-fare errors. Submission writing and file name stay identical.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import os

INPUT_DIR_CANDIDATES = ["../input", "/kaggle/input", "/kaggle/data"]
INPUT_DIR = None
for p in INPUT_DIR_CANDIDATES:
    if os.path.isdir(p):
        INPUT_DIR = p
        break
if INPUT_DIR is None:
    INPUT_DIR = "../input"

print("Using INPUT_DIR:", INPUT_DIR)
print(os.listdir(INPUT_DIR))



## === cell 1
train_path = os.path.join(INPUT_DIR, "train.csv")
test_path = os.path.join(INPUT_DIR, "test.csv")

train = pd.read_csv(train_path, nrows=5000000)
test = pd.read_csv(test_path)



## === cell 2
train.head()



## === cell 3
test.head()



## === cell 4
train.describe()



## === cell 5
test.describe()



## === cell 6
train.isnull().sum()



## === cell 7
test.isnull().sum()




## === cell 8
def handle_date(df):
    df = df.copy()
    if df["pickup_datetime"].dtype == object:
        df["pickup_datetime"] = df["pickup_datetime"].str.replace(
            " UTC", "", regex=False
        )

    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], errors="coerce")

    df["hour_of_day"] = df["pickup_datetime"].dt.hour
    iso = df["pickup_datetime"].dt.isocalendar()
    df["week"] = iso.week.astype("int16")
    df["month"] = df["pickup_datetime"].dt.month
    df["year"] = df["pickup_datetime"].dt.year
    df["day_of_year"] = df["pickup_datetime"].dt.dayofyear
    df["week_of_year"] = iso.week.astype("int16")
    df["weekday"] = df["pickup_datetime"].dt.weekday
    df["quarter"] = df["pickup_datetime"].dt.quarter
    df["day_of_month"] = df["pickup_datetime"].dt.day

    df = df.drop("pickup_datetime", axis=1)
    return df


train = handle_date(train)
test = handle_date(test)




## === cell 9
def handle_distance(df):
    df = df.copy()

    df["longitude_distance"] = (df["pickup_longitude"] - df["dropoff_longitude"]).abs()
    df["latitude_distance"] = (df["pickup_latitude"] - df["dropoff_latitude"]).abs()
    df["distance_travelled"] = (
        df["longitude_distance"] ** 2 + df["latitude_distance"] ** 2
    ) ** 0.5

    R = 6371.0
    lat1 = np.deg2rad(df["pickup_latitude"].astype(float))
    lat2 = np.deg2rad(df["dropoff_latitude"].astype(float))
    dlat = lat2 - lat1
    dlon = np.deg2rad(
        df["dropoff_longitude"].astype(float) - df["pickup_longitude"].astype(float)
    )
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    df["haversine_km"] = 2.0 * R * np.arcsin(np.sqrt(a))

    df["manhattan_approx"] = df["longitude_distance"] + df["latitude_distance"]

    df = df.drop(["longitude_distance", "latitude_distance"], axis=1)
    return df


train = handle_distance(train)
test = handle_distance(test)



## === cell 10
train.describe()




## === cell 11
def clean_up_train(train):
    train = train.dropna()
    train = train[train["fare_amount"] > 0]
    train = train[train["passenger_count"] > 0]
    train = train[train["passenger_count"] < 7]

    train = train[
        (train["pickup_longitude"].between(-74.3, -73.7))
        & (train["dropoff_longitude"].between(-74.3, -73.7))
        & (train["pickup_latitude"].between(40.5, 41.0))
        & (train["dropoff_latitude"].between(40.5, 41.0))
    ]

    train = train[(train["haversine_km"] > 0.05) & (train["haversine_km"] < 80.0)]

    train = train[train["fare_amount"] < 200]

    same_loc = (train["pickup_longitude"] == train["dropoff_longitude"]) & (
        train["pickup_latitude"] == train["dropoff_latitude"]
    )
    train = train[~same_loc]

    max_reasonable = 2.5 + 10.0 * train["haversine_km"]
    train = train[train["fare_amount"] <= max_reasonable]

    return train


train = clean_up_train(train)
train.describe()




## === cell 12
def get_samples_output(train_df, test_df):
    feature_cols = test_df.drop("key", axis=1).columns
    X = train_df[feature_cols].copy()
    y = train_df["fare_amount"].copy()
    return X, y


samples_train, samples_label = get_samples_output(
    train.drop("key", axis=1, errors="ignore"), test
)

samples_train = samples_train.apply(pd.to_numeric, errors="coerce")
test_features = test.drop("key", axis=1).apply(pd.to_numeric, errors="coerce")

samples_train = samples_train.fillna(0.0)
test_features = test_features.fillna(0.0)



## === cell 13
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    samples_train, samples_label, test_size=0.3, random_state=0
)



## === cell 14
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=10, max_depth=2, random_state=0, n_jobs=-1)

y_train_log = np.log1p(samples_label.values.astype(float))
model.fit(samples_train, y_train_log)

preds_log = model.predict(test_features)
preds = np.expm1(preds_log)

preds = np.clip(preds, 0.0, None)

submission = pd.DataFrame(
    {"key": test["key"], "fare_amount": preds}, columns=["key", "fare_amount"]
)
submission.to_csv("submission.csv", index=False)

submission.head(20)
