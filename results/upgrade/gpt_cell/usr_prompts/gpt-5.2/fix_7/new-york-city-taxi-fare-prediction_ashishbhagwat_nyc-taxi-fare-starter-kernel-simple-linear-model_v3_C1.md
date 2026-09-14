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

3.10

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
seaborn==0.12.2
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

5.6891

# 6. Current score

28.87096

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 936.92806) has done: 'The crash is caused by `LinearRegression(normalize=True)`—the `normalize` parameter was removed in scikit-learn 1.2, so passing it now raises a `TypeError`. To preserve the existing modeling semantics as closely as possible while unblocking execution, I instantiate `LinearRegression` without the removed argument and keep the rest of the training and scoring code unchanged. This keeps the same variables (`lr`) and downstream interface expected by cell 43. No other cells need modification.'
- What this solution (achieved 28.74176) has done: 'Your score is extremely far from the target (RMSE 936 vs 5.69), which strongly suggests a fundamental prediction issue rather than a small modeling gap. The smallest legitimate fix that moves you toward the target is to prevent invalid/negative/huge fares from being submitted by clamping predictions to a reasonable range and ensuring the submission file matches Kaggle’s expected format (keep `key` as a column, not as the index). Additionally, your feature scaling is inconsistent (train uses train stats; test uses test stats), which can badly distort linear regression outputs; I keep your same features/model but apply the *training* mean/variance to both train and test for those two scaled columns. These changes keep the same core logic (same features, same model, same training approach) while directly addressing the likely cause of catastrophic RMSE and producing a valid `submission.csv`.'
- What this solution (achieved 28.78336) has done: 'Your current RMSE (28.74) is worse than the target (5.6891), so we should improve accuracy with the smallest changes that don’t alter the core model/feature set. The biggest likely issue is that train/test one-hot weekday columns may not align (missing weekday columns in test can shift feature order), which can severely degrade linear regression predictions; we force identical columns and ordering between train and test. We also add a minimal, competition-standard cleaning step to remove impossible fares and out-of-bounds coordinates from the 10M training sample, which typically yields a large RMSE reduction while keeping the same linear regression approach. Finally, we ensure column alignment is preserved at prediction time and keep your existing clipping and submission format.'
- What this solution (achieved 28.87245) has done: 'Your RMSE is still far above the target, so the smallest reliable way to move it down (without changing the model) is to fix two high-impact data issues: (1) your current “scaling” divides by variance instead of standard deviation, which badly distorts feature magnitudes for linear regression; we replace it with proper z-score scaling using train mean/std and apply the same parameters to test. (2) your weekday one-hot alignment step accidentally drops the `key` column from `test_df` (because `feature_cols` is built from `train_df`, which doesn’t contain `key`), which can misalign predictions/keys; we preserve `key` and enforce identical feature column ordering between train and test. These are minimal changes that keep the same feature set and the same `LinearRegression` training flow, but should reduce prediction error substantially and always produce a valid `submission.csv`.'
- What this solution (achieved 28.87096) has done: 'The crash happens because `Weekday` was created as pandas nullable integer dtype (`Int64`) in cell 16, and `Series.replace(..., inplace=True)` attempts to write string labels (e.g., `"Monday"`) back into that integer-typed column, which raises `TypeError: Invalid value 'Monday' for dtype Int64`. The minimal fix is to perform the weekday mapping by creating a new object/string-typed Series and assigning it back to `df["Weekday"]`, avoiding incompatible in-place mutation on an `Int64` array. This preserves the same labeling semantics (0–6 → Monday–Sunday) and keeps the column name unchanged for downstream cells. No other logic is altered.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import time
import os  # reading the input files we have access to

print(os.listdir("../input"))



## === cell 1
train_df = pd.read_csv("../input/train.csv", nrows=10_000_000)
train_df.dtypes



## === cell 2
train_df.head()



## === cell 3
train_df.shape



## === cell 4
train_df.info()



## === cell 5
test_df = pd.read_csv("../input/test.csv")
test_df.dtypes



## === cell 6
test_df.head()



## === cell 7
test_df.info()



## === cell 8
test_df.shape



## === cell 9
train_df.isna().sum()




## === cell 10
def add_travel_vector_features(df):
    df["abs_diff_longitude"] = (df.dropoff_longitude - df.pickup_longitude).abs()
    df["abs_diff_latitude"] = (df.dropoff_latitude - df.pickup_latitude).abs()


add_travel_vector_features(train_df)
add_travel_vector_features(test_df)



## === cell 11
print(f"Before Dropping null values: {len(train_df)}")
train_df.dropna(inplace=True)
print(f"After Dropping null values: {len(train_df)}")



## === cell 12
print(train_df.isnull().sum())



## === cell 13
print("Old size: %d" % len(train_df))
train_df = train_df.dropna(how="any", axis="rows")
print("New size: %d" % len(train_df))



## === cell 14
plot = train_df.iloc[:2000].plot.scatter("abs_diff_longitude", "abs_diff_latitude")



## === cell 15
print("Old size: %d" % len(train_df))
train_df = train_df[
    (train_df.abs_diff_longitude < 5.0) & (train_df.abs_diff_latitude < 5.0)
]
print("New size: %d" % len(train_df))




## === cell 16
def add_time_features(df):
    dt = pd.to_datetime(df["pickup_datetime"], errors="coerce")
    df["pickuptime"] = (dt.dt.hour * 100 + dt.dt.minute).astype("Int64")
    df["Weekday"] = dt.dt.dayofweek.astype("Int64")


add_time_features(train_df)
add_time_features(test_df)

train_df = train_df.dropna(subset=["pickuptime", "Weekday"]).copy()



## === cell 17
train_df.head()



## === cell 18
test_df.head()




## === cell 19
def replace_weekday(df):
    weekday_map = {
        0: "Monday",
        1: "Tuesday",
        2: "Wednesday",
        3: "Thursday",
        4: "Friday",
        5: "Saturday",
        6: "Sunday",
    }
    df["Weekday"] = df["Weekday"].map(weekday_map)


replace_weekday(train_df)
replace_weekday(test_df)


## === cell 20
train_df.head()



## === cell 21
test_df.head()



## === cell 22
train_df.drop("pickup_datetime", inplace=True, axis=1)
test_df.drop("pickup_datetime", inplace=True, axis=1)



## === cell 23
train_one_hot = pd.get_dummies(train_df["Weekday"])
test_one_hot = pd.get_dummies(test_df["Weekday"])
train_df = pd.concat([train_df, train_one_hot], axis=1)
test_df = pd.concat([test_df, test_one_hot], axis=1)



## === cell 24
train_df.head()



## === cell 25
test_df.head()



## === cell 26
train_df.drop("Weekday", axis=1, inplace=True)
test_df.drop("Weekday", axis=1, inplace=True)



## === cell 27
train_df["pickuptime"] = train_df["pickuptime"].astype(int)
test_df["pickuptime"] = test_df["pickuptime"].fillna(0).astype(int)



## === cell 28
train_df.head()



## === cell 29
test_df.head()




## === cell 30
def finding_distance(df):
    R = 6373.0
    lat1 = np.asarray(np.radians(df["pickup_latitude"]))
    lon1 = np.asarray(np.radians(df["pickup_longitude"]))
    lat2 = np.asarray(np.radians(df["dropoff_latitude"]))
    lon2 = np.asarray(np.radians(df["dropoff_longitude"]))

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c

    df["Distance"] = np.asarray(distance) * 0.621


finding_distance(train_df)
finding_distance(test_df)




## === cell 31
def creating_pickup_dropoff_distance(df):
    R = 6373.0
    lat1 = np.asarray(np.radians(df["pickup_latitude"]))
    lon1 = np.asarray(np.radians(df["pickup_longitude"]))
    lat2 = np.asarray(np.radians(df["dropoff_latitude"]))
    lon2 = np.asarray(np.radians(df["dropoff_longitude"]))

    lat3 = np.zeros(len(df)) + np.radians(40.6413111)
    lon3 = np.zeros(len(df)) + np.radians(-73.7781391)
    dlon_pickup = lon3 - lon1
    dlat_pickup = lat3 - lat1
    d_lon_dropoff = lon3 - lon2
    d_lat_dropoff = lat3 - lat2
    a1 = (
        np.sin(dlat_pickup / 2) ** 2
        + np.cos(lat1) * np.cos(lat3) * np.sin(dlon_pickup / 2) ** 2
    )
    c1 = 2 * np.arctan2(np.sqrt(a1), np.sqrt(1 - a1))
    distance1 = R * c1
    df["Pickup_Distance_airport"] = np.asarray(distance1) * 0.621

    a2 = (
        np.sin(d_lat_dropoff / 2) ** 2
        + np.cos(lat2) * np.cos(lat3) * np.sin(d_lon_dropoff / 2) ** 2
    )
    c2 = 2 * np.arctan2(np.sqrt(a2), np.sqrt(1 - a2))
    distance2 = R * c2

    df["Dropoff_Distance_airport"] = np.asarray(distance2) * 0.621


creating_pickup_dropoff_distance(train_df)
creating_pickup_dropoff_distance(test_df)



## === cell 32
train_df["Distance"] = np.round(train_df["Distance"], 2)
train_df["Pickup_Distance_airport"] = np.round(train_df["Pickup_Distance_airport"], 2)
train_df["Dropoff_Distance_airport"] = np.round(train_df["Dropoff_Distance_airport"], 2)
test_df["Distance"] = np.round(test_df["Distance"], 2)
test_df["Pickup_Distance_airport"] = np.round(test_df["Pickup_Distance_airport"], 2)
test_df["Dropoff_Distance_airport"] = np.round(test_df["Dropoff_Distance_airport"], 2)



## === cell 33
print("Old size (before distance cleaning): %d" % len(train_df))
train_df = train_df[(train_df["Distance"] > 0) & (train_df["Distance"] <= 100)].copy()
print("New size (after distance cleaning): %d" % len(train_df))



## === cell 34
print("Old size (before fare/coord cleaning): %d" % len(train_df))
train_df = train_df[
    (train_df["fare_amount"] > 0) & (train_df["fare_amount"] <= 500)
].copy()

train_df = train_df[
    (train_df["pickup_longitude"].between(-75, -72))
    & (train_df["dropoff_longitude"].between(-75, -72))
    & (train_df["pickup_latitude"].between(40, 42))
    & (train_df["dropoff_latitude"].between(40, 42))
].copy()

train_df = train_df[
    (train_df["passenger_count"] >= 1) & (train_df["passenger_count"] <= 6)
].copy()
print("New size (after fare/coord cleaning): %d" % len(train_df))



## === cell 35
train_df.drop(
    ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"],
    axis=1,
    inplace=True,
)
test_df.drop(
    ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"],
    axis=1,
    inplace=True,
)



## === cell 36
_abs_lon_mean = float(np.mean(train_df["abs_diff_longitude"]))
_abs_lon_std = float(np.std(train_df["abs_diff_longitude"])) + 1e-12  # numerical safety
train_df["abs_diff_longitude"] = (
    train_df["abs_diff_longitude"] - _abs_lon_mean
) / _abs_lon_std



## === cell 37
_abs_lat_mean = float(np.mean(train_df["abs_diff_latitude"]))
_abs_lat_std = float(np.std(train_df["abs_diff_latitude"])) + 1e-12
train_df["abs_diff_latitude"] = (
    train_df["abs_diff_latitude"] - _abs_lat_mean
) / _abs_lat_std



## === cell 38
test_df["abs_diff_longitude"] = (
    test_df["abs_diff_longitude"] - _abs_lon_mean
) / _abs_lon_std
test_df["abs_diff_latitude"] = (
    test_df["abs_diff_latitude"] - _abs_lat_mean
) / _abs_lat_std



## === cell 39
train_feature_cols = [c for c in train_df.columns if c not in ["fare_amount", "key"]]
test_feature_cols = [c for c in test_df.columns if c != "key"]

missing_in_test = sorted(list(set(train_feature_cols) - set(test_feature_cols)))
missing_in_train = sorted(list(set(test_feature_cols) - set(train_feature_cols)))

for c in missing_in_test:
    test_df[c] = 0
for c in missing_in_train:
    train_df[c] = 0
    train_feature_cols.append(c)

train_feature_cols = sorted(train_feature_cols)

X_all = train_df[train_feature_cols]
X_test_all = test_df[["key"] + train_feature_cols]

print("Train shape (with target):", train_df.shape)
print("X_all shape:", X_all.shape)
print("Test shape (with key):", test_df.shape)
print("X_test_all shape:", X_test_all.shape)



## === cell 40
from sklearn.model_selection import train_test_split

X = X_all
y = train_df["fare_amount"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.01, random_state=80
)



## === cell 41
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train, y_train)
print(lr.score(X_test, y_test))



## === cell 42
pred = lr.predict(X_test_all.drop("key", axis=1))
pred = np.asarray(pred, dtype=float)
pred = np.clip(pred, 0.0, 500.0)  # typical competition-safe bound
pred = np.round(pred, 2)
print(pred)



## === cell 43
Submission = pd.DataFrame({"key": X_test_all["key"], "fare_amount": pred})



## === cell 44
Submission = Submission[["key", "fare_amount"]]
Submission.head()



## === cell 45
Submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", Submission.shape)
