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

3.9

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

5.68914

# 6. Current score

7.42938

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 936.92806) has done: 'Diagnosis: The crash happens in cell 35 because `LinearRegression(normalize=True)` is no longer supported in scikit-learn 1.2.2; the `normalize` parameter was removed, so passing it raises `TypeError`.  
Patch summary: Remove the unsupported `normalize=True` argument while keeping the same LinearRegression model, training, and scoring flow. This is the minimal change needed to restore execution.  
Updated cells: Only cell 35 is modified.  
Compatibility notes for cell k+1: The variable `lr` is still created and fitted the same way, so `lr.predict(...)` in cell 36 remains compatible.  
Assumptions: No other preprocessing pipeline depends on `normalize`; we keep the model as plain `LinearRegression()` to match current scikit-learn API.'
- What this solution (achieved 936.92806) has done: 'Your score is extremely far from the target (RMSE 936 vs 5.69), which strongly suggests the submission rows are not aligned to the correct `key` order (you set `key` as the index before saving, so Kaggle may read the index as an unlabeled first column and treat the `key` column as missing/misaligned). I make the smallest fix that preserves your model and feature logic: write the submission with explicit `key,fare_amount` columns (no index) and ensure predictions are finite and non-negative (a standard post-processing for taxi fares that typically reduces RMSE without changing the learning approach). I also minimally enforce that train/test one-hot columns match (same weekday dummy columns) to avoid any silent column-order mismatch risk. These changes are directly aimed at producing a valid, correctly formatted submission and moving RMSE drastically down toward the target band.'
- What this solution (achieved 752.76803) has done: 'Your RMSE (936) is far worse than the target (5.689), so we should improve score substantially; the biggest likely culprit left is inconsistent scaling: you normalize train features using train mean/var but normalize test features using *test* mean/var, which causes a major train/test distribution mismatch and can explode errors. I keep your exact model and feature set, but compute the normalization statistics on the training data once and apply them to both train and test for `Difference_longitude`/`Difference_latitude` (minimal semantic change, large RMSE impact). I also add a tiny guard to ensure train/test feature columns are in identical order before prediction (prevents silent column misalignment). The submission writing remains `key,fare_amount` with `index=False` to ensure Kaggle reads it correctly.'
- What this solution (achieved 868.37343) has done: 'Your current RMSE is far worse than the target, so we should make a small but high-impact correction that keeps your same LinearRegression + engineered features: fix the normalization bug where you divide by the *variance* instead of the *standard deviation* (this severely distorts feature scale and can blow up predictions). I change only the normalization step to use `std = sqrt(var)` computed on train and applied to both train/test (same semantics, correct scaling). I also add a minimal sanity filter for training rows (fare/passenger bounds + coordinate bounds) which is standard for this competition and typically reduces RMSE without changing your model/feature logic. The submission format (`key,fare_amount`, `index=False`) remains unchanged.'
- What this solution (achieved 7.42037) has done: 'Your RMSE is still massively above target, so we need a small but high-impact fix that preserves your LinearRegression and engineered features while correcting an evaluation mismatch. The biggest issue left is that you’re training the model on *filtered* data (NYC bounds, fare/passenger bounds) but you do **not** apply the same coordinate/passenger sanity filtering to the test set, which can create extreme out-of-distribution feature values (especially distances) and blow up predictions. I add a minimal test-time filter: compute predictions for valid rows normally and use a safe fallback (median fare from training) for invalid rows; this keeps core logic intact and usually drops RMSE dramatically. I also add a small guard to prevent any remaining row/column alignment issues when building the submission.'
- What this solution (achieved 7.50399) has done: 'We make a minimal, score-relevant adjustment to prediction post-processing to reduce RMSE toward your target: replace the global fallback (median fare) for “invalid” test rows with a simple distance-based estimate derived from your training data (median $/mile plus median base fare), which better matches taxi fare structure without changing the LinearRegression model or features. We keep your existing valid-mask logic but also guard against any remaining extreme predictions by clipping to the training fare range (a standard stabilization that usually improves RMSE). Finally, we ensure the submission remains correctly aligned (`key,fare_amount`, no index) and that feature column order matches exactly as you already intended.'
- What this solution (achieved 7.42938) has done: 'Your current RMSE (7.50399) is still above the target (5.68914), so we should make a small, metric-aligned improvement without changing your model/feature set. The biggest low-risk gain left is to apply the same NYC coordinate sanity bounds to the test set that you already apply to training, then use your existing distance-based fallback only for those invalid/out-of-bounds rows; this reduces extreme out-of-distribution predictions that inflate RMSE. I also make the fallback estimate more stable by computing the base fare and $/mile rate on a “reasonable distance” slice of training rides (still the same fallback idea, just less influenced by near-zero-distance noise). Finally, I keep your submission formatting and strict column alignment unchanged to avoid any schema/order issues.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
train_data = pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv", nrows=10_000_000
)
train_data.head()



## === cell 2
train_data.shape



## === cell 3
train_data.info()



## === cell 4
test_data = pd.read_csv("/kaggle/input/new-york-city-taxi-fare-prediction/test.csv")
test_data.head()



## === cell 5
test_data.info()



## === cell 6
train_data.isna().sum()



## === cell 7
train_data["Difference_longitude"] = np.abs(
    np.asarray(train_data["pickup_longitude"] - train_data["dropoff_longitude"])
)
train_data["Difference_latitude"] = np.abs(
    np.asarray(train_data["pickup_latitude"] - train_data["dropoff_latitude"])
)

test_data["Difference_longitude"] = np.abs(
    np.asarray(test_data["pickup_longitude"] - test_data["dropoff_longitude"])
)
test_data["Difference_latitude"] = np.abs(
    np.asarray(test_data["pickup_latitude"] - test_data["dropoff_latitude"])
)



## === cell 8
print(f"Before Dropping null values: {len(train_data)}")
train_data.dropna(inplace=True)
print(f"After Dropping null values: {len(train_data)}")



## === cell 9
plot = train_data[:2000].plot.scatter("Difference_longitude", "Difference_latitude")



## === cell 10
train_data = train_data[
    (train_data["Difference_longitude"] < 5.0)
    & (train_data["Difference_latitude"] < 5.0)
]



## === cell 11
train_data = train_data[
    (train_data["fare_amount"] > 0) & (train_data["fare_amount"] < 500)
]
train_data = train_data[
    (train_data["passenger_count"] >= 1) & (train_data["passenger_count"] <= 6)
]
train_data = train_data[
    train_data["pickup_longitude"].between(-75, -72)
    & train_data["dropoff_longitude"].between(-75, -72)
    & train_data["pickup_latitude"].between(40, 42)
    & train_data["dropoff_latitude"].between(40, 42)
]



## === cell 12
ls1 = list(train_data["pickup_datetime"])
for i in range(len(ls1)):
    ls1[i] = ls1[i][11:-7:]
train_data["pickuptime"] = ls1



## === cell 13
ls1 = list(test_data["pickup_datetime"])
for i in range(len(ls1)):
    ls1[i] = ls1[i][11:-7:]
test_data["pickuptime"] = ls1



## === cell 14
train_data.head()



## === cell 15
ls1 = list(train_data["pickup_datetime"])
for i in range(len(ls1)):
    ls1[i] = ls1[i][:-4:]
    ls1[i] = pd.Timestamp(ls1[i])
    ls1[i] = ls1[i].weekday()
train_data["Weekday"] = ls1



## === cell 16
ls1 = list(test_data["pickup_datetime"])
for i in range(len(ls1)):
    ls1[i] = ls1[i][:-4:]
    ls1[i] = pd.Timestamp(ls1[i])
    ls1[i] = ls1[i].weekday()
test_data["Weekday"] = ls1



## === cell 17
train_data.head()



## === cell 18
test_data.head()



## === cell 19
train_data.drop("pickup_datetime", inplace=True, axis=1)
test_data.drop("pickup_datetime", inplace=True, axis=1)



## === cell 20
train_data["Weekday"].replace(
    to_replace=[i for i in range(0, 7)],
    value=[
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ],
    inplace=True,
)
test_data["Weekday"].replace(
    to_replace=[i for i in range(0, 7)],
    value=[
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ],
    inplace=True,
)



## === cell 21
train_one_hot = pd.get_dummies(train_data["Weekday"])
test_one_hot = pd.get_dummies(test_data["Weekday"])
train_one_hot, test_one_hot = train_one_hot.align(
    test_one_hot, join="outer", axis=1, fill_value=0
)

train_data = pd.concat([train_data, train_one_hot], axis=1)
test_data = pd.concat([test_data, test_one_hot], axis=1)



## === cell 22
train_data.drop("Weekday", axis=1, inplace=True)
test_data.drop("Weekday", axis=1, inplace=True)



## === cell 23
ls1 = list(train_data["pickuptime"])
for i in range(len(ls1)):
    z = ls1[i].split(":")
    ls1[i] = int(z[0]) * 100 + int(z[1])
train_data["pickuptime"] = ls1



## === cell 24
ls1 = list(test_data["pickuptime"])
for i in range(len(ls1)):
    z = ls1[i].split(":")
    ls1[i] = int(z[0]) * 100 + int(z[1])
test_data["pickuptime"] = ls1



## === cell 25
train_data.head()



## === cell 26
R = 6373.0
lat1 = np.asarray(np.radians(train_data["pickup_latitude"]))
lon1 = np.asarray(np.radians(train_data["pickup_longitude"]))
lat2 = np.asarray(np.radians(train_data["dropoff_latitude"]))
lon2 = np.asarray(np.radians(train_data["dropoff_longitude"]))

dlon = lon2 - lon1
dlat = lat2 - lat1
a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
distance = R * c
train_data["Distance"] = np.asarray(distance) * 0.621

lat1 = np.asarray(np.radians(test_data["pickup_latitude"]))
lon1 = np.asarray(np.radians(test_data["pickup_longitude"]))
lat2 = np.asarray(np.radians(test_data["dropoff_latitude"]))
lon2 = np.asarray(np.radians(test_data["dropoff_longitude"]))

dlon = lon2 - lon1
dlat = lat2 - lat1
a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
distance = R * c
test_data["Distance"] = np.asarray(distance) * 0.621



## === cell 27
R = 6373.0
lat1 = np.asarray(np.radians(train_data["pickup_latitude"]))
lon1 = np.asarray(np.radians(train_data["pickup_longitude"]))
lat2 = np.asarray(np.radians(train_data["dropoff_latitude"]))
lon2 = np.asarray(np.radians(train_data["dropoff_longitude"]))

lat3 = np.zeros(len(train_data)) + np.radians(40.6413111)
lon3 = np.zeros(len(train_data)) + np.radians(-73.7781391)

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
train_data["Pickup_Distance_airport"] = np.asarray(distance1) * 0.621

a2 = (
    np.sin(d_lat_dropoff / 2) ** 2
    + np.cos(lat2) * np.cos(lat3) * np.sin(d_lon_dropoff / 2) ** 2
)
c2 = 2 * np.arctan2(np.sqrt(a2), np.sqrt(1 - a2))
distance2 = R * c2
train_data["Dropoff_Distance_airport"] = np.asarray(distance2) * 0.621

lat1 = np.asarray(np.radians(test_data["pickup_latitude"]))
lon1 = np.asarray(np.radians(test_data["pickup_longitude"]))
lat2 = np.asarray(np.radians(test_data["dropoff_latitude"]))
lon2 = np.asarray(np.radians(test_data["dropoff_longitude"]))

lat3 = np.zeros(len(test_data)) + np.radians(40.6413111)
lon3 = np.zeros(len(test_data)) + np.radians(-73.7781391)

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
test_data["Pickup_Distance_airport"] = np.asarray(distance1) * 0.621

a2 = (
    np.sin(d_lat_dropoff / 2) ** 2
    + np.cos(lat2) * np.cos(lat3) * np.sin(d_lon_dropoff / 2) ** 2
)
c2 = 2 * np.arctan2(np.sqrt(a2), np.sqrt(1 - a2))
distance2 = R * c2
test_data["Dropoff_Distance_airport"] = np.asarray(distance2) * 0.621



## === cell 28
train_data["Distance"] = np.round(train_data["Distance"], 2)
train_data["Pickup_Distance_airport"] = np.round(
    train_data["Pickup_Distance_airport"], 2
)
train_data["Dropoff_Distance_airport"] = np.round(
    train_data["Dropoff_Distance_airport"], 2
)

test_data["Distance"] = np.round(test_data["Distance"], 2)
test_data["Pickup_Distance_airport"] = np.round(test_data["Pickup_Distance_airport"], 2)
test_data["Dropoff_Distance_airport"] = np.round(
    test_data["Dropoff_Distance_airport"], 2
)



## === cell 29
train_data.drop(
    ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"],
    axis=1,
    inplace=True,
)
test_data.drop(
    ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"],
    axis=1,
    inplace=True,
)



## === cell 30
dl_mean = float(np.mean(train_data["Difference_longitude"]))
dl_var = float(np.var(train_data["Difference_longitude"]))
dlat_mean = float(np.mean(train_data["Difference_latitude"]))
dlat_var = float(np.var(train_data["Difference_latitude"]))

eps = 1e-12
dl_std = float(np.sqrt(dl_var)) if dl_var > 0 else eps
dlat_std = float(np.sqrt(dlat_var)) if dlat_var > 0 else eps



## === cell 31
train_data["Difference_longitude"] = np.abs(
    train_data["Difference_longitude"] - dl_mean
)
train_data["Difference_longitude"] = train_data["Difference_longitude"] / dl_std



## === cell 32
train_data["Difference_latitude"] = np.abs(
    train_data["Difference_latitude"] - dlat_mean
)
train_data["Difference_latitude"] = train_data["Difference_latitude"] / dlat_std



## === cell 33
test_data["Difference_longitude"] = np.abs(test_data["Difference_longitude"] - dl_mean)
test_data["Difference_longitude"] = test_data["Difference_longitude"] / dl_std

test_data["Difference_latitude"] = np.abs(test_data["Difference_latitude"] - dlat_mean)
test_data["Difference_latitude"] = test_data["Difference_latitude"] / dlat_std



## === cell 34
train_data.shape



## === cell 35
test_data.shape



## === cell 36
from sklearn.model_selection import train_test_split

X = train_data.drop(["key", "fare_amount"], axis=1)
y = train_data["fare_amount"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.01, random_state=80
)



## === cell 37
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train, y_train)
print(lr.score(X_test, y_test))



## === cell 38
valid_mask = (
    test_data["passenger_count"].between(1, 6)
    & test_data["Difference_longitude"].between(0, 5.0)
    & test_data["Difference_latitude"].between(0, 5.0)
    & test_data["Distance"].between(
        0, 200.0
    )  # generous; just removes pathological distances
    & test_data["Pickup_Distance_airport"].between(0, 200.0)
    & test_data["Dropoff_Distance_airport"].between(0, 200.0)
)


X_test_submit = test_data.drop("key", axis=1)
X_test_submit = X_test_submit.reindex(columns=X.columns, fill_value=0)

raw_pred = np.full(shape=(len(test_data),), fill_value=np.nan, dtype=np.float64)
raw_pred[valid_mask.values] = lr.predict(X_test_submit.loc[valid_mask].values)
raw_pred = np.where(np.isfinite(raw_pred), raw_pred, np.nan)

train_dist = train_data["Distance"].to_numpy(dtype=np.float64)
train_fare = train_data["fare_amount"].to_numpy(dtype=np.float64)

reasonable = (
    np.isfinite(train_dist)
    & np.isfinite(train_fare)
    & (train_dist >= 0.5)
    & (train_dist <= 50.0)
)
if reasonable.any():
    dist_used = train_dist[reasonable]
    fare_used = train_fare[reasonable]
else:
    dist_used = train_dist
    fare_used = train_fare

rate_per_mile = float(np.nanmedian(fare_used / np.clip(dist_used, 0.1, None)))
base_fare = float(np.nanmedian(fare_used - rate_per_mile * dist_used))

fallback_by_distance = base_fare + rate_per_mile * test_data["Distance"].to_numpy(
    dtype=np.float64
)
raw_pred = np.where(np.isnan(raw_pred), fallback_by_distance, raw_pred)

y_min = float(np.nanpercentile(y_train.to_numpy(dtype=np.float64), 0.1))
y_max = float(np.nanpercentile(y_train.to_numpy(dtype=np.float64), 99.9))
pred = np.round(np.clip(raw_pred, y_min, y_max), 2)



## === cell 39
pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/sample_submission.csv"
).head()



## === cell 40
Submission = pd.DataFrame({"key": test_data["key"].values, "fare_amount": pred})
Submission = Submission[["key", "fare_amount"]]



## === cell 41
Submission.to_csv("Submission.csv", index=False)
print("Wrote Submission.csv with shape:", Submission.shape)
print(Submission.head())
