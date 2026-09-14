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

8.02842

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 936.92806) has done: 'The crash happens because scikit-learn 1.2 removed the `normalize` parameter from `LinearRegression`, so passing `normalize=True` raises a `TypeError`. The minimal fix is to instantiate `LinearRegression()` without the deprecated argument while keeping the rest of the training and scoring logic identical. This preserves the `lr` variable for the next cell, so prediction code remains compatible. No other cells need changes.'
- What this solution (achieved 911.97581) has done: 'The crash happens because `train_data.align(test_data, ...)` in cell 20 makes both DataFrames share the same columns, which includes `fare_amount`. During training (cell 32) you drop `fare_amount` from `train_data` to build `X`, but during prediction (cell 34) you only drop `key` from `test_data`, leaving an extra `fare_amount` column that the model never saw at fit time. The minimal fix is to ensure the prediction feature matrix uses the exact same feature columns as `X` (same names and order), which avoids sklearn’s feature-name mismatch check. This keeps the model, training, and evaluation semantics unchanged.'
- What this solution (achieved 870.28711) has done: 'Your RMSE is extremely high because the model is being trained on a subtly corrupted feature set: after `align(...)`, the `fare_amount` column can get introduced into `test_data`, and your later standardization uses **test-set mean/variance** (and even the wrong formula: dividing by variance instead of standard deviation), creating severe train/test feature distribution mismatch. To move your score down toward the 5.689 target without changing the model or features, I (1) prevent `fare_amount` from ever being part of the alignment between train/test, and (2) compute normalization parameters on the training data and apply them consistently to both train and test (same logic, just made consistent). These are minimal, metric-relevant fixes that preserve the core linear regression approach and feature engineering while correcting a train/test pipeline bug. The script still write a valid `Submission.csv` with `key,fare_amount`.'
- What this solution (achieved 870.28711) has done: 'Your score is still huge because the normalization in cells 27–28 is mathematically incorrect (you divide by variance, not standard deviation), which badly distorts feature scales and harms linear regression. I keep the exact same features and LinearRegression training, but change those two normalizations to use training-set **standard deviation** (computed from variance) and apply it consistently to both train and test. This is a minimal, metric-relevant fix that should move RMSE sharply downward toward your 5.689 target without altering the modeling approach. The submission writing logic stays the same and still produce a valid `Submission.csv`.'
- What this solution (achieved 870.28711) has done: 'Your score is far from the target (lower is better), so we should correct a remaining pipeline issue that can catastrophically hurt RMSE without changing the model or features: the test set currently keeps raw latitude/longitude values (only dropped later), but the alignment/one-hot steps can leave subtle column-order/type differences and NaNs that propagate into LinearRegression predictions. I (1) ensure both train and test have identical numeric feature columns with no NaNs right before fitting/predicting, and (2) add the same basic sanity clipping for `passenger_count` in test as you already enforce in train (without dropping rows) to avoid extreme out-of-domain values harming predictions. These are minimal, metric-relevant changes that keep your core feature engineering and LinearRegression intact while making train/test feature distributions consistent. The script still write a valid `Submission.csv` with `key,fare_amount`.'
- What this solution (achieved 870.28711) has done: 'Your RMSE is still orders of magnitude too high, which strongly suggests the model is producing wildly wrong (often huge) fare predictions due to remaining out-of-range/dirty test inputs and a missing scale mismatch on the largest engineered numeric features. To move the score down toward the 5.68914 target without changing the core model (still `LinearRegression`) or feature set, I (1) apply the same geographic bounding-box filter logic to the test set (as clipping, not dropping) so distance-like features cannot explode, and (2) normalize the three distance-based engineered features (`Distance`, `Pickup_Distance_airport`, `Dropoff_Distance_airport`) using training-set mean/std and apply consistently to test. These are minimal, metric-relevant pipeline fixes that preserve the current approach but prevent extreme values from dominating the linear regression. The script still run end-to-end and write a valid `Submission.csv` with `key,fare_amount`.'
- What this solution (achieved 64.78705) has done: 'Your RMSE is still catastrophically high because the model is effectively being trained on raw/dirty coordinate space (after you later drop lat/long entirely), while the only remaining continuous geometry features are absolute diffs and distances that are not consistently scaled, and the `pickuptime` feature is a large-magnitude integer that dominates linear regression without proper standardization. To move your score sharply downward toward the 5.689 target without changing the model or feature set, I apply the same train-derived mean/std normalization to `pickuptime` and to the three distance-based features using the *correct* z-score (no abs), keeping everything else identical. I also remove the now-no-op latitude/longitude clipping block (it currently runs after those columns are already dropped) and instead add a safe clip for the already-engineered diff features to prevent rare extreme values from blowing up predictions. These are minimal pipeline fixes (same LinearRegression, same engineered columns) aimed at correcting scale mismatch and stabilizing prediction magnitudes.'
- What this solution (achieved 8.02362) has done: 'Your RMSE is still far above the target, which usually happens here when some test rows contain invalid/out-of-domain coordinates that create extreme engineered distances and therefore extreme predictions. With minimal change and without altering your model/features, I clip the *raw* lat/long in the test set to the same NYC bounding box used for train **before** computing Difference/Distance features, so the engineered features stay in-range. I also apply the same `Difference_longitude/latitude < 5.0` constraint to the test set as clipping (not dropping) to prevent rare explosions while keeping all rows for submission. Everything else (feature engineering, LinearRegression, training loop, submission format) stays the same.'
- What this solution (achieved 8.02908) has done: 'Your current RMSE (8.02) is still above the target (5.69), so we should make a small, metric-relevant correction that doesn’t change the model or feature set: stop rounding engineered distance features to 2 decimals, because that quantization throws away signal and typically hurts LinearRegression RMSE. I keep all feature engineering and the same LinearRegression training/prediction flow, but remove the rounding in cell 27 so distances remain continuous. I also keep the final submission rounding to 2 decimals (harmless for Kaggle formatting) and keep the same output file name/columns to ensure a valid submission is produced.'
- What this solution (achieved 8.02842) has done: 'Your current RMSE (8.029) is worse than the target (5.689), so we should make a small, metric-relevant correction that keeps the same LinearRegression model and engineered features but fixes a remaining avoidable source of error. The biggest issue left is that you never standardize `passenger_count`, even though it is on a very different scale than the z-scored distance/diff/time features; with LinearRegression this can noticeably hurt fit and generalization. I add train-mean/std normalization for `passenger_count` (computed on train and applied to test), and keep everything else identical, including the submission format and file name.'

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
test_data = pd.read_csv("/kaggle/input/new-york-city-taxi-fare-prediction/test.csv")
test_data.head()



## === cell 4
test_data.info()



## === cell 5
train_data.isna().sum()



## === cell 6
for col, lo, hi in [
    ("pickup_longitude", -74.5, -72.8),
    ("dropoff_longitude", -74.5, -72.8),
    ("pickup_latitude", 40.0, 41.8),
    ("dropoff_latitude", 40.0, 41.8),
]:
    if col in test_data.columns:
        test_data[col] = test_data[col].clip(lower=lo, upper=hi)



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
train_data = train_data[
    (train_data["fare_amount"] > 0)
    & (train_data["fare_amount"] <= 500)
    & (train_data["passenger_count"] >= 1)
    & (train_data["passenger_count"] <= 6)
    & (train_data["pickup_longitude"].between(-74.5, -72.8))
    & (train_data["dropoff_longitude"].between(-74.5, -72.8))
    & (train_data["pickup_latitude"].between(40.0, 41.8))
    & (train_data["dropoff_latitude"].between(40.0, 41.8))
].copy()



## === cell 10
plot = train_data[:2000].plot.scatter("Difference_longitude", "Difference_latitude")



## === cell 11
train_data = train_data[
    (train_data["Difference_longitude"] < 5.0)
    & (train_data["Difference_latitude"] < 5.0)
]



## === cell 12
test_data["Difference_longitude"] = test_data["Difference_longitude"].clip(upper=5.0)
test_data["Difference_latitude"] = test_data["Difference_latitude"].clip(upper=5.0)



## === cell 13
ls1 = list(train_data["pickup_datetime"])
for i in range(len(ls1)):
    ls1[i] = ls1[i][11:-7:]
train_data["pickuptime"] = ls1

ls1 = list(test_data["pickup_datetime"])
for i in range(len(ls1)):
    ls1[i] = ls1[i][11:-7:]
test_data["pickuptime"] = ls1



## === cell 14
train_data.head()



## === cell 15
test_data.head()



## === cell 16
ls1 = list(train_data["pickup_datetime"])
for i in range(len(ls1)):
    ls1[i] = ls1[i][:-4:]
    ls1[i] = pd.Timestamp(ls1[i])
    ls1[i] = ls1[i].weekday()
train_data["Weekday"] = ls1

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
train_data = pd.concat([train_data, train_one_hot], axis=1)
test_data = pd.concat([test_data, test_one_hot], axis=1)



## === cell 22
train_features = train_data.drop(columns=["fare_amount"])
train_features_aligned, test_data = train_features.align(
    test_data, join="left", axis=1, fill_value=0
)
train_data = pd.concat([train_features_aligned, train_data[["fare_amount"]]], axis=1)



## === cell 23
train_data.drop("Weekday", axis=1, inplace=True)
test_data.drop("Weekday", axis=1, inplace=True)



## === cell 24
ls1 = list(train_data["pickuptime"])
for i in range(len(ls1)):
    z = ls1[i].split(":")
    ls1[i] = int(z[0]) * 100 + int(z[1])
train_data["pickuptime"] = ls1

ls1 = list(test_data["pickuptime"])
for i in range(len(ls1)):
    z = ls1[i].split(":")
    ls1[i] = int(z[0]) * 100 + int(z[1])
test_data["pickuptime"] = ls1



## === cell 25
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



## === cell 26
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



## === cell 27
pass



## === cell 28
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



## === cell 29
dl_mu = float(train_data["Difference_longitude"].mean())
dl_var = float(train_data["Difference_longitude"].var())
dl_std = float(np.sqrt(dl_var)) if dl_var > 0.0 else 1.0
train_data["Difference_longitude"] = (
    train_data["Difference_longitude"] - dl_mu
) / dl_std
test_data["Difference_longitude"] = (test_data["Difference_longitude"] - dl_mu) / dl_std



## === cell 30
dlat_mu = float(train_data["Difference_latitude"].mean())
dlat_var = float(train_data["Difference_latitude"].var())
dlat_std = float(np.sqrt(dlat_var)) if dlat_var > 0.0 else 1.0
train_data["Difference_latitude"] = (
    train_data["Difference_latitude"] - dlat_mu
) / dlat_std
test_data["Difference_latitude"] = (
    test_data["Difference_latitude"] - dlat_mu
) / dlat_std



## === cell 31
train_data.shape



## === cell 32
test_data.shape



## === cell 33
dist_cols = ["Distance", "Pickup_Distance_airport", "Dropoff_Distance_airport"]
for col in dist_cols:
    if col in train_data.columns and col in test_data.columns:
        mu = float(train_data[col].mean())
        var = float(train_data[col].var())
        std = float(np.sqrt(var)) if var > 0.0 else 1.0
        train_data[col] = (train_data[col] - mu) / std
        test_data[col] = (test_data[col] - mu) / std

pt_mu = float(pd.to_numeric(train_data["pickuptime"], errors="coerce").mean())
pt_var = float(pd.to_numeric(train_data["pickuptime"], errors="coerce").var())
pt_std = float(np.sqrt(pt_var)) if pt_var > 0.0 else 1.0
train_data["pickuptime"] = (
    pd.to_numeric(train_data["pickuptime"], errors="coerce") - pt_mu
) / pt_std
test_data["pickuptime"] = (
    pd.to_numeric(test_data["pickuptime"], errors="coerce") - pt_mu
) / pt_std

if "passenger_count" in train_data.columns and "passenger_count" in test_data.columns:
    train_data["passenger_count"] = pd.to_numeric(
        train_data["passenger_count"], errors="coerce"
    )
    test_data["passenger_count"] = pd.to_numeric(
        test_data["passenger_count"], errors="coerce"
    )
    pc_mu = float(train_data["passenger_count"].mean())
    pc_var = float(train_data["passenger_count"].var())
    pc_std = float(np.sqrt(pc_var)) if pc_var > 0.0 else 1.0
    train_data["passenger_count"] = (train_data["passenger_count"] - pc_mu) / pc_std
    test_data["passenger_count"] = (test_data["passenger_count"] - pc_mu) / pc_std

if "passenger_count" in test_data.columns:
    if test_data["passenger_count"].abs().max() > 10:
        test_data["passenger_count"] = test_data["passenger_count"].clip(
            lower=1, upper=6
        )

for col in ["Difference_longitude", "Difference_latitude"]:
    if col in train_data.columns and col in test_data.columns:
        train_data[col] = train_data[col].clip(-10, 10)
        test_data[col] = test_data[col].clip(-10, 10)

feature_cols = [c for c in train_data.columns if c not in ("key", "fare_amount")]
for c in feature_cols:
    train_data[c] = pd.to_numeric(train_data[c], errors="coerce")
    test_data[c] = pd.to_numeric(test_data[c], errors="coerce")

train_data.replace([np.inf, -np.inf], np.nan, inplace=True)
test_data.replace([np.inf, -np.inf], np.nan, inplace=True)

train_medians = train_data[feature_cols].median(numeric_only=True)
train_data[feature_cols] = train_data[feature_cols].fillna(train_medians)
test_data[feature_cols] = test_data[feature_cols].fillna(train_medians)



## === cell 34
from sklearn.model_selection import train_test_split

X = train_data.drop(["key", "fare_amount"], axis=1)
y = train_data["fare_amount"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.01, random_state=80
)



## === cell 35
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train, y_train)
print(lr.score(X_test, y_test))



## === cell 36
X_test_submit = test_data[X.columns]

pred = np.round(lr.predict(X_test_submit), 2)
pred = np.clip(pred, 0.0, None)



## === cell 37
Submission = pd.DataFrame(data=pred, columns=["fare_amount"])
Submission["key"] = test_data["key"]
Submission = Submission[["key", "fare_amount"]]



## === cell 38
Submission.set_index("key", inplace=True)



## === cell 39
Submission.to_csv("Submission.csv")
