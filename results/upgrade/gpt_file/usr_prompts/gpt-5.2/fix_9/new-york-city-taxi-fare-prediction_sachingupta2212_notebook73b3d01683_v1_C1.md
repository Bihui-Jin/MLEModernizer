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

5.689

# 6. Current score

6.93599

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 936.93312) has done: 'I fix the runtime error caused by using the removed `normalize` parameter in scikit-learn’s `LinearRegression` by switching to an equivalent `Pipeline(StandardScaler(with_mean=False), LinearRegression())`, which preserves the same modeling intent without changing the core approach. I also ensure the one-hot encoded weekday columns align between train and test to prevent silent column mismatch issues at inference time. Finally, I make the submission creation robust by writing `key,fare_amount` exactly as required (without setting `key` as the index) and guaranteeing the output has a `.csv` suffix.'
- What this solution (achieved 752.75683) has done: 'Your score is far worse than the target (RMSE 936.9 vs 5.689), which strongly suggests a data/feature bug rather than model capacity. The single biggest issue is that you’re scaling the test set using the test set’s own mean/variance (data leakage/mismatch), so the model sees differently-scaled features at inference time, causing wildly wrong predictions. I keep your exact feature set and LinearRegression pipeline, but compute normalization stats on the training data once and apply them to both train and test. I also clip negative fare predictions to 0 (a safe constraint for this task) to avoid large RMSE blow-ups from invalid negative outputs.'
- What this solution (achieved 865.07913) has done: 'Your RMSE is still catastrophically high, which almost always means your training labels contain extreme outliers/invalid rows (e.g., negative fares, $500+ fares, bad coordinates) that dominate a plain LinearRegression fit. I keep your exact feature engineering and the same LinearRegression-in-Pipeline approach, but add the standard minimal NYC taxi cleaning filters (fare range, passenger_count range, and valid NYC-ish lat/lon bounds) so the model trains on realistic trips and generalizes to test. I also apply the same coordinate bounds to test (only for safety of engineered distances) without dropping rows, and keep your existing submission format and clipping/rounding. These are small, conventional data-quality fixes that should move RMSE dramatically toward your target without changing the model/training semantics.'
- What this solution (achieved 6.93806) has done: 'Your RMSE is still extremely far from the target, which strongly indicates a feature/scale bug rather than model capacity. The biggest issue in your current code is the custom “normalization” of `Difference_longitude/latitude` using variance (instead of standard deviation), which collapses those features toward ~0 and can destabilize LinearRegression. I keep your exact features and LinearRegression pipeline, but change that step to proper z-score scaling using the training standard deviation (and apply the same train stats to test). I also add a minimal, safe cap on extreme engineered distances (without dropping rows) to prevent a few bad rows from dominating predictions, and keep the submission format unchanged.'
- What this solution (achieved 6.88784) has done: 'You’re already reasonably close to the target (6.94 vs 5.689; lower is better), so the safest way to move toward the target is a small, metric-aligned calibration rather than changing your model or features. The biggest low-risk lever is to correct a systematic scale/bias mismatch by fitting a 1D linear calibrator on a held-out split (same split you already do) and applying it to the test predictions; this preserves the core LinearRegression approach and typically reduces RMSE without altering feature engineering. I also remove rounding of engineered distance features and final predictions (rounding usually hurts RMSE) while keeping clipping to valid ranges. Finally, I keep the submission format identical but ensure the filename is lowercase `.csv` (some Kaggle setups are picky even though usually case-insensitive).'
- What this solution (achieved 6.96704) has done: 'You’re below the target (RMSE 6.88784 vs 5.689; lower is better), so we should make a small, safe improvement without changing your model/features. The biggest likely drag is that your post-hoc calibration is fit on the same holdout used to report the score, which can overfit and not transfer to the test set; we fit the calibrator on a separate calibration split (still from the training set) while keeping the same LinearRegression pipeline and feature set. We also make the calibration numerically safer by solving for both slope and intercept using the raw model predictions from the calibration split, and we keep your clipping semantics unchanged. Everything else (feature engineering, filtering, model, and submission format) is preserved.'
- What this solution (achieved 6.93618) has done: 'We’re still outside the ±10% target band (6.967 vs 5.689), so we need a small, safe improvement without changing your model/features. The main low-risk lever is to avoid a potentially harmful calibration step when it doesn’t generalize: we only apply the post-hoc linear calibrator if it improves RMSE on the holdout; otherwise we fall back to the raw model predictions (this keeps identical core logic and evaluation semantics). We also fit the final LinearRegression pipeline on all available training data after deciding the calibration parameters, which typically gives a small, stable RMSE gain. Finally, we keep your clipping and submission formatting unchanged and ensure feature columns align exactly.'
- What this solution (achieved 6.93599) has done: 'We’re still outside the ±10% target band (need ≲6.26 RMSE, currently 6.936), so we should make a small, safe improvement that doesn’t change your core model or features. The biggest low-risk gain for this competition is to reduce sensitivity to remaining outliers by switching the regressor from plain `LinearRegression` to `Ridge` (still linear, same training loop/semantics, just adds L2 regularization). I keep your exact feature engineering/cleaning and the same scaler+linear pipeline, and I choose a single modest `alpha` value (no CV) to avoid over-tuning. Everything else—including calibration gating, clipping, and submission format—remains unchanged.'

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
train_data = train_data[
    (train_data["fare_amount"] > 0) & (train_data["fare_amount"] <= 250)
]
train_data = train_data[
    (train_data["passenger_count"] >= 1) & (train_data["passenger_count"] <= 6)
]

train_data = train_data[
    (train_data["pickup_longitude"] >= -75)
    & (train_data["pickup_longitude"] <= -72)
    & (train_data["dropoff_longitude"] >= -75)
    & (train_data["dropoff_longitude"] <= -72)
    & (train_data["pickup_latitude"] >= 40)
    & (train_data["pickup_latitude"] <= 42)
    & (train_data["dropoff_latitude"] >= 40)
    & (train_data["dropoff_latitude"] <= 42)
]

train_data = train_data[
    (train_data["Difference_longitude"] < 5.0)
    & (train_data["Difference_latitude"] < 5.0)
]



## === cell 10
ls1 = list(train_data["pickup_datetime"])
for i in range(len(ls1)):
    ls1[i] = ls1[i][11:-7:]
train_data["pickuptime"] = ls1

ls1 = list(test_data["pickup_datetime"])
for i in range(len(ls1)):
    ls1[i] = ls1[i][11:-7:]
test_data["pickuptime"] = ls1
train_data.head()



## === cell 11
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



## === cell 12
train_data.head()



## === cell 13
test_data.head()



## === cell 14
train_data.drop("pickup_datetime", inplace=True, axis=1)
test_data.drop("pickup_datetime", inplace=True, axis=1)



## === cell 15
train_data["Weekday"] = train_data["Weekday"].replace(
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
)
test_data["Weekday"] = test_data["Weekday"].replace(
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
)



## === cell 16
train_one_hot = pd.get_dummies(train_data["Weekday"])
test_one_hot = pd.get_dummies(test_data["Weekday"])
train_one_hot, test_one_hot = train_one_hot.align(
    test_one_hot, join="outer", axis=1, fill_value=0
)

train_data = pd.concat([train_data, train_one_hot], axis=1)
test_data = pd.concat([test_data, test_one_hot], axis=1)
train_data.drop("Weekday", axis=1, inplace=True)
test_data.drop("Weekday", axis=1, inplace=True)



## === cell 17
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



## === cell 18
train_data.head()



## === cell 19
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



## === cell 20
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



## === cell 21
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



## === cell 22
a2 = (
    np.sin(d_lat_dropoff / 2) ** 2
    + np.cos(lat2) * np.cos(lat3) * np.sin(d_lon_dropoff / 2) ** 2
)
c2 = 2 * np.arctan2(np.sqrt(a2), np.sqrt(1 - a2))
distance2 = R * c2

train_data["Dropoff_Distance_airport"] = np.asarray(distance2) * 0.621



## === cell 23
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



## === cell 24
for col in ["Distance", "Pickup_Distance_airport", "Dropoff_Distance_airport"]:
    train_data[col] = pd.to_numeric(train_data[col], errors="coerce")
    test_data[col] = pd.to_numeric(test_data[col], errors="coerce")



## === cell 25
for col in ["Distance", "Pickup_Distance_airport", "Dropoff_Distance_airport"]:
    train_data[col] = train_data[col].clip(lower=0, upper=200)
    test_data[col] = test_data[col].clip(lower=0, upper=200)



## === cell 26
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



## === cell 27
for col in ["Difference_longitude", "Difference_latitude"]:
    train_mean = float(train_data[col].mean())
    train_std = float(train_data[col].std(ddof=0))
    if train_std == 0 or np.isnan(train_std):
        train_std = 1.0  # safety to avoid divide-by-zero
    train_data[col] = (train_data[col] - train_mean) / train_std
    test_data[col] = (test_data[col] - train_mean) / train_std



## === cell 28
train_data.shape



## === cell 29
test_data.shape



## === cell 30
from sklearn.model_selection import train_test_split
from sklearn.linear_model import (
    Ridge,
)  # CHANGE: L2-regularized linear model to reduce outlier sensitivity and move RMSE down toward target.
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

X = train_data.drop(["key", "fare_amount"], axis=1)
y = train_data["fare_amount"]

X_train_full, X_holdout, y_train_full, y_holdout = train_test_split(
    X, y, test_size=0.01, random_state=80
)
X_train, X_cal, y_train, y_cal = train_test_split(
    X_train_full, y_train_full, test_size=0.01, random_state=81
)

ridge_alpha = 1.0
lr = Pipeline(
    [
        ("scaler", StandardScaler(with_mean=False)),
        ("model", Ridge(alpha=ridge_alpha, random_state=80)),
    ]
)
lr.fit(X_train, y_train)

print("Holdout R^2 (uncalibrated):", lr.score(X_holdout, y_holdout))



## === cell 31
cal_pred = lr.predict(X_cal)
cal_pred = np.clip(cal_pred, 0, None)

A = np.vstack([cal_pred, np.ones_like(cal_pred)]).T
a_cal, b_cal = np.linalg.lstsq(A, y_cal.to_numpy(), rcond=None)[0]
print(f"Calibration (fit on calibration split): y ≈ {a_cal:.6f} * pred + {b_cal:.6f}")

hold_pred_raw = lr.predict(X_holdout)
hold_pred_raw = np.clip(hold_pred_raw, 0, None)
hold_rmse_raw = float(np.sqrt(np.mean((hold_pred_raw - y_holdout.to_numpy()) ** 2)))

hold_pred_cal = a_cal * hold_pred_raw + b_cal
hold_pred_cal = np.clip(hold_pred_cal, 0, None)
hold_rmse_cal = float(np.sqrt(np.mean((hold_pred_cal - y_holdout.to_numpy()) ** 2)))

print("Holdout RMSE (raw):", hold_rmse_raw)
print("Holdout RMSE (calibrated):", hold_rmse_cal)

use_calibration = hold_rmse_cal < hold_rmse_raw
print("Using calibration for test predictions:", use_calibration)

lr_final = Pipeline(
    [
        ("scaler", StandardScaler(with_mean=False)),
        ("model", Ridge(alpha=ridge_alpha, random_state=80)),
    ]
)
lr_final.fit(X, y)

test_X = test_data.drop("key", axis=1)
test_X = test_X.reindex(columns=X.columns, fill_value=0)

pred = lr_final.predict(test_X)
pred = np.clip(pred, 0, None)
if use_calibration:
    pred = a_cal * pred + b_cal
    pred = np.clip(pred, 0, None)

pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/sample_submission.csv"
).head()



## === cell 32
submission = pd.DataFrame({"key": test_data["key"].values, "fare_amount": pred})
submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote: submission.csv")
