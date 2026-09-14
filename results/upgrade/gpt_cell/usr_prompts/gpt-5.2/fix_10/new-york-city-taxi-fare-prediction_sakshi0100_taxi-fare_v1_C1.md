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

5.68932

# 6. Current score

15.30423

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 936.92806) has done: 'Diagnosis: The crash in cell 29 is due to using the deprecated/removed `normalize` parameter in `sklearn.linear_model.LinearRegression`. In scikit-learn 1.2.2, `normalize` is no longer accepted, so passing it raises `TypeError`. The rest of the cell (fit/score) is fine and must remain unchanged.

Patch summary: Remove the unsupported `normalize=True` argument when constructing `LinearRegression`, keeping the same model type and training/evaluation flow so downstream cells can still use `lr` for prediction.

Updated cells: Only cell 29 is modified.

Compatibility notes for cell k+1: Cell 30 expects a fitted `lr` object with `.predict(...)`; this remains unchanged because `lr` is still a `LinearRegression` instance trained via `.fit(...)`.

Assumptions: No other scikit-learn API incompatibilities exist earlier, and data types/shapes are already compatible with `LinearRegression.fit`/`predict`.'
- What this solution (achieved 937.3389) has done: 'Your RMSE is extremely high because the model is being trained on the first 10M rows without filtering out clearly invalid/outlier target values, so the linear regression gets dominated by huge/negative fares and bad records. To move the score toward the target with minimal disruption, I add a small, standard NYC Taxi cleanup step that only removes impossible coordinates, passenger counts, and extreme/invalid `fare_amount` values before training. I also ensure the one-hot weekday columns align between train and test (in case a weekday is missing in the sampled train slice), which prevents silent feature mismatch. These changes preserve your model (LinearRegression) and feature set logic while making the training data consistent with the competition’s expected domain.'
- What this solution (achieved 1114.26746) has done: 'Your current RMSE is far from the target largely because the `Difference_*` normalization is computed separately on train and test (and with the wrong mean/variance), which badly shifts feature scales at inference time; we instead normalize test using the train mean/variance (same core feature, just consistent scaling). Also, the submission is being written with `key` as an index, but the competition expects `key` as a column; we write the CSV with `key` as a normal column to avoid format-related scoring issues. These are minimal changes that preserve your model (LinearRegression) and all feature engineering, but remove two high-impact sources of error. No training loop, model type, loss, or feature set is changed—only consistent preprocessing and correct submission formatting.'
- What this solution (achieved 15.3192) has done: 'Your current RMSE is far above target because the LinearRegression predictions can go negative or explode on rare test patterns, and RMSE heavily penalizes these outliers; adding a simple non-negative clamp (and a mild upper cap consistent with your training filter) stabilizes predictions without changing the model or features. I also make sure train/test columns are strictly aligned before fitting/predicting to avoid any subtle column-order mismatch from earlier concatenations. These are minimal post-processing and safety checks that preserve your core logic while typically bringing RMSE down substantially on this competition. The submission writing stays the same format (key as a column, correct header) and still produces a valid CSV.'
- What this solution (achieved 15.31904) has done: 'Your current RMSE (15.3192) is far above the target (5.68932), so we should improve performance with minimal, safe changes that keep your LinearRegression and feature logic intact. The biggest likely remaining issue is that `LinearRegression` is being fit without any feature scaling, while your engineered features (especially distances vs one-hot weekdays vs pickuptime) have very different magnitudes; standardizing features (fit on train only, apply to test) typically reduces RMSE a lot without changing the model class or training loop. I add a `StandardScaler` step after the final column alignment (so train/val/test all share identical column order) and keep your existing clipping/rounding and submission format unchanged. This keeps the same core pipeline (same engineered features and same LinearRegression) but makes the linear model behave much closer to what it expects numerically.'
- What this solution (achieved 15.31902) has done: 'Your current RMSE (15.319) is much worse than the target (5.689), so we should improve predictive accuracy with minimal changes while keeping your LinearRegression + existing engineered features. The biggest remaining “minimal but high impact” issue is the nonstandard “variance” normalization in cells 24–26: you divide by `np.var(...)` instead of `np.std(...)`, which overly shrinks/warps those features and can hurt the linear fit; switching to standard deviation keeps the same feature and intent (standardization) but fixes scaling. I also add a tiny numerical guard to avoid any divide-by-zero edge case and keep the rest (scaler, clipping, submission format) unchanged. This should move RMSE substantially toward the target without changing model type, training loop, or feature set.'
- What this solution (achieved 15.3192) has done: 'We need to move your RMSE down from 15.319 toward the 5.689 target, so the smallest high-impact change is to make the linear regression less sensitive to remaining mislabeled/outlier training rows while keeping the same model family and features. I switch `LinearRegression()` to `Ridge()` (still a linear model with the same training flow and prediction semantics) and keep your existing StandardScaler and clipping, which typically stabilizes coefficients and reduces RMSE substantially on this dataset. I also ensure the exact same column alignment is applied to `X_test` as well (using `align`) to prevent any silent feature-order mismatches between the validation split and the fitted model. Everything else (feature engineering, filtering, scaling, train/test split, submission format/path) stays the same and the script still writes a valid `Submission.csv`.'
- What this solution (achieved 15.3192) has done: 'Your RMSE (15.319) is far above the target (5.689), so we should improve accuracy with the smallest change that keeps your same feature engineering and linear-model training flow. The biggest remaining issue is that `pickuptime` is encoded as an HHMM integer (e.g., 930 vs 1030), which creates an artificial discontinuity at hour boundaries and weakens the linear fit; converting it to “minutes since midnight” preserves the same information but makes it linear-friendly. I keep your Ridge + StandardScaler + clipping exactly as-is, and only change how `pickuptime` is derived (for both train and test) so the rest of the pipeline remains identical. This should reduce RMSE materially without changing model class, loss, or the overall approach, and it still writes a valid `Submission.csv`.'
- What this solution (achieved 15.30423) has done: 'Your RMSE is still far above the target, so we need a small but high-impact fix that keeps your exact feature set and linear-model training flow. The main issue is the `abs(x - mean) / std` transform on `Difference_longitude/latitude`, which destroys the sign/linearity and makes the relationship to fare harder for Ridge to learn; switching to standard z-scoring `(x - mean) / std` preserves the same “normalize this feature” intent but restores linear structure. I keep your Ridge + StandardScaler + clipping + submission format unchanged, and only adjust the two normalization cells (and the corresponding test transform) to be consistent and linear-friendly. This should materially lower RMSE while staying within your core logic constraints.'

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



## === cell 7
print(f"Before Dropping null values: {len(train_data)}")
train_data.dropna(inplace=True)
print(f"After Dropping null values: {len(train_data)}")



## === cell 8
train_data = train_data[
    (train_data["fare_amount"] > 0)
    & (train_data["fare_amount"] <= 250)
    & (train_data["passenger_count"] >= 1)
    & (train_data["passenger_count"] <= 6)
    & (train_data["pickup_longitude"].between(-74.3, -72.7))
    & (train_data["dropoff_longitude"].between(-74.3, -72.7))
    & (train_data["pickup_latitude"].between(40.4, 41.3))
    & (train_data["dropoff_latitude"].between(40.4, 41.3))
].copy()



## === cell 9
plot = train_data[:2000].plot.scatter("Difference_longitude", "Difference_latitude")



## === cell 10
train_data = train_data[
    (train_data["Difference_longitude"] < 5.0)
    & (train_data["Difference_latitude"] < 5.0)
]



## === cell 11
ls1 = list(train_data["pickup_datetime"])
for i in range(len(ls1)):
    ls1[i] = ls1[i][11:-7:]
train_data["pickuptime"] = ls1

ls1 = list(test_data["pickup_datetime"])
for i in range(len(ls1)):
    ls1[i] = ls1[i][11:-7:]
test_data["pickuptime"] = ls1



## === cell 12
train_data.head()



## === cell 13
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
test_data.head()



## === cell 14
train_data.drop("pickup_datetime", inplace=True, axis=1)
test_data.drop("pickup_datetime", inplace=True, axis=1)



## === cell 15
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



## === cell 16
train_one_hot = pd.get_dummies(train_data["Weekday"])
test_one_hot = pd.get_dummies(test_data["Weekday"])

train_one_hot, test_one_hot = train_one_hot.align(
    test_one_hot, join="outer", axis=1, fill_value=0
)

train_data = pd.concat([train_data, train_one_hot], axis=1)
test_data = pd.concat([test_data, test_one_hot], axis=1)



## === cell 17
train_data.drop("Weekday", axis=1, inplace=True)
test_data.drop("Weekday", axis=1, inplace=True)



## === cell 18
ls1 = list(train_data["pickuptime"])
for i in range(len(ls1)):
    z = ls1[i].split(":")
    ls1[i] = int(z[0]) * 60 + int(z[1])
train_data["pickuptime"] = ls1

ls1 = list(test_data["pickuptime"])
for i in range(len(ls1)):
    z = ls1[i].split(":")
    ls1[i] = int(z[0]) * 60 + int(z[1])
test_data["pickuptime"] = ls1



## === cell 19
train_data.head()



## === cell 20
R = 6373.0
lat1 = np.asarray(np.radians(train_data["pickup_latitude"]))
lon1 = np.asarray(np.radians(train_data["pickup_longitude"]))
lat2 = np.asarray(np.radians(train_data["dropoff_latitude"]))
lon2 = np.asarray(np.radians(train_data["dropoff_longitude"]))

dlon = lon2 - lon1
dlat = lat2 - lat1
ls1 = []
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



## === cell 22
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



## === cell 23
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



## === cell 24
diff_lon_mean = np.mean(train_data["Difference_longitude"])
diff_lon_std = np.std(train_data["Difference_longitude"])
if diff_lon_std == 0:
    diff_lon_std = 1.0

train_data["Difference_longitude"] = (
    train_data["Difference_longitude"] - diff_lon_mean
) / diff_lon_std



## === cell 25
diff_lat_mean = np.mean(train_data["Difference_latitude"])
diff_lat_std = np.std(train_data["Difference_latitude"])
if diff_lat_std == 0:
    diff_lat_std = 1.0

train_data["Difference_latitude"] = (
    train_data["Difference_latitude"] - diff_lat_mean
) / diff_lat_std



## === cell 26
test_data["Difference_longitude"] = (
    test_data["Difference_longitude"] - diff_lon_mean
) / diff_lon_std

test_data["Difference_latitude"] = (
    test_data["Difference_latitude"] - diff_lat_mean
) / diff_lat_std



## === cell 27
train_data.shape



## === cell 28
test_data.shape



## === cell 29
from sklearn.model_selection import train_test_split

X = train_data.drop(["key", "fare_amount"], axis=1)
y = train_data["fare_amount"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.01, random_state=80
)



## === cell 30
X_test_submit = test_data.drop("key", axis=1)

X_train, X_test_submit = X_train.align(X_test_submit, join="left", axis=1, fill_value=0)
X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_test_submit_scaled = scaler.transform(X_test_submit)

lr = Ridge(alpha=1.0, random_state=80)
lr.fit(X_train_scaled, y_train)
print(lr.score(X_test_scaled, y_test))



## === cell 31
pred = lr.predict(X_test_submit_scaled)
pred = np.clip(pred, 0, 250)
pred = np.round(pred, 2)



## === cell 32
pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/sample_submission.csv"
).head()



## === cell 33
Submission = pd.DataFrame(data=pred, columns=["fare_amount"])
Submission["key"] = test_data["key"]
Submission = Submission[["key", "fare_amount"]]



## === cell 34
Submission.to_csv("Submission.csv", index=False)
print("Wrote Submission.csv with shape:", Submission.shape)
print(Submission.head())
