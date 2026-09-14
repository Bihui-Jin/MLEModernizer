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

15.25324

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 936.94064) has done: 'I fix the immediate runtime error by removing the deprecated `normalize` argument from `LinearRegression` (it was removed in recent scikit-learn versions), so the model can train. To keep the original intent of normalization without changing the core model, I wrap the same `LinearRegression` in a `Pipeline` with `StandardScaler` (this is the modern equivalent and should also improve RMSE vs the broken run). I also ensure the one-hot weekday columns align between train and test (avoids silent column mismatch) and write a valid `submission.csv` with exactly `key,fare_amount` columns.'
- What this solution (achieved 762.59107) has done: 'Your RMSE is exploding because the manual “normalization” in cells 25–27 is inconsistent between train and test (each is centered/scaled using its own distribution, and even uses variance instead of std), which makes the model see different feature scales at inference and leads to wildly wrong predictions. To move toward the target score with minimal disruption, I keep your exact feature set and LinearRegression+StandardScaler pipeline, but remove those manual scaling cells so scaling is learned on the training split and applied consistently. I also add a small safety clip to avoid negative fares (common for linear models and hurts RMSE) without changing the model itself. The script still runs end-to-end and writes a valid `submission.csv` with `key,fare_amount`.'
- What this solution (achieved 986.35052) has done: 'Your current RMSE is far from the target, so we need a couple of minimal, high-impact correctness fixes rather than tuning. The biggest issue is that training data isn’t being filtered for invalid/outlier `fare_amount` values (including negative and extremely large fares), which badly skews a linear regression fit; we add standard NYC Taxi Fare sanity filters that are widely used and don’t change your feature/model logic. Second, your train-test split is random, which can mix “future” and “past” rides and lead to unstable evaluation; we switch to a deterministic time-based split using `pickup_datetime` while keeping the same model and features. Finally, we keep your existing feature engineering intact, and still write `submission.csv` with exactly `key,fare_amount`.'
- What this solution (achieved 15.25324) has done: 'Your RMSE is still extremely high, which strongly suggests the model is being trained on corrupted targets and/or mismatched rows rather than just “weak features.” I make two minimal correctness fixes that preserve your exact feature engineering and LinearRegression pipeline: (1) load a smaller but clean subset of `train.csv` by reading only needed columns and dropping bad rows early (fixes misalignment/NaN handling and prevents the model being dominated by garbage), and (2) remove the second re-read of `train.csv` for datetimes and instead parse `pickup_datetime` once from the already-loaded data before dropping it (fixes index misalignment in the time-based split). Finally, I keep your submission format identical and keep the non-negative clipping, but also cap extreme predictions to the same upper bound used in training (250) to reduce RMSE impact from outliers without changing the model.'
- What this solution (achieved 15.25324) has done: 'Your current RMSE (15.25) is far above the target (5.69), so the most likely issue is not model capacity but a correctness/feature bug that makes predictions systematically wrong. The biggest minimal fix is that your time/weekday features are extracted via brittle string slicing; depending on timestamp formatting, this can silently produce incorrect hours/weekdays, harming the linear model a lot. I keep your exact features and LinearRegression+StandardScaler pipeline, but derive `pickuptime` and `Weekday` directly from the already-parsed `_pickup_dt` (same semantics, just correct and consistent). I also ensure `passenger_count` is numeric and missing datetimes are filled using the training median for both train and test to avoid distribution shift.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
TRAIN_PATH = "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv"
TEST_PATH = "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv"

usecols_train = [
    "key",
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
usecols_test = [
    "key",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]

train_data = pd.read_csv(TRAIN_PATH, usecols=usecols_train, nrows=2_000_000)
train_data.head()



## === cell 2
train_data.shape



## === cell 3
test_data = pd.read_csv(TEST_PATH, usecols=usecols_test)
test_data.head()



## === cell 4
test_data.info()



## === cell 5
train_data.isna().sum()



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
_ = train_data[:2000].plot.scatter("Difference_longitude", "Difference_latitude")



## === cell 9
train_data = train_data[
    (train_data["Difference_longitude"] < 5.0)
    & (train_data["Difference_latitude"] < 5.0)
]



## === cell 10
train_data = train_data[
    (train_data["fare_amount"] > 0) & (train_data["fare_amount"] <= 250)
]



## === cell 11
train_data = train_data[
    (train_data["pickup_longitude"].between(-75, -72))
    & (train_data["dropoff_longitude"].between(-75, -72))
    & (train_data["pickup_latitude"].between(40, 42))
    & (train_data["dropoff_latitude"].between(40, 42))
]



## === cell 12
train_data["_pickup_dt"] = pd.to_datetime(
    train_data["pickup_datetime"], errors="coerce"
)
test_data["_pickup_dt"] = pd.to_datetime(test_data["pickup_datetime"], errors="coerce")

train_median_dt = train_data["_pickup_dt"].median()
train_data["_pickup_dt"] = train_data["_pickup_dt"].fillna(train_median_dt)
test_data["_pickup_dt"] = test_data["_pickup_dt"].fillna(train_median_dt)



## === cell 13
train_data["pickuptime"] = (
    train_data["_pickup_dt"].dt.hour * 100 + train_data["_pickup_dt"].dt.minute
)
test_data["pickuptime"] = (
    test_data["_pickup_dt"].dt.hour * 100 + test_data["_pickup_dt"].dt.minute
)

train_data["Weekday"] = train_data["_pickup_dt"].dt.weekday
test_data["Weekday"] = test_data["_pickup_dt"].dt.weekday



## === cell 14
train_data.head()



## === cell 15
test_data.head()



## === cell 16
train_data.head()



## === cell 17
test_data.head()



## === cell 18
train_data.drop("pickup_datetime", inplace=True, axis=1)
test_data.drop("pickup_datetime", inplace=True, axis=1)



## === cell 19
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



## === cell 20
train_one_hot = pd.get_dummies(train_data["Weekday"])
test_one_hot = pd.get_dummies(test_data["Weekday"])

train_one_hot, test_one_hot = train_one_hot.align(
    test_one_hot, join="outer", axis=1, fill_value=0
)

train_data = pd.concat([train_data, train_one_hot], axis=1)
test_data = pd.concat([test_data, test_one_hot], axis=1)



## === cell 21
train_data.drop("Weekday", axis=1, inplace=True)
test_data.drop("Weekday", axis=1, inplace=True)



## === cell 22
train_data["passenger_count"] = pd.to_numeric(
    train_data["passenger_count"], errors="coerce"
)
test_data["passenger_count"] = pd.to_numeric(
    test_data["passenger_count"], errors="coerce"
)
train_data.dropna(subset=["passenger_count"], inplace=True)
test_data["passenger_count"] = test_data["passenger_count"].fillna(
    train_data["passenger_count"].median()
)



## === cell 23
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



## === cell 24
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



## === cell 25
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
pass



## === cell 28
train_data.shape



## === cell 29
test_data.shape



## === cell 30
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler



## === cell 31
pickup_dt = train_data["_pickup_dt"]
order = np.argsort(pickup_dt.values)
train_idx = train_data.index[order]

split = int(len(train_idx) * 0.99)
idx_tr = train_idx[:split]
idx_va = train_idx[split:]

X = train_data.drop(["key", "fare_amount", "_pickup_dt"], axis=1)
y = train_data["fare_amount"]

X_train = X.loc[idx_tr]
y_train = y.loc[idx_tr]
X_test = X.loc[idx_va]
y_test = y.loc[idx_va]



## === cell 32
lr = Pipeline(
    steps=[
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("model", LinearRegression()),
    ]
)

lr.fit(X_train, y_train)
print(lr.score(X_test, y_test))



## === cell 33
pred = lr.predict(test_data.drop(["key", "_pickup_dt"], axis=1))
pred = np.clip(pred, 0, 250)
pred = np.round(pred, 2)



## === cell 34
Submission = pd.DataFrame(data=pred, columns=["fare_amount"])
Submission["key"] = test_data["key"]
Submission = Submission[["key", "fare_amount"]]



## === cell 35
Submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", Submission.shape)
print(Submission.head())
