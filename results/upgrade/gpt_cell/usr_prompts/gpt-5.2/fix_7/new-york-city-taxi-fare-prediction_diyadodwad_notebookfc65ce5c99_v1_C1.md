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

1088.8098

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 936.92806) has done: 'Diagnosis: The crash occurs in cell 32 because `LinearRegression(normalize=True)` is no longer a valid constructor argument in scikit-learn 1.2.2; the `normalize` parameter was deprecated and then removed, so passing it raises a `TypeError`.  
Patch summary: Remove the unsupported `normalize` argument while keeping the same model type (`LinearRegression`) and training/evaluation flow unchanged. This unblocks fitting and scoring so later prediction code can run.  
Updated cells: Only cell 32 is changed.  
Compatibility notes for cell k+1: The variable `lr` remains a fitted `LinearRegression` object with the same interface (`predict`), so cell 33 work unchanged.  
Assumptions: No downstream code depends on the deprecated normalization behavior; we only fix the API incompatibility to prevent the crash.'
- What this solution (achieved 865.37662) has done: 'Your current RMSE is extremely high because the model is being trained on raw fares with lots of outliers/noisy rows, and also because train/test feature scaling is inconsistent (you normalize train using train stats, but normalize test using test stats). To move the score toward the 5.689 target with minimal changes and identical model/training flow, I (1) add small, standard NYC taxi data cleaning filters (fare/passenger/geo bounds) and (2) fix the normalization to use training mean/variance for both train and test. I also keep the exact LinearRegression approach and ensure the submission aligns to `sample_submission.csv` keys and is written as a valid `.csv`. These are small, legitimate fixes that typically reduce RMSE dramatically without changing the core modeling logic.'
- What this solution (achieved 865.37662) has done: 'Your current RMSE is still far from the 5.689 target, so we should make a few minimal, high-impact fixes that keep the same LinearRegression training flow. The biggest remaining issue is feature column mismatch between train and test after one-hot encoding weekdays, which can silently scramble feature alignment and explode error; we align test columns to train columns (same order, fill missing with 0). Next, we apply the same (train-derived) normalization to the other continuous engineered distance features too (not just longitude/latitude diffs), to reduce scale imbalance without changing the model. Finally, we ensure predictions are merged to sample_submission keys deterministically and write a valid `Submission.csv`.'
- What this solution (achieved 865.37662) has done: 'Your RMSE is still massively worse than the 5.689 target, which strongly suggests the submission is not aligned row-for-row with the test `key` (you currently set `key` as the index, so the file header becomes `key,fare_amount` but the first column is unnamed, which Kaggle commonly interprets incorrectly). I make the smallest change to ensure the CSV has exactly two columns `key,fare_amount` as required by the competition, with `key` as a normal column (not an index). I also keep your current merge-with-sample-submission approach but enforce the final row order to exactly match `sample_submission.csv` to avoid any accidental reordering. These changes preserve your model/feature logic and only fix submission semantics, which is the most likely cause of the extreme score.'
- What this solution (achieved 993.39484) has done: 'Your RMSE is still far from the 5.689 target, which strongly suggests the model is learning from mislabeled/outlier trips that slip through the current filters (especially “zero-distance but positive fare”, and extreme long trips still within the broad geo box). I keep your exact LinearRegression pipeline and engineered features, but add two minimal, standard NYC taxi cleaning steps: remove near-zero-distance rides with non-trivial fares, and clip remaining extreme distances to a reasonable upper bound. These are small data-quality fixes that typically reduce RMSE by orders of magnitude without changing the model architecture/loop or the evaluation semantics. I also ensure the same filtering is applied before feature normalization (so statistics aren’t skewed) and keep submission generation unchanged and valid.'
- What this solution (achieved 1088.8098) has done: 'Your RMSE is still far above the target, so we should make the smallest fixes that address the most common “blows up RMSE” issues without changing your LinearRegression pipeline. I (1) correct your feature scaling to use standard deviation (not variance) and remove the unintended `abs()` during centering, since both distort feature magnitudes and can badly harm a linear model, and (2) add one minimal, standard NYC taxi cleaning rule that removes “impossible” trips where distance is large but fare is tiny (these mislabeled/outlier rows heavily degrade RMSE). The model, features (same set), and training loop stay identical; we only adjust preprocessing to be mathematically consistent and less outlier-driven. Submission writing remains the same and still produces a valid `Submission.csv`.'

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
train_data = train_data[
    (train_data["fare_amount"] > 0)
    & (train_data["fare_amount"] <= 250)
    & (train_data["passenger_count"] >= 1)
    & (train_data["passenger_count"] <= 6)
    & (train_data["pickup_longitude"].between(-74.3, -72.9))
    & (train_data["dropoff_longitude"].between(-74.3, -72.9))
    & (train_data["pickup_latitude"].between(40.5, 41.8))
    & (train_data["dropoff_latitude"].between(40.5, 41.8))
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



## === cell 14
train_data.head()



## === cell 15
test_data.head()



## === cell 16
train_data.drop("pickup_datetime", inplace=True, axis=1)
test_data.drop("pickup_datetime", inplace=True, axis=1)



## === cell 17
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



## === cell 18
train_one_hot = pd.get_dummies(train_data["Weekday"])
test_one_hot = pd.get_dummies(test_data["Weekday"])
train_data = pd.concat([train_data, train_one_hot], axis=1)
test_data = pd.concat([test_data, test_one_hot], axis=1)



## === cell 19
train_data.drop("Weekday", axis=1, inplace=True)
test_data.drop("Weekday", axis=1, inplace=True)



## === cell 20
_missing_cols = set(train_data.columns) - set(test_data.columns)
for c in _missing_cols:
    if c not in ["fare_amount"]:  # fare_amount never exists in test anyway
        test_data[c] = 0
_extra_cols = set(test_data.columns) - set(train_data.columns)
for c in _extra_cols:
    test_data.drop(columns=[c], inplace=True)
test_data = test_data[train_data.drop(columns=["fare_amount"]).columns]



## === cell 21
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



## === cell 22
train_data.head()



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
_min_dist = 0.05  # miles (~80m): treat as "no movement"
_max_dist = 60.0  # miles: drop extreme long trips
train_data = train_data[
    (train_data["Distance"] >= _min_dist)
    & (train_data["Distance"] <= _max_dist)
    & (train_data["Pickup_Distance_airport"] <= 100.0)
    & (train_data["Dropoff_Distance_airport"] <= 100.0)
].copy()

train_data = train_data[
    ~((train_data["Distance"] >= 2.0) & (train_data["fare_amount"] < 2.5))
].copy()



## === cell 27
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



## === cell 28
_dl_mean = float(np.mean(train_data["Difference_longitude"]))
_dl_std = float(np.std(train_data["Difference_longitude"]))
if _dl_std == 0.0:
    _dl_std = 1.0

train_data["Difference_longitude"] = (
    train_data["Difference_longitude"] - _dl_mean
) / _dl_std



## === cell 29
_dlat_mean = float(np.mean(train_data["Difference_latitude"]))
_dlat_std = float(np.std(train_data["Difference_latitude"]))
if _dlat_std == 0.0:
    _dlat_std = 1.0

train_data["Difference_latitude"] = (
    train_data["Difference_latitude"] - _dlat_mean
) / _dlat_std



## === cell 30
test_data["Difference_longitude"] = (
    test_data["Difference_longitude"] - _dl_mean
) / _dl_std
test_data["Difference_latitude"] = (
    test_data["Difference_latitude"] - _dlat_mean
) / _dlat_std



## === cell 31
for _col in [
    "Distance",
    "Pickup_Distance_airport",
    "Dropoff_Distance_airport",
    "pickuptime",
]:
    _m = float(np.mean(train_data[_col]))
    _s = float(np.std(train_data[_col]))
    if _s == 0.0:
        _s = 1.0
    train_data[_col] = (train_data[_col] - _m) / _s
    test_data[_col] = (test_data[_col] - _m) / _s



## === cell 32
train_data.shape



## === cell 33
test_data.shape



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
pred = np.round(lr.predict(test_data.drop("key", axis=1)[X.columns]), 2)
pred = np.maximum(pred, 0)



## === cell 37
pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/sample_submission.csv"
).head()



## === cell 38
sample_sub = pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/sample_submission.csv"
)

pred_df = pd.DataFrame({"key": test_data["key"].values, "fare_amount": pred})

sub = sample_sub[["key"]].merge(pred_df, on="key", how="left")

if sub["fare_amount"].isna().any():
    sub["fare_amount"] = sub["fare_amount"].fillna(float(y.mean()))

Submission = sub[["key", "fare_amount"]]



## === cell 39
Submission.to_csv("Submission.csv", index=False)
print("Wrote Submission.csv with shape:", Submission.shape)
print(Submission.head())
