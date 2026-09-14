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

3.9

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

750.85581

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 936.94064) has done: 'The run is failing because scikit-learn 1.2 removed the `normalize` argument from `LinearRegression`, so the model never gets created and downstream cells error. I remove that deprecated argument (keeping the same linear regression core logic), and add a minimal preprocessing pipeline (standardization) that replaces the old normalize behavior and typically improves RMSE for this competition. I also ensure train/test one-hot columns are aligned (to avoid silent column mismatches), and I write a valid `Submission.csv` with exactly the required columns (`key,fare_amount`) without changing the overall feature engineering or training approach. Finally, I add a small safety clip to keep predictions non-negative, which is consistent with the target and helps stability.'
- What this solution (achieved 752.76916) has done: 'Your RMSE is extremely high because the test features are being scaled/normalized using *test-set statistics* (cells 26–28), while the model sees differently scaled training features—this distribution mismatch can blow up predictions. I keep your exact feature engineering and LinearRegression pipeline, but change the normalization so test uses the *training* mean/variance for the two “Difference_*” columns (a minimal, semantics-preserving fix). I also ensure no NaNs/infs sneak in after those divisions (which can destabilize LinearRegression) and keep the same submission schema/output file. These changes should move the score sharply downward (better) toward your 5.68914 target without altering the overall approach.'
- What this solution (achieved 750.85758) has done: 'Your current RMSE is still extremely high for this competition, which strongly suggests a remaining data-quality issue rather than a model issue. I keep your exact feature engineering and LinearRegression pipeline, but add the standard NYC Taxi Fare “sanity filters” on the training data (valid lat/long ranges, passenger_count range, and reasonable fare bounds) to remove clearly bad rows that otherwise dominate RMSE. I also ensure we don’t accidentally train on infinities/NaNs across *all* numeric features (not just the two scaled columns), which can destabilize the linear fit. These are minimal, well-scoped changes that typically move RMSE dramatically downward toward your 5.68914 target without altering your core logic or approach.'
- What this solution (achieved 750.85581) has done: 'Your RMSE is still orders of magnitude too high, so we should focus on a minimal “correctness” fix rather than model tweaks: you’re currently using `var` (variance) as if it were `std` (standard deviation) when scaling the Difference_* features, which massively distorts feature magnitudes and can blow up LinearRegression predictions. I keep your exact feature set and LinearRegression+StandardScaler pipeline, but change those two normalizations to use training-set standard deviation (and keep train/test using the same train-derived stats). I also add the standard latitude/longitude bounds sanity filter (minimal, common for this competition) to prevent extreme coordinates from dominating the fit. These changes should substantially lower RMSE toward your 5.68914 target while preserving the core logic and producing the same submission format.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import time
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
train_data["pickuptime"] = train_data["pickup_datetime"].astype(str).str.slice(11, -7)
test_data["pickuptime"] = test_data["pickup_datetime"].astype(str).str.slice(11, -7)



## === cell 12
train_data.head()



## === cell 13
train_data["Weekday"] = pd.to_datetime(
    train_data["pickup_datetime"].astype(str).str.slice(0, -4), errors="coerce"
).dt.weekday
test_data["Weekday"] = pd.to_datetime(
    test_data["pickup_datetime"].astype(str).str.slice(0, -4), errors="coerce"
).dt.weekday



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

train_one_hot, test_one_hot = train_one_hot.align(
    test_one_hot, join="outer", axis=1, fill_value=0
)

train_data = pd.concat([train_data, train_one_hot], axis=1)
test_data = pd.concat([test_data, test_one_hot], axis=1)



## === cell 19
train_data.drop("Weekday", axis=1, inplace=True)
test_data.drop("Weekday", axis=1, inplace=True)



## === cell 20
pt_train = train_data["pickuptime"].astype(str).str.split(":", expand=True)
train_data["pickuptime"] = (
    pt_train[0].astype(int) * 100 + pt_train[1].astype(int)
).astype(int)

pt_test = test_data["pickuptime"].astype(str).str.split(":", expand=True)
test_data["pickuptime"] = (
    pt_test[0].astype(int) * 100 + pt_test[1].astype(int)
).astype(int)



## === cell 21
train_data.head()



## === cell 22
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



## === cell 23
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



## === cell 24
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



## === cell 25
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



## === cell 26
before = len(train_data)
train_data = train_data[
    (train_data["pickup_latitude"].between(-90, 90))
    & (train_data["dropoff_latitude"].between(-90, 90))
    & (train_data["pickup_longitude"].between(-180, 180))
    & (train_data["dropoff_longitude"].between(-180, 180))
]
print("Dropped rows by lat/long bounds sanity filter:", before - len(train_data))



## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3804         try:
-> 3805             return self._engine.get_loc(casted_key)
   3806         except KeyError as err:

index.pyx in pandas._libs.index.IndexEngine.get_loc()

index.pyx in pandas._libs.index.IndexEngine.get_loc()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

KeyError: 'pickup_latitude'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/1814955101.py in <cell line: 0>()
      3 before = len(train_data)
      4 train_data = train_data[
----> 5     (train_data["pickup_latitude"].between(-90, 90))
      6     & (train_data["dropoff_latitude"].between(-90, 90))
      7     & (train_data["pickup_longitude"].between(-180, 180))

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4100             if self.columns.nlevels > 1:
   4101                 return self._getitem_multilevel(key)
-> 4102             indexer = self.columns.get_loc(key)
   4103             if is_integer(indexer):
   4104                 indexer = [indexer]

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3810             ):
   3811                 raise InvalidIndexError(key)
-> 3812             raise KeyError(key) from err
   3813         except TypeError:
   3814             # If we have a listlike key, _check_indexing_error will raise

KeyError: 'pickup_latitude'

## === cell 27
before = len(train_data)
train_data = train_data[
    (train_data["fare_amount"] > 0.0)
    & (train_data["fare_amount"] <= 250.0)
    & (train_data["passenger_count"] >= 1)
    & (train_data["passenger_count"] <= 6)
]
print("Dropped rows by fare/passenger sanity filters:", before - len(train_data))



## === cell 28
dl_mean = float(train_data["Difference_longitude"].mean())
dl_std = float(train_data["Difference_longitude"].std())
if not np.isfinite(dl_std) or dl_std == 0.0:
    dl_std = 1.0

train_data["Difference_longitude"] = np.abs(
    train_data["Difference_longitude"] - dl_mean
)
train_data["Difference_longitude"] = train_data["Difference_longitude"] / dl_std

test_data["Difference_longitude"] = np.abs(test_data["Difference_longitude"] - dl_mean)
test_data["Difference_longitude"] = test_data["Difference_longitude"] / dl_std



## === cell 29
dlat_mean = float(train_data["Difference_latitude"].mean())
dlat_std = float(train_data["Difference_latitude"].std())
if not np.isfinite(dlat_std) or dlat_std == 0.0:
    dlat_std = 1.0

train_data["Difference_latitude"] = np.abs(
    train_data["Difference_latitude"] - dlat_mean
)
train_data["Difference_latitude"] = train_data["Difference_latitude"] / dlat_std

test_data["Difference_latitude"] = np.abs(test_data["Difference_latitude"] - dlat_mean)
test_data["Difference_latitude"] = test_data["Difference_latitude"] / dlat_std



## === cell 30
for df in (train_data, test_data):
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)

before = len(train_data)
train_data.dropna(inplace=True)
print(
    "Dropped rows with NaN/inf anywhere after feature engineering:",
    before - len(train_data),
)

test_data.fillna(0.0, inplace=True)



## === cell 31
train_data.shape



## === cell 32
test_data.shape



## === cell 33
from sklearn.model_selection import train_test_split

X = train_data.drop(["key", "fare_amount"], axis=1)
y = train_data["fare_amount"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.01, random_state=80
)



## === cell 34
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

lr = Pipeline(
    steps=[
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("model", LinearRegression()),
    ]
)

lr.fit(X_train, y_train)
print(lr.score(X_test, y_test))



## === cell 35
pred = lr.predict(test_data.drop("key", axis=1))
pred = np.round(pred, 2)
pred = np.clip(pred, 0.0, None)



## === cell 36
pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/sample_submission.csv"
).head()



## === cell 37
Submission = pd.DataFrame({"key": test_data["key"].values, "fare_amount": pred})
Submission = Submission[["key", "fare_amount"]]
Submission.head()



## === cell 38
assert len(Submission) == len(test_data)
assert list(Submission.columns) == ["key", "fare_amount"]



## === cell 39
Submission.to_csv("Submission.csv", index=False)
print("Wrote Submission.csv with shape:", Submission.shape)
print(Submission.head())
