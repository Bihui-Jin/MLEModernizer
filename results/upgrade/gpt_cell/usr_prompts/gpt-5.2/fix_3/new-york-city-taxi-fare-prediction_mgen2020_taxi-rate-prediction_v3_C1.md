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

914.17869

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 936.92806) has done: 'Diagnosis: The crash happens in cell 19 because `LinearRegression(normalize=True)` is no longer a valid argument in scikit-learn 1.2.2; the `normalize` parameter was deprecated and then removed, so passing it raises `TypeError`. The rest of the cell (train/test split, `.fit`, `.score`) is fine and should remain unchanged.

Patch summary: Remove the unsupported `normalize=True` argument while keeping the same model class (`LinearRegression`) and the same training/evaluation flow. This unblocks model fitting so `lr` exists for the next cell.

Updated cells: Only cell 19 is changed.

Compatibility notes for cell k+1: Cell 20 expects a fitted `lr` object and uses `lr.predict(...)`; this patch keeps `lr` as a `LinearRegression` instance and fits it exactly as before, so the interface remains compatible.

Assumptions: No other hidden preprocessing is required; fixing the constructor argument is sufficient to proceed.'
- What this solution (achieved 914.17869) has done: 'Your current RMSE is extremely high because the model is being trained on many invalid/outlier rows (e.g., negative/zero fares, impossible coordinates, extreme fares) and because the train/test feature columns can silently mismatch due to one-hot encoding weekdays separately. I add minimal, standard NYC Taxi cleaning filters (fare bounds, passenger_count bounds, NYC coordinate bounds) to remove data that breaks a linear model, which should move RMSE sharply down toward your target. I also make the weekday one-hot encoding consistent between train and test by concatenating before `get_dummies`, preventing column misalignment that can explode predictions. Finally, I clip negative predictions to 0 (fares can’t be negative), which typically reduces RMSE without changing the modeling approach.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
td = pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv", nrows=10_000_000
)
td.head()  # td:train data # ted:test data



## === cell 2
td.shape



## === cell 3
td.info()



## === cell 4
ted = pd.read_csv("/kaggle/input/new-york-city-taxi-fare-prediction/test.csv")
ted.head()



## === cell 5
ted.info()



## === cell 6
td.isna().sum()



## === cell 7
td["Difference_longitude"] = np.abs(
    np.asarray(td["pickup_longitude"] - td["dropoff_longitude"])
)
td["Difference_latitude"] = np.abs(
    np.asarray(td["pickup_latitude"] - td["dropoff_latitude"])
)

ted["Difference_longitude"] = np.abs(
    np.asarray(ted["pickup_longitude"] - ted["dropoff_longitude"])
)
ted["Difference_latitude"] = np.abs(
    np.asarray(ted["pickup_latitude"] - ted["dropoff_latitude"])
)



## === cell 8
print(f"Before Dropping null values: {len(td)}")
td.dropna(inplace=True)
print(f"After Dropping null values: {len(td)}")



## === cell 9
plot = td[:2000].plot.scatter("Difference_longitude", "Difference_latitude")



## === cell 10
td = td[(td["Difference_longitude"] < 5.0) & (td["Difference_latitude"] < 5.0)]



## === cell 11
td = td[
    (td["fare_amount"] > 0)
    & (td["fare_amount"] <= 500)
    & (td["passenger_count"] >= 1)
    & (td["passenger_count"] <= 6)
    & (td["pickup_longitude"].between(-75, -72))
    & (td["dropoff_longitude"].between(-75, -72))
    & (td["pickup_latitude"].between(40, 42))
    & (td["dropoff_latitude"].between(40, 42))
].copy()
td.shape



## === cell 12
l1 = list(td["pickup_datetime"])
for i in range(len(l1)):
    l1[i] = l1[i][11:-7:]
td["pickuptime"] = l1

l1 = list(ted["pickup_datetime"])
for i in range(len(l1)):
    l1[i] = l1[i][11:-7:]
ted["pickuptime"] = l1



## === cell 13
td.head()



## === cell 14
l1 = list(td["pickup_datetime"])
for i in range(len(l1)):
    l1[i] = l1[i][:-4:]
    l1[i] = pd.Timestamp(l1[i])
    l1[i] = l1[i].weekday()
td["Weekday"] = l1

l1 = list(ted["pickup_datetime"])
for i in range(len(l1)):
    l1[i] = l1[i][:-4:]
    l1[i] = pd.Timestamp(l1[i])
    l1[i] = l1[i].weekday()
ted["Weekday"] = l1

td.head()



## === cell 15
ted.head()



## === cell 16
td.drop("pickup_datetime", inplace=True, axis=1)
ted.drop("pickup_datetime", inplace=True, axis=1)

td["Weekday"].replace(
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
ted["Weekday"].replace(
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

_all = pd.concat([td[["Weekday"]], ted[["Weekday"]]], axis=0)
_all_dum = pd.get_dummies(_all["Weekday"])
th = _all_dum.iloc[: len(td)].reset_index(drop=True)
teh = _all_dum.iloc[len(td) :].reset_index(drop=True)

td = td.reset_index(drop=True)
ted = ted.reset_index(drop=True)

td = pd.concat([td, th], axis=1)
ted = pd.concat([ted, teh], axis=1)

td.drop("Weekday", axis=1, inplace=True)
ted.drop("Weekday", axis=1, inplace=True)

l1 = list(td["pickuptime"])
for i in range(len(l1)):
    z = l1[i].split(":")
    l1[i] = int(z[0]) * 100 + int(z[1])
td["pickuptime"] = l1

l1 = list(ted["pickuptime"])
for i in range(len(l1)):
    z = l1[i].split(":")
    l1[i] = int(z[0]) * 100 + int(z[1])
ted["pickuptime"] = l1

td.head()



## === cell 17
R = 6373.0
lat1 = np.asarray(np.radians(td["pickup_latitude"]))
lon1 = np.asarray(np.radians(td["pickup_longitude"]))
lat2 = np.asarray(np.radians(td["dropoff_latitude"]))
lon2 = np.asarray(np.radians(td["dropoff_longitude"]))

dlon = lon2 - lon1
dlat = lat2 - lat1
l1 = []
a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
distance = R * c

td["Distance"] = np.asarray(distance) * 0.621

lat1 = np.asarray(np.radians(ted["pickup_latitude"]))
lon1 = np.asarray(np.radians(ted["pickup_longitude"]))
lat2 = np.asarray(np.radians(ted["dropoff_latitude"]))
lon2 = np.asarray(np.radians(ted["dropoff_longitude"]))

dlon = lon2 - lon1
dlat = lat2 - lat1

a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
distance = R * c
ted["Distance"] = np.asarray(distance) * 0.621



## === cell 18
R = 6373.0
lat1 = np.asarray(np.radians(td["pickup_latitude"]))
lon1 = np.asarray(np.radians(td["pickup_longitude"]))
lat2 = np.asarray(np.radians(td["dropoff_latitude"]))
lon2 = np.asarray(np.radians(td["dropoff_longitude"]))

lat3 = np.zeros(len(td)) + np.radians(40.6413111)
lon3 = np.zeros(len(td)) + np.radians(-73.7781391)
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
td["Pickup_Distance_airport"] = np.asarray(distance1) * 0.621

a2 = (
    np.sin(d_lat_dropoff / 2) ** 2
    + np.cos(lat2) * np.cos(lat3) * np.sin(d_lon_dropoff / 2) ** 2
)
c2 = 2 * np.arctan2(np.sqrt(a2), np.sqrt(1 - a2))
distance2 = R * c2

td["Dropoff_Distance_airport"] = np.asarray(distance2) * 0.621

lat1 = np.asarray(np.radians(ted["pickup_latitude"]))
lon1 = np.asarray(np.radians(ted["pickup_longitude"]))
lat2 = np.asarray(np.radians(ted["dropoff_latitude"]))
lon2 = np.asarray(np.radians(ted["dropoff_longitude"]))

lat3 = np.zeros(len(ted)) + np.radians(40.6413111)
lon3 = np.zeros(len(ted)) + np.radians(-73.7781391)
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
ted["Pickup_Distance_airport"] = np.asarray(distance1) * 0.621

a2 = (
    np.sin(d_lat_dropoff / 2) ** 2
    + np.cos(lat2) * np.cos(lat3) * np.sin(d_lon_dropoff / 2) ** 2
)
c2 = 2 * np.arctan2(np.sqrt(a2), np.sqrt(1 - a2))
distance2 = R * c2

ted["Dropoff_Distance_airport"] = np.asarray(distance2) * 0.621

td["Distance"] = np.round(td["Distance"], 2)
td["Pickup_Distance_airport"] = np.round(td["Pickup_Distance_airport"], 2)
td["Dropoff_Distance_airport"] = np.round(td["Dropoff_Distance_airport"], 2)
ted["Distance"] = np.round(ted["Distance"], 2)
ted["Pickup_Distance_airport"] = np.round(ted["Pickup_Distance_airport"], 2)
ted["Dropoff_Distance_airport"] = np.round(ted["Dropoff_Distance_airport"], 2)

td.drop(
    ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"],
    axis=1,
    inplace=True,
)
ted.drop(
    ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"],
    axis=1,
    inplace=True,
)

td["Difference_longitude"] = np.abs(
    td["Difference_longitude"] - np.mean(td["Difference_longitude"])
)
td["Difference_longitude"] = td["Difference_longitude"] / np.var(
    td["Difference_longitude"]
)

td["Difference_latitude"] = np.abs(
    td["Difference_latitude"] - np.mean(td["Difference_latitude"])
)
td["Difference_latitude"] = td["Difference_latitude"] / np.var(
    td["Difference_latitude"]
)

ted["Difference_longitude"] = np.abs(
    ted["Difference_longitude"] - np.mean(ted["Difference_longitude"])
)
ted["Difference_longitude"] = ted["Difference_longitude"] / np.var(
    ted["Difference_longitude"]
)

ted["Difference_latitude"] = np.abs(
    ted["Difference_latitude"] - np.mean(ted["Difference_latitude"])
)
ted["Difference_latitude"] = ted["Difference_latitude"] / np.var(
    ted["Difference_latitude"]
)

td.shape



## === cell 19
ted.shape



## === cell 20
from sklearn.model_selection import train_test_split

X = td.drop(["key", "fare_amount"], axis=1)
y = td["fare_amount"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.01, random_state=80
)

from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train, y_train)
print(lr.score(X_test, y_test))



## === cell 21
pred = np.round(lr.predict(ted.drop("key", axis=1)), 2)
pred = np.clip(pred, 0, None)
pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/sample_submission.csv"
).head()



## === cell 22
Submission = pd.DataFrame(data=pred, columns=["fare_amount"])
Submission["key"] = ted["key"]
Submission = Submission[["key", "fare_amount"]]

Submission.set_index("key", inplace=True)

Submission.to_csv("Submission.csv")
