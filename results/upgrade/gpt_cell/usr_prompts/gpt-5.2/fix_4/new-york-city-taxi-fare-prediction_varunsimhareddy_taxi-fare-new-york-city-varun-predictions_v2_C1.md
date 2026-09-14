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

5.68916

# 6. Current score

750.77599

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 936.92806) has done: 'Diagnosis: The crash happens because scikit-learn 1.2 removed the `normalize` parameter from `sklearn.linear_model.LinearRegression`, so passing `normalize=True` raises a `TypeError` during estimator construction. The rest of the pipeline expects `lr` to be a fitted `LinearRegression` model and uses it in cell 33 for prediction.  
Patch summary: Remove the unsupported `normalize` argument and keep the same model class and fitting/scoring logic so downstream cells remain compatible and execution unblocks.  
Updated cells: Only cell 32 is changed.  
Compatibility notes for cell k+1: `lr` remains a fitted `LinearRegression` instance, so `lr.predict(...)` in cell 33 works unchanged.  
Assumptions: No additional preprocessing (e.g., explicit scaling) is required to proceed; we only address the API incompatibility causing the crash.'
- What this solution (achieved 752.76803) has done: 'Your RMSE is extremely high because the model is trained on a feature set that includes distance-related columns, but those same columns are also (accidentally) used as inputs at test time without being normalized/processed in the same way as train (and `get_dummies` can also create mismatched weekday columns between train/test). I keep your exact feature engineering and LinearRegression model, but make train/test preprocessing consistent by computing normalization parameters on the training data and applying them to both train and test, and by aligning one-hot encoded weekday columns so train/test have identical columns. These are minimal changes that preserve your approach while fixing a core train–test feature mismatch that drives RMSE toward a reasonable range (closer to the 5.689 target). The script still run end-to-end and write `Submission.csv` with the required `key,fare_amount` columns.'
- What this solution (achieved 750.77599) has done: 'Your current RMSE is far above the target, so we should improve performance with minimal, metric-aligned fixes while keeping your LinearRegression + existing features. The biggest remaining issue is that you normalize by *variance* (not standard deviation), which badly scales features and hurts LinearRegression; switching to standard deviation is a tiny change that should move RMSE much closer to the target range. I also add a light, competition-standard cleaning step to remove obviously invalid fares (<=0 or extreme) and passenger counts, because those outliers disproportionately distort an MSE/RMSE objective. Finally, I keep your one-hot alignment and submission formatting intact so the pipeline still runs end-to-end and writes a valid `Submission.csv`.'

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
train_dt = pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv", nrows=10_000_000
)
train_dt.head()



## === cell 2
train_dt.shape



## === cell 3
train_dt.info()



## === cell 4
test_dt = pd.read_csv("/kaggle/input/new-york-city-taxi-fare-prediction/test.csv")
test_dt.head()



## === cell 5
test_dt.info()



## === cell 6
train_dt.isna().sum()



## === cell 7
train_dt["Difference_longitude"] = np.abs(
    np.asarray(train_dt["pickup_longitude"] - train_dt["dropoff_longitude"])
)
train_dt["Difference_latitude"] = np.abs(
    np.asarray(train_dt["pickup_latitude"] - train_dt["dropoff_latitude"])
)

test_dt["Difference_longitude"] = np.abs(
    np.asarray(test_dt["pickup_longitude"] - test_dt["dropoff_longitude"])
)
test_dt["Difference_latitude"] = np.abs(
    np.asarray(test_dt["pickup_latitude"] - test_dt["dropoff_latitude"])
)



## === cell 8
print(f"Before Dropping null values: {len(train_dt)}")
train_dt.dropna(inplace=True)
print(f"After Dropping null values: {len(train_dt)}")



## === cell 9
plot = train_dt[:2000].plot.scatter("Difference_longitude", "Difference_latitude")



## === cell 10
train_dt = train_dt[
    (train_dt["Difference_longitude"] < 5.0) & (train_dt["Difference_latitude"] < 5.0)
]



## === cell 11
train_dt = train_dt[
    (train_dt["fare_amount"] > 0.0)
    & (train_dt["fare_amount"] < 250.0)
    & (train_dt["passenger_count"] >= 1)
    & (train_dt["passenger_count"] <= 6)
].copy()



## === cell 12
ls1 = list(train_dt["pickup_datetime"])
for i in range(len(ls1)):
    ls1[i] = ls1[i][11:-7:]
train_dt["pickuptime"] = ls1

ls1 = list(test_dt["pickup_datetime"])
for i in range(len(ls1)):
    ls1[i] = ls1[i][11:-7:]
test_dt["pickuptime"] = ls1



## === cell 13
train_dt.head()



## === cell 14
ls1 = list(train_dt["pickup_datetime"])
for i in range(len(ls1)):
    ls1[i] = ls1[i][:-4:]
    ls1[i] = pd.Timestamp(ls1[i])
    ls1[i] = ls1[i].weekday()
train_dt["Weekday"] = ls1

ls1 = list(test_dt["pickup_datetime"])
for i in range(len(ls1)):
    ls1[i] = ls1[i][:-4:]
    ls1[i] = pd.Timestamp(ls1[i])
    ls1[i] = ls1[i].weekday()
test_dt["Weekday"] = ls1



## === cell 15
train_dt.head()



## === cell 16
test_dt.head()



## === cell 17
train_dt.drop("pickup_datetime", inplace=True, axis=1)
test_dt.drop("pickup_datetime", inplace=True, axis=1)



## === cell 18
train_dt["Weekday"].replace(
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
test_dt["Weekday"].replace(
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



## === cell 19
train_one_hot = pd.get_dummies(train_dt["Weekday"])
test_one_hot = pd.get_dummies(test_dt["Weekday"])
train_one_hot, test_one_hot = train_one_hot.align(
    test_one_hot, join="outer", axis=1, fill_value=0
)

train_dt = pd.concat([train_dt, train_one_hot], axis=1)
test_dt = pd.concat([test_dt, test_one_hot], axis=1)



## === cell 20
train_dt.drop("Weekday", axis=1, inplace=True)
test_dt.drop("Weekday", axis=1, inplace=True)



## === cell 21
ls1 = list(train_dt["pickuptime"])
for i in range(len(ls1)):
    z = ls1[i].split(":")
    ls1[i] = int(z[0]) * 100 + int(z[1])
train_dt["pickuptime"] = ls1

ls1 = list(test_dt["pickuptime"])
for i in range(len(ls1)):
    z = ls1[i].split(":")
    ls1[i] = int(z[0]) * 100 + int(z[1])
test_dt["pickuptime"] = ls1



## === cell 22
train_dt.head()



## === cell 23
R = 6373.0
lat1 = np.asarray(np.radians(train_dt["pickup_latitude"]))
lon1 = np.asarray(np.radians(train_dt["pickup_longitude"]))
lat2 = np.asarray(np.radians(train_dt["dropoff_latitude"]))
lon2 = np.asarray(np.radians(train_dt["dropoff_longitude"]))

dlon = lon2 - lon1
dlat = lat2 - lat1
ls1 = []
a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
distance = R * c
train_dt["Distance"] = np.asarray(distance) * 0.621

lat1 = np.asarray(np.radians(test_dt["pickup_latitude"]))
lon1 = np.asarray(np.radians(test_dt["pickup_longitude"]))
lat2 = np.asarray(np.radians(test_dt["dropoff_latitude"]))
lon2 = np.asarray(np.radians(test_dt["dropoff_longitude"]))

dlon = lon2 - lon1
dlat = lat2 - lat1
a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
distance = R * c
test_dt["Distance"] = np.asarray(distance) * 0.621



## === cell 24
R = 6373.0
lat1 = np.asarray(np.radians(train_dt["pickup_latitude"]))
lon1 = np.asarray(np.radians(train_dt["pickup_longitude"]))
lat2 = np.asarray(np.radians(train_dt["dropoff_latitude"]))
lon2 = np.asarray(np.radians(train_dt["dropoff_longitude"]))

lat3 = np.zeros(len(train_dt)) + np.radians(40.6413111)
lon3 = np.zeros(len(train_dt)) + np.radians(-73.7781391)
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
train_dt["Pickup_Distance_airport"] = np.asarray(distance1) * 0.621

a2 = (
    np.sin(d_lat_dropoff / 2) ** 2
    + np.cos(lat2) * np.cos(lat3) * np.sin(d_lon_dropoff / 2) ** 2
)
c2 = 2 * np.arctan2(np.sqrt(a2), np.sqrt(1 - a2))
distance2 = R * c2
train_dt["Dropoff_Distance_airport"] = np.asarray(distance2) * 0.621

lat1 = np.asarray(np.radians(test_dt["pickup_latitude"]))
lon1 = np.asarray(np.radians(test_dt["pickup_longitude"]))
lat2 = np.asarray(np.radians(test_dt["dropoff_latitude"]))
lon2 = np.asarray(np.radians(test_dt["dropoff_longitude"]))

lat3 = np.zeros(len(test_dt)) + np.radians(40.6413111)
lon3 = np.zeros(len(test_dt)) + np.radians(-73.7781391)
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
test_dt["Pickup_Distance_airport"] = np.asarray(distance1) * 0.621

a2 = (
    np.sin(d_lat_dropoff / 2) ** 2
    + np.cos(lat2) * np.cos(lat3) * np.sin(d_lon_dropoff / 2) ** 2
)
c2 = 2 * np.arctan2(np.sqrt(a2), np.sqrt(1 - a2))
distance2 = R * c2
test_dt["Dropoff_Distance_airport"] = np.asarray(distance2) * 0.621



## === cell 25
train_dt["Distance"] = np.round(train_dt["Distance"], 2)
train_dt["Pickup_Distance_airport"] = np.round(train_dt["Pickup_Distance_airport"], 2)
train_dt["Dropoff_Distance_airport"] = np.round(train_dt["Dropoff_Distance_airport"], 2)
test_dt["Distance"] = np.round(test_dt["Distance"], 2)
test_dt["Pickup_Distance_airport"] = np.round(test_dt["Pickup_Distance_airport"], 2)
test_dt["Dropoff_Distance_airport"] = np.round(test_dt["Dropoff_Distance_airport"], 2)



## === cell 26
train_dt.drop(
    ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"],
    axis=1,
    inplace=True,
)
test_dt.drop(
    ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"],
    axis=1,
    inplace=True,
)



## === cell 27
dl_mean = float(np.mean(train_dt["Difference_longitude"]))
dl_std = float(np.std(train_dt["Difference_longitude"])) + 1e-12
dlat_mean = float(np.mean(train_dt["Difference_latitude"]))
dlat_std = float(np.std(train_dt["Difference_latitude"])) + 1e-12



## === cell 28
train_dt["Difference_longitude"] = np.abs(train_dt["Difference_longitude"] - dl_mean)
train_dt["Difference_longitude"] = train_dt["Difference_longitude"] / dl_std



## === cell 29
train_dt["Difference_latitude"] = np.abs(train_dt["Difference_latitude"] - dlat_mean)
train_dt["Difference_latitude"] = train_dt["Difference_latitude"] / dlat_std



## === cell 30
test_dt["Difference_longitude"] = np.abs(test_dt["Difference_longitude"] - dl_mean)
test_dt["Difference_longitude"] = test_dt["Difference_longitude"] / dl_std

test_dt["Difference_latitude"] = np.abs(test_dt["Difference_latitude"] - dlat_mean)
test_dt["Difference_latitude"] = test_dt["Difference_latitude"] / dlat_std



## === cell 31
train_dt.shape



## === cell 32
test_dt.shape



## === cell 33
from sklearn.model_selection import train_test_split

X = train_dt.drop(["key", "fare_amount"], axis=1)
y = train_dt["fare_amount"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.01, random_state=80
)



## === cell 34
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train, y_train)
print(lr.score(X_test, y_test))



## === cell 35
pred = np.round(lr.predict(test_dt.drop("key", axis=1)), 2)



## === cell 36
pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/sample_submission.csv"
).head()



## === cell 37
Submission = pd.DataFrame(data=pred, columns=["fare_amount"])
Submission["key"] = test_dt["key"]
Submission = Submission[["key", "fare_amount"]]



## === cell 38
Submission.set_index("key", inplace=True)



## === cell 39
Submission.to_csv("Submission.csv")
print("Wrote Submission.csv with shape:", Submission.shape)
