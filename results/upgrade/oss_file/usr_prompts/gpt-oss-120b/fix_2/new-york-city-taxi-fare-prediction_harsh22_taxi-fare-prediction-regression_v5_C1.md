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

3.7

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

3.47816

# 6. Current score

4.24509

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

- What this solution (achieved 4.24509) has done: 'I fixed the DataFrame‑null handling, corrected the latitude/longitude outlier drops (using proper boolean masks), added a simple median imputer to remove any remaining NaNs before model training, and ensured the submission file is written with the required “.csv” suffix and correct column names.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os

print(os.listdir("../input"))



## === cell 1
train = pd.read_csv("../input/train.csv", nrows=1000000)
test = pd.read_csv("../input/test.csv")



## === cell 2
train = train.drop(train[train.isnull().any(axis=1)].index, axis=0)



## === cell 3
train = train.drop(train[train["fare_amount"] < 0].index, axis=0)



## === cell 4
train = train.drop(train[train["passenger_count"] == 208].index, axis=0)



## === cell 5
lat_cond = (train["pickup_latitude"] < -90) | (train["pickup_latitude"] > 90)
train = train.drop(train[lat_cond].index, axis=0)



## === cell 6
lon_cond = (train["pickup_longitude"] < -180) | (train["pickup_longitude"] > 180)
train = train.drop(train[lon_cond].index, axis=0)



## === cell 7
drop_lat_cond = (train["dropoff_latitude"] < -90) | (train["dropoff_latitude"] > 90)
train = train.drop(train[drop_lat_cond].index, axis=0)



## === cell 8
drop_lon_cond = (train["dropoff_longitude"] < -180) | (train["dropoff_longitude"] > 180)
train = train.drop(train[drop_lon_cond].index, axis=0)



## === cell 9
train["key"] = pd.to_datetime(train["key"], infer_datetime_format=True)
train["pickup_datetime"] = pd.to_datetime(
    train["pickup_datetime"], infer_datetime_format=True
)
test["key"] = pd.to_datetime(test["key"], infer_datetime_format=True)
test["pickup_datetime"] = pd.to_datetime(
    test["pickup_datetime"], infer_datetime_format=True
)




## === cell 10
def haversine_distance(df, lat1, lon1, lat2, lon2):
    r = 6371.0
    phi1 = np.radians(df[lat1])
    phi2 = np.radians(df[lat2])
    delta_phi = np.radians(df[lat2] - df[lat1])
    delta_lambda = np.radians(df[lon2] - df[lon1])
    a = (
        np.sin(delta_phi / 2.0) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return r * c


train["H_Distance"] = haversine_distance(
    train,
    "pickup_latitude",
    "pickup_longitude",
    "dropoff_latitude",
    "dropoff_longitude",
)
test["H_Distance"] = haversine_distance(
    test, "pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude"
)



## === cell 11
for df in [train, test]:
    df["Year"] = df["pickup_datetime"].dt.year
    df["Month"] = df["pickup_datetime"].dt.month
    df["Day"] = df["pickup_datetime"].dt.day
    df["DayOfWeek"] = df["pickup_datetime"].dt.dayofweek
    df["Hour"] = df["pickup_datetime"].dt.hour



## === cell 12
cond = (train["H_Distance"] == 0) & (train["fare_amount"] < 2.5)
train = train.drop(train[cond].index, axis=0)



## === cell 13
train = train.drop(["key", "pickup_datetime"], axis=1)
test = test.drop(["key", "pickup_datetime"], axis=1)



## === cell 14
x_train = train.drop("fare_amount", axis=1)
y_train = train["fare_amount"].values
x_test = test.copy()



## === cell 15
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median")
x_train = imputer.fit_transform(x_train)
x_test = imputer.transform(x_test)



## === cell 16
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(random_state=42, n_jobs=5)
rf.fit(x_train, y_train)
rf_predict = rf.predict(x_test)



## === cell 17
submission = pd.read_csv("../input/sample_submission.csv")
submission["fare_amount"] = rf_predict
submission.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")
