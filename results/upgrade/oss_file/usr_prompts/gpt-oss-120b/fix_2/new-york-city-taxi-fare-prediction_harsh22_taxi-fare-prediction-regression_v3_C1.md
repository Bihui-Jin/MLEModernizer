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

3.37198

# 6. Current score

16.67821

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

- What this solution (achieved 16.67821) has done: 'I fixed the dataframe‑null handling, corrected the latitude/longitude outlier drops, added a drop‑na step before modeling, and introduced a median imputer so the RandomForest can be trained without NaNs. These changes resolve the runtime errors and allow a proper submission CSV to be created, moving the pipeline toward the target RMS‑E score.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import os

print(os.listdir("../input"))



## === cell 1
train = pd.read_csv("../input/train.csv", nrows=1000000)
test = pd.read_csv("../input/test.csv")



## === cell 2
train.shape



## === cell 3
test.shape



## === cell 4
train.head(10)



## === cell 5
train = train.dropna()
train.isnull().sum().sort_values(ascending=False)



## === cell 6
mask = (train["pickup_latitude"] < -90) | (train["pickup_latitude"] > 90)
train = train.loc[~mask]



## === cell 7
mask = (train["pickup_longitude"] < -180) | (train["pickup_longitude"] > 180)
train = train.loc[~mask]



## === cell 8
mask = (train["dropoff_latitude"] < -90) | (train["dropoff_latitude"] > 90)
train = train.loc[~mask]



## === cell 9
train.shape



## === cell 10
train["fare_amount"].describe()



## === cell 11
from collections import Counter

Counter(train["fare_amount"] < 0)



## === cell 12
train = train.drop(train[train["fare_amount"] < 0].index, axis=0)
train.shape



## === cell 13
train["fare_amount"].describe()



## === cell 14
train["passenger_count"].describe()



## === cell 15
train = train.drop(train[train["passenger_count"] == 208].index, axis=0)



## === cell 16
train["passenger_count"].describe()



## === cell 17
train["pickup_latitude"].describe()



## === cell 18
train = train.drop(train[train["pickup_latitude"] < -90].index, axis=0)
train = train.drop(train[train["pickup_latitude"] > 90].index, axis=0)



## === cell 19
train = train.drop(train[train["pickup_longitude"] < -180].index, axis=0)
train = train.drop(train[train["pickup_longitude"] > 180].index, axis=0)



## === cell 20
train = train.drop(train[train["dropoff_latitude"] < -90].index, axis=0)
train = train.drop(train[train["dropoff_latitude"] > 90].index, axis=0)



## === cell 21
train.dtypes



## === cell 22
train["key"] = pd.to_datetime(train["key"], infer_datetime_format=True)
train["pickup_datetime"] = pd.to_datetime(
    train["pickup_datetime"], infer_datetime_format=True
)



## === cell 23
test["key"] = pd.to_datetime(test["key"], infer_datetime_format=True)
test["pickup_datetime"] = pd.to_datetime(
    test["pickup_datetime"], infer_datetime_format=True
)




## === cell 24
def haversine_distance(lat1, long1, lat2, long2):
    r = 6371  # Earth radius in km
    phi1 = np.radians(train[lat1])
    phi2 = np.radians(train[lat2])
    delta_phi = np.radians(train[lat2] - train[lat1])
    delta_lambda = np.radians(train[long2] - train[long1])
    a = (
        np.sin(delta_phi / 2.0) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    train["H_Distance"] = r * c

    phi1_t = np.radians(test[lat1])
    phi2_t = np.radians(test[lat2])
    delta_phi_t = np.radians(test[lat2] - test[lat1])
    delta_lambda_t = np.radians(test[long2] - test[long1])
    a_t = (
        np.sin(delta_phi_t / 2.0) ** 2
        + np.cos(phi1_t) * np.cos(phi2_t) * np.sin(delta_lambda_t / 2.0) ** 2
    )
    c_t = 2 * np.arctan2(np.sqrt(a_t), np.sqrt(1 - a_t))
    test["H_Distance"] = r * c_t

    return train["H_Distance"]




## === cell 25
haversine_distance(
    "pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude"
)



## === cell 26
for df in [train, test]:
    df["Year"] = df["pickup_datetime"].dt.year
    df["Month"] = df["pickup_datetime"].dt.month
    df["Date"] = df["pickup_datetime"].dt.day
    df["Day of Week"] = df["pickup_datetime"].dt.dayofweek
    df["Hour"] = df["pickup_datetime"].dt.hour



## === cell 27
cond = (
    (train["pickup_latitude"] == 0)
    & (train["pickup_longitude"] == 0)
    & (train["dropoff_latitude"] != 0)
    & (train["dropoff_longitude"] != 0)
    & (train["fare_amount"] == 0)
)
train = train.drop(train[cond].index, axis=0)

cond = (
    (train["pickup_latitude"] != 0)
    & (train["pickup_longitude"] != 0)
    & (train["dropoff_latitude"] == 0)
    & (train["dropoff_longitude"] == 0)
    & (train["fare_amount"] == 0)
)
train = train.drop(train[cond].index, axis=0)



## === cell 28
high_distance = train.loc[(train["H_Distance"] > 200) & (train["fare_amount"] != 0)]
high_distance["H_Distance"] = high_distance.apply(
    lambda row: (row["fare_amount"] - 2.50) / 1.56, axis=1
)
train.update(high_distance)



## === cell 29
train = train.drop(
    train[(train["H_Distance"] == 0) & (train["fare_amount"] == 0)].index, axis=0
)



## === cell 30
scenario_3 = train.loc[(train["H_Distance"] != 0) & (train["fare_amount"] == 0)]
scenario_3["fare_amount"] = scenario_3.apply(
    lambda row: (row["H_Distance"] * 1.56) + 2.50, axis=1
)
train.update(scenario_3)

scenario_4 = train.loc[(train["H_Distance"] == 0) & (train["fare_amount"] != 0)]
scenario_4_sub = scenario_4.loc[scenario_4["fare_amount"] > 3.0]
scenario_4_sub["H_Distance"] = scenario_4_sub.apply(
    lambda row: (row["fare_amount"] - 2.50) / 1.56, axis=1
)
train.update(scenario_4_sub)



## === cell 31
train = train.drop(["key", "pickup_datetime"], axis=1)
test = test.drop(["key", "pickup_datetime"], axis=1)



## === cell 32
train = train.dropna()
x_train = train.drop(columns=["fare_amount"])
y_train = train["fare_amount"].values
x_test = test



## === cell 33
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median")
X_train_imp = imputer.fit_transform(x_train)
X_test_imp = imputer.transform(x_test)

rf = RandomForestRegressor(random_state=42, n_estimators=200, max_depth=20, n_jobs=5)
rf.fit(X_train_imp, y_train)
rf_predict = rf.predict(X_test_imp)



## === cell 34
submission = pd.read_csv("../input/sample_submission.csv")
submission["fare_amount"] = rf_predict
submission.to_csv("submission_1.csv", index=False)
submission.head(20)
