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

3.11

# 3. Installed packages

geopandas==0.14.4
geopy==2.4.1
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

3.9197

# 6. Current score

5.2985

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 5.18249) has done: 'I add lightweight datetime features (hour, weekday, month) before dropping the original columns, switch the target to a log‑scale during training (and back‑transform predictions), and boost the RandomForest (more trees, parallelism). These changes keep the overall pipeline but should lower the RMSE toward the target.'
- What this solution (achieved 5.2985) has done: 'I load a larger training sample (500 k rows) to give the model more data, increase the RandomForest to 600 trees for better fit, and replace the slow Python‑loop distance computation with a fast vectorised haversine implementation (same semantic feature). These minimal changes keep the overall pipeline intact while improving the validation RMSE, moving it closer to the target score.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)


import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split



## === cell 2
train_df = pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv", nrows=500000
)
test_df = pd.read_csv("/kaggle/input/new-york-city-taxi-fare-prediction/test.csv")



## === cell 3
train_df



## === cell 4
test_df



## === cell 5
train_df.columns



## === cell 6
test_df.columns



## === cell 7
train_df.isnull().sum().sort_values(ascending=False)



## === cell 8
test_df.isnull().sum().sort_values(ascending=False)



## === cell 9
a = train_df[
    train_df["dropoff_longitude"].isnull() | train_df["dropoff_latitude"].isnull()
]
print(a)



## === cell 10
train_df.drop(a.index, axis=0, inplace=True)



## === cell 11
train_df.isnull().sum().sort_values(ascending=False)



## === cell 12
train_df.describe()



## === cell 13
sns.histplot(data=train_df, x="fare_amount", kde=True, bins=100)



## === cell 14
sns.boxplot(data=train_df, y="fare_amount")



## === cell 15
train_df = train_df[(train_df["fare_amount"] >= 2.5) & (train_df["fare_amount"] < 75)]



## === cell 16
train_df["fare_amount"].describe()



## === cell 17
sns.histplot(data=train_df, x="fare_amount", kde=True, bins=100)
plt.xlim(0, 80)
plt.ylim(0, 150000)



## === cell 18
sns.boxplot(data=train_df, y="fare_amount")



## === cell 19
train_df["passenger_count"].describe()



## === cell 20
train_df = train_df[train_df["passenger_count"] <= 6]
sns.histplot(data=train_df, x="passenger_count")
plt.ylim(0, 1000000)



## === cell 21
train_df = train_df[
    (train_df["pickup_longitude"] <= -73.0) & (train_df["pickup_longitude"] >= -74.5)
]
train_df = train_df[
    (train_df["pickup_latitude"] >= 40.5) & (train_df["pickup_latitude"] <= 42)
]

train_df = train_df[
    (train_df["dropoff_longitude"] <= -73.0) & (train_df["dropoff_longitude"] >= -74.5)
]
train_df = train_df[
    (train_df["dropoff_latitude"] >= 40.5) & (train_df["dropoff_latitude"] <= 42)
]



## === cell 22
train_df.describe()



## === cell 23
train_df



## === cell 24
train_df["pickup_datetime"] = pd.to_datetime(train_df["pickup_datetime"])
train_df["pickup_hour"] = train_df["pickup_datetime"].dt.hour
train_df["pickup_weekday"] = train_df["pickup_datetime"].dt.weekday
train_df["pickup_month"] = train_df["pickup_datetime"].dt.month

test_df["pickup_datetime"] = pd.to_datetime(test_df["pickup_datetime"])
test_df["pickup_hour"] = test_df["pickup_datetime"].dt.hour
test_df["pickup_weekday"] = test_df["pickup_datetime"].dt.weekday
test_df["pickup_month"] = test_df["pickup_datetime"].dt.month

train_df.drop(["key", "pickup_datetime"], axis=1, inplace=True)
test_df.drop(["key", "pickup_datetime"], axis=1, inplace=True)



## === cell 25
train_df.columns



## === cell 26
train_len = len(train_df)
print(train_len)



## === cell 27
test_df.columns



## === cell 28
test_df.describe()



## === cell 29
df = pd.concat([train_df, test_df], axis=0)



## === cell 30
from geopy.distance import (
    geodesic,
)  # retained for compatibility; not used in new implementation


def haversine_np(lat1, lon1, lat2, lon2):
    """
    Vectorised haversine distance (km) between two points.
    """
    R = 6371.0  # Earth radius in km
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c




## === cell 31
print(len(df))



## === cell 32
distances_km = haversine_np(
    df["pickup_latitude"].values,
    df["pickup_longitude"].values,
    df["dropoff_latitude"].values,
    df["dropoff_longitude"].values,
)



## === cell 33
print(distances_km[:20])



## === cell 34
import statistics

mean = statistics.mean(distances_km)
print(mean)



## === cell 35
print(len(distances_km))



## === cell 36
df["distance"] = distances_km



## === cell 37
df



## === cell 38
train = df[0:train_len]
test = df[train_len:]



## === cell 39
train



## === cell 40
train.isnull().sum().sort_values(ascending=False)



## === cell 41
test



## === cell 42
test.isnull().sum().sort_values(ascending=False)



## === cell 43
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split



## === cell 44
train_X = train.drop("fare_amount", axis=1).values
train_y = np.log1p(train["fare_amount"].values)  # log‑scale target



## === cell 45
test_X = test.drop("fare_amount", axis=1).values
train_x, valid_x, train_y, valid_y = train_test_split(
    train_X, train_y, test_size=0.3, random_state=0
)



## === cell 46
model = RandomForestRegressor(n_estimators=600, random_state=0, n_jobs=-1)
model.fit(train_x, train_y)
valid_pred_log = model.predict(valid_x)
valid_pred = np.expm1(valid_pred_log)  # back‑transform
valid_y_original = np.expm1(valid_y)
print("RMSE on validation:", np.sqrt(mean_squared_error(valid_y_original, valid_pred)))



## === cell 47
sub = pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/sample_submission.csv"
)
sub["fare_amount"] = np.expm1(model.predict(test_X))
sub.to_csv("submission.csv", index=False)
