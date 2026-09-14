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

3.12

# 3. Installed packages

No external packages required in the script and installed.

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

5.67556

# 6. Current score

6.31658

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

- What this solution (achieved 6.31658) has done: 'I fixed the import errors (geopy, catboost, lightgbm) and removed the unused plotting cells that caused crashes. I kept the original feature engineering, added a more robust GradientBoostingRegressor (which typically lowers RMSE toward the target), and ensured the script always writes a proper `Submission.csv` with the required columns.'

# 9. Code solution

## === cell 0
import os, warnings
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from datetime import datetime as dt

warnings.filterwarnings("ignore")
for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
df = pd.read_csv("../input/new-york-city-taxi-fare-prediction/train.csv", nrows=25000)
df_test = pd.read_csv("../input/new-york-city-taxi-fare-prediction/test.csv")
df.head()



## === cell 2
df.dropna(axis=0, inplace=True)
df = df[df["fare_amount"] >= 0]  # remove negative fares



## === cell 3
df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
df_test["pickup_datetime"] = pd.to_datetime(df_test["pickup_datetime"])
df["pickup_hour"] = df["pickup_datetime"].dt.hour
df["pickup_weekday"] = df["pickup_datetime"].dt.day_name()
df["pickup_date"] = df["pickup_datetime"].dt.day
df["pickup_month"] = df["pickup_datetime"].dt.month
df["pickup_dayofweek"] = df["pickup_datetime"].dt.dayofweek
df_test["pickup_hour"] = df_test["pickup_datetime"].dt.hour
df_test["pickup_weekday"] = df_test["pickup_datetime"].dt.day_name()
df_test["pickup_date"] = df_test["pickup_datetime"].dt.day
df_test["pickup_month"] = df_test["pickup_datetime"].dt.month
df_test["pickup_dayofweek"] = df_test["pickup_datetime"].dt.dayofweek




## === cell 4
def baseFare(hour):
    if hour in range(16, 20):
        return 3.50
    elif hour in range(20, 24):
        return 3.0
    else:
        return 2.50


df["base_fare"] = df["pickup_hour"].apply(baseFare)
df_test["base_fare"] = df_test["pickup_hour"].apply(baseFare)




## === cell 5
def haversine_distance(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    km = 6371 * 2 * np.arcsin(np.sqrt(a))
    return km


df["haversine_distance"] = haversine_distance(
    df["pickup_latitude"].values,
    df["pickup_longitude"].values,
    df["dropoff_latitude"].values,
    df["dropoff_longitude"].values,
)
df_test["haversine_distance"] = haversine_distance(
    df_test["pickup_latitude"].values,
    df_test["pickup_longitude"].values,
    df_test["dropoff_latitude"].values,
    df_test["dropoff_longitude"].values,
)



## === cell 6
df = df[(df["haversine_distance"] > 0) & (df["haversine_distance"] < 8)]



## === cell 7
features_to_drop = [
    "key",
    "pickup_datetime",
    "pickup_weekday",
    "fare_amount",
    "fare",
    "base_fare",
    "haversine_distance",
]
X = df.drop(columns=features_to_drop, errors="ignore")
y = df["fare_amount"]
X_test_final = df_test.drop(
    columns=[
        "key",
        "pickup_datetime",
        "pickup_weekday",
        "base_fare",
        "haversine_distance",
    ],
    errors="ignore",
)



## === cell 8
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

scaler = preprocessing.StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test_final)

X_train, X_valid, y_train, y_valid = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)



## === cell 9
from sklearn.ensemble import GradientBoostingRegressor

gbr = GradientBoostingRegressor(
    n_estimators=250, learning_rate=0.05, max_depth=5, random_state=42
)
gbr.fit(X_train, y_train)



## === cell 10
from sklearn.metrics import mean_squared_error

valid_pred = gbr.predict(X_valid)
rmse = mean_squared_error(y_valid, valid_pred, squared=False)
print(f"Validation RMSE: {rmse:.5f}")



## === cell 11
test_pred = gbr.predict(X_test_scaled)
submission = pd.DataFrame({"key": df_test["key"], "fare_amount": test_pred})
submission.head()



## === cell 12
submission_path = "Submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
