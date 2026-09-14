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

4.20479

# 6. Current score

5.49296

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 6.05507) has done: 'You’re currently losing a lot of score because the distance feature is computed with `geopy.geodesic` inside a Python loop, which is slow and also less consistent than a standard fast haversine computation; replacing it with a vectorized haversine distance keeps the same core feature idea (“distance between pickup/dropoff”) but yields a better-behaved signal for the RandomForest and typically improves RMSE toward your target. I also stop concatenating train/test just to compute distance (we can compute it separately with identical semantics), and I ensure we never accidentally include NaNs/infs in the model inputs. Finally, I keep your model/training approach intact (RandomForestRegressor with the same split logic) and still write `submission.csv` with the required columns.'
- What this solution (achieved 5.49296) has done: 'Your current gap is large (6.05507 vs target 4.20479; lower is better), so we should make a small but meaningful improvement without changing the overall approach (RandomForest on engineered distance + raw coords/passenger_count). The biggest low-risk gain is to add a couple of standard NYC Taxi baseline features derived from `pickup_datetime` (hour/day-of-week) and to use a log1p transform on the target during training with an expm1 inverse at prediction time; this preserves the same model family and training loop while typically reducing RMSE substantially on this competition. I also keep your existing cleaning rules, keep the haversine distance, and ensure the same submission schema (`key,fare_amount`) with row alignment to `sample_submission.csv`. These changes are directly aimed at improving RMSE toward your target while staying within Kaggle constraints and finishing fast on 10k rows.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd

import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
import matplotlib.pyplot as plt
import seaborn as sns



## === cell 2
train_df = pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv", nrows=10000
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
test_df.describe()



## === cell 14
sns.histplot(data=train_df, x="fare_amount", kde=True, bins=100)



## === cell 15
sns.boxplot(data=train_df, y="fare_amount")



## === cell 16
train_df["passenger_count"].describe()



## === cell 17
sns.histplot(data=train_df, x="passenger_count")
plt.ylim(0, 1000)



## === cell 18
train_df = train_df[(train_df["fare_amount"] >= 2.5) & (train_df["fare_amount"] < 100)]



## === cell 19
train_df["fare_amount"].describe()



## === cell 20
sns.histplot(data=train_df, x="fare_amount", kde=True, bins=100)
plt.xlim(0, 80)
plt.ylim(0, 150000)



## === cell 21
sns.boxplot(data=train_df, y="fare_amount")



## === cell 22
train_df = train_df[
    (train_df["passenger_count"] <= 6) & (train_df["passenger_count"] >= 1)
]
sns.histplot(data=train_df, x="passenger_count")
plt.ylim(0, 1000000)



## === cell 23
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



## === cell 24
train_df.describe()



## === cell 25
train_df



## === cell 26
for df in (train_df, test_df):
    df["pickup_datetime"] = pd.to_datetime(
        df["pickup_datetime"], errors="coerce", utc=True
    )
    df["pickup_hour"] = df["pickup_datetime"].dt.hour.astype("float32")
    df["pickup_dow"] = df["pickup_datetime"].dt.dayofweek.astype("float32")
    df["pickup_month"] = df["pickup_datetime"].dt.month.astype("float32")

test_key = test_df["key"].copy()  # needed for submission alignment
train_df.drop(["key", "pickup_datetime"], axis=1, inplace=True)
test_df.drop(["key", "pickup_datetime"], axis=1, inplace=True)



## === cell 27
train_df.columns



## === cell 28
test_df.columns




## === cell 29
def haversine_km(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon):
    R = 6371.0  # Earth radius in km
    lat1 = np.radians(pickup_lat.astype(float))
    lon1 = np.radians(pickup_lon.astype(float))
    lat2 = np.radians(dropoff_lat.astype(float))
    lon2 = np.radians(dropoff_lon.astype(float))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * (
        np.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * np.arcsin(np.sqrt(a))
    return R * c




## === cell 30
train_len = len(train_df)
print(train_len)



## === cell 31
train_df["distance"] = haversine_km(
    train_df["pickup_latitude"].values,
    train_df["pickup_longitude"].values,
    train_df["dropoff_latitude"].values,
    train_df["dropoff_longitude"].values,
)
test_df["distance"] = haversine_km(
    test_df["pickup_latitude"].values,
    test_df["pickup_longitude"].values,
    test_df["dropoff_latitude"].values,
    test_df["dropoff_longitude"].values,
)



## === cell 32
train_df.head()



## === cell 33
test_df.head()



## === cell 34
train_df = train_df.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
test_df = test_df.replace([np.inf, -np.inf], np.nan).fillna(
    test_df.median(numeric_only=True)
)



## === cell 35
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split



## === cell 36
train_X = train_df.drop("fare_amount", axis=1).values
train_y = train_df["fare_amount"].values
test_X = test_df.values

train_x, test_x, train_y_split, test_y_split = train_test_split(
    train_X, train_y, test_size=0.3, random_state=0
)



## === cell 37
model = RandomForestRegressor(random_state=0, n_jobs=-1)

train_y_split_log = np.log1p(train_y_split)
model.fit(train_x, train_y_split_log)

test_x_predict_log = model.predict(test_x)
test_x_predict = np.expm1(test_x_predict_log)

print("score:" + str(np.sqrt(mean_squared_error(test_y_split, test_x_predict))))



## === cell 38
sub = pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/sample_submission.csv"
)

test_pred_log = model.predict(test_X)
test_pred = np.expm1(test_pred_log)

test_pred = np.clip(test_pred, 0.0, None)

pred_by_key = pd.DataFrame({"key": test_key.values, "fare_amount": test_pred})
sub = sub.drop(columns=["fare_amount"]).merge(pred_by_key, on="key", how="left")



## === cell 39
sub.to_csv("submission.csv", index=False)
print(sub.head())
print("Wrote submission.csv with rows:", len(sub))
