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

3.41764

# 6. Current score

5.36121

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 7.72263) has done: 'I add two informative features—`passenger_count` and `minute`—to the model input and increase the number of trees in the RandomForest from 20 to 100 to improve predictive power while keeping the same overall pipeline. These changes should lower the RMSE toward the target score.'
- What this solution (achieved 5.36121) has done: 'I fix the NaN‑related crashes by removing rows with non‑positive fares (which make `log1p` produce NaNs) and by filling missing “busyness” values before model training. These minimal fixes let the pipeline run end‑to‑end and generate a proper `submission.csv` without altering the core modeling logic.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns



## === cell 1
train_df = pd.read_csv("../input/train.csv", nrows=1_000_000)




## === cell 2
def haversine_np(lon1, lat1, lon2, lat2):
    """
    https://stackoverflow.com/questions/29545704/fast-haversine-approximation-python-pandas
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)

    All args must be of equal length.

    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km




## === cell 3
train_df["distance"] = haversine_np(
    train_df["pickup_longitude"],
    train_df["pickup_latitude"],
    train_df["dropoff_longitude"],
    train_df["dropoff_latitude"],
)



## === cell 4
train_df["pickup_datetime"] = pd.to_datetime(train_df["pickup_datetime"])



## === cell 5
train_df["year"] = train_df["pickup_datetime"].dt.year
train_df["month"] = train_df["pickup_datetime"].dt.month
train_df["day"] = train_df["pickup_datetime"].dt.day
train_df["hour"] = train_df["pickup_datetime"].dt.hour
train_df["minute"] = train_df["pickup_datetime"].dt.minute



## === cell 6
print("Old size: %d" % len(train_df))
train_df = train_df.dropna(how="any", axis="rows")
print("New size: %d" % len(train_df))



## === cell 7
cond = True
for col in {
    "pickup_latitude",
    "pickup_longitude",
    "dropoff_latitude",
    "dropoff_longitude",
}:
    cond &= abs(train_df[col] - train_df[col].mean()) < 5
cond &= train_df["fare_amount"] > 0

print("Old size: %d" % len(train_df))
train_df = train_df[cond]
print("New size: %d" % len(train_df))



## === cell 8
for col in {
    "pickup_latitude",
    "pickup_longitude",
    "dropoff_latitude",
    "dropoff_longitude",
}:
    train_df["rough" + col] = train_df[col].round(2)



## === cell 9
a = train_df.groupby(["roughpickup_latitude", "roughpickup_longitude"])[
    ["pickup_latitude", "pickup_longitude"]
].agg(["mean", "count"])



## === cell 10
a.columns = ["mean_pickup_latitude", "c1", "mean_pickup_longitude", "c2"]
a = a[["c1"]].reset_index().rename(columns={"c1": "pickup_busyness"})



## === cell 11
b = train_df.groupby(["roughdropoff_latitude", "roughdropoff_longitude"])[
    ["dropoff_latitude", "dropoff_longitude"]
].agg(["mean", "count"])



## === cell 12
b.columns = ["mean_dropoff_latitude", "c1", "mean_dropoff_longitude", "c2"]
b = b[["c1"]].reset_index().rename(columns={"c1": "dropoff_busyness"})



## === cell 13
train_df = pd.merge(train_df, a, how="left")
train_df = pd.merge(train_df, b, how="left")
train_df["pickup_busyness"] = train_df["pickup_busyness"].fillna(1)
train_df["dropoff_busyness"] = train_df["dropoff_busyness"].fillna(1)



## === cell 14
X = train_df[
    [
        "distance",
        "year",
        "month",
        "day",
        "hour",
        "minute",
        "passenger_count",
        "pickup_busyness",
        "dropoff_busyness",
    ]
].values
Y = train_df["fare_amount"].values



## === cell 15
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

y_train_log = np.log1p(y_train)
y_val_log = np.log1p(y_val)

rf = RandomForestRegressor(
    n_estimators=300,
    max_features=3,
    min_samples_leaf=1,
    min_samples_split=2,
    bootstrap=True,
    max_depth=None,
    n_jobs=-1,
    random_state=42,
)

rf.fit(X_train, y_train_log)

val_pred_log = rf.predict(X_val)
val_pred = np.expm1(val_pred_log)
val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
print(f"Validation RMSE (log‑target): {val_rmse:.5f}")



## === cell 16
Y_log_full = np.log1p(Y)
rf_full = RandomForestRegressor(
    n_estimators=300,
    max_features=3,
    min_samples_leaf=1,
    min_samples_split=2,
    bootstrap=True,
    max_depth=None,
    n_jobs=-1,
    random_state=42,
)
rf_full.fit(X, Y_log_full)



## === cell 17
test_df = pd.read_csv("../input/test.csv")



## === cell 18
test_df["distance"] = haversine_np(
    test_df["pickup_longitude"],
    test_df["pickup_latitude"],
    test_df["dropoff_longitude"],
    test_df["dropoff_latitude"],
)



## === cell 19
test_df["pickup_datetime"] = pd.to_datetime(test_df["pickup_datetime"])



## === cell 20
test_df["year"] = test_df["pickup_datetime"].dt.year
test_df["month"] = test_df["pickup_datetime"].dt.month
test_df["day"] = test_df["pickup_datetime"].dt.day
test_df["hour"] = test_df["pickup_datetime"].dt.hour
test_df["minute"] = test_df["pickup_datetime"].dt.minute



## === cell 21
for col in {
    "pickup_latitude",
    "pickup_longitude",
    "dropoff_latitude",
    "dropoff_longitude",
}:
    test_df["rough" + col] = test_df[col].round(2)



## === cell 22
test_df = pd.merge(test_df, a, how="left")
test_df = pd.merge(test_df, b, how="left")
test_df["pickup_busyness"] = test_df["pickup_busyness"].fillna(1)
test_df["dropoff_busyness"] = test_df["dropoff_busyness"].fillna(1)



## === cell 23
X_test = test_df[
    [
        "distance",
        "year",
        "month",
        "day",
        "hour",
        "minute",
        "passenger_count",
        "pickup_busyness",
        "dropoff_busyness",
    ]
].values



## === cell 24
test_pred_log = rf_full.predict(X_test)
test_pred = np.expm1(test_pred_log)



## === cell 25
submission = pd.DataFrame(
    {"key": test_df["key"], "fare_amount": test_pred}, columns=["key", "fare_amount"]
)
submission.to_csv("submission.csv", index=False)
