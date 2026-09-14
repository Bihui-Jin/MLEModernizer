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

3.7

# 3. Installed packages

geopandas==0.14.4
lightgbm==4.6.0
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
scipy==1.15.3
sklearn-pandas==2.2.0
tqdm==4.67.1

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

3.382252995833693

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, random, pickle
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
import lightgbm as lgb
import matplotlib.pyplot as plt

print("working dir:", os.getcwd())
print("input dirs:", os.listdir("../input"))



## === cell 1
train_path = "../input/new-york-city-taxi-fare-prediction/train.csv"
test_path = "../input/new-york-city-taxi-fare-prediction/test.csv"

n = sum(1 for _ in open(train_path)) - 1
sample_size = 2_000_000  # 2 M rows – enough for a decent model yet fast
skip = sorted(random.sample(range(1, n + 1), n - sample_size))

train = pd.read_csv(train_path, skiprows=skip)
test = pd.read_csv(test_path)

test_id = test["key"].values.copy()  # keep for submission




## === cell 2
def preprocess(df):
    df = df.copy()
    df["pickup_datetime"] = pd.to_datetime(
        df["pickup_datetime"].str.slice(0, 16), utc=True, format="%Y-%m-%d %H:%M"
    )
    df["hour"] = df["pickup_datetime"].dt.hour.astype("uint8")
    df["month"] = df["pickup_datetime"].dt.month.astype("uint8")
    df["day_of_week"] = df["pickup_datetime"].dt.dayofweek.astype("uint8")
    df["year"] = df["pickup_datetime"].dt.year.astype("uint16")
    df.drop(columns=["pickup_datetime", "key"], inplace=True, errors="ignore")
    return df


train = preprocess(train)
test = preprocess(test)




## === cell 3
def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6371.01 * c
    return km


train["distance"] = haversine(
    train["pickup_latitude"],
    train["pickup_longitude"],
    train["dropoff_latitude"],
    train["dropoff_longitude"],
).astype("float32")

test["distance"] = haversine(
    test["pickup_latitude"],
    test["pickup_longitude"],
    test["dropoff_latitude"],
    test["dropoff_longitude"],
).astype("float32")



## === cell 4
train = train[train["fare_amount"] > 0].copy()
train.dropna(inplace=True)
test.dropna(inplace=True)

y = train["fare_amount"].values.astype("float32")
X = train.drop(columns=["fare_amount"])



## === cell 5
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.1, random_state=42
)

params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting": "gbdt",
    "learning_rate": 0.3,
    "num_leaves": 64,
    "max_depth": -1,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbosity": -1,
    "seed": 17,
}

d_train = lgb.Dataset(X_train, label=y_train)
d_valid = lgb.Dataset(X_valid, label=y_valid, reference=d_train)

model = lgb.train(
    params,
    d_train,
    num_boost_round=500,
    valid_sets=[d_train, d_valid],
    early_stopping_rounds=50,
    verbose_eval=100,
)

pred_valid = model.predict(X_valid, num_iteration=model.best_iteration)
rmse = sqrt(mean_squared_error(y_valid, pred_valid))
print(f"Validation RMSE: {rmse:.5f}")



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/311637815.py in <cell line: 0>()
     23 d_valid = lgb.Dataset(X_valid, label=y_valid, reference=d_train)
     24 
---> 25 model = lgb.train(
     26     params,
     27     d_train,

TypeError: train() got an unexpected keyword argument 'early_stopping_rounds'

## === cell 6
test_pred = model.predict(test, num_iteration=model.best_iteration)
test_pred = np.where(test_pred > 0, test_pred, 0)  # enforce non‑negative fares

submission = pd.DataFrame({"key": test_id, "fare_amount": test_pred})
submission.to_csv("submission_lgb.csv", index=False)
print("Submission saved to submission_lgb.csv – rows:", len(submission))



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/521966785.py in <cell line: 0>()
      2 # Predict on the test set and write submission file
      3 # ------------------------------------------------------------------
----> 4 test_pred = model.predict(test, num_iteration=model.best_iteration)
      5 test_pred = np.where(test_pred > 0, test_pred, 0)  # enforce non‑negative fares
      6 

NameError: name 'model' is not defined

## === cell 7
plt.figure(figsize=(10, 6))
plt.hist(test_pred, bins=100, color="steelblue", edgecolor="black")
plt.title("Distribution of predicted fare amounts")
plt.xlabel("Fare amount")
plt.ylabel("Count")
plt.show()

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3836677891.py in <cell line: 0>()
      3 # ------------------------------------------------------------------
      4 plt.figure(figsize=(10, 6))
----> 5 plt.hist(test_pred, bins=100, color="steelblue", edgecolor="black")
      6 plt.title("Distribution of predicted fare amounts")
      7 plt.xlabel("Fare amount")

NameError: name 'test_pred' is not defined
