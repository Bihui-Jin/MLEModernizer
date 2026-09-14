# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O
import math
import os

print("Listing ../input:")
try:
    print(os.listdir("../input"))
except FileNotFoundError:
    print("WARNING: ../input not found. Available root:", os.listdir("/"))



## === cell 1
train = pd.read_csv("../input/train.csv", nrows=1_000_000)
test = pd.read_csv("../input/test.csv")
samp = pd.read_csv("../input/sample_submission.csv")

print(
    "train shape:", train.shape, "test shape:", test.shape, "sample shape:", samp.shape
)



## === cell 2
train = train.dropna(how="any", axis="rows").reset_index(drop=True)
print("train shape after dropna:", train.shape)



## === cell 3
test.shape




## === cell 4
def clean_train_rows(df):
    df = df.copy()

    df = df[(df["fare_amount"] > 0) & (df["fare_amount"] <= 250)]

    df = df[(df["passenger_count"] >= 1) & (df["passenger_count"] <= 6)]

    df = df[
        (df["pickup_longitude"].between(-74.5, -72.8))
        & (df["dropoff_longitude"].between(-74.5, -72.8))
        & (df["pickup_latitude"].between(40.5, 41.8))
        & (df["dropoff_latitude"].between(40.5, 41.8))
    ]

    same_loc = (df["pickup_longitude"] == df["dropoff_longitude"]) & (
        df["pickup_latitude"] == df["dropoff_latitude"]
    )
    df = df[~same_loc]

    return df.reset_index(drop=True)


train = clean_train_rows(train)
print("train shape after clean_train_rows:", train.shape)



## === cell 5
all_data = pd.concat((train, test), axis=0, ignore_index=True)

y = train.fare_amount.values
n_train = len(train)
n_test = len(test)
test_id = test.key

all_data = all_data.drop(["fare_amount"], axis=1)

print("all_data shape:", all_data.shape, "n_train:", n_train, "n_test:", n_test)




## === cell 6
def week_num(day):
    """
    given the day of the month, return the week number of the month
    """
    if day <= 7:
        return "first"
    if (day > 7) and (day <= 14):
        return "second"
    if (day > 14) and (day <= 21):
        return "third"
    if (day > 21) and (day <= 28):
        return "fourth"
    return "fifth"




## === cell 7
def add_time_features(data):
    data = data.copy()
    data["pickup_datetime"] = pd.to_datetime(data["pickup_datetime"], errors="coerce")

    data["hour"] = data["pickup_datetime"].dt.hour
    data["day_of_week"] = data["pickup_datetime"].dt.day_name()
    data["day_of_month"] = data["pickup_datetime"].dt.day
    data["week_of_month"] = data["day_of_month"].map(week_num)
    data["month"] = data["pickup_datetime"].dt.month
    data["year"] = data["pickup_datetime"].dt.year

    data["hour"] = data["hour"].astype("Int64").astype(str)
    data["month"] = data["month"].astype("Int64").astype(str)
    data["year"] = data["year"].astype("Int64").astype(str)

    data = data.drop(["day_of_month"], axis=1)
    return data




## === cell 8
def add_geo_features(data):
    data = data.copy()
    data["abs_diff_longitude"] = (
        data["dropoff_longitude"] - data["pickup_longitude"]
    ).abs()
    data["abs_diff_latitude"] = (
        data["dropoff_latitude"] - data["pickup_latitude"]
    ).abs()

    data["manhattan_distance"] = data["abs_diff_longitude"] + data["abs_diff_latitude"]

    data["squared_long"] = np.power(data["abs_diff_longitude"], 2)
    data["squared_lat"] = np.power(data["abs_diff_latitude"], 2)

    data["euclid_disance"] = np.sqrt(data["squared_long"] + data["squared_lat"])
    return data




## === cell 9
all_data = add_time_features(all_data)
all_data = add_geo_features(all_data)

drop_cols = []
for c in ["pickup_datetime", "key"]:
    if c in all_data.columns:
        drop_cols.append(c)
if drop_cols:
    all_data = all_data.drop(columns=drop_cols)

print("columns after feature eng:", all_data.columns.tolist())



## === cell 10
features = [
    "passenger_count",
    "hour",
    "day_of_week",
    "week_of_month",
    "month",
    "year",
    "abs_diff_longitude",
    "abs_diff_latitude",
    "euclid_disance",
    "manhattan_distance",
]

missing = [c for c in features if c not in all_data.columns]
if missing:
    raise KeyError(f"Missing engineered feature columns: {missing}")

all_data = all_data[features]

all_data = pd.get_dummies(all_data)

print("all_data dummies shape:", all_data.shape)



## === cell 11
x = all_data.iloc[:n_train].copy()
x_test = all_data.iloc[n_train:].copy()

x, x_test = x.align(x_test, join="left", axis=1, fill_value=0)

x = x.astype(np.float32)
x_test = x_test.astype(np.float32)

print("x shape:", x.shape, "x_test shape:", x_test.shape, "y shape:", y.shape)



## === cell 12
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(random_state=42, n_jobs=-1)



## === cell 13
params = {
    "bootstrap": [True],
    "max_depth": [80, 90, 100, 110],
    "max_features": [2, 3],
    "min_samples_leaf": [3, 4, 5],
    "min_samples_split": [8, 10, 12],
    "n_estimators": [100, 200, 300, 1000],
}



## === cell 14
from sklearn.model_selection import RandomizedSearchCV

search = RandomizedSearchCV(
    estimator=model,
    param_distributions=params,
    n_iter=12,  # bounded to keep runtime < 600s while still improving over defaults
    scoring="neg_root_mean_squared_error",  # aligned with Kaggle metric
    cv=3,
    random_state=42,
    n_jobs=-1,
    verbose=1,
    refit=True,
)

search.fit(x, y)

print("Best CV RMSE:", -float(search.best_score_))
print("Best params:", search.best_params_)

best_model = search.best_estimator_



## === cell 15
test_pred = best_model.predict(x_test)

test_pred = np.maximum(test_pred, 0)

print(
    "Pred stats:",
    float(np.min(test_pred)),
    float(np.mean(test_pred)),
    float(np.max(test_pred)),
)



## === cell 16
sub = pd.DataFrame({"key": test_id.values, "fare_amount": test_pred})
assert sub.shape[0] == test.shape[0], "Submission rows do not match test rows"
sub.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())
