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
xgboost==2.0.3

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

8.30512

# 6. Current score

None

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

- What this solution (achieved 15.83674) has done: 'I fix the datetime feature extraction to work with the current pandas version by replacing deprecated `.dt.week`/`.dt.weekofyear` with ISO calendar week, and also correct a bug where `pickup_datetime` was mistakenly taken from `test` for both train and test. I update `DataFrame.drop(..., 1, ...)` to the modern keyword form and ensure only numeric columns are passed into XGBoost (and that `key` is not included as a feature). I also make the XGBoost objective compatible with xgboost 2.x (`reg:squarederror`) while keeping the same training loop semantics, and ensure a valid `submission.csv` is written. Finally, I make the correlation heatmap cell robust by computing correlations on numeric columns only (so it won’t crash if you run it).'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import datetime as dt

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
import xgboost as xgb

train = pd.read_csv("/kaggle/input/train.csv", nrows=10_000_000)
test = pd.read_csv("/kaggle/input/test.csv")

test.dtypes



## === cell 1
print(test.head())



## === cell 2
print(train.head())




## === cell 3
def add_features(dataset: pd.DataFrame) -> pd.DataFrame:
    dataset = dataset.copy()

    dataset["longitude_distance"] = (
        dataset["pickup_longitude"] - dataset["dropoff_longitude"]
    ).abs()
    dataset["latitude_distance"] = (
        dataset["pickup_latitude"] - dataset["dropoff_latitude"]
    ).abs()

    dataset["distance_travelled"] = np.sqrt(
        dataset["longitude_distance"] ** 2 + dataset["latitude_distance"] ** 2
    )

    dataset["distance_travelled_sin"] = np.sin(dataset["distance_travelled"])
    dataset["distance_travelled_cos"] = np.cos(dataset["distance_travelled"])
    dataset["distance_travelled_sin_sqrd"] = np.sin(dataset["distance_travelled"]) ** 2
    dataset["distance_travelled_cos_sqrd"] = np.cos(dataset["distance_travelled"]) ** 2

    R = 6371e3  # Metres
    phi1 = np.radians(dataset["pickup_latitude"])
    phi2 = np.radians(dataset["dropoff_latitude"])
    phi_chg = np.radians(dataset["pickup_latitude"] - dataset["dropoff_latitude"])
    delta_chg = np.radians(dataset["pickup_longitude"] - dataset["dropoff_longitude"])

    a = (np.sin(phi_chg / 2) ** 2) + np.cos(phi1) * np.cos(phi2) * (
        np.sin(delta_chg / 2) ** 2
    )
    c = 2 * np.arctan2(a**0.5, (1 - a) ** 0.5)
    d = R * c
    dataset["haversine"] = d

    y = np.sin(delta_chg * np.cos(phi2))
    x = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(delta_chg)
    dataset["bearing"] = np.degrees(np.arctan2(y, x))

    psi_chg = np.log(np.tan(np.pi / 4 + phi2 / 2) / np.tan(np.pi / 4 + phi1 / 2))
    q = phi_chg / psi_chg
    d = (phi_chg + q**2 * delta_chg**2) ** 0.5 * R
    dataset["rhumb_lines"] = d

    dataset["pickup_datetime"] = pd.to_datetime(
        dataset["pickup_datetime"], utc=True, errors="coerce"
    ).dt.tz_convert(None)

    dataset["hour_of_day"] = dataset["pickup_datetime"].dt.hour
    dataset["day"] = dataset["pickup_datetime"].dt.day
    dataset["month"] = dataset["pickup_datetime"].dt.month
    dataset["day_of_year"] = dataset["pickup_datetime"].dt.dayofyear

    iso_week = dataset["pickup_datetime"].dt.isocalendar().week.astype("int16")
    dataset["week"] = iso_week
    dataset["week_of_year"] = iso_week

    return dataset


train = add_features(train)
test = add_features(test)

train.head(3)



## === cell 4
colormap = plt.cm.RdBu
plt.figure(figsize=(20, 20))
plt.title("Pearson Correlation of Features", y=1.05, size=15)
numeric_corr = train.select_dtypes(include=[np.number]).corr()
sns.heatmap(
    numeric_corr,
    linewidths=0.1,
    vmax=1.0,
    square=True,
    cmap=colormap,
    linecolor="white",
    annot=False,
)



## === cell 5
feature_cols = [
    "haversine",
    "longitude_distance",
    "latitude_distance",
    "distance_travelled",
    "bearing",
    "rhumb_lines",
    "hour_of_day",
    "day",
    "month",
    "day_of_year",
    "week",
    "passenger_count",
]

train = train.dropna(subset=feature_cols + ["fare_amount"]).copy()
train = train[(train["fare_amount"] > 0) & (train["fare_amount"] < 500)].copy()
train = train[(train["passenger_count"] >= 1) & (train["passenger_count"] <= 6)].copy()

train = train[feature_cols + ["fare_amount"]].copy()

test = test.dropna(subset=feature_cols + ["key"]).copy()
test = test[["key"] + feature_cols].copy()



## === cell 6
x_pred = test.drop("key", axis=1)

X_full = train.drop("fare_amount", axis=1)
y_full = train["fare_amount"]


def XGBmodel_full(X, y):
    matrix_train = xgb.DMatrix(X, label=y)
    model = xgb.train(
        params={
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "eta": 0.1,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "seed": 123,
        },
        dtrain=matrix_train,
        num_boost_round=300,
        evals=[],
        verbose_eval=False,
    )
    return model


model = XGBmodel_full(X_full, y_full)

dm_pred = xgb.DMatrix(x_pred)
prediction = model.predict(dm_pred)

prediction = np.clip(prediction, 0.0, 500.0)



## === cell 7
submission = pd.DataFrame({"key": test["key"], "fare_amount": np.round(prediction, 2)})
submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
