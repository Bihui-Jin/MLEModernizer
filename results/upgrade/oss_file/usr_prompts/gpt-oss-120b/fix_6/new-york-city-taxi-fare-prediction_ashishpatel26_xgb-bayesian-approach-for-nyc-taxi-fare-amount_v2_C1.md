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
geopy==2.4.1
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

4.22957

# 6. Current score

4.91263

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 8.98306) has done: 'Implemented a safe fix for the distance feature generation by removing the failing geopy‑based row‑wise computation and using the already‑computed haversine column (converted to kilometers). This eliminates the latitude range error, restores the “distance” column needed for modeling, and allows the subsequent pipeline to run without key errors.'
- What this solution (achieved 5.02766) has done: 'I add the richer distance‑based features that were already computed (e.g., `distance_travelled`, its sin/cos and bearing) to the model’s feature list, and I give LightGBM a slightly larger capacity (more leaves and trees). These small, focused changes keep the original pipeline intact while providing the model with more informative inputs, which should lower the RMSE toward the target score.'
- What this solution (achieved 8.9245) has done: 'I streamline the feature set by dropping the noisy degree‑based distance columns and adding a simple “distance per passenger” feature, then slightly increase LightGBM capacity (more leaves and trees, a lower learning rate). These focused tweaks should lower the validation RMSE toward the target while keeping the original pipeline intact.'
- What this solution (achieved 4.91263) has done: 'I add the richer time‑ and distance‑based columns that were already computed (hour_of_day, week, day_of_year, weekday, quarter, distance_travelled and its trigonometric transforms, bearing) to the model’s feature list, and slightly increase LightGBM capacity (more leaves and trees, a lower learning rate, higher feature_fraction). These focused additions keep the original pipeline intact while providing the regressor with more predictive information, which is expected to lower the validation RMSE toward the target.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor

plt.style.use("fivethirtyeight")
print("Input directory contents:", os.listdir("../input"))
import gc




## === cell 1
def load_Data():
    train = pd.read_csv("../input/train.csv", nrows=2_000_000, low_memory=True)
    test = pd.read_csv("../input/test.csv", nrows=2_000_000, low_memory=True)
    return train, test


def prepare_distance_features(df):
    df["longitude_distance"] = np.abs(df["pickup_longitude"] - df["dropoff_longitude"])
    df["latitude_distance"] = np.abs(df["pickup_latitude"] - df["dropoff_latitude"])
    df["distance_travelled"] = np.sqrt(
        df["longitude_distance"] ** 2 + df["latitude_distance"] ** 2
    )
    df["distance_travelled_sin"] = np.sin(df["distance_travelled"])
    df["distance_travelled_cos"] = np.cos(df["distance_travelled"])
    df["distance_travelled_sin_sqrd"] = np.sin(df["distance_travelled"]) ** 2
    df["distance_travelled_cos_sqrd"] = np.cos(df["distance_travelled"]) ** 2

    R = 6371e3  # metres
    phi1 = np.radians(df["pickup_latitude"])
    phi2 = np.radians(df["dropoff_latitude"])
    phi_chg = np.radians(df["pickup_latitude"] - df["dropoff_latitude"])
    delta_chg = np.radians(df["pickup_longitude"] - df["dropoff_longitude"])
    a = np.sin(phi_chg / 2) + np.cos(phi1) * np.cos(phi2) * np.sin(delta_chg / 2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    df["haversine"] = R * c

    y = np.sin(delta_chg * np.cos(phi2))
    x = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(delta_chg)
    df["bearing"] = np.arctan2(y, x)
    return df


def prepare_time_features(df):
    df["pickup_datetime"] = df["pickup_datetime"].str.replace(" UTC", "")
    df["pickup_datetime"] = pd.to_datetime(
        df["pickup_datetime"], format="%Y-%m-%d %H:%M:%S", errors="coerce"
    )
    df["hour_of_day"] = df["pickup_datetime"].dt.hour
    df["week"] = df["pickup_datetime"].dt.isocalendar().week.astype(int)
    df["month"] = df["pickup_datetime"].dt.month
    df["year"] = df["pickup_datetime"].dt.year
    df["day_of_year"] = df["pickup_datetime"].dt.dayofyear
    df["weekday"] = df["pickup_datetime"].dt.weekday
    df["quarter"] = df["pickup_datetime"].dt.quarter
    df["day_of_month"] = df["pickup_datetime"].dt.day
    return df




## === cell 2
train, test = load_Data()



## === cell 3
train = prepare_time_features(train)
train = prepare_distance_features(train)

test = prepare_time_features(test)
test = prepare_distance_features(test)



## === cell 4
train["distance"] = train["haversine"] / 1000.0  # km
test["distance"] = test["haversine"] / 1000.0

train["distance_per_passenger"] = train["distance"] / train["passenger_count"].replace(
    0, np.nan
)
test["distance_per_passenger"] = test["distance"] / test["passenger_count"].replace(
    0, np.nan
)

train.drop(
    [
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "pickup_datetime",
        "haversine",
    ],
    axis=1,
    inplace=True,
)
test.drop(
    [
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "pickup_datetime",
        "haversine",
    ],
    axis=1,
    inplace=True,
)



## === cell 5
train["key2"] = pd.to_datetime(train["key"], errors="coerce")
test["key2"] = pd.to_datetime(test["key"], errors="coerce")

for df in [train, test]:
    df["year"] = df["key2"].dt.year
    df["month"] = df["key2"].dt.month
    df["day"] = df["key2"].dt.day
    df["day_of_week"] = df["key2"].dt.weekday
    df["hour"] = df["key2"].dt.hour



## === cell 6
feature_cols = [
    "passenger_count",
    "distance",
    "distance_per_passenger",
    "year",
    "month",
    "day",
    "day_of_week",
    "hour",
    "hour_of_day",
    "week",
    "day_of_year",
    "weekday",
    "quarter",
    "distance_travelled",
    "distance_travelled_sin",
    "distance_travelled_cos",
    "distance_travelled_sin_sqrd",
    "distance_travelled_cos_sqrd",
    "bearing",
]
X = train[feature_cols]
y = train["fare_amount"]



## === cell 7
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)



## === cell 8
lgb = LGBMRegressor(
    objective="regression",
    num_leaves=63,  # increased capacity
    learning_rate=0.02,  # finer step size
    n_estimators=2500,  # more trees
    max_bin=55,
    bagging_fraction=0.8,
    bagging_freq=5,
    feature_fraction=0.8,
    feature_fraction_seed=9,
    bagging_seed=9,
    min_data_in_leaf=6,
    min_sum_hessian_in_leaf=11,
    verbose=-1,
)

lgb.fit(X_train, y_train)



## === cell 9
val_pred = lgb.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, val_pred))
print(f"Validation RMSE: {rmse:.4f}")



## === cell 10
lgb.fit(X, y)



## === cell 11
X_test = test[feature_cols]
test_pred = lgb.predict(X_test)

submission = pd.DataFrame({"key": test["key"], "fare_amount": test_pred})
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission file written to {submission_path}")
