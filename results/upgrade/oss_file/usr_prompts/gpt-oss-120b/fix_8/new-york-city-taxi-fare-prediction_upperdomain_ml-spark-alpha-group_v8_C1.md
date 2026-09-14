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
import os, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except Exception:
    pass

print("input folder contents:", os.listdir("../input"))




## === cell 1
train_path = "../input/train.csv"
df = pd.read_csv(train_path, nrows=2_000_000)  # 2 M rows instead of 0.5 M
df.head()




## === cell 2
df = df[df.passenger_count > 0]
df = df[df.fare_amount > 0]

alpha_ang = 0.506


def add_features(df):
    df["abs_diff_longitude"] = (df.dropoff_longitude - df.pickup_longitude).abs() * 50
    df["abs_diff_latitude"] = (df.dropoff_latitude - df.pickup_latitude).abs() * 69
    df["displacement_vector"] = np.sqrt(
        df.abs_diff_latitude**2 + df.abs_diff_longitude**2
    )
    df["actual_long"] = (
        df.displacement_vector
        * np.sin(np.arctan(df.abs_diff_longitude / df.abs_diff_latitude) - alpha_ang)
    ).abs()
    df["actual_lat"] = (
        df.displacement_vector
        * np.cos(np.arctan(df.abs_diff_longitude / df.abs_diff_latitude) - alpha_ang)
    ).abs()
    df["distance_travel"] = df["actual_long"] + df["actual_lat"]

    lon1 = np.radians(df.pickup_longitude)
    lat1 = np.radians(df.pickup_latitude)
    lon2 = np.radians(df.dropoff_longitude)
    lat2 = np.radians(df.dropoff_latitude)
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    earth_radius_miles = 3956  # miles
    df["haversine_distance"] = earth_radius_miles * c

    df["distance_travel_sq"] = df["distance_travel"] ** 2
    df["haversine_distance_sq"] = df["haversine_distance"] ** 2

    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
    df["pickup_hour"] = df["pickup_datetime"].dt.hour
    df["pickup_weekday"] = df["pickup_datetime"].dt.weekday
    df["pickup_month"] = df["pickup_datetime"].dt.month


add_features(df)

df = df[df.distance_travel > 0]
df = df[df.distance_travel < 30]  # realistic trips
df = df[df.fare_amount < 100]  # cap extreme fares




## === cell 3
l = len(df)
df_train = df.iloc[: int(0.7 * l)]
df_valid = df.iloc[int(0.7 * l) :]

train_X = np.column_stack(
    (
        df_train.distance_travel,
        df_train.haversine_distance,
        df_train.distance_travel_sq,
        df_train.haversine_distance_sq,
        df_train.passenger_count,
        df_train.pickup_hour,
        df_train.pickup_weekday,
        df_train.pickup_month,
        np.ones(len(df_train)),
    )
).astype(np.float32)

valid_X = np.column_stack(
    (
        df_valid.distance_travel,
        df_valid.haversine_distance,
        df_valid.distance_travel_sq,
        df_valid.haversine_distance_sq,
        df_valid.passenger_count,
        df_valid.pickup_hour,
        df_valid.pickup_weekday,
        df_valid.pickup_month,
        np.ones(len(df_valid)),
    )
).astype(np.float32)

train_y_log = np.log1p(df_train.fare_amount.values)
valid_y = df_valid.fare_amount.values




## === cell 4
imputer = SimpleImputer(strategy="mean")
train_X_imp = imputer.fit_transform(train_X)
valid_X_imp = imputer.transform(valid_X)

scaler = StandardScaler()
train_X_scaled = scaler.fit_transform(train_X_imp)
valid_X_scaled = scaler.transform(valid_X_imp)

regr = GradientBoostingRegressor(
    n_estimators=1200,
    learning_rate=0.02,
    max_depth=5,
    subsample=1.0,  # use full data per tree
    random_state=42,
)
regr.fit(train_X_scaled, train_y_log)

pred_valid_log = regr.predict(valid_X_scaled)
pred_valid = np.expm1(pred_valid_log)  # revert log‑transform
rmse = mean_squared_error(valid_y, pred_valid, squared=False)
print(f"Validation RMSE: {rmse:.4f}")




## === cell 5
test_path = "../input/test.csv"
tdf = pd.read_csv(test_path, nrows=1_000_000)  # test set is already small
add_features(tdf)

test_X = np.column_stack(
    (
        tdf.distance_travel,
        tdf.haversine_distance,
        tdf.distance_travel_sq,
        tdf.haversine_distance_sq,
        tdf.passenger_count,
        tdf.pickup_hour,
        tdf.pickup_weekday,
        tdf.pickup_month,
        np.ones(len(tdf)),
    )
).astype(np.float32)

test_X_imp = imputer.transform(test_X)
test_X_scaled = scaler.transform(test_X_imp)
test_pred_log = regr.predict(test_X_scaled)
output = np.expm1(test_pred_log)  # convert back from log scale




## === cell 6
submission = pd.DataFrame({"key": tdf["key"], "fare_amount": output})
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
print(submission.head())
