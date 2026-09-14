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

3.9

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

5.68916

# 6. Current score

4.70011

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 10.02063) has done: 'I fixed the file‑path issue by automatically locating the CSV files, cleaned the data, added the same distance‑based features the original notebook used, trained a linear regression model, and finally wrote a correctly‑named `submission.csv` containing the required columns. This restores end‑to‑end execution and should produce an RMS score close to the target.'
- What this solution (achieved 10.03695) has done: 'I keep the overall pipeline and LinearRegression model unchanged but add modest feature engineering (hour sin/cos, squared distance and passenger count, plus outlier removal of extreme fares). These extra linear‑compatible features should help the linear model capture non‑linear patterns and reduce the RMS error, moving the score closer to the target without altering the core logic.'
- What this solution (achieved 12.2514) has done: 'The update adds simple scaling and quadratic feature expansion before fitting the linear regression, which lets the model capture interaction effects without changing its core linear‑regression nature. This richer feature set is expected to lower the validation RMSE, moving the score closer to the target while keeping the original pipeline and output intact.'
- What this solution (achieved 11.51071) has done: 'I replace the plain LinearRegression with a Ridge regression (L2‑regularized) while keeping the same feature pipeline. Regularization should reduce over‑fitting of the quadratic feature set and lower the validation RMSE, moving the score closer to the target without altering the core logic.'
- What this solution (achieved 4.70011) has done: 'I replace the polynomial‑Ridge pipeline with a Gradient Boosting regressor, which better captures the nonlinear relationships in the engineered features while keeping the existing feature construction unchanged. This change is expected to lower the validation RMSE substantially, moving the score toward the target 5.68916, and the rest of the pipeline (data loading, feature engineering, and submission writing) remains the same.'

# 9. Code solution

## === cell 0
import os, glob, pathlib, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


def find_file(filename):
    matches = glob.glob(f"/kaggle/input/**/{filename}", recursive=True)
    if not matches:
        raise FileNotFoundError(f"{filename} not found under /kaggle/input")
    return matches[0]


train_path = find_file("train.csv")
test_path = find_file("test.csv")
sample_sub_path = find_file("sample_submission.csv")



## === cell 1
train_dt = pd.read_csv(train_path, nrows=200_000)
test_dt = pd.read_csv(test_path)



## === cell 2
train_dt.dropna(inplace=True)
test_dt.dropna(inplace=True)  # just in case

train_dt = train_dt[train_dt["fare_amount"] <= 200]



## === cell 3
R = 6373.0  # Earth radius in km


def haversine(df):
    lat1 = np.radians(df["pickup_latitude"])
    lon1 = np.radians(df["pickup_longitude"])
    lat2 = np.radians(df["dropoff_latitude"])
    lon2 = np.radians(df["dropoff_longitude"])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return (R * c) * 0.621371  # miles


def airport_distance(df, lat_air=40.6413111, lon_air=-73.7781391):
    lat_air = np.radians(lat_air)
    lon_air = np.radians(lon_air)
    lat_pick = np.radians(df["pickup_latitude"])
    lon_pick = np.radians(df["pickup_longitude"])
    lat_drop = np.radians(df["dropoff_latitude"])
    lon_drop = np.radians(df["dropoff_longitude"])

    dlon = lon_air - lon_pick
    dlat = lat_air - lat_pick
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat_pick) * np.cos(lat_air) * np.sin(dlon / 2) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    pick_dist = (R * c) * 0.621371

    dlon = lon_air - lon_drop
    dlat = lat_air - lat_drop
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat_drop) * np.cos(lat_air) * np.sin(dlon / 2) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    drop_dist = (R * c) * 0.621371

    return pick_dist, drop_dist


train_dt["Distance"] = haversine(train_dt)
test_dt["Distance"] = haversine(test_dt)

train_dt["Pickup_Distance_airport"], train_dt["Dropoff_Distance_airport"] = (
    airport_distance(train_dt)
)
test_dt["Pickup_Distance_airport"], test_dt["Dropoff_Distance_airport"] = (
    airport_distance(test_dt)
)

train_dt["pickup_datetime"] = pd.to_datetime(train_dt["pickup_datetime"])
test_dt["pickup_datetime"] = pd.to_datetime(test_dt["pickup_datetime"])

train_dt["hour"] = train_dt["pickup_datetime"].dt.hour
train_dt["weekday"] = train_dt["pickup_datetime"].dt.weekday
test_dt["hour"] = test_dt["pickup_datetime"].dt.hour
test_dt["weekday"] = test_dt["pickup_datetime"].dt.weekday

train_dt["hour_sin"] = np.sin(2 * np.pi * train_dt["hour"] / 24)
train_dt["hour_cos"] = np.cos(2 * np.pi * train_dt["hour"] / 24)
test_dt["hour_sin"] = np.sin(2 * np.pi * test_dt["hour"] / 24)
test_dt["hour_cos"] = np.cos(2 * np.pi * test_dt["hour"] / 24)

train_dt["Distance_sq"] = train_dt["Distance"] ** 2
test_dt["Distance_sq"] = test_dt["Distance"] ** 2

train_dt["passenger_count_sq"] = train_dt["passenger_count"] ** 2
test_dt["passenger_count_sq"] = test_dt["passenger_count"] ** 2

train_dt.drop(
    columns=[
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "pickup_datetime",
    ],
    inplace=True,
)
test_dt.drop(
    columns=[
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "pickup_datetime",
    ],
    inplace=True,
)



## === cell 4
X = train_dt.drop(columns=["key", "fare_amount"])
y = train_dt["fare_amount"]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.05, random_state=42)

from sklearn.ensemble import GradientBoostingRegressor

model = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    random_state=42,
)

model.fit(X_train, y_train)

from sklearn.metrics import mean_squared_error

val_pred = model.predict(X_val)
val_rmse = mean_squared_error(y_val, val_pred, squared=False)
print(f"Validation RMSE: {val_rmse:.4f}")



## === cell 5
test_features = test_dt.drop(columns=["key"])
test_pred = model.predict(test_features)
test_pred = np.maximum(test_pred, 0)  # fares cannot be negative
test_pred = np.round(test_pred, 2)



## === cell 6
submission = pd.DataFrame({"key": test_dt["key"], "fare_amount": test_pred})
submission_path = os.path.join("/kaggle/working", "submission.csv")
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
