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

3.87285

# 6. Current score

5.71904

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 7.53123) has done: 'I fix the data‑filtering bug, add useful features (minute and passenger count), increase the RandomForest size for better learning, compute the true RMSE, and comment‑out the plotting cells that caused errors. These changes keep the original modeling approach while improving performance and ensuring a valid submission.csv is written.'
- What this solution (achieved 5.71904) has done: 'I speed up the heavy RandomForest training by enabling the Intel‑optimized scikit‑learn implementation (`sklearnex`) which provides a drop‑in replacement with the same API and identical results, and I cast feature matrices to `float32` to lower memory bandwidth while preserving numerical precision. No algorithmic steps are altered, only the underlying efficient implementation and data types, so the model’s predictions remain unchanged.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except Exception:
    pass  # fallback to regular scikit‑learn if sklearnex unavailable
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error




## === cell 1
cols_needed = [
    "key",
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
train_df = pd.read_csv("../input/train.csv", nrows=5_000_000, usecols=cols_needed)




## === cell 2
def haversine_np(lon1, lat1, lon2, lat2):
    """
    Calculate great‑circle distance between two points (in km).
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
train_df = train_df[(train_df["fare_amount"] > 0) & (train_df["fare_amount"] < 200)]
print("New size after dropna and fare filter: %d" % len(train_df))




## === cell 7
new_york_lat = 40
new_york_long = -74
train_df.describe()




## === cell 8
mask = (
    (train_df["pickup_latitude"].between(40, 41))
    & (train_df["dropoff_latitude"].between(40, 41))
    & (train_df["pickup_longitude"].between(-74.5, -73))
    & (train_df["dropoff_longitude"].between(-74.5, -73))
)
print("Rows before geographic filter: %d" % len(train_df))
train_df = train_df[mask]
print("Rows after geographic filter: %d" % len(train_df))




## === cell 9
train_df.describe()




## === cell 10
feature_cols = ["distance", "year", "month", "day", "hour", "minute", "passenger_count"]
X = train_df[feature_cols].astype(np.float32).values
Y = train_df["fare_amount"].astype(np.float32).values




## === cell 11
kwargs = {
    "bootstrap": True,
    "max_depth": 20,  # limit tree depth to speed training
    "max_features": "sqrt",
    "min_samples_leaf": 1,
    "min_samples_split": 2,
    "n_estimators": 200,  # reduced number of trees (still a forest)
    "random_state": 42,
    "n_jobs": -1,
    "oob_score": True,
    "max_samples": 0.5,  # each tree sees 50 % of the data
}
rand_regr = RandomForestRegressor(**kwargs)




## === cell 12
rand_regr.fit(X, Y)




## === cell 13
y_pred_train = rand_regr.predict(X)
rmse = np.sqrt(mean_squared_error(Y, y_pred_train))
print(f"RMSE on training data: {rmse:.4f}")

if hasattr(rand_regr, "oob_prediction_") and rand_regr.oob_score:
    oob_rmse = np.sqrt(mean_squared_error(Y, rand_regr.oob_prediction_))
    print(f"OOB RMSE estimate: {oob_rmse:.4f}")




## === cell 14
print(f"R^2 score on training data: {rand_regr.score(X, Y):.4f}")




## === cell 15
test_df = pd.read_csv(
    "../input/test.csv",
    usecols=[
        "key",
        "pickup_datetime",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
    ],
)




## === cell 16
test_df["distance"] = haversine_np(
    test_df["pickup_longitude"],
    test_df["pickup_latitude"],
    test_df["dropoff_longitude"],
    test_df["dropoff_latitude"],
)




## === cell 17
test_df["pickup_datetime"] = pd.to_datetime(test_df["pickup_datetime"])




## === cell 18
test_df["year"] = test_df["pickup_datetime"].dt.year
test_df["month"] = test_df["pickup_datetime"].dt.month
test_df["day"] = test_df["pickup_datetime"].dt.day
test_df["hour"] = test_df["pickup_datetime"].dt.hour
test_df["minute"] = test_df["pickup_datetime"].dt.minute




## === cell 19
X_to_pred = test_df[feature_cols].astype(np.float32).values
y_pred_test = rand_regr.predict(X_to_pred)




## === cell 20
submission = pd.DataFrame(
    {"key": test_df["key"], "fare_amount": y_pred_test}, columns=["key", "fare_amount"]
)
submission.to_csv("submission.csv", index=False)
print("Submission file written to submission.csv with", len(submission), "rows.")




## === cell 21
try:
    with sns.axes_style("white"):
        sns.jointplot(x=X.flatten(), y=Y, kind="hex", color="k", bins="log")
except Exception as e:
    print("Skipping plot due to error:", e)




## === cell 22
try:
    X_flat = X.flatten()
    mask = (X_flat < 50) & (Y < 100)
    with sns.axes_style("white"):
        p = sns.jointplot(x=X_flat[mask], y=Y[mask], kind="hex", color="k", bins="log")
        x_line = np.arange(0, 50, 1)
        y_line = np.full_like(x_line, Y.mean())
        p.ax_joint.plot(x_line, y_line, color="red")
except Exception as e:
    print("Skipping second plot due to error:", e)
