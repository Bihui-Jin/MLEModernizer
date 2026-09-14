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

4.24009

# 6. Current score

7.00011

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 7.22808) has done: 'I fixed the runtime errors by correcting the plotting code to use matching‑length arrays, added the missing `passenger_count` feature to the model, and upgraded the RandomForest to a stronger configuration (more trees and a fixed random state). These changes keep the original modeling approach intact while improving predictive power, moving the RMSE closer to the target. The script now runs end‑to‑end and writes a proper `submission.csv` file.'
- What this solution (achieved 7.00011) has done: 'We speed up data loading by reading only the needed columns, limiting rows to 500 k, and specifying lightweight dtypes, which cuts I/O and memory overhead without altering the model. The geographic filter is rewritten with a fully vectorized boolean mask, eliminating the Python‑level loop and speeding the filtering step while preserving exactly the same selection criteria. All other logic—including feature engineering, LinearRegression, and the 500‑tree RandomForest—remains unchanged.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

from sklearnex import patch_all

patch_all()
np.random.seed(42)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_11/3645571156.py in <cell line: 0>()
      7 from sklearn.ensemble import RandomForestRegressor
      8 
----> 9 from sklearnex import patch_all
     10 
     11 patch_all()

ImportError: cannot import name 'patch_all' from 'sklearnex' (/usr/local/lib/python3.11/dist-packages/sklearnex/__init__.py)

## === cell 1
usecols = [
    "key",
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
dtype_map = {
    "fare_amount": np.float32,
    "pickup_longitude": np.float32,
    "pickup_latitude": np.float32,
    "dropoff_longitude": np.float32,
    "dropoff_latitude": np.float32,
    "passenger_count": np.int8,
}
train_df = pd.read_csv(
    "../input/train.csv",
    nrows=500_000,  # smaller yet still representative sample
    usecols=usecols,
    dtype=dtype_map,
    low_memory=False,
)




## === cell 2
def haversine_np(lon1, lat1, lon2, lat2):
    """Calculate great‑circle distance (km) between two points."""
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
print("Old size:", len(train_df))
train_df = train_df.dropna(how="any")
print("New size after NA drop:", len(train_df))




## === cell 7
geo_cols = [
    "pickup_latitude",
    "pickup_longitude",
    "dropoff_latitude",
    "dropoff_longitude",
]
means = train_df[geo_cols].mean()
cond = (np.abs(train_df[geo_cols] - means) < 5).all(axis=1)
print("Old size before geo‑filter:", len(train_df))
train_df = train_df[cond]
print("New size after geo‑filter:", len(train_df))




## === cell 8
train_df["distance_per_passenger"] = np.where(
    train_df["passenger_count"] > 0,
    train_df["distance"] / train_df["passenger_count"],
    train_df["distance"],
)

feature_cols = [
    "distance",
    "distance_per_passenger",
    "year",
    "month",
    "day",
    "hour",
    "passenger_count",
]
X = train_df[feature_cols].values.astype(np.float32)  # float32 speeds up RF
Y = train_df["fare_amount"].values.astype(np.float32)




## === cell 9
lin_reg = LinearRegression()
lin_reg.fit(train_df[["distance"]].values, Y)
y_pred_lin = lin_reg.predict(train_df[["distance"]].values)
print("RMSE linear (distance only):", np.sqrt(np.mean((Y - y_pred_lin) ** 2)))




## === cell 10
rand_regr = RandomForestRegressor(
    n_estimators=500,
    random_state=42,
    n_jobs=5,
    max_features="sqrt",
    min_samples_leaf=1,
)
rand_regr.fit(X, Y)
y_pred_rf = rand_regr.predict(X)
print("RMSE RandomForest (enhanced):", np.sqrt(np.mean((Y - y_pred_rf) ** 2)))




## === cell 11
test_df = pd.read_csv("../input/test.csv")
test_df["distance"] = haversine_np(
    test_df["pickup_longitude"],
    test_df["pickup_latitude"],
    test_df["dropoff_longitude"],
    test_df["dropoff_latitude"],
)
test_df["distance_per_passenger"] = np.where(
    test_df["passenger_count"] > 0,
    test_df["distance"] / test_df["passenger_count"],
    test_df["distance"],
)
test_df["pickup_datetime"] = pd.to_datetime(test_df["pickup_datetime"])
test_df["year"] = test_df["pickup_datetime"].dt.year
test_df["month"] = test_df["pickup_datetime"].dt.month
test_df["day"] = test_df["pickup_datetime"].dt.day
test_df["hour"] = test_df["pickup_datetime"].dt.hour
test_df["minute"] = test_df["pickup_datetime"].dt.minute




## === cell 12
X_test = test_df[feature_cols].values.astype(np.float32)
test_pred = rand_regr.predict(X_test)




## === cell 13
submission = pd.DataFrame({"key": test_df["key"], "fare_amount": test_pred})
submission.to_csv("submission.csv", index=False)
print("Submission written to submission.csv with", len(submission), "rows.")




## === cell 14
try:
    distance_arr = train_df["distance"].values
    with sns.axes_style("white"):
        sns.jointplot(x=distance_arr, y=Y, kind="hex", color="k", bins="log")
except Exception as e:
    print("Plotting cell 15 skipped due to:", e)




## === cell 15
try:
    mask = (distance_arr < 50) & (Y < 100)
    with sns.axes_style("white"):
        p = sns.jointplot(
            x=distance_arr[mask], y=Y[mask], kind="hex", color="k", bins="log"
        )
    x_vals = np.arange(0, 50, 0.5).reshape(-1, 1)
    y_vals = lin_reg.predict(x_vals)
    p.ax_joint.plot(x_vals, y_vals, color="red")
except Exception as e:
    print("Plotting cell 16 skipped due to:", e)
