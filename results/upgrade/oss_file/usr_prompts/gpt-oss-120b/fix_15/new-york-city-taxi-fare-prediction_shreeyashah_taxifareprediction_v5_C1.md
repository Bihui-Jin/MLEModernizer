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

3.14

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

4.736901452993294

# 6. Current score

5.44567

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 108.82752) has done: 'I drop the non‑numeric `pickup_datetime` column before converting the training features to a NumPy array, ensuring the models can be fitted. This fixes the conversion error and the subsequent “not fitted” errors, allowing the script to run end‑to‑end and produce a valid `submission.csv` file.'
- What this solution (achieved 108.90601) has done: 'The changes keep the same model definitions and feature engineering but shrink the training set to 1 million rows (still large enough for stable models) and add a modest `max_samples=0.7` to the RandomForest so each tree trains on a subset, which cuts the fitting time dramatically without materially affecting prediction quality. All other logic, data paths, and output remain unchanged.'
- What this solution (achieved 5.48475) has done: 'I switch the models to predict the log‑transformed fare amount (using np.log1p) and then exponentiate the ensemble prediction with np.expm1, which better handles the skewed distribution and usually lowers RMSE. I also simplify the ensemble by relying solely on the RandomForest (which performed best) and increase its capacity (more trees and full‑sample training) for a modest gain without altering the overall pipeline. These targeted tweaks keep the original feature engineering and model types while moving the validation score closer to the target.'
- What this solution (achieved 5.44567) has done: 'I adjust the RandomForest hyper‑parameters to improve its generalisation while keeping the overall pipeline unchanged. Increasing the number of trees and limiting each split to a sqrt of the features usually lowers RMSE on this type of data, moving the score nearer the target.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os
import gc
import warnings
from sklearnex import patch_sklearn

warnings.filterwarnings("ignore")
patch_sklearn()
np.random.seed(2)  # ensure reproducibility for any stochastic ops




## === cell 1
dtype_train = {
    "fare_amount": "float32",
    "pickup_datetime": "object",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int8",
}
usecols_train = [
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
df_train = pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv",
    nrows=1_000_000,
    dtype=dtype_train,
    usecols=usecols_train,
)
df_train.head()




## === cell 2
df_train = df_train.iloc[:, :]




## === cell 3
print(df_train.shape)
print(df_train.isna().sum())




## === cell 4
mask = (
    df_train["fare_amount"].between(1, 100)
    & df_train["passenger_count"].between(1, 6)
    & df_train["pickup_longitude"].between(-75, -72)
    & df_train["dropoff_longitude"].between(-75, -72)
    & df_train["pickup_latitude"].between(40, 42)
    & df_train["dropoff_latitude"].between(40, 42)
)
df_train = df_train.dropna().loc[mask].reset_index(drop=True)

df_train["pickup_datetime"] = pd.to_datetime(df_train["pickup_datetime"])
df_train["hour"] = df_train["pickup_datetime"].dt.hour.astype(np.int8)
df_train["weekday"] = df_train["pickup_datetime"].dt.weekday.astype(np.int8)
df_train["month"] = df_train["pickup_datetime"].dt.month.astype(np.int8)




## === cell 5
def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371 * c  # km




## === cell 6
df_train["distance_km"] = haversine(
    df_train["pickup_latitude"],
    df_train["pickup_longitude"],
    df_train["dropoff_latitude"],
    df_train["dropoff_longitude"],
)
df_train = df_train[df_train["distance_km"].between(0.1, 30)].reset_index(drop=True)




## === cell 7
y = np.log1p(df_train["fare_amount"].to_numpy(dtype=np.float32))
X = (
    df_train.drop(columns=["pickup_datetime", "fare_amount"])
    .astype(np.float32)
    .to_numpy()
)




## === cell 8
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(
    n_estimators=800,  # increased trees for finer approximation
    max_depth=None,
    max_features="sqrt",  # use sqrt of features per split to reduce variance
    random_state=2,
    bootstrap=True,
    max_samples=1.0,
    n_jobs=-1,
)




## === cell 9
rf.fit(X, y)

del df_train, X, y
gc.collect()




## === cell 10
df_test = pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv",
    parse_dates=["pickup_datetime"],
)
key = df_test["key"]

df_test["hour"] = df_test["pickup_datetime"].dt.hour.astype(np.int8)
df_test["weekday"] = df_test["pickup_datetime"].dt.weekday.astype(np.int8)
df_test["month"] = df_test["pickup_datetime"].dt.month.astype(np.int8)

df_test = df_test.iloc[:, 2:]  # drop key and pickup_datetime




## === cell 11
df_test["distance_km"] = haversine(
    df_test["pickup_latitude"],
    df_test["pickup_longitude"],
    df_test["dropoff_latitude"],
    df_test["dropoff_longitude"],
)




## === cell 12
X_test = df_test.astype(np.float32).to_numpy()

log_pred = rf.predict(X_test)
y_pred = np.expm1(log_pred)  # inverse of log1p
y_pred = np.clip(y_pred, 0, None)




## === cell 13
results = pd.DataFrame({"key": key, "fare_amount": y_pred})
print(results.head())
results.to_csv("submission.csv", index=False)
