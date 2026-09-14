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

fastai==2.8.5
geopandas==0.14.4
google-api-python-client==2.177.0
ipython==7.34.0
ipython-genutils==0.2.0
ipython_pygments_lexers==1.1.1
ipython-sql==0.5.0
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

# 5. Target score

3.83596

# 6. Current score

4.3352

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 4.33418) has done: 'Implemented a safe import for `sklearnex` (fallback when unavailable), added the missing `set_rf_samples` stub, and reordered the cells so that all imports, functions, and variables are defined before they are used. The script now loads a subset of the training data, creates date and distance features, trains a `RandomForestRegressor`, evaluates RMSE on a validation split, preprocesses the test set with identical transformations, generates predictions, and writes a correctly formatted `submission.csv`. This resolves all NameError and ImportError issues, ensuring the pipeline runs end‑to‑end and produces a valid submission file.'
- What this solution (achieved 4.3352) has done: 'The fix adds all missing imports, defines the small helper utilities (`set_rf_samples`, `rmse`) and uses fastai’s `add_datepart` for datetime feature extraction. The script is reorganised into four sequential cells so every name is defined before it is used, and the training–validation split, model fitting, evaluation and test‑set prediction now run without errors, producing a correctly formatted `submission.csv`. These changes only address the runtime bugs and keep the original RandomForest model unchanged, preserving the intended scoring behavior.'

# 9. Code solution

## === cell 0
import os
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from fastai.tabular.all import add_datepart


def set_rf_samples(n):
    pass


def rmse(preds, targets):
    return np.sqrt(mean_squared_error(targets, preds))


def add_distance_features(df):
    df["longitude_traversed"] = (df["dropoff_longitude"] - df["pickup_longitude"]).abs()
    df["latitude_traversed"] = (df["dropoff_latitude"] - df["pickup_latitude"]).abs()
    R = 6371.0  # Earth radius in km
    lat1 = np.radians(df["pickup_latitude"])
    lon1 = np.radians(df["pickup_longitude"])
    lat2 = np.radians(df["dropoff_latitude"])
    lon2 = np.radians(df["dropoff_longitude"])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    df["haversine_km"] = R * c




## === cell 1
PATH = "../input"

df_raw = pd.read_csv(f"{PATH}/train.csv", nrows=500_000)

df_raw["pickup_datetime"] = pd.to_datetime(df_raw["pickup_datetime"])
add_datepart(df_raw, "pickup_datetime", drop=True, time=True)

add_distance_features(df_raw)

df_raw.dropna(inplace=True)

y = df_raw["fare_amount"].astype(float)
df_raw.drop(["fare_amount"], axis=1, inplace=True)

keys = df_raw["key"]
df_raw.drop(["key"], axis=1, inplace=True)



## === cell 2
X_train, X_valid, y_train, y_valid = train_test_split(
    df_raw, y, test_size=0.2, random_state=42
)

set_rf_samples(10000)  # no‑op placeholder
m = RandomForestRegressor(n_jobs=-1, random_state=42, n_estimators=200)

start = time.time()
m.fit(X_train, y_train)
print(f"Training time: {time.time() - start:.1f}s")

print("RMSE train:", rmse(m.predict(X_train), y_train))
print("RMSE valid:", rmse(m.predict(X_valid), y_valid))



## === cell 3
test_set = pd.read_csv(f"{PATH}/test.csv")
test_key = test_set["key"]
test_set.drop(["key"], axis=1, inplace=True)

test_set["pickup_datetime"] = pd.to_datetime(test_set["pickup_datetime"])
add_datepart(test_set, "pickup_datetime", drop=True, time=True)
add_distance_features(test_set)

test_set = test_set[df_raw.columns]

test_predictions = m.predict(test_set)

submission = pd.DataFrame({"key": test_key, "fare_amount": test_predictions})
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission file written to {submission_path}")
