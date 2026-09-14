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

3.50273

# 6. Current score

4.91994

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 4.65815) has done: 'I add a simple imputation step that fills missing feature values with the median of each column, drop any rows where the target `fare_amount` is missing, and then run the original RandomForest pipeline. This removes the NaN‑related errors that prevented fitting and consequently fixes the missing‑attribute issue in the prediction step, while keeping the core model unchanged.'
- What this solution (achieved 4.79571) has done: 'The update patches scikit‑learn with Intel‑optimized kernels (sklearnex), casts feature data to float32 and uses NumPy arrays for the model to reduce overhead, while keeping the exact RandomForest configuration and feature engineering unchanged. These changes speed up the heavy tree training phase and avoid unnecessary pandas copying, allowing the script to finish well within the 600 s limit without affecting prediction accuracy.'
- What this solution (achieved 4.90114) has done: 'The fix removes rows where the target `fare_amount` is missing or non‑positive, filters out any NaN/infinite values that appear after the log1p transform, and adds a log‑scaled distance feature to give the model a bit more predictive power. These changes prevent the `ValueError` during fitting and allow the script to produce a proper `RFSubmission.csv` while nudging the RMSE closer to the target.'
- What this solution (achieved 4.64363) has done: 'I keep the overall RandomForest pipeline and feature engineering, but switch the model to train on the raw fare amounts (removing the log‑transform) and increase the number of trees slightly to give the forest more capacity. These minimal adjustments should lower the validation RMSE, moving the score closer to the target while preserving the core logic and staying within the runtime limit.'
- What this solution (achieved 4.63933) has done: 'The changes focus on speeding up the RandomForest training and inference without altering the feature engineering or the overall modeling approach. We lower the number of trees, use the default “sqrt” feature selection (which is faster and still a standard RF setting), and add `max_samples` to train each tree on a random subset of rows, dramatically reducing workload while keeping the model type unchanged. We also pre‑convert the split DataFrames to NumPy arrays once to avoid repeated `.values` calls during fit and prediction.'
- What this solution (achieved 4.91994) has done: 'I keep the overall RandomForest pipeline but switch to a log‑transformed target (training on log1p fare and exponentiating predictions) and remove the aggressive lower‑bound clipping, which should reduce RMSE on the original scale. I also raise the number of trees slightly (800→1000) to give the model a bit more capacity while staying within the runtime limits. These minimal tweaks preserve the core logic and are aimed at moving the validation RMSE closer to the target score.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except ImportError:
    pass  # fallback to regular scikit‑learn if sklearnex is unavailable

n_train = 600_000  # lowered from 1_200_000

dtype_map = {
    "fare_amount": np.float32,
    "pickup_longitude": np.float64,
    "pickup_latitude": np.float64,
    "dropoff_longitude": np.float64,
    "dropoff_latitude": np.float64,
    "passenger_count": np.int8,
}

df = pd.read_csv(
    "../input/train.csv",
    nrows=n_train,
    parse_dates=["pickup_datetime"],
    dtype=dtype_map,
)

df_test = pd.read_csv(
    "../input/test.csv",
    parse_dates=["pickup_datetime"],
    dtype=dtype_map,
)

base_feature_cols = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]

df = df.dropna(subset=["fare_amount"])
df = df[df["fare_amount"] > 0]

medians = df[base_feature_cols].median()
df[base_feature_cols] = df[base_feature_cols].fillna(medians)
df_test[base_feature_cols] = df_test[base_feature_cols].fillna(medians)


def haversine_distance(lat1, lon1, lat2, lon2):
    """Return distance in km between two lat/lon points."""
    R = 6371.0  # Earth radius in km
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = (
        np.sin(dphi / 2.0) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    )
    return 2 * R * np.arcsin(np.sqrt(a))


for dataset in (df, df_test):
    dataset["distance_km"] = haversine_distance(
        dataset["pickup_latitude"],
        dataset["pickup_longitude"],
        dataset["dropoff_latitude"],
        dataset["dropoff_longitude"],
    )
    dataset["distance_km_log"] = np.log1p(dataset["distance_km"])
    dataset["pickup_hour"] = dataset["pickup_datetime"].dt.hour
    dataset["pickup_weekday"] = dataset["pickup_datetime"].dt.weekday
    dataset["pickup_month"] = dataset["pickup_datetime"].dt.month

feature_cols = base_feature_cols + [
    "distance_km",
    "distance_km_log",
    "pickup_hour",
    "pickup_weekday",
    "pickup_month",
]

y = np.log1p(df["fare_amount"]).astype(np.float32)

X = df[feature_cols].astype(np.float32)

df_train, df_val, y_train, y_val = train_test_split(
    X, y, test_size=0.1, random_state=42
)

X_train_np = df_train.to_numpy(dtype=np.float32, copy=False)
X_val_np = df_val.to_numpy(dtype=np.float32, copy=False)



## === cell 1
from sklearn.ensemble import RandomForestRegressor

reg = RandomForestRegressor(
    n_estimators=1000,  # slightly more trees for better accuracy
    max_depth=None,
    min_samples_split=2,
    max_features="sqrt",  # default fast option
    max_samples=0.7,  # train each tree on 70 % of rows
    oob_score=False,
    n_jobs=-1,
    random_state=42,
    verbose=1,
)



## === cell 2
reg.fit(X_train_np, y_train.values)

from sklearn.metrics import mean_squared_error, r2_score

y_pred_val_log = reg.predict(X_val_np)
y_pred_val = np.expm1(y_pred_val_log)

rmse = mean_squared_error(np.expm1(y_val.values), y_pred_val, squared=False)
print(f"RMSE on validation set: {rmse:.5f}")

r2 = r2_score(np.expm1(y_val.values), y_pred_val)
print(f"R2 score on validation set: {r2:.5f}")



## === cell 3
X_test = df_test[feature_cols].astype(np.float32)
test_pred_log = reg.predict(X_test.values)
predictions = np.expm1(test_pred_log)

predictions = np.clip(predictions, a_min=0.0, a_max=None)

RFSubmission = pd.DataFrame(
    {
        "key": df_test["key"].values,
        "fare_amount": predictions,
    }
)

RFSubmission.to_csv("RFSubmission.csv", index=False)
