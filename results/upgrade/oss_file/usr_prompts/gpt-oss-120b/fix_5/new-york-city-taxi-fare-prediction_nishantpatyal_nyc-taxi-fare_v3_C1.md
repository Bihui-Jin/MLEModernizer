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

3.8

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
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1

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

4.87021

# 6. Current score

6.36488

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 6.4543) has done: 'I fixed the deprecated `DataFrame.drop` usage, replaced the failing TensorFlow model with a scikit‑learn GradientBoostingRegressor (which avoids the protobuf incompatibility), added a proper RMSE calculation, and ensured the script writes a correctly‑named `submission.csv` containing the required columns.'
- What this solution (achieved 6.3248) has done: 'I keep the overall pipeline unchanged but train the model on a log‑transformed target, then exponentiate the predictions back to the original scale. This often reduces RMSE for skewed fare amounts while preserving the same features and model. I also adjust the validation comparison to the original fare values, and convert the test predictions back before writing the submission.'
- What this solution (achieved 6.4566) has done: 'I keep the overall pipeline and feature engineering unchanged, but tune the GradientBoostingRegressor to be slightly more expressive (more trees, deeper, lower learning rate, and a robust Huber loss). After validation I compute the average residual bias and apply it to the test predictions, which often reduces systematic error and moves the RMSE closer to the target.'
- What this solution (achieved 6.36488) has done: 'I add a log‑transformed distance feature (which often captures the non‑linear effect of trip length) and slightly increase the GradientBoostingRegressor capacity by using more trees with a modestly lower learning rate. These changes keep the overall pipeline intact while providing the model with richer information and a bit more fitting power, which should lower the validation RMSE and move the score closer to the target.'

# 9. Code solution

## === cell 0
import os, numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))




## === cell 1
sample_sub = pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/sample_submission.csv"
)
train_df = pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv", nrows=500000
)
test_df = pd.read_csv("/kaggle/input/new-york-city-taxi-fare-prediction/test.csv")

train_df["pickup_datetime"] = pd.to_datetime(train_df["pickup_datetime"])
test_df["pickup_datetime"] = pd.to_datetime(test_df["pickup_datetime"])

key = test_df["key"].copy()

train_df = train_df.drop(columns=["key"])
test_df = test_df.drop(columns=["key"])




## === cell 2
train_df = train_df[(train_df["fare_amount"] > 0) & (train_df["fare_amount"] < 100)]




## === cell 3
def haversine_distance(lat1, lon1, lat2, lon2):
    """Vectorised haversine distance in miles."""
    p = np.pi / 180.0
    a = (
        0.5
        - np.cos((lat2 - lat1) * p) / 2.0
        + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2.0
    )
    return 0.6213712 * 12742 * np.arcsin(np.sqrt(a))


train_df["distance_miles"] = haversine_distance(
    train_df["pickup_latitude"],
    train_df["pickup_longitude"],
    train_df["dropoff_latitude"],
    train_df["dropoff_longitude"],
)
test_df["distance_miles"] = haversine_distance(
    test_df["pickup_latitude"],
    test_df["pickup_longitude"],
    test_df["dropoff_latitude"],
    test_df["dropoff_longitude"],
)

train_df["distance_log"] = np.log1p(train_df["distance_miles"])
test_df["distance_log"] = np.log1p(test_df["distance_miles"])




## === cell 4
train_df = train_df.drop(
    columns=[
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
    ]
)
test_df = test_df.drop(
    columns=[
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
    ]
)

for df in (train_df, test_df):
    df["hour_of_day"] = df["pickup_datetime"].dt.hour
    df["dayofyear"] = df["pickup_datetime"].dt.dayofyear
    df["month"] = df["pickup_datetime"].dt.month
    df["year"] = df["pickup_datetime"].dt.year % 100  # last two digits

train_df = train_df.drop(columns=["pickup_datetime"])
test_df = test_df.drop(columns=["pickup_datetime"])

train_df = train_df[train_df["distance_miles"] < 17]




## === cell 5
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

X = train_df.drop(columns=["fare_amount"]).values
y = train_df["fare_amount"].values

y_log = np.log1p(y)

X_train, X_val, y_train_log, y_val_log = train_test_split(
    X, y_log, test_size=0.30, random_state=101
)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
test_X = scaler.transform(test_df.values)




## === cell 6
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import math

gbr = GradientBoostingRegressor(
    n_estimators=800, learning_rate=0.02, max_depth=5, loss="huber", random_state=101
)
gbr.fit(X_train, y_train_log)

val_pred_log = gbr.predict(X_val)
val_pred = np.expm1(val_pred_log)
y_val = np.expm1(y_val_log)

rmse = math.sqrt(mean_squared_error(y_val, val_pred))
print(f"Validation RMSE: {rmse:.5f}")

bias = np.mean(y_val - val_pred)
print(f"Validation bias (to be added to test predictions): {bias:.5f}")




## === cell 7
test_pred_log = gbr.predict(test_X)
test_pred = np.expm1(test_pred_log)

test_pred = test_pred + bias

test_pred = np.clip(test_pred, a_min=0, a_max=None)

submission = pd.DataFrame({"key": key, "fare_amount": test_pred})

submission.to_csv("submission.csv", index=False)
print("Submission file written to submission.csv")
