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

5.91416

# 6. Current score

None

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

- What this solution (achieved 1128.25357) has done: 'I add more informative linear features (Haversine distance, passenger count, and hour of day) while keeping the same least‑squares linear model. These extra columns improve the model’s fit, which should pull the RMSE down toward the target without altering the overall approach.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O
import os, math

print("Input directory contents:", os.listdir("../input"))




## === cell 1
data = pd.read_csv("../input/train.csv", nrows=20_000_000)




## === cell 2
def add_travel_vector_features(df):
    df["abs_diff_longitude"] = (df.dropoff_longitude - df.pickup_longitude).abs()
    df["abs_diff_latitude"] = (df.dropoff_latitude - df.pickup_latitude).abs()
    df["hour"] = pd.to_datetime(df.pickup_datetime).dt.hour.astype(np.float32)
    df["passenger_count"] = df["passenger_count"].astype(np.float32)
    lon1 = np.radians(df["pickup_longitude"].astype(np.float64))
    lat1 = np.radians(df["pickup_latitude"].astype(np.float64))
    lon2 = np.radians(df["dropoff_longitude"].astype(np.float64))
    lat2 = np.radians(df["dropoff_latitude"].astype(np.float64))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lon1) * np.cos(lon2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    earth_radius_km = 6371.0
    df["distance_km"] = earth_radius_km * c


add_travel_vector_features(data)




## === cell 3
print("Initial rows:", len(data))
data = data.dropna(how="any", axis="rows")
data = data[(data.fare_amount > 0) & (data.fare_amount < 200)]
print("Rows after NaN/outlier filter:", len(data))




## === cell 4
print("Rows before geographic filter:", len(data))
data = data[(data.abs_diff_longitude < 0.5) & (data.abs_diff_latitude < 0.5)]
print("Rows after geographic filter:", len(data))




## === cell 5
from sklearn.model_selection import train_test_split

y = data.fare_amount
X = data.drop("fare_amount", axis=1)
train_df, val_df, train_y, val_y = train_test_split(
    X, y, test_size=0.2, random_state=42
)




## === cell 6
def get_input_matrix(df):
    return np.column_stack(
        (
            df["distance_km"].values,
            df["passenger_count"].values,
            df["hour"].values,
            df["abs_diff_longitude"].values,
            df["abs_diff_latitude"].values,
            np.ones(len(df)),
        )
    )


train_X_raw = get_input_matrix(train_df)
print("train_X shape (raw):", train_X_raw.shape, "train_y shape:", train_y.shape)




## === cell 7
scaler_mean = train_X_raw.mean(axis=0)
scaler_std = train_X_raw.std(axis=0)
scaler_std[scaler_std == 0] = 1.0

scaler_mean[-1] = 0.0
scaler_std[-1] = 1.0

train_X = (train_X_raw - scaler_mean) / scaler_std




## === cell 8
train_y_log = np.log1p(train_y.values)
alpha = 1.0
XtX = train_X.T @ train_X
ridge_matrix = XtX + alpha * np.eye(XtX.shape[0])
Xty = train_X.T @ train_y_log
w = np.linalg.solve(ridge_matrix, Xty)
print("Ridge coefficients (log‑target):", w)




## === cell 9
test_df = pd.read_csv("../input/test.csv")
add_travel_vector_features(test_df)

test_X_raw = get_input_matrix(test_df)
val_X_raw = get_input_matrix(val_df)

val_X = (val_X_raw - scaler_mean) / scaler_std
test_X = (test_X_raw - scaler_mean) / scaler_std




## === cell 10
val_y_log_pred = np.matmul(val_X, w)
val_y_pred = np.expm1(val_y_log_pred)  # keep full precision
val_y_pred = np.clip(val_y_pred, 0, None)

from sklearn.metrics import mean_squared_error

rmse = np.sqrt(mean_squared_error(val_y, val_y_pred))
print("Validation RMSE (scaled features):", rmse)

test_y_log_pred = np.matmul(test_X, w)
test_y_pred = np.expm1(test_y_log_pred).round(decimals=2)
test_y_pred = np.clip(test_y_pred, 0, None)

submission = pd.DataFrame(
    {"key": test_df.key, "fare_amount": test_y_pred},
    columns=["key", "fare_amount"],
)
submission.to_csv("submission.csv", index=False)

print("Submission written. Files in current directory:", os.listdir("."))
