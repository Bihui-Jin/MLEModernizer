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

3.9

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

# 5. Code solution

## === cell 0
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import mean_squared_error

train_path = "../input/train.csv"
test_path = "../input/test.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

train_df = train_df.dropna(subset=["fare_amount"]).reset_index(drop=True)
train_df = train_df[train_df["fare_amount"] >= 0].reset_index(drop=True)




## === cell 1
def add_features(df):
    df = df.copy()
    df["abs_diff_longitude"] = (df["dropoff_longitude"] - df["pickup_longitude"]).abs()
    df["abs_diff_latitude"] = (df["dropoff_latitude"] - df["pickup_latitude"]).abs()
    R = 6371.0
    lat1 = np.radians(df["pickup_latitude"])
    lat2 = np.radians(df["dropoff_latitude"])
    dlat = lat2 - lat1
    dlon = np.radians(df["dropoff_longitude"] - df["pickup_longitude"])
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    df["haversine_distance"] = R * 2 * np.arcsin(np.sqrt(a))
    df["log_haversine_distance"] = np.log1p(df["haversine_distance"])
    df["distance_passenger"] = df["haversine_distance"] * df["passenger_count"]
    df["passenger_count_sq"] = df["passenger_count"] ** 2
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], errors="coerce")
    df["pickup_hour"] = df["pickup_datetime"].dt.hour.fillna(-1).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["pickup_hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["pickup_hour"] / 24)
    df = df.drop(columns=["pickup_datetime"])
    return df


train_df = add_features(train_df)
test_df = add_features(test_df)



## === cell 2
feature_cols = [c for c in train_df.columns if c not in ["key", "fare_amount"]]

X = train_df[feature_cols].fillna(0)

scaler = StandardScaler()
X_scaled_array = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled_array, columns=feature_cols, index=train_df.index)

y_log = np.log1p(train_df["fare_amount"])

X_train, X_val, y_train_log, y_val_log = train_test_split(
    X_scaled, y_log, test_size=0.02, random_state=80
)

ridge = Ridge(alpha=1.0, random_state=42)
ridge.fit(X_train, y_train_log)
val_pred_ridge_log = ridge.predict(X_val)
val_pred_ridge = np.expm1(val_pred_ridge_log)
val_true = np.expm1(y_val_log)
val_rmse_ridge = np.sqrt(mean_squared_error(val_true, val_pred_ridge))

linreg = LinearRegression()
linreg.fit(X_train, y_train_log)
val_pred_lr_log = linreg.predict(X_val)
val_pred_lr = np.expm1(val_pred_lr_log)
val_rmse_lr = np.sqrt(mean_squared_error(val_true, val_pred_lr))

print("Validation RMSE Ridge (α=1.0):", val_rmse_ridge)
print("Validation RMSE LinearRegression:", val_rmse_lr)

if val_rmse_ridge < val_rmse_lr:
    model = ridge
    print("Selected model: Ridge")
else:
    model = linreg
    print("Selected model: LinearRegression")

model.fit(X_scaled, y_log)



## === cell 3
test_features = test_df[feature_cols].fillna(0)
test_features_scaled_array = scaler.transform(test_features)
test_features_scaled = pd.DataFrame(
    test_features_scaled_array, columns=feature_cols, index=test_df.index
)

test_pred_log = model.predict(test_features_scaled)
test_pred = np.expm1(test_pred_log)
test_pred = np.clip(test_pred, a_min=0, a_max=None)
test_pred = np.round(test_pred, 2)

submission = pd.DataFrame({"key": test_df["key"], "fare_amount": test_pred})[
    ["key", "fare_amount"]
]



## === cell 4
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission saved to '{submission_path}'")
