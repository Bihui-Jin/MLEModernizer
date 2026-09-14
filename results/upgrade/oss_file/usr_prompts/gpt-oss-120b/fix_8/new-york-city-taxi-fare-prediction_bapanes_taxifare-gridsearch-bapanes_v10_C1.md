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

No external packages required in the script and installed.

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

3.830643716740832

# 6. Current score

4.63208

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 10.02912) has done: 'The script was failing because it referenced non‑existent data files, used an incompatible keras wrapper, and tried to load the whole 55 M‑row dataset. I replaced the broken imports, pointed the loaders to the real CSV paths, limited the training set to a manageable size, added sensible feature engineering (datetime splits and haversine distance), standardized the data, built the same three‑layer neural net with tf.keras, trained it with early stopping, and finally generated a correctly‑named submission_file.csv containing the required `key` and `fare_amount` columns. This fixes all runtime errors and produces a valid submission while keeping the original model architecture.'
- What this solution (achieved 4.51984) has done: 'The fix adds a median imputer to remove NaNs from the feature matrix, updates the scaling to use the imputed data, and applies the same imputer to the test set before scaling. This resolves the RandomForest “contains NaN” error, allows the model to be fitted, and ensures `test_pred` is defined so the submission CSV is correctly written.'
- What this solution (achieved 4.61403) has done: 'The fix drops rows with missing or non‑positive fare amounts before the log‑transform, preventing NaNs in the target vector that caused the RandomForest to fail. The data loading, feature engineering, imputation, scaling, model training, prediction, and submission steps remain unchanged, and the script now writes a correctly‑named `submission_file.csv` containing the required `key` and `fare_amount` columns.'
- What this solution (achieved 4.63208) has done: 'The changes focus on cutting down unnecessary I/O and reducing the RandomForest workload, which are the main sources of slowdown. We read only the columns required for training / prediction, avoiding the large unused “key” column, and we lower the number of trees from 800 to 200 while keeping the same model type and random seed. These adjustments keep the overall logic, feature engineering, and evaluation unchanged, but make the script finish well within the 600‑second limit.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer

print("Environment ready for training with scikit-learn")




## === cell 1
def haversine_distance(lat1, lon1, lat2, lon2):
    """Vectorised haversine distance in kilometers."""
    R = 6371.0  # Earth radius in km
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    dlat = lat2_rad - lat1_rad
    dlon = np.radians(lon2 - lon1)

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c




## === cell 2
def find_file(name):
    possible = [
        os.path.join("..", "input", name),
        os.path.join("/kaggle", "input", name),
        name,
    ]
    for p in possible:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"{name} not found in any expected location")


train_path = find_file("train.csv")
test_path = find_file("test.csv")

train_usecols = [
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
df_train = pd.read_csv(train_path, usecols=train_usecols, nrows=500000)

df_train = df_train.dropna(subset=["fare_amount"])
df_train = df_train[df_train["fare_amount"] > 0]

df_train["pickup_datetime"] = pd.to_datetime(df_train["pickup_datetime"])
df_train["hour"] = df_train["pickup_datetime"].dt.hour
df_train["dayofweek"] = df_train["pickup_datetime"].dt.dayofweek
df_train["month"] = df_train["pickup_datetime"].dt.month

df_train["distance"] = haversine_distance(
    df_train["pickup_latitude"],
    df_train["pickup_longitude"],
    df_train["dropoff_latitude"],
    df_train["dropoff_longitude"],
)

feature_cols = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
    "distance",
    "hour",
    "dayofweek",
    "month",
]

X = df_train[feature_cols].values.astype(np.float32)
y = np.log1p(df_train["fare_amount"].values.astype(np.float32))

imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X)

X_train, X_val, y_train, y_val = train_test_split(
    X_imputed, y, test_size=0.2, random_state=42
)




## === cell 3
def build_model():
    model = RandomForestRegressor(
        n_estimators=200,  # lowered from 800 for runtime constraints
        max_depth=None,
        n_jobs=-1,
        random_state=42,
    )
    return model


model = build_model()
model.fit(X_train, y_train)

val_pred_log = model.predict(X_val)
val_pred = np.expm1(val_pred_log)
val_pred = np.clip(val_pred, 0, None)  # fares cannot be negative
val_rmse = np.sqrt(mean_squared_error(np.expm1(y_val), val_pred))
print(f"Validation RMSE: {val_rmse:.4f}")



## === cell 4
test_usecols = [
    "key",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
df_test = pd.read_csv(test_path, usecols=test_usecols)

df_test["pickup_datetime"] = pd.to_datetime(df_test["pickup_datetime"])
df_test["hour"] = df_test["pickup_datetime"].dt.hour
df_test["dayofweek"] = df_test["pickup_datetime"].dt.dayofweek
df_test["month"] = df_test["pickup_datetime"].dt.month

df_test["distance"] = haversine_distance(
    df_test["pickup_latitude"],
    df_test["pickup_longitude"],
    df_test["dropoff_latitude"],
    df_test["dropoff_longitude"],
)

X_test = df_test[feature_cols].values.astype(np.float32)
X_test_imputed = imputer.transform(X_test)  # same imputer as training

test_pred_log = model.predict(X_test_imputed)
test_pred = np.expm1(test_pred_log)
test_pred = np.clip(test_pred, 0, None)



## === cell 5
submission_path = "submission_file.csv"
df_output = pd.DataFrame({"key": df_test["key"], "fare_amount": test_pred})
df_output.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
