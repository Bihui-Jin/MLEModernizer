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
lightgbm==4.6.0
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
scipy==1.15.3
sklearn-pandas==2.2.0
tqdm==4.67.1

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

3.3504463754491907

# 6. Current score

4.19796

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 5.35913) has done: 'I fix the LightGBM training call, which no longer accepts `early_stopping_rounds`. I replace it with the proper callback‑based early stopping and logging, ensuring the model variable is created for the prediction step. This resolves the TypeError and lets the script finish, write a valid `submission_lgb.csv` file, and compute the validation RMSE.'
- What this solution (achieved 4.19755) has done: 'I add useful geographic and temporal numeric features (raw coordinates, Manhattan distance, hour) to the model and modestly adjust the LightGBM learning rate and boost rounds so the model can better capture patterns, which should lower the RMSE toward the target while keeping the overall pipeline unchanged.'
- What this solution (achieved 4.19796) has done: 'I add the missing `day_of_week` feature to the one‑hot encoded categorical set and slightly boost the model capacity (more leaves/depth) while using a lower learning rate. These small adjustments should improve the LightGBM fit and move the validation RMSE closer to the target without altering the overall pipeline.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import csr_matrix, hstack
from sklearn.impute import SimpleImputer
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from math import sqrt

print("Input folder contents:", os.listdir("../input"))




## === cell 1
train_path = "../input/new-york-city-taxi-fare-prediction/train.csv"
test_path = "../input/new-york-city-taxi-fare-prediction/test.csv"

train = pd.read_csv(train_path, nrows=15_000_000)
test = pd.read_csv(test_path)

test_id = test["key"].values.copy()




## === cell 2
for df in (train, test):
    df["pickup_datetime"] = pd.to_datetime(
        df["pickup_datetime"].str.slice(0, 16), utc=True, format="%Y-%m-%d %H:%M"
    )

train.drop(columns="key", inplace=True)
test.drop(columns="key", inplace=True)

train["passenger_count"] = train["passenger_count"].astype("uint8")
test["passenger_count"] = test["passenger_count"].astype("uint8")

float_cols = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "fare_amount",
]
for col in float_cols:
    if col in train.columns:
        train[col] = train[col].astype("float32")
    if col in test.columns:
        test[col] = test[col].astype("float32")

bounds = {
    "pickup_longitude": (
        test["pickup_longitude"].min(),
        test["pickup_longitude"].max(),
    ),
    "pickup_latitude": (test["pickup_latitude"].min(), test["pickup_latitude"].max()),
    "dropoff_longitude": (
        test["dropoff_longitude"].min(),
        test["dropoff_longitude"].max(),
    ),
    "dropoff_latitude": (
        test["dropoff_latitude"].min(),
        test["dropoff_latitude"].max(),
    ),
}
train = train[
    (train["pickup_longitude"].between(*bounds["pickup_longitude"]))
    & (train["pickup_latitude"].between(*bounds["pickup_latitude"]))
    & (train["dropoff_longitude"].between(*bounds["dropoff_longitude"]))
    & (train["dropoff_latitude"].between(*bounds["dropoff_latitude"]))
].copy()

for df in (train, test):
    df["hour"] = df["pickup_datetime"].dt.hour.astype("uint8")
    df["month"] = df["pickup_datetime"].dt.month.astype("uint8")
    df["day_of_week"] = df["pickup_datetime"].dt.dayofweek.astype("uint8")
    df["year"] = df["pickup_datetime"].dt.year.astype("uint16")
    df["day_hour"] = (
        df["day_of_week"].astype(str) + "_" + df["hour"].astype(str)
    ).astype("category")


def haversine_distance(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return 6371.01 * c


train["distance"] = haversine_distance(
    train["pickup_latitude"],
    train["pickup_longitude"],
    train["dropoff_latitude"],
    train["dropoff_longitude"],
).astype("float32")
test["distance"] = haversine_distance(
    test["pickup_latitude"],
    test["pickup_longitude"],
    test["dropoff_latitude"],
    test["dropoff_longitude"],
).astype("float32")

train["manhattan_distance"] = (
    np.abs(train["pickup_latitude"] - train["dropoff_latitude"])
    + np.abs(train["pickup_longitude"] - train["dropoff_longitude"])
).astype("float32")
test["manhattan_distance"] = (
    np.abs(test["pickup_latitude"] - test["dropoff_latitude"])
    + np.abs(test["pickup_longitude"] - test["dropoff_longitude"])
).astype("float32")




## === cell 3
categorical_cols = ["day_hour", "month", "year", "passenger_count", "day_of_week"]
numerical_cols = [
    "distance",
    "manhattan_distance",
    "hour",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
]

X_cats_train = train[categorical_cols].values
X_cats_test = test[categorical_cols].values
X_cats_full = np.concatenate([X_cats_train, X_cats_test], axis=0)

ohe = OneHotEncoder(handle_unknown="ignore")
X_onehot_full = ohe.fit_transform(X_cats_full)

X_num_full = np.concatenate(
    [train[numerical_cols].values, test[numerical_cols].values], axis=0
)
X_num_sparse = csr_matrix(X_num_full)

X_full = hstack([X_onehot_full, X_num_sparse]).tocsr()

imputer = SimpleImputer()
X_full = imputer.fit_transform(X_full)

n_train = train.shape[0]
X = X_full[:n_train, :]
X_test_final = X_full[n_train:, :]

y = train["fare_amount"].values




## === cell 4
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting": "gbdt",
    "num_leaves": 80,  # increased capacity
    "max_depth": 10,  # increased depth
    "learning_rate": 0.05,  # slower learning for better generalisation
    "bagging_fraction": 0.8,
    "feature_fraction": 0.8,
    "min_split_gain": 0.02,
    "min_child_samples": 10,
    "min_child_weight": 0.02,
    "lambda_l2": 0.0475,
    "verbosity": -1,
    "seed": 17,
}

d_train = lgb.Dataset(X_train, label=y_train)
d_val = lgb.Dataset(X_val, label=y_val, reference=d_train)

callbacks = [
    lgb.early_stopping(stopping_rounds=50, verbose=False),
    lgb.log_evaluation(period=100),
]

model = lgb.train(
    lgb_params,
    d_train,
    num_boost_round=2000,  # more rounds; early stopping will limit actual number
    valid_sets=[d_train, d_val],
    callbacks=callbacks,
)

val_pred = model.predict(X_val, num_iteration=model.best_iteration)
print("Validation RMSE :", sqrt(mean_squared_error(y_val, val_pred)))




## === cell 5
test_pred = model.predict(X_test_final, num_iteration=model.best_iteration)

test_pred = np.where(test_pred > 0, test_pred, 0.0)

submission = pd.DataFrame({"key": test_id, "fare_amount": test_pred})

submission_path = "submission_lgb.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
