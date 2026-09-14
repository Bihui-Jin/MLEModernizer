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
sklearn-pandas==2.2.0
xgboost==2.0.3

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

4.00706

# 6. Current score

5.03799

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 4.77089) has done: 'I fix the target‑variable handling (use the fare_amount column as Y_train instead of an empty DataFrame), correct the XGBoost predict call, and build the submission file with the required key and fare_amount columns. These changes resolve the runtime errors and ensure a valid CSV is produced while keeping the original model logic unchanged.'
- What this solution (achieved 4.77589) has done: 'The changes add a more informative distance feature (haversine distance) and tune the XGBoost hyper‑parameters, which are small adjustments that keep the original model structure while improving predictive performance and moving the RMSE closer to the target.'
- What this solution (achieved 4.69511) has done: 'I add a few lightweight features (month and weekend flag) and switch to a train‑validation split with early‑stopping so the XGBoost model can stop at the optimal number of trees. This keeps the original model architecture while giving a modest boost in predictive power, which should bring the RMSE closer to the target 4.00706 without over‑hauling the pipeline. The script now writes a proper sample_submission.csv as before.'
- What this solution (achieved 5.55291) has done: 'The updates keep the same feature engineering and XGBoost model, but they dramatically cut the amount of data that needs to be processed. After loading the full CSV we immediately down‑sample the training rows (keeping a representative random subset) and use a smaller validation split, which reduces both memory use and the number of boosting iterations over huge arrays. All other steps—including dtype handling, feature creation, and the exact XGBoost hyper‑parameters—remain unchanged, so the prediction logic and accuracy are preserved while the runtime falls well below the 600‑second limit.'
- What this solution (achieved 4.80725) has done: 'Add a modest increase in training data size (sample 20 % instead of 10 %) to give the model more information, and allow a slightly longer early‑stopping window (150 rounds) so the booster can converge a bit further. These tiny adjustments keep the original pipeline intact while likely lowering the RMSE toward the target value.'
- What this solution (achieved 5.03799) has done: 'I slightly increase the sampled training size from 20 % to 30 % to give the model more data, add cyclical hour/month/day‑of‑week features (sin/cos) to capture periodic patterns, and give the XGBoost model a bit more capacity (max_depth 9, n_estimators 2000) with a longer early‑stopping patience. These small, targeted changes keep the original pipeline intact while encouraging a lower RMSE, moving the score toward the target 4.00706.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O
import os
from sklearn.model_selection import train_test_split

print(os.listdir("../input"))




## === cell 1
dtype_map = {
    "fare_amount": np.float32,
    "pickup_longitude": np.float32,
    "pickup_latitude": np.float32,
    "dropoff_longitude": np.float32,
    "dropoff_latitude": np.float32,
    "passenger_count": np.int8,
}
training_data = pd.read_csv(
    "../input/train.csv",
    dtype=dtype_map,
    usecols=[
        "fare_amount",
        "pickup_datetime",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
    ],
    low_memory=False,
)
training_data = training_data.sample(frac=0.30, random_state=42).reset_index(drop=True)

test_data = pd.read_csv(
    "../input/test.csv",
    dtype=dtype_map,
    usecols=[
        "key",
        "pickup_datetime",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
    ],
    low_memory=False,
)




## === cell 2
training_data




## === cell 3
Y_train = training_data["fare_amount"]
X_train = training_data.drop(columns=["fare_amount"])

X_test = test_data.copy()




## === cell 4
X_train["pickup_datetime"] = pd.to_datetime(X_train["pickup_datetime"])
X_train["hour"] = X_train["pickup_datetime"].dt.hour.astype(np.int8)
X_train["dayofweek"] = X_train["pickup_datetime"].dt.dayofweek.astype(np.int8)
X_train["month"] = X_train["pickup_datetime"].dt.month.astype(np.int8)
X_train["is_weekend"] = (X_train["dayofweek"] >= 5).astype(np.int8)
X_train["is_night"] = (
    X_train["hour"].isin([0, 1, 2, 3, 4, 5, 20, 21, 22, 23]).astype(np.int8)
)

X_train["latitude_distance"] = np.abs(
    X_train["dropoff_latitude"] - X_train["pickup_latitude"]
).astype(np.float32)
X_train["longitude_distance"] = np.abs(
    X_train["dropoff_longitude"] - X_train["pickup_longitude"]
).astype(np.float32)

X_train["pickup_lat_lon"] = (
    X_train["pickup_latitude"] * X_train["pickup_longitude"]
).astype(np.float32)
X_train["dropoff_lat_lon"] = (
    X_train["dropoff_latitude"] * X_train["dropoff_longitude"]
).astype(np.float32)

X_train["hour_sin"] = np.sin(2 * np.pi * X_train["hour"] / 24).astype(np.float32)
X_train["hour_cos"] = np.cos(2 * np.pi * X_train["hour"] / 24).astype(np.float32)
X_train["month_sin"] = np.sin(2 * np.pi * X_train["month"] / 12).astype(np.float32)
X_train["month_cos"] = np.cos(2 * np.pi * X_train["month"] / 12).astype(np.float32)
X_train["dow_sin"] = np.sin(2 * np.pi * X_train["dayofweek"] / 7).astype(np.float32)
X_train["dow_cos"] = np.cos(2 * np.pi * X_train["dayofweek"] / 7).astype(np.float32)

R = 6371.0
lat1 = np.radians(X_train["pickup_latitude"])
lat2 = np.radians(X_train["dropoff_latitude"])
dlat = lat2 - lat1
dlon = np.radians(X_train["dropoff_longitude"] - X_train["pickup_longitude"])
a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
c = 2 * np.arcsin(np.sqrt(a))
X_train["haversine"] = (R * c).astype(np.float32)

X_train = X_train.drop(
    columns=[
        "dropoff_longitude",
        "dropoff_latitude",
        "pickup_datetime",
    ]
)




## === cell 5
X_test["pickup_datetime"] = pd.to_datetime(X_test["pickup_datetime"])
X_test["hour"] = X_test["pickup_datetime"].dt.hour.astype(np.int8)
X_test["dayofweek"] = X_test["pickup_datetime"].dt.dayofweek.astype(np.int8)
X_test["month"] = X_test["pickup_datetime"].dt.month.astype(np.int8)
X_test["is_weekend"] = (X_test["dayofweek"] >= 5).astype(np.int8)
X_test["is_night"] = (
    X_test["hour"].isin([0, 1, 2, 3, 4, 5, 20, 21, 22, 23]).astype(np.int8)
)

X_test["latitude_distance"] = np.abs(
    X_test["dropoff_latitude"] - X_test["pickup_latitude"]
).astype(np.float32)
X_test["longitude_distance"] = np.abs(
    X_test["dropoff_longitude"] - X_test["pickup_longitude"]
).astype(np.float32)

X_test["pickup_lat_lon"] = (
    X_test["pickup_latitude"] * X_test["pickup_longitude"]
).astype(np.float32)
X_test["dropoff_lat_lon"] = (
    X_test["dropoff_latitude"] * X_test["dropoff_longitude"]
).astype(np.float32)

X_test["hour_sin"] = np.sin(2 * np.pi * X_test["hour"] / 24).astype(np.float32)
X_test["hour_cos"] = np.cos(2 * np.pi * X_test["hour"] / 24).astype(np.float32)
X_test["month_sin"] = np.sin(2 * np.pi * X_test["month"] / 12).astype(np.float32)
X_test["month_cos"] = np.cos(2 * np.pi * X_test["month"] / 12).astype(np.float32)
X_test["dow_sin"] = np.sin(2 * np.pi * X_test["dayofweek"] / 7).astype(np.float32)
X_test["dow_cos"] = np.cos(2 * np.pi * X_test["dayofweek"] / 7).astype(np.float32)

lat1_t = np.radians(X_test["pickup_latitude"])
lat2_t = np.radians(X_test["dropoff_latitude"])
dlat_t = lat2_t - lat1_t
dlon_t = np.radians(X_test["dropoff_longitude"] - X_test["pickup_longitude"])
a_t = (
    np.sin(dlat_t / 2) ** 2 + np.cos(lat1_t) * np.cos(lat2_t) * np.sin(dlon_t / 2) ** 2
)
c_t = 2 * np.arcsin(np.sqrt(a_t))
X_test["haversine"] = (R * c_t).astype(np.float32)

X_test = X_test.drop(
    columns=["dropoff_longitude", "dropoff_latitude", "key", "pickup_datetime"]
)




## === cell 6
"""
# Optional neural‑network baseline (commented out to avoid heavy dependencies)
from keras import models, layers, optimizers

model_nn = models.Sequential([
    layers.Dense(512, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dropout(0.2),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(1)
])

rmsprop = optimizers.RMSprop(lr=0.001)
model_nn.compile(optimizer=rmsprop, loss='mse', metrics=['mae'])
model_nn.fit(X_train, Y_train, epochs=4, batch_size=512)
Y_pred = model_nn.predict(X_test).flatten()
"""




## === cell 7
import xgboost as xgb

X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, Y_train, test_size=0.05, random_state=42
)

model = xgb.XGBRegressor(
    objective="reg:squarederror",
    eval_metric="rmse",
    n_estimators=2000,
    max_depth=9,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    n_jobs=4,
    random_state=42,
    tree_method="hist",
    verbosity=0,
)

model.fit(
    X_tr,
    y_tr,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=200,
    verbose=False,
)

Y_pred = model.predict(X_test)




## === cell 8
submission = test_data[["key"]].copy()
submission["fare_amount"] = np.clip(Y_pred, a_min=0, a_max=None)




## === cell 9
submission.to_csv("sample_submission.csv", index=False)
print("Submission saved to sample_submission.csv")
