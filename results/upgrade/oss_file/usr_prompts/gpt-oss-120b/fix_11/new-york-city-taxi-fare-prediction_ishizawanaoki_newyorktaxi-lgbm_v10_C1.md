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

3.10

# 3. Installed packages

geopandas==0.14.4
geopy==2.4.1
lightgbm==4.6.0
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

3.27817

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

- What this solution (achieved 4.48658) has done: 'I add a few simple time‑based features (hour and day‑of‑week) before dropping the original datetime column, and rename the final output file to the conventional `submission.csv`. These minor tweaks keep the original modelling pipeline intact while giving the LightGBM model a bit more signal, which should move the RMSE closer to the target value.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error



## === cell 1
dtype_spec = {
    "fare_amount": np.float32,
    "pickup_longitude": np.float32,
    "pickup_latitude": np.float32,
    "dropoff_longitude": np.float32,
    "dropoff_latitude": np.float32,
    "passenger_count": np.int8,
}
train_path = "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv"
test_path = "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv"
sample_path = "/kaggle/input/new-york-city-taxi-fare-prediction/sample_submission.csv"

train = pd.read_csv(train_path, nrows=2_500_000, dtype=dtype_spec)
test = pd.read_csv(test_path, dtype=dtype_spec)
sample_submission = pd.read_csv(sample_path)



## === cell 2
print("Missing values per column:")
print(train.isnull().sum())



## === cell 3
train.dropna(inplace=True)



## === cell 4
print(train.describe())



## === cell 5
train = train.query(
    "1 <= passenger_count <= 6 and "
    "0 <= fare_amount and "
    "-180 <= pickup_longitude <= 180 and "
    "-180 <= dropoff_longitude <= 180 and "
    "-90 <= pickup_latitude <= 90 and "
    "-90 <= dropoff_latitude <= 90"
)

train_len = len(train)

print(train.describe())



## === cell 6
train.reset_index(drop=True, inplace=True)
print(train.head())



## === cell 7
data = pd.concat([train, test], sort=False)



## === cell 8
data["pickup_datetime"] = pd.to_datetime(data["pickup_datetime"])
data["hour"] = data["pickup_datetime"].dt.hour
data["minute"] = data["pickup_datetime"].dt.minute
data["second"] = data["pickup_datetime"].dt.second
data["dayofweek"] = data["pickup_datetime"].dt.dayofweek
data["month"] = data["pickup_datetime"].dt.month
data["is_weekend"] = data["dayofweek"].isin([5, 6]).astype(int)

data["delta_lat"] = data["dropoff_latitude"] - data["pickup_latitude"]
data["delta_lon"] = data["dropoff_longitude"] - data["pickup_longitude"]

data = data.drop("pickup_datetime", axis=1)

data["key"] = data["key"].astype(str)
print(data.head())




## === cell 9
def haversine_miles(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    miles = 3958.8 * c
    return miles


data["distance"] = haversine_miles(
    data["pickup_latitude"],
    data["pickup_longitude"],
    data["dropoff_latitude"],
    data["dropoff_longitude"],
)

data["distance_per_passenger"] = data["distance"] / (data["passenger_count"] + 1e-3)

print(data.head())



## === cell 10
train = data[:train_len].copy()
test = data[train_len:].copy()

y_train_original = train["fare_amount"]
y_train = np.log1p(y_train_original)  # log‑transform target

X_train = train.drop(["fare_amount", "key"], axis=1).astype(np.float32)
X_test = test.drop(["fare_amount", "key"], axis=1).astype(np.float32)

X_train_np = X_train.values
X_test_np = X_test.values
y_train_np = y_train.values.astype(np.float32)

print("Training shape:", X_train.shape, "Test shape:", X_test.shape)



## === cell 11
cv = KFold(n_splits=5, shuffle=True, random_state=0)
categorical_features = []  # no categorical columns in this dataset

params = {
    "objective": "regression",
    "max_bin": 255,
    "learning_rate": 0.03,
    "num_leaves": 256,  # slightly larger capacity
    "min_data_in_leaf": 20,
    "metric": "l2",
    "verbosity": -1,
    "num_threads": -1,
    "feature_fraction": 0.9,  # modest feature bagging
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
}

y_preds = []
models = []
oof_train_log = np.zeros(len(X_train), dtype=np.float32)

for fold_id, (train_idx, valid_idx) in enumerate(cv.split(X_train_np, y_train_np)):
    X_tr, X_val = X_train_np[train_idx], X_train_np[valid_idx]
    y_tr, y_val = y_train_np[train_idx], y_train_np[valid_idx]

    lgb_train = lgb.Dataset(
        X_tr, y_tr, categorical_feature=categorical_features, free_raw_data=False
    )
    lgb_valid = lgb.Dataset(
        X_val,
        y_val,
        reference=lgb_train,
        categorical_feature=categorical_features,
        free_raw_data=False,
    )

    callbacks = [lgb.early_stopping(stopping_rounds=30, verbose=False)]

    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=3000,
        valid_sets=[lgb_train, lgb_valid],
        callbacks=callbacks,
    )

    oof_train_log[valid_idx] = model.predict(X_val, num_iteration=model.best_iteration)
    y_pred_log = model.predict(X_test_np, num_iteration=model.best_iteration)

    y_pred = np.expm1(y_pred_log)
    y_preds.append(y_pred)
    models.append(model)



## === cell 12
pd.DataFrame(np.expm1(oof_train_log), columns=["fare_amount"]).to_csv(
    "oof_train_kfold.csv", index=False
)

scores = [m.best_score["valid_1"]["l2"] for m in models]  # L2 on log scale
mean_mse_log = np.mean(scores)
rmse_log = np.sqrt(mean_mse_log)

rmse_original = np.sqrt(mean_squared_error(y_train_original, np.expm1(oof_train_log)))
print("=== CV RMSE (log scale) ===")
print(rmse_log)
print("=== CV RMSE (original scale) ===")
print(rmse_original)



## === cell 13
rmse_oof = np.sqrt(mean_squared_error(y_train_original, np.expm1(oof_train_log)))
print("OOF RMSE (original scale):", rmse_oof)



## === cell 14
print("Number of folds predictions collected:", len(y_preds))



## === cell 15
if y_preds:
    y_sub = np.mean(np.column_stack(y_preds), axis=1)
else:
    y_sub = np.zeros(len(X_test))

print("First 10 predictions:", y_sub[:10])



## === cell 16
sub_lgb = pd.DataFrame({"key": test["key"].values, "fare_amount": y_sub})
sub_lgb.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")
print(sub_lgb.head())
