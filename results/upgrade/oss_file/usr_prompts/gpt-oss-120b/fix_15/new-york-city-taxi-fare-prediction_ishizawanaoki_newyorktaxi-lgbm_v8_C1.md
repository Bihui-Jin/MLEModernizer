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

3.40781

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

- What this solution (achieved 4.55238) has done: 'I add a log‑transform of the fare amount so the model works on a more stable target distribution and then revert the predictions back with exp‑1. This small change often lowers RMSE without altering the core architecture or training loop, and keeps the script end‑to‑end while still writing a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import lightgbm as lgb

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))




## === cell 1
base_path = "/kaggle/input/new-york-city-taxi-fare-prediction"
train = pd.read_csv(os.path.join(base_path, "train.csv"), nrows=2_000_000)
test = pd.read_csv(os.path.join(base_path, "test.csv"))
sample_submission = pd.read_csv(os.path.join(base_path, "sample_submission.csv"))

for df in [train, test]:
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
    df["pickup_hour"] = df["pickup_datetime"].dt.hour
    df["pickup_weekday"] = df["pickup_datetime"].dt.weekday




## === cell 2
train.dropna(inplace=True)




## === cell 3
train = train.query("1 <= passenger_count <= 6 and fare_amount >= 0")




## === cell 4
train.reset_index(drop=True, inplace=True)




## === cell 5
orig_train_len = len(train)




## === cell 6
data = pd.concat([train, test], sort=False)




## === cell 7
def haversine_distance(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6371.0 * c  # Earth radius in kilometers
    return km


data["distance"] = haversine_distance(
    data["pickup_latitude"],
    data["pickup_longitude"],
    data["dropoff_latitude"],
    data["dropoff_longitude"],
)

data["pickup_month"] = data["pickup_datetime"].dt.month
data["pickup_day"] = data["pickup_datetime"].dt.day

data["month_sin"] = np.sin(2 * np.pi * data["pickup_month"] / 12.0)
data["month_cos"] = np.cos(2 * np.pi * data["pickup_month"] / 12.0)
data["day_sin"] = np.sin(2 * np.pi * data["pickup_day"] / 31.0)
data["day_cos"] = np.cos(2 * np.pi * data["pickup_day"] / 31.0)

data = data.drop("pickup_datetime", axis=1)

data["hour_sin"] = np.sin(2 * np.pi * data["pickup_hour"] / 24.0)
data["hour_cos"] = np.cos(2 * np.pi * data["pickup_hour"] / 24.0)

data["weekday_sin"] = np.sin(2 * np.pi * data["pickup_weekday"] / 7.0)
data["weekday_cos"] = np.cos(2 * np.pi * data["pickup_weekday"] / 7.0)




## === cell 8
train = data.iloc[:orig_train_len].copy()
test = data.iloc[orig_train_len:].copy()




## === cell 9
y_train = train["fare_amount"]
y_train_log = np.log1p(y_train)  # log‑transform for stable learning
X_train = train.drop(columns=["fare_amount", "key"])
X_test = test.drop(columns=["key", "fare_amount"], errors="ignore")




## === cell 10
oof_train = np.zeros(len(X_train))
y_preds = []
models = []
cv = KFold(n_splits=5, shuffle=True, random_state=0)
categorical_features = []  # no categorical columns in this feature set

params = {
    "objective": "regression",
    "metric": "rmse",  # use RMSE directly for early stopping
    "max_bin": 500,
    "learning_rate": 0.05,
    "num_leaves": 128,  # increased capacity
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "verbosity": -1,
}

for fold_id, (train_idx, valid_idx) in enumerate(cv.split(X_train, y_train)):
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[valid_idx]
    y_tr_log = y_train_log.iloc[train_idx]
    y_val_log = y_train_log.iloc[valid_idx]

    lgb_train = lgb.Dataset(X_tr, y_tr_log, categorical_feature=categorical_features)
    lgb_valid = lgb.Dataset(
        X_val, y_val_log, categorical_feature=categorical_features, reference=lgb_train
    )

    callbacks = [
        lgb.early_stopping(stopping_rounds=30, verbose=False),
        lgb.log_evaluation(period=10),
    ]

    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=2000,  # allow more iterations if beneficial
        valid_sets=[lgb_train, lgb_valid],
        callbacks=callbacks,
    )

    oof_train[valid_idx] = np.expm1(
        model.predict(X_val, num_iteration=model.best_iteration)
    )
    y_pred = np.expm1(model.predict(X_test, num_iteration=model.best_iteration))

    y_preds.append(y_pred)
    models.append(model)




## === cell 11
pd.DataFrame(oof_train, columns=["oof_pred"]).to_csv("oof_train_kfold.csv", index=False)

if models:
    scores = [m.best_score["valid_1"]["rmse"] for m in models]
    cv_score = np.mean(scores)
    print("=== CV RMSE scores ===")
    print(scores)
    print("Mean CV RMSE:", cv_score)
else:
    print("No models were trained.")




## === cell 12
rmse = np.sqrt(mean_squared_error(y_train, oof_train))
print("OOF RMSE:", rmse)




## === cell 13
if y_preds:
    y_sub = np.mean(y_preds, axis=0)
else:
    y_sub = np.full(len(X_test), y_train.mean())




## === cell 14
submission_path = "/kaggle/working/submission.csv"
submission = pd.DataFrame({"key": test["key"], "fare_amount": y_sub})
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")
