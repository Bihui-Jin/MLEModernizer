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

3.42426

# 6. Current score

4.57325

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

- What this solution (achieved 4.57325) has done: 'I fixed the data preprocessing (removed the problematic `key` conversion, parsed the datetime for useful time features, added a haversine distance column, and dropped non‑numeric columns), updated the LightGBM training call to the current API (using callbacks instead of `verbose_eval`), ensured predictions are collected correctly, and finally wrote a proper submission CSV with the required `key` and `fare_amount` columns. These changes resolve the runtime errors and produce a valid submission while keeping the original model logic, which should achieve an RMSE close to the target.'

# 9. Code solution

## === cell 0
import os, numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))




## === cell 1
train_path = "../input/new-york-city-taxi-fare-prediction/train.csv"
test_path = "../input/new-york-city-taxi-fare-prediction/test.csv"
sample_sub_path = "../input/new-york-city-taxi-fare-prediction/sample_submission.csv"

train = pd.read_csv(train_path, nrows=1_000_000)
test = pd.read_csv(test_path)
sample_submission = pd.read_csv(sample_sub_path)




## === cell 2
train.dropna(inplace=True)




## === cell 3
def haversine_np(lon1, lat1, lon2, lat2):
    """Vectorized haversine distance (km)."""
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    km = 6371.0 * 2 * np.arcsin(np.sqrt(a))
    return km


def add_time_features(df):
    dt = pd.to_datetime(df["pickup_datetime"])
    df["pickup_hour"] = dt.dt.hour
    df["pickup_dayofweek"] = dt.dt.dayofweek
    df["pickup_month"] = dt.dt.month
    return df


train = add_time_features(train)
test = add_time_features(test)

train["distance"] = haversine_np(
    train["pickup_longitude"],
    train["pickup_latitude"],
    train["dropoff_longitude"],
    train["dropoff_latitude"],
)
test["distance"] = haversine_np(
    test["pickup_longitude"],
    test["pickup_latitude"],
    test["dropoff_longitude"],
    test["dropoff_latitude"],
)

cols_to_drop = ["key", "pickup_datetime"]
train = train.drop(columns=cols_to_drop)
test = test.drop(columns=cols_to_drop)




## === cell 4
y_train = train["fare_amount"]
X_train = train.drop("fare_amount", axis=1)
X_test = test.copy()  # test has no target column




## === cell 5
cv = KFold(n_splits=5, shuffle=True, random_state=0)

oof_preds = np.zeros(len(X_train))
test_preds = []

params = {
    "objective": "regression",
    "max_bin": 300,
    "learning_rate": 0.05,
    "num_leaves": 40,
    "metric": "rmse",
    "verbosity": -1,
}

for fold, (train_idx, valid_idx) in enumerate(cv.split(X_train, y_train)):
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[valid_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[valid_idx]

    lgb_train = lgb.Dataset(X_tr, y_tr)
    lgb_valid = lgb.Dataset(X_val, y_val, reference=lgb_train)

    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=1000,
        valid_sets=[lgb_train, lgb_valid],
        callbacks=[
            lgb.log_evaluation(period=10),
            lgb.early_stopping(stopping_rounds=10, verbose=False),
        ],
    )

    oof_preds[valid_idx] = model.predict(X_val, num_iteration=model.best_iteration)
    test_preds.append(model.predict(X_test, num_iteration=model.best_iteration))




## === cell 6
rmse = np.sqrt(mean_squared_error(y_train, oof_preds))
print(f"OOF RMSE: {rmse:.5f}")

pd.DataFrame(oof_preds, columns=["oof_pred"]).to_csv("oof_train_lgb.csv", index=False)




## === cell 7
if len(test_preds) > 0:
    final_test_pred = np.mean(test_preds, axis=0)
else:
    final_test_pred = np.full(len(X_test), y_train.mean())

submission = sample_submission.copy()
submission["fare_amount"] = final_test_pred
submission_path = "submission_lightgbm.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
