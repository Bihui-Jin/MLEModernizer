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

4.24594

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 4.38139) has done: 'I fix the pipeline break in feature engineering by keeping `key` as a string ID (it must not be coerced to float) and only cleaning it in a non-destructive way. I update the LightGBM training call to be compatible with lightgbm==4.6.0 by replacing deprecated `verbose_eval`/`early_stopping_rounds` with the proper callback API, so training completes and produces predictions. I also make the distance feature computation vectorized (same feature, much faster) so it finishes within Kaggle’s time limits. Finally, I ensure the submission is written as a valid `.csv` with exactly `key` and `fare_amount` aligned to the test set.'
- What this solution (achieved 4.24594) has done: 'You’re currently far worse than the target (RMSE 4.38 vs 3.28, lower is better), so we should make the smallest changes that are likely to improve generalization without changing the core LightGBM approach. The biggest issue is that `key_clean` injects a high-cardinality, mostly-random identifier feature that encourages memorization in CV but hurts the true test score; dropping it is a minimal, high-impact fix. Next, dropping `pickup_datetime` loses strong signal; we add a few simple datetime-derived features (hour, dayofweek, month, year) while still using the same model/training loop. Finally, we use a slightly more appropriate LightGBM metric (`rmse`) for early stopping (objective stays regression), which aligns training selection with the competition RMSE without changing semantics.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os

INPUT_DIR = "/kaggle/input/new-york-city-taxi-fare-prediction"

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames[:50]:
        print(os.path.join(dirname, filename))



## === cell 1
train = pd.read_csv(f"{INPUT_DIR}/train.csv", nrows=1_000_000)
test = pd.read_csv(f"{INPUT_DIR}/test.csv")
sample_submission = pd.read_csv(f"{INPUT_DIR}/sample_submission.csv")

print(train.shape, test.shape, sample_submission.shape)
train.head()



## === cell 2
train.isnull().sum()



## === cell 3
train.dropna(inplace=True)



## === cell 4
train.describe()



## === cell 5
train.query("passenger_count > 6").head()



## === cell 6
train.query("passenger_count < 1").head()



## === cell 7
train.query("fare_amount < 0").head()



## === cell 8
train.query("pickup_longitude < -180 or pickup_longitude > 180").head()



## === cell 9
train.query("dropoff_longitude < -180 or dropoff_longitude > 180").head()



## === cell 10
train.query("pickup_latitude < -90 or pickup_latitude > 90").head()



## === cell 11
train.query("dropoff_latitude < -90 or dropoff_latitude > 90").head()



## === cell 12
train = train.query(
    "1 <= passenger_count <= 6 and "
    "0 <= fare_amount and "
    "-180 <= pickup_longitude <= 180 and "
    "-180 <= dropoff_longitude <= 180 and "
    "-90 <= pickup_latitude <= 90 and "
    "-90 <= dropoff_latitude <= 90"
)
train.describe()



## === cell 13
train.reset_index(drop=True, inplace=True)
train.head()



## === cell 14
data = pd.concat([train, test], sort=False, ignore_index=True)
data.head()



## === cell 15
data["key"] = data["key"].astype(str)

data["pickup_datetime"] = pd.to_datetime(
    data["pickup_datetime"], errors="coerce", utc=True
)
data["pickup_year"] = data["pickup_datetime"].dt.year.fillna(0).astype(np.int16)
data["pickup_month"] = data["pickup_datetime"].dt.month.fillna(0).astype(np.int8)
data["pickup_dayofweek"] = (
    data["pickup_datetime"].dt.dayofweek.fillna(0).astype(np.int8)
)
data["pickup_hour"] = data["pickup_datetime"].dt.hour.fillna(0).astype(np.int8)

data = data.drop("pickup_datetime", axis=1)

data.head()




## === cell 16
def haversine_miles(lat1, lon1, lat2, lon2):
    R = 3958.7613  # Earth radius in miles
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * (
        np.sin(dlon / 2.0) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


data["distance"] = haversine_miles(
    data["pickup_latitude"].to_numpy(),
    data["pickup_longitude"].to_numpy(),
    data["dropoff_latitude"].to_numpy(),
    data["dropoff_longitude"].to_numpy(),
)

data.head()



## === cell 17
train_fe = data.iloc[: len(train)].copy()
test_fe = data.iloc[len(train) :].copy()

y_train = train_fe["fare_amount"].astype(float)
X_train = train_fe.drop("fare_amount", axis=1)
X_test = test_fe.drop("fare_amount", axis=1, errors="ignore")

test_keys = test_fe["key"].astype(str).values

X_train = X_train.drop(columns=["key"])
X_test = X_test.drop(columns=["key"])

X_train = X_train.fillna(0)
X_test = X_test.fillna(0)

X_train.head()



## === cell 18
from sklearn.model_selection import KFold

y_preds = []
models = []
oof_train = np.zeros((len(X_train),), dtype=float)
cv = KFold(n_splits=5, shuffle=True, random_state=0)

categorical_features = []



## === cell 19
import lightgbm as lgb

params = {
    "objective": "regression",
    "metric": "rmse",
    "max_bin": 300,
    "learning_rate": 0.05,
    "num_leaves": 40,
    "verbosity": -1,
}

for fold_id, (train_index, valid_index) in enumerate(cv.split(X_train, y_train)):
    X_tr = X_train.iloc[train_index, :]
    X_val = X_train.iloc[valid_index, :]
    y_tr = y_train.iloc[train_index]
    y_val = y_train.iloc[valid_index]

    lgb_train = lgb.Dataset(
        X_tr, y_tr, categorical_feature=categorical_features, free_raw_data=False
    )
    lgb_eval = lgb.Dataset(
        X_val,
        y_val,
        reference=lgb_train,
        categorical_feature=categorical_features,
        free_raw_data=False,
    )

    callbacks = [
        lgb.early_stopping(stopping_rounds=10, verbose=False),
        lgb.log_evaluation(period=50),
    ]

    model = lgb.train(
        params,
        lgb_train,
        valid_sets=[lgb_train, lgb_eval],
        valid_names=["train", "valid"],
        num_boost_round=1000,
        callbacks=callbacks,
    )

    oof_train[valid_index] = model.predict(X_val, num_iteration=model.best_iteration)
    y_pred = model.predict(X_test, num_iteration=model.best_iteration)

    y_preds.append(y_pred)
    models.append(model)

len(models), len(y_preds)



## === cell 20
pd.DataFrame({"oof_pred": oof_train}).to_csv("oof_train_kfold.csv", index=False)

scores = [m.best_score["valid"]["rmse"] for m in models]
score = float(sum(scores) / len(scores))
print("===CV scores (rmse)===")
print(scores)
print("Mean rmse:", score)



## === cell 21
from sklearn.metrics import mean_squared_error

rmse = float(np.sqrt(mean_squared_error(y_train, oof_train)))
print("OOF RMSE:", rmse)



## === cell 22
len(y_preds)



## === cell 23
y_preds[0][:10]



## === cell 24
y_sub = np.mean(np.vstack(y_preds), axis=0)
y_sub[:10]



## === cell 25
sub_lgb = pd.DataFrame({"key": test_keys, "fare_amount": y_sub})
sub_lgb.to_csv("submission_lightgbm.csv", index=False)

print(sub_lgb.head())
print("Saved submission_lightgbm.csv with shape:", sub_lgb.shape)
print("Columns:", sub_lgb.columns.tolist())
