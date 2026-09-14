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

3.40091

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
BASE_PATH = "/kaggle/input/new-york-city-taxi-fare-prediction"

train = pd.read_csv(f"{BASE_PATH}/train.csv", nrows=1_000_000)
test = pd.read_csv(f"{BASE_PATH}/test.csv")
sample_submission = pd.read_csv(f"{BASE_PATH}/sample_submission.csv")



## === cell 2
train.isnull().sum()



## === cell 3
train.dropna(inplace=True)



## === cell 4
train.describe()



## === cell 5
train.query("passenger_count > 6")



## === cell 6
train.query("passenger_count < 1")



## === cell 7
train.query("fare_amount < 0")



## === cell 8
train.query("pickup_longitude < -180 or pickup_longitude > 180")



## === cell 9
train.query("dropoff_longitude < -180 or dropoff_longitude > 180")



## === cell 10
train.query("pickup_latitude < -90 or pickup_latitude > 90")



## === cell 11
train.query("dropoff_latitude < -90 or dropoff_latitude > 90")



## === cell 12
train = train.query(
    "1 <= passenger_count <= 6 and "
    "0 <= fare_amount and "
    "-180 <= pickup_longitude <= 180 and "
    "-180 <= dropoff_longitude <= 180 and "
    "-90 <= pickup_latitude <= 90 and "
    "-90 <= dropoff_latitude <= 90"
)

train = train[(train["fare_amount"] <= 250)].copy()
train = train[
    ~(
        (train["pickup_longitude"] == 0)
        | (train["pickup_latitude"] == 0)
        | (train["dropoff_longitude"] == 0)
        | (train["dropoff_latitude"] == 0)
    )
].copy()

train.describe()



## === cell 13
train.reset_index(drop=True, inplace=True)
train




## === cell 14
def haversine_np(lon1, lat1, lon2, lat2):
    lon1 = np.radians(lon1.astype(np.float64))
    lat1 = np.radians(lat1.astype(np.float64))
    lon2 = np.radians(lon2.astype(np.float64))
    lat2 = np.radians(lat2.astype(np.float64))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371.0 * c  # km


nyc_box = (
    (train["pickup_longitude"].between(-74.3, -73.7))
    & (train["dropoff_longitude"].between(-74.3, -73.7))
    & (train["pickup_latitude"].between(40.5, 41.0))
    & (train["dropoff_latitude"].between(40.5, 41.0))
)
train = train.loc[nyc_box].copy()

dist_km = haversine_np(
    train["pickup_longitude"].values,
    train["pickup_latitude"].values,
    train["dropoff_longitude"].values,
    train["dropoff_latitude"].values,
)
train = train.loc[(dist_km > 0) & (dist_km < 200)].copy()
train.reset_index(drop=True, inplace=True)



## === cell 15
test_key_for_submission = test["key"].astype(str).values.copy()

data = pd.concat([train, test], sort=False)



## === cell 16
data.head()



## === cell 17
data = data.drop("pickup_datetime", axis=1)

data["key_str"] = data["key"].astype(str)

data["key_hash"] = (
    pd.util.hash_pandas_object(data["key_str"], index=False)
    .astype(np.uint64)
    .astype(np.float64)
)

data["distance_km"] = haversine_np(
    data["pickup_longitude"].values,
    data["pickup_latitude"].values,
    data["dropoff_longitude"].values,
    data["dropoff_latitude"].values,
).astype(np.float32)

data["key"] = data["key_hash"].astype(np.float64)

data.head()



## === cell 18
train = data.iloc[: len(train)].copy()
test = data.iloc[len(train) :].copy()

y_train = train["fare_amount"].astype(np.float32)
X_train = train.drop(["fare_amount", "key_str", "key_hash"], axis=1)
X_test = test.drop(["fare_amount", "key_str", "key_hash"], axis=1)

X_train.head()



## === cell 19
from sklearn.model_selection import KFold

y_preds = []
models = []
oof_train = np.zeros((len(X_train),), dtype=np.float32)
cv = KFold(n_splits=5, shuffle=True, random_state=0)

categorical_features = []



## === cell 20
import lightgbm as lgb

params = {
    "objective": "regression",
    "metric": "rmse",
    "max_bin": 300,
    "learning_rate": 0.05,
    "num_leaves": 40,
    "verbosity": -1,
    "seed": 0,
    "feature_fraction_seed": 0,
    "bagging_seed": 0,
}

num_boost_round = 800  # keep fixed rounds as in original (no early stopping)

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

    model = lgb.train(
        params,
        lgb_train,
        valid_sets=[lgb_train, lgb_eval],
        valid_names=["train", "valid"],
        num_boost_round=num_boost_round,
        callbacks=[
            lgb.log_evaluation(period=50),
        ],
    )

    oof_train[valid_index] = model.predict(X_val, num_iteration=num_boost_round)
    y_pred = model.predict(X_test, num_iteration=num_boost_round)

    y_preds.append(y_pred)
    models.append(model)



## === cell 21
pd.DataFrame(oof_train, columns=["oof_pred"]).to_csv(
    "/kaggle/working/oof_train_kfold.csv", index=False
)

scores = [m.best_score["valid"]["rmse"] for m in models]
score = float(np.mean(scores))
print("===CV scores (rmse)===")
print(scores)
print(score)



## === cell 22
from sklearn.metrics import mean_squared_error

y_pred_oof = oof_train
np.sqrt(mean_squared_error(y_train, y_pred_oof))



## === cell 23
len(y_preds)



## === cell 24
y_preds[0][:10]



## === cell 25
y_sub = sum(y_preds) / len(y_preds)
y_sub[:10]



## === cell 26
y_sub = np.maximum(y_sub, 0.0)

sub_lgb = pd.DataFrame(
    {"key": test_key_for_submission, "fare_amount": y_sub.astype(np.float32)}
)

submission_path = "/kaggle/working/submission_lightgbm.csv"
sub_lgb.to_csv(submission_path, index=False)

print("Wrote:", submission_path, "rows:", len(sub_lgb), "cols:", list(sub_lgb.columns))
sub_lgb.head()
