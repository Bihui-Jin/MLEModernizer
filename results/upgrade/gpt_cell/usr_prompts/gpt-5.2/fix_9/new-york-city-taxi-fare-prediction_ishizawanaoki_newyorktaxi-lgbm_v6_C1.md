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

3.7695

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

- What this solution (achieved 12.96532) has done: 'Your code likely didn’t yield a Kaggle score because the notebook reads from `../input/...` while your environment paths are `/kaggle/input/...`, so it may fail before writing `submission_lightgbm.csv`. I make the smallest path fix so the script runs end-to-end and always writes a valid submission CSV with the required columns. To gently improve RMSE toward your target without changing the modeling approach, I remove early stopping (it can underfit with this small feature set) and use a fixed `num_boost_round` while keeping the same LightGBM model/training loop. I also ensure `key` stays aligned with test rows by building the submission directly from `X_test['key']` rather than reusing `sample_submission` (prevents any ordering/mismatch issues).'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
DATA_DIR = "/kaggle/input/new-york-city-taxi-fare-prediction"

train = pd.read_csv(f"{DATA_DIR}/train.csv", nrows=1_000_000)
test = pd.read_csv(f"{DATA_DIR}/test.csv")
sample_submission = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")



## === cell 2
train.isnull().sum()



## === cell 3
train.dropna(inplace=True)



## === cell 4
train.describe()



## === cell 5
train.quantile(0.99, numeric_only=True)



## === cell 6
train.quantile(0.01, numeric_only=True)



## === cell 7
train = train.query("1 <= passenger_count <= 6 and 3.3 <= fare_amount <= 52.33")
train.describe()



## === cell 8
train.reset_index(drop=True, inplace=True)
train



## === cell 9
coord_q = (
    "(-74.5 <= pickup_longitude <= -72.8) and (40.5 <= pickup_latitude <= 41.8) and "
    "(-74.5 <= dropoff_longitude <= -72.8) and (40.5 <= dropoff_latitude <= 41.8)"
)
train = train.query(coord_q).copy()

train = train.query(
    "(pickup_longitude != 0) and (pickup_latitude != 0) and (dropoff_longitude != 0) and (dropoff_latitude != 0)"
).copy()

train.reset_index(drop=True, inplace=True)
print("Train rows after coord cleaning:", len(train))



## === cell 10
data = pd.concat([train, test], sort=False, ignore_index=True)



## === cell 11
data.head()




## === cell 12
def haversine_np(lon1, lat1, lon2, lat2):
    lon1 = np.radians(lon1.astype(float))
    lat1 = np.radians(lat1.astype(float))
    lon2 = np.radians(lon2.astype(float))
    lat2 = np.radians(lat2.astype(float))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    r_km = 6371.0
    return c * r_km


data["pickup_datetime"] = pd.to_datetime(
    data["pickup_datetime"], errors="coerce", utc=True
)
data["pickup_hour"] = data["pickup_datetime"].dt.hour.astype("float32")
data["pickup_dow"] = data["pickup_datetime"].dt.dayofweek.astype("float32")
data = data.drop("pickup_datetime", axis=1)

data["key"] = data["key"].astype(str)
data = data[data["key"].notna()].copy()

data["distance_km"] = haversine_np(
    data["pickup_longitude"],
    data["pickup_latitude"],
    data["dropoff_longitude"],
    data["dropoff_latitude"],
)

data["abs_lon_diff"] = (
    data["pickup_longitude"].astype(float) - data["dropoff_longitude"].astype(float)
).abs()
data["abs_lat_diff"] = (
    data["pickup_latitude"].astype(float) - data["dropoff_latitude"].astype(float)
).abs()

data.head()



## === cell 13
n_train = len(train)
train_fe = data.iloc[:n_train].copy()
test_fe = data.iloc[n_train:].copy()

train_fe = train_fe.query("0 < distance_km < 100").copy()
train_fe.reset_index(drop=True, inplace=True)

y_train = train_fe["fare_amount"]
X_train = train_fe.drop(["fare_amount", "key"], axis=1)
X_test = test_fe.drop(["fare_amount", "key"], axis=1)

for df_ in (X_train, X_test):
    df_.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_.fillna(0.0, inplace=True)

print("Final X_train shape:", X_train.shape, "X_test shape:", X_test.shape)
X_train.head()



## === cell 14
from sklearn.model_selection import KFold

y_preds = []
models = []
oof_train = np.zeros((len(X_train),))
cv = KFold(n_splits=5, shuffle=True, random_state=0)

categorical_features = []



## === cell 15
import lightgbm as lgb

params = {
    "objective": "regression",
    "metric": "rmse",
    "max_bin": 300,
    "learning_rate": 0.05,
    "num_leaves": 40,
    "min_data_in_leaf": 20,
    "lambda_l2": 1.0,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 1,
    "seed": 0,
    "feature_fraction_seed": 0,
    "bagging_seed": 0,
    "verbose": -1,
}

for fold_id, (train_index, valid_index) in enumerate(cv.split(X_train, y_train)):
    X_tr = X_train.loc[train_index, :]
    X_val = X_train.loc[valid_index, :]
    y_tr = y_train.iloc[train_index]
    y_val = y_train.iloc[valid_index]

    lgb_train = lgb.Dataset(X_tr, y_tr, categorical_feature=categorical_features)
    lgb_eval = lgb.Dataset(
        X_val, y_val, reference=lgb_train, categorical_feature=categorical_features
    )

    model = lgb.train(
        params,
        lgb_train,
        valid_sets=[lgb_train, lgb_eval],
        num_boost_round=1500,
        callbacks=[
            lgb.log_evaluation(period=50),
        ],
    )

    oof_train[valid_index] = model.predict(
        X_val, num_iteration=model.current_iteration()
    )
    y_pred = model.predict(X_test, num_iteration=model.current_iteration())

    y_preds.append(y_pred)
    models.append(model)



## === cell 16
pd.DataFrame(oof_train).to_csv("oof_train_kfold.csv", index=False)
print(f"Trained {len(models)} fold models.")



## === cell 17
from sklearn.metrics import mean_squared_error

y_pred_oof = oof_train
rmse = np.sqrt(mean_squared_error(y_train, y_pred_oof))
print("OOF RMSE:", rmse)



## === cell 18
len(y_preds)



## === cell 19
y_preds[0][:10]



## === cell 20
y_sub = sum(y_preds) / len(y_preds)

y_sub = np.clip(y_sub, 0, None)

y_sub[:10]



## === cell 21
sub_lgb = pd.DataFrame({"key": test_fe["key"].astype(str).values, "fare_amount": y_sub})
sub_lgb.to_csv("submission_lightgbm.csv", index=False)

print("Wrote submission_lightgbm.csv with shape:", sub_lgb.shape)
sub_lgb.head()
