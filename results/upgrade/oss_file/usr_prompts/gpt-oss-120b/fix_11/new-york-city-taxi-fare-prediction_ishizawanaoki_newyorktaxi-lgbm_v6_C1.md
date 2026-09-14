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

# 5. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
train = pd.read_csv("/kaggle/input/new-york-city-taxi-fare-prediction/train.csv")
test = pd.read_csv("/kaggle/input/new-york-city-taxi-fare-prediction/test.csv")
sample_submission = pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/sample_submission.csv"
)



## === cell 2
train.isnull().sum()



## === cell 3
train.dropna(inplace=True)



## === cell 4
train.describe()



## === cell 5
train.select_dtypes(include=[np.number]).quantile(0.99)



## === cell 6
train.select_dtypes(include=[np.number]).quantile(0.01)



## === cell 7
train = train.query("1 <= passenger_count <= 6 and 3.3 <= fare_amount <= 52.33")
train = train[
    (train["pickup_longitude"].between(-74.5, -73.0))
    & (train["pickup_latitude"].between(40.0, 41.5))
    & (train["dropoff_longitude"].between(-74.5, -73.0))
    & (train["dropoff_latitude"].between(40.0, 41.5))
]
train.describe()



## === cell 8
train.reset_index(drop=True, inplace=True)
train



## === cell 9
data = pd.concat([train, test], sort=False)



## === cell 10
data["pickup_datetime"] = pd.to_datetime(data["pickup_datetime"], errors="coerce")
data["pickup_hour"] = data["pickup_datetime"].dt.hour
data["pickup_weekday"] = data["pickup_datetime"].dt.weekday
data["pickup_month"] = data["pickup_datetime"].dt.month

data = data.drop("pickup_datetime", axis=1)



## === cell 11
data.head()



## === cell 12
train = data[: len(train)]
test = data[len(train) :]

y_train = train["fare_amount"]
y_train_log = np.log1p(y_train)

X_train = train.drop(["fare_amount", "key"], axis=1)
X_test = test.drop(["fare_amount", "key"], axis=1)


def haversine(lon1, lat1, lon2, lat2):
    """Return distance in km between two lon/lat points."""
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def manhattan_distance(lon1, lat1, lon2, lat2):
    """Approximate Manhattan distance (km) by summing two orthogonal haversine legs."""
    leg1 = haversine(lon1, lat1, lon2, lat1)
    leg2 = haversine(lon2, lat1, lon2, lat2)
    return leg1 + leg2


distance_train = haversine(
    X_train["pickup_longitude"],
    X_train["pickup_latitude"],
    X_train["dropoff_longitude"],
    X_train["dropoff_latitude"],
)
distance_test = haversine(
    X_test["pickup_longitude"],
    X_test["pickup_latitude"],
    X_test["dropoff_longitude"],
    X_test["dropoff_latitude"],
)

X_train = X_train.assign(distance_km=distance_train)
X_test = X_test.assign(distance_km=distance_test)

X_train = X_train.assign(
    manhattan_km=manhattan_distance(
        X_train["pickup_longitude"],
        X_train["pickup_latitude"],
        X_train["dropoff_longitude"],
        X_train["dropoff_latitude"],
    )
)
X_test = X_test.assign(
    manhattan_km=manhattan_distance(
        X_test["pickup_longitude"],
        X_test["pickup_latitude"],
        X_test["dropoff_longitude"],
        X_test["dropoff_latitude"],
    )
)

X_train = X_train.assign(
    lat_diff=np.abs(X_train["pickup_latitude"] - X_train["dropoff_latitude"]),
    lon_diff=np.abs(X_train["pickup_longitude"] - X_train["dropoff_longitude"]),
)
X_test = X_test.assign(
    lat_diff=np.abs(X_test["pickup_latitude"] - X_test["dropoff_latitude"]),
    lon_diff=np.abs(X_test["pickup_longitude"] - X_test["dropoff_longitude"]),
)

X_train = X_train.assign(
    hour_passenger=X_train["pickup_hour"] * X_train["passenger_count"]
)
X_test = X_test.assign(hour_passenger=X_test["pickup_hour"] * X_test["passenger_count"])

X_train = X_train.assign(
    hour_sin=np.sin(2 * np.pi * X_train["pickup_hour"] / 24),
    hour_cos=np.cos(2 * np.pi * X_train["pickup_hour"] / 24),
    distance_passenger=X_train["distance_km"] * X_train["passenger_count"],
)
X_test = X_test.assign(
    hour_sin=np.sin(2 * np.pi * X_test["pickup_hour"] / 24),
    hour_cos=np.cos(2 * np.pi * X_test["pickup_hour"] / 24),
    distance_passenger=X_test["distance_km"] * X_test["passenger_count"],
)

X_train = X_train.assign(
    distance_log=np.log1p(X_train["distance_km"]),
    passenger_log=np.log1p(X_train["passenger_count"]),
    distance_per_passenger=X_train["distance_km"]
    / X_train["passenger_count"].replace(0, np.nan),
)
X_test = X_test.assign(
    distance_log=np.log1p(X_test["distance_km"]),
    passenger_log=np.log1p(X_test["passenger_count"]),
    distance_per_passenger=X_test["distance_km"]
    / X_test["passenger_count"].replace(0, np.nan),
)

X_train["distance_per_passenger"] = X_train["distance_per_passenger"].fillna(0)
X_test["distance_per_passenger"] = X_test["distance_per_passenger"].fillna(0)

X_train_np = X_train.values
X_test_np = X_test.values
y_train_log_np = y_train_log.values



## === cell 13
from sklearn.model_selection import KFold

y_preds = []
models = []
oof_train = np.zeros((len(X_train_np),))
cv = KFold(n_splits=5, shuffle=True, random_state=0)

categorical_features = []  # no categorical columns used



## === cell 14
import lightgbm as lgb

params = {
    "objective": "regression",
    "max_bin": 300,
    "learning_rate": 0.05,
    "num_leaves": 256,  # increased capacity
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "metric": "rmse",  # optimise RMSE directly (on log‑target)
    "verbosity": -1,
    "min_data_in_leaf": 20,  # slight regularisation to improve generalisation
    "lambda_l2": 0.1,  # tiny L2 regularisation for more stable predictions
}

for fold_id, (train_index, valid_index) in enumerate(
    cv.split(X_train_np, y_train_log_np)
):
    X_tr = X_train_np[train_index]
    X_val = X_train_np[valid_index]
    y_tr_log = y_train_log_np[train_index]
    y_val_log = y_train_log_np[valid_index]

    lgb_train = lgb.Dataset(X_tr, y_tr_log, categorical_feature=categorical_features)
    lgb_eval = lgb.Dataset(
        X_val, y_val_log, reference=lgb_train, categorical_feature=categorical_features
    )

    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=2000,
        valid_sets=[lgb_train, lgb_eval],
        callbacks=[
            lgb.early_stopping(stopping_rounds=20, verbose=False),
            lgb.log_evaluation(period=10, show_stdv=False),
        ],
    )

    oof_train[valid_index] = np.expm1(
        model.predict(X_val, num_iteration=model.best_iteration)
    )
    y_pred = np.expm1(model.predict(X_test_np, num_iteration=model.best_iteration))

    y_preds.append(y_pred)
    models.append(model)



## === cell 15
pd.DataFrame(oof_train, columns=["oof_pred"]).to_csv("oof_train_kfold.csv", index=False)

bias = (y_train - oof_train).mean()
oof_train += bias  # adjust OOF predictions
y_preds = [pred + bias for pred in y_preds]  # propagate correction to test predictions

scores = [m.best_score["valid_1"]["rmse"] for m in models]
score = sum(scores) / len(scores) if scores else float("nan")
print("===CV scores (log‑target)===")
print(scores)
print("Mean CV RMSE (log‑target):", score)



## === cell 16
from sklearn.metrics import mean_squared_error

y_pred_oof = oof_train
print("OOF RMSE (original scale):", np.sqrt(mean_squared_error(y_train, y_pred_oof)))



## === cell 17
print("Number of folds predictions collected:", len(y_preds))



## === cell 18
if y_preds:
    y_sub = np.mean(y_preds, axis=0)
else:
    y_sub = np.zeros(len(X_test_np))

print("First 10 averaged predictions:", y_sub[:10])



## === cell 19
sub_lgb = sample_submission.copy()
sub_lgb["fare_amount"] = y_sub
sub_lgb.to_csv("submission_lightgbm.csv", index=False)

print("Submission file head:")
print(sub_lgb.head())
