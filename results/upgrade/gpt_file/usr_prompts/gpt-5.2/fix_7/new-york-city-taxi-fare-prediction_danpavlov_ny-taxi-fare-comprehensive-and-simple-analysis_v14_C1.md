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
geopy==2.4.1
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
seaborn==0.12.2
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

3.16533

# 6. Current score

5.55461

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 5.63955) has done: 'I fix the runtime error caused by changes in the XGBoost API (the `best_ntree_limit` attribute is no longer present) by switching to the supported `iteration_range` prediction logic based on `best_iteration`. I also remove notebook-only magic (`%matplotlib inline`) so the script runs as a .py in Kaggle, and update the deprecated XGBoost objective from `reg:linear` to `reg:squarederror` without changing the model’s core approach. To ensure end-to-end completion within the Kaggle time limit, I replace the extremely slow row-wise `great_circle` loop with an equivalent vectorized Haversine distance computation (same “distance feature” intent, but actually feasible at 1M rows). Finally, I make sure a valid `submission.csv` (with `key,fare_amount`) is written.'
- What this solution (achieved 7.14946) has done: 'We make two minimal changes that reliably improve RMSE without changing your model class or training loop: (1) add a deterministic `random_state` to `train_test_split` so early stopping is stable and doesn’t depend on a lucky/unlucky split, and (2) use a validation set that is sampled from `X_train` (not `X_test`) so the held-out fold remains a true test-like check while early stopping tunes on validation as intended. We also set a few standard XGBoost parameters (`eta`, `max_depth`, `subsample`, `colsample_bytree`) that keep the same XGBoost training approach but usually reduce error substantially on this competition. The submission format and file writing stay identical (`submission.csv` with `key,fare_amount`).'
- What this solution (achieved 5.2048) has done: 'You’re far from the target (7.14946 vs 3.16533, lower is better), so we need a real but still minimal modeling improvement without changing the core approach (XGBoost on the same engineered features). The biggest gain here typically comes from (1) fixing label noise/outliers more aggressively (huge fares and implausible trip distances dominate RMSE), and (2) adding a single standard geospatial feature (Manhattan distance in lat/lon degrees) while keeping the same training loop and objective. I also make the early-stopping eval set include a fixed training fold and report an internal RMSE on that fold (doesn’t affect submission, but helps verify the change). Submission writing and required columns stay identical.'
- What this solution (achieved 5.3942) has done: 'Your current score (5.2048 RMSE) is worse than the target (3.16533), so we need a modest, legitimate improvement without changing the core approach (XGBoost on engineered distance + datetime features). The biggest “minimal change” win here is to add the standard NYC Taxi competition feature `abs_lat_diff + abs_lon_diff` as miles-scaled components plus a direction feature, and to filter a small set of remaining unrealistic coordinate outliers more tightly (still the same cleaning intent). I also align training/test feature dtypes and ensure the same columns are used in both, which typically reduces noise and improves generalization slightly. Everything else (model family, training loop, objective, early stopping, and submission writing) remains the same.'
- What this solution (achieved 5.25036) has done: 'You’re well below the target (RMSE 5.3942 vs 3.16533, lower is better), so we need a modest real improvement without changing the core approach (XGBoost on engineered geo + datetime features). The most “minimal but high-impact” fix for this competition is to add the standard log1p target transform for training and invert it for predictions, which reduces the influence of large-fare outliers while keeping the same model, features, and loss. I keep the same cleaning/feature engineering and the same XGBoost training loop/early stopping, just switching the labels to `log1p(fare)` and applying `expm1` at prediction time plus a non-negative clip. This typically moves RMSE substantially toward ~3.x on this dataset with otherwise similar setups.'
- What this solution (achieved 5.55461) has done: 'To move RMSE down toward your 3.16533 target without changing the overall approach (XGBoost on the same engineered geo+datetime features with early stopping and log1p target), I make two small, high-impact fixes that reduce noise and improve generalization. First, I remove a known bad subset of training rows where pickup==dropoff (near-zero distance) but fare is non-trivial, which disproportionately harms RMSE. Second, I add two standard, minimal features (pickup/dropoff distance to NYC center) that are derived only from existing columns and commonly give a noticeable boost on this competition while keeping the same model/training loop. Everything else—including the log1p target transform, early stopping, and submission formatting—remains the same, and the script still write a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from sklearn import metrics  # evaluating models
from sklearn.model_selection import train_test_split  # set splitting and validation
from sklearn.linear_model import LinearRegression
import xgboost as xgb  # XGBoost regressor
import matplotlib.pyplot as plt  # plotting
import seaborn as sns  # plotting



## === cell 1
print(os.listdir("../input"))



## === cell 2
types = {
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "uint8",
}
test_types = {
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "uint8",
}

test = pd.read_csv("../input/test.csv", dtype=test_types)



## === cell 3
test.dtypes



## === cell 4
train = pd.read_csv("../input/train.csv", nrows=1000000, dtype=types)



## === cell 5
train.head()



## === cell 6
train.describe()



## === cell 7
sns.distplot(train["fare_amount"])



## === cell 8
sns.distplot(train["passenger_count"])



## === cell 9
train.isnull().sum()



## === cell 10
train.dropna(inplace=True)



## === cell 11
train = train[train["fare_amount"] > 0]
train = train[
    train["fare_amount"] < 250
]  # common competition cleaning; reduces noisy tail

train = train[(train["pickup_longitude"] > -74.5) & (train["pickup_longitude"] < -72.5)]
train = train[
    (train["dropoff_longitude"] > -74.5) & (train["dropoff_longitude"] < -72.5)
]
train = train[(train["pickup_latitude"] > 40.5) & (train["pickup_latitude"] < 41.9)]
train = train[(train["dropoff_latitude"] > 40.5) & (train["dropoff_latitude"] < 41.9)]

train = train[(train["passenger_count"] > 0) & (train["passenger_count"] < 10)]



## === cell 12
train.describe()




## === cell 13
def add_distance_haversine_km(df):
    lat1 = np.radians(df["pickup_latitude"].astype("float64").values)
    lon1 = np.radians(df["pickup_longitude"].astype("float64").values)
    lat2 = np.radians(df["dropoff_latitude"].astype("float64").values)
    lon2 = np.radians(df["dropoff_longitude"].astype("float64").values)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * (
        np.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * np.arcsin(np.sqrt(a))

    earth_radius_km = 6371.0
    df["distance"] = (earth_radius_km * c).astype("float32")
    return df




## === cell 14
def add_manhattan_approx(df):
    df["abs_lon_diff"] = (
        (df["dropoff_longitude"] - df["pickup_longitude"]).abs().astype("float32")
    )
    df["abs_lat_diff"] = (
        (df["dropoff_latitude"] - df["pickup_latitude"]).abs().astype("float32")
    )
    df["manhattan_approx"] = (df["abs_lon_diff"] + df["abs_lat_diff"]).astype("float32")
    return df




## === cell 15
def add_geo_features(df):
    lat = df["pickup_latitude"].astype("float64").values
    dlat = (df["dropoff_latitude"] - df["pickup_latitude"]).astype("float64").values
    dlon = (df["dropoff_longitude"] - df["pickup_longitude"]).astype("float64").values

    lat_km = 111.0 * dlat
    lon_km = 111.0 * np.cos(np.radians(lat)) * dlon

    df["dlat_km"] = lat_km.astype("float32")
    df["dlon_km"] = lon_km.astype("float32")
    df["manhattan_km"] = (np.abs(lat_km) + np.abs(lon_km)).astype("float32")
    df["direction"] = np.arctan2(lat_km, lon_km).astype("float32")
    return df




## === cell 16
def add_nyc_center_features(df, center_lat=40.7141667, center_lon=-74.0063889):
    lat_c = np.radians(np.float64(center_lat))
    lon_c = np.radians(np.float64(center_lon))

    lat1 = np.radians(df["pickup_latitude"].astype("float64").values)
    lon1 = np.radians(df["pickup_longitude"].astype("float64").values)
    dlat = lat_c - lat1
    dlon = lon_c - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat_c) * (
        np.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * np.arcsin(np.sqrt(a))
    df["pickup_center_dist_km"] = (6371.0 * c).astype("float32")

    lat2 = np.radians(df["dropoff_latitude"].astype("float64").values)
    lon2 = np.radians(df["dropoff_longitude"].astype("float64").values)
    dlat = lat_c - lat2
    dlon = lon_c - lon2
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat2) * np.cos(lat_c) * (
        np.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * np.arcsin(np.sqrt(a))
    df["dropoff_center_dist_km"] = (6371.0 * c).astype("float32")

    return df




## === cell 17
train = add_distance_haversine_km(train)
test = add_distance_haversine_km(test)

train = add_manhattan_approx(train)
test = add_manhattan_approx(test)

train = add_geo_features(train)
test = add_geo_features(test)

train = add_nyc_center_features(train)
test = add_nyc_center_features(test)



## === cell 18
train = train[(train["distance"] > 0.01) & (train["distance"] < 150)].copy()



## === cell 19
train = train[~((train["distance"] < 0.05) & (train["fare_amount"] > 6.0))].copy()



## === cell 20
train["pickup_datetime"] = (
    train["pickup_datetime"].astype(str).str.replace(" UTC", "", regex=False)
)
train["pickup_datetime"] = pd.to_datetime(train["pickup_datetime"], errors="coerce")

test["pickup_datetime"] = (
    test["pickup_datetime"].astype(str).str.replace(" UTC", "", regex=False)
)
test["pickup_datetime"] = pd.to_datetime(test["pickup_datetime"], errors="coerce")

train = train.dropna(subset=["pickup_datetime"]).copy()
test = test.dropna(subset=["pickup_datetime"]).copy()



## === cell 21
train["hour"] = train.pickup_datetime.dt.hour.astype("int16")
train["weekday"] = train.pickup_datetime.dt.weekday.astype("int16")
train["month"] = train.pickup_datetime.dt.month.astype("int16")
train["year"] = train.pickup_datetime.dt.year.astype("int16")

test["hour"] = test.pickup_datetime.dt.hour.astype("int16")
test["weekday"] = test.pickup_datetime.dt.weekday.astype("int16")
test["month"] = test.pickup_datetime.dt.month.astype("int16")
test["year"] = test.pickup_datetime.dt.year.astype("int16")



## === cell 22
test.head()



## === cell 23
plt.figure(figsize=(15, 8))
sns.heatmap(
    train.drop(["key", "pickup_datetime"], axis=1).corr(), annot=True, fmt=".4f"
)



## === cell 24
X = train.drop(["key", "fare_amount", "pickup_datetime"], axis=1)
y = np.log1p(train["fare_amount"].astype("float32"))



## === cell 25
X.head()



## === cell 26
y.head()



## === cell 27
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)



## === cell 28
test_pred = test.drop(["key", "pickup_datetime"], axis=1)
test_pred = test_pred.reindex(columns=X.columns, fill_value=0)



## === cell 29
lm = LinearRegression()
lm.fit(X_train, y_train)
print(lm.score(X_train, y_train))
print(lm.score(X_test, y_test))



## === cell 30
y_pred = lm.predict(X)
lrmse = np.sqrt(metrics.mean_squared_error(y_pred, y))
lrmse



## === cell 31
LinearPredictions = lm.predict(test_pred)
LinearPredictions = np.expm1(LinearPredictions)
LinearPredictions = np.clip(LinearPredictions, 0, None)
LinearPredictions = np.round(LinearPredictions, decimals=2)
LinearPredictions



## === cell 32
LinearPredictions.size



## === cell 33
linear_submission = pd.DataFrame(
    {"key": test["key"], "fare_amount": LinearPredictions},
    columns=["key", "fare_amount"],
)



## === cell 34
linear_submission.head()




## === cell 35
def XGBoost(X_train, X_test, y_train, y_test):
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dval = xgb.DMatrix(X_val, label=y_val)

    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "eta": 0.1,
        "max_depth": 8,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "seed": 42,
    }

    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=2000,
        early_stopping_rounds=50,
        evals=[(dval, "valid")],
        verbose_eval=False,
    )

    best_iter = getattr(booster, "best_iteration", None)
    if best_iter is None:
        val_pred_log = booster.predict(dval)
    else:
        val_pred_log = booster.predict(dval, iteration_range=(0, best_iter + 1))

    val_pred = np.expm1(val_pred_log)
    y_val_true = np.expm1(y_val.values if hasattr(y_val, "values") else y_val)
    val_rmse = float(np.sqrt(metrics.mean_squared_error(y_val_true, val_pred)))
    print("Internal validation RMSE in $ space (for sanity check):", val_rmse)

    return booster




## === cell 36
xgbm = XGBoost(X_train, X_test, y_train, y_test)

best_iter = getattr(xgbm, "best_iteration", None)
dtest_full = xgb.DMatrix(test_pred)

if best_iter is None:
    XGBPredictions_log = xgbm.predict(dtest_full)
else:
    XGBPredictions_log = xgbm.predict(dtest_full, iteration_range=(0, best_iter + 1))



## === cell 37
XGBPredictions_log



## === cell 38
XGBPredictions = np.expm1(XGBPredictions_log)
XGBPredictions = np.clip(XGBPredictions, 0, None)
XGBPredictions = np.round(XGBPredictions, decimals=2)
XGBPredictions



## === cell 39
XGB_submission = pd.DataFrame(
    {"key": test["key"], "fare_amount": XGBPredictions}, columns=["key", "fare_amount"]
)
XGB_submission.head()



## === cell 40
submission = XGB_submission



## === cell 41
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())
