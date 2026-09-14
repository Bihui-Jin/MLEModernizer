# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

3.43515

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 5.38121) has done: 'Your code already trains and writes a submission CSV, but Kaggle didn’t “yield” a score likely because the file name/placement didn’t match what you uploaded, or because the pipeline can crash due to invalid target values (negative/zero fares, NaNs, out-of-range lat/lon) that LightGBM still fit but generalize poorly. I make minimal, score-relevant fixes: ensure we always write `submission.csv` (the common Kaggle default), add lightweight NYC taxi sanity filtering on the 1M training rows (removing impossible coordinates/fare/passenger_count) to improve RMSE without changing the model or training loop, and clip predictions to a valid non-negative range. These changes keep the same LightGBM setup and CV approach, just improve data correctness and submission reliability.'
- What this solution (achieved 5.05637) has done: 'Your current RMSE (5.38) is worse than the target (3.44), so we should improve generalization with minimal, score-relevant changes while keeping the same LightGBM/KFold training loop. The biggest gap is that the model is only using raw lon/lat/passenger_count after dropping datetime, which is too weak; we keep LightGBM as-is but add a few standard, cheap geospatial features (Haversine distance + coordinate deltas) derived from the same columns. We also apply the same NYC bounding-box sanitation to the test set (without dropping rows) and fill any invalid/out-of-box test coordinates with the train medians so the engineered features remain reasonable. Finally, we keep the same submission schema/path and still clip predictions to a valid range.'
- What this solution (achieved 4.99491) has done: 'We need to move RMSE down (lower is better) from 5.056 toward 3.435, so the smallest score-relevant improvements are to add a few more standard taxi features without changing the LightGBM training loop or objective. I keep your exact CV/training approach but (1) extract a handful of datetime features you currently drop, and (2) add two simple geospatial features (Manhattan distance proxy and bearing) that commonly reduce RMSE on this dataset. I also switch LightGBM’s metric from `l2` to `rmse` (same objective; just better-aligned eval/early-stopping) and increase `stopping_rounds` slightly to reduce premature stopping—this doesn’t change core logic, only training termination. Submission writing stays identical and still outputs `submission.csv`.'
- What this solution (achieved 5.30126) has done: 'Your current RMSE (4.99) is still far above the target (3.44), so we should improve generalization with minimal, score-relevant changes while keeping the same LightGBM/KFold loop and objective. The biggest remaining gap is data quality: NYC Taxi Fare has many noisy/outlier records even within a broad NYC box, and tightening the “sane trip” filters (distance, coordinate sanity, and a simple fare-vs-distance plausibility bound) typically yields a large RMSE drop without changing the model itself. I also make one small, metric-aligned tweak: train on `log1p(fare_amount)` and invert with `expm1` at prediction time (still regression + RMSE, same training loop), which usually reduces the impact of heavy-tailed fares and improves RMSE. Submission writing remains identical and still produces `submission.csv`.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd

import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
train = pd.read_csv(
    "../input/new-york-city-taxi-fare-prediction/train.csv", nrows=1_000_000
)
test = pd.read_csv("../input/new-york-city-taxi-fare-prediction/test.csv")
sample_submission = pd.read_csv(
    "../input/new-york-city-taxi-fare-prediction/sample_submission.csv"
)



## === cell 2
train = train.dropna(
    subset=[
        "fare_amount",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
        "pickup_datetime",
    ]
)

train["fare_amount"] = pd.to_numeric(train["fare_amount"], errors="coerce")
train = train.dropna(subset=["fare_amount"])

train = train[(train["fare_amount"] > 0) & (train["fare_amount"] < 250)]
train = train[(train["passenger_count"] >= 1) & (train["passenger_count"] <= 6)]


def _in_nyc_box(df):
    return (
        (df["pickup_longitude"].between(-74.3, -73.7))
        & (df["dropoff_longitude"].between(-74.3, -73.7))
        & (df["pickup_latitude"].between(40.5, 41.0))
        & (df["dropoff_latitude"].between(40.5, 41.0))
    )


train = train[_in_nyc_box(train)].copy()



## === cell 3
data = pd.concat([train, test], sort=False, ignore_index=True)



## === cell 4
data.head()




## === cell 5
def _haversine_km(lon1, lat1, lon2, lat2):
    lon1 = np.radians(np.asarray(lon1, dtype=float))
    lat1 = np.radians(np.asarray(lat1, dtype=float))
    lon2 = np.radians(np.asarray(lon2, dtype=float))
    lat2 = np.radians(np.asarray(lat2, dtype=float))
    dlon = lon2 - lon1
    dlon = (dlon + np.pi) % (2.0 * np.pi) - np.pi  # dateline-safe
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 6371.0 * (2.0 * np.arcsin(np.sqrt(a)))


def _bearing(lon1, lat1, lon2, lat2):
    lon1 = np.radians(np.asarray(lon1, dtype=float))
    lat1 = np.radians(np.asarray(lat1, dtype=float))
    lon2 = np.radians(np.asarray(lon2, dtype=float))
    lat2 = np.radians(np.asarray(lat2, dtype=float))
    dlon = lon2 - lon1
    dlon = (dlon + np.pi) % (2.0 * np.pi) - np.pi  # dateline-safe
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    brng = np.arctan2(y, x)  # radians in [-pi, pi]
    return np.asarray(brng, dtype=float)


coord_cols = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
]
train_coord_medians = train[coord_cols].median()

test_mask = data[
    "fare_amount"
].isna()  # rows belonging to test in the concatenated frame
test_rows = data.loc[test_mask, coord_cols]
invalid_test = ~_in_nyc_box(data.loc[test_mask])

data.loc[test_mask, coord_cols] = test_rows.fillna(train_coord_medians)

if invalid_test.any():
    bad_idx = data.loc[test_mask].index[invalid_test.values]
    for c in coord_cols:
        data.loc[bad_idx, c] = float(train_coord_medians[c])

data["abs_lon_diff"] = (data["pickup_longitude"] - data["dropoff_longitude"]).abs()
data["abs_lat_diff"] = (data["pickup_latitude"] - data["dropoff_latitude"]).abs()

data["haversine_km"] = _haversine_km(
    data["pickup_longitude"],
    data["pickup_latitude"],
    data["dropoff_longitude"],
    data["dropoff_latitude"],
).astype(float)

data["manhattan_km"] = (
    _haversine_km(
        data["pickup_longitude"],
        data["pickup_latitude"],
        data["dropoff_longitude"],
        data["pickup_latitude"],
    )
    + _haversine_km(
        data["pickup_longitude"],
        data["pickup_latitude"],
        data["pickup_longitude"],
        data["dropoff_latitude"],
    )
).astype(float)

data["bearing_rad"] = _bearing(
    data["pickup_longitude"],
    data["pickup_latitude"],
    data["dropoff_longitude"],
    data["dropoff_latitude"],
).astype(float)

NYC_LON, NYC_LAT = -73.985428, 40.748817  # Midtown Manhattan (Empire State Building)
data["pickup_to_center_km"] = _haversine_km(
    data["pickup_longitude"], data["pickup_latitude"], NYC_LON, NYC_LAT
).astype(float)
data["dropoff_to_center_km"] = _haversine_km(
    data["dropoff_longitude"], data["dropoff_latitude"], NYC_LON, NYC_LAT
).astype(float)

dt = pd.to_datetime(data["pickup_datetime"], errors="coerce", utc=True)
dt = dt.dt.tz_convert(None)  # naive for fast .dt access

data["pickup_hour"] = dt.dt.hour.astype("float32")
data["pickup_dayofweek"] = dt.dt.dayofweek.astype("float32")
data["pickup_month"] = dt.dt.month.astype("float32")
data["pickup_day"] = dt.dt.day.astype("float32")

if "pickup_datetime" in data.columns:
    data = data.drop("pickup_datetime", axis=1)

data["key"] = data["key"].astype(str)

data.head()



## === cell 6
len_original_train = 1_000_000  # nrows used when reading train.csv above

train = data.iloc[:len_original_train].copy()
test = data.iloc[len_original_train:].copy()

train = train.dropna(subset=["fare_amount"]).copy()
train = train[(train["haversine_km"] > 0.05) & (train["haversine_km"] < 80.0)].copy()

min_fare = 2.5 + 0.3 * train["haversine_km"]  # low slope, permissive
max_fare = 7.0 + 20.0 * train["haversine_km"]  # very permissive upper slope
train = train[
    (train["fare_amount"] >= min_fare) & (train["fare_amount"] <= max_fare)
].copy()

y_train = train["fare_amount"].astype(float)
X_train = train.drop(["fare_amount", "key"], axis=1)
X_test = test.drop(["fare_amount", "key"], axis=1)

X_train = X_train.apply(pd.to_numeric, errors="coerce")
X_test = X_test.apply(pd.to_numeric, errors="coerce")

fill_vals = X_train.median(numeric_only=True)
X_train = X_train.fillna(fill_vals)
X_test = X_test.fillna(fill_vals)

X_train.head()



## === cell 7
from sklearn.model_selection import KFold

y_preds = []
models = []
oof_train = np.zeros((len(X_train),), dtype=float)
cv = KFold(n_splits=5, shuffle=True, random_state=0)

categorical_features = []



## === cell 8
import lightgbm as lgb

params = {
    "objective": "regression",
    "metric": "rmse",
    "max_bin": 300,
    "learning_rate": 0.05,
    "num_leaves": 40,
    "verbosity": -1,
}

y_train_log = np.log1p(y_train.values.astype(float))

for fold_id, (train_index, valid_index) in enumerate(
    cv.split(X_train, y_train_log), start=1
):
    X_tr = X_train.iloc[train_index, :]
    X_val = X_train.iloc[valid_index, :]
    y_tr = y_train_log[train_index]
    y_val = y_train_log[valid_index]

    lgb_train = lgb.Dataset(
        X_tr, label=y_tr, categorical_feature=categorical_features, free_raw_data=False
    )
    lgb_eval = lgb.Dataset(
        X_val,
        label=y_val,
        reference=lgb_train,
        categorical_feature=categorical_features,
        free_raw_data=False,
    )

    model = lgb.train(
        params,
        lgb_train,
        valid_sets=[lgb_train, lgb_eval],
        valid_names=["train", "valid"],
        num_boost_round=1000,
        callbacks=[
            lgb.early_stopping(stopping_rounds=30, verbose=True),
            lgb.log_evaluation(period=10),
        ],
    )

    oof_log = model.predict(X_val, num_iteration=model.best_iteration)
    oof_train[valid_index] = np.expm1(oof_log)

    test_log = model.predict(X_test, num_iteration=model.best_iteration)
    y_pred = np.expm1(test_log)

    y_preds.append(y_pred)
    models.append(model)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2132589651.py in <cell line: 0>()
     46     oof_train[valid_index] = np.expm1(oof_log)
     47 
---> 48     test_log = model.predict(X_test, num_iteration=model.best_iteration)
     49     y_pred = np.expm1(test_log)
     50 

/usr/local/lib/python3.11/dist-packages/lightgbm/basic.py in predict(self, data, start_iteration, num_iteration, raw_score, pred_leaf, pred_contrib, data_has_header, validate_features, **kwargs)
   4765             else:
   4766                 num_iteration = -1
-> 4767         return predictor.predict(
   4768             data=data,
   4769             start_iteration=start_iteration,

/usr/local/lib/python3.11/dist-packages/lightgbm/basic.py in predict(self, data, start_iteration, num_iteration, raw_score, pred_leaf, pred_contrib, data_has_header, validate_features)
   1156 
   1157         if isinstance(data, pd_DataFrame):
-> 1158             data = _data_from_pandas(
   1159                 data=data,
   1160                 feature_name="auto",

/usr/local/lib/python3.11/dist-packages/lightgbm/basic.py in _data_from_pandas(data, feature_name, categorical_feature, pandas_categorical)
    832 ) -> Tuple[np.ndarray, List[str], Union[List[str], List[int]], List[List]]:
    833     if len(data.shape) != 2 or data.shape[0] < 1:
--> 834         raise ValueError("Input data must be 2 dimensional and non empty.")
    835 
    836     # take shallow copy in case we modify categorical columns

ValueError: Input data must be 2 dimensional and non empty.

## === cell 9
pd.DataFrame(oof_train, columns=["oof_pred"]).to_csv("oof_train_kfold.csv", index=False)

scores = [m.best_score["valid"]["rmse"] for m in models]
score = float(sum(scores) / len(scores))
print("===CV scores (rmse on log1p(fare)) ===")
print(scores)
print(score)



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
ZeroDivisionError                         Traceback (most recent call last)
/tmp/ipykernel_11/2866894169.py in <cell line: 0>()
      2 
      3 scores = [m.best_score["valid"]["rmse"] for m in models]
----> 4 score = float(sum(scores) / len(scores))
      5 print("===CV scores (rmse on log1p(fare)) ===")
      6 print(scores)

ZeroDivisionError: division by zero

## === cell 10
from sklearn.metrics import mean_squared_error

y_pred_oof = oof_train
rmse = float(np.sqrt(mean_squared_error(y_train, y_pred_oof)))
rmse



## === cell 11
len(y_preds)



## === cell 12
y_preds[0][:10]



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
IndexError                                Traceback (most recent call last)
/tmp/ipykernel_11/3456157511.py in <cell line: 0>()
----> 1 y_preds[0][:10]
      2 

IndexError: list index out of range

## === cell 13
y_sub = sum(y_preds) / len(y_preds)

y_sub = np.clip(y_sub.astype(float), 0.0, 500.0)

y_sub[:10]



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
ZeroDivisionError                         Traceback (most recent call last)
/tmp/ipykernel_11/2124624897.py in <cell line: 0>()
----> 1 y_sub = sum(y_preds) / len(y_preds)
      2 
      3 y_sub = np.clip(y_sub.astype(float), 0.0, 500.0)
      4 
      5 y_sub[:10]

ZeroDivisionError: division by zero

## === cell 14
sub_lgb = pd.DataFrame(
    {
        "key": test["key"].astype(str).values,
        "fare_amount": y_sub.astype(float),
    }
)

sub_lgb.to_csv("submission.csv", index=False)
sub_lgb.to_csv("submission_lightgbm.csv", index=False)

sub_lgb.head()

## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/924326669.py in <cell line: 0>()
      2     {
      3         "key": test["key"].astype(str).values,
----> 4         "fare_amount": y_sub.astype(float),
      5     }
      6 )

NameError: name 'y_sub' is not defined

## --- ERROR in outputing the csv:
Invalid submission: Submission should have a fare_amount column
