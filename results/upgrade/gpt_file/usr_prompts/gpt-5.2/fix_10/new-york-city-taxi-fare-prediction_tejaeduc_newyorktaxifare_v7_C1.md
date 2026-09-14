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

3.21447

# 6. Current score

5.25429

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 7.37431) has done: 'I fix the Haversine distance feature to work on NumPy arrays (your current `math.radians/sin/cos` only accept scalars), which unblocks feature creation for both train and test. Then I fix the XGBoost feature mismatch by ensuring the test DMatrix is built from the exact same feature columns used for training (the error shows `fare_amount` accidentally ended up in the test matrix). Finally, I make sure a valid submission CSV with the required `key,fare_amount` header is written end-to-end.'
- What this solution (achieved 8.18774) has done: 'Your current RMSE (7.37431) is far worse than the target (3.21447), so we should legitimately improve the model with minimal, metric-aligned changes. The biggest win without changing the overall approach is to train XGBoost on `log1p(fare_amount)` (standard for this competition to handle heavy-tailed fares) and then invert with `expm1` at prediction time—this keeps the same model family/training loop/features but usually drops RMSE substantially. I also remove early stopping (it can stop too early and underfit here) while keeping the same boosting approach/params structure, and I clip negative predictions to 0 to avoid RMSE penalties from impossible fares. The script still writes a valid `key,fare_amount` submission CSV end-to-end.'
- What this solution (achieved 8.32699) has done: 'Your current score (8.18774 RMSE) is much worse than the target (3.21447), so we should legitimately improve generalization with minimal, metric-aligned changes while keeping the same XGBoost/regression workflow and features. The biggest low-risk fix is to strengthen the XGBoost baseline by adding a few standard tree/regularization parameters (depth, subsampling, learning rate) that typically reduce RMSE substantially without changing the approach. I also add a small amount of training data (increase `nrows`) since 1.0M can underfit this problem, and switch to `tree_method='hist'` for speed so the run still fits the time budget. Finally, I keep the existing log1p target transform/inversion and ensure the submission CSV format stays exactly `key,fare_amount`.'
- What this solution (achieved 5.7545) has done: 'Your current RMSE (8.32699) is far worse than the target (3.21447), so we should make small, standard, metric-aligned improvements without changing the overall XGBoost + engineered-features approach. The biggest low-risk gain here is to use a more appropriate objective for the log-transformed target (`reg:squaredlogerror`) so training better matches the heavy-tailed fare distribution while keeping the same log1p/expm1 semantics. I also add two very common NYC-taxi features (Manhattan distance and coordinate sums/means) plus modest regularization (`gamma`, `alpha`) to reduce overfitting, and slightly increase boosting rounds while lowering `eta` to improve fit without changing the training loop. Finally, I keep the submission format identical and still write `finaloutput.csv`.'
- What this solution (achieved 5.49763) has done: 'Your current RMSE (5.7545) is still far above the target (3.21447), so we should make small, metric-aligned improvements without changing the overall XGBoost + engineered-features workflow. The biggest likely gain with minimal disruption is to add two standard, cheap geospatial features (bearing and Haversine-distance squared) and a few simple time features (hour/month) that strongly correlate with fare, while keeping the same cleaning rules, log1p target transform, and XGBoost training loop. I also add a tiny amount of stability via a fixed `verbosity` and a mild `max_delta_step` (common for this dataset) to reduce occasional extreme predictions, without changing the approach. The script still runs end-to-end and writes a valid `finaloutput.csv` with `key,fare_amount`.'
- What this solution (achieved 4.84876) has done: 'You’re still far above the target RMSE (5.50 vs 3.21, lower is better), so we need a real but minimal modeling improvement without changing the overall XGBoost-on-engineered-features approach. The smallest high-impact change here is to add a few standard NYC-taxi location priors (pickup/dropoff distance to Manhattan center + simple interaction terms) and a couple of robust time features (dayofweek and weekend flag) while keeping your existing cleaning, log1p target transform, and XGBoost training loop intact. I also increase training rows moderately (2M → 3M) to reduce underfitting while staying within the 600s budget using `tree_method='hist'`. Submission format and file writing remain identical (`finaloutput.csv` with `key,fare_amount`).'
- What this solution (achieved 4.93528) has done: 'Your current RMSE (4.84876) is still well above the target (3.21447, lower is better), so we should make a small, standard improvement that doesn’t change the overall “engineered features + XGBoost regression” approach. The lowest-risk high-impact change for this competition is to add two classic geospatial priors: the trip direction components (Δlon, Δlat) and the straight-line Euclidean distance in coordinate space, which help the model learn fare structure beyond Haversine alone. I also add a couple of very cheap time features (minute and day-of-year) that often capture rush-hour/seasonality effects better than hour/day alone. Everything else (data cleaning, log1p/expm1 target handling, XGBoost training loop/params style, and submission writing) stays the same.'
- What this solution (achieved 5.25429) has done: 'We need to move RMSE down from 4.94 toward 3.21, so the smallest legitimate gain without changing your overall “engineered features + XGBoost” approach is to (1) add a single high-signal, cheap geospatial prior: the haversine distance to JFK and LaGuardia for both pickup and dropoff, and (2) slightly tighten data cleaning to remove obvious “zero-distance but non-trivial fare” noise that tends to hurt RMSE. These changes keep the same feature-engineering style (just additional distance features using your existing haversine), the same log1p target handling and training loop, and still write the same valid `key,fare_amount` submission. Everything else is left intact to minimize risk and runtime.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from sklearn.model_selection import train_test_split
import xgboost
from sklearn.preprocessing import StandardScaler

import os

print(os.listdir("../input"))



## === cell 1
df = pd.read_csv("../input/train.csv", nrows=3000000)



## === cell 2
test = pd.read_csv("../input/test.csv")



## === cell 3
testkey = test.key



## === cell 4
df = df.dropna(how="any", axis="rows")



## === cell 5
len(df)



## === cell 6
df.head()



## === cell 7
df.describe()



## === cell 8
l = df[
    (df.pickup_latitude > 42.5)
    | (df.pickup_latitude < 40.0)
    | (df.dropoff_latitude > 42.5)
    | (df.dropoff_latitude < 40.0)
    | (df.pickup_longitude > -73.0)
    | (df.pickup_longitude < -75.0)
    | (df.dropoff_longitude > -73.0)
    | (df.dropoff_longitude < -75.0)
].index



## === cell 9
df = df.drop(l, axis=0)



## === cell 10
z = df[
    (df.fare_amount > 350.0)
    | (df.fare_amount < 0.0)
    | (df.passenger_count > 7.0)
    | (df.passenger_count < 0.0)
].index



## === cell 11
df = df.drop(z, axis=0)



## === cell 12
len(df)




## === cell 13
def distlatlong(lon1, lat1, lon2, lat2):
    lon1 = np.asarray(lon1, dtype="float64")
    lat1 = np.asarray(lat1, dtype="float64")
    lon2 = np.asarray(lon2, dtype="float64")
    lat2 = np.asarray(lat2, dtype="float64")

    lat1 = np.deg2rad(lat1)
    lat2 = np.deg2rad(lat2)
    lon1 = np.deg2rad(lon1)
    lon2 = np.deg2rad(lon2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = (np.sin(dlat / 2.0) ** 2) + np.cos(lat1) * np.cos(lat2) * (
        np.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    distance = 6373.0 * c
    return distance




## === cell 14
df["dist"] = distlatlong(
    df["pickup_longitude"].values,
    df["pickup_latitude"].values,
    df["dropoff_longitude"].values,
    df["dropoff_latitude"].values,
)



## === cell 15
test["dist"] = distlatlong(
    test["pickup_longitude"].values,
    test["pickup_latitude"].values,
    test["dropoff_longitude"].values,
    test["dropoff_latitude"].values,
)



## === cell 16
df = df[~((df["dist"] < 0.01) & (df["fare_amount"] > 3.0))].copy()



## === cell 17
test.head()



## === cell 18
df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])



## === cell 19
test["pickup_datetime"] = pd.to_datetime(test["pickup_datetime"])



## === cell 20
df.info()



## === cell 21
df["latenights"] = (df["pickup_datetime"].dt.hour < 5).astype(int)
test["latenights"] = (test["pickup_datetime"].dt.hour < 5).astype(int)



## === cell 22
df.pickup_datetime.iloc[0].weekday()



## === cell 23
df["weekday"] = (df["pickup_datetime"].dt.weekday > 4).astype(int)
test["weekday"] = (test["pickup_datetime"].dt.weekday > 4).astype(int)



## === cell 24
df.head()



## === cell 25
df["year"] = df["pickup_datetime"].dt.year



## === cell 26
test["year"] = test["pickup_datetime"].dt.year



## === cell 27
df["day"] = df["pickup_datetime"].dt.day



## === cell 28
test["day"] = test["pickup_datetime"].dt.day



## === cell 29
df["dayofweek"] = df["pickup_datetime"].dt.dayofweek
test["dayofweek"] = test["pickup_datetime"].dt.dayofweek
df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
test["is_weekend"] = (test["dayofweek"] >= 5).astype(int)

df["hour"] = df["pickup_datetime"].dt.hour
test["hour"] = test["pickup_datetime"].dt.hour
df["month"] = df["pickup_datetime"].dt.month
test["month"] = test["pickup_datetime"].dt.month

df["minute"] = df["pickup_datetime"].dt.minute
test["minute"] = test["pickup_datetime"].dt.minute
df["dayofyear"] = df["pickup_datetime"].dt.dayofyear
test["dayofyear"] = test["pickup_datetime"].dt.dayofyear



## === cell 30
df.head()



## === cell 31
test.head()



## === cell 32
feat = df.drop(["key", "pickup_datetime"], axis=1)
test = test.drop(["key", "pickup_datetime"], axis=1)



## === cell 33
test.year.unique()




## === cell 34
def bearing(lon1, lat1, lon2, lat2):
    lon1 = np.deg2rad(np.asarray(lon1, dtype="float64"))
    lat1 = np.deg2rad(np.asarray(lat1, dtype="float64"))
    lon2 = np.deg2rad(np.asarray(lon2, dtype="float64"))
    lat2 = np.deg2rad(np.asarray(lat2, dtype="float64"))
    dlon = lon2 - lon1
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return np.arctan2(y, x)


MANH_LON = -73.985428
MANH_LAT = 40.748817

JFK_LON, JFK_LAT = -73.7781, 40.6413
LGA_LON, LGA_LAT = -73.8740, 40.7769

for _df in (feat, test):
    _df["abs_lon_diff"] = (_df["pickup_longitude"] - _df["dropoff_longitude"]).abs()
    _df["abs_lat_diff"] = (_df["pickup_latitude"] - _df["dropoff_latitude"]).abs()
    _df["manhattan"] = _df["abs_lon_diff"] + _df["abs_lat_diff"]
    _df["pickup_sum"] = _df["pickup_longitude"] + _df["pickup_latitude"]
    _df["dropoff_sum"] = _df["dropoff_longitude"] + _df["dropoff_latitude"]
    _df["coord_mean"] = (
        _df["pickup_longitude"]
        + _df["pickup_latitude"]
        + _df["dropoff_longitude"]
        + _df["dropoff_latitude"]
    ) / 4.0
    _df["dist_sq"] = _df["dist"] ** 2
    _df["bearing"] = bearing(
        _df["pickup_longitude"].values,
        _df["pickup_latitude"].values,
        _df["dropoff_longitude"].values,
        _df["dropoff_latitude"].values,
    )

    _df["pickup_manh_dist"] = distlatlong(
        _df["pickup_longitude"].values,
        _df["pickup_latitude"].values,
        np.full(len(_df), MANH_LON, dtype="float64"),
        np.full(len(_df), MANH_LAT, dtype="float64"),
    )
    _df["dropoff_manh_dist"] = distlatlong(
        _df["dropoff_longitude"].values,
        _df["dropoff_latitude"].values,
        np.full(len(_df), MANH_LON, dtype="float64"),
        np.full(len(_df), MANH_LAT, dtype="float64"),
    )

    _df["dist_x_passengers"] = _df["dist"] * _df["passenger_count"]
    _df["dist_x_hour"] = _df["dist"] * _df["hour"]

    _df["lon_diff"] = _df["dropoff_longitude"] - _df["pickup_longitude"]
    _df["lat_diff"] = _df["dropoff_latitude"] - _df["pickup_latitude"]
    _df["euclid"] = np.sqrt(_df["lon_diff"] ** 2 + _df["lat_diff"] ** 2)

    _df["pickup_jfk_dist"] = distlatlong(
        _df["pickup_longitude"].values,
        _df["pickup_latitude"].values,
        np.full(len(_df), JFK_LON, dtype="float64"),
        np.full(len(_df), JFK_LAT, dtype="float64"),
    )
    _df["dropoff_jfk_dist"] = distlatlong(
        _df["dropoff_longitude"].values,
        _df["dropoff_latitude"].values,
        np.full(len(_df), JFK_LON, dtype="float64"),
        np.full(len(_df), JFK_LAT, dtype="float64"),
    )
    _df["pickup_lga_dist"] = distlatlong(
        _df["pickup_longitude"].values,
        _df["pickup_latitude"].values,
        np.full(len(_df), LGA_LON, dtype="float64"),
        np.full(len(_df), LGA_LAT, dtype="float64"),
    )
    _df["dropoff_lga_dist"] = distlatlong(
        _df["dropoff_longitude"].values,
        _df["dropoff_latitude"].values,
        np.full(len(_df), LGA_LON, dtype="float64"),
        np.full(len(_df), LGA_LAT, dtype="float64"),
    )



## === cell 35
feat_year = pd.get_dummies(feat["year"], prefix="year")
test_year = pd.get_dummies(test["year"], prefix="year")
feat = pd.concat([feat.drop("year", axis=1), feat_year], axis=1)
test = pd.concat([test.drop("year", axis=1), test_year], axis=1)
feat, test = feat.align(test, join="left", axis=1, fill_value=0)



## === cell 36
feat.head()



## === cell 37
test.head()



## === cell 38
label = feat["fare_amount"].astype("float64")
label_log = np.log1p(label.clip(lower=0.0))



## === cell 39
feat = feat.drop("fare_amount", axis=1)



## === cell 40
xtr, xts, ytr, yts = train_test_split(feat, label_log, test_size=0.2, random_state=42)



## === cell 41
test = test.reindex(columns=feat.columns, fill_value=0)

xgbtrain = xgboost.DMatrix(xtr, label=ytr, feature_names=list(feat.columns))
xgbtest = xgboost.DMatrix(xts, label=yts, feature_names=list(feat.columns))
xgbfinaltest = xgboost.DMatrix(test, feature_names=list(feat.columns))



## === cell 42
params = {
    "eval_metric": "rmse",
    "objective": "reg:squaredlogerror",
    "seed": 42,
    "tree_method": "hist",
    "max_depth": 8,
    "min_child_weight": 1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eta": 0.05,
    "lambda": 1.0,
    "alpha": 0.1,
    "gamma": 0.1,
    "max_delta_step": 1.0,
    "verbosity": 1,
}



## === cell 43
xgbmodel = xgboost.train(
    params,
    dtrain=xgbtrain,
    num_boost_round=1600,
    evals=[(xgbtest, "test")],
)



## === cell 44
pred_log = xgbmodel.predict(xgbfinaltest)



## === cell 45
pred = np.expm1(pred_log)
pred = np.clip(pred, 0.0, None)



## === cell 46
finalset = pd.DataFrame({"key": testkey, "fare_amount": pred})



## === cell 47
finalset = finalset[["key", "fare_amount"]]



## === cell 48
finalset.head()



## === cell 49
finalset.to_csv("finaloutput.csv", index=False)
print("Wrote submission:", os.path.abspath("finaloutput.csv"), "rows:", len(finalset))
