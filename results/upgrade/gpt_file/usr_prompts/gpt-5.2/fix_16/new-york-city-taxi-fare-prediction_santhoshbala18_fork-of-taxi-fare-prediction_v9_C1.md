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

3.35291

# 6. Current score

6.07091

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 7.4263) has done: 'Your current score is far worse than the target (RMSE 8.96 vs 3.35), so we should make small but high-impact correctness fixes rather than tuning for the best possible model. The biggest issue is feature mismatch: you train with `is_night/is_surge/is_airport` but later drop some columns only for the train/valid split (not for the fitted model) and you also never add `is_BeforeRaise` to training, so test-time columns don’t match what the model expects. I (1) create the exact same feature set for train/valid/test, (2) fix the incorrect airport longitude sign (NYC longitudes are negative), and (3) add a simple, standard fare cleanup (remove extreme fares) to stabilize RMSE—without changing the model type or training loop. The script still train an `XGBRegressor` and write a valid `submission_combo.csv`.'
- What this solution (achieved 5.40344) has done: 'Your current issue is that you can’t reliably measure/iterate because the run errors out before producing a graded result, so the most helpful “score toward target” move is to make the pipeline deterministic, valid, and metric-aligned while keeping the same XGBRegressor training loop and feature logic. I keep your exact feature set and model, but (1) ensure train/test use identical feature columns in the same order and with safe numeric dtypes, (2) add a minimal, standard post-processing step to clip negative predictions (invalid for fares) which usually improves RMSE without changing core modeling, and (3) write the submission to both `submission_combo.csv` and `/kaggle/working/submission_combo.csv` to ensure the file is found by graders that expect it in the working directory. These are minimal, semantics-preserving fixes that should move RMSE down toward the target band without changing the architecture or training approach.'
- What this solution (achieved 5.34723) has done: 'Your current RMSE (5.403) is worse than the target (3.353), so we should make small, reliable improvements without changing the model type or overall training flow. The biggest gain available with minimal change is to train the same XGBRegressor on a log-transformed target (still RMSE-optimized via squared error, but it reduces the influence of outliers), then invert the transform for predictions; this typically improves Taxi Fare RMSE substantially. I also switch the train/validation split to be stratified by distance bins (same split method and loop, but less distribution shift), and align XGBoost’s objective/eval_metric explicitly with RMSE for stability. Submission format and file paths remain unchanged, and we still clip negative fares to 0.'
- What this solution (achieved 6.04059) has done: 'You’re already training a solid XGBRegressor, but your RMSE (5.35) is still far from the target (3.35), so the biggest “minimal change” gains come from (1) slightly better feature signal without changing the model/training loop, and (2) stronger, standard NYC Taxi cleanup that removes impossible coordinates and extreme-distance artifacts that inflate RMSE. I add two classic, lightweight geospatial features (Haversine distance in NYC plus Manhattan distance proxy) while keeping your existing `distance_km` and all flags, and I tighten the data filtering with known-valid longitude/latitude ranges and a fare-per-km sanity filter. This preserves your overall approach (same model type, objective, training flow, log1p target) but typically moves RMSE materially downward. Submission writing stays identical and still produces `submission_combo.csv`.'
- What this solution (achieved 6.11378) has done: 'You’re well worse than the target (RMSE 6.04 vs 3.35, lower is better), so the smallest high-impact move is to fix metric alignment: Kaggle evaluates RMSE on raw dollars, but your model is trained on `log1p(fare)` with plain squared error, which is not the same objective and can hurt RMSE even if it helps relative-error. I keep the same XGBRegressor training flow and all your features/cleaning, but train directly on `fare_amount` (no log transform) and predict in dollars. To stabilize RMSE without changing the approach, I also add a single, standard sanity clip on extreme predictions (0..250) consistent with your training filter and keep the submission format identical.'
- What this solution (achieved 6.08676) has done: 'Your RMSE (6.11) is still far from the target (3.35), so the most “minimal but high-impact” move is to fix underfitting without changing your core approach (same features, same XGBRegressor, same train/fit/predict flow). I only tune XGBoost capacity in a standard way for this competition: increase `n_estimators` while lowering `learning_rate`, add `subsample/colsample_bytree` for robustness, and keep the same squared-error objective/metric. I also train with an `eval_set` so the model reports validation RMSE consistently (no early stopping added). Submission creation/format stays identical.'
- What this solution (achieved 5.98441) has done: 'I make two minimal, high-impact fixes that typically move NYC Taxi Fare RMSE down without changing your overall approach (same features, same XGBRegressor fit/predict flow). First, I replace the random split with a time-based split using `pickup_datetime`, which better matches the competition’s temporal distribution and reduces validation/public LB mismatch. Second, I use a small, deterministic sample-weighting scheme that down-weights extremely short trips (which are disproportionately noisy under your current filters) while keeping the same objective/loss and model architecture. Everything else (feature engineering, cleaning rules, XGBRegressor, clipping, and submission writing) stays the same.'
- What this solution (achieved 6.02239) has done: 'To move RMSE down toward your 3.35 target without changing the core model/training flow, I keep your exact XGBRegressor and feature set, but fix two high-impact correctness issues and one metric-aligned stability improvement. First, I remove the distance-based sample weighting (it currently down-weights longer trips where distance features are most informative, which tends to hurt RMSE); training be unweighted but otherwise identical. Second, I strengthen data cleaning in a standard, minimal way by also filtering on “fare per km” using your already-computed `distance_km` (in addition to `haversine_km`), which reduces noisy/outlier labels that inflate RMSE. Finally, I ensure the feature column order is identical between train and test by explicitly reindexing test features to `FEATURE_COLS`.'
- What this solution (achieved 6.02701) has done: 'Your current RMSE (6.02) is much worse than the 3.35 target, so the smallest high-impact move is to fix a known distribution mismatch bug: you compute time features from `pickup_datetime` but do not handle UTC (the NYC Taxi dataset timestamps are UTC), which shifts hour/weekday features and makes `is_night/is_surge` noisy; converting to NYC local time usually reduces RMSE materially without changing the model. I keep your exact feature set, cleaning logic, and XGBRegressor training loop, and only adjust datetime parsing to be timezone-aware and then convert to `America/New_York` before extracting hour/day/month/weekday/year. I also ensure train/test datetime parsing is consistent and stable by using `errors="coerce"` and dropping rows with invalid datetimes in train (rare but harmful). Submission writing remains identical and still produces `submission_combo.csv`.'
- What this solution (achieved 6.03185) has done: 'Your RMSE (6.03) is still far worse than the 3.35 target (lower is better), so we should make a small, high-impact feature/correctness improvement without changing your core model or training flow. The biggest missing signal here is that NYC taxi fares have strong spatial structure relative to Manhattan/airports; adding two standard “anchor distance” features (distance to Times Square/Manhattan center for pickup/dropoff) typically reduces RMSE materially while keeping the same XGBRegressor and same fit/predict loop. I also add one minimal data-cleaning filter to remove implausible “very long trip but tiny fare” label noise that disproportionately hurts RMSE. Everything else (existing features, timezone handling, split style, XGB params, prediction clipping, and submission writing) stays the same.'
- What this solution (achieved 5.89139) has done: 'I make two minimal, metric-aligned changes that usually reduce RMSE for this competition without changing your core XGBRegressor workflow. First, I train on the traditional NYC Taxi “log1p(fare_amount)” target and then invert with `expm1` at prediction time, which reduces the impact of label outliers while still using the same squared-error objective and training loop. Second, I add a tiny, standard coordinate-cleaning rule to drop rows with zero coordinates (a common noise source) while keeping your existing bounding-box and fare/distance filters intact. Submission columns, feature pipeline, model type/params, and file outputs remain the same.'
- What this solution (achieved 6.03185) has done: 'Your RMSE (5.89) is still much worse than the 3.35 target, so we should make a small, high-impact improvement without changing the overall XGBRegressor workflow or feature set. The biggest low-risk gain here is to stop training on `log1p(fare)` (which misaligns with Kaggle’s RMSE in dollars) and instead train directly on `fare_amount`, while keeping the same model type/params, same features, and same cleaning. To keep predictions valid and stable, we keep the same clipping to `[0, 250]` but remove the `expm1` inverse transform. Everything else (data reading, feature engineering, split, and submission writing) stays the same and still produces `submission_combo.csv`.'
- What this solution (achieved 6.03185) has done: 'Your current RMSE (6.03) is far above the target (3.35), so the most reliable way to move toward the target with minimal change is to correct a key data issue: training on a random 1M slice of the raw file includes many malformed/outlier rows that your current filters don’t fully remove, and this inflates RMSE. I keep your exact model (XGBRegressor), training flow, and existing features/flags, but add two standard NYC Taxi “sanity” filters that remove the biggest label/geo noise sources: (1) drop rows with missing coordinates/datetime before feature engineering, and (2) drop rows where pickup/dropoff are identical (near-zero trip) unless fare is small. I also ensure the same cleaned dtype/column order is applied consistently to train/valid/test (no semantic change, just prevents silent feature issues). These changes typically reduce RMSE materially without changing your core approach.'
- What this solution (achieved 6.07091) has done: 'Your RMSE (6.03) is far worse than the 3.35 target (lower is better), so the smallest likely win is to remove label noise without touching your model/training loop. I keep your exact XGBRegressor, features, datetime handling, and split, but add two standard NYC-taxi cleaning rules that directly reduce RMSE: (1) drop known-bad “fare equals 0” rows (they are common corruption and you currently keep them), and (2) add a mild upper bound on fare-per-km (your current 50 is very permissive and lets mislabeled trips through). I also apply the same basic NA/zero-coordinate cleanup to the test set (no row dropping) to avoid feature NaNs silently turning into zeros differently than train. This should move the score downward toward the target without changing the core modeling approach.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os
import random

print(os.listdir("../input"))
random.seed(113)
np.random.seed(113)

df = pd.read_csv("../input/train.csv", nrows=10**6)
test_set = pd.read_csv("../input/test.csv")




## === cell 1
def distance(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295  # Pi/180
    a = (
        0.5
        - np.cos((lat2 - lat1) * p) / 2
        + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    )
    return 12742 * np.arcsin(np.sqrt(a))  # 2*R*asin...


def add_travel_vector_features(dfx):
    dfx["abs_diff_longitude"] = (dfx.dropoff_longitude - dfx.pickup_longitude).abs()
    dfx["abs_diff_latitude"] = (dfx.dropoff_latitude - dfx.pickup_latitude).abs()
    return dfx


def add_geo_features(dfx):
    dfx["haversine_km"] = distance(
        dfx.pickup_latitude,
        dfx.pickup_longitude,
        dfx.dropoff_latitude,
        dfx.dropoff_longitude,
    )
    dfx["manhattan_km"] = distance(
        dfx.pickup_latitude,
        dfx.pickup_longitude,
        dfx.pickup_latitude,
        dfx.dropoff_longitude,
    ) + distance(
        dfx.pickup_latitude,
        dfx.pickup_longitude,
        dfx.dropoff_latitude,
        dfx.pickup_longitude,
    )
    return dfx


def add_anchor_distance_features(dfx):
    ts_lat, ts_lon = 40.7580, -73.9855
    dfx["pickup_to_ts_km"] = distance(
        dfx.pickup_latitude, dfx.pickup_longitude, ts_lat, ts_lon
    )
    dfx["dropoff_to_ts_km"] = distance(
        dfx.dropoff_latitude, dfx.dropoff_longitude, ts_lat, ts_lon
    )

    mh_lat, mh_lon = 40.7831, -73.9712
    dfx["pickup_to_mh_km"] = distance(
        dfx.pickup_latitude, dfx.pickup_longitude, mh_lat, mh_lon
    )
    dfx["dropoff_to_mh_km"] = distance(
        dfx.dropoff_latitude, dfx.dropoff_longitude, mh_lat, mh_lon
    )
    return dfx


df["distance_km"] = distance(
    df.pickup_latitude, df.pickup_longitude, df.dropoff_latitude, df.dropoff_longitude
)
test_set["distance_km"] = distance(
    test_set.pickup_latitude,
    test_set.pickup_longitude,
    test_set.dropoff_latitude,
    test_set.dropoff_longitude,
)

df = add_travel_vector_features(df)
test_set = add_travel_vector_features(test_set)

df = add_geo_features(df)
test_set = add_geo_features(test_set)

df = add_anchor_distance_features(df)
test_set = add_anchor_distance_features(test_set)



## === cell 2
BB = (-75, -73, 40, 41.5)


def select_within_boundingbox(dfx, BB):
    return (
        (dfx.pickup_longitude >= BB[0])
        & (dfx.pickup_longitude <= BB[1])
        & (dfx.pickup_latitude >= BB[2])
        & (dfx.pickup_latitude <= BB[3])
        & (dfx.dropoff_longitude >= BB[0])
        & (dfx.dropoff_longitude <= BB[1])
        & (dfx.dropoff_latitude >= BB[2])
        & (dfx.dropoff_latitude <= BB[3])
    )


print("Old size: %d" % len(df))

df = df.dropna(
    subset=[
        "fare_amount",
        "pickup_datetime",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
    ]
).copy()

df = df[select_within_boundingbox(df, BB)]
df = df[(df.passenger_count > 0) & (df.passenger_count <= 6)]

df = df[
    ~(
        (df["pickup_longitude"] == 0)
        | (df["pickup_latitude"] == 0)
        | (df["dropoff_longitude"] == 0)
        | (df["dropoff_latitude"] == 0)
    )
].copy()

print("New size: %d" % len(df))




## === cell 3
def add_datetime_info(dataset):
    dataset["pickup_datetime"] = pd.to_datetime(
        dataset["pickup_datetime"], errors="coerce", utc=True
    )
    dataset["pickup_datetime"] = (
        dataset["pickup_datetime"]
        .dt.tz_convert("America/New_York")
        .dt.tz_localize(None)
    )
    dataset["hour"] = dataset.pickup_datetime.dt.hour
    dataset["day"] = dataset.pickup_datetime.dt.day
    dataset["month"] = dataset.pickup_datetime.dt.month
    dataset["weekday"] = dataset.pickup_datetime.dt.weekday
    dataset["year"] = dataset.pickup_datetime.dt.year
    return dataset


df = add_datetime_info(df)
test_set = add_datetime_info(test_set)

before = len(df)
df = df[df["pickup_datetime"].notna()].copy()
after = len(df)
if after != before:
    print(f"Dropped {before-after} rows with invalid pickup_datetime after parsing.")

test_set["pickup_datetime"] = test_set["pickup_datetime"].fillna(
    pd.Timestamp("2009-01-01")
)




## === cell 4
def add_domain_flags(dfx):
    dfx["is_night"] = np.where(
        (
            ((dfx["hour"] >= 20) & (dfx["hour"] <= 23))
            | ((dfx["hour"] >= 0) & (dfx["hour"] < 6))
        ),
        1,
        0,
    )

    def in_box(lon, lat, lon_min, lon_max, lat_min, lat_max):
        return (lon >= lon_min) & (lon <= lon_max) & (lat >= lat_min) & (lat <= lat_max)

    jfk_pick = in_box(
        dfx["pickup_longitude"], dfx["pickup_latitude"], -73.83, -73.74, 40.61, 40.67
    )
    jfk_drop = in_box(
        dfx["dropoff_longitude"], dfx["dropoff_latitude"], -73.83, -73.74, 40.61, 40.67
    )
    lga_pick = in_box(
        dfx["pickup_longitude"], dfx["pickup_latitude"], -73.90, -73.85, 40.76, 40.79
    )
    lga_drop = in_box(
        dfx["dropoff_longitude"], dfx["dropoff_latitude"], -73.90, -73.85, 40.76, 40.79
    )

    dfx["is_airport"] = np.where((jfk_pick | jfk_drop | lga_pick | lga_drop), 1, 0)

    dfx["is_surge"] = np.where(
        (
            ((dfx["hour"] >= 16) & (dfx["hour"] < 20))
            & ((dfx["weekday"] != 5) & (dfx["weekday"] != 6))
        ),
        1,
        0,
    )

    dfx["is_BeforeRaise"] = np.where(
        (dfx["year"] < 2012) | ((dfx["year"] == 2012) & (dfx["month"] < 8)),
        1,
        0,
    )
    return dfx


df = add_domain_flags(df)
test_set = add_domain_flags(test_set)



## === cell 5
df = df.drop(df[df["fare_amount"] < 0].index, axis=0)
df = df[(df["fare_amount"] > 0) & (df["fare_amount"] <= 250)].copy()

df = df[(df["distance_km"] > 0) & (df["distance_km"] <= 100)].copy()
df = df[~((df["distance_km"] < 0.05) & (df["fare_amount"] > 15))].copy()

same_loc = (df["abs_diff_longitude"] < 1e-6) & (df["abs_diff_latitude"] < 1e-6)
df = df[~(same_loc & (df["fare_amount"] > 7.5))].copy()

df = df[(df["abs_diff_longitude"] <= 2.0) & (df["abs_diff_latitude"] <= 2.0)].copy()

df = df[
    (df["pickup_longitude"].between(-74.3, -72.9))
    & (df["dropoff_longitude"].between(-74.3, -72.9))
    & (df["pickup_latitude"].between(40.4, 41.0))
    & (df["dropoff_latitude"].between(40.4, 41.0))
].copy()

fpk_hav = df["fare_amount"] / (df["haversine_km"] + 1e-3)
df = df[fpk_hav.between(1.0, 30.0)].copy()

fpk_dist = df["fare_amount"] / (df["distance_km"] + 1e-3)
df = df[fpk_dist.between(1.0, 30.0)].copy()

df = df[~((df["haversine_km"] > 20.0) & (df["fare_amount"] < 10.0))].copy()



## === cell 6
from sklearn.model_selection import train_test_split

DROP_COLS = ["key", "fare_amount", "pickup_datetime"]
FEATURE_COLS = [c for c in df.columns if c not in DROP_COLS]

X = df[FEATURE_COLS].copy()
y = df["fare_amount"].astype(np.float32)

for c in X.columns:
    X[c] = pd.to_numeric(X[c], errors="coerce")
X = X.fillna(0.0).astype(np.float32)

df_dt = df["pickup_datetime"]
cutoff = df_dt.quantile(0.90)
train_mask = df_dt <= cutoff

X_train = X.loc[train_mask].copy()
y_train = y.loc[train_mask].copy()
X_test = X.loc[~train_mask].copy()
y_test = y.loc[~train_mask].copy()

w_train = None



## === cell 7
y_train.mean()



## === cell 8
from xgboost import XGBRegressor

regressor = XGBRegressor(
    max_depth=10,
    learning_rate=0.05,
    n_estimators=1200,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=1,
    reg_lambda=1.0,
    random_state=113,
    n_jobs=4,
    objective="reg:squarederror",
    eval_metric="rmse",
    tree_method="hist",
)
regressor.fit(
    X_train, y_train, sample_weight=w_train, eval_set=[(X_test, y_test)], verbose=False
)



## === cell 9
predictions = regressor.predict(X_test)
predictions = np.clip(predictions, 0.0, 250.0)



## === cell 10
import math
from sklearn.metrics import mean_squared_error

rmse = math.sqrt(mean_squared_error(y_test, predictions))
print(rmse)



## === cell 11
_ = None



## === cell 12
X_train.columns



## === cell 13
"""from keras import Sequential
from keras import layers
from keras.layers import Dense,Dropout
from keras.callbacks import ModelCheckpoint,  ReduceLROnPlateau
from keras import optimizers
from keras.regularizers import l2

neuralNetwork = Sequential()
neuralNetwork.add(Dense(units=1048,input_shape=(8,),activation='relu',kernel_regularizer = l2(1e-2)))
neuralNetwork.add(Dropout(0.5))
neuralNetwork.add(Dense(units=10,activation='relu',kernel_regularizer = l2(1e-2)))
neuralNetwork.add(Dropout(0.5))
neuralNetwork.add(Dense(units=1,activation='relu',kernel_regularizer = l2(1e-2)))

neuralNetwork.compile(optimizer='Adam', 
              loss='mean_squared_error')

filepath = './model_weights/weights-improvement-10M.hdf5'
best_callback = ModelCheckpoint(filepath, 
                                save_best_only=True)

history = neuralNetwork.fit(X_train, y_train, 
          epochs=20,
          verbose=0,
          batch_size=2048)

y_pred = neuralNetwork.predict(X_test)
rmse = math.sqrt(mean_squared_error(y_test,y_pred))

print(rmse)
"""



## === cell 14
test_set_features = test_set.reindex(columns=FEATURE_COLS).copy()
test_set_key = test_set["key"].copy()

for c in test_set_features.columns:
    test_set_features[c] = pd.to_numeric(test_set_features[c], errors="coerce")

test_set_features = test_set_features.fillna(0.0).astype(np.float32)

y_pred_reg = regressor.predict(test_set_features)
y_pred_reg = np.clip(y_pred_reg, 0.0, 250.0)



## === cell 15
"""test_set_features = test_set_features.drop(['abs_diff_longitude', 'abs_diff_latitude', 'year','is_airport'],axis=1)
y_pred_neuralNet = neuralNetwork.predict(test_set_features)
y_pred_neuralNet = y_pred_neuralNet.reshape(9914,)
y_pred_final = (y_pred_reg+y_pred_neuralNet)/2"""



## === cell 16
submission = pd.DataFrame(
    {"key": test_set_key, "fare_amount": y_pred_reg}, columns=["key", "fare_amount"]
)
submission = submission.round({"fare_amount": 2})

submission.to_csv("submission_combo.csv", index=False)
try:
    os.makedirs("/kaggle/working", exist_ok=True)
    submission.to_csv("/kaggle/working/submission_combo.csv", index=False)
except Exception as e:
    print("Could not also write to /kaggle/working:", repr(e))

print(submission.head())
print("Submitted")



## === cell 17
submission.shape
