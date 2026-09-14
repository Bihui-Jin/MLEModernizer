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

3.98929

# 6. Current score

5.42185

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 4.62664) has done: 'Diagnosis: Cell 8 crashes because `xgboost.XGBRegressor.predict()` (scikit-learn API) does not accept a `data=` keyword; it expects the feature matrix as the first positional argument (or `X=`). This keyword is used in the low-level `xgb.Booster.predict()` API, not the sklearn wrapper. The fix is to call `predict(X_test)` so the method signature matches xgboost==2.0.3.

Patch summary: Update only the `predict` call in cell 8 to pass `X_test` positionally (no keyword). This preserves the model, training call, and output variable `Y_pred` exactly as expected by cell 9.

Updated cells: (cell 8 only)

Compatibility notes for cell k+1: `Y_pred` remains a 1D numpy array of predictions aligned with `test_data` rows, so `submission['fare_amount']=Y_pred` in cell 9 continues to work unchanged.

Assumptions: `X_test` has the same feature columns as used in training and contains only numeric dtypes compatible with XGBoost.'
- What this solution (achieved 7.33595) has done: 'You’re currently underperforming the target RMSE (4.62664 vs 3.98929; lower is better), so we should make small, low-risk changes that improve generalization without changing the overall approach (still: simple datetime+distance features + XGBoost regressor). The biggest gain with minimal disruption is to (1) train on a bit more data (still within a 600s budget) and (2) remove clearly invalid/outlier rows (bad lat/lon ranges, zero-distance + non-trivial fare, extreme fares), which is standard for this competition and typically reduces RMSE materially. I also ensure `Y_train` is a 1D vector (not a single-column DataFrame) to match XGBoost’s expected target shape and avoid subtle fitting quirks. Submission format and paths remain unchanged, and the output CSV is still `sample_submission.csv` with `key,fare_amount`.'
- What this solution (achieved 7.55917) has done: 'Your current RMSE is worse than the target (7.33595 vs 3.98929; lower is better), so we make minimal, low-risk improvements that usually reduce error in this competition without changing the overall approach (still: simple datetime + distance features + XGBoost regressor). The main issues are (1) the model is using XGBoost defaults (often underfit and poorly regularized) and (2) the distance features are crude (axis-aligned deltas), so we add a haversine distance feature while keeping your existing ones. We also add a couple of very standard cleaning rules (remove extreme outlier fares and implausible coordinates already mostly covered) and ensure consistent dtypes for XGBoost. The submission format/path remains identical and the script still writes `sample_submission.csv`.'
- What this solution (achieved 4.785) has done: 'Your current RMSE (7.55917) is worse than the target (3.98929; lower is better), so we make small, standard NYC taxi-fare improvements without changing the overall approach (same features + XGBoost regressor). The biggest low-risk gain is to train the model in log-space (`log1p(fare_amount)`) and invert with `expm1` at prediction time, which typically reduces the impact of long-tail fares under RMSE. We also add two very common, lightweight time/location features (month and geodesic deltas) and a simple geographic “NYC bounding-box” filter on both train and test to reduce out-of-distribution effects. The script still runs end-to-end and writes a valid `sample_submission.csv` with `key,fare_amount`.'
- What this solution (achieved 4.78814) has done: 'Your RMSE (4.785) is still above the target (3.98929; lower is better), so the smallest safe way to move closer is to (1) remove a known major noise source in this competition (rare-but-huge fares) by training on `log1p(fare_amount)` but evaluating with RMSE, and (2) correct for log-space bias via a light “smearing” correction estimated from in-sample residuals (no new model, no new features, no extra data). This keeps your exact feature set and XGBoost approach unchanged, only adjusts the inversion from log space to dollars in a principled way that typically reduces RMSE. I also add a tiny guard to align columns and handle any NaNs created by datetime parsing so XGBoost won’t silently degrade. The script still runs end-to-end and writes a valid `sample_submission.csv` with `key,fare_amount`.'
- What this solution (achieved 5.5142) has done: 'Your current RMSE (4.78814) is worse than the target (3.98929), so we should make a small, low-risk improvement without changing the overall pipeline (same features and XGBoost on log1p labels). The biggest likely gain with minimal disruption is to stop “zeroing out” bad datetime parses (which injects noise) by dropping those training rows instead, and to add two standard calendar features (year and day-of-month) derived from the same `pickup_datetime` you already use. I also add a tiny filter to remove implausible near-zero-distance rides with non-trivial fares, which commonly reduces noise for this competition while preserving the same modeling approach. Submission format/path remains identical and still writes `sample_submission.csv`.'
- What this solution (achieved 5.58637) has done: 'You’re currently worse than the target RMSE (5.5142 vs 3.98929; lower is better), so we should make the smallest changes that typically reduce noise without changing the overall approach (same features, same XGBoost regressor, still training on log1p and inverting). The main likely issue is remaining label noise from outlier trips that pass the current filters, so we add one extra standard cleaning rule: drop implausible high-fare-per-km rides (using the already-computed haversine distance) and a small minimum-distance guard for non-trivial fares. We keep your model hyperparameters and training procedure unchanged, and we keep the same submission filename/format; this should move RMSE down toward the target without a big behavioral shift. We also ensure the cleaning happens before feature engineering so training/test semantics remain aligned.'
- What this solution (achieved 5.51396) has done: 'Your current RMSE (5.58637) is worse than the target (3.98929; lower is better), so the smallest likely win is to reduce remaining label noise while keeping the same feature set and XGBoost-on-log1p approach unchanged. I add one standard, lightweight cleaning rule that uses your already-computed haversine distance to remove implausible long trips with too-low fares (a common source of bad labels in this dataset), and I apply an aligned “NYC bounding box” filter to the test set to reduce out-of-distribution predictions without changing test rows (we only use it to cap predictions, not drop rows). I also make the training/test datetime-derived columns use consistent small integer dtypes (avoid pandas nullable Int16) to prevent XGBoost from seeing object/nullable types, which can silently hurt performance. The model, training loop, loss/objective, and submission schema/filename remain the same.'
- What this solution (achieved 5.406) has done: 'Your current RMSE (5.51396) is worse than the target (3.98929; lower is better), so we make the smallest change that usually improves this competition without altering your overall approach (same features + XGBoost on log1p labels). The biggest issue is that the training data is still very noisy; we add one standard filter for “minimum fare floor” (NYC-like) and one standard filter for absurdly long trips in the bbox (likely bad coordinates/labels), both of which typically reduce RMSE materially while keeping the same feature engineering and model. We keep your model hyperparameters, log1p+smearing inversion, and submission schema unchanged. We also apply the minimum-fare floor only to the *predictions* (not changing test rows), which often improves RMSE by preventing unrealistically low fares.'
- What this solution (achieved 5.43429) has done: 'To move your RMSE down toward the 3.99 target (current 5.406; lower is better) with minimal disruption, I keep the same feature set, log1p target, and XGBoost regressor approach, but reduce training-label noise that’s still likely dominating error. The smallest high-impact fix is to apply the NYC bounding-box filter to the training set with tighter, competition-standard coordinate bounds (your current [-75,-72]×[40,42] is too loose and lets in many bad points), and to add one more standard cleanup: drop rows where pickup==dropoff but the fare is non-trivial (handles float rounding, not just exact equality). I also cap passenger_count more conservatively (1–6 already) and keep everything else (model params, smearing correction, prediction post-processing, submission format/path) unchanged.'
- What this solution (achieved 5.42272) has done: 'Your current RMSE (5.43429) is above the target (3.98929), so we should make a small, low-risk improvement that typically reduces error for this competition without changing the overall approach (same engineered features + XGBoost regressor trained on log1p(fare)). The main likely issue is that the training-set coordinate bounding box is now too tight and discards many valid trips, which can hurt generalization; we loosen it slightly to a standard NYC-ish box while keeping your other cleaning rules intact. In addition, we apply the same bbox check to the test set and use it only for post-processing (as you already do) but with consistent bounds, and we add a simple per-km fare floor/ceiling only for extreme distances to reduce crazy predictions without changing the model. These are minimal edits confined to cleaning and post-processing and should move RMSE down toward the target.'
- What this solution (achieved 5.41861) has done: 'Your current RMSE (5.42272) is worse than the target (3.98929; lower is better), so we make the smallest likely-positive changes without changing the overall approach (same engineered features + XGBoost on log1p fares). The biggest low-risk issue is inconsistent handling of bad datetimes: train rows with unparseable `pickup_datetime` are dropped, but test rows get filled with zeros; instead we add a boolean `bad_datetime` feature in both train/test so the model can learn a separate behavior for those cases. We also fix a subtle train/test mismatch in `_test_in_box` (it currently uses `test_data` while features come from `X_test`) to keep post-processing aligned, and we add `min_child_weight` (a standard regularization knob) to reduce overfitting with essentially no pipeline change. All paths, output filename, and submission schema remain identical and the script still writes `sample_submission.csv`.'
- What this solution (achieved 5.42185) has done: 'Your RMSE is worse than the target, so the smallest likely-positive move is to reduce avoidable train-label noise and improve XGBoost fit stability without changing the overall approach (same features, same model family, same log1p target + smearing + post-processing). I (1) add one standard NYC taxi cleanup rule that’s missing: remove rows with non-positive `fare_amount` (already mostly handled) and, importantly, remove rows with zero/near-zero coordinates and extreme coordinate values that can still slip through, and (2) add `reg_alpha` (light L1 regularization) plus a conservative `max_bin` to make the hist tree method more stable on this large dataset. I also keep your train/test feature alignment exactly the same and ensure `haversine_km` remains available for your existing post-processing cap. These changes are confined to cleaning + small regularization knobs and are intended to move RMSE down toward the 3.99 target without changing the pipeline’s core logic.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os

print(os.listdir("../input"))



## === cell 1
training_data = pd.read_csv("../input/train.csv", nrows=5_000_000)
test_data = pd.read_csv("../input/test.csv")



## === cell 2
training_data



## === cell 3
X_train = training_data.copy()
X_test = test_data.copy()
Y_train = training_data.copy()




## === cell 4
def _clean_train(df):
    df = df.dropna()

    df = df[(df["fare_amount"] >= 2.5) & (df["fare_amount"] <= 200.0)]
    df = df[(df["passenger_count"] >= 1) & (df["passenger_count"] <= 6)]

    for c in [
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
    ]:
        df = df[
            (
                df[c].between(-180.0, 180.0)
                if "longitude" in c
                else df[c].between(-90.0, 90.0)
            )
        ]
        df = df[df[c] != 0.0]

    df = df[
        (df["pickup_longitude"].between(-74.5, -72.8))
        & (df["dropoff_longitude"].between(-74.5, -72.8))
        & (df["pickup_latitude"].between(40.5, 41.3))
        & (df["dropoff_latitude"].between(40.5, 41.3))
    ]

    lat_dist = (df["dropoff_latitude"] - df["pickup_latitude"]).abs()
    lon_dist = (df["dropoff_longitude"] - df["pickup_longitude"]).abs()
    df = df[~(((lat_dist + lon_dist) < 1e-4) & (df["fare_amount"] > 2.5))]

    return df


training_data = _clean_train(training_data)

X_train = training_data.copy()
Y_train = training_data.copy()




## === cell 5
def _haversine_km(lon1, lat1, lon2, lat2):
    lon1 = np.radians(lon1.astype(np.float64))
    lat1 = np.radians(lat1.astype(np.float64))
    lon2 = np.radians(lon2.astype(np.float64))
    lat2 = np.radians(lat2.astype(np.float64))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * (
        np.sin(dlon / 2.0) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371.0 * c




## === cell 6
_dist_km = _haversine_km(
    X_train["pickup_longitude"].values,
    X_train["pickup_latitude"].values,
    X_train["dropoff_longitude"].values,
    X_train["dropoff_latitude"].values,
).astype(np.float32)

_dist_km_safe = np.maximum(_dist_km, 0.01)  # 10m
_fare = Y_train["fare_amount"].astype(np.float32).values
_fare_per_km = _fare / _dist_km_safe

mask = (_fare_per_km <= 80.0) & ~((_dist_km < 0.2) & (_fare > 30.0))
X_train = X_train.loc[mask].copy()
Y_train = Y_train.loc[X_train.index].copy()

del _dist_km, _dist_km_safe, _fare, _fare_per_km, mask



## === cell 7
_dist_km2 = _haversine_km(
    X_train["pickup_longitude"].values,
    X_train["pickup_latitude"].values,
    X_train["dropoff_longitude"].values,
    X_train["dropoff_latitude"].values,
).astype(np.float32)
_fare2 = Y_train["fare_amount"].astype(np.float32).values

mask2 = ~((_dist_km2 > 30.0) & (_fare2 < 20.0))
X_train = X_train.loc[mask2].copy()
Y_train = Y_train.loc[X_train.index].copy()

del _dist_km2, _fare2, mask2

_dist_km3 = _haversine_km(
    X_train["pickup_longitude"].values,
    X_train["pickup_latitude"].values,
    X_train["dropoff_longitude"].values,
    X_train["dropoff_latitude"].values,
).astype(np.float32)
mask3 = _dist_km3 <= 60.0
X_train = X_train.loc[mask3].copy()
Y_train = Y_train.loc[X_train.index].copy()
del _dist_km3, mask3

X_train["pickup_datetime"] = pd.to_datetime(
    X_train["pickup_datetime"], utc=True, errors="coerce"
)
X_train["bad_datetime"] = X_train["pickup_datetime"].isna().astype(np.int8)

X_train = X_train.dropna(subset=["pickup_datetime"]).copy()
Y_train = Y_train.loc[X_train.index].copy()

X_train["hour"] = X_train["pickup_datetime"].dt.hour.astype(np.int16)
X_train["dayofweek"] = X_train["pickup_datetime"].dt.dayofweek.astype(np.int16)
X_train["month"] = X_train["pickup_datetime"].dt.month.astype(np.int16)
X_train["day"] = X_train["pickup_datetime"].dt.day.astype(np.int16)
X_train["year"] = X_train["pickup_datetime"].dt.year.astype(np.int16)

X_train["latitude_distance"] = (
    (X_train["dropoff_latitude"] - X_train["pickup_latitude"]).abs().astype(np.float32)
)
X_train["longitude_distance"] = (
    (X_train["dropoff_longitude"] - X_train["pickup_longitude"])
    .abs()
    .astype(np.float32)
)

X_train["delta_lat"] = (
    X_train["dropoff_latitude"] - X_train["pickup_latitude"]
).astype(np.float32)
X_train["delta_lon"] = (
    X_train["dropoff_longitude"] - X_train["pickup_longitude"]
).astype(np.float32)

X_train["haversine_km"] = _haversine_km(
    X_train["pickup_longitude"].values,
    X_train["pickup_latitude"].values,
    X_train["dropoff_longitude"].values,
    X_train["dropoff_latitude"].values,
).astype(np.float32)

X_train["_tiny_trip"] = X_train["haversine_km"] < 0.05  # ~50m
mask = ~(X_train["_tiny_trip"] & (Y_train["fare_amount"].astype(np.float32) > 7.0))
X_train = X_train.loc[mask].copy()
Y_train = Y_train.loc[X_train.index].copy()
X_train = X_train.drop(columns=["_tiny_trip"])

X_train = X_train.drop(
    columns=[
        "dropoff_longitude",
        "dropoff_latitude",
        "key",
        "fare_amount",
        "pickup_datetime",
    ]
)

X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0.0)



## === cell 8
X_test["pickup_datetime"] = pd.to_datetime(
    X_test["pickup_datetime"], utc=True, errors="coerce"
)
X_test["bad_datetime"] = X_test["pickup_datetime"].isna().astype(np.int8)

X_test["hour"] = X_test["pickup_datetime"].dt.hour.fillna(0).astype(np.int16)
X_test["dayofweek"] = X_test["pickup_datetime"].dt.dayofweek.fillna(0).astype(np.int16)
X_test["month"] = X_test["pickup_datetime"].dt.month.fillna(0).astype(np.int16)
X_test["day"] = X_test["pickup_datetime"].dt.day.fillna(0).astype(np.int16)
X_test["year"] = X_test["pickup_datetime"].dt.year.fillna(0).astype(np.int16)

X_test["latitude_distance"] = (
    (X_test["dropoff_latitude"] - X_test["pickup_latitude"]).abs().astype(np.float32)
)
X_test["longitude_distance"] = (
    (X_test["dropoff_longitude"] - X_test["pickup_longitude"]).abs().astype(np.float32)
)

X_test["delta_lat"] = (X_test["dropoff_latitude"] - X_test["pickup_latitude"]).astype(
    np.float32
)
X_test["delta_lon"] = (X_test["dropoff_longitude"] - X_test["pickup_longitude"]).astype(
    np.float32
)

X_test["haversine_km"] = _haversine_km(
    X_test["pickup_longitude"].values,
    X_test["pickup_latitude"].values,
    X_test["dropoff_longitude"].values,
    X_test["dropoff_latitude"].values,
).astype(np.float32)

_test_in_box = (
    (X_test["pickup_longitude"].between(-74.5, -72.8))
    & (X_test["dropoff_longitude"].between(-74.5, -72.8))
    & (X_test["pickup_latitude"].between(40.5, 41.3))
    & (X_test["dropoff_latitude"].between(40.5, 41.3))
).values

X_test = X_test.drop(
    columns=["dropoff_longitude", "dropoff_latitude", "key", "pickup_datetime"]
)

X_test = X_test.reindex(columns=X_train.columns, fill_value=0.0)
X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0.0)



## === cell 9
Y_train = Y_train["fare_amount"].astype(np.float32)
Y_train_log = np.log1p(Y_train).astype(np.float32)

"""
from keras import models
from keras import layers
from keras import optimizers
from keras.layers import Dropout

model=models.Sequential()
model.add(layers.Dense(512,activation='relu',input_shape=(X_train.shape[1],)))
model.add(Dropout(0.2))
model.add(layers.Dense(512,activation='relu'))
model.add(Dropout(0.2))
model.add(layers.Dense(1))

rmsprop=optimizers.RMSprop(lr=0.001)

model.compile(optimizer=rmsprop,loss='mse',metrics=['mae'])

model.fit(X_train,Y_train,epochs=4,batch_size=512)

Y_pred=model.predict(X_test)
"""



## === cell 10
import xgboost as xgb

model = xgb.XGBRegressor(
    n_estimators=600,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    reg_alpha=0.1,
    min_child_weight=5.0,
    objective="reg:squarederror",
    tree_method="hist",
    max_bin=256,
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train, Y_train_log)

Y_train_log_pred = model.predict(X_train)
residual = (Y_train_log.values - Y_train_log_pred).astype(np.float64)
smear = float(np.mean(np.exp(residual)))
smear = float(np.clip(smear, 0.5, 2.0))

Y_pred_log = model.predict(X_test).astype(np.float64)
Y_pred = (np.expm1(Y_pred_log) * smear).astype(np.float32)

Y_pred = np.clip(Y_pred, 2.5, None)

dist_test = X_test["haversine_km"].astype(np.float32).values
cap = np.where(dist_test > 20.0, 12.0 * dist_test + 15.0, np.inf).astype(np.float32)
Y_pred = np.minimum(Y_pred, cap)

Y_pred = Y_pred.copy()
Y_pred[~_test_in_box] = np.minimum(Y_pred[~_test_in_box], 50.0)



## === cell 11
submission = test_data.copy()
submission = submission.drop(
    columns=[
        "pickup_datetime",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
    ]
)
submission["fare_amount"] = Y_pred



## === cell 12
submission.to_csv("sample_submission.csv", index=False)
