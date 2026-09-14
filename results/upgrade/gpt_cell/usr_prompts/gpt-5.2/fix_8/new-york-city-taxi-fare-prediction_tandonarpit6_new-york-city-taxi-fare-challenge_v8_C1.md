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

4.00706

# 6. Current score

6.66407

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 4.77089) has done: 'Diagnosis: The crash happens in cell 8 because `xgboost.XGBRegressor.predict()` (the scikit-learn API) does not accept a `data=` keyword; it expects the feature matrix as the first positional argument (or `X=`). This keyword is used by the lower-level `xgboost.DMatrix`/Booster API, not `XGBRegressor`.  

Patch summary: Keep the same model and training logic, but change the prediction call to pass `X_test` positionally to `predict()`. This resolves the `TypeError` without changing outputs other than negligible dtype/path differences.  

Updated cells: Only cell 8 is modified.  

Compatibility notes for cell k+1: `Y_pred` remains a 1D numpy array aligned with `test_data` rows, so cell 9 can assign it to `submission['fare_amount']` unchanged.  

Assumptions: `X_train`, `Y_train`, and `X_test` are already prepared as in prior cells, and `Y_train` is accepted by XGBoost even if it is a single-column DataFrame.'
- What this solution (achieved 8.72228) has done: 'You’re currently worse than the target (RMSE 4.77089 vs 4.00706), so we make small, legitimate changes that usually improve RMSE without changing the overall approach: keep the same XGBoost regressor pipeline but (1) clean obviously bad training rows (invalid lat/lon, zero-distance trips, outlier fares) and (2) add a single, standard geodesic distance feature (haversine) alongside your existing time and abs-delta features. These are minimal feature-engineering and data-quality fixes that typically reduce noise and help the same model generalize better, moving your score toward the target band. We also ensure `Y_train` is a 1D vector (avoids any multioutput quirks) and clip negative predictions to 0 to prevent RMSE blowups from invalid negatives. The script still reads the same files/paths, trains the same model family (XGBRegressor), and writes a valid `sample_submission.csv`.'
- What this solution (achieved 7.7216) has done: 'Your current RMSE is much worse than the target, so the smallest likely win is to keep the same XGBoost regressor approach but fix two high-impact issues: (1) train/test feature mismatch (you dropped `dropoff_*` only, leaving `pickup_*` in training but not in test), and (2) unscaled/invalid passenger counts in test (0 and large values) plus missing datetime-derived features like month/year that usually help. I make train and test feature columns identical by dropping the same raw columns from both and adding the same simple time features to both, while keeping your existing cleaning and haversine feature. I also align with the competition’s standard baseline by adding a small `min_child_weight`/`subsample`/`colsample_bytree` regularization tweak that typically improves RMSE without changing the core model family or training loop. The script still run end-to-end quickly on 2.5M rows and write a valid `sample_submission.csv`.'
- What this solution (achieved 7.9149) has done: 'Your current RMSE is far above the target, so the smallest likely improvement is to keep the same XGBRegressor and existing features but fix data quality issues that strongly hurt generalization. I (1) parse/clean `pickup_datetime` and drop rows where it is invalid (so time features aren’t NaN-driven), (2) add one standard geographic feature (`manhattan_km`) alongside your existing haversine/delta features, and (3) apply a light, competition-standard “airport bounding box” cleanup to remove extreme/noisy trips while keeping the same training approach. These changes preserve the core logic (same model family, same training call, same loss/metric semantics) and should move RMSE down toward your target without attempting to over-optimize. The script still run end-to-end and write `sample_submission.csv` with `key,fare_amount`.'
- What this solution (achieved 6.66407) has done: 'I make two minimal, high-impact fixes that keep your XGBRegressor approach and existing features intact but typically reduce RMSE substantially on this competition. First, I add a small amount of robust target/feature cleaning: remove extreme distance outliers and cap very large fares so a few bad rows don’t dominate squared-error training. Second, I apply a simple log1p transform on the target during training and invert it at prediction time; this often improves RMSE for fare distributions while preserving the same model family, training call, and evaluation semantics (still predicting dollars). The rest of your pipeline (features, model type, submission schema/path) stays the same and still writes a valid `sample_submission.csv`.'
- What this solution (achieved 6.66407) has done: 'You’re currently worse than the target (RMSE 6.664 > 4.007), so we should make the smallest legitimate changes that reduce generalization error without changing the core XGBRegressor approach. The biggest issue is that you filter `X_train` by `haversine_km` but **do not filter `Y_train` to the same rows**, causing label/feature misalignment and heavily degrading RMSE; I align `y` to the exact `X_train` index after all filtering. I also apply the same “haversine between (0.1, 100]” filter to the training labels (implicitly via index alignment) and keep prediction post-processing identical. These are minimal, high-impact correctness fixes and should move RMSE substantially toward the target band.'
- What this solution (achieved 6.66407) has done: 'We make one correctness fix that is very likely hurting RMSE: you filter `X_train` by `haversine_km` but you don’t apply the same filter to the cleaned training frame that you later use to build `Y_train`, so your `y` can become misaligned with the filtered features (depending on how indices survive the cleaning). We instead compute the target directly from `X_train`’s aligned index before dropping columns, so labels and rows always match exactly. As a small, safe improvement toward the target, we also filter out extreme `manhattan_km` outliers (same spirit as your existing distance filter) and keep everything else (features, model, log1p transform, submission schema/path) unchanged.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os

print(os.listdir("../input"))


## === cell 1
training_data = pd.read_csv("../input/train.csv", nrows=2500000)
test_data = pd.read_csv("../input/test.csv")


## === cell 2
training_data


## === cell 3
X_train = training_data.copy()
X_test = test_data.copy()
Y_train = training_data.copy()




## === cell 4
def _basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(
        subset=[
            "fare_amount",
            "pickup_longitude",
            "pickup_latitude",
            "dropoff_longitude",
            "dropoff_latitude",
            "passenger_count",
            "pickup_datetime",
        ]
    ).copy()

    df["pickup_datetime"] = pd.to_datetime(
        df["pickup_datetime"], errors="coerce", utc=True
    )
    df = df.dropna(subset=["pickup_datetime"])
    df["pickup_datetime"] = df["pickup_datetime"].dt.tz_convert(None)

    df = df[
        (df["pickup_longitude"].between(-75.0, -72.0))
        & (df["dropoff_longitude"].between(-75.0, -72.0))
        & (df["pickup_latitude"].between(40.0, 42.0))
        & (df["dropoff_latitude"].between(40.0, 42.0))
    ]

    df = df[
        (df["pickup_longitude"].between(-74.5, -72.8))
        & (df["dropoff_longitude"].between(-74.5, -72.8))
        & (df["pickup_latitude"].between(40.5, 41.9))
        & (df["dropoff_latitude"].between(40.5, 41.9))
    ]

    df = df[df["passenger_count"].between(1, 6)]
    df = df[df["fare_amount"].between(2.5, 250.0)]

    same_loc = (df["pickup_longitude"] == df["dropoff_longitude"]) & (
        df["pickup_latitude"] == df["dropoff_latitude"]
    )
    df = df[~same_loc]

    return df


training_data = _basic_clean(training_data)

X_train = training_data.copy()
Y_train = training_data.copy()




## === cell 5
def haversine_np(lon1, lat1, lon2, lat2):
    lon1 = np.radians(lon1.astype(np.float64))
    lat1 = np.radians(lat1.astype(np.float64))
    lon2 = np.radians(lon2.astype(np.float64))
    lat2 = np.radians(lat2.astype(np.float64))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6371.0088 * c
    return km


def _manhattan_km_from_latlon(lat_dist_deg, lon_dist_deg, ref_lat_deg):
    lat_km = 111.32 * lat_dist_deg.astype(np.float64)
    lon_km = (
        111.32 * np.cos(np.radians(ref_lat_deg.astype(np.float64)))
    ) * lon_dist_deg.astype(np.float64)
    return lat_km + lon_km


X_train["hour"] = X_train["pickup_datetime"].dt.hour.astype(np.int16)
X_train["dayofweek"] = X_train["pickup_datetime"].dt.dayofweek.astype(np.int16)
X_train["month"] = X_train["pickup_datetime"].dt.month.astype(np.int16)
X_train["year"] = X_train["pickup_datetime"].dt.year.astype(np.int16)

X_train["latitude_distance"] = (
    (X_train["dropoff_latitude"] - X_train["pickup_latitude"]).abs().astype(np.float32)
)
X_train["longitude_distance"] = (
    (X_train["dropoff_longitude"] - X_train["pickup_longitude"])
    .abs()
    .astype(np.float32)
)

X_train["haversine_km"] = haversine_np(
    X_train["pickup_longitude"],
    X_train["pickup_latitude"],
    X_train["dropoff_longitude"],
    X_train["dropoff_latitude"],
).astype(np.float32)

mean_lat = ((X_train["pickup_latitude"] + X_train["dropoff_latitude"]) / 2.0).astype(
    np.float32
)
X_train["manhattan_km"] = _manhattan_km_from_latlon(
    X_train["latitude_distance"], X_train["longitude_distance"], mean_lat
).astype(np.float32)

dist_mask = X_train["haversine_km"].between(0.1, 100.0) & X_train[
    "manhattan_km"
].between(0.1, 200.0)
X_train = X_train.loc[dist_mask].copy()

X_train = X_train.drop(
    columns=[
        "key",
        "fare_amount",
        "pickup_datetime",
        "dropoff_longitude",
        "dropoff_latitude",
        "pickup_longitude",
        "pickup_latitude",
    ]
)


## === cell 6
X_test["pickup_datetime"] = pd.to_datetime(
    X_test["pickup_datetime"], errors="coerce", utc=True
)
if X_test["pickup_datetime"].isna().any():
    median_dt = training_data["pickup_datetime"].median()
    X_test["pickup_datetime"] = (
        X_test["pickup_datetime"].fillna(median_dt).dt.tz_localize("UTC")
    )
X_test["pickup_datetime"] = X_test["pickup_datetime"].dt.tz_convert(None)

X_test["hour"] = X_test["pickup_datetime"].dt.hour.astype(np.int16)
X_test["dayofweek"] = X_test["pickup_datetime"].dt.dayofweek.astype(np.int16)
X_test["month"] = X_test["pickup_datetime"].dt.month.astype(np.int16)
X_test["year"] = X_test["pickup_datetime"].dt.year.astype(np.int16)

X_test["passenger_count"] = X_test["passenger_count"].clip(1, 6)

X_test["latitude_distance"] = (
    (X_test["dropoff_latitude"] - X_test["pickup_latitude"]).abs().astype(np.float32)
)
X_test["longitude_distance"] = (
    (X_test["dropoff_longitude"] - X_test["pickup_longitude"]).abs().astype(np.float32)
)

X_test["haversine_km"] = haversine_np(
    X_test["pickup_longitude"],
    X_test["pickup_latitude"],
    X_test["dropoff_longitude"],
    X_test["dropoff_latitude"],
).astype(np.float32)

mean_lat_t = ((X_test["pickup_latitude"] + X_test["dropoff_latitude"]) / 2.0).astype(
    np.float32
)
X_test["manhattan_km"] = _manhattan_km_from_latlon(
    X_test["latitude_distance"], X_test["longitude_distance"], mean_lat_t
).astype(np.float32)

X_test = X_test.drop(
    columns=[
        "key",
        "pickup_datetime",
        "dropoff_longitude",
        "dropoff_latitude",
        "pickup_longitude",
        "pickup_latitude",
    ]
)


## === cell 7
y_raw = (
    training_data.loc[X_train.index, "fare_amount"].astype(np.float32).clip(0.0, 250.0)
)
Y_train = np.log1p(y_raw.to_numpy(dtype=np.float32))


## === cell 8
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


## === cell 9
import xgboost as xgb

model = xgb.XGBRegressor(
    random_state=42,
    n_jobs=-1,
    n_estimators=300,
    learning_rate=0.1,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=1.0,
    reg_lambda=1.0,
    objective="reg:squarederror",
)

model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)

Y_pred = np.expm1(Y_pred)
Y_pred = np.clip(Y_pred, 0.0, None)


## === cell 10
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


## === cell 11
submission.to_csv("sample_submission.csv", index=False)
