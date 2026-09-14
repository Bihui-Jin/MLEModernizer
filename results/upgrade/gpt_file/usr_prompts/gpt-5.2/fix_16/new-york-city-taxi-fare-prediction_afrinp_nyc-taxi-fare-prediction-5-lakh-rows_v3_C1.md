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

3.8

# 3. Installed packages

geopandas==0.14.4
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
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
tf_keras==2.18.0

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

3.41966

# 6. Current score

5.84429

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1125.15464) has done: 'I fix the runtime crash by switching from `keras` to `tf_keras` (TensorFlow Keras), which is compatible with Kaggle’s TF/protobuf stack and avoids the `MessageFactory.GetPrototype` error. I also fix a key logic issue that hurts score: you scale train features but scale test features independently; instead I fit a scaler on train and reuse it on validation/test to match feature distributions. To nudge RMSE toward the target, I make the train/validation split deterministic and use `validation_data` while keeping the same model, optimizer, loss, and epoch count. Finally, I ensure the submission has the correct `key,fare_amount` format and that predictions are 1D floats.'
- What this solution (achieved 5.49769) has done: 'I fix the protobuf/Keras import crash by avoiding `tf_keras` and using `sklearn`’s stable regressor instead, while keeping the same feature engineering and train/valid split semantics so the pipeline runs end-to-end. To move RMSE dramatically toward the target, I replace the neural net (which currently produces wildly wrong fares due to training instability/scale issues) with a well-calibrated `RandomForestRegressor` on the exact same engineered features. I also ensure predictions are finite and non-negative (fares can’t be negative) and that the submission has exactly `key,fare_amount` and a `.csv` suffix. All paths remain unchanged and the script write `submission.csv` in the working directory.'
- What this solution (achieved 5.28319) has done: 'Your current RMSE (5.49769) is worse than the target (3.41966), so we should carefully improve generalization without changing the core modeling approach (RandomForest on the same engineered features). The largest low-risk gain here is to use a slightly more appropriate distance feature (Haversine in km) while keeping your existing Euclidean distance feature (so we don’t remove anything), and to tune a couple of RandomForest regularization knobs (e.g., `max_features`, `min_samples_leaf`) to reduce overfitting and improve RMSE. I also ensure all numeric columns are consistent float32 and drop any remaining non-numeric leakage, while keeping the same train/valid split semantics and still writing `submission.csv` with `key,fare_amount`. These changes are minimal, fast, and typically move NYC taxi fare baselines closer to ~3–4 RMSE.'
- What this solution (achieved 5.59868) has done: 'We keep your exact feature set and RandomForest approach, but make two small changes that typically reduce RMSE toward your target: (1) add a lightweight outlier filter on the engineered distance features (very long “trips” inside the NYC bbox are usually bad GPS rows that hurt generalization), and (2) slightly increase forest averaging while regularizing a bit more via `min_samples_leaf`, which often improves this competition’s RMSE without changing core logic. We also train on the full filtered dataset after printing a validation RMSE (so the final model uses all available data), then generate the same `key,fare_amount` submission. All paths and submission format remain unchanged.'
- What this solution (achieved 5.81707) has done: 'The timeout is dominated by fitting a very large RandomForest (900 trees) on up to ~2M rows and computing extra predictions on the full training split; this is far beyond what fits in 600 seconds on Kaggle CPU. To preserve core logic (same model type, same features, same training flow), the main speed fix is to train on a deterministic, in-memory random subset of the already-loaded 2M rows (this keeps the algorithm identical but reduces asymptotic cost). Additionally, we remove redundant type-conversion loops (everything is already numeric), avoid materializing extra copies, and skip the expensive `predict(X_train)` RMSE (OOB RMSE already provides validation semantics and avoids a second full pass). These changes are equivalence-preserving for feature generation and prediction, and deterministically reduce runtime to meet the 600-second limit.'
- What this solution (achieved 6.00586) has done: 'Your current RMSE (5.817) is worse than the target (3.420), so we should make the smallest changes that improve generalization without changing the core approach (same features + RandomForestRegressor). The biggest issue is that your model is training on only 250k rows; in this competition, RMSE improves materially as you use more clean data, so we increase the deterministic fit cap while keeping the same sampling semantics to stay within runtime. We also remove a likely over-aggressive `travel_distance <= 1.0` filter (it can discard many valid trips and bias the training distribution), while keeping the NYC bbox and fare filters and retaining the `haversine_km` sanity cap. Finally, we reduce `n_estimators` so the larger training sample still fits under the 600s budget, aiming to move RMSE closer to the target band rather than maximize.'
- What this solution (achieved 5.33587) has done: 'Your current RMSE (6.00586) is worse than the target (3.41966), so we should make small, safe generalization improvements without changing the core approach (same features + RandomForestRegressor). The biggest low-risk issue is that you currently train but do not use your explicit validation split (you compute OOB RMSE on the training fold), which can hide overfitting and make hyperparameters less appropriate; we switch the RMSE print to use the existing X_valid/y_valid while keeping the same training procedure. To move RMSE downward toward the target, we add two standard NYC Taxi cleaning filters that remove label/geo outliers (plausible passenger_count range and minimum distance) and slightly adjust RandomForest regularization/averaging (more trees, a bit deeper allowed, and slightly smaller leaf) while keeping runtime within 600s by modestly reducing the fit cap. Submission format and paths remain unchanged, and we still write `submission.csv`.'
- What this solution (achieved 5.8228) has done: 'Your current RMSE (5.33587) is worse than the target (3.41966), so we should make small, low-risk generalization improvements while keeping the same core approach (same engineered features + `RandomForestRegressor`). The biggest score drag that’s still “minimal change” is missing standard NYC Taxi cleaning: removing obvious coordinate outliers, filtering unrealistic fares relative to trip distance, and constraining passenger counts in both train and test; these typically reduce noise and improve RMSE materially. I also make the test set consistent with the train cleaning (e.g., passenger_count=0 → 1, NYC bbox), and I very lightly adjust RF regularization/averaging (slightly more trees, slightly smaller leaf) to reduce variance without changing the model family or training loop. All paths remain unchanged, runtime stays bounded by the existing deterministic fit cap, and the script still writes a valid `submission.csv` with `key,fare_amount`.'
- What this solution (achieved 5.8224) has done: 'We keep your exact feature engineering and RandomForest approach, but make two small, score-relevant changes that typically reduce RMSE on this competition: (1) replace the square-root Euclidean `travel_distance` with a scale-consistent Manhattan distance in meters (still derived from the same coordinates, no new data), and (2) add a lightweight “airport trip” indicator using the standard JFK/LGA/EWR bounding boxes, which helps explain high-fare regimes with minimal complexity. To avoid hurting test generalization, we also remove the current test-time coordinate clipping (it can create unrealistic points and feature mismatch) while keeping the same NYC bbox filter for training. All paths, training loop semantics, and submission format remain unchanged, and the script still writes `submission.csv`.'
- What this solution (achieved 5.67668) has done: 'Your current RMSE (5.8224) is worse than the target (3.41966), so we should make small, safe improvements without changing the core approach (same engineered features + RandomForestRegressor). The biggest score drag is that the model is being trained on a generic NYC bbox that still includes many bad/out-of-area trips; adding the standard competition “NYC core” bounding box and a reasonable fare-vs-distance sanity filter typically removes noisy labels and improves RMSE materially. To stay minimal, we keep your exact feature set and training flow, but (a) tighten the geo filter to the common Manhattan/NYC bounds for training only, (b) add a conservative drop of extreme `fare_per_km` outliers using your existing `haversine_km`, and (c) slightly regularize the forest (`min_samples_leaf=3`) while keeping tree count similar so runtime stays within budget. Submission format/paths stay identical and the script still writes `submission.csv`.'
- What this solution (achieved 5.84429) has done: 'The timeout is dominated by fitting a very large RandomForest (800 trees) on up to 900k rows, plus some avoidable extra DataFrame copies and repeated float64 conversions during feature engineering. I keep the exact model, hyperparameters, and training semantics, but make the pipeline faster by (1) reducing Pandas overhead via vectorized NumPy feature computation and fewer `.copy()` calls, (2) avoiding repeated `astype(np.float64)`/`np.radians` conversions by computing radians once per frame, and (3) speeding up scikit-learn training by enabling Intel-optimized scikit-learn (already installed) without changing results. These changes are equivalent (same features, same filtering, same model fit) and directly target runtime.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os

base = "/kaggle/input/new-york-city-taxi-fare-prediction"
if os.path.exists(base):
    for fn in sorted(os.listdir(base))[:50]:
        print(os.path.join(base, fn))



## === cell 1
usecols_train = [
    "key",
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
dtype_train = {
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int16",
}
df = pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv",
    usecols=usecols_train,
    parse_dates=["pickup_datetime"],
    nrows=2000000,
    dtype=dtype_train,
)
df.dropna(inplace=True)



## === cell 2
print(df.shape)
print(df.head())



## === cell 3
usecols_test = [
    "key",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
dtype_test = {
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int16",
}
test = pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv",
    usecols=usecols_test,
    parse_dates=["pickup_datetime"],
    dtype=dtype_test,
)
print(test.shape)
print(test.head())



## === cell 4
pass



## === cell 5
pass



## === cell 6
nyc_min_longitude = -74.3
nyc_max_longitude = -72
nyc_min_latitude = 40.63
nyc_max_latitude = 42

core_min_longitude = -74.05
core_max_longitude = -73.75
core_min_latitude = 40.63
core_max_latitude = 40.85

mask_train = (
    (df["pickup_longitude"] > nyc_min_longitude)
    & (df["pickup_longitude"] < nyc_max_longitude)
    & (df["dropoff_longitude"] > nyc_min_longitude)
    & (df["dropoff_longitude"] < nyc_max_longitude)
    & (df["pickup_latitude"] > nyc_min_latitude)
    & (df["pickup_latitude"] < nyc_max_latitude)
    & (df["dropoff_latitude"] > nyc_min_latitude)
    & (df["dropoff_latitude"] < nyc_max_latitude)
)
df = df.loc[mask_train]

mask_test = (
    (test["pickup_longitude"] > nyc_min_longitude)
    & (test["pickup_longitude"] < nyc_max_longitude)
    & (test["dropoff_longitude"] > nyc_min_longitude)
    & (test["dropoff_longitude"] < nyc_max_longitude)
    & (test["pickup_latitude"] > nyc_min_latitude)
    & (test["pickup_latitude"] < nyc_max_latitude)
    & (test["dropoff_latitude"] > nyc_min_latitude)
    & (test["dropoff_latitude"] < nyc_max_latitude)
)
_ = mask_test



## === cell 7
df["passenger_count"] = df["passenger_count"].astype(np.int16, copy=False)
test["passenger_count"] = test["passenger_count"].astype(np.int16, copy=False)

df["passenger_count"] = df["passenger_count"].where(df["passenger_count"] != 0, 1)
test["passenger_count"] = test["passenger_count"].where(test["passenger_count"] != 0, 1)

df["passenger_count"] = df["passenger_count"].where(df["passenger_count"] >= 0, 1)
test["passenger_count"] = test["passenger_count"].where(test["passenger_count"] >= 0, 1)

df["passenger_count"] = df["passenger_count"].clip(upper=8)
test["passenger_count"] = test["passenger_count"].clip(upper=8)



## === cell 8
df = df[(df["fare_amount"] > 0) & (df["fare_amount"] <= 100)]



## === cell 9
print(df.shape)



## === cell 10
df["longitude_diff"] = (
    df["dropoff_longitude"].to_numpy() - df["pickup_longitude"].to_numpy()
).astype(np.float32, copy=False)
df["latitude_diff"] = (
    df["dropoff_latitude"].to_numpy() - df["pickup_latitude"].to_numpy()
).astype(np.float32, copy=False)



## === cell 11
test["longitude_diff"] = (
    test["dropoff_longitude"].to_numpy() - test["pickup_longitude"].to_numpy()
).astype(np.float32, copy=False)
test["latitude_diff"] = (
    test["dropoff_latitude"].to_numpy() - test["pickup_latitude"].to_numpy()
).astype(np.float32, copy=False)



## === cell 12
dt = df["pickup_datetime"].dt
df["year"] = dt.year.astype(np.int16)
df["month"] = dt.month.astype(np.int8)
df["day"] = dt.day.astype(np.int8)
df["day_of_week"] = dt.dayofweek.astype(np.int8)
df["hour"] = dt.hour.astype(np.int8)
df = df.drop(["pickup_datetime"], axis=1)



## === cell 13
dt = test["pickup_datetime"].dt
test["year"] = dt.year.astype(np.int16)
test["month"] = dt.month.astype(np.int8)
test["day"] = dt.day.astype(np.int8)
test["day_of_week"] = dt.dayofweek.astype(np.int8)
test["hour"] = dt.hour.astype(np.int8)
test = test.drop(["pickup_datetime"], axis=1)



## === cell 14
test_keys = test["key"].copy()




## === cell 15
def euc_distance(lat1, long1, lat2, long2):
    return ((lat1 - lat2) ** 2 + (long1 - long2) ** 2) ** 0.5


def manhattan_meters_from_rad(lat1r, lon1r, lat2r, lon2r):
    dlat = np.abs(lat2r - lat1r)
    dlon = np.abs(lon2r - lon1r)
    mean_lat = 0.5 * (lat1r + lat2r)
    R = 6371008.8  # meters
    x = R * dlon * np.cos(mean_lat)
    y = R * dlat
    return (x + y).astype(np.float32)


def haversine_km_from_rad(lat1r, lon1r, lat2r, lon2r):
    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1r) * np.cos(lat2r) * (
        np.sin(dlon / 2.0) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(a))
    R = 6371.0088
    return (R * c).astype(np.float32)


def add_distance_features(frame):
    plat = frame["pickup_latitude"].to_numpy(dtype=np.float64, copy=False)
    plon = frame["pickup_longitude"].to_numpy(dtype=np.float64, copy=False)
    dlat = frame["dropoff_latitude"].to_numpy(dtype=np.float64, copy=False)
    dlon = frame["dropoff_longitude"].to_numpy(dtype=np.float64, copy=False)

    plat_r = np.radians(plat)
    plon_r = np.radians(plon)
    dlat_r = np.radians(dlat)
    dlon_r = np.radians(dlon)

    frame["travel_distance"] = manhattan_meters_from_rad(plat_r, plon_r, dlat_r, dlon_r)
    frame["haversine_km"] = haversine_km_from_rad(plat_r, plon_r, dlat_r, dlon_r)
    frame["haversine_m"] = (
        frame["haversine_km"].to_numpy(dtype=np.float32, copy=False) * 1000.0
    ).astype(np.float32, copy=False)


def in_box(lat, lon, lat_min, lat_max, lon_min, lon_max):
    return (lat >= lat_min) & (lat <= lat_max) & (lon >= lon_min) & (lon <= lon_max)


JFK = (40.621, 40.670, -73.835, -73.740)
LGA = (40.767, 40.793, -73.889, -73.855)
EWR = (40.670, 40.708, -74.199, -74.159)


def airport_flag_frame(frame):
    plat = frame["pickup_latitude"].to_numpy(dtype=np.float32, copy=False)
    plon = frame["pickup_longitude"].to_numpy(dtype=np.float32, copy=False)
    dlat = frame["dropoff_latitude"].to_numpy(dtype=np.float32, copy=False)
    dlon = frame["dropoff_longitude"].to_numpy(dtype=np.float32, copy=False)

    p_jfk = in_box(plat, plon, *JFK)
    d_jfk = in_box(dlat, dlon, *JFK)
    p_lga = in_box(plat, plon, *LGA)
    d_lga = in_box(dlat, dlon, *LGA)
    p_ewr = in_box(plat, plon, *EWR)
    d_ewr = in_box(dlat, dlon, *EWR)

    p_air = (p_jfk | p_lga) | p_ewr
    d_air = (d_jfk | d_lga) | d_ewr
    return (p_air | d_air).astype(np.int8)


add_distance_features(df)
add_distance_features(test)
df["airport_trip"] = airport_flag_frame(df)
test["airport_trip"] = airport_flag_frame(test)



## === cell 16
if "haversine_km" in df.columns:
    hk = df["haversine_km"].to_numpy(dtype=np.float32, copy=False)
    fa = df["fare_amount"].to_numpy(dtype=np.float32, copy=False)
    denom = np.maximum(hk, np.float32(0.02))
    fare_per_km = fa / denom
    keep = (fare_per_km >= 1.0) & (fare_per_km <= 35.0)
    df = df.loc[keep]



## === cell 17
pass



## === cell 18
print("train nulls:", int(df.isnull().values.sum()))
print("test nulls:", int(test.isnull().values.sum()))



## === cell 19
try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except Exception as _e:
    pass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

RANDOM_STATE = 42

MAX_TRAIN_ROWS_FOR_FIT = (
    900_000  # keep runtime bounded, preserves existing training cap approach
)

df_model = df.drop(["key"], axis=1)
test_model = test.drop(["key"], axis=1)

df_model.dropna(inplace=True)  # safe: only affects train rows
test_model = test_model.fillna(0.0)

df_model = df_model[
    (df_model["passenger_count"] >= 1) & (df_model["passenger_count"] <= 8)
]

if "haversine_km" in df_model.columns:
    df_model = df_model[
        (df_model["haversine_km"] >= 0.02) & (df_model["haversine_km"] <= 80.0)
    ]

if len(df_model) > MAX_TRAIN_ROWS_FOR_FIT:
    df_model = df_model.sample(n=MAX_TRAIN_ROWS_FOR_FIT, random_state=RANDOM_STATE)

feature_cols = [c for c in df_model.columns if c != "fare_amount"]
X = df_model[feature_cols].to_numpy(dtype=np.float32, copy=False)
y = df_model["fare_amount"].to_numpy(dtype=np.float32, copy=False)
X_test = test_model[feature_cols].to_numpy(dtype=np.float32, copy=False)

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)



## === cell 20
model = RandomForestRegressor(
    n_estimators=800,
    random_state=42,
    n_jobs=-1,
    min_samples_leaf=2,
    max_features="sqrt",
    oob_score=True,
    bootstrap=True,
)

model.fit(X_train, y_train)

valid_pred = model.predict(X_valid)
valid_rmse = np.sqrt(mean_squared_error(y_valid, valid_pred))

print("Valid RMSE (holdout): {:0.4f}".format(valid_rmse))
print("------------------------")



## === cell 21
pred = model.predict(X_test).astype(np.float32)

pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
pred = np.clip(pred, 0.0, None)



## === cell 22
print(pred.shape)



## === cell 23
submission = pd.DataFrame({"key": test_keys.values, "fare_amount": pred})
submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
