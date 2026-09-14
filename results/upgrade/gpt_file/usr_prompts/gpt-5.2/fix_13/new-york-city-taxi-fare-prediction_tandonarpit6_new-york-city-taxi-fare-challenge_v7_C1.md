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

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 4.62664) has done: 'I fix the runtime error by calling `XGBRegressor.predict(X_test)` (the scikit-learn API doesn’t accept `data=`), which also unblock creation of `Y_pred` and prevent the downstream `NameError`. I also ensure `Y_train` is a 1D Series (not a single-column DataFrame) so XGBoost trains/predicts consistently. Finally, I write the submission to a new filename (to avoid overwriting Kaggle’s `sample_submission.csv`) and guarantee the output has exactly the required columns `key,fare_amount`, producing a valid `.csv` submission end-to-end.'
- What this solution (achieved 7.09168) has done: 'Your current RMSE (4.62664) is worse than the target (3.98929), so we should make small, score-relevant improvements without changing the overall approach (still XGBRegressor on the same engineered features). The biggest low-risk gains here come from (1) cleaning obviously bad training rows (invalid lat/lon, non-positive/unrealistic fares, passenger_count out of range), because NYC Taxi Fare is very sensitive to outliers, and (2) adding just a couple of standard, minimal XGBoost hyperparameters (more trees + smaller learning rate) to better fit the cleaned data while staying within time. I also ensure train/test columns align and clamp negative predictions to 0 (fares can’t be negative), which typically improves RMSE a bit. These changes keep your feature set and training flow intact while moving the score toward the target.'
- What this solution (achieved 6.07147) has done: 'We keep your feature engineering and XGBRegressor approach intact, but fix the biggest remaining score drag: training on a lot of “weird” trips that pass the broad bounding-box filter yet are still outliers (e.g., huge coordinate jumps) and missing an easy, core distance signal. Concretely, we (1) add one standard feature (haversine distance) derived only from existing lat/lon, (2) slightly tighten data cleaning with a max-distance sanity filter (removes extreme outliers that inflate RMSE), and (3) add a small amount of time-derived signal (month/year) without changing the overall training loop. These are minimal, score-relevant changes that usually move RMSE down toward your target without altering the modeling “core logic”. The submission writing stays the same and still produce a valid `submission.csv`.'
- What this solution (achieved 5.28389) has done: 'Your current RMSE (6.07147) is worse than the target (3.98929), so we should make the smallest, score-relevant improvements without changing the overall approach (same feature engineering + XGBRegressor). The biggest low-risk gain on this competition is stricter outlier removal: filter out “impossible” coordinates (0,0), extreme trip distances, and extreme fare-per-km rows that create huge squared errors. To keep the core logic intact, we only add these additional cleaning masks and keep the model/training flow the same; we also apply the same basic coordinate validity filter to the test set to avoid NaNs/infs and then fill any remaining missing values. This should reduce RMSE materially by removing high-error noise while preserving the same model and features.'
- What this solution (achieved 5.01223) has done: 'Your current RMSE (5.28389) is worse than the target (3.98929), so we should make small, score-relevant changes that keep the same overall approach (feature engineering + XGBRegressor) but reduce noisy training signal. The biggest low-risk improvement here is to align the feature set more with the physics of fares by keeping dropoff coordinates (they carry strong location information beyond just distance) and adding a standard “center point” (midpoint) feature; this doesn’t change your modeling approach, only the inputs. I also make the outlier filtering slightly more robust by removing extremely slow/long trips via a tighter maximum haversine distance (which typically reduces squared-error impact from mislabeled/outlier rows) while staying conservative. Submission writing stays identical and still produces a valid `submission.csv`.'
- What this solution (achieved 4.98522) has done: 'Your RMSE (5.01223) is still worse than the target (3.98929), so we should make small, score-relevant improvements without changing the overall XGBRegressor + feature-engineering approach. The biggest remaining drag is label noise/outliers that survive the broad filters; adding a standard “fare-per-km” and “minimum fare” consistency filter (using the already-computed haversine distance) typically reduces squared-error blowups and improves RMSE. To keep semantics stable, I also ensure we filter on finite distances and avoid divide-by-zero by computing the distance once and reusing it for filtering. Everything else (features, model family, training flow, submission format) stays the same.'
- What this solution (achieved 15.25184) has done: 'I fix the tz-aware vs tz-naive datetime subtraction by forcing both `pickup_datetime` and the datetime parsed from `key` to be tz-naive (UTC removed), so duration features and duration-based cleaning work. I also prevent XGBoost from seeing non-numeric columns by ensuring we always train on the engineered/clean numeric `X_train` (not the raw `training_data` copy) and that `X_test` is aligned to the same numeric columns. Finally, I ensure `Y_pred` is always created and a valid `submission.csv` with exactly `key,fare_amount` is written end-to-end.'
- What this solution (achieved 4.99586) has done: 'Your current RMSE (15.25184) is far worse than the target (3.98929), and the main issue is that you’re training with “trip_duration_min” derived from `key` (which is not a dropoff timestamp), creating a largely wrong/noisy feature and also using it for cleaning—this can severely degrade RMSE. I keep your XGBRegressor + feature-engineering pipeline intact, but remove the duration-based cleaning/feature entirely (minimal semantic correction) and replace it with one standard, low-risk improvement: log-transforming the target (`log1p`) and inverting with `expm1`, which typically stabilizes RMSE for this dataset without changing the model family or loop. Everything else (haversine, time-of-pickup features, outlier filters, submission format) remains the same, and the script still write a valid `submission.csv`.'
- What this solution (achieved 5.00085) has done: 'Your current RMSE (4.99586) is worse than the target (3.98929), so we should make small, score-relevant improvements without changing the overall approach (same feature engineering + XGBRegressor on log1p target). The lowest-risk gains here are (1) making the training cleaning consistent with your test-time handling by imputing any remaining missing numeric values in `X_train` (and aligning medians between train/test), and (2) adding `min_child_weight` and `gamma` to slightly reduce overfitting to residual label noise/outliers that survive filters (this often helps public LB RMSE without changing model family/loop). I also set `missing=np.nan` explicitly and keep everything else (features, log-transform, prediction inversion, submission format) unchanged. These changes are minimal and aimed at reducing squared-error blowups, moving RMSE down toward your target band.'
- What this solution (achieved 15.25184) has done: 'We need to lower RMSE from 5.00085 toward 3.98929, so we make small, score-relevant changes without altering the overall pipeline (same features, same log1p target, same XGBRegressor training flow). The biggest low-risk gain is tightening training-data cleaning with one additional sanity filter: remove rows with implausible “fare per minute” using pickup_datetime-derived trip duration (computed only from within-row information via the `key`’s timestamp, not from labels), which helps reduce squared-error blowups from mislabeled/outlier trips. We also add two very standard, minimal model regularization tweaks (`reg_alpha` and a small `gamma`) to reduce sensitivity to remaining noise, while keeping all other hyperparameters and core logic intact. Submission writing and column alignment remain unchanged.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os

BASE_INPUT = "/kaggle/input"
if not os.path.exists(BASE_INPUT):
    BASE_INPUT = "../input"

print("BASE_INPUT:", BASE_INPUT)
print("Top-level input dirs:", os.listdir(BASE_INPUT)[:20])

COMP_DIR = os.path.join(BASE_INPUT, "new-york-city-taxi-fare-prediction")
if not os.path.exists(COMP_DIR):
    COMP_DIR = BASE_INPUT

train_path = os.path.join(COMP_DIR, "train.csv")
test_path = os.path.join(COMP_DIR, "test.csv")
sample_path = os.path.join(COMP_DIR, "sample_submission.csv")

print("Resolved paths:")
print("train:", train_path)
print("test:", test_path)
print("sample_submission:", sample_path)



## === cell 1
training_data = pd.read_csv(
    train_path,
    nrows=2_000_000,
    parse_dates=["pickup_datetime"],
)
test_data = pd.read_csv(
    test_path,
    parse_dates=["pickup_datetime"],
)

training_data["pickup_datetime"] = pd.to_datetime(
    training_data["pickup_datetime"], utc=True, errors="coerce"
).dt.tz_convert(None)
test_data["pickup_datetime"] = pd.to_datetime(
    test_data["pickup_datetime"], utc=True, errors="coerce"
).dt.tz_convert(None)

print("train shape:", training_data.shape)
print("test shape:", test_data.shape)



## === cell 2
training_data



## === cell 3
X_train = training_data.copy()
X_test = test_data.copy()
Y_train = training_data.copy()




## === cell 4
def haversine_km(lon1, lat1, lon2, lat2):
    lon1 = np.radians(lon1.astype(float))
    lat1 = np.radians(lat1.astype(float))
    lon2 = np.radians(lon2.astype(float))
    lat2 = np.radians(lat2.astype(float))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371.0 * c


mask = (
    X_train["fare_amount"].notna()
    & X_train["pickup_longitude"].between(-75, -72)
    & X_train["dropoff_longitude"].between(-75, -72)
    & X_train["pickup_latitude"].between(40, 42)
    & X_train["dropoff_latitude"].between(40, 42)
    & X_train["passenger_count"].between(1, 6)
    & X_train["fare_amount"].between(2.5, 250.0)
)

mask = mask & ~(
    (X_train["pickup_longitude"].abs() < 1e-6)
    | (X_train["pickup_latitude"].abs() < 1e-6)
    | (X_train["dropoff_longitude"].abs() < 1e-6)
    | (X_train["dropoff_latitude"].abs() < 1e-6)
)

training_data = training_data.loc[mask].reset_index(drop=True)

dist_km = haversine_km(
    training_data["pickup_longitude"],
    training_data["pickup_latitude"],
    training_data["dropoff_longitude"],
    training_data["dropoff_latitude"],
)

finite_dist = np.isfinite(dist_km)
training_data = training_data.loc[finite_dist].reset_index(drop=True)
dist_km = dist_km[finite_dist].reset_index(drop=True)

same_point = (
    training_data["pickup_longitude"] == training_data["dropoff_longitude"]
) & (training_data["pickup_latitude"] == training_data["dropoff_latitude"])
training_data = training_data.loc[~same_point].reset_index(drop=True)
dist_km = dist_km.loc[~same_point].reset_index(drop=True)

dist_mask = dist_km.between(0.05, 45.0)
training_data = training_data.loc[dist_mask].reset_index(drop=True)
dist_km = dist_km.loc[dist_mask].reset_index(drop=True)

fare_per_km = training_data["fare_amount"] / dist_km
min_fare_ok = training_data["fare_amount"] >= (2.5 + 0.5 * dist_km)
fpkm_ok = fare_per_km.between(1.2, 25.0)

training_data = training_data.loc[min_fare_ok & fpkm_ok].reset_index(drop=True)

X_train = training_data.copy()
Y_train = training_data["fare_amount"].copy()

print("after cleaning train shape:", X_train.shape)



## === cell 5
y_train_model = np.log1p(Y_train)

X_train["hour"] = X_train["pickup_datetime"].dt.hour
X_train["dayofweek"] = X_train["pickup_datetime"].dt.dayofweek
X_train["month"] = X_train["pickup_datetime"].dt.month
X_train["year"] = X_train["pickup_datetime"].dt.year

X_train["latitude_distance"] = (
    X_train["dropoff_latitude"] - X_train["pickup_latitude"]
).abs()
X_train["longitude_distance"] = (
    X_train["dropoff_longitude"] - X_train["pickup_longitude"]
).abs()

X_train["haversine_km"] = haversine_km(
    X_train["pickup_longitude"],
    X_train["pickup_latitude"],
    X_train["dropoff_longitude"],
    X_train["dropoff_latitude"],
)

X_train["mid_longitude"] = (
    X_train["pickup_longitude"] + X_train["dropoff_longitude"]
) / 2.0
X_train["mid_latitude"] = (
    X_train["pickup_latitude"] + X_train["dropoff_latitude"]
) / 2.0

JFK_LON, JFK_LAT = -73.7781, 40.6413
LGA_LON, LGA_LAT = -73.8740, 40.7769
EWR_LON, EWR_LAT = -74.1745, 40.6895

X_train["pickup_to_jfk_km"] = haversine_km(
    X_train["pickup_longitude"], X_train["pickup_latitude"], JFK_LON, JFK_LAT
)
X_train["dropoff_to_jfk_km"] = haversine_km(
    X_train["dropoff_longitude"], X_train["dropoff_latitude"], JFK_LON, JFK_LAT
)
X_train["pickup_to_lga_km"] = haversine_km(
    X_train["pickup_longitude"], X_train["pickup_latitude"], LGA_LON, LGA_LAT
)
X_train["dropoff_to_lga_km"] = haversine_km(
    X_train["dropoff_longitude"], X_train["dropoff_latitude"], LGA_LON, LGA_LAT
)
X_train["pickup_to_ewr_km"] = haversine_km(
    X_train["pickup_longitude"], X_train["pickup_latitude"], EWR_LON, EWR_LAT
)
X_train["dropoff_to_ewr_km"] = haversine_km(
    X_train["dropoff_longitude"], X_train["dropoff_latitude"], EWR_LON, EWR_LAT
)

X_train = X_train.drop(
    columns=[
        "key",
        "fare_amount",
        "pickup_datetime",
    ]
)

X_train = X_train.replace([np.inf, -np.inf], np.nan)
train_medians = X_train.median(numeric_only=True)
X_train = X_train.fillna(train_medians)



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/470495921.py in <cell line: 0>()
     33 EWR_LON, EWR_LAT = -74.1745, 40.6895
     34 
---> 35 X_train["pickup_to_jfk_km"] = haversine_km(
     36     X_train["pickup_longitude"], X_train["pickup_latitude"], JFK_LON, JFK_LAT
     37 )

/tmp/ipykernel_11/3748863501.py in haversine_km(lon1, lat1, lon2, lat2)
      2     lon1 = np.radians(lon1.astype(float))
      3     lat1 = np.radians(lat1.astype(float))
----> 4     lon2 = np.radians(lon2.astype(float))
      5     lat2 = np.radians(lat2.astype(float))
      6     dlon = lon2 - lon1

AttributeError: 'float' object has no attribute 'astype'

## === cell 6
test_mask = (
    test_data["pickup_longitude"].between(-75, -72)
    & test_data["dropoff_longitude"].between(-75, -72)
    & test_data["pickup_latitude"].between(40, 42)
    & test_data["dropoff_latitude"].between(40, 42)
)

X_test = test_data.copy()
X_test.loc[
    ~test_mask,
    ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"],
] = np.nan

X_test["hour"] = X_test["pickup_datetime"].dt.hour
X_test["dayofweek"] = X_test["pickup_datetime"].dt.dayofweek
X_test["month"] = X_test["pickup_datetime"].dt.month
X_test["year"] = X_test["pickup_datetime"].dt.year

X_test["latitude_distance"] = (
    X_test["dropoff_latitude"] - X_test["pickup_latitude"]
).abs()
X_test["longitude_distance"] = (
    X_test["dropoff_longitude"] - X_test["pickup_longitude"]
).abs()

X_test["haversine_km"] = haversine_km(
    X_test["pickup_longitude"],
    X_test["pickup_latitude"],
    X_test["dropoff_longitude"],
    X_test["dropoff_latitude"],
)

X_test["mid_longitude"] = (
    X_test["pickup_longitude"] + X_test["dropoff_longitude"]
) / 2.0
X_test["mid_latitude"] = (X_test["pickup_latitude"] + X_test["dropoff_latitude"]) / 2.0

JFK_LON, JFK_LAT = -73.7781, 40.6413
LGA_LON, LGA_LAT = -73.8740, 40.7769
EWR_LON, EWR_LAT = -74.1745, 40.6895

X_test["pickup_to_jfk_km"] = haversine_km(
    X_test["pickup_longitude"], X_test["pickup_latitude"], JFK_LON, JFK_LAT
)
X_test["dropoff_to_jfk_km"] = haversine_km(
    X_test["dropoff_longitude"], X_test["dropoff_latitude"], JFK_LON, JFK_LAT
)
X_test["pickup_to_lga_km"] = haversine_km(
    X_test["pickup_longitude"], X_test["pickup_latitude"], LGA_LON, LGA_LAT
)
X_test["dropoff_to_lga_km"] = haversine_km(
    X_test["dropoff_longitude"], X_test["dropoff_latitude"], LGA_LON, LGA_LAT
)
X_test["pickup_to_ewr_km"] = haversine_km(
    X_test["pickup_longitude"], X_test["pickup_latitude"], EWR_LON, EWR_LAT
)
X_test["dropoff_to_ewr_km"] = haversine_km(
    X_test["dropoff_longitude"], X_test["dropoff_latitude"], EWR_LON, EWR_LAT
)

X_test = X_test.drop(columns=["key", "pickup_datetime"])
X_test = X_test[X_train.columns]

X_test = X_test.replace([np.inf, -np.inf], np.nan)
X_test = X_test.fillna(train_medians)



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/1926762736.py in <cell line: 0>()
     41 EWR_LON, EWR_LAT = -74.1745, 40.6895
     42 
---> 43 X_test["pickup_to_jfk_km"] = haversine_km(
     44     X_test["pickup_longitude"], X_test["pickup_latitude"], JFK_LON, JFK_LAT
     45 )

/tmp/ipykernel_11/3748863501.py in haversine_km(lon1, lat1, lon2, lat2)
      2     lon1 = np.radians(lon1.astype(float))
      3     lat1 = np.radians(lat1.astype(float))
----> 4     lon2 = np.radians(lon2.astype(float))
      5     lat2 = np.radians(lat2.astype(float))
      6     dlon = lon2 - lon1

AttributeError: 'float' object has no attribute 'astype'

## === cell 7
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



## === cell 8
import xgboost as xgb

model = xgb.XGBRegressor(
    n_estimators=600,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    reg_alpha=0.1,
    min_child_weight=5,
    gamma=0.05,
    random_state=42,
    n_jobs=-1,
    tree_method="hist",
    missing=np.nan,
)
model.fit(X_train, y_train_model)

Y_pred = np.expm1(model.predict(X_test))
Y_pred = np.clip(Y_pred, 0, None)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2385261571.py in <cell line: 0>()
     16     missing=np.nan,
     17 )
---> 18 model.fit(X_train, y_train_model)
     19 
     20 Y_pred = np.expm1(model.predict(X_test))

/usr/local/lib/python3.11/dist-packages/xgboost/core.py in inner_f(*args, **kwargs)
    728             for k, arg in zip(sig.parameters, args):
    729                 kwargs[k] = arg
--> 730             return func(**kwargs)
    731 
    732         return inner_f

/usr/local/lib/python3.11/dist-packages/xgboost/sklearn.py in fit(self, X, y, sample_weight, base_margin, eval_set, eval_metric, early_stopping_rounds, verbose, xgb_model, sample_weight_eval_set, base_margin_eval_set, feature_weights, callbacks)
   1053         with config_context(verbosity=self.verbosity):
   1054             evals_result: TrainingCallback.EvalsLog = {}
-> 1055             train_dmatrix, evals = _wrap_evaluation_matrices(
   1056                 missing=self.missing,
   1057                 X=X,

/usr/local/lib/python3.11/dist-packages/xgboost/sklearn.py in _wrap_evaluation_matrices(missing, X, y, group, qid, sample_weight, base_margin, feature_weights, eval_set, sample_weight_eval_set, base_margin_eval_set, eval_group, eval_qid, create_dmatrix, enable_categorical, feature_types)
    519     """Convert array_like evaluation matrices into DMatrix.  Perform validation on the
    520     way."""
--> 521     train_dmatrix = create_dmatrix(
    522         data=X,
    523         label=y,

/usr/local/lib/python3.11/dist-packages/xgboost/sklearn.py in _create_dmatrix(self, ref, **kwargs)
    956         if _can_use_qdm(self.tree_method) and self.booster != "gblinear":
    957             try:
--> 958                 return QuantileDMatrix(
    959                     **kwargs, ref=ref, nthread=self.n_jobs, max_bin=self.max_bin
    960                 )

/usr/local/lib/python3.11/dist-packages/xgboost/core.py in inner_f(*args, **kwargs)
    728             for k, arg in zip(sig.parameters, args):
    729                 kwargs[k] = arg
--> 730             return func(**kwargs)
    731 
    732         return inner_f

/usr/local/lib/python3.11/dist-packages/xgboost/core.py in __init__(self, data, label, weight, base_margin, missing, silent, feature_names, feature_types, nthread, max_bin, ref, group, qid, label_lower_bound, label_upper_bound, feature_weights, enable_categorical, data_split_mode)
   1527                 )
   1528 
-> 1529         self._init(
   1530             data,
   1531             ref=ref,

/usr/local/lib/python3.11/dist-packages/xgboost/core.py in _init(self, data, ref, enable_categorical, **meta)
   1586             ctypes.byref(handle),
   1587         )
-> 1588         it.reraise()
   1589         # delay check_call to throw intermediate exception first
   1590         _check_call(ret)

/usr/local/lib/python3.11/dist-packages/xgboost/core.py in reraise(self)
    574             exc = self._exception
    575             self._exception = None
--> 576             raise exc  # pylint: disable=raising-bad-type
    577 
    578     def __del__(self) -> None:

/usr/local/lib/python3.11/dist-packages/xgboost/core.py in _handle_exception(self, fn, dft_ret)
    555 
    556         try:
--> 557             return fn()
    558         except Exception as e:  # pylint: disable=broad-except
    559             # Defer the exception in order to return 0 and stop the iteration.

/usr/local/lib/python3.11/dist-packages/xgboost/core.py in <lambda>()
    639 
    640         # pylint: disable=not-callable
--> 641         return self._handle_exception(lambda: self.next(input_data), 0)
    642 
    643     @abstractmethod

/usr/local/lib/python3.11/dist-packages/xgboost/data.py in next(self, input_data)
   1278             return 0
   1279         self.it += 1
-> 1280         input_data(**self.kwargs)
   1281         return 1
   1282 

/usr/local/lib/python3.11/dist-packages/xgboost/core.py in inner_f(*args, **kwargs)
    728             for k, arg in zip(sig.parameters, args):
    729                 kwargs[k] = arg
--> 730             return func(**kwargs)
    731 
    732         return inner_f

/usr/local/lib/python3.11/dist-packages/xgboost/core.py in input_data(data, feature_names, feature_types, **kwargs)
    622                 new, cat_codes, feature_names, feature_types = self._temporary_data
    623             else:
--> 624                 new, cat_codes, feature_names, feature_types = _proxy_transform(
    625                     data,
    626                     feature_names,

/usr/local/lib/python3.11/dist-packages/xgboost/data.py in _proxy_transform(data, feature_names, feature_types, enable_categorical)
   1313         data = pd.DataFrame(data)
   1314     if _is_pandas_df(data):
-> 1315         arr, feature_names, feature_types = _transform_pandas_df(
   1316             data, enable_categorical, feature_names, feature_types
   1317         )

/usr/local/lib/python3.11/dist-packages/xgboost/data.py in _transform_pandas_df(data, enable_categorical, feature_names, feature_types, meta, meta_type)
    488             or is_pa_ext_dtype(dtype)
    489         ):
--> 490             _invalid_dataframe_dtype(data)
    491         if is_pa_ext_dtype(dtype):
    492             pyarrow_extension = True

/usr/local/lib/python3.11/dist-packages/xgboost/data.py in _invalid_dataframe_dtype(data)
    306     type_err = "DataFrame.dtypes for data must be int, float, bool or category."
    307     msg = f"""{type_err} {_ENABLE_CAT_ERR} {err}"""
--> 308     raise ValueError(msg)
    309 
    310 

ValueError: DataFrame.dtypes for data must be int, float, bool or category. When categorical type is supplied, The experimental DMatrix parameter`enable_categorical` must be set to `True`.  Invalid columns:key: object, pickup_datetime: datetime64[ns]

## === cell 9
submission = pd.DataFrame({"key": test_data["key"], "fare_amount": Y_pred})

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)

print("Wrote:", submission_path)
print(submission.head())
print("Columns:", submission.columns.tolist())
print("Rows:", len(submission))

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/379298291.py in <cell line: 0>()
----> 1 submission = pd.DataFrame({"key": test_data["key"], "fare_amount": Y_pred})
      2 
      3 submission_path = "submission.csv"
      4 submission.to_csv(submission_path, index=False)
      5 

NameError: name 'Y_pred' is not defined
