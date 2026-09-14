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

4.01327

# 6. Current score

7.28875

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 4.70062) has done: 'Diagnosis: The crash happens in cell 8 because `xgboost.XGBRegressor.predict()` (xgboost 2.0.3) does not accept a `data=` keyword argument; it expects the feature matrix as the first positional argument (`X`). This is an API mismatch (confusing the scikit-learn wrapper with the lower-level `xgb.DMatrix` API).  

Patch summary: Update the prediction call to pass `X_test` positionally (`model.predict(X_test)`), keeping the same model, training call, and output variable (`Y_pred`) so downstream cells continue to work unchanged.  

Updated cells: Only cell 8 is modified.  

Compatibility notes for cell k+1: Cell 9 uses `plot_importance(model)` and expects `model` to be a fitted XGBoost model; this remains true. `Y_pred` is still created for any later cells that might use it.  

Assumptions: `X_train`, `Y_train`, and `X_test` are valid numeric pandas objects as prepared in earlier cells, and `model.fit()` succeeds as before.'
- What this solution (achieved 9.67126) has done: 'Your current score (4.70062) is worse than the target (4.01327), so we should improve it with minimal, safe changes that keep the same overall approach (XGBoost regressor on simple engineered features). The biggest score gain for this competition typically comes from removing obviously bad training rows (invalid coordinates/passenger counts and extreme fares) and using a slightly more appropriate XGBoost configuration (objective/trees/learning rate) without changing the modeling family. I also ensure `Y_train` is a 1D vector (not a 2D DataFrame) and clip negative predictions to 0, which improves RMSE stability. Paths and the submission format remain unchanged, and the script still writes a valid `.csv`.'
- What this solution (achieved 9.75042) has done: 'To move RMSE down toward the 4.01327 target (your current 9.67126 is worse, and lower is better), the smallest high-impact change is to add one more standard, lightweight feature: haversine distance (great-circle distance), while keeping the same XGBoost regressor approach and training loop. I also ensure the test set gets identical feature engineering and apply the same geographic validity filter to training (already present) while keeping your existing columns and submission format unchanged. This preserves your model family and core workflow but typically yields a large RMSE improvement compared to raw lat/long + abs deltas. The script still run end-to-end within time by keeping `nrows=2_000_000` and writing a valid `sample_submission.csv`.'
- What this solution (achieved 8.83144) has done: 'Your RMSE (9.75042) is much worse than the target (4.01327), so we should make a small, high-impact improvement without changing the core model family (XGBRegressor) or training loop. The biggest issue is that the current feature set accidentally drops `passenger_count` and also drops the raw pickup coordinates, both of which are very informative for this task; adding them back is minimal feature-engineering change and typically reduces RMSE substantially. I also add simple, standard datetime parts (dayofweek, month) while keeping the same approach and submission semantics. All paths remain unchanged, runtime stays within limits, and the script still writes a valid `sample_submission.csv` with `key,fare_amount`.'
- What this solution (achieved 7.91711) has done: 'To move your RMSE down toward the 4.01327 target (current 8.83144 is worse, and lower is better) with minimal disruption, I keep the same XGBoost regressor pipeline but fix the largest remaining performance leak: you currently train with only engineered deltas/distance/time but you do not include the raw coordinates and passenger_count as features, which hurts accuracy. I add back the raw pickup/dropoff lat/long and passenger_count to both train and test feature matrices while keeping all existing engineered features and the same model/training approach. I also add two very standard, low-risk datetime parts (year and dayofyear) that typically help a bit without changing the overall semantics. The script still run end-to-end and write a valid `sample_submission.csv` with `key,fare_amount`.'
- What this solution (achieved 6.39435) has done: 'The crash happens because `X_train` still contains the target column `fare_amount`, while `X_test` does not; XGBoost stores the training feature names and then refuses to predict when they don’t match. The minimal fix is to drop `fare_amount` from `X_train` inside the failing cell, right before fitting, without changing the model or training logic. This keeps `Y_train` unchanged and aligns train/test feature columns deterministically. The rest of the cell remains the same, including clipping predictions.'
- What this solution (achieved 6.51841) has done: 'Diagnosis: The crash happens in `_haversine_km` because it assumes all inputs are pandas Series/NumPy arrays and calls `.astype(float)` on them. When `_add_features` computes distance to the fixed NYC center, it passes `NYC_LON`/`NYC_LAT` as Python floats; floats do not have `.astype`, causing `AttributeError`. The fix is to make `_haversine_km` (and `_bearing`, for consistency) robust to scalar inputs by converting all arguments via `np.asarray(..., dtype=float)` before calling `np.radians`.

Patch summary: Update `_haversine_km` and `_bearing` in cell 5 to safely handle both array-like and scalar float inputs using `np.asarray(..., dtype=float)`. This preserves identical feature logic and numeric behavior while preventing the scalar `.astype` crash. No other cells or model logic are changed.

Updated cells: cell 5 only.

Compatibility notes for cell k+1: Cell 6 calls `_add_features(X_test)` and expects the same added columns; this patch does not change column names, shapes, or downstream expectations—only makes the distance/bearing helpers accept scalar center coordinates.

Assumptions: NumPy broadcasting is acceptable here (it is), and converting inputs with `np.asarray(..., dtype=float)` preserves intended computation for both Series and scalars.'
- What this solution (achieved 6.88172) has done: 'Your current RMSE (6.51841) is worse than the target (4.01327), so we should improve accuracy with minimal changes while keeping the same XGBoost regressor workflow and feature set. The biggest remaining low-risk gain is better data cleaning: remove extreme/invalid coordinate rows (especially the common “0,0” and far-out points) and clamp passenger_count to the valid range in both train/test features. I also add a tiny, standard time feature (`minute`) and a simple interaction (`haversine_km * passenger_count`) which usually reduces error without changing the modeling family or training loop. Finally, I ensure the same feature columns exist in train/test (including passenger_count) and keep the submission format identical.'
- What this solution (achieved 8.23275) has done: 'To reduce RMSE from 6.88 toward the 4.01 target while keeping your XGBoost workflow and feature engineering intact, I make two minimal, high-impact adjustments: (1) train on a slightly larger random sample of the training file (still using `nrows` so it’s safe/time-bounded, but shuffled to avoid any ordering bias), and (2) add a very small set of standard location features (pickup/dropoff distances to a second well-known NYC anchor and their simple interaction) without changing the model family or training loop. I also align the train-time and test-time passenger_count handling so it’s applied exactly once in the same place (to avoid any subtle mismatch). These changes are typically enough to move RMSE materially down while staying within the same core approach and producing the same submission format.'
- What this solution (achieved 7.28875) has done: 'Your current RMSE (8.23275) is worse than the target (4.01327), so we should make the smallest changes that typically yield a large RMSE drop without changing the core approach (XGBRegressor on engineered geo/time features). The biggest likely regression is that you randomly shuffle then take the first 4M rows, which destroys the intended “sample of full distribution” and can bias training; instead, sample directly from the full file (still capped at 4M) and avoid the extra shuffle. Next, add two extremely standard, lightweight airport-anchor features (LGA + EWR distances and diffs) that often help materially with fares while keeping the same feature-engineering pattern. Finally, keep train/test feature alignment deterministic (same columns order) and ensure we still write a valid `sample_submission.csv` with `key,fare_amount`.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os

print(os.listdir("../input"))



## === cell 1
TRAIN_PATH = "../input/train.csv"
TEST_PATH = "../input/test.csv"

SAMPLE_N = 4_000_000
CHUNKSIZE = 1_000_000
rng = np.random.RandomState(42)

reservoir = []
seen = 0
for chunk in pd.read_csv(TRAIN_PATH, chunksize=CHUNKSIZE):
    n = len(chunk)
    if seen == 0 and n >= SAMPLE_N:
        reservoir.append(chunk.sample(n=SAMPLE_N, random_state=42))
        seen += n
        break

    if seen < SAMPLE_N:
        need = SAMPLE_N - sum(len(x) for x in reservoir)
        take = min(need, n)
        reservoir.append(
            chunk.sample(n=take, random_state=int(rng.randint(0, 2**31 - 1)))
        )
    else:
        current = pd.concat(reservoir, ignore_index=True)
        replace_prob = n / float(seen + n)
        k_replace = int(replace_prob * SAMPLE_N)
        if k_replace > 0:
            rep = chunk.sample(
                n=min(k_replace, n), random_state=int(rng.randint(0, 2**31 - 1))
            )
            drop_idx = rng.choice(current.index.values, size=len(rep), replace=False)
            current = current.drop(index=drop_idx).reset_index(drop=True)
            current = pd.concat([current, rep], ignore_index=True)
            reservoir = [current]
    seen += n

training_data = pd.concat(reservoir, ignore_index=True)

test_data = pd.read_csv(TEST_PATH)



## === cell 2
training_data



## === cell 3
X_train = training_data.copy()
X_test = test_data.copy()
Y_train = training_data.copy()




## === cell 4
def _clean_train(df: pd.DataFrame) -> pd.DataFrame:
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

    df = df[(df["fare_amount"] > 0) & (df["fare_amount"] < 250)]
    df = df[(df["passenger_count"] >= 1) & (df["passenger_count"] <= 6)]

    df = df[
        (df["pickup_longitude"].between(-74.3, -72.9))
        & (df["dropoff_longitude"].between(-74.3, -72.9))
        & (df["pickup_latitude"].between(40.5, 41.0))
        & (df["dropoff_latitude"].between(40.5, 41.0))
    ]

    bad_pick = (df["pickup_longitude"].abs() < 1) & (df["pickup_latitude"].abs() < 1)
    bad_drop = (df["dropoff_longitude"].abs() < 1) & (df["dropoff_latitude"].abs() < 1)
    df = df[~(bad_pick | bad_drop)]

    same_coord = (df["pickup_longitude"] == df["dropoff_longitude"]) & (
        df["pickup_latitude"] == df["dropoff_latitude"]
    )
    df = df[~same_coord]

    return df


training_data = _clean_train(training_data)

X_train = training_data.copy()
Y_train = training_data.copy()




## === cell 5
def _haversine_km(lon1, lat1, lon2, lat2):
    lon1 = np.radians(np.asarray(lon1, dtype=float))
    lat1 = np.radians(np.asarray(lat1, dtype=float))
    lon2 = np.radians(np.asarray(lon2, dtype=float))
    lat2 = np.radians(np.asarray(lat2, dtype=float))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371.0 * c


def _bearing(lon1, lat1, lon2, lat2):
    lon1 = np.radians(np.asarray(lon1, dtype=float))
    lat1 = np.radians(np.asarray(lat1, dtype=float))
    lon2 = np.radians(np.asarray(lon2, dtype=float))
    lat2 = np.radians(np.asarray(lat2, dtype=float))
    dlon = lon2 - lon1
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return (np.degrees(np.arctan2(y, x)) + 360.0) % 360.0


NYC_LON, NYC_LAT = (
    -73.985428,
    40.748817,
)  # Manhattan (Empire State Building) as a robust center proxy

JFK_LON, JFK_LAT = -73.7781, 40.6413

LGA_LON, LGA_LAT = -73.8740, 40.7769
EWR_LON, EWR_LAT = -74.1745, 40.6895


def _add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], errors="coerce")

    df["hour"] = df["pickup_datetime"].dt.hour
    df["minute"] = df["pickup_datetime"].dt.minute
    df["dayofweek"] = df["pickup_datetime"].dt.dayofweek
    df["month"] = df["pickup_datetime"].dt.month
    df["year"] = df["pickup_datetime"].dt.year
    df["dayofyear"] = df["pickup_datetime"].dt.dayofyear

    df["latitude_distance"] = (df["dropoff_latitude"] - df["pickup_latitude"]).abs()
    df["longitude_distance"] = (df["dropoff_longitude"] - df["pickup_longitude"]).abs()
    df["manhattan_approx"] = df["latitude_distance"] + df["longitude_distance"]

    df["delta_lat"] = (df["dropoff_latitude"] - df["pickup_latitude"]).astype(float)
    df["delta_lon"] = (df["dropoff_longitude"] - df["pickup_longitude"]).astype(float)
    df["delta_lat_x_delta_lon"] = df["delta_lat"] * df["delta_lon"]

    df["haversine_km"] = _haversine_km(
        df["pickup_longitude"],
        df["pickup_latitude"],
        df["dropoff_longitude"],
        df["dropoff_latitude"],
    )
    df["bearing"] = _bearing(
        df["pickup_longitude"],
        df["pickup_latitude"],
        df["dropoff_longitude"],
        df["dropoff_latitude"],
    )

    df["pickup_to_center_km"] = _haversine_km(
        df["pickup_longitude"], df["pickup_latitude"], NYC_LON, NYC_LAT
    )
    df["dropoff_to_center_km"] = _haversine_km(
        df["dropoff_longitude"], df["dropoff_latitude"], NYC_LON, NYC_LAT
    )
    df["center_dist_diff_km"] = (
        df["dropoff_to_center_km"] - df["pickup_to_center_km"]
    ).astype(float)

    df["pickup_to_jfk_km"] = _haversine_km(
        df["pickup_longitude"], df["pickup_latitude"], JFK_LON, JFK_LAT
    )
    df["dropoff_to_jfk_km"] = _haversine_km(
        df["dropoff_longitude"], df["dropoff_latitude"], JFK_LON, JFK_LAT
    )
    df["jfk_dist_diff_km"] = (df["dropoff_to_jfk_km"] - df["pickup_to_jfk_km"]).astype(
        float
    )

    df["pickup_to_lga_km"] = _haversine_km(
        df["pickup_longitude"], df["pickup_latitude"], LGA_LON, LGA_LAT
    )
    df["dropoff_to_lga_km"] = _haversine_km(
        df["dropoff_longitude"], df["dropoff_latitude"], LGA_LON, LGA_LAT
    )
    df["lga_dist_diff_km"] = (df["dropoff_to_lga_km"] - df["pickup_to_lga_km"]).astype(
        float
    )

    df["pickup_to_ewr_km"] = _haversine_km(
        df["pickup_longitude"], df["pickup_latitude"], EWR_LON, EWR_LAT
    )
    df["dropoff_to_ewr_km"] = _haversine_km(
        df["dropoff_longitude"], df["dropoff_latitude"], EWR_LON, EWR_LAT
    )
    df["ewr_dist_diff_km"] = (df["dropoff_to_ewr_km"] - df["pickup_to_ewr_km"]).astype(
        float
    )

    if "passenger_count" in df.columns:
        df["passenger_count"] = (
            pd.to_numeric(df["passenger_count"], errors="coerce").fillna(1).astype(int)
        )
        df["passenger_count"] = df["passenger_count"].clip(1, 6)
        df["haversine_x_passengers"] = df["haversine_km"] * df["passenger_count"]
        df["jfk_x_passengers"] = df["pickup_to_jfk_km"] * df["passenger_count"]

        df["lga_x_passengers"] = df["pickup_to_lga_km"] * df["passenger_count"]
        df["ewr_x_passengers"] = df["pickup_to_ewr_km"] * df["passenger_count"]

    return df


X_train = _add_features(X_train)

X_train["_fare_amount_tmp"] = training_data["fare_amount"].values
X_train = X_train[(X_train["haversine_km"] > 0.05) & (X_train["haversine_km"] < 100.0)]

fare_per_km = X_train["_fare_amount_tmp"] / (X_train["haversine_km"] + 1e-6)
X_train = X_train[(fare_per_km > 0.5) & (fare_per_km < 100.0)]

training_data = training_data.loc[X_train.index].copy()
Y_train = training_data.copy()

X_train = X_train.drop(
    columns=[
        "key",
        "pickup_datetime",
        "_fare_amount_tmp",
    ]
)



## === cell 6
X_test = _add_features(X_test)

X_test = X_test.drop(
    columns=[
        "key",
        "pickup_datetime",
    ]
)



## === cell 7
Y_train = Y_train.drop(
    columns=[
        "key",
        "pickup_datetime",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
    ]
)
Y_train = Y_train["fare_amount"]



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

if "fare_amount" in X_train.columns:
    X_train = X_train.drop(columns=["fare_amount"])

common_cols = sorted([c for c in X_train.columns if c in X_test.columns])
X_train = X_train[common_cols]
X_test = X_test[common_cols]

model = xgb.XGBRegressor(
    objective="reg:squarederror",
    n_estimators=600,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
)

model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)
Y_pred = np.clip(Y_pred, 0, None)



## === cell 10
from matplotlib import pyplot as plt
from xgboost import plot_importance

plot_importance(model)
plt.show()



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
