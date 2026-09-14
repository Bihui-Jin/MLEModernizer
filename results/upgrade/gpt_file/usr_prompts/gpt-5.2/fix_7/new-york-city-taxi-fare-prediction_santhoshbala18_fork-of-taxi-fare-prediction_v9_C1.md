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

6.11378

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 7.4263) has done: 'Your current score is far worse than the target (RMSE 8.96 vs 3.35), so we should make small but high-impact correctness fixes rather than tuning for the best possible model. The biggest issue is feature mismatch: you train with `is_night/is_surge/is_airport` but later drop some columns only for the train/valid split (not for the fitted model) and you also never add `is_BeforeRaise` to training, so test-time columns don’t match what the model expects. I (1) create the exact same feature set for train/valid/test, (2) fix the incorrect airport longitude sign (NYC longitudes are negative), and (3) add a simple, standard fare cleanup (remove extreme fares) to stabilize RMSE—without changing the model type or training loop. The script still train an `XGBRegressor` and write a valid `submission_combo.csv`.'
- What this solution (achieved 5.40344) has done: 'Your current issue is that you can’t reliably measure/iterate because the run errors out before producing a graded result, so the most helpful “score toward target” move is to make the pipeline deterministic, valid, and metric-aligned while keeping the same XGBRegressor training loop and feature logic. I keep your exact feature set and model, but (1) ensure train/test use identical feature columns in the same order and with safe numeric dtypes, (2) add a minimal, standard post-processing step to clip negative predictions (invalid for fares) which usually improves RMSE without changing core modeling, and (3) write the submission to both `submission_combo.csv` and `/kaggle/working/submission_combo.csv` to ensure the file is found by graders that expect it in the working directory. These are minimal, semantics-preserving fixes that should move RMSE down toward the target band without changing the architecture or training approach.'
- What this solution (achieved 5.34723) has done: 'Your current RMSE (5.403) is worse than the target (3.353), so we should make small, reliable improvements without changing the model type or overall training flow. The biggest gain available with minimal change is to train the same XGBRegressor on a log-transformed target (still RMSE-optimized via squared error, but it reduces the influence of outliers), then invert the transform for predictions; this typically improves Taxi Fare RMSE substantially. I also switch the train/validation split to be stratified by distance bins (same split method and loop, but less distribution shift), and align XGBoost’s objective/eval_metric explicitly with RMSE for stability. Submission format and file paths remain unchanged, and we still clip negative fares to 0.'
- What this solution (achieved 6.04059) has done: 'You’re already training a solid XGBRegressor, but your RMSE (5.35) is still far from the target (3.35), so the biggest “minimal change” gains come from (1) slightly better feature signal without changing the model/training loop, and (2) stronger, standard NYC Taxi cleanup that removes impossible coordinates and extreme-distance artifacts that inflate RMSE. I add two classic, lightweight geospatial features (Haversine distance in NYC plus Manhattan distance proxy) while keeping your existing `distance_km` and all flags, and I tighten the data filtering with known-valid longitude/latitude ranges and a fare-per-km sanity filter. This preserves your overall approach (same model type, objective, training flow, log1p target) but typically moves RMSE materially downward. Submission writing stays identical and still produces `submission_combo.csv`.'
- What this solution (achieved 6.11378) has done: 'You’re well worse than the target (RMSE 6.04 vs 3.35, lower is better), so the smallest high-impact move is to fix metric alignment: Kaggle evaluates RMSE on raw dollars, but your model is trained on `log1p(fare)` with plain squared error, which is not the same objective and can hurt RMSE even if it helps relative-error. I keep the same XGBRegressor training flow and all your features/cleaning, but train directly on `fare_amount` (no log transform) and predict in dollars. To stabilize RMSE without changing the approach, I also add a single, standard sanity clip on extreme predictions (0..250) consistent with your training filter and keep the submission format identical.'

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
df = df[select_within_boundingbox(df, BB)]
df = df[(df.passenger_count > 0) & (df.passenger_count <= 6)]
print("New size: %d" % len(df))




## === cell 3
def add_datetime_info(dataset):
    dataset["pickup_datetime"] = pd.to_datetime(dataset["pickup_datetime"])
    dataset["hour"] = dataset.pickup_datetime.dt.hour
    dataset["day"] = dataset.pickup_datetime.dt.day
    dataset["month"] = dataset.pickup_datetime.dt.month
    dataset["weekday"] = dataset.pickup_datetime.dt.weekday
    dataset["year"] = dataset.pickup_datetime.dt.year
    return dataset


df = add_datetime_info(df)
test_set = add_datetime_info(test_set)




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
df = df[(df["abs_diff_longitude"] <= 2.0) & (df["abs_diff_latitude"] <= 2.0)].copy()

df = df[
    (df["pickup_longitude"].between(-74.3, -72.9))
    & (df["dropoff_longitude"].between(-74.3, -72.9))
    & (df["pickup_latitude"].between(40.4, 41.0))
    & (df["dropoff_latitude"].between(40.4, 41.0))
].copy()

fpk = df["fare_amount"] / (df["haversine_km"] + 1e-3)
df = df[fpk.between(1.0, 50.0)].copy()



## === cell 6
from sklearn.model_selection import train_test_split

DROP_COLS = ["key", "fare_amount", "pickup_datetime"]
FEATURE_COLS = [c for c in df.columns if c not in DROP_COLS]

X = df[FEATURE_COLS].copy()

y = df["fare_amount"].astype(np.float32)

for c in X.columns:
    X[c] = pd.to_numeric(X[c], errors="coerce")
X = X.fillna(0.0).astype(np.float32)

dist_bins = pd.qcut(X["distance_km"], q=10, duplicates="drop")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=113, stratify=dist_bins
)



## === cell 7
y_train.mean()



## === cell 8
from xgboost import XGBRegressor

regressor = XGBRegressor(
    max_depth=10,
    learning_rate=0.1,
    n_estimators=200,
    random_state=113,
    n_jobs=4,
    objective="reg:squarederror",
    eval_metric="rmse",
)
regressor.fit(X_train, y_train)



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
test_set_features = test_set[FEATURE_COLS].copy()
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
