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

6.00586

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1125.15464) has done: 'I fix the runtime crash by switching from `keras` to `tf_keras` (TensorFlow Keras), which is compatible with Kaggle’s TF/protobuf stack and avoids the `MessageFactory.GetPrototype` error. I also fix a key logic issue that hurts score: you scale train features but scale test features independently; instead I fit a scaler on train and reuse it on validation/test to match feature distributions. To nudge RMSE toward the target, I make the train/validation split deterministic and use `validation_data` while keeping the same model, optimizer, loss, and epoch count. Finally, I ensure the submission has the correct `key,fare_amount` format and that predictions are 1D floats.'
- What this solution (achieved 5.49769) has done: 'I fix the protobuf/Keras import crash by avoiding `tf_keras` and using `sklearn`’s stable regressor instead, while keeping the same feature engineering and train/valid split semantics so the pipeline runs end-to-end. To move RMSE dramatically toward the target, I replace the neural net (which currently produces wildly wrong fares due to training instability/scale issues) with a well-calibrated `RandomForestRegressor` on the exact same engineered features. I also ensure predictions are finite and non-negative (fares can’t be negative) and that the submission has exactly `key,fare_amount` and a `.csv` suffix. All paths remain unchanged and the script write `submission.csv` in the working directory.'
- What this solution (achieved 5.28319) has done: 'Your current RMSE (5.49769) is worse than the target (3.41966), so we should carefully improve generalization without changing the core modeling approach (RandomForest on the same engineered features). The largest low-risk gain here is to use a slightly more appropriate distance feature (Haversine in km) while keeping your existing Euclidean distance feature (so we don’t remove anything), and to tune a couple of RandomForest regularization knobs (e.g., `max_features`, `min_samples_leaf`) to reduce overfitting and improve RMSE. I also ensure all numeric columns are consistent float32 and drop any remaining non-numeric leakage, while keeping the same train/valid split semantics and still writing `submission.csv` with `key,fare_amount`. These changes are minimal, fast, and typically move NYC taxi fare baselines closer to ~3–4 RMSE.'
- What this solution (achieved 5.59868) has done: 'We keep your exact feature set and RandomForest approach, but make two small changes that typically reduce RMSE toward your target: (1) add a lightweight outlier filter on the engineered distance features (very long “trips” inside the NYC bbox are usually bad GPS rows that hurt generalization), and (2) slightly increase forest averaging while regularizing a bit more via `min_samples_leaf`, which often improves this competition’s RMSE without changing core logic. We also train on the full filtered dataset after printing a validation RMSE (so the final model uses all available data), then generate the same `key,fare_amount` submission. All paths and submission format remain unchanged.'
- What this solution (achieved 5.81707) has done: 'The timeout is dominated by fitting a very large RandomForest (900 trees) on up to ~2M rows and computing extra predictions on the full training split; this is far beyond what fits in 600 seconds on Kaggle CPU. To preserve core logic (same model type, same features, same training flow), the main speed fix is to train on a deterministic, in-memory random subset of the already-loaded 2M rows (this keeps the algorithm identical but reduces asymptotic cost). Additionally, we remove redundant type-conversion loops (everything is already numeric), avoid materializing extra copies, and skip the expensive `predict(X_train)` RMSE (OOB RMSE already provides validation semantics and avoids a second full pass). These changes are equivalence-preserving for feature generation and prediction, and deterministically reduce runtime to meet the 600-second limit.'
- What this solution (achieved 6.00586) has done: 'Your current RMSE (5.817) is worse than the target (3.420), so we should make the smallest changes that improve generalization without changing the core approach (same features + RandomForestRegressor). The biggest issue is that your model is training on only 250k rows; in this competition, RMSE improves materially as you use more clean data, so we increase the deterministic fit cap while keeping the same sampling semantics to stay within runtime. We also remove a likely over-aggressive `travel_distance <= 1.0` filter (it can discard many valid trips and bias the training distribution), while keeping the NYC bbox and fare filters and retaining the `haversine_km` sanity cap. Finally, we reduce `n_estimators` so the larger training sample still fits under the 600s budget, aiming to move RMSE closer to the target band rather than maximize.'

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



## === cell 7
mask = (
    (df["pickup_longitude"] > nyc_min_longitude)
    & (df["pickup_longitude"] < nyc_max_longitude)
    & (df["dropoff_longitude"] > nyc_min_longitude)
    & (df["dropoff_longitude"] < nyc_max_longitude)
    & (df["pickup_latitude"] > nyc_min_latitude)
    & (df["pickup_latitude"] < nyc_max_latitude)
    & (df["dropoff_latitude"] > nyc_min_latitude)
    & (df["dropoff_latitude"] < nyc_max_latitude)
)
df = df.loc[mask].copy()



## === cell 8
df.loc[df["passenger_count"] == 0, "passenger_count"] = 1



## === cell 9
df = df[(df["fare_amount"] > 0) & (df["fare_amount"] <= 100)].copy()



## === cell 10
print(df.shape)



## === cell 11
df["longitude_diff"] = df["dropoff_longitude"] - df["pickup_longitude"]
df["latitude_diff"] = df["dropoff_latitude"] - df["pickup_latitude"]



## === cell 12
test["longitude_diff"] = test["dropoff_longitude"] - test["pickup_longitude"]
test["latitude_diff"] = test["dropoff_latitude"] - test["pickup_latitude"]



## === cell 13
df["year"] = df["pickup_datetime"].dt.year.astype(np.int16)
df["month"] = df["pickup_datetime"].dt.month.astype(np.int8)
df["day"] = df["pickup_datetime"].dt.day.astype(np.int8)
df["day_of_week"] = df["pickup_datetime"].dt.dayofweek.astype(np.int8)
df["hour"] = df["pickup_datetime"].dt.hour.astype(np.int8)
df = df.drop(["pickup_datetime"], axis=1)



## === cell 14
test["year"] = test["pickup_datetime"].dt.year.astype(np.int16)
test["month"] = test["pickup_datetime"].dt.month.astype(np.int8)
test["day"] = test["pickup_datetime"].dt.day.astype(np.int8)
test["day_of_week"] = test["pickup_datetime"].dt.dayofweek.astype(np.int8)
test["hour"] = test["pickup_datetime"].dt.hour.astype(np.int8)
test = test.drop(["pickup_datetime"], axis=1)



## === cell 15
test_keys = test["key"].copy()




## === cell 16
def euc_distance(lat1, long1, lat2, long2):
    return ((lat1 - lat2) ** 2 + (long1 - long2) ** 2) ** 0.5


df["travel_distance"] = euc_distance(
    df["pickup_latitude"],
    df["pickup_longitude"],
    df["dropoff_latitude"],
    df["dropoff_longitude"],
).astype(np.float32)
test["travel_distance"] = euc_distance(
    test["pickup_latitude"],
    test["pickup_longitude"],
    test["dropoff_latitude"],
    test["dropoff_longitude"],
).astype(np.float32)


def haversine_km(lat1, lon1, lat2, lon2):
    lat1 = np.radians(lat1.astype(np.float64))
    lon1 = np.radians(lon1.astype(np.float64))
    lat2 = np.radians(lat2.astype(np.float64))
    lon2 = np.radians(lon2.astype(np.float64))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * (
        np.sin(dlon / 2.0) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(a))
    R = 6371.0088
    return (R * c).astype(np.float32)


df["haversine_km"] = haversine_km(
    df["pickup_latitude"],
    df["pickup_longitude"],
    df["dropoff_latitude"],
    df["dropoff_longitude"],
)
test["haversine_km"] = haversine_km(
    test["pickup_latitude"],
    test["pickup_longitude"],
    test["dropoff_latitude"],
    test["dropoff_longitude"],
)



## === cell 17
pass



## === cell 18
pass



## === cell 19
print("train nulls:", int(df.isnull().values.sum()))
print("test nulls:", int(test.isnull().values.sum()))



## === cell 20
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

RANDOM_STATE = 42

MAX_TRAIN_ROWS_FOR_FIT = 1_000_000  # deterministic cap; still within 2M loaded rows

df_model = df.drop(["key"], axis=1).copy()
test_model = test.drop(["key"], axis=1).copy()

df_model.dropna(inplace=True)  # safe: only affects train rows
test_model = test_model.fillna(0.0)

if "haversine_km" in df_model.columns:
    df_model = df_model[
        (df_model["haversine_km"] >= 0.0) & (df_model["haversine_km"] <= 35.0)
    ]


if len(df_model) > MAX_TRAIN_ROWS_FOR_FIT:
    df_model = df_model.sample(n=MAX_TRAIN_ROWS_FOR_FIT, random_state=RANDOM_STATE)

X = df_model.loc[:, df_model.columns != "fare_amount"].astype(np.float32, copy=False)
y = df_model["fare_amount"].astype(np.float32, copy=False)
test_model = test_model[X.columns].astype(np.float32, copy=False)

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)



## === cell 21
model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1,
    min_samples_leaf=5,
    max_features="sqrt",
    oob_score=True,
    bootstrap=True,
)

model.fit(X_train, y_train)

oob_pred = model.oob_prediction_
valid_rmse = np.sqrt(mean_squared_error(y_train, oob_pred))

print("Valid RMSE (OOB): {:0.2f}".format(valid_rmse))
print("------------------------")



## === cell 22
pred = model.predict(test_model).astype(np.float32)

pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
pred = np.clip(pred, 0.0, None)



## === cell 23
print(pred.shape)



## === cell 24
submission = pd.DataFrame({"key": test_keys.values, "fare_amount": pred})
submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
