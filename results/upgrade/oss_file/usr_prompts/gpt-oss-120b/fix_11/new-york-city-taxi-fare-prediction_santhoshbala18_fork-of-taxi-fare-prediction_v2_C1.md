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
lightgbm==4.6.0
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

3.70513

# 6. Current score

4.96685

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 5.34467) has done: 'Implemented minimal bug fixes so the pipeline runs end‑to‑end and creates a proper `submission.csv`.  
- Removed the unsupported `silent` argument from `lgb.Dataset`.  
- Corrected the typo `reg_aplha` → `reg_alpha` in LightGBM parameters.  
- Simplified prediction calls (no `num_iteration` needed).  
- Clip negative predictions to 0 (reasonable for fare amounts).  
- Added brief comments for each fix.'
- What this solution (achieved 5.24735) has done: 'I remove the unsupported `verbose_eval` argument from the LightGBM `train` call, which caused the TypeError and prevented the model from being trained. This allows the script to create a valid `model` object, so subsequent prediction and submission steps work correctly, yielding a proper `submission.csv`. No other logic changes are needed.'
- What this solution (achieved 5.16554) has done: 'Implemented the missing imports, fixed undefined variables, corrected the LightGBM parameter typo, and added a modest hyper‑parameter tweak (increased `num_leaves`) to improve model capacity. All cells now run sequentially, producing a valid `submission.csv` with the required columns.'
- What this solution (achieved 5.17347) has done: 'I add a few inexpensive features that often help fare prediction – a squared distance term and sinusoidal hour encodings – and give LightGBM a slightly larger tree capacity (num_leaves = 127). These changes keep the original pipeline intact while providing the model with richer information, which should lower the RMSE toward the target without over‑hauling the approach.'
- What this solution (achieved 5.04866) has done: 'I add a modest interaction feature (`passenger_distance`), filter out extreme fare and distance values that hurt regression, and slightly increase model capacity (more leaves) while lowering the learning rate. These tweaks keep the original pipeline intact but give LightGBM richer information and a cleaner training set, which should lower the RMSE toward the target without over‑hauling the approach.'
- What this solution (achieved 5.43487) has done: 'I add a small “distance per passenger” feature, include it in the model, and remove the log‑transform of the target so the model is trained directly on the fare amount.  These minimal tweaks keep the core pipeline unchanged while giving LightGBM a slightly richer signal and a more appropriate loss for the RMSE metric, which should move the validation score closer to the target.'
- What this solution (achieved 4.96685) has done: 'I train the model on `log1p(fare_amount)` instead of the raw fare, then exponentiate predictions for evaluation and submission. This transformation usually stabilises variance and improves RMSE. I also increase `num_leaves` to give the trees a bit more capacity while keeping the rest of the pipeline unchanged.'

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import lightgbm as lgbm
import math

df = pd.read_csv("../input/train.csv", nrows=10**6)
test_set = pd.read_csv("../input/test.csv")




## === cell 1
def distance(lat1, lon1, lat2, lon2):
    """Haversine distance in kilometres."""
    p = 0.017453292519943295  # pi/180
    a = (
        0.5
        - np.cos((lat2 - lat1) * p) / 2
        + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    )
    return 12742 * np.arcsin(np.sqrt(a))  # 2*R*asin...


df["distance_km"] = distance(
    df["pickup_latitude"],
    df["pickup_longitude"],
    df["dropoff_latitude"],
    df["dropoff_longitude"],
)
df["distance_km_sq"] = df["distance_km"] ** 2

test_set["distance_km"] = distance(
    test_set["pickup_latitude"],
    test_set["pickup_longitude"],
    test_set["dropoff_latitude"],
    test_set["dropoff_longitude"],
)
test_set["distance_km_sq"] = test_set["distance_km"] ** 2



## === cell 2
BB = (-75, -73, 40, 41.5)


def select_within_boundingbox(df, BB):
    return (
        (df["pickup_longitude"] >= BB[0])
        & (df["pickup_longitude"] <= BB[1])
        & (df["pickup_latitude"] >= BB[2])
        & (df["pickup_latitude"] <= BB[3])
        & (df["dropoff_longitude"] >= BB[0])
        & (df["dropoff_longitude"] <= BB[1])
        & (df["dropoff_latitude"] >= BB[2])
        & (df["dropoff_latitude"] <= BB[3])
    )


print("Old size:", len(df))
df = df[select_within_boundingbox(df, BB)]
df = df[(df["passenger_count"] > 0) & (df["passenger_count"] < 10)]
print("New size:", len(df))




## === cell 3
def add_datetime_info(dataset):
    dataset["pickup_datetime"] = pd.to_datetime(dataset["pickup_datetime"])
    dataset["hour"] = dataset["pickup_datetime"].dt.hour
    dataset["day"] = dataset["pickup_datetime"].dt.day
    dataset["month"] = dataset["pickup_datetime"].dt.month
    dataset["weekday"] = dataset["pickup_datetime"].dt.weekday
    dataset["hour_sin"] = np.sin(2 * np.pi * dataset["hour"] / 24)
    dataset["hour_cos"] = np.cos(2 * np.pi * dataset["hour"] / 24)
    return dataset


df = add_datetime_info(df)
test_set = add_datetime_info(test_set)



## === cell 4
df["is_night"] = np.where(
    ((df["hour"] >= 20) & (df["hour"] <= 23)) | ((df["hour"] >= 0) & (df["hour"] < 6)),
    1,
    0,
)
df["is_airport"] = np.where(
    ((df["dropoff_longitude"] >= 73.77) & (df["dropoff_longitude"] <= 73.78))
    | ((df["dropoff_latitude"] >= 40.63) & (df["dropoff_latitude"] <= 40.64)),
    1,
    0,
)
df["is_surge"] = np.where(
    ((df["hour"] >= 16) & (df["hour"] < 20))
    & (df["weekday"] != 5)
    & (df["weekday"] != 6),
    1,
    0,
)
df["passenger_distance"] = df["passenger_count"] * df["distance_km"]
df["dist_per_passenger"] = df["distance_km"] / df["passenger_count"].replace(0, np.nan)
df["dist_per_passenger"] = df["dist_per_passenger"].fillna(0)



## === cell 5
df = df.drop(df[df["fare_amount"] < 0].index, axis=0)
df = df[(df["fare_amount"] <= 200) & (df["distance_km"] <= 100)]



## === cell 6
plt.figure(figsize=(6, 4))
plt.scatter(df["is_night"], df["fare_amount"], c="r", alpha=0.3)
plt.xlabel("is_night")
plt.ylabel("fare_amount")
plt.title("Fare amount vs Night flag")
plt.show()



## === cell 7
feature_cols = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
    "distance_km",
    "distance_km_sq",
    "hour",
    "hour_sin",
    "hour_cos",
    "day",
    "month",
    "weekday",
    "is_night",
    "is_airport",
    "is_surge",
    "passenger_distance",
    "dist_per_passenger",  # newly added feature
]

X = df[feature_cols]
y = df["fare_amount"]  # raw target
y_log = np.log1p(y)  # transformed target for training

X_train, X_val, y_train_log, y_val_log = train_test_split(
    X, y_log, test_size=0.1, random_state=42
)

y_val_original = np.expm1(y_val_log)



## === cell 8
params = {
    "boosting_type": "gbdt",
    "objective": "regression",
    "nthread": -1,
    "verbose": -1,
    "num_leaves": 511,  # slightly larger capacity
    "learning_rate": 0.03,
    "max_depth": -1,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.6,
    "reg_alpha": 1,
    "reg_lambda": 0.001,
    "metric": "rmse",
    "min_split_gain": 0.5,
    "min_child_weight": 1,
    "min_child_samples": 10,
    "seed": 42,
}

train_set = lgbm.Dataset(X_train, label=y_train_log)
valid_set = lgbm.Dataset(X_val, label=y_val_log, reference=train_set)

callbacks = [lgbm.early_stopping(stopping_rounds=100, verbose=False)]

model = lgbm.train(
    params,
    train_set=train_set,
    num_boost_round=2000,
    valid_sets=[valid_set],
    callbacks=callbacks,
)



## === cell 9
val_pred_log = model.predict(X_val, num_iteration=model.best_iteration)
val_pred = np.expm1(val_pred_log)  # back‑transform
rmse = math.sqrt(mean_squared_error(y_val_original, val_pred))
print("Validation RMSE:", rmse)



## === cell 10
test_set["is_night"] = np.where(
    ((test_set["hour"] >= 20) & (test_set["hour"] <= 23))
    | ((test_set["hour"] >= 0) & (test_set["hour"] < 6)),
    1,
    0,
)
test_set["is_airport"] = np.where(
    (
        (test_set["dropoff_longitude"] >= 73.77)
        & (test_set["dropoff_longitude"] <= 73.78)
    )
    | (
        (test_set["dropoff_latitude"] >= 40.63)
        & (test_set["dropoff_latitude"] <= 40.64)
    ),
    1,
    0,
)
test_set["is_surge"] = np.where(
    ((test_set["hour"] >= 16) & (test_set["hour"] < 20))
    & (test_set["weekday"] != 5)
    & (test_set["weekday"] != 6),
    1,
    0,
)
test_set["passenger_distance"] = test_set["passenger_count"] * test_set["distance_km"]
test_set["dist_per_passenger"] = test_set["distance_km"] / test_set[
    "passenger_count"
].replace(0, np.nan)
test_set["dist_per_passenger"] = test_set["dist_per_passenger"].fillna(0)

test_features = test_set[feature_cols]
test_keys = test_set["key"]

test_pred_log = model.predict(test_features, num_iteration=model.best_iteration)
test_pred = np.maximum(0, np.expm1(test_pred_log))  # back‑transform & clip
submission = pd.DataFrame(
    {"key": test_keys, "fare_amount": test_pred},
    columns=["key", "fare_amount"],
)
submission.to_csv("submission.csv", index=False)
print("Submission file written: submission.csv")



## === cell 11
print("Submission shape:", submission.shape)
