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

3.47309

# 6. Current score

4.56836

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 8.4997) has done: 'I fix the XGBoost inference crash by removing the deprecated `ntree_limit=model.best_ntree_limit` usage (XGBoost 2.x no longer exposes that attribute on the Booster returned by `xgb.train`) and instead use `iteration_range` when early stopping is enabled. I also make the data paths robust to your environment (preferring `/kaggle/input/...` and falling back to `../input/...`) so the notebook runs end-to-end on Kaggle. Finally, I ensure the submission is always created with exactly the required columns (`key`, `fare_amount`) and a `.csv` filename. These changes are score-neutral to slightly positive because they preserve the same training logic and simply make inference consistent with the best iteration chosen by early stopping.'
- What this solution (achieved 4.52686) has done: 'Your current RMSE (8.4997) is far worse than the target (3.47309), so we should make small, legitimate changes that improve generalization without changing the overall “XGBoost on engineered geo+time features” approach. The biggest issue is that the model is effectively undertrained: `early_stopping_rounds=100` with `num_boost_round=100` stops after just 1–2 trees and also uses the deprecated `reg:linear` objective. I keep the same training loop and feature set, but (1) switch to the correct modern objective name, (2) increase `num_boost_round` so early stopping can actually work, and (3) add the same basic coordinate/passenger cleaning to test as train plus a non-negative clip on predictions (fares can’t be negative), which usually improves RMSE without changing semantics. These are minimal, metric-aligned fixes that should move the score substantially toward the target band while still running within the time limit.'
- What this solution (achieved 4.3976) has done: 'Your current score (4.52686) is still worse than the target (3.47309), so we should make a small, metric-aligned improvement without changing the overall “XGBoost on engineered geo+time features” approach. The biggest remaining issue is that the datetime-derived features are treated as continuous numbers even though they’re categorical (hour/weekday/month/day), which commonly hurts XGBoost splits for this competition; enabling native categorical handling and casting those columns to `category` is a minimal change that typically improves RMSE. I also add a tiny bit of stable regularization (`min_child_weight`, `subsample`, `colsample_bytree`) while keeping the same training loop, features, and loss to reduce overfitting and improve generalization. Finally, I ensure train/test feature dtypes match exactly so inference uses the same representation as training.'
- What this solution (achieved 4.56836) has done: 'To move RMSE down from 4.3976 toward the 3.47309 target without changing your overall “XGBoost on engineered geo+time features” approach, I make two minimal, metric-aligned fixes: (1) remove unnecessary rounding to 2 decimals in the submission (rounding almost always worsens RMSE), and (2) ensure the train/validation split is not random but time-based (using `pickup_datetime`) to prevent temporal leakage that can hurt public/private generalization on this competition. Everything else (features, XGBoost training loop, objective, early stopping, categorical handling) stays the same. The code still run end-to-end within constraints and always write a valid `taxi_fare_submission.csv` with `key,fare_amount`.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import datetime as dt
from sklearn.model_selection import train_test_split
import xgboost as xgb
import os

INPUT_DIR_CANDIDATES = [
    "/kaggle/input",
    "../input",
    "/kaggle/data",  # user-provided environment shows /kaggle/data and /kaggle/input
]
INPUT_DIR = None
for cand in INPUT_DIR_CANDIDATES:
    if os.path.isdir(cand):
        INPUT_DIR = cand
        break

if INPUT_DIR is None:
    raise FileNotFoundError(
        "Could not find Kaggle input directory among: " + str(INPUT_DIR_CANDIDATES)
    )

DATASET_DIR = os.path.join(INPUT_DIR, "new-york-city-taxi-fare-prediction")
if not os.path.isdir(DATASET_DIR):
    DATASET_DIR = INPUT_DIR

print("Using INPUT_DIR:", INPUT_DIR)
print("Using DATASET_DIR:", DATASET_DIR)
print("Files in INPUT_DIR:", os.listdir(INPUT_DIR)[:50])

TRAIN_PATH = os.path.join(DATASET_DIR, "train.csv")
TEST_PATH = os.path.join(DATASET_DIR, "test.csv")
SAMPLE_SUB_PATH = os.path.join(DATASET_DIR, "sample_submission.csv")




## === cell 1
train_df = pd.read_csv(TRAIN_PATH, nrows=1_000_000)
train_df.dtypes




## === cell 2
print(train_df.isnull().sum())




## === cell 3
train_df = train_df.dropna(how="any", axis="rows")




## === cell 4
train_df.head()




## === cell 5
try:
    train_df.iloc[:1000].plot.scatter("pickup_longitude", "pickup_latitude")
    train_df.iloc[:1000].plot.scatter("dropoff_longitude", "dropoff_latitude")
except Exception as e:
    print("Plotting skipped due to:", repr(e))

train_df.describe()




## === cell 6
def clean_df(df):
    return df[
        (df.fare_amount > 0)
        & (df.pickup_longitude > -80)
        & (df.pickup_longitude < -70)
        & (df.pickup_latitude > 35)
        & (df.pickup_latitude < 45)
        & (df.dropoff_longitude > -80)
        & (df.dropoff_longitude < -70)
        & (df.dropoff_latitude > 35)
        & (df.dropoff_latitude < 45)
        & (df.passenger_count > 0)
        & (df.passenger_count < 10)
    ]


train_df = clean_df(train_df)
print(len(train_df))




## === cell 7
def sphere_dist(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon):
    R_earth = 6371
    pickup_lat, pickup_lon, dropoff_lat, dropoff_lon = map(
        np.radians, [pickup_lat, pickup_lon, dropoff_lat, dropoff_lon]
    )
    dlat = dropoff_lat - pickup_lat
    dlon = dropoff_lon - pickup_lon

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(pickup_lat) * np.cos(dropoff_lat) * np.sin(dlon / 2.0) ** 2
    )
    return 2 * R_earth * np.arcsin(np.sqrt(a))


def add_datetime_info(dataset):
    dataset["pickup_datetime"] = pd.to_datetime(dataset["pickup_datetime"])
    dataset["hour"] = dataset.pickup_datetime.dt.hour
    dataset["day"] = dataset.pickup_datetime.dt.day
    dataset["month"] = dataset.pickup_datetime.dt.month
    dataset["weekday"] = dataset.pickup_datetime.dt.weekday
    return dataset


train_df["distance"] = sphere_dist(
    train_df["pickup_latitude"],
    train_df["pickup_longitude"],
    train_df["dropoff_latitude"],
    train_df["dropoff_longitude"],
)

train_df = add_datetime_info(train_df)

for c in ["hour", "day", "month", "weekday"]:
    train_df[c] = train_df[c].astype("category")

train_df.head()




## === cell 8
train_df.drop(columns=["key", "pickup_datetime"], inplace=True)
train_df.head()




## === cell 9
train_df["pickup_long_15"] = train_df["pickup_longitude"] * np.cos(
    15 * np.pi / 180
) - train_df["pickup_latitude"] * np.sin(15 * np.pi / 180)
train_df["pickup_long_30"] = train_df["pickup_longitude"] * np.cos(
    30 * np.pi / 180
) - train_df["pickup_latitude"] * np.sin(30 * np.pi / 180)
train_df["pickup_long_45"] = train_df["pickup_longitude"] * np.cos(
    45 * np.pi / 180
) - train_df["pickup_latitude"] * np.sin(45 * np.pi / 180)
train_df["pickup_long_60"] = train_df["pickup_longitude"] * np.cos(
    60 * np.pi / 180
) - train_df["pickup_latitude"] * np.sin(60 * np.pi / 180)
train_df["pickup_long_75"] = train_df["pickup_longitude"] * np.cos(
    75 * np.pi / 180
) - train_df["pickup_latitude"] * np.sin(75 * np.pi / 180)

train_df["pickup_lat_15"] = train_df["pickup_longitude"] * np.sin(
    15 * np.pi / 180
) + train_df["pickup_latitude"] * np.cos(15 * np.pi / 180)
train_df["pickup_lat_30"] = train_df["pickup_longitude"] * np.sin(
    30 * np.pi / 180
) + train_df["pickup_latitude"] * np.cos(30 * np.pi / 180)
train_df["pickup_lat_45"] = train_df["pickup_longitude"] * np.sin(
    45 * np.pi / 180
) + train_df["pickup_latitude"] * np.cos(45 * np.pi / 180)
train_df["pickup_lat_60"] = train_df["pickup_longitude"] * np.sin(
    60 * np.pi / 180
) + train_df["pickup_latitude"] * np.cos(60 * np.pi / 180)
train_df["pickup_lat_75"] = train_df["pickup_longitude"] * np.sin(
    75 * np.pi / 180
) + train_df["pickup_latitude"] * np.cos(75 * np.pi / 180)

train_df["dropoff_long_15"] = train_df["dropoff_longitude"] * np.cos(
    15 * np.pi / 180
) - train_df["dropoff_latitude"] * np.sin(15 * np.pi / 180)
train_df["dropoff_long_30"] = train_df["dropoff_longitude"] * np.cos(
    30 * np.pi / 180
) - train_df["dropoff_latitude"] * np.sin(30 * np.pi / 180)
train_df["dropoff_long_45"] = train_df["dropoff_longitude"] * np.cos(
    45 * np.pi / 180
) - train_df["dropoff_latitude"] * np.sin(45 * np.pi / 180)
train_df["dropoff_long_60"] = train_df["dropoff_longitude"] * np.cos(
    60 * np.pi / 180
) - train_df["dropoff_latitude"] * np.sin(60 * np.pi / 180)
train_df["dropoff_long_75"] = train_df["dropoff_longitude"] * np.cos(
    75 * np.pi / 180
) - train_df["dropoff_latitude"] * np.sin(75 * np.pi / 180)

train_df["dropoff_lat_15"] = train_df["dropoff_longitude"] * np.sin(
    15 * np.pi / 180
) + train_df["dropoff_latitude"] * np.cos(15 * np.pi / 180)
train_df["dropoff_lat_30"] = train_df["dropoff_longitude"] * np.sin(
    30 * np.pi / 180
) + train_df["dropoff_latitude"] * np.cos(30 * np.pi / 180)
train_df["dropoff_lat_45"] = train_df["dropoff_longitude"] * np.sin(
    45 * np.pi / 180
) + train_df["dropoff_latitude"] * np.cos(45 * np.pi / 180)
train_df["dropoff_lat_60"] = train_df["dropoff_longitude"] * np.sin(
    60 * np.pi / 180
) + train_df["dropoff_latitude"] * np.cos(60 * np.pi / 180)
train_df["dropoff_lat_75"] = train_df["dropoff_longitude"] * np.sin(
    75 * np.pi / 180
) + train_df["dropoff_latitude"] * np.cos(75 * np.pi / 180)




## === cell 10
y = train_df["fare_amount"]
train = train_df.drop(columns=["fare_amount"])

order_cols = ["month", "day", "hour"]
train_sorted_idx = train[order_cols].sort_values(order_cols).index
train = train.loc[train_sorted_idx]
y = y.loc[train_sorted_idx]

split = int(0.8 * len(train))
x_train, x_test = train.iloc[:split], train.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]




## === cell 11
def XGBmodel(x_train, x_test, y_train, y_test):
    matrix_train = xgb.DMatrix(x_train, label=y_train, enable_categorical=True)
    matrix_test = xgb.DMatrix(x_test, label=y_test, enable_categorical=True)

    model = xgb.train(
        params={
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "seed": 0,
            "eta": 0.1,
            "max_depth": 6,
            "min_child_weight": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "tree_method": "hist",
            "enable_categorical": True,
        },
        dtrain=matrix_train,
        num_boost_round=500,
        early_stopping_rounds=50,
        evals=[(matrix_test, "test")],
        verbose_eval=False,
    )
    return model


model = XGBmodel(x_train, x_test, y_train, y_test)




## === cell 12
test_df = pd.read_csv(TEST_PATH)

test_mask = (
    (test_df.pickup_longitude > -80)
    & (test_df.pickup_longitude < -70)
    & (test_df.pickup_latitude > 35)
    & (test_df.pickup_latitude < 45)
    & (test_df.dropoff_longitude > -80)
    & (test_df.dropoff_longitude < -70)
    & (test_df.dropoff_latitude > 35)
    & (test_df.dropoff_latitude < 45)
    & (test_df.passenger_count > 0)
    & (test_df.passenger_count < 10)
)
fallback_fare = float(y_train.mean())

test_df["distance"] = sphere_dist(
    test_df["pickup_latitude"],
    test_df["pickup_longitude"],
    test_df["dropoff_latitude"],
    test_df["dropoff_longitude"],
)
test_df = add_datetime_info(test_df)

for c in ["hour", "day", "month", "weekday"]:
    test_df[c] = test_df[c].astype("category")

test_df["pickup_long_15"] = test_df["pickup_longitude"] * np.cos(
    15 * np.pi / 180
) - test_df["pickup_latitude"] * np.sin(15 * np.pi / 180)
test_df["pickup_long_30"] = test_df["pickup_longitude"] * np.cos(
    30 * np.pi / 180
) - test_df["pickup_latitude"] * np.sin(30 * np.pi / 180)
test_df["pickup_long_45"] = test_df["pickup_longitude"] * np.cos(
    45 * np.pi / 180
) - test_df["pickup_latitude"] * np.sin(45 * np.pi / 180)
test_df["pickup_long_60"] = test_df["pickup_longitude"] * np.cos(
    60 * np.pi / 180
) - test_df["pickup_latitude"] * np.sin(60 * np.pi / 180)
test_df["pickup_long_75"] = test_df["pickup_longitude"] * np.cos(
    75 * np.pi / 180
) - test_df["pickup_latitude"] * np.sin(75 * np.pi / 180)

test_df["pickup_lat_15"] = test_df["pickup_longitude"] * np.sin(
    15 * np.pi / 180
) + test_df["pickup_latitude"] * np.cos(15 * np.pi / 180)
test_df["pickup_lat_30"] = test_df["pickup_longitude"] * np.sin(
    30 * np.pi / 180
) + test_df["pickup_latitude"] * np.cos(30 * np.pi / 180)
test_df["pickup_lat_45"] = test_df["pickup_longitude"] * np.sin(
    45 * np.pi / 180
) + test_df["pickup_latitude"] * np.cos(45 * np.pi / 180)
test_df["pickup_lat_60"] = test_df["pickup_longitude"] * np.sin(
    60 * np.pi / 180
) + test_df["pickup_latitude"] * np.cos(60 * np.pi / 180)
test_df["pickup_lat_75"] = test_df["pickup_longitude"] * np.sin(
    75 * np.pi / 180
) + test_df["pickup_latitude"] * np.cos(75 * np.pi / 180)

test_df["dropoff_long_15"] = test_df["dropoff_longitude"] * np.cos(
    15 * np.pi / 180
) - test_df["dropoff_latitude"] * np.sin(15 * np.pi / 180)
test_df["dropoff_long_30"] = test_df["dropoff_longitude"] * np.cos(
    30 * np.pi / 180
) - test_df["dropoff_latitude"] * np.sin(30 * np.pi / 180)
test_df["dropoff_long_45"] = test_df["dropoff_longitude"] * np.cos(
    45 * np.pi / 180
) - test_df["dropoff_latitude"] * np.sin(45 * np.pi / 180)
test_df["dropoff_long_60"] = test_df["dropoff_longitude"] * np.cos(
    60 * np.pi / 180
) - test_df["dropoff_latitude"] * np.sin(60 * np.pi / 180)
test_df["dropoff_long_75"] = test_df["dropoff_longitude"] * np.cos(
    75 * np.pi / 180
) - test_df["dropoff_latitude"] * np.sin(75 * np.pi / 180)

test_df["dropoff_lat_15"] = test_df["dropoff_longitude"] * np.sin(
    15 * np.pi / 180
) + test_df["dropoff_latitude"] * np.cos(15 * np.pi / 180)
test_df["dropoff_lat_30"] = test_df["dropoff_longitude"] * np.sin(
    30 * np.pi / 180
) + test_df["dropoff_latitude"] * np.cos(30 * np.pi / 180)
test_df["dropoff_lat_45"] = test_df["dropoff_longitude"] * np.sin(
    45 * np.pi / 180
) + test_df["dropoff_latitude"] * np.cos(45 * np.pi / 180)
test_df["dropoff_lat_60"] = test_df["dropoff_longitude"] * np.sin(
    60 * np.pi / 180
) + test_df["dropoff_latitude"] * np.cos(60 * np.pi / 180)
test_df["dropoff_lat_75"] = test_df["dropoff_longitude"] * np.sin(
    75 * np.pi / 180
) + test_df["dropoff_latitude"] * np.cos(75 * np.pi / 180)

test_key = test_df["key"]
x_pred = test_df.drop(columns=["key", "pickup_datetime"])

dtest = xgb.DMatrix(x_pred, enable_categorical=True)

best_iter = getattr(model, "best_iteration", None)
if best_iter is None:
    prediction = model.predict(dtest)
else:
    prediction = model.predict(dtest, iteration_range=(0, best_iter + 1))

prediction = np.maximum(prediction, 0.0)
prediction = np.where(test_mask.values, prediction, fallback_fare)




## === cell 13
submission = pd.DataFrame({"key": test_key, "fare_amount": prediction.astype(float)})

submission_path = "taxi_fare_submission.csv"
submission.to_csv(submission_path, index=False)
print("Wrote submission:", submission_path, "shape:", submission.shape)
submission.head()
