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

4.44543

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 8.4997) has done: 'I fix the XGBoost inference crash by removing the deprecated `ntree_limit=model.best_ntree_limit` usage (XGBoost 2.x no longer exposes that attribute on the Booster returned by `xgb.train`) and instead use `iteration_range` when early stopping is enabled. I also make the data paths robust to your environment (preferring `/kaggle/input/...` and falling back to `../input/...`) so the notebook runs end-to-end on Kaggle. Finally, I ensure the submission is always created with exactly the required columns (`key`, `fare_amount`) and a `.csv` filename. These changes are score-neutral to slightly positive because they preserve the same training logic and simply make inference consistent with the best iteration chosen by early stopping.'
- What this solution (achieved 4.52686) has done: 'Your current RMSE (8.4997) is far worse than the target (3.47309), so we should make small, legitimate changes that improve generalization without changing the overall “XGBoost on engineered geo+time features” approach. The biggest issue is that the model is effectively undertrained: `early_stopping_rounds=100` with `num_boost_round=100` stops after just 1–2 trees and also uses the deprecated `reg:linear` objective. I keep the same training loop and feature set, but (1) switch to the correct modern objective name, (2) increase `num_boost_round` so early stopping can actually work, and (3) add the same basic coordinate/passenger cleaning to test as train plus a non-negative clip on predictions (fares can’t be negative), which usually improves RMSE without changing semantics. These are minimal, metric-aligned fixes that should move the score substantially toward the target band while still running within the time limit.'
- What this solution (achieved 4.3976) has done: 'Your current score (4.52686) is still worse than the target (3.47309), so we should make a small, metric-aligned improvement without changing the overall “XGBoost on engineered geo+time features” approach. The biggest remaining issue is that the datetime-derived features are treated as continuous numbers even though they’re categorical (hour/weekday/month/day), which commonly hurts XGBoost splits for this competition; enabling native categorical handling and casting those columns to `category` is a minimal change that typically improves RMSE. I also add a tiny bit of stable regularization (`min_child_weight`, `subsample`, `colsample_bytree`) while keeping the same training loop, features, and loss to reduce overfitting and improve generalization. Finally, I ensure train/test feature dtypes match exactly so inference uses the same representation as training.'
- What this solution (achieved 4.56836) has done: 'To move RMSE down from 4.3976 toward the 3.47309 target without changing your overall “XGBoost on engineered geo+time features” approach, I make two minimal, metric-aligned fixes: (1) remove unnecessary rounding to 2 decimals in the submission (rounding almost always worsens RMSE), and (2) ensure the train/validation split is not random but time-based (using `pickup_datetime`) to prevent temporal leakage that can hurt public/private generalization on this competition. Everything else (features, XGBoost training loop, objective, early stopping, categorical handling) stays the same. The code still run end-to-end within constraints and always write a valid `taxi_fare_submission.csv` with `key,fare_amount`.'
- What this solution (achieved 4.47713) has done: 'Your RMSE (4.56836) is still well above the target (3.47309), so we should make a small improvement that keeps the same “XGBoost on engineered geo+time features” core logic but reduces avoidable train/test mismatch. The biggest issue is the current time-based split: sorting by month/day/hour mixes multiple years and ignores minute/second, creating a noisy, non-chronological split that can hurt early stopping selection and generalization. I change the split to be strictly chronological using `pickup_datetime` before it gets dropped (no change to features, model, loss, or training loop), and keep the rest identical. This is a minimal, metric-aligned fix that typically lowers RMSE without altering the overall approach and still produces the same required `key,fare_amount` submission CSV.'
- What this solution (achieved 4.39233) has done: 'To move RMSE down from 4.47713 toward the 3.47309 target (lower is better), I make two minimal, metric-aligned changes without altering the overall “XGBoost on engineered geo+time features” approach. First, I fix a subtle but important train/test mismatch: `test_mask` is currently computed *before* datetime parsing/feature engineering and can be misaligned if any operations change row order; I compute the mask after loading and keep it aligned to `test_df`’s index. Second, I slightly tighten train data cleaning by capping extreme `fare_amount` outliers (common noise in this dataset) which typically improves RMSE without changing the model architecture/training loop. Everything else (features, categorical handling, chronological split, XGBoost training with early stopping, and submission schema) stays the same and the script still writes a valid `taxi_fare_submission.csv`.'
- What this solution (achieved 4.40579) has done: 'Your current RMSE (4.39233) is still worse than the target (3.47309), so we should make a small, metric-aligned improvement without changing the overall “XGBoost on engineered geo+time features” approach. The biggest remaining weakness is that the model is forced to fit raw latitude/longitude with many redundant rotated copies but no strong inductive signal about NYC geography; adding a standard NYC-taxi baseline feature set (absolute deltas + Manhattan distance approximation + simple airport flags) is a minimal feature-engineering extension that typically lowers RMSE while keeping the same model/training loop/loss. I also make train/test cleaning consistent by applying the same coordinate/passenger filters to test **before** feature creation (rather than only masking after), which reduces distribution mismatch and makes derived features (distance, deltas) better behaved. Finally, I keep the same submission schema and prediction clipping/masking so evaluation semantics remain unchanged.'
- What this solution (achieved 4.40579) has done: 'Your current RMSE (4.40579) is still meaningfully above the target (3.47309), so we should make a small improvement that keeps the same XGBoost approach and feature set but fixes a key train/test mismatch. Right now you train on a cleaned subset (coordinates/passengers filtered) but you *don’t* apply the same filtering before computing test features; this can create extreme/invalid derived features (distance/airport flags/manhattan) that then get predicted (even though you later fallback), which can still hurt due to dtype/feature-stat mismatch. I apply the same coordinate/passenger filtering to test *before* feature engineering (but without dropping rows—just using it to build a “clean copy” for feature creation and then writing predictions back), and I align categorical dtypes between train and test to avoid category-code inconsistencies. These are minimal, metric-aligned changes that typically reduce RMSE without changing the model/training loop/loss.'
- What this solution (achieved 4.6339) has done: 'To move RMSE down from 4.40579 toward the 3.47309 target (lower is better) with minimal changes, I fix a key evaluation mismatch: training/validation early stopping is currently done on a later time slice, but the final model is not re-fit on all available data up to the best number of trees, so you’re leaving signal on the table. I keep the same features, XGBoost training API, objective, and early-stopping setup, but add a second “final fit” step that trains on the full cleaned dataset using the best_iteration found on the chronological split. I also add a small, standard NYC baseline correction by using `log1p(fare_amount)` as the model target and `expm1` at inference (same model class/loop; just a target transform), which typically reduces RMSE for this competition without changing the overall approach. Submission writing, masking/fallback behavior, and all existing feature engineering remain intact.'
- What this solution (achieved 4.28683) has done: 'Your current RMSE (4.6339) is still worse than the target (3.47309), so the smallest likely win is to remove the target log-transform, which can hurt RMSE on the original fare scale when the model/feature set is already reasonable. I keep the same XGBoost training loop, early stopping, final refit, and all feature engineering unchanged, but train directly on `fare_amount` and predict directly (still clipping to non-negative). I also make the fallback prediction consistent (use the training mean fare, not expm1 of a log-mean), which reduces error on masked/invalid test rows. These minimal, metric-aligned changes should move the score toward the target without changing the overall approach.'
- What this solution (achieved 4.44543) has done: 'Your current RMSE (4.28683) is still materially worse than the target (3.47309), so we should make a small, metric-aligned improvement without changing the overall “XGBoost on engineered geo+time features” approach. The biggest avoidable error source is that the model predicts negative fares and unrealistically high fares for some cases, and RMSE penalizes these heavily; clipping predictions to a reasonable upper bound learned from the training distribution is a minimal post-processing step that usually reduces RMSE. I also make the fallback for invalid/masked test rows slightly more robust by using the cleaned-train mean (the same population the model learned from) rather than the pre-split mean, which reduces error on those rows without changing training. Everything else (data sampling size, cleaning, feature engineering, chronological split, early stopping, final refit, submission format) stays the same.'

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
        & (df.fare_amount < 250)  # tightened upper cap to reduce noisy extremes
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


def add_geo_features(df):
    df["abs_lon_diff"] = np.abs(df["pickup_longitude"] - df["dropoff_longitude"])
    df["abs_lat_diff"] = np.abs(df["pickup_latitude"] - df["dropoff_latitude"])
    df["manhattan"] = df["abs_lat_diff"] + df["abs_lon_diff"] * np.cos(
        np.radians((df["pickup_latitude"] + df["dropoff_latitude"]) / 2.0)
    )

    jfk_lon, jfk_lat = -73.7781, 40.6413
    ewr_lon, ewr_lat = -74.1745, 40.6895
    lga_lon, lga_lat = -73.8740, 40.7769

    df["pickup_jfk"] = (
        sphere_dist(df["pickup_latitude"], df["pickup_longitude"], jfk_lat, jfk_lon)
        < 2.0
    ).astype(np.int8)
    df["dropoff_jfk"] = (
        sphere_dist(df["dropoff_latitude"], df["dropoff_longitude"], jfk_lat, jfk_lon)
        < 2.0
    ).astype(np.int8)

    df["pickup_ewr"] = (
        sphere_dist(df["pickup_latitude"], df["pickup_longitude"], ewr_lat, ewr_lon)
        < 2.0
    ).astype(np.int8)
    df["dropoff_ewr"] = (
        sphere_dist(df["dropoff_latitude"], df["dropoff_longitude"], ewr_lat, ewr_lon)
        < 2.0
    ).astype(np.int8)

    df["pickup_lga"] = (
        sphere_dist(df["pickup_latitude"], df["pickup_longitude"], lga_lat, lga_lon)
        < 2.0
    ).astype(np.int8)
    df["dropoff_lga"] = (
        sphere_dist(df["dropoff_latitude"], df["dropoff_longitude"], lga_lat, lga_lon)
        < 2.0
    ).astype(np.int8)
    return df


train_df["distance"] = sphere_dist(
    train_df["pickup_latitude"],
    train_df["pickup_longitude"],
    train_df["dropoff_latitude"],
    train_df["dropoff_longitude"],
)

train_df = add_geo_features(train_df)
train_df = add_datetime_info(train_df)

for c in ["hour", "day", "month", "weekday"]:
    train_df[c] = train_df[c].astype("category")

train_df.head()



## === cell 8
train_df["_pickup_datetime_ts"] = train_df["pickup_datetime"].astype("int64")
train_df = (
    train_df.sort_values("_pickup_datetime_ts")
    .drop(columns=["_pickup_datetime_ts"])
    .reset_index(drop=True)
)

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
y = train_df["fare_amount"].astype(np.float32)
train = train_df.drop(columns=["fare_amount"])

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

best_iter = getattr(model, "best_iteration", None)
if best_iter is None:
    best_iter = 499  # fallback to full rounds if early stopping didn't set it

dall = xgb.DMatrix(train, label=y, enable_categorical=True)
final_model = xgb.train(
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
    dtrain=dall,
    num_boost_round=int(best_iter) + 1,
    verbose_eval=False,
)



## === cell 12
test_df = pd.read_csv(TEST_PATH)

fallback_fare = float(y.mean())

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

test_feat = test_df.loc[test_mask].copy()

test_feat["distance"] = sphere_dist(
    test_feat["pickup_latitude"],
    test_feat["pickup_longitude"],
    test_feat["dropoff_latitude"],
    test_feat["dropoff_longitude"],
)
test_feat = add_geo_features(test_feat)
test_feat = add_datetime_info(test_feat)

for c in ["hour", "day", "month", "weekday"]:
    test_feat[c] = test_feat[c].astype("category")

for c in ["hour", "day", "month", "weekday"]:
    test_feat[c] = test_feat[c].cat.set_categories(train_df[c].cat.categories)

test_feat["pickup_long_15"] = test_feat["pickup_longitude"] * np.cos(
    15 * np.pi / 180
) - test_feat["pickup_latitude"] * np.sin(15 * np.pi / 180)
test_feat["pickup_long_30"] = test_feat["pickup_longitude"] * np.cos(
    30 * np.pi / 180
) - test_feat["pickup_latitude"] * np.sin(30 * np.pi / 180)
test_feat["pickup_long_45"] = test_feat["pickup_longitude"] * np.cos(
    45 * np.pi / 180
) - test_feat["pickup_latitude"] * np.sin(45 * np.pi / 180)
test_feat["pickup_long_60"] = test_feat["pickup_longitude"] * np.cos(
    60 * np.pi / 180
) - test_feat["pickup_latitude"] * np.sin(60 * np.pi / 180)
test_feat["pickup_long_75"] = test_feat["pickup_longitude"] * np.cos(
    75 * np.pi / 180
) - test_feat["pickup_latitude"] * np.sin(75 * np.pi / 180)

test_feat["pickup_lat_15"] = test_feat["pickup_longitude"] * np.sin(
    15 * np.pi / 180
) + test_feat["pickup_latitude"] * np.cos(15 * np.pi / 180)
test_feat["pickup_lat_30"] = test_feat["pickup_longitude"] * np.sin(
    30 * np.pi / 180
) + test_feat["pickup_latitude"] * np.cos(30 * np.pi / 180)
test_feat["pickup_lat_45"] = test_feat["pickup_longitude"] * np.sin(
    45 * np.pi / 180
) + test_feat["pickup_latitude"] * np.cos(45 * np.pi / 180)
test_feat["pickup_lat_60"] = test_feat["pickup_longitude"] * np.sin(
    60 * np.pi / 180
) + test_feat["pickup_latitude"] * np.cos(60 * np.pi / 180)
test_feat["pickup_lat_75"] = test_feat["pickup_longitude"] * np.sin(
    75 * np.pi / 180
) + test_feat["pickup_latitude"] * np.cos(75 * np.pi / 180)

test_feat["dropoff_long_15"] = test_feat["dropoff_longitude"] * np.cos(
    15 * np.pi / 180
) - test_feat["dropoff_latitude"] * np.sin(15 * np.pi / 180)
test_feat["dropoff_long_30"] = test_feat["dropoff_longitude"] * np.cos(
    30 * np.pi / 180
) - test_feat["dropoff_latitude"] * np.sin(30 * np.pi / 180)
test_feat["dropoff_long_45"] = test_feat["dropoff_longitude"] * np.cos(
    45 * np.pi / 180
) - test_feat["dropoff_latitude"] * np.sin(45 * np.pi / 180)
test_feat["dropoff_long_60"] = test_feat["dropoff_longitude"] * np.cos(
    60 * np.pi / 180
) - test_feat["dropoff_latitude"] * np.sin(60 * np.pi / 180)
test_feat["dropoff_long_75"] = test_feat["dropoff_longitude"] * np.cos(
    75 * np.pi / 180
) - test_feat["dropoff_latitude"] * np.sin(75 * np.pi / 180)

test_feat["dropoff_lat_15"] = test_feat["dropoff_longitude"] * np.sin(
    15 * np.pi / 180
) + test_feat["dropoff_latitude"] * np.cos(15 * np.pi / 180)
test_feat["dropoff_lat_30"] = test_feat["dropoff_longitude"] * np.sin(
    30 * np.pi / 180
) + test_feat["dropoff_latitude"] * np.cos(30 * np.pi / 180)
test_feat["dropoff_lat_45"] = test_feat["dropoff_longitude"] * np.sin(
    45 * np.pi / 180
) + test_feat["dropoff_latitude"] * np.cos(45 * np.pi / 180)
test_feat["dropoff_lat_60"] = test_feat["dropoff_longitude"] * np.sin(
    60 * np.pi / 180
) + test_feat["dropoff_latitude"] * np.cos(60 * np.pi / 180)
test_feat["dropoff_lat_75"] = test_feat["dropoff_longitude"] * np.sin(
    75 * np.pi / 180
) + test_feat["dropoff_latitude"] * np.cos(75 * np.pi / 180)

test_key = test_df["key"]

prediction = np.full(shape=(len(test_df),), fill_value=fallback_fare, dtype=np.float32)

x_pred = test_feat.drop(columns=["key", "pickup_datetime"])
dtest = xgb.DMatrix(x_pred, enable_categorical=True)

pred_valid = final_model.predict(dtest).astype(np.float32)

pred_valid = np.clip(pred_valid, 0.0, float(y.quantile(0.999)))

prediction[test_mask.values] = pred_valid



## === cell 13
submission = pd.DataFrame({"key": test_key, "fare_amount": prediction.astype(float)})

submission_path = "taxi_fare_submission.csv"
submission.to_csv(submission_path, index=False)
print("Wrote submission:", submission_path, "shape:", submission.shape)
submission.head()
