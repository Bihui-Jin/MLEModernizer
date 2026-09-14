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

3.44303

# 6. Current score

5.44416

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 6.29803) has done: 'The crash happens because with xgboost==2.0.3 the `Booster` returned by `xgb.train()` may not expose `best_ntree_limit`, so accessing `model.best_ntree_limit` raises `AttributeError`. The minimal fix is to avoid relying on that attribute in prediction and instead use the best iteration if it exists. We keep the same model training logic and only adjust prediction to be compatible across XGBoost versions by conditionally passing `iteration_range`. This preserves the “use best iteration from early stopping” behavior when available, and otherwise falls back to default prediction.'
- What this solution (achieved 5.71685) has done: 'You’re currently far worse than the target RMSE (6.298 vs 3.443, lower is better), so we make small changes that legitimately improve generalization without changing the overall approach (still: same features + XGBoost via `xgb.train`). The biggest issue is that your XGBoost config is using the deprecated objective name (`reg:linear`) and is missing standard regularization / subsampling settings that typically reduce RMSE for this competition with no architectural change. I also stop rounding predictions (rounding hurts RMSE), clip to a reasonable minimum fare (prevents negative outputs), and add very light additional cleaning that removes extreme-distance outliers that can destabilize training. All paths remain the same and it still write a valid `taxi_fare_submission.csv`.'
- What this solution (achieved 6.35615) has done: 'Your current RMSE (5.71685) is far above the target (3.44303), so we should improve generalization with minimal changes while keeping the same core pipeline (same features, same `xgb.train`, same loss/metric). The biggest likely issue is the validation split: a random split leaks near-duplicate rides across train/valid in this competition and gives a misleading early-stopping point; switching to a simple time-based split better matches the test distribution and typically reduces Kaggle RMSE without changing the model. I also tighten data cleaning slightly (fare upper bound and NYC coordinate bounds) and add two very lightweight, standard engineered features derived from existing columns (abs lat/lon diffs) that don’t change the approach but help XGBoost fit distance/fare relationships more accurately. Submission writing remains identical and produces `taxi_fare_submission.csv` with `key,fare_amount`.'
- What this solution (achieved 5.23162) has done: 'Your current RMSE (6.35615) is far above the target (3.44303), so we need legitimate improvements while keeping the same overall pipeline (same feature set concept + XGBoost regression + early stopping). The biggest gap driver here is that the model is trained on raw `fare_amount` with squared error, which overweights large fares/outliers; switching to a log1p target (and expm1 back at inference) is a minimal, standard transformation that keeps the same model/approach but typically yields a large RMSE gain on this competition. I also add a tiny, safe missing-value guard in feature building (drop any rows where datetime parsing failed after cleaning) to prevent silent training noise. Everything else (data paths, core training loop with `xgb.train`, early stopping, and submission schema) stays the same, and it still writes `taxi_fare_submission.csv` with `key,fare_amount`.'
- What this solution (achieved 5.44416) has done: 'We’re still far above the target RMSE, so the smallest legitimate way to move closer is to fix two high-impact, low-intrusion issues: (1) train on a more representative sample of the huge dataset by using a simple uniform random skiprows sampler instead of the first 1M rows (which are time-biased), and (2) add one standard, minimal engineered feature (Haversine distance squared) that helps XGBoost model the fare–distance relationship without changing the approach. I keep your XGBoost training loop, log1p target, early stopping, and existing feature set intact, and I won’t add approximations or reduce convergence. The submission format and output filename remain the same.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import datetime as dt
from sklearn.model_selection import train_test_split
import xgboost as xgb
import os

print(os.listdir("../input"))



## === cell 1
TRAIN_PATH = "../input/train.csv"
N_SAMPLED = 1_000_000

rng = np.random.default_rng(42)
N_TOTAL = 55_423_856  # rows excluding header (approx from provided metadata)
p_keep = min(1.0, N_SAMPLED / float(N_TOTAL))
keep_mask = rng.random(N_TOTAL) < p_keep
keep_idx = np.flatnonzero(keep_mask) + 1  # shift by 1 because row 0 is header
if keep_idx.size > N_SAMPLED:
    keep_idx = rng.choice(keep_idx, size=N_SAMPLED, replace=False)
keep_set = set(keep_idx.tolist())


def _skiprows(i):
    if i == 0:
        return False
    return i not in keep_set


train_df = pd.read_csv(
    TRAIN_PATH,
    skiprows=_skiprows,
    parse_dates=["pickup_datetime"],
)
train_df.dtypes



## === cell 2
print(train_df.isnull().sum())



## === cell 3
train_df = train_df.dropna(how="any", axis="rows")



## === cell 4
train_df.head()



## === cell 5
train_df.iloc[:1000].plot.scatter("pickup_longitude", "pickup_latitude")
train_df.iloc[:1000].plot.scatter("dropoff_longitude", "dropoff_latitude")

train_df.describe()




## === cell 6
def clean_df(df):
    return df[
        (df.fare_amount > 0)
        & (df.fare_amount <= 250)
        & (df.pickup_longitude > -75.5)
        & (df.pickup_longitude < -72.5)
        & (df.pickup_latitude > 40.0)
        & (df.pickup_latitude < 41.8)
        & (df.dropoff_longitude > -75.5)
        & (df.dropoff_longitude < -72.5)
        & (df.dropoff_latitude > 40.0)
        & (df.dropoff_latitude < 41.8)
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
    dataset["pickup_datetime"] = pd.to_datetime(
        dataset["pickup_datetime"], errors="coerce"
    )

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

train_df["abs_lon_diff"] = (
    train_df["pickup_longitude"] - train_df["dropoff_longitude"]
).abs()
train_df["abs_lat_diff"] = (
    train_df["pickup_latitude"] - train_df["dropoff_latitude"]
).abs()

train_df["distance_sq"] = train_df["distance"] ** 2

train_df = train_df[(train_df["distance"] >= 0) & (train_df["distance"] <= 100)]

train_df = train_df.dropna(subset=["pickup_datetime"])

train_df.head()



## === cell 8
train_df.drop(columns=["key"], inplace=True)
train_df.head()



## === cell 9
y = train_df["fare_amount"]
X = train_df.drop(columns=["fare_amount"])

X = X.sort_values("pickup_datetime")
y = y.loc[X.index]

split_idx = int(len(X) * 0.8)
x_train = X.iloc[:split_idx].drop(columns=["pickup_datetime"])
y_train = y.iloc[:split_idx]
x_test = X.iloc[split_idx:].drop(columns=["pickup_datetime"])
y_test = y.iloc[split_idx:]




## === cell 10
def XGBmodel(x_train, x_test, y_train, y_test):
    y_train_log = np.log1p(y_train.values)
    y_test_log = np.log1p(y_test.values)

    matrix_train = xgb.DMatrix(x_train, label=y_train_log)
    matrix_test = xgb.DMatrix(x_test, label=y_test_log)

    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "eta": 0.05,
        "max_depth": 8,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 1.0,
        "lambda": 1.0,
        "alpha": 0.0,
        "seed": 0,
        "nthread": max(1, os.cpu_count() or 1),
    }

    model = xgb.train(
        params=params,
        dtrain=matrix_train,
        num_boost_round=600,
        early_stopping_rounds=30,
        evals=[(matrix_test, "valid")],
        verbose_eval=False,
    )
    return model


model = XGBmodel(x_train, x_test, y_train, y_test)



## === cell 11
test_df = pd.read_csv("../input/test.csv", parse_dates=["pickup_datetime"])

test_df["distance"] = sphere_dist(
    test_df["pickup_latitude"],
    test_df["pickup_longitude"],
    test_df["dropoff_latitude"],
    test_df["dropoff_longitude"],
)
test_df = add_datetime_info(test_df)

test_df["abs_lon_diff"] = (
    test_df["pickup_longitude"] - test_df["dropoff_longitude"]
).abs()
test_df["abs_lat_diff"] = (
    test_df["pickup_latitude"] - test_df["dropoff_latitude"]
).abs()

test_df["distance_sq"] = test_df["distance"] ** 2

test_key = test_df["key"]
x_pred = test_df.drop(columns=["key", "pickup_datetime"])

dtest = xgb.DMatrix(x_pred)

if hasattr(model, "best_iteration") and model.best_iteration is not None:
    pred_log = model.predict(dtest, iteration_range=(0, model.best_iteration + 1))
else:
    pred_log = model.predict(dtest)

prediction = np.expm1(pred_log)
prediction = np.clip(prediction, 0.0, None)

submission = pd.DataFrame({"key": test_key, "fare_amount": prediction})
submission.to_csv("taxi_fare_submission.csv", index=False)
submission
