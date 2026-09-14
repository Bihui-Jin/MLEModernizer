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

5.28917

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 8.4997) has done: 'Diagnosis: The crash occurs in cell 12 because with xgboost==2.0.3 the trained `Booster` object no longer exposes `best_ntree_limit`, so accessing `model.best_ntree_limit` raises `AttributeError`. This code path is only used to limit prediction to the best iteration found by early stopping. In newer XGBoost versions, the equivalent is `best_iteration` (0-based), and predictions should be limited via `iteration_range`.  
Patch summary: Update the prediction call in cell 12 to use `iteration_range=(0, best_iteration+1)` when `best_iteration` exists; otherwise fall back to the default full-model prediction. This preserves the original “use best iteration” semantics while remaining compatible with xgboost 2.x and keeps `prediction` as a NumPy array for cell 13.  
Updated cells: Only cell 12 is modified.  
Compatibility notes for cell k+1: `test_key` and `prediction` are still created with the same meanings and types expected by cell 13, so submission generation remains unchanged.  
Assumptions: The model was trained with early stopping and thus exposes `best_iteration` in xgboost 2.0.3; if not, the fallback path predicts using all trees.'
- What this solution (achieved 8.49966) has done: 'Your RMSE (8.4997) is far worse than the target (3.47309), so we should improve model fit without changing the overall approach. The biggest issue is that the model is trained with only 100 boosting rounds while also setting `early_stopping_rounds=100`, which effectively prevents early stopping from being useful and leads to a very underfit model. I keep the same XGBoost training API and features, but increase `num_boost_round` to let early stopping actually select a better iteration and update the deprecated objective to `reg:squarederror` (same semantics for RMSE regression, but correct for xgboost 2.x). I also ensure predictions are non-negative (fares can’t be negative), which typically reduces RMSE on this competition without changing core logic.'
- What this solution (achieved 9.52219) has done: 'To move RMSE down toward your target with minimal disruption, I’m keeping the same XGBoost training API, feature engineering, and loss/metric, but fixing a key generalization issue: your split is random, which leaks time patterns and typically hurts leaderboard performance on this dataset. I switch to a deterministic time-based split (train on earlier rides, validate on later rides) while keeping early stopping and the same feature set, which usually improves public RMSE without changing core semantics. I also ensure train/test feature columns align exactly (same order and any missing columns handled), preventing subtle train–test mismatch that can degrade predictions. The submission writing remains unchanged and still produces `taxi_fare_submission.csv`.'
- What this solution (achieved 6.91085) has done: 'Your RMSE (9.52) is much worse than the target (3.47), so the most likely cause is training on a non-time-sorted slice due to reading only the first 1M rows: your current “time-based split” is actually arbitrary and can badly hurt generalization. I make a minimal fix by parsing `pickup_datetime`, sorting the sampled training data by time, and then doing the same 80/20 time split so it becomes a real forward-chaining validation. I also remove the placeholder datetime code (it currently just creates NaNs and doesn’t help) and keep everything else (features, XGBoost training API, early stopping, submission format) unchanged. This should move RMSE down toward your target while preserving the core logic.'
- What this solution (achieved 5.62031) has done: 'Your current RMSE (6.91) is still far above the target (3.47), so we should improve generalization with the smallest changes that don’t alter your model/feature logic. The biggest remaining issue is that you only train on the first 1M rows, which are not representative and are biased toward early timestamps; a simple, legitimate improvement is to sample from the full training file while keeping the same row budget for speed. I switch the `read_csv` call to use `skiprows` with a fixed random seed to get a more representative 1M-row sample, and I reuse the already-parsed `pickup_datetime` for sorting (avoiding a second inconsistent read). Everything else (features, XGBoost training, early stopping, prediction, and submission format) remains the same.'
- What this solution (achieved 5.88835) has done: 'Your gap to target is large (5.62 vs 3.47, lower is better), so we need a legitimate generalization improvement while keeping the same feature engineering and XGBoost training approach. The biggest current score drag is that training uses only a 1M-row sample and then validates with a time split, but the final model is still trained only on 80% of that data (it never refits on all sampled data), which wastes signal and usually hurts leaderboard RMSE. I keep the exact same model params, early stopping semantics, and features, but refit a final model on the full sampled dataset using the best number of boosting rounds found during validation, then use that final model for test prediction. This is a minimal change that typically improves RMSE without changing your core logic.'
- What this solution (achieved 5.63676) has done: 'Diagnosis: Cell 10 reads `dt_df` with only the `pickup_datetime` column, then calls `clean_df(dt_df)`. The `clean_df` function filters on multiple columns including `fare_amount`, `pickup_longitude`, etc., so calling it on a single-column DataFrame raises `AttributeError: 'DataFrame' object has no attribute 'fare_amount'`. The intent of cell 10 is only to obtain `pickup_datetime` aligned with the already-cleaned `train_df`, so re-cleaning a datetime-only DataFrame is both invalid and unnecessary.

Patch summary: In cell 10, remove the call to `clean_df(dt_df)` and instead build `pickup_dt` directly from `train_df["pickup_datetime"]` (which still exists at this point). This keeps the subsequent masking/sorting logic unchanged while ensuring `pickup_dt` is aligned with `train_df` rows and prevents the missing-column crash.

Updated cells: Only cell 10 is modified as below.

Compatibility notes for cell k+1: The variables produced by cell 10 (`x_train`, `x_test`, `y_train`, `y_test`, `train`, `y`) retain the same meaning and compatible types/shapes for cell 11’s `XGBmodel` and final training. `pickup_dt` remains a `datetime64` Series used only within cell 10 for sorting, so downstream behavior is preserved.

Assumptions: `train_df` at the start of cell 10 still contains the original `pickup_datetime` column (it is dropped only in cell 8 from `train_df`, but cell 10 uses `train_df` as already transformed—however in the provided flow, `train_df.drop(columns=["key","pickup_datetime"], inplace=True)` happens in cell 8, so we must instead use the original `dt_df` but without `clean_df`; we assume `skip` yields identical row selection for `dt_df` and the initial `train_df` load before cleaning, and we rely on the existing `mask` logic to align rows after numeric coercion and null filtering).'
- What this solution (achieved 5.58537) has done: 'Your current RMSE (5.63676, lower is better) is still far above the target (3.47309), so the smallest likely win is to stop misaligning `pickup_datetime` with the cleaned/sampled `train_df`. Right now, cell 10 rereads only datetimes and then tries to “realign” via a mask built on already-cleaned numeric features, which can silently scramble time ordering and harm early-stopping selection and the final refit. I fix this by carrying `pickup_datetime` through feature engineering as a separate Series before dropping it, then using that aligned Series for the time sort/split (no second read of train.csv). This preserves your model, features, and training loop, but makes the time-based split genuinely consistent with the rows used for training.'
- What this solution (achieved 5.28917) has done: 'Your current RMSE (5.585) is still far above the target (3.473, lower is better), so we should make the smallest legitimate generalization improvement without changing your model/feature core. The biggest remaining issue is that you sample 1M rows uniformly, but you do not ensure the sampled/cleaned set reflects the *test* distribution well; a very small, common win on this competition is to clip extreme fare outliers (still legitimate cleaning) which otherwise dominate RMSE and degrade fit. I add a conservative upper bound filter on `fare_amount` in `clean_df` (e.g., <= 250) and keep everything else (features, split, XGBoost params/training/refit, submission format) identical. This typically reduces RMSE noticeably while staying within your current approach and runtime constraints.'

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
RANDOM_SEED = 42
NROWS = 1_000_000

train_path = "../input/train.csv"
n_total = sum(1 for _ in open(train_path)) - 1  # exclude header
rng = np.random.default_rng(RANDOM_SEED)

if NROWS >= n_total:
    skip = None
else:
    n_skip = n_total - NROWS
    skip_idx = rng.choice(
        np.arange(1, n_total + 1), size=n_skip, replace=False
    )  # 1..n_total
    skip = set(skip_idx.tolist())

train_df = pd.read_csv(train_path, skiprows=skip)
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

train_df.head()



## === cell 8
pickup_dt_aligned = train_df["pickup_datetime"].copy()

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
pickup_dt = pd.to_datetime(pickup_dt_aligned, errors="coerce").reset_index(drop=True)

y = train_df["fare_amount"]
train = train_df.drop(columns=["fare_amount"])

train = train.apply(pd.to_numeric, errors="coerce")
y = pd.to_numeric(y, errors="coerce")
mask = train.notnull().all(axis=1) & y.notnull() & pickup_dt.notnull()

train = train.loc[mask].reset_index(drop=True)
y = y.loc[mask].reset_index(drop=True)
pickup_dt = pickup_dt.loc[mask].reset_index(drop=True)

tmp = train.copy()
tmp["__y__"] = y.values
tmp["__pickup_datetime__"] = pickup_dt.values
tmp = tmp.sort_values("__pickup_datetime__").drop(columns=["__pickup_datetime__"])

y = tmp["__y__"]
train = tmp.drop(columns=["__y__"])

n = len(train)
split_idx = int(n * 0.8)

x_train = train.iloc[:split_idx].copy()
y_train = y.iloc[:split_idx].copy()
x_test = train.iloc[split_idx:].copy()
y_test = y.iloc[split_idx:].copy()




## === cell 11
def XGBmodel(x_train, x_test, y_train, y_test):
    matrix_train = xgb.DMatrix(x_train, label=y_train)
    matrix_test = xgb.DMatrix(x_test, label=y_test)

    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "eta": 0.05,
        "max_depth": 8,
        "min_child_weight": 1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "lambda": 1.0,
        "alpha": 0.0,
        "seed": RANDOM_SEED,
        "nthread": max(1, os.cpu_count() or 1),
    }

    model = xgb.train(
        params=params,
        dtrain=matrix_train,
        num_boost_round=5000,
        early_stopping_rounds=100,
        evals=[(matrix_test, "test")],
        verbose_eval=False,
    )
    return model, params


model, params = XGBmodel(x_train, x_test, y_train, y_test)

best_iter = getattr(model, "best_iteration", None)
if best_iter is None:
    final_num_boost_round = 5000
else:
    final_num_boost_round = int(best_iter) + 1  # best_iteration is 0-based

dall = xgb.DMatrix(train, label=y)
final_model = xgb.train(
    params=params,
    dtrain=dall,
    num_boost_round=final_num_boost_round,
    verbose_eval=False,
)



## === cell 12
test_df = pd.read_csv("../input/test.csv")
test_df["distance"] = sphere_dist(
    test_df["pickup_latitude"],
    test_df["pickup_longitude"],
    test_df["dropoff_latitude"],
    test_df["dropoff_longitude"],
)
test_df = add_datetime_info(test_df)
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

x_pred = x_pred.apply(pd.to_numeric, errors="coerce").fillna(0.0)

train_cols = list(x_train.columns)
for c in train_cols:
    if c not in x_pred.columns:
        x_pred[c] = 0.0
x_pred = x_pred[train_cols]

dtest = xgb.DMatrix(x_pred)

prediction = final_model.predict(dtest)

prediction = np.maximum(prediction, 0.0)

submission = pd.DataFrame({"key": test_key, "fare_amount": prediction.round(2)})

submission.to_csv("taxi_fare_submission.csv", index=False)
submission
