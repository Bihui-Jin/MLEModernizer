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

3.93504

# 6. Current score

4.73673

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 4.77304) has done: 'I fix the runtime error in the RandomForest training by ensuring there are no NaNs/Infs in the engineered feature matrix (distance features can become NaN due to a radians-conversion bug). I keep your overall workflow and model choices intact, but correct the Haversine implementation so the `distance` and airport-distance features are computed properly, which should legitimately improve RMSE toward your target. I also make train/test preprocessing consistent by applying the same dtype conversions and a final safety `dropna`/finite filter right before splitting/training. Finally, I ensure the submission uses the required `key,fare_amount` format and is saved with a `.csv` suffix.'
- What this solution (achieved 5.00486) has done: 'Your current gap to the target is large (4.77304 vs 3.93504; lower is better), so we need a modest real improvement without changing the overall approach. The biggest low-risk win here is to train on a larger sample than 5,000 rows (still tiny vs 55M), because your feature engineering + XGB model can’t generalize well with so little data; increasing `nrows` keeps the same model/loop but typically improves RMSE substantially. I also fix two small data-quality issues that hurt both training and test consistency: (1) the passenger_count filter currently keeps zeros despite earlier checks, and (2) remove extreme-distance outliers in training (but never in test) to reduce noise. Finally, I keep the same submission format but add a simple non-negative clamp to predictions, which aligns with fare semantics and tends to reduce RMSE slightly.'
- What this solution (achieved 8.80286) has done: 'Most of the timeout comes from doing heavy EDA/plotting on ~1M rows (multiple seaborn plots + pivot/heatmap) and from training extra models (RandomForest + two XGBoost fits for tuning) that don’t contribute to the final submission. I keep the exact same data loading, cleaning, feature engineering, and the final XGBoost model/parameters used for submission, but remove (or gate behind a flag) all non-essential prints/plots and the intermediate model trainings so only the final model is trained once. I also avoid a couple of expensive full-data operations (like sorting 1M distances) and consolidate redundant inf/nan handling while preserving identical semantics. Finally, I set XGBoost threading explicitly to use all cores and keep randomness deterministic via the same random_state.'
- What this solution (achieved 8.47742) has done: 'Your current RMSE (8.80) is far worse than the target (3.94, lower is better), so we need a real but minimal improvement without changing the overall model/feature approach. The main issue is that the model is missing very strong signal from `passenger_count` and basic time features (`day`, `day_of_week`), even though you already compute them—adding them to `selected_predictors` preserves the same pipeline while typically reducing RMSE substantially. I also make the train/test imputation consistent by computing medians on the training features only (currently OK) and explicitly filling any remaining NaNs, and I ensure the submission `key` alignment remains correct even if some test rows are dropped due to bad datetimes. Everything else (cleaning rules, haversine distance, XGBoost params, training approach) is kept the same.'
- What this solution (achieved 4.793) has done: 'Your current RMSE (8.48) is much worse than the target (3.94, lower is better), so we need a real improvement but with minimal, core-logic-preserving changes. The biggest likely cause is training/test distribution mismatch: you apply strict geo/fare/passenger/distance cleaning only to train, but never remove obviously invalid rows from test, so the model can produce extreme errors on those rows. I add a tiny, semantics-preserving “sanity filter” on test only where we can confidently detect invalid coordinates and passenger_count (matching your existing NYC bounding-box logic), and for any removed test rows I still output a prediction by falling back to the model prediction based on median-imputed features (so the submission keeps all keys). I also enforce consistent numeric dtypes for the engineered time features and ensure the feature matrices are fully finite before fitting/predicting (no change to model/params).'
- What this solution (achieved 4.75119) has done: 'You’re currently worse than the target (4.793 vs 3.935, lower is better), so the smallest legitimate way to move RMSE down without changing the model or feature set is to improve training data quality/representativeness. I keep the exact same feature engineering and the same final XGBRegressor hyperparameters, but (1) train on a larger slice of the 55M rows to reduce variance and distribution mismatch, and (2) add a very standard NYC Taxi Fare cleaning step that removes “zero coordinate” rows (lat/long == 0) which otherwise inject extreme noise. I also ensure we only compute medians after all cleaning (as you already do) and keep submission alignment unchanged so you still write a valid `key,fare_amount` CSV.'
- What this solution (achieved 4.75119) has done: 'Your current RMSE (4.75119) is worse than the target (3.93504; lower is better), so we should make small, legitimate improvements that reduce error without changing the overall model/feature approach. The biggest low-risk gain here is removing a small set of highly noisy training outliers (unrealistically high fares given short/medium distances) using a simple, standard “fare-per-km” cap; this preserves your feature engineering and XGBRegressor setup but typically improves RMSE. I also add a missing “zero-coordinate” sanity check to the test validity mask (you already remove those from train), so test-time fallback-to-median triggers consistently for those pathological rows. Finally, I keep your submission format identical and ensure all features remain finite.'
- What this solution (achieved 4.75119) has done: 'To move RMSE down toward your 3.935 target (lower is better) while keeping your model/feature pipeline intact, I make two small, high-impact data-quality fixes: (1) apply the same NaN/NaT handling to the test set that you already do for train, and (2) compute airport-distance features correctly (your current code accidentally swaps lat/long for those features, injecting strong noise). I also ensure the engineered feature columns used for `test_valid_mask` are finite (so invalid rows don’t slip through) and keep your current “fallback to median fare for invalid test rows” behavior unchanged. These changes preserve the same architecture/training approach and should legitimately reduce error by removing systematic feature corruption and train/test inconsistency.'
- What this solution (achieved 4.75119) has done: 'We’re currently worse than the target (4.75119 vs 3.93504; lower is better), so the smallest legitimate move is to reduce train/test mismatch without changing your model, features, or training loop. I make train and test preprocessing symmetric by dropping test rows with invalid/NaT `pickup_datetime` for feature generation and then explicitly marking them invalid in `test_valid_mask` (so they use your existing median fallback), instead of letting NaNs propagate into time features. I also ensure the submission key alignment is guaranteed by building `submission` directly from the final `test_dataset["key"]` after any safe copying, and add a final `np.isfinite` check on the full test feature matrix before prediction so invalid rows can’t slip through the mask. These are minimal pipeline-consistency fixes that typically reduce RMSE a bit without altering the core modeling approach.'
- What this solution (achieved 4.73673) has done: 'You’re currently worse than the target (4.75119 vs 3.93504 RMSE; lower is better), so the smallest safe improvement is to (1) better align the training distribution with test by switching the train/test split to `shuffle=False` (time-ordered) since this dataset is time-dependent, and (2) stabilize prediction quality by using the **mean** (rather than median) as the fallback for invalid test rows, which is closer to the evaluation-optimal constant under RMSE. I also add a minimal, standard outlier filter that removes implausible “too-low fare for too-long distance” trips (these are label/noise issues that hurt RMSE) without changing your model, features, or training loop. These changes keep your feature engineering and final XGBRegressor setup identical and still write the same `key,fare_amount` submission CSV.'
- What this solution (achieved 4.73673) has done: 'You’re currently worse than the target (4.73673 vs 3.93504 RMSE; lower is better), so we make the smallest changes that typically reduce RMSE without changing your model, features, or training approach. The biggest low-risk win is to fix a subtle but important leakage/bug: you compute `test_feature_finite` from `test_dataset[selected_predictors]` **before** filling NaNs/Infs, so many rows are wrongly marked invalid and replaced by the fallback mean—this hurts RMSE. We recompute the test validity mask **after** the exact same finite/NaN handling used to build `X_test_all_df`, keeping your fallback logic but applying it only to truly invalid geo/datetime/pax rows. We also ensure the output keeps the original test row order and all keys, producing a valid `key,fare_amount` submission CSV.'
- What this solution (achieved 4.73673) has done: 'We’re currently worse than the target (4.73673 vs 3.93504 RMSE; lower is better), so we should make a small, legitimate improvement that reduces error without changing your model, features, or training loop. The biggest remaining low-risk issue is train/test preprocessing mismatch: you filter training rows to NYC bounds and nonzero coordinates, but you never apply the same sanity filtering to test before feature construction—then you replace many such rows with a global mean fallback, which can inflate RMSE. I add a *test-only* geospatial “sanitize then restore” step: for rows with invalid coords/pax/datetime we temporarily set feature columns to NaN so they get median-imputed (consistent with your pipeline) while still using your existing fallback behavior; valid rows remain unchanged. This preserves your core logic (same features, same XGB params, same fallback mechanism) but reduces pathological feature values that can drive bad predictions.'
- What this solution (achieved 4.73673) has done: 'To move RMSE down toward your 3.935 target (lower is better) with minimal disruption, I keep your exact feature set and XGBRegressor hyperparameters but fix two train/test consistency issues that typically inflate error. First, I build the test validity mask from the same *sanitized* view you use for feature construction, so rows with NaT datetimes or invalid geo/pax don’t accidentally pass parts of the mask and get inconsistent handling. Second, I apply the same conservative coordinate “between NYC bounds” filter to the airport-distance feature computation by ensuring invalid rows are set to NaN **before** deriving those features, preventing huge airport distances from leaking into otherwise-imputed rows. This preserves your fallback behavior and model training loop, but reduces systematic feature noise and should improve RMSE modestly without changing the core approach. The script still run end-to-end and write `submission_new.csv` in the required `key,fare_amount` format.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns  # for plot visualization

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from math import sqrt

import os



## === cell 1
sns.set_style("darkgrid")

RUN_EDA = False
RUN_INTERMEDIATE_MODELS = False

np.random.seed(42)



## === cell 2
TRAIN_NROWS = 3_000_000

test_dataset = pd.read_csv("../input/test.csv")
train_dataset = pd.read_csv("../input/train.csv", nrows=TRAIN_NROWS)



## === cell 3
if RUN_EDA:
    display(train_dataset.head(5))



## === cell 4
if RUN_EDA:
    display(train_dataset.tail(5))



## === cell 5
if RUN_EDA:
    display(train_dataset.dtypes)



## === cell 6
if RUN_EDA:
    train_dataset.info(memory_usage="deep")



## === cell 7
if RUN_EDA:
    for dtype in ["float", "int", "object"]:
        selected_dtype = train_dataset.select_dtypes(include=[dtype])
        mean_usage_b = selected_dtype.memory_usage(deep=True).mean()
        mean_usage_mb = mean_usage_b / 1024**2
        print(
            "Average memory usage for {} columns: {:03.2f} MB".format(
                dtype, mean_usage_mb
            )
        )



## === cell 8
train_key = train_dataset["key"].copy()
test_key = test_dataset["key"].copy()



## === cell 9
if RUN_EDA:
    train_dataset.info(memory_usage="deep")



## === cell 10
for ds in (train_dataset, test_dataset):
    ds["passenger_count"] = ds["passenger_count"].astype("uint8", errors="ignore")
    for col in [
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
    ]:
        ds[col] = ds[col].astype("float32", errors="ignore")
train_dataset["fare_amount"] = train_dataset["fare_amount"].astype(
    "float32", errors="ignore"
)



## === cell 11
if RUN_EDA:
    train_dataset.info(memory_usage="deep")



## === cell 12
if RUN_EDA:
    display(train_dataset.isnull().sum())



## === cell 13
if RUN_EDA:
    print(len(train_dataset))



## === cell 14
core_cols_train = [
    "key",
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
core_cols_test = [
    "key",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]

if RUN_EDA:
    print(f"Row count before drop-null operation - {train_dataset.shape[0]}")
train_dataset.dropna(subset=core_cols_train, inplace=True)
if RUN_EDA:
    print(f"Row count after drop-null operation - {train_dataset.shape[0]}")

test_dataset = test_dataset.copy()
test_dataset.dropna(subset=["key"], inplace=True)



## === cell 15
train_dataset["pickup_datetime"] = pd.to_datetime(
    arg=train_dataset["pickup_datetime"], infer_datetime_format=True, errors="coerce"
)
test_dataset["pickup_datetime"] = pd.to_datetime(
    arg=test_dataset["pickup_datetime"], infer_datetime_format=True, errors="coerce"
)

before_dt = train_dataset.shape[0]
train_dataset = train_dataset[train_dataset["pickup_datetime"].notna()]
if RUN_EDA:
    print(
        "Dropped NaT pickup_datetime rows (train):", before_dt - train_dataset.shape[0]
    )

test_dataset = test_dataset.copy()
test_dt_valid_mask = test_dataset["pickup_datetime"].notna()



## === cell 16
if RUN_EDA:
    display(train_dataset.dtypes)




## === cell 17
def add_new_date_time_features(dataset):
    dataset["hour"] = dataset.pickup_datetime.dt.hour
    dataset["day"] = dataset.pickup_datetime.dt.day
    dataset["month"] = dataset.pickup_datetime.dt.month
    dataset["year"] = dataset.pickup_datetime.dt.year
    dataset["day_of_week"] = dataset.pickup_datetime.dt.dayofweek
    return dataset


train_dataset = add_new_date_time_features(train_dataset)
test_dataset = add_new_date_time_features(test_dataset)

for ds in (train_dataset, test_dataset):
    for c in ["hour", "day", "month", "year", "day_of_week"]:
        ds[c] = pd.to_numeric(ds[c], errors="coerce").astype("float32")



## === cell 18
if RUN_EDA:
    display(train_dataset.describe())



## === cell 19
nyc_long_min, nyc_long_max = -74.3, -72.9
nyc_lat_min, nyc_lat_max = 40.5, 41.0

train_dataset = train_dataset[
    train_dataset["pickup_longitude"].between(nyc_long_min, nyc_long_max)
    & train_dataset["dropoff_longitude"].between(nyc_long_min, nyc_long_max)
    & train_dataset["pickup_latitude"].between(nyc_lat_min, nyc_lat_max)
    & train_dataset["dropoff_latitude"].between(nyc_lat_min, nyc_lat_max)
]



## === cell 20
train_dataset = train_dataset[
    (train_dataset["pickup_longitude"] != 0.0)
    & (train_dataset["pickup_latitude"] != 0.0)
    & (train_dataset["dropoff_longitude"] != 0.0)
    & (train_dataset["dropoff_latitude"] != 0.0)
]



## === cell 21
if RUN_EDA:
    display(train_dataset.describe())



## === cell 22
if RUN_EDA:
    display(train_dataset.fare_amount[train_dataset.fare_amount <= 0].count())



## === cell 23
train_dataset = train_dataset[train_dataset.fare_amount > 0]



## === cell 24
train_dataset = train_dataset[train_dataset["fare_amount"] <= 250.0]



## === cell 25
if RUN_EDA:
    display(
        train_dataset.passenger_count[
            (train_dataset.passenger_count < 1) | (train_dataset.passenger_count > 8)
        ].count()
    )



## === cell 26
train_dataset = train_dataset[train_dataset.passenger_count.between(1, 7)]




## === cell 27
def degree_to_radion(degree):
    return degree * (np.pi / 180.0)


def calculate_distance(
    pickup_latitude, pickup_longitude, dropoff_latitude, dropoff_longitude
):
    from_lat = degree_to_radion(pickup_latitude)
    from_long = degree_to_radion(pickup_longitude)
    to_lat = degree_to_radion(dropoff_latitude)
    to_long = degree_to_radion(dropoff_longitude)

    radius = 6371.01  # km

    lat_diff = to_lat - from_lat
    long_diff = to_long - from_long

    a = (np.sin(lat_diff / 2.0) ** 2) + (
        np.cos(from_lat) * np.cos(to_lat) * (np.sin(long_diff / 2.0) ** 2)
    )
    a = np.clip(a, 0.0, 1.0)  # numerical safety
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))

    return radius * c




## === cell 28
train_dataset["distance"] = calculate_distance(
    train_dataset.pickup_latitude,
    train_dataset.pickup_longitude,
    train_dataset.dropoff_latitude,
    train_dataset.dropoff_longitude,
)
test_dataset["distance"] = calculate_distance(
    test_dataset.pickup_latitude,
    test_dataset.pickup_longitude,
    test_dataset.dropoff_latitude,
    test_dataset.dropoff_longitude,
)



## === cell 29
if RUN_EDA:
    display(train_dataset.sort_values(by="distance"))



## === cell 30
if RUN_EDA:
    display(train_dataset[(train_dataset.distance == 0)].count())



## === cell 31
if RUN_EDA:
    display(
        train_dataset[
            (train_dataset.pickup_latitude != train_dataset.dropoff_latitude)
            & (train_dataset.pickup_longitude != train_dataset.dropoff_latitude)
            & (train_dataset.distance == 0)
        ].count()
    )




## === cell 32
def add_distances_from_airport(dataset):
    jfk_coords = (40.639722, -73.778889)
    ewr_coords = (40.6925, -74.168611)
    lga_coords = (40.77725, -73.872611)

    dataset["pickup_jfk_distance"] = calculate_distance(
        dataset.pickup_latitude, dataset.pickup_longitude, jfk_coords[0], jfk_coords[1]
    )
    dataset["dropof_jfk_distance"] = calculate_distance(
        dataset.dropoff_latitude,
        dataset.dropoff_longitude,
        jfk_coords[0],
        jfk_coords[1],
    )

    dataset["pickup_ewr_distance"] = calculate_distance(
        dataset.pickup_latitude, dataset.pickup_longitude, ewr_coords[0], ewr_coords[1]
    )
    dataset["dropof_ewr_distance"] = calculate_distance(
        dataset.dropoff_latitude,
        dataset.dropoff_longitude,
        ewr_coords[0],
        ewr_coords[1],
    )

    dataset["pickup_lga_distance"] = calculate_distance(
        dataset.pickup_latitude, dataset.pickup_longitude, lga_coords[0], lga_coords[1]
    )
    dataset["dropof_lga_distance"] = calculate_distance(
        dataset.dropoff_latitude,
        dataset.dropoff_longitude,
        lga_coords[0],
        lga_coords[1],
    )

    return dataset


train_dataset = add_distances_from_airport(train_dataset)
test_dataset = add_distances_from_airport(test_dataset)



## === cell 33
train_dataset = train_dataset[train_dataset["distance"].between(0.0, 100.0)]



## === cell 34
eps = 1e-3
fare_per_km = train_dataset["fare_amount"] / (train_dataset["distance"] + eps)
train_dataset = train_dataset[
    (fare_per_km <= 200.0) | (train_dataset["distance"] < 1.0)
]



## === cell 35
min_fare_per_km = (
    0.5  # $/km; very conservative floor to drop near-zero fares on long distances
)
train_dataset = train_dataset[
    (train_dataset["distance"] < 2.0)
    | (
        (train_dataset["fare_amount"] / (train_dataset["distance"] + eps))
        >= min_fare_per_km
    )
]



## === cell 36
if RUN_EDA:
    sns.distplot(train_dataset.fare_amount)



## === cell 37
if RUN_EDA:
    sns.jointplot(x="distance", y="fare_amount", data=train_dataset)



## === cell 38
if RUN_EDA:
    sns.countplot(x="day_of_week", data=train_dataset)



## === cell 39
if RUN_EDA:
    tc = train_dataset.pivot_table(
        index="day_of_week", columns="month", values="fare_amount"
    )
    sns.heatmap(data=tc)



## === cell 40
selected_predictors = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
    "pickup_jfk_distance",
    "dropof_jfk_distance",
    "pickup_ewr_distance",
    "dropof_ewr_distance",
    "pickup_lga_distance",
    "dropof_lga_distance",
    "hour",
    "day",
    "day_of_week",
    "month",
    "year",
    "distance",
]

test_valid_geo = (
    test_dataset["pickup_longitude"].between(nyc_long_min, nyc_long_max)
    & test_dataset["dropoff_longitude"].between(nyc_long_min, nyc_long_max)
    & test_dataset["pickup_latitude"].between(nyc_lat_min, nyc_lat_max)
    & test_dataset["dropoff_latitude"].between(nyc_lat_min, nyc_lat_max)
)
test_nonzero_geo = (
    (test_dataset["pickup_longitude"] != 0.0)
    & (test_dataset["pickup_latitude"] != 0.0)
    & (test_dataset["dropoff_longitude"] != 0.0)
    & (test_dataset["dropoff_latitude"] != 0.0)
)
test_valid_pax = test_dataset["passenger_count"].between(1, 7)
test_valid_dt = test_dt_valid_mask
test_valid_distance = test_dataset["distance"].between(0.0, 100.0)

test_dataset_sanitized = test_dataset.copy()
invalid_for_features = ~(
    test_valid_geo & test_nonzero_geo & test_valid_pax & test_valid_dt
)
cols_to_nan = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
    "hour",
    "day",
    "day_of_week",
    "month",
    "year",
    "distance",
    "pickup_jfk_distance",
    "dropof_jfk_distance",
    "pickup_ewr_distance",
    "dropof_ewr_distance",
    "pickup_lga_distance",
    "dropof_lga_distance",
]
test_dataset_sanitized.loc[invalid_for_features, cols_to_nan] = np.nan

X_df = train_dataset.loc[:, selected_predictors].copy()
y = train_dataset["fare_amount"].values

X_test_all_df = test_dataset_sanitized.loc[:, selected_predictors].copy()

X_df = X_df.replace([np.inf, -np.inf], np.nan)
X_test_all_df = X_test_all_df.replace([np.inf, -np.inf], np.nan)

valid_mask = ~X_df.isna().any(axis=1) & np.isfinite(y)
X_df = X_df.loc[valid_mask]
y = y[valid_mask.values]

medians = X_df.median(numeric_only=True)
X_df = X_df.fillna(medians)
X_test_all_df = X_test_all_df.fillna(medians)

X_df = X_df.fillna(0.0)
X_test_all_df = X_test_all_df.fillna(0.0)

X_df = X_df.replace([np.inf, -np.inf], 0.0)
X_test_all_df = X_test_all_df.replace([np.inf, -np.inf], 0.0)

test_feature_finite = np.isfinite(X_test_all_df.values).all(axis=1)

test_valid_mask = (
    test_valid_geo
    & test_nonzero_geo
    & test_valid_pax
    & test_valid_dt
    & test_valid_distance
    & pd.Series(test_feature_finite, index=test_dataset_sanitized.index)
).fillna(False)

X = X_df.values
X_test_all = X_test_all_df.values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=1 / 5, random_state=42, shuffle=False
)
if RUN_EDA:
    print(f"X_test size - {len(X_test)}")



## === cell 41
if RUN_INTERMEDIATE_MODELS:
    rand_forest_regressor = RandomForestRegressor(
        random_state=42, n_estimators=200, n_jobs=-1
    )
    rand_forest_regressor.fit(X_train, y_train)

    y_rand_forest_predict = rand_forest_regressor.predict(X_test)
    print(f"random forest size - {len(y_rand_forest_predict)}")
    random_forest_model_error = sqrt(mean_squared_error(y_test, y_rand_forest_predict))
    print(f" Random Forest RMSE - {random_forest_model_error}")



## === cell 42
if RUN_INTERMEDIATE_MODELS:
    XGB_regressor = XGBRegressor(random_state=42, n_jobs=-1)
    XGB_regressor.fit(X_train, y_train)

    y_XGB_predict = XGB_regressor.predict(X_test)
    XGB_model_error = sqrt(mean_squared_error(y_test, y_XGB_predict))
    print(f" XGBoost RMSE - {XGB_model_error}")



## === cell 43
if RUN_EDA and RUN_INTERMEDIATE_MODELS:
    sns.barplot(
        y=list(train_dataset.loc[:, selected_predictors].columns),
        x=list(XGB_regressor.feature_importances_),
    )



## === cell 44
if RUN_INTERMEDIATE_MODELS:
    XGB_regressor = XGBRegressor(
        learning_rate=0.07, max_depth=6, n_estimators=300, random_state=42, n_jobs=-1
    )
    XGB_regressor.fit(X_train, y_train)
    y_XGB_predict = XGB_regressor.predict(X_test)

    XGB_model_error = sqrt(mean_squared_error(y_test, y_XGB_predict))
    print(f" XGBoost RMSE - {XGB_model_error}")



## === cell 45
if RUN_EDA and RUN_INTERMEDIATE_MODELS:
    sns.barplot(
        y=list(train_dataset.loc[:, selected_predictors].columns),
        x=list(XGB_regressor.feature_importances_),
    )



## === cell 46
XGB_regressor_final = XGBRegressor(
    learning_rate=0.07, max_depth=6, n_estimators=300, random_state=42, n_jobs=-1
)
XGB_regressor_final.fit(X, y)

y_XGB_predict_all = XGB_regressor_final.predict(X_test_all)

y_XGB_predict_all = np.where(np.isfinite(y_XGB_predict_all), y_XGB_predict_all, 0.0)
y_XGB_predict_all = np.maximum(y_XGB_predict_all, 0.0)

train_fare_fallback = float(np.mean(y))
y_XGB_predict_all = np.where(
    test_valid_mask.values, y_XGB_predict_all, train_fare_fallback
)

test_dataset["fare_amount"] = y_XGB_predict_all

submission = pd.DataFrame(
    {"key": test_dataset["key"].values, "fare_amount": y_XGB_predict_all}
)
if RUN_EDA:
    display(submission.head(10))



## === cell 47
if RUN_EDA:
    sns.jointplot(x="distance", y="fare_amount", data=test_dataset)



## === cell 48
if RUN_EDA:
    display(test_dataset[test_dataset.distance > 100])



## === cell 49
if RUN_EDA:
    display(test_dataset[test_dataset.fare_amount > 150])



## === cell 50
submission = submission[["key", "fare_amount"]].copy()
submission.to_csv("submission_new.csv", index=False)
print("Wrote submission_new.csv with shape:", submission.shape)
print("Unique keys:", submission["key"].nunique(), "Total rows:", len(submission))
print(
    "Invalid/sanitized test rows:",
    int((~test_valid_mask).sum()),
    "out of",
    len(test_valid_mask),
)
