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
scipy==1.15.3
sklearn-pandas==2.2.0
tqdm==4.67.1

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

3.3504463754491907

# 6. Current score

5.57826

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 4.31827) has done: 'I remove the hard dependencies on external “nyc-weather”, “us-bank-holidays”, and pre-trained kmeans pickle inputs (none exist in your environment) by generating safe fallback feature columns so downstream code keeps working with the same overall modeling pipeline. I also fix a few logic/typing bugs that would hurt both runtime and score (incorrect `drop(..., axis=0)`, incorrect SNOW/TMIN assignments, and a missing underscore in `day_hour` for test). Finally, I update the LightGBM training call to be compatible with lightgbm==4.6.0 (no `early_stop/verbose_eval` params; use callbacks), and replace deprecated `np.asscalar` with a safe float conversion so a valid `submission_lgb.csv` is always written.'
- What this solution (achieved 4.3716) has done: 'Your current score (4.31827 RMSE) is worse than the target (3.35045), so we should improve predictive quality with minimal risk while preserving the same overall pipeline. The biggest avoidable score hit in your code is a train/test mismatch: `hot_day`/`cold_day` thresholds differ between train and test, which shifts feature meaning at inference and degrades RMSE; we make them identical. We also remove early stopping (it’s currently effectively truncating training because `stopping_rounds == num_boost_round`), so the model consistently trains the full intended 100 rounds without changing the model type or loss. Finally, we keep predictions non-negative but also clip extreme outliers to a reasonable upper bound to reduce the RMSE impact of rare huge predictions (a standard, minimal post-processing step for this competition).'
- What this solution (achieved 4.27903) has done: 'Your current RMSE (4.3716) is worse than the target (3.3504), so we should make small, low-risk changes that typically improve generalization without changing the overall pipeline. The biggest win with minimal disruption is to make LightGBM use a more reasonable learning rate and enough boosting rounds (keeping GBDT + RMSE objective identical) so it can actually fit the signal rather than underfitting. We also add LightGBM’s built-in row/feature subsampling seeds for determinism and keep your existing non-negative + upper clipping post-processing unchanged. Finally, we keep all feature engineering and data handling intact and still write `submission_lgb.csv` in the required format.'
- What this solution (achieved 4.78307) has done: 'Your current RMSE (4.279) is worse than the target (3.350), so we should improve generalization with minimal, low-risk changes while keeping the same overall pipeline (same features, same LightGBM regressor, same loss/metric). The biggest avoidable issue is that you train for 1000 rounds but then force prediction with `num_iteration=num_rounds` instead of the model’s best/actual iteration; switching to `model_lgb.best_iteration` (when available) typically reduces overfitting and improves RMSE without changing the modeling approach. We also add a small, standard data cleaning step for this competition (removing obviously invalid lat/long ranges and extreme fare outliers) which usually yields a meaningful RMSE improvement while preserving core semantics. Finally, we keep your submission format identical and still write `submission_lgb.csv`.'
- What this solution (achieved 7.14613) has done: 'We’re currently worse than the target (RMSE 4.783 > 3.350; lower is better), so the safest path is to improve generalization without changing the overall approach (same feature set idea + LightGBM regressor + RMSE). The largest likely score issue is a train/test feature distribution mismatch caused by filtering training rows to the *test* min/max lat/long (this can remove a lot of valid NYC rides and distort the learned fare-vs-geo mapping); I remove that test-dependent filter while keeping the basic geo validity filter. I also make LightGBM treat the listed categorical columns as categorical (still same model family/objective), which typically improves performance with these high-cardinality IDs compared to one-hotting, and I keep the same submission schema and clipping. These changes are minimal and should move RMSE toward the target without changing the evaluation semantics.'
- What this solution (achieved 6.36117) has done: 'I fix the shape-mismatch bug in the numeric imputation step that causes cell 15 to crash (it happens because some numeric columns are all-NaN, so `SimpleImputer` drops them and returns fewer columns than expected). To keep the pipeline and model semantics the same, I ensure numeric columns exist and impute with a constant 0 only for columns that are entirely missing, while using the median imputer for the rest. This unblock training so `X_train/X_valid` are defined and the LightGBM cells run, and it reliably write `submission_lgb.csv` with the required `key,fare_amount` columns. I also add a tiny safety guard to ensure predictions are finite before clipping, without changing the model/feature logic.'
- What this solution (achieved 5.78847) has done: 'Your current RMSE (6.36117) is far worse than the target (3.35045, lower is better), so we should make small, low-risk changes that improve generalization without changing the overall LightGBM approach or feature set. The biggest likely driver is the random `train_test_split` on a huge dataset, which can leak near-duplicate rides across train/valid and cause early stopping to pick an iteration that doesn’t transfer well to Kaggle’s test; switching to a time-based split keeps the same training loop but makes the chosen `best_iteration` more robust. I also add standard NYC taxi data cleaning for a few well-known corrupt patterns (missing/zero coordinates, identical pickup/dropoff, unrealistic distances) to reduce label noise while preserving your feature engineering and model. Finally, I keep the same submission schema and clipping, ensuring the script still runs end-to-end and writes `submission_lgb.csv`.'
- What this solution (achieved 5.60962) has done: 'Your current RMSE (5.78847) is much worse than the target (3.35045; lower is better), so we should improve generalization with minimal, low-risk edits while keeping the same feature engineering + LightGBM training approach. The biggest likely issue is that with a time-based split and `early_stopping`, you’re selecting `best_iteration` on a very late-time validation slice whose distribution can differ from Kaggle’s full test period; switching back to a stratified-by-time *random* split (still respecting time as a feature, but mixing eras) typically improves leaderboard RMSE for this competition without changing the model class or features. Additionally, the current cleaning removes all “same pickup/dropoff” rides, which includes legitimate short trips and biases the model; replacing that with a “near-zero distance” filter (which you already do) keeps data quality while retaining real short rides. Finally, we keep the same clipping/post-processing and still write `submission_lgb.csv` in the required format.'
- What this solution (achieved 5.95461) has done: 'Your current RMSE (5.60962; lower is better) is far worse than the target (3.35045), so we should improve generalization with minimal changes while keeping the same feature set and LightGBM training approach. The biggest low-risk win here is to stop treating inherently-categorical integer fields (month/year/neighborhood/passenger_count/etc.) as numeric categories that can become NaN via `to_numeric` coercion; instead we cast them directly to pandas `category` (and keep `day_hour` as category), ensuring consistent category alignment between train/test. We also include `TMAX` and `TMIN` in the numeric feature set (they’re already created/merged and are used for `hot_day/cold_day`, but giving the model the raw temps typically helps and doesn’t change the pipeline style). Finally, we add a very standard NYC cleanup step to remove extreme-distance outliers (keeping your existing geo + fare + passenger filters) to reduce label noise without altering model type or loss.'
- What this solution (achieved 5.95459) has done: 'Your RMSE (5.95) is much worse than the target (3.35, lower is better), so we make the smallest changes that typically improve generalization without changing your overall LightGBM pipeline. The main score leak here is that you’re training on 15M rows without forcing LightGBM to use a sufficient number of threads, which often causes time-pressure/instability in Kaggle environments and can indirectly degrade the effective model fit; we set `num_threads` explicitly and keep everything else the same. We also fix a subtle but important categorical handling issue: for LightGBM categorical features, using pandas `category` is correct, but the train/valid split can carry unused categories differently; we ensure categories are aligned before splitting (same semantics, less noise). Finally, we keep your existing clipping but tighten the upper clip slightly to reduce RMSE sensitivity to rare extreme predictions (still legitimate post-processing for this competition).'
- What this solution (achieved 5.57826) has done: 'Your RMSE (5.95, lower is better) is far above the target (3.35), so we should fix the biggest likely generalization issue with the smallest safe edits: your model is trained on a random 10% validation split, which makes early stopping noisy and often picks a suboptimal iteration for the leaderboard. I switch to a deterministic time-based split (by `pickup_datetime`) so early stopping selects a more stable `best_iteration`, while keeping the same LightGBM model/objective/features and the same training loop structure. I also ensure the categorical columns are explicitly `category`-typed in the final split frames (sometimes the split can upcast), which improves LightGBM’s categorical handling without changing features. Submission writing stays identical (`submission_lgb.csv` with `key,fare_amount`) and within the same paths.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import pickle
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from tqdm import tqdm
from scipy.sparse import csr_matrix, hstack
from sklearn.impute import SimpleImputer
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder

np.random.seed(17)

for p in [
    "../input",
    "/kaggle/input",
    "/kaggle/data",
    "/kaggle/data/new-york-city-taxi-fare-prediction",
]:
    if os.path.exists(p):
        print(f"Listing {p}:")
        print(os.listdir(p)[:50])



## === cell 1
weather_path_candidates = [
    "../input/nyc-weather/nyc_weather.csv",
    "/kaggle/input/nyc-weather/nyc_weather.csv",
]
nyc_weather = None
for wp in weather_path_candidates:
    if os.path.exists(wp):
        nyc_weather = pd.read_csv(wp)
        break

weather_cols = ["DATE", "AWND", "PRCP", "SNOW", "TMAX", "TMIN"]
if nyc_weather is None:
    nyc_weather = pd.DataFrame(columns=weather_cols)
else:
    nyc_weather = nyc_weather[weather_cols].copy()
    nyc_weather["DATE"] = pd.to_datetime(
        nyc_weather["DATE"], utc=True, format="%m/%d/%Y", errors="coerce"
    )

nyc_weather.head()



## === cell 2
if len(nyc_weather) > 0 and {"TMAX", "TMIN"}.issubset(nyc_weather.columns):
    plt.rc("figure", figsize=(15, 8))
    plt.subplot(1, 2, 1)
    plt.hist(nyc_weather.TMAX.dropna(), bins=30)
    plt.xlabel("Temperature (C)")
    plt.ylabel("Frequency Count")
    plt.title("Max Daily Temperature")
    plt.subplot(1, 2, 2)
    plt.hist(nyc_weather.TMIN.dropna(), bins=30)
    plt.xlabel("Temperature (C)")
    plt.ylabel("Frequency Count")
    plt.title("Min Daily Temperature")
    plt.show()
else:
    print("nyc_weather not available in this environment; skipping weather plots.")



## === cell 3
if len(nyc_weather) > 0:
    try:
        display(nyc_weather.describe())
    except Exception:
        print(nyc_weather.describe())
else:
    print("nyc_weather not available in this environment; skipping describe().")



## === cell 4
train_path_candidates = [
    "../input/new-york-city-taxi-fare-prediction/train.csv",
    "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv",
    "/kaggle/data/train.csv",
    "/kaggle/data/new-york-city-taxi-fare-prediction/train.csv",
]
test_path_candidates = [
    "../input/new-york-city-taxi-fare-prediction/test.csv",
    "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv",
    "/kaggle/data/test.csv",
    "/kaggle/data/new-york-city-taxi-fare-prediction/test.csv",
]

train_path = next((p for p in train_path_candidates if os.path.exists(p)), None)
test_path = next((p for p in test_path_candidates if os.path.exists(p)), None)
if train_path is None or test_path is None:
    raise FileNotFoundError(
        "Could not locate train.csv/test.csv in the provided environment paths."
    )

train = pd.read_csv(train_path, nrows=15_000_000)
test = pd.read_csv(test_path)
test_id = test.key.values  # set this value for final submission
print(train.info())



## === cell 5
train["pickup_datetime"] = train["pickup_datetime"].str.slice(0, 16)
test["pickup_datetime"] = test["pickup_datetime"].str.slice(0, 16)

train["pickup_datetime"] = pd.to_datetime(
    train["pickup_datetime"], utc=True, format="%Y-%m-%d %H:%M", errors="coerce"
)
test["pickup_datetime"] = pd.to_datetime(
    test["pickup_datetime"], utc=True, format="%Y-%m-%d %H:%M", errors="coerce"
)

train_sample = train.dropna().copy()
del train

train_sample.drop(labels="key", axis=1, inplace=True)
test.drop(labels="key", axis=1, inplace=True)

train_sample.loc[:, "passenger_count"] = train_sample.passenger_count.astype(
    dtype="uint8"
)
train_sample["pickup_longitude"] = train_sample.pickup_longitude.astype(dtype="float32")
train_sample["pickup_latitude"] = train_sample.pickup_latitude.astype(dtype="float32")
train_sample["dropoff_longitude"] = train_sample.dropoff_longitude.astype(
    dtype="float32"
)
train_sample["dropoff_latitude"] = train_sample.dropoff_latitude.astype(dtype="float32")
train_sample["fare_amount"] = train_sample.fare_amount.astype(dtype="float32")

test["pickup_longitude"] = test.pickup_longitude.astype(dtype="float32")
test["pickup_latitude"] = test.pickup_latitude.astype(dtype="float32")
test["dropoff_longitude"] = test.dropoff_longitude.astype(dtype="float32")
test["dropoff_latitude"] = test.dropoff_latitude.astype(dtype="float32")

train_sample = train_sample.loc[train_sample.fare_amount.between(2.5, 200.0)]
train_sample = train_sample.loc[train_sample.passenger_count.between(1, 6)]

valid_geo = (
    train_sample["pickup_longitude"].between(-74.5, -72.5)
    & train_sample["dropoff_longitude"].between(-74.5, -72.5)
    & train_sample["pickup_latitude"].between(40.0, 41.8)
    & train_sample["dropoff_latitude"].between(40.0, 41.8)
)
train_sample = train_sample.loc[valid_geo]

train_sample["hour"] = train_sample["pickup_datetime"].dt.hour
train_sample["month"] = train_sample["pickup_datetime"].dt.month
train_sample["day_of_week"] = train_sample["pickup_datetime"].dt.dayofweek
train_sample["year"] = train_sample["pickup_datetime"].dt.year

test["hour"] = test["pickup_datetime"].dt.hour
test["month"] = test["pickup_datetime"].dt.month
test["day_of_week"] = test["pickup_datetime"].dt.dayofweek
test["year"] = test["pickup_datetime"].dt.year

train_sample["hour"] = train_sample.hour.astype(dtype="uint8")
train_sample["month"] = train_sample.month.astype(dtype="uint8")
train_sample["day_of_week"] = train_sample.day_of_week.astype(dtype="uint8")
train_sample["year"] = train_sample.year.astype(dtype="uint16")

test["hour"] = test.hour.astype(dtype="uint8")
test["month"] = test.month.astype(dtype="uint8")
test["day_of_week"] = test.day_of_week.astype(dtype="uint8")
test["year"] = test.year.astype(dtype="uint16")



## === cell 6
train_sample["pickup_day"] = train_sample.pickup_datetime.dt.floor("d")
test["pickup_day"] = test.pickup_datetime.dt.floor("d")

if len(nyc_weather) > 0:
    train_sample = train_sample.merge(
        nyc_weather, how="left", left_on="pickup_day", right_on="DATE"
    )
    test = test.merge(nyc_weather, how="left", left_on="pickup_day", right_on="DATE")
    train_sample.drop(columns=["pickup_day", "DATE"], inplace=True)
    test.drop(columns=["pickup_day", "DATE"], inplace=True)
else:
    for c in ["AWND", "PRCP", "SNOW", "TMAX", "TMIN"]:
        train_sample[c] = np.nan
        test[c] = np.nan
    train_sample.drop(columns=["pickup_day"], inplace=True)
    test.drop(columns=["pickup_day"], inplace=True)

for c in ["AWND", "PRCP", "SNOW", "TMAX", "TMIN"]:
    train_sample[c] = train_sample[c].astype("float16")
    test[c] = test[c].astype("float16")

HOT_TMAX_C = 30.0
COLD_TMIN_C = 0.0
train_sample["hot_day"] = np.where(train_sample.TMAX >= HOT_TMAX_C, 1, 0)
train_sample["cold_day"] = np.where(train_sample.TMIN <= COLD_TMIN_C, 1, 0)
test["hot_day"] = np.where(test.TMAX >= HOT_TMAX_C, 1, 0)
test["cold_day"] = np.where(test.TMIN <= COLD_TMIN_C, 1, 0)

train_sample["hot_day"] = train_sample.hot_day.astype(dtype="uint8")
train_sample["cold_day"] = train_sample.cold_day.astype(dtype="uint8")
test["hot_day"] = test.hot_day.astype(dtype="uint8")
test["cold_day"] = test.cold_day.astype(dtype="uint8")


def degree_to_radion(degree):
    return degree * (np.pi / 180)


def calculate_distance(
    pickup_latitude, pickup_longitude, dropoff_latitude, dropoff_longitude
):
    from_lat = degree_to_radion(pickup_latitude)
    from_long = degree_to_radion(pickup_longitude)
    to_lat = degree_to_radion(dropoff_latitude)
    to_long = degree_to_radion(dropoff_longitude)

    radius = 6371.01
    lat_diff = to_lat - from_lat
    long_diff = to_long - from_long

    a = (
        np.sin(lat_diff / 2) ** 2
        + np.cos(from_lat) * np.cos(to_lat) * np.sin(long_diff / 2) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return radius * c


train_sample["distance"] = calculate_distance(
    train_sample.pickup_latitude,
    train_sample.pickup_longitude,
    train_sample.dropoff_latitude,
    train_sample.dropoff_longitude,
)
test["distance"] = calculate_distance(
    test.pickup_latitude,
    test.pickup_longitude,
    test.dropoff_latitude,
    test.dropoff_longitude,
)

train_sample["distance"] = train_sample.distance.astype(dtype="float32")
test["distance"] = test.distance.astype(dtype="float32")

train_sample = train_sample.loc[train_sample["distance"].between(0.05, 60.0)]

train_sample["day_hour"] = (
    train_sample.day_of_week.astype(str) + "_" + train_sample.hour.astype(str)
)
train_sample["day_hour"] = train_sample["day_hour"].astype("category")

test["day_hour"] = test.day_of_week.astype(str) + "_" + test.hour.astype(str)
test["day_hour"] = test["day_hour"].astype("category")

train_sample = train_sample[train_sample.fare_amount > 0]
print(train_sample.info())



## === cell 7
holiday_path_candidates = [
    "../input/us-bank-holidays-20092018/US Bank Holidays 2012-2018.csv",
    "/kaggle/input/us-bank-holidays-20092018/US Bank Holidays 2012-2018.csv",
]
holidays_path = next((p for p in holiday_path_candidates if os.path.exists(p)), None)

if holidays_path is not None:
    holidays = pd.read_csv(holidays_path)
    holidays["Date"] = pd.to_datetime(
        holidays["Date"], utc=True, format="%m/%d/%y", errors="coerce"
    )

    train_sample["pickup_day"] = train_sample.pickup_datetime.dt.floor("d")
    train_sample = train_sample.merge(
        holidays, left_on="pickup_day", right_on="Date", how="left"
    )
    train_sample["Holiday"] = train_sample.Holiday.fillna("None")

    le = LabelEncoder()
    train_sample["holiday"] = le.fit_transform(train_sample.Holiday.values)
    train_sample.drop(["Holiday", "Date", "pickup_day"], axis=1, inplace=True)

    test["pickup_day"] = test.pickup_datetime.dt.floor("d")
    test = test.merge(holidays, left_on="pickup_day", right_on="Date", how="left")
    test["Holiday"] = test.Holiday.fillna("None")
    test["holiday"] = (
        le.transform(test.Holiday.values)
        if "None" in le.classes_
        else le.fit_transform(test.Holiday.values)
    )
    test.drop(["Holiday", "Date", "pickup_day"], axis=1, inplace=True)
else:
    train_sample["holiday"] = 0
    test["holiday"] = 0

train_sample["holiday"] = train_sample.holiday.astype(dtype="uint8")
test["holiday"] = test.holiday.astype(dtype="uint8")
print(train_sample.info())



## === cell 8
train_sample.head()



## === cell 9
full_pickups = pd.concat(
    [
        train_sample[["pickup_longitude", "pickup_latitude"]],
        test[["pickup_longitude", "pickup_latitude"]],
    ],
    axis=0,
)
full_pickups.columns = ["x", "y"]
full_dropoffs = pd.concat(
    [
        train_sample[["dropoff_longitude", "dropoff_latitude"]],
        test[["dropoff_longitude", "dropoff_latitude"]],
    ],
    axis=0,
)
full_dropoffs.columns = ["x", "y"]
full_locs = pd.concat([full_pickups, full_dropoffs], axis=0)

full_locs["x"] = full_locs.x.round(4)
full_locs["y"] = full_locs.y.round(4)

full_locs = full_locs.groupby(["x", "y"]).count().reset_index()
full_locs.info()



## === cell 10
X_df = full_locs.copy()
X_kmeans = full_locs[["x", "y"]].values

num_clusters = 200
kmeans_path_candidates = [
    "../input/taxi-weather-holidays-kmeans-neighborhoods/kmeans_200_round4.pkl",
    "/kaggle/input/taxi-weather-holidays-kmeans-neighborhoods/kmeans_200_round4.pkl",
]
kmeans_path = next((p for p in kmeans_path_candidates if os.path.exists(p)), None)

if kmeans_path is not None:
    with open(kmeans_path, "rb") as fid:
        kmeans = pickle.load(fid)
else:
    kmeans = KMeans(n_clusters=num_clusters, random_state=17, n_init=10)
    kmeans.fit(X_kmeans)



## === cell 11
z = kmeans.predict(X_kmeans)



## === cell 12
centers = kmeans.cluster_centers_

x_centers = [pair[0] for pair in centers]
y_centers = [pair[1] for pair in centers]
z_centers = np.arange(num_clusters)

plt.subplot(1, 2, 1)
plt.scatter(X_df["x"], X_df["y"], c=z, s=5)
plt.gray()
plt.xlabel("Pickup/Dropoff Longitude")
plt.ylabel("Pickup/Dropoff Latitude")
plt.title("Clusters of NYC locations")
plt.subplot(1, 2, 2)
plt.scatter(x_centers, y_centers, c=z_centers, s=20)
plt.gray()
plt.xlabel("Pickup/Dropoff Longitude")
plt.ylabel("Pickup/Dropoff Latitude")
plt.title("Cluster Centers of NYC locations")

plt.show()



## === cell 13
del X_kmeans, X_df, full_pickups, full_dropoffs

train_sample["pickup_neighborhood"] = kmeans.predict(
    np.column_stack(
        [train_sample.pickup_longitude.values, train_sample.pickup_latitude.values]
    )
)
train_sample["dropoff_neighborhood"] = kmeans.predict(
    np.column_stack(
        [train_sample.dropoff_longitude.values, train_sample.dropoff_latitude.values]
    )
)

test["pickup_neighborhood"] = kmeans.predict(
    np.column_stack([test.pickup_longitude.values, test.pickup_latitude.values])
)
test["dropoff_neighborhood"] = kmeans.predict(
    np.column_stack([test.dropoff_longitude.values, test.dropoff_latitude.values])
)

train_sample["pickup_neighborhood"] = train_sample.pickup_neighborhood.astype(
    dtype="uint8"
)
train_sample["dropoff_neighborhood"] = train_sample.dropoff_neighborhood.astype(
    dtype="uint8"
)
test["pickup_neighborhood"] = test.pickup_neighborhood.astype(dtype="uint8")
test["dropoff_neighborhood"] = test.dropoff_neighborhood.astype(dtype="uint8")

print(train_sample.info())



## === cell 14
try:
    with open("kmeans_200_round4_v2.pkl", "wb") as fid:
        pickle.dump(kmeans, fid)
except Exception as e:
    print(f"Could not save kmeans pickle: {e}")



## === cell 15
categorical_cols = [
    "day_hour",
    "month",
    "year",
    "pickup_neighborhood",
    "dropoff_neighborhood",
    "passenger_count",
    "hot_day",
    "cold_day",
    "holiday",
]
numerical_cols = ["distance", "AWND", "PRCP", "SNOW", "TMAX", "TMIN"]

for c in categorical_cols + numerical_cols:
    if c not in train_sample.columns:
        train_sample[c] = np.nan
    if c not in test.columns:
        test[c] = np.nan

train_num = train_sample.reindex(columns=numerical_cols).astype("float32")
test_num = test.reindex(columns=numerical_cols).astype("float32")

all_nan_cols = [c for c in numerical_cols if train_num[c].notna().sum() == 0]
usable_cols = [c for c in numerical_cols if c not in all_nan_cols]

X_train_num_imp = pd.DataFrame(index=train_num.index)
X_test_num_imp = pd.DataFrame(index=test_num.index)

if len(usable_cols) > 0:
    num_imputer = SimpleImputer(strategy="median")
    X_train_num_imp_arr = num_imputer.fit_transform(train_num[usable_cols].to_numpy())
    X_test_num_imp_arr = num_imputer.transform(test_num[usable_cols].to_numpy())

    X_train_num_imp = pd.DataFrame(
        X_train_num_imp_arr, columns=usable_cols, index=train_num.index
    ).astype("float32")
    X_test_num_imp = pd.DataFrame(
        X_test_num_imp_arr, columns=usable_cols, index=test_num.index
    ).astype("float32")

for c in all_nan_cols:
    X_train_num_imp[c] = np.float32(0.0)
    X_test_num_imp[c] = np.float32(0.0)

X_train_num_imp = X_train_num_imp.reindex(columns=numerical_cols).astype("float32")
X_test_num_imp = X_test_num_imp.reindex(columns=numerical_cols).astype("float32")

full_cat = pd.concat(
    [train_sample[categorical_cols], test[categorical_cols]], axis=0
).copy()
cat_categories = {}
for c in categorical_cols:
    full_cat[c] = full_cat[c].astype("category")
    cat_categories[c] = full_cat[c].cat.categories

train_cat = pd.DataFrame(index=train_sample.index)
test_cat = pd.DataFrame(index=test.index)
for c in categorical_cols:
    train_cat[c] = pd.Categorical(train_sample[c], categories=cat_categories[c])
    test_cat[c] = pd.Categorical(test[c], categories=cat_categories[c])

X_train_df = pd.concat([train_cat, X_train_num_imp], axis=1)
X_test_df = pd.concat([test_cat, X_test_num_imp], axis=1)

y = train_sample.fare_amount.values

pickup_time = train_sample["pickup_datetime"].astype(
    "int64"
)  # ns since epoch; deterministic ordering
order = np.argsort(pickup_time.values, kind="mergesort")
n = len(order)
valid_size = int(round(0.1 * n))
valid_idx = train_sample.index[order[-valid_size:]]
train_idx = train_sample.index[order[:-valid_size]]

X_train = X_train_df.loc[train_idx]
y_train = y[X_train_df.index.get_indexer(train_idx)]
X_valid = X_train_df.loc[valid_idx]
y_valid = y[X_train_df.index.get_indexer(valid_idx)]

for c in categorical_cols:
    X_train[c] = X_train[c].astype("category")
    X_valid[c] = X_valid[c].astype("category")
    X_test_df[c] = X_test_df[c].astype("category")

categorical_feature_names = categorical_cols
print("Train/valid sizes:", X_train.shape, X_valid.shape)



## === cell 16
params = {
    "objective": "regression",
    "boosting": "gbdt",
    "metric": "rmse",
    "num_leaves": 50,
    "max_depth": 8,
    "learning_rate": 0.1,
    "bagging_fraction": 0.8,
    "feature_fraction": 0.8,
    "min_split_gain": 0.02,
    "min_child_samples": 10,
    "min_child_weight": 0.02,
    "lambda_l2": 0.0475,
    "verbosity": -1,
    "data_random_seed": 17,
    "bagging_seed": 17,
    "feature_fraction_seed": 17,
    "num_threads": max(1, os.cpu_count() or 1),
}

d_train = lgb.Dataset(
    X_train,
    label=y_train,
    categorical_feature=categorical_feature_names,
    free_raw_data=False,
)
d_valid = lgb.Dataset(
    X_valid,
    label=y_valid,
    categorical_feature=categorical_feature_names,
    free_raw_data=False,
)
watchlist = [d_train, d_valid]

num_rounds = 1000
verbose_eval = 100

model_lgb = lgb.train(
    params,
    train_set=d_train,
    num_boost_round=num_rounds,
    valid_sets=watchlist,
    valid_names=["train", "valid"],
    callbacks=[
        lgb.log_evaluation(period=verbose_eval),
        lgb.early_stopping(stopping_rounds=100, verbose=False),
    ],
)

best_iter = model_lgb.best_iteration if model_lgb.best_iteration else num_rounds
pred_valid_y_lgb = model_lgb.predict(X_valid, num_iteration=best_iter)
print("LGB Loss = " + str(sqrt(mean_squared_error(y_valid, pred_valid_y_lgb))))
print("Using best_iter:", best_iter)



## === cell 17
lgb_public = model_lgb.predict(X_test_df, num_iteration=best_iter)
final_pred_public = lgb_public.flatten()

test_predictions_lgb = np.asarray(final_pred_public, dtype=np.float32)
test_predictions_lgb = np.nan_to_num(
    test_predictions_lgb, nan=0.0, posinf=250.0, neginf=0.0
)

test_predictions_lgb = np.clip(test_predictions_lgb, 0.0, 200.0)

sample = pd.DataFrame({"key": test_id, "fare_amount": test_predictions_lgb})
sample = sample.reindex(["key", "fare_amount"], axis=1)

sample.to_csv("submission_lgb.csv", index=False)
print(sample.head())
print("Wrote submission_lgb.csv with shape:", sample.shape)



## === cell 18
plt.rc("figure", figsize=(10, 10))
plt.hist(sample["fare_amount"].values, bins=100)
plt.xlabel("Predicticted Price")
plt.ylabel("Frequency")
plt.title("Predictions from LGB")
plt.show()
