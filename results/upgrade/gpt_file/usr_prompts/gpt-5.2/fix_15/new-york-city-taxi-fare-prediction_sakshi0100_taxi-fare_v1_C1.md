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

3.9

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

5.68932

# 6. Current score

36.72008

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 936.93312) has done: 'I fix the runtime error caused by the removed `normalize` parameter in scikit-learn’s `LinearRegression` by replacing it with an equivalent `Pipeline(StandardScaler(with_mean=False), LinearRegression())`, which preserves the intended normalization behavior without changing the core linear model approach. I also address a likely training-time crash due to the huge `for` loops over 10M rows by switching the datetime feature extraction to vectorized pandas operations (same semantics, but feasible within the time limit). Finally, I make sure train/test one-hot columns are aligned and the submission is written as a valid `Submission.csv` with columns exactly `key,fare_amount`.'
- What this solution (achieved 752.76918) has done: 'Your huge RMSE is primarily coming from inconsistent/incorrect scaling and column alignment: the “manual normalization” for Difference_* is fit separately on train vs test (and uses test’s own mean/var), and the weekday alignment code builds `df = df[fixed]` without actually reassigning back to the original dataframes, so train/test column order can silently differ. I (1) compute Difference_* normalization stats on the training data and apply them to both train and test, (2) fix the weekday column alignment by explicitly reindexing both dataframes to the same ordered set of columns, and (3) ensure the final feature matrix for prediction matches the exact training feature columns. These are minimal changes that preserve your linear regression approach and feature set, but should drastically reduce the RMSE toward the target.'
- What this solution (achieved 864.32864) has done: 'Your current RMSE is massively worse than the target, so we need a small but decisive correction rather than tuning: the linear model is being trained on unfiltered/invalid targets (negative/zero fares and extreme outliers) and on rows with bad coordinates, which makes predictions explode and ruins RMSE. I add the standard NYC Taxi fare competition sanity filters (fare range, passenger_count range, and lat/lon bounding box) while keeping your exact feature set and the same LinearRegression-in-a-StandardScaler pipeline. I also clip negative predictions to 0 (a valid fare can’t be negative), which typically reduces RMSE without changing the model/training logic. Everything else (feature engineering, one-hot weekday, distance features, pipeline, and submission format) remains the same.'
- What this solution (achieved 864.32864) has done: 'Your RMSE is still extremely high versus the target, which strongly suggests a remaining correctness issue rather than “model quality”. The smallest high-impact fix is to ensure train/test feature columns are aligned in *exactly the same order* after weekday one-hot creation (your current code adds missing columns but never reorders the full dataframe to a common column order). I add a single, explicit reindex of both `train_data` and `test_data` to the same `all_feature_cols` right after dummy creation, and I also guard against any non-finite feature values (NaN/inf) leaking into the linear regression, which can destabilize predictions and RMSE. Everything else (features, LinearRegression + StandardScaler pipeline, filters, clipping, and submission schema) stays the same.'
- What this solution (achieved 15.09281) has done: 'Your RMSE is still orders of magnitude above the target, so the most likely remaining issue is a feature correctness bug rather than “model strength”. The smallest high-impact fix here is that you drop rows with NaN *after* computing `feature_cols` and splitting, which can leave X/y misaligned and silently train on mismatched targets, causing wild predictions. I (1) enforce a single final `dropna()` on the full training dataframe *before* building `X`/`y`, (2) make the `train_data`/`test_data` feature columns align once at the end (after all feature engineering), and (3) add a safe upper clip on predictions (using the same `fare_max` you already use) to prevent outlier predictions from blowing up RMSE, without changing your model or features.'
- What this solution (achieved 15.26954) has done: 'Your current RMSE (15.09) is far worse than the target (5.69), so we need a small correctness-oriented fix rather than tuning. The largest remaining issue is that `Difference_*` is “normalized” by dividing by the *variance* (and taking an absolute after mean-subtraction), which badly distorts scale and can explode feature magnitudes; I switch this to the standard z-score using the *standard deviation* learned from train and applied to both train/test (same feature, same linear model, but correct scaling). I also ensure we drop any rows with non-finite engineered features *before* building `X/y` and keep train/test columns aligned exactly once at the end, preserving your architecture and training loop. The submission format and paths stay the same and a valid `Submission.csv` is produced.'
- What this solution (achieved 15.26954) has done: 'Your RMSE is still far above the target, so the most likely remaining issue is feature/target misalignment caused by filtering train rows without applying the same row selection to `y`. I make the smallest fix by (1) computing a single boolean mask of finite features and applying it to both `X` and `y` so they stay perfectly aligned, and (2) ensuring `test_data`’s feature matrix is built from the exact same `feature_cols` in the same order. Everything else (data read size, features, filters, LinearRegression+StandardScaler pipeline, clipping, and submission schema) stays the same, but this correction should move RMSE materially toward your target.'
- What this solution (achieved 15.26954) has done: 'Your RMSE is still far above the target, so the most likely remaining issue is a correctness bug rather than model capacity. The biggest one here is inconsistent missing-value handling: you drop NaNs in train very early but later create new NaNs (from `pickuptime` parsing and datetime coercion) and never drop them from train, while you fill NaNs in test with 0.0—this train/test mismatch can badly skew a linear model. I make the smallest fix by enforcing one final `dropna()` on the fully-engineered training dataframe right before building `X/y` (keeping semantics intact), and I also fill any remaining NaNs in train features with 0.0 (matching test) before fitting. Everything else (features, filters, LinearRegression + StandardScaler pipeline, clipping, and submission writing) stays the same.'
- What this solution (achieved 15.26954) has done: 'Your RMSE is still far above the target, which usually indicates a feature correctness issue rather than the linear model itself. The smallest high-impact fix is to correct the Haversine distance calculation: the code currently uses `dlon = lon2 - lon1` but should use `lon2 - lon1` where `lon1` is pickup and `lon2` is dropoff; fixing this (and the repeated airport-distance block) preserves your exact feature set and linear-regression pipeline, but makes the engineered distance features meaningful. I also enforce the same final NaN/inf handling after all distance features are created (train and test), so the model doesn’t see mismatched missing-value patterns. Everything else (filters, one-hot weekday, scaling pipeline, clipping, submission schema/path) stays the same and it still write a valid `Submission.csv`.'
- What this solution (achieved 16.42075) has done: 'Your current RMSE (15.27) is far worse than the target (5.69), so we should make a small correctness-oriented change that improves generalization without changing the model or feature set. The biggest remaining issue is that the linear model is trained on raw `fare_amount`, which has a heavy tail; switching to a log1p target transform (fit on log fares, then expm1 back) keeps the same linear regression core but typically moves this competition’s RMSE much closer to ~5–6. I also remove the hard rounding of engineered distance features (and final prediction rounding) because it throws away signal and can inflate RMSE; this preserves the same features/semantics but with full precision. Everything else (filters, one-hot, scaling pipeline, train/test alignment, clipping, and submission schema/path) stays the same and still writes a valid `Submission.csv`.'
- What this solution (achieved 15.26952) has done: 'Your RMSE is still far above the target, so we keep the same LinearRegression+StandardScaler pipeline and the same general feature set, but fix two correctness issues that strongly affect RMSE: (1) your `Difference_*` features are normalized even though you already scale all features in the pipeline (double-scaling can distort them), so we remove that manual normalization to let the pipeline handle scaling consistently; (2) the log1p target transform can hurt RMSE on this competition (metric is on dollars, not log-dollars), so we revert to training on raw `fare_amount` while keeping identical model/loop. We also add the standard “haversine distance for Distance” to be computed in kilometers consistently (previously mixed constants), but keep the same distance core logic. The script still run end-to-end and write a valid `Submission.csv` with `key,fare_amount`.'
- What this solution (achieved 15.27513) has done: 'Your current RMSE (15.27) is far worse than the target (5.69), so we should make a small, correctness-oriented fix that typically yields a large RMSE drop without changing your model/pipeline: add the standard outlier filter on the engineered `Distance` feature (remove implausibly long trips) and ensure `pickuptime` parsing can’t create NaNs/invalid ints that later get filled inconsistently. These changes keep the same LinearRegression+StandardScaler core, the same feature set, and the same training loop, but remove a major source of label/feature noise that inflates RMSE. I also align train/test NaN handling for engineered features by filling remaining train NaNs with 0.0 (matching test) right before building `X/y`, while keeping your existing dropna filters intact. The script still run end-to-end and write a valid `Submission.csv` with `key,fare_amount`.'
- What this solution (achieved 36.71979) has done: 'Your RMSE gap to target is large (15.28 vs 5.69, lower-is-better), and the biggest likely remaining issue is that you’re training on a non-representative 10M-row slice and a random split (which breaks the original Kaggle “train early, test later” distribution), plus the model still sees some noisy rows that pass basic bbox filters. I keep your exact LinearRegression+StandardScaler pipeline and the same engineered features, but (1) change the training sampling to a deterministic random sample from the full file via chunked reading (same data, just not biased to the file head) within the same 10M budget, and (2) add the standard “fare per km” sanity filter (removes mislabeled/erroneous trips that inflate RMSE) while keeping your existing distance filter. These are minimal, correctness-oriented dataset fixes that typically move this competition’s RMSE much closer to the ~5–7 band without changing the model architecture or feature extraction semantics. The script still runs end-to-end under constraints and writes a valid `Submission.csv` with `key,fare_amount`.'
- What this solution (achieved 36.72008) has done: 'Your current RMSE (36.72, lower-is-better) is far above the target (5.69), which strongly suggests a remaining data sampling correctness issue rather than model capacity. Right now you take `need` rows from every 1M-row chunk (and if a chunk is bigger you sample within it), which unintentionally keeps essentially the first 10M rows of the file and does not produce a uniform sample of the full 55M rows; this can badly hurt leaderboard RMSE due to distribution shift. I switch to a deterministic, chunk-wise uniform random sampling that draws a fixed fraction from each chunk to reach ~10M total (same core linear model, same features, same filters), and I also apply the exact same numeric sanitization to train as you already do for test (fill remaining NaNs with 0.0 after feature engineering) to avoid train/test mismatch. Everything else (feature engineering, filters, LinearRegression+StandardScaler pipeline, clipping, and submission schema/path) stays the same and still writes `Submission.csv`.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
train_path = "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv"
target_n = 10_000_000
chunk_size = 1_000_000
rng = np.random.RandomState(80)

approx_total_rows = 55_423_856
sample_frac = min(1.0, target_n / approx_total_rows)

train_chunks = []
kept = 0
for i, chunk in enumerate(pd.read_csv(train_path, chunksize=chunk_size)):
    if kept >= target_n:
        break

    n_take = int(round(len(chunk) * sample_frac))
    n_take = max(1, n_take)  # ensure we keep some rows from each chunk

    need = target_n - kept
    if n_take > need:
        n_take = need

    take = chunk.sample(n=n_take, random_state=80 + i)
    train_chunks.append(take)
    kept += len(take)

train_data = pd.concat(train_chunks, ignore_index=True)
train_data.head()



## === cell 2
train_data.shape



## === cell 3
train_data.info()



## === cell 4
test_data = pd.read_csv("/kaggle/input/new-york-city-taxi-fare-prediction/test.csv")
test_data.head()



## === cell 5
test_data.info()



## === cell 6
train_data["Difference_longitude"] = np.abs(
    np.asarray(train_data["pickup_longitude"] - train_data["dropoff_longitude"])
)
train_data["Difference_latitude"] = np.abs(
    np.asarray(train_data["pickup_latitude"] - train_data["dropoff_latitude"])
)

test_data["Difference_longitude"] = np.abs(
    np.asarray(test_data["pickup_longitude"] - test_data["dropoff_longitude"])
)
test_data["Difference_latitude"] = np.abs(
    np.asarray(test_data["pickup_latitude"] - test_data["dropoff_latitude"])
)



## === cell 7
print(f"Before Dropping null values: {len(train_data)}")
train_data.dropna(inplace=True)
print(f"After Dropping null values: {len(train_data)}")



## === cell 8
try:
    plot = train_data[:2000].plot.scatter("Difference_longitude", "Difference_latitude")
except Exception as e:
    print("Plot skipped:", repr(e))



## === cell 9
train_data = train_data[
    (train_data["Difference_longitude"] < 5.0)
    & (train_data["Difference_latitude"] < 5.0)
]



## === cell 10
fare_min, fare_max = 2.5, 250.0
train_data = train_data[
    (train_data["fare_amount"] >= fare_min) & (train_data["fare_amount"] <= fare_max)
]

train_data = train_data[
    (train_data["passenger_count"] >= 1) & (train_data["passenger_count"] <= 6)
]

train_data = train_data[
    (train_data["pickup_longitude"].between(-74.3, -72.7))
    & (train_data["dropoff_longitude"].between(-74.3, -72.7))
    & (train_data["pickup_latitude"].between(40.5, 41.8))
    & (train_data["dropoff_latitude"].between(40.5, 41.8))
]

train_data.reset_index(drop=True, inplace=True)
train_data.shape



## === cell 11
train_data["pickuptime"] = train_data["pickup_datetime"].str.slice(11, 16)
test_data["pickuptime"] = test_data["pickup_datetime"].str.slice(11, 16)



## === cell 12
train_data.head()



## === cell 13
train_dt = pd.to_datetime(
    train_data["pickup_datetime"].str.slice(0, 19), errors="coerce"
)
test_dt = pd.to_datetime(test_data["pickup_datetime"].str.slice(0, 19), errors="coerce")

train_data["Weekday"] = train_dt.dt.weekday
test_data["Weekday"] = test_dt.dt.weekday
test_data.head()



## === cell 14
train_data.drop("pickup_datetime", inplace=True, axis=1)
test_data.drop("pickup_datetime", inplace=True, axis=1)



## === cell 15
train_data["Weekday"].replace(
    to_replace=[i for i in range(0, 7)],
    value=[
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ],
    inplace=True,
)
test_data["Weekday"].replace(
    to_replace=[i for i in range(0, 7)],
    value=[
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ],
    inplace=True,
)



## === cell 16
train_one_hot = pd.get_dummies(train_data["Weekday"])
test_one_hot = pd.get_dummies(test_data["Weekday"])

train_data = pd.concat([train_data, train_one_hot], axis=1)
test_data = pd.concat([test_data, test_one_hot], axis=1)



## === cell 17
train_data.drop("Weekday", axis=1, inplace=True)
test_data.drop("Weekday", axis=1, inplace=True)



## === cell 18
weekday_cols = sorted(list(set(train_one_hot.columns).union(set(test_one_hot.columns))))
for col in weekday_cols:
    if col not in train_data.columns:
        train_data[col] = 0
    if col not in test_data.columns:
        test_data[col] = 0

non_target_cols_train = [c for c in train_data.columns if c != "fare_amount"]
all_cols = sorted(list(set(non_target_cols_train).union(set(test_data.columns))))
train_data = train_data.reindex(
    columns=(["fare_amount"] + [c for c in all_cols if c != "fare_amount"])
)
test_data = test_data.reindex(columns=[c for c in all_cols if c != "fare_amount"])



## === cell 19
hhmm_train = train_data["pickuptime"].str.split(":", expand=True)
train_hh = pd.to_numeric(hhmm_train[0], errors="coerce")
train_mm = pd.to_numeric(hhmm_train[1], errors="coerce")
train_data["pickuptime"] = (train_hh * 100 + train_mm).astype("float64")

hhmm_test = test_data["pickuptime"].str.split(":", expand=True)
test_hh = pd.to_numeric(hhmm_test[0], errors="coerce")
test_mm = pd.to_numeric(hhmm_test[1], errors="coerce")
test_data["pickuptime"] = (test_hh * 100 + test_mm).astype("float64")



## === cell 20
train_data.head()



## === cell 21
R_km = 6371.0088

lat1 = np.asarray(np.radians(train_data["pickup_latitude"]))
lon1 = np.asarray(np.radians(train_data["pickup_longitude"]))
lat2 = np.asarray(np.radians(train_data["dropoff_latitude"]))
lon2 = np.asarray(np.radians(train_data["dropoff_longitude"]))

dlon = lon2 - lon1
dlat = lat2 - lat1
a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
distance_km = R_km * c
train_data["Distance"] = np.asarray(distance_km)

lat1 = np.asarray(np.radians(test_data["pickup_latitude"]))
lon1 = np.asarray(np.radians(test_data["pickup_longitude"]))
lat2 = np.asarray(np.radians(test_data["dropoff_latitude"]))
lon2 = np.asarray(np.radians(test_data["dropoff_longitude"]))

dlon = lon2 - lon1
dlat = lat2 - lat1
a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
distance_km = R_km * c
test_data["Distance"] = np.asarray(distance_km)



## === cell 22
R = 6373.0

lat1 = np.asarray(np.radians(train_data["pickup_latitude"]))
lon1 = np.asarray(np.radians(train_data["pickup_longitude"]))
lat2 = np.asarray(np.radians(train_data["dropoff_latitude"]))
lon2 = np.asarray(np.radians(train_data["dropoff_longitude"]))

lat3 = np.zeros(len(train_data)) + np.radians(40.6413111)
lon3 = np.zeros(len(train_data)) + np.radians(-73.7781391)

dlon_pickup = lon3 - lon1
dlat_pickup = lat3 - lat1
dlon_dropoff = lon3 - lon2
dlat_dropoff = lat3 - lat2

a1 = (
    np.sin(dlat_pickup / 2) ** 2
    + np.cos(lat1) * np.cos(lat3) * np.sin(dlon_pickup / 2) ** 2
)
c1 = 2 * np.arctan2(np.sqrt(a1), np.sqrt(1 - a1))
distance1 = R * c1
train_data["Pickup_Distance_airport"] = np.asarray(distance1) * 0.621

a2 = (
    np.sin(dlat_dropoff / 2) ** 2
    + np.cos(lat2) * np.cos(lat3) * np.sin(dlon_dropoff / 2) ** 2
)
c2 = 2 * np.arctan2(np.sqrt(a2), np.sqrt(1 - a2))
distance2 = R * c2
train_data["Dropoff_Distance_airport"] = np.asarray(distance2) * 0.621

lat1 = np.asarray(np.radians(test_data["pickup_latitude"]))
lon1 = np.asarray(np.radians(test_data["pickup_longitude"]))
lat2 = np.asarray(np.radians(test_data["dropoff_latitude"]))
lon2 = np.asarray(np.radians(test_data["dropoff_longitude"]))

lat3 = np.zeros(len(test_data)) + np.radians(40.6413111)
lon3 = np.zeros(len(test_data)) + np.radians(-73.7781391)

dlon_pickup = lon3 - lon1
dlat_pickup = lat3 - lat1
dlon_dropoff = lon3 - lon2
dlat_dropoff = lat3 - lat2

a1 = (
    np.sin(dlat_pickup / 2) ** 2
    + np.cos(lat1) * np.cos(lat3) * np.sin(dlon_pickup / 2) ** 2
)
c1 = 2 * np.arctan2(np.sqrt(a1), np.sqrt(1 - a1))
distance1 = R * c1
test_data["Pickup_Distance_airport"] = np.asarray(distance1) * 0.621

a2 = (
    np.sin(dlat_dropoff / 2) ** 2
    + np.cos(lat2) * np.cos(lat3) * np.sin(dlon_dropoff / 2) ** 2
)
c2 = 2 * np.arctan2(np.sqrt(a2), np.sqrt(1 - a2))
distance2 = R * c2
test_data["Dropoff_Distance_airport"] = np.asarray(distance2) * 0.621



## === cell 23
train_data = train_data[
    (train_data["Distance"] > 0.0) & (train_data["Distance"] <= 100.0)
].reset_index(drop=True)



## === cell 24
eps = 1e-6
fare_per_km = train_data["fare_amount"] / (train_data["Distance"] + eps)
train_data = train_data[(fare_per_km >= 0.5) & (fare_per_km <= 50.0)].reset_index(
    drop=True
)



## === cell 25
train_data.drop(
    ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"],
    axis=1,
    inplace=True,
)
test_data.drop(
    ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"],
    axis=1,
    inplace=True,
)



## === cell 26
train_data.replace([np.inf, -np.inf], np.nan, inplace=True)
test_data.replace([np.inf, -np.inf], np.nan, inplace=True)

train_data.fillna(0.0, inplace=True)
test_data.fillna(0.0, inplace=True)



## === cell 27
train_data.shape



## === cell 28
test_data.shape



## === cell 29
from sklearn.model_selection import train_test_split

before_rows = len(train_data)
train_data = train_data.dropna().reset_index(drop=True)
after_rows = len(train_data)
print(f"Final engineered train dropna(): {before_rows} -> {after_rows}")

X = train_data.drop(["key", "fare_amount"], axis=1)
y = train_data["fare_amount"].astype("float64")

X = X.fillna(0.0)

finite_mask = np.isfinite(X.to_numpy(dtype="float64")).all(axis=1) & np.isfinite(
    y.to_numpy(dtype="float64")
)
X = X.loc[finite_mask].reset_index(drop=True)
y = y.loc[finite_mask].reset_index(drop=True)

feature_cols = sorted(X.columns.tolist())
X = X.reindex(columns=feature_cols)

test_data = test_data.copy()
for c in feature_cols:
    if c not in test_data.columns:
        test_data[c] = 0.0
test_data = test_data.reindex(columns=(["key"] + feature_cols), fill_value=0.0)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.01, random_state=80
)



## === cell 30
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

lr = Pipeline(
    steps=[
        ("scaler", StandardScaler(with_mean=False)),
        ("model", LinearRegression()),
    ]
)

lr.fit(X_train, y_train)
print("R^2 on raw target (diagnostic):", lr.score(X_test, y_test))



## === cell 31
X_submit = test_data.drop("key", axis=1).reindex(columns=feature_cols, fill_value=0.0)

pred = lr.predict(X_submit)

pred = np.clip(pred, 0.0, fare_max)



## === cell 32
pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/sample_submission.csv"
).head()



## === cell 33
Submission = pd.DataFrame(data=pred, columns=["fare_amount"])
Submission["key"] = test_data["key"]
Submission = Submission[["key", "fare_amount"]]



## === cell 34
Submission.to_csv("Submission.csv", index=False)
print("Wrote Submission.csv with shape:", Submission.shape)
print(Submission.head())
