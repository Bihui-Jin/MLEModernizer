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

3.50273

# 6. Current score

4.93545

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 12.97472) has done: 'You’re currently far from the target RMSE (4.44 vs 3.50, lower is better), so we make small, safe changes that reliably improve generalization without changing the core model type or feature set. The biggest issue is noisy/outlier training rows (invalid coordinates, unrealistically high fares, etc.) that a RandomForest overfit; adding standard NYC Taxi competition cleaning rules usually drops RMSE substantially while preserving the same features and training loop. I also make the train/validation split deterministic and align the internal evaluation to RMSE (your Kaggle metric) so you can track progress locally, without changing what gets submitted. Submission writing stays the same (creates `RFSubmission.csv` with `key,fare_amount`).'
- What this solution (achieved 9.49367) has done: 'You’re far above the target (RMSE 12.97 vs 3.50; lower is better), so the smallest reliable way to move toward the target is to improve data quality and align features with the metric without changing the model type or feature family. I keep your RandomForest and the same feature extraction pattern, but (1) add the standard NYC taxi “distance” feature (Haversine) while keeping your existing diffs, and (2) apply a couple of common, safe cleaning rules (remove extreme/unphysical distances) that otherwise dominate RMSE. I also clip negative test predictions to 0 (fares can’t be negative), which usually reduces RMSE a bit with no semantic change. Submission writing remains `RFSubmission.csv` with `key,fare_amount`.'
- What this solution (achieved 6.6708) has done: 'You’re currently above the target (9.49 vs 3.50 RMSE; lower is better), so we should make the smallest changes that reliably reduce RMSE without changing your RandomForest approach or feature family. The biggest remaining gap is usually caused by residual noisy/outlier rows; we tighten cleaning with standard NYC Taxi rules that don’t alter core semantics: remove “fare vs distance” inconsistencies, cap extreme coordinate deltas, and filter impossible timestamps. We also add a single robust, competition-standard feature (straight-line “manhattan” distance in km) while keeping your existing diffs and haversine. Finally, we keep submission writing identical (`RFSubmission.csv` with `key,fare_amount`) and preserve determinism.'
- What this solution (achieved 6.67642) has done: 'The timeout is dominated by fitting a 500-tree RandomForest twice (once on train/val split, then again on the full data), plus some avoidable pandas/numpy overhead when building feature matrices and computing datetime parts multiple times. I keep the exact same model, hyperparameters, features, and metrics, but eliminate the redundant first fit by training once on the full cleaned dataset and then evaluating on the held-out validation split using that same trained model (same semantics for predictions/metrics, just no extra training pass). I also speed up feature matrix construction by using a single vectorized `to_numpy()` slice (no Python loop) and compute datetime-derived columns with a single `.dt` accessor pass. These changes are correctness-preserving and reduce runtime substantially while keeping determinism (seed, random_state) intact.'
- What this solution (achieved 6.92426) has done: 'Your current RMSE (6.676) is still far above the target (3.503, lower is better), so we make the smallest changes that typically reduce RMSE without changing your core approach (same RandomForest, same training flow, same feature family). The biggest remaining lever is improving label quality by filtering out “bad” training rows that survive your current rules (notably trips with near-zero distance but non-trivial fare, and a few remaining coordinate edge cases), because these disproportionately inflate RMSE. I add two very standard NYC-taxi cleaning constraints (near-zero distance fare cap, and a slightly tighter coordinate bounding box) while keeping everything else intact. Submission writing stays identical and still produces `RFSubmission.csv` with `key,fare_amount`.'
- What this solution (achieved 6.34605) has done: 'Your current RMSE (6.924) is still much worse than the target (3.503, lower is better), so the smallest reliable lever is to improve label quality by removing remaining high-noise/outlier trips that RandomForest tends to fit badly. I keep your exact model, training flow, and feature set, but tighten cleaning with two competition-standard constraints: filter unrealistic passenger_count==0 and remove extreme fare-per-km “spikes” especially at very short distances. I also ensure the same cleaning-derived constraints are applied deterministically and keep the submission format/filename identical (`RFSubmission.csv` with `key,fare_amount`). These are minimal changes that typically reduce RMSE without changing the core approach.'
- What this solution (achieved 6.41371) has done: 'You’re still far above the target (RMSE 6.346 vs 3.503; lower is better), so we should improve generalization with the smallest changes that don’t alter your core RandomForest setup. The biggest remaining lever is training-sample representativeness: your 1,000,000 rows are currently just the *first* 1M in the file (time-ordered), which biases the model and typically hurts leaderboard RMSE; switching to a deterministic random sample via `skiprows` keeps the same feature set, cleaning, and model while usually giving a sizable RMSE drop. I also make `train_test_split` deterministic *and* representative by sampling before cleaning/feature engineering, and I align local reporting to RMSE (Kaggle metric) without changing what’s submitted. Submission writing stays identical and still outputs `RFSubmission.csv` with `key,fare_amount`.'
- What this solution (achieved 4.92728) has done: 'Your current RMSE (6.41) is still far above the target (3.50, lower is better), so we should make a small, reliable improvement without changing the model or feature set. The biggest remaining “minimal” lever is to reduce train/test feature distribution mismatch by applying the *same* coordinate/datetime sanity filters to the **test** rows that are clearly invalid (e.g., out-of-bounds NYC box, passenger_count outside 1–6), then fill those filtered predictions with a safe fallback (the cleaned-train mean fare) to keep the submission complete. This keeps your RandomForest, features, and training loop identical, but prevents extreme out-of-domain test points from producing wildly wrong predictions that inflate RMSE. I also keep submission format identical (`key,fare_amount` to `RFSubmission.csv`) and preserve determinism.'
- What this solution (achieved 4.93311) has done: 'Your current RMSE (4.927) is still well above the target (3.503, lower is better), so we should make the smallest change that reliably improves generalization without changing your RandomForest or feature set. The biggest remaining issue is that you’re training a 500-tree forest on ~1M rows, but your cleaning is still letting through “noisy but common” cases (especially very short trips with moderately high fares, and some fare-per-km extremes) that inflate RMSE on the test distribution. I add two very standard NYC Taxi cleaning constraints that only remove high-noise/outlier rows (tighten the short-trip fare cap and tighten fare-per-km bounds), and I keep everything else (features, model, training, submission writing) identical. This should move RMSE downward toward your target without any architectural changes.'
- What this solution (achieved 4.93811) has done: 'Your RMSE (4.933) is still well above the target (3.503), so we should make the smallest reliable change that improves generalization without altering your core RandomForest approach or feature set. The biggest remaining lever is that the training sample is random but still includes many “hard-noise” rows (label errors and rare outliers) that inflate RMSE; tightening *only* the most standard, competition-safe cleaning rules (without changing features/model) usually yields a meaningful drop. I add two minimal filters: (1) remove trips where pickup and dropoff are essentially identical (very tiny distance) but fare is non-trivially high, and (2) drop extremely long trips (still within your 50 km cap) that tend to be mislabeled/out-of-distribution for the public test. Everything else (sampling, features, RF hyperparameters, submission writing) stays the same.'
- What this solution (achieved 4.93545) has done: 'Your current RMSE (4.938) is still above the target (3.503, lower is better), so we make the smallest changes that typically reduce error without changing your model or feature set. The biggest remaining gain with RandomForest on this competition is usually from a couple of additional “standard” training-row cleaning rules that remove mislabeled/outlier trips your existing filters still keep (notably airport-long rides and extreme fare-per-km tails). I (1) tighten the long-trip handling by excluding very long trips unless they look plausible by fare-per-km, and (2) add a gentle upper bound on Manhattan distance (your model feature) to remove coordinate glitches that survive haversine constraints. Everything else—sampling, features, RandomForest hyperparameters/training flow, and submission writing—stays the same.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os

print(os.listdir("../input"))



## === cell 1
try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except Exception as e:
    print("sklearnex patch not applied:", repr(e))

np.random.seed(42)

n_train = 1_000_000

dtype_map = {
    "fare_amount": np.float32,
    "pickup_longitude": np.float64,
    "pickup_latitude": np.float64,
    "dropoff_longitude": np.float64,
    "dropoff_latitude": np.float64,
    "passenger_count": np.int8,
}

train_usecols = [
    "key",
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
test_usecols = [
    "key",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]

train_path = "../input/train.csv"

with open(train_path, "r") as f:
    n_total = sum(1 for _ in f) - 1  # excluding header

n_keep = min(n_train, n_total)
keep = np.random.choice(np.arange(1, n_total + 1), size=n_keep, replace=False)
keep_set = set(keep.tolist())

skiprows = (i for i in range(1, n_total + 1) if i not in keep_set)

df = pd.read_csv(
    train_path,
    usecols=train_usecols,
    parse_dates=["pickup_datetime"],
    dtype=dtype_map,
    skiprows=skiprows,
)

df_test = pd.read_csv(
    "../input/test.csv",
    usecols=test_usecols,
    parse_dates=["pickup_datetime"],
    dtype=dtype_map,
)

print("Loaded train rows:", len(df), "of total:", n_total)



## === cell 2
pass



## === cell 3
pass




## === cell 4
def add_travel_vector_features(df_):
    pickup_lon = df_["pickup_longitude"].to_numpy(dtype=np.float64, copy=False)
    pickup_lat = df_["pickup_latitude"].to_numpy(dtype=np.float64, copy=False)
    dropoff_lon = df_["dropoff_longitude"].to_numpy(dtype=np.float64, copy=False)
    dropoff_lat = df_["dropoff_latitude"].to_numpy(dtype=np.float64, copy=False)

    abs_diff_lon = np.abs(dropoff_lon - pickup_lon)
    abs_diff_lat = np.abs(dropoff_lat - pickup_lat)
    df_["abs_diff_longitude"] = abs_diff_lon
    df_["abs_diff_latitude"] = abs_diff_lat

    r = 6371.0
    lat1 = np.deg2rad(pickup_lat)
    lon1 = np.deg2rad(pickup_lon)
    lat2 = np.deg2rad(dropoff_lat)
    lon2 = np.deg2rad(dropoff_lon)

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * (
        np.sin(dlon / 2.0) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(a))
    df_["haversine_km"] = r * c

    mean_lat = np.deg2rad((pickup_lat + dropoff_lat) / 2.0)
    km_per_deg_lat = 111.32
    km_per_deg_lon = 111.32 * np.cos(mean_lat)
    df_["manhattan_km"] = abs_diff_lat * km_per_deg_lat + abs_diff_lon * km_per_deg_lon

    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    df_["bearing"] = np.arctan2(y, x)  # radians in [-pi, pi]


add_travel_vector_features(df)
add_travel_vector_features(df_test)

old_size = len(df)
df = df.dropna(how="any", axis="rows")
print("Old size: %d" % old_size)
print("New size: %d" % len(df))




## === cell 5
def clean_train_rows(df_):
    fare = df_["fare_amount"].to_numpy(copy=False)
    pax = df_["passenger_count"].to_numpy(copy=False)

    pu_lon = df_["pickup_longitude"].to_numpy(copy=False)
    pu_lat = df_["pickup_latitude"].to_numpy(copy=False)
    do_lon = df_["dropoff_longitude"].to_numpy(copy=False)
    do_lat = df_["dropoff_latitude"].to_numpy(copy=False)

    abs_dlon = df_["abs_diff_longitude"].to_numpy(copy=False)
    abs_dlat = df_["abs_diff_latitude"].to_numpy(copy=False)
    hav = df_["haversine_km"].to_numpy(copy=False)
    man = df_["manhattan_km"].to_numpy(copy=False)

    mask = np.ones(len(df_), dtype=bool)

    mask &= (fare > 0) & (fare <= 200)

    mask &= (pax >= 1) & (pax <= 6)

    mask &= (
        np.isfinite(pu_lon)
        & np.isfinite(pu_lat)
        & np.isfinite(do_lon)
        & np.isfinite(do_lat)
    )

    mask &= ~((pu_lon == 0) & (pu_lat == 0))
    mask &= ~((do_lon == 0) & (do_lat == 0))

    mask &= (pu_lon >= -74.3) & (pu_lon <= -73.7)
    mask &= (do_lon >= -74.3) & (do_lon <= -73.7)
    mask &= (pu_lat >= 40.5) & (pu_lat <= 41.0)
    mask &= (do_lat >= 40.5) & (do_lat <= 41.0)

    mask &= (abs_dlon + abs_dlat) > 0

    mask &= (hav > 0.05) & (hav < 50.0)
    mask &= (abs_dlon < 1.0) & (abs_dlat < 1.0)

    mask &= np.isfinite(man) & (man > 0.05) & (man <= 60.0)

    min_expected = 2.5 + 0.5 * hav
    max_expected = 10.0 + 15.0 * hav
    mask &= (fare >= min_expected) & (fare <= max_expected)

    years = df_["pickup_datetime"].dt.year.to_numpy(copy=False)
    mask &= (years >= 2009) & (years <= 2015)

    fare_per_km = fare / np.clip(hav, 0.1, None)
    mask &= (fare_per_km >= 2.0) & (fare_per_km <= 24.0)

    mask &= ~((hav < 0.35) & (fare > 10.0))

    mask &= ~((hav < 0.6) & (fare > 12.0))

    long_trip = hav > 30.0
    plausible_long = (fare_per_km >= 3.0) & (fare_per_km <= 12.0)
    mask &= (~long_trip) | plausible_long

    mask &= hav <= 35.0

    return df_.loc[mask]


old_len = len(df)
df = clean_train_rows(df)
print(f"After cleaning: {old_len} -> {len(df)} rows")




## === cell 6
def compute_test_valid_mask(df_):
    pu_lon = df_["pickup_longitude"].to_numpy(copy=False)
    pu_lat = df_["pickup_latitude"].to_numpy(copy=False)
    do_lon = df_["dropoff_longitude"].to_numpy(copy=False)
    do_lat = df_["dropoff_latitude"].to_numpy(copy=False)

    pax = df_["passenger_count"].to_numpy(copy=False)
    abs_dlon = df_["abs_diff_longitude"].to_numpy(copy=False)
    abs_dlat = df_["abs_diff_latitude"].to_numpy(copy=False)
    hav = df_["haversine_km"].to_numpy(copy=False)
    man = df_["manhattan_km"].to_numpy(copy=False)

    mask = np.ones(len(df_), dtype=bool)

    mask &= (pax >= 1) & (pax <= 6)

    mask &= (
        np.isfinite(pu_lon)
        & np.isfinite(pu_lat)
        & np.isfinite(do_lon)
        & np.isfinite(do_lat)
    )
    mask &= ~((pu_lon == 0) & (pu_lat == 0))
    mask &= ~((do_lon == 0) & (do_lat == 0))

    mask &= (pu_lon >= -74.3) & (pu_lon <= -73.7)
    mask &= (do_lon >= -74.3) & (do_lon <= -73.7)
    mask &= (pu_lat >= 40.5) & (pu_lat <= 41.0)
    mask &= (do_lat >= 40.5) & (do_lat <= 41.0)

    mask &= (abs_dlon + abs_dlat) > 0
    mask &= (hav > 0.05) & (hav < 50.0)
    mask &= (abs_dlon < 1.0) & (abs_dlat < 1.0)

    mask &= np.isfinite(man) & (man > 0.05) & (man <= 60.0)

    years = df_["pickup_datetime"].dt.year.to_numpy(copy=False)
    mask &= (years >= 2009) & (years <= 2015)

    return mask


test_valid_mask = compute_test_valid_mask(df_test)
print(
    "Test valid rows:",
    int(test_valid_mask.sum()),
    " / ",
    len(df_test),
    "(invalid will use mean-fare fallback)",
)



## === cell 7
min_year = df.pickup_datetime.dt.year.min()

dt_train = df["pickup_datetime"].dt
df["pickup_year"] = (dt_train.year - min_year).astype(np.int16, copy=False)
df["pickup_hour"] = dt_train.hour.astype(np.int8, copy=False)
df["pickup_day"] = dt_train.dayofyear.astype(np.int16, copy=False)

dt_test = df_test["pickup_datetime"].dt
df_test["pickup_year"] = (dt_test.year - min_year).astype(np.int16, copy=False)
df_test["pickup_hour"] = dt_test.hour.astype(np.int8, copy=False)
df_test["pickup_day"] = dt_test.dayofyear.astype(np.int16, copy=False)



## === cell 8
from sklearn.model_selection import train_test_split

df_train, df_val = train_test_split(df, test_size=0.1, random_state=42)
len(df_val)



## === cell 9
feature_cols = [
    "abs_diff_longitude",
    "abs_diff_latitude",
    "haversine_km",
    "manhattan_km",
    "bearing",
    "dropoff_longitude",
    "dropoff_latitude",
    "pickup_longitude",
    "pickup_latitude",
    "passenger_count",
    "pickup_year",
    "pickup_hour",
    "pickup_day",
]


def get_input_matrix(df_):
    X = df_.loc[:, feature_cols].to_numpy(dtype=np.float32, copy=False)
    return X


x_train, x_val = get_input_matrix(df_train), get_input_matrix(df_val)
y_train, y_val = np.asarray(df_train.fare_amount, dtype=np.float32), np.asarray(
    df_val.fare_amount, dtype=np.float32
)



## === cell 10
from sklearn.ensemble import RandomForestRegressor

reg = RandomForestRegressor(
    max_depth=25,
    n_estimators=500,
    oob_score=True,
    n_jobs=-1,
    min_samples_split=10,
    verbose=1,
    random_state=42,
)



## === cell 11
x_full = get_input_matrix(df)
y_full = np.asarray(df.fare_amount, dtype=np.float32)
reg.fit(x_full, y_full)



## === cell 12
reg.oob_score_



## === cell 13
from sklearn.metrics import r2_score

y_pred = reg.predict(x_val)



## === cell 14
score = r2_score(y_val, y_pred)
score



## === cell 15
from sklearn.metrics import mean_squared_error

rmse = mean_squared_error(y_val, y_pred, squared=False)
rmse



## === cell 16
mean_squared_error(y_val, y_pred)



## === cell 17
pass



## === cell 18
reg.score(x_val, y_val)



## === cell 19
x_test = get_input_matrix(df_test)

predictions = np.empty(len(df_test), dtype=np.float32)

train_mean_fare = float(np.mean(y_full))
predictions[~test_valid_mask] = train_mean_fare

pred_valid = reg.predict(x_test[test_valid_mask])
predictions[test_valid_mask] = pred_valid.astype(np.float32, copy=False)

predictions = np.clip(predictions, 0, None)

RFSubmission = pd.DataFrame({"key": df_test.key.ravel(), "fare_amount": predictions})
RFSubmission.to_csv("RFSubmission.csv", index=False)
print("Wrote RFSubmission.csv with shape:", RFSubmission.shape)
print("Fallback mean fare used for invalid test rows:", train_mean_fare)
print("Number of fallback rows:", int((~test_valid_mask).sum()))
