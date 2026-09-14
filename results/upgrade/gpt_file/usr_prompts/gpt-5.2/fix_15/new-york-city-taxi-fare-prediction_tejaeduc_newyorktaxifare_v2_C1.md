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
seaborn==0.12.2
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

3.22951

# 6. Current score

5.85642

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 7.3938) has done: 'Your current gap is large (RMSE 6.11 vs target 3.23, lower is better), so we should legitimately improve generalization without changing the overall model approach (still XGBoost on engineered features). The biggest score drag here is the post-processing `pred.round(2)` (it adds quantization error that directly hurts RMSE), plus a subtle train/test feature mismatch risk from one-hot encoding years separately. I (1) remove rounding, (2) make year dummies consistent by concatenating train+test before `get_dummies`, and (3) align XGBoost with modern parameters (`reg:squarederror`) and make the split deterministic; all are minimal and should move RMSE down toward your target without changing the core pipeline. The script still write `finaloutput.csv` with `key,fare_amount`.'
- What this solution (achieved 7.46635) has done: 'We keep your XGBoost-on-engineered-features pipeline intact, but remove early stopping because it can underfit and materially worsens RMSE relative to your target; instead we train the full fixed 150 rounds for a consistent boost in fit without changing the model class or features. We also make the distance/temporal feature engineering vectorized (same semantics) so the notebook reliably finishes within the time limit while still using the same features. Finally, we ensure predictions are valid (finite and non-negative) to avoid rare pathological outputs harming RMSE, and we still write `finaloutput.csv` with the required `key,fare_amount` columns.'
- What this solution (achieved 7.45943) has done: 'Your current RMSE (7.466) is much worse than the target (3.229), so we should make small, legitimate fixes that improve generalization without changing the overall “XGBoost on engineered features” pipeline. The biggest issue is that `year` is being one-hot encoded, which can unnecessarily fragment the signal and hurt RMSE on this classic baseline; we instead keep `year` as a numeric feature (same information, less sparsity) while leaving all other feature engineering intact. We also add two very standard, minimal time features (`hour` and `weekday`) derived from `pickup_datetime` (same feature-extraction approach you already use) which typically yields a large RMSE improvement for this competition. Everything else (data filtering, haversine distance, XGBoost training loop/rounds, and submission writing) stays the same and still produces `finaloutput.csv`.'
- What this solution (achieved 6.23095) has done: 'Your current RMSE (7.46) is far worse than the 3.23 target, so we should improve fit/generalization with minimal, legitimate tweaks while keeping the same “XGBoost on engineered features” pipeline. The biggest controllable gap in your current setup is that you train only on 1M rows out of ~55M; increasing the training sample (still via `nrows`, same approach) usually yields a large RMSE drop for this competition without changing model/feature logic. I also add two standard, low-risk features (`month` and `abs_lonlat` deltas) derived from existing columns (same feature-engineering style) which typically improves RMSE materially. Finally, I keep the same training loop/rounds and submission writing, only adding safe dtype casting and column alignment to avoid subtle train/test mismatches.'
- What this solution (achieved 6.46382) has done: 'Your RMSE (6.23) is still far above the target (3.229, lower is better), so we should make the smallest legitimate changes that typically yield a large RMSE drop for this competition while keeping the same “XGBoost on engineered features” pipeline. The biggest likely drag is that `passenger_count` isn’t cleaned for the common invalid value `0`, and the coordinate box filter is quite loose; tightening these filters slightly (still same approach: remove obvious outliers) usually improves generalization a lot. I also add two very standard location features (`pickup/dropoff` lat/lon) that were inadvertently excluded by your current feature set; this preserves the same model/training loop but gives XGBoost access to key signal. Everything else (haversine dist, time features, XGBoost training with 150 rounds, and writing `finaloutput.csv`) stays intact.'
- What this solution (achieved 5.2739) has done: 'You’re still far above the target RMSE (6.46 vs 3.23, lower is better), so we should make a small, legitimate improvement that typically yields a big drop for this specific competition without changing the overall “XGBoost on engineered tabular features” approach. The most impactful minimal fix is to use a log1p transform of `fare_amount` during training and then invert with expm1 at prediction time; this keeps the same model/training loop and loss (RMSE) semantics but stabilizes the heavy-tailed target and usually improves RMSE materially. I also add the standard `min_child_weight`, `subsample`, and `colsample_bytree` regularization knobs (still the same XGBoost model) to reduce overfit noise and move RMSE down. Everything else (rows read, filters, haversine/time features, DMatrix training, and `finaloutput.csv` submission writing) remains intact.'
- What this solution (achieved 4.89035) has done: 'Your current RMSE (5.2739) is still well above the target (3.2295), so we should make a small, legitimate improvement without changing the overall “XGBoost on engineered tabular features” approach. The most impactful minimal change is to add two standard, metric-aligned features that strongly correlate with fare: the straight-line “Manhattan” distance proxy (sum of absolute lat/lon deltas) and a bearing/heading feature; both are derived from the same coordinates you already use and keep the same modeling pipeline. We also add `gamma` and `reg_lambda` as mild regularization knobs to reduce overfitting noise (same objective/booster/training loop). Everything else (filters, log1p target, DMatrix training with 150 rounds, and writing `finaloutput.csv`) is kept intact.'
- What this solution (achieved 5.09032) has done: 'Your current RMSE (4.890) is still above the target (3.229, lower is better), so we should make small, legitimate improvements that keep the same “XGBoost on engineered features” core. The most impactful minimal change for this competition is to add two standard geospatial features derived from the same coordinates: distance to NYC center and distance to JFK airport, which help the model capture airport/Manhattan pricing effects without changing the training approach. We also tighten the coordinate filter slightly (still the same outlier-removal logic) to reduce noisy/invalid trips that harm RMSE. Everything else (log1p target, DMatrix + xgboost.train with 150 rounds, and writing `finaloutput.csv`) stays intact.'
- What this solution (achieved 5.78584) has done: 'We need to move RMSE down from 5.09032 toward 3.22951 (lower is better), so we make only small, legitimate improvements that keep the same XGBoost-on-engineered-features pipeline. The biggest minimal gain typically comes from adding the standard NYC taxi “trick” features that capture pricing structure: trip distance in miles plus a simple “airport flag” (near JFK/LGA) and a small set of interaction features, all derived from existing columns and keeping the same training loop/objective. We also add a very light additional outlier filter on `dist` (still the same outlier-removal approach you already use) to remove impossible trips that harm RMSE. Everything else (log1p target, DMatrix, 150 boosting rounds, and writing `finaloutput.csv`) stays unchanged.'
- What this solution (achieved 4.47484) has done: 'We need to move RMSE down from 5.78584 toward 3.22951 (lower is better), and the simplest legitimate gains for this competition usually come from (1) fixing any distribution shift due to too-loose geographic filtering and (2) adding the missing “NYC-specific” airport distance feature for Newark (EWR), which strongly affects fares but isn’t captured by JFK/LGA alone. I keep your exact XGBoost training approach (DMatrix + `xgboost.train`, 150 rounds, log1p target) and the same feature-engineering style, only adding EWR distances/flag and a small, safe tightening of the coordinate bounds to reduce obvious noise/outliers. I also apply the same distance-based outlier removal to the test set (only for impossible distances) to prevent pathological test rows from producing extreme predictions; submission alignment stays by `key`. The script still run end-to-end and write `finaloutput.csv` with `key,fare_amount`.'
- What this solution (achieved 5.28376) has done: 'Your RMSE (4.47484) is still above the target (3.22951), so we should make a small, legitimate improvement that keeps the same XGBoost + engineered-features + log1p pipeline intact. The biggest low-risk gain here is to stop “dropping” impossible-distance test rows to a constant mean (11.35), which creates large errors on those rows; instead we keep the rows and just clip `dist` into a sane range for test so the model can still produce reasonable predictions. We also add a very standard outlier filter on training fares vs distance (removing trips where fare is implausibly low/high for the computed distance), which reduces label noise and typically improves public RMSE without changing the model/training loop. Everything else (features, DMatrix + `xgboost.train`, 150 rounds, log1p target, submission format/path) stays the same.'
- What this solution (achieved 5.7645) has done: 'We need to reduce RMSE from 5.28376 toward 3.22951 (lower is better), so the smallest safe gain is to fix the remaining major source of label noise: cash-outlier trips where `fare_amount` is implausible for the computed trip distance. I keep your exact XGBoost + log1p pipeline and feature set, but replace the two very narrow fare-vs-distance filters with a single standard “fare per mile” sanity filter plus a minimal base-fare guard; this typically drops RMSE materially on this competition without changing the modeling approach. I also clip the test distance features (instead of leaving impossible `dist` values unhandled) so the model doesn’t extrapolate wildly, while preserving row alignment and writing the same `finaloutput.csv`. Everything else (rows read, engineered features, params, 150 rounds, submission format) stays the same.'
- What this solution (achieved 5.79109) has done: 'Your current RMSE (5.7645) is still far above the target (3.2295), so we should make a small, legitimate improvement without changing your core “XGBoost on engineered features + log1p target” pipeline. The biggest remaining generalization drag is that the model still has to learn strong spatial structure from raw lat/lon without an explicit “NYC zone” anchor; adding a very standard borough/zone-style proxy via discretized lat/lon bins (simple grid cells) often yields a sizable RMSE drop in this competition while keeping the same feature-extraction approach. To keep train/test semantics identical and avoid leakage, we create the same bins from fixed edges (not learned from labels), and one-hot encode them consistently by concatenating train+test before `get_dummies`. Everything else (filters, features, training loop, parameters, and submission writing) is kept the same.'
- What this solution (achieved 5.85642) has done: 'Your current RMSE (5.79) is still far above the target (3.229, lower is better), so we should focus on reducing label noise and capturing NYC fare structure with minimal, legitimate tweaks while keeping your XGBoost + engineered-features + log1p pipeline intact. The two smallest high-impact fixes are (1) adding the well-known “trip + airport + toll-ish” proxy feature `abs(bearing)` (keeps your existing bearing feature but makes it more usable) and (2) adding a simple `is_weekend` indicator (same datetime feature extraction you already do). I’m also tightening the fare-per-mile sanity filter slightly to remove a bit more obvious noise (without changing the overall filtering approach), which typically improves generalization RMSE for this competition. Everything else (data size, haversine, bins + get_dummies, XGBoost params/rounds, submission writing) stays the same and still writes `finaloutput.csv` with `key,fare_amount`.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
import xgboost
import os

print(os.listdir("../input"))



## === cell 1
df = pd.read_csv("../input/train.csv", nrows=5000000)



## === cell 2
test = pd.read_csv("../input/test.csv")



## === cell 3
testkey = test.key



## === cell 4
df = df.dropna(how="any", axis="rows")



## === cell 5
len(df)



## === cell 6
df.head()



## === cell 7
l = df[
    (df.pickup_latitude > 41.0)
    | (df.pickup_latitude < 40.55)
    | (df.dropoff_latitude > 41.0)
    | (df.dropoff_latitude < 40.55)
    | (df.pickup_longitude > -73.70)
    | (df.pickup_longitude < -74.25)
    | (df.dropoff_longitude > -73.70)
    | (df.dropoff_longitude < -74.25)
].index



## === cell 8
df = df.drop(l, axis=0)



## === cell 9
z = df[
    (df.fare_amount > 300.0)
    | (df.fare_amount < 0.0)
    | (df.passenger_count > 7.0)
    | (df.passenger_count < 1.0)
].index



## === cell 10
df = df.drop(z, axis=0)



## === cell 11
len(df)




## === cell 12
def haversine_km(lon1, lat1, lon2, lat2):
    lon1 = np.radians(lon1.astype("float64"))
    lat1 = np.radians(lat1.astype("float64"))
    lon2 = np.radians(lon2.astype("float64"))
    lat2 = np.radians(lat2.astype("float64"))

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = (np.sin(dlat / 2.0) ** 2) + np.cos(lat1) * np.cos(lat2) * (
        np.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return 6373.0 * c




## === cell 13
df["dist"] = haversine_km(
    df["pickup_longitude"].values,
    df["pickup_latitude"].values,
    df["dropoff_longitude"].values,
    df["dropoff_latitude"].values,
)



## === cell 14
test["dist"] = haversine_km(
    test["pickup_longitude"].values,
    test["pickup_latitude"].values,
    test["dropoff_longitude"].values,
    test["dropoff_latitude"].values,
)



## === cell 15
test.head()



## === cell 16
df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
test["pickup_datetime"] = pd.to_datetime(test["pickup_datetime"])



## === cell 17
df.info()



## === cell 18
df["latenights"] = (df["pickup_datetime"].dt.hour < 5).astype("int8")
test["latenights"] = (test["pickup_datetime"].dt.hour < 5).astype("int8")



## === cell 19
df["year"] = df["pickup_datetime"].dt.year.astype("int16")
test["year"] = test["pickup_datetime"].dt.year.astype("int16")



## === cell 20
df["day"] = df["pickup_datetime"].dt.day.astype("int8")
test["day"] = test["pickup_datetime"].dt.day.astype("int8")



## === cell 21
df["hour"] = df["pickup_datetime"].dt.hour.astype("int8")
test["hour"] = test["pickup_datetime"].dt.hour.astype("int8")
df["weekday"] = df["pickup_datetime"].dt.weekday.astype("int8")
test["weekday"] = test["pickup_datetime"].dt.weekday.astype("int8")

df["is_weekend"] = (df["weekday"] >= 5).astype("int8")
test["is_weekend"] = (test["weekday"] >= 5).astype("int8")



## === cell 22
df["month"] = df["pickup_datetime"].dt.month.astype("int8")
test["month"] = test["pickup_datetime"].dt.month.astype("int8")



## === cell 23
df["abs_lon_diff"] = (
    (df["pickup_longitude"] - df["dropoff_longitude"]).abs().astype("float32")
)
df["abs_lat_diff"] = (
    (df["pickup_latitude"] - df["dropoff_latitude"]).abs().astype("float32")
)
test["abs_lon_diff"] = (
    (test["pickup_longitude"] - test["dropoff_longitude"]).abs().astype("float32")
)
test["abs_lat_diff"] = (
    (test["pickup_latitude"] - test["dropoff_latitude"]).abs().astype("float32")
)



## === cell 24
df["manhattan"] = (df["abs_lon_diff"] + df["abs_lat_diff"]).astype("float32")
test["manhattan"] = (test["abs_lon_diff"] + test["abs_lat_diff"]).astype("float32")

dlon = (df["dropoff_longitude"] - df["pickup_longitude"]).astype("float64")
dlat = (df["dropoff_latitude"] - df["pickup_latitude"]).astype("float64")
df["bearing"] = np.arctan2(dlat, dlon).astype("float32")

dlon_t = (test["dropoff_longitude"] - test["pickup_longitude"]).astype("float64")
dlat_t = (test["dropoff_latitude"] - test["pickup_latitude"]).astype("float64")
test["bearing"] = np.arctan2(dlat_t, dlon_t).astype("float32")

df["abs_bearing"] = np.abs(df["bearing"]).astype("float32")
test["abs_bearing"] = np.abs(test["bearing"]).astype("float32")



## === cell 25
NYC_LON, NYC_LAT = -73.985428, 40.748817  # Midtown/Manhattan reference
JFK_LON, JFK_LAT = -73.7781, 40.6413  # JFK airport

df["pickup_to_center"] = haversine_km(
    df["pickup_longitude"].values,
    df["pickup_latitude"].values,
    np.full(len(df), NYC_LON),
    np.full(len(df), NYC_LAT),
).astype("float32")
df["dropoff_to_center"] = haversine_km(
    df["dropoff_longitude"].values,
    df["dropoff_latitude"].values,
    np.full(len(df), NYC_LON),
    np.full(len(df), NYC_LAT),
).astype("float32")
test["pickup_to_center"] = haversine_km(
    test["pickup_longitude"].values,
    test["pickup_latitude"].values,
    np.full(len(test), NYC_LON),
    np.full(len(test), NYC_LAT),
).astype("float32")
test["dropoff_to_center"] = haversine_km(
    test["dropoff_longitude"].values,
    test["dropoff_latitude"].values,
    np.full(len(test), NYC_LON),
    np.full(len(test), NYC_LAT),
).astype("float32")

df["pickup_to_jfk"] = haversine_km(
    df["pickup_longitude"].values,
    df["pickup_latitude"].values,
    np.full(len(df), JFK_LON),
    np.full(len(df), JFK_LAT),
).astype("float32")
df["dropoff_to_jfk"] = haversine_km(
    df["dropoff_longitude"].values,
    df["dropoff_latitude"].values,
    np.full(len(df), JFK_LON),
    np.full(len(df), JFK_LAT),
).astype("float32")
test["pickup_to_jfk"] = haversine_km(
    test["pickup_longitude"].values,
    test["pickup_latitude"].values,
    np.full(len(test), JFK_LON),
    np.full(len(test), JFK_LAT),
).astype("float32")
test["dropoff_to_jfk"] = haversine_km(
    test["dropoff_longitude"].values,
    test["dropoff_latitude"].values,
    np.full(len(test), JFK_LON),
    np.full(len(test), JFK_LAT),
).astype("float32")



## === cell 26
KM_TO_MILES = np.float32(0.621371)
df["dist_miles"] = (df["dist"].astype("float32") * KM_TO_MILES).astype("float32")
test["dist_miles"] = (test["dist"].astype("float32") * KM_TO_MILES).astype("float32")

LGA_LON, LGA_LAT = -73.8740, 40.7769
df["pickup_to_lga"] = haversine_km(
    df["pickup_longitude"].values,
    df["pickup_latitude"].values,
    np.full(len(df), LGA_LON),
    np.full(len(df), LGA_LAT),
).astype("float32")
df["dropoff_to_lga"] = haversine_km(
    df["dropoff_longitude"].values,
    df["dropoff_latitude"].values,
    np.full(len(df), LGA_LON),
    np.full(len(df), LGA_LAT),
).astype("float32")
test["pickup_to_lga"] = haversine_km(
    test["pickup_longitude"].values,
    test["pickup_latitude"].values,
    np.full(len(test), LGA_LON),
    np.full(len(test), LGA_LAT),
).astype("float32")
test["dropoff_to_lga"] = haversine_km(
    test["dropoff_longitude"].values,
    test["dropoff_latitude"].values,
    np.full(len(test), LGA_LON),
    np.full(len(test), LGA_LAT),
).astype("float32")

df["near_jfk"] = ((df["pickup_to_jfk"] < 2.0) | (df["dropoff_to_jfk"] < 2.0)).astype(
    "int8"
)
test["near_jfk"] = (
    (test["pickup_to_jfk"] < 2.0) | (test["dropoff_to_jfk"] < 2.0)
).astype("int8")
df["near_lga"] = ((df["pickup_to_lga"] < 2.0) | (df["dropoff_to_lga"] < 2.0)).astype(
    "int8"
)
test["near_lga"] = (
    (test["pickup_to_lga"] < 2.0) | (test["dropoff_to_lga"] < 2.0)
).astype("int8")

df["dist_x_hour"] = (df["dist_miles"] * df["hour"].astype("float32")).astype("float32")
test["dist_x_hour"] = (test["dist_miles"] * test["hour"].astype("float32")).astype(
    "float32"
)
df["dist_x_weekday"] = (df["dist_miles"] * df["weekday"].astype("float32")).astype(
    "float32"
)
test["dist_x_weekday"] = (
    test["dist_miles"] * test["weekday"].astype("float32")
).astype("float32")



## === cell 27
EWR_LON, EWR_LAT = -74.1745, 40.6895
df["pickup_to_ewr"] = haversine_km(
    df["pickup_longitude"].values,
    df["pickup_latitude"].values,
    np.full(len(df), EWR_LON),
    np.full(len(df), EWR_LAT),
).astype("float32")
df["dropoff_to_ewr"] = haversine_km(
    df["dropoff_longitude"].values,
    df["dropoff_latitude"].values,
    np.full(len(df), EWR_LON),
    np.full(len(df), EWR_LAT),
).astype("float32")
test["pickup_to_ewr"] = haversine_km(
    test["pickup_longitude"].values,
    test["pickup_latitude"].values,
    np.full(len(test), EWR_LON),
    np.full(len(test), EWR_LAT),
).astype("float32")
test["dropoff_to_ewr"] = haversine_km(
    test["dropoff_longitude"].values,
    test["dropoff_latitude"].values,
    np.full(len(test), EWR_LON),
    np.full(len(test), EWR_LAT),
).astype("float32")

df["near_ewr"] = ((df["pickup_to_ewr"] < 2.0) | (df["dropoff_to_ewr"] < 2.0)).astype(
    "int8"
)
test["near_ewr"] = (
    (test["pickup_to_ewr"] < 2.0) | (test["dropoff_to_ewr"] < 2.0)
).astype("int8")



## === cell 28
dist_out = df[(df["dist"] <= 0.0) | (df["dist"] > 100.0)].index
df = df.drop(dist_out, axis=0)

test["dist"] = test["dist"].clip(lower=0.01, upper=100.0)
test["dist_miles"] = (test["dist"].astype("float32") * KM_TO_MILES).astype("float32")
test["manhattan"] = (test["abs_lon_diff"] + test["abs_lat_diff"]).astype("float32")
test["dist_x_hour"] = (test["dist_miles"] * test["hour"].astype("float32")).astype(
    "float32"
)
test["dist_x_weekday"] = (
    test["dist_miles"] * test["weekday"].astype("float32")
).astype("float32")



## === cell 29
miles = df["dist_miles"].astype("float64").values
fare = df["fare_amount"].astype("float64").values
fare_per_mile = fare / np.maximum(miles, 0.2)

bad_pricing = (
    (fare_per_mile < 1.2) | (fare_per_mile > 45.0) | ((miles < 0.2) & (fare > 25.0))
)
df = df.loc[~bad_pricing].copy()



## === cell 30
df.head()



## === cell 31
test.head()



## === cell 32
lon_bins = np.arange(-74.25, -73.70 + 1e-9, 0.01, dtype="float64")
lat_bins = np.arange(40.55, 41.00 + 1e-9, 0.01, dtype="float64")

df["pickup_lon_bin"] = pd.cut(
    df["pickup_longitude"], bins=lon_bins, labels=False, include_lowest=True
).astype("Int16")
df["pickup_lat_bin"] = pd.cut(
    df["pickup_latitude"], bins=lat_bins, labels=False, include_lowest=True
).astype("Int16")
df["dropoff_lon_bin"] = pd.cut(
    df["dropoff_longitude"], bins=lon_bins, labels=False, include_lowest=True
).astype("Int16")
df["dropoff_lat_bin"] = pd.cut(
    df["dropoff_latitude"], bins=lat_bins, labels=False, include_lowest=True
).astype("Int16")

test["pickup_lon_bin"] = pd.cut(
    test["pickup_longitude"], bins=lon_bins, labels=False, include_lowest=True
).astype("Int16")
test["pickup_lat_bin"] = pd.cut(
    test["pickup_latitude"], bins=lat_bins, labels=False, include_lowest=True
).astype("Int16")
test["dropoff_lon_bin"] = pd.cut(
    test["dropoff_longitude"], bins=lon_bins, labels=False, include_lowest=True
).astype("Int16")
test["dropoff_lat_bin"] = pd.cut(
    test["dropoff_latitude"], bins=lat_bins, labels=False, include_lowest=True
).astype("Int16")

for c in ["pickup_lon_bin", "pickup_lat_bin", "dropoff_lon_bin", "dropoff_lat_bin"]:
    df[c] = df[c].fillna(-1).astype("int16")
    test[c] = test[c].fillna(-1).astype("int16")



## === cell 33
feat = df.drop(["key", "pickup_datetime"], axis=1)
test_feat = test.drop(["key", "pickup_datetime"], axis=1)



## === cell 34
test_feat.year.unique()



## === cell 35
y = feat["fare_amount"].copy()
X_raw = feat.drop(columns=["fare_amount"]).copy()
X_test_raw = test_feat.copy()

cat_cols = ["pickup_lon_bin", "pickup_lat_bin", "dropoff_lon_bin", "dropoff_lat_bin"]
combined = pd.concat(
    [X_raw, X_test_raw],
    axis=0,
    ignore_index=True,
    sort=False,
)

combined = pd.get_dummies(combined, columns=cat_cols, prefix=cat_cols, dtype="uint8")

X = combined.iloc[: len(X_raw), :].copy()
X_test = combined.iloc[len(X_raw) :, :].copy()



## === cell 36
X.head()



## === cell 37
X_test.head()



## === cell 38
X_test = X_test.reindex(columns=X.columns)
for c in X.columns:
    if X[c].dtype == "float64":
        X[c] = X[c].astype("float32")
    if X_test[c].dtype == "float64":
        X_test[c] = X_test[c].astype("float32")



## === cell 39
xtr, xts, ytr, yts = train_test_split(X, y, test_size=0.25, random_state=42)



## === cell 40
ytr_log = np.log1p(ytr.astype("float64"))
yts_log = np.log1p(yts.astype("float64"))

xgbtrain = xgboost.DMatrix(xtr, label=ytr_log)
xgbvalid = xgboost.DMatrix(xts, label=yts_log)
xgbfinaltest = xgboost.DMatrix(X_test)



## === cell 41
params = {
    "eval_metric": "rmse",
    "objective": "reg:squarederror",
    "seed": 42,
    "max_depth": 8,
    "eta": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 10.0,
    "gamma": 0.0,
    "reg_lambda": 1.0,
}



## === cell 42
xgbmodel = xgboost.train(
    params,
    dtrain=xgbtrain,
    num_boost_round=150,
    evals=[(xgbvalid, "valid")],
    verbose_eval=False,
)



## === cell 43
pred_log = xgbmodel.predict(xgbfinaltest)



## === cell 44
pred = np.expm1(np.asarray(pred_log, dtype="float64"))
pred = np.nan_to_num(pred, nan=0.0, posinf=300.0, neginf=0.0)
pred = np.clip(pred, 0.0, 300.0)



## === cell 45
finalset = pd.DataFrame({"key": testkey, "fare_amount": pred})



## === cell 46
finalset = finalset[["key", "fare_amount"]]



## === cell 47
finalset.head()



## === cell 48
finalset.to_csv("finaloutput.csv", index=False)
print("Wrote submission:", os.path.abspath("finaloutput.csv"), "rows:", len(finalset))
