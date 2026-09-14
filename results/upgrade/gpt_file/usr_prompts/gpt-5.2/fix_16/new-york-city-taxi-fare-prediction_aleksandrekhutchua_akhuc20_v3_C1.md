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

3.12

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

3.51454

# 6. Current score

5.87862

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 5.18792) has done: 'I make a few minimal, score-relevant fixes that should reduce RMSE toward your target by correcting a bug and tightening training stability without changing the overall approach. First, your `haversine` function accidentally swaps latitude/longitude, which creates wrong distances and hurts model quality; fixing that is a direct improvement. Second, your `features2` block mistakenly still uses `features1`, so your “second feature set” experiment isn’t real; I correct it but keep your chosen final model (xgb_model3) as the one used for submission to preserve core logic. Finally, I add deterministic `random_state` for the split and XGBoost, and clip negative predictions to 0 (fares can’t be negative), which usually improves RMSE slightly without changing evaluation semantics.'
- What this solution (achieved 4.6494) has done: 'I keep your overall pipeline and XGBoost approach intact, but make a few score-relevant fixes that typically reduce RMSE for this competition. First, I remove the line that forces all zero-distance trips to have zero fare (that injects label noise because many zero-distance rows in this dataset are data errors or have nonzero fares). Second, I add two standard, minimal geospatial features (absolute lat/lon deltas) computed from existing columns—this preserves the same model class/training loop but gives XGBoost a much easier signal than raw coordinates alone. Finally, I train the final model on all cleaned training data (still the same `xgb_model3` hyperparameters) before predicting test, which usually improves generalization vs using the fold-trained model.'
- What this solution (achieved 5.49001) has done: 'Your current score (4.6494 RMSE) is worse than the target (3.51454), so we should make small, score-relevant improvements that don’t change the overall XGBoost pipeline. The biggest likely issue is label noise from remaining bad rows (especially implausible distances/speeds and outlier coordinates within the “test bounding box”), so we add a minimal, standard cleaning step based on (1) a realistic distance cap and (2) a realistic fare-per-km cap, using only already-computed `distance` and `fare_amount`. We also add one very common, lightweight feature (`manhattan_distance` as abs_lat_diff + abs_lon_diff) to help XGBoost without changing the model class or training approach. Finally, we keep training on all cleaned data and keep your submission format/paths identical.'
- What this solution (achieved 5.48621) has done: 'Your current RMSE (5.49) is worse than the target (3.51454), so we should make small, high-impact fixes that reduce label noise without changing your overall XGBoost pipeline. The biggest score drag in this competition is almost always remaining bad training rows (impossible coords, huge distances, crazy fares per km), so I add two very standard NYC-taxi cleaning filters: a reasonable minimum fare and stricter coordinate bounding using the test set bounds (still consistent with your existing approach). I also add one lightweight, metric-aligned feature (`euclidean_distance` in degrees) derived from existing columns; this keeps the same model class/training loop but gives XGBoost another clean distance proxy. Finally, I keep your final model/hyperparameters identical and still train on all cleaned data, ensuring we still write a valid `submission.csv`.'
- What this solution (achieved 4.95796) has done: 'Your current run has no Kaggle score yet, so the most likely blocker is that you’re dropping rows from `test_df` during cleaning, which produce a submission with fewer than 9914 rows and be rejected/scoreless. I keep your core XGBoost pipeline and features the same, but change the test-time cleaning to *not drop rows* (instead, we just flag invalid rows and set their predictions to a safe value). I also preserve the required submission order/row-count by starting from `sample_submission.csv` and merging predictions by `key`. These minimal changes should yield a valid submission and typically improve RMSE versus having your submission rejected.'
- What this solution (achieved 5.07296) has done: 'Your current RMSE (4.95796, lower is better) is still far above the target (3.51454), so we need a small set of high-impact, metric-aligned fixes without changing your overall XGBoost + engineered-distance-features pipeline. The most direct improvement is to stop training on raw `fare_amount` and instead train on `log1p(fare_amount)` (a standard stabilization for this competition), then invert with `expm1` at prediction time; this keeps the same model class, loop, and features but usually reduces RMSE materially by handling heavy-tailed fares. I also tighten consistency by applying the same “bad row” logic you already use at test time to the training data (only for the relevant subset), so the model doesn’t learn from patterns you later treat as invalid. Finally, I keep your submission alignment-by-key logic intact and still clip negative fares to 0.'
- What this solution (achieved 5.03559) has done: 'Your current RMSE (5.07296) is worse than the target (3.51454), so we should make small, high-impact fixes that improve signal quality without changing your XGBoost pipeline. The main issue is that you train on `log1p(fare_amount)` but you evaluate/print RMSE in log-space, so you’re not tuning anything toward the real Kaggle metric; I change validation RMSE reporting to be computed back in dollars (expm1), leaving the training target unchanged. Then I make a minimal, standard label-noise reduction by filtering unrealistically low “fare per km” trips (in addition to your existing high cap), which usually helps this competition a lot with minimal logic changes. Finally, I keep your final model hyperparameters and submission-building logic the same, only ensuring predictions stay sane (non-negative) as you already do.'
- What this solution (achieved 5.02601) has done: 'Your current RMSE (5.03559) is still far above the target (3.51454), so we make a small, metric-aligned improvement without changing your overall XGBoost-on-engineered-features approach. The biggest direct issue in your pipeline is that you train on `log1p(fare_amount)` but then invert with `expm1` without any bias correction; because `E[exp(Y)] != exp(E[Y])`, this systematically under/over-shoots in dollar space and often hurts RMSE. I add a minimal “smearing” (bias-correction) factor computed on the validation set residuals in log-space and apply it to both validation scoring and test predictions; this keeps the same features/model/training loop and only changes the post-processing to better match the RMSE-in-dollars metric. I also clip fares to a reasonable upper bound (consistent with your `<100` training cap) to prevent a few extreme predictions from dominating RMSE.'
- What this solution (achieved 5.70329) has done: 'Your current RMSE (5.02601) is still well above the target (3.51454), so we should make one or two high-impact, low-risk changes that keep your XGBoost+engineered-distance pipeline intact. The biggest score drag I see is that your final test-time logic *replaces predictions* for any row outside a broad NYC box (or with near-zero coords) with a constant median fare; that’s likely hurting many perfectly valid test rows and inflating RMSE. I keep the same cleaning for training, the same features, the same log1p target, and the same XGBRegressor hyperparameters, but I stop overriding “invalid” test rows with a constant and instead let the trained model predict for all test rows (with simple coordinate clipping into the training/test-derived bounds to avoid extreme out-of-domain values). I also compute the smearing factor using out-of-fold-style predictions from the same data split (still your same approach) and apply it consistently to test predictions.'
- What this solution (achieved 5.6757) has done: 'Your current RMSE (5.70329, lower is better) is still far above the target (3.51454), so we need a small, high-impact fix that improves generalization without changing your core XGBoost + engineered-distance-features approach. The most likely score drag is that you train on 10M rows but don’t add the single most standard NYC-taxi feature: distance to key landmarks (JFK, LGA, EWR, Manhattan), which is still “feature extraction” and keeps the same model and training loop. I add these few landmark-distance features (computed via your existing haversine math) for both train/test and include them in `features1`, leaving your cleaning, log1p target, smearing, and XGB hyperparameters unchanged. I also ensure those new features are computed once (not inconsistently re-computed only at test time) to keep train/test feature parity.'
- What this solution (achieved 5.6757) has done: 'Your RMSE (5.6757) is still far worse than the target (3.51454), so we should make one minimal, high-impact, metric-aligned fix rather than changing the overall XGBoost/log1p pipeline. The biggest issue in your current code is that you compute `distance` and landmark distances for `test_df`, then later you **recompute** them in the final cell after coordinate clipping, but you **do not recompute** them for `train_df` after applying the NYC-box cleaning—this creates subtle train/test feature distribution mismatch (train uses pre-box distance values, test uses post-clip/post-recompute), which hurts generalization. I make train/test feature computation strictly consistent by (1) computing all engineered features in one shared function and (2) after the final cleaning steps for train, recomputing the same engineered columns for train as well (without changing model, loss, or training loop). This should move RMSE downward toward your target while keeping the core logic intact and still producing a valid `submission.csv`.'
- What this solution (achieved 5.9902) has done: 'The timeout is dominated by repeatedly materializing large boolean-indexed DataFrames, computing engineered features multiple times (including landmark distances in a Python loop), and fitting many XGBoost models on a multi‑million row dataset. I keep the same data filters, feature set, models, and smearing evaluation semantics, but eliminate redundant computations, compute all engineered features in a single vectorized pass (including landmark distances), and avoid creating extra copies/slices where not needed. I also reduce overhead from EDA plotting/printing (which doesn’t affect the trained model or submission) while keeping the training/prediction logic identical. Finally, I ensure features are computed exactly once per dataset state and reuse the resulting matrices for all model fits.'
- What this solution (achieved 5.87862) has done: 'Your current RMSE (5.9902) is far above the target (3.51454), so we need a small, score-relevant fix rather than further speed tweaks. The biggest issue in your final cell is that you clip test coordinates *after* feature engineering but never recompute `distance`/landmark distances, so the model is predicting with stale engineered features that no longer match the clipped coordinates. I make that clipping happen before feature creation and ensure we (re)compute engineered features exactly once on the final test coordinates, keeping the same model, log1p target, smearing, and hyperparameters. I also apply the same clip bounds derived from the (already-cleaned) training data to avoid compressing valid test variation using the test set’s own min/max.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

os.environ.setdefault("PYTHONHASHSEED", "0")
np.random.seed(42)



## === cell 1
train_path = "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv"
test_path = "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv"

dtypes_train = {
    "key": "string",
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int8",
}
dtypes_test = {
    "key": "string",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int8",
}

train_df = pd.read_csv(
    train_path,
    nrows=10_000_000,
    dtype=dtypes_train,
    parse_dates=["pickup_datetime"],
)
test_df = pd.read_csv(
    test_path,
    dtype=dtypes_test,
    parse_dates=["pickup_datetime"],
)



## === cell 2
_ = (train_df.shape, test_df.shape)



## === cell 3
_ = None



## === cell 4
_ = None



## === cell 5
_ = None



## === cell 6
train_df.dropna(inplace=True)



## === cell 7
_ = int((train_df["fare_amount"] < 0).sum())



## === cell 8
train_df = train_df[train_df["fare_amount"] >= 0]



## === cell 9
_ = int((train_df["fare_amount"] == 0).sum())



## === cell 10
_ = None



## === cell 11
_ = None



## === cell 12
mask_bad_coords = (
    (train_df["pickup_latitude"].lt(-90) | train_df["pickup_latitude"].gt(90))
    | (train_df["pickup_longitude"].lt(-180) | train_df["pickup_longitude"].gt(180))
    | (train_df["dropoff_latitude"].lt(-90) | train_df["dropoff_latitude"].gt(90))
    | (train_df["dropoff_longitude"].lt(-180) | train_df["dropoff_longitude"].gt(180))
)
_ = int(mask_bad_coords.sum())



## === cell 13
train_df = train_df[~mask_bad_coords]



## === cell 14
_ = None



## === cell 15
_ = None



## === cell 16
_ = None



## === cell 17
_ = None



## === cell 18
mask_bad_pax = (train_df["passenger_count"] <= 0) | (train_df["passenger_count"] > 6)
_ = int(mask_bad_pax.sum())



## === cell 19
train_df = train_df[~mask_bad_pax]



## === cell 20
_ = None



## === cell 21
_ = None



## === cell 22
import numpy as np




## === cell 23
def haversine(df: pd.DataFrame) -> None:
    lon1 = df["pickup_longitude"].to_numpy(dtype=np.float64, copy=False)
    lat1 = df["pickup_latitude"].to_numpy(dtype=np.float64, copy=False)
    lon2 = df["dropoff_longitude"].to_numpy(dtype=np.float64, copy=False)
    lat2 = df["dropoff_latitude"].to_numpy(dtype=np.float64, copy=False)

    lon1r = np.radians(lon1)
    lat1r = np.radians(lat1)
    lon2r = np.radians(lon2)
    lat2r = np.radians(lat2)

    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = np.sin(dlat / 2.0) ** 2 + (
        np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2.0) ** 2
    )
    df["distance"] = (6371.0 * 2.0 * np.arcsin(np.sqrt(a))).astype(np.float32)




## === cell 24
haversine(train_df)
haversine(test_df)



## === cell 25
_ = int((train_df["distance"] == 0).sum())



## === cell 26
_ = int((test_df["distance"] == 0).sum())



## === cell 27
_ = None



## === cell 28
mask1 = train_df["distance"] > 0
mask2 = train_df["fare_amount"] == 0
_ = int((mask1 & mask2).sum())



## === cell 29
train_df = train_df[~(mask1 & mask2)]



## === cell 30
nyc_east_long = -73.699710
nyc_west_long = -74.256890
nyc_north_lat = 40.921678
nyc_south_lat = 40.496160



## === cell 31
mask1 = (test_df["pickup_longitude"] <= nyc_east_long) & (
    test_df["pickup_longitude"] >= nyc_west_long
)
mask2 = (test_df["pickup_latitude"] <= nyc_north_lat) & (
    test_df["pickup_latitude"] >= nyc_south_lat
)
mask3 = (test_df["dropoff_longitude"] <= nyc_east_long) & (
    test_df["dropoff_longitude"] >= nyc_west_long
)
mask4 = (test_df["dropoff_latitude"] <= nyc_north_lat) & (
    test_df["dropoff_latitude"] >= nyc_south_lat
)
mask5 = test_df["distance"] != 0
test_df_not_nyc = test_df[~(mask1 & mask2 & mask3 & mask4) & mask5]
_ = test_df_not_nyc.shape



## === cell 32
tst_east_long = float(
    max(test_df["pickup_longitude"].max(), test_df["dropoff_longitude"].max())
)
tst_west_long = float(
    min(test_df["pickup_longitude"].min(), test_df["dropoff_longitude"].min())
)
tst_north_lat = float(
    max(test_df["pickup_latitude"].max(), test_df["dropoff_latitude"].max())
)
tst_south_lat = float(
    min(test_df["pickup_latitude"].min(), test_df["dropoff_latitude"].min())
)
_ = (tst_east_long, tst_west_long, tst_north_lat, tst_south_lat)



## === cell 33
mask1 = (train_df["pickup_longitude"] <= tst_east_long) & (
    train_df["pickup_longitude"] >= tst_west_long
)
mask2 = (train_df["pickup_latitude"] <= tst_north_lat) & (
    train_df["pickup_latitude"] >= tst_south_lat
)
mask3 = (train_df["dropoff_longitude"] <= tst_east_long) & (
    train_df["dropoff_longitude"] >= tst_west_long
)
mask4 = (train_df["dropoff_latitude"] <= tst_north_lat) & (
    train_df["dropoff_latitude"] >= tst_south_lat
)
mask5 = train_df["distance"] != 0
_ = int((~(mask1 & mask2 & mask3 & mask4) & mask5).sum())



## === cell 34
train_df = train_df[(mask1 & mask2 & mask3 & mask4) | ~mask5]



## === cell 35
train_df["pickup_datetime"] = pd.to_datetime(
    train_df["pickup_datetime"], errors="coerce"
)
test_df["pickup_datetime"] = pd.to_datetime(test_df["pickup_datetime"], errors="coerce")



## === cell 36
dt = train_df["pickup_datetime"]
train_df["year"] = dt.dt.year.astype(np.int16)
train_df["month"] = dt.dt.month.astype(np.int8)
train_df["weekday"] = dt.dt.dayofweek.astype(np.int8)
train_df["hour"] = dt.dt.hour.astype(np.int8)



## === cell 37
dt = test_df["pickup_datetime"]
test_df["year"] = dt.dt.year.astype(np.int16)
test_df["month"] = dt.dt.month.astype(np.int8)
test_df["weekday"] = dt.dt.dayofweek.astype(np.int8)
test_df["hour"] = dt.dt.hour.astype(np.int8)



## === cell 38
_ = None



## === cell 39
plt = None
sns = None



## === cell 40
_ = None



## === cell 41
_ = None



## === cell 42
_ = None



## === cell 43
_ = None



## === cell 44
expencives = train_df[train_df["fare_amount"] >= 100]
_ = float(len(expencives) / len(train_df) * 100.0) if len(train_df) else 0.0



## === cell 45
_ = None



## === cell 46
_ = None



## === cell 47
_ = None



## === cell 48
_ = None



## === cell 49
_ = None



## === cell 50
_ = None



## === cell 51
_ = None



## === cell 52
_ = None



## === cell 53
_ = None



## === cell 54
_ = None



## === cell 55
_ = None



## === cell 56
_ = None



## === cell 57
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


def _remove_zero_coords(df):
    eps = 1e-6
    bad = (
        (df["pickup_longitude"].abs() < eps)
        | (df["pickup_latitude"].abs() < eps)
        | (df["dropoff_longitude"].abs() < eps)
        | (df["dropoff_latitude"].abs() < eps)
    )
    return df[~bad]


train_df = _remove_zero_coords(train_df)

nyc_west_long2, nyc_east_long2 = -74.5, -72.8
nyc_south_lat2, nyc_north_lat2 = 40.3, 41.3


def _in_nyc_box(df):
    return (
        (df["pickup_longitude"].between(nyc_west_long2, nyc_east_long2))
        & (df["dropoff_longitude"].between(nyc_west_long2, nyc_east_long2))
        & (df["pickup_latitude"].between(nyc_south_lat2, nyc_north_lat2))
        & (df["dropoff_latitude"].between(nyc_south_lat2, nyc_north_lat2))
    )


train_df = train_df[_in_nyc_box(train_df)]


def _haversine_km_from_points(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    a = np.sin((lat2 - lat1) / 2.0) ** 2 + (
        np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2.0) ** 2
    )
    return 6371 * 2 * np.arcsin(np.sqrt(a))


LANDMARKS = {
    "manhattan": (-73.985130, 40.758896),
    "jfk": (-73.778139, 40.641312),
    "lga": (-73.874000, 40.776927),
    "ewr": (-74.174500, 40.689500),
}

_landmark_names = list(LANDMARKS.keys())
_landmark_lon = np.array([LANDMARKS[n][0] for n in _landmark_names], dtype=np.float64)
_landmark_lat = np.array([LANDMARKS[n][1] for n in _landmark_names], dtype=np.float64)


def _add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    haversine(df)  # defines df["distance"]

    plon = df["pickup_longitude"].to_numpy(dtype=np.float64, copy=False)
    plat = df["pickup_latitude"].to_numpy(dtype=np.float64, copy=False)
    dlon = df["dropoff_longitude"].to_numpy(dtype=np.float64, copy=False)
    dlat = df["dropoff_latitude"].to_numpy(dtype=np.float64, copy=False)

    abs_lon = np.abs(plon - dlon)
    abs_lat = np.abs(plat - dlat)

    df["abs_lon_diff"] = abs_lon.astype(np.float32)
    df["abs_lat_diff"] = abs_lat.astype(np.float32)
    manh = abs_lon + abs_lat
    df["manhattan_distance"] = manh.astype(np.float32)
    df["euclidean_distance"] = np.sqrt(abs_lon * abs_lon + abs_lat * abs_lat).astype(
        np.float32
    )
    df["log1p_distance"] = np.log1p(
        df["distance"].to_numpy(dtype=np.float64, copy=False)
    ).astype(np.float32)

    plonr = np.radians(plon)[:, None]
    platr = np.radians(plat)[:, None]
    dlonr = np.radians(dlon)[:, None]
    dlatr = np.radians(dlat)[:, None]

    llonr = np.radians(_landmark_lon)[None, :]
    llatr = np.radians(_landmark_lat)[None, :]

    def _haversine_mat(lon1r, lat1r, lon2r, lat2r):
        dlat = lat2r - lat1r
        dlon = lon2r - lon1r
        a = np.sin(dlat / 2.0) ** 2 + (
            np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2.0) ** 2
        )
        return 6371.0 * 2.0 * np.arcsin(np.sqrt(a))

    pick_d = _haversine_mat(plonr, platr, llonr, llatr).astype(np.float32)
    drop_d = _haversine_mat(dlonr, dlatr, llonr, llatr).astype(np.float32)

    for j, name in enumerate(_landmark_names):
        df[f"pickup_dist_{name}"] = pick_d[:, j]
        df[f"dropoff_dist_{name}"] = drop_d[:, j]

    return df


_add_engineered_features(train_df)
_add_engineered_features(test_df)

train_df = train_df[train_df["fare_amount"] >= 2.5]
train_df = train_df[(train_df["distance"] > 0) & (train_df["distance"] <= 80)]

fare_per_km = train_df["fare_amount"] / (train_df["distance"] + 1e-3)
train_df = train_df[(fare_per_km <= 100) & (fare_per_km >= 0.5)]

eps = 1e-6
bad_zero_tr = (
    (train_df["pickup_longitude"].abs() < eps)
    | (train_df["pickup_latitude"].abs() < eps)
    | (train_df["dropoff_longitude"].abs() < eps)
    | (train_df["dropoff_latitude"].abs() < eps)
)
bad_box_tr = ~_in_nyc_box(train_df)
train_df = train_df[~(bad_zero_tr | bad_box_tr)]

_add_engineered_features(train_df)

features1 = [
    "distance",
    "log1p_distance",
    "abs_lon_diff",
    "abs_lat_diff",
    "manhattan_distance",
    "euclidean_distance",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
    "year",
    "month",
    "weekday",
    "hour",
    "pickup_dist_manhattan",
    "dropoff_dist_manhattan",
    "pickup_dist_jfk",
    "dropoff_dist_jfk",
    "pickup_dist_lga",
    "dropoff_dist_lga",
    "pickup_dist_ewr",
    "dropoff_dist_ewr",
]
target = "fare_amount"

X1 = train_df[features1].to_numpy(copy=False)
y1 = np.log1p(train_df[target].to_numpy(dtype=np.float64, copy=False))

X_train1, X_val1, y_train1, y_val1 = train_test_split(
    X1, y1, test_size=0.2, random_state=42
)




## === cell 58
def _smearing_factor(y_true_log, y_pred_log):
    resid = y_true_log - y_pred_log
    resid = resid[np.isfinite(resid)]
    if resid.size == 0:
        return 1.0
    return float(np.mean(np.exp(resid)))


def test_model(model, X_train, y_train, X_val, y_val):
    model.fit(X_train, y_train)

    y_pred_val_log = model.predict(X_val)
    y_pred_train_log = model.predict(X_train)

    smear = _smearing_factor(y_val, y_pred_val_log)

    y_pred_val = np.expm1(y_pred_val_log) * smear
    y_pred_train = np.expm1(y_pred_train_log) * smear
    y_val_true = np.expm1(y_val)
    y_train_true = np.expm1(y_train)

    y_pred_val = np.clip(y_pred_val, 0, np.inf)
    y_pred_train = np.clip(y_pred_train, 0, np.inf)

    rmse_val = np.sqrt(mean_squared_error(y_val_true, y_pred_val))
    rmse_train = np.sqrt(mean_squared_error(y_train_true, y_pred_train))
    print(f"{type(model).__name__}: smearing_factor={smear:.6f}")
    print(f"{type(model).__name__}: RMSE($) on validation set: {rmse_val}")
    print(f"{type(model).__name__}: RMSE($) on train set: {rmse_train}")




## === cell 59
from sklearn.linear_model import LinearRegression

lr_model = LinearRegression()
test_model(lr_model, X_train1, y_train1, X_val1, y_val1)



## === cell 60
from xgboost import XGBRegressor

xgb_model1 = XGBRegressor(random_state=42, n_jobs=-1)
test_model(xgb_model1, X_train1, y_train1, X_val1, y_val1)



## === cell 61
xgb_model2 = XGBRegressor(
    n_estimators=100, learning_rate=0.2, max_depth=4, random_state=42, n_jobs=-1
)
test_model(xgb_model2, X_train1, y_train1, X_val1, y_val1)



## === cell 62
xgb_model3 = XGBRegressor(
    n_estimators=100, learning_rate=0.1, max_depth=5, n_jobs=-1, random_state=42
)
test_model(xgb_model3, X_train1, y_train1, X_val1, y_val1)



## === cell 63
features2 = ["distance", "pickup_longitude", "pickup_latitude", "year", "month", "hour"]

X2 = train_df[features2].to_numpy(copy=False)
y2 = np.log1p(train_df[target].to_numpy(dtype=np.float64, copy=False))

X_train2, X_val2, y_train2, y_val2 = train_test_split(
    X2, y2, test_size=0.2, random_state=42
)



## === cell 64
xgb_model4 = XGBRegressor(
    n_estimators=120,
    learning_rate=0.05,
    max_depth=6,
    n_jobs=-1,
    min_child_weight=3,
    random_state=42,
)
test_model(xgb_model4, X_train2, y_train2, X_val2, y_val2)



## === cell 65
final_model = XGBRegressor(
    n_estimators=100, learning_rate=0.1, max_depth=5, n_jobs=-1, random_state=42
)

_cal_model = XGBRegressor(
    n_estimators=100, learning_rate=0.1, max_depth=5, n_jobs=-1, random_state=42
)

_cal_model.fit(X_train1, y_train1)
val_pred_log = _cal_model.predict(X_val1)
smear_final = _smearing_factor(y_val1, val_pred_log)
print("Final smearing factor to use for test predictions:", smear_final)

final_model.fit(X1, y1)

margin = 1e-6
for c in ["pickup_longitude", "dropoff_longitude"]:
    lo = float(train_df[c].min()) - margin
    hi = float(train_df[c].max()) + margin
    test_df[c] = test_df[c].clip(lo, hi)

for c in ["pickup_latitude", "dropoff_latitude"]:
    lo = float(train_df[c].min()) - margin
    hi = float(train_df[c].max()) + margin
    test_df[c] = test_df[c].clip(lo, hi)

_add_engineered_features(test_df)

test_pred_all = test_df[features1].to_numpy(copy=False)

pred_test_log = final_model.predict(test_pred_all)
pred_all = np.expm1(pred_test_log) * smear_final
pred_all = np.clip(pred_all, 0, np.inf)

sample_sub = pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/sample_submission.csv",
    dtype={"key": "string", "fare_amount": "float32"},
)

pred_map = pd.DataFrame(
    {
        "key": test_df["key"].to_numpy(copy=False),
        "fare_amount": pred_all.astype(np.float32),
    }
)
submission = sample_sub[["key"]].merge(pred_map, on="key", how="left")

median_fare = float(np.expm1(np.median(y1)))
submission["fare_amount"] = submission["fare_amount"].fillna(median_fare)

submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print("Any NaNs in submission:", submission.isna().any().to_dict())
print("Submission head:\n", submission.head())
