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

No external packages required in the script and installed.

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

4.13939939564504

# 6. Current score

5.37555

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 10.67607) has done: 'I fix the TensorFlow/Keras 3 compatibility errors by replacing the deprecated `tf1.layers.dense` calls with `tf1.keras.layers.Dense` while keeping the same architecture (same number of layers/neurons, ReLU, and He initialization). I also remove the TF1 protobuf-related import crash by avoiding importing TensorFlow at the very start and by using TF1-compat mode only after data loading. Finally, I ensure the graph is built successfully so `init` and `y_test` exist, and the script always writes a valid `submission.csv` with the required `key,fare_amount` columns.'
- What this solution (achieved 9.96593) has done: 'I fix the TensorFlow import crash (`MessageFactory`/protobuf mismatch) by avoiding TensorFlow entirely and switching the training/inference to a lightweight NumPy linear regression that uses the exact same features and scaling already computed. This keeps the overall semantics (regression on the engineered/scaled inputs) while making the notebook stable in the Kaggle Python 3.7 environment and ensuring a `submission.csv` is always written. To move RMSE down toward your target, I also compute `mu`/`sigma` from the actual loaded training sample instead of using the hardcoded arrays (which likely mismatch your sampled train distribution), and use a proper train/valid shuffle split to prevent ordering bias. The rest of the pipeline (date features, feature list, clipping non-negative fares, submission format) stays the same.'
- What this solution (achieved 1066.88915) has done: 'Your current RMSE (9.96593) is far above the target (4.1394), so we need a real but still minimal modeling improvement without changing the overall approach (feature engineering + scaling + simple regression). The biggest win for this competition with your exact setup is to add a single strong engineered feature: great-circle (Haversine) distance between pickup/dropoff, while keeping the same linear ridge closed-form model. I also apply the same basic coordinate bounding-box cleaning used widely for this dataset (NYC-area filtering) to reduce label noise in the sampled training rows, which typically drops RMSE substantially. Everything else (data paths, date features, scaling, ridge solver, clipping, and submission format) stays the same and still writes a valid `submission.csv`.'
- What this solution (achieved 36.71466) has done: 'Your RMSE is catastrophically worse than the target, which strongly suggests a correctness bug rather than a modeling gap. The biggest issue is that your `y_test` predictions are being clipped at 0, but taxi fares in this dataset should never be near 0 and the competition expects realistic positive fares; more importantly, your model can produce negative values due to linear extrapolation, and clipping to 0 creates huge errors for many cases. I keep your exact pipeline (same features, haversine, scaling, ridge closed-form) and make only minimal, metric-aligned post-processing: clip predictions to a realistic fare range (e.g., [2.5, 250]) instead of [0, inf), matching the training filter and the known minimum fare behavior. I also ensure `passenger_count` is constrained to a reasonable range in both train/test (1–6) to reduce extreme outliers without changing the model.'
- What this solution (achieved 5.88226) has done: 'Your current RMSE (36.7) is far worse than the target (4.14), which strongly indicates a correctness issue rather than a small modeling gap. The biggest likely culprit is feature scale mismatch between train and test caused by out-of-range coordinates and impossible distances in `test.csv` (the model is trained only on NYC-bounded coordinates but the test set is not filtered/cleaned similarly), which can explode linear predictions and RMSE. I keep your exact model/feature set/training approach, but apply the same NYC bounding-box cleaning to the test data (and additionally cap extreme `haversine_km` values) so the test distribution matches what the model learned. Finally, I add a safe fallback to median prediction for any remaining invalid rows to guarantee stable, realistic outputs without changing core semantics.'
- What this solution (achieved 5.5339) has done: 'Your current RMSE (5.882) is worse than the target (4.139), so we should improve the model slightly without changing the overall pipeline (same feature set, same scaling, same ridge closed-form solver, same clipping/submission format). The biggest low-risk gain here is to tune the ridge regularization strength `alpha` using your existing validation split, then refit once on the full filtered training data with the chosen `alpha` before predicting test—this keeps the exact same model class and training approach. I also cap `haversine_km` in test based on the train distribution (99.9th percentile) rather than a fixed 200 km, to better match feature scaling and reduce extreme extrapolation. All other logic (NYC bounds handling, fallback for out-of-bounds, and fare clipping) is preserved.'
- What this solution (achieved 5.55016) has done: 'Your current RMSE (5.5339) is still above the target (4.1394), so we should make small, low-risk improvements that keep the same model class (closed-form ridge on the same core feature set) while reducing obvious noise/outliers. I (1) read a larger but still manageable training sample and use `skiprows` with a fixed RNG to get a more representative subset than the “first N rows”, and (2) tighten common data cleaning by removing zero-distance rides and extreme `haversine_km` outliers consistently. I also modestly refine the ridge `alpha` search grid (still ridge, still closed-form) to better match the validation optimum without changing the training approach. Submission writing and column alignment remain unchanged.'
- What this solution (achieved 24.20806) has done: 'We’re currently worse than the target RMSE (5.55 vs 4.14, lower is better), so the smallest likely improvement is to reduce remaining label noise in the training sample without changing your model class (same closed-form ridge) or feature set. I add two very common NYC Taxi sanity filters that remove extreme/invalid rides: (1) remove rows with zero/near-zero coordinates and (2) drop rows with unrealistically high implied speed (using your existing haversine + datetime elapsed), which typically helps linear models a lot. I also add one minimal feature that doesn’t change the approach (still linear ridge on engineered/scaled inputs): a “manhattan distance” approximation (|Δlat|+|Δlon|) which is strongly predictive for NYC grid travel and usually improves RMSE for linear models. Everything else (date features, haversine, scaling, ridge alpha tuning, test fallback, clipping, and submission format) remains intact.'
- What this solution (achieved 5.54981) has done: 'Your current RMSE (24.208) is far worse than the target (4.139, lower is better), which strongly suggests a correctness bug rather than a small modeling limitation. The main issue is the “speed_kmh” filter: it mistakenly derives trip duration from the absolute pickup timestamp relative to the dataset minimum, not from trip duration (which isn’t available), so it incorrectly drops a huge number of valid rows and can leave you training on a biased/garbage subset. I remove that invalid speed filter (keep your other sanity filters), and I also apply the same manhattan/haversine capping to test plus a simple “missing/invalid feature” safeguard to prevent NaNs/Infs from blowing up predictions. The model (closed-form ridge with the same features, scaling, alpha tuning, and clipping/submission format) remains unchanged.'
- What this solution (achieved 5.5587) has done: 'We’re still worse than the target RMSE (5.54981 vs 4.1394; lower is better), so the safest way to move toward the target without changing your model class is to reduce training-label noise and align train/test preprocessing more tightly. I make three minimal, high-impact changes: (1) fix the train sampling so it actually reads ~N_TRAIN rows (your current `skiprows=set(...)` is extremely memory-heavy and can behave unpredictably), (2) add a standard “outlier fare vs distance” filter using your existing haversine feature (keeps same features/model, just removes obvious bad labels), and (3) add a missing-value guard for `pickup_datetime`-derived columns in test/train so NaT rows don’t create NaNs that destabilize predictions. The ridge model, features, scaling, alpha tuning, clipping, and submission format remain the same.'
- What this solution (achieved 5.55753) has done: 'Your current RMSE (5.5587) is worse than the target (4.1394), so we should make a small, low-risk improvement that keeps the same ridge-regression core but reduces train-sample bias and label noise. The biggest issue is that you always read the *first* 1.2M rows of `train.csv`, which are not a representative sample and often hurt generalization; we switch to a deterministic random row sample using `skiprows` without changing the model/feature set. We also add two very standard, minimal sanity filters (reasonable coordinate deltas and a loose fare-vs-distance envelope) that remove obvious bad labels while preserving core semantics. Everything else (feature engineering, scaling, ridge closed-form + alpha tuning, fallback for out-of-bounds test rows, and submission writing) stays the same.'
- What this solution (achieved 5.37456) has done: 'Your current RMSE (5.55753) is still above the target (4.1394, lower is better), so the smallest likely step toward the target without changing the core “engineered features + scaling + closed-form ridge” approach is to reduce remaining label noise and improve generalization via slightly better feature expressiveness. I keep the same ridge solver and overall pipeline, but (1) add two standard, lightweight features that are strongly predictive for NYC fares (absolute deltas in lat/lon, and a simple “airport-ish” indicator from coordinate proximity), and (2) make the ridge alpha selection a bit more precise around the current best by doing a tiny second-stage local search around the best alpha (still the same validation procedure). I also ensure train/test preprocessing remains aligned for these new features (same caps computed from train, applied to test), and keep the same submission writing logic.'
- What this solution (achieved 5.37555) has done: 'Your current RMSE (5.37456) is worse than the target (4.1394), so we should make a small, low-risk improvement that keeps the same ridge-regression core but improves feature expressiveness and train/test alignment. The most impactful minimal change is to add a single additional engineered feature: a coarse bearing (direction) from pickup to dropoff, which is known to help linear models in NYC due to one-way grids/bridges. We compute it for both train and test, include it in `feature_cols`, and apply the same outlier capping derived from the training distribution to avoid destabilizing extrapolation. Everything else (sampling, cleaning, scaling, ridge closed-form with alpha tuning, bounds fallback, clipping, and submission writing) stays the same.'

# 9. Code solution

## === cell 0
import os
import re
import numpy as np
import pandas as pd

os.environ["PYTHONHASHSEED"] = "0"
np.random.seed(0)

print("Listing ../input:")
print(os.listdir("../input"))



## === cell 1
DATA_DIR = "../input/new-york-city-taxi-fare-prediction"
print("Data dir exists:", os.path.exists(DATA_DIR))
print("Files:", [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")])



## === cell 2
df_test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
print(df_test.shape)
print(df_test.columns.tolist())




## === cell 3
def add_datepart(df, fldname, drop=True):
    """
    Create datetime-derived columns; compatible with pandas versions where .dt.week is removed.
    """
    fld = df[fldname]
    if not np.issubdtype(fld.dtype, np.datetime64):
        df[fldname] = fld = pd.to_datetime(
            fld, infer_datetime_format=True, errors="coerce"
        )

    targ_pre = re.sub("[Dd]ate$", "", fldname)

    df[targ_pre + "Year"] = fld.dt.year
    df[targ_pre + "Month"] = fld.dt.month
    try:
        df[targ_pre + "Week"] = fld.dt.isocalendar().week.astype(np.int16)
    except Exception:
        df[targ_pre + "Week"] = fld.dt.strftime("%V").astype(np.int16)

    df[targ_pre + "Day"] = fld.dt.day
    df[targ_pre + "Dayofweek"] = fld.dt.dayofweek
    df[targ_pre + "Dayofyear"] = fld.dt.dayofyear
    df[targ_pre + "hour"] = fld.dt.hour

    df[targ_pre + "Is_month_end"] = fld.dt.is_month_end.astype(np.int8)
    df[targ_pre + "Is_month_start"] = fld.dt.is_month_start.astype(np.int8)
    df[targ_pre + "Is_quarter_end"] = fld.dt.is_quarter_end.astype(np.int8)
    df[targ_pre + "Is_quarter_start"] = fld.dt.is_quarter_start.astype(np.int8)
    df[targ_pre + "Is_year_end"] = fld.dt.is_year_end.astype(np.int8)
    df[targ_pre + "Is_year_start"] = fld.dt.is_year_start.astype(np.int8)

    try:
        elapsed = (fld.view("int64") // 10**9).astype(np.int64)
    except Exception:
        elapsed = (fld.astype("int64") // 10**9).astype(np.int64)
    df[targ_pre + "Elapsed"] = elapsed

    if drop:
        df.drop(fldname, axis=1, inplace=True)




## === cell 4
def haversine_km(lon1, lat1, lon2, lat2):
    lon1 = np.radians(lon1.astype(np.float64))
    lat1 = np.radians(lat1.astype(np.float64))
    lon2 = np.radians(lon2.astype(np.float64))
    lat2 = np.radians(lat2.astype(np.float64))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * (
        np.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * np.arcsin(np.sqrt(a))
    r = 6371.0
    return (r * c).astype(np.float32)


def add_bearing_feature(df):
    """
    Minimal, metric-relevant improvement: add a coarse trip direction feature.
    Using sin/cos of bearing is common; we add just sin(bearing) to keep changes minimal.
    """
    lon1 = np.radians(df["pickup_longitude"].astype(np.float64).values)
    lat1 = np.radians(df["pickup_latitude"].astype(np.float64).values)
    lon2 = np.radians(df["dropoff_longitude"].astype(np.float64).values)
    lat2 = np.radians(df["dropoff_latitude"].astype(np.float64).values)

    dlon = lon2 - lon1
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    bearing = np.arctan2(y, x)  # [-pi, pi]
    df["bearing_sin"] = np.sin(bearing).astype(np.float32)


def add_geo_features(df):
    df["abs_dlon"] = (
        (df["pickup_longitude"] - df["dropoff_longitude"]).abs().astype(np.float32)
    )
    df["abs_dlat"] = (
        (df["pickup_latitude"] - df["dropoff_latitude"]).abs().astype(np.float32)
    )

    pu = (
        (
            (df["pickup_longitude"].between(-73.90, -73.75))
            & (df["pickup_latitude"].between(40.63, 40.68))
        )
        | (
            (df["pickup_longitude"].between(-73.90, -73.85))
            & (df["pickup_latitude"].between(40.76, 40.78))
        )
        | (
            (df["pickup_longitude"].between(-74.21, -74.15))
            & (df["pickup_latitude"].between(40.67, 40.71))
        )
    )
    do = (
        (
            (df["dropoff_longitude"].between(-73.90, -73.75))
            & (df["dropoff_latitude"].between(40.63, 40.68))
        )
        | (
            (df["dropoff_longitude"].between(-73.90, -73.85))
            & (df["dropoff_latitude"].between(40.76, 40.78))
        )
        | (
            (df["dropoff_longitude"].between(-74.21, -74.15))
            & (df["dropoff_latitude"].between(40.67, 40.71))
        )
    )
    df["airport_trip"] = (pu | do).astype(np.int8)


add_datepart(df_test, "pickup_datetime", drop=True)

df_test["haversine_km"] = haversine_km(
    df_test["pickup_longitude"].values,
    df_test["pickup_latitude"].values,
    df_test["dropoff_longitude"].values,
    df_test["dropoff_latitude"].values,
)
df_test["manhattan"] = (
    (df_test["pickup_longitude"] - df_test["dropoff_longitude"]).abs()
    + (df_test["pickup_latitude"] - df_test["dropoff_latitude"]).abs()
).astype(np.float32)

add_geo_features(df_test)

add_bearing_feature(df_test)

feature_cols = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
    "pickup_datetimeYear",
    "pickup_datetimeMonth",
    "pickup_datetimeWeek",
    "pickup_datetimeDay",
    "pickup_datetimeDayofweek",
    "pickup_datetimeDayofyear",
    "pickup_datetimehour",
    "pickup_datetimeElapsed",
    "haversine_km",
    "manhattan",
    "abs_dlon",
    "abs_dlat",
    "airport_trip",
    "bearing_sin",
]

df_test["passenger_count"] = df_test["passenger_count"].clip(1, 6)

test_nyc_bounds = (
    (df_test["pickup_longitude"].between(-75, -72))
    & (df_test["dropoff_longitude"].between(-75, -72))
    & (df_test["pickup_latitude"].between(40, 42))
    & (df_test["dropoff_latitude"].between(40, 42))
)
print("Test rows within NYC bounds:", int(test_nyc_bounds.sum()), "/", len(df_test))

for c in [col for col in feature_cols if col.startswith("pickup_datetime")]:
    if c in df_test.columns:
        df_test[c] = df_test[c].fillna(0)

missing = [c for c in feature_cols if c not in df_test.columns]
print("Missing test columns:", missing)
print(df_test[feature_cols].head())



## === cell 5
train_path = os.path.join(DATA_DIR, "train.csv")

N_TRAIN = 1200000  # keep same training size for runtime

usecols = [
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]

TRAIN_TOTAL_ROWS_EST = 55423856  # from provided dataset stats (excluding header)
rng_sample = np.random.RandomState(0)
keep = np.zeros(TRAIN_TOTAL_ROWS_EST + 1, dtype=np.bool_)
keep[0] = True  # always keep header
if N_TRAIN >= TRAIN_TOTAL_ROWS_EST:
    keep[1:] = True
else:
    chosen = rng_sample.choice(
        np.arange(1, TRAIN_TOTAL_ROWS_EST + 1), size=N_TRAIN, replace=False
    )
    keep[chosen] = True


def _skiprows(i):
    return not keep[i]


df_train = pd.read_csv(train_path, usecols=usecols, skiprows=_skiprows)
df_train = df_train.dropna()

df_train = df_train[df_train["passenger_count"].between(1, 6)]
df_train = df_train[(df_train["fare_amount"] > 0) & (df_train["fare_amount"] < 250)]

nyc_bounds = (
    (df_train["pickup_longitude"].between(-75, -72))
    & (df_train["dropoff_longitude"].between(-75, -72))
    & (df_train["pickup_latitude"].between(40, 42))
    & (df_train["dropoff_latitude"].between(40, 42))
)
df_train = df_train[nyc_bounds]

coord_nonzero = (
    (df_train["pickup_longitude"].abs() > 1e-6)
    & (df_train["pickup_latitude"].abs() > 1e-6)
    & (df_train["dropoff_longitude"].abs() > 1e-6)
    & (df_train["dropoff_latitude"].abs() > 1e-6)
)
df_train = df_train[coord_nonzero]

d_lon = (df_train["pickup_longitude"] - df_train["dropoff_longitude"]).abs()
d_lat = (df_train["pickup_latitude"] - df_train["dropoff_latitude"]).abs()
df_train = df_train[(d_lon <= 2.0) & (d_lat <= 2.0)]

add_datepart(df_train, "pickup_datetime", drop=True)

df_train["haversine_km"] = haversine_km(
    df_train["pickup_longitude"].values,
    df_train["pickup_latitude"].values,
    df_train["dropoff_longitude"].values,
    df_train["dropoff_latitude"].values,
)

df_train["manhattan"] = (
    (df_train["pickup_longitude"] - df_train["dropoff_longitude"]).abs()
    + (df_train["pickup_latitude"] - df_train["dropoff_latitude"]).abs()
).astype(np.float32)

add_geo_features(df_train)

add_bearing_feature(df_train)

df_train = df_train[df_train["haversine_km"] >= 0.05]

fare_per_km = df_train["fare_amount"] / df_train["haversine_km"]
df_train = df_train[fare_per_km.between(1.0, 60.0)]

df_train = df_train[
    (df_train["fare_amount"] >= 2.5 + 0.5 * df_train["haversine_km"])
    & (df_train["fare_amount"] <= 2.5 + 25.0 * df_train["haversine_km"] + 50.0)
]

haversine_cap = float(np.percentile(df_train["haversine_km"].values, 99.9))
haversine_cap = max(10.0, min(haversine_cap, 200.0))
df_train["haversine_km"] = df_train["haversine_km"].clip(0.0, haversine_cap)
df_test["haversine_km"] = df_test["haversine_km"].clip(0.0, haversine_cap)
print("haversine_cap:", haversine_cap)

manhattan_cap = float(np.percentile(df_train["manhattan"].values, 99.9))
manhattan_cap = max(0.1, min(manhattan_cap, 10.0))
df_train["manhattan"] = df_train["manhattan"].clip(0.0, manhattan_cap)
df_test["manhattan"] = df_test["manhattan"].clip(0.0, manhattan_cap)
print("manhattan_cap:", manhattan_cap)

abs_dlon_cap = float(np.percentile(df_train["abs_dlon"].values, 99.9))
abs_dlon_cap = max(0.01, min(abs_dlon_cap, 5.0))
df_train["abs_dlon"] = df_train["abs_dlon"].clip(0.0, abs_dlon_cap)
df_test["abs_dlon"] = df_test["abs_dlon"].clip(0.0, abs_dlon_cap)
print("abs_dlon_cap:", abs_dlon_cap)

abs_dlat_cap = float(np.percentile(df_train["abs_dlat"].values, 99.9))
abs_dlat_cap = max(0.01, min(abs_dlat_cap, 5.0))
df_train["abs_dlat"] = df_train["abs_dlat"].clip(0.0, abs_dlat_cap)
df_test["abs_dlat"] = df_test["abs_dlat"].clip(0.0, abs_dlat_cap)
print("abs_dlat_cap:", abs_dlat_cap)

bearing_cap = float(np.percentile(np.abs(df_train["bearing_sin"].values), 99.9))
bearing_cap = max(0.1, min(bearing_cap, 1.0))
df_train["bearing_sin"] = df_train["bearing_sin"].clip(-bearing_cap, bearing_cap)
df_test["bearing_sin"] = df_test["bearing_sin"].clip(-bearing_cap, bearing_cap)
print("bearing_cap:", bearing_cap)

for c in feature_cols:
    if c not in df_train.columns:
        df_train[c] = 0

for c in [col for col in feature_cols if col.startswith("pickup_datetime")]:
    if c in df_train.columns:
        df_train[c] = df_train[c].fillna(0)

X_all_unscl = df_train[feature_cols].astype(np.float32).values
y_all = df_train["fare_amount"].astype(np.float32).values.reshape(-1, 1)

col_means = np.nanmean(
    np.where(np.isfinite(X_all_unscl), X_all_unscl, np.nan), axis=0
).astype(np.float32)
bad = ~np.isfinite(X_all_unscl)
if bad.any():
    X_all_unscl[bad] = np.take(col_means, np.where(bad)[1])

mu = X_all_unscl.mean(axis=0).astype(np.float32)
sigma = X_all_unscl.std(axis=0).astype(np.float32)
sigma = np.where(sigma < 1e-6, 1.0, sigma).astype(np.float32)

X_all = (X_all_unscl - mu) / sigma

rng = np.random.RandomState(0)
idx = np.arange(X_all.shape[0])
rng.shuffle(idx)

split = int(0.9 * len(idx))
tr_idx, va_idx = idx[:split], idx[split:]

X_train, X_valid = X_all[tr_idx], X_all[va_idx]
y_train, y_valid = y_all[tr_idx], y_all[va_idx]

fallback_fare = float(np.median(y_train))
print(X_train.shape, y_train.shape, X_valid.shape, y_valid.shape)
print("mu shape:", mu.shape, "sigma shape:", sigma.shape)
print("fallback_fare (median train):", fallback_fare)




## === cell 6
def fit_ridge_closed_form(X, y, alpha=1.0):
    """
    Closed-form ridge regression with intercept.
    X: (n, d) float32/float64
    y: (n, 1)
    Returns w (d,), b (scalar)
    """
    X = X.astype(np.float64, copy=False)
    y = y.astype(np.float64, copy=False)

    n, d = X.shape
    X1 = np.concatenate([X, np.ones((n, 1), dtype=np.float64)], axis=1)
    I = np.eye(d + 1, dtype=np.float64)
    I[-1, -1] = 0.0

    A = X1.T @ X1 + alpha * I
    B = X1.T @ y
    w_full = np.linalg.solve(A, B)
    w = w_full[:-1, 0]
    b = w_full[-1, 0]
    return w.astype(np.float32), np.float32(b)


def predict_linear(X, w, b):
    return (X.astype(np.float32, copy=False) @ w.reshape(-1, 1) + b).astype(np.float32)


alphas = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
best_alpha, best_rmse = None, None
rmse_by_alpha = {}
for a in alphas:
    w_tmp, b_tmp = fit_ridge_closed_form(X_train, y_train, alpha=float(a))
    y_valid_pred_tmp = predict_linear(X_valid, w_tmp, b_tmp)
    rmse_tmp = float(np.sqrt(np.mean((y_valid_pred_tmp - y_valid) ** 2)))
    rmse_by_alpha[float(a)] = rmse_tmp
    print(f"alpha={a:<7} valid_rmse={rmse_tmp:.5f}")
    if (best_rmse is None) or (rmse_tmp < best_rmse):
        best_rmse = rmse_tmp
        best_alpha = float(a)

refine_mults = [0.6, 0.8, 1.0, 1.25, 1.6]
refine_alphas = sorted({max(1e-6, best_alpha * m) for m in refine_mults})
for a in refine_alphas:
    if a in rmse_by_alpha:
        continue
    w_tmp, b_tmp = fit_ridge_closed_form(X_train, y_train, alpha=float(a))
    y_valid_pred_tmp = predict_linear(X_valid, w_tmp, b_tmp)
    rmse_tmp = float(np.sqrt(np.mean((y_valid_pred_tmp - y_valid) ** 2)))
    print(f"alpha={a:<7.6f} valid_rmse={rmse_tmp:.5f} (refine)")
    if rmse_tmp < best_rmse:
        best_rmse = rmse_tmp
        best_alpha = float(a)

print("best_alpha:", best_alpha, "best_valid_rmse:", best_rmse)

X_full = X_all
y_full = y_all
w, b = fit_ridge_closed_form(X_full, y_full, alpha=best_alpha)



## === cell 7
x_test_unscl = df_test[feature_cols].astype(np.float32).values

bad_t = ~np.isfinite(x_test_unscl)
if bad_t.any():
    x_test_unscl[bad_t] = np.take(mu, np.where(bad_t)[1])

x_test = (x_test_unscl - mu) / sigma
print("x_test:", x_test.shape)

y_test = predict_linear(x_test, w, b).reshape(-1)

y_test = y_test.astype(np.float32, copy=False)
y_test[~test_nyc_bounds.values] = np.float32(fallback_fare)

y_test = np.clip(y_test, 2.5, 250.0).astype(np.float32)

print(y_test[:5], y_test.shape)



## === cell 8
my_submission = pd.DataFrame(
    {"key": df_test["key"].astype(str).values, "fare_amount": y_test}
)
my_submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", my_submission.shape)
print(my_submission.head())
print("Exists submission.csv:", os.path.exists("submission.csv"))
