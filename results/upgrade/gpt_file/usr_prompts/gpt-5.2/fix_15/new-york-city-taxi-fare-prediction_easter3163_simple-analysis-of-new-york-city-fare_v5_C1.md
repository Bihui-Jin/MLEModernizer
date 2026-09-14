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

4.70641

# 6. Current score

6.64632

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 199.84822) has done: 'I fix the pandas datetime attribute errors by replacing deprecated `.dt.week`/`.dt.weekofyear` with ISO calendar week extraction that works in pandas 2.2+. I also make the input paths robust to your environment by resolving to the available `/kaggle/input/...` or `/kaggle/data/...` locations while keeping the same filenames. To move the RMSE score dramatically closer to the target with minimal modeling changes, I keep your exact 3-model ensemble but add a standard Haversine distance feature (a legitimate core-feature fix for taxi fare) and include the existing datetime features consistently for both train/test. Finally, I ensure the submission file is written as a valid `.csv` with the required `key,fare_amount` columns aligned to `test.csv` order.'
- What this solution (achieved 444.24131) has done: 'Your RMSE is extremely high because the model is learning on a train sample that is (a) filtered with an incorrect longitude range (`-75 < lon < 75` keeps many invalid longitudes) and (b) missing two very strong baseline signals for this competition: straight-line distance in *miles* (used implicitly by fare pricing) and airport/manhattan locality effects. With minimal changes that preserve your exact 3-model ensemble and training approach, I fix the longitude filters to NYC-appropriate bounds, add a Haversine distance in miles plus simple “center distance” features, and keep the same prediction averaging and submission writing. These changes are legitimate feature/cleaning corrections and should move RMSE dramatically down toward your target band without altering your modeling core. I also ensure deterministic sampling and safe NaN handling so the pipeline runs reliably and writes a valid submission CSV.'
- What this solution (achieved 442.54093) has done: 'Your current RMSE is far above target, so we should improve (lower) it with the smallest changes that don’t alter your model ensemble/training loop. The biggest likely issue left is that you train on a random slice of the first 500k rows, which is often noisy/dirty; using a deterministic but more representative sample across the whole file (still 500k) typically drops RMSE a lot without changing core modeling. I also fix a subtle submission bug: when `fare_amount` is missing after the merge, you currently fill with the median of an all-NaN column (stays NaN); we instead fill with a safe fallback (training median) to ensure a valid submission. Everything else (features, models, and averaging) stays the same.'
- What this solution (achieved 445.8949) has done: 'The timeout is dominated by `pd.read_csv(..., skiprows=lambda ...)`, which calls a Python lambda for every one of the ~55M rows and recreates `RandomState(42)` each time—this is catastrophically slow. I replace that with a provably-equivalent one-pass chunked read that keeps each row with the same probability (so expected sample size matches) and then applies the same fixed-size cap to 500k rows, preserving the algorithm’s semantics. I also eliminate heavy plotting (which doesn’t affect predictions) and avoid a few unnecessary large temporary allocations (e.g., `np.full(len(train), ...)`) while keeping feature logic identical. Everything else—features, models, training, and blending—remains unchanged.'
- What this solution (achieved 12.21026) has done: 'Your RMSE is wildly above target (lower is better), so we should improve it with the smallest changes that keep your exact feature set and 3-model blend intact. The main issue is that KNN and LinearRegression are extremely sensitive to feature scale; right now distance features (miles/km) and time/week features are on very different scales, which can yield terrible generalization and blow up RMSE. I add a `StandardScaler` and apply it to both train/test features (no change to model types, loss, or blending), while keeping the RandomForest on the original (unscaled) features since it doesn’t benefit from scaling. I also add a minimal post-processing clip to cap extreme predictions (still legitimate for fare amounts) to reduce outlier-driven RMSE without changing the training approach.'
- What this solution (achieved 12.74015) has done: 'We need to move RMSE down from 12.21 toward 4.71 (lower is better), so we make the smallest changes that improve generalization without changing your 3-model ensemble, features, or training loop. The biggest remaining issue is that KNN on 500k points is both impractically slow and effectively behaves poorly; we keep KNN but fit it on a small, deterministic subsample (still using the same scaled features and prediction blending), which typically improves and stabilizes results while staying within runtime. We also ensure the scaler is fit only on the same rows used by Linear/KNN (and those rows are the same filtered training set), and we add a minimal, legitimate log1p-transform on the *target for LinearRegression only* (with inverse transform at predict) to reduce skew/outlier impact while keeping the model type and blending intact. Finally, we keep your submission alignment/format exactly as required and still clip to a realistic fare range.'
- What this solution (achieved 12.22273) has done: 'Your current RMSE (12.74) is far worse than the target (4.71), so we should make small, legitimate changes that reliably lower error without changing your 3-model ensemble or training loop. The biggest low-risk gain here is fixing the LinearRegression target transform: using `log1p` on a target that can include small/near-zero values is OK, but it tends to under-predict higher fares unless you apply a proper bias correction; instead, we keep LinearRegression but remove the log transform to better match RMSE on the original scale. Next, we align scaling so LR and KNN use the exact same scaler fit on the same training data (this preserves model types and blending, but reduces distribution mismatch and improves generalization). Finally, we keep your features/filters/blend/CSV format intact, only adding a tiny safety clip after blending (still within your existing 0–200 range) to reduce outlier-driven RMSE.'
- What this solution (achieved 12.22271) has done: 'To move RMSE down from ~12.22 toward 4.71 (lower is better) without changing your 3-model ensemble or training loop, I fix the biggest remaining modeling mismatch: LinearRegression is currently unregularized and tends to overfit/behave poorly with correlated, scaled features in this dataset, hurting RMSE. I replace it with Ridge regression (same linear-model family and same training approach) while keeping the same scaler, the same KNN subsample, the same RandomForest, and the same 2:2:1 blending. I also add a tiny, legitimate cleaning step to remove obviously impossible coordinate zeros (0,0) that can sneak into the random sample and degrade generalization, without changing your feature set or metric semantics. The submission writing, columns, alignment, and filename remain unchanged and a valid `.csv` is produced.'
- What this solution (achieved 12.22053) has done: 'Your current RMSE (~12.22) is far above the target (4.71), so we should improve (lower) it with the smallest changes that keep your 3-model blend and feature set intact. The biggest low-risk fix is to make the train sample more representative across the full 55M rows: instead of keeping each row with a tiny Bernoulli probability (which produces a variable-size, potentially biased sample), we take a deterministic, evenly-spaced sample using `skiprows` so the distribution is more stable while staying at ~500k rows. We also add one minimal, standard cleaning step used in this competition: remove trips with unrealistically large straight-line distance (often bad GPS points) so they don’t distort the linear/KNN parts. Everything else (features, scaler usage, Ridge/KNN/RF models, blending, clipping, and submission writing) stays the same and still produces a valid `.csv`.'
- What this solution (achieved 12.2279) has done: 'We need to move RMSE down from 12.22 toward 4.71 (lower is better), so the smallest high-impact change is to fix the training sampling: your current `skiprows` keeps only every ~110th row, which can create a biased/aliased subset of the huge file. I switch to a deterministic hash-based sampler on the `key` column (still ~500k rows, still a single pass), which is much more representative while preserving your overall pipeline, features, and 3-model blend. I also add one minimal, standard cleaning step: remove clearly impossible coordinates (outside valid lat/lon ranges) before feature building, which reduces noise without changing model logic. Everything else—feature set, scaler usage, Ridge/KNN/RF models, blending weights, clipping, and submission schema—stays the same and it still write a valid `.csv`.'
- What this solution (achieved 5.38091) has done: 'Your current RMSE (12.2279; lower is better) is far above the target (4.70641), so we should make a small, legitimate change that improves generalization without changing your 3-model blend or feature set. The biggest remaining issue is a train/test mismatch: you apply strict NYC coordinate bounds and passenger/fare filters to train, but you don’t apply the same coordinate/passenger cleaning to test, so the model sees out-of-distribution test rows and produces large errors/outliers. I apply the same basic validity filters to `test` (lat/lon bounds + passenger_count range + drop 0/0 coords) and then align submission order to `test.csv` directly (no merge dependency), which reduces pathological predictions and should move RMSE down toward the target band while preserving your core logic. Everything else—sampling, features, scaler usage, Ridge/KNN/RF models, blending weights, and clipping—remains unchanged.'
- What this solution (achieved 6.64632) has done: 'To move RMSE down from 5.38091 toward 4.70641 (lower is better) without changing your 3-model blend or training loop, the smallest high-impact fix is to stop dropping test rows (your current filtering removes rows and then your merge fills them with a constant median, which hurts RMSE). I keep the exact same cleaning logic but change it to “soft cleaning” for test: we preserve all test keys/rows and only sanitize invalid values into safe defaults so every row gets a model-based prediction. I also add the same train-only outlier distance cleaning boundary as a prediction-time clip on distance-derived features for test (not a new feature), which reduces extreme model extrapolations. Everything else—features, scaler usage, Ridge/KNN/RF models, blending weights, and submission schema—stays the same.'
- What this solution (achieved 6.64632) has done: 'We need to lower RMSE from 6.64632 toward 4.70641 (lower is better), so we should make a small, low-risk improvement that reduces systematic error without changing your 3-model blend or training approach. The biggest remaining gap in your feature set is that you compute time features from local/naive timestamps; in this competition the provided timestamps are effectively UTC, and extracting hour/day/week in UTC (or consistently in a fixed timezone) typically improves RMSE materially. I keep all models, weights, filters, and features the same, but parse datetimes as UTC and derive the same calendar features from UTC consistently for both train and test. This is a minimal semantic fix (no new features, no model changes) that should move the score closer to the target band while preserving the pipeline and producing the same submission format.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd


def _resolve_path(filename: str) -> str:
    candidates = [
        os.path.join("/kaggle/input", filename),
        os.path.join("/kaggle/data", filename),
        os.path.join("/kaggle/input/new-york-city-taxi-fare-prediction", filename),
        os.path.join("/kaggle/data/new-york-city-taxi-fare-prediction", filename),
        os.path.join("../input", filename),
        os.path.join("../input/new-york-city-taxi-fare-prediction", filename),
        os.path.join("../data", filename),
        os.path.join("../data/new-york-city-taxi-fare-prediction", filename),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return os.path.join("../input", filename)


TRAIN_PATH = _resolve_path("train.csv")
TEST_PATH = _resolve_path("test.csv")
SAMPLE_SUB_PATH = _resolve_path("sample_submission.csv")

USECOLS = [
    "key",
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]

N_SAMPLE = 500_000


def _read_train_hash_sample(
    path: str, n_sample: int, chunksize: int = 250_000
) -> pd.DataFrame:
    modulus = 111
    parts = []
    kept = 0

    for chunk in pd.read_csv(path, usecols=USECOLS, chunksize=chunksize):
        h = pd.util.hash_pandas_object(chunk["key"], index=False).astype("uint64")
        mask = (h % modulus) == 0
        if mask.any():
            parts.append(chunk.loc[mask])
            kept += int(mask.sum())
            if kept >= n_sample:
                break

    if not parts:
        return pd.DataFrame(columns=USECOLS)

    out = pd.concat(parts, axis=0, ignore_index=True)
    if len(out) > n_sample:
        out = out.iloc[:n_sample].copy()
    return out


train = _read_train_hash_sample(TRAIN_PATH, N_SAMPLE)
test = pd.read_csv(TEST_PATH)



## === cell 1
train.shape



## === cell 2
train.head()



## === cell 3
import datetime as dt  # kept as in original environment usage (even if unused)



## === cell 4
train = train.dropna(
    subset=[
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
    ]
)
train = train.loc[
    (train["pickup_longitude"].between(-180, 180))
    & (train["dropoff_longitude"].between(-180, 180))
    & (train["pickup_latitude"].between(-90, 90))
    & (train["dropoff_latitude"].between(-90, 90))
].copy()

train["pickup_datetime"] = pd.to_datetime(
    train["pickup_datetime"], errors="coerce", utc=True
)
train = train.dropna(subset=["pickup_datetime"])

iso_train = train["pickup_datetime"].dt.isocalendar()
train["hour"] = train["pickup_datetime"].dt.hour
train["day"] = train["pickup_datetime"].dt.day
train["week"] = iso_train.week.astype("int16")
train["month"] = train["pickup_datetime"].dt.month
train["day_of_year"] = train["pickup_datetime"].dt.dayofyear
train["week_of_year"] = iso_train.week.astype("int16")



## === cell 5
test["pickup_datetime"] = pd.to_datetime(
    test["pickup_datetime"], errors="coerce", utc=True
)
iso_test = test["pickup_datetime"].dt.isocalendar()

test["hour"] = test["pickup_datetime"].dt.hour
test["day"] = test["pickup_datetime"].dt.day
test["week"] = iso_test.week.astype("int16")
test["month"] = test["pickup_datetime"].dt.month
test["day_of_year"] = test["pickup_datetime"].dt.dayofyear
test["week_of_year"] = iso_test.week.astype("int16")



## === cell 6
train.head()
train = train.dropna(how="any", axis="rows")



## === cell 7
train = train.loc[(train["fare_amount"] > 0) & (train["fare_amount"] < 200)]
train = train.loc[(train["pickup_longitude"] > -75) & (train["pickup_longitude"] < -72)]
train = train.loc[(train["pickup_latitude"] > 40) & (train["pickup_latitude"] < 42)]
train = train.loc[
    (train["dropoff_longitude"] > -75) & (train["dropoff_longitude"] < -72)
]
train = train.loc[(train["dropoff_latitude"] > 40) & (train["dropoff_latitude"] < 42)]
train = train.loc[(train["passenger_count"] >= 1) & (train["passenger_count"] <= 8)]

train = train.loc[
    ~(
        (train["pickup_longitude"].abs() < 1e-9)
        | (train["pickup_latitude"].abs() < 1e-9)
        | (train["dropoff_longitude"].abs() < 1e-9)
        | (train["dropoff_latitude"].abs() < 1e-9)
    )
].copy()



## === cell 8
for col in ["pickup_longitude", "dropoff_longitude"]:
    test[col] = pd.to_numeric(test[col], errors="coerce")
for col in ["pickup_latitude", "dropoff_latitude"]:
    test[col] = pd.to_numeric(test[col], errors="coerce")
test["passenger_count"] = pd.to_numeric(test["passenger_count"], errors="coerce")

NYC_LON, NYC_LAT = -73.985428, 40.748817
test["pickup_longitude"] = test["pickup_longitude"].fillna(NYC_LON).clip(-75, -72)
test["dropoff_longitude"] = test["dropoff_longitude"].fillna(NYC_LON).clip(-75, -72)
test["pickup_latitude"] = test["pickup_latitude"].fillna(NYC_LAT).clip(40, 42)
test["dropoff_latitude"] = test["dropoff_latitude"].fillna(NYC_LAT).clip(40, 42)

test["passenger_count"] = test["passenger_count"].fillna(1).clip(1, 8)

for c, default in [
    ("hour", 12),
    ("day", 15),
    ("week", 26),
    ("month", 6),
    ("day_of_year", 180),
    ("week_of_year", 26),
]:
    test[c] = pd.to_numeric(test[c], errors="coerce").fillna(default)



## === cell 9
train["abs_diff_longitude"] = (
    train["pickup_longitude"] - train["dropoff_longitude"]
).abs()
train["abs_diff_latitude"] = (
    train["pickup_latitude"] - train["dropoff_latitude"]
).abs()



## === cell 10
test["abs_diff_longitude"] = (
    test["pickup_longitude"] - test["dropoff_longitude"]
).abs()
test["abs_diff_latitude"] = (test["pickup_latitude"] - test["dropoff_latitude"]).abs()




## === cell 11
def haversine_km(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371.0 * c


train["haversine_km"] = haversine_km(
    train["pickup_longitude"].values,
    train["pickup_latitude"].values,
    train["dropoff_longitude"].values,
    train["dropoff_latitude"].values,
)
test["haversine_km"] = haversine_km(
    test["pickup_longitude"].values,
    test["pickup_latitude"].values,
    test["dropoff_longitude"].values,
    test["dropoff_latitude"].values,
)

train["haversine_miles"] = train["haversine_km"] * 0.621371
test["haversine_miles"] = test["haversine_km"] * 0.621371

train["manhattan_deg"] = train["abs_diff_longitude"] + train["abs_diff_latitude"]
test["manhattan_deg"] = test["abs_diff_longitude"] + test["abs_diff_latitude"]

NYC_LON, NYC_LAT = -73.985428, 40.748817

train["pickup_center_km"] = haversine_km(
    train["pickup_longitude"].values,
    train["pickup_latitude"].values,
    NYC_LON,
    NYC_LAT,
)
train["dropoff_center_km"] = haversine_km(
    train["dropoff_longitude"].values,
    train["dropoff_latitude"].values,
    NYC_LON,
    NYC_LAT,
)
test["pickup_center_km"] = haversine_km(
    test["pickup_longitude"].values,
    test["pickup_latitude"].values,
    NYC_LON,
    NYC_LAT,
)
test["dropoff_center_km"] = haversine_km(
    test["dropoff_longitude"].values,
    test["dropoff_latitude"].values,
    NYC_LON,
    NYC_LAT,
)

train = train.loc[~((train["haversine_km"] < 0.05) & (train["fare_amount"] > 5.0))]
train = train.loc[train["haversine_km"] <= 100.0].copy()

for c in ["haversine_km", "haversine_miles", "pickup_center_km", "dropoff_center_km"]:
    test[c] = pd.to_numeric(test[c], errors="coerce").fillna(0.0)
test["haversine_km"] = test["haversine_km"].clip(lower=0.0, upper=100.0)
test["haversine_miles"] = test["haversine_miles"].clip(
    lower=0.0, upper=100.0 * 0.621371
)
test["pickup_center_km"] = test["pickup_center_km"].clip(lower=0.0, upper=100.0)
test["dropoff_center_km"] = test["dropoff_center_km"].clip(lower=0.0, upper=100.0)



## === cell 12
train.head()



## === cell 13
train.head()



## === cell 14
pass



## === cell 15
feature_names = [
    "hour",
    "abs_diff_longitude",
    "abs_diff_latitude",
    "haversine_km",
    "haversine_miles",
    "manhattan_deg",
    "pickup_center_km",
    "dropoff_center_km",
    "passenger_count",
    "month",
    "day_of_year",
    "week_of_year",
]
feature_names



## === cell 16
label_name = "fare_amount"
label_name



## === cell 17
X_train = train[feature_names].copy()
y_train = train[label_name].copy()
X_test = test[feature_names].copy()

X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0.0)
X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0.0)



## === cell 18
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
import xgboost as xgb  # kept to preserve original imports even if unused



## === cell 19
KNN_FIT_N = (
    50_000  # keep the same runtime-oriented KNN subsampling as your current solution
)

if len(X_train) > KNN_FIT_N:
    knn_idx = X_train.sample(n=KNN_FIT_N, random_state=42).index
else:
    knn_idx = X_train.index

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

pos = pd.Series(np.arange(len(X_train)), index=X_train.index)
X_train_scaled_knn = X_train_scaled[pos.loc[knn_idx].values]



## === cell 20
regr = Ridge(alpha=1.0, random_state=42)
regr.fit(X_train_scaled, y_train)
regr_prediction = regr.predict(X_test_scaled)



## === cell 21
knr = KNeighborsRegressor(n_neighbors=10, weights="distance")
knr.fit(X_train_scaled_knn, y_train.loc[knn_idx])
knr_prediction = knr.predict(X_test_scaled)



## === cell 22
rfr = RandomForestRegressor(
    n_estimators=120,
    random_state=42,
    n_jobs=-1,
    min_samples_leaf=2,
)
rfr.fit(X_train, y_train)
rfr_prediction = rfr.predict(X_test)



## === cell 23
predictions = (regr_prediction * 2 + rfr_prediction * 2 + knr_prediction) / 5
predictions = np.clip(predictions, 0.0, 200.0)



## === cell 24
predictions



## === cell 25
submission = pd.DataFrame({"key": test["key"].values, "fare_amount": predictions})
submission["fare_amount"] = submission["fare_amount"].astype(float).clip(0.0, 200.0)
submission.head()



## === cell 26
out_path = "./simplenewyorktaxi.csv"
submission.to_csv(out_path, index=False)
out_path
