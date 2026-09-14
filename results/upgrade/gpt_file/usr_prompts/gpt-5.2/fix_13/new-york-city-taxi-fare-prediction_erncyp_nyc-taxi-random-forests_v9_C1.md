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
seaborn==0.12.2
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

3.41764

# 6. Current score

5.5452

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 9.34196) has done: 'Your current score is far from the target (8.14 vs 3.42, lower is better), so we can make a few minimal, metric-aligned fixes without changing the core model approach (still a RandomForest on the same feature set). The biggest issue is that you train/evaluate on noisy/outlier fares and impossible geos; adding standard NYC Taxi competition cleaning (fare range, passenger_count range, and NYC-ish coordinate bounds) typically reduces RMSE substantially. Second, your “busyness” merges are currently wrong because you merge without specifying keys, which can silently misalign/join on unintended columns; fixing the merge keys stabilizes those features and improves generalization. Finally, we make the RandomForest deterministic (random_state) and clip negative predictions to 0 (fares can’t be negative), both of which usually nudge RMSE down.'
- What this solution (achieved 5.4051) has done: 'We keep your exact feature set and RandomForest setup, but fix two issues that typically cause a large RMSE gap here: (1) stabilize the “busyness” features by ensuring the grouped tables `a` and `b` have the expected column names after the `agg(["mean","count"])` (your current renaming can silently misalign depending on pandas behavior), and (2) avoid `minute` being computed but unused by keeping everything else identical while making the train/test preprocessing fully consistent and deterministic. Additionally, we ensure busyness merges always succeed by explicitly enforcing the merge keys’ dtypes and filling missing busyness with the training median (instead of 1) to better match the training distribution without changing the model. These are minimal changes aimed at reducing the current RMSE (9.34) toward the target (3.42) without altering the core modeling approach.'
- What this solution (achieved 5.89711) has done: 'Your current RMSE (5.4051) is still worse than the target (3.41764), so we should make small, competition-standard improvements without changing the core approach (same features, same RandomForest, same training loop). The biggest remaining gap is usually from residual outliers/dirty rows and a mismatch between training sampling and the test distribution; we tighten the cleaning to remove extreme distances/fare-per-km anomalies while keeping the same engineered features. We also train on more rows (still feasible under the time limit with your 20-tree RandomForest) to reduce variance and better match the leaderboard distribution. Finally, we ensure strict feature validity (clip passenger_count, handle any remaining NaNs in both train/test busyness consistently) and keep the submission format unchanged.'
- What this solution (achieved 5.28032) has done: 'We keep your exact feature set and RandomForest setup, but fix a likely distribution mismatch introduced by over-aggressive fare-per-km filtering: replacing the hard `fpk_cond` removal with a light winsorization (clipping) of extreme `fare_per_km` rows keeps more realistic long/slow trips that appear in test, which typically reduces RMSE materially for this competition. We also add a minimal, standard cleaning step to drop obviously bad coordinates (0/0) and enforce a tighter but still NYC-typical bounding box, without changing the model or features. Finally, we make train/test preprocessing fully consistent by applying the same coordinate validity filters before distance computation, and keep the submission schema unchanged.'
- What this solution (achieved 5.22922) has done: 'Your current RMSE (5.28032, lower is better) is still far from the target (3.41764), so we make two small, metric-aligned changes that don’t alter your model/feature set: (1) stop modifying the training target via fare-per-km winsorization (this injects label noise and typically hurts RMSE against true fares), and instead only filter extreme/unrealistic fare-per-km rows; (2) add a minimal “airport flag” feature (pickup/dropoff near JFK/LGA/EWR) which is a standard, lightweight improvement for this competition and keeps the same RandomForest training approach. We also apply the same basic geo mask to test (without dropping rows) to keep preprocessing consistent, and continue clipping negative predictions to 0. These changes are expected to reduce the RMSE gap toward the target without rewriting your pipeline.'
- What this solution (achieved 5.50621) has done: 'We keep your exact RandomForest model and feature set, but make two minimal, metric-aligned fixes that typically move NYC Taxi RMSE down without changing core logic. First, we add a single standard geodesic feature (`abs_lat_diff`, `abs_lon_diff`) that complements haversine distance and is very cheap; this preserves the same training loop and model family while improving fit. Second, we apply a very small, competition-standard floor on predicted fares (>= 2.5) instead of only clipping at 0, since test fares are never below the base fare and this usually reduces RMSE from low-end underpredictions. Everything else (cleaning approach, busyness features, airport flag, RF hyperparams, I/O paths, submission schema) stays the same.'
- What this solution (achieved 5.58068) has done: 'We keep your exact RandomForest model and engineered feature set, but fix a key data mismatch that typically drives RMSE up: your `year/month/day/hour` features are currently derived from UTC timestamps, while NYC taxi patterns are strongly local-time dependent. By converting `pickup_datetime` to America/New_York before extracting these time parts (train and test consistently), we preserve the same core logic while making the features more predictive, which should move RMSE down toward the 3.42 target. We also ensure the datetime conversion is robust (handles already-naive timestamps) and keep everything else (cleaning, busyness merges, airport flag, prediction clipping, submission format) unchanged.'
- What this solution (achieved 5.29846) has done: 'We need to move RMSE down from 5.58 toward 3.42 (lower is better), and the biggest remaining issue consistent with your current pipeline is that a vanilla RandomForest struggles to extrapolate high fares; adding a very light log1p target transform (train on log-fare, then expm1 back) usually reduces RMSE materially without changing the model family or training loop. To avoid introducing bias from the transform at the low end, we keep your existing prediction floor (2.5) after inversion. We also make train/test datetime localization more robust (handle already-naive timestamps) to prevent silent NaTs that can degrade features. Everything else (features, busyness, airport flag, cleaning, RF hyperparams, I/O paths, submission schema) stays the same.'
- What this solution (achieved 386.37698) has done: 'Your RMSE is still well above the target (5.30 vs 3.42, lower is better), so we make a minimal, metric-aligned improvement without changing the core approach (same RF model family, same features, same training loop). The biggest remaining mismatch is that the model never sees the global “base fare” and “distance rate” structure, so we add a simple two-feature linear “baseline” prediction (distance + passenger_count) and train the RandomForest on the residual in log1p space; this keeps the same RandomForest architecture/loop but typically reduces RMSE substantially. We then add the baseline back at prediction time and keep your existing fare floor. Everything else (cleaning, busyness merges, airport flag, datetime localization, submission schema) remains unchanged.'
- What this solution (achieved 386.31804) has done: 'Your current RMSE (386) indicates a severe train/test mismatch or post-processing bug rather than a small modeling deficiency. The smallest fix that preserves your core approach is to make the residual modeling *signed*: right now you force residuals to be non-negative before `log1p`, so the model can never correct an overestimated baseline and systematically overpredict, which can explode RMSE. We keep the same LinearRegression baseline + RandomForest residual idea, but switch to a sign-preserving transform (`signed_log1p`) and its exact inverse at inference, allowing negative residuals while keeping the same model family and training loop. Everything else (features, cleaning, busyness, airport flag, submission format/paths) is left unchanged.'
- What this solution (achieved 5.53695) has done: 'Your RMSE exploding to ~386 strongly suggests your predictions contain NaNs/inf or extreme values that Kaggle scores very poorly, rather than a small modeling issue. The minimal, score-aligned fix is to (1) ensure train/test datetime localization never produces NaT (fallback to naive parsing if needed), (2) make the baseline prediction robust to NaNs by imputing `distance`/`passenger_count` for the test baseline inputs, and (3) hard-sanitize final predictions by replacing non-finite values and clipping to a realistic upper bound derived from the training distribution (this keeps evaluation semantics while preventing catastrophic outliers). These changes don’t alter your model family, features, or training loop; they only prevent invalid/extreme outputs that drive RMSE far from the target. The submission file format and paths remain unchanged.'
- What this solution (achieved 5.5452) has done: 'We need to move RMSE down from 5.53695 toward 3.41764 (lower is better), so we make a couple of minimal, metric-aligned fixes without changing your core approach (LinearRegression baseline + RandomForest on signed-log residuals with the same feature set). The biggest bug hurting generalization is that `passenger_count` is used in the baseline model but is not reliably numeric/clean on train (unlike test), so we coerce/impute/clip it in train exactly like test before fitting the baseline and building features. Next, your cleaning still allows some high-leverage label noise (fare vs distance inconsistency) in a way that can worsen RMSE; we add a very light, standard filter that removes only extreme “fare per km” outliers using fixed, competition-common bounds (keeps semantics, avoids leakage), which usually improves RMSE materially. Finally, we ensure feature matrices are built after the cleaned/imputed `passenger_count` so the baseline is consistent train↔test.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)



## === cell 1
train_df = pd.read_csv("../input/train.csv", nrows=2_000_000)




## === cell 2
def haversine_np(lon1, lat1, lon2, lat2):
    """
    https://stackoverflow.com/questions/29545704/fast-haversine-approximation-python-pandas
    Calculate the great circle distance between two points on the earth (specified in decimal degrees)
    All args must be of equal length.
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km




## === cell 3
coord_cols = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
]
for c in coord_cols:
    train_df[c] = pd.to_numeric(train_df[c], errors="coerce")

zero_coord = (train_df["pickup_longitude"] == 0) & (train_df["pickup_latitude"] == 0)
zero_coord |= (train_df["dropoff_longitude"] == 0) & (train_df["dropoff_latitude"] == 0)
train_df = train_df.loc[~zero_coord].copy()



## === cell 4
train_df["distance"] = haversine_np(
    train_df["pickup_longitude"],
    train_df["pickup_latitude"],
    train_df["dropoff_longitude"],
    train_df["dropoff_latitude"],
)



## === cell 5
train_df["pickup_datetime"] = pd.to_datetime(
    train_df["pickup_datetime"], errors="coerce", utc=True
)
train_df["pickup_datetime_local"] = train_df["pickup_datetime"].dt.tz_convert(
    "America/New_York"
)
bad_dt = train_df["pickup_datetime_local"].isna()
if bad_dt.any():
    dt_naive = pd.to_datetime(train_df.loc[bad_dt, "pickup_datetime"], errors="coerce")
    train_df.loc[bad_dt, "pickup_datetime_local"] = dt_naive.dt.tz_localize(
        "America/New_York", nonexistent="NaT", ambiguous="NaT"
    )



## === cell 6
train_df["year"] = train_df["pickup_datetime_local"].dt.year
train_df["month"] = train_df["pickup_datetime_local"].dt.month
train_df["day"] = train_df["pickup_datetime_local"].dt.day
train_df["hour"] = train_df["pickup_datetime_local"].dt.hour
train_df["minute"] = train_df["pickup_datetime_local"].dt.minute



## === cell 7
print("Old size: %d" % len(train_df))
train_df = train_df.dropna(how="any", axis="rows")
print("New size: %d" % len(train_df))



## === cell 8
new_york_lat = 40
new_york_long = -74
train_df.describe()



## === cell 9
train_df["passenger_count"] = pd.to_numeric(
    train_df["passenger_count"], errors="coerce"
)
pc_med_train = float(train_df["passenger_count"].median())
train_df["passenger_count"] = (
    train_df["passenger_count"].fillna(pc_med_train).clip(1, 6)
)

fare_cond = (train_df["fare_amount"] > 0) & (train_df["fare_amount"] <= 200)
passenger_cond = (train_df["passenger_count"] >= 1) & (train_df["passenger_count"] <= 6)
dist_cond = (train_df["distance"] > 0) & (train_df["distance"] <= 100)

geo_cond = (
    train_df["pickup_longitude"].between(-74.5, -73.5)
    & train_df["dropoff_longitude"].between(-74.5, -73.5)
    & train_df["pickup_latitude"].between(40.3, 41.2)
    & train_df["dropoff_latitude"].between(40.3, 41.2)
)

print("Old size: %d" % len(train_df))
train_df = train_df[fare_cond & passenger_cond & dist_cond & geo_cond].copy()
print("New size: %d" % len(train_df))

fare_per_km = train_df["fare_amount"] / (train_df["distance"] + 1e-6)
fpk_fixed_cond = (fare_per_km >= 0.5) & (fare_per_km <= 50.0)
train_df = train_df[fpk_fixed_cond].copy()

fare_per_km = train_df["fare_amount"] / (train_df["distance"] + 1e-6)
fpk_lo, fpk_hi = fare_per_km.quantile([0.005, 0.995]).astype(float).values
train_df = train_df[(fare_per_km >= fpk_lo) & (fare_per_km <= fpk_hi)].copy()



## === cell 10
train_df.describe()



## === cell 11
for col in {
    "pickup_latitude",
    "pickup_longitude",
    "dropoff_latitude",
    "dropoff_longitude",
}:
    train_df["rough" + col] = train_df[col].round(2)



## === cell 12
a = train_df.groupby(["roughpickup_latitude", "roughpickup_longitude"])[
    ["pickup_latitude", "pickup_longitude"]
].agg(["mean", "count"])
a.columns = ["_".join(c) for c in a.columns.to_flat_index()]
a.head()
a.sort_values(
    ["pickup_latitude_count", "pickup_longitude_count"], ascending=False
).head()



## === cell 13
b = train_df.groupby(["roughdropoff_latitude", "roughdropoff_longitude"])[
    ["dropoff_latitude", "dropoff_longitude"]
].agg(["mean", "count"])
b.columns = ["_".join(c) for c in b.columns.to_flat_index()]
b.head()
b.sort_values(
    ["dropoff_latitude_count", "dropoff_longitude_count"], ascending=False
).head(n=20)



## === cell 14
b[b["dropoff_latitude_mean"] < 40.7].sort_values(
    ["dropoff_latitude_count", "dropoff_longitude_count"], ascending=False
).head()



## === cell 15
b["dropoff_latitude_count"].plot.hist()
plt.title("occurance of counts of rough dropoff locations")
plt.yscale("log")



## === cell 16
a["pickup_latitude_count"].plot.hist()
plt.title("occurance of counts of rough dropoff locations")
plt.yscale("log")



## === cell 17
a[a["pickup_latitude_count"] > 1000].shape
a.shape



## === cell 18
a = (
    a[["pickup_latitude_count"]]
    .reset_index()
    .rename(columns={"pickup_latitude_count": "pickup_busyness"})
)
b = (
    b[["dropoff_latitude_count"]]
    .reset_index()
    .rename(columns={"dropoff_latitude_count": "dropoff_busyness"})
)



## === cell 19
for c in [
    "roughpickup_latitude",
    "roughpickup_longitude",
    "roughdropoff_latitude",
    "roughdropoff_longitude",
]:
    train_df[c] = train_df[c].astype(np.float64)



## === cell 20
train_df = pd.merge(
    train_df, a, how="left", on=["roughpickup_latitude", "roughpickup_longitude"]
)
train_df = pd.merge(
    train_df, b, how="left", on=["roughdropoff_latitude", "roughdropoff_longitude"]
)



## === cell 21
pickup_med = float(train_df["pickup_busyness"].median())
dropoff_med = float(train_df["dropoff_busyness"].median())
train_df["pickup_busyness"] = train_df["pickup_busyness"].fillna(pickup_med)
train_df["dropoff_busyness"] = train_df["dropoff_busyness"].fillna(dropoff_med)

train_df.head()




## === cell 22
def airport_flag(df):
    airports = np.array(
        [
            [40.6413, -73.7781],  # JFK
            [40.7769, -73.8740],  # LGA
            [40.6895, -74.1745],  # EWR
        ],
        dtype=np.float64,
    )

    p_lat = df["pickup_latitude"].to_numpy(dtype=np.float64)
    p_lon = df["pickup_longitude"].to_numpy(dtype=np.float64)
    d_lat = df["dropoff_latitude"].to_numpy(dtype=np.float64)
    d_lon = df["dropoff_longitude"].to_numpy(dtype=np.float64)

    min_pick = np.full(len(df), np.inf, dtype=np.float64)
    min_drop = np.full(len(df), np.inf, dtype=np.float64)
    for lat, lon in airports:
        min_pick = np.minimum(min_pick, haversine_np(p_lon, p_lat, lon, lat))
        min_drop = np.minimum(min_drop, haversine_np(d_lon, d_lat, lon, lat))

    return ((min_pick <= 2.0) | (min_drop <= 2.0)).astype(np.int8)


train_df["airport"] = airport_flag(train_df)



## === cell 23
train_df["abs_lat_diff"] = (
    train_df["pickup_latitude"] - train_df["dropoff_latitude"]
).abs()
train_df["abs_lon_diff"] = (
    train_df["pickup_longitude"] - train_df["dropoff_longitude"]
).abs()



## === cell 24
from sklearn.linear_model import LinearRegression

baseline_feats = ["distance", "passenger_count"]
X_base = train_df[baseline_feats].to_numpy(dtype=np.float64)
y_base = train_df["fare_amount"].to_numpy(dtype=np.float64)

base_model = LinearRegression(n_jobs=None)
base_model.fit(X_base, y_base)

train_df["base_fare_pred"] = base_model.predict(X_base)
train_df["base_fare_pred"] = np.clip(
    train_df["base_fare_pred"].to_numpy(dtype=np.float64), 2.5, None
)




## === cell 25
def signed_log1p(x):
    x = np.asarray(x, dtype=np.float64)
    return np.sign(x) * np.log1p(np.abs(x))


def signed_expm1(y):
    y = np.asarray(y, dtype=np.float64)
    return np.sign(y) * np.expm1(np.abs(y))




## === cell 26
X = train_df[
    [
        "distance",
        "year",
        "month",
        "day",
        "hour",
        "pickup_busyness",
        "dropoff_busyness",
        "airport",
        "abs_lat_diff",
        "abs_lon_diff",
    ]
].values

residual = train_df["fare_amount"].to_numpy(dtype=np.float64) - train_df[
    "base_fare_pred"
].to_numpy(dtype=np.float64)
Y = signed_log1p(residual)



## === cell 27
from sklearn.ensemble import RandomForestRegressor



## === cell 28
kwargs = {
    "bootstrap": True,
    "max_depth": None,
    "max_features": 3,
    "min_samples_leaf": 9,
    "min_samples_split": 2,
}

rand_regr = RandomForestRegressor(n_estimators=20, random_state=42, n_jobs=-1, **kwargs)



## === cell 29
rand_regr.fit(X, Y)



## === cell 30
y_pred = rand_regr.predict(X)
rmse_log = (np.mean((Y - y_pred) ** 2.0)) ** 0.5
print("RMSE on training (signed log1p residual space):", rmse_log)



## === cell 31
rand_regr.score(X, Y)



## === cell 32
test_df = pd.read_csv("../input/test.csv")



## === cell 33
for c in coord_cols:
    test_df[c] = pd.to_numeric(test_df[c], errors="coerce")
zero_coord_t = (test_df["pickup_longitude"] == 0) & (test_df["pickup_latitude"] == 0)
zero_coord_t |= (test_df["dropoff_longitude"] == 0) & (test_df["dropoff_latitude"] == 0)
test_df.loc[zero_coord_t, coord_cols] = np.nan



## === cell 34
test_df["distance"] = haversine_np(
    test_df["pickup_longitude"],
    test_df["pickup_latitude"],
    test_df["dropoff_longitude"],
    test_df["dropoff_latitude"],
)



## === cell 35
test_df["pickup_datetime"] = pd.to_datetime(
    test_df["pickup_datetime"], errors="coerce", utc=True
)
test_df["pickup_datetime_local"] = test_df["pickup_datetime"].dt.tz_convert(
    "America/New_York"
)
bad_dt_t = test_df["pickup_datetime_local"].isna()
if bad_dt_t.any():
    dt_naive_t = pd.to_datetime(
        test_df.loc[bad_dt_t, "pickup_datetime"], errors="coerce"
    )
    test_df.loc[bad_dt_t, "pickup_datetime_local"] = dt_naive_t.dt.tz_localize(
        "America/New_York", nonexistent="NaT", ambiguous="NaT"
    )



## === cell 36
test_df["year"] = test_df["pickup_datetime_local"].dt.year
test_df["month"] = test_df["pickup_datetime_local"].dt.month
test_df["day"] = test_df["pickup_datetime_local"].dt.day
test_df["hour"] = test_df["pickup_datetime_local"].dt.hour
test_df["minute"] = test_df["pickup_datetime_local"].dt.minute



## === cell 37
for col in {
    "pickup_latitude",
    "pickup_longitude",
    "dropoff_latitude",
    "dropoff_longitude",
}:
    test_df["rough" + col] = test_df[col].round(2)



## === cell 38
for c in [
    "roughpickup_latitude",
    "roughpickup_longitude",
    "roughdropoff_latitude",
    "roughdropoff_longitude",
]:
    test_df[c] = test_df[c].astype(np.float64)

test_df = pd.merge(
    test_df, a, how="left", on=["roughpickup_latitude", "roughpickup_longitude"]
)
test_df = pd.merge(
    test_df, b, how="left", on=["roughdropoff_latitude", "roughdropoff_longitude"]
)



## === cell 39
pickup_med = float(train_df["pickup_busyness"].median())
dropoff_med = float(train_df["dropoff_busyness"].median())

test_df["pickup_busyness"] = test_df["pickup_busyness"].fillna(pickup_med)
test_df["dropoff_busyness"] = test_df["dropoff_busyness"].fillna(dropoff_med)

test_df["distance"] = test_df["distance"].fillna(train_df["distance"].median())
test_df["year"] = test_df["year"].fillna(int(train_df["year"].median()))
test_df["month"] = test_df["month"].fillna(int(train_df["month"].median()))
test_df["day"] = test_df["day"].fillna(int(train_df["day"].median()))
test_df["hour"] = test_df["hour"].fillna(int(train_df["hour"].median()))

test_df["airport"] = airport_flag(test_df)

test_df["abs_lat_diff"] = (
    test_df["pickup_latitude"] - test_df["dropoff_latitude"]
).abs()
test_df["abs_lon_diff"] = (
    test_df["pickup_longitude"] - test_df["dropoff_longitude"]
).abs()
test_df["abs_lat_diff"] = test_df["abs_lat_diff"].fillna(
    train_df["abs_lat_diff"].median()
)
test_df["abs_lon_diff"] = test_df["abs_lon_diff"].fillna(
    train_df["abs_lon_diff"].median()
)



## === cell 40
X_to_pred = test_df[
    [
        "distance",
        "year",
        "month",
        "day",
        "hour",
        "pickup_busyness",
        "dropoff_busyness",
        "airport",
        "abs_lat_diff",
        "abs_lon_diff",
    ]
].values

pc_med = float(train_df["passenger_count"].median())
test_df["passenger_count"] = pd.to_numeric(test_df["passenger_count"], errors="coerce")
test_df["passenger_count"] = test_df["passenger_count"].fillna(pc_med).clip(1, 6)

X_base_test = test_df[baseline_feats].to_numpy(dtype=np.float64)
base_pred_test = base_model.predict(X_base_test)
base_pred_test = np.clip(base_pred_test, 2.5, None)

y_pred_signed_log_res = rand_regr.predict(X_to_pred)
y_pred_res = signed_expm1(y_pred_signed_log_res)

y_pred = base_pred_test + y_pred_res

upper_clip = float(
    np.nanpercentile(train_df["fare_amount"].to_numpy(dtype=np.float64), 99.9)
)
y_pred = np.where(
    np.isfinite(y_pred), y_pred, float(np.nanmedian(train_df["fare_amount"]))
)
y_pred = np.clip(y_pred, 2.5, upper_clip)



## === cell 41
submission = pd.DataFrame(
    {"key": test_df.key, "fare_amount": y_pred}, columns=["key", "fare_amount"]
)
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print("Pred stats:", pd.Series(y_pred).describe())
