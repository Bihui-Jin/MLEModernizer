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

bayesian-optimization==3.1.0
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

3.21281

# 6. Current score

6.56303

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 5.99309) has done: 'You’re currently far worse than the target (RMSE 6.19 vs 3.21, lower is better), so we make a few minimal changes that reliably reduce RMSE without changing the overall modeling approach. The biggest issue is that your `dist()` feature uses Manhattan distance in degrees, which is a poor proxy for real taxi distance; switching to a standard haversine distance (in km) keeps the same feature idea but makes it physically meaningful and typically cuts error substantially. We also ensure the train/test split is reproducible and add a standard XGBoost regression objective/eval metric plus a small learning-rate adjustment paired with more boosting rounds (same algorithm, same training style) to improve generalization toward the target. Finally, we keep the submission format identical and still write `submission.csv`.'
- What this solution (achieved 5.99194) has done: 'We’re far worse than the target (RMSE 5.99 vs 3.21, lower is better), so the smallest reliable gains come from fixing feature issues rather than changing the model type. Your `transform()` currently creates “distance_to_center” using only pickup coordinates (it mistakenly uses NYC center as the “pickup” and the pickup point as the “dropoff”), which corrupts an important feature; we correct that to compute distance from pickup to center. We also clip negative predictions to 0 (fares can’t be negative) to reduce RMSE from outlier predictions without altering the training objective. Finally, we ensure the test read uses the same columns as train (and keep the submission schema unchanged) to avoid subtle train/test feature mismatch.'
- What this solution (achieved 6.95382) has done: 'Your score is still far above the target (RMSE 5.99 vs 3.21, lower is better), so we make small, metric-aligned fixes that typically yield large RMSE drops without changing the core “engineered geo/time features + XGBoost regressor” approach. The biggest remaining issue is that the model is trained on raw coordinates, which encourages spurious splits and hurts generalization; we keep those columns but add standard coordinate sanity filtering and a simple `abs()` version of your lat/long deltas to remove sign ambiguity while preserving the same feature idea. We also train using an explicit validation set with `evals` (same training loop style) so you can see generalization and avoid silent overfitting; prediction and submission format stay identical. Finally, we clip extreme fares in training a bit tighter (still within typical competition practice) to reduce the impact of outliers on RMSE.'
- What this solution (achieved 5.94241) has done: 'You’re far worse than the target (RMSE 6.95 vs 3.21, lower is better), so we should make small, high-impact fixes that keep your “engineered geo/time features + XGBoost regressor” core intact. The biggest remaining issue is that you never remove obviously-bad coordinate cases like identical pickup/dropoff (often bad data) and extreme trip distances, which can heavily inflate RMSE; we add two simple, standard filters on training only. Next, we add a single missing but very common feature consistent with your existing logic: the bearing (direction) between pickup and dropoff, which complements distance without changing the model type. Finally, we keep the same training loop, but add a very light regularization (`reg_lambda`) and enable histogram tree method for speed/stability; submission writing stays identical.'
- What this solution (achieved 5.91695) has done: 'You’re still far above the target (RMSE 5.94 vs 3.21, lower is better), so the most “minimal but high-impact” move is to fix a key data issue: `pickup_datetime` parsing currently coerces seconds/microseconds to NaT for many rows, silently dropping lots of valid training data and harming generalization. I switch to robust datetime parsing (no fixed format, still UTC) to keep more rows, and I add two standard NYC-taxi features (straight-line distance plus a log1p(distance)) that preserve your existing feature-engineering approach but usually reduce RMSE meaningfully. Finally, I keep the same XGBoost training loop/parameters, but add a small amount of regularization via `min_child_weight`/`subsample`/`colsample_bytree` already present (unchanged) and ensure train/test feature columns match exactly to avoid subtle mismatches.'
- What this solution (achieved 5.99455) has done: 'Your RMSE is much worse than the target (5.91695 vs 3.21281, lower is better), so the smallest high-impact move is to fix feature informativeness without changing the model type or training loop. I keep your engineered geo/time features and XGBoost training as-is, but add two standard taxi-fare features that usually cut RMSE substantially: (1) Haversine distance to a few more Manhattan-related anchor points (Times Sq + Wall St) and (2) a simple “NYC grid” distance proxy (lat/long km components), which better matches how taxis drive than straight-line distance alone. I also align the train filtering with those features by removing rows with impossible datetimes (year outside a sensible range) to reduce noise. Submission writing and column schema stay identical.'
- What this solution (achieved 4.87046) has done: 'Your current gap to target is large (RMSE 5.99 vs 3.21; lower is better), so the smallest reliable improvement is to fix data-quality issues that inflate error without changing your core approach (same engineered geo/time features + XGBoost regressor + same training loop). I add two standard NYC Taxi filters: remove rows with identical pickup/dropoff coordinates and remove rows with known-bad coordinate value `0` (a common corruption), which reduces noisy labels and improves generalization. I also add two minimal, metric-aligned feature tweaks that preserve your feature-engineering style: `log1p(manhattan_km)` and `sin/cos` encoding of `hour` to capture cyclic time effects without changing the model. Submission writing/format remains identical and a valid `submission.csv` be produced.'
- What this solution (achieved 6.06831) has done: 'Your current RMSE (4.87046) is still far worse than the target (3.21281; lower is better), so we should make small, high-impact fixes that reduce noise and make the model generalize better without changing the overall “engineered geo/time features + XGBoost regressor” approach. The biggest issue is train/test distribution mismatch: you aggressively filter bad coordinates/distances in training but not in test (you even forward/back-fill coordinates), which can create unrealistic test features and inflate error; we stop imputing coordinates and instead clip them into the same plausible NYC bounding box used for training. Next, we add two very standard, minimal feature tweaks that preserve your existing feature-engineering style and often reduce RMSE materially: (1) weekend indicator and (2) cyclic encoding for day-of-week (sin/cos). Finally, we stabilize training slightly by using a modest subsample (<1) and adding a small `min_split_loss`/`gamma` regularization (same model, same training loop) to reduce overfitting toward the target.'
- What this solution (achieved 6.21974) has done: 'We’re still far above the target RMSE (6.07 vs 3.21; lower is better), so the smallest reliable improvement is to fix train/test mismatch caused by clipping and NaT-imputation in the test set, which distorts engineered distance features and hurts generalization. I apply the same core “filter/drop bad rows” philosophy to test by leaving coordinates as-is (no clipping) and instead filling only truly missing values with training medians, and I avoid forcing NaT datetimes to a constant date (use median timestamp-derived features instead). I also add two tiny, standard, core-logic-preserving geo features (pickup/dropoff Haversine-to-center) that reuse your existing distance function and typically reduce RMSE materially. Everything else (XGBoost regressor, training loop, loss/metric, submission schema/path) stays the same.'
- What this solution (achieved 6.08399) has done: 'Your RMSE is far worse than the target (6.22 vs 3.21, lower is better), so we should focus on small, high-impact fixes that reduce noisy training signal without changing the core “engineered geo/time features + XGBoost regressor” approach. The biggest likely issue is remaining label noise from bad/implausible rides; we add a couple of standard NYC-taxi filters (fare-per-km sanity and passenger_count==0 removal already exists) that typically drop RMSE materially. We also switch from a random split to a time-based split (same evaluation semantics, still RMSE) to reduce leakage and make the model generalize better to the test’s time distribution. Finally, we keep the same feature set and model, but train with early-stopping removed (per your constraint) and instead set `num_boost_round` to use the best iteration found by a fixed validation watchlist (still the same training loop) to avoid overfitting while staying deterministic.'
- What this solution (achieved 6.56303) has done: 'Your current RMSE (6.08399) is far worse than the target (3.21281, lower is better), so we should make one small but high-impact correction that keeps the same core approach (engineered geo/time features + XGBoost). The main issue is that you compute the time-based split *after* `transform()`, but `transform()` overwrites `day/month/year` with `dt.day/dt.month/dt.year`, and `day` is day-of-month (1–31), so your “time-based” ordering is effectively wrong and causes a bad/unstable validation regime and poorer generalization. I preserve your split idea but sort by a proper timestamp-derived integer (`pickup_datetime` as int64) captured before dropping the datetime column. Everything else (features, model params/training loop, submission format/path) stays the same.'

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



## === cell 1
df = pd.read_csv(
    "../input/train.csv",
    nrows=2_000_000,
    usecols=[
        "fare_amount",
        "pickup_datetime",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
    ],
)



## === cell 2
df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], utc=True, errors="coerce")



## === cell 3
df.dropna(how="any", axis="rows", inplace=True)

mask = df["pickup_longitude"].between(-75, -73)
mask &= df["dropoff_longitude"].between(-75, -73)
mask &= df["pickup_latitude"].between(40, 42)
mask &= df["dropoff_latitude"].between(40, 42)
mask &= df["passenger_count"].between(1, 8)  # drop 0-passenger noise
mask &= df["fare_amount"].between(2.5, 200)  # reduce outlier impact on RMSE

mask &= df["pickup_longitude"].between(-180, 180)
mask &= df["dropoff_longitude"].between(-180, 180)
mask &= df["pickup_latitude"].between(-90, 90)
mask &= df["dropoff_latitude"].between(-90, 90)

mask &= df["pickup_datetime"].dt.year.between(2009, 2016)

mask &= ~(
    (df["pickup_longitude"] == 0)
    | (df["pickup_latitude"] == 0)
    | (df["dropoff_longitude"] == 0)
    | (df["dropoff_latitude"] == 0)
)
mask &= ~(
    (df["pickup_longitude"] == df["dropoff_longitude"])
    & (df["pickup_latitude"] == df["dropoff_latitude"])
)

df = df[mask].copy()




## === cell 4
def dist(pickup_lat, pickup_long, dropoff_lat, dropoff_long):
    R = 6371.0
    lat1 = np.radians(pickup_lat)
    lon1 = np.radians(pickup_long)
    lat2 = np.radians(dropoff_lat)
    lon2 = np.radians(dropoff_long)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * (
        np.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * np.arcsin(np.sqrt(a))
    return R * c


def bearing(pickup_lat, pickup_long, dropoff_lat, dropoff_long):
    lat1 = np.radians(pickup_lat)
    lon1 = np.radians(pickup_long)
    lat2 = np.radians(dropoff_lat)
    lon2 = np.radians(dropoff_long)
    dlon = lon2 - lon1
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    brng = np.arctan2(y, x)
    return brng  # radians in [-pi, pi]




## === cell 5
def transform(data):
    data["hour"] = data["pickup_datetime"].dt.hour
    data["day"] = data["pickup_datetime"].dt.day
    data["month"] = data["pickup_datetime"].dt.month
    data["year"] = data["pickup_datetime"].dt.year

    data["dayofweek"] = data["pickup_datetime"].dt.dayofweek
    data["is_weekend"] = (data["dayofweek"] >= 5).astype(np.int8)
    data["dow_sin"] = np.sin(2.0 * np.pi * data["dayofweek"] / 7.0)
    data["dow_cos"] = np.cos(2.0 * np.pi * data["dayofweek"] / 7.0)

    data["hour_sin"] = np.sin(2.0 * np.pi * data["hour"] / 24.0)
    data["hour_cos"] = np.cos(2.0 * np.pi * data["hour"] / 24.0)

    data = data.drop("pickup_datetime", axis=1)

    nyc = (-74.0063889, 40.7141667)
    jfk = (-73.7822222222, 40.6441666667)
    ewr = (-74.175, 40.69)
    lgr = (-73.87, 40.77)

    times_sq = (-73.9855, 40.7580)
    wall_st = (-74.0090, 40.7060)

    data["pickup_distance_to_center"] = dist(
        data["pickup_latitude"], data["pickup_longitude"], nyc[1], nyc[0]
    )
    data["dropoff_distance_to_center"] = dist(
        data["dropoff_latitude"], data["dropoff_longitude"], nyc[1], nyc[0]
    )

    data["pickup_distance_to_jfk"] = dist(
        jfk[1], jfk[0], data["pickup_latitude"], data["pickup_longitude"]
    )
    data["dropoff_distance_to_jfk"] = dist(
        jfk[1], jfk[0], data["dropoff_latitude"], data["dropoff_longitude"]
    )
    data["pickup_distance_to_ewr"] = dist(
        ewr[1], ewr[0], data["pickup_latitude"], data["pickup_longitude"]
    )
    data["dropoff_distance_to_ewr"] = dist(
        ewr[1], ewr[0], data["dropoff_latitude"], data["dropoff_longitude"]
    )
    data["pickup_distance_to_lgr"] = dist(
        lgr[1], lgr[0], data["pickup_latitude"], data["pickup_longitude"]
    )
    data["dropoff_distance_to_lgr"] = dist(
        lgr[1], lgr[0], data["dropoff_latitude"], data["dropoff_longitude"]
    )

    data["pickup_distance_to_times_sq"] = dist(
        times_sq[1], times_sq[0], data["pickup_latitude"], data["pickup_longitude"]
    )
    data["dropoff_distance_to_times_sq"] = dist(
        times_sq[1], times_sq[0], data["dropoff_latitude"], data["dropoff_longitude"]
    )
    data["pickup_distance_to_wall_st"] = dist(
        wall_st[1], wall_st[0], data["pickup_latitude"], data["pickup_longitude"]
    )
    data["dropoff_distance_to_wall_st"] = dist(
        wall_st[1], wall_st[0], data["dropoff_latitude"], data["dropoff_longitude"]
    )

    data["long_dist"] = data["pickup_longitude"] - data["dropoff_longitude"]
    data["lat_dist"] = data["pickup_latitude"] - data["dropoff_latitude"]

    data["abs_long_dist"] = np.abs(data["long_dist"])
    data["abs_lat_dist"] = np.abs(data["lat_dist"])

    data["dist"] = dist(
        data["pickup_latitude"],
        data["pickup_longitude"],
        data["dropoff_latitude"],
        data["dropoff_longitude"],
    )
    data["log_dist"] = np.log1p(data["dist"])

    data["bearing"] = bearing(
        data["pickup_latitude"],
        data["pickup_longitude"],
        data["dropoff_latitude"],
        data["dropoff_longitude"],
    )

    km_per_deg_lat = 111.32
    km_per_deg_lon = 111.32 * np.cos(
        np.radians((data["pickup_latitude"] + data["dropoff_latitude"]) / 2.0)
    )
    data["delta_lat_km"] = data["abs_lat_dist"] * km_per_deg_lat
    data["delta_lon_km"] = data["abs_long_dist"] * km_per_deg_lon
    data["manhattan_km"] = data["delta_lat_km"] + data["delta_lon_km"]

    data["log_manhattan_km"] = np.log1p(data["manhattan_km"])

    return data




## === cell 6
df["raw_dist_km"] = dist(
    df["pickup_latitude"],
    df["pickup_longitude"],
    df["dropoff_latitude"],
    df["dropoff_longitude"],
)

mask2 = df["raw_dist_km"].between(0.2, 80)

fare_per_km = df["fare_amount"] / (df["raw_dist_km"] + 1e-3)
mask2 &= fare_per_km.between(0.5, 50.0)

df = df[mask2].copy()
df.drop(columns=["raw_dist_km"], inplace=True)

train_medians = {
    "pickup_longitude": df["pickup_longitude"].median(),
    "pickup_latitude": df["pickup_latitude"].median(),
    "dropoff_longitude": df["dropoff_longitude"].median(),
    "dropoff_latitude": df["dropoff_latitude"].median(),
    "passenger_count": int(df["passenger_count"].median()),
}
dt = df["pickup_datetime"]
train_time_medians = {
    "hour": int(dt.dt.hour.median()),
    "day": int(dt.dt.day.median()),
    "month": int(dt.dt.month.median()),
    "year": int(dt.dt.year.median()),
    "dayofweek": int(dt.dt.dayofweek.median()),
}

df["pickup_ts_int"] = df["pickup_datetime"].view("int64")

df = transform(df)



## === cell 7
import xgboost as xgb
from sklearn.metrics import mean_squared_error



## === cell 8
X = df.drop("fare_amount", axis=1)
y = df["fare_amount"]

order = np.argsort(df["pickup_ts_int"].values)
X_sorted = X.iloc[order]
y_sorted = y.iloc[order]

split_idx = int(len(X_sorted) * 0.75)
X_train, X_val = X_sorted.iloc[:split_idx], X_sorted.iloc[split_idx:]
y_train, y_val = y_sorted.iloc[:split_idx], y_sorted.iloc[split_idx:]

dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)



## === cell 9
params = {
    "max_depth": 7,
    "gamma": 0.2,
    "colsample_bytree": 0.3,
    "min_child_weight": 3.0,
    "subsample": 0.8,
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "eta": 0.05,
    "seed": 42,
    "reg_lambda": 1.0,
    "tree_method": "hist",
}



## === cell 10
watchlist = [(dtrain, "train"), (dval, "val")]
model2 = xgb.train(
    params, dtrain, num_boost_round=1200, evals=watchlist, verbose_eval=100
)

y_val_pred = model2.predict(dval)
y_train_pred = model2.predict(dtrain)

print(np.sqrt(mean_squared_error(y_val, y_val_pred)))
print(np.sqrt(mean_squared_error(y_train, y_train_pred)))



## === cell 11
import matplotlib.pyplot as plt

fscores = pd.DataFrame(
    {"X": list(model2.get_fscore().keys()), "Y": list(model2.get_fscore().values())}
)
fscores.sort_values(by="Y").plot.bar(x="X")



## === cell 12
test = pd.read_csv(
    "../input/test.csv",
    usecols=[
        "key",
        "pickup_datetime",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
    ],
).set_index("key")

test["pickup_datetime"] = pd.to_datetime(
    test["pickup_datetime"], utc=True, errors="coerce"
)

for col in [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
]:
    test[col] = test[col].astype("float64")
    test[col] = test[col].fillna(train_medians[col])

test["passenger_count"] = pd.to_numeric(
    test["passenger_count"], errors="coerce"
).fillna(train_medians["passenger_count"])
test["passenger_count"] = test["passenger_count"].clip(lower=1, upper=8)

nat_mask = test["pickup_datetime"].isna()
if nat_mask.any():
    y0 = train_time_medians["year"]
    m0 = max(1, min(12, train_time_medians["month"]))
    d0 = 15
    h0 = max(0, min(23, train_time_medians["hour"]))
    fill_ts = pd.Timestamp(year=y0, month=m0, day=d0, hour=h0, tz="UTC")
    test.loc[nat_mask, "pickup_datetime"] = fill_ts

test = transform(test)
test = test.reindex(columns=X.columns)

dtest = xgb.DMatrix(test)
y_pred_test = model2.predict(dtest)

y_pred_test = np.clip(y_pred_test, 0, None)



## === cell 13
holdout = pd.DataFrame({"key": test.index, "fare_amount": y_pred_test})
holdout.to_csv("submission.csv", index=False)
print("Wrote submission.csv with", len(holdout), "rows")
