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
geopy==2.4.1
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

4.04952

# 6. Current score

7.77473

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 5.38488) has done: 'Diagnosis: Cell 20 crashes because `geopy.distance.VincentyDistance` was removed from modern `geopy` versions (it’s no longer an attribute of `geopy.distance`). The training set distance in cell 18 already uses `geopy.distance.geodesic(...).km`, so the test set should compute distance the same way to preserve feature semantics.  
Patch summary: Replace the deprecated `VincentyDistance` call with `geopy.distance.geodesic` while keeping the same per-row `DataFrame.apply(..., axis=1)` structure and producing the same `test['distance']` column.  
Updated cells: Only cell 20 is changed.  
Compatibility notes for cell k+1: `test['distance']` is still created and the latitude/longitude columns are still present until cell 21 drops them, so cell 21 remains unchanged and works.  
Assumptions: `geopy.distance.geodesic` is available in the installed `geopy==2.4.1` and lat/long columns are numeric (or coercible) as loaded.'
- What this solution (achieved 5.34259) has done: 'Your current score (5.38488) is worse than the target (4.04952), so we should improve generalization with minimal semantic changes. The biggest low-risk gain here is fixing the datetime feature engineering: you’re parsing `key` instead of `pickup_datetime`, which makes the time features mostly invalid/noisy; switching to `pickup_datetime` preserves the same feature set but makes it meaningful. I also make the train/validation split deterministic (adds stability, doesn’t change core logic) and clip negative fare predictions to 0 (a valid constraint that typically reduces RMSE tail errors). Everything else (data size, distance feature, LightGBM model/params, training flow, submission format/path) stays the same.'
- What this solution (achieved 5.77905) has done: 'Your current RMSE (5.34259) is worse than the target (4.04952), so we should make a small, safe improvement that reduces error without changing the model or feature set. The biggest remaining issue is that you never remove clearly invalid targets (negative/zero fares) and you only validate coordinate ranges, not NYC-area plausibility; both inject heavy noise that LightGBM can’t learn well from and typically worsen RMSE. I add minimal train-side filtering to keep fares positive and restrict pickup/dropoff coordinates to a reasonable NYC bounding box (applied to train only so we don’t drop test rows), keeping the same distance + datetime-derived features and the same LightGBM setup. This should move RMSE downward toward your target while preserving the core logic and still producing a valid `submission.csv`.'
- What this solution (achieved 5.76037) has done: 'We need to move RMSE down from 5.77905 toward 4.04952 (lower is better), with minimal changes and no architecture/training-loop changes. The biggest low-risk issue left is data quality: even after bounding-box filtering, the training set still contains extreme outlier fares/distances that inflate RMSE; lightly trimming only the training rows (not test) typically reduces noise and improves generalization without changing features/model. I add a small, standard outlier filter on `fare_amount` and `distance` (after computing distance) and ensure datetime feature extraction is robust for NaT values, keeping the same feature set and LightGBM configuration. Submission writing stays identical and still produces `submission.csv`.'
- What this solution (achieved 5.75952) has done: 'Your RMSE (5.76037) is still well above the target (4.04952), so we should make small, legitimate quality fixes that reduce noise without changing the model/feature set. The biggest remaining issue is that you fill all missing values in `train` with 0 before filtering, which can create invalid coordinates/timestamps and then slip through parts of the pipeline as “valid” rows; instead we only drop rows missing essential fields used for features/target (train only). Next, we apply the same NYC bounding-box sanity filter to `test` as well (without dropping rows): out-of-box rows get their coordinates clipped so distance isn’t wildly wrong; this keeps submission row count unchanged and typically reduces extreme prediction errors. Finally, we clip negative passenger_count to 0 and cap very large passenger_count at a reasonable maximum (train+test) to reduce outlier influence while keeping the same core feature set and LightGBM training loop.'
- What this solution (achieved 5.77427) has done: 'We need to move RMSE down from 5.75952 toward the target 4.04952 (lower is better) while keeping your same LightGBM model and the same feature set. The smallest high-impact fix is to make your “distance” feature consistent with typical NYC taxi solutions: geodesic distance is overkill/noisy at city scale and introduces extra error; switching to a fast haversine km computation preserves the same “distance” feature semantics (km between pickup/dropoff) but is usually more accurate for this dataset. I also ensure `key2` parsing doesn’t leak NaT-derived zeros unnecessarily by dropping rows where `key2` is NaT in train (train-only), since your time features depend on it and NaT rows add label noise. Everything else (rows loaded, filters, feature columns, model params/training loop, and submission format/path) stays the same and still writes `submission.csv`.'
- What this solution (achieved 5.77427) has done: 'Your current RMSE (5.77427) is worse than the target (4.04952), so we should make a small, safe improvement that reduces noise without changing your model, training loop, or feature set. The biggest issue left is that `key2` in the test set can be `NaT`, which makes all time features become 0 for those rows and creates unrealistic patterns at inference; we impute missing `key2` in test with the median train pickup time so the same features remain meaningful. Next, we apply the same “drop rows missing essential fields” logic to `test` (without dropping rows) by filling missing numeric inputs with train medians before clipping, preventing NaNs from turning into NaN distances. Finally, we clip predictions to a reasonable fare range (same as your train filter) to reduce RMSE impact from extreme outliers while keeping evaluation semantics intact.'
- What this solution (achieved 5.77427) has done: 'Your RMSE (5.77427) is much worse than the target (4.04952), so we should make a small, legitimate improvement that reduces noise while keeping your LightGBM model, training loop, and feature set unchanged. The biggest remaining issue is that you filter outliers using `distance` but you never remove rows where the haversine computation becomes NaN (from non-numeric/NaN coords), so those rows can silently pass through earlier filters and then corrupt training. I compute `distance` after coercing coords and then explicitly drop rows with NaN `distance` (train only), and I also remove the now-unneeded `train.fillna(0)` that can fabricate invalid coordinates/times and degrade signal. Everything else (features, params, fit/predict flow, and submission writing) stays the same and still produces `submission.csv`.'
- What this solution (achieved 5.77427) has done: 'We need to move RMSE down from 5.77427 toward 4.04952 (lower is better), so we should improve generalization with the smallest changes that keep your LightGBM model and feature set intact. The biggest issue in your current pipeline is a bug: you drop the latitude/longitude columns from `train` before computing `train_geo_medians`, so your test coordinate imputation silently falls back to constants instead of reflecting the training sample distribution. Fixing that (compute medians from `train` before dropping, and reuse the exact same NYC clipping + haversine distance logic) reduces noisy distance features on test and typically improves RMSE materially without changing the model or training approach. I also make the train/test coordinate coercion consistent by coercing train coords before the NYC bounding-box filter (so string coords don’t get wrongly filtered/dropped), which is a data-quality fix that preserves semantics. Everything else (rows loaded, engineered features, LightGBM params, fit/predict flow, and submission writing) stays the same and still produces `submission.csv`.'
- What this solution (achieved 7.1216) has done: 'Your RMSE (5.77427) is still far above the target (4.04952), so we should reduce label/feature noise with the smallest changes that keep the same feature set and LightGBM training flow. The main issue is your `distance` feature: it ignores NYC’s actual road/grid structure and you’re not giving the model any directionality information; adding the standard minimal “delta” geo features (`abs_lon_diff`, `abs_lat_diff`, `manhattan_dist`) keeps the same approach (simple engineered features + LightGBM) but typically moves RMSE down materially. I also fix a subtle dtype issue: `isocalendar().week` can become `UInt32` and inconsistently cast between train/test; making it explicitly numeric and consistent avoids silent LightGBM handling differences. Finally, I clip predictions to the same fare range you trained on (2.5–250) to reduce RMSE impact from implausible outputs, without changing the model or training loop.'
- What this solution (achieved 7.17794) has done: 'We need to move RMSE down from 7.1216 toward 4.04952 (lower is better), so the smallest helpful changes are to remove a remaining high-noise pattern and align train/test preprocessing without changing the model or training loop. In your current pipeline, you clip `passenger_count` to 0–6 but you keep rows with `passenger_count==0` in training; those are mostly data errors and add label noise, so we drop them in train only (test rows are kept unchanged). Next, your “manhattan_dist” is currently `abs_lon_diff + abs_lat_diff` (degrees), which is poorly scaled; we keep the same feature name but convert it to an approximate km-based Manhattan distance using latitude-dependent scaling (same core feature, just better units). Finally, we make train/test feature engineering perfectly symmetric by applying the exact same km scaling for both, and we keep the same submission writing.'
- What this solution (achieved 7.33548) has done: 'Your current RMSE (7.17794) is much worse than the target (4.04952), so we should reduce train/test noise and a feature-mismatch bug with minimal, semantics-preserving changes. The biggest issue is that `train` still contains rows with `distance==0` (often bad GPS/duplicated coords) and rows where the computed km-based Manhattan distance is implausibly large, both of which inflate RMSE; we filter those in train only using the already-computed features. Next, we make passenger-count preprocessing symmetric by ensuring `test["passenger_count"]` uses the same numeric coercion + clipping and also fill missing passenger_count with the train median (rather than 0), to avoid creating a synthetic “0 passenger” mode at inference. Finally, we fix a small but impactful detail: the model currently sees both `week` and `week_of_year` as identical features; we keep the same column names (so downstream stays unchanged) but make `week_of_year` a true day-of-year-derived week number to remove redundant noise without changing the approach.'
- What this solution (achieved 7.11661) has done: 'You’re far worse than the target (RMSE 7.33548 vs 4.04952; lower is better), so we should make small changes that reduce noise without changing the model/training loop or feature set. The biggest issue is your “distance>0.05km” filter: it deletes a lot of legitimate short NYC trips (which are common) and creates train/test distribution shift; loosening it to only remove true zero/near-zero GPS errors should lower RMSE. Next, your `manhattan_dist` filter upper bound (120 km) is extremely permissive for NYC and lets through many mislabeled/outlier rows; tightening it modestly reduces label noise while keeping semantics. Finally, ensure the engineered geo columns are numeric floats before feature building (prevents any rare dtype contamination), and keep submission creation identical.'
- What this solution (achieved 7.77473) has done: 'Your current RMSE (7.11661) is worse than the target (4.04952), so we should reduce train/test distribution shift with the smallest possible, semantics-preserving changes. The main culprit is your aggressive training filter `manhattan_dist <= 60` while you do not apply any analogous handling to test, causing the model to extrapolate on many test rows; I keep all rows but cap both train and test `distance`/`manhattan_dist` at the same upper bounds instead of dropping those training rows. This keeps the same exact feature set and LightGBM training flow, but makes the feature distributions more aligned and typically lowers RMSE. I also cap (not drop) extremely large `abs_lon_diff/abs_lat_diff` values to reduce outlier leverage, and keep the existing prediction clipping/submission writing unchanged.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use("fivethirtyeight")

import geopy.distance

import os

print(os.listdir("../input"))
import gc

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn import svm
from sklearn.linear_model import SGDRegressor
from sklearn import tree
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler




## === cell 1
def load_Data():
    train = pd.read_csv("../input/train.csv", nrows=10_00_000, low_memory=True)
    test = pd.read_csv("../input/test.csv", nrows=10_00_000, low_memory=True)
    return train, test




## === cell 2
train, test = load_Data()



## === cell 3
train.head(5)



## === cell 4
train.describe()



## === cell 5
train.info()



## === cell 6
train.isnull().sum()



## === cell 7
essential_cols = [
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
train = train.dropna(subset=essential_cols).copy()



## === cell 8
train["key2"] = pd.to_datetime(train["pickup_datetime"], errors="coerce")
train = train.dropna(subset=["key2"]).copy()
train["key2"].head()
train.info()



## === cell 9
test["key2"] = pd.to_datetime(test["pickup_datetime"], errors="coerce")
median_train_key2 = train["key2"].median()
test["key2"] = test["key2"].fillna(median_train_key2)



## === cell 10
train["fare_amount"].plot(kind="box")



## === cell 11
gc.collect()
train.describe()



## === cell 12
print(
    "% of fares above 25$ - {:0.2f}".format(
        train[train["fare_amount"] > 25]["key"].count() * 100 / train["key"].count()
    )
)
print(
    "% of fares above 50$ - {:0.2f}".format(
        train[train["fare_amount"] > 50]["key"].count() * 100 / train["key"].count()
    )
)
print(
    "% of fares above 100$ - {:0.2f}".format(
        train[train["fare_amount"] > 100]["key"].count() * 100 / train["key"].count()
    )
)
print(
    "% of fares below 0$ - {:0.2f}".format(
        train[train["fare_amount"] < 0]["key"].count() * 100 / train["key"].count()
    )
)



## === cell 13
fig, axarr = plt.subplots(2, 2, figsize=(20, 10))

train[~(train["fare_amount"] > 25)]["fare_amount"].plot(kind="box", ax=axarr[0][0])
train[~(train["fare_amount"] > 50)]["fare_amount"].plot(kind="box", ax=axarr[0][1])
train[~(train["fare_amount"] > 100)]["fare_amount"].plot(kind="box", ax=axarr[1][0])
train[~(train["fare_amount"] < 0)]["fare_amount"].plot(kind="box", ax=axarr[1][1])



## === cell 14
train["passenger_count"].plot(kind="box")



## === cell 15
print(
    "Count of invalid pickup latitude",
    train[(train["pickup_latitude"] > 90) | (train["pickup_latitude"] < -90)][
        "pickup_latitude"
    ].count(),
)
print(
    "Count of invalid dropoff latitude",
    train[(train["dropoff_latitude"] > 90) | (train["dropoff_latitude"] < -90)][
        "dropoff_latitude"
    ].count(),
)
print(
    "Count of invalid pickup longitude",
    train[(train["pickup_longitude"] > 180) | (train["pickup_longitude"] < -180)][
        "pickup_longitude"
    ].count(),
)
print(
    "Count of invalid dropoff longitude",
    train[(train["dropoff_longitude"] > 180) | (train["dropoff_longitude"] < -180)][
        "dropoff_longitude"
    ].count(),
)



## === cell 16
print(
    "Count of invalid pickup latitude",
    test[(test["pickup_latitude"] > 90) | (test["pickup_latitude"] < -90)][
        "pickup_latitude"
    ].count(),
)
print(
    "Count of invalid dropoff latitude",
    test[(test["dropoff_latitude"] > 90) | (test["dropoff_latitude"] < -90)][
        "dropoff_latitude"
    ].count(),
)
print(
    "Count of invalid pickup longitude",
    test[(test["pickup_longitude"] > 180) | (test["pickup_longitude"] < -180)][
        "pickup_longitude"
    ].count(),
)
print(
    "Count of invalid dropoff longitude",
    test[(test["dropoff_longitude"] > 180) | (test["dropoff_longitude"] < -180)][
        "dropoff_longitude"
    ].count(),
)



## === cell 17
for df in (train, test):
    df["passenger_count"] = pd.to_numeric(df["passenger_count"], errors="coerce")

train_pc_median = int(
    pd.to_numeric(train["passenger_count"], errors="coerce").dropna().median()
)
test["passenger_count"] = test["passenger_count"].fillna(train_pc_median)

for df in (train, test):
    df["passenger_count"] = df["passenger_count"].fillna(train_pc_median)
    df["passenger_count"] = df["passenger_count"].clip(lower=0, upper=6).astype(int)

train = train[train["passenger_count"] > 0].copy()



## === cell 18
for c in [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
]:
    train[c] = pd.to_numeric(train[c], errors="coerce")

train = train.dropna(
    subset=[
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
    ]
).copy()

train = train[train["fare_amount"] > 0].copy()

nyc_lon_min, nyc_lon_max = -74.5, -72.8
nyc_lat_min, nyc_lat_max = 40.3, 41.2

train = train[
    (train["pickup_longitude"].between(nyc_lon_min, nyc_lon_max))
    & (train["dropoff_longitude"].between(nyc_lon_min, nyc_lon_max))
    & (train["pickup_latitude"].between(nyc_lat_min, nyc_lat_max))
    & (train["dropoff_latitude"].between(nyc_lat_min, nyc_lat_max))
].copy()



## === cell 19
train = train[~((train["pickup_latitude"] > 90) | (train["pickup_latitude"] < -90))]
train = train[~((train["dropoff_latitude"] > 90) | (train["dropoff_latitude"] < -90))]
train = train[~((train["pickup_longitude"] > 180) | (train["pickup_longitude"] < -180))]
train = train[
    ~((train["dropoff_longitude"] > 180) | (train["dropoff_longitude"] < -180))
]




## === cell 20
def haversine_km(lat1, lon1, lat2, lon2):
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(a))
    return 6371.0088 * c  # mean Earth radius in km


train["distance"] = haversine_km(
    train["pickup_latitude"].values,
    train["pickup_longitude"].values,
    train["dropoff_latitude"].values,
    train["dropoff_longitude"].values,
)

train["abs_lon_diff"] = (train["pickup_longitude"] - train["dropoff_longitude"]).abs()
train["abs_lat_diff"] = (train["pickup_latitude"] - train["dropoff_latitude"]).abs()

mid_lat = 0.5 * (train["pickup_latitude"].values + train["dropoff_latitude"].values)
km_per_deg_lat = 110.574
km_per_deg_lon = 111.320 * np.cos(np.radians(mid_lat))
train["manhattan_dist"] = (
    train["abs_lat_diff"].values * km_per_deg_lat
    + train["abs_lon_diff"].values * km_per_deg_lon
)

train = train.dropna(
    subset=["distance", "abs_lon_diff", "abs_lat_diff", "manhattan_dist"]
).copy()



## === cell 21
train = train[
    (train["distance"] > 0.001)
].copy()  # keep only true zero/near-zero GPS errors
train = train[(train["fare_amount"] >= 2.5) & (train["fare_amount"] <= 250)].copy()

DIST_CAP_KM = 100.0
MANH_CAP_KM = 60.0
DELTA_CAP_DEG = 1.0  # very generous given NYC bbox; just prevents pathological diffs

for df in (train,):
    df["distance"] = df["distance"].clip(lower=0.0, upper=DIST_CAP_KM)
    df["manhattan_dist"] = df["manhattan_dist"].clip(lower=0.0, upper=MANH_CAP_KM)
    df["abs_lon_diff"] = df["abs_lon_diff"].clip(lower=0.0, upper=DELTA_CAP_DEG)
    df["abs_lat_diff"] = df["abs_lat_diff"].clip(lower=0.0, upper=DELTA_CAP_DEG)



## === cell 22
geo_cols = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
]
train_geo_medians = {
    c: pd.to_numeric(train[c], errors="coerce").median() for c in geo_cols
}

fallback_geo = {
    "pickup_longitude": -73.985,
    "dropoff_longitude": -73.985,
    "pickup_latitude": 40.758,
    "dropoff_latitude": 40.758,
}
for c in geo_cols:
    if c not in train_geo_medians or pd.isna(train_geo_medians[c]):
        train_geo_medians[c] = fallback_geo[c]



## === cell 23
train.drop(
    ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"],
    axis=1,
    inplace=True,
)



## === cell 24
for col, lo, hi in [
    ("pickup_longitude", nyc_lon_min, nyc_lon_max),
    ("dropoff_longitude", nyc_lon_min, nyc_lon_max),
    ("pickup_latitude", nyc_lat_min, nyc_lat_max),
    ("dropoff_latitude", nyc_lat_min, nyc_lat_max),
]:
    test[col] = pd.to_numeric(test[col], errors="coerce")
    test[col] = test[col].fillna(train_geo_medians[col])
    test[col] = test[col].clip(lower=lo, upper=hi)

test["distance"] = haversine_km(
    test["pickup_latitude"].values,
    test["pickup_longitude"].values,
    test["dropoff_latitude"].values,
    test["dropoff_longitude"].values,
)

test["abs_lon_diff"] = (test["pickup_longitude"] - test["dropoff_longitude"]).abs()
test["abs_lat_diff"] = (test["pickup_latitude"] - test["dropoff_latitude"]).abs()

mid_lat_t = 0.5 * (test["pickup_latitude"].values + test["dropoff_latitude"].values)
km_per_deg_lon_t = 111.320 * np.cos(np.radians(mid_lat_t))
test["manhattan_dist"] = (
    test["abs_lat_diff"].values * 110.574
    + test["abs_lon_diff"].values * km_per_deg_lon_t
)

for c in ["distance", "abs_lon_diff", "abs_lat_diff", "manhattan_dist"]:
    test[c] = pd.to_numeric(test[c], errors="coerce").fillna(0.0)

test["distance"] = test["distance"].clip(lower=0.0, upper=DIST_CAP_KM)
test["manhattan_dist"] = test["manhattan_dist"].clip(lower=0.0, upper=MANH_CAP_KM)
test["abs_lon_diff"] = test["abs_lon_diff"].clip(lower=0.0, upper=DELTA_CAP_DEG)
test["abs_lat_diff"] = test["abs_lat_diff"].clip(lower=0.0, upper=DELTA_CAP_DEG)



## === cell 25
test.drop(
    ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"],
    axis=1,
    inplace=True,
)



## === cell 26
print(
    "% of trips above 25 KM - {:0.2f}".format(
        train[train["distance"] > 25]["key"].count() * 100 / train.count()["key"]
    )
)



## === cell 27
train["year"] = train["key2"].dt.year.fillna(0).astype(int)
train["month"] = train["key2"].dt.month.fillna(0).astype(int)
train["day"] = train["key2"].dt.day.fillna(0).astype(int)
train["day of week"] = train["key2"].dt.weekday.fillna(0).astype(int)
train["hour"] = train["key2"].dt.hour.fillna(0).astype(int)
train["week"] = (
    pd.to_numeric(train["key2"].dt.isocalendar().week, errors="coerce")
    .fillna(0)
    .astype(int)
)
train["day_of_year"] = train["key2"].dt.dayofyear.fillna(0).astype(int)

train["week_of_year"] = ((train["day_of_year"] - 1) // 7 + 1).astype(int)

train["quarter"] = train["key2"].dt.quarter.fillna(0).astype(int)



## === cell 28
train.columns



## === cell 29
test["year"] = test["key2"].dt.year.fillna(0).astype(int)
test["month"] = test["key2"].dt.month.fillna(0).astype(int)
test["day"] = test["key2"].dt.day.fillna(0).astype(int)
test["day of week"] = test["key2"].dt.weekday.fillna(0).astype(int)
test["hour"] = test["key2"].dt.hour.fillna(0).astype(int)
test["week"] = (
    pd.to_numeric(test["key2"].dt.isocalendar().week, errors="coerce")
    .fillna(0)
    .astype(int)
)
test["day_of_year"] = test["key2"].dt.dayofyear.fillna(0).astype(int)
test["week_of_year"] = ((test["day_of_year"] - 1) // 7 + 1).astype(int)
test["quarter"] = test["key2"].dt.quarter.fillna(0).astype(int)



## === cell 30
column_list = [
    "passenger_count",
    "distance",
    "abs_lon_diff",
    "abs_lat_diff",
    "manhattan_dist",
    "year",
    "month",
    "day",
    "day of week",
    "hour",
    "week",
    "day_of_year",
    "week_of_year",
    "quarter",
]
y_train_ = train["fare_amount"]
X_train_ = train.drop(["fare_amount"], axis=1)



## === cell 31
X_train_ = train[column_list]
X_test = test[column_list]



## === cell 32
X_train_.shape, y_train_.shape, X_test.shape



## === cell 33
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    X_train_, y_train_, test_size=0.1, random_state=42
)



## === cell 34
X_train.shape, X_val.shape, y_train.shape, y_val.shape



## === cell 35
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error



## === cell 36
lgb = LGBMRegressor(
    boosting_type="gbdt",
    class_weight=None,
    colsample_bytree=0.9,
    learning_rate=0.1,
    max_depth=8,
    min_child_samples=55,
    min_child_weight=0.001,
    min_split_gain=0.1,
    n_estimators=500,
    n_jobs=-1,
    num_leaves=45,
    objective=None,
    random_state=42,
    reg_alpha=5.0,
    reg_lambda=3.0,
    silent=True,
    subsample=1.0,
    subsample_for_bin=200000,
    subsample_freq=1,
)



## === cell 37
lgb.fit(X_train, y_train)



## === cell 38
pred = lgb.predict(X_val)



## === cell 39
from sklearn.metrics import r2_score

print(r2_score(y_val, pred))
print(np.sqrt(mean_squared_error(y_val, pred)))



## === cell 40
lgb.fit(X_train_, y_train_)




## === cell 41
def display_importances(feature_importance_df_, doWorst=False, n_feat=50):
    if not doWorst:
        cols = (
            feature_importance_df_[["feature", "importance"]]
            .groupby("feature")
            .mean()
            .sort_values(by="importance", ascending=False)[:n_feat]
            .index
        )
    else:
        cols = (
            feature_importance_df_[["feature", "importance"]]
            .groupby("feature")
            .mean()
            .sort_values(by="importance", ascending=False)[-n_feat:]
            .index
        )

    mean_imp = (
        feature_importance_df_[["feature", "importance"]].groupby("feature").mean()
    )
    df_2_neglect = mean_imp[mean_imp["importance"] < 1e-3]
    print("The list of features with 0 importance: ")
    print(df_2_neglect.index.values.tolist())
    del mean_imp, df_2_neglect

    best_features = feature_importance_df_.loc[
        feature_importance_df_.feature.isin(cols)
    ]

    plt.figure(figsize=(8, 10))
    sns.barplot(
        x="importance",
        y="feature",
        data=best_features.sort_values(by="importance", ascending=False),
    )
    plt.title("LightGBM Features")
    plt.tight_layout()
    plt.savefig("lgbm_importances.png")


importance_df = pd.DataFrame()
importance_df["feature"] = column_list
importance_df["importance"] = lgb.feature_importances_
display_importances(feature_importance_df_=importance_df, n_feat=20)



## === cell 42
lgb.fit(X_train_, y_train_)



## === cell 43
y_pred = lgb.predict(X_test)



## === cell 44
y_pred = np.clip(y_pred, 2.5, 250)

submission = pd.DataFrame({"key": test["key"], "fare_amount": y_pred})
submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
