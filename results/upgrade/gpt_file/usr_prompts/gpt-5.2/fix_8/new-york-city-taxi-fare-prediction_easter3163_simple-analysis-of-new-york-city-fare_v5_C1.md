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

12.74015

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 199.84822) has done: 'I fix the pandas datetime attribute errors by replacing deprecated `.dt.week`/`.dt.weekofyear` with ISO calendar week extraction that works in pandas 2.2+. I also make the input paths robust to your environment by resolving to the available `/kaggle/input/...` or `/kaggle/data/...` locations while keeping the same filenames. To move the RMSE score dramatically closer to the target with minimal modeling changes, I keep your exact 3-model ensemble but add a standard Haversine distance feature (a legitimate core-feature fix for taxi fare) and include the existing datetime features consistently for both train/test. Finally, I ensure the submission file is written as a valid `.csv` with the required `key,fare_amount` columns aligned to `test.csv` order.'
- What this solution (achieved 444.24131) has done: 'Your RMSE is extremely high because the model is learning on a train sample that is (a) filtered with an incorrect longitude range (`-75 < lon < 75` keeps many invalid longitudes) and (b) missing two very strong baseline signals for this competition: straight-line distance in *miles* (used implicitly by fare pricing) and airport/manhattan locality effects. With minimal changes that preserve your exact 3-model ensemble and training approach, I fix the longitude filters to NYC-appropriate bounds, add a Haversine distance in miles plus simple “center distance” features, and keep the same prediction averaging and submission writing. These changes are legitimate feature/cleaning corrections and should move RMSE dramatically down toward your target band without altering your modeling core. I also ensure deterministic sampling and safe NaN handling so the pipeline runs reliably and writes a valid submission CSV.'
- What this solution (achieved 442.54093) has done: 'Your current RMSE is far above target, so we should improve (lower) it with the smallest changes that don’t alter your model ensemble/training loop. The biggest likely issue left is that you train on a random slice of the first 500k rows, which is often noisy/dirty; using a deterministic but more representative sample across the whole file (still 500k) typically drops RMSE a lot without changing core modeling. I also fix a subtle submission bug: when `fare_amount` is missing after the merge, you currently fill with the median of an all-NaN column (stays NaN); we instead fill with a safe fallback (training median) to ensure a valid submission. Everything else (features, models, and averaging) stays the same.'
- What this solution (achieved 445.8949) has done: 'The timeout is dominated by `pd.read_csv(..., skiprows=lambda ...)`, which calls a Python lambda for every one of the ~55M rows and recreates `RandomState(42)` each time—this is catastrophically slow. I replace that with a provably-equivalent one-pass chunked read that keeps each row with the same probability (so expected sample size matches) and then applies the same fixed-size cap to 500k rows, preserving the algorithm’s semantics. I also eliminate heavy plotting (which doesn’t affect predictions) and avoid a few unnecessary large temporary allocations (e.g., `np.full(len(train), ...)`) while keeping feature logic identical. Everything else—features, models, training, and blending—remains unchanged.'
- What this solution (achieved 12.21026) has done: 'Your RMSE is wildly above target (lower is better), so we should improve it with the smallest changes that keep your exact feature set and 3-model blend intact. The main issue is that KNN and LinearRegression are extremely sensitive to feature scale; right now distance features (miles/km) and time/week features are on very different scales, which can yield terrible generalization and blow up RMSE. I add a `StandardScaler` and apply it to both train/test features (no change to model types, loss, or blending), while keeping the RandomForest on the original (unscaled) features since it doesn’t benefit from scaling. I also add a minimal post-processing clip to cap extreme predictions (still legitimate for fare amounts) to reduce outlier-driven RMSE without changing the training approach.'
- What this solution (achieved 12.74015) has done: 'We need to move RMSE down from 12.21 toward 4.71 (lower is better), so we make the smallest changes that improve generalization without changing your 3-model ensemble, features, or training loop. The biggest remaining issue is that KNN on 500k points is both impractically slow and effectively behaves poorly; we keep KNN but fit it on a small, deterministic subsample (still using the same scaled features and prediction blending), which typically improves and stabilizes results while staying within runtime. We also ensure the scaler is fit only on the same rows used by Linear/KNN (and those rows are the same filtered training set), and we add a minimal, legitimate log1p-transform on the *target for LinearRegression only* (with inverse transform at predict) to reduce skew/outlier impact while keeping the model type and blending intact. Finally, we keep your submission alignment/format exactly as required and still clip to a realistic fare range.'

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

P_KEEP = 500_000 / 55_423_856

rng = np.random.RandomState(42)

chunks = []
chunksize = 1_000_000
for chunk in pd.read_csv(TRAIN_PATH, usecols=USECOLS, chunksize=chunksize):
    m = rng.rand(len(chunk)) <= P_KEEP
    if m.any():
        chunks.append(chunk.loc[m])

train = (
    pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=USECOLS)
)

if len(train) > 500_000:
    train = train.sample(n=500_000, random_state=42)

test = pd.read_csv(TEST_PATH)



## === cell 1
train.shape



## === cell 2
train.head()



## === cell 3
import datetime as dt  # kept as in original environment usage (even if unused)



## === cell 4
train["pickup_datetime"] = pd.to_datetime(
    train["pickup_datetime"], errors="coerce", utc=False
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
    test["pickup_datetime"], errors="coerce", utc=False
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



## === cell 8
train["abs_diff_longitude"] = (
    train["pickup_longitude"] - train["dropoff_longitude"]
).abs()
train["abs_diff_latitude"] = (
    train["pickup_latitude"] - train["dropoff_latitude"]
).abs()



## === cell 9
test["abs_diff_longitude"] = (
    test["pickup_longitude"] - test["dropoff_longitude"]
).abs()
test["abs_diff_latitude"] = (test["pickup_latitude"] - test["dropoff_latitude"]).abs()




## === cell 10
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



## === cell 11
train.head()



## === cell 12
train.head()



## === cell 13
pass



## === cell 14
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



## === cell 15
label_name = "fare_amount"
label_name



## === cell 16
X_train = train[feature_names].copy()
y_train = train[label_name].copy()
X_test = test[feature_names].copy()

X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0.0)
X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0.0)



## === cell 17
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
import xgboost as xgb  # kept to preserve original imports even if unused



## === cell 18
KNN_FIT_N = 50_000  # small enough to run reliably; still representative

if len(X_train) > KNN_FIT_N:
    knn_idx = X_train.sample(n=KNN_FIT_N, random_state=42).index
else:
    knn_idx = X_train.index

scaler_lr = StandardScaler()
X_train_scaled_lr = scaler_lr.fit_transform(X_train)
X_test_scaled_lr = scaler_lr.transform(X_test)

scaler_knn = StandardScaler()
X_train_scaled_knn = scaler_knn.fit_transform(X_train.loc[knn_idx])
X_test_scaled_knn = scaler_knn.transform(X_test)



## === cell 19
regr = LinearRegression()
regr.fit(X_train_scaled_lr, np.log1p(y_train))
regr_prediction = np.expm1(regr.predict(X_test_scaled_lr))



## === cell 20
knr = KNeighborsRegressor(n_neighbors=10, weights="distance")
knr.fit(X_train_scaled_knn, y_train.loc[knn_idx])
knr_prediction = knr.predict(X_test_scaled_knn)



## === cell 21
rfr = RandomForestRegressor(
    n_estimators=120,
    random_state=42,
    n_jobs=-1,
    min_samples_leaf=2,
)
rfr.fit(X_train, y_train)
rfr_prediction = rfr.predict(X_test)



## === cell 22
predictions = (regr_prediction * 2 + rfr_prediction * 2 + knr_prediction) / 5



## === cell 23
predictions



## === cell 24
submission = pd.read_csv(SAMPLE_SUB_PATH)

pred_df = pd.DataFrame({"key": test["key"].values, "fare_amount": predictions})
submission = submission[["key"]].merge(pred_df, on="key", how="left")

fallback = float(y_train.median())
submission["fare_amount"] = (
    submission["fare_amount"].fillna(fallback).clip(lower=0.0, upper=200.0)
)

submission.head()



## === cell 25
out_path = "./simplenewyorktaxi.csv"
submission.to_csv(out_path, index=False)
out_path
