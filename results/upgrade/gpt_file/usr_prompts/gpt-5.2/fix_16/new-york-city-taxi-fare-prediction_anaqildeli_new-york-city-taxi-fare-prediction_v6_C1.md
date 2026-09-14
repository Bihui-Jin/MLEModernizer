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

3.10

# 3. Installed packages

folium==0.20.0
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

3.41572

# 6. Current score

4.90651

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 5.93093) has done: 'Your notebook doesn’t currently train any model or write a `submission.csv`, so no Kaggle score can be produced. I make the smallest end-to-end additions: keep your existing feature engineering/cleaning, retain the `key` from the test set (needed for submission), and add a simple scikit-learn regression model plus RMSE validation to ensure it’s working. I also add a minimal, safe missing-value drop and fare clipping to non-negative at prediction time (RMSE-appropriate and prevents invalid negatives) without changing your core feature logic. Finally, I generate `submission.csv` with exactly `key,fare_amount` and the correct row alignment.'
- What this solution (achieved 5.73105) has done: 'The timeout is dominated by fitting `GradientBoostingRegressor` with 400 trees on ~1M rows; scikit-learn’s classic GBDT scales poorly with N and can exceed 600s. To preserve the same training approach (GBDT) and evaluation semantics while drastically improving runtime, I switch to `HistGradientBoostingRegressor`, which is the same class of model but uses histogram-based training designed for large datasets and is orders of magnitude faster. I also avoid redundant pandas work by (1) converting datetime-derived features in a vectorized way, (2) dropping unused columns early, and (3) ensuring single-pass NumPy conversions without extra copies. All paths, features, target, and submission format remain unchanged.'
- What this solution (achieved 6.10424) has done: 'Your current gap to the target is large (5.73105 vs 3.41572; lower is better), so we should improve score with minimal risk while keeping the same overall pipeline (same data source, same cleaning, same model family: histogram GBDT, same fixed-iteration training). The biggest safe gain here is adding one or two high-signal, cheap geospatial features (haversine distance and manhattan distance proxy) without changing the training loop or loss, plus slightly more appropriate regularization/leaf sizing for noisy taxi fares. I also keep your existing datetime features and clipping, and ensure feature alignment remains identical between train/test. These changes are small, fast, and typically move RMSE meaningfully toward your target on this competition.'
- What this solution (achieved 7.24801) has done: 'You’re well below the target (6.10 vs 3.42 RMSE; lower is better), so we should make the smallest changes that reliably improve generalization without changing the overall approach (same data, same cleaning intent, same histogram GBDT, same fixed-iteration training). The biggest low-risk issue is that `test.dropna()` can drop rows and misalign with Kaggle’s expected 9914 keys; instead we impute missing feature values using training medians and keep all test rows. Next, we add two very standard, cheap taxi-fare signals (straight-line degrees distance and a simple “jfk/lga-ish” indicator) and tighten outlier filtering slightly (fare upper bound + coordinate sanity) to reduce label noise, which typically drops RMSE meaningfully on this competition. Finally, we print a quick holdout RMSE (not train-RMSE) for sanity while keeping the same fixed-iteration training semantics and writing a valid `submission.csv` with all keys.'
- What this solution (achieved 5.04058) has done: 'Your current RMSE (7.248) is far worse than the target (3.416, lower is better), so we should make a small, high-signal improvement without changing the overall approach (same HistGradientBoostingRegressor, fixed-iteration training, same core datetime + geo features). The biggest likely cause of poor generalization here is training-label noise/outliers that survive the current filters (e.g., huge trips/incorrect coordinates) and a heavy-tailed target; we can reduce this with one additional, standard trip-distance sanity filter and by training the same model on `log1p(fare_amount)` then inverting with `expm1`, which keeps the model family/loop intact but aligns better with RMSE on this competition. I also add two very cheap interaction features (`bearing` and `log1p(haversine_km)`) that usually help this dataset while preserving your existing feature set. Submission format/row alignment stays identical, and the script still writes a valid `submission.csv`.'
- What this solution (achieved 4.90651) has done: 'To move your RMSE down toward the 3.41572 target without changing the core approach (same HistGradientBoostingRegressor, same log1p target, same fixed-iteration training), I make two small, high-signal adjustments: (1) add a standard “toll/airport-ish” categorical flag feature using the existing airport distance features (no new data, very cheap), and (2) slightly tune the histogram GBDT regularization/leaf sizing to better fit this dataset while keeping the same training loop and no early stopping. I also add a minimal, RMSE-safe clipping of extreme log-predictions before expm1 to prevent rare blow-ups that can disproportionately hurt RMSE. Submission format, row alignment, and file path remain identical.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except Exception:
    pass

from sklearn.metrics import mean_squared_error

np.random.seed(42)



## === cell 1
fields = [
    "pickup_datetime",
    "fare_amount",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]

train_dtypes = {
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int16",
}

_read_csv_kwargs = dict(
    skipinitialspace=True,
    usecols=fields,
    parse_dates=["pickup_datetime"],
    dtype=train_dtypes,
    engine="c",
)

train = pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv",
    nrows=1_000_000,
    **_read_csv_kwargs,
)

print(f"{train.shape} shape")
train.head()



## === cell 2
test_usecols = [
    "key",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
test_dtypes = {
    "key": "string",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int16",
}

test = pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv",
    usecols=test_usecols,
    parse_dates=["pickup_datetime"],
    dtype=test_dtypes,
    engine="c",
)

print(f"{test.shape} shape")
test.head()



## === cell 3
pass



## === cell 4
pc = train["passenger_count"]
fare = train["fare_amount"]
plon = train["pickup_longitude"]
plat = train["pickup_latitude"]
dlon = train["dropoff_longitude"]
dlat = train["dropoff_latitude"]

mask_basic = (pc != 208) & (fare >= 0) & (fare <= 250)

geo_mask = (
    (plat > 40)
    & (plat < 45)
    & (dlat > 40)
    & (dlat < 45)
    & (plon < -71)
    & (plon > -79)
    & (dlon < -71)
    & (dlon > -79)
)

mask_pass = (pc >= 1) & (pc <= 6)

train = train.loc[mask_basic & geo_mask & mask_pass]




## === cell 5
def add_geo_features(df: pd.DataFrame) -> None:
    plon = df["pickup_longitude"].astype(np.float32, copy=False)
    plat = df["pickup_latitude"].astype(np.float32, copy=False)
    dlon = df["dropoff_longitude"].astype(np.float32, copy=False)
    dlat = df["dropoff_latitude"].astype(np.float32, copy=False)

    dlon_d = (dlon - plon).astype(np.float32, copy=False)
    dlat_d = (dlat - plat).astype(np.float32, copy=False)
    df["abs_dlon"] = np.abs(dlon_d).astype(np.float32, copy=False)
    df["abs_dlat"] = np.abs(dlat_d).astype(np.float32, copy=False)

    df["euclid_deg"] = np.sqrt(dlon_d * dlon_d + dlat_d * dlat_d).astype(
        np.float32, copy=False
    )

    r = np.float32(6371.0)
    lat1 = np.deg2rad(plat.to_numpy(copy=False).astype(np.float32, copy=False))
    lat2 = np.deg2rad(dlat.to_numpy(copy=False).astype(np.float32, copy=False))
    dlat_r = lat2 - lat1
    dlon_r = np.deg2rad(
        dlon.to_numpy(copy=False).astype(np.float32, copy=False)
    ) - np.deg2rad(plon.to_numpy(copy=False).astype(np.float32, copy=False))
    a = np.sin(dlat_r / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * (
        np.sin(dlon_r / 2.0) ** 2
    )
    c = 2.0 * np.arcsin(np.minimum(1.0, np.sqrt(a)))
    df["haversine_km"] = (r * c).astype(np.float32, copy=False)

    df["manhattan_km"] = (
        r * (np.abs(dlat_r) + np.abs(dlon_r) * np.cos((lat1 + lat2) / 2.0))
    ).astype(np.float32, copy=False)

    df["bearing"] = np.arctan2(dlon_r, dlat_r).astype(np.float32, copy=False)
    df["log_haversine_km"] = np.log1p(df["haversine_km"].to_numpy(copy=False)).astype(
        np.float32, copy=False
    )

    jfk_lat, jfk_lon = np.float32(40.6413), np.float32(-73.7781)
    lga_lat, lga_lon = np.float32(40.7769), np.float32(-73.8740)

    df["pickup_jfk_deg"] = np.sqrt(
        (plat - jfk_lat) ** 2 + (plon - jfk_lon) ** 2
    ).astype(np.float32, copy=False)
    df["dropoff_jfk_deg"] = np.sqrt(
        (dlat - jfk_lat) ** 2 + (dlon - jfk_lon) ** 2
    ).astype(np.float32, copy=False)
    df["pickup_lga_deg"] = np.sqrt(
        (plat - lga_lat) ** 2 + (plon - lga_lon) ** 2
    ).astype(np.float32, copy=False)
    df["dropoff_lga_deg"] = np.sqrt(
        (dlat - lga_lat) ** 2 + (dlon - lga_lon) ** 2
    ).astype(np.float32, copy=False)


for df in (train, test):
    dt = df["pickup_datetime"].dt
    df["year"] = dt.year.astype("int16", copy=False)
    df["month"] = dt.month.astype("int8", copy=False)
    df["day"] = dt.day.astype("int8", copy=False)
    df["hour"] = dt.hour.astype("int8", copy=False)
    df["minute"] = dt.minute.astype("int8", copy=False)

    add_geo_features(df)

    df["near_airport"] = (
        (df["pickup_jfk_deg"] < np.float32(0.08))
        | (df["dropoff_jfk_deg"] < np.float32(0.08))
        | (df["pickup_lga_deg"] < np.float32(0.06))
        | (df["dropoff_lga_deg"] < np.float32(0.06))
    ).astype("int8")

train.drop(columns=["pickup_datetime"], inplace=True)
test.drop(columns=["pickup_datetime"], inplace=True)



## === cell 6
train = train.loc[(train["haversine_km"] > 0.01) & (train["haversine_km"] < 80.0)]

train = train.dropna().reset_index(drop=True)

test_key = test["key"].to_numpy(copy=False)
test = test.reset_index(drop=True)

test["passenger_count"] = test["passenger_count"].clip(1, 6)

num_cols = [c for c in train.columns if c != "fare_amount"]
medians = train[num_cols].median(numeric_only=True)
test[num_cols] = test[num_cols].fillna(medians)
train[num_cols] = train[num_cols].fillna(medians)

train.shape, test.shape



## === cell 7
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split

X = train.drop(columns=["fare_amount"])

y = np.log1p(train["fare_amount"].astype(np.float32, copy=False))

feature_cols = X.columns

X_all_np = np.ascontiguousarray(X.to_numpy(dtype=np.float32, copy=False))
y_all_np = y.to_numpy(dtype=np.float32, copy=False)

X_tr, X_va, y_tr, y_va = train_test_split(
    X_all_np, y_all_np, test_size=0.1, random_state=42
)

model = HistGradientBoostingRegressor(
    random_state=42,
    max_iter=500,  # fixed iterations (no early stopping)
    learning_rate=0.05,
    max_depth=7,
    max_bins=255,
    l2_regularization=0.05,
    min_samples_leaf=25,
    early_stopping=False,
)

model.fit(X_tr, y_tr)

va_pred_log = model.predict(X_va)

va_pred_log = np.clip(va_pred_log, np.float32(0.0), np.float32(np.log1p(250.0)))

va_pred = np.expm1(va_pred_log)
y_va_fare = np.expm1(y_va)
rmse = mean_squared_error(y_va_fare, va_pred, squared=False)
print("Holdout RMSE:", rmse)

model.fit(X_all_np, y_all_np)

test_X_np = np.ascontiguousarray(
    test[feature_cols].to_numpy(dtype=np.float32, copy=False)
)
test_pred_log = model.predict(test_X_np)

test_pred_log = np.clip(test_pred_log, np.float32(0.0), np.float32(np.log1p(250.0)))

test_pred = np.expm1(test_pred_log)
test_pred = np.clip(test_pred, 0, None)

submission = pd.DataFrame({"key": test_key, "fare_amount": test_pred})

print(submission.head())
print("Submission shape:", submission.shape)

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print("Wrote:", submission_path)
