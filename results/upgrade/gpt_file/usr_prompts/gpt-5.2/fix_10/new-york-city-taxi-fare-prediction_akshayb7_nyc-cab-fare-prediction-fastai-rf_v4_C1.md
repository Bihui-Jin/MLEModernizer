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

fastai==2.8.5
geopandas==0.14.4
google-api-python-client==2.177.0
ipython==7.34.0
ipython-genutils==0.2.0
ipython_pygments_lexers==1.1.1
ipython-sql==0.5.0
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

3.76038

# 6. Current score

6.42792

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 5.08967) has done: 'The timeout is dominated by fitting a 200-tree RandomForest on 1M rows while also paying extra overhead from pandas object dtypes and non-vectorized/duplicate work. I keep the exact same feature logic and model/training semantics, but reduce constant factors by (1) converting the training matrix to contiguous NumPy arrays once (much faster for scikit-learn), (2) computing date parts with a faster vectorized implementation that yields identical columns, and (3) using a safe/compatible `set_rf_samples` patch that avoids breaking newer scikit-learn internals while still enforcing the intended 10k samples-per-tree. I also avoid unnecessary copies during quantile/outlier computation and align test columns using a precomputed column list.'
- What this solution (achieved 7.02197) has done: 'You’re currently far from the target (RMSE 5.08967 vs 3.76038, lower is better), so we need a real (but still minimal) score improvement without changing the model type or feature set. The biggest legitimate gain for NYC Taxi Fare with the same core approach is removing obviously invalid/geographically impossible rides and fare outliers that otherwise dominate RMSE; this keeps your RandomForest and your engineered features intact, but improves label/feature quality. I add a small, standard NYC bounding-box filter plus a positive-fare cap, and I also ensure passenger_count filtering is applied consistently to test data (no behavior change in features, just consistent row validity). These changes typically move RMSE substantially toward ~3.7–4.0 on this competition while keeping runtime similar.'
- What this solution (achieved 6.84283) has done: 'Your current RMSE is much worse than the target, so the smallest reliable way to move it down (without changing the model or features) is to clean the training sample a bit more to remove rides that are very likely data errors and dominate squared error. I keep your exact RandomForest setup and engineered columns, but (1) replace the weak “absolute lon/lat deltas” outlier rule with a standard NYC taxi approach: compute haversine distance and filter impossible distances/speeds, and (2) keep passenger_count/fare/NYC-bbox filtering as you already do. This typically reduces RMSE substantially on this competition while staying within your existing core logic and runtime constraints. The submission writing and column alignment remain unchanged.'
- What this solution (achieved 6.42792) has done: 'I fix the immediate runtime blocker by removing the nonexistent `dropoff_datetime` from `parse_dates` and by disabling the downstream duration/speed filter that depends on it, so the pipeline can run end-to-end. I keep your existing core feature engineering (pickup dateparts + absolute lon/lat deltas) and the same RandomForest training approach, but ensure the data-cleaning steps execute in the right order and don’t reference undefined variables after a failed read. I also make the input PATH resolution robust to the competition subfolder so `train.csv`/`test.csv` are always found. Finally, I ensure train/test feature columns align and that a valid `submission.csv` with `key,fare_amount` is always written.'

# 9. Code solution

## === cell 0
import os
import math
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from IPython.display import display

np.random.seed(42)




## === cell 1
def add_datepart(df, fldname, drop=True, time=False, errors="coerce"):
    if fldname not in df.columns:
        raise KeyError(f"{fldname} not found in dataframe columns")

    s = df[fldname]
    if not pd.api.types.is_datetime64_any_dtype(s):
        s = pd.to_datetime(s, errors=errors, utc=False)
    if pd.api.types.is_datetime64tz_dtype(s):
        s = s.dt.tz_convert(None)
    df[fldname] = s

    fld = df[fldname]
    targ_pre = fldname

    dt = fld.dt
    df[targ_pre + "Year"] = dt.year.astype(np.int16, copy=False)
    df[targ_pre + "Month"] = dt.month.astype(np.int8, copy=False)
    df[targ_pre + "Week"] = dt.isocalendar().week.astype("int16")
    df[targ_pre + "Day"] = dt.day.astype(np.int8, copy=False)
    df[targ_pre + "Dayofweek"] = dt.dayofweek.astype(np.int8, copy=False)
    df[targ_pre + "Dayofyear"] = dt.dayofyear.astype(np.int16, copy=False)

    df[targ_pre + "Is_month_end"] = dt.is_month_end
    df[targ_pre + "Is_month_start"] = dt.is_month_start
    df[targ_pre + "Is_quarter_end"] = dt.is_quarter_end
    df[targ_pre + "Is_quarter_start"] = dt.is_quarter_start
    df[targ_pre + "Is_year_end"] = dt.is_year_end
    df[targ_pre + "Is_year_start"] = dt.is_year_start

    if time:
        df[targ_pre + "Hour"] = dt.hour.astype(np.int8, copy=False)
        df[targ_pre + "Minute"] = dt.minute.astype(np.int8, copy=False)
        df[targ_pre + "Second"] = dt.second.astype(np.int8, copy=False)

    df[targ_pre + "Elapsed"] = (
        fld.to_numpy().astype("datetime64[ns]").view("int64") // 10**9
    )

    if drop:
        df.drop(columns=[fldname], inplace=True)


def set_rf_samples(n):
    try:
        from sklearn.ensemble import _forest
    except Exception:
        return

    def _generate_sample_indices(random_state, n_samples, n_samples_bootstrap):
        random_instance = np.random.RandomState(random_state)
        k = n if n is not None else n_samples_bootstrap
        if k is None:
            k = n_samples_bootstrap
        k = int(min(k, n_samples))
        return random_instance.choice(n_samples, k, replace=False)

    _forest._generate_sample_indices = _generate_sample_indices


def haversine_km(lon1, lat1, lon2, lat2):
    lon1 = np.deg2rad(lon1.astype(np.float64, copy=False))
    lat1 = np.deg2rad(lat1.astype(np.float64, copy=False))
    lon2 = np.deg2rad(lon2.astype(np.float64, copy=False))
    lat2 = np.deg2rad(lat2.astype(np.float64, copy=False))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * (
        np.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * np.arcsin(np.sqrt(a))
    return 6371.0 * c




## === cell 2
PATH = "../input"
if os.path.exists("/kaggle/input"):
    PATH = "/kaggle/input"
elif os.path.exists("/kaggle/data"):
    PATH = "/kaggle/data"


def _resolve_file(base, fname):
    p1 = os.path.join(base, fname)
    if os.path.exists(p1):
        return p1
    p2 = os.path.join(base, "new-york-city-taxi-fare-prediction", fname)
    if os.path.exists(p2):
        return p2
    raise FileNotFoundError(
        f"Could not find {fname} under {base} or its competition subfolder"
    )


train_path = _resolve_file(PATH, "train.csv")
test_path = _resolve_file(PATH, "test.csv")

NROWS = int(os.environ.get("NROWS", "1000000"))  # default 1M for runtime safety

train_dtypes = {
    "key": "object",
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int8",
}

df_raw = pd.read_csv(
    train_path,
    nrows=NROWS,
    dtype=train_dtypes,
    parse_dates=["pickup_datetime"],
)




## === cell 3
def display_all(df):
    with pd.option_context("display.max_rows", 1000):
        with pd.option_context("display.max_columns", 1000):
            display(df)




## === cell 4
add_datepart(df_raw, "pickup_datetime", drop=True, time=True)




## === cell 5
def distance(data):
    data["longitutde_traversed"] = (
        data.dropoff_longitude - data.pickup_longitude
    ).abs()
    data["latitude_traversed"] = (data.dropoff_latitude - data.pickup_latitude).abs()
    return data


df_raw = distance(df_raw)



## === cell 6
df_raw.dropna(axis=0, how="any", inplace=True)

nyc_bbox = {"lon_min": -75.0, "lon_max": -72.0, "lat_min": 40.0, "lat_max": 42.0}
coord_mask = (
    df_raw["pickup_longitude"].between(nyc_bbox["lon_min"], nyc_bbox["lon_max"])
    & df_raw["dropoff_longitude"].between(nyc_bbox["lon_min"], nyc_bbox["lon_max"])
    & df_raw["pickup_latitude"].between(nyc_bbox["lat_min"], nyc_bbox["lat_max"])
    & df_raw["dropoff_latitude"].between(nyc_bbox["lat_min"], nyc_bbox["lat_max"])
)
fare_mask = (df_raw["fare_amount"] > 0) & (df_raw["fare_amount"] <= 250)
df_raw = df_raw.loc[coord_mask & fare_mask].reset_index(drop=True)

df_raw = df_raw[
    (df_raw.passenger_count > 0) & (df_raw.passenger_count < 10)
].reset_index(drop=True)



## === cell 7
key = df_raw.key
df_raw.drop("key", axis=1, inplace=True)



## === cell 8
df = df_raw



## === cell 9
y = df.fare_amount
df.drop("fare_amount", axis=1, inplace=True)



## === cell 10
X_train, X_valid, y_train, y_valid = train_test_split(
    df, y, test_size=10000, random_state=42
)




## === cell 11
def rmse(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    return math.sqrt(((x - y) ** 2).mean())


def print_score(m, Xtr, Xva, ytr, yva):
    pred_tr = m.predict(Xtr)
    pred_va = m.predict(Xva)
    res = [
        rmse(pred_tr, ytr),
        rmse(pred_va, yva),
        m.score(Xtr, ytr),
        m.score(Xva, yva),
    ]
    print(res)




## === cell 12
set_rf_samples(10000)



## === cell 13
feature_cols = X_train.columns.tolist()

X_train_np = np.ascontiguousarray(X_train.to_numpy(dtype=np.float32, copy=False))
X_valid_np = np.ascontiguousarray(X_valid.to_numpy(dtype=np.float32, copy=False))
y_train_np = y_train.to_numpy(dtype=np.float32, copy=False)
y_valid_np = y_valid.to_numpy(dtype=np.float32, copy=False)

m = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=42)
m.fit(X_train_np, y_train_np)
print_score(m, X_train_np, X_valid_np, y_train_np, y_valid_np)



## === cell 14
test_dtypes = {
    "key": "object",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int8",
}
test_set = pd.read_csv(test_path, dtype=test_dtypes, parse_dates=["pickup_datetime"])

test_key = test_set.key
test_set.drop("key", axis=1, inplace=True)

add_datepart(test_set, "pickup_datetime", drop=True, time=True)
test_set = distance(test_set)

valid_test_mask = (test_set["passenger_count"] > 0) & (test_set["passenger_count"] < 10)

missing_cols = [c for c in feature_cols if c not in test_set.columns]
for c in missing_cols:
    test_set[c] = 0
extra_cols = [c for c in test_set.columns if c not in feature_cols]
if extra_cols:
    test_set.drop(columns=extra_cols, inplace=True)
test_set = test_set[feature_cols]

test_np_all = np.ascontiguousarray(test_set.to_numpy(dtype=np.float32, copy=False))



## === cell 15
test_predictions_all = m.predict(test_np_all)

if not np.all(valid_test_mask.to_numpy()):
    fallback = float(np.median(test_predictions_all[valid_test_mask.to_numpy()]))
    test_predictions_all = test_predictions_all.copy()
    test_predictions_all[~valid_test_mask.to_numpy()] = fallback

submission = pd.DataFrame({"key": test_key, "fare_amount": test_predictions_all})
submission.to_csv("submission.csv", index=False)

submission.head()
