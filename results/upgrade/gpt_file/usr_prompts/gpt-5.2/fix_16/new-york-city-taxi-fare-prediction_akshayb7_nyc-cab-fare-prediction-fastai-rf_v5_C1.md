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

3.83596

# 6. Current score

10.19558

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 5.15477) has done: 'I remove the deprecated fastai v0 imports and replace only the missing helper utilities (`add_datepart`, `set_rf_samples`, `reset_rf_samples`, `rf_feat_importance`) with small local equivalents so the pipeline runs in this environment. I fix the outlier loop crashing on non-numeric columns by restricting it to numeric features, and fix a logic bug where you computed outlier-filtered `df` but then trained on `df_raw`/`y` instead. I also ensure the train/test feature engineering produces identical columns (especially the date parts), and that `train_test_split` and `RandomForestRegressor` are correctly imported and executed. Finally, the code always write a valid `submission.csv` with columns `key,fare_amount` in the working directory.'
- What this solution (achieved 5.42484) has done: 'Your RMSE is worse than the target, so we should legitimately improve generalization with minimal changes while keeping the same RandomForest + same feature set and training flow. The biggest safe gain here is to (1) remove obviously invalid training rows (NYC bounding box + positive fare + sane coordinates) which otherwise teach the model nonsense, and (2) use a log1p target transform during training and invert at prediction time, which typically reduces RMSE for skewed fare distributions without changing the model type or loss. I’m also switching the validation split to a proportion (not an absolute 10k) so it scales correctly with your 10M sample and better reflects performance. Finally, I clip negative predictions to 0 to avoid invalid fares hurting RMSE.'
- What this solution (achieved 5.50976) has done: 'Your current RMSE (5.42484) is worse than the target (3.83596), so we should legitimately improve generalization with the smallest safe changes while keeping the same RandomForest + same features/training flow. The biggest issue is that you train on a log1p-transformed target but your validation RMSE is computed in log space, which misleads tuning; we compute RMSE in original fare space (via expm1) without changing training. Next, we replace the overly-aggressive global “outlier over all numeric columns” removal (which can drop many valid rows and distort the target distribution) with only the already-present distance-based outlier filter, preserving the same core cleaning idea but making it less destructive. Finally, we add a minimal derived “manhattan_distance” feature from your existing traversed deltas (same feature family, no model change), which typically improves taxi-fare RF performance.'
- What this solution (achieved 5.35737) has done: 'Your current RMSE (5.50976) is worse than the target (3.83596), so we should make small, legitimate generalization improvements without changing the RandomForest approach or feature family. The biggest safe lever here is to use a more appropriate distance feature: keep your existing “traversed deltas” but also add a proper Haversine distance (still purely derived from the same coordinates), which is a standard gain for this competition. Next, we make the RandomForest less biased by increasing `n_estimators` modestly and enabling `oob_score` (no training-loop change, still RF), and we apply the same clipping of extreme coordinates to both train and test via the same NYC mask logic to keep feature distributions aligned. Finally, we keep the log1p target training exactly as-is but clip predictions to a reasonable upper bound as well as 0 to reduce RMSE damage from rare extreme predictions.'
- What this solution (achieved 5.27883) has done: 'Your current RMSE (5.35737) is worse than the target (3.83596), so we should improve generalization with minimal, legitimate changes while keeping the same RandomForest + feature engineering flow. The biggest gap here is likely inconsistent training data quality and suboptimal weighting: we (1) add the standard airport/Manhattan location binary features derived purely from existing coordinates (no new data), and (2) use `sample_weight` to upweight longer trips so the model doesn’t underfit high-fare rides (this usually improves overall RMSE without changing the model type). We also stop clipping predictions at 250 (which can hurt RMSE on legitimate higher fares) and instead only clip at 0, while keeping the same log1p training and submission format. All changes are localized and keep the same architecture, training approach, and metric semantics.'
- What this solution (achieved 4.8306) has done: 'The main timeout drivers are (1) reading and materializing 10M training rows into pandas, (2) heavy datetime feature extraction on that large frame, and (3) training a 200-tree RandomForest on a multi-million-row dataset even though each tree only bootstraps 200k rows. To keep the exact same core logic and semantics, the fastest safe fix is to stream `train.csv` in chunks, apply the exact same filtering/feature logic per chunk, and stop once we have enough cleaned rows to support the same RF bootstrap size (200k) plus the validation split. This preserves the model, features, training loop, loss/metric, and evaluation behavior, while eliminating unnecessary work on millions of extra rows. Additionally, enabling Intel-optimized scikit-learn (if available) and avoiding a few large intermediate allocations reduces runtime without affecting results.'
- What this solution (achieved 5.20518) has done: 'We make two minimal, score-relevant fixes that preserve your RandomForest + feature engineering core logic. First, instead of clipping training coordinates into the NYC box (which creates many artificial “teleport” trips), we filter to the box like you already do later—this reduces label noise and typically improves RMSE. Second, we fix a subtle mismatch: training used clipped coords but test used raw coords for distance/flags (computed before clipping); we clip test coords *before* distance/flag features so train/test features align. Everything else (log1p target, RF settings, sampling, submission format) stays the same.'
- What this solution (achieved 5.43215) has done: 'Your current RMSE (5.20518) is worse than the target (3.83596), so we should make small, legitimate improvements without changing the RandomForest approach or feature family. The biggest score drag here is likely label noise from bad training rows and a train/test mismatch: you clip test coordinates (creating “teleport” trips) but you *filter* train to NYC, and you also compute the outlier mask only on axis deltas rather than the true trip length. I (1) add a minimal “zero-distance trip” and “minimum fare sanity” filter to remove obvious bad labels, (2) stop clipping test coordinates and instead apply the same NYC filter to test (fallback to clipping only if it would drop rows), and (3) compute the outlier mask using `haversine_km` (already computed) rather than separate lon/lat deltas. These changes preserve the same model, log1p target, and overall pipeline, but reduce noise and align distributions, which should move RMSE toward the target.'
- What this solution (achieved 10.19558) has done: 'I fix the crash in cell 22 by ensuring `pickup_datetime` is not left as a datetime column in `X_train/X_valid` (it currently survives because `add_datepart(..., drop=False)` keeps it), and by making the finite-check robust via `DataFrame.to_numpy(dtype=np.float64)` after filling NaNs. This unblocks model fitting so `m` exists for later feature importance and test prediction cells. I also apply the same datetime-drop and NaN/inf handling consistently to train/valid/test so feature types align end-to-end. The model, features, and training approach (RandomForest + log1p target + rf bootstrap sampling) are unchanged; this is primarily a correctness/stability fix to produce a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import gc
import math
import random
import numpy as np
import pandas as pd

try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except Exception:
    pass

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from IPython.display import display


def add_datepart(df, fldname, drop=True, time=False, errors="coerce"):
    """
    Adds columns relevant to a date in the column `fldname`.
    Fix: robust to timezone-aware datetimes (e.g. datetime64[ns, UTC]) which break np.issubdtype.
    """
    if fldname not in df.columns:
        raise KeyError(f"{fldname} not found in dataframe columns")

    if not pd.api.types.is_datetime64_any_dtype(df[fldname]):
        df[fldname] = pd.to_datetime(df[fldname], errors=errors, utc=True)
    if pd.api.types.is_datetime64tz_dtype(df[fldname]):
        df[fldname] = df[fldname].dt.tz_convert(None)

    fld = df[fldname]
    prefix = fldname

    attrs = [
        "Year",
        "Month",
        "Week",
        "Day",
        "Dayofweek",
        "Dayofyear",
        "Is_month_end",
        "Is_month_start",
        "Is_quarter_end",
        "Is_quarter_start",
        "Is_year_end",
        "Is_year_start",
    ]
    for attr in attrs:
        if attr == "Week":
            df[prefix + attr] = fld.dt.isocalendar().week.astype("int16")
        else:
            df[prefix + attr] = getattr(fld.dt, attr.lower())

    if time:
        df[prefix + "Hour"] = fld.dt.hour.astype("int16")
        df[prefix + "Minute"] = fld.dt.minute.astype("int16")
        df[prefix + "Second"] = fld.dt.second.astype("int16")

    if drop:
        df.drop(columns=[fldname], inplace=True)
    return df


_old_generate_sample_indices = None


def set_rf_samples(n):
    """
    Limit bootstrap sample size per tree to n (fastai v0 trick).
    """
    global _old_generate_sample_indices
    from sklearn.ensemble import _forest

    if _old_generate_sample_indices is None:
        _old_generate_sample_indices = _forest._generate_sample_indices

    def _generate_sample_indices(random_state, n_samples, n_samples_bootstrap):
        return _old_generate_sample_indices(random_state, n_samples, n)

    _forest._generate_sample_indices = _generate_sample_indices


def reset_rf_samples():
    """
    Restore sklearn's original bootstrap sampling.
    """
    global _old_generate_sample_indices
    if _old_generate_sample_indices is None:
        return
    from sklearn.ensemble import _forest

    _forest._generate_sample_indices = _old_generate_sample_indices
    _old_generate_sample_indices = None


def rf_feat_importance(m, df):
    """
    Return feature importances DataFrame (fastai v0 style).
    """
    return pd.DataFrame(
        {"cols": df.columns, "imp": m.feature_importances_}
    ).sort_values("imp", ascending=False)




## === cell 1
PATH = "/kaggle/input"  # Kaggle canonical path
train_path = f"{PATH}/train.csv"
test_path = f"{PATH}/test.csv"

USE_COLS_TRAIN = [
    "key",
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
DTYPES_TRAIN = {
    "key": "object",
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int8",
}

nyc_min_long, nyc_max_long = -74.3, -73.7
nyc_min_lat, nyc_max_lat = 40.5, 41.0

TARGET_RF_BOOTSTRAP = 200_000
VALID_FRAC = 0.01

TARGET_CLEAN_ROWS = int(TARGET_RF_BOOTSTRAP / (1.0 - VALID_FRAC) + 50_000)

CHUNK_SIZE = 1_000_000  # large chunks reduce overhead but fit typical Kaggle RAM
df_chunks = []
clean_rows = 0

reader = pd.read_csv(
    train_path,
    usecols=USE_COLS_TRAIN,
    dtype=DTYPES_TRAIN,
    parse_dates=["pickup_datetime"],
    infer_datetime_format=True,
    engine="c",
    chunksize=CHUNK_SIZE,
)

for chunk in reader:
    mask = (
        chunk["pickup_longitude"].between(nyc_min_long, nyc_max_long)
        & chunk["dropoff_longitude"].between(nyc_min_long, nyc_max_long)
        & chunk["pickup_latitude"].between(nyc_min_lat, nyc_max_lat)
        & chunk["dropoff_latitude"].between(nyc_min_lat, nyc_max_lat)
    )
    chunk = chunk.loc[mask].copy()

    if len(chunk) == 0:
        continue

    df_chunks.append(chunk)
    clean_rows += len(chunk)

    if clean_rows >= TARGET_CLEAN_ROWS:
        break

df_raw = pd.concat(df_chunks, ignore_index=True)
del df_chunks, reader, chunk
gc.collect()




## === cell 2
def display_all(df):
    with pd.option_context("display.max_rows", 1000), pd.option_context(
        "display.max_columns", 1000
    ):
        display(df)




## === cell 3
pass


## === cell 4
add_datepart(df_raw, "pickup_datetime", drop=False, time=True)




## === cell 5
def distance(data):
    plon = data["pickup_longitude"].to_numpy(dtype=np.float64, copy=False)
    plat = data["pickup_latitude"].to_numpy(dtype=np.float64, copy=False)
    dlon = data["dropoff_longitude"].to_numpy(dtype=np.float64, copy=False)
    dlat = data["dropoff_latitude"].to_numpy(dtype=np.float64, copy=False)

    lon_tr = np.abs(dlon - plon)
    lat_tr = np.abs(dlat - plat)
    data["longitutde_traversed"] = lon_tr.astype(np.float32)
    data["latitude_traversed"] = lat_tr.astype(np.float32)
    data["manhattan_distance"] = (lon_tr + lat_tr).astype(np.float32)

    lat1 = np.radians(plat)
    lon1 = np.radians(plon)
    lat2 = np.radians(dlat)
    lon2 = np.radians(dlon)

    dlat_r = lat2 - lat1
    dlon_r = lon2 - lon1
    a = np.sin(dlat_r / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * (
        np.sin(dlon_r / 2.0) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    data["haversine_km"] = (6371.0 * c).astype(np.float32)


def add_location_flags(data):
    manhattan = (-74.047285, -73.910586, 40.683928, 40.882214)
    jfk = (-73.8352, -73.7401, 40.6195, 40.6659)
    ewr = (-74.1920, -74.1460, 40.6700, 40.7080)
    lga = (-73.8895, -73.8550, 40.7660, 40.7760)

    plon = data["pickup_longitude"].to_numpy(copy=False)
    plat = data["pickup_latitude"].to_numpy(copy=False)
    dlon = data["dropoff_longitude"].to_numpy(copy=False)
    dlat = data["dropoff_latitude"].to_numpy(copy=False)

    def in_box_np(lon, lat, box):
        lon_min, lon_max, lat_min, lat_max = box
        return (lon >= lon_min) & (lon <= lon_max) & (lat >= lat_min) & (lat <= lat_max)

    pickup_manhattan = in_box_np(plon, plat, manhattan)
    dropoff_manhattan = in_box_np(dlon, dlat, manhattan)

    pickup_jfk = in_box_np(plon, plat, jfk)
    dropoff_jfk = in_box_np(dlon, dlat, jfk)

    pickup_ewr = in_box_np(plon, plat, ewr)
    dropoff_ewr = in_box_np(dlon, dlat, ewr)

    pickup_lga = in_box_np(plon, plat, lga)
    dropoff_lga = in_box_np(dlon, dlat, lga)

    data["pickup_manhattan"] = pickup_manhattan.astype("int8")
    data["dropoff_manhattan"] = dropoff_manhattan.astype("int8")
    data["pickup_jfk"] = pickup_jfk.astype("int8")
    data["dropoff_jfk"] = dropoff_jfk.astype("int8")
    data["pickup_ewr"] = pickup_ewr.astype("int8")
    data["dropoff_ewr"] = dropoff_ewr.astype("int8")
    data["pickup_lga"] = pickup_lga.astype("int8")
    data["dropoff_lga"] = dropoff_lga.astype("int8")

    is_airport_trip = (pickup_jfk | pickup_ewr | pickup_lga) ^ (
        dropoff_jfk | dropoff_ewr | dropoff_lga
    )
    data["is_airport_trip"] = is_airport_trip.astype("int8")




## === cell 6
distance(df_raw)
add_location_flags(df_raw)


## === cell 7
pass


## === cell 8
df_raw.dropna(axis=0, how="any", inplace=True)


## === cell 9
df_raw.shape


## === cell 10
key = df_raw.key
df_raw.drop("key", axis=1, inplace=True)


## === cell 11
pass


## === cell 12
df_raw = df_raw[(df_raw.passenger_count > 0) & (df_raw.passenger_count < 10)]


## === cell 13
len(df_raw)


## === cell 14
df_raw.reset_index(drop=True, inplace=True)

plon = df_raw["pickup_longitude"].to_numpy(copy=False)
dlon = df_raw["dropoff_longitude"].to_numpy(copy=False)
plat = df_raw["pickup_latitude"].to_numpy(copy=False)
dlat = df_raw["dropoff_latitude"].to_numpy(copy=False)

coord_mask = (
    (plon >= nyc_min_long)
    & (plon <= nyc_max_long)
    & (dlon >= nyc_min_long)
    & (dlon <= nyc_max_long)
    & (plat >= nyc_min_lat)
    & (plat <= nyc_max_lat)
    & (dlat >= nyc_min_lat)
    & (dlat <= nyc_max_lat)
)

fare = df_raw["fare_amount"].to_numpy(copy=False)
fare_mask = (fare > 0) & (fare < 250)

lon_tr = df_raw["longitutde_traversed"].to_numpy(copy=False)
lat_tr = df_raw["latitude_traversed"].to_numpy(copy=False)
dist_mask = (lon_tr >= 0) & (lon_tr < 2.0) & (lat_tr >= 0) & (lat_tr < 2.0)

hav = df_raw["haversine_km"].to_numpy(copy=False)
hav_mask = (hav >= 0.0) & (hav <= 100.0)

min_base_fare = 2.5
zero_dist = hav < 0.05  # ~50 meters
sanity_mask = (~zero_dist & (fare >= min_base_fare)) | (zero_dist & (fare <= 10.0))

dt = pd.to_datetime(df_raw["pickup_datetime"], errors="coerce").to_numpy(
    dtype="datetime64[ns]"
)
if len(dt) >= 2:
    dt_ns = dt.astype("datetime64[ns]").astype("int64")
    dt_next = np.empty_like(dt_ns)
    dt_next[:-1] = dt_ns[1:]
    dt_next[-1] = dt_ns[-1]
    dt_diff_s = (dt_next - dt_ns) / 1e9
    dt_diff_s = np.where(dt_diff_s > 0, dt_diff_s, np.nan)
    speed_kmh = hav / (dt_diff_s / 3600.0)
    speed_mask = np.isnan(speed_kmh) | (speed_kmh <= 120.0)
else:
    speed_mask = np.ones(len(df_raw), dtype=bool)

df_raw = df_raw[
    coord_mask & fare_mask & dist_mask & hav_mask & sanity_mask & speed_mask
].reset_index(drop=True)

del (
    plon,
    dlon,
    plat,
    dlat,
    fare,
    lon_tr,
    lat_tr,
    hav,
    coord_mask,
    fare_mask,
    dist_mask,
    hav_mask,
    zero_dist,
    sanity_mask,
    dt,
    speed_mask,
)
gc.collect()


## === cell 15
arr_hav = df_raw["haversine_km"].to_numpy(copy=False)
Q1_h, Q3_h = np.percentile(arr_hav, [25, 75])
step_h = 10 * (Q3_h - Q1_h)

mask_h = (arr_hav >= Q1_h - step_h) & (arr_hav <= Q3_h + step_h)
outliers_mask = ~mask_h
outliers = np.flatnonzero(outliers_mask).tolist()

len(outliers) / len(df_raw)


## === cell 16
df = df_raw.drop(df_raw.index[outliers]).reset_index(drop=True)

del df_raw, outliers, outliers_mask, mask_h, arr_hav
gc.collect()


## === cell 17
len(df)


## === cell 18
y = np.log1p(df.fare_amount.astype(np.float64))
finite_y_mask = np.isfinite(y.to_numpy())
if not finite_y_mask.all():
    df = df.loc[finite_y_mask].reset_index(drop=True)
    y = y.loc[finite_y_mask].reset_index(drop=True)

df.drop("fare_amount", axis=1, inplace=True)


## === cell 19
df["_pickup_dt"] = pd.to_datetime(df["pickup_datetime"], errors="coerce")
df.sort_values("_pickup_dt", inplace=True)
df.reset_index(drop=True, inplace=True)

n = len(df)
n_valid = max(1, int(round(VALID_FRAC * n)))
split_idx = n - n_valid

X_train = df.iloc[:split_idx].copy()
X_valid = df.iloc[split_idx:].copy()
y_train = y.iloc[:split_idx].copy()
y_valid = y.iloc[split_idx:].copy()

for X in (X_train, X_valid):
    drop_cols = [c for c in ["pickup_datetime", "_pickup_dt"] if c in X.columns]
    if drop_cols:
        X.drop(columns=drop_cols, inplace=True)

del df, y
gc.collect()




## === cell 20
def rmse(x, y):
    return math.sqrt(((x - y) ** 2).mean())


def print_score(m):
    pred_tr_log = m.predict(X_train)
    pred_va_log = m.predict(X_valid)

    pred_tr = np.expm1(pred_tr_log)
    pred_va = np.expm1(pred_va_log)

    y_tr = np.expm1(y_train)
    y_va = np.expm1(y_valid)

    res = [
        rmse(pred_tr, y_tr),
        rmse(pred_va, y_va),
        m.score(X_train, y_train),
        m.score(X_valid, y_valid),
    ]
    print(res)




## === cell 21
set_rf_samples(200_000)


## === cell 22
bad_cols = [c for c in X_train.columns if not pd.api.types.is_numeric_dtype(X_train[c])]
if len(bad_cols) > 0:
    raise TypeError(f"Non-numeric columns remain in X_train: {bad_cols}")

X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0.0)
X_valid = X_valid.replace([np.inf, -np.inf], np.nan).fillna(0.0)

train_finite_mask = np.isfinite(X_train.to_numpy(dtype=np.float64)).all(
    axis=1
) & np.isfinite(y_train.to_numpy(dtype=np.float64))
valid_finite_mask = np.isfinite(X_valid.to_numpy(dtype=np.float64)).all(
    axis=1
) & np.isfinite(y_valid.to_numpy(dtype=np.float64))

if not train_finite_mask.all():
    X_train = X_train.loc[train_finite_mask].reset_index(drop=True)
    y_train = y_train.loc[train_finite_mask].reset_index(drop=True)

if not valid_finite_mask.all():
    X_valid = X_valid.loc[valid_finite_mask].reset_index(drop=True)
    y_valid = y_valid.loc[valid_finite_mask].reset_index(drop=True)

train_w = 1.0 + np.clip(X_train["haversine_km"].values, 0.0, 60.0) / 10.0

m = RandomForestRegressor(
    n_estimators=200,
    n_jobs=-1,
    random_state=42,
    oob_score=True,
    bootstrap=True,
)
m.fit(X_train, y_train, sample_weight=train_w)
print_score(m)


## === cell 23
fi = rf_feat_importance(m, X_train)
fi[:10]


## === cell 24
pass


## === cell 25
USE_COLS_TEST = [
    "key",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
DTYPES_TEST = {
    "key": "object",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int8",
}
test_set = pd.read_csv(
    test_path,
    usecols=USE_COLS_TEST,
    dtype=DTYPES_TEST,
    parse_dates=["pickup_datetime"],
    infer_datetime_format=True,
    engine="c",
)


## === cell 26
test_key = test_set.key
test_set.drop("key", axis=1, inplace=True)

test_coord_mask = (
    test_set["pickup_longitude"].between(nyc_min_long, nyc_max_long)
    & test_set["dropoff_longitude"].between(nyc_min_long, nyc_max_long)
    & test_set["pickup_latitude"].between(nyc_min_lat, nyc_max_lat)
    & test_set["dropoff_latitude"].between(nyc_min_lat, nyc_max_lat)
)

if test_coord_mask.all():
    test_set_proc = test_set.copy()
else:
    test_set_proc = test_set.copy()
    for col, lo, hi in [
        ("pickup_longitude", nyc_min_long, nyc_max_long),
        ("dropoff_longitude", nyc_min_long, nyc_max_long),
        ("pickup_latitude", nyc_min_lat, nyc_max_lat),
        ("dropoff_latitude", nyc_min_lat, nyc_max_lat),
    ]:
        vals = test_set_proc[col].to_numpy(copy=False)
        bad = ~test_coord_mask.to_numpy(copy=False)
        vals[bad] = np.clip(vals[bad], lo, hi)

add_datepart(test_set_proc, "pickup_datetime", drop=False, time=True)
distance(test_set_proc)
add_location_flags(test_set_proc)

if "pickup_datetime" in test_set_proc.columns:
    test_set_proc.drop(columns=["pickup_datetime"], inplace=True)

for c in X_train.columns:
    if c not in test_set_proc.columns:
        test_set_proc[c] = 0
test_set_proc = test_set_proc[X_train.columns]

bad_cols_test = [
    c
    for c in test_set_proc.columns
    if not pd.api.types.is_numeric_dtype(test_set_proc[c])
]
if len(bad_cols_test) > 0:
    raise TypeError(f"Non-numeric columns remain in test_set: {bad_cols_test}")

test_set_proc = test_set_proc.replace([np.inf, -np.inf], np.nan).fillna(0.0)

test_predictions_log = m.predict(test_set_proc)
test_predictions = np.expm1(test_predictions_log)
test_predictions = np.clip(test_predictions, 0.0, None)

submission = pd.DataFrame({"key": test_key, "fare_amount": test_predictions})
submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
