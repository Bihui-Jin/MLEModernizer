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

3.6375985209102257

# 6. Current score

5.72582

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 18.16927) has done: 'I make the script compatible with the Kaggle runtime by (1) removing the failing TensorFlow import/setup and replacing it with a lightweight, deterministic baseline that runs without external model files, (2) fixing the datetime feature extraction bug (`.dt.week` no longer exists) so `add_datepart` creates the expected columns, and (3) correcting the distance function’s haversine computation. Since your current run yields no valid submission, the priority is to produce a correctly formatted `submission.csv` (`key,fare_amount`) end-to-end; the baseline uses a sensible distance-based fare estimate plus a small base fare, which should score materially better than a constant predictor while remaining simple and stable within the 600s limit.'
- What this solution (achieved 17.96101) has done: 'I fix the runtime error by removing the invalid `.values` access on a NumPy array when building the submission DataFrame. I also make the train/test numeric coercion consistent (same as train) so `distance()` never receives object dtypes, preventing silent NaNs and improving stability/score without changing the modeling approach. Finally, I make the submission alignment deterministic by ordering exactly like `sample_submission.csv` and filling any missing predictions with the mean prediction (should be rare), ensuring a valid `submission.csv` is always written.'
- What this solution (achieved 9.73915) has done: 'Your current RMSE (17.96) is far worse than the target (3.64), so we should legitimately improve the model while keeping the same overall “fit a simple linear model on engineered features” core logic. The biggest gain with minimal semantic change is to add a few standard NYC Taxi baseline features (absolute deltas, Euclidean distance proxy, and a Manhattan distance proxy) and to make the regression more robust by fitting on log1p(fare) then converting back—this reduces the impact of heavy-tailed fares and typically drops RMSE substantially. I also apply the same feature construction consistently to train/test and keep the same submission alignment logic. All paths remain unchanged and the script still produces `submission.csv`.'
- What this solution (achieved 9.82462) has done: 'Your current RMSE (9.739) is still far from the target (3.638), so we should improve the model but keep the same core approach: a simple linear regression on engineered features with log1p(target). The smallest high-impact change is to add two standard NYC-taxi baseline geometry features—pickup/dropoff distance to a fixed “NYC center” point and a rough bearing—while keeping the same training loop (NumPy lstsq) and the same prediction post-processing. I’m also fixing a silent bug in your `build_features`: it tries to use `pickup_datetimehour` after dropping `pickup_datetime`, which can lead to missing/incorrect hour features and hurts score. Finally, I keep the submission alignment exactly as you already do to ensure a valid `submission.csv`.'
- What this solution (achieved 9.55415) has done: 'We need to lower RMSE from 9.82 toward 3.64, so the model is underfitting and likely suffering from linear regression being too sensitive to outliers/feature scaling and a small training sample. I keep the same core “linear regression on engineered features with log1p(target)” approach, but make two minimal, high-impact, semantics-preserving upgrades: (1) standardize features using train-set mean/std before least squares (stabilizes coefficients and usually reduces RMSE), and (2) modestly increase the training sample size while staying within the runtime limit. I also add the standard NYC “airport distance” features (to JFK/LGA/EWR) which are a small extension of the existing distance-to-center idea and commonly provide a large RMSE drop without changing the model family. Submission writing, columns, and alignment remain identical.'
- What this solution (achieved 11.98909) has done: 'We need to reduce RMSE from 9.55 toward 3.64 (lower is better), so we should improve generalization without changing the core “linear regression on engineered features with log1p(target)” approach. The biggest likely issue is remaining outliers/noise in the 1.2M sample and mild under-specification around distance scaling; we keep the same features and least-squares solver, but add a single robustification step: fit with simple sample weights that downweight very long trips and unusually high fares (both are common label noise sources in this dataset). This preserves the same model family and semantics while typically giving a meaningful RMSE drop. We also make train/test preprocessing fully symmetric for NaN/inf handling before standardization to avoid distribution shift artifacts.'
- What this solution (achieved 17.55501) has done: 'I fix the runtime error caused by `np.issubdtype` not handling timezone-aware datetimes by switching the datetime check in `add_datepart` to use pandas’ dtype utilities and by normalizing any timezone-aware datetimes to naive UTC before feature extraction. This unblocks both train and test feature building, which then allows the linear regression fit and prediction steps to run end-to-end. I also add a small safeguard to ensure `pickup_datetimehour` exists even if `pickup_datetime` is missing/invalid, preventing downstream KeyErrors. No changes are made to the model family (weighted linear regression on standardized engineered features with log1p target), only robustness fixes so a valid `submission.csv` is always written.'
- What this solution (achieved 5.72582) has done: 'Your current RMSE is far above target, so we need a small, legitimate accuracy boost without changing the model family or training loop. The biggest likely issue is remaining label noise/outliers and a slight train/test mismatch due to missing “zero-distance” and coordinate sanity handling in `build_features`, which can destabilize linear least-squares. I add minimal, standard NYC-taxi cleaning (remove zero/near-zero coordinate rows; enforce lat/lon bounds and passenger_count bounds inside feature-building symmetry) and add one high-impact but still linear feature: `log1p(Herv_Dist)` to better linearize the distance–fare relationship. These are small, deterministic changes that preserve your weighted standardized linear regression on `log1p(fare_amount)` and should move RMSE down toward the target.'

# 9. Code solution

## === cell 0
import os
import re
import numpy as np
import pandas as pd

print(os.listdir("../input"))


## === cell 1
dnn_path = "../input/dnn-model"
if os.path.exists(dnn_path):
    print("Found dnn-model:", os.listdir(dnn_path))
else:
    print("dnn-model path not found, proceeding without it:", dnn_path)


## === cell 2
df_test = pd.read_csv("../input/new-york-city-taxi-fare-prediction/test.csv")


## === cell 3
df_test.head()


## === cell 4
sample_sub = pd.read_csv(
    "../input/new-york-city-taxi-fare-prediction/sample_submission.csv"
)
print(sample_sub.head())
print("test rows:", len(df_test), "sample rows:", len(sample_sub))




## === cell 5
def add_datepart(df, fldname, drop=True):
    """
    Bugfix: handle timezone-aware datetimes (e.g., datetime64[ns, UTC]) safely.
    - Use pandas dtype checks instead of np.issubdtype (which fails on tz-aware dtype).
    - Normalize tz-aware timestamps to naive UTC before extracting date parts.
    - Keep the same output column naming expected downstream (e.g., pickup_datetimehour).
    """
    fld = df[fldname]

    if not pd.api.types.is_datetime64_any_dtype(fld):
        df[fldname] = fld = pd.to_datetime(
            fld, infer_datetime_format=True, errors="coerce"
        )

    if pd.api.types.is_datetime64tz_dtype(df[fldname].dtype):
        df[fldname] = df[fldname].dt.tz_convert("UTC").dt.tz_localize(None)
        fld = df[fldname]

    targ_pre = re.sub("[Dd]ate$", "", fldname)

    df[targ_pre + "Year"] = fld.dt.year
    df[targ_pre + "Month"] = fld.dt.month
    df[targ_pre + "Day"] = fld.dt.day
    df[targ_pre + "Dayofweek"] = fld.dt.dayofweek
    df[targ_pre + "Dayofyear"] = fld.dt.dayofyear
    df[targ_pre + "hour"] = fld.dt.hour

    try:
        df[targ_pre + "Week"] = fld.dt.isocalendar().week.astype(np.int16)
    except Exception:
        df[targ_pre + "Week"] = fld.dt.strftime("%V").astype("int16")

    df[targ_pre + "Is_month_end"] = fld.dt.is_month_end
    df[targ_pre + "Is_month_start"] = fld.dt.is_month_start
    df[targ_pre + "Is_quarter_end"] = fld.dt.is_quarter_end
    df[targ_pre + "Is_quarter_start"] = fld.dt.is_quarter_start
    df[targ_pre + "Is_year_end"] = fld.dt.is_year_end
    df[targ_pre + "Is_year_start"] = fld.dt.is_year_start

    df[targ_pre + "Elapsed"] = (fld.view("int64") // 10**9).astype("float64")
    df.loc[fld.isna(), targ_pre + "Elapsed"] = np.nan

    if drop:
        df.drop(fldname, axis=1, inplace=True)


def distance(data):
    """
    Haversine distance in km.
    Input data expected as Nx4 [lon1, lat1, lon2, lat2].
    """
    radius = 6371.0  # km
    lon1 = data[:, 0]
    lat1 = data[:, 1]
    lon2 = data[:, 2]
    lat2 = data[:, 3]

    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    lat1r = np.radians(lat1)
    lat2r = np.radians(lat2)

    a = (np.sin(dlat / 2.0) ** 2) + np.cos(lat1r) * np.cos(lat2r) * (
        np.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return radius * c


def build_features(df):
    """
    Same core feature engineering; minimal accuracy-oriented fixes:
    - Add symmetric coordinate/passenger cleaning inside feature building to reduce train/test mismatch.
    - Add log1p(distance) feature to linearize the dominant distance-fare relationship while staying linear.
    """
    for c in [
        "pickup_longitude",
        "dropoff_longitude",
        "pickup_latitude",
        "dropoff_latitude",
        "passenger_count",
    ]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    nyc_lon_min, nyc_lon_max = -74.5, -72.8
    nyc_lat_min, nyc_lat_max = 40.0, 41.8
    coord_ok = (
        df["pickup_longitude"].between(nyc_lon_min, nyc_lon_max)
        & df["dropoff_longitude"].between(nyc_lon_min, nyc_lon_max)
        & df["pickup_latitude"].between(nyc_lat_min, nyc_lat_max)
        & df["dropoff_latitude"].between(nyc_lat_min, nyc_lat_max)
    )
    not_same = ~(
        (df["pickup_longitude"] == df["dropoff_longitude"])
        & (df["pickup_latitude"] == df["dropoff_latitude"])
    )
    df["_coord_ok"] = (coord_ok & not_same).astype("int8")

    df["Herv_Dist"] = distance(
        df[
            [
                "pickup_longitude",
                "pickup_latitude",
                "dropoff_longitude",
                "dropoff_latitude",
            ]
        ]
        .astype("float64")
        .values
    ).astype("float64")

    dlon = (df["dropoff_longitude"] - df["pickup_longitude"]).astype("float64")
    dlat = (df["dropoff_latitude"] - df["pickup_latitude"]).astype("float64")
    df["abs_dlon"] = np.abs(dlon)
    df["abs_dlat"] = np.abs(dlat)
    df["euclid_deg"] = np.sqrt(dlon * dlon + dlat * dlat)
    df["manhattan_deg"] = np.abs(dlon) + np.abs(dlat)

    add_datepart(df, "pickup_datetime", drop=True)

    if "pickup_datetimehour" not in df.columns:
        df["pickup_datetimehour"] = np.nan

    hour = df["pickup_datetimehour"].astype("float64").fillna(0.0)
    df["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    df["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)

    nyc_center_lon = -73.985428
    nyc_center_lat = 40.748817

    df["pickup_to_center_km"] = distance(
        np.column_stack(
            [
                df["pickup_longitude"].astype("float64").values,
                df["pickup_latitude"].astype("float64").values,
                np.full(len(df), nyc_center_lon, dtype="float64"),
                np.full(len(df), nyc_center_lat, dtype="float64"),
            ]
        )
    ).astype("float64")

    df["dropoff_to_center_km"] = distance(
        np.column_stack(
            [
                df["dropoff_longitude"].astype("float64").values,
                df["dropoff_latitude"].astype("float64").values,
                np.full(len(df), nyc_center_lon, dtype="float64"),
                np.full(len(df), nyc_center_lat, dtype="float64"),
            ]
        )
    ).astype("float64")

    bearing = np.arctan2(dlat.values, dlon.values).astype("float64")
    df["bearing_sin"] = np.sin(bearing)
    df["bearing_cos"] = np.cos(bearing)

    jfk_lon, jfk_lat = -73.7781, 40.6413
    lga_lon, lga_lat = -73.8740, 40.7769
    ewr_lon, ewr_lat = -74.1745, 40.6895

    df["pickup_to_jfk_km"] = distance(
        np.column_stack(
            [
                df["pickup_longitude"].astype("float64").values,
                df["pickup_latitude"].astype("float64").values,
                np.full(len(df), jfk_lon, dtype="float64"),
                np.full(len(df), jfk_lat, dtype="float64"),
            ]
        )
    ).astype("float64")
    df["dropoff_to_jfk_km"] = distance(
        np.column_stack(
            [
                df["dropoff_longitude"].astype("float64").values,
                df["dropoff_latitude"].astype("float64").values,
                np.full(len(df), jfk_lon, dtype="float64"),
                np.full(len(df), jfk_lat, dtype="float64"),
            ]
        )
    ).astype("float64")

    df["pickup_to_lga_km"] = distance(
        np.column_stack(
            [
                df["pickup_longitude"].astype("float64").values,
                df["pickup_latitude"].astype("float64").values,
                np.full(len(df), lga_lon, dtype="float64"),
                np.full(len(df), lga_lat, dtype="float64"),
            ]
        )
    ).astype("float64")
    df["dropoff_to_lga_km"] = distance(
        np.column_stack(
            [
                df["dropoff_longitude"].astype("float64").values,
                df["dropoff_latitude"].astype("float64").values,
                np.full(len(df), lga_lon, dtype="float64"),
                np.full(len(df), lga_lat, dtype="float64"),
            ]
        )
    ).astype("float64")

    df["pickup_to_ewr_km"] = distance(
        np.column_stack(
            [
                df["pickup_longitude"].astype("float64").values,
                df["pickup_latitude"].astype("float64").values,
                np.full(len(df), ewr_lon, dtype="float64"),
                np.full(len(df), ewr_lat, dtype="float64"),
            ]
        )
    ).astype("float64")
    df["dropoff_to_ewr_km"] = distance(
        np.column_stack(
            [
                df["dropoff_longitude"].astype("float64").values,
                df["dropoff_latitude"].astype("float64").values,
                np.full(len(df), ewr_lon, dtype="float64"),
                np.full(len(df), ewr_lat, dtype="float64"),
            ]
        )
    ).astype("float64")

    df["Herv_Dist_sq"] = (df["Herv_Dist"].astype("float64") ** 2).astype("float64")

    df["log_Herv_Dist"] = np.log1p(df["Herv_Dist"].astype("float64"))

    df["Herv_Dist"] = df["Herv_Dist"].fillna(0.0).clip(0.0, 200.0)
    df["Herv_Dist_sq"] = df["Herv_Dist_sq"].fillna(0.0).clip(0.0, 200.0**2)
    df["log_Herv_Dist"] = (
        df["log_Herv_Dist"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .clip(0.0, np.log1p(200.0))
    )

    df["abs_dlon"] = df["abs_dlon"].fillna(0.0).clip(0.0, 10.0)
    df["abs_dlat"] = df["abs_dlat"].fillna(0.0).clip(0.0, 10.0)
    df["euclid_deg"] = df["euclid_deg"].fillna(0.0).clip(0.0, 10.0)
    df["manhattan_deg"] = df["manhattan_deg"].fillna(0.0).clip(0.0, 20.0)
    df["passenger_count"] = (
        df["passenger_count"].astype("float64").fillna(1.0).clip(1.0, 6.0)
    )

    df["pickup_to_center_km"] = df["pickup_to_center_km"].fillna(0.0).clip(0.0, 200.0)
    df["dropoff_to_center_km"] = df["dropoff_to_center_km"].fillna(0.0).clip(0.0, 200.0)
    df["bearing_sin"] = df["bearing_sin"].fillna(0.0).clip(-1.0, 1.0)
    df["bearing_cos"] = df["bearing_cos"].fillna(0.0).clip(-1.0, 1.0)

    for c in [
        "pickup_to_jfk_km",
        "dropoff_to_jfk_km",
        "pickup_to_lga_km",
        "dropoff_to_lga_km",
        "pickup_to_ewr_km",
        "dropoff_to_ewr_km",
    ]:
        df[c] = df[c].fillna(0.0).clip(0.0, 200.0)

    df["dist_x_pass"] = (
        df["Herv_Dist"].astype("float64") * df["passenger_count"].astype("float64")
    ).astype("float64")
    df["dist_x_pass"] = df["dist_x_pass"].fillna(0.0).clip(0.0, 200.0 * 6.0)

    return df




## === cell 6
class DNN_Model:
    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "Original TensorFlow DNN model is not supported in this runtime (tf.contrib removed and no checkpoint provided)."
        )




## === cell 7
RANDOM_SEED = 42
TRAIN_PATH = "../input/new-york-city-taxi-fare-prediction/train.csv"

N_TRAIN_SAMPLE = 1200000

usecols_train = [
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]

df_train = pd.read_csv(
    TRAIN_PATH,
    usecols=usecols_train,
    nrows=N_TRAIN_SAMPLE,
)

df_train["fare_amount"] = pd.to_numeric(df_train["fare_amount"], errors="coerce")

df_train["pickup_datetime"] = pd.to_datetime(
    df_train["pickup_datetime"], infer_datetime_format=True, errors="coerce"
)

df_train = df_train.dropna(subset=usecols_train)

nyc_lon_min, nyc_lon_max = -74.5, -72.8
nyc_lat_min, nyc_lat_max = 40.0, 41.8

coord_mask = (
    (df_train["pickup_longitude"].between(nyc_lon_min, nyc_lon_max))
    & (df_train["dropoff_longitude"].between(nyc_lon_min, nyc_lon_max))
    & (df_train["pickup_latitude"].between(nyc_lat_min, nyc_lat_max))
    & (df_train["dropoff_latitude"].between(nyc_lat_min, nyc_lat_max))
)

fare_mask = df_train["fare_amount"].between(2.5, 300.0)
pass_mask = df_train["passenger_count"].between(1, 6)

df_train = df_train[coord_mask & fare_mask & pass_mask].copy()

df_train = build_features(df_train)

df_train = df_train[df_train["_coord_ok"] == 1].copy()

min_km = 0.05
max_km = 80.0
dist_mask = df_train["Herv_Dist"].between(min_km, max_km)
df_train = df_train[dist_mask].copy()

X_cols = [
    "Herv_Dist",
    "log_Herv_Dist",
    "Herv_Dist_sq",
    "dist_x_pass",
    "euclid_deg",
    "manhattan_deg",
    "abs_dlon",
    "abs_dlat",
    "pickup_to_center_km",
    "dropoff_to_center_km",
    "pickup_to_jfk_km",
    "dropoff_to_jfk_km",
    "pickup_to_lga_km",
    "dropoff_to_lga_km",
    "pickup_to_ewr_km",
    "dropoff_to_ewr_km",
    "bearing_sin",
    "bearing_cos",
    "passenger_count",
    "pickup_datetimeDayofweek",
    "pickup_datetimeMonth",
    "hour_sin",
    "hour_cos",
]

X = df_train[X_cols].astype("float64").values
y = df_train["fare_amount"].astype("float64").values
y_log = np.log1p(y)

dist_km = df_train["Herv_Dist"].astype("float64").values
w = 1.0 / (1.0 + (dist_km / 20.0) ** 2 + (y / 50.0) ** 2)
w = w.astype("float64")
w = np.clip(w, 0.05, 1.0)

X = np.where(np.isfinite(X), X, np.nan)
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

x_mean = np.mean(X, axis=0)
x_std = np.std(X, axis=0)
x_std = np.where(x_std == 0.0, 1.0, x_std)
Xz = (X - x_mean) / x_std

X_design = np.concatenate([np.ones((Xz.shape[0], 1), dtype="float64"), Xz], axis=1)

sqrt_w = np.sqrt(w).reshape(-1, 1)
Xw = X_design * sqrt_w
yw = y_log * sqrt_w.ravel()

beta, _, _, _ = np.linalg.lstsq(Xw, yw, rcond=None)

print("Training rows after filters:", len(df_train))
print(
    "Fitted weighted linear coefficients on standardized features (intercept + {}):".format(
        len(X_cols)
    )
)
print(beta)


## === cell 8
df_test["pickup_datetime"] = pd.to_datetime(
    df_test["pickup_datetime"], infer_datetime_format=True, errors="coerce"
)
df_test = build_features(df_test)

Xt = df_test[X_cols].astype("float64").values
Xt = np.where(np.isfinite(Xt), Xt, np.nan)
Xt = np.nan_to_num(Xt, nan=0.0, posinf=0.0, neginf=0.0)

Xtz = (Xt - x_mean) / x_std
Xt_design = np.concatenate([np.ones((Xtz.shape[0], 1), dtype="float64"), Xtz], axis=1)

fare_pred_log = Xt_design.dot(beta).astype("float64")
fare_pred = np.expm1(fare_pred_log).astype("float64")

fallback_value = (
    float(np.median(np.expm1(np.concatenate([y_log[:100000], y_log[-100000:]]))))
    if len(y_log) >= 200000
    else float(np.median(np.expm1(y_log)))
)
invalid_mask = df_test["_coord_ok"].astype("int8").values == 0
fare_pred[invalid_mask] = fallback_value

fare_pred = np.clip(fare_pred, 2.50, 300.0)


## === cell 9
my_submission = pd.DataFrame(
    {
        "key": df_test["key"].astype(str).values,
        "fare_amount": fare_pred.astype("float64"),
    }
)

if "key" in sample_sub.columns:
    my_submission = sample_sub[["key"]].merge(my_submission, on="key", how="left")

if my_submission["fare_amount"].isna().any():
    my_submission["fare_amount"] = my_submission["fare_amount"].fillna(
        float(np.nanmean(fare_pred))
    )

my_submission.to_csv("submission.csv", index=False)
print(my_submission.head())
print("Wrote submission.csv with shape:", my_submission.shape)
print("fare_amount stats:", my_submission["fare_amount"].describe())
