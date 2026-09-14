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
scipy==1.15.3
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

3.5071

# 6. Current score

4.57213

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 6.78738) has done: 'I fix the datetime handling bug in `add_datepart` caused by timezone-aware `datetime64[ns, UTC]` dtypes by switching the dtype check to Pandas’ `is_datetime64_any_dtype`, which safely supports tz-aware timestamps. This ensure `pickup_datetime` is expanded into numeric date features and then dropped, preventing `Timestamp` objects from leaking into the RandomForest input (which caused the downstream `.fit()`/`.predict()` errors). I also keep the original modeling approach intact and only add a small, standard NYC Taxi feature (`haversine` distance) to help RMSE move toward the target without changing the overall pipeline structure. Finally, the script reliably write a valid `submission.csv` with the required `key,fare_amount` columns.'
- What this solution (achieved 4.39571) has done: 'Your current RMSE (6.787) is far worse than the target (3.507), so we should improve model signal with minimal, competition-standard feature additions while keeping the same RandomForest approach and training loop. The biggest issue is that the model is only seeing raw coordinates plus dateparts, but not key engineered geography/temporal signals; adding a few lightweight distance/geometry features (Manhattan distance, bearing, and absolute deltas) typically moves NYC Taxi RMSE down substantially without changing the model class. I also apply the exact same geographic filtering bounds to the test set (but only clipping/setting out-of-bounds to NaN rather than dropping rows) so train/test distributions are more consistent and `proc_df` can impute safely. All changes are narrowly targeted to improve RMSE and preserve your existing pipeline semantics (same RF, same proc_df, same submission format).'
- What this solution (achieved 4.36388) has done: 'Your current RMSE (4.39571) is worse than the target (3.5071), so we should add small, standard NYC Taxi signal features while keeping the exact same RandomForest training approach. The most leverage with minimal disruption is (1) filtering obvious fare outliers in training (negative/too-large fares) and (2) adding a couple of well-known location features (JFK/LGA/Manhattan distance proxies) and cyclical time encodings, which RandomForest uses well. These additions keep your pipeline semantics intact (same model class, same fit/predict flow, same proc_df) but typically reduce error meaningfully. We also ensure train/test get the same new features and keep the submission format unchanged.'
- What this solution (achieved 4.34627) has done: 'Your current RMSE (4.36388) is still worse than the target (3.5071), so the smallest likely-to-help change is to make the train/test feature processing more consistent by applying the same geographic bounds mask to *both* pickup and dropoff coordinates (train currently filters only pickup bounds). I also add two lightweight, standard NYC features that don’t change the modeling approach: the raw deltas (`dlon`, `dlat`) and a simple “airport trip indicator” based on distance-to-airports already computed, which RandomForest can use effectively. These are narrow feature-engineering tweaks (no change to model class, fitting flow, or loss/metric) and should move RMSE downward toward the target band without increasing runtime meaningfully. Submission writing stays identical and still produces `submission.csv` with `key,fare_amount`.'
- What this solution (achieved 4.34371) has done: 'We need to move RMSE down from 4.346 toward 3.507, so we add a tiny amount of additional signal without changing your model class or training/prediction flow. The most “minimal but high leverage” change here is to add a simple passenger_count cleaning step (invalid 0/large counts hurt) and a well-known short-feature: straight-line “euclidean” distance in lat/lon space (a different projection than haversine/manhattan, often complementary for tree models). We also make the train/test preprocessing more consistent by applying the same passenger_count normalization and ensuring any remaining NaNs are handled identically via the existing `proc_df` median-imputation (no new training tricks). Everything else (RandomForestRegressor params, dateparts, existing engineered features, submission format/path) remains unchanged.'
- What this solution (achieved 4.35201) has done: 'Your RMSE (4.34371) is still worse than the target (3.5071), so we should improve feature signal slightly while keeping the exact same RandomForest training flow and core feature set. The most minimal high-impact fix for this competition is to add `log1p(haversine_km)` and `haversine_km^2`, which helps trees model the strongly non-linear fare-vs-distance relationship without changing the model class or loss. I also add a simple `same_loc` flag (pickup/dropoff nearly identical) to better handle very short trips that otherwise inflate error. These features are added identically to train and test, and submission writing/format stays unchanged.'
- What this solution (achieved 4.19234) has done: 'Your current RMSE (4.35201) is worse than the target (3.5071), so we should improve generalization with the smallest possible change that keeps your RandomForest pipeline intact. The highest-leverage minimal fix here is to make the training sample larger (still lightweight vs the full 55M rows) because RandomForest benefits substantially from more diverse examples, and this typically reduces leaderboard RMSE without changing model semantics. I also add a tiny, standard coordinate-validity filter to remove “zero coordinate” rows (0,0 etc.) that inject strong noise, while keeping all your existing feature engineering and model parameters unchanged. Submission writing/format stays identical.'
- What this solution (achieved 4.57213) has done: 'We need to move RMSE down from 4.192 toward 3.507 (lower is better), so the smallest likely-to-help change is to tighten training-data quality without changing your RandomForest model or training loop. I add two standard NYC Taxi cleaning filters: remove extreme/unrealistic trip distances (based on your existing haversine) and cap passenger_count to valid values while also filtering rows with missing/invalid timestamps/coordinates after parsing. These are narrow data-quality fixes that reduce label noise and outliers, which typically lowers RMSE for the same model and features. The rest of your feature engineering, model hyperparameters, and submission writing remain unchanged.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from IPython.display import display


def set_plot_sizes(sml, med, big):
    plt.rc("figure", figsize=(med, med))
    plt.rc("axes", titlesize=med)
    plt.rc("axes", labelsize=med)
    plt.rc("xtick", labelsize=sml)
    plt.rc("ytick", labelsize=sml)
    plt.rc("legend", fontsize=sml)
    plt.rc("font", size=med)


def display_all(df):
    with pd.option_context("display.max_rows", 1000, "display.max_columns", 1000):
        display(df)


def train_cats(df: pd.DataFrame):
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(
            df[col]
        ):
            df[col] = df[col].astype("category")


def add_datepart(
    df: pd.DataFrame, field_name: str, drop: bool = True, time: bool = False
):
    field = df[field_name]
    if not pd.api.types.is_datetime64_any_dtype(field):
        df[field_name] = pd.to_datetime(field, errors="coerce", utc=True)
    field = df[field_name]

    prefix = field_name.replace("date", "").replace("Date", "")
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
    for a in attrs:
        if a == "Week":
            df[prefix + a] = field.dt.isocalendar().week.astype("Int16")
        else:
            df[prefix + a] = (
                getattr(field.dt, a.lower())
                if hasattr(field.dt, a.lower())
                else getattr(field.dt, a)
            )
    if time:
        for a in ["Hour", "Minute", "Second"]:
            df[prefix + a] = getattr(field.dt, a.lower())

    if drop:
        df.drop(columns=[field_name], inplace=True)


def proc_df(
    df: pd.DataFrame, y_fld: str = None, na_dict: dict = None, add_missing: bool = True
):
    df = df.copy()
    y = None
    if y_fld is not None and y_fld in df.columns:
        y = df[y_fld].astype(np.float32).values
        df.drop(columns=[y_fld], inplace=True)

    if na_dict is None:
        na_dict = {}

    for col in df.columns:
        s = df[col]
        if pd.api.types.is_categorical_dtype(s):
            df[col] = s.cat.codes.replace({-1: np.nan}).astype("float32")
        elif pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
            df[col] = (
                s.astype("category").cat.codes.replace({-1: np.nan}).astype("float32")
            )

        if pd.api.types.is_numeric_dtype(df[col]):
            if df[col].isnull().any():
                if col not in na_dict:
                    na_dict[col] = df[col].median()
                if add_missing:
                    df[col + "_na"] = df[col].isnull().astype("int8")
                df[col] = df[col].fillna(na_dict[col])

    if y_fld is None:
        return df
    return df, y, na_dict


def rf_feat_importance(m, df):
    return pd.DataFrame(
        {"cols": df.columns, "imp": m.feature_importances_}
    ).sort_values("imp", ascending=False)


def haversine_np(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))  # km


def bearing_np(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    brng = np.degrees(np.arctan2(y, x))
    return (brng + 360.0) % 360.0


def add_geo_features(df: pd.DataFrame):
    dlon = (df["dropoff_longitude"] - df["pickup_longitude"]).astype("float32")
    dlat = (df["dropoff_latitude"] - df["pickup_latitude"]).astype("float32")
    df["dlon"] = dlon
    df["dlat"] = dlat

    df["abs_dlon"] = np.abs(dlon).astype("float32")
    df["abs_dlat"] = np.abs(dlat).astype("float32")
    df["manhattan_km"] = (df["abs_dlon"] * 85.0 + df["abs_dlat"] * 111.0).astype(
        "float32"
    )
    df["bearing"] = bearing_np(
        df["pickup_longitude"].values,
        df["pickup_latitude"].values,
        df["dropoff_longitude"].values,
        df["dropoff_latitude"].values,
    ).astype("float32")
    return df


def apply_geo_bounds_train(df: pd.DataFrame):
    df = df[df["pickup_longitude"] > -76]
    df = df[df["pickup_longitude"] < -73]
    df = df[df["pickup_latitude"] > 40]
    df = df[df["pickup_latitude"] < 44]
    df = df[df["dropoff_longitude"] > -76]
    df = df[df["dropoff_longitude"] < -73]
    df = df[df["dropoff_latitude"] > 40]
    df = df[df["dropoff_latitude"] < 44]
    return df


def apply_geo_bounds_test_to_nan(df: pd.DataFrame):
    mask = (
        (df["pickup_longitude"] > -76)
        & (df["pickup_longitude"] < -73)
        & (df["pickup_latitude"] > 40)
        & (df["pickup_latitude"] < 44)
        & (df["dropoff_longitude"] > -76)
        & (df["dropoff_longitude"] < -73)
        & (df["dropoff_latitude"] > 40)
        & (df["dropoff_latitude"] < 44)
    )
    loc_cols = [
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
    ]
    df.loc[~mask, loc_cols] = np.nan
    return df


def add_nyc_location_features(df: pd.DataFrame):
    jfk_lon, jfk_lat = -73.7781, 40.6413
    lga_lon, lga_lat = -73.8740, 40.7769
    ewr_lon, ewr_lat = -74.1745, 40.6895
    man_lon, man_lat = -73.9855, 40.7580  # Times Sq proxy

    df["pickup_to_jfk_km"] = haversine_np(
        df["pickup_longitude"].values, df["pickup_latitude"].values, jfk_lon, jfk_lat
    ).astype("float32")
    df["dropoff_to_jfk_km"] = haversine_np(
        df["dropoff_longitude"].values, df["dropoff_latitude"].values, jfk_lon, jfk_lat
    ).astype("float32")

    df["pickup_to_lga_km"] = haversine_np(
        df["pickup_longitude"].values, df["pickup_latitude"].values, lga_lon, lga_lat
    ).astype("float32")
    df["dropoff_to_lga_km"] = haversine_np(
        df["dropoff_longitude"].values, df["dropoff_latitude"].values, lga_lon, lga_lat
    ).astype("float32")

    df["pickup_to_ewr_km"] = haversine_np(
        df["pickup_longitude"].values, df["pickup_latitude"].values, ewr_lon, ewr_lat
    ).astype("float32")
    df["dropoff_to_ewr_km"] = haversine_np(
        df["dropoff_longitude"].values, df["dropoff_latitude"].values, ewr_lon, ewr_lat
    ).astype("float32")

    df["pickup_to_man_km"] = haversine_np(
        df["pickup_longitude"].values, df["pickup_latitude"].values, man_lon, man_lat
    ).astype("float32")
    df["dropoff_to_man_km"] = haversine_np(
        df["dropoff_longitude"].values, df["dropoff_latitude"].values, man_lon, man_lat
    ).astype("float32")

    min_air_km = np.minimum.reduce(
        [
            df["pickup_to_jfk_km"].values,
            df["dropoff_to_jfk_km"].values,
            df["pickup_to_lga_km"].values,
            df["dropoff_to_lga_km"].values,
            df["pickup_to_ewr_km"].values,
            df["dropoff_to_ewr_km"].values,
        ]
    ).astype("float32")
    df["is_airport_trip"] = (min_air_km < 2.0).astype("int8")
    return df


def add_cyclical_time_features(df: pd.DataFrame, prefix: str = "pickup_"):
    hr_col = prefix + "Hour"
    dow_col = prefix + "Dayofweek"
    if hr_col in df.columns:
        hr = df[hr_col].astype("float32")
        df["hour_sin"] = np.sin(2 * np.pi * hr / 24.0).astype("float32")
        df["hour_cos"] = np.cos(2 * np.pi * hr / 24.0).astype("float32")
    if dow_col in df.columns:
        dow = df[dow_col].astype("float32")
        df["dow_sin"] = np.sin(2 * np.pi * dow / 7.0).astype("float32")
        df["dow_cos"] = np.cos(2 * np.pi * dow / 7.0).astype("float32")
    return df


def clean_passenger_count(df: pd.DataFrame):
    if "passenger_count" in df.columns:
        pc = df["passenger_count"]
        bad = (pc < 1) | (pc > 6)
        if bad.any():
            df.loc[bad, "passenger_count"] = np.nan
    return df


def add_euclidean_degree_distance(df: pd.DataFrame):
    dlon = (df["dropoff_longitude"] - df["pickup_longitude"]).astype("float32")
    dlat = (df["dropoff_latitude"] - df["pickup_latitude"]).astype("float32")
    df["euclid_deg"] = np.sqrt(dlon * dlon + dlat * dlat).astype("float32")
    return df


def add_distance_nonlinear_features(df: pd.DataFrame):
    if "haversine_km" in df.columns:
        hk = df["haversine_km"].astype("float32")
        df["haversine_km2"] = (hk * hk).astype("float32")
        df["log1p_haversine_km"] = np.log1p(hk).astype("float32")
    return df


def add_same_location_flag(df: pd.DataFrame, eps_deg: float = 0.0001):
    dlon = (df["dropoff_longitude"] - df["pickup_longitude"]).astype("float32")
    dlat = (df["dropoff_latitude"] - df["pickup_latitude"]).astype("float32")
    df["same_loc"] = ((np.abs(dlon) < eps_deg) & (np.abs(dlat) < eps_deg)).astype(
        "int8"
    )
    return df


def drop_zero_coordinate_rows_train(df: pd.DataFrame):
    loc_cols = [
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
    ]
    mask = df[loc_cols].abs().sum(axis=1) > 0.0
    return df.loc[mask].copy()


def filter_unrealistic_distances_train(
    df: pd.DataFrame, min_km: float = 0.01, max_km: float = 100.0
):
    if "haversine_km" not in df.columns:
        return df
    hk = df["haversine_km"]
    return df.loc[hk.notna() & (hk >= min_km) & (hk <= max_km)].copy()


def drop_missing_critical_train(df: pd.DataFrame):
    crit = [
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "pickup_datetime",
        "fare_amount",
    ]
    keep = df[crit].notna().all(axis=1)
    return df.loc[keep].copy()


set_plot_sizes(12, 14, 16)



## === cell 1
PATH = "/kaggle/input/"

train_path = os.path.join(PATH, "train.csv")
test_path = os.path.join(PATH, "test.csv")

NROWS_TRAIN = 500000

df_raw = pd.read_csv(
    train_path,
    nrows=NROWS_TRAIN,
    parse_dates=["pickup_datetime"],
    dtype={"passenger_count": "int8", "fare_amount": "float32"},
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

df_raw_test = pd.read_csv(
    test_path, parse_dates=["pickup_datetime"], dtype={"passenger_count": "int8"}
)




## === cell 2
def display_all(df):
    with pd.option_context("display.max_rows", 1000, "display.max_columns", 1000):
        display(df)




## === cell 3
display_all(df_raw.tail().T)



## === cell 4
display_all(df_raw.describe(include="all").T)



## === cell 5
display_all(df_raw_test.describe(include="all").T)



## === cell 6
df_raw[df_raw["fare_amount"] < 0]



## === cell 7
df_raw[df_raw["pickup_longitude"] < -75]



## === cell 8
df_raw[df_raw["pickup_longitude"] > -73]



## === cell 9
df_raw[df_raw["pickup_latitude"] < 40]



## === cell 10
df_raw[df_raw["pickup_latitude"] > 42]



## === cell 11
df_raw.shape



## === cell 12
df_raw = df_raw[(df_raw["fare_amount"] >= 0) & (df_raw["fare_amount"] <= 250)].copy()



## === cell 13
df_raw = apply_geo_bounds_train(df_raw)



## === cell 14
df_raw = clean_passenger_count(df_raw)
df_raw = drop_zero_coordinate_rows_train(df_raw)

df_raw = drop_missing_critical_train(df_raw)



## === cell 15
df_raw.shape



## === cell 16
df_raw["haversine_km"] = haversine_np(
    df_raw["pickup_longitude"].values,
    df_raw["pickup_latitude"].values,
    df_raw["dropoff_longitude"].values,
    df_raw["dropoff_latitude"].values,
).astype("float32")
df_raw = add_geo_features(df_raw)
df_raw = add_euclidean_degree_distance(df_raw)
df_raw = add_nyc_location_features(df_raw)
df_raw = add_distance_nonlinear_features(df_raw)
df_raw = add_same_location_flag(df_raw)

df_raw = filter_unrealistic_distances_train(df_raw, min_km=0.01, max_km=100.0)



## === cell 17
train_cats(df_raw)



## === cell 18
add_datepart(df_raw, "pickup_datetime", time=True)
df_raw = add_cyclical_time_features(df_raw, prefix="pickup_")



## === cell 19
df_raw.info()



## === cell 20
df, y, nas = proc_df(df_raw, "fare_amount")



## === cell 21
m = RandomForestRegressor(
    n_estimators=30, min_samples_leaf=3, oob_score=True, n_jobs=-1, random_state=42
)
m.fit(df, y)



## === cell 22
fi = rf_feat_importance(m, df)
fi[:10]



## === cell 23
fi.plot("cols", "imp", figsize=(10, 6), legend=False)
plt.title("Feature Importance by Feature")




## === cell 24
def plot_fi(fi):
    return fi.plot("cols", "imp", "barh", figsize=(12, 7), legend=False)




## === cell 25
plot_fi(fi[:30])
plt.title("Feature Importance by Feature")



## === cell 26
from scipy.cluster import hierarchy as hc



## === cell 27
corr = np.round(scipy.stats.spearmanr(df).correlation, 4)
corr_condensed = hc.distance.squareform(1 - corr)
z = hc.linkage(corr_condensed, method="average")
fig = plt.figure(figsize=(16, 10))
_ = hc.dendrogram(z, labels=df.columns, orientation="left", leaf_font_size=10)
plt.title("Feature Similarities")
plt.show()



## === cell 28
df_raw_test = apply_geo_bounds_test_to_nan(df_raw_test)
df_raw_test = clean_passenger_count(df_raw_test)

df_raw_test["haversine_km"] = haversine_np(
    df_raw_test["pickup_longitude"].values,
    df_raw_test["pickup_latitude"].values,
    df_raw_test["dropoff_longitude"].values,
    df_raw_test["dropoff_latitude"].values,
).astype("float32")
df_raw_test = add_geo_features(df_raw_test)
df_raw_test = add_euclidean_degree_distance(df_raw_test)
df_raw_test = add_nyc_location_features(df_raw_test)
df_raw_test = add_distance_nonlinear_features(df_raw_test)
df_raw_test = add_same_location_flag(df_raw_test)



## === cell 29
train_cats(df_raw_test)



## === cell 30
add_datepart(df_raw_test, "pickup_datetime", time=True)
df_raw_test = add_cyclical_time_features(df_raw_test, prefix="pickup_")



## === cell 31
df_test = proc_df(df_raw_test.drop(columns=["key"]), y_fld=None, na_dict=nas)

missing_cols = [c for c in df.columns if c not in df_test.columns]
for c in missing_cols:
    df_test[c] = 0
df_test = df_test[df.columns]



## === cell 32
y_pred = m.predict(df_test).astype(np.float32)
y_pred = np.maximum(y_pred, 0)



## === cell 33
my_submission = pd.DataFrame({"key": df_raw_test["key"], "fare_amount": y_pred})
my_submission.to_csv("submission.csv", index=False)

print(my_submission.head())
print("Wrote submission.csv with shape:", my_submission.shape)
print("Submission columns:", list(my_submission.columns))
print("Submission path:", os.path.abspath("submission.csv"))
