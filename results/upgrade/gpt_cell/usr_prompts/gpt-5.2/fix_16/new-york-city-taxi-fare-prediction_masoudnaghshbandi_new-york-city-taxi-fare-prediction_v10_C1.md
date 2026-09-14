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

3.77552

# 6. Current score

5.13352

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 5.34119) has done: 'Your code likely didn’t yield a Kaggle score because the submission is missing rows: you filter test rows by nulls and geographic bounds, which drops keys and produces an invalid/incomplete submission file. To move toward the target RMSE, keep your training/modeling exactly the same but ensure the submission contains *all* test keys by predicting for every row (imputing any missing values and not dropping out-of-bound rows). Additionally, the current training filter drops `passenger_count == 208`, which looks like a typo and can remove valid signal; changing it to `passenger_count == 0` (already present) is a minimal bug-fix that should slightly improve generalization without changing core logic. These changes are directly aimed at producing a valid submission and improving score stability toward the target.'
- What this solution (achieved 6.0534) has done: 'Your current gap is 5.34119 − 3.77552 = 1.56567 RMSE (lower is better), so we should make small, safe changes that typically improve generalization without changing the core model/training loop. The biggest score drag in this competition is usually bad labels and outliers, so we add minimal, standard NYC Taxi cleaning on the training subset only: remove non-positive/huge fares and obvious coordinate outliers, and add a single strong feature (haversine distance) while keeping the same XGBoost regressor and scaling workflow. We not drop any test rows (still predict for every test key), and we also align the XGBRegressor objective to the modern equivalent (`reg:squarederror`) to avoid deprecated behavior differences. These changes are small but commonly move RMSE materially toward your target without altering the overall approach.'
- What this solution (achieved 6.50194) has done: 'Your current RMSE (6.0534) is worse than the target (3.77552), so we should make small, low-risk improvements that usually reduce error without changing the overall XGBoost + scaling approach. The largest easy gain here is to add a few standard time-based features from `pickup_datetime` (hour, dayofweek, month, year) for both train and test, while keeping the same model type and training flow. We also clip negative test predictions to 0 (fares can’t be negative), which typically improves RMSE slightly without affecting submission validity. Everything else (data loading, cleaning, scaler, XGBRegressor training, and full-row test submission) remains the same.'
- What this solution (achieved 5.14613) has done: 'Your current RMSE (6.50194) is much worse than the target (3.77552), so we should make a small, standard generalization improvement without changing the overall approach (same XGBoost regressor, same scaling, same train/valid split, same feature set). The biggest low-risk gain here is to log-transform the target (`fare_amount`) during training and invert the transform at prediction time, which typically reduces the impact of high-fare outliers under RMSE while preserving the same model/loop semantics. I keep all cleaning/feature engineering the same, only adjusting `y_train/y_valid` used in `fit` and the post-processing for both validation and test predictions. Submission creation remains identical and still includes all test keys.'
- What this solution (achieved 5.50476) has done: 'Your current RMSE (5.14613) is worse than the target (3.77552), so we should make a small, low-risk improvement without changing the overall XGBoost+scaling+feature engineering pipeline. The biggest stability gain here is to remove a few remaining high-leverage outliers in the *training subset only* (unrealistic long trips and extreme per-km fares) that can dominate RMSE even after the log1p target transform. I keep the same model, same train/valid split, same features (time features + haversine), and still predict for all test rows/keys. The submission format and path remain unchanged, still writing `submission1.csv`.'
- What this solution (achieved 5.52465) has done: 'You’re currently worse than the target (RMSE 5.50 vs 3.78), so we should make small, standard “signal” improvements without changing the core XGBoost+scaling+log1p approach. The largest low-risk gain is to add two classic NYC-taxi features derived from your existing inputs: straight-line distance in degrees and the Manhattan (L1) distance in degrees; these usually reduce RMSE materially while keeping the same modeling pipeline. I also increase `n_estimators` slightly and set a conservative `learning_rate` (still the same model/training flow) to reduce underfitting on 1M rows, and I keep the submission logic unchanged (still predicting for all test keys). All other cleaning, split, scaling, log-target, and submission formatting remains the same.'
- What this solution (achieved 5.54094) has done: 'Your current RMSE (5.52465) is worse than the target (3.77552), so we should make a small, low-risk improvement that usually reduces error without changing the overall XGBoost+scaling+feature engineering pipeline. The strongest “minimal-change” gain in this competition is to add a single NYC-specific prior: distance to Manhattan’s center (a proxy for likely trip type and base fare), computed from existing coordinates and applied identically to train/test. I also make the train/test datetime parsing more robust by explicitly using `utc=True, errors="coerce"` everywhere (same semantics as you already use) and ensure feature columns are aligned by reindexing test to the training columns before scaling (prevents silent column-order issues). Everything else (data cleaning, log1p target training, model type and training flow, submission writing) remains the same.'
- What this solution (achieved 5.2675) has done: 'We’re currently worse than the target (5.54094 vs 3.77552 RMSE; lower is better), so we should make small changes that typically reduce error without changing the overall XGBoost+scaling+log1p pipeline. The biggest likely issue is underfitting/poor calibration from training on only 1M rows with many trees but no early stopping; a minimal, safe improvement is to use XGBoost’s robust squared-log error objective (`reg:squaredlogerror`) while keeping the same log1p target workflow (still predicting fares and writing the same submission). Additionally, we add two classic low-risk geospatial features (bearing and airport-distance priors) computed from existing coordinates, which often yields a measurable RMSE drop in this competition while preserving the same model type and training loop. Finally, we apply the exact same feature engineering to both train and test via a shared function to avoid any subtle train/test mismatch that can silently hurt leaderboard score.'
- What this solution (achieved 5.24925) has done: 'Your current RMSE (5.2675) is worse than the target (3.77552), so we should make a small, low-risk improvement that preserves the same XGBoost + StandardScaler + log1p workflow. The biggest likely mismatch is using `objective="reg:squaredlogerror"` while also training on `log1p(y)`, which effectively applies a log-like error twice and can hurt calibration; switching back to `reg:squarederror` while keeping your log1p target transform is a minimal semantic fix. I also clip training predictions the same way you clip test predictions to keep train/valid behavior aligned, and I compute test imputations from `X_train` (not all `X`) to avoid slight leakage (small but safe). Everything else (cleaning rules, feature engineering, split, training loop, and submission writing) remains the same and still produces `submission1.csv`.'
- What this solution (achieved 5.26065) has done: 'Your current RMSE (5.24925) is worse than the target (3.77552), so we should make a small, standard improvement that usually reduces generalization error without changing the overall XGBoost + scaling + log1p workflow. The biggest remaining easy gain is to include the raw latitude/longitude coordinates in a better-conditioned way by adding simple “center point” features (midpoint lon/lat) and absolute diffs (replacing signed diffs) while keeping your existing haversine/bearing/airport/manhattan features and the exact same training loop. This keeps core logic identical (same model type, same fit/predict flow, same loss/metric semantics) but often improves RMSE because the model can learn location-dependent base fare effects more directly. Everything still runs end-to-end and writes a valid `submission1.csv` with all test keys.'
- What this solution (achieved 5.22767) has done: 'Your current RMSE (5.26065) is worse than the target (3.77552), so we should make a small, low-risk improvement to reduce error without changing the overall XGBoost + scaling + log1p training flow. The biggest likely remaining issue is that the model isn’t given an explicit “distance-type” signal in the same units as fare; we add one standard, competition-proven feature: a simple “euclidean distance in km” approximation (lon/lat deltas scaled by latitude), applied identically to train and test. This preserves the same model type, objective, training loop, and evaluation semantics, but usually improves RMSE meaningfully. Everything else remains the same and the script still writes a complete `submission1.csv` with all test keys.'
- What this solution (achieved 5.13352) has done: 'We’re currently worse than the target (5.22767 vs 3.77552 RMSE; lower is better), so we should make small, legitimate accuracy improvements without changing the overall pipeline (same XGBoost regressor + StandardScaler + log1p target + existing feature set). The biggest low-risk gain left is to correct the mismatch between training on a *random global sample* of 1M rows and the true data distribution: we sample 1M rows uniformly from the full training file using chunked reading (same row budget, but much more representative). Additionally, we add one classic, minimal geospatial feature that often reduces error in this competition—pickup/dropoff location “bins” (rounded lat/lon)—which helps the model learn location-dependent base fare effects while keeping the same model and training flow. Everything else (cleaning rules, feature engineering functions, model hyperparameters, log1p training, predicting all test keys, and writing `submission1.csv`) remains intact.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error as MSE




## === cell 2
def read_train_uniform_sample(
    path,
    n_sample=1_000_000,
    chunksize=1_000_000,
    random_state=12,
    usecols=None,
):
    rng = np.random.default_rng(random_state)
    sampled_chunks = []
    total_seen = 0

    approx_total_rows = 55_423_856

    for chunk in pd.read_csv(path, chunksize=chunksize, usecols=usecols):
        m = len(chunk)
        total_seen += m

        take = int(round(n_sample * (m / approx_total_rows)))
        take = max(0, min(take, m))

        if take > 0:
            idx = rng.choice(m, size=take, replace=False)
            sampled_chunks.append(chunk.iloc[idx])

        if sum(len(x) for x in sampled_chunks) >= n_sample:
            break

    if not sampled_chunks:
        return pd.read_csv(path, nrows=n_sample, usecols=usecols)

    df = pd.concat(sampled_chunks, ignore_index=True)
    if len(df) > n_sample:
        df = df.sample(n=n_sample, random_state=random_state).reset_index(drop=True)
    return df


train_path = "../input/new-york-city-taxi-fare-prediction/train.csv"
test_path = "../input/new-york-city-taxi-fare-prediction/test.csv"

usecols_train = [
    "key",
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
usecols_test = [
    "key",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]

train = read_train_uniform_sample(
    train_path,
    n_sample=1_000_000,
    chunksize=1_000_000,
    random_state=12,
    usecols=usecols_train,
)
test_raw = pd.read_csv(test_path, usecols=usecols_test)



## === cell 3
train.shape, test_raw.shape



## === cell 4
train.head()



## === cell 5
train.isnull().sum()



## === cell 6
train = train.dropna(how="any", axis="rows")



## === cell 7
test_raw.isnull().sum()



## === cell 8
train.head()



## === cell 9
train["fare_amount"].describe()



## === cell 10
train = train[(train["fare_amount"] > 0) & (train["fare_amount"] < 250)].copy()

train.drop(train[train["pickup_longitude"] == 0].index, axis=0, inplace=True)
train.drop(train[train["pickup_latitude"] == 0].index, axis=0, inplace=True)
train.drop(train[train["dropoff_longitude"] == 0].index, axis=0, inplace=True)
train.drop(train[train["dropoff_latitude"] == 0].index, axis=0, inplace=True)
train.drop(train[train["passenger_count"] > 5].index, axis=0, inplace=True)
train.drop(train[train["passenger_count"] == 0].index, axis=0, inplace=True)



## === cell 11
train.head()




## === cell 12
def add_time_features(df, dt_col="pickup_datetime"):
    dt = pd.to_datetime(df[dt_col], utc=True, errors="coerce")
    df["pickup_hour"] = dt.dt.hour.astype("float32")
    df["pickup_dayofweek"] = dt.dt.dayofweek.astype("float32")
    df["pickup_month"] = dt.dt.month.astype("float32")
    df["pickup_year"] = dt.dt.year.astype("float32")
    return df


def haversine_km(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371.0 * c


def bearing_deg(lon1, lat1, lon2, lat2):
    lon1r, lat1r, lon2r, lat2r = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2r - lon1r
    x = np.sin(dlon) * np.cos(lat2r)
    y = np.cos(lat1r) * np.sin(lat2r) - np.sin(lat1r) * np.cos(lat2r) * np.cos(dlon)
    brng = np.degrees(np.arctan2(x, y))
    return ((brng + 360.0) % 360.0).astype("float32")


MANHATTAN_LON = -73.985428
MANHATTAN_LAT = 40.748817
JFK_LON, JFK_LAT = -73.7781, 40.6413
LGA_LON, LGA_LAT = -73.8740, 40.7769
EWR_LON, EWR_LAT = -74.1745, 40.6895


def add_geo_features(df):
    df["haversine_km"] = haversine_km(
        df["pickup_longitude"].values,
        df["pickup_latitude"].values,
        df["dropoff_longitude"].values,
        df["dropoff_latitude"].values,
    ).astype("float32")

    df["mid_longitude"] = (
        (df["pickup_longitude"] + df["dropoff_longitude"]) / 2.0
    ).astype("float32")
    df["mid_latitude"] = (
        (df["pickup_latitude"] + df["dropoff_latitude"]) / 2.0
    ).astype("float32")
    df["abs_lon_diff"] = (
        (df["pickup_longitude"] - df["dropoff_longitude"]).abs().astype("float32")
    )
    df["abs_lat_diff"] = (
        (df["pickup_latitude"] - df["dropoff_latitude"]).abs().astype("float32")
    )

    df["manhattan_deg"] = (df["abs_lon_diff"] + df["abs_lat_diff"]).astype("float32")
    df["euclidean_deg"] = np.sqrt(
        df["abs_lon_diff"] ** 2 + df["abs_lat_diff"] ** 2
    ).astype("float32")

    df["bearing_deg"] = bearing_deg(
        df["pickup_longitude"].values,
        df["pickup_latitude"].values,
        df["dropoff_longitude"].values,
        df["dropoff_latitude"].values,
    )

    df["pickup_to_manhattan_km"] = haversine_km(
        df["pickup_longitude"].values,
        df["pickup_latitude"].values,
        MANHATTAN_LON,
        MANHATTAN_LAT,
    ).astype("float32")
    df["dropoff_to_manhattan_km"] = haversine_km(
        df["dropoff_longitude"].values,
        df["dropoff_latitude"].values,
        MANHATTAN_LON,
        MANHATTAN_LAT,
    ).astype("float32")

    df["pickup_to_jfk_km"] = haversine_km(
        df["pickup_longitude"].values, df["pickup_latitude"].values, JFK_LON, JFK_LAT
    ).astype("float32")
    df["dropoff_to_jfk_km"] = haversine_km(
        df["dropoff_longitude"].values, df["dropoff_latitude"].values, JFK_LON, JFK_LAT
    ).astype("float32")

    df["pickup_to_lga_km"] = haversine_km(
        df["pickup_longitude"].values, df["pickup_latitude"].values, LGA_LON, LGA_LAT
    ).astype("float32")
    df["dropoff_to_lga_km"] = haversine_km(
        df["dropoff_longitude"].values, df["dropoff_latitude"].values, LGA_LON, LGA_LAT
    ).astype("float32")

    df["pickup_to_ewr_km"] = haversine_km(
        df["pickup_longitude"].values, df["pickup_latitude"].values, EWR_LON, EWR_LAT
    ).astype("float32")
    df["dropoff_to_ewr_km"] = haversine_km(
        df["dropoff_longitude"].values, df["dropoff_latitude"].values, EWR_LON, EWR_LAT
    ).astype("float32")

    lat_mean_rad = np.radians(
        ((df["pickup_latitude"] + df["dropoff_latitude"]) / 2.0).astype("float32")
    )
    dlat_km = (df["abs_lat_diff"].astype("float32")) * 111.32
    dlon_km = (df["abs_lon_diff"].astype("float32")) * (
        111.32 * np.cos(lat_mean_rad).astype("float32")
    )
    df["euclidean_km_approx"] = np.sqrt(dlat_km * dlat_km + dlon_km * dlon_km).astype(
        "float32"
    )

    return df


def add_location_bins(df, precision=2):
    df["pickup_lon_bin"] = df["pickup_longitude"].round(precision).astype("float32")
    df["pickup_lat_bin"] = df["pickup_latitude"].round(precision).astype("float32")
    df["dropoff_lon_bin"] = df["dropoff_longitude"].round(precision).astype("float32")
    df["dropoff_lat_bin"] = df["dropoff_latitude"].round(precision).astype("float32")
    return df




## === cell 13
train = add_time_features(train, "pickup_datetime")



## === cell 14
train.drop(["key"], axis=1, inplace=True)



## === cell 15
train.drop(["pickup_datetime"], axis=1, inplace=True)



## === cell 16
train.dropna(inplace=True)

train.drop(
    train.index[
        (train.pickup_longitude < -75)
        | (train.pickup_longitude > -72)
        | (train.pickup_latitude < 40)
        | (train.pickup_latitude > 42)
    ],
    inplace=True,
)
train.drop(
    train.index[
        (train.dropoff_longitude < -75)
        | (train.dropoff_longitude > -72)
        | (train.dropoff_latitude < 40)
        | (train.dropoff_latitude > 42)
    ],
    inplace=True,
)



## === cell 17
train = add_geo_features(train)
train = add_location_bins(train, precision=2)

train = train[(train["haversine_km"] >= 0) & (train["haversine_km"] <= 100)].copy()



## === cell 18
eps_km = 0.05
train["fare_per_km"] = train["fare_amount"] / (train["haversine_km"] + eps_km)
train = train[(train["fare_per_km"] > 0) & (train["fare_per_km"] < 50)].copy()
train = train[~((train["haversine_km"] < 0.2) & (train["fare_amount"] > 50))].copy()
train.drop(columns=["fare_per_km"], inplace=True)



## === cell 19
train.head()



## === cell 20
X, y = train.drop("fare_amount", axis=1), train["fare_amount"]
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=12
)



## === cell 21
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)



## === cell 22
xgb_r = xgb.XGBRegressor(
    objective="reg:squarederror",
    n_estimators=600,
    learning_rate=0.05,
    max_depth=6,
    subsample=1.0,
    colsample_bytree=1.0,
    reg_lambda=1.0,
    seed=123,
    n_jobs=-1,
)



## === cell 23
y_train_log = np.log1p(y_train.values)
xgb_r.fit(X_train_scaled, y_train_log)



## === cell 24
y_pred_log = xgb_r.predict(X_valid_scaled)
y_pred = np.expm1(y_pred_log)
y_pred = np.clip(y_pred, 0.0, None)



## === cell 25
rmse = np.sqrt(MSE(y_valid, y_pred))
print("RMSE : % f" % (rmse))



## === cell 26
test_keys = test_raw[["key"]].copy()
test = test_raw.drop(["key"], axis=1)



## === cell 27
test.head()



## === cell 28
test = add_time_features(test.copy(), "pickup_datetime")
test.drop(["pickup_datetime"], axis=1, inplace=True)

test = add_geo_features(test)
test = add_location_bins(test, precision=2)

train_feature_medians = X_train.median(numeric_only=True)
for col in test.columns:
    if col in train_feature_medians.index:
        test[col] = test[col].fillna(train_feature_medians[col])
    else:
        test[col] = test[col].fillna(0.0)



## === cell 29
test.head()



## === cell 30
test = test.reindex(columns=X.columns, fill_value=0.0)



## === cell 31
test_scaled = scaler.transform(test)



## === cell 32
new_pred_log = xgb_r.predict(test_scaled)
new_pred = np.expm1(new_pred_log)
new_pred = np.clip(new_pred, 0.0, None)



## === cell 33
submission = test_keys.copy()
submission["fare_amount"] = new_pred



## === cell 34
submission.head()



## === cell 35
submission = submission[["key", "fare_amount"]]
submission.to_csv("submission1.csv", index=False)

print("Wrote submission1.csv with shape:", submission.shape)
print("Expected test rows:", test_raw.shape[0])
print(submission.head())
print("Any missing keys in submission:", submission["key"].isnull().any())
print("Any missing fares in submission:", submission["fare_amount"].isnull().any())
