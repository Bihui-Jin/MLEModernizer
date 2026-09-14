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

4.10469

# 6. Current score

6.81541

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 5.37712) has done: 'I fix the runtime errors caused by seaborn’s updated API by switching positional arguments in `regplot`/`lmplot` to explicit `x=`/`y=` keywords. I also fix a major modeling bug in the train/validation split where `fare_amount` was mistakenly included as an input feature, causing leakage and harming generalization; the model now train on `df` with `fare_amount` removed, while keeping the same feature set and XGBoost regressor approach. Finally, I make the run deterministic (`random_state`) and ensure the submission is written with a `.csv` suffix and correct columns (`key,fare_amount`). These changes are minimal, preserve the core approach, and should improve RMSE toward your target.'
- What this solution (achieved 5.44725) has done: 'I make two minimal changes aimed at lowering RMSE toward your 4.10469 target without changing the overall XGBoost-regression approach: (1) add a few simple, standard time-based features from `pickup_datetime` (hour/day/month/weekday) that are already present but currently unused, and (2) apply a basic, competition-common geographic outlier filter on lat/lon and distance to remove clearly invalid trips that hurt generalization. I keep the same model type and training loop, only adjusting the feature list and adding the same feature engineering to both train and test to preserve semantics. This should improve score materially from ~5.38 while remaining stable and within Kaggle constraints, and it still writes a valid `submission.csv` with `key,fare_amount`.'
- What this solution (achieved 6.88736) has done: 'Your current RMSE (5.44725) is worse than the target (4.10469), so we should make small, legitimate improvements without changing the core XGBoost regression approach. The biggest low-risk gain here is to add a couple of standard, competition-common geographic features (Manhattan distance and bearing) derived from the same coordinates you already use; this keeps the same model type/training loop but gives the model more signal. I also make the train/test feature engineering fully symmetric by applying the same geographic filters to test (with safe clipping rather than dropping rows) and by ensuring no NaNs/Infs reach XGBoost, which can otherwise degrade predictions. These changes are minimal, keep evaluation semantics intact, and still write a valid `submission.csv` with `key,fare_amount`.'
- What this solution (achieved 6.64383) has done: 'Your current RMSE (6.88736) is worse than the target (4.10469), so we should make small, legitimate improvements without changing the core XGBoost-regression approach. The biggest issue is that your train set uses geo-filtering (drops out-of-bounds rows), while your test set is only clipped after feature creation; this train/test mismatch can hurt generalization, so we make preprocessing symmetric by clipping coordinates before computing distance/bearing features for both train and test, and only then applying the same haversine-range filter on train. We also include the already-computed `haversine` and `euclidean` consistently by recomputing them after clipping (instead of mixing unclipped/clipped), and keep the same model type/training loop and submission format. These are minimal, metric-aligned fixes aimed at lowering RMSE toward your target without altering the overall solution structure.'
- What this solution (achieved 5.57054) has done: 'We need to move RMSE down from 6.64383 toward 4.10469 (lower is better), so we should make a small, legitimate improvement without changing the overall XGBoost-regression approach. The biggest low-risk issue is that we currently feed raw lat/lon (and the raw parsed datetime column) into `X_train` but not into the model’s `features` list, while also not using a few very standard location/temporal features that usually help a lot for this competition. I keep the same model class and training flow, but (1) add a minimal NYC-centric feature set (center distance + simple direction components) computed from existing columns, and (2) slightly tighten the fare/passenger/outlier filtering to remove obviously noisy training rows that inflate RMSE. Submission writing stays identical (`submission.csv` with `key,fare_amount`), and feature engineering is applied symmetrically to train and test.'
- What this solution (achieved 5.17321) has done: 'Your current RMSE (5.57054) is worse than the target (4.10469), so we should make small, legitimate improvements without changing the XGBoost regressor approach. The lowest-risk gain here is to fix a feature bug in `minkowski_distance` (it incorrectly computes `abs(y2 - y1)` without exponentiating), which corrupts `euclidean` and anything derived from it; this is a direct modeling-quality fix while preserving the same feature set and flow. Next, we add two standard airport-distance features (to JFK and LGA) using the same haversine function you already use; these are minimal, competition-common features that usually reduce RMSE materially. Finally, we keep everything deterministic and ensure the submission format remains `key,fare_amount` written to `submission.csv`.'
- What this solution (achieved 5.69654) has done: 'To move RMSE down from 5.17321 toward your 4.10469 target (lower is better) without changing the core XGBoost-regression approach, I make three minimal, high-impact fixes: (1) remove the hard fare cap at 60 that biases the target distribution and hurts generalization, replacing it with a standard, looser upper bound; (2) tighten the geographic/outlier cleaning with a small set of common NYC constraints (bounding box + passenger_count + reasonable fare upper bound) while keeping the same feature engineering and training loop; and (3) set a small, deterministic set of XGBoost hyperparameters (same model class, same fit/predict flow) that typically improves RMSE for this competition without adding training complexity. The submission writing remains unchanged (`submission.csv` with `key,fare_amount`). These are incremental changes expected to reduce RMSE toward the target band while preserving your existing logic and semantics.'
- What this solution (achieved 5.74909) has done: 'Your RMSE (5.69654) is worse than the target (4.10469), so we should make a small, legitimate improvement without changing the overall XGBoost-regression approach. The biggest low-risk issue is that the model can predict unrealistically low fares for short trips; since you already clip at 0, tightening this to the known competition minimum (about $2.50) is a simple post-processing step aligned with the target distribution and often reduces RMSE. Second, your current coordinate “clipping” can distort distances for out-of-bounds points (creating artificial short/long trips); switching to dropping out-of-bounds rows in train (already done via `geo_mask`) and *not* clipping test (instead, just compute features on raw values and then safely fill/clip derived distances) reduces feature distortion while keeping the same feature set and training loop. These changes preserve core logic (same features, same model, same training flow) and should move RMSE downward toward the target.'
- What this solution (achieved 6.81541) has done: 'Your current RMSE (5.74909) is worse than the target (4.10469), so we should make the smallest model-quality improvements that keep your XGBoost approach and feature set intact. The biggest low-risk issue is train/test preprocessing mismatch: you drop geo-outliers in train but only clip a single derived feature in test; we instead apply the same geo bounding-box filter to test by clipping coordinates *before* computing distances, mirroring what the model learned. Next, we tighten the haversine-based training filter to remove extreme-distance outliers that inflate RMSE while keeping the same features and model. Finally, we ensure the model sees identical feature distributions by recomputing all geo primitives after clipping for both train and test, then writing a valid `submission.csv` as before.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
df = pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv",
    nrows=500_000,
    parse_dates=["pickup_datetime"],
).drop(columns="key")

df = df.dropna()
df.head()



## === cell 2
df.describe()



## === cell 3
plt.figure(figsize=(10, 8))
plt.hist(df["fare_amount"])
plt.title("Fare Distribution")



## === cell 4
print(f"Number of negative fares: {len(df[df['fare_amount'] < 0])}")
print(f"Number of fares equal to 0: {len(df[df['fare_amount'] == 0])}")



## === cell 5
df = df[df["fare_amount"].between(left=2.5, right=250.0)]




## === cell 6
def ecdf(x):
    x = np.sort(x)
    n = len(x)
    y = np.arange(1, n + 1, 1) / n
    return x, y




## === cell 7
x, y = ecdf(df["fare_amount"])
plt.figure(figsize=(8, 6))
plt.plot(x, y)
plt.ylabel("Percentile")
plt.xlabel("Fare Amount")
plt.title("Fare Amount ECDF")



## === cell 8
x, y = ecdf(df["fare_amount"])
plt.figure(figsize=(8, 6))
plt.plot(x, y)
plt.ylabel("Percentile")
plt.xlabel("Fare Amount")
plt.title("Fare Amount ECDF (after cleaning)")



## === cell 9
df["passenger_count"].value_counts().plot.bar()
plt.title("Passenger Counts")
plt.xlabel("Passengers Numbers")
plt.ylabel("Frequency")



## === cell 10
df = df.loc[(df["passenger_count"] > 0) & (df["passenger_count"] < 7)]



## === cell 11
fig, axes = plt.subplots(1, 2, figsize=(20, 8), sharex=True, sharey=True)
axes = axes.flatten()

sns.regplot(
    x="pickup_longitude", y="pickup_latitude", fit_reg=False, data=df, ax=axes[0]
)
sns.regplot(
    x="dropoff_longitude", y="dropoff_latitude", fit_reg=False, data=df, ax=axes[1]
)
axes[0].set_title("Pickup Locations")
axes[1].set_title("Dropoff Locations")



## === cell 12
df["abs_lat_diff"] = (df["dropoff_latitude"] - df["pickup_latitude"]).abs()
df["abs_lon_diff"] = (df["dropoff_longitude"] - df["pickup_longitude"]).abs()



## === cell 13
sns.lmplot(x="abs_lat_diff", y="abs_lon_diff", fit_reg=False, data=df)
plt.title("Absolute latitude difference vs Absolute longitude difference")



## === cell 14
zero_diff = df[(df["abs_lat_diff"] == 0) & (df["abs_lon_diff"] == 0)]
zero_diff.shape




## === cell 15
def minkowski_distance(x1, x2, y1, y2, p):
    return ((abs(x2 - x1) ** p) + (abs(y2 - y1) ** p)) ** (1 / p)




## === cell 16
df["euclidean"] = minkowski_distance(
    df["pickup_longitude"],
    df["dropoff_longitude"],
    df["pickup_latitude"],
    df["dropoff_latitude"],
    2,
)



## === cell 17
plt.figure(figsize=(10, 8))
plt.hist(df["euclidean"])
plt.title("Euclidean Distance Distribution")
ax = plt.subplot(111)
ax.set_xlim([0, 500])



## === cell 18
plt.figure(figsize=(10, 6))
for p, grouped in df.groupby("passenger_count"):
    sns.kdeplot(grouped["fare_amount"], label=f"{p} passengers")

plt.xlabel("Fare Amount")
plt.ylabel("Density")
plt.title("Distribution of Fare Amount by Number of Passengers")



## === cell 19
df.groupby("passenger_count")["fare_amount"].agg(["mean", "count"])



## === cell 20
df.groupby("passenger_count")["fare_amount"].mean().plot.bar(color="b")
plt.title("Average Fare by Passenger Count")



## === cell 21
R = 6378


def haversine_np(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points on the earth (specified in decimal degrees)
    All args must be of equal length.
    source: https://stackoverflow.com/a/29546836
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    km = R * c
    return km




## === cell 22
df["haversine"] = haversine_np(
    df["pickup_longitude"],
    df["pickup_latitude"],
    df["dropoff_longitude"],
    df["dropoff_latitude"],
)



## === cell 23
sns.kdeplot(df["haversine"])



## === cell 24
corrs = df.corr(numeric_only=True)
corrs["fare_amount"].plot.bar(color="b")
plt.title("Correlation with Fare Amount")




## === cell 25
def bearing_np(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    brng = np.degrees(np.arctan2(y, x))
    return (brng + 360.0) % 360.0


def add_time_features(frame: pd.DataFrame) -> pd.DataFrame:
    dt = frame["pickup_datetime"]
    frame["pickup_hour"] = dt.dt.hour.astype(np.int16)
    frame["pickup_day"] = dt.dt.day.astype(np.int16)
    frame["pickup_month"] = dt.dt.month.astype(np.int16)
    frame["pickup_weekday"] = dt.dt.weekday.astype(np.int16)
    return frame


def clip_geo_inplace(frame: pd.DataFrame) -> pd.DataFrame:
    frame["pickup_longitude"] = frame["pickup_longitude"].clip(-75, -72)
    frame["dropoff_longitude"] = frame["dropoff_longitude"].clip(-75, -72)
    frame["pickup_latitude"] = frame["pickup_latitude"].clip(40, 42)
    frame["dropoff_latitude"] = frame["dropoff_latitude"].clip(40, 42)
    return frame


def recompute_geo_primitives(frame: pd.DataFrame) -> pd.DataFrame:
    frame["abs_lat_diff"] = (frame["dropoff_latitude"] - frame["pickup_latitude"]).abs()
    frame["abs_lon_diff"] = (
        frame["dropoff_longitude"] - frame["pickup_longitude"]
    ).abs()
    frame["euclidean"] = minkowski_distance(
        frame["pickup_longitude"],
        frame["dropoff_longitude"],
        frame["pickup_latitude"],
        frame["dropoff_latitude"],
        2,
    )
    frame["haversine"] = haversine_np(
        frame["pickup_longitude"],
        frame["pickup_latitude"],
        frame["dropoff_longitude"],
        frame["dropoff_latitude"],
    )
    return frame


def apply_haversine_filter_train(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[frame["haversine"].between(0.05, 60.0)]


def add_geo_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame["manhattan"] = frame["abs_lat_diff"] + frame["abs_lon_diff"]
    frame["bearing"] = bearing_np(
        frame["pickup_longitude"],
        frame["pickup_latitude"],
        frame["dropoff_longitude"],
        frame["dropoff_latitude"],
    ).astype(np.float32)
    return frame


NYC_LON, NYC_LAT = -73.985428, 40.748817  # Midtown Manhattan (simple fixed reference)

JFK_LON, JFK_LAT = -73.778137, 40.641312
LGA_LON, LGA_LAT = -73.8740, 40.7769


def add_center_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame["pickup_dist_center"] = haversine_np(
        frame["pickup_longitude"], frame["pickup_latitude"], NYC_LON, NYC_LAT
    ).astype(np.float32)
    frame["dropoff_dist_center"] = haversine_np(
        frame["dropoff_longitude"], frame["dropoff_latitude"], NYC_LON, NYC_LAT
    ).astype(np.float32)
    frame["lon_diff"] = (frame["dropoff_longitude"] - frame["pickup_longitude"]).astype(
        np.float32
    )
    frame["lat_diff"] = (frame["dropoff_latitude"] - frame["pickup_latitude"]).astype(
        np.float32
    )

    frame["pickup_dist_jfk"] = haversine_np(
        frame["pickup_longitude"], frame["pickup_latitude"], JFK_LON, JFK_LAT
    ).astype(np.float32)
    frame["dropoff_dist_jfk"] = haversine_np(
        frame["dropoff_longitude"], frame["dropoff_latitude"], JFK_LON, JFK_LAT
    ).astype(np.float32)
    frame["pickup_dist_lga"] = haversine_np(
        frame["pickup_longitude"], frame["pickup_latitude"], LGA_LON, LGA_LAT
    ).astype(np.float32)
    frame["dropoff_dist_lga"] = haversine_np(
        frame["dropoff_longitude"], frame["dropoff_latitude"], LGA_LON, LGA_LAT
    ).astype(np.float32)

    return frame


geo_mask = (
    df["pickup_longitude"].between(-75, -72)
    & df["dropoff_longitude"].between(-75, -72)
    & df["pickup_latitude"].between(40, 42)
    & df["dropoff_latitude"].between(40, 42)
)
df = df.loc[geo_mask].copy()

df = recompute_geo_primitives(df)
df = add_time_features(df)
df = add_geo_features(df)
df = add_center_features(df)
df = apply_haversine_filter_train(df)

df = df.replace([np.inf, -np.inf], np.nan).dropna()



## === cell 26
test = pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv",
    parse_dates=["pickup_datetime"],
)

test_id = list(test.pop("key"))

test = clip_geo_inplace(test)

test = recompute_geo_primitives(test)
test = add_time_features(test)
test = add_geo_features(test)
test = add_center_features(test)

test["haversine"] = test["haversine"].clip(0.05, 60.0)

test = test.replace([np.inf, -np.inf], np.nan).fillna(0)

test.describe()



## === cell 27
from sklearn.model_selection import train_test_split

y = df["fare_amount"].to_numpy()
X = df.drop(columns=["fare_amount", "pickup_datetime"])

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.30, random_state=42
)



## === cell 28
import xgboost as xgb

features = [
    "haversine",
    "manhattan",
    "bearing",
    "abs_lat_diff",
    "abs_lon_diff",
    "lon_diff",
    "lat_diff",
    "pickup_dist_center",
    "dropoff_dist_center",
    "pickup_dist_jfk",
    "dropoff_dist_jfk",
    "pickup_dist_lga",
    "dropoff_dist_lga",
    "passenger_count",
    "pickup_hour",
    "pickup_weekday",
    "pickup_month",
]

xgbr = xgb.XGBRegressor(
    random_state=42,
    n_estimators=600,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.0,
    reg_lambda=1.0,
    objective="reg:squarederror",
    n_jobs=-1,
)
xgbr.fit(X_train[features], y_train)



## === cell 29
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


def metrics(train_pred, valid_pred, y_train, y_valid):
    """Calculate metrics: Root mean squared error and mean absolute percentage error"""
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    valid_rmse = np.sqrt(mean_squared_error(y_valid, valid_pred))

    train_ape = abs((y_train - train_pred) / y_train)
    valid_ape = abs((y_valid - valid_pred) / y_valid)

    train_ape[train_ape == np.inf] = 0
    train_ape[train_ape == -np.inf] = 0
    valid_ape[valid_ape == np.inf] = 0
    valid_ape[valid_ape == -np.inf] = 0

    train_mape = 100 * np.mean(train_ape)
    valid_mape = 100 * np.mean(valid_ape)

    return train_rmse, valid_rmse, train_mape, valid_mape


def evaluate(model, features, X_train, X_valid, y_train, y_valid):
    """Print RMSE and MAPE on train/validation splits"""
    train_pred = model.predict(X_train[features])
    valid_pred = model.predict(X_valid[features])

    train_rmse, valid_rmse, train_mape, valid_mape = metrics(
        train_pred, valid_pred, y_train, y_valid
    )

    print(f"Training:   rmse = {round(train_rmse, 2)} \t mape = {round(train_mape, 2)}")
    print(f"Validation: rmse = {round(valid_rmse, 2)} \t mape = {round(valid_mape, 2)}")




## === cell 30
evaluate(
    xgbr,
    features,
    X_train,
    X_valid,
    y_train,
    y_valid,
)



## === cell 31
test_model = test.drop(columns=["pickup_datetime"])
preds = xgbr.predict(test_model[features])

preds = np.clip(preds, 2.5, None)

sub = pd.DataFrame({"key": test_id, "fare_amount": preds})
sub.to_csv("submission.csv", index=False)

sns.histplot(sub["fare_amount"], kde=True)
plt.title("Distribution of Predicted Fare Amount")
