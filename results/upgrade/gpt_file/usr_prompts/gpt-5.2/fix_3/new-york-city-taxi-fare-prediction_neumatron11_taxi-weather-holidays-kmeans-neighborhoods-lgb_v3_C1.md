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
scipy==1.15.3
sklearn-pandas==2.2.0
tqdm==4.67.1

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

3.3504463754491907

# 6. Current score

4.3716

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 4.31827) has done: 'I remove the hard dependencies on external “nyc-weather”, “us-bank-holidays”, and pre-trained kmeans pickle inputs (none exist in your environment) by generating safe fallback feature columns so downstream code keeps working with the same overall modeling pipeline. I also fix a few logic/typing bugs that would hurt both runtime and score (incorrect `drop(..., axis=0)`, incorrect SNOW/TMIN assignments, and a missing underscore in `day_hour` for test). Finally, I update the LightGBM training call to be compatible with lightgbm==4.6.0 (no `early_stop/verbose_eval` params; use callbacks), and replace deprecated `np.asscalar` with a safe float conversion so a valid `submission_lgb.csv` is always written.'
- What this solution (achieved 4.3716) has done: 'Your current score (4.31827 RMSE) is worse than the target (3.35045), so we should improve predictive quality with minimal risk while preserving the same overall pipeline. The biggest avoidable score hit in your code is a train/test mismatch: `hot_day`/`cold_day` thresholds differ between train and test, which shifts feature meaning at inference and degrades RMSE; we make them identical. We also remove early stopping (it’s currently effectively truncating training because `stopping_rounds == num_boost_round`), so the model consistently trains the full intended 100 rounds without changing the model type or loss. Finally, we keep predictions non-negative but also clip extreme outliers to a reasonable upper bound to reduce the RMSE impact of rare huge predictions (a standard, minimal post-processing step for this competition).'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import pickle
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from tqdm import tqdm
from scipy.sparse import csr_matrix, hstack
from sklearn.impute import SimpleImputer
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder

for p in [
    "../input",
    "/kaggle/input",
    "/kaggle/data",
    "/kaggle/data/new-york-city-taxi-fare-prediction",
]:
    if os.path.exists(p):
        print(f"Listing {p}:")
        print(os.listdir(p)[:50])



## === cell 1
weather_path_candidates = [
    "../input/nyc-weather/nyc_weather.csv",
    "/kaggle/input/nyc-weather/nyc_weather.csv",
]
nyc_weather = None
for wp in weather_path_candidates:
    if os.path.exists(wp):
        nyc_weather = pd.read_csv(wp)
        break

weather_cols = ["DATE", "AWND", "PRCP", "SNOW", "TMAX", "TMIN"]
if nyc_weather is None:
    nyc_weather = pd.DataFrame(columns=weather_cols)
else:
    nyc_weather = nyc_weather[weather_cols].copy()
    nyc_weather["DATE"] = pd.to_datetime(
        nyc_weather["DATE"], utc=True, format="%m/%d/%Y", errors="coerce"
    )

nyc_weather.head()



## === cell 2
if len(nyc_weather) > 0 and {"TMAX", "TMIN"}.issubset(nyc_weather.columns):
    plt.rc("figure", figsize=(15, 8))
    plt.subplot(1, 2, 1)
    plt.hist(nyc_weather.TMAX.dropna(), bins=30)
    plt.xlabel("Temperature (C)")
    plt.ylabel("Frequency Count")
    plt.title("Max Daily Temperature")
    plt.subplot(1, 2, 2)
    plt.hist(nyc_weather.TMIN.dropna(), bins=30)
    plt.xlabel("Temperature (C)")
    plt.ylabel("Frequency Count")
    plt.title("Min Daily Temperature")
    plt.show()
else:
    print("nyc_weather not available in this environment; skipping weather plots.")



## === cell 3
if len(nyc_weather) > 0:
    display(nyc_weather.describe())
else:
    print("nyc_weather not available in this environment; skipping describe().")



## === cell 4
train_path_candidates = [
    "../input/new-york-city-taxi-fare-prediction/train.csv",
    "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv",
    "/kaggle/data/train.csv",
    "/kaggle/data/new-york-city-taxi-fare-prediction/train.csv",
]
test_path_candidates = [
    "../input/new-york-city-taxi-fare-prediction/test.csv",
    "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv",
    "/kaggle/data/test.csv",
    "/kaggle/data/new-york-city-taxi-fare-prediction/test.csv",
]

train_path = next((p for p in train_path_candidates if os.path.exists(p)), None)
test_path = next((p for p in test_path_candidates if os.path.exists(p)), None)
if train_path is None or test_path is None:
    raise FileNotFoundError(
        "Could not locate train.csv/test.csv in the provided environment paths."
    )

train = pd.read_csv(train_path, nrows=15_000_000)
test = pd.read_csv(test_path)
test_id = test.key.values  # set this value for final submission
print(train.info())



## === cell 5
train["pickup_datetime"] = train["pickup_datetime"].str.slice(0, 16)
test["pickup_datetime"] = test["pickup_datetime"].str.slice(0, 16)

train["pickup_datetime"] = pd.to_datetime(
    train["pickup_datetime"], utc=True, format="%Y-%m-%d %H:%M", errors="coerce"
)
test["pickup_datetime"] = pd.to_datetime(
    test["pickup_datetime"], utc=True, format="%Y-%m-%d %H:%M", errors="coerce"
)

train_sample = train.dropna().copy()
del train

train_sample.drop(labels="key", axis=1, inplace=True)
test.drop(labels="key", axis=1, inplace=True)

train_sample.loc[:, "passenger_count"] = train_sample.passenger_count.astype(
    dtype="uint8"
)
train_sample["pickup_longitude"] = train_sample.pickup_longitude.astype(dtype="float32")
train_sample["pickup_latitude"] = train_sample.pickup_latitude.astype(dtype="float32")
train_sample["dropoff_longitude"] = train_sample.dropoff_longitude.astype(
    dtype="float32"
)
train_sample["dropoff_latitude"] = train_sample.dropoff_latitude.astype(dtype="float32")
train_sample["fare_amount"] = train_sample.fare_amount.astype(dtype="float32")

test["pickup_longitude"] = test.pickup_longitude.astype(dtype="float32")
test["pickup_latitude"] = test.pickup_latitude.astype(dtype="float32")
test["dropoff_longitude"] = test.dropoff_longitude.astype(dtype="float32")
test["dropoff_latitude"] = test.dropoff_latitude.astype(dtype="float32")

train_sample = train_sample.loc[
    train_sample.pickup_longitude.between(
        test.pickup_longitude.min(), test.pickup_longitude.max()
    )
]
train_sample = train_sample.loc[
    train_sample.pickup_latitude.between(
        test.pickup_latitude.min(), test.pickup_latitude.max()
    )
]
train_sample = train_sample.loc[
    train_sample.dropoff_longitude.between(
        test.dropoff_longitude.min(), test.dropoff_longitude.max()
    )
]
train_sample = train_sample.loc[
    train_sample.dropoff_latitude.between(
        test.dropoff_latitude.min(), test.dropoff_latitude.max()
    )
]

train_sample["hour"] = train_sample["pickup_datetime"].dt.hour
train_sample["month"] = train_sample["pickup_datetime"].dt.month
train_sample["day_of_week"] = train_sample["pickup_datetime"].dt.dayofweek
train_sample["year"] = train_sample["pickup_datetime"].dt.year

test["hour"] = test["pickup_datetime"].dt.hour
test["month"] = test["pickup_datetime"].dt.month
test["day_of_week"] = test["pickup_datetime"].dt.dayofweek
test["year"] = test["pickup_datetime"].dt.year

train_sample["hour"] = train_sample.hour.astype(dtype="uint8")
train_sample["month"] = train_sample.month.astype(dtype="uint8")
train_sample["day_of_week"] = train_sample.day_of_week.astype(dtype="uint8")
train_sample["year"] = train_sample.year.astype(dtype="uint16")

test["hour"] = test.hour.astype(dtype="uint8")
test["month"] = test.month.astype(dtype="uint8")
test["day_of_week"] = test.day_of_week.astype(dtype="uint8")
test["year"] = test.year.astype(dtype="uint16")



## === cell 6
train_sample["pickup_day"] = train_sample.pickup_datetime.dt.floor("d")
test["pickup_day"] = test.pickup_datetime.dt.floor("d")

if len(nyc_weather) > 0:
    train_sample = train_sample.merge(
        nyc_weather, how="left", left_on="pickup_day", right_on="DATE"
    )
    test = test.merge(nyc_weather, how="left", left_on="pickup_day", right_on="DATE")
    train_sample.drop(columns=["pickup_day", "DATE"], inplace=True)
    test.drop(columns=["pickup_day", "DATE"], inplace=True)
else:
    for c in ["AWND", "PRCP", "SNOW", "TMAX", "TMIN"]:
        train_sample[c] = np.nan
        test[c] = np.nan
    train_sample.drop(columns=["pickup_day"], inplace=True)
    test.drop(columns=["pickup_day"], inplace=True)

for c in ["AWND", "PRCP", "SNOW", "TMAX", "TMIN"]:
    train_sample[c] = train_sample[c].astype("float16")
    test[c] = test[c].astype("float16")

HOT_TMAX_C = 30.0
COLD_TMIN_C = 0.0
train_sample["hot_day"] = np.where(train_sample.TMAX >= HOT_TMAX_C, 1, 0)
train_sample["cold_day"] = np.where(train_sample.TMIN <= COLD_TMIN_C, 1, 0)
test["hot_day"] = np.where(test.TMAX >= HOT_TMAX_C, 1, 0)
test["cold_day"] = np.where(test.TMIN <= COLD_TMIN_C, 1, 0)

train_sample["hot_day"] = train_sample.hot_day.astype(dtype="uint8")
train_sample["cold_day"] = train_sample.cold_day.astype(dtype="uint8")
test["hot_day"] = test.hot_day.astype(dtype="uint8")
test["cold_day"] = test.cold_day.astype(dtype="uint8")


def degree_to_radion(degree):
    return degree * (np.pi / 180)


def calculate_distance(
    pickup_latitude, pickup_longitude, dropoff_latitude, dropoff_longitude
):
    from_lat = degree_to_radion(pickup_latitude)
    from_long = degree_to_radion(pickup_longitude)
    to_lat = degree_to_radion(dropoff_latitude)
    to_long = degree_to_radion(dropoff_longitude)

    radius = 6371.01
    lat_diff = to_lat - from_lat
    long_diff = to_long - from_long

    a = (
        np.sin(lat_diff / 2) ** 2
        + np.cos(from_lat) * np.cos(to_lat) * np.sin(long_diff / 2) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return radius * c


train_sample["distance"] = calculate_distance(
    train_sample.pickup_latitude,
    train_sample.pickup_longitude,
    train_sample.dropoff_latitude,
    train_sample.dropoff_longitude,
)
test["distance"] = calculate_distance(
    test.pickup_latitude,
    test.pickup_longitude,
    test.dropoff_latitude,
    test.dropoff_longitude,
)

train_sample["distance"] = train_sample.distance.astype(dtype="float32")
test["distance"] = test.distance.astype(dtype="float32")

train_sample["day_hour"] = (
    train_sample.day_of_week.astype(str) + "_" + train_sample.hour.astype(str)
)
train_sample["day_hour"] = train_sample["day_hour"].astype("category")

test["day_hour"] = test.day_of_week.astype(str) + "_" + test.hour.astype(str)
test["day_hour"] = test["day_hour"].astype("category")

train_sample = train_sample[train_sample.fare_amount > 0]
print(train_sample.info())



## === cell 7
holiday_path_candidates = [
    "../input/us-bank-holidays-20092018/US Bank Holidays 2012-2018.csv",
    "/kaggle/input/us-bank-holidays-20092018/US Bank Holidays 2012-2018.csv",
]
holidays_path = next((p for p in holiday_path_candidates if os.path.exists(p)), None)

if holidays_path is not None:
    holidays = pd.read_csv(holidays_path)
    holidays["Date"] = pd.to_datetime(
        holidays["Date"], utc=True, format="%m/%d/%y", errors="coerce"
    )

    train_sample["pickup_day"] = train_sample.pickup_datetime.dt.floor("d")
    train_sample = train_sample.merge(
        holidays, left_on="pickup_day", right_on="Date", how="left"
    )
    train_sample["Holiday"] = train_sample.Holiday.fillna("None")

    le = LabelEncoder()
    train_sample["holiday"] = le.fit_transform(train_sample.Holiday.values)
    train_sample.drop(["Holiday", "Date", "pickup_day"], axis=1, inplace=True)

    test["pickup_day"] = test.pickup_datetime.dt.floor("d")
    test = test.merge(holidays, left_on="pickup_day", right_on="Date", how="left")
    test["Holiday"] = test.Holiday.fillna("None")
    test["holiday"] = (
        le.transform(test.Holiday.values)
        if "None" in le.classes_
        else le.fit_transform(test.Holiday.values)
    )
    test.drop(["Holiday", "Date", "pickup_day"], axis=1, inplace=True)
else:
    train_sample["holiday"] = 0
    test["holiday"] = 0

train_sample["holiday"] = train_sample.holiday.astype(dtype="uint8")
test["holiday"] = test.holiday.astype(dtype="uint8")
print(train_sample.info())



## === cell 8
train_sample.head()



## === cell 9
full_pickups = pd.concat(
    [
        train_sample[["pickup_longitude", "pickup_latitude"]],
        test[["pickup_longitude", "pickup_latitude"]],
    ],
    axis=0,
)
full_pickups.columns = ["x", "y"]
full_dropoffs = pd.concat(
    [
        train_sample[["dropoff_longitude", "dropoff_latitude"]],
        test[["dropoff_longitude", "dropoff_latitude"]],
    ],
    axis=0,
)
full_dropoffs.columns = ["x", "y"]
full_locs = pd.concat([full_pickups, full_dropoffs], axis=0)

full_locs["x"] = full_locs.x.round(4)
full_locs["y"] = full_locs.y.round(4)

full_locs = full_locs.groupby(["x", "y"]).count().reset_index()
full_locs.info()



## === cell 10
X_df = full_locs.copy()
X_kmeans = full_locs[["x", "y"]].values

num_clusters = 200
kmeans_path_candidates = [
    "../input/taxi-weather-holidays-kmeans-neighborhoods/kmeans_200_round4.pkl",
    "/kaggle/input/taxi-weather-holidays-kmeans-neighborhoods/kmeans_200_round4.pkl",
]
kmeans_path = next((p for p in kmeans_path_candidates if os.path.exists(p)), None)

if kmeans_path is not None:
    with open(kmeans_path, "rb") as fid:
        kmeans = pickle.load(fid)
else:
    kmeans = KMeans(n_clusters=num_clusters, random_state=17, n_init=10)
    kmeans.fit(X_kmeans)



## === cell 11
z = kmeans.predict(X_kmeans)



## === cell 12
centers = kmeans.cluster_centers_

x_centers = [pair[0] for pair in centers]
y_centers = [pair[1] for pair in centers]
z_centers = np.arange(num_clusters)

plt.subplot(1, 2, 1)
plt.scatter(X_df["x"], X_df["y"], c=z, s=5)
plt.gray()
plt.xlabel("Pickup/Dropoff Longitude")
plt.ylabel("Pickup/Dropoff Latitude")
plt.title("Clusters of NYC locations")
plt.subplot(1, 2, 2)
plt.scatter(x_centers, y_centers, c=z_centers, s=20)
plt.gray()
plt.xlabel("Pickup/Dropoff Longitude")
plt.ylabel("Pickup/Dropoff Latitude")
plt.title("Cluster Centers of NYC locations")

plt.show()



## === cell 13
del X_kmeans, X_df, full_pickups, full_dropoffs

train_sample["pickup_neighborhood"] = kmeans.predict(
    np.column_stack(
        [train_sample.pickup_longitude.values, train_sample.pickup_latitude.values]
    )
)
train_sample["dropoff_neighborhood"] = kmeans.predict(
    np.column_stack(
        [train_sample.dropoff_longitude.values, train_sample.dropoff_latitude.values]
    )
)

test["pickup_neighborhood"] = kmeans.predict(
    np.column_stack([test.pickup_longitude.values, test.pickup_latitude.values])
)
test["dropoff_neighborhood"] = kmeans.predict(
    np.column_stack([test.dropoff_longitude.values, test.dropoff_latitude.values])
)

train_sample["pickup_neighborhood"] = train_sample.pickup_neighborhood.astype(
    dtype="uint8"
)
train_sample["dropoff_neighborhood"] = train_sample.dropoff_neighborhood.astype(
    dtype="uint8"
)
test["pickup_neighborhood"] = test.pickup_neighborhood.astype(dtype="uint8")
test["dropoff_neighborhood"] = test.dropoff_neighborhood.astype(dtype="uint8")

print(train_sample.info())



## === cell 14
try:
    with open("kmeans_200_round4_v2.pkl", "wb") as fid:
        pickle.dump(kmeans, fid)
except Exception as e:
    print(f"Could not save kmeans pickle: {e}")



## === cell 15
categorical_cols = [
    "day_hour",
    "month",
    "year",
    "pickup_neighborhood",
    "dropoff_neighborhood",
    "passenger_count",
    "hot_day",
    "cold_day",
    "holiday",
]
numerical_cols = ["distance", "AWND", "PRCP", "SNOW"]

for c in categorical_cols + numerical_cols:
    if c not in train_sample.columns:
        train_sample[c] = np.nan
    if c not in test.columns:
        test[c] = np.nan

X_cats = train_sample[categorical_cols].values
X_cats_test = test[categorical_cols].values
X_cats_full = np.append(X_cats, X_cats_test, axis=0)

ohe = OneHotEncoder(categories="auto", handle_unknown="ignore")
X_onehot = ohe.fit_transform(X_cats_full)
del X_cats, X_cats_test, X_cats_full

X_nums = train_sample[numerical_cols].values
X_nums_test = test[numerical_cols].values
X_nums_full = np.append(X_nums, X_nums_test, axis=0)
X_nums_sparse = csr_matrix(X_nums_full)
del X_nums, X_nums_test, X_nums_full

X_full = hstack([X_onehot, X_nums_sparse]).tocsr()

si = SimpleImputer()
X_full_imputed = si.fit_transform(X_full)

X = X_full_imputed[: train_sample.shape[0], :]
X_public = X_full_imputed[train_sample.shape[0] :, :]

y = train_sample.fare_amount.values
del X_onehot, X_nums_sparse, X_full, X_full_imputed

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=17
)



## === cell 16
params = {
    "objective": "regression",
    "boosting": "gbdt",
    "metric": "rmse",
    "num_leaves": 50,
    "max_depth": 8,
    "learning_rate": 0.5,
    "bagging_fraction": 0.8,
    "feature_fraction": 0.8,
    "min_split_gain": 0.02,
    "min_child_samples": 10,
    "min_child_weight": 0.02,
    "lambda_l2": 0.0475,
    "verbosity": -1,
    "data_random_seed": 17,
}

d_train = lgb.Dataset(X_train, label=y_train)
d_test = lgb.Dataset(X_test, label=y_test)
watchlist = [d_train, d_test]

num_rounds = 100
verbose_eval = 100

model_lgb = lgb.train(
    params,
    train_set=d_train,
    num_boost_round=num_rounds,
    valid_sets=watchlist,
    valid_names=["train", "valid"],
    callbacks=[
        lgb.log_evaluation(period=verbose_eval),
    ],
)

pred_test_y_lgb = model_lgb.predict(X_test, num_iteration=num_rounds)
print("LGB Loss = " + str(sqrt(mean_squared_error(y_test, pred_test_y_lgb))))



## === cell 17
lgb_public = model_lgb.predict(X_public, num_iteration=num_rounds)
final_pred_public = lgb_public.flatten()

test_predictions_lgb = np.asarray(final_pred_public, dtype=np.float32)

test_predictions_lgb = np.clip(test_predictions_lgb, 0.0, 250.0)

sample = pd.DataFrame({"key": test_id, "fare_amount": test_predictions_lgb})
sample = sample.reindex(["key", "fare_amount"], axis=1)

sample.to_csv("submission_lgb.csv", index=False)
print(sample.head())
print("Wrote submission_lgb.csv with shape:", sample.shape)



## === cell 18
plt.rc("figure", figsize=(10, 10))
plt.hist(sample["fare_amount"].values, bins=100)
plt.xlabel("Predicticted Price")
plt.ylabel("Frequency")
plt.title("Predictions from LGB")
plt.show()
