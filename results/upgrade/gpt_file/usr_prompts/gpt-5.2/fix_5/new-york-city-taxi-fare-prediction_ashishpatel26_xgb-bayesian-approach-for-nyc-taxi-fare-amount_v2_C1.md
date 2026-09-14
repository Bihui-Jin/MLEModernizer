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

4.22957

# 6. Current score

None

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 5.77234) has done: 'I fix the runtime errors caused by deprecated pandas datetime accessors (`.dt.week`, `.dt.weekofyear`) by switching to the supported `isocalendar()` equivalents. I replace the removed `geopy.distance.VincentyDistance` with `geopy.distance.geodesic` while keeping the same “distance in km” feature and ensure it’s computed efficiently enough for 1M rows. I also make sure LightGBM only receives numeric feature columns (not `key`/datetimes), so `fit()` succeeds and the train/validation split works. Finally, I ensure a valid `submission.csv` (with `key,fare_amount`) is always written to the working directory.'
- What this solution (achieved 5.63018) has done: 'Your current RMSE (5.77) is worse than the target (4.23), so we should make small, legitimate improvements that usually reduce error without changing the model/training loop. The biggest win here is fixing the haversine calculation (it currently misses the required squares, which makes the feature wrong) and removing redundant/low-signal time features derived from `key` (which is not a real timestamp feature) while keeping the same overall feature-engineering + LightGBM approach. I also add the standard NYC Taxi sanity filters (fare bounds, passenger_count bounds, and NYC-ish coordinate bounds) to reduce training noise; this is a common minimal improvement for this competition and typically improves RMSE. Finally, I keep the submission schema identical (`key,fare_amount`) and still clip negative predictions to 0.'
- What this solution (achieved 5.43711) has done: 'Your RMSE (5.63) is worse than the target (4.23), so we should make small, legitimate improvements without changing the overall LightGBM + feature-engineering pipeline. The biggest issue is that you currently drop the raw lat/long columns after creating distance features; keeping the raw coordinates is a standard small improvement for this competition and usually reduces RMSE while preserving the same model/training approach. I also remove the unused `key2` parsing (it adds work and can introduce NaTs/side-effects but is not used for modeling), and I add a minimal, metric-aligned post-process to clip extreme predictions to the same fare range used in training filters (0–250), which typically slightly improves RMSE stability. All paths and the submission schema (`key,fare_amount`) remain unchanged and a valid `submission.csv` is always written.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use("fivethirtyeight")

import geopy.distance

import os
import gc

INPUT_DIR_CANDIDATES = [
    "../input",
    "/kaggle/input",
    "/kaggle/data",
    "../kaggle/input",
    "../kaggle/data",
]
INPUT_DIR = None
for d in INPUT_DIR_CANDIDATES:
    if os.path.isdir(d):
        INPUT_DIR = d
        break

print("Detected INPUT_DIR:", INPUT_DIR)
if INPUT_DIR is not None:
    try:
        print("Top-level listing:", os.listdir(INPUT_DIR)[:20])
    except Exception as e:
        print("Could not list input dir:", e)




## === cell 1
def _resolve_path(filename: str) -> str:
    """
    Keep original relative paths if they exist; otherwise try common Kaggle locations.
    """
    p1 = os.path.join("../input", filename)
    if os.path.exists(p1):
        return p1

    if INPUT_DIR is not None:
        p2 = os.path.join(INPUT_DIR, filename)
        if os.path.exists(p2):
            return p2

        p3 = os.path.join(INPUT_DIR, "new-york-city-taxi-fare-prediction", filename)
        if os.path.exists(p3):
            return p3

    return filename


def load_Data():
    train_path = _resolve_path("train.csv")
    test_path = _resolve_path("test.csv")
    train = pd.read_csv(train_path, nrows=1_000_000, low_memory=True)
    test = pd.read_csv(test_path, nrows=10_000_000, low_memory=True)
    return train, test


def prepare_distance_features(df):
    df["longitude_distance"] = abs(df["pickup_longitude"] - df["dropoff_longitude"])
    df["latitude_distance"] = abs(df["pickup_latitude"] - df["dropoff_latitude"])

    df["distance_travelled"] = (
        df["longitude_distance"] ** 2 + df["latitude_distance"] ** 2
    ) ** 0.5
    df["distance_travelled_sin"] = np.sin(
        (df["longitude_distance"] ** 2 * df["latitude_distance"] ** 2) ** 0.5
    )
    df["distance_travelled_cos"] = np.cos(
        (df["longitude_distance"] ** 2 * df["latitude_distance"] ** 2) ** 0.5
    )
    df["distance_travelled_sin_sqrd"] = (
        np.sin((df["longitude_distance"] ** 2 * df["latitude_distance"] ** 2) ** 0.5)
        ** 2
    )
    df["distance_travelled_cos_sqrd"] = (
        np.cos((df["longitude_distance"] ** 2 * df["latitude_distance"] ** 2) ** 0.5)
        ** 2
    )

    R = 6371e3  # Metres
    phi1 = np.radians(df["pickup_latitude"])
    phi2 = np.radians(df["dropoff_latitude"])
    phi_chg = np.radians(df["pickup_latitude"] - df["dropoff_latitude"])
    delta_chg = np.radians(df["pickup_longitude"] - df["dropoff_longitude"])
    a = (np.sin(phi_chg / 2) ** 2) + np.cos(phi1) * np.cos(phi2) * (
        np.sin(delta_chg / 2) ** 2
    )
    c = 2 * np.arctan2(a**0.5, (1 - a) ** 0.5)
    d = R * c
    df["haversine"] = d

    y = np.sin(delta_chg * np.cos(phi2))
    x = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(delta_chg)
    df["bearing"] = np.arctan2(y, x)

    return df


def prepare_time_features(df):
    df["pickup_datetime"] = (
        df["pickup_datetime"].astype(str).str.replace(" UTC", "", regex=False)
    )
    df["pickup_datetime"] = pd.to_datetime(
        df["pickup_datetime"], format="%Y-%m-%d %H:%M:%S", errors="coerce"
    )

    df["hour_of_day"] = df.pickup_datetime.dt.hour

    iso = df.pickup_datetime.dt.isocalendar()
    df["week"] = iso.week.astype("int16")
    df["week_of_year"] = iso.week.astype("int16")

    df["month"] = df.pickup_datetime.dt.month
    df["year"] = df.pickup_datetime.dt.year
    df["day_of_year"] = df.pickup_datetime.dt.dayofyear
    df["weekday"] = df.pickup_datetime.dt.weekday
    df["quarter"] = df.pickup_datetime.dt.quarter
    df["day_of_month"] = df.pickup_datetime.dt.day

    return df


def _apply_common_filters_train(df):
    df = df[df["fare_amount"].between(0, 250)]
    df = df[df["passenger_count"].between(1, 6)]
    df = df[df["pickup_longitude"].between(-75, -72)]
    df = df[df["dropoff_longitude"].between(-75, -72)]
    df = df[df["pickup_latitude"].between(40, 42)]
    df = df[df["dropoff_latitude"].between(40, 42)]
    return df


def _apply_common_filters_test(df):
    df = df[df["passenger_count"].between(1, 6)]
    df = df[df["pickup_longitude"].between(-75, -72)]
    df = df[df["dropoff_longitude"].between(-75, -72)]
    df = df[df["pickup_latitude"].between(40, 42)]
    df = df[df["dropoff_latitude"].between(40, 42)]
    return df




## === cell 2
train, test = load_Data()



## === cell 3
train = prepare_time_features(train)
train = prepare_distance_features(train)

test = prepare_time_features(test)
test = prepare_distance_features(test)



## === cell 4
train.describe()



## === cell 5
train.info()



## === cell 6
train.isnull().sum()



## === cell 7
train = train.fillna(0)



## === cell 8
train.info()



## === cell 9
pass



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
train = _apply_common_filters_train(train)



## === cell 18
train = train[~((train["pickup_latitude"] > 90) | (train["pickup_latitude"] < -90))]
train = train[~((train["dropoff_latitude"] > 90) | (train["dropoff_latitude"] < -90))]
train = train[~((train["pickup_longitude"] > 180) | (train["pickup_longitude"] < -180))]
train = train[
    ~((train["dropoff_longitude"] > 180) | (train["dropoff_longitude"] < -180))
]



## === cell 19
test = _apply_common_filters_test(test)




## === cell 20
def _geodesic_km(lat1, lon1, lat2, lon2):
    try:
        return geopy.distance.geodesic((lat1, lon1), (lat2, lon2)).km
    except Exception:
        return np.nan


HAVERSINE_CAP_METERS = (
    100_000.0  # cap at 100km to reduce extreme outliers consistently in train/test
)
train["haversine"] = np.clip(
    train["haversine"].astype(float), 0.0, HAVERSINE_CAP_METERS
)
test["haversine"] = np.clip(test["haversine"].astype(float), 0.0, HAVERSINE_CAP_METERS)

train["distance"] = train["haversine"] / 1000.0
test["distance"] = test["haversine"] / 1000.0

gc.collect()



## === cell 21
pass



## === cell 22
pass



## === cell 23
print(
    "% of trips above 25 KM - {:0.2f}".format(
        train[train["distance"] > 25]["key"].count() * 100 / train.count()["key"]
    )
)



## === cell 24
column_list = [
    "passenger_count",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "distance",
    "haversine",
    "bearing",
    "hour_of_day",
    "week",
    "month",
    "year",
    "day_of_year",
    "weekday",
    "quarter",
    "day_of_month",
]

train[column_list] = train[column_list].apply(pd.to_numeric, errors="coerce").fillna(0)
test[column_list] = test[column_list].apply(pd.to_numeric, errors="coerce").fillna(0)

y_train_full = train["fare_amount"].astype(float)
X_train_full = train[column_list]
X_test = test[column_list]



## === cell 25
X_train_full.shape, y_train_full.shape



## === cell 26
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.1, random_state=42
)



## === cell 27
X_train.shape, X_val.shape, y_train.shape, y_val.shape



## === cell 28
from lightgbm import LGBMRegressor



## === cell 29
lgb = LGBMRegressor(
    objective="regression",
    num_leaves=5,
    learning_rate=0.05,
    n_estimators=720,
    max_bin=55,
    bagging_fraction=0.8,
    bagging_freq=5,
    feature_fraction=0.2319,
    feature_fraction_seed=9,
    bagging_seed=9,
    min_data_in_leaf=6,
    min_sum_hessian_in_leaf=11,
)



## === cell 30
lgb.fit(X_train, y_train)



## === cell 31
pred = lgb.predict(X_val)



## === cell 32
from sklearn.metrics import mean_squared_error, r2_score

rmse = mean_squared_error(y_val, pred, squared=False)
r2 = r2_score(y_val, pred)
rmse, r2



## === cell 33
lgb.fit(X_train_full, y_train_full)




## === cell 34
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



## === cell 35
y_pred = lgb.predict(X_test)



## === cell 36
y_pred = np.clip(y_pred, 0, 250)

submission = pd.DataFrame({"key": test["key"], "fare_amount": y_pred})
submission.to_csv("submission.csv", index=False)
submission.head()



## === cell 37
print("Wrote submission to:", os.path.abspath("submission.csv"))
print("Submission shape:", submission.shape)
print("Submission columns:", submission.columns.tolist())
print(submission.describe())
