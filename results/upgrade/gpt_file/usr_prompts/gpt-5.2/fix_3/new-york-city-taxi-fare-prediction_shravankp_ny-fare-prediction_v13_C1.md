# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

eli5==0.13.0
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

4.57128

# 6. Current score

8.2181

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 8.21817) has done: 'I fix the datetime parsing bug by handling already tz-aware timestamps (using `utc=True` and avoiding `tz_localize`), and replace the platform-dependent `strftime('%-H')`/`'%-d'`/`'%-y'` with robust `.dt.hour/.dt.day/.dt.year`. I also prevent the exploratory cells that currently error (the `df/grouped` analysis and `eli5` permutation importance) from stopping the pipeline by guarding them, since they are not needed to train or create `submission.csv`. Finally, I keep the model/training approach intact but set reasonable default XGBoost hyperparameters (and a fixed random seed) to improve RMSE toward your target without changing the overall logic.'
- What this solution (achieved 8.2181) has done: 'I fix the pathing so the notebook reliably reads the competition CSVs from `/kaggle/input/new-york-city-taxi-fare-prediction/` (your current `../input/...` can fail depending on where it’s run). I also remove an unnecessary `.round(2)` on predictions (this is score-harming for RMSE and is not required for submission), which should improve your score toward the 4.57 target without changing the model/training approach. Finally, I keep the `eli5` block guarded (it errors due to protobuf incompatibility) and ensure the submission is written as `submission.csv` with the exact required columns and row alignment.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd

import os

print(os.listdir("/kaggle/input"))

import seaborn as sns
import matplotlib.pyplot as plt
import datetime as dt
import math

DATA_DIR = "/kaggle/input/new-york-city-taxi-fare-prediction"
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")



## === cell 1
train = pd.read_csv(TRAIN_PATH, nrows=1000000)
test = pd.read_csv(TEST_PATH)
train.head()



## === cell 2
train.isnull().sum()



## === cell 3
train = train.dropna(how="any", axis=0)



## === cell 4
train["abs_diff_longitude"] = np.abs(
    train["dropoff_longitude"] - train["pickup_longitude"]
)
train["abs_diff_latitude"] = np.abs(
    train["dropoff_latitude"] - train["pickup_latitude"]
)
test["abs_diff_longitude"] = np.abs(
    test["dropoff_longitude"] - test["pickup_longitude"]
)
test["abs_diff_latitude"] = np.abs(test["dropoff_latitude"] - test["pickup_latitude"])



## === cell 5
train = train.loc[train["fare_amount"] > 0, :]
train = train.loc[(train["passenger_count"] <= 6) & (train["passenger_count"] > 0), :]
train = train.loc[
    (train["abs_diff_latitude"] < 2) & (train["abs_diff_longitude"] < 2), :
]
train = train.loc[
    (train["abs_diff_latitude"] > 0) & (train["abs_diff_longitude"] > 0), :
]



## === cell 6
train.loc[:, "timestamp_with_key"] = train.loc[:, "key"]
test.loc[:, "timestamp_with_key"] = test.loc[:, "key"]
train.key = pd.DataFrame({"key": train["key"].str.split(".").str[1].astype("int")})
test.key = pd.DataFrame({"key": test["key"].str.split(".").str[1].astype("int")})



## === cell 7
train["pickup_datetime"] = pd.to_datetime(
    train["pickup_datetime"], utc=True, errors="coerce"
)
test["pickup_datetime"] = pd.to_datetime(
    test["pickup_datetime"], utc=True, errors="coerce"
)

train = train.dropna(subset=["pickup_datetime"]).reset_index(drop=True)
test = test.dropna(subset=["pickup_datetime"]).reset_index(drop=True)

train.loc[:, "hour_no"] = train["pickup_datetime"].dt.hour.astype("int")
test.loc[:, "hour_no"] = test["pickup_datetime"].dt.hour.astype("int")

train.loc[:, "weekday_no"] = train["pickup_datetime"].dt.weekday.astype(
    "int"
)  # Monday=0
test.loc[:, "weekday_no"] = test["pickup_datetime"].dt.weekday.astype("int")

train.loc[:, "day_no"] = train["pickup_datetime"].dt.day.astype("int")
test.loc[:, "day_no"] = test["pickup_datetime"].dt.day.astype("int")

train.loc[:, "year_no"] = (train["pickup_datetime"].dt.year % 100).astype("int")
test.loc[:, "year_no"] = (test["pickup_datetime"].dt.year % 100).astype("int")




## === cell 8
def dist_haversine(x):
    R = 6371
    picklat = math.radians(x[1])
    droplat = math.radians(x[3])
    latdiff = abs(droplat - picklat)
    picklon = math.radians(x[0])
    droplon = math.radians(x[2])
    londiff = abs(droplon - picklon)

    a = math.sin(latdiff / 2) * math.sin(latdiff / 2) + math.cos(picklat) * math.cos(
        droplat
    ) * math.sin(londiff / 2) * math.sin(londiff / 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


train["dist_haversine_km"] = pd.DataFrame(
    list(
        map(
            lambda x: dist_haversine(x),
            train[
                [
                    "pickup_longitude",
                    "pickup_latitude",
                    "dropoff_longitude",
                    "dropoff_latitude",
                ]
            ].values,
        )
    ),
    index=train.index,
)

test["dist_haversine_km"] = pd.DataFrame(
    list(
        map(
            lambda x: dist_haversine(x),
            test[
                [
                    "pickup_longitude",
                    "pickup_latitude",
                    "dropoff_longitude",
                    "dropoff_latitude",
                ]
            ].values,
        )
    ),
    index=test.index,
)



## === cell 9
train["fare_per_km"] = train["fare_amount"] / (train["dist_haversine_km"])
train["fare_per_km_passenger"] = train["fare_amount"] / (
    train["dist_haversine_km"] * train["passenger_count"]
)



## === cell 10
train.groupby("key").agg(
    {
        "fare_per_km_passenger": "mean",
        "key": "count",
        "passenger_count": "mean",
        "fare_amount": "mean",
    }
)



## === cell 11
train.loc[
    train["fare_per_km_passenger"] > 20,
    ["fare_per_km_passenger", "fare_amount", "dist_haversine_km"],
]



## === cell 12
try:
    grouped_df = train.groupby("key")
    count = 0
    df = None
    for key, item in grouped_df:
        count += 1
        if count == 2:
            filtered = grouped_df.get_group(key)["dist_haversine_km"] > 1
            df = pd.DataFrame(
                grouped_df.get_group(key)
                .loc[filtered, :]
                .sort_values(by="pickup_datetime")
            )
            break

    if (
        df is not None
        and not df.empty
        and pd.api.types.is_datetime64_any_dtype(df["pickup_datetime"])
    ):
        indexes = ["key", df["pickup_datetime"].dt.strftime("%a"), "hour_no"]
        grouped = df.groupby(indexes).agg(
            {"fare_per_km_passenger": "mean", "hour_no": "count"}
        )
        grouped.rename(columns={"hour_no": "count"}, inplace=True)
        grouped
except Exception as e:
    print("Skipping exploratory grouping block due to:", repr(e))



## === cell 13
try:
    if "grouped" in globals():
        reindexed = grouped.reset_index().drop("key", axis=1)
        get_max_count = reindexed.groupby(["pickup_datetime"]).agg({"count": "max"})
        get_max_count = get_max_count.reindex(
            reindexed["pickup_datetime"], method="ffill"
        )
        reindexed = reindexed.set_index("pickup_datetime")
        reindexed.loc[get_max_count["count"] == reindexed["count"], :]
except Exception as e:
    print("Skipping exploratory reindexing block due to:", repr(e))



## === cell 14
train = train.loc[~((train["fare_per_km"] < 0.2) & (train["dist_haversine_km"] > 1))]
train = train.loc[~((train["dist_haversine_km"] < 0.01) & (train["fare_per_km"] > 50))]



## === cell 15
print(
    train.shape[0]
    - train.loc[
        train["pickup_latitude"].between(40.5, 41)
        | train["dropoff_latitude"].between(40.5, 41)
        | train["pickup_longitude"].between(-74, -73.9)
        | train["dropoff_longitude"].between(-74, -73.9)
    ].shape[0]
)
old_train = train.copy()
train = train.loc[
    train["pickup_latitude"].between(40.5, 41)
    & train["dropoff_latitude"].between(40.5, 41)
    & train["pickup_longitude"].between(-74, -73.9)
    & train["dropoff_longitude"].between(-74, -73.9)
]



## === cell 16
train.loc[:, "pickuplat_no"], pick_lat_bin = pd.cut(
    train["pickup_latitude"], 100, labels=False, retbins=True
)
train.loc[:, "pickuplong_no"], pick_long_bin = pd.cut(
    train["pickup_longitude"], 100, labels=False, retbins=True
)
train.loc[:, "dropofflat_no"], drop_lat_bin = pd.cut(
    train["dropoff_latitude"], 100, labels=False, retbins=True
)
train.loc[:, "dropofflong_no"], drop_long_bin = pd.cut(
    train["dropoff_longitude"], 100, labels=False, retbins=True
)
test["pickuplat_no"] = pd.cut(
    test["pickup_latitude"], pick_lat_bin, labels=False
).fillna(int(train.loc[:, "pickuplat_no"].mean()))
test["pickuplong_no"] = pd.cut(
    test["pickup_longitude"], pick_long_bin, labels=False
).fillna(int(train.loc[:, "pickuplong_no"].mean()))



## === cell 17
test["dropofflat_no"] = pd.cut(
    test["dropoff_latitude"], drop_lat_bin, labels=False
).fillna(int(train.loc[:, "dropofflat_no"].mean()))
test["dropofflong_no"] = pd.cut(
    test["dropoff_longitude"], drop_long_bin, labels=False
).fillna(int(train.loc[:, "dropofflong_no"].mean()))



## === cell 18
train.loc[:, "pickdrop_lat_diff"] = abs(
    train["pickuplat_no"].astype(int) - train["dropofflat_no"].astype(int)
)
train.loc[:, "pickdrop_long_diff"] = abs(
    train["pickuplong_no"].astype(int) - train["dropofflong_no"].astype(int)
)
train.loc[:, "final_dist_factor"] = train["pickdrop_lat_diff"].astype(int) + train[
    "pickdrop_long_diff"
].astype(int)
test.loc[:, "pickdrop_lat_diff"] = abs(
    test["pickuplat_no"].astype(int) - test["dropofflat_no"].astype(int)
)
test.loc[:, "pickdrop_long_diff"] = abs(
    test["pickuplong_no"].astype(int) - test["dropofflong_no"].astype(int)
)
test.loc[:, "final_dist_factor"] = test["pickdrop_lat_diff"].astype(int) + test[
    "pickdrop_long_diff"
].astype(int)



## === cell 19
print(train.shape, test.shape)



## === cell 20
train = train.drop(["fare_per_km_passenger", "fare_per_km"], axis=1)




## === cell 21
def calc_cwd_factor(df, col):
    new_df = df.groupby(col)["key"].count().sort_values(ascending=False).reset_index()
    new_df["cwd_factor"] = 1
    count = 1
    for i in range(1, new_df.shape[0]):
        count += 1
        if new_df.loc[i - 1, "key"] == new_df.loc[i, "key"]:
            count -= 1
        new_df.loc[i, "cwd_factor"] = count
    new_df.index = new_df[col]
    return new_df


fact_df = calc_cwd_factor(train, "pickuplat_no")
train.loc[:, "pickuplat_cwd_factor"] = list(
    map(lambda x: fact_df.loc[x, "cwd_factor"], train["pickuplat_no"])
)
fact_df = calc_cwd_factor(train, "pickuplong_no")
train.loc[:, "pickuplong_cwd_factor"] = list(
    map(lambda x: fact_df.loc[x, "cwd_factor"], train["pickuplong_no"])
)
fact_df = calc_cwd_factor(train, "dropofflat_no")
train.loc[:, "dropofflat_cwd_factor"] = list(
    map(lambda x: fact_df.loc[x, "cwd_factor"], train["dropofflat_no"])
)
fact_df = calc_cwd_factor(train, "dropofflong_no")
train.loc[:, "dropofflong_cwd_factor"] = list(
    map(lambda x: fact_df.loc[x, "cwd_factor"], train["dropofflong_no"])
)
fact_df = calc_cwd_factor(test, "pickuplat_no")
test.loc[:, "pickuplat_cwd_factor"] = list(
    map(lambda x: fact_df.loc[x, "cwd_factor"], test["pickuplat_no"])
)
fact_df = calc_cwd_factor(test, "pickuplong_no")
test.loc[:, "pickuplong_cwd_factor"] = list(
    map(lambda x: fact_df.loc[x, "cwd_factor"], test["pickuplong_no"])
)
fact_df = calc_cwd_factor(test, "dropofflat_no")
test.loc[:, "dropofflat_cwd_factor"] = list(
    map(lambda x: fact_df.loc[x, "cwd_factor"], test["dropofflat_no"])
)
fact_df = calc_cwd_factor(test, "dropofflong_no")
test.loc[:, "dropofflong_cwd_factor"] = list(
    map(lambda x: fact_df.loc[x, "cwd_factor"], test["dropofflong_no"])
)



## === cell 22
print(train.shape, test.shape)



## === cell 23
import sklearn
from sklearn import *
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle



## === cell 24
orig_train = train.copy()
orig_test = test.copy()



## === cell 25
train = shuffle(train.iloc[:, :], random_state=42).reset_index(drop=True)
val = train.iloc[int(0.9 * train.shape[0]) :, :].copy()
train = train.iloc[: int(0.9 * train.shape[0]), :].copy()

cols_to_normalize = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "abs_diff_longitude",
    "abs_diff_latitude",
    "dist_haversine_km",
]

transformer = Normalizer().fit(train.loc[:, cols_to_normalize])
train.loc[:, cols_to_normalize] = transformer.transform(train.loc[:, cols_to_normalize])
val.loc[:, cols_to_normalize] = transformer.transform(val.loc[:, cols_to_normalize])
test.loc[:, cols_to_normalize] = transformer.transform(test.loc[:, cols_to_normalize])



## === cell 26
import seaborn as sns

categorical_cols = [
    i
    for i in train.columns
    if i
    not in [
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "abs_diff_longitude",
        "abs_diff_latitude",
        "dist_haversine_km",
        "key",
        "pickup_datetime",
        "timestamp_with_key",
        "fare_amount",
    ]
]
numerical_cols = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "abs_diff_longitude",
    "abs_diff_latitude",
    "dist_haversine_km",
    "fare_amount",
]
object_cols = ["key", "pickup_datetime", "timestamp_with_key"]
cor = train.loc[:, numerical_cols]
f, ax = plt.subplots(1, 1, figsize=(18, 7))
sns.heatmap(cor.corr(), annot=True)



## === cell 27
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import mutual_info_regression

obj = SelectKBest(f_regression, k=10)
obj.fit(train[categorical_cols], train["fare_amount"])



## === cell 28
categories_selected = []
for i in range(len(categorical_cols)):
    if obj.get_support()[i]:
        categories_selected.append(categorical_cols[i])
categories_selected



## === cell 29
train_y = train["fare_amount"][:]
val_y = val["fare_amount"][:]
cols = [
    i
    for i in categories_selected + numerical_cols
    if i not in ["fare_amount"] + object_cols
]
train_x = train.loc[:, cols]
val_x = val.loc[:, cols]
test_x = test.loc[:, cols]



## === cell 30
import xgboost as xgb
from xgboost import XGBRegressor



## === cell 31
xgbr = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1,
)
xgb_train = xgbr.fit(train_x, train_y)



## === cell 32
pred_train = xgbr.predict(train_x)
pred_val = xgbr.predict(val_x)
pred_test = xgbr.predict(test_x)



## === cell 33
from sklearn.metrics import mean_squared_error

rmse_train = np.sqrt(mean_squared_error(train_y, pred_train))
rmse_val = np.sqrt(mean_squared_error(val_y, pred_val))
print(rmse_train, rmse_val)



## === cell 34
try:
    import eli5
    from eli5.sklearn import PermutationImportance

    perm = PermutationImportance(xgb_train, random_state=1).fit(val_x, val_y)
    eli5.show_weights(perm, feature_names=val_x.columns.tolist())
except Exception as e:
    print("Skipping eli5 permutation importance due to:", repr(e))



## --- ERROR in cell 34, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 35
final = pd.DataFrame(
    {"key": test["timestamp_with_key"].values, "fare_amount": pred_test},
    columns=["key", "fare_amount"],
)

final.to_csv("submission.csv", index=False)

print(final.shape)
final.head()
