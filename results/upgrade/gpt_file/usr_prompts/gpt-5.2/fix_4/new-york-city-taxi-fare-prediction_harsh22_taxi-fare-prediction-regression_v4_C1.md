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

5.48885

# 6. Current score

1042.10002

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 800.26126) has done: 'I fix the pandas boolean logic errors causing the early crashes by switching to boolean masks (and using `axis=1` with `.any()`), so the cleaning steps actually execute. Then I ensure the feature matrix going into scikit-learn has no NaNs by dropping/imputing remaining missing values consistently in both train/test, which resolves the `RandomForestRegressor`/`LinearRegression` fit failures. Finally, I write a valid Kaggle submission CSV with exactly `key,fare_amount` using the original test `key` column (kept separately before dropping), so the notebook produces an uploadable `.csv` end-to-end. These are correctness/stability fixes; they should also improve RMSE versus the current “no submission” state by enabling the intended model to train and predict.'
- What this solution (achieved 800.26125) has done: 'Your RMSE (800) indicates severe feature corruption and/or prediction blow-ups; the biggest issue is converting `key` to datetime and then later stringifying it, which destroys the required `key` IDs and makes submissions effectively mismatched. I keep your cleaning + feature engineering + models intact, but preserve the original test `key` exactly as provided (string), and stop converting `key` to datetime in both train/test. I also clip predictions to a valid non-negative range (fares can’t be negative), which is a minimal post-processing step that reduces catastrophic RMSE from a small number of negative/huge outputs without changing the models. Finally, I ensure NaNs introduced by datetime parsing are removed before modeling so training stays consistent.'
- What this solution (achieved 1042.10002) has done: 'Your RMSE of ~800 strongly suggests the engineered datetime features are mostly NaN (from unparsable `pickup_datetime` in test and/or train) and then imputed to extreme/invalid values, which makes both models output wildly wrong fares. I make a minimal, metric-aligned fix: parse `pickup_datetime` with the known NYC taxi format first (fast and reliable), and drop rows where datetime failed in train while filling test datetime-derived fields with sensible defaults. I also add one small but high-impact data-cleaning guardrail: remove obviously invalid longitude/latitude zeros and out-of-NYC-ish coordinate outliers (still preserving your feature set and models) so the models stop learning from corrupted geos. These changes keep your core logic (same features, same haversine, same RandomForest + LinearRegression and training flow) but should bring RMSE down sharply toward your target band.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt

import os

if os.path.exists("../input"):
    INPUT_DIR = "../input"
elif os.path.exists("/kaggle/input"):
    INPUT_DIR = "/kaggle/input"
else:
    INPUT_DIR = "/kaggle/input"

print("INPUT_DIR:", INPUT_DIR)
print(os.listdir(INPUT_DIR)[:20])



## === cell 1
train = pd.read_csv(f"{INPUT_DIR}/train.csv", nrows=1000000)
test = pd.read_csv(f"{INPUT_DIR}/test.csv")



## === cell 2
train.shape



## === cell 3
test.shape



## === cell 4
train.head(10)



## === cell 5
train.describe()



## === cell 6
train.isnull().sum().sort_values(ascending=False)



## === cell 7
train = train.drop(train[train.isnull().any(axis=1)].index, axis=0)



## === cell 8
train.shape



## === cell 9
train["fare_amount"].describe()



## === cell 10
from collections import Counter

Counter(train["fare_amount"] < 0)



## === cell 11
train = train.drop(train[train["fare_amount"] < 0].index, axis=0)
train.shape



## === cell 12
train["fare_amount"].describe()



## === cell 13
train["fare_amount"].sort_values(ascending=False)



## === cell 14
train["passenger_count"].describe()



## === cell 15
train[train["passenger_count"] > 8]



## === cell 16
train = train.drop(train[train["passenger_count"] == 208].index, axis=0)



## === cell 17
train["passenger_count"].describe()



## === cell 18
train["pickup_latitude"].describe()



## === cell 19
train[train["pickup_latitude"] < -90]



## === cell 20
train[train["pickup_latitude"] > 90]



## === cell 21
mask_bad_pickup_lat = (train["pickup_latitude"] < -90) | (train["pickup_latitude"] > 90)
train = train.drop(train[mask_bad_pickup_lat].index, axis=0)



## === cell 22
train.shape



## === cell 23
train["pickup_longitude"].describe()



## === cell 24
train[train["pickup_longitude"] < -180]



## === cell 25
train[train["pickup_longitude"] > 180]



## === cell 26
mask_bad_pickup_lon = (train["pickup_longitude"] < -180) | (
    train["pickup_longitude"] > 180
)
train = train.drop(train[mask_bad_pickup_lon].index, axis=0)



## === cell 27
train[train["dropoff_latitude"] < -90]



## === cell 28
train[train["dropoff_latitude"] > 90]



## === cell 29
mask_bad_dropoff_lat = (train["dropoff_latitude"] < -90) | (
    train["dropoff_latitude"] > 90
)
train = train.drop(train[mask_bad_dropoff_lat].index, axis=0)



## === cell 30
train[(train["dropoff_longitude"] < -180) | (train["dropoff_longitude"] > 180)].head()



## === cell 31
train.dtypes




## === cell 32
def parse_pickup_datetime(s: pd.Series) -> pd.Series:
    dt = pd.to_datetime(s, format="%Y-%m-%d %H:%M:%S UTC", errors="coerce")
    if dt.isna().any():
        dt2 = pd.to_datetime(s, errors="coerce", infer_datetime_format=True)
        dt = dt.fillna(dt2)
    return dt


train["pickup_datetime"] = parse_pickup_datetime(train["pickup_datetime"])
test["pickup_datetime"] = parse_pickup_datetime(test["pickup_datetime"])

train = train.drop(train[train["pickup_datetime"].isna()].index, axis=0)



## === cell 33
train.dtypes




## === cell 34
def haversine_distance(lat1, long1, lat2, long2):
    data = [train, test]
    for i in data:
        r = 6371
        phi1 = np.radians(i[lat1])
        phi2 = np.radians(i[lat2])

        delta_phi = np.radians(i[lat2] - i[lat1])
        delta_lambda = np.radians(i[long2] - i[long1])

        a = (
            np.sin(delta_phi / 2.0) ** 2
            + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2
        )
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

        d = r * c  # in kilometers
        i["H_Distance"] = d
    return d




## === cell 35
haversine_distance(
    "pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude"
)



## === cell 36
train["H_Distance"].head(10)



## === cell 37
data = [train, test]
for i in data:
    i["Year"] = i["pickup_datetime"].dt.year
    i["Month"] = i["pickup_datetime"].dt.month
    i["Date"] = i["pickup_datetime"].dt.day
    i["Day of Week"] = i["pickup_datetime"].dt.dayofweek
    i["Hour"] = i["pickup_datetime"].dt.hour

for col in ["Year", "Month", "Date", "Day of Week", "Hour"]:
    if test[col].isna().any():
        fill_val = int(train[col].median())
        test[col] = test[col].fillna(fill_val)



## === cell 38
train.loc[
    ((train["pickup_latitude"] == 0) & (train["pickup_longitude"] == 0))
    & ((train["dropoff_latitude"] != 0) & (train["dropoff_longitude"] != 0))
    & (train["fare_amount"] == 0)
]



## === cell 39
train = train.drop(
    train.loc[
        ((train["pickup_latitude"] == 0) & (train["pickup_longitude"] == 0))
        & ((train["dropoff_latitude"] != 0) & (train["dropoff_longitude"] != 0))
        & (train["fare_amount"] == 0)
    ].index,
    axis=0,
)



## === cell 40
train.shape



## === cell 41
train.loc[
    ((train["pickup_latitude"] != 0) & (train["pickup_longitude"] != 0))
    & ((train["dropoff_latitude"] == 0) & (train["dropoff_longitude"] == 0))
    & (train["fare_amount"] == 0)
]



## === cell 42
train = train.drop(
    train.loc[
        ((train["pickup_latitude"] != 0) & (train["pickup_longitude"] != 0))
        & ((train["dropoff_latitude"] == 0) & (train["dropoff_longitude"] == 0))
        & (train["fare_amount"] == 0)
    ].index,
    axis=0,
)



## === cell 43
high_distance = train.loc[(train["H_Distance"] > 200) & (train["fare_amount"] != 0)]



## === cell 44
high_distance



## === cell 45
high_distance = high_distance.copy()
high_distance.loc[:, "H_Distance"] = high_distance.apply(
    lambda row: (row["fare_amount"] - 2.50) / 1.56, axis=1
)



## === cell 46
train.update(high_distance)



## === cell 47
train[train["H_Distance"] == 0]



## === cell 48
train[(train["H_Distance"] == 0) & (train["fare_amount"] == 0)]



## === cell 49
train = train.drop(
    train[(train["H_Distance"] == 0) & (train["fare_amount"] == 0)].index, axis=0
)



## === cell 50
train[(train["H_Distance"] == 0)].shape



## === cell 51
rush_hour = train.loc[
    (
        ((train["Hour"] >= 6) & (train["Hour"] <= 20))
        & ((train["Day of Week"] >= 1) & (train["Day of Week"] <= 5))
    )
    & (train["H_Distance"] == 0)
    & (train["fare_amount"] < 2.5)
]
rush_hour



## === cell 52
train = train.drop(rush_hour.index, axis=0)



## === cell 53
non_rush_hour = train.loc[
    (
        ((train["Hour"] < 6) | (train["Hour"] > 20))
        & ((train["Day of Week"] >= 1) & (train["Day of Week"] <= 5))
    )
    & (train["H_Distance"] == 0)
    & (train["fare_amount"] < 3.0)
]
non_rush_hour



## === cell 54
weekends = train.loc[
    ((train["Day of Week"] == 0) | (train["Day of Week"] == 6))
    & (train["H_Distance"] == 0)
    & (train["fare_amount"] < 3.0)
]
weekends



## === cell 55
train.loc[(train["H_Distance"] != 0) & (train["fare_amount"] == 0)]



## === cell 56
scenario_3 = train.loc[(train["H_Distance"] != 0) & (train["fare_amount"] == 0)]



## === cell 57
len(scenario_3)



## === cell 58
scenario_3.sort_values("H_Distance", ascending=False)



## === cell 59
scenario_3 = scenario_3.copy()
scenario_3.loc[:, "fare_amount"] = scenario_3.apply(
    lambda row: ((row["H_Distance"] * 1.56) + 2.50), axis=1
)



## === cell 60
scenario_3["fare_amount"].head()



## === cell 61
train.update(scenario_3)



## === cell 62
train.loc[(train["H_Distance"] == 0) & (train["fare_amount"] != 0)]



## === cell 63
scenario_4 = train.loc[(train["H_Distance"] == 0) & (train["fare_amount"] != 0)]



## === cell 64
len(scenario_4)



## === cell 65
scenario_4.loc[(scenario_4["fare_amount"] <= 3.0) & (scenario_4["H_Distance"] == 0)]



## === cell 66
scenario_4.loc[(scenario_4["fare_amount"] > 3.0) & (scenario_4["H_Distance"] == 0)]



## === cell 67
scenario_4_sub = scenario_4.loc[
    (scenario_4["fare_amount"] > 3.0) & (scenario_4["H_Distance"] == 0)
]



## === cell 68
scenario_4_sub = scenario_4.loc[
    (scenario_4["fare_amount"] > 3.0) & (scenario_4["H_Distance"] == 0)
]



## === cell 69
len(scenario_4_sub)



## === cell 70
scenario_4_sub = scenario_4_sub.copy()
scenario_4_sub.loc[:, "H_Distance"] = scenario_4_sub.apply(
    lambda row: ((row["fare_amount"] - 2.50) / 1.56), axis=1
)



## === cell 71
train.update(scenario_4_sub)



## === cell 72
train.columns



## === cell 73
test.columns



## === cell 74
nyc_bbox = {
    "min_lon": -74.5,
    "max_lon": -72.8,
    "min_lat": 40.0,
    "max_lat": 41.8,
}

geo_ok = (
    (train["pickup_longitude"].between(nyc_bbox["min_lon"], nyc_bbox["max_lon"]))
    & (train["dropoff_longitude"].between(nyc_bbox["min_lon"], nyc_bbox["max_lon"]))
    & (train["pickup_latitude"].between(nyc_bbox["min_lat"], nyc_bbox["max_lat"]))
    & (train["dropoff_latitude"].between(nyc_bbox["min_lat"], nyc_bbox["max_lat"]))
    & ~(
        ((train["pickup_longitude"] == 0) & (train["pickup_latitude"] == 0))
        | ((train["dropoff_longitude"] == 0) & (train["dropoff_latitude"] == 0))
    )
)
train = train.loc[geo_ok].copy()

test_key = test["key"].astype(str).copy()

train = train.drop(["key", "pickup_datetime"], axis=1)
test = test.drop(["key", "pickup_datetime"], axis=1)



## === cell 75
required_cols = [
    "Year",
    "Month",
    "Date",
    "Day of Week",
    "Hour",
    "H_Distance",
    "passenger_count",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "fare_amount",
]
required_cols = [c for c in required_cols if c in train.columns]
train = train.dropna(subset=required_cols)

x_train = train.iloc[:, train.columns != "fare_amount"]
y_train = train["fare_amount"].values
x_test = test

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median")
x_train_imp = pd.DataFrame(
    imputer.fit_transform(x_train), columns=x_train.columns, index=x_train.index
)
x_test_imp = pd.DataFrame(
    imputer.transform(x_test), columns=x_test.columns, index=x_test.index
)

mask_y_ok = ~pd.isna(y_train)
x_train_imp = x_train_imp.loc[mask_y_ok]
y_train = y_train[mask_y_ok]



## === cell 76
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(random_state=42, n_jobs=-1)
rf.fit(x_train_imp, y_train)
rf_predict = rf.predict(x_test_imp)

rf_predict = np.clip(rf_predict, 0.0, None)



## === cell 77
submission = pd.DataFrame({"key": test_key, "fare_amount": rf_predict})
submission.to_csv("submission_1.csv", index=False)
submission.head(20)



## === cell 78
from sklearn import linear_model

lr = linear_model.LinearRegression()
lr.fit(x_train_imp, y_train)
lr_predict = lr.predict(x_test_imp)

lr_predict = np.clip(lr_predict, 0.0, None)

submission2 = pd.DataFrame({"key": test_key, "fare_amount": lr_predict})
submission2.to_csv("submission_2.csv", index=False)
submission2.head(20)
