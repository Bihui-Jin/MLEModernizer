# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.ensemble import RandomForestRegressor
import os

np.random.seed(42)

try:
    print(os.listdir("../input"))
except Exception as e:
    print("Could not list ../input:", repr(e))



## === cell 1
TRAIN_PATH = "../input/train.csv"
TEST_PATH = "../input/test.csv"

train_usecols = [
    "key",
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
test_usecols = [
    "key",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]

dtypes_train = {
    "key": "string",
    "fare_amount": "float64",
    "pickup_longitude": "float64",
    "pickup_latitude": "float64",
    "dropoff_longitude": "float64",
    "dropoff_latitude": "float64",
    "passenger_count": "int16",
}
dtypes_test = {
    "key": "string",
    "pickup_longitude": "float64",
    "pickup_latitude": "float64",
    "dropoff_longitude": "float64",
    "dropoff_latitude": "float64",
    "passenger_count": "int16",
}

train_df = pd.read_csv(
    TRAIN_PATH,
    nrows=1000000,
    usecols=train_usecols,
    dtype=dtypes_train,
    low_memory=False,
)
test_df = pd.read_csv(
    TEST_PATH, usecols=test_usecols, dtype=dtypes_test, low_memory=False
)



## === cell 2
train_df.shape



## === cell 3
train_df.columns



## === cell 4
train_df.head(3)



## === cell 5
None



## === cell 6
None



## === cell 7
None



## === cell 8
None



## === cell 9
train_df = train_df.dropna()



## === cell 10
None



## === cell 11
Counter(train_df["fare_amount"] < 0)



## === cell 12
m = (train_df["fare_amount"] >= 0) & (train_df["passenger_count"] <= 6)
train_df = train_df.loc[m].copy()
train_df.shape



## === cell 13
None



## === cell 14
Counter(train_df["passenger_count"] > 6)



## === cell 15
None



## === cell 16
Counter(train_df["pickup_latitude"] < -90)



## === cell 17
Counter(train_df["pickup_latitude"] > 90)



## === cell 18
m = train_df["pickup_latitude"].between(-90, 90) & train_df["pickup_longitude"].between(
    -180, 180
)
train_df = train_df.loc[m].copy()



## === cell 19
train_df.shape



## === cell 20
Counter(train_df["pickup_longitude"] < -180)



## === cell 21
Counter(train_df["pickup_longitude"] > 180)



## === cell 22
train_df = train_df.loc[train_df["pickup_longitude"] >= -180].copy()



## === cell 23
train_df.shape



## === cell 24
train_df.dtypes



## === cell 25
train_df.head(3)



## === cell 26
train_df["pickup_datetime"] = pd.to_datetime(
    train_df["pickup_datetime"], errors="coerce", cache=True
)



## === cell 27
train_df.dtypes



## === cell 28
test_df.dtypes



## === cell 29
train_df.head(3)



## === cell 30
test_df["pickup_datetime"] = pd.to_datetime(
    test_df["pickup_datetime"], errors="coerce", cache=True
)



## === cell 31
test_df.dtypes



## === cell 32
test_df.head(3)



## === cell 33
train_df.head(3)



## === cell 34
train_df = train_df.dropna(subset=["pickup_datetime"]).copy()

for df in (train_df, test_df):
    dt = df["pickup_datetime"].dt
    df["date"] = dt.day.astype("int16")
    df["month"] = dt.month.astype("int16")
    df["day_of_week"] = dt.dayofweek.astype("int16")
    df["hour"] = dt.hour.astype("int16")
    df["year"] = dt.year.astype("int16")



## === cell 35
train_df.head(3)



## === cell 36
None



## === cell 37
nyc_mask_early = (
    train_df["pickup_latitude"].between(40.5, 41.0)
    & train_df["dropoff_latitude"].between(40.5, 41.0)
    & train_df["pickup_longitude"].between(-74.5, -73.5)
    & train_df["dropoff_longitude"].between(-74.5, -73.5)
)
train_df = train_df.loc[nyc_mask_early].copy()




## === cell 38
def _haversine_km(df, lat1, long1, lat2, long2, R=6367.0):
    lat1v = df[lat1].to_numpy()
    lat2v = df[lat2].to_numpy()
    lon1v = df[long1].to_numpy()
    lon2v = df[long2].to_numpy()

    phi1 = np.radians(lat1v)
    phi2 = np.radians(lat2v)
    dphi = np.radians(lat2v - lat1v)
    dlmb = np.radians(lon2v - lon1v)

    a = (np.sin(dphi / 2.0) ** 2) + np.cos(phi1) * np.cos(phi2) * (
        np.sin(dlmb / 2.0) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


def sphere_distance(lat1, long1, lat2, long2):
    train_df["S_Distance"] = _haversine_km(train_df, lat1, long1, lat2, long2)
    test_df["S_Distance"] = _haversine_km(test_df, lat1, long1, lat2, long2)
    return train_df["S_Distance"]




## === cell 39
sphere_distance(
    "pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude"
)



## === cell 40
train_df.head(3)



## === cell 41
if False:
    plt.hist(train_df["passenger_count"], bins=15)
    plt.xlabel("No. of Passengers")
    plt.ylabel("Frequency")



## === cell 42
if False:
    plt.scatter(x=train_df["passenger_count"], y=train_df["fare_amount"], s=2.0)
    plt.xlabel("No. of Passengers")
    plt.ylabel("Fare")



## === cell 43
if False:
    plt.scatter(x=train_df["date"], y=train_df["fare_amount"])
    plt.xlabel("Date")
    plt.ylabel("Fare")



## === cell 44
if False:
    plt.hist(train_df["hour"], bins=50)
    plt.xlabel("Date")
    plt.ylabel("Fare")



## === cell 45
if False:
    plt.hist(train_df["day_of_week"], bins=20)
    plt.xlabel("Date")
    plt.ylabel("Fare")



## === cell 46
if False:
    plt.scatter(x=train_df["day_of_week"], y=train_df["fare_amount"])
    plt.xlabel("Date of week")
    plt.ylabel("Fare")



## === cell 47
len(train_df)



## === cell 48
if False:
    train_df.sort_values(["S_Distance", "fare_amount"], ascending=False)



## === cell 49
if False:
    dis_0 = train_df.loc[(train_df["S_Distance"] == 0), ["S_Distance"]]
    dis_1 = train_df.loc[
        (train_df["S_Distance"] > 0) & (train_df["S_Distance"] <= 10), ["S_Distance"]
    ]
    dis_2 = train_df.loc[
        (train_df["S_Distance"] > 10) & (train_df["S_Distance"] <= 50), ["S_Distance"]
    ]
    dis_3 = train_df.loc[
        (train_df["S_Distance"] > 50) & (train_df["S_Distance"] <= 100), ["S_Distance"]
    ]
    dis_4 = train_df.loc[
        (train_df["S_Distance"] > 100) & (train_df["S_Distance"] <= 200), ["S_Distance"]
    ]
    dis_5 = train_df.loc[
        (train_df["S_Distance"] > 200) & (train_df["S_Distance"] <= 300), ["S_Distance"]
    ]
    dis_6 = train_df.loc[
        (train_df["S_Distance"] > 300) & (train_df["S_Distance"] <= 500), ["S_Distance"]
    ]
    dis_7 = train_df.loc[(train_df["S_Distance"] > 500), ["S_Distance"]]
    dis_0["bins"] = "0"
    dis_1["bins"] = "0-10"
    dis_2["bins"] = "11-50"
    dis_3["bins"] = "51-100"
    dis_4["bins"] = "101-200"
    dis_5["bins"] = "201-300"
    dis_6["bins"] = "301-500"
    dis_7["bins"] = ">500"
    dis_bin = pd.concat([dis_0, dis_1, dis_2, dis_3, dis_4, dis_5, dis_6, dis_7])
    dis_bin



## === cell 50
if False:
    x = Counter(dis_bin["bins"])
    x



## === cell 51
if False:
    train_df.loc[
        ((train_df["pickup_latitude"] == 0) & (train_df["pickup_longitude"] == 0))
        & ((train_df["dropoff_latitude"] != 0) & (train_df["dropoff_longitude"] != 0))
        & (train_df["fare_amount"] == 0)
    ]



## === cell 52
if False:
    train_df.loc[
        ((train_df["pickup_latitude"] == 0) & (train_df["pickup_longitude"] == 0))
        & ((train_df["dropoff_latitude"] != 0) & (train_df["dropoff_longitude"] != 0))
        & (train_df["fare_amount"] == 0)
    ]



## === cell 53
mask_scenario_dup = (
    (train_df["pickup_latitude"] == 0)
    & (train_df["pickup_longitude"] == 0)
    & (train_df["dropoff_latitude"] != 0)
    & (train_df["dropoff_longitude"] != 0)
    & (train_df["fare_amount"] == 0)
)
train_df = train_df.loc[~mask_scenario_dup].copy()



## === cell 54
train_df.shape



## === cell 55
train_df = train_df



## === cell 56
train_df.shape



## === cell 57
high_distance = train_df.loc[
    (train_df["S_Distance"] > 200)
    & (train_df["fare_amount"] > 0)
    & (train_df["fare_amount"] < 50)
]



## === cell 58
high_distance.head(3)



## === cell 59
high_distance.shape



## === cell 60
train_df = train_df.loc[
    ~(
        (train_df["S_Distance"] > 200)
        & (train_df["fare_amount"] > 0)
        & (train_df["fare_amount"] < 50)
    )
].copy()



## === cell 61
train_df.head(3)



## === cell 62
train_df.loc[train_df["S_Distance"] == 0].head(3)



## === cell 63
train_df[(train_df["S_Distance"] == 0) & (train_df["fare_amount"] == 0)].head(3)



## === cell 64
train_df = train_df.loc[
    ~((train_df["S_Distance"] == 0) & (train_df["fare_amount"] == 0))
].copy()



## === cell 65
rush_hour = train_df.loc[
    (
        ((train_df["hour"] >= 6) & (train_df["hour"] <= 20))
        & ((train_df["day_of_week"] >= 1) & (train_df["day_of_week"] <= 5))
        & (train_df["S_Distance"] == 0)
        & (train_df["fare_amount"] < 2.5)
    )
]
rush_hour.head(3)



## === cell 66
mask_rush = (
    (train_df["hour"].between(6, 20))
    & (train_df["day_of_week"].between(1, 5))
    & (train_df["S_Distance"] == 0)
    & (train_df["fare_amount"] < 2.5)
)
train_df = train_df.loc[~mask_rush].copy()



## === cell 67
train_df.shape



## === cell 68
non_rush_hour = train_df.loc[
    (
        ((train_df["hour"] < 6) | (train_df["hour"] > 20))
        & ((train_df["day_of_week"] >= 1) & (train_df["day_of_week"] <= 5))
        & (train_df["S_Distance"] == 0)
        & (train_df["fare_amount"] < 3.0)
    )
]
non_rush_hour.head(3)



## === cell 69
non_rush_hour.shape



## === cell 70
non_rush_hour = train_df.loc[
    (
        ((train_df["hour"] < 6) | (train_df["hour"] > 20))
        & ((train_df["day_of_week"] >= 1) & (train_df["day_of_week"] <= 5))
        & (train_df["S_Distance"] == 0)
        & (train_df["fare_amount"] < 3.0)
    )
]
non_rush_hour.head(3)



## === cell 71
mask_non_rush = (
    (~train_df["hour"].between(6, 20))
    & (train_df["day_of_week"].between(1, 5))
    & (train_df["S_Distance"] == 0)
    & (train_df["fare_amount"] < 3.0)
)
train_df = train_df.loc[~mask_non_rush].copy()



## === cell 72
train_df.loc[(train_df["S_Distance"] != 0) & (train_df["fare_amount"] == 0)].head(3)



## === cell 73
scenario_3 = train_df.loc[
    (train_df["S_Distance"] != 0) & (train_df["fare_amount"] == 0)
]
scenario_3.head(3)



## === cell 74
scenario_3 = train_df.loc[
    (train_df["S_Distance"] != 0) & (train_df["fare_amount"] == 0)
]



## === cell 75
scenario_3 = scenario_3.copy()
scenario_3["fare_amount"] = (scenario_3["S_Distance"] * 1.56) + 2.50



## === cell 76
scenario_3["fare_amount"].head(3)



## === cell 77
train_df.loc[(train_df["S_Distance"] == 0) & (train_df["fare_amount"] != 0)].head(3)



## === cell 78
scenario_4 = train_df.loc[
    (train_df["S_Distance"] == 0) & (train_df["fare_amount"] != 0)
]



## === cell 79
scenario_4.head(3)



## === cell 80
len(scenario_3)



## === cell 81
len(scenario_4)



## === cell 82
scenario_4.loc[
    (scenario_4["fare_amount"] <= 3.0) & (scenario_4["S_Distance"] == 0)
].head(3)



## === cell 83
scenario_4.loc[
    (scenario_4["fare_amount"] > 3.0) & (scenario_4["S_Distance"] == 0)
].head(3)



## === cell 84
scenario_4_sub = scenario_4.loc[
    (scenario_4["fare_amount"] > 3.0) & (scenario_4["S_Distance"] == 0)
]



## === cell 85
len(scenario_4_sub)



## === cell 86
train_df = train_df.loc[
    ~((train_df["S_Distance"] == 0) & (train_df["fare_amount"] > 3.0))
].copy()



## === cell 87
len(train_df)



## === cell 88
train_df.columns



## === cell 89
test_df.columns



## === cell 90
bad_drop_lat = (train_df["dropoff_latitude"] < -90) | (
    train_df["dropoff_latitude"] > 90
)
bad_drop_lon = (train_df["dropoff_longitude"] < -180) | (
    train_df["dropoff_longitude"] > 180
)
coord_mask_train = (
    train_df["pickup_latitude"].between(-90, 90)
    & train_df["dropoff_latitude"].between(-90, 90)
    & train_df["pickup_longitude"].between(-180, 180)
    & train_df["dropoff_longitude"].between(-180, 180)
)
train_df = train_df.loc[~(bad_drop_lat | bad_drop_lon) & coord_mask_train].copy()



## === cell 91
not_null_island = ~(
    (train_df["pickup_latitude"].abs() < 0.01)
    & (train_df["pickup_longitude"].abs() < 0.01)
) & ~(
    (train_df["dropoff_latitude"].abs() < 0.01)
    & (train_df["dropoff_longitude"].abs() < 0.01)
)
train_df = train_df.loc[not_null_island].copy()



## === cell 92
train_df = train_df.loc[train_df["fare_amount"].between(2.5, 250)].copy()

same_point = (train_df["pickup_latitude"] == train_df["dropoff_latitude"]) & (
    train_df["pickup_longitude"] == train_df["dropoff_longitude"]
)
train_df = train_df.loc[(train_df["S_Distance"] > 0) | same_point].copy()



## === cell 93
nyc_mask = (
    train_df["pickup_latitude"].between(40.5, 41.0)
    & train_df["dropoff_latitude"].between(40.5, 41.0)
    & train_df["pickup_longitude"].between(-74.5, -73.5)
    & train_df["dropoff_longitude"].between(-74.5, -73.5)
)
train_df = train_df.loc[nyc_mask].copy()



## === cell 94
min_reasonable_fare = 2.5 + 0.5 * train_df["S_Distance"]
train_df = train_df.loc[
    (train_df["fare_amount"] >= min_reasonable_fare) | (train_df["S_Distance"] == 0)
].copy()



## === cell 95
ymin = int(train_df["year"].min())
year = train_df["year"].to_numpy(dtype=np.int32)
month = train_df["month"].to_numpy(dtype=np.int32)
date = train_df["date"].to_numpy(dtype=np.int32)
hour = train_df["hour"].to_numpy(dtype=np.int32)
train_df["time_bucket"] = (year - ymin) * 8760 + month * 744 + date * 24 + hour

train_df = train_df.loc[train_df["S_Distance"].between(0, 60)].copy()



## === cell 96
bucket_counts = train_df["time_bucket"].value_counts()
valid_buckets = bucket_counts.index[bucket_counts >= 50]

bucket_med = train_df.groupby("time_bucket")["S_Distance"].median()
tb = train_df["time_bucket"]
med_re = bucket_med.reindex(tb).to_numpy()
valid_mask = tb.isin(valid_buckets).to_numpy()
sd = train_df["S_Distance"].to_numpy()
train_df = train_df.loc[(~valid_mask) | (sd <= (med_re + 25.0))].copy()



## === cell 97
train_df["S_Distance"] = train_df["S_Distance"].to_numpy()




## === cell 98
def add_manhattan_km(df):
    lat1 = np.radians(df["pickup_latitude"].to_numpy())
    lat2 = np.radians(df["dropoff_latitude"].to_numpy())
    dlat = np.abs(lat2 - lat1)
    dlon = np.abs(
        np.radians((df["dropoff_longitude"] - df["pickup_longitude"]).to_numpy())
    )
    Rloc = 6367.0
    return Rloc * (dlat + np.cos((lat1 + lat2) / 2.0) * dlon)


train_df["M_Distance"] = add_manhattan_km(train_df)



## === cell 99
for col, lo, hi in [
    ("pickup_latitude", -90, 90),
    ("dropoff_latitude", -90, 90),
    ("pickup_longitude", -180, 180),
    ("dropoff_longitude", -180, 180),
]:
    test_df.loc[~test_df[col].between(lo, hi), col] = np.nan



## === cell 100
test_df["S_Distance"] = _haversine_km(
    test_df,
    "pickup_latitude",
    "pickup_longitude",
    "dropoff_latitude",
    "dropoff_longitude",
)



## === cell 101
test_df["M_Distance"] = add_manhattan_km(test_df)



## === cell 102
test_key = test_df["key"].copy()

if "time_bucket" in train_df.columns:
    train_df = train_df.drop(["time_bucket"], axis=1)

train_df = train_df.drop(["key", "pickup_datetime"], axis=1)
test_df = test_df.drop(["key", "pickup_datetime"], axis=1)



## === cell 103
train_df.columns



## === cell 104
test_df.columns



## === cell 105
x_train = train_df.iloc[:, train_df.columns != "fare_amount"]
y_train = train_df["fare_amount"].values
x_test = test_df



## === cell 106
x_train.shape



## === cell 107
y_train.shape



## === cell 108
x_train = x_train.replace([np.inf, -np.inf], np.nan)
x_test = x_test.replace([np.inf, -np.inf], np.nan)

x_test = x_test.reindex(columns=x_train.columns)

med = x_train.median(numeric_only=True)
x_train = x_train.fillna(med)
x_test = x_test.fillna(med)

rg = RandomForestRegressor(
    random_state=42,
    n_jobs=-1,
    n_estimators=300,
    min_samples_leaf=2,
    min_samples_split=4,
)
rg.fit(x_train, y_train)
y_predict = rg.predict(x_test)

y_predict = np.clip(y_predict, 0.0, None)

y_predict



## === cell 109
submission = pd.DataFrame({"key": test_key.values, "fare_amount": y_predict})
submission.to_csv("submission_1.csv", index=False)
submission.head(10)
