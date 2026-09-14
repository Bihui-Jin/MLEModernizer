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

3.8

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

3.09737

# 6. Current score

None

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 3.95899) has done: 'The crash happens because the model was trained on `df_train` which includes the engineered `distance` feature, but `df_test` does not have that same engineered feature (it has `direction` instead, coming from `train_df`/`df_test` mismatch), so prediction sees 12 columns vs 11 during training. The minimal safe fix is to align test features to exactly the training feature columns and order right before calling `model.predict()`. This preserves the same model, training procedure, and semantics while ensuring LightGBM gets identical feature schema. The patch in cell 37 simply reindexes `df_test` to `x_train.columns`, which also deterministically drops any extra columns (like `direction`) and adds any missing ones as NaN (none expected here beyond the bug).'
- What this solution (achieved 4.01807) has done: 'You’re currently worse than the target (RMSE 3.95899 vs 3.09737, lower is better), so we make a small, safe improvement that doesn’t change the model type or feature engineering approach. The biggest issue is that `direction` is engineered on `train_df` (the raw pre-features dataframe) but the model is trained on `df_train`, so the model never sees `direction` while the test set does—this wastes a potentially useful feature and can hurt RMSE. I compute `direction` for the actual training dataframe (`df_train`) as well (keeping everything else the same), and also fix the “radians apply” cells to actually assign results (they currently do nothing). Finally, we keep the existing column alignment safeguard before prediction and still write a valid `taxi_fare_submission.csv`.'
- What this solution (achieved 11.95564) has done: 'Your score (4.01807 RMSE) is worse than the target (3.09737), so we make the smallest changes that typically yield a sizable, legitimate RMSE drop without changing the model type or training loop. The biggest issue is that the model is trained on just 2M rows but uses very weak/buggy geospatial preprocessing: converting coordinates to radians after engineering distance makes features inconsistent between train/test, and the current “manhattan distance” approximation can produce instabilities (division by zero) and ignores the well-known NYC outlier cleanup that strongly improves RMSE. I (1) compute a stable haversine distance (still a single engineered distance feature—same core idea), (2) add minimal, standard NYC filtering (bounding box + passenger_count + fare caps) while keeping your 2M-row limit, and (3) ensure the exact same feature schema/order is used for train and test before predicting and writing the submission CSV.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os

print(os.listdir("../input"))



## === cell 1
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("dark_background")
sns.set_style("darkgrid")



## === cell 2
train_path = "../input/train.csv"

traintypes = {
    "key": "str",
    "fare_amount": "float32",
    "pickup_datetime": "str",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "uint8",
}

cols = list(traintypes.keys())
train_df = pd.read_csv(train_path, usecols=cols, dtype=traintypes, nrows=2_000_000)



## === cell 3
train_df.to_feather("nyc_taxi_data_raw.feather")



## === cell 4
df_train = pd.read_feather("nyc_taxi_data_raw.feather")



## === cell 5
df_train.dtypes



## === cell 6
df_train.describe()



## === cell 7
len(df_train[df_train.fare_amount > 0])



## === cell 8
df_train = df_train[df_train.fare_amount > 0]



## === cell 9
sns.histplot(df_train[df_train.fare_amount < 100].fare_amount, bins=50, kde=False)



## === cell 10
df_train.isnull().sum()



## === cell 11
df_train = df_train.dropna(how="any", axis="rows")



## === cell 12
df_test = pd.read_csv("../input/test.csv")
df_test.head(5)



## === cell 13
df_test.describe()



## === cell 14
df_train["pickup_datetime"] = pd.to_datetime(
    df_train["pickup_datetime"], format="%Y-%m-%d %H:%M:%S UTC"
)



## === cell 15
df_train["pickup_datetime"]



## === cell 16
df_test["pickup_datetime"] = pd.to_datetime(
    df_test["pickup_datetime"], format="%Y-%m-%d %H:%M:%S UTC"
)




## === cell 17
def add_new_date_time_features(dataset):
    dataset["hour"] = dataset.pickup_datetime.dt.hour
    dataset["day"] = dataset.pickup_datetime.dt.day
    dataset["month"] = dataset.pickup_datetime.dt.month
    dataset["year"] = dataset.pickup_datetime.dt.year
    dataset["day_of_week"] = dataset.pickup_datetime.dt.dayofweek
    return dataset




## === cell 18
df_train = add_new_date_time_features(df_train)
df_test = add_new_date_time_features(df_test)



## === cell 19
df_train.describe()



## === cell 20
NYC_BOUNDS = {
    "lon_min": -74.3,
    "lon_max": -73.7,
    "lat_min": 40.5,
    "lat_max": 41.0,
}

df_train = df_train[
    (df_train["pickup_longitude"].between(NYC_BOUNDS["lon_min"], NYC_BOUNDS["lon_max"]))
    & (
        df_train["dropoff_longitude"].between(
            NYC_BOUNDS["lon_min"], NYC_BOUNDS["lon_max"]
        )
    )
    & (
        df_train["pickup_latitude"].between(
            NYC_BOUNDS["lat_min"], NYC_BOUNDS["lat_max"]
        )
    )
    & (
        df_train["dropoff_latitude"].between(
            NYC_BOUNDS["lat_min"], NYC_BOUNDS["lat_max"]
        )
    )
]
df_train = df_train[
    (df_train["passenger_count"] >= 1) & (df_train["passenger_count"] <= 6)
]
df_train = df_train[(df_train["fare_amount"] >= 2.5) & (df_train["fare_amount"] <= 250)]

df_test = df_test[
    (df_test["pickup_longitude"].between(NYC_BOUNDS["lon_min"], NYC_BOUNDS["lon_max"]))
    & (
        df_test["dropoff_longitude"].between(
            NYC_BOUNDS["lon_min"], NYC_BOUNDS["lon_max"]
        )
    )
    & (df_test["pickup_latitude"].between(NYC_BOUNDS["lat_min"], NYC_BOUNDS["lat_max"]))
    & (
        df_test["dropoff_latitude"].between(
            NYC_BOUNDS["lat_min"], NYC_BOUNDS["lat_max"]
        )
    )
]
df_test = df_test[(df_test["passenger_count"] >= 1) & (df_test["passenger_count"] <= 6)]



## === cell 21
df_train.shape




## === cell 22
def haversine_np(lon1, lat1, lon2, lat2):
    lon1 = np.radians(lon1.astype("float64"))
    lat1 = np.radians(lat1.astype("float64"))
    lon2 = np.radians(lon2.astype("float64"))
    lat2 = np.radians(lat2.astype("float64"))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6371.0 * c
    return km.astype("float32")


df_train["distance"] = haversine_np(
    df_train["pickup_longitude"].values,
    df_train["pickup_latitude"].values,
    df_train["dropoff_longitude"].values,
    df_train["dropoff_latitude"].values,
)
df_test["distance"] = haversine_np(
    df_test["pickup_longitude"].values,
    df_test["pickup_latitude"].values,
    df_test["dropoff_longitude"].values,
    df_test["dropoff_latitude"].values,
)



## === cell 23
df_train.head()




## === cell 24
def calculate_direction(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon):
    """
    Return bearing-like direction value (as in the original notebook).
    """
    pickup_lat, pickup_lon, dropoff_lat, dropoff_lon = map(
        np.radians, [pickup_lat, pickup_lon, dropoff_lat, dropoff_lon]
    )
    dlon = pickup_lon - dropoff_lon
    a = np.arctan2(
        np.sin(dlon) * np.cos(dropoff_lat),
        np.cos(pickup_lat) * np.sin(dropoff_lat)
        - np.sin(pickup_lat) * np.cos(dropoff_lat) * np.cos(dlon),
    )
    return a.astype("float32")




## === cell 25
df_train["direction"] = calculate_direction(
    df_train["pickup_latitude"].values,
    df_train["pickup_longitude"].values,
    df_train["dropoff_latitude"].values,
    df_train["dropoff_longitude"].values,
)
df_test["direction"] = calculate_direction(
    df_test["pickup_latitude"].values,
    df_test["pickup_longitude"].values,
    df_test["dropoff_latitude"].values,
    df_test["dropoff_longitude"].values,
)



## === cell 26
sns.jointplot(
    x="distance",
    y="fare_amount",
    data=df_train.sample(n=min(50000, len(df_train)), random_state=123),
)



## === cell 27
if "key" in df_train.columns:
    df_train.drop(columns=["key"], inplace=True)

df_train.drop(columns=["pickup_datetime"], inplace=True)

y = df_train["fare_amount"]
df_train = df_train.drop(columns=["fare_amount"])



## === cell 28
df_train.head()



## === cell 29
from sklearn.model_selection import train_test_split
import lightgbm as lgbm

x_train, x_valid, y_train, y_valid = train_test_split(
    df_train, y, random_state=123, test_size=0.1
)



## === cell 30
params = {
    "boosting_type": "gbdt",
    "objective": "regression",
    "nthread": 4,
    "num_leaves": 31,
    "learning_rate": 0.1,
    "max_depth": -1,
    "subsample": 0.8,
    "bagging_fraction": 1,
    "max_bin": 10000,
    "bagging_freq": 10,
    "metric": "rmse",
    "zero_as_missing": True,
    "num_rounds": 50000,
}



## === cell 31
train_set = lgbm.Dataset(
    x_train, y_train, categorical_feature=["year", "month", "day", "day_of_week"]
)
valid_set = lgbm.Dataset(
    x_valid, y_valid, categorical_feature=["year", "month", "day", "day_of_week"]
)

model = lgbm.train(
    params,
    train_set=train_set,
    num_boost_round=10000,
    valid_sets=[valid_set],
    callbacks=[
        lgbm.early_stopping(stopping_rounds=1000),
        lgbm.log_evaluation(period=500),
    ],
)



## === cell 32
df_train.describe()



## === cell 33
test_key = df_test["key"]
df_test.drop(columns=["pickup_datetime", "key"], axis=1, inplace=True)



## === cell 34
df_test = df_test.reindex(columns=x_train.columns)

prediction = model.predict(df_test, num_iteration=model.best_iteration)
prediction = np.maximum(prediction, 0.0)

submission = pd.DataFrame({"key": test_key, "fare_amount": prediction})
submission.to_csv("taxi_fare_submission.csv", index=False)
print(submission.head())
print("Wrote taxi_fare_submission.csv with shape:", submission.shape)
