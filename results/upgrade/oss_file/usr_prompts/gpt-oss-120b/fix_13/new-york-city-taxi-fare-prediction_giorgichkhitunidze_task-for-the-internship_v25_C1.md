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
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1
xgboost==2.0.3

# 4. Data file paths

```
/
    kaggle/
        data/
            new-york-city-taxi-fare-prediction/
                GCP-Coupons-Instructions.rtf (486 Bytes)
                description.md (100 lines)
                ... and 3 other files
        input/
            new-york-city-taxi-fare-prediction/
                GCP-Coupons-Instructions.rtf (486 Bytes)
                description.md (100 lines)
                ... and 3 other files
        working/
            new-york-city-taxi-fare-prediction/
                GCP-Coupons-Instructions.rtf (486 Bytes)
                description.md (100 lines)
                ... and 3 other files
```

-> data/new-york-city-taxi-fare-prediction/labels.csv has 55413942 rows and 8 columns.
The columns are: key, fare_amount, pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count

-> data/new-york-city-taxi-fare-prediction/sample_submission.csv has 9914 rows and 2 columns.
The columns are: key, fare_amount

-> data/new-york-city-taxi-fare-prediction/test.csv has 9914 rows and 7 columns.
The columns are: key, pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count

-> input/new-york-city-taxi-fare-prediction/labels.csv has 55413942 rows and 8 columns.
The columns are: key, fare_amount, pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count

-> input/new-york-city-taxi-fare-prediction/sample_submission.csv has 9914 rows and 2 columns.
The columns are: key, fare_amount

-> input/new-york-city-taxi-fare-prediction/test.csv has 9914 rows and 7 columns.
The columns are: key, pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count

-> working/new-york-city-taxi-fare-prediction/labels.csv has 55413942 rows and 8 columns.
The columns are: key, fare_amount, pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

plt.rcParams["figure.figsize"] = (16, 8)

dtypes = {
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int16",
}
train_cols = [
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]

train_df = pd.read_csv(
    "../input/new-york-city-taxi-fare-prediction/labels.csv",
    usecols=train_cols,
    dtype=dtypes,
    nrows=5_000_000,
)
test_df = pd.read_csv(
    "../input/new-york-city-taxi-fare-prediction/test.csv", dtype=dtypes
)



## === cell 1
train_df.isnull().sum()



## === cell 2
train_df.dropna(
    subset=[
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
    ],
    inplace=True,
)
train_df.reset_index(drop=True, inplace=True)



## === cell 3
pd.set_option("display.float_format", lambda x: "%.5f" % x)
train_df.describe()



## === cell 4
valid_geo = (
    train_df["pickup_longitude"].between(-180, 180)
    & train_df["pickup_latitude"].between(-90, 90)
    & train_df["dropoff_longitude"].between(-180, 180)
    & train_df["dropoff_latitude"].between(-90, 90)
)
train_df = train_df.loc[valid_geo].reset_index(drop=True)



## === cell 5
swap_idx = train_df["pickup_longitude"] >= 40
train_df.loc[swap_idx, ["dropoff_longitude", "dropoff_latitude"]] = train_df.loc[
    swap_idx, ["dropoff_latitude", "dropoff_longitude"]
].values
train_df.loc[swap_idx, ["pickup_longitude", "pickup_latitude"]] = train_df.loc[
    swap_idx, ["pickup_latitude", "pickup_longitude"]
].values



## === cell 6
nyc_mask = (
    train_df["pickup_longitude"].between(-75, -72)
    & train_df["dropoff_longitude"].between(-75, -72)
    & train_df["pickup_latitude"].between(40, 42)
    & train_df["dropoff_latitude"].between(40, 42)
)
train_df = train_df.loc[nyc_mask].reset_index(drop=True)



## === cell 7
train_df.describe()



## === cell 8
train_df = train_df[train_df["passenger_count"] > 0]
train_df = train_df[train_df["fare_amount"] > 0].reset_index(drop=True)



## === cell 9
test_df.isna().sum()



## === cell 10
test_df.describe()



## === cell 11
train_df.dtypes



## === cell 12
train_df["pickup_datetime"] = pd.to_datetime(train_df["pickup_datetime"])
test_df["pickup_datetime"] = pd.to_datetime(test_df["pickup_datetime"])




## === cell 13
def date_splitter(df):
    df["Year"] = df["pickup_datetime"].dt.year.astype("int16")
    df["Month"] = df["pickup_datetime"].dt.month.astype("int16")
    df["Day"] = df["pickup_datetime"].dt.day.astype("int16")
    df["Weekday"] = df["pickup_datetime"].dt.dayofweek.astype("int16")
    df["Hour"] = df["pickup_datetime"].dt.hour.astype("int16")


date_splitter(train_df)
date_splitter(test_df)
train_df.drop(columns=["pickup_datetime"], inplace=True)
test_df.drop(columns=["pickup_datetime"], inplace=True)




## === cell 14
def haversine_distance(df):
    rad = np.radians(
        df[
            [
                "pickup_latitude",
                "pickup_longitude",
                "dropoff_latitude",
                "dropoff_longitude",
            ]
        ].values.astype("float32")
    )
    phi1 = rad[:, 0]
    lambda1 = rad[:, 1]
    phi2 = rad[:, 2]
    lambda2 = rad[:, 3]
    R = 6371.0  # Earth radius in km
    dphi = phi2 - phi1
    dlambda = lambda2 - lambda1
    a = (
        np.sin(dphi / 2.0) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    df["Distance"] = (R * c).astype("float32")
    df["ManhattanDist"] = (
        np.abs(df["pickup_latitude"] - df["dropoff_latitude"])
        + np.abs(df["pickup_longitude"] - df["dropoff_longitude"])
    ).astype("float32")


haversine_distance(train_df)
haversine_distance(test_df)



## === cell 15
train_df["Dist_log"] = np.log1p(train_df["Distance"]).astype("float32")
test_df["Dist_log"] = np.log1p(test_df["Distance"]).astype("float32")

train_df["Hour_sin"] = np.sin(2 * np.pi * train_df["Hour"] / 24).astype("float32")
train_df["Hour_cos"] = np.cos(2 * np.pi * train_df["Hour"] / 24).astype("float32")
test_df["Hour_sin"] = np.sin(2 * np.pi * test_df["Hour"] / 24).astype("float32")
test_df["Hour_cos"] = np.cos(2 * np.pi * test_df["Hour"] / 24).astype("float32")

train_df["Dist_pass"] = (train_df["Distance"] * train_df["passenger_count"]).astype(
    "float32"
)
test_df["Dist_pass"] = (test_df["Distance"] * test_df["passenger_count"]).astype(
    "float32"
)

train_df["Is_weekend"] = (train_df["Weekday"] >= 5).astype("int8")
test_df["Is_weekend"] = (test_df["Weekday"] >= 5).astype("int8")

train_df["Is_night"] = ((train_df["Hour"] < 5) | (train_df["Hour"] > 22)).astype("int8")
test_df["Is_night"] = ((test_df["Hour"] < 5) | (test_df["Hour"] > 22)).astype("int8")



## === cell 16
train_df = train_df[train_df["Distance"] >= 0.5].reset_index(drop=True)



## === cell 17
features = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
    "Year",
    "Month",
    "Day",
    "Weekday",
    "Hour",
    "Distance",
    "ManhattanDist",
    "Dist_log",
    "Hour_sin",
    "Hour_cos",
    "Dist_pass",
    "Is_weekend",
    "Is_night",
]
outcome = "fare_amount"



## === cell 18
X_train, X_valid, y_train, y_valid = train_test_split(
    train_df[features],
    train_df[outcome],
    test_size=0.30,
    random_state=42,
)



## === cell 19
scaled_train = X_train.astype("float32").to_numpy()
scaled_valid = X_valid.astype("float32").to_numpy()
scaled_test = test_df[features].astype("float32").to_numpy()
y_train_np = y_train.astype("float32").to_numpy()
y_valid_np = y_valid.astype("float32").to_numpy()



## === cell 20
model = XGBRegressor(
    n_estimators=1200,  # allow more trees; early stopping will keep optimal count
    learning_rate=0.05,
    max_depth=12,  # a bit deeper to capture extra interactions
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    eval_metric="rmse",
    n_jobs=5,
    random_state=42,
    reg_lambda=1.0,
    tree_method="hist",
    max_bin=256,
)
model.fit(
    scaled_train,
    y_train_np,
    eval_set=[(scaled_valid, y_valid_np)],
    early_stopping_rounds=30,
    verbose=False,
)



## === cell 21
prediction = model.predict(scaled_test)
prediction = np.where(prediction < 0, 0, prediction)
submission = pd.DataFrame({"key": test_df["key"], "fare_amount": prediction})
submission.to_csv("taxi_fare_submission.csv", index=False)
print("Submission saved to taxi_fare_submission.csv")
