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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")
np.random.seed(42)



## === cell 1
usecols = [
    "key",
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
dtype = {
    "key": "object",
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int8",
}
train_df = pd.read_csv(
    "../input/new-york-city-taxi-fare-prediction/train.csv",
    usecols=usecols,
    dtype=dtype,
    low_memory=False,
)
test_df = pd.read_csv(
    "../input/new-york-city-taxi-fare-prediction/test.csv",
    usecols=[c for c in usecols if c != "fare_amount"],
    dtype={k: v for k, v in dtype.items() if k != "fare_amount"},
    low_memory=False,
)



## === cell 2
train_df.isnull().sum()



## === cell 3
train_df.dropna(
    axis=0,
    subset=[
        "dropoff_longitude",
        "dropoff_latitude",
        "pickup_longitude",
        "pickup_latitude",
    ],
    inplace=True,
)
train_df.reset_index(drop=True, inplace=True)



## === cell 4
pd.set_option("display.float_format", lambda x: "%.5f" % x)
train_df.describe()



## === cell 5
print("Number of observations out of valid range in coordinate columns:")

print(
    "pickup_longitude:",
    (train_df.pickup_longitude < -180).sum() + (train_df.pickup_longitude > 180).sum(),
)
print(
    "pickup_latitude :",
    (train_df.pickup_latitude < -90).sum() + (train_df.pickup_latitude > 90).sum(),
)
print(
    "dropoff_longitude:",
    (train_df.dropoff_longitude < -180).sum()
    + (train_df.dropoff_longitude > 180).sum(),
)
print(
    "dropoff_latitude :",
    (train_df.dropoff_latitude < -90).sum() + (train_df.dropoff_latitude > 90).sum(),
)



## === cell 6
mask = (
    (train_df.pickup_longitude >= -180)
    & (train_df.pickup_longitude <= 180)
    & (train_df.pickup_latitude >= -90)
    & (train_df.pickup_latitude <= 90)
    & (train_df.dropoff_longitude >= -180)
    & (train_df.dropoff_longitude <= 180)
    & (train_df.dropoff_latitude >= -90)
    & (train_df.dropoff_latitude <= 90)
    & (train_df.pickup_longitude >= -75)
    & (train_df.pickup_longitude <= -72)
    & (train_df.dropoff_longitude >= -75)
    & (train_df.dropoff_longitude <= -72)
    & (train_df.pickup_latitude >= 40)
    & (train_df.pickup_latitude <= 42)
    & (train_df.dropoff_latitude >= 40)
    & (train_df.dropoff_latitude <= 42)
    & (train_df.passenger_count > 0)
    & (train_df.fare_amount > 0)
)
train_df = train_df.loc[mask].reset_index(drop=True)



## === cell 7
train_df.describe()



## === cell 8
idx = train_df[train_df.pickup_longitude >= 40].index
if not idx.empty:
    train_df.loc[idx, ["pickup_longitude", "pickup_latitude"]] = train_df.loc[
        idx, ["pickup_latitude", "pickup_longitude"]
    ].values
    train_df.loc[idx, ["dropoff_longitude", "dropoff_latitude"]] = train_df.loc[
        idx, ["dropoff_latitude", "dropoff_longitude"]
    ].values



## === cell 11
train_df.describe()



## === cell 12
train_df.passenger_count.value_counts()



## === cell 14
train_df.fare_amount.sort_values(ascending=False)



## === cell 16
test_df.isna().sum()



## === cell 17
test_df.describe()



## === cell 18
train_df.dtypes



## === cell 19
train_df["pickup_datetime"] = pd.to_datetime(train_df["pickup_datetime"])
test_df["pickup_datetime"] = pd.to_datetime(test_df["pickup_datetime"])




## === cell 20
def date_splitter(df):
    df["Year"] = df["pickup_datetime"].dt.year.astype(np.int16)
    df["Month"] = df["pickup_datetime"].dt.month.astype(np.int8)
    df["Day"] = df["pickup_datetime"].dt.day.astype(np.int8)
    df["Weekday"] = df["pickup_datetime"].dt.dayofweek.astype(np.int8)
    df["Hour"] = df["pickup_datetime"].dt.hour.astype(np.int8)


date_splitter(train_df)
date_splitter(test_df)

train_df.drop(columns=["pickup_datetime"], inplace=True)
test_df.drop(columns=["pickup_datetime"], inplace=True)




## === cell 21
def haversine_distance(df):
    phi1 = np.radians(df["pickup_latitude"].astype(np.float32))
    phi2 = np.radians(df["dropoff_latitude"].astype(np.float32))
    lambda1 = np.radians(df["pickup_longitude"].astype(np.float32))
    lambda2 = np.radians(df["dropoff_longitude"].astype(np.float32))
    R = 6371.0  # Earth radius in km
    dphi = phi2 - phi1
    dlambda = lambda2 - lambda1
    a = (
        np.sin(dphi / 2.0) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    df["Distance"] = R * c


haversine_distance(train_df)
haversine_distance(test_df)



## === cell 22
train_df.Distance.sort_values()



## === cell 23
train_df = train_df[train_df.Distance >= 0.5].reset_index(drop=True)



## === cell 24
sns.scatterplot(x="passenger_count", y="fare_amount", data=train_df)



## === cell 25
sns.scatterplot(x="Year", y="fare_amount", data=train_df)



## === cell 26
sns.scatterplot(x="Month", y="fare_amount", data=train_df)



## === cell 27
sns.scatterplot(x="Day", y="fare_amount", data=train_df)



## === cell 28
sns.scatterplot(x="Weekday", y="fare_amount", data=train_df)



## === cell 29
sns.scatterplot(x="Hour", y="fare_amount", data=train_df)



## === cell 30
sns.scatterplot(x="Distance", y="fare_amount", data=train_df)



## === cell 31
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
]



## === cell 32
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler



## === cell 33
X = train_df[features]
y = train_df["fare_amount"]
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.30, random_state=42
)



## === cell 34
scaler = StandardScaler()
scaled_train = scaler.fit_transform(X_train.astype(np.float32))
scaled_valid = scaler.transform(X_valid.astype(np.float32))
scaled_test = scaler.transform(test_df[features].astype(np.float32))



## === cell 35
model = XGBRegressor(
    n_estimators=1500,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    n_jobs=5,
    random_state=42,
    verbosity=0,
    tree_method="hist",
)



## === cell 36
model.fit(
    scaled_train,
    y_train,
    eval_set=[(scaled_valid, y_valid)],
    early_stopping_rounds=50,
    verbose=False,
)
val_pred = model.predict(scaled_valid)
rmse = mean_squared_error(y_valid, val_pred, squared=False)
print(f"Validation RMSE: {rmse:.5f}")



## === cell 37
prediction = model.predict(scaled_test)



## === cell 38
prediction = np.clip(prediction, 0, None)



## === cell 39
submission = pd.DataFrame({"key": test_df["key"], "fare_amount": prediction})
submission.to_csv("taxi_fare_submission.csv", index=False)
