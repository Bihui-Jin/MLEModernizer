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

3.12

# 3. Installed packages

geopandas==0.14.4
geopy==2.4.1
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
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
tf_keras==2.18.0
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
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))




## === cell 1
train_df = pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv"
)  # removed nrows limit
test_df = pd.read_csv("/kaggle/input/new-york-city-taxi-fare-prediction/test.csv")




## === cell 2
print(train_df)
print(test_df)

missing_values = train_df.isnull()
ans = missing_values.sum()
print(ans)




## === cell 3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error




## === cell 4
print(train_df.isnull().sum())

train_df.dropna(inplace=True)

train_df = train_df[train_df["passenger_count"] < 8]
train_df = train_df[train_df["fare_amount"] > 0]

train_df = train_df[train_df["fare_amount"] < 200]

print(train_df.isnull().sum())




## === cell 5
min(test_df.pickup_longitude.min(), test_df.dropoff_longitude.min()), max(
    test_df.pickup_longitude.max(), test_df.dropoff_longitude.max()
)




## === cell 6
min(test_df.pickup_latitude.min(), test_df.dropoff_latitude.min()), max(
    test_df.pickup_latitude.max(), test_df.dropoff_latitude.max()
)




## === cell 7
RANGE = (-74.26, -72.99, 40.56, 41.71)


def select_within_boundingbox(df, RANGE):
    return (
        (df.pickup_longitude >= RANGE[0])
        & (df.pickup_longitude <= RANGE[1])
        & (df.pickup_latitude >= RANGE[2])
        & (df.pickup_latitude <= RANGE[3])
        & (df.dropoff_longitude >= RANGE[0])
        & (df.dropoff_longitude <= RANGE[1])
        & (df.dropoff_latitude >= RANGE[2])
        & (df.dropoff_latitude <= RANGE[3])
    )


print("Old size: %d" % len(train_df))
train_df = train_df[select_within_boundingbox(train_df, RANGE)]
print("New size: %d" % len(train_df))




## === cell 8
from geopy.distance import great_circle


def calculate_distance(df):
    pickup_coords = list(zip(df["pickup_latitude"], df["pickup_longitude"]))
    dropoff_coords = list(zip(df["dropoff_latitude"], df["dropoff_longitude"]))
    distances = [
        great_circle(pickup, dropoff).miles
        for pickup, dropoff in zip(pickup_coords, dropoff_coords)
    ]
    return distances


def add_distance_to_df(df):
    df["distance"] = calculate_distance(df)

    df["distance_log"] = np.log1p(df["distance"])

    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
    df["hour"] = df["pickup_datetime"].dt.hour
    df["day"] = df["pickup_datetime"].dt.day
    df["month"] = df["pickup_datetime"].dt.month
    df["year"] = df["pickup_datetime"].dt.year

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)


add_distance_to_df(train_df)
add_distance_to_df(test_df)




## === cell 9
import matplotlib.pyplot as plt


def plt_distance_to_fare(df, sample_size=100_000):
    df = df.sample(sample_size)
    plt.scatter(df["distance"], df["fare_amount"], s=1)
    plt.title("Distance and Fare Amount")
    plt.xlabel("Distance")
    plt.ylabel("Fare Amount")
    plt.show()


plt_distance_to_fare(train_df)




## === cell 10
nn_df = train_df.copy()

plt_distance_to_fare(train_df)




## === cell 11
features = [
    "passenger_count",
    "distance",
    "distance_log",  # added feature
    "hour",
    "hour_sin",
    "hour_cos",
    "day",
    "month",
    "pickup_latitude",
    "pickup_longitude",
    "dropoff_latitude",
    "dropoff_longitude",
]
X = train_df[features]
y = train_df["fare_amount"]




## === cell 12
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_valid)

rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
print("Linear Regression RMSE:", rmse)




## === cell 13
def plot_linear(X_valid, y_valid, y_pred, column="distance"):
    plt.scatter(X_valid[column], y_valid, color="blue", label="Data", s=2)
    plt.plot(
        X_valid[column], y_pred, color="red", linewidth=1, label="Linear Regression"
    )
    plt.title("Linear Regression Model")
    plt.xlabel("Distance")
    plt.ylabel("Fare Amount")
    plt.legend()
    plt.show()


plot_linear(X_valid, y_valid, y_pred)




## === cell 14
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

X = train_df[features]
y = train_df["fare_amount"]

y_log = np.log1p(y)

X_train, X_valid, y_train_log, y_valid_log = train_test_split(
    X, y_log, test_size=0.2, random_state=42
)

model = XGBRegressor(
    n_estimators=1200,  # more trees for stronger fit
    learning_rate=0.03,  # lower LR to keep training stable
    max_depth=10,  # deeper trees for richer interactions
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    n_jobs=4,
    random_state=42,
)

model.fit(X_train, y_train_log)

y_valid_pred_log = model.predict(X_valid)
y_valid_pred = np.expm1(y_valid_pred_log)
y_valid_pred = np.clip(y_valid_pred, 0, 200)

rmse = np.sqrt(mean_squared_error(np.expm1(y_valid_log), y_valid_pred))
print("XGBoost RMSE (log‑target, original scale):", rmse)




## === cell 15
X_test = test_df[features]

y_test_pred_log = model.predict(X_test)
y_test_pred = np.expm1(y_test_pred_log)
y_test_pred = np.clip(y_test_pred, 0, 200)

submission_df = test_df[["key"]].copy()
submission_df["fare_amount"] = y_test_pred
submission_df.to_csv("submission.csv", index=False)
