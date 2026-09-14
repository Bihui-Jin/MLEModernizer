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
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import warnings

warnings.filterwarnings("ignore")

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))

dtype_dict = {
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int8",
}
parse_dates = ["pickup_datetime"]

train_iop_path = "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv"
test_iop_path = "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv"

dataset_train = pd.read_csv(
    train_iop_path,
    index_col="key",
    dtype=dtype_dict,
    parse_dates=parse_dates,
    infer_datetime_format=True,
    low_memory=False,
)
dataset_test = pd.read_csv(
    test_iop_path,
    index_col="key",
    dtype={k: v for k, v in dtype_dict.items() if k != "fare_amount"},
    parse_dates=parse_dates,
    infer_datetime_format=True,
    low_memory=False,
)



## === cell 1
print("dataset_train old size", len(dataset_train))
dataset_train = dataset_train[dataset_train.dropoff_longitude != 0]
print("new size", len(dataset_train))



## === cell 2
print("dataset_test old size", len(dataset_test))
dataset_test = dataset_test[dataset_test.dropoff_longitude != 0]
print("new size", len(dataset_test))




## === cell 3
def get_year(pickup_date):
    return pickup_date.year


def get_month(pickup_date):
    return pickup_date.month


def get_day(pickup_date):
    return pickup_date.day


def get_hour(pickup_date):
    return pickup_date.hour




## === cell 4
def preparedataset2(datasetname, apply_bbox=True):
    """
    Clean and engineer features.
    If apply_bbox is True (default), rows outside a reasonable NYC bounding box are dropped.
    For the test set we set apply_bbox=False to keep every row so the submission matches the required size.
    """

    datasetname["pickup_year"] = datasetname["pickup_datetime"].dt.year.astype(np.int16)
    datasetname["pickup_month"] = datasetname["pickup_datetime"].dt.month.astype(
        np.int8
    )
    datasetname["pickup_day"] = datasetname["pickup_datetime"].dt.day.astype(np.int8)
    datasetname["pickup_hour"] = datasetname["pickup_datetime"].dt.hour.astype(np.int8)

    datasetname["x_dis"] = (
        datasetname["dropoff_longitude"] - datasetname["pickup_longitude"]
    )
    datasetname["y_dis"] = (
        datasetname["dropoff_latitude"] - datasetname["pickup_latitude"]
    )
    datasetname["dis"] = np.sqrt(datasetname["x_dis"] ** 2 + datasetname["y_dis"] ** 2)
    datasetname["manhattan_dis"] = (
        datasetname["x_dis"].abs() + datasetname["y_dis"].abs()
    )

    if apply_bbox:
        bbox_mask = (
            (datasetname["pickup_longitude"].between(-75, -73))
            & (datasetname["dropoff_longitude"].between(-75, -73))
            & (datasetname["pickup_latitude"].between(40, 42))
            & (datasetname["dropoff_latitude"].between(40, 42))
        )
        datasetname = datasetname[bbox_mask]

    R = 6371.0  # Earth radius in km
    lat1 = np.radians(datasetname["pickup_latitude"])
    lat2 = np.radians(datasetname["dropoff_latitude"])
    dlat = lat2 - lat1
    dlon = np.radians(
        datasetname["dropoff_longitude"] - datasetname["pickup_longitude"]
    )
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    haversine_km = R * 2 * np.arcsin(np.sqrt(a))
    datasetname["haversine_km"] = haversine_km

    datasetname = datasetname.drop(
        [
            "pickup_datetime",
            "pickup_longitude",
            "dropoff_latitude",
            "dropoff_longitude",
            "pickup_latitude",
        ],
        axis=1,
    )
    if "fare_amount" in datasetname.columns:
        datasetname = datasetname[datasetname["fare_amount"] > 0]

    return datasetname




## === cell 5
df = preparedataset2(dataset_train, apply_bbox=True)
df = df.fillna(0)  # safeguard against any missing values



## === cell 6
test_df = preparedataset2(dataset_test, apply_bbox=False)
test_df = test_df.fillna(0)  # safeguard against any missing values



## === cell 7
y = np.log1p(df["fare_amount"])
X = df.drop("fare_amount", axis=1)

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.25, random_state=42
)

X_train_sub = X_train.sample(frac=0.05, random_state=42)
y_train_sub = y_train.loc[X_train_sub.index]

best_n_estimators = 0
best_learning_rate = 0
best_rmse = float("inf")
result = {}

learning_rates = [0.03, 0.05, 0.07] + [x / 100 for x in range(10, 50, 5)]
n_estimators_range = list(range(400, 2001, 200))

for lr in learning_rates:
    for ns in n_estimators_range:
        model = XGBRegressor(
            n_estimators=ns,
            learning_rate=lr,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=4,
            objective="reg:squarederror",
            verbosity=0,
            random_state=42,
            tree_method="hist",
        )
        model.fit(X_train_sub, y_train_sub)

        preds_log = model.predict(X_valid)
        preds = np.expm1(preds_log)
        y_valid_original = np.expm1(y_valid)

        rmse = mean_squared_error(y_valid_original, preds, squared=False)

        result[(ns, lr)] = rmse
        if rmse < best_rmse:
            best_rmse = rmse
            best_n_estimators = ns
            best_learning_rate = lr
            print("Better hyper‑params found:", ns, lr, rmse)

print("Best n_estimators:", best_n_estimators)
print("Best learning_rate:", best_learning_rate)
print("Best validation RMSE:", best_rmse)



## === cell 8
final_model = XGBRegressor(
    n_estimators=best_n_estimators,
    learning_rate=best_learning_rate,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    n_jobs=4,
    objective="reg:squarederror",
    verbosity=0,
    random_state=42,
    tree_method="hist",
)
final_model.fit(X_train_sub, y_train_sub)

test_predictions_log = final_model.predict(test_df)
test_predictions = np.expm1(test_predictions_log)

test_predictions = np.maximum(test_predictions, 0)

submission = pd.DataFrame({"key": test_df.index, "fare_amount": test_predictions})
submission.to_csv("submission.csv", index=False)
print("Submission written to submission.csv")
