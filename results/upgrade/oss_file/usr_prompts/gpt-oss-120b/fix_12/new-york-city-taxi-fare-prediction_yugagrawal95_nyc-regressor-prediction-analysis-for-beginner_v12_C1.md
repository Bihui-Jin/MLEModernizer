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




## === cell 1
train_cols = [
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
test_cols = [
    "key",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]

dtype_dict = {
    "passenger_count": "uint8",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "fare_amount": "float32",
}

train_data = pd.read_csv(
    "../input/train.csv",
    nrows=1_000_000,
    usecols=train_cols,
    dtype=dtype_dict,
    parse_dates=["pickup_datetime"],
    infer_datetime_format=True,
)

test_data = pd.read_csv(
    "../input/test.csv",
    usecols=test_cols,
    dtype=dtype_dict,
    parse_dates=["pickup_datetime"],
    infer_datetime_format=True,
)

train_data.head()
test_data.head()




## === cell 2
train_data.describe()




## === cell 3
train_data.shape




## === cell 4
train_data.info()
test_data.info()




## === cell 5
train_data["pickup_datetime"].head()




## === cell 6
train_data.isnull().sum()
train_data = train_data.dropna(axis=0)
train_data.isnull().sum()




## === cell 7
pd.set_option("float_format", "{:f}".format)
train_data.describe()




## === cell 8
train_data = train_data.loc[train_data["fare_amount"] > 0]




## === cell 9
train_data = train_data.loc[train_data["fare_amount"] < 400]




## === cell 10
train_data = train_data[train_data["passenger_count"] <= 6]




## === cell 11
train_data = train_data[
    (train_data["pickup_latitude"] >= -90)
    & (train_data["pickup_latitude"] <= 90)
    & (train_data["pickup_longitude"] >= -180)
    & (train_data["pickup_longitude"] <= 180)
    & (train_data["dropoff_latitude"] >= -90)
    & (train_data["dropoff_latitude"] <= 90)
    & (train_data["dropoff_longitude"] >= -180)
    & (train_data["dropoff_longitude"] <= 180)
]




## === cell 12
def degree_to_radian(degree):
    return degree * (np.pi / 180)


def calculate_distance(
    pickup_latitude, pickup_longitude, dropoff_latitude, dropoff_longitude
):
    from_lat = degree_to_radian(pickup_latitude)
    from_long = degree_to_radian(pickup_longitude)
    to_lat = degree_to_radian(dropoff_latitude)
    to_long = degree_to_radian(dropoff_longitude)
    radius = 6371.01  # Earth radius in km
    lat_diff = to_lat - from_lat
    long_diff = to_long - from_long
    a = (
        np.sin(lat_diff / 2) ** 2
        + np.cos(from_lat) * np.cos(to_lat) * np.sin(long_diff / 2) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return radius * c




## === cell 13
train_data["distance"] = calculate_distance(
    train_data["pickup_latitude"],
    train_data["pickup_longitude"],
    train_data["dropoff_latitude"],
    train_data["dropoff_longitude"],
)




## === cell 14
test_data["distance"] = calculate_distance(
    test_data["pickup_latitude"],
    test_data["pickup_longitude"],
    test_data["dropoff_latitude"],
    test_data["dropoff_longitude"],
)




## === cell 15
train_data = train_data.loc[train_data["distance"] < 200]




## === cell 16
test_data_key = test_data["key"]
train_data = train_data.drop(columns="key", errors="ignore")
test_data = test_data.drop(columns="key")




## === cell 17
for df in (train_data, test_data):
    df["Year"] = df["pickup_datetime"].dt.year
    df["Month"] = df["pickup_datetime"].dt.month
    df["Date"] = df["pickup_datetime"].dt.day
    df["Day_of_Week"] = df["pickup_datetime"].dt.dayofweek
    df["Hour"] = df["pickup_datetime"].dt.hour




## === cell 18
train_data = train_data.drop(columns="pickup_datetime")
test_data = test_data.drop(columns="pickup_datetime")




## === cell 19
X = train_data.drop(columns="fare_amount")
y = train_data["fare_amount"]

X_np = X.values.astype(np.float32)
y_np = y.values.astype(np.float32)




## === cell 20
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X_train, X_valid, y_train, y_valid = train_test_split(
    X_np, y_np, test_size=0.3, random_state=0
)




## === cell 21
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
import lightgbm as lgb
from joblib import Parallel, delayed

gradient_reg = GradientBoostingRegressor(
    n_estimators=800, learning_rate=0.05, max_depth=3, random_state=0
)
xgreg = XGBRegressor(
    n_estimators=800,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    n_jobs=-1,
    tree_method="hist",
    predictor="cpu_predictor",
    random_state=0,
    verbosity=0,
)
model_lgb = lgb.LGBMRegressor(
    n_estimators=800,
    learning_rate=0.05,
    num_leaves=63,
    objective="regression",
    n_jobs=-1,
    random_state=0,
    verbose=-1,
)


def fit_model(model, X, y):
    model.fit(X, y)
    return model


gradient_reg, xgreg, model_lgb = Parallel(n_jobs=3, backend="threading")(
    delayed(fit_model)(m, X_train, y_train) for m in (gradient_reg, xgreg, model_lgb)
)




## === cell 22
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


pred_gb = gradient_reg.predict(X_valid)
pred_xgb = xgreg.predict(X_valid)
pred_lgb = model_lgb.predict(X_valid)

rmse_gb = rmse(y_valid, pred_gb)
rmse_xgb = rmse(y_valid, pred_xgb)
rmse_lgb = rmse(y_valid, pred_lgb)

inv_errors = np.array([1 / rmse_gb, 1 / rmse_xgb, 1 / rmse_lgb])
weights = inv_errors / inv_errors.sum()

val_preds = weights[0] * pred_gb + weights[1] * pred_xgb + weights[2] * pred_lgb
validation_rmse = rmse(y_valid, val_preds)
validation_rmse




## === cell 23
test_np = test_data.values.astype(np.float32)

test_pred_gb = gradient_reg.predict(test_np)
test_pred_xgb = xgreg.predict(test_np)
test_pred_lgb = model_lgb.predict(test_np)

test_preds = (
    weights[0] * test_pred_gb + weights[1] * test_pred_xgb + weights[2] * test_pred_lgb
)




## === cell 24
submission = pd.DataFrame({"key": test_data_key, "fare_amount": test_preds})
submission.to_csv("submission.csv", index=False)
print("Submission saved, rows:", submission.shape[0])
