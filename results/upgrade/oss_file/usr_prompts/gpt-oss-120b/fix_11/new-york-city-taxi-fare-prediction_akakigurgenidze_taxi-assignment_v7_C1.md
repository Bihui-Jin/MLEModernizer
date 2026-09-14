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
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import xgboost as xgb

usecols_train = [
    "key",
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
usecols_test = [
    "key",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]

dtype_train = {
    "fare_amount": np.float32,
    "pickup_longitude": np.float32,
    "pickup_latitude": np.float32,
    "dropoff_longitude": np.float32,
    "dropoff_latitude": np.float32,
    "passenger_count": np.int8,
}
dtype_test = {
    "pickup_longitude": np.float32,
    "pickup_latitude": np.float32,
    "dropoff_longitude": np.float32,
    "dropoff_latitude": np.float32,
    "passenger_count": np.int8,
}

train_df = pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv",
    nrows=10_000_000,
    usecols=usecols_train,
    parse_dates=["pickup_datetime"],
    dtype=dtype_train,
)
test_df = pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv",
    usecols=usecols_test,
    parse_dates=["pickup_datetime"],
    dtype=dtype_test,
)




## === cell 1
num_rows = len(train_df)
train_df = train_df[train_df["fare_amount"] > 0]
print(f"Drop {num_rows - len(train_df)} rows with non‑positive fare")




## === cell 2
def change_outliers_by_range(df, column_name, min_range, max_range):
    before_len = df.shape[0]
    mask = df[column_name].between(min_range, max_range)
    selected_rows = df[mask]
    changed_rows = before_len - selected_rows.shape[0]
    dtype_of_column = df[column_name].dtype
    mean_of_column = selected_rows[column_name].mean()
    if dtype_of_column == np.int64:
        mean_of_column = round(mean_of_column)
    df.loc[~mask, column_name] = mean_of_column
    return changed_rows


def change_outliers(df, name):
    print(f"{name} outliers:")
    print(
        "  pickup_latitude:",
        change_outliers_by_range(df, "pickup_latitude", 40.5, 41.0),
    )
    print(
        "  dropoff_latitude:",
        change_outliers_by_range(df, "dropoff_latitude", 40.5, 41.0),
    )
    print(
        "  pickup_longitude:",
        change_outliers_by_range(df, "pickup_longitude", -74.3, -73.60),
    )
    print(
        "  dropoff_longitude:",
        change_outliers_by_range(df, "dropoff_longitude", -74.3, -73.6),
    )
    print("  passenger_count:", change_outliers_by_range(df, "passenger_count", 1, 10))


change_outliers(train_df, "Training")
change_outliers(test_df, "Test")




## === cell 3
def preprocess_data(df):
    airport_lat_long = (40.644600, -73.779700)
    la_guardia_lat_long = (40.7733, -73.8718)

    near_airport = (
        (
            df["pickup_latitude"].between(
                airport_lat_long[0] - 0.005, airport_lat_long[0] + 0.005
            )
            & df["pickup_longitude"].between(
                airport_lat_long[1] - 0.005, airport_lat_long[1] + 0.005
            )
        )
        | (
            df["pickup_latitude"].between(
                la_guardia_lat_long[0] - 0.003, la_guardia_lat_long[0] + 0.002
            )
            & df["pickup_longitude"].between(
                la_guardia_lat_long[1] - 0.005, la_guardia_lat_long[1] + 0.005
            )
        )
    ).astype(int)
    df["near_airport"] = near_airport

    df["manhattan_distance"] = abs(
        df["pickup_longitude"] - df["dropoff_longitude"]
    ) + abs(df["pickup_latitude"] - df["dropoff_latitude"])
    df["manhattan_distance"] = np.clip(df["manhattan_distance"], 0, 200)

    def haversine_np(lat1, lon1, lat2, lon2):
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = (
            np.sin(dlat / 2.0) ** 2
            + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
        )
        c = 2 * np.arcsin(np.sqrt(a))
        return 6371.0 * c

    df["haversine_distance"] = haversine_np(
        df["pickup_latitude"],
        df["pickup_longitude"],
        df["dropoff_latitude"],
        df["dropoff_longitude"],
    )
    df["haversine_distance"] = np.clip(df["haversine_distance"], 0, 100)

    df["pickup_year"] = df["pickup_datetime"].dt.year
    df["pickup_month"] = df["pickup_datetime"].dt.month
    df["pickup_hour"] = df["pickup_datetime"].dt.hour
    df["pickup_day"] = df["pickup_datetime"].dt.dayofweek
    df["is_weekend"] = ((df["pickup_day"] >= 5) & (df["pickup_day"] <= 6)).astype(int)

    df["is_holiday"] = (
        ((df["pickup_month"] == 12) & df["pickup_datetime"].dt.day.isin([25, 26, 31]))
        | ((df["pickup_month"] == 1) & (df["pickup_datetime"].dt.day == 1))
        | ((df["pickup_month"] == 7) & (df["pickup_datetime"].dt.day == 4))
    ).astype(int)

    df["hour_sin"] = np.sin(2 * np.pi * df["pickup_hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["pickup_hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["pickup_month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["pickup_month"] / 12)


preprocess_data(train_df)
preprocess_data(test_df)




## === cell 4
features = [
    "pickup_latitude",
    "pickup_longitude",
    "dropoff_latitude",
    "dropoff_longitude",
    "near_airport",
    "manhattan_distance",
    "haversine_distance",
    "passenger_count",
    "pickup_year",
    "pickup_hour",
    "is_weekend",
    "is_holiday",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
]

for col in features:
    median_val = train_df[col].median()
    train_df[col].fillna(median_val, inplace=True)
    test_df[col].fillna(median_val, inplace=True)

train_df[features] = train_df[features].astype(np.float32, copy=False)
test_df[features] = test_df[features].astype(np.float32, copy=False)

test_keys = test_df["key"].values

X = train_df[features].values
y = train_df["fare_amount"].astype(np.float32).values
X_test = test_df[features].values

del train_df, test_df




## === cell 5
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=69)




## === cell 6
print("Shapes -> X:", X.shape, "y:", y.shape)
print("Train:", X_train.shape, y_train.shape)
print("Val  :", X_val.shape, y_val.shape)




## === cell 7
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
val_pred_lr = lr_model.predict(X_val)
print(
    "Linear Regression validation RMSE:",
    np.sqrt(mean_squared_error(y_val, val_pred_lr)),
)




## === cell 8
xgb_params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "max_depth": 12,  # a bit deeper to capture more interactions
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eta": 0.01,  # lower learning rate for finer fitting
    "min_child_weight": 1,
    "gamma": 0.0,
    "seed": 42,
    "tree_method": "hist",
    "nthread": -1,
    "verbosity": 0,
}
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)
dtest = xgb.DMatrix(X_test)
dall = xgb.DMatrix(X)

xgb_model = xgb.train(
    params=xgb_params,
    dtrain=dtrain,
    num_boost_round=2000,  # larger upper bound, early stopping will trim
    evals=[(dval, "validation")],
    early_stopping_rounds=100,  # allow more rounds before stopping
    verbose_eval=False,
)




## === cell 9
val_pred_xgb = xgb_model.predict(dval)
print("XGBoost validation RMSE:", np.sqrt(mean_squared_error(y_val, val_pred_xgb)))
train_pred_xgb = xgb_model.predict(dall)
print("XGBoost training RMSE:", np.sqrt(mean_squared_error(y, train_pred_xgb)))

test_pred_xgb = xgb_model.predict(dtest)
submission_df = pd.DataFrame({"key": test_keys, "fare_amount": test_pred_xgb})
submission_df.to_csv("submission.csv", index=False)
