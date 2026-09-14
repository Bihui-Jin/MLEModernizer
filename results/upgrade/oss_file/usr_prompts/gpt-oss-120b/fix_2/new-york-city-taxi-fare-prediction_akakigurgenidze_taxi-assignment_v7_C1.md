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

# 5. Target score

3.39438

# 6. Current score

4.47595

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

- What this solution (achieved 4.47595) has done: 'The script now avoids the TensorFlow import that caused the protobuf AttributeError, adds a proper haversine distance feature (a key predictor for taxi fares), includes this feature in the model inputs, and trains the XGBoost model with early‑stopping to improve validation RMSE while keeping the original workflow intact. All cells are renumbered sequentially and the final submission file is written to **submission.csv**.'

# 9. Code solution

## === cell 0
markdown


## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2950632611.py in <cell line: 0>()
----> 1 markdown

NameError: name 'markdown' is not defined

## === cell 1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import xgboost as xgb

train_df = pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv", nrows=10_000_000
)
test_df = pd.read_csv("/kaggle/input/new-york-city-taxi-fare-prediction/test.csv")


## === cell 2
train_df.head()


## === cell 3
test_df.head()


## === cell 4
train_df.dtypes


## === cell 5
train_df.describe()


## === cell 6
num_rows = len(train_df)
train_df = train_df[train_df["fare_amount"] > 0]
print(f"Drop {num_rows - len(train_df)} rows")




## === cell 7
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


def change_outliers(df):
    print(
        "Change",
        change_outliers_by_range(df, "pickup_latitude", 40.5, 41.0),
        "rows by pickup lat",
    )
    print(
        "Change",
        change_outliers_by_range(df, "dropoff_latitude", 40.5, 41.0),
        "rows by dropoff lat",
    )
    print(
        "Change",
        change_outliers_by_range(df, "pickup_longitude", -74.3, -73.60),
        "rows by pickup long",
    )
    print(
        "Change",
        change_outliers_by_range(df, "dropoff_longitude", -74.3, -73.6),
        "rows by dropoff long",
    )
    print(
        "Change",
        change_outliers_by_range(df, "passenger_count", 1, 10),
        "rows by passenger cnt",
    )


print("Training data outliers:")
change_outliers(train_df)
print("\nTest data outliers:")
change_outliers(test_df)


## === cell 8
train_df.describe()


## === cell 9
train_df.isnull().sum()




## === cell 10
def preprocess_data(df):
    airport_lat_long = (40.644600, -73.779700)
    la_guardia_lat_long = (40.7733, -73.8718)
    near_airport = (
        (
            (
                df["pickup_latitude"].between(
                    airport_lat_long[0] - 0.005, airport_lat_long[0] + 0.005
                )
            )
            & (
                df["pickup_longitude"].between(
                    airport_lat_long[1] - 0.005, airport_lat_long[1] + 0.005
                )
            )
        )
        | (
            (
                df["pickup_latitude"].between(
                    la_guardia_lat_long[0] - 0.003, la_guardia_lat_long[0] + 0.002
                )
            )
            & (
                df["pickup_longitude"].between(
                    la_guardia_lat_long[1] - 0.005, la_guardia_lat_long[1] + 0.005
                )
            )
        )
    ).astype(int)
    df["near_airport"] = near_airport

    df["manhattan_distance"] = abs(
        df["pickup_longitude"] - df["dropoff_longitude"]
    ) + abs(df["pickup_latitude"] - df["dropoff_latitude"])

    def haversine_np(lat1, lon1, lat2, lon2):
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = (
            np.sin(dlat / 2.0) ** 2
            + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
        )
        c = 2 * np.arcsin(np.sqrt(a))
        km = 6371.0 * c
        return km

    df["haversine_distance"] = haversine_np(
        df["pickup_latitude"],
        df["pickup_longitude"],
        df["dropoff_latitude"],
        df["dropoff_longitude"],
    )

    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
    df["pickup_year"] = df["pickup_datetime"].dt.year
    df["pickup_month"] = df["pickup_datetime"].dt.month
    df["pickup_hour"] = df["pickup_datetime"].dt.hour
    df["pickup_day"] = df["pickup_datetime"].dt.dayofweek
    df["is_weekend"] = ((df["pickup_day"] >= 5) & (df["pickup_day"] <= 6)).astype(int)

    df["is_holiday"] = (
        ((df["pickup_month"] == 12) & (df["pickup_datetime"].dt.day.isin([25, 26, 31])))
        | ((df["pickup_month"] == 1) & (df["pickup_datetime"].dt.day == 1))
        | ((df["pickup_month"] == 7) & (df["pickup_datetime"].dt.day == 4))
    ).astype(int)


preprocess_data(train_df)
preprocess_data(test_df)


## === cell 11
train_df.head()


## === cell 12
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
]
X = train_df[features].values
y = train_df["fare_amount"].values


## === cell 13
X_test = test_df[features].values


## === cell 14
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=69)


## === cell 15
print("Shapes -> X:", X.shape, "y:", y.shape)
print("Train:", X_train.shape, y_train.shape)
print("Val  :", X_val.shape, y_val.shape)


## === cell 16
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
val_pred_lr = lr_model.predict(X_val)
print(
    "Linear Regression validation RMSE:",
    np.sqrt(mean_squared_error(y_val, val_pred_lr)),
)


## === cell 17
xgb_params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "max_depth": 10,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "eta": 0.05,
    "min_child_weight": 3,
    "gamma": 0.1,
    "seed": 42,
    "tree_method": "hist",
    "nthread": -1,
}
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)
dtest = xgb.DMatrix(X_test)
dall = xgb.DMatrix(X)

xgb_model = xgb.train(
    params=xgb_params,
    dtrain=dtrain,
    num_boost_round=500,
    evals=[(dval, "validation")],
    early_stopping_rounds=30,
    verbose_eval=False,
)


## === cell 18
val_pred_xgb = xgb_model.predict(dval)
print("XGBoost validation RMSE:", np.sqrt(mean_squared_error(y_val, val_pred_xgb)))
train_pred_xgb = xgb_model.predict(dall)
print("XGBoost training RMSE:", np.sqrt(mean_squared_error(y, train_pred_xgb)))


## === cell 19
test_pred_xgb = xgb_model.predict(dtest)
submission_df = pd.DataFrame({"key": test_df["key"], "fare_amount": test_pred_xgb})
submission_df.to_csv("submission.csv", index=False)
