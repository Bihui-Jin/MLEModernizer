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

3.9

# 3. Installed packages

geopandas==0.14.4
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

5.6891

# 6. Current score

10.25603

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 936.92806) has done: 'I fix the deprecated `normalize` argument in `LinearRegression`, ensure the test set has exactly the same feature columns as the training set (adding missing one‑hot columns with zeros), and correctly write the prediction dataframe to a CSV file named `submission.csv`. These minimal changes resolve the runtime errors and produce a valid submission while keeping the original modeling approach unchanged.'
- What this solution (achieved 10.25603) has done: 'The fix adds the missing imports, ensures all preprocessing functions run on the loaded data, aligns test columns with the training feature set, removes the deprecated `normalize` argument from `LinearRegression`, and writes a proper `submission.csv` with the required `key` and `fare_amount` columns. These changes resolve the NameError cascade and produce a valid submission while preserving the original modeling approach.'

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error



## === cell 1
train_path = "../input/train.csv"
test_path = "../input/test.csv"
train_df = pd.read_csv(train_path, nrows=1000000)  # 1M rows for demo
test_df = pd.read_csv(test_path)




## === cell 2
def add_travel_vector_features(df):
    df["abs_diff_longitude"] = (df["dropoff_longitude"] - df["pickup_longitude"]).abs()
    df["abs_diff_latitude"] = (df["dropoff_latitude"] - df["pickup_latitude"]).abs()


def creating_time(df):
    df["pickuptime"] = df["pickup_datetime"].str[11:16]


def creating_weekdays(df):
    df["Weekday"] = pd.to_datetime(df["pickup_datetime"]).dt.weekday


def replace_weekday(df):
    df["Weekday"].replace(
        to_replace=list(range(7)),
        value=[
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ],
        inplace=True,
    )


def creating_pickupdate(df):
    df["pickuptime"] = df["pickuptime"].apply(
        lambda x: int(x.split(":")[0]) * 100 + int(x.split(":")[1])
    )


def finding_distance(df):
    R = 6373.0
    lat1 = np.radians(df["pickup_latitude"])
    lon1 = np.radians(df["pickup_longitude"])
    lat2 = np.radians(df["dropoff_latitude"])
    lon2 = np.radians(df["dropoff_longitude"])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c * 0.621  # miles
    df["Distance"] = distance


def creating_pickup_dropoff_distance(df):
    R = 6373.0
    lat_air = np.radians(40.6413111)
    lon_air = np.radians(-73.7781391)
    lat1 = np.radians(df["pickup_latitude"])
    lon1 = np.radians(df["pickup_longitude"])
    dlon_p = lon_air - lon1
    dlat_p = lat_air - lat1
    a1 = (
        np.sin(dlat_p / 2) ** 2
        + np.cos(lat1) * np.cos(lat_air) * np.sin(dlon_p / 2) ** 2
    )
    c1 = 2 * np.arctan2(np.sqrt(a1), np.sqrt(1 - a1))
    df["Pickup_Distance_airport"] = R * c1 * 0.621
    lat2 = np.radians(df["dropoff_latitude"])
    lon2 = np.radians(df["dropoff_longitude"])
    dlon_d = lon_air - lon2
    dlat_d = lat_air - lat2
    a2 = (
        np.sin(dlat_d / 2) ** 2
        + np.cos(lat2) * np.cos(lat_air) * np.sin(dlon_d / 2) ** 2
    )
    c2 = 2 * np.arctan2(np.sqrt(a2), np.sqrt(1 - a2))
    df["Dropoff_Distance_airport"] = R * c2 * 0.621




## === cell 3
add_travel_vector_features(train_df)
add_travel_vector_features(test_df)

creating_time(train_df)
creating_time(test_df)

creating_weekdays(train_df)
creating_weekdays(test_df)

replace_weekday(train_df)
replace_weekday(test_df)

creating_pickupdate(train_df)
creating_pickupdate(test_df)

finding_distance(train_df)
finding_distance(test_df)

creating_pickup_dropoff_distance(train_df)
creating_pickup_dropoff_distance(test_df)



## === cell 4
train_df.drop(columns=["pickup_datetime"], inplace=True)
test_df.drop(columns=["pickup_datetime"], inplace=True)

train_one_hot = pd.get_dummies(train_df["Weekday"])
test_one_hot = pd.get_dummies(test_df["Weekday"])
train_df = pd.concat([train_df, train_one_hot], axis=1)
test_df = pd.concat([test_df, test_one_hot], axis=1)

train_df.drop(columns=["Weekday"], inplace=True)
test_df.drop(columns=["Weekday"], inplace=True)



## === cell 5
for col in ["Distance", "Pickup_Distance_airport", "Dropoff_Distance_airport"]:
    train_df[col] = np.round(train_df[col], 2)
    test_df[col] = np.round(test_df[col], 2)

train_df.drop(
    columns=[
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
    ],
    inplace=True,
)
test_df.drop(
    columns=[
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
    ],
    inplace=True,
)



## === cell 6
for df in (train_df, test_df):
    df["abs_diff_longitude"] = np.abs(
        df["abs_diff_longitude"] - df["abs_diff_longitude"].mean()
    )
    df["abs_diff_longitude"] = df["abs_diff_longitude"] / df["abs_diff_longitude"].var()
    df["abs_diff_latitude"] = np.abs(
        df["abs_diff_latitude"] - df["abs_diff_latitude"].mean()
    )
    df["abs_diff_latitude"] = df["abs_diff_latitude"] / df["abs_diff_latitude"].var()



## === cell 7
X = train_df.drop(columns=["key", "fare_amount"])
y = train_df["fare_amount"]
y_log = np.log1p(y)

X_train, X_val, y_train_log, y_val_log = train_test_split(
    X, y_log, test_size=0.01, random_state=80
)

valid_mask = X_train.notnull().all(axis=1) & np.isfinite(y_train_log)
X_train = X_train[valid_mask]
y_train_log = y_train_log[valid_mask]



## === cell 8
lr = LinearRegression()
lr.fit(X_train, y_train_log)

val_pred_log = lr.predict(X_val)
val_rmse_log = np.sqrt(mean_squared_error(y_val_log, val_pred_log))
print("Validation RMSE on log‑target:", val_rmse_log)



## === cell 9
test_features = test_df.drop(columns=["key"])
test_features = test_features.reindex(columns=X.columns, fill_value=0)

pred_log = lr.predict(test_features)
pred = np.expm1(pred_log)
pred = np.clip(pred, a_min=0, a_max=None)
pred = np.round(pred, 2)



## === cell 10
submission = pd.DataFrame({"key": test_df["key"], "fare_amount": pred})
submission = submission[["key", "fare_amount"]]



## === cell 11
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission saved to '{submission_path}'")
