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

5.76027

# 6. Current score

inf

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 936.87448) has done: 'I remove the deprecated `normalize` parameter from `LinearRegression` and make sure the test features are aligned to the same columns used for training before predicting. This fixes the runtime errors while keeping the original modeling pipeline intact, allowing a valid `.csv` submission to be generated.'
- What this solution (achieved inf) has done: 'I relax the geographic distance filter (increase the threshold from 5 to 10 degrees) to keep more training rows, and strengthen the ridge regularization (set alpha to 1.0). These minimal tweaks keep the core pipeline unchanged while likely improving the model’s generalisation and moving the RMSE closer to the target value.'
- What this solution (achieved inf) has done: 'I adjust the model regularisation slightly (use a smaller alpha so the Ridge model can fit the data a bit better) and make the prediction step robust to possible NaN values by replacing them with 0 after applying the non‑negative clamp. These two tiny tweaks keep the original pipeline intact while preventing invalid submissions that caused an infinite score and should move the RMSE closer to the target.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os

print(os.listdir("../input"))




## === cell 1
train_df = pd.read_csv("../input/train.csv", nrows=10_000_000)




## === cell 2
test_df = pd.read_csv("../input/test.csv")




## === cell 3
print("Test shape:", test_df.shape)




## === cell 4
def add_travel_vector_features(df):
    df["abs_diff_longitude"] = (df.dropoff_longitude - df.pickup_longitude).abs()
    df["abs_diff_latitude"] = (df.dropoff_latitude - df.pickup_latitude).abs()


add_travel_vector_features(train_df)
add_travel_vector_features(test_df)




## === cell 5
print(train_df.isnull().sum())




## === cell 6
print(f"Old train size: {len(train_df)}")
train_df = train_df.dropna(how="any", axis="rows")
print(f"New train size: {len(train_df)}")




## === cell 7
print(f"Before distance filter: {len(train_df)}")
train_df = train_df[
    (train_df.abs_diff_longitude < 10.0) & (train_df.abs_diff_latitude < 10.0)
]
print(f"After distance filter: {len(train_df)}")




## === cell 8
train_df["pickuptime"] = (
    pd.to_datetime(train_df["pickup_datetime"]).dt.hour * 100
    + pd.to_datetime(train_df["pickup_datetime"]).dt.minute
)
test_df["pickuptime"] = (
    pd.to_datetime(test_df["pickup_datetime"]).dt.hour * 100
    + pd.to_datetime(test_df["pickup_datetime"]).dt.minute
)




## === cell 9
def extract_weekday(series):
    weekdays = []
    for ts in series:
        ts = ts[:-4]  # remove seconds and timezone
        weekdays.append(pd.Timestamp(ts).weekday())
    return weekdays


train_df["weekday"] = extract_weekday(train_df["pickup_datetime"])
test_df["weekday"] = extract_weekday(test_df["pickup_datetime"])




## === cell 10
train_df.drop(columns="pickup_datetime", inplace=True)
test_df.drop(columns="pickup_datetime", inplace=True)




## === cell 11
weekday_map = {
    0: "Monday",
    1: "Tuesday",
    2: "Wednesday",
    3: "Thursday",
    4: "Friday",
    5: "Saturday",
    6: "Sunday",
}
train_df["weekday"] = train_df["weekday"].map(weekday_map)
test_df["weekday"] = test_df["weekday"].map(weekday_map)




## === cell 12
train_one_hot = pd.get_dummies(train_df["weekday"])
test_one_hot = pd.get_dummies(test_df["weekday"])
train_df = pd.concat([train_df, train_one_hot], axis=1)
test_df = pd.concat([test_df, test_one_hot], axis=1)




## === cell 13
train_df.drop(columns="weekday", inplace=True)
test_df.drop(columns="weekday", inplace=True)




## === cell 14
train_df["pickuptime"] = train_df["pickuptime"].astype(int)
test_df["pickuptime"] = test_df["pickuptime"].astype(int)




## === cell 15
R = 6373.0  # Earth radius in km


def haversine(df):
    lat1 = np.radians(df["pickup_latitude"])
    lon1 = np.radians(df["pickup_longitude"])
    lat2 = np.radians(df["dropoff_latitude"])
    lon2 = np.radians(df["dropoff_longitude"])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c * 0.621  # convert km to miles


train_df["Distance"] = haversine(train_df)
test_df["Distance"] = haversine(test_df)




## === cell 16
airport_lat = np.radians(40.6413111)
airport_lon = np.radians(-73.7781391)


def airport_dist(df, lat_col, lon_col):
    lat = np.radians(df[lat_col])
    lon = np.radians(df[lon_col])
    dlon = airport_lon - lon
    dlat = airport_lat - lat
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat) * np.cos(airport_lat) * np.sin(dlon / 2) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c * 0.621  # miles


train_df["Pickup_Distance_airport"] = airport_dist(
    train_df, "pickup_latitude", "pickup_longitude"
)
train_df["Dropoff_Distance_airport"] = airport_dist(
    train_df, "dropoff_latitude", "dropoff_longitude"
)
test_df["Pickup_Distance_airport"] = airport_dist(
    test_df, "pickup_latitude", "pickup_longitude"
)
test_df["Dropoff_Distance_airport"] = airport_dist(
    test_df, "dropoff_latitude", "dropoff_longitude"
)




## === cell 17
train_df["Total_Distance"] = (
    train_df["Distance"]
    + train_df["Pickup_Distance_airport"]
    + train_df["Dropoff_Distance_airport"]
)
test_df["Total_Distance"] = (
    test_df["Distance"]
    + test_df["Pickup_Distance_airport"]
    + test_df["Dropoff_Distance_airport"]
)




## === cell 18
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




## === cell 19
for col in ["abs_diff_longitude", "abs_diff_latitude"]:
    mean = train_df[col].mean()
    std = train_df[col].std()
    if std == 0:
        std = 1.0
    train_df[col] = (train_df[col] - mean) / std
    test_df[col] = (test_df[col] - mean) / std




## === cell 20
print("Train shape:", train_df.shape, "Test shape:", test_df.shape)




## === cell 21
train_df = train_df[train_df["fare_amount"] > 0].reset_index(drop=True)

X = train_df.drop(columns=["key", "fare_amount"])
Y = np.log1p(train_df["fare_amount"])
X = X.fillna(0)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.01, random_state=80)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)




## === cell 22
from sklearn.linear_model import Ridge

lr = Ridge(alpha=0.1, random_state=42)
lr.fit(X_train_scaled, Y_train)

val_pred = np.expm1(lr.predict(X_val_scaled))
val_rmse = np.sqrt(((np.expm1(Y_val) - val_pred) ** 2).mean())
print(f"Validation RMSE: {val_rmse:.4f}")




## === cell 23
test_features = test_df.drop(columns="key")
test_features = test_features.reindex(columns=X.columns, fill_value=0)
test_features_scaled = scaler.transform(test_features)

pred = np.expm1(lr.predict(test_features_scaled))
pred = np.maximum(pred, 0)  # fares cannot be negative
pred = np.where(np.isnan(pred), 0, pred)




## === cell 24
submission = pd.DataFrame({"key": test_df["key"], "fare_amount": pred})
submission = submission[["key", "fare_amount"]]




## === cell 25
submission.to_csv("submission.csv", index=False)
