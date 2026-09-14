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

3.10

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

None

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 936.92806) has done: 'I remove the deprecated `normalize` argument from `LinearRegression`, and make sure the test features are aligned with the training columns before prediction. This fixes the runtime errors and guarantees a valid `Submission.csv` with the correct columns.'
- What this solution (achieved 752.76803) has done: 'I keep the overall pipeline and model unchanged, but fix the feature scaling so that the same training‑derived statistics are applied to both the train and test sets. This removes the mismatch that caused wildly inaccurate predictions and should bring the RMSE much closer to the target ≈ 5.69.'
- What this solution (achieved 752.76916) has done: 'I add a standard‑scaler step that learns the mean and standard‑deviation from the training split and applies the same transformation to the validation and test features. Using consistent scaling for all numeric columns (including the engineered ones) typically brings the linear model’s predictions much closer to the true fares, reducing the RMSE toward the target while keeping the original model and overall pipeline unchanged.'
- What this solution (achieved 762.5723) has done: 'I correct the scaling of the travel‑vector features so they use the proper mean‑std standardisation (instead of the incorrect variance‑based, absolute‑value formula) and then clip the model’s predictions to non‑negative values before writing the submission. These small fixes keep the original pipeline unchanged while removing the extreme distortion that caused the huge RMSE, moving the score much closer to the target.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import time
import os  # reading the input files we have access to

print(os.listdir("../input"))



## === cell 1
train_df = pd.read_csv("../input/train.csv", nrows=10_000_000)
train_df.dtypes



## === cell 2
train_df.head()



## === cell 3
train_df.shape



## === cell 4
train_df.info()



## === cell 5
test_df = pd.read_csv("../input/test.csv")
test_df.dtypes



## === cell 6
test_df.head()



## === cell 7
test_df.info()



## === cell 8
test_df.shape



## === cell 9
train_df.isna().sum()




## === cell 10
def add_travel_vector_features(df):
    df["abs_diff_longitude"] = (df.dropoff_longitude - df.pickup_longitude).abs()
    df["abs_diff_latitude"] = (df.dropoff_latitude - df.pickup_latitude).abs()


add_travel_vector_features(train_df)
add_travel_vector_features(test_df)



## === cell 11
print(f"Before Dropping null values: {len(train_df)}")
train_df.dropna(inplace=True)
print(f"After Dropping null values: {len(train_df)}")



## === cell 12
print(train_df.isnull().sum())



## === cell 13
print("Old size: %d" % len(train_df))
train_df = train_df.dropna(how="any", axis="rows")
print("New size: %d" % len(train_df))



## === cell 14
plot = train_df.iloc[:2000].plot.scatter("abs_diff_longitude", "abs_diff_latitude")



## === cell 15
print("Old size: %d" % len(train_df))
train_df = train_df[
    (train_df.abs_diff_longitude < 5.0) & (train_df.abs_diff_latitude < 5.0)
]
print("New size: %d" % len(train_df))




## === cell 16
def creating_time(df):
    ls1 = list(df["pickup_datetime"])
    for i in range(len(ls1)):
        ls1[i] = ls1[i][11:-7:]
    df["pickuptime"] = ls1


creating_time(train_df)
creating_time(test_df)



## === cell 17
train_df.head()



## === cell 18
test_df.head()




## === cell 19
def creating_weekdays(df):
    ls1 = list(df["pickup_datetime"])
    for i in range(len(ls1)):
        ls1[i] = ls1[i][:-4:]
        ls1[i] = pd.Timestamp(ls1[i])
        ls1[i] = ls1[i].weekday()
    df["Weekday"] = ls1


creating_weekdays(train_df)
creating_weekdays(test_df)



## === cell 20
train_df.head()



## === cell 21
test_df.head()



## === cell 22
train_df.drop("pickup_datetime", inplace=True, axis=1)
test_df.drop("pickup_datetime", inplace=True, axis=1)




## === cell 23
def replace_weekday(df):
    df["Weekday"].replace(
        to_replace=[i for i in range(0, 7)],
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


replace_weekday(train_df)
replace_weekday(test_df)



## === cell 24
train_one_hot = pd.get_dummies(train_df["Weekday"])
test_one_hot = pd.get_dummies(test_df["Weekday"])
train_df = pd.concat([train_df, train_one_hot], axis=1)
test_df = pd.concat([test_df, test_one_hot], axis=1)



## === cell 25
train_df.head()



## === cell 26
test_df.head()



## === cell 27
train_df.drop("Weekday", axis=1, inplace=True)
test_df.drop("Weekday", axis=1, inplace=True)




## === cell 28
def creating_pickupdate(df):
    ls1 = list(df["pickuptime"])
    for i in range(len(ls1)):
        z = ls1[i].split(":")
        ls1[i] = int(z[0]) * 100 + int(z[1])
    df["pickuptime"] = ls1


creating_pickupdate(train_df)
creating_pickupdate(test_df)



## === cell 29
train_df.head()



## === cell 30
test_df.head()




## === cell 31
def add_time_and_weekend_features(df):
    df["hour"] = df["pickuptime"] // 100
    if "Saturday" in df.columns and "Sunday" in df.columns:
        df["is_weekend"] = df["Saturday"] + df["Sunday"]
    else:
        df["is_weekend"] = 0


add_time_and_weekend_features(train_df)
add_time_and_weekend_features(test_df)




## === cell 32
def finding_distance(df):
    R = 6373.0
    lat1 = np.asarray(np.radians(df["pickup_latitude"]))
    lon1 = np.asarray(np.radians(df["pickup_longitude"]))
    lat2 = np.asarray(np.radians(df["dropoff_latitude"]))
    lon2 = np.asarray(np.radians(df["dropoff_longitude"]))

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c

    df["Distance"] = np.asarray(distance) * 0.621


finding_distance(train_df)
finding_distance(test_df)




## === cell 33
def creating_pickup_dropoff_distance(df):
    R = 6373.0
    lat1 = np.asarray(np.radians(df["pickup_latitude"]))
    lon1 = np.asarray(np.radians(df["pickup_longitude"]))
    lat2 = np.asarray(np.radians(df["dropoff_latitude"]))
    lon2 = np.asarray(np.radians(df["dropoff_longitude"]))

    lat3 = np.zeros(len(df)) + np.radians(40.6413111)
    lon3 = np.zeros(len(df)) + np.radians(-73.7781391)
    dlon_pickup = lon3 - lon1
    dlat_pickup = lat3 - lat1
    d_lon_dropoff = lon3 - lon2
    d_lat_dropoff = lat3 - lat2
    a1 = (
        np.sin(dlat_pickup / 2) ** 2
        + np.cos(lat1) * np.cos(lat3) * np.sin(dlon_pickup / 2) ** 2
    )
    c1 = 2 * np.arctan2(np.sqrt(a1), np.sqrt(1 - a1))
    distance1 = R * c1
    df["Pickup_Distance_airport"] = np.asarray(distance1) * 0.621

    a2 = (
        np.sin(d_lat_dropoff / 2) ** 2
        + np.cos(lat2) * np.cos(lat3) * np.sin(d_lon_dropoff / 2) ** 2
    )
    c2 = 2 * np.arctan2(np.sqrt(a2), np.sqrt(1 - a2))
    distance2 = R * c2

    df["Dropoff_Distance_airport"] = np.asarray(distance2) * 0.621


creating_pickup_dropoff_distance(train_df)
creating_pickup_dropoff_distance(test_df)



## === cell 35
train_df.drop(
    ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"],
    axis=1,
    inplace=True,
)
test_df.drop(
    ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"],
    axis=1,
    inplace=True,
)



## === cell 36
abs_long_mean = train_df["abs_diff_longitude"].mean()
abs_long_std = train_df["abs_diff_longitude"].std()
abs_lat_mean = train_df["abs_diff_latitude"].mean()
abs_lat_std = train_df["abs_diff_latitude"].std()



## === cell 37
print(train_df.shape)
print(test_df.shape)



## === cell 38
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = train_df.drop(["key", "fare_amount"], axis=1)
y = train_df["fare_amount"]

valid_mask = y.notna() & (y > 0)
X = X.loc[valid_mask]
y = y.loc[valid_mask]

y_log = np.log1p(y)

X_train, X_val, y_train_log, y_val = train_test_split(
    X, y_log, test_size=0.01, random_state=80
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)



## === cell 39
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

ridge = Ridge(alpha=0.5, random_state=80)
ridge.fit(X_train_scaled, y_train_log)

val_pred = np.expm1(ridge.predict(X_val_scaled))
val_rmse = np.sqrt(mean_squared_error(np.expm1(y_val), val_pred))
print(f"Validation RMSE (original scale): {val_rmse:.4f}")



## === cell 40
test_features = test_df.drop("key", axis=1).reindex(columns=X.columns, fill_value=0)
test_features_scaled = scaler.transform(test_features)
pred_log = ridge.predict(test_features_scaled)
pred = np.expm1(pred_log)  # invert log1p
pred = np.round(pred, 2)
pred = np.clip(pred, 0, None)
print(pred[:5])



## === cell 41
Submission = pd.DataFrame({"key": test_df["key"], "fare_amount": pred})
Submission = Submission[["key", "fare_amount"]]



## === cell 42
Submission.head()



## === cell 43
Submission.to_csv("submission.csv", index=False)



## === cell 44
print("Submission saved to submission.csv")
