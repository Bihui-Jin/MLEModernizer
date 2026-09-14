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

3.9

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

5.68914

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

- What this solution (achieved 936.94064) has done: 'The script was failing because `LinearRegression` no longer accepts the `normalize` argument, which caused the model not to be created and all subsequent cells to crash. I removed the unsupported argument and wrapped the regressor in a small pipeline that standard‑scales the features, keeping the original linear‑model approach. I also fixed the submission‑creation steps so the prediction array is correctly turned into a DataFrame with the required columns and written to a CSV file. All other preprocessing steps are retained.'

# 9. Code solution

## === cell 0
print("Data directory:", "/kaggle/input/new-york-city-taxi-fare-prediction")


## === cell 1
dtype_map = {
    "fare_amount": np.float32,
    "pickup_longitude": np.float32,
    "pickup_latitude": np.float32,
    "dropoff_longitude": np.float32,
    "dropoff_latitude": np.float32,
    "passenger_count": np.int8,
}
train_data = pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv",
    nrows=2_000_000,  # <<< reduced from 10 M rows
    dtype=dtype_map,
)
train_data.head()


## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/19877274.py in <cell line: 0>()
      1 dtype_map = {
----> 2     "fare_amount": np.float32,
      3     "pickup_longitude": np.float32,
      4     "pickup_latitude": np.float32,
      5     "dropoff_longitude": np.float32,

NameError: name 'np' is not defined

## === cell 2
train_data.shape


## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2272128234.py in <cell line: 0>()
----> 1 train_data.shape

NameError: name 'train_data' is not defined

## === cell 3
train_data.info()


## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/352001865.py in <cell line: 0>()
----> 1 train_data.info()

NameError: name 'train_data' is not defined

## === cell 4
test_data = pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv",
    dtype=dtype_map,
)
test_data.head()


## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3202061986.py in <cell line: 0>()
----> 1 test_data = pd.read_csv(
      2     "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv",
      3     dtype=dtype_map,
      4 )
      5 test_data.head()

NameError: name 'pd' is not defined

## === cell 5
test_data.info()


## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3438184762.py in <cell line: 0>()
----> 1 test_data.info()

NameError: name 'test_data' is not defined

## === cell 6
train_data.isna().sum()


## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2674134246.py in <cell line: 0>()
----> 1 train_data.isna().sum()

NameError: name 'train_data' is not defined

## === cell 7
train_data["Difference_longitude"] = np.abs(
    train_data["pickup_longitude"] - train_data["dropoff_longitude"]
)
train_data["Difference_latitude"] = np.abs(
    train_data["pickup_latitude"] - train_data["dropoff_latitude"]
)

test_data["Difference_longitude"] = np.abs(
    test_data["pickup_longitude"] - test_data["dropoff_longitude"]
)
test_data["Difference_latitude"] = np.abs(
    test_data["pickup_latitude"] - test_data["dropoff_latitude"]
)


## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3284048867.py in <cell line: 0>()
----> 1 train_data["Difference_longitude"] = np.abs(
      2     train_data["pickup_longitude"] - train_data["dropoff_longitude"]
      3 )
      4 train_data["Difference_latitude"] = np.abs(
      5     train_data["pickup_latitude"] - train_data["dropoff_latitude"]

NameError: name 'np' is not defined

## === cell 8
print(f"Before Dropping null values: {len(train_data)}")
train_data.dropna(inplace=True)
print(f"After Dropping null values: {len(train_data)}")


## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/844105544.py in <cell line: 0>()
----> 1 print(f"Before Dropping null values: {len(train_data)}")
      2 train_data.dropna(inplace=True)
      3 print(f"After Dropping null values: {len(train_data)}")

NameError: name 'train_data' is not defined

## === cell 9
train_data = train_data[
    (train_data["Difference_longitude"] < 5.0)
    & (train_data["Difference_latitude"] < 5.0)
]


## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1606394833.py in <cell line: 0>()
----> 1 train_data = train_data[
      2     (train_data["Difference_longitude"] < 5.0)
      3     & (train_data["Difference_latitude"] < 5.0)
      4 ]

NameError: name 'train_data' is not defined

## === cell 10
train_dt = pd.to_datetime(train_data["pickup_datetime"])
train_data["pickuptime"] = train_dt.dt.hour * 100 + train_dt.dt.minute

test_dt = pd.to_datetime(test_data["pickup_datetime"])
test_data["pickuptime"] = test_dt.dt.hour * 100 + test_dt.dt.minute


## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3369269993.py in <cell line: 0>()
----> 1 train_dt = pd.to_datetime(train_data["pickup_datetime"])
      2 train_data["pickuptime"] = train_dt.dt.hour * 100 + train_dt.dt.minute
      3 
      4 test_dt = pd.to_datetime(test_data["pickup_datetime"])
      5 test_data["pickuptime"] = test_dt.dt.hour * 100 + test_dt.dt.minute

NameError: name 'pd' is not defined

## === cell 11
train_data.head()


## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1664430853.py in <cell line: 0>()
----> 1 train_data.head()

NameError: name 'train_data' is not defined

## === cell 12
train_data["Weekday"] = pd.to_datetime(train_data["pickup_datetime"]).dt.weekday
test_data["Weekday"] = pd.to_datetime(test_data["pickup_datetime"]).dt.weekday


## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2784509118.py in <cell line: 0>()
----> 1 train_data["Weekday"] = pd.to_datetime(train_data["pickup_datetime"]).dt.weekday
      2 test_data["Weekday"] = pd.to_datetime(test_data["pickup_datetime"]).dt.weekday

NameError: name 'pd' is not defined

## === cell 13
train_data.head()


## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1664430853.py in <cell line: 0>()
----> 1 train_data.head()

NameError: name 'train_data' is not defined

## === cell 14
test_data.head()


## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3713091046.py in <cell line: 0>()
----> 1 test_data.head()

NameError: name 'test_data' is not defined

## === cell 15
train_data.drop("pickup_datetime", inplace=True, axis=1)
test_data.drop("pickup_datetime", inplace=True, axis=1)


## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1969310352.py in <cell line: 0>()
----> 1 train_data.drop("pickup_datetime", inplace=True, axis=1)
      2 test_data.drop("pickup_datetime", inplace=True, axis=1)

NameError: name 'train_data' is not defined

## === cell 16
train_data["Weekday"].replace(
    to_replace=[i for i in range(7)],
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

test_data["Weekday"].replace(
    to_replace=[i for i in range(7)],
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


## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2561112985.py in <cell line: 0>()
----> 1 train_data["Weekday"].replace(
      2     to_replace=[i for i in range(7)],
      3     value=[
      4         "Monday",
      5         "Tuesday",

NameError: name 'train_data' is not defined

## === cell 17
train_one_hot = pd.get_dummies(train_data["Weekday"]).astype(np.float32)
test_one_hot = pd.get_dummies(test_data["Weekday"]).astype(np.float32)
train_data = pd.concat([train_data, train_one_hot], axis=1)
test_data = pd.concat([test_data, test_one_hot], axis=1)


## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2043580175.py in <cell line: 0>()
----> 1 train_one_hot = pd.get_dummies(train_data["Weekday"]).astype(np.float32)
      2 test_one_hot = pd.get_dummies(test_data["Weekday"]).astype(np.float32)
      3 train_data = pd.concat([train_data, train_one_hot], axis=1)
      4 test_data = pd.concat([test_data, test_one_hot], axis=1)

NameError: name 'pd' is not defined

## === cell 18
train_data.drop("Weekday", axis=1, inplace=True)
test_data.drop("Weekday", axis=1, inplace=True)


## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1049282192.py in <cell line: 0>()
----> 1 train_data.drop("Weekday", axis=1, inplace=True)
      2 test_data.drop("Weekday", axis=1, inplace=True)

NameError: name 'train_data' is not defined

## === cell 19
train_data.head()


## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1664430853.py in <cell line: 0>()
----> 1 train_data.head()

NameError: name 'train_data' is not defined

## === cell 20
R = 6373.0
lat1 = np.radians(train_data["pickup_latitude"])
lon1 = np.radians(train_data["pickup_longitude"])
lat2 = np.radians(train_data["dropoff_latitude"])
lon2 = np.radians(train_data["dropoff_longitude"])

dlon = lon2 - lon1
dlat = lat2 - lat1
a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
distance = R * c
train_data["Distance"] = np.round(distance * 0.621, 2)

lat1 = np.radians(test_data["pickup_latitude"])
lon1 = np.radians(test_data["pickup_longitude"])
lat2 = np.radians(test_data["dropoff_latitude"])
lon2 = np.radians(test_data["dropoff_longitude"])

dlon = lon2 - lon1
dlat = lat2 - lat1
a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
distance = R * c
test_data["Distance"] = np.round(distance * 0.621, 2)


## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2837450510.py in <cell line: 0>()
      1 R = 6373.0
----> 2 lat1 = np.radians(train_data["pickup_latitude"])
      3 lon1 = np.radians(train_data["pickup_longitude"])
      4 lat2 = np.radians(train_data["dropoff_latitude"])
      5 lon2 = np.radians(train_data["dropoff_longitude"])

NameError: name 'np' is not defined

## === cell 21
R = 6373.0
lat1 = np.radians(train_data["pickup_latitude"])
lon1 = np.radians(train_data["pickup_longitude"])
lat2 = np.radians(train_data["dropoff_latitude"])
lon2 = np.radians(train_data["dropoff_longitude"])

lat3 = np.radians(40.6413111)  # JFK airport latitude
lon3 = np.radians(-73.7781391)  # JFK airport longitude

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
train_data["Pickup_Distance_airport"] = np.round(distance1 * 0.621, 2)

a2 = (
    np.sin(d_lat_dropoff / 2) ** 2
    + np.cos(lat2) * np.cos(lat3) * np.sin(d_lon_dropoff / 2) ** 2
)
c2 = 2 * np.arctan2(np.sqrt(a2), np.sqrt(1 - a2))
distance2 = R * c2
train_data["Dropoff_Distance_airport"] = np.round(distance2 * 0.621, 2)

lat1 = np.radians(test_data["pickup_latitude"])
lon1 = np.radians(test_data["pickup_longitude"])
lat2 = np.radians(test_data["dropoff_latitude"])
lon2 = np.radians(test_data["dropoff_longitude"])

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
test_data["Pickup_Distance_airport"] = np.round(distance1 * 0.621, 2)

a2 = (
    np.sin(d_lat_dropoff / 2) ** 2
    + np.cos(lat2) * np.cos(lat3) * np.sin(d_lon_dropoff / 2) ** 2
)
c2 = 2 * np.arctan2(np.sqrt(a2), np.sqrt(1 - a2))
distance2 = R * c2
test_data["Dropoff_Distance_airport"] = np.round(distance2 * 0.621, 2)


## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/510476878.py in <cell line: 0>()
      1 R = 6373.0
----> 2 lat1 = np.radians(train_data["pickup_latitude"])
      3 lon1 = np.radians(train_data["pickup_longitude"])
      4 lat2 = np.radians(train_data["dropoff_latitude"])
      5 lon2 = np.radians(train_data["dropoff_longitude"])

NameError: name 'np' is not defined

## === cell 22
train_data.shape


## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2272128234.py in <cell line: 0>()
----> 1 train_data.shape

NameError: name 'train_data' is not defined

## === cell 23
test_data.shape


## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4073087099.py in <cell line: 0>()
----> 1 test_data.shape

NameError: name 'test_data' is not defined

## === cell 24
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np

X = train_data.drop(["key", "fare_amount"], axis=1)
y = train_data["fare_amount"]

X_train_df, X_test_df, y_train, y_test = train_test_split(
    X, y, test_size=0.01, random_state=80
)

X_train = X_train_df.to_numpy(dtype=np.float32)
X_test = X_test_df.to_numpy(dtype=np.float32)
y_train = y_train.to_numpy(dtype=np.float32)
y_test = y_test.to_numpy(dtype=np.float32)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

gbr = GradientBoostingRegressor(
    n_estimators=100,  # same as original
    learning_rate=0.05,
    max_depth=4,
    random_state=42,
    loss="huber",
    subsample=0.8,
)
gbr.fit(X_train_scaled, y_train)
print("Training completed.")


## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3096672406.py in <cell line: 0>()
      4 import numpy as np
      5 
----> 6 X = train_data.drop(["key", "fare_amount"], axis=1)
      7 y = train_data["fare_amount"]
      8 

NameError: name 'train_data' is not defined

## === cell 25
from sklearn.metrics import mean_squared_error

val_pred = gbr.predict(X_test_scaled)
val_rmse = np.sqrt(mean_squared_error(y_test, val_pred))
print(f"Validation RMSE: {val_rmse:.4f}")

test_features = test_data.drop("key", axis=1).to_numpy(dtype=np.float32)
test_features_scaled = scaler.transform(test_features)
pred = np.round(gbr.predict(test_features_scaled), 2)


## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3639508794.py in <cell line: 0>()
      1 from sklearn.metrics import mean_squared_error
      2 
----> 3 val_pred = gbr.predict(X_test_scaled)
      4 val_rmse = np.sqrt(mean_squared_error(y_test, val_pred))
      5 print(f"Validation RMSE: {val_rmse:.4f}")

NameError: name 'gbr' is not defined

## === cell 26
pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/sample_submission.csv"
).head()


## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/18449898.py in <cell line: 0>()
----> 1 pd.read_csv(
      2     "/kaggle/input/new-york-city-taxi-fare-prediction/sample_submission.csv"
      3 ).head()

NameError: name 'pd' is not defined

## === cell 27
Submission = pd.DataFrame({"key": test_data["key"], "fare_amount": pred})
Submission = Submission[["key", "fare_amount"]]


## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3925564468.py in <cell line: 0>()
----> 1 Submission = pd.DataFrame({"key": test_data["key"], "fare_amount": pred})
      2 Submission = Submission[["key", "fare_amount"]]

NameError: name 'pd' is not defined

## === cell 28
Submission.to_csv("Submission.csv", index=False)

## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3446740269.py in <cell line: 0>()
----> 1 Submission.to_csv("Submission.csv", index=False)

NameError: name 'Submission' is not defined
