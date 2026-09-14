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

5.48885

# 6. Current score

4.18534

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 4.18648) has done: 'I add the missing imports for pandas, numpy, and sklearn, and renumber the notebook cells so they start at 1. The code now loads the data, performs the cleaning and feature engineering steps, trains the RandomForest model, generates predictions, and writes a proper `submission_1.csv` file with the required columns. No other logic is changed, preserving the original modeling approach.'
- What this solution (achieved 4.18534) has done: 'Implemented targeted speed-ups while keeping the original modeling approach unchanged.

- Removed unnecessary float64 casts in the haversine computation to stay in float32, reducing memory and compute.
- Set `n_jobs=-1` for the RandomForest to utilize all available CPU cores, dramatically cutting training time.
- Added brief comments explaining each optimization.'

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

dtype_dict = {
    "key": "object",
    "fare_amount": "float32",
    "pickup_datetime": "object",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int8",
}
train = pd.read_csv(
    "../input/train.csv",
    nrows=1_000_000,
    dtype=dtype_dict,
    usecols=list(dtype_dict.keys()),
    low_memory=False,
)
test_dtype = {k: v for k, v in dtype_dict.items() if k != "fare_amount"}
test = pd.read_csv(
    "../input/test.csv",
    dtype=test_dtype,
    low_memory=False,
)



## === cell 1
print("train shape:", train.shape)
print("test shape:", test.shape)



## === cell 2
train = train.dropna(axis=0, how="any")



## === cell 3
train = train[train["fare_amount"] >= 0]



## === cell 4
train = train[train["passenger_count"].between(1, 8)]



## === cell 5
cond_lat = (
    (train["pickup_latitude"] < -90)
    | (train["pickup_latitude"] > 90)
    | (train["dropoff_latitude"] < -90)
    | (train["dropoff_latitude"] > 90)
)
cond_long = (
    (train["pickup_longitude"] < -180)
    | (train["pickup_longitude"] > 180)
    | (train["dropoff_longitude"] < -180)
    | (train["dropoff_longitude"] > 180)
)
train = train[~(cond_lat | cond_long)]



## === cell 6
train["pickup_datetime"] = pd.to_datetime(
    train["pickup_datetime"], infer_datetime_format=True
)
test["pickup_datetime"] = pd.to_datetime(
    test["pickup_datetime"], infer_datetime_format=True
)




## === cell 7
def haversine_distance(df, lat1, lon1, lat2, lon2):
    """
    Vectorised Haversine distance using float32 throughout.
    Avoids costly astype('float64') conversions present in the original code.
    """
    r = 6371.0
    phi1 = np.radians(df[lat1])
    phi2 = np.radians(df[lat2])
    delta_phi = np.radians(df[lat2] - df[lat1])
    delta_lambda = np.radians(df[lon2] - df[lon1])
    a = (
        np.sin(delta_phi / 2.0) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return r * c


train["H_Distance"] = haversine_distance(
    train,
    "pickup_latitude",
    "pickup_longitude",
    "dropoff_latitude",
    "dropoff_longitude",
)
test["H_Distance"] = haversine_distance(
    test,
    "pickup_latitude",
    "pickup_longitude",
    "dropoff_latitude",
    "dropoff_longitude",
)



## === cell 8
for df in [train, test]:
    df["Year"] = df["pickup_datetime"].dt.year
    df["Month"] = df["pickup_datetime"].dt.month
    df["Day"] = df["pickup_datetime"].dt.day
    df["DayOfWeek"] = df["pickup_datetime"].dt.dayofweek
    df["Hour"] = df["pickup_datetime"].dt.hour



## === cell 9
cond_zero = (
    (train["pickup_latitude"] == 0)
    & (train["pickup_longitude"] == 0)
    & (train["dropoff_latitude"] != 0)
    & (train["dropoff_longitude"] != 0)
    & (train["fare_amount"] == 0)
)
train = train[~cond_zero]



## === cell 10
train = train.dropna(axis=0, how="any")
test = test.dropna(axis=0, how="any")



## === cell 11
train = train.drop(["key", "pickup_datetime"], axis=1)
test = test.drop(["key", "pickup_datetime"], axis=1)

X_train = train.drop("fare_amount", axis=1)
y_train = train["fare_amount"].values
X_test = test



## === cell 12
rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=20,
    n_jobs=-1,  # changed from 5 to -1 for maximal parallelism
    random_state=42,
    min_samples_leaf=1,
)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)



## === cell 13
submission = pd.read_csv("../input/sample_submission.csv")
submission["fare_amount"] = rf_pred
submission.to_csv("submission_1.csv", index=False)
print("Submission saved to submission_1.csv")
