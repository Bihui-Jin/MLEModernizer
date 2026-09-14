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

folium==0.20.0
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

3.41572

# 6. Current score

4.85311

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 5.3998) has done: 'I keep the existing data loading, cleaning, and feature‑engineering steps unchanged, then add a lightweight gradient‑boosting model to evaluate RMSE on a held‑out split and finally train on all available data to generate predictions for the test set. I also reload the test file to retain the `key` column required for the submission, apply the same preprocessing, predict fares, and write a proper `submission.csv` file. This minimal addition ensures the notebook runs end‑to‑end and produces a valid Kaggle submission while moving the RMSE toward the target.'
- What this solution (achieved 5.43134) has done: 'I add a haversine distance feature to both the training and test data, because the straight‑line distance between pickup and drop‑off points is a strong predictor of fare and can noticeably lower RMSE with only a few extra lines. I compute this feature right after the temporal columns are created (cells 13 and 23) and keep the existing columns. I also increase the HistGradientBoostingRegressor `max_iter` to 400 to give the model a bit more capacity to capture the added information, which should improve validation RMSE and move the score closer to the target while preserving the overall pipeline.'
- What this solution (achieved 5.54211) has done: 'I added aggressive but safe outlier filtering (reasonable fare range, passenger count limits, and max distance) right after the distance feature is created, and I slightly increased the model capacity by raising `max_iter` to 800 and lowering the learning rate to 0.03. These changes keep the original pipeline intact while expected to lower the validation RMSE, moving it closer to the target score.'
- What this solution (achieved 5.06341) has done: 'I add a simple engineered feature (`distance_sq`) to give the model a non‑linear view of travel length, and I train the HistGradientBoostingRegressor on the log‑transformed target (`log1p(fare_amount)`). After predicting, I invert the transformation with `expm1` before computing RMSE and creating the submission. These lightweight tweaks keep the original pipeline intact while typically lowering the RMSE, moving the score nearer to the target.'
- What this solution (achieved 5.12079) has done: 'I added cyclical encodings for hour, month, and weekday (sine / cosine) and introduced a weekday feature derived from the original datetime. These extra time‑of‑day and day‑of‑week signals are cheap to compute, keep the existing pipeline untouched, and are known to improve tree‑based models for taxi‑fare data, moving the RMSE closer to the target. The new features are created for both training and test sets before dropping the raw datetime column, and the same columns are used when fitting and predicting.'
- What this solution (achieved 5.04309) has done: 'I added simple latitude/longitude delta features (Δlat, Δlon) and their Manhattan‑style absolute sum to give the model a direct sense of directional travel, then recomputed them for the test set. I also increased the tree depth to 10, the number of boosting iterations to 1000, and lowered the learning rate to 0.02, which provides a bit more capacity without altering the overall pipeline. These modest, targeted changes are expected to reduce the validation RMSE and move the score closer to the target while keeping the original workflow intact.'
- What this solution (achieved 4.93241) has done: 'We accelerate the heavy model training by enabling the Intel® oneAPI sklearn acceleration (sklearnex) and, more importantly, eliminate the duplicated reading and feature‑engineering of the test set. The test data is read once, its key column saved before it is dropped, and the already‑engineered feature matrix is reused for the final prediction, cutting a large amount of I/O and computation while keeping the exact same model and features.'
- What this solution (achieved 4.90363) has done: 'Implemented missing imports, corrected the data‑loading flow, ensured the key column is saved before dropping, added required sklearn utilities, and imported folium with its HeatMap plugin so the map cell works. No core logic was altered; the model and feature engineering remain unchanged, and the script now runs end‑to‑end and writes a valid `submission.csv`.'
- What this solution (achieved 4.85311) has done: 'The script now avoids the costly double‑fit of the Gradient Boosting model: it creates a single train/validation split only for evaluation, then fits the model once on the entire training set. This halves the training time while keeping the same model architecture, features, and prediction pipeline, ensuring identical final predictions and preserving result accuracy. Minor refactoring also moves the split creation before model definition for clarity.'

# 9. Code solution

## === cell 0
from sklearnex import patch_sklearn

patch_sklearn()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import folium
from folium.plugins import HeatMap

data_root = "/kaggle/input"



## === cell 1
fields = [
    "pickup_datetime",
    "fare_amount",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]

train = pd.read_csv(
    f"{data_root}/new-york-city-taxi-fare-prediction/train.csv",
    nrows=5_000_000,
    skipinitialspace=True,
    usecols=fields,
    parse_dates=["pickup_datetime"],
)
print(f"{train.shape} shape")
train.head()



## === cell 2
print(train.info())
train.describe()



## === cell 3
test = pd.read_csv(
    f"{data_root}/new-york-city-taxi-fare-prediction/test.csv",
    parse_dates=["pickup_datetime"],
)
print(f"{test.shape} shape")
test.head()



## === cell 4
print(test.info())
test.describe()



## === cell 5
test_key = test["key"].copy()
test = test.drop(columns=["key"])



## === cell 6
sns.countplot(x="passenger_count", data=train)



## === cell 7
train = train.drop(train[train["passenger_count"] == 208].index)



## === cell 8
train = train.drop(train[train["fare_amount"] < 0].index)



## === cell 9
train = train[
    (train["passenger_count"] >= 1)
    & (train["passenger_count"] <= 6)
    & (train["fare_amount"] > 0)
    & (train["fare_amount"] < 200)
    & (train["pickup_latitude"] > 40)
    & (train["pickup_latitude"] < 45)
    & (train["dropoff_latitude"] > 40)
    & (train["dropoff_latitude"] < 45)
    & (train["pickup_longitude"] < -71)
    & (train["pickup_longitude"] > -79)
    & (train["dropoff_longitude"] < -71)
    & (train["dropoff_longitude"] > -79)
].copy()

train["year"] = train["pickup_datetime"].dt.year
train["month"] = train["pickup_datetime"].dt.month
train["day"] = train["pickup_datetime"].dt.day
train["hour"] = train["pickup_datetime"].dt.hour
train["minute"] = train["pickup_datetime"].dt.minute
train["weekday"] = train["pickup_datetime"].dt.weekday

test["year"] = test["pickup_datetime"].dt.year
test["month"] = test["pickup_datetime"].dt.month
test["day"] = test["pickup_datetime"].dt.day
test["hour"] = test["pickup_datetime"].dt.hour
test["minute"] = test["pickup_datetime"].dt.minute
test["weekday"] = test["pickup_datetime"].dt.weekday


def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0  # Earth radius in km
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = (
        np.sin(dphi / 2.0) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    )
    return 2 * R * np.arcsin(np.sqrt(a))


train["distance"] = haversine(
    train["pickup_latitude"],
    train["pickup_longitude"],
    train["dropoff_latitude"],
    train["dropoff_longitude"],
)

test["distance"] = haversine(
    test["pickup_latitude"],
    test["pickup_longitude"],
    test["dropoff_latitude"],
    test["dropoff_longitude"],
)

train["distance_log"] = np.log1p(train["distance"]).astype(np.float32)
test["distance_log"] = np.log1p(test["distance"]).astype(np.float32)

train["delta_lat"] = train["dropoff_latitude"] - train["pickup_latitude"]
train["delta_lon"] = train["dropoff_longitude"] - train["pickup_longitude"]
train["manhattan_delta"] = np.abs(train["delta_lat"]) + np.abs(train["delta_lon"])

test["delta_lat"] = test["dropoff_latitude"] - test["pickup_latitude"]
test["delta_lon"] = test["dropoff_longitude"] - test["pickup_longitude"]
test["manhattan_delta"] = np.abs(test["delta_lat"]) + np.abs(test["delta_lon"])

train["distance_sq"] = train["distance"] ** 2
test["distance_sq"] = test["distance"] ** 2

for df in [train, test]:
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7)
    df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7)

train = train[train["distance"] < 100]
test.loc[test["distance"] > 100, "distance"] = 100

train_float_cols = train.select_dtypes(include=["float64"]).columns.tolist()
test_float_cols = test.select_dtypes(include=["float64"]).columns.tolist()

train[train_float_cols] = train[train_float_cols].astype(np.float32)
test[test_float_cols] = test[test_float_cols].astype(np.float32)



## === cell 10
train = train.drop(columns=["pickup_datetime"])
test = test.drop(columns=["pickup_datetime"])



## === cell 11
train.describe()



## === cell 12
variables = ["year", "month", "day", "hour"]
for var in variables:
    plt.figure()
    sns.countplot(x=var, data=train).set(title=f"Count plot of {var}")



## === cell 13
ny_map = folium.Map(location=[40.71, -74.00], tiles="cartodbpositron", zoom_start=12)
HeatMap(data=train[["pickup_latitude", "pickup_longitude"]], radius=10).add_to(ny_map)
ny_map



## === cell 14
X = train.drop(columns=["fare_amount"]).astype(np.float32).values
y = train["fare_amount"].values.astype(np.float32)
y_log = np.log1p(y).astype(np.float32)

X_train, X_val, y_train_log, y_val_log = train_test_split(
    X, y_log, test_size=0.1, random_state=42
)

from sklearn.ensemble import HistGradientBoostingRegressor

model = HistGradientBoostingRegressor(
    max_depth=10,
    learning_rate=0.02,
    max_iter=2000,
    random_state=42,
)

model.fit(X, y_log)

val_pred_log = model.predict(X_val)
val_pred = np.expm1(val_pred_log)
val_rmse = mean_squared_error(np.expm1(y_val_log), val_pred, squared=False)
print(f"Validation RMSE: {val_rmse:.5f}")

test_features = test.astype(np.float32).values
test_pred_log = model.predict(test_features)
test_pred = np.expm1(test_pred_log)

submission = pd.DataFrame({"key": test_key, "fare_amount": test_pred})
output_path = "/kaggle/working/submission.csv"
submission.to_csv(output_path, index=False)
print(f"Submission file written to {output_path}")
