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

5.689

# 6. Current score

10.3447

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 936.92806) has done: 'The fixes address three runtime errors: (1) `LinearRegression` no longer accepts the `normalize` argument, so it’s removed; (2) the model variable `lr` was undefined in later cells because the previous error stopped execution, now it runs correctly; (3) the submission file was written with the index as the key column, which mismatches the required format, so we keep `key` as a regular column and write it to `submission.csv`. An optional clipping of negative predictions is added to keep fares realistic, which can modestly improve the RMSE.'
- What this solution (achieved 936.72428) has done: 'I replace the plain LinearRegression with a small preprocessing‑plus‑Ridge pipeline, which keeps the original linear‑model approach but adds scaling and regularisation. This should dramatically lower the RMSE (moving the score much closer to the target) while preserving the overall workflow and output format.'
- What this solution (achieved 762.43239) has done: 'I remove the custom “mean‑absolute‑variance” scaling applied to the longitude/latitude difference features, because it inflates those columns and harms the linear model. The StandardScaler in the Ridge pipeline already normalizes all features appropriately, so keeping the raw difference values gives a much more sensible feature scale and should lower the RMSE toward the target. No other logic is changed.'
- What this solution (achieved 5.06202) has done: 'The updates vectorize all datetime‑related loops (pick‑up time extraction, weekday computation, and time‑to‑int conversion) using pandas datetime accessors, and enable Intel® Extension for Scikit‑Learn to accelerate the RandomForest training. These changes eliminate costly Python for‑loops over millions of rows while keeping every derived feature and model hyper‑parameter identical, so the resulting predictions and validation RMSE remain unchanged but the overall runtime drops well below the 600‑second limit.'
- What this solution (achieved 14.77858) has done: 'The fix updates the file paths to the correct dataset location, aligns one‑hot‑encoded weekday columns between train and test, and adds a safety check for missing dummy columns. These changes let the pipeline load the data, generate features, train the RandomForest model, and write a proper `submission.csv` without runtime errors, moving the RMSE toward the target.'
- What this solution (achieved 12.13573) has done: 'I replace the RandomForest with a faster, more accurate HistGradientBoostingRegressor (still a tree‑based model) and keep the same log‑target handling. The change is limited to the model definition/training cell, preserving all feature engineering and preprocessing steps, and should lower the validation RMSE toward the target while still producing a correct `submission.csv`.'
- What this solution (achieved 10.46733) has done: 'I keep the overall pipeline unchanged but add the original latitude/longitude coordinates back as features (they were previously dropped) and give the HistGradientBoostingRegressor a few more boosting iterations (max_iter = 500). Keeping more geometric features typically improves tree‑based models, and a modest increase in iterations lets the model fit the data better, both of which should lower the RMSE and move it toward the target value.'
- What this solution (achieved 10.35777) has done: 'I added a few inexpensive features that are known to help fare prediction while keeping the overall pipeline unchanged.  
1. In the weekday processing step I create a binary `IsWeekend` column (1 for Saturday/Sunday) before dropping the original weekday string.  
2. In the time‑feature step I also extract the separate `Hour` and `Minute` columns from the datetime, keeping the existing `pickuptime` representation.  
These extra numeric features give the model more granular temporal information and a simple weekend indicator, which should modestly lower the RMSE and move the score closer to the target without altering the core model or training logic.'
- What this solution (achieved 10.3447) has done: 'I fixed the file‑path errors that prevented the data from loading and consequently stopped the whole pipeline. The script now defines a `DATA_ROOT` pointing to the correct competition folder (`new‑york‑city‑taxi‑fare‑prediction`) and uses it for loading both `train.csv` and `test.csv`. All later cells operate on the correctly loaded DataFrames, so the feature engineering, model training, and submission generation run without raising errors, producing a valid `submission.csv` file.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os

DATA_ROOT = "/kaggle/input/new-york-city-taxi-fare-prediction"

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))




## === cell 1
train_data = pd.read_csv(os.path.join(DATA_ROOT, "train.csv"), nrows=2_000_000)
train_data.head()




## === cell 2
train_data.shape




## === cell 3
train_data.info()




## === cell 4
test_data = pd.read_csv(os.path.join(DATA_ROOT, "test.csv"))
test_data.head()




## === cell 5
test_data.info()




## === cell 6
train_data.isna().sum()




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




## === cell 8
print(f"Before Dropping null values: {len(train_data)}")
train_data.dropna(inplace=True)
print(f"After Dropping null values: {len(train_data)}")




## === cell 9
train_data = train_data[
    (train_data["Difference_longitude"] < 5.0)
    & (train_data["Difference_latitude"] < 5.0)
]




## === cell 10
train_data["pickuptime_dt"] = pd.to_datetime(train_data["pickup_datetime"])
test_data["pickuptime_dt"] = pd.to_datetime(test_data["pickup_datetime"])
train_data.head()




## === cell 11
train_data["Weekday"] = pd.to_datetime(train_data["pickup_datetime"]).dt.weekday
test_data["Weekday"] = pd.to_datetime(test_data["pickup_datetime"]).dt.weekday




## === cell 12
train_data.head()




## === cell 13
test_data.head()




## === cell 14
train_data.drop("pickup_datetime", inplace=True, axis=1)
test_data.drop("pickup_datetime", inplace=True, axis=1)




## === cell 15
train_data["Weekday"].replace(
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
test_data["Weekday"].replace(
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




## === cell 16
train_one_hot = pd.get_dummies(train_data["Weekday"], prefix="Weekday")
test_one_hot = pd.get_dummies(test_data["Weekday"], prefix="Weekday")
test_one_hot = test_one_hot.reindex(columns=train_one_hot.columns, fill_value=0)

train_data["IsWeekend"] = train_data["Weekday"].isin(["Saturday", "Sunday"]).astype(int)
test_data["IsWeekend"] = test_data["Weekday"].isin(["Saturday", "Sunday"]).astype(int)

train_data = pd.concat([train_data, train_one_hot], axis=1)
test_data = pd.concat([test_data, test_one_hot], axis=1)
train_data.drop("Weekday", axis=1, inplace=True)
test_data.drop("Weekday", axis=1, inplace=True)




## === cell 17
train_data["Hour"] = train_data["pickuptime_dt"].dt.hour
train_data["Minute"] = train_data["pickuptime_dt"].dt.minute
test_data["Hour"] = test_data["pickuptime_dt"].dt.hour
test_data["Minute"] = test_data["pickuptime_dt"].dt.minute

train_data["pickuptime"] = (
    train_data["pickuptime_dt"].dt.hour * 100 + train_data["pickuptime_dt"].dt.minute
)
test_data["pickuptime"] = (
    test_data["pickuptime_dt"].dt.hour * 100 + test_data["pickuptime_dt"].dt.minute
)
train_data.drop(columns=["pickuptime_dt"], inplace=True)
test_data.drop(columns=["pickuptime_dt"], inplace=True)




## === cell 18
train_data.head()




## === cell 19
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
train_data["Distance"] = distance * 0.621  # miles




## === cell 20
lat1 = np.radians(test_data["pickup_latitude"])
lon1 = np.radians(test_data["pickup_longitude"])
lat2 = np.radians(test_data["dropoff_latitude"])
lon2 = np.radians(test_data["dropoff_longitude"])

dlon = lon2 - lon1
dlat = lat2 - lat1
a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
distance = R * c
test_data["Distance"] = distance * 0.621




## === cell 21
R = 6373.0
lat1 = np.radians(train_data["pickup_latitude"])
lon1 = np.radians(train_data["pickup_longitude"])
lat2 = np.radians(train_data["dropoff_latitude"])
lon2 = np.radians(train_data["dropoff_longitude"])

lat3 = np.full(len(train_data), np.radians(40.6413111))  # JFK Airport lat
lon3 = np.full(len(train_data), np.radians(-73.7781391))  # JFK Airport lon

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
train_data["Pickup_Distance_airport"] = distance1 * 0.621




## === cell 22
a2 = (
    np.sin(d_lat_dropoff / 2) ** 2
    + np.cos(lat2) * np.cos(lat3) * np.sin(d_lon_dropoff / 2) ** 2
)
c2 = 2 * np.arctan2(np.sqrt(a2), np.sqrt(1 - a2))
distance2 = R * c2
train_data["Dropoff_Distance_airport"] = distance2 * 0.621




## === cell 23
lat1 = np.radians(test_data["pickup_latitude"])
lon1 = np.radians(test_data["pickup_longitude"])
lat2 = np.radians(test_data["dropoff_latitude"])
lon2 = np.radians(test_data["dropoff_longitude"])

lat3 = np.full(len(test_data), np.radians(40.6413111))
lon3 = np.full(len(test_data), np.radians(-73.7781391))

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
test_data["Pickup_Distance_airport"] = distance1 * 0.621

a2 = (
    np.sin(d_lat_dropoff / 2) ** 2
    + np.cos(lat2) * np.cos(lat3) * np.sin(d_lon_dropoff / 2) ** 2
)
c2 = 2 * np.arctan2(np.sqrt(a2), np.sqrt(1 - a2))
distance2 = R * c2
test_data["Dropoff_Distance_airport"] = distance2 * 0.621




## === cell 24
train_data["Distance"] = np.round(train_data["Distance"], 2)
train_data["Pickup_Distance_airport"] = np.round(
    train_data["Pickup_Distance_airport"], 2
)
train_data["Dropoff_Distance_airport"] = np.round(
    train_data["Dropoff_Distance_airport"], 2
)

test_data["Distance"] = np.round(test_data["Distance"], 2)
test_data["Pickup_Distance_airport"] = np.round(test_data["Pickup_Distance_airport"], 2)
test_data["Dropoff_Distance_airport"] = np.round(
    test_data["Dropoff_Distance_airport"], 2
)




## === cell 25
train_data.shape




## === cell 26
test_data.shape




## === cell 27
from sklearnex import patch_sklearn

patch_sklearn()
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import HistGradientBoostingRegressor

train_data = train_data[train_data["fare_amount"] > 0].copy()

feature_cols = train_data.columns.difference(["key", "fare_amount"])
X = train_data[feature_cols].fillna(0).astype(np.float32).values
y_log = np.log1p(train_data["fare_amount"].values)

X_train, X_valid, y_train_log, y_valid_log = train_test_split(
    X, y_log, test_size=0.01, random_state=80
)

hgb = HistGradientBoostingRegressor(
    max_depth=12,
    learning_rate=0.05,
    max_iter=800,
    l2_regularization=0.1,
    random_state=42,
)

hgb.fit(X_train, y_train_log)

valid_pred_log = hgb.predict(X_valid)
valid_pred = np.expm1(valid_pred_log)
valid_true = np.expm1(y_valid_log)

rmse = np.sqrt(mean_squared_error(valid_true, valid_pred))
print(f"Validation RMSE: {rmse:.4f}")




## === cell 28
test_features = test_data.drop("key", axis=1).fillna(0).astype(np.float32).values

test_pred_log = hgb.predict(test_features)
test_pred = np.expm1(test_pred_log)

pred = np.round(test_pred, 2)
pred = np.clip(pred, a_min=0, a_max=None)




## === cell 29
submission = pd.DataFrame({"key": test_data["key"], "fare_amount": pred})
submission.to_csv("submission.csv", index=False)
