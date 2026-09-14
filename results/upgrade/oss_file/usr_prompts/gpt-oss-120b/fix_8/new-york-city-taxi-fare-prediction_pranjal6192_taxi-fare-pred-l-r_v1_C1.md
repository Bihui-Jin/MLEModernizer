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

4.71563

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 936.92806) has done: 'I remove the obsolete `normalize=True` argument from the `LinearRegression` constructor (it caused a TypeError) so the model can be trained, which also restores the downstream variables (`lr`, `pred`, `Submission`) needed for creating a valid CSV submission.'
- What this solution (achieved 936.92806) has done: 'I replace the use of `LinearRegression.score` (R²) with a proper RMSE calculation, because the competition evaluates RMSE. Computing the correct metric move the reported score from the huge 936 value to a realistic value near the target (≈5‑6). This change only affects the evaluation line and adds the necessary import.'
- What this solution (achieved 4.71563) has done: 'The script failed because the test set was read without the required **key** column, causing a KeyError when creating the submission. I updated the data loading step to retain the **key** column while still applying the specified dtypes to the numeric features. This restores the key for the final DataFrame and allows the submission CSV to be generated correctly.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error




## === cell 1
base_path = "/kaggle/input"
train_path = os.path.join(base_path, "new-york-city-taxi-fare-prediction", "train.csv")
test_path = os.path.join(base_path, "new-york-city-taxi-fare-prediction", "test.csv")

dtype_train = {
    "fare_amount": np.float32,
    "pickup_longitude": np.float32,
    "pickup_latitude": np.float32,
    "dropoff_longitude": np.float32,
    "dropoff_latitude": np.float32,
    "passenger_count": np.int8,
}
train_data = pd.read_csv(
    train_path, nrows=5_000_000, dtype=dtype_train, usecols=lambda c: c != "key"
)
test_data = pd.read_csv(
    test_path, dtype=dtype_train  # key column is read automatically as object
)




## === cell 2
train_data.dropna(inplace=True)
test_data.dropna(inplace=True)  # precautionary




## === cell 3
def add_features(df):
    df["pickup_dt"] = pd.to_datetime(df["pickup_datetime"])
    df["pickuptime"] = df["pickup_dt"].dt.hour * 100 + df["pickup_dt"].dt.minute
    df["Weekday"] = df["pickup_dt"].dt.weekday

    R = 6373.0  # Earth radius in km
    lat1 = np.radians(df["pickup_latitude"])
    lon1 = np.radians(df["pickup_longitude"])
    lat2 = np.radians(df["dropoff_latitude"])
    lon2 = np.radians(df["dropoff_longitude"])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance_km = R * c
    df["Distance"] = distance_km * 0.621371  # miles

    lat_air = np.radians(40.6413111)
    lon_air = np.radians(-73.7781391)

    dlon_pa = lon_air - lon1
    dlat_pa = lat_air - lat1
    a_pa = (
        np.sin(dlat_pa / 2) ** 2
        + np.cos(lat1) * np.cos(lat_air) * np.sin(dlon_pa / 2) ** 2
    )
    c_pa = 2 * np.arctan2(np.sqrt(a_pa), np.sqrt(1 - a_pa))
    df["Pickup_Distance_airport"] = (R * c_pa) * 0.621371

    dlon_da = lon_air - lon2
    dlat_da = lat_air - lat2
    a_da = (
        np.sin(dlat_da / 2) ** 2
        + np.cos(lat2) * np.cos(lat_air) * np.sin(dlon_da / 2) ** 2
    )
    c_da = 2 * np.arctan2(np.sqrt(a_da), np.sqrt(1 - a_da))
    df["Dropoff_Distance_airport"] = (R * c_da) * 0.621371

    weekday_dummies = pd.get_dummies(df["Weekday"], prefix="wd")
    df = pd.concat([df, weekday_dummies], axis=1)

    drop_cols = [
        "pickup_datetime",
        "pickup_dt",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "Weekday",
    ]
    df.drop(columns=drop_cols, inplace=True, errors="ignore")
    return df


train_data = add_features(train_data)
test_data = add_features(test_data)




## === cell 4
feature_cols = [c for c in train_data.columns if c not in ["key", "fare_amount"]]

X_test = test_data.reindex(columns=feature_cols, fill_value=0)




## === cell 5
X = train_data[feature_cols]
y = train_data["fare_amount"]

X_sample, _, y_sample, _ = train_test_split(
    X, y, train_size=1_000_000, random_state=42, stratify=None
)

X_train, X_val, y_train, y_val = train_test_split(
    X_sample, y_sample, test_size=0.01, random_state=80
)

gbr = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=3,
    subsample=0.8,
    random_state=42,
)
gbr.fit(X_train, y_train)

val_pred = gbr.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, val_pred))
print(f"Validation RMSE: {rmse:.5f}")




## === cell 6
test_pred = np.round(gbr.predict(X_test), 2)




## === cell 7
submission = pd.DataFrame({"key": test_data["key"], "fare_amount": test_pred})




## === cell 8
submission.to_csv("Submission.csv", index=False)
print("Submission file 'Submission.csv' written.")
