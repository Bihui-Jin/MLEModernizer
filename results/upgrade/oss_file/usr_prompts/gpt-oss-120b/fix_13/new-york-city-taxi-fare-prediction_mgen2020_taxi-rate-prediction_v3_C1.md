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

8.2434

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 936.92806) has done: 'I remove the deprecated `normalize` argument from `LinearRegression`, ensure the model variable persists for prediction, and adjust the submission writing step to output a proper CSV with the required columns and no index column. This resolves the runtime errors and creates a valid `submission.csv` file.'
- What this solution (achieved 7.98509) has done: 'Implemented a lightweight acceleration by patching scikit‑learn with Intel® ‑ oneAPI optimizations before the RandomForest model is instantiated. This change does not alter any algorithmic parameters, data preprocessing, or evaluation logic, but it significantly speeds up tree construction and prediction, keeping the entire pipeline within the 600 s limit.'
- What this solution (achieved 7.0889) has done: 'I filter out any rows with non‑positive fares (which cause NaNs after the log 1p transform) and guard against infinities that can appear when normalising the longitude/latitude differences. These small fixes ensure the target vector is finite and the feature matrix contains only valid numbers, allowing the RandomForest to fit and the script to generate a proper `submission.csv`.'
- What this solution (achieved 8.10079) has done: 'I add a simple cyclical time feature (hour sin / cos) to give the model a better sense of daily patterns, and I slightly adjust the RandomForest hyper‑parameters (more trees, use `sqrt` features and a minimum leaf size of 2) which typically reduces over‑fitting and improves RMSE without changing the overall model architecture. These minimal changes keep the original pipeline intact while steering the validation error closer to the target 5.689.'
- What this solution (achieved 8.2434) has done: 'I adjust two small preprocessing and model settings that are likely to reduce over‑fitting and give a cleaner feature scale.  
1. Replace the custom “abs‑mean‑over‑variance” scaling of the longitude/latitude differences with standard‑score scaling (mean / std) – a more appropriate normalization.  
2. Limit the RandomForest depth (e.g., `max_depth=20`) to curb excessive tree growth while keeping all other hyper‑parameters unchanged. These minimal tweaks keep the overall pipeline intact but should move the validation RMSE closer to the target 5.689.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))




## === cell 1
usecols = [
    "key",
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
dtypes = {
    "key": "object",
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int8",
}
td = pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv",
    usecols=usecols,
    dtype=dtypes,
    nrows=10_000_000,
)
if len(td) > 2_000_000:
    td = td.sample(frac=2_000_000 / len(td), random_state=42).reset_index(drop=True)
td.head()




## === cell 2
td.shape




## === cell 3
td.info()




## === cell 4
test_usecols = [col for col in usecols if col != "fare_amount"]
test_dtypes = {k: v for k, v in dtypes.items() if k != "fare_amount"}

ted = pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv",
    usecols=test_usecols,  # test does not have fare_amount
    dtype=test_dtypes,
)
ted.head()




## === cell 5
ted.info()




## === cell 6
td.isna().sum()




## === cell 7
td["Difference_longitude"] = np.abs(td["pickup_longitude"] - td["dropoff_longitude"])
td["Difference_latitude"] = np.abs(td["pickup_latitude"] - td["dropoff_latitude"])
ted["Difference_longitude"] = np.abs(ted["pickup_longitude"] - ted["dropoff_longitude"])
ted["Difference_latitude"] = np.abs(ted["pickup_latitude"] - ted["dropoff_latitude"])




## === cell 8
print(f"Before Dropping null values: {len(td)}")
td.dropna(inplace=True)
td = td[td["fare_amount"] > 0].reset_index(drop=True)
print(f"After Dropping null and non‑positive fare values: {len(td)}")




## === cell 9
plot = td[:2000].plot.scatter("Difference_longitude", "Difference_latitude")




## === cell 10
td = td[(td["Difference_longitude"] < 5.0) & (td["Difference_latitude"] < 5.0)]




## === cell 11
td["pickup_datetime"] = pd.to_datetime(td["pickup_datetime"])
td["pickuptime"] = td["pickup_datetime"].dt.hour * 100 + td["pickup_datetime"].dt.minute
td["Weekday"] = td["pickup_datetime"].dt.weekday.map(
    {
        0: "Monday",
        1: "Tuesday",
        2: "Wednesday",
        3: "Thursday",
        4: "Friday",
        5: "Saturday",
        6: "Sunday",
    }
)

ted["pickup_datetime"] = pd.to_datetime(ted["pickup_datetime"])
ted["pickuptime"] = (
    ted["pickup_datetime"].dt.hour * 100 + ted["pickup_datetime"].dt.minute
)
ted["Weekday"] = ted["pickup_datetime"].dt.weekday.map(
    {
        0: "Monday",
        1: "Tuesday",
        2: "Wednesday",
        3: "Thursday",
        4: "Friday",
        5: "Saturday",
        6: "Sunday",
    }
)


def add_cyclical_time(df):
    df["hour"] = df["pickuptime"] // 100
    df["minute"] = df["pickuptime"] % 100
    df["hour_frac"] = df["hour"] + df["minute"] / 60.0
    df["hour_sin"] = np.sin(2 * np.pi * df["hour_frac"] / 24.0).astype(np.float32)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour_frac"] / 24.0).astype(np.float32)
    return df


td = add_cyclical_time(td)
ted = add_cyclical_time(ted)




## === cell 12
td.head()




## === cell 13
td.head()




## === cell 14
ted.head()




## === cell 15
td.drop("pickup_datetime", inplace=True, axis=1)
ted.drop("pickup_datetime", inplace=True, axis=1)

th = pd.get_dummies(td["Weekday"], dtype=np.float32)
teh = pd.get_dummies(ted["Weekday"], dtype=np.float32)

td = pd.concat([td, th], axis=1)
ted = pd.concat([ted, teh], axis=1)

td.drop("Weekday", axis=1, inplace=True)
ted.drop("Weekday", axis=1, inplace=True)

td.head()




## === cell 16
def add_haversine_features(df):
    R = 6373.0
    lat1 = np.radians(df["pickup_latitude"].values.astype(np.float32))
    lon1 = np.radians(df["pickup_longitude"].values.astype(np.float32))
    lat2 = np.radians(df["dropoff_latitude"].values.astype(np.float32))
    lon2 = np.radians(df["dropoff_longitude"].values.astype(np.float32))

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    df["Distance"] = (R * c * 0.621).round(2)  # miles

    lat3 = np.full(len(df), np.radians(40.6413111), dtype=np.float32)
    lon3 = np.full(len(df), np.radians(-73.7781391), dtype=np.float32)

    dlon_pickup = lon3 - lon1
    dlat_pickup = lat3 - lat1
    a1 = (
        np.sin(dlat_pickup / 2) ** 2
        + np.cos(lat1) * np.cos(lat3) * np.sin(dlon_pickup / 2) ** 2
    )
    df["Pickup_Distance_airport"] = (
        R * 2 * np.arctan2(np.sqrt(a1), np.sqrt(1 - a1)) * 0.621
    ).round(2)

    dlon_dropoff = lon3 - lon2
    dlat_dropoff = lat3 - lat2
    a2 = (
        np.sin(dlon_dropoff / 2) ** 2
        + np.cos(lat2) * np.cos(lat3) * np.sin(dlon_dropoff / 2) ** 2
    )
    df["Dropoff_Distance_airport"] = (
        R * 2 * np.arctan2(np.sqrt(a2), np.sqrt(1 - a2)) * 0.621
    ).round(2)

    df.drop(
        [
            "pickup_longitude",
            "pickup_latitude",
            "dropoff_longitude",
            "dropoff_latitude",
        ],
        axis=1,
        inplace=True,
    )
    return df


td = add_haversine_features(td)
ted = add_haversine_features(ted)

for col in ["Difference_longitude", "Difference_latitude"]:
    td[col] = (td[col] - td[col].mean()) / td[col].std()
    ted[col] = (ted[col] - ted[col].mean()) / ted[col].std()

td.replace([np.inf, -np.inf], np.nan, inplace=True)
ted.replace([np.inf, -np.inf], np.nan, inplace=True)

td.shape




## === cell 17
ted.shape




## === cell 18
from sklearnex import patch_sklearn

patch_sklearn()

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

X = td.drop(["key", "fare_amount"], axis=1).astype(np.float32)
y = td["fare_amount"].astype(np.float32)

mask = np.isfinite(y) & (y > 0)
X = X[mask]
y = y[mask]

y_log = np.log1p(y).astype(np.float32)

median_vals = X.median()
X = X.fillna(median_vals)

X_train, X_val, y_train_log, y_val = train_test_split(
    X, y_log, test_size=0.01, random_state=80
)

rf = RandomForestRegressor(
    n_estimators=400,  # more trees for stability
    max_depth=20,  # limit depth to reduce over‑fitting
    max_features="sqrt",  # reduce variance
    min_samples_leaf=2,  # prevent over‑fitting tiny leaves
    n_jobs=5,
    random_state=42,
)
rf.fit(X_train, y_train_log)

val_pred_log = rf.predict(X_val)
val_pred = np.expm1(val_pred_log)  # back to original scale
rmse = np.sqrt(mean_squared_error(y_val, val_pred))
print(f"Validation RMSE: {rmse:.4f}")




## === cell 19
test_features = ted.drop("key", axis=1).astype(np.float32)
test_features = test_features.fillna(median_vals)  # use training medians
test_pred_log = rf.predict(test_features)
pred = np.expm1(test_pred_log)
pred = np.round(pred, 2)

print(pred[:5])




## === cell 20
Submission = pd.DataFrame({"key": ted["key"], "fare_amount": pred})
Submission.to_csv("submission.csv", index=False)
