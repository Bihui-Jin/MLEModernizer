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
scipy==1.15.3
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

4.08766

# 6. Current score

5.35387

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 5.52155) has done: 'I replace the obsolete `Imputer` with `SimpleImputer`, ensure all libraries are imported before use, and reorganize the cells so every variable is defined before it is referenced. The core feature engineering (distance calculation) and model (GradientBoostingRegressor) remain unchanged, only the missing‑value handling and a modest increase in `n_estimators` are applied to push the RMSE a bit closer to the target while still producing a valid `submission.csv`.'
- What this solution (achieved 5.77251) has done: 'I add a few inexpensive datetime‑based features (hour, weekday, month) and a proper haversine distance column to the existing feature set. These extra predictors are known to help taxi‑fare models and can be incorporated without changing the core GradientBoostingRegressor workflow. The new columns are included in the training, validation, and test matrices, while the rest of the pipeline (imputation, model, CSV output) stays unchanged, so the script still runs end‑to‑end and should move the RMSE closer to the target.'
- What this solution (achieved 5.70216) has done: 'I add a lightweight standard‑scaling step after the imputation and increase the GradientBoostingRegressor capacity (more trees with a lower learning rate). These tweaks keep the same model type and feature set while typically reducing RMSE, moving the score closer to the target.'
- What this solution (achieved 14.56888) has done: 'The changes keep the same feature engineering and model type, but reduce the amount of data read for training and enable subsampling inside the GradientBoostingRegressor, which cuts the amount of work per tree while preserving the algorithm’s semantics. Both adjustments are deterministic and maintain the original pipeline’s logic and output format.'
- What this solution (achieved 5.35387) has done: 'The script is slowed mainly by loading 2 million rows and training a GradientBoosting model on ~1.4 million samples. By loading a smaller, still representative slice of the data (e.g., 500 k rows) we keep the same feature‑engineering and model‑definition logic while reducing memory movement and training time enough to stay under the 600 s limit. The change is limited to the `read_csv` call; all other steps, hyper‑parameters and processing remain unchanged.'

# 9. Code solution

## === cell 0
try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except Exception:
    pass

import os, math, pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

BASE_INPUT = "/kaggle/input" if pathlib.Path("/kaggle/input").exists() else "../input"
if not pathlib.Path(BASE_INPUT).exists() and pathlib.Path("./data").exists():
    BASE_INPUT = "./data"
print("Using input folder:", BASE_INPUT)
print("Folder contents:", os.listdir(BASE_INPUT))




## === cell 1
dtype_map = {
    "fare_amount": np.float32,
    "pickup_longitude": np.float32,
    "pickup_latitude": np.float32,
    "dropoff_longitude": np.float32,
    "dropoff_latitude": np.float32,
    "passenger_count": np.int8,
    "key": "object",
}
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
train_path = os.path.join(BASE_INPUT, "train.csv")
df = pd.read_csv(
    train_path,
    nrows=500_000,  # <-- reduced from 2_000_000 for faster runtime
    usecols=usecols,
    dtype=dtype_map,
    parse_dates=["pickup_datetime"],
    low_memory=False,
)

df = df[df.passenger_count > 0]
df = df[df.fare_amount > 0]

alpha_ang = 0.506


def add_features(df):
    dlon = (df["dropoff_longitude"] - df["pickup_longitude"]).abs().values * 50
    dlat = (df["dropoff_latitude"] - df["pickup_latitude"]).abs().values * 69

    df["abs_diff_longitude"] = dlon
    df["abs_diff_latitude"] = dlat

    displacement = np.sqrt(dlat**2 + dlon**2)
    df["displacement_vector"] = displacement

    angle = np.arctan(dlon / dlat) - alpha_ang
    df["actual_long"] = np.abs(displacement * np.sin(angle))
    df["actual_lat"] = np.abs(displacement * np.cos(angle))
    df["distance_travel"] = df["actual_long"] + df["actual_lat"]

    lon1 = np.radians(df["pickup_longitude"].values)
    lat1 = np.radians(df["pickup_latitude"].values)
    lon2 = np.radians(df["dropoff_longitude"].values)
    lat2 = np.radians(df["dropoff_latitude"].values)
    dlon_rad = lon2 - lon1
    dlat_rad = lat2 - lat1
    a = (
        np.sin(dlat_rad / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon_rad / 2) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(a))
    earth_radius_miles = 3956
    df["haversine_distance"] = earth_radius_miles * c

    df["distance_travel_sq"] = df["distance_travel"] ** 2
    df["haversine_distance_sq"] = df["haversine_distance"] ** 2

    if not pd.api.types.is_datetime64_any_dtype(df["pickup_datetime"]):
        df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
    df["pickup_hour"] = df["pickup_datetime"].dt.hour
    df["pickup_weekday"] = df["pickup_datetime"].dt.weekday
    df["pickup_month"] = df["pickup_datetime"].dt.month


add_features(df)

df = df[df.distance_travel > 0]
df = df[df.distance_travel < 30]  # realistic trips
df = df[df.fare_amount < 100]  # cap extreme fares




## === cell 2
l = len(df)
df_train = df.iloc[: int(0.7 * l)]
df_valid = df.iloc[int(0.7 * l) :]

feature_cols = [
    "distance_travel",
    "haversine_distance",
    "distance_travel_sq",
    "haversine_distance_sq",
    "passenger_count",
    "pickup_hour",
    "pickup_weekday",
    "pickup_month",
]

train_X = np.hstack(
    (
        df_train[feature_cols].to_numpy(dtype=np.float32),
        np.ones((len(df_train), 1), dtype=np.float32),
    )
)

valid_X = np.hstack(
    (
        df_valid[feature_cols].to_numpy(dtype=np.float32),
        np.ones((len(df_valid), 1), dtype=np.float32),
    )
)

train_y_log = np.log1p(df_train.fare_amount.values)
valid_y = df_valid.fare_amount.values




## === cell 3
imputer = SimpleImputer(strategy="mean")
train_X_imp = imputer.fit_transform(train_X)
valid_X_imp = imputer.transform(valid_X)

scaler = StandardScaler()
train_X_scaled = scaler.fit_transform(train_X_imp)
valid_X_scaled = scaler.transform(valid_X_imp)

regr = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.015,
    max_depth=5,
    subsample=1.0,
    random_state=42,
)
regr.fit(train_X_scaled, train_y_log)

pred_valid_log = regr.predict(valid_X_scaled)
pred_valid = np.expm1(pred_valid_log)
rmse = mean_squared_error(valid_y, pred_valid, squared=False)
print(f"Validation RMSE: {rmse:.4f}")




## === cell 4
test_path = os.path.join(BASE_INPUT, "test.csv")
test_usecols = [c for c in usecols if c != "fare_amount"]
test_dtype = {k: v for k, v in dtype_map.items() if k != "fare_amount"}

tdf = pd.read_csv(
    test_path,
    usecols=test_usecols,
    dtype=test_dtype,
    parse_dates=["pickup_datetime"],
    low_memory=False,
)
add_features(tdf)

test_X = np.hstack(
    (
        tdf[feature_cols].to_numpy(dtype=np.float32),
        np.ones((len(tdf), 1), dtype=np.float32),
    )
)

test_X_imp = imputer.transform(test_X)
test_X_scaled = scaler.transform(test_X_imp)
test_pred_log = regr.predict(test_X_scaled)
output = np.expm1(test_pred_log)




## === cell 5
submission = pd.DataFrame({"key": tdf["key"], "fare_amount": output})
submission_path = "/kaggle/working/submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
print(submission.head())
