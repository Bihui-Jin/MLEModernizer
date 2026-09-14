# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import pathlib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
import gc


def load_csv(filename):
    paths = [
        pathlib.Path(filename),
        pathlib.Path("/kaggle/input") / filename,
        pathlib.Path("/kaggle/working") / filename,
        pathlib.Path("data") / filename,
    ]
    dtype_map = {
        "fare_amount": "float32",
        "pickup_longitude": "float32",
        "pickup_latitude": "float32",
        "dropoff_longitude": "float32",
        "dropoff_latitude": "float32",
        "passenger_count": "int8",
    }
    for p in paths:
        if p.is_file():
            return pd.read_csv(
                p,
                low_memory=False,
                dtype=dtype_map,
                parse_dates=["pickup_datetime"],
                infer_datetime_format=True,
            )
    raise FileNotFoundError(f"{filename} not found")


train_df = load_csv("train.csv")
test_df = load_csv("test.csv")

train_df = train_df[(train_df["fare_amount"] > 0) & (train_df["fare_amount"] < 300)]

train_df = train_df.sample(frac=0.10, random_state=42).reset_index(drop=True)




## === cell 1
def add_travel_vector_features(df):
    df["abs_diff_longitude"] = (df["dropoff_longitude"] - df["pickup_longitude"]).abs()
    df["abs_diff_latitude"] = (df["dropoff_latitude"] - df["pickup_latitude"]).abs()
    lon1 = np.radians(df["pickup_longitude"].values.astype(np.float32))
    lat1 = np.radians(df["pickup_latitude"].values.astype(np.float32))
    lon2 = np.radians(df["dropoff_longitude"].values.astype(np.float32))
    lat2 = np.radians(df["dropoff_latitude"].values.astype(np.float32))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    earth_radius_km = np.float32(6371.0)
    df["haversine_km"] = earth_radius_km * c
    df["haversine_km_sq"] = df["haversine_km"] ** 2
    for col in [
        "abs_diff_longitude",
        "abs_diff_latitude",
        "haversine_km",
        "haversine_km_sq",
    ]:
        df[col] = df[col].astype(np.float32, copy=False)


def add_datetime_features(df):
    dt = df["pickup_datetime"]
    df["pickup_hour"] = dt.dt.hour.astype(np.int8)
    df["pickup_dayofweek"] = dt.dt.dayofweek.astype(np.int8)
    df["pickup_month"] = dt.dt.month.astype(np.int8)


for d in (train_df, test_df):
    add_travel_vector_features(d)
    add_datetime_features(d)



## === cell 2
coords_to_drop = ["pickup_datetime"]
X = train_df.drop(columns=["key", "fare_amount"] + coords_to_drop)
y = train_df["fare_amount"]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.01, random_state=42)

del train_df
gc.collect()



## === cell 3
model = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler(),
    Ridge(alpha=1.0, random_state=42, solver="sag", n_jobs=-1, max_iter=100),
)

X_train_np = X_train.to_numpy(dtype=np.float32, copy=False)
X_val_np = X_val.to_numpy(dtype=np.float32, copy=False)

model.fit(X_train_np, y_train.to_numpy(dtype=np.float32, copy=False))
val_pred = model.predict(X_val_np)
rmse = mean_squared_error(y_val, val_pred, squared=False)
print(f"Hold‑out RMSE: {rmse:.4f}")



## === cell 4
test_features = test_df.drop(columns=["key"] + coords_to_drop, errors="ignore")
missing = set(X.columns) - set(test_features.columns)
for col in missing:
    test_features[col] = 0
test_features = test_features[X.columns]

test_features_np = test_features.to_numpy(dtype=np.float32, copy=False)

test_pred = np.round(model.predict(test_features_np), 2)
test_pred = np.where(test_pred < 0, 0, test_pred)



## === cell 5
submission = pd.DataFrame({"key": test_df["key"], "fare_amount": test_pred})
submission.to_csv("submission.csv", index=False)
print("Saved submission.csv with", len(submission), "rows.")
