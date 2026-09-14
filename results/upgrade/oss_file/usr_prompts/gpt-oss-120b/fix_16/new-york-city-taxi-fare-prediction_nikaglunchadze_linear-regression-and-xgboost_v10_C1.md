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

3.12

# 3. Installed packages

geopandas==0.14.4
joblib==1.5.2
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
xgboost==2.0.3

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
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import gc

dtype_spec = {
    "key": "object",
    "fare_amount": "float32",
    "pickup_datetime": "object",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int8",
}

train_usecols = [
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]

ny_latitude_min, ny_latitude_max = 40.4772, 45.0153
ny_longitude_min, ny_longitude_max = -79.7624, -71.7517

ny_center = ("ny_center", (40.7128, -74.0060))
jfk_airport = ("jfk_airport", (40.6446, -73.7797))
lga_airport = ("lga_airport", (40.7733, -73.8718))
ewr_airport = ("ewr_airport", (40.6895, -74.1745))
locs = [ny_center, jfk_airport, lga_airport, ewr_airport]




## === cell 1
def add_datetime_features(df):
    """Extract year, month, day, weekday and hour from pickup_datetime."""
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
    df["year"] = df["pickup_datetime"].dt.year
    df["month"] = df["pickup_datetime"].dt.month
    df["day"] = df["pickup_datetime"].dt.day
    df["weekday"] = df["pickup_datetime"].dt.weekday
    df["hour"] = df["pickup_datetime"].dt.hour
    df.drop(columns=["pickup_datetime"], inplace=True)


def add_haversine_features(df):
    """Vectorised distance features without redundant radian conversions."""
    plat_rad = np.radians(df["pickup_latitude"].astype("float64"))
    plon_rad = np.radians(df["pickup_longitude"].astype("float64"))
    dlat_rad = np.radians(df["dropoff_latitude"].astype("float64"))
    dlon_rad = np.radians(df["dropoff_longitude"].astype("float64"))

    dlon_pair = dlon_rad - plon_rad
    dlat_pair = dlat_rad - plat_rad
    a = (
        np.sin(dlat_pair / 2.0) ** 2
        + np.cos(plat_rad) * np.cos(dlat_rad) * np.sin(dlon_pair / 2.0) ** 2
    )
    df["ride_distance"] = 6367 * 2 * np.arcsin(np.sqrt(a))

    for name, (lat_ref, lon_ref) in locs:
        lat_ref_rad = np.radians(lat_ref)
        lon_ref_rad = np.radians(lon_ref)

        dlon_ref = lon_ref_rad - plon_rad
        dlat_ref = lat_ref_rad - plat_rad
        a_ref = (
            np.sin(dlat_ref / 2.0) ** 2
            + np.cos(plat_rad) * np.cos(lat_ref_rad) * np.sin(dlon_ref / 2.0) ** 2
        )
        df[f"pickup_dist_to_{name}"] = 6367 * 2 * np.arcsin(np.sqrt(a_ref))

        dlon_ref = lon_ref_rad - dlon_rad
        dlat_ref = lat_ref_rad - dlat_rad
        a_ref = (
            np.sin(dlat_ref / 2.0) ** 2
            + np.cos(dlat_rad) * np.cos(lat_ref_rad) * np.sin(dlon_ref / 2.0) ** 2
        )
        df[f"dropoff_dist_to_{name}"] = 6367 * 2 * np.arcsin(np.sqrt(a_ref))

    df["lat_diff"] = df["dropoff_latitude"] - df["pickup_latitude"]
    df["lon_diff"] = df["dropoff_longitude"] - df["pickup_longitude"]
    df["abs_lat_diff"] = df["lat_diff"].abs()
    df["abs_lon_diff"] = df["lon_diff"].abs()

    float_cols = df.select_dtypes(include=["float64"]).columns
    df[float_cols] = df[float_cols].astype("float32")


def preprocess_chunk(chunk):
    """Filter rows and perform feature engineering on a dataframe chunk."""
    mask = (
        chunk["pickup_longitude"].between(ny_longitude_min, ny_longitude_max)
        & chunk["pickup_latitude"].between(ny_latitude_min, ny_latitude_max)
        & chunk["dropoff_longitude"].between(ny_longitude_min, ny_longitude_max)
        & chunk["dropoff_latitude"].between(ny_latitude_min, ny_latitude_max)
        & chunk["passenger_count"].between(1, 6)
    )
    chunk = chunk.loc[mask]
    chunk = chunk[(chunk["fare_amount"] >= 1) & (chunk["fare_amount"] <= 200)]
    chunk = chunk.dropna()
    add_datetime_features(chunk)
    add_haversine_features(chunk)
    if "ride_distance" in chunk.columns:
        chunk = chunk[chunk["ride_distance"] > 0]
    return chunk




## === cell 2
train_path = "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv"
chunks = []
chunksize = 10_000_000  # larger chunk reduces loop overhead
for chunk in pd.read_csv(
    train_path,
    dtype={k: v for k, v in dtype_spec.items() if k != "key"},
    usecols=train_usecols,
    chunksize=chunksize,
):
    processed = preprocess_chunk(chunk)
    chunks.append(processed)
    del chunk, processed
    gc.collect()
df = pd.concat(chunks, ignore_index=True)
del chunks
gc.collect()

test_path = "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv"
test_df = pd.read_csv(
    test_path,
    dtype={k: v for k, v in dtype_spec.items() if k != "fare_amount"},
)
test_keys = test_df["key"].copy()
add_datetime_features(test_df)
add_haversine_features(test_df)



## === cell 3
train_df, validation_df = train_test_split(df, test_size=0.2, random_state=42)



## === cell 4
features = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
    "year",
    "month",
    "day",
    "hour",
    "weekday",
    "ride_distance",
    "lat_diff",
    "lon_diff",
    "abs_lat_diff",
    "abs_lon_diff",
]
features += [f"pickup_dist_to_{x[0]}" for x in locs]
features += [f"dropoff_dist_to_{x[0]}" for x in locs]
fare_amount = "fare_amount"

train_features = train_df[features]
train_fare_amount = train_df[fare_amount]

validation_features = validation_df[features]
validation_fare_amount = validation_df[fare_amount]



## === cell 5
linear_model = LinearRegression()




## === cell 6
def estimate_model(model, df):
    X = df[features]
    y = df[fare_amount]
    print("Data shape for estimation:", X.shape)




## === cell 7
estimate_model(linear_model, train_df)



## === cell 8
linear_model.fit(train_features, train_fare_amount)



## === cell 9
linear_predictions = linear_model.predict(validation_features)
print(
    "Linear RMSE:",
    mean_squared_error(validation_fare_amount, linear_predictions, squared=False),
)



## === cell 10
log_train_target = np.log1p(train_fare_amount)
log_val_target = np.log1p(validation_fare_amount)

xgb_log_model = XGBRegressor(
    objective="reg:squarederror",
    learning_rate=0.03,
    n_estimators=800,
    max_depth=8,
    min_child_weight=1,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_lambda=1.0,
    n_jobs=-1,
    random_state=42,
    tree_method="hist",
)

xgb_log_model.fit(
    train_features,
    log_train_target,
    eval_set=[(validation_features, log_val_target)],
    early_stopping_rounds=200,
    verbose=False,
)

val_log_pred = xgb_log_model.predict(validation_features)
val_pred_raw = np.expm1(val_log_pred)
val_pred_clipped = np.clip(val_pred_raw, 1, 200)

validation_rmse_raw = mean_squared_error(
    validation_fare_amount, val_pred_raw, squared=False
)
validation_rmse_clip = mean_squared_error(
    validation_fare_amount, val_pred_clipped, squared=False
)
print("Validation RMSE (raw, no clipping):", validation_rmse_raw)
print("Validation RMSE (clipped to [1,200]):", validation_rmse_clip)



## === cell 11
train_log_pred = xgb_log_model.predict(train_features)
train_pred_raw = np.expm1(train_log_pred)
train_pred_clipped = np.clip(train_pred_raw, 1, 200)

train_rmse_raw = mean_squared_error(train_fare_amount, train_pred_raw, squared=False)
train_rmse_clip = mean_squared_error(
    train_fare_amount, train_pred_clipped, squared=False
)
print("Training RMSE (raw):", train_rmse_raw)
print("Training RMSE (clipped):", train_rmse_clip)



## === cell 12
xgb_val_pred = val_pred_clipped
xgb_rmse = mean_squared_error(validation_fare_amount, xgb_val_pred, squared=False)
print("XGBoost Validation RMSE (final clipped predictions):", xgb_rmse)



## === cell 13
test_pred_raw = np.expm1(xgb_log_model.predict(test_df[features]))
test_pred = np.clip(test_pred_raw, 1, 200)

holdout = pd.DataFrame({"key": test_keys, "fare_amount": test_pred})
holdout.to_csv("submission.csv", index=False)
print("Submission file written to submission.csv")
