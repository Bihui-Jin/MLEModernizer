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
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
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

# 5. Target score

4.00706

# 6. Current score

5.12465

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 4.77089) has done: 'I fix the target‑variable handling (use the fare_amount column as Y_train instead of an empty DataFrame), correct the XGBoost predict call, and build the submission file with the required key and fare_amount columns. These changes resolve the runtime errors and ensure a valid CSV is produced while keeping the original model logic unchanged.'
- What this solution (achieved 4.77589) has done: 'The changes add a more informative distance feature (haversine distance) and tune the XGBoost hyper‑parameters, which are small adjustments that keep the original model structure while improving predictive performance and moving the RMSE closer to the target.'
- What this solution (achieved 4.69511) has done: 'I add a few lightweight features (month and weekend flag) and switch to a train‑validation split with early‑stopping so the XGBoost model can stop at the optimal number of trees. This keeps the original model architecture while giving a modest boost in predictive power, which should bring the RMSE closer to the target 4.00706 without over‑hauling the pipeline. The script now writes a proper sample_submission.csv as before.'
- What this solution (achieved 5.55291) has done: 'The updates keep the same feature engineering and XGBoost model, but they dramatically cut the amount of data that needs to be processed. After loading the full CSV we immediately down‑sample the training rows (keeping a representative random subset) and use a smaller validation split, which reduces both memory use and the number of boosting iterations over huge arrays. All other steps—including dtype handling, feature creation, and the exact XGBoost hyper‑parameters—remain unchanged, so the prediction logic and accuracy are preserved while the runtime falls well below the 600‑second limit.'
- What this solution (achieved 4.80725) has done: 'Add a modest increase in training data size (sample 20 % instead of 10 %) to give the model more information, and allow a slightly longer early‑stopping window (150 rounds) so the booster can converge a bit further. These tiny adjustments keep the original pipeline intact while likely lowering the RMSE toward the target value.'
- What this solution (achieved 5.03799) has done: 'I slightly increase the sampled training size from 20 % to 30 % to give the model more data, add cyclical hour/month/day‑of‑week features (sin/cos) to capture periodic patterns, and give the XGBoost model a bit more capacity (max_depth 9, n_estimators 2000) with a longer early‑stopping patience. These small, targeted changes keep the original pipeline intact while encouraging a lower RMSE, moving the score toward the target 4.00706.'
- What this solution (achieved 5.08371) has done: 'I fixed the undefined variables, added a reusable feature‑engineering function, correctly split the sampled training data into features X and target y, train an XGBoost regressor on the engineered features, and finally generate a submission file containing the required **key** and **fare_amount** columns. All steps are kept consistent with the original pipeline while ensuring the script runs end‑to‑end and writes a valid CSV.'
- What this solution (achieved 4.9763) has done: 'I increase the training sample size from 40 % to 50 % of each chunk and add two inexpensive distance‑based features (Manhattan distance and a log‑scaled haversine) inside the feature‑engineering function. These changes keep the original model architecture and training loop intact while giving the model more data and richer inputs, which should lower the RMSE toward the target score.'
- What this solution (achieved 5.12465) has done: 'I increase the training sample fraction to use more data (60 % instead of 50 %) and set a fixed random seed for reproducibility. I also slightly adjust the XGBoost hyper‑parameters: more trees, a lower learning rate, and a tighter early‑stopping patience, which should modestly improve validation performance without changing the overall model structure. These minimal tweaks aim to reduce the RMSE toward the target while keeping the core pipeline unchanged.'

# 9. Code solution

## === cell 0
import os, random, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb

np.random.seed(42)

dtype_map = {
    "fare_amount": np.float32,
    "pickup_longitude": np.float32,
    "pickup_latitude": np.float32,
    "dropoff_longitude": np.float32,
    "dropoff_latitude": np.float32,
    "passenger_count": np.int8,
}
usecols_train = [
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]

train_chunks = pd.read_csv(
    "../input/train.csv",
    dtype=dtype_map,
    usecols=usecols_train,
    low_memory=False,
    chunksize=500_000,
)

samples = []
for chunk in train_chunks:
    chunk = chunk.dropna()
    chunk = chunk[chunk["fare_amount"] > 0]
    mask = np.random.rand(len(chunk)) < 0.60
    samples.append(chunk[mask])

training_data = pd.concat(samples, ignore_index=True)

test_data = pd.read_csv(
    "../input/test.csv",
    dtype=dtype_map,
    usecols=[
        "key",
        "pickup_datetime",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
    ],
    low_memory=False,
)




## === cell 1
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create engineered features for both train and test."""
    df = df.copy()

    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
    df["hour"] = df["pickup_datetime"].dt.hour.astype(np.int8)
    df["dayofweek"] = df["pickup_datetime"].dt.dayofweek.astype(np.int8)
    df["month"] = df["pickup_datetime"].dt.month.astype(np.int8)
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(np.int8)
    df["is_night"] = df["hour"].isin([0, 1, 2, 3, 4, 5, 20, 21, 22, 23]).astype(np.int8)

    df["latitude_distance"] = np.abs(
        df["dropoff_latitude"] - df["pickup_latitude"]
    ).astype(np.float32)
    df["longitude_distance"] = np.abs(
        df["dropoff_longitude"] - df["pickup_longitude"]
    ).astype(np.float32)

    df["manhattan"] = (df["latitude_distance"] + df["longitude_distance"]).astype(
        np.float32
    )

    df["pickup_lat_lon"] = (df["pickup_latitude"] * df["pickup_longitude"]).astype(
        np.float32
    )
    df["dropoff_lat_lon"] = (df["dropoff_latitude"] * df["dropoff_longitude"]).astype(
        np.float32
    )

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24).astype(np.float32)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24).astype(np.float32)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12).astype(np.float32)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12).astype(np.float32)
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7).astype(np.float32)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7).astype(np.float32)

    R = 6371.0
    lat1 = np.radians(df["pickup_latitude"])
    lat2 = np.radians(df["dropoff_latitude"])
    dlat = lat2 - lat1
    dlon = np.radians(df["dropoff_longitude"] - df["pickup_longitude"])
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    df["haversine"] = (R * c).astype(np.float32)

    df["log_haversine"] = np.log1p(df["haversine"]).astype(np.float32)

    df = df.drop(columns=["dropoff_longitude", "dropoff_latitude", "pickup_datetime"])
    return df


X_raw = training_data.drop(columns=["fare_amount"])
y = training_data["fare_amount"].astype(np.float32)

X = engineer_features(X_raw)

numeric_cols = X.select_dtypes(include=["float64", "int64", "int8"]).columns
X[numeric_cols] = X[numeric_cols].astype(np.float32)

X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.05, random_state=42)

model = xgb.XGBRegressor(
    objective="reg:squarederror",
    eval_metric="rmse",
    n_estimators=1200,  # allow more trees; early stopping will limit actual count
    max_depth=9,
    learning_rate=0.04,  # slightly slower learning for better generalisation
    subsample=0.9,
    colsample_bytree=0.9,
    n_jobs=4,
    random_state=42,
    tree_method="hist",
    verbosity=0,
)

model.fit(
    X_tr,
    y_tr,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=30,  # tighter patience to avoid over‑fitting
    verbose=False,
)



## === cell 2
test_features_raw = test_data.drop(columns=["key"])
X_test = engineer_features(test_features_raw)

numeric_cols_test = X_test.select_dtypes(include=["float64", "int64", "int8"]).columns
X_test[numeric_cols_test] = X_test[numeric_cols_test].astype(np.float32)

y_pred = model.predict(X_test)
y_pred = np.maximum(y_pred, 0.0)

submission = pd.DataFrame({"key": test_data["key"], "fare_amount": y_pred})
submission.to_csv("sample_submission.csv", index=False)
print("Submission saved to sample_submission.csv")
