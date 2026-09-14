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
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1
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

4.19822

# 6. Current score

6.68772

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 5.83099) has done: 'Your code already trains and predicts, but it can fail Kaggle scoring because (1) you’re filtering out test rows where `dropoff_longitude == 0`, which makes the submission row count not match the sample/test set, and (2) you optimize hyperparameters using MAE even though the competition metric is RMSE. I keep the same overall approach (XGBoost on the same engineered features) and make minimal changes: don’t drop any test rows, switch the validation metric to RMSE to better match the leaderboard, and add a deterministic split/seed so results are stable. The submission be created with exactly the same `key` order/row count as `test.csv`.'
- What this solution (achieved 5.98272) has done: 'I keep your XGBoost + the same engineered features, but reduce RMSE by fixing the main data-quality issue: the model is trained on raw fares/coords with outliers that dominate RMSE. I add minimal, standard NYC Taxi Fare cleaning on the *training* subset only (valid fare range, lat/lon bounding box, passenger_count range, and nonzero distance), without dropping any test rows so the submission length stays correct. To move toward the target score, I also use XGBoost’s squared-error objective and a slightly more regularized/default-stable tree setup (max_depth/min_child_weight/subsample/colsample), keeping your same hyperparameter search over `n_estimators` and `learning_rate` and still selecting by RMSE. Finally, I clip negative predictions to 0 to avoid RMSE inflation from impossible fares.'
- What this solution (achieved 6.00572) has done: 'Your current gap is 5.98272 − 4.19822 = +1.7845 RMSE, so we need a genuine improvement (lower RMSE) with minimal, low-risk changes. The biggest limiter is that you only train on 10,000 rows from a 55M-row dataset; increasing the training sample (while keeping the same model/features/training loop) is the most direct way to move toward the target. To avoid slowing down too much, I keep your exact hyperparameter search structure but reduce the grid size slightly and use XGBoost’s built-in early stopping **only during the hyperparameter selection stage** is not allowed per your constraints, so I do not add it; instead I keep the loop but make it smaller to offset the larger training sample. I also ensure the test feature frame is numeric/finite (no NaNs/Infs) so predictions can’t silently degrade due to datetime parsing issues.'
- What this solution (achieved 5.7898) has done: 'You’re currently worse than the target (6.00572 vs 4.19822 RMSE), so we need a real RMSE improvement with minimal, low-risk changes. The biggest gain without changing your model/feature logic is to (1) train the final model on **all cleaned training rows** (train+valid combined) after selecting hyperparameters, and (2) add a single strong, standard feature for this competition: **Haversine distance** (keeping your existing distance features intact). I also keep the hyperparameter search structure but modestly expand it toward commonly better settings (still small grid), and I enforce identical train/test columns ordering to avoid any silent mismatch. These changes preserve the same core approach (XGBoost regressor on engineered time/distance features with an explicit train/valid split and RMSE selection) while typically moving RMSE materially downward toward your target.'
- What this solution (achieved 6.68772) has done: 'You’re currently worse than the target (RMSE 5.7898 vs 4.19822), so we should make a small, safe improvement without changing the model family or training loop. The biggest low-risk gain for this competition is to add a few standard, cheap features (absolute deltas and Manhattan distance) while keeping your existing time/distance features and XGBoost setup intact. I also add a minimal training-only filter for unrealistically long trips (using your already-computed haversine distance) to reduce RMSE sensitivity to outliers without touching the test set. Finally, I ensure numeric coercion + consistent NaN handling for both train/test so the model sees identical feature semantics at inference.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"

print(
    "Skipping TensorFlow import due to TensorFlow/protobuf incompatibility in this environment."
)



## === cell 1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

import warnings

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

train_iop_path = "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv"
test_iop_path = "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv"

TRAIN_NROWS = 1_000_000

dataset_train = pd.read_csv(train_iop_path, nrows=TRAIN_NROWS, index_col="key")
dataset_test = pd.read_csv(test_iop_path, index_col="key")

print("train rows:", len(dataset_train), "test rows:", len(dataset_test))




## === cell 2
def clean_train_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    num_cols = [
        "fare_amount",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
    ]
    for c in num_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out = out.dropna(subset=num_cols + ["pickup_datetime"])

    out = out[(out["fare_amount"] > 0) & (out["fare_amount"] <= 250)]
    out = out[(out["passenger_count"] >= 1) & (out["passenger_count"] <= 6)]

    out = out[
        (out["pickup_longitude"].between(-74.3, -73.7))
        & (out["dropoff_longitude"].between(-74.3, -73.7))
        & (out["pickup_latitude"].between(40.5, 41.0))
        & (out["dropoff_latitude"].between(40.5, 41.0))
    ]

    out = out[
        ~(
            (out["pickup_longitude"] == out["dropoff_longitude"])
            & (out["pickup_latitude"] == out["dropoff_latitude"])
        )
    ]

    return out


print("dataset_train old size", len(dataset_train))
dataset_train = clean_train_df(dataset_train)
print("dataset_train new size (cleaned)", len(dataset_train))

print("dataset_test size (kept intact)", len(dataset_test))




## === cell 3
def get_year(pickup_date):
    return pickup_date.year


def get_month(pickup_date):
    return pickup_date.month


def get_day(pickup_date):
    return pickup_date.day


def get_hour(pickup_date):
    return pickup_date.hour




## === cell 4
def _haversine_km(lon1, lat1, lon2, lat2):
    lon1 = np.radians(lon1.astype(float))
    lat1 = np.radians(lat1.astype(float))
    lon2 = np.radians(lon2.astype(float))
    lat2 = np.radians(lat2.astype(float))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371.0 * c


def preparedataset2(datasetname: pd.DataFrame) -> pd.DataFrame:
    df = datasetname.copy()

    for c in [
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], errors="coerce")

    df["pickup_year"] = df["pickup_datetime"].dt.year
    df["pickup_month"] = df["pickup_datetime"].dt.month
    df["pickup_day"] = df["pickup_datetime"].dt.day
    df["pickup_hour"] = df["pickup_datetime"].dt.hour

    df["x_dis"] = df["dropoff_longitude"] - df["pickup_longitude"]
    df["y_dis"] = df["dropoff_latitude"] - df["pickup_latitude"]
    df["dis"] = np.sqrt(
        (df["dropoff_longitude"] - df["pickup_longitude"]) ** 2
        + (df["dropoff_latitude"] - df["pickup_latitude"]) ** 2
    )

    df["haversine_km"] = _haversine_km(
        df["pickup_longitude"].values,
        df["pickup_latitude"].values,
        df["dropoff_longitude"].values,
        df["dropoff_latitude"].values,
    )

    df["abs_x_dis"] = np.abs(df["x_dis"])
    df["abs_y_dis"] = np.abs(df["y_dis"])
    df["manhattan_dis"] = df["abs_x_dis"] + df["abs_y_dis"]

    df = df.drop(["pickup_datetime"], axis=1)
    df = df.drop(["pickup_longitude"], axis=1)
    df = df.drop(["dropoff_latitude"], axis=1)
    df = df.drop(["dropoff_longitude"], axis=1)
    df = df.drop(["pickup_latitude"], axis=1)

    return df




## === cell 5
df = preparedataset2(dataset_train)
test_df = preparedataset2(dataset_test)

df = df.replace([np.inf, -np.inf], np.nan).dropna()
test_df = test_df.replace([np.inf, -np.inf], np.nan)

if "haversine_km" in df.columns:
    df = df[(df["haversine_km"] > 0) & (df["haversine_km"] <= 60)].copy()

train_feature_medians = df.drop(columns=["fare_amount"], errors="ignore").median(
    numeric_only=True
)
test_df = test_df.fillna(train_feature_medians)

print("Prepared train shape:", df.shape)
print("Prepared test shape:", test_df.shape)
df.head(3)



## === cell 6
y = df["fare_amount"]
X = df.drop("fare_amount", axis=1)

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

print("X_train:", X_train.shape, "X_valid:", X_valid.shape)



## === cell 7
best_n_estimators = None
best_learning_rate = None
best_rmse = float("inf")

common_params = dict(
    objective="reg:squarederror",
    max_depth=8,
    min_child_weight=1.0,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.0,
    reg_lambda=1.0,
    n_jobs=4,
    random_state=RANDOM_STATE,
)

for lr in [0.03, 0.05, 0.07, 0.10]:
    for ns in [600, 900, 1200]:
        my_model = XGBRegressor(
            n_estimators=ns,
            learning_rate=lr,
            **common_params,
        )
        my_model.fit(X_train, y_train)

        preds = my_model.predict(X_valid)
        rmse = mean_squared_error(y_valid, preds, squared=False)

        if rmse < best_rmse:
            best_rmse = rmse
            best_n_estimators = ns
            best_learning_rate = lr
            print("better found ->", "n_estimators:", ns, "lr:", lr, "RMSE:", rmse)

print("best_n_estimators:", best_n_estimators)
print("best_learning_rate:", best_learning_rate)
print("Best validation RMSE:", best_rmse)



## === cell 8
final_model = XGBRegressor(
    n_estimators=best_n_estimators,
    learning_rate=best_learning_rate,
    **common_params,
)

final_model.fit(X, y)

valid_preds = final_model.predict(X_valid)
valid_rmse = mean_squared_error(y_valid, valid_preds, squared=False)
print("Validation RMSE (refit-on-all model, evaluated on X_valid):", valid_rmse)



## === cell 9
test_df_aligned = test_df.reindex(columns=X.columns)

test_preds = final_model.predict(test_df_aligned)

test_preds = np.clip(test_preds, 0.0, None)

output = pd.DataFrame(
    {
        "key": test_df_aligned.index,
        "fare_amount": test_preds,
    }
)

output = output[["key", "fare_amount"]]
print("Submission rows:", len(output), "Expected:", len(dataset_test))

output.to_csv("submission.csv", index=False)
print("Wrote submission.csv")
