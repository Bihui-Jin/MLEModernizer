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

No external packages required in the script and installed.

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

3.830643716740832

# 6. Current score

1066.87503

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 5.21833) has done: 'I fix the environment-breaking TensorFlow/Keras imports by removing the incompatible scikit-learn/Keras wrapper grid-search section and switching to `tf.keras` only, which is enough to train and predict with the same neural-network core logic. I also fix the file-path issues by reading the provided `../input/new-york-city-taxi-fare-prediction/train.csv` and `test.csv` instead of non-existent `my-taxi-fare-data/*` files. Because your code expects a `distance` feature that is not in the raw CSVs, I minimally add it (haversine distance) during preprocessing for both train and test. Finally, I ensure normalization uses the training-set mean/std (not test-set stats) and always writes a valid `submission_file.csv` with columns `key,fare_amount`.'
- What this solution (achieved 1066.87501) has done: 'I fix the TensorFlow import crash (`MessageFactory` / protobuf mismatch) by avoiding TensorFlow entirely and replacing it with a small NumPy-based linear regression that preserves your existing feature engineering (including the haversine `distance`) and normalization pipeline. This change removes the environment-breaking dependency while keeping the same supervised regression semantics and produces a valid `submission_file.csv` with `key,fare_amount`. I also add a minimal, standard NYC Taxi cleanup filter on the training shards (valid coordinate bounds and fare/passenger sanity) to improve RMSE toward the target without changing the feature set or introducing extra data. Finally, I keep the same file paths and ensure test keys stay aligned with predictions.'
- What this solution (achieved 1066.87503) has done: 'Your huge RMSE is most likely coming from a feature/normalization mismatch bug: you compute normalized features for training, but you accidentally fit the ridge regression on the *unnormalized* features. I make the minimal fix by training on `mynp_train_norm_*` (the already-normalized matrices) while keeping the exact same ridge model, same features, and same prediction pipeline. I also add a tiny numerical-stability tweak to the normalizer (cast mean/std to float64 during normalization) to avoid dtype drift, without changing semantics. This should move RMSE dramatically down toward the target by making train/test feature scales consistent.'

# 9. Code solution

## === cell 0
print("hello moto")



## === cell 1
import os
import math
import numpy as np
import pandas as pd

np.random.seed(42)

TRAIN_PATH = "../input/new-york-city-taxi-fare-prediction/train.csv"
TEST_PATH = "../input/new-york-city-taxi-fare-prediction/test.csv"
SAMPLE_SUB_PATH = "../input/new-york-city-taxi-fare-prediction/sample_submission.csv"

for p in [TRAIN_PATH, TEST_PATH, SAMPLE_SUB_PATH]:
    if not os.path.exists(p):
        raise FileNotFoundError(f"Expected file not found: {p}")

print("Using NumPy-only training to avoid TF/protobuf runtime crash.")
print(
    "train exists:",
    os.path.exists(TRAIN_PATH),
    "test exists:",
    os.path.exists(TEST_PATH),
)




## === cell 2
def _haversine_np(lon1, lat1, lon2, lat2):
    """
    Vectorized haversine distance in kilometers.
    """
    lon1 = np.asarray(lon1, dtype=np.float64)
    lat1 = np.asarray(lat1, dtype=np.float64)
    lon2 = np.asarray(lon2, dtype=np.float64)
    lat2 = np.asarray(lat2, dtype=np.float64)

    rlon1 = np.radians(lon1)
    rlat1 = np.radians(lat1)
    rlon2 = np.radians(lon2)
    rlat2 = np.radians(lat2)

    dlon = rlon2 - rlon1
    dlat = rlat2 - rlat1

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(rlat1) * np.cos(rlat2) * np.sin(dlon / 2.0) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(a))
    R = 6371.0  # km
    return R * c


def _add_distance_feature(df):
    """
    Adds the 'distance' column expected by the original code.
    """
    df = df.copy()
    df["distance"] = _haversine_np(
        df["pickup_longitude"].values,
        df["pickup_latitude"].values,
        df["dropoff_longitude"].values,
        df["dropoff_latitude"].values,
    )
    return df


def data_to_np(input_file, nrows=900000):
    """
    Reads a slice of the training file, builds the same feature set the original code expects,
    and returns (X, y).
    """
    df = pd.read_csv(input_file, sep=",", nrows=nrows)

    df = df.dropna(
        subset=[
            "fare_amount",
            "pickup_longitude",
            "pickup_latitude",
            "dropoff_longitude",
            "dropoff_latitude",
            "passenger_count",
        ]
    )

    df = _add_distance_feature(df)

    header_names = [
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
        "distance",
    ]

    df_train = df[header_names].astype(np.float32)
    np_df_train = df_train.values

    df_label = df["fare_amount"].astype(np.float32)
    np_df_label = df_label.values

    return np_df_train, np_df_label




## === cell 3
def global_mean_per_column(mynp_train_list):
    sum_mean = 0
    for con in range(len(mynp_train_list)):
        sum_mean = sum_mean + np.mean(mynp_train_list[con], axis=0)
    mean = sum_mean / len(mynp_train_list)
    return mean




## === cell 4
def global_std_per_column(mynp_train_list, global_mean):
    sum_mean_x2 = 0
    for con in range(len(mynp_train_list)):
        sum_mean_x2 += np.mean((mynp_train_list[con] - global_mean) ** 2, axis=0)
    std = np.sqrt(sum_mean_x2 / len(mynp_train_list))
    std = np.where(std == 0, 1.0, std)
    return std




## === cell 5
def norm_mynp_train(mynp_train, mean, std):
    mean64 = np.asarray(mean, dtype=np.float64)
    std64 = np.asarray(std, dtype=np.float64)
    X64 = np.asarray(mynp_train, dtype=np.float64)
    mynp_train_norm = (X64 - mean64) / std64
    return mynp_train_norm.astype(np.float32)




## === cell 6
def _basic_nyc_sanity_filter(df):
    """
    Bugfix/score-improvement: remove clearly invalid rows that hurt RMSE.
    This keeps the same features/target, only cleans corrupted/outlier training points.
    """
    cond = (
        df["pickup_longitude"].between(-75, -72)
        & df["dropoff_longitude"].between(-75, -72)
        & df["pickup_latitude"].between(40, 42)
        & df["dropoff_latitude"].between(40, 42)
        & df["passenger_count"].between(1, 6)
        & df["fare_amount"].between(2.5, 250.0)
    )
    return df.loc[cond].copy()


def load_train_shards(train_path, shard_rows=300000, n_shards=3):
    mynp_trains, mynp_labels = [], []
    for i in range(n_shards):
        skip = 1 + i * shard_rows  # +1 to skip header row when skiprows is used
        df = pd.read_csv(train_path, nrows=shard_rows, skiprows=range(1, skip))
        df = df.dropna(
            subset=[
                "fare_amount",
                "pickup_longitude",
                "pickup_latitude",
                "dropoff_longitude",
                "dropoff_latitude",
                "passenger_count",
            ]
        )

        df = _basic_nyc_sanity_filter(df)

        df = _add_distance_feature(df)

        header_names = [
            "pickup_longitude",
            "pickup_latitude",
            "dropoff_longitude",
            "dropoff_latitude",
            "passenger_count",
            "distance",
        ]
        X = df[header_names].astype(np.float32).values
        y = df["fare_amount"].astype(np.float32).values

        mynp_trains.append(X)
        mynp_labels.append(y)
    return mynp_trains, mynp_labels




## === cell 7
mynp_train_list, mynp_label_list = load_train_shards(
    TRAIN_PATH, shard_rows=300000, n_shards=3
)

mynp_train_0, mynp_train_1, mynp_train_2 = mynp_train_list
mynp_label_0, mynp_label_1, mynp_label_2 = mynp_label_list

print([m.shape for m in mynp_train_list], [l.shape for l in mynp_label_list])




## === cell 8
def _shuffle_pair(X, y):
    order = np.argsort(np.random.random(y.shape))
    return X[order], y[order]


mynp_train_0, mynp_label_0 = _shuffle_pair(mynp_train_0, mynp_label_0)
mynp_train_1, mynp_label_1 = _shuffle_pair(mynp_train_1, mynp_label_1)
mynp_train_2, mynp_label_2 = _shuffle_pair(mynp_train_2, mynp_label_2)



## === cell 9
mynp_train_list = [mynp_train_0, mynp_train_1, mynp_train_2]
mynp_label_list = [mynp_label_0, mynp_label_1, mynp_label_2]



## === cell 10
global_mean = global_mean_per_column(mynp_train_list)
global_std = global_std_per_column(mynp_train_list, global_mean)
print("global_mean:", global_mean)
print("global_std :", global_std)



## === cell 11
mynp_train_norm_0 = norm_mynp_train(mynp_train_list[0], global_mean, global_std)
mynp_train_norm_1 = norm_mynp_train(mynp_train_list[1], global_mean, global_std)
mynp_train_norm_2 = norm_mynp_train(mynp_train_list[2], global_mean, global_std)

mynp_train_concat = np.concatenate(
    (mynp_train_norm_0, mynp_train_norm_1, mynp_train_norm_2), axis=0
)
mynp_label_concat = np.concatenate((mynp_label_0, mynp_label_1, mynp_label_2), axis=0)

print(mynp_train_concat.shape, mynp_label_concat.shape)




## === cell 12
def fit_ridge_regression_closed_form(X, y, l2=1.0):
    """
    Fit ridge regression with intercept using a closed-form solve:
    w = (X^T X + l2*I)^(-1) X^T y, with intercept unregularized.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1, 1)

    Xb = np.concatenate([np.ones((X.shape[0], 1), dtype=np.float64), X], axis=1)

    XtX = Xb.T @ Xb
    reg = np.eye(XtX.shape[0], dtype=np.float64) * float(l2)
    reg[0, 0] = 0.0  # don't regularize intercept

    Xty = Xb.T @ y
    w = np.linalg.solve(XtX + reg, Xty)  # (d+1, 1)
    return w.reshape(-1)


def predict_ridge_regression(X, w):
    X = np.asarray(X, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    Xb = np.concatenate([np.ones((X.shape[0], 1), dtype=np.float64), X], axis=1)
    return (Xb @ w).astype(np.float64)




## === cell 13
w = fit_ridge_regression_closed_form(mynp_train_concat, mynp_label_concat, l2=1.0)
print("Trained ridge weights shape:", w.shape)



## === cell 14
df_test = pd.read_csv(TEST_PATH, sep=",")
df_test = df_test.dropna(
    subset=[
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
    ]
)

df_test = _add_distance_feature(df_test)

df_test = df_test[
    [
        "key",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
        "distance",
    ]
]

df_test_fn = df_test[
    [
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
        "distance",
    ]
].astype(np.float32)

mynp_test = df_test_fn.values
mynp_test = (
    mynp_test.astype(np.float64) - np.asarray(global_mean, dtype=np.float64)
) / np.asarray(global_std, dtype=np.float64)
mynp_test = mynp_test.astype(np.float32)

print("Test matrix shape:", mynp_test.shape)



## === cell 15
test_predictions = predict_ridge_regression(mynp_test, w).reshape(-1)

test_predictions = np.clip(test_predictions, 0.0, None)

print(test_predictions[:5], test_predictions.shape)



## === cell 16
test_key_array = df_test["key"].values
df_output = pd.DataFrame({"key": test_key_array, "fare_amount": test_predictions})
df_output = df_output[["key", "fare_amount"]]
print(df_output.head())

out_path = "submission_file.csv"
df_output.to_csv(out_path, index=False)

print("Wrote:", out_path, "rows:", len(df_output))
print(df_output.dtypes)
