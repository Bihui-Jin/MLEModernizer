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

3.8575708313744257

# 6. Current score

36.48669

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 5.20914) has done: 'I fix the environment-breaking TensorFlow/Keras import usage (your current TF/Keras stack triggers a protobuf `MessageFactory.GetPrototype` crash) by switching to scikit-learn’s `MLPRegressor`, keeping the same core idea: a multilayer dense neural network trained on normalized numeric features with MSE loss. I also fix the missing/incorrect input file paths by reading the provided Kaggle `train.csv`/`test.csv` files and computing the missing `distance` feature from coordinates so the rest of your pipeline remains consistent. To ensure the notebook finishes under the time limit, I keep the training subset size similar to your original intent (you were reading 3×900k rows from custom shards, which aren’t available here). Finally, I generate a valid `submission_file.csv` with exactly the required `key,fare_amount` columns aligned to the test rows.'
- What this solution (achieved 1187.25401) has done: 'Your score (RMSE 5.209) is worse than the target (3.858), so we should cautiously improve generalization without changing the model family or overall pipeline. The biggest low-risk gain here is data quality: NYC taxi fare is extremely sensitive to outliers (bad coordinates, impossible passenger counts, extreme distances/fare), and your current script trains on them, which typically hurts RMSE a lot. I add minimal, standard train-time filtering for obviously invalid rows (coords bounds, passenger_count, positive and capped fare, reasonable distance) while keeping the same features, normalization approach, and `MLPRegressor` architecture/training. I also make the chunk reads consistent by applying the same `dropna(subset=cols+['fare_amount'])` logic to all chunks to avoid training on rows with missing critical fields.'
- What this solution (achieved 36.53649) has done: 'Your current RMSE (1187) is far worse than the target (3.86), which strongly suggests a submission alignment/format issue rather than pure model quality. The minimal fix is to ensure the prediction vector aligns 1:1 with the original test rows: don’t drop test rows with NaNs (keep all keys), compute distance with NaN-safe handling, and fill missing feature values using train means before normalization so every test key gets a real prediction. I also clamp extreme predictions to a reasonable upper bound (consistent with your train filtering cap) to reduce the impact of any remaining bad rows, without changing the model family, features, or training approach. These changes keep your core logic intact (same MLPRegressor, same features, same normalization method) while directly addressing the likely source of the catastrophic score.'
- What this solution (achieved 36.48669) has done: 'Your current RMSE (36.54) is far worse than the target (3.86), so we should improve generalization without changing the model family or training loop. The biggest low-risk issue in this exact script is that you’re computing `distance` in kilometers but using filtering and model expectations that are typically tuned for miles-like scales in many NYC Taxi kernels; this mismatch can severely distort learned relationships and clipping, inflating error. I keep the same features and MLPRegressor setup, but switch the distance feature to miles and adjust only the distance filtering bounds accordingly (same idea: filter obviously bad trips), which should move RMSE substantially toward the target. I also remove the unused/incorrect `train_feature_means` variable to avoid confusion, but leave prediction alignment and submission generation intact.'

# 9. Code solution

## === cell 0
print("hello moto")



## === cell 1
import os
import pandas as pd
import numpy as np
import math

from sklearn.neural_network import MLPRegressor

np.random.seed(42)

DATA_DIR = "/kaggle/input"
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")

print("Train exists:", os.path.exists(TRAIN_PATH), TRAIN_PATH)
print("Test exists:", os.path.exists(TEST_PATH), TEST_PATH)
print("Sample exists:", os.path.exists(SAMPLE_SUB_PATH), SAMPLE_SUB_PATH)




## === cell 2
def haversine_distance_miles(
    pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude
):
    """
    Vectorized haversine distance in miles.
    NaN-safe: if any coord is NaN, output will be NaN for that row (handled later).
    """
    lon1 = np.radians(pickup_longitude.astype(float))
    lat1 = np.radians(pickup_latitude.astype(float))
    lon2 = np.radians(dropoff_longitude.astype(float))
    lat2 = np.radians(dropoff_latitude.astype(float))

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * (
        np.sin(dlon / 2.0) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(a))
    r_miles = 3958.756  # Earth radius in miles
    return r_miles * c


def add_distance_feature(df):
    df = df.copy()
    df["distance"] = haversine_distance_miles(
        df["pickup_longitude"].values,
        df["pickup_latitude"].values,
        df["dropoff_longitude"].values,
        df["dropoff_latitude"].values,
    )
    return df


def filter_train_rows(df):
    df = df.copy()

    req = [
        "fare_amount",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
        "distance",
    ]
    df = df.dropna(subset=req)

    df = df[
        (df["pickup_longitude"].between(-74.3, -73.6))
        & (df["dropoff_longitude"].between(-74.3, -73.6))
        & (df["pickup_latitude"].between(40.5, 41.0))
        & (df["dropoff_latitude"].between(40.5, 41.0))
    ]

    df = df[df["passenger_count"].between(1, 6)]

    df = df[df["fare_amount"].between(2.5, 250.0)]

    df = df[df["distance"].between(0.03, 60.0)]

    return df




## === cell 3
def data_to_np(input_file, nrows=900000, skiprows=None):
    """
    Read Kaggle-provided CSV and compute 'distance' on the fly.
    """
    df = pd.read_csv(input_file, sep=",", nrows=nrows, skiprows=skiprows)

    df = add_distance_feature(df)

    header_names = [
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
        "distance",
    ]

    if "fare_amount" in df.columns:
        df = filter_train_rows(df)
        X = df[header_names].values.astype(np.float32)
        y = df["fare_amount"].values.astype(np.float32)
        return X, y
    else:
        X = df[header_names].values.astype(np.float32)
        return X, None




## === cell 4
def global_mean_per_column(mynp_train_list):
    sum_mean = 0.0
    for con in range(len(mynp_train_list)):
        sum_mean = sum_mean + np.mean(mynp_train_list[con], axis=0)
    mean = sum_mean / len(mynp_train_list)
    return mean


def global_std_per_column(mynp_train_list, global_mean):
    sum_mean_x2 = 0.0
    for con in range(len(mynp_train_list)):
        sum_mean_x2 += np.mean((mynp_train_list[con] - global_mean) ** 2, axis=0)
    std = np.sqrt(sum_mean_x2 / len(mynp_train_list))
    return std


def norm_mynp_train(mynp_train, mean, std):
    std_safe = np.where(std == 0, 1.0, std)
    mynp_train_norm = (mynp_train - mean) / std_safe
    return mynp_train_norm




## === cell 5
NROWS_PER_CHUNK = 250000  # 3 chunks -> up to 750k raw rows (fewer after filtering)

mynp_train_0, mynp_label_0 = data_to_np(
    TRAIN_PATH, nrows=NROWS_PER_CHUNK, skiprows=None
)

mynp_train_1, mynp_label_1 = data_to_np(
    TRAIN_PATH,
    nrows=NROWS_PER_CHUNK,
    skiprows=range(1, 1 + NROWS_PER_CHUNK),
)

mynp_train_2, mynp_label_2 = data_to_np(
    TRAIN_PATH,
    nrows=NROWS_PER_CHUNK,
    skiprows=range(1, 1 + 2 * NROWS_PER_CHUNK),
)

print(mynp_train_0.shape, mynp_train_1.shape, mynp_train_2.shape)
print("Labels:", mynp_label_0.shape, mynp_label_1.shape, mynp_label_2.shape)




## === cell 6
def shuffle_in_unison(X, y, seed=42):
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(y))
    return X[order], y[order]


mynp_train_0, mynp_label_0 = shuffle_in_unison(mynp_train_0, mynp_label_0, seed=1)
mynp_train_1, mynp_label_1 = shuffle_in_unison(mynp_train_1, mynp_label_1, seed=2)
mynp_train_2, mynp_label_2 = shuffle_in_unison(mynp_train_2, mynp_label_2, seed=3)



## === cell 7
mynp_train_list = [mynp_train_0, mynp_train_1, mynp_train_2]
mynp_label_list = [mynp_label_0, mynp_label_1, mynp_label_2]

global_mean = global_mean_per_column(mynp_train_list)
global_std = global_std_per_column(mynp_train_list, global_mean)

print("global_mean:", global_mean)
print("global_std :", global_std)



## === cell 8
mynp_train_norm_0 = norm_mynp_train(mynp_train_list[0], global_mean, global_std)
mynp_train_norm_1 = norm_mynp_train(mynp_train_list[1], global_mean, global_std)
mynp_train_norm_2 = norm_mynp_train(mynp_train_list[2], global_mean, global_std)

mynp_train_concat = np.concatenate(
    (mynp_train_norm_0, mynp_train_norm_1, mynp_train_norm_2), axis=0
)
mynp_label_concat = np.concatenate((mynp_label_0, mynp_label_1, mynp_label_2), axis=0)

print("Train concat:", mynp_train_concat.shape, mynp_label_concat.shape)
print(
    "y stats: min/mean/max:",
    float(np.min(mynp_label_concat)),
    float(np.mean(mynp_label_concat)),
    float(np.max(mynp_label_concat)),
)



## === cell 9
cols = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
    "distance",
]

df_test_raw = pd.read_csv(TEST_PATH, sep=",")
df_test_raw = add_distance_feature(df_test_raw)

raw_means = (global_mean).astype(np.float32)

df_test_feat = df_test_raw[cols].copy()

for i, c in enumerate(cols):
    df_test_feat[c] = df_test_feat[c].astype(float)
    df_test_feat[c] = df_test_feat[c].fillna(float(raw_means[i]))

test_key_array = df_test_raw["key"].values
mynp_test = df_test_feat[cols].values.astype(np.float32)
mynp_test = norm_mynp_train(mynp_test, global_mean, global_std)

print("Test:", mynp_test.shape, test_key_array.shape)
print("Any NaN in test features after fill:", bool(np.isnan(mynp_test).any()))



## === cell 10
model_after_gridSearch = MLPRegressor(
    hidden_layer_sizes=(64, 64, 64),
    activation="relu",
    solver="adam",
    alpha=0.0001,
    batch_size=256,
    learning_rate_init=0.001,
    max_iter=30,
    random_state=42,
    verbose=False,
)

model_after_gridSearch.fit(mynp_train_concat, mynp_label_concat)



## === cell 11
test_predictions = model_after_gridSearch.predict(mynp_test).astype(np.float32)

test_predictions = np.clip(test_predictions, 0.0, 250.0)

print(test_predictions[:10])
print(
    "pred stats: min/mean/max:",
    float(np.min(test_predictions)),
    float(np.mean(test_predictions)),
    float(np.max(test_predictions)),
)



## === cell 12
df_output = pd.DataFrame({"key": test_key_array, "fare_amount": test_predictions})

sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
df_output = sample_sub[["key"]].merge(df_output, on="key", how="left")

df_output["fare_amount"] = df_output["fare_amount"].fillna(
    float(np.mean(test_predictions))
)

df_output.head()



## === cell 13
out_path = "submission_file.csv"
df_output.to_csv(out_path, index=False)
print("Wrote:", out_path, "rows:", len(df_output), "cols:", list(df_output.columns))
print(df_output.head())
