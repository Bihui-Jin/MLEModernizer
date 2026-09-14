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

5.20914

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

- What this solution (achieved 5.20914) has done: 'I fix the environment-breaking TensorFlow/Keras import usage (your current TF/Keras stack triggers a protobuf `MessageFactory.GetPrototype` crash) by switching to scikit-learn’s `MLPRegressor`, keeping the same core idea: a multilayer dense neural network trained on normalized numeric features with MSE loss. I also fix the missing/incorrect input file paths by reading the provided Kaggle `train.csv`/`test.csv` files and computing the missing `distance` feature from coordinates so the rest of your pipeline remains consistent. To ensure the notebook finishes under the time limit, I keep the training subset size similar to your original intent (you were reading 3×900k rows from custom shards, which aren’t available here). Finally, I generate a valid `submission_file.csv` with exactly the required `key,fare_amount` columns aligned to the test rows.'

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




## === cell 2
def haversine_distance_km(
    pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude
):
    """
    Vectorized haversine distance in kilometers.
    This replaces the missing precomputed 'distance' column from the original script's custom files.
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
    r_km = 6371.0
    return r_km * c


def add_distance_feature(df):
    df = df.copy()
    df["distance"] = haversine_distance_km(
        df["pickup_longitude"].values,
        df["pickup_latitude"].values,
        df["dropoff_longitude"].values,
        df["dropoff_latitude"].values,
    )
    return df




## === cell 3
def data_to_np(input_file, nrows=900000):
    """
    Fix: original code expected custom pre-sharded files with a 'distance' column.
    Here we read Kaggle-provided CSV and compute 'distance' on the fly.
    """
    df = pd.read_csv(input_file, sep=",", nrows=nrows)

    df = add_distance_feature(df)

    header_names = [
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
        "distance",
    ]

    df = df.dropna(
        subset=header_names + (["fare_amount"] if "fare_amount" in df.columns else [])
    )

    X = df[header_names].values.astype(np.float32)

    if "fare_amount" in df.columns:
        y = df["fare_amount"].values.astype(np.float32)
        return X, y
    else:
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

NROWS_PER_CHUNK = 250000  # 3 chunks -> 750k rows total

mynp_train_0, mynp_label_0 = data_to_np(TRAIN_PATH, nrows=NROWS_PER_CHUNK)

df1 = pd.read_csv(
    TRAIN_PATH, nrows=NROWS_PER_CHUNK, skiprows=range(1, 1 + NROWS_PER_CHUNK)
)
df1 = add_distance_feature(df1).dropna()
cols = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
    "distance",
]
mynp_train_1 = df1[cols].values.astype(np.float32)
mynp_label_1 = df1["fare_amount"].values.astype(np.float32)

df2 = pd.read_csv(
    TRAIN_PATH, nrows=NROWS_PER_CHUNK, skiprows=range(1, 1 + 2 * NROWS_PER_CHUNK)
)
df2 = add_distance_feature(df2).dropna()
mynp_train_2 = df2[cols].values.astype(np.float32)
mynp_label_2 = df2["fare_amount"].values.astype(np.float32)

print(mynp_train_0.shape, mynp_train_1.shape, mynp_train_2.shape)




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



## === cell 9

df_test = pd.read_csv(TEST_PATH, sep=",")
df_test = add_distance_feature(df_test)

df_test = df_test[["key"] + cols].dropna(subset=cols)

test_key_array = df_test["key"].values
mynp_test = df_test[cols].values.astype(np.float32)
mynp_test = norm_mynp_train(mynp_test, global_mean, global_std)

print("Test:", mynp_test.shape, test_key_array.shape)



## === cell 10

model_after_gridSearch = MLPRegressor(
    hidden_layer_sizes=(64, 64, 64),
    activation="relu",
    solver="adam",
    alpha=0.0001,
    batch_size=256,
    learning_rate_init=0.001,
    max_iter=30,  # practical runtime; original had EPOCHS=3 but Keras epochs differ from sklearn iterations
    random_state=42,
    verbose=False,
)

model_after_gridSearch.fit(mynp_train_concat, mynp_label_concat)



## === cell 11
test_predictions = model_after_gridSearch.predict(mynp_test).astype(np.float32)

test_predictions = np.maximum(test_predictions, 0.0)

print(test_predictions[:10])



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
