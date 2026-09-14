# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

5.21833

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

- What this solution (achieved 5.21833) has done: 'I fix the environment-breaking TensorFlow/Keras imports by removing the incompatible scikit-learn/Keras wrapper grid-search section and switching to `tf.keras` only, which is enough to train and predict with the same neural-network core logic. I also fix the file-path issues by reading the provided `../input/new-york-city-taxi-fare-prediction/train.csv` and `test.csv` instead of non-existent `my-taxi-fare-data/*` files. Because your code expects a `distance` feature that is not in the raw CSVs, I minimally add it (haversine distance) during preprocessing for both train and test. Finally, I ensure normalization uses the training-set mean/std (not test-set stats) and always writes a valid `submission_file.csv` with columns `key,fare_amount`.'

# 9. Code solution

## === cell 0
print("hello moto")



## === cell 1
import os
import math
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

np.random.seed(42)
tf.random.set_seed(42)

print("TF version:", tf.__version__)

TRAIN_PATH = "../input/new-york-city-taxi-fare-prediction/train.csv"
TEST_PATH = "../input/new-york-city-taxi-fare-prediction/test.csv"
SAMPLE_SUB_PATH = "../input/new-york-city-taxi-fare-prediction/sample_submission.csv"

for p in [TRAIN_PATH, TEST_PATH, SAMPLE_SUB_PATH]:
    if not os.path.exists(p):
        raise FileNotFoundError(f"Expected file not found: {p}")




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

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
    mynp_train_norm = (mynp_train - mean) / std
    return mynp_train_norm




## === cell 6
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

[m.shape for m in mynp_train_list], [l.shape for l in mynp_label_list]




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
global_mean, global_std



## === cell 11
mynp_train_norm_0 = norm_mynp_train(mynp_train_list[0], global_mean, global_std)
mynp_train_norm_1 = norm_mynp_train(mynp_train_list[1], global_mean, global_std)
mynp_train_norm_2 = norm_mynp_train(mynp_train_list[2], global_mean, global_std)

mynp_train_concat = np.concatenate(
    (mynp_train_norm_0, mynp_train_norm_1, mynp_train_norm_2), axis=0
)
mynp_label_concat = np.concatenate((mynp_label_0, mynp_label_1, mynp_label_2), axis=0)

mynp_train_concat.shape, mynp_label_concat.shape




## === cell 12
def build_model(shape_of_np_array):
    model = keras.Sequential(
        [
            keras.layers.Dense(
                64, activation=tf.nn.relu, input_shape=(shape_of_np_array,)
            ),
            keras.layers.Dense(64, activation=tf.nn.relu),
            keras.layers.Dense(64, activation=tf.nn.relu),
            keras.layers.Dense(1),
        ]
    )

    optimizer = keras.optimizers.RMSprop(learning_rate=0.001)

    model.compile(loss="mse", optimizer=optimizer, metrics=["mae"])
    return model




## === cell 13
class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 5 == 0:
            print("epoch", epoch)
        print(".", end="")




## === cell 14
early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=20)



## === cell 15
EPOCHS = 7

model_after_gridSearch = build_model(mynp_train_concat.shape[1])
history = model_after_gridSearch.fit(
    mynp_train_concat,
    mynp_label_concat,
    epochs=EPOCHS,
    validation_split=0.2,
    verbose=0,
    callbacks=[PrintDot()],
)
print("\nTraining done.")



## === cell 16
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
mynp_test = (mynp_test - global_mean.astype(np.float32)) / global_std.astype(np.float32)

mynp_test.shape



## === cell 17
test_predictions = model_after_gridSearch.predict(mynp_test, verbose=0).reshape(-1)

test_predictions = np.clip(test_predictions, 0.0, None)

test_predictions[:5], test_predictions.shape



## === cell 18
test_key_array = df_test["key"].values
df_output = pd.DataFrame({"key": test_key_array, "fare_amount": test_predictions})

df_output = df_output[["key", "fare_amount"]]
df_output.head()



## === cell 19
out_path = "submission_file.csv"
df_output.to_csv(out_path, index=False)

print("Wrote:", out_path, "rows:", len(df_output))
print(df_output.dtypes)
