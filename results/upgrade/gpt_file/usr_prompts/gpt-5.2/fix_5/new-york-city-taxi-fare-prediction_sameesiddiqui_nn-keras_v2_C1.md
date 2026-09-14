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

3.62636

# 6. Current score

5.24361

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 5.22189) has done: 'I fix the feature-extraction dtype check that crashes on timezone-aware datetimes by using pandas’ datetime type utilities instead of `np.issubdtype`. I also ensure `validate_df` gets the same engineered columns before outlier removal (so it doesn’t depend on mutation order), which prevents `X` from being undefined and unblocks the validation and test inference cells. To keep core logic identical, I won’t change the model, training loop, or feature engineering math—only make the datetime check robust and the pipeline deterministic end-to-end. Finally, I make sure `mean_y` is always defined before it’s used for clipping and that a valid `nn_submission.csv` is written.'
- What this solution (achieved 5.13764) has done: 'Your current gap to the target (5.22189 vs 3.62636, lower is better) suggests the model is underperforming mainly due to feature binning/one-hot issues rather than architecture. The smallest high-impact fix that preserves your core model/training is to correct `bucketize_feature`: it currently turns all negative longitudes into the “9” bin, collapsing most NYC coordinates into one bucket and destroying the cross features. I change bucketization to use proper bin edges via `np.digitize` (still 10 bins, still quantile-based from `train_df`, same downstream one-hot/cross sizes), and keep everything else (features, model, epochs, batch size, loss) the same. This should materially reduce RMSE and move you closer to the target while keeping the pipeline deterministic and producing the same submission schema.'
- What this solution (achieved 5.24361) has done: 'Your current RMSE is worse than the target, so we should make a small, legitimate improvement without changing the model or training loop. The biggest remaining low-risk issue is that your validation and test feature extraction re-computes bucket edges from the *current* df via a function that implicitly depends on the global `train_df`; this is fragile and can drift if `train_df` changes (e.g., due to outlier filtering) and also recomputes quantiles repeatedly. I freeze the quantile bin edges once (after outlier removal) and use those fixed edges consistently for train/validation/test, keeping the same 10-bin scheme, the same one-hot/cross sizes, and the same network. I also add the same high-end clipping you already do for validation to test predictions (still using the same `mean_y`) to reduce extreme outlier penalties under RMSE.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

INPUT_DIR = "../input"
if not os.path.exists(INPUT_DIR):
    INPUT_DIR = "/kaggle/input"

print("INPUT_DIR:", INPUT_DIR)
print(os.listdir(INPUT_DIR)[:20])



## === cell 1
import sys
import subprocess


def _ensure_protobuf_compat():
    try:
        import google.protobuf
        from packaging.version import parse as vparse

        cur = vparse(google.protobuf.__version__)
        if cur.major >= 5:
            print(
                "Downgrading protobuf from",
                google.protobuf.__version__,
                "to 4.25.3 for TensorFlow compatibility...",
            )
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", "protobuf==4.25.3"]
            )
            import importlib

            importlib.invalidate_caches()
    except Exception as e:
        print("Warning: protobuf compatibility check failed:", repr(e))


_ensure_protobuf_compat()



## === cell 2
datatypes = {
    "key": "str",
    "fare_amount": "float32",
    "pickup_datetime": "str",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "uint8",
}

train_path = os.path.join(INPUT_DIR, "train.csv")
test_path = os.path.join(INPUT_DIR, "test.csv")
sample_path = os.path.join(INPUT_DIR, "sample_submission.csv")

train_df = pd.read_csv(train_path, nrows=1_000_000, dtype=datatypes)
train_df.head()



## === cell 3
train_df.describe()



## === cell 4
test_df = pd.read_csv(
    test_path, dtype={k: v for k, v in datatypes.items() if k != "fare_amount"}
)
test_df.describe()




## === cell 5
def distance_between_points(df):
    df["diff_lat"] = (df["dropoff_latitude"] - df["pickup_latitude"]).abs()
    df["diff_long"] = (df["dropoff_longitude"] - df["pickup_longitude"]).abs()
    df["manhattan_dist"] = df["diff_lat"] + df["diff_long"]


distance_between_points(train_df)




## === cell 6
def extract_date_details(df):
    dt = pd.to_datetime(df["pickup_datetime"], errors="coerce", utc=True)
    df["pickup_datetime"] = dt
    df["year"] = dt.dt.year.astype("Int64")
    df["month"] = dt.dt.month.astype("Int64")
    df["day"] = dt.dt.weekday.astype("Int64")
    df["hour"] = dt.dt.hour.astype("Int64")


extract_date_details(train_df)
train_df.head()




## === cell 7
def remove_outliers(df):
    df = df.dropna()

    if "diff_lat" not in df.columns or "diff_long" not in df.columns:
        distance_between_points(df)

    df = df[(df["diff_lat"] < 5.0) & (df["diff_long"] < 5.0)]
    df = df[(df["diff_lat"] > 0.001) & (df["diff_long"] > 0.001)]

    df = df[(df["pickup_longitude"] < -72) & (df["pickup_longitude"] > -75)]
    df = df[(df["pickup_latitude"] < 42) & (df["pickup_latitude"] > 39)]
    df = df[(df["dropoff_longitude"] < -72) & (df["dropoff_longitude"] > -75)]
    df = df[(df["dropoff_latitude"] < 42) & (df["dropoff_latitude"] > 39)]

    if "fare_amount" in df.columns:
        df = df[
            (df["fare_amount"] > 2.50)
            & (df["fare_amount"] < 200)
            & (df["passenger_count"] <= 6)
            & (df["passenger_count"] > 0)
        ]
    else:
        df = df[(df["passenger_count"] <= 6) & (df["passenger_count"] > 0)]
    return df


train_df = remove_outliers(train_df)
len(train_df)



## === cell 8
plt.scatter(train_df[:10000]["manhattan_dist"], train_df[:10000]["fare_amount"])
plt.xlabel("manhattan distance")
plt.ylabel("fare")
plt.show()



## === cell 9
train_df.describe()




## === cell 10
def convert_to_one_hot(column, num_buckets, df, starting_index=0):
    df_size = df.shape[0]
    one_hots = np.zeros((df_size, num_buckets), dtype="byte")
    idx = df[column].astype("int32").values - starting_index
    idx = np.clip(idx, 0, num_buckets - 1)
    one_hots[np.arange(df_size), idx] = 1
    return one_hots




## === cell 11
year = convert_to_one_hot("year", 7, train_df, 2009)
hour = convert_to_one_hot("hour", 24, train_df, 0)



## === cell 12
train_df.shape




## === cell 13
def compute_bucket_edges(train_df_local, column):
    return (
        train_df_local[column]
        .quantile([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        .values.astype(np.float32)
    )


BUCKET_EDGES = {
    "pickup_longitude": compute_bucket_edges(train_df, "pickup_longitude"),
    "pickup_latitude": compute_bucket_edges(train_df, "pickup_latitude"),
    "dropoff_longitude": compute_bucket_edges(train_df, "dropoff_longitude"),
    "dropoff_latitude": compute_bucket_edges(train_df, "dropoff_latitude"),
}


def bucketize_feature(df, column, edges=None):
    if edges is None:
        edges = BUCKET_EDGES[column]
    vals = df[column].values.astype(np.float32)
    bins = np.digitize(vals, edges, right=False).astype("byte")
    return bins


p_long = bucketize_feature(train_df, "pickup_longitude")
p_lat = bucketize_feature(train_df, "pickup_latitude")
d_long = bucketize_feature(train_df, "dropoff_longitude")
d_lat = bucketize_feature(train_df, "dropoff_latitude")



## === cell 14
print(p_long)
print(p_lat)




## === cell 15
def feature_cross(a1, a2):
    rows = a1.shape[0]
    cols = 100
    cross = np.zeros((rows, cols), dtype="byte")
    cross[np.arange(rows), (a1.astype("int32") * 10) + a2.astype("int32")] = 1
    return cross


p_lat_x_long = feature_cross(p_lat, p_long)
d_lat_x_long = feature_cross(d_lat, d_long)



## === cell 16
unique, counts = np.unique(p_long, return_counts=True)
print(np.asarray((unique, counts)).T)
unique, counts = np.unique(p_lat, return_counts=True)
print(np.asarray((unique, counts)).T)



## === cell 17
print(p_lat_x_long.shape)
print(d_lat_x_long.shape)
print(year.shape)
print(hour.shape)
print(train_df["manhattan_dist"].shape)



## === cell 18
manhattan = train_df["manhattan_dist"].values.reshape(len(train_df), 1)

train_X = np.concatenate((p_lat_x_long, d_lat_x_long, year, hour, manhattan), axis=1)
train_y = train_df["fare_amount"].values
print(train_X.shape)
print(train_y.shape)



## === cell 19
validate_df = pd.read_csv(
    train_path, skiprows=range(1, 1_000_001), nrows=10_000, dtype=datatypes
)



## === cell 20
distance_between_points(validate_df)
extract_date_details(validate_df)
validate_df = remove_outliers(validate_df)


def extract_features(df):
    if "pickup_datetime" not in df.columns or not pd.api.types.is_datetime64_any_dtype(
        df["pickup_datetime"]
    ):
        extract_date_details(df)
    if "manhattan_dist" not in df.columns:
        distance_between_points(df)

    p_lo = bucketize_feature(df, "pickup_longitude")
    p_la = bucketize_feature(df, "pickup_latitude")
    d_lo = bucketize_feature(df, "dropoff_longitude")
    d_la = bucketize_feature(df, "dropoff_latitude")
    p_la_x_lo = feature_cross(p_la, p_lo)
    d_la_x_lo = feature_cross(d_la, d_lo)
    yr = convert_to_one_hot("year", 7, df, 2009)
    hr = convert_to_one_hot("hour", 24, df, 0)
    manhattan_local = df["manhattan_dist"].values.reshape(len(df), 1)

    X = np.concatenate((p_la_x_lo, d_la_x_lo, yr, hr, manhattan_local), axis=1)
    return X


X = extract_features(validate_df)
true_y = validate_df["fare_amount"].values



## === cell 21
import tensorflow as tf
from tensorflow.keras import layers

model = tf.keras.Sequential()
model.add(layers.Dense(128, activation="relu", input_dim=232))
model.add(layers.Dense(128, activation="relu"))
model.add(layers.Dense(1))
model.compile(optimizer="adam", loss="mse", metrics=["mae"])
model.fit(train_X, train_y, epochs=5, batch_size=256, verbose=1)



## === cell 22
result = model.predict(X, verbose=0).flatten()



## === cell 23
mean_y = float(np.mean(train_df["fare_amount"].values))
result[result > 100] = mean_y
diff = true_y - result
mse = np.sum(diff**2) / len(diff)
rmse = np.sqrt(mse)
print(rmse)



## === cell 24
distance_between_points(test_df)
extract_date_details(test_df)
X_test = extract_features(test_df)
pred_y_test = model.predict(X_test, verbose=0).flatten()

pred_y_test[pred_y_test > 100] = mean_y



## === cell 25
submission = pd.DataFrame(
    {"key": test_df["key"].values, "fare_amount": pred_y_test.astype(np.float32)}
)

submission.loc[submission["fare_amount"] < 0, "fare_amount"] = mean_y

submission.to_csv("nn_submission.csv", index=False)
print(submission.head())
print("Wrote nn_submission.csv with shape:", submission.shape)
