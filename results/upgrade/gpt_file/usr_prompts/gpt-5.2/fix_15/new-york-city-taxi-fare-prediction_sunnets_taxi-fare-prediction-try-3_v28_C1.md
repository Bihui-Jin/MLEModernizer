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
google-api-python-client==2.177.0
ipython==7.34.0
ipython-genutils==0.2.0
ipython_pygments_lexers==1.1.1
ipython-sql==0.5.0
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
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
sklearn-pandas==2.2.0
tf_keras==2.18.0

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

4.15415

# 6. Current score

9.07429

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 27.47228) has done: 'I fix the runtime issues caused by using Keras 3 APIs and by trying to read an external URL (internet is not available in Kaggle). The water-mask cleaning step be made robust by skipping it when the mask cannot be loaded, keeping the rest of your cleaning logic intact. I also correct the time parsing/feature logic bugs and use the proper Keras optimizer constructor so training runs. Finally, I ensure the submission is written with the required `key,fare_amount` columns to a `.csv` file.'
- What this solution (achieved 6.16458) has done: 'I fix the crash in the first cell caused by importing `tensorflow` in this Kaggle environment (protobuf incompatibility), since it prevents the notebook from running end-to-end. I also correct the training data read to include the `key` column and then explicitly drop it before scaling/training, which avoids accidentally training on the target-leaking `key` identifier and should move RMSE substantially toward your target. Finally, I ensure test/train feature columns are aligned and the submission is written as a valid `.csv` with `key,fare_amount` columns.'
- What this solution (achieved 6.03042) has done: 'I fix the hard crash in the first cell by preventing any `tensorflow` import attempt (it’s currently raising a protobuf `MessageFactory` error before your try/except can catch it). Then I fix a score-hurting logic issue in your cleaning: several “airport removal” filters use `&` with “!=” checks, which unintentionally drops far too many normal rows; switching those to the intended `|` keeps only the exact coordinate matches removed and should move RMSE toward your target without changing the model/training core. Finally, I keep your feature pipeline and model identical, ensure consistent column alignment, and still write a valid `key,fare_amount` submission `.csv`.'
- What this solution (achieved 6.01914) has done: 'I fix the hard crash in the first cell caused by importing `tf_keras` at module import time (it triggers the protobuf `MessageFactory` error in this environment). To keep your model/training logic unchanged, I delay-import `tf_keras` (and all Keras symbols) until just before the model is built/compiled, so the rest of the preprocessing runs and the notebook executes end-to-end. I also fix the incorrect input paths (`../input/...`) to the actual provided Kaggle paths so the data loads reliably and a submission `.csv` is always written. These changes are runtime/stability focused and should be score-neutral (your earlier score gains from the cleaning logic are preserved).'
- What this solution (achieved 41.32125) has done: 'Your current pipeline doesn’t yield a Kaggle score because it writes predictions without ever evaluating on Kaggle; to move RMSE toward the 4.15415 target, the smallest legitimate improvement is to (1) train on the full cleaned training sample instead of discarding half of it into `test_df` (which currently never reaches Kaggle), and (2) stop applying the heavy `clean()` filters to the Kaggle `test.csv` rows (those filters can drop rows and misalign `key` to predictions, producing invalid/shifted submissions and worse scores). These changes keep your model, features, scaler, loss, epochs, and batch size identical, but use more training signal and ensure the submission has exactly one prediction per test `key`. I also add a safety alignment assert and reindex to guarantee submission row order matches `testKaggle` keys. The result should produce a valid `submissiontry_water.csv` and improve RMSE substantially toward your target.'
- What this solution (achieved 244.36891) has done: 'We make the smallest changes that legitimately reduce RMSE from 41 toward the 4.15 target by (1) increasing the effective training signal while keeping your model and feature pipeline identical, and (2) reducing train/test distribution mismatch introduced by overly aggressive coordinate filtering. Concretely: raise `DATASET_SIZE` to use more rows (still within runtime), and relax the “only NYC bounding box” filter so trips that start/end slightly outside the box (common for airports/nearby areas) aren’t dropped, which otherwise biases the model and inflates Kaggle RMSE. We keep the same architecture, epochs, batch size, loss, and feature set; only data selection changes. Submission generation, key alignment, and post-processing remain unchanged.'
- What this solution (achieved 84.78799) has done: 'Your current RMSE (244) indicates the submission is likely badly miscalibrated rather than slightly underfit, so the smallest legitimate change is to make the model output closer to the evaluation target scale without changing the model/feature/training core. The key issue is that `MinMaxScaler` is fit on raw features, but distance features (`distance`, `manhattan`) can produce extreme values from a few bad coordinate pairs that slip through cleaning, which then squashes most samples into a tiny range and destabilizes training/predictions. I add a minimal, train-fitted clipping of the distance-based features (winsorization) on train/val/testKaggle before scaling, keeping the same features, model, loss, epochs, and training loop. This should dramatically reduce extreme predictions and move RMSE down toward the 4.15 target while preserving identical semantics.'
- What this solution (achieved 5.8441) has done: 'I fix the runtime crash caused by calling `y_scaler.inverse_transform` on an empty `prediction_scaled` array (your `test_df` is intentionally empty), by skipping local inverse-transform when there are 0 samples while keeping Kaggle test predictions unchanged. This also prevent the downstream `NameError` for `predictionKaggle`/`prediction` by ensuring those variables are always defined. I keep the model, features, cleaning, scaling, training loop, and prediction post-processing the same, only adding minimal guards and shape handling so the script completes and writes a valid `key,fare_amount` submission CSV.'
- What this solution (achieved 9.07429) has done: 'Your RMSE (5.8441) is worse than the target (4.15415), so we should make a small, low-risk improvement without changing your model or feature set. The biggest issue is that you’re training on a random split of the first 500k rows without shuffling, which makes the train/validation distributions time-skewed and typically hurts generalization to the Kaggle test distribution. I (1) read a random subsample from the full `train.csv` using `skiprows` (still 500k rows, same runtime budget) and (2) enable shuffling inside `model.fit` (training approach unchanged; it’s still standard supervised training), both of which usually reduce RMSE for this competition. Everything else (cleaning, feature engineering, scaler, architecture, loss, epochs, batch size, submission format) stays the same.'

# 9. Code solution

## === cell 0
import os

os.environ["KERAS_BACKEND"] = "numpy"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

TRAIN_PATH = "/kaggle/data/train.csv"
TEST_PATH = "/kaggle/data/test.csv"
SUBMISSION_NAME = "submissiontry_water.csv"

BATCH_SIZE = 256
EPOCHS = 30
LEARNING_RATE = 0.001

DATASET_SIZE = 500000

np.random.seed(1)

print("Using Keras backend:", os.environ.get("KERAS_BACKEND"))



## === cell 1
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


def read_random_train_sample(csv_path, n_rows, dtype, usecols, seed=1):
    try:
        with open(csv_path, "rb") as f:
            total_lines = 0
            buf_size = 1024 * 1024
            while True:
                b = f.read(buf_size)
                if not b:
                    break
                total_lines += b.count(b"\n")
        total_rows = max(0, total_lines - 1)
        if total_rows <= n_rows:
            return pd.read_csv(csv_path, dtype=dtype, usecols=usecols)

        rng = np.random.RandomState(seed)
        keep = rng.choice(total_rows, size=n_rows, replace=False)
        keep_set = set(int(i) for i in keep)

        skiprows = [i for i in range(1, total_rows + 1) if (i - 1) not in keep_set]

        return pd.read_csv(csv_path, dtype=dtype, usecols=usecols, skiprows=skiprows)
    except Exception as e:
        print(
            "Random subsample read failed; falling back to head(nrows). Error:", repr(e)
        )
        return pd.read_csv(
            csv_path,
            nrows=n_rows,
            dtype=dtype,
            usecols=usecols,
        )


trainKaggle = read_random_train_sample(
    TRAIN_PATH,
    n_rows=DATASET_SIZE,
    dtype=datatypes,
    usecols=[
        "key",
        "fare_amount",
        "pickup_datetime",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
    ],
    seed=1,
)

testKaggle = pd.read_csv(
    TEST_PATH, dtype={k: v for k, v in datatypes.items() if k != "fare_amount"}
)



## === cell 2
train_df, validation_df = train_test_split(trainKaggle, test_size=0.10, random_state=1)

test_df = train_df.iloc[:0].copy()



## === cell 3
print("testKaggle Size %d" % len(testKaggle))
print("train_df Size %d" % len(train_df))
print("validation_df Size %d" % len(validation_df))
print("test_df Size %d" % len(test_df))



## === cell 4
train_df.describe()



## === cell 5
validation_df.describe()



## === cell 6
test_df.describe()




## === cell 7
def remove_datapoints_from_water(df):
    def lonlat_to_xy(longitude, latitude, dx, dy, BB):
        return (dx * (longitude - BB[0]) / (BB[1] - BB[0])).astype("int"), (
            dy - dy * (latitude - BB[2]) / (BB[3] - BB[2])
        ).astype("int")

    BB = (-74.5, -72.8, 40.5, 41.8)

    local_mask = "/kaggle/input/nyc-mask/nyc_mask-74.5_-72.8_40.5_41.8.png"
    if os.path.exists(local_mask):
        nyc_mask_img = plt.imread(local_mask)
    else:
        return df

    nyc_mask = nyc_mask_img[:, :, 0] > 0.9

    pickup_x, pickup_y = lonlat_to_xy(
        df.pickup_longitude,
        df.pickup_latitude,
        nyc_mask.shape[1],
        nyc_mask.shape[0],
        BB,
    )
    dropoff_x, dropoff_y = lonlat_to_xy(
        df.dropoff_longitude,
        df.dropoff_latitude,
        nyc_mask.shape[1],
        nyc_mask.shape[0],
        BB,
    )

    pickup_x = np.clip(pickup_x, 0, nyc_mask.shape[1] - 1)
    dropoff_x = np.clip(dropoff_x, 0, nyc_mask.shape[1] - 1)
    pickup_y = np.clip(pickup_y, 0, nyc_mask.shape[0] - 1)
    dropoff_y = np.clip(dropoff_y, 0, nyc_mask.shape[0] - 1)

    idx = nyc_mask[pickup_y, pickup_x] & nyc_mask[dropoff_y, dropoff_x]
    return df[idx]


def clean(df):
    print(" Old size: %d" % len(df))
    df = df.dropna(how="any", axis="rows")
    print(" New size after dropna: %d" % len(df))

    df = df[
        (df["dropoff_longitude"] != df["pickup_longitude"])
        & (df["dropoff_latitude"] != df["pickup_latitude"])
    ]
    print(" New size after removing same long lat: %d" % len(df))

    df = df[
        (df["dropoff_longitude"] != 0)
        & (df["pickup_longitude"] != 0)
        & (df["dropoff_latitude"] != 0)
        & (df["pickup_latitude"] != 0)
    ]
    print(" New size after removing 0 long lat: %d" % len(df))

    MinMax = (-75.5, -72.0, 40.0, 42.0)
    df = df[
        (MinMax[0] <= df["pickup_longitude"]) & (df["pickup_longitude"] <= MinMax[1])
    ]
    df = df[
        (MinMax[0] <= df["dropoff_longitude"]) & (df["dropoff_longitude"] <= MinMax[1])
    ]
    df = df[(MinMax[2] <= df["pickup_latitude"]) & (df["pickup_latitude"] <= MinMax[3])]
    df = df[
        (MinMax[2] <= df["dropoff_latitude"]) & (df["dropoff_latitude"] <= MinMax[3])
    ]
    print(" New size after only NYC (relaxed BB): %d" % len(df))

    if "fare_amount" in df.columns:
        df = df[(0 < df["fare_amount"]) & (df["fare_amount"] <= 50)]
        print(" New size after removing outliers: %d" % len(df))

    df = df[(df["passenger_count"] > 0) & (df["passenger_count"] <= 6)]
    print(" New size after removing 6=>passenger_count > 0 : %d" % len(df))

    nyc_coord = (40.7141667, -74.0063889)
    fk_coord = (40.639722, -73.778889)
    ewr_coord = (40.6925, -74.168611)
    lga_coord = (40.77725, -73.872611)
    sol_coord = (40.6892, -74.0445)

    df = df[
        (nyc_coord[1] != df["pickup_longitude"])
        | (df["pickup_latitude"] != nyc_coord[0])
    ]
    df = df[
        (nyc_coord[1] != df["dropoff_longitude"])
        | (df["dropoff_latitude"] != nyc_coord[0])
    ]
    print(" New size after NY airport: %d" % len(df))

    df = df[
        (fk_coord[1] != df["pickup_longitude"]) | (df["pickup_latitude"] != fk_coord[0])
    ]
    df = df[
        (fk_coord[1] != df["dropoff_longitude"])
        | (df["dropoff_latitude"] != fk_coord[0])
    ]
    print(" New size after jfk airport: %d" % len(df))

    df = df[
        (ewr_coord[1] != df["pickup_longitude"])
        | (df["pickup_latitude"] != ewr_coord[0])
    ]
    df = df[
        (ewr_coord[1] != df["dropoff_longitude"])
        | (df["dropoff_latitude"] != ewr_coord[0])
    ]
    print(" New size after ewr airport: %d" % len(df))

    df = df[
        (lga_coord[1] != df["pickup_longitude"])
        | (df["pickup_latitude"] != lga_coord[0])
    ]
    df = df[
        (lga_coord[1] != df["dropoff_longitude"])
        | (df["dropoff_latitude"] != lga_coord[0])
    ]
    print(" New size after lgr airport: %d" % len(df))

    df = df[
        (sol_coord[1] != df["pickup_longitude"])
        | (df["pickup_latitude"] != sol_coord[0])
    ]
    df = df[
        (sol_coord[1] != df["dropoff_longitude"])
        | (df["dropoff_latitude"] != sol_coord[0])
    ]
    print(" New size after sol removed: %d" % len(df))

    print("Old size: %d" % len(df))
    df2 = remove_datapoints_from_water(df)
    if len(df2) != len(df):
        df = df2
    print("New size: %d" % len(df))

    print(" New size: %d" % len(df))
    return df


def manhattan(pickup_lat, pickup_long, dropoff_lat, dropoff_long):
    return np.abs(dropoff_lat - pickup_lat) + np.abs(dropoff_long - pickup_long)


def add_time_features(df):
    dt = pd.to_datetime(df["pickup_datetime"], errors="coerce", utc=False)
    df["year"] = dt.dt.year.astype("int16")
    df["month"] = dt.dt.month.astype("int8")
    df["day"] = dt.dt.day.astype("int8")
    df["hour"] = dt.dt.hour.astype("int8")
    df["weekday"] = dt.dt.weekday.astype("int8")

    h = df["hour"].astype("int16")
    wd = df["weekday"].astype("int16")

    df["night"] = (((h > 20) | (h < 6)) & (wd < 5)).astype("int8")
    df["late_night"] = (h <= 3).astype("int8")
    df["rush_hour"] = (((h >= 16) & (h <= 20)) & (wd < 5)).astype("int8")

    df["pickup_datetime"] = df["pickup_datetime"].astype(str)
    return df


def add_coordinate_features(df):
    lat1 = df["pickup_latitude"]
    lat2 = df["dropoff_latitude"]
    lon1 = df["pickup_longitude"]
    lon2 = df["dropoff_longitude"]
    df["latdiff"] = lat1 - lat2
    df["londiff"] = lon1 - lon2
    return df


def distance(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295  # Pi/180
    a = (
        0.5
        - np.cos((lat2 - lat1) * p) / 2
        + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    )
    return 0.6213712 * 12742 * np.arcsin(np.sqrt(a))


def distanceP(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295
    a = (
        0.5
        - np.cos((lat2 - lat1) * p) / 2
        + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    )
    return 0.6213712 * 12742 * np.arcsin(np.sqrt(a))


def add_distances_features(df):
    lat1 = df["pickup_latitude"]
    lat2 = df["dropoff_latitude"]
    lon1 = df["pickup_longitude"]
    lon2 = df["dropoff_longitude"]
    df["manhattan"] = manhattan(lat1, lon1, lat2, lon2)
    df["distance"] = distance(lat1, lon1, lat2, lon2)
    return df


def output_submission(raw_test, prediction, id_column, prediction_column, file_name):
    df = pd.DataFrame(prediction, columns=[prediction_column])
    df[id_column] = raw_test[id_column].values
    df[[id_column, prediction_column]].to_csv(file_name, index=False)
    print("Output complete:", file_name, "shape=", df.shape)


def plot_loss_accuracy_rmse(history):
    plt.figure(figsize=(20, 10))
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "val"], loc="upper right")
    plt.show()

    if "rmse" in history.history:
        plt.figure(figsize=(20, 10))
        plt.plot(history.history["rmse"])
        plt.plot(history.history["val_rmse"])
        plt.title("Model rmse")
        plt.ylabel("rmse")
        plt.xlabel("epoch")
        plt.legend(["train", "val"], loc="upper right")
        plt.show()


def clip_distance_features_inplace(
    train_df, other_dfs, cols=("distance", "manhattan"), q_low=0.001, q_high=0.999
):
    bounds = {}
    for c in cols:
        if c in train_df.columns:
            lo = float(train_df[c].quantile(q_low))
            hi = float(train_df[c].quantile(q_high))
            if not np.isfinite(lo):
                lo = -np.inf
            if not np.isfinite(hi):
                hi = np.inf
            if hi < lo:
                lo, hi = hi, lo
            bounds[c] = (lo, hi)

    def _clip(df):
        for c, (lo, hi) in bounds.items():
            if c in df.columns:
                df[c] = df[c].clip(lower=lo, upper=hi)
        return df

    train_df = _clip(train_df)
    clipped_others = []
    for df in other_dfs:
        clipped_others.append(_clip(df))
    return train_df, clipped_others, bounds




## === cell 8
print("train_df clean")
train_df = clean(train_df)
print("validation_df clean")
validation_df = clean(validation_df)



## === cell 9
train_df.describe()



## === cell 10
validation_df.describe()



## === cell 11
print(
    "testKaggle clean skipped to preserve 1:1 key alignment with the provided test.csv"
)
testKaggle = testKaggle.dropna(how="any", axis="rows").copy()

print("train_df add_time_features")
train_df = add_time_features(train_df)
print("validation_df add_time_features")
validation_df = add_time_features(validation_df)
print("test_df add_time_features")
test_df = add_time_features(test_df)
print("testKaggle add_time_features")
testKaggle = add_time_features(testKaggle)



## === cell 12
print("train_df add_coordinate_features")
train_df = add_coordinate_features(train_df)
print("validation_df add_coordinate_features")
validation_df = add_coordinate_features(validation_df)
print("test_df add_coordinate_features Disabled!")
test_df = add_coordinate_features(test_df)
print("testKaggle add_coordinate_features")
testKaggle = add_coordinate_features(testKaggle)



## === cell 13
print("train_df add_distances_features")
train_df = add_distances_features(train_df)
print("validation_df add_distances_features")
validation_df = add_distances_features(validation_df)
print("test_df add_distances_features")
test_df = add_distances_features(test_df)
print("testKaggle add_distances_features")
testKaggle = add_distances_features(testKaggle)

print("Done with Adding features")



## === cell 14
train_df, (validation_df, test_df, testKaggle), clip_bounds = (
    clip_distance_features_inplace(
        train_df,
        [validation_df, test_df, testKaggle],
        cols=("distance", "manhattan"),
        q_low=0.001,
        q_high=0.999,
    )
)
print("Clipping bounds used:", clip_bounds)



## === cell 15
train_df.describe()



## === cell 16
validation_df.describe()



## === cell 17
dropped_columns = ["passenger_count", "pickup_datetime", "key"]

train_df = train_df.drop(dropped_columns, axis=1)
test_df = test_df.drop(dropped_columns, axis=1)
validation_df = validation_df.drop(dropped_columns, axis=1)

testKaggle_clean = testKaggle.drop(
    ["passenger_count", "pickup_datetime", "key"], axis=1
)

print("Done with dropped_columns")



## === cell 18
train_labels = train_df["fare_amount"].values
validation_labels = validation_df["fare_amount"].values
test_labels = (
    test_df["fare_amount"].values if "fare_amount" in test_df.columns else np.array([])
)

train_df = train_df.drop(["fare_amount"], axis=1)
validation_df = validation_df.drop(["fare_amount"], axis=1)
if "fare_amount" in test_df.columns:
    test_df = test_df.drop(["fare_amount"], axis=1)

print("Done with Labels")



## === cell 19
train_df.shape



## === cell 20
test_df.shape



## === cell 21
validation_df.shape



## === cell 22
testKaggle_clean = testKaggle_clean.reindex(columns=train_df.columns)

scaler = preprocessing.MinMaxScaler()
train_df_scaled = scaler.fit_transform(train_df)
validation_df_scaled = scaler.transform(validation_df)
test_scaled = (
    scaler.transform(test_df)
    if len(test_df)
    else np.zeros((0, train_df_scaled.shape[1]), dtype=np.float32)
)
testKaggle_scaled = scaler.transform(testKaggle_clean)



## === cell 23
import keras
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization
from keras import optimizers
from keras import regularizers
from keras import ops


def rmse(y_true, y_pred):
    y_true = ops.convert_to_tensor(y_true)
    y_pred = ops.convert_to_tensor(y_pred)
    return ops.sqrt(ops.mean(ops.square(y_pred - y_true), axis=-1))




## === cell 24
y_scaler = preprocessing.StandardScaler()
train_labels_scaled = y_scaler.fit_transform(train_labels.reshape(-1, 1)).reshape(-1)
validation_labels_scaled = y_scaler.transform(validation_labels.reshape(-1, 1)).reshape(
    -1
)

model = Sequential()
model.add(
    Dense(
        256,
        activation="relu",
        input_dim=train_df_scaled.shape[1],
        activity_regularizer=regularizers.l1(0.01),
    )
)
model.add(BatchNormalization())
model.add(Dense(128, activation="relu"))
model.add(BatchNormalization())
model.add(Dense(64, activation="relu"))
model.add(BatchNormalization())
model.add(Dense(32, activation="relu"))
model.add(BatchNormalization())
model.add(Dense(8, activation="relu"))
model.add(BatchNormalization())
model.add(Dense(1))

adam = optimizers.Adam(learning_rate=LEARNING_RATE)
model.compile(loss="mean_squared_error", optimizer=adam, metrics=["mae", rmse])

print("Dataset size: %s" % DATASET_SIZE)
print("Epochs: %s" % EPOCHS)
print("Learning rate: %s" % LEARNING_RATE)
print("Batch size: %s" % BATCH_SIZE)
print("Input dimension: %s" % train_df_scaled.shape[1])
print("Features used: %s" % list(train_df.columns))
model.summary()

history = None
trained_with = None

try:
    history = model.fit(
        x=train_df_scaled,
        y=train_labels_scaled,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        verbose=1,
        validation_data=(validation_df_scaled, validation_labels_scaled),
        shuffle=True,
    )
    trained_with = "keras"
except Exception as e:
    print(
        "Keras model.fit() is not available with this backend; falling back to sklearn. Error:",
        repr(e),
    )
    from sklearn.neural_network import MLPRegressor

    mlp = MLPRegressor(
        hidden_layer_sizes=(256, 128, 64, 32, 8),
        activation="relu",
        solver="adam",
        alpha=0.0001,
        batch_size=BATCH_SIZE,
        learning_rate_init=LEARNING_RATE,
        max_iter=EPOCHS,
        shuffle=True,
        random_state=1,
        verbose=True,
    )
    mlp.fit(train_df_scaled, train_labels_scaled)
    trained_with = "sklearn"
    model = mlp  # reuse downstream "model" name

print("Training engine:", trained_with)



## === cell 25
try:
    from IPython.display import SVG
    from keras.utils import model_to_dot

    if trained_with == "keras":
        SVG(model_to_dot(model).create(prog="dot", format="svg"))
    else:
        print("Model visualization skipped: sklearn fallback model.")
except Exception as e:
    print("Model visualization skipped:", repr(e))



## === cell 26
if "history" in globals() and history is not None:
    plot_loss_accuracy_rmse(history)
else:
    print("Skipping plot: history is not available.")



## === cell 27
if trained_with == "keras":
    prediction_scaled = (
        model.predict(test_scaled, batch_size=128, verbose=0)
        if len(test_scaled)
        else np.zeros((0, 1), dtype=np.float32)
    )
    predictionKaggle_scaled = model.predict(
        testKaggle_scaled, batch_size=128, verbose=1
    )
    prediction_scaled = np.asarray(prediction_scaled).reshape(-1, 1)
    predictionKaggle_scaled = np.asarray(predictionKaggle_scaled).reshape(-1, 1)
else:
    prediction_scaled = (
        model.predict(test_scaled).reshape(-1, 1)
        if len(test_scaled)
        else np.zeros((0, 1), dtype=np.float32)
    )
    predictionKaggle_scaled = model.predict(testKaggle_scaled).reshape(-1, 1)

predictionKaggle = y_scaler.inverse_transform(predictionKaggle_scaled).reshape(-1, 1)

if prediction_scaled.shape[0] > 0:
    prediction = y_scaler.inverse_transform(prediction_scaled).reshape(-1, 1)
else:
    prediction = np.zeros((0, 1), dtype=np.float32)



## === cell 28
try:
    if len(test_labels) and len(prediction):
        mse_sample = np.mean((test_labels[:1000] - prediction[:1000, 0]) ** 2)
        print("Sample MSE (first 1000):", float(mse_sample))
    else:
        print("Sample MSE skipped: no local test split in this run.")
except Exception as e:
    print("Sample MSE computation skipped:", repr(e))



## === cell 29
predictionKaggle = np.asarray(predictionKaggle).reshape(-1)
predictionKaggle = np.where(np.isfinite(predictionKaggle), predictionKaggle, 0.0)

predictionKaggle = np.maximum(predictionKaggle, 0.0)
predictionKaggle = np.minimum(predictionKaggle, 60.0)

if len(predictionKaggle) != len(testKaggle):
    raise ValueError(
        f"Prediction length {len(predictionKaggle)} does not match test rows {len(testKaggle)}; "
        "refusing to write a misaligned submission."
    )



## === cell 30
if not SUBMISSION_NAME.lower().endswith(".csv"):
    SUBMISSION_NAME = SUBMISSION_NAME + ".csv"

output_submission(testKaggle, predictionKaggle, "key", "fare_amount", SUBMISSION_NAME)



## === cell 31
if len(prediction) > 10000:
    print(prediction[10000])
    print(test_labels[10000] if len(test_labels) > 10000 else "no test_labels")
else:
    print("Prediction/test split sizes:", len(prediction), len(test_labels))
