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

4.44194

# 6. Current score

5.56825

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 6.07597) has done: 'I fix the Keras import crash by switching to `tf_keras` (available in your environment) while keeping the same Sequential/Dense/BatchNorm architecture and training loop. I fix the timezone conversion error in `add_time_features` by parsing as UTC and converting to timezone-naive before extracting date parts. I also fix the feature-name mismatch during scaling by ensuring all splits (train/validation/test/kaggle-test) receive the same feature-engineering and column drops, then passing NumPy arrays into `MinMaxScaler` to avoid strict feature-name checks. Finally, I ensure the pipeline reaches inference and writes a valid `submissiontry_water.csv` with `key,fare_amount`.'
- What this solution (achieved 27.54086) has done: 'I fix the import-time crash by switching the Keras imports to `tf_keras`, which is installed and compatible in this environment, while keeping the same Sequential/Dense/BatchNorm model and training loop. I also fix the scaler failure by ensuring `key` (a string) is dropped from the feature matrices (it was still present in train/val/test, causing “could not convert string to float”). Finally, I add a small safety alignment step so train/validation/test/testKaggle have identical numeric feature columns before scaling, and ensure the pipeline always writes a valid `submissiontry_water.csv` with `key,fare_amount`.'
- What this solution (achieved 6.90849) has done: 'I fix the import-time crash (`MessageFactory` / protobuf issue) by removing the `tf_keras` dependency and switching to `sklearn`’s `RandomForestRegressor`, which is already installed and stable in this environment. This keeps the same overall pipeline (same cleaning + time/coordinate/distance feature engineering, same train/val/test split, same scaling, same RMSE evaluation semantics) while avoiding the broken deep-learning stack. I also make the datetime feature creation faster and deterministic by avoiding `df.apply` and computing the night/rush flags vectorized, which is score-neutral but prevents slowdowns. Finally, I keep the submission format identical and ensure the script always writes `submissiontry_water.csv` with `key,fare_amount`.'
- What this solution (achieved 6.86054) has done: 'Your current RMSE (6.908) is worse than the target (4.441), so we should make a small, legitimate improvement without changing the core model choice (RandomForest) or the overall pipeline. The biggest score leak here is that you drop `passenger_count`, which is a real predictive feature, and RandomForest does not need MinMax scaling (scaling can slightly hurt tree splits). I (1) keep `passenger_count` as a feature, (2) stop scaling for the RF while keeping the same train/val/test split and feature engineering, and (3) add a tiny amount of robustness by filling any remaining NaNs after reindexing so inference can’t degrade from missing values. These are minimal changes that typically improve NYC taxi fare baselines substantially and should move RMSE toward your target.'
- What this solution (achieved 5.59921) has done: 'Your current RMSE is materially worse than the target, so we should make a small, legitimate improvement without changing the overall approach (same RandomForest, same feature engineering, same split/training loop). The biggest likely score drag is label noise from remaining bad rows (e.g., extreme fare/distance mismatches) that survive current cleaning; adding one conservative, competition-standard filter tying fare to trip distance typically improves generalization. I add a minimal “distance sanity” filter (drop very short trips with high fares and very long trips with tiny fares) using your already-created `distance` feature, applied only on train/val/test (not on Kaggle test). I also keep everything deterministic and ensure the submission CSV is still written with `key,fare_amount`.'
- What this solution (achieved 5.38807) has done: 'Your current RMSE (5.599) is still well above the target (4.442), so we should make a small, legitimate improvement without changing the overall pipeline or switching model families. The biggest low-risk gain for RandomForest on this competition is to add the standard longitude/latitude “bearing” and “haversine distance (km)” features (you already compute a rough Euclidean distance and Manhattan distance), which improves generalization while keeping the same training loop and estimator. I also set `max_features="sqrt"` (a common RF default that reduces overfitting/noise) while keeping the same RF approach and row sampling. Everything else (cleaning, split, label handling, submission writing) stays the same and still produces `submissiontry_water.csv`.'
- What this solution (achieved 5.61176) has done: 'We make two minimal, score-relevant adjustments that keep your current RandomForest + feature engineering pipeline intact. First, we increase the training sample size (your current 80k is a key limiter) while keeping runtime reasonable, which typically reduces RMSE materially for this competition. Second, we add a standard, conservative geographic “center point” feature (midpoint lat/lon) that often helps tree models without changing the overall approach. Everything else (cleaning, splits, RF model family, submission format/path) stays the same and it still write `submissiontry_water.csv`.'
- What this solution (achieved 5.56825) has done: 'We need to reduce RMSE from 5.61 toward 4.44 (lower is better), so we should make a small, legitimate improvement without changing the overall RandomForest approach or feature engineering. The biggest low-risk gain here is to stop randomly splitting the first 300k rows into “train/validation/test”, because this competition has strong time drift; instead we do a deterministic time-based split using `pickup_datetime` so validation better matches Kaggle’s generalization and the model trains on earlier rides and validates on later rides. This keeps the same estimator, same features, same cleaning, and same evaluation semantics, but typically improves leaderboard RMSE for NYC Taxi baselines. We also keep the submission writing unchanged and ensure the split is stable and fast.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

TRAIN_PATH = "../input/train.csv"
TEST_PATH = "../input/test.csv"
SUBMISSION_NAME = "submissiontry_water.csv"

BATCH_SIZE = 1000
EPOCHS = 30
LEARNING_RATE = 0.001

DATASET_SIZE = 300000

np.random.seed(1)



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

train_cols = [
    "key",
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
trainKaggle = pd.read_csv(
    TRAIN_PATH, nrows=DATASET_SIZE, dtype=datatypes, usecols=train_cols
)

testKaggle = pd.read_csv(
    TEST_PATH, dtype={k: v for k, v in datatypes.items() if k != "fare_amount"}
)



## === cell 2
trainKaggle["_pickup_dt"] = pd.to_datetime(
    trainKaggle["pickup_datetime"], errors="coerce", utc=True
)
trainKaggle = (
    trainKaggle.dropna(subset=["_pickup_dt"])
    .sort_values("_pickup_dt")
    .reset_index(drop=True)
)

n_total = len(trainKaggle)
n_test = int(0.50 * n_total)
n_val = int(0.10 * (n_total - n_test))

train_df = trainKaggle.iloc[: (n_total - n_test - n_val)].copy()
validation_df = trainKaggle.iloc[(n_total - n_test - n_val) : (n_total - n_test)].copy()
test_df = trainKaggle.iloc[(n_total - n_test) :].copy()

for _df in (train_df, validation_df, test_df):
    _df.drop(columns=["_pickup_dt"], inplace=True)



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
def remove_datapoints_from_water(df, mask_path=None):
    if mask_path is None or (
        isinstance(mask_path, str) and (not os.path.exists(mask_path))
    ):
        return df

    def lonlat_to_xy(longitude, latitude, dx, dy, BB):
        return (dx * (longitude - BB[0]) / (BB[1] - BB[0])).astype("int"), (
            dy - dy * (latitude - BB[2]) / (BB[3] - BB[2])
        ).astype("int")

    BB = (-74.5, -72.8, 40.5, 41.8)
    nyc_mask = plt.imread(mask_path)[:, :, 0] > 0.9

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

    MinMax = (-74.5, -72.8, 40.5, 41.8)
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
    print(" New size after only NYC: %d" % len(df))

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
        & (df["pickup_latitude"] != nyc_coord[0])
    ]
    df = df[
        (nyc_coord[1] != df["dropoff_longitude"])
        & (df["dropoff_latitude"] != nyc_coord[0])
    ]
    print(" New size after NY airport: %d" % len(df))

    df = df[
        (fk_coord[1] != df["pickup_longitude"]) & (df["pickup_latitude"] != fk_coord[0])
    ]
    df = df[
        (fk_coord[1] != df["dropoff_longitude"])
        & (df["dropoff_latitude"] != fk_coord[0])
    ]
    print(" New size after jfk airport: %d" % len(df))

    df = df[
        (ewr_coord[1] != df["pickup_longitude"])
        & (df["pickup_latitude"] != ewr_coord[0])
    ]
    df = df[
        (ewr_coord[1] != df["dropoff_longitude"])
        & (df["dropoff_latitude"] != ewr_coord[0])
    ]
    print(" New size after ewr airport: %d" % len(df))

    df = df[
        (lga_coord[1] != df["pickup_longitude"])
        & (df["pickup_latitude"] != lga_coord[0])
    ]
    df = df[
        (lga_coord[1] != df["dropoff_longitude"])
        & (df["dropoff_latitude"] != lga_coord[0])
    ]
    print(" New size after lgr airport: %d" % len(df))

    df = df[
        (sol_coord[1] != df["pickup_longitude"])
        & (df["pickup_latitude"] != sol_coord[0])
    ]
    df = df[
        (sol_coord[1] != df["dropoff_longitude"])
        & (df["dropoff_latitude"] != sol_coord[0])
    ]
    print(" New size after sol removed: %d" % len(df))

    print("Old size: %d" % len(df))
    df = remove_datapoints_from_water(df, mask_path=None)
    print("New size: %d" % len(df))

    print(" New size: %d" % len(df))
    return df


def manhattan(pickup_lat, pickup_long, dropoff_lat, dropoff_long):
    return np.abs(dropoff_lat - pickup_lat) + np.abs(dropoff_long - pickup_long)


def add_time_features(df):
    dt = pd.to_datetime(df["pickup_datetime"], errors="coerce", utc=True)
    dt = dt.dt.tz_convert(None)

    year = dt.dt.year.fillna(0).astype("int16")
    month = dt.dt.month.fillna(0).astype("int8")
    day = dt.dt.day.fillna(0).astype("int8")
    hour = dt.dt.hour.fillna(0).astype("int8")
    weekday = dt.dt.weekday.fillna(0).astype("int8")

    df["year"] = year
    df["month"] = month
    df["day"] = day
    df["hour"] = hour
    df["weekday"] = weekday

    df["pickup_datetime"] = dt.astype("datetime64[ns]").astype(str)

    df["night"] = (((hour >= 20) | (hour <= 6)) & (weekday < 5)).astype("int8")
    df["late_night"] = (hour <= 3).astype("int8")
    df["rush_hour"] = (((hour >= 16) & (hour <= 20)) & (weekday < 5)).astype("int8")
    return df


def add_coordinate_features(df):
    lat1 = df["pickup_latitude"]
    lat2 = df["dropoff_latitude"]
    lon1 = df["pickup_longitude"]
    lon2 = df["dropoff_longitude"]
    df["latdiff"] = lat1 - lat2
    df["londiff"] = lon1 - lon2

    df["mid_lat"] = ((lat1 + lat2) * 0.5).astype("float32")
    df["mid_lon"] = ((lon1 + lon2) * 0.5).astype("float32")
    return df


def add_distances_features(df):
    lat1 = df["pickup_latitude"]
    lat2 = df["dropoff_latitude"]
    lon1 = df["pickup_longitude"]
    lon2 = df["dropoff_longitude"]

    df["manhattan"] = manhattan(lat1, lon1, lat2, lon2)
    df["distance"] = np.sqrt(np.abs(lon1 - lon2) ** 2 + np.abs(lat1 - lat2) ** 2)
    return df


def add_geo_features(df):
    lat1 = np.deg2rad(df["pickup_latitude"].astype("float64"))
    lon1 = np.deg2rad(df["pickup_longitude"].astype("float64"))
    lat2 = np.deg2rad(df["dropoff_latitude"].astype("float64"))
    lon2 = np.deg2rad(df["dropoff_longitude"].astype("float64"))

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.minimum(1.0, np.sqrt(a)))
    earth_radius_km = 6371.0
    df["haversine_km"] = (earth_radius_km * c).astype("float32")

    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    df["bearing"] = np.arctan2(y, x).astype("float32")
    return df


def filter_by_fare_distance(df):
    if ("fare_amount" not in df.columns) or ("distance" not in df.columns):
        return df

    d = df["distance"].astype("float32")
    f = df["fare_amount"].astype("float32")

    keep = np.ones(len(df), dtype=bool)
    keep &= ~((d < 0.001) & (f > 30.0))
    keep &= ~((d > 0.30) & (f < 3.0))

    keep &= np.isfinite(d.to_numpy())
    keep &= np.isfinite(f.to_numpy())

    before = len(df)
    df2 = df.loc[keep]
    print(
        f" New size after fare/distance sanity filter: {len(df2)} (dropped {before - len(df2)})"
    )
    return df2


def output_submission(raw_test, prediction, id_column, prediction_column, file_name):
    df = pd.DataFrame(
        {
            id_column: raw_test[id_column].values,
            prediction_column: prediction.reshape(-1),
        }
    )
    df.to_csv(file_name, index=False)
    print("Output complete:", file_name, "rows=", len(df))


def rmse_np(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))




## === cell 8
print("train_df clean")
train_df = clean(train_df)
print("validation_df clean")
validation_df = clean(validation_df)

print("test_df clean")
test_df = clean(test_df)



## === cell 9
train_df.describe()



## === cell 10
validation_df.describe()



## === cell 11
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
print("test_df add_coordinate_features")
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
print("train_df add_geo_features")
train_df = add_geo_features(train_df)
print("validation_df add_geo_features")
validation_df = add_geo_features(validation_df)
print("test_df add_geo_features")
test_df = add_geo_features(test_df)
print("testKaggle add_geo_features")
testKaggle = add_geo_features(testKaggle)



## === cell 15
print("Applying fare/distance sanity filter")
train_df = filter_by_fare_distance(train_df)
validation_df = filter_by_fare_distance(validation_df)
test_df = filter_by_fare_distance(test_df)



## === cell 16
train_df.describe()



## === cell 17
validation_df.describe()



## === cell 18
dropped_columns = ["key", "pickup_datetime"]

train_df = train_df.drop(dropped_columns, axis=1)
test_df = test_df.drop(dropped_columns, axis=1)
validation_df = validation_df.drop(dropped_columns, axis=1)
testKaggle_clean = testKaggle.drop(dropped_columns, axis=1)

print("Done with dropped_columns")



## === cell 19
train_labels = train_df["fare_amount"].values.astype("float32")
validation_labels = validation_df["fare_amount"].values.astype("float32")
test_labels = test_df["fare_amount"].values.astype("float32")

train_df = train_df.drop(["fare_amount"], axis=1)
validation_df = validation_df.drop(["fare_amount"], axis=1)
test_df = test_df.drop(["fare_amount"], axis=1)

print("Done with Labels")



## === cell 20
train_df.shape



## === cell 21
test_df.shape



## === cell 22
validation_df.shape



## === cell 23
feature_cols = list(train_df.columns)
validation_df = validation_df.reindex(columns=feature_cols)
test_df = test_df.reindex(columns=feature_cols)
testKaggle_clean = testKaggle_clean.reindex(columns=feature_cols)

train_df = train_df.fillna(0.0)
validation_df = validation_df.fillna(0.0)
test_df = test_df.fillna(0.0)
testKaggle_clean = testKaggle_clean.fillna(0.0)

train_X = train_df.to_numpy(dtype=np.float32)
val_X = validation_df.to_numpy(dtype=np.float32)
test_X = test_df.to_numpy(dtype=np.float32)
kaggle_X = testKaggle_clean.to_numpy(dtype=np.float32)



## === cell 24
rf = RandomForestRegressor(
    n_estimators=300,
    random_state=1,
    n_jobs=-1,
    min_samples_leaf=2,
    max_features="sqrt",
)
rf.fit(train_X, train_labels)



## === cell 25
val_pred = rf.predict(val_X)
val_rmse = rmse_np(validation_labels, val_pred)
print("Validation RMSE:", val_rmse)



## === cell 26
test_pred = rf.predict(test_X)
test_rmse = rmse_np(test_labels, test_pred)
print("Test RMSE:", test_rmse)



## === cell 27
predictionKaggle = rf.predict(kaggle_X)



## === cell 28
mse_sample = np.mean(
    (test_labels[:1000].reshape(-1) - test_pred[:1000].reshape(-1)) ** 2
)
print("Sample MSE (first 1000):", float(mse_sample))



## === cell 29
predictionKaggle_pp = np.clip(predictionKaggle.reshape(-1), 0.0, 200.0).astype(
    "float32"
)



## === cell 30
output_submission(
    testKaggle, predictionKaggle_pp, "key", "fare_amount", SUBMISSION_NAME
)

print("Example prediction:", float(test_pred.reshape(-1)[0]))
print("Submission head:")
print(pd.read_csv(SUBMISSION_NAME).head())
