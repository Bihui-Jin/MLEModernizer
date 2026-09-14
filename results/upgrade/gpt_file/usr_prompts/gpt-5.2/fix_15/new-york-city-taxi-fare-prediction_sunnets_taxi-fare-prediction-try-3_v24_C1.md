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

6.24476

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

- What this solution (achieved 10.49555) has done: 'I fix the crashes by (1) switching imports to `tf_keras` to avoid the protobuf/Keras 3 `MessageFactory` error, (2) removing the internet dependency in the water-mask cleaning step (fallback to no-op when the mask cannot be loaded), and (3) updating the optimizer call to the modern API (`Adam(learning_rate=...)`). I also correct the datetime parsing to not require a timezone token, and fix the feature-engineering boolean logic bugs that were making your time flags almost always 1/0 incorrectly—this should legitimately improve RMSE toward your target without changing the core model/training approach. Finally, I ensure the submission is written as a valid `.csv` with the correct columns and aligned `key` order.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor

TRAIN_PATH = "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv"
TEST_PATH = "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv"
SUBMISSION_NAME = "submissiontry_water.csv"  # must end with .csv

DATASET_SIZE = 80000
MIN_TRAIN_ROWS_AFTER_CLEAN = 5000

np.random.seed(1)

print("Using scikit-learn backend (no TensorFlow/Keras) to avoid protobuf/Keras crash.")

if not os.path.exists(TRAIN_PATH):
    raise FileNotFoundError(f"TRAIN_PATH not found: {TRAIN_PATH}")
if not os.path.exists(TEST_PATH):
    raise FileNotFoundError(f"TEST_PATH not found: {TEST_PATH}")



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


def load_data(nrows):
    trainKaggle_local = pd.read_csv(
        TRAIN_PATH,
        nrows=nrows,
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
    )
    testKaggle_local = pd.read_csv(
        TEST_PATH, dtype={k: v for k, v in datatypes.items() if k != "fare_amount"}
    )
    return trainKaggle_local, testKaggle_local


trainKaggle, testKaggle = load_data(DATASET_SIZE)
print("Loaded:", trainKaggle.shape, testKaggle.shape)




## === cell 2
def remove_datapoints_from_water(df):
    mask_local = (
        "/kaggle/input/nyc-mask/nyc_mask-74.5_-72.8_40.5_41.8.png"  # may not exist
    )
    if not os.path.exists(mask_local):
        return df

    def lonlat_to_xy(longitude, latitude, dx, dy, BB):
        return (dx * (longitude - BB[0]) / (BB[1] - BB[0])).astype("int"), (
            dy - dy * (latitude - BB[2]) / (BB[3] - BB[2])
        ).astype("int")

    BB = (-74.5, -72.8, 40.5, 41.8)
    nyc_mask = plt.imread(mask_local)[:, :, 0] > 0.9

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


def clean(df, relax: bool = False, allow_coord_drops: bool = True):
    print(" Old size: %d" % len(df))
    df = df.dropna(how="any", axis="rows")
    print(" New size after dropna: %d" % len(df))

    tol = 1e-6
    df = df[
        (np.abs(df["dropoff_longitude"] - df["pickup_longitude"]) > tol)
        | (np.abs(df["dropoff_latitude"] - df["pickup_latitude"]) > tol)
    ]
    print(" New size after removing same long/lat (tol): %d" % len(df))

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
        df = df[(0.99 < df["fare_amount"]) & (df["fare_amount"] <= 50)]
        print(" New size after removing outliers: %d" % len(df))

    df = df[(df["passenger_count"] > 0) & (df["passenger_count"] <= 6)]
    print(" New size after removing 6=>passenger_count > 0 : %d" % len(df))

    if allow_coord_drops:
        nyc_coord = (40.7141667, -74.0063889)
        fk_coord = (40.639722, -73.778889)
        ewr_coord = (40.6925, -74.168611)
        lga_coord = (40.77725, -73.872611)
        sol_coord = (40.6892, -74.0445)

        def drop_near_coord(df_in, lat, lon, tol=1e-7):
            return df_in[
                ~(
                    (np.abs(df_in["pickup_longitude"] - lon) <= tol)
                    & (np.abs(df_in["pickup_latitude"] - lat) <= tol)
                )
            ].copy()

        def drop_near_coord_dropoff(df_in, lat, lon, tol=1e-7):
            return df_in[
                ~(
                    (np.abs(df_in["dropoff_longitude"] - lon) <= tol)
                    & (np.abs(df_in["dropoff_latitude"] - lat) <= tol)
                )
            ].copy()

        def safe_apply_drop(df_in, fn, *args, min_keep=2000, name="rule"):
            out = fn(df_in, *args)
            if len(out) < min_keep and len(df_in) >= min_keep:
                print(f" Skipping {name} (would reduce {len(df_in)} -> {len(out)}).")
                return df_in
            return out

        df = safe_apply_drop(
            df, drop_near_coord, nyc_coord[0], nyc_coord[1], name="NYC pickup exact"
        )
        df = safe_apply_drop(
            df,
            drop_near_coord_dropoff,
            nyc_coord[0],
            nyc_coord[1],
            name="NYC dropoff exact",
        )
        print(" New size after NY coord removed: %d" % len(df))

        df = safe_apply_drop(
            df, drop_near_coord, fk_coord[0], fk_coord[1], name="JFK pickup exact"
        )
        df = safe_apply_drop(
            df,
            drop_near_coord_dropoff,
            fk_coord[0],
            fk_coord[1],
            name="JFK dropoff exact",
        )
        print(" New size after jfk airport removed: %d" % len(df))

        df = safe_apply_drop(
            df, drop_near_coord, ewr_coord[0], ewr_coord[1], name="EWR pickup exact"
        )
        df = safe_apply_drop(
            df,
            drop_near_coord_dropoff,
            ewr_coord[0],
            ewr_coord[1],
            name="EWR dropoff exact",
        )
        print(" New size after ewr airport removed: %d" % len(df))

        df = safe_apply_drop(
            df, drop_near_coord, lga_coord[0], lga_coord[1], name="LGA pickup exact"
        )
        df = safe_apply_drop(
            df,
            drop_near_coord_dropoff,
            lga_coord[0],
            lga_coord[1],
            name="LGA dropoff exact",
        )
        print(" New size after lga airport removed: %d" % len(df))

        df = safe_apply_drop(
            df, drop_near_coord, sol_coord[0], sol_coord[1], name="SOL pickup exact"
        )
        df = safe_apply_drop(
            df,
            drop_near_coord_dropoff,
            sol_coord[0],
            sol_coord[1],
            name="SOL dropoff exact",
        )
        print(" New size after sol removed: %d" % len(df))
    else:
        print(" Skipping exact-coordinate drop rules (allow_coord_drops=False).")

    if relax:
        print(" Relaxed cleaning: skipping water-mask filtering.")
    else:
        print("Old size: %d" % len(df))
        before = len(df)
        df2 = remove_datapoints_from_water(df)
        if len(df2) < 2000 and before >= 2000:
            print(
                f" Skipping water-mask filtering (would reduce {before} -> {len(df2)})."
            )
        else:
            df = df2
        print("New size: %d" % len(df))

    print(" New size: %d" % len(df))
    return df


def late_night(row):
    return 1 if (row["hour"] <= 3) else 0


def night(row):
    return 1 if ((row["hour"] > 20) and (row["weekday"] < 5)) else 0


def rush_hour(row):
    return 1 if ((16 <= row["hour"] <= 20) and (row["weekday"] < 5)) else 0


def manhattan(pickup_lat, pickup_long, dropoff_lat, dropoff_long):
    return np.abs(dropoff_lat - pickup_lat) + np.abs(dropoff_long - pickup_long)


def add_time_features(df):
    dt = pd.to_datetime(
        df["pickup_datetime"], errors="coerce", infer_datetime_format=True
    )
    df["year"] = dt.dt.year.astype("float32")
    df["month"] = dt.dt.month.astype("float32")
    df["day"] = dt.dt.day.astype("float32")
    df["hour"] = dt.dt.hour.astype("float32")
    df["weekday"] = dt.dt.weekday.astype("float32")
    df["night"] = df.apply(lambda x: night(x), axis=1).astype("int8")
    df["late_night"] = df.apply(lambda x: late_night(x), axis=1).astype("int8")
    df["rush_hour"] = df.apply(lambda x: rush_hour(x), axis=1).astype("int8")
    return df


def add_coordinate_features(df):
    lat1 = df["pickup_latitude"]
    lat2 = df["dropoff_latitude"]
    lon1 = df["pickup_longitude"]
    lon2 = df["dropoff_longitude"]
    df["latdiff"] = lat1 - lat2
    df["londiff"] = lon1 - lon2
    return df


def add_distances_features(df):
    lat1 = df["pickup_latitude"]
    lat2 = df["dropoff_latitude"]
    lon1 = df["pickup_longitude"]
    lon2 = df["dropoff_longitude"]
    df["manhattan"] = manhattan(lat1, lon1, lat2, lon2)
    df["distance"] = np.sqrt(np.abs(lon1 - lon2) ** 2 + np.abs(lat1 - lat2) ** 2)
    return df


def output_submission(raw_test, prediction, id_column, prediction_column, file_name):
    df = pd.DataFrame(
        {
            id_column: raw_test[id_column].values,
            prediction_column: np.asarray(prediction).reshape(-1),
        }
    )
    if not file_name.lower().endswith(".csv"):
        file_name = file_name + ".csv"
    df.to_csv(file_name, index=False)
    print("Output complete:", file_name, "rows=", len(df))




## === cell 3
def prepare_clean_training_sample(initial_nrows, max_nrows=2000000):
    nrows = int(initial_nrows)
    while True:
        print(f"Cleaning sampled training data (nrows={nrows}) before splitting")
        raw = pd.read_csv(
            TRAIN_PATH,
            nrows=nrows,
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
        )

        cleaned = clean(raw, relax=False, allow_coord_drops=True)

        if len(cleaned) < MIN_TRAIN_ROWS_AFTER_CLEAN:
            print(
                f"Strict cleaning produced too few rows ({len(cleaned)}). "
                "Trying relaxed cleaning."
            )
            cleaned_relax = clean(raw, relax=True, allow_coord_drops=True)
            if len(cleaned_relax) > len(cleaned):
                cleaned = cleaned_relax

        if len(cleaned) < MIN_TRAIN_ROWS_AFTER_CLEAN:
            print(
                f"Still too few rows ({len(cleaned)}). "
                "Trying to disable exact-coordinate drop rules."
            )
            cleaned_relax2 = clean(raw, relax=True, allow_coord_drops=False)
            if len(cleaned_relax2) > len(cleaned):
                cleaned = cleaned_relax2

        if len(cleaned) >= MIN_TRAIN_ROWS_AFTER_CLEAN:
            return cleaned, nrows

        if nrows >= max_nrows:
            raise RuntimeError(
                f"Even after increasing nrows up to {max_nrows}, cleaned rows={len(cleaned)} "
                f"< {MIN_TRAIN_ROWS_AFTER_CLEAN}. Adjust cleaning rules or increase limits."
            )
        nrows = min(max_nrows, nrows * 2)


trainKaggle_cleaned, final_nrows = prepare_clean_training_sample(DATASET_SIZE)
print("Final cleaned sample:", trainKaggle_cleaned.shape, "from nrows=", final_nrows)

print("Splitting cleaned data into train/validation/test")
train_df, test_df = train_test_split(
    trainKaggle_cleaned, test_size=0.50, random_state=1
)
train_df, validation_df = train_test_split(train_df, test_size=0.10, random_state=1)

print("testKaggle Size %d" % len(testKaggle))
print("train_df Size %d" % len(train_df))
print("validation_df Size %d" % len(validation_df))
print("test_df Size %d" % len(test_df))



## === cell 4
print("train_df add_time_features")
train_df = add_time_features(train_df)
print("validation_df add_time_features")
validation_df = add_time_features(validation_df)
print("test_df add_time_features")
test_df = add_time_features(test_df)
print("testKaggle add_time_features")
testKaggle = add_time_features(testKaggle)


def drop_bad_time_rows(df, require_target: bool):
    time_cols = ["year", "month", "day", "hour", "weekday"]
    before = len(df)
    df2 = df.dropna(subset=time_cols)
    if require_target and "fare_amount" in df2.columns:
        df2 = df2.dropna(subset=["fare_amount"])
    after = len(df2)
    if after != before:
        print(f"Dropped {before-after} rows due to invalid pickup_datetime parsing.")
    return df2


train_df = drop_bad_time_rows(train_df, require_target=True)
validation_df = drop_bad_time_rows(validation_df, require_target=True)
test_df = drop_bad_time_rows(test_df, require_target=True)
testKaggle = drop_bad_time_rows(testKaggle, require_target=False)

train_df["pickup_datetime"] = train_df["pickup_datetime"].astype(str)
validation_df["pickup_datetime"] = validation_df["pickup_datetime"].astype(str)
test_df["pickup_datetime"] = test_df["pickup_datetime"].astype(str)
testKaggle["pickup_datetime"] = testKaggle["pickup_datetime"].astype(str)



## === cell 5
print("train_df add_coordinate_features")
train_df = add_coordinate_features(train_df)
print("validation_df add_coordinate_features")
validation_df = add_coordinate_features(validation_df)
print("test_df add_coordinate_features")
test_df = add_coordinate_features(test_df)
print("testKaggle add_coordinate_features")
testKaggle = add_coordinate_features(testKaggle)



## === cell 6
print("train_df add_distances_features")
train_df = add_distances_features(train_df)
print("validation_df add_distances_features")
validation_df = add_distances_features(validation_df)
print("test_df add_distances_features")
test_df = add_distances_features(test_df)
print("testKaggle add_distances_features")
testKaggle = add_distances_features(testKaggle)
print("Done with Adding features")



## === cell 7
dropped_columns = [
    "passenger_count",
    "pickup_datetime",
    "pickup_latitude",
    "pickup_longitude",
    "dropoff_longitude",
    "dropoff_latitude",
]

train_df = train_df.drop(dropped_columns, axis=1)
test_df = test_df.drop(dropped_columns, axis=1)
validation_df = validation_df.drop(dropped_columns, axis=1)
testKaggle_clean = testKaggle.drop(dropped_columns + ["key"], axis=1)

print("Done with dropped_columns")



## === cell 8
train_labels = train_df["fare_amount"].values
validation_labels = validation_df["fare_amount"].values
test_labels = test_df["fare_amount"].values

train_df = train_df.drop(["fare_amount"], axis=1)
validation_df = validation_df.drop(["fare_amount"], axis=1)
test_df = test_df.drop(["fare_amount"], axis=1)

print("Done with Labels")




## === cell 9
def coerce_numeric_and_align(X_df: pd.DataFrame, y=None, name: str = "data"):
    X = X_df.copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X_np = X.to_numpy(dtype=np.float64)
    valid = np.isfinite(X_np).all(axis=1)

    dropped = int((~valid).sum())
    if dropped:
        print(
            f"{name}: dropped {dropped} rows due to non-numeric/NaN features (aligned)."
        )

    X2 = X.loc[valid].reset_index(drop=True)
    if y is None:
        return X2, None
    y2 = np.asarray(y)[valid]
    return X2, y2


def build_features_and_splits(initial_nrows: int):
    cleaned, used_nrows = prepare_clean_training_sample(initial_nrows)
    train_df0, test_df0 = train_test_split(cleaned, test_size=0.50, random_state=1)
    train_df0, validation_df0 = train_test_split(
        train_df0, test_size=0.10, random_state=1
    )

    train_df0 = add_distances_features(
        add_coordinate_features(add_time_features(train_df0))
    )
    validation_df0 = add_distances_features(
        add_coordinate_features(add_time_features(validation_df0))
    )
    test_df0 = add_distances_features(
        add_coordinate_features(add_time_features(test_df0))
    )

    train_df0 = drop_bad_time_rows(train_df0, require_target=True)
    validation_df0 = drop_bad_time_rows(validation_df0, require_target=True)
    test_df0 = drop_bad_time_rows(test_df0, require_target=True)

    train_df0 = train_df0.drop(dropped_columns, axis=1)
    validation_df0 = validation_df0.drop(dropped_columns, axis=1)
    test_df0 = test_df0.drop(dropped_columns, axis=1)

    y_train = train_df0["fare_amount"].values
    y_val = validation_df0["fare_amount"].values
    y_test = test_df0["fare_amount"].values

    X_train = train_df0.drop(["fare_amount"], axis=1)
    X_val = validation_df0.drop(["fare_amount"], axis=1)
    X_test = test_df0.drop(["fare_amount"], axis=1)

    X_train, y_train = coerce_numeric_and_align(X_train, y_train, name="train")
    X_val, y_val = coerce_numeric_and_align(X_val, y_val, name="validation")
    X_test, y_test = coerce_numeric_and_align(X_test, y_test, name="test")

    return X_train, y_train, X_val, y_val, X_test, y_test, used_nrows


train_df, train_labels = coerce_numeric_and_align(train_df, train_labels, name="train")
validation_df, validation_labels = coerce_numeric_and_align(
    validation_df, validation_labels, name="validation"
)
test_df, test_labels = coerce_numeric_and_align(test_df, test_labels, name="test")

testKaggle_clean, _ = coerce_numeric_and_align(
    testKaggle_clean, None, name="kaggle_test"
)

if len(train_df) == 0 or len(validation_df) == 0:
    print(
        "WARNING: training/validation empty after feature engineering. "
        "Retrying with larger samples until non-empty (up to 2,000,000 rows)."
    )
    retry_nrows = max(DATASET_SIZE * 2, 160000)
    while True:
        (
            train_df,
            train_labels,
            validation_df,
            validation_labels,
            test_df,
            test_labels,
            used_nrows,
        ) = build_features_and_splits(retry_nrows)
        print(
            f"Retry result: nrows={used_nrows} -> train={len(train_df)}, val={len(validation_df)}, test={len(test_df)}"
        )
        if len(train_df) > 0 and len(validation_df) > 0 and len(test_df) > 0:
            break
        if retry_nrows >= 2000000:
            raise RuntimeError(
                "After retries up to 2,000,000 rows, training/validation still empty. "
                "Cleaning rules are too strict for the environment."
            )
        retry_nrows = min(2000000, retry_nrows * 2)

if len(testKaggle_clean) == 0:
    raise RuntimeError(
        "After feature engineering, Kaggle test features are empty; cannot create submission."
    )


def fill_na_with_median(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if out[c].isna().any():
            med = out[c].median()
            if not np.isfinite(med):
                med = 0.0
            out[c] = out[c].fillna(med)
    return out


train_df = fill_na_with_median(train_df)
validation_df = fill_na_with_median(validation_df)
test_df = fill_na_with_median(test_df)
testKaggle_clean = fill_na_with_median(testKaggle_clean)

scaler = preprocessing.MinMaxScaler()
train_df_scaled = scaler.fit_transform(train_df)
validation_df_scaled = scaler.transform(validation_df)
test_scaled = scaler.transform(test_df)
testKaggle_scaled = scaler.transform(testKaggle_clean)

print(
    "Scaled shapes:",
    train_df_scaled.shape,
    validation_df_scaled.shape,
    test_scaled.shape,
    testKaggle_scaled.shape,
)



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_11/2874135343.py in <cell line: 0>()
     93             break
     94         if retry_nrows >= 2000000:
---> 95             raise RuntimeError(
     96                 "After retries up to 2,000,000 rows, training/validation still empty. "
     97                 "Cleaning rules are too strict for the environment."

RuntimeError: After retries up to 2,000,000 rows, training/validation still empty. Cleaning rules are too strict for the environment.

## === cell 10
model = HistGradientBoostingRegressor(
    loss="squared_error",
    learning_rate=0.05,
    max_depth=8,
    max_iter=400,
    random_state=1,
)

model.fit(train_df_scaled, train_labels)

val_pred = model.predict(validation_df_scaled)
val_rmse = float(np.sqrt(np.mean((val_pred - validation_labels) ** 2)))
print("Validation RMSE:", val_rmse)

test_pred = model.predict(test_scaled)
test_rmse = float(np.sqrt(np.mean((test_pred - test_labels) ** 2)))
print("Holdout RMSE:", test_rmse)



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2688947985.py in <cell line: 0>()
      7 )
      8 
----> 9 model.fit(train_df_scaled, train_labels)
     10 
     11 val_pred = model.predict(validation_df_scaled)

NameError: name 'train_df_scaled' is not defined

## === cell 11
predictionKaggle = model.predict(testKaggle_scaled)
predictionKaggle = np.clip(np.asarray(predictionKaggle).reshape(-1), 0.0, 500.0)

output_submission(testKaggle, predictionKaggle, "key", "fare_amount", SUBMISSION_NAME)

print("Wrote submission file:", SUBMISSION_NAME)
print(
    "Exists?",
    os.path.exists(SUBMISSION_NAME),
    "Size:",
    os.path.getsize(SUBMISSION_NAME) if os.path.exists(SUBMISSION_NAME) else None,
)
print(pd.read_csv(SUBMISSION_NAME).head())

## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/795336576.py in <cell line: 0>()
----> 1 predictionKaggle = model.predict(testKaggle_scaled)
      2 predictionKaggle = np.clip(np.asarray(predictionKaggle).reshape(-1), 0.0, 500.0)
      3 
      4 # Ensure submission keys align with the feature rows used (testKaggle may have had bad-time rows dropped earlier).
      5 output_submission(testKaggle, predictionKaggle, "key", "fare_amount", SUBMISSION_NAME)

NameError: name 'testKaggle_scaled' is not defined
