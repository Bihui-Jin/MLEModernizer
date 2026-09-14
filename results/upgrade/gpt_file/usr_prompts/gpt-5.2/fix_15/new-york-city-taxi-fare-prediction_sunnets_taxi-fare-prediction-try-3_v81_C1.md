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

4.30388

# 6. Current score

5.82523

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 5.68564) has done: 'I fix the runtime errors caused by (1) Keras 3 API changes (optimizer naming/args, ModelCheckpoint file format, removed `vis_utils`) and (2) the NYC water-mask step trying to download an image (no internet; Matplotlib no longer reads URLs directly). I keep your model architecture and training loop intact, but make the smallest necessary edits so training actually runs and a valid `submissiontry_water.csv` is written. To improve RMSE toward your target (from ~15 → ~4.3) without changing the core approach, I correct two major logic bugs in the time-based features (`late_night` and `night`) that currently make those features almost always 1/0 incorrectly, which severely hurts model quality. All changes are localized and do not add new modeling approaches or reduce training.'
- What this solution (achieved 5.76353) has done: 'I fix the runtime error happening immediately on import by forcing the pure-Python protobuf implementation (this is a common incompatibility that triggers `MessageFactory.GetPrototype` failures in Kaggle images). Then I correct the two time-feature logic bugs (`late_night` and `night`) so those binary flags reflect the intended hour ranges; this is a minimal, semantics-preserving feature bugfix that should improve RMSE toward your target without changing the model or training loop. Finally, I keep all paths and outputs the same and ensure the submission CSV is always written with the required columns.'
- What this solution (achieved 5.78231) has done: 'I fix the protobuf/Keras import crash by ensuring the protobuf pure-Python fallback is applied early enough and (if needed) by avoiding the internal TF/keras protobuf path that triggers `MessageFactory.GetPrototype`. Then I keep your exact model/training loop and features, but fix a key data bug: you currently drop the `key` column from the sampled training read while later expecting `key` in the submission; this can silently misalign ID handling and also prevents easy consistency checks. Finally, I keep all paths and outputs the same and ensure we always write a valid submission CSV with columns `key,fare_amount`.'
- What this solution (achieved 5.88558) has done: 'I fix the immediate runtime crash (`MessageFactory.GetPrototype`) by ensuring TensorFlow/Keras is imported in a way that avoids the protobuf API mismatch in this Kaggle image, while keeping your tf_keras-based model and training loop unchanged. Then I address a score-relevant data logic issue: you currently remove `key` before train/test splitting and then run `clean()` on frames that don’t always have `key`; more importantly, your feature engineering uses slow row-wise `apply`, which is prone to timeouts and can silently coerce types—so I replace those with equivalent vectorized computations (same semantics) to stabilize training and generally improve RMSE. Finally, I keep all paths/output names the same and ensure the submission CSV is written with correct columns and row alignment.'
- What this solution (achieved 5.76366) has done: 'I fix the immediate runtime crash caused by an incompatible protobuf version by force-installing a protobuf release that matches TensorFlow/tf_keras expectations within the Kaggle session, before importing TensorFlow/Keras. Then I keep your model, training loop, and features the same, but correct one score-relevant data bug: `clean()` is incorrectly applied to `test_df_full` (which includes `fare_amount`) even though it is used only as a holdout; for true test inference you should *not* filter rows based on `fare_amount`, and this also causes inconsistent preprocessing behavior. Finally, I ensure the submission file is always produced with the required `key,fare_amount` columns and a `.csv` suffix.'
- What this solution (achieved 17.96886) has done: 'I fix one score-critical data-preprocessing mistake that currently makes your model worse than necessary: you are applying `clean()` to the real Kaggle test set, which drops rows (no labels) and causes a mismatched submission (and forces the model to predict on a filtered distribution). Instead, I keep `clean()` for training/holdout only, and for Kaggle test I apply a “no-row-dropping” version that only does safe `dropna` and basic type coercions while keeping all test rows/keys. I also make the feature generation functions robust to missing/invalid datetimes without changing the feature definitions. These are minimal edits that preserve your architecture/training loop and should improve RMSE toward the 4.30388 target by fixing distribution/row-alignment issues and avoiding unintended test-time filtering.'
- What this solution (achieved 8.48697) has done: 'Your current leaderboard score (17.97 RMSE) suggests the submission predictions are badly miscalibrated versus the training distribution; the biggest minimal/legitimate fix is to make train/validation/test use the same preprocessing as Kaggle test, instead of applying the very aggressive `clean()` row-filtering (which changes the training distribution and makes scaling inconsistent). I keep your model, training loop, and engineered features identical, but (1) stop using `clean()` on the sampled training data and instead use the same “no-row-dropping” cleaner everywhere, (2) ensure the datetime-derived features are created after a safe datetime parse without converting back to string until the end, and (3) keep the submission alignment exactly by key with no row drops. This should move RMSE strongly toward your target (4.30) without changing the architecture or optimization settings.'
- What this solution (achieved 5.43324) has done: 'Your current score (8.49 RMSE) is far above the target (4.30), so we should make a small, score-relevant fix that improves generalization without changing your model/training loop. The biggest minimal issue is that the “minimal clean” keeps many physically invalid coordinates (0s, outside NYC bounds, identical pickup/dropoff), which the model can’t fit well and harm test-time predictions; we apply a *training-only* row filter that removes only those coordinate-invalid rows while keeping the Kaggle test set unfiltered (so submission row alignment stays perfect). We also ensure passenger_count is numeric and clipped consistently for train/val/test, and keep feature engineering and the neural net exactly the same. This should move RMSE down materially toward the target while preserving the core approach and producing the same submission format.'
- What this solution (achieved 130.62634) has done: 'We’re currently worse than the target (RMSE 5.433 > 4.304), so we want a small, legitimate improvement without changing your model or training loop. The biggest low-risk issue is that the scaler is fit on training data that still contains label/feature outliers (since we only filtered coords), which can compress most “normal” trips and hurt regression accuracy. I add a *training-only* fare-based filter (same bounds already present in your original `clean()` logic) and fit the scaler on the cleaned training features only, while keeping Kaggle test unfiltered and identically feature-engineered for perfect submission alignment. This keeps architecture/loss/training identical and should move RMSE down toward your target.'
- What this solution (achieved 249.54365) has done: 'Your current RMSE (130.63) indicates the model is likely training on mostly-NaN/zero features due to a subtle index/label misalignment bug introduced when you `dropna()` after splitting labels/features. I make the smallest score-relevant fix: ensure we drop non-finite rows by creating a single finite-row mask and applying it to both `X` and `y` (for train/val/test) so labels always match their feature rows. I also apply the same “finite/NaN handling” to the Kaggle test features using the exact same column order as training to avoid any silent column mismatch that can explode RMSE. Core model, features, training loop, optimizer, and loss stay unchanged; we only fix data alignment and stability.'
- What this solution (achieved 10.43319) has done: 'Your current RMSE (249) is wildly worse than the target (4.30), which strongly suggests a preprocessing mismatch or a feature/label alignment problem that’s corrupting training, not a “model capacity” issue. The smallest high-impact fix is to stop dropping `pickup_datetime` from the model inputs (you currently engineer time features but then throw them away), and to prevent the `_finite_align_xy` step from silently desynchronizing labels by using an explicit joint finite mask on both `X` and `y`. Finally, because taxi fares are strictly positive and the model can output extreme values when training is unstable, we apply a conservative clipping of predictions to the same training fare range used by your outlier filter to stabilize the submission distribution (this affects only post-processing, not training).'
- What this solution (achieved 5.90376) has done: 'Your current RMSE (10.433) is far above the target (4.304), so we should make a small, high-impact correction to reduce obvious preprocessing/training mismatch without changing the model or training loop. The biggest score-damaging issue is that you train on a heavily truncated fare range via `TRAIN_MAX_FARE=200` while also using an even stricter training-time filter (<=200) and then clipping predictions to 200, which still allows extreme outputs relative to the original classic baseline (most public solutions clip around 0–50/0–100 for stability). We tighten the training-only fare filter and prediction clipping to match the intended “NYC typical ride” range already present in your original `clean()` logic (<=50), which usually reduces RMSE materially on this competition by preventing the network from chasing rare large fares. This keeps architecture/loss/optimizer/training loop identical and only adjusts an outlier-handling constant and consistent post-processing.'
- What this solution (achieved 5.82523) has done: 'To move RMSE down toward your 4.30388 target (from 5.90376) without changing the model/training loop, I make one minimal, high-impact preprocessing fix: ensure `passenger_count` is treated consistently between train/validation and Kaggle test by *not* dropping it from the model inputs (right now you accidentally keep it for train/val but drop it for Kaggle test, which harms predictions). I also reindex `testKaggle_clean` to exactly match the training feature columns without silently filling a missing `passenger_count` with zeros, which is an unrealistic value. These are localized changes that preserve your architecture, loss, optimizer, epochs, and feature engineering while improving train–test feature consistency (a common cause of inflated RMSE). The pipeline still run end-to-end and write the same submission CSV.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

import sys
import subprocess


def _ensure_protobuf_compat():
    try:
        import google.protobuf  # noqa: F401
        from google.protobuf import __version__ as pb_ver

        major = int(pb_ver.split(".")[0])
        if major >= 5:
            raise RuntimeError(f"protobuf too new: {pb_ver}")
    except Exception as e:
        print(
            "Adjusting protobuf version for TF/tf_keras compatibility due to:", repr(e)
        )
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-q",
                "--no-deps",
                "protobuf==3.20.3",
            ]
        )
        for m in list(sys.modules.keys()):
            if m.startswith("google.protobuf"):
                del sys.modules[m]


_ensure_protobuf_compat()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

try:
    import tensorflow as tf  # noqa: F401
except Exception as e:
    print("WARNING: tensorflow import failed; continuing with tf_keras only:", repr(e))

import tf_keras as keras
from tf_keras.models import Sequential
from tf_keras.layers import Dense, BatchNormalization
from tf_keras.callbacks import ModelCheckpoint
from tf_keras import optimizers
from tf_keras import regularizers
from tf_keras import backend

TRAIN_PATH = "/kaggle/input/train.csv"
TEST_PATH = "/kaggle/input/test.csv"
SUBMISSION_NAME = "submissiontry_water.csv"

BATCH_SIZE = 256
EPOCHS = 100
LEARNING_RATE = 0.001
DATASET_SIZE = 80000

TRAIN_MAX_FARE = 50.0

np.random.seed(1)
keras.utils.set_random_seed(1)




## === cell 1
def remove_datapoints_from_water(df):
    def lonlat_to_xy(longitude, latitude, dx, dy, BB):
        return (dx * (longitude - BB[0]) / (BB[1] - BB[0])).astype("int"), (
            dy - dy * (latitude - BB[2]) / (BB[3] - BB[2])
        ).astype("int")

    BB = (-74.5, -72.8, 40.5, 41.8)

    local_mask_paths = [
        "/kaggle/input/nyc_mask-74.5_-72.8_40.5_41.8.png",
        "/kaggle/working/nyc_mask-74.5_-72.8_40.5_41.8.png",
    ]
    mask_path = None
    for p in local_mask_paths:
        if os.path.exists(p):
            mask_path = p
            break

    if mask_path is None:
        print(
            "WARNING: NYC water mask image not found locally; skipping water filtering step."
        )
        return df

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
    print(" New size after NYC lang lot: %d" % len(df))

    df = df[(df["pickup_latitude"] != 0)]
    df = df[(df["pickup_latitude"] != 0)]
    df = df[(df["dropoff_longitude"] != 0)]
    df = df[(df["dropoff_latitude"] != 0)]
    print(" New size after lang lot > 0: %d" % len(df))

    df = df[((df["pickup_latitude"] - df["dropoff_latitude"]).abs() > 0.001)]
    df = df[((df["pickup_longitude"] - df["dropoff_longitude"]).abs() > 0.001)]
    print(" New size after lang - lot > 0.001: %d" % len(df))

    print(" New size after only NYC: %d" % len(df))

    if "fare_amount" in df.columns:
        df = df[(0 < df["fare_amount"]) & (df["fare_amount"] <= 50)]
        print(" New size after removing outliers: %d" % len(df))

    df = df[(df["passenger_count"] > 0)]
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
    df = remove_datapoints_from_water(df)
    print("New size: %d" % len(df))

    print(" New size: %d" % len(df))
    return df


def clean_test_no_drop(df):
    df = df.copy()
    print(" Size before minimal clean:", len(df))

    df = df.dropna(how="any", axis="rows")
    print(" Size after dropna:", len(df))

    if "pickup_datetime" in df.columns:
        df["pickup_datetime"] = df["pickup_datetime"].astype(str)

    if "passenger_count" in df.columns:
        df["passenger_count"] = pd.to_numeric(
            df["passenger_count"], errors="coerce"
        ).fillna(1)
        df["passenger_count"] = (
            df["passenger_count"].clip(lower=1, upper=8).astype("uint8")
        )

    print(" Size after minimal clean:", len(df))
    return df


def filter_train_coordinate_outliers(df):
    df = df.copy()
    old = len(df)

    for c in [
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
    ]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(
        subset=[
            "pickup_longitude",
            "pickup_latitude",
            "dropoff_longitude",
            "dropoff_latitude",
        ]
    )

    df = df[
        (df["pickup_longitude"] != 0)
        & (df["pickup_latitude"] != 0)
        & (df["dropoff_longitude"] != 0)
        & (df["dropoff_latitude"] != 0)
    ]

    MinMax = (-74.5, -72.8, 40.5, 41.8)
    df = df[
        (MinMax[0] <= df["pickup_longitude"])
        & (df["pickup_longitude"] <= MinMax[1])
        & (MinMax[0] <= df["dropoff_longitude"])
        & (df["dropoff_longitude"] <= MinMax[1])
        & (MinMax[2] <= df["pickup_latitude"])
        & (df["pickup_latitude"] <= MinMax[3])
        & (MinMax[2] <= df["dropoff_latitude"])
        & (df["dropoff_latitude"] <= MinMax[3])
    ]

    df = df[
        (df["dropoff_longitude"] != df["pickup_longitude"])
        | (df["dropoff_latitude"] != df["pickup_latitude"])
    ]

    if "passenger_count" in df.columns:
        df["passenger_count"] = pd.to_numeric(
            df["passenger_count"], errors="coerce"
        ).fillna(1)
        df = df[df["passenger_count"] > 0]
        df["passenger_count"] = (
            df["passenger_count"].clip(lower=1, upper=8).astype("uint8")
        )

    print(f" Train coord filter: {old} -> {len(df)}")
    return df


def filter_train_fare_outliers(df, max_fare=200.0):
    df = df.copy()
    if "fare_amount" not in df.columns:
        return df
    old = len(df)
    df["fare_amount"] = pd.to_numeric(df["fare_amount"], errors="coerce")
    df = df.dropna(subset=["fare_amount"])
    df = df[(df["fare_amount"] > 0) & (df["fare_amount"] <= float(max_fare))]
    print(f" Train fare filter (0, {max_fare}]: {old} -> {len(df)}")
    return df


def manhattan(pickup_lat, pickup_long, dropoff_lat, dropoff_long):
    return np.abs(dropoff_lat - pickup_lat) + np.abs(dropoff_long - pickup_long)


def add_time_features(df):
    dt = pd.to_datetime(df["pickup_datetime"], errors="coerce", utc=True)
    df["year"] = dt.dt.year.fillna(0).astype("int16")
    df["month"] = dt.dt.month.fillna(0).astype("int8")
    df["day"] = dt.dt.day.fillna(0).astype("int8")
    df["hour"] = dt.dt.hour.fillna(0).astype("int8")
    df["weekday"] = dt.dt.weekday.fillna(0).astype("int8")

    hour = df["hour"].astype("float32")
    weekday = df["weekday"].astype("float32")

    df["late_night"] = ((hour >= 0) & (hour <= 3)).astype("int8")
    df["night"] = (((hour > 20) | (hour < 6)) & (weekday < 5)).astype("int8")
    df["rush_hour"] = (((hour <= 20) & (hour >= 16)) & (weekday < 5)).astype("int8")

    df["pickup_datetime"] = df["pickup_datetime"].astype(str)
    return df


def add_coordinate_features(df):
    lat1 = df["pickup_latitude"]
    lat2 = df["dropoff_latitude"]
    lon1 = df["pickup_longitude"]
    lon2 = df["dropoff_longitude"]

    df["latdiff"] = (lat1 - lat2).abs()
    df["londiff"] = (lon1 - lon2).abs()
    return df


def add_distances_features(df):
    lat1 = df["pickup_latitude"]
    lat2 = df["dropoff_latitude"]
    lon1 = df["pickup_longitude"]
    lon2 = df["dropoff_longitude"]

    df["manhattan"] = manhattan(lat1, lon1, lat2, lon2)
    df["distance"] = np.sqrt(
        np.abs(df["pickup_longitude"] - df["dropoff_longitude"]) ** 2
        + np.abs(df["pickup_latitude"] - df["dropoff_latitude"]) ** 2
    )
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
    plt.legend(["train", "test"], loc="upper right")
    plt.show()

    if "rmse" in history.history:
        plt.figure(figsize=(20, 10))
        plt.plot(history.history["rmse"])
        plt.plot(history.history["val_rmse"])
        plt.title("Model rmse")
        plt.ylabel("rmse")
        plt.xlabel("epoch")
        plt.legend(["train", "test"], loc="upper right")
        plt.show()




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

trainKaggle = pd.read_csv(
    TRAIN_PATH,
    nrows=DATASET_SIZE,
    dtype=datatypes,
    usecols=[0, 1, 2, 3, 4, 5, 6, 7],  # key + fare_amount + features
)
testKaggle = pd.read_csv(
    TEST_PATH, dtype={k: v for k, v in datatypes.items() if k != "fare_amount"}
)



## === cell 3
train_df_full, test_df_full = train_test_split(
    trainKaggle, test_size=0.50, random_state=1
)
test_df_full = test_df_full[:10000].copy()



## === cell 4
print("testKaggle Size %d" % len(testKaggle))
print("train_df Size %d" % len(train_df_full))
print("test_df Size %d" % len(test_df_full))



## === cell 5
train_df_full.describe()



## === cell 6
test_df_full.describe()



## === cell 7
print("train_df minimal clean (match Kaggle test preprocessing)")
train_df_full = clean_test_no_drop(train_df_full)
print("test_df minimal clean (match Kaggle test preprocessing)")
test_df_full = clean_test_no_drop(test_df_full)

print("testKaggle minimal clean (keep all rows/keys for valid submission alignment)")
testKaggle = clean_test_no_drop(testKaggle)

train_df_full = filter_train_coordinate_outliers(train_df_full)
test_df_full = filter_train_coordinate_outliers(test_df_full)

train_df_full = filter_train_fare_outliers(train_df_full, max_fare=TRAIN_MAX_FARE)
test_df_full = filter_train_fare_outliers(test_df_full, max_fare=TRAIN_MAX_FARE)



## === cell 8
train_df_full.head()



## === cell 9
print("train_df add_time_features")
train_df_full = add_time_features(train_df_full)
print("test_df add_time_features")
test_df_full = add_time_features(test_df_full)
print("testKaggle add_time_features")
testKaggle = add_time_features(testKaggle)



## === cell 10
train_df_full.describe()



## === cell 11
print("train_df add_coordinate_features")
train_df_full = add_coordinate_features(train_df_full)
print("test_df add_coordinate_features")
test_df_full = add_coordinate_features(test_df_full)
print("testKaggle add_coordinate_features")
testKaggle = add_coordinate_features(testKaggle)



## === cell 12
train_df_full.describe()



## === cell 13
print("train_df add_distances_features")
train_df_full = add_distances_features(train_df_full)
print("test_df add_distances_features")
test_df_full = add_distances_features(test_df_full)
print("testKaggle add_distances_features")
testKaggle = add_distances_features(testKaggle)
print("Done with Adding features")



## === cell 14
train_df_full.describe()



## === cell 15
_ = train_df_full.iloc[:2000].plot.scatter("latdiff", "londiff")
_ = train_df_full.iloc[:2000].plot.scatter("fare_amount", "passenger_count")



## === cell 16
dropped_columns = [
    "pickup_datetime",  # drop raw string; keep engineered year/month/day/hour/weekday flags
    "pickup_latitude",
    "pickup_longitude",
    "dropoff_longitude",
    "dropoff_latitude",
]

train_df = train_df_full.drop(dropped_columns, axis=1)
test_df = test_df_full.drop(dropped_columns, axis=1)

testKaggle_clean = testKaggle.drop(dropped_columns + ["key"], axis=1)

print("Done with dropped_columns")



## === cell 17
train_df.shape



## === cell 18
train_df.describe()



## === cell 19
test_df.describe()



## === cell 20
train_df, validation_df = train_test_split(train_df, test_size=0.10, random_state=1)



## === cell 21
train_df_main = train_df
validation_df_main = validation_df



## === cell 22
validation_df.describe()



## === cell 23
if "key" in train_df.columns:
    train_df = train_df.drop(columns=["key"])
if "key" in validation_df.columns:
    validation_df = validation_df.drop(columns=["key"])
if "key" in test_df.columns:
    test_df = test_df.drop(columns=["key"])

train_labels = train_df["fare_amount"].values
validation_labels = validation_df["fare_amount"].values
test_labels = test_df["fare_amount"].values

train_df = train_df.drop(["fare_amount"], axis=1)
validation_df = validation_df.drop(["fare_amount"], axis=1)
test_df = test_df.drop(["fare_amount"], axis=1)

print("Done with Labels")




## === cell 24
def _finite_align_xy(X_df, y_arr, name=""):
    X_df = X_df.replace([np.inf, -np.inf], np.nan)
    y_arr = np.asarray(y_arr)

    x_mask = np.isfinite(X_df.to_numpy(dtype="float64")).all(axis=1)
    y_mask = np.isfinite(y_arr)

    mask = x_mask & y_mask
    X_out = X_df.loc[mask].copy()
    y_out = y_arr[mask]
    print(f" Finite-align {name}: {len(X_df)} -> {len(X_out)}")
    return X_out, y_out


train_df, train_labels = _finite_align_xy(train_df, train_labels, name="train")
validation_df, validation_labels = _finite_align_xy(
    validation_df, validation_labels, name="validation"
)
test_df, test_labels = _finite_align_xy(test_df, test_labels, name="test")

testKaggle_clean = testKaggle_clean.replace([np.inf, -np.inf], np.nan)

if (
    "passenger_count" in train_df.columns
    and "passenger_count" in testKaggle_clean.columns
):
    testKaggle_clean["passenger_count"] = pd.to_numeric(
        testKaggle_clean["passenger_count"], errors="coerce"
    ).fillna(1.0)
    testKaggle_clean["passenger_count"] = testKaggle_clean["passenger_count"].clip(1, 8)

testKaggle_clean = testKaggle_clean.fillna(0.0)
testKaggle_clean = testKaggle_clean.reindex(
    columns=list(train_df.columns), fill_value=0.0
)

scaler = preprocessing.MinMaxScaler()
train_df_scaled = scaler.fit_transform(train_df)
validation_df_scaled = scaler.transform(validation_df)
test_scaled = scaler.transform(test_df)
testKaggle_scaled = scaler.transform(testKaggle_clean)




## === cell 25
def rmse(y_true, y_pred):
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))




## === cell 26
checkpoint = ModelCheckpoint(
    filepath="my_model.keras",
    verbose=1,
    save_best_only=True,
    monitor="val_loss",
    mode="min",
)

model = Sequential()
model.add(
    Dense(
        256,
        activation="linear",
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
model.compile(loss="mean_squared_error", optimizer=adam, metrics=["mae", rmse, "mse"])

print("Dataset size: %s" % DATASET_SIZE)
print("Epochs: %s" % EPOCHS)
print("Learning rate: %s" % LEARNING_RATE)
print("Batch size: %s" % BATCH_SIZE)
print("Input dimension: %s" % train_df_scaled.shape[1])
print("Features used: %s" % list(train_df.columns))
model.summary()

history = model.fit(
    x=train_df_scaled,
    y=train_labels,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    verbose=1,
    callbacks=[checkpoint],
    validation_data=(validation_df_scaled, validation_labels),
    shuffle=True,
)



## === cell 27
plot_loss_accuracy_rmse(history)



## === cell 28
score = model.evaluate(train_df_scaled, train_labels, verbose=1)
print(score)
print("train mean_squared_error:", score[0])
print("train mae:", score[1])
print("train rmse:", score[2])
print("train mse:", score[3])



## === cell 29
score = model.evaluate(validation_df_scaled, validation_labels, verbose=1)
print(score)
print("Validation mean_squared_error:", score[0])
print("Validation mae:", score[1])
print("Validation rmse:", score[2])
print("Validation mse:", score[3])



## === cell 30
score = model.evaluate(test_scaled, test_labels, verbose=1)
print(score)
print("Test mean_squared_error:", score[0])
print("Test mae:", score[1])
print("Test rmse:", score[2])
print("Test mse:", score[3])



## === cell 31
validation_predictions = model.predict(validation_df_scaled).flatten()

plt.scatter(validation_labels, validation_predictions, s=5)
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.axis("equal")
plt.xlim(plt.xlim())
plt.ylim(plt.ylim())
_ = plt.plot(
    [validation_predictions.min(), validation_predictions.max()],
    [validation_predictions.min(), validation_predictions.max()],
    "k--",
    lw=2,
)



## === cell 32
test_predictions = model.predict(test_scaled).flatten()

plt.scatter(test_labels, test_predictions, s=5)
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.axis("equal")
plt.xlim(plt.xlim())
plt.ylim(plt.ylim())
_ = plt.plot(
    [test_predictions.min(), test_predictions.max()],
    [test_predictions.min(), test_predictions.max()],
    "k--",
    lw=2,
)



## === cell 33
error = test_predictions - test_labels
errorGreaterZero = error[(np.logical_or(error <= -1, error >= 1))]
print(len(error))
print(len(errorGreaterZero))

plt.hist(errorGreaterZero, bins=100)
plt.xlabel("Prediction Error (|err|>=1)")
_ = plt.ylabel("Count")



## === cell 34
predictionKaggle = model.predict(testKaggle_scaled, batch_size=128, verbose=1)

predictionKaggle = np.clip(predictionKaggle, 0.0, TRAIN_MAX_FARE)

if not SUBMISSION_NAME.lower().endswith(".csv"):
    SUBMISSION_NAME = SUBMISSION_NAME + ".csv"

output_submission(testKaggle, predictionKaggle, "key", "fare_amount", SUBMISSION_NAME)

print("Saved submission file:", os.path.abspath(SUBMISSION_NAME))
print(pd.read_csv(SUBMISSION_NAME).head())
print("Submission rows:", len(pd.read_csv(SUBMISSION_NAME)))
print("Expected test rows:", len(testKaggle))
