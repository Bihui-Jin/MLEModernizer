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
seaborn==0.12.2
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

4.29156

# 6. Current score

12.09678

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 558.61119) has done: 'I fix the environment/runtime errors by removing the broken TensorFlow Estimator/`tf.contrib` usage (not available in TF 2.18 here) and replacing it with an equivalent Keras model trained on the same engineered/tabular features, so the pipeline runs end-to-end. I also fix a logic bug where you computed `add_feats(test)` but then overwrote it by dropping columns from the original `test` (losing engineered features), and I vectorize the distance feature computation so it finishes quickly. Finally, I ensure the script always writes a valid `submission_file.csv` with the required `key,fare_amount` columns.'
- What this solution (achieved 10.25103) has done: 'We fix the immediate runtime crash before any training happens: TensorFlow 2.18 is currently failing due to an incompatible `protobuf` (the `MessageFactory.GetPrototype` AttributeError). The minimal, Kaggle-safe fix is to pin the protobuf Python implementation to the pure-Python backend (and disable C++ descriptors) via environment variables before importing TensorFlow. Next, to move RMSE drastically toward the target (your current score indicates a severe evaluation mismatch), we keep the exact same feature set and Keras model, but train on `log1p(fare_amount)` and invert with `expm1` at inference; this is a standard calibration change for this competition that reduces the impact of outliers without changing the core architecture or training loop. Finally, we keep the submission format and path unchanged and ensure predictions are finite and non-negative.'
- What this solution (achieved 1.9564537640881563e+22) has done: 'I fix the TensorFlow import crash caused by the protobuf API mismatch by forcing the pure-Python protobuf implementation and disabling C++ descriptors before importing TensorFlow, plus adding a safe fallback path if TF still fails. Then I keep your exact feature engineering and Keras model/training loop, but ensure the train/test files are read from the correct Kaggle `../input/new-york-city-taxi-fare-prediction/` directory (with a fallback to `../input/`) so the pipeline runs reliably. Finally, I keep the log1p/expm1 target transform (a calibration change that should reduce RMSE toward your target) and ensure a valid `submission_file.csv` with `key,fare_amount` is always written.'
- What this solution (achieved 2.626143387451563e+35) has done: 'We fix the TensorFlow import crash (`MessageFactory.GetPrototype`) by setting the additional Kaggle-safe env flags that force the pure-Python protobuf runtime and prevent the C++ implementation from being used, and we do it before any TensorFlow import. Next, we make the submission robust even if TensorFlow still fails by using the provided `sample_submission.csv` to guarantee correct row count/order, and by aligning predictions to that key order. Finally, we eliminate the extreme-score failure mode by clipping/cleaning features and predictions to avoid NaNs/Infs propagating through scaling or inference, without changing the model architecture, features, or training loop.'
- What this solution (achieved 8.73857949869891e+31) has done: 'I fix the TensorFlow import crash that’s preventing any real model from training (your astronomically bad RMSE strongly suggests the TF fallback path is being used or predictions are invalid). The minimal robust fix is to set protobuf/TensorFlow environment flags correctly and early, and if TF still can’t import, automatically fall back to a non-TF sklearn model using the exact same engineered features (this keeps the overall approach tabular+ML and produces sane predictions). I also ensure the submission aligns exactly to `sample_submission.csv` key order and that all predictions are finite and non-negative to avoid score blow-ups. These changes are directly aimed at moving RMSE down toward the target without changing your feature engineering or the Keras architecture/training when TF is available.'
- What this solution (achieved 10.25103) has done: 'You’re getting an astronomically bad RMSE because TensorFlow fails to import (protobuf API mismatch), then your fallback path trains on `log1p(fare_amount)` but fills missing predictions using the *raw* mean fare (mixing target spaces), which can create extreme, miscalibrated outputs when any key mapping mismatch happens. I (1) make TF import robust by forcing the pure-Python protobuf runtime *and* falling back cleanly without crashing the notebook, (2) fix the fallback fill value to be in the correct (raw-fare) space and guarantee strict key alignment to `sample_submission.csv`, and (3) add a minimal safety clamp for log targets/predictions to avoid inf/overflow while keeping your exact features and model/training logic unchanged. These are correctness/stability fixes plus a small calibration correction intended to move RMSE sharply down toward your target without changing the core approach.'
- What this solution (achieved 234.49134) has done: 'I fix the early TensorFlow import crash by moving all protobuf/TensorFlow environment variables to the very top of the script and adding the additional safety flag `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` + `PROTOCOL_BUFFERS_PYTHON_DISABLE_CPP_IMPLEMENTATION=1` before any TensorFlow-related import happens (including indirect imports). Then, to move RMSE down toward your target without changing your feature engineering or model architecture/training loop, I remove the log1p/expm1 target transform (it changes the training objective vs the RMSE-on-fare metric) and train/predict directly on raw `fare_amount` while keeping the same network/loss, split, scaling, and prediction clipping. Finally, I keep the submission aligned to `sample_submission.csv` key order and guarantee finite non-negative predictions so a valid `submission_file.csv` is always written.'
- What this solution (achieved 428.51677) has done: 'We need to fix the TensorFlow import crash (`MessageFactory.GetPrototype`) which is causing the sklearn fallback to run and giving a very poor RMSE. The minimal, Kaggle-safe way is to pin `protobuf` to the pure-Python runtime and avoid the incompatible implementation-version flag, plus import TensorFlow only after setting env vars. Then, without changing your core feature engineering or model architecture/training loop, we keep the same pipeline but ensure keys/predictions align exactly to `sample_submission.csv` and that we never drop test rows needed for submission (avoid NaN-related row drops for test by imputing instead). These changes are directly targeted at (1) making TF actually run, and (2) preventing key-mismatch/row-count issues that can inflate RMSE.'
- What this solution (achieved 150.00895) has done: 'I fix the TensorFlow/protobuf import crash by pinning protobuf to the pure-Python implementation and also forcing `google.protobuf` to use the Python backend before TensorFlow is imported; this directly addresses the `MessageFactory.GetPrototype` error so the Keras path runs instead of the weak sklearn fallback. I keep your exact feature engineering, filtering, scaling, and Keras architecture/training loop unchanged. I also make the train/test file path selection work in both Kaggle layouts (`/kaggle/input/...` and `../input/...`) without changing filenames, and ensure the submission is always aligned to `sample_submission.csv` keys and written as `submission_file.csv`.'
- What this solution (achieved 533.57439) has done: 'I fix the hard runtime failure in the first cell by removing the incompatible protobuf/TensorFlow forcing that triggers the `MessageFactory.GetPrototype` crash, and instead import TensorFlow normally with safe env vars only. To move RMSE substantially toward your target (your current 150 indicates a broken/weak modeling path), I keep your exact feature engineering and Keras architecture/training loop, but ensure the TF path actually runs by preventing the crash and adding a robust fallback only if TF truly cannot import. I also make one minimal, score-improving correction: use the exact same filtering logic to remove outliers from training, but never drop/lose test rows and always align predictions to `sample_submission.csv` keys. The script always write a valid `submission_file.csv` with `key,fare_amount`.'
- What this solution (achieved 2.269381187510173e+29) has done: 'I fix the TensorFlow import crash caused by an incompatible protobuf runtime by setting the required protobuf environment variables *before* importing anything that might touch TensorFlow/protobuf, ensuring the Keras training path actually runs instead of falling back. I also add a safe, minimal try/except around TF import so the notebook always completes and writes a valid `submission_file.csv`. To move RMSE sharply toward your target (533 is far off), I keep the exact same features and Keras architecture/training loop, but switch to a log1p target transform with expm1 inversion at inference (a minimal calibration change commonly used for this competition). Finally, I preserve strict key alignment to `sample_submission.csv` and ensure predictions are finite and non-negative.'
- What this solution (achieved 2.175095411435537e+36) has done: 'We need to stop the TensorFlow import crash caused by an incompatible protobuf runtime; right now the notebook errors in the first cell so no model trains and you end up with invalid/astronomical predictions. The minimal fix is to force protobuf to use the pure-Python backend *and* pre-import `google.protobuf` before importing TensorFlow, which avoids the `MessageFactory.GetPrototype` AttributeError in this environment. Then the rest of your pipeline (same features, same Keras architecture, same log1p/expm1 calibration) can run unchanged and should bring RMSE dramatically down toward the target band. I also add a small safety fallback so that even if TF still fails, the sklearn path writes a valid submission aligned to `sample_submission.csv`.'
- What this solution (achieved 2.8829300743815713e+32) has done: 'I fix the crash happening before any training by removing the protobuf/TensorFlow forcing that triggers the `MessageFactory.GetPrototype` AttributeError in this environment, and instead import TensorFlow normally (with only safe log-level env var). I also add a minimal, robust fallback so if TensorFlow still can’t import for any reason, the script still trains a simple sklearn model and writes a valid `submission_file.csv`. To move your astronomically bad RMSE back toward the target, I ensure predictions are always finite, aligned exactly to `sample_submission.csv` key order, and never missing (no NaNs from key mismatches). Core feature engineering, filtering, scaling, and the Keras architecture/training loop are kept unchanged.'
- What this solution (achieved 1.0848252812780274e+16) has done: 'I fix the runtime crash by forcing protobuf to use the pure-Python implementation before *any* TensorFlow/protobuf import happens, which directly addresses the `MessageFactory.GetPrototype` AttributeError in this Kaggle environment. I also keep the existing Keras model, features, training loop, and log1p/expm1 target calibration unchanged, because the current astronomical RMSE strongly indicates the model never trained correctly (TF import failure leading to bad fallback behavior). Additionally, I make the submission mapping robust to duplicate `key` values (present in this dataset) by constructing a stable key→prediction mapping that won’t break `Series.map` semantics. These changes are minimal, unblock end-to-end execution, and should move the RMSE drastically down toward the target by ensuring the intended TF path runs and produces sane predictions.'
- What this solution (achieved 12.09678) has done: 'We need to stop the TensorFlow import crash (`MessageFactory.GetPrototype`) which is preventing the intended Keras model from running and forcing the weak/incorrect fallback path, leading to astronomical RMSE. The minimal robust fix is to avoid importing TensorFlow entirely (since it’s broken in this environment with the installed protobuf) and always use a stable non-TF model on the exact same engineered/scaled tabular features, producing finite predictions aligned to `sample_submission.csv`. We also fix a logic issue in the fallback: the RandomForest was incorrectly trained on the log-transformed target while using raw-fare fallback values; we train/predict directly on raw `fare_amount` to match the RMSE metric and keep calibration consistent. Finally, we keep strict key alignment and always write `submission_file.csv` with the required `key,fare_amount` columns and correct row count.'

# 9. Code solution

## === cell 0
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd

np.random.seed(1)

print("../input exists:", os.path.exists("../input"))
if os.path.exists("../input"):
    print("../input entries (head):", os.listdir("../input")[:30])
print("/kaggle/input exists:", os.path.exists("/kaggle/input"))
if os.path.exists("/kaggle/input"):
    print("/kaggle/input entries (head):", os.listdir("/kaggle/input")[:30])



## === cell 1
BASE_CANDIDATES = [
    "/kaggle/input/new-york-city-taxi-fare-prediction",
    "/kaggle/input",
    "../input/new-york-city-taxi-fare-prediction",
    "../input",
    "/kaggle/data/new-york-city-taxi-fare-prediction",
    "/kaggle/data",
]


def pick_path(fname):
    for base in BASE_CANDIDATES:
        p = os.path.join(base, fname)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Could not find {fname} in any of: {BASE_CANDIDATES}")


train_path = pick_path("train.csv")
test_path = pick_path("test.csv")
sample_path = pick_path("sample_submission.csv")

print("train_path:", train_path)
print("test_path:", test_path)
print("sample_path:", sample_path)

df = pd.read_csv(train_path, nrows=100000, parse_dates=["pickup_datetime"])
test = pd.read_csv(test_path, parse_dates=["pickup_datetime"])
sample_sub = pd.read_csv(sample_path)

print(
    "Train rows loaded:",
    len(df),
    "Test rows loaded:",
    len(test),
    "Sample rows:",
    len(sample_sub),
)




## === cell 2
def haversine_km(lat1, lon1, lat2, lon2):
    lat1 = np.radians(lat1.astype(np.float64))
    lon1 = np.radians(lon1.astype(np.float64))
    lat2 = np.radians(lat2.astype(np.float64))
    lon2 = np.radians(lon2.astype(np.float64))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371.0 * c




## === cell 3
def add_feats(dfin):
    dfout = dfin.copy()
    dfout["distance"] = haversine_km(
        dfout["pickup_latitude"].values,
        dfout["pickup_longitude"].values,
        dfout["dropoff_latitude"].values,
        dfout["dropoff_longitude"].values,
    )
    dfout["hour"] = dfout["pickup_datetime"].dt.hour.astype(np.int16)
    dfout["weekday"] = dfout["pickup_datetime"].dt.weekday.astype(np.int16)
    return dfout




## === cell 4
df = add_feats(df)
test = add_feats(test)

num_cols_train = df.select_dtypes(include=[np.number]).columns
df[num_cols_train] = df[num_cols_train].replace([np.inf, -np.inf], np.nan)
test_cols_num = test.select_dtypes(include=[np.number]).columns
test[test_cols_num] = test[test_cols_num].replace([np.inf, -np.inf], np.nan)

df = df.dropna(
    subset=[
        "fare_amount",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
        "distance",
    ]
)

test[test_cols_num] = test[test_cols_num].fillna(0.0)

print("After sanitization -> Train rows:", len(df), "Test rows:", len(test))



## === cell 5
dfc = df[
    ((df.pickup_longitude >= -75.0) & (df.pickup_longitude <= -72))
    & ((df.pickup_latitude >= 38) & (df.pickup_latitude <= 42))
    & ((df.dropoff_longitude >= -75.0) & (df.dropoff_longitude <= -72))
    & ((df.dropoff_latitude >= 38) & (df.dropoff_latitude <= 42))
    & (df.fare_amount > 2.5)
    & (df.passenger_count > 0)
    & (df.passenger_count < 7)
    & (df.distance > 0.2)
].copy()

print("Filtered rows:", len(dfc), "of", len(df))



## === cell 6
msk = np.random.rand(len(dfc)) < 0.8
traindf = dfc[msk].drop(["key", "pickup_datetime"], axis=1)
evaldf = dfc[~msk].drop(["key", "pickup_datetime"], axis=1)

testdf = test.drop(["key", "pickup_datetime"], axis=1)

print(
    "Train shape:",
    traindf.shape,
    "Eval shape:",
    evaldf.shape,
    "Test shape:",
    testdf.shape,
)



## === cell 7
FEATURES = [c for c in traindf.columns if c != "fare_amount"]

X_train = traindf[FEATURES].astype(np.float32).values
y_train = (
    traindf["fare_amount"].astype(np.float32).values
)  # train on raw fare to match RMSE metric

X_eval = evaldf[FEATURES].astype(np.float32).values
y_eval = evaldf["fare_amount"].astype(np.float32).values

X_test = testdf[FEATURES].astype(np.float32).values

X_train = np.where(np.isfinite(X_train), X_train, 0.0).astype(np.float32)
X_eval = np.where(np.isfinite(X_eval), X_eval, 0.0).astype(np.float32)
X_test = np.where(np.isfinite(X_test), X_test, 0.0).astype(np.float32)

mu = X_train.mean(axis=0)
sigma = X_train.std(axis=0)
sigma = np.where(np.isfinite(sigma) & (sigma != 0), sigma, 1.0).astype(np.float32)

X_train_s = (X_train - mu) / sigma
X_eval_s = (X_eval - mu) / sigma
X_test_s = (X_test - mu) / sigma

y_train = np.where(np.isfinite(y_train), y_train, 0.0).astype(np.float32)
y_eval = np.where(np.isfinite(y_eval), y_eval, 0.0).astype(np.float32)

print("Num features:", X_train_s.shape[1])



## === cell 8
submission_path = "submission_file.csv"

sample_keys = sample_sub["key"].astype(str).values
test_keys = test["key"].astype(str).values


def build_submission_from_pred(pred_raw_fare, test_keys_arr, sample_df, fallback_raw):
    pred_raw_fare = np.asarray(pred_raw_fare).reshape(-1).astype(np.float32)
    pred_raw_fare = np.where(
        np.isfinite(pred_raw_fare), pred_raw_fare, fallback_raw
    ).astype(np.float32)
    pred_raw_fare = np.clip(pred_raw_fare, 0.0, None).astype(np.float32)

    key_to_pred = {}
    for k, v in zip(test_keys_arr, pred_raw_fare):
        ks = str(k)
        if ks not in key_to_pred:
            key_to_pred[ks] = float(v)

    submission = sample_df.copy()
    submission["key"] = submission["key"].astype(str)
    submission["fare_amount"] = submission["key"].map(key_to_pred)

    submission["fare_amount"] = (
        submission["fare_amount"].fillna(fallback_raw).astype(np.float32)
    )
    submission["fare_amount"] = np.where(
        np.isfinite(submission["fare_amount"].values),
        submission["fare_amount"].values,
        fallback_raw,
    ).astype(np.float32)
    submission["fare_amount"] = np.clip(
        submission["fare_amount"].values, 0.0, None
    ).astype(np.float32)
    return submission


fallback_raw = float(np.mean(y_train)) if len(y_train) else 11.35

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(
    n_estimators=300,
    random_state=1,
    n_jobs=-1,
    min_samples_leaf=1,
)

rf.fit(X_train_s, y_train)

pred = rf.predict(X_test_s).reshape(-1).astype(np.float32)
pred = np.where(np.isfinite(pred), pred, fallback_raw).astype(np.float32)
pred = np.clip(pred, 0.0, None).astype(np.float32)

submission = build_submission_from_pred(pred, test_keys, sample_sub, fallback_raw)
submission.to_csv(submission_path, index=False)

print("Wrote:", submission_path, "shape:", submission.shape)
print(submission.head())
print("Submission columns:", list(submission.columns))
print(
    "Unique keys in submission:", submission["key"].nunique(), "rows:", len(submission)
)
