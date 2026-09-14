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
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
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

3.97644

# 6. Current score

35.61661

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 5.65568) has done: 'I fix the Pandas datetime feature extraction by replacing the removed `weekday_name` accessor with the supported `dt.day_name()` and by ensuring `pickup_datetime` is parsed consistently. Then I prevent datetime columns from leaking into the model matrices (which caused the RandomForest and Keras dtype errors) by selecting only the engineered numeric/categorical feature columns and forcing the final design matrices to `float32`. Finally, I fix the Keras import/runtime issue in this environment by using `tf_keras` (already installed) so the neural net trains and produces `submission_nn.csv` with the required `key,fare_amount` columns.'
- What this solution (achieved 177.63536) has done: 'I fix the runtime crash in the neural-network part by avoiding the `tf_keras` import path that triggers the protobuf `MessageFactory.GetPrototype` error in this Kaggle environment, and instead use `tensorflow.keras` (which is compatible with TF’s bundled protobuf). I keep the same model architecture, loss, optimizer, and training call, only changing the import to unblock execution. To move the RMSE down toward your target with minimal logic change, I also add a small, standard training-data filter to remove extreme/out-of-NYC coordinate outliers (a common source of large errors in this competition) while keeping your existing feature engineering and model choices intact. The script still write valid `submission_rf.csv` and `submission_nn.csv` with the required `key,fare_amount` columns.'
- What this solution (achieved 79.80632) has done: 'I fix the TensorFlow/Keras import crash caused by an incompatible protobuf runtime (the `MessageFactory.GetPrototype` error) by switching the neural-network section to use the installed `tf_keras` package instead of `tensorflow.keras`, keeping the exact same model architecture, loss, optimizer, and fit/predict flow. I also make the train/test read paths robust to Kaggle’s dataset directory structure without changing filenames, so the notebook runs end-to-end regardless of whether files live in `../input/` or `../input/new-york-city-taxi-fare-prediction/`. Finally, I ensure the NN training data contains no NaNs after datetime parsing (dropping only rows with invalid `pickup_datetime`), which prevents silent NaNs from poisoning training and should move RMSE down toward your target without altering core logic. The script still write valid `submission_rf.csv` and `submission_nn.csv` with columns `key,fare_amount`.'
- What this solution (achieved 27.78105) has done: 'I fix the crash in the neural-network section by avoiding the `tf_keras` import path that triggers the protobuf `MessageFactory.GetPrototype` error in this environment, and instead import Keras from TensorFlow (`tensorflow.keras`), keeping the same model architecture, loss, optimizer, and fit/predict flow. I also ensure `pickup_datetime` is parsed with `utc=True` so time-feature extraction is consistent and avoids timezone-related NaNs. Finally, I keep the existing feature engineering and filtering intact and ensure the submission files are written with the required `key,fare_amount` columns and `.csv` suffix.'
- What this solution (achieved 5.21112) has done: 'I fix the TensorFlow/Keras import crash (`MessageFactory.GetPrototype`) by switching only the neural-network imports to the already-installed `tf_keras`, keeping the exact same NN architecture, loss, optimizer, and fit/predict flow. I also make the `test.csv` read consistent (apply the same dtypes and only needed columns) so feature engineering and `get_dummies` are aligned and stable between train/test. Finally, I preserve your existing feature engineering and filtering but ensure no datetime-parsing NaNs in the concatenated data silently poison the model matrices, so both `submission_rf.csv` and `submission_nn.csv` are produced correctly.'
- What this solution (achieved 5.21112) has done: 'I fix the runtime crash in the neural-network section caused by importing `tf_keras` (protobuf `MessageFactory.GetPrototype` incompatibility in this environment) by switching only those imports to `tensorflow.keras`, keeping the exact same model architecture, loss, optimizer, and fit/predict flow. I also ensure datetime-derived categorical features don’t create missing-category NaNs by explicitly filling invalid/NaT-derived strings before `get_dummies`, which is a score-improving stability fix without changing the modeling approach. Finally, I keep the output format identical but guarantee both submission files are written successfully as `.csv` with `key,fare_amount`.'
- What this solution (achieved 19.01724) has done: 'I fix the runtime crash in the neural-network section by switching the Keras imports from `tensorflow.keras` (which triggers the protobuf `MessageFactory.GetPrototype` error in this environment) to the already-installed `tf_keras`, keeping the exact same model architecture, loss, optimizer, and fit/predict flow. I also ensure the NN training call uses an explicit, stable default `epochs=1` (Keras 3/tf_keras requires clarity and avoids accidental longer training), which is consistent with your current intent and keeps runtime safe. No feature engineering, filtering, or model logic be changed otherwise, and both `submission_rf.csv` and `submission_nn.csv` be written with the required `key,fare_amount` columns.'
- What this solution (achieved 5.22649) has done: 'I fix the runtime crash in the neural-network section by avoiding the `tf_keras` import path that triggers the protobuf `MessageFactory.GetPrototype` error, and instead import Keras from TensorFlow (`tensorflow.keras`), keeping the exact same model architecture, loss, optimizer, and training call. I also add a small, standard fare-amount upper-bound filter to remove extreme-label outliers that heavily inflate RMSE; this is a minimal data-cleaning change that preserves the modeling approach but should move your score down toward the target. Finally, I ensure the submission files are always written as valid `.csv` with the required `key,fare_amount` columns and aligned row order.'
- What this solution (achieved 10.02912) has done: 'I fix the runtime crash in the neural-network section by switching the Keras imports from `tensorflow.keras` (which is triggering the protobuf `MessageFactory.GetPrototype` error in this environment) to the already-installed `tf_keras`, without changing the model architecture, loss, optimizer, or training loop. I also make the taxi “key” read as a string to prevent any accidental dtype/format changes when writing the submission (submission alignment/format stability, score-neutral). Everything else (feature engineering, filtering, RandomForest setup, and the 1-epoch NN training) remain the same so any score change comes only from getting the NN to run correctly. The script still write valid `submission_rf.csv` and `submission_nn.csv` with `key,fare_amount` columns.'
- What this solution (achieved 112.77074) has done: 'I fix the runtime crash in the neural-network section caused by importing `tf_keras` in this environment (protobuf `MessageFactory.GetPrototype` error) by switching only the NN imports to `tensorflow.keras`, keeping the exact same model architecture, loss, optimizer, and training call. I also make the input file discovery work with your actual Kaggle directory layout (`/kaggle/input/...`) so the script runs end-to-end here without manual path edits. Finally, I ensure `test_id` stays aligned after feature engineering by explicitly copying it from `test["key"]` after reading and writing submissions with the required `key,fare_amount` columns.'
- What this solution (achieved 5.17907) has done: 'I fix the runtime crash in the neural-network section caused by importing `tensorflow.keras` in this environment (protobuf `MessageFactory.GetPrototype` issue) by switching only those imports to the already-installed `tf_keras`, keeping the same model architecture, loss, optimizer, and training/inference flow. I also make the RandomForest section stable by setting a reasonable default `n_estimators` (the current default can be too weak and is likely a big contributor to the very poor RMSE), without changing the overall modeling approach. Finally, I ensure submission writing always uses the untouched `test_id` ordering and produces valid `submission_rf.csv` and `submission_nn.csv` files with `key,fare_amount`.'
- What this solution (achieved 48.48846) has done: 'The timeout is dominated by fitting a 200-tree `RandomForestRegressor` on ~1M rows with a wide one-hot encoded feature matrix, plus avoidable extra copies/materializations during feature engineering and NaN checks. I keep the exact same feature logic and model hyperparameters, but (1) remove redundant DataFrame copies, (2) make feature engineering operate in-place on a single concatenated frame, (3) avoid creating large intermediate NumPy arrays multiple times, and (4) use a faster, correctness-preserving backend for scikit-learn (Intel Extension) when available to accelerate the RandomForest fit/predict. These changes are equivalent in results (up to negligible floating-point differences) and target only constant-factor runtime/memory reductions so the script completes within 600 seconds.'
- What this solution (achieved 115.01629) has done: 'I fix the runtime crash in the neural-network section by avoiding the `tf_keras` import path that triggers the protobuf `MessageFactory.GetPrototype` error in this Kaggle environment, and instead importing Keras via `tensorflow.keras` (leaving the exact same NN architecture, loss, optimizer, and training call). I also make the training file read include the `key` column (without using it as a feature) so we can safely drop obviously bad duplicated/invalid rows by key if present and keep indices consistent; this is a correctness/stability fix and should not worsen score. Finally, I keep the RandomForest logic unchanged and ensure both submission files are written with the required `key,fare_amount` columns and `.csv` suffix.'
- What this solution (achieved 35.61661) has done: 'I fix the TensorFlow/Keras import crash (`MessageFactory.GetPrototype`) by switching the neural network section to use the already-installed `tf_keras` package, keeping the exact same Sequential/Dense architecture, loss, optimizer, and training call. This unblocks the notebook so it runs end-to-end and always writes valid `submission_rf.csv` and `submission_nn.csv` files. I also add a small safety clamp to ensure predictions are finite and non-negative (RMSE-safe, minimal post-processing) so extreme/NaN values can’t blow up your score. No other feature engineering, filtering, or model hyperparameters are changed.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing
import math
import os

os.environ.setdefault("PYTHONHASHSEED", "0")
np.random.seed(42)

INPUT_ROOTS = ["../input", "/kaggle/input", "/kaggle/data"]

CANDIDATE_DIRS = []
for root in INPUT_ROOTS:
    CANDIDATE_DIRS.append(root)
    CANDIDATE_DIRS.append(os.path.join(root, "new-york-city-taxi-fare-prediction"))


def _find_input_file(fname: str) -> str:
    for d in CANDIDATE_DIRS:
        p = os.path.join(d, fname)
        if os.path.exists(p):
            return p
    return os.path.join(INPUT_ROOTS[0], fname)


for r in INPUT_ROOTS:
    if os.path.exists(r):
        print(f"Listing {r}:", os.listdir(r))

train_path = _find_input_file("train.csv")
test_path = _find_input_file("test.csv")
sample_path = _find_input_file("sample_submission.csv")

print("Using paths:")
print(" train:", train_path)
print(" test :", test_path)
print(" samp :", sample_path)



## === cell 1
types = {
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "uint8",
}

cols = [
    "key",
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]

test_cols = [
    "key",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]



## === cell 2
train = pd.read_csv(
    train_path,
    nrows=1000000,
    usecols=cols,
    dtype={**types, **{"key": "string"}},
)
test = pd.read_csv(
    test_path,
    usecols=test_cols,
    dtype={
        **{k: v for k, v in types.items() if k != "fare_amount"},
        **{"key": "string"},
    },
)
samp = pd.read_csv(sample_path, dtype={"key": "string"})

test_id = test["key"].copy()



## === cell 3
train.dropna(how="any", axis="rows", inplace=True)

if "key" in train.columns:
    before = len(train)
    train = train.drop_duplicates(subset=["key"], keep="first")
    if len(train) != before:
        print(f"Dropped {before - len(train)} duplicated training rows by key.")

mask = (
    (train["fare_amount"] > 0)
    & (train["fare_amount"] <= 250)
    & (train["passenger_count"] <= 6)
    & (train["pickup_latitude"] > -90)
    & (train["pickup_latitude"] < 90)
    & (train["dropoff_latitude"] > -90)
    & (train["dropoff_latitude"] < 90)
    & (train["pickup_longitude"] > -180)
    & (train["pickup_longitude"] < 180)
    & (train["dropoff_longitude"] > -180)
    & (train["dropoff_longitude"] < 180)
)
train = train.loc[mask]



## === cell 4
nyc_geo = (
    train["pickup_longitude"].between(-74.5, -72.8)
    & train["dropoff_longitude"].between(-74.5, -72.8)
    & train["pickup_latitude"].between(40.0, 41.8)
    & train["dropoff_latitude"].between(40.0, 41.8)
)
train = train.loc[nyc_geo].copy()



## === cell 5
all_data = pd.concat((train, test), ignore_index=True)

y = train["fare_amount"].to_numpy(dtype=np.float32, copy=True)
n_train = len(train)
n_test = len(test)

all_data.drop(["fare_amount", "key"], axis=1, inplace=True)




## === cell 6
def week_num(day):
    """
    given the day of the month, return the week number of the month
    """
    if day <= 7:
        return "first"
    if (day > 7) and (day <= 14):
        return "second"
    if (day > 14) and (day <= 21):
        return "third"
    if (day > 21) and (day <= 28):
        return "fourth"
    return "fifth"




## === cell 7
def add_time_features(data):
    data["pickup_datetime"] = pd.to_datetime(
        data["pickup_datetime"], errors="coerce", utc=True
    )

    dt = data["pickup_datetime"].dt
    data["hour"] = dt.hour
    data["day_of_week"] = dt.day_name()
    data["day_of_month"] = dt.day
    data["week_of_month"] = data["day_of_month"].map(week_num)
    data["month"] = dt.month
    data["year"] = dt.year

    data["day_of_week"] = data["day_of_week"].fillna("Unknown").astype(str)
    data["week_of_month"] = data["week_of_month"].fillna("Unknown").astype(str)

    data["hour"] = data["hour"].astype("Int64").astype(str).fillna("Unknown")
    data["month"] = data["month"].astype("Int64").astype(str).fillna("Unknown")
    data["year"] = data["year"].astype("Int64").astype(str).fillna("Unknown")

    data.drop(["day_of_month"], axis=1, inplace=True)
    return data




## === cell 8
def add_geo_features(data):
    plon = data["pickup_longitude"].to_numpy(dtype=np.float32, copy=False)
    plat = data["pickup_latitude"].to_numpy(dtype=np.float32, copy=False)
    dlon = data["dropoff_longitude"].to_numpy(dtype=np.float32, copy=False)
    dlat = data["dropoff_latitude"].to_numpy(dtype=np.float32, copy=False)

    abs_diff_long = np.abs(dlon - plon)
    abs_diff_lat = np.abs(dlat - plat)

    data["abs_diff_longitude"] = abs_diff_long
    data["abs_diff_latitude"] = abs_diff_lat
    data["manhattan_distance"] = abs_diff_long + abs_diff_lat

    squared_long = abs_diff_long * abs_diff_long
    squared_lat = abs_diff_lat * abs_diff_lat
    data["squared_long"] = squared_long
    data["squared_lat"] = squared_lat
    data["euclid_distance"] = np.sqrt(squared_long + squared_lat)
    return data




## === cell 9
all_data = add_time_features(all_data)



## === cell 10
all_data = add_geo_features(all_data)



## === cell 11
features = [
    "passenger_count",
    "hour",
    "day_of_week",
    "week_of_month",
    "month",
    "year",
    "abs_diff_longitude",
    "abs_diff_latitude",
    "manhattan_distance",
    "euclid_distance",
]

all_data = all_data[features]
all_data = pd.get_dummies(all_data)
all_data = all_data.astype(np.float32, copy=False)



## === cell 12
x = all_data.iloc[:n_train]
x_test = all_data.iloc[n_train:]

X = np.ascontiguousarray(x.to_numpy(dtype=np.float32, copy=False))
X_test = np.ascontiguousarray(x_test.to_numpy(dtype=np.float32, copy=False))

train_nan_mask = ~np.isfinite(X).all(axis=1)
if train_nan_mask.any():
    keep = ~train_nan_mask
    X = X[keep]
    y = y[keep]
    n_train = X.shape[0]
    print(
        f"Dropped {train_nan_mask.sum()} training rows with NaN/inf engineered features."
    )

test_nan_mask = ~np.isfinite(X_test).all(axis=1)
if test_nan_mask.any():
    np.nan_to_num(X_test, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    print(
        f"Replaced NaN/inf in {test_nan_mask.sum()} test rows' engineered features with 0.0."
    )



## === cell 13
try:
    from sklearnex import patch_sklearn

    patch_sklearn()
    print("Enabled sklearnex acceleration.")
except Exception as e:
    print("sklearnex not enabled:", repr(e))

from sklearn.ensemble import RandomForestRegressor

model_1 = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)



## === cell 14
model_1.fit(X, y)

model_1_pred = model_1.predict(X_test)

model_1_pred = np.asarray(model_1_pred, dtype=np.float32)
model_1_pred = np.nan_to_num(model_1_pred, nan=0.0, posinf=0.0, neginf=0.0)
model_1_pred = np.clip(model_1_pred, 0.0, None)

sub_1 = pd.DataFrame({"key": test_id, "fare_amount": model_1_pred})
sub_1.to_csv("submission_rf.csv", index=False)



## === cell 15
import tf_keras as tfk
from tf_keras.models import Sequential
from tf_keras.layers import Dense



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 16
num_features = X.shape[1]



## === cell 17
model = Sequential()
model.add(Dense(30, input_dim=num_features, activation="relu"))
model.add(Dense(15, activation="relu"))
model.add(Dense(7, activation="relu"))
model.add(Dense(3, activation="relu"))
model.add(Dense(1, activation="linear"))



## === cell 18
model.summary()



## === cell 19
model.compile(loss="mean_squared_error", optimizer="adam")



## === cell 20
model.fit(X, y, verbose=1, epochs=1)



## === cell 21
test_pred = model.predict(X_test, verbose=0).reshape(-1)

test_pred = np.asarray(test_pred, dtype=np.float32)
test_pred = np.nan_to_num(test_pred, nan=0.0, posinf=0.0, neginf=0.0)
test_pred = np.clip(test_pred, 0.0, None)

sub = pd.DataFrame({"key": test_id, "fare_amount": test_pred})
sub.to_csv("submission_nn.csv", index=False)

print("Wrote: submission_rf.csv and submission_nn.csv")
print(sub.head())
