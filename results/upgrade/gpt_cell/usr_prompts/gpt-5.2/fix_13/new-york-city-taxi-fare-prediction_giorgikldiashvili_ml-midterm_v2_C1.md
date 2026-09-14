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

3.12

# 3. Installed packages

geopandas==0.14.4
geopy==2.4.1
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
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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
tf_keras==2.18.0
xgboost==2.0.3

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

3.61245

# 6. Current score

5.17396

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1086.87849) has done: 'The crash in cell 17 is triggered by an incompatibility between TensorFlow 2.18 and the installed `protobuf==6.33.0`, which leads to the `MessageFactory.GetPrototype` AttributeError during TensorFlow/Keras initialization. The minimal fix is to set the protobuf implementation to the pure-Python backend *before* importing TensorFlow, which avoids the missing API on some protobuf builds. This keeps the exact same model/training code and outputs; it only changes the protobuf runtime backend to make imports deterministic and non-crashing. No other cells need to change, and `model`, `history`, and data variables remain the same for cell 18.'
- What this solution (achieved 1086.87849) has done: 'Diagnosis: The crash happens in cell 17 when importing/initializing TensorFlow/Keras due to an incompatibility between TensorFlow 2.18 and the installed `protobuf==6.33.0`, which triggers `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'`. The current attempt to force the pure-Python protobuf implementation is applied too late (after protobuf/TensorFlow may already be loaded) and cannot fix the underlying version mismatch. The minimal deterministic fix is to pin protobuf to a TensorFlow-compatible major version (protobuf 4.x) inside the notebook before importing TensorFlow.

Patch summary: In cell 17 only, install a compatible protobuf version (`protobuf<5`) at runtime, then import TensorFlow/Keras as originally intended. Keep the model architecture, compile, data, and training logic unchanged; only adjust environment/package setup to prevent the import-time crash.

Updated cells: Below is the complete updated cell 17.

Compatibility notes for cell k+1: The variable `model`, and the training/validation variables (`X_valid`, `y_valid`, etc.) are still created exactly as before, so cell 18 can run unchanged.

Assumptions: The environment allows `pip` installs during runtime (standard in Kaggle-style notebook environments), and downgrading protobuf does not conflict with other already-imported libraries in earlier cells.'
- What this solution (achieved 5.68937) has done: 'Your score is extremely worse than the target (RMSE 1086 vs 3.61), which strongly suggests the submission has a formatting/data issue rather than model quality. The main bug is that the neural-net predictions are written as a 2D array (shape `(n,1)`) into the DataFrame, producing an invalid/garbled `fare_amount` column in the CSV that Kaggle can’t interpret correctly (often leading to astronomically bad scores). I keep your exact training logic/model the same, but force predictions to a 1D float vector and clamp negatives to 0 (consistent with your own training filter `fare_amount > 0`), then write a clean submission CSV. I also ensure the submission uses `key` from `sample_submission.csv` order to avoid any potential row alignment issues.'
- What this solution (achieved 5.43905) has done: 'Your current RMSE (5.689) is worse than the target (3.612), so we should improve generalization with the smallest possible change while keeping your exact feature set and model/loop intact. The biggest issue is that the neural net is trained on `nn_df` (unfiltered, includes extreme fares/outliers and bad coordinates), while your better-performing filters are applied only to `train_df`; this mismatch likely inflates error. I keep the same NN architecture/compile/epochs/batch_size, but train the NN on the same cleaned/filtered dataframe as the other models, and apply the same bounding-box + distance-range filters to `test_df` predictions with a safe fallback (median) for any rows filtered out so the submission stays complete. This should move RMSE down toward the target without changing the core modeling approach.'
- What this solution (achieved 1086.87849) has done: 'Your current RMSE (5.43905) is worse than the target (3.61245), so we should improve generalization with the smallest safe changes that keep your exact feature set and model architecture/training loop intact. The biggest likely issue is that the neural net is trained on raw, unscaled numeric features with very different magnitudes (lat/long ~ 40/-73, distance ~ 0-25, time parts), which makes Adam training for only 2 epochs underfit and behave poorly; standardizing inputs usually yields a large RMSE drop without changing model structure. I add a `StandardScaler` fitted on the training split and apply it to validation/test for the NN only, leaving your LinearRegression/XGBoost logic unchanged. I also compute the fallback (for filtered-out test rows) from training labels (robust median) instead of potentially NaN predictions, which stabilizes the submission without gaming the metric.'
- What this solution (achieved 1086.87849) has done: 'Your RMSE (1086) is far worse than the target (3.61), so the most likely issue is an invalid/poor submission caused by misalignment between `sample_submission` keys and your predictions after the merge (the merge can reorder rows and/or introduce NaNs if any key mismatch occurs). I keep your exact feature engineering and NN training loop intact, but change the submission construction to preserve the exact row order of `sample_submission.csv` via an index-based reindex, which is a minimal, high-impact correctness fix for Kaggle scoring. I also ensure predictions are strictly 1D float and clipped non-negative (already done) and add a final assert to guarantee the submission row count matches the sample submission. These changes should move the score dramatically down toward the target by fixing evaluation alignment rather than “improving the model.”'
- What this solution (achieved 5.11556) has done: 'Your current RMSE is dramatically worse than the target, which usually indicates a submission/key-alignment or NaN issue rather than model quality. The smallest safe fix is to ensure `key` is treated identically in both `test_df` and `sample_submission` (strip whitespace, same dtype) and to guarantee *every* key gets a finite prediction by filling any missing reindex results with the fallback. I also add a deterministic check that there are no missing keys after reindexing and, if there are, fall back to writing predictions in `sample_submission` row order directly (still using your existing model outputs, not changing training). Core model/feature engineering/training remain unchanged.'
- What this solution (achieved 1086.87849) has done: 'You’re currently worse than the target (RMSE 5.11556 vs 3.61245), so we should make a small, low-risk generalization improvement without changing your feature set or model architecture/training loop. The biggest remaining issue is that the NN is trained directly on raw `fare_amount`, which is heavy-tailed; a minimal metric-consistent fix is to train the same NN on `log1p(fare_amount)` and invert with `expm1` at prediction time, which typically reduces RMSE for this competition. I keep epochs/batch size/architecture identical and leave your linear/XGBoost parts intact; only the NN target transform and its inverse (plus a non-negativity clip) change. This should move your score downward toward the target while preserving the overall approach and producing the same valid submission format.'
- What this solution (achieved 1086.87849) has done: 'Your current RMSE is far worse than the target, so the most likely remaining issue is still correctness-related: `geopy.great_circle` in `calculate_distance()` is extremely slow and can silently break/timeout or behave inconsistently at scale, which can cascade into bad features/predictions. I keep your exact feature set and modeling logic the same, but replace the distance computation with a fast, vectorized Haversine calculation (still “great-circle miles”), which should reliably produce correct `distance` for all rows and improve RMSE substantially toward the target. I also ensure `pickup_datetime` parsing is robust (UTC-naive consistent) and keep your existing filtering, scaling, log1p target transform, and submission alignment unchanged. These are minimal, directly score-relevant changes that preserve the overall approach and still produce `nn_submission.csv`.'
- What this solution (achieved 5.6568) has done: 'Your current RMSE is massively worse than the target, so this is almost certainly still a correctness issue in how the submission is built (key alignment/duplicate keys) rather than model quality. The smallest high-impact fix is to guarantee a 1:1 mapping between each test `key` and exactly one prediction by (a) forcing `key` to string consistently, (b) dropping any duplicate keys in `test_df` before building the prediction Series, and (c) using an explicit “sample_submission order” mapping with a safe fallback for any missing keys. I also ensure the NN submission writes clean float values (1D, finite, non-negative) and add assertions that catch the exact failure mode that leads to enormous Kaggle scores. No model architecture, features, training loop, epochs, batch size, or loss are changed.'
- What this solution (achieved 5.55948) has done: 'Your current RMSE (5.6568) is worse than the target (3.61245), so we should make the smallest change that improves generalization without changing your feature set, model architecture, epochs, batch size, or loss. The biggest remaining issue is that the NN is trained to predict only `log1p(fare_amount)` with plain MSE, which underweights large-dollar errors; for an RMSE-in-dollars metric, a minimal fix is to add a **bias-correction** on the validation set (a single global scaling factor in log-space) and apply the same correction to test predictions. This keeps your training loop identical and only adjusts the final prediction calibration in a metric-consistent way, typically reducing RMSE noticeably on this competition. I also clamp extreme fares to a reasonable upper bound before writing submission (no negatives already), which avoids rare exploding predictions that can disproportionately hurt RMSE.'
- What this solution (achieved 5.17396) has done: 'Your current RMSE (5.55948) is worse than the target (3.61245), so we should make a minimal, low-risk change that improves correctness/generalization without changing your model architecture or training loop. The biggest score drag remaining is likely inconsistent/weak feature scaling for the NN because the NN is trained on standardized inputs but the *distribution shift* from unscaled latitude/longitude and distance still benefits from stabilizing the target distribution further; we keep your `log1p` target but add the standard lognormal bias correction (a single scalar derived from validation residual variance) instead of (or alongside) the alpha grid, which is more metric-aligned for RMSE in dollars. We keep your existing alpha search (so core behavior stays similar) but apply a deterministic bias-correction factor computed on the validation set after choosing `best_alpha`, and apply the same correction to test predictions. This is a tiny post-processing calibration step (no training changes) that typically reduces RMSE on this competition and should move you closer to the target.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
train_df = pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv", nrows=1_000_000
)
test_df = pd.read_csv("/kaggle/input/new-york-city-taxi-fare-prediction/test.csv")



## === cell 2
print(train_df)
print(test_df)

missing_values = train_df.isnull()
ans = missing_values.sum()
print(ans)



## === cell 3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error



## === cell 4
print(train_df.isnull().sum())

train_df.dropna(inplace=True)

train_df = train_df[train_df["passenger_count"] < 8]
train_df = train_df[train_df["fare_amount"] > 0]

print(train_df.isnull().sum())



## === cell 5
min(test_df.pickup_longitude.min(), test_df.dropoff_longitude.min()), max(
    test_df.pickup_longitude.max(), test_df.dropoff_longitude.max()
)



## === cell 6
min(test_df.pickup_latitude.min(), test_df.dropoff_latitude.min()), max(
    test_df.pickup_latitude.max(), test_df.dropoff_latitude.max()
)



## === cell 7
RANGE = (-74.26, -72.99, 40.56, 41.71)


def select_within_boundingbox(df, RANGE):
    return (
        (df.pickup_longitude >= RANGE[0])
        & (df.pickup_longitude <= RANGE[1])
        & (df.pickup_latitude >= RANGE[2])
        & (df.pickup_latitude <= RANGE[3])
        & (df.dropoff_longitude >= RANGE[0])
        & (df.dropoff_longitude <= RANGE[1])
        & (df.dropoff_latitude >= RANGE[2])
        & (df.dropoff_latitude <= RANGE[3])
    )


print("Old size: %d" % len(train_df))
train_df = train_df[select_within_boundingbox(train_df, RANGE)]
print("New size: %d" % len(train_df))




## === cell 8
def haversine_miles(lat1, lon1, lat2, lon2):
    lat1 = np.asarray(lat1, dtype=np.float64)
    lon1 = np.asarray(lon1, dtype=np.float64)
    lat2 = np.asarray(lat2, dtype=np.float64)
    lon2 = np.asarray(lon2, dtype=np.float64)

    rlat1 = np.radians(lat1)
    rlon1 = np.radians(lon1)
    rlat2 = np.radians(lat2)
    rlon2 = np.radians(lon2)

    dlat = rlat2 - rlat1
    dlon = rlon2 - rlon1

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(rlat1) * np.cos(rlat2) * np.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * np.arcsin(np.minimum(1.0, np.sqrt(a)))
    earth_radius_miles = 3958.7613
    return earth_radius_miles * c


def add_distance_to_df(df):
    df["distance"] = haversine_miles(
        df["pickup_latitude"],
        df["pickup_longitude"],
        df["dropoff_latitude"],
        df["dropoff_longitude"],
    )

    df["pickup_datetime"] = pd.to_datetime(
        df["pickup_datetime"], errors="coerce", utc=True
    ).dt.tz_convert(None)
    df["hour"] = df["pickup_datetime"].dt.hour
    df["day"] = df["pickup_datetime"].dt.day
    df["month"] = df["pickup_datetime"].dt.month
    df["year"] = df["pickup_datetime"].dt.year


add_distance_to_df(train_df)
add_distance_to_df(test_df)



## === cell 9
import matplotlib.pyplot as plt


def plt_distance_to_fare(df, sample_size=100_000):
    df = df.sample(sample_size)
    plt.scatter(df["distance"], df["fare_amount"], s=1)
    plt.title("Distance and Fare Amount")
    plt.xlabel("Distance")
    plt.ylabel("Fare Amount")
    plt.show()


plt_distance_to_fare(train_df)



## === cell 10
nn_df = train_df.copy()

train_df = train_df[train_df["distance"] < 25]
train_df = train_df[train_df["distance"] > 0.1]

plt_distance_to_fare(train_df)



## === cell 11
features = [
    "passenger_count",
    "distance",
    "hour",
    "day",
    "month",
    "pickup_latitude",
    "pickup_longitude",
    "dropoff_latitude",
    "dropoff_longitude",
]
X = train_df[features]
y = train_df["fare_amount"]



## === cell 12
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_valid)

rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
print("RMSE:", rmse)




## === cell 13
def plot_linear(X_valid, y_valid, y_pred, column="distance"):
    plt.scatter(X_valid[column], y_valid, color="blue", label="Data", s=2)

    plt.plot(
        X_valid[column], y_pred, color="red", linewidth=1, label="Linear Regression"
    )

    plt.title("Linear Regression Model")
    plt.xlabel("Distance")
    plt.ylabel("Fare Amount")
    plt.legend()
    plt.show()


plot_linear(X_valid, y_valid, y_pred)



## === cell 14
X = test_df[features]

y_pred = model.predict(X)

print(y_pred)
submission_df = test_df[["key"]]
submission_df["fare_amount"] = y_pred
print(submission_df)

submission_df.to_csv("linear_submission.csv", index=False)



## === cell 15
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

X = train_df[features]
y = train_df["fare_amount"]

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = XGBRegressor(learning_rate=0.1)  # You can specify parameters here if needed
model.fit(X_train, y_train)

y_pred = model.predict(X_valid)

rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
print("RMSE:", rmse)



## === cell 16
X = test_df[features]

y_pred = model.predict(X)

print(y_pred)
test_df["fare_amount"] = y_pred
submission_df = test_df[["key", "fare_amount"]]
print(submission_df)
submission_df.to_csv("xgboost_submission.csv", index=False)



## === cell 17
import os
import sys
import subprocess

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "protobuf<5"])
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from keras import backend as K

from sklearn.preprocessing import StandardScaler

X = train_df[features]
y = train_df["fare_amount"]
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_valid_s = scaler.transform(X_valid)

model = Sequential(
    [
        Dense(32, activation="relu", input_shape=(len(features),)),
        Dense(16, activation="relu"),
        Dense(1),
    ]
)

model.compile(optimizer="adam", loss="mean_squared_error")

y_train_nn = np.log1p(y_train.to_numpy(dtype=np.float64))
y_valid_nn = np.log1p(y_valid.to_numpy(dtype=np.float64))

history = model.fit(
    X_train_s,
    y_train_nn,
    validation_data=(X_valid_s, y_valid_nn),
    epochs=2,
    batch_size=32,
)  # keep original training loop/epochs/batch_size



## === cell 18
valid_pred_log = model.predict(X_valid_s, verbose=0)
valid_pred_log = np.asarray(valid_pred_log).reshape(-1).astype(np.float64)

y_valid_d = y_valid.to_numpy(dtype=np.float64)
y_valid_d = np.clip(y_valid_d, 0.0, None)

alphas = np.linspace(0.80, 1.20, 81, dtype=np.float64)  # small, deterministic grid
best_alpha = 1.0
best_rmse = np.inf
best_pred_log = None
for a in alphas:
    pred_d = np.expm1(a * valid_pred_log)
    pred_d = np.clip(pred_d, 0.0, None)
    rmse_a = np.sqrt(mean_squared_error(y_valid_d, pred_d))
    if rmse_a < best_rmse:
        best_rmse = rmse_a
        best_alpha = float(a)
        best_pred_log = a * valid_pred_log

print("Chosen log-space calibration alpha:", best_alpha)
print("NN valid RMSE (dollars, after expm1 + alpha):", float(best_rmse))

resid = (y_valid_nn - best_pred_log).astype(np.float64)
sigma2 = float(np.mean(resid * resid))
bias_correction = float(np.exp(0.5 * sigma2))
print("Validation-derived bias_correction:", bias_correction)

pred_d_bc = np.expm1(best_pred_log) * bias_correction
pred_d_bc = np.clip(pred_d_bc, 0.0, None)
rmse_bc = float(np.sqrt(mean_squared_error(y_valid_d, pred_d_bc)))
print("NN valid RMSE (dollars, after expm1 + alpha + bias_correction):", rmse_bc)

test_mask = (
    select_within_boundingbox(test_df, RANGE)
    & (test_df["distance"] < 25)
    & (test_df["distance"] > 0.1)
)

X_test_filt = test_df.loc[test_mask, features]
X_test_filt_s = scaler.transform(X_test_filt)

y_pred_all = np.empty(len(test_df), dtype=np.float64)
y_pred_all[:] = np.nan

y_pred_filt_log = model.predict(X_test_filt_s, verbose=0)
y_pred_filt_log = np.asarray(y_pred_filt_log).reshape(-1).astype(np.float64)

y_pred_filt = np.expm1(best_alpha * y_pred_filt_log) * bias_correction
y_pred_filt = np.clip(y_pred_filt, 0.0, None)

y_pred_all[test_mask.to_numpy()] = y_pred_filt

fallback = float(np.median(y_train.to_numpy(dtype=np.float64)))
y_pred_all = np.where(np.isnan(y_pred_all), fallback, y_pred_all)

y_pred_all = np.clip(y_pred_all, 0.0, 250.0)

sample_sub = pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/sample_submission.csv"
)

test_keys = test_df["key"].astype(str).str.strip().to_numpy()
sub_keys = sample_sub["key"].astype(str).str.strip().to_numpy()

pred_df = pd.DataFrame({"key": test_keys, "pred": y_pred_all})
dup_ct = int(pred_df["key"].duplicated(keep="first").sum())
if dup_ct > 0:
    print(
        f"[WARN] Found {dup_ct} duplicate test keys; dropping duplicates (keep=first) for safe mapping."
    )
pred_df = pred_df.drop_duplicates(subset=["key"], keep="first")

pred_by_key = pd.Series(
    pred_df["pred"].to_numpy(dtype=np.float64), index=pred_df["key"].to_numpy()
)

reindexed = pred_by_key.reindex(sub_keys)
missing_ct = int(reindexed.isna().sum())
if missing_ct > 0:
    print(
        f"[WARN] {missing_ct} keys missing after reindex; filling with fallback={fallback:.4f}"
    )

sub = sample_sub.copy()
sub["key"] = sub_keys
sub["fare_amount"] = reindexed.to_numpy()

sub["fare_amount"] = pd.to_numeric(sub["fare_amount"], errors="coerce").fillna(fallback)
sub["fare_amount"] = np.clip(sub["fare_amount"].to_numpy(dtype=np.float64), 0.0, 250.0)

assert len(sub) == len(sample_sub), "Submission row count must match sample_submission."
assert list(sub.columns) == [
    "key",
    "fare_amount",
], "Submission must have columns: key,fare_amount"
assert np.isfinite(
    sub["fare_amount"].to_numpy()
).all(), "All predictions must be finite."
assert (
    sub["key"].astype(str).str.strip().equals(sample_sub["key"].astype(str).str.strip())
), "Key order must match sample_submission."

print(sub.head())
print(sub.dtypes)
sub.to_csv("nn_submission.csv", index=False)
