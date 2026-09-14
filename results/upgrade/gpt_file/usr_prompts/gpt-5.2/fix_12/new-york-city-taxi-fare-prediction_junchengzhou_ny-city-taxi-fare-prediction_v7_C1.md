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
seaborn==0.12.2
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

4.12199

# 6. Current score

6.24147

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 5.61574) has done: 'I fix the Keras import/backend breakage by switching to `tf_keras` (which matches the installed environment) and implement RMSE using TensorFlow ops so training runs. I also correct the RMSE metric definition (it was missing the mean) while keeping the same model architecture, optimizer, loss, and training loop, which should legitimately move the score down toward your target. Finally, I prevent key parsing issues by keeping `key` as a string ID (not converting to datetime/int), and I ensure the submission uses the correct `key,fare_amount` columns and row alignment.'
- What this solution (achieved 18.08208) has done: 'I fix the runtime crash caused by an incompatibility between `tensorflow` and `tf_keras` in this Kaggle image by switching the model import to the `tf.keras` API (same layers/architecture/training loop) while keeping the custom RMSE metric identical in semantics. I also make the data load paths robust to this notebook’s `/kaggle/input` layout so it runs in your environment without manual path edits. Finally, I add a minimal, competition-standard data cleaning step (filtering obvious invalid coordinates/passenger_count/fare outliers) before scaling/training; this keeps the same feature set and model but typically improves RMSE materially toward your target. The script still write a valid `submission.csv` with `key,fare_amount` and correct row alignment.'
- What this solution (achieved 65.27421) has done: 'I fix the TensorFlow import crash (`MessageFactory.GetPrototype`) by switching the model code back to `tf_keras` (which is installed) while keeping the exact same network architecture, optimizer, loss, and training loop semantics. I also make the input-path resolution robust for this competition’s nested folder layout so `train.csv`/`test.csv` are always found under `/kaggle/input/.../new-york-city-taxi-fare-prediction/`. Finally, I keep the existing feature engineering/cleaning intact and ensure the submission is written as `submission.csv` with the required `key,fare_amount` columns and aligned row order.'
- What this solution (achieved 6.21345) has done: 'I fix the crash in the deep learning imports by using the `tf.keras` API from the already-installed TensorFlow package and avoiding the `tf_keras`/protobuf `MessageFactory.GetPrototype` incompatibility that stops execution. This keeps the exact same network architecture, loss, optimizer, metric, and training loop semantics; it only changes the import surface so the code runs end-to-end. I also make the run deterministic (seeds) to stabilize the resulting score without changing the modeling approach. The pipeline still write a valid `submission.csv` with `key,fare_amount` aligned to the original test order.'
- What this solution (achieved 6.85402) has done: 'I fix the TensorFlow/Keras import crash (`MessageFactory.GetPrototype`) by avoiding `tensorflow.keras` entirely and switching the exact same model code to the installed `tf_keras` package, which is compatible in this environment. I keep the model architecture, loss, optimizer, and training loop identical, and implement RMSE via `tf_keras.backend`/TensorFlow ops so it compiles cleanly. I also add a small fallback to load fewer rows only if the 10M-row read hits memory limits, so the notebook reliably runs end-to-end within Kaggle constraints. The script still write `submission.csv` with exactly the required `key,fare_amount` columns aligned to the original test order.'
- What this solution (achieved 6.21345) has done: 'I fix the runtime crash caused by importing `tf_keras` (protobuf `MessageFactory.GetPrototype` incompatibility) by switching to the already-imported `tensorflow` Keras API (`tf.keras`) while keeping the exact same model architecture, loss, optimizer, and training loop. I also add a small safety cast to ensure the scaled feature matrices are `float32`, avoiding dtype-related slowdowns/issues in TF and keeping behavior consistent. Everything else (data loading, cleaning, feature engineering, scaling, split, epochs, batch size, and submission formatting) remain the same so the pipeline runs end-to-end and writes a valid `submission.csv`.'
- What this solution (achieved 6.2797) has done: 'The crash comes from importing `tensorflow.keras` in this Kaggle image, which triggers a protobuf `MessageFactory.GetPrototype` incompatibility. To keep your exact same model architecture/training loop while restoring execution, I switch only the Keras layer/model imports to the installed `tf_keras` package and keep TensorFlow only for ops/seeding and the RMSE function. I also add a small guard to ensure the submission predictions are finite and clipped to a non-negative fare, which is consistent with the competition target and typically nudges RMSE down without changing the core modeling approach. Everything else (data loading, cleaning, features, scaling, split, epochs, batch size) is left intact, and the script still writes a valid `submission.csv` with `key,fare_amount`.'
- What this solution (achieved 6.21345) has done: 'I fix the runtime crash by avoiding the `tf_keras` import path that triggers the protobuf `MessageFactory.GetPrototype` error and instead use the compatible `tensorflow.keras` API within the already-imported TensorFlow package. I also make `y` explicitly `float32` (and keep feature arrays `float32`) to prevent dtype/object issues during TensorFlow training while keeping the exact same model architecture, loss, optimizer, and training loop. Finally, I keep the existing data loading/cleaning/feature engineering intact and ensure the submission is still written as `submission.csv` with `key,fare_amount` aligned to the original test row order.'
- What this solution (achieved 6.26321) has done: 'I fix the runtime crash in the deep learning imports (`MessageFactory.GetPrototype`) by switching only the Keras layer/model imports from `tensorflow.keras` to the installed `tf_keras` package, while keeping TensorFlow for ops/seeding and keeping the exact same model architecture, loss, optimizer, and training loop. I also ensure the TensorFlow backend used by `tf_keras` is selected before importing `tf_keras`, which avoids backend-mismatch issues in Keras 3 environments. Everything else (data loading, cleaning, feature engineering, scaling, split, epochs, batch size, and submission formatting) remain the same so behavior is preserved and score should move back toward your previously better runs. The script still write a valid `submission.csv` with `key,fare_amount` aligned to the original test order.'
- What this solution (achieved 6.21345) has done: 'I fix the runtime crash coming from `tf_keras`/protobuf (`MessageFactory.GetPrototype`) by switching only the model imports to the stable `tensorflow.keras` API, keeping the exact same network architecture, loss, optimizer, and training loop. To keep score movement minimal and in the right direction, I won’t change feature engineering or training hyperparameters; this is primarily a correctness/unblocking fix. I also ensure TensorFlow is imported early and deterministically seeded, and keep the submission formatting/alignment exactly as required (`key,fare_amount`, `submission.csv`). The rest of the pipeline (data loading, cleaning, scaling, split, fit, predict) remains unchanged.'
- What this solution (achieved 6.24147) has done: 'I fix the runtime crash caused by importing `tensorflow.keras` in this Kaggle image (protobuf `MessageFactory.GetPrototype` issue) by switching the layer/model imports to the installed `tf_keras` package while keeping TensorFlow only for ops/seeding; this preserves the exact same network architecture, optimizer, loss, metric, and training loop. I also set the Keras backend to TensorFlow before importing `tf_keras` to avoid backend-mismatch issues in Keras 3 environments. These changes are execution-unblocking and should also move your score back toward your previously better runs (closer to the 4.12 target) without altering core modeling semantics. The pipeline still write a valid `submission.csv` with the required `key,fare_amount` columns aligned to the original test order.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os

INPUT_DIR_CANDIDATES = ["../input", "/kaggle/input"]
INPUT_DIR = next((p for p in INPUT_DIR_CANDIDATES if os.path.exists(p)), "../input")
print("Using INPUT_DIR =", INPUT_DIR)
print("Top-level contents:", os.listdir(INPUT_DIR)[:50])

COMP_SUBDIR_CANDIDATES = [
    os.path.join(INPUT_DIR, "new-york-city-taxi-fare-prediction"),
    os.path.join(
        INPUT_DIR,
        "new-york-city-taxi-fare-prediction",
        "new-york-city-taxi-fare-prediction",
    ),
    INPUT_DIR,
]
DATA_DIR = next(
    (
        p
        for p in COMP_SUBDIR_CANDIDATES
        if os.path.exists(os.path.join(p, "train.csv"))
        and os.path.exists(os.path.join(p, "test.csv"))
    ),
    None,
)
if DATA_DIR is None:
    raise FileNotFoundError(
        "Could not find train.csv/test.csv under expected Kaggle input locations. "
        f"Tried: {COMP_SUBDIR_CANDIDATES}"
    )
print("Using DATA_DIR =", DATA_DIR)
print("DATA_DIR contents sample:", os.listdir(DATA_DIR)[:20])



## === cell 1
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings("ignore")



## === cell 2
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")

try:
    train = pd.read_csv(TRAIN_PATH, nrows=10_000_000)
except Exception as e:
    print("Warning: failed to load 10,000,000 rows due to:", repr(e))
    print("Falling back to 2,000,000 rows to ensure the notebook completes.")
    train = pd.read_csv(TRAIN_PATH, nrows=2_000_000)



## === cell 3
train.head()



## === cell 4
test = pd.read_csv(TEST_PATH)



## === cell 5
test.head()



## === cell 6
train.dtypes




## === cell 7
def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points on the earth (km)
    """
    lon1 = np.radians(lon1)
    lat1 = np.radians(lat1)
    lon2 = np.radians(lon2)
    lat2 = np.radians(lat2)
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371.0
    return c * r




## === cell 8
def add_travel_distance_vector_features(df):
    df["distance"] = haversine(
        df.dropoff_longitude,
        df.dropoff_latitude,
        df.pickup_longitude,
        df.pickup_latitude,
    )


add_travel_distance_vector_features(train)
add_travel_distance_vector_features(test)



## === cell 9
train.dtypes



## === cell 10
train = train.dropna(how="any", axis="rows")

train = train[(train["fare_amount"] > 0) & (train["fare_amount"] <= 300)]
train = train[(train["passenger_count"] >= 1) & (train["passenger_count"] <= 6)]

for col, lo, hi in [
    ("pickup_longitude", -75, -72),
    ("dropoff_longitude", -75, -72),
    ("pickup_latitude", 40, 42),
    ("dropoff_latitude", 40, 42),
]:
    train = train[(train[col] >= lo) & (train[col] <= hi)]

train = train[(train["distance"] > 0) & (train["distance"] <= 200)]



## === cell 11
train.drop(
    [
        "dropoff_longitude",
        "dropoff_latitude",
        "pickup_longitude",
        "pickup_latitude",
        "pickup_datetime",
    ],
    axis=1,
    inplace=True,
)
test.drop(
    [
        "dropoff_longitude",
        "dropoff_latitude",
        "pickup_longitude",
        "pickup_latitude",
        "pickup_datetime",
    ],
    axis=1,
    inplace=True,
)



## === cell 12
train.head()



## === cell 13
train.isnull().sum()



## === cell 14
train.dropna(how="any", axis="rows", inplace=True)



## === cell 15
train.describe().astype("float16")



## === cell 16
try:
    sns.kdeplot(train.distance, fill=True)
except Exception:
    pass



## === cell 17
train["key"] = train["key"].astype(str)
test["key"] = test["key"].astype(str)



## === cell 18
train.head()



## === cell 19
train.distance.describe().astype("float16")



## === cell 20
y = train.pop("fare_amount")
X = train



## === cell 21
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_key = X["key"].values
X_num = X.drop(["key"], axis=1)

X_num = scaler.fit_transform(X_num)
test_key = test["key"].values
test_num = scaler.transform(test.drop(["key"], axis=1))

X_num = X_num.astype(np.float32, copy=False)
test_num = test_num.astype(np.float32, copy=False)

y = pd.to_numeric(y, errors="coerce").astype(np.float32)



## === cell 22
X_train, X_test, y_train, y_test = train_test_split(
    X_num, y, test_size=0.33, random_state=42
)



## === cell 23
import random
import os as _os

_os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
_os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import tensorflow as tf
import tf_keras
from tf_keras.layers import Dense, Input, Dropout, LeakyReLU
from tf_keras.models import Model

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)




## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 24
def rmse(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))




## === cell 25
def nn(n_feature, k=10):
    model_in = Input(shape=(n_feature,))
    model = Dense(k)(model_in)
    model = LeakyReLU(0.15)(model)
    model = Dropout(0.2)(model)

    model = Dense(k * 4)(model)
    model = LeakyReLU(0.15)(model)
    model = Dropout(0.2)(model)

    model = Dense(k * 16)(model)
    model = LeakyReLU(0.15)(model)
    model = Dropout(0.2)(model)

    model = Dense(k * 16)(model)
    model = LeakyReLU(0.15)(model)
    model = Dropout(0.2)(model)

    model = Dense(k * 4)(model)
    model = LeakyReLU(0.15)(model)
    model = Dropout(0.2)(model)

    model = Dense(k)(model)
    model = LeakyReLU(0.15)(model)
    model = Dropout(0.2)(model)

    model = Dense(1, activation="linear")(model)

    model = Model(inputs=model_in, outputs=model)
    model.compile(loss="mse", optimizer="adam", metrics=[rmse])
    return model




## === cell 26
model = nn(X_train.shape[1])



## === cell 27
history = model.fit(
    X_train,
    y_train,
    batch_size=10000,
    epochs=10,
    verbose=1,
    validation_data=(X_test, y_test),
)



## === cell 28
plt.plot(history.history["rmse"])
plt.plot(history.history["val_rmse"])
plt.title("model rmse")
plt.ylabel("rmse")
plt.xlabel("epoch")
plt.legend(["train", "test"], loc="upper left")
plt.show()



## === cell 29
pres = model.predict(test_num, batch_size=10000)



## === cell 30
test_reload = pd.read_csv(TEST_PATH)
test_reload["key"] = test_reload["key"].astype(str)



## === cell 31
pred = pres.reshape(-1).astype(np.float64)
pred = np.where(np.isfinite(pred), pred, np.nan)
if np.isnan(pred).any():
    pred = np.nan_to_num(pred, nan=float(np.nanmedian(pred)))
pred = np.clip(pred, 0.0, None)

submission = pd.DataFrame(
    {"key": test_reload["key"].values, "fare_amount": pred},
    columns=["key", "fare_amount"],
)



## === cell 32
submission.to_csv("submission.csv", index=False)



## === cell 33
print(os.listdir("."))
print(submission.head())
print("Saved submission.csv with shape:", submission.shape)
