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
            new-york-city-taxi-fare-prediction/
                GCP-Coupons-Instructions.rtf (486 Bytes)
                description.md (100 lines)
                ... and 3 other files
        input/
            new-york-city-taxi-fare-prediction/
                GCP-Coupons-Instructions.rtf (486 Bytes)
                description.md (100 lines)
                ... and 3 other files
        working/
            new-york-city-taxi-fare-prediction/
                GCP-Coupons-Instructions.rtf (486 Bytes)
                description.md (100 lines)
                ... and 3 other files
```

-> data/new-york-city-taxi-fare-prediction/labels.csv has 55413942 rows and 8 columns.
The columns are: key, fare_amount, pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count

-> data/new-york-city-taxi-fare-prediction/sample_submission.csv has 9914 rows and 2 columns.
The columns are: key, fare_amount

-> data/new-york-city-taxi-fare-prediction/test.csv has 9914 rows and 7 columns.
The columns are: key, pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count

-> input/new-york-city-taxi-fare-prediction/labels.csv has 55413942 rows and 8 columns.
The columns are: key, fare_amount, pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count

-> input/new-york-city-taxi-fare-prediction/sample_submission.csv has 9914 rows and 2 columns.
The columns are: key, fare_amount

-> input/new-york-city-taxi-fare-prediction/test.csv has 9914 rows and 7 columns.
The columns are: key, pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count

-> working/new-york-city-taxi-fare-prediction/labels.csv has 55413942 rows and 8 columns.
The columns are: key, fare_amount, pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count

-> (stopped after 10 files for performance)

# 5. Target score

4.60368

# 6. Current score

None

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 15.30778) has done: 'I fix the import/optimizer errors, remove the failing water‑mask step, keep the useful passenger_count feature, and adjust the column‑dropping logic. These changes let the notebook run end‑to‑end, produce a valid submission.csv, and should lower the RMSE toward the target.'
- What this solution (achieved 276.72863) has done: 'The changes fix the import errors caused by using the outdated keras API, register the custom rmse metric correctly, and adjust the metric imports so the model can be compiled and trained without crashing. This enables the notebook to run end‑to‑end and produce a valid submissiontry_water.csv file while keeping the original modeling logic unchanged, allowing the RMSE to improve toward the target.'

# 9. Code solution

## === cell 0
import os, pathlib, numpy as np, pandas as pd, matplotlib.pyplot as plt
from datetime import datetime
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from keras.models import Sequential
from keras.layers import Dense, BatchNormalization
from keras import regularizers, optimizers, backend as K

possible_base = [
    "./input/new-york-city-taxi-fare-prediction",
    "./data/new-york-city-taxi-fare-prediction",
    "./working/new-york-city-taxi-fare-prediction",
    "/kaggle/input/new-york-city-taxi-fare-prediction",
    "/kaggle/working/new-york-city-taxi-fare-prediction",
]
BASE_PATH = next(
    (p for p in possible_base if pathlib.Path(p).exists()),
    "./input/new-york-city-taxi-fare-prediction",
)

TRAIN_PATH = os.path.join(BASE_PATH, "labels.csv")
TEST_PATH = os.path.join(BASE_PATH, "test.csv")
SUBMISSION_NAME = "submissiontry_water.csv"

BATCH_SIZE = 256
EPOCHS = 20  # modest epochs for quick run
LEARNING_RATE = 0.001
DATASET_SIZE = 80000  # use a manageable slice of the data

print(f"Using BASE_PATH: {BASE_PATH}")




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Remove obvious outliers and rows with missing values."""
    df = df.dropna()
    lon_min, lon_max = -74.5, -73.5
    lat_min, lat_max = 40.0, 41.0
    mask = (
        df["pickup_longitude"].between(lon_min, lon_max)
        & df["dropoff_longitude"].between(lon_min, lon_max)
        & df["pickup_latitude"].between(lat_min, lat_max)
        & df["dropoff_latitude"].between(lat_min, lat_max)
    )
    if "fare_amount" in df.columns:
        mask &= (df["fare_amount"] > 0) & (df["fare_amount"] < 500)
    return df[mask].reset_index(drop=True)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract year, month, day, hour, weekday and simple time‑of‑day flags."""
    dt = pd.to_datetime(df["pickup_datetime"])
    df["year"] = dt.dt.year
    df["month"] = dt.dt.month
    df["day"] = dt.dt.day
    df["hour"] = dt.dt.hour
    df["weekday"] = dt.dt.weekday
    df["night"] = ((df["hour"] >= 20) | (df["hour"] <= 5)).astype(int)
    df["late_night"] = ((df["hour"] >= 0) & (df["hour"] <= 4)).astype(int)
    df["rush_hour"] = (
        ((df["hour"] >= 7) & (df["hour"] <= 9))
        | ((df["hour"] >= 16) & (df["hour"] <= 19))
    ).astype(int)
    return df


def haversine_np(lon1, lat1, lon2, lat2):
    """Vectorised haversine distance (km)."""
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371 * c


def add_distances_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add haversine distance and simple coordinate differences."""
    df["distance_km"] = haversine_np(
        df["pickup_longitude"],
        df["pickup_latitude"],
        df["dropoff_longitude"],
        df["dropoff_latitude"],
    )
    df["latdiff"] = df["dropoff_latitude"] - df["pickup_latitude"]
    df["londiff"] = df["dropoff_longitude"] - df["pickup_longitude"]
    return df


def plot_loss_accuracy_rmse(history):
    """Plot training/validation loss, MAE and RMSE."""
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.plot(history.history["loss"], label="train")
    plt.plot(history.history["val_loss"], label="val")
    plt.title("Loss")
    plt.legend()
    plt.subplot(1, 3, 2)
    plt.plot(history.history["mae"], label="train")
    plt.plot(history.history["val_mae"], label="val")
    plt.title("MAE")
    plt.legend()
    if "rmse" in history.history:
        plt.subplot(1, 3, 3)
        plt.plot(history.history["rmse"], label="train")
        plt.plot(history.history["val_rmse"], label="val")
        plt.title("RMSE")
        plt.legend()
    plt.tight_layout()
    plt.show()


def output_submission(
    df_test: pd.DataFrame,
    preds: np.ndarray,
    key_col: str,
    target_col: str,
    filename: str,
):
    """Write Kaggle submission file."""
    out = pd.DataFrame({key_col: df_test[key_col], target_col: preds})
    out.to_csv(filename, index=False)
    print(f"Submission written to {filename} (shape {out.shape})")




## === cell 2
train_dtypes = {
    "key": "str",
    "fare_amount": "float32",
    "pickup_datetime": "str",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "uint8",
}
test_dtypes = {
    "key": "str",
    "pickup_datetime": "str",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "uint8",
}

print("Loading data...")
train_full = pd.read_csv(
    TRAIN_PATH,
    nrows=DATASET_SIZE,
    dtype=train_dtypes,
    usecols=list(train_dtypes.keys()),
)
testKaggle = pd.read_csv(
    TEST_PATH,
    dtype=test_dtypes,
    usecols=list(test_dtypes.keys()),
)
print(f"Loaded train rows: {len(train_full)}, test rows: {len(testKaggle)}")



## === cell 3
train_full = clean(train_full)
testKaggle = clean(testKaggle)

train_full = add_time_features(train_full)
testKaggle = add_time_features(testKaggle)

train_full = add_distances_features(train_full)
testKaggle = add_distances_features(testKaggle)

print("Features after engineering (train head):")
print(train_full.head())



## === cell 4
drop_cols = ["key", "pickup_datetime"]
X = train_full.drop(columns=drop_cols + ["fare_amount"])
y = train_full["fare_amount"].values

X_test_raw = testKaggle.drop(columns=drop_cols)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.10, random_state=1)

scaler = preprocessing.MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test_raw)

print(
    f"Train shape: {X_train_scaled.shape}, "
    f"Val shape: {X_val_scaled.shape}, "
    f"Test shape: {X_test_scaled.shape}"
)




## === cell 5
def rmse(y_true, y_pred):
    """Root‑Mean‑Square‑Error metric for Keras."""
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


model = Sequential()
model.add(
    Dense(
        256,
        activation="linear",
        input_dim=X_train_scaled.shape[1],
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
model.compile(
    loss="mean_squared_error",
    optimizer=adam,
    metrics=["mae", rmse, "mse"],
)

print("Model summary:")
model.summary()



## === cell 6
history = model.fit(
    X_train_scaled,
    y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    verbose=1,
    validation_data=(X_val_scaled, y_val),
    shuffle=True,
)

plot_loss_accuracy_rmse(history)



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/1564946326.py in <cell line: 0>()
----> 1 history = model.fit(
      2     X_train_scaled,
      3     y_train,
      4     batch_size=BATCH_SIZE,
      5     epochs=EPOCHS,

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/tmp/ipykernel_11/295588820.py in rmse(y_true, y_pred)
      1 def rmse(y_true, y_pred):
      2     """Root‑Mean‑Square‑Error metric for Keras."""
----> 3     return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))
      4 
      5 

AttributeError: module 'keras.api.backend' has no attribute 'sqrt'

## === cell 7
val_score = model.evaluate(X_val_scaled, y_val, verbose=0)
print(
    f"Validation - MSE: {val_score[0]:.4f}, "
    f"MAE: {val_score[1]:.4f}, RMSE: {val_score[2]:.4f}"
)



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/4161393562.py in <cell line: 0>()
----> 1 val_score = model.evaluate(X_val_scaled, y_val, verbose=0)
      2 print(
      3     f"Validation - MSE: {val_score[0]:.4f}, "
      4     f"MAE: {val_score[1]:.4f}, RMSE: {val_score[2]:.4f}"
      5 )

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/tmp/ipykernel_11/295588820.py in rmse(y_true, y_pred)
      1 def rmse(y_true, y_pred):
      2     """Root‑Mean‑Square‑Error metric for Keras."""
----> 3     return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))
      4 
      5 

AttributeError: module 'keras.api.backend' has no attribute 'sqrt'

## === cell 8
pred_test_raw = model.predict(X_test_scaled, batch_size=128, verbose=0).flatten()
global_mean = y_train.mean()
alpha = 0.5  # modest blend with global mean for calibration
pred_test = alpha * pred_test_raw + (1 - alpha) * global_mean

output_submission(
    testKaggle,
    pred_test,
    key_col="key",
    target_col="fare_amount",
    filename=SUBMISSION_NAME,
)
