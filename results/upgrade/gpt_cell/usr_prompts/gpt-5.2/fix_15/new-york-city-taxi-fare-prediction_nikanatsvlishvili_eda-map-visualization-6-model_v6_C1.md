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

3.10

# 3. Installed packages

folium==0.20.0
geopandas==0.14.4
haversine==2.9.0
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

22.32348

# 6. Current score

141244.91462

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1008.51236) has done: 'Your notebook currently never reaches a training/prediction/submission step, so it can’t yield a Kaggle score; the minimal improvement is to add the missing final cells that (1) build the feature matrix from the columns you already engineered, (2) fit the existing Keras Dense/BatchNorm/Dropout regression model (same core approach), and (3) write a valid `submission.csv` with `key,fare_amount` aligned to `test`. To keep runtime within limits while improving RMSE toward your target, I’m also making one small but directly score-relevant fix: compute `fare_to_dist_ratio` only on train features is a leakage feature (it uses the label) and also can’t be computed for test, so it must be excluded from the model feature set (but I keep the column creation unchanged to preserve your workflow). Finally, I add a lightweight validation split for sanity (no early stopping) and clip negative predictions to 0 to avoid obviously invalid fares that typically hurt RMSE.'
- What this solution (achieved 829.1278) has done: 'Your current pipeline can’t yield a valid Kaggle score because you drop outlier rows from `test` and then submit fewer than the required 9914 predictions; Kaggle reject or score it incorrectly. I keep the same feature engineering and the same Keras Dense/BatchNorm/Dropout regressor, but change test handling to never drop rows—instead, we keep all `key`s and only clip out-of-bounds coordinates to the NYC bounding box so the feature distribution stays reasonable. I also ensure `train` and `test` get identical numeric preprocessing (inf/NaN handling) and keep the non-negative fare clipping, which typically improves RMSE slightly without changing the modeling approach. These minimal changes should produce a valid `submission.csv` and move RMSE toward your target by fixing the submission validity and avoiding pathological test features.'
- What this solution (achieved 141244.91462) has done: 'Your RMSE is extremely high because the model is likely producing wildly out-of-range predictions on the test set (even after non-negative clipping), which usually comes from training on a noisy/unrepresentative sample plus no robust handling of coordinate outliers in training and target skew. To move your score much closer to the 22.32348 target while keeping the same core feature engineering and the same Keras Dense/BatchNorm/Dropout regressor, I make two minimal, score-relevant adjustments: (1) clip *train* coordinates to the same NYC bounding box as test (instead of dropping them, which can distort learning), and (2) train on `log1p(fare_amount)` and invert with `expm1` at prediction time (same MSE loss, but stabilizes regression and reduces catastrophic errors). I also ensure the train/test preprocessing remains identical (NaN/inf handling and scaling) and keep the non-negative fare clipping so submissions remain valid.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "3"

import sys, subprocess

try:
    import google.protobuf as _pb
    from packaging import version as _version

    if _version.parse(getattr(_pb, "__version__", "0")) >= _version.parse("6.0.0"):
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "protobuf<6"]
        )
        for _m in list(sys.modules.keys()):
            if _m.startswith("google.protobuf"):
                del sys.modules[_m]
except Exception:
    pass

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

get_ipython().run_line_magic("matplotlib", "inline")
from mpl_toolkits import mplot3d
import seaborn as sns

import math
from math import sqrt

from numpy import absolute
from numpy import mean
from numpy import std

from sklearn import metrics
from sklearn.feature_selection import f_regression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import RepeatedKFold
from sklearn import neighbors

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.layers import BatchNormalization



## === cell 2
train = pd.read_csv(
    "../input/new-york-city-taxi-fare-prediction/train.csv",
    nrows=100000,
    parse_dates=["pickup_datetime"],
)
test = pd.read_csv("../input/new-york-city-taxi-fare-prediction/test.csv")



## === cell 3
print(train.shape)
print(test.shape)



## === cell 4
train.head()



## === cell 5
train.dtypes



## === cell 6
train.describe()



## === cell 7
print(train.isnull().sum())



## === cell 8
train = train.dropna(how="any", axis="rows")



## === cell 9
print("Old size: %d" % len(train))



## === cell 10
train = train.drop(train[train.fare_amount < 2.5].index, axis=0)
train = train.drop(train[train.fare_amount > 300].index, axis=0)



## === cell 11
train = train.drop(train[train["passenger_count"] > 6].index, axis=0)
train = train.drop(train[train["passenger_count"] < 0].index, axis=0)



## === cell 12
train = train.drop(train[train["pickup_latitude"] < -90].index, axis=0)
train = train.drop(train[train["pickup_latitude"] > 90].index, axis=0)



## === cell 13
train = train.drop(train[train["pickup_longitude"] < -180].index, axis=0)
train = train.drop(train[train["pickup_longitude"] > 180].index, axis=0)



## === cell 14
train = train.drop(train[train["dropoff_latitude"] < -90].index, axis=0)
train = train.drop(train[train["dropoff_latitude"] > 90].index, axis=0)

train = train.drop(train[train["dropoff_longitude"] < -180].index, axis=0)
train = train.drop(train[train["dropoff_longitude"] > 180].index, axis=0)




## === cell 15
def select_outside_boundingbox(df, BB):
    filter_df = df.loc[
        (df["pickup_longitude"] < BB[0])
        | (df["pickup_longitude"] > BB[1])
        | (df["pickup_latitude"] < BB[2])
        | (df["pickup_latitude"] > BB[3])
        | (df["dropoff_longitude"] < BB[0])
        | (df["dropoff_longitude"] > BB[1])
        | (df["dropoff_latitude"] < BB[2])
        | (df["dropoff_latitude"] > BB[3])
    ]
    return filter_df


NYC_BB = (-74.5, -72.8, 40.5, 41.8)



## === cell 16
outliers = select_outside_boundingbox(train, NYC_BB)
outliers




## === cell 17
def clip_to_boundingbox(df, BB):
    df = df.copy()
    df["pickup_longitude"] = df["pickup_longitude"].clip(BB[0], BB[1])
    df["dropoff_longitude"] = df["dropoff_longitude"].clip(BB[0], BB[1])
    df["pickup_latitude"] = df["pickup_latitude"].clip(BB[2], BB[3])
    df["dropoff_latitude"] = df["dropoff_latitude"].clip(BB[2], BB[3])
    return df


train = clip_to_boundingbox(train, NYC_BB)



## === cell 18
print("New size: %d" % len(train))



## === cell 19
test.dtypes



## === cell 20
train["loc1"] = train[["pickup_latitude", "pickup_longitude"]].apply(tuple, axis=1)
train["loc2"] = train[["dropoff_latitude", "dropoff_longitude"]].apply(tuple, axis=1)




## === cell 21
def haversine_vectorized(lat1, lon1, lat2, lon2):
    R = 6371.0  # km
    lat1 = np.radians(lat1.astype("float64"))
    lon1 = np.radians(lon1.astype("float64"))
    lat2 = np.radians(lat2.astype("float64"))
    lon2 = np.radians(lon2.astype("float64"))

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(a))
    return (R * c).astype("float32")


train["H_Distance"] = haversine_vectorized(
    train["pickup_latitude"].values,
    train["pickup_longitude"].values,
    train["dropoff_latitude"].values,
    train["dropoff_longitude"].values,
)




## === cell 22
def chebyshev(pickup_long, dropoff_long, pickup_lat, dropoff_lat):
    return np.maximum(
        np.absolute(pickup_long - dropoff_long), np.absolute(pickup_lat - dropoff_lat)
    )


train["Chebyshev"] = chebyshev(
    train["pickup_longitude"],
    train["dropoff_longitude"],
    train["pickup_latitude"],
    train["dropoff_latitude"],
)



## === cell 23
train.head()



## === cell 24
train["hour"] = train.pickup_datetime.dt.hour
train["day_of_week"] = train.pickup_datetime.dt.weekday
train["day_of_month"] = train.pickup_datetime.dt.day
train["week"] = train.pickup_datetime.dt.isocalendar().week.astype(int)
train["month"] = train.pickup_datetime.dt.month
train["year"] = train.pickup_datetime.dt.year - 2000
train["minute"] = train["pickup_datetime"].dt.minute
train["second"] = train["pickup_datetime"].dt.second
train["dayofyear"] = train["pickup_datetime"].dt.dayofyear



## === cell 25
train.head()



## === cell 26
train["fare_to_dist_ratio"] = train["fare_amount"] / (train["H_Distance"] + 0.0001)



## === cell 27
train = train.drop(train[train["loc1"] == train["loc2"]].index, axis=0)




## === cell 28
def add_distances_from_airport(dataset):
    jfk_lat, jfk_lon = 40.639722, -73.778889
    ewr_lat, ewr_lon = 40.6925, -74.168611
    lga_lat, lga_lon = 40.77725, -73.872611

    dataset["pickup_jfk_distance"] = haversine_vectorized(
        dataset["pickup_latitude"].values,
        dataset["pickup_longitude"].values,
        np.full(len(dataset), jfk_lat, dtype="float32"),
        np.full(len(dataset), jfk_lon, dtype="float32"),
    )
    dataset["dropof_jfk_distance"] = haversine_vectorized(
        dataset["dropoff_latitude"].values,
        dataset["dropoff_longitude"].values,
        np.full(len(dataset), jfk_lat, dtype="float32"),
        np.full(len(dataset), jfk_lon, dtype="float32"),
    )

    dataset["pickup_ewr_distance"] = haversine_vectorized(
        dataset["pickup_latitude"].values,
        dataset["pickup_longitude"].values,
        np.full(len(dataset), ewr_lat, dtype="float32"),
        np.full(len(dataset), ewr_lon, dtype="float32"),
    )
    dataset["dropof_ewr_distance"] = haversine_vectorized(
        dataset["dropoff_latitude"].values,
        dataset["dropoff_longitude"].values,
        np.full(len(dataset), ewr_lat, dtype="float32"),
        np.full(len(dataset), ewr_lon, dtype="float32"),
    )

    dataset["pickup_lga_distance"] = haversine_vectorized(
        dataset["pickup_latitude"].values,
        dataset["pickup_longitude"].values,
        np.full(len(dataset), lga_lat, dtype="float32"),
        np.full(len(dataset), lga_lon, dtype="float32"),
    )
    dataset["dropof_lga_distance"] = haversine_vectorized(
        dataset["dropoff_latitude"].values,
        dataset["dropoff_longitude"].values,
        np.full(len(dataset), lga_lat, dtype="float32"),
        np.full(len(dataset), lga_lon, dtype="float32"),
    )

    return dataset


train = add_distances_from_airport(train)



## === cell 29
test["pickup_datetime"] = pd.to_datetime(test["pickup_datetime"])

test["loc1"] = test[["pickup_latitude", "pickup_longitude"]].apply(tuple, axis=1)
test["loc2"] = test[["dropoff_latitude", "dropoff_longitude"]].apply(tuple, axis=1)

test["H_Distance"] = haversine_vectorized(
    test["pickup_latitude"].values,
    test["pickup_longitude"].values,
    test["dropoff_latitude"].values,
    test["dropoff_longitude"].values,
)

test["Chebyshev"] = chebyshev(
    test["pickup_longitude"],
    test["dropoff_longitude"],
    test["pickup_latitude"],
    test["dropoff_latitude"],
)

test["hour"] = test.pickup_datetime.dt.hour
test["day_of_week"] = test.pickup_datetime.dt.weekday
test["day_of_month"] = test.pickup_datetime.dt.day
test["week"] = test.pickup_datetime.dt.isocalendar().week.astype(int)
test["month"] = test.pickup_datetime.dt.month
test["year"] = test.pickup_datetime.dt.year - 2000
test["minute"] = test["pickup_datetime"].dt.minute
test["second"] = test["pickup_datetime"].dt.second
test["dayofyear"] = test["pickup_datetime"].dt.dayofyear

test = add_distances_from_airport(test)




## === cell 30
def downcast(df):
    df_int = df.select_dtypes(include=["int64", "int32", "int16", "int8", "int"])
    df[df_int.columns] = df_int.apply(pd.to_numeric, downcast="unsigned")

    df_float = df.select_dtypes(include=["float64", "float32", "float16", "float"])
    df[df_float.columns] = df_float.apply(pd.to_numeric, downcast="float")

    return df


downcast(train)
downcast(test)
train.dtypes



## === cell 31
train.plot(y="pickup_latitude", x="pickup_longitude", kind="scatter", alpha=0.7, s=0.02)
city_long_border = (-74.03, -73.75)
city_lat_border = (40.63, 40.85)
plt.title("Pickups Data")
plt.ylim(city_lat_border)
plt.xlim(city_long_border)
plt.show()



## === cell 32
train.plot(
    y="dropoff_latitude", x="dropoff_longitude", kind="scatter", alpha=0.5, s=0.02
)
city_long_border = (-74.03, -73.75)
city_lat_border = (40.63, 40.85)
plt.title("Dropoff Data")
plt.ylim(city_lat_border)
plt.xlim(city_long_border)
plt.show()



## === cell 33
import folium

long_trips = train[train["H_Distance"] >= 10]
drop_map = folium.Map(location=[40.730610, -73.935242], zoom_start=12)

for index, row in long_trips.iterrows():
    folium.CircleMarker(
        [row["dropoff_latitude"], row["dropoff_longitude"]],
        radius=3,
        color="green",
        fill_opacity=0.9,
    ).add_to(drop_map)

for index, row in long_trips.iterrows():
    folium.CircleMarker(
        [row["pickup_latitude"], row["pickup_longitude"]],
        radius=3,
        color="blue",
        fill_opacity=0.9,
    ).add_to(drop_map)

drop_map



## === cell 34
test = clip_to_boundingbox(test, NYC_BB)

feature_cols = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
    "H_Distance",
    "Chebyshev",
    "hour",
    "day_of_week",
    "day_of_month",
    "week",
    "month",
    "year",
    "minute",
    "second",
    "dayofyear",
    "pickup_jfk_distance",
    "dropof_jfk_distance",
    "pickup_ewr_distance",
    "dropof_ewr_distance",
    "pickup_lga_distance",
    "dropof_lga_distance",
]

missing_train = [c for c in feature_cols if c not in train.columns]
missing_test = [c for c in feature_cols if c not in test.columns]
print("Missing in train:", missing_train)
print("Missing in test:", missing_test)

X = train[feature_cols].copy()
y_raw = train["fare_amount"].astype("float32").copy()
X_test = test[feature_cols].copy()

X = X.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
y_raw = y_raw.loc[X.index]

train_medians = X.median(numeric_only=True)
X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(train_medians)

y = np.log1p(y_raw.values).astype("float32")



## === cell 35
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X.values)
X_test_scaled = scaler.transform(X_test.values)

X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print(X_train.shape, X_val.shape, X_test_scaled.shape)



## === cell 36
tf.random.set_seed(42)
np.random.seed(42)

model = Sequential(
    [
        Dense(128, activation="relu", input_shape=(X_train.shape[1],)),
        BatchNormalization(),
        Dropout(0.2),
        Dense(64, activation="relu"),
        BatchNormalization(),
        Dropout(0.2),
        Dense(1, activation="linear"),
    ]
)

model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss=MeanSquaredError(),
)

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=15,
    batch_size=1024,
    verbose=2,
)

val_pred_log = model.predict(X_val, batch_size=4096).reshape(-1)
val_pred = np.expm1(val_pred_log)
val_pred = np.clip(val_pred, 0.0, None)
val_rmse = mean_squared_error(np.expm1(y_val), val_pred, squared=False)
print("Validation RMSE:", val_rmse)



## === cell 37
test_pred_log = model.predict(X_test_scaled, batch_size=4096).reshape(-1)
test_pred = np.expm1(test_pred_log)

test_pred = np.clip(test_pred, 0.0, None)

submission = pd.DataFrame(
    {
        "key": test["key"].values,
        "fare_amount": test_pred.astype("float32"),
    }
)

assert submission.shape[0] == len(
    test
), f"Submission rows {submission.shape[0]} != {len(test)}"
assert submission["key"].isna().sum() == 0, "Found NA keys in submission"

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print("Wrote:", submission_path)
print(submission.head())
print("Rows:", len(submission), "Expected:", len(test))
