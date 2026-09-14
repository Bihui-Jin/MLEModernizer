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

5.45325

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1086.87849) has done: 'The fix removes the broken Keras import, adds a log‑transform to the XGBoost training (which greatly improves RMSE), updates the validation metric to use the original scale, and ensures the test‑set predictions are inverse‑transformed before saving. The neural‑network cells that caused the `AttributeError` are replaced with a harmless comment so the notebook runs end‑to‑end and produces a valid `xgboost_submission.csv` file.'
- What this solution (achieved 1086.87849) has done: 'I correct the RMSE calculation for the XGBoost model so it compares predictions on the original fare scale (removing the erroneous `np.expm1` on the true values). This small fix brings the validation score down dramatically toward the target while keeping all other logic unchanged.'
- What this solution (achieved 1096.83159) has done: 'I keep the overall workflow and model unchanged but stop discarding data based on the distance feature, which was needlessly shrinking the training set and hurting performance. I also ensure the final prediction file follows Kaggle’s expected name “submission.csv”. These minimal adjustments keep the core logic intact while moving the RMSE much closer to the target.'
- What this solution (achieved 1085.85472) has done: 'I add a modest fare‑amount cap to remove extreme outliers, and clip both validation and test predictions to a realistic upper bound. This keeps the core model unchanged while preventing huge erroneous predictions that inflate RMSE, moving the score much closer to the target.'
- What this solution (achieved 1085.85472) has done: 'I correct the validation scoring in the XGBoost part by keeping the target values in the same transformed scale as the predictions. After splitting, both train and validation targets remain log‑scaled; I inverse‑transform the validation target before computing RMSE so the metric is calculated on the original fare scale, which dramatically reduces the error and moves the score toward the target.'
- What this solution (achieved 7.67547) has done: 'I fix the training pipeline so the XGBoost model is trained directly on the original fare amount (removing the unnecessary log‑transform that inflated the validation error) and I add simple cyclic time features (hour sin/cos) to give the model more useful information. The feature list is updated accordingly, and the XGBoost hyper‑parameters are slightly strengthened (more trees, deeper). These minimal changes keep the overall workflow unchanged while moving the RMSE dramatically closer to the target score.'
- What this solution (achieved 5.8765) has done: 'I keep the overall workflow and feature set unchanged, but switch the XGBoost model to train on a log‑transformed target (`log1p`) and then inverse‑transform the predictions before computing RMSE and creating the submission. This small change aligns the training objective with the skewed fare distribution and is expected to lower the RMSE, moving the score toward the target while preserving the core logic.'
- What this solution (achieved 5.3765) has done: 'The changes add a deterministic down‑sampling step after cleaning to keep the training set to a manageable size and reduce XGBoost’s workload, and they lower the number of trees (n_estimators) to speed up model fitting. Both adjustments keep the same feature engineering, model type, loss function, and evaluation logic, so the results remain comparable while ensuring the whole script finishes within the 600 s limit.'
- What this solution (achieved 5.45325) has done: 'Implemented two fixes: (1) Adjusted the linear‑regression plot to sample a manageable number of points, preventing the Matplotlib overflow error. (2) Enhanced the XGBoost model by increasing tree depth/number and adding early‑stopping on the validation split, which improves predictive performance and moves the RMSE closer to the target while keeping the original workflow intact. The script now runs end‑to‑end and writes a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np

dtype_dict = {
    "key": "object",
    "fare_amount": "float32",
    "pickup_datetime": "object",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int8",
}
usecols = list(dtype_dict.keys())
train_df = pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv",
    usecols=usecols,
    dtype=dtype_dict,
    low_memory=False,
)
test_df = pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv",
    usecols=[c for c in usecols if c != "fare_amount"],
    dtype={k: v for k, v in dtype_dict.items() if k != "fare_amount"},
    low_memory=False,
)

train_df["key"] = train_df["key"].astype("category")
test_df["key"] = test_df["key"].astype("category")



## === cell 1
print(train_df.head())
print(test_df.head())

missing_values = train_df.isnull()
ans = missing_values.sum()
print(ans)



## === cell 2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error



## === cell 3
print(train_df.isnull().sum())

train_df = train_df.dropna().query("passenger_count < 8").query("0 < fare_amount < 200")
print(train_df.isnull().sum())



## === cell 4
min(test_df.pickup_longitude.min(), test_df.dropoff_longitude.min()), max(
    test_df.pickup_longitude.max(), test_df.dropoff_longitude.max()
)



## === cell 5
min(test_df.pickup_latitude.min(), test_df.dropoff_latitude.min()), max(
    test_df.pickup_latitude.max(), test_df.dropoff_latitude.max()
)



## === cell 6
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
train_df = train_df.loc[select_within_boundingbox(train_df, RANGE)].copy()
print("After bounding box: %d" % len(train_df))

MAX_TRAIN_ROWS = 5_000_000  # keep ~5 M rows for fast training
if len(train_df) > MAX_TRAIN_ROWS:
    train_df = train_df.sample(MAX_TRAIN_ROWS, random_state=42)
    print("After down‑sampling: %d" % len(train_df))
else:
    print("No down‑sampling needed.")



## === cell 7
from math import radians, sin, cos, sqrt, atan2


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Vectorized haversine distance in miles.
    """
    R = 3958.8  # Earth radius in miles
    lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
    lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


def add_distance_to_df(df):
    df["distance"] = haversine_distance(
        df["pickup_latitude"],
        df["pickup_longitude"],
        df["dropoff_latitude"],
        df["dropoff_longitude"],
    )
    df["distance_log"] = np.log1p(df["distance"])

    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
    df["hour"] = df["pickup_datetime"].dt.hour
    df["day"] = df["pickup_datetime"].dt.day
    df["month"] = df["pickup_datetime"].dt.month
    df["year"] = df["pickup_datetime"].dt.year

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    df.drop(columns=["pickup_datetime"], inplace=True)


add_distance_to_df(train_df)
add_distance_to_df(test_df)



## === cell 8
import matplotlib.pyplot as plt


def plt_distance_to_fare(df, sample_size=100_000):
    df = df.sample(sample_size, random_state=42)
    plt.scatter(df["distance"], df["fare_amount"], s=1)
    plt.title("Distance and Fare Amount")
    plt.xlabel("Distance")
    plt.ylabel("Fare Amount")
    plt.show()


plt_distance_to_fare(train_df)



## === cell 9
features = [
    "passenger_count",
    "distance",
    "distance_log",
    "hour",
    "hour_sin",
    "hour_cos",
    "day",
    "month",
    "pickup_latitude",
    "pickup_longitude",
    "dropoff_latitude",
    "dropoff_longitude",
]
X = train_df[features].astype("float32")
y = train_df["fare_amount"]



## === cell 10
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_valid)

rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
print("Linear Regression RMSE:", rmse)

del X_train, X_valid, y_train, y_valid, y_pred




## === cell 11
def plot_linear(X_valid, y_valid, y_pred, column="distance"):
    """
    Plot a sample of the validation data to avoid Matplotlib overflow.
    """
    max_points = 200_000
    if len(X_valid) > max_points:
        idx = np.random.RandomState(42).choice(len(X_valid), max_points, replace=False)
        X_plot = X_valid.iloc[idx]
        y_plot = y_valid.iloc[idx]
        y_pred_plot = y_pred[idx]
    else:
        X_plot = X_valid
        y_plot = y_valid
        y_pred_plot = y_pred

    plt.figure(figsize=(8, 5))
    plt.scatter(
        X_plot[column],
        y_plot,
        color="blue",
        label="Data",
        s=2,
        alpha=0.6,
    )
    plt.plot(
        X_plot[column],
        y_pred_plot,
        color="red",
        linewidth=1,
        label="Linear Regression",
    )
    plt.title("Linear Regression Model")
    plt.xlabel(column.capitalize())
    plt.ylabel("Fare Amount")
    plt.legend()
    plt.show()


X_valid_plot, _, y_valid_plot, _ = train_test_split(
    X, y, test_size=0.2, random_state=42
)
y_pred_plot = model.predict(X_valid_plot)
plot_linear(X_valid_plot, y_valid_plot, y_pred_plot)



## === cell 12
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

X = train_df[features].astype("float32")
y = train_df["fare_amount"]

y_log = np.log1p(y)

X_train, X_valid, y_train_log, y_valid_log = train_test_split(
    X, y_log, test_size=0.2, random_state=42
)

model = XGBRegressor(
    n_estimators=400,  # more trees for better fit
    learning_rate=0.05,
    max_depth=12,  # deeper trees
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    n_jobs=-1,
    max_bin=64,
    random_state=42,
    tree_method="hist",
)

model.fit(
    X_train,
    y_train_log,
    eval_set=[(X_valid, y_valid_log)],
    early_stopping_rounds=30,
    verbose=False,
)

y_valid_pred_log = model.predict(X_valid)
y_valid_pred = np.expm1(y_valid_pred_log)
y_valid_pred = np.clip(y_valid_pred, 0, 200)

rmse = np.sqrt(mean_squared_error(np.expm1(y_valid_log), y_valid_pred))
print("XGBoost RMSE (log‑target, original scale):", rmse)

del X_train, X_valid, y_train_log, y_valid_log, y_valid_pred_log, y_valid_pred



## === cell 13
X_test = test_df[features].astype("float32")

y_test_pred_log = model.predict(X_test)
y_test_pred = np.expm1(y_test_pred_log)
y_test_pred = np.clip(y_test_pred, 0, 200)

submission_df = test_df[["key"]].copy()
submission_df["fare_amount"] = y_test_pred
submission_df.to_csv("submission.csv", index=False)
