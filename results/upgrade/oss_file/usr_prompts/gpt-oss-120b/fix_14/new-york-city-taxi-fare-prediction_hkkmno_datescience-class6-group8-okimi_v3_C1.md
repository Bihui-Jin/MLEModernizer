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
pyproj==3.7.1
pyproject_hooks==1.2.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
scipy==1.15.3
seaborn==0.12.2
sklearn-pandas==2.2.0
ydata-profiling==4.17.0

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

3.53576

# 6. Current score

5.82113

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 6.91209) has done: 'I fixed the syntax errors, added handling for missing values, and introduced a small preprocessing step (standard scaling) to improve the K‑Nearest Neighbours model without altering its core logic. The script now runs end‑to‑end, evaluates a validation RMSE, and writes a correct `submission.csv` with the required columns.'
- What this solution (achieved 5.19283) has done: 'I replace the K‑Nearest Neighbours model with a GradientBoostingRegressor (still using scikit‑learn, so the core pipeline stays the same) and add a simple “is_weekend” feature. I also increase the sampled training size modestly (to 500 k rows) to give the tree‑based model more data without exhausting memory. These minimal changes keep the overall structure intact while expected to lower the RMSE toward the target of 3.53576.'
- What this solution (achieved 5.50507) has done: 'The changes replace the plain GradientBoostingRegressor with the much faster HistGradientBoostingRegressor (a drop‑in gradient‑boosting implementation) and enable the experimental import needed for it. This provides the same boosting logic and hyper‑parameters while dramatically reducing training time, keeping all feature engineering and data handling unchanged.'
- What this solution (achieved 5.57503) has done: 'I keep the overall pipeline unchanged but make three small, performance‑oriented tweaks:  
1. Increase the training sample to 2 million rows (still well within memory limits).  
2. Train the model on the log‑transformed target `log1p(fare_amount)` and convert predictions back with `expm1`, a common technique that reduces RMSE for skewed fare data.  
3. Report the validation RMSE on the original scale after the inverse transformation, ensuring the metric reflects the true competition loss.'
- What this solution (achieved 5.61205) has done: 'I increase the training sample to 3 million rows, add a squared‑distance feature, and give the HistGradientBoostingRegressor a bit more capacity (larger max depth and more iterations). These modest changes keep the original pipeline and target‑log transformation while providing the model with richer data, which should lower the validation RMSE and move it closer to the target score.'
- What this solution (achieved 5.82481) has done: 'I add a simple outlier filter (dropping rides with implausibly large distances or fares) and modestly adjust the HistGradientBoostingRegressor’s hyper‑parameters (more boosting iterations, a lower learning rate, and a slightly deeper tree). These minimal changes keep the original pipeline and target‑log transformation while reducing validation RMSE, moving the score closer to the target.'
- What this solution (achieved 10.34505) has done: 'I add cyclic hour/month features (sin / cos) to capture periodic patterns, increase the training sample to 5 million rows, and modestly boost the HistGradientBoostingRegressor capacity (more iterations, deeper trees, lower learning rate). These small, targeted tweaks keep the original pipeline and log‑target handling intact while expected to lower the validation RMSE and thus move the score nearer the target.'
- What this solution (achieved 5.82113) has done: 'The update fixes the haversine distance computation by correcting the latitude difference (`dlat`) to use `lat2 - lat1` instead of the erroneous `lat2 - lon1`. This provides accurate distance features, which are crucial for the model and should lower the RMSE toward the target. No other parts of the pipeline are altered, preserving the original logic and model.'

# 9. Code solution

## === cell 0
import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split as t_t_s
from sklearn.metrics import mean_squared_error as m_sq_er
from sklearn.pipeline import Pipeline

from sklearn.experimental import enable_hist_gradient_boosting  # noqa: F401
from sklearn.ensemble import HistGradientBoostingRegressor

pd.set_option("display.float_format", lambda x: "%.4f" % x)

TRAIN_PATH = "../input/new-york-city-taxi-fare-prediction/train.csv"
TEST_PATH = "../input/new-york-city-taxi-fare-prediction/test.csv"




## === cell 1
def _haversine_km(lon1, lat1, lon2, lat2):
    """
    Fast vectorized haversine distance (km) using numpy.
    Correct calculation of latitude difference ensures accurate distance features.
    """
    R = 6371.0  # Earth radius in km
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1  # corrected latitude difference
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


def add_features(df):
    df = df.dropna(
        subset=[
            "pickup_longitude",
            "pickup_latitude",
            "dropoff_longitude",
            "dropoff_latitude",
            "pickup_datetime",
        ]
    )

    df["distance_km"] = _haversine_km(
        df["pickup_longitude"].values,
        df["pickup_latitude"].values,
        df["dropoff_longitude"].values,
        df["dropoff_latitude"].values,
    )
    df["distance_km_sq"] = df["distance_km"] ** 2
    df["log_distance"] = np.log1p(df["distance_km"])

    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
    df["hour"] = df["pickup_datetime"].dt.hour
    df["dayofweek"] = df["pickup_datetime"].dt.dayofweek
    df["month"] = df["pickup_datetime"].dt.month
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    return df


FEATURES = [
    "distance_km",
    "distance_km_sq",
    "log_distance",
    "hour",
    "dayofweek",
    "month",
    "passenger_count",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
]



## === cell 2
TRAIN_SAMPLE_SIZE = 5_000_000

dtype_map = {
    "key": "string",
    "fare_amount": "float32",
    "pickup_datetime": "string",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int8",
}
usecols = list(dtype_map.keys())

df = pd.read_csv(
    TRAIN_PATH,
    nrows=TRAIN_SAMPLE_SIZE,
    dtype=dtype_map,
    usecols=usecols,
)

df = add_features(df)

outlier_mask = (df["distance_km"] <= 100) & (df["fare_amount"] <= 300)
df = df[outlier_mask]

X = df[FEATURES].astype(np.float32)  # feature matrix
y = df["fare_amount"].astype(np.float32)  # original target

y_log = np.log1p(y)

mask = X.isnull().any(axis=1) | y_log.isnull()
X = X[~mask]
y_log = y_log[~mask]

X_train, X_val, y_train, y_val = t_t_s(X, y_log, test_size=0.2, random_state=42)

gb_pipe = Pipeline(
    [
        (
            "gb",
            HistGradientBoostingRegressor(
                max_iter=800,  # a bit more boosting rounds
                learning_rate=0.03,  # smaller step size for finer fitting
                max_depth=12,  # slightly deeper trees
                max_bins=255,
                random_state=42,
            ),
        )
    ]
)

gb_pipe.fit(X_train, y_train)

val_pred_log = gb_pipe.predict(X_val)
val_pred = np.expm1(val_pred_log)  # back to original fare scale
y_val_original = np.expm1(y_val)  # original fare values

rmse = np.sqrt(m_sq_er(y_val_original, val_pred))
print(f"Validation RMSE (original scale): {rmse:.5f}")



## === cell 3
test_df = pd.read_csv(
    TEST_PATH,
    dtype=dtype_map,
    usecols=[c for c in usecols if c != "fare_amount"],  # test set has no target
)
test_df = add_features(test_df)

X_test = test_df[FEATURES].astype(np.float32)
test_pred_log = gb_pipe.predict(X_test)
test_pred = np.expm1(test_pred_log)  # convert back to fare amount

submission = pd.DataFrame({"key": test_df["key"], "fare_amount": test_pred})

output_path = "submission.csv"
submission.to_csv(output_path, index=False)
print(f"Submission file written to {output_path}")
