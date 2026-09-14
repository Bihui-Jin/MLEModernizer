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
geopy==2.4.1
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

3.4594

# 6. Current score

6.40629

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 13.37112) has done: 'The changes remove the invalid `%matplotlib inline` line, fix the XGBoost prediction by using the correct Booster API (no `best_ntree_limit` attribute) and update the training parameters to the current objective name. These fixes let the notebook run end‑to‑end and correctly write a `submission.csv` file with the required columns.'
- What this solution (achieved 431.76956) has done: 'I keep the overall workflow and feature set but average the LinearRegression and XGBoost predictions (both already computed) before creating the submission. Averaging usually reduces variance and brings the RMSE down, moving the score closer to the target while leaving the core modeling logic unchanged. The final submission file is still written as `submission.csv` with the required columns.'
- What this solution (achieved 1.4083565406871365e+89) has done: 'Implemented a precise fix for loading the test set:
- Defined `test_usecols` to exclude the non‑existent `fare_amount` column.
- Updated the `load_csv` call for the test data to use these columns.
- Added a short comment explaining the change.

All other logic remains unchanged, ensuring the pipeline runs end‑to‑end and writes a valid `submission.csv` with the required columns.'
- What this solution (achieved 1.4083565406871365e+89) has done: 'Implemented minimal NaN handling to prevent exploding predictions. After feature engineering, any missing values in the training feature matrix and test feature set are filled with 0 so models receive valid inputs, avoiding `NaN`/`inf` outputs that caused the astronomically high RMSE. The change is confined to the feature‑preparation steps, preserving the core modeling logic while steering the validation score toward the target range.'
- What this solution (achieved 37.95211) has done: 'I add a simple outlier filter on the fare amount to remove extreme values that can cause exploding log‑targets, and I clip the predicted log‑values before converting them back to the original scale. These tiny safeguards prevent infinities and keep the RMSE from blowing up, moving the score much closer to the target while preserving the existing model logic.'
- What this solution (achieved 15.84386) has done: 'I remove the log‑transform of the target and train both the LinearRegression and XGBoost models directly on the original fare amounts. This aligns the training objective with the RMSE evaluation metric, which should dramatically lower the validation error and move the score toward the target while keeping the overall pipeline unchanged.'
- What this solution (achieved 6.80285) has done: 'Implemented missing imports, added necessary library initializations, and fixed the execution order so all variables are defined before use. The core modeling logic (feature engineering, LinearRegression, XGBoost, and weighted averaging) remains unchanged, ensuring a valid `submission.csv` is produced while keeping the pipeline stable.'
- What this solution (achieved 6.72851) has done: 'I increase the training sample size (allow up to 5 million rows) and simplify the final prediction to use only the XGBoost model, which is the stronger learner in this pipeline. These minimal tweaks keep the overall architecture unchanged while expected to lower the validation RMSE, moving the score closer to the target.'
- What this solution (achieved 6.40629) has done: 'I keep the overall pipeline unchanged but make three minimal tweaks that are expected to lower the validation RMSE and bring the score closer to the target: (1) strengthen the XGBoost model by using a slightly deeper tree and a smaller learning rate; (2) compute a simple optimal linear‑XGBoost weight on the validation split; (3) apply this weighted ensemble to the test predictions before writing the submission file. This preserves the core logic while improving performance.'

# 9. Code solution

## === cell 0
import pathlib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import xgboost as xgb


def load_csv(fname, **kwargs):
    """Try several possible locations for a CSV file."""
    possible_paths = [
        pathlib.Path(fname),
        pathlib.Path("../input") / fname,
        pathlib.Path("/kaggle/input") / fname,
        pathlib.Path("/kaggle/working") / fname,
    ]
    for p in possible_paths:
        if p.exists():
            return pd.read_csv(p, **kwargs)
    raise FileNotFoundError(f"Could not find {fname}")




## === cell 1
types = {
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "uint8",
}

usecols = [
    "key",
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]

train = load_csv("train.csv", dtype=types, usecols=usecols)

test_usecols = [c for c in usecols if c != "fare_amount"]
test = load_csv("test.csv", dtype=types, usecols=test_usecols)




## === cell 2
max_rows = 5_000_000
if len(train) > max_rows:
    train = train.sample(n=max_rows, random_state=42).reset_index(drop=True)




## === cell 3
train = train.dropna()
train = train[train["fare_amount"] > 0]
train = train[train["fare_amount"] < 500]
train = train[train["pickup_longitude"] < -72]
train = train[(train["pickup_latitude"] > 40) & (train["pickup_latitude"] < 44)]
train = train[train["dropoff_longitude"] < -72]
train = train[(train["dropoff_latitude"] > 40) & (train["dropoff_latitude"] < 44)]
train = train[(train["passenger_count"] > 0) & (train["passenger_count"] < 10)]




## === cell 4
def quick_dist_calc(df):
    """Add Haversine distance (km) between pickup and dropoff."""
    R = 6373.0
    lat1 = np.radians(df["pickup_latitude"].astype("float64"))
    lon1 = np.radians(df["pickup_longitude"].astype("float64"))
    lat2 = np.radians(df["dropoff_latitude"].astype("float64"))
    lon2 = np.radians(df["dropoff_longitude"].astype("float64"))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    df["distance"] = (R * c).astype("float32")
    df["distance_sq"] = (df["distance"] ** 2).astype("float32")
    df["passenger_sq"] = (df["passenger_count"] ** 2).astype("uint16")




## === cell 5
quick_dist_calc(train)
quick_dist_calc(test)




## === cell 6
def parse_datetime(series):
    return pd.to_datetime(
        series.str.slice(0, 19), format="%Y-%m-%d %H:%M:%S", errors="coerce"
    )


train["pickup_datetime"] = parse_datetime(train["pickup_datetime"])
test["pickup_datetime"] = parse_datetime(test["pickup_datetime"])




## === cell 7
for df in (train, test):
    df["hour"] = df["pickup_datetime"].dt.hour.astype("uint8")
    df["weekday"] = df["pickup_datetime"].dt.weekday.astype("uint8")
    df["month"] = df["pickup_datetime"].dt.month.astype("uint8")
    df["year"] = df["pickup_datetime"].dt.year.astype("uint16")




## === cell 8
def sphere_dist(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon):
    R_earth = 6371.0
    pickup_lat, pickup_lon, dropoff_lat, dropoff_lon = map(
        np.radians, [pickup_lat, pickup_lon, dropoff_lat, dropoff_lon]
    )
    dlat = dropoff_lat - pickup_lat
    dlon = dropoff_lon - pickup_lon
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(pickup_lat) * np.cos(dropoff_lat) * np.sin(dlon / 2.0) ** 2
    )
    return 2 * R_earth * np.arcsin(np.sqrt(a))


def add_airport_dist(df):
    jfk = (40.639722, -73.778889)
    ewr = (40.6925, -74.168611)
    lga = (40.77725, -73.872611)
    df["jfk_dist"] = pd.concat(
        [
            sphere_dist(df["pickup_latitude"], df["pickup_longitude"], *jfk),
            sphere_dist(df["dropoff_latitude"], df["dropoff_longitude"], *jfk),
        ],
        axis=1,
    ).min(axis=1)
    df["ewr_dist"] = pd.concat(
        [
            sphere_dist(df["pickup_latitude"], df["pickup_longitude"], *ewr),
            sphere_dist(df["dropoff_latitude"], df["dropoff_longitude"], *ewr),
        ],
        axis=1,
    ).min(axis=1)
    df["lga_dist"] = pd.concat(
        [
            sphere_dist(df["pickup_latitude"], df["pickup_longitude"], *lga),
            sphere_dist(df["dropoff_latitude"], df["dropoff_longitude"], *lga),
        ],
        axis=1,
    ).min(axis=1)
    return df


train = add_airport_dist(train)
test = add_airport_dist(test)




## === cell 9
X = train.drop(columns=["key", "fare_amount", "pickup_datetime"])
X = X.fillna(0)
y = train["fare_amount"]  # original target, no log‑transform




## === cell 10
np.random.seed(42)
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)




## === cell 11
lm = LinearRegression()
lm.fit(X_train, y_train)
val_pred = lm.predict(X_valid)
val_pred = np.clip(val_pred, a_min=0, a_max=500)
linear_rmse = np.sqrt(metrics.mean_squared_error(y_valid, val_pred))
print(f"Linear model validation RMSE (original scale): {linear_rmse:.4f}")




## === cell 12
def train_xgb(X_tr, X_va, y_tr, y_va):
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dvalid = xgb.DMatrix(X_va, label=y_va)
    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "seed": 42,
        "max_depth": 10,  # slightly deeper trees
        "learning_rate": 0.05,  # smaller step size
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "tree_method": "hist",
        "max_bin": 256,
        "nthread": -1,
    }
    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=5000,
        evals=[(dvalid, "valid")],
        early_stopping_rounds=30,
        verbose_eval=False,
    )
    return bst


xgbm = train_xgb(X_train, X_valid, y_train, y_valid)

xgb_val = xgbm.predict(xgb.DMatrix(X_valid))
xgb_val = np.clip(xgb_val, a_min=0, a_max=500)
xgb_rmse = np.sqrt(metrics.mean_squared_error(y_valid, xgb_val))
print(f"XGBoost validation RMSE (original scale): {xgb_rmse:.4f}")




## === cell 13
diff = val_pred - xgb_val
if np.sum(diff**2) > 0:
    w_opt = np.sum((xgb_val - y_valid) * diff) / np.sum(diff**2)
    w_opt = np.clip(w_opt, 0.0, 1.0)
else:
    w_opt = 0.5  # fallback
print(f"Ensemble weight for Linear model: {w_opt:.4f}")

test_features = test.drop(columns=["key", "pickup_datetime"])
test_features = test_features.fillna(0)

lin_test_pred = lm.predict(test_features)
lin_test_pred = np.clip(lin_test_pred, a_min=0, a_max=500)

xgb_test_pred = xgbm.predict(xgb.DMatrix(test_features))
xgb_test_pred = np.clip(xgb_test_pred, a_min=0, a_max=500)

final_test_pred = w_opt * lin_test_pred + (1 - w_opt) * xgb_test_pred
final_test_pred = np.round(final_test_pred, 2)

submission = pd.DataFrame(
    {"key": test["key"], "fare_amount": final_test_pred},
    columns=["key", "fare_amount"],
)

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")




## === cell 14
for var in [
    "train",
    "test",
    "X",
    "y",
    "X_train",
    "X_valid",
    "y_train",
    "y_valid",
    "xgb_test_pred",
    "lin_test_pred",
    "final_test_pred",
]:
    if var in globals():
        del globals()[var]
