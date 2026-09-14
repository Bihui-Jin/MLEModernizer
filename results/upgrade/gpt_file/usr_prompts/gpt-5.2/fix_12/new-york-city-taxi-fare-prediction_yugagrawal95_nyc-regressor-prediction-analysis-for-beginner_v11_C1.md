# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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
lightgbm==4.6.0
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

# 5. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

np.random.seed(42)
os.environ.setdefault("PYTHONHASHSEED", "42")

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")



## === cell 1
DATA_CANDIDATES = [
    "/kaggle/input/new-york-city-taxi-fare-prediction",
    "/kaggle/input",
    "../input",  # keep original as fallback
]
DATA_DIR = None
for d in DATA_CANDIDATES:
    if os.path.exists(os.path.join(d, "train.csv")) and os.path.exists(
        os.path.join(d, "test.csv")
    ):
        DATA_DIR = d
        break
if DATA_DIR is None:
    raise FileNotFoundError(
        "Could not find train.csv/test.csv in expected locations: "
        + ", ".join(DATA_CANDIDATES)
    )

TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")

print("Using DATA_DIR:", DATA_DIR)
print("TRAIN_PATH:", TRAIN_PATH)
print("TEST_PATH:", TEST_PATH)



## === cell 2
TRAIN_NROWS = 200000
usecols_train = [
    "key",
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
usecols_test = [
    "key",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
dtype_common = {
    "passenger_count": "uint8",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
}
dtype_train = dict(dtype_common)
dtype_train["fare_amount"] = "float32"

train_data = pd.read_csv(
    TRAIN_PATH,
    nrows=TRAIN_NROWS,
    usecols=usecols_train,
    dtype=dtype_train,
    parse_dates=["pickup_datetime"],
)
test_data = pd.read_csv(
    TEST_PATH,
    usecols=usecols_test,
    dtype=dtype_common,
    parse_dates=["pickup_datetime"],
)



## === cell 3
_ = train_data.shape
_ = test_data.shape




## === cell 4
def changeDataType(dataset):
    dataset["pickup_datetime"] = pd.to_datetime(
        dataset["pickup_datetime"], errors="coerce", utc=True
    )


changeDataType(train_data)
changeDataType(test_data)



## === cell 5
pass



## === cell 6
pass



## === cell 7
train_data = train_data.dropna(axis=0)



## === cell 8
pass



## === cell 9
pass



## === cell 10
train_data = train_data.loc[train_data["fare_amount"] > 0]



## === cell 11
pass



## === cell 12
pass



## === cell 13
pass



## === cell 14
train_data = train_data[train_data.fare_amount < 400]



## === cell 15
pass



## === cell 16
pass



## === cell 17
train_data = train_data[train_data.passenger_count <= 6]



## === cell 18
pass



## === cell 19
pass



## === cell 20
mask = (
    train_data["pickup_latitude"].between(-90, 90)
    & train_data["dropoff_latitude"].between(-90, 90)
    & train_data["pickup_longitude"].between(-180, 180)
    & train_data["dropoff_longitude"].between(-180, 180)
)
train_data = train_data.loc[mask]



## === cell 21
pl_min, pl_max = test_data.pickup_latitude.min(), test_data.pickup_latitude.max()
plo_min, plo_max = test_data.pickup_longitude.min(), test_data.pickup_longitude.max()
dl_min, dl_max = test_data.dropoff_latitude.min(), test_data.dropoff_latitude.max()
dlo_min, dlo_max = test_data.dropoff_longitude.min(), test_data.dropoff_longitude.max()

mask = (
    train_data.pickup_latitude.between(pl_min, pl_max)
    & train_data.pickup_longitude.between(plo_min, plo_max)
    & train_data.dropoff_latitude.between(dl_min, dl_max)
    & train_data.dropoff_longitude.between(dlo_min, dlo_max)
)
train_data = train_data.loc[mask]



## === cell 22
pass




## === cell 23
def degree_to_radion(degree):
    return degree * (np.pi / 180.0)


def calculate_distance(
    pickup_latitude, pickup_longitude, dropoff_latitude, dropoff_longitude
):
    """
    Keep same distance feature (Haversine), vectorized via numpy.
    """
    from_lat = degree_to_radion(pickup_latitude)
    from_long = degree_to_radion(pickup_longitude)
    to_lat = degree_to_radion(dropoff_latitude)
    to_long = degree_to_radion(dropoff_longitude)

    radius = 6371.01
    lat_diff = to_lat - from_lat
    long_diff = to_long - from_long

    a = (
        np.sin(lat_diff / 2.0) ** 2
        + np.cos(from_lat) * np.cos(to_lat) * np.sin(long_diff / 2.0) ** 2
    )
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return radius * c




## === cell 24
train_data["distance"] = calculate_distance(
    train_data.pickup_latitude,
    train_data.pickup_longitude,
    train_data.dropoff_latitude,
    train_data.dropoff_longitude,
)



## === cell 25
pass



## === cell 26
test_data["distance"] = calculate_distance(
    test_data.pickup_latitude,
    test_data.pickup_longitude,
    test_data.dropoff_latitude,
    test_data.dropoff_longitude,
)



## === cell 27
pass



## === cell 28
pass



## === cell 29
train_data = train_data.loc[train_data.distance < 200]



## === cell 30
pass



## === cell 31
pass



## === cell 32
train_data = train_data.drop(columns="key")



## === cell 33
pass



## === cell 34
test_data_key = test_data["key"].copy()
test_data = test_data.drop(columns="key")
assert len(test_data_key) == len(test_data), "Test key length mismatch with test rows."



## === cell 35
pass



## === cell 36
for df in (train_data, test_data):
    dt = df["pickup_datetime"].dt
    df["Year"] = dt.year
    df["Month"] = dt.month
    df["Date"] = dt.day
    df["Day of Week"] = dt.dayofweek
    df["Hour"] = dt.hour



## === cell 37
pass



## === cell 38
pass



## === cell 39
pass



## === cell 40
pass



## === cell 41
pass



## === cell 42
pass



## === cell 43
pass



## === cell 44
pass



## === cell 45
pass



## === cell 46
pass



## === cell 47
pass



## === cell 48
pass



## === cell 49
pass



## === cell 50
pass



## === cell 51
pass



## === cell 52
pass



## === cell 53
pass



## === cell 54
pass



## === cell 55
pass



## === cell 56
pass



## === cell 57
nz_mask = (
    (train_data.pickup_latitude != 0)
    & (train_data.pickup_longitude != 0)
    & (train_data.dropoff_latitude != 0)
    & (train_data.dropoff_longitude != 0)
)
train_data = train_data.loc[nz_mask]



## === cell 58
zero_cols = [
    "pickup_latitude",
    "pickup_longitude",
    "dropoff_latitude",
    "dropoff_longitude",
]
for c in zero_cols:
    test_data.loc[test_data[c] == 0, c] = np.nan
test_data[zero_cols] = test_data[zero_cols].fillna(test_data[zero_cols].median())



## === cell 59
train_data = train_data.drop(columns="pickup_datetime", axis=1)
test_data = test_data.drop(columns="pickup_datetime", axis=1)



## === cell 60
X = train_data.loc[:, train_data.columns != "fare_amount"]
y = train_data["fare_amount"]



## === cell 61
try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except Exception:
    pass

from sklearn.model_selection import KFold
from sklearn.base import BaseEstimator, clone
import lightgbm as lgb
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor



## === cell 62
pass



## === cell 63
pass



## === cell 64
pass



## === cell 65
pass



## === cell 66
pass



## === cell 67
pass



## === cell 68
pass



## === cell 69
pass



## === cell 70
pass



## === cell 71
n_folds = 5


def rmsle_cv(model):
    raise RuntimeError(
        "rmsle_cv was disabled for runtime; not needed for submission generation."
    )




## === cell 72
GBoost = GradientBoostingRegressor(
    n_estimators=3000,
    learning_rate=0.05,
    max_depth=4,
    max_features="sqrt",
    min_samples_leaf=15,
    min_samples_split=10,
    loss="huber",
    random_state=5,
)



## === cell 73
model_xgb = XGBRegressor(
    colsample_bytree=0.4603,
    gamma=0.0468,
    learning_rate=0.05,
    max_depth=3,
    min_child_weight=1.7817,
    n_estimators=2200,
    reg_alpha=0.4640,
    reg_lambda=0.8571,
    subsample=0.5213,
    random_state=7,
    nthread=4,
    verbosity=0,
    tree_method="hist",
)



## === cell 74
model_lgb1 = lgb.LGBMRegressor(
    objective="regression",
    num_leaves=5,
    learning_rate=0.05,
    n_estimators=720,
    max_bin=55,
    bagging_fraction=0.8,
    bagging_freq=5,
    feature_fraction=0.2319,
    feature_fraction_seed=9,
    bagging_seed=9,
    min_data_in_leaf=6,
    min_sum_hessian_in_leaf=11,
    n_jobs=4,
    force_col_wise=True,
)



## === cell 75
pass



## === cell 76
pass



## === cell 77
pass




## === cell 78
class AverageModel(BaseEstimator):
    def __init__(self, models):
        self.models = models

    def fit(self, X, y):
        for model in self.models:
            model.fit(X, y)
        return self

    def predict(self, X):
        predictions = np.column_stack([model.predict(X) for model in self.models])
        return np.mean(predictions, axis=1)




## === cell 79
pass



## === cell 80
pass




## === cell 81
class stackingModel(BaseEstimator):
    def __init__(self, base_model, meta_model, k_fold=5):
        self.base_model = base_model
        self.meta_model = meta_model
        self.k_fold = k_fold

    def fit(self, X, y):
        kfold = KFold(n_splits=self.k_fold, shuffle=True, random_state=156)
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_model)))
        for i, model in enumerate(self.base_model):
            for train_index, holdout_index in kfold.split(X, y):
                model.fit(X.iloc[train_index], y.iloc[train_index])
                y_pred = model.predict(X.iloc[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred
        self.meta_model.fit(out_of_fold_predictions, y)
        return self

    def predict(self, X):
        meta_features = np.column_stack([model.predict(X) for model in self.base_model])
        return self.meta_model.predict(meta_features)




## === cell 82
base_model = [GBoost, model_xgb, model_lgb1]


def test1(X, y):
    raise RuntimeError(
        "test1 disabled for runtime; final stacking pipeline computes OOF directly."
    )




## === cell 83
pass



## === cell 84
X = X.reindex(sorted(X.columns), axis=1)
test_data = test_data.reindex(X.columns, axis=1)

train_medians = X.median(numeric_only=True)
X = X.fillna(train_medians)
test_data = test_data.fillna(train_medians)

X_np = np.ascontiguousarray(X.to_numpy(dtype=np.float32, copy=False))
y_np = np.ascontiguousarray(y.to_numpy(dtype=np.float32, copy=False))
test_np = np.ascontiguousarray(test_data.to_numpy(dtype=np.float32, copy=False))

kfold_full = KFold(n_splits=5, shuffle=True, random_state=0)
fold_indices = list(kfold_full.split(X_np, y_np))

base_model_prototypes = [clone(m) for m in base_model]
n_models = len(base_model_prototypes)
n_rows = X_np.shape[0]
oof_full = np.zeros((n_rows, n_models), dtype=np.float32)

for i, proto in enumerate(base_model_prototypes):
    for tr_idx, ho_idx in fold_indices:
        fold_model = clone(proto)
        fold_model.fit(X_np[tr_idx], y_np[tr_idx])
        oof_full[ho_idx, i] = fold_model.predict(X_np[ho_idx]).astype(
            np.float32, copy=False
        )

meta_model_full = lgb.LGBMRegressor(
    objective="regression",
    learning_rate=0.05,
    n_estimators=800,
    random_state=0,
    n_jobs=4,
    force_col_wise=True,
)
meta_model_full.fit(oof_full, y_np)

base_models_fitted_full = []
for proto in base_model_prototypes:
    m = clone(proto)
    m.fit(X_np, y_np)
    base_models_fitted_full.append(m)

feature_data = np.empty((test_np.shape[0], n_models), dtype=np.float32)
for i, m in enumerate(base_models_fitted_full):
    feature_data[:, i] = m.predict(test_np).astype(np.float32, copy=False)

assert feature_data.shape[0] == len(
    test_data_key
), "Meta-feature/test key row mismatch with test rows."
_ = feature_data.shape



## === cell 85
meta_y = meta_model_full.predict(feature_data)



## === cell 86
meta_y = np.clip(meta_y, 0, None)

submission = pd.DataFrame({"key": test_data_key.values, "fare_amount": meta_y})
submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
