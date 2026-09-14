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
import warnings
import random
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

SEED = 0
random.seed(SEED)
np.random.seed(SEED)

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

INPUT_DIR_CANDIDATES = [
    "/kaggle/input",
    "../input",
    "/kaggle/data",
]


def _resolve_input_file(fname: str) -> str:
    for d in INPUT_DIR_CANDIDATES:
        p = os.path.join(d, fname)
        if os.path.exists(p):
            return p
    for d in INPUT_DIR_CANDIDATES:
        if os.path.exists(d):
            for root, _, files in os.walk(d):
                if fname in files:
                    return os.path.join(root, fname)
    raise FileNotFoundError(f"Could not find {fname} under {INPUT_DIR_CANDIDATES}")


TRAIN_PATH = _resolve_input_file("train.csv")
TEST_PATH = _resolve_input_file("test.csv")
SAMPLE_SUB_PATH = _resolve_input_file("sample_submission.csv")

DO_PLOTS = False
if DO_PLOTS:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns



## === cell 1
try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except Exception:
    pass



## === cell 2
NROWS_TRAIN = 500000  # keep as-is (core logic expectation)

train_usecols = [
    "key",
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
test_usecols = [
    "key",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]

dtype_train = {
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "uint8",
}
dtype_test = {
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "uint8",
}

train_data = pd.read_csv(
    TRAIN_PATH,
    nrows=NROWS_TRAIN,
    usecols=train_usecols,
    dtype=dtype_train,
    parse_dates=["pickup_datetime"],
)
test_data = pd.read_csv(
    TEST_PATH,
    usecols=test_usecols,
    dtype=dtype_test,
    parse_dates=["pickup_datetime"],
)

train_data.head(), test_data.head()



## === cell 3
train_data.shape




## === cell 4
def changeDataType(dataset):
    if not pd.api.types.is_datetime64_any_dtype(dataset["pickup_datetime"]):
        dataset["pickup_datetime"] = pd.to_datetime(
            dataset["pickup_datetime"], errors="coerce", utc=True
        )
    else:
        try:
            if dataset["pickup_datetime"].dt.tz is None:
                dataset["pickup_datetime"] = dataset["pickup_datetime"].dt.tz_localize(
                    "UTC"
                )
        except Exception:
            dataset["pickup_datetime"] = pd.to_datetime(
                dataset["pickup_datetime"], errors="coerce", utc=True
            )
    return dataset


train_data = changeDataType(train_data)
test_data = changeDataType(test_data)



## === cell 5
train_data["pickup_datetime"].head()



## === cell 6
train_data.isnull().sum()



## === cell 7
train_data = train_data.dropna(axis=0)



## === cell 8
pd.set_option("float_format", "{:f}".format)
train_data.shape



## === cell 9
if DO_PLOTS:
    plt.figure(figsize=(8, 5), dpi=80)
    sns.distplot(train_data["fare_amount"], color="red", kde=False)
train_data = train_data.loc[train_data["fare_amount"] > 0]
train_data["fare_amount"].head()



## === cell 10
if DO_PLOTS:
    sns.distplot(a=train_data.fare_amount, kde=False)



## === cell 11
pass



## === cell 12
train_data = train_data[train_data.fare_amount < 400]



## === cell 13
if DO_PLOTS:
    sns.kdeplot(data=train_data.fare_amount)



## === cell 14
if DO_PLOTS:
    sns.countplot(x=train_data.passenger_count)



## === cell 15
train_data = train_data[train_data.passenger_count <= 6]



## === cell 16
if DO_PLOTS:
    sns.distplot(a=train_data.passenger_count, kde=False)



## === cell 17
train_data.shape



## === cell 18
mask_valid_geo = (
    train_data["pickup_latitude"].between(-90, 90)
    & train_data["dropoff_latitude"].between(-90, 90)
    & train_data["pickup_longitude"].between(-180, 180)
    & train_data["dropoff_longitude"].between(-180, 180)
)
train_data = train_data.loc[mask_valid_geo]



## === cell 19
t_plat_min, t_plat_max = float(test_data.pickup_latitude.min()), float(
    test_data.pickup_latitude.max()
)
t_plon_min, t_plon_max = float(test_data.pickup_longitude.min()), float(
    test_data.pickup_longitude.max()
)
t_dlat_min, t_dlat_max = float(test_data.dropoff_latitude.min()), float(
    test_data.dropoff_latitude.max()
)
t_dlon_min, t_dlon_max = float(test_data.dropoff_longitude.min()), float(
    test_data.dropoff_longitude.max()
)

mask_in_test_bbox = (
    train_data.pickup_latitude.between(t_plat_min, t_plat_max)
    & train_data.pickup_longitude.between(t_plon_min, t_plon_max)
    & train_data.dropoff_latitude.between(t_dlat_min, t_dlat_max)
    & train_data.dropoff_longitude.between(t_dlon_min, t_dlon_max)
)
train_data = train_data.loc[mask_in_test_bbox]



## === cell 20
if DO_PLOTS:
    sns.scatterplot(x=train_data.pickup_latitude, y=train_data.pickup_longitude)
    sns.scatterplot(x=train_data.dropoff_latitude, y=train_data.dropoff_longitude)




## === cell 21
def degree_to_radion(degree):
    return degree * (np.pi / 180.0)


def calculate_distance(
    pickup_latitude, pickup_longitude, dropoff_latitude, dropoff_longitude
):
    from_lat = degree_to_radion(pickup_latitude)
    from_long = degree_to_radion(pickup_longitude)
    to_lat = degree_to_radion(dropoff_latitude)
    to_long = degree_to_radion(dropoff_longitude)

    radius = 6371.01  # km
    lat_diff = to_lat - from_lat
    long_diff = to_long - from_long

    a = (
        np.sin(lat_diff / 2) ** 2
        + np.cos(from_lat) * np.cos(to_lat) * np.sin(long_diff / 2) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return radius * c




## === cell 22
train_data["distance"] = calculate_distance(
    train_data.pickup_latitude,
    train_data.pickup_longitude,
    train_data.dropoff_latitude,
    train_data.dropoff_longitude,
)



## === cell 23
train_data.shape



## === cell 24
test_data["distance"] = calculate_distance(
    test_data.pickup_latitude,
    test_data.pickup_longitude,
    test_data.dropoff_latitude,
    test_data.dropoff_longitude,
)



## === cell 25
test_data.shape



## === cell 26
pass



## === cell 27
train_data = train_data.loc[train_data.distance < 200]



## === cell 28
train_data.shape



## === cell 29
if DO_PLOTS:
    sns.distplot(train_data.distance, kde=False)



## === cell 30
train_data = train_data.drop(columns="key")



## === cell 31
train_data.shape



## === cell 32
test_data_key = test_data["key"]
test_data = test_data.drop(columns="key")



## === cell 33
test_data.head()



## === cell 34
dt_tr = train_data["pickup_datetime"].dt
train_data["Year"] = dt_tr.year.astype("int16", copy=False)
train_data["Month"] = dt_tr.month.astype("int8", copy=False)
train_data["Date"] = dt_tr.day.astype("int8", copy=False)
train_data["Day of Week"] = dt_tr.dayofweek.astype("int8", copy=False)
train_data["Hour"] = dt_tr.hour.astype("int8", copy=False)

dt_te = test_data["pickup_datetime"].dt
test_data["Year"] = dt_te.year.astype("int16", copy=False)
test_data["Month"] = dt_te.month.astype("int8", copy=False)
test_data["Date"] = dt_te.day.astype("int8", copy=False)
test_data["Day of Week"] = dt_te.dayofweek.astype("int8", copy=False)
test_data["Hour"] = dt_te.hour.astype("int8", copy=False)



## === cell 35
train_data.head()



## === cell 36
test_data.head()



## === cell 37
if DO_PLOTS:
    sns.scatterplot(x=train_data["passenger_count"], y=train_data["fare_amount"])



## === cell 38
if DO_PLOTS:
    sns.scatterplot(x=train_data["distance"], y=train_data["fare_amount"])



## === cell 39
if DO_PLOTS:
    g = sns.FacetGrid(train_data, col="Year")
    g.map(sns.scatterplot, "distance", "fare_amount")



## === cell 40
train_data.shape



## === cell 41
pass



## === cell 42
if DO_PLOTS:
    sns.scatterplot(x=train_data["Year"], y=train_data["fare_amount"])



## === cell 43
pass



## === cell 44
if DO_PLOTS:
    sns.scatterplot(
        x=train_data["Month"], y=train_data["fare_amount"], hue=train_data["Year"]
    )



## === cell 45
if DO_PLOTS:
    sns.scatterplot(x=train_data["Month"], y=train_data["fare_amount"])



## === cell 46
if DO_PLOTS:
    w = sns.FacetGrid(train_data, col="Year")
    w.map(sns.scatterplot, "Month", "fare_amount")



## === cell 47
if DO_PLOTS:
    sns.barplot(x=train_data["Day of Week"], y=train_data["fare_amount"])



## === cell 48
if DO_PLOTS:
    plt.figure(figsize=(10, 10), dpi=150)
    w = sns.FacetGrid(train_data, col="Month")
    w.map(sns.barplot, "Day of Week", "fare_amount")



## === cell 49
if DO_PLOTS:
    sns.barplot(x=train_data["Hour"], y=train_data["fare_amount"])



## === cell 50
mask_nonzero_geo = (
    (train_data.pickup_latitude != 0)
    & (train_data.pickup_longitude != 0)
    & (train_data.dropoff_latitude != 0)
    & (train_data.dropoff_longitude != 0)
)
train_data = train_data.loc[mask_nonzero_geo]



## === cell 51
cols_geo = [
    "pickup_latitude",
    "pickup_longitude",
    "dropoff_latitude",
    "dropoff_longitude",
]
test_data[cols_geo] = test_data[cols_geo].replace(0, np.nan)
test_data[cols_geo] = test_data[cols_geo].fillna(test_data[cols_geo].median())



## === cell 52
train_data = train_data.drop(columns="pickup_datetime", axis=1)
test_data = test_data.drop(columns="pickup_datetime", axis=1)



## === cell 53
X = train_data.loc[:, train_data.columns != "fare_amount"]
y = train_data["fare_amount"]



## === cell 54
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)



## === cell 55
X_train.shape



## === cell 56
SKIP_EXPLORATORY_MODELS = True

if not SKIP_EXPLORATORY_MODELS:
    from sklearn.ensemble import RandomForestRegressor

    ran_for_reg = RandomForestRegressor(max_depth=50, random_state=0, n_jobs=-1)
    ran_for_reg.fit(X_train, y_train)
    y_ranfor_pred = ran_for_reg.predict(X_test)
    error = np.sqrt(mean_squared_error(y_test, y_ranfor_pred))
    error
else:
    error = None
error



## === cell 57
if DO_PLOTS and (not SKIP_EXPLORATORY_MODELS):
    sns.barplot(x=ran_for_reg.feature_importances_, y=X_test.columns)



## === cell 58
if not SKIP_EXPLORATORY_MODELS:
    from sklearn.ensemble import BaggingRegressor
    from sklearn.tree import DecisionTreeRegressor

    bagreg = BaggingRegressor(
        base_estimator=DecisionTreeRegressor(random_state=0),
        n_estimators=10,
        bootstrap=True,
        random_state=0,
        n_jobs=-1,
    )
    bagreg.fit(X_train, y_train)
    y_bagg_pred = bagreg.predict(X_test)
    error = np.sqrt(mean_squared_error(y_test, y_bagg_pred))
    error
else:
    error = None
error



## === cell 59
if not SKIP_EXPLORATORY_MODELS:
    from sklearn.ensemble import AdaBoostRegressor
    from sklearn.tree import DecisionTreeRegressor

    adareg = AdaBoostRegressor(DecisionTreeRegressor(random_state=0), random_state=0)
    adareg.fit(X_train, y_train)
    y_adareg_pred = adareg.predict(X_test)
    error = np.sqrt(mean_squared_error(y_test, y_adareg_pred))
    error
else:
    error = None
error



## === cell 60
from sklearn.ensemble import GradientBoostingRegressor

gradient_reg = GradientBoostingRegressor(random_state=0)
if not SKIP_EXPLORATORY_MODELS:
    gradient_reg.fit(X_train, y_train)
    y_gradient_pred = gradient_reg.predict(X_test)
    error = np.sqrt(mean_squared_error(y_test, y_gradient_pred))
    error
else:
    error = None
error



## === cell 61
from xgboost import XGBRegressor

xgreg = XGBRegressor(
    random_state=0,
    n_estimators=200,
    n_jobs=-1,
    verbosity=0,
    tree_method="hist",
)
if not SKIP_EXPLORATORY_MODELS:
    xgreg.fit(X_train, y_train)
    y_xgreg_pred = xgreg.predict(X_test)
    error = np.sqrt(mean_squared_error(y_test, y_xgreg_pred))
    error
else:
    error = None
error



## === cell 62
import lightgbm as lgb

model_lgb = lgb.LGBMRegressor(random_state=0, n_jobs=-1, verbose=-1)
if not SKIP_EXPLORATORY_MODELS:
    model_lgb.fit(X_train, y_train)
    y_lgb_pred = model_lgb.predict(X_test)
    error = np.sqrt(mean_squared_error(y_test, y_lgb_pred))
    error
else:
    error = None
error



## === cell 63
from sklearn.model_selection import KFold
from sklearn.base import BaseEstimator

n_folds = 5


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




## === cell 64
average_model = AverageModel(models=(gradient_reg, xgreg, model_lgb))
average_model  # keep object creation (core logic continuity)



## === cell 65
from sklearn.base import clone

base_model = [gradient_reg, xgreg, model_lgb]


def make_oof_predictions(X_full, y_full, models, n_splits=5, random_state=156):
    from joblib import Parallel, delayed

    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    Xv = np.ascontiguousarray(X_full.to_numpy(copy=False), dtype=np.float32)
    yv = np.ascontiguousarray(y_full.to_numpy(copy=False), dtype=np.float32)

    splits = list(kfold.split(Xv, yv))  # deterministic; reused across models
    n_models = len(models)

    def _fit_predict_one_model(i_model, base):
        oof_col = np.zeros(Xv.shape[0], dtype=np.float32)
        for tr_idx, ho_idx in splits:
            X_tr, y_tr = Xv[tr_idx], yv[tr_idx]
            X_ho = Xv[ho_idx]
            m = clone(base)
            m.fit(X_tr, y_tr)
            oof_col[ho_idx] = np.asarray(m.predict(X_ho), dtype=np.float32)
        return i_model, oof_col

    results = Parallel(n_jobs=min(n_models, 3), prefer="processes")(
        delayed(_fit_predict_one_model)(i, m) for i, m in enumerate(models)
    )

    oof = np.zeros((Xv.shape[0], n_models), dtype=np.float32)
    for i_model, col in results:
        oof[:, i_model] = col
    return oof


out_of_fold_predictions_full = make_oof_predictions(
    X, y, base_model, n_splits=5, random_state=156
)
out_of_fold_predictions_full.shape



## === cell 66
oof_avg = out_of_fold_predictions_full.mean(axis=1)
y_np = y.to_numpy(copy=False)
score = np.sqrt(mean_squared_error(y_np, oof_avg))
np.array([score], dtype=np.float64)



## === cell 67
score.mean()



## === cell 68
meta_model = lgb.LGBMRegressor(random_state=0, n_jobs=-1, verbose=-1)
meta_model.fit(out_of_fold_predictions_full, y)



## === cell 69
from sklearn.base import clone

X_np = np.ascontiguousarray(X.to_numpy(copy=False), dtype=np.float32)
y_np = np.ascontiguousarray(y.to_numpy(copy=False), dtype=np.float32)
test_np = np.ascontiguousarray(test_data.to_numpy(copy=False), dtype=np.float32)

gradient_full = clone(gradient_reg).fit(X_np, y_np)
xgb_full = clone(xgreg).fit(X_np, y_np)
lgb_full = clone(model_lgb).fit(X_np, y_np)
base_model_fitted = [gradient_full, xgb_full, lgb_full]

feature_data = np.column_stack([m.predict(test_np) for m in base_model_fitted]).astype(
    np.float32,
    copy=False,
)
meta_y = meta_model.predict(feature_data)

submission = pd.DataFrame(
    {"key": test_data_key.values, "fare_amount": meta_y.astype(np.float32, copy=False)}
)
submission.to_csv("submission.csv", index=False)

submission.head(), submission.shape
