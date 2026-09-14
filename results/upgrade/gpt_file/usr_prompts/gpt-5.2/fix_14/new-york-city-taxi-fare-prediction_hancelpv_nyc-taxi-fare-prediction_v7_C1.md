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

3.89954

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 4.85034) has done: 'I fix the datetime feature extraction to work with your installed pandas version (the `.dt.weekday_name` attribute was removed), and make sure the engineered features actually exist before selecting them. Then I ensure the model only receives numeric columns by one-hot encoding categoricals and aligning train/test matrices so `RandomForestRegressor` can fit/predict without string-conversion errors. Finally, I guarantee a valid `submission.csv` is written with exactly the required `key,fare_amount` columns and row count matching the test set.'
- What this solution (achieved 6.54046) has done: 'Your current score (4.85034 RMSE) is worse than the target (3.89954), so we should improve performance with minimal, low-risk changes that keep the same core model and feature pipeline. The biggest issue is that the model is trained on 1M raw rows without filtering obvious bad/noisy NYC taxi records (invalid coordinates, zero-distance rides, extreme fares), which typically hurts RMSE a lot; adding standard sanity filters improves signal without changing the modeling approach. I add a small set of deterministic row filters and also ensure train/test one-hot columns are aligned explicitly (to avoid any subtle mismatch), while keeping the RandomForestRegressor and feature engineering intact. This should move the score down toward the target band without introducing new training tricks or changing the core logic.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O
import os

os.environ.setdefault("PYTHONHASHSEED", "42")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

np.random.seed(42)

print("Listing ../input:")
try:
    print(os.listdir("../input"))
except FileNotFoundError:
    print("WARNING: ../input not found. Available root:", os.listdir("/"))



## === cell 1
train_cols = [
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

train_dtypes = {
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int16",
}
test_dtypes = {
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int16",
}

train = pd.read_csv(
    "../input/train.csv", nrows=1_000_000, usecols=train_cols, dtype=train_dtypes
)
test = pd.read_csv("../input/test.csv", usecols=test_cols, dtype=test_dtypes)
samp = pd.read_csv("../input/sample_submission.csv")

print(
    "train shape:", train.shape, "test shape:", test.shape, "sample shape:", samp.shape
)



## === cell 2
train = train.dropna(how="any", axis="rows")
print("train shape after dropna:", train.shape)



## === cell 3
test.shape




## === cell 4
def clean_train_rows(df):
    mask = (
        (df["fare_amount"] > 0)
        & (df["fare_amount"] <= 250)
        & (df["passenger_count"] >= 1)
        & (df["passenger_count"] <= 6)
        & (df["pickup_longitude"].between(-74.5, -72.8))
        & (df["dropoff_longitude"].between(-74.5, -72.8))
        & (df["pickup_latitude"].between(40.5, 41.8))
        & (df["dropoff_latitude"].between(40.5, 41.8))
        & ~(
            (df["pickup_longitude"] == df["dropoff_longitude"])
            & (df["pickup_latitude"] == df["dropoff_latitude"])
        )
    )
    return df.loc[mask].reset_index(drop=True)


train = clean_train_rows(train)
print("train shape after clean_train_rows:", train.shape)



## === cell 5
y = train["fare_amount"].to_numpy()
n_train = len(train)
n_test = len(test)
test_id = test["key"]

train_X = train.drop(columns=["fare_amount"])
test_X = test.drop(columns=["key"])
all_data = pd.concat((train_X, test_X), axis=0, ignore_index=True)

print("all_data shape:", all_data.shape, "n_train:", n_train, "n_test:", n_test)




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
    dt = pd.to_datetime(data["pickup_datetime"], errors="coerce", cache=True)

    out = data.drop(columns=["pickup_datetime"], errors="ignore")

    out["hour"] = dt.dt.hour.astype("int16")
    out["day_of_week"] = dt.dt.weekday.astype("int16")

    dom = dt.dt.day.astype("int16")
    wom = pd.cut(
        dom,
        bins=[0, 7, 14, 21, 28, 31],
        labels=[0, 1, 2, 3, 4],
        include_lowest=True,
        right=True,
    )
    out["week_of_month"] = wom.astype("int16")

    out["month"] = dt.dt.month.astype("int16")
    out["year"] = dt.dt.year.astype("int16")
    return out




## === cell 8
def add_geo_features(data):
    p_long = data["pickup_longitude"].to_numpy()
    d_long = data["dropoff_longitude"].to_numpy()
    p_lat = data["pickup_latitude"].to_numpy()
    d_lat = data["dropoff_latitude"].to_numpy()

    abs_diff_long = np.abs(d_long - p_long)
    abs_diff_lat = np.abs(d_lat - p_lat)

    data["abs_diff_longitude"] = abs_diff_long
    data["abs_diff_latitude"] = abs_diff_lat
    manhattan = abs_diff_long + abs_diff_lat
    data["manhattan_distance"] = manhattan

    squared_long = abs_diff_long * abs_diff_long
    squared_lat = abs_diff_lat * abs_diff_lat
    data["squared_long"] = squared_long
    data["squared_lat"] = squared_lat
    data["euclid_disance"] = np.sqrt(squared_long + squared_lat)

    return data




## === cell 9
all_data = add_time_features(all_data)
all_data = add_geo_features(all_data)

drop_cols = []
for c in ["key"]:
    if c in all_data.columns:
        drop_cols.append(c)
if drop_cols:
    all_data = all_data.drop(columns=drop_cols)

print("columns after feature eng:", all_data.columns.tolist())



## === cell 10
features = [
    "passenger_count",
    "hour",
    "day_of_week",
    "week_of_month",
    "month",
    "year",
    "abs_diff_longitude",
    "abs_diff_latitude",
    "euclid_disance",
    "manhattan_distance",
]

missing = [c for c in features if c not in all_data.columns]
if missing:
    raise KeyError(f"Missing engineered feature columns: {missing}")

all_data = all_data[features]

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

cat_cols = ["hour", "day_of_week", "week_of_month", "month", "year"]
num_cols = [c for c in features if c not in cat_cols]

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ("num", "passthrough", num_cols),
    ],
    sparse_threshold=0.0,  # ensure dense
)

X_all = all_data  # keep as DataFrame for ColumnTransformer column selection
X_train = X_all.iloc[:n_train]
X_test = X_all.iloc[n_train:]

print(
    "Prepared split: X_train shape:",
    X_train.shape,
    "X_test shape:",
    X_test.shape,
    "y shape:",
    y.shape,
)



## === cell 11
try:
    from sklearnex import patch_sklearn

    patch_sklearn()
    print("Applied sklearnex patch for speed.")
except Exception as e:
    print("sklearnex patch not applied (continuing):", repr(e))

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(random_state=42, n_jobs=1, warm_start=False)



## === cell 12
params = {
    "bootstrap": [True],
    "max_depth": [80, 90, 100, 110],
    "max_features": [2, 3],
    "min_samples_leaf": [3, 4, 5],
    "min_samples_split": [8, 10, 12],
    "n_estimators": [100, 200, 300, 1000],
}



## === cell 13
import tempfile
import joblib
import shutil

X_train_enc = preprocess.fit_transform(X_train)
X_test_enc = preprocess.transform(X_test)

X_train_enc = np.asarray(X_train_enc, dtype=np.float32, order="C")
X_test_enc = np.asarray(X_test_enc, dtype=np.float32, order="C")

print(
    "Encoded shapes: X_train_enc:",
    getattr(X_train_enc, "shape", None),
    "X_test_enc:",
    getattr(X_test_enc, "shape", None),
)

from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.model_selection import HalvingRandomSearchCV

search = HalvingRandomSearchCV(
    estimator=model,
    param_distributions=params,
    scoring="neg_root_mean_squared_error",
    cv=3,
    random_state=42,
    n_jobs=-1,
    verbose=1,
    refit=True,
    resource="n_estimators",
    min_resources=100,
    max_resources=1000,
    factor=3,
    aggressive_elimination=False,
)

tmpdir = tempfile.mkdtemp(prefix="joblib_memmap_")
try:
    with joblib.parallel_config(temp_folder=tmpdir, max_nbytes="128M"):
        search.fit(X_train_enc, y)
finally:
    try:
        shutil.rmtree(tmpdir, ignore_errors=True)
    except Exception:
        pass

print("Best CV RMSE:", -float(search.best_score_))
print("Best params:", search.best_params_)

best_model = search.best_estimator_



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1275025929.py in <cell line: 0>()
     46 try:
     47     with joblib.parallel_config(temp_folder=tmpdir, max_nbytes="128M"):
---> 48         search.fit(X_train_enc, y)
     49 finally:
     50     try:

/usr/local/lib/python3.11/dist-packages/sklearn/model_selection/_search_successive_halving.py in fit(self, X, y, groups, **fit_params)
    271         self._n_samples_orig = _num_samples(X)
    272 
--> 273         super().fit(X, y=y, groups=groups, **fit_params)
    274 
    275         # Set best_score_: BaseSearchCV does not set it, as refit is a callable

/usr/local/lib/python3.11/dist-packages/sklearn/model_selection/_search.py in fit(self, X, y, groups, **fit_params)
    872                 return results
    873 
--> 874             self._run_search(evaluate_candidates)
    875 
    876             # multimetric is determined here because in the case of a callable

/usr/local/lib/python3.11/dist-packages/sklearn/model_selection/_search_successive_halving.py in _run_search(self, evaluate_candidates)
    285         ):
    286             # Can only check this now since we need the candidates list
--> 287             raise ValueError(
    288                 f"Cannot use parameter {self.resource} as the resource since "
    289                 "it is part of the searched parameters."

ValueError: Cannot use parameter n_estimators as the resource since it is part of the searched parameters.

## === cell 14
test_pred = best_model.predict(X_test_enc)

test_pred = np.maximum(test_pred, 0)

print(
    "Pred stats:",
    float(np.min(test_pred)),
    float(np.mean(test_pred)),
    float(np.max(test_pred)),
)



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2432356339.py in <cell line: 0>()
----> 1 test_pred = best_model.predict(X_test_enc)
      2 
      3 test_pred = np.maximum(test_pred, 0)
      4 
      5 print(

NameError: name 'best_model' is not defined

## === cell 15
sub = pd.DataFrame({"key": test_id.values, "fare_amount": test_pred})
assert sub.shape[0] == test.shape[0], "Submission rows do not match test rows"
sub.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())

## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2757507452.py in <cell line: 0>()
----> 1 sub = pd.DataFrame({"key": test_id.values, "fare_amount": test_pred})
      2 assert sub.shape[0] == test.shape[0], "Submission rows do not match test rows"
      3 sub.to_csv("submission.csv", index=False)
      4 
      5 print("Wrote submission.csv with shape:", sub.shape)

NameError: name 'test_pred' is not defined
