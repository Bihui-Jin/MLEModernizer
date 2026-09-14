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

3.14

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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
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

4.736901452993294

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

- What this solution (achieved 108.82752) has done: 'I drop the non‑numeric `pickup_datetime` column before converting the training features to a NumPy array, ensuring the models can be fitted. This fixes the conversion error and the subsequent “not fitted” errors, allowing the script to run end‑to‑end and produce a valid `submission.csv` file.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os
import gc
import warnings
from sklearnex import patch_sklearn

warnings.filterwarnings("ignore")
patch_sklearn()
np.random.seed(2)  # ensure reproducibility for any stochastic ops




## === cell 1
dtype_train = {
    "fare_amount": "float32",
    "pickup_datetime": "object",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int8",
}
usecols_train = [
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
df_train = pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv",
    nrows=2_000_000,
    dtype=dtype_train,
    usecols=usecols_train,
)
df_train.head()




## === cell 2
df_train = df_train.iloc[:, :]




## === cell 3
print(df_train.shape)
print(df_train.isna().sum())




## === cell 4
mask = (
    df_train["fare_amount"].between(1, 100)
    & df_train["passenger_count"].between(1, 6)
    & df_train["pickup_longitude"].between(-75, -72)
    & df_train["dropoff_longitude"].between(-75, -72)
    & df_train["pickup_latitude"].between(40, 42)
    & df_train["dropoff_latitude"].between(40, 42)
)
df_train = df_train.dropna().loc[mask].reset_index(drop=True)

df_train["pickup_datetime"] = pd.to_datetime(df_train["pickup_datetime"])
df_train["hour"] = df_train["pickup_datetime"].dt.hour.astype(np.int8)
df_train["weekday"] = df_train["pickup_datetime"].dt.weekday.astype(np.int8)
df_train["month"] = df_train["pickup_datetime"].dt.month.astype(np.int8)




## === cell 5
def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371 * c  # km




## === cell 6
df_train["distance_km"] = haversine(
    df_train["pickup_latitude"],
    df_train["pickup_longitude"],
    df_train["dropoff_latitude"],
    df_train["dropoff_longitude"],
)
df_train = df_train[df_train["distance_km"].between(0.1, 30)].reset_index(drop=True)




## === cell 7
y = df_train["fare_amount"].to_numpy(dtype=np.float32)
X = df_train.drop(columns=["pickup_datetime"]).astype(np.float32).to_numpy()




## === cell 8
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=None,
    random_state=2,
    bootstrap=True,
    max_samples=None,
    n_jobs=-1,
)




## === cell 9
from sklearn.linear_model import LinearRegression

lr = LinearRegression()




## === cell 10
from sklearn.svm import SVR
from sklearn.ensemble import BaggingRegressor

sr = SVR(kernel="rbf")
br = BaggingRegressor(
    base_estimator=sr,
    n_estimators=10,
    max_samples=5000,
    bootstrap=True,
    n_jobs=-1,
    verbose=0,
)




## === cell 11
rf.fit(X, y)
lr.fit(X, y)
br.fit(X, y)

del df_train, X, y
gc.collect()




## === cell 12
df_test = pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv",
    parse_dates=["pickup_datetime"],
)
key = df_test["key"]

df_test["hour"] = df_test["pickup_datetime"].dt.hour.astype(np.int8)
df_test["weekday"] = df_test["pickup_datetime"].dt.weekday.astype(np.int8)
df_test["month"] = df_test["pickup_datetime"].dt.month.astype(np.int8)

df_test = df_test.iloc[:, 2:]  # drop key and pickup_datetime




## === cell 13
df_test["distance_km"] = haversine(
    df_test["pickup_latitude"],
    df_test["pickup_longitude"],
    df_test["dropoff_latitude"],
    df_test["dropoff_longitude"],
)




## === cell 14
X_test = df_test.astype(np.float32).to_numpy()

pred_lr = lr.predict(X_test)
pred_rf = rf.predict(X_test)
pred_br = br.predict(X_test)

y_pred = 0.1 * pred_lr + 0.8 * pred_rf + 0.1 * pred_br
y_pred = np.clip(y_pred, 0, None)




## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2330048709.py in <cell line: 0>()
      2 X_test = df_test.astype(np.float32).to_numpy()
      3 
----> 4 pred_lr = lr.predict(X_test)
      5 pred_rf = rf.predict(X_test)
      6 pred_br = br.predict(X_test)

/usr/local/lib/python3.11/dist-packages/daal4py/sklearn/_n_jobs_support.py in n_jobs_wrapper(self, *args, **kwargs)
    120         old_n_threads = get_n_threads()
    121         if n_jobs == old_n_threads:
--> 122             return method(self, *args, **kwargs)
    123 
    124         try:

/usr/local/lib/python3.11/dist-packages/sklearnex/_device_offload.py in wrapper(self, *args, **kwargs)
    180     @wraps(func)
    181     def wrapper(self, *args, **kwargs) -> Any:
--> 182         result = func(self, *args, **kwargs)
    183         if not (len(args) == 0 and len(kwargs) == 0):
    184             data = (*args, *kwargs.values())[0]

/usr/local/lib/python3.11/dist-packages/sklearnex/linear_model/linear.py in predict(self, X)
    129     def predict(self, X):
    130         check_is_fitted(self)
--> 131         return dispatch(
    132             self,
    133             "predict",

/usr/local/lib/python3.11/dist-packages/sklearnex/_device_offload.py in dispatch(obj, method_name, branches, *args, **kwargs)
    150             queue = QM.get_global_queue()
    151             patching_status.write_log(queue=queue, transferred_to_host=False)
--> 152             return branches["onedal"](obj, *hostargs, **hostkwargs, queue=queue)
    153         else:
    154             if sklearn_array_api and not has_usm_data:

/usr/local/lib/python3.11/dist-packages/sklearnex/linear_model/linear.py in _onedal_predict(self, X, queue)
    309 
    310         if not get_config()["use_raw_input"]:
--> 311             X = validate_data(
    312                 self, X, accept_sparse=False, dtype=[xp.float64, xp.float32], reset=False
    313             )

/usr/local/lib/python3.11/dist-packages/daal4py/sklearn/utils/validation.py in new_function(*args, **kwargs)
    738         )
    739         def new_function(*args, **kwargs):
--> 740             return dispatcher_function(*args, **kwargs)
    741 
    742         new_function.__doc__ = (

/usr/local/lib/python3.11/dist-packages/sklearnex/utils/validation.py in validate_data(_estimator, X, y, **kwargs)
    112     kwargs[_finite_keyword] = False
    113 
--> 114     out = _sklearn_validate_data(
    115         _estimator,
    116         X=X,

/usr/local/lib/python3.11/dist-packages/sklearn/base.py in _validate_data(self, X, y, reset, validate_separately, **check_params)
    586 
    587         if not no_val_X and check_params.get("ensure_2d", True):
--> 588             self._check_n_features(X, reset=reset)
    589 
    590         return out

/usr/local/lib/python3.11/dist-packages/sklearn/base.py in _check_n_features(self, X, reset)
    387 
    388         if n_features != self.n_features_in_:
--> 389             raise ValueError(
    390                 f"X has {n_features} features, but {self.__class__.__name__} "
    391                 f"is expecting {self.n_features_in_} features as input."

ValueError: X has 9 features, but LinearRegression is expecting 10 features as input.

## === cell 15
results = pd.DataFrame({"key": key, "fare_amount": y_pred})
print(results.head())
results.to_csv("submission.csv", index=False)

## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3964246742.py in <cell line: 0>()
----> 1 results = pd.DataFrame({"key": key, "fare_amount": y_pred})
      2 print(results.head())
      3 results.to_csv("submission.csv", index=False)

NameError: name 'y_pred' is not defined
