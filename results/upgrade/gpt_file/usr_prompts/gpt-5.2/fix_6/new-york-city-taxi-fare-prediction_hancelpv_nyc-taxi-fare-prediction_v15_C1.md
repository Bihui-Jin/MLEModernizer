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

3.94377

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

- What this solution (achieved 5.29223) has done: 'I fix the pandas datetime feature extraction by replacing the removed `weekday_name` accessor with `day_name()` and ensuring the datetime conversion doesn’t introduce timezone-aware dtypes. Then I make sure the model inputs contain only numeric columns by selecting the engineered feature set and one-hot encoding, which prevents `Timestamp`/`datetime64` dtypes from reaching scikit-learn/Keras. I also fix variable naming bugs (`model` vs `model_1`) and the Keras import issue in this environment by using `tf_keras` consistently. Finally, I ensure both models can run end-to-end and that at least one valid `.csv` submission file with the required `key,fare_amount` columns is written.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import math
import os

try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except Exception:
    pass

print(os.listdir("../input"))

np.random.seed(42)



## === cell 1
types = {
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "uint8",
}

cols = [
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]



## === cell 2
train = pd.read_csv(
    "../input/train.csv",
    nrows=1_000_000,
    usecols=cols,
    dtype=types,
    parse_dates=["pickup_datetime"],
    infer_datetime_format=True,
    low_memory=False,
)
test = pd.read_csv(
    "../input/test.csv",
    usecols=[
        "key",
        "pickup_datetime",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
    ],
    dtype={
        "pickup_longitude": "float32",
        "pickup_latitude": "float32",
        "dropoff_longitude": "float32",
        "dropoff_latitude": "float32",
        "passenger_count": "uint8",
    },
    parse_dates=["pickup_datetime"],
    infer_datetime_format=True,
    low_memory=False,
)
samp = pd.read_csv("../input/sample_submission.csv")



## === cell 3
train.dropna(how="any", axis="rows", inplace=True)
train = train[(train.fare_amount > 0) & (train["passenger_count"] <= 6)]



## === cell 4
mask = (
    (train.pickup_latitude > -90)
    & (train.pickup_latitude < 90)
    & (train.dropoff_latitude > -90)
    & (train.dropoff_latitude < 90)
    & (train.pickup_longitude > -180)
    & (train.pickup_longitude < 180)
    & (train.dropoff_longitude > -180)
    & (train.dropoff_longitude < 180)
)
train = train[mask]



## === cell 5
y = train.fare_amount.values
n_train = len(train)
n_test = len(test)
test_id = test.key

train_feat = train.drop(["fare_amount"], axis=1)
all_data = pd.concat(
    (train_feat, test.drop(columns=["key"])),
    axis=0,
    ignore_index=True,
    copy=False,
)




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
    dt = data["pickup_datetime"]
    if not np.issubdtype(dt.dtype, np.datetime64):
        dt = pd.to_datetime(dt, errors="coerce", utc=False)
        data["pickup_datetime"] = dt

    data["hour"] = dt.dt.hour.astype("Int16")
    data["day_of_week"] = dt.dt.day_name()

    dom = dt.dt.day
    bins = [0, 7, 14, 21, 28, 31]
    labels = ["first", "second", "third", "fourth", "fifth"]
    data["week_of_month"] = pd.cut(
        dom, bins=bins, labels=labels, include_lowest=True, right=True
    )

    data["month"] = dt.dt.month.astype("Int16")
    data["year"] = dt.dt.year.astype("Int16")

    data["hour"] = data["hour"].astype("category")
    data["month"] = data["month"].astype("category")
    data["year"] = data["year"].astype("category")
    data["day_of_week"] = data["day_of_week"].astype("category")
    data["week_of_month"] = data["week_of_month"].astype("category")

    return data




## === cell 8
def add_geo_features(data):
    plon = data["pickup_longitude"].astype(np.float32, copy=False)
    plat = data["pickup_latitude"].astype(np.float32, copy=False)
    dlon = data["dropoff_longitude"].astype(np.float32, copy=False)
    dlat = data["dropoff_latitude"].astype(np.float32, copy=False)

    abs_diff_long = (dlon - plon).abs()
    abs_diff_lat = (dlat - plat).abs()
    data["abs_diff_longitude"] = abs_diff_long
    data["abs_diff_latitude"] = abs_diff_lat

    manhattan = abs_diff_long + abs_diff_lat
    data["manhattan_distance"] = manhattan

    squared_long = np.square(abs_diff_long.to_numpy(dtype=np.float32, copy=False))
    squared_lat = np.square(abs_diff_lat.to_numpy(dtype=np.float32, copy=False))
    data["squared_long"] = squared_long
    data["squared_lat"] = squared_lat

    data["euclid_distance"] = np.sqrt(squared_long + squared_lat).astype(
        np.float32, copy=False
    )
    return data




## === cell 9
all_data = add_time_features(all_data)
all_data = add_geo_features(all_data)

if "pickup_datetime" in all_data.columns:
    all_data.drop(["pickup_datetime"], axis=1, inplace=True)



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/951134752.py in <cell line: 0>()
----> 1 all_data = add_time_features(all_data)
      2 all_data = add_geo_features(all_data)
      3 
      4 if "pickup_datetime" in all_data.columns:
      5     all_data.drop(["pickup_datetime"], axis=1, inplace=True)

/tmp/ipykernel_11/3267061766.py in add_time_features(data)
      3 def add_time_features(data):
      4     dt = data["pickup_datetime"]
----> 5     if not np.issubdtype(dt.dtype, np.datetime64):
      6         dt = pd.to_datetime(dt, errors="coerce", utc=False)
      7         data["pickup_datetime"] = dt

/usr/local/lib/python3.11/dist-packages/numpy/core/numerictypes.py in issubdtype(arg1, arg2)
    415     """
    416     if not issubclass_(arg1, generic):
--> 417         arg1 = dtype(arg1).type
    418     if not issubclass_(arg2, generic):
    419         arg2 = dtype(arg2).type

TypeError: Cannot interpret 'datetime64[ns, UTC]' as a data type

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
    "manhattan_distance",
    "euclid_distance",
]

all_data = all_data[features]

all_data["hour"] = all_data["hour"].cat.set_categories(list(range(24)))
all_data["month"] = all_data["month"].cat.set_categories(list(range(1, 13)))
all_data["day_of_week"] = all_data["day_of_week"].cat.set_categories(
    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
)
all_data["week_of_month"] = all_data["week_of_month"].cat.set_categories(
    ["first", "second", "third", "fourth", "fifth"]
)
all_data["year"] = all_data["year"].cat.set_categories(
    sorted(all_data["year"].cat.categories)
)

all_data = pd.get_dummies(all_data, sparse=True)
all_data = all_data.fillna(0)



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/3453877888.py in <cell line: 0>()
     12 ]
     13 
---> 14 all_data = all_data[features]
     15 
     16 # Speed: pre-set known finite category sets so get_dummies doesn't scan all values to discover categories.

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4106             if is_iterator(key):
   4107                 key = list(key)
-> 4108             indexer = self.columns._get_indexer_strict(key, "columns")[1]
   4109 
   4110         # take() does not accept boolean indexers

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _get_indexer_strict(self, key, axis_name)
   6198             keyarr, indexer, new_indexer = self._reindex_non_unique(keyarr)
   6199 
-> 6200         self._raise_if_missing(keyarr, indexer, axis_name)
   6201 
   6202         keyarr = self.take(indexer)

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _raise_if_missing(self, key, indexer, axis_name)
   6250 
   6251             not_found = list(ensure_index(key)[missing_mask.nonzero()[0]].unique())
-> 6252             raise KeyError(f"{not_found} not in index")
   6253 
   6254     @overload

KeyError: "['hour', 'day_of_week', 'week_of_month', 'month', 'year', 'abs_diff_longitude', 'abs_diff_latitude', 'manhattan_distance', 'euclid_distance'] not in index"

## === cell 11
x = all_data[:n_train]
x_test = all_data[n_train:]

from scipy import sparse

x = sparse.csr_matrix(x)
x_test = sparse.csr_matrix(x_test)



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/4037072207.py in <cell line: 0>()
      5 from scipy import sparse
      6 
----> 7 x = sparse.csr_matrix(x)
      8 x_test = sparse.csr_matrix(x_test)
      9 

/usr/local/lib/python3.11/dist-packages/scipy/sparse/_compressed.py in __init__(self, arg1, shape, dtype, copy, maxprint)
     92                 raise ValueError(f"CSR arrays don't yet support {arg1.ndim}D.")
     93 
---> 94             coo = self._coo_container(arg1, dtype=dtype)
     95             arrays = coo._coo_to_compressed(self._swap)
     96             self.indptr, self.indices, self.data, self._shape = arrays

/usr/local/lib/python3.11/dist-packages/scipy/sparse/_coo.py in __init__(self, arg1, shape, dtype, copy, maxprint)
     93                 self.coords = tuple(idx.astype(index_dtype, copy=False)
     94                                      for idx in coords)
---> 95                 self.data = getdata(M[coords], copy=copy, dtype=dtype)
     96                 self.has_canonical_format = True
     97 

/usr/local/lib/python3.11/dist-packages/scipy/sparse/_sputils.py in getdata(obj, dtype, copy)
    148     # Defer to getdtype for checking that the dtype is OK.
    149     # This is called for the validation only; we don't need the return value.
--> 150     getdtype(data.dtype)
    151     return data
    152 

/usr/local/lib/python3.11/dist-packages/scipy/sparse/_sputils.py in getdtype(dtype, a, default)
    135     if newdtype not in supported_dtypes:
    136         supported_dtypes_fmt = ", ".join(t.__name__ for t in supported_dtypes)
--> 137         raise ValueError(f"scipy.sparse does not support dtype {newdtype.name}. "
    138                          f"The only supported types are: {supported_dtypes_fmt}.")
    139     return newdtype

ValueError: scipy.sparse does not support dtype object. The only supported types are: bool_, int8, uint8, int16, uint16, int32, uint32, int64, uint64, longlong, ulonglong, float32, float64, longdouble, complex64, complex128, clongdouble.

## === cell 12
from sklearn.ensemble import RandomForestRegressor

model_1 = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1,
    min_samples_leaf=2,
    warm_start=True,
)



## === cell 13
prev = 0
for n_estimators in (100, 200, 300):
    if n_estimators <= prev:
        continue
    model_1.set_params(n_estimators=n_estimators)
    model_1.fit(x, y)
    prev = n_estimators

model_1_pred = model_1.predict(x_test)
model_1_pred = np.maximum(model_1_pred, 0.0)

sub_1 = pd.DataFrame({"key": test_id, "fare_amount": model_1_pred})
sub_1.to_csv("submission_rf.csv", index=False)

print("Wrote submission_rf.csv with shape:", sub_1.shape)



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1199089752.py in <cell line: 0>()
      7         continue
      8     model_1.set_params(n_estimators=n_estimators)
----> 9     model_1.fit(x, y)
     10     prev = n_estimators
     11 

/usr/local/lib/python3.11/dist-packages/daal4py/sklearn/_n_jobs_support.py in n_jobs_wrapper(self, *args, **kwargs)
    120         old_n_threads = get_n_threads()
    121         if n_jobs == old_n_threads:
--> 122             return method(self, *args, **kwargs)
    123 
    124         try:

/usr/local/lib/python3.11/dist-packages/sklearnex/ensemble/_forest.py in fit(self, X, y, sample_weight)
   1173 
   1174     def fit(self, X, y, sample_weight=None):
-> 1175         dispatch(
   1176             self,
   1177             "fit",

/usr/local/lib/python3.11/dist-packages/sklearnex/_device_offload.py in dispatch(obj, method_name, branches, *args, **kwargs)
    158             else:
    159                 patching_status.write_log()
--> 160                 return branches["sklearn"](obj, *hostargs, **hostkwargs)
    161 
    162 

/usr/local/lib/python3.11/dist-packages/sklearn/ensemble/_forest.py in fit(self, X, y, sample_weight)
    343         if issparse(y):
    344             raise ValueError("sparse multilabel-indicator for y is not supported.")
--> 345         X, y = self._validate_data(
    346             X, y, multi_output=True, accept_sparse="csc", dtype=DTYPE
    347         )

/usr/local/lib/python3.11/dist-packages/sklearn/base.py in _validate_data(self, X, y, reset, validate_separately, **check_params)
    582                 y = check_array(y, input_name="y", **check_y_params)
    583             else:
--> 584                 X, y = check_X_y(X, y, **check_params)
    585             out = X, y
    586 

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in check_X_y(X, y, accept_sparse, accept_large_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, multi_output, ensure_min_samples, ensure_min_features, y_numeric, estimator)
   1104         )
   1105 
-> 1106     X = check_array(
   1107         X,
   1108         accept_sparse=accept_sparse,

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in check_array(array, accept_sparse, accept_large_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, ensure_min_samples, ensure_min_features, estimator, input_name)
    877                     array = xp.astype(array, dtype, copy=False)
    878                 else:
--> 879                     array = _asarray_with_order(array, order=order, dtype=dtype, xp=xp)
    880             except ComplexWarning as complex_warning:
    881                 raise ValueError(

/usr/local/lib/python3.11/dist-packages/sklearn/utils/_array_api.py in _asarray_with_order(array, dtype, order, copy, xp)
    183     if xp.__name__ in {"numpy", "numpy.array_api"}:
    184         # Use NumPy API to support order
--> 185         array = numpy.asarray(array, order=order, dtype=dtype)
    186         return xp.asarray(array, copy=copy)
    187     else:

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in __array__(self, dtype, copy)
   2151     ) -> np.ndarray:
   2152         values = self._values
-> 2153         arr = np.asarray(values, dtype=dtype)
   2154         if (
   2155             astype_is_view(values.dtype, arr.dtype)

TypeError: float() argument must be a string or a real number, not 'Timestamp'

## === cell 14
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler



## === cell 15
num_features = x.shape[1]
print("Num features:", num_features)



## === cell 16
scaler = StandardScaler(with_mean=False)
x_scaled = scaler.fit_transform(x)
x_test_scaled = scaler.transform(x_test)



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1764502090.py in <cell line: 0>()
      1 # Speed: StandardScaler on sparse CSR is efficient; keep with_mean=False as before (required for sparse).
      2 scaler = StandardScaler(with_mean=False)
----> 3 x_scaled = scaler.fit_transform(x)
      4 x_test_scaled = scaler.transform(x_test)
      5 

/usr/local/lib/python3.11/dist-packages/sklearn/utils/_set_output.py in wrapped(self, X, *args, **kwargs)
    138     @wraps(f)
    139     def wrapped(self, X, *args, **kwargs):
--> 140         data_to_wrap = f(self, X, *args, **kwargs)
    141         if isinstance(data_to_wrap, tuple):
    142             # only wrap the first output for cross decomposition

/usr/local/lib/python3.11/dist-packages/sklearn/base.py in fit_transform(self, X, y, **fit_params)
    876         if y is None:
    877             # fit method of arity 1 (unsupervised transformation)
--> 878             return self.fit(X, **fit_params).transform(X)
    879         else:
    880             # fit method of arity 2 (supervised transformation)

/usr/local/lib/python3.11/dist-packages/sklearn/preprocessing/_data.py in fit(self, X, y, sample_weight)
    822         # Reset internal state before fitting
    823         self._reset()
--> 824         return self.partial_fit(X, y, sample_weight)
    825 
    826     def partial_fit(self, X, y=None, sample_weight=None):

/usr/local/lib/python3.11/dist-packages/sklearn/preprocessing/_data.py in partial_fit(self, X, y, sample_weight)
    859 
    860         first_call = not hasattr(self, "n_samples_seen_")
--> 861         X = self._validate_data(
    862             X,
    863             accept_sparse=("csr", "csc"),

/usr/local/lib/python3.11/dist-packages/sklearn/base.py in _validate_data(self, X, y, reset, validate_separately, **check_params)
    563             raise ValueError("Validation should be done on X, y or both.")
    564         elif not no_val_X and no_val_y:
--> 565             X = check_array(X, input_name="X", **check_params)
    566             out = X
    567         elif no_val_X and not no_val_y:

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in check_array(array, accept_sparse, accept_large_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, ensure_min_samples, ensure_min_features, estimator, input_name)
    877                     array = xp.astype(array, dtype, copy=False)
    878                 else:
--> 879                     array = _asarray_with_order(array, order=order, dtype=dtype, xp=xp)
    880             except ComplexWarning as complex_warning:
    881                 raise ValueError(

/usr/local/lib/python3.11/dist-packages/sklearn/utils/_array_api.py in _asarray_with_order(array, dtype, order, copy, xp)
    183     if xp.__name__ in {"numpy", "numpy.array_api"}:
    184         # Use NumPy API to support order
--> 185         array = numpy.asarray(array, order=order, dtype=dtype)
    186         return xp.asarray(array, copy=copy)
    187     else:

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in __array__(self, dtype, copy)
   2151     ) -> np.ndarray:
   2152         values = self._values
-> 2153         arr = np.asarray(values, dtype=dtype)
   2154         if (
   2155             astype_is_view(values.dtype, arr.dtype)

TypeError: float() argument must be a string or a real number, not 'Timestamp'

## === cell 17
mlp = MLPRegressor(
    hidden_layer_sizes=(30, 15, 7, 3),
    activation="relu",
    solver="adam",
    alpha=0.0001,
    batch_size=1024,
    learning_rate_init=0.001,
    max_iter=5,  # matches original epochs=5 intent
    shuffle=True,
    random_state=42,
    verbose=True,
)



## === cell 18
y_np = y.astype(np.float32, copy=False)
mlp.fit(x_scaled, y_np)

test_pred = mlp.predict(x_test_scaled).reshape(-1)
test_pred = np.maximum(test_pred, 0.0)

sub = pd.DataFrame({"key": test_id, "fare_amount": test_pred})
sub.to_csv("submission_nn.csv", index=False)

print("Wrote submission_nn.csv with shape:", sub.shape)
print(sub.head())

## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1666790481.py in <cell line: 0>()
      1 y_np = y.astype(np.float32, copy=False)
----> 2 mlp.fit(x_scaled, y_np)
      3 
      4 test_pred = mlp.predict(x_test_scaled).reshape(-1)
      5 test_pred = np.maximum(test_pred, 0.0)

NameError: name 'x_scaled' is not defined
