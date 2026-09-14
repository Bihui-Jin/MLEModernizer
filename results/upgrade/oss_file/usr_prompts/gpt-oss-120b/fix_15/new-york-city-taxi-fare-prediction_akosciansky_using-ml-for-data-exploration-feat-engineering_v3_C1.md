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

fastai==2.8.5
geopandas==0.14.4
google-api-python-client==2.177.0
ipython==7.34.0
ipython-genutils==0.2.0
ipython_pygments_lexers==1.1.1
ipython-sql==0.5.0
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
scipy==1.15.3
sklearn-pandas==2.2.0

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

3.5071

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 5.06791) has done: 'I fix the datetime handling by removing the unsupported “Week” attribute (or replacing it with the ISO week) in `add_datepart`, guard the attribute access, and then the preprocessing run correctly so `df` and `y` are defined. This enables the RandomForest model to train and the script to produce a valid `submission.csv` without further errors.'
- What this solution (achieved 5.78883) has done: 'I add a geographic distance feature (haversine) to both train and test data, because fare strongly depends on trip length, and increase the RandomForest size to capture this information better. The distance column is created before categorical conversion, then the existing pipeline runs unchanged. I also raise `n_estimators` from 30 to 200 for a more stable model, keeping all other logic intact.'

# 9. Code solution

## === cell 0
import os, warnings

warnings.filterwarnings("ignore")

import pandas as pd, numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer  # added for NaN handling


def _find_data_path(fname: str) -> str:
    """Return the first existing path for *fname* in known data folders."""
    base_dirs = [
        Path("./input/new-york-city-taxi-fare-prediction"),
        Path("./working/new-york-city-taxi-fare-prediction"),
        Path("./data/new-york-city-taxi-fare-prediction"),
        Path("/kaggle/input/new-york-city-taxi-fare-prediction"),
        Path("/kaggle/working/new-york-city-taxi-fare-prediction"),
    ]
    for b in base_dirs:
        p = b / fname
        if p.is_file():
            return str(p)
    raise FileNotFoundError(f"{fname} not found in any known data directory.")


def haversine_np(lon1, lat1, lon2, lat2):
    """Vectorised haversine distance (km) between two points."""
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371.0 * c


def _add_datepart(df: pd.DataFrame, field_name: str, drop: bool = True):
    """Expand a datetime column into many useful columns."""
    dt = getattr(df, field_name)
    df[field_name + "_Year"] = dt.dt.year
    df[field_name + "_Month"] = dt.dt.month
    df[field_name + "_Day"] = dt.dt.day
    df[field_name + "_Hour"] = dt.dt.hour
    df[field_name + "_Minute"] = dt.dt.minute
    df[field_name + "_Second"] = dt.dt.second
    df[field_name + "_Weekofyear"] = dt.dt.isocalendar().week.astype(int)
    df[field_name + "_Dayofweek"] = dt.dt.dayofweek
    df[field_name + "_Is_month_end"] = dt.dt.is_month_end.astype(int)
    df[field_name + "_Is_month_start"] = dt.dt.is_month_start.astype(int)
    df[field_name + "_Is_quarter_end"] = dt.dt.is_quarter_end.astype(int)
    df[field_name + "_Is_quarter_start"] = dt.dt.is_quarter_start.astype(int)
    df[field_name + "_Is_year_end"] = dt.dt.is_year_end.astype(int)
    df[field_name + "_Is_year_start"] = dt.dt.is_year_start.astype(int)
    if drop:
        df.drop(columns=[field_name], inplace=True)


def rf_feat_importance(model, X):
    """Return a pandas Series of feature importances sorted descending."""
    importances = model.feature_importances_
    return pd.Series(importances, index=X.columns).sort_values(ascending=False)




## === cell 1
train_path = _find_data_path("labels.csv")
test_path = _find_data_path("test.csv")

df_raw = pd.read_csv(
    train_path,
    nrows=500_000,
    parse_dates=["pickup_datetime"],
    dtype={"passenger_count": "int8", "fare_amount": "float32"},
    usecols=[
        "fare_amount",
        "pickup_datetime",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
    ],
)

df_raw = df_raw.dropna(subset=["fare_amount"])

df_raw_test = pd.read_csv(
    test_path,
    parse_dates=["pickup_datetime"],
    dtype={"passenger_count": "int8"},
    usecols=[
        "key",
        "pickup_datetime",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
    ],
)

df_raw["distance"] = haversine_np(
    df_raw["pickup_longitude"],
    df_raw["pickup_latitude"],
    df_raw["dropoff_longitude"],
    df_raw["dropoff_latitude"],
)

df_raw_test["distance"] = haversine_np(
    df_raw_test["pickup_longitude"],
    df_raw_test["pickup_latitude"],
    df_raw_test["dropoff_longitude"],
    df_raw_test["dropoff_latitude"],
)

df_raw = df_raw[
    (df_raw["pickup_longitude"] > -76)
    & (df_raw["pickup_longitude"] < -73)
    & (df_raw["pickup_latitude"] > 40)
    & (df_raw["pickup_latitude"] < 44)
]

_add_datepart(df_raw, "pickup_datetime")
_add_datepart(df_raw_test, "pickup_datetime")




## === cell 2
y = np.log1p(df_raw["fare_amount"].values)

X = df_raw.drop(columns=["fare_amount"])

imputer = SimpleImputer(strategy="median")
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

m = RandomForestRegressor(
    n_estimators=800,
    min_samples_leaf=1,
    oob_score=False,  # disable oob to speed up training
    n_jobs=-1,
    random_state=42,
)
m.fit(X, y)




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_10/3130362963.py in <cell line: 0>()
     17     random_state=42,
     18 )
---> 19 m.fit(X, y)
     20 
     21 

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
   1120     )
   1121 
-> 1122     y = _check_y(y, multi_output=multi_output, y_numeric=y_numeric, estimator=estimator)
   1123 
   1124     check_consistent_length(X, y)

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in _check_y(y, multi_output, y_numeric, estimator)
   1130     """Isolated part of check_X_y dedicated to y validation"""
   1131     if multi_output:
-> 1132         y = check_array(
   1133             y,
   1134             accept_sparse="csr",

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in check_array(array, accept_sparse, accept_large_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, ensure_min_samples, ensure_min_features, estimator, input_name)
    919 
    920         if force_all_finite:
--> 921             _assert_all_finite(
    922                 array,
    923                 input_name=input_name,

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in _assert_all_finite(X, allow_nan, msg_dtype, estimator_name, input_name)
    159                 "#estimators-that-handle-nan-values"
    160             )
--> 161         raise ValueError(msg_err)
    162 
    163 

ValueError: Input y contains NaN.

## === cell 3
fi = rf_feat_importance(m, X)
print("Top features:", fi.head())

X_test = df_raw_test.drop(columns=["key"])
X_test = X_test[X.columns]  # ensure same column order as training
X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

log_pred = m.predict(X_test)
y_pred = np.expm1(log_pred)
y_pred = np.clip(y_pred, 0, None)  # fares cannot be negative

submission = pd.DataFrame({"key": df_raw_test["key"], "fare_amount": y_pred})
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}, shape: {submission.shape}")

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_10/151955673.py in <cell line: 0>()
      1 # Feature importance (optional insight)
----> 2 fi = rf_feat_importance(m, X)
      3 print("Top features:", fi.head())
      4 
      5 # Prepare test features, align column order, and impute

/tmp/ipykernel_10/411653868.py in rf_feat_importance(model, X)
     58 def rf_feat_importance(model, X):
     59     """Return a pandas Series of feature importances sorted descending."""
---> 60     importances = model.feature_importances_
     61     return pd.Series(importances, index=X.columns).sort_values(ascending=False)
     62 

/usr/local/lib/python3.11/dist-packages/sklearn/ensemble/_forest.py in feature_importances_(self)
    630         all_importances = Parallel(n_jobs=self.n_jobs, prefer="threads")(
    631             delayed(getattr)(tree, "feature_importances_")
--> 632             for tree in self.estimators_
    633             if tree.tree_.node_count > 1
    634         )

AttributeError: 'RandomForestRegressor' object has no attribute 'estimators_'
