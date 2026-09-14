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

12.50832

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import datetime as dt
import random
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
import xgboost as xgb

from pandas.tseries.holiday import USFederalHolidayCalendar as calendar



## === cell 1
train = pd.read_csv("../input/train.csv", nrows=1_000_000)
print("train shape:", train.shape)
train.head()



## === cell 2
test = pd.read_csv("../input/test.csv")
test.head()



## === cell 3
combine = [train, test]

for dataset in combine:
    dataset["longitude_distance"] = abs(
        dataset["pickup_longitude"] - dataset["dropoff_longitude"]
    )
    dataset["latitude_distance"] = abs(
        dataset["pickup_latitude"] - dataset["dropoff_latitude"]
    )
    dataset["distance_travelled"] = (
        dataset["longitude_distance"] ** 2 * dataset["latitude_distance"] ** 2
    ) ** 0.5

    R = 6371e3  # metres
    phi1 = np.radians(dataset["pickup_latitude"])
    phi2 = np.radians(dataset["dropoff_latitude"])
    phi_chg = np.radians(dataset["pickup_latitude"] - dataset["dropoff_latitude"])
    delta_chg = np.radians(dataset["pickup_longitude"] - dataset["dropoff_longitude"])
    a = np.sin(phi_chg / 2) + np.cos(phi1) * np.cos(phi2) * np.sin(delta_chg / 2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    dataset["haversine"] = R * c

    y = np.sin(delta_chg * np.cos(phi2))
    x = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(delta_chg)
    dataset["bearing"] = np.degrees(np.arctan2(y, x))

    psi_chg = np.log(np.tan(np.pi / 4 + phi2 / 2) / np.tan(np.pi / 4 + phi1 / 2))
    q = phi_chg / psi_chg
    d = (phi_chg + q**2 * delta_chg**2) ** 0.5 * R
    dataset["rhumb_lines"] = d

    dataset["pickup_datetime"] = pd.to_datetime(
        dataset["pickup_datetime"], infer_datetime_format=True
    )

    dataset["hour_of_day"] = dataset["pickup_datetime"].dt.hour
    dataset["day"] = dataset["pickup_datetime"].dt.day
    dataset["week"] = dataset["pickup_datetime"].dt.isocalendar().week
    dataset["month"] = dataset["pickup_datetime"].dt.month
    dataset["dayofweek"] = dataset["pickup_datetime"].dt.dayofweek
    dataset["day_of_year"] = dataset["pickup_datetime"].dt.dayofyear
    dataset["week_of_year"] = dataset["pickup_datetime"].dt.isocalendar().week

    cal = calendar()
    holidays = cal.holidays()
    dataset["usFedHoliday"] = (
        dataset["pickup_datetime"].dt.date.astype("datetime64").isin(holidays)
    )

train.head(3)



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/348673704.py in <cell line: 0>()
     52     holidays = cal.holidays()
     53     dataset["usFedHoliday"] = (
---> 54         dataset["pickup_datetime"].dt.date.astype("datetime64").isin(holidays)
     55     )
     56 

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in astype(self, dtype, copy, errors)
   6641         else:
   6642             # else, only a single dtype is given
-> 6643             new_data = self._mgr.astype(dtype=dtype, copy=copy, errors=errors)
   6644             res = self._constructor_from_mgr(new_data, axes=new_data.axes)
   6645             return res.__finalize__(self, method="astype")

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/managers.py in astype(self, dtype, copy, errors)
    428             copy = False
    429 
--> 430         return self.apply(
    431             "astype",
    432             dtype=dtype,

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/managers.py in apply(self, f, align_keys, **kwargs)
    361                 applied = b.apply(f, **kwargs)
    362             else:
--> 363                 applied = getattr(b, f)(**kwargs)
    364             result_blocks = extend_blocks(applied, result_blocks)
    365 

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/blocks.py in astype(self, dtype, copy, errors, using_cow, squeeze)
    756             values = values[0, :]  # type: ignore[call-overload]
    757 
--> 758         new_values = astype_array_safe(values, dtype, copy=copy, errors=errors)
    759 
    760         new_values = maybe_coerce_values(new_values)

/usr/local/lib/python3.11/dist-packages/pandas/core/dtypes/astype.py in astype_array_safe(values, dtype, copy, errors)
    235 
    236     try:
--> 237         new_values = astype_array(values, dtype, copy=copy)
    238     except (ValueError, TypeError):
    239         # e.g. _astype_nansafe can fail on object-dtype of strings

/usr/local/lib/python3.11/dist-packages/pandas/core/dtypes/astype.py in astype_array(values, dtype, copy)
    180 
    181     else:
--> 182         values = _astype_nansafe(values, dtype, copy=copy)
    183 
    184     # in pandas we don't store numpy str dtypes, so convert to object

/usr/local/lib/python3.11/dist-packages/pandas/core/dtypes/astype.py in _astype_nansafe(arr, dtype, copy, skipna)
    108             from pandas.core.arrays import DatetimeArray
    109 
--> 110             dta = DatetimeArray._from_sequence(arr, dtype=dtype)
    111             return dta._ndarray
    112 

/usr/local/lib/python3.11/dist-packages/pandas/core/arrays/datetimes.py in _from_sequence(cls, scalars, dtype, copy)
    325     @classmethod
    326     def _from_sequence(cls, scalars, *, dtype=None, copy: bool = False):
--> 327         return cls._from_sequence_not_strict(scalars, dtype=dtype, copy=copy)
    328 
    329     @classmethod

/usr/local/lib/python3.11/dist-packages/pandas/core/arrays/datetimes.py in _from_sequence_not_strict(cls, data, dtype, copy, tz, freq, dayfirst, yearfirst, ambiguous)
    352             tz = timezones.maybe_get_tz(tz)
    353 
--> 354         dtype = _validate_dt64_dtype(dtype)
    355         # if dtype has an embedded tz, capture it
    356         tz = _validate_tz_from_dtype(dtype, tz, explicit_tz_none)

/usr/local/lib/python3.11/dist-packages/pandas/core/arrays/datetimes.py in _validate_dt64_dtype(dtype)
   2542                 "Please pass in 'datetime64[ns]' instead."
   2543             )
-> 2544             raise ValueError(msg)
   2545 
   2546         if (

ValueError: Passing in 'datetime64' dtype with no precision is not allowed. Please pass in 'datetime64[ns]' instead.

## === cell 4
numeric_corr = train.select_dtypes(include=[np.number]).corr()
colormap = plt.cm.RdBu
plt.figure(figsize=(20, 20))
plt.title("Pearson Correlation of Numeric Features", y=1.05, size=15)
sns.heatmap(
    numeric_corr,
    linewidths=0.1,
    vmax=1.0,
    square=True,
    cmap=colormap,
    linecolor="white",
    annot=True,
)



## === cell 5
train.drop(["key", "pickup_datetime"], axis=1, inplace=True)
train.dropna(inplace=True)

test.drop(["pickup_datetime"], axis=1, inplace=True)  # keep 'key' for submission



## === cell 6
x_pred = test.drop("key", axis=1)

X = train.drop("fare_amount", axis=1)
y = train["fare_amount"]

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123
)


def XGBmodel(x_train, x_test, y_train, y_test):
    dtrain = xgb.DMatrix(x_train, label=y_train)
    dtest = xgb.DMatrix(x_test, label=y_test)
    params = {
        "objective": "reg:squarederror",  # modern equivalent of reg:linear
        "eval_metric": "rmse",
    }
    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=400,
        early_stopping_rounds=30,
        evals=[(dtest, "test")],
        verbose_eval=False,
    )
    return model


model = XGBmodel(x_train, x_test, y_train, y_test)

best_ntree = (
    model.best_iteration if hasattr(model, "best_iteration") else model.best_ntree_limit
)
prediction = model.predict(xgb.DMatrix(x_pred), ntree_limit=best_ntree)



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2459440026.py in <cell line: 0>()
     34     model.best_iteration if hasattr(model, "best_iteration") else model.best_ntree_limit
     35 )
---> 36 prediction = model.predict(xgb.DMatrix(x_pred), ntree_limit=best_ntree)
     37 

TypeError: Booster.predict() got an unexpected keyword argument 'ntree_limit'

## === cell 7
submission = pd.DataFrame({"key": test["key"], "fare_amount": np.round(prediction, 2)})

submission.to_csv("sub_fare.csv", index=False)



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/118694137.py in <cell line: 0>()
----> 1 submission = pd.DataFrame({"key": test["key"], "fare_amount": np.round(prediction, 2)})
      2 
      3 submission.to_csv("sub_fare.csv", index=False)
      4 

NameError: name 'prediction' is not defined

## === cell 8
submission.head()

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3365464162.py in <cell line: 0>()
----> 1 submission.head()

NameError: name 'submission' is not defined
