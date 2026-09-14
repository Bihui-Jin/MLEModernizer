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
scipy==1.15.3
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

4.22828

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

- What this solution (achieved 6.55718) has done: 'The main fixes are:
* Replace the removed `Imputer` with `SimpleImputer`.
* Ensure the import error no longer stops the notebook, so all functions (including `chunck_generator`) are defined.
* Adjust the imputer initialization to use `np.nan`.
* Keep the overall logic unchanged while making the script runnable and capable of writing a proper `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split

try:
    from sklearnex import patch_all

    patch_all()
except Exception:
    pass

print(os.listdir("../input"))


def chunck_generator(filename, header=False, chunk_size=5 * 10**6):
    """Yield CSV chunks for large files, loading only needed columns with optimal dtypes."""
    usecols = [
        "fare_amount",
        "pickup_datetime",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
    ]
    dtype = {
        "fare_amount": np.float32,
        "pickup_longitude": np.float32,
        "pickup_latitude": np.float32,
        "dropoff_longitude": np.float32,
        "dropoff_latitude": np.float32,
        "passenger_count": np.int8,  # passenger count never exceeds 127
    }
    parse_dates = ["pickup_datetime"]
    date_parser = lambda cols: pd.to_datetime(
        cols, format="%Y-%m-%d %H:%M:%S", errors="coerce"
    )
    for chunk in pd.read_csv(
        filename,
        delimiter=",",
        iterator=True,
        chunksize=chunk_size,
        usecols=usecols,
        dtype=dtype,
        parse_dates=parse_dates,
        date_parser=date_parser,
    ):
        yield chunk




## === cell 1
alpha_ang = 0.506


def distance_travel(df):
    """Vectorised travel‑distance features without repeated DataFrame look‑ups."""
    plon = df["pickup_longitude"].values
    plat = df["pickup_latitude"].values
    dlon = df["dropoff_longitude"].values
    dlat = df["dropoff_latitude"].values

    abs_diff_longitude = np.abs(dlon - plon) * 50.0
    abs_diff_latitude = np.abs(dlat - plat) * 69.0
    displacement_vector = np.sqrt(abs_diff_longitude**2 + abs_diff_latitude**2)

    eps = 1e-12
    theta = np.arctan(abs_diff_longitude / (abs_diff_latitude + eps))

    actual_long = np.abs(displacement_vector * np.sin(theta - alpha_ang))
    actual_lat = np.abs(displacement_vector * np.cos(theta - alpha_ang))
    distance = actual_long + actual_lat

    df["abs_diff_longitude"] = abs_diff_longitude.astype(np.float32)
    df["abs_diff_latitude"] = abs_diff_latitude.astype(np.float32)
    df["displacement_vector"] = displacement_vector.astype(np.float32)
    df["actual_long"] = actual_long.astype(np.float32)
    df["actual_lat"] = actual_lat.astype(np.float32)
    df["distance_travel"] = distance.astype(np.float32)
    return df




## === cell 2
def add_time_features(df):
    """Extract hour / dayofweek / month from already‑parsed datetime."""
    dt = df["pickup_datetime"]
    df["hour"] = dt.dt.hour.astype(np.int8)
    df["dayofweek"] = dt.dt.dayofweek.astype(np.int8)
    df["month"] = dt.dt.month.astype(np.int8)
    return df


def data_clean(df):
    df = df[df.passenger_count > 0]
    df = df[df.fare_amount > 0]
    distance_travel(df)  # adds travel columns in‑place
    add_time_features(df)  # adds hour, dayofweek, month
    df = df[df.distance_travel > 0]
    return df




## === cell 3
def remove_outliers(df):
    """Simple deterministic outlier filter."""
    df = df[df.distance_travel < 30]
    df = df[df.fare_amount < 100]
    return df




## === cell 4
def incremental_training(train_X, train_y, regr):
    """Fit the regressor on the whole training set (single call)."""
    regr.fit(train_X, train_y)
    return regr




## === cell 5
filename = r"../input/train.csv"
gen = chunck_generator(filename=filename)

all_X = []
all_y = []

chunk_idx = 0
while True:
    try:
        df = next(gen)
    except StopIteration:
        break

    chunk_idx += 1
    print(f"Chunk {chunk_idx}")

    df = data_clean(df)
    df = remove_outliers(df)

    if df.empty:
        continue

    X_chunk = np.column_stack(
        (
            df.distance_travel.values,
            df.passenger_count.values.astype(np.float32),
            df.hour.values.astype(np.float32),
            df.dayofweek.values.astype(np.float32),
            df.month.values.astype(np.float32),
        )
    )
    y_chunk = df.fare_amount.values

    all_X.append(X_chunk)
    all_y.append(y_chunk)

train_X_full = np.vstack(all_X)
train_y_full = np.concatenate(all_y)

imp = SimpleImputer(missing_values=np.nan, strategy="mean")
train_X_full = imp.fit_transform(train_X_full).astype(np.float32)

X_tr, X_val, y_tr, y_val = train_test_split(
    train_X_full, train_y_full, test_size=0.1, random_state=42
)

regr = HistGradientBoostingRegressor(
    max_iter=200,
    learning_rate=0.05,
    max_depth=5,
    random_state=42,
)

regr = incremental_training(X_tr, y_tr, regr)

val_pred = regr.predict(X_val)
val_rmse = np.sqrt(((val_pred - y_val) ** 2).mean())
print("Validation RMSE:", val_rmse)




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
IntCastingNaNError                        Traceback (most recent call last)
/tmp/ipykernel_11/305874348.py in <cell line: 0>()
     15     print(f"Chunk {chunk_idx}")
     16 
---> 17     df = data_clean(df)
     18     df = remove_outliers(df)
     19 

/tmp/ipykernel_11/3098255419.py in data_clean(df)
     13     df = df[df.fare_amount > 0]
     14     distance_travel(df)  # adds travel columns in‑place
---> 15     add_time_features(df)  # adds hour, dayofweek, month
     16     df = df[df.distance_travel > 0]
     17     return df

/tmp/ipykernel_11/3098255419.py in add_time_features(df)
      3     # datetime already parsed in chunck_generator; no extra conversion needed
      4     dt = df["pickup_datetime"]
----> 5     df["hour"] = dt.dt.hour.astype(np.int8)
      6     df["dayofweek"] = dt.dt.dayofweek.astype(np.int8)
      7     df["month"] = dt.dt.month.astype(np.int8)

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
     99 
    100     elif np.issubdtype(arr.dtype, np.floating) and dtype.kind in "iu":
--> 101         return _astype_float_to_int_nansafe(arr, dtype, copy)
    102 
    103     elif arr.dtype == object:

/usr/local/lib/python3.11/dist-packages/pandas/core/dtypes/astype.py in _astype_float_to_int_nansafe(values, dtype, copy)
    143     """
    144     if not np.isfinite(values).all():
--> 145         raise IntCastingNaNError(
    146             "Cannot convert non-finite values (NA or inf) to integer"
    147         )

IntCastingNaNError: Cannot convert non-finite values (NA or inf) to integer

## === cell 6
test_df = pd.read_csv(
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
        "pickup_longitude": np.float32,
        "pickup_latitude": np.float32,
        "dropoff_longitude": np.float32,
        "dropoff_latitude": np.float32,
        "passenger_count": np.int8,
    },
    parse_dates=["pickup_datetime"],
    date_parser=lambda cols: pd.to_datetime(
        cols, format="%Y-%m-%d %H:%M:%S", errors="coerce"
    ),
)

distance_travel(test_df)
add_time_features(test_df)

test_X = np.column_stack(
    (
        test_df.distance_travel.values,
        test_df.passenger_count.values.astype(np.float32),
        test_df.hour.values.astype(np.float32),
        test_df.dayofweek.values.astype(np.float32),
        test_df.month.values.astype(np.float32),
    )
)

test_X = imp.transform(test_X)

predicted_fare = regr.predict(test_X)
print("Sample predictions:", predicted_fare[:5])




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
IntCastingNaNError                        Traceback (most recent call last)
/tmp/ipykernel_11/875501683.py in <cell line: 0>()
     24 
     25 distance_travel(test_df)
---> 26 add_time_features(test_df)
     27 
     28 test_X = np.column_stack(

/tmp/ipykernel_11/3098255419.py in add_time_features(df)
      3     # datetime already parsed in chunck_generator; no extra conversion needed
      4     dt = df["pickup_datetime"]
----> 5     df["hour"] = dt.dt.hour.astype(np.int8)
      6     df["dayofweek"] = dt.dt.dayofweek.astype(np.int8)
      7     df["month"] = dt.dt.month.astype(np.int8)

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
     99 
    100     elif np.issubdtype(arr.dtype, np.floating) and dtype.kind in "iu":
--> 101         return _astype_float_to_int_nansafe(arr, dtype, copy)
    102 
    103     elif arr.dtype == object:

/usr/local/lib/python3.11/dist-packages/pandas/core/dtypes/astype.py in _astype_float_to_int_nansafe(values, dtype, copy)
    143     """
    144     if not np.isfinite(values).all():
--> 145         raise IntCastingNaNError(
    146             "Cannot convert non-finite values (NA or inf) to integer"
    147         )

IntCastingNaNError: Cannot convert non-finite values (NA or inf) to integer

## === cell 7
my_submission = pd.DataFrame({"key": test_df.key, "fare_amount": predicted_fare})
my_submission.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv (first rows):")
print(my_submission.head())

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3845590556.py in <cell line: 0>()
----> 1 my_submission = pd.DataFrame({"key": test_df.key, "fare_amount": predicted_fare})
      2 my_submission.to_csv("submission.csv", index=False)
      3 print("Submission saved to submission.csv (first rows):")
      4 print(my_submission.head())

NameError: name 'predicted_fare' is not defined
