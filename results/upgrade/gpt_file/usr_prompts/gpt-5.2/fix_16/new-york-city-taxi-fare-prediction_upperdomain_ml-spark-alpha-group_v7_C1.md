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

4.07608

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 5.66111) has done: 'I fix the immediate import/runtime errors by replacing the removed `Imputer` with `SimpleImputer`, ensuring `pandas/numpy` are imported before use, and correcting the data paths to the Kaggle-provided `/kaggle/input/...` structure with a safe fallback to `../input` when present. I also make the distance feature computation robust to division-by-zero so the model can fit/predict without producing NaNs/Infs. To keep the core logic unchanged, I preserve the same feature set (distance_travel, passenger_count, bias term) and the same `GradientBoostingRegressor` settings, only ensuring the imputer is actually applied to both train and test matrices. Finally, I guarantee a valid `submission.csv` with exactly `key,fare_amount` is written.'
- What this solution (achieved 6.47829) has done: 'To move RMSE down toward the 4.07608 target without changing the core model/feature set, I make the training split representative by shuffling before the 70/30 split (the current sequential split can be biased by time ordering). I also apply the exact same basic sanity filters used in training to the test feature generation (distance>0, passenger_count>0) and safely handle any remaining NaN/Inf distances so the imputer/model sees consistent inputs. Finally, I keep the same `GradientBoostingRegressor` configuration and feature columns, but ensure predictions are aligned to the original test row order and always produce a valid `submission.csv` with `key,fare_amount`.'
- What this solution (achieved 5.29348) has done: 'To reduce RMSE toward the 4.07608 target without changing the model or feature set, I keep the same `GradientBoostingRegressor` and the same three input columns, but make the train/test preprocessing consistent by applying the *same* outlier/validity filters to the training data that the distance feature expects (bounding lat/longs and removing extreme coordinate zeros) so the model learns from cleaner, more realistic trips. I also compute `distance_travel` on a copy to avoid chained-assignment issues and ensure missing/invalid values become NaN consistently for the imputer. Finally, I fix a subtle alignment risk in the submission creation by predicting in the same order as `tdf_sorted` (rather than indexing predictions by original DataFrame indices), while still writing `submission.csv` with exactly `key,fare_amount`.'
- What this solution (achieved 5.28674) has done: 'You’re currently worse than the target (RMSE 5.29348 vs 4.07608, lower is better), so the smallest safe move is to improve generalization without changing the model or feature set. The biggest low-risk issue is that the model is being trained on only the first 1,000,000 rows (time-ordered), which is less representative; we instead sample 1,000,000 rows uniformly across the full training file using `skiprows` so it stays fast and within memory. We keep the exact same filters, the exact same 3-feature matrix, and the same `GradientBoostingRegressor` config, but train on a more representative subset. This should reduce public RMSE while preserving your core logic and still writing a valid `submission.csv`.'
- What this solution (achieved 5.28637) has done: 'Your current RMSE (5.28674) is still meaningfully worse than the target (4.07608, lower is better), so we should improve generalization with the smallest changes that don’t alter the core model/features. The biggest low-risk issue is train/serving skew: you filter training to `distance_travel < 30` and `fare_amount < 100`, but you do not apply the analogous `distance_travel < 30` filter to test, so the model is forced to extrapolate on longer trips, hurting RMSE. I keep the exact same feature set (distance_travel, passenger_count, bias) and the same `GradientBoostingRegressor`, but I (1) clamp test `distance_travel` to the same upper bound (set >30 to NaN so the existing imputer handles it) and (2) prevent huge passenger_count values from leaking into inference by setting unrealistic counts (>6) to NaN (again handled by the same imputer). These changes keep evaluation semantics intact, avoid changing the training loop/architecture, and should move the score down toward the target.'
- What this solution (achieved 5.28912) has done: 'Your current RMSE (5.28637, lower is better) is still far from the target (4.07608), so we should make a small, safe improvement that doesn’t change the model or feature set. The biggest remaining low-risk issue is that we only train on 1,000,000 rows; increasing the sampled training size (while keeping the same filtering, same distance feature, and same `GradientBoostingRegressor`) typically improves generalization and reduces Kaggle RMSE without altering core logic. To keep runtime/memory under control and preserve determinism, we bump the sample to 2,000,000 and keep the same `skiprows` sampling approach and seeds. Everything else (features, filters, imputation, model params, submission format) stays the same.'

# 9. Code solution

## === cell 0
import os
import math
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.interpolate import griddata  # noqa: F401

try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except Exception:
    pass

from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error


def _resolve_input_path(filename: str) -> str:
    """
    Kaggle typically mounts datasets under /kaggle/input/<dataset-slug>/.
    This notebook's original code used ../input/, so we support both.
    """
    candidates = [
        f"/kaggle/input/new-york-city-taxi-fare-prediction/{filename}",
        f"/kaggle/input/{filename}",
        f"../input/{filename}",
        f"/kaggle/data/new-york-city-taxi-fare-prediction/{filename}",
        f"/kaggle/data/{filename}",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return candidates[0]


print("Listing /kaggle/input (if present):")
if os.path.exists("/kaggle/input"):
    print(os.listdir("/kaggle/input")[:50])
else:
    print("No /kaggle/input directory; will rely on fallback paths if available.")


## === cell 1
train_path = _resolve_input_path("train.csv")

N_TOTAL = 55_423_856  # provided dataset size
N_SAMPLE = 3_000_000

rng = np.random.RandomState(21)

train_usecols = [
    "key",
    "fare_amount",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]

_train_dtypes = {
    "key": "string",
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int16",
}


def sample_csv_by_skiprows(
    path: str,
    usecols,
    n_total: int,
    n_sample: int,
    rng_: np.random.RandomState,
    dtype: dict,
) -> pd.DataFrame:
    keep = rng_.choice(
        np.arange(1, n_total + 1, dtype=np.int64), size=n_sample, replace=False
    )
    keep.sort()

    skip_mask = np.ones(n_total + 1, dtype=bool)
    skip_mask[0] = False  # keep header
    skip_mask[keep] = False  # keep sampled rows

    df_ = pd.read_csv(
        path,
        usecols=usecols,
        dtype=dtype,
        low_memory=False,
        engine="c",
        skiprows=skip_mask,
    )
    if len(df_) != n_sample:
        raise RuntimeError(f"Sampling failed: got {len(df_)} rows, expected {n_sample}")
    return df_


df = sample_csv_by_skiprows(
    train_path,
    usecols=train_usecols,
    n_total=N_TOTAL,
    n_sample=N_SAMPLE,
    rng_=rng,
    dtype=_train_dtypes,
)

df["passenger_count"] = df["passenger_count"].astype("float32", copy=False)

print("Loaded sampled train rows:", df.shape)
df.head()


## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/4125324848.py in <cell line: 0>()
     69 
     70 
---> 71 df = sample_csv_by_skiprows(
     72     train_path,
     73     usecols=train_usecols,

/tmp/ipykernel_11/4125324848.py in sample_csv_by_skiprows(path, usecols, n_total, n_sample, rng_, dtype)
     56     skip_mask[keep] = False  # keep sampled rows
     57 
---> 58     df_ = pd.read_csv(
     59         path,
     60         usecols=usecols,

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in read_csv(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, date_format, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, encoding_errors, dialect, on_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options, dtype_backend)
   1024     kwds.update(kwds_defaults)
   1025 
-> 1026     return _read(filepath_or_buffer, kwds)
   1027 
   1028 

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _read(filepath_or_buffer, kwds)
    618 
    619     # Create the parser.
--> 620     parser = TextFileReader(filepath_or_buffer, **kwds)
    621 
    622     if chunksize or iterator:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in __init__(self, f, engine, **kwds)
   1618 
   1619         self.handles: IOHandles | None = None
-> 1620         self._engine = self._make_engine(f, self.engine)
   1621 
   1622     def close(self) -> None:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _make_engine(self, f, engine)
   1896 
   1897         try:
-> 1898             return mapping[engine](f, **self.options)
   1899         except Exception:
   1900             if self.handles is not None:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/c_parser_wrapper.py in __init__(self, src, **kwds)
    138                 self.orig_names
    139             ):
--> 140                 self._validate_usecols_names(usecols, self.orig_names)
    141 
    142             # error: Cannot determine type of 'names'

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/base_parser.py in _validate_usecols_names(self, usecols, names)
    977         missing = [c for c in usecols if c not in names]
    978         if len(missing) > 0:
--> 979             raise ValueError(
    980                 f"Usecols do not match columns, columns expected but not found: "
    981                 f"{missing}"

ValueError: Usecols do not match columns, columns expected but not found: ['fare_amount', 'passenger_count', 'dropoff_latitude', 'key', 'pickup_longitude', 'dropoff_longitude', 'pickup_latitude']

## === cell 2
m = (df.passenger_count > 0) & (df.fare_amount > 0)
m &= (
    df.pickup_longitude.between(-75, -72)
    & df.dropoff_longitude.between(-75, -72)
    & df.pickup_latitude.between(40, 42)
    & df.dropoff_latitude.between(40, 42)
)
df = df.loc[m].copy()

df.head()


## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1136830214.py in <cell line: 0>()
      1 # SPEED FIX (correctness-preserving):
      2 # Combine filters via boolean masks to avoid creating multiple intermediate DataFrames.
----> 3 m = (df.passenger_count > 0) & (df.fare_amount > 0)
      4 m &= (
      5     df.pickup_longitude.between(-75, -72)

NameError: name 'df' is not defined

## === cell 3
alpha_ang = 0.506


def distance_travel(df_):
    p_lo = df_["pickup_longitude"].to_numpy(dtype=np.float32, copy=False)
    d_lo = df_["dropoff_longitude"].to_numpy(dtype=np.float32, copy=False)
    p_la = df_["pickup_latitude"].to_numpy(dtype=np.float32, copy=False)
    d_la = df_["dropoff_latitude"].to_numpy(dtype=np.float32, copy=False)

    abs_diff_longitude = np.abs(d_lo - p_lo) * np.float32(50.0)
    abs_diff_latitude = np.abs(d_la - p_la) * np.float32(69.0)
    displacement_vector = np.sqrt(
        abs_diff_latitude * abs_diff_latitude + abs_diff_longitude * abs_diff_longitude
    ).astype(np.float32, copy=False)

    denom = abs_diff_latitude.copy()
    denom[denom == 0.0] = np.nan
    angle = np.arctan(abs_diff_longitude / denom).astype(np.float32, copy=False)

    a = np.float32(alpha_ang)
    actual_long = np.abs(displacement_vector * np.sin(angle - a)).astype(
        np.float32, copy=False
    )
    actual_lat = np.abs(displacement_vector * np.cos(angle - a)).astype(
        np.float32, copy=False
    )
    distance = (actual_long + actual_lat).astype(np.float32, copy=False)
    distance[~np.isfinite(distance)] = np.nan

    df_["abs_diff_longitude"] = abs_diff_longitude
    df_["abs_diff_latitude"] = abs_diff_latitude
    df_["displacement_vector"] = displacement_vector
    df_["actual_long"] = actual_long
    df_["actual_lat"] = actual_lat
    df_["distance_travel"] = distance


distance_travel(df)
df = df[df.distance_travel > 0]
df.head()


## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3338811282.py in <cell line: 0>()
     36 
     37 
---> 38 distance_travel(df)
     39 df = df[df.distance_travel > 0]
     40 df.head()

NameError: name 'df' is not defined

## === cell 4
if False:
    test = df[df.passenger_count == 1]
    _ = test.iloc[:20000].plot.scatter("distance_travel", "fare_amount")


## === cell 5
df = df[df.distance_travel < 30]
df = df[df.fare_amount < 100]

if False:
    _ = df.iloc[:100000].plot.scatter("distance_travel", "fare_amount")


## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/685119702.py in <cell line: 0>()
----> 1 df = df[df.distance_travel < 30]
      2 df = df[df.fare_amount < 100]
      3 
      4 if False:
      5     _ = df.iloc[:100000].plot.scatter("distance_travel", "fare_amount")

NameError: name 'df' is not defined

## === cell 6
perm = np.random.RandomState(21).permutation(len(df))
df = df.iloc[perm].reset_index(drop=True)

l = len(df)
print(l)
df_train = df[: int(0.7 * l)]
df_test = df[int(0.7 * l) :]

ones_train = np.ones(len(df_train), dtype=np.float32)
ones_test = np.ones(len(df_test), dtype=np.float32)

train_X = np.column_stack(
    (
        df_train["distance_travel"].to_numpy(dtype=np.float32, copy=False),
        df_train["passenger_count"].to_numpy(dtype=np.float32, copy=False),
        ones_train,
    )
)
test_X = np.column_stack(
    (
        df_test["distance_travel"].to_numpy(dtype=np.float32, copy=False),
        df_test["passenger_count"].to_numpy(dtype=np.float32, copy=False),
        ones_test,
    )
)
train_y = df_train["fare_amount"].to_numpy(dtype=np.float32, copy=False)
test_y = df_test["fare_amount"].to_numpy(dtype=np.float32, copy=False)


## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3848584904.py in <cell line: 0>()
----> 1 perm = np.random.RandomState(21).permutation(len(df))
      2 df = df.iloc[perm].reset_index(drop=True)
      3 
      4 l = len(df)
      5 print(l)

NameError: name 'df' is not defined

## === cell 7
imp = SimpleImputer(missing_values=np.nan, strategy="mean")
train_X = imp.fit_transform(train_X)
test_X = imp.transform(test_X)

regr = GradientBoostingRegressor(random_state=21, n_estimators=400)
regr.fit(train_X, train_y)

pred_val = regr.predict(test_X)
rmse = mean_squared_error(test_y, pred_val, squared=False)
print("Holdout RMSE:", rmse)


## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1152172831.py in <cell line: 0>()
      1 imp = SimpleImputer(missing_values=np.nan, strategy="mean")
----> 2 train_X = imp.fit_transform(train_X)
      3 test_X = imp.transform(test_X)
      4 
      5 regr = GradientBoostingRegressor(random_state=21, n_estimators=400)

NameError: name 'train_X' is not defined

## === cell 8
test_path = _resolve_input_path("test.csv")

test_usecols = [
    "key",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]

_test_dtypes = {
    "key": "string",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int16",
}
tdf = pd.read_csv(test_path, usecols=test_usecols, dtype=_test_dtypes)

tdf["_row_id"] = np.arange(len(tdf))  # preserve original order for submission alignment

coord_ok = (
    tdf.pickup_longitude.between(-75, -72)
    & tdf.dropoff_longitude.between(-75, -72)
    & tdf.pickup_latitude.between(40, 42)
    & tdf.dropoff_latitude.between(40, 42)
)
for c in [
    "pickup_longitude",
    "dropoff_longitude",
    "pickup_latitude",
    "dropoff_latitude",
]:
    tdf.loc[~coord_ok, c] = np.nan

distance_travel(tdf)

tdf.loc[tdf["passenger_count"] <= 0, "passenger_count"] = np.nan

tdf.loc[tdf["distance_travel"] <= 0, "distance_travel"] = np.nan
tdf.loc[tdf["distance_travel"] >= 30, "distance_travel"] = np.nan

tdf.loc[tdf["passenger_count"] > 6, "passenger_count"] = np.nan

tdf.head()


## === cell 9
tdf_sorted = tdf.sort_values("_row_id").reset_index(drop=True)

ones_pred = np.ones(len(tdf_sorted), dtype=np.float32)
ttrain_X = np.column_stack(
    (
        tdf_sorted["distance_travel"].to_numpy(dtype=np.float32, copy=False),
        tdf_sorted["passenger_count"].to_numpy(dtype=np.float32, copy=False),
        ones_pred,
    )
)
ttrain_X = imp.transform(ttrain_X)
output = regr.predict(ttrain_X)

output = np.clip(output, 0.0, None)

print(output[:10])


## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NotFittedError                            Traceback (most recent call last)
/tmp/ipykernel_11/4245393736.py in <cell line: 0>()
      9     )
     10 )
---> 11 ttrain_X = imp.transform(ttrain_X)
     12 output = regr.predict(ttrain_X)
     13 

/usr/local/lib/python3.11/dist-packages/sklearn/utils/_set_output.py in wrapped(self, X, *args, **kwargs)
    138     @wraps(f)
    139     def wrapped(self, X, *args, **kwargs):
--> 140         data_to_wrap = f(self, X, *args, **kwargs)
    141         if isinstance(data_to_wrap, tuple):
    142             # only wrap the first output for cross decomposition

/usr/local/lib/python3.11/dist-packages/sklearn/impute/_base.py in transform(self, X)
    547             `X` with imputed values.
    548         """
--> 549         check_is_fitted(self)
    550 
    551         X = self._validate_input(X, in_fit=False)

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in check_is_fitted(estimator, attributes, msg, all_or_any)
   1388 
   1389     if not fitted:
-> 1390         raise NotFittedError(msg % {"name": type(estimator).__name__})
   1391 
   1392 

NotFittedError: This SimpleImputer instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.

## === cell 10
my_submission = pd.DataFrame({"key": tdf_sorted.key.values, "fare_amount": output})
my_submission.to_csv("submission.csv", index=False)
print(my_submission.head())
print("Wrote submission.csv with shape:", my_submission.shape)

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3504799084.py in <cell line: 0>()
----> 1 my_submission = pd.DataFrame({"key": tdf_sorted.key.values, "fare_amount": output})
      2 my_submission.to_csv("submission.csv", index=False)
      3 print(my_submission.head())
      4 print("Wrote submission.csv with shape:", my_submission.shape)

NameError: name 'output' is not defined
