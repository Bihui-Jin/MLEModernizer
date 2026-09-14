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

5.48885

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 800.26126) has done: 'The changes enable Intel‑optimized scikit‑learn (sklearnex) and parallel execution of the RandomForest, which dramatically cuts training time without altering the model’s hyper‑parameters or prediction logic. Only the import of `sklearnex` and the `n_jobs=-1` argument are added; all other code and data handling remain identical, preserving exact results while staying under the 600‑second limit.'
- What this solution (achieved 16.9774) has done: 'The fix adds a safe import for the Intel‑optimized scikit‑learn patch (using a try/except fallback) so the notebook can run, and clips model predictions to non‑negative values to avoid wildly erroneous fares that hurt RMSE. No core modeling logic is changed.'
- What this solution (achieved 16.9774) has done: 'I keep the existing cleaning, feature engineering, and model code unchanged and add a new, simple distance‑based prediction rule (fare ≈ 2.5 + 1.56 × H_Distance) which is known to be a strong baseline for this dataset. This rule is applied after the models are trained, and its predictions are written to a third submission file (`submission_3.csv`). Using this deterministic estimate should move the RMSE much closer to the target score while preserving the core workflow.'
- What this solution (achieved 797.51326) has done: 'I replace the simple distance‑based rule with a tiny linear fit of fare ≈ a + b·H_Distance computed on the cleaned training data. This keeps the same features and preprocessing, only calibrates the two coefficients to better match the target RMSE, and writes the final predictions to submission.csv (the required output file).'

# 9. Code solution

## === cell 0
import pandas as pd, numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression



## === cell 1
dtypes = {
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int8",
}
parse_dates = ["pickup_datetime"]
train = pd.read_csv(
    "../input/train.csv",
    usecols=list(dtypes.keys()) + ["pickup_datetime"],
    dtype=dtypes,
    parse_dates=parse_dates,
    infer_datetime_format=True,
    memory_map=True,
    nrows=500_000,  # reasonable sample for quick training
)
test = pd.read_csv(
    "../input/test.csv",
    usecols=list(dtypes.keys()) + ["pickup_datetime"],
    dtype=dtypes,
    parse_dates=parse_dates,
    infer_datetime_format=True,
    memory_map=True,
)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2340312181.py in <cell line: 0>()
     18     nrows=500_000,  # reasonable sample for quick training
     19 )
---> 20 test = pd.read_csv(
     21     "../input/test.csv",
     22     usecols=list(dtypes.keys()) + ["pickup_datetime"],

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

ValueError: Usecols do not match columns, columns expected but not found: ['fare_amount']

## === cell 2
train.dropna(inplace=True)
train = train[train["fare_amount"] >= 0]
train = train[train["passenger_count"] <= 8]

lat_cond = (train["pickup_latitude"].between(-90, 90)) & (
    train["dropoff_latitude"].between(-90, 90)
)
lon_cond = (train["pickup_longitude"].between(-180, 180)) & (
    train["dropoff_longitude"].between(-180, 180)
)
train = train[lat_cond & lon_cond]




## === cell 3
def haversine_distance(df):
    r = 6371.0
    phi1 = np.radians(df["pickup_latitude"].values)
    phi2 = np.radians(df["dropoff_latitude"].values)
    dphi = np.radians(df["dropoff_latitude"].values - df["pickup_latitude"].values)
    dlambda = np.radians(df["dropoff_longitude"].values - df["pickup_longitude"].values)
    a = (
        np.sin(dphi / 2.0) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return r * c


train["H_Distance"] = haversine_distance(train)
test["H_Distance"] = haversine_distance(test)



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1277818981.py in <cell line: 0>()
     14 
     15 train["H_Distance"] = haversine_distance(train)
---> 16 test["H_Distance"] = haversine_distance(test)
     17 

NameError: name 'test' is not defined

## === cell 4
for df in (train, test):
    dt = df["pickup_datetime"].dt
    df["Year"] = dt.year
    df["Month"] = dt.month
    df["Date"] = dt.day
    df["Day_of_Week"] = dt.dayofweek
    df["Hour"] = dt.hour



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2575303785.py in <cell line: 0>()
----> 1 for df in (train, test):
      2     dt = df["pickup_datetime"].dt
      3     df["Year"] = dt.year
      4     df["Month"] = dt.month
      5     df["Date"] = dt.day

NameError: name 'test' is not defined

## === cell 5
baseline_fare = 2.5 + 1.56 * test["H_Distance"]
baseline_fare = np.clip(baseline_fare, 0, None)



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/232083141.py in <cell line: 0>()
      1 # Apply simple distance‑based fare rule (baseline known to be strong)
----> 2 baseline_fare = 2.5 + 1.56 * test["H_Distance"]
      3 baseline_fare = np.clip(baseline_fare, 0, None)
      4 

NameError: name 'test' is not defined

## === cell 6
submission = pd.read_csv("../input/sample_submission.csv")
submission["fare_amount"] = baseline_fare
submission.to_csv("submission.csv", index=False)

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2488092598.py in <cell line: 0>()
      1 # Write final submission using the baseline predictions
      2 submission = pd.read_csv("../input/sample_submission.csv")
----> 3 submission["fare_amount"] = baseline_fare
      4 submission.to_csv("submission.csv", index=False)

NameError: name 'baseline_fare' is not defined
