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

3.10

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
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1
xgboost==2.0.3

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

3.41747

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 5.29775) has done: 'I remove the TensorFlow imports that cause the protobuf error and replace the neural‑network model with an XGBoost regressor, which works with the existing pre‑processed features and scaling. This fixes the runtime crash, enables training, and should bring the RMS error much closer to the target while keeping the rest of the preprocessing pipeline unchanged.'
- What this solution (achieved 5.34524) has done: 'I add a simple Manhattan distance feature, include it in the feature list, drop the unnecessary StandardScaler (tree models don’t need scaling), and modestly increase the XGBRegressor capacity so the model can better capture patterns. These tweaks keep the overall pipeline unchanged while expected to lower the RMS error toward the target.'

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (16, 8)
import seaborn as sns

from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error



## === cell 1
train_df = pd.read_csv(
    "../input/new-york-city-taxi-fare-prediction/train.csv", nrows=1000000
)
test_df = pd.read_csv("../input/new-york-city-taxi-fare-prediction/test.csv")




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/1476743133.py in <cell line: 0>()
----> 1 train_df = pd.read_csv(
      2     "../input/new-york-city-taxi-fare-prediction/train.csv", nrows=1000000
      3 )
      4 test_df = pd.read_csv("../input/new-york-city-taxi-fare-prediction/test.csv")
      5 

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
   1878                 if "b" not in mode:
   1879                     mode += "b"
-> 1880             self.handles = get_handle(
   1881                 f,
   1882                 mode,

/usr/local/lib/python3.11/dist-packages/pandas/io/common.py in get_handle(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)
    871         if ioargs.encoding and "b" not in ioargs.mode:
    872             # Encoding
--> 873             handle = open(
    874                 handle,
    875                 ioargs.mode,

FileNotFoundError: [Errno 2] No such file or directory: '../input/new-york-city-taxi-fare-prediction/train.csv'

## === cell 2
train_df.isnull().sum()




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2639615703.py in <cell line: 0>()
----> 1 train_df.isnull().sum()
      2 
      3 

NameError: name 'train_df' is not defined

## === cell 3
train_df.dropna(axis=0, subset=["dropoff_longitude", "dropoff_latitude"], inplace=True)
train_df = train_df.reset_index(drop=True)




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1356006980.py in <cell line: 0>()
----> 1 train_df.dropna(axis=0, subset=["dropoff_longitude", "dropoff_latitude"], inplace=True)
      2 train_df = train_df.reset_index(drop=True)
      3 
      4 

NameError: name 'train_df' is not defined

## === cell 4
pd.set_option("display.float_format", lambda x: "%.5f" % x)
train_df.describe()




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2250610215.py in <cell line: 0>()
      1 pd.set_option("display.float_format", lambda x: "%.5f" % x)
----> 2 train_df.describe()
      3 
      4 

NameError: name 'train_df' is not defined

## === cell 5
print("Number of observations out of valid range in coordinate columns:", end="\n")
print("pickup_longitude", end=": ")
print(
    (train_df.pickup_longitude < -180).sum() + (train_df.pickup_longitude > 180).sum()
)
print("pickup_latitude", end=": ")
print((train_df.pickup_latitude < -90).sum() + (train_df.pickup_latitude > 90).sum())
print("dropoff_longitude", end=": ")
print(
    (train_df.dropoff_longitude < -180).sum() + (train_df.dropoff_longitude > 180).sum()
)
print("dropoff_latitude", end=": ")
print((train_df.dropoff_latitude < -90).sum() + (train_df.dropoff_latitude > 90).sum())




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/697398380.py in <cell line: 0>()
      2 print("pickup_longitude", end=": ")
      3 print(
----> 4     (train_df.pickup_longitude < -180).sum() + (train_df.pickup_longitude > 180).sum()
      5 )
      6 print("pickup_latitude", end=": ")

NameError: name 'train_df' is not defined

## === cell 6
train_df = train_df.drop(
    train_df[
        (train_df.pickup_longitude < -180) | (train_df.pickup_longitude > 180)
    ].index,
    axis=0,
)
train_df = train_df.drop(
    train_df[(train_df.pickup_latitude < -90) | (train_df.pickup_latitude > 90)].index,
    axis=0,
)
train_df = train_df.drop(
    train_df[
        (train_df.dropoff_longitude < -180) | (train_df.dropoff_longitude > 180)
    ].index,
    axis=0,
)
train_df = train_df.drop(
    train_df[
        (train_df.dropoff_latitude < -90) | (train_df.dropoff_latitude > 90)
    ].index,
    axis=0,
)




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1632762843.py in <cell line: 0>()
----> 1 train_df = train_df.drop(
      2     train_df[
      3         (train_df.pickup_longitude < -180) | (train_df.pickup_longitude > 180)
      4     ].index,
      5     axis=0,

NameError: name 'train_df' is not defined

## === cell 7
train_df.describe()




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/911278541.py in <cell line: 0>()
----> 1 train_df.describe()
      2 
      3 

NameError: name 'train_df' is not defined

## === cell 8
train_df[(train_df.pickup_longitude >= 40)]




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1353903298.py in <cell line: 0>()
----> 1 train_df[(train_df.pickup_longitude >= 40)]
      2 
      3 

NameError: name 'train_df' is not defined

## === cell 9
indx = train_df[(train_df.pickup_longitude >= 40)].index
train_df.loc[indx, ["dropoff_longitude", "dropoff_latitude"]] = train_df.loc[
    indx, ["dropoff_latitude", "dropoff_longitude"]
].values
train_df.loc[indx, ["pickup_longitude", "pickup_latitude"]] = train_df.loc[
    indx, ["pickup_latitude", "pickup_longitude"]
].values




## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/375527993.py in <cell line: 0>()
----> 1 indx = train_df[(train_df.pickup_longitude >= 40)].index
      2 train_df.loc[indx, ["dropoff_longitude", "dropoff_latitude"]] = train_df.loc[
      3     indx, ["dropoff_latitude", "dropoff_longitude"]
      4 ].values
      5 train_df.loc[indx, ["pickup_longitude", "pickup_latitude"]] = train_df.loc[

NameError: name 'train_df' is not defined

## === cell 10
train_df = train_df.drop(
    train_df[
        (train_df.pickup_longitude < -75) | (train_df.pickup_longitude > -72)
    ].index,
    axis=0,
)
train_df = train_df.drop(
    train_df[
        (train_df.dropoff_longitude < -75) | (train_df.dropoff_longitude > -72)
    ].index,
    axis=0,
)
train_df = train_df.drop(
    train_df[(train_df.pickup_latitude < 40) | (train_df.pickup_latitude > 42)].index,
    axis=0,
)
train_df = train_df.drop(
    train_df[(train_df.dropoff_latitude < 40) | (train_df.dropoff_latitude > 42)].index,
    axis=0,
)




## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2406975378.py in <cell line: 0>()
----> 1 train_df = train_df.drop(
      2     train_df[
      3         (train_df.pickup_longitude < -75) | (train_df.pickup_longitude > -72)
      4     ].index,
      5     axis=0,

NameError: name 'train_df' is not defined

## === cell 11
train_df.describe()




## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/911278541.py in <cell line: 0>()
----> 1 train_df.describe()
      2 
      3 

NameError: name 'train_df' is not defined

## === cell 12
train_df.passenger_count.value_counts()




## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1112773242.py in <cell line: 0>()
----> 1 train_df.passenger_count.value_counts()
      2 
      3 

NameError: name 'train_df' is not defined

## === cell 13
train_df = train_df.drop(train_df[train_df.passenger_count == 0].index, axis=0)




## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3815171884.py in <cell line: 0>()
----> 1 train_df = train_df.drop(train_df[train_df.passenger_count == 0].index, axis=0)
      2 
      3 

NameError: name 'train_df' is not defined

## === cell 14
train_df.fare_amount.sort_values(ascending=False)




## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2229512276.py in <cell line: 0>()
----> 1 train_df.fare_amount.sort_values(ascending=False)
      2 
      3 

NameError: name 'train_df' is not defined

## === cell 15
train_df = train_df.drop(train_df[train_df.fare_amount <= 0].index, axis=0)
train_df["fare_amount"].sort_values(ascending=False)




## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4140262451.py in <cell line: 0>()
----> 1 train_df = train_df.drop(train_df[train_df.fare_amount <= 0].index, axis=0)
      2 train_df["fare_amount"].sort_values(ascending=False)
      3 
      4 

NameError: name 'train_df' is not defined

## === cell 16
test_df.isna().sum()




## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2363194077.py in <cell line: 0>()
----> 1 test_df.isna().sum()
      2 
      3 

NameError: name 'test_df' is not defined

## === cell 17
test_df.describe()




## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2532408863.py in <cell line: 0>()
----> 1 test_df.describe()
      2 
      3 

NameError: name 'test_df' is not defined

## === cell 18
train_df.dtypes




## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2615532969.py in <cell line: 0>()
----> 1 train_df.dtypes
      2 
      3 

NameError: name 'train_df' is not defined

## === cell 19
train_df["pickup_datetime"] = pd.to_datetime(train_df["pickup_datetime"])
test_df["pickup_datetime"] = pd.to_datetime(test_df["pickup_datetime"])




## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/403545854.py in <cell line: 0>()
----> 1 train_df["pickup_datetime"] = pd.to_datetime(train_df["pickup_datetime"])
      2 test_df["pickup_datetime"] = pd.to_datetime(test_df["pickup_datetime"])
      3 
      4 

NameError: name 'train_df' is not defined

## === cell 20
def date_splitter(df):
    df["Year"] = df["pickup_datetime"].dt.year
    df["Month"] = df["pickup_datetime"].dt.month
    df["Day"] = df["pickup_datetime"].dt.day
    df["Weekday"] = df["pickup_datetime"].dt.dayofweek
    df["Hour"] = df["pickup_datetime"].dt.hour


date_splitter(train_df)
date_splitter(test_df)
train_df.drop(["pickup_datetime"], axis=1, inplace=True)
test_df.drop(["pickup_datetime"], axis=1, inplace=True)




## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2219508502.py in <cell line: 0>()
      7 
      8 
----> 9 date_splitter(train_df)
     10 date_splitter(test_df)
     11 train_df.drop(["pickup_datetime"], axis=1, inplace=True)

NameError: name 'train_df' is not defined

## === cell 21
import math


def haversine_distance(df):
    coord = [
        "pickup_latitude",
        "pickup_longitude",
        "dropoff_latitude",
        "dropoff_longitude",
    ]
    phi1, lambda1, phi2, lambda2 = [df[i] * math.pi / 180.0 for i in coord]
    R = 6371
    dPhi = phi2 - phi1
    dLambda = lambda2 - lambda1
    a = (
        np.sin(dPhi / 2.0) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(dLambda / 2.0) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    d = R * c
    df["Distance"] = d
    df["ManhattanDist"] = np.abs(
        df["pickup_latitude"] - df["dropoff_latitude"]
    ) + np.abs(df["pickup_longitude"] - df["dropoff_longitude"])


haversine_distance(train_df)
haversine_distance(test_df)




## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1279153460.py in <cell line: 0>()
     25 
     26 
---> 27 haversine_distance(train_df)
     28 haversine_distance(test_df)
     29 

NameError: name 'train_df' is not defined

## === cell 22
train_df.Distance.sort_values()




## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1934675045.py in <cell line: 0>()
----> 1 train_df.Distance.sort_values()
      2 
      3 

NameError: name 'train_df' is not defined

## === cell 23
train_df = train_df.drop(train_df[train_df.Distance < 0.5].index, axis=0)




## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/878006848.py in <cell line: 0>()
----> 1 train_df = train_df.drop(train_df[train_df.Distance < 0.5].index, axis=0)
      2 
      3 

NameError: name 'train_df' is not defined

## === cell 24
f, axes = plt.subplots(1, 2)
sns.barplot(x="passenger_count", y="fare_amount", data=train_df, ax=axes[0])
sns.scatterplot(x="passenger_count", y="fare_amount", data=train_df, ax=axes[1])




## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2246989474.py in <cell line: 0>()
      1 f, axes = plt.subplots(1, 2)
----> 2 sns.barplot(x="passenger_count", y="fare_amount", data=train_df, ax=axes[0])
      3 sns.scatterplot(x="passenger_count", y="fare_amount", data=train_df, ax=axes[1])
      4 
      5 

NameError: name 'train_df' is not defined

## === cell 25
f, axes = plt.subplots(1, 2)
sns.barplot(x="Year", y="fare_amount", data=train_df, ax=axes[0])
sns.scatterplot(x="Year", y="fare_amount", data=train_df, ax=axes[1])




## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2064762055.py in <cell line: 0>()
      1 f, axes = plt.subplots(1, 2)
----> 2 sns.barplot(x="Year", y="fare_amount", data=train_df, ax=axes[0])
      3 sns.scatterplot(x="Year", y="fare_amount", data=train_df, ax=axes[1])
      4 
      5 

NameError: name 'train_df' is not defined

## === cell 26
f, axes = plt.subplots(1, 2)
sns.barplot(x="Month", y="fare_amount", data=train_df, ax=axes[0])
sns.scatterplot(x="Month", y="fare_amount", data=train_df, ax=axes[1])




## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3348318888.py in <cell line: 0>()
      1 f, axes = plt.subplots(1, 2)
----> 2 sns.barplot(x="Month", y="fare_amount", data=train_df, ax=axes[0])
      3 sns.scatterplot(x="Month", y="fare_amount", data=train_df, ax=axes[1])
      4 
      5 

NameError: name 'train_df' is not defined

## === cell 27
f, axes = plt.subplots(1, 2)
sns.barplot(x="Day", y="fare_amount", data=train_df, ax=axes[0])
sns.scatterplot(x="Day", y="fare_amount", data=train_df, ax=axes[1])




## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3834160633.py in <cell line: 0>()
      1 f, axes = plt.subplots(1, 2)
----> 2 sns.barplot(x="Day", y="fare_amount", data=train_df, ax=axes[0])
      3 sns.scatterplot(x="Day", y="fare_amount", data=train_df, ax=axes[1])
      4 
      5 

NameError: name 'train_df' is not defined

## === cell 28
f, axes = plt.subplots(1, 2)
sns.barplot(x="Weekday", y="fare_amount", data=train_df, ax=axes[0])
sns.scatterplot(x="Weekday", y="fare_amount", data=train_df, ax=axes[1])




## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2623932136.py in <cell line: 0>()
      1 f, axes = plt.subplots(1, 2)
----> 2 sns.barplot(x="Weekday", y="fare_amount", data=train_df, ax=axes[0])
      3 sns.scatterplot(x="Weekday", y="fare_amount", data=train_df, ax=axes[1])
      4 
      5 

NameError: name 'train_df' is not defined

## === cell 29
f, axes = plt.subplots(1, 2)
sns.barplot(x="Hour", y="fare_amount", data=train_df, ax=axes[0])
sns.scatterplot(x="Hour", y="fare_amount", data=train_df, ax=axes[1])




## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/543036698.py in <cell line: 0>()
      1 f, axes = plt.subplots(1, 2)
----> 2 sns.barplot(x="Hour", y="fare_amount", data=train_df, ax=axes[0])
      3 sns.scatterplot(x="Hour", y="fare_amount", data=train_df, ax=axes[1])
      4 
      5 

NameError: name 'train_df' is not defined

## === cell 30
train_df["DistanceGroups"] = pd.qcut(train_df["Distance"], 10)




## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1767556939.py in <cell line: 0>()
----> 1 train_df["DistanceGroups"] = pd.qcut(train_df["Distance"], 10)
      2 
      3 

NameError: name 'train_df' is not defined

## === cell 31
f, axes = plt.subplots(1, 2)
plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=70)
sns.barplot(x="DistanceGroups", y="fare_amount", data=train_df, ax=axes[0])
sns.scatterplot(x="Distance", y="fare_amount", data=train_df, ax=axes[1])




## --- ERROR in cell 31, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2545151100.py in <cell line: 0>()
      1 f, axes = plt.subplots(1, 2)
      2 plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=70)
----> 3 sns.barplot(x="DistanceGroups", y="fare_amount", data=train_df, ax=axes[0])
      4 sns.scatterplot(x="Distance", y="fare_amount", data=train_df, ax=axes[1])
      5 

NameError: name 'train_df' is not defined

## === cell 32
train_df.columns




## --- ERROR in cell 32, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2029752390.py in <cell line: 0>()
----> 1 train_df.columns
      2 
      3 

NameError: name 'train_df' is not defined

## === cell 33
features = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
    "Year",
    "Month",
    "Day",
    "Weekday",
    "Hour",
    "Distance",
    "ManhattanDist",
]
outcome = "fare_amount"




## === cell 34
X_train, X_test, y_train, y_test = train_test_split(
    train_df[features],
    train_df[outcome],
    test_size=0.30,
    random_state=42,
)

y_train_log = np.log1p(y_train)
y_test_log = np.log1p(y_test)




## --- ERROR in cell 34, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2093500161.py in <cell line: 0>()
      1 # Split data
      2 X_train, X_test, y_train, y_test = train_test_split(
----> 3     train_df[features],
      4     train_df[outcome],
      5     test_size=0.30,

NameError: name 'train_df' is not defined

## === cell 35
scaled_train = X_train
scaled_valid = X_test
scaled_test = test_df[features]




## --- ERROR in cell 35, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3584331969.py in <cell line: 0>()
----> 1 scaled_train = X_train
      2 scaled_valid = X_test
      3 scaled_test = test_df[features]
      4 
      5 

NameError: name 'X_train' is not defined

## === cell 36
model = XGBRegressor(
    n_estimators=1200,  # allow more trees for the transformed target
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    n_jobs=5,
    random_state=42,
    reg_lambda=1.0,
)
model.fit(
    scaled_train,
    y_train_log,
    eval_set=[(scaled_valid, y_test_log)],
    early_stopping_rounds=10,
    verbose=False,
)




## --- ERROR in cell 36, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2761095335.py in <cell line: 0>()
     12 )
     13 model.fit(
---> 14     scaled_train,
     15     y_train_log,
     16     eval_set=[(scaled_valid, y_test_log)],

NameError: name 'scaled_train' is not defined

## === cell 37
prediction_log = model.predict(scaled_test)
prediction = np.expm1(prediction_log)  # back to original fare scale
submission = pd.DataFrame({"key": test_df["key"], "fare_amount": prediction})
submission.to_csv("taxi_fare_submission.csv", index=False)

## --- ERROR in cell 37, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1343843120.py in <cell line: 0>()
      1 # Predict and reverse the log‑transform
----> 2 prediction_log = model.predict(scaled_test)
      3 prediction = np.expm1(prediction_log)  # back to original fare scale
      4 submission = pd.DataFrame({"key": test_df["key"], "fare_amount": prediction})
      5 submission.to_csv("taxi_fare_submission.csv", index=False)

NameError: name 'scaled_test' is not defined
