# Goal

You will receive environment details and a partial notebook export.

# Requirements

- Fix the bug that causes the error in cell k.
- Do NOT adjust any other non-buggy cells.
- You may reference cell k+1 only to preserve variable/interface compatibility.
- Do not complete or extend code logic in cell k, k+1, or later cells.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (bug fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Output must follow your strict format: Diagnosis / Patch summary / Updated cells / Compatibility notes for cell k+1 / Assumptions.


# 1. Python version

3.7

# 2. Installed packages

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

# 3. Data file paths

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

# 4. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from sklearn import metrics  # evaluating models
from sklearn.model_selection import train_test_split  # set splitting and validation
from sklearn.linear_model import LinearRegression
import xgboost as xgb  # XGBoost regressor
import matplotlib.pyplot as plt  # plotting
import seaborn as sns  # plotting
import warnings

warnings.filterwarnings("ignore")

try:
    get_ipython().run_line_magic("matplotlib", "inline")
except Exception:
    pass

np.random.seed(42)

DO_PLOTS = False



## === cell 1
print(os.listdir("../input"))



## === cell 2
types = {
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "uint8",
}
test = pd.read_csv(
    "../input/test.csv",
    dtype={k: v for k, v in types.items() if k != "fare_amount"},
)



## === cell 3
test.dtypes



## === cell 4
types = {
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "uint8",
}




## === cell 5
def fast_sample_csv_exact_n(
    path,
    n,
    dtype,
    seed=42,
    usecols=None,
):
    with open(path, "rb") as f:
        total_lines = sum(1 for _ in f)
    nrows_total = max(total_lines - 1, 0)
    if n >= nrows_total:
        return pd.read_csv(path, dtype=dtype, usecols=usecols)

    rng = np.random.default_rng(seed)

    keep = rng.choice(nrows_total, size=n, replace=False)
    keep.sort()

    keep_ptr = 0
    keep_len = len(keep)
    skip = [
        0
    ]  # don't treat header as data; include 0 to ensure header is read correctly when header=0
    for data_idx in range(nrows_total):
        if keep_ptr < keep_len and data_idx == keep[keep_ptr]:
            keep_ptr += 1
        else:
            skip.append(data_idx + 1)

    df = pd.read_csv(path, dtype=dtype, usecols=usecols, skiprows=skip)
    return df


train = fast_sample_csv_exact_n(
    "../input/train.csv",
    n=500000,
    dtype=types,
    seed=42,
)



## === cell 6
train.head()



## === cell 7
train.describe()



## === cell 8
if DO_PLOTS:
    sns.distplot(train["fare_amount"])



## === cell 9
if DO_PLOTS:
    sns.distplot(train["passenger_count"])



## === cell 10
train.isnull().sum()



## === cell 11
train.dropna(inplace=True)



## === cell 12
required_cols = [
    "fare_amount",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]

if (
    "train" not in globals()
    or not isinstance(train, pd.DataFrame)
    or any(c not in train.columns for c in required_cols)
):
    try:
        train = fast_sample_csv_exact_n(
            "../input/train.csv",
            n=500000,
            dtype=types,
            seed=42,
            usecols=required_cols,
        )
    except ValueError as e:
        if "Usecols do not match columns" not in str(e):
            raise
        with open("../input/train.csv", "rb") as f:
            total_lines = sum(1 for _ in f)
        nrows_total = max(total_lines - 1, 0)
        if 500000 >= nrows_total:
            train = pd.read_csv(
                "../input/train.csv",
                dtype=types,
                usecols=required_cols,
                engine="python",
            )
        else:
            rng = np.random.default_rng(42)
            keep = rng.choice(nrows_total, size=500000, replace=False)
            keep.sort()
            keep_ptr = 0
            keep_len = len(keep)
            skip = [0]
            for data_idx in range(nrows_total):
                if keep_ptr < keep_len and data_idx == keep[keep_ptr]:
                    keep_ptr += 1
                else:
                    skip.append(data_idx + 1)
            train = pd.read_csv(
                "../input/train.csv",
                dtype=types,
                usecols=required_cols,
                skiprows=skip,
                engine="python",
            )

train = train[(train["fare_amount"] > 0) & (train["fare_amount"] <= 250)]

train = train[train["pickup_longitude"] < -72]
train = train[(train["pickup_latitude"] > 40) & (train["pickup_latitude"] < 44)]
train = train[train["dropoff_longitude"] < -72]
train = train[(train["dropoff_latitude"] > 40) & (train["dropoff_latitude"] < 44)]
train = train[(train["passenger_count"] > 0) & (train["passenger_count"] < 10)]


## --- ERROR in cell 12, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mValueError[0m                                Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/1417780555.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     18[0m     [0;32mtry[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 19[0;31m         train = fast_sample_csv_exact_n(
[0m[1;32m     20[0m             [0;34m"../input/train.csv"[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/tmp/ipykernel_11/1537261691.py[0m in [0;36mfast_sample_csv_exact_n[0;34m(path, n, dtype, seed, usecols)[0m
[1;32m     41[0m [0;34m[0m[0m
[0;32m---> 42[0;31m     [0mdf[0m [0;34m=[0m [0mpd[0m[0;34m.[0m[0mread_csv[0m[0;34m([0m[0mpath[0m[0;34m,[0m [0mdtype[0m[0;34m=[0m[0mdtype[0m[0;34m,[0m [0musecols[0m[0;34m=[0m[0musecols[0m[0;34m,[0m [0mskiprows[0m[0;34m=[0m[0mskip[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     43[0m     [0;32mreturn[0m [0mdf[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py[0m in [0;36mread_csv[0;34m(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, date_format, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, encoding_errors, dialect, on_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options, dtype_backend)[0m
[1;32m   1025[0m [0;34m[0m[0m
[0;32m-> 1026[0;31m     [0;32mreturn[0m [0m_read[0m[0;34m([0m[0mfilepath_or_buffer[0m[0;34m,[0m [0mkwds[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1027[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py[0m in [0;36m_read[0;34m(filepath_or_buffer, kwds)[0m
[1;32m    619[0m     [0;31m# Create the parser.[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 620[0;31m     [0mparser[0m [0;34m=[0m [0mTextFileReader[0m[0;34m([0m[0mfilepath_or_buffer[0m[0;34m,[0m [0;34m**[0m[0mkwds[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    621[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py[0m in [0;36m__init__[0;34m(self, f, engine, **kwds)[0m
[1;32m   1619[0m         [0mself[0m[0;34m.[0m[0mhandles[0m[0;34m:[0m [0mIOHandles[0m [0;34m|[0m [0;32mNone[0m [0;34m=[0m [0;32mNone[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1620[0;31m         [0mself[0m[0;34m.[0m[0m_engine[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0m_make_engine[0m[0;34m([0m[0mf[0m[0;34m,[0m [0mself[0m[0;34m.[0m[0mengine[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1621[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py[0m in [0;36m_make_engine[0;34m(self, f, engine)[0m
[1;32m   1897[0m         [0;32mtry[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1898[0;31m             [0;32mreturn[0m [0mmapping[0m[0;34m[[0m[0mengine[0m[0;34m][0m[0;34m([0m[0mf[0m[0;34m,[0m [0;34m**[0m[0mself[0m[0;34m.[0m[0moptions[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1899[0m         [0;32mexcept[0m [0mException[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/c_parser_wrapper.py[0m in [0;36m__init__[0;34m(self, src, **kwds)[0m
[1;32m    139[0m             ):
[0;32m--> 140[0;31m                 [0mself[0m[0;34m.[0m[0m_validate_usecols_names[0m[0;34m([0m[0musecols[0m[0;34m,[0m [0mself[0m[0;34m.[0m[0morig_names[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    141[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/base_parser.py[0m in [0;36m_validate_usecols_names[0;34m(self, usecols, names)[0m
[1;32m    978[0m         [0;32mif[0m [0mlen[0m[0;34m([0m[0mmissing[0m[0;34m)[0m [0;34m>[0m [0;36m0[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 979[0;31m             raise ValueError(
[0m[1;32m    980[0m                 [0;34mf"Usecols do not match columns, columns expected but not found: "[0m[0;34m[0m[0;34m[0m[0m

[0;31mValueError[0m: Usecols do not match columns, columns expected but not found: ['pickup_longitude', 'fare_amount', 'dropoff_latitude', 'dropoff_longitude', 'passenger_count', 'pickup_latitude']

During handling of the above exception, another exception occurred:

[0;31mValueError[0m                                Traceback (most recent call last)
[0;32m/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/python_parser.py[0m in [0;36m_handle_usecols[0;34m(self, columns, usecols_key, num_original_columns)[0m
[1;32m    606[0m                         [0;32mtry[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 607[0;31m                             [0mcol_indices[0m[0;34m.[0m[0mappend[0m[0;34m([0m[0musecols_key[0m[0;34m.[0m[0mindex[0m[0;34m([0m[0mcol[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    608[0m                         [0;32mexcept[0m [0mValueError[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;31mValueError[0m: 'pickup_longitude' is not in list

During handling of the above exception, another exception occurred:

[0;31mValueError[0m                                Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/1417780555.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     49[0m                 [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m     50[0m                     [0mskip[0m[0;34m.[0m[0mappend[0m[0;34m([0m[0mdata_idx[0m [0;34m+[0m [0;36m1[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 51[0;31m             train = pd.read_csv(
[0m[1;32m     52[0m                 [0;34m"../input/train.csv"[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m     53[0m                 [0mdtype[0m[0;34m=[0m[0mtypes[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py[0m in [0;36mread_csv[0;34m(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, date_format, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, encoding_errors, dialect, on_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options, dtype_backend)[0m
[1;32m   1024[0m     [0mkwds[0m[0;34m.[0m[0mupdate[0m[0;34m([0m[0mkwds_defaults[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1025[0m [0;34m[0m[0m
[0;32m-> 1026[0;31m     [0;32mreturn[0m [0m_read[0m[0;34m([0m[0mfilepath_or_buffer[0m[0;34m,[0m [0mkwds[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1027[0m [0;34m[0m[0m
[1;32m   1028[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py[0m in [0;36m_read[0;34m(filepath_or_buffer, kwds)[0m
[1;32m    618[0m [0;34m[0m[0m
[1;32m    619[0m     [0;31m# Create the parser.[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 620[0;31m     [0mparser[0m [0;34m=[0m [0mTextFileReader[0m[0;34m([0m[0mfilepath_or_buffer[0m[0;34m,[0m [0;34m**[0m[0mkwds[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    621[0m [0;34m[0m[0m
[1;32m    622[0m     [0;32mif[0m [0mchunksize[0m [0;32mor[0m [0miterator[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py[0m in [0;36m__init__[0;34m(self, f, engine, **kwds)[0m
[1;32m   1618[0m [0;34m[0m[0m
[1;32m   1619[0m         [0mself[0m[0;34m.[0m[0mhandles[0m[0;34m:[0m [0mIOHandles[0m [0;34m|[0m [0;32mNone[0m [0;34m=[0m [0;32mNone[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1620[0;31m         [0mself[0m[0;34m.[0m[0m_engine[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0m_make_engine[0m[0;34m([0m[0mf[0m[0;34m,[0m [0mself[0m[0;34m.[0m[0mengine[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1621[0m [0;34m[0m[0m
[1;32m   1622[0m     [0;32mdef[0m [0mclose[0m[0;34m([0m[0mself[0m[0;34m)[0m [0;34m->[0m [0;32mNone[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py[0m in [0;36m_make_engine[0;34m(self, f, engine)[0m
[1;32m   1896[0m [0;34m[0m[0m
[1;32m   1897[0m         [0;32mtry[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1898[0;31m             [0;32mreturn[0m [0mmapping[0m[0;34m[[0m[0mengine[0m[0;34m][0m[0;34m([0m[0mf[0m[0;34m,[0m [0;34m**[0m[0mself[0m[0;34m.[0m[0moptions[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1899[0m         [0;32mexcept[0m [0mException[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1900[0m             [0;32mif[0m [0mself[0m[0;34m.[0m[0mhandles[0m [0;32mis[0m [0;32mnot[0m [0;32mNone[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/python_parser.py[0m in [0;36m__init__[0;34m(self, f, **kwds)[0m
[1;32m    131[0m             [0mself[0m[0;34m.[0m[0mnum_original_columns[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m    132[0m             [0mself[0m[0;34m.[0m[0munnamed_cols[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 133[0;31m         ) = self._infer_columns()
[0m[1;32m    134[0m [0;34m[0m[0m
[1;32m    135[0m         [0;31m# Now self.columns has the set of columns that we will process.[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/python_parser.py[0m in [0;36m_infer_columns[0;34m(self)[0m
[1;32m    540[0m                     [0mcolumns[0m [0;34m=[0m [0;34m[[0m[0mnames[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[1;32m    541[0m             [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 542[0;31m                 columns = self._handle_usecols(
[0m[1;32m    543[0m                     [0mcolumns[0m[0;34m,[0m [0mcolumns[0m[0;34m[[0m[0;36m0[0m[0;34m][0m[0;34m,[0m [0mnum_original_columns[0m[0;34m[0m[0;34m[0m[0m
[1;32m    544[0m                 )

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/python_parser.py[0m in [0;36m_handle_usecols[0;34m(self, columns, usecols_key, num_original_columns)[0m
[1;32m    607[0m                             [0mcol_indices[0m[0;34m.[0m[0mappend[0m[0;34m([0m[0musecols_key[0m[0;34m.[0m[0mindex[0m[0;34m([0m[0mcol[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    608[0m                         [0;32mexcept[0m [0mValueError[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 609[0;31m                             [0mself[0m[0;34m.[0m[0m_validate_usecols_names[0m[0;34m([0m[0mself[0m[0;34m.[0m[0musecols[0m[0;34m,[0m [0musecols_key[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    610[0m                     [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    611[0m                         [0mcol_indices[0m[0;34m.[0m[0mappend[0m[0;34m([0m[0mcol[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/base_parser.py[0m in [0;36m_validate_usecols_names[0;34m(self, usecols, names)[0m
[1;32m    977[0m         [0mmissing[0m [0;34m=[0m [0;34m[[0m[0mc[0m [0;32mfor[0m [0mc[0m [0;32min[0m [0musecols[0m [0;32mif[0m [0mc[0m [0;32mnot[0m [0;32min[0m [0mnames[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[1;32m    978[0m         [0;32mif[0m [0mlen[0m[0;34m([0m[0mmissing[0m[0;34m)[0m [0;34m>[0m [0;36m0[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 979[0;31m             raise ValueError(
[0m[1;32m    980[0m                 [0;34mf"Usecols do not match columns, columns expected but not found: "[0m[0;34m[0m[0;34m[0m[0m
[1;32m    981[0m                 [0;34mf"{missing}"[0m[0;34m[0m[0;34m[0m[0m

[0;31mValueError[0m: Usecols do not match columns, columns expected but not found: ['pickup_longitude', 'fare_amount', 'dropoff_latitude', 'dropoff_longitude', 'passenger_count', 'pickup_latitude']

## === cell 13
train.describe()
