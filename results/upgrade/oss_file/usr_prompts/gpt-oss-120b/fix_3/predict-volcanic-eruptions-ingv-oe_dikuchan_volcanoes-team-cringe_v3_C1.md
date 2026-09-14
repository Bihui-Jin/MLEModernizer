# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


# 1. Kaggle task description

## Overview
Given readings from several seismic sensors around a volcano, estimate how long it will be until the next eruption.

## Metric
Mean absolute error (MAE) between the predicted loss and the actual loss.

## Submission Format
For every id in the test set, you should predict the time until the next eruption. The file should contain a header and have the following format:

```
segment_id,time_to_eruption
1,1
2,2
3,3
etc.
```

## Data
### Dataset Description

#### Files
**train.csv** Metadata for the train files.

- `segment_id`: ID code for the data segment. Matches the name of the associated data file.
- `time_to_eruption`: The target value, the time until the next eruption.

**[train|test]/*.csv**: the data files. Each file contains ten minutes of logs from ten different sensors arrayed around a volcano. The readings have been normalized within each segment, in part to ensure that the readings fall within the range of int16 values. If you are using the Pandas library you may find that you still need to load the data as float32 due to the presence of some nulls.

# 2. Python version

3.9

# 3. Installed packages

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
optuna==4.5.0
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
tsfresh==0.21.0
xgboost==2.0.3

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (70 lines)
            sample_submission.csv (445 lines)
            sample_submission.csv.zip (2.8 kB)
            test.zip (514.3 MB)
            train.csv (3988 lines)
            train.csv.zip (39.1 kB)
            train.zip (4.6 GB)
            predict-volcanic-eruptions-ingv-oe/
                description.md (70 lines)
                sample_submission.csv (445 lines)
                ... and 5 other files
                predict-volcanic-eruptions-ingv-oe/
                test/
                    1003520023.csv (60002 lines)
                    1004346803.csv (60002 lines)
                    ... and 442 other files
                    test/
                train/
                    1000015382.csv (60002 lines)
                    1000554676.csv (60002 lines)
                    ... and 3985 other files
                    train/
            test/
                1003520023.csv (60002 lines)
                1004346803.csv (60002 lines)
                ... and 442 other files
                test/
            train/
                1000015382.csv (60002 lines)
                1000554676.csv (60002 lines)
                ... and 3985 other files
                train/
        input/
            description.md (70 lines)
            sample_submission.csv (445 lines)
            sample_submission.csv.zip (2.8 kB)
            test.zip (514.3 MB)
            train.csv (3988 lines)
            train.csv.zip (39.1 kB)
            train.zip (4.6 GB)
            predict-volcanic-eruptions-ingv-oe/
                description.md (70 lines)
                sample_submission.csv (445 lines)
                ... and 5 other files
                predict-volcanic-eruptions-ingv-oe/
                test/
                    1003520023.csv (60002 lines)
                    1004346803.csv (60002 lines)
                    ... and 442 other files
                    test/
                train/
                    1000015382.csv (60002 lines)
                    1000554676.csv (60002 lines)
                    ... and 3985 other files
                    train/
            test/
                1003520023.csv (60002 lines)
                1004346803.csv (60002 lines)
                ... and 442 other files
                test/
                    1003520023.csv (60002 lines)
                    1004346803.csv (60002 lines)
                    ... and 442 other files
                    test/
            train/
                1000015382.csv (60002 lines)
                1000554676.csv (60002 lines)
                ... and 3985 other files
                train/
                    1000015382.csv (60002 lines)
                    1000554676.csv (60002 lines)
                    ... and 3985 other files
                    train/
        working/
            predict-volcanic-eruptions-ingv-oe/
                description.md (70 lines)
                sample_submission.csv (445 lines)
                ... and 5 other files
                predict-volcanic-eruptions-ingv-oe/
                test/
                    1003520023.csv (60002 lines)
                    1004346803.csv (60002 lines)
                    ... and 442 other files
                    test/
                train/
                    1000015382.csv (60002 lines)
                    1000554676.csv (60002 lines)
                    ... and 3985 other files
                    train/
```

-> data/predict-volcanic-eruptions-ingv-oe/sample_submission.csv has 444 rows and 2 columns.
The columns are: segment_id, time_to_eruption

-> data/predict-volcanic-eruptions-ingv-oe/test/1003520023.csv has 60001 rows and 10 columns.
The columns are: sensor_1, sensor_2, sensor_3, sensor_4, sensor_5, sensor_6, sensor_7, sensor_8, sensor_9, sensor_10

-> data/predict-volcanic-eruptions-ingv-oe/test/1004346803.csv has 60001 rows and 10 columns.
The columns are: sensor_1, sensor_2, sensor_3, sensor_4, sensor_5, sensor_6, sensor_7, sensor_8, sensor_9, sensor_10

-> data/predict-volcanic-eruptions-ingv-oe/test/1007996426.csv has 60001 rows and 10 columns.
The columns are: sensor_1, sensor_2, sensor_3, sensor_4, sensor_5, sensor_6, sensor_7, sensor_8, sensor_9, sensor_10

-> data/predict-volcanic-eruptions-ingv-oe/test/1009749143.csv has 60001 rows and 10 columns.
The columns are: sensor_1, sensor_2, sensor_3, sensor_4, sensor_5, sensor_6, sensor_7, sensor_8, sensor_9, sensor_10

-> data/predict-volcanic-eruptions-ingv-oe/test/1016956864.csv has 60001 rows and 10 columns.
The columns are: sensor_1, sensor_2, sensor_3, sensor_4, sensor_5, sensor_6, sensor_7, sensor_8, sensor_9, sensor_10

-> data/predict-volcanic-eruptions-ingv-oe/test/1024522044.csv has 60001 rows and 10 columns.
The columns are: sensor_1, sensor_2, sensor_3, sensor_4, sensor_5, sensor_6, sensor_7, sensor_8, sensor_9, sensor_10

-> data/predict-volcanic-eruptions-ingv-oe/test/1028325789.csv has 60001 rows and 10 columns.
The columns are: sensor_1, sensor_2, sensor_3, sensor_4, sensor_5, sensor_6, sensor_7, sensor_8, sensor_9, sensor_10

-> (stopped after 10 files for performance)

# 5. Target score

5176291.478340709

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import TransformerMixin
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error

from xgboost import XGBRegressor
from tsfresh import extract_features
from tsfresh.feature_extraction import MinimalFCParameters




## === cell 1
data_folder = Path("../input/predict-volcanic-eruptions-ingv-oe/")




## === cell 2
train_meta = pd.read_csv(data_folder / "train.csv")
train_meta.head()




## === cell 3
tsfresh_parameters = MinimalFCParameters()
del tsfresh_parameters["length"]
tsfresh_parameters["skewness"] = None
tsfresh_parameters["kurtosis"] = None
tsfresh_parameters["last_location_of_maximum"] = None
tsfresh_parameters["first_location_of_maximum"] = None
tsfresh_parameters["last_location_of_minimum"] = None
tsfresh_parameters["first_location_of_minimum"] = None
tsfresh_parameters["benford_correlation"] = None
tsfresh_parameters["percentage_of_reoccurring_values_to_all_values"] = None
tsfresh_parameters["percentage_of_reoccurring_datapoints_to_all_datapoints"] = None
tsfresh_parameters["number_peaks"] = [
    {"n": 1},
    {"n": 3},
    {"n": 5},
    {"n": 10},
    {"n": 50},
]
tsfresh_parameters["binned_entropy"] = [{"max_bins": 10}]
tsfresh_parameters["fft_aggregated"] = [
    {"aggtype": "centroid"},
    {"aggtype": "variance"},
    {"aggtype": "skew"},
    {"aggtype": "kurtosis"},
]
tsfresh_parameters["autocorrelation"] = [{"lag": i} for i in range(10)]
tsfresh_parameters["agg_autocorrelation"] = [
    {"f_agg": "mean", "maxlag": 40},
    {"f_agg": "median", "maxlag": 40},
    {"f_agg": "var", "maxlag": 40},
]
tsfresh_parameters["friedrich_coefficients"] = [
    {"coeff": c, "m": 3, "r": 30} for c in range(4)
]
tsfresh_parameters["count_above"] = [{"t": 0}]
tsfresh_parameters["count_below"] = [{"t": 0}]




## === cell 4
def preprocess_timeseries(meta_df, parameters, is_train=True, batch_size=200):
    """
    Efficiently extract TSFresh features for all segments.
    - For training, meta_df provides the target values.
    - For test, meta_df is ignored.
    The function processes files in batches to limit memory usage
    and uses TSFresh's built‑in parallelism (n_jobs=-1).
    Returns a DataFrame with one row per segment, retaining the same
    columns as the original implementation.
    """
    if is_train:
        segments = meta_df["segment_id"].astype(str).tolist()
        target_map = meta_df.set_index("segment_id")["time_to_eruption"]
    else:
        segments = sorted(os.listdir(data_folder / "test"))
        target_map = None

    results = []
    for start in range(0, len(segments), batch_size):
        batch = segments[start : start + batch_size]
        batch_frames = []
        for seg in batch:
            csv_path = data_folder / ("train" if is_train else "test") / f"{seg}.csv"
            ts = pd.read_csv(csv_path, dtype=np.float32).fillna(0)
            ts["segment"] = seg
            batch_frames.append(ts)
        batch_df = pd.concat(batch_frames, ignore_index=True)

        feats = extract_features(
            batch_df,
            column_id="segment",
            default_fc_parameters=parameters,
            disable_progressbar=True,
            n_jobs=-1,
        )
        feats["segment"] = batch_df["segment"].unique()
        if is_train:
            feats["time_to_eruption"] = feats["segment"].map(target_map)
        results.append(feats)

    return pd.concat(results, ignore_index=True)




## === cell 5
df_train = preprocess_timeseries(train_meta, tsfresh_parameters, is_train=True)

df_train = df_train.dropna(axis="columns")
df_train.head()




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3472115669.py in <cell line: 0>()
----> 1 df_train = preprocess_timeseries(train_meta, tsfresh_parameters, is_train=True)
      2 
      3 df_train = df_train.dropna(axis="columns")
      4 df_train.head()
      5 

/tmp/ipykernel_11/2671612103.py in preprocess_timeseries(meta_df, parameters, is_train, batch_size)
     31 
     32         # Extract features for this batch; parallel across all CPU cores
---> 33         feats = extract_features(
     34             batch_df,
     35             column_id="segment",

/usr/local/lib/python3.11/dist-packages/tsfresh/feature_extraction/extraction.py in extract_features(timeseries_container, default_fc_parameters, kind_to_fc_parameters, column_id, column_sort, column_kind, column_value, chunksize, n_jobs, show_warnings, disable_progressbar, impute_function, profile, profiling_filename, profiling_sorting, distributor, pivot)
    162             warnings.simplefilter("default")
    163 
--> 164         result = _do_extraction(
    165             df=timeseries_container,
    166             column_id=column_id,

/usr/local/lib/python3.11/dist-packages/tsfresh/feature_extraction/extraction.py in _do_extraction(df, column_id, column_value, column_kind, column_sort, default_fc_parameters, kind_to_fc_parameters, n_jobs, chunk_size, disable_progressbar, show_warnings, distributor, pivot)
    268                 )
    269             else:
--> 270                 distributor = MultiprocessingDistributor(
    271                     n_workers=n_jobs,
    272                     disable_progressbar=disable_progressbar,

/usr/local/lib/python3.11/dist-packages/tsfresh/utilities/distribution.py in __init__(self, n_workers, disable_progressbar, progressbar_title, show_warnings)
    460         :type show_warnings: bool
    461         """
--> 462         self.pool = Pool(
    463             processes=n_workers,
    464             initializer=initialize_warnings_in_workers,

/usr/lib/python3.11/multiprocessing/context.py in Pool(self, processes, initializer, initargs, maxtasksperchild)
    117         '''Returns a process pool object'''
    118         from .pool import Pool
--> 119         return Pool(processes, initializer, initargs, maxtasksperchild,
    120                     context=self.get_context())
    121 

/usr/lib/python3.11/multiprocessing/pool.py in __init__(self, processes, initializer, initargs, maxtasksperchild, context)
    203             processes = os.cpu_count() or 1
    204         if processes < 1:
--> 205             raise ValueError("Number of processes must be at least 1")
    206         if maxtasksperchild is not None:
    207             if not isinstance(maxtasksperchild, int) or maxtasksperchild <= 0:

ValueError: Number of processes must be at least 1

## === cell 6
df_test = preprocess_timeseries(None, tsfresh_parameters, is_train=False)
df_test = df_test.dropna(axis="columns")
df_test.head()




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/4232696060.py in <cell line: 0>()
----> 1 df_test = preprocess_timeseries(None, tsfresh_parameters, is_train=False)
      2 df_test = df_test.dropna(axis="columns")
      3 df_test.head()
      4 
      5 

/tmp/ipykernel_11/2671612103.py in preprocess_timeseries(meta_df, parameters, is_train, batch_size)
     24             csv_path = data_folder / ("train" if is_train else "test") / f"{seg}.csv"
     25             # read as float32 to reduce memory, fill NaNs with 0
---> 26             ts = pd.read_csv(csv_path, dtype=np.float32).fillna(0)
     27             ts["segment"] = seg
     28             batch_frames.append(ts)

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

FileNotFoundError: [Errno 2] No such file or directory: '../input/predict-volcanic-eruptions-ingv-oe/test/1003520023.csv.csv'

## === cell 7
feature_cols = [c for c in df_train.columns if c not in ["time_to_eruption", "segment"]]
X = df_train[feature_cols]
y = df_train["time_to_eruption"]

X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=1337)




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4188788755.py in <cell line: 0>()
----> 1 feature_cols = [c for c in df_train.columns if c not in ["time_to_eruption", "segment"]]
      2 X = df_train[feature_cols]
      3 y = df_train["time_to_eruption"]
      4 
      5 X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=1337)

NameError: name 'df_train' is not defined

## === cell 8
model_params = {
    "lambda": 0.0020555245431348778,
    "alpha": 0.11298627316540845,
    "colsample_bytree": 0.6,
    "subsample": 1.0,
    "learning_rate": 0.01,
    "max_depth": 20,
    "random_state": 48,
    "min_child_weight": 18,
    "objective": "reg:squarederror",
    "n_estimators": 1000,
    "tree_method": "hist",  # fall back to CPU if GPU not available
}


class CorrelationSelector(TransformerMixin):
    def __init__(self, threshold=0.85):
        self.threshold = threshold
        self.columns_to_drop = []

    def fit(self, X, y=None):
        corr = X.corr().abs()
        cols = corr.columns
        for i in range(len(cols)):
            for j in range(i):
                if corr.iloc[i, j] >= self.threshold:
                    self.columns_to_drop.append(cols[i])
                    break
        return self

    def transform(self, X, y=None):
        return X.drop(columns=self.columns_to_drop, errors="ignore")


pipe = Pipeline(
    [
        ("corr_sel", CorrelationSelector(threshold=0.85)),
        ("scaler", MinMaxScaler()),
        ("xgb", XGBRegressor(**model_params)),
    ]
)

pipe.fit(X_tr, y_tr)

val_pred = pipe.predict(X_val)
print("Validation MAE:", mean_absolute_error(y_val, val_pred))




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3903798949.py in <cell line: 0>()
     41 )
     42 
---> 43 pipe.fit(X_tr, y_tr)
     44 
     45 val_pred = pipe.predict(X_val)

NameError: name 'X_tr' is not defined

## === cell 9
pipe.fit(X, y)

test_features = df_test[feature_cols]
test_pred = pipe.predict(test_features)

submission = pd.DataFrame(
    {"segment_id": df_test["segment"], "time_to_eruption": test_pred}
)

submission = submission.sort_values("segment_id").reset_index(drop=True)

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3587083852.py in <cell line: 0>()
----> 1 pipe.fit(X, y)
      2 
      3 test_features = df_test[feature_cols]
      4 test_pred = pipe.predict(test_features)
      5 

NameError: name 'X' is not defined
