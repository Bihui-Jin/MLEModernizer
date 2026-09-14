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
import gc  # explicit memory cleanup

from sklearn.base import TransformerMixin
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error

from xgboost import XGBRegressor
from tsfresh import extract_features
from tsfresh.feature_extraction import MinimalFCParameters

data_folder = Path("/kaggle/input/predict-volcanic-eruptions-ingv-oe/")

train_feat_path = data_folder / "train_features.parquet"
test_feat_path = data_folder / "test_features.parquet"




## === cell 1
train_meta = pd.read_csv(data_folder / "train.csv")
train_meta.head()




## === cell 2
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




## === cell 3
def preprocess_timeseries(meta_df, parameters, is_train=True, batch_size=2000):
    """
    Faster TSFresh feature extraction:
    * Larger batch_size (default 2000) dramatically reduces the number of
      expensive `extract_features` calls.
    * `n_jobs=-1` uses all available CPU cores.
    * File paths are built once, avoiding per‑row Path joins.
    * Returns the identical feature DataFrames as the original logic.
    """
    n_jobs = -1

    if is_train:
        base_path = data_folder / "train"
        segments = [
            base_path / f"{seg}.csv"
            for seg in meta_df["segment_id"].astype(str).tolist()
        ]
        target_map = meta_df.set_index("segment_id")["time_to_eruption"]
    else:
        base_path = data_folder / "test"
        segments = sorted(base_path.glob("*.csv"))
        target_map = None

    results = []
    for start in range(0, len(segments), batch_size):
        batch_paths = segments[start : start + batch_size]

        batch_frames = [
            pd.read_csv(p, dtype=np.float32).fillna(0).assign(segment=p.stem)
            for p in batch_paths
        ]
        batch_df = pd.concat(batch_frames, ignore_index=True, copy=False)

        feats = extract_features(
            batch_df,
            column_id="segment",
            default_fc_parameters=parameters,
            disable_progressbar=True,
            n_jobs=n_jobs,
        )
        feats["segment"] = batch_df["segment"].unique()
        if is_train:
            feats["time_to_eruption"] = feats["segment"].map(target_map)

        results.append(feats)

        del batch_frames, batch_df, feats
        gc.collect()

    return pd.concat(results, ignore_index=True, copy=False)




## === cell 4
if train_feat_path.is_file():
    df_train = pd.read_parquet(train_feat_path)
else:
    df_train = preprocess_timeseries(train_meta, tsfresh_parameters, is_train=True)
    df_train = df_train.fillna(0)
    df_train.to_parquet(train_feat_path, index=False)

df_train.head()




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2073676921.py in <cell line: 0>()
      5     df_train = pd.read_parquet(train_feat_path)
      6 else:
----> 7     df_train = preprocess_timeseries(train_meta, tsfresh_parameters, is_train=True)
      8     df_train = df_train.fillna(0)
      9     df_train.to_parquet(train_feat_path, index=False)

/tmp/ipykernel_11/1831005827.py in preprocess_timeseries(meta_df, parameters, is_train, batch_size)
     36 
     37         # Extract TSFresh features for the whole batch
---> 38         feats = extract_features(
     39             batch_df,
     40             column_id="segment",

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

## === cell 5
if test_feat_path.is_file():
    df_test = pd.read_parquet(test_feat_path)
else:
    df_test = preprocess_timeseries(None, tsfresh_parameters, is_train=False)
    df_test = df_test.fillna(0)
    df_test.to_parquet(test_feat_path, index=False)

df_test.head()




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2604686291.py in <cell line: 0>()
      5     df_test = pd.read_parquet(test_feat_path)
      6 else:
----> 7     df_test = preprocess_timeseries(None, tsfresh_parameters, is_train=False)
      8     df_test = df_test.fillna(0)
      9     df_test.to_parquet(test_feat_path, index=False)

/tmp/ipykernel_11/1831005827.py in preprocess_timeseries(meta_df, parameters, is_train, batch_size)
     36 
     37         # Extract TSFresh features for the whole batch
---> 38         feats = extract_features(
     39             batch_df,
     40             column_id="segment",

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
feature_cols = [c for c in df_train.columns if c not in ["time_to_eruption", "segment"]]
X = df_train[feature_cols]
y = df_train["time_to_eruption"]

X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=1337)




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4188788755.py in <cell line: 0>()
----> 1 feature_cols = [c for c in df_train.columns if c not in ["time_to_eruption", "segment"]]
      2 X = df_train[feature_cols]
      3 y = df_train["time_to_eruption"]
      4 
      5 X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=1337)

NameError: name 'df_train' is not defined

## === cell 7
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
    "tree_method": "hist",
}


class CorrelationSelector(TransformerMixin):
    def __init__(self, threshold=0.85):
        self.threshold = threshold
        self.columns_to_drop = []

    def fit(self, X, y=None):
        corr = X.corr().abs()
        cols = corr.columns
        mask = np.triu(np.ones(corr.shape), k=1).astype(bool)
        high_corr = corr.where(mask)
        for i, col in enumerate(cols):
            if (high_corr[col] >= self.threshold).any():
                self.columns_to_drop.append(col)
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




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3024365569.py in <cell line: 0>()
     41 )
     42 
---> 43 pipe.fit(X_tr, y_tr)
     44 
     45 val_pred = pipe.predict(X_val)

NameError: name 'X_tr' is not defined

## === cell 8
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

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3587083852.py in <cell line: 0>()
----> 1 pipe.fit(X, y)
      2 
      3 test_features = df_test[feature_cols]
      4 test_pred = pipe.predict(test_features)
      5 

NameError: name 'X' is not defined
