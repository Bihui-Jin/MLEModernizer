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

import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.base import TransformerMixin
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error

import optuna

from xgboost import XGBRegressor

RANDOM_STATE = 1337
np.random.seed(RANDOM_STATE)

os.environ.setdefault("OMP_NUM_THREADS", str(os.cpu_count() or 1))
os.environ.setdefault("MKL_NUM_THREADS", str(os.cpu_count() or 1))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(os.cpu_count() or 1))
os.environ.setdefault("NUMEXPR_NUM_THREADS", str(os.cpu_count() or 1))



## === cell 1
data_folder = Path("../input/predict-volcanic-eruptions-ingv-oe/")

train_meta_path = data_folder / "train.csv"
sample_sub_path = data_folder / "sample_submission.csv"
train_dir = data_folder / "train"
test_dir = data_folder / "test"

assert train_meta_path.exists(), f"Missing {train_meta_path}"
assert sample_sub_path.exists(), f"Missing {sample_sub_path}"
assert train_dir.exists(), f"Missing {train_dir}"
assert test_dir.exists(), f"Missing {test_dir}"

df = pd.read_csv(train_meta_path)
df.head().T



## === cell 2
print(df.shape)
print(df.columns.tolist())
print(df["segment_id"].nunique(), "unique segment_ids in train meta")

sample_sub = pd.read_csv(sample_sub_path)
print("sample_submission:", sample_sub.shape)
print(sample_sub.head())



## === cell 3
df_sorted = df.sort_values("time_to_eruption")

time_min_segment = str(df_sorted.head(1).iloc[0]["segment_id"])
time_max_segment = str(df_sorted.tail(1).iloc[0]["segment_id"])

df_min = pd.read_csv(train_dir / f"{time_min_segment}.csv")
df_max = pd.read_csv(train_dir / f"{time_max_segment}.csv")

df_min["time_to_eruption"] = df_sorted.head(1).iloc[0]["time_to_eruption"]
df_max["time_to_eruption"] = df_sorted.tail(1).iloc[0]["time_to_eruption"]

df_min.head()



## === cell 4
df_min.describe()



## === cell 5
df_max.describe()



## === cell 6
for i in range(10):
    sensor = f"sensor_{i + 1}"
    if sensor in df_min.columns and sensor in df_max.columns:
        if df_min[sensor].isnull().all() or df_max[sensor].isnull().all():
            del df_min[sensor]
            del df_max[sensor]

print("Columns kept for plotting:", df_min.columns.tolist())




## === cell 7
def plot_sensors_data(ld, rd):
    cols = [c for c in ld.columns if c.startswith("sensor_")]
    n = min(len(cols), 5)
    figure, axs = plt.subplots(5, 2, figsize=(16, 20))
    for ax in axs.ravel():
        ax.set_visible(False)

    for i, c in enumerate(cols[:n]):
        axs[i, 0].set_visible(True)
        axs[i, 1].set_visible(True)
        axs[i, 0].plot(ld[c].values)
        axs[i, 0].set_title(f"Minimal time to eruption, {c}")
        axs[i, 1].plot(rd[c].values)
        axs[i, 1].set_title(f"Maximal time to eruption, {c}")
    plt.tight_layout()


plot_sensors_data(df_min, df_max)



## === cell 8
plot_sensors_data(df_min[:100], df_max[:100])



## === cell 9
from tsfresh import extract_features
from tsfresh.feature_extraction import MinimalFCParameters



## === cell 10
tsfresh_parameters = MinimalFCParameters()



## === cell 11
if "length" in tsfresh_parameters:
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
tsfresh_parameters["autocorrelation"] = [{"lag": k} for k in range(10)]
tsfresh_parameters["agg_autocorrelation"] = [
    {"f_agg": "mean", "maxlag": 40},
    {"f_agg": "median", "maxlag": 40},
    {"f_agg": "var", "maxlag": 40},
]
tsfresh_parameters["friedrich_coefficients"] = [
    {"coeff": 0, "m": 3, "r": 30},
    {"coeff": 1, "m": 3, "r": 30},
    {"coeff": 2, "m": 3, "r": 30},
    {"coeff": 3, "m": 3, "r": 30},
]
tsfresh_parameters["count_above"] = [{"t": 0}]
tsfresh_parameters["count_below"] = [{"t": 0}]

print("Number of tsfresh feature calculators:", len(tsfresh_parameters))




## === cell 12
def _read_segment_csv(path: Path, sensor_cols=None):
    if sensor_cols is None:
        df_seg = pd.read_csv(path, engine="c")
        sensor_cols = [c for c in df_seg.columns if c.startswith("sensor_")]
        df_seg = df_seg[sensor_cols]
    else:
        df_seg = pd.read_csv(
            path,
            usecols=sensor_cols,
            dtype={c: np.float32 for c in sensor_cols},
            engine="c",
        )
    df_seg = df_seg.fillna(0)
    df_seg = df_seg.reset_index(drop=False).rename(columns={"index": "time_index"})
    return df_seg, sensor_cols


def preprocess_timeseries(train_meta_df, parameters, is_train=True, max_segments=None):
    df_result_parts = []

    if is_train:
        seg_rows = train_meta_df[["segment_id", "time_to_eruption"]]
        if max_segments is not None:
            seg_rows = seg_rows.iloc[:max_segments]
        seg_ids = seg_rows["segment_id"].astype(str).tolist()
        y_map = dict(
            zip(
                seg_rows["segment_id"].astype(str).tolist(),
                seg_rows["time_to_eruption"].astype(float).tolist(),
            )
        )
        file_paths = [train_dir / f"{sid}.csv" for sid in seg_ids]
    else:
        fnames = sorted([p.name for p in test_dir.glob("*.csv")])
        if max_segments is not None:
            fnames = fnames[:max_segments]
        seg_ids = [f.replace(".csv", "") for f in fnames]
        file_paths = [test_dir / f for f in fnames]

    first_df, sensor_cols = _read_segment_csv(file_paths[0], sensor_cols=None)
    first_df["id"] = 0
    first_df["segment"] = seg_ids[0]
    df_all = [first_df]

    for idx in range(1, len(file_paths)):
        p = file_paths[idx]
        sid = seg_ids[idx]
        seg_df, _ = _read_segment_csv(p, sensor_cols=sensor_cols)
        seg_df["id"] = 0
        seg_df["segment"] = sid
        df_all.append(seg_df)
        if (idx + 1) % 200 == 0:
            print(f"Loaded segments into RAM: {idx + 1}/{len(file_paths)}")

    big_df = pd.concat(df_all, axis=0, ignore_index=True, sort=False)

    feats = extract_features(
        big_df.drop(columns=["id"]),
        column_id="segment",
        column_sort="time_index",
        disable_progressbar=True,
        default_fc_parameters=parameters,
        n_jobs=-1,
    ).reset_index()

    feats = feats.rename(columns={"index": "segment"})
    if is_train:
        feats["time_to_eruption"] = feats["segment"].map(y_map).astype(float)

    return feats




## === cell 13
def build_or_load_features(parameters):
    train_feats_csv = Path("tsfresh_train_features.csv")
    test_feats_csv = Path("tsfresh_test_features.csv")
    train_feats_parquet = Path("tsfresh_train_features.parquet")
    test_feats_parquet = Path("tsfresh_test_features.parquet")

    if train_feats_parquet.exists() and test_feats_parquet.exists():
        df_train_feats = pd.read_parquet(train_feats_parquet)
        df_test_feats = pd.read_parquet(test_feats_parquet)
        return df_train_feats, df_test_feats

    if train_feats_csv.exists() and test_feats_csv.exists():
        df_train_feats = pd.read_csv(train_feats_csv)
        df_test_feats = pd.read_csv(test_feats_csv)
        return df_train_feats, df_test_feats

    df_train_feats = preprocess_timeseries(
        df, parameters, is_train=True, max_segments=None
    )
    df_test_feats = preprocess_timeseries(
        None, parameters, is_train=False, max_segments=None
    )

    try:
        df_train_feats.to_parquet(train_feats_parquet, index=False)
        df_test_feats.to_parquet(test_feats_parquet, index=False)
    except Exception:
        df_train_feats.to_csv(train_feats_csv, index=False)
        df_test_feats.to_csv(test_feats_csv, index=False)

    return df_train_feats, df_test_feats




## === cell 14
df_train_feats, df_test_feats = build_or_load_features(tsfresh_parameters)

print("df_train_feats:", df_train_feats.shape)
print("df_test_feats:", df_test_feats.shape)
print(df_train_feats[["segment", "time_to_eruption"]].head())
print(df_test_feats[["segment"]].head())



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1544714467.py in <cell line: 0>()
----> 1 df_train_feats, df_test_feats = build_or_load_features(tsfresh_parameters)
      2 
      3 print("df_train_feats:", df_train_feats.shape)
      4 print("df_test_feats:", df_test_feats.shape)
      5 print(df_train_feats[["segment", "time_to_eruption"]].head())

/tmp/ipykernel_11/1833806393.py in build_or_load_features(parameters)
     16         return df_train_feats, df_test_feats
     17 
---> 18     df_train_feats = preprocess_timeseries(
     19         df, parameters, is_train=True, max_segments=None
     20     )

/tmp/ipykernel_11/1782034171.py in preprocess_timeseries(train_meta_df, parameters, is_train, max_segments)
     64     # Speed: one extraction call, using segment as the grouping id and parallel extraction.
     65     # Correctness: equivalent to extracting each segment independently because ids separate windows.
---> 66     feats = extract_features(
     67         big_df.drop(columns=["id"]),
     68         column_id="segment",

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

## === cell 15
df_train_model = df_train_feats.dropna(axis="columns")
df_test_model = df_test_feats.dropna(axis="columns")

drop_cols_train = {"time_to_eruption", "segment"}
drop_cols_test = {"segment"}

train_feature_cols = [c for c in df_train_model.columns if c not in drop_cols_train]
test_feature_cols = [c for c in df_test_model.columns if c not in drop_cols_test]

common_features = sorted(
    list(set(train_feature_cols).intersection(set(test_feature_cols)))
)
print("Common feature count:", len(common_features))

X = df_train_model[common_features]
y = df_train_model["time_to_eruption"].astype(float)

X_predict = df_test_model[common_features]
segments_predict = df_test_model["segment"].astype(str)

print("X:", X.shape, "y:", y.shape, "X_predict:", X_predict.shape)




## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/401224744.py in <cell line: 0>()
----> 1 df_train_model = df_train_feats.dropna(axis="columns")
      2 df_test_model = df_test_feats.dropna(axis="columns")
      3 
      4 drop_cols_train = {"time_to_eruption", "segment"}
      5 drop_cols_test = {"segment"}

NameError: name 'df_train_feats' is not defined

## === cell 16
class LowImportanceSelector(TransformerMixin):
    def __init__(self, threshold, n_estimators=100):
        self.features = None
        self.threshold = threshold
        self.n_estimators = n_estimators

    def fit(self, X, y):
        estimator = RandomForestRegressor(
            n_estimators=self.n_estimators, random_state=RANDOM_STATE, n_jobs=-1
        )
        estimator.fit(X, y)

        importances = pd.DataFrame(
            {"feature": X.columns, "importance": estimator.feature_importances_}
        )
        importances = importances[importances["importance"] > self.threshold]
        self.features = importances["feature"].tolist()
        return self

    def transform(self, X):
        return X[self.features]




## === cell 17
class CorrelationSelector(TransformerMixin):
    def __init__(self, threshold):
        self.columns = None
        self.threshold = threshold

    def fit(self, X, y=None):
        self.columns = set()
        C = X.corr()
        cols = C.columns
        for i in range(len(cols)):
            for j in range(i):
                if (C.iat[i, j] >= self.threshold) and (cols[j] not in self.columns):
                    self.columns.add(cols[i])
        return self

    def transform(self, X, y=None):
        return X.drop(columns=list(self.columns), errors="ignore")




## === cell 18
def objective(trial, data, target):
    parameters = {
        "tree_method": "hist",
        "lambda": trial.suggest_float("lambda", 1e-3, 10.0, log=True),
        "alpha": trial.suggest_float("alpha", 1e-3, 10.0, log=True),
        "colsample_bytree": trial.suggest_categorical(
            "colsample_bytree", [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        ),
        "subsample": trial.suggest_categorical(
            "subsample", [0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
        ),
        "learning_rate": trial.suggest_categorical(
            "learning_rate", [0.008, 0.009, 0.01, 0.012, 0.014, 0.016, 0.018, 0.02]
        ),
        "n_estimators": 1000,
        "max_depth": trial.suggest_categorical(
            "max_depth", [5, 7, 9, 11, 13, 15, 17, 20]
        ),
        "random_state": trial.suggest_categorical("random_state", [24, 48, 2020]),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 300),
    }

    X_tr, X_te, y_tr, y_te = train_test_split(
        data, target, test_size=0.2, random_state=RANDOM_STATE
    )
    model = XGBRegressor(objective="reg:squarederror", **parameters)

    model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)

    return mean_absolute_error(y_te, model.predict(X_te))




## === cell 19
parameters = {
    "lambda": 0.0020555245431348778,
    "alpha": 0.11298627316540845,
    "colsample_bytree": 0.6,
    "subsample": 1.0,
    "learning_rate": 0.01,
    "max_depth": 20,
    "random_state": 48,
    "min_child_weight": 18,
}



## === cell 20
pipe = Pipeline(
    [
        ("correlation", CorrelationSelector(threshold=0.85)),
        ("scaler", MinMaxScaler()),
        (
            "xgboost",
            XGBRegressor(
                objective="reg:squarederror",
                n_estimators=1000,
                tree_method="hist",
                n_jobs=-1,
                **parameters,
            ),
        ),
    ]
)



## === cell 21
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

pipe.fit(X_train, y_train)
valid_pred = pipe.predict(X_valid)
mae = mean_absolute_error(y_valid, valid_pred)
print("Validation MAE:", mae)



## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1659597326.py in <cell line: 0>()
      1 X_train, X_valid, y_train, y_valid = train_test_split(
----> 2     X, y, test_size=0.2, random_state=RANDOM_STATE
      3 )
      4 
      5 pipe.fit(X_train, y_train)

NameError: name 'X' is not defined

## === cell 22
pipe.fit(X, y)



## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1251256995.py in <cell line: 0>()
----> 1 pipe.fit(X, y)
      2 

NameError: name 'X' is not defined

## === cell 23
test_pred = pipe.predict(X_predict)

submission = pd.DataFrame(
    {
        "segment_id": segments_predict.astype(str),
        "time_to_eruption": test_pred.astype(float),
    }
)

submission = (
    submission.set_index("segment_id")
    .reindex(sample_sub["segment_id"].astype(str))
    .reset_index()
)

assert submission.shape[0] == sample_sub.shape[0], "Submission row count mismatch"
assert list(submission.columns) == [
    "segment_id",
    "time_to_eruption",
], "Submission columns mismatch"
assert submission["time_to_eruption"].notnull().all(), "Found null predictions"

submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())

## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/237466155.py in <cell line: 0>()
----> 1 test_pred = pipe.predict(X_predict)
      2 
      3 submission = pd.DataFrame(
      4     {
      5         "segment_id": segments_predict.astype(str),

NameError: name 'X_predict' is not defined
