# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

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

_CPU = int(os.cpu_count() or 1)
os.environ.setdefault("OMP_NUM_THREADS", str(_CPU))
os.environ.setdefault("MKL_NUM_THREADS", str(_CPU))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(_CPU))
os.environ.setdefault("NUMEXPR_NUM_THREADS", str(_CPU))

try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except Exception:
    pass


def _safe_n_jobs(n_jobs: int) -> int:
    if n_jobs is None:
        return 1
    try:
        n = int(n_jobs)
    except Exception:
        return 1
    if n == -1:
        return max(1, int(os.cpu_count() or 1))
    return max(1, n)


TSFRESH_N_JOBS = _safe_n_jobs(-1)

DO_PLOTS = False




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


if DO_PLOTS:
    plot_sensors_data(df_min, df_max)




## === cell 8
if DO_PLOTS:
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
def _infer_sensor_cols_and_dtypes(path: Path):
    df0 = pd.read_csv(path, nrows=1, engine="c")
    sensor_cols = [c for c in df0.columns if c.startswith("sensor_")]
    sensor_dtypes = {c: np.float32 for c in sensor_cols}
    return sensor_cols, sensor_dtypes


def _read_segment_array(path: Path, sensor_cols, sensor_dtypes) -> np.ndarray:
    seg = pd.read_csv(
        path,
        usecols=sensor_cols,
        dtype=sensor_dtypes,
        engine="c",
        na_values=["", "nan", "NaN", "NULL", "null"],
        memory_map=True,
    )
    if seg.isnull().values.any():
        seg = seg.fillna(0.0)
    return seg.to_numpy(dtype=np.float32, copy=False)


def _batch_to_long_df(seg_ids, seg_arrays, sensor_cols):
    T = seg_arrays[0].shape[0]
    S = seg_arrays[0].shape[1]
    B = len(seg_arrays)

    total_rows = B * T * S

    segment_col = np.repeat(np.asarray(seg_ids, dtype=object), T * S)
    time_col = np.tile(np.repeat(np.arange(T, dtype=np.int32), S), B)
    kind_col = np.tile(np.tile(np.asarray(sensor_cols, dtype=object), T), B)

    values = (
        np.stack(seg_arrays, axis=0).reshape(total_rows).astype(np.float32, copy=False)
    )

    return pd.DataFrame(
        {
            "segment": segment_col,
            "time_index": time_col,
            "kind": kind_col,
            "value": values,
        }
    )


def preprocess_timeseries(
    train_meta_df, parameters, is_train=True, max_segments=None, batch_size=32
):
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

    sensor_cols, sensor_dtypes = _infer_sensor_cols_and_dtypes(file_paths[0])

    feats_parts = []
    n_total = len(file_paths)
    for start in range(0, n_total, batch_size):
        end = min(n_total, start + batch_size)
        batch_ids = seg_ids[start:end]
        batch_paths = file_paths[start:end]

        seg_arrays = [
            _read_segment_array(p, sensor_cols, sensor_dtypes) for p in batch_paths
        ]
        long_df = _batch_to_long_df(batch_ids, seg_arrays, sensor_cols)

        feats = (
            extract_features(
                long_df,
                column_id="segment",
                column_sort="time_index",
                column_kind="kind",
                column_value="value",
                disable_progressbar=True,
                default_fc_parameters=parameters,
                n_jobs=TSFRESH_N_JOBS,
            )
            .reset_index()
            .rename(columns={"index": "segment"})
        )
        feats_parts.append(feats)

        if end % 200 == 0 or end == n_total:
            print(f"Processed segments: {end}/{n_total}")

    feats_all = pd.concat(feats_parts, axis=0, ignore_index=True, sort=False)
    if is_train:
        feats_all["time_to_eruption"] = feats_all["segment"].map(y_map).astype(float)
    return feats_all




## === cell 13
def build_or_load_features(parameters):
    cache_dir = (
        Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")
    )
    train_feats_csv = cache_dir / "tsfresh_train_features.csv"
    test_feats_csv = cache_dir / "tsfresh_test_features.csv"
    train_feats_parquet = cache_dir / "tsfresh_train_features.parquet"
    test_feats_parquet = cache_dir / "tsfresh_test_features.parquet"

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
        cols = list(X.columns)
        A = X.to_numpy(dtype=np.float32, copy=False)

        mean = A.mean(axis=0, keepdims=True)
        std = A.std(axis=0, ddof=1, keepdims=True)
        std[std == 0] = 1.0
        Z = (A - mean) / std

        n = Z.shape[0]
        C = (Z.T @ Z) / (n - 1)

        for i in range(len(cols)):
            for j in range(i):
                if (C[i, j] >= self.threshold) and (cols[j] not in self.columns):
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




## === cell 22
pipe.fit(X, y)




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
