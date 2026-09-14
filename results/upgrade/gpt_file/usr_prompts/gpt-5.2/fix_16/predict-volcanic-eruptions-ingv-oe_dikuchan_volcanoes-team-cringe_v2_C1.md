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

from matplotlib import pyplot as plt

from sklearn.base import TransformerMixin
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from xgboost import XGBRegressor

import random

os.environ.setdefault("PYTHONHASHSEED", "1337")
random.seed(1337)
np.random.seed(1337)

_cpu = os.cpu_count() or 2
_default_threads = max(1, _cpu - 1)
os.environ.setdefault("OMP_NUM_THREADS", str(_default_threads))
os.environ.setdefault("MKL_NUM_THREADS", os.environ["OMP_NUM_THREADS"])
os.environ.setdefault("OPENBLAS_NUM_THREADS", os.environ["OMP_NUM_THREADS"])
os.environ.setdefault("NUMEXPR_NUM_THREADS", os.environ["OMP_NUM_THREADS"])

from joblib import Parallel, delayed




## === cell 1
def resolve_data_folder():
    candidates = [
        Path("../input/predict-volcanic-eruptions-ingv-oe/"),
        Path("/kaggle/input/predict-volcanic-eruptions-ingv-oe/"),
        Path("/kaggle/data/predict-volcanic-eruptions-ingv-oe/"),
        Path("../kaggle/data/predict-volcanic-eruptions-ingv-oe/"),
        Path("/kaggle/data/input/predict-volcanic-eruptions-ingv-oe/"),
    ]
    for p in candidates:
        if (p / "train.csv").exists():
            return p
    return Path("../input/predict-volcanic-eruptions-ingv-oe/")


data_folder = resolve_data_folder()
data_folder




## === cell 2
df = pd.read_csv(data_folder / "train.csv")
df.head().T




## === cell 3
df_sorted = df.sort_values("time_to_eruption")

time_min = df_sorted.head(1).iloc[0]["segment_id"]
time_max = df_sorted.tail(1).iloc[0]["segment_id"]

_SENSOR_DTYPES = {f"sensor_{i}": np.float32 for i in range(1, 11)}
df_min = pd.read_csv(
    data_folder / "train" / f"{time_min}.csv", dtype=_SENSOR_DTYPES, engine="c"
).fillna(0)
df_max = pd.read_csv(
    data_folder / "train" / f"{time_max}.csv", dtype=_SENSOR_DTYPES, engine="c"
).fillna(0)

df_min["time_to_eruption"] = df_sorted.head(1).iloc[0]["time_to_eruption"]
df_max["time_to_eruption"] = df_sorted.tail(1).iloc[0]["time_to_eruption"]




## === cell 4
df_min.describe()




## === cell 5
df_max.describe()




## === cell 6
pass




## === cell 7
for i in range(10):
    sensor = f"sensor_{i + 1}"
    if df_min[sensor].isnull().all() or df_max[sensor].isnull().all():
        del df_min[sensor]
        del df_max[sensor]




## === cell 8
def plot_sensors_data(ld, rd):
    figure, axs = plt.subplots(5, 2, figsize=(16, 20))
    cols = [c for c in ld.columns if c.startswith("sensor_")]

    for i, c in zip(range(min(5, len(cols))), cols):
        axs[i, 0].plot(ld[c])
        axs[i, 0].set_title(f"Minimal time to eruption, {c}")

        axs[i, 1].plot(rd[c])
        axs[i, 1].set_title(f"Maximal time to eruption, {c}")
    plt.tight_layout()




## === cell 9
try:
    plot_sensors_data(df_min, df_max)
except Exception as e:
    print("Plot skipped:", repr(e))




## === cell 10
pass




## === cell 11
try:
    plot_sensors_data(df_min[:100], df_max[:100])
except Exception as e:
    print("Plot skipped:", repr(e))




## === cell 12
pass




## === cell 13
from tsfresh import extract_features
from tsfresh.feature_extraction import EfficientFCParameters




## === cell 14
tsfresh_parameters = EfficientFCParameters()




## === cell 15
pass




## === cell 16
tsfresh_parameters.pop("length", None)

for k in [
    "skewness",
    "kurtosis",
    "last_location_of_maximum",
    "first_location_of_maximum",
    "last_location_of_minimum",
    "first_location_of_minimum",
    "benford_correlation",
    "percentage_of_reoccurring_values_to_all_values",
    "percentage_of_reoccurring_datapoints_to_all_datapoints",
]:
    tsfresh_parameters[k] = None

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
tsfresh_parameters["autocorrelation"] = [{"lag": lag} for lag in range(10)]
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




## === cell 17
pass




## === cell 18
import hashlib
import json

_SENSOR_DTYPES = {f"sensor_{i}": np.float32 for i in range(1, 11)}
_SENSOR_COLS = [f"sensor_{i}" for i in range(1, 11)]

try:
    import pyarrow.csv as pacsv
    import pyarrow as pa

    _HAS_PYARROW = True
except Exception:
    _HAS_PYARROW = False


def _read_segment_array_fast(path: Path) -> np.ndarray:
    if _HAS_PYARROW:
        read_opts = pacsv.ReadOptions(use_threads=True, block_size=1 << 20)
        convert_opts = pacsv.ConvertOptions(
            include_columns=_SENSOR_COLS,
            column_types={c: pa.float32() for c in _SENSOR_COLS},
            strings_can_be_null=True,
        )
        parse_opts = pacsv.ParseOptions(delimiter=",")
        table = pacsv.read_csv(
            str(path),
            read_options=read_opts,
            parse_options=parse_opts,
            convert_options=convert_opts,
        )
        try:
            arr = table.to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
        except Exception:
            arr = table.to_pandas(types_mapper=None).to_numpy(
                dtype=np.float32, copy=False
            )
    else:
        df_seg = pd.read_csv(
            path,
            dtype=_SENSOR_DTYPES,
            usecols=_SENSOR_COLS,
            engine="c",
        )
        arr = df_seg.to_numpy(dtype=np.float32, copy=False)

    np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


def _params_fingerprint(parameters) -> str:
    dumped = json.dumps(parameters, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.md5(dumped.encode("utf-8")).hexdigest()


_PARAMS_FP = _params_fingerprint(tsfresh_parameters)


def _segment_array_cache_path(cache_dir: Path, is_train: bool, seg_id: str) -> Path:
    return cache_dir / f"{'train' if is_train else 'test'}_{seg_id}.npy"


def _load_or_read_segment_array(
    cache_dir: Path, is_train: bool, seg_id: str
) -> np.ndarray:
    npy_path = _segment_array_cache_path(cache_dir, is_train, seg_id)
    if npy_path.exists():
        return np.load(npy_path, allow_pickle=False, mmap_mode="r")
    seg_path = data_folder / ("train" if is_train else "test") / f"{seg_id}.csv"
    arr = _read_segment_array_fast(seg_path)
    np.save(npy_path, arr, allow_pickle=False)
    return arr


def _build_long_df_from_arrays(seg_ids, arrays) -> pd.DataFrame:
    counts = np.fromiter(
        (a.shape[0] for a in arrays), dtype=np.int32, count=len(arrays)
    )
    total_rows = int(counts.sum())
    if total_rows == 0:
        return pd.DataFrame(columns=_SENSOR_COLS + ["index", "id"])

    big = np.empty((total_rows, len(_SENSOR_COLS)), dtype=np.float32)
    idx = np.empty(total_rows, dtype=np.int32)
    ids = np.empty(total_rows, dtype=object)

    pos = 0
    for seg_id, arr, n in zip(seg_ids, arrays, counts.tolist()):
        big[pos : pos + n] = arr
        idx[pos : pos + n] = np.arange(n, dtype=np.int32)
        ids[pos : pos + n] = seg_id
        pos += n

    df_long = pd.DataFrame(big, columns=_SENSOR_COLS, copy=False)
    df_long["index"] = idx
    df_long["id"] = ids
    return df_long


def preprocess_timeseries(
    meta_df, parameters, is_train=True, verbose_every=25, max_segments=None
):
    if is_train:
        seg_tuples = meta_df[["segment_id", "time_to_eruption"]].itertuples(
            index=False, name=None
        )
        seg_tuples = (
            list(seg_tuples)
            if max_segments is None
            else list(seg_tuples)[:max_segments]
        )
        seg_ids = [str(s) for s, _ in seg_tuples]
        targets = {str(s): float(t) for s, t in seg_tuples}
    else:
        test_dir = data_folder / "test"
        files = sorted([p.stem for p in test_dir.glob("*.csv")])
        if max_segments is not None:
            files = files[:max_segments]
        seg_ids = [str(s) for s in files]
        targets = None

    cache_dir = Path("tsfresh_cache")
    cache_dir.mkdir(exist_ok=True)

    batch_size = 64

    total = len(seg_ids)
    n_workers = max(1, min(_default_threads, total))

    out_frames = []
    done = 0

    for start in range(0, total, batch_size):
        batch_ids = seg_ids[start : start + batch_size]

        arrays = Parallel(
            n_jobs=n_workers,
            backend="threading",
            batch_size=8,
            pre_dispatch="2*n_jobs",
        )(
            delayed(_load_or_read_segment_array)(cache_dir, is_train, sid)
            for sid in batch_ids
        )

        df_long = _build_long_df_from_arrays(batch_ids, arrays)

        feats = extract_features(
            df_long,
            column_id="id",
            column_sort="index",
            disable_progressbar=True,
            default_fc_parameters=parameters,
            n_jobs=0,
            chunksize=32,
        )

        feats = feats.reset_index(names="segment")
        if is_train:
            feats["time_to_eruption"] = feats["segment"].map(targets).astype(np.float32)

        feats = feats.replace([np.inf, -np.inf], np.nan).fillna(0)
        out_frames.append(feats)

        done = min(total, start + len(batch_ids))
        if verbose_every and (done % verbose_every == 0 or done == total):
            print(f"Processed {done} segments")

    df_result = pd.concat(out_frames, axis=0, ignore_index=True, sort=False, copy=False)
    df_result = df_result.replace([np.inf, -np.inf], np.nan).fillna(0)
    return df_result




## === cell 19
pass




## === cell 20
def save(parameters):
    ts_train = pd.read_csv(data_folder / "train.csv")
    df_train_local = preprocess_timeseries(ts_train, parameters, is_train=True)
    df_train_local.to_csv("train_features.csv", index=False)

    df_test_local = preprocess_timeseries(None, parameters, is_train=False)
    df_test_local.to_csv("test_features.csv", index=False)




## === cell 21
pass




## === cell 22
def _find_precomputed_feature_csv(name: str) -> Path | None:
    candidates = [
        Path(name),
        data_folder / name,
        Path("/kaggle/input/predict-volcanic-eruptions-ingv-oe") / name,
        Path("/kaggle/data/predict-volcanic-eruptions-ingv-oe") / name,
        Path("../input/predict-volcanic-eruptions-ingv-oe") / name,
        Path("../kaggle/data/predict-volcanic-eruptions-ingv-oe") / name,
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _find_precomputed_feature_parquet(name: str) -> Path | None:
    candidates = [
        Path(name),
        data_folder / name,
        Path("/kaggle/input/predict-volcanic-eruptions-ingv-oe") / name,
        Path("/kaggle/data/predict-volcanic-eruptions-ingv-oe") / name,
        Path("../input/predict-volcanic-eruptions-ingv-oe") / name,
        Path("../kaggle/data/predict-volcanic-eruptions-ingv-oe") / name,
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


TRAIN_FEAT_PATH = Path("train_features.csv")
TRAIN_FEAT_PARQUET = Path("train_features.parquet")

precomp_train = _find_precomputed_feature_csv("train_features.csv")
precomp_train_parq = _find_precomputed_feature_parquet("train_features.parquet")

if TRAIN_FEAT_PARQUET.exists():
    df_train_feat = pd.read_parquet(TRAIN_FEAT_PARQUET)
elif precomp_train_parq is not None and precomp_train_parq != TRAIN_FEAT_PARQUET:
    df_train_feat = pd.read_parquet(precomp_train_parq)
    df_train_feat.to_parquet(TRAIN_FEAT_PARQUET, index=False)
    df_train_feat.to_csv(TRAIN_FEAT_PATH, index=False)
elif TRAIN_FEAT_PATH.exists():
    df_train_feat = pd.read_csv(TRAIN_FEAT_PATH)
    df_train_feat.to_parquet(TRAIN_FEAT_PARQUET, index=False)
elif precomp_train is not None and precomp_train != TRAIN_FEAT_PATH:
    df_train_feat = pd.read_csv(precomp_train)
    df_train_feat.to_csv(TRAIN_FEAT_PATH, index=False)
    df_train_feat.to_parquet(TRAIN_FEAT_PARQUET, index=False)
else:
    df_train_feat = preprocess_timeseries(
        df, tsfresh_parameters, is_train=True, verbose_every=200
    )
    df_train_feat.to_csv(TRAIN_FEAT_PATH, index=False)
    df_train_feat.to_parquet(TRAIN_FEAT_PARQUET, index=False)

df_train_feat = df_train_feat.dropna(axis="columns")  # keep original intent

features = [
    c for c in df_train_feat.columns if c not in ["time_to_eruption", "segment"]
]
X = df_train_feat[features]
y = df_train_feat["time_to_eruption"]

df_train_feat.head()




## === cell 23
X_np = np.asarray(X, dtype=np.float32)
np.nan_to_num(X_np, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
X = pd.DataFrame(X_np, columns=X.columns, index=X.index)

y_np = np.asarray(y, dtype=np.float32)
np.nan_to_num(y_np, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
y = pd.Series(y_np, index=y.index, name="time_to_eruption")




## === cell 24
class LowImportanceSelector(TransformerMixin):
    def __init__(self, threshold, n_estimators=100):
        self.features = None
        self.threshold = threshold
        self.n_estimators = n_estimators

    def fit(self, X, y):
        estimator = RandomForestRegressor(
            n_estimators=self.n_estimators, random_state=1337, n_jobs=-1
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




## === cell 25
pass




## === cell 26
class CorrelationSelector(TransformerMixin):
    def __init__(self, threshold):
        self.columns = None
        self.threshold = threshold

    def fit(self, X, y=None):
        X_num = X.select_dtypes(include=[np.number])
        Xv = X_num.to_numpy(copy=False)
        if Xv.dtype != np.float64:
            Xv = Xv.astype(np.float64, copy=False)

        C = np.corrcoef(Xv, rowvar=False)
        cols = np.asarray(X_num.columns.to_list(), dtype=object)

        iu = np.triu_indices(C.shape[0], k=1)
        high = C[iu] >= self.threshold

        to_drop = set()
        pairs_i = iu[1][high]
        pairs_j = iu[0][high]
        order = np.argsort(pairs_i, kind="mergesort")
        pairs_i = pairs_i[order]
        pairs_j = pairs_j[order]

        last_i = -1
        js = []
        for i, j in zip(pairs_i.tolist(), pairs_j.tolist()):
            if i != last_i:
                if last_i != -1 and (cols[last_i] not in to_drop):
                    for jj in js:
                        if cols[jj] not in to_drop:
                            to_drop.add(cols[last_i])
                            break
                last_i = i
                js = [j]
            else:
                js.append(j)

        if last_i != -1 and (cols[last_i] not in to_drop):
            for jj in js:
                if cols[jj] not in to_drop:
                    to_drop.add(cols[last_i])
                    break

        self.columns = to_drop
        return self

    def transform(self, X, y=None):
        return X.drop(columns=list(self.columns), errors="ignore")




## === cell 27
pipe = Pipeline(
    [
        ("scaler", MinMaxScaler()),
        (
            "xgboost",
            XGBRegressor(
                objective="reg:squarederror",
                n_estimators=100,
                random_state=1337,
                n_jobs=_default_threads,
            ),
        ),
    ]
)




## === cell 28
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=1337
)




## === cell 29
pipe.fit(X_train, y_train)




## === cell 30
valid_r2 = pipe.score(X_valid, y_valid)
valid_r2




## === cell 31
pass




## === cell 32
TEST_FEAT_PATH = Path("test_features.csv")
TEST_FEAT_PARQUET = Path("test_features.parquet")

precomp_test = _find_precomputed_feature_csv("test_features.csv")
precomp_test_parq = _find_precomputed_feature_parquet("test_features.parquet")

if TEST_FEAT_PARQUET.exists():
    df_test_feat = pd.read_parquet(TEST_FEAT_PARQUET)
elif precomp_test_parq is not None and precomp_test_parq != TEST_FEAT_PARQUET:
    df_test_feat = pd.read_parquet(precomp_test_parq)
    df_test_feat.to_parquet(TEST_FEAT_PARQUET, index=False)
    df_test_feat.to_csv(TEST_FEAT_PATH, index=False)
elif TEST_FEAT_PATH.exists():
    df_test_feat = pd.read_csv(TEST_FEAT_PATH)
    df_test_feat.to_parquet(TEST_FEAT_PARQUET, index=False)
elif precomp_test is not None and precomp_test != TEST_FEAT_PATH:
    df_test_feat = pd.read_csv(precomp_test)
    df_test_feat.to_csv(TEST_FEAT_PATH, index=False)
    df_test_feat.to_parquet(TEST_FEAT_PARQUET, index=False)
else:
    df_test_feat = preprocess_timeseries(
        None, tsfresh_parameters, is_train=False, verbose_every=200
    )
    df_test_feat.to_csv(TEST_FEAT_PATH, index=False)
    df_test_feat.to_parquet(TEST_FEAT_PARQUET, index=False)

df_test_feat = df_test_feat.dropna(axis="columns")
df_test_feat["segment"] = df_test_feat["segment"].astype(str)

X_test_feat = df_test_feat.drop(columns=["segment"], errors="ignore")

Xt_np = np.asarray(X_test_feat, dtype=np.float32)
np.nan_to_num(Xt_np, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
X_test_feat = pd.DataFrame(Xt_np, columns=X_test_feat.columns, index=X_test_feat.index)

X_test_feat = X_test_feat.reindex(columns=X.columns, fill_value=0.0)

pipe.fit(X, y)

pred = pipe.predict(X_test_feat).astype(np.float32, copy=False)

submission = pd.DataFrame(
    {"segment_id": df_test_feat["segment"].astype(str), "time_to_eruption": pred}
)
submission = submission[["segment_id", "time_to_eruption"]]

submission.to_csv("submission.csv", index=False)

submission.head()
