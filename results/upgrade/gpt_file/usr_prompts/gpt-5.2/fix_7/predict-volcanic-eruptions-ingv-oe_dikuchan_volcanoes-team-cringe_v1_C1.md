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
import seaborn as sns

from matplotlib import pyplot as plt

from sklearn.base import TransformerMixin
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from xgboost import XGBRegressor

os.environ.setdefault("PYTHONHASHSEED", "1337")
np.random.seed(1337)

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")



## === cell 1
data_folder = Path("/kaggle/input/predict-volcanic-eruptions-ingv-oe/")
if not data_folder.exists():
    data_folder = Path("/kaggle/data/predict-volcanic-eruptions-ingv-oe/")

train_meta_path = data_folder / "train.csv"
sample_sub_path = data_folder / "sample_submission.csv"

df = pd.read_csv(train_meta_path)
df.head().T



## === cell 2
assert train_meta_path.exists(), f"train.csv not found at: {train_meta_path}"
assert (
    data_folder / "train"
).exists(), f"train folder not found at: {data_folder / 'train'}"
assert (
    data_folder / "test"
).exists(), f"test folder not found at: {data_folder / 'test'}"
assert (
    sample_sub_path.exists()
), f"sample_submission.csv not found at: {sample_sub_path}"



## === cell 3
df = df.sort_values("time_to_eruption")

time_min = str(df.head(1).iloc[0]["segment_id"])
time_max = str(df.tail(1).iloc[0]["segment_id"])

sensor_cols = [f"sensor_{i}" for i in range(1, 11)]
dtypes = {c: np.float32 for c in sensor_cols}

df_min = pd.read_csv(
    data_folder / "train" / f"{time_min}.csv",
    usecols=sensor_cols,
    dtype=dtypes,
)
df_max = pd.read_csv(
    data_folder / "train" / f"{time_max}.csv",
    usecols=sensor_cols,
    dtype=dtypes,
)

df_min["time_to_eruption"] = df.head(1).iloc[0]["time_to_eruption"]
df_max["time_to_eruption"] = df.tail(1).iloc[0]["time_to_eruption"]



## === cell 4
df_min.describe()



## === cell 5
df_max.describe()



## === cell 6
expected_sensors = [f"sensor_{i}" for i in range(1, 11)]
missing_min = [c for c in expected_sensors if c not in df_min.columns]
missing_max = [c for c in expected_sensors if c not in df_max.columns]
if missing_min or missing_max:
    print("Warning: missing sensor columns:", {"min": missing_min, "max": missing_max})



## === cell 7
for i in range(10):
    sensor = f"sensor_{i + 1}"
    if sensor in df_min.columns and sensor in df_max.columns:
        if df_min[sensor].isnull().all() or df_max[sensor].isnull().all():
            del df_min[sensor]
            del df_max[sensor]




## === cell 8
def plot_sensors_data(ld, rd):
    figure, axs = plt.subplots(5, 2, figsize=(16, 20))

    cols = [c for c in ld.columns if c.startswith("sensor_")][:5]
    for i, c in enumerate(cols):
        axs[i, 0].plot(ld[c].to_numpy(copy=False))
        axs[i, 0].set_title(f"Minimal time to eruption, {c}")

        axs[i, 1].plot(rd[c].to_numpy(copy=False))
        axs[i, 1].set_title(f"Maximal time to eruption, {c}")
    plt.tight_layout()




## === cell 9
plt.close("all")



## === cell 10
plt.close("all")



## === cell 11
from tsfresh import extract_features
from tsfresh.feature_extraction import MinimalFCParameters



## === cell 12
tsfresh_parameters = MinimalFCParameters()



## === cell 13
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
tsfresh_parameters["autocorrelation"] = [
    {"lag": 0},
    {"lag": 1},
    {"lag": 2},
    {"lag": 3},
    {"lag": 4},
    {"lag": 5},
    {"lag": 6},
    {"lag": 7},
    {"lag": 8},
    {"lag": 9},
]
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




## === cell 14
def _safe_n_jobs(n_jobs: int) -> int:
    if n_jobs is None:
        return 1
    if n_jobs == 0:
        return 1
    if n_jobs < 0:
        cpu = os.cpu_count() or 1
        return max(1, cpu)
    return max(1, int(n_jobs))


_sample_segment_ids = (
    pd.read_csv(sample_sub_path, usecols=["segment_id"])["segment_id"]
    .astype(str)
    .tolist()
)

_SENSOR_COLS = [f"sensor_{i}" for i in range(1, 11)]
_DTYPES = {c: np.float32 for c in _SENSOR_COLS}


def _load_segment_timeseries_array(segment_timeseries_path: Path) -> np.ndarray:
    segment_timeseries = pd.read_csv(
        segment_timeseries_path,
        usecols=_SENSOR_COLS,
        dtype=_DTYPES,
        engine="c",
        low_memory=False,
    )
    arr = segment_timeseries.to_numpy(copy=False)
    if np.isnan(arr).any():
        np.nan_to_num(arr, copy=False, nan=0.0)
    return np.ascontiguousarray(arr, dtype=np.float32)


def preprocess_timeseries(meta_df, parameters, is_train=True, n_jobs=-1, batch_size=16):
    results = []

    if is_train:
        segments = meta_df[["segment_id", "time_to_eruption"]].itertuples(
            index=False, name=None
        )
    else:
        segments = ((sid,) for sid in _sample_segment_ids)

    n_jobs_eff = _safe_n_jobs(n_jobs)

    batch_arrays = []
    batch_segment_ids = []
    batch_targets = []
    batch_count = 0
    global_idx = 0

    def _flush_batch():
        nonlocal batch_arrays, batch_segment_ids, batch_targets, batch_count, results
        if not batch_arrays:
            return

        lengths = np.fromiter((a.shape[0] for a in batch_arrays), dtype=np.int64)
        total_len = int(lengths.sum())
        n_sensors = len(_SENSOR_COLS)

        sensor_block = np.empty((total_len, n_sensors), dtype=np.float32)
        id_col = np.empty(total_len, dtype=np.int32)
        index_col = np.empty(total_len, dtype=np.int32)

        pos = 0
        for local_id, arr in enumerate(batch_arrays):
            n = arr.shape[0]
            sensor_block[pos : pos + n] = arr
            id_col[pos : pos + n] = local_id
            index_col[pos : pos + n] = np.arange(n, dtype=np.int32)
            pos += n

        big_df = pd.DataFrame(sensor_block, columns=_SENSOR_COLS, copy=False)
        big_df.insert(0, "index", index_col)
        big_df.insert(0, "id", id_col)

        extracted = extract_features(
            big_df,
            column_id="id",
            column_sort="index",
            disable_progressbar=True,
            default_fc_parameters=parameters,
            n_jobs=n_jobs_eff,
        )

        extracted = extracted.reset_index(drop=True)
        extracted["segment"] = batch_segment_ids
        if is_train:
            extracted["time_to_eruption"] = batch_targets

        results.append(extracted)

        batch_arrays = []
        batch_segment_ids = []
        batch_targets = []
        batch_count = 0

    for row in segments:
        if is_train:
            segment_id, time_to_eruption = row
            segment_id = str(segment_id)
            segment_timeseries_path = data_folder / "train" / f"{segment_id}.csv"
        else:
            segment_id = str(row[0])
            segment_timeseries_path = data_folder / "test" / f"{segment_id}.csv"

        arr = _load_segment_timeseries_array(segment_timeseries_path)

        batch_arrays.append(arr)
        batch_segment_ids.append(segment_id)
        if is_train:
            batch_targets.append(time_to_eruption)

        batch_count += 1
        global_idx += 1

        if batch_count >= batch_size:
            _flush_batch()

        if global_idx % 200 == 0 or global_idx == 1:
            print(f"Processed segment #{global_idx}")

    _flush_batch()
    df_result = pd.concat(results, axis=0, ignore_index=True, sort=True)
    return df_result




## === cell 15
train_feat_path = Path("train_tsfresh.csv")
test_feat_path = Path("test_tsfresh.csv")

if train_feat_path.exists():
    df_train_feat = pd.read_csv(train_feat_path)
else:
    df_train_feat = preprocess_timeseries(
        df, tsfresh_parameters, is_train=True, n_jobs=-1, batch_size=16
    )
    df_train_feat.to_csv(train_feat_path, index=False)

if test_feat_path.exists():
    df_test_feat = pd.read_csv(test_feat_path)
else:
    df_test_feat = preprocess_timeseries(
        None, tsfresh_parameters, is_train=False, n_jobs=-1, batch_size=16
    )
    df_test_feat.to_csv(test_feat_path, index=False)

assert (
    "segment" in df_train_feat.columns and "time_to_eruption" in df_train_feat.columns
)
assert "segment" in df_test_feat.columns
df_train_feat.shape, df_test_feat.shape



## === cell 16
df_train_feat = df_train_feat.dropna(axis="columns")
df_test_feat = df_test_feat.dropna(axis="columns")

train_cols = set(df_train_feat.columns)
test_cols = set(df_test_feat.columns)

id_cols_train = {"time_to_eruption", "segment"}
id_cols_test = {"segment"}

common_features = sorted(
    list((train_cols - id_cols_train) & (test_cols - id_cols_test))
)

X = df_train_feat[common_features]
y = df_train_feat["time_to_eruption"]

df_train_feat.head()




## === cell 17
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




## === cell 18
class CorrelationSelector(TransformerMixin):
    def __init__(self, threshold):
        self.columns = None
        self.threshold = threshold

    def fit(self, X, y=None):
        cols = X.columns.to_list()
        C = X.corr().to_numpy(copy=False)
        to_drop = set()
        n = C.shape[0]
        for i in range(n):
            if cols[i] in to_drop:
                continue
            row = C[i, :i]
            for j in range(i):
                if row[j] >= self.threshold and cols[j] not in to_drop:
                    to_drop.add(cols[i])
                    break
        self.columns = to_drop
        return self

    def transform(self, X, y=None):
        return X.drop(columns=list(self.columns), errors="ignore")




## === cell 19
pipe = Pipeline(
    [
        ("correlation", CorrelationSelector(threshold=0.75)),
        ("importance", LowImportanceSelector(threshold=1e-4)),
        ("scaler", MinMaxScaler()),
        (
            "xgboost",
            XGBRegressor(
                objective="reg:squarederror",
                n_estimators=10,
                random_state=1337,
                n_jobs=os.cpu_count() or 1,
            ),
        ),
    ]
)



## === cell 20
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=1337
)



## === cell 21
pipe.fit(X_train, y_train)



## === cell 22
from sklearn.metrics import mean_absolute_error

valid_pred = pipe.predict(X_valid)
mae = mean_absolute_error(y_valid, valid_pred)
mae



## === cell 23
pipe.fit(X, y)



## === cell 24
X_pred = df_test_feat[common_features]
test_pred = pipe.predict(X_pred)

pred_df = pd.DataFrame(
    {"segment_id": df_test_feat["segment"].astype(str), "time_to_eruption": test_pred}
)

sample_sub = pd.read_csv(sample_sub_path)
sample_sub["segment_id"] = sample_sub["segment_id"].astype(str)

pred_df = sample_sub[["segment_id"]].merge(pred_df, on="segment_id", how="left")

if pred_df["time_to_eruption"].isna().any():
    pred_df["time_to_eruption"] = pred_df["time_to_eruption"].fillna(
        float(np.median(y))
    )

pred_df.to_csv("submission.csv", index=False)
pred_df.head()



## === cell 25
assert Path("submission.csv").exists(), "submission.csv was not created"
sub_check = pd.read_csv("submission.csv")
assert list(sub_check.columns) == [
    "segment_id",
    "time_to_eruption",
], f"Bad submission columns: {sub_check.columns.tolist()}"
assert len(sub_check) == len(
    sample_sub
), f"Bad submission length: {len(sub_check)} vs {len(sample_sub)}"
sub_check.describe(include="all")
