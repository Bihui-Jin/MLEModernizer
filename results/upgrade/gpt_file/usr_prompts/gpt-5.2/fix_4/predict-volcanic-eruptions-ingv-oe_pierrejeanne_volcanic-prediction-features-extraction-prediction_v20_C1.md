# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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
librosa==0.11.0
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
seaborn==0.12.2
sklearn-pandas==2.2.0
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

13666381.132300884

# 6. Current score

4703257.0

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 4925768.0) has done: 'I remove the dependency on missing external “features-from-version-16” CSVs by computing features directly from the provided per-segment sensor files in `/kaggle/input/predict-volcanic-eruptions-ingv-oe/{train,test}`. I keep the same core modeling logic (RobustScaler for X and y, then an XGBoost regressor with the same hyperparameters) but fix a scaling bug by transforming the test set with `scalerx.transform` (not `fit_transform`) to avoid leakage and mismatched scaling. I also ensure train/test feature columns align and that predictions are inverse-transformed back to the original target scale before writing a valid `submission4.csv` with the required columns. Finally, I keep runtime within limits by using a lightweight set of per-sensor statistical features (mean/std/min/max/skew/kurtosis + quantiles), which is sufficient to run end-to-end and usually yields a reasonable MAE for this competition.'
- What this solution (achieved 4703257.0) has done: 'The timeout is dominated by per-file CSV parsing across ~4k train segments and repeated heavy feature computations (quantiles + scipy skew/kurtosis) inside Python loops. I keep the exact same features and model, but make them cheaper by: (1) reading faster with a deterministic dtype + `float32` + `na_values` and avoiding the costly pyarrow fallback attempt per file, (2) computing skew/kurtosis with fully vectorized NumPy formulas (equivalent to SciPy with `bias=True, fisher=True`) to remove SciPy overhead, (3) reducing overhead in the multiprocessing pipeline (initializer + larger chunksize, streamlined task passing), and (4) avoiding pandas object overhead by building a fixed-column NumPy feature matrix and converting to DataFrame once. These changes preserve evaluation semantics (same features, scaling, and XGBoost training) and should fit within 600 seconds on Kaggle CPUs.'

# 9. Code solution

## === cell 0
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from scipy.signal import butter, filtfilt
from scipy.stats import skew, kurtosis
from numpy.fft import fft, fftfreq

import librosa as lr
from librosa.core import stft, amplitude_to_db

from sklearn import preprocessing, model_selection, metrics
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)



## === cell 1
filename_list = []
file_path = r"/kaggle/input/predict-volcanic-eruptions-ingv-oe/train"
all_files = glob.glob(file_path + "/*.csv")

filename_list.append(all_files)
filename_list.append(all_files)

list_sequence = []
for file in all_files:
    file = file.split("/")[-1]
    file = file.split(".")[-2]
    list_sequence.append(int(file))

df_list_sequence = pd.DataFrame(list_sequence)
df_list_sequence.columns = ["segment_id"]



## === cell 2
df_list_sequence



## === cell 3
filename_list_test = []
file_path_test = r"/kaggle/input/predict-volcanic-eruptions-ingv-oe/test"
all_files_test = glob.glob(file_path_test + "/*.csv")

filename_list_test.append(all_files_test)
filename_list_test.append(all_files_test)

list_sequence_test = []
for file in all_files_test:
    file = file.split("/")[-1]
    file = file.split(".")[-2]
    list_sequence_test.append(int(file))

df_list_sequence_test = pd.DataFrame(list_sequence_test)
df_list_sequence_test.columns = ["segment_id"]



## === cell 4
#     """
#     remove frequency lower than cutoff
#     Parameters
#     ----------
#     data : array_like
#     cutoff: int
#     fs: float
#     order: int
#     Returns
#     -------
#     y:numpy.ndarray



## === cell 5
from pathlib import Path

DATA_ROOT = Path("/kaggle/input/predict-volcanic-eruptions-ingv-oe")
TRAIN_META_PATH = DATA_ROOT / "train.csv"
TRAIN_DIR = DATA_ROOT / "train"
TEST_DIR = DATA_ROOT / "test"
SAMPLE_SUB_PATH = DATA_ROOT / "sample_submission.csv"

train_meta = pd.read_csv(TRAIN_META_PATH)
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

train_meta["segment_id"] = train_meta["segment_id"].astype(np.int64)
sample_sub["segment_id"] = sample_sub["segment_id"].astype(np.int64)

train_meta.head()



## === cell 6
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

SENSOR_COLS = [f"sensor_{i}" for i in range(1, 11)]
DTYPE_MAP = {c: np.float32 for c in SENSOR_COLS}

_READ_KW = dict(
    usecols=SENSOR_COLS,
    dtype=DTYPE_MAP,
    na_values=["", "nan", "NaN", "NA", "null", "None"],
    keep_default_na=True,
)


def _read_segment_csv(fp: Path) -> np.ndarray:
    df = pd.read_csv(fp, **_READ_KW)
    return df.to_numpy(dtype=np.float32, copy=False)


def _skew_kurtosis_bias_fisher(x64: np.ndarray):
    n = x64.shape[0]
    mu = x64.mean(axis=0)
    d = x64 - mu
    m2 = np.mean(d * d, axis=0)
    m3 = np.mean(d * d * d, axis=0)
    m4 = np.mean(d * d * d * d, axis=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        skew_biased = m3 / np.power(m2, 1.5)
        kurt_fisher_biased = m4 / (m2 * m2) - 3.0
    return skew_biased, kurt_fisher_biased


def extract_features_from_segment_array(X: np.ndarray) -> np.ndarray:
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0, copy=False)

    means = X.mean(axis=0, dtype=np.float64)
    stds = X.std(axis=0, dtype=np.float64)
    mins = X.min(axis=0)
    maxs = X.max(axis=0)

    q25 = np.quantile(X, 0.25, axis=0)
    q50 = np.quantile(X, 0.50, axis=0)
    q75 = np.quantile(X, 0.75, axis=0)

    x64 = X.astype(np.float64, copy=False)
    sk, ku = _skew_kurtosis_bias_fisher(x64)

    out = np.empty(10 * 9, dtype=np.float64)
    k = 0
    for i in range(10):
        out[k] = means[i]
        k += 1
        out[k] = stds[i]
        k += 1
        out[k] = mins[i]
        k += 1
        out[k] = maxs[i]
        k += 1
        out[k] = q25[i]
        k += 1
        out[k] = q50[i]
        k += 1
        out[k] = q75[i]
        k += 1
        out[k] = sk[i] if np.isfinite(sk[i]) else 0.0
        k += 1
        out[k] = ku[i] if np.isfinite(ku[i]) else 0.0
        k += 1
    return out


_FEATURE_SUFFIXES = [
    "mean",
    "std",
    "min",
    "max",
    "q25",
    "q50",
    "q75",
    "skew",
    "kurtosis",
]
FEATURE_COLS = [f"{c}_{s}" for c in SENSOR_COLS for s in _FEATURE_SUFFIXES]


def _featurize_one(args):
    sid, folder = args
    fp = folder / f"{int(sid)}.csv"
    X = _read_segment_csv(fp)
    feats = extract_features_from_segment_array(X)
    return int(sid), feats


def build_feature_table(segment_ids, folder: Path) -> pd.DataFrame:
    segment_ids = np.asarray(segment_ids, dtype=np.int64)
    n = segment_ids.shape[0]
    feats_mat = np.empty((n, len(FEATURE_COLS)), dtype=np.float64)

    tasks = [(int(sid), folder) for sid in segment_ids.tolist()]

    max_workers = min(8, mp.cpu_count())
    if max_workers <= 1:
        for i, t in enumerate(tasks):
            sid, feats = _featurize_one(t)
            feats_mat[i, :] = feats
        out = pd.DataFrame(feats_mat, columns=FEATURE_COLS)
        out.insert(0, "segment_id", segment_ids)
        return out

    chunksize = 64
    ids_out = np.empty(n, dtype=np.int64)
    i = 0
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        for sid, feats in ex.map(_featurize_one, tasks, chunksize=chunksize):
            ids_out[i] = sid
            feats_mat[i, :] = feats
            i += 1

    out = pd.DataFrame(feats_mat, columns=FEATURE_COLS)
    out.insert(0, "segment_id", ids_out)
    return out




## === cell 7
train_features = build_feature_table(train_meta["segment_id"].values, TRAIN_DIR)
train_features = train_features.merge(
    train_meta[["segment_id", "time_to_eruption"]], on="segment_id", how="left"
)

test_features = build_feature_table(sample_sub["segment_id"].values, TEST_DIR)

train_features.head(2), test_features.head(2)



## === cell 8
feature_cols = [
    c for c in train_features.columns if c not in ["segment_id", "time_to_eruption"]
]
missing_in_test = [c for c in feature_cols if c not in test_features.columns]
for c in missing_in_test:
    test_features[c] = 0.0

test_features = test_features[["segment_id"] + feature_cols]
train_features = train_features[["segment_id"] + feature_cols + ["time_to_eruption"]]



## === cell 9
X_train = train_features.drop(["time_to_eruption", "segment_id"], axis=1)
y_train = train_features[["time_to_eruption"]]

X_test = test_features.drop(["segment_id"], axis=1)

X_train.shape, X_test.shape, y_train.shape



## === cell 10
scalerx = preprocessing.RobustScaler(quantile_range=(25.0, 75.0))
X_train_scaled = scalerx.fit_transform(X_train)
X_test_scaled = scalerx.transform(X_test)

scalery = preprocessing.RobustScaler(quantile_range=(25.0, 75.0))
y_train_scaled = scalery.fit_transform(y_train)

pd.DataFrame(y_train_scaled, columns=y_train.columns, index=y_train.index).head()



## === cell 11
import xgboost as xgb



## === cell 12
gbm = xgb.XGBRegressor(
    objective="reg:squarederror",  # 'reg:linear' deprecated/removed in newer xgboost
    n_estimators=100,
    max_depth=6,
    eta=0.3,
    colsample_bytree=0.4,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

gbm.fit(X_train_scaled, y_train_scaled.ravel())

preds = gbm.predict(X_test_scaled)



## === cell 13
predicted4 = scalery.inverse_transform(preds.reshape(-1, 1))



## === cell 14
submission = sample_sub.copy()
pred_map = dict(
    zip(test_features["segment_id"].astype(np.int64).values, predicted4.reshape(-1))
)
submission["time_to_eruption"] = submission["segment_id"].map(pred_map).astype(float)

if submission["time_to_eruption"].isna().any():
    submission["time_to_eruption"] = submission["time_to_eruption"].fillna(
        float(np.nanmedian(predicted4))
    )

submission.to_csv("submission4.csv", header=True, index=False)
submission.head()
