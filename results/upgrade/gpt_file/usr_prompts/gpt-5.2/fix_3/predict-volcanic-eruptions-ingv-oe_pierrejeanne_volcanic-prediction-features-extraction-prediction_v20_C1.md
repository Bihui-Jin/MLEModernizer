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

# 5. Code solution

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


def _read_segment_csv(fp: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(fp, usecols=SENSOR_COLS, dtype=DTYPE_MAP, engine="pyarrow")
    except Exception:
        return pd.read_csv(fp, usecols=SENSOR_COLS, dtype=DTYPE_MAP)


def extract_features_from_segment(df: pd.DataFrame) -> dict:
    feats = {}
    X = df.to_numpy(dtype=np.float32, copy=False)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    means = X.mean(axis=0, dtype=np.float64)
    stds = X.std(axis=0, dtype=np.float64)
    mins = X.min(axis=0)
    maxs = X.max(axis=0)

    q25 = np.quantile(X, 0.25, axis=0)
    q50 = np.quantile(X, 0.50, axis=0)
    q75 = np.quantile(X, 0.75, axis=0)

    sk = skew(X.astype(np.float64, copy=False), axis=0, nan_policy="omit", bias=True)
    ku = kurtosis(
        X.astype(np.float64, copy=False),
        axis=0,
        nan_policy="omit",
        bias=True,
        fisher=True,
    )

    for i, col in enumerate(df.columns):
        feats[f"{col}_mean"] = float(means[i])
        feats[f"{col}_std"] = float(stds[i])
        feats[f"{col}_min"] = float(mins[i])
        feats[f"{col}_max"] = float(maxs[i])
        feats[f"{col}_q25"] = float(q25[i])
        feats[f"{col}_q50"] = float(q50[i])
        feats[f"{col}_q75"] = float(q75[i])
        feats[f"{col}_skew"] = float(sk[i]) if np.isfinite(sk[i]) else 0.0
        feats[f"{col}_kurtosis"] = float(ku[i]) if np.isfinite(ku[i]) else 0.0
    return feats


def _featurize_one(sid_folder):
    sid, folder = sid_folder
    fp = folder / f"{int(sid)}.csv"
    seg = _read_segment_csv(fp)
    feats = extract_features_from_segment(seg)
    feats["segment_id"] = int(sid)
    return feats


def build_feature_table(segment_ids, folder: Path) -> pd.DataFrame:
    segment_ids = [int(x) for x in segment_ids]
    tasks = [(sid, folder) for sid in segment_ids]

    max_workers = min(8, mp.cpu_count())
    rows = []
    if max_workers <= 1:
        for t in tasks:
            rows.append(_featurize_one(t))
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            for feats in ex.map(_featurize_one, tasks, chunksize=32):
                rows.append(feats)

    return pd.DataFrame(rows)




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
