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

np.random.seed(123)



## === cell 1
filename_list = []
file_path = r"/kaggle/input/predict-volcanic-eruptions-ingv-oe/train"
all_files = glob.glob(file_path + "/*.csv")

filename_list = all_files

list_sequence = []
for file in all_files:
    file = file.split("/")[-1]
    file = file.split(".")[-2]
    list_sequence.append(int(file))

df_list_sequence = pd.DataFrame(list_sequence)
df_list_sequence.columns = ["segment_id"]



## === cell 2
df_list_sequence.head()



## === cell 3
filename_list_test = []
file_path_test = r"/kaggle/input/predict-volcanic-eruptions-ingv-oe/test"
all_files_test = glob.glob(file_path_test + "/*.csv")

filename_list_test = all_files_test

list_sequence_test = []
for file in all_files_test:
    file = file.split("/")[-1]
    file = file.split(".")[-2]
    list_sequence_test.append(int(file))

df_list_sequence_test = pd.DataFrame(list_sequence_test)
df_list_sequence_test.columns = ["segment_id"]



## === cell 4
train = pd.read_csv("/kaggle/input/predict-volcanic-eruptions-ingv-oe/train.csv")
train.head()



## === cell 5

SENSOR_COLS = [f"sensor_{i}" for i in range(1, 11)]


def _safe_stats(x: np.ndarray) -> dict:
    x = x.astype(np.float32, copy=False)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
            "median": np.nan,
            "q25": np.nan,
            "q75": np.nan,
            "skew": np.nan,
            "kurt": np.nan,
            "rms": np.nan,
        }
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "median": float(np.median(x)),
        "q25": float(np.quantile(x, 0.25)),
        "q75": float(np.quantile(x, 0.75)),
        "skew": float(skew(x, bias=False)) if x.size > 2 else np.nan,
        "kurt": float(kurtosis(x, fisher=True, bias=False)) if x.size > 3 else np.nan,
        "rms": float(np.sqrt(np.mean(x * x))),
    }


def extract_features_from_file(csv_path: str) -> dict:
    seg_id = int(os.path.basename(csv_path).replace(".csv", ""))
    df = pd.read_csv(csv_path, usecols=SENSOR_COLS)
    feats = {"segment_id": seg_id}

    for c in SENSOR_COLS:
        arr = df[c].to_numpy(dtype=np.float32, copy=False)
        st = _safe_stats(arr)
        for k, v in st.items():
            feats[f"{c}_{k}"] = v

        dif = np.diff(arr)
        dif = dif[np.isfinite(dif)]
        feats[f"{c}_diff_abs_mean"] = (
            float(np.mean(np.abs(dif))) if dif.size else np.nan
        )

    mat = df.to_numpy(dtype=np.float32, copy=False)
    row_mean = np.nanmean(mat, axis=1)
    st_rm = _safe_stats(row_mean)
    for k, v in st_rm.items():
        feats[f"rowmean_{k}"] = v

    return feats


def build_feature_dataframe(file_paths: list) -> pd.DataFrame:
    rows = []
    for p in sorted(file_paths):
        rows.append(extract_features_from_file(p))
    out = pd.DataFrame(rows)
    for col in out.columns:
        if col != "segment_id":
            out[col] = out[col].astype(np.float32)
    return out




## === cell 6
train_features = build_feature_dataframe(all_files)
test_features = build_feature_dataframe(all_files_test)

train_features = train_features.merge(
    train[["segment_id", "time_to_eruption"]], on="segment_id", how="left"
)

feature_cols = [
    c for c in train_features.columns if c not in ["segment_id", "time_to_eruption"]
]
medians = train_features[feature_cols].median(numeric_only=True)
train_features[feature_cols] = train_features[feature_cols].fillna(medians)
test_features[feature_cols] = test_features[feature_cols].fillna(medians)

train_features.head(2)



## === cell 7
X_train = train_features.drop(["time_to_eruption", "segment_id"], axis=1)
y_train = train_features[["time_to_eruption"]]

X_test = test_features.drop(["segment_id"], axis=1)

X_names = list(X_train.columns)



## === cell 8
scalerx = preprocessing.RobustScaler(quantile_range=(25.0, 75.0))
X_train_scaled = scalerx.fit_transform(X_train)
X_train_scaled = pd.DataFrame(
    X_train_scaled, columns=X_train.columns, index=X_train.index
)

X_test_scaled = scalerx.transform(X_test)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

scalery = preprocessing.RobustScaler(quantile_range=(25.0, 75.0))
y_train_scaled = scalery.fit_transform(y_train)
y_train_scaled = pd.DataFrame(
    y_train_scaled, columns=y_train.columns, index=y_train.index
)

y_train_scaled.head()



## === cell 9
X_train_scaled = X_train_scaled[X_names].values
X_test_scaled = X_test_scaled[X_names].values
y_train_scaled = y_train_scaled.values.ravel()



## === cell 10
df_X_train = pd.DataFrame(X_train_scaled, columns=X_names)
df_y_train = pd.DataFrame(y_train_scaled, columns=["time_to_eruption"])
Xy_train = pd.concat([df_X_train, df_y_train], axis=1, sort=False)




## === cell 11
from sklearn.svm import SVR

svr_rbf = SVR(kernel="rbf", C=10, gamma=0.1, degree=2, epsilon=0.1, coef0=0)

svr_rbf.fit(X_train_scaled, y_train_scaled)

predicted_scaled = svr_rbf.predict(X_test_scaled)
predicted = scalery.inverse_transform(predicted_scaled.reshape(-1, 1)).reshape(-1)



## === cell 12
import xgboost as xgb

xg_reg = xgb.XGBRegressor(
    n_estimators=25,
    seed=123,
    max_depth=6,
    eta=0.5,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    n_jobs=4,
    subsample=1.0,
    reg_lambda=1.0,
)

xg_reg.fit(X_train_scaled, y_train_scaled)

preds_scaled = xg_reg.predict(X_test_scaled)
predicted3 = scalery.inverse_transform(preds_scaled.reshape(-1, 1)).reshape(-1)



## === cell 13
sample_sub = pd.read_csv(
    "/kaggle/input/predict-volcanic-eruptions-ingv-oe/sample_submission.csv"
)
test_pred_df = pd.DataFrame(
    {
        "segment_id": test_features["segment_id"].astype(np.int64),
        "time_to_eruption": predicted3,
    }
)

submission = sample_sub.merge(
    test_pred_df, on="segment_id", how="left", suffixes=("", "_pred")
)
if submission["time_to_eruption"].isna().any():
    submission["time_to_eruption"] = submission["time_to_eruption"].fillna(
        np.nanmedian(predicted3)
    )

submission = submission[["segment_id", "time_to_eruption"]]
submission.to_csv("submission.csv", header=True, index=False)

submission.head()
