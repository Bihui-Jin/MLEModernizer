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

11742913.23097345

# 6. Current score

5574561.0

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

- What this solution (achieved 5574561.0) has done: 'I remove the dependency on the missing `../input/features-from-version-16/*.csv` files by computing the same kind of tabular features directly from the provided per-segment sensor CSVs in `/kaggle/input/predict-volcanic-eruptions-ingv-oe/train` and `/kaggle/input/predict-volcanic-eruptions-ingv-oe/test`. I keep the core modeling approach (SVR with RBF kernel) and keep the same general scaling workflow, but fix a critical bug where the test data scaler was incorrectly refit on test instead of using the train-fitted scaler (this is score-improving and still semantically consistent). I also ensure `segment_id` ordering matches `sample_submission.csv` so predictions align with the required output rows, and I write a valid `submission.csv`. Finally, I remove/disable the unfinished XGBoost randomized search section that currently cannot run due to undefined variables and missing imports.'

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

from sklearn import preprocessing
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



## === cell 1
TRAIN_DIR = "/kaggle/input/predict-volcanic-eruptions-ingv-oe/train"
TEST_DIR = "/kaggle/input/predict-volcanic-eruptions-ingv-oe/test"
TRAIN_META_PATH = "/kaggle/input/predict-volcanic-eruptions-ingv-oe/train.csv"
SAMPLE_SUB_PATH = (
    "/kaggle/input/predict-volcanic-eruptions-ingv-oe/sample_submission.csv"
)

train_meta = pd.read_csv(TRAIN_META_PATH)
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

train_meta.head()



## === cell 2
all_train_files = glob.glob(os.path.join(TRAIN_DIR, "*.csv"))
all_test_files = glob.glob(os.path.join(TEST_DIR, "*.csv"))

train_segment_ids = [
    int(os.path.splitext(os.path.basename(p))[0]) for p in all_train_files
]
test_segment_ids = [
    int(os.path.splitext(os.path.basename(p))[0]) for p in all_test_files
]

df_list_sequence = pd.DataFrame({"segment_id": train_segment_ids})
df_list_sequence_test = pd.DataFrame({"segment_id": test_segment_ids})

df_list_sequence.head(), df_list_sequence_test.head()



## === cell 3
SENSOR_COLS = [f"sensor_{i}" for i in range(1, 11)]


def _safe_stats(x: np.ndarray):
    x = x.astype(np.float32, copy=False)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "q01": 0.0,
            "q05": 0.0,
            "q50": 0.0,
            "q95": 0.0,
            "q99": 0.0,
            "skew": 0.0,
            "kurt": 0.0,
            "mad": 0.0,
            "energy": 0.0,
            "abs_mean": 0.0,
            "abs_max": 0.0,
        }
    q = np.quantile(x, [0.01, 0.05, 0.50, 0.95, 0.99])
    mean = float(np.mean(x))
    std = float(np.std(x))
    mad = float(np.mean(np.abs(x - mean)))
    return {
        "mean": mean,
        "std": std,
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "q01": float(q[0]),
        "q05": float(q[1]),
        "q50": float(q[2]),
        "q95": float(q[3]),
        "q99": float(q[4]),
        "skew": float(skew(x, bias=False)) if x.size > 2 else 0.0,
        "kurt": float(kurtosis(x, fisher=True, bias=False)) if x.size > 3 else 0.0,
        "mad": mad,
        "energy": float(np.mean(x * x)),
        "abs_mean": float(np.mean(np.abs(x))),
        "abs_max": float(np.max(np.abs(x))),
    }


def extract_features_for_segment(csv_path: str, segment_id: int) -> dict:
    df = pd.read_csv(csv_path)
    cols = [c for c in SENSOR_COLS if c in df.columns]
    if not cols:
        raise ValueError(
            f"No sensor columns found in {csv_path}. Columns: {df.columns.tolist()}"
        )

    feats = {"segment_id": int(segment_id)}
    for col in cols:
        x = df[col].to_numpy(dtype=np.float32, copy=False)
        s = _safe_stats(x)
        for k, v in s.items():
            feats[f"{col}_{k}"] = v

        dx = np.diff(x)
        sd = _safe_stats(dx)
        for k, v in sd.items():
            feats[f"{col}_diff_{k}"] = v

    return feats


def build_feature_table(segment_ids, directory, max_rows=None):
    rows = []
    for i, sid in enumerate(segment_ids):
        if max_rows is not None and i >= max_rows:
            break
        path = os.path.join(directory, f"{sid}.csv")
        rows.append(extract_features_for_segment(path, sid))
    return pd.DataFrame(rows)




## === cell 4
train_features = build_feature_table(train_meta["segment_id"].tolist(), TRAIN_DIR)
train_features = train_meta[["segment_id", "time_to_eruption"]].merge(
    train_features, on="segment_id", how="left"
)

test_features = build_feature_table(sample_sub["segment_id"].tolist(), TEST_DIR)
test_features = sample_sub[["segment_id"]].merge(
    test_features, on="segment_id", how="left"
)

train_features.head(2), test_features.head(2)



## === cell 5
X_train = train_features.drop(["time_to_eruption", "segment_id"], axis=1)
y_train = train_features[["time_to_eruption"]]
X_test = test_features.drop(["segment_id"], axis=1)

X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0.0)
X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0.0)
y_train = y_train.replace([np.inf, -np.inf], np.nan).fillna(y_train.median())

X_train.shape, X_test.shape, y_train.shape



## === cell 6
scalerx = preprocessing.RobustScaler(quantile_range=(25.0, 75.0))
X_train_scaled = scalerx.fit_transform(X_train)
X_train_scaled = pd.DataFrame(
    X_train_scaled, columns=X_train.columns, index=X_train.index
)

X_test_scaled = scalerx.transform(X_test)  # FIX
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

scalery = preprocessing.RobustScaler(quantile_range=(25.0, 75.0))
y_train_scaled = scalery.fit_transform(y_train)
y_train_scaled = pd.DataFrame(
    y_train_scaled, columns=y_train.columns, index=y_train.index
)

y_train_scaled.head()



## === cell 7
Xy_train_scaled = pd.concat([X_train_scaled, y_train_scaled], axis=1, sort=False)

correlation_coef_scale = Xy_train_scaled.corr(numeric_only=True)[
    "time_to_eruption"
].drop("time_to_eruption")
df_cor = pd.DataFrame(
    correlation_coef_scale.sort_values(ascending=False), columns=["time_to_eruption"]
)

selected = df_cor[df_cor["time_to_eruption"] < -0.24]
if selected.shape[0] == 0:
    selected = df_cor.reindex(
        df_cor["time_to_eruption"].abs().sort_values(ascending=False).head(100).index
    )

X_names = selected.index.tolist()
len(X_names), X_names[:10]



## === cell 8
X_train_sel = X_train_scaled[X_names].to_numpy()
X_test_sel = X_test_scaled[X_names].to_numpy()
y_train_sel = y_train_scaled.to_numpy().ravel()

X_train_sel.shape, X_test_sel.shape, y_train_sel.shape



## === cell 9
DO_PLOTS = False
if DO_PLOTS:
    df_X_train = pd.DataFrame(X_train_sel, columns=X_names)
    df_y_train = pd.DataFrame(y_train_sel, columns=["time_to_eruption"])
    Xy_train_plot = pd.concat([df_X_train, df_y_train], axis=1, sort=False)
    Xy_train_plot.plot(
        x="time_to_eruption",
        y=X_names[0:],
        kind="line",
        legend=False,
        subplots=True,
        sharex=True,
        figsize=(20, 30),
        ls="none",
        marker="o",
        layout=(10, 9),
    )
    plt.show()



## === cell 10
from sklearn.svm import SVR

svr_rbf = SVR(kernel="rbf", C=10, gamma=0.1, degree=2, epsilon=0.1, coef0=0)
svr_rbf.fit(X_train_sel, y_train_sel)

pred_scaled = svr_rbf.predict(X_test_sel).reshape(-1, 1)
predicted = scalery.inverse_transform(pred_scaled).ravel()

predicted[:5], predicted.shape



## === cell 11
submission = sample_sub.copy()
submission["time_to_eruption"] = predicted

submission["segment_id"] = submission["segment_id"].astype(int)
submission["time_to_eruption"] = submission["time_to_eruption"].astype(float)

submission.head()



## === cell 12
SUB_PATH = "submission.csv"
submission.to_csv(SUB_PATH, index=False)
print(
    f"Wrote {SUB_PATH} with shape {submission.shape} and columns {submission.columns.tolist()}"
)



## === cell 13
pass
