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

df_min = pd.read_csv(data_folder / "train" / f"{time_min}.csv")
df_max = pd.read_csv(data_folder / "train" / f"{time_max}.csv")

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
from tsfresh.feature_extraction import MinimalFCParameters



## === cell 14
tsfresh_parameters = MinimalFCParameters()



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
_SENSOR_DTYPES = {f"sensor_{i}": np.float32 for i in range(1, 11)}


def _read_segment_csv_fast(path: Path) -> pd.DataFrame:
    seg = pd.read_csv(path, dtype=_SENSOR_DTYPES)
    return seg.fillna(0)


def preprocess_timeseries(
    meta_df, parameters, is_train=True, verbose_every=25, max_segments=None
):
    rows = []

    if is_train:
        segments_iter = meta_df[["segment_id", "time_to_eruption"]].itertuples(
            index=False, name=None
        )
    else:
        test_dir = data_folder / "test"
        files = sorted([p.name for p in test_dir.glob("*.csv")])
        segments_iter = ((fn.replace(".csv", ""), None) for fn in files)

    n_jobs = max(1, (os.cpu_count() or 2) - 1)

    for idx, (segment, time_to_eruption) in enumerate(segments_iter):
        if max_segments is not None and idx >= max_segments:
            break

        if is_train:
            segment_timeseries_path = data_folder / "train" / f"{segment}.csv"
        else:
            segment_timeseries_path = data_folder / "test" / f"{segment}.csv"

        segment_timeseries = _read_segment_csv_fast(segment_timeseries_path)

        segment_timeseries = segment_timeseries.reset_index(drop=False)
        segment_timeseries["id"] = 0  # single id per segment for tsfresh

        extracted_features = extract_features(
            segment_timeseries,
            column_id="id",
            column_sort="index",
            disable_progressbar=True,
            default_fc_parameters=parameters,
            n_jobs=n_jobs,
        )
        extracted_features["segment"] = str(segment)

        if is_train:
            extracted_features["time_to_eruption"] = float(time_to_eruption)

        rows.append(extracted_features)

        if verbose_every and ((idx + 1) % verbose_every == 0):
            print(f"Processed {idx + 1} segments")

    if not rows:
        df_result = pd.DataFrame()
    else:
        df_result = pd.concat(rows, axis=0, ignore_index=True, sort=False)

    df_result = df_result.replace([np.inf, -np.inf], np.nan)
    df_result = df_result.fillna(0)
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
TRAIN_FEAT_PATH = Path("train_features.csv")
if TRAIN_FEAT_PATH.exists():
    df_train_feat = pd.read_csv(TRAIN_FEAT_PATH)
else:
    df_train_feat = preprocess_timeseries(
        df, tsfresh_parameters, is_train=True, verbose_every=100
    )
    df_train_feat.to_csv(TRAIN_FEAT_PATH, index=False)

df_train_feat = df_train_feat.dropna(axis="columns")  # keep original intent

features = [
    c for c in df_train_feat.columns if c not in ["time_to_eruption", "segment"]
]
X = df_train_feat[features]
y = df_train_feat["time_to_eruption"]

df_train_feat.head()



## === cell 23
X = X.apply(pd.to_numeric, errors="coerce").fillna(0)




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
        Xc = X.copy()
        self.columns = set()
        C = Xc.corr(numeric_only=True)
        for i in range(len(C.columns)):
            for j in range(i):
                if (C.iloc[i, j] >= self.threshold) and (
                    C.columns[j] not in self.columns
                ):
                    c = C.columns[i]
                    self.columns.add(c)
                    if c in Xc.columns:
                        del Xc[c]
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
                n_jobs=-1,
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
pipe.fit(X, y)



## === cell 32
TEST_FEAT_PATH = Path("test_features.csv")
if TEST_FEAT_PATH.exists():
    df_test_feat = pd.read_csv(TEST_FEAT_PATH)
else:
    df_test_feat = preprocess_timeseries(
        None, tsfresh_parameters, is_train=False, verbose_every=100
    )
    df_test_feat.to_csv(TEST_FEAT_PATH, index=False)

df_test_feat = df_test_feat.dropna(axis="columns")

df_test_feat["segment"] = df_test_feat["segment"].astype(str)

X_test_feat = df_test_feat.drop(columns=["segment"], errors="ignore")
X_test_feat = X_test_feat.apply(pd.to_numeric, errors="coerce").fillna(0)

X_test_feat = X_test_feat.reindex(columns=X.columns, fill_value=0)



## === cell 33
pred = pipe.predict(X_test_feat)

submission = pd.DataFrame(
    {"segment_id": df_test_feat["segment"].astype(str), "time_to_eruption": pred}
)
submission = submission[["segment_id", "time_to_eruption"]]
submission.to_csv("submission.csv", index=False)

submission.head()
