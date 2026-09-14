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

from tsfresh import extract_features
from tsfresh.feature_extraction import MinimalFCParameters



## === cell 1
data_folder = Path("../input/predict-volcanic-eruptions-ingv-oe")

df_meta = pd.read_csv(data_folder / "train.csv")
df_meta.head()



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
    {"coeff": i, "m": 3, "r": 30} for i in range(4)
]
tsfresh_parameters["count_above"] = [{"t": 0}]
tsfresh_parameters["count_below"] = [{"t": 0}]




## === cell 3
def preprocess_timeseries(meta_df, parameters, is_train=True):
    """
    Extract tsfresh features for every segment.
    For speed we keep only the first 1000 rows of each segment.
    """
    feats_list = []  # collect per‑segment feature DataFrames

    if is_train:
        iterator = enumerate(meta_df.itertuples(index=False, name="TrainRow"))
    else:
        test_files = [f for f in os.listdir(data_folder / "test") if f.endswith(".csv")]
        iterator = enumerate(test_files)

    for idx, row in iterator:
        if is_train:
            segment_id = row.segment_id
            time_to_eruption = row.time_to_eruption
            segment_path = data_folder / "train" / f"{segment_id}.csv"
        else:
            filename = row  # string like "12345.csv"
            segment_path = data_folder / "test" / filename
            segment_id = Path(filename).stem
            time_to_eruption = np.nan  # placeholder, not used for test

        ts = pd.read_csv(segment_path).fillna(0).reset_index()
        ts = ts.head(1000)
        ts["id"] = idx  # unique id for tsfresh

        feats = extract_features(
            ts,
            column_id="id",
            column_sort="index",
            default_fc_parameters=parameters,
            disable_progressbar=True,
            n_jobs=1,
        )
        feats["segment"] = segment_id

        if is_train:
            feats["time_to_eruption"] = time_to_eruption

        feats_list.append(feats)

        print(f"Processed segment #{idx} – {segment_id}")

    result_df = pd.concat(feats_list, axis=0, ignore_index=True, sort=True)
    return result_df




## === cell 4
def save_features(parameters):
    """Create processed train / test CSVs in the current working directory."""
    print("--- processing TRAIN ---")
    train_features = preprocess_timeseries(df_meta, parameters, is_train=True)
    train_features.to_csv("train.csv", index=False)

    print("--- processing TEST ---")
    test_features = preprocess_timeseries(None, parameters, is_train=False)
    test_features.to_csv("test.csv", index=False)


save_features(tsfresh_parameters)



## === cell 5
df_train = pd.read_csv("train.csv")
df_train = df_train.dropna(axis="columns", how="all")

feature_cols = [c for c in df_train.columns if c not in ["time_to_eruption", "segment"]]
X = df_train[feature_cols]
y = df_train["time_to_eruption"]




## === cell 6
class LowImportanceSelector(TransformerMixin):
    def __init__(self, threshold, n_estimators=100):
        self.threshold = threshold
        self.n_estimators = n_estimators
        self.features = None

    def fit(self, X, y):
        rf = RandomForestRegressor(n_estimators=self.n_estimators, random_state=42)
        rf.fit(X, y)
        imp = pd.DataFrame(
            {"feature": X.columns, "importance": rf.feature_importances_}
        )
        self.features = imp[imp["importance"] > self.threshold]["feature"].tolist()
        return self

    def transform(self, X):
        return X[self.features]


class CorrelationSelector(TransformerMixin):
    def __init__(self, threshold):
        self.threshold = threshold
        self.to_drop = None

    def fit(self, X, y=None):
        corr = X.corr().abs()
        self.to_drop = set()
        for i in range(len(corr.columns)):
            for j in range(i):
                if corr.iloc[i, j] >= self.threshold:
                    col_i = corr.columns[i]
                    self.to_drop.add(col_i)
        return self

    def transform(self, X, y=None):
        return X.drop(columns=list(self.to_drop), errors="ignore")




## === cell 7
pipe = Pipeline(
    [
        ("correlation", CorrelationSelector(threshold=0.75)),
        ("importance", LowImportanceSelector(threshold=1e-4)),
        ("scaler", MinMaxScaler()),
        (
            "xgboost",
            XGBRegressor(
                objective="reg:squarederror",
                n_estimators=50,
                learning_rate=0.1,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=4,
            ),
        ),
    ]
)



## === cell 8
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=1337
)

pipe.fit(X_train, y_train)



## === cell 9
df_test = pd.read_csv("test.csv")
df_test = df_test.dropna(axis="columns", how="all")
X_test = df_test[feature_cols]



## === cell 10
test_pred = pipe.predict(X_test)

submission = pd.DataFrame(
    {"segment_id": df_test["segment"], "time_to_eruption": test_pred}
)

submission.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")
