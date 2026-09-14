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
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import TransformerMixin
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error

from xgboost import XGBRegressor
from tsfresh import extract_features
from tsfresh.feature_extraction import MinimalFCParameters



## === cell 1
data_folder = Path("../input/predict-volcanic-eruptions-ingv-oe/")



## === cell 2
train_meta = pd.read_csv(data_folder / "train.csv")
train_meta.head()



## === cell 3
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
    {"coeff": c, "m": 3, "r": 30} for c in range(4)
]
tsfresh_parameters["count_above"] = [{"t": 0}]
tsfresh_parameters["count_below"] = [{"t": 0}]




## === cell 4
def preprocess_timeseries(meta_df, parameters, is_train=True):
    """
    Extract TSFresh features for each segment.
    - If is_train=True, meta_df must contain 'segment_id' and 'time_to_eruption'.
    - If is_train=False, meta_df is ignored and all files in the test folder are processed.
    Returns a DataFrame with one row per segment.
    """
    results = []
    if is_train:
        iterator = meta_df.iterrows()
    else:
        files = sorted(os.listdir(data_folder / "test"))
        iterator = enumerate(files)

    for idx, row in iterator:
        if is_train:
            segment = str(row["segment_id"])
            target = row["time_to_eruption"]
        else:
            segment = str(row)  # file name, e.g. "12345.csv"
            target = None

        csv_path = data_folder / ("train" if is_train else "test") / f"{segment}.csv"
        ts = pd.read_csv(csv_path)

        ts = ts.fillna(0).reset_index()
        ts["id"] = idx

        feats = extract_features(
            ts,
            column_id="id",
            column_sort="index",
            default_fc_parameters=parameters,
            disable_progressbar=True,
        )
        feats["segment"] = segment
        if is_train:
            feats["time_to_eruption"] = target

        results.append(feats)
        print(f"Processed segment {segment}")
    return pd.concat(results, ignore_index=True)




## === cell 5
df_train = preprocess_timeseries(train_meta, tsfresh_parameters, is_train=True)

df_train = df_train.dropna(axis="columns")
df_train.head()



## === cell 6
df_test = preprocess_timeseries(None, tsfresh_parameters, is_train=False)
df_test = df_test.dropna(axis="columns")
df_test.head()



## === cell 7
feature_cols = [c for c in df_train.columns if c not in ["time_to_eruption", "segment"]]
X = df_train[feature_cols]
y = df_train["time_to_eruption"]

X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=1337)



## === cell 8
model_params = {
    "lambda": 0.0020555245431348778,
    "alpha": 0.11298627316540845,
    "colsample_bytree": 0.6,
    "subsample": 1.0,
    "learning_rate": 0.01,
    "max_depth": 20,
    "random_state": 48,
    "min_child_weight": 18,
    "objective": "reg:squarederror",
    "n_estimators": 1000,
    "tree_method": "hist",  # fall back to CPU if GPU not available
}


class CorrelationSelector(TransformerMixin):
    def __init__(self, threshold=0.85):
        self.threshold = threshold
        self.columns_to_drop = []

    def fit(self, X, y=None):
        corr = X.corr().abs()
        cols = corr.columns
        for i in range(len(cols)):
            for j in range(i):
                if corr.iloc[i, j] >= self.threshold:
                    self.columns_to_drop.append(cols[i])
                    break
        return self

    def transform(self, X, y=None):
        return X.drop(columns=self.columns_to_drop, errors="ignore")


pipe = Pipeline(
    [
        ("corr_sel", CorrelationSelector(threshold=0.85)),
        ("scaler", MinMaxScaler()),
        ("xgb", XGBRegressor(**model_params)),
    ]
)

pipe.fit(X_tr, y_tr)

val_pred = pipe.predict(X_val)
print("Validation MAE:", mean_absolute_error(y_val, val_pred))



## === cell 9
pipe.fit(X, y)

test_features = df_test[feature_cols]
test_pred = pipe.predict(test_features)

submission = pd.DataFrame(
    {"segment_id": df_test["segment"], "time_to_eruption": test_pred}
)

submission = submission.sort_values("segment_id").reset_index(drop=True)

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
