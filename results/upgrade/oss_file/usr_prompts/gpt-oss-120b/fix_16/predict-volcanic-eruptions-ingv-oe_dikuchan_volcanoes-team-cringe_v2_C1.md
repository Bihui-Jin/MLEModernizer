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

# 5. Target score

6079440.246681859

# 6. Current score

8164858.0

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 8607143.0) has done: 'I filter the test directory to read only CSV files (skipping the stray sub‑directory), ensure the test feature matrix has exactly the same columns as the training data (filling missing ones with 0), and then run the prediction and write a proper `submission.csv`. These small fixes resolve the `IsADirectoryError` and the subsequent `NameError`, allowing the pipeline to finish and produce a valid submission file.'
- What this solution (achieved 7193000.0) has done: 'The changes add richer statistical descriptors (min, max, median, 25th and 75th percentiles) to each segment’s feature set, giving the model more information while preserving the original pipeline. The XGBoost regressor is also given a modest boost in capacity (more trees and deeper depth) to better capture the added features. These adjustments are targeted to lower the MAE and move the score closer to the target without altering the overall workflow.'
- What this solution (achieved 8099401.0) has done: 'The changes switch the parallel feature extraction from process‑based to thread‑based, which is faster for the I/O‑heavy CSV reads, and enable XGBoost’s fast histogram tree building (`tree_method='hist'`). Both adjustments keep the exact same feature set and model semantics while reducing the overall runtime well below the 600‑second limit.'
- What this solution (achieved 7935380.0) has done: 'I add two informative descriptors (skewness and kurtosis) for each sensor to the feature set and slightly increase the X‑GBoost model capacity (more trees and a deeper depth). These modest extensions give the model a bit more signal to reduce MAE, moving the score closer to the target while keeping the overall pipeline unchanged.'
- What this solution (achieved 8190787.0) has done: 'I add two simple descriptors—range and variance—for each sensor to give the model a bit more signal, and I slightly increase the XGBoost capacity (more trees and deeper depth) which should lower the MAE toward the target while keeping the overall pipeline unchanged.'
- What this solution (achieved 8106188.0) has done: 'I remove the noisy “range” and “variance” features that worsened the score, add a few robust descriptors (inter‑quartile range, median absolute deviation, and overall median), and slightly adjust the XGBoost hyper‑parameters (more trees, lower learning rate, shallower depth). These small, targeted changes keep the overall pipeline unchanged while providing extra signal and reducing over‑fitting, moving the MAE closer to the target.'
- What this solution (achieved 8506090.0) has done: 'I add a per‑sensor variance feature (computed from the existing standard deviations) and include it in the feature list, then give the XGBoost model a slightly larger capacity (more trees and a deeper depth with a modestly lower learning rate). These minimal, targeted changes provide the model with extra signal while keeping the overall pipeline unchanged, aiming to lower the validation MAE toward the target.'
- What this solution (achieved 8304798.0) has done: 'I keep the overall feature extraction and data handling unchanged, but replace the Pipeline with an explicit scaler + XGBoost fitting that uses early stopping and a slightly tighter set of hyper‑parameters (more trees, lower learning‑rate, a bit shallower depth and L2 regularisation). This modest change generally reduces over‑fitting and brings the validation MAE closer to the target without altering the core logic or feature set.'
- What this solution (achieved 8302446.0) has done: 'I drop the unnecessary Min‑Max scaling (tree models don’t need it) and switch XGBoost to the MAE‑optimising objective while giving the model a bit more depth. These minimal tweaks keep the overall pipeline intact but should lower the validation MAE and move the score closer to the target.'
- What this solution (achieved 8897463.0) has done: 'I adjust the XGBoost hyper‑parameters to reduce over‑fitting and better align the model with the MAE metric: shallower trees, a lower learning rate, more potential boosting rounds, stronger L2 regularisation, and a higher early‑stopping patience. These targeted tweaks keep the overall pipeline and feature engineering unchanged while aiming to lower the validation MAE toward the target value.'
- What this solution (achieved 8164858.0) has done: 'I slightly adjust the XGBoost hyper‑parameters to give the model more capacity while keeping regularisation modest (increase max depth, raise learning rate a bit, reduce L2 strength and allow more trees). These minor tweaks aim to improve the fit and therefore lower the validation MAE, moving the score closer to the target without altering any feature engineering or overall pipeline.'

# 9. Code solution

## === cell 0
import os
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt

from sklearn.base import TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from xgboost import XGBRegressor

import concurrent.futures




## === cell 1
data_folder = Path("../input/predict-volcanic-eruptions-ingv-oe/")




## === cell 2
df_meta = pd.read_csv(data_folder / "train.csv")
df_meta.head().T




## === cell 3
_sensor_names = [f"sensor_{i+1}" for i in range(10)]
_FEATURE_IDX = (
    [f"mean_{s}" for s in _sensor_names]
    + [f"std_{s}" for s in _sensor_names]
    + [f"var_{s}" for s in _sensor_names]  # new variance descriptor
    + [f"min_{s}" for s in _sensor_names]
    + [f"max_{s}" for s in _sensor_names]
    + [f"median_{s}" for s in _sensor_names]
    + [f"q25_{s}" for s in _sensor_names]
    + [f"q75_{s}" for s in _sensor_names]
    + [f"iqr_{s}" for s in _sensor_names]  # inter‑quartile range
    + [f"mad_{s}" for s in _sensor_names]  # median absolute deviation
    + [f"skew_{s}" for s in _sensor_names]  # existing descriptor
    + [f"kurt_{s}" for s in _sensor_names]  # existing descriptor
    + ["overall_mean", "overall_std", "overall_median"]  # new overall descriptors
)


def extract_simple_features(segment_path):
    """Fast NumPy‑based extraction of column‑wise statistics and additional
    descriptors for a segment CSV, now including variance, IQR, MAD and overall median.
    """
    arr = pd.read_csv(segment_path, header=0, dtype=np.float32).to_numpy(
        copy=False
    )  # shape (rows, 10)

    means = np.mean(arr, axis=0)
    stds = np.std(arr, axis=0, ddof=1)
    vars_ = stds**2  # variance per sensor
    mins = np.min(arr, axis=0)
    maxs = np.max(arr, axis=0)
    medians = np.median(arr, axis=0)

    q25, q75 = np.percentile(arr, [25, 75], axis=0)
    iqr = q75 - q25  # inter‑quartile range per sensor
    mad = np.median(
        np.abs(arr - medians), axis=0
    )  # median absolute deviation per sensor

    df_arr = pd.DataFrame(arr)
    skews = df_arr.skew().values
    kurts = df_arr.kurtosis().values

    overall_mean = np.mean(arr)
    overall_std = np.mean(np.std(arr, axis=1, ddof=1))
    overall_median = np.median(arr)

    values = np.concatenate(
        [
            means,
            stds,
            vars_,
            mins,
            maxs,
            medians,
            q25,
            q75,
            iqr,
            mad,
            skews,
            kurts,
            [overall_mean, overall_std, overall_median],
        ]
    )
    return pd.Series(values, index=_FEATURE_IDX)




## === cell 4
def _process_train_row(row):
    seg_id = row["segment_id"]
    target = row["time_to_eruption"]
    seg_path = data_folder / "train" / f"{seg_id}.csv"
    feats = extract_simple_features(seg_path)
    feats["segment_id"] = seg_id
    return feats, target


train_features = []
train_targets = []

with concurrent.futures.ThreadPoolExecutor(
    max_workers=min(os.cpu_count() or 4, 8)
) as executor:
    for feats, target in executor.map(
        _process_train_row, [r for _, r in df_meta.iterrows()]
    ):
        train_features.append(feats)
        train_targets.append(target)

df_train = pd.DataFrame(train_features)
df_train["time_to_eruption"] = train_targets
df_train.head()




## === cell 5
features = [c for c in df_train.columns if c not in ["time_to_eruption", "segment_id"]]
X = df_train[features].fillna(0)
y = df_train["time_to_eruption"]




## === cell 6
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=1337
)




## === cell 7
xgb_model = XGBRegressor(
    objective="reg:absoluteerror",  # optimises MAE directly
    eval_metric="mae",
    n_estimators=8000,
    learning_rate=0.01,
    max_depth=12,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=4,
    tree_method="hist",
    verbosity=0,
)




## === cell 8
xgb_model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=400,  # allow a bit more room to converge
    verbose=False,
)




## === cell 9
from sklearn.metrics import mean_absolute_error

val_pred = xgb_model.predict(X_val)
val_mae = mean_absolute_error(y_val, val_pred)
print(f"Validation MAE: {val_mae:.4f}")




## === cell 10
test_dir = data_folder / "test"
test_files = [f for f in os.listdir(test_dir) if f.lower().endswith(".csv")]
test_features = []


def _process_test_file(filename):
    seg_path = test_dir / filename
    feats = extract_simple_features(seg_path)
    feats["segment_id"] = filename.split(".")[0]
    return feats


with concurrent.futures.ThreadPoolExecutor(
    max_workers=min(os.cpu_count() or 4, 8)
) as executor:
    for feats in executor.map(_process_test_file, sorted(test_files)):
        test_features.append(feats)

df_test = pd.DataFrame(test_features)
df_test = df_test.reindex(columns=features + ["segment_id"], fill_value=0)




## === cell 11
test_pred = xgb_model.predict(df_test[features])




## === cell 12
submission = pd.DataFrame(
    {"segment_id": df_test["segment_id"], "time_to_eruption": test_pred}
)
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
