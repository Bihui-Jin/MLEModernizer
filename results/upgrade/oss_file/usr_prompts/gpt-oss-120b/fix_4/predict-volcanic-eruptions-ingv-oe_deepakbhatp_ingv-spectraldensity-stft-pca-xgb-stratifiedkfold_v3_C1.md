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

5140185.634133191

# 6. Current score

13684299.0

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 13769488.0) has done: 'The changes parallelize the expensive CSV feature extraction and make it leaner by reading data as float32 and returning a simple NumPy array of means and standard deviations, avoiding costly Pandas concatenations.  This reduces I/O and memory overhead while preserving exactly the same 20‑dimensional feature vector for each segment, so the model training and predictions remain unchanged.  All other logic, model hyper‑parameters, and evaluation remain identical.'
- What this solution (achieved 13684299.0) has done: 'I enhance the feature extraction to include additional descriptive statistics (min, max, median) for each sensor, expanding the feature vector from 20 to 50 elements. These richer features should help the XGBoost model predict more accurately and lower the MAE toward the target while keeping the modeling pipeline unchanged.'

# 9. Code solution

## === cell 0
import os
import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import xgboost as xgb
from multiprocessing import Pool, cpu_count



## === cell 1
BASE_PATH = "../input/predict-volcanic-eruptions-ingv-oe"
TRAIN_META_PATH = os.path.join(BASE_PATH, "train.csv")
TRAIN_SEG_PATH = os.path.join(BASE_PATH, "train")
TEST_SEG_PATH = os.path.join(BASE_PATH, "test")
SAMPLE_SUB_PATH = os.path.join(BASE_PATH, "sample_submission.csv")


def extract_features_arr(csv_path):
    """
    Read a segment CSV as float32 and return a 50‑element array:
    For each of the 10 sensors compute:
        mean, std, min, max, median  (5 stats * 10 sensors)
    This richer representation provides more predictive power while
    preserving the original workflow.
    """
    df = pd.read_csv(csv_path, dtype=np.float32)
    means = df.mean().values
    stds = df.std().values
    mins = df.min().values
    maxs = df.max().values
    medians = df.median().values
    return np.concatenate([means, stds, mins, maxs, medians])




## === cell 2
train_meta = pd.read_csv(TRAIN_META_PATH)
train_meta = train_meta.sort_values("segment_id").reset_index(drop=True)

train_paths = [
    os.path.join(TRAIN_SEG_PATH, f"{sid}.csv") for sid in train_meta["segment_id"]
]

with Pool(cpu_count()) as p:
    train_features = p.map(extract_features_arr, train_paths)

train_X = pd.DataFrame(train_features)
train_y = train_meta["time_to_eruption"].values



## === cell 3
test_files = sorted(glob.glob(os.path.join(TEST_SEG_PATH, "*.csv")))
test_segment_ids = [int(os.path.splitext(os.path.basename(p))[0]) for p in test_files]

with Pool(cpu_count()) as p:
    test_features = p.map(extract_features_arr, test_files)

test_X = pd.DataFrame(test_features)



## === cell 4
kf = KFold(n_splits=5, shuffle=True, random_state=42)
test_pred = np.zeros(len(test_X))

for fold, (train_idx, val_idx) in enumerate(kf.split(train_X), 1):
    X_tr, X_val = train_X.iloc[train_idx], train_X.iloc[val_idx]
    y_tr, y_val = train_y[train_idx], train_y[val_idx]

    model = xgb.XGBRegressor(
        n_estimators=12000,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.6,
        colsample_bytree=0.5,
        tree_method="hist",
        eval_metric="mae",
        verbosity=0,
        seed=42,
    )
    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=30,
        verbose=False,
    )
    test_pred += model.predict(test_X)

test_pred /= 5  # average over folds



## === cell 5
submission = pd.read_csv(SAMPLE_SUB_PATH)
submission = submission.sort_values("segment_id").reset_index(drop=True)
submission["time_to_eruption"] = test_pred
submission.to_csv("xgb_5fold_features.csv", index=False)
