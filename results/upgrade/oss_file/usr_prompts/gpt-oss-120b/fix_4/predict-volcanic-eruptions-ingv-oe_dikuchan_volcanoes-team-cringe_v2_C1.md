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

7193000.0

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 8607143.0) has done: 'I filter the test directory to read only CSV files (skipping the stray sub‑directory), ensure the test feature matrix has exactly the same columns as the training data (filling missing ones with 0), and then run the prediction and write a proper `submission.csv`. These small fixes resolve the `IsADirectoryError` and the subsequent `NameError`, allowing the pipeline to finish and produce a valid submission file.'
- What this solution (achieved 7193000.0) has done: 'The changes add richer statistical descriptors (min, max, median, 25th and 75th percentiles) to each segment’s feature set, giving the model more information while preserving the original pipeline. The XGBoost regressor is also given a modest boost in capacity (more trees and deeper depth) to better capture the added features. These adjustments are targeted to lower the MAE and move the score closer to the target without altering the overall workflow.'

# 9. Code solution

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




## === cell 1
data_folder = Path("../input/predict-volcanic-eruptions-ingv-oe/")




## === cell 2
df_meta = pd.read_csv(data_folder / "train.csv")
df_meta.head().T




## === cell 3
def extract_simple_features(segment_path):
    """Compute a richer set of column‑wise statistics for a segment CSV."""
    df = pd.read_csv(segment_path)
    means = df.mean().add_prefix("mean_")
    stds = df.std().add_prefix("std_")
    mins = df.min().add_prefix("min_")
    maxs = df.max().add_prefix("max_")
    medians = df.median().add_prefix("median_")
    q25 = df.quantile(0.25).add_prefix("q25_")
    q75 = df.quantile(0.75).add_prefix("q75_")
    return pd.concat([means, stds, mins, maxs, medians, q25, q75])




## === cell 4
train_features = []
train_targets = []
for _, row in df_meta.iterrows():
    seg_id = row["segment_id"]
    target = row["time_to_eruption"]
    seg_path = data_folder / "train" / f"{seg_id}.csv"
    feats = extract_simple_features(seg_path)
    feats["segment_id"] = seg_id
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
pipe = Pipeline(
    [
        ("scaler", MinMaxScaler()),
        (
            "xgboost",
            XGBRegressor(
                objective="reg:squarederror",
                n_estimators=400,  # modest increase for better fit
                learning_rate=0.05,
                max_depth=8,  # deeper trees to capture extra features
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=4,
            ),
        ),
    ]
)




## === cell 8
pipe.fit(X_train, y_train)




## === cell 9
from sklearn.metrics import mean_absolute_error

val_pred = pipe.predict(X_val)
val_mae = mean_absolute_error(y_val, val_pred)
print(f"Validation MAE: {val_mae:.4f}")




## === cell 10
test_dir = data_folder / "test"
test_files = [f for f in os.listdir(test_dir) if f.lower().endswith(".csv")]
test_features = []
for seg in sorted(test_files):
    seg_path = test_dir / seg
    feats = extract_simple_features(seg_path)
    feats["segment_id"] = seg.split(".")[0]  # strip .csv extension
    test_features.append(feats)

df_test = pd.DataFrame(test_features)

df_test = df_test.reindex(columns=features + ["segment_id"], fill_value=0)




## === cell 11
test_pred = pipe.predict(df_test[features])




## === cell 12
submission = pd.DataFrame(
    {"segment_id": df_test["segment_id"], "time_to_eruption": test_pred}
)
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
