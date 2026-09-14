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
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1
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

6646327.0

# 6. Current score

5138894.0

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 4107646.0) has done: 'The fix removes the failing tensorflow‑addons imports, corrects the Keras Input shape usage, adds a proper XGBoost import, and streamlines the pipeline to compute statistical features, train a simple XGBRegressor, and write a valid `submission.csv` matching the required column names. These changes resolve the runtime errors and ensure a submission file is produced; the resulting MAE be far below the huge target, moving the score into the acceptable lower‑is‑better range.'
- What this solution (achieved 5138894.0) has done: 'I keep the original pipeline intact but slightly downgrade the XGBoost model so the MAE moves upward (worse) toward the target value. This is done by reducing the number of trees, shallower depth, a higher learning rate, a much smaller early‑stopping window, and fixing a random seed for reproducibility. No other logic is altered, and the script still writes a correctly formatted `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb



## === cell 1
train_meta = pd.read_csv("/kaggle/input/predict-volcanic-eruptions-ingv-oe/train.csv")
n_features = 12 * 10  # 12 statistics per sensor, 10 sensors

train_X = np.empty((train_meta.shape[0], n_features), dtype=np.float32)
train_y = np.empty((train_meta.shape[0],), dtype=np.float32)

for idx, seg_id in enumerate(train_meta["segment_id"]):
    seg_path = f"/kaggle/input/predict-volcanic-eruptions-ingv-oe/train/{seg_id}.csv"
    df = pd.read_csv(seg_path).fillna(0)
    feats = np.concatenate(
        (
            df.abs().mean().to_numpy(),
            df.std().to_numpy(),
            df.mean().to_numpy(),
            df.var().to_numpy(),
            df.min().to_numpy(),
            df.max().to_numpy(),
            df.median().to_numpy(),
            df.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).to_numpy().reshape(-1),
        )
    )
    train_X[idx, :] = feats
    train_y[idx] = train_meta.loc[idx, "time_to_eruption"]



## === cell 2
sample_sub = pd.read_csv(
    "/kaggle/input/predict-volcanic-eruptions-ingv-oe/sample_submission.csv"
)
test_X = np.empty((sample_sub.shape[0], n_features), dtype=np.float32)

for idx, seg_id in enumerate(sample_sub["segment_id"]):
    seg_path = f"/kaggle/input/predict-volcanic-eruptions-ingv-oe/test/{seg_id}.csv"
    df = pd.read_csv(seg_path).fillna(0)
    feats = np.concatenate(
        (
            df.abs().mean().to_numpy(),
            df.std().to_numpy(),
            df.mean().to_numpy(),
            df.var().to_numpy(),
            df.min().to_numpy(),
            df.max().to_numpy(),
            df.median().to_numpy(),
            df.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).to_numpy().reshape(-1),
        )
    )
    test_X[idx, :] = feats



## === cell 3
X_tr, X_val, y_tr, y_val = train_test_split(
    train_X, train_y, test_size=0.1, random_state=3
)

model = xgb.XGBRegressor(
    n_estimators=500,  # fewer trees
    max_depth=4,  # shallower trees
    learning_rate=0.1,  # higher learning rate
    subsample=0.6,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    eval_metric="mae",
    tree_method="hist",
    verbosity=0,
    random_state=42,  # reproducibility
)

model.fit(
    X_tr,
    y_tr,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=5,  # stop early to limit over‑fitting
    verbose=False,
)



## === cell 4
preds = model.predict(test_X)
submission = sample_sub.copy()
submission["time_to_eruption"] = preds
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
