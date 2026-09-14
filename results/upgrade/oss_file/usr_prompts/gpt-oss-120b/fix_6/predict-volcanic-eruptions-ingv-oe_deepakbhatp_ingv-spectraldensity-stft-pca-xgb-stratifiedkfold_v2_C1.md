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

5381042.07761754

# 6. Current score

14581617.0

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 14419211.0) has done: 'The changes load the raw segment CSVs, compute simple statistical features (mean and standard deviation for each sensor) for both train and test sets, merge them with the provided metadata, and train an XGBoost regressor using a validation split. The script then predicts on the test data and writes a correctly‑formatted submission file, fixing all missing‑file errors and the mismatched fold assignment.'
- What this solution (achieved 14543839.0) has done: 'The fix expands the feature extraction to include additional descriptive statistics (min, max, median) for each sensor, giving the model richer information while keeping the overall pipeline and model unchanged. This modest change is expected to lower the validation MAE, moving the score closer to the target. The rest of the script is left intact, and the submission file is still written correctly.'
- What this solution (achieved 14694165.0) has done: 'I enhance the feature set by adding additional descriptive statistics (25th/75th percentiles, range, and overall segment stats) and slightly increase the model’s depth to capture more complex patterns. These changes keep the original pipeline intact while giving the XGBoost regressor richer information, which should lower the MAE and move the score closer to the target.'
- What this solution (achieved 14581617.0) has done: 'The changes focus on speeding up feature extraction, which dominates runtime. The new implementation reads each CSV directly into a NumPy array (float32) and computes all required statistics with vectorized NumPy calls in a single pass. It processes files in parallel using `ProcessPoolExecutor` (matching the 4‑core setting of XGBoost) while preserving the original segment order, so the resulting feature matrix is identical. No model logic is altered, ensuring the same predictions and evaluation.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import xgboost as xgb




## === cell 1
base_path = Path("../input/predict-volcanic-eruptions-ingv-oe")
train_meta_path = base_path / "train.csv"
test_meta_path = base_path / "test.csv"  # not used, only for consistency
train_folder = base_path / "train"
test_folder = base_path / "test"

train_meta = pd.read_csv(train_meta_path)
train_meta["segment_id"] = train_meta["segment_id"].astype(str)

import concurrent.futures


def _process_segment(args):
    """Helper for parallel execution – returns a dict of features for one segment."""
    seg_id, folder = args
    file_path = folder / f"{seg_id}.csv"
    data = pd.read_csv(file_path, dtype=np.float32).values  # shape (60001, 10)

    means = data.mean(axis=0)
    stds = data.std(axis=0, ddof=0)
    mins = data.min(axis=0)
    maxs = data.max(axis=0)
    medians = np.median(data, axis=0)
    q25 = np.quantile(data, 0.25, axis=0)
    q75 = np.quantile(data, 0.75, axis=0)
    iqr = q75 - q25
    ranges = maxs - mins

    flat = data.ravel()
    overall_mean = flat.mean()
    overall_std = flat.std()
    overall_min = flat.min()
    overall_max = flat.max()
    overall_median = np.median(flat)
    overall_range = overall_max - overall_min

    sensor_cols = [f"sensor_{i+1}" for i in range(data.shape[1])]
    feat = {}
    for idx, col in enumerate(sensor_cols):
        feat[f"{col}_mean"] = means[idx]
        feat[f"{col}_std"] = stds[idx]
        feat[f"{col}_min"] = mins[idx]
        feat[f"{col}_max"] = maxs[idx]
        feat[f"{col}_median"] = medians[idx]
        feat[f"{col}_q25"] = q25[idx]
        feat[f"{col}_q75"] = q75[idx]
        feat[f"{col}_iqr"] = iqr[idx]
        feat[f"{col}_range"] = ranges[idx]

    feat.update(
        {
            "overall_mean": overall_mean,
            "overall_std": overall_std,
            "overall_min": overall_min,
            "overall_max": overall_max,
            "overall_median": overall_median,
            "overall_range": overall_range,
            "segment_id": seg_id,
        }
    )
    return feat


def extract_features(folder: Path, ids):
    """
    Parallel extraction of statistical features for a list of segment IDs.
    Returns a DataFrame with the same schema as the original implementation.
    """
    tasks = [(seg_id, folder) for seg_id in ids]

    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(_process_segment, tasks))

    return pd.DataFrame(results)


train_features = extract_features(train_folder, train_meta["segment_id"])
train_df = train_meta.merge(train_features, on="segment_id")
y = train_df["time_to_eruption"]
X = train_df.drop(columns=["segment_id", "time_to_eruption"])

test_ids = [p.stem for p in test_folder.glob("*.csv")]
test_features = extract_features(test_folder, test_ids)
test_df = test_features.copy()  # same columns as X except segment_id
test_X = test_df.drop(columns=["segment_id"])




## === cell 2
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

model = xgb.XGBRegressor(
    n_estimators=12000,
    max_depth=10,  # deeper trees to exploit richer features
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    objective="reg:squarederror",
    eval_metric="mae",
    n_jobs=4,
    reg_lambda=1.0,
)

eval_set = [(X_val, y_val)]
model.fit(
    X_train,
    y_train,
    eval_set=eval_set,
    early_stopping_rounds=150,
    verbose=False,
)

val_pred = model.predict(X_val)
print("Validation MAE:", mean_absolute_error(y_val, val_pred))

test_predictions = model.predict(test_X)




## === cell 3
sample_submission_path = base_path / "sample_submission.csv"
submission = pd.read_csv(sample_submission_path)
submission["time_to_eruption"] = test_predictions
submission.to_csv("volcano_xgb_submission.csv", index=False)
print("Submission saved to volcano_xgb_submission.csv")
