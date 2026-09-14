# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 14419211.0) has done: 'The changes load the raw segment CSVs, compute simple statistical features (mean and standard deviation for each sensor) for both train and test sets, merge them with the provided metadata, and train an XGBoost regressor using a validation split. The script then predicts on the test data and writes a correctly‑formatted submission file, fixing all missing‑file errors and the mismatched fold assignment.'
- What this solution (achieved 14543839.0) has done: 'The fix expands the feature extraction to include additional descriptive statistics (min, max, median) for each sensor, giving the model richer information while keeping the overall pipeline and model unchanged. This modest change is expected to lower the validation MAE, moving the score closer to the target. The rest of the script is left intact, and the submission file is still written correctly.'
- What this solution (achieved 14694165.0) has done: 'I enhance the feature set by adding additional descriptive statistics (25th/75th percentiles, range, and overall segment stats) and slightly increase the model’s depth to capture more complex patterns. These changes keep the original pipeline intact while giving the XGBoost regressor richer information, which should lower the MAE and move the score closer to the target.'
- What this solution (achieved 14581617.0) has done: 'The changes focus on speeding up feature extraction, which dominates runtime. The new implementation reads each CSV directly into a NumPy array (float32) and computes all required statistics with vectorized NumPy calls in a single pass. It processes files in parallel using `ProcessPoolExecutor` (matching the 4‑core setting of XGBoost) while preserving the original segment order, so the resulting feature matrix is identical. No model logic is altered, ensuring the same predictions and evaluation.'
- What this solution (achieved 14689648.0) has done: 'I adjust the XGBoost model to better match the MAE metric by switching the objective to `reg:absoluteerror`, reducing tree depth to prevent over‑fitting, and slightly increasing the learning rate. I also set a seed for reproducibility and clip negative predictions to zero, which is sensible for time‑to‑eruption values. These minimal changes keep the overall pipeline unchanged while aiming to lower the validation MAE and move the score closer to the target.'
- What this solution (achieved 14527027.0) has done: 'I add a few lightweight descriptive features (mean and std of the first‑order differences for each sensor) that often capture trend information without altering the overall pipeline. Then I slightly adjust the XGBoost hyper‑parameters – increase depth, lower the learning rate and L2 regularisation, and raise the maximum number of trees – so the model can make better use of the richer feature set while still using the same MAE‑aligned objective and early stopping. These changes keep the core logic intact but should lower the validation MAE and move the score closer to the target.'
- What this solution (achieved 14573872.0) has done: 'I apply a log‑transform to the target variable before training the XGBoost model and then invert the transform for predictions. Modeling the log of `time_to_eruption` often stabilises variance and reduces MAE, moving the score toward the lower target while keeping the overall pipeline unchanged. The only code changes are the target transformation, corresponding split handling, and the inverse transform for validation and test predictions.'
- What this solution (achieved 14527027.0) has done: 'I remove the log‑transform of the target and train the XGBoost regressor directly on the original `time_to_eruption` values. This aligns the training objective (MAE) with the competition’s evaluation metric, which should lower the validation MAE and move the score closer to the target while keeping the rest of the pipeline unchanged.'
- What this solution (achieved 14579592.0) has done: 'I keep the overall pipeline unchanged but train the model on a log‑transformed target (log1p) and then invert the transformation for validation and test predictions.  MAE is still computed on the original scale, which often reduces error for the skewed “time‑to‑eruption” distribution while preserving the same feature set and XGBoost settings.  The only code changes are the target transformation, inverse‑transform of predictions, and a slightly deeper tree (max_depth = 10) to give the model a bit more capacity.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
import concurrent.futures



## === cell 1
base_path = Path("../input/predict-volcanic-eruptions-ingv-oe")
train_meta_path = base_path / "train.csv"
test_meta_path = base_path / "test.csv"  # not used, only for consistency
train_folder = base_path / "train"
test_folder = base_path / "test"

train_meta = pd.read_csv(train_meta_path)
train_meta["segment_id"] = train_meta["segment_id"].astype(str)


def _process_segment(args):
    """Extract statistical and simple trend features for one segment."""
    seg_id, folder = args
    file_path = folder / f"{seg_id}.csv"
    data = pd.read_csv(file_path, dtype=np.float32).values  # (60001, 10)

    means = data.mean(axis=0)
    stds = data.std(axis=0, ddof=0)
    mins = data.min(axis=0)
    maxs = data.max(axis=0)
    medians = np.median(data, axis=0)
    q25 = np.quantile(data, 0.25, axis=0)
    q75 = np.quantile(data, 0.75, axis=0)
    iqr = q75 - q25
    ranges = maxs - mins

    diffs = np.diff(data, axis=0)  # shape (60000, 10)
    diff_means = diffs.mean(axis=0)
    diff_stds = diffs.std(axis=0, ddof=0)

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
        feat[f"{col}_diff_mean"] = diff_means[idx]
        feat[f"{col}_diff_std"] = diff_stds[idx]

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
    """Parallel extraction of features for a list of segment IDs."""
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
test_df = test_features.copy()
test_X = test_df.drop(columns=["segment_id"])



## === cell 2
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

model = xgb.XGBRegressor(
    n_estimators=30000,
    max_depth=10,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    objective="reg:absoluteerror",
    eval_metric="mae",
    n_jobs=4,
    reg_lambda=1.0,
    random_state=42,
)

eval_set = [(X_val, y_val)]
model.fit(
    X_train,
    y_train,
    eval_set=eval_set,
    early_stopping_rounds=200,
    verbose=False,
)

val_pred = model.predict(X_val)
print("Validation MAE:", mean_absolute_error(y_val, val_pred))

test_pred = model.predict(test_X)
test_pred = np.clip(test_pred, 0, None)  # ensure non‑negative predictions



## === cell 3
sample_submission_path = base_path / "sample_submission.csv"
submission = pd.read_csv(sample_submission_path)

pred_df = pd.DataFrame(
    {"segment_id": test_df["segment_id"], "time_to_eruption": test_pred}
)

submission = submission.drop(columns=["time_to_eruption"]).merge(
    pred_df, on="segment_id", how="left"
)

submission.to_csv("volcano_xgb_submission.csv", index=False)
print("Submission saved to volcano_xgb_submission.csv")

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1923472352.py in <cell line: 0>()
      8 
      9 # Merge to ensure alignment with the sample submission order
---> 10 submission = submission.drop(columns=["time_to_eruption"]).merge(
     11     pred_df, on="segment_id", how="left"
     12 )

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in merge(self, right, how, on, left_on, right_on, left_index, right_index, sort, suffixes, copy, indicator, validate)
  10830         from pandas.core.reshape.merge import merge
  10831 
> 10832         return merge(
  10833             self,
  10834             right,

/usr/local/lib/python3.11/dist-packages/pandas/core/reshape/merge.py in merge(left, right, how, on, left_on, right_on, left_index, right_index, sort, suffixes, copy, indicator, validate)
    168         )
    169     else:
--> 170         op = _MergeOperation(
    171             left_df,
    172             right_df,

/usr/local/lib/python3.11/dist-packages/pandas/core/reshape/merge.py in __init__(self, left, right, how, on, left_on, right_on, left_index, right_index, sort, suffixes, indicator, validate)
    805         # validate the merge keys dtypes. We may need to coerce
    806         # to avoid incompatible dtypes
--> 807         self._maybe_coerce_merge_keys()
    808 
    809         # If argument passed to validate,

/usr/local/lib/python3.11/dist-packages/pandas/core/reshape/merge.py in _maybe_coerce_merge_keys(self)
   1506                     inferred_right in string_types and inferred_left not in string_types
   1507                 ):
-> 1508                     raise ValueError(msg)
   1509 
   1510             # datetimelikes must match exactly

ValueError: You are trying to merge on int64 and object columns for key 'segment_id'. If you wish to proceed you should use pd.concat
