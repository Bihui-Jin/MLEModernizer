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
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
rgf-python==3.12.0
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

# 5. Code solution

## === cell 0
import os
import glob
import warnings

import numpy as np
import pandas as pd

pd.set_option("display.max_columns", None)
warnings.filterwarnings("ignore")

RANDOM_STATE = 0

BASE_PATH = "../input/predict-volcanic-eruptions-ingv-oe"
TRAIN_META_PATH = os.path.join(BASE_PATH, "train.csv")
SAMPLE_SUB_PATH = os.path.join(BASE_PATH, "sample_submission.csv")
TRAIN_DIR = os.path.join(BASE_PATH, "train")
TEST_DIR = os.path.join(BASE_PATH, "test")

assert os.path.exists(TRAIN_META_PATH), f"Missing: {TRAIN_META_PATH}"
assert os.path.exists(SAMPLE_SUB_PATH), f"Missing: {SAMPLE_SUB_PATH}"
assert os.path.isdir(TRAIN_DIR), f"Missing dir: {TRAIN_DIR}"
assert os.path.isdir(TEST_DIR), f"Missing dir: {TEST_DIR}"




## === cell 1
def build_features_for_segment(csv_path: str) -> pd.Series:
    """
    Minimal, deterministic feature extraction:
    aggregate stats per sensor column to produce fixed-length tabular features.
    This replaces the missing external 'volcano_train.csv/volcano_test.csv' files.
    """
    df = pd.read_csv(csv_path, dtype=np.float32)
    df = df.replace([np.inf, -np.inf], np.nan)

    feats = {}
    for col in df.columns:
        x = df[col].astype(np.float32)
        feats[f"{col}_mean"] = float(np.nanmean(x))
        feats[f"{col}_std"] = float(np.nanstd(x))
        feats[f"{col}_min"] = float(np.nanmin(x))
        feats[f"{col}_max"] = float(np.nanmax(x))
        feats[f"{col}_median"] = float(np.nanmedian(x))
        q25 = float(np.nanpercentile(x, 25))
        q75 = float(np.nanpercentile(x, 75))
        feats[f"{col}_iqr"] = q75 - q25

    return pd.Series(feats, dtype=np.float32)


def build_feature_matrix(segment_ids, data_dir: str) -> pd.DataFrame:
    rows = []
    idx = []
    for sid in segment_ids:
        csv_path = os.path.join(data_dir, f"{sid}.csv")
        if not os.path.exists(csv_path):
            csv_path = os.path.join(data_dir, f"{int(sid)}.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"Segment file not found for segment_id={sid} in {data_dir}"
            )
        rows.append(build_features_for_segment(csv_path))
        idx.append(int(sid))
    X = pd.DataFrame(rows, index=idx)
    X.index.name = "segment_id"
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))
    return X


train_meta = pd.read_csv(TRAIN_META_PATH)
train_meta["segment_id"] = train_meta["segment_id"].astype(int)

y_train = train_meta.set_index("segment_id")["time_to_eruption"].astype(float)

X_train_full = build_feature_matrix(train_meta["segment_id"].values, TRAIN_DIR)

X_train_full = X_train_full.loc[y_train.index]

print("Train feature matrix:", X_train_full.shape, "Target:", y_train.shape)



## === cell 2
selected_columns = list(X_train_full.columns)

try:
    from xgboost import XGBRegressor
    from BorutaShap import BorutaShap

    base_model = XGBRegressor(
        random_state=RANDOM_STATE,
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="reg:squarederror",
        n_jobs=4,
    )

    Feature_Selector = BorutaShap(
        model=base_model,
        importance_measure="shap",
        classification=False,
    )
    Feature_Selector.fit(
        X=X_train_full,
        y=y_train.values,
        n_trials=35,
        random_state=RANDOM_STATE,
    )

    try:
        subset_df = Feature_Selector.Subset()
        if isinstance(subset_df, pd.DataFrame) and subset_df.shape[1] > 0:
            selected_columns = list(subset_df.columns)
    except Exception:
        pass

except Exception as e:
    print("BorutaShap step skipped/fell back to all features due to:", repr(e))

print("Selected features:", len(selected_columns))



## === cell 3
try:
    if "Feature_Selector" in globals():
        Feature_Selector.plot(which_features="accepted", figsize=(20, 12))
except Exception as e:
    print("Plot skipped:", repr(e))



## === cell 4
X_train = X_train_full[selected_columns].copy()

pd.Series(selected_columns, name="feature").to_csv("selected_features.csv", index=False)

print("Final training matrix:", X_train.shape)



## === cell 5
sample = pd.read_csv(SAMPLE_SUB_PATH)
sample["segment_id"] = sample["segment_id"].astype(int)

X_test_full = build_feature_matrix(sample["segment_id"].values, TEST_DIR)
X_test = X_test_full[selected_columns].copy()

print("Test feature matrix:", X_test.shape)



## === cell 6
from rgf.sklearn import RGFRegressor

regressor = RGFRegressor(
    max_leaf=10000,
    algorithm="RGF_Sib",
    test_interval=100,
    loss="LS",
    verbose=False,
)

regressor.fit(X_train, y_train.values)
predictions = regressor.predict(X_test)

predictions = np.asarray(predictions, dtype=float)



## === cell 7
submission = sample.copy()
submission["time_to_eruption"] = predictions

submission = submission[["segment_id", "time_to_eruption"]]

submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())

## --- ERROR in outputing the csv:
Invalid submission: Target column time_to_eruption not found in submissions DataFrame.
