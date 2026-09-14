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
from sklearn.model_selection import train_test_split
from tsfresh import extract_features
from tsfresh.feature_extraction import EfficientFCParameters
from xgboost import XGBRegressor

data_folder = Path("/kaggle/input/predict-volcanic-eruptions-ingv-oe")



## === cell 1
df_meta = pd.read_csv(data_folder / "train.csv")
df_meta.head()




## === cell 2
def preprocess_timeseries(meta_df, parameters, is_train=True):
    """
    Extract tsfresh features for every segment.
    For speed we keep only the first 1000 rows of each segment and
    run a single parallel extraction instead of one call per segment.
    """
    segment_ids = []
    times = []  # only for training
    dfs = []  # list of per‑row dataframes to concatenate

    if is_train:
        iterator = list(meta_df.itertuples(index=False, name="TrainRow"))
    else:
        test_files = [f for f in os.listdir(data_folder / "test") if f.endswith(".csv")]
        iterator = test_files

    for idx, row in enumerate(iterator):
        if is_train:
            segment_id = row.segment_id
            time_to_eruption = row.time_to_eruption
            segment_path = data_folder / "train" / f"{segment_id}.csv"
        else:
            filename = row  # e.g. "12345.csv"
            segment_path = data_folder / "test" / filename
            segment_id = Path(filename).stem
            time_to_eruption = np.nan  # placeholder

        ts = pd.read_csv(
            segment_path,
            dtype=np.float32,
            nrows=1000,  # pandas will stop after 1000 rows
        ).fillna(0)

        ts = ts.reset_index()  # creates column "index" for ordering
        ts["id"] = idx  # tsfresh grouping id

        dfs.append(ts)
        segment_ids.append(segment_id)
        if is_train:
            times.append(time_to_eruption)

        if (idx + 1) % 200 == 0:
            print(f"Loaded {idx + 1} segments …")

    all_ts = pd.concat(dfs, ignore_index=True)

    feats = extract_features(
        all_ts,
        column_id="id",
        column_sort="index",
        default_fc_parameters=parameters,
        disable_progressbar=True,
        n_jobs=1,  # enforce a single process to avoid pool errors
    )
    feats.reset_index(inplace=True)
    feats.rename(columns={"id": "segment_index"}, inplace=True)

    feats["segment"] = segment_ids

    if is_train:
        feats["time_to_eruption"] = times

    feats.drop(columns=["segment_index"], inplace=True)

    return feats




## === cell 3
tsfresh_parameters = EfficientFCParameters()




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
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=1337
)

X_train = X_train.fillna(0)
X_valid = X_valid.fillna(0)

pipe = XGBRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    objective="reg:squarederror",
    n_jobs=5,
    random_state=42,
    reg_lambda=1.0,
)

pipe.fit(X_train, y_train)



## === cell 7
df_test = pd.read_csv("test.csv")
df_test = df_test.dropna(axis="columns", how="all")
X_test = df_test[feature_cols].fillna(0)



## === cell 8
test_pred = pipe.predict(X_test)

submission = pd.DataFrame(
    {
        "segment_id": df_test["segment"],
        "time_to_eruption": test_pred,
    }
)

submission.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")
