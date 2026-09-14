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

5009493.05033489

# 6. Current score

24402200.0

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

- What this solution (achieved 24402200.0) has done: 'Implemented a robust end‑to‑end pipeline that replaces missing pickle feature files with on‑the‑fly feature extraction, fixes the fold creation mismatch, adjusts XGBoost to a universally‑available histogram tree method, and ensures a correctly‑named CSV submission is written.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import pathlib
from sklearn.model_selection import KFold
import xgboost as xgb



## === cell 1
meta_path = pathlib.Path("../input/predict-volcanic-eruptions-ingv-oe/train.csv")
meta_df = pd.read_csv(meta_path)

meta_df = meta_df.sort_values("segment_id").reset_index(drop=True)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
folds = np.empty(len(meta_df), dtype=int)
for fold_idx, (_, val_idx) in enumerate(kf.split(meta_df), start=1):
    folds[val_idx] = fold_idx
meta_df["fold"] = folds




## === cell 2
def extract_features(segment_dir: pathlib.Path) -> pd.DataFrame:
    """
    For each segment CSV in `segment_dir`, compute mean and std for every sensor.
    Returns a DataFrame indexed by segment_id with 20 feature columns.
    """
    feature_rows = []
    for csv_path in sorted(segment_dir.glob("*.csv")):
        seg_id = csv_path.stem  # filename without .csv
        df = pd.read_csv(csv_path, dtype=np.float32)
        means = df.mean()
        stds = df.std()
        row = {"segment_id": seg_id}
        for col in means.index:
            row[f"{col}_mean"] = means[col]
            row[f"{col}_std"] = stds[col]
        feature_rows.append(row)
    return pd.DataFrame(feature_rows).set_index("segment_id")


train_seg_dir = pathlib.Path("../input/predict-volcanic-eruptions-ingv-oe/train")
test_seg_dir = pathlib.Path("../input/predict-volcanic-eruptions-ingv-oe/test")

train_feat = extract_features(train_seg_dir)
test_feat = extract_features(test_seg_dir)



## === cell 3
train_feat = train_feat.loc[meta_df["segment_id"]].reset_index(drop=True)
test_feat = test_feat.reset_index(drop=True)

y = meta_df["time_to_eruption"].values



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/3444802875.py in <cell line: 0>()
      1 # Align feature tables with metadata
----> 2 train_feat = train_feat.loc[meta_df["segment_id"]].reset_index(drop=True)
      3 test_feat = test_feat.reset_index(drop=True)
      4 
      5 # Target variable

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in __getitem__(self, key)
   1189             maybe_callable = com.apply_if_callable(key, self.obj)
   1190             maybe_callable = self._check_deprecated_callable_usage(key, maybe_callable)
-> 1191             return self._getitem_axis(maybe_callable, axis=axis)
   1192 
   1193     def _is_scalar_access(self, key: tuple):

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in _getitem_axis(self, key, axis)
   1418                     raise ValueError("Cannot index with multidimensional key")
   1419 
-> 1420                 return self._getitem_iterable(key, axis=axis)
   1421 
   1422             # nested tuple slicing

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in _getitem_iterable(self, key, axis)
   1358 
   1359         # A collection of keys
-> 1360         keyarr, indexer = self._get_listlike_indexer(key, axis)
   1361         return self.obj._reindex_with_indexers(
   1362             {axis: [keyarr, indexer]}, copy=True, allow_dups=True

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in _get_listlike_indexer(self, key, axis)
   1556         axis_name = self.obj._get_axis_name(axis)
   1557 
-> 1558         keyarr, indexer = ax._get_indexer_strict(key, axis_name)
   1559 
   1560         return keyarr, indexer

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _get_indexer_strict(self, key, axis_name)
   6198             keyarr, indexer, new_indexer = self._reindex_non_unique(keyarr)
   6199 
-> 6200         self._raise_if_missing(keyarr, indexer, axis_name)
   6201 
   6202         keyarr = self.take(indexer)

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _raise_if_missing(self, key, indexer, axis_name)
   6247         if nmissing:
   6248             if nmissing == len(indexer):
-> 6249                 raise KeyError(f"None of [{key}] are in the [{axis_name}]")
   6250 
   6251             not_found = list(ensure_index(key)[missing_mask.nonzero()[0]].unique())

KeyError: "None of [Index([    513181,     603314,    1257785,    1846591,    2666652,    5149608,\n          6070985,    6452624,    6577398,    6825894,\n       ...\n       2143457307, 2144701601, 2144756834, 2145274131, 2145499088, 2145859440,\n       2146132452, 2146270165, 2146361020, 2146908755],\n      dtype='int64', name='segment_id', length=3987)] are in the [index]"

## === cell 4
preds = np.zeros(len(test_feat))
for fold in range(1, 6):
    train_idx = meta_df[meta_df["fold"] != fold].index
    val_idx = meta_df[meta_df["fold"] == fold].index

    X_train = train_feat.iloc[train_idx]
    y_train = y[train_idx]
    X_val = train_feat.iloc[val_idx]
    y_val = y[val_idx]

    model = xgb.XGBRegressor(
        n_estimators=100000,
        tree_method="hist",  # histogram method works on CPU & GPU
        max_depth=8,
        learning_rate=0.05,
        alpha=0.1,
        subsample=0.6,
        colsample_bytree=0.5,
        objective="reg:squarederror",
        eval_metric="mae",
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=5,
        verbose=False,
    )
    print(f"Fold {fold} MAE:", model.evals_result()["validation_0"]["mae"][-1])
    preds += model.predict(test_feat)

preds /= 5.0



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/506433682.py in <cell line: 0>()
      5 
      6     X_train = train_feat.iloc[train_idx]
----> 7     y_train = y[train_idx]
      8     X_val = train_feat.iloc[val_idx]
      9     y_val = y[val_idx]

NameError: name 'y' is not defined

## === cell 5
sample_sub_path = pathlib.Path(
    "../input/predict-volcanic-eruptions-ingv-oe/sample_submission.csv"
)
submission = pd.read_csv(sample_sub_path)
submission["time_to_eruption"] = preds
submission.to_csv("submission.csv", index=False)
print("Submission written to submission.csv")
