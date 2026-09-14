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

catboost==1.2.8
geopandas==0.14.4
lightgbm==4.6.0
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
tqdm==4.67.1
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

11757378.0

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import glob
from tqdm import tqdm
import time

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")



## === cell 1
segment_csvs = glob.glob("../input/predict-volcanic-eruptions-ingv-oe/train/*")
len(segment_csvs)



## === cell 2
test_csvs = glob.glob("../input/predict-volcanic-eruptions-ingv-oe/test/*")
len(test_csvs)



## === cell 3
sample_submission = pd.read_csv(
    "../input/predict-volcanic-eruptions-ingv-oe/sample_submission.csv"
)
sample_submission.head()




## === cell 4
def sensor_show(df):
    f, axes = plt.subplots(10, 1, figsize=(16, 8))
    f.tight_layout()
    for i in range(1, 11):
        axes[i - 1].plot(df[f"sensor_{i}"].values)
        axes[i - 1].set_title(f"Sensor_{i}")
        axes[i - 1].set_xlabel("time")




## === cell 5
first_train = pd.read_csv(segment_csvs[0])
train_means = pd.DataFrame(first_train.mean()).T
train_means["segment_id"] = segment_csvs[0].split("/")[-1].split(".")[0]

for csv_path in tqdm(segment_csvs[1:]):
    seg_id = csv_path.split("/")[-1].split(".")[0]
    df = pd.read_csv(csv_path)
    seg_mean = pd.DataFrame(df.mean()).T
    seg_mean["segment_id"] = seg_id
    train_means = pd.concat([train_means, seg_mean], ignore_index=True)

train_means.head()



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
IsADirectoryError                         Traceback (most recent call last)
/tmp/ipykernel_11/3095659919.py in <cell line: 0>()
      9 for csv_path in tqdm(segment_csvs[1:]):
     10     seg_id = csv_path.split("/")[-1].split(".")[0]
---> 11     df = pd.read_csv(csv_path)
     12     seg_mean = pd.DataFrame(df.mean()).T
     13     seg_mean["segment_id"] = seg_id

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in read_csv(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, date_format, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, encoding_errors, dialect, on_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options, dtype_backend)
   1024     kwds.update(kwds_defaults)
   1025 
-> 1026     return _read(filepath_or_buffer, kwds)
   1027 
   1028 

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _read(filepath_or_buffer, kwds)
    618 
    619     # Create the parser.
--> 620     parser = TextFileReader(filepath_or_buffer, **kwds)
    621 
    622     if chunksize or iterator:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in __init__(self, f, engine, **kwds)
   1618 
   1619         self.handles: IOHandles | None = None
-> 1620         self._engine = self._make_engine(f, self.engine)
   1621 
   1622     def close(self) -> None:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _make_engine(self, f, engine)
   1878                 if "b" not in mode:
   1879                     mode += "b"
-> 1880             self.handles = get_handle(
   1881                 f,
   1882                 mode,

/usr/local/lib/python3.11/dist-packages/pandas/io/common.py in get_handle(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)
    871         if ioargs.encoding and "b" not in ioargs.mode:
    872             # Encoding
--> 873             handle = open(
    874                 handle,
    875                 ioargs.mode,

IsADirectoryError: [Errno 21] Is a directory: '../input/predict-volcanic-eruptions-ingv-oe/train/train'

## === cell 6
df_train_meta = pd.read_csv("../input/predict-volcanic-eruptions-ingv-oe/train.csv")
train_features = train_means.merge(df_train_meta, on="segment_id")
train_features.head()



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1625182402.py in <cell line: 0>()
      1 # Load target values and join with the mean features
      2 df_train_meta = pd.read_csv("../input/predict-volcanic-eruptions-ingv-oe/train.csv")
----> 3 train_features = train_means.merge(df_train_meta, on="segment_id")
      4 train_features.head()
      5 

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

ValueError: You are trying to merge on object and int64 columns for key 'segment_id'. If you wish to proceed you should use pd.concat

## === cell 7
y_train = train_features["time_to_eruption"]
X_train = train_features.drop(["segment_id", "time_to_eruption"], axis=1)
X_train = X_train.fillna(X_train.mean())

scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1877224451.py in <cell line: 0>()
      1 # Prepare X and y
----> 2 y_train = train_features["time_to_eruption"]
      3 X_train = train_features.drop(["segment_id", "time_to_eruption"], axis=1)
      4 X_train = X_train.fillna(X_train.mean())
      5 

NameError: name 'train_features' is not defined

## === cell 8
first_test = pd.read_csv(test_csvs[0])
test_means = pd.DataFrame(first_test.mean()).T
test_means["segment_id"] = test_csvs[0].split("/")[-1].split(".")[0]

for csv_path in tqdm(test_csvs[1:]):
    seg_id = csv_path.split("/")[-1].split(".")[0]
    df = pd.read_csv(csv_path)
    seg_mean = pd.DataFrame(df.mean()).T
    seg_mean["segment_id"] = seg_id
    test_means = pd.concat([test_means, seg_mean], ignore_index=True)

test_means.head()



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
IsADirectoryError                         Traceback (most recent call last)
/tmp/ipykernel_11/1110790128.py in <cell line: 0>()
      8 for csv_path in tqdm(test_csvs[1:]):
      9     seg_id = csv_path.split("/")[-1].split(".")[0]
---> 10     df = pd.read_csv(csv_path)
     11     seg_mean = pd.DataFrame(df.mean()).T
     12     seg_mean["segment_id"] = seg_id

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in read_csv(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, date_format, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, encoding_errors, dialect, on_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options, dtype_backend)
   1024     kwds.update(kwds_defaults)
   1025 
-> 1026     return _read(filepath_or_buffer, kwds)
   1027 
   1028 

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _read(filepath_or_buffer, kwds)
    618 
    619     # Create the parser.
--> 620     parser = TextFileReader(filepath_or_buffer, **kwds)
    621 
    622     if chunksize or iterator:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in __init__(self, f, engine, **kwds)
   1618 
   1619         self.handles: IOHandles | None = None
-> 1620         self._engine = self._make_engine(f, self.engine)
   1621 
   1622     def close(self) -> None:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _make_engine(self, f, engine)
   1878                 if "b" not in mode:
   1879                     mode += "b"
-> 1880             self.handles = get_handle(
   1881                 f,
   1882                 mode,

/usr/local/lib/python3.11/dist-packages/pandas/io/common.py in get_handle(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)
    871         if ioargs.encoding and "b" not in ioargs.mode:
    872             # Encoding
--> 873             handle = open(
    874                 handle,
    875                 ioargs.mode,

IsADirectoryError: [Errno 21] Is a directory: '../input/predict-volcanic-eruptions-ingv-oe/test/test'

## === cell 9
X_test = test_means.drop(["segment_id"], axis=1)
X_test = X_test.fillna(X_test.mean())
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)




## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2828713592.py in <cell line: 0>()
      2 X_test = test_means.drop(["segment_id"], axis=1)
      3 X_test = X_test.fillna(X_test.mean())
----> 4 X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
      5 
      6 

NameError: name 'scaler' is not defined

## === cell 10
def train_model(
    X, X_test, y, folds, params, model_type="lgb", plot_feature_importance=False
):
    """
    Train a model with K-fold CV and return OOF predictions and test predictions.
    Currently only LightGBM ('lgb') is used.
    """
    oof = np.zeros(len(X))
    prediction = np.zeros(len(X_test))
    scores = []
    feature_importance = pd.DataFrame()

    for fold_n, (train_idx, valid_idx) in enumerate(folds.split(X)):
        print(f"Fold {fold_n} started at {time.ctime()}")
        X_tr, X_val = X.iloc[train_idx], X.iloc[valid_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[valid_idx]

        if model_type == "lgb":
            model = lgb.LGBMRegressor(
                **params, n_estimators=20000, n_jobs=-1, random_state=42
            )
            model.fit(
                X_tr,
                y_tr,
                eval_set=[(X_tr, y_tr), (X_val, y_val)],
                eval_metric="mae",
                verbose=10000,
                early_stopping_rounds=200,
            )

            oof_val = model.predict(X_val)
            test_pred = model.predict(X_test, num_iteration=model.best_iteration_)

            fold_imp = pd.DataFrame(
                {
                    "feature": X.columns,
                    "importance": model.feature_importances_,
                    "fold": fold_n + 1,
                }
            )
            feature_importance = pd.concat([feature_importance, fold_imp], axis=0)

        else:
            raise NotImplementedError("Only LightGBM is implemented in this script.")

        oof[valid_idx] = oof_val
        prediction += test_pred
        scores.append(mean_absolute_error(y_val, oof_val))

    prediction /= folds.get_n_splits()
    print(f"CV mean MAE: {np.mean(scores):.4f} ± {np.std(scores):.4f}")

    if plot_feature_importance:
        avg_imp = (
            feature_importance.groupby("feature")["importance"]
            .mean()
            .sort_values(ascending=False)
        )
        top_features = avg_imp.head(30).index
        plt.figure(figsize=(12, 8))
        sns.barplot(x=avg_imp.loc[top_features], y=top_features)
        plt.title("Top 30 Feature Importances (average over folds)")
        plt.show()
        return oof, prediction, feature_importance

    return oof, prediction




## === cell 11
n_folds = 5
folds = KFold(n_splits=n_folds, shuffle=True, random_state=11)

lgb_params = {
    "num_leaves": 54,
    "min_data_in_leaf": 79,
    "objective": "huber",
    "max_depth": -1,
    "learning_rate": 0.01,
    "boosting": "gbdt",
    "bagging_freq": 3,
    "bagging_fraction": 0.8126672064208567,
    "bagging_seed": 11,
    "metric": "mae",
    "verbosity": -1,
    "reg_alpha": 1.1302650970728192,
    "reg_lambda": 0.3603427518866501,
}

oof_lgb, pred_lgb, feature_imp = train_model(
    X=X_train_scaled,
    X_test=X_test_scaled,
    y=y_train,
    folds=folds,
    params=lgb_params,
    model_type="lgb",
    plot_feature_importance=True,
)



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1640818613.py in <cell line: 0>()
     22 # Train LightGBM model
     23 oof_lgb, pred_lgb, feature_imp = train_model(
---> 24     X=X_train_scaled,
     25     X_test=X_test_scaled,
     26     y=y_train,

NameError: name 'X_train_scaled' is not defined

## === cell 12
submission = pd.DataFrame(
    {"segment_id": test_means["segment_id"], "time_to_eruption": pred_lgb}
)
submission.head()



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1859289962.py in <cell line: 0>()
      1 # Create submission file – ensure column names match the competition requirement
      2 submission = pd.DataFrame(
----> 3     {"segment_id": test_means["segment_id"], "time_to_eruption": pred_lgb}
      4 )
      5 submission.head()

NameError: name 'pred_lgb' is not defined

## === cell 13
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")

## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3218387231.py in <cell line: 0>()
      1 # Write to CSV
      2 submission_path = "submission.csv"
----> 3 submission.to_csv(submission_path, index=False)
      4 print(f"Submission saved to {submission_path}")

NameError: name 'submission' is not defined
