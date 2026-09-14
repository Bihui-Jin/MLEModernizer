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
import os
import multiprocessing as mp  # added for parallel file processing

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")




## === cell 1
segment_csvs = glob.glob("../input/predict-volcanic-eruptions-ingv-oe/train/*.csv")
test_csvs = glob.glob("../input/predict-volcanic-eruptions-ingv-oe/test/*.csv")
print(f"train files: {len(segment_csvs)}, test files: {len(test_csvs)}")




## === cell 2
sample_submission = pd.read_csv(
    "../input/predict-volcanic-eruptions-ingv-oe/sample_submission.csv"
)
print(sample_submission.head())




## === cell 3
def sensor_show(df):
    f, axes = plt.subplots(10, 1, figsize=(16, 8))
    f.tight_layout()
    for i in range(1, 11):
        axes[i - 1].plot(df[f"sensor_{i}"].values)
        axes[i - 1].set_title(f"Sensor_{i}")
        axes[i - 1].set_xlabel("time")




## === cell 4
def _process_train_file(csv_path):
    seg_id = int(os.path.basename(csv_path).split(".")[0])
    df = pd.read_csv(csv_path, dtype=np.float32)
    row = df.mean().to_dict()
    row["segment_id"] = seg_id
    return row


workers = max(1, mp.cpu_count() - 1)  # leave one core free for other tasks
with mp.Pool(processes=workers) as pool:
    train_rows = list(
        tqdm(
            pool.imap_unordered(_process_train_file, segment_csvs),
            total=len(segment_csvs),
        )
    )
train_means = pd.DataFrame(train_rows)
print(train_means.head())




## === cell 5
df_train_meta = pd.read_csv("../input/predict-volcanic-eruptions-ingv-oe/train.csv")
df_train_meta["segment_id"] = df_train_meta["segment_id"].astype(int)

train_features = train_means.merge(df_train_meta, on="segment_id")
print(train_features.head())




## === cell 6
y_train = train_features["time_to_eruption"]
X_train = train_features.drop(["segment_id", "time_to_eruption"], axis=1)
X_train = X_train.fillna(X_train.mean())

scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)




## === cell 7
def _process_test_file(csv_path):
    seg_id = int(os.path.basename(csv_path).split(".")[0])
    df = pd.read_csv(csv_path, dtype=np.float32)
    row = df.mean().to_dict()
    row["segment_id"] = seg_id
    return row


with mp.Pool(processes=workers) as pool:
    test_rows = list(
        tqdm(pool.imap_unordered(_process_test_file, test_csvs), total=len(test_csvs))
    )
test_means = pd.DataFrame(test_rows)
print(test_means.head())




## === cell 8
X_test = test_means.drop(["segment_id"], axis=1)
X_test = X_test.fillna(X_test.mean())
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)




## === cell 9
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




## === cell 10
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

oof_lgb, pred_lgb = train_model(
    X=X_train_scaled,
    X_test=X_test_scaled,
    y=y_train,
    folds=folds,
    params=lgb_params,
    model_type="lgb",
    plot_feature_importance=True,
)




## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3622625304.py in <cell line: 0>()
     18 }
     19 
---> 20 oof_lgb, pred_lgb = train_model(
     21     X=X_train_scaled,
     22     X_test=X_test_scaled,

/tmp/ipykernel_11/2852432470.py in train_model(X, X_test, y, folds, params, model_type, plot_feature_importance)
     20                 **params, n_estimators=20000, n_jobs=-1, random_state=42
     21             )
---> 22             model.fit(
     23                 X_tr,
     24                 y_tr,

TypeError: LGBMRegressor.fit() got an unexpected keyword argument 'verbose'

## === cell 11
submission = pd.DataFrame(
    {"segment_id": test_means["segment_id"], "time_to_eruption": pred_lgb}
)
print(submission.head())




## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2913554979.py in <cell line: 0>()
      1 submission = pd.DataFrame(
----> 2     {"segment_id": test_means["segment_id"], "time_to_eruption": pred_lgb}
      3 )
      4 print(submission.head())
      5 

NameError: name 'pred_lgb' is not defined

## === cell 12
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")

## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3907420166.py in <cell line: 0>()
      1 submission_path = "submission.csv"
----> 2 submission.to_csv(submission_path, index=False)
      3 print(f"Submission saved to {submission_path}")

NameError: name 'submission' is not defined
