# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


# 1. Kaggle task description

## Task
Given simulated manufacturing control data, predict whether the machine is in state `0` or state `1`.

## Metric
Area under the ROC curve.

## Submission Format
For each `id` in the test set, you must predict a probability for the `target` variable. The file should contain a header and have the following format:

```
id,target
900000,0.65
900001,0.97
900002,0.02
etc.
```

## Dataset
- **train.csv** - the training data, which includes normalized continuous data and categorical data
- **test.csv** - the test set; your task is to predict binary `target` variable which represents the state of a manufacturing process
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.10

# 3. Installed packages

geopandas==0.14.4
lightgbm==4.6.0
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
optuna==4.5.0
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (79 lines)
            sample_submission.csv (100001 lines)
            sample_submission.csv.zip (224.9 kB)
            test.csv (100001 lines)
            test.csv.zip (16.6 MB)
            train.csv (800001 lines)
            train.csv.zip (133.3 MB)
            tabular-playground-series-may-2022/
                description.md (79 lines)
                sample_submission.csv (100001 lines)
                ... and 5 other files
                tabular-playground-series-may-2022/
        input/
            description.md (79 lines)
            sample_submission.csv (100001 lines)
            sample_submission.csv.zip (224.9 kB)
            test.csv (100001 lines)
            test.csv.zip (16.6 MB)
            train.csv (800001 lines)
            train.csv.zip (133.3 MB)
            tabular-playground-series-may-2022/
                description.md (79 lines)
                sample_submission.csv (100001 lines)
                ... and 5 other files
                tabular-playground-series-may-2022/
        working/
            tabular-playground-series-may-2022/
                description.md (79 lines)
                sample_submission.csv (100001 lines)
                ... and 5 other files
                tabular-playground-series-may-2022/
```

-> data/sample_submission.csv has 100000 rows and 2 columns.
The columns are: id, target

-> data/tabular-playground-series-may-2022/sample_submission.csv has 100000 rows and 2 columns.
The columns are: id, target

-> data/tabular-playground-series-may-2022/test.csv has 100000 rows and 32 columns.
The columns are: id, f_00, f_01, f_02, f_03, f_04, f_05, f_06, f_07, f_08, f_09, f_10, f_11, f_12, f_13... and 17 more columns

-> data/tabular-playground-series-may-2022/train.csv has 800000 rows and 33 columns.
The columns are: id, f_00, f_01, f_02, f_03, f_04, f_05, f_06, f_07, f_08, f_09, f_10, f_11, f_12, f_13... and 18 more columns

-> data/test.csv has 100000 rows and 32 columns.
The columns are: id, f_00, f_01, f_02, f_03, f_04, f_05, f_06, f_07, f_08, f_09, f_10, f_11, f_12, f_13... and 17 more columns

-> data/train.csv has 800000 rows and 33 columns.
The columns are: id, f_00, f_01, f_02, f_03, f_04, f_05, f_06, f_07, f_08, f_09, f_10, f_11, f_12, f_13... and 18 more columns

-> input/sample_submission.csv has 100000 rows and 2 columns.
The columns are: id, target

-> (stopped after 10 files for performance)

# 5. Target score

0.99044

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt


import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
train_df = pd.read_csv("/kaggle/input/tabular-playground-series-may-2022/train.csv")
test_df = pd.read_csv("/kaggle/input/tabular-playground-series-may-2022/test.csv")



## === cell 2
train_df.count(), train_df.dtypes



## === cell 3
train_df.target.value_counts()



## === cell 4
fs = [f"f_{i:02d}" for i in range(0, 31)]
num_df = train_df[["target"] + fs].drop(["f_27"], axis=1)



## === cell 5
fig, axs = plt.subplots(6, 6, figsize=(24, 20))
for i, col in enumerate(num_df.columns):
    num_df[f"{col}"].plot(kind="hist", bins=100, ax=axs[i // 6, i % 6], title=f"{col}")
fig.tight_layout()



## === cell 6
num_df.f_18.unique(), num_df.f_29.unique(), num_df.f_30.unique()



## === cell 7
num_df.corr()



## === cell 8
display(train_df["f_27"])
print(
    "% of unique sequence in train",
    len(train_df["f_27"].unique()) / len(train_df["f_27"]),
)
print(
    "% of unique sequence in test", len(test_df["f_27"].unique()) / len(test_df["f_27"])
)



## === cell 9
f27_target_seq_counts = (
    train_df[["target", "f_27"]]
    .groupby(["target", "f_27"])
    .size()
    .unstack(fill_value=0)
    .T
)

most_occur_in_0 = f27_target_seq_counts.sort_values(by=[0], ascending=False)
most_occur_in_1 = f27_target_seq_counts.sort_values(by=[1], ascending=False)

fig, axs = plt.subplots(2, figsize=(20, 10))
most_occur_in_0.head(5000).plot(
    kind="area",
    stacked=False,
    ax=axs[0],
    title="Top 5000 occuring duplicated sequences for target 0",
)
most_occur_in_1.head(5000).plot(
    kind="area",
    stacked=False,
    ax=axs[1],
    title="Top 5000 occuring duplicated sequences for target 1",
)
fig.tight_layout()



## === cell 10
f27_map = most_occur_in_0.reset_index()
f27_map.columns = ["f_27", "f_27_tar0", "f_27_tar1"]
f27_map



## === cell 11
print("Among all sequences in test set")
print(
    "% of same duplicated sequence as in train set:",
    test_df["f_27"].isin(f27_target_seq_counts.index).sum() / len(test_df),
)
print(
    "# of same top 100 duplicated sequence as in train set, for target 0:",
    test_df["f_27"].isin(most_occur_in_0.index).sum(),
)
print(
    "# of same top 100 duplicated sequence as in train set, for target 1:",
    test_df["f_27"].isin(most_occur_in_1.index).sum(),
)



## === cell 12
f27_split = train_df["f_27"].str.split(pat="\s*", expand=True).iloc[:, 1:-1]
f27_split_test = test_df["f_27"].str.split(pat="\s*", expand=True).iloc[:, 1:-1]



## === cell 13
fig, axs = plt.subplots(5, 2, figsize=(10, 20))
for i in range(10):
    f27_split.iloc[:, i].value_counts().plot(
        kind="bar", ax=axs[i // 2, i % 2], title=f"sequence position {i}"
    )
fig.tight_layout()



## === cell 14
fig, axs = plt.subplots(5, 2, figsize=(10, 20))
for i in range(10):
    f27_split_test.iloc[:, i].value_counts().plot(
        kind="bar", ax=axs[i // 2, i % 2], title=f"sequence position {i}"
    )
fig.tight_layout()



## === cell 15
f27_split["target"] = train_df["target"]



## === cell 16
fig, axs = plt.subplots(10, figsize=(10, 30))
for i in range(10):
    target_seqpos_counts = (
        f27_split[["target", i + 1]]
        .groupby(["target", i + 1])
        .size()
        .unstack(fill_value=0)
        .T
    )
    target_seqpos_counts.plot(kind="bar", ax=axs[i], title=f"sequence position {i}")
fig.tight_layout()



## === cell 17
f27_split_df = pd.DataFrame(f27_split)
f27_split_df.columns = [f"f_27_pos{i}" for i in range(10)] + ["target"]
full_df = pd.concat([train_df, f27_split_df], axis=1)

f27_split_test_df = pd.DataFrame(f27_split_test)
f27_split_test_df.columns = [f"f_27_pos{i}" for i in range(10)]
full_test_df = pd.concat([test_df, f27_split_test_df], axis=1)




## === cell 18
def aggregate_features(df):
    df["i_02_21"] = (df.f_21 + df.f_02 > 5.2).astype(int) - (
        df.f_21 + df.f_02 < -5.3
    ).astype(int)
    df["i_05_22"] = (df.f_22 + df.f_05 > 5.1).astype(int) - (
        df.f_22 + df.f_05 < -5.4
    ).astype(int)
    i_00_01_26 = df.f_00 + df.f_01 + df.f_26
    df["i_00_01_26"] = (i_00_01_26 > 5.0).astype(int) - (i_00_01_26 < -5.0).astype(int)

    df["f_27_unique_len"] = df["f_27"].apply(lambda x: len(set(x)))
    df["f_27_unique_char"] = df["f_27"].apply(lambda x: "".join(sorted(set(x))))
    return df


full_df = aggregate_features(full_df)
full_test_df = aggregate_features(full_test_df)



## === cell 19
full_df[["f_27", "f_27_unique_char"] + [f"f_27_pos{i}" for i in range(10)]]



## === cell 20
cat_cols = ["f_27", "f_27_unique_char"] + [f"f_27_pos{i}" for i in range(10)]
cat_cols += [f"f_{i:02d}" for i in range(7, 19)] + [
    "f_29",
    "f_30",
    "i_02_21",
    "i_05_22",
    "i_00_01_26",
]

X_train = full_df.drop(["id", "target"], axis=1)
X_train[cat_cols] = X_train[cat_cols].astype("category")
y_train = train_df.target

X_test = full_test_df.drop(["id"], axis=1)
X_test[cat_cols] = X_test[cat_cols].astype("category")



## === cell 21
X_train.dtypes



## === cell 22
import lightgbm as lgb

train_data = lgb.Dataset(X_train, label=y_train)



## === cell 23
import optuna
import sklearn.metrics
from sklearn.model_selection import train_test_split


def objective(trial):
    param = {
        "objective": "cross_entropy",
        "metric": "auc",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "num_leaves": trial.suggest_int("num_leaves", 2, 256),
        "max_depth": trial.suggest_int("max_depth", 2, 10),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "num_iterations": trial.suggest_int("num_iterations", 20, 200),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 30),
    }

    train_data = lgb.Dataset(X_train, label=y_train)
    pruning_callback = optuna.integration.LightGBMPruningCallback(trial, "auc")
    hist = lgb.cv(param, train_data, nfold=3, callbacks=[pruning_callback])
    auc = hist["auc-mean"][-1]
    return auc




## === cell 24
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)



## === cell 25
val_params = {
    "objective": "cross_entropy",
    "metric": "auc",
    "verbosity": -1,
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 241,
    "max_depth": 10,
    "feature_fraction": 0.9751915273276082,
    "bagging_fraction": 0.9183132982296186,
    "bagging_freq": 5,
    "num_iterations": 500,
    "min_child_samples": 11,
}
val_train_data = lgb.Dataset(X_tr, label=y_tr)
val_clf = lgb.train(val_params, val_train_data)
val_pred = val_clf.predict(X_val)
val_auc = sklearn.metrics.roc_auc_score(y_val, val_pred)
print(f"Validation AUC (internal estimate): {val_auc:.5f}")



## === cell 26
params = {
    "num_leaves": 241,
    "max_depth": 10,
    "feature_fraction": 0.9751915273276082,
    "bagging_fraction": 0.9183132982296186,
    "bagging_freq": 5,
    "num_iterations": 500,  # increased from 193
    "min_child_samples": 11,
    "learning_rate": 0.05,  # added for finer boosting
}

clf = lgb.train(params, train_data)



## === cell 27
y_pred = clf.predict(X_test)



## === cell 28
submission = test_df[["id"]].copy()
submission["target"] = y_pred
submission.loc[submission.target < 0, "target"] = 0
submission.to_csv("submission.csv", index=False)



## === cell 29
pd.read_csv("submission.csv")
