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
sklearn-pandas==2.2.0
xgboost==2.0.3

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

0.50004

# 6. Current score

0.97241

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.97241) has done: 'I fix the pipeline so it runs end-to-end in Kaggle CPU-only environments by switching XGBoost from `gpu_hist` to a CPU tree method and ensuring categorical handling works reliably. Your feature engineering function currently doesn’t return anything and can leave object/categorical dtypes that XGBoost may not accept as-is, so I make it return the modified DataFrame and convert categorical/object columns to integer codes consistently for both train and test. I also remove the notebook-only `display()` call (which can error in a .py run) and ensure the submission is written with the required columns `id,target` to `submission.csv`. These changes are execution/stability fixes and should yield a sensible AUC above random (thus moving toward your ~0.50 target).'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (20, 10)



## === cell 1
train_df = pd.read_csv(
    "../input/tabular-playground-series-may-2022/train.csv", index_col="id"
)
test_df = pd.read_csv(
    "../input/tabular-playground-series-may-2022/test.csv", index_col="id"
)



## === cell 2
import string

characters = list(string.ascii_uppercase)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bug fix: original function mutates df in-place but returned None; keep same core
    feature logic but return df to make usage robust and explicit.
    Also add a safe fill for f_27 and stable categorical coding.
    """
    if "f_27" in df.columns:
        df["f_27"] = df["f_27"].astype("string").fillna("")

        for ch in characters:
            df[ch] = df["f_27"].str.count(ch)

        df["unique_text_str"] = df["f_27"].apply(
            lambda x: "".join([str(n) for n in list(set(x))])
        )
        df["unique_text_str"] = df["unique_text_str"].astype("category")
        df["unique_text_len"] = df["f_27"].apply(lambda s: len(set(s)))

        df.drop("f_27", axis=1, inplace=True)

    df["i_02_21"] = (df.f_21 + df.f_02 > 5.2).astype(int) - (
        df.f_21 + df.f_02 < -5.3
    ).astype(int)
    df["i_05_22"] = (df.f_22 + df.f_05 > 5.1).astype(int) - (
        df.f_22 + df.f_05 < -5.4
    ).astype(int)
    i_00_01_26 = df.f_00 + df.f_01 + df.f_26
    df["i_00_01_26"] = (i_00_01_26 > 5.0).astype(int) - (i_00_01_26 < -5.0).astype(int)

    return df


def make_xgb_compatible(X: pd.DataFrame) -> pd.DataFrame:
    """
    Bug fix: XGBClassifier with enable_categorical can be finicky with pandas category/object.
    Minimal change: convert object/category columns to deterministic integer codes.
    """
    X = X.copy()
    cat_cols = X.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()
    for c in cat_cols:
        X[c] = X[c].astype("category").cat.codes.astype(np.int32)
    return X




## === cell 3
X_train = train_df.drop(["target"], axis=1)
y_train = train_df["target"].astype(np.int32)
X_test = test_df.copy()

X_train = engineer_features(X_train)
X_test = engineer_features(X_test)

missing_in_test = [c for c in X_train.columns if c not in X_test.columns]
missing_in_train = [c for c in X_test.columns if c not in X_train.columns]
if missing_in_test:
    for c in missing_in_test:
        X_test[c] = 0
if missing_in_train:
    for c in missing_in_train:
        X_train[c] = 0
X_test = X_test[X_train.columns]

X_train = make_xgb_compatible(X_train)
X_test = make_xgb_compatible(X_test)

submission = pd.DataFrame({"id": X_test.index})

print("Train shape:", X_train.shape, "Test shape:", X_test.shape)
print("Dtypes summary (train):")
print(X_train.dtypes.value_counts())



## === cell 4
from xgboost import XGBClassifier

model_xgb = XGBClassifier(
    tree_method="hist",  # CPU-friendly; fixes "Must have at least one device"
    enable_categorical=False,  # we already encoded categoricals as codes
    eval_metric="auc",
    random_state=42,
)

model_xgb.fit(X_train, y_train)



## === cell 5
y_xgb = model_xgb.predict_proba(X_test)[:, 1]
submission["target"] = y_xgb

print(submission.head())
print("Prediction range:", float(np.min(y_xgb)), float(np.max(y_xgb)))



## === cell 6
try:
    from xgboost import plot_importance

    plot_importance(model_xgb, max_num_features=30)
    plt.show()
except Exception as e:
    print("Skipping feature importance plot due to:", repr(e))



## === cell 7
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
