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

0.95988

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.95988) has done: 'The fix switches XGBoost to CPU‑only training by using `tree_method="hist"` (the original GPU setting caused a runtime error). After fitting, we store the predicted probability of class 1 in a column named **target** to match the submission format, and write the CSV correctly. No other logic is changed.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (20, 10)  # make plots a bit bigger




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


def engineer_features(df):
    for ch in characters:
        df[ch] = df["f_27"].str.count(ch)

    df["unique_text_str"] = df["f_27"].apply(
        lambda x: "".join([str(n) for n in list(set(x))])
    )
    df["unique_text_str"] = df["unique_text_str"].astype("category")
    df["unique_text_len"] = df.f_27.apply(lambda s: len(set(s)))

    df.drop("f_27", axis=1, inplace=True)

    df["i_02_21"] = (df.f_21 + df.f_02 > 5.2).astype(int) - (
        df.f_21 + df.f_02 < -5.3
    ).astype(int)
    df["i_05_22"] = (df.f_22 + df.f_05 > 5.1).astype(int) - (
        df.f_22 + df.f_05 < -5.4
    ).astype(int)
    i_00_01_26 = df.f_00 + df.f_01 + df.f_26
    df["i_00_01_26"] = (i_00_01_26 > 5.0).astype(int) - (i_00_01_26 < -5.0).astype(int)




## === cell 3
X_train = train_df.drop(["target"], axis=1)
y_train = train_df["target"]
X_test = test_df

engineer_features(X_train)
engineer_features(X_test)

submission = pd.DataFrame(index=X_test.index)  # prepare df for submission

display(X_train.head(), y_train.head(), X_test.head())




## === cell 4
from xgboost import XGBClassifier

model_xgb = XGBClassifier(
    tree_method="hist",
    enable_categorical=True,
    eval_metric="logloss",
    use_label_encoder=False,
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
)

model_xgb.fit(X_train, y_train)




## === cell 5
y_xgb = model_xgb.predict_proba(X_test)
submission["target"] = y_xgb[:, 1]




## === cell 6
from xgboost import plot_importance

plot_importance(model_xgb, max_num_features=30)
plt.show()




## === cell 7
submission.to_csv("submission.csv", index=True)
