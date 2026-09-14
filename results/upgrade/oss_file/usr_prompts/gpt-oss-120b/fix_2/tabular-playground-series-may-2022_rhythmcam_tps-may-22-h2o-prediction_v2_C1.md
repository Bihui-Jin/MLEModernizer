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
h2o==3.46.0.8
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
scipy==1.15.3
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

0.9776

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd

import h2o
from h2o.automl import H2OAutoML

TRAIN_PATH = "../input/tabular-playground-series-may-2022/train.csv"
TEST_PATH = "../input/tabular-playground-series-may-2022/test.csv"
SAMPLE_SUBMISSION_PATH = (
    "../input/tabular-playground-series-may-2022/sample_submission.csv"
)
SUBMISSION_PATH = "submission.csv"  # removed trailing space

ID = "id"
TARGET = "target"

NEW_TRAIN_PATH = "train.csv"
NEW_TEST_PATH = "test.csv"

SEED_LIST = [7, 77]
MAX_RUNTIME_SECS = 60 * 3




## === cell 1
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

for df in [train, test]:
    for i in range(10):
        df[f"ch{i}"] = df.f_27.str.get(i).apply(ord) - ord("A")
    df["unique_characters"] = df.f_27.apply(lambda s: len(set(s)))

train.to_csv(NEW_TRAIN_PATH, index=False)
test.to_csv(NEW_TEST_PATH, index=False)




## === cell 2
h2o.init()

train_h2o = h2o.import_file(NEW_TRAIN_PATH)
test_h2o = h2o.import_file(NEW_TEST_PATH)

train_h2o[TARGET] = train_h2o[TARGET].asfactor()

x_cols = train_h2o.columns
x_cols.remove(TARGET)
x_cols.remove(ID)  # drop identifier

pred_prob_list = []  # store probability of class 1 for each seed

for selSeed in SEED_LIST:
    aml = H2OAutoML(max_runtime_secs=MAX_RUNTIME_SECS, seed=selSeed)
    aml.train(x=x_cols, y=TARGET, training_frame=train_h2o)

    preds_h2o = aml.predict(test_h2o).as_data_frame()
    prob_class1 = preds_h2o["p1"]  # probability of target == 1
    pred_prob_list.append(prob_class1.values)




## === cell 3
final_test_pred = np.mean(np.column_stack(pred_prob_list), axis=1)

submission = pd.read_csv(SAMPLE_SUBMISSION_PATH)
submission[TARGET] = final_test_pred
submission.to_csv(SUBMISSION_PATH, index=False)

submission.head()
