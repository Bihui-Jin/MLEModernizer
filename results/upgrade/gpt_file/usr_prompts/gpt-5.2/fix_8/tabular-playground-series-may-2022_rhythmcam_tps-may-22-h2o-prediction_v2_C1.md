# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

import h2o
from h2o.automl import H2OAutoML

TRAIN_PATH = "/kaggle/input/tabular-playground-series-may-2022/train.csv"
TEST_PATH = "/kaggle/input/tabular-playground-series-may-2022/test.csv"
SAMPLE_SUBMISSION_PATH = (
    "/kaggle/input/tabular-playground-series-may-2022/sample_submission.csv"
)

if not os.path.exists(TRAIN_PATH):
    TRAIN_PATH = "/kaggle/data/tabular-playground-series-may-2022/train.csv"
    TEST_PATH = "/kaggle/data/tabular-playground-series-may-2022/test.csv"
    SAMPLE_SUBMISSION_PATH = (
        "/kaggle/data/tabular-playground-series-may-2022/sample_submission.csv"
    )

SUBMISSION_PATH = "submission.csv"

ID = "id"
TARGET = "target"

SEED_LIST = [7, 77]
MAX_RUNTIME_SECS = 60 * 4  # 240

os.environ.setdefault("PYTHONHASHSEED", "0")
np.random.seed(0)



## === cell 1
read_csv_kwargs = {}
try:
    import pyarrow  # noqa: F401

    read_csv_kwargs["engine"] = "pyarrow"
except Exception:
    pass

read_csv_kwargs["dtype"] = {"f_27": "string"}

train = pd.read_csv(TRAIN_PATH, **read_csv_kwargs)
test = pd.read_csv(TEST_PATH, **read_csv_kwargs)

A_ORD = ord("A")


def add_f27_features(df: pd.DataFrame) -> None:
    s = df["f_27"].astype("string").fillna("")
    s = s.str.pad(width=10, side="right", fillchar="A").str.slice(0, 10)
    df["f_27"] = s.astype(str)

    arr = s.to_numpy(dtype="U10", copy=False)  # shape (n,), each entry length 10
    mat = arr.view("U1").reshape(-1, 10)  # shape (n,10), each char a 'U1'

    codes = (mat.view(np.uint32) - np.uint32(A_ORD)).astype(np.int16, copy=False)
    for i in range(10):
        df[f"ch{i}"] = codes[:, i]

    mat_u32 = mat.view(np.uint32)
    mat_sorted = np.sort(mat_u32, axis=1)
    uniq_cnt = (1 + (mat_sorted[:, 1:] != mat_sorted[:, :-1]).sum(axis=1)).astype(
        np.int16
    )
    df["unique_characters"] = uniq_cnt


add_f27_features(train)
add_f27_features(test)



## === cell 2
ICE_ROOT = "/kaggle/working/h2o_ice"
os.makedirs(ICE_ROOT, exist_ok=True)

H2O_PORT = 54321

try:
    h2o.connection()
    try:
        h2o.shutdown(prompt=False)
    except Exception:
        pass
except Exception:
    pass

h2o.init(
    start_h2o=True,
    ip="127.0.0.1",
    port=H2O_PORT,
    max_mem_size="4G",
    nthreads=-1,
    ice_root=ICE_ROOT,
)

train_h2o = h2o.H2OFrame(train)
test_h2o = h2o.H2OFrame(test)

train_h2o[TARGET] = train_h2o[TARGET].asfactor()

types = train_h2o.types
cols = train_h2o.columns
for col in cols:
    if col == ID or col == TARGET:
        continue
    if types.get(col) == "string":
        train_h2o[col] = train_h2o[col].asfactor()
        test_h2o[col] = test_h2o[col].asfactor()

x = train_h2o.columns
y = TARGET
x.remove(y)
x.remove(ID)

pred_test = []
for selSeed in SEED_LIST:
    aml_y = H2OAutoML(
        max_runtime_secs=MAX_RUNTIME_SECS,
        seed=selSeed,
    )
    aml_y.train(x=x, y=y, training_frame=train_h2o)

    preds_df = aml_y.predict(test_h2o).as_data_frame()

    if "p1" in preds_df.columns:
        prob = preds_df["p1"].to_numpy(dtype=float)
    elif "p0" in preds_df.columns:
        prob = 1.0 - preds_df["p0"].to_numpy(dtype=float)
    else:
        prob = preds_df["predict"].astype(float).to_numpy()

    pred_test.append(prob)



## === cell 3
if not pred_test:
    raise RuntimeError(
        "No predictions were generated (pred_test is empty). Check H2O training/prediction."
    )

final_test_pred = np.mean(np.vstack(pred_test), axis=0)

submission = pd.read_csv(SAMPLE_SUBMISSION_PATH)

if len(final_test_pred) != len(submission):
    raise ValueError(
        f"Prediction length {len(final_test_pred)} != submission length {len(submission)}"
    )

submission[TARGET] = final_test_pred
submission[[ID, TARGET]].to_csv(SUBMISSION_PATH, index=False)

print(submission.head())
print(
    f"Wrote: {SUBMISSION_PATH} with shape={submission.shape} and columns={list(submission.columns)}"
)

try:
    h2o.shutdown(prompt=False)
except Exception:
    pass
