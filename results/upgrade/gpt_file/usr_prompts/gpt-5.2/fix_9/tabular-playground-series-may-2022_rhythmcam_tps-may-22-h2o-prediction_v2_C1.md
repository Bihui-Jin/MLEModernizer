# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

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

    arr = s.to_numpy(dtype=object, copy=False)

    arr_u = arr.astype("U")
    arr_u = np.char.substr(arr_u, 0, 10)
    lens = np.char.str_len(arr_u).astype(np.int32, copy=False)
    pad_len = (10 - lens).clip(min=0)
    arr_u10 = np.char.add(arr_u, np.char.multiply("A", pad_len)).astype(
        "U10", copy=False
    )

    df["f_27"] = pd.Series(arr_u10, index=df.index, dtype="string").astype(str)

    mat = arr_u10.view("U1").reshape(-1, 10)

    codes = (mat.view(np.uint32) - np.uint32(A_ORD)).astype(np.int16, copy=False)
    for i in range(10):
        df[f"ch{i}"] = codes[:, i]

    mat_u32 = mat.view(np.uint32)
    mat_sorted = np.sort(mat_u32, axis=1)
    uniq_cnt = (1 + (mat_sorted[:, 1:] != mat_sorted[:, :-1]).sum(axis=1)).astype(
        np.int16, copy=False
    )
    df["unique_characters"] = uniq_cnt


add_f27_features(train)
add_f27_features(test)




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/1850995684.py in <cell line: 0>()
     57 
     58 
---> 59 add_f27_features(train)
     60 add_f27_features(test)
     61 

/tmp/ipykernel_11/1850995684.py in add_f27_features(df)
     29     # Using np.char is vectorized and avoids per-row Python loops.
     30     arr_u = arr.astype("U")
---> 31     arr_u = np.char.substr(arr_u, 0, 10)
     32     lens = np.char.str_len(arr_u).astype(np.int32, copy=False)
     33     pad_len = (10 - lens).clip(min=0)

AttributeError: module 'numpy.core.defchararray' has no attribute 'substr'

## === cell 2
ICE_ROOT = "/kaggle/working/h2o_ice"
os.makedirs(ICE_ROOT, exist_ok=True)

H2O_PORT = 54321

try:
    h2o.connection()
    _need_init = False
except Exception:
    _need_init = True

if _need_init:
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
for col, t in types.items():
    if col == ID or col == TARGET:
        continue
    if t == "string":
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




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
H2OConnectionError                        Traceback (most recent call last)
/tmp/ipykernel_11/41775038.py in <cell line: 0>()
     23 
     24 # Optimization: create H2OFrames once; avoid repeated conversions.
---> 25 train_h2o = h2o.H2OFrame(train)
     26 test_h2o = h2o.H2OFrame(test)
     27 

/usr/local/lib/python3.11/dist-packages/h2o/frame.py in __init__(self, python_obj, destination_frame, header, separator, column_names, column_types, na_strings, skipped_columns, force_col_types)
    118             self._ex._children = None
    119             if python_obj is not None:
--> 120                 self._upload_python_object(python_obj, destination_frame, header, separator,
    121                                            column_names, column_types, na_strings, skipped_columns, force_col_types)
    122 

/usr/local/lib/python3.11/dist-packages/h2o/frame.py in _upload_python_object(self, python_obj, destination_frame, header, separator, column_names, column_types, na_strings, skipped_columns, force_col_types)
    159             csv_writer.writerows(data_to_write)
    160         tmp_file.close()  # close the streams
--> 161         self._upload_parse(tmp_path, destination_frame, 1, separator, column_names, column_types, na_strings,
    162                            skipped_columns, force_col_types)
    163         os.remove(tmp_path)  # delete the tmp file

/usr/local/lib/python3.11/dist-packages/h2o/frame.py in _upload_parse(self, path, destination_frame, header, sep, column_names, column_types, na_strings, skipped_columns, force_col_types, quotechar, escapechar)
    464     def _upload_parse(self, path, destination_frame, header, sep, column_names, column_types, na_strings, 
    465                       skipped_columns=None, force_col_types=False, quotechar=None, escapechar=None):
--> 466         ret = h2o.api("POST /3/PostFile", filename=path)
    467         rawkey = ret["destination_frame"]
    468         self._parse(rawkey, destination_frame, header, sep, column_names, column_types, na_strings, skipped_columns,

/usr/local/lib/python3.11/dist-packages/h2o/h2o.py in api(endpoint, data, json, filename, save_to)
    120     """
    121     # type checks are performed in H2OConnection class
--> 122     _check_connection()
    123     return h2oconn.request(endpoint, data=data, json=json, filename=filename, save_to=save_to)
    124 

/usr/local/lib/python3.11/dist-packages/h2o/h2o.py in _check_connection()
   2532 def _check_connection():
   2533     if not cluster():
-> 2534         raise H2OConnectionError("Not connected to a cluster. Did you run `h2o.init()` or `h2o.connect()`?")
   2535 
   2536 

H2OConnectionError: Not connected to a cluster. Did you run `h2o.init()` or `h2o.connect()`?

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

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/496265647.py in <cell line: 0>()
----> 1 if not pred_test:
      2     raise RuntimeError(
      3         "No predictions were generated (pred_test is empty). Check H2O training/prediction."
      4     )
      5 

NameError: name 'pred_test' is not defined
