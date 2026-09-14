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



## === cell 1
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

for df in [train, test]:
    s = df["f_27"].astype("string").fillna("")
    s = s.str.pad(width=10, side="right", fillchar="A").str.slice(0, 10)
    df["f_27"] = s.astype(str)

    for i in range(10):
        df[f"ch{i}"] = (df["f_27"].str.get(i).apply(ord) - ord("A")).astype(np.int16)

    df["unique_characters"] = df["f_27"].apply(lambda x: len(set(x))).astype(np.int16)



## === cell 2
h2o.init(
    max_mem_size="4G",
    nthreads=-1,
    ice_root="/kaggle/working/h2o_ice",
)

train_h2o = h2o.H2OFrame(train)
test_h2o = h2o.H2OFrame(test)

train_h2o[TARGET] = train_h2o[TARGET].asfactor()

for col in train_h2o.columns:
    if col in (ID, TARGET):
        continue
    if train_h2o.types.get(col) == "string":
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
    elif "p1" in preds_df.columns:
        prob = preds_df["p1"].to_numpy(dtype=float)
    else:
        prob = preds_df["predict"].astype(float).to_numpy()

    pred_test.append(prob)



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
H2OConnectionError                        Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/h2o/h2o.py in init(url, ip, port, name, https, cacert, insecure, username, password, cookies, proxy, start_h2o, nthreads, ice_root, log_dir, log_level, max_log_file_size, enable_assertions, max_mem_size, min_mem_size, strict_version_check, ignore_config, extra_classpath, jvm_custom_args, bind_to_localhost, verbose, **kwargs)
    269     try:
--> 270         h2oconn = H2OConnection.open(url=url, ip=ip, port=port, name=name, https=https,
    271                                      verify_ssl_certificates=verify_ssl_certificates, cacert=cacert,

/usr/local/lib/python3.11/dist-packages/h2o/backend/connection.py in open(server, url, ip, port, name, https, auth, verify_ssl_certificates, cacert, proxy, cookies, verbose, msgs, strict_version_check)
    405             conn._timeout = 3.0
--> 406             conn._cluster = conn._test_connection(retries, messages=msgs)
    407             # If a server is unable to respond within 1s, it should be considered a bug. However we disable this

/usr/local/lib/python3.11/dist-packages/h2o/backend/connection.py in _test_connection(self, max_retries, messages)
    712         else:
--> 713             raise H2OConnectionError("Could not establish link to the H2O cloud %s after %d retries\n%s"
    714                                      % (self._base_url, max_retries, "\n".join(errors)))

H2OConnectionError: Could not establish link to the H2O cloud http://localhost:54321 after 5 retries
[23:53.46] H2OConnectionError: Unexpected HTTP error: HTTPConnectionPool(host='localhost', port=54321): Max retries exceeded with url: /3/Metadata/schemas/CloudV3 (Caused by NewConnectionError('<urllib3.connection.HTTPConnection object at 0x7fffaaf746d0>: Failed to establish a new connection: [Errno 111] Connection refused'))
[23:53.66] H2OConnectionError: Unexpected HTTP error: HTTPConnectionPool(host='localhost', port=54321): Max retries exceeded with url: /3/Metadata/schemas/CloudV3 (Caused by NewConnectionError('<urllib3.connection.HTTPConnection object at 0x7fffaaaa6cd0>: Failed to establish a new connection: [Errno 111] Connection refused'))
[23:53.86] H2OConnectionError: Unexpected HTTP error: HTTPConnectionPool(host='localhost', port=54321): Max retries exceeded with url: /3/Metadata/schemas/CloudV3 (Caused by NewConnectionError('<urllib3.connection.HTTPConnection object at 0x7fffaaab8f90>: Failed to establish a new connection: [Errno 111] Connection refused'))
[23:54.07] H2OConnectionError: Unexpected HTTP error: HTTPConnectionPool(host='localhost', port=54321): Max retries exceeded with url: /3/Metadata/schemas/CloudV3 (Caused by NewConnectionError('<urllib3.connection.HTTPConnection object at 0x7fffaaabaf90>: Failed to establish a new connection: [Errno 111] Connection refused'))
[23:54.27] H2OConnectionError: Unexpected HTTP error: HTTPConnectionPool(host='localhost', port=54321): Max retries exceeded with url: /3/Metadata/schemas/CloudV3 (Caused by NewConnectionError('<urllib3.connection.HTTPConnection object at 0x7fffaaac0e50>: Failed to establish a new connection: [Errno 111] Connection refused'))

During handling of the above exception, another exception occurred:

H2OTypeError                              Traceback (most recent call last)
/tmp/ipykernel_11/3645484382.py in <cell line: 0>()
      2 # This prevents failures due to IO and temp storage issues and should reliably produce a submission.
      3 # Also set H2O's working directory to /kaggle/working to avoid /tmp space problems.
----> 4 h2o.init(
      5     max_mem_size="4G",
      6     nthreads=-1,

/usr/local/lib/python3.11/dist-packages/h2o/h2o.py in init(url, ip, port, name, https, cacert, insecure, username, password, cookies, proxy, start_h2o, nthreads, ice_root, log_dir, log_level, max_log_file_size, enable_assertions, max_mem_size, min_mem_size, strict_version_check, ignore_config, extra_classpath, jvm_custom_args, bind_to_localhost, verbose, **kwargs)
    285                                      ' instance of H2O with https manually '
    286                                      '(https://docs.h2o.ai/h2o/latest-stable/h2o-docs/welcome.html#new-user-quick-start).')
--> 287         hs = H2OLocalServer.start(nthreads=nthreads, enable_assertions=enable_assertions, max_mem_size=mmax,
    288                                   min_mem_size=mmin, ice_root=ice_root, log_dir=log_dir, log_level=log_level,
    289                                   max_log_file_size=max_log_file_size, port=port, name=name,

/usr/local/lib/python3.11/dist-packages/h2o/backend/server.py in start(jar_path, nthreads, enable_assertions, max_mem_size, min_mem_size, ice_root, log_dir, log_level, max_log_file_size, port, name, extra_classpath, verbose, jvm_custom_args, bind_to_localhost)
    105         assert_satisfies(log_level, log_level in [None, "TRACE", "DEBUG", "INFO", "WARN", "ERRR", "FATA"])
    106         assert_is_type(max_log_file_size, str, None)
--> 107         assert_is_type(ice_root, None, I(str, os.path.isdir))
    108         assert_is_type(extra_classpath, None, [str])
    109         assert_is_type(jvm_custom_args, list, None)

/usr/local/lib/python3.11/dist-packages/h2o/utils/typechecks.py in assert_is_type(var, *types, **kwargs)
    442     etn = _get_type_name(expected_type, dump=", ".join(args[1:]))
    443     vtn = _get_type_name(type(var))
--> 444     raise H2OTypeError(var_name=vname, var_value=var, var_type_name=vtn, exp_type_name=etn, message=message,
    445                        skip_frames=skip_frames)
    446 

H2OTypeError: Argument `ice_root` should be a ?string & isdir, got string /kaggle/working/h2o_ice

## === cell 3
final_test_pred = np.mean(np.vstack(pred_test), axis=0)

submission = pd.read_csv(SAMPLE_SUBMISSION_PATH)

if len(final_test_pred) != len(submission):
    raise ValueError(
        f"Prediction length {len(final_test_pred)} != submission length {len(submission)}"
    )

submission[TARGET] = final_test_pred
submission.to_csv(SUBMISSION_PATH, index=False)

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
/tmp/ipykernel_11/217988984.py in <cell line: 0>()
----> 1 final_test_pred = np.mean(np.vstack(pred_test), axis=0)
      2 
      3 submission = pd.read_csv(SAMPLE_SUBMISSION_PATH)
      4 
      5 # Change: ensure alignment by id using the sample submission order (prevents accidental mis-order).

NameError: name 'pred_test' is not defined
