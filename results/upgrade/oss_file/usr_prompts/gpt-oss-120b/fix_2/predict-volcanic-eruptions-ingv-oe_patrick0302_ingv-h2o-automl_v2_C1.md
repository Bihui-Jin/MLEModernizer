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
h2o==3.46.0.8
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
sklearn-pandas==2.2.0

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

7791545.688837831

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import h2o
from h2o.automl import H2OAutoML

h2o.init(max_mem_size="16G", nthreads=-1, quiet=True)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
H2OTypeError                              Traceback (most recent call last)
/tmp/ipykernel_11/3211032229.py in <cell line: 0>()
      6 
      7 # initialise H2O cluster
----> 8 h2o.init(max_mem_size="16G", nthreads=-1, quiet=True)
      9 

/usr/local/lib/python3.11/dist-packages/h2o/h2o.py in init(url, ip, port, name, https, cacert, insecure, username, password, cookies, proxy, start_h2o, nthreads, ice_root, log_dir, log_level, max_log_file_size, enable_assertions, max_mem_size, min_mem_size, strict_version_check, ignore_config, extra_classpath, jvm_custom_args, bind_to_localhost, verbose, **kwargs)
    214     assert_is_type(jvm_custom_args, [str], None)
    215     assert_is_type(bind_to_localhost, bool)
--> 216     assert_is_type(kwargs, {"proxies": {str: str}, "max_mem_size_GB": int, "min_mem_size_GB": int,
    217                             "force_connect": bool, "as_port": bool})
    218 

/usr/local/lib/python3.11/dist-packages/h2o/utils/typechecks.py in assert_is_type(var, *types, **kwargs)
    442     etn = _get_type_name(expected_type, dump=", ".join(args[1:]))
    443     vtn = _get_type_name(type(var))
--> 444     raise H2OTypeError(var_name=vname, var_value=var, var_type_name=vtn, exp_type_name=etn, message=message,
    445                        skip_frames=skip_frames)
    446 

H2OTypeError: Argument `kwargs` should be a dict("proxies": dict(string: string), "max_mem_size_GB": integer, "min_mem_size_GB": integer, "force_connect": bool, "as_port": bool), got dict {'quiet': True}

## === cell 1
BASE_PATH = "/kaggle/input/predict-volcanic-eruptions-ingv-oe"
TRAIN_META_PATH = os.path.join(BASE_PATH, "train.csv")
SAMPLE_SUB_PATH = os.path.join(BASE_PATH, "sample_submission.csv")
TRAIN_SEG_PATH = os.path.join(BASE_PATH, "train")
TEST_SEG_PATH = os.path.join(BASE_PATH, "test")

train_meta = pd.read_csv(TRAIN_META_PATH)
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)




## === cell 2
def extract_features_from_folder(ids, folder):
    """
    For each segment id, read the corresponding CSV and compute simple
    statistical features per sensor.
    Returns a DataFrame whose index aligns with `ids`.
    """
    features = []
    for seg_id in ids:
        file_path = os.path.join(folder, f"{seg_id}.csv")
        df = pd.read_csv(file_path)
        sensor_means = df.mean()
        sensor_stds = df.std()
        sensor_mins = df.min()
        sensor_maxs = df.max()
        feats = pd.concat([sensor_means, sensor_stds, sensor_mins, sensor_maxs])
        feats.index = (
            [f"{col}_mean" for col in df.columns]
            + [f"{col}_std" for col in df.columns]
            + [f"{col}_min" for col in df.columns]
            + [f"{col}_max" for col in df.columns]
        )
        features.append(feats)
    feature_df = pd.DataFrame(features)
    feature_df.insert(0, "segment_id", ids)
    return feature_df


train_features = extract_features_from_folder(
    train_meta["segment_id"].values, TRAIN_SEG_PATH
)
train_features = train_features.merge(
    train_meta[["segment_id", "time_to_eruption"]], on="segment_id", how="left"
)

test_features = extract_features_from_folder(
    sample_sub["segment_id"].values, TEST_SEG_PATH
)



## === cell 3
train_h2o = h2o.H2OFrame(train_features)
test_h2o = h2o.H2OFrame(test_features)

train_h2o["time_to_eruption"] = train_h2o["time_to_eruption"].asnumeric()

x_cols = [c for c in train_h2o.columns if c not in ["segment_id", "time_to_eruption"]]
y_col = "time_to_eruption"



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
H2OConnectionError                        Traceback (most recent call last)
/tmp/ipykernel_11/239243824.py in <cell line: 0>()
      1 # convert to H2O frames
----> 2 train_h2o = h2o.H2OFrame(train_features)
      3 test_h2o = h2o.H2OFrame(test_features)
      4 
      5 # set target column type to numeric

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

## === cell 4
aml = H2OAutoML(max_models=20, seed=121, max_runtime_secs=5 * 60)  # 5 minutes
aml.train(x=x_cols, y=y_col, training_frame=train_h2o)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
H2OConnectionError                        Traceback (most recent call last)
/tmp/ipykernel_11/3907726257.py in <cell line: 0>()
      1 # run AutoML
----> 2 aml = H2OAutoML(max_models=20, seed=121, max_runtime_secs=5 * 60)  # 5 minutes
      3 aml.train(x=x_cols, y=y_col, training_frame=train_h2o)
      4 

/usr/local/lib/python3.11/dist-packages/h2o/automl/_estimator.py in __init__(self, nfolds, balance_classes, class_sampling_factors, max_after_balance_size, max_runtime_secs, max_runtime_secs_per_model, max_models, distribution, stopping_metric, stopping_tolerance, stopping_rounds, seed, project_name, exclude_algos, include_algos, exploitation_ratio, modeling_plan, preprocessing, monotone_constraints, keep_cross_validation_predictions, keep_cross_validation_models, keep_cross_validation_fold_assignment, sort_metric, custom_metric_func, export_checkpoints_dir, verbosity, **kwargs)
    309         # Check if H2O jar contains AutoML
    310         try:
--> 311             h2o.api("GET /3/Metadata/schemas/AutoMLV99")
    312         except h2o.exceptions.H2OResponseError as e:
    313             print(e)

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

## === cell 5
preds_h2o = aml.leader.predict(test_h2o)
preds = preds_h2o.as_data_frame()["predict"].values

submission = pd.DataFrame(
    {"segment_id": sample_sub["segment_id"], "time_to_eruption": preds}
)

submission_path = "submission_recent.csv"
submission.to_csv(submission_path, index=False)

print(f"Submission written to {submission_path}")
print(submission.head())



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/940639435.py in <cell line: 0>()
      1 # predict on test set
----> 2 preds_h2o = aml.leader.predict(test_h2o)
      3 preds = preds_h2o.as_data_frame()["predict"].values
      4 
      5 # build submission dataframe matching sample format

NameError: name 'aml' is not defined

## === cell 6
h2o.shutdown(prompt=False)

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
H2OConnectionError                        Traceback (most recent call last)
/tmp/ipykernel_11/4127463089.py in <cell line: 0>()
      1 # clean up H2O
----> 2 h2o.shutdown(prompt=False)

/usr/local/lib/python3.11/dist-packages/h2o/utils/metaclass.py in wrapper(*args, **kwargs)
    185         def wrapper(*args, **kwargs):
    186             warnings.warn(msg, H2ODeprecationWarning, 2)
--> 187             return call_fn(*args, **kwargs)
    188 
    189         return wrapper

/usr/local/lib/python3.11/dist-packages/h2o/h2o.py in shutdown(prompt)
   2586 def shutdown(prompt=False):
   2587     """Deprecated."""
-> 2588     _check_connection()
   2589     cluster().shutdown(prompt)
   2590 

/usr/local/lib/python3.11/dist-packages/h2o/h2o.py in _check_connection()
   2532 def _check_connection():
   2533     if not cluster():
-> 2534         raise H2OConnectionError("Not connected to a cluster. Did you run `h2o.init()` or `h2o.connect()`?")
   2535 
   2536 

H2OConnectionError: Not connected to a cluster. Did you run `h2o.init()` or `h2o.connect()`?
