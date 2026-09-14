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

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os

## === cell 1
import h2o
print(h2o.__version__)
from h2o.automl import H2OAutoML

h2o.init(max_mem_size='16G')

## === cell 2
train = pd.read_csv('/kaggle/input/predict-volcanic-eruptions-ingv-oe/train.csv')
test = pd.read_csv('/kaggle/input/predict-volcanic-eruptions-ingv-oe/sample_submission.csv')

## === cell 3
scaled_feature_df = pd.read_pickle('../input/ingv-volcanic-eruption-prediction-add-resampling/scaled_feature_df.pickle')
scaled_test_df = pd.read_pickle('../input/ingv-volcanic-eruption-prediction-add-resampling/scaled_test_df.pickle')

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/92423872.py in <cell line: 0>()
----> 1 scaled_feature_df = pd.read_pickle('../input/ingv-volcanic-eruption-prediction-add-resampling/scaled_feature_df.pickle')
      2 scaled_test_df = pd.read_pickle('../input/ingv-volcanic-eruption-prediction-add-resampling/scaled_test_df.pickle')

/usr/local/lib/python3.11/dist-packages/pandas/io/pickle.py in read_pickle(filepath_or_buffer, compression, storage_options)
    183     """
    184     excs_to_catch = (AttributeError, ImportError, ModuleNotFoundError, TypeError)
--> 185     with get_handle(
    186         filepath_or_buffer,
    187         "rb",

/usr/local/lib/python3.11/dist-packages/pandas/io/common.py in get_handle(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)
    880         else:
    881             # Binary mode
--> 882             handle = open(handle, ioargs.mode)
    883         handles.append(handle)
    884 

FileNotFoundError: [Errno 2] No such file or directory: '../input/ingv-volcanic-eruption-prediction-add-resampling/scaled_feature_df.pickle'

## === cell 4
train_h2o = h2o.H2OFrame(scaled_feature_df)
train_label_h2o = h2o.H2OFrame(train[['time_to_eruption']])
train_h2o['time_to_eruption'] = train_label_h2o['time_to_eruption']

test_feature_h2o = h2o.H2OFrame(scaled_test_df)

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3797286598.py in <cell line: 0>()
----> 1 train_h2o = h2o.H2OFrame(scaled_feature_df)
      2 train_label_h2o = h2o.H2OFrame(train[['time_to_eruption']])
      3 train_h2o['time_to_eruption'] = train_label_h2o['time_to_eruption']
      4 
      5 test_feature_h2o = h2o.H2OFrame(scaled_test_df)

NameError: name 'scaled_feature_df' is not defined

## === cell 5
print(train_h2o.shape)
print(test_feature_h2o.shape)

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4139462414.py in <cell line: 0>()
----> 1 print(train_h2o.shape)
      2 print(test_feature_h2o.shape)

NameError: name 'train_h2o' is not defined

## === cell 6
x = test_feature_h2o.columns
y = 'time_to_eruption'

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4244305951.py in <cell line: 0>()
----> 1 x = test_feature_h2o.columns
      2 y = 'time_to_eruption'

NameError: name 'test_feature_h2o' is not defined

## === cell 7
aml = H2OAutoML(max_models=1000, seed=121, max_runtime_secs=1*60)
aml.train(x=x, y=y, training_frame=train_h2o)

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/523175500.py in <cell line: 0>()
      1 aml = H2OAutoML(max_models=1000, seed=121, max_runtime_secs=1*60)
----> 2 aml.train(x=x, y=y, training_frame=train_h2o)

NameError: name 'x' is not defined

## === cell 8
lb = aml.leaderboard
lb.head(rows=lb.nrows)  # Print all rows instead of default (10 rows)

## === cell 9
aml.leader

## === cell 10

preds = aml.predict(test_feature_h2o)

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1612864829.py in <cell line: 0>()
      3 # model object directly
      4 
----> 5 preds = aml.predict(test_feature_h2o)

NameError: name 'test_feature_h2o' is not defined

## === cell 11
submission = pd.DataFrame()
submission['segment_id'] = test['segment_id']
submission['time_to_eruption'] = preds.as_data_frame().values.flatten()
submission.to_csv('submission_recent.csv', header=True, index=False)

## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2841173454.py in <cell line: 0>()
      1 submission = pd.DataFrame()
      2 submission['segment_id'] = test['segment_id']
----> 3 submission['time_to_eruption'] = preds.as_data_frame().values.flatten()
      4 submission.to_csv('submission_recent.csv', header=True, index=False)

NameError: name 'preds' is not defined

## === cell 12
submission
