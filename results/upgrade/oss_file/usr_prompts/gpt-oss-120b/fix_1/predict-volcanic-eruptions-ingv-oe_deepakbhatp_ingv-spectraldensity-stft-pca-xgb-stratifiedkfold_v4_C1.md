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
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
scipy==1.15.3
sklearn-pandas==2.2.0
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

5009493.05033489

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 2
import numpy as np
import pandas as pd

import scipy
from scipy import signal
import random
import pickle

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

import xgboost

## === cell 13
train_df_stft = pd.read_pickle('../input/volcano-features/train_df_stft.pkl')
time = train_df_stft['time_to_eruption']
train_df_stft = train_df_stft.drop(['segment_id', 'time_to_eruption'], axis=1)

train_df_sd = pd.read_pickle('../input/volcano-features/train_df_sd.pkl')

test_df_stft = pd.read_pickle('../input/volcano-features/test_df_stft.pkl')
test_df_stft = test_df_stft.drop(['segment_id'],axis = 1)

test_df_sd = pd.read_pickle('../input/volcano-features/test_df_sd.pkl')

## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/149820931.py in <cell line: 0>()
----> 1 train_df_stft = pd.read_pickle('../input/volcano-features/train_df_stft.pkl')
      2 time = train_df_stft['time_to_eruption']
      3 train_df_stft = train_df_stft.drop(['segment_id', 'time_to_eruption'], axis=1)
      4 
      5 # train_df_sd = pd.read_pickle('../input/volcano-features/train_df_sd_only.pkl')

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

FileNotFoundError: [Errno 2] No such file or directory: '../input/volcano-features/train_df_stft.pkl'

## === cell 15
from sklearn.decomposition import PCA
pca = PCA(n_components=0.999,svd_solver="full") #all components that explain upto 99.9% of the variance
train_pca = pca.fit_transform(train_df_sd)
test_pca = pca.transform(test_df_sd)
train_pca = pd.DataFrame(train_pca)
test_pca = pd.DataFrame(test_pca)

## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/411406233.py in <cell line: 0>()
      1 from sklearn.decomposition import PCA
      2 pca = PCA(n_components=0.999,svd_solver="full") #all components that explain upto 99.9% of the variance
----> 3 train_pca = pca.fit_transform(train_df_sd)
      4 test_pca = pca.transform(test_df_sd)
      5 train_pca = pd.DataFrame(train_pca)

NameError: name 'train_df_sd' is not defined

## === cell 16
train_df = pd.concat([train_df_stft,train_pca] ,axis=1)
test_df = pd.concat([test_df_stft,test_pca],axis = 1)

## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/955833613.py in <cell line: 0>()
----> 1 train_df = pd.concat([train_df_stft,train_pca] ,axis=1)
      2 test_df = pd.concat([test_df_stft,test_pca],axis = 1)

NameError: name 'train_df_stft' is not defined

## === cell 18
input_df = pd.read_csv('../input/predict-volcanic-eruptions-ingv-oe/train.csv')
input_df = input_df.sort_values("time_to_eruption")
fold_list = [1,2,3,4,5]
folds = []
for i in range(int((input_df.shape[0]-1)/5)):
    random.shuffle(fold_list)
    folds.extend(fold_list)
folds = folds + [1] #adding a remaining solitary record to fold 1
input_df['fold'] = folds


## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/724199037.py in <cell line: 0>()
      7     folds.extend(fold_list)
      8 folds = folds + [1] #adding a remaining solitary record to fold 1
----> 9 input_df['fold'] = folds
     10 #input_df.head(20)

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __setitem__(self, key, value)
   4309         else:
   4310             # set column
-> 4311             self._set_item(key, value)
   4312 
   4313     def _setitem_slice(self, key: slice, value) -> None:

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in _set_item(self, key, value)
   4522         ensure homogeneity.
   4523         """
-> 4524         value, refs = self._sanitize_column(value)
   4525 
   4526         if (

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in _sanitize_column(self, value)
   5264 
   5265         if is_list_like(value):
-> 5266             com.require_length_match(value, self.index)
   5267         arr = sanitize_array(value, self.index, copy=True, allow_2d=True)
   5268         if (

/usr/local/lib/python3.11/dist-packages/pandas/core/common.py in require_length_match(data, index)
    571     """
    572     if len(data) != len(index):
--> 573         raise ValueError(
    574             "Length of values "
    575             f"({len(data)}) "

ValueError: Length of values (3986) does not match length of index (3987)

## === cell 19
predictions = np.zeros(len(test_df))
for fold in range(1,6):
    train_index_list = input_df[input_df['fold'] != fold].index
    test_index_list = input_df[input_df['fold'] == fold].index

    X_train = train_df.iloc[train_index_list]
    y_train = time[train_index_list]
    X_val = train_df.iloc[test_index_list]
    y_val = time[test_index_list]

    model = xgboost.XGBRegressor(n_estimators=100000,tree_method='gpu_hist',max_depth=8,learning_rate=0.05,alpha=0.1,SUBSAMPLE=0.6)#,colsample_bytree=0.5)
    eval_set = [(X_val, y_val)]
    model.fit(X_train, y_train,early_stopping_rounds=5,eval_metric='mae', eval_set=eval_set, verbose=False)
    print(model.evals_result()['validation_0']['mae'][-5:])
    predictions += model.predict(test_df)
predictions = predictions/5

sample_submission_df1=pd.read_csv('../input/predict-volcanic-eruptions-ingv-oe/sample_submission.csv')
sample_submission_df1['time_to_eruption']=predictions
sample_submission_df1.to_csv('xgb_5fldst_ft_sdpca_stft.csv',index=False)

## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3921501357.py in <cell line: 0>()
----> 1 predictions = np.zeros(len(test_df))
      2 for fold in range(1,6):
      3     train_index_list = input_df[input_df['fold'] != fold].index
      4     test_index_list = input_df[input_df['fold'] == fold].index
      5 

NameError: name 'test_df' is not defined
