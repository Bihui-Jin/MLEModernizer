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
Predict a neutrino particle's direction. 

## Metric
Mean angular error between the predicted and true event origins.

## Submission Format
For each `event_id` in the test set, you must predict the `azimuth` and `zenith`. The file should contain a header and have the following format:

```
event_id,azimuth,zenith
730,1,1
769,1,1
774,1,1
etc.
```

## Dataset 
[train/test]_meta.parquet

-   `batch_id` (`int`): the ID of the batch the event was placed into.
-   `event_id` (`int`): the event ID.
-   `[first/last]_pulse_index` (`int`): index of the first/last row in the features dataframe belonging to this event.
-   `[azimuth/zenith]` (`float32`): the [azimuth/zenith] angle in radians of the neutrino. A value between 0 and 2*pi for the azimuth and 0 and pi for zenith. The target columns. Not provided for the test set. The direction vector represented by zenith and azimuth points to where the neutrino came from.
-   NB: Other quantities regarding the event, such as the interaction point in `x, y, z` (vertex position), the neutrino energy, or the interaction type and kinematics are not included in the dataset.

[train/test]/batch_[n].parquet Each batch contains tens of thousands of events. Each event may contain thousands of pulses, each of which is the digitized output from a photomultiplier tube and occupies one row.

-   `event_id` (`int`): the event ID. Saved as the index column in parquet.
-   `time` (`int`): the time of the pulse in nanoseconds in the current event time window. The absolute time of a pulse has no relevance, and only the relative time with respect to other pulses within an event is of relevance.
-   `sensor_id` (`int`): the ID of which of the 5160 IceCube photomultiplier sensors recorded this pulse.
-   `charge` (`float32`): An estimate of the amount of light in the pulse, in units of photoelectrons (p.e.). A physical photon does not exactly result in a measurement of 1 p.e. but rather can take values spread around 1 p.e. As an example, a pulse with charge 2.7 p.e. could quite likely be the result of two or three photons hitting the photomultiplier tube around the same time. This data has `float16` precision but is stored as `float32` due to limitations of the version of pyarrow the data was prepared with.
-   `auxiliary` (`bool`): If `True`, the pulse was not fully digitized, is of lower quality, and was more likely to originate from noise. If `False`, then this pulse was contributed to the trigger decision and the pulse was fully digitized.

sample_submission.parquet An example submission with the correct columns and properly ordered event IDs. The sample submission is provided in the parquet format so it can be read quickly but *your final submission must be a csv*.

`sensor_geometry.csv` The `x`, `y`, and `z` positions for each of the 5160 IceCube sensors. The row index corresponds to the `sensor_idx` feature of pulses. The `x`, `y`, and `z` coordinates are in units of meters, with the origin at the center of the IceCube detector. The coordinate system is right-handed, and the z-axis points upwards when standing at the South Pole. You can convert from these coordinates to `azimuth` and `zenith` with the following formulas (here the vector (x,y,z) is normalized):

```
x = cos(azimuth) * sin(zenith)
y = sin(azimuth) * sin(zenith)
z = cos(zenith)

```

# 2. Python version

3.11

# 3. Installed packages

geopandas==0.14.4
lightgbm==4.6.0
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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
tqdm==4.67.1
xgboost==2.0.3

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (231 lines)
            sample_submission.csv (13200001 lines)
            sample_submission.csv.zip (35.3 MB)
            sensor_geometry.csv (5161 lines)
            sensor_geometry.csv.zip (36.0 kB)
            test.zip (9.3 GB)
            test_meta.parquet (172.5 MB)
            train.zip (83.9 GB)
            train_meta.parquet (3.5 GB)
            icecube-neutrinos-in-deep-ice/
                description.md (231 lines)
                sample_submission.csv (13200001 lines)
                ... and 7 other files
                icecube-neutrinos-in-deep-ice/
                test/
                    batch_104.parquet (172.9 MB)
                    batch_128.parquet (172.7 MB)
                    ... and 64 other files
                    test/
                train/
                    batch_1.parquet (172.1 MB)
                    batch_10.parquet (173.4 MB)
                    ... and 592 other files
                    train/
            test/
                batch_104.parquet (172.9 MB)
                batch_128.parquet (172.7 MB)
                ... and 64 other files
                test/
            train/
                batch_1.parquet (172.1 MB)
                batch_10.parquet (173.4 MB)
                ... and 592 other files
                train/
        input/
            description.md (231 lines)
            sample_submission.csv (13200001 lines)
            sample_submission.csv.zip (35.3 MB)
            sensor_geometry.csv (5161 lines)
            sensor_geometry.csv.zip (36.0 kB)
            test.zip (9.3 GB)
            test_meta.parquet (172.5 MB)
            train.zip (83.9 GB)
            train_meta.parquet (3.5 GB)
            icecube-neutrinos-in-deep-ice/
                description.md (231 lines)
                sample_submission.csv (13200001 lines)
                ... and 7 other files
                icecube-neutrinos-in-deep-ice/
                test/
                    batch_104.parquet (172.9 MB)
                    batch_128.parquet (172.7 MB)
                    ... and 64 other files
                    test/
                train/
                    batch_1.parquet (172.1 MB)
                    batch_10.parquet (173.4 MB)
                    ... and 592 other files
                    train/
            test/
                batch_104.parquet (172.9 MB)
                batch_128.parquet (172.7 MB)
                ... and 64 other files
                test/
                    batch_104.parquet (172.9 MB)
                    batch_128.parquet (172.7 MB)
                    ... and 64 other files
                    test/
            train/
                batch_1.parquet (172.1 MB)
                batch_10.parquet (173.4 MB)
                ... and 592 other files
                train/
                    batch_1.parquet (172.1 MB)
                    batch_10.parquet (173.4 MB)
                    ... and 592 other files
                    train/
        working/
            icecube-neutrinos-in-deep-ice/
                description.md (231 lines)
                sample_submission.csv (13200001 lines)
                ... and 7 other files
                icecube-neutrinos-in-deep-ice/
                test/
                    batch_104.parquet (172.9 MB)
                    batch_128.parquet (172.7 MB)
                    ... and 64 other files
                    test/
                train/
                    batch_1.parquet (172.1 MB)
                    batch_10.parquet (173.4 MB)
                    ... and 592 other files
                    train/
```

-> data/icecube-neutrinos-in-deep-ice/sample_submission.csv has 13200000 rows and 3 columns.
The columns are: event_id, azimuth, zenith

-> data/icecube-neutrinos-in-deep-ice/sensor_geometry.csv has 5160 rows and 4 columns.
The columns are: sensor_id, x, y, z

-> data/sample_submission.csv has 13200000 rows and 3 columns.
The columns are: event_id, azimuth, zenith

-> data/sensor_geometry.csv has 5160 rows and 4 columns.
The columns are: sensor_id, x, y, z

-> input/icecube-neutrinos-in-deep-ice/sample_submission.csv has 13200000 rows and 3 columns.
The columns are: event_id, azimuth, zenith

-> input/icecube-neutrinos-in-deep-ice/sensor_geometry.csv has 5160 rows and 4 columns.
The columns are: sensor_id, x, y, z

-> (stopped after 10 files for performance)

# 5. Target score

1.568163

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
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
%matplotlib inline

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PATH_DATASET = "/kaggle/input/icecube-neutrinos-in-deep-ice"
meta_test = pd.read_parquet(os.path.join(PATH_DATASET, "test_meta.parquet"))
del meta_test


## === cell 2
meta_train = pd.read_parquet(os.path.join(PATH_DATASET, "train_meta.parquet"))
meta_train.head()


## === cell 3
from tqdm.auto import tqdm
def transform_batch(path_batch_parquet, nb_sensors=5160):
    df = pd.read_parquet(path_batch_parquet)
    data = np.zeros((len(df.index.unique()), nb_sensors), dtype=np.float16)
    event_ids = []
    for i, (idx, dfg) in tqdm(enumerate(df.groupby(level=0))):
        event_ids.append(idx)
        data[i, dfg['sensor_id']] = dfg['charge']
    return event_ids, data


## === cell 4
event_ids, data = transform_batch(os.path.join(PATH_DATASET, "train/batch_10.parquet"))
print(f"data size: {data.shape}")
meta_train_ = meta_train[meta_train['event_id'].isin(event_ids)]
meta_train_ = dict(zip(meta_train_['event_id'].values, meta_train_[["azimuth", "zenith"]].values.tolist()))
print(f"LUT size: {len(meta_train_)}")
angles = np.array([meta_train_[eid] for eid in event_ids], dtype=np.float16)
print(f"angles size: {angles.shape}")


## === cell 5
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data, angles, train_size=0.8)
del data, angles


## === cell 6
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
preprocess = Pipeline([
    ('scaler', StandardScaler()),
     ("PCA", PCA(
        n_components=510,
        copy=False,
    ))])
X_train = preprocess.fit_transform(X_train)
X_test = preprocess.transform(X_test)


## === cell 8
from lightgbm import LGBMRegressor


## === cell 9
params_lgb={'colsample_bytree': 0.8825309754631472, 
            'learning_rate': 0.13432792418150802, 
            'max_depth': 4, 
            'n_estimators': 165,
            'num_leaves': 15, 
            'reg_alpha': 0.676985054081822, 
            'reg_lambda': 580.8583113057583, 'subsample': 0.9447267339812311}
lgb=LGBMRegressor(**params_lgb)


## === cell 10
lgb.fit(
    X_train, y_train[:,0],
    eval_set=[(X_test, y_test[:,0])], 
    eval_metric=["rmse"],
    verbose=True,
    early_stopping_rounds=5
)


## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/816369907.py in <cell line: 0>()
----> 1 lgb.fit(
      2     X_train, y_train[:,0],
      3     eval_set=[(X_test, y_test[:,0])],
      4     eval_metric=["rmse"],
      5     verbose=True,

TypeError: LGBMRegressor.fit() got an unexpected keyword argument 'verbose'

## === cell 11
lgb1=LGBMRegressor(**params_lgb)
lgb1.fit(
    X_train, y_train[:,1],
    eval_set=[(X_test, y_test[:,1])], 
    eval_metric=["rmse"],
    verbose=True,
    early_stopping_rounds=5
)


## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/860481941.py in <cell line: 0>()
      1 lgb1=LGBMRegressor(**params_lgb)
----> 2 lgb1.fit(
      3     X_train, y_train[:,1],
      4     eval_set=[(X_test, y_test[:,1])],
      5     eval_metric=["rmse"],

TypeError: LGBMRegressor.fit() got an unexpected keyword argument 'verbose'

## === cell 12
import gc, time
gc.collect()
time.sleep(9)
ssub = pd.read_parquet(os.path.join(PATH_DATASET, "sample_submission.parquet"))
ssub.set_index("event_id", inplace=True)
ssub = ssub.apply(np.float16)


## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/2783220334.py in <cell line: 0>()
      2 gc.collect()
      3 time.sleep(9)
----> 4 ssub = pd.read_parquet(os.path.join(PATH_DATASET, "sample_submission.parquet"))
      5 ssub.set_index("event_id", inplace=True)
      6 ssub = ssub.apply(np.float16)

/usr/local/lib/python3.11/dist-packages/pandas/io/parquet.py in read_parquet(path, engine, columns, storage_options, use_nullable_dtypes, dtype_backend, filesystem, filters, **kwargs)
    665     check_dtype_backend(dtype_backend)
    666 
--> 667     return impl.read(
    668         path,
    669         columns=columns,

/usr/local/lib/python3.11/dist-packages/pandas/io/parquet.py in read(self, path, columns, filters, use_nullable_dtypes, dtype_backend, storage_options, filesystem, **kwargs)
    265             to_pandas_kwargs["split_blocks"] = True  # type: ignore[assignment]
    266 
--> 267         path_or_handle, handles, filesystem = _get_path_or_handle(
    268             path,
    269             filesystem,

/usr/local/lib/python3.11/dist-packages/pandas/io/parquet.py in _get_path_or_handle(path, fs, storage_options, mode, is_dir)
    138         # fsspec resources can also point to directories
    139         # this branch is used for example when reading from non-fsspec URLs
--> 140         handles = get_handle(
    141             path_or_handle, mode, is_text=False, storage_options=storage_options
    142         )

/usr/local/lib/python3.11/dist-packages/pandas/io/common.py in get_handle(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)
    880         else:
    881             # Binary mode
--> 882             handle = open(handle, ioargs.mode)
    883         handles.append(handle)
    884 

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/input/icecube-neutrinos-in-deep-ice/sample_submission.parquet'

## === cell 13
ssub['zenith'] = meta_train['zenith'].mean()


## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1678292486.py in <cell line: 0>()
----> 1 ssub['zenith'] = meta_train['zenith'].mean()

NameError: name 'ssub' is not defined

## === cell 14
ls = glob.glob(os.path.join(PATH_DATASET, "test", "*.parquet"))

for batch_file in ls:
    print(f"processing: {batch_file}")
    event_ids, data = transform_batch(batch_file)
    preds=np.array(list(zip(lgb.predict(preprocess.transform(data)),
                            lgb1.predict(preprocess.transform(data)))))
    del data
    for eid, (a, z) in zip(event_ids, preds):
        ssub.at[eid, "azimuth"] = a
        ssub.at[eid, "zenith"] = z
    del event_ids, preds
    gc.collect()
    time.sleep(9)


## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NotFittedError                            Traceback (most recent call last)
/tmp/ipykernel_11/473430933.py in <cell line: 0>()
      4     print(f"processing: {batch_file}")
      5     event_ids, data = transform_batch(batch_file)
----> 6     preds=np.array(list(zip(lgb.predict(preprocess.transform(data)),
      7                             lgb1.predict(preprocess.transform(data)))))
      8     del data

/usr/local/lib/python3.11/dist-packages/lightgbm/sklearn.py in predict(self, X, raw_score, start_iteration, num_iteration, pred_leaf, pred_contrib, validate_features, **kwargs)
   1104         """Docstring is set after definition, using a template."""
   1105         if not self.__sklearn_is_fitted__():
-> 1106             raise LGBMNotFittedError("Estimator not fitted, call fit before exploiting the model.")
   1107         if not isinstance(X, (pd_DataFrame, dt_DataTable)):
   1108             X = _LGBMValidateData(

NotFittedError: Estimator not fitted, call fit before exploiting the model.

## === cell 15
ssub.to_csv('submission.csv', index=True)

!head submission.csv


## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/739921399.py in <cell line: 0>()
----> 1 ssub.to_csv('submission.csv', index=True)
      2 
      3 get_ipython().system('head submission.csv')

NameError: name 'ssub' is not defined
