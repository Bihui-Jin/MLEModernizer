# Goal

You will receive environment details and a partial notebook export.

# Requirements

- Fix the bug that causes the error in cell k.
- Do NOT adjust any other non-buggy cells.
- You may reference cell k+1 only to preserve variable/interface compatibility.
- Do not complete or extend code logic in cell k, k+1, or later cells.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (bug fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Output must follow your strict format: Diagnosis / Patch summary / Updated cells / Compatibility notes for cell k+1 / Assumptions.


# 1. Python version

3.9

# 2. Installed packages

geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1
xgboost==2.0.3

# 3. Data file paths

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

# 4. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

import numpy as np
import pandas as pd

tf = None
bk = None
ly = None
ml = None
EarlyStopping = None
ReduceLROnPlateau = None

tfa = None

from sklearn.model_selection import train_test_split
import xgboost


## === cell 1
%%time
train_df=pd.read_csv('/kaggle/input/predict-volcanic-eruptions-ingv-oe/train.csv')
n_f=12
total_data=np.empty((train_df.shape[0],n_f*10))
time_=np.empty((train_df.shape[0],1))
for i_,seg_ in enumerate(train_df['segment_id']):
    the_df=pd.read_csv(f'/kaggle/input/predict-volcanic-eruptions-ingv-oe/train/{seg_}.csv').fillna(0)
    total_data[i_,:]=np.concatenate((the_df.abs().mean().to_numpy(),
                                    the_df.std().to_numpy(),
                                    the_df.mean().to_numpy(),
                                    the_df.var().to_numpy(),
                                    the_df.min().to_numpy(),
                                    the_df.max().to_numpy(),
                                    the_df.median().to_numpy(),
                                    the_df.quantile([0.1,0.25,0.5,0.75,0.9]).to_numpy().reshape(1,-1)[0]))
    time_[i_,0]=train_df.loc[i_,'time_to_eruption']


## === cell 2
%%time
sample_submission_df=pd.read_csv('../input/predict-volcanic-eruptions-ingv-oe/sample_submission.csv')
n_f=12
total_data_test_=np.empty((sample_submission_df.shape[0],n_f*10))
for i_,seg_ in enumerate(sample_submission_df['segment_id']):
    the_df=pd.read_csv(f'/kaggle/input/predict-volcanic-eruptions-ingv-oe/test/{seg_}.csv').fillna(0)
    total_data_test_[i_,:]=np.concatenate((the_df.abs().mean().to_numpy(),
                                    the_df.std().to_numpy(),
                                    the_df.mean().to_numpy(),
                                    the_df.var().to_numpy(),
                                    the_df.min().to_numpy(),
                                    the_df.max().to_numpy(),
                                    the_df.median().to_numpy(),
                                    the_df.quantile([0.1,0.25,0.5,0.75,0.9]).to_numpy().reshape(1,-1)[0]))


## === cell 3
del the_df


## === cell 4
def create_my_model():
    model = ml.Sequential()
    model.add(ly.Input(total_data.shape[1]))
    model.add(ly.BatchNormalization())
    model.add(tfa.layers.WeightNormalization(ly.Dense(1000,activation='relu')))
    model.add(ly.BatchNormalization())
    model.add(ly.Dropout(0.7))
    model.add(tfa.layers.WeightNormalization(ly.Dense(1,activation='relu')))


    model.compile(optimizer=tfa.optimizers.AdamW(lr = 1, weight_decay = 1e-5, clipvalue = 900),loss='mean_absolute_error')
    return model


## === cell 5
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
os.environ["PROTOCOL_BUFFERS_PYTHON_DISABLE_CXX_IMPLEMENTATION"] = "1"

import tensorflow as tf
from tensorflow.keras import backend as bk
from tensorflow.keras import layers as ly
from tensorflow.keras import models as ml
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import types


class _IdentityWeightNormalization(ly.Wrapper):
    def __init__(self, layer, **kwargs):
        super().__init__(layer, **kwargs)

    def call(self, inputs, training=None):
        return self.layer(inputs, training=training)


tfa = types.SimpleNamespace(
    layers=types.SimpleNamespace(WeightNormalization=_IdentityWeightNormalization),
    optimizers=types.SimpleNamespace(AdamW=tf.keras.optimizers.AdamW),
)

cb_lr = ReduceLROnPlateau(
    monitor="val_loss", factor=0.1, min_lr=1e-7, patience=2, verbose=1, mode="min"
)
cb_early = EarlyStopping(
    monitor="val_loss", mode="min", restore_best_weights=True, patience=5, verbose=1
)
model = create_my_model()
X_train1, X_val, y_train1, y_val = train_test_split(
    total_data, time_, test_size=0.1, random_state=3
)
model.fit(
    X_train1,
    y_train1,
    batch_size=8,
    epochs=600,
    verbose=1,
    validation_data=(X_val, y_val),
    callbacks=[cb_lr, cb_early],
)


## --- ERROR in cell 5, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mAttributeError[0m                            Traceback (most recent call last)
[0;31mAttributeError[0m: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 6
sample_submission_df1=pd.read_csv('../input/predict-volcanic-eruptions-ingv-oe/sample_submission.csv')
sample_submission_df['time_to_eruption']=model.predict(total_data_test_)
sample_submission_df.to_csv('nn_res.csv',index=False)
