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

6646327.0

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as bk
import tensorflow.keras.layers as ly
import tensorflow.keras.models as ml
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau
import tensorflow_addons as tfa
from sklearn.model_selection import train_test_split
import xgboost


## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

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
cb_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, min_lr=1e-7, patience=2, verbose=1, mode='min')
cb_early = EarlyStopping(monitor="val_loss", mode="min", restore_best_weights=True, patience= 5, verbose = 1)
model=create_my_model()
X_train1, X_val, y_train1, y_val = train_test_split(total_data, time_, test_size=0.1, random_state=3)
model.fit(X_train1,y_train1,batch_size=8,epochs=600,verbose=1,validation_data=(X_val,y_val),callbacks=[cb_lr,cb_early])


## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3754797525.py in <cell line: 0>()
      1 cb_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, min_lr=1e-7, patience=2, verbose=1, mode='min')
      2 cb_early = EarlyStopping(monitor="val_loss", mode="min", restore_best_weights=True, patience= 5, verbose = 1)
----> 3 model=create_my_model()
      4 X_train1, X_val, y_train1, y_val = train_test_split(total_data, time_, test_size=0.1, random_state=3)
      5 model.fit(X_train1,y_train1,batch_size=8,epochs=600,verbose=1,validation_data=(X_val,y_val),callbacks=[cb_lr,cb_early])

/tmp/ipykernel_11/802413106.py in create_my_model()
      1 def create_my_model():
      2     model = ml.Sequential()
----> 3     model.add(ly.Input(total_data.shape[1]))
      4     model.add(ly.BatchNormalization())
      5     model.add(tfa.layers.WeightNormalization(ly.Dense(1000,activation='relu')))

/usr/local/lib/python3.11/dist-packages/keras/src/layers/core/input_layer.py in Input(shape, batch_size, dtype, sparse, batch_shape, name, tensor, optional)
    189     ```
    190     """
--> 191     layer = InputLayer(
    192         shape=shape,
    193         batch_size=batch_size,

/usr/local/lib/python3.11/dist-packages/keras/src/layers/core/input_layer.py in __init__(self, shape, batch_size, dtype, sparse, batch_shape, input_tensor, optional, name, **kwargs)
     90 
     91             if shape is not None:
---> 92                 shape = backend.standardize_shape(shape)
     93                 batch_shape = (batch_size,) + shape
     94 

/usr/local/lib/python3.11/dist-packages/keras/src/backend/common/variables.py in standardize_shape(shape)
    560             raise ValueError("Undefined shapes are not supported.")
    561         if not hasattr(shape, "__iter__"):
--> 562             raise ValueError(f"Cannot convert '{shape}' to a shape.")
    563         if config.backend() == "tensorflow":
    564             if isinstance(shape, tf.TensorShape):

ValueError: Cannot convert '120' to a shape.

## === cell 6
sample_submission_df1=pd.read_csv('../input/predict-volcanic-eruptions-ingv-oe/sample_submission.csv')
sample_submission_df['time_to_eruption']=model.predict(total_data_test_)
sample_submission_df.to_csv('nn_res.csv',index=False)


## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1521150988.py in <cell line: 0>()
      1 sample_submission_df1=pd.read_csv('../input/predict-volcanic-eruptions-ingv-oe/sample_submission.csv')
----> 2 sample_submission_df['time_to_eruption']=model.predict(total_data_test_)
      3 sample_submission_df.to_csv('nn_res.csv',index=False)

NameError: name 'model' is not defined

## === cell 7
model1 = xgboost.XGBRegressor(n_estimators=100000,tree_method='gpu_hist',max_depth=8,learning_rate=0.05,alpha=0.1,SUBSAMPLE=0.6)
X_train1, X_val, y_train1, y_val = train_test_split(total_data, time_, test_size=0.1, random_state=3)
eval_set = [(X_val, y_val)]
model1.fit(X_train1, y_train1,early_stopping_rounds=5,eval_metric='mae', eval_set=eval_set, verbose=True)


## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/468542703.py in <cell line: 0>()
      1 #remove `tree_method` if you dont have gpu as processor
----> 2 model1 = xgboost.XGBRegressor(n_estimators=100000,tree_method='gpu_hist',max_depth=8,learning_rate=0.05,alpha=0.1,SUBSAMPLE=0.6)
      3 X_train1, X_val, y_train1, y_val = train_test_split(total_data, time_, test_size=0.1, random_state=3)
      4 eval_set = [(X_val, y_val)]
      5 model1.fit(X_train1, y_train1,early_stopping_rounds=5,eval_metric='mae', eval_set=eval_set, verbose=True)

NameError: name 'xgboost' is not defined

## === cell 8
sample_submission_df1=pd.read_csv('../input/predict-volcanic-eruptions-ingv-oe/sample_submission.csv')
sample_submission_df1['time_to_eruption']=model1.predict(total_data_test_)[:,None]
sample_submission_df1.to_csv('xgb_res.csv',index=False)


## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2337033851.py in <cell line: 0>()
      1 sample_submission_df1=pd.read_csv('../input/predict-volcanic-eruptions-ingv-oe/sample_submission.csv')
----> 2 sample_submission_df1['time_to_eruption']=model1.predict(total_data_test_)[:,None]
      3 sample_submission_df1.to_csv('xgb_res.csv',index=False)

NameError: name 'model1' is not defined
