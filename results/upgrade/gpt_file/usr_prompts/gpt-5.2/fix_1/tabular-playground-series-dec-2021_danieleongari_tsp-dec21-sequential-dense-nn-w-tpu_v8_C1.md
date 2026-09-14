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
Predict the class of a given image from a synthetic dataset.

## MetricMulti-class classification accuracy.

## Submission FormatFor each `Id` in the test set, you must predict the `Cover_Type` class. The file should contain a header and have the following format:
```
Id,Cover_Type
4000000,2
4000001,1
4000001,3
etc.
```

## Dataset 
- train.csv - the training data with the target `Cover_Type` column
- test.csv - the test set; you will be predicting the `Cover_Type` for each row in this file (the target integer class)
- sample_submission.csv - a sample submission file in the correct format

# 2. Python version

3.10

# 3. Installed packages

geopandas==0.14.4
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
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
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

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (59 lines)
            sample_submission.csv (400001 lines)
            sample_submission.csv.zip (1.6 MB)
            test.csv (400001 lines)
            test.csv.zip (10.7 MB)
            train.csv (3600001 lines)
            train.csv.zip (97.9 MB)
            tabular-playground-series-dec-2021/
                description.md (59 lines)
                sample_submission.csv (400001 lines)
                ... and 5 other files
                tabular-playground-series-dec-2021/
        input/
            description.md (59 lines)
            sample_submission.csv (400001 lines)
            sample_submission.csv.zip (1.6 MB)
            test.csv (400001 lines)
            test.csv.zip (10.7 MB)
            train.csv (3600001 lines)
            train.csv.zip (97.9 MB)
            tabular-playground-series-dec-2021/
                description.md (59 lines)
                sample_submission.csv (400001 lines)
                ... and 5 other files
                tabular-playground-series-dec-2021/
        working/
            tabular-playground-series-dec-2021/
                description.md (59 lines)
                sample_submission.csv (400001 lines)
                ... and 5 other files
                tabular-playground-series-dec-2021/
```

-> data/sample_submission.csv has 400000 rows and 2 columns.
The columns are: Id, Cover_Type

-> data/tabular-playground-series-dec-2021/sample_submission.csv has 400000 rows and 2 columns.
The columns are: Id, Cover_Type

-> data/tabular-playground-series-dec-2021/test.csv has 400000 rows and 55 columns.
The columns are: Id, Elevation, Aspect, Slope, Horizontal_Distance_To_Hydrology, Vertical_Distance_To_Hydrology, Horizontal_Distance_To_Roadways, Hillshade_9am, Hillshade_Noon, Hillshade_3pm, Horizontal_Distance_To_Fire_Points, Wilderness_Area1, Wilderness_Area2, Wilderness_Area3, Wilderness_Area4... and 40 more columns

-> data/tabular-playground-series-dec-2021/train.csv has 3600000 rows and 56 columns.
The columns are: Id, Elevation, Aspect, Slope, Horizontal_Distance_To_Hydrology, Vertical_Distance_To_Hydrology, Horizontal_Distance_To_Roadways, Hillshade_9am, Hillshade_Noon, Hillshade_3pm, Horizontal_Distance_To_Fire_Points, Wilderness_Area1, Wilderness_Area2, Wilderness_Area3, Wilderness_Area4... and 41 more columns

-> data/test.csv has 400000 rows and 55 columns.
The columns are: Id, Elevation, Aspect, Slope, Horizontal_Distance_To_Hydrology, Vertical_Distance_To_Hydrology, Horizontal_Distance_To_Roadways, Hillshade_9am, Hillshade_Noon, Hillshade_3pm, Horizontal_Distance_To_Fire_Points, Wilderness_Area1, Wilderness_Area2, Wilderness_Area3, Wilderness_Area4... and 40 more columns

-> data/train.csv has 3600000 rows and 56 columns.
The columns are: Id, Elevation, Aspect, Slope, Horizontal_Distance_To_Hydrology, Vertical_Distance_To_Hydrology, Horizontal_Distance_To_Roadways, Hillshade_9am, Hillshade_Noon, Hillshade_3pm, Horizontal_Distance_To_Fire_Points, Wilderness_Area1, Wilderness_Area2, Wilderness_Area3, Wilderness_Area4... and 41 more columns

-> input/sample_submission.csv has 400000 rows and 2 columns.
The columns are: Id, Cover_Type

-> (stopped after 10 files for performance)

# 5. Target score

0.95437

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import gc
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint


## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train = pd.read_csv('../input/tabular-playground-series-dec-2021/train.csv')
test = pd.read_csv('../input/tabular-playground-series-dec-2021/test.csv')
train


## === cell 2
train.describe().T


## === cell 3
test.describe().T


## === cell 4
display(train['Cover_Type'].value_counts().sort_index())


## === cell 5
for df in [train, test]:
    df = df.drop(columns = ['Id', 'Soil_Type7', 'Soil_Type15'])


## === cell 6
for df in [train, test]:
    df["Aspect_cos"] = np.cos(np.radians(df["Aspect"]))
    df["Aspect_sin"] = np.sin(np.radians(df["Aspect"]))
    df = df.drop(columns=["Aspect"])


## === cell 7
for df in train, test:
    for col in ["Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm"]:
        df[col] = df[col].clip(lower=0, upper=255)


## === cell 8
for df in [train, test]:
    df['Sum_Hydrology'] = np.abs(df['Horizontal_Distance_To_Hydrology']) + np.abs(df['Vertical_Distance_To_Hydrology'])
    df['Sub_Hydrology'] = np.abs(df['Horizontal_Distance_To_Hydrology']) - np.abs(df['Vertical_Distance_To_Hydrology'])


## === cell 9
for df in [train, test]:
    df['EHiElv'] = df['Horizontal_Distance_To_Roadways'] * df['Elevation']
    df['EViElv'] = df['Vertical_Distance_To_Hydrology'] * df['Elevation']
    df['Highwater'] = (df.Vertical_Distance_To_Hydrology < 0).astype(int)
    df['EVDtH'] = df.Elevation - df.Vertical_Distance_To_Hydrology
    df['EHDtH'] = df.Elevation - df.Horizontal_Distance_To_Hydrology * 0.2
    df['Euclidean_Distance_to_Hydrolody'] = (df['Horizontal_Distance_To_Hydrology']**2 + df['Vertical_Distance_To_Hydrology']**2)**0.5 # A bit redundant with Sum/Sub_Hydrology, but I keep it
    df['Manhattan_Distance_to_Hydrolody'] = df['Horizontal_Distance_To_Hydrology'] + df['Vertical_Distance_To_Hydrology']              # A bit redundant with Sum/Sub_Hydrology, but I keep it
    df['Hydro_Fire_1'] = df['Horizontal_Distance_To_Hydrology'] + df['Horizontal_Distance_To_Fire_Points']
    df['Hydro_Fire_2'] = abs(df['Horizontal_Distance_To_Hydrology'] - df['Horizontal_Distance_To_Fire_Points'])
    df['Hydro_Road_1'] = abs(df['Horizontal_Distance_To_Hydrology'] + df['Horizontal_Distance_To_Roadways'])
    df['Hydro_Road_2'] = abs(df['Horizontal_Distance_To_Hydrology'] - df['Horizontal_Distance_To_Roadways'])
    df['Fire_Road_1'] = abs(df['Horizontal_Distance_To_Fire_Points'] + df['Horizontal_Distance_To_Roadways'])
    df['Fire_Road_2'] = abs(df['Horizontal_Distance_To_Fire_Points'] - df['Horizontal_Distance_To_Roadways'])
    df['Hillshade_3pm_is_zero'] = (df.Hillshade_3pm == 0).astype(int)


## === cell 10
train = train.drop(index = train[train['Cover_Type'] == 5].index).reset_index(drop = True)
display(train['Cover_Type'].value_counts())


## === cell 11
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

le = LabelEncoder()
target = le.fit_transform(train['Cover_Type']) # REMEMBER: need to run `le.inverse_transform(test_pred)` at the end
target = to_categorical(target)                # REMEMBER: need to run `np.argmax(test_pred, axis = 1)` at the end

train = train.drop(columns = ['Cover_Type'])

gc.collect()


## === cell 12
from sklearn.preprocessing import RobustScaler
rb = RobustScaler()

cols = train.columns

train[cols] = rb.fit_transform(train[cols].values) # note: df[cols] is a trick to keep the df as a DataFrame for later (instead of an array)
test[cols] = rb.transform(test[cols].values)       # note: df[cols] is a trick to keep the df as a DataFrame for later (instead of an array)


## === cell 13
def reduce_mem_usage(df, verbose=True):
    """Make the dataframe lighter for the RAM: in this case by ca. 50%.
    """
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2

    for col in df.columns:
        col_type = df[col].dtypes

        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()

            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose:
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
 
    return df

train = reduce_mem_usage(train).values
test = reduce_mem_usage(test).values

gc.collect()


## === cell 14
shapes = {
    'nsamples': train.shape[0],
    'nfeatures': train.shape[1],
    'ncategories': target.shape[1]
}
print(shapes)


## === cell 15
tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
print('Device:', tpu.master())
tf.config.experimental_connect_to_cluster(tpu)
tf.tpu.experimental.initialize_tpu_system(tpu)
strategy = tf.distribute.experimental.TPUStrategy(tpu)
print('Number of replicas:', strategy.num_replicas_in_sync)


## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1291752464.py in <cell line: 0>()
      1 # Configuring TPU (https://www.kaggle.com/docs/tpu)
      2 # NOTE: will fail if the notebook doe not have Accelerator=TPU!
----> 3 tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
      4 print('Device:', tpu.master())
      5 tf.config.experimental_connect_to_cluster(tpu)

/usr/local/lib/python3.11/dist-packages/tensorflow/python/distribute/cluster_resolver/tpu/tpu_cluster_resolver.py in __init__(self, tpu, zone, project, job_name, coordinator_name, coordinator_address, credentials, service, discovery_url)
    233     if tpu != 'local':
    234       # Default Cloud environment
--> 235       self._cloud_tpu_client = client.Client(
    236           tpu=tpu,
    237           zone=zone,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/tpu/client/client.py in __init__(self, tpu, zone, project, credentials, service, discovery_url)
    157         zone = zone or tpu_node_config.get('zone')
    158       else:
--> 159         raise ValueError('Please provide a TPU Name to connect to.')
    160 
    161     self._tpu = _as_text(tpu)

ValueError: Please provide a TPU Name to connect to.

## === cell 16
CURRENT_MODEL = 2


## === cell 17
if CURRENT_MODEL==1:
    with strategy.scope(): # necessary for using the TPU
        model1 = keras.models.Sequential([
            keras.layers.Input((shapes['nfeatures'],)),
            keras.layers.Dense(200, activation='relu'),
            keras.layers.Dense(200, activation='relu'),
            keras.layers.Dense(200, activation='relu'),
            keras.layers.Dense(shapes['ncategories'], activation='softmax')
        ])
        display(keras.utils.plot_model(model1, show_shapes=True, show_dtype=True))
        model1.compile(
            loss='categorical_crossentropy',
            optimizer='Adam', 
            metrics=['accuracy'] # same as the competition: "Submissions are evaluated on multi-class classification accuracy."
        )
    


## === cell 18
if CURRENT_MODEL==1:
    earlystop = EarlyStopping(patience=3, restore_best_weights=True)
    model1.fit(
        x=train,
        y=target,
        epochs=20,
        batch_size=128 * strategy.num_replicas_in_sync,# https://www.kaggle.com/docs/tpu
        validation_split=0.2,
        callbacks=[earlystop]
    )


## === cell 19
if CURRENT_MODEL==1:
    pd.DataFrame(model1.history.history).plot(subplots=True, sharex=True, figsize=[15,8], grid=True)
    plt.show()


## === cell 20
if CURRENT_MODEL==2:
    dropout_rate = 0.1
    with strategy.scope(): # necessary for using the TPU
        model2 = keras.models.Sequential([
            keras.layers.Input((shapes['nfeatures'],)),
            keras.layers.Dense(200, kernel_initializer="he_normal", use_bias=False),
            keras.layers.BatchNormalization(),
            keras.layers.Activation("relu"),
            keras.layers.Dropout(rate=dropout_rate),
            keras.layers.Dense(200, kernel_initializer="he_normal", use_bias=False),
            keras.layers.BatchNormalization(),
            keras.layers.Activation("relu"),
            keras.layers.Dropout(rate=dropout_rate),
            keras.layers.Dense(200, kernel_initializer="he_normal", use_bias=False),
            keras.layers.BatchNormalization(),
            keras.layers.Activation("relu"),
            keras.layers.Dense(shapes['ncategories'], activation='softmax')
        ])
        display(keras.utils.plot_model(model2, show_shapes=True, show_dtype=True))
        model2.compile(
            loss='categorical_crossentropy',
            optimizer='Adam', 
            metrics=['accuracy'] # same as the competition: "Submissions are evaluated on multi-class classification accuracy."
        )


## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2071438796.py in <cell line: 0>()
      1 if CURRENT_MODEL==2:
      2     dropout_rate = 0.1
----> 3     with strategy.scope(): # necessary for using the TPU
      4         model2 = keras.models.Sequential([
      5             keras.layers.Input((shapes['nfeatures'],)),

NameError: name 'strategy' is not defined

## === cell 21
if CURRENT_MODEL==2:
    earlystop = EarlyStopping(patience=3, restore_best_weights=True) 
    model2.fit(
        x=train,
        y=target,
        epochs=10,
        batch_size=128 * strategy.num_replicas_in_sync,# https://www.kaggle.com/docs/tpu
        validation_split=0.00, # I want to use the more possible samples for training, just a little to monitor validation (I don't need it anymore for EarlyStopping)
    )


## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2338645647.py in <cell line: 0>()
      1 if CURRENT_MODEL==2:
      2     earlystop = EarlyStopping(patience=3, restore_best_weights=True)
----> 3     model2.fit(
      4         x=train,
      5         y=target,

NameError: name 'model2' is not defined

## === cell 22
if CURRENT_MODEL==2:
    pd.DataFrame(model2.history.history).plot(subplots=True, sharex=True, figsize=[15,8], grid=True)
    plt.show()


## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3610858992.py in <cell line: 0>()
      1 if CURRENT_MODEL==2:
----> 2     pd.DataFrame(model2.history.history).plot(subplots=True, sharex=True, figsize=[15,8], grid=True)
      3     plt.show()

NameError: name 'model2' is not defined

## === cell 23
if CURRENT_MODEL==1:
    test_pred = model1.predict(test, verbose=1)
elif CURRENT_MODEL==2:
    test_pred = model2.predict(test, verbose=1)


## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2322612285.py in <cell line: 0>()
      2     test_pred = model1.predict(test, verbose=1)
      3 elif CURRENT_MODEL==2:
----> 4     test_pred = model2.predict(test, verbose=1)

NameError: name 'model2' is not defined

## === cell 24
sub = pd.read_csv('../input/tabular-playground-series-dec-2021/sample_submission.csv')
sub['Cover_Type'] = le.inverse_transform(np.argmax(test_pred, axis = 1))
sub.to_csv('submission.csv', index=False)
display(sub)


## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4146097905.py in <cell line: 0>()
      1 sub = pd.read_csv('../input/tabular-playground-series-dec-2021/sample_submission.csv')
----> 2 sub['Cover_Type'] = le.inverse_transform(np.argmax(test_pred, axis = 1))
      3 sub.to_csv('submission.csv', index=False)
      4 display(sub)

NameError: name 'test_pred' is not defined
