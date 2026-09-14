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

3.10

# 2. Installed packages

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

# 3. Data file paths

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

# 4. Code solution

## === cell 0
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import sys
import subprocess

subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-q", "--no-deps", "protobuf<5"]
)

import gc
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint


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
[0;31m---------------------------------------------------------------------------[0m
[0;31mValueError[0m                                Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/1291752464.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m      1[0m [0;31m# Configuring TPU (https://www.kaggle.com/docs/tpu)[0m[0;34m[0m[0;34m[0m[0m
[1;32m      2[0m [0;31m# NOTE: will fail if the notebook doe not have Accelerator=TPU![0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 3[0;31m [0mtpu[0m [0;34m=[0m [0mtf[0m[0;34m.[0m[0mdistribute[0m[0;34m.[0m[0mcluster_resolver[0m[0;34m.[0m[0mTPUClusterResolver[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      4[0m [0mprint[0m[0;34m([0m[0;34m'Device:'[0m[0;34m,[0m [0mtpu[0m[0;34m.[0m[0mmaster[0m[0;34m([0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m      5[0m [0mtf[0m[0;34m.[0m[0mconfig[0m[0;34m.[0m[0mexperimental_connect_to_cluster[0m[0;34m([0m[0mtpu[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow/python/distribute/cluster_resolver/tpu/tpu_cluster_resolver.py[0m in [0;36m__init__[0;34m(self, tpu, zone, project, job_name, coordinator_name, coordinator_address, credentials, service, discovery_url)[0m
[1;32m    233[0m     [0;32mif[0m [0mtpu[0m [0;34m!=[0m [0;34m'local'[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    234[0m       [0;31m# Default Cloud environment[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 235[0;31m       self._cloud_tpu_client = client.Client(
[0m[1;32m    236[0m           [0mtpu[0m[0;34m=[0m[0mtpu[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m    237[0m           [0mzone[0m[0;34m=[0m[0mzone[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow/python/tpu/client/client.py[0m in [0;36m__init__[0;34m(self, tpu, zone, project, credentials, service, discovery_url)[0m
[1;32m    157[0m         [0mzone[0m [0;34m=[0m [0mzone[0m [0;32mor[0m [0mtpu_node_config[0m[0;34m.[0m[0mget[0m[0;34m([0m[0;34m'zone'[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    158[0m       [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 159[0;31m         [0;32mraise[0m [0mValueError[0m[0;34m([0m[0;34m'Please provide a TPU Name to connect to.'[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    160[0m [0;34m[0m[0m
[1;32m    161[0m     [0mself[0m[0;34m.[0m[0m_tpu[0m [0;34m=[0m [0m_as_text[0m[0;34m([0m[0mtpu[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;31mValueError[0m: Please provide a TPU Name to connect to.

## === cell 16
CURRENT_MODEL = 3
