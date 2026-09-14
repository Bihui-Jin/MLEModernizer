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

3.7

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
            GCP-Coupons-Instructions.rtf (486 Bytes)
            description.md (100 lines)
            labels.csv (55413943 lines)
            labels.csv.zip (1.6 GB)
            sample_submission.csv (9915 lines)
            sample_submission.csv.zip (76.2 kB)
            test.csv (9915 lines)
            test.csv.zip (273.0 kB)
            train.csv (55423857 lines)
            train.csv.zip (1.6 GB)
            new-york-city-taxi-fare-prediction/
                GCP-Coupons-Instructions.rtf (486 Bytes)
                description.md (100 lines)
                ... and 8 other files
                new-york-city-taxi-fare-prediction/
        input/
            GCP-Coupons-Instructions.rtf (486 Bytes)
            description.md (100 lines)
            labels.csv (55413943 lines)
            labels.csv.zip (1.6 GB)
            sample_submission.csv (9915 lines)
            sample_submission.csv.zip (76.2 kB)
            test.csv (9915 lines)
            test.csv.zip (273.0 kB)
            train.csv (55423857 lines)
            train.csv.zip (1.6 GB)
            new-york-city-taxi-fare-prediction/
                GCP-Coupons-Instructions.rtf (486 Bytes)
                description.md (100 lines)
                ... and 8 other files
                new-york-city-taxi-fare-prediction/
        working/
            new-york-city-taxi-fare-prediction/
                GCP-Coupons-Instructions.rtf (486 Bytes)
                description.md (100 lines)
                ... and 8 other files
                new-york-city-taxi-fare-prediction/
```

-> data/labels.csv has 55413942 rows and 8 columns.
The columns are: key, fare_amount, pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count

-> data/new-york-city-taxi-fare-prediction/labels.csv has 55413942 rows and 8 columns.
The columns are: key, fare_amount, pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count

-> data/new-york-city-taxi-fare-prediction/sample_submission.csv has 9914 rows and 2 columns.
The columns are: key, fare_amount

-> data/new-york-city-taxi-fare-prediction/test.csv has 9914 rows and 7 columns.
The columns are: key, pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count

-> data/new-york-city-taxi-fare-prediction/train.csv has 55423856 rows and 8 columns.
The columns are: key, fare_amount, pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count

-> data/sample_submission.csv has 9914 rows and 2 columns.
The columns are: key, fare_amount

-> data/test.csv has 9914 rows and 7 columns.
The columns are: key, pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count

-> data/train.csv has 55423856 rows and 8 columns.
The columns are: key, fare_amount, pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count

-> (stopped after 10 files for performance)

# 4. Code solution

## === cell 0
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
%matplotlib inline


## === cell 1
train_df = pd.read_csv('../input/train.csv', nrows = 1000000)


## === cell 2
train_df.shape


## === cell 3
test_df = pd.read_csv('../input/test.csv')


## === cell 4
test_df.shape


## === cell 5
train_df.head(5)


## === cell 6
train_df.isnull().sum()


## === cell 7
train_df.dropna(inplace=True)


## === cell 8
train_df.describe()


## === cell 9
train_df = train_df[train_df['fare_amount']>0]


## === cell 10
train_df.shape


## === cell 11
def distance(lat1, lon1, lat2, lon2):
    a = 0.5 - np.cos((lat2 - lat1) *  0.017453292519943295)/2 + np.cos(lat1 * 0.017453292519943295) * np.cos(lat2 * 0.017453292519943295) * (1 - np.cos((lon2 - lon1) *  0.017453292519943295)) / 2
    res = 0.6213712 * 12742 * np.arcsin(np.sqrt(a))
    return res


## === cell 12
train_df['distance'] = distance(train_df.pickup_latitude, train_df.pickup_longitude, \
                                      train_df.dropoff_latitude,train_df.dropoff_longitude)


## === cell 13
test_df['distance'] = distance(test_df.pickup_latitude, test_df.pickup_longitude, \
                                      test_df.dropoff_latitude,test_df.dropoff_longitude)


## === cell 14
train_df = train_df[train_df['distance']<15]


## === cell 15
train_df.describe()


## === cell 16
train_df = train_df[(train_df['passenger_count']!=0) & (train_df['passenger_count']<10)]


## === cell 18
feat_cols = ['distance','passenger_count']

X = train_df[feat_cols]
y = train_df['fare_amount']


## === cell 19
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)


## === cell 20
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

model_lin = Pipeline((
        ("standard_scaler", StandardScaler()),
        ("lin_reg", LinearRegression())
    ))
model_lin.fit(X_train, y_train)


## === cell 21
y_pred_final = model_lin.predict(test_df[feat_cols])

submission = pd.DataFrame(
    {'key': test_df.key, 'fare_amount': y_pred_final},
    columns = ['key', 'fare_amount'])
submission.to_csv('linear_reg.csv', index = False)


## === cell 22
import sys, subprocess

subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-q", "protobuf>=4.21.0,<5"]
)
import tensorflow as tf


## === cell 23
x_data = X
y_val = y


## === cell 24
from sklearn.model_selection import train_test_split


## === cell 25
X_train, X_test, y_train, y_test = train_test_split(x_data,y_val,test_size=0.1,random_state=101)


## === cell 26
from sklearn.preprocessing import MinMaxScaler


## === cell 27
scaler = MinMaxScaler()


## === cell 28
scaler.fit(X_train)


## === cell 29
X_train = pd.DataFrame(data=scaler.transform(X_train),columns = X_train.columns,index=X_train.index)


## === cell 30
X_test = pd.DataFrame(data=scaler.transform(X_test),columns = X_test.columns,index=X_test.index)


## === cell 31
passenger_count = tf.feature_column.numeric_column('passenger_count')
distance = tf.feature_column.numeric_column('distance')


## === cell 32
feat_cols = [ passenger_count,distance]


## === cell 33
import tensorflow as tf

import tensorflow_decision_forests as tfdf

tf.estimator = tfdf.keras.estimator

input_func = tf.estimator.inputs.pandas_input_fn(
    x=X_train, y=y_train, batch_size=10, num_epochs=1000, shuffle=True
)


## --- ERROR in cell 33, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mImportError[0m                               Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/3919927395.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m      5[0m [0;31m# Use TF-DF's compatible Estimator bridge to provide tf.estimator API expected[0m[0;34m[0m[0;34m[0m[0m
[1;32m      6[0m [0;31m# by the rest of the notebook (inputs.pandas_input_fn, DNNRegressor, etc.).[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 7[0;31m [0;32mimport[0m [0mtensorflow_decision_forests[0m [0;32mas[0m [0mtfdf[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      8[0m [0;34m[0m[0m
[1;32m      9[0m [0mtf[0m[0;34m.[0m[0mestimator[0m [0;34m=[0m [0mtfdf[0m[0;34m.[0m[0mkeras[0m[0;34m.[0m[0mestimator[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow_decision_forests/__init__.py[0m in [0;36m<module>[0;34m[0m
[1;32m     62[0m [0mcheck_version[0m[0;34m.[0m[0mcheck_version[0m[0;34m([0m[0m__version__[0m[0;34m,[0m [0mcompatible_tf_versions[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     63[0m [0;34m[0m[0m
[0;32m---> 64[0;31m [0;32mfrom[0m [0mtensorflow_decision_forests[0m [0;32mimport[0m [0mkeras[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     65[0m [0;32mfrom[0m [0mtensorflow_decision_forests[0m[0;34m.[0m[0mcomponent[0m [0;32mimport[0m [0mpy_tree[0m[0;34m[0m[0;34m[0m[0m
[1;32m     66[0m [0;32mfrom[0m [0mtensorflow_decision_forests[0m[0;34m.[0m[0mcomponent[0m[0;34m.[0m[0mbuilder[0m [0;32mimport[0m [0mbuilder[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow_decision_forests/keras/__init__.py[0m in [0;36m<module>[0;34m[0m
[1;32m     51[0m [0;32mfrom[0m [0mtyping[0m [0;32mimport[0m [0mCallable[0m[0;34m,[0m [0mList[0m[0;34m[0m[0;34m[0m[0m
[1;32m     52[0m [0;34m[0m[0m
[0;32m---> 53[0;31m [0;32mfrom[0m [0mtensorflow_decision_forests[0m[0;34m.[0m[0mkeras[0m [0;32mimport[0m [0mcore[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     54[0m [0;32mfrom[0m [0mtensorflow_decision_forests[0m[0;34m.[0m[0mkeras[0m [0;32mimport[0m [0mwrappers[0m[0;34m[0m[0;34m[0m[0m
[1;32m     55[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow_decision_forests/keras/core.py[0m in [0;36m<module>[0;34m[0m
[1;32m     60[0m [0;32mfrom[0m [0mtensorflow[0m[0;34m.[0m[0mpython[0m[0;34m.[0m[0mdata[0m[0;34m.[0m[0mops[0m [0;32mimport[0m [0mdataset_ops[0m[0;34m[0m[0;34m[0m[0m
[1;32m     61[0m [0;32mfrom[0m [0mtensorflow[0m[0;34m.[0m[0mpython[0m[0;34m.[0m[0mdata[0m[0;34m.[0m[0mops[0m [0;32mimport[0m [0mload_op[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 62[0;31m [0;32mfrom[0m [0mtensorflow_decision_forests[0m[0;34m.[0m[0mcomponent[0m[0;34m.[0m[0minspector[0m [0;32mimport[0m [0minspector[0m [0;32mas[0m [0minspector_lib[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     63[0m [0;32mfrom[0m [0mtensorflow_decision_forests[0m[0;34m.[0m[0mcomponent[0m[0;34m.[0m[0mtuner[0m [0;32mimport[0m [0mtuner[0m [0;32mas[0m [0mtuner_lib[0m[0;34m[0m[0;34m[0m[0m
[1;32m     64[0m [0;32mfrom[0m [0mtensorflow_decision_forests[0m[0;34m.[0m[0mkeras[0m [0;32mimport[0m [0mcore_inference[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow_decision_forests/component/inspector/inspector.py[0m in [0;36m<module>[0;34m[0m
[1;32m     62[0m [0;32mimport[0m [0mtensorflow[0m [0;32mas[0m [0mtf[0m[0;34m[0m[0;34m[0m[0m
[1;32m     63[0m [0;34m[0m[0m
[0;32m---> 64[0;31m [0;32mfrom[0m [0mtensorflow_decision_forests[0m[0;34m.[0m[0mcomponent[0m [0;32mimport[0m [0mpy_tree[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     65[0m [0;32mfrom[0m [0mtensorflow_decision_forests[0m[0;34m.[0m[0mcomponent[0m[0;34m.[0m[0minspector[0m [0;32mimport[0m [0mblob_sequence[0m[0;34m[0m[0;34m[0m[0m
[1;32m     66[0m [0;32mfrom[0m [0myggdrasil_decision_forests[0m[0;34m.[0m[0mdataset[0m [0;32mimport[0m [0mdata_spec_pb2[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow_decision_forests/component/py_tree/__init__.py[0m in [0;36m<module>[0;34m[0m
[1;32m     18[0m """
[1;32m     19[0m [0;34m[0m[0m
[0;32m---> 20[0;31m [0;32mfrom[0m [0mtensorflow_decision_forests[0m[0;34m.[0m[0mcomponent[0m[0;34m.[0m[0mpy_tree[0m [0;32mimport[0m [0mcondition[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     21[0m [0;32mfrom[0m [0mtensorflow_decision_forests[0m[0;34m.[0m[0mcomponent[0m[0;34m.[0m[0mpy_tree[0m [0;32mimport[0m [0mdataspec[0m[0;34m[0m[0;34m[0m[0m
[1;32m     22[0m [0;32mfrom[0m [0mtensorflow_decision_forests[0m[0;34m.[0m[0mcomponent[0m[0;34m.[0m[0mpy_tree[0m [0;32mimport[0m [0mnode[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow_decision_forests/component/py_tree/condition.py[0m in [0;36m<module>[0;34m[0m
[1;32m     24[0m [0;32mimport[0m [0msix[0m[0;34m[0m[0;34m[0m[0m
[1;32m     25[0m [0;34m[0m[0m
[0;32m---> 26[0;31m [0;32mfrom[0m [0mtensorflow_decision_forests[0m[0;34m.[0m[0mcomponent[0m[0;34m.[0m[0mpy_tree[0m [0;32mimport[0m [0mdataspec[0m [0;32mas[0m [0mdataspec_lib[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     27[0m [0;32mfrom[0m [0myggdrasil_decision_forests[0m[0;34m.[0m[0mdataset[0m [0;32mimport[0m [0mdata_spec_pb2[0m[0;34m[0m[0;34m[0m[0m
[1;32m     28[0m [0;32mfrom[0m [0myggdrasil_decision_forests[0m[0;34m.[0m[0mmodel[0m[0;34m.[0m[0mdecision_tree[0m [0;32mimport[0m [0mdecision_tree_pb2[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow_decision_forests/component/py_tree/dataspec.py[0m in [0;36m<module>[0;34m[0m
[1;32m     22[0m [0;32mfrom[0m [0mtyping[0m [0;32mimport[0m [0mNamedTuple[0m[0;34m,[0m [0mUnion[0m[0;34m,[0m [0mOptional[0m[0;34m,[0m [0mList[0m[0;34m[0m[0;34m[0m[0m
[1;32m     23[0m [0;34m[0m[0m
[0;32m---> 24[0;31m [0;32mfrom[0m [0myggdrasil_decision_forests[0m[0;34m.[0m[0mdataset[0m [0;32mimport[0m [0mdata_spec_pb2[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     25[0m [0;34m[0m[0m
[1;32m     26[0m [0mColumnType[0m [0;34m=[0m [0mdata_spec_pb2[0m[0;34m.[0m[0mColumnType[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/yggdrasil_decision_forests/dataset/data_spec_pb2.py[0m in [0;36m<module>[0;34m[0m
[1;32m      7[0m [0;32mfrom[0m [0mgoogle[0m[0;34m.[0m[0mprotobuf[0m [0;32mimport[0m [0mdescriptor[0m [0;32mas[0m [0m_descriptor[0m[0;34m[0m[0;34m[0m[0m
[1;32m      8[0m [0;32mfrom[0m [0mgoogle[0m[0;34m.[0m[0mprotobuf[0m [0;32mimport[0m [0mdescriptor_pool[0m [0;32mas[0m [0m_descriptor_pool[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 9[0;31m [0;32mfrom[0m [0mgoogle[0m[0;34m.[0m[0mprotobuf[0m [0;32mimport[0m [0mruntime_version[0m [0;32mas[0m [0m_runtime_version[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     10[0m [0;32mfrom[0m [0mgoogle[0m[0;34m.[0m[0mprotobuf[0m [0;32mimport[0m [0msymbol_database[0m [0;32mas[0m [0m_symbol_database[0m[0;34m[0m[0;34m[0m[0m
[1;32m     11[0m [0;32mfrom[0m [0mgoogle[0m[0;34m.[0m[0mprotobuf[0m[0;34m.[0m[0minternal[0m [0;32mimport[0m [0mbuilder[0m [0;32mas[0m [0m_builder[0m[0;34m[0m[0;34m[0m[0m

[0;31mImportError[0m: cannot import name 'runtime_version' from 'google.protobuf' (/usr/local/lib/python3.11/dist-packages/google/protobuf/__init__.py)

## === cell 34
model = tf.estimator.DNNRegressor(hidden_units=[100,100,100,100],feature_columns=feat_cols)
