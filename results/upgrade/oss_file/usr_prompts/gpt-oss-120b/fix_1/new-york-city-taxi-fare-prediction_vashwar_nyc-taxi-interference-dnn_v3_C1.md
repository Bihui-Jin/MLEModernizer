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
Predict the fare amount for a taxi ride given the pickup and dropoff locations.

## Metric
Root mean-squared error.

## Submission Format
For each `key` in the test set, you must predict a value for the `fare_amount` variable. The file should contain a header and have the following format:

```
key,fare_amount
2015-01-27 13:08:24.0000002,11.00
2015-02-27 13:08:24.0000002,12.05
2015-03-27 13:08:24.0000002,11.23
2015-04-27 13:08:24.0000002,14.17
2015-05-27 13:08:24.0000002,15.12
etc
```

## Dataset
- **train.csv** - Input features and target `fare_amount` values for the training set (about 55M rows).
- **test.csv** - Input features for the test set (about 10K rows). Your goal is to predict `fare_amount` for each row.
- **sample_submission.csv** - a sample submission file in the correct format (columns `key` and `fare_amount`). This file 'predicts' `fare_amount` to be $`11.35` for all rows, which is the mean `fare_amount` from the training set.

### Data fields
**ID**

- **key** - Unique `string` identifying each row in both the training and test sets. Comprised of **pickup_datetime** plus a unique integer, but this doesn't matter, it should just be used as a unique ID field.Required in your submission CSV. Not necessarily needed in the training set, but could be useful to simulate a 'submission file' while doing cross-validation within the training set.

**Features**

- **pickup_datetime** - `timestamp` value indicating when the taxi ride started.
- **pickup_longitude** - `float` for longitude coordinate of where the taxi ride started.
- **pickup_latitude** - `float` for latitude coordinate of where the taxi ride started.
- **dropoff_longitude** - `float` for longitude coordinate of where the taxi ride ended.
- **dropoff_latitude** - `float` for latitude coordinate of where the taxi ride ended.
- **passenger_count** - `integer` indicating the number of passengers in the taxi ride.

**Target**

- **fare_amount** - `float` dollar amount of the cost of the taxi ride. This value is only in the training set; this is what you are predicting in the test set and it is required in your submission CSV.

# 2. Python version

3.7

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

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

# 5. Target score

3.6375985209102257

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 1

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re
import tensorflow as tf
import os
print(os.listdir("../input"))



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
os.listdir("../input/dnn-model")

## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/507771417.py in <cell line: 0>()
----> 1 os.listdir("../input/dnn-model")

FileNotFoundError: [Errno 2] No such file or directory: '../input/dnn-model'

## === cell 3
df_test=pd.read_csv("../input/new-york-city-taxi-fare-prediction/test.csv")

## === cell 4
df_test.head()

## === cell 6
def add_datepart(df, fldname, drop=True):
    fld = df[fldname]
    if not np.issubdtype(fld.dtype, np.datetime64):
        df[fldname] = fld = pd.to_datetime(fld, infer_datetime_format=True)
    targ_pre = re.sub('[Dd]ate$', '', fldname)
    for n in ('Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear','hour',
            'Is_month_end', 'Is_month_start', 'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start'):
        df[targ_pre+n] = getattr(fld.dt,n.lower())
        
    df[targ_pre+'Elapsed'] = fld.astype(np.int64) // 10**9
    if drop: df.drop(fldname, axis=1, inplace=True)
        
def distance( data):

    radius = 6371 # km
    lon1=data[:,0]
    lat1=data[:,1]
    lon2=data[:,2]
    lat2=data[:,3]
    dlat = np.radians(lat2-lat1)
    dlon = np.radians(lon2-lon1)
    a = np.sin(dlat/2) * np.sin(dlat/2) + np.cos(np.radians(lat1)) \
            * np.cos(np.radians(lat2)) * np.sin(dlon/2) * np.sin(dlon/2)
    c = 2 * np.arctan(np.sqrt(a), np.sqrt(1-a))
    d = radius * c

    return d

## === cell 7
he_init = tf.contrib.layers.variance_scaling_initializer()

def dnn(inputs,training, n_hidden_layers=8, name=None,
        activation=tf.nn.relu, initializer=he_init):
    n_neurons=[2000,1000,500,250,125,50,25,10]
 
    with tf.variable_scope(name, "dnn"):
        for layer in range(n_hidden_layers):
            
            
       
            inputs = tf.layers.dense(inputs, n_neurons[layer],
                                     kernel_initializer=initializer,
                                     name="hidden%d" % (layer + 1))
        

            inputs=tf.nn.relu(inputs, name="hidden%d_out" % (layer + 1))
        
        return inputs

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/4106984658.py in <cell line: 0>()
----> 1 he_init = tf.contrib.layers.variance_scaling_initializer()
      2 
      3 def dnn(inputs,training, n_hidden_layers=8, name=None,
      4         activation=tf.nn.relu, initializer=he_init):
      5     n_neurons=[2000,1000,500,250,125,50,25,10]

AttributeError: module 'tensorflow' has no attribute 'contrib'

## === cell 8
df_test['Herv_Dist'] =distance(np.float64(df_test.values[:,2:6]))

## === cell 9
add_datepart(df_test, 'pickup_datetime', drop=True)

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/3039125103.py in <cell line: 0>()
----> 1 add_datepart(df_test, 'pickup_datetime', drop=True)

/tmp/ipykernel_11/117838631.py in add_datepart(df, fldname, drop)
      7     for n in ('Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear','hour',
      8             'Is_month_end', 'Is_month_start', 'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start'):
----> 9         df[targ_pre+n] = getattr(fld.dt,n.lower())
     10 
     11     df[targ_pre+'Elapsed'] = fld.astype(np.int64) // 10**9

AttributeError: 'DatetimeProperties' object has no attribute 'week'

## === cell 10
feature_cols=['pickup_longitude', 'pickup_latitude', 'dropoff_longitude',
       'dropoff_latitude', 'passenger_count', 'pickup_datetimeYear',
       'pickup_datetimeMonth', 'pickup_datetimeWeek', 'pickup_datetimeDay',
       'pickup_datetimeDayofweek', 'pickup_datetimeDayofyear',
       'pickup_datetimehour', 'pickup_datetimeElapsed', 'Herv_Dist']

## === cell 11
df_test[feature_cols].head()

## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/3372810872.py in <cell line: 0>()
----> 1 df_test[feature_cols].head()

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4106             if is_iterator(key):
   4107                 key = list(key)
-> 4108             indexer = self.columns._get_indexer_strict(key, "columns")[1]
   4109 
   4110         # take() does not accept boolean indexers

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _get_indexer_strict(self, key, axis_name)
   6198             keyarr, indexer, new_indexer = self._reindex_non_unique(keyarr)
   6199 
-> 6200         self._raise_if_missing(keyarr, indexer, axis_name)
   6201 
   6202         keyarr = self.take(indexer)

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _raise_if_missing(self, key, indexer, axis_name)
   6250 
   6251             not_found = list(ensure_index(key)[missing_mask.nonzero()[0]].unique())
-> 6252             raise KeyError(f"{not_found} not in index")
   6253 
   6254     @overload

KeyError: "['pickup_datetimeWeek', 'pickup_datetimeDay', 'pickup_datetimeDayofweek', 'pickup_datetimeDayofyear', 'pickup_datetimehour', 'pickup_datetimeElapsed'] not in index"

## === cell 12
mu=np.array([ -7.39752352e+01,   4.07510864e+01,  -7.39743620e+01,
         4.07514412e+01,   1.69111912e+00,   2.01173779e+03,
         6.26937910e+00,   2.54649417e+01,   1.57119467e+01,
         3.04109087e+00,   1.75307310e+02,   1.35101716e+01,
         1.33224990e+09,   3.34143936e+00])
sigma=np.array([  4.26467712e-02,   3.18110081e-02,   4.13962939e-02,
         3.48417371e-02,   1.30694141e+00,   1.86550121e+00,
         3.43641982e+00,   1.49473195e+01,   8.68516050e+00,
         1.94912410e+00,   1.04798866e+02,   6.51677611e+00,
         5.84916113e+07,   4.08371701e+00])

## === cell 13
x_test_unscl=df_test[feature_cols].values

## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/4073259518.py in <cell line: 0>()
----> 1 x_test_unscl=df_test[feature_cols].values

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4106             if is_iterator(key):
   4107                 key = list(key)
-> 4108             indexer = self.columns._get_indexer_strict(key, "columns")[1]
   4109 
   4110         # take() does not accept boolean indexers

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _get_indexer_strict(self, key, axis_name)
   6198             keyarr, indexer, new_indexer = self._reindex_non_unique(keyarr)
   6199 
-> 6200         self._raise_if_missing(keyarr, indexer, axis_name)
   6201 
   6202         keyarr = self.take(indexer)

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _raise_if_missing(self, key, indexer, axis_name)
   6250 
   6251             not_found = list(ensure_index(key)[missing_mask.nonzero()[0]].unique())
-> 6252             raise KeyError(f"{not_found} not in index")
   6253 
   6254     @overload

KeyError: "['pickup_datetimeWeek', 'pickup_datetimeDay', 'pickup_datetimeDayofweek', 'pickup_datetimeDayofyear', 'pickup_datetimehour', 'pickup_datetimeElapsed'] not in index"

## === cell 14
x_test=(x_test_unscl-mu)/sigma

## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2611217.py in <cell line: 0>()
----> 1 x_test=(x_test_unscl-mu)/sigma

NameError: name 'x_test_unscl' is not defined

## === cell 16
tf.reset_default_graph()
n_inputs=14
n_outputs=1

## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/2689729292.py in <cell line: 0>()
----> 1 tf.reset_default_graph()
      2 n_inputs=14
      3 n_outputs=1

AttributeError: module 'tensorflow' has no attribute 'reset_default_graph'

## === cell 17
X=tf.placeholder(tf.float32,shape=(None,n_inputs),name="X")
y=tf.placeholder(tf.float32,shape=(None,1),name='y')
training=tf.placeholder_with_default(False,shape=(),name='training')

## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/1251178249.py in <cell line: 0>()
----> 1 X=tf.placeholder(tf.float32,shape=(None,n_inputs),name="X")
      2 y=tf.placeholder(tf.float32,shape=(None,1),name='y')
      3 training=tf.placeholder_with_default(False,shape=(),name='training')

AttributeError: module 'tensorflow' has no attribute 'placeholder'

## === cell 18
with tf.name_scope("dnn"):
    y_pred=tf.layers.dense(dnn(X,training),n_outputs,name="Outputs",kernel_initializer=he_init,activation=tf.nn.relu)

## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/1607384089.py in <cell line: 0>()
      1 with tf.name_scope("dnn"):
----> 2     y_pred=tf.layers.dense(dnn(X,training),n_outputs,name="Outputs",kernel_initializer=he_init,activation=tf.nn.relu)

AttributeError: module 'tensorflow' has no attribute 'layers'

## === cell 19
with tf.name_scope("loss"):

    error=y_pred-y
    mse=tf.reduce_mean(tf.square(error),name='MSE')

## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2858619765.py in <cell line: 0>()
      1 with tf.name_scope("loss"):
      2 
----> 3     error=y_pred-y
      4     mse=tf.reduce_mean(tf.square(error),name='MSE')

NameError: name 'y_pred' is not defined

## === cell 20
l_rate=0.0001
with tf.name_scope("train"):

    optimizer=tf.train.AdamOptimizer(learning_rate=l_rate)
    t_op=optimizer.minimize(mse)

## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/896694006.py in <cell line: 0>()
      2 with tf.name_scope("train"):
      3 
----> 4     optimizer=tf.train.AdamOptimizer(learning_rate=l_rate)
      5     t_op=optimizer.minimize(mse)

AttributeError: module 'tensorflow._api.v2.train' has no attribute 'AdamOptimizer'

## === cell 21
init=tf.global_variables_initializer()
saver=tf.train.Saver()

## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/1252750975.py in <cell line: 0>()
----> 1 init=tf.global_variables_initializer()
      2 saver=tf.train.Saver()

AttributeError: module 'tensorflow' has no attribute 'global_variables_initializer'

## === cell 22
sess=tf.InteractiveSession()
sess.run(init)

## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/3147797020.py in <cell line: 0>()
----> 1 sess=tf.InteractiveSession()
      2 sess.run(init)

AttributeError: module 'tensorflow' has no attribute 'InteractiveSession'

## === cell 23
saver.restore(sess,'../input/dnn-model/model_.ckpt')

## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1081872608.py in <cell line: 0>()
----> 1 saver.restore(sess,'../input/dnn-model/model_.ckpt')

NameError: name 'saver' is not defined

## === cell 24
y_test=y_pred.eval(feed_dict={X:x_test})

## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2113956095.py in <cell line: 0>()
----> 1 y_test=y_pred.eval(feed_dict={X:x_test})

NameError: name 'y_pred' is not defined

## === cell 25
sess.close()

## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2017321283.py in <cell line: 0>()
----> 1 sess.close()

NameError: name 'sess' is not defined

## === cell 26
my_submission =pd.DataFrame(np.concatenate((df_test['key'].values.reshape(-1,1),y_test),axis=1),columns=['key','fare_amount'])

## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/163389177.py in <cell line: 0>()
----> 1 my_submission =pd.DataFrame(np.concatenate((df_test['key'].values.reshape(-1,1),y_test),axis=1),columns=['key','fare_amount'])

NameError: name 'y_test' is not defined

## === cell 27
my_submission.to_csv('submission.csv', index=False)

## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1373934707.py in <cell line: 0>()
----> 1 my_submission.to_csv('submission.csv', index=False)

NameError: name 'my_submission' is not defined

## === cell 28
y_test

## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4075987148.py in <cell line: 0>()
----> 1 y_test

NameError: name 'y_test' is not defined
