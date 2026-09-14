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
class DNN_Model:
    def __init__(self,feature,predict,hidden_layers,neurons,layer_dropout,iterations,path_read,path_write,save_model,load_model):
        self.x_f=feature
        self.y_pr=predict
        self.n_hidden_layers=hidden_layers
        self.n_neurons=neurons
        self.dropout=layer_dropout
        self.n_iterations=iterations
        self.initializer = tf.contrib.layers.variance_scaling_initializer()
        self.path_r=path_read
        self.path_w=path_write
        self.save_model=save_model
        self.load_model=load_model

    def dnn(self,inputs,training):

        with tf.variable_scope("dnn"):
            for layer in range(self.n_hidden_layers):


                inputs = tf.layers.dropout(inputs, self.dropout[layer], training=self.training)
                inputs = tf.layers.dense(inputs, self.n_neurons[layer],
                                         kernel_initializer=self.initializer,
                                         name="hidden%d" % (layer + 1))

                inputs=tf.nn.relu(inputs, name="hidden%d_out" % (layer + 1))

            return inputs
    def build_graph(self):
        n_inputs=14
        n_outputs=1
        self.X=tf.placeholder(tf.float32,shape=(None,n_inputs),name="X")
        self.y=tf.placeholder(tf.float32,shape=(None,1),name='y')
        self.training=tf.placeholder_with_default(False,shape=(),name='training')
        with tf.name_scope("dnn"):
            self.y_pred=tf.layers.dense(self.dnn(self.X,self.training),n_outputs,name="Outputs",kernel_initializer=self.initializer,activation=tf.nn.relu)
        with tf.name_scope("loss"):

            error=self.y_pred-self.y
            self.mse=tf.reduce_mean(tf.square(error),name='MSE')
        l_rate=0.0001
        with tf.name_scope("train"):

            optimizer=tf.train.AdamOptimizer(learning_rate=l_rate)
            self.t_op=optimizer.minimize(self.mse)
        


    def train_data(self):
                
        np.random.seed(4)
        index_shuffled=np.random.permutation(x_sc.shape[0])
        ind_train=index_shuffled[0:54000000]
        ind_valid=index_shuffled[54000000:]
        er_=np.zeros((1,2))
        batch_size=128
        tf.reset_default_graph()
        self.build_graph()
        
        saver=tf.train.Saver()
        sess=tf.InteractiveSession()
        if load_model==1:
            saver.restore(sess,self.path_r+'/model_.ckpt')
        else:
            sess.run(init)

        
        for i in range(0,0+self.n_iterations):

            ind_use=np.random.randint(0,54000000,batch_size)
            rand_test=np.random.randint(0,ind_valid.shape[0],1000)
            X_batch=self.x_f[ind_use,:]
            y_batch=self.y_pr[ind_use,:]
            sess.run(self.t_op,feed_dict={self.X:X_batch,self.y:y_batch,self.training:True})
            temp_er=np.vstack((self.mse.eval(feed_dict={self.X:X_batch,self.y:y_batch}),
            self.mse.eval(feed_dict={self.X:x_sc[ind_valid[rand_test]],self.y:y_sc[ind_valid[rand_test]]}))).T
            er_=np.concatenate((er_,temp_er))
        if self.save_model==1:
            saver.save(sess,path_r+'/model_.ckpt')
        sess.close()
        return er_
    def model_interference(self):
        tf.reset_default_graph()
        self.build_graph()
        sess=tf.InteractiveSession()
        saver=tf.train.Saver()
        saver.restore(sess,self.path_r+'/model_.ckpt')
        y_prediction=self.y_pred.eval(feed_dict={self.X:self.x_f}) 
        sess.close()
        return y_prediction

## === cell 9
df_test['Herv_Dist'] =distance(np.float64(df_test.values[:,2:6]))

## === cell 10
add_datepart(df_test, 'pickup_datetime', drop=True)

## --- ERROR in cell 10, traceback:
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

## === cell 11
feature_cols=['pickup_longitude', 'pickup_latitude', 'dropoff_longitude',
       'dropoff_latitude', 'passenger_count', 'pickup_datetimeYear',
       'pickup_datetimeMonth', 'pickup_datetimeWeek', 'pickup_datetimeDay',
       'pickup_datetimeDayofweek', 'pickup_datetimeDayofyear',
       'pickup_datetimehour', 'pickup_datetimeElapsed', 'Herv_Dist']

## === cell 12
df_test[feature_cols].head()

## --- ERROR in cell 12, traceback:
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

## === cell 13
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

## === cell 14
x_test_unscl=df_test[feature_cols].values

## --- ERROR in cell 14, traceback:
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

## === cell 15
x_test=(x_test_unscl-mu)/sigma

## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2611217.py in <cell line: 0>()
----> 1 x_test=(x_test_unscl-mu)/sigma

NameError: name 'x_test_unscl' is not defined

## === cell 18
path_model='../input/dnn-model'
n_neurons_=[2000,1000,500,250,125,50,25,10]
dropout_=[0.25,0,0,0,0,0,0,0]
model=DNN_Model(x_test,[],8,n_neurons_,dropout_,1,path_model,[],0,1)
y_test=model.model_interference()

## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3701557032.py in <cell line: 0>()
      2 n_neurons_=[2000,1000,500,250,125,50,25,10]
      3 dropout_=[0.25,0,0,0,0,0,0,0]
----> 4 model=DNN_Model(x_test,[],8,n_neurons_,dropout_,1,path_model,[],0,1)
      5 y_test=model.model_interference()

NameError: name 'x_test' is not defined

## === cell 19
my_submission =pd.DataFrame(np.concatenate((df_test['key'].values.reshape(-1,1),y_test),axis=1),columns=['key','fare_amount'])
my_submission.to_csv('submission.csv', index=False)

## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1120997317.py in <cell line: 0>()
----> 1 my_submission =pd.DataFrame(np.concatenate((df_test['key'].values.reshape(-1,1),y_test),axis=1),columns=['key','fare_amount'])
      2 my_submission.to_csv('submission.csv', index=False)

NameError: name 'y_test' is not defined
