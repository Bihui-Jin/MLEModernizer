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

5.92188

# 6. Current score

1038.40577

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

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
import tensorflow as tf


## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

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
input_func = tf.estimator.inputs.pandas_input_fn(x=X_train,y=y_train ,batch_size=10,num_epochs=1000,
                                            shuffle=True)


## --- ERROR in cell 33, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/1908080178.py in <cell line: 0>()
----> 1 input_func = tf.estimator.inputs.pandas_input_fn(x=X_train,y=y_train ,batch_size=10,num_epochs=1000,
      2                                             shuffle=True)

AttributeError: module 'tensorflow' has no attribute 'estimator'

## === cell 34
model = tf.estimator.DNNRegressor(hidden_units=[20,10,20,20],feature_columns=feat_cols)


## --- ERROR in cell 34, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/3948258498.py in <cell line: 0>()
----> 1 model = tf.estimator.DNNRegressor(hidden_units=[20,10,20,20],feature_columns=feat_cols)

AttributeError: module 'tensorflow' has no attribute 'estimator'

## === cell 35
model.train(input_fn=input_func,steps=25000)


## --- ERROR in cell 35, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3755604716.py in <cell line: 0>()
----> 1 model.train(input_fn=input_func,steps=25000)

NameError: name 'model' is not defined

## === cell 36
predict_input_func = tf.estimator.inputs.pandas_input_fn(
      x=X_test,
      batch_size=10,
      num_epochs=1,
      shuffle=False)


## --- ERROR in cell 36, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/3482710786.py in <cell line: 0>()
----> 1 predict_input_func = tf.estimator.inputs.pandas_input_fn(
      2       x=X_test,
      3       batch_size=10,
      4       num_epochs=1,
      5       shuffle=False)

AttributeError: module 'tensorflow' has no attribute 'estimator'

## === cell 37
pred_gen = model.predict(predict_input_func)


## --- ERROR in cell 37, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1047493184.py in <cell line: 0>()
----> 1 pred_gen = model.predict(predict_input_func)

NameError: name 'model' is not defined

## === cell 38
predictions = list(pred_gen)


## --- ERROR in cell 38, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1949265483.py in <cell line: 0>()
----> 1 predictions = list(pred_gen)

NameError: name 'pred_gen' is not defined

## === cell 39
final_preds = []
for pred in predictions:
    final_preds.append(pred['predictions'])


## --- ERROR in cell 39, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1347375219.py in <cell line: 0>()
      1 final_preds = []
----> 2 for pred in predictions:
      3     final_preds.append(pred['predictions'])

NameError: name 'predictions' is not defined

## === cell 40
from sklearn.metrics import mean_squared_error


## === cell 41
mean_squared_error(y_test,final_preds)**0.5


## --- ERROR in cell 41, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/954863904.py in <cell line: 0>()
----> 1 mean_squared_error(y_test,final_preds)**0.5

/usr/local/lib/python3.11/dist-packages/sklearn/metrics/_regression.py in mean_squared_error(y_true, y_pred, sample_weight, multioutput, squared)
    440     0.825...
    441     """
--> 442     y_type, y_true, y_pred, multioutput = _check_reg_targets(
    443         y_true, y_pred, multioutput
    444     )

/usr/local/lib/python3.11/dist-packages/sklearn/metrics/_regression.py in _check_reg_targets(y_true, y_pred, multioutput, dtype)
     98         correct keyword.
     99     """
--> 100     check_consistent_length(y_true, y_pred)
    101     y_true = check_array(y_true, ensure_2d=False, dtype=dtype)
    102     y_pred = check_array(y_pred, ensure_2d=False, dtype=dtype)

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in check_consistent_length(*arrays)
    395     uniques = np.unique(lengths)
    396     if len(uniques) > 1:
--> 397         raise ValueError(
    398             "Found input variables with inconsistent numbers of samples: %r"
    399             % [int(l) for l in lengths]

ValueError: Found input variables with inconsistent numbers of samples: [99341, 0]

## === cell 42
feat_cols = ['distance','passenger_count']

X_t = test_df[feat_cols]


## === cell 43
X_test = pd.DataFrame(data=scaler.transform(X_t),columns = X_t.columns,index=X_t.index)


## === cell 44
passenger_count = tf.feature_column.numeric_column('passenger_count')
distance = tf.feature_column.numeric_column('distance')


## === cell 45
feat_cols = [ passenger_count,distance]


## === cell 46
predict_input_func = tf.estimator.inputs.pandas_input_fn(
      X_test,
      batch_size=10,
      num_epochs=1,
      shuffle=False)


## --- ERROR in cell 46, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/2754653891.py in <cell line: 0>()
----> 1 predict_input_func = tf.estimator.inputs.pandas_input_fn(
      2       X_test,
      3       batch_size=10,
      4       num_epochs=1,
      5       shuffle=False)

AttributeError: module 'tensorflow' has no attribute 'estimator'

## === cell 47
pred_gen = model.predict(predict_input_func)
predictions = list(pred_gen)


## --- ERROR in cell 47, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1948927792.py in <cell line: 0>()
----> 1 pred_gen = model.predict(predict_input_func)
      2 predictions = list(pred_gen)

NameError: name 'model' is not defined

## === cell 48
final_preds = []
for pred in predictions:
    final_preds.append(float(pred['predictions']))


## --- ERROR in cell 48, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/184923341.py in <cell line: 0>()
      1 final_preds = []
----> 2 for pred in predictions:
      3     final_preds.append(float(pred['predictions']))

NameError: name 'predictions' is not defined

## === cell 49
filename = 'DNNRegressor.csv'

submission = pd.DataFrame(
    {'key': test_df.key, 'fare_amount': final_preds},
    columns = ['key', 'fare_amount'])
submission.to_csv(filename, index = False)


## --- ERROR in cell 49, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1454382884.py in <cell line: 0>()
      1 filename = 'DNNRegressor.csv'
      2 
----> 3 submission = pd.DataFrame(
      4     {'key': test_df.key, 'fare_amount': final_preds},
      5     columns = ['key', 'fare_amount'])

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __init__(self, data, index, columns, dtype, copy)
    776         elif isinstance(data, dict):
    777             # GH#38939 de facto copy defaults to False only in non-dict cases
--> 778             mgr = dict_to_mgr(data, index, columns, dtype=dtype, copy=copy, typ=manager)
    779         elif isinstance(data, ma.MaskedArray):
    780             from numpy.ma import mrecords

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/construction.py in dict_to_mgr(data, index, columns, dtype, typ, copy)
    446             # GH10856
    447             # raise ValueError if only scalars in dict
--> 448             index = _extract_index(arrays[~missing])
    449         else:
    450             index = ensure_index(index)

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/construction.py in _extract_index(data)
    688                     f"length {len(index)}"
    689                 )
--> 690                 raise ValueError(msg)
    691         else:
    692             index = default_index(lengths[0])

ValueError: array length 0 does not match index length 9914
