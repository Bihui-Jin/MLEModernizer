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
Predict a patient’s severity of decline in lung function based on a CT scan of their lungs. Lung function is assessed based on output from a spirometer, which measures the forced vital capacity (`FVC`), i.e. the volume of air exhaled.

## Metric
A modified version of the Laplace Log Likelihood. 

For each true FVC measurement, you will predict both an FVC and a confidence measure (standard deviation 𝜎𝜎). The metric is computed as:

$$
\begin{gathered}
\sigma_{\text {clipped }}=\max (\sigma, 70), \\
\Delta=\min \left(\left|F V C_{\text {true }}-F V C_{\text {predicted }}\right|, 1000\right), \\
\text { metric }=-\frac{\sqrt{2} \Delta}{\sigma_{\text {clipped }}}-\ln \left(\sqrt{2} \sigma_{\text {clipped }}\right) .
\end{gathered}
$$

The error is thresholded at 1000 ml to avoid large errors adversely penalizing results, while the confidence values are clipped at 70 ml to reflect the approximate measurement uncertainty in FVC. The final score is calculated by averaging the metric across all test set `Patient_Week`s (three per patient). 

Metric values will be negative and higher is better.

## Submission Format
For each `Patient_Week`, you must predict the `FVC` and a confidence. You are asked to predict every patient's `FVC` measurement for every possible week. Those weeks which are not in the final three visits are ignored in scoring.

The file should contain a header and have the following format:

```
Patient_Week,FVC,Confidence
ID00002637202176704235138_1,2000,100
ID00002637202176704235138_2,2000,100
ID00002637202176704235138_3,2000,100
etc.

```

## Dataset
In the dataset, you are provided with a baseline chest CT scan and associated clinical information for a set of patients. A patient has an image acquired at time `Week = 0` and has numerous follow up visits over the course of approximately 1-2 years, at which time their `FVC` is measured.

- In the training set, you are provided with an anonymized, baseline CT scan and the entire history of FVC measurements.
- In the test set, you are provided with a baseline CT scan and only the initial FVC measurement. **You are asked to predict the final three `FVC` measurements for each patient, as well as a confidence value in your prediction.**

- **train.csv** - the training set, contains full history of clinical information
- **test.csv** - the test set, contains only the baseline measurement
- **train/** - contains the training patients' baseline CT scan in DICOM format
- **test/** - contains the test patients' baseline CT scan in DICOM format
- **sample_submission.csv** - demonstrates the submission format

**train.csv and test.csv**

- `Patient`a unique Id for each patient (also the name of the patient's DICOM folder)
- `Weeks`the relative number of weeks pre/post the baseline CT (may be negative)
- `FVC` - the recorded lung capacity in ml
- `Percent`a computed field which approximates the patient's FVC as a percent of the typical FVC for a person of similar characteristics
- `Age`
- `Sex`
- `SmokingStatus`

**sample submission.csv**

- `Patient_Week` - a unique Id formed by concatenating the `Patient` and `Weeks` columns (i.e. ABC_22 is a prediction for patient ABC at week 22)
- `FVC` - the predicted FVC in ml
- `Confidence` - a confidence value of your prediction (also has units of ml)

# 2. Python version

3.8

# 3. Installed packages

geopandas==0.14.4
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
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
tf_keras==2.18.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (122 lines)
            sample_submission.csv (1909 lines)
            sample_submission.csv.zip (5.7 kB)
            test.csv (19 lines)
            test.csv.zip (748 Bytes)
            test.zip (1.2 GB)
            train.csv (1395 lines)
            train.csv.zip (23.6 kB)
            train.zip (12.7 GB)
            osic-pulmonary-fibrosis-progression/
                description.md (122 lines)
                sample_submission.csv (1909 lines)
                ... and 7 other files
                osic-pulmonary-fibrosis-progression/
                test/
                    ID00014637202177757139317/
                        1.dcm (1.5 MB)
                        10.dcm (1.5 MB)
                        ... and 29 other files
                    ID00019637202178323708467/
                        1.dcm (525.5 kB)
                        10.dcm (525.5 kB)
                        ... and 27 other files
                    ... and 17 other folders
                train/
                    ID00007637202177411956430/
                        1.dcm (525.6 kB)
                        10.dcm (525.6 kB)
                        ... and 28 other files
                    ID00009637202177434476278/
                        1.dcm (1.2 MB)
                        10.dcm (1.2 MB)
                        ... and 392 other files
                    ... and 157 other folders
            test/
                ID00014637202177757139317/
                    1.dcm (1.5 MB)
                    10.dcm (1.5 MB)
                    ... and 29 other files
                ID00019637202178323708467/
                    1.dcm (525.5 kB)
                    10.dcm (525.5 kB)
                    ... and 27 other files
                ... and 17 other folders
            train/
                ID00007637202177411956430/
                    1.dcm (525.6 kB)
                    10.dcm (525.6 kB)
                    ... and 28 other files
                ID00009637202177434476278/
                    1.dcm (1.2 MB)
                    10.dcm (1.2 MB)
                    ... and 392 other files
                ... and 157 other folders
        input/
            description.md (122 lines)
            sample_submission.csv (1909 lines)
            sample_submission.csv.zip (5.7 kB)
            test.csv (19 lines)
            test.csv.zip (748 Bytes)
            test.zip (1.2 GB)
            train.csv (1395 lines)
            train.csv.zip (23.6 kB)
            train.zip (12.7 GB)
            osic-pulmonary-fibrosis-progression/
                description.md (122 lines)
                sample_submission.csv (1909 lines)
                ... and 7 other files
                osic-pulmonary-fibrosis-progression/
                test/
                    ID00014637202177757139317/
                        1.dcm (1.5 MB)
                        10.dcm (1.5 MB)
                        ... and 29 other files
                    ID00019637202178323708467/
                        1.dcm (525.5 kB)
                        10.dcm (525.5 kB)
                        ... and 27 other files
                    ... and 17 other folders
                train/
                    ID00007637202177411956430/
                        1.dcm (525.6 kB)
                        10.dcm (525.6 kB)
                        ... and 28 other files
                    ID00009637202177434476278/
                        1.dcm (1.2 MB)
                        10.dcm (1.2 MB)
                        ... and 392 other files
                    ... and 157 other folders
            test/
                ID00014637202177757139317/
                    1.dcm (1.5 MB)
                    10.dcm (1.5 MB)
                    ... and 29 other files
                ID00019637202178323708467/
                    1.dcm (525.5 kB)
                    10.dcm (525.5 kB)
                    ... and 27 other files
                ... and 17 other folders
            train/
                ID00007637202177411956430/
                    1.dcm (525.6 kB)
                    10.dcm (525.6 kB)
                    ... and 28 other files
                ID00009637202177434476278/
                    1.dcm (1.2 MB)
                    10.dcm (1.2 MB)
                    ... and 392 other files
                ... and 157 other folders
        working/
            osic-pulmonary-fibrosis-progression/
                description.md (122 lines)
                sample_submission.csv (1909 lines)
                ... and 7 other files
                osic-pulmonary-fibrosis-progression/
                test/
                    ID00014637202177757139317/
                        1.dcm (1.5 MB)
                        10.dcm (1.5 MB)
                        ... and 29 other files
                    ID00019637202178323708467/
                        1.dcm (525.5 kB)
                        10.dcm (525.5 kB)
                        ... and 27 other files
                    ... and 17 other folders
                train/
                    ID00007637202177411956430/
                        1.dcm (525.6 kB)
                        10.dcm (525.6 kB)
                        ... and 28 other files
                    ID00009637202177434476278/
                        1.dcm (1.2 MB)
                        10.dcm (1.2 MB)
                        ... and 392 other files
                    ... and 157 other folders
```

-> data/osic-pulmonary-fibrosis-progression/sample_submission.csv has 1908 rows and 3 columns.
The columns are: Patient_Week, FVC, Confidence

-> data/osic-pulmonary-fibrosis-progression/test.csv has 18 rows and 7 columns.
The columns are: Patient, Weeks, FVC, Percent, Age, Sex, SmokingStatus

-> data/osic-pulmonary-fibrosis-progression/train.csv has 1394 rows and 7 columns.
The columns are: Patient, Weeks, FVC, Percent, Age, Sex, SmokingStatus

-> data/sample_submission.csv has 1908 rows and 3 columns.
The columns are: Patient_Week, FVC, Confidence

-> data/test.csv has 18 rows and 7 columns.
The columns are: Patient, Weeks, FVC, Percent, Age, Sex, SmokingStatus

-> data/train.csv has 1394 rows and 7 columns.
The columns are: Patient, Weeks, FVC, Percent, Age, Sex, SmokingStatus

-> (stopped after 10 files for performance)

# 5. Target score

-9.2867

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf

from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import keras.backend as K


## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
!pip install sklearn


## === cell 2
import pathlib
train_df = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv')


## === cell 3
train_df.head()


## === cell 4
test_df = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')
test_df.head()


## === cell 5
train_df['Source'] = 'train'
test_df['Source'] = 'test'

dataframe = train_df.append([test_df])
dataframe.reset_index(inplace = True)


## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/4115221079.py in <cell line: 0>()
      2 test_df['Source'] = 'test'
      3 
----> 4 dataframe = train_df.append([test_df])
      5 dataframe.reset_index(inplace = True)

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in __getattr__(self, name)
   6297         ):
   6298             return self[name]
-> 6299         return object.__getattribute__(self, name)
   6300 
   6301     @final

AttributeError: 'DataFrame' object has no attribute 'append'

## === cell 6
'''
Add patient level Baseline information, only the information that the test dataset will also have
1. Number of visits
2. Visit Number (0,1,2,3,4)
4. Variation in Percent
5. Change in smoking status
6. Range of Percent

'''


## === cell 7
def df_to_dataset(dataframe, shuffle=True, batch_size=32):
  dataframe = dataframe.copy()
  labels = dataframe.pop('target')
  ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
  if shuffle:
    ds = ds.shuffle(buffer_size=len(dataframe))
  ds = ds.batch(batch_size)
  return ds


## === cell 8
def own_ZScaler(df, columns):
    for col in columns:
        new_col_name = col + 'Z'
        col_min = df[col].min()
        col_max = df[col].max()        
        df[new_col_name] = (df[col] - col_min) / (col_max - col_min)


## === cell 9
numeric_columns = ['Weeks', 'Age', "Percent"]


## === cell 10
own_ZScaler(dataframe, numeric_columns)


## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1977800796.py in <cell line: 0>()
----> 1 own_ZScaler(dataframe, numeric_columns)

NameError: name 'dataframe' is not defined

## === cell 11
dataframe.head()


## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2260102822.py in <cell line: 0>()
----> 1 dataframe.head()

NameError: name 'dataframe' is not defined

## === cell 12
dataframe['target'] = dataframe['FVC']


## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/90034700.py in <cell line: 0>()
----> 1 dataframe['target'] = dataframe['FVC']

NameError: name 'dataframe' is not defined

## === cell 13
feature_columns = []

for header in ['WeeksZ', 'AgeZ', "PercentZ"]:
  feature_columns.append(feature_column.numeric_column(header))


## === cell 14

indicator_column_names = ['Sex','SmokingStatus']
for col_name in indicator_column_names:
  categorical_column = feature_column.categorical_column_with_vocabulary_list(
      col_name, dataframe[col_name].unique())
  indicator_column = feature_column.indicator_column(categorical_column)
  feature_columns.append(indicator_column)


## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1179704669.py in <cell line: 0>()
      4 for col_name in indicator_column_names:
      5   categorical_column = feature_column.categorical_column_with_vocabulary_list(
----> 6       col_name, dataframe[col_name].unique())
      7   indicator_column = feature_column.indicator_column(categorical_column)
      8   feature_columns.append(indicator_column)

NameError: name 'dataframe' is not defined

## === cell 15
'''## Try instead embedding columns
# embedding columns
embedded_column_names = ['Sex','SmokingStatus']
for col_name in embedded_column_names:
  m = len(dataframe[col_name].unique())
  categorical_column = feature_column.categorical_column_with_vocabulary_list(
      col_name, dataframe[col_name].unique())
  embedded_column = feature_column.embedding_column(categorical_column, dimension = min(50,m//2))
  feature_columns.append(embedded_column)'''


## === cell 16
'''sex_smoker_feature = feature_column.crossed_column(['Sex', 'SmokingStatus'], hash_bucket_size=100)
feature_columns.append(feature_column.indicator_column(sex_smoker_feature))'''


## === cell 17
feature_columns


## === cell 18
feature_layer = tf.keras.layers.DenseFeatures(feature_columns)


## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/3243415958.py in <cell line: 0>()
----> 1 feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

AttributeError: module 'keras._tf_keras.keras.layers' has no attribute 'DenseFeatures'

## === cell 19
train_df = dataframe.loc[dataframe.Source == 'train']
test_df = dataframe.loc[dataframe.Source == 'test']


## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2783746458.py in <cell line: 0>()
      1 # get back original data split
----> 2 train_df = dataframe.loc[dataframe.Source == 'train']
      3 test_df = dataframe.loc[dataframe.Source == 'test']

NameError: name 'dataframe' is not defined

## === cell 20
train, val = train_test_split(train_df, test_size=0.2)
print(len(train), 'train examples')
print(len(val), 'validation examples')


## === cell 21
if len(test_df) < 10:
    EPOCHS = 1000
else:
    EPOCHS = 1000
batch_size = 128


## === cell 22
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)


## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3804         try:
-> 3805             return self._engine.get_loc(casted_key)
   3806         except KeyError as err:

index.pyx in pandas._libs.index.IndexEngine.get_loc()

index.pyx in pandas._libs.index.IndexEngine.get_loc()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

KeyError: 'target'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/1165145611.py in <cell line: 0>()
----> 1 train_ds = df_to_dataset(train, batch_size=batch_size)
      2 val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)

/tmp/ipykernel_11/702872946.py in df_to_dataset(dataframe, shuffle, batch_size)
      2 def df_to_dataset(dataframe, shuffle=True, batch_size=32):
      3   dataframe = dataframe.copy()
----> 4   labels = dataframe.pop('target')
      5   ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
      6   if shuffle:

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in pop(self, item)
   5817         3  monkey        NaN
   5818         """
-> 5819         return super().pop(item=item)
   5820 
   5821     def _replace_columnwise(

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in pop(self, item)
    945 
    946     def pop(self, item: Hashable) -> Series | Any:
--> 947         result = self[item]
    948         del self[item]
    949 

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4100             if self.columns.nlevels > 1:
   4101                 return self._getitem_multilevel(key)
-> 4102             indexer = self.columns.get_loc(key)
   4103             if is_integer(indexer):
   4104                 indexer = [indexer]

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3810             ):
   3811                 raise InvalidIndexError(key)
-> 3812             raise KeyError(key) from err
   3813         except TypeError:
   3814             # If we have a listlike key, _check_indexing_error will raise

KeyError: 'target'

## === cell 23
model = tf.keras.Sequential([
  feature_layer,
  layers.Dense(64, activation='relu'),
  layers.Dropout(.2),
  layers.Dense(64, activation='relu'),
  layers.Dropout(.2),
  layers.Dense(1, activation='linear')
])

ADAM = tf.keras.optimizers.Adam(lr = 0.001)

optimizer = ADAM

model.compile(optimizer= optimizer,
              loss='mae',
              metrics=['mae'])

model.fit(train_ds,
          validation_data=val_ds,
          epochs=EPOCHS)


## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2366093640.py in <cell line: 0>()
      1 model = tf.keras.Sequential([
----> 2   feature_layer,
      3   layers.Dense(64, activation='relu'),
      4   layers.Dropout(.2),
      5   layers.Dense(64, activation='relu'),

NameError: name 'feature_layer' is not defined

## === cell 24
test_df.head()


## === cell 25
column_names = ["Patient", "Weeks", 'PercentZ' , 'AgeZ', 'Sex', 'SmokingStatus']
df = pd.DataFrame(columns = column_names)

for i in range(len(test_df)):
    Patient  = test_df.iloc[i]['Patient']
    PercentZ  = test_df.iloc[i]['PercentZ']
    AgeZ  = test_df.iloc[i]['AgeZ']
    Sex  = test_df.iloc[i]['Sex']
    SmokingStatus  = test_df.iloc[i]['SmokingStatus']
    for week in np.arange(-12,134):
        df = df.append({'Patient': Patient, 
                        "Weeks": week, 
                        'PercentZ' : PercentZ, 
                        'AgeZ': AgeZ, 
                        'Sex': Sex,
                        'SmokingStatus': SmokingStatus 
                         }, ignore_index=True)
df['target'] = 0


## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3804         try:
-> 3805             return self._engine.get_loc(casted_key)
   3806         except KeyError as err:

index.pyx in pandas._libs.index.IndexEngine.get_loc()

index.pyx in pandas._libs.index.IndexEngine.get_loc()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

KeyError: 'PercentZ'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/2216584570.py in <cell line: 0>()
      4 for i in range(len(test_df)):
      5     Patient  = test_df.iloc[i]['Patient']
----> 6     PercentZ  = test_df.iloc[i]['PercentZ']
      7     AgeZ  = test_df.iloc[i]['AgeZ']
      8     Sex  = test_df.iloc[i]['Sex']

/usr/local/lib/python3.11/dist-packages/pandas/core/series.py in __getitem__(self, key)
   1119 
   1120         elif key_is_scalar:
-> 1121             return self._get_value(key)
   1122 
   1123         # Convert generator to list before going through hashable part

/usr/local/lib/python3.11/dist-packages/pandas/core/series.py in _get_value(self, label, takeable)
   1235 
   1236         # Similar to Index.get_value, but we do not fall back to positional
-> 1237         loc = self.index.get_loc(label)
   1238 
   1239         if is_integer(loc):

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3810             ):
   3811                 raise InvalidIndexError(key)
-> 3812             raise KeyError(key) from err
   3813         except TypeError:
   3814             # If we have a listlike key, _check_indexing_error will raise

KeyError: 'PercentZ'

## === cell 26
own_ZScaler(df, ['Weeks'])


## === cell 27
df.head()


## === cell 28
df['Weeks'] = df['Weeks'].astype(int)
df['WeeksZ'] = df['WeeksZ'].astype(float)
df['AgeZ'] = df['AgeZ'].astype(float)
df['PercentZ'] = df['PercentZ'].astype(float)
df['Patient'] = df['Patient'].astype(str)


## === cell 29
test_ds = df_to_dataset(df, shuffle=False, batch_size=batch_size)


## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3804         try:
-> 3805             return self._engine.get_loc(casted_key)
   3806         except KeyError as err:

index.pyx in pandas._libs.index.IndexEngine.get_loc()

index.pyx in pandas._libs.index.IndexEngine.get_loc()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

KeyError: 'target'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/3866573591.py in <cell line: 0>()
----> 1 test_ds = df_to_dataset(df, shuffle=False, batch_size=batch_size)

/tmp/ipykernel_11/702872946.py in df_to_dataset(dataframe, shuffle, batch_size)
      2 def df_to_dataset(dataframe, shuffle=True, batch_size=32):
      3   dataframe = dataframe.copy()
----> 4   labels = dataframe.pop('target')
      5   ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
      6   if shuffle:

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in pop(self, item)
   5817         3  monkey        NaN
   5818         """
-> 5819         return super().pop(item=item)
   5820 
   5821     def _replace_columnwise(

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in pop(self, item)
    945 
    946     def pop(self, item: Hashable) -> Series | Any:
--> 947         result = self[item]
    948         del self[item]
    949 

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4100             if self.columns.nlevels > 1:
   4101                 return self._getitem_multilevel(key)
-> 4102             indexer = self.columns.get_loc(key)
   4103             if is_integer(indexer):
   4104                 indexer = [indexer]

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3810             ):
   3811                 raise InvalidIndexError(key)
-> 3812             raise KeyError(key) from err
   3813         except TypeError:
   3814             # If we have a listlike key, _check_indexing_error will raise

KeyError: 'target'

## === cell 30
preds = model.predict(test_ds, batch_size = 100, verbose = 0)


## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3520259785.py in <cell line: 0>()
----> 1 preds = model.predict(test_ds, batch_size = 100, verbose = 0)

NameError: name 'model' is not defined

## === cell 31
preds


## --- ERROR in cell 31, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/565913920.py in <cell line: 0>()
----> 1 preds

NameError: name 'preds' is not defined

## === cell 32
df['FVC'] = preds
df['Weeks'] = df['Weeks'].astype(str)
df['Confidence'] = 100


## --- ERROR in cell 32, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/209935102.py in <cell line: 0>()
----> 1 df['FVC'] = preds
      2 df['Weeks'] = df['Weeks'].astype(str)
      3 df['Confidence'] = 100

NameError: name 'preds' is not defined

## === cell 33
df['Patient'] = df['Patient'].str.cat(df['Weeks'],sep="_")
df.rename(columns={"Patient": "Patient_Week"}, inplace = True)


## === cell 34
df = df[['Patient_Week','FVC', 'Confidence']]


## --- ERROR in cell 34, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/1638967288.py in <cell line: 0>()
----> 1 df = df[['Patient_Week','FVC', 'Confidence']]

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

KeyError: "['FVC', 'Confidence'] not in index"

## === cell 35
df[["Patient_Week","FVC","Confidence"]].to_csv("submission.csv", index = False)


## --- ERROR in cell 35, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/1083086328.py in <cell line: 0>()
----> 1 df[["Patient_Week","FVC","Confidence"]].to_csv("submission.csv", index = False)

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

KeyError: "['FVC', 'Confidence'] not in index"

## === cell 36
df.head()
