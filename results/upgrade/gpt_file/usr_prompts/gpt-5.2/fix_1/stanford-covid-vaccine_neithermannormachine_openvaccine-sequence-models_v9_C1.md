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
Predict likely degradation rates at each base of an RNA molecule.

## Metric
Mean columnwise root mean squared error:

$\textrm{MCRMSE} = \frac{1}{N_{t}}\sum_{j=1}^{N_{t}}\sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_{ij} - \hat{y}_{ij})^2}$

where $N_{t}$ is the number of scored ground truth target columns, and $y$ and $\hat{y}$ are the actual and predicted values, respectively.

There are multiple ground truth values provided in the training data. While the submission format requires all 5 to be predicted, only the following are scored: reactivity, deg_Mg_pH10, and deg_Mg_50C.

## Submission Formats
For each sample `id` in the test set, you must predict targets for *each* sequence position (`seqpos`), one per row. If the length of the `sequence` of an `id` is, e.g., 107, then you should make 107 predictions. Positions greater than the `seq_scored` value of a sample are not scored, but still need a value in the solution file.

```csv
id_seqpos,reactivity,deg_Mg_pH10,deg_pH10,deg_Mg_50C,deg_50C
id_d190610e8_0,0.1,0.3,0.2,0.5,0.4
id_d190610e8_1,0.3,0.2,0.5,0.4,0.2
id_d190610e8_2,0.5,0.4,0.2,0.1,0.2
etc.
```

## Dataset 
- **train.json** - the training data
- **test.json** - the test set, without any columns associated with the ground truth.
- **sample_submission.csv** - a sample submission file in the correct format

#### Columns
- `id` - An arbitrary identifier for each sample.
- `seq_scored` - (68 in Train and Public Test, 68 in Private Test) Integer value denoting the number of positions used in scoring with predicted values. This should match the length of `reactivity`, `deg_*` and `*_error_*` columns.
- `seq_length` - (107 in Train and Public Test, 107 in Private Test) Integer values, denotes the length of `sequence`.
- `sequence` - (1x107 string in Train and Public Test, 107 in Private Test) Describes the RNA sequence, a combination of `A`, `G`, `U`, and `C` for each sample. Should be 107 characters long, and the first 68 bases should correspond to the 68 positions specified in `seq_scored` (note: indexed starting at 0).
- `structure` - (1x107 string in Train and Public Test, 107 in Private Test) An array of `(`, `)`, and `.` characters that describe whether a base is estimated to be paired or unpaired. Paired bases are denoted by opening and closing parentheses e.g. (....) means that base 0 is paired to base 5, and bases 1-4 are unpaired.
- `reactivity` - (1x68 vector in Train and Public Test, 1x68 in Private Test) An array of floating point numbers, should have the same length as `seq_scored`. These numbers are reactivity values for the first 68 bases as denoted in `sequence`, and used to determine the likely secondary structure of the RNA sample.
- `deg_pH10` - (1x68 vector in Train and Public Test, 1x68 in Private Test) An array of floating point numbers, should have the same length as `seq_scored`. These numbers are reactivity values for the first 68 bases as denoted in `sequence`, and used to determine the likelihood of degradation at the base/linkage after incubating without magnesium at high pH (pH 10).
- `deg_Mg_pH10` - (1x68 vector in Train and Public Test, 1x68 in Private Test) An array of floating point numbers, should have the same length as `seq_scored`. These numbers are reactivity values for the first 68 bases as denoted in `sequence`, and used to determine the likelihood of degradation at the base/linkage after incubating with magnesium in high pH (pH 10).
- `deg_50C` - (1x68 vector in Train and Public Test, 1x68 in Private Test) An array of floating point numbers, should have the same length as `seq_scored`. These numbers are reactivity values for the first 68 bases as denoted in `sequence`, and used to determine the likelihood of degradation at the base/linkage after incubating without magnesium at high temperature (50 degrees Celsius).
- `deg_Mg_50C` - (1x68 vector in Train and Public Test, 1x68 in Private Test) An array of floating point numbers, should have the same length as `seq_scored`. These numbers are reactivity values for the first 68 bases as denoted in `sequence`, and used to determine the likelihood of degradation at the base/linkage after incubating with magnesium at high temperature (50 degrees Celsius).
- `*_error_*` - An array of floating point numbers, should have the same length as the corresponding `reactivity` or `deg_*` columns, calculated errors in experimental values obtained in `reactivity` and `deg_*` columns.
- `predicted_loop_type` - (1x107 string) Describes the structural context (also referred to as 'loop type')of each character in `sequence`. Loop types assigned by bpRNA from Vienna RNAfold 2 structure. From the bpRNA_documentation: S: paired "Stem" M: Multiloop I: Internal loop B: Bulge H: Hairpin loop E: dangling End X: eXternal loop
    - `S/N filter` Indicates if the sample passed filters described below in `Additional Notes`.

#### Additional Notes
At the beginning of the competition, Stanford scientists have data on 2400 RNA sequences of length 107. For technical reasons, measurements cannot be carried out on the final bases of these RNA sequences, so we have experimental data (ground truth) in 5 conditions for the first 68 bases.

We have split out 240 of these 2400 sequences for a public test set to allow for continuous evaluation through the competition, on the public leaderboard. These sequences, in `test.json`, have been additionally filtered based on three criteria detailed below to ensure that this subset is not dominated by any large cluster of RNA molecules with poor data, which might bias the public leaderboard. The remaining 2160 sequences for which we have data are in `train.json`.

For our final and most important scoring (the Private Leaderbooard), Stanford scientists are carrying out measurements on 240 new RNAs. For these data, we expect to have measurements for the first 68 bases, again missing the ends of the RNA. These sequences constitute the 240 sequences in `test.json`.

For those interested in how the sequences in `test.json` were filtered, here were the steps to ensure a diverse and high quality test set for public leaderboard scoring:

1. Minimum value across all 5 conditions must be greater than -0.5.
2. Mean signal/noise across all 5 conditions must be greater than 1.0. [Signal/noise is defined as mean( measurement value over 68 nts )/mean( statistical error in measurement value over 68 nts)]
3. To help ensure sequence diversity, the resulting sequences were clustered into clusters with less than 50% sequence similarity, and the 240 test set sequences were chosen from clusters with 3 or fewer members. That is, any sequence in the test set should be sequence similar to at most 2 other sequences.

Note that these filters have not been applied to the 2160 RNAs in the public training data `train.json` -- some of those measurements have negative values or poor signal-to-noise, or some RNA sequences have near-identical sequences in that set. But we are providing all those data in case competitors can squeeze out more signal.

# 2. Python version

3.8

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
            description.md (125 lines)
            sample_submission.csv (25681 lines)
            sample_submission.csv.zip (74.8 kB)
            test.json (240 lines)
            train.json (2160 lines)
            stanford-covid-vaccine/
                description.md (125 lines)
                sample_submission.csv (25681 lines)
                ... and 3 other files
                stanford-covid-vaccine/
        input/
            description.md (125 lines)
            sample_submission.csv (25681 lines)
            sample_submission.csv.zip (74.8 kB)
            test.json (240 lines)
            train.json (2160 lines)
            stanford-covid-vaccine/
                description.md (125 lines)
                sample_submission.csv (25681 lines)
                ... and 3 other files
                stanford-covid-vaccine/
        working/
            stanford-covid-vaccine/
                description.md (125 lines)
                sample_submission.csv (25681 lines)
                ... and 3 other files
                stanford-covid-vaccine/
```

-> data/sample_submission.csv has 25680 rows and 6 columns.
The columns are: id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C

-> data/stanford-covid-vaccine/sample_submission.csv has 25680 rows and 6 columns.
The columns are: id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C

-> data/stanford-covid-vaccine/test.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "index": {
      "type": "integer"
    },
    "id": {
      "type": "string"
    },
    "sequence": {
      "type": "string"
    },
    "structure": {
      "type": "string"
    },
    "predicted_loop_type": {
      "type": "string"
    },
    "seq_length": {
      "type": "integer"
    },
    "seq_scored": {
      "type": "integer"
    }
  },
  "required": [
    "id",
    "index",
    "predicted_loop_type",
    "seq_length",
    "seq_scored",
    "sequence",
    "structure"
  ]
}

-> data/stanford-covid-vaccine/train.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "index": {
      "type": "integer"
    },
    "id": {
      "type": "string"
    },
    "sequence": {
      "type": "string"
    },
    "structure": {
      "type": "string"
    },
    "predicted_loop_type": {
      "type": "string"
    },
    "signal_to_noise": {
      "type": "number"
    },
    "SN_filter": {
      "type": "integer"
    },
    "seq_length": {
      "type": "integer"
    },
    "seq_scored": {
      "type": "integer"
    },
    "reactivity_error": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_error_Mg_pH10": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_error_pH10": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_error_Mg_50C": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_error_50C": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "reactivity": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_Mg_pH10": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_pH10": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_Mg_50C": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_50C": {
      "type": "array",
      "items": {
        "type": "number"
      }
    }
  },
  "required": [
    "SN_filter",
    "deg_50C",
    "deg_Mg_50C",
    "deg_Mg_pH10",
    "deg_error_50C",
    "deg_error_Mg_50C",
    "deg_error_Mg_pH10",
    "deg_error_pH10",
    "deg_pH10",
    "id",
    "index",
    "predicted_loop_type",
    "reactivity",
    "reactivity_error",
    "seq_length",
    "seq_scored",
    "sequence",
    "signal_to_noise",
    "structure"
  ]
}

-> data/test.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "index": {
      "type": "integer"
    },
    "id": {
      "type": "string"
    },
    "sequence": {
      "type": "string"
    },
    "structure": {
      "type": "string"
    },
    "predicted_loop_type": {
      "type": "string"
    },
    "seq_length": {
      "type": "integer"
    },
    "seq_scored": {
      "type": "integer"
    }
  },
  "required": [
    "id",
    "index",
    "predicted_loop_type",
    "seq_length",
    "seq_scored",
    "sequence",
    "structure"
  ]
}

-> data/train.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "index": {
      "type": "integer"
    },
    "id": {
      "type": "string"
    },
    "sequence": {
      "type": "string"
    },
    "structure": {
      "type": "string"
    },
    "predicted_loop_type": {
      "type": "string"
    },
    "signal_to_noise": {
      "type": "number"
    },
    "SN_filter": {
      "type": "integer"
    },
    "seq_length": {
      "type": "integer"
    },
    "seq_scored": {
      "type": "integer"
    },
    "reactivity_error": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_error_Mg_pH10": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_error_pH10": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_error_Mg_50C": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_error_50C": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "reactivity": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_Mg_pH10": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_pH10": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_Mg_50C": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_50C": {
      "type": "array",
      "items": {
        "type": "number"
      }
    }
  },
  "required": [
    "SN_filter",
    "deg_50C",
    "deg_Mg_50C",
    "deg_Mg_pH10",
    "deg_error_50C",
    "deg_error_Mg_50C",
    "deg_error_Mg_pH10",
    "deg_error_pH10",
    "deg_pH10",
    "id",
    "index",
    "predicted_loop_type",
    "reactivity",
    "reactivity_error",
    "seq_length",
    "seq_scored",
    "sequence",
    "signal_to_noise",
    "structure"
  ]
}

-> input/sample_submission.csv has 25680 rows and 6 columns.
The columns are: id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C

-> (stopped after 10 files for performance)

# 5. Target score

0.41379

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
import sklearn
import matplotlib.pyplot as plt

'''
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
'''


## === cell 1
target_cols = ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']


## === cell 2
def read_json(filename):
    '''
    reads in train/test json data as pandas DataFrame
    '''
    file = open(filename)
    df = pd.read_json(path_or_buf = file, orient = 'records', lines = True)
    return df


## === cell 3

train_df = read_json('../input/stanford-covid-vaccine/train.json')

print(train_df['id'].nunique())
print(train_df.columns)
train_df


## === cell 4

test_df = read_json('../input/stanford-covid-vaccine/test.json')


print('Features only in training set (not including target columns):') 
set(train_df.columns) - set(test_df.columns) - set(target_cols)


## === cell 5
test_df


## === cell 6
'''! ls
#! ls draw_rna
! ls forna

! python forna/forna_server.py -s -d'''


## === cell 7
'''seq = train_df.loc[0, 'sequence']
struct = train_df.loc[0, 'structure']

seq, struct'''


## === cell 8
def unpack_df_lists(df, col_names):
    '''
    turn list-like elements of dataframe into tabular data
    
    works great
    '''
    if isinstance(col_names, str): #if string is passed in, convert to list for convenience
        col_names = [col_names]
    
    all_series = [df[c] for c in col_names] #select relevant columns
    unpacked = [ser.explode() for ser in all_series] #unpack lists for each feature series
    
    data = pd.concat(unpacked, axis = 1) #concat unpacked columns together
    
    original = df.drop(col_names, axis = 1) #drop columns with list elements
    data = original.join(data) #then join unpacked data to original df
    
    return data


## === cell 9
'''
def feature_engineer(df, train = True, **kwargs):
    
    
    unpack_cols = ['reactivity_error', 'deg_error_Mg_pH10', 'deg_error_pH10',
       'deg_error_Mg_50C', 'deg_error_50C', 'reactivity', 'deg_Mg_pH10',
       'deg_pH10', 'deg_Mg_50C', 'deg_50C'] #only need to unpack things in training set
    
    #unpack list elements (only training set needs to be unpacked)
    if train:
        data = unpack_df_lists(df, unpack_cols)
    else: #if test data, need to add rows manually
        data = df.copy()
        data['temp'] = data.apply(lambda row: [0] * row['seq_length'], axis = 1) #adds temp column with list-like elements, of len(seq_scored) for that row 
        data = unpack_df_lists(data, 'temp') #unpack to right length using this function
        del data['temp'] #delete the temp column
        #this works great!
        
    #adds seqpos column to record position of each row in each id's individual sequence
    data['seqpos'] = 1
    data['seqpos'] = data.groupby('id').cumsum()['seqpos'] - 1
    
    #adds nucleotide column to record base (A,C,G,U) at position seqpos
    seq_temp = pd.concat([data['sequence'],data['seqpos']], axis = 1)
    data['nucleotide'] = seq_temp.apply(lambda row: row['sequence'][row['seqpos']], axis = 1) #get base at seqpos in sequence string
    
    #adds pred_loop_seqpos column to record predicted loop type at position seqpos
    loop_temp = pd.concat([data['predicted_loop_type'],data['seqpos']], axis = 1)
    data['pred_loop_seqpos'] = loop_temp.apply(lambda row: row['predicted_loop_type'][row['seqpos']], axis = 1) #get type at seqpos in predicted_loop_type string 
    
    #do one-hot encoding on nucleotide column? or label encoding?
    data = pd.get_dummies(data, columns = ['nucleotide','pred_loop_seqpos']) #do one-hot encoding on predicted_loop_type & nucleotide column
    
    return data
'''


## === cell 10

tokenize_cols = ['sequence', 'structure', 'predicted_loop_type']
def tokenize_df(df, tokenizer, cols = tokenize_cols):
    '''
    tokenizer is a tensorflow keras Tokenizer that has been already fitted on text
    '''
    data = df.copy()
    for c in tokenize_cols:
        data[c] = tokenizer.texts_to_sequences(data[c])
    return data


## === cell 11
from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer(filters = None, lower = False, char_level = True)
tokenizer.fit_on_texts('().ACGUBEHIMSX')

temp = train_df[train_df['SN_filter'] == 1]
temp = tokenize_df(temp, tokenizer)

train_df = temp

train_df


## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 13
import matplotlib.pyplot as plt
import seaborn as sns

corr_data = train_df.drop(['index','id','sequence','structure','predicted_loop_type', 'seq_length','seq_scored'], axis = 1)
corr_data = unpack_df_lists(corr_data, ['reactivity_error', 'deg_error_Mg_pH10', 'deg_error_pH10',
       'deg_error_Mg_50C', 'deg_error_50C', 'reactivity', 'deg_Mg_pH10',
       'deg_pH10', 'deg_Mg_50C', 'deg_50C']).convert_dtypes()


print(corr_data)

corr_data = corr_data.corr()

corr_data


## === cell 14
mask = np.ma.masked_inside(corr_data.values, -0.15, 0.15).mask #get most powerful features

plt.figure(figsize = (8,8))
sns.heatmap(corr_data, annot = True, mask = mask)


## === cell 15
'''
For tensorflow compatibility, metrics should have signature f(y_true, y_pred)
For sklearn compatibility, metrics should have signature f(y_true, y_pred, **kwargs)
'''


def score(raw_values = False, use_tf = False, **kwargs):
    '''
    This is competition metric: Mean Columnwise Root Mean Square Error (MCRMSE)
    Averages RMSE loss over all scored columns (only 3 are scored)
    
    Parameters:
    For now, kwargs is ignored
    tf -- True if using in tensorflow, false if not
    col_dict is a dictionary that maps column number index to column name
        keys 'reactivity', 'deg_Mg_pH10', 'deg_Mg_50C'
        values are numeric index of that column in y_pred
    raw_values determines if losses for each column are returned or just the average
        if True, losses for each of columns are returned
        if False, only average is returned
    
    Returns a loss function that computes MCRMSE for scored columns
    '''    
    
    col_dict = {
        'reactivity':0,
        'deg_Mg_pH10':1,
        'deg_Mg_50C':3
    }
    
    unscored = set([0,1,2,3,4]) - set(col_dict.values())
    
    multi = 'uniform_average'
    if raw_values:
        multi = 'raw_values'
    
    def loss(y_true, y_pred):
        '''
        y_true & y_pred may have more columns than needed for scoring
        select only necessary ones for scoring

        y_true & y_pred have shapes (n, 5, len), where n is # of id_seqpos combos, len is length of sequence
        '''
        from sklearn.metrics import mean_squared_error
        y_true = np.array(y_true) #convert to np for convenience
        y_pred = np.array(y_pred)

        
        y_true = y_true[:, list(col_dict.values())]
        y_pred = y_pred[:, list(col_dict.values())]
        

        metric = mean_squared_error(y_true, y_pred, squared = False, multioutput = multi)
        return metric
    
    def loss_tf(y_true, y_pred):
        '''
        WIP -- need to prioritize scored columns
        
        '''
        
        import tensorflow as tf
        import tensorflow.keras.backend as tf_kb
        
        
        
        
        '''for c in unscored:
            y_pred[:, c] = y_true[:, c]'''
            
        '''
        colwise_mse = tf_kb.mean(tf_kb.square(y_true - y_pred)) #first take average squared difference over row examples (shape 1x3)
        return tf_kb.mean(tf_kb.sqrt(colwise_mse)) #then sqrt and take average over columns (shape 1x1 / scalar)
        '''        
        
        
        colwise_mse = tf.reduce_mean(tf.square(y_true - y_pred), axis=1)
        return tf.reduce_mean(tf.sqrt(colwise_mse), axis=1)
        
        
    if not use_tf:
        return loss
    else:
        return loss_tf


## === cell 16

train_only_cols = ['reactivity_error', 'deg_error_Mg_pH10', 'deg_error_pH10', 'deg_error_Mg_50C', 'deg_error_50C'] #features only in train set
signal_cols = ['signal_to_noise','SN_filter']
drop_cols = [
             'seq_length', 'seq_scored', #don't actually use seq_length and seq_scored for training - just metadata
            'index', 'id']  #also not actually useful for training
train_drop_cols = drop_cols + target_cols + train_only_cols + signal_cols


X_train = (train_df.drop(train_drop_cols, axis = 1) #selects only relevant columns
                    .apply(lambda row: [e for e in row], axis = 1) #concatenates each element in row to list
                    .apply(lambda e : np.array(e)) #creates 2d numpy array from those elements
          )
X_train = np.stack(X_train.values, axis = 0) #stacking might not work for test data, since there is two different shapes

y_train = (train_df[target_cols]
               .apply(lambda row: [e for e in row], axis = 1) #concatenates each element in row to list
                .apply(lambda e : np.array(e)) #creates 2d numpy array from those elements
          )
y_train = np.stack(y_train.values, axis = 0)

'''
#maybe can use this as example weights -- higher signal_to_noise means higher weight?
#probably gotta make sure to cap the weight though, otherwise training dominated by top signal_to_noise
signal_to_noise = train_df[signal_cols]  #not necessary anymore, although might still be useful -- try experimenting
'''




## === cell 17
X_train


## === cell 18
y_train


## === cell 19
sw = train_df['signal_to_noise'].values


import matplotlib.pyplot as plt

q = np.quantile(sw, np.linspace(0,1,21))
plt.plot(q, np.log1p(q + 5)/2)


sw = np.log1p(sw + 5)/2


## === cell 20
import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras.optimizers import Adam


def make_model():
    '''
    Creates a tensorflow keras sequence model
    '''
    EMBEDDING_PARAMS = {'input_dim': len(tokenizer.word_index) + 1,
                        'output_dim': 100, #play around with this number
                       }
    
    shape = (3,None) #3 sequences of unknown length (should all be same length though)
    
    seq_inputs = tf.keras.Input(shape = shape) #shape (n, 3, seq_length)
    embed = layers.Embedding(**EMBEDDING_PARAMS)(seq_inputs) #(n, 3, seq_length, output_dim)
    
    def rnn_layer(num_neurons):
        return layers.Bidirectional(layers.LSTM(num_neurons, return_sequences = True)) #define the rnn type to use
    
    rnn_layers = []
    for i in range(embed.shape[1]): #loop thru sequences
        r = rnn_layer(30)(embed[:,i])
        r = rnn_layer(30)(r)
        rnn_layers.append(r) #r is shape (n, seq_length, num_rnn_neurons * 2 (b/c bidirectional))
    
    x = layers.Concatenate()(rnn_layers) #concatenate the rnn layers for each sequence (n, seq_length, num_rnn_neurons * 2 * 3)
    x = layers.Dense(100, activation = 'relu')(x) #(n, seq_length, 100)
    x = layers.Dense(5, activation = 'linear')(x) #need output layer of 5 x seq_scored; shape (n, seq_length, 5)
    
    x = tf.transpose(x, (0,2,1)) #reshape prediction for submitting output & computing loss -- (n, 5, seq_length)
    x = x[:, :, :-39] #compact sequence of 107/130 into 68/91 by removing last 39 elements (n, 5, seq_scored)
    
    model = tf.keras.Model(inputs = seq_inputs, outputs = x)
    
    optimizer = Adam(learning_rate = 0.01) #try different learning rates, optimizer hyperparameters
    loss = score(use_tf = True)
    
    model.compile(optimizer = optimizer, loss = loss, metrics = ['mse'])
    
    
    return model


## === cell 21
'''
work on callbacks (lr scheduling, logging, etc)
'''
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping
from sklearn.model_selection import train_test_split

TF_FITPARAMS = {'epochs': 150,
               'batch_size':100,
               'validation_batch_size':50}
fp = TF_FITPARAMS

def schedule_func(epoch, lr):
    '''
    function passed to LearningRateScheduler to determine learning rate at each epoch
    keeps lr constant until certain percent of epochs have elapsed, then exponentially decreases lr
    '''
    if epoch < 50:
        return lr
    else:
        return lr * np.exp(-0.07)
    
callbacks = [
    LearningRateScheduler(schedule_func),
    EarlyStopping(monitor = 'val_loss', mode = 'min', min_delta = 5e-5, patience = 10)
]



model = make_model()
model.summary()
print(X_train.shape, y_train.shape)




X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train)

%time history = model.fit(X_tr, y_tr, validation_data = (X_val, y_val), callbacks = callbacks, **fp)


## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3663907571.py in <cell line: 0>()
     31 
     32 #instantiate model
---> 33 model = make_model()
     34 model.summary()
     35 print(X_train.shape, y_train.shape)

/tmp/ipykernel_11/1895267959.py in make_model()
     41     x = layers.Dense(5, activation = 'linear')(x) #need output layer of 5 x seq_scored; shape (n, seq_length, 5)
     42 
---> 43     x = tf.transpose(x, (0,2,1)) #reshape prediction for submitting output & computing loss -- (n, 5, seq_length)
     44     x = x[:, :, :-39] #compact sequence of 107/130 into 68/91 by removing last 39 elements (n, 5, seq_scored)
     45 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/weak_tensor_ops.py in wrapper(*args, **kwargs)
     86   def wrapper(*args, **kwargs):
     87     if not ops.is_auto_dtype_conversion_enabled():
---> 88       return op(*args, **kwargs)
     89     bound_arguments = signature.bind(*args, **kwargs)
     90     bound_arguments.apply_defaults()

/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/traceback_utils.py in error_handler(*args, **kwargs)
    151     except Exception as e:
    152       filtered_tb = _process_traceback_frames(e.__traceback__)
--> 153       raise e.with_traceback(filtered_tb) from None
    154     finally:
    155       del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/backend/common/keras_tensor.py in __tf_tensor__(self, dtype, name)
    136 
    137     def __tf_tensor__(self, dtype=None, name=None):
--> 138         raise ValueError(
    139             "A KerasTensor cannot be used as input to a TensorFlow function. "
    140             "A KerasTensor is a symbolic placeholder for a shape and dtype, "

ValueError: A KerasTensor cannot be used as input to a TensorFlow function. A KerasTensor is a symbolic placeholder for a shape and dtype, used when constructing Keras Functional models or Keras Functions. You can only use it as input to a Keras layer or a Keras operation (from the namespaces `keras.layers` and `keras.operations`). You are likely doing something like:

```
x = Input(...)
...
tf_fn(x)  # Invalid.
```

What you should do instead is wrap `tf_fn` in a layer:

```
class MyLayer(Layer):
    def call(self, x):
        return tf_fn(x)

x = MyLayer()(x)
```


## === cell 22
def save_model(model):
    '''
    WIP
    model is a fitted tensorflow keras model 
    '''
    
    pass


## === cell 23
import matplotlib.pyplot as plt

plt.figure(figsize = (13,7))


all_metrics = set(history.history.keys()) - set(['lr']) #plot learning rate seperately
for metric in all_metrics:
    metric_history = history.history[metric]
    plt.plot(metric_history, label = metric)
plt.legend()
plt.show()

plt.plot(history.history['lr'], label = 'lr')
plt.legend()
plt.show()


## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/762938380.py in <cell line: 0>()
      7 
      8 #first plot loss and related metrics
----> 9 all_metrics = set(history.history.keys()) - set(['lr']) #plot learning rate seperately
     10 for metric in all_metrics:
     11     metric_history = history.history[metric]

NameError: name 'history' is not defined

## === cell 24
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

param_grid = {}
model_sk = KerasRegressor(build_fn = make_model) #make sk_learn wrapped model to use sklearn hyperparameter optimization


## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/2478277528.py in <cell line: 0>()
      1 from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
----> 2 from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
      3 #wip
      4 
      5 param_grid = {}

ModuleNotFoundError: No module named 'tensorflow.keras.wrappers.scikit_learn'

## === cell 27
test_df = read_json('../input/stanford-covid-vaccine/test.json')

test_df


## === cell 28
test_public = test_df[test_df['seq_length'] == 107]
test_public = tokenize_df(test_public, tokenizer)

test_public


## === cell 29
test_private = test_df[test_df['seq_length'] == 130]
test_private = tokenize_df(test_private, tokenizer)


test_private


## === cell 30
X_test_public = (test_public.drop(drop_cols, axis = 1) #selects only relevant columns
                    .apply(lambda row: [e for e in row], axis = 1) #concatenates each element in row to list
                    .apply(lambda e : np.array(e)) #creates 2d numpy array from those elements
          )
X_test_public = np.stack(X_test_public.values, axis = 0) #shape (n,3,107)

X_test_public


## === cell 31
X_test_private = (test_private.drop(drop_cols, axis = 1) #selects only relevant columns
                    .apply(lambda row: [e for e in row], axis = 1) #concatenates each element in row to list
                    .apply(lambda e : np.array(e)) #creates 2d numpy array from those elements
          )
X_test_private = np.stack(X_test_private.values, axis = 0) #shape (n,3,130)

X_test_private


## --- ERROR in cell 31, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1379439377.py in <cell line: 0>()
      4                     .apply(lambda e : np.array(e)) #creates 2d numpy array from those elements
      5           )
----> 6 X_test_private = np.stack(X_test_private.values, axis = 0) #shape (n,3,130)
      7 
      8 X_test_private

/usr/local/lib/python3.11/dist-packages/numpy/core/shape_base.py in stack(arrays, axis, out, dtype, casting)
    443     arrays = [asanyarray(arr) for arr in arrays]
    444     if not arrays:
--> 445         raise ValueError('need at least one array to stack')
    446 
    447     shapes = {arr.shape for arr in arrays}

ValueError: need at least one array to stack

## === cell 32

test_pred_public = model.predict(X_test_public) #shape (n,5,68)
test_pred_public


## --- ERROR in cell 32, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4103711864.py in <cell line: 0>()
      1 #do actual prediction
      2 
----> 3 test_pred_public = model.predict(X_test_public) #shape (n,5,68)
      4 test_pred_public

NameError: name 'model' is not defined

## === cell 33
test_pred_private = model.predict(X_test_private) #shape (n,5,91)
test_pred_private


## --- ERROR in cell 33, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3179537886.py in <cell line: 0>()
----> 1 test_pred_private = model.predict(X_test_private) #shape (n,5,91)
      2 test_pred_private

NameError: name 'model' is not defined

## === cell 34
def create_sub_df(test_df, predictions, length):
    sub_df = test_df.drop(tokenize_cols + ['index','seq_scored'], axis = 1) #first drop feature and irrelevant columns
    sub_df['seqpos'] = sub_df.apply(lambda row: list(range(row['seq_length'])), axis = 1)
    sub_df = unpack_df_lists(sub_df, 'seqpos') #now unroll
    sub_df['id_seqpos'] = sub_df.apply(lambda row: row['id'] + '_' + str(row['seqpos']), axis = 1)
    
    
    def pad_pred(p, i):
        '''
        p is the prediction
        i is final length
        fills unscored rows with 0s
        '''
        start = list(p) #first elements
        padding = [0] * (i - len(start)) #add padding to make up difference
        return start + padding

    pred_df = pd.DataFrame([[pad_pred(l, length) for l in e] for e in predictions], columns = target_cols) #here, each row holds a list in each column
    pred_df = unpack_df_lists(pred_df, target_cols) #then unroll those lists
    pred_df.index = sub_df.index
    pred_df = pred_df.reset_index()

    sub_df = sub_df.reset_index()
    sub_df = sub_df[['id_seqpos']]
    
    print(sub_df.shape, pred_df.shape)
    
    sub_df = sub_df.join(pred_df).set_index('index')

    
    return sub_df


## === cell 35

sub_public = create_sub_df(test_public, test_pred_public, 107)
sub_public


## --- ERROR in cell 35, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1591648568.py in <cell line: 0>()
      1 #create submission dataframe for public test data
      2 
----> 3 sub_public = create_sub_df(test_public, test_pred_public, 107)
      4 sub_public

NameError: name 'test_pred_public' is not defined

## === cell 36
"""
test_private = test_df[test_df['seq_length'] == 130]

sub_private = test_private[['id', 'seq_length']]
sub_private['seqpos'] = sub_private.apply(lambda row: list(range(row['seq_length'])), axis = 1) #create seqpos column


test_pred_private = model.predict(X_)

for c in target_cols: #fill in targets with 0s
    sub_private[c] = sub_private.apply(lambda row: [0] * row['seq_length'], axis = 1)

sub_private = unpack_df_lists(sub_private, ['seqpos'] + target_cols) #unroll lists
sub_private['id_seqpos'] = sub_private.apply(lambda row: row["id"]+'_'+str(row["seqpos"]), axis = 1) #create id_seqpos column
sub_private = sub_private.drop(['id','seq_length','seqpos'], axis = 1)
sub_private = sub_private[['id_seqpos'] + target_cols] #rearrange column order
"""


## === cell 37

sub_private = create_sub_df(test_private, test_pred_private, 130)
sub_private


## --- ERROR in cell 37, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3703929723.py in <cell line: 0>()
      1 #create submission dataframe for private data
      2 
----> 3 sub_private = create_sub_df(test_private, test_pred_private, 130)
      4 sub_private

NameError: name 'test_pred_private' is not defined

## === cell 38

sub_df = pd.concat([sub_public, sub_private]).convert_dtypes()
sub_df


## --- ERROR in cell 38, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1088416219.py in <cell line: 0>()
      1 #concat public and private predictions into single submission dataframe
      2 
----> 3 sub_df = pd.concat([sub_public, sub_private]).convert_dtypes()
      4 sub_df

NameError: name 'sub_public' is not defined

## === cell 39
sub_df.to_csv('submission.csv', index = False)


## --- ERROR in cell 39, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1903632625.py in <cell line: 0>()
----> 1 sub_df.to_csv('submission.csv', index = False)

NameError: name 'sub_df' is not defined
