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

3.8

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

# 4. Code solution

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

def feature_engineer(df, train = True, **kwargs):
    
    
    unpack_cols = ['reactivity_error', 'deg_error_Mg_pH10', 'deg_error_pH10',
       'deg_error_Mg_50C', 'deg_error_50C', 'reactivity', 'deg_Mg_pH10',
       'deg_pH10', 'deg_Mg_50C', 'deg_50C'] #only need to unpack things in training set
    
    if train:
        data = unpack_df_lists(df, unpack_cols)
    else: #if test data, need to add rows manually
        data = df.copy()
        data['temp'] = data.apply(lambda row: [0] * row['seq_length'], axis = 1) #adds temp column with list-like elements, of len(seq_scored) for that row 
        data = unpack_df_lists(data, 'temp') #unpack to right length using this function
        del data['temp'] #delete the temp column
        
    data['seqpos'] = 1
    data['seqpos'] = data.groupby('id').cumsum()['seqpos'] - 1
    
    seq_temp = pd.concat([data['sequence'],data['seqpos']], axis = 1)
    data['nucleotide'] = seq_temp.apply(lambda row: row['sequence'][row['seqpos']], axis = 1) #get base at seqpos in sequence string
    
    loop_temp = pd.concat([data['predicted_loop_type'],data['seqpos']], axis = 1)
    data['pred_loop_seqpos'] = loop_temp.apply(lambda row: row['predicted_loop_type'][row['seqpos']], axis = 1) #get type at seqpos in predicted_loop_type string 
    
    data = pd.get_dummies(data, columns = ['nucleotide','pred_loop_seqpos']) #do one-hot encoding on predicted_loop_type & nucleotide column
    
    return data


## === cell 10
sequences = train_df["sequence"].values

print(sequences)

sequences.shape


## === cell 11

tokenize_cols = ['sequence', 'structure', 'predicted_loop_type']
def tokenize_df(df, tokenizer, cols = tokenize_cols):
    '''
    tokenizer is a tensorflow keras Tokenizer that has been already fitted on text
    '''
    data = df.copy()
    for c in tokenize_cols:
        data[c] = tokenizer.texts_to_sequences(data[c])
    return data


## === cell 12
import os
import sys
import site
import subprocess

try:
    import google.protobuf  # noqa: F401
    from google.protobuf import __version__ as _pb_ver
except Exception:
    _pb_ver = None


def _pb_major(ver):
    try:
        return int(str(ver).split(".")[0])
    except Exception:
        return None


if _pb_ver is None or (_pb_major(_pb_ver) is not None and _pb_major(_pb_ver) >= 5):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "protobuf<5"])

for m in list(sys.modules.keys()):
    if m.startswith("google.protobuf"):
        del sys.modules[m]

try:
    sp = site.getsitepackages()
except Exception:
    sp = []
if sp:
    tf_pkg_path = os.path.join(sp[0], "tensorflow")
    if os.path.isdir(tf_pkg_path) and tf_pkg_path not in sys.path:
        sys.path.insert(0, tf_pkg_path)

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "3")

from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer(filters=None, lower=False, char_level=True)
tokenizer.fit_on_texts("().ACGUBEHIMSX")

temp = train_df[train_df["SN_filter"] == 1]
temp = tokenize_df(temp, tokenizer)

train_df = temp


## === cell 14
import matplotlib.pyplot as plt
import seaborn as sns

_base_drop = [
    "index",
    "id",
    "sequence",
    "structure",
    "predicted_loop_type",
    "seq_length",
    "seq_scored",
]
_numeric_df = train_df.drop(_base_drop, axis=1, errors="ignore").select_dtypes(
    include=[np.number]
)

corr_data = _numeric_df.corr()

corr_data


## === cell 15
mask = np.ma.masked_inside(corr_data.values, -0.15, 0.15).mask #get most powerful features

plt.figure(figsize = (13,13))
sns.heatmap(corr_data, annot = True, mask = mask)


## === cell 16
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

        y_true & y_pred have shapes (n, 5), where n is # of id_seqpos combos
        '''
        from sklearn.metrics import mean_squared_error
        y_true = np.array(y_true) #convert to np for convenience
        y_pred = np.array(y_pred)

        
        y_true = y_true[:, list(col_dict.values())]
        y_pred = y_pred[:, list(col_dict.values())]
        

        metric = mean_squared_error(y_true, y_pred, squared = False, multioutput = multi)
        return metric
    
    def loss_tf(y_true, y_pred):
        import tensorflow as tf
        import tensorflow.keras.backend as tf_kb
        
        
        
        
        for c in unscored:
            y_pred[:, c] = y_true[:, c]
        
        colwise_mse = tf_kb.mean(tf_kb.square(y_true - y_pred)) #first take average squared difference over row examples (shape 1x3)
        return tf_kb.mean(tf_kb.sqrt(colwise_mse)) #then sqrt and take average over columns (shape 1x1 / scalar)
        
    if not use_tf:
        return loss
    else:
        return loss_tf


## === cell 17

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
X_train = np.stack(X_train.values, axis = 0) #might not work for test data, since there is two different shapes

y_train = (train_df[target_cols]
               .apply(lambda row: [e for e in row], axis = 1) #concatenates each element in row to list
                .apply(lambda e : np.array(e)) #creates 2d numpy array from those elements
          )
y_train = np.stack(y_train.values, axis = 0)

'''
#maybe can use this as example weights -- higher signal_to_noise means higher weight?
#probably gotta make sure to cap the weight though, otherwise training dominated by top signal_to_noise
signal_to_noise = train_df[signal_cols]  #not necessary anymore
'''


np.array(train_df.drop(train_drop_cols, axis = 1).iloc[0].explode()).reshape(3,-1).astype('int')

'''
l = []
for arr in X_train.values[0]:
    l.append(arr)
    
np.array(l)'''


X_train


## === cell 18
y_train


## === cell 19
import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor


def make_model():
    
    EMBEDDING_PARAMS = {'input_dim': len(tokenizer.word_index) + 1,
                        'output_dim': 100, #play around with this
                       }
    
    shape = (3,None) #3 sequences of unknown length
    
    inputs = tf.keras.Input(shape = shape)
    embed = layers.Embedding(**EMBEDDING_PARAMS)(inputs)
    
    rnn_layer = layers.Bidirectional(layers.LSTM(30, return_sequences = True)) #define the rnn type to use
    
    rnn_layers = []
    for i in range(embed.shape[1]):
        rnn_layers.append(rnn_layer(embed[:,i]))
    
    x = layers.Concatenate()(rnn_layers)
    x = layers.Dense(100, activation = 'relu')(x)
    x = layers.Dense(5, activation = 'linear')(x) #need output layer of 5 x seq_scored
    
    x = tf.transpose(x, (0,2,1)) #reshape output so it makes more sense -- 5 layers of seq_scored length
    print(x.shape)
    x = x[:,:,:68] #compact sequence of 107 into 68
    
    model = tf.keras.Model(inputs = inputs, outputs = x)
    
        
    optimizer = 'adam'
    loss = 'mse'
    
    model.compile(optimizer = optimizer, loss = loss, metrics = [])
    
    
    return model


## --- ERROR in cell 19, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mModuleNotFoundError[0m                       Traceback (most recent call last)
[0;32m/tmp/ipykernel_12/428219339.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m      1[0m [0;32mimport[0m [0mtensorflow[0m [0;32mas[0m [0mtf[0m[0;34m[0m[0;34m[0m[0m
[1;32m      2[0m [0;32mimport[0m [0mtensorflow[0m[0;34m.[0m[0mkeras[0m[0;34m.[0m[0mlayers[0m [0;32mas[0m [0mlayers[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 3[0;31m [0;32mfrom[0m [0mtensorflow[0m[0;34m.[0m[0mkeras[0m[0;34m.[0m[0mwrappers[0m[0;34m.[0m[0mscikit_learn[0m [0;32mimport[0m [0mKerasRegressor[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      4[0m [0;34m[0m[0m
[1;32m      5[0m [0;31m#model should be highly generalizable, given limited public LB test data[0m[0;34m[0m[0;34m[0m[0m

[0;31mModuleNotFoundError[0m: No module named 'tensorflow.keras.wrappers.scikit_learn'

## === cell 20

from tensorflow.keras.wrappers.scikit_learn import KerasRegressor


TF_FITPARAMS = {'epochs': 100,
               'batch_size':100}
fp = TF_FITPARAMS


model = make_model()
model.summary()

print(X_train.shape, y_train.shape)

history = model.fit(X_train, y_train, **fp)
