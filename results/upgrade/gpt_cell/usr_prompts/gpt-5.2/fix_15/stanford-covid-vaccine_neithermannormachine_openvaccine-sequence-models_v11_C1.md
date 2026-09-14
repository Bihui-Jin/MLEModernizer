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
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")


class Tokenizer:
    def __init__(self, filters=None, lower=False, char_level=True):
        self.filters = filters
        self.lower = lower
        self.char_level = char_level
        self.word_index = {}

    def fit_on_texts(self, texts):
        if isinstance(texts, str):
            iterable = [texts]
        else:
            iterable = list(texts)

        chars = []
        for t in iterable:
            if t is None:
                continue
            s = str(t)
            if self.lower:
                s = s.lower()
            if self.char_level:
                chars.extend(list(s))
            else:
                chars.extend(s.split())

        uniq = sorted(set(chars))
        self.word_index = {ch: i + 1 for i, ch in enumerate(uniq)}

    def texts_to_sequences(self, texts):
        if isinstance(texts, str):
            texts = [texts]

        sequences = []
        for t in texts:
            if t is None:
                sequences.append([])
                continue
            s = str(t)
            if self.lower:
                s = s.lower()
            if self.char_level:
                sequences.append([self.word_index.get(ch, 0) for ch in s])
            else:
                sequences.append([self.word_index.get(tok, 0) for tok in s.split()])
        return sequences


tokenizer = Tokenizer(filters=None, lower=False, char_level=True)
tokenizer.fit_on_texts("().ACGUBEHIMSX")

temp = train_df[train_df["SN_filter"] == 1]
temp = tokenize_df(temp, tokenizer)

train_df = temp

train_df


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
import os
import sys
import subprocess
import importlib

try:
    import google.protobuf  # noqa: F401
    import pkgutil
    import pkg_resources

    pb_ver = pkg_resources.get_distribution("protobuf").version
    major = int(pb_ver.split(".", 1)[0])
except Exception:
    major = None

if major is None or major >= 5:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "protobuf<5"])
    for m in list(sys.modules.keys()):
        if m.startswith("google.protobuf"):
            del sys.modules[m]
    importlib.invalidate_caches()

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"

import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras.optimizers import Adam


def make_model():
    """
    Creates a tensorflow keras sequence model
    """
    EMBEDDING_PARAMS = {
        "input_dim": len(tokenizer.word_index) + 1,
        "output_dim": 100,  # play around with this number
    }

    shape = (
        3,
        None,
    )  # 3 sequences of unknown length (should all be same length though)

    seq_inputs = tf.keras.Input(shape=shape)  # shape (n, 3, seq_length)
    embed = layers.Embedding(**EMBEDDING_PARAMS)(
        seq_inputs
    )  # (n, 3, seq_length, output_dim)

    def rnn_layer(num_neurons):
        return layers.Bidirectional(
            layers.LSTM(num_neurons, return_sequences=True, dropout=0.2)
        )  # define the rnn type to use

    rnn_layers = []
    for i in range(embed.shape[1]):  # loop thru sequences
        r = rnn_layer(30)(embed[:, i])
        r = rnn_layer(30)(r)
        rnn_layers.append(
            r
        )  # r is shape (n, seq_length, num_rnn_neurons * 2 (b/c bidirectional))

    x = layers.Concatenate()(
        rnn_layers
    )  # concatenate the rnn layers for each sequence (n, seq_length, num_rnn_neurons * 2 * 3)
    x = layers.Dense(100, activation="relu")(x)  # (n, seq_length, 100)
    x = layers.Dense(5, activation="linear")(
        x
    )  # need output layer of 5 x seq_scored; shape (n, seq_length, 5)

    x = tf.transpose(
        x, (0, 2, 1)
    )  # reshape prediction for submitting output & computing loss -- (n, 5, seq_length)
    x = x[
        :, :, :-39
    ]  # compact sequence of 107/130 into 68/91 by removing last 39 elements (n, 5, seq_scored)

    model = tf.keras.Model(inputs=seq_inputs, outputs=x)

    optimizer = Adam(
        learning_rate=0.01
    )  # try different learning rates, optimizer hyperparameters
    loss = score(use_tf=True)

    model.compile(optimizer=optimizer, loss=loss, metrics=["mse"])

    return model


## === cell 21
"""
work on callbacks (lr scheduling, logging, etc)
"""
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping
from sklearn.model_selection import train_test_split

TF_FITPARAMS = {"epochs": 150, "batch_size": 100, "validation_batch_size": 50}
fp = TF_FITPARAMS


def schedule_func(epoch, lr):
    """
    function passed to LearningRateScheduler to determine learning rate at each epoch
    keeps lr constant until certain percent of epochs have elapsed, then exponentially decreases lr
    """
    if epoch < 50:
        return lr
    else:
        return lr * np.exp(-0.13)


callbacks = [
    LearningRateScheduler(schedule_func),
    EarlyStopping(monitor="val_loss", mode="min", min_delta=5e-5, patience=10),
]


def make_model():
    EMBEDDING_PARAMS = {
        "input_dim": len(tokenizer.word_index) + 1,
        "output_dim": 100,
    }

    shape = (3, None)  # 3 sequences of unknown length

    seq_inputs = tf.keras.Input(shape=shape)  # (n, 3, seq_length)
    embed = layers.Embedding(**EMBEDDING_PARAMS)(
        seq_inputs
    )  # (n, 3, seq_length, output_dim)

    def rnn_layer(num_neurons):
        return layers.Bidirectional(
            layers.LSTM(num_neurons, return_sequences=True, dropout=0.2)
        )

    rnn_layers = []
    for i in range(embed.shape[1]):
        r = rnn_layer(30)(embed[:, i])
        r = rnn_layer(30)(r)
        rnn_layers.append(r)

    x = layers.Concatenate()(rnn_layers)
    x = layers.Dense(100, activation="relu")(x)
    x = layers.Dense(5, activation="linear")(x)  # (n, seq_length, 5)

    x = layers.Permute((2, 1), name="permute_to_batch_5_seq")(x)  # (n, 5, seq_length)
    x = layers.Lambda(lambda t: t[:, :, :-39], name="crop_seq_scored")(
        x
    )  # (n, 5, seq_scored)

    model = tf.keras.Model(inputs=seq_inputs, outputs=x)

    optimizer = Adam(learning_rate=0.01)
    loss = score(use_tf=True)
    model.compile(optimizer=optimizer, loss=loss, metrics=["mse"])
    return model


model = make_model()

model.summary()
print(X_train.shape, y_train.shape)

X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.1)

get_ipython().run_line_magic(
    "time",
    "history = model.fit(X_tr, y_tr, validation_data = (X_val, y_val), callbacks = callbacks, **fp)",
)


## === cell 22
tf.keras.utils.plot_model(model, show_shapes=True)


## === cell 23
def save_model(model):
    '''
    WIP
    model is a fitted tensorflow keras model 
    '''
    
    pass


## === cell 24
import matplotlib.pyplot as plt

plt.figure(figsize=(13, 7))


all_metrics = set(history.history.keys()) - set(["lr"])  # plot learning rate seperately
for metric in all_metrics:
    metric_history = history.history[metric]
    plt.plot(metric_history, label=metric)
plt.legend()
plt.show()

if "lr" in history.history:
    plt.plot(history.history["lr"], label="lr")
    plt.legend()
    plt.show()


## === cell 25
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

try:
    from scikeras.wrappers import KerasRegressor  # type: ignore
except Exception:
    try:
        from tensorflow.keras.wrappers.scikit_learn import (
            KerasRegressor,
        )  # legacy fallback
    except Exception:
        KerasRegressor = None

param_grid = {}

model_sk = (
    KerasRegressor(build_fn=make_model) if KerasRegressor is not None else None
)  # WIP


## === cell 28
test_df = read_json('../input/stanford-covid-vaccine/test.json')

test_df


## === cell 29
test_public = test_df[test_df['seq_length'] == 107]
test_public = tokenize_df(test_public, tokenizer)

test_public


## === cell 30
test_private = test_df[test_df['seq_length'] == 130]
test_private = tokenize_df(test_private, tokenizer)


test_private


## === cell 31
X_test_public = (test_public.drop(drop_cols, axis = 1) #selects only relevant columns
                    .apply(lambda row: [e for e in row], axis = 1) #concatenates each element in row to list
                    .apply(lambda e : np.array(e)) #creates 2d numpy array from those elements
          )
X_test_public = np.stack(X_test_public.values, axis = 0) #shape (n,3,107)

X_test_public


## === cell 32
X_test_private = (
    test_private.drop(drop_cols, axis=1)  # selects only relevant columns
    .apply(
        lambda row: [e for e in row], axis=1
    )  # concatenates each element in row to list
    .apply(lambda e: np.array(e))  # creates 2d numpy array from those elements
)

if len(X_test_private) == 0:
    X_test_private = np.empty((0, 3, 130), dtype=np.int32)
else:
    X_test_private = np.stack(X_test_private.values, axis=0)  # shape (n,3,130)

X_test_private


## === cell 33

test_pred_public = model.predict(X_test_public) #shape (n,5,68)
test_pred_public


## === cell 34
if X_test_private.shape[0] == 0:
    test_pred_private = np.empty((0, 5, 91), dtype=np.float32)
else:
    test_pred_private = model.predict(X_test_private)  # shape (n,5,91)
test_pred_private


## === cell 35
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


## === cell 36

sub_public = create_sub_df(test_public, test_pred_public, 107)
sub_public


## === cell 37
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


## === cell 38


def create_sub_df(test_df, predictions, length):
    sub_df = test_df.drop(
        tokenize_cols + ["index", "seq_scored"], axis=1
    )  # first drop feature and irrelevant columns

    sub_df["seqpos"] = sub_df["seq_length"].map(lambda n: list(range(int(n))))

    sub_df = unpack_df_lists(sub_df, "seqpos")  # now unroll
    sub_df["id_seqpos"] = sub_df.apply(
        lambda row: row["id"] + "_" + str(row["seqpos"]), axis=1
    )

    def pad_pred(p, i):
        """
        p is the prediction
        i is final length
        fills unscored rows with 0s
        """
        start = list(p)  # first elements
        padding = [0] * (i - len(start))  # add padding to make up difference
        return start + padding

    pred_df = pd.DataFrame(
        [[pad_pred(l, length) for l in e] for e in predictions], columns=target_cols
    )  # here, each row holds a list in each column
    pred_df = unpack_df_lists(pred_df, target_cols)  # then unroll those lists
    pred_df.index = sub_df.index
    pred_df = pred_df.reset_index()

    sub_df = sub_df.reset_index()
    sub_df = sub_df[["id_seqpos"]]

    print(sub_df.shape, pred_df.shape)

    sub_df = sub_df.join(pred_df).set_index("index")

    return sub_df


sub_private = create_sub_df(test_private, test_pred_private, 130)
sub_private


## --- ERROR in cell 38, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mValueError[0m                                Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/3678313688.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     42[0m [0;34m[0m[0m
[1;32m     43[0m [0;34m[0m[0m
[0;32m---> 44[0;31m [0msub_private[0m [0;34m=[0m [0mcreate_sub_df[0m[0;34m([0m[0mtest_private[0m[0;34m,[0m [0mtest_pred_private[0m[0;34m,[0m [0;36m130[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     45[0m [0msub_private[0m[0;34m[0m[0;34m[0m[0m

[0;32m/tmp/ipykernel_11/3678313688.py[0m in [0;36mcreate_sub_df[0;34m(test_df, predictions, length)[0m
[1;32m     11[0m [0;34m[0m[0m
[1;32m     12[0m     [0msub_df[0m [0;34m=[0m [0munpack_df_lists[0m[0;34m([0m[0msub_df[0m[0;34m,[0m [0;34m"seqpos"[0m[0;34m)[0m  [0;31m# now unroll[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 13[0;31m     sub_df["id_seqpos"] = sub_df.apply(
[0m[1;32m     14[0m         [0;32mlambda[0m [0mrow[0m[0;34m:[0m [0mrow[0m[0;34m[[0m[0;34m"id"[0m[0;34m][0m [0;34m+[0m [0;34m"_"[0m [0;34m+[0m [0mstr[0m[0;34m([0m[0mrow[0m[0;34m[[0m[0;34m"seqpos"[0m[0;34m][0m[0;34m)[0m[0;34m,[0m [0maxis[0m[0;34m=[0m[0;36m1[0m[0;34m[0m[0;34m[0m[0m
[1;32m     15[0m     )

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py[0m in [0;36m__setitem__[0;34m(self, key, value)[0m
[1;32m   4299[0m             [0mself[0m[0;34m.[0m[0m_setitem_array[0m[0;34m([0m[0mkey[0m[0;34m,[0m [0mvalue[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m   4300[0m         [0;32melif[0m [0misinstance[0m[0;34m([0m[0mvalue[0m[0;34m,[0m [0mDataFrame[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 4301[0;31m             [0mself[0m[0;34m.[0m[0m_set_item_frame_value[0m[0;34m([0m[0mkey[0m[0;34m,[0m [0mvalue[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   4302[0m         elif (
[1;32m   4303[0m             [0mis_list_like[0m[0;34m([0m[0mvalue[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py[0m in [0;36m_set_item_frame_value[0;34m(self, key, value)[0m
[1;32m   4457[0m [0;34m[0m[0m
[1;32m   4458[0m         [0;32mif[0m [0mlen[0m[0;34m([0m[0mvalue[0m[0;34m.[0m[0mcolumns[0m[0;34m)[0m [0;34m>[0m [0;36m1[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 4459[0;31m             raise ValueError(
[0m[1;32m   4460[0m                 [0;34m"Cannot set a DataFrame with multiple columns to the single "[0m[0;34m[0m[0;34m[0m[0m
[1;32m   4461[0m                 [0;34mf"column {key}"[0m[0;34m[0m[0;34m[0m[0m

[0;31mValueError[0m: Cannot set a DataFrame with multiple columns to the single column id_seqpos

## === cell 39

sub_df = pd.concat([sub_public, sub_private]).convert_dtypes()
sub_df
