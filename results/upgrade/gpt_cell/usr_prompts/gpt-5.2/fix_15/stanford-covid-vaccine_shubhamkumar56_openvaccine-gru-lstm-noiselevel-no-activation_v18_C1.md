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
tqdm==4.67.1

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
import warnings

warnings.filterwarnings("ignore")

import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
for _k in [
    "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION",
    "PROTOCOL_BUFFERS_PYTHON_USE_C_DESCRIPTORS",
]:
    os.environ.pop(_k, None)

import pandas as pd, numpy as np
import math, json, gc, random, sys
from matplotlib import pyplot as plt
from tqdm import tqdm

try:
    import google.protobuf as _pb
    from packaging.version import Version as _V

    if _V(getattr(_pb, "__version__", "0")) >= _V("5.0.0"):
        import subprocess

        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "protobuf<5"]
        )
        import importlib

        importlib.invalidate_caches()
        if "google.protobuf" in sys.modules:
            importlib.reload(sys.modules["google.protobuf"])
except Exception:
    pass

import tensorflow as tf

try:
    import tensorflow_addons as tfa
except Exception:
    tfa = None

import tensorflow.keras.backend as K
import tensorflow.keras.layers as L
from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.keras.layers import LSTM

from sklearn.model_selection import train_test_split, KFold

print("set up complete!")


## === cell 1
train = pd.read_json('/kaggle/input/stanford-covid-vaccine/train.json', lines=True)
test = pd.read_json('/kaggle/input/stanford-covid-vaccine/test.json', lines=True)
sample_sub = pd.read_csv('/kaggle/input/stanford-covid-vaccine/sample_submission.csv')


print ("Data Load Complete")


## === cell 2
print(train.shape)
if ~ train.isnull().values.any(): print('No missing values')
train.head()


## === cell 3
def add_list(test_list1, test_list2):
    res_list = [] 
    for i in range(0, len(test_list1)): 
        res_list.append(test_list1[i] + test_list2[i])
    return res_list

def subtract_list(test_list1, test_list2):
    res_list = [] 
    for i in range(0, len(test_list1)): 
        res_list.append(test_list1[i] - test_list2[i])
    return res_list


## === cell 4

train['reactivity_over'] = train.apply(lambda x: add_list(x.reactivity,x.reactivity_error), axis=1)
train['deg_Mg_pH10_over'] = train.apply(lambda x: add_list(x.deg_Mg_pH10,x.deg_error_Mg_pH10), axis=1)
train['deg_pH10_over'] = train.apply(lambda x: add_list(x.deg_pH10,x.deg_error_pH10), axis=1)
train['deg_Mg_50C_over'] = train.apply(lambda x: add_list(x.deg_Mg_50C,x.deg_error_Mg_50C), axis=1)
train['deg_50C_over'] = train.apply(lambda x: add_list(x.deg_50C,x.deg_error_50C), axis=1)

train['reactivity_under'] = train.apply(lambda x: subtract_list(x.reactivity,x.reactivity_error), axis=1)
train['deg_Mg_pH10_under'] = train.apply(lambda x: subtract_list(x.deg_Mg_pH10,x.deg_error_Mg_pH10), axis=1)
train['deg_pH10_under'] = train.apply(lambda x: subtract_list(x.deg_pH10,x.deg_error_pH10), axis=1)
train['deg_Mg_50C_under'] = train.apply(lambda x: subtract_list(x.deg_Mg_50C,x.deg_error_Mg_50C), axis=1)
train['deg_50C_under'] = train.apply(lambda x: subtract_list(x.deg_50C,x.deg_error_50C), axis=1)

print ("Additional target variables created!")


## === cell 5
train.head()


## === cell 6
print(test.shape)
if ~ test.isnull().values.any(): print('No missing values')
test.head()


## === cell 7
print(sample_sub.shape)
if ~ sample_sub.isnull().values.any(): print('No missing values')
sample_sub.head()


## === cell 8
target_cols_actual = ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']
target_cols_over = ['reactivity_over', 'deg_Mg_pH10_over', 'deg_pH10_over', 'deg_Mg_50C_over', 'deg_50C_over']
target_cols_under = ['reactivity_under', 'deg_Mg_pH10_under', 'deg_pH10_under', 'deg_Mg_50C_under', 'deg_50C_under']


## === cell 9
token2int = {x:i for i, x in enumerate('().ACGUBEHIMSX')}


## === cell 10
token2int['U']


## === cell 11
cols=['sequence', 'structure', 'predicted_loop_type']
train[cols].applymap(lambda seq: [token2int[x] for x in seq])


## === cell 12
def preprocess_inputs(df, cols=['sequence', 'structure', 'predicted_loop_type']):
    return np.transpose(
        np.array(
            df[cols]
            .applymap(lambda seq: [token2int[x] for x in seq])
            .values
            .tolist()
        ),
        (0, 2, 1)
    )


## === cell 13
train_inputs = preprocess_inputs(train[train.signal_to_noise > 1])
train_y_actual = np.array(train[train.signal_to_noise > 1][target_cols_actual].values.tolist()).transpose((0, 2, 1))
train_y_over = np.array(train[train.signal_to_noise > 1][target_cols_over].values.tolist()).transpose((0, 2, 1))
train_y_under = np.array(train[train.signal_to_noise > 1][target_cols_under].values.tolist()).transpose((0, 2, 1))


## === cell 14
print (train_inputs.shape)

print (train_y_actual.shape)
print (train_y_over.shape)
print (train_y_under.shape)


## === cell 15
inputs = tf.keras.layers.Input(shape=(107, 3))
embed = tf.keras.layers.Embedding(input_dim=len(token2int), output_dim=75)(inputs)

reshaped = tf.keras.layers.Reshape((107, 75 * 3))(embed)


## === cell 16
embed.shape


## === cell 17
def gru_layer(hidden_dim, dropout):
    return tf.keras.layers.Bidirectional(
                                tf.keras.layers.GRU(hidden_dim,
                                dropout=dropout,
                                return_sequences=True,
                                kernel_initializer = 'orthogonal'))

def lstm_layer(hidden_dim, dropout):
    return tf.keras.layers.Bidirectional(
                                tf.keras.layers.LSTM(hidden_dim,
                                dropout=dropout,
                                return_sequences=True,
                                kernel_initializer = 'orthogonal'))

def build_model(gru=False,seq_len=107, pred_len=68, dropout=0.5,
                embed_dim=100, hidden_dim=128):
    
    inputs = tf.keras.layers.Input(shape=(seq_len, 3))

    embed = tf.keras.layers.Embedding(input_dim=len(token2int), output_dim=embed_dim)(inputs)
    reshaped = tf.reshape(
        embed, shape=(-1, embed.shape[1],  embed.shape[2] * embed.shape[3]))
    
    reshaped = tf.keras.layers.SpatialDropout1D(.2)(reshaped)
    
    if gru:
        hidden = gru_layer(hidden_dim, dropout)(reshaped)
        hidden = gru_layer(hidden_dim, dropout)(hidden)
        hidden = gru_layer(hidden_dim, dropout)(hidden)
        
    else:
        hidden = lstm_layer(hidden_dim, dropout)(reshaped)
        hidden = lstm_layer(hidden_dim, dropout)(hidden)
        hidden = lstm_layer(hidden_dim, dropout)(hidden)
    
    truncated = hidden[:, :pred_len]
    
    out = tf.keras.layers.Dense(5, activation='linear')(truncated)

    model = tf.keras.Model(inputs=inputs, outputs=out)

    adam = tf.optimizers.Adam()
    radam = tfa.optimizers.RectifiedAdam()
    lookahead = tfa.optimizers.Lookahead(adam, sync_period=6)
    ranger = tfa.optimizers.Lookahead(radam, sync_period=6)
    
    model.compile(optimizer = adam, loss='mse')
    
    return model

print ("Model structure defined")


## === cell 18


def lstm_model (seq_len = 107,output_dim = 100,dropout = 0.5, pred_len = 68):
    
    
    inputs = tf.keras.layers.Input(shape=(seq_len, 3))

    embed = tf.keras.layers.Embedding(input_dim=len(token2int), output_dim=output_dim)(inputs)
    reshaped = tf.keras.layers.Reshape((seq_len,3*output_dim), input_shape=(seq_len, 3, output_dim))(embed)
    


    hidden = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM (128,  dropout = dropout,kernel_initializer = 'orthogonal', return_sequences = True ))(reshaped)
    hidden = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM (128,  dropout = dropout,kernel_initializer = 'orthogonal', return_sequences = True ))(hidden)
    hidden = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM (128,  dropout = dropout,kernel_initializer = 'orthogonal', return_sequences = True ))(hidden)
    truncated = hidden[:,:pred_len]
    output = tf.keras.layers.Dense(5, activation='linear')(truncated)
    
    model = tf.keras.Model(inputs=inputs, outputs=output)

    adam = tf.optimizers.Adam(learning_rate = 0.01, decay = 0.0001)

    model.compile(loss='mse',
                optimizer= adam ,
                )
    return model


## === cell 19
train_data, val_data, train_labels, val_labels = train_test_split(train_inputs, train_y_over,
                                                                     test_size=.2, random_state=4)


## === cell 20
lr_callback = tf.keras.callbacks.ReduceLROnPlateau()


## === cell 21
smpl_lstm = lstm_model(seq_len = 107,output_dim = 100,dropout = 0.5, pred_len = 68)
sv_smpl_lstm = tf.keras.callbacks.ModelCheckpoint('model_smpl_lstm.h5')


## === cell 22
smpl_lstm.summary()


## === cell 23
history_smpl_lstm = smpl_lstm.fit(
    train_data, train_labels, 
    validation_data=(val_data,val_labels),
    batch_size=64,
    epochs=80,
    callbacks=[lr_callback,sv_smpl_lstm],
    verbose = 2
)

print(f"Min training loss={min(history_smpl_lstm.history['loss'])}, min validation loss={min(history_smpl_lstm.history['val_loss'])}")


## === cell 24
def build_model(
    gru=False, seq_len=107, pred_len=68, dropout=0.5, embed_dim=100, hidden_dim=128
):
    inputs = tf.keras.layers.Input(shape=(seq_len, 3))

    embed = tf.keras.layers.Embedding(input_dim=len(token2int), output_dim=embed_dim)(
        inputs
    )
    reshaped = tf.keras.layers.Reshape((seq_len, embed_dim * 3))(embed)

    reshaped = tf.keras.layers.SpatialDropout1D(0.2)(reshaped)

    if gru:
        hidden = gru_layer(hidden_dim, dropout)(reshaped)
        hidden = gru_layer(hidden_dim, dropout)(hidden)
        hidden = gru_layer(hidden_dim, dropout)(hidden)
    else:
        hidden = lstm_layer(hidden_dim, dropout)(reshaped)
        hidden = lstm_layer(hidden_dim, dropout)(hidden)
        hidden = lstm_layer(hidden_dim, dropout)(hidden)

    truncated = hidden[:, :pred_len]
    out = tf.keras.layers.Dense(5, activation="linear")(truncated)

    model = tf.keras.Model(inputs=inputs, outputs=out)

    adam = tf.optimizers.Adam()
    if tfa is not None:
        radam = tfa.optimizers.RectifiedAdam()
        lookahead = tfa.optimizers.Lookahead(adam, sync_period=6)
        ranger = tfa.optimizers.Lookahead(radam, sync_period=6)

    model.compile(optimizer=adam, loss="mse")
    return model


gru = build_model(gru=True)
sv_gru = tf.keras.callbacks.ModelCheckpoint("model_gru.h5")


## === cell 25
gru.summary()


## === cell 26
history_gru = gru.fit(
    train_data, train_labels, 
    validation_data=(val_data,val_labels),
    batch_size=64,
    epochs=70,
    callbacks=[lr_callback,sv_gru],
    verbose = 2
)

print(f"Min training loss={min(history_gru.history['loss'])}, min validation loss={min(history_gru.history['val_loss'])}")


## === cell 27
fig, ax = plt.subplots(1, 2,  figsize = (20, 10))

ax[0].plot(history_gru.history['loss'])
ax[0].plot(history_gru.history['val_loss'])



ax[1].plot(history_smpl_lstm.history['loss'])
ax[1].plot(history_smpl_lstm.history['val_loss'])

ax[0].set_title('GRU')

ax[1].set_title('SMPL_LSTM')

ax[0].legend(['train', 'validation'], loc = 'upper right')

ax[1].legend(['train', 'validation'], loc = 'upper right')

ax[0].set_ylabel('Loss')
ax[0].set_xlabel('Epoch')
ax[1].set_ylabel('Loss')
ax[1].set_xlabel('Epoch')


## === cell 28
public_df = test.query("seq_length == 107").copy()
private_df = test.query("seq_length == 130").copy()


def _preprocess_inputs_fixed(
    df, seq_len, cols=["sequence", "structure", "predicted_loop_type"]
):
    col_arrays = []
    for c in cols:
        arr = np.stack(
            df[c].map(lambda s: [token2int[ch] for ch in s]).to_list(), axis=0
        )
        if arr.shape[1] != seq_len:
            raise ValueError(
                f"Unexpected {c} length: got {arr.shape[1]}, expected {seq_len}"
            )
        col_arrays.append(arr)
    return np.stack(col_arrays, axis=-1)


public_inputs = _preprocess_inputs_fixed(public_df, seq_len=107)
private_inputs = _preprocess_inputs_fixed(private_df, seq_len=130)

print("Test data prepared!")


## --- ERROR in cell 28, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mValueError[0m                                Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/612133550.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     22[0m [0;34m[0m[0m
[1;32m     23[0m [0mpublic_inputs[0m [0;34m=[0m [0m_preprocess_inputs_fixed[0m[0;34m([0m[0mpublic_df[0m[0;34m,[0m [0mseq_len[0m[0;34m=[0m[0;36m107[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 24[0;31m [0mprivate_inputs[0m [0;34m=[0m [0m_preprocess_inputs_fixed[0m[0;34m([0m[0mprivate_df[0m[0;34m,[0m [0mseq_len[0m[0;34m=[0m[0;36m130[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     25[0m [0;34m[0m[0m
[1;32m     26[0m [0mprint[0m[0;34m([0m[0;34m"Test data prepared!"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/tmp/ipykernel_11/612133550.py[0m in [0;36m_preprocess_inputs_fixed[0;34m(df, seq_len, cols)[0m
[1;32m     10[0m     [0mcol_arrays[0m [0;34m=[0m [0;34m[[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[1;32m     11[0m     [0;32mfor[0m [0mc[0m [0;32min[0m [0mcols[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 12[0;31m         arr = np.stack(
[0m[1;32m     13[0m             [0mdf[0m[0;34m[[0m[0mc[0m[0;34m][0m[0;34m.[0m[0mmap[0m[0;34m([0m[0;32mlambda[0m [0ms[0m[0;34m:[0m [0;34m[[0m[0mtoken2int[0m[0;34m[[0m[0mch[0m[0;34m][0m [0;32mfor[0m [0mch[0m [0;32min[0m [0ms[0m[0;34m][0m[0;34m)[0m[0;34m.[0m[0mto_list[0m[0;34m([0m[0;34m)[0m[0;34m,[0m [0maxis[0m[0;34m=[0m[0;36m0[0m[0;34m[0m[0;34m[0m[0m
[1;32m     14[0m         )

[0;32m/usr/local/lib/python3.11/dist-packages/numpy/core/shape_base.py[0m in [0;36mstack[0;34m(arrays, axis, out, dtype, casting)[0m
[1;32m    443[0m     [0marrays[0m [0;34m=[0m [0;34m[[0m[0masanyarray[0m[0;34m([0m[0marr[0m[0;34m)[0m [0;32mfor[0m [0marr[0m [0;32min[0m [0marrays[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[1;32m    444[0m     [0;32mif[0m [0;32mnot[0m [0marrays[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 445[0;31m         [0;32mraise[0m [0mValueError[0m[0;34m([0m[0;34m'need at least one array to stack'[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    446[0m [0;34m[0m[0m
[1;32m    447[0m     [0mshapes[0m [0;34m=[0m [0;34m{[0m[0marr[0m[0;34m.[0m[0mshape[0m [0;32mfor[0m [0marr[0m [0;32min[0m [0marrays[0m[0;34m}[0m[0;34m[0m[0;34m[0m[0m

[0;31mValueError[0m: need at least one array to stack

## === cell 29
gru_short = build_model(gru=True, seq_len=107, pred_len=107)
gru_long = build_model(gru=True, seq_len=130, pred_len=130)
lstm_short = lstm_model(seq_len = 107,output_dim = 100,dropout = 0.5, pred_len = 107)
lstm_long = lstm_model(seq_len = 130,output_dim = 100,dropout = 0.5, pred_len = 130)

gru_short.load_weights('model_gru.h5')
gru_long.load_weights('model_gru.h5')
lstm_short.load_weights('model_smpl_lstm.h5')
lstm_long.load_weights('model_smpl_lstm.h5')

print ("models built and trained")
