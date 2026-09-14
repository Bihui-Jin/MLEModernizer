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

import pandas as pd, numpy as np
import math, json, gc, random, os, sys
from matplotlib import pyplot as plt
from tqdm import tqdm

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

try:
    import importlib
    from importlib import metadata as _importlib_metadata

    _pb_ver = _importlib_metadata.version("protobuf")
    _major = int(_pb_ver.split(".")[0])
    if _major >= 5:
        import subprocess

        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "protobuf<5"]
        )
        for _m in list(sys.modules.keys()):
            if _m.startswith("google.protobuf"):
                sys.modules.pop(_m, None)
        importlib.invalidate_caches()
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
print(test.shape)
if ~ test.isnull().values.any(): print('No missing values')
test.head()


## === cell 4
print(sample_sub.shape)
if ~ sample_sub.isnull().values.any(): print('No missing values')
sample_sub.head()


## === cell 5
target_cols = ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']


## === cell 6
token2int = {x:i for i, x in enumerate('().ACGUBEHIMSX')}


## === cell 7
token2int['U']


## === cell 8
cols=['sequence', 'structure', 'predicted_loop_type']
train[cols].applymap(lambda seq: [token2int[x] for x in seq])


## === cell 9
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


## === cell 10
train_inputs = preprocess_inputs(train[train.signal_to_noise > 1])
train_y = np.array(train[train.signal_to_noise > 1][target_cols].values.tolist()).transpose((0, 2, 1))


## === cell 11
print (train_inputs.shape)

print (train_y.shape)


## === cell 12
inputs = tf.keras.layers.Input(shape=(107, 3))
embed = tf.keras.layers.Embedding(input_dim=len(token2int), output_dim=75)(inputs)

reshaped = tf.keras.layers.Reshape(
    (embed.shape[1], int(embed.shape[2] * embed.shape[3]))
)(embed)


## === cell 13
embed.shape


## === cell 14
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
                embed_dim=75, hidden_dim=128):
    
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


## === cell 15


def lstm_model (seq_len = 107,output_dim = 75,dropout = 0.5, pred_len = 68):
    
    
    inputs = tf.keras.layers.Input(shape=(seq_len, 3))

    embed = tf.keras.layers.Embedding(input_dim=len(token2int), output_dim=output_dim)(inputs)
    reshaped = tf.keras.layers.Reshape((seq_len,225), input_shape=(seq_len, 3, 75))(embed)
    


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


## === cell 16
train_data, val_data, train_labels, val_labels = train_test_split(train_inputs, train_y,
                                                                     test_size=.2, random_state=4)


## === cell 17
lr_callback = tf.keras.callbacks.ReduceLROnPlateau()


## === cell 18
smpl_lstm = lstm_model(seq_len = 107,output_dim = 75,dropout = 0.5, pred_len = 68)
sv_smpl_lstm = tf.keras.callbacks.ModelCheckpoint('model_smpl_lstm.h5')


## === cell 19
smpl_lstm.summary()


## === cell 20
history_smpl_lstm = smpl_lstm.fit(
    train_data, train_labels, 
    validation_data=(val_data,val_labels),
    batch_size=64,
    epochs=80,
    callbacks=[lr_callback,sv_smpl_lstm],
    verbose = 2
)

print(f"Min training loss={min(history_smpl_lstm.history['loss'])}, min validation loss={min(history_smpl_lstm.history['val_loss'])}")


## === cell 21
def build_model(
    gru=False, seq_len=107, pred_len=68, dropout=0.5, embed_dim=75, hidden_dim=128
):

    inputs = tf.keras.layers.Input(shape=(seq_len, 3))

    embed = tf.keras.layers.Embedding(input_dim=len(token2int), output_dim=embed_dim)(
        inputs
    )

    reshaped = tf.keras.layers.Reshape((seq_len, int(embed_dim * 3)))(embed)

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
    else:
        radam = None
        lookahead = None
        ranger = None

    model.compile(optimizer=adam, loss="mse")

    return model


gru = build_model(gru=True)
sv_gru = tf.keras.callbacks.ModelCheckpoint("model_gru.h5")


## === cell 22
gru.summary()


## === cell 23
history_gru = gru.fit(
    train_data, train_labels, 
    validation_data=(val_data,val_labels),
    batch_size=64,
    epochs=70,
    callbacks=[lr_callback,sv_gru],
    verbose = 2
)

print(f"Min training loss={min(history_gru.history['loss'])}, min validation loss={min(history_gru.history['val_loss'])}")


## === cell 24
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


## === cell 25
public_df = test.query("seq_length == 107").copy()
private_df = test.query("seq_length == 130").copy()

_req_cols = ["sequence", "structure", "predicted_loop_type"]
public_df[_req_cols] = public_df[_req_cols].astype(str)
private_df[_req_cols] = private_df[_req_cols].astype(str)

_pad_map = {"sequence": "A", "structure": ".", "predicted_loop_type": "S"}

_valid_tokens = set(token2int.keys())
_unknown_token = "X" if "X" in token2int else "A"


def _sanitize_and_fixlen(s: str, L: int, pad: str) -> str:
    s = ("" if s is None else str(s)).strip()
    s = s[:L].ljust(L, pad)
    if not _valid_tokens.issuperset(set(s)):
        s = "".join(ch if ch in _valid_tokens else _unknown_token for ch in s)
    return s


for _df, _L in ((public_df, 107), (private_df, 130)):
    for _c in _req_cols:
        _pad = _pad_map[_c]
        _df[_c] = _df[_c].map(lambda v: _sanitize_and_fixlen(v, _L, _pad))


def _to_token_tensor(df, L: int, cols=_req_cols):
    n = len(df)
    arr = np.empty((n, L, len(cols)), dtype=np.int32)
    for j, c in enumerate(cols):
        arr[:, :, j] = np.array(
            [[token2int[ch] for ch in s] for s in df[c].tolist()], dtype=np.int32
        )
    return arr


public_inputs = _to_token_tensor(public_df, 107)
private_inputs = _to_token_tensor(private_df, 130)

print("Test data prepared!")


## --- ERROR in cell 25, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mValueError[0m                                Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/2428707884.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     41[0m [0;34m[0m[0m
[1;32m     42[0m [0mpublic_inputs[0m [0;34m=[0m [0m_to_token_tensor[0m[0;34m([0m[0mpublic_df[0m[0;34m,[0m [0;36m107[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 43[0;31m [0mprivate_inputs[0m [0;34m=[0m [0m_to_token_tensor[0m[0;34m([0m[0mprivate_df[0m[0;34m,[0m [0;36m130[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     44[0m [0;34m[0m[0m
[1;32m     45[0m [0mprint[0m[0;34m([0m[0;34m"Test data prepared!"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/tmp/ipykernel_11/2428707884.py[0m in [0;36m_to_token_tensor[0;34m(df, L, cols)[0m
[1;32m     34[0m     [0;32mfor[0m [0mj[0m[0;34m,[0m [0mc[0m [0;32min[0m [0menumerate[0m[0;34m([0m[0mcols[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m     35[0m         [0;31m# Each entry is already sanitized to length L and to valid tokens[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 36[0;31m         arr[:, :, j] = np.array(
[0m[1;32m     37[0m             [0;34m[[0m[0;34m[[0m[0mtoken2int[0m[0;34m[[0m[0mch[0m[0;34m][0m [0;32mfor[0m [0mch[0m [0;32min[0m [0ms[0m[0;34m][0m [0;32mfor[0m [0ms[0m [0;32min[0m [0mdf[0m[0;34m[[0m[0mc[0m[0;34m][0m[0;34m.[0m[0mtolist[0m[0;34m([0m[0;34m)[0m[0;34m][0m[0;34m,[0m [0mdtype[0m[0;34m=[0m[0mnp[0m[0;34m.[0m[0mint32[0m[0;34m[0m[0;34m[0m[0m
[1;32m     38[0m         )

[0;31mValueError[0m: could not broadcast input array from shape (0,) into shape (0,130)

## === cell 26
gru_short = build_model(gru=True, seq_len=107, pred_len=107)
gru_long = build_model(gru=True, seq_len=130, pred_len=130)
lstm_short = lstm_model(seq_len = 107,output_dim = 75,dropout = 0.5, pred_len = 107)
lstm_long = lstm_model(seq_len = 130,output_dim = 75,dropout = 0.5, pred_len = 130)

gru_short.load_weights('model_gru.h5')
gru_long.load_weights('model_gru.h5')
lstm_short.load_weights('model_smpl_lstm.h5')
lstm_long.load_weights('model_smpl_lstm.h5')

print ("models built and trained")
