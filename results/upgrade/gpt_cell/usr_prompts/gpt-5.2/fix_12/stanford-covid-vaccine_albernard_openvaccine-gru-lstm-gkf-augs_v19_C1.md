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

import os, sys, subprocess, importlib

try:
    import google.protobuf  # noqa: F401

    try:
        import pkg_resources

        pb_ver = pkg_resources.get_distribution("protobuf").version
    except Exception:
        pb_ver = None

    def _major(v):
        try:
            return int(str(v).split(".", 1)[0])
        except Exception:
            return None

    if pb_ver is not None and _major(pb_ver) is not None and _major(pb_ver) >= 6:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "protobuf==5.28.3"]
        )
        for m in list(sys.modules):
            if m.startswith("google.protobuf") or m == "protobuf":
                sys.modules.pop(m, None)
        importlib.invalidate_caches()
except Exception:
    pass

import pandas as pd, numpy as np
import math, json, gc, random

from matplotlib import pyplot as plt
from tqdm import tqdm

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

import tensorflow as tf

try:
    import tensorflow_addons as tfa
except Exception:
    tfa = None

import tensorflow.keras.backend as K
import tensorflow.keras.layers as L

from sklearn.model_selection import train_test_split, KFold


## === cell 1
train = pd.read_json('/kaggle/input/stanford-covid-vaccine/train.json', lines=True)
test = pd.read_json('/kaggle/input/stanford-covid-vaccine/test.json', lines=True)
sample_sub = pd.read_csv('/kaggle/input/stanford-covid-vaccine/sample_submission.csv')


## === cell 2
target_cols = ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']


## === cell 3
token2int = {x:i for i, x in enumerate('().ACGUBEHIMSX')}


## === cell 4
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


## === cell 5
train_inputs = preprocess_inputs(train[train.signal_to_noise > 1])
train_labels = np.array(train[train.signal_to_noise > 1][target_cols].values.tolist()).transpose((0, 2, 1))


## === cell 6
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

def build_model(gru=1,seq_len=107, pred_len=68, dropout=0.5,
                embed_dim=75, hidden_dim=128):
    
    inputs = tf.keras.layers.Input(shape=(seq_len, 3))

    embed = tf.keras.layers.Embedding(input_dim=len(token2int), output_dim=embed_dim)(inputs)
    reshaped = tf.reshape(
        embed, shape=(-1, embed.shape[1],  embed.shape[2] * embed.shape[3]))
    
    reshaped = tf.keras.layers.SpatialDropout1D(.2)(reshaped)
    
    if gru==1:
        hidden = gru_layer(hidden_dim, dropout)(reshaped)
        hidden = gru_layer(hidden_dim, dropout)(hidden)
        hidden = gru_layer(hidden_dim, dropout)(hidden)
        
    elif gru==0:
        hidden = lstm_layer(hidden_dim, dropout)(reshaped)
        hidden = lstm_layer(hidden_dim, dropout)(hidden)
        hidden = lstm_layer(hidden_dim, dropout)(hidden)
        
    elif gru==3:
        hidden = gru_layer(hidden_dim, dropout)(reshaped)
        hidden = lstm_layer(hidden_dim, dropout)(hidden)
        hidden = lstm_layer(hidden_dim, dropout)(hidden)
        
    elif gru==4:
        hidden = lstm_layer(hidden_dim, dropout)(reshaped)
        hidden = gru_layer(hidden_dim, dropout)(hidden)
        hidden = gru_layer(hidden_dim, dropout)(hidden)
    
    truncated = hidden[:, :pred_len]
    
    out = tf.keras.layers.Dense(5, activation='linear')(truncated)

    model = tf.keras.Model(inputs=inputs, outputs=out)

    adam = tf.optimizers.Adam()
    radam = tfa.optimizers.RectifiedAdam()
    lookahead = tfa.optimizers.Lookahead(adam, sync_period=6)
    ranger = tfa.optimizers.Lookahead(radam, sync_period=6)
    
    model.compile(optimizer = adam, loss='mse')
    
    return model


## === cell 7
train_inputs, val_inputs, train_labels, val_labels = train_test_split(train_inputs, train_labels,
                                                                     test_size=.1, random_state=34)


## === cell 8
if tf.config.list_physical_devices('GPU') is not None:
    print('Training on GPU')


## === cell 9
lr_callback = tf.keras.callbacks.ReduceLROnPlateau()


## === cell 13
def gru_layer(hidden_dim, dropout):
    return tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(
            hidden_dim,
            dropout=dropout,
            return_sequences=True,
            kernel_initializer="orthogonal",
        )
    )


def lstm_layer(hidden_dim, dropout):
    return tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            hidden_dim,
            dropout=dropout,
            return_sequences=True,
            kernel_initializer="orthogonal",
        )
    )


def build_model(
    gru=1, seq_len=107, pred_len=68, dropout=0.5, embed_dim=75, hidden_dim=128
):

    inputs = tf.keras.layers.Input(shape=(seq_len, 3))

    embed = tf.keras.layers.Embedding(input_dim=len(token2int), output_dim=embed_dim)(
        inputs
    )

    reshaped = tf.keras.layers.Reshape((seq_len, 3 * embed_dim))(embed)

    reshaped = tf.keras.layers.SpatialDropout1D(0.2)(reshaped)

    if gru == 1:
        hidden = gru_layer(hidden_dim, dropout)(reshaped)
        hidden = gru_layer(hidden_dim, dropout)(hidden)
        hidden = gru_layer(hidden_dim, dropout)(hidden)

    elif gru == 0:
        hidden = lstm_layer(hidden_dim, dropout)(reshaped)
        hidden = lstm_layer(hidden_dim, dropout)(hidden)
        hidden = lstm_layer(hidden_dim, dropout)(hidden)

    elif gru == 3:
        hidden = gru_layer(hidden_dim, dropout)(reshaped)
        hidden = lstm_layer(hidden_dim, dropout)(hidden)
        hidden = lstm_layer(hidden_dim, dropout)(hidden)

    elif gru == 4:
        hidden = lstm_layer(hidden_dim, dropout)(reshaped)
        hidden = gru_layer(hidden_dim, dropout)(hidden)
        hidden = gru_layer(hidden_dim, dropout)(hidden)

    truncated = hidden[:, :pred_len]

    out = tf.keras.layers.Dense(5, activation="linear")(truncated)

    model = tf.keras.Model(inputs=inputs, outputs=out)

    adam = tf.optimizers.Adam()
    radam = tfa.optimizers.RectifiedAdam()
    lookahead = tfa.optimizers.Lookahead(adam, sync_period=6)
    ranger = tfa.optimizers.Lookahead(radam, sync_period=6)

    model.compile(optimizer=adam, loss="mse")

    return model


## === cell 14
public_df = test.query("seq_length == 107").copy()
private_df = test.query("seq_length == 130").copy()

public_df["sequence"] = public_df["sequence"].astype(str).str.slice(0, 107)
public_df["structure"] = public_df["structure"].astype(str).str.slice(0, 107)
public_df["predicted_loop_type"] = (
    public_df["predicted_loop_type"].astype(str).str.slice(0, 107)
)

private_df["sequence"] = private_df["sequence"].astype(str).str.slice(0, 130)
private_df["structure"] = private_df["structure"].astype(str).str.slice(0, 130)
private_df["predicted_loop_type"] = (
    private_df["predicted_loop_type"].astype(str).str.slice(0, 130)
)

_req_len = 130
_min_len = (
    private_df[["sequence", "structure", "predicted_loop_type"]]
    .astype(str)
    .apply(lambda c: c.str.len())
    .min(axis=1)
)
private_df = private_df.loc[_min_len >= _req_len].copy()

public_inputs = preprocess_inputs(public_df)
private_inputs = preprocess_inputs(private_df)


## --- ERROR in cell 14, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mValueError[0m                                Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/2470718811.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     26[0m [0;34m[0m[0m
[1;32m     27[0m [0mpublic_inputs[0m [0;34m=[0m [0mpreprocess_inputs[0m[0;34m([0m[0mpublic_df[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 28[0;31m [0mprivate_inputs[0m [0;34m=[0m [0mpreprocess_inputs[0m[0;34m([0m[0mprivate_df[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m
[0;32m/tmp/ipykernel_11/3126562086.py[0m in [0;36mpreprocess_inputs[0;34m(df, cols)[0m
[1;32m      1[0m [0;32mdef[0m [0mpreprocess_inputs[0m[0;34m([0m[0mdf[0m[0;34m,[0m [0mcols[0m[0;34m=[0m[0;34m[[0m[0;34m'sequence'[0m[0;34m,[0m [0;34m'structure'[0m[0;34m,[0m [0;34m'predicted_loop_type'[0m[0;34m][0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 2[0;31m     return np.transpose(
[0m[1;32m      3[0m         np.array(
[1;32m      4[0m             [0mdf[0m[0;34m[[0m[0mcols[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[1;32m      5[0m             [0;34m.[0m[0mapplymap[0m[0;34m([0m[0;32mlambda[0m [0mseq[0m[0;34m:[0m [0;34m[[0m[0mtoken2int[0m[0;34m[[0m[0mx[0m[0;34m][0m [0;32mfor[0m [0mx[0m [0;32min[0m [0mseq[0m[0;34m][0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/numpy/core/fromnumeric.py[0m in [0;36mtranspose[0;34m(a, axes)[0m
[1;32m    653[0m [0;34m[0m[0m
[1;32m    654[0m     """
[0;32m--> 655[0;31m     [0;32mreturn[0m [0m_wrapfunc[0m[0;34m([0m[0ma[0m[0;34m,[0m [0;34m'transpose'[0m[0;34m,[0m [0maxes[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    656[0m [0;34m[0m[0m
[1;32m    657[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/numpy/core/fromnumeric.py[0m in [0;36m_wrapfunc[0;34m(obj, method, *args, **kwds)[0m
[1;32m     57[0m [0;34m[0m[0m
[1;32m     58[0m     [0;32mtry[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 59[0;31m         [0;32mreturn[0m [0mbound[0m[0;34m([0m[0;34m*[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mkwds[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     60[0m     [0;32mexcept[0m [0mTypeError[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m     61[0m         [0;31m# A TypeError occurs if the object does have such a method in its[0m[0;34m[0m[0;34m[0m[0m

[0;31mValueError[0m: axes don't match array

## === cell 15
hyb2_short = build_model(gru=4, seq_len=107, pred_len=107)
hyb2_long = build_model(gru=4, seq_len=130, pred_len=130)


hyb2_short.load_weights('model_hyb2.h5')
hyb2_long.load_weights('model_hyb2.h5')

hyb2_public_preds = hyb2_short.predict(public_inputs)
hyb2_private_preds = hyb2_long.predict(private_inputs)
