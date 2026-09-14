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
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
plotly==5.24.1
plotly-express==0.4.1
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sentence-transformers==4.1.0
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
transformers==4.53.3

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
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "3")

try:
    from google.protobuf.message_factory import MessageFactory

    if not hasattr(MessageFactory, "GetPrototype"):

        def _GetPrototype(self, descriptor):
            if hasattr(self, "GetMessageClass"):
                return self.GetMessageClass(descriptor)
            raise AttributeError("MessageFactory has no GetPrototype/GetMessageClass")

        MessageFactory.GetPrototype = _GetPrototype
except Exception:
    pass

import pandas as pd
import numpy as np
import json
import tensorflow.keras.layers as L
import tensorflow as tf
import plotly.express as px
from sklearn.preprocessing import quantile_transform, StandardScaler

from transformers import BertConfig, TFBertModel


## === cell 1
pred_cols = ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']


## === cell 2
def gru_layer(hidden_dim, dropout):
    return L.Bidirectional(L.GRU(hidden_dim, dropout=dropout, return_sequences=True))

def build_model(seq_len=107, pred_len=68, dropout=0.5, embed_dim=100, hidden_dim=128):
    ids = L.Input(shape=(seq_len,3), dtype=tf.int32)
    config = BertConfig() 
    config.vocab_size = len(token2int)
    config.num_hidden_layers = 3
    config.attention_probs_dropout_prob  = 0.133
    config.hidden_act= tf.tanh
    bert_model = TFBertModel(config=config)

    embed = bert_model(ids[:,0])[0]
    embed1 = bert_model(ids[:,1])[0]
    embed2 = bert_model(ids[:,2])[0]
    bert_embeddings = L.Concatenate()([embed,embed1,embed2])
    reshaped = tf.reshape(
        bert_embeddings, shape=(-1, bert_embeddings.shape[2],  bert_embeddings.shape[1]))

    hidden = L.AveragePooling1D(pool_size=16)(reshaped)
    truncated = hidden[:,:pred_len, :]
    out = L.Dense(5, activation='linear')(truncated)

    model = tf.keras.Model(inputs=ids, outputs=out)

    model.compile(tf.keras.optimizers.Adam(), loss='mse')
    
    return model


## === cell 3
token2int = {x:i for i, x in enumerate('().ACGUBEHIMSX')}

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


## === cell 4
train = pd.read_json('/kaggle/input/stanford-covid-vaccine/train.json', lines=True)
test = pd.read_json('/kaggle/input/stanford-covid-vaccine/test.json', lines=True)
sample_df = pd.read_csv('/kaggle/input/stanford-covid-vaccine/sample_submission.csv')


## === cell 5
train_inputs = preprocess_inputs(train)
train_labels = np.array(train[pred_cols].values.tolist()).transpose((0, 2, 1))


## === cell 6
train_inputs.shape


## === cell 7
for df in [train,test]:
    df['Paired']=[sum([i=='(' or i==')' for i in j]) for j in df['structure']]
    df['Unpaired']=[sum([i=='.' for i in j]) for j in df['structure']]
    for col in ['E','S','H','I','G','A','U']:
        if col in ['E','S','H','I']:
            df[col]=[sum([i==col for i in j])/len(j) for j in df['predicted_loop_type']]
        else:
            df[col]=[sum([i==col for i in j])/len(j) for j in df['sequence']]
for a in [ 'G', 'A', 'C', 'U']:
    train[a+'_position']=[np.sum([i for i in range(len(j)) if j[i]==a])/len([i for i in range(len(j)) if j[i]==a]) for j in train['sequence']]
    test[a+'_position']=[np.sum([i for i in range(len(j)) if j[i]==a])/len([i for i in range(len(j)) if j[i]==a]) for j in test['sequence']]
for a in [ 'E', 'S', 'H',]:
    train[a+'_position']=[np.sum([i for i in range(len(j)) if j[i]==a])/len([i for i in range(len(j)) if j[i]==a]) for j in train['predicted_loop_type']]
    test[a+'_position']=[np.sum([i for i in range(len(j)) if j[i]==a])/len([i for i in range(len(j)) if j[i]==a]) for j in test['predicted_loop_type']]
for a in [ 'E', 'S', 'H',]:
    train[a+'']=[np.sum([i for i in range(len(j)) if j[i]==a])/len([i for i in range(len(j)) if j[i]==a]) for j in train['predicted_loop_type']]
    test[a+'_position']=[np.sum([i for i in range(len(j)) if j[i]==a])/len([i for i in range(len(j)) if j[i]==a]) for j in test['predicted_loop_type']]


## === cell 8
target_columns = ['reactivity', 'deg_Mg_pH10','deg_pH10', 'deg_Mg_50C', 'deg_50C']
target_columns.extend(['SN_filter', 'signal_to_noise'])
target_columns.extend(['deg_error_pH10', 'deg_error_Mg_50C', 'deg_error_50C', 'reactivity_error', 'deg_error_Mg_pH10'] )
train.drop(target_columns,axis=1,inplace=True)


## === cell 9
SC = StandardScaler()
train_measurements = SC.fit_transform(pd.concat((train.select_dtypes('float64'),train.select_dtypes('int64')),axis=1))


## === cell 10
train_measurements.shape


## === cell 11
np.min(train_measurements),np.max(train_measurements)


## === cell 12
def gru_layer(hidden_dim, dropout):
    return L.Bidirectional(L.GRU(hidden_dim, dropout=dropout, return_sequences=True))


def build_model(seq_len=107, pred_len=68, dropout=0.5, embed_dim=100, hidden_dim=128):
    ids = L.Input(shape=(seq_len, 3), dtype=tf.int32)
    config = BertConfig()
    config.vocab_size = len(token2int)
    config.num_hidden_layers = 3
    config.attention_probs_dropout_prob = 0.133
    config.hidden_act = tf.tanh
    bert_model = TFBertModel(config=config)

    ids0 = L.Lambda(lambda x: x[:, :, 0], name="ids0")(ids)
    ids1 = L.Lambda(lambda x: x[:, :, 1], name="ids1")(ids)
    ids2 = L.Lambda(lambda x: x[:, :, 2], name="ids2")(ids)

    embed = bert_model(input_ids=ids0)[0]
    embed1 = bert_model(input_ids=ids1)[0]
    embed2 = bert_model(input_ids=ids2)[0]

    bert_embeddings = L.Concatenate()([embed, embed1, embed2])
    reshaped = tf.reshape(
        bert_embeddings, shape=(-1, bert_embeddings.shape[2], bert_embeddings.shape[1])
    )

    hidden = L.AveragePooling1D(pool_size=16)(reshaped)
    truncated = hidden[:, :pred_len, :]
    out = L.Dense(5, activation="linear")(truncated)

    model = tf.keras.Model(inputs=ids, outputs=out)

    model.compile(tf.keras.optimizers.Adam(), loss="mse")

    return model


## === cell 13
train_inputs.shape,train_labels.shape


## === cell 14
_original_build_model = build_model


def build_model(seq_len=107, pred_len=68, dropout=0.5, embed_dim=100, hidden_dim=128):
    ids = L.Input(shape=(seq_len, 3), dtype=tf.int32)
    config = BertConfig()
    config.vocab_size = len(token2int)
    config.num_hidden_layers = 3
    config.attention_probs_dropout_prob = 0.133
    config.hidden_act = tf.tanh
    bert_model = TFBertModel(config=config)

    ids0 = L.Lambda(lambda x: tf.cast(x[:, :, 0], tf.int32), name="ids0")(ids)
    ids1 = L.Lambda(lambda x: tf.cast(x[:, :, 1], tf.int32), name="ids1")(ids)
    ids2 = L.Lambda(lambda x: tf.cast(x[:, :, 2], tf.int32), name="ids2")(ids)

    embed = L.Lambda(
        lambda x: bert_model(input_ids=x).last_hidden_state,
        output_shape=(seq_len, config.hidden_size),
        name="bert0",
    )(ids0)
    embed1 = L.Lambda(
        lambda x: bert_model(input_ids=x).last_hidden_state,
        output_shape=(seq_len, config.hidden_size),
        name="bert1",
    )(ids1)
    embed2 = L.Lambda(
        lambda x: bert_model(input_ids=x).last_hidden_state,
        output_shape=(seq_len, config.hidden_size),
        name="bert2",
    )(ids2)

    bert_embeddings = L.Concatenate()([embed, embed1, embed2])

    reshaped = L.Reshape((config.hidden_size, 3 * seq_len), name="reshape_bert_concat")(
        bert_embeddings
    )

    hidden = L.AveragePooling1D(pool_size=16)(reshaped)
    truncated = hidden[:, :pred_len, :]
    out = L.Dense(5, activation="linear")(truncated)

    model = tf.keras.Model(inputs=ids, outputs=out)
    model.compile(tf.keras.optimizers.Adam(), loss="mse")
    return model


model = build_model()

with tf.device("/gpu"):
    history = model.fit(
        train_inputs,
        train_labels,
        batch_size=64,
        epochs=100,
        validation_split=0.05,
        callbacks=[
            tf.keras.callbacks.ReduceLROnPlateau(),
            tf.keras.callbacks.ModelCheckpoint(
                "model.h5", save_weights_only=True, save_best_only=True
            ),
        ],
    )


## --- ERROR in cell 14, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mValueError[0m                                Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/4248801718.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     59[0m         callbacks=[
[1;32m     60[0m             [0mtf[0m[0;34m.[0m[0mkeras[0m[0;34m.[0m[0mcallbacks[0m[0;34m.[0m[0mReduceLROnPlateau[0m[0;34m([0m[0;34m)[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 61[0;31m             tf.keras.callbacks.ModelCheckpoint(
[0m[1;32m     62[0m                 [0;34m"model.h5"[0m[0;34m,[0m [0msave_weights_only[0m[0;34m=[0m[0;32mTrue[0m[0;34m,[0m [0msave_best_only[0m[0;34m=[0m[0;32mTrue[0m[0;34m[0m[0;34m[0m[0m
[1;32m     63[0m             ),

[0;32m/usr/local/lib/python3.11/dist-packages/keras/src/callbacks/model_checkpoint.py[0m in [0;36m__init__[0;34m(self, filepath, monitor, verbose, save_best_only, save_weights_only, mode, save_freq, initial_value_threshold)[0m
[1;32m    182[0m         [0;32mif[0m [0msave_weights_only[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    183[0m             [0;32mif[0m [0;32mnot[0m [0mself[0m[0;34m.[0m[0mfilepath[0m[0;34m.[0m[0mendswith[0m[0;34m([0m[0;34m".weights.h5"[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 184[0;31m                 raise ValueError(
[0m[1;32m    185[0m                     [0;34m"When using `save_weights_only=True` in `ModelCheckpoint`"[0m[0;34m[0m[0;34m[0m[0m
[1;32m    186[0m                     [0;34m", the filepath provided must end in `.weights.h5` "[0m[0;34m[0m[0;34m[0m[0m

[0;31mValueError[0m: When using `save_weights_only=True` in `ModelCheckpoint`, the filepath provided must end in `.weights.h5` (Keras weights format). Received: filepath=model.h5

## === cell 15
model.save_weights('model.h5')
