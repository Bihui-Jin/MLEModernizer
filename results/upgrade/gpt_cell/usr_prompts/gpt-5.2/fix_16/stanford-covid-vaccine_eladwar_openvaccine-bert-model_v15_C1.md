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

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

import sys
import subprocess

try:
    import google.protobuf  # noqa: F401
    from google.protobuf import __version__ as _pb_ver

    _pb_major = int(_pb_ver.split(".", 1)[0])
except Exception:
    _pb_major = None

if _pb_major is not None and _pb_major >= 5:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "protobuf<5"])
    import importlib
    import google.protobuf as _gp

    importlib.reload(_gp)

import pandas as pd
import numpy as np
import json
import tensorflow as tf
import tensorflow.keras.layers as L

from sklearn.preprocessing import quantile_transform, StandardScaler, MinMaxScaler


## === cell 1
pred_cols = ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']


## === cell 2
from transformers import BertConfig

config = BertConfig()


## === cell 3
config.num_attention_heads


## === cell 4
def MCRMSE(y_true, y_pred):
    colwise_mse = tf.reduce_mean(tf.square(y_true - y_pred), axis=1)
    return tf.reduce_mean(tf.sqrt(colwise_mse), axis=1)

def gru_layer(hidden_dim, dropout):
    return L.Bidirectional(L.GRU(hidden_dim, dropout=dropout, return_sequences=True))

def build_model(seq_len=107, pred_len=68, dropout=0.5, embed_dim=100, hidden_dim=128):
    ids = L.Input(shape=(seq_len,3), dtype=tf.int32)
    flat = L.Flatten()(ids)
    config = BertConfig() 
    config.vocab_size = 7
    config.num_hidden_layers = 3
    config.num_attention_heads = 1
    config.attention_probs_dropout_prob  = 0.5
    config.hidden_size = 120
    config.hidden_act= tf.sinh #tf.tanh
    bert_model = TFBertModel(config=config)

    bert_embeddings = bert_model(flat)[0]

    hidden = L.AveragePooling1D(pool_size=2)(bert_embeddings)
    truncated = hidden[:,:pred_len, :]
    out = L.Dense(5, activation='linear')(truncated)

    model = tf.keras.Model(inputs=ids, outputs=out)

    model.compile(tf.keras.optimizers.Adam(), loss=MCRMSE)
    
    return model


## === cell 5
vocab = {
    'sequence': {x:i for i, x in enumerate("A C G U".split())},
    'structure': {x:i for i, x in enumerate("( . )".split())},
    'predicted_loop_type': {x:i for i, x in enumerate("B E H I M S X".split())},
}
def preprocess_inputs(df, cols=['sequence', 'structure', 'predicted_loop_type']):
    
    def f(x):
        return [vocab['sequence'][x] for x in x[0]],\
                [vocab['structure'][x] for x in x[1]],\
                [vocab['predicted_loop_type'][x] for x in x[2]],

    return np.array(
            df[cols]
            .apply(f, axis=1)
            .values
            .tolist()
        )


## === cell 6
train = pd.read_json('/kaggle/input/stanford-covid-vaccine/train.json', lines=True)
test = pd.read_json('/kaggle/input/stanford-covid-vaccine/test.json', lines=True)
sample_df = pd.read_csv('/kaggle/input/stanford-covid-vaccine/sample_submission.csv')


## === cell 7
print(pd.Series(list(train['structure'][0])).value_counts())
print(pd.Series(list(train['sequence'][0])).value_counts())
print(pd.Series(list(train['predicted_loop_type'][0])).value_counts())


## === cell 8
train_inputs = preprocess_inputs(train)
train_labels = np.array(train[pred_cols].values.tolist()).transpose((0, 2, 1))


## === cell 9
train_inputs.shape


## === cell 10
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


## === cell 11
target_columns = ['reactivity', 'deg_Mg_pH10','deg_pH10', 'deg_Mg_50C', 'deg_50C']
target_columns.extend(['SN_filter', 'signal_to_noise'])
target_columns.extend(['deg_error_pH10', 'deg_error_Mg_50C', 'deg_error_50C', 'reactivity_error', 'deg_error_Mg_pH10'] )
train.drop(target_columns,axis=1,inplace=True)


## === cell 12
SC = MinMaxScaler(feature_range=(-1, 1))
train_measurements = SC.fit_transform(pd.concat((train.select_dtypes('float64'),train.select_dtypes('int64')),axis=1))


## === cell 13
train_measurements.shape


## === cell 14
np.min(train_measurements),np.max(train_measurements)


## === cell 15
pd.DataFrame(train_measurements).describe().T


## === cell 16
from transformers import TFBertModel

model = build_model()
model.summary()


## --- ERROR in cell 16, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mValueError[0m                                Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/3958902010.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m      2[0m [0;32mfrom[0m [0mtransformers[0m [0;32mimport[0m [0mTFBertModel[0m[0;34m[0m[0;34m[0m[0m
[1;32m      3[0m [0;34m[0m[0m
[0;32m----> 4[0;31m [0mmodel[0m [0;34m=[0m [0mbuild_model[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      5[0m [0mmodel[0m[0;34m.[0m[0msummary[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/tmp/ipykernel_11/3646264068.py[0m in [0;36mbuild_model[0;34m(seq_len, pred_len, dropout, embed_dim, hidden_dim)[0m
[1;32m     18[0m     [0mbert_model[0m [0;34m=[0m [0mTFBertModel[0m[0;34m([0m[0mconfig[0m[0;34m=[0m[0mconfig[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     19[0m [0;34m[0m[0m
[0;32m---> 20[0;31m     [0mbert_embeddings[0m [0;34m=[0m [0mbert_model[0m[0;34m([0m[0mflat[0m[0;34m)[0m[0;34m[[0m[0;36m0[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     21[0m [0;34m[0m[0m
[1;32m     22[0m     [0mhidden[0m [0;34m=[0m [0mL[0m[0;34m.[0m[0mAveragePooling1D[0m[0;34m([0m[0mpool_size[0m[0;34m=[0m[0;36m2[0m[0;34m)[0m[0;34m([0m[0mbert_embeddings[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/tf_keras/src/utils/traceback_utils.py[0m in [0;36merror_handler[0;34m(*args, **kwargs)[0m
[1;32m     68[0m             [0;31m# To get the full stack trace, call:[0m[0;34m[0m[0;34m[0m[0m
[1;32m     69[0m             [0;31m# `tf.debugging.disable_traceback_filtering()`[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 70[0;31m             [0;32mraise[0m [0me[0m[0;34m.[0m[0mwith_traceback[0m[0;34m([0m[0mfiltered_tb[0m[0;34m)[0m [0;32mfrom[0m [0;32mNone[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     71[0m         [0;32mfinally[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m     72[0m             [0;32mdel[0m [0mfiltered_tb[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/transformers/modeling_tf_utils.py[0m in [0;36mrun_call_with_unpacked_inputs[0;34m(self, *args, **kwargs)[0m
[1;32m    434[0m             [0mconfig[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0mconfig[0m[0;34m[0m[0;34m[0m[0m
[1;32m    435[0m [0;34m[0m[0m
[0;32m--> 436[0;31m         [0munpacked_inputs[0m [0;34m=[0m [0minput_processing[0m[0;34m([0m[0mfunc[0m[0;34m,[0m [0mconfig[0m[0;34m,[0m [0;34m**[0m[0mfn_args_and_kwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    437[0m         [0;32mreturn[0m [0mfunc[0m[0;34m([0m[0mself[0m[0;34m,[0m [0;34m**[0m[0munpacked_inputs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    438[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/transformers/modeling_tf_utils.py[0m in [0;36minput_processing[0;34m(func, config, **kwargs)[0m
[1;32m    564[0m             [0moutput[0m[0;34m[[0m[0mmain_input_name[0m[0;34m][0m [0;34m=[0m [0mmain_input[0m[0;34m[0m[0;34m[0m[0m
[1;32m    565[0m         [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 566[0;31m             raise ValueError(
[0m[1;32m    567[0m                 [0;34mf"Data of type {type(main_input)} is not allowed only {allowed_types} is accepted for"[0m[0;34m[0m[0;34m[0m[0m
[1;32m    568[0m                 [0;34mf" {main_input_name}."[0m[0;34m[0m[0;34m[0m[0m

[0;31mValueError[0m: Exception encountered when calling layer 'tf_bert_model' (type TFBertModel).

Data of type <class 'keras.src.backend.common.keras_tensor.KerasTensor'> is not allowed only (<class 'tensorflow.python.framework.tensor.Tensor'>, <class 'bool'>, <class 'int'>, <class 'transformers.utils.generic.ModelOutput'>, <class 'tuple'>, <class 'list'>, <class 'dict'>, <class 'numpy.ndarray'>) is accepted for input_ids.

Call arguments received by layer 'tf_bert_model' (type TFBertModel):
  • input_ids=<KerasTensor shape=(None, 321), dtype=int32, sparse=False, name=keras_tensor_1>
  • attention_mask=None
  • token_type_ids=None
  • position_ids=None
  • head_mask=None
  • inputs_embeds=None
  • encoder_hidden_states=None
  • encoder_attention_mask=None
  • past_key_values=None
  • use_cache=None
  • output_attentions=None
  • output_hidden_states=None
  • return_dict=None
  • training=False

## === cell 17
train_inputs.shape,train_labels.shape
