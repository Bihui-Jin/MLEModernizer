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
import sys
import subprocess

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "protobuf<5"])

import gc

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras import layers as L
from tensorflow.keras.models import Model

from sklearn.preprocessing import LabelEncoder


## === cell 1
train_df = pd.read_json('/kaggle/input/stanford-covid-vaccine/train.json', lines=True)
test_df = pd.read_json('/kaggle/input/stanford-covid-vaccine/test.json', lines=True)
sample_df = pd.read_csv('/kaggle/input/stanford-covid-vaccine/sample_submission.csv')


## === cell 2
train_df.head()


## === cell 3
sample_df.head()


## === cell 4
train_df.columns


## === cell 5
train_df["sequence"].str.split("").apply(lambda x: (np.unique(x), len(x)))


## === cell 6
np.unique(train_df["seq_length"].values)


## === cell 7
feature_columns = ['sequence', 'structure', 'predicted_loop_type']
target_columns = ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']


## === cell 8
train_df.head(3)


## === cell 9
label_encoders = dict()
for column in feature_columns:
    encoder = LabelEncoder()
    encoder.fit(list(set(train_df[column].apply(list).sum())))
    label_encoders[column]=encoder
    del encoder
    gc.collect()
    


## === cell 10
def transform_(df:pd.DataFrame, label_encoders:dict):
    for column in feature_columns:
        df[column+"_n"] = df[column].apply(lambda seq: label_encoders[column].transform(list(seq)))        
    return df


## === cell 11
train_df = transform_(train_df, label_encoders)
test_df = transform_(test_df, label_encoders)

feature_columns_n = [column for column in train_df.columns if "e_n" in column];
feature_columns_n

train_x = np.array(train_df[feature_columns_n].values.tolist()).transpose((0, 2, 1))
train_y = np.array(train_df[target_columns].values.tolist()).transpose((0, 2, 1))


## === cell 12
def build_model(seq_len=107, pred_len=68, dropout=0.5, embed_dim=100, hidden_dim=128):
    
    inputs = L.Input(shape = (seq_len, 3))
    
    inputs_as = L.Lambda(lambda x: tf.split(x, inputs.shape[-1], axis=-1))(inputs)
    print(len(inputs_as), inputs_as[0].shape)
    
    
    embeddings = []
    for i, (inp_a, col)in enumerate(zip(inputs_as, feature_columns)):
        
        embedding = L.Embedding(input_dim = len(label_encoders[col].classes_),
                               output_dim = embed_dim)(inp_a)
        embedding = L.Reshape((-1, embedding.shape[2] * embedding.shape[3]))(embedding)
        
        embeddings.append(embedding)
        
    embed_cat = L.Concatenate()(embeddings)
    
    
    lstm1 = L.Bidirectional(L.LSTM(hidden_dim, dropout=dropout, return_sequences=True))(embeddings[0])
    lstm1 = L.Bidirectional(L.LSTM(hidden_dim//2, dropout=dropout, return_sequences=True))(lstm1)
    
    lstm2 = L.Bidirectional(L.LSTM(hidden_dim, dropout=dropout, return_sequences=True))(embeddings[1])
    lstm2 = L.Bidirectional(L.LSTM(hidden_dim//2, dropout=dropout, return_sequences=True))(lstm2)
    
    lstm3 = L.Bidirectional(L.LSTM(hidden_dim, dropout=dropout, return_sequences=True))(embeddings[2])
    lstm3 = L.Bidirectional(L.LSTM(hidden_dim//2, dropout=dropout, return_sequences=True))(lstm3)
    
    cat = L.Add()([lstm1, lstm2, lstm3])
    """
    lstm1 = L.Bidirectional(L.LSTM(hidden_dim, dropout=dropout, return_sequences=True))(embed_cat)
    lstm2 = L.Bidirectional(L.LSTM(hidden_dim, dropout=dropout, return_sequences=True))(lstm1)
    lstm3 = L.Bidirectional(L.LSTM(hidden_dim, dropout=dropout, return_sequences=True))(lstm2)
    """
    
    cat = cat[:, :pred_len]
    
    dense = L.Dense(5, activation='linear')(cat)
    
    model = Model(inputs=inputs, outputs = dense)
    model.compile(loss="mse", optimizer='adam')
        
    return model
        


## === cell 13
model = build_model()
model.summary()


## === cell 14
tf.keras.utils.plot_model(model, show_shapes=True)


## === cell 15
model.fit(train_x, train_y, 
          batch_size=64,
          epochs=60,
          callbacks=[
              tf.keras.callbacks.ReduceLROnPlateau(),
          ],
          validation_split=0.01)


## === cell 16
public_df = test_df.query("seq_length == 107").copy()
private_df = test_df.query("seq_length == 130").copy()

public_df = transform_(public_df, label_encoders)
private_df = transform_(private_df, label_encoders)

public_test_x = np.array(public_df[feature_columns_n].values.tolist()).transpose((0, 2, 1))
private_test_x = np.array(private_df[feature_columns_n].values.tolist()).transpose((0, 2, 1))


## --- ERROR in cell 16, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mValueError[0m                                Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/471079810.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m      6[0m [0;34m[0m[0m
[1;32m      7[0m [0mpublic_test_x[0m [0;34m=[0m [0mnp[0m[0;34m.[0m[0marray[0m[0;34m([0m[0mpublic_df[0m[0;34m[[0m[0mfeature_columns_n[0m[0;34m][0m[0;34m.[0m[0mvalues[0m[0;34m.[0m[0mtolist[0m[0;34m([0m[0;34m)[0m[0;34m)[0m[0;34m.[0m[0mtranspose[0m[0;34m([0m[0;34m([0m[0;36m0[0m[0;34m,[0m [0;36m2[0m[0;34m,[0m [0;36m1[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 8[0;31m [0mprivate_test_x[0m [0;34m=[0m [0mnp[0m[0;34m.[0m[0marray[0m[0;34m([0m[0mprivate_df[0m[0;34m[[0m[0mfeature_columns_n[0m[0;34m][0m[0;34m.[0m[0mvalues[0m[0;34m.[0m[0mtolist[0m[0;34m([0m[0;34m)[0m[0;34m)[0m[0;34m.[0m[0mtranspose[0m[0;34m([0m[0;34m([0m[0;36m0[0m[0;34m,[0m [0;36m2[0m[0;34m,[0m [0;36m1[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m
[0;31mValueError[0m: axes don't match array

## === cell 17
model_short = build_model(seq_len=107, pred_len=107)
model_long = build_model(seq_len=130, pred_len=130)

model_short.set_weights(model.get_weights())
model_long.set_weights(model.get_weights())

public_preds = model_short.predict(public_test_x, verbose=1)
private_preds = model_long.predict(private_test_x, verbose=1)
