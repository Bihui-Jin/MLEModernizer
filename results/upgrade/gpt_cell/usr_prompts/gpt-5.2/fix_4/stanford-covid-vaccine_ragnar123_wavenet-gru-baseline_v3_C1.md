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

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
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
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "protobuf==4.25.3"]
    )
    for _m in list(sys.modules.keys()):
        if _m.startswith("google.protobuf"):
            del sys.modules[_m]

import pandas as pd
import numpy as np
import seaborn as sns
import math
import random
import json
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import tensorflow as tf

try:
    import tensorflow_addons as tfa
except Exception:
    tfa = None

import tensorflow.keras.backend as K
from sklearn.model_selection import train_test_split, KFold
from sklearn import metrics


## === cell 1
FOLDS = 5
EPOCHS = 130
BATCH_SIZE = 64
LR = 0.001
VERBOSE = 2
SEED = 123

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)
    
seed_everything(SEED)

train = pd.read_json('../input/stanford-covid-vaccine/train.json', lines = True)
test = pd.read_json('../input/stanford-covid-vaccine/test.json', lines = True)
sample_sub = pd.read_csv('../input/stanford-covid-vaccine/sample_submission.csv')


target_cols = ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']

token2int = {x: i for i, x in enumerate('().ACGUBEHIMSX')}

def preprocess_inputs(df, cols = ['sequence', 'structure', 'predicted_loop_type']):
    return np.transpose(
        np.array(
            df[cols]\
            .applymap(lambda seq: [token2int[x] for x in seq])\
            .values\
            .tolist()
        ),
        (0, 2, 1)
    )

train_inputs = preprocess_inputs(train.loc[train['SN_filter'] == 1])
train_labels = np.array(train[train['SN_filter'] == 1][target_cols].values.tolist()).transpose(0, 2, 1)
public_test_df = test[test['seq_length'] == 107]
private_test_df = test[test['seq_length'] == 130]
public_test = preprocess_inputs(public_test_df)
private_test = preprocess_inputs(private_test_df)


## --- ERROR in cell 1, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mValueError[0m                                Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/458248589.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     55[0m [0;31m# Preprocess the test sets to the same format as our training data[0m[0;34m[0m[0;34m[0m[0m
[1;32m     56[0m [0mpublic_test[0m [0;34m=[0m [0mpreprocess_inputs[0m[0;34m([0m[0mpublic_test_df[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 57[0;31m [0mprivate_test[0m [0;34m=[0m [0mpreprocess_inputs[0m[0;34m([0m[0mprivate_test_df[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m
[0;32m/tmp/ipykernel_11/458248589.py[0m in [0;36mpreprocess_inputs[0;34m(df, cols)[0m
[1;32m     36[0m [0;31m# Preprocesing function to transform features to 3d format[0m[0;34m[0m[0;34m[0m[0m
[1;32m     37[0m [0;32mdef[0m [0mpreprocess_inputs[0m[0;34m([0m[0mdf[0m[0;34m,[0m [0mcols[0m [0;34m=[0m [0;34m[[0m[0;34m'sequence'[0m[0;34m,[0m [0;34m'structure'[0m[0;34m,[0m [0;34m'predicted_loop_type'[0m[0;34m][0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 38[0;31m     return np.transpose(
[0m[1;32m     39[0m         np.array(
[1;32m     40[0m             [0mdf[0m[0;34m[[0m[0mcols[0m[0;34m][0m[0;31m\[0m[0;34m[0m[0;34m[0m[0m

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

## === cell 2
def build_model(seq_len = 107, pred_len = 68, embed_dim = 75, dropout = 0.10):
    
    def wave_block(x, filters, kernel_size, n):
        dilation_rates = [2 ** i for i in range(n)]
        x = tf.keras.layers.Conv1D(filters = filters, 
                                   kernel_size = 1,
                                   padding = 'same')(x)
        res_x = x
        for dilation_rate in dilation_rates:
            tanh_out = tf.keras.layers.Conv1D(filters = filters,
                              kernel_size = kernel_size,
                              padding = 'same', 
                              activation = 'tanh', 
                              dilation_rate = dilation_rate)(x)
            sigm_out = tf.keras.layers.Conv1D(filters = filters,
                              kernel_size = kernel_size,
                              padding = 'same',
                              activation = 'sigmoid', 
                              dilation_rate = dilation_rate)(x)
            x = tf.keras.layers.Multiply()([tanh_out, sigm_out])
            x = tf.keras.layers.Conv1D(filters = filters,
                       kernel_size = 1,
                       padding = 'same')(x)
            res_x = tf.keras.layers.Add()([res_x, x])
        return res_x
    
    inputs = tf.keras.layers.Input(shape = (seq_len, 3))
    embed = tf.keras.layers.Embedding(input_dim = len(token2int), output_dim = embed_dim)(inputs)
    reshaped = tf.reshape(embed, shape = (-1, embed.shape[1], embed.shape[2] * embed.shape[3]))
    reshaped = tf.keras.layers.SpatialDropout1D(dropout)(reshaped)
    
    x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(256, 
                                                          dropout = dropout, 
                                                          return_sequences = True, 
                                                          kernel_initializer = 'orthogonal'))(reshaped)
    x = wave_block(reshaped, 16, 3, 12)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(256, 
                                                          dropout = dropout, 
                                                          return_sequences = True, 
                                                          kernel_initializer = 'orthogonal'))(x)
    x = wave_block(reshaped, 32, 3, 8)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(256, 
                                                          dropout = dropout, 
                                                          return_sequences = True, 
                                                          kernel_initializer = 'orthogonal'))(x)
    x = wave_block(reshaped, 64, 3, 4)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(256, 
                                                          dropout = dropout, 
                                                          return_sequences = True, 
                                                          kernel_initializer = 'orthogonal'))(x)
    
    truncated = x[:, :pred_len]
    out = tf.keras.layers.Dense(5, activation = 'linear')(truncated)
    model = tf.keras.models.Model(inputs = inputs, outputs = out)
    opt = tf.keras.optimizers.Adam(learning_rate = LR)
    opt = tfa.optimizers.SWA(opt)
    model.compile(optimizer = opt,
                  loss = tf.keras.losses.MeanSquaredError(),
                  metrics = [tf.keras.metrics.RootMeanSquaredError()])
    
    return model

def mcrmse(y_true, y_pred):
    y_true_ = y_true.reshape(108052, 5)
    y_pred_ = y_pred.reshape(108052, 5)
    y_true0 = y_true_[:, 0]
    y_true1 = y_true_[:, 1]
    y_true2 = y_true_[:, 2]
    y_true3 = y_true_[:, 3]
    y_true4 = y_true_[:, 4]
    y_pred0 = y_pred_[:, 0]
    y_pred1 = y_pred_[:, 1]
    y_pred2 = y_pred_[:, 2]
    y_pred3 = y_pred_[:, 3]
    y_pred4 = y_pred_[:, 4]
    rmse0 = math.sqrt(metrics.mean_squared_error(y_true0, y_pred0))
    rmse1 = math.sqrt(metrics.mean_squared_error(y_true1, y_pred1))
    rmse2 = math.sqrt(metrics.mean_squared_error(y_true2, y_pred2))
    rmse3 = math.sqrt(metrics.mean_squared_error(y_true3, y_pred3))
    rmse4 = math.sqrt(metrics.mean_squared_error(y_true4, y_pred4))
    return np.mean([rmse0, rmse1, rmse2, rmse3, rmse4])


def train_and_evaluate(train_inputs, train_labels, public_test, private_test):
        
    oof_preds = np.zeros((train_inputs.shape[0], 68, 5))
    public_preds = np.zeros((public_test.shape[0], 107, 5))
    private_preds = np.zeros((private_test.shape[0], 130, 5))

    kfold = KFold(FOLDS, shuffle = True, random_state = SEED)
    for fold, (train_index, val_index) in enumerate(kfold.split(train_inputs)):
        
        print(f'Training fold {fold + 1}')
    
        checkpoint = tf.keras.callbacks.ModelCheckpoint(f'fold_{fold + 1}.h5', 
                                                        monitor = 'val_loss',
                                                        save_best_only = True,
                                                        save_weights_only = True
                                                       )
        cb_lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss', 
                                                              mode = 'min', 
                                                              factor = 0.5, 
                                                              patience = 5, 
                                                              verbose = 1, 
                                                              min_delta = 0.00001
                                                             )
    
        x_train, x_val = train_inputs[train_index], train_inputs[val_index]
        y_train, y_val = train_labels[train_index], train_labels[val_index]
        K.clear_session()
        model = build_model()
        history = model.fit(x_train, y_train,
                            validation_data = (x_val, y_val),
                            batch_size = BATCH_SIZE,
                            epochs = EPOCHS,
                            callbacks = [checkpoint, cb_lr_schedule],
                            verbose = VERBOSE)
    
        model.load_weights(f'fold_{fold + 1}.h5')
        oof_preds[val_index] = model.predict(x_val)

        short = build_model(seq_len = 107, pred_len = 107)
        short.load_weights(f'fold_{fold + 1}.h5')
        public_preds += short.predict(public_test) / FOLDS

        long = build_model(seq_len = 130, pred_len = 130)
        long.load_weights(f'fold_{fold + 1}.h5')
        private_preds += long.predict(private_test) / FOLDS
        
        print('-'*50)
        print('\n')
    
    mean_col_rmse = mcrmse(train_labels, oof_preds)

    print(f'Our out of folds mean columnwise root mean squared error is {mean_col_rmse}')
    
    return public_preds, private_preds
