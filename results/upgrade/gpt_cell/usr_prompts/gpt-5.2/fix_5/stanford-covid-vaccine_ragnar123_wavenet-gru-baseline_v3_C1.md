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
    os.environ["PYTHONHASHSEED"] = str(seed)
    tf.random.set_seed(seed)


seed_everything(SEED)

train = pd.read_json("../input/stanford-covid-vaccine/train.json", lines=True)
test = pd.read_json("../input/stanford-covid-vaccine/test.json", lines=True)
sample_sub = pd.read_csv("../input/stanford-covid-vaccine/sample_submission.csv")


target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

token2int = {x: i for i, x in enumerate("().ACGUBEHIMSX")}


def preprocess_inputs(df, cols=["sequence", "structure", "predicted_loop_type"]):
    if df is None or len(df) == 0:
        return np.zeros((0, 0, len(cols)), dtype=np.int32)

    arr = np.array(
        df[cols].applymap(lambda seq: [token2int[x] for x in seq]).values.tolist()
    )
    return np.transpose(arr, (0, 2, 1))


train_inputs = preprocess_inputs(train.loc[train["SN_filter"] == 1])
train_labels = np.array(
    train[train["SN_filter"] == 1][target_cols].values.tolist()
).transpose(0, 2, 1)
public_test_df = test[test["seq_length"] == 107]
private_test_df: pd.DataFrame = test[test["seq_length"] == 130]
public_test = preprocess_inputs(public_test_df)
private_test = preprocess_inputs(private_test_df)


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


## === cell 3
public_preds, private_preds = train_and_evaluate(train_inputs, train_labels, public_test, private_test)


## --- ERROR in cell 3, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mValueError[0m                                Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/1579376146.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[0;32m----> 1[0;31m [0mpublic_preds[0m[0;34m,[0m [0mprivate_preds[0m [0;34m=[0m [0mtrain_and_evaluate[0m[0;34m([0m[0mtrain_inputs[0m[0;34m,[0m [0mtrain_labels[0m[0;34m,[0m [0mpublic_test[0m[0;34m,[0m [0mprivate_test[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m
[0;32m/tmp/ipykernel_11/2636264495.py[0m in [0;36mtrain_and_evaluate[0;34m(train_inputs, train_labels, public_test, private_test)[0m
[1;32m    101[0m         [0mprint[0m[0;34m([0m[0;34mf'Training fold {fold + 1}'[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    102[0m [0;34m[0m[0m
[0;32m--> 103[0;31m         checkpoint = tf.keras.callbacks.ModelCheckpoint(f'fold_{fold + 1}.h5', 
[0m[1;32m    104[0m                                                         [0mmonitor[0m [0;34m=[0m [0;34m'val_loss'[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m    105[0m                                                         [0msave_best_only[0m [0;34m=[0m [0;32mTrue[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/keras/src/callbacks/model_checkpoint.py[0m in [0;36m__init__[0;34m(self, filepath, monitor, verbose, save_best_only, save_weights_only, mode, save_freq, initial_value_threshold)[0m
[1;32m    182[0m         [0;32mif[0m [0msave_weights_only[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    183[0m             [0;32mif[0m [0;32mnot[0m [0mself[0m[0;34m.[0m[0mfilepath[0m[0;34m.[0m[0mendswith[0m[0;34m([0m[0;34m".weights.h5"[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 184[0;31m                 raise ValueError(
[0m[1;32m    185[0m                     [0;34m"When using `save_weights_only=True` in `ModelCheckpoint`"[0m[0;34m[0m[0;34m[0m[0m
[1;32m    186[0m                     [0;34m", the filepath provided must end in `.weights.h5` "[0m[0;34m[0m[0;34m[0m[0m

[0;31mValueError[0m: When using `save_weights_only=True` in `ModelCheckpoint`, the filepath provided must end in `.weights.h5` (Keras weights format). Received: filepath=fold_1.h5

## === cell 4
def inference_format(public_test_df, public_preds, private_test_df, private_preds, target_cols):
    predictions = []
    for test, preds in [(public_test_df, public_preds), (private_test_df, private_preds)]:
        for index, uid in enumerate(test['id']):
            single_pred = preds[index]
            single_df = pd.DataFrame(single_pred, columns = target_cols)
            single_df['id_seqpos'] = [f'{uid}_{x}' for x in range(single_df.shape[0])]
            predictions.append(single_df)
            
    predictions = pd.concat(predictions)
    return predictions

predictions = inference_format(public_test_df, public_preds, private_test_df, private_preds, target_cols)
submission = sample_sub[['id_seqpos']].merge(predictions, on = ['id_seqpos'])
submission.to_csv('submission.csv', index = False)
print('Submission saved')
submission.head()
