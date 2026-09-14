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
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import sklearn
import matplotlib.pyplot as plt



## === cell 1
target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]




## === cell 2
def read_json(filename):
    """
    reads in train/test json data as pandas DataFrame
    """
    file = open(filename)
    df = pd.read_json(path_or_buf=file, orient="records", lines=True)
    return df




## === cell 3
train_df = read_json("../input/stanford-covid-vaccine/train.json")

print(train_df["id"].nunique())
print(train_df.columns)
train_df



## === cell 4
test_df = read_json("../input/stanford-covid-vaccine/test.json")

print("Features only in training set (not including target columns):")
set(train_df.columns) - set(test_df.columns) - set(target_cols)



## === cell 5
test_df



## === cell 6
"""! ls
#! ls draw_rna
! ls forna

! python forna/forna_server.py -s -d"""



## === cell 7
"""seq = train_df.loc[0, 'sequence']
struct = train_df.loc[0, 'structure']

seq, struct"""




## === cell 8
def unpack_df_lists(df, col_names):
    """
    turn list-like elements of dataframe into tabular data

    works great
    """
    if isinstance(
        col_names, str
    ):  # if string is passed in, convert to list for convenience
        col_names = [col_names]

    all_series = [df[c] for c in col_names]  # select relevant columns
    unpacked = [
        ser.explode() for ser in all_series
    ]  # unpack lists for each feature series

    data = pd.concat(unpacked, axis=1)  # concat unpacked columns together

    original = df.drop(col_names, axis=1)  # drop columns with list elements
    data = original.join(data)  # then join unpacked data to original df

    return data




## === cell 9
"""
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
"""



## === cell 10
tokenize_cols = ["sequence", "structure", "predicted_loop_type"]


def tokenize_df(df, tokenizer, cols=tokenize_cols):
    """
    tokenizer is a tensorflow keras Tokenizer that has been already fitted on text
    """
    data = df.copy()
    for c in tokenize_cols:
        data[c] = tokenizer.texts_to_sequences(data[c])
    return data




## === cell 11
import os

import google.protobuf
from packaging import version
import sys
import subprocess

if version.parse(google.protobuf.__version__) >= version.parse("6.0.0"):
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "protobuf==5.28.3"]
    )
    import importlib

    importlib.invalidate_caches()
    importlib.reload(google.protobuf)

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer(filters=None, lower=False, char_level=True)
tokenizer.fit_on_texts("().ACGUBEHIMSX")

temp = train_df[train_df["SN_filter"] == 1]
temp = tokenize_df(temp, tokenizer)

train_df = temp

train_df



## === cell 12
import matplotlib.pyplot as plt
import seaborn as sns

corr_data = train_df.drop(
    [
        "index",
        "id",
        "sequence",
        "structure",
        "predicted_loop_type",
        "seq_length",
        "seq_scored",
    ],
    axis=1,
)
corr_data = unpack_df_lists(
    corr_data,
    [
        "reactivity_error",
        "deg_error_Mg_pH10",
        "deg_error_pH10",
        "deg_error_Mg_50C",
        "deg_error_50C",
        "reactivity",
        "deg_Mg_pH10",
        "deg_pH10",
        "deg_Mg_50C",
        "deg_50C",
    ],
).convert_dtypes()

print(corr_data)

corr_data = corr_data.corr()

corr_data



## === cell 13
mask = np.ma.masked_inside(
    corr_data.values, -0.15, 0.15
).mask  # get most powerful features

plt.figure(figsize=(8, 8))
sns.heatmap(corr_data, annot=True, mask=mask)



## === cell 14
"""
For tensorflow compatibility, metrics should have signature f(y_true, y_pred)
For sklearn compatibility, metrics should have signature f(y_true, y_pred, **kwargs)
"""


def score(raw_values=False, use_tf=False, **kwargs):
    """
    This is competition metric: Mean Columnwise Root Mean Square Error (MCRMSE)
    Averages RMSE loss over all scored columns (only 3 are scored)
    """

    col_dict = {"reactivity": 0, "deg_Mg_pH10": 1, "deg_Mg_50C": 3}

    multi = "uniform_average"
    if raw_values:
        multi = "raw_values"

    def loss(y_true, y_pred):
        """
        y_true & y_pred may have more columns than needed for scoring
        select only necessary ones for scoring

        Expected shapes are (n, 5, seq_scored)
        """
        from sklearn.metrics import mean_squared_error

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        y_true = y_true[:, list(col_dict.values())]
        y_pred = y_pred[:, list(col_dict.values())]

        metric = mean_squared_error(y_true, y_pred, squared=False, multioutput=multi)
        return metric

    def loss_tf(y_true, y_pred):
        """
        Change is only to make the TF version compute the same MCRMSE intent:
        - select scored columns
        - compute RMSE over sequence positions, then mean over columns
        This avoids accidental axis misuse that can change training dynamics unpredictably.
        """
        import tensorflow as tf

        idx = tf.constant(list(col_dict.values()), dtype=tf.int32)
        y_true_s = tf.gather(y_true, idx, axis=1)  # (n,3,L)
        y_pred_s = tf.gather(y_pred, idx, axis=1)  # (n,3,L)

        mse_col = tf.reduce_mean(tf.square(y_true_s - y_pred_s), axis=2)  # (n,3)
        rmse_col = tf.sqrt(mse_col)  # (n,3)
        return tf.reduce_mean(rmse_col, axis=1)  # (n,)

    if not use_tf:
        return loss
    else:
        return loss_tf




## === cell 15
train_only_cols = [
    "reactivity_error",
    "deg_error_Mg_pH10",
    "deg_error_pH10",
    "deg_error_Mg_50C",
    "deg_error_50C",
]  # features only in train set
signal_cols = ["signal_to_noise", "SN_filter"]
drop_cols = [
    "seq_length",
    "seq_scored",  # don't actually use seq_length and seq_scored for training - just metadata
    "index",
    "id",
]  # also not actually useful for training
train_drop_cols = drop_cols + target_cols + train_only_cols + signal_cols

X_train = (
    train_df.drop(train_drop_cols, axis=1)  # selects only relevant columns
    .apply(
        lambda row: [e for e in row], axis=1
    )  # concatenates each element in row to list
    .apply(lambda e: np.array(e))  # creates 2d numpy array from those elements
)
X_train = np.stack(
    X_train.values, axis=0
)  # stacking might not work for test data, since there is two different shapes

y_train = (
    train_df[target_cols]
    .apply(
        lambda row: [e for e in row], axis=1
    )  # concatenates each element in row to list
    .apply(lambda e: np.array(e))  # creates 2d numpy array from those elements
)
y_train = np.stack(y_train.values, axis=0)



## === cell 16
X_train



## === cell 17
y_train



## === cell 18
sw = train_df["signal_to_noise"].values

import matplotlib.pyplot as plt

q = np.quantile(sw, np.linspace(0, 1, 21))
plt.plot(q, np.log1p(q + 5) / 2)

sw = np.log1p(sw + 5) / 2



## === cell 19
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




## === cell 20
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

    x = layers.Permute((2, 1))(x)  # (n, 5, seq_length)

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
def save_model(model):
    """
    WIP
    model is a fitted tensorflow keras model
    """
    pass




## === cell 22
import matplotlib.pyplot as plt

plt.figure(figsize=(13, 7))

all_metrics = set(history.history.keys()) - set(["lr", "learning_rate"])
for metric in all_metrics:
    metric_history = history.history[metric]
    plt.plot(metric_history, label=metric)
plt.legend()
plt.show()

lr_key = None
for k in ("lr", "learning_rate"):
    if k in history.history:
        lr_key = k
        break

if lr_key is not None:
    plt.plot(history.history[lr_key], label=lr_key)
    plt.legend()
    plt.show()



## --- ERROR in cell 22, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mNameError[0m                                 Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/1417779799.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m      3[0m [0mplt[0m[0;34m.[0m[0mfigure[0m[0;34m([0m[0mfigsize[0m[0;34m=[0m[0;34m([0m[0;36m13[0m[0;34m,[0m [0;36m7[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m      4[0m [0;34m[0m[0m
[0;32m----> 5[0;31m [0mall_metrics[0m [0;34m=[0m [0mset[0m[0;34m([0m[0mhistory[0m[0;34m.[0m[0mhistory[0m[0;34m.[0m[0mkeys[0m[0;34m([0m[0;34m)[0m[0;34m)[0m [0;34m-[0m [0mset[0m[0;34m([0m[0;34m[[0m[0;34m"lr"[0m[0;34m,[0m [0;34m"learning_rate"[0m[0;34m][0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      6[0m [0;32mfor[0m [0mmetric[0m [0;32min[0m [0mall_metrics[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m      7[0m     [0mmetric_history[0m [0;34m=[0m [0mhistory[0m[0;34m.[0m[0mhistory[0m[0;34m[[0m[0mmetric[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m

[0;31mNameError[0m: name 'history' is not defined

## === cell 23
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

try:
    from scikeras.wrappers import KerasRegressor  # type: ignore

    _USING_SCIKERAS = True
except Exception:
    from tensorflow.keras.wrappers import SKLearnRegressor as KerasRegressor  # type: ignore

    _USING_SCIKERAS = False

param_grid = {}

if _USING_SCIKERAS:
    model_sk = KerasRegressor(build_fn=make_model)
else:
    model_sk = KerasRegressor(
        model=make_model
    )  # make sk_learn wrapped model to use sklearn hyperparameter optimization
