# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


# 1. Kaggle task description

## Overview
Predict likely degradation rates at each base of an RNA molecule.

## Metric
Mean columnwise root mean squared error:

$\textrm{MCRMSE} = \frac{1}{N_{t}}\sum_{j=1}^{N_{t}}\sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_{ij} - \hat{y}_{ij})^2}$

where $N_{t}$ is the number of scored ground truth target columns, and $y$ and $\hat{y}$ are the actual and predicted values, respectively.

There are multiple ground truth values provided in the training data. While the submission format requires all 5 to be predicted, only the following are scored: reactivity, deg_Mg_pH10, and deg_Mg_50C.

## Submission Formats
For each sample `id` in the test set, you must predict targets for *each* sequence position (`seqpos`), one per row. If the length of the `sequence` of an `id` is, e.g., 107, then you should make 107 predictions. Positions greater than the `seq_scored` value of a sample are not scored, but still need a value in the solution file.

```csv
id_seqpos,reactivity,deg_Mg_pH10,deg_pH10,deg_Mg_50C,deg_50C
id_d190610e8_0,0.1,0.3,0.2,0.5,0.4
id_d190610e8_1,0.3,0.2,0.5,0.4,0.2
id_d190610e8_2,0.5,0.4,0.2,0.1,0.2
etc.
```

## Dataset 
- **train.json** - the training data
- **test.json** - the test set, without any columns associated with the ground truth.
- **sample_submission.csv** - a sample submission file in the correct format

#### Columns
- `id` - An arbitrary identifier for each sample.
- `seq_scored` - (68 in Train and Public Test, 68 in Private Test) Integer value denoting the number of positions used in scoring with predicted values. This should match the length of `reactivity`, `deg_*` and `*_error_*` columns.
- `seq_length` - (107 in Train and Public Test, 107 in Private Test) Integer values, denotes the length of `sequence`.
- `sequence` - (1x107 string in Train and Public Test, 107 in Private Test) Describes the RNA sequence, a combination of `A`, `G`, `U`, and `C` for each sample. Should be 107 characters long, and the first 68 bases should correspond to the 68 positions specified in `seq_scored` (note: indexed starting at 0).
- `structure` - (1x107 string in Train and Public Test, 107 in Private Test) An array of `(`, `)`, and `.` characters that describe whether a base is estimated to be paired or unpaired. Paired bases are denoted by opening and closing parentheses e.g. (....) means that base 0 is paired to base 5, and bases 1-4 are unpaired.
- `reactivity` - (1x68 vector in Train and Public Test, 1x68 in Private Test) An array of floating point numbers, should have the same length as `seq_scored`. These numbers are reactivity values for the first 68 bases as denoted in `sequence`, and used to determine the likely secondary structure of the RNA sample.
- `deg_pH10` - (1x68 vector in Train and Public Test, 1x68 in Private Test) An array of floating point numbers, should have the same length as `seq_scored`. These numbers are reactivity values for the first 68 bases as denoted in `sequence`, and used to determine the likelihood of degradation at the base/linkage after incubating without magnesium at high pH (pH 10).
- `deg_Mg_pH10` - (1x68 vector in Train and Public Test, 1x68 in Private Test) An array of floating point numbers, should have the same length as `seq_scored`. These numbers are reactivity values for the first 68 bases as denoted in `sequence`, and used to determine the likelihood of degradation at the base/linkage after incubating with magnesium in high pH (pH 10).
- `deg_50C` - (1x68 vector in Train and Public Test, 1x68 in Private Test) An array of floating point numbers, should have the same length as `seq_scored`. These numbers are reactivity values for the first 68 bases as denoted in `sequence`, and used to determine the likelihood of degradation at the base/linkage after incubating without magnesium at high temperature (50 degrees Celsius).
- `deg_Mg_50C` - (1x68 vector in Train and Public Test, 1x68 in Private Test) An array of floating point numbers, should have the same length as `seq_scored`. These numbers are reactivity values for the first 68 bases as denoted in `sequence`, and used to determine the likelihood of degradation at the base/linkage after incubating with magnesium at high temperature (50 degrees Celsius).
- `*_error_*` - An array of floating point numbers, should have the same length as the corresponding `reactivity` or `deg_*` columns, calculated errors in experimental values obtained in `reactivity` and `deg_*` columns.
- `predicted_loop_type` - (1x107 string) Describes the structural context (also referred to as 'loop type')of each character in `sequence`. Loop types assigned by bpRNA from Vienna RNAfold 2 structure. From the bpRNA_documentation: S: paired "Stem" M: Multiloop I: Internal loop B: Bulge H: Hairpin loop E: dangling End X: eXternal loop
    - `S/N filter` Indicates if the sample passed filters described below in `Additional Notes`.

#### Additional Notes
At the beginning of the competition, Stanford scientists have data on 2400 RNA sequences of length 107. For technical reasons, measurements cannot be carried out on the final bases of these RNA sequences, so we have experimental data (ground truth) in 5 conditions for the first 68 bases.

We have split out 240 of these 2400 sequences for a public test set to allow for continuous evaluation through the competition, on the public leaderboard. These sequences, in `test.json`, have been additionally filtered based on three criteria detailed below to ensure that this subset is not dominated by any large cluster of RNA molecules with poor data, which might bias the public leaderboard. The remaining 2160 sequences for which we have data are in `train.json`.

For our final and most important scoring (the Private Leaderbooard), Stanford scientists are carrying out measurements on 240 new RNAs. For these data, we expect to have measurements for the first 68 bases, again missing the ends of the RNA. These sequences constitute the 240 sequences in `test.json`.

For those interested in how the sequences in `test.json` were filtered, here were the steps to ensure a diverse and high quality test set for public leaderboard scoring:

1. Minimum value across all 5 conditions must be greater than -0.5.
2. Mean signal/noise across all 5 conditions must be greater than 1.0. [Signal/noise is defined as mean( measurement value over 68 nts )/mean( statistical error in measurement value over 68 nts)]
3. To help ensure sequence diversity, the resulting sequences were clustered into clusters with less than 50% sequence similarity, and the 240 test set sequences were chosen from clusters with 3 or fewer members. That is, any sequence in the test set should be sequence similar to at most 2 other sequences.

Note that these filters have not been applied to the 2160 RNAs in the public training data `train.json` -- some of those measurements have negative values or poor signal-to-noise, or some RNA sequences have near-identical sequences in that set. But we are providing all those data in case competitors can squeeze out more signal.

# 2. Python version

3.8

# 3. Installed packages

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

# 4. Data file paths

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

# 5. Target score

0.41112

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import sys
import numpy as np
import pandas as pd

import subprocess

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "protobuf==4.25.3"], check=False
)

import tensorflow as tf
import tensorflow.keras.layers as layers
import matplotlib.pyplot as plt

np.random.seed(42)
tf.random.set_seed(42)



## === cell 1
target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]




## === cell 2
def read_json(filename):
    """
    reads in train/test json data as pandas DataFrame
    """
    with open(filename, "r") as f:
        df = pd.read_json(path_or_buf=f, orient="records", lines=True)
    return df




## === cell 3
train_path = "../input/stanford-covid-vaccine/train.json"
test_path = "../input/stanford-covid-vaccine/test.json"
sample_sub_path = "../input/stanford-covid-vaccine/sample_submission.csv"

train_df = read_json(train_path)

print(train_df["id"].nunique())
print(train_df.columns)
train_df.head()



## === cell 4
test_df = read_json(test_path)

print("Features only in training set (not including target columns):")
set(train_df.columns) - set(test_df.columns) - set(target_cols)



## === cell 5
test_df.head()




## === cell 6
def unpack_df_lists(df, col_names):
    """
    turn list-like elements of dataframe into tabular data
    """
    if isinstance(col_names, str):
        col_names = [col_names]
    all_series = [df[c] for c in col_names]
    unpacked = [ser.explode() for ser in all_series]
    data = pd.concat(unpacked, axis=1)
    original = df.drop(col_names, axis=1)
    data = original.join(data)
    return data




## === cell 7
tokenize_cols = ["sequence", "structure", "predicted_loop_type"]


def tokenize_df(df, tokenizer, cols=tokenize_cols):
    """
    tokenizer is a tensorflow keras Tokenizer that has been already fitted on text
    """
    data = df.copy()
    for c in cols:
        data[c] = tokenizer.texts_to_sequences(data[c])
    return data




## === cell 8
from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer(filters=None, lower=False, char_level=True)
tokenizer.fit_on_texts("().ACGUBEHIMSX")

temp = train_df[train_df["SN_filter"] == 1].copy()
temp = tokenize_df(temp, tokenizer)

train_df = temp
train_df.head()




## === cell 9
def score(raw_values=False, use_tf=False, **kwargs):
    """
    Competition metric: MCRMSE over scored columns only:
    reactivity, deg_Mg_pH10, deg_Mg_50C

    Expected model output shape: (batch, 5, L_scored)
    """
    col_dict = {"reactivity": 0, "deg_Mg_pH10": 1, "deg_Mg_50C": 3}
    scored_idx = list(col_dict.values())

    def loss_np(y_true, y_pred):
        from sklearn.metrics import mean_squared_error

        y_true = np.asarray(y_true)[:, scored_idx, :]
        y_pred = np.asarray(y_pred)[:, scored_idx, :]
        yt = y_true.transpose(0, 2, 1).reshape(-1, len(scored_idx))
        yp = y_pred.transpose(0, 2, 1).reshape(-1, len(scored_idx))
        multi = "raw_values" if raw_values else "uniform_average"
        return mean_squared_error(yt, yp, squared=False, multioutput=multi)

    def loss_tf(y_true, y_pred):
        y_true_s = tf.gather(y_true, scored_idx, axis=1)
        y_pred_s = tf.gather(y_pred, scored_idx, axis=1)

        mse_per_target = tf.reduce_mean(
            tf.square(y_true_s - y_pred_s), axis=[0, 2]
        )  # (3,)
        rmse_per_target = tf.sqrt(mse_per_target)  # (3,)
        return tf.reduce_mean(rmse_per_target)  # scalar

    return loss_tf if use_tf else loss_np




## === cell 10
train_only_cols = [
    "reactivity_error",
    "deg_error_Mg_pH10",
    "deg_error_pH10",
    "deg_error_Mg_50C",
    "deg_error_50C",
]
signal_cols = ["signal_to_noise", "SN_filter"]
drop_cols = ["seq_length", "seq_scored", "index", "id"]
train_drop_cols = drop_cols + target_cols + train_only_cols + signal_cols

X_train = (
    train_df.drop(train_drop_cols, axis=1)
    .apply(lambda row: [e for e in row], axis=1)
    .apply(lambda e: np.array(e, dtype=np.int32))
)
X_train = np.stack(X_train.values, axis=0)

y_train = (
    train_df[target_cols]
    .apply(lambda row: [e for e in row], axis=1)
    .apply(lambda e: np.array(e, dtype=np.float32))
)
y_train = np.stack(y_train.values, axis=0)

y_train = y_train[:, :, :68].astype(np.float32)

print("X_train:", X_train.shape, X_train.dtype)
print("y_train:", y_train.shape, y_train.dtype)



## === cell 11
sw = train_df["signal_to_noise"].values.astype(np.float32)

q = np.quantile(sw, np.linspace(0, 1, 21))
plt.figure(figsize=(6, 3))
plt.plot(q, np.log1p(q + 5) / 2)
plt.title("sample_weight transform")
plt.tight_layout()
plt.show()

sw = np.log1p(sw + 5).astype(np.float32) / 2




## === cell 12
def make_model():
    EMBEDDING_PARAMS = {
        "input_dim": len(tokenizer.word_index) + 1,
        "output_dim": 100,
    }

    shape = (3, None)
    inputs = tf.keras.Input(shape=shape)
    embed = layers.Embedding(**EMBEDDING_PARAMS)(inputs)

    def rnn_layer():
        return layers.Bidirectional(layers.LSTM(30, return_sequences=True))

    rnn_layers = []
    for i in range(embed.shape[1]):
        r = rnn_layer()(embed[:, i])
        r = rnn_layer()(r)
        rnn_layers.append(r)

    x = layers.Concatenate()(rnn_layers)
    x = layers.Dense(100, activation="relu")(x)
    x = layers.Dense(5, activation="linear")(x)  # (B, L, 5)

    x = layers.Permute((2, 1))(x)  # (B, 5, L)

    x = layers.Cropping1D(cropping=(0, 39))(x)  # crop last 39 => length 68

    model = tf.keras.Model(inputs=inputs, outputs=x)
    model.compile(optimizer="adam", loss=score(use_tf=True), metrics=["mse"])
    return model




## === cell 13
TF_FITPARAMS = {"epochs": 100, "batch_size": 100, "sample_weight": sw}

model = make_model()
model.summary()

history = model.fit(X_train, y_train, **TF_FITPARAMS)



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3952833090.py in <cell line: 0>()
      1 TF_FITPARAMS = {"epochs": 100, "batch_size": 100, "sample_weight": sw}
      2 
----> 3 model = make_model()
      4 model.summary()
      5 

/tmp/ipykernel_11/260529440.py in make_model()
     28 
     29     # Keep exactly first 68 positions (107 -> 68) matching training labels
---> 30     x = layers.Cropping1D(cropping=(0, 39))(x)  # crop last 39 => length 68
     31 
     32     model = tf.keras.Model(inputs=inputs, outputs=x)

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/layers/reshaping/cropping1d.py in compute_output_shape(self, input_shape)
     53             length = input_shape[1] - self.cropping[0] - self.cropping[1]
     54             if length <= 0:
---> 55                 raise ValueError(
     56                     "`cropping` parameter of `Cropping1D` layer must be "
     57                     "smaller than the input length. Received: input_shape="

ValueError: Exception encountered when calling Cropping1D.call().

`cropping` parameter of `Cropping1D` layer must be smaller than the input length. Received: input_shape=(None, 5, None), cropping=(0, 39)

Arguments received by Cropping1D.call():
  • args=('<KerasTensor shape=(None, 5, None), dtype=float32, sparse=False, name=keras_tensor_14>',)
  • kwargs=<class 'inspect._empty'>

## === cell 14
plt.figure(figsize=(7, 4))
for metric, metric_history in history.history.items():
    plt.plot(metric_history, label=metric)
plt.legend()
plt.title("Training history")
plt.tight_layout()
plt.show()



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4264063169.py in <cell line: 0>()
      1 plt.figure(figsize=(7, 4))
----> 2 for metric, metric_history in history.history.items():
      3     plt.plot(metric_history, label=metric)
      4 plt.legend()
      5 plt.title("Training history")

NameError: name 'history' is not defined

## === cell 15
test_df = read_json(test_path)
sample_sub = pd.read_csv(sample_sub_path)

test_tok = tokenize_df(test_df.copy(), tokenizer)

X_test = (
    test_tok.drop(drop_cols, axis=1)
    .apply(lambda row: [e for e in row], axis=1)
    .apply(lambda e: np.array(e, dtype=np.int32))
)
X_test = np.stack(X_test.values, axis=0)

print("X_test:", X_test.shape)



## === cell 16
test_pred = model.predict(X_test, batch_size=100)  # expected (n_test,5,68)
print("test_pred:", test_pred.shape, test_pred.dtype)




## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/681917830.py in <cell line: 0>()
----> 1 test_pred = model.predict(X_test, batch_size=100)  # expected (n_test,5,68)
      2 print("test_pred:", test_pred.shape, test_pred.dtype)
      3 
      4 

NameError: name 'model' is not defined

## === cell 17
def build_submission_aligned_to_sample(
    test_df_raw, preds_5x68, sample_submission, seq_length=107
):
    """
    Create predictions for all (id, seqpos 0..seq_length-1) and then align to sample_submission['id_seqpos'].
    For seqpos >= 68 (unscored), pad with 0 as in original code.
    """
    id_list = test_df_raw["id"].tolist()
    n = len(id_list)

    if preds_5x68.shape[0] != n or preds_5x68.shape[1] != 5:
        raise ValueError(
            f"Unexpected preds shape {preds_5x68.shape}, expected ({n},5,68)"
        )

    pred_by_id = {id_list[i]: preds_5x68[i] for i in range(n)}

    out = sample_submission[["id_seqpos"]].copy()
    id_seqpos_split = out["id_seqpos"].str.rsplit("_", n=1, expand=True)
    out["_id"] = id_seqpos_split[0]
    out["_seqpos"] = id_seqpos_split[1].astype(int)

    for j, col in enumerate(target_cols):
        vals = np.zeros(len(out), dtype=np.float32)
        mask_scored = out["_seqpos"].values < 68
        ids_scored = out.loc[mask_scored, "_id"].values
        pos_scored = out.loc[mask_scored, "_seqpos"].values

        gathered = np.empty(mask_scored.sum(), dtype=np.float32)
        for k in range(len(gathered)):
            gathered[k] = pred_by_id[ids_scored[k]][j, pos_scored[k]]
        vals[mask_scored] = gathered

        out[col] = vals

    out = out.drop(columns=["_id", "_seqpos"])
    return out


sub_df = build_submission_aligned_to_sample(
    test_df, test_pred, sample_sub, seq_length=107
)
print(sub_df.shape)
sub_df.head()



## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1313181113.py in <cell line: 0>()
     39 
     40 sub_df = build_submission_aligned_to_sample(
---> 41     test_df, test_pred, sample_sub, seq_length=107
     42 )
     43 print(sub_df.shape)

NameError: name 'test_pred' is not defined

## === cell 18
expected_cols = ["id_seqpos"] + target_cols
assert list(sub_df.columns) == expected_cols, f"Bad columns: {sub_df.columns}"
assert sub_df.shape[0] == sample_sub.shape[0], "Row count mismatch vs sample_submission"

sub_path = "submission.csv"
sub_df.to_csv(sub_path, index=False)
print("Wrote:", sub_path)
print(pd.read_csv(sub_path).head())

## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2300746578.py in <cell line: 0>()
      1 expected_cols = ["id_seqpos"] + target_cols
----> 2 assert list(sub_df.columns) == expected_cols, f"Bad columns: {sub_df.columns}"
      3 assert sub_df.shape[0] == sample_sub.shape[0], "Row count mismatch vs sample_submission"
      4 
      5 sub_path = "submission.csv"

NameError: name 'sub_df' is not defined
