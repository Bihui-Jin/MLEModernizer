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

0.40796

# 6. Current score

0.28834

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 0.63848) has done: 'I fix the two root runtime blockers first: (1) remove the failing `tensorflow_addons` import (it’s incompatible with the provided TF/protobuf stack and isn’t actually used for training), and (2) replace the invalid `tf.reshape` on a KerasTensor with a proper Keras `Reshape` layer inside `build_model`. Next I restore missing imports and make the test preprocessing robust by not assuming any 130-length sequences (this dataset has only 107), preventing the “axes don’t match array” error. Finally, I ensure predictions are generated for all 107 positions, mapped back to the exact `sample_submission.csv` order, and written to a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.29786) has done: 'I fix the TensorFlow/protobuf import crash by setting the safe environment flag before importing TensorFlow and by avoiding unused TF components that trigger the protobuf factory path. Then I fix the Keras 2.18+ `ModelCheckpoint(save_weights_only=True)` filename requirement by switching to the required `.weights.h5` suffix and updating the corresponding `load_weights` calls so training artifacts are found. I also keep the existing architecture/training loops intact, but ensure the test preprocessing and submission building always align to the `sample_submission.csv` row order and write a valid `submission.csv`. These changes are runtime/stability fixes and should also improve the score versus the current broken-training path (since weights actually be saved/loaded and used for prediction).'
- What this solution (achieved 0.28834) has done: 'I fix the immediate runtime crash in the first cell caused by an incompatibility between TensorFlow 2.18 and the protobuf runtime by forcing the pure-Python protobuf implementation earlier and pinning the compatible protobuf API surface via environment variables before importing TensorFlow. Then I fix a couple of small logic bugs that can silently hurt training quality and score (the misuse of bitwise `~` in missing-value checks, and an invalid/ignored optimizer argument `decay` that can behave unexpectedly in TF 2.18). Finally, I keep the same models/training loops and submission-building logic, but add a safety alignment check to ensure the blended predictions match `sample_submission.csv` order and all required rows/columns are present before writing `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import warnings

warnings.filterwarnings("ignore")

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import gc, json, math, random, sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

import tensorflow as tf
import tensorflow.keras.layers as L
from tensorflow import keras
from sklearn.model_selection import train_test_split, KFold

SEED = 4
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("set up complete!")
print("TF version:", tf.__version__)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train = pd.read_json("/kaggle/input/stanford-covid-vaccine/train.json", lines=True)
test = pd.read_json("/kaggle/input/stanford-covid-vaccine/test.json", lines=True)
sample_sub = pd.read_csv("/kaggle/input/stanford-covid-vaccine/sample_submission.csv")

print("Data Load Complete")



## === cell 2
print(train.shape)
if not train.isnull().values.any():
    print("No missing values")
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
train["reactivity_over"] = train.apply(
    lambda x: add_list(x.reactivity, x.reactivity_error), axis=1
)
train["deg_Mg_pH10_over"] = train.apply(
    lambda x: add_list(x.deg_Mg_pH10, x.deg_error_Mg_pH10), axis=1
)
train["deg_pH10_over"] = train.apply(
    lambda x: add_list(x.deg_pH10, x.deg_error_pH10), axis=1
)
train["deg_Mg_50C_over"] = train.apply(
    lambda x: add_list(x.deg_Mg_50C, x.deg_error_Mg_50C), axis=1
)
train["deg_50C_over"] = train.apply(
    lambda x: add_list(x.deg_50C, x.deg_error_50C), axis=1
)

train["reactivity_under"] = train.apply(
    lambda x: subtract_list(x.reactivity, x.reactivity_error), axis=1
)
train["deg_Mg_pH10_under"] = train.apply(
    lambda x: subtract_list(x.deg_Mg_pH10, x.deg_error_Mg_pH10), axis=1
)
train["deg_pH10_under"] = train.apply(
    lambda x: subtract_list(x.deg_pH10, x.deg_error_pH10), axis=1
)
train["deg_Mg_50C_under"] = train.apply(
    lambda x: subtract_list(x.deg_Mg_50C, x.deg_error_Mg_50C), axis=1
)
train["deg_50C_under"] = train.apply(
    lambda x: subtract_list(x.deg_50C, x.deg_error_50C), axis=1
)

print("Additional target variables created!")



## === cell 5
train.head()



## === cell 6
print(test.shape)
if not test.isnull().values.any():
    print("No missing values")
test.head()



## === cell 7
print(sample_sub.shape)
if not sample_sub.isnull().values.any():
    print("No missing values")
sample_sub.head()



## === cell 8
target_cols_actual = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
target_cols_over = [
    "reactivity_over",
    "deg_Mg_pH10_over",
    "deg_pH10_over",
    "deg_Mg_50C_over",
    "deg_50C_over",
]
target_cols_under = [
    "reactivity_under",
    "deg_Mg_pH10_under",
    "deg_pH10_under",
    "deg_Mg_50C_under",
    "deg_50C_under",
]



## === cell 9
token2int = {x: i for i, x in enumerate("().ACGUBEHIMSX")}



## === cell 10
token2int["U"]



## === cell 11
cols = ["sequence", "structure", "predicted_loop_type"]
train[cols].applymap(lambda seq: [token2int[x] for x in seq])




## === cell 12
def preprocess_inputs(df, cols=["sequence", "structure", "predicted_loop_type"]):
    return np.transpose(
        np.array(
            df[cols].applymap(lambda seq: [token2int[x] for x in seq]).values.tolist()
        ),
        (0, 2, 1),
    )




## === cell 13
train_inputs = preprocess_inputs(train[train.signal_to_noise > 1])
train_y_actual = np.array(
    train[train.signal_to_noise > 1][target_cols_actual].values.tolist()
).transpose((0, 2, 1))
train_y_over = np.array(
    train[train.signal_to_noise > 1][target_cols_over].values.tolist()
).transpose((0, 2, 1))
train_y_under = np.array(
    train[train.signal_to_noise > 1][target_cols_under].values.tolist()
).transpose((0, 2, 1))



## === cell 14
print(train_inputs.shape)

print(train_y_actual.shape)
print(train_y_over.shape)
print(train_y_under.shape)



## === cell 15
inputs = tf.keras.layers.Input(shape=(107, 3))
embed = tf.keras.layers.Embedding(input_dim=len(token2int), output_dim=75)(inputs)
reshaped = tf.keras.layers.Reshape((107, 3 * 75))(embed)



## === cell 16
embed.shape




## === cell 17
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
    gru=False, seq_len=107, pred_len=68, dropout=0.5, embed_dim=100, hidden_dim=128
):
    inputs = tf.keras.layers.Input(shape=(seq_len, 3))

    embed = tf.keras.layers.Embedding(input_dim=len(token2int), output_dim=embed_dim)(
        inputs
    )

    reshaped = tf.keras.layers.Reshape((seq_len, 3 * embed_dim))(embed)

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

    adam = tf.keras.optimizers.Adam()
    model.compile(optimizer=adam, loss="mse")

    return model


print("Model structure defined")




## === cell 18
def lstm_model(seq_len=107, output_dim=100, dropout=0.5, pred_len=68):
    inputs = tf.keras.layers.Input(shape=(seq_len, 3))

    embed = tf.keras.layers.Embedding(input_dim=len(token2int), output_dim=output_dim)(
        inputs
    )
    reshaped = tf.keras.layers.Reshape(
        (seq_len, 3 * output_dim), input_shape=(seq_len, 3, output_dim)
    )(embed)

    hidden = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            128, dropout=dropout, kernel_initializer="orthogonal", return_sequences=True
        )
    )(reshaped)
    hidden = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            128, dropout=dropout, kernel_initializer="orthogonal", return_sequences=True
        )
    )(hidden)
    hidden = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            128, dropout=dropout, kernel_initializer="orthogonal", return_sequences=True
        )
    )(hidden)

    truncated = hidden[:, :pred_len]
    output = tf.keras.layers.Dense(5, activation="linear")(truncated)

    model = tf.keras.Model(inputs=inputs, outputs=output)

    adam = tf.keras.optimizers.Adam(learning_rate=0.01)

    model.compile(loss="mse", optimizer=adam)
    return model




## === cell 19
train_data, val_data, train_labels, val_labels = train_test_split(
    train_inputs, train_y_over, test_size=0.2, random_state=4
)



## === cell 20
lr_callback = tf.keras.callbacks.ReduceLROnPlateau()



## === cell 21
smpl_lstm = lstm_model(seq_len=107, output_dim=100, dropout=0.5, pred_len=68)

sv_smpl_lstm = tf.keras.callbacks.ModelCheckpoint(
    "model_smpl_lstm.weights.h5", save_weights_only=True
)



## === cell 22
smpl_lstm.summary()



## === cell 23
history_smpl_lstm = smpl_lstm.fit(
    train_data,
    train_labels,
    validation_data=(val_data, val_labels),
    batch_size=64,
    epochs=80,
    callbacks=[lr_callback, sv_smpl_lstm],
    verbose=2,
)

print(
    f"Min training loss={min(history_smpl_lstm.history['loss'])}, min validation loss={min(history_smpl_lstm.history['val_loss'])}"
)



## === cell 24
gru = build_model(gru=True, seq_len=107, pred_len=68)

sv_gru = tf.keras.callbacks.ModelCheckpoint(
    "model_gru.weights.h5", save_weights_only=True
)



## === cell 25
gru.summary()



## === cell 26
history_gru = gru.fit(
    train_data,
    train_labels,
    validation_data=(val_data, val_labels),
    batch_size=64,
    epochs=70,
    callbacks=[lr_callback, sv_gru],
    verbose=2,
)

print(
    f"Min training loss={min(history_gru.history['loss'])}, min validation loss={min(history_gru.history['val_loss'])}"
)



## === cell 27
fig, ax = plt.subplots(1, 2, figsize=(20, 10))

ax[0].plot(history_gru.history["loss"])
ax[0].plot(history_gru.history["val_loss"])

ax[1].plot(history_smpl_lstm.history["loss"])
ax[1].plot(history_smpl_lstm.history["val_loss"])

ax[0].set_title("GRU")
ax[1].set_title("SMPL_LSTM")

ax[0].legend(["train", "validation"], loc="upper right")
ax[1].legend(["train", "validation"], loc="upper right")

ax[0].set_ylabel("Loss")
ax[0].set_xlabel("Epoch")
ax[1].set_ylabel("Loss")
ax[1].set_xlabel("Epoch")



## === cell 28
public_df = test.query("seq_length == 107").copy()
private_df = test.query("seq_length == 130").copy()  # likely empty in this environment

public_inputs = preprocess_inputs(public_df)
private_inputs = None
if len(private_df) > 0:
    private_inputs = preprocess_inputs(private_df)

print("Test data prepared!")
print("public_df:", public_df.shape, "private_df:", private_df.shape)



## === cell 29
gru_short = build_model(gru=True, seq_len=107, pred_len=107)
lstm_short = lstm_model(seq_len=107, output_dim=100, dropout=0.5, pred_len=107)

gru_short.load_weights("model_gru.weights.h5")
lstm_short.load_weights("model_smpl_lstm.weights.h5")

gru_long = None
lstm_long = None
if private_inputs is not None:
    gru_long = build_model(gru=True, seq_len=130, pred_len=130)
    lstm_long = lstm_model(seq_len=130, output_dim=100, dropout=0.5, pred_len=130)
    gru_long.load_weights("model_gru.weights.h5")
    lstm_long.load_weights("model_smpl_lstm.weights.h5")

print("models built and weights loaded")



## === cell 30
gru_public_preds = gru_short.predict(public_inputs, verbose=0)
lstm_public_preds = lstm_short.predict(public_inputs, verbose=0)

gru_private_preds = None
lstm_private_preds = None
if private_inputs is not None:
    gru_private_preds = gru_long.predict(private_inputs, verbose=0)
    lstm_private_preds = lstm_long.predict(private_inputs, verbose=0)

print("Predictions generated")
print(
    "gru_public_preds:",
    gru_public_preds.shape,
    "lstm_public_preds:",
    lstm_public_preds.shape,
)



## === cell 31
preds_gru = []

for df, preds in [(public_df, gru_public_preds)] + (
    [(private_df, gru_private_preds)] if gru_private_preds is not None else []
):
    for i, uid in enumerate(df.id):
        single_pred = preds[i]
        single_df = pd.DataFrame(single_pred, columns=target_cols_over)
        single_df["id_seqpos"] = [f"{uid}_{x}" for x in range(single_df.shape[0])]
        preds_gru.append(single_df)

preds_gru_df = pd.concat(preds_gru, ignore_index=True)
preds_gru_df.head()



## === cell 32
preds_lstm = []

for df, preds in [(public_df, lstm_public_preds)] + (
    [(private_df, lstm_private_preds)] if lstm_private_preds is not None else []
):
    for i, uid in enumerate(df.id):
        single_pred = preds[i]
        single_df = pd.DataFrame(single_pred, columns=target_cols_over)
        single_df["id_seqpos"] = [f"{uid}_{x}" for x in range(single_df.shape[0])]
        preds_lstm.append(single_df)

preds_lstm_df = pd.concat(preds_lstm, ignore_index=True)
preds_lstm_df.head()



## === cell 33
lstm_weight = 0.65
gru_weight = 0.35

blend_preds_df = pd.DataFrame()
blend_preds_df["id_seqpos"] = preds_gru_df["id_seqpos"]
blend_preds_df["reactivity"] = (
    gru_weight * preds_gru_df["reactivity_over"]
    + lstm_weight * preds_lstm_df["reactivity_over"]
)
blend_preds_df["deg_Mg_pH10"] = (
    gru_weight * preds_gru_df["deg_Mg_pH10_over"]
    + lstm_weight * preds_lstm_df["deg_Mg_pH10_over"]
)
blend_preds_df["deg_pH10"] = (
    gru_weight * preds_gru_df["deg_pH10_over"]
    + lstm_weight * preds_lstm_df["deg_pH10_over"]
)
blend_preds_df["deg_Mg_50C"] = (
    gru_weight * preds_gru_df["deg_Mg_50C_over"]
    + lstm_weight * preds_lstm_df["deg_Mg_50C_over"]
)
blend_preds_df["deg_50C"] = (
    gru_weight * preds_gru_df["deg_50C_over"]
    + lstm_weight * preds_lstm_df["deg_50C_over"]
)

blend_preds_df.head()



## === cell 34
submission = sample_sub[["id_seqpos"]].merge(blend_preds_df, on="id_seqpos", how="left")

pred_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
submission[pred_cols] = submission[pred_cols].fillna(0.0)

submission = (
    submission.set_index("id_seqpos").reindex(sample_sub["id_seqpos"]).reset_index()
)

print(submission.head())
print("Submission rows:", len(submission), "Expected:", len(sample_sub))
print("Any NaNs left?", submission[pred_cols].isna().any().any())



## === cell 35
submission.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")
print("submission.csv columns:", list(submission.columns))
print("submission.csv shape:", submission.shape)
