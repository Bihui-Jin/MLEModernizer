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

0.48857

# 6. Current score

0.38909

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 0.38909) has done: 'I fix the runtime blockers without changing the intended modeling approach: (1) make `seqpos` creation robust by using `groupby().cumcount()` instead of `cumsum()` (which fails with object dtypes), (2) remove/avoid broken TensorFlow sklearn wrappers and train the same Keras model directly with `model.fit`, and (3) ensure train/test one-hot columns are aligned so `predict()` gets the same feature set. I also avoid the exploratory correlation/pairplot cells that currently error, so the notebook runs end-to-end. Finally, I build `id_seqpos` correctly and write a valid `submission.csv` with the exact required columns.'
- What this solution (achieved 0.38909) has done: 'I fix the TensorFlow import crash caused by the protobuf 6.x incompatibility by forcing TensorFlow to use the pure-Python protobuf implementation before importing TF (this is the standard workaround in Kaggle-like environments). I also make the training targets robust by dropping rows where any of the 5 target values are missing/NaN after `explode()`, which otherwise silently poisons training and hurts MCRMSE. These are minimal, score-aligned changes that preserve your feature engineering and exact Keras model/training loop. Finally, I keep the submission construction as-is but ensure it always writes a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os, random

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PYTHONHASHSEED"] = "0"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn

random.seed(0)
np.random.seed(0)



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
train_df = read_json("../input/stanford-covid-vaccine/train.json")

print(train_df["id"].nunique())
print(train_df.columns)
train_df.head()



## === cell 4
test_df = read_json("../input/stanford-covid-vaccine/test.json")

print("Features only in training set (not including target columns):")
set(train_df.columns) - set(test_df.columns) - set(target_cols)



## === cell 5
test_df.head()



## === cell 6
_ = """
! ls
! ls forna
! python forna/forna_server.py -s -d
"""



## === cell 7
_ = """
seq = train_df.loc[0, 'sequence']
struct = train_df.loc[0, 'structure']
seq, struct
"""




## === cell 8
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




## === cell 9
def feature_engineer(df, train=True, **kwargs):
    """
    Core logic preserved: explode lists to per-position rows, add seqpos, derive nucleotide and loop-type
    at each position, then one-hot encode categorical features.
    """
    unpack_cols = [
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
    ]

    if train:
        data = unpack_df_lists(df, unpack_cols)
    else:
        data = df.copy()
        data["temp"] = data.apply(lambda row: [0] * int(row["seq_length"]), axis=1)
        data = unpack_df_lists(data, "temp")
        del data["temp"]

    data["seqpos"] = data.groupby("id").cumcount().astype(np.int16)

    seq_temp = pd.concat([data["sequence"], data["seqpos"]], axis=1)
    data["nucleotide"] = seq_temp.apply(
        lambda row: row["sequence"][int(row["seqpos"])], axis=1
    )

    loop_temp = pd.concat([data["predicted_loop_type"], data["seqpos"]], axis=1)
    data["pred_loop_seqpos"] = loop_temp.apply(
        lambda row: row["predicted_loop_type"][int(row["seqpos"])], axis=1
    )

    data = pd.get_dummies(data, columns=["nucleotide", "pred_loop_seqpos"])
    return data




## === cell 10
unpack_cols = [
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
]

temp = train_df[train_df["SN_filter"] == 1].copy()
temp = feature_engineer(temp)

for b in ["A", "C", "G", "U"]:
    col = f"nucleotide_{b}"
    if col in temp.columns:
        print(b, temp[col].mean())

for b in ["S", "M", "I", "B", "H", "E", "X"]:
    col = f"pred_loop_seqpos_{b}"
    if col in temp.columns:
        print(b, temp[col].mean())

print("train_df memory (MB):", train_df.memory_usage(deep=True).sum() * 1e-6)
print("temp_df memory before (MB):", temp.memory_usage(deep=True).sum() * 1e-6)

temp["SN_filter"] = temp["SN_filter"].astype("uint8")
temp["seq_length"] = temp["seq_length"].astype("uint16")
temp["seq_scored"] = temp["seq_scored"].astype("uint16")
temp["seqpos"] = temp["seqpos"].astype("uint16")

for c in unpack_cols + ["signal_to_noise"]:
    if c in temp.columns:
        temp[c] = pd.to_numeric(temp[c], errors="coerce")

before = len(temp)
temp = temp.dropna(subset=target_cols).reset_index(drop=True)
after = len(temp)
print(f"Dropped {before-after} rows with NaN targets after explode/convert.")

print("temp_df memory after (MB):", temp.memory_usage(deep=True).sum() * 1e-6)
print(temp.dtypes.head(20))

train_df = temp
train_df.head()



## === cell 11
corr_data = None



## === cell 12
_ = None




## === cell 13
def score(raw_values=False, use_tf=False, **kwargs):
    """
    Competition metric: Mean Columnwise RMSE (MCRMSE).
    This implementation matches original intent but is not used for training in this script.
    """
    multi = "raw_values" if raw_values else "uniform_average"

    def loss(y_true, y_pred):
        from sklearn.metrics import mean_squared_error

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        return mean_squared_error(y_true, y_pred, squared=False, multioutput=multi)

    return loss




## === cell 14
train_only_cols = [
    "reactivity_error",
    "deg_error_Mg_pH10",
    "deg_error_pH10",
    "deg_error_Mg_50C",
    "deg_error_50C",
]
signal_cols = ["signal_to_noise", "SN_filter"]
drop_cols = [
    "sequence",
    "predicted_loop_type",
    "structure",
    "seq_length",
    "seq_scored",
    "index",
    "id",
]
train_drop_cols = drop_cols + target_cols + train_only_cols + signal_cols

X_train = train_df.drop(train_drop_cols, axis=1)
y_train = train_df[target_cols]

X_train = X_train.astype("float32")
y_train = y_train.astype("float32")

X_train.head()



## === cell 15
y_train.head()



## === cell 16
import tensorflow as tf
import tensorflow.keras.layers as layers

tf.random.set_seed(0)


def make_model(input_dim):
    inputs = tf.keras.Input(shape=(input_dim,))
    x = layers.Dense(100, activation="relu")(inputs)
    x = layers.Dense(30, activation="relu")(x)
    x = layers.Dense(5, activation="linear")(x)

    model = tf.keras.Model(inputs=inputs, outputs=x)
    model.compile(optimizer="adam", loss="mse", metrics=[])
    return model




## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 17
TF_FITPARAMS = {"epochs": 100, "batch_size": 5000, "verbose": 1}

model = make_model(X_train.shape[1])
history = model.fit(X_train, y_train, **TF_FITPARAMS)



## === cell 18
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
)  # kept from original; not used



## === cell 19
_ = None



## === cell 20
_ = None



## === cell 21
test_df = read_json("../input/stanford-covid-vaccine/test.json")
test_df.head()



## === cell 22
test_df = feature_engineer(test_df, train=False)

X_test_full = test_df.drop(drop_cols, axis=1)
X_test = X_test_full.reindex(columns=X_train.columns, fill_value=0).astype("float32")

test_df.head(), X_test.shape



## === cell 23
X_test.head()



## === cell 24
test_pred = model.predict(X_test, batch_size=8192, verbose=1)
test_pred[:3]



## === cell 25
sub_df = (
    test_df["id"].astype(str) + "_" + test_df["seqpos"].astype(int).astype(str)
).to_frame(name="id_seqpos")
pred_df = pd.DataFrame(test_pred, columns=target_cols)
sub_df = pd.concat(
    [sub_df.reset_index(drop=True), pred_df.reset_index(drop=True)], axis=1
)

sub_df.head()



## === cell 26
_ = None



## === cell 27
_ = None



## === cell 28
sample_sub = pd.read_csv("../input/stanford-covid-vaccine/sample_submission.csv")
assert list(sample_sub.columns) == list(
    sub_df.columns
), "Submission columns do not match required format."
assert len(sample_sub) == len(
    sub_df
), f"Row count mismatch: expected {len(sample_sub)}, got {len(sub_df)}"

sub_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub_df.shape)
print(sub_df.head())
