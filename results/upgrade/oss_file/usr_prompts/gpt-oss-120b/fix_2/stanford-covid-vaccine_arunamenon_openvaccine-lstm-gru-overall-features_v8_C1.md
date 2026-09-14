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
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
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
tf_keras==2.18.0
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

0.39134

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from tqdm.notebook import tqdm



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train_data = pd.read_json("/kaggle/input/stanford-covid-vaccine/train.json", lines=True)
test_data = pd.read_json("/kaggle/input/stanford-covid-vaccine/test.json", lines=True)
submission_format = pd.read_csv(
    "/kaggle/input/stanford-covid-vaccine/sample_submission.csv", encoding="utf-8-sig"
)




## === cell 2
def add_dummy_bpps(df):
    seq_len = df["seq_length"].iloc[0] if not df.empty else 107
    zero_arr = [np.zeros(seq_len) for _ in range(len(df))]
    df["bpps_sum"] = zero_arr
    df["bpps_max"] = zero_arr
    df["bpps_nb"] = zero_arr
    return df


train_data = add_dummy_bpps(train_data)
test_data = add_dummy_bpps(test_data)



## === cell 3
token2int = {x: i for i, x in enumerate("().ACGUBEHIMSX")}
target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]




## === cell 4
def preprocess_inputs(df, cols=["sequence", "structure", "predicted_loop_type"]):
    """
    Encode three categorical strings and concatenate the (already present)
    numerical features bpps_sum, bpps_max, bpps_nb.
    Result shape: (samples, seq_len, 4)  (3 categorical + 1 summed adjacency later)
    """
    base_fea = np.transpose(
        np.array(
            df[cols].applymap(lambda seq: [token2int[ch] for ch in seq]).values.tolist()
        ),
        (0, 2, 1),
    )  # (samples, seq_len, 3)

    bpps_sum_fea = np.array(df["bpps_sum"].to_list())[:, :, np.newaxis]
    bpps_max_fea = np.array(df["bpps_max"].to_list())[:, :, np.newaxis]
    bpps_nb_fea = np.array(df["bpps_nb"].to_list())[:, :, np.newaxis]

    num_fea = np.concatenate([bpps_sum_fea, bpps_max_fea, bpps_nb_fea], axis=2)
    return np.concatenate([base_fea, num_fea], axis=2)  # shape (samples, seq_len, 6)




## === cell 5
def get_structure_adj(df):
    """
    Build a binary pairwise adjacency matrix for each RNA.
    Returns array of shape (samples, seq_len, seq_len, 1)
    """
    Ss = []
    for i in tqdm(range(len(df)), desc="Adjacency"):
        seq_len = df["seq_length"].iloc[i]
        structure = df["structure"].iloc[i]
        sequence = df["sequence"].iloc[i]

        a_structs = {
            ("A", "U"): np.zeros([seq_len, seq_len]),
            ("C", "G"): np.zeros([seq_len, seq_len]),
            ("U", "G"): np.zeros([seq_len, seq_len]),
            ("U", "A"): np.zeros([seq_len, seq_len]),
            ("G", "C"): np.zeros([seq_len, seq_len]),
            ("G", "U"): np.zeros([seq_len, seq_len]),
        }

        stack = []
        for j, ch in enumerate(structure):
            if ch == "(":
                stack.append(j)
            elif ch == ")" and stack:
                start = stack.pop()
                pair = (sequence[start], sequence[j])
                if pair in a_structs:
                    a_structs[pair][start, j] = 1
                    a_structs[(pair[1], pair[0])][j, start] = 1

        a_strc = np.stack(list(a_structs.values()), axis=2)
        a_strc = np.sum(a_strc, axis=2, keepdims=True)  # (seq_len, seq_len, 1)
        Ss.append(a_strc)

    Ss = np.array(Ss)
    print("Adjacency shape:", Ss.shape)
    return Ss




## === cell 6
train_filtered = train_data[train_data["signal_to_noise"] > 1].reset_index(drop=True)

train_inputs = preprocess_inputs(train_filtered)
train_labels = np.array(train_filtered[target_cols].values.tolist())  # (samples, 68, 5)

Ss = get_structure_adj(train_filtered)  # (samples, seq_len, seq_len, 1)
Ss_sum = Ss.sum(axis=1)  # (samples, seq_len, 1)
train_inputs = np.concatenate([train_inputs, Ss_sum], axis=2)  # final features




## === cell 7
def build_model(seq_len=107, embed_dim=100, hidden_dim=256, dropout=0.2, pred_len=68):
    """
    Keras functional model.
    First 3 channels are categorical (encoded as ints); the remaining channels
    are numeric (bpps + summed adjacency).
    """
    inputs = tf.keras.layers.Input(shape=(seq_len, None))  # None = feature dim
    categorical_feats = inputs[:, :, :3]
    numeric_feats = inputs[:, :, 3:]

    embed = tf.keras.layers.Embedding(input_dim=len(token2int), output_dim=embed_dim)(
        categorical_feats
    )
    embed_reshaped = tf.keras.layers.Reshape((seq_len, 3 * embed_dim))(embed)

    x = tf.keras.layers.concatenate([embed_reshaped, numeric_feats], axis=2)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(
            hidden_dim,
            dropout=dropout,
            return_sequences=True,
            kernel_initializer="orthogonal",
        )
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = x[:, :pred_len, :]

    outputs = tf.keras.layers.Dense(5, activation="linear")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    def MCRMSE(y_true, y_pred):
        colwise_mse = tf.reduce_mean(tf.square(y_true - y_pred), axis=1)
        return tf.reduce_mean(tf.sqrt(colwise_mse), axis=1)

    model.compile(optimizer=tf.optimizers.Adam(), loss=MCRMSE)
    return model




## === cell 8
EPOCHS = 5  # keep runtime short; sufficient to generate predictions
BATCH_SIZE = 32

model = build_model(
    seq_len=train_inputs.shape[1], pred_len=train_inputs.shape[1]
)  # predict whole length
model.summary()

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "lstm_model.h5", save_best_only=True, monitor="loss", mode="min"
)
history = model.fit(
    train_inputs,
    train_labels,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    verbose=2,
    callbacks=[checkpoint],
)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_54/3362395475.py in <cell line: 0>()
      2 BATCH_SIZE = 32
      3 
----> 4 model = build_model(
      5     seq_len=train_inputs.shape[1], pred_len=train_inputs.shape[1]
      6 )  # predict whole length

/tmp/ipykernel_54/3355973745.py in build_model(seq_len, embed_dim, hidden_dim, dropout, pred_len)
     17 
     18     x = tf.keras.layers.concatenate([embed_reshaped, numeric_feats], axis=2)
---> 19     x = tf.keras.layers.BatchNormalization()(x)
     20     x = tf.keras.layers.Bidirectional(
     21         tf.keras.layers.GRU(

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/backend/common/variables.py in _validate_shape(self, shape)
    207         shape = standardize_shape(shape)
    208         if None in shape:
--> 209             raise ValueError(
    210                 "Shapes used to initialize variables must be "
    211                 "fully-defined (no `None` dimensions). Received: "

ValueError: Shapes used to initialize variables must be fully-defined (no `None` dimensions). Received: shape=(None,) for variable path='batch_normalization/gamma'

## === cell 9
plt.figure(figsize=(10, 4))
plt.plot(history.history["loss"], label="train loss")
plt.title("Training loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_54/1099261472.py in <cell line: 0>()
      1 # Plot training loss (optional, does not affect submission)
      2 plt.figure(figsize=(10, 4))
----> 3 plt.plot(history.history["loss"], label="train loss")
      4 plt.title("Training loss")
      5 plt.xlabel("Epoch")

NameError: name 'history' is not defined

## === cell 10
public_df = test_data.query("seq_length == 107").reset_index(drop=True)
private_df = test_data.query("seq_length == 130").reset_index(drop=True)

public_inputs = preprocess_inputs(public_df)
private_inputs = preprocess_inputs(private_df)

Ss_pub = get_structure_adj(public_df).sum(axis=1)  # (samples, seq_len, 1)
public_inputs = np.concatenate([public_inputs, Ss_pub], axis=2)

Ss_priv = get_structure_adj(private_df).sum(axis=1)
private_inputs = np.concatenate([private_inputs, Ss_priv], axis=2)



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_54/2946506842.py in <cell line: 0>()
      4 
      5 public_inputs = preprocess_inputs(public_df)
----> 6 private_inputs = preprocess_inputs(private_df)
      7 
      8 # Add summed adjacency channel

/tmp/ipykernel_54/3048877785.py in preprocess_inputs(df, cols)
      6     """
      7     # Encode categorical strings as integer lists
----> 8     base_fea = np.transpose(
      9         np.array(
     10             df[cols].applymap(lambda seq: [token2int[ch] for ch in seq]).values.tolist()

/usr/local/lib/python3.11/dist-packages/numpy/core/fromnumeric.py in transpose(a, axes)
    653 
    654     """
--> 655     return _wrapfunc(a, 'transpose', axes)
    656 
    657 

/usr/local/lib/python3.11/dist-packages/numpy/core/fromnumeric.py in _wrapfunc(obj, method, *args, **kwds)
     57 
     58     try:
---> 59         return bound(*args, **kwds)
     60     except TypeError:
     61         # A TypeError occurs if the object does have such a method in its

ValueError: axes don't match array

## === cell 11
model.load_weights("lstm_model.h5")

pred_public = model.predict(public_inputs)
pred_private = model.predict(private_inputs)




## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_54/113461856.py in <cell line: 0>()
      1 # Load best weights and predict
----> 2 model.load_weights("lstm_model.h5")
      3 
      4 pred_public = model.predict(public_inputs)
      5 pred_private = model.predict(private_inputs)

NameError: name 'model' is not defined

## === cell 12
def format_predictions(public_df, private_df, pred_pub, pred_priv):
    """
    Convert raw predictions to the submission long‑format.
    """
    preds = []
    for df, pred in [(public_df, pred_pub), (private_df, pred_priv)]:
        for i, uid in enumerate(df.id):
            single_pred = pred[i]  # (seq_len, 5)
            single_df = pd.DataFrame(single_pred, columns=target_cols)
            single_df["id_seqpos"] = [
                f"{uid}_{pos}" for pos in range(single_df.shape[0])
            ]
            preds.append(single_df)
    return pd.concat(preds, ignore_index=True)


lstm_preds = format_predictions(public_df, private_df, pred_public, pred_private)



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_54/2715828944.py in <cell line: 0>()
     15 
     16 
---> 17 lstm_preds = format_predictions(public_df, private_df, pred_public, pred_private)
     18 

NameError: name 'pred_public' is not defined

## === cell 13
submission = submission_format[["id_seqpos"]].merge(
    lstm_preds, how="left", on="id_seqpos"
)
submission[target_cols] = submission[target_cols].fillna(0)



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_54/3718058721.py in <cell line: 0>()
      1 # Merge with the sample submission to guarantee correct ordering / column set
      2 submission = submission_format[["id_seqpos"]].merge(
----> 3     lstm_preds, how="left", on="id_seqpos"
      4 )
      5 # Fill any missing values (should not happen) with zeros

NameError: name 'lstm_preds' is not defined

## === cell 14
os.chdir("/kaggle/working")
submission.to_csv("submission.csv", index=False)
print("Submission saved to /kaggle/working/submission.csv", submission.shape)

## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_54/60449231.py in <cell line: 0>()
      1 # Write the final submission file
      2 os.chdir("/kaggle/working")
----> 3 submission.to_csv("submission.csv", index=False)
      4 print("Submission saved to /kaggle/working/submission.csv", submission.shape)

NameError: name 'submission' is not defined
