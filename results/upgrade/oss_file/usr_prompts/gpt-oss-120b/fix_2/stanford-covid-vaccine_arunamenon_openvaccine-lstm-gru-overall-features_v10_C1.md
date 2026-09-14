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

0.37982

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, json
import numpy as np, pandas as pd
import tensorflow as tf



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
os.chdir("/kaggle/working/")
print("cwd:", os.getcwd())



## === cell 2
train_path = "/kaggle/input/stanford-covid-vaccine/train.json"
test_path = "/kaggle/input/stanford-covid-vaccine/test.json"
sample_sub_path = "/kaggle/input/stanford-covid-vaccine/sample_submission.csv"

train_data = pd.read_json(train_path, lines=True)
test_data = pd.read_json(test_path, lines=True)
submission_format = pd.read_csv(sample_sub_path, encoding="utf-8-sig")
print(train_data.shape, test_data.shape, submission_format.shape)



## === cell 3
token2int = {c: i for i, c in enumerate("().ACGUBEHIMSX")}
target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]




## === cell 4
def preprocess_inputs(df):
    """
    Returns an array of shape (samples, 107, 4):
    - 3 categorical integer columns (sequence, structure, loop type)
    - 1 numeric column: summed structure adjacency (added later)
    """
    base_fea = np.transpose(
        np.array(
            df[["sequence", "structure", "predicted_loop_type"]]
            .applymap(lambda seq: [token2int[x] for x in seq])
            .values.tolist()
        ),
        (0, 2, 1),
    )  # (samples, 107, 3)

    adj_placeholder = np.zeros((len(df), 107, 1), dtype=np.float32)
    return np.concatenate([base_fea, adj_placeholder], axis=2)




## === cell 5
def get_structure_adj(df):
    """Compute a summed adjacency matrix per sample and return shape (samples,107,1)."""
    Ss = []
    for i in range(len(df)):
        seq_len = df["seq_length"].iloc[i]
        structure = df["structure"].iloc[i]
        sequence = df["sequence"].iloc[i]

        a_structures = {
            ("A", "U"): np.zeros([seq_len, seq_len]),
            ("C", "G"): np.zeros([seq_len, seq_len]),
            ("U", "G"): np.zeros([seq_len, seq_len]),
            ("U", "A"): np.zeros([seq_len, seq_len]),
            ("G", "C"): np.zeros([seq_len, seq_len]),
            ("G", "U"): np.zeros([seq_len, seq_len]),
        }
        cue = []
        for j, ch in enumerate(structure):
            if ch == "(":
                cue.append(j)
            elif ch == ")":
                start = cue.pop()
                pair = (sequence[start], sequence[j])
                if pair in a_structures:
                    a_structures[pair][start, j] = 1
                    a_structures[(pair[1], pair[0])][j, start] = 1

        stacked = np.stack(list(a_structures.values()), axis=2)
        summed = np.sum(stacked, axis=2, keepdims=True)  # (seq_len, seq_len, 1)
        Ss.append(summed.sum(axis=1))  # (seq_len, 1)

    return np.array(Ss, dtype=np.float32)  # (samples, seq_len, 1)




## === cell 6
train_mask = train_data["signal_to_noise"] > 1
train_df = train_data[train_mask].reset_index(drop=True)

train_inputs = preprocess_inputs(train_df)
train_adj = get_structure_adj(train_df)
train_inputs = np.concatenate(
    [train_inputs[..., :3], train_adj], axis=2
)  # replace placeholder


def pad_labels(labels, seq_scored):
    padded = np.zeros((len(labels), 107, 5), dtype=np.float32)
    for i, (lab, sc) in enumerate(zip(labels, seq_scored)):
        padded[i, :sc, :] = lab
    return padded


raw_labels = np.array(train_df[target_cols].values.tolist())  # (samples,68,5)
train_labels = pad_labels(raw_labels, train_df["seq_scored"].values)




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/2412769922.py in <cell line: 0>()
     19 
     20 raw_labels = np.array(train_df[target_cols].values.tolist())  # (samples,68,5)
---> 21 train_labels = pad_labels(raw_labels, train_df["seq_scored"].values)
     22 
     23 

/tmp/ipykernel_55/2412769922.py in pad_labels(labels, seq_scored)
     14     padded = np.zeros((len(labels), 107, 5), dtype=np.float32)
     15     for i, (lab, sc) in enumerate(zip(labels, seq_scored)):
---> 16         padded[i, :sc, :] = lab
     17     return padded
     18 

ValueError: could not broadcast input array from shape (5,68) into shape (68,5)

## === cell 7
def build_model(
    seq_len=107, embed_dim=64, hidden_dim=128, dropout=0.2, pred_len=68, gru_flag=False
):
    inputs = tf.keras.layers.Input(shape=(seq_len, 4))  # 3 cat + 1 num
    categorical = tf.cast(inputs[:, :, :3], tf.int32)  # (B, L, 3)
    numerical = inputs[:, :, 3:]  # (B, L, 1)

    embed = tf.keras.layers.Embedding(input_dim=len(token2int), output_dim=embed_dim)(
        categorical
    )
    reshaped = tf.keras.layers.Reshape((seq_len, 3 * embed_dim))(embed)

    merged = tf.keras.layers.concatenate([reshaped, numerical], axis=2)

    x = tf.keras.layers.BatchNormalization()(merged)

    if gru_flag:
        rnn = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(
                hidden_dim,
                dropout=dropout,
                return_sequences=True,
                kernel_initializer="orthogonal",
            )
        )(x)
    else:
        rnn = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                hidden_dim,
                dropout=dropout,
                return_sequences=True,
                kernel_initializer="orthogonal",
            )
        )(x)

    x = tf.keras.layers.BatchNormalization()(rnn)

    x = x[:, :pred_len, :]  # (B, pred_len, hidden*2)

    out = tf.keras.layers.Dense(5, activation="linear")(x)  # 5 target columns

    model = tf.keras.Model(inputs, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=lambda y_true, y_pred: tf.reduce_mean(
            tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred), axis=2)), axis=1
        ),
    )
    return model




## === cell 8
EPOCHS = 30
BATCH_SIZE = 32

model = build_model(gru_flag=True)  # GRU was slightly better in original code
model.summary()
model.fit(train_inputs, train_labels, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/2884350146.py in <cell line: 0>()
      2 BATCH_SIZE = 32
      3 
----> 4 model = build_model(gru_flag=True)  # GRU was slightly better in original code
      5 model.summary()
      6 model.fit(train_inputs, train_labels, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2)

/tmp/ipykernel_55/1325003590.py in build_model(seq_len, embed_dim, hidden_dim, dropout, pred_len, gru_flag)
      3 ):
      4     inputs = tf.keras.layers.Input(shape=(seq_len, 4))  # 3 cat + 1 num
----> 5     categorical = tf.cast(inputs[:, :, :3], tf.int32)  # (B, L, 3)
      6     numerical = inputs[:, :, 3:]  # (B, L, 1)
      7 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/traceback_utils.py in error_handler(*args, **kwargs)
    151     except Exception as e:
    152       filtered_tb = _process_traceback_frames(e.__traceback__)
--> 153       raise e.with_traceback(filtered_tb) from None
    154     finally:
    155       del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/backend/common/keras_tensor.py in __tf_tensor__(self, dtype, name)
    136 
    137     def __tf_tensor__(self, dtype=None, name=None):
--> 138         raise ValueError(
    139             "A KerasTensor cannot be used as input to a TensorFlow function. "
    140             "A KerasTensor is a symbolic placeholder for a shape and dtype, "

ValueError: A KerasTensor cannot be used as input to a TensorFlow function. A KerasTensor is a symbolic placeholder for a shape and dtype, used when constructing Keras Functional models or Keras Functions. You can only use it as input to a Keras layer or a Keras operation (from the namespaces `keras.layers` and `keras.operations`). You are likely doing something like:

```
x = Input(...)
...
tf_fn(x)  # Invalid.
```

What you should do instead is wrap `tf_fn` in a layer:

```
class MyLayer(Layer):
    def call(self, x):
        return tf_fn(x)

x = MyLayer()(x)
```


## === cell 9
public_df = test_data[test_data["seq_length"] == 107].reset_index(drop=True)
private_df = test_data[test_data["seq_length"] == 130].reset_index(drop=True)

public_inputs = preprocess_inputs(public_df)
public_adj = get_structure_adj(public_df)
public_inputs = np.concatenate([public_inputs[..., :3], public_adj], axis=2)

private_inputs = preprocess_inputs(private_df)
private_adj = get_structure_adj(private_df)
private_inputs = np.concatenate([private_inputs[..., :3], private_adj], axis=2)



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/3397530612.py in <cell line: 0>()
      7 public_inputs = np.concatenate([public_inputs[..., :3], public_adj], axis=2)
      8 
----> 9 private_inputs = preprocess_inputs(private_df)
     10 private_adj = get_structure_adj(private_df)
     11 private_inputs = np.concatenate([private_inputs[..., :3], private_adj], axis=2)

/tmp/ipykernel_55/1504576037.py in preprocess_inputs(df)
      6     """
      7     # categorical part
----> 8     base_fea = np.transpose(
      9         np.array(
     10             df[["sequence", "structure", "predicted_loop_type"]]

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

## === cell 10
pred_public = model.predict(public_inputs)  # shape (n_pub, 68, 5)
pred_private = model.predict(private_inputs)  # shape (n_priv, 68, 5)




## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/237612042.py in <cell line: 0>()
      1 # predict for both splits (pred_len equals sequence length)
----> 2 pred_public = model.predict(public_inputs)  # shape (n_pub, 68, 5)
      3 pred_private = model.predict(private_inputs)  # shape (n_priv, 68, 5)
      4 
      5 

NameError: name 'model' is not defined

## === cell 11
def format_predictions(df, preds):
    rows = []
    for uid, arr in zip(df["id"], preds):
        seq_len = df.loc[df["id"] == uid, "seq_length"].values[0]
        padded = np.zeros((seq_len, 5), dtype=np.float32)
        padded[: arr.shape[0], :] = arr
        df_pred = pd.DataFrame(padded, columns=target_cols)
        df_pred["id_seqpos"] = [f"{uid}_{i}" for i in range(seq_len)]
        rows.append(df_pred)
    return pd.concat(rows, ignore_index=True)


public_preds_df = format_predictions(public_df, pred_public)
private_preds_df = format_predictions(private_df, pred_private)



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1100520361.py in <cell line: 0>()
     12 
     13 
---> 14 public_preds_df = format_predictions(public_df, pred_public)
     15 private_preds_df = format_predictions(private_df, pred_private)
     16 

NameError: name 'pred_public' is not defined

## === cell 12
all_preds = pd.concat([public_preds_df, private_preds_df], ignore_index=True)
submission = submission_format[["id_seqpos"]].merge(
    all_preds, on="id_seqpos", how="left"
)
submission.to_csv("submission.csv", index=False)
print("Submission file written:", os.path.abspath("submission.csv"))

## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4114726171.py in <cell line: 0>()
      1 # combine and align with the sample submission
----> 2 all_preds = pd.concat([public_preds_df, private_preds_df], ignore_index=True)
      3 submission = submission_format[["id_seqpos"]].merge(
      4     all_preds, on="id_seqpos", how="left"
      5 )

NameError: name 'public_preds_df' is not defined
