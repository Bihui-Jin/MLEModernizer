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
plotly==5.24.1
plotly-express==0.4.1
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

0.39657

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

import numpy as np, pandas as pd, json, tensorflow as tf
from matplotlib import pyplot as plt
from tqdm.notebook import tqdm



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train_path = "/kaggle/input/stanford-covid-vaccine/train.json"
test_path = "/kaggle/input/stanford-covid-vaccine/test.json"
sample_sub_path = "/kaggle/input/stanford-covid-vaccine/sample_submission.csv"

train_data = pd.read_json(train_path, lines=True)
test_data = pd.read_json(test_path, lines=True)
submission_format = pd.read_csv(sample_sub_path, encoding="utf-8-sig")



## === cell 2
token2int = {x: i for i, x in enumerate("().ACGUBEHIMSX")}
target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]




## === cell 3
def get_bases(df):
    bases = []
    for seq in df["sequence"]:
        cnt = pd.Series(list(seq)).value_counts()
        bases.append(
            [
                cnt.get("A", 0) / 107,
                cnt.get("G", 0) / 107,
                cnt.get("C", 0) / 107,
                cnt.get("U", 0) / 107,
            ]
        )
    return pd.DataFrame(
        bases, columns=["A_percent", "G_percent", "C_percent", "U_percent"]
    )


def get_pairs_rate(df):
    rates = []
    for struct in df["structure"]:
        rates.append(struct.count("(") / 53.5)
    return pd.DataFrame(rates, columns=["pairs_rate"])


def get_pairs(df):
    pairs = []
    for i in range(len(df)):
        partners = [-1] * 130
        pairs_dict = {}
        queue = []
        seq = df.iloc[i]["sequence"]
        struct = df.iloc[i]["structure"]
        for j, ch in enumerate(struct):
            if ch == "(":
                queue.append(j)
            elif ch == ")":
                start = queue.pop()
                pair = (seq[start], seq[j])
                pairs_dict[pair] = pairs_dict.get(pair, 0) + 1
                partners[start] = j
                partners[j] = start
        total = sum(pairs_dict.values())
        uniq = [("U", "G"), ("C", "G"), ("U", "A"), ("G", "C"), ("A", "U"), ("G", "U")]
        pairs.append([pairs_dict.get(p, 0) / total if total > 0 else 0 for p in uniq])
    return pd.DataFrame(pairs, columns=["U-G", "C-G", "U-A", "G-C", "A-U", "G-U"])


def get_loops(df):
    loops = []
    avail = ["E", "S", "H", "B", "X", "I", "M"]
    for lt in df["predicted_loop_type"]:
        cnt = pd.Series(list(lt)).value_counts()
        loops.append([cnt.get(c, 0) / 107 for c in avail])
    return pd.DataFrame(loops, columns=avail)


train_feat = pd.concat(
    [
        train_data,
        get_bases(train_data),
        get_pairs(train_data),
        get_loops(train_data),
        get_pairs_rate(train_data),
    ],
    axis=1,
)

test_feat = pd.concat(
    [
        test_data,
        get_bases(test_data),
        get_pairs(test_data),
        get_loops(test_data),
        get_pairs_rate(test_data),
    ],
    axis=1,
)




## === cell 4
def preprocess_inputs(df, seq_length=107):
    cat_array = np.stack(
        df[["sequence", "structure", "predicted_loop_type"]]
        .applymap(lambda s: [token2int[ch] for ch in s])
        .values
    )
    cat_array = np.transpose(cat_array, (1, 2, 0))  # (samples, seq_len, 3)

    bpps_sum = np.zeros((len(df), seq_length, 1))
    bpps_max = np.zeros((len(df), seq_length, 1))
    bpps_nb = np.zeros((len(df), seq_length, 1))

    def get_structure_adj(df_local):
        Ss = []
        for _, row in df_local.iterrows():
            seq_len = row["seq_length"]
            struct = row["structure"]
            seq = row["sequence"]
            a_struct = np.zeros((seq_len, seq_len))
            stack = []
            for idx, ch in enumerate(struct):
                if ch == "(":
                    stack.append(idx)
                elif ch == ")":
                    start = stack.pop()
                    a_struct[start, idx] = 1
                    a_struct[idx, start] = 1
            Ss.append(a_struct.sum(axis=1, keepdims=True))
        return np.stack(Ss, axis=0)  # (samples, seq_len, 1)

    struct_adj = get_structure_adj(df)  # (samples, seq_len, 1)

    data = np.concatenate([cat_array, bpps_sum, bpps_max, bpps_nb, struct_adj], axis=2)

    scalar_cols = [
        "A_percent",
        "G_percent",
        "C_percent",
        "U_percent",
        "U-G",
        "C-G",
        "U-A",
        "G-C",
        "A-U",
        "G-U",
        "E",
        "S",
        "H",
        "B",
        "X",
        "I",
        "M",
        "pairs_rate",
    ]
    for col in scalar_cols:
        col_arr = np.repeat(df[col].values[:, None], seq_length, axis=1)[:, :, None]
        data = np.concatenate([data, col_arr], axis=2)

    return data




## === cell 5
train_high = train_feat[train_feat["signal_to_noise"] > 1]
train_inputs = preprocess_inputs(train_high, seq_length=107)
train_labels = np.stack(train_high[target_cols].values)  # (samples, 68, 5)
pad_len = 107 - train_labels.shape[1]
train_labels = np.concatenate(
    [train_labels, np.zeros((train_labels.shape[0], pad_len, train_labels.shape[2]))],
    axis=1,
)




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_56/602187676.py in <cell line: 0>()
      1 # Keep only high‑quality samples for training
      2 train_high = train_feat[train_feat["signal_to_noise"] > 1]
----> 3 train_inputs = preprocess_inputs(train_high, seq_length=107)
      4 train_labels = np.stack(train_high[target_cols].values)  # (samples, 68, 5)
      5 # Pad labels to full sequence length (107) with zeros for positions 68‑106

/tmp/ipykernel_56/391333185.py in preprocess_inputs(df, seq_length)
      8     )
      9     # shape -> (3, samples, seq_len)
---> 10     cat_array = np.transpose(cat_array, (1, 2, 0))  # (samples, seq_len, 3)
     11 
     12     # Numerical features (bpps placeholders – zeros)

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

## === cell 6
def build_model(
    seq_len=107,
    num_features=None,
    embed_dim=64,
    sp_dropout=0.2,
    hidden_dim=128,
    dropout=0.3,
    pred_len=68,
    gru_flag=False,
):
    if num_features is None:
        raise ValueError("num_features must be provided")
    inputs = tf.keras.layers.Input(shape=(seq_len, num_features))
    cat_feats = (
        inputs[:, :, :, 0:3] if False else inputs[:, :, :, 0:3]
    )  # placeholder, not used
    categorical = inputs[:, :, 0:3]  # (batch, seq_len, 3) – integer tokens
    numeric = inputs[:, :, 3:]  # remaining numeric features

    embed = tf.keras.layers.Embedding(input_dim=len(token2int), output_dim=embed_dim)(
        categorical
    )
    reshaped = tf.keras.layers.Reshape((seq_len, embed_dim * 3))(embed)

    x = tf.keras.layers.Concatenate(axis=-1)([reshaped, numeric])

    x = tf.keras.layers.SpatialDropout1D(sp_dropout)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    rnn_layer = tf.keras.layers.GRU if gru_flag else tf.keras.layers.LSTM
    for _ in range(2):
        x = rnn_layer(
            hidden_dim,
            dropout=dropout,
            return_sequences=True,
            kernel_initializer="orthogonal",
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)

    overall = inputs  # using the full input as “gene‑level” features
    x = tf.keras.layers.Concatenate(axis=-1)([x, overall])

    x = tf.keras.layers.Dense(50, activation="linear")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(sp_dropout)(x)

    x = x[:, :pred_len, :]

    outputs = tf.keras.layers.Dense(5, activation="linear")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer="adam",
        loss=lambda y, t: tf.reduce_mean(
            tf.sqrt(tf.reduce_mean(tf.square(y - t), axis=-1)), axis=-1
        ),
    )
    return model




## === cell 7
seq_len = train_inputs.shape[1]
num_feat = train_inputs.shape[2]
model = build_model(seq_len=seq_len, num_features=num_feat, pred_len=68, gru_flag=False)
model.summary()

EPOCHS = 30
BATCH_SIZE = 32
model.fit(train_inputs, train_labels, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2)



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/1627482278.py in <cell line: 0>()
      1 # Train the model
----> 2 seq_len = train_inputs.shape[1]
      3 num_feat = train_inputs.shape[2]
      4 model = build_model(seq_len=seq_len, num_features=num_feat, pred_len=68, gru_flag=False)
      5 model.summary()

NameError: name 'train_inputs' is not defined

## === cell 8
public_df = test_feat[test_feat["seq_length"] == 107].copy()
private_df = test_feat[test_feat["seq_length"] == 130].copy()

public_inputs = preprocess_inputs(public_df, seq_length=107)
private_inputs = preprocess_inputs(private_df, seq_length=130)

pred_public = model.predict(public_inputs)  # (samples, 68, 5)
pred_private = model.predict(private_inputs)  # (samples, 68, 5)




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_56/2621004434.py in <cell line: 0>()
      3 private_df = test_feat[test_feat["seq_length"] == 130].copy()
      4 
----> 5 public_inputs = preprocess_inputs(public_df, seq_length=107)
      6 private_inputs = preprocess_inputs(private_df, seq_length=130)
      7 

/tmp/ipykernel_56/391333185.py in preprocess_inputs(df, seq_length)
      8     )
      9     # shape -> (3, samples, seq_len)
---> 10     cat_array = np.transpose(cat_array, (1, 2, 0))  # (samples, seq_len, 3)
     11 
     12     # Numerical features (bpps placeholders – zeros)

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

## === cell 9
def format_predictions(df, preds):
    rows = []
    for i, uid in enumerate(df["id"]):
        pred = preds[i]  # (68,5)
        df_pred = pd.DataFrame(pred, columns=target_cols)
        df_pred["id_seqpos"] = [f"{uid}_{pos}" for pos in range(df_pred.shape[0])]
        rows.append(df_pred)
    return pd.concat(rows, ignore_index=True)


preds_public_df = format_predictions(public_df, pred_public)
preds_private_df = format_predictions(private_df, pred_private)

all_preds = pd.concat([preds_public_df, preds_private_df], ignore_index=True)

submission = submission_format[["id_seqpos"]].merge(
    all_preds, on="id_seqpos", how="left"
)
submission[target_cols] = submission[target_cols].fillna(0)



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/4016887017.py in <cell line: 0>()
     10 
     11 
---> 12 preds_public_df = format_predictions(public_df, pred_public)
     13 preds_private_df = format_predictions(private_df, pred_private)
     14 

NameError: name 'pred_public' is not defined

## === cell 10
os.chdir("/kaggle/working/")
submission.to_csv("submission.csv", index=False)
print("Submission file saved as /kaggle/working/submission.csv")

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/4099936586.py in <cell line: 0>()
      1 # Write final CSV
      2 os.chdir("/kaggle/working/")
----> 3 submission.to_csv("submission.csv", index=False)
      4 print("Submission file saved as /kaggle/working/submission.csv")

NameError: name 'submission' is not defined
