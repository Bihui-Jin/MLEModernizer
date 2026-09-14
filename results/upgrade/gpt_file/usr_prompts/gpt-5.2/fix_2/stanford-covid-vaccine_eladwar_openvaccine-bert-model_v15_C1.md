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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sentence-transformers==4.1.0
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
transformers==4.53.3

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

0.54715

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import json
import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow.keras.layers as L

import plotly.express as px

from sklearn.preprocessing import MinMaxScaler

from transformers import BertConfig, TFBertModel

np.random.seed(42)
tf.random.set_seed(42)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
pred_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]



## === cell 2
config = BertConfig()



## === cell 3
config.num_attention_heads




## === cell 4
def MCRMSE(y_true, y_pred):
    colwise_mse = tf.reduce_mean(tf.square(y_true - y_pred), axis=1)  # (batch, 5)
    return tf.reduce_mean(tf.sqrt(colwise_mse + 1e-9), axis=1)  # (batch,)


def gru_layer(hidden_dim, dropout):
    return L.Bidirectional(L.GRU(hidden_dim, dropout=dropout, return_sequences=True))


def build_model(seq_len=107, pred_len=68, dropout=0.5, embed_dim=100, hidden_dim=128):
    ids = L.Input(shape=(seq_len,), dtype=tf.int32, name="input_ids")

    config = BertConfig()
    config.vocab_size = 64  # small safe vocab for combined tokens
    config.num_hidden_layers = 3
    config.num_attention_heads = 1
    config.attention_probs_dropout_prob = 0.5
    config.hidden_size = 120
    config.hidden_act = tf.sinh  # preserve original intent

    bert_model = TFBertModel(config=config)
    bert_embeddings = bert_model(
        ids, training=True
    ).last_hidden_state  # (batch, seq_len, hidden)

    pooled = L.AveragePooling1D(pool_size=2)(bert_embeddings)  # (batch, 53, hidden)

    def pad_to_predlen(x):
        cur_len = tf.shape(x)[1]
        pad_len = tf.maximum(pred_len - cur_len, 0)
        last = x[:, -1:, :]
        pad = tf.tile(last, [1, pad_len, 1])
        return tf.concat([x, pad], axis=1)

    padded = L.Lambda(pad_to_predlen, name="pad_to_predlen")(
        pooled
    )  # (batch, >=pred_len, hidden)
    truncated = L.Lambda(lambda t: t[:, :pred_len, :], name="truncate")(padded)

    out = L.Dense(5, activation="linear")(truncated)
    model = tf.keras.Model(inputs=ids, outputs=out)

    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=MCRMSE)
    return model




## === cell 5
vocab = {
    "sequence": {x: i for i, x in enumerate("A C G U".split())},
    "structure": {x: i for i, x in enumerate("( . )".split())},
    "predicted_loop_type": {x: i for i, x in enumerate("B E H I M S X".split())},
}


def preprocess_inputs(df, cols=["sequence", "structure", "predicted_loop_type"]):
    seqs = df[cols[0]].values
    strs = df[cols[1]].values
    loops = df[cols[2]].values

    n = len(df)
    seq_len = len(seqs[0])

    out = np.zeros((n, seq_len), dtype=np.int32)

    off_seq = 0
    off_str = 8
    off_loop = 16

    for i in range(n):
        s = seqs[i]
        st = strs[i]
        lp = loops[i]
        for t in range(seq_len):
            out[i, t] = (
                off_seq
                + vocab["sequence"][s[t]]
                + 4 * (off_str + vocab["structure"][st[t]])
                + 16 * (off_loop + vocab["predicted_loop_type"][lp[t]])
            ) % 64  # keep within vocab_size
    return out




## === cell 6
train = pd.read_json("/kaggle/input/stanford-covid-vaccine/train.json", lines=True)
test = pd.read_json("/kaggle/input/stanford-covid-vaccine/test.json", lines=True)
sample_df = pd.read_csv("/kaggle/input/stanford-covid-vaccine/sample_submission.csv")



## === cell 7
print(pd.Series(list(train["structure"][0])).value_counts())
print(pd.Series(list(train["sequence"][0])).value_counts())
print(pd.Series(list(train["predicted_loop_type"][0])).value_counts())



## === cell 8
train_inputs = preprocess_inputs(train)
train_labels = np.array(train[pred_cols].values.tolist()).transpose(
    (0, 2, 1)
)  # (n, 68, 5)



## === cell 9
train_inputs.shape



## === cell 10
for df in [train, test]:
    df["Paired"] = [
        sum([(ch == "(") or (ch == ")") for ch in s]) for s in df["structure"]
    ]
    df["Unpaired"] = [sum([ch == "." for ch in s]) for s in df["structure"]]
    for col in ["E", "S", "H", "I", "G", "A", "U"]:
        if col in ["E", "S", "H", "I"]:
            df[col] = [
                sum([ch == col for ch in s]) / len(s) for s in df["predicted_loop_type"]
            ]
        else:
            df[col] = [sum([ch == col for ch in s]) / len(s) for s in df["sequence"]]


def safe_mean_pos(seq, ch):
    idx = [i for i, c in enumerate(seq) if c == ch]
    if len(idx) == 0:
        return 0.0
    return float(np.mean(idx))


for a in ["G", "A", "C", "U"]:
    train[a + "_position"] = [safe_mean_pos(s, a) for s in train["sequence"]]
    test[a + "_position"] = [safe_mean_pos(s, a) for s in test["sequence"]]

for a in ["E", "S", "H"]:
    train[a + "_position"] = [safe_mean_pos(s, a) for s in train["predicted_loop_type"]]
    test[a + "_position"] = [safe_mean_pos(s, a) for s in test["predicted_loop_type"]]



## === cell 11
target_columns = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
target_columns.extend(["SN_filter", "signal_to_noise"])
target_columns.extend(
    [
        "deg_error_pH10",
        "deg_error_Mg_50C",
        "deg_error_50C",
        "reactivity_error",
        "deg_error_Mg_pH10",
    ]
)
train.drop(target_columns, axis=1, inplace=True)



## === cell 12
SC = MinMaxScaler(feature_range=(-1, 1))
train_measurements = SC.fit_transform(
    pd.concat((train.select_dtypes("float64"), train.select_dtypes("int64")), axis=1)
)



## === cell 13
train_measurements.shape



## === cell 14
np.min(train_measurements), np.max(train_measurements)



## === cell 15
pd.DataFrame(train_measurements).describe().T



## === cell 16
model = build_model(seq_len=107, pred_len=68)
model.summary()



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/587590391.py in <cell line: 0>()
----> 1 model = build_model(seq_len=107, pred_len=68)
      2 model.summary()
      3 

/tmp/ipykernel_11/1784160057.py in build_model(seq_len, pred_len, dropout, embed_dim, hidden_dim)
     24 
     25     bert_model = TFBertModel(config=config)
---> 26     bert_embeddings = bert_model(
     27         ids, training=True
     28     ).last_hidden_state  # (batch, seq_len, hidden)

/usr/local/lib/python3.11/dist-packages/tf_keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
     68             # To get the full stack trace, call:
     69             # `tf.debugging.disable_traceback_filtering()`
---> 70             raise e.with_traceback(filtered_tb) from None
     71         finally:
     72             del filtered_tb

/usr/local/lib/python3.11/dist-packages/transformers/modeling_tf_utils.py in run_call_with_unpacked_inputs(self, *args, **kwargs)
    434             config = self.config
    435 
--> 436         unpacked_inputs = input_processing(func, config, **fn_args_and_kwargs)
    437         return func(self, **unpacked_inputs)
    438 

/usr/local/lib/python3.11/dist-packages/transformers/modeling_tf_utils.py in input_processing(func, config, **kwargs)
    564             output[main_input_name] = main_input
    565         else:
--> 566             raise ValueError(
    567                 f"Data of type {type(main_input)} is not allowed only {allowed_types} is accepted for"
    568                 f" {main_input_name}."

ValueError: Exception encountered when calling layer 'tf_bert_model' (type TFBertModel).

Data of type <class 'keras.src.backend.common.keras_tensor.KerasTensor'> is not allowed only (<class 'tensorflow.python.framework.tensor.Tensor'>, <class 'bool'>, <class 'int'>, <class 'transformers.utils.generic.ModelOutput'>, <class 'tuple'>, <class 'list'>, <class 'dict'>, <class 'numpy.ndarray'>) is accepted for input_ids.

Call arguments received by layer 'tf_bert_model' (type TFBertModel):
  • input_ids=<KerasTensor shape=(None, 107), dtype=int32, sparse=False, name=input_ids>
  • attention_mask=None
  • token_type_ids=None
  • position_ids=None
  • head_mask=None
  • inputs_embeds=None
  • encoder_hidden_states=None
  • encoder_attention_mask=None
  • past_key_values=None
  • use_cache=None
  • output_attentions=None
  • output_hidden_states=None
  • return_dict=None
  • training=True

## === cell 17
train_inputs.shape, train_labels.shape



## === cell 18
device = "/GPU:0" if tf.config.list_physical_devices("GPU") else "/CPU:0"
with tf.device(device):
    history = model.fit(
        train_inputs,
        train_labels,
        batch_size=64,
        epochs=100,
        validation_split=0.05,
        callbacks=[
            tf.keras.callbacks.ReduceLROnPlateau(),
            tf.keras.callbacks.ModelCheckpoint(
                "model.weights.h5", save_weights_only=True, save_best_only=True
            ),
        ],
        verbose=2,
    )



## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/925951503.py in <cell line: 0>()
      2 device = "/GPU:0" if tf.config.list_physical_devices("GPU") else "/CPU:0"
      3 with tf.device(device):
----> 4     history = model.fit(
      5         train_inputs,
      6         train_labels,

NameError: name 'model' is not defined

## === cell 19
fig = px.line(
    history.history,
    y=["loss", "val_loss"],
    labels={"index": "epoch", "value": "MCRMSE"},
    title="Training History",
)
fig.show()



## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1635172173.py in <cell line: 0>()
      1 fig = px.line(
----> 2     history.history,
      3     y=["loss", "val_loss"],
      4     labels={"index": "epoch", "value": "MCRMSE"},
      5     title="Training History",

NameError: name 'history' is not defined

## === cell 20
test_df = test.copy()
test_inputs = preprocess_inputs(test_df)



## === cell 21
infer_model = build_model(seq_len=107, pred_len=68)
infer_model.load_weights("model.weights.h5")

test_preds_68 = infer_model.predict(
    test_inputs, batch_size=64, verbose=0
)  # (n_test, 68, 5)

n_test = test_preds_68.shape[0]
preds_107 = np.zeros((n_test, 107, 5), dtype=np.float32)
preds_107[:, :68, :] = test_preds_68
preds_107[:, 68:, :] = test_preds_68[:, -1:, :]



## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/4015238749.py in <cell line: 0>()
      1 # Build same-shape model for inference and load best weights
----> 2 infer_model = build_model(seq_len=107, pred_len=68)
      3 infer_model.load_weights("model.weights.h5")
      4 
      5 test_preds_68 = infer_model.predict(

/tmp/ipykernel_11/1784160057.py in build_model(seq_len, pred_len, dropout, embed_dim, hidden_dim)
     24 
     25     bert_model = TFBertModel(config=config)
---> 26     bert_embeddings = bert_model(
     27         ids, training=True
     28     ).last_hidden_state  # (batch, seq_len, hidden)

/usr/local/lib/python3.11/dist-packages/tf_keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
     68             # To get the full stack trace, call:
     69             # `tf.debugging.disable_traceback_filtering()`
---> 70             raise e.with_traceback(filtered_tb) from None
     71         finally:
     72             del filtered_tb

/usr/local/lib/python3.11/dist-packages/transformers/modeling_tf_utils.py in run_call_with_unpacked_inputs(self, *args, **kwargs)
    434             config = self.config
    435 
--> 436         unpacked_inputs = input_processing(func, config, **fn_args_and_kwargs)
    437         return func(self, **unpacked_inputs)
    438 

/usr/local/lib/python3.11/dist-packages/transformers/modeling_tf_utils.py in input_processing(func, config, **kwargs)
    564             output[main_input_name] = main_input
    565         else:
--> 566             raise ValueError(
    567                 f"Data of type {type(main_input)} is not allowed only {allowed_types} is accepted for"
    568                 f" {main_input_name}."

ValueError: Exception encountered when calling layer 'tf_bert_model_1' (type TFBertModel).

Data of type <class 'keras.src.backend.common.keras_tensor.KerasTensor'> is not allowed only (<class 'tensorflow.python.framework.tensor.Tensor'>, <class 'bool'>, <class 'int'>, <class 'transformers.utils.generic.ModelOutput'>, <class 'tuple'>, <class 'list'>, <class 'dict'>, <class 'numpy.ndarray'>) is accepted for input_ids.

Call arguments received by layer 'tf_bert_model_1' (type TFBertModel):
  • input_ids=<KerasTensor shape=(None, 107), dtype=int32, sparse=False, name=input_ids>
  • attention_mask=None
  • token_type_ids=None
  • position_ids=None
  • head_mask=None
  • inputs_embeds=None
  • encoder_hidden_states=None
  • encoder_attention_mask=None
  • past_key_values=None
  • use_cache=None
  • output_attentions=None
  • output_hidden_states=None
  • return_dict=None
  • training=True

## === cell 22
print(preds_107.shape)



## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/274572136.py in <cell line: 0>()
----> 1 print(preds_107.shape)
      2 

NameError: name 'preds_107' is not defined

## === cell 23
preds_ls = []
for i, uid in enumerate(test_df.id.values):
    single_pred = preds_107[i]  # (107, 5)
    single_df = pd.DataFrame(single_pred, columns=pred_cols)
    single_df["id_seqpos"] = [f"{uid}_{x}" for x in range(single_df.shape[0])]
    preds_ls.append(single_df)

preds_df = pd.concat(preds_ls, axis=0, ignore_index=True)



## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3773261372.py in <cell line: 0>()
      1 preds_ls = []
      2 for i, uid in enumerate(test_df.id.values):
----> 3     single_pred = preds_107[i]  # (107, 5)
      4     single_df = pd.DataFrame(single_pred, columns=pred_cols)
      5     single_df["id_seqpos"] = [f"{uid}_{x}" for x in range(single_df.shape[0])]

NameError: name 'preds_107' is not defined

## === cell 24
submission = sample_df[["id_seqpos"]].merge(preds_df, on="id_seqpos", how="left")

for c in pred_cols:
    submission[c] = submission[c].fillna(0.0)

submission.to_csv("submission.csv", index=False)
print(submission.shape)
print(submission.head())
print("Wrote submission.csv")

## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/580885612.py in <cell line: 0>()
      1 # Fix: ensure exact row order/coverage matches sample_submission by merging on id_seqpos and keeping sample order.
----> 2 submission = sample_df[["id_seqpos"]].merge(preds_df, on="id_seqpos", how="left")
      3 
      4 # Safety: fill any missing (shouldn't happen) with 0.0 to keep valid file
      5 for c in pred_cols:

NameError: name 'preds_df' is not defined
