# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

3.9

# 3. Installed packages

geopandas==0.14.4
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
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

0.36425

# 6. Current score

0.22882

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.69303) has done: 'I fix the environment/import issues by removing the incompatible `spektral` dependency and replacing `GraphConv` with a small in-notebook graph-convolution layer that keeps the same call signature `([node_features, adjacency])`. I also fix sequence tokenization bugs (wrong dictionary used, incorrect handling of empty private set) and ensure `tf`, `L`, and `K` are consistently imported under TensorFlow/Keras 3. Next, I make BPPS optional: if the `bpps/` directory isn’t present, the code fall back to zero matrices (so the pipeline always runs end-to-end without missing files). Finally, I ensure predictions are created for all test rows (107 length here), aligned to `sample_submission.csv` ordering, and written to `submission.csv` with the exact required columns.'
- What this solution (achieved 0.22882) has done: 'I fix the environment crash caused by an incompatible protobuf/TF import combination by pinning protobuf to the Kaggle-compatible version at runtime before importing TensorFlow. Then I fix the training-time error by replacing the removed `tf.keras.losses.mean_squared_error` functional call with an equivalent backend-safe computation that preserves the exact loss semantics (per-position RMSE then averaged across targets). Finally, I remove the unintended EarlyStopping callback (it violates your stated “no early stopping/relaxed convergence” requirement and can hurt score) while keeping the rest of the training loop unchanged, and I fix a small logic bug in `preds_to_rows` (early `return`) without changing the submission formatting behavior.'

# 9. Code solution

## === cell 0
import os
import sys
import json
import subprocess
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import train_test_split

try:
    import google.protobuf  # noqa: F401
    from google.protobuf import __version__ as _pb_ver  # noqa: F401
except Exception:
    _pb_ver = "0"


def _major(v):
    try:
        return int(str(v).split(".")[0])
    except Exception:
        return 0


if _major(_pb_ver) >= 5:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "--no-deps", "protobuf==3.20.3"]
    )
    import importlib
    import google.protobuf

    importlib.reload(google.protobuf)

import tensorflow as tf
from tensorflow.keras import layers as L
from tensorflow.keras import backend as K

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("tf:", tf.__version__)
print("numpy:", np.__version__)
print("pandas:", pd.__version__)




## === cell 1
class GraphConv(L.Layer):
    def __init__(self, channels, activation=None, use_bias=True, **kwargs):
        super().__init__(**kwargs)
        self.channels = int(channels)
        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias
        self.dense = L.Dense(self.channels, use_bias=self.use_bias)

    def call(self, inputs):
        x, a = inputs  # x: (B,N,F), a: (B,N,N)
        n = tf.shape(a)[-1]
        eye = tf.eye(n, batch_shape=[tf.shape(a)[0]], dtype=a.dtype)
        a_hat = a + eye

        d = tf.reduce_sum(a_hat, axis=-1)  # (B,N)
        d_inv_sqrt = tf.math.rsqrt(tf.maximum(d, tf.cast(1e-6, d.dtype)))
        d_inv_sqrt = tf.linalg.diag(d_inv_sqrt)  # (B,N,N)
        a_norm = tf.matmul(tf.matmul(d_inv_sqrt, a_hat), d_inv_sqrt)  # (B,N,N)

        xw = self.dense(x)  # (B,N,C)
        out = tf.matmul(a_norm, xw)  # (B,N,C)
        if self.activation is not None:
            out = self.activation(out)
        return out




## === cell 2
train_json_path = "/kaggle/input/stanford-covid-vaccine/train.json"
test_json_path = "/kaggle/input/stanford-covid-vaccine/test.json"
sample_sub_path = "/kaggle/input/stanford-covid-vaccine/sample_submission.csv"
bpps_path = "/kaggle/input/stanford-covid-vaccine/bpps"  # may not exist in this dataset snapshot
output_path = "./"

train_df = pd.read_json(train_json_path, lines=True)
test_df = pd.read_json(test_json_path, lines=True)

public_df = test_df.query("seq_length == 107").copy()
private_df = test_df.query("seq_length == 130").copy()

print("train_df:", train_df.shape, "test_df:", test_df.shape)
print("public_df:", public_df.shape, "private_df:", private_df.shape)



## === cell 3
pred_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
train_y = np.array(train_df[pred_cols].values.tolist()).transpose(
    (0, 2, 1)
)  # (n, 68, 5)
print("train_y:", train_y.shape)



## === cell 4
sequence_token2int = {x: i for i, x in enumerate("AGUC")}
structure_token2int = {".": 0, "(": 1, ")": 2}
loop_token2int = {x: i for i, x in enumerate("SMIBHEX")}
token2int_map = {
    "sequence": sequence_token2int,
    "structure": structure_token2int,
    "predicted_loop_type": loop_token2int,
}
sequence_columns = ["sequence", "structure", "predicted_loop_type"]


def to_one_hot(df):
    if df.shape[0] == 0:
        return np.zeros((0, 107, 14), dtype=np.float32)

    temp = np.transpose(
        np.array(
            [
                df[col]
                .apply(lambda seq: [token2int_map[col][x] for x in seq])
                .values.tolist()
                for col in sequence_columns
            ]
        ),
        (1, 2, 0),
    )
    ohe_1 = tf.keras.utils.to_categorical(temp[:, :, 0], 4)
    ohe_2 = tf.keras.utils.to_categorical(temp[:, :, 1], 3)
    ohe_3 = tf.keras.utils.to_categorical(temp[:, :, 2], 7)
    out = np.concatenate([ohe_1, ohe_2, ohe_3], axis=2).astype(np.float32)
    return out


train_ohe = to_one_hot(train_df)
public_ohe = to_one_hot(public_df)
private_ohe = to_one_hot(private_df)

print(
    "train_ohe:",
    train_ohe.shape,
    "public_ohe:",
    public_ohe.shape,
    "private_ohe:",
    private_ohe.shape,
)




## === cell 5
def to_int_tokens(df):
    if df.shape[0] == 0:
        return np.zeros((0, 107, 3), dtype=np.int32)

    temp = np.stack(
        [
            df[col]
            .apply(lambda seq: [token2int_map[col][x] for x in seq])
            .values.tolist()
            for col in sequence_columns
        ],
        axis=-1,
    )  # (n, seq_len, 3)
    return temp.astype(np.int32)


train_tok = to_int_tokens(train_df)
public_tok = to_int_tokens(public_df)
private_tok = to_int_tokens(private_df)

print(
    "train_tok:",
    train_tok.shape,
    "public_tok:",
    public_tok.shape,
    "private_tok:",
    private_tok.shape,
)




## === cell 6
def get_adjacency_matrix(inps):
    if inps.shape[0] == 0:
        return np.zeros((0, inps.shape[1], inps.shape[1]), dtype=np.float32)

    As = np.zeros((inps.shape[0], inps.shape[1], inps.shape[1]), dtype=np.float32)
    for row in range(inps.shape[0]):
        stack = []
        for seqpos in range(inps.shape[1]):
            if inps[row, seqpos, 1] == 1:  # '('
                stack.append(seqpos)
            elif inps[row, seqpos, 1] == 2:  # ')'
                if stack:
                    openpos = stack.pop()
                    As[row, openpos, seqpos] = 1.0
                    As[row, seqpos, openpos] = 1.0
    return As


train_adj = get_adjacency_matrix(train_tok)
public_adj = get_adjacency_matrix(public_tok)
private_adj = get_adjacency_matrix(private_tok)

print(
    "train_adj:",
    train_adj.shape,
    "public_adj:",
    public_adj.shape,
    "private_adj:",
    private_adj.shape,
)




## === cell 7
def get_bpps(mRNA_ids, seq_len=107):
    if (bpps_path is None) or (not os.path.isdir(bpps_path)):
        return np.zeros((len(mRNA_ids), seq_len, seq_len), dtype=np.float32)

    bpps = []
    for mRNA_id in tqdm(mRNA_ids, desc="Loading BPPS"):
        fp = f"{bpps_path}/{mRNA_id}.npy"
        if os.path.exists(fp):
            bpps.append(np.load(fp).astype(np.float32))
        else:
            bpps.append(np.zeros((seq_len, seq_len), dtype=np.float32))
    return np.array(bpps, dtype=np.float32)


train_bpps = get_bpps(train_df.id.values, seq_len=int(train_df.seq_length.iloc[0]))
public_bpps = (
    get_bpps(public_df.id.values, seq_len=107)
    if len(public_df)
    else np.zeros((0, 107, 107), np.float32)
)
private_bpps = (
    get_bpps(private_df.id.values, seq_len=130)
    if len(private_df)
    else np.zeros((0, 130, 130), np.float32)
)

print(
    "train_bpps:",
    train_bpps.shape,
    "public_bpps:",
    public_bpps.shape,
    "private_bpps:",
    private_bpps.shape,
)




## === cell 8
def bpps_stats(bpps):
    if bpps.shape[0] == 0:
        return np.zeros((0, bpps.shape[1], 2), dtype=np.float32)
    mean_ = bpps.mean(axis=2)
    max_ = bpps.max(axis=2)
    return np.concatenate([mean_[:, :, None], max_[:, :, None]], axis=2).astype(
        np.float32
    )


train_bpps_stats = bpps_stats(train_bpps)
public_bpps_stats = bpps_stats(public_bpps)
private_bpps_stats = bpps_stats(private_bpps)

print(
    "train_bpps_stats:",
    train_bpps_stats.shape,
    "public_bpps_stats:",
    public_bpps_stats.shape,
    "private_bpps_stats:",
    private_bpps_stats.shape,
)



## === cell 9
scored_seq_length = 68


def rmse(y_actual, y_pred):
    y_actual = tf.cast(y_actual, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    mse = tf.reduce_mean(tf.square(y_actual - y_pred), axis=-1)
    return tf.sqrt(mse)


def mcrmse(y_actual, y_pred):
    y_actual = tf.cast(y_actual, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    score = 0.0
    n_t = tf.shape(y_actual)[-1]
    t_dim = y_actual.shape[-1]
    if t_dim is not None:
        for i in range(int(t_dim)):
            score += rmse(y_actual[:, :, i], y_pred[:, :, i]) / tf.cast(n_t, tf.float32)
    else:
        for i in tf.range(n_t):
            score += rmse(y_actual[:, :, i], y_pred[:, :, i]) / tf.cast(n_t, tf.float32)
    return score


def build_model(input_seq_len=107, output_seq_len=scored_seq_length):
    def _bi_gru_block(x, hidden_dim, dropout):
        gru = L.Bidirectional(
            L.GRU(hidden_dim, dropout=dropout, return_sequences=True),
        )(x)
        return gru

    def _conv_block(x, adj_m, bpp_m, conv_filters, graph_channels):
        conv = L.Conv1D(conv_filters, 5, padding="same", activation="tanh")(x)

        gcn_1 = GraphConv(graph_channels, activation="tanh")([conv, adj_m])
        gcn_2 = GraphConv(graph_channels, activation="tanh")([conv, bpp_m])

        conv = L.Concatenate()([conv, gcn_1, gcn_2])
        conv = L.Activation("relu")(conv)
        conv = L.SpatialDropout1D(0.1)(conv)
        return conv

    one_hot_encoding_inputs = L.Input(shape=(input_seq_len, 14), name="onehot")
    adj_matrix_inputs = L.Input((input_seq_len, input_seq_len), name="adjmatrix")
    base_pair_proba_inputs = L.Input((input_seq_len, input_seq_len), name="pairproba")
    base_pair_proba_stats_inputs = L.Input(
        shape=(input_seq_len, 2), name="pairprobastats"
    )

    merged_inputs = L.Concatenate()(
        [one_hot_encoding_inputs, base_pair_proba_stats_inputs]
    )

    hidden = _conv_block(
        merged_inputs, adj_matrix_inputs, base_pair_proba_inputs, 512, 80
    )
    hidden = _bi_gru_block(hidden, 256, 0.5)
    hidden = _conv_block(hidden, adj_matrix_inputs, base_pair_proba_inputs, 512, 80)
    hidden = _bi_gru_block(hidden, 256, 0.5)

    out = hidden[:, :output_seq_len]
    out = L.Dense(5, activation="linear")(out)

    model = tf.keras.Model(
        inputs=[
            one_hot_encoding_inputs,
            adj_matrix_inputs,
            base_pair_proba_inputs,
            base_pair_proba_stats_inputs,
        ],
        outputs=out,
    )
    return model


model = build_model()
model.summary()



## === cell 10
split_results = train_test_split(
    train_ohe,
    train_adj,
    train_bpps,
    train_bpps_stats,
    train_y,
    train_df.signal_to_noise.values.astype(np.float32),
    train_df.SN_filter.values.astype(np.int32),
    test_size=0.1,
    random_state=SEED,
)

(
    trn_ohe,
    val_ohe,
    trn_adj,
    val_adj,
    trn_bpps,
    val_bpps,
    trn_bpps_stats,
    val_bpps_stats,
    trn_y,
    val_y,
    trn_snr,
    val_snr,
    trn_snf,
    val_snf,
) = split_results

trn_inputs = [trn_ohe, trn_adj, trn_bpps, trn_bpps_stats]
val_inputs = [val_ohe, val_adj, val_bpps, val_bpps_stats]

val_mask = np.where((val_snf == 1))[0]
val_inputs = [v[val_mask] for v in val_inputs]
val_y = val_y[val_mask]

sample_weight = np.log(trn_snr + 1.11) / 2.0

print("Train size:", trn_ohe.shape[0], "Val size (SN_filter=1):", val_y.shape[0])



## === cell 11
model = build_model()
model.compile(tf.keras.optimizers.Adam(), loss=mcrmse)

history = model.fit(
    trn_inputs,
    trn_y,
    validation_data=(val_inputs, val_y),
    batch_size=64,
    epochs=300,
    sample_weight=sample_weight,
    callbacks=[
        tf.keras.callbacks.ReduceLROnPlateau(verbose=1, monitor="val_loss"),
        tf.keras.callbacks.ModelCheckpoint(
            "model.weights.h5",
            save_best_only=True,
            verbose=0,
            monitor="val_loss",
            save_weights_only=True,
        ),
    ],
    verbose=2,
)
print(f"Min validation loss history={min(history.history['val_loss'])}")



## === cell 12
if os.path.exists("model.weights.h5"):
    model.load_weights("model.weights.h5")

val_preds = model.predict(val_inputs, verbose=0)
val_score = tf.reduce_mean(mcrmse(val_y, val_preds)).numpy()
print("Validation MCRMSE:", float(val_score))



## === cell 13
model_public = build_model(107, 107)
if os.path.exists("model.weights.h5"):
    model_public.load_weights("model.weights.h5")
else:
    model_public.set_weights(model.get_weights())

public_inputs = [public_ohe, public_adj, public_bpps, public_bpps_stats]
public_preds = model_public.predict(public_inputs, verbose=0)  # (n_test, 107, 5)
print("public_preds:", public_preds.shape)

private_preds = None
if len(private_df) > 0:
    model_private = build_model(130, 130)
    if os.path.exists("model.weights.h5"):
        model_private.load_weights("model.weights.h5")
    else:
        model_private.set_weights(model.get_weights())
    private_inputs = [private_ohe, private_adj, private_bpps, private_bpps_stats]
    private_preds = model_private.predict(private_inputs, verbose=0)
    print("private_preds:", private_preds.shape)



## === cell 14
preds_ls = []


def preds_to_rows(df, preds, seq_length):
    rows = []
    for i, uid in enumerate(df.id.values):
        single_pred = preds[i]  # (seq_length, 5)
        single_df = pd.DataFrame(single_pred, columns=pred_cols)
        single_df["id_seqpos"] = [f"{uid}_{x}" for x in range(seq_length)]
        rows.append(single_df)
    if len(rows) == 0:
        return pd.DataFrame(columns=["id_seqpos"] + pred_cols)
    return pd.concat(rows, axis=0, ignore_index=True)


for i, uid in tqdm(
    list(enumerate(public_df.id.values)), desc="Formatting public preds"
):
    single_pred = public_preds[i]  # (107,5)
    single_df = pd.DataFrame(single_pred, columns=pred_cols)
    single_df["id_seqpos"] = [f"{uid}_{x}" for x in range(single_df.shape[0])]
    preds_ls.append(single_df)

if private_preds is not None:
    for i, uid in tqdm(
        list(enumerate(private_df.id.values)), desc="Formatting private preds"
    ):
        single_pred = private_preds[i]
        single_df = pd.DataFrame(single_pred, columns=pred_cols)
        single_df["id_seqpos"] = [f"{uid}_{x}" for x in range(single_df.shape[0])]
        preds_ls.append(single_df)

preds_df = pd.concat(preds_ls, axis=0, ignore_index=True)

sample_sub = pd.read_csv(sample_sub_path)
submission = sample_sub[["id_seqpos"]].merge(preds_df, on="id_seqpos", how="left")

for c in pred_cols:
    if c not in submission.columns:
        submission[c] = 0.0
submission[pred_cols] = submission[pred_cols].fillna(0.0).astype(np.float32)

submission = submission[["id_seqpos"] + pred_cols]
print("submission shape:", submission.shape, "sample_sub shape:", sample_sub.shape)

submission.to_csv("submission.csv", index=False)
print("wrote to submission.csv")



## === cell 15
chk = pd.read_csv("submission.csv")
print(chk.head())
print("Columns:", list(chk.columns))
print("Any NaNs:", chk.isna().any().to_dict())
print("Row count ok:", chk.shape[0] == pd.read_csv(sample_sub_path).shape[0])
