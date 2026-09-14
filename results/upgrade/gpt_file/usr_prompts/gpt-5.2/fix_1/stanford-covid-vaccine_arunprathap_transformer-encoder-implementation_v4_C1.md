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
plotly==5.24.1
plotly-express==0.4.1
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

0.47792

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 2
import os 
import sys
import json
import math
import random
import numpy as np
import pandas as pd
import gc
from tqdm import tqdm

import matplotlib.pyplot as plt 
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split, KFold,  StratifiedKFold

import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras.backend as K
import tensorflow.keras.layers as L

import warnings
warnings.filterwarnings("ignore")


## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 3
seed = 42


## === cell 4
DEVICE = "TPU"
if DEVICE == "TPU":
    print("connecting to TPU...")
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        print('Running on TPU ', tpu.master())
    except ValueError:
        print("Could not connect to TPU")
        tpu = None

    if tpu:
        try:
            print("initializing  TPU ...")
            tf.config.experimental_connect_to_cluster(tpu)
            tf.tpu.experimental.initialize_tpu_system(tpu)
            strategy = tf.distribute.experimental.TPUStrategy(tpu)
            print("TPU initialized")
        except _:
            print("failed to initialize TPU")
    else:
        DEVICE = "GPU"

if DEVICE != "TPU":
    strategy = tf.distribute.get_strategy()

if DEVICE == "GPU":
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))  
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))


## === cell 5
dropout_model = 0.36
hidden_dim_first = 128
hidden_dim_second = 256
hidden_dim_third = 128


## === cell 6
train = pd.read_json('/kaggle/input/stanford-covid-vaccine/train.json', lines=True)
test = pd.read_json('/kaggle/input/stanford-covid-vaccine/test.json', lines=True)
sample_sub = pd.read_csv("/kaggle/input/stanford-covid-vaccine/sample_submission.csv")
train = train[train.signal_to_noise > 1]


## === cell 7
target_cols = ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']


## === cell 8
token2int = {x:i for i, x in enumerate('().ACGUBEHIMSX')}


## === cell 9
def preprocess_inputs(df, cols=['sequence','predicted_loop_type','structure']):
    base_features = np.transpose(
        np.array(
            df[cols]
            .applymap(lambda seq: [token2int[x] for x in seq])
            .values
            .tolist()
        ),
        (0, 2, 1)
    )
    return base_features

train_inputs_all = preprocess_inputs(train)
train_labels_all = np.array(train[target_cols].values.tolist()).transpose((0, 2, 1))


## === cell 10
def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead) 
    but it must be broadcastable for addition.
  
    Args:
      q: query shape == (..., seq_len_q, depth)
      k: key shape == (..., seq_len_k, depth)
      v: value shape == (..., seq_len_v, depth_v)
      mask: Float tensor with shape broadcastable 
            to (..., seq_len_q, seq_len_k). Defaults to None.
    
    Returns:
     output, attention_weights
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
  
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    if mask is not None:
        scaled_attention_logits += (mask * -1e9)  

    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)
        
    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention, 
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights
    
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'depth': self.depth,
            'wq': self.wq,
            'qk': self.wk,
            'wv': self.wv,
            'dense': self.dense,
        })
        
        return config
def point_wise_feed_forward_network(d_model, dff):
      return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
      tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
  ])

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.rate = rate
        
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
    
    def call(self, x, training):
        attn_output, _ = self.mha(x, x, x, None)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2

    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'num_heads': self.num_heads,
            'rate': self.rate,
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'dropout1': self.dropout1,
            'dropout2': self.dropout2,
            'layernorm1': self.layernorm1,
            'layernorm2': self.layernorm2,
            'mha': self.mha,
            'ffn': self.ffn,
        })
        return config


## === cell 11
def MCRMSE(y_true, y_pred):
    colwise_mse = tf.reduce_mean(tf.square(y_true - y_pred), axis=1)
    return tf.reduce_mean(tf.sqrt(colwise_mse), axis=1)

def build_model(model_type=1, seq_len=107, pred_len=68, embed_dim=32, 
                dropout=dropout_model, hidden_dim_first = hidden_dim_first, 
                hidden_dim_second = hidden_dim_second, hidden_dim_third = hidden_dim_third):
    
    inputs = tf.keras.layers.Input(shape=(seq_len, 3))

    categorical_feat_dim = 3
    categorical_fea = inputs[:, :, :categorical_feat_dim]
    
    embed = tf.keras.layers.Embedding(input_dim=len(token2int), output_dim=embed_dim)(categorical_fea)
    reshaped = tf.reshape(
        embed, shape=(-1, embed.shape[1],  embed.shape[2] * embed.shape[3]))
    
    reshaped = tf.keras.layers.SpatialDropout1D(.2)(reshaped)
    
    hidden = EncoderLayer(96, 8, 256)(reshaped)
    hidden = EncoderLayer(96, 8, 256)(hidden)
        
    truncated = hidden[:, :pred_len]

    out = tf.keras.layers.Dense(len(target_cols), activation='linear')(truncated)

    model = tf.keras.Model(inputs=inputs, outputs=out)

    adam = tf.optimizers.Adam()
    model.compile(optimizer=adam, loss=MCRMSE, metrics=[tf.keras.metrics.RootMeanSquaredError()])
    
    return model


## === cell 12
tf.keras.backend.clear_session()
from tqdm.keras import TqdmCallback
lr_callback = tf.keras.callbacks.ReduceLROnPlateau()
es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',restore_best_weights=True,min_delta=0.001, patience=10)


## === cell 13
def train_and_predict(n_folds=5, model_name="model", model_type=0, epochs=100, debug=True,
                      dropout_model=dropout_model, hidden_dim_first = hidden_dim_first, 
                      hidden_dim_second = hidden_dim_second, hidden_dim_third = hidden_dim_third,
                      seed=seed):

    print("Model:", model_name)

    ensemble_preds = pd.DataFrame(index=sample_sub.index, columns=target_cols).fillna(0) # test dataframe with 0 values
    kf = KFold(n_folds, shuffle=True, random_state=seed)
    skf = StratifiedKFold(n_folds, shuffle=True, random_state=seed)
    val_losses = []
    historys = []

    for i, (train_index, val_index) in enumerate(skf.split(train_inputs_all, train['SN_filter'])):
        print("Fold:", str(i+1))
        with strategy.scope():
            model_train = build_model(model_type=model_type, 
                                      dropout=dropout_model, 
                                      hidden_dim_first = hidden_dim_first, 
                                      hidden_dim_second = hidden_dim_second, 
                                      hidden_dim_third = hidden_dim_third)
            model_short = build_model(model_type=model_type, seq_len=107, pred_len=107,
                                      dropout=dropout_model, 
                                      hidden_dim_first = hidden_dim_first, 
                                      hidden_dim_second = hidden_dim_second, 
                                      hidden_dim_third = hidden_dim_third)
            model_long = build_model(model_type=model_type, seq_len=130, pred_len=130,
                                     dropout=dropout_model, 
                                     hidden_dim_first = hidden_dim_first, 
                                     hidden_dim_second = hidden_dim_second, 
                                     hidden_dim_third = hidden_dim_third)

        train_inputs, train_labels = train_inputs_all[train_index], train_labels_all[train_index]
        val_inputs, val_labels = train_inputs_all[val_index], train_labels_all[val_index]

        checkpoint = tf.keras.callbacks.ModelCheckpoint(f'{model_name}_Fold_{str(i+1)}.h5')

        history = model_train.fit(
            train_inputs , train_labels, 
            validation_data=(val_inputs,val_labels),
            batch_size=64,
            epochs=epochs, 
            callbacks=[checkpoint,
                       lr_callback,
                       TqdmCallback(),
                       tf.keras.callbacks.TerminateOnNaN(),
                       es_callback],
            verbose= 0
        )

        print(f"{model_name} Min training loss={min(history.history['loss'])}, min validation loss={min(history.history['val_loss'])}")

        val_losses.append(min(history.history['val_loss']))
        historys.append(history)

        model_short.load_weights(f'{model_name}_Fold_{str(i+1)}.h5')
        model_long.load_weights(f'{model_name}_Fold_{str(i+1)}.h5')

        public_preds = model_short.predict(public_inputs)
        private_preds = model_long.predict(private_inputs)

        preds_model = []
        for df, preds in [(public_df, public_preds), (private_df, private_preds)]:
            for i, uid in enumerate(df.id):
                single_pred = preds[i]

                single_df = pd.DataFrame(single_pred, columns=target_cols)
                single_df['id_seqpos'] = [f'{uid}_{x}' for x in range(single_df.shape[0])]

                preds_model.append(single_df)

        preds_model_df = pd.concat(preds_model)
        ensemble_preds[target_cols] += preds_model_df[target_cols].values / n_folds

        if debug:
            print("Intermediate ensemble result")
            print(ensemble_preds[target_cols].head())

    ensemble_preds["id_seqpos"] = preds_model_df["id_seqpos"].values
    ensemble_preds = pd.merge(sample_sub["id_seqpos"], ensemble_preds, on="id_seqpos", how="left")

    print("Mean Validation loss:", str(np.mean(val_losses)))

    if debug:
        fig, ax = plt.subplots(1, 3, figsize = (20, 10))
        for i, history in enumerate(historys):
            ax[0].plot(history.history['loss'])
            ax[0].plot(history.history['val_loss'])
            ax[0].set_title('model_'+str(i+1))
            ax[0].set_ylabel('Loss')
            ax[0].set_xlabel('Epoch')
            
            ax[1].plot(history.history['root_mean_squared_error'])
            ax[1].plot(history.history['val_root_mean_squared_error'])
            ax[1].set_title('model_'+str(i+1))
            ax[1].set_ylabel('RMSE')
            ax[1].set_xlabel('Epoch')
            
            ax[2].plot(history.history['lr'])
            ax[2].set_title('model_'+str(i+1))
            ax[2].set_ylabel('LR')
            ax[2].set_xlabel('Epoch')
        plt.show()

    return ensemble_preds


public_df = test.query("seq_length == 107").copy()
private_df = test.query("seq_length == 130").copy()
public_inputs = preprocess_inputs(public_df)
private_inputs = preprocess_inputs(private_df)

ensembles = []

for i in range(1):
    model_name = "model_"+str(i+1)

    ensemble = train_and_predict(n_folds=5, model_name=model_name, model_type=i, epochs=40,
                                 dropout_model=dropout_model, hidden_dim_first = hidden_dim_first, 
                                 hidden_dim_second = hidden_dim_second, hidden_dim_third = hidden_dim_third,
                                 seed=seed)
    ensembles.append(ensemble)


## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/152820280.py in <cell line: 0>()
    109 private_df = test.query("seq_length == 130").copy()
    110 public_inputs = preprocess_inputs(public_df)
--> 111 private_inputs = preprocess_inputs(private_df)
    112 
    113 ensembles = []

/tmp/ipykernel_11/3769076311.py in preprocess_inputs(df, cols)
      1 def preprocess_inputs(df, cols=['sequence','predicted_loop_type','structure']):
----> 2     base_features = np.transpose(
      3         np.array(
      4             df[cols]
      5             .applymap(lambda seq: [token2int[x] for x in seq])

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

## === cell 14
ensemble_final = ensembles[0].copy()
ensemble_final[target_cols] = 0

for ensemble in ensembles:
    ensemble_final[target_cols] += ensemble[target_cols].values / len(ensembles)

ensemble_final.head().T


## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2699847937.py in <cell line: 0>()
      1 # Ensembling the solutions
----> 2 ensemble_final = ensembles[0].copy()
      3 ensemble_final[target_cols] = 0
      4 
      5 for ensemble in ensembles:

NameError: name 'ensembles' is not defined

## === cell 15
blend_preds_df = pd.DataFrame()
blend_preds_df['id_seqpos'] = ensemble_final['id_seqpos']
blend_preds_df['reactivity'] = ensemble_final['reactivity'] 
blend_preds_df['deg_Mg_pH10'] = ensemble_final['deg_Mg_pH10']
blend_preds_df['deg_pH10'] = ensemble_final['deg_pH10']
blend_preds_df['deg_Mg_50C'] = ensemble_final['deg_Mg_50C']
blend_preds_df['deg_50C'] = ensemble_final['deg_50C']
blend_preds_df.head().T


## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1350554416.py in <cell line: 0>()
      1 blend_preds_df = pd.DataFrame()
----> 2 blend_preds_df['id_seqpos'] = ensemble_final['id_seqpos']
      3 blend_preds_df['reactivity'] = ensemble_final['reactivity']
      4 blend_preds_df['deg_Mg_pH10'] = ensemble_final['deg_Mg_pH10']
      5 blend_preds_df['deg_pH10'] = ensemble_final['deg_pH10']

NameError: name 'ensemble_final' is not defined

## === cell 16
blend_preds_df.to_csv('ensemble_final.csv', index=False)


## --- ERROR in outputing the csv:
Invalid submission: Expected submission to be the same length as answers, but got 0 instead of 25680.
