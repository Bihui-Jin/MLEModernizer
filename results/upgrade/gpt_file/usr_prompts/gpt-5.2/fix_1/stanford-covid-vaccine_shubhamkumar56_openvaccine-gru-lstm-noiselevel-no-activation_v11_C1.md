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

0.38739

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import warnings
warnings.filterwarnings('ignore')

import pandas as pd, numpy as np
import math, json, gc, random, os, sys
from matplotlib import pyplot as plt
from tqdm import tqdm

import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras.backend as K
import tensorflow.keras.layers as L
from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.keras.layers import LSTM


from sklearn.model_selection import train_test_split, KFold


print("set up complete!")


## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train = pd.read_json('/kaggle/input/stanford-covid-vaccine/train.json', lines=True)
test = pd.read_json('/kaggle/input/stanford-covid-vaccine/test.json', lines=True)
sample_sub = pd.read_csv('/kaggle/input/stanford-covid-vaccine/sample_submission.csv')


print ("Data Load Complete")


## === cell 2
print(train.shape)
if ~ train.isnull().values.any(): print('No missing values')
train.head()


## === cell 3
print(test.shape)
if ~ test.isnull().values.any(): print('No missing values')
test.head()


## === cell 4
print(sample_sub.shape)
if ~ sample_sub.isnull().values.any(): print('No missing values')
sample_sub.head()


## === cell 5
target_cols = ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']


## === cell 6
token2int = {x:i for i, x in enumerate('().ACGUBEHIMSX')}


## === cell 7
token2int['U']


## === cell 8
cols=['sequence', 'structure', 'predicted_loop_type']
train[cols].applymap(lambda seq: [token2int[x] for x in seq])


## === cell 9
def preprocess_inputs(df, cols=['sequence', 'structure', 'predicted_loop_type']):
    return np.transpose(
        np.array(
            df[cols]
            .applymap(lambda seq: [token2int[x] for x in seq])
            .values
            .tolist()
        ),
        (0, 2, 1)
    )


## === cell 10
train_inputs = preprocess_inputs(train[train.signal_to_noise > 1])
train_y = np.array(train[train.signal_to_noise > 1][target_cols].values.tolist()).transpose((0, 2, 1))


## === cell 11
print (train_inputs.shape)

print (train_y.shape)


## === cell 12
inputs = tf.keras.layers.Input(shape=(107, 3))
embed = tf.keras.layers.Embedding(input_dim=len(token2int), output_dim=75)(inputs)
reshaped = tf.reshape(
        embed, shape=(-1, embed.shape[1],  embed.shape[2] * embed.shape[3]))


## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/436362424.py in <cell line: 0>()
      1 inputs = tf.keras.layers.Input(shape=(107, 3))
      2 embed = tf.keras.layers.Embedding(input_dim=len(token2int), output_dim=75)(inputs)
----> 3 reshaped = tf.reshape(
      4         embed, shape=(-1, embed.shape[1],  embed.shape[2] * embed.shape[3]))

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/weak_tensor_ops.py in wrapper(*args, **kwargs)
     86   def wrapper(*args, **kwargs):
     87     if not ops.is_auto_dtype_conversion_enabled():
---> 88       return op(*args, **kwargs)
     89     bound_arguments = signature.bind(*args, **kwargs)
     90     bound_arguments.apply_defaults()

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


## === cell 13
embed.shape


## === cell 14
def gru_layer(hidden_dim, dropout):
    return tf.keras.layers.Bidirectional(
                                tf.keras.layers.GRU(hidden_dim,
                                dropout=dropout,
                                return_sequences=True,
                                kernel_initializer = 'orthogonal'))

def lstm_layer(hidden_dim, dropout):
    return tf.keras.layers.Bidirectional(
                                tf.keras.layers.LSTM(hidden_dim,
                                dropout=dropout,
                                return_sequences=True,
                                kernel_initializer = 'orthogonal'))

def build_model(gru=False,seq_len=107, pred_len=68, dropout=0.5,
                embed_dim=75, hidden_dim=128):
    
    inputs = tf.keras.layers.Input(shape=(seq_len, 3))

    embed = tf.keras.layers.Embedding(input_dim=len(token2int), output_dim=embed_dim)(inputs)
    reshaped = tf.reshape(
        embed, shape=(-1, embed.shape[1],  embed.shape[2] * embed.shape[3]))
    
    reshaped = tf.keras.layers.SpatialDropout1D(.2)(reshaped)
    
    if gru:
        hidden = gru_layer(hidden_dim, dropout)(reshaped)
        hidden = gru_layer(hidden_dim, dropout)(hidden)
        hidden = gru_layer(hidden_dim, dropout)(hidden)
        
    else:
        hidden = lstm_layer(hidden_dim, dropout)(reshaped)
        hidden = lstm_layer(hidden_dim, dropout)(hidden)
        hidden = lstm_layer(hidden_dim, dropout)(hidden)
    
    truncated = hidden[:, :pred_len]
    
    out = tf.keras.layers.Dense(5, activation='linear')(truncated)

    model = tf.keras.Model(inputs=inputs, outputs=out)

    adam = tf.optimizers.Adam()
    radam = tfa.optimizers.RectifiedAdam()
    lookahead = tfa.optimizers.Lookahead(adam, sync_period=6)
    ranger = tfa.optimizers.Lookahead(radam, sync_period=6)
    
    model.compile(optimizer = adam, loss='mse')
    
    return model

print ("Model structure defined")


## === cell 15


def lstm_model (seq_len = 107,output_dim = 75,dropout = 0.5, pred_len = 68):
    
    
    inputs = tf.keras.layers.Input(shape=(seq_len, 3))

    embed = tf.keras.layers.Embedding(input_dim=len(token2int), output_dim=output_dim)(inputs)
    reshaped = tf.keras.layers.Reshape((seq_len,225), input_shape=(seq_len, 3, 75))(embed)
    


    hidden = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM (128,  dropout = dropout,kernel_initializer = 'orthogonal', return_sequences = True ))(reshaped)
    hidden = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM (128,  dropout = dropout,kernel_initializer = 'orthogonal', return_sequences = True ))(hidden)
    hidden = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM (128,  dropout = dropout,kernel_initializer = 'orthogonal', return_sequences = True ))(hidden)
    truncated = hidden[:,:pred_len]
    output = tf.keras.layers.Dense(5, activation='linear')(truncated)
    
    model = tf.keras.Model(inputs=inputs, outputs=output)

    adam = tf.optimizers.Adam(learning_rate = 0.01, decay = 0.0001)

    model.compile(loss='mse',
                optimizer= adam ,
                )
    return model


## === cell 16
train_data, val_data, train_labels, val_labels = train_test_split(train_inputs, train_y,
                                                                     test_size=.2, random_state=4)


## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1778787158.py in <cell line: 0>()
      1 # preparing data for simple LSTM
----> 2 train_data, val_data, train_labels, val_labels = train_test_split(train_inputs, train_y,
      3                                                                      test_size=.2, random_state=4)

NameError: name 'train_test_split' is not defined

## === cell 17
lr_callback = tf.keras.callbacks.ReduceLROnPlateau()


## === cell 18
smpl_lstm = lstm_model(seq_len = 107,output_dim = 75,dropout = 0.5, pred_len = 68)
sv_smpl_lstm = tf.keras.callbacks.ModelCheckpoint('model_smpl_lstm.h5')


## === cell 19
smpl_lstm.summary()


## === cell 20
history_smpl_lstm = smpl_lstm.fit(
    train_data, train_labels, 
    validation_data=(val_data,val_labels),
    batch_size=64,
    epochs=80,
    callbacks=[lr_callback,sv_smpl_lstm],
    verbose = 2
)

print(f"Min training loss={min(history_smpl_lstm.history['loss'])}, min validation loss={min(history_smpl_lstm.history['val_loss'])}")


## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1357393778.py in <cell line: 0>()
      1 history_smpl_lstm = smpl_lstm.fit(
----> 2     train_data, train_labels,
      3     validation_data=(val_data,val_labels),
      4     batch_size=64,
      5     epochs=80,

NameError: name 'train_data' is not defined

## === cell 21
gru = build_model(gru=True)
sv_gru = tf.keras.callbacks.ModelCheckpoint('model_gru.h5')


## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3901974226.py in <cell line: 0>()
----> 1 gru = build_model(gru=True)
      2 sv_gru = tf.keras.callbacks.ModelCheckpoint('model_gru.h5')

/tmp/ipykernel_11/2355881675.py in build_model(gru, seq_len, pred_len, dropout, embed_dim, hidden_dim)
     19 
     20     embed = tf.keras.layers.Embedding(input_dim=len(token2int), output_dim=embed_dim)(inputs)
---> 21     reshaped = tf.reshape(
     22         embed, shape=(-1, embed.shape[1],  embed.shape[2] * embed.shape[3]))
     23 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/weak_tensor_ops.py in wrapper(*args, **kwargs)
     86   def wrapper(*args, **kwargs):
     87     if not ops.is_auto_dtype_conversion_enabled():
---> 88       return op(*args, **kwargs)
     89     bound_arguments = signature.bind(*args, **kwargs)
     90     bound_arguments.apply_defaults()

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


## === cell 22
gru.summary()


## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2830488234.py in <cell line: 0>()
----> 1 gru.summary()

NameError: name 'gru' is not defined

## === cell 23
history_gru = gru.fit(
    train_data, train_labels, 
    validation_data=(val_data,val_labels),
    batch_size=64,
    epochs=70,
    callbacks=[lr_callback,sv_gru],
    verbose = 2
)

print(f"Min training loss={min(history_gru.history['loss'])}, min validation loss={min(history_gru.history['val_loss'])}")


## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3656801670.py in <cell line: 0>()
----> 1 history_gru = gru.fit(
      2     train_data, train_labels,
      3     validation_data=(val_data,val_labels),
      4     batch_size=64,
      5     epochs=70,

NameError: name 'gru' is not defined

## === cell 24
fig, ax = plt.subplots(1, 2,  figsize = (20, 10))

ax[0].plot(history_gru.history['loss'])
ax[0].plot(history_gru.history['val_loss'])



ax[1].plot(history_smpl_lstm.history['loss'])
ax[1].plot(history_smpl_lstm.history['val_loss'])

ax[0].set_title('GRU')

ax[1].set_title('SMPL_LSTM')

ax[0].legend(['train', 'validation'], loc = 'upper right')

ax[1].legend(['train', 'validation'], loc = 'upper right')

ax[0].set_ylabel('Loss')
ax[0].set_xlabel('Epoch')
ax[1].set_ylabel('Loss')
ax[1].set_xlabel('Epoch')


## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3299055180.py in <cell line: 0>()
      1 fig, ax = plt.subplots(1, 2,  figsize = (20, 10))
      2 
----> 3 ax[0].plot(history_gru.history['loss'])
      4 ax[0].plot(history_gru.history['val_loss'])
      5 

NameError: name 'history_gru' is not defined

## === cell 25
public_df = test.query("seq_length == 107").copy()
private_df = test.query("seq_length == 130").copy()

public_inputs = preprocess_inputs(public_df)
private_inputs = preprocess_inputs(private_df )

print ("Test data prepared!")


## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/588087126.py in <cell line: 0>()
      3 
      4 public_inputs = preprocess_inputs(public_df)
----> 5 private_inputs = preprocess_inputs(private_df )
      6 
      7 print ("Test data prepared!")

/tmp/ipykernel_11/3126562086.py in preprocess_inputs(df, cols)
      1 def preprocess_inputs(df, cols=['sequence', 'structure', 'predicted_loop_type']):
----> 2     return np.transpose(
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

## === cell 26
gru_short = build_model(gru=True, seq_len=107, pred_len=107)
gru_long = build_model(gru=True, seq_len=130, pred_len=130)
lstm_short = lstm_model(seq_len = 107,output_dim = 75,dropout = 0.5, pred_len = 107)
lstm_long = lstm_model(seq_len = 130,output_dim = 75,dropout = 0.5, pred_len = 130)

gru_short.load_weights('model_gru.h5')
gru_long.load_weights('model_gru.h5')
lstm_short.load_weights('model_smpl_lstm.h5')
lstm_long.load_weights('model_smpl_lstm.h5')

print ("models built and trained")


## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/398307571.py in <cell line: 0>()
      1 #build all models
----> 2 gru_short = build_model(gru=True, seq_len=107, pred_len=107)
      3 gru_long = build_model(gru=True, seq_len=130, pred_len=130)
      4 lstm_short = lstm_model(seq_len = 107,output_dim = 75,dropout = 0.5, pred_len = 107)
      5 lstm_long = lstm_model(seq_len = 130,output_dim = 75,dropout = 0.5, pred_len = 130)

/tmp/ipykernel_11/2355881675.py in build_model(gru, seq_len, pred_len, dropout, embed_dim, hidden_dim)
     19 
     20     embed = tf.keras.layers.Embedding(input_dim=len(token2int), output_dim=embed_dim)(inputs)
---> 21     reshaped = tf.reshape(
     22         embed, shape=(-1, embed.shape[1],  embed.shape[2] * embed.shape[3]))
     23 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/weak_tensor_ops.py in wrapper(*args, **kwargs)
     86   def wrapper(*args, **kwargs):
     87     if not ops.is_auto_dtype_conversion_enabled():
---> 88       return op(*args, **kwargs)
     89     bound_arguments = signature.bind(*args, **kwargs)
     90     bound_arguments.apply_defaults()

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


## === cell 27
gru_public_preds = gru_short.predict(public_inputs)
gru_private_preds = gru_long.predict(private_inputs)
lstm_public_preds = lstm_short.predict(public_inputs)
lstm_private_preds = lstm_long.predict(private_inputs)


## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2162102869.py in <cell line: 0>()
      1 #and predict
----> 2 gru_public_preds = gru_short.predict(public_inputs)
      3 gru_private_preds = gru_long.predict(private_inputs)
      4 lstm_public_preds = lstm_short.predict(public_inputs)
      5 lstm_private_preds = lstm_long.predict(private_inputs)

NameError: name 'gru_short' is not defined

## === cell 28
preds_gru = []

for df, preds in [(public_df, gru_public_preds), (private_df, gru_private_preds)]:
    for i, uid in enumerate(df.id):
        single_pred = preds[i]

        single_df = pd.DataFrame(single_pred, columns=target_cols)
        single_df['id_seqpos'] = [f'{uid}_{x}' for x in range(single_df.shape[0])]

        preds_gru.append(single_df)

preds_gru_df = pd.concat(preds_gru)
preds_gru_df.head()


## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1858050573.py in <cell line: 0>()
      1 preds_gru = []
      2 
----> 3 for df, preds in [(public_df, gru_public_preds), (private_df, gru_private_preds)]:
      4     for i, uid in enumerate(df.id):
      5         single_pred = preds[i]

NameError: name 'gru_public_preds' is not defined

## === cell 29
preds_lstm = []

for df, preds in [(public_df, lstm_public_preds), (private_df, lstm_private_preds)]:
    for i, uid in enumerate(df.id):
        single_pred = preds[i]

        single_df = pd.DataFrame(single_pred, columns=target_cols)
        single_df['id_seqpos'] = [f'{uid}_{x}' for x in range(single_df.shape[0])]

        preds_lstm.append(single_df)

preds_lstm_df = pd.concat(preds_lstm)
preds_lstm_df.head()


## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3685069532.py in <cell line: 0>()
      1 preds_lstm = []
      2 
----> 3 for df, preds in [(public_df, lstm_public_preds), (private_df, lstm_private_preds)]:
      4     for i, uid in enumerate(df.id):
      5         single_pred = preds[i]

NameError: name 'lstm_public_preds' is not defined

## === cell 30
lstm_weight = 0.5
gru_weight = 0.5
blend_preds_df = pd.DataFrame()
blend_preds_df['id_seqpos'] = preds_gru_df['id_seqpos']
blend_preds_df['reactivity'] = gru_weight*preds_gru_df['reactivity'] + lstm_weight*preds_lstm_df['reactivity']
blend_preds_df['deg_Mg_pH10'] = gru_weight*preds_gru_df['deg_Mg_pH10'] + lstm_weight*preds_lstm_df['deg_Mg_pH10']
blend_preds_df['deg_pH10'] = gru_weight*preds_gru_df['deg_pH10'] + lstm_weight*preds_lstm_df['deg_pH10']
blend_preds_df['deg_Mg_50C'] = gru_weight*preds_gru_df['deg_Mg_50C'] + lstm_weight*preds_lstm_df['deg_Mg_50C']
blend_preds_df['deg_50C'] = gru_weight*preds_gru_df['deg_50C'] + lstm_weight*preds_lstm_df['deg_50C']


## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/927656639.py in <cell line: 0>()
      2 gru_weight = 0.5
      3 blend_preds_df = pd.DataFrame()
----> 4 blend_preds_df['id_seqpos'] = preds_gru_df['id_seqpos']
      5 blend_preds_df['reactivity'] = gru_weight*preds_gru_df['reactivity'] + lstm_weight*preds_lstm_df['reactivity']
      6 blend_preds_df['deg_Mg_pH10'] = gru_weight*preds_gru_df['deg_Mg_pH10'] + lstm_weight*preds_lstm_df['deg_Mg_pH10']

NameError: name 'preds_gru_df' is not defined

## === cell 31
submission = sample_sub[['id_seqpos']].merge(blend_preds_df, on=['id_seqpos'])

submission.head()
len(submission)


## --- ERROR in cell 31, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/1449467614.py in <cell line: 0>()
----> 1 submission = sample_sub[['id_seqpos']].merge(blend_preds_df, on=['id_seqpos'])
      2 
      3 #sanity check
      4 submission.head()
      5 len(submission)

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in merge(self, right, how, on, left_on, right_on, left_index, right_index, sort, suffixes, copy, indicator, validate)
  10830         from pandas.core.reshape.merge import merge
  10831 
> 10832         return merge(
  10833             self,
  10834             right,

/usr/local/lib/python3.11/dist-packages/pandas/core/reshape/merge.py in merge(left, right, how, on, left_on, right_on, left_index, right_index, sort, suffixes, copy, indicator, validate)
    168         )
    169     else:
--> 170         op = _MergeOperation(
    171             left_df,
    172             right_df,

/usr/local/lib/python3.11/dist-packages/pandas/core/reshape/merge.py in __init__(self, left, right, how, on, left_on, right_on, left_index, right_index, sort, suffixes, indicator, validate)
    792             left_drop,
    793             right_drop,
--> 794         ) = self._get_merge_keys()
    795 
    796         if left_drop:

/usr/local/lib/python3.11/dist-packages/pandas/core/reshape/merge.py in _get_merge_keys(self)
   1295                         rk = cast(Hashable, rk)
   1296                         if rk is not None:
-> 1297                             right_keys.append(right._get_label_or_level_values(rk))
   1298                         else:
   1299                             # work-around for merge_asof(right_index=True)

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in _get_label_or_level_values(self, key, axis)
   1909             values = self.axes[axis].get_level_values(key)._values
   1910         else:
-> 1911             raise KeyError(key)
   1912 
   1913         # Check for duplicates

KeyError: 'id_seqpos'

## === cell 32
submission.to_csv('submission.csv', index=False)
print('Submission saved')


## --- ERROR in cell 32, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1957518295.py in <cell line: 0>()
----> 1 submission.to_csv('submission.csv', index=False)
      2 print('Submission saved')

NameError: name 'submission' is not defined
