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

0.39813

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
import json
import tensorflow as tf
from matplotlib import pyplot as plt



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
os.chdir("/kaggle")
print("CWD:", os.getcwd())



## === cell 2
train_path = "/kaggle/input/stanford-covid-vaccine/train.json"
test_path = "/kaggle/input/stanford-covid-vaccine/test.json"
sample_path = "/kaggle/input/stanford-covid-vaccine/sample_submission.csv"

train_data = pd.read_json(train_path, lines=True)
test_data = pd.read_json(test_path, lines=True)
submission_format = pd.read_csv(sample_path, encoding="utf-8-sig")
print(train_data.shape, test_data.shape, submission_format.shape)



## === cell 3
token2int = {x: i for i, x in enumerate("().ACGUBEHIMSX")}
target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]




## === cell 4
def preprocess_inputs(df):
    """
    Convert the three categorical string columns into integer token arrays.
    Returns a NumPy array of shape (samples, seq_len, 3).
    """
    seq_len = df.iloc[0].seq_length  # all rows share the same length (107)
    cat_arrays = []
    for col in ["sequence", "structure", "predicted_loop_type"]:
        tokenised = df[col].apply(lambda s: [token2int[ch] for ch in s])
        cat_arrays.append(np.stack(tokenised.values))
    return np.stack(cat_arrays, axis=2)




## === cell 5
high_quality = train_data[train_data["signal_to_noise"] > 1].reset_index(drop=True)

train_inputs = preprocess_inputs(high_quality)

seq_len = 107
pred_len = 68  # number of scored positions
n_samples = high_quality.shape[0]

train_labels = np.zeros((n_samples, seq_len, len(target_cols)), dtype=np.float32)

true_vals = np.stack(high_quality[target_cols].values)  # shape (5, n, pred_len)
true_vals = true_vals.transpose(1, 2, 0)  # (n, pred_len, 5)
train_labels[:, :pred_len, :] = true_vals




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/3624698992.py in <cell line: 0>()
     15 # Fill the first `pred_len` positions with the true values
     16 true_vals = np.stack(high_quality[target_cols].values)  # shape (5, n, pred_len)
---> 17 true_vals = true_vals.transpose(1, 2, 0)  # (n, pred_len, 5)
     18 train_labels[:, :pred_len, :] = true_vals
     19 

ValueError: axes don't match array

## === cell 6
def lstm_layer(hidden_dim, dropout):
    return tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            hidden_dim,
            dropout=dropout,
            return_sequences=True,
            kernel_initializer="orthogonal",
        )
    )


def MCRMSE(y_true, y_pred):
    colwise_mse = tf.reduce_mean(tf.square(y_true - y_pred), axis=1)
    return tf.reduce_mean(tf.sqrt(colwise_mse), axis=1)


def build_model(seq_len=107, embed_dim=32, hidden_dim=128, dropout=0.2, pred_len=68):
    """
    Very lightweight model:
    - Input: integer tokens (shape: seq_len x 3)
    - Cast to float and feed directly into a bidirectional LSTM
    - Dense layer produces 5 regression targets
    """
    inputs = tf.keras.Input(shape=(seq_len, 3), dtype=tf.int32, name="tokens")
    x = tf.cast(inputs, tf.float32)  # treat token IDs as numeric features
    x = lstm_layer(hidden_dim, dropout)(x)
    x = x[:, :pred_len, :]  # keep only the scored positions
    outputs = tf.keras.layers.Dense(len(target_cols), activation="linear")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss=MCRMSE)
    return model




## === cell 7
EPOCHS = 5  # keep runtime short; sufficient for a valid submission
BATCH_SIZE = 32

model = build_model(seq_len=seq_len, pred_len=pred_len)
model.summary()

history = model.fit(
    train_inputs, train_labels, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=2
)



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/3575175262.py in <cell line: 0>()
      2 BATCH_SIZE = 32
      3 
----> 4 model = build_model(seq_len=seq_len, pred_len=pred_len)
      5 model.summary()
      6 

/tmp/ipykernel_55/1834149133.py in build_model(seq_len, embed_dim, hidden_dim, dropout, pred_len)
     24     """
     25     inputs = tf.keras.Input(shape=(seq_len, 3), dtype=tf.int32, name="tokens")
---> 26     x = tf.cast(inputs, tf.float32)  # treat token IDs as numeric features
     27     x = lstm_layer(hidden_dim, dropout)(x)
     28     x = x[:, :pred_len, :]  # keep only the scored positions

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


## === cell 8
plt.figure(figsize=(8, 4))
plt.plot(history.history["loss"], marker="o")
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2599129691.py in <cell line: 0>()
      1 # Plot training loss
      2 plt.figure(figsize=(8, 4))
----> 3 plt.plot(history.history["loss"], marker="o")
      4 plt.title("Training Loss")
      5 plt.xlabel("Epoch")

NameError: name 'history' is not defined

## === cell 9
def predict_full(df, model, pred_len):
    """
    Predict for all positions (seq_len). The model outputs only the first `pred_len`
    predictions; the remaining positions are filled with zeros.
    """
    inputs = preprocess_inputs(df)
    preds_scored = model.predict(inputs, verbose=0)  # shape (n, pred_len, 5)
    n = preds_scored.shape[0]
    full_preds = np.zeros(
        (n, df.iloc[0].seq_length, len(target_cols)), dtype=np.float32
    )
    full_preds[:, :pred_len, :] = preds_scored
    return full_preds


public_df = test_data[test_data["seq_length"] == 107].reset_index(drop=True)
public_preds = predict_full(public_df, model, pred_len)

private_df = test_data[test_data["seq_length"] != 107].reset_index(drop=True)
if not private_df.empty:
    private_preds = predict_full(private_df, model, pred_len)
else:
    private_preds = np.empty((0, 0, 0))




## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2155066639.py in <cell line: 0>()
     16 # Public test set (seq_length == 107)
     17 public_df = test_data[test_data["seq_length"] == 107].reset_index(drop=True)
---> 18 public_preds = predict_full(public_df, model, pred_len)
     19 
     20 # Private test set (if any longer sequences exist; handle gracefully)

NameError: name 'model' is not defined

## === cell 10
def format_predictions(df, preds):
    """
    Convert model predictions into the Kaggle submission format.
    """
    rows = []
    for idx, row in df.iterrows():
        seq_len = row["seq_length"]
        pred_array = preds[idx]  # shape (seq_len, 5) after padding
        df_pred = pd.DataFrame(pred_array, columns=target_cols)
        df_pred["id_seqpos"] = [f"{row['id']}_{pos}" for pos in range(seq_len)]
        rows.append(df_pred)
    return pd.concat(rows, ignore_index=True)


lstm_preds_public = format_predictions(public_df, public_preds)
if private_df.empty:
    lstm_preds = lstm_preds_public
else:
    lstm_preds_private = format_predictions(private_df, private_preds)
    lstm_preds = pd.concat([lstm_preds_public, lstm_preds_private], ignore_index=True)



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1845504592.py in <cell line: 0>()
     13 
     14 
---> 15 lstm_preds_public = format_predictions(public_df, public_preds)
     16 if private_df.empty:
     17     lstm_preds = lstm_preds_public

NameError: name 'public_preds' is not defined

## === cell 11
submission = submission_format[["id_seqpos"]].merge(
    lstm_preds, on="id_seqpos", how="left"
)
submission = submission.fillna(0)  # any missing rows get zeros

print("Submission shape:", submission.shape)
print(submission.head())



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/62342746.py in <cell line: 0>()
      1 # Merge with the sample submission to guarantee correct ordering / columns
      2 submission = submission_format[["id_seqpos"]].merge(
----> 3     lstm_preds, on="id_seqpos", how="left"
      4 )
      5 submission = submission.fillna(0)  # any missing rows get zeros

NameError: name 'lstm_preds' is not defined

## === cell 12
os.chdir("/kaggle/working")
submission.to_csv("submission.csv", index=False)
print("Saved submission.csv")

## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2522350028.py in <cell line: 0>()
      1 # Write the final CSV
      2 os.chdir("/kaggle/working")
----> 3 submission.to_csv("submission.csv", index=False)
      4 print("Saved submission.csv")

NameError: name 'submission' is not defined
