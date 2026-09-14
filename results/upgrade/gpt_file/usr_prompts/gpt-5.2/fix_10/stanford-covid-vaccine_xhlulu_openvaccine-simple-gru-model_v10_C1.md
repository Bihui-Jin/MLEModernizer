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

0.40561

# 6. Current score

0.24917

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 0.6355) has done: 'I fix the environment-breaking import that triggers the protobuf/plotly `MessageFactory` error by removing the unused Plotly dependency. Then I correct the input preprocessing so it always returns a numeric `(n, 107, 3)` array (the current version can produce an object array and breaks transpose for empty/private splits), and I simplify test handling since this competition’s test has only length 107 (so the “private 130” branch is invalid). Finally, I fix the Keras Functional model build error by replacing the raw `tf.reshape` on a KerasTensor with a Keras `Reshape` layer and by feeding integer token IDs (not floats) into the embedding; the rest of the GRU architecture/training loop stays the same and now run end-to-end to write `submission.csv`.'
- What this solution (achieved 0.27343) has done: 'I fix the environment-breaking `MessageFactory.GetPrototype` import error by forcing protobuf to use the pure-Python implementation before TensorFlow (and anything that triggers protobuf/plotly) loads. Then I fix the Keras `ModelCheckpoint` filename requirement by switching from `model.h5` to `model.weights.h5` and consistently loading that same file in inference, which resolves the downstream `FileNotFoundError`. Finally, I keep the model and training loop identical otherwise, ensuring the pipeline runs end-to-end and writes a valid `submission.csv` with the exact required columns.'
- What this solution (achieved 0.25296) has done: 'I fix the protobuf/TensorFlow import crash by forcing the pure-Python protobuf implementation *before* any TensorFlow import and by pinning a safe protobuf version via a runtime check (fallback: avoid importing modules that trigger the incompatible C++ protobuf path). Then I fix a silent train/val leakage/quality issue by applying the competition’s standard `SN_filter==1` training filter (score-relevant but minimal and consistent with the dataset notes) while keeping the same GRU model, loss, and training loop. Finally, I ensure inference outputs correctly fill all 107 positions (but remain scored on first 68) and always write a valid `submission.csv` matching `sample_submission.csv` row order and columns.'
- What this solution (achieved 0.25631) has done: 'To fix the crash, I remove the protobuf environment forcing that is now incompatible with the Kaggle image (it triggers the `MessageFactory.GetPrototype` failure during TensorFlow import). I keep the GRU model, loss, preprocessing, training loop, and submission-building logic the same, only adjusting the import/seed cell so the notebook runs end-to-end again. This should restore your previous working pipeline and produce a valid `submission.csv` without changing model semantics or score behavior beyond negligible randomness (we keep the same seeds). No score-tuning changes are made since your current score is already substantially better than the target band and the priority is correctness/execution.'
- What this solution (achieved 0.25334) has done: 'I fix the TensorFlow import crash caused by an incompatible protobuf runtime by forcing protobuf to use the pure-Python implementation *before* any TensorFlow-related imports occur, which avoids the `MessageFactory.GetPrototype` failure. I keep the model, preprocessing, training loop, and submission-building logic unchanged to preserve score behavior (your current score is already better than the target, and lower is better). I also add a small safety fallback to locate the dataset whether it’s in `/kaggle/input/stanford-covid-vaccine/` or directly in `/kaggle/input/`, without changing paths if the original exists. Finally, I ensure the pipeline always writes a valid `submission.csv` with the exact required columns and row alignment.'
- What this solution (achieved 0.25375) has done: 'We fix the immediate runtime crash by removing the protobuf environment forcing that is incompatible with this Kaggle image and triggers the `MessageFactory.GetPrototype` error during TensorFlow import. To keep the rest of the pipeline identical (and score behavior stable), we won’t change the model, preprocessing, training loop, or prediction logic beyond this import stability fix. We also add a tiny, score-neutral fallback to pick the correct dataset directory between `/kaggle/input/stanford-covid-vaccine/` and `/kaggle/input/` as your script already intends. The output still be a valid `submission.csv` with the exact required columns and row alignment.'
- What this solution (achieved 0.25109) has done: 'We fix the immediate TensorFlow import crash caused by an incompatible protobuf runtime by forcing protobuf to use the pure-Python implementation *before* importing TensorFlow, which avoids the `MessageFactory.GetPrototype` error in this environment. This is an execution-only change and should be score-neutral (model/data/training remain identical). To keep the pipeline robust, we also ensure the environment variables are set early and deterministically, but we won’t change the GRU model, preprocessing, training loop, or submission formatting. The script then run end-to-end and write a valid `submission.csv` with the required columns and row alignment.'
- What this solution (achieved 0.25186) has done: 'I fix the TensorFlow import crash by removing the protobuf environment forcing that is incompatible with this Kaggle image (it triggers the `MessageFactory.GetPrototype` error). Then I keep the model, preprocessing, training loop, and submission formatting identical so score behavior remains stable (your current score is already better than the target, and lower is better). I also add a tiny safety check around the token mapping to fail early if an unexpected character appears, avoiding silent bad encodings. The script run end-to-end and write a valid `submission.csv` with the exact required columns and row alignment.'
- What this solution (achieved 0.24917) has done: 'The current failure happens before any training because TensorFlow’s protobuf dependency is incompatible with the default C++ protobuf runtime in this environment, causing `MessageFactory.GetPrototype` to be missing. The minimal fix is to force protobuf to use the pure-Python implementation *before* importing TensorFlow, which avoids that crash while keeping the model, preprocessing, training loop, and submission logic unchanged (so score behavior should remain essentially the same). I also keep the existing dataset path fallback and ensure the submission is written as `submission.csv` with the exact required columns and row alignment.'

# 9. Code solution

## === cell 0
import os
import json
import numpy as np
import pandas as pd

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"

import tensorflow as tf
import tensorflow.keras.layers as L
from sklearn.model_selection import train_test_split

print("TF version:", tf.__version__)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
tf.random.set_seed(2020)
np.random.seed(2020)



## === cell 2
pred_cols = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C", "deg_pH10", "deg_50C"]



## === cell 3
y_true = tf.random.normal((32, 68, 3))
y_pred = tf.random.normal((32, 68, 3))




## === cell 4
def MCRMSE(y_true, y_pred):
    colwise_mse = tf.reduce_mean(tf.square(y_true - y_pred), axis=(0, 1))
    return tf.reduce_mean(tf.sqrt(colwise_mse), axis=-1)




## === cell 5
def gru_layer(hidden_dim, dropout):
    return L.Bidirectional(
        L.GRU(
            hidden_dim,
            dropout=dropout,
            return_sequences=True,
            kernel_initializer="orthogonal",
        )
    )




## === cell 6
def build_model(
    embed_size,
    seq_len=107,
    pred_len=68,
    dropout=0.5,
    sp_dropout=0.2,
    embed_dim=75,
    hidden_dim=128,
    n_layers=2,
):
    inputs = L.Input(shape=(seq_len, 3), dtype=tf.int32)

    embed = L.Embedding(input_dim=embed_size, output_dim=embed_dim)(inputs)
    reshaped = L.Reshape((seq_len, 3 * embed_dim))(embed)
    hidden = L.SpatialDropout1D(sp_dropout)(reshaped)

    for _ in range(n_layers):
        hidden = gru_layer(hidden_dim, dropout)(hidden)

    truncated = hidden[:, :pred_len]
    out = L.Dense(5, activation="linear")(truncated)

    model = tf.keras.Model(inputs=inputs, outputs=out)
    model.compile(tf.optimizers.Adam(), loss=MCRMSE)
    return model




## === cell 7
def pandas_list_to_array(df):
    """
    Input: dataframe of shape (n, c), containing list-like of length L in each cell
    Return: np.array of shape (n, L, c)
    """
    vals = df.to_numpy()
    n, c = vals.shape
    arr = np.stack([np.stack(vals[i], axis=1) for i in range(n)], axis=0)
    return arr




## === cell 8
def preprocess_inputs(
    df, token2int, cols=["sequence", "structure", "predicted_loop_type"]
):
    allowed = set(token2int.keys())
    for col in cols:
        bad = set("".join(df[col].astype(str).tolist())) - allowed
        if bad:
            raise ValueError(f"Unexpected tokens in {col}: {sorted(bad)}")

    mapped = df[cols].applymap(lambda seq: [token2int[x] for x in seq])
    return pandas_list_to_array(mapped).astype(np.int32)




## === cell 9
data_dir = "/kaggle/input/stanford-covid-vaccine/"
if not os.path.exists(os.path.join(data_dir, "train.json")):
    data_dir = "/kaggle/input/"

train = pd.read_json(os.path.join(data_dir, "train.json"), lines=True)
test = pd.read_json(os.path.join(data_dir, "test.json"), lines=True)
sample_df = pd.read_csv(os.path.join(data_dir, "sample_submission.csv"))

print(
    "Using data_dir:",
    data_dir,
    "\ntrain shape:",
    train.shape,
    "\ntest shape:",
    test.shape,
    "\nsample shape:",
    sample_df.shape,
)



## === cell 10
if "SN_filter" in train.columns:
    train = train.query("SN_filter == 1").reset_index(drop=True)
print("train after SN_filter==1:", train.shape)

token2int = {x: i for i, x in enumerate("().ACGUBEHIMSX")}

train_inputs = preprocess_inputs(train, token2int)
train_labels = pandas_list_to_array(train[pred_cols]).astype(np.float32)

print("train_inputs:", train_inputs.shape, train_inputs.dtype)
print("train_labels:", train_labels.shape, train_labels.dtype)



## === cell 11
x_train, x_val, y_train, y_val = train_test_split(
    train_inputs, train_labels, test_size=0.1, random_state=34
)

print("x_train:", x_train.shape, "x_val:", x_val.shape)



## === cell 12
test_df = test.query("seq_length == 107").copy()
test_inputs = preprocess_inputs(test_df, token2int)

print("test_inputs:", test_inputs.shape, test_inputs.dtype)



## === cell 13
model = build_model(embed_size=len(token2int), seq_len=107, pred_len=68)
model.summary()



## === cell 14
weights_path = "model.weights.h5"

history = model.fit(
    x_train,
    y_train,
    validation_data=(x_val, y_val),
    batch_size=64,
    epochs=100,
    verbose=2,
    callbacks=[
        tf.keras.callbacks.ReduceLROnPlateau(patience=5),
        tf.keras.callbacks.ModelCheckpoint(
            weights_path,
            save_weights_only=True,
            save_best_only=True,
            monitor="val_loss",
            mode="min",
        ),
    ],
)



## === cell 15
model_test = build_model(seq_len=107, pred_len=107, embed_size=len(token2int))
model_test.load_weights(weights_path)



## === cell 16
test_preds = model_test.predict(test_inputs, batch_size=64, verbose=1)
print("test_preds:", test_preds.shape)



## === cell 17
preds_ls = []
for i, uid in enumerate(test_df.id):
    single_pred = test_preds[i]  # (107, 5)
    single_df = pd.DataFrame(single_pred, columns=pred_cols)
    single_df["id_seqpos"] = [f"{uid}_{x}" for x in range(single_df.shape[0])]
    preds_ls.append(single_df)

preds_df = pd.concat(preds_ls, ignore_index=True)
print("preds_df:", preds_df.shape)



## === cell 18
submission = sample_df[["id_seqpos"]].merge(preds_df, on=["id_seqpos"], how="left")

for c in pred_cols:
    submission[c] = submission[c].astype(np.float32).fillna(0.0)

submission.to_csv("submission.csv", index=False)
print(submission.shape)
print(submission.head())
print("Wrote submission.csv")
