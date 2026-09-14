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

0.38677

# 6. Current score

0.23892

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.2427) has done: 'I remove the import that triggers the protobuf/plotly `MessageFactory.GetPrototype` crash and keep plotting optional so the pipeline can run in the Kaggle environment. I fix the model-building errors by making the input integer tokens (not float) and replacing the invalid `tf.reshape` on a KerasTensor with a Keras `Reshape` layer, preserving the same architecture intent. I also fix the test split logic: in this competition the test set is all `seq_length==107`, so we won’t try to build a nonexistent 130-length “private” branch; instead we generate predictions for all test rows and write a correctly-aligned submission matching `sample_submission.csv`. Finally, I ensure the saved weights filename is compatible with TF/Keras 2.18 and that `submission.csv` is always produced end-to-end.'
- What this solution (achieved 0.23892) has done: 'We fix the protobuf-triggered `MessageFactory.GetPrototype` crash by forcing the pure-Python protobuf implementation before TensorFlow imports, which avoids the incompatible fast C++ API path in this Kaggle image. We also make the custom MCRMSE loss return a scalar (the current version returns a per-sample vector), which is a silent logic bug that can destabilize training and harm score, while preserving the intended metric. Finally, we add a couple of small stability guards (determinism seeds, token default mapping) and keep the model/training loop and submission construction unchanged so it still writes a valid `submission.csv` aligned to `sample_submission.csv`.'
- What this solution (achieved 0.2403) has done: 'We fix the protobuf crash that currently happens at import time by forcing the pure-Python protobuf implementation *and* disabling the C++ implementation explicitly before importing TensorFlow. This is a runtime-only stability fix and won’t change your model logic or training semantics. We also add a small, safe fallback to load the dataset from either `/kaggle/input/stanford-covid-vaccine/` or `/kaggle/input/` so the notebook runs regardless of which path is mounted. Everything else (model architecture, loss, training loop, prediction shaping, and submission construction) is kept the same so your score behavior should remain essentially unchanged aside from being able to run end-to-end.'
- What this solution (achieved 0.24111) has done: 'We fix the import-time protobuf crash by forcing the pure-Python protobuf implementation *and* disabling the C++ implementation before TensorFlow is imported (the current `setdefault` can be too late/ineffective in this Kaggle image). This is a runtime stability fix and should be score-neutral: it doesn’t change the model, training loop, data, or evaluation semantics. We also keep the existing fallback input path logic and ensure the script always writes `submission.csv` with the exact required columns aligned to `sample_submission.csv`. No model architecture or training behavior is changed beyond making the environment imports succeed.'
- What this solution (achieved 0.23872) has done: 'We fix the import-time protobuf crash that prevents the notebook from running by pinning protobuf to the pure-Python backend *before* any TensorFlow/protobuf-dependent imports and by avoiding any optional imports that trigger the `MessageFactory.GetPrototype` path. This change is runtime/stability-only and won’t alter your model, training loop, or prediction logic, so it should be score-neutral while unblocking execution. We also add a small environment sanity check and ensure the submission is always written as `submission.csv` with the exact required columns and row alignment to `sample_submission.csv`. No architecture, loss, epochs, batch size, or data filtering logic is changed.'
- What this solution (achieved 0.24253) has done: 'The crash happens before training because TensorFlow’s protobuf stack is still hitting the incompatible `MessageFactory.GetPrototype` path in this Kaggle image, even with the env vars you set. I fix this by forcing the pure-Python protobuf backend *and* importing `google.protobuf` early (before TensorFlow) so the backend selection is applied deterministically, which is runtime-only and score-neutral. I also add a small safety fallback: if the crash still occurs, we hard-disable TensorFlow and write a valid baseline `submission.csv` from `sample_submission.csv` (zeros) so note­books always complete with a valid CSV. Everything else (data processing, model, loss, training loop, and submission alignment) is unchanged.'
- What this solution (achieved 0.23892) has done: 'The notebook is failing before training because TensorFlow’s import chain hits a protobuf API mismatch (`MessageFactory.GetPrototype`), so the main fix is to patch protobuf compatibility *before* importing TensorFlow. I keep your model, loss, training loop, and submission-building logic unchanged, and only add a small, targeted monkey-patch for `google.protobuf.message_factory.MessageFactory.GetPrototype` to map to `GetMessageClass` when missing. This unblocks TensorFlow import in the given environment, allowing the model to train and produce predictions rather than falling back to zeros, which should move the score back toward your target band. The script still writes `submission.csv` with the exact required columns aligned to `sample_submission.csv`.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_DISABLE_CXX"] = "1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import json
import pandas as pd
import numpy as np

try:
    import google.protobuf  # noqa: F401
    from google.protobuf import message_factory as _message_factory

    if hasattr(_message_factory, "MessageFactory") and not hasattr(
        _message_factory.MessageFactory, "GetPrototype"
    ):

        def _GetPrototype(self, descriptor):
            return self.GetMessageClass(descriptor)

        _message_factory.MessageFactory.GetPrototype = _GetPrototype
except Exception as e:
    print("Warning: protobuf patch failed:", repr(e))

try:
    from google.protobuf.internal import api_implementation as _api_impl

    try:
        _api_impl._SetImplementationType("python")
    except Exception:
        pass
except Exception:
    pass

TF_AVAILABLE = True
try:
    import tensorflow as tf
    import tensorflow.keras.layers as L
except Exception as e:
    TF_AVAILABLE = False
    TF_IMPORT_ERROR = e

from sklearn.model_selection import train_test_split

print("TF available:", TF_AVAILABLE)
if TF_AVAILABLE:
    print("TF version:", tf.__version__)
print(
    "Protobuf implementation env:",
    os.environ.get("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"),
)



## === cell 1
data_dir_candidates = [
    "/kaggle/input/stanford-covid-vaccine/",
    "/kaggle/input/",
]
data_dir = None
for cand in data_dir_candidates:
    if os.path.exists(os.path.join(cand, "train.json")) and os.path.exists(
        os.path.join(cand, "test.json")
    ):
        data_dir = cand
        break
if data_dir is None:
    raise FileNotFoundError(
        "Could not find train.json/test.json in expected Kaggle input locations."
    )

train = pd.read_json(os.path.join(data_dir, "train.json"), lines=True)
test = pd.read_json(os.path.join(data_dir, "test.json"), lines=True)
sample_df = pd.read_csv(os.path.join(data_dir, "sample_submission.csv"))

print("Using data_dir:", data_dir)
print(
    "train shape:",
    train.shape,
    "test shape:",
    test.shape,
    "sample_df shape:",
    sample_df.shape,
)



## === cell 2
if not TF_AVAILABLE:
    print(
        "TensorFlow failed to import; writing baseline submission from sample_submission.csv."
    )
    print("TF import error:", repr(TF_IMPORT_ERROR))
    submission = sample_df.copy()
    submission.to_csv("submission.csv", index=False)
    print("Wrote submission.csv with shape:", submission.shape)
    raise SystemExit(0)



## === cell 3
tf.random.set_seed(2020)
np.random.seed(2020)



## === cell 4
pred_cols = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C", "deg_pH10", "deg_50C"]



## === cell 5
y_true = tf.random.normal((32, 68, 3))
y_pred = tf.random.normal((32, 68, 3))




## === cell 6
def MCRMSE(y_true, y_pred):
    """
    MCRMSE as used by the competition (mean over targets of RMSE), computed on the
    per-position dimension (axis=1). Return a scalar mean across the batch for stability.
    """
    colwise_mse = tf.reduce_mean(tf.square(y_true - y_pred), axis=1)  # (B, n_targets)
    per_sample = tf.reduce_mean(tf.sqrt(colwise_mse), axis=1)  # (B,)
    return tf.reduce_mean(per_sample)  # scalar




## === cell 7
def gru_layer(hidden_dim, dropout):
    return L.Bidirectional(
        L.GRU(
            hidden_dim,
            dropout=dropout,
            return_sequences=True,
            kernel_initializer="orthogonal",
        )
    )




## === cell 8
def build_model(
    embed_size,
    seq_len=107,
    pred_len=68,
    dropout=0.5,
    sp_dropout=0.2,
    embed_dim=200,
    hidden_dim=256,
    n_layers=3,
):
    inputs = L.Input(shape=(seq_len, 3), dtype=tf.int32)

    embed = L.Embedding(input_dim=embed_size, output_dim=embed_dim)(
        inputs
    )  # (B, seq_len, 3, embed_dim)

    reshaped = L.Reshape((seq_len, 3 * embed_dim))(embed)  # (B, seq_len, 3*embed_dim)

    hidden = L.SpatialDropout1D(sp_dropout)(reshaped)

    for _ in range(n_layers):
        hidden = gru_layer(hidden_dim, dropout)(hidden)

    truncated = hidden[:, :pred_len]
    out = L.Dense(5, activation="linear")(truncated)

    model = tf.keras.Model(inputs=inputs, outputs=out)
    model.compile(tf.optimizers.Adam(), loss=MCRMSE)
    return model




## === cell 9
def pandas_list_to_array(df):
    """
    Input: dataframe of shape (x, y), containing list of length l
    Return: np.array of shape (x, l, y)
    """
    return np.transpose(np.array(df.values.tolist()), (0, 2, 1))




## === cell 10
def preprocess_inputs(
    df, token2int, cols=["sequence", "structure", "predicted_loop_type"]
):
    def encode(seq):
        return [token2int.get(x, 0) for x in seq]

    arr = pandas_list_to_array(df[cols].applymap(encode))
    return arr.astype(np.int32)




## === cell 11
train = train.query("signal_to_noise >= 1").reset_index(drop=True)



## === cell 12
token2int = {x: i for i, x in enumerate("().ACGUBEHIMSX")}

train_inputs = preprocess_inputs(train, token2int)
train_labels = pandas_list_to_array(train[pred_cols]).astype(np.float32)



## === cell 13
x_train, x_val, y_train, y_val = train_test_split(
    train_inputs, train_labels, test_size=0.1, random_state=34, stratify=train.SN_filter
)



## === cell 14
test_df = test.query("seq_length == 107").reset_index(drop=True)
test_inputs = preprocess_inputs(test_df, token2int)



## === cell 15
model = build_model(embed_size=len(token2int), seq_len=107, pred_len=68)
model.summary()



## === cell 16
weights_path = "model.weights.h5"
history = model.fit(
    x_train,
    y_train,
    validation_data=(x_val, y_val),
    batch_size=64,
    epochs=75,
    verbose=2,
    callbacks=[
        tf.keras.callbacks.ReduceLROnPlateau(patience=5),
        tf.keras.callbacks.ModelCheckpoint(
            weights_path,
            save_weights_only=True,
            monitor="val_loss",
            mode="min",
            save_best_only=True,
        ),
    ],
)



## === cell 17
model_full = build_model(seq_len=107, pred_len=107, embed_size=len(token2int))
model_full.load_weights(weights_path)



## === cell 18
test_preds = model_full.predict(
    test_inputs, batch_size=128, verbose=1
)  # (n_test, 107, 5)



## === cell 19
preds_ls = []
for i, uid in enumerate(test_df.id.values):
    single_pred = test_preds[i]  # (107, 5)
    single_df = pd.DataFrame(single_pred, columns=pred_cols)
    single_df["id_seqpos"] = [f"{uid}_{x}" for x in range(single_df.shape[0])]
    preds_ls.append(single_df)

preds_df = pd.concat(preds_ls, ignore_index=True)

submission = sample_df[["id_seqpos"]].merge(preds_df, on="id_seqpos", how="left")

for c in pred_cols:
    if c not in submission.columns:
        submission[c] = 0.0
submission[pred_cols] = submission[pred_cols].fillna(0.0)

submission = submission[["id_seqpos"] + pred_cols]
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
print("Nulls per column:\n", submission.isna().sum())
