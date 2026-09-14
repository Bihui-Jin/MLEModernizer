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

0.40838

# 6. Current score

0.29754

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.29211) has done: 'I fix the environment/import crash by removing the optional `plotly` dependency (it triggers a protobuf compatibility issue in this Kaggle image) and replacing the training-history plot with a lightweight pandas print. Then I fix the preprocessing/model-shape bugs: the model currently feeds a 3D float tensor into an `Embedding` layer and uses `tf.reshape` directly on a `KerasTensor`, both of which error in TF/Keras 2.18; I instead embed each of the 3 categorical channels separately and `Concatenate` them (same core GRU stack and loss). Finally, I fix the incorrect public/private split (this dataset is length 107 only) and ensure predictions are generated for full `seq_length` while keeping training targets at `seq_scored=68`, and write a valid `submission.csv` matching `sample_submission.csv` ordering.'
- What this solution (achieved 0.28803) has done: 'I fix the current runtime crash caused by a protobuf/TensorFlow incompatibility by pinning the protobuf Python implementation before importing TensorFlow, which is the minimal environment-level change needed to run. I also make the preprocessing robust to any unexpected characters by mapping unknown tokens to a safe default, preventing rare KeyErrors without changing the modeling approach. Finally, I keep the same model/training logic but ensure the checkpoint filename is compatible across Keras versions and always produces a valid `submission.csv` matching `sample_submission.csv` row order and required columns (this should be score-neutral and stabilize execution).'
- What this solution (achieved 0.29697) has done: 'I fix the TensorFlow import crash by switching the protobuf implementation env var from `cpp` (which fails in this image) to the safe `python` implementation before importing TensorFlow. Then I make the script run end-to-end by ensuring `tf`, `L`, `train_test_split`, and `pred_cols` are defined even if earlier cells fail, while keeping the same GRU model, preprocessing, training loop, and prediction flow. Finally, I ensure the model checkpoint filename is compatible with TF/Keras 2.18 and that predictions are expanded to the full 107 positions and merged in the exact `sample_submission.csv` row order, producing a valid `submission.csv`.'
- What this solution (achieved 0.29754) has done: 'I fix the current hard crash at TensorFlow import (`MessageFactory.GetPrototype`), which comes from an incompatible protobuf runtime in this Kaggle image, by forcing a safe protobuf version before importing TensorFlow (this is execution-unblocking and score-neutral). Then I keep your exact preprocessing, GRU architecture, training loop, and submission construction, but make the protobuf/TensorFlow initialization deterministic and robust so training/inference completes reliably. Finally, I ensure the submission is written as `submission.csv` with correct columns and exact `sample_submission.csv` row order as you already intended.'

# 9. Code solution

## === cell 0
import os

import sys
import subprocess

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

try:
    import google.protobuf  # noqa: F401
    from google.protobuf import __version__ as pb_ver
except Exception:
    pb_ver = None


def _version_tuple(v):
    try:
        return tuple(int(x) for x in v.split(".")[:3])
    except Exception:
        return (999, 999, 999)


need_pin = (pb_ver is None) or (_version_tuple(pb_ver) >= (5, 0, 0))
if need_pin:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "protobuf>=4.21.0,<5"]
    )

import json
import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow.keras.layers as L
from sklearn.model_selection import train_test_split

print("TF version:", tf.__version__)
try:
    from google.protobuf import __version__ as pb_ver2

    print("protobuf version:", pb_ver2)
except Exception as e:
    print("protobuf version: (unavailable)", repr(e))



## === cell 1
tf.random.set_seed(2020)
np.random.seed(2020)

pred_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]




## === cell 2
def gru_layer(hidden_dim, dropout):
    return L.Bidirectional(
        L.GRU(
            hidden_dim,
            dropout=dropout,
            return_sequences=True,
            kernel_initializer="orthogonal",
        )
    )




## === cell 3
def build_model(
    embed_size,
    seq_len=107,
    pred_len=68,
    dropout=0.5,
    sp_dropout=0.2,
    embed_dim=75,
    hidden_dim=128,
    n_layers=3,
):
    inputs = L.Input(shape=(seq_len, 3), dtype="int32")

    emb_layers = []
    for k in range(3):
        xk = inputs[:, :, k]  # (batch, seq_len)
        ek = L.Embedding(input_dim=embed_size, output_dim=embed_dim, name=f"embed_{k}")(
            xk
        )  # (batch, seq_len, embed_dim)
        emb_layers.append(ek)

    hidden = L.Concatenate(axis=-1)(emb_layers)  # (batch, seq_len, 3*embed_dim)
    hidden = L.SpatialDropout1D(sp_dropout)(hidden)

    for _ in range(n_layers):
        hidden = gru_layer(hidden_dim, dropout)(hidden)

    truncated = hidden[:, :pred_len]
    out = L.Dense(5, activation="linear")(truncated)

    model = tf.keras.Model(inputs=inputs, outputs=out)
    model.compile(tf.optimizers.Adam(), loss="mse")
    return model




## === cell 4
def pandas_list_to_array(df):
    """
    Input: dataframe of shape (n_samples, n_cols), where each cell is a list (length L).
    Return: np.array of shape (n_samples, L, n_cols)
    """
    arr = np.array(df.values.tolist())  # (n_samples, n_cols, L)
    return np.transpose(arr, (0, 2, 1))




## === cell 5
def preprocess_inputs(
    df, token2int, cols=("sequence", "structure", "predicted_loop_type")
):
    unk = token2int.get(".", 0)
    arr = pandas_list_to_array(
        df[list(cols)].applymap(lambda seq: [token2int.get(x, unk) for x in seq])
    ).astype(np.int32)
    return arr




## === cell 6
candidate_dirs = [
    "/kaggle/input/stanford-covid-vaccine/",
    "/kaggle/data/stanford-covid-vaccine/",
    "/kaggle/input/stanford-covid-vaccine/stanford-covid-vaccine/",
    "/kaggle/data/stanford-covid-vaccine/stanford-covid-vaccine/",
]
data_dir = None
for d in candidate_dirs:
    if os.path.exists(os.path.join(d, "train.json")):
        data_dir = d
        break
if data_dir is None:
    raise FileNotFoundError(
        "Could not find train.json in expected locations. Checked: "
        + ", ".join(candidate_dirs)
    )

train = pd.read_json(os.path.join(data_dir, "train.json"), lines=True)
test = pd.read_json(os.path.join(data_dir, "test.json"), lines=True)
sample_df = pd.read_csv(os.path.join(data_dir, "sample_submission.csv"))

print("Using data_dir:", data_dir)
print(train.shape, test.shape, sample_df.shape)
print("Unique test seq_length:", sorted(test["seq_length"].unique().tolist()))
print("Unique test seq_scored:", sorted(test["seq_scored"].unique().tolist()))



## === cell 7
token2int = {x: i for i, x in enumerate("().ACGUBEHIMSX")}

train_inputs = preprocess_inputs(train, token2int)
train_labels = pandas_list_to_array(train[pred_cols]).astype(np.float32)

print("train_inputs:", train_inputs.shape, train_inputs.dtype)
print("train_labels:", train_labels.shape, train_labels.dtype)



## === cell 8
x_train, x_val, y_train, y_val = train_test_split(
    train_inputs, train_labels, test_size=0.1, random_state=34
)

print("x_train/x_val:", x_train.shape, x_val.shape)
print("y_train/y_val:", y_train.shape, y_val.shape)



## === cell 9
test_df = test.copy()
test_inputs = preprocess_inputs(test_df, token2int)

seq_len = int(test_df["seq_length"].iloc[0])  # 107
pred_len_train = int(train["seq_scored"].iloc[0])  # 68
pred_len_test = int(test_df["seq_length"].iloc[0])  # 107

print(
    "Using seq_len:",
    seq_len,
    "pred_len_train:",
    pred_len_train,
    "pred_len_test:",
    pred_len_test,
)
print("test_inputs:", test_inputs.shape, test_inputs.dtype)



## === cell 10
model = build_model(embed_size=len(token2int), seq_len=seq_len, pred_len=pred_len_train)
model.summary()



## === cell 11
weights_path = "model.weights.h5"

history = model.fit(
    x_train,
    y_train,
    validation_data=(x_val, y_val),
    batch_size=64,
    epochs=70,
    verbose=2,
    callbacks=[
        tf.keras.callbacks.ReduceLROnPlateau(),
        tf.keras.callbacks.ModelCheckpoint(
            weights_path, save_best_only=True, save_weights_only=True
        ),
    ],
)



## === cell 12
hist_df = pd.DataFrame(history.history)
keep_cols = [c for c in ["loss", "val_loss"] if c in hist_df.columns]
print(hist_df.tail(5)[keep_cols])



## === cell 13
model_full = build_model(
    embed_size=len(token2int), seq_len=seq_len, pred_len=pred_len_test
)
model_full.load_weights(weights_path)



## === cell 14
test_preds = model_full.predict(
    test_inputs, batch_size=64, verbose=1
)  # (n_test, 107, 5)
print("test_preds:", test_preds.shape, test_preds.dtype)



## === cell 15
preds_ls = []
for i, uid in enumerate(test_df.id.values):
    single_pred = test_preds[i]  # (107, 5)
    single_df = pd.DataFrame(single_pred, columns=pred_cols)
    single_df["id_seqpos"] = [f"{uid}_{x}" for x in range(single_df.shape[0])]
    preds_ls.append(single_df)

preds_df = pd.concat(preds_ls, axis=0, ignore_index=True)
print(preds_df.head())
print("preds_df:", preds_df.shape)



## === cell 16
submission = sample_df[["id_seqpos"]].merge(preds_df, on="id_seqpos", how="left")

for c in pred_cols:
    if c not in submission.columns:
        submission[c] = 0.0
submission[pred_cols] = submission[pred_cols].fillna(0.0)

submission = submission[["id_seqpos"] + pred_cols]
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
print("Any NaNs:", submission.isna().any().to_dict())
print("Saved to:", os.path.abspath("submission.csv"))
assert os.path.exists("submission.csv") and os.path.getsize("submission.csv") > 0
