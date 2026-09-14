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

0.55896

# 6. Current score

0.47555

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 0.47548) has done: 'I remove the incompatible `transformers` imports that trigger the protobuf `MessageFactory.GetPrototype` error and replace the BERT-based embedding inside `build_model()` with a small trainable token `Embedding` applied to the same integer-encoded inputs; this keeps the overall approach (tokenize 3 strings → concatenate embeddings → pool → dense per position) intact while making it runnable in TF/Keras. I also fix ordering/NameError issues (e.g., `token2int` used before defined, stray `init_checkpoint`) and stabilize feature-engineering divisions by zero. Finally, I simplify inference to the actual test shape (all are length 107 here), generate predictions for all 107 positions (as required by the submission), and ensure the submission CSV matches `sample_submission.csv` exactly.'
- What this solution (achieved 0.47555) has done: 'I fix the immediate runtime crash coming from the protobuf/transformers `MessageFactory.GetPrototype` issue by ensuring we do not import `transformers` (or any TF-Hub text stack that pulls it in indirectly) and by pinning TensorFlow to a safe protobuf implementation path at runtime. I also make Keras graph-safe tensor slicing in `build_model()` by replacing Python-style slicing on KerasTensors with `Lambda` layers, which avoids backend-dependent errors. Finally, I keep the model/training logic intact but align training to the competition metric by training only the 3 scored targets and copying them into the 5-column submission (unscored columns duplicated), which is a minimal semantic change that typically improves public LB MCRMSE when unscored targets would otherwise inject noise.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.layers as L

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("TF:", tf.__version__)
print("NP:", np.__version__)
print("PD:", pd.__version__)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
pred_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

scored_cols = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]

subcol_to_scored_idx = {
    "reactivity": 0,
    "deg_Mg_pH10": 1,
    "deg_Mg_50C": 2,
}



## === cell 2
token2int = {x: i for i, x in enumerate("().ACGUBEHIMSX")}
VOCAB_SIZE = len(token2int)


def preprocess_inputs(df, cols=("sequence", "structure", "predicted_loop_type")):
    """
    Returns int array of shape (n, seq_len, 3).
    """
    arr = (
        df.loc[:, list(cols)]
        .applymap(lambda seq: [token2int[x] for x in seq])
        .values.tolist()
    )
    arr = np.array(arr)  # (n, 3, seq_len)
    arr = np.transpose(arr, (0, 2, 1))  # (n, seq_len, 3)
    return arr.astype(np.int32)




## === cell 3
def build_model(
    seq_len=107, pred_len=68, dropout=0.5, embed_dim=64, hidden_dim=128, out_dim=3
):
    """
    Core logic preserved: integer tokens -> Embedding for each of 3 channels -> concat ->
    permute -> pooling -> upsample -> per-position dense outputs.
    Bugfix: KerasTensor slicing via Lambda to be graph-safe across TF/Keras versions.
    """
    ids = L.Input(shape=(seq_len, 3), dtype=tf.int32)  # (B, L, 3)

    tok = L.Embedding(VOCAB_SIZE, embed_dim, name="tok_embed")

    ch0 = L.Lambda(lambda x: x[:, :, 0])(ids)
    ch1 = L.Lambda(lambda x: x[:, :, 1])(ids)
    ch2 = L.Lambda(lambda x: x[:, :, 2])(ids)

    e0 = tok(ch0)  # (B, L, D)
    e1 = tok(ch1)
    e2 = tok(ch2)

    x = L.Concatenate(axis=-1)([e0, e1, e2])  # (B, L, 3D)

    x = L.Permute((2, 1))(x)  # (B, 3D, L)
    x = L.AveragePooling1D(pool_size=16)(x)  # (B, 3D, floor(L/16))
    x = L.Permute((2, 1))(x)  # (B, floor(L/16), 3D)
    x = L.UpSampling1D(size=16)(x)  # approx back to L

    x = L.Lambda(lambda t: t[:, :pred_len, :])(x)

    out = L.Dense(out_dim, activation="linear")(x)

    model = tf.keras.Model(inputs=ids, outputs=out)
    model.compile(tf.keras.optimizers.Adam(), loss="mse")
    return model




## === cell 4
train = pd.read_json("/kaggle/input/stanford-covid-vaccine/train.json", lines=True)
test = pd.read_json("/kaggle/input/stanford-covid-vaccine/test.json", lines=True)
sample_df = pd.read_csv("/kaggle/input/stanford-covid-vaccine/sample_submission.csv")

print("train:", train.shape, "test:", test.shape, "sample:", sample_df.shape)
print("train seq_length unique:", sorted(train["seq_length"].unique())[:5], "...")
print("train seq_scored unique:", sorted(train["seq_scored"].unique())[:5], "...")



## === cell 5
train_inputs = preprocess_inputs(train)
train_labels = (
    np.array(train[scored_cols].values.tolist()).transpose((0, 2, 1)).astype(np.float32)
)

print("train_inputs:", train_inputs.shape, train_inputs.dtype)
print("train_labels:", train_labels.shape, train_labels.dtype)




## === cell 6
def safe_mean_position(seq, ch):
    idx = [i for i, c in enumerate(seq) if c == ch]
    if len(idx) == 0:
        return 0.0
    return float(np.mean(idx))


for df in [train, test]:
    df["Paired"] = [sum((c == "(") or (c == ")") for c in s) for s in df["structure"]]
    df["Unpaired"] = [sum((c == ".") for c in s) for s in df["structure"]]

    for col in ["E", "S", "H", "I", "G", "A", "U"]:
        if col in ["E", "S", "H", "I"]:
            df[col] = [
                sum(c == col for c in s) / len(s) for s in df["predicted_loop_type"]
            ]
        else:
            df[col] = [sum(c == col for c in s) / len(s) for s in df["sequence"]]

for a in ["G", "A", "C", "U"]:
    train[a + "_position"] = [safe_mean_position(s, a) for s in train["sequence"]]
    test[a + "_position"] = [safe_mean_position(s, a) for s in test["sequence"]]

for a in ["E", "S", "H"]:
    train[a + "_position"] = [
        safe_mean_position(s, a) for s in train["predicted_loop_type"]
    ]
    test[a + "_position"] = [
        safe_mean_position(s, a) for s in test["predicted_loop_type"]
    ]



## === cell 7
target_columns = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
target_columns += ["SN_filter", "signal_to_noise"]
target_columns += [
    "deg_error_pH10",
    "deg_error_Mg_50C",
    "deg_error_50C",
    "reactivity_error",
    "deg_error_Mg_pH10",
]
train_fe = train.drop(columns=target_columns, errors="ignore").copy()



## === cell 8
model = build_model(seq_len=107, pred_len=68, out_dim=len(scored_cols))
model.summary()



## === cell 9
history = model.fit(
    train_inputs,
    train_labels,
    batch_size=64,
    epochs=100,
    validation_split=0.05,
    callbacks=[tf.keras.callbacks.ReduceLROnPlateau()],
    verbose=2,
)



## === cell 10
model.save_weights("model.weights.h5")



## === cell 11
test_inputs = preprocess_inputs(test)
preds_68 = model.predict(test_inputs, batch_size=64, verbose=0)  # (n_test, 68, 3)

print("test_inputs:", test_inputs.shape)
print("preds_68:", preds_68.shape)



## === cell 12
n_test = preds_68.shape[0]
full_len = int(test["seq_length"].iloc[0])
assert full_len == 107, f"Unexpected test seq_length={full_len}; update code if needed."

preds_107_scored = np.zeros((n_test, full_len, len(scored_cols)), dtype=np.float32)
preds_107_scored[:, :68, :] = preds_68
preds_107_scored[:, 68:, :] = preds_68[:, -1:, :]  # repeat last pos

print("preds_107_scored:", preds_107_scored.shape)



## === cell 13
preds_ls = []
for i, uid in enumerate(test.id.values):
    p3 = preds_107_scored[i]  # (107, 3)
    df_pred = pd.DataFrame(
        {
            "reactivity": p3[:, subcol_to_scored_idx["reactivity"]],
            "deg_Mg_pH10": p3[:, subcol_to_scored_idx["deg_Mg_pH10"]],
            "deg_pH10": p3[:, subcol_to_scored_idx["deg_Mg_pH10"]],
            "deg_Mg_50C": p3[:, subcol_to_scored_idx["deg_Mg_50C"]],
            "deg_50C": p3[:, subcol_to_scored_idx["deg_Mg_50C"]],
        }
    )
    df_pred["id_seqpos"] = [f"{uid}_{x}" for x in range(full_len)]
    preds_ls.append(df_pred)

preds_df = pd.concat(preds_ls, axis=0, ignore_index=True)

submission = sample_df[["id_seqpos"]].merge(preds_df, on="id_seqpos", how="left")
for c in pred_cols:
    submission[c] = submission[c].astype(np.float32).fillna(0.0)

submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
print("Columns:", submission.columns.tolist())
