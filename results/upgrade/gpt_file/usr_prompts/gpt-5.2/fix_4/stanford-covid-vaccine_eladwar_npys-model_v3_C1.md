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
joblib==1.5.2
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

0.54125

# 6. Current score

0.30153

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 0.30896) has done: 'I first fix the environment/import crash by removing the unused `ResNet50` import that triggers a protobuf/TensorFlow incompatibility in this Kaggle image. Then I fix the missing BPPS `.npy` path issue by falling back to a pure-sequence/structure/loop-type one-hot feature tensor (same overall training flow: build arrays → fit Keras model → predict → write submission). I also fix the `Invalid dtype: object` issue by ensuring all model inputs/targets are `float32` numpy arrays with consistent shapes. Finally, I ensure the submission is exactly aligned to `sample_submission.csv` and always writes a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.30292) has done: 'I fix the TensorFlow/protobuf crash that happens at import time by forcing the pure-Python protobuf implementation before importing TensorFlow (a common Kaggle workaround for the `MessageFactory.GetPrototype` error). Then I keep the same data pipeline/model/training loop, but ensure the labels are built robustly as float32 arrays of shape `(n_samples, 68, 5)` to avoid any subtle object-dtype issues. Finally, I keep the submission creation logic identical while adding a strict alignment check to guarantee every `id_seqpos` is present and the output is a valid `submission.csv`. These changes are score-neutral (or negligible) and primarily restore end-to-end execution.'
- What this solution (achieved 0.30153) has done: 'We fix the import-time crash coming from the protobuf/TensorFlow mismatch by pinning protobuf to the pure-Python implementation *and* forcing the safe “python” backend before TensorFlow loads, plus disabling C++ descriptors that trigger `MessageFactory.GetPrototype` errors in some Kaggle images. Then we keep your exact data pipeline/model/training logic, only adding a safe fallback that retries the TensorFlow import with XLA disabled if the first import still fails (score-neutral but unblocks execution). Finally, we keep the submission-building logic the same while ensuring IDs are constructed from the full `id` (not `id_hash`) to avoid any accidental formatting mismatch and guarantee perfect alignment to `sample_submission.csv` (this is correctness-focused and should not worsen score).'

# 9. Code solution

## === cell 0
import os
import gc
import numpy as np
import pandas as pd

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_DISABLE_CPP_DESCRIPTORS", "1")

try:
    import tensorflow as tf
    import tensorflow.keras.layers as L
except Exception as e:
    os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"
    import tensorflow as tf
    import tensorflow.keras.layers as L

from joblib import Parallel, delayed  # kept as-is (unused but harmless)

np.random.seed(42)
tf.random.set_seed(42)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
try:
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except Exception:
    pass

gc.collect()



## === cell 2
BASE_INPUT = "/kaggle/input/stanford-covid-vaccine"
if not os.path.exists(BASE_INPUT):
    BASE_INPUT = "../input/stanford-covid-vaccine"

train_path = os.path.join(BASE_INPUT, "train.json")
test_path = os.path.join(BASE_INPUT, "test.json")
sample_path = os.path.join(BASE_INPUT, "sample_submission.csv")

train_df = pd.read_json(train_path, lines=True)
test_df = pd.read_json(test_path, lines=True)
sample_df = pd.read_csv(sample_path)

train_df["id_hash"] = train_df["id"].apply(lambda x: x.split("_")[1])
test_df["id_hash"] = test_df["id"].apply(lambda x: x.split("_")[1])

train_df.shape, test_df.shape, sample_df.shape




## === cell 3
def build_feature_tensor(df: pd.DataFrame) -> np.ndarray:
    """
    Returns X with shape (n_samples, 107, n_features) float32.
    Features: one-hot for sequence (4), structure (3), loop_type (7) => 14 dims.
    """
    seq_map = {"A": 0, "C": 1, "G": 2, "U": 3}
    struct_map = {"(": 0, ")": 1, ".": 2}
    loop_map = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}

    n = len(df)
    LSEQ = int(df["seq_length"].iloc[0])
    assert LSEQ == 107, f"Unexpected seq_length={LSEQ}; expected 107."
    nfeat = 4 + 3 + 7

    X = np.zeros((n, LSEQ, nfeat), dtype=np.float32)

    for i, (seq, struct, loop) in enumerate(
        zip(df["sequence"], df["structure"], df["predicted_loop_type"])
    ):
        for t in range(LSEQ):
            b = seq_map.get(seq[t], None)
            if b is not None:
                X[i, t, b] = 1.0
            s = struct_map.get(struct[t], None)
            if s is not None:
                X[i, t, 4 + s] = 1.0
            lp = loop_map.get(loop[t], None)
            if lp is not None:
                X[i, t, 7 + lp] = 1.0

    return X


target_columns = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

y_list = []
for _, r in train_df.iterrows():
    y_list.append(
        [np.asarray(r[c], dtype=np.float32) for c in target_columns]
    )  # 5 x 68
y = np.stack(y_list, axis=0)  # (n, 5, 68)
y = np.transpose(y, (0, 2, 1)).astype(np.float32)  # (n, 68, 5)

X_train = build_feature_tensor(train_df).astype(np.float32)
X_test = build_feature_tensor(test_df).astype(np.float32)

X_train.shape, y.shape, X_test.shape




## === cell 4
def build_model(seq_len=107, pred_len=68, n_features=14, hidden_dim=128, dropout=0.2):
    inp = L.Input(shape=(seq_len, n_features), dtype=tf.float32)
    x = L.SpatialDropout1D(dropout)(inp)
    x = L.Bidirectional(L.LSTM(hidden_dim, return_sequences=True))(x)
    x = L.Bidirectional(L.LSTM(hidden_dim // 2, return_sequences=True))(x)
    x = x[:, :pred_len, :]
    out = L.Dense(5, activation="linear")(x)
    model = tf.keras.Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss="mse")
    return model


try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

model = build_model(seq_len=107, pred_len=68, n_features=X_train.shape[-1])
model.summary()



## === cell 5
ckpt_path = "model.weights.h5"

callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=3, verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        ckpt_path,
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True,
        verbose=1,
    ),
]

history = model.fit(
    X_train,
    y,
    batch_size=64,
    epochs=100,
    validation_split=0.05,
    callbacks=callbacks,
    verbose=2,
)

if os.path.exists(ckpt_path):
    model.load_weights(ckpt_path)



## === cell 6
test_pred_68 = model.predict(X_test, batch_size=64, verbose=0).astype(
    np.float32
)  # (n_test, 68, 5)

n_test = test_pred_68.shape[0]
full_len = 107
pred_len = test_pred_68.shape[1]

test_pred_full = np.zeros((n_test, full_len, 5), dtype=np.float32)
test_pred_full[:, :pred_len, :] = test_pred_68

last_vals = test_pred_68[:, -1:, :]  # (n_test, 1, 5)
test_pred_full[:, pred_len:, :] = np.repeat(last_vals, full_len - pred_len, axis=1)

test_pred_full.shape



## === cell 7
pred_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

rows = []
for i, uid in enumerate(test_df["id"].tolist()):
    df_i = pd.DataFrame(test_pred_full[i], columns=pred_cols)
    df_i["id_seqpos"] = [f"{uid}_{pos}" for pos in range(full_len)]
    rows.append(df_i)

preds_df = pd.concat(rows, axis=0, ignore_index=True)

submission = sample_df[["id_seqpos"]].merge(preds_df, on="id_seqpos", how="left")
for c in pred_cols:
    submission[c] = submission[c].astype(np.float32).fillna(0.0)

if submission[pred_cols].isna().any().any():
    raise RuntimeError("Submission contains NaNs after merge/fill; alignment failed.")
if len(submission) != len(sample_df):
    raise RuntimeError(
        f"Submission row count mismatch: {len(submission)} vs {len(sample_df)}"
    )

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)

submission.head(), submission.shape, submission_path
