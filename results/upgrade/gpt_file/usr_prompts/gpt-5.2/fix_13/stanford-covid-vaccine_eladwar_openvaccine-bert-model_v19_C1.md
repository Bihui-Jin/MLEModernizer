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

0.56098

# 6. Current score

0.24393

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 0.2432) has done: 'I remove the incompatible `transformers` BERT dependency that’s currently crashing imports and model building under TF/Keras 2.18, and replace it with a tiny embedding + BiGRU block that keeps the same input/output semantics (sequence-length → per-position 5 targets) so the pipeline runs end-to-end. I also fix preprocessing to always return a proper `(n, seq_len, 3)` int32 tensor, handle the fact that this dataset has no `seq_length==130` rows (so private_df is empty), and ensure we still generate predictions for all 107 positions required by the sample submission. Finally, I make the fold training loop actually use the KFold indices (instead of `validation_split`) and generate a valid `submission.csv` with the exact required columns and row order.'
- What this solution (achieved 0.2439) has done: 'I fix the runtime import crash caused by an incompatible protobuf version by forcing the pure-Python protobuf implementation before TensorFlow is imported, which avoids the `MessageFactory.GetPrototype` AttributeError in this Kaggle image. Then I make the embedding vocabulary 1-indexed (and increase `vocab_size`) so token id 0 is reserved/unused rather than a real symbol; this is a minimal, score-positive change that improves representation without changing the model architecture or training loop. Finally, I keep all I/O paths and the submission-building logic the same, ensuring a valid `submission.csv` is always written with the required columns and row order.'
- What this solution (achieved 0.24323) has done: 'I fix the protobuf/TensorFlow import crash by forcing the pure-Python protobuf implementation *before the protobuf runtime is loaded* and by clearing any already-imported `google.protobuf` modules, which is the root cause of the `MessageFactory.GetPrototype` error in this Kaggle image. I keep the model and training loop identical, only making the environment/import sequence robust so the notebook runs end-to-end. I also add a small safety fallback to read data from either `/kaggle/input/stanford-covid-vaccine/` or `/kaggle/input/` in case the dataset is mounted in the alternative path, without changing filenames or submission format. This should be score-neutral (your current score is already better than the target for a lower-is-better metric), focusing strictly on correctness and stability.'
- What this solution (achieved 0.24295) has done: 'I fix the runtime import crash coming from the protobuf/TensorFlow incompatibility by forcing the pure-Python protobuf implementation before *any* protobuf-related imports occur, and by also setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` at the very top to make it effective in this Kaggle image. I keep the model, preprocessing, CV loop, and submission-building logic the same to avoid score-changing changes (your current score is already better than the target for a lower-is-better metric). I also add a small, safe fallback to skip TPU discovery if TF fails to initialize TPU in this environment, without changing training behavior on CPU/GPU. The script run end-to-end and always write a valid `submission.csv` with the exact required columns and row order.'
- What this solution (achieved 0.24305) has done: 'I fix the TensorFlow import crash caused by the protobuf runtime mismatch by forcing the pure-Python protobuf implementation and disabling the C++ implementation *before any TensorFlow/protobuf code loads*, plus clearing already-imported protobuf modules to make the setting effective. This is a correctness/stability fix only and should not meaningfully change model behavior or score. I also keep all paths, model/training loop, and submission-building logic the same, ensuring a valid `submission.csv` is always written with the required columns and row order.'
- What this solution (achieved 0.24413) has done: 'I fix the TensorFlow import crash caused by the protobuf runtime mismatch by forcing the pure-Python protobuf implementation earlier and more completely (including setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` and removing already-imported protobuf modules) before TensorFlow is imported. This is a stability/correctness change only and should not materially change model behavior or score (your current score is already better than the target for a lower-is-better metric, so we avoid score-improving edits). I also make the TPU detection safer so a TPU init failure can’t stop execution, while keeping the same training loop and model. The rest of the pipeline (data loading, preprocessing, CV training, prediction expansion to 107, and submission formatting) remains unchanged and still write a valid `submission.csv`.'
- What this solution (achieved 0.24348) has done: 'I fix the TensorFlow import crash caused by a protobuf runtime mismatch by enforcing the pure-Python protobuf implementation *and* preventing the C++ backend from loading before any protobuf/TensorFlow import occurs. To make that enforcement actually take effect in this Kaggle image, I also clear any already-imported `google.protobuf` modules and import `google.protobuf` once after setting the env vars (so it binds to the python implementation) before importing TensorFlow. The rest of your pipeline (model, preprocessing, CV loop, prediction expansion, and submission formatting) is kept identical to avoid unintended score changes (your current score is already better than the target for a lower-is-better metric). The script run end-to-end and always write a valid `submission.csv` with the required columns and row order.'
- What this solution (achieved 0.2434) has done: 'I fix the immediate runtime crash caused by a protobuf API mismatch by forcing the legacy pure-Python protobuf implementation version 2, and doing it before any `google.protobuf`/TensorFlow import (the current env var combination still allows the newer runtime that lacks `MessageFactory.GetPrototype`). This is a stability-only change: it keeps your model, preprocessing, CV training loop, and submission-building logic identical, so it should not intentionally move the score (your current score is already better than the target for a lower-is-better metric). I also add a small import-time sanity print to confirm which protobuf backend/version is active, without altering training. The script then run end-to-end and always write a valid `submission.csv` with the exact required columns and row order.'
- What this solution (achieved 0.244) has done: 'I fix the immediate runtime crash caused by the protobuf/TensorFlow incompatibility by forcing TensorFlow to use the pure-Python protobuf runtime before importing TensorFlow, and by falling back to a safe “protobuf<4” re-install inside the Kaggle kernel if the import still fails. This is a stability-only change and keeps your model, preprocessing, CV loop, and submission formatting identical (so it should not intentionally move your score, which is already better than the target for a lower-is-better metric). I also keep the same input paths and ensure the script always reaches the final cell that writes `submission.csv`. No changes are made to architecture, loss, or training hyperparameters.'
- What this solution (achieved 0.2443) has done: 'I fix the TensorFlow import crash by enforcing the pure-Python protobuf backend *before* anything from `google.protobuf` is imported, and by also disabling the C++ backend explicitly; this resolves the `MessageFactory.GetPrototype` issue without touching your model/training logic. I remove the in-notebook `pip install protobuf<4` fallback (it can’t reliably fix the already-imported-runtime situation in Kaggle and may break the environment), keeping the rest of the pipeline identical. I keep all paths, preprocessing, KFold training, prediction expansion to 107, and submission formatting unchanged to avoid score-moving changes (your current score is already better than the target for a lower-is-better metric). The script run end-to-end and always write `submission.csv` with the required columns/order.'
- What this solution (achieved 0.2432) has done: 'I fix the immediate runtime crash (`MessageFactory.GetPrototype`) by removing the protobuf version forcing that is incompatible with this Kaggle image, and instead reliably forcing the pure-Python protobuf backend before TensorFlow imports (the key part to avoid the C++ implementation mismatch). I also add a safe fallback to import TensorFlow in a clean process state by clearing any preloaded protobuf modules, and I keep your model, preprocessing, CV loop, hyperparameters, and submission formatting identical to avoid intentionally moving your score (your current 0.2443 is already better than the 0.56098 target for a lower-is-better metric). Finally, I keep the same I/O paths and ensure `submission.csv` is always written with the exact required columns/order.'
- What this solution (achieved 0.24393) has done: 'I fix the immediate runtime crash (`MessageFactory.GetPrototype`) by ensuring the protobuf runtime is forced to the pure-Python implementation *before* TensorFlow (and any protobuf internals) can load, and by avoiding importing protobuf internals at all. This is a stability-only change and should not intentionally move your score (your current 0.2432 is already better than the 0.56098 target for a lower-is-better metric), while keeping the model, preprocessing, CV loop, and submission formatting identical. I also keep the same input path fallback logic and ensure we always reach the final cell that writes a valid `submission.csv` with the exact required columns/order.'

# 9. Code solution

## === cell 0
import os
import sys

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_DISABLE_CPP"] = "1"

for m in list(sys.modules.keys()):
    if m.startswith("google.protobuf") or m.startswith("tensorflow"):
        sys.modules.pop(m, None)

import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow.keras.layers as L
from sklearn.model_selection import KFold

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("TF version:", tf.__version__)
print(
    "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION:",
    os.environ.get("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"),
)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
AUTO = tf.data.experimental.AUTOTUNE

try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print("Running on TPU ", tpu.master())
except Exception:
    tpu = None

if tpu:
    try:
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.experimental.TPUStrategy(tpu)
    except Exception:
        strategy = tf.distribute.get_strategy()
else:
    strategy = tf.distribute.get_strategy()

print("REPLICAS: ", strategy.num_replicas_in_sync)




## === cell 2
pred_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]




## === cell 3
def MCRMSE(y_true, y_pred):
    colwise_mse = tf.reduce_mean(tf.square(y_true - y_pred), axis=1)  # (batch, 5)
    return tf.reduce_mean(tf.sqrt(colwise_mse + 1e-9), axis=1)  # (batch,)




## === cell 4
def build_model(
    seq_len=107, pred_len=68, dropout=0.5, embed_dim=64, hidden_dim=128, vocab_size=32
):
    ids = L.Input(shape=(seq_len, 3), dtype=tf.int32)

    x_seq = L.Embedding(input_dim=vocab_size, output_dim=embed_dim, mask_zero=False)(
        ids[:, :, 0]
    )
    x_str = L.Embedding(input_dim=vocab_size, output_dim=embed_dim, mask_zero=False)(
        ids[:, :, 1]
    )
    x_loop = L.Embedding(input_dim=vocab_size, output_dim=embed_dim, mask_zero=False)(
        ids[:, :, 2]
    )
    x = L.Concatenate(axis=-1)([x_seq, x_str, x_loop])

    x = L.Bidirectional(L.GRU(hidden_dim, dropout=dropout, return_sequences=True))(x)
    x = L.Bidirectional(L.GRU(hidden_dim, dropout=dropout, return_sequences=True))(x)

    x = x[:, :pred_len, :]
    out = L.Dense(5, activation="linear")(x)

    model = tf.keras.Model(inputs=ids, outputs=out)
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=MCRMSE)
    return model




## === cell 5
vocab = {
    "sequence": {x: i + 1 for i, x in enumerate("A C G U".split())},
    "structure": {x: i + 1 for i, x in enumerate("( . )".split())},
    "predicted_loop_type": {x: i + 1 for i, x in enumerate("B E H I M S X".split())},
}


def preprocess_inputs(df, cols=("sequence", "structure", "predicted_loop_type")):
    seqs = df[cols[0]].values
    structs = df[cols[1]].values
    loops = df[cols[2]].values

    n = len(df)
    seq_len = len(seqs[0]) if n > 0 else 0
    X = np.zeros((n, seq_len, 3), dtype=np.int32)

    for i in range(n):
        X[i, :, 0] = [vocab["sequence"][c] for c in seqs[i]]
        X[i, :, 1] = [vocab["structure"][c] for c in structs[i]]
        X[i, :, 2] = [vocab["predicted_loop_type"][c] for c in loops[i]]

    return X




## === cell 6
base_candidates = [
    "/kaggle/input/stanford-covid-vaccine",
    "/kaggle/input",
]

data_base = None
for b in base_candidates:
    if os.path.exists(os.path.join(b, "train.json")) and os.path.exists(
        os.path.join(b, "test.json")
    ):
        data_base = b
        break

if data_base is None:
    raise FileNotFoundError(
        "Could not find train.json/test.json in expected Kaggle input paths."
    )

train_path = os.path.join(data_base, "train.json")
test_path = os.path.join(data_base, "test.json")
sample_path = os.path.join(data_base, "sample_submission.csv")

train = pd.read_json(train_path, lines=True)
test = pd.read_json(test_path, lines=True)
sample_df = pd.read_csv(sample_path)

print("Using data_base:", data_base)
print(train.shape, test.shape, sample_df.shape)
print(sample_df.columns.tolist())




## === cell 7
train = train.query("signal_to_noise >= 4").copy()
if len(train) == 0:
    raise ValueError(
        "After filtering signal_to_noise >= 4, training set became empty. Lower the threshold."
    )

train_inputs = preprocess_inputs(train)
train_labels = (
    np.array(train[pred_cols].values.tolist()).transpose((0, 2, 1)).astype(np.float32)
)  # (n, 68, 5)

print("train_inputs:", train_inputs.shape, train_inputs.dtype)
print("train_labels:", train_labels.shape, train_labels.dtype)




## === cell 8
test_inputs = preprocess_inputs(test)
print("test_inputs:", test_inputs.shape)




## === cell 9
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

EPOCHS = 100
BATCH_SIZE = 64

test_preds_accum = None

with strategy.scope():
    for fold, (idxT, idxV) in enumerate(kf.split(train_inputs), start=0):
        model = build_model(seq_len=107, pred_len=68, vocab_size=32)

        ckpt_path = f"model_fold{fold}.weights.h5"
        callbacks = [
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=5, verbose=0
            ),
            tf.keras.callbacks.ModelCheckpoint(
                ckpt_path,
                monitor="val_loss",
                save_weights_only=True,
                save_best_only=True,
                verbose=0,
            ),
        ]

        history = model.fit(
            train_inputs[idxT],
            train_labels[idxT],
            validation_data=(train_inputs[idxV], train_labels[idxV]),
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            callbacks=callbacks,
            verbose=0,
        )

        model.load_weights(ckpt_path)

        fold_pred_68 = model.predict(
            test_inputs, batch_size=BATCH_SIZE, verbose=0
        )  # (n_test, 68, 5)

        n_test, pred_len, n_targets = fold_pred_68.shape
        full_len = test_inputs.shape[1]
        fold_pred_full = np.zeros((n_test, full_len, n_targets), dtype=np.float32)
        fold_pred_full[:, :pred_len, :] = fold_pred_68
        if full_len > pred_len:
            fold_pred_full[:, pred_len:, :] = fold_pred_68[
                :, pred_len - 1 : pred_len, :
            ]

        if test_preds_accum is None:
            test_preds_accum = fold_pred_full / kf.get_n_splits()
        else:
            test_preds_accum += fold_pred_full / kf.get_n_splits()

print("test_preds_accum:", test_preds_accum.shape)




## === cell 10
preds_ls = []
for i, uid in enumerate(test.id.values):
    single_pred = test_preds_accum[i]  # (seq_len, 5)
    single_df = pd.DataFrame(single_pred, columns=pred_cols)
    single_df["id_seqpos"] = [f"{uid}_{x}" for x in range(single_df.shape[0])]
    preds_ls.append(single_df)

preds_df = pd.concat(preds_ls, axis=0, ignore_index=True)

submission = sample_df[["id_seqpos"]].merge(preds_df, on="id_seqpos", how="left")

for c in pred_cols:
    submission[c] = submission[c].astype(np.float32)
submission[pred_cols] = submission[pred_cols].fillna(0.0)

print(submission.shape)
print(submission.head())




## === cell 11
out_path = "submission.csv"
submission.to_csv(out_path, index=False)
print("Wrote:", out_path, "size:", os.path.getsize(out_path))
print("Columns:", submission.columns.tolist())
print("Any NaNs:", submission.isna().any().to_dict())
