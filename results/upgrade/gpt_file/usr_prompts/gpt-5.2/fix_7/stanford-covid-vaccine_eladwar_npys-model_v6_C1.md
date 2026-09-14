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

0.54286

# 6. Current score

0.37989

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 0.38041) has done: 'I fix the environment/runtime issues first: remove the problematic `ResNet50` import that triggers the protobuf `MessageFactory` error, and replace notebook-only magics (`%%time`). Then I fix the missing `bpps/` dependency by adding a safe fallback feature generator that uses the available `sequence/structure/loop_type` to build a 130×130 numeric matrix per sample so the existing ConvLSTM model can run unchanged. Next, I ensure all arrays are proper `float32` (your current pipeline creates `object` arrays, causing the `Invalid dtype: object` error) and correct the model so the ConvLSTM output is actually used (currently it’s overwritten). Finally, I generate predictions for all test rows and write a valid `submission.csv` aligned to `sample_submission.csv` `id_seqpos` ordering.'
- What this solution (achieved 0.38147) has done: 'I fix the crash happening at import time by setting a compatible protobuf implementation *before* importing TensorFlow, which avoids the `MessageFactory.GetPrototype` AttributeError in this Kaggle environment. I also make checkpoint loading robust for Keras 2.18 by saving/loading the whole model (not just weights), since your checkpoint filename is `model.keras`. Finally, I keep the rest of your pipeline (fallback feature matrices, ConvLSTM model, training loop, and submission building) unchanged so the score behavior remains consistent while restoring end-to-end execution and a valid `submission.csv`.'
- What this solution (achieved 0.38055) has done: 'I fix the import-time protobuf/TensorFlow crash by forcing TensorFlow to use the C++ protobuf runtime (and removing the incompatible “python” implementation override) before importing TF. Then I make checkpointing/loading compatible with Keras 2.18 by saving/loading weights only (with a `.weights.h5` file) so `load_model()` doesn’t fail on serialization issues. Finally, I keep your existing data pipeline, fallback feature matrices, ConvLSTM model, training loop, and submission construction unchanged so the scoring behavior remains essentially the same while the notebook runs end-to-end and writes a valid `submission.csv`.'
- What this solution (achieved 0.37896) has done: 'We fix the import-time protobuf/TensorFlow crash by forcing TensorFlow to use the pure-Python protobuf implementation *before* importing TensorFlow, which is the most reliable workaround for the `MessageFactory.GetPrototype` incompatibility. We keep your data pipeline, fallback matrix generation, model definition, training loop, and submission construction unchanged to preserve evaluation semantics and keep score movement minimal. We also add a small defensive guard so the code doesn’t crash if the checkpoint file isn’t written for any reason (still producing a valid submission). Finally, we keep paths and output filename the same (`submission.csv`) and ensure the submission columns/dtypes match the sample submission.'
- What this solution (achieved 0.38356) has done: 'I fix the import-time TensorFlow/protobuf crash by setting protobuf-related environment variables in the safest way *before* importing TensorFlow, and by falling back automatically if the first choice still fails. I also make the training data parsing robust against any non-numeric/NaN values inside the target arrays (a common silent source of dtype/object issues) while keeping the same target definitions and shapes. Finally, I keep your model, training loop, and submission-building logic unchanged, ensuring `submission.csv` is always written in the exact sample-submission row order. These changes should restore end-to-end execution and, by preventing accidental bad target values from poisoning training, should move the score modestly toward the target without altering the core approach.'
- What this solution (achieved 0.37989) has done: 'I fix the TensorFlow import crash by setting protobuf environment variables in a way that is compatible with TF 2.18 (avoid forcing the pure-Python protobuf implementation, which triggers the `MessageFactory.GetPrototype` error here) and only falling back if needed. Then I keep the model/training/prediction logic unchanged, but ensure the script always reaches the submission-writing step and produces `submission.csv` with the exact required columns/order. These changes are runtime/stability fixes and should be score-neutral (your training/inference pipeline remains the same), while restoring end-to-end execution so you can iterate toward the target score.'

# 9. Code solution

## === cell 0
import os
import gc
import numpy as np
import pandas as pd

from joblib import Parallel, delayed
from sklearn.metrics import mean_absolute_error

SEED = 42
np.random.seed(SEED)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)


def _import_tf_safely():
    try:
        import tensorflow as tf  # noqa: F401

        return tf
    except Exception as e1:
        os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
        os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
        try:
            import importlib

            tf = importlib.import_module("tensorflow")
            return tf
        except Exception as e2:
            print("TensorFlow import failed with default protobuf runtime:", repr(e1))
            print(
                "TensorFlow import also failed with python protobuf runtime:", repr(e2)
            )
            raise


tf = _import_tf_safely()
import tensorflow.keras.layers as L

tf.random.set_seed(SEED)

BASE_INPUT = "/kaggle/input/stanford-covid-vaccine"
if not os.path.exists(BASE_INPUT):
    BASE_INPUT = "/kaggle/data/stanford-covid-vaccine"

print("Using BASE_INPUT:", BASE_INPUT)
print("TF version:", tf.__version__)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        print("Could not set memory growth:", repr(e))

gc.collect()



## === cell 2
BPPS_DIR = os.path.join(BASE_INPUT, "bpps")
has_bpps = os.path.isdir(BPPS_DIR)

print("BPPS directory exists:", has_bpps, BPPS_DIR)


def _safe_pad(s, n=130, pad_char="N"):
    s = "" if s is None else str(s)
    if len(s) >= n:
        return s[:n]
    return s + (pad_char * (n - len(s)))


_base_to_int = {"A": 1.0, "C": 2.0, "G": 3.0, "U": 4.0, "N": 0.0}
_struct_to_int = {"(": 1.0, ")": -1.0, ".": 0.0, "N": 0.0}
_loop_to_int = {
    "S": 1.0,
    "M": 2.0,
    "I": 3.0,
    "B": 4.0,
    "H": 5.0,
    "E": 6.0,
    "X": 7.0,
    "N": 0.0,
}


def build_fallback_matrix(sequence, structure, loop_type, n=130):
    """
    Deterministic numeric (n x n) matrix from 1D annotations.
    Produces float32 and fixed size so the ConvLSTM model can run as designed.
    """
    seq = _safe_pad(sequence, n=n, pad_char="N")
    st = _safe_pad(structure, n=n, pad_char="N")
    lp = _safe_pad(loop_type, n=n, pad_char="N")

    seq_v = np.fromiter(
        (_base_to_int.get(c, 0.0) for c in seq), dtype=np.float32, count=n
    )
    st_v = np.fromiter(
        (_struct_to_int.get(c, 0.0) for c in st), dtype=np.float32, count=n
    )
    lp_v = np.fromiter(
        (_loop_to_int.get(c, 0.0) for c in lp), dtype=np.float32, count=n
    )

    m = (
        0.50 * np.outer(seq_v, seq_v)
        + 0.30 * np.outer(st_v, st_v)
        + 0.20 * np.outer(lp_v, lp_v)
    ).astype(np.float32)

    diag = (0.25 * seq_v + 0.15 * st_v + 0.10 * lp_v).astype(np.float32)
    m[np.arange(n), np.arange(n)] += diag

    m /= np.max(np.abs(m)) + 1e-6
    return m.astype(np.float32)


def load_npy(x):
    arr = np.load(x)
    return np.resize(arr, (130, 130)).astype(np.float32)




## === cell 3
train = pd.read_json(os.path.join(BASE_INPUT, "train.json"), lines=True)
test = pd.read_json(os.path.join(BASE_INPUT, "test.json"), lines=True)
sub = pd.read_csv(os.path.join(BASE_INPUT, "sample_submission.csv"))

train["id_hash"] = train["id"].apply(lambda x: x.split("_")[1])
test["id_hash"] = test["id"].apply(lambda x: x.split("_")[1])

print(train.shape, test.shape, sub.shape)
print("train seq_length unique:", sorted(train["seq_length"].unique().tolist()))
print("test seq_length unique:", sorted(test["seq_length"].unique().tolist()))



## === cell 4
if has_bpps:
    npy_files = sorted([f for f in os.listdir(BPPS_DIR) if f.endswith(".npy")])
    npys_paths = pd.Series([os.path.join(BPPS_DIR, f) for f in npy_files])
    npys_ids = npys_paths.apply(lambda x: os.path.basename(x).split("_")[1]).apply(
        lambda x: x.split(".")[0]
    )
    npys = pd.DataFrame([*zip(npys_paths, npys_ids)], columns=["path", "id_hash"])
    print("Found npy files:", len(npys))

    paths = npys["path"].tolist()
    mats = Parallel(n_jobs=4)(delayed(load_npy)(p) for p in paths)
    npys["genetic_probs"] = mats
    npys = npys[["genetic_probs", "id_hash"]]
else:
    all_df = pd.concat(
        [
            train[["id_hash", "sequence", "structure", "predicted_loop_type"]],
            test[["id_hash", "sequence", "structure", "predicted_loop_type"]],
        ],
        axis=0,
        ignore_index=True,
    ).drop_duplicates("id_hash")

    def _gen_row(row):
        return build_fallback_matrix(
            row["sequence"], row["structure"], row["predicted_loop_type"], n=130
        )

    mats = Parallel(n_jobs=4)(delayed(_gen_row)(row) for _, row in all_df.iterrows())
    npys = pd.DataFrame({"id_hash": all_df["id_hash"].values, "genetic_probs": mats})

print("npys shape:", npys.shape)
print(
    "Example matrix dtype/shape:",
    npys["genetic_probs"].iloc[0].dtype,
    npys["genetic_probs"].iloc[0].shape,
)



## === cell 5
target_columns = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]


def _to_float_array(x, length=68):
    arr = np.asarray(x, dtype=np.float32)
    if arr.shape[0] != length:
        arr = np.resize(arr, (length,)).astype(np.float32, copy=False)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(
        np.float32, copy=False
    )
    return arr


Y_list = []
for col in target_columns:
    Y_list.append(
        np.stack([_to_float_array(v, length=68) for v in train[col].values], axis=0)
    )
y = np.stack(Y_list, axis=-1).astype(np.float32, copy=False)

print("y shape:", y.shape, y.dtype)



## === cell 6
train_merged = train.merge(npys, on="id_hash", how="left")
test_merged = test.merge(npys, on="id_hash", how="left")

if (
    train_merged["genetic_probs"].isna().any()
    or test_merged["genetic_probs"].isna().any()
):
    print("Warning: missing genetic_probs; filling with zeros.")

    def _fill_na(x):
        return (
            np.zeros((130, 130), dtype=np.float32)
            if x is None or (isinstance(x, float) and np.isnan(x))
            else x
        )

    train_merged["genetic_probs"] = train_merged["genetic_probs"].apply(_fill_na)
    test_merged["genetic_probs"] = test_merged["genetic_probs"].apply(_fill_na)

X_train = np.stack(train_merged["genetic_probs"].values).astype(
    np.float32
)  # (N,130,130)
X_test = np.stack(test_merged["genetic_probs"].values).astype(
    np.float32
)  # (Nt,130,130)

print("X_train:", X_train.shape, X_train.dtype)
print("X_test:", X_test.shape, X_test.dtype)




## === cell 7
def MCRMSE(y_true, y_pred):
    colwise_mse = tf.reduce_mean(tf.square(y_true - y_pred), axis=1)  # (batch, 5)
    return tf.reduce_mean(tf.sqrt(colwise_mse + 1e-8), axis=1)  # (batch,)


def build_model(seq_len=107, pred_len=68, dropout=0.5, embed_dim=100, hidden_dim=128):
    image_tensor = L.Input(shape=(130, 130), dtype=tf.float32)

    im = L.GaussianDropout(0.2)(image_tensor)
    im = L.Reshape((1, 130, 130, 1))(im)  # time=1, H, W, C=1

    im = L.ConvLSTM2D(
        32, 4, recurrent_dropout=0.0, padding="same", return_sequences=True
    )(im)
    im = L.ConvLSTM2D(
        16, 4, recurrent_dropout=0.0, padding="same", return_sequences=False
    )(
        im
    )  # (H,W,filters)

    im = L.Conv2D(1, kernel_size=1, padding="same", activation="linear")(im)
    im = L.Reshape((130, 130))(im)  # (130,130)

    truncated = im[:, :pred_len, :]  # (batch, pred_len, 130)
    out = L.Dense(5, activation="linear")(truncated)  # (batch, pred_len, 5)

    model = tf.keras.Model(inputs=image_tensor, outputs=out)
    model.compile(tf.keras.optimizers.Adam(), loss=MCRMSE)
    return model




## === cell 8
tf.config.optimizer.set_jit(True)
model = build_model(seq_len=107, pred_len=68)
model.summary()



## === cell 9
ckpt_path = "model.weights.h5"

history = model.fit(
    X_train,
    y,
    batch_size=64,
    epochs=100,
    validation_split=0.05,
    callbacks=[
        tf.keras.callbacks.ReduceLROnPlateau(),
        tf.keras.callbacks.ModelCheckpoint(
            ckpt_path,
            save_best_only=True,
            monitor="val_loss",
            mode="min",
            save_weights_only=True,
        ),
    ],
    verbose=2,
)



## === cell 10
train_pred = model.predict(X_train, batch_size=128, verbose=0)
print("train_pred:", train_pred.shape, train_pred.dtype)
print(
    "MAE(flat):",
    mean_absolute_error(train_pred.reshape(len(train_pred), -1), y.reshape(len(y), -1)),
)



## === cell 11
best_model = build_model(seq_len=107, pred_len=68)

if os.path.exists(ckpt_path):
    best_model.load_weights(ckpt_path)
else:
    print(
        f"Warning: checkpoint not found at {ckpt_path}; using last-epoch model weights."
    )
    best_model.set_weights(model.get_weights())

test_pred_68 = best_model.predict(X_test, batch_size=128, verbose=0)  # (Nt,68,5)
print("test_pred_68:", test_pred_68.shape)



## === cell 12
pred_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

rows = []
for i in range(len(test_merged)):
    uid = test_merged.loc[i, "id"]
    seq_len = int(test_merged.loc[i, "seq_length"])
    seq_scored = int(test_merged.loc[i, "seq_scored"])

    pred68 = test_pred_68[i]  # (68,5)

    full = np.zeros((seq_len, 5), dtype=np.float32)
    use = min(seq_scored, pred68.shape[0], seq_len)
    full[:use, :] = pred68[:use, :]
    if seq_len > use:
        full[use:, :] = full[use - 1, :] if use > 0 else 0.0

    for pos in range(seq_len):
        r = {"id_seqpos": f"{uid}_{pos}"}
        for j, c in enumerate(pred_cols):
            r[c] = float(full[pos, j])
        rows.append(r)

preds_df = pd.DataFrame(rows)
print("preds_df shape:", preds_df.shape)
print(preds_df.head())



## === cell 13
sample_df = pd.read_csv(os.path.join(BASE_INPUT, "sample_submission.csv"))

submission = sample_df[["id_seqpos"]].merge(preds_df, on="id_seqpos", how="left")

for c in pred_cols:
    submission[c] = submission[c].fillna(0.0).astype(np.float32)

submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())
