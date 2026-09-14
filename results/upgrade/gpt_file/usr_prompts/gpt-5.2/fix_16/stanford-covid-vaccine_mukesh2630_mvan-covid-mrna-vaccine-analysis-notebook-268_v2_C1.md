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

0.37958

# 6. Current score

0.23039

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 0.23208) has done: 'I fix the environment import crash by pinning protobuf to the pure-Python implementation before importing TensorFlow, which avoids the `MessageFactory.GetPrototype` issue in Kaggle’s runtime. Then I fix the Keras 2.18 `ModelCheckpoint(save_weights_only=True)` filename requirement by using the `.weights.h5` suffix and loading those same files at inference time, eliminating the missing-file errors. Finally, I correct the public/private test split (this competition’s test is length 107 only) and ensure predictions are generated for exactly the `sample_submission.csv` rows, writing a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.23032) has done: 'I fix the TensorFlow import crash by forcing the pure-Python protobuf backend *and* disabling the C++ implementation via `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION=2`, which is the most reliable workaround for the `MessageFactory.GetPrototype` issue in Kaggle TF 2.18 environments. Then I correct a small but real logic bug in `build_model(gru=5)` where the last recurrent layer mistakenly reuses `reshaped` instead of the current `hidden` tensor (this preserves the intended architecture path and should improve score, moving it toward your target). Finally, I make submission generation more robust by asserting row alignment with `sample_submission.csv` and ensuring the output has the exact required columns and length.'
- What this solution (achieved 0.23055) has done: 'I fix the TensorFlow import crash (`MessageFactory.GetPrototype`) by setting the protobuf environment variables *and* ensuring the pure-Python protobuf runtime is used before TensorFlow loads; then I add a safe fallback to force the pure-Python implementation if Kaggle still loads the C++ one. I also correct the `MCRMSE` loss reduction axes so it matches the competition’s columnwise RMSE definition for tensors shaped `(batch, seq, targets)`, which should legitimately move your score upward toward the target without changing the model architecture. Finally, I keep the rest of the training/inference pipeline intact and ensure the submission CSV is always produced with the exact required shape and columns.'
- What this solution (achieved 0.23054) has done: 'We fix the TensorFlow import crash by ensuring the pure-Python protobuf implementation is selected *before* TensorFlow is imported (including setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` and also disabling the upb/C++ path via `TF_PROTOBUF_IMPLEMENTATION=python`). This is the only blocker preventing the notebook from running end-to-end and producing `submission.csv`. All modeling/training/inference logic be kept identical; changes are limited to environment setup and a small safety check so the run fails fast with a clear message if protobuf is still not in Python mode. This should restore the prior working score trajectory (and at minimum produce a valid submission).'
- What this solution (achieved 0.23048) has done: 'We fix the TensorFlow import crash by avoiding the protobuf/python forcing logic (which is now triggering the `MessageFactory.GetPrototype` failure in this environment) and instead letting Kaggle’s provided TF/protobuf stack load normally. Then we keep the model/training/inference logic unchanged, only adding small robustness checks so missing weight files fail fast with a clearer error message and ensuring the submission aligns exactly to `sample_submission.csv`. This should restore an end-to-end run and produce a valid `submission.csv` without altering the core modeling approach or intended score behavior.'
- What this solution (achieved 0.23042) has done: 'I fix the TensorFlow/protobuf import crash by forcing protobuf to use the pure-Python backend *before* importing TensorFlow, which resolves the `MessageFactory.GetPrototype` error seen in cell 0 on Kaggle TF 2.18 runtimes. I keep the model/training/inference logic intact, only adding a small safety check to confirm the protobuf backend is correctly selected. I also keep the existing weight filename suffixes (`.weights.h5`) to satisfy Keras 2.18 requirements and ensure inference loads the same files. Finally, I leave submission generation unchanged except for path robustness (fallback to the already-listed `/kaggle/data/...` paths if needed) so it always writes a valid `submission.csv`.'
- What this solution (achieved 0.23015) has done: 'The only blocker is the TensorFlow import crash caused by forcing the pure-Python protobuf backend in this Kaggle environment; we remove that strict forcing and instead make TensorFlow import robust by trying a normal import first and only falling back to python-protobuf if needed. This change is score-neutral (it just makes the notebook run end-to-end again). I also keep the existing training/inference logic intact, but add a tiny safety fix to ensure the custom loss reduces to a scalar per batch (so it’s always compatible with Keras’ expectations). Finally, submission generation stays the same, still producing a valid `submission.csv` with the required columns and row alignment to `sample_submission.csv`.'
- What this solution (achieved 0.23035) has done: 'We fix the runtime crash in cell 0 by forcing protobuf to use the pure-Python implementation *before* TensorFlow is imported, and by hard-restarting the TF/protobuf module imports if the first import fails with the known `MessageFactory.GetPrototype` issue. This is an execution blocker and should be score-neutral (it only affects import stability). All model/training/inference logic remains unchanged, including architecture, loss definition, training loops, and blending; we only adjust the import bootstrap so the notebook runs end-to-end again. The rest of the pipeline continue to write a valid `submission.csv` aligned exactly to `sample_submission.csv`.'
- What this solution (achieved 0.23027) has done: 'I fix the runtime crash happening before any training by making TensorFlow import robust to the protobuf `MessageFactory.GetPrototype` failure in this environment, using the standard safe workaround (force pure-Python protobuf and fully purge/reload protobuf/TensorFlow modules). This change is execution-critical and score-neutral (it only affects imports). I keep the model, loss, training loops, blending, and submission logic intact, only adding a stricter import bootstrap and a clear check that the protobuf backend is actually “python” before proceeding. The rest of the pipeline continue to train, load weights, and write a valid `submission.csv` with the required columns/row alignment.'
- What this solution (achieved 0.23014) has done: 'The crash happens before training because TensorFlow is being imported in an environment where protobuf’s `MessageFactory.GetPrototype` is missing/incompatible; the current “retry with python protobuf” path still triggers that same failure. I make the TensorFlow import bootstrap more robust by trying a normal import first, then (only if it fails) forcing the pure-Python protobuf backend and purging/re-importing modules without calling protobuf internals that can themselves crash. This is execution-critical and score-neutral (it doesn’t change model/training logic), and it let the notebook run end-to-end again to produce a valid `submission.csv`. I also add a tiny deterministic session cleanup between model trainings to reduce the chance of OOM/retrace issues in the 5-model loop (no change to architecture/training semantics).'
- What this solution (achieved 0.23011) has done: 'I fix the TensorFlow import crash by setting protobuf-related environment variables to use the pure-Python implementation *before* any TensorFlow/protobuf import happens, and by purging already-loaded protobuf/TensorFlow modules before retrying. This is execution-critical and score-neutral: it only affects import stability, not the model/training/inference logic. I keep the model, loss, training loops, ensembling, and submission formatting unchanged, only adding a small backend check and a controlled retry path to avoid the `MessageFactory.GetPrototype` failure. The script then run end-to-end and always write a valid `submission.csv` with the required columns and row alignment.'
- What this solution (achieved 0.2305) has done: 'The crash is happening at TensorFlow import time due to an incompatible protobuf backend: forcing pure-Python protobuf in this Kaggle image triggers `MessageFactory.GetPrototype` errors. I make the TensorFlow import bootstrap robust by first trying the default environment (which matches Kaggle’s TF build), and only if that fails, retry with pure-Python protobuf env vars and a full module purge. This is execution-critical but score-neutral (no model/training/inference logic changes), and it let the rest of your existing pipeline run end-to-end and write a valid `submission.csv`. I also keep your existing loss/model/training and submission alignment untouched.'
- What this solution (achieved 0.23044) has done: 'I fix the TensorFlow import crash by forcing protobuf to use the pure-Python implementation from the start (this specific `MessageFactory.GetPrototype` error is caused by an incompatible C++/upb protobuf path in this Kaggle TF 2.18 environment). I keep all model/training/inference logic identical, only changing the import bootstrap so the notebook can run end-to-end again and produce `submission.csv`. I also add a tiny, score-neutral safety check to confirm TF actually imported before proceeding, and keep the existing submission alignment assertions intact.'
- What this solution (achieved 0.23039) has done: 'I fix the execution blocker in cell 0 by making TensorFlow import robust: try a normal import first (which matches Kaggle’s TF/protobuf build), and only if it fails with the known protobuf `GetPrototype` issue, retry after forcing the pure-Python protobuf backend and purging modules. This change is score-neutral but required to run end-to-end and produce `submission.csv`. I keep the model/training/inference logic unchanged, and add a small sanity check that TensorFlow imported successfully before proceeding. The rest of the pipeline (training, weight saving/loading, blending, and submission formatting) remains identical to preserve the current score behavior while ensuring stability.'

# 9. Code solution

## === cell 0
import os
import warnings

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
warnings.filterwarnings("ignore")

import gc, json, math, random
import numpy as np
import pandas as pd
from tqdm import tqdm


def _purge_modules(prefixes):
    import sys

    for m in list(sys.modules.keys()):
        if any(m == p or m.startswith(p + ".") for p in prefixes):
            sys.modules.pop(m, None)


def _import_tensorflow_safely():
    """
    Execution blocker fix:
    Some Kaggle TF 2.18 images can error at import time with:
      AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'
    due to a protobuf backend mismatch.
    Approach: try default TF import first; if it fails, retry with pure-Python protobuf
    env vars and a full module purge. This is score-neutral (imports only).
    """
    try:
        _purge_modules(["tensorflow", "google.protobuf"])
        import tensorflow as tf  # noqa: F401

        return tf
    except Exception as e1:
        os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
        os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
        os.environ["TF_PROTOBUF_IMPLEMENTATION"] = "python"

        _purge_modules(["tensorflow", "google.protobuf"])
        try:
            import tensorflow as tf  # noqa: F401

            return tf
        except Exception as e2:
            raise RuntimeError(
                "TensorFlow import failed in both default and python-protobuf modes.\n"
                f"Default import error: {repr(e1)}\n"
                f"Python-protobuf retry error: {repr(e2)}"
            )


tf = _import_tensorflow_safely()

import tensorflow.keras.backend as K
import tensorflow.keras.layers as L

from sklearn.model_selection import train_test_split

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("TF version:", tf.__version__)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def _read_json_fallback(primary_path, fallback_path):
    if os.path.exists(primary_path):
        return pd.read_json(primary_path, lines=True)
    if os.path.exists(fallback_path):
        return pd.read_json(fallback_path, lines=True)
    raise FileNotFoundError(f"Could not find {primary_path} or {fallback_path}")


def _read_csv_fallback(primary_path, fallback_path):
    if os.path.exists(primary_path):
        return pd.read_csv(primary_path)
    if os.path.exists(fallback_path):
        return pd.read_csv(fallback_path)
    raise FileNotFoundError(f"Could not find {primary_path} or {fallback_path}")


train = _read_json_fallback(
    "/kaggle/input/stanford-covid-vaccine/train.json",
    "/kaggle/data/stanford-covid-vaccine/train.json",
)
test = _read_json_fallback(
    "/kaggle/input/stanford-covid-vaccine/test.json",
    "/kaggle/data/stanford-covid-vaccine/test.json",
)
sample_sub = _read_csv_fallback(
    "/kaggle/input/stanford-covid-vaccine/sample_submission.csv",
    "/kaggle/data/stanford-covid-vaccine/sample_submission.csv",
)

target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

token2int = {x: i for i, x in enumerate("().ACGUBEHIMSX")}


def preprocess_inputs(df, cols=("sequence", "structure", "predicted_loop_type")):
    if df is None or len(df) == 0:
        return np.zeros((0, 0, 0), dtype=np.int32)
    arr = (
        df.loc[:, list(cols)]
        .applymap(lambda seq: [token2int[x] for x in seq])
        .values.tolist()
    )
    arr = np.array(arr, dtype=np.int32)  # (n, 3, seq_len)
    return np.transpose(arr, (0, 2, 1))  # (n, seq_len, 3)


train_filt = train[train.signal_to_noise > 1].copy()
train_inputs = preprocess_inputs(train_filt)
train_labels = np.array(
    train_filt[target_cols].values.tolist(), dtype=np.float32
).transpose((0, 2, 1))

print("train_inputs:", train_inputs.shape, train_inputs.dtype)
print("train_labels:", train_labels.shape, train_labels.dtype)
print("test:", test.shape, "sample_sub:", sample_sub.shape)



## === cell 2
train.head()




## === cell 3
def MCRMSE(y_true, y_pred):
    colwise_mse = tf.reduce_mean(tf.square(y_true - y_pred), axis=1)  # (batch, 5)
    per_row = tf.reduce_mean(tf.sqrt(colwise_mse + 1e-9), axis=1)  # (batch,)
    return tf.reduce_mean(per_row)  # scalar


def gru_layer(hidden_dim, dropout):
    return tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(
            hidden_dim,
            dropout=dropout,
            return_sequences=True,
            kernel_initializer="orthogonal",
        )
    )


def lstm_layer(hidden_dim, dropout):
    return tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            hidden_dim,
            dropout=dropout,
            return_sequences=True,
            kernel_initializer="orthogonal",
        )
    )


def build_model(
    gru=1, seq_len=107, pred_len=68, dropout=0.5, embed_dim=75, hidden_dim=128
):
    inputs = tf.keras.layers.Input(shape=(seq_len, 3), dtype=tf.int32)

    embed = tf.keras.layers.Embedding(input_dim=len(token2int), output_dim=embed_dim)(
        inputs
    )
    reshaped = tf.keras.layers.Reshape((seq_len, 3 * embed_dim))(embed)

    reshaped = tf.keras.layers.SpatialDropout1D(0.2)(reshaped)

    if gru == 1:
        hidden = gru_layer(hidden_dim, dropout)(reshaped)
        hidden = gru_layer(hidden_dim, dropout)(hidden)
        hidden = gru_layer(hidden_dim, dropout)(hidden)
    elif gru == 0:
        hidden = lstm_layer(hidden_dim, dropout)(reshaped)
        hidden = lstm_layer(hidden_dim, dropout)(hidden)
        hidden = lstm_layer(hidden_dim, dropout)(hidden)
    elif gru == 3:
        hidden = gru_layer(hidden_dim, dropout)(reshaped)
        hidden = lstm_layer(hidden_dim, dropout)(hidden)
        hidden = lstm_layer(hidden_dim, dropout)(hidden)
    elif gru == 4:
        hidden = lstm_layer(hidden_dim, dropout)(reshaped)
        hidden = gru_layer(hidden_dim, dropout)(hidden)
        hidden = gru_layer(hidden_dim, dropout)(hidden)
    elif gru == 5:
        hidden = lstm_layer(hidden_dim, dropout)(reshaped)
        hidden = gru_layer(hidden_dim, dropout)(hidden)
        hidden = lstm_layer(hidden_dim, dropout)(hidden)
    else:
        raise ValueError(f"Unknown gru mode: {gru}")

    truncated = hidden[:, :pred_len]
    out = tf.keras.layers.Dense(5, activation="linear")(truncated)

    model = tf.keras.Model(inputs=inputs, outputs=out)
    adam = tf.optimizers.Adam()
    model.compile(optimizer=adam, loss=MCRMSE)
    return model




## === cell 4
train_inputs, val_inputs, train_labels, val_labels = train_test_split(
    train_inputs, train_labels, test_size=0.1, random_state=34
)
print("split:", train_inputs.shape, val_inputs.shape)



## === cell 5
tf.keras.backend.clear_session()
gc.collect()

lr_callback = tf.keras.callbacks.ReduceLROnPlateau()

gru = build_model(gru=1)
sv_gru = tf.keras.callbacks.ModelCheckpoint(
    "model_gru.weights.h5",
    save_weights_only=True,
    save_best_only=True,
    monitor="val_loss",
)

history_gru = gru.fit(
    train_inputs,
    train_labels,
    validation_data=(val_inputs, val_labels),
    batch_size=64,
    epochs=100,
    callbacks=[lr_callback, sv_gru],
    verbose=2,
)

print(
    f"Min training loss={min(history_gru.history['loss'])}, min validation loss={min(history_gru.history['val_loss'])}"
)



## === cell 6
tf.keras.backend.clear_session()
gc.collect()

lstm = build_model(gru=0)
sv_lstm = tf.keras.callbacks.ModelCheckpoint(
    "model_lstm.weights.h5",
    save_weights_only=True,
    save_best_only=True,
    monitor="val_loss",
)

history_lstm = lstm.fit(
    train_inputs,
    train_labels,
    validation_data=(val_inputs, val_labels),
    batch_size=64,
    epochs=100,
    callbacks=[lr_callback, sv_lstm],
    verbose=2,
)

print(
    f"Min training loss={min(history_lstm.history['loss'])}, min validation loss={min(history_lstm.history['val_loss'])}"
)



## === cell 7
tf.keras.backend.clear_session()
gc.collect()

hyb1 = build_model(gru=3)
sv_hyb1 = tf.keras.callbacks.ModelCheckpoint(
    "model_hyb1.weights.h5",
    save_weights_only=True,
    save_best_only=True,
    monitor="val_loss",
)

history_hyb1 = hyb1.fit(
    train_inputs,
    train_labels,
    validation_data=(val_inputs, val_labels),
    batch_size=64,
    epochs=100,
    callbacks=[lr_callback, sv_hyb1],
    verbose=2,
)

print(
    f"Min training loss={min(history_hyb1.history['loss'])}, min validation loss={min(history_hyb1.history['val_loss'])}"
)



## === cell 8
tf.keras.backend.clear_session()
gc.collect()

hyb2 = build_model(gru=4)
sv_hyb2 = tf.keras.callbacks.ModelCheckpoint(
    "model_hyb2.weights.h5",
    save_weights_only=True,
    save_best_only=True,
    monitor="val_loss",
)

history_hyb2 = hyb2.fit(
    train_inputs,
    train_labels,
    validation_data=(val_inputs, val_labels),
    batch_size=64,
    epochs=100,
    callbacks=[lr_callback, sv_hyb2],
    verbose=2,
)

print(
    f"Min training loss={min(history_hyb2.history['loss'])}, min validation loss={min(history_hyb2.history['val_loss'])}"
)



## === cell 9
tf.keras.backend.clear_session()
gc.collect()

hyb3 = build_model(gru=5)
sv_hyb3 = tf.keras.callbacks.ModelCheckpoint(
    "model_hyb3.weights.h5",
    save_weights_only=True,
    save_best_only=True,
    monitor="val_loss",
)

history_hyb3 = hyb3.fit(
    train_inputs,
    train_labels,
    validation_data=(val_inputs, val_labels),
    batch_size=64,
    epochs=100,
    callbacks=[lr_callback, sv_hyb3],
    verbose=2,
)

print(
    f"Min training loss={min(history_hyb3.history['loss'])}, min validation loss={min(history_hyb3.history['val_loss'])}"
)



## === cell 10
test_df = test.copy()
test_inputs = preprocess_inputs(test_df)

gru_test = build_model(gru=1, seq_len=107, pred_len=107)
lstm_test = build_model(gru=0, seq_len=107, pred_len=107)
hyb1_test = build_model(gru=3, seq_len=107, pred_len=107)
hyb2_test = build_model(gru=4, seq_len=107, pred_len=107)
hyb3_test = build_model(gru=5, seq_len=107, pred_len=107)

for f in [
    "model_gru.weights.h5",
    "model_lstm.weights.h5",
    "model_hyb1.weights.h5",
    "model_hyb2.weights.h5",
    "model_hyb3.weights.h5",
]:
    if not os.path.exists(f):
        raise FileNotFoundError(
            f"Missing weights file: {f}. Training should have created it via ModelCheckpoint."
        )

gru_test.load_weights("model_gru.weights.h5")
lstm_test.load_weights("model_lstm.weights.h5")
hyb1_test.load_weights("model_hyb1.weights.h5")
hyb2_test.load_weights("model_hyb2.weights.h5")
hyb3_test.load_weights("model_hyb3.weights.h5")

gru_preds = gru_test.predict(test_inputs, verbose=0)
lstm_preds = lstm_test.predict(test_inputs, verbose=0)
hyb1_preds = hyb1_test.predict(test_inputs, verbose=0)
hyb2_preds = hyb2_test.predict(test_inputs, verbose=0)
hyb3_preds = hyb3_test.predict(test_inputs, verbose=0)


def preds_to_long_df(df, preds):
    out = []
    for i, uid in enumerate(df.id.values):
        single_pred = preds[i]
        single_df = pd.DataFrame(single_pred, columns=target_cols)
        single_df["id_seqpos"] = [f"{uid}_{x}" for x in range(single_df.shape[0])]
        out.append(single_df)
    if len(out) == 0:
        return pd.DataFrame(columns=target_cols + ["id_seqpos"])
    return pd.concat(out, ignore_index=True)


preds_gru_df = preds_to_long_df(test_df, gru_preds)
preds_lstm_df = preds_to_long_df(test_df, lstm_preds)
preds_hyb1_df = preds_to_long_df(test_df, hyb1_preds)
preds_hyb2_df = preds_to_long_df(test_df, hyb2_preds)
preds_hyb3_df = preds_to_long_df(test_df, hyb3_preds)

blend_preds_df = pd.DataFrame({"id_seqpos": preds_gru_df["id_seqpos"]})
for col in target_cols:
    blend_preds_df[col] = (
        0.2 * preds_gru_df[col].astype(np.float32)
        + 0.2 * preds_lstm_df[col].astype(np.float32)
        + 0.2 * preds_hyb1_df[col].astype(np.float32)
        + 0.2 * preds_hyb2_df[col].astype(np.float32)
        + 0.2 * preds_hyb3_df[col].astype(np.float32)
    )

submission = sample_sub[["id_seqpos"]].merge(blend_preds_df, on="id_seqpos", how="left")
for col in target_cols:
    submission[col] = submission[col].fillna(0.0).astype(np.float32)

submission = submission[["id_seqpos"] + target_cols]
assert (
    submission.shape[0] == sample_sub.shape[0]
), "Submission row count mismatch vs sample_submission."
assert (
    list(submission.columns) == ["id_seqpos"] + target_cols
), "Submission columns mismatch."

print(submission.head())
print("submission shape:", submission.shape)

submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv")
