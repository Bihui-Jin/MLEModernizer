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

0.38981

# 6. Current score

0.24403

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 0.24444) has done: 'I first remove the `tensorflow_addons` dependency causing the protobuf `MessageFactory` crash, and replace the optimizer with plain Adam (keeping the same loss/model core). Next, I fix the Keras Functional error by replacing the raw `tf.reshape` call with a Keras `Reshape` layer, which preserves the intended tensor shape. Then I make preprocessing robust for both 107- and 130-length sequences by padding/truncating to a requested `seq_len`, eliminating the “axes don’t match array” error and ensuring predictions cover all `seq_length` positions required by the submission. Finally, I fix the missing import for `train_test_split`, correct GPU detection, and build the submission by aligning predictions to `sample_submission.csv` (always writing a valid `submission.csv` with the required columns).'
- What this solution (achieved 0.24432) has done: 'I fix the immediate crash in the first cell caused by the protobuf 6.x / TensorFlow import incompatibility by forcing the pure-Python protobuf implementation before importing TensorFlow, which is the standard Kaggle-safe workaround and should be score-neutral. I also fix a couple of small logic bugs in the “no missing values” checks (bitwise NOT on a boolean) that can silently misbehave, without changing modeling behavior. Finally, I add a minimal safety check to ensure the submission matches `sample_submission.csv` row count/order and always writes a valid `submission.csv`.'
- What this solution (achieved 0.24443) has done: 'You’re currently crashing before any training due to an incompatibility between TensorFlow 2.18 and protobuf 6.x (the `MessageFactory.GetPrototype` AttributeError). I fix this by forcing the pure-Python protobuf runtime *and* disabling the C++ implementation explicitly **before** importing TensorFlow, which is the standard Kaggle-safe workaround and should be score-neutral. I also add a tiny safety fallback that imports TensorFlow after setting the env vars (and fails loudly with a clearer message if it still can’t), while keeping the exact same model/training/prediction logic. Finally, I keep the submission alignment checks so the pipeline always produces a valid `submission.csv` matching `sample_submission.csv`.'
- What this solution (achieved 0.24422) has done: 'I fix the immediate crash in cell 1 by forcing TensorFlow to use the pure-Python protobuf implementation (and disabling the C++ implementation) before importing TensorFlow, which avoids the `MessageFactory.GetPrototype` error in this Kaggle environment. I also add a robust fallback that sets `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` unconditionally (not just `setdefault`) and clears any partially imported TensorFlow modules before retrying, without changing your model/training logic. These changes are score-neutral and only unblock end-to-end execution so your existing pipeline can train, predict, and write a valid `submission.csv`. No modeling/architecture/training-loop changes are made since your current score (0.24443) is already better than the target (0.38981) for a lower-is-better metric.'
- What this solution (achieved 0.24458) has done: 'I fix the TensorFlow/protobuf crash by setting the correct environment variables *before any TensorFlow import* (the current ones include an invalid key and still allow the C++ protobuf runtime to load). I also renumber the notebook cells to start at 1 (Kaggle expects execution in order; your provided “cell 0” needs to be “cell 1”) while keeping all model/training/prediction logic identical. Since your current score (0.24422, lower-is-better) is already better than the target (0.38981), I won’t make any score-improving changes—only stability fixes to ensure it runs end-to-end and writes a valid `submission.csv`. Finally, I keep the existing submission alignment checks so the output always matches `sample_submission.csv` exactly.'
- What this solution (achieved 0.24451) has done: 'I fix the TensorFlow/protobuf import crash by using only the valid environment variables for forcing the pure-Python protobuf runtime, and I ensure those are set before *any* TensorFlow-related import occurs (the current extra env keys don’t help and can still allow the failing C++ path). I also renumber cells to start at 1 so the notebook executes in order as provided. No model/training/prediction logic be changed because your current score (0.24458, lower-is-better) is already better than the target (0.38981), so the goal is stability and producing a valid `submission.csv`. The rest of the pipeline (data loading, preprocessing, training, inference, and submission alignment to `sample_submission.csv`) stays the same.'
- What this solution (achieved 0.2445) has done: 'I fix the TensorFlow/protobuf `MessageFactory.GetPrototype` crash by enforcing the pure-Python protobuf implementation *and* preventing the C++ one from loading before any TensorFlow import, plus clearing any partially-imported `google.protobuf` modules on retry. This is the root cause of your current runtime error and is score-neutral (it only unblocks execution). I also renumber the cells to start at 1 (your execution currently fails at “cell 0”), keeping all model/training/inference logic identical. Since your current score (0.24451, lower-is-better) is already better than the target (0.38981), I won’t make any score-improving changes—only stability fixes to ensure a valid `submission.csv` is produced end-to-end.'
- What this solution (achieved 0.24441) has done: 'I fix the TensorFlow/protobuf crash that happens before any training by forcing the pure-Python protobuf runtime in the only supported way for this environment (and removing the unsupported/ineffective env keys that still allow the failing C++ path). I also renumber the notebook cells to start from 1 so the execution order matches what Kaggle expects, without altering your model/training/inference logic. Because your current score (0.2445, lower-is-better) is already better than the target (0.38981), I not make any score-improving changes—only stability fixes so it runs end-to-end and always writes a valid `submission.csv` matching `sample_submission.csv` order/rows.'
- What this solution (achieved 0.24455) has done: 'I fix the TensorFlow/protobuf crash by setting both required environment variables *before any TensorFlow import* and clearing any already-imported protobuf/TensorFlow modules before retrying, which resolves the `MessageFactory.GetPrototype` error in this Kaggle environment. I also renumber the notebook cells to start at 1 so execution order matches the expected format, without altering any model/training/prediction logic. Since your current score (0.24441, lower-is-better) is already much better than the target (0.38981), I not make score-improving changes—only stability fixes to ensure it runs end-to-end and writes a valid `submission.csv` matching `sample_submission.csv` rows/order.'
- What this solution (achieved 0.24458) has done: 'I fix the TensorFlow/protobuf crash by setting the required protobuf environment variables even earlier and preventing the C++ protobuf implementation from loading, then clearing any partially imported protobuf/TensorFlow modules before retrying the import. This is the root cause of the `MessageFactory.GetPrototype` error and is score-neutral (it only unblocks execution so your existing training/inference can run). I also renumber the cells to start from 1 (your provided notebook starts at cell 0), keeping all model/training/prediction logic identical. Finally, I keep the existing submission alignment logic so the script always writes a valid `submission.csv` with the exact required columns and row order.'
- What this solution (achieved 0.24443) has done: 'I fix the TensorFlow/protobuf crash in the first cell by forcing the pure-Python protobuf runtime in the only reliable way for TF 2.18 + protobuf 6.x on Kaggle: set the env vars before any TF import and also force the Python implementation via `google.protobuf.internal.api_implementation` before importing TensorFlow. This is purely an execution/stability fix and should be score-neutral (no model/training/inference changes). I also renumber the cells to start at 1 (your current script starts at cell 0) while keeping all subsequent logic identical. Finally, I keep the existing submission alignment checks to guarantee a valid `submission.csv` with the exact required columns/row order.'
- What this solution (achieved 0.24447) has done: 'I fix the TensorFlow/protobuf crash happening before training by setting the protobuf environment variables *before* any protobuf/TensorFlow imports and by importing `google.protobuf.internal.api_implementation` early, which is the reliable workaround in this Kaggle TF 2.18 + protobuf 6.x environment. I also renumber the notebook to start at cell 1 (your current script starts at cell 0) so it runs in-order as expected. These are execution/stability-only changes; I won’t change the model, training loop, preprocessing, blending, or submission construction since your current score (0.24443, lower-is-better) is already better than the target (0.38981). The pipeline run end-to-end and always write a valid `submission.csv` with the required columns and row order.'
- What this solution (achieved 0.24444) has done: 'I fix the TensorFlow/protobuf import crash by setting the protobuf env vars at the very top and forcing the Python protobuf implementation before importing anything that could transitively load protobuf/TensorFlow; the current workaround still hits the `MessageFactory.GetPrototype` path in this environment. I also renumber cells to start at 1 (to match the required format) without changing any model/training/inference logic. Since your current score (0.24447, lower-is-better) is already much better than the target (0.38981), I avoid any score-improving changes and only make stability/execution fixes so the script runs end-to-end and writes a valid `submission.csv`.'
- What this solution (achieved 0.24403) has done: 'I fix the runtime crash caused by the TensorFlow 2.18 + protobuf 6.x incompatibility by forcing the pure-Python protobuf implementation *before any protobuf/TensorFlow import*, and by removing the unsupported env var that can still allow the C++ path to load. I also renumber the notebook cells to start at 1 (your provided script starts at cell 0), keeping all modeling/training/inference logic identical. Since your current score (0.24444, lower-is-better) is already better than the target (0.38981), I won’t make any score-improving changes—only execution/stability fixes so it runs end-to-end and writes a valid `submission.csv`. The submission alignment logic remains unchanged to guarantee the correct row order/columns.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"

import warnings

warnings.filterwarnings("ignore")

import gc, random, math, json, sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm


def _import_tensorflow_safely():
    try:
        from google.protobuf.internal import api_implementation

        try:
            api_implementation._SetImplementationType("python")
        except Exception:
            pass
    except Exception:
        pass

    try:
        import tensorflow as tf  # noqa
        import tensorflow.keras.backend as K  # noqa
        import tensorflow.keras.layers as L  # noqa

        return tf, K, L
    except Exception as e:
        for m in list(sys.modules.keys()):
            if m.startswith(("tensorflow", "google.protobuf")):
                sys.modules.pop(m, None)
        gc.collect()

        os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
        os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"

        try:
            from google.protobuf.internal import api_implementation

            try:
                api_implementation._SetImplementationType("python")
            except Exception:
                pass
        except Exception:
            pass

        try:
            import tensorflow as tf  # noqa
            import tensorflow.keras.backend as K  # noqa
            import tensorflow.keras.layers as L  # noqa

            return tf, K, L
        except Exception as e2:
            raise RuntimeError(
                "TensorFlow import failed even after forcing pure-Python protobuf.\n"
                f"Original error: {repr(e)}\nRetry error: {repr(e2)}"
            )


tf, K, L = _import_tensorflow_safely()

from sklearn.model_selection import train_test_split, KFold

SEED = 34
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train = pd.read_json("/kaggle/input/stanford-covid-vaccine/train.json", lines=True)
test = pd.read_json("/kaggle/input/stanford-covid-vaccine/test.json", lines=True)
sample_sub = pd.read_csv("/kaggle/input/stanford-covid-vaccine/sample_submission.csv")



## === cell 2
print(train.shape)
if not train.isnull().values.any():
    print("No missing values")
train.head()



## === cell 3
print(test.shape)
if not test.isnull().values.any():
    print("No missing values")
test.head()



## === cell 4
print(sample_sub.shape)
if not sample_sub.isnull().values.any():
    print("No missing values")
sample_sub.head()



## === cell 5
target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]



## === cell 6
token2int = {x: i for i, x in enumerate("().ACGUBEHIMSX")}
PAD_TOKEN = "."  # safe pad char that exists in token2int




## === cell 7
def preprocess_inputs(
    df, cols=("sequence", "structure", "predicted_loop_type"), seq_len=107
):
    arrs = []
    for _, row in df[list(cols)].iterrows():
        per_col = []
        for c in cols:
            s = row[c]
            if len(s) < seq_len:
                s = s + (PAD_TOKEN * (seq_len - len(s)))
            elif len(s) > seq_len:
                s = s[:seq_len]
            per_col.append([token2int[ch] for ch in s])
        arrs.append(np.stack(per_col, axis=1))
    return np.asarray(arrs, dtype=np.int32)




## === cell 8
train_filt = train[train.signal_to_noise > 1].reset_index(drop=True)

train_inputs = preprocess_inputs(train_filt, seq_len=107)
train_labels = (
    np.array(train_filt[target_cols].values.tolist())
    .transpose((0, 2, 1))
    .astype(np.float32)
)

print("train_inputs:", train_inputs.shape, train_inputs.dtype)
print("train_labels:", train_labels.shape, train_labels.dtype)




## === cell 9
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
    gru=False, seq_len=107, pred_len=68, dropout=0.5, embed_dim=75, hidden_dim=128
):
    inputs = tf.keras.layers.Input(shape=(seq_len, 3), dtype=tf.int32)

    embed = tf.keras.layers.Embedding(input_dim=len(token2int), output_dim=embed_dim)(
        inputs
    )
    reshaped = tf.keras.layers.Reshape((seq_len, 3 * embed_dim))(embed)

    reshaped = tf.keras.layers.SpatialDropout1D(0.2)(reshaped)

    if gru:
        hidden = gru_layer(hidden_dim, dropout)(reshaped)
        hidden = gru_layer(hidden_dim, dropout)(hidden)
        hidden = gru_layer(hidden_dim, dropout)(hidden)
    else:
        hidden = lstm_layer(hidden_dim, dropout)(reshaped)
        hidden = lstm_layer(hidden_dim, dropout)(hidden)
        hidden = lstm_layer(hidden_dim, dropout)(hidden)

    truncated = hidden[:, :pred_len]
    out = tf.keras.layers.Dense(5, activation="linear")(truncated)

    model = tf.keras.Model(inputs=inputs, outputs=out)

    adam = tf.optimizers.Adam()
    model.compile(optimizer=adam, loss="mse")
    return model




## === cell 10
train_inputs, val_inputs, train_labels, val_labels = train_test_split(
    train_inputs, train_labels, test_size=0.1, random_state=SEED
)
print(train_inputs.shape, val_inputs.shape, train_labels.shape, val_labels.shape)



## === cell 11
if len(tf.config.list_physical_devices("GPU")) > 0:
    print("Training on GPU")
else:
    print("Training on CPU")



## === cell 12
lr_callback = tf.keras.callbacks.ReduceLROnPlateau()
sv_gru = tf.keras.callbacks.ModelCheckpoint(
    "model_gru.weights.h5",
    save_weights_only=True,
    monitor="val_loss",
    save_best_only=True,
)
sv_lstm = tf.keras.callbacks.ModelCheckpoint(
    "model_lstm.weights.h5",
    save_weights_only=True,
    monitor="val_loss",
    save_best_only=True,
)



## === cell 13
gru = build_model(gru=True, seq_len=107, pred_len=68)
history_gru = gru.fit(
    train_inputs,
    train_labels,
    validation_data=(val_inputs, val_labels),
    batch_size=64,
    epochs=70,
    callbacks=[lr_callback, sv_gru],
    verbose=2,
)
print(
    f"Min training loss={min(history_gru.history['loss'])}, min validation loss={min(history_gru.history['val_loss'])}"
)



## === cell 14
lstm = build_model(gru=False, seq_len=107, pred_len=68)
history_lstm = lstm.fit(
    train_inputs,
    train_labels,
    validation_data=(val_inputs, val_labels),
    batch_size=64,
    epochs=75,
    callbacks=[lr_callback, sv_lstm],
    verbose=2,
)
print(
    f"Min training loss={min(history_lstm.history['loss'])}, min validation loss={min(history_lstm.history['val_loss'])}"
)



## === cell 15
fig, ax = plt.subplots(1, 2, figsize=(20, 10))

ax[0].plot(history_gru.history["loss"])
ax[0].plot(history_gru.history["val_loss"])

ax[1].plot(history_lstm.history["loss"])
ax[1].plot(history_lstm.history["val_loss"])

ax[0].set_title("GRU")
ax[1].set_title("LSTM")

ax[0].legend(["train", "validation"], loc="upper right")
ax[1].legend(["train", "validation"], loc="upper right")

ax[0].set_ylabel("Loss")
ax[0].set_xlabel("Epoch")
ax[1].set_ylabel("Loss")
ax[1].set_xlabel("Epoch")



## === cell 16
gru.load_weights("model_gru.weights.h5")
lstm.load_weights("model_lstm.weights.h5")




## === cell 17
def predict_df(model, df, seq_len_model=107, pred_len_model=68):
    """
    Returns predictions shaped (n, pred_len_model, 5) for the first pred_len_model positions.
    """
    x = preprocess_inputs(df, seq_len=seq_len_model)
    preds = model.predict(x, batch_size=64, verbose=0)
    return preds


test_preds_gru_scored = predict_df(gru, test, seq_len_model=107, pred_len_model=68)
test_preds_lstm_scored = predict_df(lstm, test, seq_len_model=107, pred_len_model=68)

print(test_preds_gru_scored.shape, test_preds_lstm_scored.shape)




## === cell 18
def expand_to_seq_length(preds_scored, seq_length=107):
    n, scored_len, k = preds_scored.shape
    if seq_length == scored_len:
        return preds_scored
    if seq_length < scored_len:
        return preds_scored[:, :seq_length, :]
    pad_len = seq_length - scored_len
    last = preds_scored[:, -1:, :]
    pad = np.repeat(last, pad_len, axis=1)
    return np.concatenate([preds_scored, pad], axis=1)


all_rows = []
for i, uid in enumerate(test["id"].values):
    seq_len = int(test.loc[i, "seq_length"])
    gru_full = expand_to_seq_length(
        test_preds_gru_scored[i : i + 1], seq_length=seq_len
    )[0]
    lstm_full = expand_to_seq_length(
        test_preds_lstm_scored[i : i + 1], seq_length=seq_len
    )[0]
    blend_full = 0.5 * gru_full + 0.5 * lstm_full

    single_df = pd.DataFrame(blend_full, columns=target_cols)
    single_df["id_seqpos"] = [f"{uid}_{p}" for p in range(seq_len)]
    all_rows.append(single_df)

blend_preds_df = pd.concat(all_rows, axis=0, ignore_index=True)
blend_preds_df = blend_preds_df[["id_seqpos"] + target_cols]
print(blend_preds_df.shape)
blend_preds_df.head()



## === cell 19
submission = sample_sub[["id_seqpos"]].merge(blend_preds_df, on="id_seqpos", how="left")

for c in target_cols:
    submission[c] = submission[c].astype(np.float32)

missing = submission[target_cols].isna().sum().sum()
if missing > 0:
    submission[target_cols] = submission[target_cols].fillna(0.0)

submission = (
    submission.set_index("id_seqpos").reindex(sample_sub["id_seqpos"]).reset_index()
)

print("submission shape:", submission.shape, "missing filled:", int(missing))
submission.head()



## === cell 20
submission.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")
print(submission.columns.tolist())
print(submission.head(3))
