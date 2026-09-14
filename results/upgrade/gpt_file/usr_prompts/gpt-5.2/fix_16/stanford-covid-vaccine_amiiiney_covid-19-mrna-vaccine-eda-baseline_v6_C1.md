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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
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

0.52869

# 6. Current score

0.45291

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 0.45376) has done: 'I make the plotting cells robust (or no-op) so they don’t crash the run, because they are exploratory and currently stop execution due to seaborn/pandas type issues. I fix the Keras/TensorFlow import clash that triggers the protobuf `MessageFactory` error by using only `tf.keras` consistently, which also resolves the missing `Sequential` name. I also fix the test feature construction to match this competition’s data (all sequences are length 107 and require 107 rows per id), and ensure train/test one-hot columns are aligned so inference runs without shape errors. Finally, I generate a valid `submission.csv` with the exact required columns and `id_seqpos` order matching `sample_submission.csv`.'
- What this solution (achieved 0.45478) has done: 'I fix the TensorFlow/protobuf `MessageFactory.GetPrototype` crash by pinning protobuf to the TF-compatible pure-Python implementation at runtime (before importing TensorFlow), which is the minimal change that unblocks training. I also ensure the Keras imports remain strictly `tf.keras` to avoid mixed-keras conflicts, and add a safe fallback to load the best checkpointed model if available. Finally, I keep the existing feature engineering and model exactly the same, and ensure the submission is written as `submission.csv` with the required columns and the exact `sample_submission.csv` row order.'
- What this solution (achieved 0.46234) has done: 'I fix the TensorFlow/protobuf `MessageFactory.GetPrototype` crash by forcing the pure-Python protobuf implementation *before* any TensorFlow import and by also setting the versioned env var that TF sometimes checks, which unblocks model training without changing the model logic. I also make the train data expansion respect `seq_scored` (instead of hard-coding 68) so the pipeline is robust and consistent with the dataset metadata while keeping the same supervised targets and features. Finally, I keep the same architecture/training loop and ensure the submission rows align exactly to `sample_submission.csv` and that any missing predictions are safely filled.'
- What this solution (achieved 0.45507) has done: 'I fix the TensorFlow/protobuf `MessageFactory.GetPrototype` crash by forcing the pure-Python protobuf implementation *and* ensuring the already-imported `google.protobuf` module is reloaded from that implementation before TensorFlow is imported. This is a minimal, execution-unblocking change that doesn’t alter the model/feature logic. I also keep all imports strictly `tf.keras` and add a small safety fallback so the notebook still produces a valid `submission.csv` even if training fails (it write zeros rather than crash). No score-tuning changes are introduced beyond making the pipeline reliably run end-to-end.'
- What this solution (achieved 0.45386) has done: 'I fix the TensorFlow/protobuf crash by forcing the pure-Python protobuf runtime *before* any protobuf/TensorFlow-related imports and by reloading `google.protobuf` cleanly, which unblocks training instead of falling back to all-zeros predictions. I also add a deterministic seed to stabilize training behavior without changing the model, features, or training loop semantics. Finally, I keep the existing feature engineering/model architecture intact and ensure the submission is still written in the exact `sample_submission.csv` row order with the required columns.'
- What this solution (achieved 0.45509) has done: 'I fix the TensorFlow/protobuf `MessageFactory.GetPrototype` crash by forcing the pure-Python protobuf implementation *and* reloading protobuf cleanly before importing TensorFlow, which is the root execution blocker preventing training and causing zero-pred fallbacks. I also make the training/test row expansion slightly faster/safer without changing the feature set (still one-hot of sequence/structure/loop_type) or the model/training loop. Finally, I keep the same submission construction but add a strict alignment check against `sample_submission.csv` to guarantee correct row order and required columns, ensuring a valid `submission.csv` is always produced. This should restore the trained-model predictions (improving score from the current degraded state caused by the crash) while preserving the core logic.'
- What this solution (achieved 0.45603) has done: 'I fix the TensorFlow/protobuf crash (`MessageFactory` has no `GetPrototype`) by forcing the pure-Python protobuf runtime *and* patching the missing method in a minimal, TF-safe way before importing TensorFlow. This unblocks training so you stop falling back to all-zero predictions, which should move the MCRMSE downward toward the target (lower is better) without changing the model architecture, features, or training loop. I also make the protobuf reload step deterministic and ensure we always write a valid `submission.csv` aligned exactly to `sample_submission.csv`. No score-tuning beyond restoring intended training/inference behavior is introduced.'
- What this solution (achieved 0.45427) has done: 'I fix the TensorFlow/protobuf crash by applying the `MessageFactory.GetPrototype` patch earlier and more robustly, before any TensorFlow import, and by patching both `google.protobuf.message_factory` and `google.protobuf.internal.message_factory` (TF may import either). This should unblock training (removing the current all-zero/failed-training fallback behavior), which should reduce MCRMSE (lower is better) toward your target. I keep the same feature engineering, model architecture, optimizer, loss, and training loop, only changing the protobuf setup order/coverage and making sure the run always writes a valid `submission.csv` in the exact `sample_submission.csv` row order.'
- What this solution (achieved 0.45571) has done: 'I fix the TensorFlow/protobuf `MessageFactory.GetPrototype` crash by ensuring the pure-Python protobuf runtime is forced early and by patching `GetPrototype` on the actual `MessageFactory` class objects that TensorFlow may use (including any already-imported/internal variants) before importing TensorFlow. This unblocks model training so you no longer fall back to all-zero predictions, which should reduce MCRMSE (lower is better) toward your target while preserving the same features, model, loss, optimizer, and training loop. I also keep the submission construction identical but add a small robustness check so the `id_seqpos` alignment is guaranteed even if any ordering changes occur. No score-changing modeling edits are introduced beyond restoring intended training/inference execution.'
- What this solution (achieved 0.4543) has done: 'I fix the TensorFlow/protobuf crash by forcing the pure-Python protobuf implementation and patching `MessageFactory.GetPrototype` *before* any TensorFlow import, including patching both the public and internal protobuf message_factory modules and their concrete class. This is the root blocker currently preventing training and causing the pipeline to fail (and/or fall back to zeros), so unblocking it should legitimately improve MCRMSE toward your target without changing the model/feature/training semantics. I also keep all `keras` usage strictly via `tf.keras` to avoid mixed-keras/protobuf import paths. Finally, I keep the submission construction identical but ensure it always aligns to `sample_submission.csv` order and writes `submission.csv`.'
- What this solution (achieved 0.45409) has done: 'I fix the protobuf/TensorFlow crash that prevents training by forcing the pure-Python protobuf runtime and patching `MessageFactory.GetPrototype` in a way that definitely executes *before* importing TensorFlow (including patching the already-instantiated factory class). This should restore normal training/inference (instead of failing and producing zero predictions), which reduce MCRMSE toward your target (lower is better) without changing the model architecture, features, loss, or training loop. I also keep the submission construction identical but add one small safety check to guarantee `tdf` and `sample_submission` alignment stays correct. No other score-changing modeling edits are introduced.'
- What this solution (achieved 0.45482) has done: 'I fix the TensorFlow/protobuf crash by applying the `MessageFactory.GetPrototype` patch in a way that definitely affects the exact factory objects TensorFlow uses, including patching the *instance* method (not just the class) and patching both public and internal protobuf message_factory modules before importing TensorFlow. This is an execution-unblocking change that restores real training (instead of failing and writing near-zero predictions), which should move MCRMSE downward toward your target (lower is better) without changing the model/feature/training logic. I also ensure the protobuf module reload happens from the forced pure-Python implementation, then proceed with the same pipeline and keep the submission aligned exactly to `sample_submission.csv`. No model/feature/loop changes are made beyond making training actually run.'
- What this solution (achieved 0.45278) has done: 'I fix the TensorFlow/protobuf `MessageFactory.GetPrototype` crash by applying a more direct patch that targets the *actual* `google.protobuf.message_factory.MessageFactory` class and internal variants, and by also patching `symbol_database.Default().pool._InternalCreateMessageClass` compatibility when present, all strictly before importing TensorFlow. This is an execution-unblocking fix (not a modeling change) that allow training to run instead of failing and producing zero predictions, which should reduce MCRMSE toward your target (lower is better). I keep the feature engineering, model architecture, loss, optimizer, and training loop unchanged, and ensure the submission is still written as `submission.csv` in the exact `sample_submission.csv` row order with the required columns.'
- What this solution (achieved 0.45402) has done: 'I fix the TensorFlow/protobuf crash by forcing the pure-Python protobuf runtime and patching `MessageFactory.GetPrototype` in a way that runs before *any* TensorFlow import and also patches both public and internal protobuf factories (including already-created instances). This is an execution-unblocking change only; it preserves your feature engineering, model architecture, loss, optimizer, and training loop. I also add a tiny safety check to confirm TensorFlow imports successfully right after patching, so the run fails early with a clear error instead of failing mid-training. Finally, I keep submission creation identical but ensure we always write `submission.csv` with the exact sample submission order and required columns.'
- What this solution (achieved 0.45291) has done: 'I fix the TensorFlow/protobuf crash by ensuring the `MessageFactory.GetPrototype` patch is applied to the exact `MessageFactory` class object that TensorFlow imports, including reloading protobuf modules in a clean order before importing TensorFlow. This is an execution-unblocking change only; it keeps the same features, model architecture, loss, optimizer, and training loop. Once training runs, predictions come from the trained model instead of the current fallback/failed state, which should move MCRMSE downward (better) toward your target. I also keep the submission generation logic the same but add a strict early TF import check so failures happen before long preprocessing.'

# 9. Code solution

## === cell 0
import os
import sys
import warnings
import importlib

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"

os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

for k in list(sys.modules.keys()):
    if k.startswith("google.protobuf"):
        del sys.modules[k]


def _patch_getprototype_on_class(cls):
    """Ensure MessageFactory has GetPrototype, mapping to GetMessageClass when available."""
    if cls is None:
        return False
    if hasattr(cls, "GetPrototype"):
        return False
    if hasattr(cls, "GetMessageClass"):

        def _GetPrototype(self, descriptor):
            return self.GetMessageClass(descriptor)

        setattr(cls, "GetPrototype", _GetPrototype)
        return True
    return False


def _patch_getprototype_on_instance(obj):
    """Patch GetPrototype on a specific factory instance if missing."""
    if obj is None:
        return False
    if hasattr(obj, "GetPrototype"):
        return False
    if hasattr(obj, "GetMessageClass"):

        def _GetPrototype(descriptor, _obj=obj):
            return _obj.GetMessageClass(descriptor)

        try:
            setattr(obj, "GetPrototype", _GetPrototype)
            return True
        except Exception:
            return False
    return False


def _reload_protobuf_clean():
    gp = importlib.import_module("google.protobuf")
    importlib.reload(gp)
    mf = importlib.import_module("google.protobuf.message_factory")
    importlib.reload(mf)
    try:
        imf = importlib.import_module("google.protobuf.internal.message_factory")
        importlib.reload(imf)
    except Exception:
        imf = None
    return mf, imf


def _patch_getprototype_everywhere():
    patched = False
    try:
        mf, imf = _reload_protobuf_clean()

        if hasattr(mf, "MessageFactory"):
            patched |= _patch_getprototype_on_class(mf.MessageFactory)
        if imf is not None and hasattr(imf, "MessageFactory"):
            patched |= _patch_getprototype_on_class(imf.MessageFactory)

        try:
            from google.protobuf import symbol_database as _symbol_database

            _db = _symbol_database.Default()
            for attr in ["_factory", "factory", "_message_factory"]:
                patched |= _patch_getprototype_on_instance(getattr(_db, attr, None))
            if hasattr(_db, "pool"):
                patched |= _patch_getprototype_on_instance(
                    getattr(_db.pool, "_message_factory", None)
                )
        except Exception:
            pass

        return patched
    except Exception:
        return patched


patched_any = _patch_getprototype_everywhere()
if patched_any:
    print("[protobuf patch] Patched MessageFactory.GetPrototype for TF compatibility.")
else:
    print("[protobuf patch] No patch needed (or could not patch).")

warnings.simplefilter(action="ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(0)

train = pd.read_json("../input/stanford-covid-vaccine/train.json", lines=True)
test = pd.read_json("../input/stanford-covid-vaccine/test.json", lines=True)
sub = pd.read_csv("../input/stanford-covid-vaccine/sample_submission.csv")


def plotd(f1, f2):
    try:
        plt.style.use("seaborn-v0_8")
        sns.set_style("whitegrid")
        fig = plt.figure(figsize=(15, 5))
        plt.subplot2grid((1, 2), (0, 0))
        plt.hist(a[f1], bins=20, color="black", alpha=0.5)
        plt.title(f"{f1}", weight="bold", fontsize=18)
        plt.subplot2grid((1, 2), (0, 1))
        plt.hist(a[f2], bins=20, color="crimson", alpha=0.5)
        plt.title(f"{f2}", weight="bold", fontsize=18)
        plt.show()
    except Exception as e:
        print(f"[plotd skipped] {e}")


def plotc(f1, f2):
    try:
        plt.style.use("seaborn-v0_8")
        sns.set_style("whitegrid")
        fig = plt.figure(figsize=(15, 5))
        plt.subplot2grid((1, 2), (0, 0))
        if pd.api.types.is_numeric_dtype(a[f1]):
            plt.hist(a[f1], bins=20, color="black", alpha=0.7)
        else:
            sns.countplot(x=a[f1], color="black")
        plt.title(f"{f1}", weight="bold", fontsize=18)

        plt.subplot2grid((1, 2), (0, 1))
        if pd.api.types.is_numeric_dtype(a[f2]):
            plt.hist(a[f2], bins=20, color="crimson", alpha=0.7)
        else:
            sns.countplot(x=a[f2], color="crimson")
        plt.title(f"{f2}", weight="bold", fontsize=18)
        plt.xticks(weight="bold", rotation=0)
        plt.show()
    except Exception as e:
        print(f"[plotc skipped] {e}")


def ploth(data, w=15, h=9):
    try:
        plt.figure(figsize=(w, h))
        num = data.select_dtypes(include=[np.number])
        if num.shape[1] == 0:
            print("[ploth skipped] no numeric columns to correlate.")
            return
        sns.heatmap(num.corr(), cmap="hot", annot=False)
        plt.title("Correlation between the features", fontsize=18, weight="bold")
        plt.xticks(weight="bold")
        plt.yticks(weight="bold")
        plt.show()
    except Exception as e:
        print(f"[ploth skipped] {e}")




## === cell 1
train.head()




## === cell 2
def length(feature):
    column = train[[feature]].copy()
    column["length"] = column[feature].apply(len)
    return column.head()


length("sequence")



## === cell 3
length("reactivity")



## === cell 4
train_data = []
for row in train.itertuples(index=False):
    seq_scored = int(row.seq_scored)
    seq = row.sequence
    struct = row.structure
    loop = row.predicted_loop_type
    for i in range(seq_scored):
        train_data.append(
            (
                row.id,
                seq[i],
                struct[i],
                loop[i],
                row.reactivity[i],
                row.reactivity_error[i],
                row.deg_Mg_pH10[i],
                row.deg_error_Mg_pH10[i],
                row.deg_pH10[i],
                row.deg_error_pH10[i],
                row.deg_Mg_50C[i],
                row.deg_error_Mg_50C[i],
                row.deg_50C[i],
                row.deg_error_50C[i],
            )
        )



## === cell 5
a = pd.DataFrame(
    train_data,
    columns=[
        "id",
        "sequence",
        "structure",
        "predicted_loop_type",
        "reactivity",
        "reactivity_error",
        "deg_Mg_pH10",
        "deg_error_Mg_pH10",
        "deg_pH10",
        "deg_error_pH10",
        "deg_Mg_50C",
        "deg_error_Mg_50C",
        "deg_50C",
        "deg_error_50C",
    ],
)
a.head()



## === cell 6
plotd("reactivity", "reactivity_error")



## === cell 7
plotd("deg_50C", "deg_Mg_50C")



## === cell 8
plotd("deg_pH10", "deg_Mg_pH10")



## === cell 9
plotc("predicted_loop_type", "structure")



## === cell 10
try:
    plt.style.use("seaborn-v0_8")
    sns.set_style("whitegrid")
    sns.countplot(x="sequence", data=a, palette="terrain")
    plt.title("Nucleotides count per sequence", weight="bold", fontsize=12)
    plt.show()
except Exception as e:
    print(f"[countplot skipped] {e}")



## === cell 11
b = a[["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]]
ploth(b, 10, 4)



## === cell 12
c = a[["id", "sequence", "structure", "predicted_loop_type"]].copy()
c = pd.get_dummies(c, columns=["sequence", "structure", "predicted_loop_type"])
ploth(c, 12, 6)



## === cell 13
test_data = []
for row in test.itertuples(index=False):
    seq_len = int(row.seq_length)
    seq = row.sequence
    struct = row.structure
    loop = row.predicted_loop_type
    for i in range(seq_len):
        test_data.append((f"{row.id}_{i}", seq[i], struct[i], loop[i]))

tdf = pd.DataFrame(
    test_data, columns=["id", "sequence", "structure", "predicted_loop_type"]
)
X_test_raw = pd.get_dummies(
    tdf, columns=["sequence", "structure", "predicted_loop_type"]
)

X_train = c.drop("id", axis=1)
X_test = X_test_raw.drop("id", axis=1)

X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)
y_train = b.astype(np.float32)

print("X_train:", X_train.shape, "y_train:", y_train.shape, "X_test:", X_test.shape)

assert (
    tdf.shape[0] == sub.shape[0]
), "tdf rows must equal sample_submission rows (107 * #test_ids)."



## === cell 14
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

tf.random.set_seed(0)

_ = tf.constant([0.0])


def build_model(n_inputs, n_outputs):
    model = Sequential()
    model.add(
        layers.Dense(
            1024,
            input_shape=(n_inputs,),
            kernel_initializer="he_uniform",
            activation="relu",
        )
    )
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(512, activation="relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(256, activation="relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dense(n_outputs))
    model.compile(loss="mae", optimizer="adam")
    return model


callbacks = [
    ReduceLROnPlateau(),
    ModelCheckpoint("model.keras", save_best_only=True, monitor="val_loss"),
]

n_inputs, n_outputs = X_train.shape[1], y_train.shape[1]
model = build_model(n_inputs, n_outputs)

history = None
training_ok = True
try:
    history = model.fit(
        X_train,
        y_train,
        batch_size=128,
        epochs=100,
        callbacks=callbacks,
        validation_split=0.3,
        verbose=2,
    )

    if tf.io.gfile.exists("model.keras"):
        try:
            model = tf.keras.models.load_model("model.keras")
            print("Loaded best checkpoint: model.keras")
        except Exception as e:
            print(f"[checkpoint load skipped] {e}")
except Exception as e:
    training_ok = False
    print(f"[training skipped due to error] {type(e).__name__}: {e}")



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 15
try:
    plt.style.use("seaborn-v0_8")
    sns.set_style("whitegrid")
    if history is None:
        raise ValueError("No training history available.")
    fig = plt.figure(figsize=(15, 5))
    train_loss = history.history["loss"]
    test_loss = history.history["val_loss"]
    x = list(range(1, len(test_loss) + 1))
    plt.plot(x, test_loss, color="cyan", label="Val loss")
    plt.plot(x, train_loss, label="Train loss")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.title("Loss vs. Epoch", weight="bold", fontsize=18)
    plt.show()
except Exception as e:
    print(f"[loss plot skipped] {e}")



## === cell 16
if training_ok:
    test_preds = model.predict(X_test, batch_size=2048, verbose=1)
else:
    test_preds = np.zeros((X_test.shape[0], y_train.shape[1]), dtype=np.float32)



## === cell 17
pred_df = pd.DataFrame(test_preds, columns=y_train.columns)
pred_df["id_seqpos"] = tdf["id"].values  # already id_seqpos format (id_i)

sub1 = sub.merge(pred_df, on="id_seqpos", how="left", suffixes=("_x", ""))
for col in ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]:
    if f"{col}_x" in sub1.columns:
        sub1 = sub1.drop(columns=[f"{col}_x"])

required_cols = [
    "id_seqpos",
    "reactivity",
    "deg_Mg_pH10",
    "deg_pH10",
    "deg_Mg_50C",
    "deg_50C",
]
for col in required_cols:
    if col not in sub1.columns:
        sub1[col] = 0.0

sub1[required_cols[1:]] = sub1[required_cols[1:]].astype(np.float32)
submission = sub1[required_cols].fillna(0.0)

submission = sub[["id_seqpos"]].merge(submission, on="id_seqpos", how="left")
submission[required_cols[1:]] = (
    submission[required_cols[1:]].fillna(0.0).astype(np.float32)
)

assert (
    submission.shape[0] == sub.shape[0]
), "Submission row count mismatch vs sample_submission."
assert submission.columns.tolist() == required_cols, "Submission columns mismatch."

submission.head()



## === cell 18
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print("Columns:", submission.columns.tolist())
print(submission.tail())
