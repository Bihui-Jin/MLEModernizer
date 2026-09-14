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

0.40854

# 6. Current score

0.25445

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.25539) has done: 'I fix the TensorFlow/Keras import crash by forcing the pure-TensorFlow protobuf implementation before importing TensorFlow, which avoids the `MessageFactory` error in this Kaggle environment. I also make BPPS feature loading robust to missing `bpps/` files by substituting safe zero-valued features (so preprocessing and the model input shape stay identical and the pipeline can run end-to-end). Next, I replace the invalid `tf.reshape` on a KerasTensor with an equivalent Keras layer (`Reshape`), which preserves the model architecture but makes it Functional-API compatible. Finally, I ensure test prediction covers the actual `seq_length` (107 here) and write a correctly formatted `submission.csv` matching `sample_submission.csv` rows/columns.'
- What this solution (achieved 0.25662) has done: 'I fix the TensorFlow import crash by setting the protobuf env var *before* any TensorFlow/Keras-related import and by using the compatible `google.protobuf` API fallback if needed, so the notebook runs end-to-end in this Kaggle environment. I also correct a small logic bug in the negative-error scan loop (it was always checking row 0), which is score-neutral but fixes incorrect diagnostics. Finally, I keep the existing model/training/inference core logic unchanged and ensure the submission is always fully populated and written as a valid `submission.csv` matching `sample_submission.csv` rows/columns.'
- What this solution (achieved 0.25434) has done: 'I fix the TensorFlow/protobuf crash by pinning protobuf to the pure-Python implementation before importing any TF/Keras modules and adding a small compatibility patch for the missing `MessageFactory.GetPrototype` method seen in this environment. I also make the Keras loss import compatible with Keras 3 by using `tf.keras.losses.mean_squared_error` (or a safe fallback) to prevent runtime errors during `model.compile`. These are execution/stability fixes and should be score-neutral, preserving your model, training loop, and submission formatting exactly as-is. The rest of the pipeline (feature building, model architecture, training, inference, and CSV writing) is kept unchanged.'
- What this solution (achieved 0.25593) has done: 'You’re hitting the protobuf `MessageFactory.GetPrototype` crash during TensorFlow import; the prior patch checks the class but the error arises from an instance method missing inside protobuf’s runtime. I make the protobuf compatibility patch robust by monkey‑patching both the class and the default instance method (when available) before importing TensorFlow, so TF loads cleanly. I keep the model, preprocessing, training loop, and submission formatting unchanged to preserve the current score behavior (only stability fixes). Finally, I also ensure the environment variables are set at the very top before any TF/Keras-related import to avoid the crash deterministically.'
- What this solution (achieved 0.25336) has done: 'I fix the TensorFlow/protobuf import crash by applying a more robust protobuf compatibility monkey‑patch before importing TensorFlow (patching both the class and instance behavior so TF won’t call a missing `GetPrototype`). This is an execution/stability fix only and won’t change your model/training logic or predictions aside from letting the notebook run. I also keep all paths, preprocessing, model architecture, training loop, and submission formatting exactly the same to preserve score behavior (your current score is already better than the target for a lower-is-better metric). Finally, I ensure the script always writes a valid `submission.csv` matching `sample_submission.csv`.'
- What this solution (achieved 0.25708) has done: 'You’re failing before any training because TensorFlow import triggers a protobuf runtime call to `MessageFactory().GetPrototype`, and your patch only adds a class-level alias (it doesn’t reliably patch the instance method TF ends up using). I make the protobuf patch robust by explicitly adding `GetPrototype` to both the `MessageFactory` class and the already-instantiated default factory object (when present), before importing TensorFlow; this is an execution fix and should be score-neutral once it runs. I also keep the rest of the pipeline (data loading, preprocessing, model, training loop, inference, submission formatting) unchanged to preserve the current score behavior relative to your target. Finally, I ensure the script always writes `submission.csv` with the exact required columns and row count.'
- What this solution (achieved 0.25812) has done: 'The crash happens during TensorFlow import because TF calls `google.protobuf.message_factory.MessageFactory().GetPrototype()` but the installed protobuf version removed/changed that method; the current patch doesn’t reliably add it to the *instance* TF ends up using. I patch protobuf more robustly by adding `GetPrototype` both to the `MessageFactory` class and to the already-created default factory instance used internally, before importing TensorFlow. This is an execution-only fix and keeps the model, preprocessing, training loop, and submission formatting unchanged, so the score behavior should remain essentially the same (already within ±10% of the 0.40854 target for a lower-is-better metric). The script then run end-to-end and write a valid `submission.csv` with the correct shape/columns.'
- What this solution (achieved 0.25513) has done: 'I fix the TensorFlow import crash by forcing the pure-Python protobuf implementation and adding a more robust protobuf compatibility patch that also covers the `MessageFactory` instance TF uses internally (this is the current blocker). I keep the model, preprocessing, training loop, and prediction logic unchanged so the score behavior stays essentially the same (your current score is already better than the target band for a lower-is-better metric). I also keep all paths and the submission-writing logic intact, ensuring `submission.csv` is always produced with the exact required columns and row count. No score-optimization changes are introduced—this is primarily an execution/stability fix.'
- What this solution (achieved 0.25445) has done: 'I fix the immediate runtime blocker: TensorFlow is crashing on import due to a protobuf API mismatch where TF expects `MessageFactory.GetPrototype`. The minimal, score-neutral fix is to force the pure-Python protobuf implementation and monkey-patch `MessageFactory.GetPrototype` onto both the class and any default factory instances *before* importing TensorFlow, ensuring the patch is applied even if TF imports protobuf internals during startup. I keep the model/training/inference logic unchanged, and only touch the protobuf patch cell plus minor safety around env var placement so the notebook runs end-to-end. The submission-writing logic remain the same and still output a valid `submission.csv` with the required columns/row count.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames[:50]:
        print(os.path.join(dirname, filename))



## === cell 1
import json

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"


def _patch_protobuf_getprototype():
    try:
        from google.protobuf import message_factory as _mf  # noqa: F401
        from google.protobuf import symbol_database as _sym_db
    except Exception as e:
        print("Warning: could not import google.protobuf to patch:", repr(e))
        return

    try:
        from google.protobuf import message_factory as _mf

        if not hasattr(_mf, "MessageFactory"):
            return

        cls = _mf.MessageFactory

        def _get_prototype(self, descriptor):
            if hasattr(self, "GetMessageClass"):
                return self.GetMessageClass(descriptor)
            try:
                _db = _sym_db.Default()
                return _db.GetPrototype(descriptor)
            except Exception:
                raise AttributeError(
                    "GetPrototype is unavailable and no fallback worked."
                )

        if not hasattr(cls, "GetPrototype"):
            try:
                cls.GetPrototype = _get_prototype  # type: ignore[attr-defined]
            except Exception:
                pass

        def _ensure_instance(obj):
            if obj is None:
                return
            if not hasattr(obj, "GetPrototype"):
                try:
                    obj.GetPrototype = obj.GetMessageClass  # type: ignore[attr-defined]
                except Exception:
                    try:
                        obj.GetPrototype = _get_prototype.__get__(obj, obj.__class__)  # type: ignore[attr-defined]
                    except Exception:
                        pass

        try:
            _ensure_instance(cls())
        except Exception:
            pass

        for attr in (
            "Default",
            "_DEFAULT_FACTORY",
            "default_factory",
            "_default_factory",
        ):
            try:
                if hasattr(_mf, attr):
                    candidate = getattr(_mf, attr)
                    candidate = (
                        candidate()
                        if callable(candidate) and attr == "Default"
                        else candidate
                    )
                    _ensure_instance(candidate)
            except Exception:
                pass

        try:
            _db = _sym_db.Default()
            if hasattr(_db, "_factory"):
                _ensure_instance(getattr(_db, "_factory"))
        except Exception:
            pass

    except Exception as e:
        print("Warning: protobuf patch issue (continuing):", repr(e))


_patch_protobuf_getprototype()

import tensorflow as tf
from matplotlib import pyplot as plt

print("TensorFlow:", tf.__version__)



## === cell 2
os.chdir("/kaggle/")
os.getcwd()



## === cell 3
train_data = pd.read_json("/kaggle/input/stanford-covid-vaccine/train.json", lines=True)
test_data = pd.read_json("/kaggle/input/stanford-covid-vaccine/test.json", lines=True)
submission_format = pd.read_csv(
    "/kaggle/input/stanford-covid-vaccine/sample_submission.csv", encoding="utf-8-sig"
)



## === cell 4
train_data.head()



## === cell 5
train_data.shape



## === cell 6
train_data.groupby(["SN_filter"]).size()



## === cell 7
test_data.head()



## === cell 8
test_data.shape



## === cell 9
submission_format.head()



## === cell 10
print(train_data.shape)
print(test_data.shape)
print(submission_format.shape)



## === cell 11
print("Training data:\n", train_data["seq_scored"].value_counts())
print("Test data:\n", test_data["seq_scored"].value_counts())
len(train_data["reactivity"].iloc[0])



## === cell 12
len(train_data["sequence"].iloc[0])



## === cell 13
flag = False
for i in range(0, len(train_data)):
    if (
        ([x < 0 for x in train_data["reactivity_error"].iloc[i]].count(True) > 0)
        | ([x < 0 for x in train_data["deg_error_Mg_pH10"].iloc[i]].count(True) > 0)
        | ([x < 0 for x in train_data["deg_error_pH10"].iloc[i]].count(True) > 0)
        | ([x < 0 for x in train_data["deg_error_Mg_50C"].iloc[i]].count(True) > 0)
        | ([x < 0 for x in train_data["deg_error_50C"].iloc[i]].count(True) > 0)
    ):
        flag = True
        break
print(flag)



## === cell 14
train_data.columns



## === cell 15
token2int = {x: i for i, x in enumerate("().ACGUBEHIMSX")}
target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]



## === cell 16
token2int




## === cell 17
def _bpps_path(mol_id: str) -> str:
    return f"/kaggle/input/stanford-covid-vaccine/bpps/{mol_id}.npy"


def read_bpps_sum(df):
    bpps_arr = []
    for mol_id, seq_len in zip(df.id.to_list(), df.seq_length.to_list()):
        p = _bpps_path(mol_id)
        if os.path.exists(p):
            bpps_arr.append(np.load(p).sum(axis=1).astype(np.float32))
        else:
            bpps_arr.append(np.zeros((seq_len,), dtype=np.float32))
    return bpps_arr


def read_bpps_max(df):
    bpps_arr = []
    for mol_id, seq_len in zip(df.id.to_list(), df.seq_length.to_list()):
        p = _bpps_path(mol_id)
        if os.path.exists(p):
            bpps_arr.append(np.load(p).max(axis=1).astype(np.float32))
        else:
            bpps_arr.append(np.zeros((seq_len,), dtype=np.float32))
    return bpps_arr


def read_bpps_nb(df):
    bpps_nb_mean = 0.077522
    bpps_nb_std = 0.08914
    bpps_arr = []
    for mol_id, seq_len in zip(df.id.to_list(), df.seq_length.to_list()):
        p = _bpps_path(mol_id)
        if os.path.exists(p):
            bpps = np.load(p).astype(np.float32)
            bpps_nb = (bpps > 0).sum(axis=0) / bpps.shape[0]
            bpps_nb = (bpps_nb - bpps_nb_mean) / bpps_nb_std
            bpps_arr.append(bpps_nb.astype(np.float32))
        else:
            bpps_arr.append(np.zeros((seq_len,), dtype=np.float32))
    return bpps_arr


os.chdir("/kaggle/working/")
train_data["bpps_sum"] = read_bpps_sum(train_data)
test_data["bpps_sum"] = read_bpps_sum(test_data)
train_data["bpps_max"] = read_bpps_max(train_data)
test_data["bpps_max"] = read_bpps_max(test_data)
train_data["bpps_nb"] = read_bpps_nb(train_data)
test_data["bpps_nb"] = read_bpps_nb(test_data)

train_data.head()




## === cell 18
def preprocess_inputs(df, cols=["sequence", "structure", "predicted_loop_type"]):
    base_fea = np.transpose(
        np.array(
            df[cols].applymap(lambda seq: [token2int[x] for x in seq]).values.tolist()
        ),
        (0, 2, 1),
    ).astype(np.int32)

    bpps_sum_fea = np.array(df["bpps_sum"].to_list(), dtype=np.float32)[
        :, :, np.newaxis
    ]
    bpps_max_fea = np.array(df["bpps_max"].to_list(), dtype=np.float32)[
        :, :, np.newaxis
    ]
    bpps_nb_fea = np.array(df["bpps_nb"].to_list(), dtype=np.float32)[:, :, np.newaxis]

    out = np.concatenate([base_fea, bpps_sum_fea, bpps_max_fea, bpps_nb_fea], 2).astype(
        np.float32
    )
    return out




## === cell 19
train_inputs = preprocess_inputs(train_data.loc[train_data["signal_to_noise"] > 1])
train_labels = np.array(
    train_data.loc[train_data["signal_to_noise"] > 1][target_cols].values.tolist(),
    dtype=np.float32,
).transpose((0, 2, 1))

print("train_inputs:", train_inputs.shape, train_inputs.dtype)
print("train_labels:", train_labels.shape, train_labels.dtype)



## === cell 20
train_data.loc[[0]]



## === cell 21
preprocess_inputs(train_data.loc[[0]]).shape



## === cell 22
test_data.head()



## === cell 23
try:
    mean_squared_error = tf.keras.losses.mean_squared_error
except Exception:

    def mean_squared_error(y_true, y_pred):
        return tf.reduce_mean(tf.square(y_true - y_pred), axis=-1)


def root_mean_squared_error(y_true, y_pred):
    return tf.sqrt(mean_squared_error(y_true, y_pred))


def MCRMSE(y_true, y_pred):
    colwise_mse = tf.reduce_mean(tf.square(y_true - y_pred), axis=1)
    return tf.reduce_mean(tf.sqrt(colwise_mse), axis=1)


def lstm_layer(hidden_dim, dropout):
    return tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            hidden_dim,
            dropout=dropout,
            return_sequences=True,
            kernel_initializer="orthogonal",
        )
    )


def build_model(seq_len=107, embed_dim=100, hidden_dim=256, dropout=0.2, pred_len=68):
    inputs = tf.keras.layers.Input(shape=(seq_len, 6))
    categorical_feats = inputs[:, :, :3]
    numerical_feats = inputs[:, :, 3:]

    embed = tf.keras.layers.Embedding(input_dim=len(token2int), output_dim=embed_dim)(
        categorical_feats
    )

    reshaped = tf.keras.layers.Reshape((seq_len, 3 * embed_dim))(embed)

    reshaped = tf.keras.layers.Concatenate(axis=2)([reshaped, numerical_feats])

    LSTM_layer = lstm_layer(hidden_dim, dropout)(reshaped)
    truncated = LSTM_layer[:, :pred_len]
    out = tf.keras.layers.Dense(5, activation="linear")(truncated)

    model = tf.keras.Model(inputs=inputs, outputs=out)
    adam = tf.optimizers.Adam()
    model.compile(optimizer=adam, loss=MCRMSE)
    return model




## === cell 24
EPOCHS = 100
BATCH_SIZE = 32

model_on_train_data = build_model(seq_len=train_inputs.shape[1], pred_len=68)
model_on_train_data.summary()

ckpt_path = "/kaggle/working/lstm_model.weights.h5"
model_callback = tf.keras.callbacks.ModelCheckpoint(
    ckpt_path,
    save_weights_only=True,
    save_best_only=True,
    monitor="loss",
    mode="min",
    verbose=0,
)

history = model_on_train_data.fit(
    train_inputs,
    train_labels,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    verbose=2,
    callbacks=[model_callback],
)



## === cell 25
print(f"LSTM min training loss: {min(history.history['loss'])}")

fig, ax = plt.subplots(1, 1, figsize=(20, 6))
ax.plot(history.history["loss"])
ax.set_title("LSTM Model - Training Loss")
ax.set_ylabel("Loss")
ax.set_xlabel("Epoch")
plt.show()



## === cell 26
public_df = test_data.query("seq_length == 107").copy()
private_df = test_data.query("seq_length == 130").copy()  # will likely be empty here

public_inputs = preprocess_inputs(public_df)
private_inputs = preprocess_inputs(private_df) if len(private_df) else None

print("public_df:", public_df.shape, "private_df:", private_df.shape)



## === cell 27
model_on_test_data_public = build_model(seq_len=107, pred_len=107)
if os.path.exists(ckpt_path):
    model_on_test_data_public.load_weights(ckpt_path)
pred_test_data_public = model_on_test_data_public.predict(
    public_inputs, batch_size=64, verbose=0
)

pred_test_data_private = None
if private_inputs is not None and len(private_df):
    model_on_test_data_private = build_model(seq_len=130, pred_len=130)
    if os.path.exists(ckpt_path):
        model_on_test_data_private.load_weights(ckpt_path)
    pred_test_data_private = model_on_test_data_private.predict(
        private_inputs, batch_size=64, verbose=0
    )

print("pred_test_data_public:", pred_test_data_public.shape)




## === cell 28
def format_predictions(public_preds, private_preds):
    preds = []

    for df, preds_ in [(public_df, public_preds), (private_df, private_preds)]:
        if df is None or len(df) == 0 or preds_ is None:
            continue
        for i, uid in enumerate(df.id):
            single_pred = preds_[i]
            single_df = pd.DataFrame(single_pred, columns=target_cols)
            single_df["id_seqpos"] = [f"{uid}_{x}" for x in range(single_df.shape[0])]
            preds.append(single_df)

    return (
        pd.concat(preds).reset_index(drop=True)
        if len(preds)
        else pd.DataFrame(columns=["id_seqpos"] + target_cols)
    )




## === cell 29
lstm_preds = format_predictions(pred_test_data_public, pred_test_data_private)
lstm_preds.head()



## === cell 30
submission = submission_format[["id_seqpos"]].merge(
    lstm_preds, how="left", on="id_seqpos"
)

for c in target_cols:
    if c not in submission.columns:
        submission[c] = 0.0
submission[target_cols] = submission[target_cols].fillna(0.0)

submission.shape, submission.head()



## === cell 31
os.chdir("/kaggle/working/")
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print("Wrote:", submission_path, "rows:", len(submission), "cols:", submission.shape[1])
print(submission.head())
