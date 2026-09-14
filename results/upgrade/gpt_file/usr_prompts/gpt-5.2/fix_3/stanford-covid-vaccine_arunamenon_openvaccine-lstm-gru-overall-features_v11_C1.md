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

0.38042

# 6. Current score

0.25967

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 0.25609) has done: 'I fix the TensorFlow/protobuf import crash by forcing the pure-Python protobuf implementation before importing TensorFlow. Then I make the BPPS feature loading robust to missing `bpps/*.npy` files by substituting zeros with correct shapes so preprocessing can proceed and the model input feature count stays consistent. Next I replace the invalid `tf.reshape` on a KerasTensor with a Keras `Reshape`/`Lambda` layer so the functional model builds and trains. Finally, I ensure test inference runs on the actual 107-length test set (no 130 split for this dataset) and always writes a valid `submission.csv` in `/kaggle/working/` with the required columns and row order matching `sample_submission.csv`.'
- What this solution (achieved 0.25967) has done: 'I fix the TensorFlow/protobuf crash by setting both the pure-Python protobuf implementation and a compatible protobuf version environment variable before importing TensorFlow, and by avoiding optional imports that trigger the bad code path. I keep the model and training logic identical, but correct one inference-time bug: the model currently predicts 107 positions, while training labels are only 68; I set `pred_len=68` for test-time models and then pad predictions to 107 so submission rows are fully populated and consistent. This change is score-improving (lower is better) because it avoids untrained extrapolation for positions 68–106 and preserves the trained output semantics for the scored region. Finally, I ensure the submission is written as `/kaggle/working/submission.csv` with correct columns and row order matching `sample_submission.csv`.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"

import numpy as np
import pandas as pd

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames[:50]:
        print(os.path.join(dirname, filename))



## === cell 1
import json
import tensorflow as tf

print("TensorFlow:", tf.__version__)
print("Eager:", tf.executing_eagerly())



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

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
print(flag)



## === cell 14
min_reactivity_value = min(train_data["reactivity"].iloc[0])
min_deg_Mg_pH10_value = min(train_data["deg_Mg_pH10"].iloc[0])
min_deg_pH10_value = min(train_data["deg_pH10"].iloc[0])
min_deg_Mg_50C_value = min(train_data["deg_Mg_50C"].iloc[0])
min_deg_50C_value = min(train_data["deg_50C"].iloc[0])

for i in range(0, len(train_data)):
    if min(train_data["reactivity"].iloc[i]) < min_reactivity_value:
        min_reactivity_value = min(train_data["reactivity"].iloc[i])

    if min(train_data["deg_Mg_pH10"].iloc[i]) < min_deg_Mg_pH10_value:
        min_deg_Mg_pH10_value = min(train_data["deg_Mg_pH10"].iloc[i])

    if min(train_data["deg_pH10"].iloc[i]) < min_deg_pH10_value:
        min_deg_pH10_value = min(train_data["deg_pH10"].iloc[i])

    if min(train_data["deg_Mg_50C"].iloc[i]) < min_deg_Mg_50C_value:
        min_deg_Mg_50C_value = min(train_data["deg_Mg_50C"].iloc[i])

    if min(train_data["deg_50C"].iloc[i]) < min_deg_50C_value:
        min_deg_50C_value = min(train_data["deg_50C"].iloc[i])

print(
    min_reactivity_value,
    min_deg_Mg_pH10_value,
    min_deg_pH10_value,
    min_deg_Mg_50C_value,
    min_deg_50C_value,
)



## === cell 15
train_data.columns



## === cell 16
token2int = {x: i for i, x in enumerate("().ACGUBEHIMSX")}
target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]



## === cell 17
token2int



## === cell 18
BPPS_DIR = "/kaggle/input/stanford-covid-vaccine/bpps"


def _safe_load_bpps(mol_id, seq_len):
    path = os.path.join(BPPS_DIR, f"{mol_id}.npy")
    if os.path.exists(path):
        return np.load(path)
    return np.zeros((seq_len, seq_len), dtype=np.float32)


def read_bpps_sum(df):
    bpps_arr = []
    for mol_id, seq_len in zip(df.id.to_list(), df.seq_length.to_list()):
        bpps = _safe_load_bpps(mol_id, seq_len)
        bpps_arr.append(bpps.sum(axis=1))
    return bpps_arr


def read_bpps_max(df):
    bpps_arr = []
    for mol_id, seq_len in zip(df.id.to_list(), df.seq_length.to_list()):
        bpps = _safe_load_bpps(mol_id, seq_len)
        bpps_arr.append(bpps.max(axis=1))
    return bpps_arr


def read_bpps_nb(df):
    bpps_nb_mean = 0.077522
    bpps_nb_std = 0.08914
    bpps_arr = []
    for mol_id, seq_len in zip(df.id.to_list(), df.seq_length.to_list()):
        bpps = _safe_load_bpps(mol_id, seq_len)
        bpps_nb = (bpps > 0).sum(axis=0) / bpps.shape[0]
        bpps_nb = (bpps_nb - bpps_nb_mean) / bpps_nb_std
        bpps_arr.append(bpps_nb.astype(np.float32))
    return bpps_arr


os.chdir("/kaggle/working/")
train_data["bpps_sum"] = read_bpps_sum(train_data)
test_data["bpps_sum"] = read_bpps_sum(test_data)
train_data["bpps_max"] = read_bpps_max(train_data)
test_data["bpps_max"] = read_bpps_max(test_data)
train_data["bpps_nb"] = read_bpps_nb(train_data)
test_data["bpps_nb"] = read_bpps_nb(test_data)

train_data.head()




## === cell 19
def preprocess_inputs(df, cols=["sequence", "structure", "predicted_loop_type"]):
    base_fea = np.transpose(
        np.array(
            df[cols].applymap(lambda seq: [token2int[x] for x in seq]).values.tolist()
        ),
        (0, 2, 1),
    )

    bpps_sum_fea = np.array(df["bpps_sum"].to_list())[:, :, np.newaxis]
    bpps_max_fea = np.array(df["bpps_max"].to_list())[:, :, np.newaxis]
    bpps_nb_fea = np.array(df["bpps_nb"].to_list())[:, :, np.newaxis]

    return np.concatenate(
        [base_fea, bpps_sum_fea, bpps_max_fea, bpps_nb_fea], 2
    ).astype(np.float32)




## === cell 20
from tqdm.notebook import tqdm


def get_structure_adj(train):
    Ss = []
    for i in tqdm(range(len(train))):
        seq_length = train["seq_length"].iloc[i]
        structure = train["structure"].iloc[i]
        sequence = train["sequence"].iloc[i]

        cue = []
        a_structures = {
            ("A", "U"): np.zeros([seq_length, seq_length], dtype=np.float32),
            ("C", "G"): np.zeros([seq_length, seq_length], dtype=np.float32),
            ("U", "G"): np.zeros([seq_length, seq_length], dtype=np.float32),
            ("U", "A"): np.zeros([seq_length, seq_length], dtype=np.float32),
            ("G", "C"): np.zeros([seq_length, seq_length], dtype=np.float32),
            ("G", "U"): np.zeros([seq_length, seq_length], dtype=np.float32),
        }

        for j in range(seq_length):
            if structure[j] == "(":
                cue.append(j)
            elif structure[j] == ")":
                start = cue.pop()
                a_structures[(sequence[start], sequence[j])][start, j] = 1.0
                a_structures[(sequence[j], sequence[start])][j, start] = 1.0

        a_strc = np.stack([a for a in a_structures.values()], axis=2)
        a_strc = np.sum(a_strc, axis=2, keepdims=True)
        Ss.append(a_strc)

    Ss = np.array(Ss, dtype=np.float32)
    print(Ss.shape)
    return Ss




## === cell 21
train_filtered = train_data.loc[train_data["signal_to_noise"] > 1].reset_index(
    drop=True
)
train_inputs = preprocess_inputs(train_filtered)
train_labels = np.array(
    train_filtered[target_cols].values.tolist(), dtype=np.float32
).transpose((0, 2, 1))

Ss = get_structure_adj(train_filtered)
Ss = Ss.sum(axis=1)  # (N, seq_len, 1)
train_inputs = np.concatenate([train_inputs, Ss], 2)  # adds 1 numerical feature

print("train_inputs:", train_inputs.shape)
print("train_labels:", train_labels.shape)



## === cell 22
preprocess_inputs(train_data.loc[[0]]).shape



## === cell 23
test_data.head()




## === cell 24
def MCRMSE(y_true, y_pred):
    colwise_mse = tf.reduce_mean(tf.square(y_true - y_pred), axis=1)  # (batch, 5)
    return tf.reduce_mean(tf.sqrt(colwise_mse + 1e-12), axis=1)  # (batch,)


def lstm_layer(hidden_dim, dropout):
    return tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            hidden_dim,
            dropout=dropout,
            return_sequences=True,
            kernel_initializer="orthogonal",
        )
    )


def gru_layer(hidden_dim, dropout):
    return tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(
            hidden_dim,
            dropout=dropout,
            return_sequences=True,
            kernel_initializer="orthogonal",
        )
    )


def build_model(
    seq_len=107,
    num_features=7,
    embed_dim=100,
    hidden_dim=256,
    dropout=0.2,
    pred_len=68,
    gru_flag=False,
):
    inputs = tf.keras.layers.Input(shape=(seq_len, num_features))

    categorical_feats = inputs[:, :, :3]
    numerical_feats = inputs[:, :, 3:]

    embed = tf.keras.layers.Embedding(input_dim=len(token2int), output_dim=embed_dim)(
        categorical_feats
    )

    reshaped = tf.keras.layers.Reshape((seq_len, 3 * embed_dim))(embed)

    reshaped = tf.keras.layers.Concatenate(axis=2)([reshaped, numerical_feats])
    normalized_layer_1 = tf.keras.layers.BatchNormalization()(reshaped)

    if gru_flag:
        rnn_out = gru_layer(hidden_dim, dropout)(normalized_layer_1)
    else:
        rnn_out = lstm_layer(hidden_dim, dropout)(normalized_layer_1)

    normalized_layer_2 = tf.keras.layers.BatchNormalization()(rnn_out)
    truncated = normalized_layer_2[:, :pred_len]
    out = tf.keras.layers.Dense(5, activation="linear")(truncated)

    model = tf.keras.Model(inputs=inputs, outputs=out)
    adam = tf.optimizers.Adam()
    model.compile(optimizer=adam, loss=MCRMSE)
    return model




## === cell 25
SEQ_LEN = train_inputs.shape[1]
NUM_FEATURES = train_inputs.shape[2]
PRED_LEN = train_labels.shape[1]

EPOCHS = 60
BATCH_SIZE = 32

model_GRU_on_train_data = build_model(
    seq_len=SEQ_LEN, num_features=NUM_FEATURES, pred_len=PRED_LEN, gru_flag=True
)
model_GRU_on_train_data.summary()
model_GRU_callback = tf.keras.callbacks.ModelCheckpoint(
    "GRU_model.weights.h5",
    save_weights_only=True,
    monitor="loss",
    mode="min",
    save_best_only=True,
)

history_GRU = model_GRU_on_train_data.fit(
    train_inputs,
    train_labels,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    verbose=2,
    callbacks=[model_GRU_callback],
)



## === cell 26
model_LSTM_on_train_data = build_model(
    seq_len=SEQ_LEN, num_features=NUM_FEATURES, pred_len=PRED_LEN, gru_flag=False
)
model_LSTM_on_train_data.summary()
model_LSTM_callback = tf.keras.callbacks.ModelCheckpoint(
    "LSTM_model.weights.h5",
    save_weights_only=True,
    monitor="loss",
    mode="min",
    save_best_only=True,
)

history_LSTM = model_LSTM_on_train_data.fit(
    train_inputs,
    train_labels,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    verbose=2,
    callbacks=[model_LSTM_callback],
)



## === cell 27
from matplotlib import pyplot as plt

print(f" LSTM loss: {min(history_LSTM.history['loss'])}")
print(f" GRU loss: {min(history_GRU.history['loss'])}")

fig, ax = plt.subplots(1, 1, figsize=(20, 10))
ax.plot(history_LSTM.history["loss"], label="LSTM")
ax.plot(history_GRU.history["loss"], label="GRU")
ax.set_title("Model - LSTM vs GRU")
ax.set_ylabel("Loss")
ax.set_xlabel("Epoch")
ax.legend()
plt.show()



## === cell 28
seq_lens = sorted(test_data["seq_length"].unique().tolist())
print("Test seq_length unique:", seq_lens)

public_df = test_data.query("seq_length == 107").copy()
private_df = test_data.query("seq_length != 107").copy()  # may be empty

public_inputs = preprocess_inputs(public_df)
Ss_pub = get_structure_adj(public_df).sum(axis=1)
public_inputs = np.concatenate([public_inputs, Ss_pub], 2).astype(np.float32)

if len(private_df) > 0:
    private_inputs = preprocess_inputs(private_df)
    Ss_pri = get_structure_adj(private_df).sum(axis=1)
    private_inputs = np.concatenate([private_inputs, Ss_pri], 2).astype(np.float32)
else:
    private_inputs = None

print(
    "public_inputs:",
    public_inputs.shape,
    "private_inputs:",
    None if private_inputs is None else private_inputs.shape,
)



## === cell 29
TRAIN_PRED_LEN = int(PRED_LEN)  # 68

model_LSTM_on_test_data_public = build_model(
    seq_len=public_inputs.shape[1],
    num_features=public_inputs.shape[2],
    pred_len=TRAIN_PRED_LEN,
    gru_flag=False,
)
model_LSTM_on_test_data_public.load_weights("LSTM_model.weights.h5")
pred_test_data_public_LSTM = model_LSTM_on_test_data_public.predict(
    public_inputs, batch_size=64, verbose=0
)

model_GRU_on_test_data_public = build_model(
    seq_len=public_inputs.shape[1],
    num_features=public_inputs.shape[2],
    pred_len=TRAIN_PRED_LEN,
    gru_flag=True,
)
model_GRU_on_test_data_public.load_weights("GRU_model.weights.h5")
pred_test_data_public_GRU = model_GRU_on_test_data_public.predict(
    public_inputs, batch_size=64, verbose=0
)

if private_inputs is not None:
    model_LSTM_on_test_data_private = build_model(
        seq_len=private_inputs.shape[1],
        num_features=private_inputs.shape[2],
        pred_len=TRAIN_PRED_LEN,
        gru_flag=False,
    )
    model_LSTM_on_test_data_private.load_weights("LSTM_model.weights.h5")
    pred_test_data_private_LSTM = model_LSTM_on_test_data_private.predict(
        private_inputs, batch_size=64, verbose=0
    )

    model_GRU_on_test_data_private = build_model(
        seq_len=private_inputs.shape[1],
        num_features=private_inputs.shape[2],
        pred_len=TRAIN_PRED_LEN,
        gru_flag=True,
    )
    model_GRU_on_test_data_private.load_weights("GRU_model.weights.h5")
    pred_test_data_private_GRU = model_GRU_on_test_data_private.predict(
        private_inputs, batch_size=64, verbose=0
    )
else:
    pred_test_data_private_LSTM = np.zeros((0, TRAIN_PRED_LEN, 5), dtype=np.float32)
    pred_test_data_private_GRU = np.zeros((0, TRAIN_PRED_LEN, 5), dtype=np.float32)

print(pred_test_data_public_LSTM.shape, pred_test_data_public_GRU.shape)




## === cell 30
def _pad_to_seq_len(pred_68, seq_len=107):
    """Pad (N, pred_len, 5) to (N, seq_len, 5) with zeros after pred_len."""
    n, pred_len, c = pred_68.shape
    if pred_len == seq_len:
        return pred_68
    out = np.zeros((n, seq_len, c), dtype=pred_68.dtype)
    out[:, :pred_len, :] = pred_68
    return out


def format_predictions(public_preds, private_preds):
    preds = []
    for df, preds_ in [(public_df, public_preds), (private_df, private_preds)]:
        if len(df) == 0:
            continue
        seq_len = int(df["seq_length"].iloc[0])
        preds_seq = _pad_to_seq_len(preds_, seq_len=seq_len)
        for i, uid in enumerate(df.id):
            single_pred = preds_seq[i]
            single_df = pd.DataFrame(single_pred, columns=target_cols)
            single_df["id_seqpos"] = [f"{uid}_{x}" for x in range(single_df.shape[0])]
            preds.append(single_df)
    return pd.concat(preds).reset_index(drop=True)


lstm_preds = format_predictions(pred_test_data_public_LSTM, pred_test_data_private_LSTM)
gru_preds = format_predictions(pred_test_data_public_GRU, pred_test_data_private_GRU)

print(lstm_preds.shape, gru_preds.shape)
lstm_preds.head()



## === cell 31
submission_LSTM = submission_format[["id_seqpos"]].merge(
    lstm_preds, how="left", on="id_seqpos"
)
submission_GRU = submission_format[["id_seqpos"]].merge(
    gru_preds, how="left", on="id_seqpos"
)

submission_LSTM[target_cols] = submission_LSTM[target_cols].fillna(0.0)
submission_GRU[target_cols] = submission_GRU[target_cols].fillna(0.0)

print(submission_LSTM.shape, submission_GRU.shape)
submission_LSTM.head()



## === cell 32
submission_lstm_gru_combined = submission_GRU.merge(
    submission_LSTM, how="inner", on="id_seqpos", suffixes=("_gru", "_lstm")
)

gru_weight = 0.6
lstm_weight = 0.4
for c in target_cols:
    submission_lstm_gru_combined[c] = (
        submission_lstm_gru_combined[f"{c}_gru"] * gru_weight
        + submission_lstm_gru_combined[f"{c}_lstm"] * lstm_weight
    )

submission_lstm_gru_combined = submission_lstm_gru_combined[["id_seqpos"] + target_cols]
submission_lstm_gru_combined.head()



## === cell 33
os.chdir("/kaggle/working/")
submission_LSTM.to_csv("submission_LSTM.csv", index=False)
submission_GRU.to_csv("submission_GRU.csv", index=False)
submission_lstm_gru_combined.to_csv("submission.csv", index=False)

print("Wrote:")
print("/kaggle/working/submission.csv")
print("/kaggle/working/submission_LSTM.csv")
print("/kaggle/working/submission_GRU.csv")
print(submission_lstm_gru_combined.shape)
print(submission_lstm_gru_combined.columns.tolist())

assert submission_lstm_gru_combined.shape[0] == submission_format.shape[0]
assert list(submission_lstm_gru_combined.columns) == ["id_seqpos"] + target_cols
assert submission_lstm_gru_combined[target_cols].isna().sum().sum() == 0
