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

3.9

# 3. Installed packages

geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
sklearn-pandas==2.2.0

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

0.3518779703353691

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 0.47906) has done: 'The fix removes the missing “worst‑submission” file dependency, loads the true training data, computes simple column‑wise mean values for the five targets, and fills the provided sample submission template with those means so a valid `submission.csv` is always written. This resolves the NameError and FileNotFoundError and gives a reasonable baseline that moves the score toward the target.'
- What this solution (achieved 0.43024) has done: 'The fix adds a robust JSON loader that can handle both a single JSON array and line‑delimited JSON objects, preventing the `JSONDecodeError`. With the data correctly loaded, the training, feature creation, and prediction steps run, and a valid `submission.csv` is written. No core modeling logic is changed, preserving the original approach while ensuring the script finishes without errors.'
- What this solution (achieved 0.39379) has done: 'The update adds simple positional one‑hot encoding (treating the base index as a categorical feature) and switches from an ordinary linear regression to a Ridge regression, which usually gives better general‑ization for this kind of high‑dimensional sparse data. These minimal changes keep the overall modelling pipeline intact while expectedly lowering the MCRMSE toward the target score.'
- What this solution (achieved 0.39379) has done: 'I add a numeric position feature to both training and test matrices and slightly reduce the Ridge regularization (alpha = 0.1) to give the model a bit more flexibility. These minimal tweaks keep the overall pipeline unchanged while likely lowering the MCRMSE toward the target score.'
- What this solution (achieved 0.39381) has done: 'The change lowers the regularization strength by setting `alpha=0.0` in the Ridge regressor, which effectively makes it an ordinary least‑squares linear model. This small tweak keeps the overall pipeline unchanged while giving the model more flexibility, expected to reduce the validation MCRMSE and move the score closer to the target (lower is better).'
- What this solution (achieved 0.39345) has done: 'I add the RNA secondary‑structure character as an extra categorical feature (one‑hot encoded) for both training and test data. This small, non‑intrusive change keeps the overall linear‑Ridge pipeline intact while giving the model more information, which is expected to lower the MCRMSE and move the score toward the target.'

# 9. Code solution

## === cell 0
import json
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

DATA_ROOT = "./data"
TRAIN_PATH = f"{DATA_ROOT}/train.json"
TEST_PATH = f"{DATA_ROOT}/test.json"
SAMPLE_SUB_PATH = f"{DATA_ROOT}/sample_submission.csv"
SUBMISSION_OUT = "submission.csv"




## === cell 1
def parse_id_seqpos(id_seqpos: str):
    """Split 'id_xyz_12' into ('id_xyz', 12)."""
    try:
        sample_id, pos_str = id_seqpos.rsplit("_", 1)
        return sample_id, int(pos_str)
    except Exception:
        return id_seqpos, 0  # fallback (should not happen)


def load_json(path):
    with open(path, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            f.seek(0)
            return [json.loads(line) for line in f if line.strip()]


train_data = load_json(TRAIN_PATH)
test_data = load_json(TEST_PATH)

submission = pd.read_csv(SAMPLE_SUB_PATH)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/3718346906.py in <cell line: 0>()
     19 
     20 
---> 21 train_data = load_json(TRAIN_PATH)
     22 test_data = load_json(TEST_PATH)
     23 

/tmp/ipykernel_11/3718346906.py in load_json(path)
     10 # load json files (handles both array and line‑delimited formats)
     11 def load_json(path):
---> 12     with open(path, "r") as f:
     13         try:
     14             return json.load(f)

FileNotFoundError: [Errno 2] No such file or directory: './data/train.json'

## === cell 2
train_rows = []
target_vals = {
    col: []
    for col in ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
}

for entry in train_data:
    seq = entry["sequence"]
    struct = entry["structure"]
    loop = entry["predicted_loop_type"]
    scored_len = entry["seq_scored"]
    for pos in range(scored_len):
        nuc = seq[pos]
        prev_nuc = seq[pos - 1] if pos > 0 else "N"
        next_nuc = seq[pos + 1] if pos + 1 < len(seq) else "N"
        loop_ch = loop[pos]
        struct_ch = struct[pos]

        train_rows.append(
            {
                "pos": pos,
                "pos_str": str(pos),
                "nuc": nuc,
                "prev_nuc": prev_nuc,
                "next_nuc": next_nuc,
                "loop": loop_ch,
                "struct": struct_ch,
                "pos_sq": pos * pos,
            }
        )
        target_vals["reactivity"].append(entry["reactivity"][pos])
        target_vals["deg_Mg_pH10"].append(entry["deg_Mg_pH10"][pos])
        target_vals["deg_pH10"].append(entry["deg_pH10"][pos])
        target_vals["deg_Mg_50C"].append(entry["deg_Mg_50C"][pos])
        target_vals["deg_50C"].append(entry["deg_50C"][pos])

train_features = pd.DataFrame(train_rows)



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1613984110.py in <cell line: 0>()
      6 }
      7 
----> 8 for entry in train_data:
      9     seq = entry["sequence"]
     10     struct = entry["structure"]

NameError: name 'train_data' is not defined

## === cell 3
cat_cols = ["pos_str", "nuc", "prev_nuc", "next_nuc", "loop", "struct"]
X_train = pd.get_dummies(train_features[cat_cols], columns=cat_cols)
X_train["pos"] = train_features["pos"]
X_train["pos_sq"] = train_features["pos_sq"]

target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
y_train = {col: np.array(target_vals[col]) for col in target_cols}



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/694636520.py in <cell line: 0>()
      1 cat_cols = ["pos_str", "nuc", "prev_nuc", "next_nuc", "loop", "struct"]
----> 2 X_train = pd.get_dummies(train_features[cat_cols], columns=cat_cols)
      3 X_train["pos"] = train_features["pos"]
      4 X_train["pos_sq"] = train_features["pos_sq"]
      5 

NameError: name 'train_features' is not defined

## === cell 4
X_tr, X_val, y_tr_dict, y_val_dict = train_test_split(
    X_train, pd.DataFrame(y_train), test_size=0.2, random_state=42
)

models = {}
for col in target_cols:
    model = Ridge(alpha=0.0, fit_intercept=True)  # ordinary least‑squares
    model.fit(X_tr, y_tr_dict[col])
    models[col] = model


def mcrmse(y_true, y_pred):
    return np.mean(np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0)))


val_preds = {}
for col in ["reactivity", "deg_Mg_pH10", "deg_50C"]:
    val_preds[col] = models[col].predict(X_val)

val_score = mcrmse(
    np.column_stack(
        [y_val_dict[col] for col in ["reactivity", "deg_Mg_pH10", "deg_50C"]]
    ),
    np.column_stack(
        [val_preds[col] for col in ["reactivity", "deg_Mg_pH10", "deg_50C"]]
    ),
)
print(f"Validation MCRMSE (scored columns): {val_score:.5f}")



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2955025333.py in <cell line: 0>()
      1 # optional validation to see current score (not required for submission)
      2 X_tr, X_val, y_tr_dict, y_val_dict = train_test_split(
----> 3     X_train, pd.DataFrame(y_train), test_size=0.2, random_state=42
      4 )
      5 

NameError: name 'X_train' is not defined

## === cell 5
test_lookup = {e["id"]: e for e in test_data}
test_feature_rows = []
for _, row in submission.iterrows():
    sample_id, pos = parse_id_seqpos(row["id_seqpos"])
    entry = test_lookup.get(sample_id)
    if entry is None or pos >= entry["seq_length"]:
        nuc = prev_nuc = next_nuc = loop = struct = "N"
    else:
        seq = entry["sequence"]
        nuc = seq[pos]
        prev_nuc = seq[pos - 1] if pos > 0 else "N"
        next_nuc = seq[pos + 1] if pos + 1 < len(seq) else "N"
        loop = entry["predicted_loop_type"][pos]
        struct = entry["structure"][pos]
    test_feature_rows.append(
        {
            "pos": pos,
            "pos_str": str(pos),
            "nuc": nuc,
            "prev_nuc": prev_nuc,
            "next_nuc": next_nuc,
            "loop": loop,
            "struct": struct,
            "pos_sq": pos * pos,
        }
    )

test_features = pd.DataFrame(test_feature_rows)
X_test = pd.get_dummies(test_features[cat_cols], columns=cat_cols)
X_test["pos"] = test_features["pos"]
X_test["pos_sq"] = test_features["pos_sq"]
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

for col in target_cols:
    submission[col] = models[col].predict(X_test)



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1881972815.py in <cell line: 0>()
      1 # ---------- prepare test features ----------
----> 2 test_lookup = {e["id"]: e for e in test_data}
      3 test_feature_rows = []
      4 for _, row in submission.iterrows():
      5     sample_id, pos = parse_id_seqpos(row["id_seqpos"])

NameError: name 'test_data' is not defined

## === cell 6
submission.to_csv(SUBMISSION_OUT, index=False)
print(f"Submission file written to '{SUBMISSION_OUT}' with shape:", submission.shape)

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2340370376.py in <cell line: 0>()
----> 1 submission.to_csv(SUBMISSION_OUT, index=False)
      2 print(f"Submission file written to '{SUBMISSION_OUT}' with shape:", submission.shape)

NameError: name 'submission' is not defined
