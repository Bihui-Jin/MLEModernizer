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

0.39494

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.63824) has done: 'Your notebook fails because it tries to read a non-existent file (`../input/worst-submission/ensemble52.csv`), so `df` is never created and every later cell errors. I change the input to the provided competition `sample_submission.csv` so the pipeline runs end-to-end and always produces a valid `submission.csv` with the correct columns/row count. I also make the “set one id’s reactivity to 0” step safe by only applying it if that id exists (score-neutral, prevents accidental errors). This won’t target the leaderboard score (no model is trained here), but it generate a valid submission file.'
- What this solution (achieved 0.42418) has done: 'Your current submission is essentially the all-zeros sample submission (plus one id forced to zero), which explains the high (bad) MCRMSE. To move the score down toward your target while keeping changes minimal and within the same “no training” core logic, we can replace the constant predictions with per-position priors learned from `train.json`: the mean target value at each `seqpos` across the training set. This is a lightweight, leakage-free baseline that typically improves MCRMSE substantially versus zeros, without changing any model/training approach. We keep the submission format identical and still write `submission.csv`, and we also keep your special-case id edit (but apply it after filling priors).'
- What this solution (achieved 0.39494) has done: 'You’re currently using per-seqpos means from `train.json`, which is a good minimal baseline, but it leaves a lot of score on the table because it ignores obvious sample-level information (sequence/structure/loop context) that strongly shifts degradation/reactivity. To move the MCRMSE down toward your target while keeping the same “no model training” approach, I add small, leakage-free per-row adjustments using test-known features: global base composition and per-position one-hot of (sequence, structure, loop_type) learned via simple least-squares ridge (closed form, no iterative training loop). I keep your existing per-position mean as the intercept/baseline so the change is incremental and stable, and I preserve the exact submission schema/paths and keep your special-case id override. This should improve from 0.424 toward ~0.35 without changing evaluation semantics or introducing heavy dependencies.'

# 9. Code solution

## === cell 0
import os
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)



## === cell 1
CANDIDATE_SUB_PATHS = [
    "/kaggle/input/sample_submission.csv",
    "/kaggle/input/stanford-covid-vaccine/sample_submission.csv",
    "/kaggle/data/sample_submission.csv",
    "/kaggle/data/stanford-covid-vaccine/sample_submission.csv",
]
CANDIDATE_TRAIN_PATHS = [
    "/kaggle/input/train.json",
    "/kaggle/input/stanford-covid-vaccine/train.json",
    "/kaggle/data/train.json",
    "/kaggle/data/stanford-covid-vaccine/train.json",
]
CANDIDATE_TEST_PATHS = [
    "/kaggle/input/test.json",
    "/kaggle/input/stanford-covid-vaccine/test.json",
    "/kaggle/data/test.json",
    "/kaggle/data/stanford-covid-vaccine/test.json",
]

sub_path = None
for p in CANDIDATE_SUB_PATHS:
    if os.path.exists(p):
        sub_path = p
        break
if sub_path is None:
    raise FileNotFoundError(
        "Could not find sample_submission.csv in expected Kaggle paths. "
        f"Tried: {CANDIDATE_SUB_PATHS}"
    )

train_path = None
for p in CANDIDATE_TRAIN_PATHS:
    if os.path.exists(p):
        train_path = p
        break
if train_path is None:
    raise FileNotFoundError(
        "Could not find train.json in expected Kaggle paths. "
        f"Tried: {CANDIDATE_TRAIN_PATHS}"
    )

test_path = None
for p in CANDIDATE_TEST_PATHS:
    if os.path.exists(p):
        test_path = p
        break
if test_path is None:
    raise FileNotFoundError(
        "Could not find test.json in expected Kaggle paths. "
        f"Tried: {CANDIDATE_TEST_PATHS}"
    )

df = pd.read_csv(sub_path)

required_cols = [
    "id_seqpos",
    "reactivity",
    "deg_Mg_pH10",
    "deg_pH10",
    "deg_Mg_50C",
    "deg_50C",
]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Submission template missing required columns: {missing}")

for c in required_cols[1:]:
    df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)



## === cell 2
train = pd.read_json(train_path, lines=True)
test = pd.read_json(test_path, lines=True)

target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

seq_scored = int(train["seq_scored"].iloc[0])  # expected 68
seq_length = int(train["seq_length"].iloc[0])  # expected 107

pos_means = {}
for t in target_cols:
    arr = np.vstack(train[t].values).astype(np.float64)  # (n_train, 68)
    pos_means[t] = arr.mean(axis=0)  # (68,)

BASES = "ACGU"
STRUCTS = "()."
LOOPS = "SMIBHEX"  # expected loop types

base_to_i = {c: i for i, c in enumerate(BASES)}
struct_to_i = {c: i for i, c in enumerate(STRUCTS)}
loop_to_i = {c: i for i, c in enumerate(LOOPS)}


def _safe_idx(mapper, ch, default=0):
    return mapper.get(ch, default)


def build_features(df_json: pd.DataFrame, upto: int):
    """
    Build per-seqpos feature matrix for first `upto` positions only.
    Features:
      - one-hot base (4)
      - one-hot structure (3)
      - one-hot loop_type (7)
      - global base composition for the whole sequence (4), repeated per position
      - bias term (1)
    Output:
      X: (n_samples*upto, d)
    """
    n = len(df_json)
    d = 4 + 3 + 7 + 4 + 1
    X = np.zeros((n * upto, d), dtype=np.float64)

    for i in range(n):
        seq = str(df_json.loc[i, "sequence"])
        st = str(df_json.loc[i, "structure"])
        lp = str(df_json.loc[i, "predicted_loop_type"])

        counts = np.zeros(4, dtype=np.float64)
        for ch in seq:
            if ch in base_to_i:
                counts[base_to_i[ch]] += 1.0
        denom = max(1.0, float(len(seq)))
        comp = counts / denom

        for p in range(upto):
            r = i * upto + p
            b = seq[p] if p < len(seq) else "A"
            X[r, _safe_idx(base_to_i, b, 0)] = 1.0

            s = st[p] if p < len(st) else "."
            X[r, 4 + _safe_idx(struct_to_i, s, 2)] = 1.0  # default "."

            l = lp[p] if p < len(lp) else "X"
            X[r, 7 + _safe_idx(loop_to_i, l, loop_to_i["X"])] = 1.0

            X[r, 14:18] = comp

            X[r, -1] = 1.0
    return X


n_train = len(train)
Y = np.zeros((n_train * seq_scored, len(target_cols)), dtype=np.float64)
for k, t in enumerate(target_cols):
    arr = np.vstack(train[t].values).astype(np.float64)  # (n_train, 68)
    Y[:, k] = arr.reshape(-1)

baseline = np.zeros((n_train * seq_scored, len(target_cols)), dtype=np.float64)
for k, t in enumerate(target_cols):
    baseline[:, k] = np.tile(pos_means[t], n_train)

Y_delta = Y - baseline

X = build_features(train, upto=seq_scored)

alpha = 10.0
XtX = X.T @ X
XtX.flat[:: XtX.shape[0] + 1] += alpha  # add alpha to diagonal
XtY = X.T @ Y_delta
W = np.linalg.solve(XtX, XtY)  # (d, 5)

X_test = build_features(test, upto=seq_scored)
Yd_test = X_test @ W  # (n_test*68, 5)



## === cell 3
seqpos = df["id_seqpos"].astype(str).str.split("_").str[-1].astype(int).values
id_only = df["id_seqpos"].astype(str).str.rsplit("_", n=1).str[0]

test_id_to_idx = {rid: i for i, rid in enumerate(test["id"].astype(str).values)}

for k, t in enumerate(target_cols):
    pm = pos_means[t]
    base_fill = np.where(
        seqpos < seq_scored, pm[np.clip(seqpos, 0, seq_scored - 1)], pm[-1]
    ).astype(np.float64)
    df[t] = base_fill

delta_add = np.zeros((len(df), len(target_cols)), dtype=np.float64)
valid_mask = (seqpos < seq_scored) & id_only.isin(test_id_to_idx.keys())

if valid_mask.any():
    valid_idx = np.where(valid_mask.values)[0]
    test_rows = id_only.iloc[valid_idx].map(test_id_to_idx).astype(int).values
    pos = seqpos[valid_idx].astype(int)
    flat = test_rows * seq_scored + pos  # index into Yd_test
    delta_add[valid_idx, :] = Yd_test[flat, :]

for k, t in enumerate(target_cols):
    df[t] = (df[t].values.astype(np.float64) + delta_add[:, k]).astype(float)



## === cell 4
mask = df.id_seqpos.astype(str).str.startswith("id_79819a72b")
if mask.any():
    df.loc[mask, "reactivity"] = 0.0



## === cell 5
df = df[required_cols]
df.to_csv("submission.csv", index=False)

print("Wrote submission.csv")
print(df.head())
print("rows:", len(df), "cols:", df.shape[1])
