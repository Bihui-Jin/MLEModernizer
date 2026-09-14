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

0.41711

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.63824) has done: 'I remove the dependency on the missing `../input/worst-submission/ensemble52.csv` and instead build a submission directly from the provided `sample_submission.csv`, which guarantees the correct row count and column names. I also fix the logic bug where you tried to select `seq_length == 130` (this dataset uses 107) and eliminate downstream `NameError`s caused by `df` never being created. Since no valid submission was previously generated, this focuses on correctness/stability rather than score tuning; it produce a valid `submission.csv` end-to-end.'
- What this solution (achieved 0.63824) has done: 'Your current code intentionally injects a huge incorrect value (`reactivity = 100`) for one specific test id prefix, which dramatically worsen MCRMSE (lower is better) and explains the poor 0.63824 score. I remove that sabotage line while keeping the rest of your submission-building logic identical (still based on `sample_submission.csv` for correct formatting/row count). I also add a small safety check to ensure all predictions are finite numeric values before writing, without changing your approach or introducing any new modeling. This minimal change should move your score substantially down toward the target 0.3519.'
- What this solution (achieved 0.42166) has done: 'Your current pipeline always submits the all-zeros baseline copied from `sample_submission.csv`, which is why the score is stuck around ~0.64 (this competition’s baseline). To move the score down toward the target (lower is better), the smallest legitimate improvement that preserves your “no ML model” core logic is to replace the zeros with a simple per-position prior computed from the training set targets (mean target value at each `seqpos`). This keeps the exact same submission-building flow and format checks, but uses information available in `train.json` to produce more realistic predictions for each base position. I also fill positions 68–106 (not scored but required) with the last scored position’s mean to keep output well-defined and stable.'
- What this solution (achieved 0.3912) has done: 'Your current approach (per-position mean priors) is stable but leaves score on the table because it ignores sample-specific signals already present in test.json (sequence/structure/loop type). To move your MCRMSE down toward the target with minimal logic change, I keep the same “predict by seqpos priors” core, but add small additive adjustments learned from train.json based on (a) nucleotide identity at each position and (b) loop-type at each position. This remains a simple closed-form baseline (no model training loops), is fully deterministic, and still fills unscored positions 68–106 with the last scored position’s estimate. The submission format/row order remains anchored to sample_submission.csv to guarantee a valid file.'
- What this solution (achieved 0.3891) has done: 'Your current score (0.3912) is worse than the target (0.3519) for a lower-is-better metric, so we should make a small, low-risk improvement. Keeping your same “positional mean + additive base and loop residual effects” core logic, I add a tiny amount of regularization (shrinkage) to the base/loop residuals to reduce noise/overfitting, which typically improves generalization on this dataset. I also add a lightweight clipping of predictions to the training target quantile range per column to avoid rare extreme values that can hurt RMSE, without changing the overall approach. Submission format and row ordering remain anchored to `sample_submission.csv`, and runtime stays well under the limit.'
- What this solution (achieved 0.41711) has done: 'Your current score (0.3891) is worse than the target (0.3519) for a lower-is-better metric, so we should make a small generalization-focused improvement without changing the core “positional mean + additive base/loop residual effects” approach. The simplest low-risk gain here is to add a third residual correction for the `structure` character at each position (paired/unpaired context), learned from train.json the same way as base/loop, with the same shrinkage to avoid overfitting. This leverages an already-available per-position signal in both train and test and typically improves MCRMSE while preserving identical evaluation semantics and keeping runtime small. Everything else (format anchored to sample_submission, filling unscored positions with last scored mean, clipping) remains the same.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os



## === cell 1
BASE = "/kaggle/input/stanford-covid-vaccine"
train_path = os.path.join(BASE, "train.json")
test_path = os.path.join(BASE, "test.json")
sample_sub_path = os.path.join(BASE, "sample_submission.csv")

df_train = pd.read_json(train_path, lines=True)
df_test = pd.read_json(test_path, lines=True)
sample_sub = pd.read_csv(sample_sub_path)

print(
    "train rows:",
    df_train.shape,
    "test rows:",
    df_test.shape,
    "sample_sub rows:",
    sample_sub.shape,
)
print("sample_sub columns:", sample_sub.columns.tolist())



## === cell 2
seq_lengths = df_test["seq_length"].value_counts().sort_index()
print("test seq_length distribution:\n", seq_lengths)

sequences = sorted(df_test.loc[df_test["seq_length"] == 107, "id"].unique().tolist())
print("num sequences with length 107:", len(sequences))
print("last 10 ids:", sequences[-10:])



## === cell 3
target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
scored_target_cols = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]  # scored by Kaggle

if "SN_filter" in df_train.columns:
    df_train_use = df_train[df_train["SN_filter"] == 1].copy()
    if len(df_train_use) == 0:
        df_train_use = df_train.copy()
else:
    df_train_use = df_train.copy()

records = []
for _, row in df_train_use.iterrows():
    seq_scored = int(row["seq_scored"])
    L = min(seq_scored, 68)
    seq = str(row["sequence"])
    loop = str(row["predicted_loop_type"])
    struct = str(row["structure"])
    for pos in range(L):
        rec = {"seqpos": pos}
        rec["base"] = seq[pos] if pos < len(seq) else np.nan
        rec["loop_type"] = loop[pos] if pos < len(loop) else np.nan
        rec["struct_char"] = struct[pos] if pos < len(struct) else np.nan
        for c in target_cols:
            v = row[c][pos] if pos < len(row[c]) else np.nan
            rec[c] = v
        records.append(rec)

train_long = pd.DataFrame.from_records(records)
for c in target_cols:
    train_long[c] = pd.to_numeric(train_long[c], errors="coerce")

pos_means = train_long.groupby("seqpos")[target_cols].mean(numeric_only=True)

global_means = train_long[target_cols].mean(numeric_only=True)
for pos in range(68):
    if pos not in pos_means.index:
        pos_means.loc[pos] = global_means
pos_means = pos_means.sort_index()

resid = train_long.copy()
for c in target_cols:
    resid[c] = resid[c] - resid["seqpos"].map(pos_means[c])

base_mean_resid = resid.groupby(["seqpos", "base"])[target_cols].mean(numeric_only=True)
base_cnt = resid.groupby(["seqpos", "base"]).size().rename("n").astype(np.float32)

loop_mean_resid = resid.groupby(["seqpos", "loop_type"])[target_cols].mean(
    numeric_only=True
)
loop_cnt = resid.groupby(["seqpos", "loop_type"]).size().rename("n").astype(np.float32)

struct_mean_resid = resid.groupby(["seqpos", "struct_char"])[target_cols].mean(
    numeric_only=True
)
struct_cnt = (
    resid.groupby(["seqpos", "struct_char"]).size().rename("n").astype(np.float32)
)

K_BASE = 30.0
K_LOOP = 30.0
K_STRUCT = 30.0

base_w = (base_cnt / (base_cnt + K_BASE)).to_frame()  # align by index
loop_w = (loop_cnt / (loop_cnt + K_LOOP)).to_frame()
struct_w = (struct_cnt / (struct_cnt + K_STRUCT)).to_frame()

base_effect = base_mean_resid.mul(base_w["n"], axis=0)
loop_effect = loop_mean_resid.mul(loop_w["n"], axis=0)
struct_effect = struct_mean_resid.mul(struct_w["n"], axis=0)

clip_lo = train_long[target_cols].quantile(0.001, numeric_only=True)
clip_hi = train_long[target_cols].quantile(0.999, numeric_only=True)

print("Per-position means (head):")
print(pos_means.head())
print("Learned base_effect index levels:", base_effect.index.names)
print("Learned loop_effect index levels:", loop_effect.index.names)
print("Learned struct_effect index levels:", struct_effect.index.names)
print("Clip ranges (0.1%..99.9%):")
print(pd.DataFrame({"lo": clip_lo, "hi": clip_hi}))



## === cell 4
test_lookup = df_test.set_index("id")[
    ["sequence", "predicted_loop_type", "structure"]
].to_dict(orient="index")

df = sample_sub.copy()
seqpos = df["id_seqpos"].astype(str).str.rsplit("_", n=1).str[-1].astype(int)
ids = df["id_seqpos"].astype(str).str.rsplit("_", n=1).str[0]

df["_seqpos"] = seqpos
df["_id"] = ids

bases = np.empty(len(df), dtype=object)
loops = np.empty(len(df), dtype=object)
structs = np.empty(len(df), dtype=object)

for i, (rid, sp_i) in enumerate(zip(df["_id"].to_numpy(), df["_seqpos"].to_numpy())):
    info = test_lookup.get(rid, None)
    if info is None:
        bases[i] = np.nan
        loops[i] = np.nan
        structs[i] = np.nan
        continue
    s = info["sequence"]
    lt = info["predicted_loop_type"]
    st = info["structure"]
    bases[i] = s[sp_i] if sp_i < len(s) else np.nan
    loops[i] = lt[sp_i] if sp_i < len(lt) else np.nan
    structs[i] = st[sp_i] if sp_i < len(st) else np.nan

df["_base"] = bases
df["_loop_type"] = loops
df["_struct_char"] = structs

sp = df["_seqpos"].to_numpy()
scored_mask = sp < 68

last_scored_mean = pos_means.loc[67]

for c in target_cols:
    vals = np.empty(len(df), dtype=np.float32)
    vals.fill(np.float32(last_scored_mean[c]))

    if scored_mask.any():
        sp_sc = df.loc[scored_mask, "_seqpos"].to_numpy()
        base_sc = df.loc[scored_mask, "_base"].to_numpy()
        loop_sc = df.loc[scored_mask, "_loop_type"].to_numpy()
        struct_sc = df.loc[scored_mask, "_struct_char"].to_numpy()

        pred = pos_means.loc[sp_sc, c].to_numpy(dtype=np.float32)

        idx_base = pd.MultiIndex.from_arrays([sp_sc, base_sc], names=["seqpos", "base"])
        be = base_effect[c].reindex(idx_base).to_numpy()
        be = np.where(np.isfinite(be), be, 0.0).astype(np.float32)
        pred = pred + be

        idx_loop = pd.MultiIndex.from_arrays(
            [sp_sc, loop_sc], names=["seqpos", "loop_type"]
        )
        le = loop_effect[c].reindex(idx_loop).to_numpy()
        le = np.where(np.isfinite(le), le, 0.0).astype(np.float32)
        pred = pred + le

        idx_struct = pd.MultiIndex.from_arrays(
            [sp_sc, struct_sc], names=["seqpos", "struct_char"]
        )
        se = struct_effect[c].reindex(idx_struct).to_numpy()
        se = np.where(np.isfinite(se), se, 0.0).astype(np.float32)
        pred = pred + se

        lo = np.float32(clip_lo[c])
        hi = np.float32(clip_hi[c])
        pred = np.clip(pred, lo, hi)

        vals[scored_mask] = pred

    vals = np.clip(vals, np.float32(clip_lo[c]), np.float32(clip_hi[c]))
    df[c] = vals

df = df.drop(columns=["_seqpos", "_id", "_base", "_loop_type", "_struct_char"])

print(df.head())



## === cell 5
mask = df["id_seqpos"].astype(str).str.startswith("id_ff4593941_")
n = int(mask.sum())
print("rows matching id_ff4593941_*:", n)



## === cell 6
expected_cols = ["id_seqpos"] + target_cols
missing = [c for c in expected_cols if c not in df.columns]
extra = [c for c in df.columns if c not in expected_cols]
if missing:
    raise ValueError(f"Submission missing required columns: {missing}")
if extra:
    df = df[expected_cols]

if df.shape[0] != sample_sub.shape[0]:
    raise ValueError(
        f"Row count mismatch: df has {df.shape[0]} rows, sample has {sample_sub.shape[0]} rows"
    )

for c in target_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce").astype(np.float32)
    if not np.isfinite(df[c].to_numpy()).all():
        raise ValueError(f"Non-finite values found in column: {c}")

print("submission shape:", df.shape)
print(df.describe(include="all").transpose().head(10))



## === cell 7
df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", df.shape)
