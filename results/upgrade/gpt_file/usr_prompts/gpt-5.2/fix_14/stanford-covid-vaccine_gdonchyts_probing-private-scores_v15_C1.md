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

0.3568481764028675

# 6. Current score

0.50911

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.63824) has done: 'The current notebook fails immediately because it tries to read a non-existent CSV (`../input/worst-submission/ensemble52.csv`). I replace that dependency by loading the provided `sample_submission.csv` from the competition dataset path, so the pipeline always runs in this environment. I keep your existing “calibration” logic (`/= 1.06`) but apply it safely only to the numeric target columns and ensure all required submission columns exist and are in the correct order. Finally, I write a valid `submission.csv` with the required header and row count.'
- What this solution (achieved 0.63824) has done: 'Your current code is effectively submitting (almost) all zeros (because `sample_submission.csv` target columns are 0), and dividing by 1.06 keeps them at 0; this explains the weak 0.63824. To move the score down toward the 0.3568 target with minimal change and without changing the overall “no-model” approach, I replace the constant-zero predictions with per-position mean target values computed from `train.json` (a simple baseline that often lands around the mid-0.3s on this competition). I keep your existing calibration step (`/= 1.06`) but apply it after inserting these baseline predictions. I also ensure the predictions are aligned to the correct `seqpos` extracted from `id_seqpos` and that the submission preserves the required row order and columns.'
- What this solution (achieved 0.426) has done: 'The crash comes from indexing the per-position mean array `m` with `_seqpos` values that include 68–106; `np.where` still evaluates `m[df["_seqpos"]]` for all rows, so it raises `IndexError`. I fix this by computing a clipped index (min(seqpos, 67)) and indexing `m` only with that safe index, preserving your exact baseline logic (use pos-mean for 0–67 and reuse pos-67 mean for the rest). I also keep your existing calibration (`/ 1.06`) and submission column/order checks unchanged so it runs end-to-end and writes a valid `submission.csv`.'
- What this solution (achieved 0.4218) has done: 'Your current baseline is already valid and stable, but its MCRMSE (0.426) is worse than the target (0.3568), so we should make a small legitimate improvement without changing the overall “no-model, per-position statistics from train.json” approach. The biggest low-risk gain here is to compute the per-position means using only the “clean” training subset (SN_filter==1), because the competition’s test distribution is filtered for quality and this alignment typically improves generalization. I keep your exact per-position mean logic, keep the same 68→107 clipping behavior, and keep your calibration step but make it slightly less aggressive (from /1.06 to /1.03) to reduce underprediction bias that often hurts RMSE on this task. The script still run end-to-end and write a valid `submission.csv` with the required columns and row order.'
- What this solution (achieved 0.42181) has done: 'You’re currently above the target (0.4218 vs 0.3568; lower is better), so we should make a small, low-risk improvement rather than changing the baseline approach. I keep your exact “per-position mean from train.json with 68→107 clipping” logic, but compute those means using a signal-to-noise weighted average (still pure train-set statistics, no model) so cleaner, higher-confidence sequences influence the baseline more. This typically improves MCRMSE on this competition without changing evaluation semantics or requiring new packages. I keep your SN_filter==1 subset and keep the same calibration (/1.03) unless weighting already shifts scale too much.'
- What this solution (achieved 0.42175) has done: 'We keep your exact “SN_filter==1 + signal_to_noise weighted per-position mean + 68→107 clipping” baseline, but fix a small statistical mismatch: right now each target column is averaged independently even when some sequences contain NaNs at certain positions, which can bias means and hurt RMSE. I compute a weighted mean per position with a *per-position valid-weight denominator* (ignore NaNs in the numerator and denominator) to better match how the ground truth is defined. I also make the calibration slightly less aggressive (1.03 → 1.02) to nudge predictions upward a touch; since you’re worse than target (0.42181 vs 0.35685; lower is better), this is a minimal, low-risk improvement. All paths, output format, and submission writing remain unchanged.'
- What this solution (achieved 0.44493) has done: 'Your current gap to target is sizable (0.42175 vs 0.35685; lower is better), so we need a small but meaningful improvement while preserving the same “per-position statistics baseline from train.json” core logic. The lowest-risk gain on this competition is to compute a more robust per-position estimate by using a weighted **median** (instead of weighted mean) on the SN_filter==1 subset, which reduces the impact of heavy-tailed/noisy labels and typically lowers RMSE without changing the overall approach. I keep your exact seqpos extraction and 68→107 clipping behavior, and I keep your existing calibration step (still a single global divisor) to avoid changing semantics too much. The script still runs end-to-end, uses only the provided data paths, and writes a valid `submission.csv` with the required columns/order.'
- What this solution (achieved 0.42269) has done: 'Your current score (0.44493) is worse than the target (0.35685; lower is better), so we need a small improvement while keeping the same “per-position statistics baseline” core logic. The safest gain without changing the overall approach is to stop using a weighted median (which can be too “step-like” and underfit for RMSE) and instead use a *robust weighted mean* computed after winsorizing each position (clip to per-position weighted quantiles). This keeps the exact same training-data-only, per-position aggregation concept, but reduces the impact of heavy tails while still optimizing for an L2-type metric more directly than a median. I also keep the same SN_filter==1 subset, the same seqpos extraction and 68→107 clipping, and the same global calibration divisor (1.02) to avoid shifting semantics too much.'
- What this solution (achieved 0.42198) has done: 'Your current score (0.42269) is still worse than the target (0.35685; lower is better), so we should make a small, legitimate improvement while keeping the exact same “SN_filter==1 + (winsorized) weighted per-position mean + 68→107 clipping + global calibration divisor” core logic. The lowest-risk gain here is to tune the winsorization quantiles a bit wider (less clipping) so the per-position means better match an L2/RMSE objective; your current 5–95% clipping can underfit by suppressing real signal. I also slightly relax the signal_to_noise cap (10→20) to let very clean sequences contribute more (still bounded to avoid a few points dominating), which often improves generalization on this competition’s filtered test. Everything else (paths, row alignment, columns, 1.02 divisor, CSV writing) stays the same to preserve semantics and stability.'
- What this solution (achieved 0.43735) has done: 'Your current score (0.42198) is worse than the target (0.35685; lower is better), so we should make a small, low-risk improvement while preserving the exact “SN_filter==1 + winsorized S/N-weighted per-position mean + 68→107 clipping + global calibration” baseline. The most impactful minimal tweak here is to (a) use the correct aggregation for an RMSE metric by switching from mean to **weighted mean of squared targets then sqrt** (per-position RMS), which is still purely train-statistics per position and often fits RMSE better, and (b) adjust the global calibration slightly to avoid inflating predictions too much after the RMS change. Everything else (data paths, seqpos extraction/clipping, required columns/order, and writing `submission.csv`) stays the same.'
- What this solution (achieved 0.42211) has done: 'We keep your exact “no-model, per-position train statistics + 68→107 clipping + global calibration” core logic, but make two minimal tweaks aimed at reducing RMSE (lower-is-better) toward the 0.3568 target. First, we switch the per-position aggregator back from RMS to a winsorized weighted mean, because RMS of targets is a biased estimator for the conditional mean and tends to inflate predictions and hurt RMSE here. Second, we adjust the single global calibration divisor slightly (1.06 → 1.03) to correct the scale after changing the aggregator, while leaving all paths, row alignment, and submission formatting unchanged.'
- What this solution (achieved 0.42191) has done: 'Your score (0.42211) is worse than the target (0.35685; lower is better), so we should make a small, low-risk improvement while keeping the same core “SN_filter==1 + S/N-weighted per-position winsorized mean + seqpos clipping + single global calibration divisor” approach. The most likely issue is that a single global calibration factor is suboptimal because the five targets have different scales; we keep the exact same baseline predictions but replace the single divisor with per-target divisors computed from the training set (still train-only statistics), which usually reduces MCRMSE without changing the modeling logic. To avoid overfitting, we compute these divisors using only the scored positions (0–67) and only SN_filter==1, matching the test distribution you already use. Everything else (paths, aggregation, row alignment, column order, and writing `submission.csv`) stays the same.'
- What this solution (achieved 0.50911) has done: 'Your current score (0.42191; lower-is-better) is still above the target (0.35685), so we should make a small, legitimate improvement without changing the overall “per-position train statistics baseline” logic. The biggest low-risk mismatch is that we’re calibrating each target using all 68 positions equally, but the public/private test sets are filtered and the *ends* (later positions) often behave differently; we compute the per-target calibration on the same SN_filter==1 subset but with an additional mild downweighting of noisier sequences via their provided per-target error arrays (still pure train-only statistics). Concretely, we keep the same winsorized S/N-weighted per-position means, but in the calibration step we use effective weights `w / (error^2 + eps)` per target/position so high-uncertainty labels influence the scaling less, which typically reduces RMSE. Everything else (paths, seqpos extraction/clipping, required columns/order, and writing `submission.csv`) remains unchanged.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

np.random.seed(0)



## === cell 1
CANDIDATE_PATHS = [
    "/kaggle/input/stanford-covid-vaccine/sample_submission.csv",
    "/kaggle/input/sample_submission.csv",
    "../input/stanford-covid-vaccine/sample_submission.csv",
    "../input/sample_submission.csv",
    "/kaggle/data/stanford-covid-vaccine/sample_submission.csv",
    "/kaggle/data/sample_submission.csv",
]

sub_path = None
for p in CANDIDATE_PATHS:
    if os.path.exists(p):
        sub_path = p
        break

if sub_path is None:
    raise FileNotFoundError(
        "Could not find sample_submission.csv in expected locations. "
        f"Tried: {CANDIDATE_PATHS}"
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
    raise ValueError(f"sample_submission is missing required columns: {missing}")

for c in required_cols[1:]:
    df[c] = pd.to_numeric(df[c], errors="coerce").astype(np.float32)

df.head()



## === cell 2
sequences = list(set(["_".join(v.split("_")[:-1]) for v in df.id_seqpos.values]))
sequences[-10:], len(sequences)



## === cell 3
TRAIN_CANDIDATE_PATHS = [
    "/kaggle/input/stanford-covid-vaccine/train.json",
    "/kaggle/input/train.json",
    "../input/stanford-covid-vaccine/train.json",
    "../input/train.json",
    "/kaggle/data/stanford-covid-vaccine/train.json",
    "/kaggle/data/train.json",
]

train_path = None
for p in TRAIN_CANDIDATE_PATHS:
    if os.path.exists(p):
        train_path = p
        break
if train_path is None:
    raise FileNotFoundError(
        "Could not find train.json in expected locations. "
        f"Tried: {TRAIN_CANDIDATE_PATHS}"
    )

train = pd.read_json(train_path, lines=True)

target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

if "SN_filter" in train.columns:
    train = train.loc[train["SN_filter"] == 1].reset_index(drop=True)

if "signal_to_noise" in train.columns:
    w = (
        pd.to_numeric(train["signal_to_noise"], errors="coerce")
        .fillna(0.0)
        .to_numpy(np.float32)
    )
    w = np.clip(w, 0.0, 20.0).astype(np.float32)
else:
    w = np.ones(len(train), dtype=np.float32)

w_sum = float(w.sum())
if not np.isfinite(w_sum) or w_sum <= 0:
    w = np.ones(len(train), dtype=np.float32)


def weighted_quantile_1d(x: np.ndarray, w: np.ndarray, q: float) -> np.float32:
    m = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if not np.any(m):
        return np.float32(0.0)
    x = x[m].astype(np.float32)
    w = w[m].astype(np.float32)
    order = np.argsort(x, kind="mergesort")
    x = x[order]
    w = w[order]
    cw = np.cumsum(w, dtype=np.float64)
    cutoff = q * float(cw[-1])
    k = int(np.searchsorted(cw, cutoff, side="left"))
    if k < 0:
        k = 0
    elif k >= len(x):
        k = len(x) - 1
    return np.float32(x[k])


def winsorized_weighted_mean_per_position(
    arr_2d: np.ndarray,
    weights_1d: np.ndarray,
    q_low: float = 0.05,
    q_high: float = 0.95,
) -> np.ndarray:
    n, p = arr_2d.shape
    out = np.zeros(p, dtype=np.float32)
    wloc = weights_1d.astype(np.float32)

    for j in range(p):
        x = arr_2d[:, j].astype(np.float32)
        lo = weighted_quantile_1d(x, wloc, q_low)
        hi = weighted_quantile_1d(x, wloc, q_high)

        m = np.isfinite(x) & np.isfinite(wloc) & (wloc > 0)
        if not np.any(m):
            out[j] = np.float32(0.0)
            continue

        xj = x[m]
        wj = wloc[m].astype(np.float64)

        xj = np.clip(xj, lo, hi).astype(np.float64)

        denom = float(wj.sum())
        if not np.isfinite(denom) or denom <= 0:
            out[j] = np.float32(0.0)
            continue

        out[j] = np.float32((xj * wj).sum() / denom)

    return out


means_by_pos = {}
w_f32 = w.astype(np.float32)

for col in target_cols:
    arr = np.stack(train[col].values).astype(np.float32)  # (n_train, 68)
    means_by_pos[col] = winsorized_weighted_mean_per_position(
        arr, w_f32, q_low=0.02, q_high=0.98
    )

df["_seqpos"] = df["id_seqpos"].str.rsplit("_", n=1).str[-1].astype(np.int32)
idx = np.minimum(df["_seqpos"].values, 67).astype(np.int32)

for col in target_cols:
    m = means_by_pos[col]
    df[col] = m[idx].astype(np.float32)

calib_div = {}
eps = 1e-12

error_map = {
    "reactivity": "reactivity_error",
    "deg_Mg_pH10": "deg_error_Mg_pH10",
    "deg_pH10": "deg_error_pH10",
    "deg_Mg_50C": "deg_error_Mg_50C",
    "deg_50C": "deg_error_50C",
}

for col in target_cols:
    y = np.stack(train[col].values).astype(np.float32)  # (n_train, 68)
    pred_pos = means_by_pos[col].astype(np.float32)[None, :]  # (1, 68) broadcast

    err_col = error_map.get(col, None)
    if err_col is not None and err_col in train.columns:
        e = np.stack(train[err_col].values).astype(np.float32)  # (n_train, 68)
        e2 = (e.astype(np.float64) ** 2) + 1e-6  # keep finite, avoid division by zero
        ww = (w_f32[:, None].astype(np.float64) / e2).astype(np.float64)
    else:
        ww = w_f32[:, None].astype(np.float64)

    yy = y.astype(np.float64)
    pp = pred_pos.astype(np.float64)

    msk = np.isfinite(yy) & np.isfinite(pp) & np.isfinite(ww) & (ww > 0)
    if not np.any(msk):
        calib_div[col] = np.float32(1.03)
        continue

    num = (ww * (yy * pp) * msk).sum()
    den = (ww * (pp * pp) * msk).sum()
    if not np.isfinite(num) or not np.isfinite(den) or den <= eps:
        calib_div[col] = np.float32(1.03)
        continue

    a = num / den
    if not np.isfinite(a) or a <= 0:
        calib_div[col] = np.float32(1.03)
        continue

    calib_div[col] = np.float32(1.0 / a)

for col in target_cols:
    df[col] = (df[col].astype(np.float32) / calib_div[col]).astype(np.float32)

df[target_cols] = (
    df[target_cols].replace([np.inf, -np.inf], np.nan).fillna(np.float32(0.0))
)

df.drop(columns=["_seqpos"], inplace=True)
df.describe()



## === cell 4
df = df[required_cols].copy()

if df["id_seqpos"].isna().any():
    raise ValueError("Found NaN in id_seqpos, submission would be invalid.")
if df.duplicated(subset=["id_seqpos"]).any():
    raise ValueError("Duplicate id_seqpos detected; submission likely malformed.")

df.tail()



## === cell 5
out_path = "submission.csv"
df.to_csv(out_path, index=False)

print("Wrote:", out_path)
print("Shape:", df.shape)
print("Columns:", list(df.columns))
print(
    "Per-target calibration divisors used:", {k: float(v) for k, v in calib_div.items()}
)
print(df.head(3).to_string(index=False))
