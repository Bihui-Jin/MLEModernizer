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

0.39025

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.63824) has done: 'Your notebook fails because it tries to read a non-existent file (`../input/worst-submission/ensemble52.csv`), so `df` is never created and every later cell errors. I change the input to the provided competition `sample_submission.csv` so the pipeline runs end-to-end and always produces a valid `submission.csv` with the correct columns/row count. I also make the “set one id’s reactivity to 0” step safe by only applying it if that id exists (score-neutral, prevents accidental errors). This won’t target the leaderboard score (no model is trained here), but it generate a valid submission file.'
- What this solution (achieved 0.42418) has done: 'Your current submission is essentially the all-zeros sample submission (plus one id forced to zero), which explains the high (bad) MCRMSE. To move the score down toward your target while keeping changes minimal and within the same “no training” core logic, we can replace the constant predictions with per-position priors learned from `train.json`: the mean target value at each `seqpos` across the training set. This is a lightweight, leakage-free baseline that typically improves MCRMSE substantially versus zeros, without changing any model/training approach. We keep the submission format identical and still write `submission.csv`, and we also keep your special-case id edit (but apply it after filling priors).'
- What this solution (achieved 0.39494) has done: 'You’re currently using per-seqpos means from `train.json`, which is a good minimal baseline, but it leaves a lot of score on the table because it ignores obvious sample-level information (sequence/structure/loop context) that strongly shifts degradation/reactivity. To move the MCRMSE down toward your target while keeping the same “no model training” approach, I add small, leakage-free per-row adjustments using test-known features: global base composition and per-position one-hot of (sequence, structure, loop_type) learned via simple least-squares ridge (closed form, no iterative training loop). I keep your existing per-position mean as the intercept/baseline so the change is incremental and stable, and I preserve the exact submission schema/paths and keep your special-case id override. This should improve from 0.424 toward ~0.35 without changing evaluation semantics or introducing heavy dependencies.'
- What this solution (achieved 0.39147) has done: 'You’re currently worse than the target (0.39494 vs 0.35188, lower-is-better), so we should make a small, safe improvement without changing the “per-position mean + closed-form ridge correction” core approach. The biggest low-risk gain here is to (1) train the ridge only on high-quality training rows (`SN_filter==1`) to reduce label noise and (2) weight the ridge regression by per-position measurement uncertainty using the provided `*_error_*` arrays (a metric-aligned, leakage-free change that keeps the same closed-form solve). I also keep the per-seqpos mean baseline but compute it on the same filtered set for consistency. These changes are minimal (same features, same linear model, same no-iteration training) and should move MCRMSE down toward your target.'
- What this solution (achieved 0.39168) has done: 'We’re currently above (worse than) the target MCRMSE, so the safest way to move down toward it without changing your core “per-position mean + closed-form ridge correction” logic is to improve the ridge fit quality with minimal, metric-aligned tweaks. I keep the exact same feature set and closed-form solve, but (1) compute the ridge correction only on the 3 scored targets (reactivity, deg_Mg_pH10, deg_Mg_50C) to reduce multi-task interference from unscored labels, and (2) use per-target inverse-variance weights (from the provided error arrays) instead of a single averaged weight per row, so each task is trained with its own uncertainty weighting. Unscored targets (deg_pH10, deg_50C) remain as the per-seqpos mean baseline (stable and score-neutral). The submission format/paths remain identical and `submission.csv` is still written.'
- What this solution (achieved 0.39139) has done: 'You’re currently worse than the target (0.39168 vs 0.35188, lower-is-better), so we should make a small, low-risk improvement within the same “per-seqpos mean baseline + closed-form weighted ridge correction” approach. The largest likely gain without changing the model family is to make the ridge penalty scale-invariant by standardizing features (and applying the same transform to test), which typically improves conditioning and fit quality for one-hot + continuous mixed features under a fixed alpha. I also standardize the training targets’ deltas per task (then unscale predictions back) so the same alpha behaves similarly across tasks, while preserving identical semantics (still linear ridge on the same features, closed-form solve, same baseline). Finally, I keep submission formatting identical and continue to fill unscored positions/targets exactly as before.'
- What this solution (achieved 0.39122) has done: 'We’re currently worse than the target (0.39139 vs 0.35188, lower-is-better), so the goal is a small, low-risk improvement that preserves your existing “per-seqpos mean baseline + closed-form weighted ridge correction” core logic. The biggest likely gain without changing the model family is to add a tiny amount of position information (normalized `seqpos` and its square) to the ridge features, letting the linear correction capture smooth per-position trends that one-hots/composition don’t fully model. I also make the ridge regularization slightly stronger on the bias term (last column) to reduce overfitting drift while keeping everything closed-form and deterministic. Submission format/paths remain identical and `submission.csv` is still written.'
- What this solution (achieved 0.39122) has done: 'You’re still worse than the target (0.39122 vs 0.35188, lower-is-better), so we should make a small, low-risk improvement without changing your core “per-seqpos mean baseline + closed-form weighted ridge correction” approach. The most likely incremental gain is to fit the ridge corrections separately per target with a target-specific regularization strength (same model/solve, just better-conditioned per-task), while keeping all features, weighting, and the baseline identical. This avoids under/over-regularizing any single target due to different noise/scale, and it’s deterministic and cheap. Submission formatting and the special-case id override remain unchanged.'
- What this solution (achieved 0.39122) has done: 'You’re currently above (worse than) the target MCRMSE, so we should make a small, low-risk improvement while keeping the exact same “per-seqpos mean baseline + closed-form weighted ridge correction” approach. The most likely incremental gain without changing the model family is to tune regularization strength slightly downward (your current alpha=10 is likely over-regularizing after standardization), and to remove the extra intercept penalty that can underfit the global shift. I keep all features, weighting, standardization, targets, and submission formatting identical, and only adjust these ridge penalty details (still closed-form solve, deterministic). This should move the score down toward your target band without any architectural/training-loop changes.'
- What this solution (achieved 0.3912) has done: 'You’re worse than the target (0.39122 vs 0.35188, lower-is-better), so we need a small, low-risk improvement while keeping the same “per-seqpos mean baseline + closed-form weighted ridge correction” approach. The most likely score gain without changing the model family is to train the ridge correction on a more robust target: winsorize (clip) the training deltas using per-position quantiles, which reduces the impact of noisy/outlier labels that this dataset is known to contain. This keeps the exact same features, weighting, standardization, and closed-form solve; it only makes the fitted linear coefficients less dominated by extreme values, usually improving MCRMSE. Submission formatting/paths remain unchanged and `submission.csv` is still written.'
- What this solution (achieved 0.39122) has done: 'We’re currently worse than the target (0.3912 vs 0.35188, lower-is-better), so we should make a small, safe improvement while preserving your exact baseline + closed-form weighted ridge correction core logic. The biggest likely win with minimal risk is to slightly improve the robustness of the winsorization by clipping using weighted quantiles (using the same per-target inverse-variance weights you already compute), so noisy high-error points don’t dominate the clipping thresholds. This keeps the same features, same closed-form ridge solve, same weighting scheme, and same submission formatting; it only refines how outliers are damped before fitting. Everything remains deterministic and finishes quickly.'
- What this solution (achieved 0.39122) has done: 'We’re still worse than the target (0.39122 vs 0.35188, lower-is-better), so we make a small improvement while preserving your exact “per-seqpos mean baseline + closed-form weighted ridge correction” approach. The least invasive change likely to help is to include the provided `predicted_loop_type` one-hot consistently by fixing the feature dimension bug (you reserve 7 dims but your loop alphabet has 7 and needs 7, while your `d` currently mismatches offsets), which can silently waste/corrupt signal. I also make the `valid_mask` mapping faster/safer by using a set for membership (score-neutral but avoids any accidental pandas dtype edge cases). Everything else (features, weighting, winsorization, closed-form solve, baseline fill, and submission format/path) stays the same.'
- What this solution (achieved 0.39025) has done: 'We’re worse than the target (0.39122 vs 0.35188, lower-is-better), so we make a small, low-risk improvement that keeps your exact “per-seqpos mean baseline + closed-form weighted ridge correction” approach. The largest likely gain with minimal change is to use a slightly richer but still deterministic linear feature: add per-position base×structure interaction one-hots (4×3=12) so the ridge can learn that the same base behaves differently when paired/unpaired, without changing any training loop or model family. To avoid overfitting from the extra dimensions, we increase ridge regularization slightly for these new interaction features only (same closed-form solve). Submission format/paths and your special-case id override remain unchanged, and it still writes `submission.csv`.'

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
error_cols = [
    "reactivity_error",
    "deg_error_Mg_pH10",
    "deg_error_pH10",
    "deg_error_Mg_50C",
    "deg_error_50C",
]

seq_scored = int(train["seq_scored"].iloc[0])  # expected 68
seq_length = int(train["seq_length"].iloc[0])  # expected 107

if "SN_filter" in train.columns:
    train_fit = train[train["SN_filter"].astype(int) == 1].reset_index(drop=True)
    if len(train_fit) < 10:  # safety fallback
        train_fit = train.reset_index(drop=True)
else:
    train_fit = train.reset_index(drop=True)

pos_means = {}
for t in target_cols:
    arr = np.vstack(train_fit[t].values).astype(np.float64)  # (n_fit, 68)
    pos_means[t] = arr.mean(axis=0)  # (68,)

BASES = "ACGU"
STRUCTS = "()."
LOOPS = "SMIBHEX"  # 7 loop types

base_to_i = {c: i for i, c in enumerate(BASES)}
struct_to_i = {c: i for i, c in enumerate(STRUCTS)}
loop_to_i = {c: i for i, c in enumerate(LOOPS)}


def _safe_idx(mapper, ch, default=0):
    return mapper.get(ch, default)


def build_features(df_json: pd.DataFrame, upto: int):
    """
    Build per-seqpos feature matrix for first `upto` positions only.
    Features (same core as before) plus one minimal extension:
      - one-hot base (4)
      - one-hot structure (3)
      - one-hot loop_type (7)
      - global base composition for the whole sequence (4), repeated per position
      - normalized position features: pos and pos^2 (2)
      - base×structure interaction one-hot (4*3=12)  [Change: small linear capacity boost]
      - bias term (1)
    Output:
      X: (n_samples*upto, d)
    """
    n = len(df_json)

    n_base = 4
    n_struct = 3
    n_loop = len(LOOPS)  # 7
    n_comp = 4
    n_pos = 2
    n_bx = n_base * n_struct  # 12
    n_bias = 1

    off_base = 0
    off_struct = off_base + n_base
    off_loop = off_struct + n_struct
    off_comp = off_loop + n_loop
    off_pos = off_comp + n_comp
    off_bx = off_pos + n_pos
    off_bias = off_bx + n_bx

    d = off_bias + n_bias  # 4+3+7+4+2+12+1 = 33
    X = np.zeros((n * upto, d), dtype=np.float64)

    denom_pos = max(1.0, float(upto - 1))
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
            bi = _safe_idx(base_to_i, b, 0)
            X[r, off_base + bi] = 1.0

            s = st[p] if p < len(st) else "."
            si = _safe_idx(struct_to_i, s, 2)  # default "."
            X[r, off_struct + si] = 1.0

            l = lp[p] if p < len(lp) else "X"
            X[r, off_loop + _safe_idx(loop_to_i, l, loop_to_i["X"])] = 1.0

            X[r, off_comp : off_comp + n_comp] = comp

            pos = float(p) / denom_pos  # [0,1]
            X[r, off_pos + 0] = pos
            X[r, off_pos + 1] = pos * pos

            X[r, off_bx + (bi * n_struct + si)] = 1.0

            X[r, off_bias] = 1.0

    return X


def weighted_quantile(values, quantiles, sample_weight=None):
    """
    Minimal helper to compute weighted quantiles for 1D arrays.
    Deterministic, numpy-only, and used only to make the existing winsorization
    more robust under the same inverse-variance weights (metric-aligned).
    """
    v = np.asarray(values, dtype=np.float64)
    q = np.asarray(quantiles, dtype=np.float64)

    if sample_weight is None:
        return np.quantile(v, q)

    w = np.asarray(sample_weight, dtype=np.float64)
    w = np.clip(w, 0.0, np.inf)

    if v.size == 0:
        return np.full_like(q, np.nan, dtype=np.float64)

    sorter = np.argsort(v, kind="mergesort")
    v_sorted = v[sorter]
    w_sorted = w[sorter]

    total = w_sorted.sum()
    if not np.isfinite(total) or total <= 0.0:
        return np.quantile(v_sorted, q)

    cdf = np.cumsum(w_sorted) / total
    return np.interp(q, cdf, v_sorted, left=v_sorted[0], right=v_sorted[-1])


n_fit = len(train_fit)

scored_target_cols = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
scored_error_cols = ["reactivity_error", "deg_error_Mg_pH10", "deg_error_Mg_50C"]

Y_scored = np.zeros((n_fit * seq_scored, len(scored_target_cols)), dtype=np.float64)
for k, t in enumerate(scored_target_cols):
    arr = np.vstack(train_fit[t].values).astype(np.float64)  # (n_fit, 68)
    Y_scored[:, k] = arr.reshape(-1)

baseline_scored = np.zeros(
    (n_fit * seq_scored, len(scored_target_cols)), dtype=np.float64
)
for k, t in enumerate(scored_target_cols):
    baseline_scored[:, k] = np.tile(pos_means[t], n_fit)

Y_delta_scored = Y_scored - baseline_scored

X = build_features(train_fit, upto=seq_scored)

eps = 1e-6
W_task = []
if all(c in train_fit.columns for c in scored_error_cols):
    for ec in scored_error_cols:
        e = np.vstack(train_fit[ec].values).astype(np.float64).reshape(-1)
        w = 1.0 / (e**2 + eps)
        w = np.clip(w, 0.05, 20.0)
        W_task.append(w)
else:
    for _ in scored_error_cols:
        W_task.append(np.ones((n_fit * seq_scored,), dtype=np.float64))

x_mu = X.mean(axis=0)
x_sigma = X.std(axis=0)
x_sigma = np.where(x_sigma < 1e-12, 1.0, x_sigma)  # protect constant columns
Xn = (X - x_mu) / x_sigma

alpha_base = 3.0
d = Xn.shape[1]
W_coef = np.zeros((d, len(scored_target_cols)), dtype=np.float64)

y_sigma = Y_delta_scored.std(axis=0)
y_sigma = np.where(y_sigma < 1e-12, 1.0, y_sigma)

Y_delta_scored_2d = Y_delta_scored.reshape(n_fit, seq_scored, len(scored_target_cols))
lo_q, hi_q = 0.01, 0.99
clip_lo = np.zeros((seq_scored, len(scored_target_cols)), dtype=np.float64)
clip_hi = np.zeros((seq_scored, len(scored_target_cols)), dtype=np.float64)
for k in range(len(scored_target_cols)):
    w_flat = W_task[k].reshape(n_fit, seq_scored)
    for p in range(seq_scored):
        vals = Y_delta_scored_2d[:, p, k]
        wts = w_flat[:, p]
        lo, hi = weighted_quantile(vals, [lo_q, hi_q], sample_weight=wts)
        clip_lo[p, k] = lo
        clip_hi[p, k] = hi

Y_delta_scored_clip = np.clip(Y_delta_scored_2d, clip_lo, clip_hi).reshape(
    -1, len(scored_target_cols)
)

n_base = 4
n_struct = 3
n_loop = len(LOOPS)
n_comp = 4
n_pos = 2
n_bx = n_base * n_struct
n_bias = 1
off_base = 0
off_struct = off_base + n_base
off_loop = off_struct + n_struct
off_comp = off_loop + n_loop
off_pos = off_comp + n_comp
off_bx = off_pos + n_pos
off_bias = off_bx + n_bx

diag_add = np.ones(d, dtype=np.float64)
diag_add[off_bx : off_bx + n_bx] = 1.35  # modest extra shrinkage only for interactions

for k in range(len(scored_target_cols)):
    w = W_task[k]
    sqrtw = np.sqrt(w).reshape(-1, 1)
    Xw = Xn * sqrtw
    yw = (Y_delta_scored_clip[:, k : k + 1] / y_sigma[k]) * sqrtw  # (n,1)

    XtX = Xw.T @ Xw

    w_mean = float(np.mean(w))
    alpha_k = alpha_base * (1.0 + 0.25 * np.clip(w_mean - 1.0, -0.8, 3.0))

    XtX.flat[:: XtX.shape[0] + 1] += alpha_k * diag_add

    XtY = Xw.T @ yw
    coef = np.linalg.solve(XtX, XtY).reshape(-1)
    W_coef[:, k] = coef

X_test = build_features(test, upto=seq_scored)
X_testn = (X_test - x_mu) / x_sigma
Yd_test_scored = (X_testn @ W_coef) * y_sigma.reshape(1, -1)  # (n_test*68, 3)



## === cell 3
seqpos = df["id_seqpos"].astype(str).str.split("_").str[-1].astype(int).values
id_only = df["id_seqpos"].astype(str).str.rsplit("_", n=1).str[0]

test_ids = test["id"].astype(str).values
test_id_to_idx = {rid: i for i, rid in enumerate(test_ids)}
test_id_set = set(test_id_to_idx.keys())

for t in target_cols:
    pm = pos_means[t]
    base_fill = np.where(
        seqpos < seq_scored, pm[np.clip(seqpos, 0, seq_scored - 1)], pm[-1]
    ).astype(np.float64)
    df[t] = base_fill

delta_add_scored = np.zeros((len(df), len(scored_target_cols)), dtype=np.float64)
valid_mask = (seqpos < seq_scored) & id_only.isin(test_id_set)

if valid_mask.any():
    valid_idx = np.where(valid_mask.values)[0]
    test_rows = id_only.iloc[valid_idx].map(test_id_to_idx).astype(int).values
    pos = seqpos[valid_idx].astype(int)
    flat = test_rows * seq_scored + pos
    delta_add_scored[valid_idx, :] = Yd_test_scored[flat, :]

for k, t in enumerate(scored_target_cols):
    df[t] = (df[t].values.astype(np.float64) + delta_add_scored[:, k]).astype(float)



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
