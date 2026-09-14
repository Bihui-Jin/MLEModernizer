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

0.48578

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.63824) has done: 'The current notebook fails immediately because it tries to load a non-existent `../input/worst-submission/ensemble52.csv`, so `df` is never created and all later cells error. I replace that input with the competition-provided `sample_submission.csv` (same required columns/row count) to guarantee a valid submission is produced. To keep the core intent (“make a submission”) while being score-safe and stable, I output constant predictions (zeros) and preserve the existing row-wise tweak (which becomes a no-op). I also add a small path-fallback so it works with either `/kaggle/input/...` style or your provided `kaggle/data/...` layout.'
- What this solution (achieved 0.63824) has done: 'Your current approach writes an all-zeros submission, which is valid but far from the target score; the smallest legitimate improvement (without changing the “no model training” core logic) is to use a stronger constant baseline derived only from the training labels. I compute the per-target global mean over all scored positions in `train.json` (using only the 68 scored entries), and fill every test row with those means (and reuse the same means for the unscored targets to keep the required columns). I also preserve your existing path-fallback logic and keep the submission row order identical to `sample_submission.csv` to avoid any alignment issues. This should move MCRMSE substantially downward toward the target while remaining stable and fast under 600 seconds.'
- What this solution (achieved 0.47906) has done: 'I fix the runtime error by loading `train.json` with the correct JSON orientation (it is newline-delimited “records”, so `pd.read_json(..., lines=True)`), which removes the “Trailing data” failure. I also make the mean computation match the competition’s scored setup by averaging only the first `seq_scored` positions for each training row, which should legitimately improve MCRMSE toward your target while keeping the “constant baseline from train” core logic unchanged. Finally, I keep the sample submission row order/columns intact and ensure numeric, non-NaN outputs before writing `submission.csv`.'
- What this solution (achieved 0.47889) has done: 'Your current constant-mean baseline is valid but still far from the target (lower is better), so the smallest score-improving change is to keep the exact same “no training, constant prediction” core logic while making the constants better match the public test distribution. Concretely, we compute means using only high-quality training rows (`SN_filter==1`) since the test set was similarly filtered, and (optionally) weight each row’s contribution by `signal_to_noise` to emphasize more reliable measurements—this stays within the same baseline approach and aligns with the metric. We also remove the no-op/tweak cell that doesn’t change predictions, to avoid any accidental dtype quirks, while preserving the exact submission ordering from `sample_submission.csv`. The rest (paths, columns, CSV writing, NaN checks) stays the same.'
- What this solution (achieved 0.47888) has done: 'We keep your “constant baseline from train” core logic, but make the constants better match the test distribution by (1) using only `SN_filter==1` rows (already done) and (2) additionally filtering out the noisiest remaining rows via a simple, deterministic `signal_to_noise` quantile cutoff, which is consistent with how the test set was curated. We also remove the signal-to-noise weighting (which can overfit to a few extreme values) and instead use a plain mean over the filtered high-quality subset—this is still the same constant-prediction approach, just a more robust estimate. Finally, we keep the submission row order identical to `sample_submission.csv` and still write a valid `submission.csv`.'
- What this solution (achieved 0.47889) has done: 'Your current constant-mean baseline is stable but still above the target (lower is better), so we keep the same “no model, constant predictions derived from train” core logic and make the mean estimate slightly more test-like. Concretely, we (1) continue using `SN_filter==1`, and (2) replace the weak 10% S/N cutoff with a more selective, deterministic cutoff to better match the curated test distribution, while ensuring we never end up with an empty training subset. We also compute the constants using only the *scored* targets’ first `seq_scored` positions (already aligned), and then reuse those constants for the two unscored columns to keep required submission columns valid. The rest (paths, ordering from `sample_submission.csv`, NaN checks, and writing `submission.csv`) stays unchanged.'
- What this solution (achieved 0.47906) has done: 'We keep your constant-prediction-from-train baseline (same core logic, no model training) but make the estimated constants more robust and closer to the curated test distribution. Specifically, we compute *trimmed means* (winsorized) on the SN_filter==1 subset after applying the same S/N quantile filter you already use, which reduces the influence of extreme/noisy label outliers that hurt RMSE. We also clip final predictions to a conservative range derived from the filtered training distribution (e.g., 0.5%–99.5% quantiles) to avoid rare extreme constants causing unnecessary error. Submission row order/format and file writing remain unchanged.'
- What this solution (achieved 0.50559) has done: 'We keep your “constant baseline from train” approach intact, but make the constant estimates slightly more robust by using a per-target median (instead of mean) over the same scored positions and the same filtered training subset, which typically reduces RMSE under heavy-tailed noise. We also ensure the constants stay within the existing conservative clip bounds you already compute, so predictions remain numerically safe and stable. No model/training/feature changes are introduced; the only change is the aggregation statistic used to produce the constants. This is a minimal, legitimate adjustment that should reduce the leaderboard score (lower is better) toward your target.'
- What this solution (achieved 0.47892) has done: 'We need to lower MCRMSE from 0.50559 toward 0.35188 (lower is better), so we should improve predictions while keeping your constant-baseline core logic intact. The minimal lever that’s directly relevant is how we aggregate training labels into constants: for RMSE, a (winsorized) mean typically beats a median, so we switch back from winsorized median to winsorized mean while keeping the same SN_filter and signal_to_noise quantile filtering. To better match the curated test distribution without changing the approach, we also tune the S/N cutoff slightly more selective (50% quantile with safe fallbacks) but keep the same fallback logic to avoid empty/too-small subsets. Everything else (paths, scored-position slicing, clipping, output ordering, and writing submission.csv) stays the same to preserve semantics and stability.'
- What this solution (achieved 0.47894) has done: 'To move your MCRMSE down toward the target while preserving the same “constant predictions derived from train” core logic, I only adjust how the constants are estimated. Specifically, I compute the constant per target as a *row-wise mean over each sequence’s scored positions*, then take a winsorized mean across sequences; this better matches the metric’s per-sequence averaging and reduces overweighting any single sequence. I keep your existing SN_filter==1 and signal_to_noise quantile filtering (with the same safe fallbacks) and keep the same clipping step to remain numerically stable. Everything else (paths, columns, ordering, and writing `submission.csv`) remains unchanged.'
- What this solution (achieved 0.42336) has done: 'To reduce MCRMSE from 0.47894 toward the 0.35188 target (lower is better) while keeping your constant-baseline core logic, I only change how the constants are estimated. Instead of a single global constant per target, we compute a **position-wise constant profile** (length 68) from the filtered training set, which better matches the per-position scoring and typically improves RMSE without introducing any model/training. We keep your existing SN_filter and signal_to_noise quantile filtering and add the smallest safe robustness: winsorize per-position means and then fill all test rows by seqpos (and reuse the last scored position for unscored positions >67). Submission ordering/format stays identical to `sample_submission.csv`, and we still write `submission.csv`.'
- What this solution (achieved 0.48578) has done: 'We need to decrease your MCRMSE from 0.42336 toward the 0.35188 target (lower is better) while keeping the same “no model, constant(ish) predictions derived from train” core logic. The smallest, directly relevant improvement is to estimate the position-wise profiles with a metric-aligned aggregation: compute per-sequence RMSE-optimal mean per position, but weight each training sequence by the inverse variance implied by its provided per-position experimental errors (downweight noisy measurements). We keep your existing SN_filter and signal_to_noise filtering, keep the same position-wise profile approach, and keep the submission formatting/order identical. We also add safe fallbacks when error columns are missing/non-finite so the pipeline always produces a valid `submission.csv`.'
- What this solution (achieved 0.48578) has done: 'We need to lower your MCRMSE (0.48578) toward the target 0.35188 (lower is better), so we should make a small, metric-aligned improvement without changing the “position-wise constant profile from filtered train” core approach. The biggest issue in your current code is that the “robust” step recomputes per-position weighted averages using weights from the full (unwinsorized) set but values from a reduced (winsorized) set, which misaligns weights/values and can silently degrade the estimate. I fix this by winsorizing via clipping (not dropping) so weights and values stay aligned, and compute the final robust per-position estimate as a proper weighted average over the aligned masked arrays. Everything else (paths, filtering, 68-length profile, seqpos mapping, required columns, and writing `submission.csv`) remains unchanged.'

# 9. Code solution

## === cell 0
import os
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)



## === cell 1
candidate_paths = [
    "/kaggle/input/stanford-covid-vaccine/sample_submission.csv",
    "/kaggle/input/sample_submission.csv",
    "../input/stanford-covid-vaccine/sample_submission.csv",
    "../input/sample_submission.csv",
    "/kaggle/data/stanford-covid-vaccine/sample_submission.csv",
    "/kaggle/data/sample_submission.csv",
]

sample_path = next((p for p in candidate_paths if os.path.exists(p)), None)
if sample_path is None:
    raise FileNotFoundError(
        "Could not locate sample_submission.csv in expected Kaggle paths. "
        f"Tried: {candidate_paths}"
    )

df = pd.read_csv(sample_path)

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
    raise ValueError(f"sample_submission.csv missing required columns: {missing}")
df = df[required_cols].copy()



## === cell 2
train_candidate_paths = [
    "/kaggle/input/stanford-covid-vaccine/train.json",
    "/kaggle/input/train.json",
    "../input/stanford-covid-vaccine/train.json",
    "../input/train.json",
    "/kaggle/data/stanford-covid-vaccine/train.json",
    "/kaggle/data/train.json",
]
train_path = next((p for p in train_candidate_paths if os.path.exists(p)), None)
if train_path is None:
    raise FileNotFoundError(
        "Could not locate train.json in expected Kaggle paths. "
        f"Tried: {train_candidate_paths}"
    )

train_df = pd.read_json(train_path, lines=True)

target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
err_map = {
    "reactivity": "reactivity_error",
    "deg_Mg_pH10": "deg_error_Mg_pH10",
    "deg_pH10": "deg_error_pH10",
    "deg_Mg_50C": "deg_error_Mg_50C",
    "deg_50C": "deg_error_50C",
}

if "SN_filter" in train_df.columns:
    train_use = train_df.loc[train_df["SN_filter"].astype(int) == 1].copy()
else:
    train_use = train_df.copy()
if len(train_use) == 0:
    train_use = train_df.copy()

sn_thr_used = None
if "signal_to_noise" in train_use.columns:
    s2n = pd.to_numeric(train_use["signal_to_noise"], errors="coerce").astype(
        np.float64
    )
    if np.isfinite(s2n.values).any():
        for q, min_rows in [(0.50, 200), (0.25, 200), (0.10, 1)]:
            thr = float(np.nanquantile(s2n.values, q))
            filtered = train_use.loc[s2n >= thr].copy()
            if len(filtered) >= min_rows:
                train_use = filtered
                sn_thr_used = thr
                break

if len(train_use) == 0:
    train_use = (
        train_df.loc[train_df["SN_filter"].astype(int) == 1].copy()
        if "SN_filter" in train_df.columns
        else train_df.copy()
    )

SEQ_SCORED = 68


def _winsorize_clip_1d(x, lo_q=0.02, hi_q=0.98):
    """
    Keep array length unchanged but clip extreme values (winsorization).
    This preserves alignment with any parallel arrays (e.g., weights).
    """
    x = np.asarray(x, dtype=np.float64)
    mask = np.isfinite(x)
    if not np.any(mask):
        return x
    lo = float(np.quantile(x[mask], lo_q))
    hi = float(np.quantile(x[mask], hi_q))
    out = x.copy()
    out[mask] = np.clip(out[mask], lo, hi)
    return out


def _build_position_matrix(values, seq_scored_col, max_len=SEQ_SCORED):
    """
    Returns matrix shape (n_samples, max_len) padded with NaNs,
    containing only the scored prefix for each sample.
    """
    n = len(values)
    mat = np.full((n, max_len), np.nan, dtype=np.float64)
    seq_sc = seq_scored_col.astype(int).values
    for i, (v, m) in enumerate(zip(values, seq_sc)):
        m = int(min(m, max_len))
        arr = np.asarray(v, dtype=np.float64)[:m]
        if arr.size:
            mat[i, :m] = arr
    return mat


def _build_weight_matrix_from_errors(
    err_values, seq_scored_col, max_len=SEQ_SCORED, eps=1e-6
):
    """
    Build inverse-variance weights from provided per-position errors:
    w = 1 / (err^2 + eps). Returns (n_samples, max_len) with NaNs where missing.
    """
    n = len(err_values)
    wmat = np.full((n, max_len), np.nan, dtype=np.float64)
    seq_sc = seq_scored_col.astype(int).values
    for i, (e, m) in enumerate(zip(err_values, seq_sc)):
        m = int(min(m, max_len))
        arr = np.asarray(e, dtype=np.float64)[:m]
        if arr.size:
            w = 1.0 / (np.square(arr) + eps)
            w[~np.isfinite(w)] = np.nan
            wmat[i, :m] = w
    return wmat


pos_profiles = {}
pos_clip_bounds = {}

seq_scored_col = train_use["seq_scored"]

for col in target_cols:
    mat = _build_position_matrix(
        train_use[col].values, seq_scored_col, max_len=SEQ_SCORED
    )

    err_col = err_map.get(col, None)
    if err_col is not None and err_col in train_use.columns:
        wmat = _build_weight_matrix_from_errors(
            train_use[err_col].values, seq_scored_col, max_len=SEQ_SCORED
        )
        num = np.nansum(mat * wmat, axis=0)
        den = np.nansum(wmat, axis=0)
        pos_mean = np.divide(
            num, den, out=np.full(SEQ_SCORED, np.nan, dtype=np.float64), where=(den > 0)
        )
        fallback = np.nanmean(mat, axis=0).astype(np.float64)
        pos_mean = np.where(np.isfinite(pos_mean), pos_mean, fallback).astype(
            np.float64
        )
    else:
        pos_mean = np.nanmean(mat, axis=0).astype(np.float64)

    finite_pm = pos_mean[np.isfinite(pos_mean)]
    if finite_pm.size:
        lo = float(np.quantile(finite_pm, 0.005))
        hi = float(np.quantile(finite_pm, 0.995))
    else:
        lo, hi = -10.0, 10.0
    pos_clip_bounds[col] = (lo, hi)
    pos_mean = np.clip(pos_mean, lo, hi)

    pos_mean_robust = np.empty(SEQ_SCORED, dtype=np.float64)
    if err_col is not None and err_col in train_use.columns:
        for j in range(SEQ_SCORED):
            yj = mat[:, j].astype(np.float64, copy=False)
            wj = wmat[:, j].astype(np.float64, copy=False)
            mask = np.isfinite(yj) & np.isfinite(wj) & (wj > 0)
            if np.any(mask):
                yj_wins = _winsorize_clip_1d(yj, lo_q=0.02, hi_q=0.98)
                pos_mean_robust[j] = float(np.average(yj_wins[mask], weights=wj[mask]))
            else:
                yj_wins = _winsorize_clip_1d(yj, lo_q=0.02, hi_q=0.98)
                finite = yj_wins[np.isfinite(yj_wins)]
                pos_mean_robust[j] = (
                    float(np.mean(finite)) if finite.size else float(pos_mean[j])
                )
    else:
        for j in range(SEQ_SCORED):
            yj = mat[:, j]
            yj_wins = _winsorize_clip_1d(yj, lo_q=0.02, hi_q=0.98)
            finite = yj_wins[np.isfinite(yj_wins)]
            pos_mean_robust[j] = (
                float(np.mean(finite)) if finite.size else float(pos_mean[j])
            )

    pos_mean_robust = np.clip(pos_mean_robust, lo, hi)
    pos_profiles[col] = pos_mean_robust

print("Using SN_filter==1 + S/N quantile filter (try 50%, fallback 25% then 10%)")
print("S/N threshold used:", sn_thr_used)
print("Train rows used for estimation:", len(train_use), "out of", len(train_df))
print(
    "Per-target clip bounds applied to position-wise means (0.5%..99.5% across positions):",
    pos_clip_bounds,
)

seqpos = df["id_seqpos"].astype(str).str.rsplit("_", n=1).str[-1].astype(int).values
seqpos_clipped = np.clip(seqpos, 0, SEQ_SCORED - 1)

for col in target_cols:
    df[col] = pos_profiles[col][seqpos_clipped]



## === cell 3
sequences = list(set(["_".join(v.split("_")[:-1]) for v in df.id_seqpos.values]))
print("Unique sequence ids in submission:", len(sequences))



## === cell 4
for c in required_cols[1:]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
if df[required_cols].isna().any().any():
    raise ValueError("Submission contains NaNs after coercion; aborting.")



## === cell 5
df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", df.shape)
print(df.head())
