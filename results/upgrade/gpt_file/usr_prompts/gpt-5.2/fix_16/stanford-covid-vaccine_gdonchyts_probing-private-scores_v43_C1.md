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

0.42186

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.63824) has done: 'I remove the dependency on the missing `../input/worst-submission/ensemble52.csv` by instead loading the provided `sample_submission.csv` as the baseline, which fixes the `FileNotFoundError` and subsequent `NameError`s. I keep the existing “core logic” intact (i.e., edit one id’s `reactivity` values) but make it robust by only applying the change if that id actually exists in the submission. I also standardize input paths to the known-available `../input/stanford-covid-vaccine/` location and add a small validation to ensure the submission has the correct columns and row count. Finally, it always write a valid `submission.csv` with the required `.csv` suffix.'
- What this solution (achieved 0.63824) has done: 'Your current code intentionally destroys performance by setting one sequence’s `reactivity` to 100, which pushes the MCRMSE far away from your (lower-is-better) target. To move the score down toward 0.3519 with the smallest possible change, I remove that harmful override so predictions remain identical to the provided `sample_submission.csv` baseline. I keep your existing submission validation/alignment logic intact to guarantee a valid CSV with correct ordering and columns. This should substantially reduce error versus your current 0.63824 while preserving the same overall “baseline-submission” approach.'
- What this solution (achieved 0.42418) has done: 'Your current pipeline is effectively submitting the unmodified `sample_submission.csv`, which is already a reasonable baseline but is still far from your (lower-is-better) target. With minimal changes and without changing the overall “no-model” approach, we can legitimately improve by replacing the constant baseline predictions with per-position averages learned from the training set (a simple, stable prior that usually beats the sample baseline on MCRMSE). This keeps the core workflow (read train/test, create a submission in the required long format, write `submission.csv`) while only changing how the prediction columns are filled. We also keep your existing submission alignment/validation to guarantee the output format is correct.'
- What this solution (achieved 0.42166) has done: 'We keep your current “per-position mean prior from train” approach, but compute those means on the higher-quality subset of training data (`SN_filter == 1`) which is closer to the test distribution and typically reduces MCRMSE with a very small change. We also fill unscored positions (seqpos ≥ 68) with the last scored position’s mean instead of leaving the sample_submission constants there, because those rows still contribute (slightly) and this is consistent with the same prior. Finally, we keep your existing id_seqpos alignment/validation so the output stays a valid Kaggle submission.'
- What this solution (achieved 0.42178) has done: 'We keep your current “per-position mean prior from train” core logic intact, but make one small, targeted improvement: compute means with signal-to-noise weighting so higher-quality measurements influence the prior more (this often reduces MCRMSE vs a plain mean). We still restrict to `SN_filter==1` as you already do, and we keep the same submission alignment/validation and output format. This is a minimal change (only how `pos_means` is computed) and should move your score down toward the 0.3519 target without changing the overall approach.'
- What this solution (achieved 0.44436) has done: 'We keep your exact “per-position prior from train” core logic and submission alignment, but make the prior slightly closer to the evaluation distribution by (1) computing position-wise **medians** (more robust than means for noisy/outlier-prone targets) on `SN_filter==1`, and (2) applying a tiny amount of **shrinkage** toward the overall per-target median to reduce variance without changing the approach. This is still a deterministic, no-model baseline and only changes how `pos_means` is estimated (the rest of the pipeline, including filling unscored positions and submission formatting, stays the same). These two changes typically reduce MCRMSE a bit versus a plain/weighted mean, moving your 0.42178 down toward the 0.3519 target while remaining minimal and stable. The script still writes a valid `submission.csv` with correct columns/order/row count.'
- What this solution (achieved 0.43481) has done: 'We keep your exact “per-position prior from train” approach and submission alignment, but adjust the estimator to be closer to what MCRMSE rewards: position-wise central tendency on `SN_filter==1` with slightly less bias from heavy tails. Concretely, we (1) switch from plain median to a small convex blend of **median and mean** per position (often improves RMSE vs median-only), and (2) reduce the global shrinkage a bit so we don’t over-flatten real positional structure (your current shrink=0.10 is likely hurting). These are minimal changes limited to how `pos_means` is computed; no model/training loop is introduced and output format stays identical. The script still write a valid `submission.csv` with the correct columns/order/row count.'
- What this solution (achieved 0.42252) has done: 'Your current approach is a simple per-position prior; to move the (lower-is-better) MCRMSE down from 0.43481 toward 0.35188 with minimal risk, we keep the exact same pipeline but adjust the estimator to be slightly more RMSE-friendly. Specifically, we (1) switch from the current median/mean blend to a small amount of trimming (winsorization) before taking the mean, which usually reduces RMSE under heavy-tailed noise without changing the “per-position central tendency” logic, and (2) set shrinkage to ~0 (your current shrink can over-flatten true position structure and hurt). Everything else (data loading, SN_filter usage, filling unscored positions, id_seqpos alignment, and writing submission.csv) stays the same to preserve evaluation semantics and ensure a valid submission.'
- What this solution (achieved 0.42186) has done: 'We keep your exact “per-position prior from train (SN_filter==1) + fill unscored with last position” pipeline, but make the estimator slightly more MCRMSE-friendly with minimal risk. Concretely, we (1) compute position-wise means using **per-sample weights = signal_to_noise** so higher-quality RNAs contribute more, and (2) tune winsorization slightly (from 5% to 2%) to reduce over-clipping of legitimate signal while still damping heavy tails. This stays within your current core logic (no model/training loop changes), is deterministic, and should move the score down (better) from 0.42252 toward your 0.35188 target. Submission alignment/validation and output format remain unchanged to guarantee a valid `submission.csv`.'
- What this solution (achieved 0.42555) has done: 'We keep your exact “per-position prior from train (SN_filter==1) + fill unscored with last position” pipeline, but make one small estimator tweak that is often more RMSE-friendly: instead of winsorizing then taking a mean, we winsorize then take a **trimmed mean** (drop the most extreme values after clipping) per position. This remains the same core logic (a deterministic per-position central tendency prior) and only changes how `pos_means` is computed, which should cautiously reduce your MCRMSE from 0.42186 toward the 0.35188 target. We keep your SNR weighting, submission alignment, and column/order validation unchanged to ensure a valid `submission.csv`. The change is lightweight and runs comfortably within the time limit.'
- What this solution (achieved 0.42186) has done: 'Your current score (0.42555, lower-is-better) is worse than the target (0.35188), so we should cautiously improve while keeping the same “per-position prior from SN_filter==1 with SNR weighting” core logic. The smallest high-impact fix is to align training aggregation with the metric by computing the per-position prior using only the **3 scored targets** (reactivity, deg_Mg_pH10, deg_Mg_50C) and then copy those priors into the unscored targets (deg_pH10, deg_50C), since the latter are not evaluated and currently add noise to the prior estimation. We also remove the extra trimmed-mean step (keep winsorization + weighted mean) because MCRMSE is RMSE-based and trimmed means can introduce bias; this is a minimal estimator tweak that keeps the same overall approach. Submission formatting/alignment logic is kept intact to guarantee a valid `submission.csv`.'
- What this solution (achieved 0.42162) has done: 'We keep your exact “per-position prior from train (SN_filter==1) + SNR-weighted winsorized mean + fill unscored with last position” pipeline, but make one small adjustment that typically reduces RMSE for this competition: apply a light, deterministic **per-position shrinkage toward the global center** for each scored target. This reduces variance from position-specific noise while keeping the same estimator family and semantics, and it’s especially helpful when test distribution is smoother than raw per-position estimates. We keep your “copy scored priors into unscored targets” behavior and all submission alignment/validation exactly as-is to guarantee a valid `submission.csv`. The only functional change is setting `shrink` to a small nonzero value.'
- What this solution (achieved 0.42186) has done: 'We keep your exact “SN_filter==1 + SNR-weighted winsorized per-position mean + fill unscored with last position + copy scored priors into unscored targets” pipeline, but tune the only knob currently influencing bias/variance: the shrinkage strength. Since your current score (0.42162) is still worse than the target (0.35188) in a lower-is-better metric, we should cautiously improve; the smallest plausible improvement is reducing shrinkage (your 0.05 likely over-flattens genuine positional signal). I change `shrink` from `0.05` to `0.0` (no shrinkage), leaving every other detail identical to preserve core logic and semantics. The script still write a valid `submission.csv` with correct columns and ordering.'
- What this solution (achieved 0.42186) has done: 'Your current score (0.42186, lower-is-better) is still far from the target (0.35188), so we should make a small, low-risk improvement while keeping the exact same “per-position prior from train (SN_filter==1) + SNR-weighted winsorized mean” core logic. The most direct fix is to ensure the train aggregation matches what the metric scores: compute the per-position prior using only the first `seq_scored` positions (already done) but also compute the global center using the same per-position weighting without reshaping/`repeat` (which can unintentionally distort weighting and add unnecessary numerical noise). We keep winsorization, SNR weighting, and the same filling rules, but make the weighted-global-center computation consistent and stable (weighted mean of per-position weighted means). This should nudge MCRMSE down slightly without changing the approach or submission semantics.'
- What this solution (achieved 0.42186) has done: 'To move your lower-is-better MCRMSE down toward the 0.3519 target with minimal changes, I keep your exact “SN_filter==1 + SNR-weighted winsorized per-position prior + fill unscored with last position + copy scored priors into unscored targets” pipeline. The single adjustment is to compute the global center in a way that’s consistent with your SNR weighting (a weighted mean over all clipped values), instead of an unweighted mean of per-position weighted means, which can slightly mis-calibrate the shrink/centering behavior even when `shrink=0`. This is a tiny, safe change (no model/training loop changes) that often improves RMSE a bit by improving calibration. Output formatting/alignment and submission writing remain unchanged so you still get a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)



## === cell 1
df_test = pd.read_json("../input/stanford-covid-vaccine/test.json", lines=True)
df_train = pd.read_json("../input/stanford-covid-vaccine/train.json", lines=True)

df = pd.read_csv("../input/stanford-covid-vaccine/sample_submission.csv")



## === cell 2
sequences = list(set(df_test[df_test.seq_length != 130].id))
sequences.sort()



## === cell 3
sequences[-10:] if len(sequences) >= 10 else sequences



## === cell 4
target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
scored_target_cols = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]

seq_scored = int(df_train["seq_scored"].iloc[0])  # expected 68

df_train_used = df_train
if "SN_filter" in df_train.columns:
    df_train_used = df_train[df_train["SN_filter"] == 1].reset_index(drop=True)
    if len(df_train_used) == 0:
        df_train_used = df_train

pos_means = {}

winsor_q = 0.02  # keep your current clipping

shrink = 0.0

use_snr_weighting = "signal_to_noise" in df_train_used.columns
if use_snr_weighting:
    w = df_train_used["signal_to_noise"].astype(np.float32).values
    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    w = np.clip(w, 0.0, 1e6)
    if float(w.sum()) <= 0:
        use_snr_weighting = False
        w = None
else:
    w = None


def _weighted_mean_1d(x: np.ndarray, w: np.ndarray) -> np.float32:
    x = x.astype(np.float32, copy=False)
    w = w.astype(np.float32, copy=False)
    mask = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if not np.any(mask):
        return np.float32(np.nan)
    xv = x[mask]
    wv = w[mask]
    den = float(np.sum(wv))
    if den <= 0:
        return np.float32(np.nan)
    return np.float32(np.sum(xv * wv) / den)


for c in scored_target_cols:
    arr = np.asarray(df_train_used[c].tolist(), dtype=np.float32)  # (n_samples, 68)
    if arr.ndim != 2 or arr.shape[1] != seq_scored:
        raise ValueError(
            f"Unexpected shape for {c}: {arr.shape}, expected (*, {seq_scored})"
        )

    lo = np.nanquantile(arr, winsor_q, axis=0).astype(np.float32)
    hi = np.nanquantile(arr, 1.0 - winsor_q, axis=0).astype(np.float32)
    arr_clip = np.clip(arr, lo[None, :], hi[None, :])

    if use_snr_weighting:
        pos_center = np.empty((seq_scored,), dtype=np.float32)
        for j in range(seq_scored):
            pos_center[j] = _weighted_mean_1d(arr_clip[:, j], w)

        num = np.nansum(arr_clip * w[:, None])
        den = np.nansum(w) * seq_scored
        global_center = np.float32(num / den) if den > 0 else np.float32(np.nan)

        if not np.isfinite(global_center):
            global_center = np.float32(np.nanmean(arr_clip))
    else:
        pos_center = np.nanmean(arr_clip, axis=0).astype(np.float32)
        global_center = np.float32(np.nanmean(arr_clip))

    if not np.all(np.isfinite(pos_center)):
        fallback = global_center
        if not np.isfinite(fallback):
            fallback = np.float32(0.0)
        pos_center = np.where(np.isfinite(pos_center), pos_center, fallback).astype(
            np.float32
        )

    pos_means[c] = (1.0 - shrink) * pos_center + shrink * global_center

pos_means["deg_pH10"] = pos_means["deg_Mg_pH10"]
pos_means["deg_50C"] = pos_means["deg_Mg_50C"]

seqpos = df["id_seqpos"].str.rsplit("_", n=1).str[-1].astype(int).values
scored_mask = seqpos < seq_scored
unscored_mask = ~scored_mask

for c in target_cols:
    vals = df[c].values.astype(np.float32, copy=True)
    vals[scored_mask] = pos_means[c][seqpos[scored_mask]]

    if np.any(unscored_mask):
        vals[unscored_mask] = float(pos_means[c][-1])

    df[c] = vals

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
    raise ValueError(f"Submission is missing required columns: {missing}")

sample = pd.read_csv("../input/stanford-covid-vaccine/sample_submission.csv")
if len(df) != len(sample) or not df["id_seqpos"].equals(sample["id_seqpos"]):
    df = sample[["id_seqpos"]].merge(
        df, on="id_seqpos", how="left", suffixes=("", "_y")
    )
    for c in required_cols[1:]:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = df[c].fillna(sample[c].values)

df = df[required_cols]



## === cell 5
df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", df.shape)
print(df.head())
