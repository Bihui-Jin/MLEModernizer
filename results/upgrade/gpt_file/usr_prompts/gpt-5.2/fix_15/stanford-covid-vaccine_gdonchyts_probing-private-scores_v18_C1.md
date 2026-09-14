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

0.3601225065950412

# 6. Current score

0.46467

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.63824) has done: 'I remove the dependency on the missing `../input/worst-submission/ensemble52.csv` and instead load the provided competition `sample_submission.csv` as the base submission template so the notebook always runs in this environment. Since your current code is only doing a simple global scaling, I preserve that core behavior by applying the same division to the prediction columns (leaving `id_seqpos` untouched). I also make the input path resolution robust by trying the known available locations you listed. Finally, I ensure a valid `submission.csv` is written with exactly the required 6 columns and correct row count.'
- What this solution (achieved 0.42418) has done: 'Your current pipeline is only scaling the provided sample submission (which contains placeholder zeros), so it cannot approach the target score; we need to generate real predictions from `train.json`/`test.json` while keeping the approach simple and fast. I replace the “divide-by-1.09” step with a minimal baseline model that predicts the per-position mean of each target across the training set (same evaluation semantics: per-base regression, no architecture/training loop changes because there currently isn’t one). This drastically reduce MCRMSE versus zeros and should move you much closer to the target without adding heavy dependencies. I also ensure the submission rows align exactly to `sample_submission.csv`’s `id_seqpos` ordering and that positions beyond `seq_scored` get a safe fill value.'
- What this solution (achieved 0.42196) has done: 'We keep your core “per-position mean baseline” intact, but make a small change that typically improves MCRMSE: compute those means using only high-quality training samples (`SN_filter==1`) so noisy sequences don’t inflate error. We also add a lightweight shrinkage blend between per-position means and global means to reduce variance at positions with higher noise, which tends to help a mean-based baseline without changing the modeling approach. Finally, we keep the same submission row alignment logic, still filling non-scored positions safely, and ensure `submission.csv` is produced identically formatted.'
- What this solution (achieved 0.4923) has done: 'We keep your exact “per-position mean baseline + global-mean shrinkage + SN_filter==1” core logic, but make two minimal score-improving adjustments that typically reduce MCRMSE without changing the modeling approach. First, we compute **weighted means** per position using the provided per-position measurement errors as inverse-variance weights, so high-uncertainty labels contribute less. Second, we tune the shrinkage strength slightly upward in a conservative way (still a simple convex blend) to reduce variance from noisy positions; this should move your 0.42196 closer to the 0.3601 target. Submission formatting, ordering, and fill behavior remain identical, and the script still finishes quickly and writes a valid `submission.csv`.'
- What this solution (achieved 0.4923) has done: 'We keep your exact baseline (SN_filter==1, inverse-variance weighted per-position means, and global-mean shrinkage) but fix the main score regression source: you are training on all 5 targets even though Kaggle scores only 3, so the shrinkage/blending is being influenced by unscored targets and can indirectly worsen the scored ones via shared choices like `shrink_alpha` and tail fill. We compute the statistics (pos_means/global_means/tail) using only the three scored targets, then still output all five targets by filling the two unscored columns with their (simple) global weighted means (stable and harmless to the metric). This preserves the core logic and semantics (mean baseline + shrinkage + weighting) while focusing it on what is actually scored, which should move MCRMSE down toward the 0.3601 target without introducing any new modeling complexity. Submission formatting and row alignment remain identical.'
- What this solution (achieved 0.4923) has done: 'I keep your exact baseline approach (SN_filter==1, inverse-variance weighted per-position means + global-mean shrinkage, and safe tail/unscored fills), but fix the main reason the score is stuck: the training labels contain many invalid/noisy entries (notably values < -0.5) that the competition itself filtered out for test, and those outliers distort the weighted means. Concretely, I add a minimal per-position validity mask (drop targets < -0.5 and non-finite values) before computing weighted means, and I compute the “tail” fill as the mean of the last few scored positions instead of a single last position to reduce variance. These are small statistical hygiene changes that preserve your core logic and should move MCRMSE down toward the 0.3601 target without adding any new model/training complexity. Submission formatting, ordering, and row alignment remain identical and it still write a valid `submission.csv`.'
- What this solution (achieved 0.49179) has done: 'Your current baseline is likely being hurt by (1) an overly aggressive inverse-variance weighting (very small reported errors can dominate means) and (2) a shrinkage factor that’s too low given the label noise/outliers, which can increase per-position variance and worsen MCRMSE. I keep the exact same core approach (SN_filter==1, per-position weighted means + global-mean shrinkage, scored-only stats, and safe tail/unscored fills) but add a minimal, standard stabilization: clip the per-position errors to a small floor and cap the maximum weight to prevent single observations from dominating. I also slightly increase `shrink_alpha` (still a convex blend) to reduce variance, which typically moves this kind of baseline downward toward your 0.360 target. Submission formatting and ordering remain identical, and the script still writes a valid `submission.csv`.'
- What this solution (achieved 0.48709) has done: 'I keep your current “SN_filter==1 + inverse-variance weighted per-position mean + global shrinkage + tail fill” baseline intact, but make two minimal statistical stabilizations that usually reduce MCRMSE for this competition. First, I compute a *softly trimmed* weighted mean by clipping extreme target values per position (winsorization) before averaging, which reduces the impact of remaining outliers that still pass your `valid_min` filter. Second, I replace the fixed `shrink_alpha` with a tiny, data-driven per-position shrinkage based on the effective sample size (Neff) of the weights, so noisy positions get slightly more global shrinkage while stable positions remain mostly per-position—this preserves your exact semantics (convex blend) but improves calibration. Submission alignment/format stays exactly the same and it still write a valid `submission.csv`.'
- What this solution (achieved 0.46758) has done: 'We keep your exact baseline (SN_filter==1, weighted per-position means with winsorization, and per-position shrinkage toward global means) but fix two small statistical choices that are likely keeping your score far from the 0.3601 target. First, we compute the inverse-variance weights using the competition’s provided `*_error_*` more appropriately by adding the known measurement noise floor (and clipping weights), which prevents overweighting artificially tiny errors. Second, we make the winsorization and tail-fill slightly more robust (based only on scored positions) and tune the shrinkage to be a bit less aggressive on well-supported positions while still shrinking noisy ones—these are minimal changes that should reduce MCRMSE without changing the modeling approach. The submission formatting/order/alignment stays identical and still writes a valid `submission.csv`.'
- What this solution (achieved 0.46768) has done: 'Your current baseline is already valid and stable, so the smallest likely improvement toward the 0.3601 target is to fix a subtle weighting inconsistency: you winsorize target values per position but then compute the global mean without the same winsorization, which can skew the per-position shrinkage. I keep the exact same core approach (SN_filter==1, inverse-variance weighting with a noise floor + cap, per-position weighted mean, per-position shrinkage to global mean, same tail fill, same unscored fill), but compute the **global means using the same per-position winsorized values** so shrinkage is aligned. This is a minimal statistical hygiene change (no model/loop/feature changes) that typically reduces MCRMSE for this competition. Submission formatting, ordering, and file output remain identical.'
- What this solution (achieved 0.46768) has done: 'I keep your exact baseline approach (SN_filter==1, inverse-variance weighted per-position mean with winsorization, and per-position shrinkage toward a global mean), but make two tightly-scoped changes that typically reduce MCRMSE without changing semantics. First, I compute the global mean in a way that is *consistent with the winsorization and weighting at each position* by using per-position clipped values and weights (rather than a single pooled clip), so the shrinkage target better matches the per-position statistics. Second, I make the shrinkage slightly more position-adaptive by computing Neff from the same clipped/weighted aggregation used for means (small numerical alignment), keeping the same functional form and hyperparameters. Submission formatting, row alignment to `sample_submission.csv`, and tail/unscored fill behavior remain the same, and it still writes `submission.csv` end-to-end.'
- What this solution (achieved 0.46224) has done: 'I keep your exact mean-baseline + inverse-variance weighting + per-position winsorization + Neff-based shrinkage logic, but make two minimal statistical adjustments aimed at lowering MCRMSE toward your 0.3601 target. First, I compute winsorization quantiles using **weighted quantiles** (instead of unweighted `groupby().quantile`), so clipping reflects measurement confidence and better matches your weighting scheme. Second, I slightly soften the dominance of extremely confident points by using a **gentle weight tempering** (`w ** 0.85`) after the existing floor/cap, which typically improves robustness without changing the overall approach. Submission formatting, ordering, alignment to `sample_submission.csv`, and all fill behaviors remain unchanged, and it still writes a valid `submission.csv`.'
- What this solution (achieved 0.46468) has done: 'I keep your existing “SN_filter==1 + inverse-variance weighted per-position mean + weighted winsorization + Neff-based shrinkage + tail fill” baseline intact, but make two minimal adjustments aimed at lowering MCRMSE from 0.46224 toward the 0.3601 target. First, I slightly relax the weight tempering (increase `w_power` from 0.85 to 0.93) so genuinely high-quality (low-error) measurements contribute a bit more, which usually improves this competition’s baseline means. Second, I make shrinkage slightly less aggressive on well-supported positions by reducing `base_alpha` a bit while keeping the same Neff-based functional form; this typically helps when the per-position signal is strong. Everything else (data paths, validity filtering, winsorization, tail fill, submission alignment/format) remains unchanged and it still writes a valid `submission.csv`.'
- What this solution (achieved 0.46467) has done: 'Your current baseline is stable but still far from the 0.3601 target (lower is better), so the smallest likely improvement is to reduce bias from outlier-heavy training tails and better match the test-set filtering. I keep the exact same core approach (SN_filter==1, inverse-variance weighted per-position means, weighted winsorization, Neff-based shrinkage, same submission alignment and fills), but additionally (1) drop low-signal sequences using the provided `signal_to_noise` to mimic the test set quality and (2) compute a more robust tail fill using a **weighted mean of the last K scored positions** (using Neff as weights) instead of a plain average. These changes only affect the statistics used by the same mean/shrinkage predictor, and should nudge MCRMSE down toward the target without changing the overall modeling semantics. The script still runs end-to-end and writes a valid `submission.csv` with the required columns/order.'

# 9. Code solution

## === cell 0
import os
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)



## === cell 1
candidate_paths = [
    "/kaggle/input/sample_submission.csv",
    "/kaggle/input/stanford-covid-vaccine/sample_submission.csv",
    "/kaggle/data/sample_submission.csv",
    "/kaggle/data/stanford-covid-vaccine/sample_submission.csv",
    "../input/sample_submission.csv",
    "../input/stanford-covid-vaccine/sample_submission.csv",
]

sample_path = None
for p in candidate_paths:
    if os.path.exists(p):
        sample_path = p
        break

if sample_path is None:
    raise FileNotFoundError(
        "Could not find sample_submission.csv in expected locations. "
        f"Tried: {candidate_paths}"
    )

sub = pd.read_csv(sample_path)

required_cols = [
    "id_seqpos",
    "reactivity",
    "deg_Mg_pH10",
    "deg_pH10",
    "deg_Mg_50C",
    "deg_50C",
]
missing = [c for c in required_cols if c not in sub.columns]
if missing:
    raise ValueError(f"sample_submission.csv missing required columns: {missing}")

sub = sub[required_cols].copy()
sub.head()



## === cell 2
data_candidates = [
    ("/kaggle/input/train.json", "/kaggle/input/test.json"),
    (
        "/kaggle/input/stanford-covid-vaccine/train.json",
        "/kaggle/input/stanford-covid-vaccine/test.json",
    ),
    ("/kaggle/data/train.json", "/kaggle/data/test.json"),
    (
        "/kaggle/data/stanford-covid-vaccine/train.json",
        "/kaggle/data/stanford-covid-vaccine/test.json",
    ),
    ("../input/train.json", "../input/test.json"),
    (
        "../input/stanford-covid-vaccine/train.json",
        "../input/stanford-covid-vaccine/test.json",
    ),
]

train_path = test_path = None
for tr, te in data_candidates:
    if os.path.exists(tr) and os.path.exists(te):
        train_path, test_path = tr, te
        break

if train_path is None:
    raise FileNotFoundError(
        f"Could not find train.json/test.json in expected locations: {data_candidates}"
    )

train_df = pd.read_json(train_path, lines=True)
test_df = pd.read_json(test_path, lines=True)

scored_target_cols = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
all_target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

for c in all_target_cols:
    if c not in train_df.columns:
        raise ValueError(f"train.json missing target column: {c}")

err_map = {
    "reactivity": "reactivity_error",
    "deg_Mg_pH10": "deg_error_Mg_pH10",
    "deg_pH10": "deg_error_pH10",
    "deg_Mg_50C": "deg_error_Mg_50C",
    "deg_50C": "deg_error_50C",
}
for t, e in err_map.items():
    if e not in train_df.columns:
        raise ValueError(f"train.json missing required error column for weighting: {e}")

snr_min = 1.0
if "SN_filter" in train_df.columns:
    m = train_df["SN_filter"] == 1
    if "signal_to_noise" in train_df.columns:
        m = m & (train_df["signal_to_noise"].astype(float) >= snr_min)
    train_df_use = train_df[m].copy()
    if len(train_df_use) == 0:
        train_df_use = train_df[train_df["SN_filter"] == 1].copy()
        if len(train_df_use) == 0:
            train_df_use = train_df
else:
    train_df_use = train_df

valid_min = -0.5

train_rows = []
for _, r in train_df_use.iterrows():
    rid = r["id"]
    Ls = int(r["seq_scored"])
    for pos in range(Ls):
        row = {"id": rid, "seqpos": pos}
        ok_any = False
        for c in scored_target_cols:
            v = float(r[c][pos])
            e = float(r[err_map[c]][pos])
            if np.isfinite(v) and np.isfinite(e) and (v > valid_min) and (e > 0):
                row[c] = v
                row[c + "_err"] = e
                ok_any = True
            else:
                row[c] = np.nan
                row[c + "_err"] = np.nan
        if ok_any:
            train_rows.append(row)

train_long = pd.DataFrame(train_rows)

eps = 1e-6

err_floor = 0.03
w_cap = 1500.0

winsor_lo_q = 0.02
winsor_hi_q = 0.98


def _weighted_quantile(values, weights, q):
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    m = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(m):
        return np.nan
    v = values[m]
    w = weights[m]
    idx = np.argsort(v)
    v = v[idx]
    w = w[idx]
    cw = np.cumsum(w)
    tot = cw[-1]
    if tot <= 0:
        return np.nan
    target = q * tot
    j = int(np.searchsorted(cw, target, side="left"))
    j = min(max(j, 0), len(v) - 1)
    return float(v[j])


w_power = 0.93

pos_means = pd.DataFrame(index=np.sort(train_long["seqpos"].unique()))
pos_neff = pd.DataFrame(
    index=pos_means.index, columns=scored_target_cols, dtype=np.float32
)

vclip_store = {}  # c -> dict with keys: seqpos, v_clip, w

for c in scored_target_cols:
    v_all = train_long[c].astype(np.float32)
    e_all = train_long[c + "_err"].astype(np.float32)
    m = np.isfinite(v_all) & np.isfinite(e_all)
    v = v_all[m]
    e = e_all[m]
    seqpos = train_long.loc[m, "seqpos"].astype(np.int16)

    w = 1.0 / (np.square(e) + (err_floor * err_floor) + eps)
    w = np.minimum(w, w_cap).astype(np.float32)
    w = np.power(w, w_power).astype(np.float32)

    v_df = pd.DataFrame({"v": v.values, "w": w.values, "seqpos": seqpos.values})

    qlo_map = {}
    qhi_map = {}
    for sp, g in v_df.groupby("seqpos", sort=False):
        qlo_map[int(sp)] = _weighted_quantile(g["v"].values, g["w"].values, winsor_lo_q)
        qhi_map[int(sp)] = _weighted_quantile(g["v"].values, g["w"].values, winsor_hi_q)

    qlo = pd.Series(qlo_map, dtype=np.float32)
    qhi = pd.Series(qhi_map, dtype=np.float32)

    v_clip = v_df["v"].to_numpy(dtype=np.float32).copy()
    sp_idx = v_df["seqpos"].to_numpy(dtype=np.int16)

    hi_vals = qhi.loc[sp_idx].to_numpy(dtype=np.float32)
    lo_vals = qlo.loc[sp_idx].to_numpy(dtype=np.float32)
    v_clip = np.minimum(v_clip, hi_vals)
    v_clip = np.maximum(v_clip, lo_vals)
    v_df["v_clip"] = v_clip.astype(np.float32)

    num = (v_df["v_clip"] * v_df["w"]).groupby(v_df["seqpos"]).sum()
    den = v_df["w"].groupby(v_df["seqpos"]).sum()
    pos_means[c] = (num / (den + eps)).astype(np.float32)

    sw = v_df["w"].groupby(v_df["seqpos"]).sum()
    sw2 = (v_df["w"] ** 2).groupby(v_df["seqpos"]).sum()
    neff = (sw * sw) / (sw2 + eps)
    pos_neff[c] = neff.astype(np.float32)

    vclip_store[c] = {
        "seqpos": v_df["seqpos"].to_numpy(dtype=np.int16),
        "v_clip": v_df["v_clip"].to_numpy(dtype=np.float32),
        "w": v_df["w"].to_numpy(dtype=np.float32),
    }

global_means = {}
for c in scored_target_cols:
    store = vclip_store[c]
    if store["v_clip"].size == 0:
        global_means[c] = 0.0
        continue
    df_tmp = pd.DataFrame(
        {"seqpos": store["seqpos"], "v_clip": store["v_clip"], "w": store["w"]}
    )
    num_pos = (df_tmp["v_clip"] * df_tmp["w"]).groupby(df_tmp["seqpos"]).sum()
    den_pos = df_tmp["w"].groupby(df_tmp["seqpos"]).sum()
    pos_mean_tmp = num_pos / (den_pos + eps)

    global_means[c] = float((pos_mean_tmp * den_pos).sum() / (den_pos.sum() + eps))

global_means = pd.Series(global_means, dtype=np.float32)

base_alpha = 0.22
extra_alpha = 0.26
tau_neff = 55.0
alpha_cap = 0.72

alpha_pos = pd.DataFrame(
    index=pos_means.index, columns=scored_target_cols, dtype=np.float32
)
for c in scored_target_cols:
    neff = pos_neff[c].fillna(0.0).astype(np.float32)
    a = base_alpha + extra_alpha * np.exp(-neff / tau_neff)
    a = np.clip(a, 0.0, alpha_cap).astype(np.float32)
    alpha_pos[c] = a

for c in scored_target_cols:
    pos_means[c] = (1.0 - alpha_pos[c]) * pos_means[c] + alpha_pos[c] * global_means[c]

unscored_target_cols = ["deg_pH10", "deg_50C"]
unscored_global_means = {}
for c in unscored_target_cols:
    vals = []
    errs = []
    for _, r in train_df_use.iterrows():
        Ls = int(r["seq_scored"])
        vv = np.asarray(r[c][:Ls], dtype=np.float32)
        ee = np.asarray(r[err_map[c]][:Ls], dtype=np.float32)

        m = np.isfinite(vv) & np.isfinite(ee) & (vv > valid_min) & (ee > 0)
        if m.any():
            vals.append(vv[m])
            errs.append(ee[m])

    if len(vals) == 0:
        unscored_global_means[c] = 0.0
    else:
        vals = np.concatenate(vals)
        errs = np.concatenate(errs)

        w = 1.0 / (np.square(errs) + (err_floor * err_floor) + eps)
        w = np.minimum(w, w_cap).astype(np.float32)
        w = np.power(w, w_power).astype(np.float32)

        unscored_global_means[c] = float((vals * w).sum() / (w.sum() + eps))
unscored_global_means = pd.Series(unscored_global_means, dtype=np.float32)

pos_means.head(), global_means, unscored_global_means



## === cell 3
pred_cols = required_cols[1:]

parts = sub["id_seqpos"].str.rsplit("_", n=1, expand=True)
sub_id = parts[0].values
sub_pos = parts[1].astype(int).values

test_seq_scored = test_df.set_index("id")["seq_scored"].to_dict()

pred = np.zeros((len(sub), len(pred_cols)), dtype=np.float32)

if len(pos_means.index):
    tail_k = 10
    last_positions = list(pos_means.index[-tail_k:])

    neff_tail = (
        pos_neff.loc[last_positions, scored_target_cols]
        .astype(np.float32)
        .mean(axis=1)
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
    )
    neff_tail = np.maximum(neff_tail, 0.0)

    vals_tail = pos_means.loc[last_positions, scored_target_cols].to_numpy(
        dtype=np.float32
    )
    wsum = float(neff_tail.sum())
    if wsum > 0:
        tail_vals_scored = (vals_tail * neff_tail[:, None]).sum(axis=0) / (wsum + eps)
        tail_vals_scored = tail_vals_scored.astype(np.float32)
    else:
        tail_vals_scored = vals_tail.mean(axis=0).astype(np.float32)
else:
    tail_vals_scored = global_means.values.astype(np.float32)

col_to_j = {c: j for j, c in enumerate(pred_cols)}

for i, (rid, pos) in enumerate(zip(sub_id, sub_pos)):
    scored_len = int(test_seq_scored.get(rid, 68))

    if pos < scored_len and pos in pos_means.index:
        for c in scored_target_cols:
            pred[i, col_to_j[c]] = float(pos_means.loc[pos, c])
    elif pos < scored_len:
        for c in scored_target_cols:
            pred[i, col_to_j[c]] = float(global_means[c])
    else:
        for k, c in enumerate(scored_target_cols):
            pred[i, col_to_j[c]] = float(tail_vals_scored[k])

    for c in unscored_target_cols:
        pred[i, col_to_j[c]] = float(unscored_global_means[c])

sub[pred_cols] = pred
sub[pred_cols] = sub[pred_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

assert sub.shape[1] == 6, f"Expected 6 columns, got {sub.shape[1]}"
assert (
    sub.columns.tolist() == required_cols
), "Submission columns are not in the required order"
assert sub["id_seqpos"].is_unique, "id_seqpos must be unique"
sub.head()



## === cell 4
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())
