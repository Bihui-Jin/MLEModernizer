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

0.48369

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.42418) has done: 'I remove the dependency on the missing `../input/worst-submission/ensemble52.csv` file and instead build predictions directly from the provided competition files so the notebook runs end-to-end. To keep the logic minimal and stable, I generate a valid submission by using simple per-position means from the training targets (a common baseline) and filling unscored positions with the last available mean. I also ensure the output matches `sample_submission.csv` exactly in row order and column names, and write `submission.csv` to the working directory. The remaining “sequence leakage/manipulation” cells be made no-ops to avoid runtime errors while preserving the overall structure.'
- What this solution (achieved 0.42606) has done: 'Your current score is worse than the target (lower is better), so we should make the smallest legitimate change that improves MCRMSE. The biggest issue hurting score is cell 4, which manually overwrites predictions for one test id (this cannot help generalization and usually worsen error), so we remove that overwrite while keeping the same per-position-mean baseline core logic. Additionally, we align the baseline with the competition’s standard practice by computing means only on high-quality training rows (SN_filter==1) and clipping predictions to a reasonable range; both are minimal post-processing steps that typically reduce RMSE without changing the modeling approach. The script still runs end-to-end, keeps the same submission schema/ordering, and writes a valid `submission.csv`.'
- What this solution (achieved 0.48491) has done: 'We keep your per-position-mean baseline intact, but make one small change that typically improves MCRMSE: compute the mean targets as a *weighted mean* using the provided per-position measurement errors (higher weight for more reliable measurements). This stays within the same “training-set average per position” logic (no model/loop/feature changes), but uses information already in `train.json` to reduce noise. We apply the same SN_filter==1 selection, keep the same padding to seq_length, keep the same clipping, and preserve the exact sample_submission row order/columns while writing `submission.csv`.'
- What this solution (achieved 0.48345) has done: 'Your current score (0.48491) is worse than the target (0.35188) for a lower-is-better metric, so we should make a small, legitimate improvement without changing the “per-position mean baseline” core logic. The main change is to compute the per-position mean only over the *scored region* (0..seq_scored-1) and fill unscored positions (68..106) with a more robust value (the mean over the last few scored positions) instead of repeating just the last position, which typically reduces error spillover patterns. We also add a minimal safeguard to keep weights from extreme errors from dominating by clipping the inverse-variance weights, which often stabilizes the weighted mean and improves MCRMSE slightly. Output format, ordering, and columns remain identical and `submission.csv` is still written.'
- What this solution (achieved 0.48344) has done: 'Your current score (0.48345, lower-is-better) is still far from the target (0.35188), so we should make a small but meaningful improvement without changing the “per-position averaged baseline” core logic. The most impactful minimal fix is to compute position-wise means separately for each target and also use the per-sample `signal_to_noise` as an additional (capped) weight multiplier on top of inverse-variance error weights; this typically denoises the baseline while staying the same approach (still a weighted per-position mean, just with a better reliability weight). I’m also adding a tiny robustness safeguard: replace any non-finite values in targets/errors before weighting to avoid accidental NaNs propagating into the means. Output format, ordering, clipping, and the `submission.csv` write remain unchanged.'
- What this solution (achieved 0.48344) has done: 'Your current score (0.48344, lower-is-better) is far worse than the target (0.35188), so we should make a small, legitimate improvement while keeping the same “per-position weighted mean baseline” core logic. The biggest gain with minimal semantic change is to (1) drop a few extreme/invalid training rows using the competition’s standard quality filters (signal_to_noise > 1 and min target > -0.5), and (2) use the known training distribution to fill unscored positions (68..106) more realistically (position-wise global means over all 107 positions computed from training) instead of repeating a tail value. We keep your inverse-variance + capped SNR weighting for the scored region exactly as the main predictor, preserve submission order/columns, and still write a valid `submission.csv`. These changes typically reduce MCRMSE for this competition without changing the modeling approach.'
- What this solution (achieved 0.48339) has done: 'Your current score (0.48344, lower-is-better) is still far from the target (0.35188), so we need a small but meaningful improvement while keeping the same “per-position weighted mean baseline” core logic. The biggest legitimate gain here is to stop treating every position equally in the average and instead weight each training example by the competition’s provided per-position errors *and* the sample’s SNR in a more score-aligned way: use inverse-variance weighting with an SNR exponent (sub-linear) so high-SNR rows help but don’t dominate. Additionally, for the unscored tail positions (68..106), we keep your simple constant padding idea but make it target-specific and derived from a smoother estimate (average of last K scored means) to avoid sharp discontinuities that can slightly hurt generalization. These are minimal changes (still a weighted per-position mean baseline, same I/O and submission order) and should improve MCRMSE without changing evaluation semantics.'
- What this solution (achieved 0.48339) has done: 'We keep your per-position weighted-mean baseline exactly the same, but make the weighting more score-aligned by downweighting noisier targets using the *reported per-position errors* as a proxy for label reliability (i.e., winsorize extreme target values based on their error before computing the weighted mean). This is still the same “weighted average per position” logic (no model, no new features), but it reduces the influence of extreme/outlier measurements that disproportionately hurt RMSE. We also apply the same error-aware winsorization to the per-position aggregation only (not to submission post-processing), keeping your submission order/format unchanged and still writing `submission.csv`. These changes are small, fast, and commonly reduce MCRMSE for this competition.'
- What this solution (achieved 0.48339) has done: 'Your current score (0.48339, lower-is-better) is far worse than the target (0.35188), so we should make a small, legitimate improvement while keeping the same “weighted per-position mean baseline” core logic. The most impactful minimal fix here is that you’re averaging over *all five* targets (including the two unscored ones) in your quality filter `minvals > -0.5`, which can unnecessarily discard useful training rows and degrade the scored targets; we apply that filter only to the *scored* targets. In addition, we compute the per-position aggregation only for the three scored targets and then copy those predictions into the unscored columns (deg_pH10, deg_50C) to reduce noise without changing the scoring semantics. Everything else (inverse-variance + SNR weighting, error-aware winsorization, tail padding, submission order/format, and writing `submission.csv`) stays the same.'
- What this solution (achieved 0.48341) has done: 'We keep your exact “weighted per-position mean baseline” logic and submission alignment, but make the weights better match the metric by computing the per-position aggregation on a training fold that excludes the noisiest label measurements more directly. Concretely, we add a minimal per-row/per-target filter based on the provided per-position errors (drop a small fraction of rows with the largest *mean error* for each scored target), then recompute the same inverse-variance + SNR-weighted means on the remaining rows. This is still the same approach (a weighted average baseline), just with a more reliable set of measurements feeding the mean, which typically lowers RMSE. Everything else (tail padding, copying scored→unscored columns, clipping, row order, and writing `submission.csv`) remains unchanged.'
- What this solution (achieved 0.48415) has done: 'Your current score (0.48341, lower-is-better) is still far from the target (0.35188), so we need a small but meaningful improvement while keeping the same “weighted per-position mean baseline” approach. The most likely issue is that the recent “drop the noisiest labels by mean error quantile” step can accidentally remove a lot of useful signal and destabilize the weighted mean; we keep the same weighting logic but *replace hard-dropping with soft downweighting* via an extra per-row reliability multiplier derived from mean error. We also stop filling the unscored tail (positions 68–106) with an arbitrary constant and instead use a simple linear extrapolation from the last scored region, which is still the same baseline logic but typically reduces error because the tail is correlated with nearby positions. Submission ordering/columns remain exactly aligned to `sample_submission.csv`, and the script still writes `submission.csv` end-to-end.'
- What this solution (achieved 0.48415) has done: 'We keep your exact “weighted per-position mean baseline” core logic, but remove the only remaining avoidable source of score inflation/instability: the linear extrapolation used to fill unscored tail positions (68–106), which can create out-of-distribution values and slightly worsen RMSE on the scored region through clipping/interactions. Instead, we fill the tail with a very stable constant equal to the mean of the last K scored-position means (target-specific), which is a minimal post-processing change and commonly performs better for this competition while preserving evaluation semantics. Everything else (filters, inverse-variance + SNR weighting, winsorization, copying scored→unscored columns, clipping, and submission alignment to `sample_submission.csv`) remains unchanged. The script still run end-to-end and write a valid `submission.csv`.'
- What this solution (achieved 0.48415) has done: 'Your current score (0.48415, lower-is-better) is far from the target (0.35188), so we should make a small but meaningful improvement while keeping the same “weighted per-position mean baseline” core logic. The biggest safe gain here is to stop using the two *unscored* columns as noisy copies and instead predict them with their own (same-method) weighted per-position means from train, while leaving the scored columns unchanged in approach. This does not change the model family (still the same weighted mean with the same filters/weights/winsorization), but it typically reduces MCRMSE slightly because the submission is more coherent and avoids injecting extra noise patterns. We also keep the same submission row order, clipping, and `submission.csv` output.'
- What this solution (achieved 0.4837) has done: 'You’re still far from the target (0.48415 vs 0.35188; lower is better), so we should make a small, legitimate improvement without changing the “weighted per-position mean baseline” core approach. The biggest issue in your current code is that you compute `seq_scored`/`seq_length` from train and assume they’re constant; instead we should respect each test sample’s `seq_scored` and only use the scored-region means for the scored rows, which avoids mismatches if any ids differ and aligns predictions to what’s actually evaluated. Next, your row-level reliability multiplier `row_rel` multiplies three target reliabilities together, which can overly crush weights; we keep the same concept but replace the product with a capped geometric mean to preserve denoising while preventing over-downweighting. Finally, we ensure the submission rows beyond `seq_scored` are filled stably per-id (tail constant from last K scored means), but without affecting scored rows, preserving evaluation semantics and keeping changes minimal.'
- What this solution (achieved 0.48369) has done: 'Your current score (0.4837) is worse than the target (0.3519) for a lower-is-better metric, so we should make the smallest likely-helpful improvement while keeping the same “weighted per-position mean baseline” core logic. The main issue is that you aggregate all training samples into one global per-position mean, even though test/public LB is filtered for higher-quality (SN_filter-like) sequences; a minimal, competition-standard improvement is to compute the same weighted mean **within a few coarse bins of `signal_to_noise`** and then select the appropriate bin for each test id based on its predicted loop/sequence-independent metadata isn’t available, so we approximate by using a fixed “high quality” bin (since test is filtered). To keep changes minimal and stable, we implement this as a simple mixture: combine your current “all-used” mean with an additional “very-high-SNR” mean (same weighting math), which usually moves predictions closer to the test distribution without changing architecture/loops/loss. Everything else (filters, inverse-variance weights, winsorization, tail fill, submission alignment, and writing `submission.csv`) is preserved.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

DATA_DIR = "../input/stanford-covid-vaccine"

train_path = os.path.join(DATA_DIR, "train.json")
test_path = os.path.join(DATA_DIR, "test.json")
sample_path = os.path.join(DATA_DIR, "sample_submission.csv")

df_train = pd.read_json(train_path, lines=True)
df_test = pd.read_json(test_path, lines=True)
sample_sub = pd.read_csv(sample_path)

TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
SCORED_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]

seq_scored_train = int(df_train["seq_scored"].iloc[0])  # typically 68
seq_length_train = int(df_train["seq_length"].iloc[0])  # typically 107

use_mask = df_train["SN_filter"] == 1
if "signal_to_noise" in df_train.columns:
    use_mask &= df_train["signal_to_noise"].fillna(0.0) > 1.0


def row_min_over_cols(row, cols) -> float:
    m = np.inf
    for c in cols:
        v = row[c]
        if isinstance(v, (list, tuple, np.ndarray)) and len(v) > 0:
            mv = float(np.nanmin(np.asarray(v, dtype=np.float64)))
            if mv < m:
                m = mv
    return m


minvals = df_train.apply(lambda r: row_min_over_cols(r, SCORED_COLS), axis=1)
use_mask &= minvals > -0.5

df_train_use = df_train.loc[use_mask].copy()
if len(df_train_use) == 0:
    df_train_use = df_train.copy()

ERR_COL_MAP = {
    "reactivity": "reactivity_error",
    "deg_Mg_pH10": "deg_error_Mg_pH10",
    "deg_pH10": "deg_error_pH10",
    "deg_Mg_50C": "deg_error_Mg_50C",
    "deg_50C": "deg_error_50C",
}

EPS = 1e-6

W_MAX = 1e4  # cap inverse-variance weights to avoid extreme domination
SNR_MAX = 10.0  # cap SNR influence; keeps weighting stable

SNR_EXP = 0.5
Y_CLIP_K = 3.0  # conservative error-aware winsorization strength
ERR_SOFT_POW = 1.0  # gentle; keeps behavior close to current while improving stability
ERR_ROW_W_FLOOR = 0.2  # avoid zeroing out rows entirely


def compute_weighted_means(df_in: pd.DataFrame):
    means = {}

    snr = df_in.get(
        "signal_to_noise", pd.Series(np.ones(len(df_in)), index=df_in.index)
    ).to_numpy(dtype=np.float64)
    snr = np.nan_to_num(snr, nan=1.0, posinf=1.0, neginf=1.0)
    snr = np.clip(snr, 0.0, SNR_MAX)  # (n_train,)
    snr_w = np.power(np.maximum(snr, 0.0), SNR_EXP)

    rels = []
    for col in SCORED_COLS:
        err_col = ERR_COL_MAP.get(col)
        if err_col in df_in.columns:
            e = np.vstack(df_in[err_col].values).astype(np.float64)
            e = np.nan_to_num(e, nan=np.nan, posinf=np.nan, neginf=np.nan)
            mean_e = np.nanmean(e, axis=1)
            mean_e = np.nan_to_num(mean_e, nan=np.inf, posinf=np.inf, neginf=np.inf)
            med = (
                float(np.nanmedian(mean_e[np.isfinite(mean_e)]))
                if np.any(np.isfinite(mean_e))
                else 1.0
            )
            med = max(med, 1e-3)
            rel = (med / np.clip(mean_e, 1e-3, None)) ** ERR_SOFT_POW
            rel = np.clip(rel, ERR_ROW_W_FLOOR, 1.0)
            rels.append(rel)

    if len(rels) > 0:
        rels = np.vstack(rels)  # (n_targets, n_train)
        row_rel = np.exp(
            np.mean(np.log(np.clip(rels, 1e-12, None)), axis=0)
        )  # geometric mean
        row_rel = np.clip(row_rel, ERR_ROW_W_FLOOR, 1.0)
    else:
        row_rel = np.ones(len(df_in), dtype=np.float64)

    row_rel = np.nan_to_num(row_rel, nan=1.0, posinf=1.0, neginf=1.0)
    row_w = snr_w * row_rel  # (n_train,)

    for col in TARGET_COLS:
        y = np.vstack(df_in[col].values).astype(
            np.float64
        )  # (n_train, seq_scored_train)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

        err_col = ERR_COL_MAP.get(col, None)
        if err_col is not None and err_col in df_in.columns:
            e = np.vstack(df_in[err_col].values).astype(np.float64)
            e = np.nan_to_num(e, nan=1.0, posinf=1.0, neginf=1.0)

            e_safe = np.clip(e, 1e-3, None)
            y = np.clip(y, y - Y_CLIP_K * e_safe, y + Y_CLIP_K * e_safe)

            w = 1.0 / (np.square(e_safe) + EPS)
            w = np.clip(w, 0.0, W_MAX)

            w = w * row_w[:, None]
            means[col] = (w * y).sum(axis=0) / np.clip(w.sum(axis=0), EPS, None)
        else:
            w = row_w[:, None]
            means[col] = (w * y).sum(axis=0) / np.clip(w.sum(axis=0), EPS, None)

        means[col] = means[col].astype(np.float64)

    return means


means_all = compute_weighted_means(df_train_use)

snr_full = df_train_use.get(
    "signal_to_noise", pd.Series(np.ones(len(df_train_use)), index=df_train_use.index)
).to_numpy(dtype=np.float64)
snr_full = np.nan_to_num(snr_full, nan=0.0, posinf=0.0, neginf=0.0)

df_train_hi = df_train_use.loc[snr_full >= 2.0].copy()
if len(df_train_hi) < 200:
    df_train_hi = df_train_use.copy()

means_hi = compute_weighted_means(df_train_hi)

ALPHA_HI = 0.35
means_scored = {}
for col in TARGET_COLS:
    means_scored[col] = (1.0 - ALPHA_HI) * means_all[col] + ALPHA_HI * means_hi[col]

K_tail = min(12, seq_scored_train)
tail_fill = {col: float(np.mean(means_scored[col][-K_tail:])) for col in TARGET_COLS}


def parse_seqpos(id_seqpos: str) -> int:
    return int(id_seqpos.rsplit("_", 1)[1])


def parse_id(id_seqpos: str) -> str:
    return id_seqpos.rsplit("_", 1)[0]


seqpos = sample_sub["id_seqpos"].map(parse_seqpos).to_numpy(dtype=np.int64)
ids = sample_sub["id_seqpos"].map(parse_id).to_numpy()

test_meta = df_test.set_index("id")[["seq_scored", "seq_length"]]
id_to_scored = test_meta["seq_scored"].to_dict()
id_to_length = test_meta["seq_length"].to_dict()

df = sample_sub.copy()

scored_cut = np.array(
    [int(id_to_scored.get(i, seq_scored_train)) for i in ids], dtype=np.int64
)

for col in TARGET_COLS:
    base = means_scored[col]  # length seq_scored_train
    vals = np.empty(len(df), dtype=np.float64)

    in_scored = seqpos < scored_cut
    sp = np.clip(seqpos[in_scored], 0, len(base) - 1)
    vals[in_scored] = base[sp]

    vals[~in_scored] = tail_fill[col]
    df[col] = vals

for col in TARGET_COLS:
    df[col] = np.clip(df[col].to_numpy(dtype=np.float64), -0.5, 1.5)



## === cell 1
sequences_public = set(df_train.id)



## === cell 2
sequences = list(set(["_".join(v.split("_")[:-1]) for v in df.id_seqpos.values]))
sequences = [s for s in sequences if s not in sequences_public]
sequences[-10:] if len(sequences) >= 10 else sequences



## === cell 3
len(sequences), len(sequences_public)



## === cell 4
pass



## === cell 5
df = df[["id_seqpos"] + TARGET_COLS]

for c in TARGET_COLS:
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)



## === cell 6
assert (
    df.shape == sample_sub.shape
), f"Submission shape mismatch: {df.shape} vs {sample_sub.shape}"
assert list(df.columns) == list(
    sample_sub.columns
), f"Submission columns mismatch: {df.columns} vs {sample_sub.columns}"



## === cell 7
df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", df.shape)
print(df.head())
