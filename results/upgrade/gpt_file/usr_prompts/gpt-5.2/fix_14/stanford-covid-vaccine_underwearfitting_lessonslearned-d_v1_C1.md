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

0.3647779140395166

# 6. Current score

0.4327

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.63824) has done: 'The current notebook fails because it tries to read multiple external submissions from `../input/...` that do not exist in your environment, so none of the `sub*` DataFrames are created and downstream cells error out. To keep the ensemble-style “blend submissions then write CSV” core logic intact while making it runnable, I replace missing inputs with a safe fallback baseline submission built directly from `sample_submission.csv` (constant zeros), and I load any of the referenced files only if they actually exist. I also make the blending robust to missing `sub*` inputs by averaging only over successfully loaded submissions, preserving the original weights where possible. Finally, it writes a valid `.csv` submission with the exact required columns.'
- What this solution (achieved 0.63824) has done: 'Your current score is poor because almost all ensemble inputs are missing in this environment, so the blend collapses to an all-zeros baseline. To move toward the target (lower is better), the smallest legitimate improvement is to replace that baseline with a simple per-position mean predictor learned from `train.json` (computed only from the training targets), while keeping the ensemble/blending logic intact. This preserves the notebook’s “blend submissions then write CSV” core behavior but ensures a much stronger fallback when external submission files aren’t available. The submission is still aligned to `sample_submission.csv` and written with the exact required columns.'
- What this solution (achieved 0.42418) has done: 'I fix the IndexError in the baseline-per-position mean fallback by safely indexing only the scored positions (0–67) and filling the remaining positions (68–106) with an overall mean, instead of attempting `mu[seqpos]` for out-of-range indices. This preserves your core “fallback baseline + optional ensemble blend” logic and only changes the baseline value construction so the notebook runs end-to-end. I also make the `id_seqpos` parsing slightly more robust and ensure the baseline remains aligned to the sample submission order. These changes are score-improving relative to the current broken run while keeping everything else intact.'
- What this solution (achieved 0.42178) has done: 'Your current score (0.42418, lower-is-better) is worse than the target (0.36478), and in this environment none of the external `sub*` files exist, so the final prediction collapses to the train-derived baseline. To move the score toward the target with minimal, safe changes (keeping the same ensemble/blending structure), I improve only the baseline by (1) filtering training rows to `SN_filter==1` (matching test-set quality) and (2) using signal/noise–weighted per-position means so noisy sequences contribute less. This keeps the same per-position mean fallback logic, only changing how the mean is computed from training targets, and still fills unscored positions (68–106) with an overall mean. The rest of the pipeline (alignment, blending, submission writing) stays identical and still produce a valid CSV.'
- What this solution (achieved 0.48258) has done: 'Your current score (0.42178, lower-is-better) is still above the target (0.36478), and since none of the external `sub*` files exist here the leaderboard score is driven almost entirely by the train-derived baseline. With minimal changes and identical “per-position mean baseline + optional blending” core logic, I improve only how the baseline mean is computed by switching from global `signal_to_noise` weights to per-target, per-position inverse-variance weights derived from the provided `*_error_*` arrays (a closer match to how noisy labels should be averaged). I also keep your `SN_filter==1` filtering and the same out-of-range handling for positions 68–106, but compute the fill value as an error-weighted overall mean for consistency. Everything else (alignment, blending, and submission writing) stays the same and still produces a valid `final_v1.csv`.'
- What this solution (achieved 0.48258) has done: 'Your current score (0.48258, lower-is-better) is worse than the target (0.36478), so we need a small, legitimate accuracy gain without changing the overall “train-derived baseline + optional blending” logic. Since none of the external `sub*` files exist here, the score is dominated by the baseline; the safest minimal improvement is to compute the baseline means only from reliable labels by filtering out low signal/noise training examples (in addition to `SN_filter==1`). To keep semantics identical, we keep the same inverse-variance (error) weighted per-position mean, but apply a conservative `signal_to_noise >= 1.0` filter (matching the test filtering description) and fall back gracefully if that filter would remove too much data. Everything else (alignment, blending, CSV output schema) remains unchanged.'
- What this solution (achieved 0.48357) has done: 'Your current score (0.48258, lower-is-better) is still worse than the target (0.36478), and since none of the external `sub*` files exist here the public score is almost entirely determined by the train-derived baseline. To move the score downward with minimal change while keeping the same “baseline + optional blending” logic, I only improve the baseline estimator by shrinking the noisy per-position means toward an overall mean using the effective sample size implied by your inverse-variance weights (this reduces variance/overfit in positions with weak support). I keep the same `SN_filter==1` and `signal_to_noise>=1.0` filtering, the same inverse-variance weighting from `*_error_*`, and the same filling rule for positions 68–106 (still using the overall mean). Everything else (alignment, blending, and writing `final_v1.csv` with exact required columns and row order) remains unchanged.'
- What this solution (achieved 0.48357) has done: 'We keep your ensemble/blending structure intact (since external `sub*` files are still missing here, your score is essentially the baseline). The smallest score-improving change is to compute the baseline per-position means only on the *scored* targets (reactivity, deg_Mg_pH10, deg_Mg_50C) and then copy those predictions into the two unscored columns (deg_pH10, deg_50C), because those two are not evaluated and currently add avoidable noise. This preserves evaluation semantics (same file format and valid values everywhere) while reducing the effective error contribution coming from weak/noisy estimates of unscored columns. Everything else (filters, inverse-variance weighting, shrinkage, alignment, blending, CSV writing) remains unchanged.'
- What this solution (achieved 0.48252) has done: 'Your current score is worse than the target (lower is better), and because the external `sub*` files don’t exist here the final predictions are effectively just the train-derived baseline. With minimal changes and identical “train-derived per-position baseline + optional blending” core logic, I (1) compute the baseline per-position means using **only the same 3 scored targets** (as you already do) but **fit a tiny linear calibration per target** on the training data to reduce systematic bias, and (2) keep your existing shrinkage/filters and still copy scored columns into the two unscored columns (since they don’t affect the metric). This keeps the same prediction structure (position-wise mean baseline) while improving alignment to the evaluation metric without changing the ensemble/blend logic or adding new models. The submission writing, ordering, and schema remain unchanged.'
- What this solution (achieved 0.63824) has done: 'We keep your ensemble/blending structure exactly the same and only adjust the train-derived baseline, since in this environment the external `sub*` files are missing and the baseline drives the score. The minimal improvement is to compute a per-position baseline conditioned on the available **test-side features** (`sequence`, `structure`, `predicted_loop_type`) by grouping training rows with the same per-position token triple and taking the same inverse-variance weighted mean you already use, with the same filters/shrinkage, then falling back to your global per-position mean when a group is unseen. This preserves your “per-position mean baseline” core logic (still means, still error-weighted, still shrinkage) but makes it less biased for specific base/structure contexts, which should move MCRMSE down toward the target. All alignment, blending weights, and submission writing remain unchanged and it still writes `final_v1.csv` with the required columns and row order.'
- What this solution (achieved 0.44484) has done: 'I fix the runtime error coming from trying to “add” NumPy unicode arrays when building token keys (sequence/structure/loop_type), by switching to safe string concatenation that works on object arrays. This is a correctness-only change: it preserves the exact baseline logic (same conditioning, weighting, shrinkage, calibration, and blending), but makes the code run end-to-end in your Kaggle environment. I also keep the rest of the pipeline unchanged so it still aligns to `sample_submission.csv` order and writes a valid `final_v1.csv` submission.'
- What this solution (achieved 0.43728) has done: 'Your current score (0.44484; lower is better) is still worse than the target (0.36478), and in this environment the external `sub*` inputs are missing, so the leaderboard score is driven almost entirely by the train-derived baseline. With minimal change and the same “token-conditioned error-weighted mean + shrinkage + calibration + blending” core logic, I adjust only the shrinkage strength (`prior_strength`) downward so the model relies more on the token-conditioned estimates (which should reduce bias) while keeping all weighting, filters, and calibration intact. I also add a small guard to prevent over-trusting extremely low-support conditional groups by clamping `cond_neff_at_row` to a minimum of 1.0 (stability-only; same semantics). Everything else (I/O paths, alignment to `sample_submission.csv`, and writing `final_v1.csv`) remains unchanged.'
- What this solution (achieved 0.4327) has done: 'Your current score (0.43728, lower-is-better) is still worse than the target (0.36478), and because the external `sub*` submissions are missing in this environment the leaderboard score is almost entirely determined by the train-derived baseline. To move toward the target with minimal risk while preserving the exact “token-conditioned error-weighted mean + shrinkage + calibration + blending” core logic, I only adjust the shrinkage `prior_strength` slightly downward again so the model relies a bit more on the token-conditioned estimates (reducing bias) while keeping the same guards and calibration. I also add a tiny stability clamp to the final predictions (range clipping) to avoid rare extreme values from sparse conditional groups hurting RMSE; this is a post-processing safety step that preserves semantics and typically improves MCRMSE. Everything else (paths, feature extraction, weighting, blending, and writing `final_v1.csv`) stays unchanged.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os



## === cell 1
SUB_COLS = [
    "id_seqpos",
    "reactivity",
    "deg_Mg_pH10",
    "deg_pH10",
    "deg_Mg_50C",
    "deg_50C",
]


def _find_first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


sample_path = _find_first_existing(
    [
        "/kaggle/input/sample_submission.csv",
        "/kaggle/input/stanford-covid-vaccine/sample_submission.csv",
        "../input/sample_submission.csv",
        "../input/stanford-covid-vaccine/sample_submission.csv",
        "/kaggle/data/sample_submission.csv",
        "/kaggle/data/stanford-covid-vaccine/sample_submission.csv",
    ]
)

if sample_path is None:
    raise FileNotFoundError(
        "Could not locate sample_submission.csv in expected Kaggle paths."
    )

sample_sub = pd.read_csv(sample_path)
missing_cols = [c for c in SUB_COLS if c not in sample_sub.columns]
if missing_cols:
    raise ValueError(f"sample_submission.csv missing required columns: {missing_cols}")



## === cell 2
train_path = _find_first_existing(
    [
        "/kaggle/input/train.json",
        "/kaggle/input/stanford-covid-vaccine/train.json",
        "../input/train.json",
        "../input/stanford-covid-vaccine/train.json",
        "/kaggle/data/train.json",
        "/kaggle/data/stanford-covid-vaccine/train.json",
    ]
)

test_path = _find_first_existing(
    [
        "/kaggle/input/test.json",
        "/kaggle/input/stanford-covid-vaccine/test.json",
        "../input/test.json",
        "../input/stanford-covid-vaccine/test.json",
        "/kaggle/data/test.json",
        "/kaggle/data/stanford-covid-vaccine/test.json",
    ]
)

baseline = sample_sub[SUB_COLS].copy()

if train_path is None:
    for c in SUB_COLS[1:]:
        baseline[c] = 0.0
else:
    train_df = pd.read_json(train_path, lines=True)

    if "SN_filter" in train_df.columns:
        train_df = train_df.loc[train_df["SN_filter"].astype(int) == 1].reset_index(
            drop=True
        )

    if "signal_to_noise" in train_df.columns:
        sn_mask = train_df["signal_to_noise"].astype("float64") >= 1.0
        if sn_mask.sum() >= 200:  # safety to avoid collapsing to a tiny subset
            train_df = train_df.loc[sn_mask].reset_index(drop=True)

    scored_targets = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]

    err_map = {
        "reactivity": "reactivity_error",
        "deg_Mg_pH10": "deg_error_Mg_pH10",
        "deg_pH10": "deg_error_pH10",
        "deg_Mg_50C": "deg_error_Mg_50C",
        "deg_50C": "deg_error_50C",
    }

    if test_path is None:
        test_df = None
    else:
        test_df = pd.read_json(test_path, lines=True)

    eps = 1e-6

    seqpos = (
        baseline["id_seqpos"]
        .astype(str)
        .str.rsplit("_", n=1, expand=True)[1]
        .astype(int)
        .to_numpy()
    )

    def _tokens_matrix(df, col):
        arr = df[col].astype(str).values
        return np.array([list(s) for s in arr], dtype=object)

    have_tokens = (
        (test_df is not None)
        and all(
            c in train_df.columns
            for c in ["sequence", "structure", "predicted_loop_type"]
        )
        and all(
            c in test_df.columns
            for c in ["sequence", "structure", "predicted_loop_type"]
        )
    )

    if have_tokens:
        tr_seq = _tokens_matrix(train_df, "sequence")[:, :68]
        tr_str = _tokens_matrix(train_df, "structure")[:, :68]
        tr_loop = _tokens_matrix(train_df, "predicted_loop_type")[:, :68]

        te_seq = _tokens_matrix(test_df, "sequence")  # (n_test, 107)
        te_str = _tokens_matrix(test_df, "structure")
        te_loop = _tokens_matrix(test_df, "predicted_loop_type")
    else:
        tr_seq = tr_str = tr_loop = None
        te_seq = te_str = te_loop = None

    per_pos_means = {}
    per_pos_wsum = {}

    for t in scored_targets:
        y = np.vstack(train_df[t].values).astype("float64")  # (n_train, 68)

        err_col = err_map.get(t, None)
        if err_col is not None and err_col in train_df.columns:
            e = np.vstack(train_df[err_col].values).astype("float64")  # (n_train, 68)
            w = 1.0 / np.clip(e, eps, None) ** 2
            w = np.where(np.isfinite(y), w, 0.0)
        else:
            w = np.ones_like(y, dtype="float64")
            w = np.where(np.isfinite(y), w, 0.0)

        wsum = np.sum(w, axis=0) + 1e-12
        mu = np.nansum(np.where(np.isfinite(y), y, 0.0) * w, axis=0) / wsum

        per_pos_means[t] = mu
        per_pos_wsum[t] = wsum

    cond_stats = {t: [dict() for _ in range(68)] for t in scored_targets}

    if have_tokens:
        token_keys_train = np.char.add(
            np.char.add(np.char.add(tr_seq.astype(str), "|"), tr_str.astype(str)),
            np.char.add("|", tr_loop.astype(str)),
        )  # (n_train, 68)

        for t in scored_targets:
            y = np.vstack(train_df[t].values).astype("float64")  # (n_train, 68)
            err_col = err_map.get(t, None)
            if err_col is not None and err_col in train_df.columns:
                e = np.vstack(train_df[err_col].values).astype("float64")
                w = 1.0 / np.clip(e, eps, None) ** 2
                w = np.where(np.isfinite(y), w, 0.0)
            else:
                w = np.ones_like(y, dtype="float64")
                w = np.where(np.isfinite(y), w, 0.0)

            for p in range(68):
                keys_p = token_keys_train[:, p]
                y_p = y[:, p]
                w_p = w[:, p]
                m = np.isfinite(y_p) & (w_p > 0)
                if not np.any(m):
                    continue
                keys_p = keys_p[m]
                y_p = y_p[m]
                w_p = w_p[m]

                order = np.argsort(keys_p)
                keys_s = keys_p[order]
                y_s = y_p[order]
                w_s = w_p[order]

                change = np.empty(keys_s.shape[0], dtype=bool)
                change[0] = True
                change[1:] = keys_s[1:] != keys_s[:-1]
                idx = np.nonzero(change)[0]
                idx_next = np.r_[idx[1:], keys_s.shape[0]]

                d = cond_stats[t][p]
                for a, b in zip(idx, idx_next):
                    k = keys_s[a]
                    wsum_k = float(np.sum(w_s[a:b]))
                    if wsum_k <= 0:
                        continue
                    wy = float(np.sum(w_s[a:b] * y_s[a:b]))
                    wsum2 = float(np.sum(w_s[a:b] ** 2))
                    d[k] = (wy, wsum_k, wsum2)

    baseline_pred_scored = {}

    prior_strength = 6.0

    if have_tokens:
        test_df = test_df.reset_index(drop=True)
        id_to_row = pd.Series(
            test_df.index.values, index=test_df["id"].astype(str)
        ).to_dict()

        ids = (
            baseline["id_seqpos"]
            .astype(str)
            .str.rsplit("_", n=1, expand=True)[0]
            .values
        )
        row_idx = np.array([id_to_row.get(i, -1) for i in ids], dtype=int)

        token_key_rows = np.empty(len(baseline), dtype=object)
        valid_row = row_idx >= 0
        valid_pos = (seqpos >= 0) & (seqpos < 107)
        valid = valid_row & valid_pos
        token_key_rows[:] = None
        if np.any(valid):
            token_key_rows[valid] = np.char.add(
                np.char.add(
                    np.char.add(te_seq[row_idx[valid], seqpos[valid]].astype(str), "|"),
                    te_str[row_idx[valid], seqpos[valid]].astype(str),
                ),
                np.char.add("|", te_loop[row_idx[valid], seqpos[valid]].astype(str)),
            )

    for t in scored_targets:
        mu = per_pos_means[t]
        wsum_pos = per_pos_wsum[t]
        overall = float(np.sum(mu * wsum_pos) / (np.sum(wsum_pos) + 1e-12))

        base_mu_at_row = np.full(seqpos.shape[0], overall, dtype="float64")
        mask_scored = (seqpos >= 0) & (seqpos < 68)
        base_mu_at_row[mask_scored] = mu[seqpos[mask_scored]]

        if have_tokens:
            cond_mu_at_row = base_mu_at_row.copy()
            cond_neff_at_row = np.zeros_like(cond_mu_at_row, dtype="float64")

            m = mask_scored & (token_key_rows != None)
            if np.any(m):
                for i in np.where(m)[0]:
                    p = int(seqpos[i])
                    k = token_key_rows[i]
                    stat = cond_stats[t][p].get(k, None)
                    if stat is None:
                        continue
                    wy, sw, sw2 = stat
                    if sw <= 0:
                        continue
                    cond_mu_at_row[i] = wy / sw
                    neff_k = (sw * sw) / (sw2 + 1e-12)
                    cond_neff_at_row[i] = max(1.0, float(neff_k))

            y_train = np.vstack(train_df[t].values).astype("float64")
            err_col = err_map.get(t, None)
            if err_col is not None and err_col in train_df.columns:
                e = np.vstack(train_df[err_col].values).astype("float64")
                w_full = 1.0 / np.clip(e, eps, None) ** 2
                w_full = np.where(np.isfinite(y_train), w_full, 0.0)
                sumw = np.sum(w_full, axis=0) + 1e-12
                sumw2 = np.sum(w_full**2, axis=0) + 1e-12
                neff_pos = (sumw**2) / sumw2
            else:
                neff_pos = np.sum(np.isfinite(y_train), axis=0).astype("float64")

            neff_at_row = np.zeros(len(base_mu_at_row), dtype="float64")
            neff_at_row[mask_scored] = neff_pos[seqpos[mask_scored]]
            use_cond = cond_neff_at_row > 0
            neff_at_row[use_cond] = cond_neff_at_row[use_cond]

            alpha = neff_at_row / (neff_at_row + prior_strength)
            pred = alpha * cond_mu_at_row + (1.0 - alpha) * overall
        else:
            y_full = np.vstack(train_df[t].values).astype("float64")
            err_col = err_map.get(t, None)
            if err_col is not None and err_col in train_df.columns:
                e = np.vstack(train_df[err_col].values).astype("float64")
                w_full = 1.0 / np.clip(e, eps, None) ** 2
                w_full = np.where(np.isfinite(y_full), w_full, 0.0)
                sumw = np.sum(w_full, axis=0) + 1e-12
                sumw2 = np.sum(w_full**2, axis=0) + 1e-12
                neff = (sumw**2) / sumw2
            else:
                neff = np.sum(np.isfinite(y_full), axis=0).astype("float64")

            alpha_pos = neff / (neff + prior_strength)
            mu_shrunk = alpha_pos * mu + (1.0 - alpha_pos) * overall

            pred = np.full(seqpos.shape[0], overall, dtype="float64")
            pred[mask_scored] = mu_shrunk[seqpos[mask_scored]]

        baseline_pred_scored[t] = pred

    cal_params = {}
    for t in scored_targets:
        y_train = np.vstack(train_df[t].values).astype("float64")  # (n_train, 68)

        mu = per_pos_means[t]
        wsum = per_pos_wsum[t]
        overall = float(np.sum(mu * wsum) / (np.sum(wsum) + 1e-12))

        err_col = err_map.get(t, None)
        if err_col is not None and err_col in train_df.columns:
            e = np.vstack(train_df[err_col].values).astype("float64")
            w_full = 1.0 / np.clip(e, eps, None) ** 2
            w_full = np.where(np.isfinite(y_train), w_full, 0.0)
        else:
            w_full = np.ones_like(y_train, dtype="float64")
            w_full = np.where(np.isfinite(y_train), w_full, 0.0)

        if err_col is not None and err_col in train_df.columns:
            sumw = np.sum(w_full, axis=0) + 1e-12
            sumw2 = np.sum(w_full**2, axis=0) + 1e-12
            neff = (sumw**2) / sumw2
        else:
            neff = np.sum(np.isfinite(y_train), axis=0).astype("float64")

        alpha = neff / (neff + prior_strength)
        mu_shrunk = alpha * mu + (1.0 - alpha) * overall  # (68,)

        x_train = np.broadcast_to(mu_shrunk[None, :], y_train.shape).astype("float64")

        mask = np.isfinite(y_train)
        x = x_train[mask]
        y = y_train[mask]
        w = w_full[mask]

        if x.size < 10 or np.sum(w) <= 0:
            a, b = 1.0, 0.0
        else:
            Sw = float(np.sum(w))
            Sx = float(np.sum(w * x))
            Sy = float(np.sum(w * y))
            Sxx = float(np.sum(w * x * x))
            Sxy = float(np.sum(w * x * y))
            denom = Sw * Sxx - Sx * Sx
            if abs(denom) < 1e-12:
                a = 1.0
                b = (Sy / Sw) - a * (Sx / Sw)
            else:
                a = (Sw * Sxy - Sx * Sy) / denom
                b = (Sy - a * Sx) / Sw

        a = float(np.clip(a, 0.5, 1.5))
        b = float(np.clip(b, -0.5, 0.5))
        cal_params[t] = (a, b)

    for t in scored_targets:
        a, b = cal_params.get(t, (1.0, 0.0))
        baseline[t] = baseline_pred_scored[t] * a + b

    baseline["deg_pH10"] = baseline["deg_Mg_pH10"].astype("float64")
    baseline["deg_50C"] = baseline["deg_Mg_50C"].astype("float64")




## === cell 3
def load_submission_or_none(path, name):
    """Load a submission if it exists and has required columns, else return None."""
    if path is None or (not os.path.exists(path)):
        return None
    df = pd.read_csv(path)
    if "id_seqpos" not in df.columns:
        return None
    if all(c in df.columns for c in SUB_COLS):
        df = df[SUB_COLS].copy()
    else:
        return None
    return df


sub1 = load_submission_or_none(
    "../input/ae-pretrained-local/submission_ae_nonpretrained_local.csv", "sub1"
)  # @sin
sub2 = load_submission_or_none(
    "../input/ae-pretrained-local/submission_ae_pretrained_local.csv", "sub2"
)  # @sin
sub3 = load_submission_or_none(
    "../input/ae-pretrained-local/submission_ae_pretrained_local2.csv", "sub3"
)  # @sin
sub4 = load_submission_or_none(
    "../input/openvaccine-pytorch-ae-pretrain/submission.csv", "sub4"
)  # @public



## === cell 4
sub5 = load_submission_or_none(
    "../input/24551-ae-gcn/submission(3).csv", "sub5"
)  # @sin
sub6 = load_submission_or_none(
    "../input/hawkey-ae-pretrained-gcn-3ensemble/submission(4).csv", "sub6"
)  # @hawkey



## === cell 5
sub7 = load_submission_or_none(
    "../input/lstm-gru-5fold-2seeds-knncv/submission.csv", "sub7"
)  # @sin
sub8 = load_submission_or_none(
    "../input/hrunic-lstm-25177/submission(6).csv", "sub8"
)  # @hrunic
sub9 = load_submission_or_none(
    "../input/hrunic-gru-25365/submission(7).csv", "sub9"
)  # @hrunic
sub10 = load_submission_or_none(
    "../input/gru-lstm-with-feature-engineering-and-augmentation/submission.csv",
    "sub10",
)  # @public



## === cell 6
sub11 = load_submission_or_none(
    "../input/hawkey-gcn-only-25301/submission(5).csv", "sub11"
)  # @hawkey



## === cell 7
final_index = baseline.set_index("id_seqpos").index


def align_to_final(df):
    if df is None:
        return None
    df = df.drop_duplicates("id_seqpos")
    df = df.set_index("id_seqpos").reindex(final_index)
    for c in SUB_COLS[1:]:
        if c not in df.columns:
            df[c] = 0.0
    df = df[SUB_COLS[1:]].astype("float64")
    df = df.fillna(0.0)
    return df


sub1 = align_to_final(sub1)
sub2 = align_to_final(sub2)
sub3 = align_to_final(sub3)
sub4 = align_to_final(sub4)
sub5 = align_to_final(sub5)
sub6 = align_to_final(sub6)
sub7 = align_to_final(sub7)
sub8 = align_to_final(sub8)
sub9 = align_to_final(sub9)
sub10 = align_to_final(sub10)
sub11 = align_to_final(sub11)

baseline_aligned = align_to_final(baseline)




## === cell 8
def mean_of_available(dfs):
    dfs = [d for d in dfs if d is not None]
    if len(dfs) == 0:
        return None
    out = dfs[0].copy()
    for d in dfs[1:]:
        out = out + d
    return out / len(dfs)


block_a = mean_of_available([sub1, sub2, sub3, sub4])
block_b = mean_of_available([sub5, sub6])

if block_a is None and block_b is None:
    ac_pretrained = baseline_aligned
elif block_a is None:
    ac_pretrained = block_b
elif block_b is None:
    ac_pretrained = block_a
else:
    ac_pretrained = block_a * 0.5 + block_b * 0.5

block_c = mean_of_available([sub7, sub8, sub9, sub10])
block_d = sub11

if block_c is None and block_d is None:
    non_ac_pretrained = baseline_aligned
elif block_c is None:
    non_ac_pretrained = block_d
elif block_d is None:
    non_ac_pretrained = block_c
else:
    non_ac_pretrained = block_c * 0.6 + block_d * 0.4



## === cell 9
final_pred = ac_pretrained * 0.5 + non_ac_pretrained * 0.5
final_pred = final_pred.replace([np.inf, -np.inf], np.nan).fillna(0.0)

for c in SUB_COLS[1:]:
    final_pred[c] = final_pred[c].clip(lower=-0.5, upper=2.0)

final = pd.DataFrame({"id_seqpos": final_index})
for c in SUB_COLS[1:]:
    final[c] = final_pred[c].values



## === cell 10
out_path = "final_v1.csv"
final.to_csv(out_path, index=False)

assert out_path.endswith(".csv")
assert list(final.columns) == SUB_COLS
assert len(final) == len(sample_sub)
assert final["id_seqpos"].iloc[0] == sample_sub["id_seqpos"].iloc[0]
