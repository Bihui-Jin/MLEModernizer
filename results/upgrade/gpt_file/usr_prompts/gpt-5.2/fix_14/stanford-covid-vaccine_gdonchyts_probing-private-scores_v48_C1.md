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

0.49187

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.63824) has done: 'I remove the dependency on a missing external “baseline submission” file and instead build a valid submission directly from the provided `sample_submission.csv`, which guarantees the required shape/columns and a `.csv` output. I also fix the cell numbering to start at 1 (your current script starts at cell 0) and make the data paths robust to the provided `/kaggle/input/...` layout. Since you currently cannot produce any submission (no score yielded), these changes are necessary to run end-to-end; the resulting submission be valid (though it be a simple baseline using the sample values). Finally, I keep the original intent of editing a specific `id`’s `reactivity`, but only if that `id` exists in the submission index (otherwise it’s a no-op).'
- What this solution (achieved 0.47889) has done: 'Your current submission is essentially the sample_submission with one id’s reactivity set to 100, which badly hurts MCRMSE; the smallest change to move toward your target is to stop injecting that extreme value and instead output a conservative constant baseline derived from training labels. I keep the same “no model” core logic (no training loop/architecture) but compute per-target means on train (only for SN_filter==1 to better match test quality) and fill all rows with those means, which is a standard minimal baseline that should substantially reduce error vs zeros/extremes. I also make sure the output rows align exactly to sample_submission (same index/order/columns) to avoid any submission-format mismatch. These changes are directly aimed at lowering the score toward 0.3519 while remaining simple and stable.'
- What this solution (achieved 0.47889) has done: 'We keep your “constant baseline from train” core logic, but compute those constants only over the scored region (first `seq_scored` positions) because Kaggle only scores those and using unscored tail positions adds noise to the means. We also compute means only for the three scored targets and set the two unscored targets to the average of the two most related scored degradation means (a conservative fill that typically reduces error versus using their own noisier means). Finally, we keep exact `sample_submission.csv` row order and add a strict sanity check that `id_seqpos` aligns to `test.json` ids/seq_length so the submission can’t silently misalign and hurt MCRMSE.'
- What this solution (achieved 0.42166) has done: 'Your current constant-mean baseline is stable but it leaves easy signal unused: the labels vary by sequence position, and MCRMSE is computed positionwise. I keep the same “no model, no training loop” core logic but switch from a single global mean per target to a per-position mean vector (length `seq_scored`) computed from `train.json` (still using `SN_filter==1`), then fill each `id_seqpos` row by its position. This is a minimal change that typically improves MCRMSE materially versus a flat constant, moving your 0.47889 down toward the 0.3519 target without altering evaluation semantics or requiring extra packages. I also add a safe clamp on `seqpos` so unscored tail positions (>=68) reuse the last scored mean, keeping the submission valid for all 107 positions.'
- What this solution (achieved 0.48258) has done: 'We keep your current “per-position mean vector” baseline (core logic) and make a minimal, score-relevant improvement by filtering out low-quality training samples more strictly using both `SN_filter==1` and a `signal_to_noise` threshold, which better matches the curated test distribution and typically lowers MCRMSE. We also compute the per-position mean with a simple weighted average using the provided per-position error arrays (inverse-variance weighting), which is still a “mean-per-position” baseline but uses label reliability to reduce noise—often a small but consistent improvement. Finally, we ensure we only use the first `seq_scored` positions for both values and errors and keep the exact `sample_submission.csv` order/shape so the submission remains valid.'
- What this solution (achieved 0.48256) has done: 'We keep your exact “per-position weighted mean baseline” core logic, but make two minimal, score-relevant corrections: (1) compute the per-position means only from the *scored region* while explicitly dropping any non-finite label values (these exist in this dataset and can silently bias weighted means), and (2) align the training filter closer to the curated test distribution by using a slightly stricter `signal_to_noise` threshold (this typically reduces MCRMSE vs 1.0 without changing the approach). We also make the weighting more robust by zeroing weights where errors are non-finite/zero and by ignoring non-finite y values rather than letting them propagate. Everything else (no model, same per-position fill, same submission format/ordering) stays the same to keep changes minimal and stable.'
- What this solution (achieved 0.48258) has done: 'We’re currently worse than the target (0.48256 vs 0.35188, lower is better), so we should make a small, safe change that improves generalization without changing the “per-position weighted mean baseline” core logic. The biggest lever left in this setup is how we filter training rows to better match the curated test distribution; your current `signal_to_noise >= 1.25` is likely excluding too much useful data and making the per-position estimates noisy. I keep the same weighted-mean-per-position computation, but tune the filter to `signal_to_noise >= 1.0` (the test curation threshold) and add a tiny weight cap to prevent a few extremely small errors from dominating the weighted mean. Everything else (feature-free baseline, same targets, same submission order/shape) remains unchanged.'
- What this solution (achieved 0.48251) has done: 'You’re currently worse than the target (0.4826 vs 0.3519; lower is better), so we should improve with the smallest change that keeps your “per-position weighted mean baseline” core logic. The biggest likely issue is distribution mismatch from using a hard `signal_to_noise >= 1.0` cutoff plus inverse-variance weighting, which can overfit per-position noise; instead, we keep all `SN_filter==1` rows but incorporate `signal_to_noise` as a *soft sample weight* in the same weighted-mean computation. Concretely: multiply the per-position inverse-variance weights by a clipped `signal_to_noise` factor (downweight noisy samples, but don’t discard them), while keeping the existing error-weight cap and non-finite handling. This should stabilize the per-position estimates and typically reduce MCRMSE toward your target without changing the submission format, targets, or overall approach.'
- What this solution (achieved 0.48224) has done: 'We keep your current “per-position weighted mean baseline” exactly, but adjust two score-critical knobs that can reduce MCRMSE without changing the approach: (1) use only the curated-like subset by softly weighting by `signal_to_noise` *and* excluding extreme low-quality rows via a very mild floor (keeps core logic but reduces noisy label influence), and (2) change the soft-weight clipping range to be wider so high-quality rows can contribute more (your current `[0.5, 2.0]` is likely too compressed). We also stop forcing unscored targets to the average of two scored targets and instead compute their own per-position weighted means from train (still the same baseline logic), because although unscored they can interact indirectly with sanity/format and typically don’t hurt. All I/O paths, submission shape/order, and the weighted-mean-per-position filling remain the same.'
- What this solution (achieved 0.5183) has done: 'Your current approach is a per-position (length `seq_scored`) weighted-mean baseline; the simplest way to move the MCRMSE down toward the target is to make that weighting more robust rather than changing the overall method. I keep the same per-position inverse-variance aggregation, but (1) Winsorize (clip) training label values per position to reduce the impact of extreme/outlier measurements that can skew means, and (2) add a small global shrinkage that blends each per-position estimate toward the overall mean for that target to reduce per-position noise/overfitting. Both are minimal post-processing around the same “compute per-position mean from train then fill submission by `seqpos`” core logic, and they keep all I/O and submission formatting identical.'
- What this solution (achieved 0.48583) has done: 'We’re currently worse than the target (0.5183 vs 0.3519; lower is better), and the largest likely reason is that the new winsorization + strong shrinkage (alpha=0.85) is over-smoothing per-position signal and hurting MCRMSE. I keep the exact same “per-position inverse-variance weighted mean” core logic, but reduce shrinkage substantially (so we use more of the per-position information) and make the outlier clipping less aggressive (wider quantiles) so true signal extremes aren’t flattened. These are minimal, metric-aligned tweaks that should move the score down toward the target without changing the approach, data paths, or submission format. The script still run end-to-end and write a valid `submission.csv` aligned to `sample_submission.csv`.'
- What this solution (achieved 0.48247) has done: 'We’re worse than the target (0.48583 vs 0.35188; lower is better), so we should make the smallest change likely to improve MCRMSE without changing your “per-position weighted mean” core logic. The biggest likely regression in your recent tweaks is over-regularization: quantile clipping plus shrinkage can wash out real per-position signal, so I reduce shrinkage to near-zero and widen the clipping quantiles further (making winsorization almost a no-op but still protecting against pathological outliers). I also lightly relax the inverse-variance cap so reliable measurements can contribute more, while keeping all the same data paths, filters, and submission alignment logic. Everything else (no model/training loop, same per-position aggregation, same output schema and order) remains unchanged.'
- What this solution (achieved 0.49187) has done: 'We’re currently worse than the target (0.48247 vs 0.35188; lower is better), so the smallest likely-to-help adjustment without changing your “per-position inverse-variance weighted mean baseline” core logic is to reduce overfitting/noise in the per-position estimates. I keep the same weighted-mean computation but (1) compute clip bounds from the *same finite/valid subset* used for weighting (so bounds aren’t driven by junk/outliers), (2) add a very light per-position smoothing across neighboring positions (RNA measurements are locally correlated), and (3) slightly increase shrinkage toward the global mean just enough to stabilize high-variance positions. Submission format, row ordering, and all I/O paths remain unchanged and it still write a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd



## === cell 1
BASE_INPUT = "/kaggle/input/stanford-covid-vaccine"
if not os.path.exists(BASE_INPUT):
    BASE_INPUT = "../input/stanford-covid-vaccine"

train_path = os.path.join(BASE_INPUT, "train.json")
test_path = os.path.join(BASE_INPUT, "test.json")
sample_path = os.path.join(BASE_INPUT, "sample_submission.csv")

df_train = pd.read_json(train_path, lines=True)
df_test = pd.read_json(test_path, lines=True)
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



## === cell 2
targets = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
scored_targets = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]  # only these are evaluated

df_train_use = df_train.copy()

if "SN_filter" in df_train_use.columns:
    df_train_use = df_train_use[df_train_use["SN_filter"] == 1]

df_train_use = df_train_use.reset_index(drop=True)

k_scored = 68
if "seq_scored" in df_train_use.columns and len(df_train_use):
    k_scored = int(df_train_use["seq_scored"].mode().iloc[0])

err_col = {
    "reactivity": "reactivity_error",
    "deg_Mg_pH10": "deg_error_Mg_pH10",
    "deg_pH10": "deg_error_pH10",
    "deg_Mg_50C": "deg_error_Mg_50C",
    "deg_50C": "deg_error_50C",
}


def compute_pos_clip_bounds(
    values_series: pd.Series, k: int, lo_q: float = 0.001, hi_q: float = 0.999
) -> tuple[np.ndarray, np.ndarray]:
    mat = np.full((len(values_series), k), np.nan, dtype=np.float64)
    for i, v in enumerate(values_series.values):
        if v is None:
            continue
        arr = np.asarray(v, dtype=np.float64)
        m = min(len(arr), k)
        if m > 0:
            mat[i, :m] = arr[:m]
    lo = np.nanquantile(mat, lo_q, axis=0)
    hi = np.nanquantile(mat, hi_q, axis=0)

    all_finite = mat[np.isfinite(mat)]
    if all_finite.size == 0:
        glo_lo, glo_hi = -np.inf, np.inf
    else:
        glo_lo = np.nanquantile(all_finite, lo_q)
        glo_hi = np.nanquantile(all_finite, hi_q)

    lo = np.where(np.isfinite(lo), lo, glo_lo)
    hi = np.where(np.isfinite(hi), hi, glo_hi)
    return lo.astype(np.float64), hi.astype(np.float64)


def weighted_mean_vector_first_k(
    values_series: pd.Series,
    errors_series: pd.Series,
    k: int,
    sample_weight: np.ndarray | None = None,
    clip_lo: np.ndarray | None = None,
    clip_hi: np.ndarray | None = None,
) -> np.ndarray:
    """
    Per-position inverse-variance weighted mean (core logic preserved).
    """
    num = np.zeros(k, dtype=np.float64)
    den = np.zeros(k, dtype=np.float64)

    eps = 1e-3
    w_max = 3e6

    vvals = values_series.values
    evals = (
        errors_series.values
        if errors_series is not None
        else [None] * len(values_series)
    )

    if sample_weight is None:
        sample_weight = np.ones(len(values_series), dtype=np.float64)
    else:
        sample_weight = np.asarray(sample_weight, dtype=np.float64)
        if len(sample_weight) != len(values_series):
            raise ValueError("sample_weight length mismatch")

    for v, e, sw in zip(vvals, evals, sample_weight):
        if v is None:
            continue
        if not np.isfinite(sw) or sw <= 0:
            continue

        vv = np.asarray(v, dtype=np.float64)
        m = min(len(vv), k)
        if m <= 0:
            continue

        yy = vv[:m]
        ymask = np.isfinite(yy)
        if not np.any(ymask):
            continue

        if clip_lo is not None and clip_hi is not None:
            yy = yy.copy()
            yy[ymask] = np.clip(yy[ymask], clip_lo[:m][ymask], clip_hi[:m][ymask])

        if e is None:
            w = np.ones(m, dtype=np.float64)
        else:
            ee = np.asarray(e, dtype=np.float64)
            if len(ee) >= m:
                ee = ee[:m]
            else:
                ee = np.pad(ee, (0, m - len(ee)), constant_values=np.nan)

            emask = np.isfinite(ee) & (ee > 0)
            w = np.zeros(m, dtype=np.float64)
            inv_var = 1.0 / np.maximum(ee[emask], eps) ** 2
            w[emask] = np.minimum(inv_var, w_max)

        w = w * ymask.astype(np.float64) * float(sw)
        if w.sum() <= 0:
            continue

        num[:m] += w * yy
        den[:m] += w

    vec = np.divide(num, np.maximum(den, 1e-12))
    vec = vec.astype(np.float32)
    vec = np.where(np.isfinite(vec), vec, np.float32(0.0)).astype(np.float32)
    return vec


if "signal_to_noise" in df_train_use.columns:
    sn_all = pd.to_numeric(df_train_use["signal_to_noise"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    keep = np.isfinite(sn_all) & (sn_all >= 0.5)  # mild floor (kept)
    df_train_use = df_train_use.loc[keep].reset_index(drop=True)
    sn = sn_all[keep]
else:
    sn = None

if sn is not None:
    sample_w = np.clip(sn, 0.5, 5.0)
    sample_w = np.where(np.isfinite(sample_w), sample_w, 1.0)
else:
    sample_w = np.ones(len(df_train_use), dtype=np.float64)


def _finite_series_only(s: pd.Series) -> pd.Series:
    vals = []
    for v in s.values:
        if v is None:
            vals.append(None)
            continue
        a = np.asarray(v, dtype=np.float64)
        if a.size == 0:
            vals.append(None)
            continue
        a = a.copy()
        a[~np.isfinite(a)] = np.nan
        vals.append(a.tolist())
    return pd.Series(vals, index=s.index)


def smooth_vec(vec: np.ndarray) -> np.ndarray:
    v = vec.astype(np.float32)
    if v.size < 3:
        return v
    out = v.copy()
    out[1:-1] = 0.25 * v[:-2] + 0.50 * v[1:-1] + 0.25 * v[2:]
    out[0] = 0.75 * v[0] + 0.25 * v[1]
    out[-1] = 0.75 * v[-1] + 0.25 * v[-2]
    return out.astype(np.float32)


clip_bounds = {}
pos_means_raw = {}

for t in targets:
    t_series = _finite_series_only(df_train_use[t])
    clip_lo, clip_hi = compute_pos_clip_bounds(t_series, k_scored, 0.001, 0.999)
    clip_bounds[t] = (clip_lo, clip_hi)

    ecol = err_col.get(t)
    errs = df_train_use[ecol] if (ecol in df_train_use.columns) else None
    pos_means_raw[t] = weighted_mean_vector_first_k(
        t_series,
        errs,
        k_scored,
        sample_weight=sample_w,
        clip_lo=clip_lo,
        clip_hi=clip_hi,
    )

pos_means = {}
train_means = {}

shrink_alpha = 0.12  # was 0.05

for t in targets:
    g = np.float32(np.nanmean(pos_means_raw[t]))
    train_means[t] = g
    v = (shrink_alpha * g + (1.0 - shrink_alpha) * pos_means_raw[t]).astype(np.float32)
    pos_means[t] = smooth_vec(v)

train_means



## === cell 3
for c in targets:
    df[c] = pd.to_numeric(df[c], errors="coerce").astype(np.float32)

seqpos = (
    df["id_seqpos"]
    .astype(str)
    .str.rsplit("_", n=1, expand=True)[1]
    .astype(int)
    .to_numpy()
)
seqpos_clamped = np.clip(seqpos, 0, k_scored - 1)

for c in targets:
    vec = pos_means.get(c, None)
    if vec is None or len(vec) != k_scored:
        df[c] = train_means[c]
    else:
        df[c] = vec[seqpos_clamped].astype(np.float32)

df[targets] = (
    df[targets].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
)



## === cell 4
expected_rows = int(df_test["seq_length"].sum()) if len(df_test) else len(df)
print("Expected rows (sum seq_length over test):", expected_rows)
print("Submission rows:", len(df))
print("Columns:", list(df.columns))

if len(df) != expected_rows:
    raise ValueError(
        f"Row count mismatch: submission has {len(df)} rows but test expects {expected_rows}"
    )

test_ids = set(df_test["id"].astype(str).tolist())


def extract_id(id_seqpos: str) -> str:
    parts = str(id_seqpos).split("_")
    return "_".join(parts[1:-1]) if len(parts) >= 3 else ""


if (
    extract_id(df["id_seqpos"].iloc[0]) not in test_ids
    or extract_id(df["id_seqpos"].iloc[-1]) not in test_ids
):
    print(
        "Warning: id_seqpos ids don't match test.json ids on boundary rows; check dataset paths."
    )



## === cell 5
out_path = "submission.csv"
df.to_csv(out_path, index=False)

print("Wrote:", out_path)
print("Submission shape:", df.shape)
print("Head:\n", df.head())
