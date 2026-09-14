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

0.4495

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.63824) has done: 'I remove the dependency on the missing `../input/worst-submission/ensemble52.csv` and instead build a valid submission directly from the provided `sample_submission.csv` (guarantees correct shape/columns and fixes the `df` undefined errors). To keep the original core intent (a “worst” baseline with a small manual override), I fill predictions with zeros for all targets and then apply your existing override for the specified `id` on `reactivity` (it becomes a no-op but preserves semantics). I also make the input path robust by falling back between the two provided dataset locations. Finally, I ensure the output is written as `submission.csv` with the exact required columns.'
- What this solution (achieved 0.42418) has done: 'Your current submission is essentially all-zeros, which is why the MCRMSE is far from the target (lower is better), so the smallest legitimate improvement is to replace the constant predictions with simple position-wise averages learned from the training set. This keeps the “no model / no training loop” core logic intact while moving score down toward the target by using real signal from `train.json`. To preserve required submission shape, we still start from `sample_submission.csv` and fill its rows in-order by `seqpos`, using train-derived means for the first 68 positions and a safe fallback (position 67 mean) for unscored positions > 67. We keep your existing manual override line (it remains effectively a no-op) and ensure the output columns/order match exactly.'
- What this solution (achieved 0.44283) has done: 'We keep your “position-wise mean baseline” core logic, but make it slightly stronger in a minimal, safe way by computing means only on high-quality training rows (using the provided `SN_filter`), which typically reduces noise and improves MCRMSE toward your lower target. We also add a tiny amount of smoothing across adjacent positions (a 3-point moving average) to reduce per-position variance without changing the overall approach or adding a model/training loop. Finally, we preserve your submission-building method from `sample_submission.csv` to guarantee correct ordering/shape and keep the existing manual override line as-is. These changes should move the score down (better) from 0.42418 toward the 0.3519 target without altering evaluation semantics.'
- What this solution (achieved 0.49857) has done: 'Your current score (0.44283, lower-is-better) is still worse than the target (0.35188), so we should make a small, legitimate improvement without changing the overall “position-wise mean baseline” approach. I keep the same pipeline (train-derived position means → fill sample_submission rows by seqpos) but switch from plain means to a more robust, noise-aware aggregation: inverse-variance weighting using the provided per-position error arrays, computed on SN_filter==1 rows. This typically reduces the influence of noisy measurements and should move MCRMSE downward toward the target while preserving the same core logic and submission semantics. Everything else (paths, smoothing, fallback for unscored positions, required columns, and writing submission.csv) remains intact.'
- What this solution (achieved 0.46464) has done: 'We should move the score down (better) from 0.49857 toward the 0.35188 target, but with minimal changes and without changing the overall “position-wise baseline from train.json → fill sample_submission by seqpos” approach. Your last change (inverse-variance weighting by per-position errors) likely over-trusted very small reported errors, which can hurt generalization; we keep the same weighting idea but add a small error floor (regularization) and use a more robust weight of 1/(e^2 + floor^2). We also use SN_filter==1 (as you already do) and keep the same 3-point smoothing and the same fallback for unscored positions to preserve semantics. This should typically improve stability and reduce MCRMSE toward the target without any new model/training loop.'
- What this solution (achieved 0.44773) has done: 'Your current baseline is still worse than the target (0.46464 vs 0.35188, lower-is-better), so we should make a small, legitimate improvement without changing the overall “train-derived position baseline → fill sample_submission by seqpos” logic. The most likely issue is that pure error-weighting can overfit per-position noise; a minimal fix is to shrink the error-weighted estimate toward the plain mean estimate using a single global mixing factor (keeps the same baseline approach, just stabilizes it). We also tune the error floor slightly to reduce extreme weights and keep the existing 3-point smoothing and SN_filter selection. These changes should improve generalization and move MCRMSE down toward the target while preserving submission shape/semantics.'
- What this solution (achieved 0.44928) has done: 'We should move your score down (better) from 0.44773 toward the 0.35188 target while keeping the exact same “position-wise baseline from train.json → fill sample_submission by seqpos” logic. The smallest likely gain is to (1) use a slightly stricter, competition-standard training filter (SN_filter==1 and signal_to_noise>1) to reduce noisy labels, and (2) tune the shrinkage toward the plain mean so the error-weighted estimate doesn’t overfit (reduce mix_plain_mean a bit). I keep your same 1/(err^2+floor^2) weighting, 3-point smoothing, and the same submission-building pipeline/format. These tweaks are minimal, deterministic, and aimed at improving generalization on the scored targets without changing the overall approach.'
- What this solution (achieved 0.44852) has done: 'We keep your exact “position-wise baseline from train.json → fill sample_submission by seqpos” approach, but make two minimal changes aimed at reducing MCRMSE toward the (lower-is-better) target: (1) compute the baseline only over the 3 scored targets (and then copy those predictions into the 2 unscored columns to reduce noise from learning separate baselines that aren’t evaluated), and (2) use a tiny, deterministic per-position shrinkage toward the global mean of each target to stabilize the baseline against position-specific over/under-shoots. This does not add any model, training loop, or new features; it only adjusts how the same per-position averages are aggregated. The submission-building logic, ordering, and output format remain unchanged and still write a valid `submission.csv`.'
- What this solution (achieved 0.44852) has done: 'Your current score (0.44852, lower-is-better) is still far worse than the target (0.35188), so we should make a small, legitimate improvement while keeping the same “position-wise baseline from train.json → fill sample_submission by seqpos” core logic. The largest issue is that you currently force the two unscored columns to copy scored columns; while not scored directly, they are correlated with the scored degradation targets and learning them separately from train typically helps overall calibration and avoids distribution mismatches. I keep your same SN_filter + signal_to_noise filter, the same error-regularized inverse-variance weighting, the same 3-point smoothing, and the same shrink-to-global stabilization, but compute baselines for all 5 targets (instead of copying). This is a minimal change, preserves evaluation semantics and runtime, and should move MCRMSE downward toward the target band.'
- What this solution (achieved 0.4486) has done: 'To move your MCRMSE down toward the target with minimal disruption, I keep your exact “train-derived position baseline → fill sample_submission by seqpos” pipeline and only adjust the aggregation so it matches the leaderboard distribution better. Specifically, I (1) remove the `signal_to_noise > 1.0` cutoff (which applies to the filtered test set but can bias the training mean estimates when used as a hard filter), and (2) add a tiny “use-unfiltered-for-global-shrink” stabilization: position means are still computed from the filtered set, but the global mean used in your shrinkage term comes from all training rows to better match overall target scale. Everything else (inverse-variance weighting with floor, smoothing, shrink, submission formatting, and writing `submission.csv`) remains intact and deterministic.'
- What this solution (achieved 0.44874) has done: 'We keep your exact “train-derived position baseline → fill sample_submission by seqpos” pipeline, but make the aggregation slightly more robust to outliers and noisy tails to improve MCRMSE (lower-is-better) toward your 0.3519 target. Concretely, we winsorize (clip) training values per target using global percentiles before computing the same error-regularized inverse-variance weighted means; this preserves the same baseline logic but reduces the influence of extreme measurements. We also (minimally) compute the global shrink mean from the same winsorized values to keep calibration consistent, while leaving your smoothing, shrink strength, weighting form, and submission formatting unchanged. This should be a small, legitimate improvement without introducing any new model/training loop and still write a valid `submission.csv`.'
- What this solution (achieved 0.44874) has done: 'Your current approach is a deterministic “position-wise baseline” from train → fill `sample_submission` by `seqpos`, and your score (0.44874, lower-is-better) is still far from the target (0.35188), so we should make a small improvement without changing the core method. The most impactful minimal fix is to stop optimizing the two *unscored* targets with their own noisy baselines and instead predict them as simple, stable linear transforms of the scored degradation targets learned from train (still no model/training loop; just a global least-squares mapping). This typically improves calibration/consistency and reduces distribution mismatch that can indirectly hurt the scored columns in MCRMSE-style setups, while keeping the same main baseline and submission construction. Everything else (SN_filter usage, error-regularized weighting, smoothing, shrinkage, winsorization, and CSV writing) is left intact.'
- What this solution (achieved 0.44979) has done: 'Your current score (0.44874, lower-is-better) is still worse than the target (0.35188), so we should make a small, legitimate improvement while keeping the same “train-derived per-position baseline → fill sample_submission by seqpos” pipeline. The most likely low-risk gain is to align the training aggregation closer to the metric by (a) filtering out the noisiest training samples using a gentle `signal_to_noise` cutoff in addition to `SN_filter==1`, and (b) using the provided per-position errors not only as weights but also to compute a *weighted* global mean for the shrinkage term (so shrinkage reflects measurement confidence). These are minimal changes that preserve your core logic (same inverse-variance weighting, smoothing, shrinkage, and linear mapping for unscored targets) and should move MCRMSE downward toward the target. The submission building and output format remain unchanged and still write a valid `submission.csv`.'
- What this solution (achieved 0.44979) has done: 'We need to move your MCRMSE down (better) from 0.44979 toward the 0.35188 target, so the smallest likely improvement is to stop enforcing a test-like `signal_to_noise>=1.0` cutoff on training aggregation (it can bias the per-position means upward/downward and hurt generalization). I keep your exact core pipeline (winsorized position-wise aggregation + error-regularized inverse-variance weighting + 3-point smoothing + global shrink + linear mapping for unscored targets + fill `sample_submission` by `seqpos`) and only adjust the training filter to use `SN_filter==1` only, while keeping the weighted-global-mean shrinkage you added. This is minimal, deterministic, and should typically reduce the error by using more training signal without changing model semantics or output format. The script still write a valid `submission.csv` with the required columns and ordering.'
- What this solution (achieved 0.4495) has done: 'Your current score (0.44979, lower-is-better) is still worse than the target (0.35188), so we should make a small improvement without changing the overall “train-derived per-position baseline → fill sample_submission by seqpos” pipeline. The main low-risk fix is to better match the evaluation focus by (a) computing the per-position baselines using only the *scored length* (first 68) exactly (guarding against any accidental length mismatch) and (b) tuning down the shrinkage-to-global and mapping blend slightly so we rely a bit more on true per-position signal (which is what MCRMSE rewards) rather than overly-smoothed/global values. I keep your inverse-variance weighting with error floor, winsorization, 3-point smoothing, SN_filter==1 filter, and the same submission construction/format. These changes are minimal, deterministic, and should move MCRMSE downward toward the target band while still producing a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd



## === cell 1
BASE_CANDIDATES = [
    "../input/stanford-covid-vaccine",
    "/kaggle/input/stanford-covid-vaccine",
    "../kaggle/input/stanford-covid-vaccine",
    "/kaggle/data/stanford-covid-vaccine",
    "/kaggle/data",
]
base_path = next((p for p in BASE_CANDIDATES if os.path.exists(p)), None)
if base_path is None:
    raise FileNotFoundError(f"Could not find dataset in any of: {BASE_CANDIDATES}")


def _resolve(path_root: str, fname: str) -> str:
    p1 = os.path.join(path_root, fname)
    p2 = os.path.join(path_root, "stanford-covid-vaccine", fname)
    if os.path.exists(p1):
        return p1
    if os.path.exists(p2):
        return p2
    raise FileNotFoundError(f"Could not find {fname} under {path_root}")


train_path = _resolve(base_path, "train.json")
test_path = _resolve(base_path, "test.json")
sample_sub_path = _resolve(base_path, "sample_submission.csv")

df_train = pd.read_json(train_path, lines=True)
df_test = pd.read_json(test_path, lines=True)
sample_sub = pd.read_csv(sample_sub_path)

print("Loaded:", df_train.shape, df_test.shape, sample_sub.shape)



## === cell 2
target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
scored_targets = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]

seq_scored = int(df_train["seq_scored"].iloc[0])  # 68
seq_len = int(df_train["seq_length"].iloc[0])  # 107

df_train_used = df_train.copy()

if "SN_filter" in df_train_used.columns:
    df_train_used = df_train_used[df_train_used["SN_filter"].astype(int) == 1]

sn_min = None  # keep printed for transparency

df_train_used = df_train_used.reset_index(drop=True)
if len(df_train_used) == 0:
    df_train_used = df_train


def _smooth_3(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    y = x.copy()
    if len(x) >= 3:
        y[1:-1] = (x[:-2] + x[1:-1] + x[2:]) / 3.0
        y[0] = (x[0] + x[1]) / 2.0
        y[-1] = (x[-2] + x[-1]) / 2.0
    return y


err_map = {
    "reactivity": "reactivity_error",
    "deg_Mg_pH10": "deg_error_Mg_pH10",
    "deg_pH10": "deg_error_pH10",
    "deg_Mg_50C": "deg_error_Mg_50C",
    "deg_50C": "deg_error_50C",
}

pos_means = {}

eps = 1e-6
error_floor = 0.10
mix_plain_mean = 0.15  # 0 => pure error-weighted; 1 => pure plain mean

shrink_to_global = 0.03  # was 0.06

winsor_q_low = 0.005
winsor_q_high = 0.995

cols_to_fit = list(target_cols)

for col in cols_to_fit:
    y = np.vstack(df_train_used[col].values).astype(np.float32)[
        :, :seq_scored
    ]  # (n_samples, 68)

    y_all = y.reshape(-1)
    lo = float(np.quantile(y_all, winsor_q_low))
    hi = float(np.quantile(y_all, winsor_q_high))
    y = np.clip(y, lo, hi)

    plain = y.mean(axis=0).astype(np.float32)

    err_col = err_map.get(col, None)

    if err_col is not None and err_col in df_train.columns:
        y_global = np.vstack(df_train[col].values).astype(np.float32)[
            :, :seq_scored
        ]  # (n_all, 68)
        y_global_flat = y_global.reshape(-1)
        glo = float(np.quantile(y_global_flat, winsor_q_low))
        ghi = float(np.quantile(y_global_flat, winsor_q_high))
        y_global_flat = np.clip(y_global_flat, glo, ghi).astype(np.float32)

        e_global = (
            np.vstack(df_train[err_col].values)
            .astype(np.float32)[:, :seq_scored]
            .reshape(-1)
        )
        denom_g = (np.maximum(e_global, 0.0) ** 2 + (error_floor**2)).astype(np.float32)
        w_g = 1.0 / np.maximum(denom_g, eps)
        w_g = np.clip(w_g, 0.0, 1e6).astype(np.float32)

        global_mean = float((w_g * y_global_flat).sum() / (w_g.sum() + eps))
    else:
        y_global = (
            np.vstack(df_train[col].values)
            .astype(np.float32)[:, :seq_scored]
            .reshape(-1)
        )
        glo = float(np.quantile(y_global, winsor_q_low))
        ghi = float(np.quantile(y_global, winsor_q_high))
        global_mean = float(np.clip(y_global, glo, ghi).mean())

    if err_col is not None and err_col in df_train_used.columns:
        e = np.vstack(df_train_used[err_col].values).astype(np.float32)[:, :seq_scored]
        denom = (np.maximum(e, 0.0) ** 2 + (error_floor**2)).astype(np.float32)
        w = 1.0 / np.maximum(denom, eps)
        w = np.clip(w, 0.0, 1e6).astype(np.float32)

        weighted = (w * y).sum(axis=0) / (w.sum(axis=0) + eps)
        m = (1.0 - mix_plain_mean) * weighted + mix_plain_mean * plain
    else:
        m = plain

    m = _smooth_3(m)

    m = ((1.0 - shrink_to_global) * m + shrink_to_global * global_mean).astype(
        np.float32
    )

    pos_means[col] = m

fallback = {col: float(pos_means[col][-1]) for col in target_cols}

print("Using train rows:", len(df_train_used), "of", len(df_train))
print("Computed position baselines for:", list(pos_means.keys()))
print(
    "Weighting:",
    f"1/(err^2 + {error_floor}^2), then mixed with plain mean at {mix_plain_mean}",
    f"and shrink_to_global={shrink_to_global}",
)
print(f"Winsorization per target: q=[{winsor_q_low}, {winsor_q_high}]")
print("Train filter: SN_filter==1 (if present); signal_to_noise cutoff removed")




## === cell 3
def _fit_linear_map(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Fit y ≈ b0 + b1*X1 + b2*X2 via least squares; returns coef [b0,b1,b2].
    X shape: (n,2)
    y shape: (n,)
    """
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    A = np.concatenate([np.ones((X.shape[0], 1), dtype=np.float32), X], axis=1)  # (n,3)
    coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    return coef.astype(np.float32)


deg_mg_ph10 = (
    np.vstack(df_train_used["deg_Mg_pH10"].values)
    .astype(np.float32)[:, :seq_scored]
    .reshape(-1)
)
deg_mg_50c = (
    np.vstack(df_train_used["deg_Mg_50C"].values)
    .astype(np.float32)[:, :seq_scored]
    .reshape(-1)
)

X = np.stack([deg_mg_ph10, deg_mg_50c], axis=1)  # (n*68, 2)

y_deg_ph10 = (
    np.vstack(df_train_used["deg_pH10"].values)
    .astype(np.float32)[:, :seq_scored]
    .reshape(-1)
)
y_deg_50c = (
    np.vstack(df_train_used["deg_50C"].values)
    .astype(np.float32)[:, :seq_scored]
    .reshape(-1)
)


def _winsorize_flat(v: np.ndarray, ql: float, qh: float) -> np.ndarray:
    lo = float(np.quantile(v, ql))
    hi = float(np.quantile(v, qh))
    return np.clip(v, lo, hi).astype(np.float32)


Xw = X.copy()
Xw[:, 0] = _winsorize_flat(Xw[:, 0], winsor_q_low, winsor_q_high)
Xw[:, 1] = _winsorize_flat(Xw[:, 1], winsor_q_low, winsor_q_high)
y_deg_ph10_w = _winsorize_flat(y_deg_ph10, winsor_q_low, winsor_q_high)
y_deg_50c_w = _winsorize_flat(y_deg_50c, winsor_q_low, winsor_q_high)

coef_ph10 = _fit_linear_map(Xw, y_deg_ph10_w)
coef_50c = _fit_linear_map(Xw, y_deg_50c_w)


def _apply_linear_map(coef: np.ndarray, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    return (coef[0] + coef[1] * x1 + coef[2] * x2).astype(np.float32)


mapped_deg_ph10 = _apply_linear_map(
    coef_ph10, pos_means["deg_Mg_pH10"], pos_means["deg_Mg_50C"]
)
mapped_deg_50c = _apply_linear_map(
    coef_50c, pos_means["deg_Mg_pH10"], pos_means["deg_Mg_50C"]
)

mapped_deg_ph10 = _smooth_3(mapped_deg_ph10)
mapped_deg_50c = _smooth_3(mapped_deg_50c)

map_blend = 0.65  # was 0.80
pos_means["deg_pH10"] = (
    map_blend * mapped_deg_ph10 + (1.0 - map_blend) * pos_means["deg_pH10"]
).astype(np.float32)
pos_means["deg_50C"] = (
    map_blend * mapped_deg_50c + (1.0 - map_blend) * pos_means["deg_50C"]
).astype(np.float32)

fallback = {col: float(pos_means[col][-1]) for col in target_cols}

print("Linear map coefficients:")
print("deg_pH10 ~ b0 + b1*deg_Mg_pH10 + b2*deg_Mg_50C:", coef_ph10)
print("deg_50C  ~ b0 + b1*deg_Mg_pH10 + b2*deg_Mg_50C:", coef_50c)
print("Applied mapped baselines with blend=", map_blend)



## === cell 4
df = sample_sub.copy()

seqpos = df["id_seqpos"].astype(str).str.rsplit("_", n=1).str[-1].astype(int).values

for col in target_cols:
    vals = np.empty(len(df), dtype=np.float32)
    mask_scored = seqpos < seq_scored
    vals[mask_scored] = pos_means[col][seqpos[mask_scored]]
    vals[~mask_scored] = fallback[col]
    df[col] = vals



## === cell 5
df.loc[df["id_seqpos"].astype(str).str.startswith("id_ff2d18b94"), "reactivity"] = (
    df.loc[
        df["id_seqpos"].astype(str).str.startswith("id_ff2d18b94"), "reactivity"
    ].astype(float)
)



## === cell 6
required_cols = ["id_seqpos"] + target_cols
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Submission is missing required columns: {missing}")

df = df[required_cols]
df.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", df.shape)
print(df.head())
print("Means (first 5 positions) reactivity:", pos_means["reactivity"][:5])
