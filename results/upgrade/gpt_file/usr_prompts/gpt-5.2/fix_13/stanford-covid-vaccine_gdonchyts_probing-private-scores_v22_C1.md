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

0.3663833104739942

# 6. Current score

0.42156

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.63824) has done: 'Your current notebook fails immediately because it tries to read a non-existent `../input/worst-submission/ensemble52.csv`, so `df` is never created and later cells crash. To make it run end-to-end and always produce a valid submission, I instead load the provided `sample_submission.csv` from the competition input path, which guarantees correct row count/columns and alignment. I keep your existing scaling step (`/= 1.14`) intact (score likely be poor, but your current score is “Not yielded”, so the priority is generating a valid CSV). I also add a small path fallback so it works whether the dataset is mounted at `../input/stanford-covid-vaccine` or directly under `../input`.'
- What this solution (achieved 0.42541) has done: 'Your current submission is effectively a constant all-zeros prediction (the `/= 1.14` does nothing to zeros), which explains the weak score. To move the score down toward your target with minimal change and without altering the “no real model” core approach, I replace the constant zeros with per-position empirical means computed from the training set (a strong baseline for this competition). This keeps the same overall pipeline (read data → fill predictions → write `submission.csv`) while making predictions statistically closer to the ground truth under MCRMSE. I also ensure the per-id/per-position ordering exactly matches `sample_submission.csv` so the submission stays valid.'
- What this solution (achieved 0.42181) has done: 'We’re currently worse than the target (0.42541 vs 0.36638; lower is better), so we should cautiously improve score with minimal changes while keeping the same “no real model” baseline approach. The biggest easy win is to optimize the single post-scaling factor: your hard-coded `/= 1.14` is likely miscalibrated for the per-position-mean baseline, so we learn the best global scale on the training data (SN_filter==1) by minimizing MCRMSE (same metric family) via a tiny grid search. This preserves your core logic (position-wise means, then one global scaling) but chooses the scaling that better matches the label magnitudes, which should move the score down toward the target. Everything else (paths, submission alignment, output format) stays the same and we still write `submission.csv`.'
- What this solution (achieved 0.42165) has done: 'Your current approach (position-wise means + single global scale) is already stable and close to the target, so I keep the core logic unchanged and only make a minimal calibration improvement that tends to reduce MCRMSE. Specifically, instead of fitting the global scale on all training rows, I fit it on a small internal validation split (still SN_filter==1) to avoid a slight “train-fit” bias that can miscalibrate the scale and hurt leaderboard generalization. I also compute the optimal scale analytically (least-squares) for the scored targets, which is equivalent to a finer-than-grid search and typically gives a small but consistent improvement without changing the model. All paths, submission alignment, and output format remain identical, and it still writes `submission.csv`.'
- What this solution (achieved 0.4843) has done: 'You’re currently worse than the target (0.42165 vs 0.36638; lower is better), so we should make the smallest legitimate change that’s likely to reduce MCRMSE without changing the baseline “per-position mean + global scaling” core logic. The biggest low-risk improvement is to fit the single global scale using all 3 scored targets and weighting each training sample/position by the provided measurement errors (inverse-variance), which better matches the metric by down-weighting noisy labels. This keeps the same prediction shape (position-wise means broadcasted) and still applies just one scalar calibration, but typically moves the score downward. Submission alignment/format and paths remain unchanged, and it still writes a valid `submission.csv`.'
- What this solution (achieved 0.4843) has done: 'Your score (0.4843) is worse than the target (0.36638; lower is better), so we should make a small, low-risk improvement without changing the “per-position mean + global scalar calibration” core logic. The current code computes position-wise means using all 2160 train sequences, including low-quality ones (SN_filter==0), which tends to hurt this competition; we restrict the mean computation to SN_filter==1 (higher-quality) while keeping the rest identical. We still fit the same single global scale on a held-out validation split using the same weighted least-squares formula and apply it to the submission. This is a minimal change that usually reduces MCRMSE and should move you toward the target.'
- What this solution (achieved 0.42181) has done: 'We keep your core baseline unchanged (position-wise means from train, broadcast to test, then one global scalar calibration) but remove a likely regression: the weighted least-squares scale is currently optimized for weighted MSE, while you report/optimize an *unweighted* MCRMSE, so the chosen scale can easily be miscalibrated for the leaderboard metric. With minimal change, we compute the global scale by directly minimizing the *unweighted* MCRMSE on the same validation split via a small deterministic grid search around 1.0, and then apply that single scale to the submission. Everything else (SN_filter==1 restriction, per-position means, paths, submission format) stays the same and still writes a valid `submission.csv`.'
- What this solution (achieved 0.42181) has done: 'You’re currently above the target (0.42181 vs 0.36638; lower is better), so we should make a small, legitimate improvement without changing the baseline logic (position-wise means + single global scalar calibration). The main issue is that your scale is tuned by minimizing MCRMSE on a single random holdout, which is noisy and can pick a suboptimal scale for the leaderboard; we keep the exact same calibration idea but choose the scale using out-of-fold (K-fold) validation for a more stable estimate. This is still “one global scale” and uses the same metric, but reduces variance and typically lowers MCRMSE. Everything else (data loading, per-position mean construction, submission alignment/format, writing `submission.csv`) stays the same.'
- What this solution (achieved 0.42967) has done: 'We’re currently worse than the target (0.42181 vs 0.36638; lower is better), so we should make the smallest change that legitimately improves generalization while preserving your baseline (per-position means + one global scalar). The biggest low-risk issue is that the per-position means are computed on raw values, but this competition’s standard preprocessing clips extreme negative/positive measurements; adding a light clip to both the training targets (when computing means and calibration) and the final predictions typically reduces MCRMSE without changing the modeling approach. I keep your K-fold scale search intact, but apply the same clipping consistently in the CV objective so the chosen scale matches the post-processed predictions. All paths, alignment with `sample_submission.csv`, output columns, and writing `submission.csv` stay unchanged.'
- What this solution (achieved 0.42967) has done: 'You’re currently worse than the target (0.42967 vs 0.36638; lower is better), so we make the smallest calibration change that tends to reduce MCRMSE without changing your baseline logic (position-wise means + one global scale + clipping). The key fix is that your CV scale selection is computed on unpermuted arrays while the validation indices come from a permutation, which makes the CV objective inconsistent and can pick a poor scale. I apply the same permutation to `y_all_true` and `y_all_base` before doing K-fold selection so the scale is actually optimized on the intended folds. Everything else (data loading, SN_filter==1 restriction, per-position mean construction, clipping bounds, output format, writing `submission.csv`) stays unchanged.'
- What this solution (achieved 0.42193) has done: 'Your current score (0.42967; lower is better) is still worse than the target (0.36638), so we make the smallest legitimate improvement that keeps your core baseline intact (position-wise means + one global scale + clipping). The most likely issue now is miscalibration from using a generic clip upper bound (1.5) and filling unscored positions with the last-scored mean; both can inflate error and don’t reflect the train distribution. I (1) set the clip bounds from the training data itself (robust percentiles) and use the same bounds everywhere, and (2) for positions beyond `seq_scored`, fill with the mean of the last few scored positions (smoother tail) instead of just the last position. Everything else (data loading, SN_filter==1 restriction, CV scale selection, submission alignment/format, writing `submission.csv`) stays the same.'
- What this solution (achieved 0.42156) has done: 'We’re currently worse than the target (0.42193 vs 0.36638; lower is better), so we should make the smallest legitimate improvement without changing your core baseline (per-position means + one global scale + clipping + tail fill). The biggest easy gain is to compute the per-position means only from the *scored* positions and then calibrate the *global scale* against the same per-position baseline under the exact MCRMSE definition (flatten all positions/targets), because the current `mcrmse()` averages over the wrong axis and can pick a suboptimal scale. I fix `mcrmse()` to match the competition definition and run the same 5-fold grid search, keeping everything else identical (same features, same scale-only calibration, same clip bounds, same submission alignment). This should move the score downward toward the target while remaining stable and within Kaggle constraints.'

# 9. Code solution

## === cell 0
import os
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)



## === cell 1
base_candidates = [
    "../input/stanford-covid-vaccine",
    "../input",
    "/kaggle/input/stanford-covid-vaccine",
    "/kaggle/input",
    "/kaggle/data/stanford-covid-vaccine",
    "/kaggle/data",
]
sample_path = None
train_path = None

for base in base_candidates:
    p = os.path.join(base, "sample_submission.csv")
    if os.path.exists(p):
        sample_path = p
        break

for base in base_candidates:
    p = os.path.join(base, "train.json")
    if os.path.exists(p):
        train_path = p
        break

if sample_path is None:
    raise FileNotFoundError(
        "Could not find sample_submission.csv in expected Kaggle input locations."
    )
if train_path is None:
    raise FileNotFoundError(
        "Could not find train.json in expected Kaggle input locations."
    )

df = pd.read_csv(sample_path)

target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
missing = [c for c in target_cols if c not in df.columns]
if missing:
    raise ValueError(f"sample_submission.csv missing expected columns: {missing}")

train = pd.read_json(train_path, lines=True)

train_hq = train
if "SN_filter" in train_hq.columns:
    train_hq = train_hq[train_hq["SN_filter"] == 1].reset_index(drop=True)

seq_scored = int(train_hq["seq_scored"].iloc[0])  # expected 68

stack_all = []
for c in target_cols:
    stack_all.append(np.stack(train_hq[c].values).astype(np.float32).reshape(-1))
stack_all = np.concatenate(stack_all, axis=0)
CLIP_LO = float(np.percentile(stack_all, 0.5))
CLIP_HI = float(np.percentile(stack_all, 99.5))
if not np.isfinite(CLIP_LO):
    CLIP_LO = -0.5
if not np.isfinite(CLIP_HI):
    CLIP_HI = 1.5
if CLIP_HI <= CLIP_LO:
    CLIP_LO, CLIP_HI = -0.5, 1.5

pos_means = {}
for c in target_cols:
    arr = np.stack(train_hq[c].values).astype(np.float32)  # (n_samples, 68)
    arr = np.clip(arr, CLIP_LO, CLIP_HI)
    pos_means[c] = arr.mean(axis=0)  # (68,)

seqpos = df["id_seqpos"].str.rsplit("_", n=1).str[-1].astype(int).values

TAIL_K = 8  # unchanged baseline behavior
for c in target_cols:
    mu = pos_means[c]
    tail_val = float(np.mean(mu[max(0, len(mu) - TAIL_K) :]))
    preds = np.empty(len(df), dtype=np.float32)
    in_scored = seqpos < seq_scored
    preds[in_scored] = mu[seqpos[in_scored]]
    preds[~in_scored] = tail_val
    df[c] = preds



## === cell 2
sequences = list(set(["_".join(v.split("_")[:-1]) for v in df.id_seqpos.values]))
sequences[-10:]




## === cell 3
def mcrmse_competition(y_true, y_pred, eps=1e-12):
    diff2 = (y_true - y_pred) ** 2
    mse_per_target = np.mean(diff2, axis=(0, 1))  # (n_targets,)
    rmse_per_target = np.sqrt(mse_per_target + eps)
    return float(np.mean(rmse_per_target))


scored_cols = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]  # only these are scored
train_cal = train_hq.reset_index(drop=True)

rng = np.random.RandomState(0)
n = len(train_cal)
perm = rng.permutation(n)

K = 5
fold_sizes = np.full(K, n // K, dtype=int)
fold_sizes[: n % K] += 1
fold_starts = np.concatenate([[0], np.cumsum(fold_sizes)[:-1]])
fold_ends = np.cumsum(fold_sizes)

y_all_true = np.stack(
    [np.stack(train_cal[c].values) for c in scored_cols], axis=-1
).astype(
    np.float32
)  # (n, 68, 3)
y_all_true = np.clip(y_all_true, CLIP_LO, CLIP_HI)

y_all_base = np.stack(
    [np.broadcast_to(pos_means[c], (n, seq_scored)) for c in scored_cols], axis=-1
).astype(
    np.float32
)  # (n, 68, 3)

y_all_true = y_all_true[perm]
y_all_base = y_all_base[perm]

scales = np.linspace(0.60, 1.40, 161, dtype=np.float32)
cv_scores = np.zeros_like(scales, dtype=np.float32)

for k in range(K):
    va_pos = np.arange(fold_starts[k], fold_ends[k])  # positions within permuted arrays
    y_va_true = y_all_true[va_pos]
    y_va_base = y_all_base[va_pos]
    for i, s in enumerate(scales):
        y_pred = np.clip(y_va_base * float(s), CLIP_LO, CLIP_HI)
        cv_scores[i] += mcrmse_competition(y_va_true, y_pred)

cv_scores /= float(K)
best_i = int(np.argmin(cv_scores))
best_scale = float(scales[best_i])
cv_score = float(cv_scores[best_i])

df[target_cols] = df[target_cols].astype(np.float32) * best_scale
df[target_cols] = np.clip(df[target_cols].values.astype(np.float32), CLIP_LO, CLIP_HI)

print(
    f"Chosen global scale={best_scale:.6f} by {K}-fold CV MCRMSE grid-search; "
    f"CV MCRMSE={cv_score:.6f} (n={n}); clip=[{CLIP_LO:.6f},{CLIP_HI:.6f}]"
)



## === cell 4
assert df.shape[1] == 6, f"Expected 6 columns, got {df.shape[1]}"
assert (
    df.columns.tolist() == ["id_seqpos"] + target_cols
), f"Unexpected columns: {df.columns.tolist()}"
assert df["id_seqpos"].isna().sum() == 0, "Found NaNs in id_seqpos"
assert df[target_cols].isna().sum().sum() == 0, "Found NaNs in predictions"



## === cell 5
df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", df.shape)
print(df.head())
