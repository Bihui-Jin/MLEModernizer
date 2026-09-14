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

0.3519914239007519

# 6. Current score

0.4226

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.63824) has done: 'I remove the dependency on the missing `../input/worst-submission/ensemble52.csv` file by instead loading the provided `sample_submission.csv` from the competition dataset path that exists in your environment. To keep the “core logic” (editing a subset of rows and then saving) intact while making the pipeline valid end-to-end, I preserve the same downstream operations but ensure `df` is defined and has the required columns. I also make the id handling deterministic (no `set()` order dependence) and add a small sanity check that the submission columns match the required format before writing `submission.csv`. This produce a valid submission file; since no model is trained here, score targeting is not meaningfully adjustable beyond producing a correct-format file.'
- What this solution (achieved 0.42418) has done: 'You’re currently submitting essentially the sample submission (all zeros except a tiny tweak), which explains the high (worse) MCRMSE; to move toward the much lower target score, we need predictions that reflect training signal while staying minimal and fast. I keep the “no deep model” core approach but replace the zero-fill with a simple, legitimate baseline: per-position mean targets computed from `train.json`, then copy those means onto every test `id_seqpos` by `seqpos`. This preserves evaluation semantics and produces a valid submission with correct row alignment to `sample_submission.csv`. I also keep the existing special-case row edit (it becomes a no-op improvement-wise but preserves your prior logic pattern) and add a small alignment assert to ensure we never mis-order rows.'
- What this solution (achieved 0.42526) has done: 'To move your MCRMSE down toward the target with minimal disruption, I keep the same “per-seqpos baseline from train.json applied to sample_submission rows” core logic, but improve it in two small, metric-relevant ways. First, I compute the per-position mean using only high-quality rows (`SN_filter==1`) and also compute per-position standard deviation; then I apply a light, per-target, per-position shrinkage toward the global mean to reduce noise amplification (this usually improves RMSE without changing the modeling approach). Second, I clip predictions to a reasonable range derived from training quantiles to avoid extreme values that can hurt RMSE. The submission writing, row alignment, and required column handling remain unchanged.'
- What this solution (achieved 0.4246) has done: 'Your current baseline is already close to the target but still worse (higher MCRMSE) than desired, so the smallest likely improvement is to use a more appropriate weighting of training rows without changing the “per-seqpos mean applied to test” core logic. I compute per-position means using *all* training rows but weighted by `signal_to_noise` (and still respecting `SN_filter` as a mild upweight rather than a hard filter), which usually reduces RMSE versus an unweighted mean. I also make the shrinkage strength data-driven but gentler (closer to your existing approach) and keep clipping, submission ordering, and the special-case row edit intact. This should move the score down toward the 0.352 target without introducing a new model or changing evaluation semantics.'
- What this solution (achieved 0.42457) has done: 'Your current score (0.4246, lower-is-better) is still worse than the target (0.35199), so we should make a small, low-risk improvement without changing the overall “per-seqpos baseline from train applied to test” approach. I keep your weighted-per-position mean + shrinkage + clipping core logic, but (1) compute *weighted* per-position means and (optionally) a weighted global mean more consistently, and (2) compute per-position dispersion using the same weights (instead of unweighted std) so the shrinkage strength is better aligned with the metric. I also change clipping bounds to be computed on the *scored* positions only (first 68 already, but explicitly) and use slightly less aggressive quantiles to avoid over-clipping toward the mean. All I/O, submission ordering checks, and the special-case edit remain intact.'
- What this solution (achieved 0.42818) has done: 'We keep your exact “per-seqpos baseline from train applied to sample_submission rows” approach, but make two minimal, metric-aligned tweaks that usually reduce RMSE without changing the core logic: (1) compute per-position means using a robust weighted Huber IRLS step on residuals (still just a weighted mean, but less sensitive to outlier sequences), and (2) make clipping bounds weighted-quantile based (instead of unweighted) so they better reflect the same weighting used for the mean. Everything else (paths, columns, shrinkage, row ordering checks, and the special-case id edit) remains intact. This should move your score downward (better) toward the 0.35199 target without introducing a new model or changing evaluation semantics.'
- What this solution (achieved 0.42362) has done: 'We keep your exact per-seqpos weighted baseline + shrinkage + clipping pipeline, but remove the extra robust Huber IRLS step that likely over-smooths/perturbs the weighted means and has recently made the score worse (0.42818). Then we make the shrinkage slightly gentler (closer to raw per-position means) while keeping the same functional form, which is a minimal, metric-aligned way to reduce bias and typically lowers RMSE for this competition. Finally, we keep all I/O, ordering checks, and the special-case `id_dadd4b985` edit intact so the script still produces a valid `submission.csv` with the correct schema and alignment.'
- What this solution (achieved 0.42358) has done: 'Your current score (0.42362, lower-is-better) is still worse than the target (0.35199), so we should make a very small, low-risk improvement without changing the core “per-seqpos weighted mean from train applied to sample_submission rows” approach. The main tweak is to compute the per-position mean and variance after winsorizing the training targets per position (weighted 0.5%/99.5% bounds), which reduces the influence of extreme/noisy training labels and typically lowers RMSE for this competition while keeping the same baseline logic. I keep your existing shrinkage and output clipping, but I align the clip bounds to the same (slightly less aggressive) weighted quantiles used for winsorization to avoid rare extreme predictions. All I/O paths, row ordering, columns, and the special-case `id_dadd4b985` edit remain intact, and it still writes a valid `submission.csv`.'
- What this solution (achieved 0.4226) has done: 'We keep your exact “per-seqpos weighted mean from train applied to sample_submission rows” pipeline, but make the smallest metric-aligned adjustment that’s likely to reduce MCRMSE: tune the shrinkage to be a bit *less aggressive on average* (closer to the per-position means) and make it depend slightly on estimated per-position noise in a smoother way. This aims to reduce bias from over-shrinking toward the global mean, which can hurt in this competition when real per-position patterns matter. Everything else (winsorized weighted means, weighted variance, weighted quantile clipping, submission ordering/format, and the special-case edit) stays intact and the script still writes a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)



## === cell 1
CANDIDATE_PATHS = [
    "/kaggle/input/stanford-covid-vaccine/sample_submission.csv",
    "/kaggle/input/sample_submission.csv",
    "/kaggle/data/stanford-covid-vaccine/sample_submission.csv",
    "/kaggle/data/sample_submission.csv",
    "../input/stanford-covid-vaccine/sample_submission.csv",
    "../input/sample_submission.csv",
]
sub_path = next((p for p in CANDIDATE_PATHS if os.path.exists(p)), None)
if sub_path is None:
    raise FileNotFoundError(
        "Could not find sample_submission.csv in expected Kaggle input/data locations. "
        f"Tried: {CANDIDATE_PATHS}"
    )

df = pd.read_csv(sub_path)
df.head()



## === cell 2
sequences = (
    pd.Series(df["id_seqpos"].astype(str)).str.rsplit("_", n=1).str[0].unique().tolist()
)
sequences[-10:]



## === cell 3
TRAIN_CANDIDATE_PATHS = [
    "/kaggle/input/stanford-covid-vaccine/train.json",
    "/kaggle/input/train.json",
    "/kaggle/data/stanford-covid-vaccine/train.json",
    "/kaggle/data/train.json",
    "../input/stanford-covid-vaccine/train.json",
    "../input/train.json",
]
train_path = next((p for p in TRAIN_CANDIDATE_PATHS if os.path.exists(p)), None)
if train_path is None:
    raise FileNotFoundError(
        "Could not find train.json in expected Kaggle input/data locations. "
        f"Tried: {TRAIN_CANDIDATE_PATHS}"
    )

train = pd.read_json(train_path, lines=True)

target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
seq_scored = int(train["seq_scored"].iloc[0])  # 68

sn = (
    pd.to_numeric(train["signal_to_noise"], errors="coerce")
    .fillna(0.0)
    .astype(np.float32)
    .values
)
sn_filter = (
    pd.to_numeric(train["SN_filter"], errors="coerce").fillna(0).astype(int).values
)

w = np.clip(sn, 0.0, 5.0)  # cap extreme influence
w = (w + 0.5).astype(np.float32)  # avoid zero weights
w *= np.where(sn_filter == 1, 1.25, 1.0).astype(np.float32)

means = {}
global_means = {}
pos_stds = {}
clip_bounds = {}

eps = 1e-6


def _weighted_mean(arr_2d, weights_1d):
    wsum = float(weights_1d.sum()) + eps
    wn = (weights_1d / wsum).astype(np.float32)
    return (arr_2d * wn[:, None]).sum(axis=0).astype(np.float32)


def _weighted_quantile(x, weights, qs):
    x = x.astype(np.float64)
    wq = weights.astype(np.float64)
    m = np.isfinite(x) & np.isfinite(wq) & (wq > 0)
    x = x[m]
    wq = wq[m]
    if x.size == 0:
        return [0.0 for _ in qs]
    idx = np.argsort(x)
    x = x[idx]
    wq = wq[idx]
    cdf = np.cumsum(wq) / (wq.sum() + eps)
    return [float(x[np.searchsorted(cdf, q)]) for q in qs]


for col in target_cols:
    arr_raw = np.vstack(train[col].values).astype(np.float32)  # (n_samples, 68)

    lo_pos = np.zeros(seq_scored, dtype=np.float32)
    hi_pos = np.zeros(seq_scored, dtype=np.float32)
    for j in range(seq_scored):
        ql, qh = _weighted_quantile(
            arr_raw[:, j].astype(np.float32), w.astype(np.float32), [0.005, 0.995]
        )
        lo_pos[j] = np.float32(ql)
        hi_pos[j] = np.float32(qh)

    arr = np.clip(arr_raw[:, :seq_scored], lo_pos[None, :], hi_pos[None, :]).astype(
        np.float32
    )

    mu_pos = _weighted_mean(arr, w)

    means[col] = mu_pos.astype(np.float32)
    global_means[col] = float(mu_pos.mean())

    wsum = float(w.sum()) + eps
    wn = (w / wsum).astype(np.float32)
    var_pos = (wn[:, None] * (arr - mu_pos[None, :]) ** 2).sum(axis=0)
    pos_stds[col] = np.sqrt(np.maximum(var_pos, 0.0)).astype(np.float32)

    flat = arr.reshape(-1).astype(np.float32)
    w_flat = np.repeat(w.astype(np.float32), seq_scored)
    q_low, q_high = _weighted_quantile(flat, w_flat, [0.005, 0.995])
    clip_bounds[col] = (float(q_low), float(q_high))

seqpos = df["id_seqpos"].astype(str).str.rsplit("_", n=1).str[1].astype(int).values
seqpos_clip = np.clip(seqpos, 0, seq_scored - 1)

for col in target_cols:
    std = pos_stds[col]

    std_mean = float(std.mean()) + eps
    rel = (std / std_mean).astype(np.float32)

    alpha = 0.992 - 0.10 * rel  # gentler than previous (was 0.985 - 0.15 * rel)
    alpha = np.clip(alpha, 0.88, 0.995).astype(
        np.float32
    )  # slightly higher floor/ceiling than before

    base = means[col].astype(np.float32)
    gm = np.float32(global_means[col])
    shrunk = alpha * base + (1.0 - alpha) * gm  # (68,)

    pred = shrunk[seqpos_clip].astype(np.float32)
    lo, hi = clip_bounds[col]
    pred = np.clip(pred, lo, hi).astype(np.float32)
    df[col] = pred

assert len(df) == pd.read_csv(sub_path).shape[0], "Row count changed unexpectedly."



## === cell 4
mask = df["id_seqpos"].astype(str).str.startswith("id_dadd4b985")
if mask.any():
    df.loc[mask, "reactivity"] = 0.0



## === cell 5
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
    for c in missing:
        df[c] = 0.0

df = df[required_cols]

for c in required_cols[1:]:
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).astype(np.float32)

df.shape, df.columns.tolist()



## === cell 6
out_path = "submission.csv"
df.to_csv(out_path, index=False)

check = pd.read_csv(out_path, nrows=5)
assert (
    check.columns.tolist() == required_cols
), f"Submission columns mismatch: {check.columns.tolist()}"

template_ids = (
    pd.read_csv(sub_path, usecols=["id_seqpos"])["id_seqpos"].astype(str).values
)
out_ids = pd.read_csv(out_path, usecols=["id_seqpos"])["id_seqpos"].astype(str).values
assert np.array_equal(
    template_ids, out_ids
), "id_seqpos order mismatch vs sample_submission.csv."

out_path
