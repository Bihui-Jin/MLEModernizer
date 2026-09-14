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

0.3578912193294472

# 6. Current score

0.42581

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.42418) has done: 'I remove the dependency on the missing `../input/worst-submission/ensemble52.csv` and instead generate predictions directly from the provided competition files. To keep core logic minimal and stable while still getting a reasonable score, I implement a simple baseline that uses the per-position mean targets from the training set (a standard, legitimate approach for this competition) and applies them to every test sequence position. I also ensure the submission rows exactly match `sample_submission.csv` (`id_seqpos` order and required 5 target columns), and always write a `submission.csv` file.'
- What this solution (achieved 0.42166) has done: 'Your current baseline uses a single per-position mean for all training rows, but the competition score is computed only on the first `seq_scored` positions and the train set contains many low-quality/noisy rows that are not filtered like the test set. To move the score down toward your target with minimal logic change, we compute the same per-position means using only training rows that pass the provided `SN_filter==1`, which better matches the test distribution while preserving the same “position-mean template” approach. We also make padding for unscored positions use the mean of the last few scored positions (still constant beyond 68) to reduce edge artifacts without changing semantics. The submission format and row ordering remain exactly aligned to `sample_submission.csv`, and we still always write `submission.csv`.'
- What this solution (achieved 0.42166) has done: 'We keep your “per-position mean template” baseline intact, but make the means better match the test distribution by adding a second lightweight filter on `signal_to_noise` (the same concept used in the competition notes) in addition to `SN_filter==1`. This is a minimal change that typically reduces MCRMSE because it down-weights low-quality training rows without changing any modeling approach. We also add a tiny bit of robustness: if the filtered subset becomes too small, we fall back to the previous `SN_filter==1` subset (and then to all rows), preventing score regressions from over-filtering. Submission generation, ordering, columns, and padding logic remain the same, and the script still always writes a valid `submission.csv`.'
- What this solution (achieved 0.42172) has done: 'We keep your per-position mean “template” baseline exactly intact, but make the filtering slightly closer to the test distribution by tightening the `signal_to_noise` threshold from 1.0 to 1.5 (still a minimal, legitimate data-quality filter described in the competition notes). This should reduce noise in the computed per-position means and typically lowers MCRMSE, nudging your score down toward the 0.3579 target without changing any modeling logic. To avoid regressions from over-filtering, we keep the same robust fallback ladder (S/N+SN_filter → SN_filter → all rows). Submission row order, padding approach, and output schema remain identical, and we still always write `submission.csv`.'
- What this solution (achieved 0.42173) has done: 'We keep your per-position mean “template” baseline exactly the same, but adjust the data-quality filtering to better match the test-set distribution and reduce noise in the estimated per-position means (which should lower MCRMSE toward your 0.3579 target). Concretely, we (1) keep `SN_filter==1`, (2) raise the `signal_to_noise` threshold slightly (from 1.5 → 2.0), and (3) add a simple fallback ladder (2.0 → 1.5 → 1.0 → SN_filter-only → all rows) so we don’t over-filter and accidentally worsen the score. Submission formatting, ordering, padding logic, and output filename remain unchanged, and the script still writes a valid `submission.csv`.'
- What this solution (achieved 0.48217) has done: 'Your current approach is a per-position mean “template” baseline, and the main lever that stays within your constraints is improving how those per-position means are estimated from the training set. To move the score down toward the 0.3579 target (lower is better) with minimal change, I keep the exact same template/padding/submission logic but replace the plain mean with an error-weighted mean using the provided per-position experimental error columns (down-weighting noisier measurements). I also keep your existing SN_filter + signal_to_noise filtering and fallback ladder for stability, so we don’t risk regressions from an overly small subset. This should legitimately reduce MCRMSE because it reduces the influence of low-quality labels without changing any model architecture or training loop (there is none).'
- What this solution (achieved 0.42578) has done: 'You’re currently well above the target (0.48217 vs 0.35789, lower is better), so we need a small, legitimate improvement without changing the “per-position template” core logic. The biggest issue in your last change is that pure error-weighting can distort the per-position means (and be sensitive to error scale/outliers), which can worsen MCRMSE. I keep the same filtering/template/padding/submission pipeline, but switch to a *robust* error-weighted estimator: compute both plain mean and error-weighted mean per position, then blend them with a small fixed mixing factor to stabilize the template. This is minimal, keeps semantics intact, and should move the score back down toward the target.'
- What this solution (achieved 0.42578) has done: 'You’re currently worse than the target (0.42578 vs 0.35789, lower is better), so we make a small, legitimate change that should reduce MCRMSE without changing the core “per-position template” logic. The most impactful minimal fix is to compute the per-position template using only the *scored targets* (reactivity, deg_Mg_pH10, deg_Mg_50C) and then fill the two unscored columns in a stable way from those scored predictions (rather than trying to estimate them from noisy labels). This usually improves the scored metric because it removes noise from estimating unscored columns while keeping the same submission schema and per-position constant template approach. Everything else (filters, padding, row alignment to sample_submission, and writing `submission.csv`) remains the same.'
- What this solution (achieved 0.42518) has done: 'We keep your per-position “template” baseline intact, but make two small changes that typically lower MCRMSE for this competition without changing the approach: (1) compute the per-position template using a trimmed mean (winsorized) before averaging to reduce the influence of outliers/noisy labels, and (2) make the error-weight blending slightly more conservative so it can’t drift too far from the stable plain estimate. These are minimal, legitimate estimator tweaks (still “position-wise constant template from filtered training”) and should improve score versus your current 0.42578 while staying well within runtime limits. Submission row ordering, padding logic, filters/fallback ladder, and output schema/filename remain unchanged.'
- What this solution (achieved 0.42833) has done: 'We keep your exact “per-position template” baseline and the same filtering/fallback ladder, but make the template estimator slightly more metric-aligned and stable. Specifically, we (1) compute the per-position template with a *weighted winsorized mean* (winsorize first, then error-weight) to avoid the error-weighted mean being distorted by outliers, and (2) very slightly increase the blend toward the weighted estimate (alpha) since your earlier pure weighting was worse but a cautious increase should improve over the current conservative blend. These are minimal estimator tweaks (no new model, no new features, same submission/padding logic) and should reduce MCRMSE from 0.42518 toward your 0.35789 target. The script still runs end-to-end and writes a valid `submission.csv` with correct ordering/columns.'
- What this solution (achieved 0.42524) has done: 'Your current score (0.42833, lower-is-better) is still far above the target (0.35789), so we should make a small, legitimate estimator tweak while keeping the same per-position “template” approach and the same filtering/submission logic. The most minimal high-impact change here is to better align the per-position template with the evaluation metric by computing it on a per-position basis using a soft reliability weighting derived from the provided per-position errors, but in a *more stable* way than your current 1/e² scheme. Concretely, we switch the weight to 1/(e+floor)² with a small error-floor (robust against tiny/unstable errors), and we slightly increase the blend toward the weighted estimate (alpha) now that the weighting is stabilized. Everything else (winsorization, SN/S2N fallback ladder, padding, unscored-column fill, exact sample_submission row order, and writing `submission.csv`) remains unchanged.'
- What this solution (achieved 0.42581) has done: 'We keep your exact “per-position template” baseline and submission/padding logic, but adjust the estimator in a minimal, metric-relevant way to reduce noise in the template. Specifically, we slightly increase winsorization strength (trim more outliers) and make the stabilized error-weighting a bit more conservative by raising the error floor, which should prevent overweighting dubious low-error points that can worsen MCRMSE. Then we slightly reduce the blend alpha toward the weighted estimate to keep the template closer to the robust plain mean while still using reliability information. These are small knobs that preserve your core approach and should move the score down from ~0.425 toward the 0.358 target (lower is better) without changing any I/O paths or output format.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd



## === cell 1
CANDIDATE_BASES = [
    "/kaggle/input/stanford-covid-vaccine",
    "/kaggle/data/stanford-covid-vaccine",
    "/kaggle/input",
    "/kaggle/data",
]


def find_file(filename: str) -> str:
    for base in CANDIDATE_BASES:
        path = os.path.join(base, filename)
        if os.path.exists(path):
            return path
    for base in CANDIDATE_BASES:
        if os.path.exists(base):
            for root, _, files in os.walk(base):
                if filename in files:
                    return os.path.join(root, filename)
    raise FileNotFoundError(
        f"Could not find {filename} under any of: {CANDIDATE_BASES}"
    )


train_path = find_file("train.json")
test_path = find_file("test.json")
sample_sub_path = find_file("sample_submission.csv")

train_path, test_path, sample_sub_path



## === cell 2
train = pd.read_json(train_path, lines=True)
test = pd.read_json(test_path, lines=True)
sample_sub = pd.read_csv(sample_sub_path)

train.shape, test.shape, sample_sub.shape



## === cell 3
targets_all = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
targets_scored = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]

err_map = {
    "reactivity": "reactivity_error",
    "deg_Mg_pH10": "deg_error_Mg_pH10",
    "deg_pH10": "deg_error_pH10",
    "deg_Mg_50C": "deg_error_Mg_50C",
    "deg_50C": "deg_error_50C",
}

seq_length = int(test["seq_length"].iloc[0])
seq_scored = int(test["seq_scored"].iloc[0])

train_used = train
if "SN_filter" in train.columns:
    sn_ok = train["SN_filter"].astype(int) == 1
else:
    sn_ok = pd.Series(True, index=train.index)

if "signal_to_noise" in train.columns:
    s2n = pd.to_numeric(train["signal_to_noise"], errors="coerce")
else:
    s2n = pd.Series(np.nan, index=train.index)

S2N_CANDIDATES = [2.0, 1.5, 1.0]

picked = None
for thr in S2N_CANDIDATES:
    if "signal_to_noise" in train.columns:
        cand = train.loc[sn_ok & (s2n >= thr)].copy()
    else:
        cand = train.loc[sn_ok].copy()

    if len(cand) >= 200:
        train_used = cand
        picked = ("SN_filter & signal_to_noise >= ", thr, len(cand))
        break

if picked is None:
    cand2 = train.loc[sn_ok].copy()
    if len(cand2) > 0:
        train_used = cand2
        picked = ("SN_filter only", None, len(cand2))
    else:
        train_used = train
        picked = ("all rows", None, len(train))


def _stack_float32(series_of_lists) -> np.ndarray:
    return np.vstack(series_of_lists.values).astype(np.float32)


train_target_arrays = {}
train_error_arrays = {}
for t in targets_scored:
    train_target_arrays[t] = _stack_float32(train_used[t])  # (n_used, 68)
    err_col = err_map.get(t)
    if err_col is not None and err_col in train_used.columns:
        train_error_arrays[t] = _stack_float32(train_used[err_col])  # (n_used, 68)
    else:
        train_error_arrays[t] = None


def _winsorize_per_position(
    y: np.ndarray, lower_q: float = 0.10, upper_q: float = 0.90
) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    lo = np.nanquantile(y, lower_q, axis=0).astype(np.float32)
    hi = np.nanquantile(y, upper_q, axis=0).astype(np.float32)
    return np.clip(y, lo[None, :], hi[None, :]).astype(np.float32)


def weighted_position_mean(y: np.ndarray, err: np.ndarray | None) -> np.ndarray:
    """
    Same core logic: a single per-position template from filtered training.

    Changes (minimal, score-relevant):
    - Use a slightly larger error floor inside weights to avoid over-trusting tiny errors.
    - Slightly reduce alpha to keep closer to robust plain mean (stability).
    """
    y = np.asarray(y, dtype=np.float32)

    y_clip = _winsorize_per_position(y, lower_q=0.10, upper_q=0.90)
    plain = np.nanmean(y_clip, axis=0).astype(np.float32)

    if err is None:
        return plain

    e = np.asarray(err, dtype=np.float32)

    err_floor = 0.15  # was 0.10; more conservative weighting to reduce instability
    valid = np.isfinite(e) & (e > 0) & np.isfinite(y_clip)
    w = np.zeros_like(e, dtype=np.float32)
    w[valid] = 1.0 / np.square(e[valid] + err_floor)

    wsum = w.sum(axis=0)
    y_safe = np.nan_to_num(y_clip, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    ywsum = (w * y_safe).sum(axis=0)

    wmean = np.divide(ywsum, wsum, out=np.zeros_like(ywsum), where=wsum > 0).astype(
        np.float32
    )
    wmean = np.where(wsum > 0, wmean, plain).astype(np.float32)

    alpha = 0.25  # was 0.35; more stable blend tends to reduce MCRMSE in this baseline
    blended = (1.0 - alpha) * plain + alpha * wmean
    return blended.astype(np.float32)


train_position_template = {}
for t in targets_scored:
    train_position_template[t] = weighted_position_mean(
        train_target_arrays[t], train_error_arrays[t]
    )  # (68,)

full_template = {}
tail_k = 5
for t in targets_scored:
    m = train_position_template[t]  # (68,)
    pad_val = float(np.mean(m[max(0, seq_scored - tail_k) : seq_scored]))
    full = np.concatenate(
        [m[:seq_scored], np.full(seq_length - seq_scored, pad_val, dtype=np.float32)],
        axis=0,
    )
    full_template[t] = full  # (107,)

full_template["deg_pH10"] = np.clip(
    0.50 * full_template["deg_Mg_pH10"] + 0.50 * full_template["reactivity"],
    -5.0,
    5.0,
).astype(np.float32)
full_template["deg_50C"] = np.clip(
    0.50 * full_template["deg_Mg_50C"] + 0.50 * full_template["reactivity"],
    -5.0,
    5.0,
).astype(np.float32)

for t in targets_all:
    full_template[t] = np.nan_to_num(
        full_template[t], nan=0.0, posinf=0.0, neginf=0.0
    ).astype(np.float32)

{t: full_template[t].shape for t in targets_all}



## === cell 4
sub = sample_sub[["id_seqpos"]].copy()

seqpos = sub["id_seqpos"].str.rsplit("_", n=1, expand=True)[1].astype(int).values

for t in targets_all:
    sub[t] = full_template[t][seqpos]

expected_cols = ["id_seqpos"] + targets_all
assert list(sub.columns) == expected_cols, f"Bad columns: {sub.columns}"
assert len(sub) == len(sample_sub), "Row count mismatch vs sample_submission"
assert sub.isna().sum().sum() == 0, "NaNs in submission"

sub.head()



## === cell 5
sub.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", sub.shape)
print("Used train rows:", len(train_used), "of", len(train))
print("Picked subset rule:", picked)
print(
    "Filters available:",
    {
        "SN_filter": "SN_filter" in train.columns,
        "signal_to_noise": "signal_to_noise" in train.columns,
    },
)
print("S2N thresholds tried (if available):", S2N_CANDIDATES)
print(
    "Used error columns available (scored only):",
    {t: (err_map[t] in train_used.columns) for t in targets_scored},
)
print("Estimator: winsorize(10%-90%) then stabilized error-weight; blended with plain")
print("Weighting: w = 1/(error + err_floor)^2; err_floor:", 0.15)
print("Blend alpha (0=plain mean, 1=pure stabilized error-weighted):", 0.25)
print(
    "Unscored fill formulas: deg_pH10=0.5*deg_Mg_pH10+0.5*reactivity; deg_50C=0.5*deg_Mg_50C+0.5*reactivity"
)
print(sub.iloc[0].to_dict())
print(sub.iloc[-1].to_dict())
