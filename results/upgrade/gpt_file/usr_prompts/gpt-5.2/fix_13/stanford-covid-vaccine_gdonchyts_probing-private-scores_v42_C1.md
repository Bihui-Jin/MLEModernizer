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

0.42682

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.63824) has done: 'I remove the dependency on the missing `../input/worst-submission/ensemble52.csv` file and instead build a valid submission directly from the provided `sample_submission.csv`, ensuring the correct row count and columns. I keep the existing “edit a specific id’s reactivity” core behavior, but guard it so it won’t crash if the id isn’t present. I also fix the cell numbering/order so the notebook runs end-to-end and always writes `submission.csv`. This yield a valid (though not necessarily strong) baseline submission rather than failing before output.'
- What this solution (achieved 0.63824) has done: 'Your current submission is essentially the sample submission with one id’s `reactivity` set to a huge constant, which severely hurts MCRMSE; the smallest change that moves you toward the (lower-is-better) target is to stop injecting that large error. I keep the same pipeline (read sample_submission → optionally edit specific id → enforce columns/numerics → write submission.csv) but change the edit to a harmless no-op by setting that id’s `reactivity` back to the baseline value it already has (0.0 in the provided sample). This preserves your “edit a specific id’s reactivity” core behavior while reducing avoidable error and should improve the score toward your target. The script still run end-to-end and always write a valid `submission.csv` with the correct schema and row count.'
- What this solution (achieved 0.42418) has done: 'You’re currently submitting (almost exactly) the sample submission, which is far from the target; the smallest legitimate improvement (without changing your overall “use sample_submission as baseline” core logic) is to replace the constant-0 predictions with simple per-position means learned from the training set. This keeps your pipeline structure intact (read train/test/sample → build a submission DataFrame → write submission.csv), but makes predictions data-driven and typically yields a large MCRMSE drop for this competition. I compute mean target values by `seqpos` (0–67) for all 5 targets from `train.json`, then fill those into every test `id` for positions 0–67 while leaving >67 as zeros, matching evaluation semantics. I also keep your “edit a specific id’s reactivity” behavior as a final optional override (still harmless), and ensure the output stays aligned to `sample_submission.csv` rows/columns.'
- What this solution (achieved 0.42166) has done: 'You’re currently using simple per-seqpos means over all training rows; a minimal, score-improving step for this competition is to apply the dataset’s intended quality filter (`SN_filter==1`) and also drop low-signal samples, so the per-position means are estimated from cleaner targets. This keeps the exact same core approach (position-wise mean imputation into `sample_submission.csv`) while typically reducing noise and lowering MCRMSE toward your target. I also make the `seq_length != 130` test filtering logic a no-op safeguard (it’s irrelevant here) and keep your “edit a specific id’s reactivity” override harmless. The script still runs end-to-end and writes a valid `submission.csv` with the correct schema and row count.'
- What this solution (achieved 0.44369) has done: 'You’re currently using position-wise means from a filtered training subset; to move the (lower-is-better) score closer to the target, the smallest safe improvement is to make those means more robust to outliers/noisy measurements without changing the overall “per-seqpos aggregate → fill submission” logic. I keep the exact pipeline but switch the aggregate from mean to median (still per-position, still learned only from train), which typically lowers MCRMSE for this competition. I also ensure the filter fallback remains intact and add a deterministic safety check that the produced submission aligns 1:1 with `sample_submission.csv` rows. The script still run end-to-end and write a valid `submission.csv` with the required columns and row count.'
- What this solution (achieved 0.42196) has done: 'Your current score (0.44369, lower-is-better) is worse than the target (0.35188), so we need a small, legitimate improvement without changing the overall “per-seqpos aggregate from train → fill sample_submission” approach. The lowest-risk change is to keep using the same filtered training subset but switch from a pure median to a robust clipped-mean per position: clip each target at reasonable global quantiles computed from the filtered training set, then take the mean by `seqpos`. This typically reduces error versus median on this competition while still being robust to outliers, and it preserves the same prediction semantics (single aggregate per position applied to all test IDs). I also keep your existing safety checks and the harmless id override, and ensure the submission remains perfectly aligned to `sample_submission.csv`.'
- What this solution (achieved 0.42145) has done: 'We need to reduce your MCRMSE (lower-is-better) from 0.42196 toward 0.35188, while keeping the same “per-seqpos aggregate from filtered train → fill sample_submission” core logic. The smallest likely win without changing modeling semantics is to compute the per-position aggregate using only the *scored* target columns (3) for the scored positions, since the other two targets are not scored and can be safely set to 0.0 (or kept as-is) without affecting the metric; this also avoids any noise from estimating unscored targets. Additionally, we make the clipping slightly less aggressive (0.005–0.995) to reduce bias from over-clipping, which can hurt RMSE more than it helps here. Everything else (filters, long-format construction, merge back to sample_submission, row/column checks, writing `submission.csv`) stays the same.'
- What this solution (achieved 0.63824) has done: 'We need to reduce MCRMSE from 0.42145 toward 0.35188 (lower is better) while keeping your “per-seqpos aggregate from filtered train → fill sample_submission” logic unchanged. The smallest likely improvement is to (1) compute the aggregate only over scored positions but weight training samples by inverse-variance from the provided per-position error arrays, which better matches the metric by trusting high-quality measurements more, and (2) slightly tighten your quality filter by using a higher `signal_to_noise` cutoff (still using the same filter mechanism) to reduce noise in the aggregate. Everything else (data reading, long-format creation, clipping, merge back to sample_submission, unscored targets set to 0.0, harmless id override, and submission writing) stays the same and remains deterministic and valid.'
- What this solution (achieved 0.48284) has done: 'We fix a key bug that currently makes your per-position aggregates unused: after merging, you never copy the `_agg` columns into the target columns, so you’re effectively submitting mostly zeros (which explains the regression to 0.638). Keeping your exact “per-seqpos aggregate from filtered train → fill sample_submission” logic, we minimally assign the merged `*_agg` values into the scored target columns for scored positions (0–67), and keep >67 as 0.0. We also replace the slow/fragile `groupby.apply` weighted mean with an equivalent vectorized weighted aggregation (same semantics, just correct and deterministic) to stay within time and reduce risk. Everything else (filters, clipping, unscored targets set to 0.0, harmless id override, schema/row-count checks, writing `submission.csv`) stays the same.'
- What this solution (achieved 0.44455) has done: 'Your current approach (per-seqpos aggregate from filtered train → fill sample_submission) is sound but still underperforming the target, so the smallest likely improvement is to align the training filter with what the organizers recommend: keep only higher-quality training samples by requiring both `SN_filter==1` and `signal_to_noise>=1.0` (instead of 1.5, which can over-prune and add variance). Next, because the metric is unweighted RMSE, we should avoid overly aggressive inverse-variance weighting that can introduce bias; we keep the same aggregation logic but “soften” the weights by using `1/(err+eps)` instead of `1/(err^2+eps)`. Everything else (same targets, same per-position aggregation, same clipping, same submission assembly and schema checks) stays the same to preserve core logic and evaluation semantics while nudging the score downward toward your target.'
- What this solution (achieved 0.42145) has done: 'We need to reduce your MCRMSE (lower-is-better) from 0.44455 toward 0.35188, so we make a minimal improvement within the same “per-seqpos aggregate from filtered train → fill sample_submission” core logic. The smallest likely win is to stop using inverse-error weighting altogether (set weights to 1) because the competition metric is unweighted RMSE and error-based weights can bias estimates away from the unweighted optimum, especially after clipping. We keep your exact filtering, clipping, aggregation-by-seqpos, and submission assembly unchanged, only neutralizing the weighting and adding a tiny safety fallback to ensure no NaNs after merge. This should nudge the score downward without changing the overall approach or runtime.'
- What this solution (achieved 0.42682) has done: 'We keep your exact “per-seqpos aggregate from filtered train → fill sample_submission” approach, but improve the aggregate slightly to better match the unweighted RMSE metric: use the per-position **mean** without clipping (clipping can bias the mean and worsen RMSE when outliers aren’t extreme). We also expand the training filter minimally to include only sequences with non-negative minima across the 3 scored targets (a common-quality heuristic consistent with the competition’s filtering rationale) while retaining your existing SN/signal_to_noise filters and fallback. Everything else—data loading, long-format creation, merge into sample_submission, unscored targets set to 0.0, optional id override, and writing `submission.csv`—stays the same and deterministic.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd



## === cell 1
DATA_DIR = "../input/stanford-covid-vaccine"

train_path = f"{DATA_DIR}/train.json"
test_path = f"{DATA_DIR}/test.json"
sample_sub_path = f"{DATA_DIR}/sample_submission.csv"

df_train = pd.read_json(train_path, lines=True)
df_test = pd.read_json(test_path, lines=True)
df = pd.read_csv(sample_sub_path)

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
    raise ValueError(f"sample_submission.csv missing columns: {missing}")



## === cell 2
sequences = list(df_test["id"].astype(str).unique())
sequences.sort()



## === cell 3
sequences[-10:] if len(sequences) >= 10 else sequences



## === cell 4
all_targets = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
scored_targets = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
unscored_targets = ["deg_pH10", "deg_50C"]

seq_scored = int(df_train["seq_scored"].iloc[0])  # expected 68
if seq_scored <= 0:
    raise ValueError("seq_scored must be positive")

df_train_f = df_train.copy()

if "SN_filter" in df_train_f.columns:
    df_train_f = df_train_f[df_train_f["SN_filter"] == 1]

if "signal_to_noise" in df_train_f.columns:
    df_train_f = df_train_f[df_train_f["signal_to_noise"] >= 1.0]

train_ids = df_train_f["id"].astype(str).values
n_train = len(df_train_f)

train_long = pd.DataFrame(
    {
        "id": np.repeat(train_ids, seq_scored),
        "seqpos": np.tile(np.arange(seq_scored, dtype=int), n_train),
    }
)

for t in scored_targets:
    arr = np.asarray(df_train_f[t].tolist(), dtype=np.float32)  # (n_train, 68)
    if arr.ndim != 2 or arr.shape[1] != seq_scored:
        raise ValueError(
            f"Unexpected {t} shape {arr.shape}, expected (n_train, {seq_scored})"
        )
    train_long[t] = arr.reshape(-1)

mins = None
for t in scored_targets:
    m = train_long.groupby("id", sort=False)[t].min()
    mins = m if mins is None else np.minimum(mins, m)
good_ids = mins.index[(mins >= 0.0).to_numpy()]
if len(good_ids) > 0:
    train_long = train_long[train_long["id"].isin(good_ids)]

if len(train_long) == 0:
    train_ids = df_train["id"].astype(str).values
    n_train = len(df_train)
    train_long = pd.DataFrame(
        {
            "id": np.repeat(train_ids, seq_scored),
            "seqpos": np.tile(np.arange(seq_scored, dtype=int), n_train),
        }
    )
    for t in scored_targets:
        arr = np.asarray(df_train[t].tolist(), dtype=np.float32)
        train_long[t] = arr.reshape(-1)

pos_aggs = pd.DataFrame({"seqpos": np.arange(seq_scored, dtype=int)})
for t in scored_targets:
    y = train_long[t].to_numpy(np.float64)
    seqpos_arr = train_long["seqpos"].to_numpy(np.int64)
    num = np.bincount(seqpos_arr, weights=y, minlength=seq_scored)
    den = np.bincount(seqpos_arr, minlength=seq_scored).astype(np.float64)
    den = np.where(den == 0.0, 1.0, den)
    pos_aggs[t] = (num / den).astype(np.float32)

sub = df[required_cols].copy()
sub["seqpos"] = sub["id_seqpos"].astype(str).str.split("_").str[-1].astype(int)

sub = sub.merge(pos_aggs, on="seqpos", how="left", suffixes=("", "_agg"))

is_scored = sub["seqpos"] < seq_scored

for t in scored_targets:
    agg_col = f"{t}_agg"
    if agg_col in sub.columns:
        vals = sub.loc[is_scored, agg_col].astype(np.float32).fillna(0.0)
        sub.loc[is_scored, t] = vals
    sub.loc[~is_scored, t] = 0.0

for t in unscored_targets:
    sub[t] = 0.0

drop_cols = ["seqpos"] + [
    f"{t}_agg" for t in scored_targets if f"{t}_agg" in sub.columns
]
sub = sub.drop(columns=drop_cols)
sub[all_targets] = sub[all_targets].apply(pd.to_numeric, errors="coerce").fillna(0.0)

df = sub

if df.shape[0] != pd.read_csv(sample_sub_path).shape[0]:
    raise ValueError(
        "Row count mismatch vs sample_submission.csv; refusing to write invalid submission."
    )



## === cell 5
target_id_prefix = "id_ff2d18b94"
mask = df["id_seqpos"].astype(str).str.startswith(target_id_prefix)
if mask.any():
    df.loc[mask, "reactivity"] = 0.0



## === cell 6
df = df[required_cols].copy()
df[required_cols[1:]] = (
    df[required_cols[1:]].apply(pd.to_numeric, errors="coerce").fillna(0.0)
)



## === cell 7
df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", df.shape)
print(df.head())
