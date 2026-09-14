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

0.42197

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.63824) has done: 'I remove the dependency on the missing `../input/worst-submission/ensemble52.csv` file (the root cause of all downstream `NameError`s) and instead build predictions from the provided `sample_submission.csv`, which is guaranteed to exist. To keep the core “logic” of this script (a submission post-processing hack) intact while making it runnable, I preserve the same edit that forces `reactivity=1` for the specified `id` prefix. I also add a lightweight validation step to ensure the submission has the exact required columns/row count and is written as `submission.csv`. This produce a valid CSV submission end-to-end; since no model is trained here, score improvement beyond validity isn’t possible without introducing modeling (which would violate the “minimal changes/preserve core logic” constraint).'
- What this solution (achieved 0.63824) has done: 'Your current script is essentially a “submission edit hack” on top of `sample_submission.csv`, which caps performance around a weak baseline and explains the 0.638 score; to move toward the 0.3519 target (lower is better), we need a minimal-but-legitimate predictor while keeping the same overall workflow (read train, build per-position predictions, write `submission.csv`). The smallest change that improves MCRMSE without introducing new modeling frameworks is to replace the constant sample-submission values with per-position mean targets learned from the training set (a standard baseline for this competition), while preserving your existing special-case override (`id_c5e79dcb9*` reactivity=1.0). This keeps the “no deep model/training loops” core approach intact but uses real signal from `train.json`. We also keep strict alignment to `sample_submission.csv` row order by merging on `seqpos`, ensuring a valid submission format.'
- What this solution (achieved 0.42166) has done: 'We keep your current “position-wise mean from train.json” baseline (core logic) but fix a subtle bug: after the merge you are still using the original sample_submission columns (zeros) instead of the newly-merged mean columns, which caps performance. We also restrict the mean calculation to the high-quality subset (`SN_filter==1`) which is a minimal, legitimate data-cleaning step commonly used for this competition and tends to reduce MCRMSE. Finally, we extend predictions from 68 scored positions to all 107 positions by carrying forward the last available mean (pos 67) so the non-scored rows are reasonable while preserving required submission shape and ordering.'
- What this solution (achieved 0.48245) has done: 'We keep your position-wise mean baseline intact, but make two minimal adjustments that typically reduce MCRMSE for this competition: (1) compute the per-position means using signal-to-noise as sample weights (using the provided `*_error_*` columns) rather than a plain mean, and (2) align the training objective with the metric by only learning the three scored targets and filling the two unscored targets with their own (unweighted) position means to keep the submission well-formed. This preserves the same end-to-end approach (read train → aggregate by seqpos → merge into sample submission → write CSV) and doesn’t introduce any new model/training loop. We also keep your `SN_filter==1` restriction and your special-case override for `id_c5e79dcb9*` reactivity. The output remains a valid `submission.csv` with the exact required columns and row count.'
- What this solution (achieved 0.47944) has done: 'We keep your exact “position-wise aggregate from train.json → merge into sample_submission → write submission.csv” pipeline, but fix two small issues that can push MCRMSE down from 0.482 toward your 0.3519 target. First, we replace the current `1/(error^2)` weighting (which can explode for tiny errors and overfit noise) with the competition-standard inverse-variance weighting **clipped to robust percentiles**, preserving the same core idea while stabilizing the estimate. Second, we compute means using **only the scored positions (0..67)** and then fill the unscored positions (68..106) with the last scored-position value as you already do, but we also ensure the merge always prefers the newly-merged columns (not any `_sample` remnants) and that seqpos extraction is strictly correct. These are minimal changes aimed at legitimately improving the per-position baseline without introducing a model or changing submission semantics.'
- What this solution (achieved 0.51504) has done: 'We keep your exact “train.json → position-wise aggregate → merge into sample_submission → write submission.csv” baseline, but make one targeted change to reduce MCRMSE: compute the per-position aggregates using a robust **weighted median** (with the same clipped inverse-variance weights you already use) for the three scored targets, instead of a weighted mean that is still sensitive to remaining outliers. This preserves the same features, targets, and overall flow, and stays within a small code change while typically improving this competition’s metric. We also ensure the merge always prefers the newly-computed columns (even when `_sample` columns exist), preventing accidental fallback to the sample values. The special-case override and submission validation/writing remain unchanged.'
- What this solution (achieved 0.47944) has done: 'We move back from the weighted-median aggregation (which likely over-robustifies and hurt your score) to a stabilized clipped inverse-variance **weighted mean** per position for the three scored targets, keeping the exact same pipeline (train.json → per-position aggregate → merge into sample_submission → write submission.csv). This is a minimal change that better matches MCRMSE’s squared-error nature (mean tends to outperform median under RMSE), while preserving your SN_filter usage, seqpos handling, and the “fill positions 68..106 with pos 67” logic. We also keep the two unscored targets as simple per-position means as you already do, and preserve your special-case override for `id_c5e79dcb9*`. No model/training loop or feature/architecture changes are introduced—only the aggregation statistic changes to legitimately reduce error toward your target.'
- What this solution (achieved 0.4804) has done: 'We keep your exact “train.json → per-position aggregate → merge into sample_submission → write submission.csv” pipeline, but make one minimal, competition-relevant adjustment to move the MCRMSE down toward the 0.3519 target. Specifically, we compute the scored-target aggregates using the provided per-position experimental errors as inverse-variance weights (as you do), but additionally (a) drop positions with clearly unreliable measurements (very large errors) via a gentle percentile-based filter and (b) clip extreme target values before averaging to reduce the impact of outliers that RMSE heavily penalizes. This preserves your core logic (still a per-position weighted mean baseline, no model/training loop), but typically improves this competition metric materially versus using all points. Submission alignment, the `id_c5e79dcb9*` override, and the 68→107 fill behavior remain unchanged.'
- What this solution (achieved 0.48024) has done: 'We keep your exact “train.json → position-wise aggregate → merge into sample_submission → write submission.csv” pipeline, but make one targeted adjustment to reduce MCRMSE from 0.4804 toward the 0.3519 target: compute the per-position weighted means using an additional, gentle **S/N-based sample weight** (so high-quality sequences contribute a bit more) while keeping your existing inverse-variance weighting and clipping/filters. This is a minimal change that does not introduce any new model/loop and is aligned with the competition’s known data-quality issues. We also make the groupby aggregation explicitly use `include_groups=False` (pandas 2.2 compatibility) to avoid subtle behavior changes and ensure deterministic output. Everything else (SN_filter usage, 68→107 fill, id override, submission validation) remains intact.'
- What this solution (achieved 0.42197) has done: 'Your current pipeline is already a strong “train.json → per-position aggregate → merge into sample submission” baseline, but the weighting/clipping stack is likely overcomplicating and hurting MCRMSE versus a simpler, more metric-aligned estimate. To move your score down (lower is better) toward the 0.3519 target with minimal core-logic changes, I (1) switch the scored-target aggregation from inverse-variance weighting to a plain per-position mean computed on the high-quality subset (SN_filter==1) with a light value clipping, and (2) also compute unscored targets on the same subset for consistency. Everything else stays the same: same long-format expansion, same 68→107 fill with position 67, same merge/alignment to sample_submission order, same special-case `id_c5e79dcb9*` override, and the script still writes a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd



## === cell 1
train_path = "../input/stanford-covid-vaccine/train.json"
if not os.path.exists(train_path):
    train_path = "../input/train.json"

df_train = pd.read_json(train_path, lines=True)



## === cell 2
sample_sub_path = "../input/stanford-covid-vaccine/sample_submission.csv"
if not os.path.exists(sample_sub_path):
    sample_sub_path = "../input/sample_submission.csv"

df = pd.read_csv(sample_sub_path)



## === cell 3
targets = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
scored_targets = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
unscored_targets = ["deg_pH10", "deg_50C"]

seq_scored = int(df_train["seq_scored"].iloc[0])  # 68
seq_length = int(df_train["seq_length"].iloc[0])  # 107

if "SN_filter" in df_train.columns:
    df_train_use = df_train[df_train["SN_filter"] == 1].copy()
    if df_train_use.shape[0] == 0:
        df_train_use = df_train
else:
    df_train_use = df_train

rows = []
for _, r in df_train_use.iterrows():
    sn = (
        float(r["signal_to_noise"])
        if "signal_to_noise" in r and pd.notna(r["signal_to_noise"])
        else 1.0
    )
    for pos in range(seq_scored):
        rows.append(
            {
                "seqpos": pos,
                "reactivity": float(r["reactivity"][pos]),
                "deg_Mg_pH10": float(r["deg_Mg_pH10"][pos]),
                "deg_pH10": float(r["deg_pH10"][pos]),
                "deg_Mg_50C": float(r["deg_Mg_50C"][pos]),
                "deg_50C": float(r["deg_50C"][pos]),
                "reactivity_error": float(r["reactivity_error"][pos]),
                "deg_error_Mg_pH10": float(r["deg_error_Mg_pH10"][pos]),
                "deg_error_pH10": float(r["deg_error_pH10"][pos]),
                "deg_error_Mg_50C": float(r["deg_error_Mg_50C"][pos]),
                "deg_error_50C": float(r["deg_error_50C"][pos]),
                "signal_to_noise": sn,
            }
        )
train_long = pd.DataFrame(rows)


def _clipped_mean_by_pos(
    df_long: pd.DataFrame,
    value_col: str,
    clip_val_low_q: float = 0.01,
    clip_val_high_q: float = 0.99,
) -> pd.Series:
    tmp = pd.DataFrame(
        {
            "seqpos": df_long["seqpos"],
            "val": pd.to_numeric(df_long[value_col], errors="coerce"),
        }
    ).dropna(subset=["seqpos", "val"])
    if tmp.shape[0] == 0:
        return pd.Series(index=np.arange(seq_scored), data=0.0, name=value_col)

    v_lo = float(tmp["val"].quantile(clip_val_low_q))
    v_hi = float(tmp["val"].quantile(clip_val_high_q))
    if np.isfinite(v_lo) and np.isfinite(v_hi) and v_lo < v_hi:
        tmp["val"] = tmp["val"].clip(lower=v_lo, upper=v_hi)

    out = tmp.groupby("seqpos", sort=True)["val"].mean().rename(value_col)
    return out


pos_scored = pd.concat(
    [
        _clipped_mean_by_pos(train_long, "reactivity"),
        _clipped_mean_by_pos(train_long, "deg_Mg_pH10"),
        _clipped_mean_by_pos(train_long, "deg_Mg_50C"),
    ],
    axis=1,
)

pos_unscored = train_long.groupby("seqpos", sort=True)[unscored_targets].mean()

pos_means = (
    pd.concat([pos_scored, pos_unscored], axis=1)
    .reset_index()
    .rename(columns={"index": "seqpos"})
)

if seq_length > seq_scored:
    last_vals = pos_means.loc[pos_means["seqpos"] == (seq_scored - 1), targets].iloc[0]
    extra = pd.DataFrame({"seqpos": np.arange(seq_scored, seq_length, dtype=int)})
    for t in targets:
        extra[t] = float(last_vals[t])
    pos_means_full = pd.concat([pos_means, extra], ignore_index=True)
else:
    pos_means_full = pos_means

sub = df.copy()
sub["seqpos"] = sub["id_seqpos"].astype(str).str.extract(r"_(\d+)$")[0].astype(int)
sub = sub.merge(pos_means_full, on="seqpos", how="left", suffixes=("_sample", ""))

for t in targets:
    if f"{t}_sample" in sub.columns and t in sub.columns:
        sub[t] = sub[t].where(sub[t].notna(), sub[f"{t}_sample"])
    elif t not in sub.columns and f"{t}_sample" in sub.columns:
        sub[t] = sub[f"{t}_sample"]

sub[targets] = sub[targets].apply(pd.to_numeric, errors="coerce").fillna(0.0)
df = sub[["id_seqpos"] + targets].copy()



## === cell 4
df.loc[df.id_seqpos.astype(str).str.startswith("id_c5e79dcb9"), "reactivity"] = 1.0



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
    raise ValueError(f"Submission is missing required columns: {missing}")

df = df[required_cols]

for c in required_cols[1:]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df[required_cols[1:]] = df[required_cols[1:]].fillna(0.0)

if df.shape[0] != pd.read_csv(sample_sub_path).shape[0]:
    raise ValueError(f"Row count mismatch vs sample_submission: {df.shape[0]}")



## === cell 6
out_path = "submission.csv"
df.to_csv(out_path, index=False)

print("Wrote:", out_path)
print("Shape:", df.shape)
print(df.head())
