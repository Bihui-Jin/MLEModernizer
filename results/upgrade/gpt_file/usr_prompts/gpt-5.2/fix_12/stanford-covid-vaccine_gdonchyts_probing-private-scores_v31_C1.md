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

0.3521099280679826

# 6. Current score

0.42295

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.42418) has done: 'I fix the JSON loading error (`ValueError: Trailing data`) by reading the dataset as JSON Lines (`lines=True`) with a fallback to standard JSON so it works across Kaggle file variants. Then I keep your baseline logic intact (per-position means for first 68, global means for the remaining positions) and ensure `seq_scored`/`seq_length` are taken robustly from the loaded frames. Finally, I rebuild the submission by merging onto `sample_submission.csv` to guarantee correct row order and completeness, and write a valid `submission.csv` with the exact required columns.'
- What this solution (achieved 0.42418) has done: 'Your current score is worse than the target (lower is better), so the smallest safe improvement is to remove the intentional, id-specific “reactivity = 1.0” override that injects noise and hurts MCRMSE. I keep the core baseline logic (per-position means for first 68, global means for the rest) identical otherwise, and just ensure the submission stays aligned to `sample_submission.csv`. This should move the score downward toward the target without changing the modeling approach or adding any new dependencies.'
- What this solution (achieved 0.42166) has done: 'Your current score (0.42418, lower-is-better) is worse than the target (0.35211), so we should make a small, legitimate improvement without changing the baseline modeling idea. The biggest safe gain here is to compute means using only the high-quality training subset (`SN_filter == 1`) to reduce noise in the per-position/global averages while keeping the same “position means for first 68, global mean for the rest” logic. I also make the JSON reading more robust to both JSON-lines and normal JSON formats (without changing behavior when your current files already load fine). Everything else (targets, aggregation scheme, merge onto sample submission, output format) stays the same and still produces a valid `submission.csv`.'
- What this solution (achieved 0.48258) has done: 'Your score is worse than the target (lower is better), so we should make a small, legitimate improvement without changing the baseline “per-position means for first 68, global mean for the rest” logic. The most impactful minimal tweak in this baseline family is to compute the means as *error-weighted* averages using the provided `*_error_*` columns, which better matches the metric by downweighting noisy positions/samples while keeping the same prediction structure. We keep using the `SN_filter==1` subset (as you already do) and only change how the position/global means are aggregated. The submission generation, alignment to `sample_submission.csv`, paths, and output schema remain unchanged.'
- What this solution (achieved 0.43957) has done: 'Your current score (0.48258, lower-is-better) is worse than the target (0.35211), so we should make a small, legitimate improvement while keeping the same “per-position means for first 68, global mean for the rest” baseline. The main issue is that inverse-variance weighting can become unstable when some error values are tiny or inconsistent, which can overfit noisy samples/positions and worsen public LB. I keep your weighted-mean idea but stabilize it by (1) clipping errors to a reasonable floor per target, and (2) using a softer weighting exponent (1/error instead of 1/error²), which is still error-aware but less extreme. Submission alignment, paths, schema, and overall prediction structure remain unchanged and it still write a valid `submission.csv`.'
- What this solution (achieved 0.42166) has done: 'Your current score (0.43957, lower-is-better) is still worse than the target (0.35211), so we should make a small, legitimate improvement while keeping the exact same baseline structure (per-position means for first 68, global mean for the rest). The biggest low-risk issue in the current weighted-mean variant is that the error-based weighting can still be noisy/unstable; switching back to simple (unweighted) means within the high-quality subset (`SN_filter==1`) is a minimal change that often improves this competition’s baseline performance. I keep the robust JSON loading, the same seq_scored/seq_length handling, and the same merge onto `sample_submission.csv` to preserve submission order and completeness. This preserves evaluation semantics and should move the score downward toward the target band.'
- What this solution (achieved 0.42178) has done: 'Your score is worse than the target (lower is better), so we make the smallest “same-baseline-family” improvement that usually helps this competition without changing the prediction structure. We keep the exact core logic (per-position means for first 68, global mean for the rest) but compute those means as *signal-to-noise weighted* averages over the existing `SN_filter==1` subset, which tends to better match the evaluation while staying a pure averaging baseline. This uses only an existing numeric column (`signal_to_noise`) and does not change targets, shapes, loops, or submission alignment. We also clip weights to avoid a few very high-SN samples dominating, improving stability and typically lowering MCRMSE toward your target.'
- What this solution (achieved 0.42214) has done: 'Your current score (0.42178, lower-is-better) is still worse than the target (0.35211), so we should make a small, low-risk improvement while keeping the exact same baseline structure (per-position means for first 68, global mean for the rest). The safest gain in this “mean baseline” family is to add a mild per-target shrinkage: blend the per-position mean toward the global mean for that target, which reduces variance from position-specific noise without changing the prediction shape or using any new model. I’m also switching `global_means` to use the same S/N weights you already use for `pos_means` (currently it’s unweighted), to keep the aggregation consistent and slightly more robust. Everything else (data loading, SN_filter usage, loops, submission alignment/format, output path) stays the same and still produces a valid `submission.csv`.'
- What this solution (achieved 0.42214) has done: 'Your current score (0.42214, lower-is-better) is worse than the target (0.35211), so we want a small, low-risk improvement without changing the baseline “per-position means for first 68, global mean for the rest” logic. The simplest legit gain is to use the competition’s known training-quality heuristic: compute means only from sequences with `SN_filter==1` *and* reasonably high `signal_to_noise`, which reduces label noise while keeping the same averaging approach. To avoid harming generalization, we apply a mild S/N threshold and keep your existing signal-to-noise weighting and shrinkage unchanged. Everything else (robust JSON loading, shapes, seq_scored/seq_length handling, merge onto sample for order, and writing `submission.csv`) remains the same.'
- What this solution (achieved 0.42295) has done: 'We keep your exact baseline structure (position means for the first 68, global means for the rest, merge onto `sample_submission.csv`) and only make two small, score-relevant adjustments. First, we apply the same shrinkage to the `global_means` (mixing with an overall grand mean) to reduce noise from a few high-weight sequences, which tends to lower MCRMSE without changing prediction semantics. Second, we tune the shrinkage strength slightly (a bit more regularization) because your current score is still worse than the target, and this is the smallest stable lever within the same averaging baseline family. Everything else (robust JSON loading, SN_filter/SN thresholding, S/N weighting, output format/path) remains unchanged and still writes a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os

BASE_INPUT = "../input/stanford-covid-vaccine"
if not os.path.exists(BASE_INPUT):
    BASE_INPUT = "/kaggle/input/stanford-covid-vaccine"

train_path = os.path.join(BASE_INPUT, "train.json")
test_path = os.path.join(BASE_INPUT, "test.json")
sample_path = os.path.join(BASE_INPUT, "sample_submission.csv")


def read_json_robust(path: str) -> pd.DataFrame:
    """
    Robustly read either JSON Lines or standard JSON.

    Why this helps score/stability:
    - Prevents exceptions / partial reads that could lead to malformed arrays and bad predictions.
    - Does not change the modeling logic; only ensures the intended data is loaded.
    """
    try:
        return pd.read_json(path, lines=True)
    except ValueError:
        try:
            return pd.read_json(path)
        except ValueError:
            with open(path, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
            return pd.DataFrame(
                [
                    pd.read_json(pd.io.common.StringIO(line), typ="series")
                    for line in lines
                ]
            )


df_train = read_json_robust(train_path)
df_test = read_json_robust(test_path)
sample = pd.read_csv(sample_path)

print("df_train:", df_train.shape, "df_test:", df_test.shape, "sample:", sample.shape)
print("train cols:", list(df_train.columns))



## === cell 1
target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

seq_scored = int(pd.Series(df_train["seq_scored"]).mode().iloc[0])  # expected 68
seq_length = int(pd.Series(df_train["seq_length"]).mode().iloc[0])  # expected 107

df_train_means = df_train.copy()
if "SN_filter" in df_train_means.columns:
    df_train_means = df_train_means[df_train_means["SN_filter"].astype(int) == 1].copy()

if "signal_to_noise" in df_train_means.columns:
    sn = df_train_means["signal_to_noise"].astype(np.float64)
    df_train_means = df_train_means[sn >= 1.0].copy()

if len(df_train_means) == 0:
    df_train_means = df_train.copy()

print("Using rows for mean computation:", len(df_train_means), "of", len(df_train))

if "signal_to_noise" in df_train_means.columns:
    w = df_train_means["signal_to_noise"].astype(np.float64).values
    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    w = np.clip(w, 0.0, 10.0)
else:
    w = np.ones(len(df_train_means), dtype=np.float64)

w_sum = float(np.sum(w))
if w_sum <= 0:
    w = np.ones(len(df_train_means), dtype=np.float64)
    w_sum = float(np.sum(w))

pos_means = {}
for col in target_cols:
    y = np.stack(df_train_means[col].values).astype(np.float64)  # (n, 68)
    mu = np.sum(y * w[:, None], axis=0) / w_sum
    pos_means[col] = mu.astype(np.float32)

global_means = {}
grand_means = {}

for col in target_cols:
    y = np.stack(df_train_means[col].values).astype(np.float64)  # (n, 68)

    grand_means[col] = float(np.mean(y))

    weighted_global = float(np.sum(y * w[:, None]) / (w_sum * y.shape[1]))
    beta = 0.90  # mild stabilization of global constants (small change, usually helps MCRMSE)
    global_means[col] = beta * weighted_global + (1.0 - beta) * grand_means[col]

print("seq_scored:", seq_scored, "seq_length:", seq_length)
print("grand_means:", grand_means)
print("global_means:", global_means)

alpha = 0.80  # was 0.85; small increase in regularization toward global means
for col in target_cols:
    pos_means[col] = (
        alpha * pos_means[col] + (1.0 - alpha) * np.float32(global_means[col])
    ).astype(np.float32)



## === cell 2
rows = []
for _id in df_test["id"].astype(str).values:
    for pos in range(seq_length):
        row = {"id_seqpos": f"{_id}_{pos}"}
        for col in target_cols:
            if pos < seq_scored:
                row[col] = float(pos_means[col][pos])
            else:
                row[col] = float(global_means[col])
        rows.append(row)

df = pd.DataFrame(rows, columns=["id_seqpos"] + target_cols)

df = sample[["id_seqpos"]].merge(df, on="id_seqpos", how="left")

for col in target_cols:
    df[col] = df[col].fillna(global_means[col]).astype(np.float32)

print(df.head())
print("Submission frame shape:", df.shape)



## === cell 3
sequences_public = set(df_train.id.astype(str).values)

sequences = list(
    set(["_".join(v.split("_")[:-1]) for v in df.id_seqpos.astype(str).values])
)
sequences = [s for s in sequences if s not in sequences_public]
print("Derived sequences (not used):", sequences[:5], "count:", len(sequences))



## === cell 4
expected_cols = [
    "id_seqpos",
    "reactivity",
    "deg_Mg_pH10",
    "deg_pH10",
    "deg_Mg_50C",
    "deg_50C",
]
df = df[expected_cols]

assert df.shape[0] == sample.shape[0], "Row count mismatch vs sample submission"
assert (
    list(df.columns) == expected_cols
), "Column mismatch vs expected submission format"

out_path = "submission.csv"
df.to_csv(out_path, index=False)
print("Wrote", out_path, "with shape:", df.shape)
print(df.head())
