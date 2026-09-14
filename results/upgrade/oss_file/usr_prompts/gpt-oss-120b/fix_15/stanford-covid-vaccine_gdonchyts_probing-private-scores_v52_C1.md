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

0.43509

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.47906) has done: 'I replace the missing baseline file with the official sample submission, compute simple global means from the training data for each target, fill the submission with those means (a valid baseline), and correctly write out `submission.csv`. This fixes the FileNotFoundError, the undefined‑`df` errors, and ensures a proper CSV is produced.'
- What this solution (achieved 0.42418) has done: 'I replace the single global‑mean predictions with per‑position means computed from the training data. For each target I compute a 68‑length vector of column‑wise averages, then fill the submission rows by extracting the position index from `id_seqpos`. Positions beyond the scored region fall back to the overall mean. This modest change keeps the original pipeline intact while giving more accurate, position‑aware predictions, which should lower the MCRMSE toward the target score.'
- What this solution (achieved 0.3944) has done: 'I add a lightweight per‑position, per‑loop‑type mean model.  
First I build a dictionary mapping each test‑sample id to its 68‑character loop‑type string.  
Then I compute, from the training set, means for every target at each position **and** for each loop‑type character.  
During prediction I look up the sample’s loop‑type at the required position; if a specific mean exists I use it, otherwise I fall back to the plain position mean (or the overall mean). This keeps the original logic but adds a small amount of useful signal, which should lower the MCRMSE toward the target without changing the core pipeline.'
- What this solution (achieved 0.39444) has done: 'I add a lightweight per‑loop‑type overall mean as an extra fallback: after trying the per‑position‑per‑loop mean, the code use the overall mean for that loop character (if available) before falling back to the plain position mean or the global mean. This adds a tiny amount of signal without changing the core pipeline, and should reduce the MCRMSE toward the target.'
- What this solution (achieved 0.39444) has done: 'I add a lightweight per‑position‑per‑nucleotide mean as an additional fallback. After building the existing per‑position‑per‑loop and per‑loop means, I compute means for each target conditioned on the nucleotide (A,C,G,U) at each of the 68 scored positions. In `get_pred_value` I first try the loop‑aware means, then the nucleotide‑aware mean, and finally fall back to the plain position mean or global mean. This adds a bit more signal without changing the overall modeling pipeline and should nudge the MCRMSE toward the target score.'
- What this solution (achieved 0.39444) has done: 'I add a lightweight overall‑nucleotide mean as an extra fallback (computed from the training set) and modify the prediction function to use it before resorting to the plain per‑position mean. This small extra signal is expected to nudge the MCRMSE down toward the target without altering the core modeling logic.'
- What this solution (achieved 0.39444) has done: 'I add a lightweight per‑position‑per‑loop‑and‑nucleotide mean lookup, which provides a more specific fallback before the existing nucleotide‑only fallback. This adds a small amount of extra signal while keeping the original pipeline unchanged, and is expected to nud‑ge the MCRMSE closer to the target score.'
- What this solution (achieved 0.43509) has done: 'I replace the simple un‑weighted averages with weighted averages that use the provided error columns as inverse‑variance weights. This keeps the same fallback hierarchy but gives each mean a more reliable estimate, which should nudge the MCRMSE down toward the target. The rest of the pipeline and the submission format remain unchanged.'
- What this solution (achieved 0.43509) has done: 'I replace the inverse‑variance weighted averages with simple un‑weighted means for the fallback predictions. Weighted means can amplify noisy measurements and are the main reason the current score (0.435) is above the target; using plain averages should give a more stable baseline and move the MCRMSE toward the desired 0.3519 while preserving the existing lookup hierarchy.'
- What this solution (achieved 0.43509) has done: 'I replace the plain per‑position simple means used as the final fallback with the inverse‑variance weighted per‑position means that were already computed (`*_pos_weighted`). These weighted averages better reflect measurement confidence, so predictions be slightly more accurate and the MCRMSE should move closer to the target value. No other logic is altered.'
- What this solution (achieved 0.43509) has done: 'I switch the primary fallback from the weighted per‑position means to the un‑weighted (plain) per‑position means, and add lightweight un‑weighted nucleotide‑specific means as an extra fallback. This reduces the influence of noisy error‑based weights while keeping the existing hierarchy, moving the MCRMSE closer to the target score.'
- What this solution (achieved 0.43509) has done: 'I replace the un‑weighted per‑position and global means with the already‑computed inverse‑variance weighted versions, so the fallback hierarchy now uses higher‑quality estimates without altering the core pipeline.'
- What this solution (achieved 0.43509) has done: 'I adjust the prediction fallback hierarchy to favour the more stable un‑weighted per‑position and global means, which have shown to reduce the error compared to the current weighted versions. The function now receives both weighted and simple means and returns the simple mean earlier, keeping all other look‑ups unchanged. This small change keeps the core pipeline intact while moving the MCRMSE closer to the target.'
- What this solution (achieved 0.43509) has done: 'I adjust the prediction fallback hierarchy so that the more reliable inverse‑variance weighted per‑position means are used before the un‑weighted per‑position means. This small change keeps all existing statistics and look‑ups intact while giving higher priority to the weighted averages, which should lower the MCRMSE toward the target score.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
from collections import defaultdict

df_test = pd.read_json("../input/stanford-covid-vaccine/test.json", lines=True)
df_sub = pd.read_csv("../input/sample_submission.csv")
df_train = pd.read_json("../input/stanford-covid-vaccine/train.json", lines=True)


def flatten_column(col):
    return np.concatenate(col.values)


error_map = {
    "reactivity": "reactivity_error",
    "deg_Mg_pH10": "deg_error_Mg_pH10",
    "deg_pH10": "deg_error_pH10",
    "deg_Mg_50C": "deg_error_Mg_50C",
    "deg_50C": "deg_error_50C",
}

eps = 1e-6  # avoid division by zero in weighting

global_weighted_means = {}
simple_global_means = {}
for target, err_col in error_map.items():
    vals = flatten_column(df_train[target])
    errs = flatten_column(df_train[err_col])
    weights = 1.0 / ((errs + eps) ** 2)
    global_weighted_means[target] = np.average(vals, weights=weights)
    simple_global_means[target] = np.mean(vals)


def weighted_pos_mean(target):
    vals = np.vstack(df_train[target].values)
    errs = np.vstack(df_train[error_map[target]].values)
    weights = 1.0 / ((errs + eps) ** 2)
    return np.average(vals, axis=0, weights=weights)


def simple_pos_mean(target):
    return np.mean(np.vstack(df_train[target].values), axis=0)


reactivity_pos_weighted = weighted_pos_mean("reactivity")
deg_Mg_pH10_pos_weighted = weighted_pos_mean("deg_Mg_pH10")
deg_pH10_pos_weighted = weighted_pos_mean("deg_pH10")
deg_Mg_50C_pos_weighted = weighted_pos_mean("deg_Mg_50C")
deg_50C_pos_weighted = weighted_pos_mean("deg_50C")

reactivity_pos_simple = simple_pos_mean("reactivity")
deg_Mg_pH10_pos_simple = simple_pos_mean("deg_Mg_pH10")
deg_pH10_pos_simple = simple_pos_mean("deg_pH10")
deg_Mg_50C_pos_simple = simple_pos_mean("deg_Mg_50C")
deg_50C_pos_simple = simple_pos_mean("deg_50C")

weighted_pos_means = {
    "reactivity": reactivity_pos_weighted,
    "deg_Mg_pH10": deg_Mg_pH10_pos_weighted,
    "deg_pH10": deg_pH10_pos_weighted,
    "deg_Mg_50C": deg_Mg_50C_pos_weighted,
    "deg_50C": deg_50C_pos_weighted,
}
simple_pos_means = {
    "reactivity": reactivity_pos_simple,
    "deg_Mg_pH10": deg_Mg_pH10_pos_simple,
    "deg_pH10": deg_pH10_pos_simple,
    "deg_Mg_50C": deg_Mg_50C_pos_simple,
    "deg_50C": deg_50C_pos_simple,
}

id_to_loop = (
    df_test.set_index("id")["predicted_loop_type"].apply(lambda s: s[:68]).to_dict()
)

id_to_seq = df_train.set_index("id")["sequence"].apply(lambda s: s[:68]).to_dict()

targets = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
loop_chars = set("SMIBHEX")  # observed loop characters
nuc_chars = set("ACGU")  # nucleotides

acc = defaultdict(lambda: [0.0, 0.0])
loop_acc = defaultdict(lambda: [0.0, 0.0])
nuc_acc = defaultdict(lambda: [0.0, 0.0])
nuc_total = defaultdict(lambda: [0.0, 0.0])  # overall nucleotide
combo_acc = defaultdict(lambda: [0.0, 0.0])

simple_nuc_acc = defaultdict(lambda: [0.0, 0])
simple_nuc_total = defaultdict(lambda: [0.0, 0])

for _, row in df_train.iterrows():
    loops = row["predicted_loop_type"][:68]  # first 68 loop chars
    seq = row["sequence"][:68]  # first 68 nucleotides
    for target in targets:
        vals = row[target]  # length‑68 list/array
        errs = row[error_map[target]]  # corresponding errors
        for pos, (val, err, lc, nuc) in enumerate(zip(vals, errs, loops, seq)):
            w = 1.0 / ((err + eps) ** 2)
            if lc in loop_chars:
                acc[(target, pos, lc)][0] += val * w
                acc[(target, pos, lc)][1] += w
                loop_acc[(target, lc)][0] += val * w
                loop_acc[(target, lc)][1] += w
            if nuc in nuc_chars:
                nuc_acc[(target, pos, nuc)][0] += val * w
                nuc_acc[(target, pos, nuc)][1] += w
                nuc_total[(target, nuc)][0] += val * w
                nuc_total[(target, nuc)][1] += w

                simple_nuc_acc[(target, pos, nuc)][0] += val
                simple_nuc_acc[(target, pos, nuc)][1] += 1
                simple_nuc_total[(target, nuc)][0] += val
                simple_nuc_total[(target, nuc)][1] += 1

            if lc in loop_chars and nuc in nuc_chars:
                combo_acc[(target, pos, lc, nuc)][0] += val * w
                combo_acc[(target, pos, lc, nuc)][1] += w

loop_pos_mean = {
    (t, p, lc): s / w if w > 0 else np.nan for (t, p, lc), (s, w) in acc.items()
}
loop_type_mean = {
    (t, lc): s / w if w > 0 else np.nan for (t, lc), (s, w) in loop_acc.items()
}
nuc_pos_mean = {
    (t, p, nuc): s / w if w > 0 else np.nan for (t, p, nuc), (s, w) in nuc_acc.items()
}
nuc_type_mean = {
    (t, nuc): s / w if w > 0 else np.nan for (t, nuc), (s, w) in nuc_total.items()
}
combo_pos_mean = {
    (t, p, lc, nuc): s / w if w > 0 else np.nan
    for (t, p, lc, nuc), (s, w) in combo_acc.items()
}
simple_nuc_pos_mean = {
    (t, p, nuc): s / cnt if cnt > 0 else np.nan
    for (t, p, nuc), (s, cnt) in simple_nuc_acc.items()
}
simple_nuc_type_mean = {
    (t, nuc): s / cnt if cnt > 0 else np.nan
    for (t, nuc), (s, cnt) in simple_nuc_total.items()
}




## === cell 1
def get_pred_value(
    row, target, weighted_pos, simple_pos, weighted_global, simple_global
):
    """
    Return prediction for a given target.
    Updated fallback order (most specific → most general):
    1) per‑position‑per‑loop‑and‑nucleotide weighted mean
    2) per‑position‑per‑loop weighted mean
    3) overall loop‑type weighted mean
    4) per‑position‑per‑nucleotide weighted mean
    5) overall nucleotide weighted mean
    6) per‑position‑per‑nucleotide **un‑weighted** mean
    7) overall nucleotide **un‑weighted** mean
    8) **inverse‑variance weighted** per‑position mean   ← higher priority now
    9) **un‑weighted** per‑position mean
    10) **un‑weighted** global mean
    11) weighted global mean
    """
    try:
        id_part, pos_str = row["id_seqpos"].rsplit("_", 1)
        pos = int(pos_str)
    except Exception:
        return simple_global

    if not (0 <= pos < len(weighted_pos)):
        return simple_global

    loop_seq = id_to_loop.get(id_part, None)
    seq_str = id_to_seq.get(id_part, None)

    loop_char = None
    nuc_char = None
    if loop_seq is not None and pos < len(loop_seq):
        loop_char = loop_seq[pos]
    if seq_str is not None and pos < len(seq_str):
        nuc_char = seq_str[pos]

    if loop_char is not None and nuc_char is not None:
        key_combo = (target, pos, loop_char, nuc_char)
        if key_combo in combo_pos_mean:
            return combo_pos_mean[key_combo]

    if loop_char is not None:
        key_pos = (target, pos, loop_char)
        if key_pos in loop_pos_mean:
            return loop_pos_mean[key_pos]
        key_loop = (target, loop_char)
        if key_loop in loop_type_mean:
            return loop_type_mean[key_loop]

    if nuc_char is not None:
        key_nuc = (target, pos, nuc_char)
        if key_nuc in nuc_pos_mean:
            return nuc_pos_mean[key_nuc]
        key_nuc_type = (target, nuc_char)
        if key_nuc_type in nuc_type_mean:
            return nuc_type_mean[key_nuc_type]

        if key_nuc in simple_nuc_pos_mean:
            return simple_nuc_pos_mean[key_nuc]
        if key_nuc_type in simple_nuc_type_mean:
            return simple_nuc_type_mean[key_nuc_type]

    if weighted_pos is not None:
        return weighted_pos[pos]

    if simple_pos is not None:
        return simple_pos[pos]

    return simple_global if simple_global is not None else weighted_global




## === cell 2
df_sub["reactivity"] = df_sub.apply(
    lambda r: get_pred_value(
        r,
        "reactivity",
        weighted_pos_means["reactivity"],
        simple_pos_means["reactivity"],
        global_weighted_means["reactivity"],
        simple_global_means["reactivity"],
    ),
    axis=1,
)
df_sub["deg_Mg_pH10"] = df_sub.apply(
    lambda r: get_pred_value(
        r,
        "deg_Mg_pH10",
        weighted_pos_means["deg_Mg_pH10"],
        simple_pos_means["deg_Mg_pH10"],
        global_weighted_means["deg_Mg_pH10"],
        simple_global_means["deg_Mg_pH10"],
    ),
    axis=1,
)
df_sub["deg_pH10"] = df_sub.apply(
    lambda r: get_pred_value(
        r,
        "deg_pH10",
        weighted_pos_means["deg_pH10"],
        simple_pos_means["deg_pH10"],
        global_weighted_means["deg_pH10"],
        simple_global_means["deg_pH10"],
    ),
    axis=1,
)
df_sub["deg_Mg_50C"] = df_sub.apply(
    lambda r: get_pred_value(
        r,
        "deg_Mg_50C",
        weighted_pos_means["deg_Mg_50C"],
        simple_pos_means["deg_Mg_50C"],
        global_weighted_means["deg_Mg_50C"],
        simple_global_means["deg_Mg_50C"],
    ),
    axis=1,
)
df_sub["deg_50C"] = df_sub.apply(
    lambda r: get_pred_value(
        r,
        "deg_50C",
        weighted_pos_means["deg_50C"],
        simple_pos_means["deg_50C"],
        global_weighted_means["deg_50C"],
        simple_global_means["deg_50C"],
    ),
    axis=1,
)




## === cell 3
df_sub.to_csv("submission.csv", index=False)
