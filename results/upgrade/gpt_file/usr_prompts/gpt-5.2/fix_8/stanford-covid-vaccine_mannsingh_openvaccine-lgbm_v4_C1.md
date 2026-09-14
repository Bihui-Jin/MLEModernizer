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

3.8

# 3. Installed packages

geopandas==0.14.4
lightgbm==4.6.0
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
xgboost==2.0.3

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

0.71038

# 6. Current score

5015.94579

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 4432.75698) has done: 'I remove the broken BPPS-loading cells (the dataset in this environment doesn’t include the `bpps/` folder), and fix LightGBM training to work with the installed LightGBM version by using callback-based early stopping instead of the deprecated `early_stopping_rounds` argument. I also fix a couple of feature/label bugs (loop-type counts were mistakenly computed from `sequence`, and “mean_*_error” columns were computed from the wrong arrays) while keeping the same overall approach (sequence/structure categorical features + LightGBM on mean targets). Finally, I make the merge robust and ensure we always write a valid `submission.csv` with exactly the sample submission columns.'
- What this solution (achieved 5015.94579) has done: 'Your current score is catastrophically high because the submission repeats one constant (per-id) prediction for all 107 positions, but the metric is computed per-position for the first 68 scored positions. Keeping your LightGBM “mean target per id” core logic intact, the smallest legitimate fix is to expand those per-id mean predictions into per-position predictions by using the training-set average *shape* over sequence position: predict = per-id mean × (position_factor[target][pos]). This preserves your modeling approach (same features, same models, same loss/fit), but makes the output aligned with the evaluation semantics and should drastically reduce MCRMSE toward the 0.71 range. I also keep submission formatting robust by merging on id then mapping by seqpos extracted from `id_seqpos`, while still writing `submission.csv` with exactly the sample columns.'
- What this solution (achieved 5015.94579) has done: 'Your score is still catastrophically high (lower-is-better) so we should make the smallest changes that correct evaluation semantics without changing your core modeling approach (LightGBM on per-id mean targets). The biggest remaining issue is that you’re submitting zeros for the two unscored columns, which can indirectly hurt because your positional expansion currently only covers 3 columns; we can fill the missing two columns using the same “per-position average shape × per-id mean” trick computed from the corresponding training targets. Additionally, we apply the positional factors for all 5 targets and compute mean targets for all 5 in the same way you already do, while keeping the same feature set, same model class, same training loop, and same loss/metric. This should move the score dramatically down toward the 0.71 range (and certainly far below thousands) while staying within your solution’s intended logic.'
- What this solution (achieved 4432.75697) has done: 'Your current score is still orders of magnitude worse than the target (lower-is-better), which strongly suggests the submission is misaligned with the required per-position evaluation semantics. Keeping your core approach (LightGBM predicting per-id mean targets from the same features), the smallest fix that should dramatically reduce MCRMSE is to generate per-position predictions via an additive “mean + position_delta” decomposition learned from the training set, rather than a multiplicative scaling that can explode when the overall mean is near zero. I compute position-wise deltas for each target from the training arrays, expand each per-id mean prediction across seqpos using those deltas (and fill positions 68–106 with the last delta), and ensure the submission rows/ids align exactly to `sample_submission.csv`. This keeps the same models, features, and training loop, but corrects the per-position output in a more numerically stable way.'
- What this solution (achieved 4647.22712) has done: 'Your current score is far worse than the target (lower-is-better), which strongly indicates the main remaining issue is still *submission alignment/semantics*, not model quality. The smallest fix that should massively reduce MCRMSE is to stop using “mean+global positional delta” and instead expand per-id predictions into per-position predictions using the training set’s *position-wise mean shape after per-id demeaning* (a stable multiplicative shape on centered values). This keeps your core logic intact (same features, same LightGBM per-id mean models, same training loop/loss), but makes the per-position outputs much more realistic and avoids huge blow-ups when means are near zero. I also add a strict sanity check to ensure the constructed `id` matches the test ids exactly, and I merge predictions by `id` in the sample submission order to guarantee row alignment.'
- What this solution (achieved 4432.75698) has done: 'Your score is still orders of magnitude worse than the target (lower is better), so the remaining issue is almost certainly prediction scale/semantics rather than model quality. The minimal fix is to stop using the multiplicative `(1 + alpha*shape)` expansion (which can explode when per-id means are negative/small) and instead use an additive expansion that preserves each id’s predicted mean exactly: `pred(pos) = mean_pred + beta * mean_abs * shape(pos)`. I compute the per-target `mean_abs` from the training arrays (a stable amplitude) and use a small `beta` to introduce realistic positional variation without changing the core LightGBM setup. I keep the merge/alignment with `sample_submission.csv` intact and still output a valid `submission.csv` with the required columns/order.'
- What this solution (achieved 5015.94579) has done: 'Your score is still orders of magnitude worse than the target (lower is better), which points to a submission-semantics mismatch rather than model quality. The smallest fix that preserves your core approach (LightGBM predicting per-id mean targets) is to expand those mean predictions to per-position predictions using a *per-target, per-position mean profile* learned from train, and to enforce that the first 68 positions’ average equals the predicted mean (so we don’t shift the mean inadvertently). Concretely, we compute a stable “shape” = position-wise mean of the training targets (not centered), normalize it so its mean over the first 68 equals 1, and then output `pred(pos) = mean_pred * shape[pos]` for all 5 columns. We also add a small safety clamp on the shape to avoid rare division blow-ups, while keeping the rest of the pipeline and submission alignment identical.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

from sklearn.model_selection import train_test_split
import lightgbm as lgb

RANDOM_STATE = 42



## === cell 1
train = pd.read_json("../input/stanford-covid-vaccine/train.json", lines=True)
test = pd.read_json("../input/stanford-covid-vaccine/test.json", lines=True)
ss = pd.read_csv("../input/stanford-covid-vaccine/sample_submission.csv")



## === cell 2
train = train.set_index("index")
test = test.set_index("index")



## === cell 3
ss.head()



## === cell 4
train.head(3)



## === cell 5
test.seq_length.value_counts()



## === cell 6
test.head(3)



## === cell 7
print("Size of training examples: ", np.shape(train))
print("Size of test examples: ", np.shape(test))



## === cell 8
print("========= train columns ==========")
print([c for c in train.columns])

print("========= test columns ==========")
print([c for c in test.columns])



## === cell 9
train.info()



## === cell 10
BPPS_DIR = "../input/stanford-covid-vaccine/bpps/"
if os.path.isdir(BPPS_DIR):
    bpps_list = sorted(os.listdir(BPPS_DIR))
    bpps_npy = np.load(os.path.join(BPPS_DIR, bpps_list[0]))
    print("Count of npy files: ", len(bpps_list))
    print("Size of image: ", bpps_npy.shape)
else:
    bpps_list = []
    print(f"BPPS directory not found at {BPPS_DIR}. Skipping BPPS-related steps.")



## === cell 11
if bpps_list:
    NO_OF_EXAMPLES = min(15, len(bpps_list))
    fig = plt.figure(figsize=(15, 15))
    for i in range(NO_OF_EXAMPLES):
        bpps_eg = np.load(os.path.join(BPPS_DIR, bpps_list[i]))
        sub = fig.add_subplot(5, 5, i + 1)
        sub.imshow(bpps_eg)
    plt.show()



## === cell 12
Counter(train["sequence"].values[0])



## === cell 13
Counter(train["predicted_loop_type"].values[0])




## === cell 14
def featurize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["total_A_count"] = df["sequence"].apply(lambda s: s.count("A"))
    df["total_G_count"] = df["sequence"].apply(lambda s: s.count("G"))
    df["total_U_count"] = df["sequence"].apply(lambda s: s.count("U"))
    df["total_C_count"] = df["sequence"].apply(lambda s: s.count("C"))

    df["total_dot_count"] = df["structure"].apply(lambda s: s.count("."))
    df["total_ob_count"] = df["structure"].apply(lambda s: s.count("("))
    df["total_cb_count"] = df["structure"].apply(lambda s: s.count(")"))

    df["total_S_count"] = df["predicted_loop_type"].apply(lambda s: s.count("S"))
    df["total_M_count"] = df["predicted_loop_type"].apply(lambda s: s.count("M"))
    df["total_I_count"] = df["predicted_loop_type"].apply(lambda s: s.count("I"))
    df["total_X_count"] = df["predicted_loop_type"].apply(lambda s: s.count("X"))
    df["total_B_count"] = df["predicted_loop_type"].apply(lambda s: s.count("B"))
    df["total_H_count"] = df["predicted_loop_type"].apply(lambda s: s.count("H"))
    df["total_E_count"] = df["predicted_loop_type"].apply(lambda s: s.count("E"))

    return df




## === cell 15
train = featurize(train)
test = featurize(test)



## === cell 16
train["mean_reactivity_error"] = train["reactivity_error"].apply(
    lambda x: float(np.mean(x))
)
train["mean_deg_error_Mg_pH10"] = train["deg_error_Mg_pH10"].apply(
    lambda x: float(np.mean(x))
)
train["mean_deg_error_pH10"] = train["deg_error_pH10"].apply(
    lambda x: float(np.mean(x))
)
train["mean_deg_error_Mg_50C"] = train["deg_error_Mg_50C"].apply(
    lambda x: float(np.mean(x))
)
train["mean_deg_error_50C"] = train["deg_error_50C"].apply(lambda x: float(np.mean(x)))



## === cell 17
train["mean_reactivity"] = (
    train["reactivity"].apply(lambda x: float(np.mean(x)))
    + train["mean_reactivity_error"]
)
train["mean_deg_Mg_pH10"] = (
    train["deg_Mg_pH10"].apply(lambda x: float(np.mean(x)))
    + train["mean_deg_error_Mg_pH10"]
)
train["mean_deg_pH10"] = (
    train["deg_pH10"].apply(lambda x: float(np.mean(x))) + train["mean_deg_error_pH10"]
)
train["mean_deg_Mg_50C"] = (
    train["deg_Mg_50C"].apply(lambda x: float(np.mean(x)))
    + train["mean_deg_error_Mg_50C"]
)
train["mean_deg_50C"] = (
    train["deg_50C"].apply(lambda x: float(np.mean(x))) + train["mean_deg_error_50C"]
)



## === cell 18
for n in range(107):
    train[f"sequence_{n}"] = train["sequence"].str[n].astype("category")
    test[f"sequence_{n}"] = test["sequence"].str[n].astype("category")



## === cell 19
for n in range(107):
    train[f"structure_{n}"] = train["structure"].str[n].astype("category")
    test[f"structure_{n}"] = test["structure"].str[n].astype("category")



## === cell 20
SEQUENCE_COLS = [c for c in train.columns if c.startswith("sequence_")]
STRUCTURE_COLS = [c for c in train.columns if c.startswith("structure_")]
OTHERS = [
    "total_A_count",
    "total_G_count",
    "total_U_count",
    "total_C_count",
    "total_dot_count",
    "total_ob_count",
    "total_cb_count",
    "total_S_count",
    "total_M_count",
    "total_I_count",
    "total_X_count",
    "total_B_count",
    "total_H_count",
    "total_E_count",
]
MY_COLS = SEQUENCE_COLS + STRUCTURE_COLS + OTHERS

print("Number of features:", len(MY_COLS))



## === cell 21
for target in ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]:
    X = train[MY_COLS]
    y = train[f"mean_{target}"]
    X_test = test[MY_COLS]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    reg = lgb.LGBMRegressor(random_state=RANDOM_STATE, n_estimators=5000)

    reg.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=True)],
    )

    test[f"mean_{target}_pred"] = reg.predict(X_test, num_iteration=reg.best_iteration_)



## === cell 22
test[
    [
        "id",
        "mean_reactivity_pred",
        "mean_deg_Mg_pH10_pred",
        "mean_deg_pH10_pred",
        "mean_deg_Mg_50C_pred",
        "mean_deg_50C_pred",
    ]
].head()




## === cell 23
def _positional_profile(
    train_df: pd.DataFrame, target_col: str, seq_length: int = 107
) -> np.ndarray:
    arr68 = np.vstack(train_df[target_col].values).astype(np.float64)  # (n, 68)
    prof68 = arr68.mean(axis=0)  # (68,)
    denom = float(
        np.mean(prof68) + 1e-12
    )  # normalize so mean over scored positions is 1
    profile68 = (prof68 / denom).astype(np.float64)

    profile68 = np.clip(profile68, -5.0, 5.0)

    profile107 = np.ones(seq_length, dtype=np.float64)
    profile107[: len(profile68)] = profile68
    profile107[len(profile68) :] = float(profile68[-1])
    return profile107


pos_profile = {
    "reactivity": _positional_profile(train, "reactivity", seq_length=107),
    "deg_Mg_pH10": _positional_profile(train, "deg_Mg_pH10", seq_length=107),
    "deg_pH10": _positional_profile(train, "deg_pH10", seq_length=107),
    "deg_Mg_50C": _positional_profile(train, "deg_Mg_50C", seq_length=107),
    "deg_50C": _positional_profile(train, "deg_50C", seq_length=107),
}

for k in ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]:
    print(
        k,
        "profile mean(first68):",
        float(np.mean(pos_profile[k][:68])),
        "min/max:",
        float(pos_profile[k][:68].min()),
        float(pos_profile[k][:68].max()),
    )



## === cell 24
ss_work = ss.copy()
tmp = ss_work["id_seqpos"].str.split("_", expand=True)

ss_work["id"] = "id_" + tmp[1]
ss_work["seqpos"] = tmp[2].astype(int)

test_ids = set(test["id"].values.tolist())
parsed_ids = set(ss_work["id"].unique().tolist())
missing_in_test = sorted(list(parsed_ids - test_ids))
if len(missing_in_test) > 0:
    raise ValueError(
        f"Parsed {len(missing_in_test)} ids from sample_submission not found in test.json. Example: {missing_in_test[:5]}"
    )

preds_mean = test[
    [
        "id",
        "mean_reactivity_pred",
        "mean_deg_Mg_pH10_pred",
        "mean_deg_pH10_pred",
        "mean_deg_Mg_50C_pred",
        "mean_deg_50C_pred",
    ]
].copy()

ss_new = ss_work.drop(
    ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"],
    axis=1,
).merge(preds_mean, on="id", how="left", validate="m:1")

mean_pred_cols = [
    "mean_reactivity_pred",
    "mean_deg_Mg_pH10_pred",
    "mean_deg_pH10_pred",
    "mean_deg_Mg_50C_pred",
    "mean_deg_50C_pred",
]

if ss_new[mean_pred_cols].isna().any().any():
    bad = ss_new.loc[
        ss_new[mean_pred_cols].isna().any(axis=1), ["id_seqpos", "id"]
    ].head(5)
    raise ValueError(f"Found NaN predictions after merge; example rows:\n{bad}")

seqpos = ss_new["seqpos"].values

ss_new["reactivity"] = (
    ss_new["mean_reactivity_pred"].values * pos_profile["reactivity"][seqpos]
)
ss_new["deg_Mg_pH10"] = (
    ss_new["mean_deg_Mg_pH10_pred"].values * pos_profile["deg_Mg_pH10"][seqpos]
)
ss_new["deg_pH10"] = (
    ss_new["mean_deg_pH10_pred"].values * pos_profile["deg_pH10"][seqpos]
)
ss_new["deg_Mg_50C"] = (
    ss_new["mean_deg_Mg_50C_pred"].values * pos_profile["deg_Mg_50C"][seqpos]
)
ss_new["deg_50C"] = ss_new["mean_deg_50C_pred"].values * pos_profile["deg_50C"][seqpos]

ss_new = ss_new.drop(mean_pred_cols + ["id", "seqpos"], axis=1)

for c in ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]:
    ss_new[c] = ss_new[c].astype(float).fillna(0.0)



## === cell 25
submission = ss_new[ss.columns].copy()
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print("Columns:", list(submission.columns))
submission.head()
