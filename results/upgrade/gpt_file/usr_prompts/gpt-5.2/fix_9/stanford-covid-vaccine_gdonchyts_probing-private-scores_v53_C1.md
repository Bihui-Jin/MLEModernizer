# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 0.63824) has done: 'I remove the dependency on a missing external “worst-submission/ensemble52.csv” file and instead start from the provided `sample_submission.csv`, which guarantees the correct row count and column names. I also make the “private set seq_length == 130” logic safe for this dataset (which has seq_length 107) so it doesn’t crash when indexing an empty list. Since there is no model here (it was attempting to hack one id’s reactivity), I keep the core intent (a baseline submission) but make it valid and deterministic by filling predictions with zeros and only applying the provided tweak if the id exists. Finally, I always write a `submission.csv` with the required columns.'
- What this solution (achieved 0.47906) has done: 'Your current submission is essentially a zero baseline with one large outlier tweak, which explains the relatively poor MCRMSE. To move the score closer to the target (lower is better) without changing the overall “no real model” core logic, I (1) remove the harmful hardcoded `reactivity = 100.0` spike and (2) replace the all-zero fill with a slightly stronger constant baseline equal to the training-set per-target mean over the scored 68 positions (still a constant predictor, but better calibrated). This keeps the approach simple/deterministic and uses only legitimate training statistics. The output still be a valid `submission.csv` with the required schema and row count.'
- What this solution (achieved 0.47906) has done: 'You’re currently using a constant-per-target baseline computed as the *mean of per-sequence means*, which is slightly miscalibrated versus the metric that scores every base equally. I keep the same “constant predictor” core logic, but compute constants as the *global mean over all scored positions (first 68)* across all training rows, which better matches MCRMSE weighting and should move your score down toward the target. I also safely drop the unused `seq_length == 130` block (it has no effect on predictions) while keeping I/O paths and submission schema identical. The script still run end-to-end and write a valid `submission.csv`.'
- What this solution (achieved 0.47602) has done: 'You’re already using the best “minimal-core-logic” improvement for a constant baseline (global mean over all scored positions), so to move closer to the much lower target we need a small, legitimate modeling step without changing the overall pipeline shape. The smallest effective upgrade is to keep the same per-row submission construction but replace the constant fill with an OLS linear regression on simple sequence/structure/loop one-hot counts (still fast, deterministic, and uses only sklearn). Because only 3 targets are scored, we fit models for those and keep the other 2 as global means to avoid overfitting/instability while still improving MCRMSE. We also correctly train on the first 68 scored positions (flattened) so the training objective matches the metric’s equal-per-position weighting.'
- What this solution (achieved 0.42861) has done: 'We keep your Ridge-on-simple-counts approach intact but make it better match the per-position scoring by training on position-level features instead of repeating the same id-level vector for all 68 positions. Concretely, we expand features to include per-base one-hots for sequence/structure/loop at each position plus a normalized position index, then fit the same Ridge models for the three scored targets. This is a minimal, legitimate modeling upgrade (same model family, same training loop shape) that should reduce MCRMSE from 0.476 toward your 0.3519 target. We also clip predictions to a reasonable range derived from training quantiles to avoid occasional extreme values that can inflate RMSE, without changing the overall semantics of “predict continuous values”.'
- What this solution (achieved 0.42858) has done: 'You’re already using a position-level Ridge with simple one-hots, so the most “minimal but real” lever left is to make training better match the leaderboard distribution: train only on high-quality `SN_filter==1` rows (the test set is similarly filtered), and use `signal_to_noise` as sample weights so noisier measurements contribute less to the fit. This keeps the same model family (Ridge), same features, and same prediction construction, but should reduce MCRMSE (lower is better) from 0.42861 toward your 0.3519 target. I keep the same clipping strategy and global-mean fallback, but compute global means/clip bounds from the same filtered/weighted training subset for consistency.'
- What this solution (achieved 0.42862) has done: 'We keep your exact Ridge + per-position one-hot feature approach, but tune it minimally to reduce MCRMSE (lower is better) from 0.42858 toward 0.35188. The smallest likely win is to (1) include only scored positions (0–67) **and** drop NaN targets consistently when computing weights, and (2) slightly increase Ridge regularization (alpha) to improve generalization on the filtered test distribution without changing the model family or features. We also compute clipping bounds a bit more conservatively (0.5%–99.5%) from the same filtered subset to reduce RMSE inflation from tail errors while keeping the same “clip predictions to training-derived quantiles” semantics. Output format and paths stay identical and we still write a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing
from sklearn.linear_model import Ridge, MultiTaskRidge



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_11/2658855273.py in <cell line: 0>()
      1 import numpy as np  # linear algebra
      2 import pandas as pd  # data processing
----> 3 from sklearn.linear_model import Ridge, MultiTaskRidge
      4 

ImportError: cannot import name 'MultiTaskRidge' from 'sklearn.linear_model' (/usr/local/lib/python3.11/dist-packages/sklearn/linear_model/__init__.py)

## === cell 1
df_test = pd.read_json("../input/stanford-covid-vaccine/test.json", lines=True)
df_train = pd.read_json("../input/stanford-covid-vaccine/train.json", lines=True)

df_sub = pd.read_csv("../input/stanford-covid-vaccine/sample_submission.csv")

required_cols = [
    "id_seqpos",
    "reactivity",
    "deg_Mg_pH10",
    "deg_pH10",
    "deg_Mg_50C",
    "deg_50C",
]
missing = [c for c in required_cols if c not in df_sub.columns]
if missing:
    raise ValueError(f"sample_submission.csv missing required columns: {missing}")

target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
scored_cols = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
SEQ_SCORED = 68



## === cell 2
SEQ_VOCAB = ["A", "C", "G", "U"]
STRUCT_VOCAB = ["(", ")", "."]
LOOP_VOCAB = ["S", "M", "I", "B", "H", "E", "X"]

seq_to_i = {c: i for i, c in enumerate(SEQ_VOCAB)}
struct_to_i = {c: i for i, c in enumerate(STRUCT_VOCAB)}
loop_to_i = {c: i for i, c in enumerate(LOOP_VOCAB)}


def featurize_positions(row, n_pos=SEQ_SCORED):
    seq = row["sequence"]
    struct = row["structure"]
    loop = row["predicted_loop_type"]

    seq_scored = seq[:n_pos]
    struct_scored = struct[:n_pos]
    loop_scored = loop[:n_pos]

    X = np.zeros(
        (n_pos, 1 + len(SEQ_VOCAB) + len(STRUCT_VOCAB) + len(LOOP_VOCAB)),
        dtype=np.float64,
    )

    denom = float(max(n_pos - 1, 1))
    X[:, 0] = np.arange(n_pos, dtype=np.float64) / denom

    for p in range(n_pos):
        s = seq_scored[p] if p < len(seq_scored) else None
        st = struct_scored[p] if p < len(struct_scored) else None
        lp = loop_scored[p] if p < len(loop_scored) else None

        if s in seq_to_i:
            X[p, 1 + seq_to_i[s]] = 1.0
        if st in struct_to_i:
            X[p, 1 + len(SEQ_VOCAB) + struct_to_i[st]] = 1.0
        if lp in loop_to_i:
            X[p, 1 + len(SEQ_VOCAB) + len(STRUCT_VOCAB) + loop_to_i[lp]] = 1.0

    return X


X_train_pos = np.vstack(
    [featurize_positions(r, SEQ_SCORED) for _, r in df_train.iterrows()]
)
X_test_pos = np.vstack(
    [featurize_positions(r, SEQ_SCORED) for _, r in df_test.iterrows()]
)




## === cell 3
def flatten_targets(df, col):
    ys = []
    for arr in df[col].values:
        if isinstance(arr, (list, np.ndarray)) and len(arr):
            a = np.asarray(arr, dtype=np.float64)[:SEQ_SCORED]
        else:
            a = np.full((SEQ_SCORED,), np.nan, dtype=np.float64)
        ys.append(a)
    y = np.concatenate(ys, axis=0)
    return y


train_keep = df_train["SN_filter"].astype(int).values == 1
keep_idx = np.flatnonzero(train_keep)

sn_keep = df_train.loc[train_keep, "signal_to_noise"].astype(np.float64).values
sn_keep = np.nan_to_num(sn_keep, nan=0.0, posinf=0.0, neginf=0.0)
sn_keep = np.clip(sn_keep, 0.0, 10.0)
w_pos_keep_full = np.repeat(sn_keep, SEQ_SCORED).astype(np.float64)

X_train_pos_keep = X_train_pos.reshape(len(df_train), SEQ_SCORED, -1)[keep_idx].reshape(
    -1, X_train_pos.shape[1]
)

models = {}
global_means = {}
clip_bounds = {}

CLIP_Q_LOW, CLIP_Q_HIGH = 0.005, 0.995

for c in target_cols:
    y_flat_all = flatten_targets(df_train.loc[train_keep], c)
    global_means[c] = float(np.nanmean(y_flat_all))

finite_pool = []
for c in scored_cols:
    y_flat_all_sc = flatten_targets(df_train.loc[train_keep], c)
    finite_pool.append(y_flat_all_sc[np.isfinite(y_flat_all_sc)])
finite_pool = (
    np.concatenate(finite_pool) if len(finite_pool) else np.array([], dtype=np.float64)
)

if finite_pool.size:
    pooled_lo = float(np.quantile(finite_pool, CLIP_Q_LOW))
    pooled_hi = float(np.quantile(finite_pool, CLIP_Q_HIGH))
else:
    pooled_lo, pooled_hi = -1.0, 1.0

for c in scored_cols:
    clip_bounds[c] = (pooled_lo, pooled_hi)
for c in target_cols:
    if c not in clip_bounds:
        clip_bounds[c] = (-np.inf, np.inf)

alpha = 3.0

Y_list = []
mask_list = []
for c in scored_cols:
    y_flat = flatten_targets(df_train.loc[train_keep], c)
    m = np.isfinite(y_flat)
    Y_list.append(y_flat)
    mask_list.append(m)

mask_all = mask_list[0] & mask_list[1] & mask_list[2]

X_mt = X_train_pos_keep[mask_all]
Y_mt = np.vstack(
    [Y_list[i][mask_all] for i in range(len(scored_cols))]
).T  # (n_samples, 3)
w_mt = w_pos_keep_full[mask_all]

mt_model = MultiTaskRidge(alpha=alpha, random_state=0)
mt_model.fit(X_mt, Y_mt, sample_weight=w_mt)
models["__multitask_scored__"] = mt_model

models.keys(), global_means, clip_bounds



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2999586643.py in <cell line: 0>()
     78 w_mt = w_pos_keep_full[mask_all]
     79 
---> 80 mt_model = MultiTaskRidge(alpha=alpha, random_state=0)
     81 mt_model.fit(X_mt, Y_mt, sample_weight=w_mt)
     82 models["__multitask_scored__"] = mt_model

NameError: name 'MultiTaskRidge' is not defined

## === cell 4
test_ids = df_test["id"].tolist()
test_seq_length = df_test["seq_length"].astype(int).tolist()  # expected 107 for all
n_test = len(df_test)

p_mt = (
    models["__multitask_scored__"].predict(X_test_pos).astype(np.float64)
)  # (n_test*68, 3)

preds_scored = {}
for j, c in enumerate(scored_cols):
    lo, hi = clip_bounds[c]
    p_flat = np.clip(p_mt[:, j], lo, hi)
    preds_scored[c] = p_flat.reshape(n_test, SEQ_SCORED)

rows = []
for i, rid in enumerate(test_ids):
    L = int(test_seq_length[i])
    for pos in range(L):
        out = {"id_seqpos": f"{rid}_{pos}"}
        for c in target_cols:
            if c in scored_cols and pos < SEQ_SCORED:
                val = float(preds_scored[c][i, pos])
            else:
                val = float(global_means[c])
            out[c] = val
        rows.append(out)

df_pred = pd.DataFrame(rows)

df_out = df_sub[["id_seqpos"]].merge(df_pred, on="id_seqpos", how="left")

for c in target_cols:
    df_out[c] = df_out[c].astype(np.float64)
    df_out[c] = df_out[c].fillna(global_means[c])



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/5900777.py in <cell line: 0>()
      5 # Predict scored targets jointly then split back into per-target arrays
      6 p_mt = (
----> 7     models["__multitask_scored__"].predict(X_test_pos).astype(np.float64)
      8 )  # (n_test*68, 3)
      9 

KeyError: '__multitask_scored__'

## === cell 5
df_out = df_out[required_cols]
df_out.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", df_out.shape)
print(df_out.head())
print("Used MultiTaskRidge for scored targets:", scored_cols)
print("Ridge alpha:", alpha)
print("Global means (fallback / unscored targets):", global_means)
print(
    f"Pooled clip bounds ({CLIP_Q_LOW*100:.1f}%-{CLIP_Q_HIGH*100:.1f}%) for scored targets:",
    (pooled_lo, pooled_hi),
)

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3295107402.py in <cell line: 0>()
----> 1 df_out = df_out[required_cols]
      2 df_out.to_csv("submission.csv", index=False)
      3 print("Wrote submission.csv with shape:", df_out.shape)
      4 print(df_out.head())
      5 print("Used MultiTaskRidge for scored targets:", scored_cols)

NameError: name 'df_out' is not defined
