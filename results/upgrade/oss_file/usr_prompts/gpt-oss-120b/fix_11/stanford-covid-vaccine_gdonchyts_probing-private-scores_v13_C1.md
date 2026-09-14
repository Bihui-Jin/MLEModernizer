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

0.35405

# 6. Current score

0.41637

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.42418) has done: 'I correct the file paths so the script can locate the training and test JSON files in the Kaggle environment, add a fallback lookup, and restructure the notebook cells to load the data, compute per‑position means, build the required submission rows, and finally write a `submission.csv` file with the proper column order. These changes fix the runtime errors while preserving the original simple averaging model, ensuring a valid submission is produced.'
- What this solution (achieved 0.41641) has done: 'I replace the simple per‑position averaging with a per‑position + base (A,C,G,U) averaging scheme, which keeps the overall “mean‑based” approach but adds a modest, data‑driven adjustment that is expected to lower the MCRMSE toward the target. The rest of the pipeline (loading data, building rows, writing CSV) stays unchanged.'
- What this solution (achieved 0.41638) has done: 'I add a lightweight Bayesian shrinkage when computing the base‑specific means: instead of using raw averages (which can be noisy for rarely‑seen bases), I blend each base‑mean with the overall position mean using a small pseudocount α = 5. This keeps the original simple averaging approach while reducing variance, which should lower the MCRMSE toward the target. No other logic is changed.'
- What this solution (achieved 0.41625) has done: 'I fixed the shape‑mismatch when applying shrinkage by explicitly indexing the positions and targets for entries with zero counts, and kept the rest of the workflow unchanged so the script now runs through training‑validation split, selects the best α, builds predictions for every test position, and writes a correctly‑named `submission.csv` with all required columns.'
- What this solution (achieved 0.42482) has done: 'I add a lightweight positional smoothing step to the base‑specific mean matrix, which reduces noise without changing the overall averaging logic. The smoothing is applied after the Bayesian shrinkage calculation, keeping the core model intact while giving slightly more stable predictions. I also expand the α candidates a little to let the validation split pick a better regularisation strength. These minimal changes are expected to lower the MCRMSE toward the target score.'
- What this solution (achieved 0.4173) has done: 'I add a lightweight tuneable smoothing step after the Bayesian shrinkage, trying a few weight triples and selecting the best combination on the validation split. This keeps the original averaging‑based model while reducing variance, which should lower the MCRMSE toward the target.'
- What this solution (achieved 0.41637) has done: 'I expand the regularisation grid (add a finer α = 0.05 and a few stronger smoothing kernels) and use the average of the base‑specific means when a nucleotide is unknown, rather than falling back directly to the overall mean. These modest tweaks keep the averaging‑based core unchanged while giving the model a better bias‑variance trade‑off, which should lower the MCRMSE toward the target.'

# 9. Code solution

## === cell 0
import json
import pandas as pd
import numpy as np
from pathlib import Path


def resolve_path(rel_path: str) -> Path:
    p = Path(rel_path)
    if p.is_file():
        return p
    alt = Path("/kaggle/input/stanford-covid-vaccine") / Path(rel_path).name
    if alt.is_file():
        return alt
    fallback = Path("/kaggle/input") / Path(rel_path).name
    if fallback.is_file():
        return fallback
    raise FileNotFoundError(f"Unable to locate {rel_path}")


train_path = resolve_path("train.json")
test_path = resolve_path("test.json")



## === cell 1
train = pd.read_json(train_path, lines=True)
test = pd.read_json(test_path, lines=True)



## === cell 2
target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
base_to_idx = {"A": 0, "C": 1, "G": 2, "U": 3}
n_pos = 68
n_bases = 4
n_targets = len(target_cols)


def smooth_base_means(base_means: np.ndarray, weights: tuple) -> np.ndarray:
    """Apply 1‑D edge‑aware smoothing with the given three‑weight kernel."""
    w0, w1, w2 = weights
    padded = np.pad(base_means, ((1, 1), (0, 0), (0, 0)), mode="edge")
    return w0 * padded[:-2] + w1 * padded[1:-1] + w2 * padded[2:]


def compute_means(
    df: pd.DataFrame, alpha: float, smooth_weights: tuple = (0.25, 0.5, 0.25)
):
    """Return overall_means (pos×targets) and smoothed base_means (pos×base×targets)."""
    sums = np.zeros((n_pos, n_bases, n_targets), dtype=float)
    counts = np.zeros((n_pos, n_bases, n_targets), dtype=int)

    for _, row in df.iterrows():
        seq = row["sequence"]
        seq_scored = int(row["seq_scored"])
        for pos in range(seq_scored):
            base = seq[pos]
            if base not in base_to_idx:
                continue
            b_idx = base_to_idx[base]
            for t_idx, col in enumerate(target_cols):
                val = row[col][pos]
                sums[pos, b_idx, t_idx] += val
                counts[pos, b_idx, t_idx] += 1

    overall_means = np.zeros((n_pos, n_targets), dtype=float)
    for t_idx, col in enumerate(target_cols):
        stacked = np.vstack(df[col].values)  # shape (samples, n_pos)
        overall_means[:, t_idx] = stacked.mean(axis=0)

    overall_exp = overall_means[:, np.newaxis, :]  # (pos,1,targets)
    base_means = (sums + alpha * overall_exp) / (counts + alpha)

    mask = counts == 0
    if np.any(mask):
        pos_idx, _, t_idx = np.where(mask)
        base_means[pos_idx, :, t_idx] = overall_means[pos_idx, t_idx][:, np.newaxis]

    base_means = smooth_base_means(base_means, smooth_weights)

    return overall_means, base_means


def mcrmse(val_df: pd.DataFrame, overall_means: np.ndarray, base_means: np.ndarray):
    """Compute the Mean Columnwise RMSE for the three scored targets."""
    scored_idx = [
        target_cols.index("reactivity"),
        target_cols.index("deg_Mg_pH10"),
        target_cols.index("deg_pH10"),
    ]
    sq_err = np.zeros(len(scored_idx), dtype=float)
    cnt = np.zeros(len(scored_idx), dtype=int)

    base_means_avg = base_means.mean(axis=1)  # shape (pos, targets)

    for _, row in val_df.iterrows():
        seq = row["sequence"]
        seq_scored = int(row["seq_scored"])
        for pos in range(seq_scored):
            base = seq[pos]
            b_idx = base_to_idx.get(base, None)
            for i, t_idx in enumerate(scored_idx):
                true_val = row[target_cols[t_idx]][pos]
                if b_idx is not None:
                    pred = base_means[pos, b_idx, t_idx]
                else:
                    pred = base_means_avg[pos, t_idx]
                err = (true_val - pred) ** 2
                sq_err[i] += err
                cnt[i] += 1

    rmse = np.sqrt(sq_err / cnt)
    return rmse.mean()


val_df = train.sample(frac=0.2, random_state=42)
train_sub = train.drop(val_df.index)

candidate_alphas = [0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1, 2, 5, 10, 20, 50, 100]
candidate_smooth = [
    (0.25, 0.5, 0.25),  # original
    (0.2, 0.6, 0.2),
    (0.1, 0.8, 0.1),
    (0.33, 0.34, 0.33),
    (0.05, 0.9, 0.05),  # stronger centre weight
    (0.0, 1.0, 0.0),  # no smoothing (edge case)
]

best_alpha = candidate_alphas[0]
best_smooth = candidate_smooth[0]
best_score = np.inf

for a in candidate_alphas:
    for w in candidate_smooth:
        overall_means_a, base_means_a = compute_means(
            train_sub, alpha=a, smooth_weights=w
        )
        score = mcrmse(val_df, overall_means_a, base_means_a)
        if score < best_score:
            best_score = score
            best_alpha = a
            best_smooth = w

overall_means, base_means = compute_means(
    train, alpha=best_alpha, smooth_weights=best_smooth
)



## === cell 3
rows = []
base_means_avg = base_means.mean(axis=1)  # (pos, targets)

for _, row in test.iterrows():
    sample_id = row["id"]
    seq_len = int(row["seq_length"])
    seq_scored = int(row["seq_scored"])
    seq = row["sequence"]
    for pos in range(seq_len):
        id_seqpos = f"{sample_id}_{pos}"
        if pos < seq_scored:
            base = seq[pos]
            b_idx = base_to_idx.get(base, None)
            if b_idx is not None:
                values = [
                    base_means[pos, b_idx, t_idx] for t_idx in range(len(target_cols))
                ]
            else:
                values = base_means_avg[pos].tolist()
        else:
            values = overall_means[pos % n_pos].tolist()
        rows.append([id_seqpos] + values)

submission = pd.DataFrame(rows, columns=["id_seqpos"] + target_cols)



## === cell 4
submission_path = Path("submission.csv")
submission.to_csv(submission_path, index=False)
