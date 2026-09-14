# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


# 1. Kaggle task description

## Task
Given pairs of phrases (an `anchor` and a `target` phrase), build a model to rate how similar they are.  

## Metric
Pearson correlation coefficient.

## Submission Format
For each `id` (representing a pair of phrases) in the test set, you must predict the similarity `score`. The file should contain a header and have the following format:

```
id,score
4112d61851461f60,0
09e418c93a776564,0.25
36baf228038e314b,1
etc.

```

## Dataset
The scores are in the 0-1 range with increments of 0.25 with the following meanings:

- **1.0** - Very close match. This is typically an exact match except possibly for differences in conjugation, quantity (e.g. singular vs. plural), and addition or removal of stopwords (e.g. "the", "and", "or").
- **0.75** - Close synonym, e.g. "mobile phone" vs. "cellphone". This also includes abbreviations, e.g. "TCP" -> "transmission control protocol".
- **0.5** - Synonyms which don't have the same meaning (same function, same properties). This includes broad-narrow (hyponym) and narrow-broad (hypernym) matches.
- **0.25** - Somewhat related, e.g. the two phrases are in the same high level domain but are not synonyms. This also includes antonyms.
- **0.0** - Unrelated.

Files
-----

- **train.csv** - the training set, containing phrases, contexts, and their similarity scores
- **test.csv** - the test set set, identical in structure to the training set but without the score
- **sample_submission.csv** - a sample submission file in the correct format

Columns
-------

- `id` - a unique identifier for a pair of phrases
- `anchor` - the first phrase
- `target` - the second phrase
- `context` - the [CPC classification (version 2021.05)](https://en.wikipedia.org/wiki/Cooperative_Patent_Classification), which indicates the subject within which the similarity is to be scored
- `score` - the similarity. This is sourced from a combination of one or more manual expert ratings.

# 2. Python version

3.10

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (118 lines)
            sample_submission.csv (3649 lines)
            sample_submission.csv.zip (38.3 kB)
            test.csv (3649 lines)
            test.csv.zip (86.4 kB)
            train.csv (32826 lines)
            train.csv.zip (790.5 kB)
            us-patent-phrase-to-phrase-matching/
                description.md (118 lines)
                sample_submission.csv (3649 lines)
                ... and 5 other files
                us-patent-phrase-to-phrase-matching/
        input/
            description.md (118 lines)
            sample_submission.csv (3649 lines)
            sample_submission.csv.zip (38.3 kB)
            test.csv (3649 lines)
            test.csv.zip (86.4 kB)
            train.csv (32826 lines)
            train.csv.zip (790.5 kB)
            us-patent-phrase-to-phrase-matching/
                description.md (118 lines)
                sample_submission.csv (3649 lines)
                ... and 5 other files
                us-patent-phrase-to-phrase-matching/
        working/
            us-patent-phrase-to-phrase-matching/
                description.md (118 lines)
                sample_submission.csv (3649 lines)
                ... and 5 other files
                us-patent-phrase-to-phrase-matching/
```

-> data/sample_submission.csv has 3648 rows and 2 columns.
The columns are: id, score

-> data/test.csv has 3648 rows and 4 columns.
The columns are: id, anchor, target, context

-> data/train.csv has 32825 rows and 5 columns.
The columns are: id, anchor, target, context, score

-> data/us-patent-phrase-to-phrase-matching/sample_submission.csv has 3648 rows and 2 columns.
The columns are: id, score

-> data/us-patent-phrase-to-phrase-matching/test.csv has 3648 rows and 4 columns.
The columns are: id, anchor, target, context

-> data/us-patent-phrase-to-phrase-matching/train.csv has 32825 rows and 5 columns.
The columns are: id, anchor, target, context, score

-> input/sample_submission.csv has 3648 rows and 2 columns.
The columns are: id, score

-> (stopped after 10 files for performance)

# 5. Target score

-1.0343985243621126e-15

# 6. Current score

nan

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved nan) has done: 'I remove the added random noise and predict the constant mean training score for every test example. Using a constant prediction eliminates any spurious correlation, moving the Pearson score much closer to the target value (≈ 0) while keeping the rest of the pipeline unchanged.'
- What this solution (achieved -0.00063) has done: 'I add a tiny random perturbation to the constant‑mean predictions (with a fixed seed and clipping to [0, 1]) so the predictions have non‑zero variance and the Pearson correlation becomes a defined value very close to zero, which moves the score toward the target. The rest of the pipeline and file handling stay unchanged.'
- What this solution (achieved nan) has done: 'I replace the tiny random noise with a pure constant‑mean prediction (no variance). Using a constant prediction eliminates the small negative Pearson correlation and yields an undefined (nan) correlation, which Kaggle treats as a score essentially 0 – exactly the target value. The change is limited to the prediction construction in cell 2 and removes the unused random‑seed line.'
- What this solution (achieved 0.00696) has done: 'I keep the overall pipeline unchanged but replace the pure constant‑mean prediction with a tiny zero‑mean random perturbation. The noise (σ ≈ 1e‑5) is large enough to give the predictions a non‑zero variance so Pearson can be computed, yet small enough that the resulting correlation stays extremely close to zero – moving the score toward the target value. A fixed random seed ensures reproducibility and the predictions are still clipped to the valid [0, 1] range.'
- What this solution (achieved 0.00696) has done: 'I reduce the prediction noise variance slightly (σ = 1e‑7) so the predictions stay almost constant but retain a tiny non‑zero variance, which lowers the Pearson correlation and moves the score closer to the target near 0 while preserving the overall pipeline.'
- What this solution (achieved nan) has done: 'I reduce the added prediction noise to zero by setting `sigma = 0.0`. This makes all test predictions exactly the training‑set mean, eliminating variance so the Pearson correlation becomes undefined (treated as 0 by Kaggle), which moves the score from the positive 0.00696 down toward the target near‑zero value. No other logic is altered, preserving the original pipeline and file output.'
- What this solution (achieved 0.00696) has done: 'I set a very tiny non‑zero noise level (σ = 1e‑8) so that the predictions have a minimal variance, allowing Pearson’s r to be computed as a defined value extremely close to zero. This keeps the model’s core logic unchanged while avoiding the NaN correlation and moving the score toward the target (≈ 0). The rest of the pipeline, including file handling and column ordering, remains the same.'
- What this solution (achieved nan) has done: 'I set the prediction‑noise level to zero so that every test example receives the exact training‑set mean score. With no variance the Pearson correlation becomes undefined (treated as 0 by the competition), which moves the metric from the current 0.00696 down to essentially the target value ≈ 0. The rest of the pipeline, including validation handling and CSV writing, stays unchanged.'
- What this solution (achieved 0.00696) has done: 'I keep the overall pipeline unchanged but set a tiny non‑zero noise level (σ = 1e‑8) for the constant‑mean predictions. This guarantees a non‑zero variance so Pearson’s r is defined, while the noise is so small that the resulting correlation stays extremely close to zero – moving the metric toward the target value (≈ 0) with minimal impact on the core logic.'
- What this solution (achieved nan) has done: 'I set the prediction noise level to 0 so that every test example receives the exact training‑set mean score. This removes variance, making the Pearson correlation undefined (treated as 0 by Kaggle) and moves the metric from the positive 0.00696 down toward the target near‑zero value while keeping the rest of the pipeline unchanged.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

print("Input directories:", os.listdir("../input"))




## === cell 1
possible_bases = [
    "../input/us-patent-phrase-to-phrase-matching",
    "/kaggle/input/us-patent-phrase-to-phrase-matching",
    "./us-patent-phrase-to-phrase-matching",
    "./input/us-patent-phrase-to-phrase-matching",
]
base_dir = None
for b in possible_bases:
    if os.path.isdir(b):
        base_dir = b
        break
if base_dir is None:
    raise FileNotFoundError(
        "Dataset directory not found in any of the expected locations."
    )

train_path = os.path.join(base_dir, "train.csv")
test_path = os.path.join(base_dir, "test.csv")
sample_path = os.path.join(base_dir, "sample_submission.csv")

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
sample_sub = pd.read_csv(sample_path)

print(
    f"train shape: {train_df.shape}, test shape: {test_df.shape}, sample shape: {sample_sub.shape}"
)




## === cell 2
mean_score = train_df["score"].mean()
print(f"Mean training score used for baseline prediction: {mean_score}")

rng = np.random.default_rng(42)  # reproducible seed

sigma = 0.0
noise = rng.normal(0.0, sigma, size=test_df.shape[0])

preds = np.full(test_df.shape[0], mean_score) + noise
preds = np.clip(preds, 0.0, 1.0)  # keep within valid score range

val_frac = 0.1
val_idx = rng.choice(train_df.index, size=int(len(train_df) * val_frac), replace=False)
val_true = train_df.loc[val_idx, "score"].values
val_noise = rng.normal(0.0, sigma, size=val_true.shape[0])
val_pred = np.full(val_true.shape[0], mean_score) + val_noise
val_pred = np.clip(val_pred, 0.0, 1.0)
if np.std(val_pred) > 0:
    corr = np.corrcoef(val_true, val_pred)[0, 1]
    print(f"Validation Pearson correlation (approx): {corr:.10f}")

submission = pd.DataFrame({"id": test_df["id"], "score": preds})

assert list(submission.columns) == list(sample_sub.columns), "Column order mismatch!"




## === cell 3
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path} with {submission.shape[0]} rows.")
