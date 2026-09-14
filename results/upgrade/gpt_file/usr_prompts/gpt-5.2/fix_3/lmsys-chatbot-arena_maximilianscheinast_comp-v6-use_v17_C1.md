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
Predict which chatbot response a user will prefer in a competition between two chatbots.

## Metric
Log loss with "eps=auto"

## Submission Format
For each id in the test set, you must predict the probability for each target class. The file should contain a header and have the following format:

```
 id,winner_model_a,winner_model_b,winner_tie
 136060,0.33,0,33,0.33
 211333,0.33,0,33,0.33
 1233961,0.33,0,33,0.33
 etc
```

## Dataset
**train.csv**

- `id` - A unique identifier for the row.
- `model_[a/b]` - The identity of model_[a/b]. Included in train.csv but not test.csv.
- `prompt` - The prompt that was given as an input (to both models).
- `response_[a/b]` - The response from model_[a/b] to the given prompt.
- `winner_model_[a/b/tie]` - Binary columns marking the judge's selection. The ground truth target column.

**test.csv**

- `id`
- `prompt`
- `response_[a/b]`

**sample_submission.csv** A submission file in the correct format.

- `id`
- `winner_model_[a/b/tie]` - This is what is predicted from the test set.

# 2. Python version

3.12

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (98 lines)
            sample_submission.csv (5749 lines)
            sample_submission.csv.zip (38.3 kB)
            test.csv (5749 lines)
            test.csv.zip (5.9 MB)
            train.csv (51730 lines)
            train.csv.zip (53.4 MB)
            lmsys-chatbot-arena/
                description.md (98 lines)
                sample_submission.csv (5749 lines)
                ... and 5 other files
                lmsys-chatbot-arena/
        input/
            description.md (98 lines)
            sample_submission.csv (5749 lines)
            sample_submission.csv.zip (38.3 kB)
            test.csv (5749 lines)
            test.csv.zip (5.9 MB)
            train.csv (51730 lines)
            train.csv.zip (53.4 MB)
            lmsys-chatbot-arena/
                description.md (98 lines)
                sample_submission.csv (5749 lines)
                ... and 5 other files
                lmsys-chatbot-arena/
        working/
            lmsys-chatbot-arena/
                description.md (98 lines)
                sample_submission.csv (5749 lines)
                ... and 5 other files
                lmsys-chatbot-arena/
```

-> data/lmsys-chatbot-arena/sample_submission.csv has 5748 rows and 4 columns.
The columns are: id, winner_model_a, winner_model_b, winner_tie

-> data/lmsys-chatbot-arena/test.csv has 5748 rows and 4 columns.
The columns are: id, prompt, response_a, response_b

-> data/lmsys-chatbot-arena/train.csv has 51729 rows and 9 columns.
The columns are: id, model_a, model_b, prompt, response_a, response_b, winner_model_a, winner_model_b, winner_tie

-> data/sample_submission.csv has 5748 rows and 4 columns.
The columns are: id, winner_model_a, winner_model_b, winner_tie

-> data/test.csv has 5748 rows and 4 columns.
The columns are: id, prompt, response_a, response_b

-> data/train.csv has 51729 rows and 9 columns.
The columns are: id, model_a, model_b, prompt, response_a, response_b, winner_model_a, winner_model_b, winner_tie

-> (stopped after 10 files for performance)

# 5. Target score

1.1029188742467997

# 6. Current score

4.64087

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

- What this solution (achieved 4.64087) has done: 'I remove the TensorFlow dependency (it fails to import in this Kaggle Python 3.12 image and the referenced model file path is also missing), and replace the missing model inference with a small deterministic baseline that produces properly normalized class probabilities. This make the notebook run end-to-end and always write a valid `submission.csv` with the required columns and row count. The baseline uses only the provided text fields and simple token-overlap features, so it stays within the “no extra packages” constraint and is stable/reproducible. Since no valid score has been produced yet, the priority is to generate a correct submission file rather than tuning.'

# 9. Code solution

## === cell 0
import os
import re
import numpy as np
import pandas as pd

SEED = 42
np.random.seed(SEED)

DATA_DIR = "/kaggle/input/lmsys-chatbot-arena"
TEST_PATH = os.path.join(DATA_DIR, "test.csv")

print("Using DATA_DIR:", DATA_DIR)
print("Reading:", TEST_PATH)



## === cell 1
test_data = pd.read_csv(TEST_PATH)

required_cols = {"id", "prompt", "response_a", "response_b"}
missing = required_cols - set(test_data.columns)
if missing:
    raise ValueError(f"Missing required columns in test.csv: {missing}")

test_data["prompt"] = test_data["prompt"].fillna("").astype(str)
test_data["response_a"] = test_data["response_a"].fillna("").astype(str)
test_data["response_b"] = test_data["response_b"].fillna("").astype(str)

print("test_data shape:", test_data.shape)
test_data.head()



## === cell 2
_token_re = re.compile(r"[a-z0-9]+")


def tokenize(s: str):
    return _token_re.findall(s.lower())


def softmax3(logits_3):
    x = np.asarray(logits_3, dtype=np.float64)
    x = x - np.max(x, axis=1, keepdims=True)
    ex = np.exp(x)
    return (ex / np.sum(ex, axis=1, keepdims=True)).astype(np.float32)


def baseline_predict_proba(df: pd.DataFrame):
    """
    Deterministic baseline:
    - compute token overlap between prompt and each response
    - add mild length priors
    - convert to 3-class probabilities: A, B, Tie via softmax
    """
    n = len(df)
    logits = np.zeros((n, 3), dtype=np.float64)

    for i, (p, a, b) in enumerate(
        zip(df["prompt"], df["response_a"], df["response_b"])
    ):
        tp = tokenize(p)
        ta = tokenize(a)
        tb = tokenize(b)

        sp = set(tp)
        sa = set(ta)
        sb = set(tb)

        overlap_a = len(sp & sa)
        overlap_b = len(sp & sb)

        len_a = len(ta)
        len_b = len(tb)

        score_a = overlap_a + 0.02 * np.sqrt(len_a)
        score_b = overlap_b + 0.02 * np.sqrt(len_b)

        closeness = -abs(score_a - score_b)
        both_relevant = 0.2 * (1.0 if (overlap_a > 0 and overlap_b > 0) else 0.0)
        tie_score = closeness + both_relevant

        logits[i, 0] = 0.6 * score_a
        logits[i, 1] = 0.6 * score_b
        logits[i, 2] = 0.9 * tie_score

    proba = softmax3(logits)

    proba = np.nan_to_num(proba, nan=1 / 3, posinf=1 / 3, neginf=1 / 3).astype(
        np.float32
    )
    row_sums = proba.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0.0, 1.0, row_sums)
    proba = (proba / row_sums).astype(np.float32)
    return proba


predictions = baseline_predict_proba(test_data)
print("Predictions shape:", predictions.shape)
print(
    "Row sums (min/max):", predictions.sum(axis=1).min(), predictions.sum(axis=1).max()
)



## === cell 3
submission = pd.DataFrame(
    {
        "id": test_data["id"].values,
        "winner_model_a": predictions[:, 0],
        "winner_model_b": predictions[:, 1],
        "winner_tie": predictions[:, 2],
    }
)

sample_path = os.path.join(DATA_DIR, "sample_submission.csv")
if os.path.exists(sample_path):
    sample = pd.read_csv(sample_path)
    expected_cols = list(sample.columns)
    if list(submission.columns) != expected_cols:
        submission = submission[expected_cols]
    if len(submission) != len(sample):
        raise ValueError(
            f"Row count mismatch vs sample_submission: {len(submission)} vs {len(sample)}"
        )

submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())
print("submission.csv saved at:", os.path.abspath("submission.csv"))
