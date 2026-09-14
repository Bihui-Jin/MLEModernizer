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
Predict the score of student essays.

## Metric
Quadratic weighted kappa.

## Submission Format
For each `essay_id` in the test set, you must predict the corresponding `score` (between 1-6, see [rubric](https://storage.googleapis.com/kaggle-forum-message-attachments/2733927/20538/Rubric_%20Holistic%20Essay%20Scoring.pdf) for more details). The file should contain a header and have the following format:

```
essay_id,score
000d118,3
000fe60,3
001ab80,4
...
```

## Dataset
- **train.csv** - Essays and scores to be used as training data.
    - `essay_id` - The unique ID of the essay
    - `full_text` - The full essay response
    - `score` - Holistic score of the essay on a 1-6 scale
- **test.csv** - The essays to be used as test data. Contains the same fields as `train.csv`, aside from exclusion of `score`.
- **sample_submission.csv** - A submission file in the correct format.
    - `essay_id` - The unique ID of the essay
    - `score` - The predicted holistic score of the essay on a 1-6 scale

# 2. Python version

3.12

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (153 lines)
            sample_submission.csv (1732 lines)
            sample_submission.csv.zip (9.4 kB)
            test.csv (15336 lines)
            test.csv.zip (1.2 MB)
            train.csv (139231 lines)
            train.csv.zip (11.0 MB)
            learning-agency-lab-automated-essay-scoring-2/
                description.md (153 lines)
                sample_submission.csv (1732 lines)
                ... and 5 other files
                learning-agency-lab-automated-essay-scoring-2/
        input/
            description.md (153 lines)
            sample_submission.csv (1732 lines)
            sample_submission.csv.zip (9.4 kB)
            test.csv (15336 lines)
            test.csv.zip (1.2 MB)
            train.csv (139231 lines)
            train.csv.zip (11.0 MB)
            learning-agency-lab-automated-essay-scoring-2/
                description.md (153 lines)
                sample_submission.csv (1732 lines)
                ... and 5 other files
                learning-agency-lab-automated-essay-scoring-2/
        working/
            learning-agency-lab-automated-essay-scoring-2/
                description.md (153 lines)
                sample_submission.csv (1732 lines)
                ... and 5 other files
                learning-agency-lab-automated-essay-scoring-2/
```

-> data/learning-agency-lab-automated-essay-scoring-2/sample_submission.csv has 1731 rows and 2 columns.
The columns are: essay_id, score

-> data/learning-agency-lab-automated-essay-scoring-2/test.csv has 15335 rows and 2 columns.
The columns are: essay_id, full_text

-> data/learning-agency-lab-automated-essay-scoring-2/train.csv has 139230 rows and 3 columns.
The columns are: essay_id, full_text, score

-> data/sample_submission.csv has 1731 rows and 2 columns.
The columns are: essay_id, score

-> data/test.csv has 15335 rows and 2 columns.
The columns are: essay_id, full_text

-> data/train.csv has 139230 rows and 3 columns.
The columns are: essay_id, full_text, score

-> (stopped after 10 files for performance)

# 5. Target score

0.8077619023103844

# 6. Current score

0.60809

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.64219) has done: 'I replace the failing transformer‑based pipeline with a lightweight, deterministic baseline that loads the CSV files directly, computes average scores per essay length bucket from the training data, and uses those averages to predict scores for the test set. This removes the protobuf‑related errors, avoids invalid model path handling, and guarantees that a `submission.csv` file with the correct columns is written, allowing the notebook to finish successfully while still providing reasonable predictions.'
- What this solution (achieved 0.61115) has done: 'I replace the simple mean‑per‑length‑bin rule with a more robust per‑bin mode (most frequent score) and add a small count‑threshold check so that rarely‑seen bins fall back to the overall mean. This keeps the overall deterministic, lightweight approach while giving predictions that better match the discrete scoring distribution, which should lift the quadratic weighted kappa toward the target.'
- What this solution (achieved 0.60809) has done: 'The plan is to make the length‑bin predictions a bit more granular and use the average score per bin (instead of the mode) while keeping the same fallback to the overall mean for rare bins. By decreasing the bin size from 50 to 30 words and applying a lower count threshold, we can capture finer score variations and hopefully raise the quadratic weighted kappa toward the target score.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np

DATA_ROOT = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2"
TRAIN_PATH = os.path.join(DATA_ROOT, "train.csv")
TEST_PATH = os.path.join(DATA_ROOT, "test.csv")
SUBMISSION_PATH = "submission.csv"



## === cell 1
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

assert {"essay_id", "full_text", "score"}.issubset(train_df.columns)
assert {"essay_id", "full_text"}.issubset(test_df.columns)



## === cell 2
train_df["word_cnt"] = train_df["full_text"].str.split().apply(len)
BIN_SIZE = 30  # finer granularity than the previous 50
train_df["len_bin"] = (train_df["word_cnt"] // BIN_SIZE).astype(int)

overall_mean = train_df["score"].mean()

bin_counts = train_df.groupby("len_bin").size().to_dict()
bin_mean_map = train_df.groupby("len_bin")["score"].mean().to_dict()



## === cell 3
test_df["word_cnt"] = test_df["full_text"].str.split().apply(len)
test_df["len_bin"] = (test_df["word_cnt"] // BIN_SIZE).astype(int)

MIN_BIN_COUNT = (
    20  # allow bins with fewer examples to still be used if they have enough data
)


def predict_score(row):
    bin_id = row["len_bin"]
    if bin_id in bin_mean_map and bin_counts.get(bin_id, 0) >= MIN_BIN_COUNT:
        score = bin_mean_map[bin_id]
    else:
        score = overall_mean
    return int(np.clip(round(score), 1, 6))


test_df["score"] = test_df.apply(predict_score, axis=1)



## === cell 4
submission_df = test_df[["essay_id", "score"]].copy()
submission_df.to_csv(SUBMISSION_PATH, index=False)

submission_df.head()
