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

geopandas==0.14.4
google-api-python-client==2.177.0
ipython==7.34.0
ipython-genutils==0.2.0
ipython_pygments_lexers==1.1.1
ipython-sql==0.5.0
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
sentence-transformers==4.1.0
sklearn-pandas==2.2.0
torch==2.6.0+cu124
torchao==0.10.0
torchaudio==2.6.0+cu124
torchdata==0.11.0
torchinfo==1.8.0
torchmetrics==1.8.2
torchsummary==1.5.1
torchtune==0.6.1
torchvision==0.21.0+cu124
transformers==4.53.3

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

0.233718277572957

# 6. Current score

0.27341

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I add a safe fallback that avoids the failing LLM pipeline by loading the training data, computing the overall mean score, and using this constant as the prediction for every essay. The pipeline creation is wrapped in a try/except so the script continues even if the model cannot be loaded. This fixes the NameError issues, ensures the “score” column is present, and produces a valid `submission.csv` that should achieve a baseline QWK close to the target.'
- What this solution (achieved 0.61194) has done: 'I remove the failing LLM pipeline creation and replace it with a lightweight heuristic that predicts scores based on essay length buckets derived from the training data. This fixes the runtime error, guarantees a valid `submission.csv`, and adds a simple data‑driven rule that should lift the quadratic weighted kappa toward the target without altering the core modeling approach.'
- What this solution (achieved 0.03226) has done: 'I slightly dampen the length‑bucket predictions by blending each bucket’s average score with the overall mean score (using a small weight α). This reduces the prediction variance, lowering the quadratic weighted kappa toward the target value while keeping the original heuristic and overall pipeline unchanged.'
- What this solution (achieved 0.57733) has done: 'I add the missing pandas import and the data‑loading steps, keep the existing length‑bucket heuristic unchanged, and then create a third cell that loads the test set, applies the `predict_score` function to each essay, and writes a properly formatted `submission.csv`. This fixes the NameError, guarantees a valid CSV with the required columns, and preserves the core logic while providing reasonable predictions that should achieve a score near the target.'
- What this solution (achieved 0.0032) has done: 'I lower the blending weight `alpha` so the prediction relies more on the overall mean score and less on the length‑bucket average. This reduces prediction variance, which lowers the quadratic weighted kappa and moves the score from the current 0.57733 toward the target ≈ 0.23 while keeping the original heuristic unchanged.'
- What this solution (achieved 0.31763) has done: 'I raise the blending weight `alpha` to give more influence to the length‑bucket averages (≈ 0.5) and keep the overall mean as a float (instead of an integer) so the blend is smoother. These minimal tweaks should increase the quadratic weighted kappa toward the target ≈ 0.23 while preserving the original heuristic.'
- What this solution (achieved 0.0032) has done: 'I lower the blending weight `alpha` from 0.5 to 0.2 so the prediction relies more on the overall mean and less on the length‑bucket averages. This reduces the variance of the predictions, which should bring the quadratic weighted kappa down from 0.317 toward the target ≈ 0.234 while keeping the original heuristic intact.'
- What this solution (achieved 0.27341) has done: 'I raise the blending weight `alpha` from 0.2 to 0.42 so the predictions rely more on the length‑bucket averages rather than the overall mean. This modest change is expected to increase the quadratic weighted kappa from the current 0.0032 toward the target 0.2337 while preserving the original heuristic and overall pipeline.'

# 9. Code solution

## === cell 0
import pandas as pd

train_path = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/train.csv"
train_df = pd.read_csv(train_path)

default_score = train_df["score"].mean()

train_df["word_count"] = train_df["full_text"].str.split().str.len()
train_df["bucket"] = (train_df["word_count"] // 50).astype(int)
bucket_score = train_df.groupby("bucket")["score"].mean().to_dict()

print(f"Default constant prediction (mean): {default_score:.4f}")
print(f"Number of length buckets (50‑word size): {len(bucket_score)}")




## === cell 1
alpha = 0.42  # blend: 42% bucket mean, 58% overall mean


def predict_score(text):
    """
    Predict essay score using a blended length‑bucket heuristic.
    Falls back to the overall mean if the bucket is unseen.
    """
    word_count = len(text.split())
    bucket = word_count // 50
    bucket_mean = bucket_score.get(bucket, default_score)
    blended_score = alpha * bucket_mean + (1 - alpha) * default_score
    score = int(round(blended_score))
    return max(1, min(6, score))




## === cell 2
test_path = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv"
test_df = pd.read_csv(test_path)

test_df["score"] = test_df["full_text"].apply(predict_score)

submission = test_df[["essay_id", "score"]]
submission.to_csv("submission.csv", index=False)

print("Submission file written to submission.csv")
