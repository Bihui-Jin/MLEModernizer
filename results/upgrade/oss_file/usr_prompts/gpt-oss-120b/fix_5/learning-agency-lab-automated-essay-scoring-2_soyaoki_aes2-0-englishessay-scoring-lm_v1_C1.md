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
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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
tqdm==4.67.1
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

0.3194937486952924

# 6. Current score

0.5615

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.66108) has done: 'The fix replaces the unavailable HuggingFace model with a lightweight TF‑IDF + LinearRegression pipeline, removes the failing tokenization steps, and ensures the script creates a proper `submission.csv` containing `essay_id,score`. This restores end‑to‑end execution, produces valid predictions clipped to the 1‑6 range, and keeps the original linear‑regression‑based approach while staying within the required logic.'
- What this solution (achieved 0.23964) has done: 'I slightly under‑fit the predictions by blending the linear‑regression output with the overall mean score, which reduces variance and therefore lowers the quadratic weighted kappa toward the target value (while still keeping the original TF‑IDF + LinearRegression pipeline). This change is minimal, preserves all core logic, and ensures a valid `submission.csv` is written.'
- What this solution (achieved 0.5615) has done: 'I increase the blending weight `w` so the predictions rely more on the raw LinearRegression output (which tends to give a higher quadratic weighted kappa) while still keeping a modest pull toward the global mean. This small adjustment should raise the score toward the target without changing the core TF‑IDF + LinearRegression pipeline.'

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression




## === cell 1
train_path = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/train.csv"
test_path = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv"

df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)




## === cell 2
vectorizer = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1, 2),
    stop_words="english",
    dtype=np.float32,
)
X_train = vectorizer.fit_transform(df_train["full_text"])
X_test = vectorizer.transform(df_test["full_text"])

y_train = df_train["score"].astype(float).values  # target scores 1‑6




## === cell 3
lr = LinearRegression()
lr.fit(X_train, y_train)




## === cell 4
train_preds = lr.predict(X_train)
print("Train predictions sample:", train_preds[:5])




## === cell 5
test_preds_raw = lr.predict(X_test)

global_mean = y_train.mean()

w = 0.6  # previously 0.3
test_preds_blended = w * test_preds_raw + (1 - w) * global_mean

test_preds_clipped = np.clip(test_preds_blended, 1, 6)
test_preds_rounded = np.rint(test_preds_clipped).astype(int)




## === cell 6
submission = pd.DataFrame(
    {"essay_id": df_test["essay_id"], "score": test_preds_rounded}
)




## === cell 7
submission.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv, shape:", submission.shape)
