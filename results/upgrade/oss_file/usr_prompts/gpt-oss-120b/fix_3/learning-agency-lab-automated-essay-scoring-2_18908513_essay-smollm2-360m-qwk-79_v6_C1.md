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

3.13

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

0.7797599670654481

# 6. Current score

0.69654

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.6191) has done: 'I replace the failing model‑loading code with a simple but effective TF‑IDF + LogisticRegression pipeline that trains on the provided train.csv, evaluates Quadratic Weighted Kappa on a validation split, and finally produces a valid submission.csv with the required columns. This fixes the HFValidationError and ensures a realistic score that moves toward the target.'
- What this solution (achieved 0.69654) has done: 'I increased the expressive power of the TF‑IDF representation (more features, trigram support, sublinear term frequency) and relaxed the regularisation of the logistic regression while adding balanced class weights. These modest hyper‑parameter tweaks stay within the original TF‑IDF + LogisticRegression pipeline but should boost validation QWK, moving the score nearer the target. The rest of the workflow—including the train/validation split, full‑data training, test prediction, clipping, and CSV output—remains unchanged.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score
import warnings

warnings.filterwarnings("ignore")



## === cell 1
train_path = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/train.csv"
test_path = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

assert "full_text" in train_df.columns and "score" in train_df.columns
assert "full_text" in test_df.columns

train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_df["full_text"].astype(str).tolist(),
    train_df["score"].astype(int).tolist(),
    test_size=0.2,
    random_state=42,
    stratify=train_df["score"],
)



## === cell 2
vectorizer = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1, 3),
    stop_words="english",
    sublinear_tf=True,
)

X_train = vectorizer.fit_transform(train_texts)
X_val = vectorizer.transform(val_texts)

clf = LogisticRegression(
    max_iter=500,
    n_jobs=-1,
    multi_class="multinomial",
    solver="lbfgs",
    C=4.0,
    class_weight="balanced",
)
clf.fit(X_train, train_labels)

val_pred = clf.predict(X_val)
qwk = cohen_kappa_score(val_labels, val_pred, weights="quadratic")
print(f"Validation Quadratic Weighted Kappa: {qwk:.5f}")



## === cell 3
X_full = vectorizer.fit_transform(train_df["full_text"].astype(str))
clf_full = LogisticRegression(
    max_iter=500,
    n_jobs=-1,
    multi_class="multinomial",
    solver="lbfgs",
    C=4.0,
    class_weight="balanced",
)
clf_full.fit(X_full, train_df["score"].astype(int))

X_test = vectorizer.transform(test_df["full_text"].astype(str))
test_pred = clf_full.predict(X_test)

test_pred = np.clip(test_pred, 1, 6)

submission = pd.DataFrame({"essay_id": test_df["essay_id"], "score": test_pred})
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)

print(f"Predictions saved to '{submission_path}'")
