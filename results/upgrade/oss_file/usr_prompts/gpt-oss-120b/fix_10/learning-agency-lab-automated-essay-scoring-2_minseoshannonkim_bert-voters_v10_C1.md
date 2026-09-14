# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

datasets==4.4.1
geopandas==0.14.4
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
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sentence-transformers==4.1.0
sklearn-pandas==2.2.0
tensorflow-datasets==4.9.9
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
vega-datasets==0.9.0

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

# 5. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np

from sklearnex import patch_sklearn

patch_sklearn()  # Must be called before importing sklearn classes

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score
from scipy.sparse import hstack



## === cell 1
base_path = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2"
train_path = os.path.join(base_path, "train.csv")
test_path = os.path.join(base_path, "test.csv")

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)



## === cell 2
word_vectorizer = TfidfVectorizer(
    max_features=250000,
    ngram_range=(1, 2),
    stop_words="english",
    sublinear_tf=True,
    dtype=np.float32,
)

char_vectorizer = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(3, 5),
    max_features=50000,
    sublinear_tf=True,
    dtype=np.float32,
)

X_full_word = word_vectorizer.fit_transform(train_df["full_text"])
X_full_char = char_vectorizer.fit_transform(train_df["full_text"])
X_full = hstack([X_full_word, X_full_char])
y_full = train_df["score"].values

_, valid_idx = train_test_split(
    np.arange(len(train_df)),
    test_size=0.2,
    random_state=42,
    stratify=train_df["score"],
)



## === cell 3
clf = LogisticRegression(
    max_iter=1000,
    n_jobs=-1,
    multi_class="multinomial",
    solver="sag",
    C=6.0,
    class_weight="balanced",
    random_state=42,
    warm_start=True,
)

clf.fit(X_full, y_full)

proba_valid = clf.predict_proba(X_full[valid_idx])
classes = np.arange(1, 7)  # possible scores 1‑6
expected_valid = np.rint(proba_valid @ classes).astype(int)
expected_valid = np.clip(expected_valid, 1, 6)

kappa = cohen_kappa_score(
    train_df["score"].values[valid_idx], expected_valid, weights="quadratic"
)
print(f"Validation QWK (probability‑based rounding): {kappa:.6f}")

del X_full_word, X_full_char, X_full, y_full, proba_valid, expected_valid



## === cell 4
pass



## === cell 5
X_test_word = word_vectorizer.transform(test_df["full_text"])
X_test_char = char_vectorizer.transform(test_df["full_text"])
X_test = hstack([X_test_word, X_test_char])

del X_test_word, X_test_char



## === cell 6
proba_test = clf.predict_proba(X_test)
expected_test = np.rint(proba_test @ classes).astype(int)
expected_test = np.clip(expected_test, 1, 6)

submission = pd.DataFrame({"essay_id": test_df["essay_id"], "score": expected_test})
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission file written to {submission_path}")
