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

3.14

# 3. Installed packages

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

# 5. Code solution

## === cell 0
import os
import glob
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score


def find_file(pattern):
    matches = glob.glob(f"**/{pattern}", recursive=True)
    if not matches:
        raise FileNotFoundError(f"Unable to locate {pattern}")
    return matches[0]


TRAIN_PATH = find_file("train.csv")
TEST_PATH = find_file("test.csv")
OUTPUT_PATH = os.path.join(os.getcwd(), "submission_output")
os.makedirs(OUTPUT_PATH, exist_ok=True)




## === cell 1
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

assert {"essay_id", "full_text", "score"}.issubset(train_df.columns)
assert {"essay_id", "full_text"}.issubset(test_df.columns)

train_split, val_split = train_test_split(
    train_df,
    test_size=0.1,
    stratify=train_df["score"],
    random_state=42,
)


def build_vectorizers(max_features_word):
    word_vec = TfidfVectorizer(
        max_features=max_features_word,
        ngram_range=(1, 6),
        stop_words="english",
        dtype=np.float32,
        sublinear_tf=True,
        min_df=1,
    )
    char_vec = TfidfVectorizer(
        analyzer="char",
        max_features=200_000,
        ngram_range=(3, 5),
        dtype=np.float32,
        sublinear_tf=True,
        min_df=1,
    )
    return word_vec, char_vec


word_vectorizer, char_vectorizer = build_vectorizers(max_features_word=800_000)

X_word_train = word_vectorizer.fit_transform(train_split["full_text"])
X_char_train = char_vectorizer.fit_transform(train_split["full_text"])
X_train = hstack([X_word_train, X_char_train])

X_word_val = word_vectorizer.transform(val_split["full_text"])
X_char_val = char_vectorizer.transform(val_split["full_text"])
X_val = hstack([X_word_val, X_char_val])

y_train = train_split["score"].astype(int)
y_val = val_split["score"].astype(int)

model = LogisticRegression(
    max_iter=1000,
    n_jobs=5,
    multi_class="multinomial",
    solver="lbfgs",
    class_weight="balanced",
    C=5.0,  # reduced regularisation to improve generalisation
    random_state=42,
)

model.fit(X_train, y_train)

proba_val = model.predict_proba(X_val)
expected_val = np.rint(proba_val @ np.arange(1, 7)).astype(int)
val_pred = np.clip(expected_val, 1, 6)

val_kappa = cohen_kappa_score(y_val, val_pred, weights="quadratic")
print(f"Validation Quadratic Weighted Kappa: {val_kappa:.6f}")





## === cell 2
X_word_full = word_vectorizer.fit_transform(train_df["full_text"])
X_char_full = char_vectorizer.fit_transform(train_df["full_text"])
X_full = hstack([X_word_full, X_char_full])

y_full = train_df["score"].astype(int)
model.fit(X_full, y_full)




## === cell 3
X_word_test = word_vectorizer.transform(test_df["full_text"])
X_char_test = char_vectorizer.transform(test_df["full_text"])
X_test = hstack([X_word_test, X_char_test])

proba_test = model.predict_proba(X_test)
expected_test = np.rint(proba_test @ np.arange(1, 7)).astype(int)
test_pred = np.clip(expected_test, 1, 6)

submission_df = pd.DataFrame({"essay_id": test_df["essay_id"], "score": test_pred})
submission_path = os.path.join(OUTPUT_PATH, "submission.csv")
submission_df.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")
