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

cudf-polars-cu12==25.6.0
geopandas==0.14.4
joblib==1.5.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
polars==1.25.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
spacy==3.8.7
spacy-legacy==3.0.12
spacy-loggers==1.0.5
tqdm==4.67.1
xgboost==2.0.3

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

0.0108952342277427

# 6. Current score

0.7473

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.7473) has done: 'I replace the missing utility imports and model files with a small, self‑contained training pipeline that loads the provided CSVs, creates a few simple text features, fits a multi‑class XGBoost model, generates predictions for the test set, and writes a correctly formatted `submission.csv`. This resolves all import and file‑not‑found errors and produces a valid submission; given the extremely low target metric, any reasonable model fall within the acceptable range.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import polars as pl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import xgboost as xgb

DATA_ROOT = "/kaggle/input"
TRAIN_PATH = os.path.join(DATA_ROOT, "train.csv")
TEST_PATH = os.path.join(DATA_ROOT, "test.csv")
SUBMISSION_PATH = "submission.csv"



## === cell 1
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

print("Train shape:", train_df.shape, "Test shape:", test_df.shape)



## === cell 2
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english")

X_text_train = tfidf.fit_transform(train_df["full_text"].fillna(""))
X_text_test = tfidf.transform(test_df["full_text"].fillna(""))


def extra_features(df):
    txt = df["full_text"].fillna("")
    return np.vstack(
        [
            txt.str.len().values,  # character length
            txt.str.split().str.len().values,  # word count
        ]
    ).T


X_num_train = extra_features(train_df)
X_num_test = extra_features(test_df)

from scipy import sparse

X_train = sparse.hstack([X_text_train, X_num_train])
X_test = sparse.hstack([X_text_test, X_num_test])



## === cell 3
le = LabelEncoder()
y = le.fit_transform(train_df["score"])



## === cell 4
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y, test_size=0.1, stratify=y, random_state=42
)

clf = xgb.XGBClassifier(
    objective="multi:softprob",
    eval_metric="mlogloss",
    num_class=len(le.classes_),
    learning_rate=0.1,
    max_depth=6,
    n_estimators=200,
    subsample=0.8,
    colsample_bytree=0.8,
    n_jobs=4,
    tree_method="hist",
    use_label_encoder=False,
    verbosity=0,
)

clf.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], early_stopping_rounds=20, verbose=False)

print("Best iteration:", clf.best_iteration)



## === cell 5
test_pred_proba = clf.predict_proba(X_test)
test_pred_labels = np.argmax(test_pred_proba, axis=1)
test_scores = le.inverse_transform(test_pred_labels)



## === cell 6
submission = pd.DataFrame(
    {"essay_id": test_df["essay_id"], "score": test_scores.astype(int)}
)

submission.to_csv(SUBMISSION_PATH, index=False)
print(f"Submission saved to {SUBMISSION_PATH}")
print(submission.head())
