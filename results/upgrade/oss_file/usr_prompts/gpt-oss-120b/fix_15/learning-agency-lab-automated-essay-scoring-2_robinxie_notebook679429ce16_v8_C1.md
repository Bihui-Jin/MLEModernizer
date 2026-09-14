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

cudf-polars-cu12==25.6.0
gensim==4.4.0
geopandas==0.14.4
joblib==1.5.2
lightgbm==4.6.0
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
polars==1.25.0
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
scipy==1.15.3
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
import re
import gc
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
import scipy.sparse as sp
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

HTML_RE = re.compile(r"<.*?>")
PATTERNS = [
    (HTML_RE, ""),
    (re.compile(r"@\w+"), ""),
    (re.compile(r"'\d+"), ""),
    (re.compile(r"\d+"), ""),
    (re.compile(r"http\w+"), ""),
    (re.compile(r"\s+"), " "),
    (re.compile(r"\.+"), "."),
    (re.compile(r"\,+"), ","),
]


def vectorized_preprocess(series: pd.Series) -> pd.Series:
    """Lower‑case and apply all regex substitutions using pandas vectorized string ops."""
    s = series.str.lower()
    for pat, repl in PATTERNS:
        s = s.str.replace(pat, repl, regex=True)
    return s.str.strip()


PATH = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2"
OUTPUT_PATH = "/kaggle/working/"

train_df = pd.read_csv(os.path.join(PATH, "train.csv"))
test_df = pd.read_csv(os.path.join(PATH, "test.csv"))

train_df["full_text"] = vectorized_preprocess(train_df["full_text"])
test_df["full_text"] = vectorized_preprocess(test_df["full_text"])

train_df["char_len"] = train_df["full_text"].str.len()
train_df["word_len"] = train_df["full_text"].str.split().apply(len)

test_df["char_len"] = test_df["full_text"].str.len()
test_df["word_len"] = test_df["full_text"].str.split().apply(len)

vectorizer = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1, 2),
    stop_words="english",
    dtype=np.float32,
)

train_tfid = vectorizer.fit_transform(train_df["full_text"])
test_tfid = vectorizer.transform(test_df["full_text"])

scaler = StandardScaler()
train_extra = scaler.fit_transform(train_df[["char_len", "word_len"]].values)
test_extra = scaler.transform(test_df[["char_len", "word_len"]].values)

train_extra_sparse = sp.csr_matrix(train_extra, dtype=np.float32)
test_extra_sparse = sp.csr_matrix(test_extra, dtype=np.float32)

X = sp.hstack([train_tfid, train_extra_sparse]).tocsr()
X_test = sp.hstack([test_tfid, test_extra_sparse]).tocsr()

y = train_df["score"].values - 1  # LightGBM expects 0‑based classes




## === cell 1
n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
fold_splits = list(skf.split(X, y))  # list of (train_idx, val_idx)


def _train_fold(fold_idx, tr_idx, val_idx):
    """Train a single LightGBM fold (no validation scoring) and return its test‑set probabilities."""
    X_tr, X_val = X[tr_idx], X[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]

    lgb_model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=6,
        learning_rate=0.05,
        n_estimators=1000,
        max_depth=-1,
        colsample_bytree=0.8,
        subsample=0.8,
        class_weight="balanced",
        random_state=fold_idx,
        n_jobs=1,
        verbose=-1,
    )
    lgb_model.fit(
        X_tr,
        y_tr,
        eval_metric="multi_logloss",
    )
    prob = lgb_model.predict_proba(X_test)

    del X_tr, X_val, y_tr, y_val, lgb_model
    gc.collect()
    return prob


prob_list = joblib.Parallel(
    n_jobs=n_folds,
    backend="loky",
    verbose=0,
)(
    joblib.delayed(_train_fold)(fold_idx + 1, tr_idx, val_idx)
    for fold_idx, (tr_idx, val_idx) in enumerate(fold_splits)
)

prob_sum = np.sum(prob_list, axis=0)
avg_prob = prob_sum / n_folds

final_pred = np.argmax(avg_prob, axis=1) + 1
final_pred = np.clip(final_pred, 1, 6)

submission = pd.DataFrame({"essay_id": test_df["essay_id"], "score": final_pred})
submission_path = os.path.join(OUTPUT_PATH, "submission.csv")
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
