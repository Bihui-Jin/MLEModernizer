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

0.0018343368837958

# 6. Current score

0.74882

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.74882) has done: 'Main bottlenecks are (1) building a 200k-feature word+bigram TF‑IDF on 139k essays and (2) fitting 1200 boosting rounds on a very high‑dimensional sparse matrix. To stay within 600s without changing the algorithm, I (a) make TF‑IDF construction faster via vectorized pre-tokenization and multi-threaded vocabulary/DF counting, and (b) make XGBoost training faster by enabling its built-in sparse-aware column sampling (`colsample_bynode`) while keeping the same estimator, objective, and number of trees. I also remove redundant work (avoid `eval_set` on the training data which adds overhead each boosting round while not affecting the final model when no early stopping is used) and ensure deterministic settings are preserved. All paths, model type, feature type (TF‑IDF 1–2 grams), and prediction post-processing remain the same.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

INPUT_DIR = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2"
TRAIN_PATH = os.path.join(INPUT_DIR, "train.csv")
TEST_PATH = os.path.join(INPUT_DIR, "test.csv")
SAMPLE_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

assert os.path.exists(TRAIN_PATH), f"Missing: {TRAIN_PATH}"
assert os.path.exists(TEST_PATH), f"Missing: {TEST_PATH}"
assert os.path.exists(SAMPLE_PATH), f"Missing: {SAMPLE_PATH}"

train_df = pd.read_csv(
    TRAIN_PATH,
    usecols=["essay_id", "full_text", "score"],
    dtype={"essay_id": "string", "full_text": "string", "score": "int8"},
)
test_df = pd.read_csv(
    TEST_PATH,
    usecols=["essay_id", "full_text"],
    dtype={"essay_id": "string", "full_text": "string"},
)
sample_df = pd.read_csv(
    SAMPLE_PATH,
    usecols=["essay_id", "score"],
    dtype={"essay_id": "string"},
)

required_train_cols = {"essay_id", "full_text", "score"}
required_test_cols = {"essay_id", "full_text"}
assert required_train_cols.issubset(
    train_df.columns
), f"train.csv missing cols: {required_train_cols - set(train_df.columns)}"
assert required_test_cols.issubset(
    test_df.columns
), f"test.csv missing cols: {required_test_cols - set(test_df.columns)}"
assert {"essay_id", "score"}.issubset(
    sample_df.columns
), "sample_submission.csv must have essay_id, score"

train_df["full_text"] = train_df["full_text"].fillna("")
test_df["full_text"] = test_df["full_text"].fillna("")

y = train_df["score"].astype(np.float32).to_numpy(copy=False)
X_text = train_df["full_text"].to_numpy(copy=False)
X_test_text = test_df["full_text"].to_numpy(copy=False)

print("Train shape:", train_df.shape, "Test shape:", test_df.shape)
print("Score range in train:", (float(np.min(y)), float(np.max(y))))



## === cell 1
try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except Exception:
    pass

import re
from sklearn.feature_extraction.text import TfidfVectorizer

_token_re = re.compile(r"(?u)\b\w\w+\b")


def _analyze_doc(doc: str):
    return _token_re.findall(doc.lower())


tfidf = TfidfVectorizer(
    lowercase=False,  # already lowercased in analyzer
    strip_accents="unicode",
    analyzer=_analyze_doc,  # pre-tokenize; vectorizer will build 1-2 grams from tokens
    ngram_range=(1, 2),
    max_features=200000,
    min_df=2,
    max_df=0.9,
    sublinear_tf=True,
)

X = tfidf.fit_transform(X_text)
X_test = tfidf.transform(X_test_text)

print("TF-IDF train matrix:", X.shape, "TF-IDF test matrix:", X_test.shape)



## === cell 2
import xgboost as xgb

SEED = 42
nthread = int(os.environ.get("OMP_NUM_THREADS", "0")) or (os.cpu_count() or 1)

model = xgb.XGBRegressor(
    n_estimators=1200,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    colsample_bynode=0.8,  # additional sampling inside each tree node (constant-factor speed win)
    reg_alpha=0.0,
    reg_lambda=1.0,
    objective="reg:squarederror",
    tree_method="hist",
    random_state=SEED,
    n_jobs=nthread,
    verbosity=0,
)

model.fit(X, y, verbose=False)



## === cell 3
pred = model.predict(X_test)
pred_score = np.rint(np.clip(pred, 1, 6)).astype(np.int32)

sub = pd.DataFrame({"essay_id": test_df["essay_id"], "score": pred_score})

if len(sample_df) == len(sub) and sample_df["essay_id"].equals(sub["essay_id"]):
    sub = pd.DataFrame(
        {"essay_id": sample_df["essay_id"], "score": sub["score"].to_numpy()}
    )
else:
    sub = sample_df[["essay_id"]].merge(sub, on="essay_id", how="left")

assert (
    sub.shape[0] == test_df.shape[0] or sub.shape[0] == sample_df.shape[0]
), "Unexpected submission row count"
assert sub["score"].notna().all(), "NaNs in submission scores"
assert sub["score"].between(1, 6).all(), "Scores out of range 1..6"

out_path = "submission.csv"
sub.to_csv(out_path, index=False)

print("Wrote:", out_path)
print(sub.head())
print(sub["score"].value_counts().sort_index())
