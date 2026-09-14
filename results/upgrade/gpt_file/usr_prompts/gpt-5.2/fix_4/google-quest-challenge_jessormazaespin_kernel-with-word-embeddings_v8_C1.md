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
Given questions and answers from various StackExchange properties, predict target values of 30 labels for each question-answer pair.

## Metric
Mean column-wise Spearman's correlation coefficient. The Spearman's rank correlation is computed for each target column, and the mean of these values is calculated for the submission score.

## Submission Format
For each qa_id in the test set, you must predict a probability for each target variable. The predictions should be in the range [0,1]. The file should contain a header and have the following format:

```
qa_id,question_asker_intent_understanding,...,answer_well_written
6,0.0,...,0.5
8,0.5,...,0.1
18,1.0,...,0.0
etc.
```

## Dataset
The list of 30 target labels are the same as the column names in the `sample_submission.csv` file. Target labels with the prefix `question_` relate to the `question_title` and/or `question_body` features in the data. Target labels with the prefix `answer_` relate to the `answer` feature.

Target labels are aggregated from multiple raters, and can have continuous values in the range `[0,1]`. Therefore, predictions must also be in that range.

- **train.csv** - the training data (target labels are the last 30 columns)
- **test.csv** - the test set (you must predict 30 labels for each test set row)
- **sample_submission.csv** - a sample submission file in the correct format; column names are the 30 target labels

# 2. Python version

3.8

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (83 lines)
            sample_submission.csv (609 lines)
            sample_submission.csv.zip (8.9 kB)
            test.csv (19551 lines)
            test.csv.zip (471.7 kB)
            train.csv (159837 lines)
            train.csv.zip (4.2 MB)
            google-quest-challenge/
                description.md (83 lines)
                sample_submission.csv (609 lines)
                ... and 5 other files
                google-quest-challenge/
        input/
            description.md (83 lines)
            sample_submission.csv (609 lines)
            sample_submission.csv.zip (8.9 kB)
            test.csv (19551 lines)
            test.csv.zip (471.7 kB)
            train.csv (159837 lines)
            train.csv.zip (4.2 MB)
            google-quest-challenge/
                description.md (83 lines)
                sample_submission.csv (609 lines)
                ... and 5 other files
                google-quest-challenge/
        working/
            google-quest-challenge/
                description.md (83 lines)
                sample_submission.csv (609 lines)
                ... and 5 other files
                google-quest-challenge/
```

-> data/google-quest-challenge/sample_submission.csv has 608 rows and 31 columns.
The columns are: qa_id, question_asker_intent_understanding, question_body_critical, question_conversational, question_expect_short_answer, question_fact_seeking, question_has_commonly_accepted_answer, question_interestingness_others, question_interestingness_self, question_multi_intent, question_not_really_a_question, question_opinion_seeking, question_type_choice, question_type_compare, question_type_consequence... and 16 more columns

-> data/google-quest-challenge/test.csv has 19550 rows and 11 columns.
The columns are: qa_id, question_title, question_body, question_user_name, question_user_page, answer, answer_user_name, answer_user_page, url, category, host

-> data/google-quest-challenge/train.csv has 159836 rows and 41 columns.
The columns are: qa_id, question_title, question_body, question_user_name, question_user_page, answer, answer_user_name, answer_user_page, url, category, host, question_asker_intent_understanding, question_body_critical, question_conversational, question_expect_short_answer... and 26 more columns

-> data/sample_submission.csv has 608 rows and 31 columns.
The columns are: qa_id, question_asker_intent_understanding, question_body_critical, question_conversational, question_expect_short_answer, question_fact_seeking, question_has_commonly_accepted_answer, question_interestingness_others, question_interestingness_self, question_multi_intent, question_not_really_a_question, question_opinion_seeking, question_type_choice, question_type_compare, question_type_consequence... and 16 more columns

-> data/test.csv has 19550 rows and 11 columns.
The columns are: qa_id, question_title, question_body, question_user_name, question_user_page, answer, answer_user_name, answer_user_page, url, category, host

-> data/train.csv has 159836 rows and 41 columns.
The columns are: qa_id, question_title, question_body, question_user_name, question_user_page, answer, answer_user_name, answer_user_page, url, category, host, question_asker_intent_understanding, question_body_critical, question_conversational, question_expect_short_answer... and 26 more columns

-> (stopped after 10 files for performance)

# 5. Target score

0.2336372618481733

# 6. Current score

0.3764

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.16445) has done: 'The notebook currently fails before producing predictions because TensorFlow is crashing on import (protobuf incompatibility) and because it expects external model/tokenizer files that don’t exist in this Kaggle environment. To make it run end-to-end while preserving the overall “tokenize text → model predicts 30 targets → write submission.csv” logic, I replace the missing pre-trained Keras model with a lightweight in-notebook Keras model trained on the provided `train.csv` (same multi-input structure: title/body/answer/category). I also replace the missing pickled tokenizer with a fitted Keras `Tokenizer` trained on the training texts. Finally, I ensure the submission columns match `sample_submission.csv` exactly and clip predictions to `[0,1]` to satisfy submission requirements.'
- What this solution (achieved 0.3764) has done: 'I fix the runtime crash by switching Ridge to a solver that avoids SciPy’s `cg(tol=...)` incompatibility in this environment, which currently prevents the model from fitting and therefore stops submission generation. Then I make cell 2 robust by ensuring it only runs after successful training/prediction and by keeping the same isotonic calibration + clipping logic so predictions remain in `[0,1]`. These changes preserve the core approach (TF‑IDF + Ridge multioutput + isotonic calibration) while unblocking end-to-end execution and producing a valid `submission.csv`. Since no score was yielded, the priority is correctness and a valid submission; the solver change should be score-neutral or slightly positive.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

PATH = "/kaggle/input/google-quest-challenge/"
df_train = pd.read_csv(os.path.join(PATH, "train.csv"))
df_test = pd.read_csv(os.path.join(PATH, "test.csv"))
sample_sub = pd.read_csv(os.path.join(PATH, "sample_submission.csv"))

target_cols = [c for c in sample_sub.columns if c != "qa_id"]
assert len(target_cols) == 30, f"Expected 30 targets, got {len(target_cols)}"

print(df_train.shape, df_test.shape, sample_sub.shape)
print("n_targets:", len(target_cols))
print("Train columns head:", df_train.columns[:12].tolist())



## === cell 1
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error
from sklearn.isotonic import IsotonicRegression

TEXT_COLS = ["question_title", "question_body", "answer"]
CAT_COLS = ["category", "host"]

for col in TEXT_COLS + CAT_COLS:
    if col in df_train.columns:
        df_train[col] = df_train[col].fillna("").astype(str)
    if col in df_test.columns:
        df_test[col] = df_test[col].fillna("").astype(str)

X = df_train[TEXT_COLS + CAT_COLS].copy()
y = df_train[target_cols].astype(np.float32).values
X_test = df_test[TEXT_COLS + CAT_COLS].copy()

X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.10, random_state=SEED)

preprocess = ColumnTransformer(
    transformers=[
        (
            "title_tfidf",
            TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=2,
                max_features=30000,
                strip_accents="unicode",
                lowercase=True,
            ),
            "question_title",
        ),
        (
            "body_tfidf",
            TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=2,
                max_features=60000,
                strip_accents="unicode",
                lowercase=True,
            ),
            "question_body",
        ),
        (
            "ans_tfidf",
            TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=2,
                max_features=60000,
                strip_accents="unicode",
                lowercase=True,
            ),
            "answer",
        ),
        ("cat_ohe", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
    ],
    remainder="drop",
    sparse_threshold=0.3,
)

base_ridge = Ridge(alpha=2.0, solver="lsqr", random_state=SEED)

reg = MultiOutputRegressor(base_ridge, n_jobs=1)
model = Pipeline(steps=[("prep", preprocess), ("reg", reg)])

model.fit(X_tr, y_tr)
va_pred = model.predict(X_va)
va_pred = np.clip(va_pred, 0.0, 1.0)

print("Validation MSE (sanity):", mean_squared_error(y_va, va_pred))



## === cell 2
calibrators = []
va_pred_cal = np.empty_like(va_pred, dtype=np.float32)

for j in range(y_va.shape[1]):
    ir = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    ir.fit(va_pred[:, j], y_va[:, j])
    calibrators.append(ir)
    va_pred_cal[:, j] = ir.transform(va_pred[:, j]).astype(np.float32)

print("Validation MSE after isotonic (sanity):", mean_squared_error(y_va, va_pred_cal))

model.fit(X, y)

test_pred = model.predict(X_test)
test_pred = np.clip(test_pred, 0.0, 1.0)

test_pred_cal = np.empty_like(test_pred, dtype=np.float32)
for j, ir in enumerate(calibrators):
    test_pred_cal[:, j] = ir.transform(test_pred[:, j]).astype(np.float32)

test_pred_cal = np.clip(test_pred_cal, 0.0, 1.0)

submission = pd.DataFrame(test_pred_cal, columns=target_cols)
submission.insert(0, "qa_id", df_test["qa_id"].values)
submission = submission[sample_sub.columns]
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Saved submission.csv with shape:", submission.shape)
print("Columns OK:", list(submission.columns) == list(sample_sub.columns))
print(
    "Pred range:",
    float(submission[target_cols].min().min()),
    float(submission[target_cols].max().max()),
)
