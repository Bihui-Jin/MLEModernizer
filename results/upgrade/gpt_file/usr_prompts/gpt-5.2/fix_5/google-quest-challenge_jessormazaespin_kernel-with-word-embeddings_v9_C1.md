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

0.2412229471076906

# 6. Current score

0.35156

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.35156) has done: 'I fix the runtime error caused by an incompatibility between scikit-learn’s Ridge solver (`sag`/`sparse_cg`) and the SciPy version in this environment by switching to a stable closed-form solver that works reliably on sparse TF‑IDF features. I keep the same overall pipeline (TF‑IDF + one-hot category + multioutput Ridge regression) and only change the Ridge `solver` (and set a deterministic `random_state`) so training runs end-to-end. Finally, I ensure the script always writes a correctly ordered `submission.csv` with predictions clipped to `[0,1]` and the exact columns from `sample_submission.csv`.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

PATH = "/kaggle/input/google-quest-challenge/"

train_path = os.path.join(PATH, "train.csv")
test_path = os.path.join(PATH, "test.csv")
sample_path = os.path.join(PATH, "sample_submission.csv")

df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)
sample_submission = pd.read_csv(sample_path)

target_cols = [c for c in sample_submission.columns if c != "qa_id"]

assert "qa_id" in df_train.columns and "qa_id" in df_test.columns
assert set(target_cols).issubset(
    set(df_train.columns)
), "Targets not found in train.csv"
assert len(target_cols) == 30, f"Expected 30 targets, got {len(target_cols)}"

print(
    "train:", df_train.shape, "test:", df_test.shape, "sample:", sample_submission.shape
)



## === cell 1
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor

text_features = ["question_title", "question_body", "answer"]
cat_features = ["category"]

df_train_proc = df_train.copy()
df_test_proc = df_test.copy()

for c in text_features + cat_features:
    if c not in df_train_proc.columns:
        df_train_proc[c] = ""
    if c not in df_test_proc.columns:
        df_test_proc[c] = ""

df_train_proc[text_features] = df_train_proc[text_features].fillna("")
df_test_proc[text_features] = df_test_proc[text_features].fillna("")
df_train_proc[cat_features] = df_train_proc[cat_features].fillna("unknown")
df_test_proc[cat_features] = df_test_proc[cat_features].fillna("unknown")

df_train_proc["all_text"] = (
    df_train_proc["question_title"].astype(str)
    + " [SEP] "
    + df_train_proc["question_body"].astype(str)
    + " [SEP] "
    + df_train_proc["answer"].astype(str)
)
df_test_proc["all_text"] = (
    df_test_proc["question_title"].astype(str)
    + " [SEP] "
    + df_test_proc["question_body"].astype(str)
    + " [SEP] "
    + df_test_proc["answer"].astype(str)
)

X_train = df_train_proc[["all_text"] + cat_features]
y_train = df_train_proc[target_cols].astype(np.float32).values
X_test = df_test_proc[["all_text"] + cat_features]

preprocess = ColumnTransformer(
    transformers=[
        (
            "txt",
            TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=2,
                max_features=60000,
                strip_accents="unicode",
                lowercase=True,
            ),
            "all_text",
        ),
        (
            "cat",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]
            ),
            ["category"],
        ),
    ],
    remainder="drop",
    sparse_threshold=0.3,
)

ridge = Ridge(alpha=2.0, solver="lsqr", random_state=RANDOM_STATE)

model = MultiOutputRegressor(ridge)
clf = Pipeline(steps=[("preprocess", preprocess), ("model", model)])

clf



## === cell 2
clf.fit(X_train, y_train)
pred = clf.predict(X_test).astype(np.float32)

pred = np.clip(pred, 0.0, 1.0)

submission = pd.DataFrame(pred, columns=target_cols)
submission.insert(0, "qa_id", df_test["qa_id"].values)

submission = submission[sample_submission.columns]

out_path = "submission.csv"
submission.to_csv(out_path, index=False)

print("Wrote:", out_path)
print(submission.head())
print("submission shape:", submission.shape)
assert out_path.endswith(".csv")
assert submission.shape[1] == sample_submission.shape[1]
assert list(submission.columns) == list(sample_submission.columns)
