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

0.3475238241448931

# 6. Current score

0.29207

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.29207) has done: 'I fixed the failure by switching the Ridge solver to one that works with the current SciPy version (`solver="lsqr"`). This avoids the `cg()` keyword‑argument error, lets the model train on the validation split, then on the full data, and finally produces a proper `submission.csv` with the required columns.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from scipy.stats import spearmanr



## === cell 1
DATA_PATH = "../input/google-quest-challenge"
TRAIN_PATH = os.path.join(DATA_PATH, "train.csv")
TEST_PATH = os.path.join(DATA_PATH, "test.csv")
SAMPLE_SUB_PATH = os.path.join(DATA_PATH, "sample_submission.csv")
OUTPUT_SUB_PATH = "submission.csv"



## === cell 2
df_train = pd.read_csv(TRAIN_PATH)
df_test = pd.read_csv(TEST_PATH)
df_sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

text_cols = ["question_title", "question_body", "answer"]
target_cols = df_train.columns[-30:].tolist()




## === cell 3
def combine_text(row):
    parts = [str(row[col]) if pd.notnull(row[col]) else "" for col in text_cols]
    return " ".join(parts)


df_train["combined_text"] = df_train.apply(combine_text, axis=1)
df_test["combined_text"] = df_test.apply(combine_text, axis=1)



## === cell 4
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    stop_words="english",
    token_pattern=r"(?u)\b\w\w+\b",
)

X_train_vec = vectorizer.fit_transform(df_train["combined_text"])
X_test_vec = vectorizer.transform(df_test["combined_text"])

y_train = df_train[target_cols].values



## === cell 5
base_model = Ridge(alpha=1.0, solver="lsqr", random_state=42)
model = MultiOutputRegressor(base_model)

X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_vec, y_train, test_size=0.1, random_state=42
)
model.fit(X_tr, y_tr)


def spearman_mean(y_true, y_pred):
    rhos = []
    for tcol, pcol in zip(y_true.T, y_pred.T):
        rho = spearmanr(tcol, pcol).correlation
        if np.isnan(rho):
            rho = 0.0
        rhos.append(rho)
    return np.mean(rhos)


val_pred = model.predict(X_val)
val_score = spearman_mean(y_val, val_pred)
print(f"Validation Spearman mean: {val_score:.5f}")



## === cell 6
model.fit(X_train_vec, y_train)

test_pred = model.predict(X_test_vec)
test_pred = np.clip(test_pred, 0.0, 1.0)



## === cell 7
df_submission = df_sample_sub.copy()
df_submission = df_submission.set_index("qa_id").loc[df_test["qa_id"]].reset_index()
df_submission[target_cols] = test_pred
df_submission.to_csv(OUTPUT_SUB_PATH, index=False)
print(f"Submission written to {OUTPUT_SUB_PATH}")
