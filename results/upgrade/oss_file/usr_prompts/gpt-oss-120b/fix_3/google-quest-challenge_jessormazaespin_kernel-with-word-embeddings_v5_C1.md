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

0.2639521134234983

# 6. Current score

0.30993

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.30993) has done: 'I fix the failure caused by the Ridge regression solver, which is incompatible with the current SciPy version. By explicitly setting `solver='lsqr'` the model fits correctly on sparse data, allowing the script to run end‑to‑end and generate a proper `submission.csv`. No other logic is altered.'

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import OneHotEncoder
from scipy import sparse

PATH = "../input/google-quest-challenge/"
train_path = PATH + "train.csv"
test_path = PATH + "test.csv"

df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)

target_cols = df_train.columns[11:]  # after the first 11 meta columns
print(f"Target columns ({len(target_cols)}): {list(target_cols)}")


def combine_text(df):
    return (
        df["question_title"].fillna("")
        + " "
        + df["question_body"].fillna("")
        + " "
        + df["answer"].fillna("")
    )


train_text = combine_text(df_train)
test_text = combine_text(df_test)

vectorizer = TfidfVectorizer(
    max_features=20000, ngram_range=(1, 2), stop_words="english"
)
X_train_text = vectorizer.fit_transform(train_text)
X_test_text = vectorizer.transform(test_text)

categorical = df_train[["category"]].astype(str)
enc = OneHotEncoder(sparse=True, handle_unknown="ignore")
X_train_cat = enc.fit_transform(categorical)

X_test_cat = enc.transform(df_test[["category"]].astype(str))

X_train = sparse.hstack([X_train_text, X_train_cat])
X_test = sparse.hstack([X_test_text, X_test_cat])

y = df_train[target_cols].values.astype(np.float32)

X_tr, X_val, y_tr, y_val = train_test_split(X_train, y, test_size=0.1, random_state=42)

base_model = Ridge(alpha=1.0, random_state=42, solver="lsqr")
model = MultiOutputRegressor(base_model)

model.fit(X_tr, y_tr)



## === cell 1
test_pred = model.predict(X_test)

test_pred = np.clip(test_pred, 0.0, 1.0)

submission = pd.DataFrame(test_pred, columns=target_cols)
submission.insert(0, "qa_id", df_test["qa_id"].values)

sample_sub_path = PATH + "sample_submission.csv"
sample_sub = pd.read_csv(sample_sub_path, nrows=0)  # just header
submission = submission[sample_sub.columns]  # reorder if needed

submission.to_csv("submission.csv", index=False)
print("Submission written to submission.csv with shape:", submission.shape)
