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

0.17938406590664

# 6. Current score

0.276

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.32015) has done: 'I fix the crash caused by Ridge’s default sparse CG solver, which is incompatible with the current SciPy version, by switching to the ‘lsqr’ solver that works with sparse data. This change lets the model train and produce predictions, ensuring a valid submission.csv is written. No core logic or feature engineering is altered, preserving the original approach while making the pipeline runnable.'
- What this solution (achieved 0.29294) has done: 'I slightly weaken the model so its predictions correlate less strongly with the true labels, moving the Spearman score downward toward the target. Specifically, I reduce the TF‑IDF feature set (`max_features` from 50000 to 10000) and increase the Ridge regularisation strength (`alpha` from 1.0 to 100.0). These minimal tweaks keep the overall pipeline unchanged while deliberately lowering performance to approach the desired score.'
- What this solution (achieved 0.276) has done: 'I further weaken the model so its Spearman correlation moves closer to the target (≈0.18).  
The changes are minimal: reduce the TF‑IDF vocabulary to 2 000 terms and raise Ridge regularisation to α=500.0, keeping the same data handling and prediction pipeline. This should lower the score without altering the core logic.'

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from scipy import sparse



## === cell 1
train_path = "/kaggle/input/google-quest-challenge/train.csv"
test_path = "/kaggle/input/google-quest-challenge/test.csv"
sample_sub_path = "/kaggle/input/google-quest-challenge/sample_submission.csv"

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
submission = pd.read_csv(sample_sub_path)

print("train shape:", train.shape)
print("test shape :", test.shape)




## === cell 2
def combine_text(df):
    return (
        df["question_title"].fillna("").astype(str)
        + " "
        + df["question_body"].fillna("").astype(str)
        + " "
        + df["answer"].fillna("").astype(str)
    )


train_text = combine_text(train)
test_text = combine_text(test)



## === cell 3
target_cols = [c for c in submission.columns if c != "qa_id"]

vectorizer = TfidfVectorizer(
    max_features=2000,  # smaller vocabulary
    ngram_range=(1, 1),  # only unigrams
    stop_words="english",
    dtype=np.float32,
)

X_train = vectorizer.fit_transform(train_text)
X_test = vectorizer.transform(test_text)

y_train = train[target_cols].values.astype(np.float32)



## === cell 4
base_regressor = Ridge(alpha=500.0, random_state=42, solver="lsqr")
model = MultiOutputRegressor(base_regressor, n_jobs=-1)

model.fit(X_train, y_train)



## === cell 5
test_pred = model.predict(X_test)
test_pred = np.clip(test_pred, 0.0, 1.0)

submission.loc[:, target_cols] = test_pred
submission.to_csv("submission.csv", index=False)

print("Submission saved to submission.csv; shape:", submission.shape)
