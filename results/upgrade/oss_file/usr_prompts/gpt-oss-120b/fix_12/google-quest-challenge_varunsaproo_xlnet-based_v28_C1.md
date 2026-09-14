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

0.2496310240701485

# 6. Current score

0.28218

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -0.00332) has done: 'I replace the failing BERT‑based pipeline with a lightweight TF‑IDF + dense‑network model that uses only the available libraries. The new code cleans the text, builds a TF‑IDF representation, trains a simple Keras regressor on the 30 continuous targets, and writes a correctly‑formatted `submission.csv`. This fixes all NameError / import errors and ensures a valid submission while keeping the overall approach (text preprocessing → feature extraction → neural model) unchanged.'
- What this solution (achieved 0.30307) has done: 'Implemented a fix for the Ridge regression solver by switching to `solver='lsqr'`, which is compatible with the current SciPy version and avoids the `cg()` argument error. This allows the `MultiOutputRegressor` to train successfully, after which predictions are made, clipped to the required [0, 1] range, and written to a correctly formatted `submission.csv`. No other logic or model architecture changes were made, preserving the original pipeline while ensuring a valid submission file is generated.'
- What this solution (achieved 0.31217) has done: 'I slightly increase the Ridge regularisation strength (alpha = 5.0) so the model under‑fits a bit, which is expected to lower the Spearman correlation and move the public score from 0.30307 down toward the target 0.2496. This change is minimal, keeps the same pipeline and output format, and does not affect any other logic.'
- What this solution (achieved 0.30222) has done: 'I increase the Ridge regularisation strength from 5.0 to 20.0 so the model under‑fits more, which is expected to lower the Spearman correlation and bring the score closer to the target (while keeping the same pipeline and output format). This is the only change needed.'
- What this solution (achieved 0.29055) has done: 'I increase the Ridge regularisation strength from 20.0 to 80.0. A larger alpha makes the linear model under‑fit more, which typically lowers the Spearman correlation and moves the score closer to the target 0.2496310240701485 while keeping the original pipeline unchanged. The only modification is the alpha value in the MultiOutputRegressor initialisation; all other logic and file handling remain the same.'
- What this solution (achieved 0.28628) has done: 'I increase the Ridge regularisation strength (alpha) from 80.0 to 200.0 in the MultiOutputRegressor. A larger alpha makes the linear model under‑fit more, which should lower the Spearman correlation and move the score from 0.29055 closer to the target 0.249631… while keeping the rest of the pipeline unchanged.'
- What this solution (achieved 0.28328) has done: 'I increase the Ridge regularisation strength (alpha) from 200.0 to 800.0 in the `MultiOutputRegressor`. A larger alpha forces the linear model to under‑fit more, which typically lowers the Spearman correlation and moves the score from 0.28628 closer to the target 0.249631… while preserving the original pipeline and output format.'
- What this solution (achieved 0.28264) has done: 'I keep the overall pipeline unchanged and only increase the Ridge regularisation strength to make the model under‑fit more, which should lower the Spearman correlation and move the score from 0.28328 closer to the target 0.24963. The only modification is setting `alpha=2000.0` in the Ridge estimator.'
- What this solution (achieved 0.2822) has done: 'I slightly increase the Ridge regularisation strength (alpha) to make the model under‑fit a bit more, which should lower the Spearman correlation and move the score from 0.28264 toward the target 0.24963. The only code change is setting `alpha=8000.0` in the `Ridge` estimator while keeping the rest of the pipeline unchanged.'
- What this solution (achieved 0.28218) has done: 'I keep the entire pipeline unchanged and only increase the Ridge regularization strength (alpha) from 8000 to 40000 in the `MultiOutputRegressor`. A larger alpha makes the linear model under‑fit more, which is expected to lower the Spearman correlation and move the validation score from 0.2822 into the target band around 0.25 while preserving all other logic and the correct CSV output.'

# 9. Code solution

## === cell 0
import os, re, numpy as np, pandas as pd

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
DIR = "/kaggle/input/google-quest-challenge"
BATCH_SIZE = 64



## === cell 2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor




## === cell 3
def func(s):
    s = re.sub("\n+", " ", s)
    s = re.sub("[?]", " . ", s)
    s = re.sub("[!\{\}]", " . ", s)
    s = re.sub("\.{2,}", "", s)
    s = re.sub("\s+", " ", s)
    return s


def clean_data(df):
    df["question_body"] = df["question_body"].apply(func)
    df["question_title"] = df["question_title"].apply(func)
    df["answer"] = df["answer"].apply(func)
    return df


def create_aggregate(df):
    df["aggregate"] = (
        df["question_title"] + " " + df["question_body"] + " " + df["answer"]
    )
    return df




## === cell 4
train_df = pd.read_csv(os.path.join(DIR, "train.csv"))
test_df = pd.read_csv(os.path.join(DIR, "test.csv"))

train_df = clean_data(train_df)
test_df = clean_data(test_df)

train_df = create_aggregate(train_df)
test_df = create_aggregate(test_df)



## === cell 5
vectorizer = TfidfVectorizer(max_features=8000, stop_words="english")
train_X = vectorizer.fit_transform(train_df["aggregate"]).astype(np.float32)
test_X = vectorizer.transform(test_df["aggregate"]).astype(np.float32)



## === cell 6
numeric_cols = train_df.select_dtypes(include=[np.number]).columns
label_cols = numeric_cols[-30:]
labels = train_df[label_cols].values.astype(np.float32)



## === cell 7
model = MultiOutputRegressor(Ridge(alpha=40000.0, random_state=42, solver="lsqr"))



## === cell 8
X_tr, X_val, y_tr, y_val = train_test_split(
    train_X, labels, test_size=0.1, random_state=42
)

model.fit(X_tr, y_tr)



## === cell 9
preds = model.predict(test_X)
preds = np.clip(preds, 0.0, 1.0)



## === cell 10
sample_submission = pd.read_csv(os.path.join(DIR, "sample_submission.csv"))
submission = pd.DataFrame(preds, columns=sample_submission.columns[1:])
submission.insert(0, "qa_id", test_df["qa_id"])
submission = submission[sample_submission.columns]
submission.to_csv("submission.csv", index=False)
