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

0.2585599180325146

# 6. Current score

0.33128

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -0.00639) has done: 'I fixed the import errors, replaced the unavailable BERT/Universal‑Sentence‑Encoder components with a lightweight TF‑IDF text vectorizer, built a simple dense neural network for the 30 target columns, and rewrote the pipeline so it runs from data loading to saving a proper `submission.csv` file.'
- What this solution (achieved 0.33007) has done: 'I fix the runtime error caused by the Ridge solver incompatibility with the installed SciPy version by switching to a solver that works with sparse inputs (`solver='lsqr'`). This allows the model to train and produce predictions, enabling the pipeline to generate a valid `submission.csv` file and compute a reasonable validation Spearman score.'
- What this solution (achieved 0.33128) has done: 'I raise the Ridge regularisation strength (alpha) from 1.0 to 10.0 and slightly reduce the TF‑IDF vocabulary size (max_features = 15000). Stronger regularisation and fewer features lower the model’s predictive power, decreasing the validation Spearman score from the current 0.33 toward the target 0.258 while keeping the core pipeline unchanged.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import re
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
import warnings

warnings.filterwarnings("ignore", category=UserWarning)



## === cell 1
DIR = "/kaggle/input/google-quest-challenge"




## === cell 2
def func(s):
    s = re.sub("\n+", " ", s)
    s = re.sub("[?]", " . ", s)
    s = re.sub("[!{}]", " . ", s)
    s = re.sub("\.{2,}", "", s)
    s = re.sub("\s+", " ", s)
    return s


def clean_data(df):
    df["question_body"] = df["question_body"].apply(func)
    df["question_title"] = df["question_title"].apply(func)
    df["answer"] = df["answer"].apply(func)
    return df




## === cell 3
train_df = pd.read_csv(DIR + "/train.csv")
test_df = pd.read_csv(DIR + "/test.csv")

train_df = clean_data(train_df)
test_df = clean_data(test_df)

label_cols = train_df.columns[-30:]

train_df["combined"] = (
    train_df["question_title"]
    + " "
    + train_df["question_body"]
    + " "
    + train_df["answer"]
)
test_df["combined"] = (
    test_df["question_title"] + " " + test_df["question_body"] + " " + test_df["answer"]
)

vectorizer = TfidfVectorizer(max_features=15000, dtype=np.float32)
train_vec = vectorizer.fit_transform(train_df["combined"])
test_vec = vectorizer.transform(test_df["combined"])

labels = train_df[label_cols].values.astype(np.float32)




## === cell 4
def mean_spearman(y_true, y_pred):
    """Compute mean column‑wise Spearman correlation."""
    total = 0.0
    for i in range(y_true.shape[1]):
        corr, _ = spearmanr(y_true[:, i], y_pred[:, i])
        if np.isnan(corr):
            corr = 0.0
        total += corr
    return total / y_true.shape[1]


X_tr, X_val, y_tr, y_val = train_test_split(
    train_vec, labels, test_size=0.1, random_state=42
)

base_model = Ridge(alpha=10.0, random_state=42, solver="lsqr")
model = MultiOutputRegressor(base_model)

model.fit(X_tr, y_tr)
val_pred = model.predict(X_val)

val_score = mean_spearman(y_val, val_pred)
print(f"Validation mean Spearman correlation: {val_score:.5f}")

model.fit(train_vec, labels)



## === cell 5
test_pred = model.predict(test_vec)
test_pred = np.clip(test_pred, 0.0, 1.0)



## === cell 6
sample_submission = pd.read_csv(DIR + "/sample_submission.csv")
submission = pd.DataFrame(test_pred, columns=label_cols)
submission["qa_id"] = test_df["qa_id"].values
submission = submission[sample_submission.columns]
submission.to_csv("submission.csv", index=False)
