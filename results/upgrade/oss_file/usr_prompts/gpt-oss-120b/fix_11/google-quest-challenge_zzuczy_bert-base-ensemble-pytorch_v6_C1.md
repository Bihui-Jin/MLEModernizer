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

0.3567914536991772

# 6. Current score

0.08665

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.31371) has done: 'The fix changes the Ridge regression to use the “lsqr” solver, which avoids the SciPy cg incompatibility that caused the fit to fail. With the model now fitting correctly, the downstream prediction, clipping, and submission‑creation steps work, producing a valid `submission.csv` file containing the required `qa_id` column and the 30 target columns.'
- What this solution (achieved 0.32003) has done: 'I slightly enrich the text feature by concatenating the `category` and `host` fields, increase the TF‑IDF capacity and enable sub‑linear TF scaling, and lower the Ridge regularisation (α=0.5). These modest adjustments keep the original linear‑regression pipeline while giving the model a bit more expressive power, which should raise the Spearman correlation toward the target score.'
- What this solution (achieved 0.31508) has done: 'I slightly reduce the Ridge regularisation (α = 0.3) and increase the TF‑IDF capacity to 60 k features. These minimal tweaks keep the same linear‑regression pipeline while giving the model a bit more flexibility, which should raise the Spearman correlation toward the target score.'
- What this solution (achieved 0.31019) has done: 'I modestly increase the TF‑IDF vocabulary size and n‑gram range to capture more textual patterns, and lower the Ridge regularisation (α) slightly. These small adjustments keep the original linear‑regression pipeline intact while giving the model a bit more flexibility, which should raise the Spearman correlation toward the target score.'
- What this solution (achieved 0.30513) has done: 'I slightly increase the TF‑IDF vocabulary size and make the ridge regularisation weaker (α = 0.1). This gives the linear model a bit more capacity to capture useful patterns without altering the overall pipeline, which should raise the Spearman correlation into the target‑score band.'
- What this solution (achieved 0.08664) has done: 'I add a simple numeric signal (the raw text length) to the sparse TF‑IDF matrices, increase the TF‑IDF vocabulary size a bit, and weaken the Ridge regularisation (α = 0.05). These tiny, low‑risk changes keep the original linear‑model pipeline intact while giving the model a bit more information, which should raise the Spearman correlation toward the target score.'
- What this solution (achieved 0.08664) has done: 'I tighten the TF‑IDF representation and use a slightly stronger Ridge regularisation, which in earlier experiments moved the Spearman score from the low 0.08 range up toward the target (~0.30). The changes keep the overall pipeline intact while adjusting only the hyper‑parameters that control model capacity.'
- What this solution (achieved 0.29691) has done: 'I raise the model’s expressive power and reduce noisy signals by (1) enlarging the TF‑IDF vocabulary, (2) discarding extremely rare terms, (3) removing the raw‑length feature that previously hurt performance, and (4) using a weaker Ridge regularisation (α = 0.1). These minimal adjustments stay within the original linear‑regression pipeline while aiming to lift the Spearman correlation toward the target score.'
- What this solution (achieved 0.08665) has done: 'I slightly enrich the textual representation by adding the `category` and `host` fields, increase the TF‑IDF capacity, and attach a simple numeric signal (the character length of the combined text) as an additional feature. I also lower the Ridge regularisation (α) a bit. These minimal, targeted tweaks keep the overall linear‑model pipeline unchanged while giving the model a bit more expressive power, which should raise the Spearman correlation toward the target score.'

# 9. Code solution

## === cell 0
import os, re, numpy as np, pandas as pd

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from scipy import sparse



## === cell 2
train_path = "../input/google-quest-challenge/train.csv"
test_path = "../input/google-quest-challenge/test.csv"
sample_sub_path = "../input/google-quest-challenge/sample_submission.csv"

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
sample_submission = pd.read_csv(sample_sub_path)




## === cell 3
def txt_clean(txt: str) -> str:
    """basic cleaning used in the original notebook"""
    if not isinstance(txt, str):
        return ""
    txt = txt.strip()
    txt = re.sub(r"https?.*$", "", txt)
    txt = re.sub(r"https?.*\s", "", txt)
    txt = re.sub(r"\n+", " ", txt)
    txt = re.sub(r"\r+", " ", txt)
    txt = re.sub(r"\t+", " ", txt)
    txt = re.sub(r"&gt;", ">", txt)
    txt = re.sub(r"&lt;", "<", txt)
    txt = re.sub(r"&amp;", "&", txt)
    txt = re.sub(r"&quot;", '"', txt)
    return txt




## === cell 4
target_cols = train.columns.values[11:]


def combine_text(df):
    """
    Concatenate the main textual fields plus low‑cardinality categorical fields.
    Adding `category` and `host` provides extra signal without altering the core
    linear‑model pipeline.
    """
    return (
        df["question_title"].fillna("").astype(str)
        + " "
        + df["question_body"].fillna("").astype(str)
        + " "
        + df["answer"].fillna("").astype(str)
        + " "
        + df["category"].fillna("").astype(str)
        + " "
        + df["host"].fillna("").astype(str)
    ).apply(txt_clean)


train_text = combine_text(train)
test_text = combine_text(test)



## === cell 5
vectorizer = TfidfVectorizer(
    max_features=150000,  # slightly larger vocab for more expressive power
    min_df=2,
    ngram_range=(1, 2),
    stop_words="english",
    sublinear_tf=True,
)

X_train = vectorizer.fit_transform(train_text)
X_test = vectorizer.transform(test_text)

train_len = train_text.str.len().values.reshape(-1, 1)
test_len = test_text.str.len().values.reshape(-1, 1)

train_len_sparse = sparse.csr_matrix(train_len)
test_len_sparse = sparse.csr_matrix(test_len)

X_train = sparse.hstack([X_train, train_len_sparse])
X_test = sparse.hstack([X_test, test_len_sparse])



## === cell 6
y_train = train[target_cols].values

base_model = Ridge(alpha=0.05, random_state=42, solver="lsqr")  # weaker regularisation
model = MultiOutputRegressor(base_model)
model.fit(X_train, y_train)



## === cell 7
test_pred = model.predict(X_test)

test_pred = np.clip(test_pred, 0.0, 1.0)



## === cell 8
submission = pd.DataFrame(test_pred, columns=target_cols)
submission.insert(0, "qa_id", test["qa_id"].values)



## === cell 9
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
