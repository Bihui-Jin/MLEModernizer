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

0.3655933516945904

# 6. Current score

0.28529

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.03533) has done: 'I fix the import errors, make tokenisation fallback to the standard HuggingFace model, guard directory creation, skip saving/loading non‑existent pre‑processed files, and build a simple dataset that tokenises the test rows on the fly. This lets the script run end‑to‑end, load the available pretrained model weights, generate predictions for the test set, and write a correctly formatted `submission.csv` file.'
- What this solution (achieved 0.16444) has done: 'I replace the failing Transformer imports with safe fallbacks, add a simple deterministic tokeniser for cases without a tokenizer, and compute predictions using a lightweight linear‑regression model built from question/answer text lengths. This removes the crash‑prone BERT model while still providing varied predictions that correlate better with the targets, moving the score toward the target range. The script now writes a proper `submission.csv` with the required columns.'
- What this solution (achieved 0.23157) has done: 'I added the missing `os` import to fix the directory‑creation errors, and expanded the feature engineering in `build_features` to include word counts and simple punctuation flags for title, body and answer texts. These extra numeric signals give the linear regression a richer representation, which should raise the Spearman correlation toward the target while keeping the core model unchanged. The rest of the pipeline remains the same, now successfully creating the required `submission.csv`.'
- What this solution (achieved 0.25194) has done: 'I add a few informative numeric features (log‑scaled lengths and average word lengths) to the existing feature set used by the linear‑regression model. These extra monotonic signals often improve the rank‑based Spearman metric while keeping the original linear‑regression approach unchanged. The change is limited to the feature‑building function, so the overall pipeline, model architecture, and output format remain the same.'
- What this solution (achieved 0.28013) has done: 'The fix adds the missing imports, loads the data, builds numeric features, fits a simple linear‑regression model (using NumPy least‑squares) on the training set, generates predictions for the test set, clips them to the required [0, 1] range and writes a correctly‑formatted `submission.csv` with the exact column order from the sample submission. No core modelling logic is changed, only the surrounding pipeline is completed so the script runs end‑to‑end and can achieve a score closer to the target.'
- What this solution (achieved 0.28529) has done: 'The update adds a few extra numeric signals that are monotonic (digit ratios, punctuation counts, and log‑scaled word counts) which tend to improve rank‑based metrics like Spearman while keeping the linear‑regression pipeline unchanged. The regularisation strength is also increased slightly (ridge_alpha = 1e‑2) to reduce over‑fitting, which should raise the validation correlation toward the target score.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np

sample_submission = pd.read_csv("../input/google-quest-challenge/sample_submission.csv")
test = pd.read_csv("../input/google-quest-challenge/test.csv")
train = pd.read_csv("../input/google-quest-challenge/train.csv")

_category_counts = train["category"].value_counts().to_dict()
_host_counts = train["host"].value_counts().to_dict()




## === cell 1
def build_features(df):
    title_len = df.question_title.astype(str).apply(len).values
    body_len = df.question_body.astype(str).apply(len).values
    answer_len = df.answer.astype(str).apply(len).values

    title_wc = df.question_title.astype(str).apply(lambda x: len(x.split())).values
    body_wc = df.question_body.astype(str).apply(lambda x: len(x.split())).values
    answer_wc = df.answer.astype(str).apply(lambda x: len(x.split())).values

    title_q = df.question_title.astype(str).apply(lambda x: int("?" in x)).values
    body_q = df.question_body.astype(str).apply(lambda x: int("?" in x)).values
    answer_q = df.answer.astype(str).apply(lambda x: int("?" in x)).values

    title_exc = df.question_title.astype(str).apply(lambda x: int("!" in x)).values
    body_exc = df.question_body.astype(str).apply(lambda x: int("!" in x)).values
    answer_exc = df.answer.astype(str).apply(lambda x: int("!" in x)).values

    title_up = (
        df.question_title.astype(str)
        .apply(lambda x: sum(1 for c in x if c.isupper()) / max(len(x), 1))
        .values
    )
    body_up = (
        df.question_body.astype(str)
        .apply(lambda x: sum(1 for c in x if c.isupper()) / max(len(x), 1))
        .values
    )
    answer_up = (
        df.answer.astype(str)
        .apply(lambda x: sum(1 for c in x if c.isupper()) / max(len(x), 1))
        .values
    )

    title_len_log = np.log1p(title_len)
    body_len_log = np.log1p(body_len)
    answer_len_log = np.log1p(answer_len)

    title_avg_word_len = title_len / np.maximum(title_wc, 1)
    body_avg_word_len = body_len / np.maximum(body_wc, 1)
    answer_avg_word_len = answer_len / np.maximum(answer_wc, 1)

    title_digit_ratio = (
        df.question_title.astype(str)
        .apply(lambda x: sum(c.isdigit() for c in x) / max(len(x), 1))
        .values
    )
    body_digit_ratio = (
        df.question_body.astype(str)
        .apply(lambda x: sum(c.isdigit() for c in x) / max(len(x), 1))
        .values
    )
    answer_digit_ratio = (
        df.answer.astype(str)
        .apply(lambda x: sum(c.isdigit() for c in x) / max(len(x), 1))
        .values
    )

    title_punct = (
        df.question_title.astype(str)
        .apply(lambda x: x.count("?") + x.count("!"))
        .values
    )
    body_punct = (
        df.question_body.astype(str).apply(lambda x: x.count("?") + x.count("!")).values
    )
    answer_punct = (
        df.answer.astype(str).apply(lambda x: x.count("?") + x.count("!")).values
    )

    title_wc_log = np.log1p(title_wc)
    body_wc_log = np.log1p(body_wc)
    answer_wc_log = np.log1p(answer_wc)

    cat_freq = df["category"].map(_category_counts).fillna(0).values
    host_freq = df["host"].map(_host_counts).fillna(0).values

    bias = np.ones_like(title_len)

    feats = np.vstack(
        [
            title_len,
            body_len,
            answer_len,
            title_wc,
            body_wc,
            answer_wc,
            title_q,
            body_q,
            answer_q,
            title_exc,
            body_exc,
            answer_exc,
            title_up,
            body_up,
            answer_up,
            title_len_log,
            body_len_log,
            answer_len_log,
            title_avg_word_len,
            body_avg_word_len,
            answer_avg_word_len,
            title_digit_ratio,
            body_digit_ratio,
            answer_digit_ratio,
            title_punct,
            body_punct,
            answer_punct,
            title_wc_log,
            body_wc_log,
            answer_wc_log,
            cat_freq,
            host_freq,
            bias,
        ]
    ).T
    return feats




## === cell 2
target_cols = [c for c in sample_submission.columns if c != "qa_id"]
X_train = build_features(train)
y_train = train[target_cols].values.astype(np.float32)

X_test = build_features(test)




## === cell 3
ridge_alpha = 1e-2
A = X_train.T @ X_train + ridge_alpha * np.eye(X_train.shape[1])
B = X_train.T @ y_train
weights = np.linalg.solve(A, B)  # shape (features, targets)




## === cell 4
preds = X_test @ weights
preds = np.clip(preds, 0.0, 1.0)




## === cell 5
submission = pd.DataFrame(preds, columns=target_cols)
submission.insert(0, "qa_id", test["qa_id"])

output_path = "./submission.csv"
submission.to_csv(output_path, index=False)
print(f"Submission written to {output_path}")
